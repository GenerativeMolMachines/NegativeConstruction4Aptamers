"""Controlled negative dataset construction for ERE.

All methods are forced to the same target size and approximately the same
ligand distribution as the original experimental negatives in the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from data import detect_alphabet, gc_content, known_pairs, normalize_sequence
from learned import generated_by_molecule, load_generated_pairs
from methods import (
    generate_composition_shuffle,
    generate_hard_negatives,
    generate_random_sequence,
    generate_structure_filtered,
)


NEGATIVE_METHODS: tuple[str, ...] = (
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
    "hard_negative",
    "learned_generated",
    "edit_based",
    "gru_edit",
    "learned_real_hard_negative",
    "experimental",
)

# Generation does not need contrastive/LGBM scorer or GRU CSV; only size/ligand matching.
STATIC_NEGATIVE_METHODS: frozenset[str] = frozenset({
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
})


@dataclass
class NegativeDataset:
    method: str
    frame: pd.DataFrame
    target_n: int
    matched_ligand_fraction: float
    notes: str = ""


def build_controlled_negative_dataset(
    method: str,
    positives: pd.DataFrame,
    reference_negatives: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    target_n: int,
    generated_path: str,
    shuffle_struct_diff: float = 0.5,
    score_fn: Callable | None = None,
    candidate_positives: pd.DataFrame | None = None,
) -> NegativeDataset:
    """Build one negative set with size and ligand distribution controls."""
    if target_n <= 0:
        return NegativeDataset(method, _empty_negatives(), target_n, 0.0, "empty_target")

    if method == "experimental":
        out = _sample_frame(reference_negatives, target_n, rng)
        return NegativeDataset(method, _pairs_to_frame(out), target_n, 1.0, "original_experimental_negatives")

    sources, matched_frac = sample_positive_sources_by_ligand(positives, reference_negatives, target_n, rng)

    if method == "random_sequence":
        raw = generate_random_sequence(sources, rng, neg_ratio=1)
    elif method == "composition_shuffle":
        raw = generate_composition_shuffle(sources, rng, neg_ratio=1, min_struct_diff=shuffle_struct_diff)
    elif method == "structure_filtered":
        raw = generate_structure_filtered(sources, rng, neg_ratio=1)
    elif method == "hard_negative":
        raw = generate_hard_negatives(sources, full_df, rng, neg_ratio=1, score_fn=score_fn,
                                      candidate_frame=candidate_positives)
    elif method == "random_pairing":
        raw = generate_ligand_matched_random_pairing(sources, full_df, rng,
                                                   candidate_frame=candidate_positives)
    elif method in ("learned_generated", "edit_based", "gru_edit"):
        raw = generate_ligand_matched_learned(
            sources,
            generated_path,
            rng,
            reference_negatives=reference_negatives,
            positives=positives,
            score_fn=score_fn,
            method_label=method,
        )
    elif method == "learned_real_hard_negative":
        raw = generate_ligand_matched_learned_real_hard(
            sources,
            reference_negatives,
            full_df,
            rng,
            score_fn=score_fn,
        )
    else:
        raise ValueError(f"Unknown negative method: {method}")

    out = _pairs_to_frame(raw)
    out = _force_size(out, target_n, rng)
    notes = "" if len(out) == target_n else f"size_mismatch:{len(out)}/{target_n}"
    return NegativeDataset(method, out, target_n, matched_frac, notes)


def sample_positive_sources_by_ligand(
    positives: pd.DataFrame,
    reference_negatives: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """Sample source positives following the ligand counts of reference negatives."""
    positives = positives.reset_index(drop=True)
    if positives.empty:
        return positives, 0.0
    ref = _sample_frame(reference_negatives, target_n, rng)
    grouped_pos = {str(k): g for k, g in positives.groupby("canonical_smiles", sort=False)}
    rows: list[pd.Series] = []
    matched = 0
    for neg in ref.itertuples(index=False):
        mol = str(getattr(neg, "canonical_smiles"))
        pool = grouped_pos.get(mol)
        if pool is None or pool.empty:
            pool = positives
        else:
            matched += 1
        pick = int(rng.integers(len(pool)))
        rows.append(pool.iloc[pick])
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out, matched / max(target_n, 1)


def generate_ligand_matched_random_pairing(
    sources: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    candidate_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Random-pairing baseline with ligand distribution fixed by sources."""
    forbidden = known_pairs(full_df)
    candidates = full_df if candidate_frame is None else candidate_frame
    seqs = candidates["sequence"].astype(str).unique()
    meta = (
        candidates.drop_duplicates("sequence")
        .set_index("sequence")[[c for c in ("type", "length") if c in candidates.columns]]
    )
    rows: list[dict] = []
    for i, r in sources.reset_index(drop=True).iterrows():
        mol = str(r["canonical_smiles"])
        pos_seq = str(r["sequence"])
        legal = [seq for seq in seqs if seq != pos_seq and (seq, mol) not in forbidden]
        if not legal:
            continue
        seq = str(rng.choice(legal))
        m = meta.loc[seq] if seq in meta.index else {}
        rows.append(
            {
                "method": "random_pairing",
                "sequence": normalize_sequence(seq),
                "canonical_smiles": mol,
                "label": 0,
                "type": m["type"] if "type" in meta.columns and seq in meta.index else r.get("type", "DNA"),
                "source_positive_idx": i,
                "notes": "ligand_matched_random_real_sequence",
            }
        )
    return pd.DataFrame(rows)


def generate_ligand_matched_learned(
    sources: pd.DataFrame,
    generated_path: str,
    rng: np.random.Generator,
    *,
    reference_negatives: pd.DataFrame | None = None,
    positives: pd.DataFrame | None = None,
    score_fn: Callable | None = None,
    method_label: str = "learned_generated",
) -> pd.DataFrame:
    """Target-conditioned generated negatives with distribution-aware reranking.

    Prefer candidates inside experimental length/GC bands, close to the
    experimental k-mer centroid, novel vs train positives, and only mildly
    non-binding (semi-hard decoded_sim band) so ultra-low-sim GRU artifacts
    do not dominate.
    """
    gen = load_generated_pairs(generated_path)
    by_mol = generated_by_molecule(gen)
    rows: list[dict] = []
    used_by_mol: dict[str, set[str]] = {}
    used_global: set[str] = set()
    ref_profile = _distribution_profile(reference_negatives)
    pos_kmers = _precompute_kmers(
        positives["sequence"].astype(str).map(normalize_sequence).tolist()
        if positives is not None and not positives.empty
        else []
    )
    for i, r in sources.reset_index(drop=True).iterrows():
        mol = str(r["canonical_smiles"])
        pool = by_mol.get(mol)
        if pool is None or pool.empty:
            continue
        ranked = _rank_generated_pool(
            pool,
            source_row=r,
            ref_profile=ref_profile,
            pos_kmers=pos_kmers,
            used_global=used_global,
            used_for_mol=used_by_mol.setdefault(mol, set()),
            score_fn=score_fn,
        )
        if ranked.empty:
            continue
        g = ranked.iloc[0]
        seq = normalize_sequence(g["sequence"])
        used_by_mol[mol].add(seq)
        used_global.add(seq)
        rows.append(
            {
                "method": method_label,
                "sequence": seq,
                "canonical_smiles": mol,
                "label": 0,
                "type": r.get("type", "DNA"),
                "source_positive_idx": i,
                "contrastive_sim": g.get("decoded_sim", np.nan),
                "profile_score": g.get("_profile_score", np.nan),
                "positive_kmer_jaccard": g.get("_positive_kmer_jaccard", np.nan),
                "exp_kmer_l2": g.get("_exp_kmer_l2", np.nan),
                "length_delta": g.get("_length_delta", np.nan),
                "gc_delta": g.get("_gc_delta", np.nan),
                "notes": (
                    "ligand_matched_gru_edit"
                    if method_label == "gru_edit"
                    else (
                        "ligand_matched_edit_based"
                        if method_label == "edit_based"
                        else "ligand_matched_gru_generated_profile_hardened"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_ligand_matched_learned_real_hard(
    sources: pd.DataFrame,
    reference_negatives: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    score_fn: Callable | None = None,
    candidate_multiplier: int = 64,
) -> pd.DataFrame:
    """Real-aptamer target-aware hard negatives reranked by learned score.

    This is the hybrid method: stay on the real aptamer manifold, but use the
    learned contrastive scorer to select target-aware hard non-interacting pairs.
    """
    forbidden = known_pairs(full_df)
    seqs = full_df["sequence"].astype(str).map(normalize_sequence).unique()
    seq_meta = (
        full_df.assign(sequence=lambda x: x["sequence"].astype(str).map(normalize_sequence))
        .drop_duplicates("sequence")
        .set_index("sequence")[[c for c in ("type", "length", "gc_content") if c in full_df.columns]]
    )
    ref_profile = _distribution_profile(reference_negatives)
    score_profile = _score_profile(reference_negatives, score_fn)
    rows: list[dict] = []
    n_cand = min(len(seqs), max(24, candidate_multiplier))

    for i, r in sources.reset_index(drop=True).iterrows():
        mol = str(r["canonical_smiles"])
        pos_seq = normalize_sequence(str(r["sequence"]))
        candidates = rng.choice(seqs, size=n_cand, replace=False)
        scored: list[dict] = []
        batch_scorer = getattr(score_fn, "__self__", None) if score_fn is not None else None
        valid_seqs = [
            normalize_sequence(str(seq))
            for seq in candidates
            if normalize_sequence(str(seq)) != pos_seq and (normalize_sequence(str(seq)), mol) not in forbidden
        ]
        score_map: dict[str, float] = {}
        if batch_scorer is not None and hasattr(batch_scorer, "score_many") and valid_seqs:
            batch_scores = batch_scorer.score_many(valid_seqs, mol)
            score_map = {
                seq: float(val) if np.isfinite(val) else -1.0
                for seq, val in zip(valid_seqs, batch_scores)
            }
        for seq in valid_seqs:
            meta = seq_meta.loc[seq] if seq in seq_meta.index else {}
            typ = meta["type"] if "type" in seq_meta.columns and seq in seq_meta.index else r.get("type", "DNA")
            if seq in score_map:
                model_score = score_map[seq]
            elif score_fn is not None:
                model_score = float(score_fn(seq, mol))
                if not np.isfinite(model_score):
                    model_score = -1.0
            else:
                model_score = float(rng.random())
            profile_penalty = _profile_penalty(seq, typ, ref_profile, r)
            score_penalty = _score_band_penalty(model_score, score_profile)
            # Match the harder side of experimental negatives rather than
            # blindly maximizing the score, which can select false positives.
            composite = score_penalty + 0.35 * profile_penalty
            scored.append(
                {
                    "method": "learned_real_hard_negative",
                    "sequence": seq,
                    "canonical_smiles": mol,
                    "label": 0,
                    "type": typ,
                    "source_positive_idx": i,
                    "contrastive_sim": model_score,
                    "profile_score": profile_penalty,
                    "score_band_penalty": score_penalty,
                    "_composite": composite,
                    "notes": "real_aptamer_target_aware_lgbm_hard",
                }
            )
        if scored:
            rows.append(min(scored, key=lambda x: x["_composite"]))

    out = pd.DataFrame(rows)
    return out.drop(columns=["_composite"], errors="ignore")


def _distribution_profile(reference_negatives: pd.DataFrame | None) -> dict:
    if reference_negatives is None or reference_negatives.empty:
        return {
            "length_mean": 45.0,
            "length_std": 10.0,
            "length_p10": 30.0,
            "length_p90": 60.0,
            "gc_mean": 0.55,
            "gc_std": 0.12,
            "gc_p10": 0.40,
            "gc_p90": 0.70,
            "dna_frac": 1.0,
            "kmer_centroid": None,
            "decoded_sim_p25": None,
            "decoded_sim_p75": None,
        }
    seq = reference_negatives["sequence"].astype(str).map(normalize_sequence)
    typ = reference_negatives.get("type", pd.Series(["DNA"] * len(reference_negatives))).astype(str).str.upper()
    lengths = seq.map(len).to_numpy(dtype=float)
    gcs = seq.map(gc_content).to_numpy(dtype=float)
    length_std = float(max(np.std(lengths), 3.0))
    gc_std = float(max(np.std(gcs), 0.05))
    length_mean = float(np.mean(lengths))
    gc_mean = float(np.mean(gcs))
    if len(lengths) >= 5:
        length_p10 = float(np.quantile(lengths, 0.10))
        length_p90 = float(np.quantile(lengths, 0.90))
        gc_p10 = float(np.quantile(gcs, 0.10))
        gc_p90 = float(np.quantile(gcs, 0.90))
    else:
        length_p10 = length_mean - 1.5 * length_std
        length_p90 = length_mean + 1.5 * length_std
        gc_p10 = gc_mean - 1.5 * gc_std
        gc_p90 = gc_mean + 1.5 * gc_std
    return {
        "length_mean": length_mean,
        "length_std": length_std,
        "length_p10": length_p10,
        "length_p90": length_p90,
        "gc_mean": gc_mean,
        "gc_std": gc_std,
        "gc_p10": gc_p10,
        "gc_p90": gc_p90,
        "dna_frac": float((typ == "DNA").mean()),
        "kmer_centroid": _kmer_centroid(seq.tolist()),
        "decoded_sim_p25": None,
        "decoded_sim_p75": None,
    }


def _score_profile(reference_negatives: pd.DataFrame | None, score_fn: Callable | None) -> dict:
    if score_fn is None or reference_negatives is None or reference_negatives.empty:
        return {"target": None, "scale": 1.0}
    sample = reference_negatives
    if len(sample) > 256:
        sample = sample.sample(n=256, random_state=13)
    batch_scorer = getattr(score_fn, "__self__", None)
    if batch_scorer is not None and hasattr(batch_scorer, "score_table"):
        arr = np.asarray(batch_scorer.score_table(sample), dtype=float)
        arr = arr[np.isfinite(arr)]
    else:
        vals = []
        for seq, mol in zip(sample["sequence"], sample["canonical_smiles"]):
            val = float(score_fn(str(seq), str(mol)))
            if np.isfinite(val):
                vals.append(val)
        arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {"target": None, "scale": 1.0}
    # Hard but still experimental-like: upper quartile of measured negatives.
    return {"target": float(np.quantile(arr, 0.75)), "scale": float(max(np.std(arr), 0.03))}


def _score_band_penalty(score: float, score_profile: dict) -> float:
    target = score_profile.get("target")
    if target is None or not np.isfinite(score):
        return 0.0
    return float(abs(score - target) / max(float(score_profile.get("scale", 1.0)), 1e-6))


def _rank_generated_pool(
    pool: pd.DataFrame,
    *,
    source_row: pd.Series,
    ref_profile: dict,
    pos_kmers: list[tuple[str, set[str]]],
    used_global: set[str],
    used_for_mol: set[str],
    score_fn: Callable | None,
) -> pd.DataFrame:
    if pool.empty:
        return pool
    items = pool.copy()
    items["sequence"] = items["sequence"].astype(str).map(normalize_sequence)
    items = items.drop_duplicates("sequence").reset_index(drop=True)
    rows: list[dict] = []
    mol = str(source_row["canonical_smiles"])
    in_band_rows: list[dict] = []
    for _, item in items.iterrows():
        seq = str(item["sequence"])
        typ = str(source_row.get("type", "DNA"))
        decoded = float(item.get("decoded_sim", np.nan))
        if not np.isfinite(decoded):
            decoded = 0.0
        learned_score = float(score_fn(seq, mol)) if score_fn is not None else decoded
        if not np.isfinite(learned_score):
            learned_score = decoded
        length = len(seq)
        gc = gc_content(seq)
        length_delta = abs(length - float(ref_profile["length_mean"]))
        gc_delta = abs(gc - float(ref_profile["gc_mean"]))
        exp_l2 = _exp_kmer_l2(seq, ref_profile.get("kmer_centroid"))
        profile_penalty = _profile_penalty(seq, typ, ref_profile, source_row)
        pos_jaccard = _nearest_kmer_jaccard(seq, pos_kmers)
        reuse_penalty = 2.0 * float(seq in used_for_mol) + 0.6 * float(seq in used_global)
        semi_hard = _semi_hard_decoded_penalty(decoded if score_fn is None else learned_score)
        # Prefer experimental-like length/GC/k-mer; keep mild non-binding scores.
        composite = (
            learned_score
            + 1.0 * profile_penalty
            + 0.8 * exp_l2
            + 0.35 * pos_jaccard
            + reuse_penalty
            + semi_hard
        )
        rec = item.to_dict()
        rec["_profile_score"] = float(profile_penalty)
        rec["_positive_kmer_jaccard"] = float(pos_jaccard)
        rec["_exp_kmer_l2"] = float(exp_l2)
        rec["_length_delta"] = float(length_delta)
        rec["_gc_delta"] = float(gc_delta)
        rec["_composite"] = float(composite)
        rows.append(rec)
        if _in_length_gc_band(length, gc, ref_profile):
            in_band_rows.append(rec)
    # Hard preference: if any candidates sit in experimental [p10,p90], use only them.
    chosen = in_band_rows if in_band_rows else rows
    return pd.DataFrame(chosen).sort_values("_composite", ascending=True).reset_index(drop=True)


def _in_length_gc_band(length: float, gc: float, ref_profile: dict) -> bool:
    return (
        float(ref_profile["length_p10"]) <= length <= float(ref_profile["length_p90"])
        and float(ref_profile["gc_p10"]) <= gc <= float(ref_profile["gc_p90"])
    )


def _semi_hard_decoded_penalty(score: float) -> float:
    """Penalize ultra-low similarity artifacts; prefer mild non-binders."""
    if not np.isfinite(score):
        return 0.0
    # Soft target near slightly-negative / near-zero non-binder scores.
    target = -0.05
    penalty = 0.4 * abs(float(score) - target)
    if score < -0.35:
        penalty += 2.0 * abs(float(score) + 0.35)
    return float(penalty)


def _profile_penalty(seq: str, typ: str, ref_profile: dict, source_row: pd.Series | None = None) -> float:
    seq = normalize_sequence(seq)
    length = len(seq)
    gc = gc_content(seq)
    length_z = abs(length - ref_profile["length_mean"]) / max(ref_profile["length_std"], 1e-6)
    gc_z = abs(gc - ref_profile["gc_mean"]) / max(ref_profile["gc_std"], 1e-6)
    band_penalty = 0.0
    if not _in_length_gc_band(length, gc, ref_profile):
        # Hard reject outside experimental quantiles (still rankable as last resort).
        band_penalty = 50.0
    typ = str(typ).upper()
    dna_target = ref_profile.get("dna_frac", 1.0)
    type_penalty = 0.0
    if typ == "DNA":
        type_penalty = max(0.0, 0.5 - dna_target)
    elif typ == "RNA":
        type_penalty = max(0.0, dna_target - 0.5)
    source_penalty = 0.0
    if source_row is not None:
        if "length" in source_row and pd.notna(source_row["length"]):
            source_penalty += 0.15 * abs(length - float(source_row["length"])) / max(ref_profile["length_std"], 1e-6)
        if "gc_content" in source_row and pd.notna(source_row["gc_content"]):
            source_penalty += 0.15 * abs(gc - float(source_row["gc_content"])) / max(ref_profile["gc_std"], 1e-6)
    return float(band_penalty + length_z + gc_z + type_penalty + source_penalty)


_BASES = ("A", "C", "G", "T")
_KMER4_INDEX = {a + b + c + d: i for i, (a, b, c, d) in enumerate(
    (w, x, y, z) for w in _BASES for x in _BASES for y in _BASES for z in _BASES
)}


def _kmer_freq(seq: str, k: int = 4) -> np.ndarray:
    seq = normalize_sequence(seq)
    dim = 4**k
    vec = np.zeros(dim, dtype=np.float64)
    if len(seq) < k:
        return vec
    for i in range(len(seq) - k + 1):
        mer = seq[i : i + k]
        idx = _KMER4_INDEX.get(mer) if k == 4 else None
        if idx is None:
            continue
        vec[idx] += 1.0
    total = float(vec.sum())
    if total > 0:
        vec /= total
    return vec


def _kmer_centroid(sequences: list[str], k: int = 4) -> np.ndarray | None:
    if not sequences:
        return None
    mats = [_kmer_freq(s, k=k) for s in sequences]
    if not mats:
        return None
    return np.mean(np.vstack(mats), axis=0)


def _exp_kmer_l2(seq: str, centroid: np.ndarray | None) -> float:
    if centroid is None:
        return 0.0
    return float(np.linalg.norm(_kmer_freq(seq) - centroid))


def _kmer_set(seq: str, k: int = 4) -> set[str]:
    seq = normalize_sequence(seq)
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _precompute_kmers(sequences: list[str], k: int = 4) -> list[tuple[str, set[str]]]:
    unique = sorted({normalize_sequence(s) for s in sequences if str(s)})
    return [(s, _kmer_set(s, k)) for s in unique]


def _nearest_kmer_jaccard(seq: str, pool_kmers: list[tuple[str, set[str]]], k: int = 4) -> float:
    if not pool_kmers:
        return 0.0
    sk = _kmer_set(seq, k)
    if not sk:
        return 0.0
    best = 0.0
    for _, pk in pool_kmers:
        union = len(sk | pk)
        if union:
            best = max(best, len(sk & pk) / union)
    return float(best)


def diagnostics(frame: pd.DataFrame, positives: pd.DataFrame, reference_negatives: pd.DataFrame) -> dict:
    seq = frame["sequence"].astype(str).map(normalize_sequence)
    ref_seq = reference_negatives["sequence"].astype(str).map(normalize_sequence)
    pos_seq = positives["sequence"].astype(str).map(normalize_sequence)
    out = {
        "n": int(len(frame)),
        "mean_len": float(seq.map(len).mean()) if len(seq) else None,
        "mean_gc": float(seq.map(gc_content).mean()) if len(seq) else None,
        "dna_frac": float((frame.get("type", pd.Series(["DNA"] * len(frame))).astype(str).str.upper() == "DNA").mean())
        if len(frame)
        else None,
        "n_ligands": int(frame["canonical_smiles"].nunique()) if len(frame) else 0,
        "ref_mean_len": float(ref_seq.map(len).mean()) if len(ref_seq) else None,
        "ref_mean_gc": float(ref_seq.map(gc_content).mean()) if len(ref_seq) else None,
        "pos_mean_len": float(pos_seq.map(len).mean()) if len(pos_seq) else None,
        "pos_mean_gc": float(pos_seq.map(gc_content).mean()) if len(pos_seq) else None,
    }
    for col in (
        "contrastive_sim",
        "profile_score",
        "positive_kmer_jaccard",
        "score_band_penalty",
        "exp_kmer_l2",
        "length_delta",
        "gc_delta",
    ):
        if col in frame.columns and len(frame):
            vals = pd.to_numeric(frame[col], errors="coerce")
            out[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else None
    return out


def _pairs_to_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return _empty_negatives()
    out = pairs.loc[:, ["sequence", "canonical_smiles"]].copy()
    out["sequence"] = out["sequence"].map(normalize_sequence)
    out["canonical_smiles"] = out["canonical_smiles"].astype(str).str.strip()
    out["label"] = 0
    if "type" in pairs.columns:
        out["type"] = pairs["type"].astype(str).values
    else:
        out["type"] = out["sequence"].map(lambda s: "RNA" if "U" in str(s).upper() else "DNA")
    for col in (
        "contrastive_sim",
        "profile_score",
        "positive_kmer_jaccard",
        "score_band_penalty",
        "exp_kmer_l2",
        "length_delta",
        "gc_delta",
    ):
        if col in pairs.columns:
            out[col] = pd.to_numeric(pairs[col], errors="coerce").values
    return out.reset_index(drop=True)


def _force_size(frame: pd.DataFrame, target_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) == target_n:
        return frame.reset_index(drop=True)
    if len(frame) > target_n:
        return frame.sample(n=target_n, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    if frame.empty:
        return frame
    extra = frame.sample(
        n=target_n - len(frame),
        replace=True,
        random_state=int(rng.integers(1_000_000)),
    )
    return pd.concat([frame, extra], ignore_index=True).reset_index(drop=True)


def _sample_frame(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) >= n:
        return frame.sample(n=n, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    return frame.sample(n=n, replace=True, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def _empty_negatives() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence", "canonical_smiles", "label", "type"])

"""Six synthetic negative generators + experimental AptaBench reference."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from data import detect_alphabet, gc_content, known_pairs, normalize_sequence
from learned import generated_by_molecule, load_generated_pairs
from schema import METHOD_SPECS, OUTPUT_COLUMNS
from structure import is_aptamer_like, structure_diff_frac, structure_features

NUCLEOTIDES = {
    "ACGT": list("ACGT"),
    "ACGU": list("ACGU"),
}


def _spec_row(method: str, **kwargs) -> dict:
    spec = METHOD_SPECS[method]
    row = {
        "method": spec["method"],
        "method_name": spec["method_name"],
        "literature": spec["literature"],
        "is_reference": spec["is_reference"],
        "label": 0,
        "sequence": "",
        "canonical_smiles": "",
        "type": "DNA",
        "source_positive_idx": -1,
        "pair_rank": 0,
        "length": 0,
        "gc_content": 0.0,
        "alphabet": "ACGT",
        "contrastive_sim": np.nan,
        "mfe_per_nt": np.nan,
        "paired_frac": np.nan,
        "notes": "",
    }
    row.update(kwargs)
    seq = row["sequence"]
    if seq:
        row["length"] = len(normalize_sequence(seq))
        row["gc_content"] = gc_content(seq)
        row["alphabet"] = detect_alphabet(seq)
        if row["type"] not in ("DNA", "RNA"):
            row["type"] = "RNA" if row["alphabet"] == "ACGU" else "DNA"
    return row


def _finalize(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in {"contrastive_sim", "mfe_per_nt", "paired_frac"} else ""
    return df[list(OUTPUT_COLUMNS)].reset_index(drop=True)


def _random_seq(length: int, alphabet: str, rng: np.random.Generator) -> str:
    letters = NUCLEOTIDES.get(alphabet, list("ACGT"))
    return "".join(rng.choice(letters, size=max(int(length), 1)))


def _shuffle_seq(
    seq: str,
    rng: np.random.Generator,
    *,
    max_tries: int = 40,
    min_struct_diff: float = 0.0,
) -> str:
    chars = list(str(seq))
    if len(chars) < 2:
        return "".join(chars)
    original = "".join(chars)
    for _ in range(max_tries):
        perm = chars.copy()
        rng.shuffle(perm)
        cand = "".join(perm)
        if cand == original:
            continue
        if min_struct_diff <= 0 or structure_diff_frac(original, cand) >= min_struct_diff:
            return cand
    rng.shuffle(chars)
    cand = "".join(chars)
    if cand == original and len(chars) > 1:
        cand = chars[1] + "".join(chars[2:]) + chars[0]
    return cand


def generate_experimental(df: pd.DataFrame) -> pd.DataFrame:
    negs = df.loc[df["label"] == 0].reset_index(drop=True)
    rows = []
    for i, r in negs.iterrows():
        feats = structure_features(r["sequence"])
        rows.append(
            _spec_row(
                "experimental",
                sequence=r["sequence"],
                canonical_smiles=r["canonical_smiles"],
                type=r.get("type", "DNA"),
                source_positive_idx=-1,
                pair_rank=int(i),
                mfe_per_nt=feats.get("mfe_per_nt", np.nan),
                paired_frac=feats.get("paired_frac", np.nan),
                notes="aptabench_label0",
            )
        )
    return _finalize(rows)


def generate_random_sequence(
    positives: pd.DataFrame,
    rng: np.random.Generator,
    neg_ratio: int = 4,
) -> pd.DataFrame:
    rows = []
    for i, r in positives.iterrows():
        alphabet = r.get("alphabet") or detect_alphabet(r["sequence_raw"] if "sequence_raw" in r else r["sequence"])
        n = int(r["length"] if "length" in r else len(r["sequence"]))
        for k in range(neg_ratio):
            seq = _random_seq(n, alphabet, rng)
            rows.append(
                _spec_row(
                    "random_sequence",
                    sequence=seq,
                    canonical_smiles=r["canonical_smiles"],
                    type=r.get("type", "DNA"),
                    source_positive_idx=int(i),
                    pair_rank=k + 1,
                    notes="uniform_same_length",
                )
            )
    return _finalize(rows)


def generate_composition_shuffle(
    positives: pd.DataFrame,
    rng: np.random.Generator,
    neg_ratio: int = 4,
    min_struct_diff: float = 0.5,
) -> pd.DataFrame:
    """AptaBLE: 4 composition-preserving shuffles per positive."""
    rows = []
    for i, r in positives.iterrows():
        raw = str(r["sequence_raw"] if "sequence_raw" in r.index and pd.notna(r.get("sequence_raw")) else r["sequence"])
        for k in range(neg_ratio):
            seq = _shuffle_seq(raw, rng, min_struct_diff=min_struct_diff)
            feats = structure_features(seq)
            rows.append(
                _spec_row(
                    "composition_shuffle",
                    sequence=normalize_sequence(seq),
                    canonical_smiles=r["canonical_smiles"],
                    type=r.get("type", "DNA"),
                    source_positive_idx=int(i),
                    pair_rank=k + 1,
                    mfe_per_nt=feats.get("mfe_per_nt", np.nan),
                    paired_frac=feats.get("paired_frac", np.nan),
                    notes=f"aptable_shuffle min_struct_diff={min_struct_diff}",
                )
            )
    return _finalize(rows)


def generate_random_pairing(
    positives: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    neg_ratio: int = 4,
    exclude_experimental: bool = True,
    max_tries_factor: int = 40,
) -> pd.DataFrame:
    """SMRTnet / interaction-prediction: rewire real sequences × real molecules."""
    forbidden = known_pairs(full_df, {1})
    if exclude_experimental:
        forbidden |= known_pairs(full_df, {0})
    # SMRTnet draws from the full corpus of real fragments and ligands, not only positives.
    seqs = full_df["sequence"].astype(str).unique()
    mols = full_df["canonical_smiles"].astype(str).unique()
    type_by_seq = (
        full_df.drop_duplicates("sequence").set_index("sequence")["type"].to_dict()
        if "type" in full_df.columns
        else {}
    )
    rows = []
    max_tries = max(20, neg_ratio * max_tries_factor)
    for i, r in positives.iterrows():
        seen: set[tuple[str, str]] = set()
        k = 0
        tries = 0
        while k < neg_ratio and tries < max_tries:
            tries += 1
            seq = str(rng.choice(seqs))
            mol = str(rng.choice(mols))
            key = (seq, mol)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            k += 1
            rows.append(
                _spec_row(
                    "random_pairing",
                    sequence=seq,
                    canonical_smiles=mol,
                    type=type_by_seq.get(seq, r.get("type", "DNA")),
                    source_positive_idx=int(i),
                    pair_rank=k,
                    notes="rewired_real_pair",
                )
            )
    return _finalize(rows)


def generate_structure_filtered(
    positives: pd.DataFrame,
    rng: np.random.Generator,
    neg_ratio: int = 4,
    max_tries: int = 200,
) -> pd.DataFrame:
    rows = []
    n_filtered = 0
    n_fallback = 0
    for i, r in positives.iterrows():
        alphabet = r.get("alphabet") or detect_alphabet(r["sequence"])
        n = int(r["length"] if "length" in r else len(r["sequence"]))
        k = 0
        tries = 0
        while k < neg_ratio and tries < max_tries * max(neg_ratio, 1):
            tries += 1
            seq = _random_seq(n, alphabet, rng)
            ok, feats = is_aptamer_like(seq)
            if not ok:
                continue
            k += 1
            n_filtered += 1
            rows.append(
                _spec_row(
                    "structure_filtered",
                    sequence=seq,
                    canonical_smiles=r["canonical_smiles"],
                    type=r.get("type", "DNA"),
                    source_positive_idx=int(i),
                    pair_rank=k,
                    mfe_per_nt=feats.get("mfe_per_nt", np.nan),
                    paired_frac=feats.get("paired_frac", np.nan),
                    notes=f"ppai_style backend={feats.get('backend', 'unknown')}",
                )
            )
        while k < neg_ratio:
            seq = _random_seq(n, alphabet, rng)
            feats = structure_features(seq)
            k += 1
            n_fallback += 1
            rows.append(
                _spec_row(
                    "structure_filtered",
                    sequence=seq,
                    canonical_smiles=r["canonical_smiles"],
                    type=r.get("type", "DNA"),
                    source_positive_idx=int(i),
                    pair_rank=k,
                    mfe_per_nt=feats.get("mfe_per_nt", np.nan),
                    paired_frac=feats.get("paired_frac", np.nan),
                    notes="fallback_unfiltered_random",
                )
            )
    out = _finalize(rows)
    out.attrs["n_structure_filtered"] = n_filtered
    out.attrs["n_structure_fallback"] = n_fallback
    return out


def generate_hard_negatives(
    positives: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    neg_ratio: int = 4,
    candidate_multiplier: int = 32,
    score_fn=None,
    *,
    candidate_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Robinson-style: keep high-scoring (binder-like) non-interacting pairs.

    Candidate pool = real aptamers that are not a known pair with the source
    molecule. `score_fn(seq, mol)` should return contrastive cosine (higher =
    more binder-like = harder negative).
    """
    forbidden = known_pairs(full_df)
    candidates_frame = full_df if candidate_frame is None else candidate_frame
    seqs = np.asarray(candidates_frame["sequence"].astype(str).unique(), dtype=object)
    keep_cols = [c for c in ("type", "length", "gc_content") if c in candidates_frame.columns]
    seq_meta = candidates_frame.drop_duplicates("sequence").set_index("sequence")[keep_cols]

    rows = []
    n_cand = max(neg_ratio * candidate_multiplier, neg_ratio)
    for i, r in positives.iterrows():
        mol = str(r["canonical_smiles"])
        pos_seq = str(r["sequence"])
        # Cheap reject: sample with replacement from full unique set, then filter.
        # Avoid O(|seqs|) list comprehension per source on large tables.
        take = min(max(n_cand * 3, n_cand), len(seqs))
        sampled = rng.choice(seqs, size=take, replace=False)
        pool = [s for s in sampled if s != pos_seq and (s, mol) not in forbidden]
        if len(pool) < n_cand:
            pool = [s for s in seqs if s != pos_seq and (s, mol) not in forbidden]
        if not pool:
            continue
        take = min(n_cand, len(pool))
        chosen = rng.choice(np.asarray(pool, dtype=object), size=take, replace=False)
        candidates = []
        batch_scorer = getattr(score_fn, "__self__", None) if score_fn is not None else None
        if batch_scorer is not None and hasattr(batch_scorer, "score_many"):
            seq_list = [str(s) for s in chosen if (str(s), mol) not in forbidden]
            if seq_list:
                scores = batch_scorer.score_many(seq_list, mol)
                for seq, sim in zip(seq_list, scores):
                    val = float(sim) if np.isfinite(sim) else -np.inf
                    candidates.append((val, seq))
        else:
            for seq in chosen:
                seq = str(seq)
                if (seq, mol) in forbidden:
                    continue
                sim = float(score_fn(seq, mol)) if score_fn is not None else float("nan")
                candidates.append((sim if np.isfinite(sim) else -np.inf, seq))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for k, (sim, seq) in enumerate(candidates[:neg_ratio], start=1):
            meta = seq_meta.loc[seq] if seq in seq_meta.index else None
            rows.append(
                _spec_row(
                    "hard_negative",
                    sequence=seq,
                    canonical_smiles=mol,
                    type=(meta.get("type", r.get("type", "DNA")) if meta is not None else r.get("type", "DNA")),
                    source_positive_idx=int(i),
                    pair_rank=k,
                    contrastive_sim=sim if np.isfinite(sim) else np.nan,
                    notes="robinson_hard_from_real_pool_lgbm" if batch_scorer is not None else "robinson_hard_from_real_pool",
                )
            )
    return _finalize(rows)


def generate_learned(
    positives: pd.DataFrame,
    generated_path: str,
    neg_ratio: int = 4,
) -> pd.DataFrame:
    gen = load_generated_pairs(generated_path)
    by_mol = generated_by_molecule(gen)
    rows = []
    n_hit = 0
    for i, r in positives.iterrows():
        mol = str(r["canonical_smiles"])
        pool = by_mol.get(mol)
        if pool is None or pool.empty:
            continue
        n_hit += 1
        take = pool.head(neg_ratio)
        for k, g in enumerate(take.itertuples(index=False), start=1):
            rows.append(
                _spec_row(
                    "learned_generated",
                    sequence=g.sequence,
                    canonical_smiles=mol,
                    type=r.get("type", "DNA"),
                    source_positive_idx=int(i),
                    pair_rank=k,
                    contrastive_sim=g.decoded_sim,
                    notes=f"gru_decoded_sim={g.decoded_sim:.4f}",
                )
            )
    out = _finalize(rows)
    out.attrs["n_positives_with_generation"] = n_hit
    out.attrs["n_positives"] = int(len(positives))
    out.attrs["n_generated_molecules"] = int(len(by_mol))
    return out


GENERATOR_REGISTRY: dict[str, Callable] = {
    "random_sequence": generate_random_sequence,
    "composition_shuffle": generate_composition_shuffle,
    "random_pairing": generate_random_pairing,
    "structure_filtered": generate_structure_filtered,
    "hard_negative": generate_hard_negatives,
    "learned_generated": generate_learned,
    "experimental": generate_experimental,
}

"""Fraction-local retraining for learned negative methods (mix / fraction_retrain).

At supervision fraction f > 0, contrastive / GRU / bootstrap LGBM are fit only on:
  train_pos + sample(f × |train_neg_exp|) experimental negatives

At f=0 learned methods are not run (no experimental seeds; random-pairing proxy is
forbidden — that would mix methods). Static methods like random_pairing handle f=0.

GRU decoder is trained **from scratch** on fold train positives (no global init).
Those seeds are **not** mixed into classifier train negatives; the method output is
resampled to |train_neg_exp| synthetic rows only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from contrastive_path import ensure_contrastive_on_path
from data import embedding_columns, normalize_sequence
from lgbm_scoring import build_bootstrap_lgbm_scorer
from negative_dataset_controls import (
    NegativeDataset,
    build_controlled_negative_dataset,
)

FRACTION_LOCAL_LEARNED_METHODS = frozenset({"learned_generated", "learned_real_hard_negative"})

# Faster defaults for mix sweeps (full pipeline uses more epochs).
DEFAULT_CONTRASTIVE_EPOCHS = 8
DEFAULT_DECODER_EPOCHS = 10
DEFAULT_NEGATIVE_RATIO = 16


@dataclass
class FractionLocalBundle:
    exp_fraction: float
    n_supervision_neg: int
    contrastive_ckpt: Path
    generation_csv: Path | None
    hard_score_fn: Callable | None
    notes: str = ""


def fraction_local_supervision_negatives(
    train_neg_exp: pd.DataFrame,
    *,
    target_n: int,
    exp_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Experimental negatives visible to generative/scoring models at this mix fraction."""
    exp_fraction = float(np.clip(exp_fraction, 0.0, 1.0))
    n_exp = int(round(exp_fraction * target_n))
    n_exp = min(max(n_exp, 0), target_n, len(train_neg_exp))
    if n_exp <= 0:
        return pd.DataFrame(columns=list(train_neg_exp.columns))
    idx = rng.choice(len(train_neg_exp), size=n_exp, replace=False if n_exp <= len(train_neg_exp) else True)
    return train_neg_exp.iloc[idx].reset_index(drop=True)


def contrastive_supervision_negatives(
    train_pos: pd.DataFrame,
    supervision_neg: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    target_n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str]:
    """Negatives used to train contrastive/GRU at this fraction.

    Only experimental supervision seeds are allowed. Random-pairing (or any other
    method) must not be used as a proxy — that mixes negative-construction methods.
    """
    del train_pos, full_df, target_n, rng  # API kept stable for callers
    if supervision_neg.empty:
        raise ValueError(
            "No experimental supervision negatives at this f; "
            "refusing random-pairing proxy (would mix methods)"
        )
    return supervision_neg, "experimental_subset"


def bootstrap_negatives_for_scorer(
    train_pos: pd.DataFrame,
    supervision_neg: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap LGBM train negatives: fraction-local experimental negs only."""
    del train_pos, full_df, target_n, rng
    return supervision_neg


def _prepare_raw_lookup(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()
    seq_col = "sequence" if "sequence" in out.columns else "aptamer"
    out["sequence"] = out[seq_col].astype(str).map(normalize_sequence)
    out["canonical_smiles"] = out["canonical_smiles"].astype(str).str.strip()
    return out.drop_duplicates(["sequence", "canonical_smiles"]).reset_index(drop=True)


def _rows_to_embeddings(rows: pd.DataFrame, raw_lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    work = rows.copy()
    work["sequence"] = work["sequence"].astype(str).map(normalize_sequence)
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()

    seq_cols = embedding_columns(work, "seq_emb_")
    smi_cols = embedding_columns(work, "smi_emb_")
    if not seq_cols or not smi_cols:
        seq_cols = embedding_columns(raw_lookup, "seq_emb_")
        smi_cols = embedding_columns(raw_lookup, "smi_emb_")
    if not seq_cols or not smi_cols:
        raise ValueError("Need seq_emb_* and smi_emb_* columns in rows or raw_df")

    if not all(c in work.columns for c in seq_cols):
        lookup_cols = ["sequence", "canonical_smiles"] + seq_cols + smi_cols
        work = work.merge(
            raw_lookup[lookup_cols],
            on=["sequence", "canonical_smiles"],
            how="inner",
        )
    if work.empty:
        seq_map = (
            raw_lookup.drop_duplicates("sequence")
            .set_index("sequence")[seq_cols]
        )
        smi_map = (
            raw_lookup.drop_duplicates("canonical_smiles")
            .set_index("canonical_smiles")[smi_cols]
        )
        apt_parts: list[np.ndarray] = []
        smi_parts: list[np.ndarray] = []
        keep_rows: list[dict[str, str]] = []
        for _, row in work.iterrows() if not work.empty else rows.iterrows():
            seq = str(row["sequence"])
            smi = str(row["canonical_smiles"])
            if seq not in seq_map.index or smi not in smi_map.index:
                continue
            apt_parts.append(seq_map.loc[seq].to_numpy(dtype=np.float32))
            smi_parts.append(smi_map.loc[smi].to_numpy(dtype=np.float32))
            keep_rows.append({"sequence": seq, "canonical_smiles": smi})
        if not keep_rows:
            raise ValueError("No embedding rows matched supervision frame")
        work = pd.DataFrame(keep_rows)
        apt = np.stack(apt_parts).astype(np.float32)
        smi = np.stack(smi_parts).astype(np.float32)
        return apt, smi, work.reset_index(drop=True)
    work = work.dropna(subset=[seq_cols[0], smi_cols[0]])
    if work.empty:
        raise ValueError("Supervision rows missing embedding values")
    apt = work[seq_cols].to_numpy(dtype=np.float32)
    smi = work[smi_cols].to_numpy(dtype=np.float32)
    return apt, smi, work.reset_index(drop=True)


def train_fraction_contrastive(
    train_pos: pd.DataFrame,
    train_neg: pd.DataFrame,
    val_pos: pd.DataFrame,
    val_neg: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    out_ckpt: Path,
    seed: int,
    epochs: int = DEFAULT_CONTRASTIVE_EPOCHS,
    device: str | None = None,
) -> Path:
    """Train MicroContrastiveModel from scratch on fraction-local supervision only."""
    ensure_contrastive_on_path()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import torch
    from torch.utils.data import DataLoader

    from DataPrepare import FinalContrastiveDataset
    from Loss import TemperatureScaledLoss
    from Model import MicroContrastiveModel

    raw_lookup = _prepare_raw_lookup(raw_df)
    apt_pos, smi_pos, pos_merged = _rows_to_embeddings(train_pos, raw_lookup)
    mol_pos = pos_merged["canonical_smiles"].to_numpy()

    if train_neg.empty:
        apt_neg = np.zeros((0, apt_pos.shape[1]), dtype=np.float32)
        smi_neg = np.zeros((0, smi_pos.shape[1]), dtype=np.float32)
        mol_neg = np.array([], dtype=object)
    else:
        apt_neg, smi_neg, neg_merged = _rows_to_embeddings(train_neg, raw_lookup)
        mol_neg = neg_merged["canonical_smiles"].to_numpy()

    use_protocol_val = not val_pos.empty

    train_ds = FinalContrastiveDataset(
        apt_pos,
        smi_pos,
        apt_neg,
        smi_neg,
        mol_keys_pos=mol_pos,
        mol_keys_neg=mol_neg if len(mol_neg) else None,
        negative_ratio=DEFAULT_NEGATIVE_RATIO,
        training=True,
    )

    def collate_fn(batch):
        return {
            "anchor_smi": torch.stack([b["anchor_smi"] for b in batch]),
            "positive_apts": torch.stack([b["positive_apts"] for b in batch]),
            "negative_apts": torch.stack([b["negative_apts"] for b in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = None
    if use_protocol_val:
        apt_val_pos, smi_val_pos, val_pos_m = _rows_to_embeddings(val_pos, raw_lookup)
        apt_val_neg, smi_val_neg, val_neg_m = _rows_to_embeddings(val_neg, raw_lookup)
        val_ds = FinalContrastiveDataset(
            apt_val_pos,
            smi_val_pos,
            apt_val_neg,
            smi_val_neg,
            mol_keys_pos=val_pos_m["canonical_smiles"].to_numpy(),
            mol_keys_neg=val_neg_m["canonical_smiles"].to_numpy(),
            negative_ratio=DEFAULT_NEGATIVE_RATIO,
            training=False,
            eval_seed=seed + 17,
        )
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MicroContrastiveModel().to(dev)
    criterion = TemperatureScaledLoss().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    patience = 3
    stale = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            anchor = batch["anchor_smi"].to(dev)
            positive = batch["positive_apts"].to(dev)
            negatives = batch["negative_apts"].to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = model(anchor, positive, negatives)
            loss, _ = criterion(out["z_anchor"], out["z_positive"], out["z_negatives"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = anchor.size(0)
            train_loss += float(loss.item()) * bs
            n_train += bs
        train_loss /= max(n_train, 1)

        if val_loader is None:
            print(f"    contrastive epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}", flush=True)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                anchor = batch["anchor_smi"].to(dev)
                positive = batch["positive_apts"].to(dev)
                negatives = batch["negative_apts"].to(dev)
                out = model(anchor, positive, negatives)
                loss, _ = criterion(out["z_anchor"], out["z_positive"], out["z_negatives"])
                bs = anchor.size(0)
                val_loss += float(loss.item()) * bs
                n_val += bs
        val_loss /= max(n_val, 1)
        print(f"    contrastive epoch {epoch + 1}/{epochs} val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed}, out_ckpt)
    return out_ckpt


def finetune_fraction_decoder(
    contrastive_ckpt: Path,
    train_pos: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    init_decoder: Path | None,
    out_decoder: Path,
    seed: int,
    epochs: int = DEFAULT_DECODER_EPOCHS,
    device: str | None = None,
    from_scratch: bool = True,
) -> Path:
    """Train GRU on fold train positives + fraction-local contrastive aux.

    Default ``from_scratch=True`` avoids leaking a globally pretrained decoder that
    saw the full experimental negative pool.
    """
    ensure_contrastive_on_path()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import torch
    from GRU import ConditionalGRUDecoder, train_conditional_decoder
    from Model import MicroContrastiveModel

    raw_lookup = _prepare_raw_lookup(raw_df)
    apt, smi, merged = _rows_to_embeddings(train_pos, raw_lookup)
    sequences = merged["sequence"].astype(str).tolist()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    contrastive = MicroContrastiveModel().to(dev)
    ckpt = torch.load(contrastive_ckpt, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    contrastive.load_state_dict(state)
    contrastive.eval()

    decoder = ConditionalGRUDecoder(
        mol_dim=768, latent_dim=768, hidden_dim=512, max_len=50, min_len=20, num_layers=2, dropout=0.3
    )
    if not from_scratch and init_decoder is not None:
        init = Path(init_decoder)
        if init.exists():
            decoder.load_state_dict(torch.load(init, map_location=dev, weights_only=False))
    decoder.to(dev)

    out_decoder.parent.mkdir(parents=True, exist_ok=True)
    train_conditional_decoder(
        decoder,
        smi,
        apt,
        sequences,
        epochs=epochs,
        batch_size=32,
        lr=8e-4,
        device=str(dev),
        scheduled_sampling_max=0.25,
        save_path=str(out_decoder),
        contrastive_model=contrastive,
        aux_loss_weight=0.08,
        aux_warmup_epochs=2,
        collapse_loss_weight=0.10,
    )
    return out_decoder


def _make_cached_aptamer_encode(device: Any, *, batch_size: int = 64):
    """GENA encode with de-dup cache; molecules reuse CSV, new GRU seqs hit GENA once."""
    from Embeddings import aptamer_encode

    cache: dict[str, np.ndarray] = {}

    def _encode(sequences, **_kwargs):
        keys = [str(s).upper().replace("U", "T") for s in sequences]
        missing = [k for k in dict.fromkeys(keys) if k not in cache]
        if missing:
            embs = aptamer_encode(missing, batch_size=batch_size, device=str(device))
            if embs is None:
                raise RuntimeError("aptamer_encode returned None")
            for seq, emb in zip(missing, embs):
                cache[seq] = np.asarray(emb, dtype=np.float32)
        return np.stack([cache[k] for k in keys]).astype(np.float32)

    return _encode, cache


def generate_fraction_local_pairs(
    *,
    target_molecules: pd.Series | np.ndarray,
    contrastive_ckpt: Path,
    decoder_ckpt: Path,
    raw_df: pd.DataFrame,
    supervision_neg: pd.DataFrame,
    out_csv: Path,
    seed: int,
    device: str | None = None,
) -> Path:
    """Generate GRU negatives for train-fold molecules using fraction-local checkpoints.

    Speed path: decode all molecules first (no per-mol GENA), then one batched GENA
    pass over unique generated sequences, then filter/rank (same scoring as before).
    Molecule ChemBERTa vectors stay CSV lookups (`smi_emb_*`).
    """
    ensure_contrastive_on_path()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import torch
    from GRU import (
        ConditionalGRUDecoder,
        build_generation_target_from_dataframe,
        generate_ranked_aptamers_for_molecule,
        rank_candidates_with_motif_penalty,
        score_and_filter_candidates_by_sequence_sim,
        select_diverse_output_candidates,
    )
    from Model import MicroContrastiveModel

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MicroContrastiveModel().to(dev)
    ckpt = torch.load(contrastive_ckpt, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    decoder = ConditionalGRUDecoder(
        mol_dim=768, latent_dim=768, hidden_dim=512, max_len=50, min_len=20, num_layers=2, dropout=0.3
    )
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=dev, weights_only=False))
    decoder.to(dev)
    decoder.eval()

    raw_lookup = _prepare_raw_lookup(raw_df)
    local_neg_latents = _encode_negative_latents(model, supervision_neg, raw_lookup, dev)
    encode_fn, _emb_cache = _make_cached_aptamer_encode(dev, batch_size=64)

    gen_cfg = {
        "n_latent_points": 24,
        "samples_per_latent": 3,
        "latent_jitter_copies": 1,
        "latent_jitter_std": 0.10,
        "diversity_threshold": 0.84,
        "temperature": 1.05,
        "top_k": 8,
        "max_latent_similarity": 0.15,
        "n_keep": 6,
        "max_latent_sim_for_decode": -0.05,
        "motif_penalty_weight": 0.15,
        "max_kmer_repeat": 2,
        "motif_kmer_sizes": (5, 6),
        "max_homopolymer_motif": 5,
        "reject_high_motif_repeat": True,
        "reject_overused_prefixes": True,
        "use_sequence_sim_filter": True,
        "use_structure_filter": False,
        "allow_motif_fallback": True,
        "max_same_prefix": 1,
        "motif_prefix_len": 8,
        "min_sequence_distance": 0.18,
        "max_motif_penalty_for_tools": 0.08,
    }
    rank_kwargs = dict(
        motif_penalty_weight=gen_cfg["motif_penalty_weight"],
        reject_high_motif_repeat=gen_cfg["reject_high_motif_repeat"],
        reject_overused_prefixes=gen_cfg["reject_overused_prefixes"],
        kmer_sizes=gen_cfg["motif_kmer_sizes"],
        max_kmer_repeat=gen_cfg["max_kmer_repeat"],
        max_homopolymer=gen_cfg["max_homopolymer_motif"],
        prefix_len=gen_cfg["motif_prefix_len"],
    )

    mols = pd.Series(target_molecules).astype(str).str.strip().unique().tolist()
    # Warm up GENA once (not per molecule).
    encode_fn(["ACGTACGTACGTACGTACGTACGT"])

    pending: list[dict[str, Any]] = []
    print(f"    [fast-gen] decode phase: {len(mols)} molecules (CSV mol emb, no per-mol GENA)", flush=True)
    for mol_idx, smiles in enumerate(mols):
        built = build_generation_target_from_dataframe(
            df=raw_lookup,
            target_smiles=smiles,
            contrastive_model=model,
            device=dev,
            global_negative_latents=local_neg_latents,
            local_negative_override=local_neg_latents if local_neg_latents.size else None,
        )
        if built is None:
            continue
        try:
            raw_cands = generate_ranked_aptamers_for_molecule(
                decoder=decoder,
                mol_embedding=built["mol_embedding"],
                local_negative_points=built.get("local_negative_latents", local_neg_latents),
                global_negative_points=local_neg_latents,
                device=dev,
                n_latent_points=gen_cfg["n_latent_points"],
                samples_per_latent=gen_cfg["samples_per_latent"],
                temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"],
                max_similarity=gen_cfg["max_latent_similarity"],
                diversity_threshold=gen_cfg["diversity_threshold"],
                latent_jitter_copies=gen_cfg["latent_jitter_copies"],
                latent_jitter_std=gen_cfg["latent_jitter_std"],
                min_seq_len=int(decoder.min_len),
                max_seq_len=int(decoder.max_len),
                use_structure_filter=gen_cfg["use_structure_filter"],
                skip_post_decode_scoring=True,
            )
        except Exception as exc:
            print(f"    generation skip {smiles[:40]}: {exc}", flush=True)
            continue
        if not raw_cands:
            continue
        pending.append({"mol_idx": mol_idx, "smiles": smiles, "built": built, "raw": raw_cands})

    all_seqs = sorted(
        {
            str(c["sequence"]).upper().replace("U", "T")
            for item in pending
            for c in item["raw"]
        }
    )
    print(f"    [fast-gen] batched GENA: {len(all_seqs)} unique sequences", flush=True)
    if all_seqs:
        encode_fn(all_seqs)

    rows: list[dict[str, Any]] = []
    for item in pending:
        built = item["built"]
        try:
            scored = score_and_filter_candidates_by_sequence_sim(
                candidates=item["raw"],
                raw_smi_embedding=built.get("raw_smi_embedding"),
                contrastive_model=model,
                device=dev,
                aptamer_encode_fn=encode_fn,
                baseline_positive_mean=built.get("positive_mean"),
                max_latent_sim=gen_cfg["max_latent_sim_for_decode"],
                use_sequence_sim_filter=gen_cfg["use_sequence_sim_filter"],
            )
            ranked = rank_candidates_with_motif_penalty(scored, **rank_kwargs)
        except Exception as exc:
            print(f"    score skip {item['smiles'][:40]}: {exc}", flush=True)
            continue
        picked = select_diverse_output_candidates(
            ranked or [],
            n_keep=gen_cfg["n_keep"],
            max_same_prefix=gen_cfg["max_same_prefix"],
            prefix_len=gen_cfg["motif_prefix_len"],
            min_sequence_distance=gen_cfg["min_sequence_distance"],
            max_motif_penalty=gen_cfg["max_motif_penalty_for_tools"],
            backfill_to_n_keep=gen_cfg["allow_motif_fallback"],
        )
        for rank, cand in enumerate(picked, start=1):
            decoded = cand.get("latent_sim", cand.get("sequence_sim"))
            rows.append(
                {
                    "molecule_index": item["mol_idx"],
                    "canonical_smiles": item["smiles"],
                    "aptamer_rank": rank,
                    "dna_sequence": cand["sequence"],
                    "sequence": cand["sequence"],
                    "decoded_sim": float(decoded) if decoded is not None else float("nan"),
                    "length": cand.get("length"),
                    "gc_content": cand.get("gc"),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    fraction-local generation: {len(rows)} seq / {len(mols)} mols -> {out_csv}", flush=True)
    return out_csv


def _encode_negative_latents(
    model: Any,
    supervision_neg: pd.DataFrame,
    raw_lookup: pd.DataFrame,
    device: Any,
) -> np.ndarray:
    import torch

    if supervision_neg.empty:
        return np.zeros((0, 768), dtype=np.float32)
    apt, _, _ = _rows_to_embeddings(supervision_neg, raw_lookup)
    with torch.no_grad():
        z = model.encode_aptamer(torch.tensor(apt, device=device)).cpu().numpy().astype(np.float32)
    return z


def build_fraction_local_bundle(
    *,
    split_type: str,
    fold: int,
    exp_fraction: float,
    train_pos: pd.DataFrame,
    train_neg_exp: pd.DataFrame,
    val_pos: pd.DataFrame,
    val_neg_exp: pd.DataFrame,
    full_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
    seed: int,
    init_decoder: Path | None = None,
    retrain_decoder: bool = True,
    contrastive_epochs: int = DEFAULT_CONTRASTIVE_EPOCHS,
    decoder_epochs: int = DEFAULT_DECODER_EPOCHS,
    device: str | None = None,
    need_generation: bool = True,
) -> FractionLocalBundle:
    """Retrain contrastive (+ optional GRU) and build fraction-local scorers/generators.

    need_generation=False skips GRU decode/GENA when only learned_real_hard is needed.
    """
    target_n = len(train_neg_exp)
    supervision_neg = fraction_local_supervision_negatives(
        train_neg_exp, target_n=target_n, exp_fraction=exp_fraction, rng=rng
    )
    contrastive_neg, contrastive_neg_source = contrastive_supervision_negatives(
        train_pos, supervision_neg, full_df, target_n=target_n, rng=rng
    )
    tag = f"{split_type}_fold{fold}_f{exp_fraction:.2f}".replace(".", "p")
    ckpt_dir = out_dir / "fraction_local_models" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    contrastive_ckpt = ckpt_dir / "micro_contrastive.pth"
    if not contrastive_ckpt.exists():
        print(
            f"  [fraction-local] training contrastive on "
            f"pos={len(train_pos)} neg={len(contrastive_neg)} "
            f"(f={exp_fraction:.0%}, source={contrastive_neg_source})",
            flush=True,
        )
        train_fraction_contrastive(
            train_pos,
            contrastive_neg,
            val_pos,
            val_neg_exp,
            raw_df,
            out_ckpt=contrastive_ckpt,
            seed=seed,
            epochs=contrastive_epochs,
            device=device,
        )
    else:
        print(f"  [fraction-local] reuse contrastive {contrastive_ckpt}", flush=True)

    decoder_ckpt = ckpt_dir / "conditional_decoder.pth"
    if not retrain_decoder:
        raise ValueError(
            "fraction_retrain protocol requires per-f decoder training "
            "(do not use --no-retrain-decoder; that reuses a global GRU and leaks full-pool supervision)"
        )
    if not decoder_ckpt.exists():
        print(
            f"  [fraction-local] training GRU decoder from scratch "
            f"({decoder_epochs} epochs; no global init)",
            flush=True,
        )
        finetune_fraction_decoder(
            contrastive_ckpt,
            train_pos,
            raw_df,
            init_decoder=None,
            out_decoder=decoder_ckpt,
            seed=seed + 1,
            epochs=decoder_epochs,
            device=device,
            from_scratch=True,
        )
    else:
        print(f"  [fraction-local] reuse decoder {decoder_ckpt}", flush=True)

    gen_csv = ckpt_dir / "generation_summary.csv"
    if need_generation and (not gen_csv.exists() or gen_csv.stat().st_size == 0):
        generate_fraction_local_pairs(
            target_molecules=train_neg_exp["canonical_smiles"],
            contrastive_ckpt=contrastive_ckpt,
            decoder_ckpt=decoder_ckpt,
            raw_df=raw_df,
            supervision_neg=supervision_neg,
            out_csv=gen_csv,
            seed=seed + 2,
            device=device,
        )
    elif not need_generation:
        print("  [fraction-local] skip GRU generation (not required for pending methods)", flush=True)
    else:
        print(f"  [fraction-local] reuse generation {gen_csv}", flush=True)

    bootstrap_neg = bootstrap_negatives_for_scorer(
        train_pos, supervision_neg, full_df, target_n=target_n, rng=rng
    )
    hard_score_fn = None
    try:
        hard_score_fn = build_bootstrap_lgbm_scorer(
            train_pos,
            bootstrap_neg,
            full_df,
            split_type,
            seed + 3,
        ).as_score_fn()
        print(
            f"  [fraction-local] bootstrap LGBM on pos={len(train_pos)} "
            f"neg={len(bootstrap_neg)} (f={exp_fraction:.0%})",
            flush=True,
        )
    except ValueError as exc:
        print(f"  WARNING: fraction-local bootstrap LGBM unavailable: {exc}", flush=True)

    return FractionLocalBundle(
        exp_fraction=float(exp_fraction),
        n_supervision_neg=int(len(supervision_neg)),
        contrastive_ckpt=contrastive_ckpt,
        generation_csv=gen_csv,
        hard_score_fn=hard_score_fn,
        notes=f"supervision_neg={len(supervision_neg)} contrastive_neg={len(contrastive_neg)} ({contrastive_neg_source}) bootstrap_neg={len(bootstrap_neg)}",
    )


def build_fraction_local_negative_dataset(
    method: str,
    train_pos: pd.DataFrame,
    train_neg_exp: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
    bundle: FractionLocalBundle,
) -> NegativeDataset:
    """Build learned_generated / learned_real_hard from a fraction-local bundle."""
    target_n = len(train_neg_exp)
    if method == "learned_generated":
        if bundle.generation_csv is None or not bundle.generation_csv.exists():
            return NegativeDataset(method, pd.DataFrame(), target_n, 0.0, "missing_fraction_local_generation")
        return build_controlled_negative_dataset(
            method,
            train_pos,
            train_neg_exp,
            full_df,
            rng,
            target_n=target_n,
            generated_path=str(bundle.generation_csv),
            score_fn=None,
        )
    if method == "learned_real_hard_negative":
        return build_controlled_negative_dataset(
            method,
            train_pos,
            train_neg_exp,
            full_df,
            rng,
            target_n=target_n,
            generated_path="",
            score_fn=bundle.hard_score_fn,
        )
    raise ValueError(f"Not a fraction-local learned method: {method}")

"""Load and clean AptaBench-style pair tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    BENCHMARK_ROOT / "data" / "aptabench" / "embeddings" / "aptabench_with_embeddings_v3_clamp.parquet"
)
DEFAULT_RAW = BENCHMARK_ROOT / "data" / "aptabench" / "dataset" / "AptaBench_dataset.csv"
DEFAULT_GENERATION = (
    BENCHMARK_ROOT / "data" / "aptabench" / "generation" / "train_fold_generation_summary.csv"
)
DEFAULT_CHECKPOINT = BENCHMARK_ROOT / "checkpoints" / "final_micro_model.pth"
APTABENCH_TABLE = BENCHMARK_ROOT / "data" / "aptabench" / "dataset" / "AptaBench_dataset.csv"
APTABENCH_SPLITS_DIR = BENCHMARK_ROOT / "data" / "aptabench" / "dataset" / "splits"
APTABENCH_SPLIT_FILES = {
    "random": "stratified.json",
    "stratified": "stratified.json",
    "aptamer": "disjoint_aptamer.json",
    "molecule": "disjoint_molecule.json",
}

def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace("U", "T")


def detect_alphabet(seq: str) -> str:
    raw = str(seq).strip().upper()
    return "ACGU" if "U" in raw and "T" not in raw else "ACGT"


def gc_content(seq: str) -> float:
    s = normalize_sequence(seq)
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def sequence_column(df: pd.DataFrame) -> str:
    for col in ("sequence", "aptamer", "dna_sequence"):
        if col in df.columns:
            return col
    raise ValueError("Need a sequence / aptamer column")


def smiles_column(df: pd.DataFrame) -> str:
    for col in ("canonical_smiles", "smiles"):
        if col in df.columns:
            return col
    raise ValueError("Need canonical_smiles or smiles")


def _annotate_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    seq_col = sequence_column(df)
    smi_col = smiles_column(df)
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["sequence_raw"] = out[seq_col].astype(str)
    out["sequence"] = out["sequence_raw"].map(normalize_sequence)
    out["canonical_smiles"] = out[smi_col].astype(str).str.strip()
    if "type" in out.columns:
        out["type"] = out["type"].astype(str).str.upper()
    else:
        out["type"] = [
            "RNA" if detect_alphabet(s) == "ACGU" else "DNA" for s in out["sequence_raw"]
        ]
    out["alphabet"] = [detect_alphabet(s) for s in out["sequence_raw"]]
    out["length"] = out["sequence"].str.len()
    out["gc_content"] = out["sequence"].map(gc_content)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_pairs(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    out = _annotate_pair_frame(_read_table(path))
    raw_n = len(out)
    out = out.drop_duplicates(["sequence", "canonical_smiles", "label"]).reset_index(drop=True)
    conflict = out.groupby(["sequence", "canonical_smiles"])["label"].transform("nunique")
    n_conflict = int((conflict > 1).sum())
    out = out.loc[conflict == 1].reset_index(drop=True)
    stats = {
        "path": str(path),
        "raw_rows": int(raw_n),
        "clean_rows": int(len(out)),
        "n_pos": int((out["label"] == 1).sum()),
        "n_neg": int((out["label"] == 0).sum()),
        "conflicting_rows_dropped": n_conflict,
        "n_molecules": int(out["canonical_smiles"].nunique()),
        "n_aptamers": int(out["sequence"].nunique()),
    }
    return out, stats


def load_aptabench_indexed_pairs(
    embeddings_path: str | Path = DEFAULT_DATA,
    *,
    aptabench_table: str | Path = APTABENCH_TABLE,
) -> tuple[pd.DataFrame, dict]:
    """6289 AptaBench rows in published order, embeddings joined by SMILES.

    JSON split indices refer to this table, not the cleaned embeddings CSV.
    Duplicate published rows are kept. Conflicting labels are kept as published.
    """
    aptabench_table = Path(aptabench_table)
    embeddings_path = Path(embeddings_path)
    if not aptabench_table.exists():
        raise FileNotFoundError(f"AptaBench table not found: {aptabench_table}")
    aptabench = _annotate_pair_frame(_read_table(aptabench_table))
    if len(aptabench) != 6289:
        raise ValueError(f"AptaBench table must have 6289 rows, got {len(aptabench)}")
    aptabench["aptabench_idx"] = np.arange(len(aptabench), dtype=int)

    emb = _annotate_pair_frame(_read_table(embeddings_path))
    emb_cols = [c for c in emb.columns if c not in aptabench.columns]
    smi_map = emb.drop_duplicates("canonical_smiles").set_index("canonical_smiles")[emb_cols]
    joined = aptabench.join(smi_map, on="canonical_smiles", how="left")
    n_smi_hit = int(joined["canonical_smiles"].isin(smi_map.index).sum())
    stats = {
        "path": str(embeddings_path),
        "aptabench_table": str(aptabench_table),
        "split_source": "aptabench_json",
        "raw_rows": int(len(aptabench)),
        "clean_rows": int(len(joined)),
        "n_pos": int((joined["label"] == 1).sum()),
        "n_neg": int((joined["label"] == 0).sum()),
        "conflicting_rows_dropped": 0,
        "n_molecules": int(joined["canonical_smiles"].nunique()),
        "n_aptamers": int(joined["sequence"].nunique()),
        "rows_with_smiles_embeddings": n_smi_hit,
        "embeddings_csv_rows": int(len(emb)),
    }
    return joined, stats


def embedding_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def known_pairs(df: pd.DataFrame, labels: set[int] | None = None) -> set[tuple[str, str]]:
    sub = df if labels is None else df.loc[df["label"].isin(labels)]
    return set(zip(sub["sequence"].astype(str), sub["canonical_smiles"].astype(str)))

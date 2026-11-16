"""Parse existing target-conditioned generation outputs (our method)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import normalize_sequence


def load_generated_pairs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "canonical_smiles",
                "sequence",
                "decoded_sim",
                "seed_sim",
                "length",
                "gc_content",
                "aptamer_rank",
            ]
        )
    df = pd.read_csv(path)
    seq_col = "dna_sequence" if "dna_sequence" in df.columns else "sequence"
    smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    out = pd.DataFrame(
        {
            "canonical_smiles": df[smi_col].astype(str).str.strip(),
            "sequence": df[seq_col].map(normalize_sequence),
            "decoded_sim": pd.to_numeric(df.get("decoded_sim", 0.0), errors="coerce"),
            "seed_sim": pd.to_numeric(df.get("seed_sim", float("nan")), errors="coerce"),
            "length": pd.to_numeric(df.get("length", df[seq_col].astype(str).str.len()), errors="coerce"),
            "gc_content": pd.to_numeric(df.get("gc_content", float("nan")), errors="coerce"),
            "aptamer_rank": pd.to_numeric(df.get("aptamer_rank", 0), errors="coerce"),
        }
    )
    out = out.dropna(subset=["sequence", "canonical_smiles"])
    out = out.loc[out["sequence"].str.len() >= 12].reset_index(drop=True)
    return out


def generated_by_molecule(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    # Lowest decoded_sim = strongest predicted non-binder for that molecule.
    ranked = df.sort_values(["canonical_smiles", "decoded_sim", "aptamer_rank"], ascending=[True, True, True])
    return {smi: g.reset_index(drop=True) for smi, g in ranked.groupby("canonical_smiles")}

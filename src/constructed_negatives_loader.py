"""Load published constructed-negative CSV tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from negative_dataset_controls import NegativeDataset

PUBLISHED_CONSTRUCTED_METHODS: tuple[str, ...] = (
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
    "hard_negative",
    "learned_generated",
    "experimental",
)

METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "learned_generated": ("learned_generated", "learned_generated_GRU_contrastive"),
    "random_pairing": ("random_pairing",),
    "hard_negative": ("hard_negative",),
    "random_sequence": ("random_sequence",),
    "composition_shuffle": ("composition_shuffle",),
    "structure_filtered": ("structure_filtered",),
    "experimental": ("experimental",),
}

PAIR_COLS = ("sequence", "canonical_smiles", "type", "label")

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in METHOD_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical


def constructed_split_key(split_type: str) -> str:
    st = str(split_type).strip().lower()
    if st in {"stratified", "random"}:
        return "random"
    return st


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    keep = [c for c in PAIR_COLS if c in out.columns]
    out = out.loc[:, keep].copy()
    if "label" not in out.columns:
        out["label"] = 0
    return out.reset_index(drop=True)


def _by_fold_path(
    root: Path,
    split_type: str,
    fold: int,
    method: str,
    scope: str,
) -> Path:
    split_key = constructed_split_key(split_type)
    names = METHOD_ALIASES.get(method, (method,))
    for name in names:
        path = root / "by_fold" / split_key / f"fold{fold}" / scope / f"{name}.csv"
        if path.exists():
            return path
    return root / "by_fold" / split_key / f"fold{fold}" / scope / f"{method}.csv"


@lru_cache(maxsize=64)
def _load_by_method_table(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _load_from_by_method(
    root: Path,
    split_type: str,
    fold: int,
    method: str,
    scope: str,
) -> tuple[pd.DataFrame, float, int, str]:
    split_key = constructed_split_key(split_type)
    candidates: list[Path] = []
    for name in METHOD_ALIASES.get(method, (method,)):
        candidates.extend(
            [
                root / "by_method" / f"{name}.csv",
                root / "by_method" / f"{name}_train_only.csv",
            ]
        )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        table = _load_by_method_table(str(path))
        if "scope" in table.columns:
            part = table.loc[
                (table["split_type"].astype(str) == split_key)
                & (table["fold"].astype(int) == int(fold))
                & (table["scope"].astype(str) == scope)
            ]
        elif scope != "train":
            continue
        else:
            part = table.loc[
                (table["split_type"].astype(str) == split_key)
                & (table["fold"].astype(int) == int(fold))
            ]
        if not part.empty:
            notes = ""
            if "notes" in part.columns and not part["notes"].dropna().empty:
                notes = str(part["notes"].dropna().iloc[0])
            matched = (
                float(part["matched_ligand_fraction"].iloc[0])
                if "matched_ligand_fraction" in part.columns
                else 0.0
            )
            target_n = int(part["target_n"].iloc[0]) if "target_n" in part.columns else len(part)
            frame = _normalize_frame(part)
            return frame, matched, target_n, notes or "constructed_by_method_csv"
    return pd.DataFrame(), 0.0, 0, ""


def load_constructed_negative_frame(
    root: str | Path,
    split_type: str,
    fold: int,
    method: str,
    scope: str = "train",
) -> pd.DataFrame:
    nd = load_constructed_negative_dataset(root, split_type, fold, method, scope=scope)
    return nd.frame


def load_constructed_negative_dataset(
    root: str | Path,
    split_type: str,
    fold: int,
    method: str,
    scope: str = "train",
) -> NegativeDataset:
    root = Path(root)
    path = _by_fold_path(root, split_type, fold, method, scope)
    if path.exists():
        frame = _normalize_frame(pd.read_csv(path))
        notes = "constructed_by_fold_csv"
        matched = 0.0
        target_n = len(frame)
        summary = root / "negatives_summary.csv"
        if summary.exists():
            summary_df = pd.read_csv(summary)
            split_key = constructed_split_key(split_type)
            row = summary_df.loc[
                (summary_df["split_type"].astype(str) == split_key)
                & (summary_df["fold"].astype(int) == int(fold))
                & (summary_df["scope"].astype(str) == scope)
                & (
                    summary_df["neg_method"]
                    .astype(str)
                    .isin(METHOD_ALIASES.get(method, (method,)))
                )
            ]
            if not row.empty:
                matched = float(row["matched_ligand_fraction"].iloc[0])
                target_n = int(row["n"].iloc[0])
        return NegativeDataset(method, frame, target_n, matched, notes)

    frame, matched, target_n, notes = _load_from_by_method(root, split_type, fold, method, scope)
    if frame.empty:
        return NegativeDataset(method, frame, 0, 0.0, f"missing_constructed_csv:{method}:{scope}")
    return NegativeDataset(method, frame, target_n or len(frame), matched, notes)


def bootstrap_expand_pool(
    base: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if target_n <= 0 or base.empty:
        return base.iloc[0:0].copy()
    if len(base) >= target_n:
        idx = rng.choice(len(base), size=target_n, replace=False)
    else:
        idx = rng.choice(len(base), size=target_n, replace=True)
    out = base.iloc[idx].reset_index(drop=True)
    out["label"] = 0
    return out

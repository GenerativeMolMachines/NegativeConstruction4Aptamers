"""Source-disjoint splits for AptaBench, only when a provenance id is reliable.

``source`` is a complete 8-way database tag. ``origin`` (DOI/PMID) and ``buffer``
are not used: coverage is below 95%, and one origin covers most experimental
negatives. Callers must go through ``reliable_provenance_columns`` and must not
invent a study-disjoint split from ``origin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data import normalize_sequence

PROVENANCE_CANDIDATES: tuple[str, ...] = ("source", "origin", "buffer")
MIN_COVERAGE = 0.95
MAX_TOP_FRACTION = 0.90
MAX_NUNIQUE = 80
MIN_HELD_OUT_POS = 30
MIN_HELD_OUT_NEG = 50
MIN_TRAIN_POS = 30
MIN_TRAIN_NEG = 30


@dataclass
class ProvenanceAudit:
    column: str
    present: bool
    coverage: float
    nunique: int
    top_value: str
    top_fraction: float
    reliable: bool
    reason: str


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def audit_provenance_column(df: pd.DataFrame, column: str) -> ProvenanceAudit:
    if column not in df.columns:
        return ProvenanceAudit(column, False, 0.0, 0, "", 1.0, False, "missing_column")
    values = _clean(df[column])
    coverage = float((values != "").mean()) if len(values) else 0.0
    nonempty = values.loc[values != ""]
    nunique = int(nonempty.nunique()) if len(nonempty) else 0
    if nonempty.empty:
        top_value, top_fraction = "", 1.0
    else:
        counts = nonempty.value_counts(normalize=True)
        top_value = str(counts.index[0])
        top_fraction = float(counts.iloc[0])
    reasons: list[str] = []
    if coverage < MIN_COVERAGE:
        reasons.append(f"coverage={coverage:.3f}<{MIN_COVERAGE}")
    if nunique < 2:
        reasons.append("nunique<2")
    if nunique > MAX_NUNIQUE:
        reasons.append(f"nunique={nunique}>{MAX_NUNIQUE}")
    if top_fraction > MAX_TOP_FRACTION:
        reasons.append(f"top_fraction={top_fraction:.3f}>{MAX_TOP_FRACTION}")
    return ProvenanceAudit(
        column=column,
        present=True,
        coverage=coverage,
        nunique=nunique,
        top_value=top_value,
        top_fraction=top_fraction,
        reliable=not reasons,
        reason="ok" if not reasons else ";".join(reasons),
    )


def audit_provenance(df: pd.DataFrame, columns: tuple[str, ...] = PROVENANCE_CANDIDATES) -> pd.DataFrame:
    rows = [audit_provenance_column(df, col) for col in columns]
    return pd.DataFrame([row.__dict__ for row in rows])


def reliable_provenance_columns(df: pd.DataFrame, columns: tuple[str, ...] = PROVENANCE_CANDIDATES) -> list[str]:
    audit = audit_provenance(df, columns)
    return audit.loc[audit["reliable"].astype(bool), "column"].astype(str).tolist()


def source_disjoint_holdouts(
    df: pd.DataFrame,
    *,
    column: str = "source",
    min_held_out_pos: int = MIN_HELD_OUT_POS,
    min_held_out_neg: int = MIN_HELD_OUT_NEG,
    min_train_pos: int = MIN_TRAIN_POS,
    min_train_neg: int = MIN_TRAIN_NEG,
) -> list[dict]:
    """Leave-one-source-out bundles. Empty if ``column`` fails the reliability audit."""
    if column not in reliable_provenance_columns(df, (column,)):
        return []
    tagged = df.copy()
    tagged["_prov"] = _clean(tagged[column])
    tagged = tagged.loc[tagged["_prov"] != ""].copy()
    holdouts: list[dict] = []
    for source, held in tagged.groupby("_prov", sort=True):
        train = tagged.loc[tagged["_prov"] != source]
        test_pos = held.loc[held["label"] == 1]
        test_neg = held.loc[held["label"] == 0]
        train_pos = train.loc[train["label"] == 1]
        train_neg = train.loc[train["label"] == 0]
        if (
            len(test_pos) < min_held_out_pos
            or len(test_neg) < min_held_out_neg
            or len(train_pos) < min_train_pos
            or len(train_neg) < min_train_neg
        ):
            continue
        test_mols = set(held["canonical_smiles"].astype(str))
        train_mols = set(train["canonical_smiles"].astype(str))
        test_apts = set(held["sequence"].astype(str))
        train_apts = set(train["sequence"].astype(str))
        holdouts.append(
            {
                "provenance_column": column,
                "held_out_source": str(source),
                "train_pos": train_pos.drop(columns=["_prov"]).reset_index(drop=True),
                "train_neg_exp": train_neg.drop(columns=["_prov"]).reset_index(drop=True),
                "test_pos": test_pos.drop(columns=["_prov"]).reset_index(drop=True),
                "test_neg_exp": test_neg.drop(columns=["_prov"]).reset_index(drop=True),
                "n_train_pos": int(len(train_pos)),
                "n_train_neg": int(len(train_neg)),
                "n_test_pos": int(len(test_pos)),
                "n_test_neg": int(len(test_neg)),
                "frac_test_molecules_in_train": float(len(test_mols & train_mols) / max(len(test_mols), 1)),
                "frac_test_aptamers_in_train": float(len(test_apts & train_apts) / max(len(test_apts), 1)),
            }
        )
    return holdouts


def _seq_set(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty:
        return set()
    return set(frame["sequence"].map(normalize_sequence))


def _mol_set(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty:
        return set()
    return set(frame["canonical_smiles"].astype(str).str.strip())


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame is None or frame.empty:
        return set()
    seq = frame["sequence"].map(normalize_sequence)
    mol = frame["canonical_smiles"].astype(str).str.strip()
    return set(zip(seq, mol))


def _drop_entity_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Drop train rows that share an aptamer, a molecule, or an exact pair with test."""
    if train.empty or test.empty:
        return train
    ban_seq = _seq_set(test)
    ban_mol = _mol_set(test)
    ban_pairs = _pair_set(test)
    seq = train["sequence"].map(normalize_sequence)
    mol = train["canonical_smiles"].astype(str).str.strip()
    pairs = list(zip(seq, mol))
    keep = [
        (s not in ban_seq) and (m not in ban_mol) and ((s, m) not in ban_pairs)
        for s, m in pairs
    ]
    return train.loc[keep].reset_index(drop=True)


def aptabench_fold_source_holdouts(
    folds: list,
    *,
    split_type: str,
    column: str = "source",
    min_test_pos: int = 20,
    min_test_neg: int = 20,
    min_train_pos: int = 30,
    min_train_neg: int = 30,
) -> list[dict]:
    """Source holdout inside each AptaBench fold, with entity overlap removed.

    Test pairs are the AptaBench test fold restricted to one source.
    Train pairs are the AptaBench training fold without that source, then
    without any aptamer or molecule that appears in that test slice.
    A cell is skipped unless both overlaps are exactly zero and the size
    gates pass. This is not a new split of the corpus.
    """
    cells: list[dict] = []
    for split in folds:
        fold_i = int(split.stats["fold"])
        parts = {
            "train_pos": split.train_pos,
            "train_neg": split.train_neg_exp,
            "test_pos": split.test_pos,
            "test_neg": split.test_neg_exp,
        }
        if any(column not in frame.columns for frame in parts.values()):
            raise ValueError(f"AptaBench fold frames lack column {column!r}")
        tagged = {name: frame.copy() for name, frame in parts.items()}
        for frame in tagged.values():
            frame["_prov"] = _clean(frame[column])
        sources = sorted(
            {
                s
                for frame in tagged.values()
                for s in frame["_prov"].unique()
                if s
            }
        )
        for source in sources:
            test_pos = tagged["test_pos"].loc[tagged["test_pos"]["_prov"] == source]
            test_neg = tagged["test_neg"].loc[tagged["test_neg"]["_prov"] == source]
            train_pos_src = tagged["train_pos"].loc[tagged["train_pos"]["_prov"] != source]
            train_neg_src = tagged["train_neg"].loc[tagged["train_neg"]["_prov"] != source]
            n_pos_before = int(len(train_pos_src))
            n_neg_before = int(len(train_neg_src))
            test_all = pd.concat([test_pos, test_neg], ignore_index=True)
            train_pos = _drop_entity_overlap(train_pos_src, test_all)
            train_neg = _drop_entity_overlap(train_neg_src, test_all)
            test_mols = _mol_set(test_all)
            test_apts = _seq_set(test_all)
            train_mols = _mol_set(train_pos) | _mol_set(train_neg)
            train_apts = _seq_set(train_pos) | _seq_set(train_neg)
            mol_overlap = len(test_mols & train_mols)
            apt_overlap = len(test_apts & train_apts)
            pair_overlap = len(_pair_set(test_all) & (_pair_set(train_pos) | _pair_set(train_neg)))
            same_source = int(
                ((train_pos["_prov"] == source).sum() if len(train_pos) else 0)
                + ((train_neg["_prov"] == source).sum() if len(train_neg) else 0)
            )
            reason = ""
            if mol_overlap or apt_overlap or pair_overlap or same_source:
                reason = "leak_after_filter"
            elif len(test_pos) < min_test_pos or len(test_neg) < min_test_neg:
                reason = "test_too_small"
            elif len(train_pos) < min_train_pos or len(train_neg) < min_train_neg:
                reason = "train_too_small"
            drop_prov = ["_prov"]
            cells.append(
                {
                    "split_type": split_type,
                    "fold": int(fold_i),
                    "held_out_source": str(source),
                    "skipped": bool(reason),
                    "skip_reason": reason,
                    "train_pos": train_pos.drop(columns=drop_prov).reset_index(drop=True),
                    "train_neg_exp": train_neg.drop(columns=drop_prov).reset_index(drop=True),
                    "test_pos": test_pos.drop(columns=drop_prov).reset_index(drop=True),
                    "test_neg_exp": test_neg.drop(columns=drop_prov).reset_index(drop=True),
                    "n_train_pos": int(len(train_pos)),
                    "n_train_neg": int(len(train_neg)),
                    "n_test_pos": int(len(test_pos)),
                    "n_test_neg": int(len(test_neg)),
                    "n_train_pos_before_entity_filter": n_pos_before,
                    "n_train_neg_before_entity_filter": n_neg_before,
                    "n_test_molecules_in_train": int(mol_overlap),
                    "n_test_aptamers_in_train": int(apt_overlap),
                    "n_test_pairs_in_train": int(pair_overlap),
                    "n_train_rows_same_source": int(same_source),
                }
            )
    return cells

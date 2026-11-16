"""Evaluation helpers."""

from __future__ import annotations

import pandas as pd

__all__ = ["cross_negative_pivot"]


def cross_negative_pivot(cross_csv: str | pd.DataFrame) -> pd.DataFrame:
    """Train-negative x test-negative mean ROC-AUC heatmap table."""
    df = pd.read_csv(cross_csv) if isinstance(cross_csv, str) else cross_csv
    if "skipped" in df.columns:
        df = df[~df["skipped"].fillna(False).astype(bool)]
    value_col = "roc_auc" if "roc_auc" in df.columns else "test_auc"
    train_col = "train_negative" if "train_negative" in df.columns else "train_method"
    test_col = "test_negative" if "test_negative" in df.columns else "test_method"
    return df.groupby([train_col, test_col])[value_col].mean().unstack()

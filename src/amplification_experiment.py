"""Fixed 10% experimental-subset amplification experiment (LightGBM / molecule-disjoint)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constructed_negatives_loader import load_constructed_negative_frame
from data import load_official_indexed_pairs
from ere_lgbm import _eval_model, _fit_model, _labeled_eval, make_folds

AMPLIFICATION_PROTOCOL = "amplification_exp_v1"
EXP_FRACTION = 0.1

MODES = (
    "exp_10_only",
    "exp_10_oversample",
    "exp_10_plus_rp",
    "exp_10_plus_generated",
)


def _sample_n(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    if frame.empty:
        return frame.copy()
    replace = len(frame) < n
    idx = rng.choice(len(frame), size=n, replace=replace)
    return frame.iloc[idx].reset_index(drop=True)


def fixed_exp_subset(
    train_neg_exp: pd.DataFrame,
    *,
    exp_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_total = len(train_neg_exp)
    if n_total == 0:
        return train_neg_exp.iloc[0:0].copy()
    n_exp = max(1, int(round(exp_fraction * n_total)))
    n_exp = min(n_exp, n_total)
    if n_exp >= n_total:
        return train_neg_exp.copy().reset_index(drop=True)
    idx = rng.choice(n_total, size=n_exp, replace=False)
    return train_neg_exp.iloc[idx].reset_index(drop=True)


def amplification_train_negatives(
    subset_exp: pd.DataFrame,
    *,
    mode: str,
    total_n: int,
    synthetic_pool: pd.DataFrame | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int, int]:
    if mode == "exp_10_only":
        out = subset_exp.copy()
        if not out.empty:
            out["label"] = 0
        return out, len(out), 0

    if mode == "exp_10_oversample":
        out = _sample_n(subset_exp, total_n, rng)
        if not out.empty:
            out = out.copy()
            out["label"] = 0
        return out, len(out), 0

    if mode in ("exp_10_plus_rp", "exp_10_plus_generated"):
        n_exp = len(subset_exp)
        n_syn = max(0, total_n - n_exp)
        parts: list[pd.DataFrame] = []
        if n_exp > 0:
            parts.append(subset_exp.copy())
        if n_syn > 0:
            pool = synthetic_pool if synthetic_pool is not None else pd.DataFrame()
            if pool.empty:
                raise ValueError(f"empty synthetic pool for mode={mode}")
            parts.append(_sample_n(pool, n_syn, rng))
        if not parts:
            return subset_exp.iloc[0:0].copy(), 0, 0
        out = pd.concat(parts, ignore_index=True)
        out["label"] = 0
        return out, n_exp, n_syn

    raise ValueError(f"unknown amplification mode: {mode}")


def _build_synthetic_pools(
    *,
    split_type: str,
    fold: int,
    constructed_negatives_root: str | Path,
) -> dict[str, pd.DataFrame]:
    pools: dict[str, pd.DataFrame] = {}
    for method in ("random_pairing", "learned_generated"):
        frame = load_constructed_negative_frame(
            constructed_negatives_root,
            split_type,
            fold,
            method,
            scope="train",
        )
        if frame.empty:
            raise FileNotFoundError(
                f"Missing constructed train/{split_type}/fold{fold}/{method}"
            )
        pools[method] = frame
        print(f"  pool {method}: {len(frame)} (constructed CSV)", flush=True)
    return pools


def _eval_cell(
    row_base: dict[str, Any],
    train_pos: pd.DataFrame,
    train_neg: pd.DataFrame,
    *,
    n_exp: int,
    n_syn: int,
    exp_test_eval: pd.DataFrame,
    df: pd.DataFrame,
    split_type: str,
    seed: int,
) -> dict[str, Any]:
    row = {
        **row_base,
        "n_train_neg": len(train_neg),
        "n_train_neg_experimental": int(n_exp),
        "n_train_neg_synthetic": int(n_syn),
    }
    if train_neg.empty:
        return {**row, "skipped": True, "skip_reason": "empty_train_negatives"}
    train_frame = _labeled_eval(train_pos, train_neg)
    try:
        model, threshold, scaler = _fit_model(train_frame, df, split_type, seed)
        test_metrics = _eval_model(model, exp_test_eval, df, split_type, threshold, scaler)
    except ValueError as exc:
        return {**row, "skipped": True, "skip_reason": str(exc)}
    return {
        **row,
        "skipped": False,
        "skip_reason": "",
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_mcc": test_metrics["mcc"],
        "threshold": threshold,
    }


def _save_checkpoint(
    out: Path,
    metrics: pd.DataFrame,
    subset_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "amplification_metrics.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(out / "fixed_exp_subsets.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_checkpoint(out: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    path = out / "amplification_metrics.csv"
    if not path.exists():
        return [], set()
    metrics = pd.read_csv(path)
    if metrics.empty:
        return [], set()
    done = {
        (r["split_type"], int(r["fold"]), str(r["mode"]))
        for _, r in metrics.iterrows()
        if not bool(r.get("skipped", False))
    }
    return metrics.to_dict("records"), done


def run_amplification_experiment(
    *,
    data_path: Path,
    out: Path,
    split_type: str = "molecule",
    n_folds: int = 5,
    seed: int = 42,
    exp_fraction: float = EXP_FRACTION,
    resume: bool = True,
    constructed_negatives_root: str | Path,
) -> pd.DataFrame:
    """Run fixed-subset amplification modes on one split type."""
    root = Path(constructed_negatives_root)
    if not root.is_dir():
        raise FileNotFoundError(f"constructed_negatives_root not found: {root}")

    df, _ = load_official_indexed_pairs(data_path)
    folds = make_folds(df, split_type, n_folds, seed)

    metric_rows, done = _load_checkpoint(out) if resume else ([], set())
    subset_rows: list[dict[str, Any]] = []
    subset_path = out / "fixed_exp_subsets.csv"
    if subset_path.exists():
        subset_rows = pd.read_csv(subset_path).to_dict("records")

    meta = {
        "protocol": AMPLIFICATION_PROTOCOL,
        "split_type": split_type,
        "n_folds": n_folds,
        "seed": seed,
        "exp_fraction": exp_fraction,
        "modes": list(MODES),
        "constructed_negatives_root": str(root),
    }

    print(f"\n=== {AMPLIFICATION_PROTOCOL}: {split_type} ===", flush=True)

    for fold, split in enumerate(folds):
        print(f"\n--- {split_type}/fold{fold} ---", flush=True)
        rng = np.random.default_rng(seed + fold)
        target_n = len(split.train_neg_exp)
        exp_test_eval = _labeled_eval(split.test_pos, split.test_neg_exp)

        subset = fixed_exp_subset(split.train_neg_exp, exp_fraction=exp_fraction, rng=rng)
        subset_rows.append(
            {
                "split_type": split_type,
                "fold": fold,
                "n_train_neg_exp_total": target_n,
                "n_subset_exp": len(subset),
                "exp_fraction_target": exp_fraction,
                "seed": seed + fold,
            }
        )
        print(f"  fixed subset: {len(subset)}/{target_n} experimental negatives", flush=True)

        syn_pools = _build_synthetic_pools(
            split_type=split_type,
            fold=fold,
            constructed_negatives_root=root,
        )

        for mode in MODES:
            ck = (split_type, fold, mode)
            if ck in done:
                continue

            row_base = {
                "protocol": AMPLIFICATION_PROTOCOL,
                "split_type": split_type,
                "fold": fold,
                "mode": mode,
                "exp_fraction": exp_fraction,
                "n_train_pos": len(split.train_pos),
                "n_target_neg": target_n,
                "n_subset_exp": len(subset),
            }

            syn_pool: pd.DataFrame | None = None
            if mode == "exp_10_plus_rp":
                syn_pool = syn_pools.get("random_pairing", pd.DataFrame())
            elif mode == "exp_10_plus_generated":
                syn_pool = syn_pools.get("learned_generated", pd.DataFrame())

            try:
                train_neg, n_exp, n_syn = amplification_train_negatives(
                    subset,
                    mode=mode,
                    total_n=target_n,
                    synthetic_pool=syn_pool,
                    rng=rng,
                )
            except ValueError as exc:
                metric_rows.append(
                    {
                        **row_base,
                        "skipped": True,
                        "skip_reason": str(exc),
                        "n_train_neg": 0,
                        "n_train_neg_experimental": 0,
                        "n_train_neg_synthetic": 0,
                    }
                )
                done.add(ck)
                _save_checkpoint(out, pd.DataFrame(metric_rows), subset_rows, meta)
                print(f"  SKIP {mode}: {exc}", flush=True)
                continue

            metric_rows.append(
                _eval_cell(
                    row_base,
                    split.train_pos,
                    train_neg,
                    n_exp=n_exp,
                    n_syn=n_syn,
                    exp_test_eval=exp_test_eval,
                    df=df,
                    split_type=split_type,
                    seed=seed + fold,
                )
            )
            done.add(ck)
            _save_checkpoint(out, pd.DataFrame(metric_rows), subset_rows, meta)
            last = metric_rows[-1]
            if last.get("skipped"):
                print(f"  SKIP {mode}: {last.get('skip_reason')}", flush=True)
            else:
                print(
                    f"  {mode} (neg={last['n_train_neg']}, "
                    f"exp={last['n_train_neg_experimental']}, syn={last['n_train_neg_synthetic']}): "
                    f"AUC={last['test_roc_auc']:.4f}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows)
    _save_checkpoint(out, metrics, subset_rows, meta)
    return metrics


def summarize_amplification(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["mode", "mean_roc_auc", "std_roc_auc", "n_folds"])
    skipped = metrics.get("skipped", pd.Series(False, index=metrics.index)).fillna(False).astype(bool)
    ok = metrics.loc[~skipped].copy()
    if ok.empty:
        return pd.DataFrame(columns=["mode", "mean_roc_auc", "std_roc_auc", "n_folds"])
    g = ok.groupby("mode", as_index=False)["test_roc_auc"]
    summary = g.agg(mean_roc_auc="mean", std_roc_auc="std", n_folds="count")
    mode_order = {m: i for i, m in enumerate(MODES)}
    summary["_ord"] = summary["mode"].map(mode_order)
    return summary.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

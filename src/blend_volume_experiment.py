"""Blend + volume negative experiments (train/test only).

Two complementary axes (same LightGBM, experimental-only test):

1. **blend** — fixed total negatives ``N = |train_neg_exp|``:
   ``n_exp = round(f * N)`` experimental + ``n_syn = N - n_exp`` synthetic.
   f ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}.

2. **volume** — pure synthetic, count relative to train positives:
   ``n_syn = round(r * |train_pos|)``, r ∈ {0.25, 0.5, 1, 2, 4}.
   Separates generation-method effect from negative-count effect.

Synthetic pools are built once per (split, fold, method) at size
``max(N, 4 * |train_pos|)``. Learned methods use the provided generation CSV /
fold-bootstrap LGBM on **full** train experimental negatives (this arm is the
classic blend/volume study; leak-free per-f retrain is ``fraction_retrain_v2``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aptabench_features import feature_spec_for_split
from data import DEFAULT_DATA, load_aptabench_indexed_pairs
from lgbm_scoring import HARD_SCORE_METHODS, build_bootstrap_lgbm_scorer
from negative_dataset_controls import (
    NEGATIVE_METHODS,
    build_controlled_negative_dataset,
)
from ere_lgbm import (
    SPLIT_TYPES,
    _eval_model,
    _fit_model,
    _labeled_eval,
    make_folds,
)

BLEND_PROTOCOL = "blend_volume_v1"
DEFAULT_BLEND_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
DEFAULT_VOLUME_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_METHODS = (
    "random_pairing",
    "learned_generated",
    "hard_negative",
    "composition_shuffle",
    "random_sequence",
)


def _sample_n(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    if frame.empty:
        return frame.copy()
    if len(frame) >= n:
        idx = rng.choice(len(frame), size=n, replace=False)
    else:
        idx = rng.choice(len(frame), size=n, replace=True)
    return frame.iloc[idx].reset_index(drop=True)


def blend_train_negatives(
    train_neg_exp: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    exp_fraction: float,
    total_n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int, int]:
    """Return (train_neg, n_exp, n_syn) with fixed total_n."""
    f = float(np.clip(exp_fraction, 0.0, 1.0))
    n_exp = int(round(f * total_n))
    n_exp = min(max(n_exp, 0), total_n)
    n_syn = total_n - n_exp
    parts: list[pd.DataFrame] = []
    if n_exp > 0:
        parts.append(_sample_n(train_neg_exp, n_exp, rng))
    if n_syn > 0:
        parts.append(_sample_n(synthetic, n_syn, rng))
    if not parts:
        return train_neg_exp.iloc[0:0].copy(), 0, 0
    out = pd.concat(parts, ignore_index=True)
    out["label"] = 0
    return out, n_exp, n_syn


def volume_train_negatives(
    synthetic: pd.DataFrame,
    *,
    n_pos: int,
    syn_to_pos_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    n_syn = int(round(float(syn_to_pos_ratio) * max(n_pos, 1)))
    out = _sample_n(synthetic, n_syn, rng)
    if not out.empty:
        out = out.copy()
        out["label"] = 0
    return out, n_syn


def run_blend_volume_experiment(
    *,
    data_path: str | Path = DEFAULT_DATA,
    generated_path: str | Path,
    out_dir: str | Path,
    methods: list[str] | None = None,
    split_types: list[str] | None = None,
    blend_fractions: list[float] | tuple[float, ...] | None = None,
    volume_ratios: list[float] | tuple[float, ...] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    lgbm_hard_score: bool = True,
    run_blend: bool = True,
    run_volume: bool = True,
    constructed_root: str | Path | None = None,
) -> pd.DataFrame:
    methods = [m for m in (methods or list(DEFAULT_METHODS)) if m != "experimental"]
    if not methods:
        raise ValueError("Need at least one synthetic method")
    split_types = split_types or ["aptamer", "molecule"]
    blend_fractions = tuple(blend_fractions or DEFAULT_BLEND_FRACTIONS)
    volume_ratios = tuple(volume_ratios or DEFAULT_VOLUME_RATIOS)
    rng = np.random.default_rng(seed)
    df, data_stats = load_aptabench_indexed_pairs(data_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_rows, done = _load_checkpoint(out)
    if done:
        print(f"Resuming ({BLEND_PROTOCOL}) with {len(done)} completed cells...", flush=True)

    meta = {
        "protocol": BLEND_PROTOCOL,
        "methods": methods,
        "split_types": split_types,
        "blend_fractions": list(blend_fractions),
        "volume_ratios": list(volume_ratios),
        "blend_constructed_root": str(constructed_root) if constructed_root is not None else None,
        "n_folds": n_folds,
        "seed": seed,
        "lgbm_hard_score": lgbm_hard_score,
        "split_source": "aptabench_json",
        "data_stats": data_stats,
        "note": (
            "AptaBench JSON folds (val_idx=test). "
            "blend: fixed N=|train_neg_exp|; volume: pure syn, n=r*|train_pos|; "
            "train/test only; learned pool=generation CSV; hard scorer=full train exp"
        ),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for split_type in split_types:
        spec = feature_spec_for_split(split_type)
        print(
            f"\n=== {BLEND_PROTOCOL}: {split_type} | "
            f"{spec.aptamer_encoder}+{spec.molecule_encoder} ===",
            flush=True,
        )
        folds = make_folds(df, split_type, n_folds, seed)
        for fold, split in enumerate(folds):
            print(f"\n--- {split_type}/fold{fold} ---", flush=True)
            target_n = len(split.train_neg_exp)
            n_pos = len(split.train_pos)
            pool_n = max(target_n, int(max(volume_ratios) * n_pos) + 8)
            exp_test_eval = _labeled_eval(split.test_pos, split.test_neg_exp)

            hard_score_fn = None
            needs_generated_pool = run_volume or (run_blend and constructed_root is None)
            if needs_generated_pool and lgbm_hard_score and any(m in HARD_SCORE_METHODS for m in methods):
                try:
                    hard_score_fn = build_bootstrap_lgbm_scorer(
                        split.train_pos,
                        split.train_neg_exp,
                        df,
                        split_type,
                        seed + fold,
                    ).as_score_fn()
                    print(
                        f"  Bootstrap LGBM hard scorer "
                        f"(pos={n_pos}, neg={target_n})",
                        flush=True,
                    )
                except ValueError as exc:
                    print(f"  WARNING: hard scorer unavailable: {exc}", flush=True)

            syn_pools: dict[str, pd.DataFrame] = {}
            for method in methods if needs_generated_pool else []:
                score_fn = (
                    hard_score_fn
                    if method in HARD_SCORE_METHODS and hard_score_fn is not None
                    else None
                )
                try:
                    neg_ds = build_controlled_negative_dataset(
                        method,
                        split.train_pos,
                        split.train_neg_exp,
                        df,
                        rng,
                        target_n=pool_n,
                        generated_path=str(generated_path),
                        score_fn=score_fn,
                        candidate_positives=split.train_pos,
                    )
                    syn_pools[method] = neg_ds.frame.reset_index(drop=True)
                    print(
                        f"  pool {method}: {len(syn_pools[method])} "
                        f"(requested {pool_n})",
                        flush=True,
                    )
                except Exception as exc:
                    syn_pools[method] = pd.DataFrame()
                    print(f"  SKIP pool {method}: {exc}", flush=True)

            if run_blend:
                for method in methods:
                    if constructed_root is None:
                        syn = syn_pools.get(method, pd.DataFrame())
                    else:
                        path = Path(constructed_root) / "by_fold" / split_type / f"fold{fold}" / "train" / f"{method}.csv"
                        syn = pd.read_csv(path)
                        if len(syn) != target_n:
                            raise ValueError(f"Expected {target_n} pairs in {path}, got {len(syn)}")
                    for frac in blend_fractions:
                        ck = _checkpoint_key(split_type, fold, method, "blend", float(frac))
                        if ck in done:
                            continue
                        row_base = {
                            "protocol": BLEND_PROTOCOL,
                            "arm": "blend",
                            "split_type": split_type,
                            "fold": fold,
                            "train_negative": method,
                            "exp_fraction": float(frac),
                            "syn_to_pos_ratio": np.nan,
                            "n_train_pos": n_pos,
                            "aptamer_encoder": spec.aptamer_encoder,
                            "molecule_encoder": spec.molecule_encoder,
                        }
                        if float(frac) < 1.0 and syn.empty:
                            metric_rows.append(
                                {
                                    **row_base,
                                    "skipped": True,
                                    "skip_reason": "empty_synthetic_pool",
                                    "n_train_neg": 0,
                                    "n_train_neg_experimental": 0,
                                    "n_train_neg_synthetic": 0,
                                }
                            )
                            done.add(ck)
                            _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, meta)
                            continue
                        if constructed_root is not None and float(frac) == 0:
                            train_neg, n_exp, n_syn = syn, 0, len(syn)
                        else:
                            train_neg, n_exp, n_syn = blend_train_negatives(
                                split.train_neg_exp,
                                syn,
                                exp_fraction=float(frac),
                                total_n=target_n,
                                rng=rng,
                            )
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
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, meta)
                        last = metric_rows[-1]
                        if last.get("skipped"):
                            print(
                                f"  SKIP blend {method}@f={frac:.0%}: {last.get('skip_reason')}",
                                flush=True,
                            )
                        else:
                            print(
                                f"  blend {method} f={frac:.0%} "
                                f"(exp={n_exp}, syn={n_syn}): "
                                f"AUC={last['test_roc_auc']:.4f} MCC={last['test_mcc']:.4f}",
                                flush=True,
                            )

            if run_volume:
                for method in methods:
                    syn = syn_pools.get(method, pd.DataFrame())
                    for ratio in volume_ratios:
                        ck = _checkpoint_key(split_type, fold, method, "volume", float(ratio))
                        if ck in done:
                            continue
                        row_base = {
                            "protocol": BLEND_PROTOCOL,
                            "arm": "volume",
                            "split_type": split_type,
                            "fold": fold,
                            "train_negative": method,
                            "exp_fraction": 0.0,
                            "syn_to_pos_ratio": float(ratio),
                            "n_train_pos": n_pos,
                            "aptamer_encoder": spec.aptamer_encoder,
                            "molecule_encoder": spec.molecule_encoder,
                        }
                        if syn.empty:
                            metric_rows.append(
                                {
                                    **row_base,
                                    "skipped": True,
                                    "skip_reason": "empty_synthetic_pool",
                                    "n_train_neg": 0,
                                    "n_train_neg_experimental": 0,
                                    "n_train_neg_synthetic": 0,
                                }
                            )
                            done.add(ck)
                            _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, meta)
                            continue
                        train_neg, n_syn = volume_train_negatives(
                            syn, n_pos=n_pos, syn_to_pos_ratio=float(ratio), rng=rng
                        )
                        metric_rows.append(
                            _eval_cell(
                                row_base,
                                split.train_pos,
                                train_neg,
                                n_exp=0,
                                n_syn=n_syn,
                                exp_test_eval=exp_test_eval,
                                df=df,
                                split_type=split_type,
                                seed=seed + fold,
                            )
                        )
                        done.add(ck)
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, meta)
                        last = metric_rows[-1]
                        if last.get("skipped"):
                            print(
                                f"  SKIP volume {method}@r={ratio:g}x: {last.get('skip_reason')}",
                                flush=True,
                            )
                        else:
                            print(
                                f"  volume {method} r={ratio:g}x (syn={n_syn}): "
                                f"AUC={last['test_roc_auc']:.4f} MCC={last['test_mcc']:.4f}",
                                flush=True,
                            )

    metrics = pd.DataFrame(metric_rows)
    _save_checkpoint(out, metrics, data_stats, meta)
    return metrics


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


def _checkpoint_key(
    split_type: str,
    fold: int,
    method: str,
    arm: str,
    value: float,
) -> tuple[Any, ...]:
    return (BLEND_PROTOCOL, split_type, int(fold), str(method), str(arm), float(value))


def _load_checkpoint(out: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    path = out / "blend_volume_metrics.csv"
    if not path.exists():
        return [], set()
    metrics = pd.read_csv(path)
    if metrics.empty:
        return [], set()
    done = {
        _checkpoint_key(s, f, m, a, v)
        for s, f, m, a, v in zip(
            metrics["split_type"],
            metrics["fold"],
            metrics["train_negative"],
            metrics["arm"],
            metrics.apply(
                lambda r: r["exp_fraction"] if r["arm"] == "blend" else r["syn_to_pos_ratio"],
                axis=1,
            ),
        )
    }
    return metrics.to_dict(orient="records"), done


def _save_checkpoint(
    out: Path,
    metrics: pd.DataFrame,
    data_stats: dict,
    meta: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "blend_volume_metrics.csv", index=False)
    summary = _summarize(metrics)
    summary.to_csv(out / "blend_volume_summary.csv", index=False)
    _write_report(metrics, summary, out / "REPORT.md", data_stats, meta)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ok = metrics.loc[~metrics.get("skipped", False).fillna(False).astype(bool)].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["arm", "split_type", "train_negative", "exp_fraction", "syn_to_pos_ratio"], dropna=False)
        .agg(
            auc_mean=("test_roc_auc", "mean"),
            auc_std=("test_roc_auc", "std"),
            mcc_mean=("test_mcc", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["arm", "split_type", "train_negative", "exp_fraction", "syn_to_pos_ratio"])
    )


def _write_report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    data_stats: dict,
    meta: dict,
) -> None:
    lines = [
        f"# Blend + Volume Experiment — {BLEND_PROTOCOL}",
        "",
        "- **blend**: train negs = f×experimental + (1−f)×synthetic, total = |train_neg_exp|",
        "- **volume**: pure synthetic, n_syn = r × |train_pos|, r ∈ volume_ratios",
        "- Test: experimental-only; splits: train/test only (no protocol val)",
        "",
        f"Meta: {json.dumps(meta)}",
        "",
    ]
    if not summary.empty:
        try:
            lines.append(summary.to_markdown(index=False))
        except Exception:
            lines.append(summary.to_string(index=False))
    lines += ["", f"Data: {json.dumps(data_stats)}", "", "Outputs:", "- blend_volume_metrics.csv", "- blend_volume_summary.csv"]
    path.write_text("\n".join(lines), encoding="utf-8")

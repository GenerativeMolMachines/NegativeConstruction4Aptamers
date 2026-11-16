"""ERE with AptaBench Table 13-style LightGBM baselines.

For every split/fold:
1. Build equal-size negative datasets for all methods using only the split train rows.
2. Train the same LightGBM classifier on positives + one negative dataset.
3. Evaluate on the same held-out experimental test set.
4. Cross-negative evaluation: train negative type × test negative type heatmap data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from aptabench_features import build_pair_features, feature_spec_for_split
from data import (
    DEFAULT_DATA,
    OFFICIAL_SPLIT_FILES,
    OFFICIAL_SPLITS_DIR,
    load_official_indexed_pairs,
    normalize_sequence,
)
from distribution_closeness import distribution_closeness
from lgbm_scoring import HARD_SCORE_METHODS, build_bootstrap_lgbm_scorer
from negative_dataset_controls import (
    NEGATIVE_METHODS,
    NegativeDataset,
    build_controlled_negative_dataset,
    diagnostics,
)


SPLIT_TYPES = ("random", "aptamer", "molecule")
LGBM_EVAL_PROTOCOL = "train_test_only"
SPLIT_SOURCE = "aptabench_official_json"


@dataclass
class SplitBundle:
    train_pos: pd.DataFrame
    val_pos: pd.DataFrame
    test_pos: pd.DataFrame
    train_neg_exp: pd.DataFrame
    val_neg_exp: pd.DataFrame
    test_neg_exp: pd.DataFrame
    stats: dict[str, Any]


def make_folds(
    df: pd.DataFrame,
    split_type: str,
    n_folds: int,
    seed: int,
    *,
    splits_dir: str | Path | None = None,
) -> list[SplitBundle]:
    """Official AptaBench 5-fold JSON: train_idx vs val_idx (held-out test).

    ``val_idx`` is the published held-out fold (~20%). ERE is train/test
    only, so we do not carve a further validation set. Empty val frames stay on
    ``SplitBundle`` for API compatibility. ``seed`` does not reshuffle splits.
    """
    del seed  # splits are fixed; callers still pass seed for generators
    if n_folds != 5:
        raise ValueError(f"Official AptaBench splits are 5-fold, got n_folds={n_folds}")
    if len(df) != 6289:
        raise ValueError(
            f"Official splits index the 6289-row AptaBench table, got len(df)={len(df)}. "
            "Use load_official_indexed_pairs()."
        )
    key = "random" if split_type == "stratified" else split_type
    if key not in OFFICIAL_SPLIT_FILES:
        raise ValueError(f"Unknown split type: {split_type}")
    path = Path(splits_dir or OFFICIAL_SPLITS_DIR) / OFFICIAL_SPLIT_FILES[key]
    folds_json = json.loads(path.read_text(encoding="utf-8"))
    if len(folds_json) != 5:
        raise ValueError(f"{path} must contain 5 folds, got {len(folds_json)}")

    empty_pos = df.iloc[0:0].copy()
    empty_neg = df.iloc[0:0].copy()
    bundles: list[SplitBundle] = []
    for spec in folds_json:
        fold = int(spec["fold"])
        train_idx = np.asarray(spec["train_idx"], dtype=int)
        test_idx = np.asarray(spec["val_idx"], dtype=int)
        if set(train_idx) & set(test_idx):
            raise ValueError(f"{split_type} fold {fold}: train/val overlap")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        dropped_emb = 0
        if split_type in ("random", "stratified", "aptamer"):
            marker = next(
                (c for c in df.columns if c.startswith(("clamp_", "smi_emb_", "chemberta_"))),
                None,
            )
            if marker is not None:
                before = len(train) + len(test)
                train = train.loc[train[marker].notna()]
                test = test.loc[test[marker].notna()]
                dropped_emb = int(before - len(train) - len(test))
        train_pos = train.loc[train["label"] == 1].reset_index(drop=True)
        test_pos = test.loc[test["label"] == 1].reset_index(drop=True)
        train_neg = train.loc[train["label"] == 0].reset_index(drop=True)
        test_neg = test.loc[test["label"] == 0].reset_index(drop=True)
        bundles.append(
            SplitBundle(
                train_pos=train_pos,
                val_pos=empty_pos.copy(),
                test_pos=test_pos,
                train_neg_exp=train_neg,
                val_neg_exp=empty_neg.copy(),
                test_neg_exp=test_neg,
                stats={
                    "split_type": split_type,
                    "split_source": SPLIT_SOURCE,
                    "fold": fold,
                    "protocol_val": False,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_pos": int(len(train_pos)),
                    "train_neg": int(len(train_neg)),
                    "val_pos": 0,
                    "val_neg": 0,
                    "test_pos": int(len(test_pos)),
                    "test_neg": int(len(test_neg)),
                    "dropped_missing_mol_emb": dropped_emb,
                },
            )
        )
    bundles.sort(key=lambda b: int(b.stats["fold"]))
    return bundles


def run_ere_lgbm(
    *,
    data_path: str | Path = DEFAULT_DATA,
    generated_path: str | Path,
    out_dir: str | Path,
    methods: list[str] | None = None,
    split_types: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    strict_features: bool = True,
    score_fn=None,
    hard_score_fn=None,
    lgbm_hard_score: bool = False,
    report_synthetic_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = methods or list(NEGATIVE_METHODS)
    split_types = split_types or list(SPLIT_TYPES)
    rng = np.random.default_rng(seed)
    df, data_stats = load_official_indexed_pairs(data_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_rows, cross_rows, diag_rows, distance_rows, done = _load_checkpoint(out)
    if done:
        print(f"Resuming with {len(done)} completed folds: {sorted(done)}", flush=True)

    for split_type in split_types:
        spec = feature_spec_for_split(split_type)
        print(f"\n=== ERE LightGBM: {split_type} | {spec.aptamer_encoder}+{spec.molecule_encoder} ===", flush=True)
        folds = make_folds(df, split_type, n_folds, seed)
        for fold, split in enumerate(folds):
            if (split_type, int(fold)) in done:
                print(f"\n--- {split_type}/fold{fold} --- SKIP (checkpoint)", flush=True)
                continue
            print(f"\n--- {split_type}/fold{fold} ---", flush=True)
            fold_hard_score_fn = hard_score_fn
            if lgbm_hard_score and fold_hard_score_fn is None:
                try:
                    bootstrap = build_bootstrap_lgbm_scorer(
                        split.train_pos,
                        split.train_neg_exp,
                        df,
                        split_type,
                        seed + fold,
                    )
                    fold_hard_score_fn = bootstrap.as_score_fn()
                    print(
                        "  Bootstrap LGBM hard-negative scorer ready "
                        f"(train pos={len(split.train_pos)}, neg={len(split.train_neg_exp)})",
                        flush=True,
                    )
                except ValueError as e:
                    print(f"  WARNING: bootstrap LGBM scorer unavailable: {e}", flush=True)

            print("  Building train negative sets...", flush=True)
            train_neg_sets = _build_neg_sets(
                methods,
                split.train_pos,
                split.train_neg_exp,
                df,
                rng,
                generated_path,
                score_fn=score_fn,
                hard_score_fn=fold_hard_score_fn,
            )
            print("  Building test negative sets...", flush=True)
            test_neg_sets = _build_neg_sets(
                methods,
                split.test_pos,
                split.test_neg_exp,
                df,
                rng,
                generated_path,
                score_fn=score_fn,
                hard_score_fn=fold_hard_score_fn,
            )

            for scope, neg_sets, pos_ref, neg_ref in (
                ("train", train_neg_sets, split.train_pos, split.train_neg_exp),
                ("test", test_neg_sets, split.test_pos, split.test_neg_exp),
            ):
                for nd in neg_sets.values():
                    d = diagnostics(nd.frame, pos_ref, neg_ref)
                    d.update(
                        {
                            "split_type": split_type,
                            "fold": fold,
                            "scope": scope,
                            "method": nd.method,
                            "target_n": nd.target_n,
                            "actual_n": len(nd.frame),
                            "matched_ligand_fraction": nd.matched_ligand_fraction,
                            "notes": nd.notes,
                        }
                    )
                    diag_rows.append(d)
                    if nd.method != "experimental" and not nd.frame.empty:
                        dist_score_fn = _score_fn_for_method(nd.method, score_fn, fold_hard_score_fn)
                        dist = distribution_closeness(
                            nd.frame,
                            neg_ref,
                            method=nd.method,
                            score_fn=dist_score_fn,
                            positives=pos_ref,
                            seed=seed + fold,
                        )
                        dist.update(
                            {
                                "split_type": split_type,
                                "fold": fold,
                                "scope": scope,
                                "reference": "experimental",
                            }
                        )
                        distance_rows.append(dist)

            exp_test_eval = _labeled_eval(split.test_pos, split.test_neg_exp)

            for train_method, train_neg in train_neg_sets.items():
                row_base = {
                    "lgbm_eval_protocol": LGBM_EVAL_PROTOCOL,
                    "split_type": split_type,
                    "fold": fold,
                    "train_negative": train_method,
                    "aptamer_encoder": spec.aptamer_encoder,
                    "molecule_encoder": spec.molecule_encoder,
                    "n_train_pos": len(split.train_pos),
                    "n_train_neg": len(train_neg.frame),
                }
                if train_neg.frame.empty:
                    metric_rows.append({**row_base, "skipped": True, "skip_reason": "empty_train_negatives"})
                    continue

                train_frame = _labeled_eval(split.train_pos, train_neg.frame)
                try:
                    model, threshold, scaler = _fit_model(train_frame, df, split_type, seed + fold)
                    test_metrics = _eval_model(model, exp_test_eval, df, split_type, threshold, scaler)
                except ValueError as e:
                    if strict_features:
                        metric_rows.append({**row_base, "skipped": True, "skip_reason": str(e)})
                        print(f"  SKIP {train_method}: {e}", flush=True)
                        continue
                    raise

                syn_metrics = {"roc_auc": None, "pr_auc": None, "mcc": None}
                if report_synthetic_test and train_method in test_neg_sets:
                    matched_neg = test_neg_sets[train_method]
                    if not matched_neg.frame.empty:
                        syn_eval = _labeled_eval(split.test_pos, matched_neg.frame)
                        syn_metrics = _eval_model(model, syn_eval, df, split_type, threshold, scaler)

                exp_auc = test_metrics["roc_auc"]
                syn_auc = syn_metrics["roc_auc"]
                auc_inflation = (
                    float(syn_auc) - float(exp_auc)
                    if syn_auc is not None and exp_auc is not None
                    else None
                )

                metric_rows.append(
                    {
                        **row_base,
                        "skipped": False,
                        "skip_reason": "",
                        "test_roc_auc": exp_auc,
                        "test_pr_auc": test_metrics["pr_auc"],
                        "test_mcc": test_metrics["mcc"],
                        "test_roc_auc_experimental": exp_auc,
                        "test_mcc_experimental": test_metrics["mcc"],
                        "test_roc_auc_synthetic_matched": syn_auc,
                        "test_mcc_synthetic_matched": syn_metrics["mcc"],
                        "test_negative_matched": train_method,
                        "auc_inflation_syn_minus_exp": auc_inflation,
                        "threshold": threshold,
                    }
                )
                print(
                    f"  train={train_method}: experimental test AUC={exp_auc:.4f} "
                    f"MCC={test_metrics['mcc']:.4f}",
                    flush=True,
                )
                if report_synthetic_test and syn_auc is not None:
                    print(
                        f"    matched synthetic test ({train_method}) "
                        f"AUC={syn_auc:.4f} MCC={syn_metrics['mcc']:.4f} "
                        f"inflation={auc_inflation:+.4f}",
                        flush=True,
                    )

                for test_method, test_neg in test_neg_sets.items():
                    if test_neg.frame.empty:
                        cross_rows.append(
                            {
                                "split_type": split_type,
                                "fold": fold,
                                "train_negative": train_method,
                                "test_negative": test_method,
                                "skipped": True,
                                "skip_reason": "empty_test_negatives",
                            }
                        )
                        continue
                    test_eval = _labeled_eval(split.test_pos, test_neg.frame)
                    cm = _eval_model(model, test_eval, df, split_type, threshold, scaler)
                    cross_rows.append(
                        {
                            "split_type": split_type,
                            "fold": fold,
                            "train_negative": train_method,
                            "test_negative": test_method,
                            "roc_auc": cm["roc_auc"],
                            "pr_auc": cm["pr_auc"],
                            "mcc": cm["mcc"],
                            "skipped": False,
                            "skip_reason": "",
                        }
                    )

            done.add((split_type, int(fold)))
            metrics = pd.DataFrame(metric_rows)
            cross = pd.DataFrame(cross_rows)
            diags = pd.DataFrame(diag_rows)
            distances = pd.DataFrame(distance_rows)
            _save_checkpoint(out, metrics, cross, diags, distances, data_stats)
            print(f"  Checkpoint saved after {split_type}/fold{fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    cross = pd.DataFrame(cross_rows)
    diags = pd.DataFrame(diag_rows)
    distances = pd.DataFrame(distance_rows)
    _save_checkpoint(out, metrics, cross, diags, distances, data_stats)
    return metrics, cross, diags


def _load_checkpoint(out: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], set[tuple[str, int]]]:
    metrics_path = out / "ere_metrics.csv"
    if not metrics_path.exists():
        return [], [], [], [], set()
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "split_type" not in metrics.columns or "fold" not in metrics.columns:
        return [], [], [], [], set()
    done = {(str(s), int(f)) for s, f in zip(metrics["split_type"], metrics["fold"])}

    def _rows(name: str) -> list[dict]:
        path = out / name
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        return frame.to_dict(orient="records") if not frame.empty else []

    return (
        metrics.to_dict(orient="records"),
        _rows("cross_negative_metrics.csv"),
        _rows("negative_distribution_diagnostics.csv"),
        _rows("distribution_closeness_metrics.csv"),
        done,
    )


def _save_checkpoint(
    out: Path,
    metrics: pd.DataFrame,
    cross: pd.DataFrame,
    diags: pd.DataFrame,
    distances: pd.DataFrame,
    data_stats: dict,
) -> None:
    metrics.to_csv(out / "ere_metrics.csv", index=False)
    cross.to_csv(out / "cross_negative_metrics.csv", index=False)
    diags.to_csv(out / "negative_distribution_diagnostics.csv", index=False)
    distances.to_csv(out / "distribution_closeness_metrics.csv", index=False)
    _write_inflation_table(metrics, out / "synthetic_inflation_metrics.csv")
    _write_report(metrics, cross, diags, distances, out / "REPORT.md", data_stats)


def _score_fn_for_method(method: str, score_fn, hard_score_fn):
    if method in HARD_SCORE_METHODS and hard_score_fn is not None:
        return hard_score_fn
    return score_fn


def _build_neg_sets(
    methods,
    positives,
    ref_neg,
    df,
    rng,
    generated_path,
    score_fn,
    hard_score_fn=None,
) -> dict[str, NegativeDataset]:
    target_n = len(ref_neg)
    out: dict[str, NegativeDataset] = {}
    for method in methods:
        print(f"    method={method} (n={target_n})...", flush=True)
        out[method] = build_controlled_negative_dataset(
            method,
            positives,
            ref_neg,
            df,
            rng,
            target_n=target_n,
            generated_path=str(generated_path),
            score_fn=_score_fn_for_method(method, score_fn, hard_score_fn),
        )
        print(f"    method={method}: got {len(out[method].frame)} pairs", flush=True)
    return out


def _write_inflation_table(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    ok = metrics.loc[~metrics["skipped"].fillna(False).astype(bool)].copy()
    if ok.empty or "test_roc_auc_synthetic_matched" not in ok.columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    agg = (
        ok.groupby(["split_type", "train_negative"])
        .agg(
            experimental_auc_mean=("test_roc_auc_experimental", "mean"),
            synthetic_matched_auc_mean=("test_roc_auc_synthetic_matched", "mean"),
            auc_inflation_mean=("auc_inflation_syn_minus_exp", "mean"),
            experimental_mcc_mean=("test_mcc_experimental", "mean"),
            synthetic_matched_mcc_mean=("test_mcc_synthetic_matched", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["split_type", "auc_inflation_mean"], ascending=[True, False])
    )
    agg.to_csv(path, index=False)


def _fit_model(train_frame, reference_df, split_type, seed):
    """Fit LightGBM with AptaBench-style StandardScaler (no Optuna)."""
    x_train, _ = build_pair_features(train_frame, reference_df, split_type)
    y_train = train_frame["label"].to_numpy(dtype=int)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    train_score = model.predict_proba(x_train)[:, 1]
    threshold = _best_mcc_threshold(y_train, train_score)
    return model, threshold, scaler


def _eval_model(model, frame, reference_df, split_type, threshold, scaler=None):
    x, _ = build_pair_features(frame, reference_df, split_type)
    if scaler is not None:
        x = scaler.transform(x)
    y = frame["label"].to_numpy(dtype=int)
    score = model.predict_proba(x)[:, 1]
    return _metrics(y, score, threshold)


def _metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(y)) == 2 else None,
    }


def _best_mcc_threshold(y, score) -> float:
    if len(set(y)) < 2:
        return 0.5
    thresholds = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best_t = 0.5
    best = -1.0
    for t in thresholds:
        m = matthews_corrcoef(y, (score >= t).astype(int))
        if m > best:
            best = m
            best_t = float(t)
    return best_t


def _labeled_eval(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    p = pos.loc[:, ["sequence", "canonical_smiles"] + (["type"] if "type" in pos.columns else [])].copy()
    p["sequence"] = p["sequence"].map(normalize_sequence)
    p["label"] = 1
    n = neg.loc[:, ["sequence", "canonical_smiles"] + (["type"] if "type" in neg.columns else [])].copy()
    n["sequence"] = n["sequence"].map(normalize_sequence)
    n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _take_val(indices: np.ndarray, rng: np.random.Generator, frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated: ERE is train/test only. Kept for external imports."""
    perm = rng.permutation(len(indices))
    n_val = max(1, int(frac * len(indices)))
    return indices[perm[:n_val]], indices[perm[n_val:]]


def _write_report(
    metrics: pd.DataFrame,
    cross: pd.DataFrame,
    diags: pd.DataFrame,
    distances: pd.DataFrame,
    path: Path,
    data_stats: dict,
) -> None:
    lines = [
        "# ERE LightGBM — Negative Construction Benchmark",
        "",
        "Narrative: experimental negatives are scarce, so synthetic negative construction changes the negative distribution, decision boundary, and measured generalization.",
        "Experimental AptaBench negatives are treated as the reference distribution; learned target-conditioned generation is evaluated against literature-based synthetic baselines.",
        "",
        "Model/features follow AptaBench Table 13 best LightGBM configurations:",
        "- random/stratified: Kmer4 + CLAMP",
        "- molecule-disjoint: OneHot + MorganFP",
        "- aptamer-disjoint: Kmer4 + ChemBERTa",
        "- StandardScaler fit on train fold (AptaBench screening.py); fixed LightGBM (no Optuna)",
        "",
        "Splits: official AptaBench JSON (stratified / disjoint_aptamer / disjoint_molecule); `val_idx` is the held-out test fold.",
        "Negative datasets are matched to the original split negative count and ligand distribution.",
        "",
    ]
    if not metrics.empty:
        ok = metrics.loc[~metrics["skipped"].fillna(False).astype(bool)]
        if not ok.empty:
            agg = (
                ok.groupby(["split_type", "train_negative"])
                .agg(
                    auc_mean=("test_roc_auc", "mean"),
                    auc_std=("test_roc_auc", "std"),
                    mcc_mean=("test_mcc", "mean"),
                    folds=("fold", "count"),
                )
                .reset_index()
                .sort_values(["split_type", "auc_mean"], ascending=[True, False])
            )
            try:
                lines.append(agg.to_markdown(index=False))
            except Exception:
                lines.append(agg.to_string(index=False))
    if not metrics.empty and "auc_inflation_syn_minus_exp" in metrics.columns:
        ok = metrics.loc[~metrics["skipped"].fillna(False).astype(bool)]
        if not ok.empty:
            infl = (
                ok.groupby(["split_type", "train_negative"])
                .agg(
                    experimental_auc=("test_roc_auc_experimental", "mean"),
                    synthetic_matched_auc=("test_roc_auc_synthetic_matched", "mean"),
                    auc_inflation=("auc_inflation_syn_minus_exp", "mean"),
                    folds=("fold", "count"),
                )
                .reset_index()
                .sort_values(["split_type", "auc_inflation"], ascending=[True, False])
            )
            lines += [
                "",
                "## Synthetic Test Inflation (matched train/test negative method)",
                "",
                "Positive = held-out test positives. Negative = synthetic set built with the same method as train.",
                "auc_inflation = synthetic_matched_auc - experimental_auc (higher means optimistic bias).",
                "",
            ]
            try:
                lines.append(infl.to_markdown(index=False))
            except Exception:
                lines.append(infl.to_string(index=False))
    if not distances.empty:
        train_dist = distances.loc[distances["scope"] == "train"].copy()
        if not train_dist.empty:
            lines += ["", "## Distribution Closeness To Experimental Negatives", ""]
            cols = [
                "length_wasserstein",
                "gc_wasserstein",
                "kmer_mmd_rbf",
                "sequence_c2st_auc",
                "ligand_jaccard",
                "molecule_c2st_auc",
            ]
            have = [c for c in cols if c in train_dist.columns]
            dist_agg = train_dist.groupby(["split_type", "method"])[have].mean().reset_index()
            try:
                lines.append(dist_agg.to_markdown(index=False))
            except Exception:
                lines.append(dist_agg.to_string(index=False))
    lines += [
        "",
        f"Data: {json.dumps(data_stats)}",
        "",
        "Outputs:",
        "- ere_metrics.csv",
        "- cross_negative_metrics.csv",
        "- negative_distribution_diagnostics.csv",
        "- distribution_closeness_metrics.csv",
        "- synthetic_inflation_metrics.csv",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

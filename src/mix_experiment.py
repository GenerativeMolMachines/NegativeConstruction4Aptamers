"""Mix experiment: f = fraction of experimental negatives kept as supervision seeds.

For each synthetic method and each f in {0, 0.10, …, 1.0}:

1. **Supervision** (not in classifier train mix): ``sample(f × |train_neg_exp|)`` experimental
   negatives — seeds for contrastive / bootstrap LGBM (hard) / GRU.
2. **Generate** ``|train_neg_exp|`` negatives with the method (trained/scored using only
   supervision at this f — never another method's negatives as a proxy).
3. **Classifier train set**: ``train_pos`` + **100% synthetic** (step 2). No raw
   experimental negatives mixed into train at f < 1.

At f=0 (no experimental seeds):
- ``random_pairing`` (and other static decoys) run — pure cross-pairing / decoys.
- ``hard_negative`` / ``learned_real_hard_negative`` / ``learned_generated`` are
  **skipped** — those need experimental supervision to train their own scorer/generator;
  using pairing negatives to train GRU/contrastive would mix methods.

At f>0: contrastive + GRU (+ bootstrap LGBM for hard) are trained from scratch on
train_pos + only the f% experimental negatives, then used to sample synthetic train negs.

Classifier LGBM is always fit on train_pos + the synthetic set for that (split, f, method).
No protocol-level validation holdout (train/test only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aptabench_features import feature_spec_for_split
from data import DEFAULT_DATA, load_aptabench_indexed_pairs, normalize_sequence
from fraction_retrain import (
    FRACTION_LOCAL_LEARNED_METHODS,
    build_fraction_local_bundle,
    build_fraction_local_negative_dataset,
    bootstrap_negatives_for_scorer,
    fraction_local_supervision_negatives,
)
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

DEFAULT_MIX_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
SYNTHETIC_METHODS = tuple(m for m in NEGATIVE_METHODS if m != "experimental")
MIX_PROTOCOL = "fraction_retrain_v2"

# Learned / hard methods need experimental supervision seeds at this f.
# Do NOT train them on random-pairing proxies — that mixes methods.
METHODS_REQUIRE_SUPERVISION_NEG = frozenset({
    "hard_negative",
    "learned_generated",
    "learned_real_hard_negative",
})


def supervision_negatives_at_fraction(
    train_neg_exp: pd.DataFrame,
    exp_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    target_n = len(train_neg_exp)
    return fraction_local_supervision_negatives(
        train_neg_exp, target_n=target_n, exp_fraction=exp_fraction, rng=rng
    )


def ligand_reference_for_sources(
    train_pos: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """When f=0, match ligand counts using train positives (no exp neg seeds)."""
    cols = [c for c in ("sequence", "canonical_smiles", "type", "label") if c in train_pos.columns]
    base = train_pos.loc[:, cols].copy()
    if len(base) >= target_n:
        idx = rng.choice(len(base), size=target_n, replace=False)
    else:
        idx = rng.choice(len(base), size=target_n, replace=True)
    out = base.iloc[idx].reset_index(drop=True)
    out["label"] = 0
    return out


def build_fold_hard_score_fn(
    train_pos: pd.DataFrame,
    supervision_neg: pd.DataFrame,
    full_df: pd.DataFrame,
    split_type: str,
    seed: int,
    *,
    target_n: int,
    lgbm_hard_score: bool,
) -> Any:
    if not lgbm_hard_score:
        return None
    rng = np.random.default_rng(seed)
    bootstrap_neg = bootstrap_negatives_for_scorer(
        train_pos,
        supervision_neg,
        full_df,
        target_n=target_n,
        rng=rng,
    )
    if bootstrap_neg.empty:
        return None
    try:
        return build_bootstrap_lgbm_scorer(
            train_pos, bootstrap_neg, full_df, split_type, seed
        ).as_score_fn()
    except ValueError:
        return None


def train_negatives_from_pool(
    synthetic: pd.DataFrame,
    *,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Size method output to classifier train negative count (100% synthetic)."""
    return _sample_n(synthetic, target_n, rng)


def run_mix_experiment(
    *,
    data_path: str | Path = DEFAULT_DATA,
    generated_path: str | Path,
    out_dir: str | Path,
    methods: list[str] | None = None,
    split_types: list[str] | None = None,
    mix_fractions: list[float] | tuple[float, ...] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    lgbm_hard_score: bool = True,
    fraction_local_learned: bool = True,
    contrastive_epochs: int = 8,
    decoder_epochs: int = 10,
    retrain_decoder: bool = True,
) -> pd.DataFrame:
    """Seed-multiply mix sweep (see module docstring).

    ``fraction_local_learned`` is kept for CLI compatibility; learned methods always
    use fold-local bundles. Static methods use the same supervision_neg at each f.
    """
    del fraction_local_learned  # always seed-multiply protocol

    methods = [m for m in (methods or list(SYNTHETIC_METHODS)) if m != "experimental"]
    if not methods:
        raise ValueError("Need at least one synthetic method")
    split_types = split_types or list(SPLIT_TYPES)
    mix_fractions = tuple(mix_fractions or DEFAULT_MIX_FRACTIONS)
    rng = np.random.default_rng(seed)
    df, data_stats = load_aptabench_indexed_pairs(data_path)
    raw_df = (pd.read_parquet(data_path) if Path(data_path).suffix.lower() == ".parquet"
              else pd.read_csv(data_path, low_memory=False))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_rows, done = _load_checkpoint(out)
    if done:
        print(f"Resuming ({MIX_PROTOCOL}) with {len(done)} completed cells...", flush=True)

    for split_type in split_types:
        spec = feature_spec_for_split(split_type)
        print(
            f"\n=== MIX ({MIX_PROTOCOL}): {split_type} | "
            f"{spec.aptamer_encoder}+{spec.molecule_encoder} ===",
            flush=True,
        )
        folds = make_folds(df, split_type, n_folds, seed)
        for fold, split in enumerate(folds):
            print(f"\n--- {split_type}/fold{fold} ---", flush=True)
            target_n = len(split.train_neg_exp)
            exp_test_eval = _labeled_eval(split.test_pos, split.test_neg_exp)

            for frac in mix_fractions:
                if all(_checkpoint_key(split_type, fold, method, frac) in done for method in methods):
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([
                    seed, SPLIT_TYPES.index(split_type), fold, round(float(frac) * 1000),
                ]))
                supervision_neg = supervision_negatives_at_fraction(
                    split.train_neg_exp, float(frac), rng
                )
                n_supervision = int(len(supervision_neg))
                hard_score_fn = build_fold_hard_score_fn(
                    split.train_pos,
                    supervision_neg,
                    df,
                    split_type,
                    seed + fold + int(round(float(frac) * 1000)),
                    target_n=target_n,
                    lgbm_hard_score=lgbm_hard_score,
                )
                ref_neg = (
                    supervision_neg
                    if not supervision_neg.empty
                    else ligand_reference_for_sources(split.train_pos, target_n, rng)
                )

                bundle = None
                learned_pending = [m for m in methods if m in FRACTION_LOCAL_LEARNED_METHODS]
                static_pending = [m for m in methods if m not in FRACTION_LOCAL_LEARNED_METHODS]

                # Never build learned bundles at f=0 / empty supervision — no cross-method proxy.
                if learned_pending and not supervision_neg.empty:
                    need_gen = "learned_generated" in learned_pending
                    ck0 = _checkpoint_key(split_type, fold, learned_pending[0], frac)
                    if ck0 not in done or any(
                        _checkpoint_key(split_type, fold, m, frac) not in done for m in learned_pending
                    ):
                        bundle = build_fraction_local_bundle(
                            split_type=split_type,
                            fold=fold,
                            exp_fraction=float(frac),
                            train_pos=split.train_pos,
                            train_neg_exp=split.train_neg_exp,
                            val_pos=split.val_pos,
                            val_neg_exp=split.val_neg_exp,
                            full_df=df,
                            raw_df=raw_df,
                            out_dir=out,
                            rng=rng,
                            seed=seed + fold * 100 + int(round(float(frac) * 100)),
                            retrain_decoder=retrain_decoder,
                            contrastive_epochs=contrastive_epochs,
                            decoder_epochs=decoder_epochs,
                            need_generation=need_gen,
                        )

                all_at_frac = learned_pending + static_pending
                for method in all_at_frac:
                    ck_key = _checkpoint_key(split_type, fold, method, frac)
                    if ck_key in done:
                        continue

                    row_base = {
                        "mix_protocol": MIX_PROTOCOL,
                        "split_type": split_type,
                        "fold": fold,
                        "train_negative": method,
                        "exp_fraction": float(frac),
                        "n_train_pos": len(split.train_pos),
                        "n_supervision_neg": n_supervision,
                        "aptamer_encoder": spec.aptamer_encoder,
                        "molecule_encoder": spec.molecule_encoder,
                    }

                    if method in METHODS_REQUIRE_SUPERVISION_NEG and supervision_neg.empty:
                        metric_rows.append(
                            {
                                **row_base,
                                "skipped": True,
                                "skip_reason": "requires_supervision_neg_at_f0",
                                "n_train_neg": 0,
                                "n_train_neg_experimental": 0,
                                "n_train_neg_synthetic": 0,
                            }
                        )
                        done.add(ck_key)
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, mix_fractions)
                        print(f"  SKIP {method}@{frac:.0%}: no supervision negatives", flush=True)
                        continue

                    try:
                        if method in FRACTION_LOCAL_LEARNED_METHODS:
                            assert bundle is not None
                            neg_ds = build_fraction_local_negative_dataset(
                                method,
                                split.train_pos,
                                split.train_neg_exp,
                                df,
                                rng,
                                bundle,
                            )
                            notes = bundle.notes
                        else:
                            score_fn = (
                                hard_score_fn
                                if method in HARD_SCORE_METHODS and hard_score_fn is not None
                                else None
                            )
                            neg_ds = build_controlled_negative_dataset(
                                method,
                                split.train_pos,
                                ref_neg,
                                df,
                                rng,
                                target_n=target_n,
                                generated_path=str(generated_path),
                                score_fn=score_fn,
                                candidate_positives=split.train_pos,
                            )
                            notes = neg_ds.notes or ""

                        train_neg = train_negatives_from_pool(
                            neg_ds.frame, target_n=target_n, rng=rng
                        )
                    except Exception as e:
                        metric_rows.append(
                            {
                                **row_base,
                                "skipped": True,
                                "skip_reason": str(e),
                                "n_train_neg": 0,
                                "n_train_neg_experimental": 0,
                                "n_train_neg_synthetic": 0,
                            }
                        )
                        done.add(ck_key)
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, mix_fractions)
                        print(f"  SKIP {method}@{frac:.0%}: {e}", flush=True)
                        continue

                    row_base.update(
                        {
                            "n_train_neg": len(train_neg),
                            "n_train_neg_experimental": 0,
                            "n_train_neg_synthetic": len(train_neg),
                            "fraction_local_notes": notes,
                        }
                    )

                    if train_neg.empty:
                        metric_rows.append(
                            {**row_base, "skipped": True, "skip_reason": "empty_train_negatives"}
                        )
                        done.add(ck_key)
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, mix_fractions)
                        continue

                    train_frame = _labeled_eval(split.train_pos, train_neg)
                    try:
                        model, threshold, scaler = _fit_model(
                            train_frame, df, split_type, seed + fold
                        )
                        test_metrics = _eval_model(
                            model, exp_test_eval, df, split_type, threshold, scaler
                        )
                    except ValueError as e:
                        metric_rows.append({**row_base, "skipped": True, "skip_reason": str(e)})
                        done.add(ck_key)
                        _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, mix_fractions)
                        print(f"  SKIP {method}@{frac:.0%}: {e}", flush=True)
                        continue

                    metric_rows.append(
                        {
                            **row_base,
                            "skipped": False,
                            "skip_reason": "",
                            "test_roc_auc": test_metrics["roc_auc"],
                            "test_pr_auc": test_metrics["pr_auc"],
                            "test_mcc": test_metrics["mcc"],
                            "threshold": threshold,
                        }
                    )
                    print(
                        f"  train={method} f={frac:.0%} "
                        f"(supervision={n_supervision}, train_syn={len(train_neg)}): "
                        f"test AUC={test_metrics['roc_auc']:.4f} "
                        f"MCC={test_metrics['mcc']:.4f}",
                        flush=True,
                    )
                    done.add(ck_key)
                    _save_checkpoint(out, pd.DataFrame(metric_rows), data_stats, mix_fractions)

    metrics = pd.DataFrame(metric_rows)
    _save_checkpoint(out, metrics, data_stats, mix_fractions)
    return metrics


def _checkpoint_key(
    split_type: str,
    fold: int,
    method: str,
    exp_fraction: float,
) -> tuple[Any, ...]:
    return (MIX_PROTOCOL, split_type, int(fold), str(method), float(exp_fraction))


def _sample_n(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame is None or frame.empty:
        return pd.DataFrame(columns=["sequence", "canonical_smiles", "label", "type"])
    cols = ["sequence", "canonical_smiles"] + (["type"] if "type" in frame.columns else [])
    base = frame.loc[:, cols].copy()
    base["sequence"] = base["sequence"].map(normalize_sequence)
    if "label" not in base.columns:
        base["label"] = 0
    if len(base) >= n:
        idx = rng.choice(len(base), size=n, replace=False)
    else:
        idx = rng.choice(len(base), size=n, replace=True)
    return base.iloc[idx].reset_index(drop=True)


def _load_checkpoint(out: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    path = out / "mix_metrics.csv"
    if not path.exists():
        return [], set()
    metrics = pd.read_csv(path)
    if metrics.empty or "split_type" not in metrics.columns:
        return [], set()
    protocol = MIX_PROTOCOL
    if "mix_protocol" in metrics.columns:
        metrics = metrics.loc[metrics["mix_protocol"].astype(str) == protocol].copy()
    ok = metrics.loc[~metrics.get("skipped", False).fillna(False).astype(bool)]
    if ok.empty or "train_negative" not in ok.columns or "exp_fraction" not in ok.columns:
        return metrics.to_dict(orient="records"), set()
    done = {
        (protocol, str(s), int(f), str(m), float(x))
        for s, f, m, x in zip(ok["split_type"], ok["fold"], ok["train_negative"], ok["exp_fraction"])
    }
    return metrics.to_dict(orient="records"), done


def _save_checkpoint(
    out: Path,
    metrics: pd.DataFrame,
    data_stats: dict,
    mix_fractions: tuple[float, ...] | list[float],
) -> None:
    metrics.to_csv(out / "mix_metrics.csv", index=False)
    summary = _summarize(metrics)
    summary.to_csv(out / "mix_summary.csv", index=False)
    _write_report(metrics, summary, out / "REPORT.md", data_stats, mix_fractions)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ok = metrics.loc[~metrics.get("skipped", False).fillna(False).astype(bool)].copy()
    if ok.empty:
        return pd.DataFrame()
    if "mix_protocol" in ok.columns:
        ok = ok.loc[ok["mix_protocol"] == MIX_PROTOCOL]
    return (
        ok.groupby(["split_type", "train_negative", "exp_fraction"])
        .agg(
            auc_mean=("test_roc_auc", "mean"),
            auc_std=("test_roc_auc", "std"),
            mcc_mean=("test_mcc", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["split_type", "train_negative", "exp_fraction"])
    )


def _write_report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    data_stats: dict,
    mix_fractions: tuple[float, ...] | list[float],
) -> None:
    lines = [
        f"# Mix Experiment — {MIX_PROTOCOL}",
        "",
        "**f** = fraction of experimental train negatives kept as **supervision seeds**",
        "(contrastive / bootstrap LGBM / GRU trained only on that slice; decoder from scratch).",
        "Classifier train negatives are **100% synthetic**",
        f"from the method, count = |train_neg_exp|. Raw experimental negatives are **not**",
        "mixed into train at f < 1.",
        "",
        "At f=0: only static methods (e.g. `random_pairing`). "
        "`hard_negative` / `learned_generated` / `learned_real_hard_negative` skipped "
        "(no experimental seeds; no cross-method proxy). "
        "Test: experimental-only (train/test protocol, no val).",
        "",
        f"Mix fractions: {list(mix_fractions)}",
        "",
    ]
    if not summary.empty:
        try:
            lines.append(summary.to_markdown(index=False))
        except Exception:
            lines.append(summary.to_string(index=False))
    lines += ["", f"Data: {json.dumps(data_stats)}", "", "Outputs:", "- mix_metrics.csv", "- mix_summary.csv"]
    path.write_text("\n".join(lines), encoding="utf-8")


# Legacy alias — old blend mix removed; kept so imports do not break.
def mix_train_negatives(*_args, **_kwargs):
    raise NotImplementedError(
        "mix_train_negatives (exp/syn blend) removed; use seed_multiply protocol in run_mix_experiment"
    )

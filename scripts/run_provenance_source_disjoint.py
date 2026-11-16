#!/usr/bin/env python3
"""Leave-one-source-out ERE on AptaBench if a reliable provenance id exists.

Does not build a study-disjoint split from origin/buffer. Those columns fail
the coverage / dominance audit in provenance_splits.py.
Train negatives: experimental, random_sequence, random_pairing (no learned
constructors; those pools are official-fold scoped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DEFAULT_DATA, load_official_indexed_pairs  # noqa: E402
from ere_lgbm import _eval_model, _fit_model, _labeled_eval  # noqa: E402
from negative_dataset_controls import build_controlled_negative_dataset  # noqa: E402
from provenance_splits import audit_provenance, reliable_provenance_columns, source_disjoint_holdouts  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "provenance_source_disjoint_v1"
METHODS = ("experimental", "random_sequence", "random_pairing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df, _ = load_official_indexed_pairs(args.data)
    audit = audit_provenance(df)
    audit.to_csv(args.out / "provenance_audit.csv", index=False)
    reliable = reliable_provenance_columns(df)
    print(audit.to_string(index=False), flush=True)
    print(f"reliable columns: {reliable or 'none'}", flush=True)
    if "source" not in reliable:
        (args.out / "run_meta.json").write_text(
            json.dumps(
                {
                    "ran_splits": False,
                    "reason": "no reliable source/study id",
                    "reliable_columns": reliable,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("No reliable provenance id; source-disjoint training skipped.", flush=True)
        return 0
    if args.audit_only:
        return 0

    rows = []
    for hold in source_disjoint_holdouts(df, column="source"):
        source = hold["held_out_source"]
        print(
            f"\n=== hold out {source}: train {hold['n_train_pos']}+/{hold['n_train_neg']}- "
            f"test {hold['n_test_pos']}+/{hold['n_test_neg']}- "
            f"mol_overlap={hold['frac_test_molecules_in_train']:.3f} ===",
            flush=True,
        )
        test = _labeled_eval(hold["test_pos"], hold["test_neg_exp"])
        train_pool = pd.concat([hold["train_pos"], hold["train_neg_exp"]], ignore_index=True)
        rng = np.random.default_rng(args.seed)
        for method in METHODS:
            neg = build_controlled_negative_dataset(
                method,
                positives=hold["train_pos"],
                reference_negatives=hold["train_neg_exp"],
                full_df=train_pool,
                rng=rng,
                target_n=len(hold["train_neg_exp"]),
                generated_path="",
            )
            if neg.frame.empty:
                rows.append(
                    {
                        "held_out_source": source,
                        "train_negative": method,
                        "skipped": True,
                        "skip_reason": neg.notes,
                    }
                )
                continue
            train = _labeled_eval(hold["train_pos"], neg.frame)
            model, threshold, scaler = _fit_model(train, df, "molecule", args.seed)
            metrics = _eval_model(model, test, df, "molecule", threshold, scaler)
            row = {
                "held_out_source": source,
                "train_negative": method,
                "skipped": False,
                "skip_reason": "",
                "n_train_pos": hold["n_train_pos"],
                "n_train_neg": len(neg.frame),
                "n_test_pos": hold["n_test_pos"],
                "n_test_neg": hold["n_test_neg"],
                "frac_test_molecules_in_train": hold["frac_test_molecules_in_train"],
                "frac_test_aptamers_in_train": hold["frac_test_aptamers_in_train"],
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
                "test_mcc": metrics["mcc"],
            }
            rows.append(row)
            print(
                f"  {method}: AUC={row['test_roc_auc']:.4f} AUPR={row['test_pr_auc']:.4f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.out / "source_disjoint_metrics.csv", index=False)
    (args.out / "run_meta.json").write_text(
        json.dumps({"ran_splits": True, "reliable_columns": reliable, "methods": list(METHODS)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out / 'source_disjoint_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

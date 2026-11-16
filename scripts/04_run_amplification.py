#!/usr/bin/env python3
"""Fixed 10% experimental-subset amplification (molecule-disjoint, LightGBM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amplification_experiment import (
    run_amplification_experiment,
    summarize_amplification,
    summarize_paired_subset_deltas,
)

DEFAULT_DATA = ROOT / "data" / "aptabench" / "embeddings" / "aptabench_with_embeddings_v3_clamp.parquet"
DEFAULT_CONSTRUCTED = ROOT / "data" / "constructed_negatives" / "ere_official_scaled_v1"
DEFAULT_OUT = ROOT / "results" / "amplification_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--constructed-root", default=str(DEFAULT_CONSTRUCTED))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--split-type", default="molecule")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp-fraction", type=float, default=0.1)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    metrics = run_amplification_experiment(
        data_path=Path(args.data),
        out=out,
        split_type=args.split_type,
        n_folds=args.n_folds,
        seed=args.seed,
        exp_fraction=args.exp_fraction,
        n_repeats=args.n_repeats,
        resume=not args.no_resume,
        constructed_negatives_root=Path(args.constructed_root),
    )
    summary = summarize_amplification(metrics)
    summary.to_csv(out / "amplification_summary.csv", index=False)
    summarize_paired_subset_deltas(metrics).to_csv(out / "amplification_paired_deltas.csv", index=False)

    print("\n=== Amplification summary (ROC-AUC mean ± std) ===")
    for _, row in summary.iterrows():
        print(
            f"  {row['mode']:24s}  "
            f"{row['mean_roc_auc']:.4f} ± {row['std_roc_auc']:.4f}  "
            f"(n={int(row['n_folds'])})"
        )
    print(f"\nWrote {out / 'amplification_metrics.csv'}")
    print(f"Wrote {out / 'amplification_summary.csv'}")


if __name__ == "__main__":
    main()

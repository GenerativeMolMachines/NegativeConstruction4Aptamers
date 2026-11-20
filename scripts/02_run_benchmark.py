#!/usr/bin/env python3
"""ERE LightGBM on AptaBench folds (StandardScaler, fixed LGBM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_dataset_controls import NEGATIVE_METHODS
from ere_lgbm import SPLIT_TYPES, run_ere_lgbm

DEFAULT_DATA = ROOT / "data" / "aptabench" / "embeddings" / "aptabench_embeddings.parquet"
DEFAULT_GENERATION = ROOT / "data" / "aptabench" / "generation" / "train_fold_generation_summary.csv"
DEFAULT_OUT = ROOT / "results" / "ere"
PAPER_METHODS = (
    "random_sequence", "composition_shuffle", "random_pairing",
    "structure_filtered", "hard_negative", "learned_generated", "experimental",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--generation", default=str(DEFAULT_GENERATION))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--methods", default="paper")
    p.add_argument("--split-types", default="all")
    p.add_argument("--lgbm-hard-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = (
        list(PAPER_METHODS)
        if args.methods == "paper"
        else list(NEGATIVE_METHODS)
        if args.methods == "all"
        else [m.strip() for m in args.methods.split(",") if m.strip()]
    )
    split_types = (
        list(SPLIT_TYPES)
        if args.split_types == "all"
        else [s.strip() for s in args.split_types.split(",") if s.strip()]
    )
    run_ere_lgbm(
        data_path=args.data,
        generated_path=args.generation,
        out_dir=args.out,
        methods=methods,
        split_types=split_types,
        n_folds=args.n_folds,
        seed=args.seed,
        lgbm_hard_score=args.lgbm_hard_score,
        report_synthetic_test=True,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Blend + volume sweep on official AptaBench folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blend_volume_experiment import (
    BLEND_PROTOCOL,
    DEFAULT_BLEND_FRACTIONS,
    DEFAULT_METHODS,
    DEFAULT_VOLUME_RATIOS,
    run_blend_volume_experiment,
)

DEFAULT_DATA = ROOT / "data" / "aptabench" / "embeddings" / "aptabench_with_embeddings_v3_clamp.parquet"
DEFAULT_GENERATION = ROOT / "data" / "aptabench" / "generation" / "train_fold_generation_summary.csv"
DEFAULT_OUT = ROOT / "results" / "ere_blend_volume_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--generation", default=str(DEFAULT_GENERATION))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--split-types", default="random,aptamer,molecule")
    p.add_argument("--blend-fractions", default=",".join(str(x) for x in DEFAULT_BLEND_FRACTIONS))
    p.add_argument("--volume-ratios", default=",".join(str(x) for x in DEFAULT_VOLUME_RATIOS))
    p.add_argument("--no-lgbm-hard-score", action="store_true")
    p.add_argument("--blend-only", action="store_true")
    p.add_argument("--volume-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    split_types = [s.strip() for s in args.split_types.split(",") if s.strip()]
    blend_fractions = [float(x) for x in args.blend_fractions.split(",") if x.strip()]
    volume_ratios = [float(x) for x in args.volume_ratios.split(",") if x.strip()]
    print(f"Protocol: {BLEND_PROTOCOL}", flush=True)
    run_blend_volume_experiment(
        data_path=args.data,
        generated_path=args.generation,
        out_dir=args.out,
        methods=methods,
        split_types=split_types,
        blend_fractions=blend_fractions,
        volume_ratios=volume_ratios,
        n_folds=args.n_folds,
        seed=args.seed,
        lgbm_hard_score=not args.no_lgbm_hard_score,
        run_blend=not args.volume_only,
        run_volume=not args.blend_only,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

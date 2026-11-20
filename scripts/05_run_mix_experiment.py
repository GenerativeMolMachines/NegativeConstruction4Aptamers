#!/usr/bin/env python3
"""Fraction-retrain mix experiment (AptaBench folds, StandardScaler)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mix_experiment import DEFAULT_MIX_FRACTIONS, MIX_PROTOCOL, SYNTHETIC_METHODS, run_mix_experiment
from ere_lgbm import SPLIT_TYPES

DEFAULT_DATA = ROOT / "data" / "aptabench" / "embeddings" / "aptabench_embeddings.parquet"
DEFAULT_GENERATION = ROOT / "data" / "aptabench" / "generation" / "train_fold_generation_summary.csv"
DEFAULT_OUT = ROOT / "results" / "fraction_retrain"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--generation", default=str(DEFAULT_GENERATION))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--methods", default="random_pairing,learned_generated,hard_negative")
    p.add_argument("--split-types", default="aptamer,molecule")
    p.add_argument(
        "--mix-fractions",
        default=",".join(str(x) for x in DEFAULT_MIX_FRACTIONS),
    )
    p.add_argument("--contrastive-epochs", type=int, default=5)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--no-lgbm-hard-score", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    split_types = [s.strip() for s in args.split_types.split(",") if s.strip()]
    mix_fractions = [float(x) for x in args.mix_fractions.split(",") if x.strip()]
    print(f"Protocol: {MIX_PROTOCOL}", flush=True)
    run_mix_experiment(
        data_path=args.data,
        generated_path=args.generation,
        out_dir=args.out,
        methods=methods,
        split_types=split_types,
        mix_fractions=mix_fractions,
        n_folds=args.n_folds,
        seed=args.seed,
        lgbm_hard_score=not args.no_lgbm_hard_score,
        contrastive_epochs=args.contrastive_epochs,
        decoder_epochs=args.decoder_epochs,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

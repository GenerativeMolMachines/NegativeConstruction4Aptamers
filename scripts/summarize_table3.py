"""Validate the new Table 3 inputs and write a common fold CSV and summary."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["split_type", "fold", "train_negative", "exp_fraction"]
SPLITS = ("random", "aptamer", "molecule")
BLEND = ("random_sequence", "composition_shuffle", "random_pairing", "structure_filtered")
RETRAIN = ("hard_negative", "learned_generated")
FRACTIONS = (0.10, 0.25, 0.50, 0.75)


def validate(path, methods, fractions, protocol):
    data = pd.read_csv(path)
    keys = {(s, fold, m, f) for s in SPLITS for fold in range(5)
            for m in methods for f in fractions}
    actual = set(data[KEY].itertuples(index=False, name=None))
    if actual != keys or data.duplicated(KEY).any():
        raise ValueError(f"{protocol}: expected {len(keys)} distinct rows, found {len(data)}")
    if data.skipped.fillna(False).any() or not data.test_roc_auc.between(0, 1).all():
        raise ValueError(f"{protocol}: skipped or invalid scores")
    return data.assign(table3_protocol=protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=ROOT / "results/table3_blend/blend_volume_metrics.csv")
    parser.add_argument("--retrain", type=Path, default=ROOT / "results/table3_fraction_retrain/mix_metrics.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/table3")
    args = parser.parse_args()
    blend = validate(args.blend, BLEND, (0.0, *FRACTIONS, 1.0), "Blend")
    retrain = validate(args.retrain, RETRAIN, FRACTIONS, "Fraction-retrain")
    if retrain.n_train_neg_experimental.ne(0).any():
        raise ValueError("Fraction-retrain classifier received measured negatives")
    ere = pd.read_csv(ROOT / "analyses/results/ere_metrics.csv")
    measured = ere.loc[ere.train_negative.eq("experimental")].assign(exp_fraction=1.0)
    baseline = ere.loc[ere.train_negative.isin(BLEND)].assign(exp_fraction=0.0)
    check = blend.loc[blend.exp_fraction.eq(0)].merge(baseline[KEY + ["test_roc_auc"]], on=KEY,
                                                       suffixes=("", "_ere"), validate="one_to_one")
    if len(check) != len(baseline) or not np.allclose(check.test_roc_auc, check.test_roc_auc_ere, atol=1e-12, rtol=0):
        raise ValueError("Blend f=0 scores differ from the canonical ERE")
    # Print the experimental reference once per split in the paper.
    combined = pd.concat([blend.loc[blend.exp_fraction.lt(1)], retrain,
                          measured.assign(table3_protocol="Reference")], ignore_index=True)
    summary = combined.groupby(["split_type", "table3_protocol", "train_negative", "exp_fraction"],
                               as_index=False).test_roc_auc.agg(["mean", "std", "count"])
    if not summary["count"].eq(5).all():
        raise ValueError("Each Table 3 cell must contain five folds")
    args.out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out / "table3_fold_metrics.csv", index=False)
    summary.to_csv(args.out / "table3_summary.csv", index=False)


if __name__ == "__main__":
    main()

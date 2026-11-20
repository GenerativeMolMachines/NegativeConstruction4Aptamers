"""Combine the complete new architecture comparison with its LightGBM reference."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"composition_shuffle", "random_pairing", "learned_generated", "experimental"}
SPLITS = {"random", "molecule"}
ARCHITECTURES = {"mlp", "two_tower_mlp", "cnn_dual_encoder"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/architecture_comparison")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    files = sorted(args.run.glob(f"*_e{args.epochs}.csv"))
    if len(files) != 120:
        raise SystemExit(f"Expected 120 scored neural runs, found {len(files)}")
    neural = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["split_type", "fold", "method", "architecture"]
    expected = {(s, f, m, a) for s in SPLITS for f in range(5) for m in METHODS for a in ARCHITECTURES}
    if set(neural[keys].itertuples(index=False, name=None)) != expected or neural.duplicated(keys).any():
        raise SystemExit("Incomplete or duplicate architecture comparison")
    reference = pd.read_csv(ROOT / "analyses/results/ere_metrics.csv")
    reference = reference.loc[reference.split_type.isin(SPLITS) & reference.train_negative.isin(METHODS)]
    reference = reference.rename(columns={"train_negative": "method", "test_roc_auc": "roc_auc", "test_pr_auc": "ap"})
    reference = reference.assign(architecture="lightgbm", seed=42)
    if len(reference) != 40 or reference.duplicated(keys).any():
        raise SystemExit("Expected 40 LightGBM reference scores")
    selected = keys + ["seed", "roc_auc", "ap"]
    combined = pd.concat([neural[selected + ["epochs", "n_train", "n_test", "pool_sha256"]],
                          reference[selected]], ignore_index=True).sort_values(keys)
    if combined[["roc_auc", "ap"]].isna().any().any():
        raise SystemExit("Missing ROC-AUC or AP")
    summary = combined.groupby(["split_type", "method", "architecture"], as_index=False).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        ap_mean=("ap", "mean"), ap_sd=("ap", "std"), folds=("fold", "nunique"),
    )
    combined.to_csv(args.run / "architecture_fold_metrics.csv", index=False)
    summary.to_csv(args.run / "architecture_summary.csv", index=False)
    print(f"Saved {len(combined)} fold results and {len(summary)} summary rows.", flush=True)


if __name__ == "__main__":
    main()

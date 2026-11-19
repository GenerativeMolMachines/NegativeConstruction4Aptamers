#!/usr/bin/env python3
"""Compare the two paper ERE tables against complete fold-level metrics."""
import argparse
import re
from pathlib import Path

import pandas as pd

METHODS = {
    "Random sequence": "random_sequence",
    "AptaBLE shuffle": "composition_shuffle",
    "Random pairing": "random_pairing",
    "Structure-filtered": "structure_filtered",
    "Hard negatives": "hard_negative",
    "Profile-GRU": "learned_generated",
    "Experimental": "experimental",
}
SPLITS = ["random", "aptamer", "molecule"]
APPENDIX_METHODS = ["composition_shuffle", "random_pairing", "learned_generated", "experimental"]


def table(source, label):
    start = source.index(r"\label{" + label + "}")
    return source[start:source.index(r"\end{tabular}", start)]


def scores(row):
    return re.findall(r"(0\.\d{3})\s*\$?\s*\\pm\s*\$?\s*(0\.\d{3})", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    source = args.manuscript.read_text(encoding="utf-8")
    frame = pd.read_csv(args.metrics)
    frame = frame.loc[frame.train_negative.isin(METHODS.values())]
    if (len(frame) != 105 or frame.test_roc_auc.isna().any()
            or frame.duplicated(["split_type", "fold", "train_negative"]).any()
            or set(frame.fold) != set(range(5))):
        raise SystemExit("Expected all seven methods on five folds of each of the three splits")
    stats = frame.groupby(["split_type", "train_negative"]).test_roc_auc.agg(["mean", "std"])

    def expected(split, method):
        row = stats.loc[split, method]
        return f"{row['mean']:.3f}", f"{row['std']:.3f}"

    checks = []
    main_table = table(source, "tab:experimental_test")
    for label, method in METHODS.items():
        row = re.search(rf"(?m)^{re.escape(label)}\s*\n(.*?)\\\\", main_table, re.S)
        actual = scores(row.group(1)) if row else []
        checks.append((f"Table 1 / {label}", actual, [expected(split, method) for split in SPLITS]))
    appendix = table(source, "tab:robust-multimodel")
    rows = [line for line in appendix.splitlines() if "& LightGBM" in line]
    if len(rows) != 2:
        raise SystemExit("Expected two LightGBM rows in architecture comparison")
    for split, row in zip(("random", "molecule"), rows):
        checks.append((f"Architecture comparison / {split}", scores(row),
                       [expected(split, method) for method in APPENDIX_METHODS]))
    mismatches = [(label, actual, reference) for label, actual, reference in checks if actual != reference]
    for label, actual, reference in mismatches:
        print(f"{label}: manuscript {actual} | CSV {reference}")
    if mismatches:
        raise SystemExit(1)
    print("Verified Table 1 and both LightGBM rows of the architecture comparison.")


if __name__ == "__main__":
    main()

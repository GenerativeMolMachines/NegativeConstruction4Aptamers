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
    parser.add_argument("--amplification", type=Path, help="Directory with fold-level amplification metrics")
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
    if args.amplification:
        methods = {
            "exp_10_only": r"10\% experimental only",
            "exp_10_oversample": r"10\% experimental, oversampled",
            "exp_10_plus_rp": r"10\% experimental + random pairing",
            "exp_10_plus_generated": r"10\% experimental + Profile-GRU",
        }
        raw = pd.read_csv(args.amplification / "amplification_metrics.csv")
        if len(raw) != 200 or raw.duplicated(["fold", "repeat", "mode"]).any():
            raise SystemExit("Expected 200 distinct amplification scores")
        draws = raw.pivot(index=["fold", "repeat"], columns="mode", values="test_roc_auc")
        if draws.shape != (50, 4) or set(draws.columns) != set(methods) or draws.isna().any().any():
            raise SystemExit("Expected four scored methods in each of 50 matched draws")
        rendered = table(source, "tab:robust-amplification-repeats")
        for method, name in methods.items():
            values = draws[method]
            delta = values - draws["exp_10_only"]
            mean = "$%.3f \\pm %.3f$" % (values.mean(), values.std())
            change = "$0$" if method == "exp_10_only" else "$%.4f \\pm %.4f$" % (delta.mean(), delta.std())
            wins = "reference" if method == "exp_10_only" else f"{int(delta.gt(0).sum())}/50"
            row = next((line for line in rendered.splitlines() if line.startswith(name + " &")), "")
            checks.append((name, [part.strip() for part in row.removesuffix(r"\\").split("&")[1:]],
                           [mean, change, wins]))
    mismatches = [(label, actual, reference) for label, actual, reference in checks if actual != reference]
    for label, actual, reference in mismatches:
        print(f"{label}: manuscript {actual} | CSV {reference}")
    if mismatches:
        raise SystemExit(1)
    print(f"Verified {len(checks)} manuscript rows against fold-level CSVs.")


if __name__ == "__main__":
    main()

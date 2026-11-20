#!/usr/bin/env python3
"""Compare manuscript ERE, architecture, and audit tables against their CSVs."""
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
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "analyses/results")
    parser.add_argument("--table3", type=Path, help="Validated Table 3 summary CSV")
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
    architectures = {
        "LightGBM": "lightgbm",
        "CNN dual encoder": "cnn_dual_encoder",
        "Two-tower MLP": "two_tower_mlp",
        "MLP": "mlp",
    }
    model_scores = pd.read_csv(args.results / "architecture_fold_metrics.csv")
    keys = ["split_type", "architecture", "method", "fold"]
    expected_index = pd.MultiIndex.from_product(
        [["random", "molecule"], architectures.values(), APPENDIX_METHODS, range(5)], names=keys
    )
    indexed = model_scores.set_index(keys).roc_auc
    if indexed.index.has_duplicates or len(indexed) != len(expected_index):
        raise SystemExit("Expected 160 distinct architecture results")
    indexed = indexed.reindex(expected_index)
    if indexed.isna().any():
        raise SystemExit("Missing scores in the architecture comparison")
    model_stats = indexed.groupby(level=keys[:3]).agg(["mean", "std"])
    for split, section in zip(("random", "molecule"), appendix.split(r"\multicolumn")[1:]):
        for name, architecture in architectures.items():
            row = next((line for line in section.splitlines() if line.strip().startswith(f"& {name} &")), "")
            target = model_stats.loc[(split, architecture)].reindex(APPENDIX_METHODS)
            checks.append((f"Architecture / {split} / {name}", scores(row),
                           [(f"{mean:.3f}", f"{std:.3f}") for mean, std in target.itertuples(index=False)]))
    split_labels = {"random": "Stratified", "aptamer": "Aptamer-disjoint", "molecule": "Molecule-disjoint"}
    audit = pd.read_csv(args.results / "leakage_audit.csv")
    cols = ["heldout_sequences", "heldout_molecules", "test_pairs", "known_positives"]
    methods = list(METHODS.values())[:-1]
    audit_keys = ["split_type", "fold", "method"]
    audit_index = pd.MultiIndex.from_product([SPLITS, range(5), methods], names=audit_keys)
    audit = audit.set_index(audit_keys)
    if audit.index.has_duplicates or audit.reindex(audit_index)[cols].isna().any().any():
        raise SystemExit("Missing or duplicate constructor audit rows")
    audit_table = table(source, "tab:supp-leakage-audit")
    for split, label in split_labels.items():
        group = audit.loc[split].loc[(slice(None), methods), :]
        restricted = group.loc[(slice(None), ["random_pairing", "hard_negative"]), :]
        counts = [*group[cols].max(), restricted.outside_train_positive_pool.max()]
        match = re.search(rf"(?m)^{label} & (.*?)\s*\\\\$", audit_table)
        checks.append((f"Leakage / {split}", match.group(1).split(" & ") if match else [],
                       [str(int(count)) for count in counts]))
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
    source_metrics = pd.read_csv(args.results / "fold_source_metrics.csv")
    source_metrics = source_metrics.dropna(subset=["test_roc_auc"])
    source_methods = ["experimental", "random_sequence", "random_pairing"]
    source_draws = source_metrics.pivot(index=["split_type", "held_out_source", "fold"],
                                        columns="train_negative", values="test_roc_auc")
    if len(source_draws) != 28 or set(source_draws.columns) != set(source_methods) or source_draws.isna().any().any():
        raise SystemExit("Expected 28 source folds with three scored methods each")
    source_table = table(source, "tab:source-fold")
    for (split, held_out), group in source_draws.groupby(level=[0, 1]):
        label = split_labels["random" if split == "stratified" else split]
        prefix = f"{label} & {held_out} & "
        row = next((line for line in source_table.splitlines() if line.startswith(prefix)), "")
        expected_cells = [str(len(group))]
        for method in source_methods:
            values = group[method]
            mean = f"{values.mean():.3f}"
            expected_cells.append(f"${mean}\\pm{values.std():.3f}$" if len(group) > 1 else f"${mean}$")
        actual = row.removeprefix(prefix).removesuffix(r"\\").strip().split(" & ")
        checks.append((f"Source holdout / {label} / {held_out}", actual, expected_cells))
    blend_path = args.results / "table3_blend_fold_metrics.csv"
    if blend_path.exists() and not args.table3:
        blend = pd.read_csv(blend_path)
        keys = ["split_type", "train_negative", "exp_fraction"]
        blend_methods = dict(list(METHODS.items())[:4])
        fractions = (0, 0.1, 0.25, 0.5, 0.75, 1)
        expected_keys = pd.MultiIndex.from_product(
            [SPLITS, blend_methods.values(), fractions, range(5)], names=keys + ["fold"]
        )
        values = blend.set_index(keys + ["fold"]).test_roc_auc
        if values.index.has_duplicates or len(values) != len(expected_keys) or values.reindex(expected_keys).isna().any():
            raise SystemExit("Expected all 360 Blend fold scores")
        blend_stats = values.groupby(level=keys).agg(["mean", "std"])
        split = None
        verified = set()
        for line in table(source, "tab:experimental_fraction").splitlines():
            cells = [cell.strip() for cell in line.split("&")]
            if len(cells) != 9 or cells[2] != "Blend":
                continue
            if cells[0]:
                split = next(key for key, label in split_labels.items() if label == cells[0])
            method = blend_methods[cells[1]]
            if (split, method) in verified:
                raise SystemExit("Duplicate Table 3 Blend row")
            verified.add((split, method))
            reference = [blend_stats.loc[(split, method, f)] for f in fractions[:-1]]
            checks.append((f"Table 3 Blend / {split} / {method}", scores(line),
                           [(f"{r['mean']:.3f}", f"{r['std']:.3f}") for r in reference]))
            original = stats.loc[split, method]
            if blend_stats.loc[(split, method, 0)].ne(original).any():
                raise SystemExit(f"Table 3 Blend f=0 differs from ERE for {split}/{method}")
        if len(verified) != 12:
            raise SystemExit("Missing Table 3 Blend rows")
    if args.table3:
        mix = pd.read_csv(args.table3)
        if len(mix) != 87 or not mix["count"].eq(5).all():
            raise SystemExit("Expected 87 Table 3 cells, each based on five folds")
        stats = mix.set_index(["split_type", "table3_protocol", "train_negative", "exp_fraction"])
        composition = table(source, "tab:experimental_fraction")
        split = None
        verified = set()
        for line in composition.splitlines():
            cells = line.removesuffix(r"\\").split("&")
            if len(cells) != 9 or cells[2].strip() not in {"Blend", "Fraction-retrain", "Reference"}:
                continue
            label = cells[0].replace(r"\rowcolor{gray!15}", "").strip()
            if label:
                split = next(key for key, value in split_labels.items() if value == label)
            protocol = cells[2].strip()
            method = "experimental" if protocol == "Reference" else METHODS[cells[1].strip()]
            for frac, cell in zip((0, 0.1, 0.25, 0.5, 0.75, 1), cells[3:]):
                key = (split, protocol, method, frac)
                present = key in stats.index
                actual = scores(cell)
                if present:
                    if key in verified:
                        raise SystemExit(f"Duplicate Table 3 cell {key}")
                    verified.add(key)
                    ref = stats.loc[key]
                    checks.append((f"Table 3 / {key}", actual, [(f"{ref['mean']:.3f}", f"{ref['std']:.3f}")]))
                elif cell.strip() != "--":
                    checks.append((f"Unexpected Table 3 cell / {key}", cell.strip(), "--"))
        if verified != set(stats.index):
            raise SystemExit("Table 3 is missing cells from the summary")
    mismatches = [(label, actual, reference) for label, actual, reference in checks if actual != reference]
    for label, actual, reference in mismatches:
        print(f"{label}: manuscript {actual} | CSV {reference}")
    if mismatches:
        raise SystemExit(1)
    print(f"Verified {len(checks)} manuscript rows against fold-level CSVs.")


if __name__ == "__main__":
    main()

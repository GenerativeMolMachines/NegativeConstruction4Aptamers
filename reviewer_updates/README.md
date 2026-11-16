# Analyses added after review

These CSVs accompany the revised paper. The original AptaBench ERE protocol and pre-exported fold data remain in the repository root.

## Repeated 10% experimental-negative subsets

- `results/amplification_metrics.csv`: one row per fold, redraw, and training mode (200 rows).
- `results/amplification_summary.csv`: mean and dispersion across the repeated runs.
- `results/amplification_paired_deltas.csv`: differences between augmentation modes and the same 10% draw.

The 10% versus full experimental pool comparison is descriptive because the training sets are not a matched pair of independent redraws. The ten redraws within a fold share the same held-out test data.

## Source-held-out sensitivity analysis

- `results/fold_source_metrics.csv`: metrics by official fold, held-out source, and negative-construction method.
- `results/fold_source_audit.csv`: overlap and eligibility checks.

The scored train pool excludes the test slice's source and all aptamers and molecules occurring in that slice. Only experimental, random-sequence, and random-pairing negatives were evaluated in this analysis. This is a database-source sensitivity analysis within the official folds, not an assay-disjoint evaluation. Some source-fold combinations are skipped by the minimum class-size rule, so rows must not be interpreted as independent panels.

These files are supplementary evidence for the revised claims; the original benchmark table uses the official ERE folds.

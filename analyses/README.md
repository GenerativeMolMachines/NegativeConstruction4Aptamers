# Analysis results

These CSVs summarize repeated-subset experiments and source-held-out sensitivity analyses on AptaBench.

To reproduce the analyses from the repository root:

```bash
python scripts/04_run_amplification.py --n-repeats 10 --out results/amplification_repeats10_v1
python scripts/run_provenance_aptabench_folds.py --out results/provenance_aptabench_folds_v1
```

The provenance runner uses the published AptaBench folds; its source filtering logic is in `src/provenance_splits.py`. The optional `scripts/run_provenance_source_disjoint.py` performs a global source holdout and should not be confused with the AptaBench fold sensitivity analysis.

## Repeated 10% experimental-negative subsets

- `results/amplification_metrics.csv`: one row per fold, redraw, and training mode (200 rows).
- `results/amplification_summary.csv`: mean and dispersion across the repeated runs.
- `results/amplification_paired_deltas.csv`: differences between augmentation modes and the same 10% draw.

The 10% versus full experimental pool comparison is descriptive because the training sets are not a matched pair of independent redraws. The ten redraws within a fold share the same held-out test data.

## Source-held-out sensitivity analysis

- `results/fold_source_metrics.csv`: metrics by aptabench fold, held-out source, and negative-construction method.
- `results/fold_source_audit.csv`: overlap and eligibility checks.

The scored train pool excludes the test slice's source and all aptamers and molecules occurring in that slice. Only experimental, random-sequence, and random-pairing negatives were evaluated in this analysis. This is a database-source sensitivity analysis within the AptaBench folds, not an assay-disjoint evaluation. Some source-fold combinations are skipped by the minimum class-size rule, so rows must not be interpreted as independent panels.

The main benchmark table uses the main ERE folds; source-held-out results are a separate sensitivity analysis.

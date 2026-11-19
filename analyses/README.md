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

The main benchmark table uses the AptaBench folds. The source sensitivity experiment additionally removes all shared aptamers and molecules.

## Main ERE protocol revision

The archived inputs reproduce the historical figures. The previous Random pairing and Hard negative implementation used the complete sequence table as its candidate pool. An audit of the released pools found test-exclusive sequences in both methods. The revised runner passes the positive sequences of each training fold as the candidate pool. The complete measured pair table serves as the exclusion set. Results that reused the older pools have the same provenance limitation.

The published Profile-GRU candidate CSV contains no split or fold identifiers. The pair audit checks the emitted pairs. Generator training provenance requires separate fold-level records. The architecture comparison requires the original fold-level outputs or a new matched rerun.

The completed revised LightGBM run is recorded in [ere_metrics.csv](results/ere_metrics.csv), with 105 rows across 15 folds. The [paired audit](results/leakage_audit.csv) checks 90 synthetic training sets. The [released fold pools](../data/constructed_negatives/ere_fold_local_v2/by_fold/) are the exact pairs emitted by this run. Cross-negative scores and matched-test inflation are in [cross_negative_metrics.csv](results/cross_negative_metrics.csv) and [synthetic_inflation_metrics.csv](results/synthetic_inflation_metrics.csv).

The existing `amplification_*.csv` files record the historical repeated-subset experiment. Its RP and Profile-GRU inputs came from the earlier export. Running the amplification command now uses the revised pools.

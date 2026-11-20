# Analysis results

These CSVs summarize repeated-subset experiments and source-held-out sensitivity analyses on AptaBench.

To reproduce the analyses from the repository root:

```bash
python scripts/04_run_amplification.py --n-repeats 10 --out results/amplification_v2
python scripts/run_provenance_aptabench_folds.py --out results/provenance_aptabench_folds_v1
```

The provenance runner uses the published AptaBench folds; its source filtering logic is in `src/provenance_splits.py`. The optional `scripts/run_provenance_source_disjoint.py` performs a global source holdout and should not be confused with the AptaBench fold sensitivity analysis.

## Repeated 10% experimental-negative subsets

- `results/amplification_metrics.csv`: one row per fold, redraw, and training mode (200 rows).
- `results/amplification_summary.csv`: mean and dispersion across the repeated runs.
- `results/amplification_paired_deltas.csv`: differences between augmentation modes and the same 10% draw.
- `results/amplification_fixed_exp_subsets.csv`: selected experimental negatives for each draw.
- `results/amplification_run_meta.json`: split, seed, redraw count, and input pools.

The 10% versus full experimental pool comparison is descriptive because the training sets are not a matched pair of independent redraws. The ten redraws within a fold share the same held-out test data.

## Source-held-out sensitivity analysis

- `results/fold_source_metrics.csv`: Table 6 metrics by AptaBench fold, held-out source, and negative-construction method. Use `split_type=random` for Stratified, as in the main ERE CSVs.
- `results/fold_source_audit.csv`: overlap and eligibility checks.

The scored train pool excludes the test slice's source and all aptamers and molecules occurring in that slice. Only experimental, random-sequence, and random-pairing negatives were evaluated in this analysis. This is a database-source sensitivity analysis within the AptaBench folds, not an assay-disjoint evaluation. Some source-fold combinations are skipped by the minimum class-size rule, so rows must not be interpreted as independent panels.

The main benchmark table uses the AptaBench folds. The source sensitivity experiment additionally removes all shared aptamers and molecules.

## Main ERE protocol revision

The archived inputs reproduce the historical figures. The previous Random pairing and Hard negative implementation used the complete sequence table as its candidate pool. An audit of the released pools found test-exclusive sequences in both methods. The revised runner passes the positive sequences of each training fold as the candidate pool. The complete measured pair table serves as the exclusion set. Results that reused the older pools have the same provenance limitation.

The published Profile-GRU candidate CSV contains no split or fold identifiers. The pair audit checks the emitted pairs. Generator training provenance for the main ERE requires separate fold-level records. The revised architecture comparison below uses the published candidate pools.

For the historical stratified comparison, the original main ERE and the later LightGBM run on exported negatives used different inputs. The revised LightGBM results supersede both as the benchmark reference. The updated Table 3 reports Blend and Fraction-retrain separately in the [cell summary](results/table3_summary.csv) and [fold scores](results/table3_fold_metrics.csv). The separate volume sweep retains its historical inputs.

The stratified LightGBM Profile-GRU result in the current paper's Table 4 is **0.669 ± 0.013 ROC-AUC** across five folds, computed from [ere_metrics.csv](results/ere_metrics.csv) at `split_type=random` and `train_negative=learned_generated`. The earlier **0.672 ± 0.017** appears only in the [archived Table 4 transcription](results/archived_table4_reported_summary.csv). The available archived runs do not reproduce that earlier value.

To recalculate Table 3 from the repository root:

```bash
python scripts/06_run_blend_volume.py --out results/table3_blend_v2 --blend-only --constructed-root data/constructed_negatives/ere_v2 --methods random_sequence,composition_shuffle,random_pairing,structure_filtered
python scripts/05_run_mix_experiment.py --out results/table3_fraction_retrain_v2 --methods hard_negative,learned_generated --split-types random,aptamer,molecule --mix-fractions 0.10,0.25,0.50,0.75
python scripts/summarize_table3.py
```

The summarizer validates both complete runs and checks that the 0% Blend values reproduce the per-fold ERE results before writing the fold CSV and summary.

The completed revised LightGBM run is recorded in [ere_metrics.csv](results/ere_metrics.csv), with 105 rows across 15 folds. The [paired audit](results/leakage_audit.csv) checks 90 synthetic training sets. The [released fold pools](../data/constructed_negatives/ere_v2/by_fold/) are the exact pairs emitted by this run. Cross-negative scores and matched-test inflation are in [cross_negative_metrics.csv](results/cross_negative_metrics.csv) and [synthetic_inflation_metrics.csv](results/synthetic_inflation_metrics.csv).

The `amplification_*.csv` files record the repeated-subset experiment on the revised ERE pools. Files with the `archived_amplification_` prefix preserve the earlier experiment. Profile-GRU candidates were precomputed. The 10% limit governs the measured negatives supplied directly to LightGBM.

## Revised architecture comparison

```bash
python scripts/run_architecture_comparison.py
python scripts/summarize_architecture_comparison.py
```

The new protocol fits MLP, two-tower MLP and a CNN dual encoder for 20 epochs with seed 42. For each negative method, all architectures use the released revised training pool and the same five experimental test folds per split. The runner saves a CSV after every model fit and records the SHA-256 of its negative pool. The summary adds the 40 corresponding LightGBM scores and reports ROC-AUC and average precision.

The revised Table 4 uses the [160 fold scores](results/architecture_fold_metrics.csv) and [32 summary rows](results/architecture_summary.csv).

The [historical Table 4 transcription](results/archived_table4_reported_summary.csv) preserves the reported means and standard deviations. The files [archived_hybrid_mlp_metrics.csv](results/archived_hybrid_mlp_metrics.csv) and [archived_dual_encoder_metrics.csv](results/archived_dual_encoder_metrics.csv) preserve separate exploratory runs. These files do not establish the provenance of the original Table 4 numbers.

See [Upstream assets](UPSTREAM_ASSETS.md) for checkpoint terms and attribution.

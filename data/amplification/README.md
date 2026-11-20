# Archived fixed-subset amplification

Committed summary for **fixed 10% experimental-subset amplification**
(`amplification_exp_v1`, molecule-disjoint, 5 folds, seed 42).

| `mode` | Paper label |
|--------|-------------|
| `exp_10_only` | 10% experimental only |
| `exp_10_oversample` | 10% experimental, oversampled to full N |
| `exp_10_plus_rp` | 10% exp + random pairing to N |
| `exp_10_plus_generated` | 10% exp + learned generated to N |

The archived run used [earlier synthetic pools](../constructed_negatives/ere_aptabench_scaled_v1/). For the repeated-subset results in the paper, use the [current fold metrics](../../analyses/results/amplification_metrics.csv) and [current negative pools](../constructed_negatives/ere/).

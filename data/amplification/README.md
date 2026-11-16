# Amplification (robustness appendix)

Committed summary for **fixed 10% experimental-subset amplification**
(`amplification_exp_v1`, molecule-disjoint, 5 folds, seed 42).

| `mode` | Paper label |
|--------|-------------|
| `exp_10_only` | 10% experimental only |
| `exp_10_oversample` | 10% experimental, oversampled to full N |
| `exp_10_plus_rp` | 10% exp + random pairing to N |
| `exp_10_plus_generated` | 10% exp + learned generated to N |

Regenerate: `python scripts/04_run_amplification.py` → `results/amplification_v1/`.

Synthetic pools: `data/constructed_negatives/ere_official_scaled_v1/`.

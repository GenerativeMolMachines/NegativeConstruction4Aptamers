# Constructed negatives (read-only release)

Pre-exported fold-local synthetic negative CSVs from the main ERE run
(`ere_aptabench_scaled_v1`). Used by:

- **Amplification experiment** (`scripts/04_run_amplification.py`) — pools for
  `exp_10_plus_rp` and `exp_10_plus_generated`
- Optional inspection / external re-analysis

Main ERE pipeline (`02_run_benchmark.py`) builds negatives at runtime and does
**not** read this tree.

```text
constructed_negatives/ere_aptabench_scaled_v1/
  by_fold/{random,aptamer,molecule}/fold*/{train,test}/*.csv
  volume_pools/...
  READONLY_RELEASE.txt
```

Amplification needs only **molecule / train / random_pairing + learned_generated**
for folds 0–4; the full tree is committed for completeness.

# Constructed negatives

- [Current paper pools](ere_fold_local_v2/by_fold/) are the pools used for the revised ERE, architecture comparison, and repeated 10% amplification analysis.
- [Archived pools](ere_aptabench_scaled_v1/) preserve earlier experiment inputs. Use them only when reproducing historical analyses. They can yield different ROC-AUC values from the current paper.

The main ERE runner constructs its own negatives. Downstream analyses that load exported pools use the current release by default.

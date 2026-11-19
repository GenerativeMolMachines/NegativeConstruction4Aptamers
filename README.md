<p align="center">
  <img src="assets/ere_protocol.png" alt="AptaBench pairs, six negative constructors, and experimental-reference evaluation" width="960">
</p>

# Evaluating Negative Construction for Aptamer–Small Molecule Interaction Learning

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-AIDaR%20poster-204a70)](#about)
[![Dataset](https://img.shields.io/badge/Dataset-AptaBench-269e91)](https://github.com/GenerativeMolMachines/AptaBench)
[![Code license](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

## About

How much does the way we construct **synthetic non-binders** change an aptamer–small molecule interaction model? This repository compares six constructors on AptaBench. Experimental positives, held-out experimental test pairs, AptaBench folds, and the main LightGBM pipeline stay fixed; only the training negatives change.

| Data | Evaluation | Reference |
|---|---|---|
| AptaBench aptamer–ligand pairs | Five folds each: stratified, aptamer-disjoint, molecule-disjoint | Experimentally characterized binders and non-binders |
| Six synthetic constructors | Experimental-reference evaluation (ERE) on fixed held-out pairs | Experimental training negatives |

**Main finding:** matched synthetic tests can substantially overstate performance against experimental non-binders. In the revised LightGBM run, Random pairing has the highest experimental-test ROC-AUC among synthetic methods on all three splits. See the [fold metrics](analyses/results/ere_metrics.csv) and [leakage audit](analyses/results/leakage_audit.csv).

## How the benchmark works

1. Use one released AptaBench train/test fold.
2. Keep its positives and experimental test pairs fixed.
3. Construct training negatives using only permitted fold-local information.
4. Fit the same classifier and evaluate on the same experimental test pairs.

| Synthetic constructor | Core idea | Experimental-negative information used by the constructor |
|---|---|---|
| Random sequence | Generate decoy sequences | Ligand counts for matching |
| Composition shuffle | Shuffle aptamers while preserving composition | Ligand counts for matching |
| Structure-filtered | Filter candidate sequences by structure | Ligand counts for matching |
| Random pairing | Rewire observed aptamers and ligands | Ligand counts; known pairs excluded |
| Hard negative | Select difficult candidates with a fold-local scorer | Experimental negatives train the scorer |
| Learned generated | Generate and rank sequences with a learned model | Experimental-negative profiles and contrastive supervision |

The **experimental** row is a reference training condition, not a synthetic constructor. Except for that reference and explicit blend controls, the downstream classifier trains on emitted synthetic negatives.

## Install and run

```bash
git clone https://github.com/OpiumProger/Neurips_AIDAR.git
cd Neurips_AIDAR
python -m venv .venv
# Activate .venv for your shell, then:
pip install -r requirements.txt
python scripts/02_run_benchmark.py --lgbm-hard-score
python scripts/audit_ere_pairs.py results/ere_fold_local_v2/constructed --out results/ere_fold_local_v2/leakage_audit.csv
```

The main ERE run uses the released embeddings and generation input. Some optional generation and structure-filtering experiments need pretrained models, a GPU, or ViennaRNA; see [contrastive/README.md](contrastive/README.md).

## Reproduce the analyses

| Analysis | Entry point | Output |
|---|---|---|
| Main ERE on AptaBench folds | `scripts/02_run_benchmark.py` | `results/ere_fold_local_v2/` |
| Cross-negative comparison | `scripts/03_cross_negative_eval.py` | `results/ere_fold_local_v2/cross_negative_heatmap.csv` |
| Repeated 10% experimental subsets | `scripts/04_run_amplification.py --n-repeats 10 --out results/amplification_fold_local_v2` | `results/amplification_fold_local_v2/` |
| Source-held-out sensitivity | `scripts/run_provenance_aptabench_folds.py` | `results/provenance_aptabench_folds_v1/` |
| Experimental/synthetic mixes | `scripts/05_run_mix_experiment.py` | `results/ere_fraction_retrain_v1/mix_metrics.csv` and `mix_summary.csv` |
| Blend and volume controls | `scripts/06_run_blend_volume.py` | `results/ere_blend_volume_v1/blend_volume_metrics.csv` and `blend_volume_summary.csv` |

Summary CSVs and interpretation are in [analyses/](analyses/). The commands write the outputs listed above when run from the repository root. Released analysis CSVs are available under `analyses/results/`.

Run `python scripts/07_make_figures.py` to regenerate the current ERE plots from the included CSVs. Use `--ere-inputs results/ere_fold_local_v2` to plot a new run. The volume and mix plots reproduce separate historical experiments.

## Repository map

| Path | Contents |
|---|---|
| [data/aptabench/](data/aptabench/) | AptaBench table, split indices, embeddings, and generation input |
| [Revised fold pools](data/constructed_negatives/ere_fold_local_v2/by_fold/) | Negative sets used by the revised LightGBM run |
| [src/](src/) | Constructors, data loading, ERE, and analysis code |
| [contrastive/](contrastive/) | Contrastive model and GRU generation code |
| [scripts/](scripts/) | Reproduction commands |
| [analyses/](analyses/) | Repeated-subset and source sensitivity analyses |

## Interpretation

- AptaBench aptamer- and molecule-disjoint folds can share assays and database sources across train and test. The source sensitivity analysis holds out a database source within each AptaBench fold.
- In the historical 10% control, ten independent redraws per molecule-disjoint fold give mean ROC-AUC **0.826 ± 0.070** for 10%-only experimental negatives. Redraws share test folds. The historical synthetic augmentation arms used the earlier pools.
- The revised comparison holds the LightGBM classifier fixed. Other architectures require evaluation on the revised pools.

## License and data

Code in this release is under [MIT](LICENSE). The [AptaBench dataset](https://huggingface.co/datasets/GenerativeMolMachines/AptaBench_dataset) by Mariia Eremeyeva and Nikita Serov is released under CC BY 4.0. The release contains 1,610 published aptamer sequences. Normalizing U to T yields 1,605. The released ligand embedding join covers 6,224 of 6,289 rows. See the [AptaBench repository](https://github.com/GenerativeMolMachines/AptaBench) for the citation.

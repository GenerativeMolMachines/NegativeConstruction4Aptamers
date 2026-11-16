<p align="center">
  <img src="assets/ere_protocol.png" alt="AptaBench pairs, six negative constructors, and experimental-reference evaluation" width="960">
</p>

# Evaluating Negative Construction for Aptamer–Small Molecule Interaction Learning

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-AIDaR%20poster-204a70)](#about)
[![Dataset](https://img.shields.io/badge/Dataset-AptaBench-269e91)](https://github.com/GenerativeMolMachines/AptaBench)
[![Code license](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

## About

How much does the way we construct **synthetic non-binders** change an aptamer–small molecule interaction model? This repository compares six constructors on AptaBench. Experimental positives, held-out experimental test pairs, official folds, and the main LightGBM pipeline stay fixed; only the training negatives change.

| Data | Evaluation | Reference |
|---|---|---|
| AptaBench aptamer–ligand pairs | Five folds each: stratified, aptamer-disjoint, molecule-disjoint | Experimentally characterized binders and non-binders |
| Six synthetic constructors | Experimental-reference evaluation (ERE) on fixed held-out pairs | Experimental training negatives |

**Main finding:** matched synthetic tests can substantially overstate performance against experimental non-binders. The ranking of synthetic methods can change with the prediction architecture and test-source provenance.

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
python scripts/02_run_benchmark.py --out results/ere_official_scaled_v1 --lgbm-hard-score
```

The main ERE run uses the released embeddings and generation input. Some optional generation and structure-filtering experiments need pretrained models, a GPU, or ViennaRNA; see [contrastive/README.md](contrastive/README.md).

## Reproduce the analyses

| Analysis | Entry point | Output |
|---|---|---|
| Main ERE on official folds | `scripts/02_run_benchmark.py` | `results/ere_official_scaled_v1/` |
| Cross-negative comparison | `scripts/03_cross_negative_eval.py` | Local `results/` |
| Repeated 10% experimental subsets | `scripts/04_run_amplification.py --n-repeats 10` | `results/amplification_repeats10_v1/` |
| Source-held-out sensitivity | `scripts/run_provenance_official_folds.py` | `results/provenance_official_folds_v1/` |
| Experimental/synthetic mixes | `scripts/05_run_mix_experiment.py` | Local `results/` |
| Blend and volume controls | `scripts/06_run_blend_volume.py` | Local `results/` |

The additional CSVs and their interpretation are in [reviewer_updates/](reviewer_updates/). Generated `results/` files are local; released supplementary CSVs are versioned under `reviewer_updates/results/`.

## Repository map

| Path | Contents |
|---|---|
| [data/aptabench/](data/aptabench/) | AptaBench table, split indices, embeddings, and generation input |
| [data/constructed_negatives/](data/constructed_negatives/) | Released fold-specific negative pools |
| [src/](src/) | Constructors, data loading, ERE, and analysis code |
| [contrastive/](contrastive/) | Contrastive model and GRU generation code |
| [scripts/](scripts/) | Reproduction commands |
| [reviewer_updates/](reviewer_updates/) | Repeated-subset and source sensitivity analyses |

## Interpretation

- Official aptamer- and molecule-disjoint folds do not hold out assay or database source. The added source analysis is a sensitivity check within official folds, not an assay-disjoint benchmark.
- Ten independent 10% redraws per molecule-disjoint fold give mean ROC-AUC **0.826 ± 0.070** for 10%-only experimental negatives. Redraws share test folds; comparison with full-data training is descriptive.
- LightGBM isolates negative construction in the main experiment. Synthetic-constructor rankings under molecule shift are architecture-dependent.

## License and data

Code in this release is under [MIT](LICENSE). AptaBench is the upstream benchmark; consult its [repository](https://github.com/GenerativeMolMachines/AptaBench) for dataset terms and citation.

# Oligonucleotide Negative Benchmark

Code and released fold-level data for synthetic negative construction and experimental-reference evaluation (ERE) on AptaBench. The `reviewer_updates/` directory contains the additional analyses reported in the revised paper.

## Layout

```text
src/                 # negative constructors + ERE / mix / blend-volume
contrastive/         # vendored contrastive + GRU generation code
scripts/02–07, 04       # experiment runners (+ amplification)
configs/             # protocol defaults
data/aptabench/      # official table, splits, embeddings, generation input
data/constructed_negatives/  # pre-exported fold CSVs (amplification pools)
data/amplification/        # committed amplification_summary.csv (appendix table)
```

Synthetic negatives are built at runtime per fold (seed 42). Script outputs go to `results/` (generated locally, not committed).

## Contrastive / GRU (mix + learned generation)

The `contrastive/` folder contains the contrastive model, GRU decoder, GENA/ChemBERTa
encoders, and optional ViennaRNA structure helpers. It is added to `PYTHONPATH`
automatically when running mix (`05`) or fraction-local generation.

- **CPU:** main ERE (`02`) uses precomputed `data/aptabench/generation/*.csv`
- **GPU recommended:** `05_run_mix_experiment.py` retrains contrastive + GRU per fold/fraction
- First run downloads HuggingFace models (~1 GB cache)
- Optional: `pip install ViennaRNA` for exact `structure_filtered` behavior

Override vendored copy: `set CONTRASTIVE_CODE_ROOT=C:\path\to\contrastive`

## Setup

```bash
git clone https://github.com/OpiumProger/Neurips_AIDAR.git
cd Neurips_AIDAR
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
set PYTHONPATH=src              # Windows cmd
```

## Run

```bash
python scripts/02_run_benchmark.py --out results/ere_official_scaled_v1 --lgbm-hard-score
python scripts/02_run_benchmark.py --out results/ere_official_v1
python scripts/03_cross_negative_eval.py --cross-csv results/ere_official_scaled_v1/cross_negative_metrics.csv
python scripts/05_run_mix_experiment.py
python scripts/06_run_blend_volume.py
python scripts/04_run_amplification.py
python scripts/07_make_figures.py
```

## License

MIT — see [LICENSE](LICENSE).

## Reviewer-driven updates

The revised paper makes three limits explicit: the official aptamer and molecule splits do not isolate assay or database provenance; the synthetic-constructor ranking changes with the predictor architecture under molecule shift; and some constructors use experimental-negative information during construction. These distinctions concern the interpretation of the original ERE results.

The additional `reviewer_updates/results/` files report ten independent 10% experimental-negative redraws per molecule-disjoint fold and a database-source-held-out sensitivity analysis within the official folds. The 10% experiment comprises five folds, ten redraws, and four training modes (200 model cells). Its 10%-only mean ROC-AUC is 0.826; this is descriptive evidence of robustness to subset redraw, not a paired significance test against the full-data model. Source holdout excludes the held-out database source and test entities from training; it is not an assay- or study-disjoint evaluation. See [the update notes](reviewer_updates/README.md) for file definitions and interpretation.

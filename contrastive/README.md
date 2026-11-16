# Contrastive + GRU generation (vendored)

Minimal copy of the contrastive aptamer–ligand pipeline used by:

- `learned_generated` (precomputed pool + fraction-retrain regeneration)
- `hard_negative` bootstrap scoring (via benchmark `src/`)
- `structure_filtered` when ViennaRNA is installed

## Modules

| File | Role |
|------|------|
| `Model.py` | `MicroContrastiveModel` |
| `DataPrepare.py` | `FinalContrastiveDataset` |
| `Loss.py` | `TemperatureScaledLoss` |
| `GRU.py` | conditional decoder + generation helpers |
| `Embeddings.py` | GENA aptamer + ChemBERTa encoders (HF cache) |
| `aptamer_structure.py` | ViennaRNA MFE filter (optional) |

The benchmark adds this directory to `sys.path` automatically (`src/contrastive_path.py`).
Override with `CONTRASTIVE_CODE_ROOT` if needed.

## Extra dependencies

See root `requirements.txt`: `transformers`, `tqdm`, `python-Levenshtein`.
Optional: `ViennaRNA` for AptaBLE-style structure filtering.

First run downloads HF models (`AIRI-Institute/gena-lm-bert-base`, ChemBERTa).

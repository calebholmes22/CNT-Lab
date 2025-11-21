# CNT Lab Starter Kit

A tidy, opinionated scaffold for your CNT Jupyter workflow.

## Quick Start (Windows / PowerShell)

1. **Choose a root folder** (recommended): `C:\Users\caleb\CNT_Lab`.
2. Unzip this bundle into that folder.
3. Open PowerShell **in the project root** and run:
   ```powershell
   scripts\bootstrap.ps1
   ```
   This sets the `CNT_LAB_DIR` env var (system user scope) and creates the standard folders if missing.
4. Install the helper library in editable mode:
   ```powershell
   python -m pip install -e .
   ```
5. Start JupyterLab and open `notebooks\00_SETUP_TEMPLATE.ipynb`. Run the first cell.

## Philosophy

- **Everything has a place.** Data in `data/`, derived outputs in `artifacts/`, logs in `logs/`.
- **Deterministic saving.** Save via `cntlab.io.save_*` to auto-tag, hash, and register in the manifest.
- **Fetch by tags, not by guess.** Load via `cntlab.manifest.find_artifacts(...)` and `cntlab.io.load_*`.
- **One source of truth** for paths in `configs/settings.toml` and `cntlab.paths`.

## Standard Trees

```
configs/
data/
  raw/         # immutable inputs
  interim/     # transformed but not final
  processed/   # final clean datasets ready for modeling/analysis
artifacts/
  figures/     # .png, .svg
  models/      # model binaries
  metrics/     # .json metrics
  tables/      # .csv, .parquet
  manifests/   # registry of all saved artifacts
logs/
notebooks/
reports/
scripts/
src/cntlab/    # helper library installed in editable mode
```


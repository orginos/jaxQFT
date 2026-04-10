# Processed Phi4 Data Tree

This directory documents the intended location of processed `phi4` analysis products.

Large processed outputs should normally live at NERSC, not in the repo. Keep only small, curated products under version control when needed for papers or regression tests.

## Intended Structure
- `data/phi4/hmc-g2-scan/summary/`
- `data/phi4/hmc-g2-scan/timeseries/`
- `data/phi4/hmc-g2-scan/figures/`
- `data/phi4/rg-comparison/...`

## Typical Contents
- small summary JSON files
- CSV or TeX tables for papers
- curated figure inputs
- compact derived products

Do not store:
- full raw MCMC streams
- large checkpoint files
- unfiltered Slurm logs

# Phi4 HMC `g2` Scan Inputs

This directory is reserved for repository-tracked inputs for the paper-2 HMC campaign.

## Scope
- theory: `2d` scalar `phi^4`
- fixed coupling: `g4 = lambda = 2.4`
- scan variable: `g2 = m^2`
- production volumes: `L = 16, 32, 64, 128, 256`

## Intended Contents
This directory should hold only small, versioned inputs such as:
- batch-scan cards
- production cards
- helper JSON/TOML grids
- campaign-specific README files

It should not contain:
- checkpoints
- raw MCMC histories
- Slurm outputs
- large analysis products

## Planned Input Families

### Batch-size scan
One input family per volume:
- `batch_scan_L16.*`
- `batch_scan_L32.*`
- `batch_scan_L64.*`
- `batch_scan_L128.*`
- `batch_scan_L256.*`

These cards should sweep candidate batched HMC sizes and use the cost metric
`time_per_trajectory * tau_int / batch_size`.

### Production scan
One input family per volume and `g2` value:
- `prod_L{L}_g2_{value}.*`

## Current Status
The existing `scripts/phi4/hmc_phi4.py` already provides:
- HMC evolution
- magnetization
- connected susceptibility
- `C2p`
- `xi`
- IAT-aware errors on the batch-mean series

Still missing before the full campaign:
- Binder cumulant support
- full time-history output
- a standalone IAT-aware reanalysis path from saved histories
- a batch-scan driver

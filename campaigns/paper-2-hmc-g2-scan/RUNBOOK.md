# Paper 2 HMC `g2` Scan Runbook

## Purpose
Operational rules for the `phi^4` HMC scan at fixed `g4 = 2.4`.

This file is the execution contract for future Codex sessions and for manual NERSC work.

## Repo vs Runtime Split

### Keep in the repository
- scripts
- input cards / config files
- campaign docs
- plotting scripts
- analysis code
- summary tables and selected small JSON outputs

### Keep outside the repository
- large run outputs
- checkpoints
- raw MCMC histories
- Slurm logs
- large NPZ archives

Primary runtime root at NERSC:
- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan`

Primary processed-data root at NERSC:
- `/global/cfs/cdirs/hadron/jaxQFT/data/phi4/hmc-g2-scan`

## Runtime Directory Layout

### Batch-scan runs
- `runs/phi4/hmc-g2-scan/batch-scan/L{L}/g2_{g2}/b{batch}/s{seed}/`

### Production runs
- `runs/phi4/hmc-g2-scan/production/L{L}/g2_{g2}/s{seed}/`

### Analysis outputs
- `data/phi4/hmc-g2-scan/summary/`
- `data/phi4/hmc-g2-scan/timeseries/`
- `data/phi4/hmc-g2-scan/figures/`

Each run directory should contain at least:
- `input.toml` or equivalent saved input snapshot
- `job.sbatch`
- `metrics.json`
- `observables_history.npz`
- `slurm/*.out`
- `slurm/*.err`

## Failure Policy

### Non-finite dynamics / NaN observables
- mark run as failed
- do not resume blindly
- resubmit as a fresh replacement seed
- preserve the failed directory for audit

### Timeout
- resume from the last valid saved state if checkpointing exists
- otherwise rerun the same seed with a longer walltime

### OOM / memory pressure
- reduce batch size
- rerun the batch-scan family for that volume if needed
- do not manually edit logs or partial outputs

### Missing or corrupt analysis output
- rerun analysis only
- do not rerun HMC unless the raw history is missing or corrupt

## Resubmission Policy
- production replacement seeds should use a new seed id
- failed runs should never overwrite a completed seed directory
- replacement seeds should be logged explicitly in the campaign summary

## Inspection Policy
After each completed family, inspect:
- acceptance
- mean and error of `xi_2`
- `tau_int` for slow observables
- batch-scan cost metric
- missing/corrupt outputs

Do not unlock the next production family until:
- the analysis outputs are readable
- the cost metric has selected a production batch
- no unresolved systematic code bug is present

## Production Batch Selection Rule
For fixed volume `L`, choose the batch minimizing:
- `time_per_trajectory * tau_int / batch_size`

Use the slowest relevant observable as the decision criterion:
- start with `xi_2`
- cross-check with `chi_m`
- if the two disagree strongly, take the more conservative choice

## Data Movement Policy
- no ad hoc `scp`
- NERSC code updates via `git pull`
- large runtime data stays under the NERSC `runs/` and `data/` roots
- only curated summaries, small JSONs, and publication figures are copied back into the repo when needed

## Immediate Next Physics Step
Implement and test locally:
1. HMC histories with `m`, `|m|`, `m^2`, `m^4`, `chi_m`, `C2p_x`, `C2p_y`
2. Binder cumulant analysis
3. IAT-aware summary analysis
4. batch-scan driver

Only after that:
- run the canonical-point HMC pilot
- then launch the full `g2` matrix

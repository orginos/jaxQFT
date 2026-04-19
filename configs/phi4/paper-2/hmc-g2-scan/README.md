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

The current tracked tuning grid is:
- `configs/phi4/paper-2/hmc-g2-scan/tuning_grid.tsv`

The current tracked `g2` scan points are:
- `configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv`

The blocked-reference RT validation point lists are:
- `configs/phi4/paper-2/hmc-g2-scan/blocked_core_points.tsv`
- `configs/phi4/paper-2/hmc-g2-scan/blocked_interp_points.tsv`

The tuned production settings are:
- `configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv`

### Production scan
One input family per volume and `g2` value:
- `prod_L{L}_g2_{value}.*`

## Runtime Tree
Tracked inputs stay in the repo. Runtime outputs on NERSC should live under:

- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/tuning/`
- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production/`

Recommended tuning layout:
- `runs/phi4/hmc-g2-scan/tuning/L{L}/b{batch}_nmd{nmd}/`

Recommended production layout:
- `runs/phi4/hmc-g2-scan/production/L{L}/g2_{value}/seed_{s}/`

Each run directory should contain:
- `hmc_phi4.json`
- `hmc_phi4.npz`
- `job.sbatch`
- `slurm/*.out`
- `slurm/*.err`

## Current Status
The existing phi4 HMC stack now provides:
- HMC evolution:
  - `scripts/phi4/hmc_phi4.py`
- optional blocked-level measurement using the same average `2x2` RG split as
  the flow analysis:
  - `--measure-blocked-levels`
  - `--blocked-rg-mode average`
- saved raw histories:
  - `m`
  - `E/V`
  - `C2p_x`, `C2p_y`
  - optional low-momentum scan `k=1..k_max`
  - optional per-level blocked histories:
    - `level{l}_magnetization`
    - `level{l}_magnetization2`
    - `level{l}_magnetization4`
    - `level{l}_c2pk_x`
    - `level{l}_c2pk_y`
    - `level{l}_momenta_k`
- observables and errors:
  - `m`, `|m|`, `m^2`, `m^4`
  - `chi_m`
  - Binder ratio `B4`
  - Binder cumulant `U4`
  - `xi2`, `xi2(k)`, fitted low-momentum `xi2`
  - Gamma-analysis IAT-aware primitive errors
  - blocked-jackknife derived errors
- standalone reanalysis:
  - `scripts/phi4/analysis/analyze_hmc_phi4.py`
  - now also emits a `blocked_levels` section summarizing every saved blocked
    level
- tuning metrics recorded in JSON:
  - `measure_usec_per_traj`
  - `tau_int`
  - `time_per_trajectory * tau_int / batch_size`

NERSC launch helpers now exist:
- `scripts/phi4/run_hmc_phi4.sh`
- `scripts/phi4/submit_hmc_phi4_nersc.sh`
- `scripts/phi4/submit_hmc_phi4_tuning_campaign_nersc.sh`
- `scripts/phi4/submit_hmc_phi4_production_campaign_nersc.sh`
- `scripts/phi4/submit_hmc_phi4_blocked_reference_campaign_nersc.sh`
- `scripts/phi4/submit_hmc_phi4_blocked_core_campaign_nersc.sh`
- `scripts/phi4/submit_hmc_phi4_blocked_interp_campaign_nersc.sh`

The intended campaign order is:
1. tune `(batch_size, nmd)` at the canonical point `g2=-0.4`, `g4=2.4`
2. choose settings with acceptance above `90%`
3. launch the full `g2` production matrix with conservative walltimes

Current tuned production choices:
- `L=16`: `batch=128`, `nmd=4`
- `L=32`: `batch=128`, `nmd=4`
- `L=64`: `batch=64`, `nmd=6`
- `L=128`: `batch=32`, `nmd=8`
- `L=256`: `batch=16`, `nmd=12`

## Blocked-Reference RT Campaign

The blocked-reference campaign reuses the tuned production settings at the
paper-critical volumes:

- `L=64`: `batch=64`, `nmd=6`, walltime `04:00:00`
- `L=128`: `batch=32`, `nmd=8`, walltime `06:00:00`

Default runtime root:

- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/blocked-reference`

Tracked submitters:

- core points (`-0.4`, `-0.5`, `-0.585`, `4` replicas):

```bash
scripts/phi4/submit_hmc_phi4_blocked_core_campaign_nersc.sh
```

- interpolation points (`-0.45`, `-0.55`, `-0.57`, `-0.58`, `-0.59`, `-0.60`,
  `2` replicas):

```bash
scripts/phi4/submit_hmc_phi4_blocked_interp_campaign_nersc.sh
```

Both wrappers call the generic blocked-reference submitter, which always enables
`--measure-blocked-levels` and defaults to `--blocked-rg-mode average`.

Operational note:
- the current HMC runner does not checkpoint or resume, so failed jobs restart from scratch
- production walltimes should therefore be intentionally conservative

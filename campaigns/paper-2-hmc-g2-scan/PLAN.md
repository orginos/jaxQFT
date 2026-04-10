# Paper 2 HMC `g2` Scan Plan

## Scope
- Model: `2d` scalar `phi^4`
- Fixed coupling: `g4 = lambda = 2.4`
- Scan variable: `g2 = m^2`
- Initial volume cap: `L = 16, 32, 64, 128, 256`
- Exploratory later volumes: `512`, `1024` only after the first campaign is stable

## Physics Goal
Construct a statistically controlled HMC reference data set for the canonical `phi^4` theory at fixed `g4`, scanning `g2`, and measuring:
- second-moment correlation length `xi_2`
- magnetic susceptibility `chi_m`
- magnetization `m`
- absolute magnetization `|m|`
- Binder cumulant `U_4`
- integrated autocorrelation times `tau_int`

This data set is the physics baseline for the RG paper. It is needed for:
- matching the learned-model blocked physics to target HMC
- comparing locality length `xi_local` to physical `xi_2`
- finite-size scaling and critical-line studies at fixed `g4`

## `g2` Grid
- `-0.40`
- `-0.45`
- `-0.50`
- `-0.55`
- `-0.57`
- `-0.58`
- `-0.585`
- `-0.59`
- `-0.60`
- `-0.65`
- `-0.70`
- `-0.75`
- `-0.80`

## Phase Structure of the Campaign

### Phase 0: Local Code Completion and Validation
Before launching the full matrix at NERSC, implement and test locally:
- Binder cumulant measurement in the `phi4` HMC workflow
- full time-history output for all primary observables
- IAT-aware error analysis for all reported observables
- batch-size scan driver
- persistent run-record output suitable for later re-analysis

Required local checks:
- small-volume smoke tests at `L=16, 32`
- agreement of `xi_2`, `chi_m`, and magnetization with current `hmc_phi4.py`
- Binder sanity checks:
  - symmetric phase: small `|m|`
  - broken/near-critical regime: nontrivial `U_4`
- stable reconstruction of means/errors from saved histories without rerunning dynamics

### Phase 1: Batch-Size Optimization
For each volume `L`, determine the best batched HMC size before scanning `g2`.

Optimization metric:
`time_per_trajectory * tau_int / batch_size`

This should be measured for at least:
- magnetization
- susceptibility
- `C2p` or `xi_2`

The production batch for volume `L` should minimize the worst relevant cost among the slow observables.

### Phase 2: Canonical Point Pilot
At the canonical point:
- `g4 = 2.4`
- `g2 = -0.4`

Run all target volumes:
- `16, 32, 64, 128, 256`

Purpose:
- validate the measurement pipeline
- establish volume scaling of `xi_2`, `chi_m`, `U_4`, `tau_int`
- confirm production wall times and storage requirements

### Phase 3: Full `g2` Scan
Launch the full matrix:
- all `g2` values
- all volumes up to `256`
- same analysis pipeline

Initial priority order:
1. `L=64`
2. `L=128`
3. `L=32`
4. `L=256`
5. `L=16`

Reason:
- `64` and `128` are the most informative for the RG paper
- `32` is cheap and useful for cross-checks
- `256` is the large-volume target
- `16` mainly anchors the small-volume end

## Observables to Store Per Measurement
The full MCMC time history must be retained for later analysis.

At minimum store:
- trajectory index
- action density `E/V`
- signed magnetization `m`
- absolute magnetization `|m|`
- `m^2`
- `m^4`
- connected susceptibility estimator
- `C2p_x`
- `C2p_y`
- averaged `C2p`
- acceptance indicator or acceptance history
- HMC energy violation diagnostics if available

Derived later from the stored histories:
- `chi_m`
- Binder cumulant `U_4`
- `xi_2`
- `tau_int`
- ESS
- blocked-jackknife / IAT-aware mean errors

## Statistical Standard
All final errors must be autocorrelation-aware.

Required analysis standard:
- estimate `tau_int` with the existing `jaxqft.stats.autocorr` tools
- use block sizes tied to `tau_int`
- compute final mean errors with blocked jackknife or an equivalent IAT-aware procedure
- keep the raw histories so analysis can be rerun with improved choices later

## Deliverables for the RG Paper
- HMC reference table at the canonical point
- `g2` scan tables of `xi_2`, `chi_m`, `m`, `|m|`, `U_4`, `tau_int`
- data products usable for:
  - finite-size scaling
  - comparison to learned blocked hierarchies
  - `xi_local / xi_2` locality checks

## Immediate Implementation Backlog
1. Extend the `phi4` HMC measurement code to include Binder cumulant ingredients and full histories.
2. Add a standalone analysis script that ingests saved histories and produces IAT-aware summary JSON.
3. Add a batch-scan script for the cost metric.
4. Add a lightweight run database format for HMC runs.
5. Only then launch the NERSC production matrix.

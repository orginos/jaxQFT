# Paper 2 Experiment Plan

This is the research plan for the RG/universality paper.

Unlike paper 1, this is not primarily an architecture-comparison paper. The
central object is the exact blocked hierarchy of the learned action.

## Main questions

1. Does the learned hierarchy define useful RG trajectories in observable space?
2. Do different bare points flow toward a common renormalized trajectory?
3. Can the learned hierarchy remain informative at large volume even when
   fine-level reweighting is already difficult?
4. Can one treat the learned action as an operational ``quantum-perfect'' action
   in the sense that all blocked levels are exact samplers for the learned
   theory?

## Essential numerical program

### R1. Volume scaling at a fixed target point

Run the current best Wilsonian Gaussian branch at:

- `L = 16`
- `L = 32`
- `L = 64`
- `L = 128` if feasible

Record:

- `std(ΔS)`
- `std(ΔS)/L`
- `ESS`

This establishes whether the residual mismatch remains intensive.

### R2. Blocked observable trajectories

For final production checkpoints, measure at every RG level:

- `xi/L`
- Binder cumulant
- susceptibility
- blocked two-point function at low momentum
- connected four-point observable

Do this first at the canonical point:

- `m^2 = -0.4`
- `lambda = 2.4`

### R3. Bare-point scan

At fixed `lambda = 2.4`, scan several `m^2` values around the target point.

Suggested first scan:

- `m^2 in {-0.55, -0.50, -0.45, -0.40, -0.35, -0.30}`

The goal is to see whether the blocked trajectories merge after several RG
levels.

### R4. Optional projection to coupling space

Only after the observable-space story is clear, fit a truncated coarse action
and extract projected running couplings.

## Required new tooling

This paper will need analysis scripts that are not yet finished:

1. blocked-observable measurement at each RG depth
2. comparison of blocked trajectories across target points
3. optional fitting of a truncated coarse action

The first two are essential; the third is optional.

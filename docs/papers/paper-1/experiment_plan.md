# Paper 1 Experiment Plan

This file is the run plan for the short Wilsonian-transport-map paper in
`docs/papers/paper-1/`.

The purpose is not to do a huge architecture sweep. The purpose is to run the
smallest set of experiments that can support a clean paper narrative within a
two-week window.

## 1. Paper Narrative

The paper claim is:

- a Wilsonian transport map that keeps the coarse field fixed and trivializes
  only local block fluctuations
- together with learned Gaussian priors at each RG level
- produces a small **intensive** mismatch to the target action
- with residual error that scales extensively with volume
- and can be understood with level-resolved diagnostics

Therefore the experiments should answer:

1. Does the learned-Gaussian Wilsonian model beat the earlier simpler branch?
2. Does per-level capacity allocation help?
3. How does the residual mismatch scale with volume?
4. Which RG levels and modules matter most?

The experiments below are prioritized to answer exactly those questions.


## 2. Primary Metrics

Rank runs by:

1. `std(action diff)` = centered `std(ΔS)`
2. `ESS`
3. `std(reweighting factor)`
4. wall-clock cost

Across volumes, the main scaling observable is:

- `std(ΔS) / L`

This is more meaningful than raw `std(ΔS)` because the residual mismatch is
expected to behave like a local extensive sum.


## 3. Canonical Model Family

Use the current Gaussian-prior coarse-eta branch:

- `jaxqft/models/phi4_rg_coarse_eta_gaussian_flow.py`
- `scripts/phi4/train_rg_coarse_eta_gaussian_flow.py`
- `scripts/phi4/analyze_rg_coarse_eta_gaussian_flow.py`

Keep fixed unless explicitly testing an ablation:

- `mass = -0.4`
- `lam = 2.4`
- `parity = sym`
- `rg_type = average`
- `eta_gaussian = level`
- `terminal_prior = learned`
- `terminal_n_layers = 2`
- `terminal_width = 64`
- `output_init_scale = 1e-2`
- `log_scale_clip = 5.0`
- `offdiag_clip = 2.0`
- `validation.each_stage = true`
- `save_every = 1000`


## 4. Core Production Runs

These are the runs needed for the paper itself.

### P1. L=16 production

Current best known per-level setup:

- `width_levels = [96, 64, 96]`
- `n_cycles_levels = [3, 2, 3]`
- `radius_levels = [1, 1, 2]`
- `gaussian_width_levels = [64, 64, 64]`
- `gaussian_radius_levels = [1, 1, 1]`

Required seeds:

- `seed = 0, 1, 2`

Purpose:

- establish the final `16^2` production point
- estimate run-to-run spread
- produce a robust `16^2` reference for the scaling plot


### P2. L=32 production

Current best direct extension:

- `width_levels = [128, 96, 64, 96]`
- `n_cycles_levels = [3, 3, 2, 3]`
- `radius_levels = [1, 1, 1, 2]`
- `gaussian_width_levels = [64, 64, 64, 64]`
- `gaussian_radius_levels = [1, 1, 1, 1]`

Required seeds:

- `seed = 0, 1`

Purpose:

- complete the current promising `32^2` line through annealing
- provide the second point in the main scaling plot


### P3. L=64 direct extension

Recommended direct extension:

- `width_levels = [160, 128, 96, 64, 96]`
- `n_cycles_levels = [3, 3, 3, 2, 3]`
- `radius_levels = [1, 1, 1, 1, 2]`
- `gaussian_width_levels = [64, 64, 64, 64, 64]`
- `gaussian_radius_levels = [1, 1, 1, 1, 1]`

Required seeds:

- `seed = 0`

Purpose:

- first large-volume scaling point beyond `32^2`
- show whether `std(ΔS)/L` remains roughly constant


### P4. L=128 exploratory extension

Recommended direct extension:

- `width_levels = [192, 160, 128, 96, 64, 96]`
- `n_cycles_levels = [3, 3, 3, 3, 2, 3]`
- `radius_levels = [1, 1, 1, 1, 1, 2]`
- `gaussian_width_levels = [64, 64, 64, 64, 64, 64]`
- `gaussian_radius_levels = [1, 1, 1, 1, 1, 1]`

Required seeds:

- `seed = 0`

Purpose:

- exploratory point only
- if the run is affordable, it becomes a very strong scaling figure
- if it is too expensive, drop it without harming the core paper


## 5. Essential Ablations

These are the only ablations that are really needed for the paper.

### A1. Gaussian priors matter

Run at `L=16` and `L=32`:

1. full model:
   - `eta_gaussian = level`
   - `terminal_prior = learned`
2. no learned Gaussian priors:
   - `eta_gaussian = none`
   - `terminal_prior = std`

Purpose:

- isolate the effect of learned Gaussian whitening

Required seeds:

- `L=16`: `seed = 0, 1`
- `L=32`: `seed = 0`


### A2. Per-level capacity matters

Run at `L=16` and `L=32`:

1. per-level model:
   - as in the production configuration
2. uniform-capacity model:
   - same branch
   - scalar `width`, `n_cycles`, `radius`
   - use the best known uniform-capacity settings as the control

Recommended uniform controls:

- `L=16`: `width = 64`, `n_cycles = 2`, `radius = 1`
- `L=32`: `width = 96`, `n_cycles = 3`, `radius = 1`

Purpose:

- demonstrate that level-dependent capacity allocation is useful

Required seeds:

- `L=16`: `seed = 0, 1`
- `L=32`: `seed = 0`


### A3. Simpler terminal is enough

Run at `L=16`:

1. default terminal:
   - `terminal_n_layers = 2`
   - `terminal_width = 64`
2. larger terminal:
   - `terminal_n_layers = 4`
   - `terminal_width = 128`

Purpose:

- support the claim that the simpler terminal is adequate and that the gains are
  not coming simply from a bigger terminal RealNVP

Required seeds:

- `seed = 0`


## 6. Diagnostic Runs

These are needed for the figures and interpretation.

### D1. HMC level diagnostics and knockouts

Run on final or near-final checkpoints for:

- `L=16` production winner
- `L=32` production winner

Command family:

```bash
source /opt/python/jax/bin/activate
MPLCONFIGDIR=/tmp/mpl-cache JAX_PLATFORMS=cpu \
python scripts/phi4/analyze_rg_coarse_eta_gaussian_flow.py \
  --resume <checkpoint.pkl> \
  --nwarm 100 --nmeas 32 --nskip 5 \
  --batch-size 16 \
  --chunk-size 128 \
  --include-grouped \
  --tests hmc,knockout,conditional
```

Purpose:

- produce the knockout-importance figure
- produce the conditional-decomposition figure
- support the interpretation of where the residual mismatch lives


### D2. Validation-only snapshots at every stage

Use:

```bash
source /opt/python/jax/bin/activate
MPLCONFIGDIR=/tmp/mpl-cache JAX_PLATFORMS=cpu \
python scripts/phi4/train_rg_coarse_eta_gaussian_flow.py \
  --config <config.toml> \
  --resume <checkpoint.pkl> \
  --validate --validate-only
```

Purpose:

- collect stage-by-stage `std(ΔS)`, `std(w)`, and `ESS`
- support a training-curve figure if needed


## 7. Training Schedules

Use staged schedules in TOML. On A100 GPUs, the target should be the largest
affordable batch rather than a fixed universal value.

### S16

Recommended for `L=16`:

- ramp:
  - `32 -> 64 -> 128 -> 256 -> 512 -> 1024`
  - `1000` epochs per stage
- if memory allows, continue to `2048`
- anneal at `Bmax`:
  - `1e-4` for `1000` epochs
  - `3e-5` for `1000` epochs
  - optional `1e-5` for `1000` epochs


### S32

Recommended for `L=32`:

- ramp:
  - `16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024`
  - `1000` epochs per stage
- anneal at `1024`:
  - `1e-4` for `1000` epochs
  - `3e-5` for `1000` epochs


### S64

Recommended for `L=64`:

- ramp:
  - start at `8` or `16`
  - double up to the largest affordable batch (`256` or `512` are realistic targets)
  - `1000` epochs per stage
- anneal at `Bmax`:
  - `1e-4` for `1000` epochs
  - `3e-5` for `1000` epochs


### S128

Recommended for `L=128`:

- exploratory only
- ramp:
  - start at `4` or `8`
  - double up to the largest affordable batch
- do not overcommit to a full exhaustive anneal unless the early stages are
  clearly promising


## 8. File Naming

Use checkpoint names that encode the volume, the per-level schedule, and the
seed.

Recommended stem:

`rg_coarse_eta_gauss_L{L}_m{m2}_l{lam}_perlevel_w{wlist}_nc{nclist}_r{rlist}_eglevel_tglearned_s{seed}`

Examples:

- `rg_coarse_eta_gauss_L16_m-0.4_l2.4_perlevel_w96-64-96_nc3-2-3_r1-1-2_eglevel_tglearned_s0.pkl`
- `rg_coarse_eta_gauss_L64_m-0.4_l2.4_perlevel_w160-128-96-64-96_nc3-3-3-2-3_r1-1-1-1-2_eglevel_tglearned_s0.pkl`


## 9. Minimal Run Set For The Paper

If compute becomes tight, do only this:

1. `L=16` production winner, seeds `0,1,2`
2. `L=32` production winner, seeds `0,1`
3. `L=64` production winner, seed `0`
4. Ablation A1 at `L=16`
5. Ablation A2 at `L=16`
6. HMC diagnostics for final `L=16` and `L=32`

That is sufficient to support the short paper.


## 10. Nice-To-Have Extensions

If there is spare compute:

1. `L=128` exploratory run
2. `L=32` ablations with seed `1`
3. `L=64` uniform-capacity control
4. one transport-HMC pilot using the final `L=16` or `L=32` winner


## 11. Final Deliverables For The Paper

The student should return, for each completed production run:

1. checkpoint file
2. full stdout log
3. final validation summary:
   - `mean action diff`
   - `std action diff`
   - `std reweighting factor`
   - `ESS`
4. stage-by-stage table with:
   - epoch
   - batch
   - lr
   - `std(ΔS)`
   - `std(w)`
   - `ESS`
5. wall-clock time per stage

This is enough to make the main paper plots and tables.

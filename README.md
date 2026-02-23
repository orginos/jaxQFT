# jaxQFT

JAX transcription of core `torchQFT` utilities.

## Ported modules (initial pass)

- `jaxqft.models.phi4.Phi4`: lattice scalar `phi^4` action/force + momentum refresh.
- `jaxqft.models.phi4_mg`: hierarchical MG normalizing flow for `phi^4`.
- `jaxqft.models.stacked_mg`: stacked variant of MGFlow.
- `jaxqft.models.su3_ym.SU3YangMills`: pure SU(3) Yang-Mills theory + HMC-compatible API.
- `jaxqft.core.integrators`: `leapfrog`, `minnorm2`, `minnorm4pf4`, and r-RESPA variants.
- `jaxqft.core.update.HMC`: HMC evolution with Metropolis accept/reject tracking.
- `jaxqft.flows`: minimal `Shift2D`, `AffineCoupling`, and `FlowModel`.
- `jaxqft.LieGroups`: JAX ports of `so3`, `su2`, `su3`, with shared Lie helpers.

## Package layout

- `jaxqft/models`: model-specific code (`phi4`, and future `o3`, etc.).
- `jaxqft/core`: generic algorithms (`integrators`, `update`, `hmc_sampler`).
- `jaxqft/LieGroups`: general Lie-group utilities.
- `scripts`: runnable scripts grouped by model.
  - `scripts/phi4/hmc_phi4.py`
  - `scripts/phi4/train_mg_single.py`
  - `scripts/phi4/train_stacked_mg.py`
  - `scripts/su3_ym/hmc_su3_ym.py`
  - `scripts/su3_ym/bench_hmc_su3.py`

## Install

```bash
pip install -e .
```

## Run

```bash
python scripts/phi4/hmc_phi4.py
python scripts/su3_ym/hmc_su3_ym.py --L 4 --Nd 4 --beta 5.8 --layout auto
python scripts/su3_ym/bench_hmc_su3.py --shape 8,8,8,8 --integrator forcegrad --nmd 8 --tau 1.0
```

```bash
python scripts/phi4/train_mg_single.py
python scripts/phi4/train_stacked_mg.py
```

## Notes

- API compatibility aliases from Torch naming are kept (`refreshP`, `evolveQ`, `calc_Acceptance`, etc.).
- RNG is explicit in JAX; these classes keep an internal key for a practical Torch-like workflow.
- Higher-level training scripts from `torchQFT` are not yet ported in this first step.

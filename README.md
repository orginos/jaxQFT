# jaxQFT

JAX transcription of core `torchQFT` utilities.

## Ported modules (initial pass)

- `jaxqft.models.phi4.Phi4`: lattice scalar `phi^4` action/force + momentum refresh.
- `jaxqft.models.phi4_mg`: hierarchical MG normalizing flow for `phi^4`.
- `jaxqft.models.phi4_rg_cond_flow`: RG-first conditional flow with local-transformer coupling nets for `phi^4`, with selectable `Z_2` parity symmetry (`--parity {none,sym}`, default `sym`).
- `jaxqft.models.phi4_rg_checkerboard_flow`: RG checkerboard conditional flow for `phi^4`, using red/black `\eta` updates plus a `(1,1)` reblocking shift cycle controlled by `--n-cycles`, with selectable conditioner `--conditioner {transformer,mlp}`.
- `jaxqft.models.phi4_rg_checkerboard_inblock_flow`: checkerboard RG flow with the same red/black plus shifted-blocking cycle, but each color pass now does masked in-block mixing among the three block fluctuations via `--n-inner-couplings`.
- `jaxqft.models.phi4_rg_coarse_eta_flow`: RG coarse-lattice fluctuation flow for `phi^4`, which keeps the coarse field fixed at each level and trivializes only the local `\eta \in \mathbb{R}^3` fluctuations using red/black coarse-lattice sweeps with a local neighborhood MLP.
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
  - `scripts/phi4/check_rg_cond_flow.py`
  - `scripts/phi4/check_rg_checkerboard_flow.py`
  - `scripts/phi4/check_rg_checkerboard_inblock_flow.py`
  - `scripts/phi4/check_rg_coarse_eta_flow.py`
  - `scripts/phi4/train_mg_single.py`
  - `scripts/phi4/train_rg_cond_flow.py`
  - `scripts/phi4/train_rg_checkerboard_flow.py`
  - `scripts/phi4/train_rg_checkerboard_inblock_flow.py`
  - `scripts/phi4/train_rg_coarse_eta_flow.py`
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
JAX_PLATFORMS=cpu python scripts/phi4/check_rg_cond_flow.py --selfcheck-fail
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_cond_flow.py --L 16 --lam 2.4 --mass -0.4 --validate
JAX_PLATFORMS=cpu python scripts/phi4/check_rg_checkerboard_flow.py --selfcheck-fail
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_checkerboard_flow.py --L 16 --lam 2.4 --mass -0.4 --n-cycles 1 --validate
JAX_PLATFORMS=cpu python scripts/phi4/check_rg_checkerboard_inblock_flow.py --selfcheck-fail
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_checkerboard_inblock_flow.py --L 16 --lam 2.4 --mass -0.4 --n-cycles 1 --n-inner-couplings 3 --validate
JAX_PLATFORMS=cpu python scripts/phi4/check_rg_coarse_eta_flow.py --selfcheck-fail
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py --L 16 --lam 2.4 --mass -0.4 --n-cycles 2 --radius 1 --validate
python scripts/phi4/train_stacked_mg.py
```

## Notes

- API compatibility aliases from Torch naming are kept (`refreshP`, `evolveQ`, `calc_Acceptance`, etc.).
- RNG is explicit in JAX; these classes keep an internal key for a practical Torch-like workflow.
- Higher-level training scripts from `torchQFT` are not yet ported in this first step.
- A precise code-level note for the new RG-conditional model lives in `docs/notes/phi4_rg_conditional_transformer_flow.tex`.
- A precise code-level note for the checkerboard RG model lives in `docs/notes/phi4_rg_checkerboard_flow.tex`.
- A precise code-level note for the checkerboard in-block RG model lives in `docs/notes/phi4_rg_checkerboard_inblock_flow.tex`.
- A precise code-level note for the coarse-lattice fluctuation RG model lives in `docs/notes/phi4_rg_coarse_eta_flow.tex`.
- The new RG-conditional trainer/checker expose `--parity {none,sym}`; `sym` enforces the old-map-style `z \mapsto -z`, `\phi \mapsto -\phi` symmetry and is the default.
- The checkerboard RG trainer/checker expose `--n-cycles`, which counts red/black + shifted red/black update cycles per non-terminal RG level.
- The checkerboard RG trainer/checker also expose `--conditioner {transformer,mlp}`; the MLP option is a much cheaper pointwise conditioner.
- The new in-block checkerboard trainer/checker keep the previous checkerboard code unchanged and add `--n-inner-couplings`, which controls how many masked in-block `\eta` updates are done inside each red or black color pass.
- The new coarse-eta trainer/checker keep the coarse field fixed at each RG level and expose `--radius`, which sets the neighborhood size on the coarse lattice used to parametrize the local `\eta` trivialization.

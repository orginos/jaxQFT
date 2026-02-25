# HANDOFF

Last updated: 2026-02-24

## Project Snapshot
- Repository: `jaxQFT`
- Goal: high-performance JAX lattice QFT framework, ported from `torchQFT`.
- Core split:
  - `jaxqft/models`: model-specific physics.
  - `jaxqft/core`: generic integrators and update algorithms.
  - `jaxqft/LieGroups`: reusable Lie-group math.
  - `scripts/<model>/`: runnable production/benchmark scripts.

## Implemented Status
- Pure gauge:
  - `SU3YangMills`, `SU2YangMills`, `U1YangMills`.
  - HMC production + dedicated trajectory benchmarks for each:
    - `scripts/su3_ym/hmc_su3_ym.py`, `scripts/su3_ym/bench_hmc_su3.py`
    - `scripts/su2_ym/hmc_su2_ym.py`, `scripts/su2_ym/bench_hmc_su2.py`
    - `scripts/u1_ym/hmc_u1_ym.py`, `scripts/u1_ym/bench_hmc_u1.py`
- Update algorithms:
  - `HMC` and `SMD` (`GHMC` alias) in `jaxqft/core/update.py`.
  - Production scripts support `--update {hmc,smd,ghmc}` with default `hmc`.
  - SMD rejection branch flips post-OU momentum (`p <- -p_ou`) per Lüscher SMD convention.
  - If a theory sets `requires_trajectory_refresh=True`, updates call `prepare_trajectory(q, traj_length)` internally once per trajectory.
  - Fast-scan update path is disabled automatically for such theories.
- Monomial Hamiltonian infrastructure:
  - `jaxqft/core/hamiltonian.py` with `Monomial` protocol and `HamiltonianModel`.
  - Initial SU3 Wilson Nf=2 monomial composition implemented in `jaxqft/models/su3_wilson_nf2.py`.
    - Gauge monomial uses optimized pure-gauge SU3 action/force.
    - Pseudofermion monomial owns pseudofermion field state and computes action/force.
    - Theory exposes `prepare_trajectory(U, traj_length)` for stochastic monomial refresh.
    - Pseudofermion monomial supports refresh modes:
      - `heatbath` (independent redraw)
      - `ou` (OU rotation with `c1=exp(-gamma*traj_length)` on underlying Gaussian field)
      - if pseudofermion gamma is omitted and refresh mode is `ou`, it defaults to SMD/GHMC `gamma`
  - `SU3WilsonNf2` sets `requires_trajectory_refresh=True` when stochastic monomials are present.
- Integrators:
  - 2nd-order (`leapfrog`, `minnorm2`) and 4th-order (`forcegrad`, `minnorm4pf4`) in `jaxqft/core/integrators.py`.
- Diagnostics/tests in model CLIs:
  - `layout`, `timing`, `fd`, `autodiff`, `eps2`, `eps4`, `gauge`, `forcecov`, `forceimpl`, `selfcheck`.
- Autocorrelation/statistics:
  - `jaxqft/stats/autocorr.py` with IPS, Sokal, and Gamma methods.
- Fermion building blocks:
  - `jaxqft/fermions/gamma.py`, `jaxqft/fermions/wilson.py`.
  - Wilson Nf=2 theory wrappers for SU3/SU2/U1:
    - `su3_wilson_nf2.py`, `su2_wilson_nf2.py`, `u1_wilson_nf2.py`.

## Checkpoint Semantics (Production HMC scripts)
- Saved payload contains:
  - Gauge field `q`
  - `theory_key`
  - `update_key` (generic updater RNG)
  - `update_momentum` (for SMD/GHMC; `None` for HMC)
  - `hmc_key` (legacy compatibility)
  - `accept_reject` history
  - `state` and `config`
- Resume behavior:
  - If `--update` is omitted, updater comes from checkpoint config (fallback `hmc`).
  - Passing `--update ...` overrides checkpoint updater.
  - Resuming HMC checkpoint with SMD is supported; momentum is initialized as needed.

## Known Runtime Notes
- macOS default backend policy in model CLIs is CPU unless explicitly overridden.
- METAL backend can be unstable depending on JAX build/version.
- CPU performance depends strongly on thread config:
  - `--cpu-threads`
  - `--cpu-onednn`

## Quick Commands
- SU3 production:
  - `python scripts/su3_ym/hmc_su3_ym.py --shape 8,8,8,8 --integrator forcegrad --nmd 7 --tau 1.0`
- SU3 Wilson Nf=2 production:
  - `python scripts/su3_wilson_nf2/hmc_su3_wilson_nf2.py --shape 4,4,4,8 --integrator minnorm2 --nmd 8 --tau 1.0 --mass 0.05 --beta 5.8`
- SU3 benchmark:
  - `python scripts/su3_ym/bench_hmc_su3.py --shape 16,16,16,16 --integrator forcegrad --nmd 12 --tau 1.0`
- SU2/U1 selfcheck:
  - `python -m jaxqft.models.su2_ym --selfcheck --selfcheck-fail`
  - `python -m jaxqft.models.u1_ym --selfcheck --selfcheck-fail`
- Autodiff force check (example):
  - `python -m jaxqft.models.su3_ym --tests autodiff --shape 4,4,4,8 --n-iter-timing 10`
- SU3 Wilson monomial check:
  - `python -m jaxqft.models.su3_wilson_nf2 --tests hamiltonian --shape 4,4,4,8`
- SU3 Wilson pseudofermion refresh diagnostics:
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfrefresh --pf-refresh heatbath --shape 4,4,4,8`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfrefresh --pf-refresh ou --smd-gamma 0.3 --traj-length 1.0 --shape 4,4,4,8`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfid --shape 4,4,4,8`

## Priority Backlog
1. Add nested integrator schedules over monomial timescales (Sexton-Weingarten style).
2. Add Hasenbusch-preconditioned monomials for SU3 Wilson Nf=2.
3. Add RHMC monomials for odd-flavor support.
4. Add backend abstraction for fermion operators/solvers (`jax` default, `quda` optional later).
5. Multi-device/domain decomposition across spacetime dimensions (not batch-only).
6. Keep benchmark baselines per model/lattice/backend in versioned docs.

## Session Resume Protocol
1. Read `AGENTS.md` and this `HANDOFF.md`.
2. Run one target selfcheck and one target benchmark.
3. Continue from open backlog item with explicit command + expected metric.

## Public-Safe Template
Copy/paste for public progress notes:

```md
### Handoff Update (Public)
- Date: YYYY-MM-DD
- Branch/Commit: <branch> / <sha>
- Completed:
  - <item 1>
  - <item 2>
- Verified with:
  - `<command 1>` -> <key result>
  - `<command 2>` -> <key result>
- Open Issues:
  - <issue or risk>
- Next Step:
  - <single next action>
```

Do not include secrets, private hostnames, internal file paths, or non-public data references.

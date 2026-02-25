# HANDOFF

Last updated: 2026-02-25

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
    - Pseudofermion monomial now has force modes:
      - `autodiff` (reference)
      - `analytic` (CG + explicit force kernel)
    - Solver modularity for pseudofermions:
      - `solver_kind`: `cg`, `bicgstab`, or `gmres`
      - `solver_form`: `normal` (`D^\dagger D`) or `split` (`D^\dagger` then `D`)
      - matrix-free stencil operator application (no assembled sparse matrix)
      - split-form action optimization: one solve (`D^\dagger \chi=\phi`, then `S=\chi^\dagger\chi`)
      - split-form force still uses two solves; `Y` is reused from first split solve
    - EO-preconditioning groundwork:
      - even/odd parity masks and Schur matvec blocks added (`apply_eo_schur_even/odd`)
      - CLI `--tests eo` validates EO Schur operator against full-$D$ block reference and reports timing
      - EO split solve path integrated as `solver_form=eo_split` for pseudofermion action/force solves
      - unpreconditioned `eo_split` currently uses full-lattice masked vectors
    - EO-preconditioned monomial:
      - new fermion monomial kind `eo_preconditioned` (select via `fermion_monomial_kind`)
      - action defined on even Schur operator, with refresh `phi_e = S_e^\dagger eta_e`
      - EO-preconditioned solves now use compact even-checkerboard unknown vectors internally (no odd-subspace identity augmentation)
      - compact checkerboard-native `K_oe/K_eo` kernels added for EO Schur matvec (precomputed parity-neighbor maps)
      - force supports both `autodiff` and `analytic` modes
      - analytic EO-preconditioned force implemented from Schur variation using two mapped dslash-force terms
    - CLI test `forcecmp` compares analytic vs autodiff force, checks Lie-directional consistency, and reports:
      - analytic split timing (inversion vs force-kernel)
      - solve residual metrics
      - best-effort solver iteration metrics from solver `info`
      - fallback solver diagnostics when `info` has no iterations:
        - `DdagD` (or `D/Ddag`) matvec sec/call
        - solve operator-application equivalent (`solve_op_apply_equiv`)
    - `u1_wilson_nf2` `forcecmp` now reports the same solver fallback diagnostics.
    - CLI test `eopforce` reports:
      - forward/central finite-difference directional slope
      - direct autodiff directional derivative
      - `-<F,H>` from autodiff force and analytic force
      - action/force timing (autodiff and analytic)
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
  - Wilson hop convention aligned with Chroma dslash (`forward: r-\gamma_\mu`, `backward: r+\gamma_\mu` for undaggered branch).
  - `WilsonDiracOperator` now exposes explicit `apply_diag` + `apply_dslash` decomposition.
  - In `SU3WilsonNf2`, parity-coupling helpers `K_oe/K_eo` and daggered variants are explicit and used in EO Schur operators.
  - Wilson Nf=2 theory wrappers for SU3/SU2/U1:
    - `su3_wilson_nf2.py`, `su2_wilson_nf2.py`, `u1_wilson_nf2.py`.
  - Wilson SU3/U1 now default to JIT-wrapped Dirac kernels and solver paths for runtime use:
    - `jit_dirac_kernels=True` (default)
    - `jit_solvers=True` (default)
    - CLI toggles available:
      - `--no-jit-dirac-kernels`
      - `--no-jit-solvers`
  - U1 Wilson no longer wraps links as `[...,1,1]` in Dirac operator application:
    - scalar-link path is used directly (`take_mu=self._take_mu`, dagger=`conj`)
    - generic `WilsonDiracOperator.color_mul_left` now supports scalar links.
  - `u1_wilson_nf2.py` now mirrors the SU3 Wilson CLI harness and monomial architecture:
    - monomial composition via `HamiltonianModel` with `gauge` + fermion monomials
    - fermion monomial kinds: `unpreconditioned`, `eo_preconditioned`
    - pseudofermion refresh modes: `heatbath`, `ou`
    - pseudofermion force modes: `autodiff`, `analytic`
    - solver modularity: `cg`, `bicgstab`, `gmres`; forms `normal`, `split`, `eo_split`
    - EO Schur operators and EO-preconditioned action/force path
    - SU3-parity CLI test set available for U1:
      `gamma,adjoint,normal,perf,eo,hamiltonian,pfrefresh,pfid,gauge,conventions,forcecmp,eopforce`
  - New gamma representation note: `docs/notes/gamma_conventions.tex`.

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
- SU3 Wilson pseudofermion force comparison:
  - `python -m jaxqft.models.su3_wilson_nf2 --tests forcecmp --shape 4,4,4,8 --pf-force-mode analytic --force-compare-trials 2 --force-compare-iters 2`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests forcecmp --shape 4,4,4,8 --solver-kind bicgstab --solver-form split --pf-force-mode analytic`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests eopforce --shape 4,4,4,8 --fermion-monomial-kind eo_preconditioned --solver-kind cg --solver-form normal --pf-fd-eps 5e-4 --pf-fd-trials 2`
- SU3 Wilson pseudofermion refresh diagnostics:
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfrefresh --pf-refresh heatbath --shape 4,4,4,8`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfrefresh --pf-refresh ou --smd-gamma 0.3 --traj-length 1.0 --shape 4,4,4,8`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfid --shape 4,4,4,8`
  - `python -m jaxqft.models.su3_wilson_nf2 --tests pfid --shape 4,4,4,8 --fermion-monomial-kind eo_preconditioned`
- U1 Wilson Nf=2 (2D default) full harness checks:
  - `python -m jaxqft.models.u1_wilson_nf2 --tests all --shape 8,8`
  - `python -m jaxqft.models.u1_wilson_nf2 --tests forcecmp --shape 8,8 --pf-force-mode analytic`
  - `python -m jaxqft.models.u1_wilson_nf2 --tests eopforce --shape 8,8 --fermion-monomial-kind eo_preconditioned --pf-force-mode analytic`
 - Cross-model Wilson benchmark matrix:
   - `python scripts/wilson/bench_wilson_matrix.py --kernel-iters 20 --solve-iters 3`
   - includes U1/SU3 eager-vs-jit and SU2 manual eager-vs-jit wrappers
   - optional raw output: `--json-out wilson_matrix.json`

### JIT Regime Note
- Earlier “optimization had little effect” conclusions for Wilson fermion kernels were confounded by eager-dispatch overhead.
- With default JIT on:
  - U1 `8x8` D kernel optimized/reference ratio ~`1e-2` in `--tests perf`.
  - U1 `256x256` D kernel optimized/reference ratio ~`1.3e-1`.
  - SU3 `4^3x8` D kernel optimized/reference ratio ~`2.7e-1`.
- Re-evaluate performance decisions in this JIT regime before pruning kernel variants.

### SU2 Wilson Status
- `su2_wilson_nf2.py` is still a minimal module and is not yet feature-parity with SU3/U1 Wilson modules.
- Missing parity items include monomial composition, EO-preconditioned fermion monomial, and full CLI harness (`forcecmp/eopforce/pfrefresh/...`).

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

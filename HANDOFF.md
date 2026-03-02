# HANDOFF

Last updated: 2026-02-27

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
- Unified production control (new):
  - `scripts/mcmc/mcmc.py` is a control-file-driven MCMC entry point (current theory target: `su3_wilson_nf2`).
  - Supports phased run control:
    - `warmup_no_ar` (HMC-only, accept/reject disabled)
    - `warmup_ar` (Metropolis enabled, optional `nmd` adaptation)
    - `measure`
  - Separates restart checkpoints from saved gauge-field configurations.
  - Supports SU3 Wilson LIME initialization from control file:
    - `[input].init_cfg_lime`, `[input].init_mom_lime`, `[input].init_pf_lime`
    - optional PF leaf index and checkerboard fix controls:
      - `init_pf_field_index`, `init_pf_cb_fix={none,auto,shiftx1}`
    - one-shot skip of first pseudofermion refresh after load:
      - `init_use_loaded_pf_first_traj=true`
    - on `output.resume`, `input.init_*_lime` is ignored (checkpoint state wins).
  - TOML template generation: `--write-template`.
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
      - chronological warm-start for iterative solves:
        - cached previous solution used as `x0` for subsequent CG calls in the same pseudofermion trajectory
        - cache is reset on pseudofermion refresh/clear
        - enabled for both unpreconditioned normal solves and EO-preconditioned normal Schur solves
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
      - standalone `eopforce` timings now disable chrono-guess reuse between timed calls (no same-field last-solution timing inflation)
    - Theory exposes `prepare_trajectory(U, traj_length)` for stochastic monomial refresh.
    - Pseudofermion monomial supports refresh modes:
      - `heatbath` (independent redraw)
      - `ou` (OU rotation with `c1=exp(-gamma*traj_length)` on underlying Gaussian field)
      - if pseudofermion gamma is omitted and refresh mode is `ou`, it defaults to SMD/GHMC `gamma`
    - `SU3WilsonNf2.set_loaded_pseudofermion(phi, skip_next_refresh=True)` added for importing external pseudofermion states.
      - `prepare_trajectory(...)` now supports one-shot refresh skip so the loaded pseudofermion is actually used on the first trajectory.
  - `SU3WilsonNf2` sets `requires_trajectory_refresh=True` when stochastic monomials are present.
- Integrators:
  - 2nd-order (`leapfrog`, `minnorm2`) and 4th-order (`forcegrad`, `minnorm4pf4`) in `jaxqft/core/integrators.py`.
- Diagnostics/tests in model CLIs:
  - `layout`, `timing`, `fd`, `autodiff`, `eps2`, `eps4`, `gauge`, `forcecov`, `forceimpl`, `selfcheck`.
  - `su3_wilson_nf2` now also includes:
    - `pfsolve`: pseudofermion solver-path consistency (`normal` residuals; split/eo-split cross-checks when non-Hermitian solvers are used)
    - `pfcov`: pseudofermion-force gauge covariance (autodiff and analytic)
- Inline measurement infrastructure:
  - `jaxqft/core/measurements.py`
  - `MeasurementContext` provides per-step data handoff between ordered measurements.
  - Measurement registry/execution helpers:
    - `build_inline_measurements(...)`
    - `run_inline_measurements(...)`
  - Current built-in measurement: `plaquette`.
- SciDAC/ILDG LIME I/O:
  - New module: `jaxqft/io/lime.py`
  - Reads LIME headers/records and decodes SciDAC fields (big-endian, lexicographic site order).
  - Validated on Chroma checkpoint files:
    - `run-6_cfg_100.lime` gauge decode gives plaquette `0.4005066` (matches XMLDAT at update 100).
    - `smd-6_mom_100.lime` decodes as traceless anti-Hermitian SU(3) momentum matrices.
    - `smd-1_pf_{100,200,300,400,500}.lime` decode works:
      - files contain extra global metadata records (`datatype=char`, `precision=U`, `recordtype=1`)
      - pseudofermion leaf is the `recordtype=0` floating field (`datatype=Lattice`, `spins=4`, `colors=3`, `typesize=96`, `datacount=1`)
      - decoded pseudofermion shape: `(1, Lx, Ly, Lz, Lt, 4, 3)` for SU3 Wilson runs
  - CLI helper:
    - `python -m jaxqft.io.lime <file.lime> --decode`
    - `python -m jaxqft.io.lime <cfg.lime> --decode-gauge --check-plaq --beta <beta>`
    - `python -m jaxqft.io.lime <mom.lime> --decode-momentum`
    - `python -m jaxqft.io.lime <pf.lime> --decode-pf` (auto-selects pseudofermion leaf field)
- Autocorrelation/statistics:
  - `jaxqft/stats/autocorr.py` with IPS, Sokal, and Gamma methods.
- Fermion building blocks:
  - `jaxqft/fermions/gamma.py`, `jaxqft/fermions/wilson.py`.
  - Wilson hop convention aligned with Chroma dslash (`forward: r-\gamma_\mu`, `backward: r+\gamma_\mu` for undaggered branch).
  - Fermion boundary phases are now supported in SU3 Wilson Nf=2 via `fermion_bc` (e.g. `1,1,1,-1`):
    - implemented as a precomputed link-factor tensor
    - applied to links once per operator/solve input gauge field (no boundary-branching inside hot dslash loops)
    - wired consistently through `D`, `D^\dagger`, Schur operators, and pseudofermion force kernels
  - `WilsonDiracOperator` now exposes explicit `apply_diag` + `apply_dslash` decomposition.
  - Wilson color multiply now has specialized small-Nc kernels:
    - SU3 (`3x3`) and SU2 (`2x2`) paths in `WilsonDiracOperator.color_mul_left`
    - fallback remains generic `einsum` for other Nc
  - SU3 Wilson matvec throughput improved substantially after small-Nc specialization.
  - `WilsonDiracOperator` now includes FLOP estimators for Wilson matvecs:
    - `flops_per_site_dslash(...)`
    - `flops_per_site_matvec(...)`
    - `flops_per_matvec_call(...)`
  - Wilson model `--tests perf` now reports:
    - FLOPs/site for `D` (sparse and dense gamma paths)
    - timing (sec/call) for `D` and `D^\dagger D`
    - achieved GFLOP/s for `D` and `D^\dagger D`
  - In `SU3WilsonNf2`, parity-coupling helpers `K_oe/K_eo` and daggered variants are explicit and used in EO Schur operators.
  - Wilson Nf=2 theory wrappers for SU3/SU2/U1:
    - `su3_wilson_nf2.py`, `su2_wilson_nf2.py`, `u1_wilson_nf2.py`.
  - `su2_wilson_nf2.py` now mirrors SU3/U1 Wilson architecture and CLI harness:
    - monomial composition via `HamiltonianModel` with `gauge` + fermion monomials
    - fermion monomial kinds: `unpreconditioned`, `eo_preconditioned`
    - pseudofermion refresh modes: `heatbath`, `ou`
    - pseudofermion force modes: `autodiff`, `analytic`
    - solver modularity: `cg`, `bicgstab`, `gmres`; forms `normal`, `split`, `eo_split`
    - EO Schur operators and EO-preconditioned action/force path
    - SU3-parity CLI test set available for SU2:
      `gamma,adjoint,normal,perf,eo,hamiltonian,pfrefresh,pfid,gauge,conventions,forcecmp,eopforce`
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
- Unified control-file MCMC driver:
  - `python scripts/mcmc/mcmc.py --write-template control.toml`
  - `python scripts/mcmc/mcmc.py --config control.toml`
- SU3 Wilson Nf=2 production:
  - `python scripts/su3_wilson_nf2/hmc_su3_wilson_nf2.py --shape 4,4,4,8 --integrator minnorm2 --nmd 8 --tau 1.0 --mass 0.05 --beta 5.8`
  - Chroma-like fermion APBC in time:
    - `--fermion-bc 1,1,1,-1` (or alias `--fermion-bc antiperiodic-t`)
  - two-stage warmup is supported in HMC:
    - `--warmup-no-ar <N0>`: initial trajectories with Metropolis disabled
    - `--warmup <N1>`: subsequent warmup trajectories with Metropolis enabled (and optional `nmd` adaptation)
    - measurement then runs with standard Metropolis as before
  - defaults now target production throughput:
    - `fermion_monomial_kind=eo_preconditioned`
    - `pf_force_mode=analytic`
  - can initialize from Chroma LIME files (no resume):
    - `--init-cfg-lime <cfg.lime>`
    - `--init-mom-lime <mom.lime>` (SMD/GHMC only)
    - `--init-pf-lime <pf.lime>`
    - pseudofermion leaf selection: `--init-pf-field-index -1` (auto, default)
    - EO checkerboard convention fix for imported pseudofermions:
      - `--init-pf-cb-fix auto` (default; applies `x` shift by +1 if needed)
      - `--init-pf-cb-fix shiftx1|none`
    - use loaded pseudofermion for first trajectory:
      - `--init-use-loaded-pf-first-traj` (default true)
  - explicit monomial timing profile (force/action):
    - `--profile-monomials` enables per-monomial timing output (default off for performance runs)
  - explicit HMC loop timing profile:
    - `--profile-hmc-components` reports per-trajectory timing for `refresh`, `kinetic`, `action`, `integrate`, and residual `other`
    - `--profile-hmc-every N` controls per-trajectory print frequency
- SU2 Wilson Nf=2 full harness checks:
  - `python -m jaxqft.models.su2_wilson_nf2 --tests all --shape 4,4,4,8`
  - `python -m jaxqft.models.su2_wilson_nf2 --tests forcecmp --shape 4,4,4,8 --pf-force-mode analytic`
  - `python -m jaxqft.models.su2_wilson_nf2 --tests eopforce --shape 4,4,4,8 --fermion-monomial-kind eo_preconditioned --pf-force-mode analytic`
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
  - EO-preconditioned fixed-iteration solve benchmark:
    - `python -m jaxqft.models.su3_wilson_nf2 --tests eoperf --shape 8,8,8,16 --solver-kind cg --solver-form normal --eop-perf-maxiter 86 --eop-perf-tol 0.0 --eop-perf-repeat 4 --eop-perf-warmup 1`
    - optional half-spinor toggle: `--no-eo-use-half-spinor` (or benchmark both via default `--eop-perf-compare-halfspinor`)
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
   - includes U1/SU2/SU3 eager-vs-jit with solver variants
   - reports per-row `D/N` timing, solver timing, `solve/N` operator-apply equivalent, and `D/N` GFLOP/s
   - optional raw output: `--json-out wilson_matrix.json`

### JIT Regime Note
- Earlier “optimization had little effect” conclusions for Wilson fermion kernels were confounded by eager-dispatch overhead.
- With default JIT on:
  - U1 `8x8` D kernel optimized/reference ratio ~`1e-2` in `--tests perf`.
  - U1 `256x256` D kernel optimized/reference ratio ~`1.3e-1`.
  - SU3 `4^3x8` D kernel optimized/reference ratio ~`2.7e-1`.
- Re-evaluate performance decisions in this JIT regime before pruning kernel variants.

### SU3 Wilson vs Chroma Snapshot (CPU, weak field, 20 threads)
- Chroma reference files:
  - `/Users/kostas/Work/qcd_codes/chromaform/runs/test_perf/unprec_wilson.out`
  - `/Users/kostas/Work/qcd_codes/chromaform/runs/test_perf/prec_wilson.out`
- Matched JAX command baseline:
  - `JAX_PLATFORMS=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=20 --xla_cpu_use_onednn=true' python -m jaxqft.models.su3_wilson_nf2 --tests perf --shape 8,8,8,16 --mass 0.05 --n-iter-timing 10`
- Current measured points:
  - SU3 `D` matvec (optimized): ~`0.85 ms` per call on `8^3x16`
  - SU3 `DdagD` matvec (optimized): ~`1.64 ms` per call on `8^3x16` after split-normal path update
  - Fixed-iteration CG unpreconditioned (`243` iters, `tol=0`): ~`1.44-1.46 s` total (`~5.93-6.00 ms/iter`)
  - Fixed-iteration CG EO-preconditioned Schur-normal (`86` iters, `tol=0`):
    - with EO half-spinor compact kernel: ~`0.40 s` total (`~4.63 ms/iter`)
    - without EO half-spinor compact kernel: ~`0.74 s` total (`~8.65 ms/iter`)
  - Chroma `hmc.out` (provided run, early weak-field phase):
    - first `wilson_two_flav` force call: `0.667797 s`
    - calls `2..11` mean: `0.524933 s`
  - JAX EO-preconditioned analytic force (matched `8^3x16`, `beta=5.7`, `mass=0.05`, `cg tol=1e-7`):
    - `python -m jaxqft.models.su3_wilson_nf2 --tests eopforce --shape 8,8,8,16 --beta 5.7 --mass 0.05 --fermion-monomial-kind eo_preconditioned --solver-kind cg --solver-form normal --solver-tol 1e-7 --solver-maxiter 1000 --pf-force-mode analytic --pf-fd-trials 1 --pf-fd-iters 4`
    - steady analytic force timing: ~`6.82e-02 s/call` (after warm-start path is active)
  - One-trajectory matched-leapfrog probe (`tau=0.5`, `nmd=25`) gave:
    - force time ~`0.375 s/call` averaged over force calls in-trajectory
- Relative to the provided Chroma logs:
  - unpreconditioned iteration time is currently faster in JAX
  - EO-preconditioned force path is now faster than the early Chroma weak-field force-time window
 - Important implementation note:
  - `SU3WilsonNf2.apply_normal` now uses split calls `Ddag(D(psi))` by default (with jitted `D`/`Ddag` kernels),
    and `apply_normal` is no longer separately jitted as one monolithic kernel.
  - This removed a major normal-kernel overhead while preserving solver behavior/accuracy.
  - `SU3WilsonNf2` now supports EO compact half-spinor acceleration:
    - dataclass/CLI flag: `eo_use_half_spinor` / `--eo-use-half-spinor` (default `True`)
    - this path is used only in compact EO Schur dslash kernels (`_apply_dslash_parity_compact_one`)
    - for `8^3x16`, this gave ~`1.87x` speedup for EO-preconditioned solve wall-time at fixed 86 iterations.

### SU2 Wilson Status
- SU2 Wilson Nf=2 is feature-parity with SU3/U1 Wilson module architecture.
- Validated on CPU in both monomial modes:
  - unpreconditioned: `--tests all --fermion-monomial-kind unpreconditioned`
  - EO-preconditioned: `--tests all --fermion-monomial-kind eo_preconditioned --pf-force-mode analytic`

## SAD (Stochastic Automatic Differentiation)
- Reference: Catumba & Ramos, arXiv:2502.15570 (2025).
- Status: **Implemented and validated** for phi^4 in 2D.
- Eliminates exponential signal-to-noise degradation in correlator measurements by reformulating the connected two-point function C(t) as a one-point function of a tangent field, computed via forward-mode AD (`jax.jvp`) through the SMD integrator.

### Key files:
- `jaxqft/models/phi4_sad.py`: Core library — `action_J`, `force_J`, `leapfrog_scan`, `smd_sad_step`, `free_propagator`.
- `scripts/phi4/sad_phi4.py`: Measurement driver — batched chains, warmup, measurement, jackknife statistics, cosh m_eff, JSON output.
- `scripts/phi4/plot_sad.py`: Plotting — correlator, effective mass, signal-to-noise, error bar comparison. Reads JSON from `sad_phi4.py`.
- `docs/phi4_sad/`: Results directory — LaTeX note (`sad_note.tex`/`.pdf`), plots (`.png`), data (`.json`).

### Architecture:
- `smd_sad_step(phi, pi, phi_tan, pi_tan, key, ...)` takes and returns accumulated tangent state.
- The tangent pair `(phi_tan, pi_tan)` is initialized to zeros and propagated through every SMD step (warmup + measurement).
- At each step: OU refresh gives `pi_ou_tan = c1 * pi_tan`, then `jax.jvp` through `leapfrog_scan` propagates `(phi_tan, pi_ou_tan, 1.0)`.
- The J-tangent `1.0` injects a fresh source at t=0 every step; OU damping `c1 < 1` ensures convergence.
- Observable: `C_sad(t) = mean_x(phi_tan(x,t))` — the spatial mean of the tangent field.
- Cost: ~2x a standard SMD step (forward-mode AD doubles the leapfrog work).
- Batching: `jax.vmap` over independent chains, compiled with `jax.jit`.

### Validated results:
- Free field (m2=1.0, 16x16): SAD matches exact propagator to machine precision.
- phi^4 (m2=-0.40, lam=2.4, 16x16): 3-6x noise reduction across timeslices.
- phi^4 (m2=-0.40, lam=2.4, 64x32): Up to 950x noise reduction at t=T/2, ~450,000x effective speedup.
- phi^4 (m2=-0.55, lam=2.4, 256x128): Peak R~12,000 at t~120.
- phi^4 (m2=-0.55, lam=2.4, 512x256, B=32, N=40000): Peak R~67,000 at t~252, m_eff = 0.090 ± 0.001 for t=3-20.

### Known pitfalls (see AGENTS.md for details):
1. Tangent must accumulate across full Markov chain (not per-trajectory).
2. Leapfrog must use `jax.lax.scan` (not Python for-loop) for efficient JVP.
3. Source sign convention: `S = S - J * sum(phi[0])` for positive C(t).
4. Connected subtraction uses per-chain phi_bar, not global mean.
5. Cosh m_eff solver overflows for large T — use log-space bisection (fixed in `cosh_meff_solve`).

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

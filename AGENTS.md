# AGENTS

This file is the persistent development contract for this repository.

## Scope
- Repository: `jaxQFT`
- Domain: lattice QFT in JAX (pure gauge + Wilson fermion building blocks + flows).

## Design Rules
- Keep generic algorithms in `jaxqft/core`.
- Keep model-specific code in `jaxqft/models`.
- Keep Lie-group utilities in `jaxqft/LieGroups`.
- Keep runnable entry points under `scripts/<model>/`.
- Avoid compatibility wrappers unless explicitly requested.
- Prefer model CLIs as test harnesses (`--tests ...`, `--selfcheck`).
- For fermions, build the total action from explicit monomials:
  - gauge monomials
  - pseudofermion/rational monomials
  - future preconditioning monomials (Hasenbusch, RHMC terms)
- Keep pseudofermion fields owned by monomials (not by update algorithms).
- Pseudofermion refresh policy belongs to the monomial:
  - `heatbath` (independent redraw, equivalent to c1=0)
  - `ou` (partial OU rotation on underlying Gaussian field, then mapped through Dirac operator)
  - if pseudofermion OU gamma is not explicitly provided, default it to the SMD/GHMC `gamma`

## Performance Rules
- Preserve optimized kernels and reference kernels side-by-side for correctness checks.
- Any optimization touching force/action/evolve must keep:
  - force-action consistency tests (`fd`, `autodiff`)
  - gauge invariance/covariance checks where relevant
  - epsilon scaling checks (`eps2`, `eps4`)
- Benchmark with realistic lattices (8^4 and larger), not only 4^4.

## Update Algorithm Rules
- Production scripts must support update selection via:
  - `--update {hmc,smd,ghmc}` (default `hmc` for new runs)
- Checkpoint resume must preserve updater state:
  - `update_key`
  - updater momentum when applicable (`update_momentum`)
- Legacy HMC checkpoints must remain loadable.
- Monomial-based theories with stochastic internal fields should expose `prepare_trajectory(U, traj_length)`.
- Update classes should call `prepare_trajectory(...)` once per trajectory when theory declares `requires_trajectory_refresh=True`.
- `prepare_trajectory(U, traj_length)` should pass trajectory length to stochastic monomials so OU coefficients can be computed consistently.

## Testing Expectations
- For gauge models (SU3/SU2/U1), keep CLI tests for:
  - `layout`, `timing`, `fd`, `autodiff`, `eps2`, `eps4`, `gauge`, `forcecov`, `forceimpl`, `selfcheck`
- `selfcheck --selfcheck-fail` should be CI-friendly (nonzero on failure).

## Public Repo Hygiene
- Safe to keep public:
  - source code
  - scripts
  - benchmark summaries
  - documentation
- Do not commit:
  - secrets/tokens/credentials
  - private cluster hostnames or internal paths
  - large run artifacts/checkpoints unless intentionally publishing data
- Keep `.gitignore` updated for caches and large runtime outputs.

## Public-Safe Template
Use this template for issue updates, commit notes, or handoff snippets that will be public:

```md
### Public Update
- Scope: <what changed>
- Why: <problem addressed>
- Validation: <tests/bench commands + summary metrics>
- Artifacts: <public files only; no private paths or credentials>
- Next: <short actionable follow-up>
```

## Collaboration Handoff
- Maintain `HANDOFF.md` as the first resume target.
- For each major change:
  - update `HANDOFF.md` status/backlog/commands
  - include reproducible command lines and key metrics
  - note any changed defaults or checkpoint schema changes

## SAD (Stochastic Automatic Differentiation) Rules
- Reference: Catumba & Ramos, arXiv:2502.15570 (2025).
- SAD library code lives in `jaxqft/models/phi4_sad.py`.
- SAD measurement script: `scripts/phi4/sad_phi4.py`.
- SAD plotting script: `scripts/phi4/plot_sad.py`.
- Results and LaTeX note: `docs/phi4_sad/`.

### Critical Implementation Rules
1. **Tangent propagation through the Markov chain**: The tangent `(phi_tan, pi_tan)` MUST be accumulated across ALL SMD steps. Never differentiate a single leapfrog trajectory in isolation — the tangent must carry chain-rule information from all prior steps.
   - WRONG: `jax.jvp(lambda J: leapfrog(phi, pi_ou, J), (0.0,), (1.0,))` — treats phi, pi_ou as constants.
   - RIGHT: `jax.jvp(lambda phi, pi, J: leapfrog(phi, pi, J), (phi, pi_ou, 0.0), (phi_tan, pi_ou_tan, 1.0))` — propagates accumulated tangent.
2. **OU momentum tangent**: `pi_ou_tan = c1 * pi_tan` (noise is J-independent, so its tangent is zero).
3. **Leapfrog MUST use `jax.lax.scan`**: Python for-loops cause `jax.jvp` to unroll the full computation graph, leading to catastrophically slow compilation or OOM. `lax.scan` differentiates the loop body once.
4. **Source convention**: `S_J = S - J * sum(phi[0])` (minus sign) gives positive correlator `C(t) = d/dJ <phi(t)>`.
5. **Connected subtraction for standard estimator**: Use per-chain `phi_bar` means, not the global mean: `C_std -= L * phi_bar_mean * phi_bar_mean[:, 0:1]`.
6. **Cosh effective mass** for periodic correlators: solve `cosh(m(t-T/2)) / cosh(m(t+1-T/2)) = C(t)/C(t+1)` via bisection, not log-ratio.

### Testing Expectations
- Free-field test (`lam=0, m2>0`): SAD must reproduce the analytic propagator `C(t) = Re[ifft(1/D)]` to machine precision (zero variance across chains).
- Interacting phi^4: SAD and standard estimators should agree within errors, with SAD showing 3-1000x noise reduction depending on timeslice.
- Run command examples:
  - Free field: `python scripts/phi4/sad_phi4.py --shape 16,16 --m2 1.0 --lam 0.0 --json-out docs/phi4_sad/sad_free.json`
  - phi^4 16x16: `python scripts/phi4/sad_phi4.py --shape 16,16 --m2 -0.40 --lam 2.4 --nwarm 1000 --nmeas 5000 --batch-size 8 --json-out docs/phi4_sad/sad_phi4.json`
  - phi^4 64x32: `python scripts/phi4/sad_phi4.py --shape 64,32 --m2 -0.40 --lam 2.4 --nwarm 1000 --nmeas 5000 --batch-size 8 --json-out docs/phi4_sad/sad_phi4_64x32.json`
  - Plotting: `python scripts/phi4/plot_sad.py --free-json docs/phi4_sad/sad_free.json --phi4-json docs/phi4_sad/sad_phi4.json docs/phi4_sad/sad_phi4_64x32.json --outdir docs/phi4_sad/`

## Fermion Roadmap (Current)
1. SU3 Wilson Nf=2 refactor to monomial-based Hamiltonian model.
2. Performance optimization until we get performace close to 7.3s/traj (Nf=2 EO prec. kappa=0.11 beta=5.7 8^3x16 kappa=0.11-> m_0 = 0.54545)
   Algorithm CG inverter, Force Gradient Integrator NMD=20 with a single time scale.
   Fermion force 0.093s Gauge force 0.013s)   CG iterations ~ 13
3. Add nested-timescale schedule object for monomial groups.
4. Add Hasenbusch monomials.
5. Add RHMC monomials.
6. Add optional QUDA backend under a solver/operator interface (keep JAX-native default).


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

## Fermion Roadmap (Current)
1. SU3 Wilson Nf=2 refactor to monomial-based Hamiltonian model.
2. Performance optimization until we get performace close to 7.3s/traj (Nf=2 EO prec. kappa=0.11 beta=5.7 8^3x16)
   Algorithm CG inverter, Force Gradient Integrator NMD=20 with a single time scale.
   Fermion force 0.093s Gauge force 0.013s)   CG iterations ~ 13
3. Add nested-timescale schedule object for monomial groups.
4. Add Hasenbusch monomials.
5. Add RHMC monomials.
6. Add optional QUDA backend under a solver/operator interface (keep JAX-native default).


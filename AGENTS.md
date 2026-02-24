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

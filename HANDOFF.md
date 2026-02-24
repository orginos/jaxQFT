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
- SU3 benchmark:
  - `python scripts/su3_ym/bench_hmc_su3.py --shape 16,16,16,16 --integrator forcegrad --nmd 12 --tau 1.0`
- SU2/U1 selfcheck:
  - `python -m jaxqft.models.su2_ym --selfcheck --selfcheck-fail`
  - `python -m jaxqft.models.u1_ym --selfcheck --selfcheck-fail`
- Autodiff force check (example):
  - `python -m jaxqft.models.su3_ym --tests autodiff --shape 4,4,4,8 --n-iter-timing 10`

## Priority Backlog
1. Multi-device/domain decomposition across spacetime dimensions (not batch-only).
2. More kernel-level optimization for dominant drift/force paths per backend.
3. Tighten Wilson-fermion validation and production-grade solver workflow.
4. Keep benchmark baselines per model/lattice/backend in versioned docs.

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

# Phi4 Multi-GPU Plan

Last updated: 2026-04-07

## Scope

This note records the current plan for extending the `phi^4` RG coarse-eta Gaussian
training workflow to multiple GPUs on a single node.

The intent is to resume this work later without having to reconstruct the
decision process from terminal logs.

## Current Status

The current production trainer is:

- `scripts/phi4/train_rg_coarse_eta_gaussian_flow.py`

It is a single-device trainer:

- the training step is a plain `jax.jit`
- there is no `pmap`, `pjit`, `NamedSharding`, or `shard_map` path in the
  current `phi^4` Gaussian-flow training stack

The current NERSC launcher is:

- `scripts/phi4/run_rg_coarse_eta_gaussian.sh`

## Observed Single-GPU Limits

### `L=64`

Current `64^2` production card:

- `configs/phi4/rg_coarse_eta_gaussian_L64_perlevel.toml`

Observed behavior on one A100:

- runs through lower batch stages
- OOM at batch `512`

Implication:

- the practical single-GPU ceiling for the current `L64` setup is below `512`
- a resumed one-GPU schedule should likely top out at `256`

### `L=128`

Current `128^2` exploratory card:

- `configs/phi4/rg_coarse_eta_gaussian_L128_perlevel.toml`

Observed behavior on one A100:

- runs successfully through batch `64`
- fails at batch `128`

Important compiler warning at the failed `128` stage:

- XLA reported that rematerialization could not reduce the step memory below
  `28.13 GiB`
- the best it could do was approximately `59.55 GiB`
- the unreduced step wanted approximately `70.47 GiB`

Implication:

- `128^2` is viable on one GPU
- but the current single-GPU `128^2` schedule should top out at `64`
- trying to force batch `128` on one GPU is not useful

## Main Conclusion

Multi-GPU work is not required to make `128^2` possible at all.

However, multi-GPU batch parallelism is still worthwhile because it should:

- raise the maximum feasible global batch
- reduce wall time for larger lattices
- avoid getting stuck at the one-GPU memory wall during late-stage training

## Agreed Design Constraint

Do **not** modify the current single-GPU trainer as the primary development
path.

Instead:

- keep `scripts/phi4/train_rg_coarse_eta_gaussian_flow.py` intact
- add a separate multi-GPU trainer script
- keep the core model implementation unchanged unless strictly necessary

This keeps the current single-GPU workflow stable while enabling a clean
experimental multi-GPU path.

## Proposed Implementation Path

### Phase 1: Batch data parallelism only

Create a new trainer, for example:

- `scripts/phi4/train_rg_coarse_eta_gaussian_flow_dp.py`

Design goals:

- shard the **batch** across local GPUs on one node
- replicate parameters and optimizer state across devices
- compute local losses and local gradients on each GPU
- average gradients and scalar diagnostics across devices
- keep the model code unchanged:
  - `jaxqft/models/phi4_rg_coarse_eta_gaussian_flow.py`
  - `jaxqft/models/phi4.py`

Why this is the right first step:

- the model already uses batch-first arrays of shape `(batch, H, W)`
- `Phi4.action()` is per-sample and reduces only over spatial axes
- this is the least invasive way to get a `4x` reduction in per-GPU local batch

Recommended first implementation style:

- use `pmap` first for a bounded implementation
- move to `shard_map` / explicit sharding later only if needed

Rationale:

- `pmap` is enough for straightforward single-node data parallelism
- this minimizes engineering risk for the first version

### Phase 2: Preserve checkpoint compatibility

The multi-GPU trainer should save the same logical checkpoint payload shape as
the current single-GPU trainer:

- plain NumPy weights
- plain NumPy optimizer state
- same metadata fields

That means:

- replicate during training
- unreplicate before writing the `.pkl`

Goal:

- existing analysis scripts should keep working
- laptop/macOS checkpoint loading should keep working
- resume paths remain familiar

### Phase 3: Optional memory follow-up

If data parallelism is still not enough for later targets:

1. add explicit `jax.checkpoint` / `jax.remat` in the new multi-GPU trainer or
   in the model call path
2. only after that, consider true spatial sharding of the lattice

Spatial sharding is a much larger project because the RG blocking hierarchy
operates on `2x2` structure and would need careful cross-device layout choices.

## Decision Boundary

### If the goal is only to finish current `128^2` science runs

Do not start multi-GPU work yet.

Use single-GPU resume cards that:

- ramp only through `64`
- anneal at `64`

### If the goal is to push `128^2` to larger effective batch or improve wall time

Start the batch-parallel multi-GPU trainer.

This is now justified by the observed one-GPU `128` memory wall.

## Concrete Next Steps

1. Add `scripts/phi4/train_rg_coarse_eta_gaussian_flow_dp.py`.
2. Keep the current trainer untouched.
3. Reuse the current config schema where possible.
4. Add a separate launcher mode or wrapper for the DP trainer.
5. Preserve the current checkpoint schema on save.
6. Smoke-test on `4` local GPUs with `L=32`.
7. Benchmark `L=64` and `L=128` with global batch held fixed and then increased.

## Suggested Smoke-Test Goal

First target:

- `4` GPUs on one node
- `L=32`
- small global batch divisible by `4`
- verify:
  - loss matches the single-GPU trainer within expected numerical tolerance
  - checkpoints can be loaded by existing scripts
  - resume works

Only after that:

- try `L=64`
- then `L=128`

## Notes For Tomorrow

- The multi-GPU project is an optimization path, not an emergency rescue path.
- The main near-term value is larger stable global batch and shorter wall time.
- Keep the current single-GPU workflow usable while the DP trainer is being developed.

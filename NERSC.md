# NERSC Playbook

This file is for future Codex sessions running inside the private research clone
of this repository on NERSC Perlmutter.

It is not a general project overview. It is the operational guide for getting
productive quickly on A100 hardware without repeating setup mistakes.

## Read This First

On a fresh NERSC session, read these files in this order:

1. `HANDOFF.md`
2. `NERSC.md`
3. `AGENTS.md`
4. `docs/papers/paper-1/manuscript.tex`
5. `docs/papers/paper-1/experiment_plan.md`

`HANDOFF.md` remains the main project resume log. `NERSC.md` is only the
machine/workflow supplement.


## Ground Rules

- Use the **private research clone**, not the public repo, for active phi4 flow
  and RG work.
- Use **Codex CLI on the NERSC login node** for editing, log inspection, job
  preparation, and analysis.
- Use **Slurm jobs on GPU nodes** for JAX training and GPU validation.
- Do **not** run long JAX training on login nodes.
- Do **not** put API keys or tokens into tracked files or Slurm scripts.
- Do **not** assume the macOS CPU environment from the local machine applies on
  Perlmutter.


## What Codex Should Do On NERSC

Codex is most useful on NERSC for:

- building or repairing the JAX GPU environment
- writing and updating Slurm job scripts
- checking logs and checkpoints
- generating/adjusting TOML run cards
- postprocessing results into tables and figures
- patching code after GPU-specific failures appear

Codex should **not** be used as the long-running process inside the training
job itself.


## Codex CLI Setup

The practical mode on NERSC is the terminal CLI.

### Install

If Codex is not already installed:

```bash
npm i -g @openai/codex
```

### Login

Prefer one of:

```bash
codex login --device-auth
```

or

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
```

Do not hardcode the API key in shell scripts.

### Recommended Usage

Use `tmux`:

```bash
tmux new -s codex
cd /path/to/private/jaxQFT
codex --full-auto
```

For scripted one-shot tasks:

```bash
codex exec -C /path/to/private/jaxQFT "Inspect the latest Slurm logs and update the L64 config if batch 512 OOMs."
```


## JAX On A100: First Priority

Future Codex should assume that the GPU environment is **not** ready until all
of the checks below pass.

### Required Questions To Answer

1. Which Python environment is active?
2. Which CUDA/toolchain stack is provided by NERSC for the current session?
3. Does `jax` import successfully?
4. Does `jaxlib` see the A100 GPU?
5. Does a small phi4 training step run on GPU without OOM or backend errors?

### Minimum Validation Sequence

#### 1. Basic JAX import

Run on login node or interactive node:

```bash
python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

#### 2. GPU check on an interactive GPU allocation

Use `salloc` or a short `sbatch` smoke job. The output must show GPU devices,
not CPU only.

#### 3. Small phi4 smoke train

Run a tiny single-GPU smoke test before any real production job:

```bash
python scripts/phi4/train_rg_coarse_eta_gaussian_flow.py \
  --L 16 --mass -0.4 --lam 2.4 \
  --width 32 \
  --n-cycles 1 \
  --radius 1 \
  --eta-gaussian level \
  --terminal-prior learned \
  --batch 8 \
  --lr 3e-4 \
  --epochs 2 \
  --validate
```

If this fails, do not launch production runs.


## Recommended Workflow On Perlmutter

### Login node

Use the login node for:

- editing files
- running Codex
- preparing environments
- submitting jobs
- checking logs
- doing very small CPU-side checks

### GPU node

Use GPU nodes for:

- JAX GPU validation
- production training
- larger analysis jobs if they actually need GPU


## Slurm Strategy

The current trainer supports staged schedules in TOML, stage-boundary saving,
and stage-boundary validation.

That means the safest production strategy is:

- one checkpoint file per model
- one Slurm job per stage or per short group of stages
- resume from the checkpoint file

This is more robust than trying to fit a full long schedule into one very long
batch job.

### Why

- later stages at large batch sizes can be much slower
- walltime estimates become uncertain
- stage checkpoints are already available every `1000` epochs
- validation metrics are printed at stage boundaries


## Current Phi4 Priorities

### Main model family

Use:

- `jaxqft/models/phi4_rg_coarse_eta_gaussian_flow.py`
- `scripts/phi4/train_rg_coarse_eta_gaussian_flow.py`
- `scripts/phi4/analyze_rg_coarse_eta_gaussian_flow.py`

This is the current best branch for the short paper and the RG follow-up.

### Current paper workspace

- `docs/papers/paper-1/manuscript.tex`
- `docs/papers/paper-1/experiment_plan.md`

### Current main configs

- `configs/phi4/rg_coarse_eta_gaussian_L32_perlevel.toml`
- `configs/phi4/rg_coarse_eta_gaussian_L64_perlevel.toml`
- `configs/phi4/rg_coarse_eta_gaussian_L128_perlevel.toml`


## Current Scientific Priorities

### Paper 1

Primary goal:

- establish the Wilsonian transport-map formulation
- show learned Gaussian priors help
- show per-level capacity helps
- show the residual mismatch scales extensively, i.e. roughly constant
  `std(ΔS) / L`

Needed runs are listed in:

- `docs/papers/paper-1/experiment_plan.md`

### Paper 2 / RG program

Longer-term goal:

- treat the learned model as an exact blocked hierarchy for the learned action
- sample coarse levels directly
- study RG trajectories in observable space and later in projected coupling
  space

This is more important scientifically than squeezing a tiny additional ESS gain
out of paper 1.


## Current Performance Picture

Known reference point at the canonical target:

- `m^2 = -0.4`
- `lambda = 2.4`

Best current `16^2` result:

- per-level Gaussian coarse-eta model
- `std(ΔS) = 0.7869`
- `ESS = 0.5697`

Current `32^2` run:

- same model family, direct larger-volume extension
- at the `512` batch stage:
  - `std(ΔS) = 1.4740`
  - `ESS = 0.2213`

The important scaling variable is:

- `std(ΔS) / L`

The current interpretation is that the residual mismatch is behaving like a
small local extensive sum, not like catastrophic model failure.


## What Future Codex Should Do First On NERSC

1. Verify JAX on A100 works.
2. Run one tiny phi4 smoke job on GPU.
3. Make or update Slurm wrappers for the `L=32`, `L=64`, and `L=128` TOML
   schedules.
4. Confirm stage-boundary saving and validation are visible in logs.
5. Keep a clean table of:
   - stage
   - batch
   - learning rate
   - `std(ΔS)`
   - `std(w)`
   - `ESS`
   - wall-clock time


## Expected Failure Modes

Future Codex should actively check for:

- JAX falling back to CPU silently
- CUDA/JAX version mismatch
- OOM at larger batches
- very slow first-step compile masking later throughput
- Slurm jobs dying before stage checkpoint save
- validation PDFs/logs being written in unexpected working directories


## Recovery Rules

If a production run fails:

1. find the last saved checkpoint
2. run `--validate --validate-only` on it
3. determine whether the failure is:
   - environment
   - OOM
   - walltime
   - numerical instability
4. patch the TOML or Slurm script, not the scientific model, unless there is a
   real model bug


## Do Not Forget

- The current short-paper path and the larger RG paper path are different.
- Do not let a large architecture sweep consume GPU budget meant for scaling
  studies.
- The `32^2`, `64^2`, and `128^2` runs are valuable even when fine-level ESS is
  not spectacular, because the RG interpretation is now central.

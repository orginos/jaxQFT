# Paper 2 Canonical Scaling Baseline

This directory defines the fixed-architecture baseline for the RG paper.

Scientific purpose:
- hold the model architecture fixed across volumes
- hold the staged training schedule fixed across volumes
- measure how the intensive quality metric `std(Delta S) / L` scales with `L`
- only after this baseline is established, inspect bottlenecks and introduce
  volume-dependent tuning

Baseline model:
- Wilsonian Gaussian coarse-eta flow
- uniform across all nonterminal RG levels
- `width = 64`
- `n_cycles = 2`
- `radius = 1`
- `eta_gaussian = "level"`
- `gaussian_width = 64`
- `gaussian_radius = 1`
- `terminal_prior = "learned"`
- `terminal_n_layers = 2`
- `terminal_width = 64`

Baseline schedule:
- `1000` epochs at batch `16`, lr `3e-4`
- `1000` epochs at batch `32`, lr `3e-4`
- `1000` epochs at batch `64`, lr `3e-4`
- `2000` epochs at batch `64`, lr `1e-4`
- `2000` epochs at batch `64`, lr `3e-5`
- `4000` epochs at batch `64`, lr `1e-5`

This schedule is intentionally the same for `L = 16, 32, 64, 128`.

## Config files

- `L16_uniform.toml`
- `L32_uniform.toml`
- `L64_uniform.toml`
- `L128_uniform.toml`

## Checkpoint layout

Committed directories exist under:

- `ckpts/phi4/paper-2/canonical-scaling/L16/`
- `ckpts/phi4/paper-2/canonical-scaling/L32/`
- `ckpts/phi4/paper-2/canonical-scaling/L64/`
- `ckpts/phi4/paper-2/canonical-scaling/L128/`

The TOML files default to seed `0`. For production runs with multiple seeds,
override both `--seed` and `--save`.

Example:

```bash
scripts/phi4/run_rg_coarse_eta_gaussian.sh \
  --config configs/phi4/paper-2/canonical-scaling/L128_uniform.toml \
  --workdir runs/phi4/paper-2/canonical-scaling/L128/s7 \
  --gpu all \
  -- \
  --seed 7 \
  --save checkpoint.pkl
```

Resume:

```bash
scripts/phi4/run_rg_coarse_eta_gaussian.sh \
  --config configs/phi4/paper-2/canonical-scaling/L128_uniform.toml \
  --workdir runs/phi4/paper-2/canonical-scaling/L128/s7 \
  --gpu all \
  -- \
  --resume checkpoint.pkl
```

Checkpoint-only validation:

```bash
scripts/phi4/run_rg_coarse_eta_gaussian.sh \
  --config configs/phi4/paper-2/canonical-scaling/L128_uniform.toml \
  --workdir runs/phi4/paper-2/canonical-scaling/L128/s7 \
  --gpu all \
  -- \
  --resume checkpoint.pkl \
  --validate \
  --validate-only
```

For Perlmutter batch launch, use:

```bash
sbatch scripts/phi4/rg_coarse_eta_gaussian_4seed_perlmutter.slurm
```

after editing the variables at the top of the Slurm script.

For arbitrary regular-qos bundles of `4N` independent flow tasks on `N` nodes,
use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh \
  --tasks /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/my_bundle/tasks_in.tsv \
  --bundle-root /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/my_bundle \
  --time 08:00:00
```

The manifest is tab-separated with a nonzero multiple of 4 non-comment rows.
The submit helper computes `nodes = task_count / 4`. The first four columns are:

- `task_name`
- `config`
- `run_dir`
- `seed`

Any additional tab-separated columns are passed verbatim to the trainer after
`--save checkpoint.pkl`. This is where point-specific overrides such as
`--lam`, `--mass`, and `--width` belong.

## High-batch continuations

Continuation cards that keep the architecture fixed and only extend the
schedule live under:

- `configs/phi4/paper-2/canonical-scaling-continuation/`

These are intended to test the volume-versus-batch tradeoff directly:

- `L16_uniform_continue.toml`
- `L32_uniform_continue.toml`
- `L64_uniform_continue.toml`
- `L128_uniform_continue.toml`

The intended continuation targets are:

- `L=16`: continue to batch `2048`
- `L=32`: continue to batch `1024`
- `L=64`: continue to batch `256`
- `L=128`: keep batch `64` and extend the final low-learning-rate anneal

## Canonical HMC reference runs

The existing HMC script already reports the observables needed for the canonical
reference ensemble:
- magnetization
- connected susceptibility
- second-moment correlation length
- IAT / ESS

Suggested commands:

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/hmc_phi4.py \
  --shape 16,16 --lam 2.4 --mass -0.4 \
  --nwarm 1000 --nmeas 5000 --nskip 10 --batch-size 8 \
  --json-out runs/phi4/paper-2/hmc/hmc_phi4_L16_canonical.json
```

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/hmc_phi4.py \
  --shape 32,32 --lam 2.4 --mass -0.4 \
  --nwarm 1000 --nmeas 5000 --nskip 10 --batch-size 8 \
  --json-out runs/phi4/paper-2/hmc/hmc_phi4_L32_canonical.json
```

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/hmc_phi4.py \
  --shape 64,64 --lam 2.4 --mass -0.4 \
  --nwarm 1000 --nmeas 5000 --nskip 10 --batch-size 4 \
  --json-out runs/phi4/paper-2/hmc/hmc_phi4_L64_canonical.json
```

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/hmc_phi4.py \
  --shape 128,128 --lam 2.4 --mass -0.4 \
  --nwarm 1000 --nmeas 5000 --nskip 10 --batch-size 2 \
  --json-out runs/phi4/paper-2/hmc/hmc_phi4_L128_canonical.json
```

On NERSC, the same commands can be run on GPU nodes after the JAX/CUDA
environment is validated.

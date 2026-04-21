# Canonical Point Scan

This campaign extends the original canonical `g4 = 2.4`, `g2 = -0.4` study in
two directions:

1. a cheaper uniform model with reduced transport width (`width = 48`)
2. additional physics points approaching criticality and entering the broken phase

The flow family is kept as close as possible to the original canonical baseline:

- uniform transport architecture across RG levels
- `n_cycles = 2`
- `radius = 1`
- `eta_gaussian = "level"`
- `gaussian_width = 64`
- `terminal_width = 64`
- same staged training schedule as `configs/phi4/paper-2/canonical-scaling/L*_uniform.toml`

Only two knobs are changed in this scan:

- `physics.mass` (`g2`)
- `model.width`

An additional fixed-across-volume architecture variant is now available in the
repo cards:

- `w64c3`: `width = 64`, `n_cycles = 3`, `radius = 1`

This variant is not part of the original width-only submitter. It should be
treated as a separate architecture family using the dedicated base cards:

- `configs/phi4/paper-2/canonical-scaling/L16_uniform_c3.toml`
- `configs/phi4/paper-2/canonical-scaling/L32_uniform_c3.toml`
- `configs/phi4/paper-2/canonical-scaling/L64_uniform_c3.toml`
- `configs/phi4/paper-2/canonical-scaling/L128_uniform_c3.toml`
- `configs/phi4/paper-2/canonical-point-scan/L128_uniform_c3_batch64_then_anneal.toml`

For the bundled `w64c3` campaign, use the dedicated point list:

- `configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv`

This list includes the original canonical point in addition to the shifted
near-critical and broken-phase points:

- `canonical`: `g2 = -0.4`
- `canonical2`: `g2 = -0.5`
- `canonical3`: `g2 = -0.585`
- `canonical4`: `g2 = -0.70`

## Physics Points

See `points.tsv` for the tracked point list. The current campaign uses:

- `canonical2`: `g2 = -0.5`
- `canonical3`: `g2 = -0.585`
- `canonical4`: `g2 = -0.70`

Interpretation:

- `canonical2`: closer to criticality than the original canonical point
- `canonical3`: near-critical
- `canonical4`: deep in the broken phase; `phi` histograms should show a double-peak structure

For the clean forward-loss rerun campaign, use:

- `forward_points.tsv`

This explicitly includes all four physics points:

- `canonical`: `g2 = -0.4`
- `canonical2`: `g2 = -0.5`
- `canonical3`: `g2 = -0.585`
- `canonical4`: `g2 = -0.70`

## Width Variants

- `w64`: original canonical transport width
- `w48`: cheaper width-only variant
- `w64c3`: same width as `w64`, but with one extra transport cycle per level

The Gaussian/terminal subnet widths remain at `64` so this scan isolates the
effect of changing the main transport-map capacity.

## Volumes and Seeds

The current submitter targets:

- `L = 16, 32, 64, 128`
- seeds `0,1,2,3`

If a seed fails with non-finite loss, the replacement policy is tracked in
`replacement_seeds.tsv`. The logical campaign still has four seed slots per
point/width/volume, but a failed slot can be reassigned to a fresh active seed
such as `s0 -> s4`.

## NERSC Submitter

Use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_campaign_nersc.sh
```

For the bundled `w64c3` family, use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_bundles_nersc.sh
```

For the clean forward-loss rerun campaign, use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_forward_bundles_nersc.sh
```

and for the bundled `w64c3` forward rerun:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_forward_bundles_nersc.sh
```

This submitter groups tasks by common volume and emits one regular-qos bundle
per `L`, with one task per point/seed combination.

All runtime products go under:

```text
/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan/
```

with layout

```text
<point>/<arch>/L<L>/s<seed>/
```

where `<arch>` is one of:

- `w64`
- `w48`
- `w64c3`

Each run directory contains:

- `input.toml`
- `job.sbatch`
- `checkpoint.pkl`
- `slurm/train.out`
- `slurm/train.err`

Important:

- the point-specific `mass` and `width` overrides are carried by the generated
  `job.sbatch` command line
- `input.toml` is the copied base card, not by itself a complete record of the
  runtime overrides used by the canonical submitter
- for the failed `canonical3/w64/L128` rescue, use
  `L128_uniform_batch64_then_anneal.toml`, which replaces the `16 -> 32 -> 64`
  ramp with `3000` epochs directly at batch `64`, followed by the original
  anneal stages
- for the analogous bundled `w64c3` launch, `canonical3/L128` uses
  `L128_uniform_c3_batch64_then_anneal.toml`
- the forward-loss rerun campaign is meant to be apples-to-apples against the
  original baseline, so it uses the original base cards plus explicit
  `--loss-path forward` rather than the conservative rescue cards
- the recommended runtime root for the forward rerun is:
  `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward`
  which preserves the original `canonical-point-scan` tree as the historical
  inverse-path archive

## Bundled Forward Level Analysis

To analyze the completed forward checkpoints before any refinement wave, use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_forward_level_analysis_bundles_nersc.sh
```

Default scope:

- points: `canonical`, `canonical2`, `canonical3`, `canonical4`
- arches: `w64`, `w48`
- volumes: `L = 16, 32, 64, 128`
- seeds: `0,1,2,3`
- shards: `4` independent analysis calls per checkpoint

Important:

- each shard gets its own run directory
  `.../s<seed>/shardXX/`
- this is deliberate: large aggregate statistics are accumulated through
  multiple independent model-sampling calls rather than a single oversized run
- the default output root is:
  `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-level-analysis/before_refine4096`

The analysis bundler uses the generic 4-task Perlmutter bundle helper with the
dedicated level-analysis task runner:

- `scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh`
- `scripts/phi4/rg_coarse_eta_gaussian_level_analysis_bundle_task.sh`

For the matching post-refinement analysis wave, use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_forward_level_analysis_after_refine4096_bundles_nersc.sh
```

This reads:

- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-refine4096`

and writes:

- `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-level-analysis/after_refine4096`

using `checkpoint_refine4096.pkl` as the checkpoint file name.

## Refine4096 Resume Wave

To continue the stable forward checkpoints with very large effective batch
training, use:

```bash
scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_forward_refine4096_bundles_nersc.sh
```

Default scope:

- arches: `w64`, `w48`
- points: `canonical`, `canonical2`, `canonical3`, `canonical4`
- volumes: `L = 16, 32, 64, 128`
- seeds: `0,1,2,3`

This wave:

- resumes from the existing forward checkpoints
- writes into a separate runtime root:
  `/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-refine4096`
- saves a different checkpoint file:
  `checkpoint_refine4096.pkl`
- runs for an additional `10000` epochs (`target epoch = 21000`)
- uses a fixed refinement lr `5e-6`
- reaches effective batch `4096` through gradient accumulation, not by trying to
  fit a real batch of `4096` in memory

Important:

- early `refine4096` bundles launched before the resume-target fix were no-ops
- the bad signature in `train.out` is:
  - `start_epoch: 11000`
  - `target_epochs: 11000`
  - `remaining_epochs: 0`
- the corrected behavior is:
  - `target_epochs: 21000`
  - `remaining_epochs: 10000`
  - and the refinement-card `batch` / `grad_accum_steps` / `lr` are used instead
    of the old checkpoint schedule

The volume-specific microbatch / accumulation pairs are:

- `L16`: `batch=512`, `grad_accum_steps=8`
- `L32`: `batch=256`, `grad_accum_steps=16`
- `L64`: `batch=128`, `grad_accum_steps=32`
- `L128`: `batch=64`, `grad_accum_steps=64`

The schedule-free resume cards are:

- `L16_uniform_refine4096_resume.toml`
- `L32_uniform_refine4096_resume.toml`
- `L64_uniform_refine4096_resume.toml`
- `L128_uniform_refine4096_resume.toml`

The default refinement wave intentionally excludes `w64c3` because that family
is still in repair mode at `L=128`.

## Conservative Repair Wave

The unresolved April 14, 2026 repair wave is tracked in:

- `configs/phi4/paper-2/canonical-point-scan/replacement_seeds.tsv`
- `configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/`
- `scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_conservative_repairs_nersc.sh`

The conservative rescue schedule is:

- `3000` epochs at batch `64`, lr `1e-4`
- anneal to epoch ends `5000`, `7000`, `11000`
- anneal lrs `5e-5`, `2e-5`, `1e-5`

Tracked rescue cards:

- `L32_uniform_batch64_lr1e4_then_anneal.toml`
- `L64_uniform_batch64_lr1e4_then_anneal.toml`
- `L128_uniform_batch64_lr1e4_then_anneal.toml`
- `L32_uniform_c3_batch64_lr1e4_then_anneal.toml`
- `L64_uniform_c3_batch64_lr1e4_then_anneal.toml`
- `L128_uniform_c3_batch64_lr1e4_then_anneal.toml`

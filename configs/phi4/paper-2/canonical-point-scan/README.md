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

## Physics Points

See `points.tsv` for the tracked point list. The current campaign uses:

- `canonical2`: `g2 = -0.5`
- `canonical3`: `g2 = -0.585`
- `canonical4`: `g2 = -0.70`

Interpretation:

- `canonical2`: closer to criticality than the original canonical point
- `canonical3`: near-critical
- `canonical4`: deep in the broken phase; `phi` histograms should show a double-peak structure

## Width Variants

- `w64`: original canonical transport width
- `w48`: cheaper width-only variant

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

All runtime products go under:

```text
/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan/
```

with layout

```text
<point>/w<width>/L<L>/s<seed>/
```

Each run directory contains:

- `input.toml`
- `job.sbatch`
- `checkpoint.pkl`
- `slurm/train.out`
- `slurm/train.err`

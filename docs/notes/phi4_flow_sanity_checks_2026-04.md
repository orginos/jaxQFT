# Phi4 Flow Sanity Checks (2026-04)

This note records the concrete checks that were run after the broken-phase
`L=128` difficulties, so the diagnosis does not live only in chat history.

## Scope

Questions addressed:

- Is there evidence of a global mistake in the RG coarse-eta Gaussian flow?
- Or is the observed failure localized to the hardest broken-phase point?

Short answer:

- There is **no evidence of a global flow-model failure** from the checks below.
- The problematic case is localized to the deep broken-phase, large-volume,
  deep-architecture point `canonical4 / w64c3 / L128`.

## Main conclusion

The current evidence supports the following interpretation:

1. Symmetric-phase and near-critical controls match HMC reasonably well at
   small volume.
2. Broken phase is representable at small volume (`L=16`) by the standard
   `w48` and `w64` models.
3. The serious mismatch appears only for the hard large-volume broken-phase
   `w64c3` point, where training remained fragile despite large effective
   batch.

So the correct reading is:

- the framework is not coming apart globally;
- the unresolved issue is a **specific broken-phase modeling/training problem**
  at large volume and higher depth.

## Symmetric / Near-Critical Anchors Against HMC

All comparisons below are fine-level observables at the same bare couplings,
using a locally copied flow checkpoint and a fresh local HMC run.

### `canonical3 / w64 / L16 / s0` versus HMC

Bare point:

- `L=16`
- `lambda = 2.4`
- `mass^2 = -0.585`

Observed values:

| source | xi2 | chi_m | U4 |
|---|---:|---:|---:|
| flow (raw) | 11.5701 | 87.1901 | 0.51013 |
| flow (reweighted) | 11.7319 | 92.3906 | 0.54023 |
| HMC | 11.5567 | 92.0674 | 0.54521 |

Interpretation:

- `xi2` is already essentially correct at the raw-model level.
- Raw flow samples underestimate `chi_m` and `U4`.
- Reweighting brings `chi_m` and `U4` into good agreement with HMC.

### `canonical2 / w64 / L16 / s0` versus HMC

Bare point:

- `L=16`
- `lambda = 2.4`
- `mass^2 = -0.5`

Observed values:

| source | xi2 | chi_m | U4 |
|---|---:|---:|---:|
| flow (raw) | 5.9016 | 29.0924 | 0.27722 |
| flow (reweighted) | 5.8717 | 31.5930 | 0.29119 |
| HMC | 5.8651 | 31.8406 | 0.30997 |

Interpretation:

- Same pattern as `canonical3`.
- `xi2` is already very close in the raw model.
- Reweighting moves `chi_m` and `U4` toward HMC and largely closes the gap.

## Broken-Phase Small-Volume Check: `canonical4 / L16`

This check was added after the large-volume broken-phase `w64c3` mismatch.

Bare point:

- `L=16`
- `lambda = 2.4`
- `mass^2 = -0.70`

Compared objects:

- HMC
- `canonical4 / w48 / L16 / s0`
- `canonical4 / w64 / L16 / s0`

### Fine-level observables

| source | xi2 | chi_m | U4 |
|---|---:|---:|---:|
| `w48` raw | 25.9281 | 212.5925 | 0.63675 |
| `w48` reweighted | 27.5956 | 219.2860 | 0.65174 |
| `w64` raw | 25.6474 | 213.0242 | 0.63775 |
| `w64` reweighted | 27.4638 | 218.4869 | 0.65117 |
| HMC | 26.8558 | 216.0574 | 0.64975 |

Interpretation:

- Both `w48` and `w64` are already close at the raw-model level.
- Reweighting places both almost directly on top of HMC.
- `w64` is only marginally better than `w48`.

### Histogram comparison

The `L=16` broken-phase histograms are qualitatively healthy for both `w48`
and `w64`.

Key site-field histogram summaries:

| source | peak separation | valley / peak |
|---|---:|---:|
| `w48` | 1.897 | 0.508 |
| `w64` | 1.951 | 0.459 |
| HMC | 2.107 | 0.457 |

Interpretation:

- Both models are clearly bimodal.
- `w64` tracks the HMC site histogram slightly more closely.
- Broken phase is therefore **not** generically impossible for the model
  family; it is under control at `L=16`.

## Large-Volume Broken-Phase Failure Is Localized

The problematic case is:

- `canonical4 / w64c3 / L128`

What was observed:

- `accum16 + warmup`: `1/4` successful seeds
- `accum32 + warmup`: `2/4` successful seeds
- even successful `L=128 w64c3` broken-phase seeds still showed a site-field
  histogram much more washed out than HMC

Representative comparison:

- HMC `L=128` warmed-up site histogram:
  - peaks near `phi ~ +-1`
  - valley/peak ratio about `0.52`
- best local flow probe (`s26`) at `L=128 w64c3`:
  - peaks near `phi ~ -0.10, +0.05`
  - valley/peak ratio about `0.94`

Interpretation:

- the `L=128 w64c3` broken-phase issue is real;
- but it is a **localized hard-point failure**, not evidence that the whole
  flow construction is wrong.

## Practical takeaway for paper writing

Safe claims:

- Symmetric / near-critical flow models are physically sane and validate
  against HMC on small-volume anchors.
- Broken phase is representable at small volume by `w48` and `w64`.
- The unresolved issue is the hard large-volume broken-phase deep model,
  especially `w64c3 / L128`.

Unsafe claim:

- Do **not** claim that the broken-phase `L=128 w64c3` point is solved or
  trustworthy.

## Minimal reproducibility notes

Useful local tools:

- model/HMC observable comparison:
  - `scripts/phi4/analysis/analyze_rg_coarse_eta_gaussian_levels.py`
  - `scripts/phi4/hmc_phi4.py`
- model/HMC histogram comparison:
  - `scripts/phi4/analysis/sample_rg_coarse_eta_gaussian_histograms.py`

For histogram sanity checks:

- one batch is already enough for the pooled site-field histogram at large
  volume;
- magnetization histograms need more than one batch to be meaningful.

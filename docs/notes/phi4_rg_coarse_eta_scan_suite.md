# RG Coarse-Eta Flow: Comprehensive Parameter-Scan Suite

This note defines a practical scan plan for the current preferred model:

- `jaxqft/models/phi4_rg_coarse_eta_flow.py`
- `scripts/phi4/train_rg_coarse_eta_flow.py`
- `scripts/phi4/check_rg_coarse_eta_flow.py`

The goal is to identify the best architectural settings and training schedule
for the scalar `phi^4` target ensemble

- `mass = -0.4`
- `lam = 2.4`

across volumes

- `8 x 8`
- `16 x 16`
- `32 x 32`
- `64 x 64`

The current evidence is that:

- large batch sizes are important for late-stage convergence
- gradual learning-rate reduction stabilizes the final optimization
- a staged resume schedule is the right way to train this model


## 1. Metrics To Optimize

For model selection, rank runs by:

1. centered `std(action diff)`
2. `ESS`
3. wall-clock cost per epoch
4. final loss

Use the centered width of

`DeltaS = log p_theta(phi) + S(phi)`

as the primary metric. That is the most reliable proxy for reweighting quality.


## 2. Canonical Checkpoint Naming

Use a checkpoint name that encodes only the architecture and seed, not the
training stage. Resume into the same file across the full schedule.

Recommended stem:

`coarseeta_L{L}_w{W}_nc{NC}_r{R}_tnl{TNL}_tw{TW}_par{PAR}_rg{RG}_lsc{LSC}_odc{ODC}_s{SEED}`

Examples:

- `coarseeta_L16_w64_nc2_r1_tnl2_tw64_parsym_rgavg_lsc5p0_odc2p0_s0.pkl`
- `coarseeta_L32_w32_nc3_r1_tnl1_tw32_parsym_rgavg_lsc5p0_odc1p0_s1.pkl`

Recommended directory layout:

- `ckpts/` for checkpoints
- `logs/` for stdout/stderr logs

Use the same checkpoint file for all resume stages and a new log file per
stage:

- `logs/${STEM}_e1000_b32_lr3e-4.log`
- `logs/${STEM}_e6000_b1024_lr3e-4.log`
- `logs/${STEM}_e8000_b1024_lr3e-5.log`


## 3. Standard Training Schedule

Use a staged schedule. The base schedule assumes the batch can eventually reach
`1024`.

### Schedule S0: Quick Screen

Purpose:

- cheap ranking of many architecture choices

Run:

- epochs `0 -> 1000`
- batch `32`
- lr `3e-4`

Command template:

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py \
  --L ${L} \
  --mass -0.4 \
  --lam 2.4 \
  --width ${W} \
  --n-cycles ${NC} \
  --radius ${R} \
  --terminal-n-layers ${TNL} \
  --terminal-width ${TW} \
  --parity ${PAR} \
  --rg-type ${RG} \
  --log-scale-clip ${LSC} \
  --offdiag-clip ${ODC} \
  --batch 32 \
  --lr 3e-4 \
  --epochs 1000 \
  --validate \
  --save ckpts/${STEM}.pkl
```


### Schedule S1: Batch Ramp

Purpose:

- move promising runs into the large-batch regime

Resume blocks:

- `1000 -> 2000`: batch `64`, lr `3e-4`
- `2000 -> 3000`: batch `128`, lr `3e-4`
- `3000 -> 4000`: batch `256`, lr `3e-4`
- `4000 -> 5000`: batch `512`, lr `3e-4`
- `5000 -> 6000`: batch `1024`, lr `3e-4`

Resume template:

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py \
  --resume ckpts/${STEM}.pkl \
  --batch ${BATCH} \
  --lr 3e-4 \
  --epochs ${TARGET_EPOCH} \
  --validate
```


### Schedule S2: Fine-Tune At Maximum Batch

Purpose:

- stabilize and squeeze out the final improvement

Recommended blocks after reaching `Bmax`:

- `6000 -> 7000`: batch `Bmax`, lr `1e-4`
- `7000 -> 8000`: batch `Bmax`, lr `3e-5`
- `8000 -> 10000`: batch `Bmax`, lr `1e-5`

Optional final tail if still improving:

- `10000 -> 12000`: batch `Bmax`, lr `3e-6`

Resume template:

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py \
  --resume ckpts/${STEM}.pkl \
  --batch ${BMAX} \
  --lr ${LR} \
  --epochs ${TARGET_EPOCH} \
  --validate
```


### Volume-Dependent Batch Guidance

If `1024` does not fit on the target GPU:

- use the largest affordable batch size `Bmax`
- keep doubling up to `Bmax`
- then perform the same learning-rate decay schedule at that `Bmax`

Practical expectations:

- `L = 8`: `Bmax = 1024` or larger should be easy
- `L = 16`: `Bmax = 1024` should be a standard target
- `L = 32`: try `512`, then `1024` if memory allows
- `L = 64`: use the largest affordable batch, even if it is below `1024`


## 4. Fixed Defaults For The Main Scan

Keep these fixed unless explicitly testing them:

- `mass = -0.4`
- `lam = 2.4`
- `parity = sym`
- `rg-type = average`
- `log-scale-clip = 5.0`
- `offdiag-clip = 2.0`
- `terminal-width = width`

These are control ablations, not core-scan axes:

- `parity = none`
- `rg-type = select`
- `log-scale-clip = 3.0, 7.0`
- `offdiag-clip = 1.0, 4.0`
- `terminal-width = 2 * width`


## 5. Core Architecture Axes

The main architecture knobs for this model are:

- `width`
- `n-cycles`
- `radius`
- `terminal-n-layers`

Recommended core grid:

- `width in {32, 64, 96}`
- `n-cycles in {1, 2, 3}`
- `radius in {0, 1, 2}`
- `terminal-n-layers in {1, 2}`

This gives:

- `3 * 3 * 3 * 2 = 54` core configurations per volume

If GPU budget is tight, use the reduced grid:

- `width in {32, 64}`
- `n-cycles in {1, 2, 3}`
- `radius in {0, 1}`
- `terminal-n-layers in {1, 2}`

This gives:

- `2 * 3 * 2 * 2 = 24` configurations per volume


## 6. Scan Stages

### Stage A: Sanity

For every volume:

- run `check_rg_coarse_eta_flow.py --selfcheck-fail`
- run one smoke training job at the baseline config

Baseline config:

- `width = 64`
- `n-cycles = 2`
- `radius = 1`
- `terminal-n-layers = 2`


### Stage B: L8 Pilot Sweep

Purpose:

- cheap architecture ranking
- identify obviously bad regions

Run:

- full core grid at `L = 8`
- schedule `S0`
- one seed only: `seed = 0`

Keep:

- top `12` configurations by centered `std(action diff)`
- also keep any run that is in the top `8` by `ESS`


### Stage C: L16 Main Sweep

Purpose:

- main architecture selection at the target medium volume

Run:

- the `12` survivors from `L = 8`
- plus the full `L = 16` reduced grid if GPU budget allows
- schedule `S0`
- one seed only

Then:

- keep the top `8` configurations by centered `std(action diff)`


### Stage D: L16 Ablations Around The Winners

Purpose:

- determine the best stabilizing settings

For each of the top `8` `L = 16` core winners, run the following single-knob
ablations under schedule `S0`:

1. `offdiag-clip = 1.0`
2. `offdiag-clip = 4.0`
3. `log-scale-clip = 3.0`
4. `log-scale-clip = 7.0`
5. `terminal-width = 2 * width`
6. `parity = none`
7. `rg-type = select`

This identifies:

- whether the triangular mixing strength is too weak or too strong
- whether the diagonal scale cap is limiting learning
- whether the symmetry is genuinely helping
- whether the `average` blocking choice is really better than `select`


### Stage E: L16 Long Schedule

Purpose:

- determine the truly best configurations after large-batch fine tuning

Run on the top `4` configurations from Stage D:

- schedule `S1 + S2`
- seeds `0, 1, 2`

Rank by:

- mean centered `std(action diff)` across seeds
- then mean `ESS`
- then mean wall-clock cost

This stage should produce the `L = 16` winner.


### Stage F: L32 Transfer

Purpose:

- test how well the best `L = 16` settings scale

Run on the top `4` configurations from Stage E:

- `L = 32`
- schedule `S0`, then `S1` for the top `2`
- use the largest affordable batch

If `1024` fits, use the same ramp.
If not, stop the ramp at `Bmax` and use `S2` there.


### Stage G: L64 Final Scale Test

Purpose:

- determine whether the architecture remains competitive at the largest volume

Run on the top `2` configurations from `L = 32`:

- `L = 64`
- schedule `S0`
- then `S1 + S2` only for the single best candidate

At this stage, wall-clock efficiency matters heavily, so record:

- step time
- total time to each schedule milestone
- centered `std(action diff)`
- `ESS`


## 7. Minimal Command Recipes

### Base Fresh Command

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py \
  --L ${L} \
  --mass -0.4 \
  --lam 2.4 \
  --width ${W} \
  --n-cycles ${NC} \
  --radius ${R} \
  --terminal-n-layers ${TNL} \
  --terminal-width ${TW} \
  --parity sym \
  --rg-type average \
  --log-scale-clip ${LSC} \
  --offdiag-clip ${ODC} \
  --batch 32 \
  --lr 3e-4 \
  --epochs 1000 \
  --validate \
  --save ckpts/${STEM}.pkl
```

### Resume Command

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/train_rg_coarse_eta_flow.py \
  --resume ckpts/${STEM}.pkl \
  --batch ${BATCH} \
  --lr ${LR} \
  --epochs ${TARGET_EPOCH} \
  --validate
```

### Checker Command

```bash
source /opt/python/jax/bin/activate
JAX_PLATFORMS=cpu python scripts/phi4/check_rg_coarse_eta_flow.py \
  --shape ${L},${L} \
  --jac-shape 2,2 \
  --width ${W} \
  --n-cycles ${NC} \
  --radius ${R} \
  --terminal-n-layers ${TNL} \
  --terminal-width ${TW} \
  --parity sym \
  --rg-type average \
  --log-scale-clip ${LSC} \
  --offdiag-clip ${ODC} \
  --selfcheck-fail
```


## 8. Recommended First Full GPU Scan

If you want one concrete scan to hand off immediately, use this:

### Volumes

- `L in {8, 16, 32, 64}`

### Core grid

- `width in {32, 64}`
- `n-cycles in {1, 2, 3}`
- `radius in {0, 1, 2}`
- `terminal-n-layers in {1, 2}`
- `terminal-width = width`
- `parity = sym`
- `rg-type = average`
- `log-scale-clip = 5.0`
- `offdiag-clip = 2.0`

This is:

- `2 * 3 * 3 * 2 = 36` runs per volume

### Schedule

- all 36 runs: `S0`
- top 8 at each volume: `S1`
- top 4 at each volume: `S2`
- top 2 at `L = 16` and above: repeat with seeds `1` and `2`

This is the best balance of breadth and cost for a first serious GPU campaign.


## 9. What To Expect To Matter Most

Based on current results, the most likely important knobs are:

1. `n-cycles`
2. `radius`
3. `width`

The less likely but still worth checking knobs are:

1. `terminal-n-layers`
2. `offdiag-clip`
3. `log-scale-clip`

The current default baseline to beat is:

- `width = 64`
- `n-cycles = 2`
- `radius = 1`
- `terminal-n-layers = 2`
- `terminal-width = 64`
- `parity = sym`
- `rg-type = average`
- `log-scale-clip = 5.0`
- `offdiag-clip = 2.0`


## 10. Suggested Record Sheet Columns

For each completed run, record:

- checkpoint stem
- volume `L`
- `width`
- `n-cycles`
- `radius`
- `terminal-n-layers`
- `terminal-width`
- `parity`
- `rg-type`
- `log-scale-clip`
- `offdiag-clip`
- seed
- total epochs
- final batch size
- learning-rate schedule
- final centered `std(action diff)`
- final `ESS`
- final loss
- total wall time
- mean seconds per epoch

This makes it easy to compare both quality and cost across the scan.

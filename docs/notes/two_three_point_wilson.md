# Two- and Three-Point Functions for Unimproved Wilson Nf=2

This note documents the correlator setup for the SU(3) unimproved Wilson Nf=2 workflow in this repository.

## Scope and Conventions

- Fermion action: unimproved Wilson (no clover term).
- Boundary condition for production spectroscopy runs: anti-periodic in time for fermions (`fermion_bc = 1,1,1,-1`).
- Time direction is the last lattice axis (`t = x[3]` for 4D).
- Proton interpolator:
  - `chi_alpha(x) = eps_{abc} (u_a^T(x) C gamma_5 d_b(x)) u_{c,alpha}(x)`.
- Positive-parity projector:
  - `P_+ = (1 + gamma_t)/2`.

The implementation follows the classic Wuppertal-era strategy (point/smeared hadron operators and sequential-source 3pt construction).

## Two-Point Functions

### Pion 2pt

For a point source at `x0`, define the quark propagator `S(x, x0)` from `D S = eta` with spin-color unit sources.

For the local pseudoscalar channel:

`C_pi(t) = sum_{x, x4=t} Tr[ S(x, x0) S^dagger(x, x0) ]`

with separation index `dt = (t - t0) mod Lt`.

In code (`jaxqft/core/measurements.py`), this is `type = "pion_2pt"` and returns scalar keys:

- `c_t0`, `c_t1`, ..., `c_t(Lt-1)`.

### Proton 2pt

Using the local interpolator above and spin projection:

`C_N(dt) = sum_{x, x4=t} Re Tr[ P_+ G_N(x, x0) ]`

where `G_N` is the standard direct-minus-exchange color-spin contraction from three quark propagators with `C gamma_5` diquark structure.

In code this is `type = "proton_2pt"` (or `nucleon_2pt`) with:

- `parity = "+"` for `P_+`, `parity = "-"` for `P_-`.

## Inline Measurement Integration (`mcmc.py`)

`mcmc.py` calls `run_inline_measurements(...)` before each measurement update step and stores each scalar value in `inline_history`.

Example TOML snippet:

```toml
[[measurements.inline]]
type = "pion_2pt"
name = "pion_2pt"
every = 20
source = [0, 0, 0, 0]

[[measurements.inline]]
type = "proton_2pt"
name = "proton_2pt"
every = 20
source = [0, 0, 0, 0]
parity = "+"
```

Notes:

- These measurements are expensive because each call performs 12 Dirac solves for a full spin-color point-source propagator.
- If both pion and proton are requested at the same step and source, the propagator is reused through `MeasurementContext.cache`.
- Each inline record now includes inversion timing fields:
  - `inv_n_solves`, `inv_solve_total_sec_step`, `inv_solve_total_sec_this_call`
  - `inv_solve_mean_sec`, `inv_solve_min_sec`, `inv_solve_max_sec`
  - `inv_prop_build_wall_sec`, `inv_cache_hit`
  - plus measurement wall timers `wall_total_sec`, `wall_after_prop_sec`

## Practical External-Ensemble Workflow

If HMC generation is done externally (e.g. Chroma), use:

- `scripts/su3_wilson_nf2/measure_cfgs_2pt.py`

to scan `.lime`/`.npy`/`.pkl` configurations and measure `pion_2pt`/`proton_2pt` with inversion timing summaries.

## Three-Point Functions (Sequential Source)

For current `J_mu` insertion and sink time `t_sink`:

`C_3pt^J(t_sink, tau; p', p) = sum_{x,y} e^{-i p' x} e^{+i q y} < chi_N(x,t_sink) J_mu(y,tau) chibar_N(0) >`

with `q = p' - p`.

### Fixed-sink sequential method

1. Compute forward quark propagators from source: `S_q(y,0)`.
2. Build sink-contracted sequential source at fixed `t_sink`, sink momentum `p'`, and sink projector.
3. Solve one sequential inversion per sink choice:
   - `D psi_seq = eta_seq`.
4. Contract `psi_seq` with current insertion and remaining forward propagator to get all `tau` for that sink setup.

Tradeoff:

- Efficient for scanning many insertion times `tau` and momenta transfers `q` at fixed sink quantum numbers.
- Requires a new sequential solve when changing sink momentum/projector/operator structure.

## Ratios and Matrix Elements

Define 2pt and 3pt ratios with matched kinematics to cancel overlap factors. At large Euclidean times:

- `0 << tau << t_sink`,
- excited-state contamination is exponentially suppressed,
- ratio plateaus to the desired matrix element combination.

### Vector current decomposition

`<N(p')|V_mu|N(p)> = ubar(p') [ gamma_mu F1(Q^2) + i sigma_{mu nu} q_nu/(2 M_N) F2(Q^2) ] u(p)`

`G_E = F1 - (Q^2 / 4 M_N^2) F2`,
`G_M = F1 + F2`.

### Axial current decomposition

`<N(p')|A_mu|N(p)> = ubar(p') [ gamma_mu gamma_5 G_A(Q^2) + q_mu gamma_5/(2 M_N) G_P(Q^2) ] u(p)`

At each `Q^2`, solve a linear system built from all measured projector/current/momentum channels to extract `F1, F2` (or `G_E, G_M`) and `G_A, G_P`.

## Statistical Procedure (Rigorous, IAT-aware)

For each measured observable (`2pt` timeslice values, ratio values, fitted amplitudes):

1. Estimate integrated autocorrelation time `tau_int` (IPS/Sokal/Gamma; code has all three in `jaxqft.stats.autocorr`).
2. Use effective sample size `N_eff = N / (2 tau_int)` when reporting mean errors.
3. Build blocked bins with block size `>= 2 tau_int` before jackknife/bootstrap fits.
4. Perform correlated fits (covariance from blocked samples), with stability checks under fit-window changes.
5. Quote systematic spread from fit-range variation and (for 3pt) source-sink separation scans.

## Historical References (Wuppertal-era)

- S. Gusken et al., Phys. Lett. B227 (1989) 266.
- S. Gusken, Nucl. Phys. B (Proc. Suppl.) 17 (1990) 361.
- M. Gockeler et al. (Wuppertal group), early 1990s nucleon matrix-element and form-factor studies using Wilson fermions and sequential-source techniques.

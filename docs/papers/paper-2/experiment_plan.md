# Paper 2 Experiment Plan

This is the research plan for the RG/universality paper.

Unlike paper 1, this is not primarily an architecture-comparison paper. The
central object is the exact blocked hierarchy of the learned action.

## Main questions

1. Does the learned hierarchy define useful RG trajectories in observable space?
2. Do different bare points flow toward a common renormalized trajectory?
3. Can the learned hierarchy remain informative at large volume even when
   fine-level reweighting is already difficult?
4. Can one treat the learned action as an operational ``quantum-perfect'' action
   in the sense that all blocked levels are exact samplers for the learned
   theory?

## Phase A: fixed-architecture canonical baseline

Before doing any volume-dependent tuning, establish a clean baseline in which
both the architecture and the staged training schedule are held fixed across
volumes.

Baseline architecture:

- Wilsonian Gaussian coarse-eta branch
- uniform across all nonterminal levels
- `width = 64`
- `n_cycles = 2`
- `radius = 1`
- `eta_gaussian = level`
- `gaussian_width = 64`
- `gaussian_radius = 1`
- `terminal_prior = learned`
- `terminal_n_layers = 2`
- `terminal_width = 64`

Baseline schedule:

- `1000` epochs at batch `16`, lr `3e-4`
- `1000` epochs at batch `32`, lr `3e-4`
- `1000` epochs at batch `64`, lr `3e-4`
- `2000` epochs at batch `64`, lr `1e-4`
- `2000` epochs at batch `64`, lr `3e-5`
- `4000` epochs at batch `64`, lr `1e-5`

This is the right scaling test because it removes the ambiguity introduced by
changing per-level capacity with the volume. Only after this fixed-architecture
baseline is in hand should bottlenecks be identified and tuned models be
introduced.

## Essential numerical program

### R1. Volume scaling at a fixed target point

Run the current best Wilsonian Gaussian branch at:

- `L = 16`
- `L = 32`
- `L = 64`
- `L = 128` if feasible

Record:

- `std(ΔS)`
- `std(ΔS)/L`
- `ESS`
- time to threshold at fixed `std(ΔS)/L`

This establishes whether the residual mismatch remains intensive.

### R2. Blocked observable trajectories

For final production checkpoints, measure at every RG level:

- `xi/L`
- Binder cumulant
- susceptibility
- blocked two-point function at low momentum
- connected four-point observable

Do this first at the canonical point:

- `m^2 = -0.4`
- `lambda = 2.4`

In parallel, generate canonical HMC reference ensembles and record:

- correlation length
- connected susceptibility
- integrated autocorrelation time
- ESS

### R3. Bare-point scan

At fixed `lambda = 2.4`, scan several `m^2` values around the target point.

Suggested first scan:

- `m^2 in {-0.55, -0.50, -0.45, -0.40, -0.35, -0.30}`

The goal is to see whether the blocked trajectories merge after several RG
levels.

### R4. Optional projection to coupling space

Only after the observable-space story is clear, fit a truncated coarse action
and extract projected running couplings.

### R5. Training criticality and intensive mismatch scaling

This is now a central physics question rather than an optimizer-only
diagnostic.

The main observable should be

- `std(Delta S) / L`

in two dimensions, compared across volumes at approximately fixed `xi/L`. The
working hypothesis is:

- away from criticality, `std(Delta S) / L` is approximately intensive
- near criticality, it acquires a critical enhancement controlled by the
  integrated correlator of the local mismatch density
- if the mismatch lies mainly in the `Z2`-even, energy-like sector, then in the
  2d Ising class one expects a logarithmic enhancement rather than a strong
  power law

Numerical checks to perform:

1. determine `xi` and `g2_c` from HMC finite-size scaling
2. measure `std(Delta S) / L` for the canonical model family across
   `g2 = -0.4, -0.5, -0.585, -0.70` and volumes `L = 16, 32, 64, 128`
3. compare `std(Delta S) / L` directly against `xi`
4. test finite-size scaling near criticality using:
   - `A + B sqrt(log xi)`
   - `A + B sqrt(log L)` at approximately fixed `xi/L`
   - a generic power-law fit as a null comparison
5. check whether the same story holds for cheaper and more expressive model
   variants

Related training-side observable:

- fluctuations of representative gradient components or gradient-density
  components as a function of `L`, `xi`, and batch size

The relevant quantity is not an ad hoc ``minimal stable batch size'' by itself,
but the variance of the batch-averaged gradient density. If that quantity shows
critical enhancement, then the observed need for larger batches near the
critical line has a direct physical interpretation.

## Required new tooling

This paper will need analysis scripts that are not yet finished:

1. blocked-observable measurement at each RG depth
2. comparison of blocked trajectories across target points
3. optional fitting of a truncated coarse action
4. locality diagnostics based on score/Hessian decay
5. threshold-crossing analysis for fixed `std(ΔS)/L`

The first two are essential; the third is optional. The fourth and fifth are
central to the RG/locality scaling argument.

## Related Work

The closest literature does not yet appear to analyze the critical scaling of
flow-training observables themselves. The present plan should therefore cite
the nearest precedents clearly and delimit what is new here.

- Singha, Chakrabarti, and Arora study a conditional normalizing flow for
  scalar `phi^4` theory in the critical region and emphasize interpolation or
  extrapolation from non-critical training points into the critical regime
  [@Singha2022ConditionalNF]. This is close in physical setting, but it does
  not analyze the critical scaling of training diagnostics such as
  `std(Delta S) / L`, gradient fluctuations, or batch-size requirements.
- Albergo et al. demonstrate successful flow-based sampling in the critical
  Schwinger model [@Albergo2022SchwingerCritical]. This is strong evidence that
  flow methods can work at criticality, but it is not a study of the critical
  behavior of training itself.
- Bacchio et al. introduce learned trivializing gradient flows for lattice
  gauge theories and argue for favorable scaling with a very lightweight
  parametrization [@Bacchio2023TrivializingFlows]. This is highly relevant to
  the scaling discussion, but it does not formulate a critical-scaling theory
  for training observables or gradient noise.
- Bialas, Korcyl, and Stebel analyze alternative gradient estimators for normalizing-flow
  training with computationally intensive target distributions, including a
  Schwinger-model test at criticality [@Bialas2024NFComputationalTargets]. They
  study estimator cost, memory use, and numerical stability, but not the
  critical scaling of training quality metrics.
- Bialas, Korcyl, and Stebel also study autocorrelation times in neural Markov
  chain Monte Carlo simulations for the 2d Ising model
  [@Bialas2023NeuralMCMCAutocorr]. That work examines loss choices, symmetry
  constraints, and autocorrelation behavior as functions of temperature, but it
  does not develop a theory of critical scaling for normalizing-flow training.
- In broader machine learning, McCandlish et al. develop the gradient-noise
  scale picture and the notion of a critical batch size
  [@McCandlish2018LargeBatch], while Shallue et al. provide a broad empirical
  study of data parallelism and batch-size scaling across workloads
  [@Shallue2019DataParallelism]. These works are conceptually relevant for the
  batch-size discussion, but they do not connect batch scaling to physical
  criticality or to lattice-field-theory observables.

The gap left by this literature is exactly the one to be tested here:

- whether a physically meaningful training observable such as
  `std(Delta S) / L` exhibits universal or near-universal critical behavior
- whether gradient-fluctuation growth near criticality can be tied to standard
  field-theory critical exponents
- whether severe degradation in exact reweighting metrics can coexist with
  accurate long-distance physics in a controlled, local learned action

Bibliography for this section:

- [related_work_refs.bib](/Users/kostas/Box%20Sync/Work/projects/ECP/CSD/TrivializingMaps/jaxQFT/docs/papers/paper-2/related_work_refs.bib)

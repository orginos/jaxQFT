# Domain Decomposition Notes for U(1) Wilson Fermions

This note documents the **current** domain-decomposition code in `jaxQFT` for the
2D U(1) Wilson-fermion Schwinger model. It is intended as the repository-level
reference for what is implemented now, not for what we may eventually replace it with.

For a derivation-style write-up with the phase-1 and phase-2 formulas in LaTeX, see:

- `docs/notes/domain_decomposition_wilson_u1.tex`

Status date: `2026-03-25`

## Scope

There are three distinct DD paths in the repo:

1. **Phase 1: exact DD observables, measurement-only**
   - exact interior Schur-complement observables on a time-slab decomposition
   - no DD update algorithm
   - no determinant factorization

2. **Phase 2: quenched multilevel pion measurement**
   - pure-gauge level-0 chain
   - internal pure-gauge conditional level-1 chain inside the measurement
   - factorized pion observable plus level-0 bias correction
   - no fermion determinant in HMC

3. **Phase 3: exact dense determinant-factorization reference**
   - tiny-lattice dense reference only
   - exact block determinant split
   - no multiboson correction yet
   - no Giusti-style stochastic projector yet

## Code Map

- Geometry and generic DD indexing:
  - `jaxqft/core/domain_decomposition.py`
- Exact phase-1 DD observables:
  - `jaxqft/core/measurements.py`
  - measurement names:
    - `pion_2pt_dd`
    - `eta_2pt_dd`
- Phase-2 multilevel quenched estimator:
  - `jaxqft/core/multilevel_quenched.py`
  - `jaxqft/core/measurements.py`
  - measurement name:
    - `pion_2pt_ml_dd`
- Pure-gauge conditional DD updates:
  - `jaxqft/models/u1_ym_dd.py`
- Phase-3 exact dense determinant reference:
  - `jaxqft/models/u1_wilson_nf2_dd.py`

## Gauge Convention

The link transformation is

\[
U_\mu(x) \to U_\mu^\Omega(x)
= \Omega^\dagger(x)\,U_\mu(x)\,\Omega(x+\hat\mu).
\]

The fermion field transforms as

\[
\psi(x) \to \psi^\Omega(x)=\Omega^\dagger(x)\psi(x).
\]

This is the convention used in:

- `jaxqft/models/u1_ym.py`
- `jaxqft/models/u1_wilson_nf2.py`

The phase-2 multilevel measurement is tested against this convention directly:
the actual `pion_2pt_ml_dd` output is invariant under a site gauge transform in the
deterministic `level1_ncfg=1` limit.

## Time-Slab Geometry

The DD geometry is always built from a set of frozen time slabs.

For the multilevel pion measurement:

- the source must lie in one unfrozen connected component
- the sink lies in the other unfrozen connected component
- the frozen region is the union

\[
\Sigma = \Sigma_L \cup \Sigma_R.
\]

The current phase-2 code assumes exactly **two** unfrozen domains:

- source domain: \(A\)
- sink domain: \(C\)
- frozen overlap region: \(\Sigma\)

with block structure

\[
D =
\begin{pmatrix}
D_{AA} & D_{A\Sigma} & 0 \\
D_{\Sigma A} & D_{\Sigma\Sigma} & D_{\Sigma C} \\
0 & D_{C\Sigma} & D_{CC}
\end{pmatrix}.
\]

The source is constrained to lie in the bulk of its unfrozen domain by
`source_margin`.

## Phase 1: Exact Interior DD Observables

The phase-1 observables are exact interior Schur-complement measurements.

Split the lattice sites into interior \(I\) and frozen boundary \(B\):

\[
D =
\begin{pmatrix}
D_{II} & D_{IB} \\
D_{BI} & D_{BB}
\end{pmatrix}.
\]

The exact interior operator is

\[
S_I = D_{II} - D_{IB} D_{BB}^{-1} D_{BI},
\]

and the exact interior propagator is

\[
G_I = S_I^{-1}.
\]

The current phase-1 measurements use this exact interior propagator and therefore:

- are exact for the chosen DD geometry
- are measurement-only
- do not change HMC/SMD/GHMC

This is what `pion_2pt_dd` and `eta_2pt_dd` implement.

## Phase 2: Quenched Multilevel Pion Estimator

Phase 2 now contains two factorization families:

- `factorization_kind = "boundary_transfer"`
- `factorization_kind = "giusti"`

The default is still `boundary_transfer`, so existing cards keep their previous
behavior unless they opt into the Giusti-style path.

### 2A. Boundary-Transfer Factorization

This is the older phase-2 formulation. It uses a projected boundary transfer
built from the bare slab block \(D_{\Sigma\Sigma}\).

### Projector Basis

Let

\[
V \in \mathbb{C}^{|\Sigma|\times N_\text{proj}}
\]

denote the projector basis on the union of the two frozen slabs.

Current supported basis choices:

- `full`: full orthonormal basis on the overlap
- `probe`: sparse canonical probing basis
- `laplace` / `distillation`: low covariant-Laplacian modes, split evenly by slab
- `svd` / `singular`: low singular-vector basis of the slab-restricted
  \(D_{\Sigma\Sigma}\), split evenly by slab

### Implemented Transfer Matrix

The implemented projected transfer matrix is

\[
M = V^\dagger D_{\Sigma\Sigma} V.
\]

This is the key approximation of the boundary-transfer path.

### Source and Sink Factors

For a point source in the source domain \(A\), the code constructs

\[
S_x
=
M^{-1} V^\dagger D_{\Sigma A} D_{AA}^{-1} \eta_x,
\]

where \(\eta_x\) is the spin-color point source at the chosen source site.

For the sink domain \(C\), the code constructs

\[
L_y
=
\left[D_{CC}^{-1} D_{C\Sigma} V\right]_y,
\]

where \(y\) runs over sink-domain sites.

These are implemented in

- `compute_factorized_pion_factors(...)`

in `jaxqft/core/multilevel_quenched.py`.

### Factorized Propagator

The implemented factorized propagator is therefore

\[
G_{\text{fact}}(y,x) = L_y S_x.
\]

This is local in the source and sink domains once the frozen boundary and the
projector basis are fixed.

### Pion Contraction

The point-to-all factorized pion contraction is

\[
C_{\pi,\text{fact}}(y,x)
=
\mathrm{tr}\!\left[
G_{\text{fact}}(y,x)\,G_{\text{fact}}(y,x)^\dagger
\right].
\]

The code works with block objects instead of forming the full matrix at every sink:

\[
S_{\text{blk}} = S_x S_x^\dagger,
\qquad
L_{\text{blk}}(y) = L_y^\dagger L_y.
\]

Then

\[
C_{\pi,\text{fact}}(y,x)
=
\mathrm{tr}\!\left[S_{\text{blk}}\,L_{\text{blk}}(y)\right].
\]

In index form the implemented contraction is

\[
\sum_{p,q} S_{pq}\,L_{qp}(y),
\]

which is why the sink block enters with the transpose in the final contraction.

This transpose was a real bug once; the current code includes the fix and the
validation suite now checks the block-contraction identity explicitly.

### 2B. Giusti-Style Surface Factorization

The Giusti-style path is enabled with

\[
\texttt{factorization\_kind = "giusti"}.
\]

It uses two asymmetric approximations and then averages them:

1. source side dressed, sink side bare
2. sink side dressed, source side bare

Let

\[
\bar A = A \cup \Sigma,
\qquad
\bar C = \Sigma \cup C.
\]

For the partition \((\bar A, C)\), the exact two-block Schur form is

\[
G_{CA}
=
-\left(D_{CC} - D_{C\bar A} D_{\bar A \bar A}^{-1} D_{\bar A C}\right)^{-1}
\, D_{C\bar A} D_{\bar A\bar A}^{-1}.
\]

The implemented Giusti-style approximation drops the cross-domain Schur term and
keeps only the local dressing on the \(\bar A\) side:

\[
G^{(\bar A|C)}_{\mathrm{fact}}
\approx
- D_{CC}^{-1} D_{C\bar A} D_{\bar A\bar A}^{-1}.
\]

Similarly, for the partition \((A,\bar C)\),

\[
G^{(A|\bar C)}_{\mathrm{fact}}
\approx
- D_{\bar C\bar C}^{-1} D_{\bar C A} D_{AA}^{-1}.
\]

Because Wilson fermions are nearest-neighbor, the cross-domain couplings only
touch the surface of the dressed domain. The current Giusti implementation
therefore builds its projector basis only on the relevant surface slices:

- source-dressed branch: sink-facing slab surface
- sink-dressed branch: source-facing slab surface

For the source-dressed branch, with support \(S_C \subset \Sigma\),

\[
G^{(\bar A|C)}_{\mathrm{fact}}(y,x)
=
\left[D_{CC}^{-1} D_{C S_C} V_C\right]_y
\left[V_C^\dagger \left(D_{\bar A\bar A}^{-1}\eta_x\right)_{S_C}\right].
\]

For the sink-dressed branch, with support \(S_A \subset \Sigma\),

\[
G^{(A|\bar C)}_{\mathrm{fact}}(y,x)
=
\left[\left(D_{\bar C\bar C}^{-1} V_A\right)_{C,S_A}\right]_y
\left[V_A^\dagger D_{S_A A} D_{AA}^{-1}\eta_x\right].
\]

The code builds the pion correlator from these two branches separately and then averages:

\[
C_{\pi,\mathrm{fact}}^{\mathrm{giusti}}
=
\frac12\left(
C_{\pi,\mathrm{fact}}^{(\bar A|C)}
+
C_{\pi,\mathrm{fact}}^{(A|\bar C)}
\right).
\]

The level-0 bias correction is then

\[
B = C_{\pi,\mathrm{exact}} - C_{\pi,\mathrm{fact}}^{\mathrm{giusti}}.
\]

Supported projector choices on the Giusti surface supports are:

- `full`
- `probe`
- `laplace` / `distillation`

`svd` is intentionally not enabled for the Giusti path.

### Momentum Projection

For sink times \(t\) in the sink domain,

\[
C_{\pi,\text{fact}}(t,p)
=
\sum_{y_0-x_0=t}
e^{ip(y_1-x_1)} C_{\pi,\text{fact}}(y,x).
\]

When `average_pm = true`, the code uses the cosine projection

\[
C_\pi^{\cos}(t,p)
=
\frac12\left(C_\pi(t,+p)+C_\pi(t,-p)\right).
\]

### Level-0 / Level-1 Estimator

For a level-0 gauge configuration \(U_n\), the code computes:

Level-0 factorized observable:

\[
A_n^{(0)}(t,p)=C_{\text{fact}}(t,p;U_n).
\]

Exact observable:

\[
C_n^{\text{exact}}(t,p)=C_{\text{exact}}(t,p;U_n).
\]

Bias:

\[
B_n(t,p)=C_n^{\text{exact}}(t,p)-A_n^{(0)}(t,p).
\]

Then, with an internal level-1 conditional DD chain at fixed boundaries,

\[
\langle\!\langle C_{\text{fact}}(t,p)\rangle\!\rangle_1(U_n)
\equiv
A_n^{(1)}(t,p).
\]

The corrected level-0 sample is

\[
X_n(t,p)=A_n^{(1)}(t,p)+B_n(t,p).
\]

This is exactly what the measurement exports:

- `approx_l0`
- `approx_ml`
- `bias`
- `corrected`
- `exact`

### Covariance Used for Fits

The fit covariance is built from the corrected level-0 samples \(X_n\), after
blocking along the outer Markov chain:

\[
\widehat{\mathrm{Cov}}_{tt'}(p)
=
\frac{1}{N_0(N_0-1)}
\sum_{n=1}^{N_0}
\left(X_n(t,p)-\bar X(t,p)\right)
\left(X_n(t',p)-\bar X(t',p)\right).
\]

The fitter writes this covariance to the sidecar

- `p{p}_cov_mean`

inside the `*.npz` output of `scripts/mcmc/fit_2pt_dispersion.py`.

## Phase 2 Validation Coverage

The current validation suite is

- `scripts/u1_wilson_nf2/validate_phase2_multilevel.py`

It currently checks:

1. projector construction
2. source/sink domain locality under fixed-boundary splicing
3. block-contraction identity against an explicit dense factorized propagator
4. pair-average identity
5. bias identity

\[
\texttt{approx\_l0} + \texttt{bias} = \texttt{exact}
\]

6. gauge invariance of the multilevel measurement output in the deterministic limit

This suite should be treated as mandatory whenever the DD contraction logic changes.

## Phase 2 Empirical Status

For the current bare-\(D_{\Sigma\Sigma}\) factorization:

- light pion + thin slabs was not good enough
- heavier pion helped
- increasing slab width from `1` to `3` improved the level-0 correlation substantially
- increasing slab width from `3` to `5` did **not** improve further

So for the current implementation, simply making the slab thicker is not a monotonic
win.

## Phase 3: Exact Dense Determinant-Factorization Reference

The phase-3 code in `jaxqft/models/u1_wilson_nf2_dd.py` is a **tiny-lattice dense reference**.

It factorizes

\[
\det D
=
\det D_{bb}
\left(\prod_i \det D_i\right)
\det R_b,
\]

more precisely

\[
\det D
=
\det D_{bb}\,
\left(\prod_i \det D_i\right)\,
\det R_b,
\]

with

\[
R_b
=
I - D_{bb}^{-1}\sum_i D_{bi} D_i^{-1} D_{ib}.
\]

The active-link-dependent Hamiltonian then keeps:

- the conditional gauge action
- local-domain determinant monomials
- exact boundary-correction monomial

This is a **reference path only**. It is not the scalable production algorithm from
the Giusti paper.

## What Is Not Implemented Yet

The following should be stated explicitly so the note does not drift away from the code:

- a stochastic Giusti projector is **not** implemented yet
- a deterministic distillation basis built from the **dressed interface operator**
  is **not** implemented yet
- the phase-3 multiboson correction algorithm is **not** implemented yet

At present, the phase-2 multilevel code should be described as:

- quenched
- measurement-only at level 0
- pure-gauge conditional level-1 updates internal to the measurement
- with two available factorization families:
  - projected bare-slab transfer
  \[
  M = V^\dagger D_{\Sigma\Sigma} V
  \]
  - Giusti-style asymmetric dressed-domain surface factorization

## Recommended Reading Order in the Repo

For someone resuming the DD work, the clean order is:

1. `HANDOFF.md`
2. `docs/notes/domain_decomposition_wilson_u1.md`
3. `jaxqft/core/domain_decomposition.py`
4. `jaxqft/core/multilevel_quenched.py`
5. `jaxqft/core/measurements.py`
6. `scripts/u1_wilson_nf2/validate_phase2_multilevel.py`
7. `jaxqft/models/u1_wilson_nf2_dd.py`

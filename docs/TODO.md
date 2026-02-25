# TODO

## Fermions (Priority)
- Replace autodiff pseudofermion force with analytic Wilson force kernel.
- Add inverter diagnostics: iteration counts, residual norms, timing per solve.
- Add even-odd preconditioning for Wilson operator.
- Add Hasenbusch monomial factorization for Nf=2.
- Add RHMC monomials for odd/flavor-fractional determinants.

## Update Algorithms
- Document HMC/SMD/GHMC formulas and exact implementation choices.
- Add references section in docs for each update scheme.
- Add nested multi-timescale integrator schedule for monomial lists.
- Add explicit regression test for SMD rejection branch: verify momentum flip on reject.

## Gauge Sector Docs
- Document gauge action and force formulas for SU3/SU2/U1 implementations.
- Record sign/normalization conventions and map to code paths.

## Testing and Validation
- Keep monomial-level correctness tests before production trajectory tests.
- Extend Wilson tests with force finite-difference/autodiff consistency at monomial level.
- Add regression thresholds and CI-friendly selfcheck gates.

## Performance
- Add per-monomial timing in production scripts.
- Add benchmark scripts for Wilson monomial solve/force split.
- Prepare optional backend interface for QUDA while keeping JAX-native default.

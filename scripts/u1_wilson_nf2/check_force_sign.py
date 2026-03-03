#!/usr/bin/env python3
"""Check pseudofermion force sign via finite differences.

For a correct force F = -dS/da (in the Lie algebra), a small step
q' = exp(eps * F) * q should DECREASE the action: S(q') < S(q).
"""
from __future__ import annotations
import os, platform, sys
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_ym import U1YangMills


def main():
    th = U1WilsonNf2(
        lattice_shape=(6, 6), beta=2.0, mass=0.1, batch_size=1,
        seed=42, pseudofermion_force_mode="autodiff",
        solver_kind="cg", solver_form="normal",
        cg_tol=1e-12, cg_maxiter=2000,
        jit_dirac_kernels=True, jit_solvers=True,
    )

    U = th.hot_start(scale=0.1)

    # Fix a pseudofermion
    th.prepare_trajectory(U, traj_length=1.0)
    phi = th.fermion_monomial.phi.copy()

    print("=== Force sign check ===")
    print("If force = -dS/da, then S(exp(eps*F)*U) < S(U) for small eps > 0.\n")

    # Gauge-only force and action
    F_gauge = U1YangMills.force(th, U)
    S_gauge_0 = float(jnp.mean(U1YangMills.action(th, U)))

    # Full force (gauge + fermion) and full action
    F_full = th.force(U)
    S_full_0 = float(jnp.mean(th.action(U)))

    # Fermion-only force
    F_ferm = F_full - F_gauge

    # Pseudofermion action only
    S_pf_0 = float(jnp.mean(th.pseudofermion_action(U, phi)))

    print(f"S_gauge(U) = {S_gauge_0:.8f}")
    print(f"S_pf(U)    = {S_pf_0:.8f}")
    print(f"S_full(U)  = {S_full_0:.8f}")
    print(f"||F_gauge|| = {float(jnp.sqrt(jnp.sum(jnp.abs(F_gauge)**2))):.6f}")
    print(f"||F_ferm||  = {float(jnp.sqrt(jnp.sum(jnp.abs(F_ferm)**2))):.6f}")
    print()

    for eps in [1e-2, 1e-3, 1e-4]:
        # Step along gauge force
        U_g = jnp.exp(eps * F_gauge) * U
        S_g = float(jnp.mean(U1YangMills.action(th, U_g)))
        dS_g = S_g - S_gauge_0

        # Step along fermion force
        U_f = jnp.exp(eps * F_ferm) * U
        S_pf_f = float(jnp.mean(th.pseudofermion_action(U_f, phi)))
        dS_pf = S_pf_f - S_pf_0

        # Step along full force
        U_full = jnp.exp(eps * F_full) * U
        S_full = float(jnp.mean(th.action(U_full)))
        dS_full = S_full - S_full_0

        # Also check: finite-diff derivative vs force inner product
        # dS/deps = <F, dS/da> should be -||F||^2 if F = -dS/da
        F_gauge_norm2 = float(jnp.sum(jnp.real(F_gauge * jnp.conj(F_gauge))))
        F_ferm_norm2 = float(jnp.sum(jnp.real(F_ferm * jnp.conj(F_ferm))))

        print(f"eps={eps:.0e}:")
        print(f"  gauge:   dS_g/eps = {dS_g/eps:.6f}  (expect ~ -{F_gauge_norm2:.6f} = -||F_g||^2)")
        print(f"  fermion: dS_pf/eps = {dS_pf/eps:.6f}  (expect ~ -{F_ferm_norm2:.6f} = -||F_f||^2)")
        print(f"  full:    dS_full/eps = {dS_full/eps:.6f}")
        # Sign check
        sign_g = "OK (decreasing)" if dS_g < 0 else "WRONG (increasing!)"
        sign_f = "OK (decreasing)" if dS_pf < 0 else "WRONG (increasing!)"
        print(f"  gauge sign: {sign_g}")
        print(f"  fermion sign: {sign_f}")
        print()

    print("=== Lie-algebra finite-difference force check ===")
    print("Compare: analytic force vs finite-diff of action along each algebra direction\n")

    # For a single link, compute dS/da by finite difference
    eps = 1e-5
    # Pick link at (batch=0, mu=0, x=0, y=0)
    F_analytic = th.force(U)

    # Finite-diff: perturb U_mu(x) -> exp(i*eps) * U_mu(x)
    # For U(1), the algebra element is i*delta (anti-Hermitian)
    # Use the algebra inner product: F_a = -Re(F * conj(i)) = Im(F)
    # But let's just do the full field perturbation

    # Perturb ALL links simultaneously along F direction
    dU = eps * F_analytic  # anti-Hermitian perturbation
    U_plus = jnp.exp(dU) * U
    U_minus = jnp.exp(-dU) * U
    S_plus = float(jnp.sum(th.action(U_plus)))
    S_minus = float(jnp.sum(th.action(U_minus)))
    dS_fd = (S_plus - S_minus) / (2 * eps)

    # Analytic: dS/deps along F direction = <dS/da, F>_algebra
    # For U(1), <A, B>_algebra = -Re(sum A*B) = -sum Re(A*B)
    # Since F = -dS/da, <dS/da, F> = -||F||^2
    F_norm2 = -float(jnp.sum(jnp.real(F_analytic * F_analytic)))  # positive since F is anti-Hermitian
    print(f"  Finite-diff dS/deps = {dS_fd:.6f}")
    print(f"  Expected (= -||F||^2) = {-F_norm2:.6f}")
    print(f"  Ratio: {dS_fd / (-F_norm2 + 1e-30):.6f} (should be ~1.0)")


if __name__ == "__main__":
    main()

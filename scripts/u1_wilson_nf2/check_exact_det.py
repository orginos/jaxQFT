#!/usr/bin/env python3
"""Validate pseudofermion machinery against exact determinant on a tiny lattice.

On a small lattice, we can build the full D matrix explicitly and compute
det(D^dag D) exactly. This allows us to:
1. Verify D adjoint relationship
2. Compare pseudofermion action average with -ln det(D^dag D)
3. Check if HMC samples the correct distribution
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


def build_full_D_matrix(th, U_single):
    """Build the full Dirac matrix for a single batch element."""
    Vol = int(np.prod(th.lattice_shape))
    Ns = th.Ns
    Nc = 1  # U(1)
    N = Vol * Ns * Nc
    D = np.zeros((N, N), dtype=np.complex64)

    for i in range(N):
        # Create unit vector e_i
        e = np.zeros((1, *th.lattice_shape, Ns, Nc), dtype=np.complex64)
        # Flatten index: i = x_flat * Ns * Nc + s * Nc + c
        x_flat = i // (Ns * Nc)
        s = (i % (Ns * Nc)) // Nc
        c = i % Nc
        # Convert flat index to lattice coords
        coords = np.unravel_index(x_flat, th.lattice_shape)
        idx = (0,) + coords + (s, c)
        e[idx] = 1.0

        e_jax = jnp.asarray(e)
        U_batch = U_single[None, ...]
        De = th.apply_D(U_batch, e_jax)
        D[:, i] = np.asarray(De).reshape(-1)

    return D


def main():
    L = 4
    lattice_shape = (L, L)
    beta = 2.0
    mass = 0.1
    seed = 42

    th = U1WilsonNf2(
        lattice_shape=lattice_shape, beta=beta, mass=mass,
        batch_size=1, seed=seed,
        solver_kind="cg", solver_form="normal",
        cg_tol=1e-12, cg_maxiter=2000,
        jit_dirac_kernels=False, jit_solvers=False,
    )

    U = th.hot_start(scale=0.1)
    U_single = np.asarray(U[0])

    print(f"Lattice: {lattice_shape}, beta={beta}, mass={mass}")
    Vol = int(np.prod(lattice_shape))
    Ns = th.Ns
    N = Vol * Ns
    print(f"D matrix size: {N}x{N} (Vol={Vol}, Ns={Ns})")
    print()

    # Build D and D^dag matrices
    print("Building full D matrix...")
    D_mat = build_full_D_matrix(th, U_single)

    print("=== Test 1: D^dag relationship ===")
    Ddag_mat = build_full_D_matrix_dagger(th, U_single)
    diff = np.linalg.norm(Ddag_mat - D_mat.conj().T) / (np.linalg.norm(D_mat) + 1e-12)
    print(f"  ||D^dag_applied - D^T*||/||D|| = {diff:.2e}")
    print(f"  => D^dag {'CORRECT' if diff < 1e-5 else 'WRONG'}")
    print()

    print("=== Test 2: D^dag D positive definiteness ===")
    DdD = D_mat.conj().T @ D_mat
    eigs = np.linalg.eigvalsh(DdD)
    print(f"  min eigenvalue of D^dag D: {eigs.min():.6e}")
    print(f"  max eigenvalue of D^dag D: {eigs.max():.6e}")
    print(f"  condition number: {eigs.max() / (eigs.min() + 1e-30):.2e}")
    print(f"  => D^dag D {'positive definite' if eigs.min() > 0 else 'NOT positive definite!'}")
    print()

    print("=== Test 3: Exact determinant vs pseudofermion action ===")
    det_D = np.linalg.det(D_mat)
    det_DdD = np.linalg.det(DdD)
    ln_det_DdD = np.real(np.log(det_DdD))
    print(f"  det(D) = {det_D:.6e}")
    print(f"  |det(D)|^2 = {abs(det_D)**2:.6e}")
    print(f"  det(D^dag D) = {det_DdD:.6e}")
    print(f"  ln det(D^dag D) = {ln_det_DdD:.6f}")
    print()

    # Average <eta^dag eta> = N (number of complex components) check
    n_samples = 5000
    pf_actions = []
    for _ in range(n_samples):
        eta = th.random_fermion()
        phi = th.apply_Ddag(U, eta)
        s_pf = float(jnp.mean(th.pseudofermion_action(U, phi)))
        pf_actions.append(s_pf)
    pf_mean = np.mean(pf_actions)
    pf_err = np.std(pf_actions) / np.sqrt(n_samples)
    print(f"  <S_pf(U_fixed, phi)> = {pf_mean:.4f} +/- {pf_err:.4f}")
    print(f"  Expected (= N_dof = {N}) = {N}")
    print(f"  Ratio: {pf_mean / N:.4f}")
    print()

    print("=== Test 4: CG solve accuracy ===")
    eta = th.random_fermion()
    phi = th.apply_Ddag(U, eta)
    x = th.solve_normal(U, phi)
    residual = th.apply_normal(U, x) - phi
    rel_res = float(jnp.linalg.norm(residual) / (jnp.linalg.norm(phi) + 1e-12))
    print(f"  CG relative residual: {rel_res:.2e}")
    print()

    # Exact solve vs CG
    phi_np = np.asarray(phi[0]).reshape(-1)
    x_exact = np.linalg.solve(DdD, phi_np)
    x_cg = np.asarray(x[0]).reshape(-1)
    x_diff = np.linalg.norm(x_exact - x_cg) / (np.linalg.norm(x_exact) + 1e-12)
    print(f"  Exact solve vs CG: rel diff = {x_diff:.2e}")
    print()

    print("=== Test 5: Exact reweighting check ===")
    # For exact sampling: <O>_Nf2 = <O * det(D^dag D)> / <det(D^dag D)>
    # where <...> is over the quenched ensemble
    # Compare plaquette: HMC vs exact reweighting from quenched
    from jaxqft.core.integrators import minnorm2
    from jaxqft.core.update import HMC

    # Run quenched HMC
    th_q = U1WilsonNf2(
        lattice_shape=lattice_shape, beta=beta, mass=mass,
        batch_size=1, seed=seed, include_fermion_monomial=False,
        jit_dirac_kernels=False, jit_solvers=False,
    )
    I_q = minnorm2(th_q.force, th_q.evolveQ, 10, 1.0)
    ch_q = HMC(T=th_q, I=I_q, verbose=False, seed=seed)
    q_q = th_q.hot_start(scale=0.2)
    for _ in range(200):
        q_q = ch_q.evolve(q_q, 1)

    plaq_quenched = []
    ln_det_vals = []
    for _ in range(2000):
        q_q = ch_q.evolve(q_q, 3)
        plaq = float(jnp.mean(th_q.average_plaquette(q_q)))
        plaq_quenched.append(plaq)
        # Compute exact det for reweighting
        U_s = np.asarray(q_q[0])
        D_s = build_full_D_matrix(th, U_s)
        ln_d = np.real(np.log(np.linalg.det(D_s.conj().T @ D_s) + 1e-300))
        ln_det_vals.append(ln_d)

    plaq_q = np.array(plaq_quenched)
    ln_det = np.array(ln_det_vals)

    # Reweight: <P>_Nf2 = <P * exp(ln det)> / <exp(ln det)>
    # To avoid overflow, subtract the mean
    ln_det_shifted = ln_det - ln_det.mean()
    w = np.exp(ln_det_shifted)
    plaq_rw = np.sum(plaq_q * w) / np.sum(w)
    plaq_q_mean = plaq_q.mean()
    plaq_q_err = plaq_q.std() / np.sqrt(len(plaq_q))

    print(f"  Quenched <plaq>      = {plaq_q_mean:.6f} +/- {plaq_q_err:.6f}")
    print(f"  Reweighted <plaq>_Nf2 = {plaq_rw:.6f}")
    print()

    # Now run dynamical HMC
    th_d = U1WilsonNf2(
        lattice_shape=lattice_shape, beta=beta, mass=mass,
        batch_size=1, seed=seed+1000,
        solver_kind="cg", solver_form="normal",
        cg_tol=1e-12, cg_maxiter=2000,
        pseudofermion_force_mode="autodiff",
    )
    I_d = minnorm2(th_d.force, th_d.evolveQ, 10, 1.0)
    ch_d = HMC(T=th_d, I=I_d, verbose=False, seed=seed+1000)
    q_d = th_d.hot_start(scale=0.2)
    for _ in range(200):
        q_d = ch_d.evolve(q_d, 1)

    plaq_dyn = []
    for _ in range(500):
        q_d = ch_d.evolve(q_d, 3)
        plaq = float(jnp.mean(th_d.average_plaquette(q_d)))
        plaq_dyn.append(plaq)

    plaq_d_mean = np.mean(plaq_dyn)
    plaq_d_err = np.std(plaq_dyn) / np.sqrt(len(plaq_dyn))

    print(f"  HMC dynamical <plaq> = {plaq_d_mean:.6f} +/- {plaq_d_err:.6f}")
    print()
    diff_sigma = abs(plaq_rw - plaq_d_mean) / max(plaq_d_err, 1e-10)
    print(f"  Reweight vs HMC: {diff_sigma:.1f} sigma")
    if diff_sigma > 3:
        print("  => DISAGREEMENT: HMC is likely sampling the WRONG distribution!")
    else:
        print("  => Agreement: HMC samples the correct distribution.")


def build_full_D_matrix_dagger(th, U_single):
    """Build the full D^dag matrix for a single batch element."""
    Vol = int(np.prod(th.lattice_shape))
    Ns = th.Ns
    Nc = 1
    N = Vol * Ns * Nc
    Ddag = np.zeros((N, N), dtype=np.complex64)

    for i in range(N):
        e = np.zeros((1, *th.lattice_shape, Ns, Nc), dtype=np.complex64)
        x_flat = i // (Ns * Nc)
        s = (i % (Ns * Nc)) // Nc
        c = i % Nc
        coords = np.unravel_index(x_flat, th.lattice_shape)
        idx = (0,) + coords + (s, c)
        e[idx] = 1.0

        e_jax = jnp.asarray(e)
        U_batch = jnp.asarray(U_single)[None, ...]
        Ddage = th.apply_Ddag(U_batch, e_jax)
        Ddag[:, i] = np.asarray(Ddage).reshape(-1)

    return Ddag


if __name__ == "__main__":
    main()

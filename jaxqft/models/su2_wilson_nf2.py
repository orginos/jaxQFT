"""SU(2) gauge + Nf=2 Wilson-fermion building blocks."""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from jaxqft.fermions import WilsonDiracOperator, check_gamma_algebra
from jaxqft.models.su2_ym import SU2YangMills
from jaxqft.models.su3_ym import _dagger


Array = jax.Array


def _vdot_field(a: Array, b: Array) -> Array:
    return jnp.sum(jnp.conj(a) * b)


@dataclass
class SU2WilsonNf2(SU2YangMills):
    mass: float = 0.05
    wilson_r: float = 1.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 500

    def __post_init__(self):
        super().__post_init__()
        self.dirac = WilsonDiracOperator(ndim=self.Nd, mass=self.mass, wilson_r=self.wilson_r, dtype=self.dtype)
        self.gamma = self.dirac.gamma
        self.Ns = self.dirac.Ns

    def fermion_shape(self) -> Tuple[int, ...]:
        return (self.Bs, *self.lattice_shape, self.Ns, 2)

    def _roll_site(self, x: Array, shift: int, direction: int) -> Array:
        return jnp.roll(x, shift=shift, axis=1 + direction)

    def apply_D(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(U, psi, take_mu=self._take_mu, roll_site=self._roll_site, dagger=_dagger, sign=-1)

    def apply_Ddag(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(U, psi, take_mu=self._take_mu, roll_site=self._roll_site, dagger=_dagger, sign=+1)

    def apply_normal(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply_normal(U, psi, take_mu=self._take_mu, roll_site=self._roll_site, dagger=_dagger)

    def apply_D_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U, psi, take_mu=self._take_mu, roll_site=self._roll_site, dagger=_dagger, sign=-1, use_sparse_gamma=False
        )

    def apply_normal_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply_normal(
            U, psi, take_mu=self._take_mu, roll_site=self._roll_site, dagger=_dagger, use_sparse_gamma=False
        )

    def random_fermion(self) -> Array:
        k1 = self._split_key()
        re = jax.random.normal(k1, self.fermion_shape(), dtype=jnp.float32)
        k2 = self._split_key()
        im = jax.random.normal(k2, self.fermion_shape(), dtype=jnp.float32)
        return (re + 1j * im).astype(self.dtype)

    def sample_pseudofermion(self, U: Array) -> Array:
        chi = self.random_fermion()
        return self.apply_D(U, chi)

    def pseudofermion_action(self, U: Array, phi: Array) -> Array:
        def solve_one(Ub, pb):
            def A(x):
                xb = x[None, ...]
                yb = self.apply_normal(Ub[None, ...], xb)
                return yb[0]

            xb, _ = cg(A, pb, tol=self.cg_tol, maxiter=self.cg_maxiter)
            return jnp.real(_vdot_field(pb, xb))

        return jax.vmap(solve_one, in_axes=(0, 0))(U, phi)


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def test_dirac_adjoint(th: SU2WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    chi = th.random_fermion()
    lhs = _vdot_field(chi, th.apply_D(U, psi))
    rhs = _vdot_field(th.apply_Ddag(U, chi), psi)
    rel = jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12)
    return {"rel_adjoint_error": float(rel)}


def test_normal_positivity(th: SU2WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    npsi = th.apply_normal(U, psi)
    return {"normal_quadratic_form": float(jnp.real(_vdot_field(psi, npsi)))}


def test_perf(th: SU2WilsonNf2, n_iter: int = 10) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    th.apply_D(U, psi).block_until_ready()
    th.apply_D_dense(U, psi).block_until_ready()
    th.apply_normal(U, psi).block_until_ready()
    th.apply_normal_dense(U, psi).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_D(U, psi).block_until_ready()
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_D_dense(U, psi).block_until_ready()
    t3 = time.perf_counter()
    t4 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_normal(U, psi).block_until_ready()
    t5 = time.perf_counter()
    t6 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_normal_dense(U, psi).block_until_ready()
    t7 = time.perf_counter()

    d_sparse = th.apply_D(U, psi)
    d_dense = th.apply_D_dense(U, psi)
    n_sparse = th.apply_normal(U, psi)
    n_dense = th.apply_normal_dense(U, psi)
    rel_d = float(jnp.linalg.norm(d_sparse - d_dense) / (jnp.linalg.norm(d_dense) + 1e-12))
    rel_n = float(jnp.linalg.norm(n_sparse - n_dense) / (jnp.linalg.norm(n_dense) + 1e-12))
    return {
        "D_sparse_sec_per_call": (t1 - t0) / max(1, n_iter),
        "D_dense_sec_per_call": (t3 - t2) / max(1, n_iter),
        "N_sparse_sec_per_call": (t5 - t4) / max(1, n_iter),
        "N_dense_sec_per_call": (t7 - t6) / max(1, n_iter),
        "rel_D_sparse_vs_dense": rel_d,
        "rel_N_sparse_vs_dense": rel_n,
        "sparse_gamma_available": float(1.0 if th.dirac.sparse_gamma_available else 0.0),
    }


def main():
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(description="SU2 Wilson Nf=2 checks")
    ap.add_argument("--shape", type=str, default="8,8")
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ")
    ap.add_argument("--exp-method", type=str, default="su2", choices=["su2", "expm"])
    ap.add_argument("--n-iter-timing", type=int, default=10)
    ap.add_argument("--tests", type=str, default="all", help="comma-separated: gamma,adjoint,normal,perf,all")
    args = ap.parse_args()

    tests = {t.strip().lower() for t in args.tests.split(",") if t.strip()}
    if "all" in tests:
        tests = {"gamma", "adjoint", "normal", "perf"}

    th = SU2WilsonNf2(
        lattice_shape=_parse_shape(args.shape),
        beta=args.beta,
        mass=args.mass,
        wilson_r=args.r,
        batch_size=args.batch,
        layout=args.layout,
        exp_method=args.exp_method,
    )
    print("JAX backend:", jax.default_backend())
    print("Theory config:")
    print(f"  shape: {th.lattice_shape}")
    print(f"  Nd: {th.Nd}")
    print(f"  spin_dim: {th.Ns}")
    print(f"  fermion_shape: {th.fermion_shape()}")

    if "gamma" in tests:
        g = check_gamma_algebra(th.gamma)
        gs = th.dirac.gamma_sparse_dense_error(th.random_fermion())
        print("Gamma algebra:")
        print(f"  max rel clifford error: {g['max_rel_clifford_error']:.6e}")
        print(f"  max hermiticity error:  {g['max_hermiticity_error']:.6e}")
        if bool(int(gs["sparse_available"])):
            print(f"  max rel sparse-vs-dense gamma apply error: {gs['max_rel_error']:.6e}")
        else:
            print("  sparse-vs-dense gamma apply check: skipped (no sparse gamma form)")
    if "adjoint" in tests:
        print("Dirac adjoint consistency:")
        print(f"  rel error: {test_dirac_adjoint(th)['rel_adjoint_error']:.6e}")
    if "normal" in tests:
        print("Normal operator check:")
        print(f"  <psi, DdagD psi>: {test_normal_positivity(th)['normal_quadratic_form']:.6e}")
    if "perf" in tests:
        p = test_perf(th, n_iter=max(1, args.n_iter_timing))
        print("Performance test (sparse gamma vs dense gamma):")
        print(f"  sparse gamma available: {bool(int(p['sparse_gamma_available']))}")
        print(f"  D      sec/call sparse/dense: {p['D_sparse_sec_per_call']:.6e} / {p['D_dense_sec_per_call']:.6e}")
        print(f"  DdagD  sec/call sparse/dense: {p['N_sparse_sec_per_call']:.6e} / {p['N_dense_sec_per_call']:.6e}")
        print(f"  rel diff D (sparse,dense): {p['rel_D_sparse_vs_dense']:.6e}")
        print(f"  rel diff DdagD (sparse,dense): {p['rel_N_sparse_vs_dense']:.6e}")


if __name__ == "__main__":
    main()


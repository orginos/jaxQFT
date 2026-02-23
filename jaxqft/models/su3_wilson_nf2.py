"""SU(3) gauge + Nf=2 Wilson-fermion building blocks."""

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
from jaxqft.models.su3_ym import SU3YangMills, _dagger


Array = jax.Array


def _vdot_field(a: Array, b: Array) -> Array:
    return jnp.sum(jnp.conj(a) * b)


@dataclass
class SU3WilsonNf2(SU3YangMills):
    mass: float = 0.0
    wilson_r: float = 1.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 500

    def __post_init__(self):
        super().__post_init__()
        self.dirac = WilsonDiracOperator(ndim=self.Nd, mass=self.mass, wilson_r=self.wilson_r, dtype=self.dtype)
        self.gamma = self.dirac.gamma
        self.Ns = self.dirac.Ns

    def fermion_shape(self) -> Tuple[int, ...]:
        # Spin-major for cheap gamma/projector ops and contiguous color vectors.
        return (self.Bs, *self.lattice_shape, self.Ns, 3)

    def _roll_site(self, x: Array, shift: int, direction: int) -> Array:
        return jnp.roll(x, shift=shift, axis=1 + direction)

    def apply_D(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=-1,
            use_sparse_gamma=True,
        )

    def apply_Ddag(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=+1,
            use_sparse_gamma=True,
        )

    def apply_normal(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply_normal(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            use_sparse_gamma=True,
        )

    def apply_D_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=-1,
            use_sparse_gamma=False,
        )

    def apply_Ddag_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=+1,
            use_sparse_gamma=False,
        )

    def apply_normal_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply_normal(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            use_sparse_gamma=False,
        )

    def random_fermion(self) -> Array:
        k1 = self._split_key()
        re = jax.random.normal(k1, self.fermion_shape(), dtype=jnp.float32)
        k2 = self._split_key()
        im = jax.random.normal(k2, self.fermion_shape(), dtype=jnp.float32)
        return (re + 1j * im).astype(self.dtype)

    def sample_pseudofermion(self, U: Array) -> Array:
        # For Nf=2, phi = D(U) chi gives weight exp(-phi^\dagger (D^\dagger D)^-1 phi).
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


def test_dirac_adjoint(th: SU3WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    chi = th.random_fermion()
    lhs = _vdot_field(chi, th.apply_D(U, psi))
    rhs = _vdot_field(th.apply_Ddag(U, chi), psi)
    rel = jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12)
    return {"rel_adjoint_error": float(rel), "lhs_real": float(jnp.real(lhs)), "rhs_real": float(jnp.real(rhs))}


def test_normal_positivity(th: SU3WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    npsi = th.apply_normal(U, psi)
    val = jnp.real(_vdot_field(psi, npsi))
    return {"normal_quadratic_form": float(val)}


def test_gamma_sparse_dense(th: SU3WilsonNf2) -> Dict[str, float]:
    if not th.dirac.sparse_gamma_available:
        return {"sparse_available": 0.0, "max_rel_error": float("nan")}
    psi = th.random_fermion()
    out = th.dirac.gamma_sparse_dense_error(psi)
    return {"sparse_available": out["sparse_available"], "max_rel_error": out["max_rel_error"]}


def test_perf(th: SU3WilsonNf2, n_iter: int = 10) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()

    d_sparse = th.apply_D(U, psi)
    d_dense = th.apply_D_dense(U, psi)
    n_sparse = th.apply_normal(U, psi)
    n_dense = th.apply_normal_dense(U, psi)
    d_sparse.block_until_ready()
    d_dense.block_until_ready()
    n_sparse.block_until_ready()
    n_dense.block_until_ready()

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

    rel_d = float(jnp.linalg.norm(d_sparse - d_dense) / (jnp.linalg.norm(d_dense) + 1e-12))
    rel_n = float(jnp.linalg.norm(n_sparse - n_dense) / (jnp.linalg.norm(n_dense) + 1e-12))
    return {
        "sparse_enabled": float(1.0 if th.dirac.sparse_gamma_available else 0.0),
        "D_sparse_sec_per_call": (t1 - t0) / max(1, n_iter),
        "D_dense_sec_per_call": (t3 - t2) / max(1, n_iter),
        "N_sparse_sec_per_call": (t5 - t4) / max(1, n_iter),
        "N_dense_sec_per_call": (t7 - t6) / max(1, n_iter),
        "rel_D_sparse_vs_dense": rel_d,
        "rel_N_sparse_vs_dense": rel_n,
    }


def main():
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(description="Wilson-fermion building-block checks")
    ap.add_argument("--shape", type=str, default="4,4,4,8")
    ap.add_argument("--beta", type=float, default=5.8)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ")
    ap.add_argument("--exp-method", type=str, default="su3", choices=["expm", "su3"])
    ap.add_argument("--n-iter-timing", type=int, default=10)
    ap.add_argument("--tests", type=str, default="all", help="comma-separated: gamma,adjoint,normal,perf,all")
    args = ap.parse_args()

    tests = {t.strip().lower() for t in args.tests.split(",") if t.strip()}
    if "all" in tests:
        tests = {"gamma", "adjoint", "normal", "perf"}

    shape = _parse_shape(args.shape)
    th = SU3WilsonNf2(
        lattice_shape=shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=args.layout,
        exp_method=args.exp_method,
        mass=args.mass,
        wilson_r=args.r,
    )

    print("JAX backend:", jax.default_backend())
    print("Theory config:")
    print(f"  shape: {shape}")
    print(f"  Nd: {th.Nd}")
    print(f"  spin_dim: {th.Ns}")
    print(f"  fermion_shape: {th.fermion_shape()}")

    if "gamma" in tests:
        g = check_gamma_algebra(th.gamma)
        print("Gamma algebra:")
        print(f"  max rel clifford error: {g['max_rel_clifford_error']:.6e}")
        print(f"  max hermiticity error:  {g['max_hermiticity_error']:.6e}")
        gs = test_gamma_sparse_dense(th)
        if bool(int(gs["sparse_available"])):
            print(f"  max rel sparse-vs-dense gamma apply error: {gs['max_rel_error']:.6e}")
        else:
            print("  sparse-vs-dense gamma apply check: skipped (no sparse gamma form)")

    if "adjoint" in tests:
        a = test_dirac_adjoint(th)
        print("Dirac adjoint consistency:")
        print(f"  rel error <chi,Dpsi> - <Ddag chi,psi>: {a['rel_adjoint_error']:.6e}")

    if "normal" in tests:
        n = test_normal_positivity(th)
        print("Normal operator check:")
        print(f"  <psi, DdagD psi> real value: {n['normal_quadratic_form']:.6e}")

    if "perf" in tests:
        p = test_perf(th, n_iter=max(1, args.n_iter_timing))
        print("Performance test (sparse gamma vs dense gamma):")
        print(f"  sparse gamma available: {bool(int(p['sparse_enabled']))}")
        print(f"  D      sec/call sparse/dense: {p['D_sparse_sec_per_call']:.6e} / {p['D_dense_sec_per_call']:.6e}")
        print(f"  DdagD  sec/call sparse/dense: {p['N_sparse_sec_per_call']:.6e} / {p['N_dense_sec_per_call']:.6e}")
        print(f"  rel diff D (sparse,dense): {p['rel_D_sparse_vs_dense']:.6e}")
        print(f"  rel diff DdagD (sparse,dense): {p['rel_N_sparse_vs_dense']:.6e}")


if __name__ == "__main__":
    main()

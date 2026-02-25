"""SU(3) gauge + Nf=2 Wilson-fermion model with monomial Hamiltonian composition."""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from jaxqft.core import HamiltonianModel
from jaxqft.fermions import WilsonDiracOperator, check_gamma_algebra, gamma5
from jaxqft.models.su3_ym import (
    SU3YangMills,
    _dagger,
    gauge_transform_links,
    project_su3_algebra,
    random_site_gauge_with_method,
)


Array = jax.Array


def _vdot_field(a: Array, b: Array) -> Array:
    return jnp.sum(jnp.conj(a) * b)


@dataclass
class GaugeActionMonomial:
    """Gauge plaquette monomial using the optimized pure-gauge kernels."""

    model: "SU3WilsonNf2"
    name: str = "gauge"
    timescale: int = 0
    stochastic: bool = False

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        _ = q
        _ = traj_length

    def action(self, q: Array) -> Array:
        return SU3YangMills.action(self.model, q)

    def force(self, q: Array) -> Array:
        return SU3YangMills.force(self.model, q)


@dataclass
class WilsonNf2PseudofermionMonomial:
    """Nf=2 pseudofermion monomial: S_pf = phi^dagger (D^dagger D)^-1 phi."""

    model: "SU3WilsonNf2"
    name: str = "wilson_nf2_pf"
    timescale: int = 1
    stochastic: bool = True
    refresh_mode: str = "heatbath"
    gamma: float = 0.3
    eta: Optional[Array] = field(default=None, init=False, repr=False)
    phi: Optional[Array] = field(default=None, init=False, repr=False)
    _force_fn: Optional[object] = field(default=None, init=False, repr=False)

    def clear(self) -> None:
        self.eta = None
        self.phi = None

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        mode = str(self.refresh_mode).lower()
        if mode == "heatbath":
            self.eta = self.model.random_fermion()
        elif mode == "ou":
            zeta = self.model.random_fermion()
            if self.eta is None:
                self.eta = zeta
            else:
                c1 = float(jnp.exp(-float(self.gamma) * float(traj_length)))
                c2 = float(jnp.sqrt(max(0.0, 1.0 - c1 * c1)))
                self.eta = c1 * self.eta + c2 * zeta
        else:
            raise ValueError(f"Unknown pseudofermion refresh mode: {self.refresh_mode}")
        # Nf=2 pseudofermion refresh convention:
        # phi = D^\dagger eta, with S_pf = phi^\dagger (D^\dagger D)^{-1} phi.
        self.phi = self.model.apply_Ddag(q, self.eta)

    def action(self, q: Array) -> Array:
        if self.phi is None:
            self.refresh(q, traj_length=1.0)
        assert self.phi is not None
        return self.model.pseudofermion_action(q, self.phi)

    def _build_force_fn(self):
        def _pf_action_sum(u_re: Array, u_im: Array, phi: Array) -> Array:
            u = (u_re + 1j * u_im).astype(self.model.dtype)
            return jnp.sum(self.model.pseudofermion_action(u, phi))

        grad_re_im = jax.grad(_pf_action_sum, argnums=(0, 1))

        def force_fn(u: Array, phi: Array) -> Array:
            g_re, g_im = grad_re_im(jnp.real(u), jnp.imag(u), phi)
            dS_dU = (g_re - 1j * g_im).astype(self.model.dtype)
            G = jnp.swapaxes(dS_dU, -1, -2)
            return project_su3_algebra(u @ G)

        return force_fn

    def force(self, q: Array) -> Array:
        if self.phi is None:
            self.refresh(q, traj_length=1.0)
        assert self.phi is not None
        if self._force_fn is None:
            self._force_fn = self._build_force_fn()
        return self._force_fn(q, self.phi)


@dataclass
class SU3WilsonNf2(SU3YangMills):
    mass: float = 0.0
    wilson_r: float = 1.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 500
    include_gauge_monomial: bool = True
    include_fermion_monomial: bool = True
    gauge_timescale: int = 0
    fermion_timescale: int = 1
    pseudofermion_refresh: str = "heatbath"
    pseudofermion_gamma: Optional[float] = None
    smd_gamma: float = 0.3
    auto_refresh_pseudofermions: bool = True
    requires_trajectory_refresh: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.pseudofermion_refresh).lower()
        if self.pseudofermion_gamma is None:
            self.pseudofermion_gamma = float(self.smd_gamma) if mode == "ou" else 0.3
        self.dirac = WilsonDiracOperator(ndim=self.Nd, mass=self.mass, wilson_r=self.wilson_r, dtype=self.dtype)
        self.gamma = self.dirac.gamma
        self.Ns = self.dirac.Ns
        monomials = []
        if self.include_gauge_monomial:
            monomials.append(GaugeActionMonomial(self, timescale=int(self.gauge_timescale)))
        self.fermion_monomial: Optional[WilsonNf2PseudofermionMonomial]
        if self.include_fermion_monomial:
            self.fermion_monomial = WilsonNf2PseudofermionMonomial(
                self,
                timescale=int(self.fermion_timescale),
                refresh_mode=str(self.pseudofermion_refresh),
                gamma=float(self.pseudofermion_gamma),
            )
            monomials.append(self.fermion_monomial)
        else:
            self.fermion_monomial = None
        self.hamiltonian = HamiltonianModel(tuple(monomials))
        self.requires_trajectory_refresh = any(bool(getattr(m, "stochastic", False)) for m in self.hamiltonian.monomials)

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
        # For Nf=2, use phi = D^\dagger(U) eta with
        # S_pf = phi^\dagger (D^\dagger D)^-1 phi.
        chi = self.random_fermion()
        return self.apply_Ddag(U, chi)

    def pseudofermion_action(self, U: Array, phi: Array) -> Array:
        def solve_one(Ub, pb):
            def A(x):
                xb = x[None, ...]
                yb = self.apply_normal(Ub[None, ...], xb)
                return yb[0]

            xb, _ = cg(A, pb, tol=self.cg_tol, maxiter=self.cg_maxiter)
            return jnp.real(_vdot_field(pb, xb))

        return jax.vmap(solve_one, in_axes=(0, 0))(U, phi)

    def prepare_trajectory(self, U: Array, traj_length: float = 1.0) -> None:
        """Refresh stochastic monomials (pseudofermions) once per trajectory."""
        self.hamiltonian.prepare_trajectory(U, traj_length=float(traj_length))

    def clear_pseudofermion(self) -> None:
        if self.fermion_monomial is not None:
            self.fermion_monomial.clear()

    def action_breakdown(self, U: Array) -> Dict[str, Array]:
        if self.auto_refresh_pseudofermions and self.fermion_monomial is not None and self.fermion_monomial.phi is None:
            self.fermion_monomial.refresh(U)
        return self.hamiltonian.action_breakdown(U)

    def action(self, U: Array) -> Array:
        if self.auto_refresh_pseudofermions and self.fermion_monomial is not None and self.fermion_monomial.phi is None:
            self.fermion_monomial.refresh(U)
        return self.hamiltonian.action_from_monomials(U)

    def force(self, U: Array) -> Array:
        if self.auto_refresh_pseudofermions and self.fermion_monomial is not None and self.fermion_monomial.phi is None:
            self.fermion_monomial.refresh(U)
        return self.hamiltonian.force_from_monomials(U)


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _gauge_transform_fermion(psi: Array, Omega: Array) -> Array:
    # Link convention: U' = Omega^\dag(x) U(x) Omega(x+mu), so psi' = Omega^\dag psi.
    return jnp.einsum("...ab,...sb->...sa", _dagger(Omega), psi)


def _identity_links(th: SU3WilsonNf2) -> Array:
    eye = jnp.eye(3, dtype=th.dtype)
    if th.layout == "BMXYIJ":
        shape = (th.Bs, th.Nd, *th.lattice_shape, 3, 3)
    else:
        shape = (th.Bs, *th.lattice_shape, th.Nd, 3, 3)
    return jnp.broadcast_to(eye, shape)


def _gamma5_apply(g5: Array, psi: Array) -> Array:
    # psi: [..., Ns, Nc]
    return jnp.einsum("st,...tc->...sc", g5, psi)


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


def test_hamiltonian_monomials(th: SU3WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    th.prepare_trajectory(U, traj_length=1.0)
    breakdown = th.action_breakdown(U)
    S = th.action(U)
    F = th.force(U)
    total_breakdown = None
    for v in breakdown.values():
        total_breakdown = v if total_breakdown is None else (total_breakdown + v)
    assert total_breakdown is not None
    rel = jnp.linalg.norm(S - total_breakdown) / (jnp.linalg.norm(S) + 1e-12)
    return {
        "n_monomials": float(len(th.hamiltonian.monomials)),
        "action_sum_consistency": float(rel),
        "mean_action": float(jnp.mean(S)),
        "mean_force_norm": float(jnp.mean(jnp.linalg.norm(F, axis=(-2, -1)))),
    }


def test_pseudofermion_refresh(th: SU3WilsonNf2, traj_length: float = 1.0, n_refresh: int = 8) -> Dict[str, float]:
    if th.fermion_monomial is None:
        return {"enabled": 0.0}
    m = th.fermion_monomial
    U = th.hot_start(scale=0.05)
    m.clear()
    corrs = []
    dphi = []
    eta_prev = None
    phi_prev = None
    for _ in range(max(1, int(n_refresh))):
        m.refresh(U, traj_length=float(traj_length))
        assert m.eta is not None
        assert m.phi is not None
        eta_curr = m.eta
        phi_curr = m.phi
        if eta_prev is not None:
            num = jnp.real(_vdot_field(eta_prev, eta_curr))
            den = jnp.sqrt(jnp.real(_vdot_field(eta_prev, eta_prev)) * jnp.real(_vdot_field(eta_curr, eta_curr)) + 1e-12)
            corrs.append(float(num / den))
        if phi_prev is not None:
            rel = jnp.linalg.norm(phi_curr - phi_prev) / (jnp.linalg.norm(phi_prev) + 1e-12)
            dphi.append(float(rel))
        eta_prev = eta_curr
        phi_prev = phi_curr

    mode = str(m.refresh_mode).lower()
    c1_exp = 0.0
    if mode == "ou":
        c1_exp = float(jnp.exp(-float(m.gamma) * float(traj_length)))
    return {
        "enabled": 1.0,
        "refresh_mode": 1.0 if mode == "ou" else 0.0,
        "expected_c1": c1_exp,
        "mean_eta_corr": float(jnp.mean(jnp.asarray(corrs))) if corrs else float("nan"),
        "std_eta_corr": float(jnp.std(jnp.asarray(corrs))) if corrs else float("nan"),
        "mean_rel_phi_change": float(jnp.mean(jnp.asarray(dphi))) if dphi else float("nan"),
    }


def test_twoflavor_gauge_invariance(
    th: SU3WilsonNf2,
    q_scale: float = 0.05,
    omega_scale: float = 0.05,
    n_trials: int = 3,
    gauge_exp_method: str = "expm",
) -> Dict[str, float]:
    rel_D = []
    rel_N = []
    rel_phi = []
    rel_Spf = []

    for _ in range(max(1, int(n_trials))):
        U = th.hot_start(scale=q_scale)
        Om = random_site_gauge_with_method(th, scale=omega_scale, method=gauge_exp_method)
        Up = gauge_transform_links(th, U, Om)

        psi = th.random_fermion()
        psi_p = _gauge_transform_fermion(psi, Om)
        Dpsi = th.apply_D(U, psi)
        Dpsi_p = th.apply_D(Up, psi_p)
        Dpsi_cov = _gauge_transform_fermion(Dpsi, Om)
        rel_D.append(float(jnp.linalg.norm(Dpsi_p - Dpsi_cov) / (jnp.linalg.norm(Dpsi_cov) + 1e-12)))

        Npsi = th.apply_normal(U, psi)
        Npsi_p = th.apply_normal(Up, psi_p)
        Npsi_cov = _gauge_transform_fermion(Npsi, Om)
        rel_N.append(float(jnp.linalg.norm(Npsi_p - Npsi_cov) / (jnp.linalg.norm(Npsi_cov) + 1e-12)))

        eta = th.random_fermion()
        eta_p = _gauge_transform_fermion(eta, Om)
        phi = th.apply_Ddag(U, eta)
        phi_p = th.apply_Ddag(Up, eta_p)
        phi_cov = _gauge_transform_fermion(phi, Om)
        rel_phi.append(float(jnp.linalg.norm(phi_p - phi_cov) / (jnp.linalg.norm(phi_cov) + 1e-12)))

        Spf = th.pseudofermion_action(U, phi)
        Spf_p = th.pseudofermion_action(Up, phi_p)
        rel_Spf.append(float(jnp.linalg.norm(Spf_p - Spf) / (jnp.linalg.norm(Spf) + 1e-12)))

    return {
        "max_rel_D_cov": float(np.max(rel_D)),
        "mean_rel_D_cov": float(np.mean(rel_D)),
        "max_rel_DdagD_cov": float(np.max(rel_N)),
        "mean_rel_DdagD_cov": float(np.mean(rel_N)),
        "max_rel_phi_cov": float(np.max(rel_phi)),
        "mean_rel_phi_cov": float(np.mean(rel_phi)),
        "max_rel_Spf_invariance": float(np.max(rel_Spf)),
        "mean_rel_Spf_invariance": float(np.mean(rel_Spf)),
    }


def test_dirac_conventions(th: SU3WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()

    # gamma5-hermiticity: D^\dagger = gamma5 D gamma5 (even dimensions only).
    g5_rel = float("nan")
    if th.Nd % 2 == 0:
        g5 = gamma5(th.gamma)
        lhs = th.apply_Ddag(U, psi)
        rhs = _gamma5_apply(g5, th.apply_D(U, _gamma5_apply(g5, psi)))
        g5_rel = float(jnp.linalg.norm(lhs - rhs) / (jnp.linalg.norm(lhs) + 1e-12))

    # Free-field stencil consistency for implemented D convention.
    U1 = _identity_links(th)
    psi0 = th.random_fermion()
    D_num = th.apply_D(U1, psi0)
    D_ref = (th.mass + th.wilson_r * th.Nd) * psi0
    for mu in range(th.Nd):
        psi_xpmu = th._roll_site(psi0, -1, mu)
        psi_xmmu = th._roll_site(psi0, +1, mu)
        # Matches apply(..., sign=-1): forward coeff=+1, backward coeff=-1.
        fwd = th.dirac.spin_project(psi_xpmu, mu, coeff=+1, use_sparse=False)
        bwd = th.dirac.spin_project(psi_xmmu, mu, coeff=-1, use_sparse=False)
        D_ref = D_ref - 0.5 * (fwd + bwd)
    rel_free = float(jnp.linalg.norm(D_num - D_ref) / (jnp.linalg.norm(D_ref) + 1e-12))

    return {
        "rel_gamma5_hermiticity": g5_rel,
        "rel_freefield_stencil": rel_free,
    }


def test_pseudofermion_refresh_identity(
    th: SU3WilsonNf2,
    q_scale: float = 0.05,
    traj_length: float = 1.0,
    n_trials: int = 3,
) -> Dict[str, float]:
    if th.fermion_monomial is None:
        return {"enabled": 0.0}
    m = th.fermion_monomial
    rels = []
    means_spf = []
    means_eta = []
    for _ in range(max(1, int(n_trials))):
        U = th.hot_start(scale=q_scale)
        m.refresh(U, traj_length=float(traj_length))
        assert m.eta is not None
        assert m.phi is not None
        Spf = th.pseudofermion_action(U, m.phi)
        eta2 = jnp.real(_vdot_field(m.eta, m.eta))
        spf_m = jnp.mean(Spf)
        rel = jnp.abs(spf_m - eta2) / (jnp.abs(eta2) + 1e-12)
        rels.append(float(rel))
        means_spf.append(float(spf_m))
        means_eta.append(float(eta2))
    return {
        "enabled": 1.0,
        "max_rel_spf_vs_eta2": float(np.max(rels)),
        "mean_rel_spf_vs_eta2": float(np.mean(rels)),
        "mean_spf": float(np.mean(means_spf)),
        "mean_eta2": float(np.mean(means_eta)),
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
    ap.add_argument("--include-gauge-monomial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-fermion-monomial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gauge-timescale", type=int, default=0)
    ap.add_argument("--fermion-timescale", type=int, default=1)
    ap.add_argument("--pf-refresh", type=str, default="heatbath", choices=["heatbath", "ou"])
    ap.add_argument(
        "--pf-gamma",
        type=float,
        default=None,
        help="OU pseudofermion gamma. Default: smd-gamma when --pf-refresh ou, else 0.3",
    )
    ap.add_argument(
        "--smd-gamma",
        type=float,
        default=0.3,
        help="SMD/GHMC OU friction gamma used as the default pseudofermion OU gamma",
    )
    ap.add_argument("--traj-length", type=float, default=1.0, help="trajectory length used in pseudofermion refresh tests")
    ap.add_argument("--n-refresh", type=int, default=8, help="number of refreshes for pseudofermion refresh diagnostics")
    ap.add_argument("--gauge-trials", type=int, default=3)
    ap.add_argument("--gauge-omega-scale", type=float, default=0.05)
    ap.add_argument("--gauge-exp-method", type=str, default="expm", choices=["expm", "su3", "auto"])
    ap.add_argument("--pf-id-trials", type=int, default=3)
    ap.add_argument("--auto-refresh-pseudofermions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--n-iter-timing", type=int, default=10)
    ap.add_argument(
        "--tests",
        type=str,
        default="all",
        help="comma-separated: gamma,adjoint,normal,perf,hamiltonian,pfrefresh,pfid,gauge,conventions,all",
    )
    args = ap.parse_args()

    tests = {t.strip().lower() for t in args.tests.split(",") if t.strip()}
    if "all" in tests:
        tests = {"gamma", "adjoint", "normal", "perf", "hamiltonian", "pfrefresh", "pfid", "gauge", "conventions"}

    shape = _parse_shape(args.shape)
    pf_gamma = args.pf_gamma
    if pf_gamma is None:
        pf_gamma = float(args.smd_gamma) if str(args.pf_refresh).lower() == "ou" else 0.3

    th = SU3WilsonNf2(
        lattice_shape=shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=args.layout,
        exp_method=args.exp_method,
        mass=args.mass,
        wilson_r=args.r,
        include_gauge_monomial=bool(args.include_gauge_monomial),
        include_fermion_monomial=bool(args.include_fermion_monomial),
        gauge_timescale=int(args.gauge_timescale),
        fermion_timescale=int(args.fermion_timescale),
        pseudofermion_refresh=str(args.pf_refresh),
        pseudofermion_gamma=float(pf_gamma),
        smd_gamma=float(args.smd_gamma),
        auto_refresh_pseudofermions=bool(args.auto_refresh_pseudofermions),
    )

    print("JAX backend:", jax.default_backend())
    print("Theory config:")
    print(f"  shape: {shape}")
    print(f"  Nd: {th.Nd}")
    print(f"  spin_dim: {th.Ns}")
    print(f"  fermion_shape: {th.fermion_shape()}")
    print(f"  monomials: {', '.join(th.hamiltonian.monomial_names())}")
    print(f"  monomial timescales: gauge={th.gauge_timescale}, fermion={th.fermion_timescale}")
    print(f"  smd gamma: {float(th.smd_gamma):.6g}")
    print(f"  pseudofermion refresh: mode={th.pseudofermion_refresh}, gamma={th.pseudofermion_gamma}")
    print(f"  auto_refresh_pseudofermions: {bool(th.auto_refresh_pseudofermions)}")

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

    if "hamiltonian" in tests:
        h = test_hamiltonian_monomials(th)
        print("Monomial Hamiltonian check:")
        print(f"  monomials: {int(h['n_monomials'])} ({', '.join(th.hamiltonian.monomial_names())})")
        print(f"  rel action sum consistency: {h['action_sum_consistency']:.6e}")
        print(f"  mean action: {h['mean_action']:.6e}")
        print(f"  mean force Frobenius norm: {h['mean_force_norm']:.6e}")

    if "pfrefresh" in tests:
        r = test_pseudofermion_refresh(th, traj_length=float(args.traj_length), n_refresh=int(args.n_refresh))
        if not bool(int(r.get("enabled", 0.0))):
            print("Pseudofermion refresh check: skipped (no fermion monomial)")
        else:
            print("Pseudofermion refresh check:")
            print(f"  mode: {th.pseudofermion_refresh}")
            print(f"  expected eta corr c1: {r['expected_c1']:.6e}")
            print(f"  measured eta corr mean/std: {r['mean_eta_corr']:.6e} / {r['std_eta_corr']:.6e}")
            print(f"  mean relative phi change per refresh: {r['mean_rel_phi_change']:.6e}")

    if "pfid" in tests:
        pfi = test_pseudofermion_refresh_identity(
            th,
            q_scale=0.05,
            traj_length=float(args.traj_length),
            n_trials=int(args.pf_id_trials),
        )
        if not bool(int(pfi.get("enabled", 0.0))):
            print("Pseudofermion refresh identity check: skipped (no fermion monomial)")
        else:
            print("Pseudofermion refresh identity check:")
            print(f"  max rel |Spf-eta2|/|eta2|:  {pfi['max_rel_spf_vs_eta2']:.6e}")
            print(f"  mean rel |Spf-eta2|/|eta2|: {pfi['mean_rel_spf_vs_eta2']:.6e}")
            print(f"  mean Spf / mean eta2:       {pfi['mean_spf']:.6e} / {pfi['mean_eta2']:.6e}")

    if "gauge" in tests:
        gauge_exp_method = args.gauge_exp_method
        if gauge_exp_method == "auto":
            gauge_exp_method = str(th.exp_method)
        g = test_twoflavor_gauge_invariance(
            th,
            q_scale=0.05,
            omega_scale=float(args.gauge_omega_scale),
            n_trials=int(args.gauge_trials),
            gauge_exp_method=gauge_exp_method,
        )
        print("Two-flavor monomial gauge-invariance/covariance test:")
        print(f"  max rel D covariance:       {g['max_rel_D_cov']:.6e}")
        print(f"  mean rel D covariance:      {g['mean_rel_D_cov']:.6e}")
        print(f"  max rel DdagD covariance:   {g['max_rel_DdagD_cov']:.6e}")
        print(f"  mean rel DdagD covariance:  {g['mean_rel_DdagD_cov']:.6e}")
        print(f"  max rel phi covariance:     {g['max_rel_phi_cov']:.6e}")
        print(f"  mean rel phi covariance:    {g['mean_rel_phi_cov']:.6e}")
        print(f"  max rel Spf invariance:     {g['max_rel_Spf_invariance']:.6e}")
        print(f"  mean rel Spf invariance:    {g['mean_rel_Spf_invariance']:.6e}")

    if "conventions" in tests:
        c = test_dirac_conventions(th)
        print("Dirac convention checks:")
        print(f"  rel gamma5-hermiticity:     {c['rel_gamma5_hermiticity']:.6e}")
        print(f"  rel free-field stencil D:   {c['rel_freefield_stencil']:.6e}")


if __name__ == "__main__":
    main()

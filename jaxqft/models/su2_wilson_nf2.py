"""SU(2) gauge + Nf=2 Wilson-fermion model with monomial Hamiltonian composition."""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg
try:
    from jax.scipy.sparse.linalg import bicgstab  # type: ignore
except Exception:
    bicgstab = None
try:
    from jax.scipy.sparse.linalg import gmres  # type: ignore
except Exception:
    gmres = None

from jaxqft.core import HamiltonianModel
from jaxqft.fermions import WilsonDiracOperator, check_gamma_algebra, gamma5
from jaxqft.models.su2_ym import (
    SU2YangMills,
    _dagger,
    gauge_transform_links,
    project_su2_algebra,
    random_site_gauge_with_method,
)


Array = jax.Array


def _vdot_field(a: Array, b: Array) -> Array:
    return jnp.sum(jnp.conj(a) * b)


def _field_norm(x: Array) -> Array:
    return jnp.sqrt(jnp.real(_vdot_field(x, x)))


def _algebra_inner(a: Array, b: Array) -> Array:
    """Positive-definite inner product on anti-Hermitian matrices: -Re Tr(a b)."""
    trab = jnp.real(jnp.einsum("...ab,...ba->...", a, b))
    return -jnp.sum(trab, axis=tuple(range(1, trab.ndim)))


def _extract_cg_num_iters(info: object) -> float:
    """Best-effort extraction of CG iteration count from SciPy/JAX-style info."""
    if info is None:
        return float("nan")
    if isinstance(info, (int, float, np.integer, np.floating)):
        return float(info)

    if isinstance(info, dict):
        for key in ("num_iters", "num_iter", "iterations", "niter", "iter"):
            if key in info:
                try:
                    return float(np.asarray(info[key]).item())
                except Exception:
                    pass

    for attr in ("num_iters", "num_iter", "iterations", "niter", "iter"):
        if hasattr(info, attr):
            try:
                return float(np.asarray(getattr(info, attr)).item())
            except Exception:
                pass

    try:
        return float(np.asarray(info).item())
    except Exception:
        return float("nan")


@dataclass
class GaugeActionMonomial:
    """Gauge plaquette monomial using the optimized pure-gauge kernels."""

    model: "SU2WilsonNf2"
    name: str = "gauge"
    timescale: int = 0
    stochastic: bool = False

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        _ = q
        _ = traj_length

    def action(self, q: Array) -> Array:
        return SU2YangMills.action(self.model, q)

    def force(self, q: Array) -> Array:
        return SU2YangMills.force(self.model, q)


@dataclass
class WilsonNf2PseudofermionMonomial:
    """Nf=2 pseudofermion monomial: S_pf = phi^dagger (D^dagger D)^-1 phi."""

    model: "SU2WilsonNf2"
    name: str = "wilson_nf2_pf"
    timescale: int = 1
    stochastic: bool = True
    refresh_mode: str = "heatbath"
    gamma: float = 0.3
    force_mode: str = "autodiff"
    eta: Optional[Array] = field(default=None, init=False, repr=False)
    phi: Optional[Array] = field(default=None, init=False, repr=False)
    _grad_links_ad_fn: Optional[object] = field(default=None, init=False, repr=False)
    _force_ad_fn: Optional[object] = field(default=None, init=False, repr=False)

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

    def _build_grad_links_ad_fn(self):
        def _pf_action_sum(u_re: Array, u_im: Array, phi: Array) -> Array:
            u = (u_re + 1j * u_im).astype(self.model.dtype)
            return jnp.sum(self.model.pseudofermion_action(u, phi))

        grad_re_im = jax.grad(_pf_action_sum, argnums=(0, 1))

        def grad_links(u: Array, phi: Array) -> Array:
            g_re, g_im = grad_re_im(jnp.real(u), jnp.imag(u), phi)
            return (g_re - 1j * g_im).astype(self.model.dtype)

        return grad_links

    def action_grad_links_autodiff(self, q: Array, phi: Array) -> Array:
        if self._grad_links_ad_fn is None:
            self._grad_links_ad_fn = self._build_grad_links_ad_fn()
        return self._grad_links_ad_fn(q, phi)

    def _build_force_ad_fn(self):
        grad_links = self.action_grad_links_autodiff

        def force_fn(u: Array, phi: Array) -> Array:
            dS_dU = grad_links(u, phi)
            G = jnp.swapaxes(dS_dU, -1, -2)
            return project_su2_algebra(u @ G)

        return force_fn

    def force_autodiff(self, q: Array, phi: Array) -> Array:
        if self._force_ad_fn is None:
            self._force_ad_fn = self._build_force_ad_fn()
        return self._force_ad_fn(q, phi)

    def force_analytic(self, q: Array, phi: Array) -> Array:
        if self.model.solver_form in ("split", "eo_split"):
            # Reuse Y from the first split solve: D^\dagger Y = phi, D X = Y.
            x, y = self.model.solve_split_with_intermediate(q, phi)
        else:
            x = self.model.solve_pseudofermion(q, phi)
            y = self.model.apply_D(q, x)
        return self.model.pseudofermion_force_from_solution(q, x, y)

    def force(self, q: Array) -> Array:
        if self.phi is None:
            self.refresh(q, traj_length=1.0)
        assert self.phi is not None
        mode = str(self.force_mode).lower()
        if mode == "autodiff":
            return self.force_autodiff(q, self.phi)
        if mode == "analytic":
            return self.force_analytic(q, self.phi)
        raise ValueError(f"Unknown pseudofermion force mode: {self.force_mode}")


@dataclass
class WilsonNf2EOPreconditionedMonomial:
    """Nf=2 EO-preconditioned pseudofermion monomial on the even Schur operator."""

    model: "SU2WilsonNf2"
    name: str = "wilson_nf2_eop_pf"
    timescale: int = 1
    stochastic: bool = True
    refresh_mode: str = "heatbath"
    gamma: float = 0.3
    force_mode: str = "autodiff"
    eta: Optional[Array] = field(default=None, init=False, repr=False)
    phi: Optional[Array] = field(default=None, init=False, repr=False)
    _grad_links_ad_fn: Optional[object] = field(default=None, init=False, repr=False)
    _force_ad_fn: Optional[object] = field(default=None, init=False, repr=False)

    def clear(self) -> None:
        self.eta = None
        self.phi = None

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        mode = str(self.refresh_mode).lower()
        if mode == "heatbath":
            eta = self.model.random_fermion()
        elif mode == "ou":
            zeta = self.model.random_fermion()
            if self.eta is None:
                eta = zeta
            else:
                c1 = float(jnp.exp(-float(self.gamma) * float(traj_length)))
                c2 = float(jnp.sqrt(max(0.0, 1.0 - c1 * c1)))
                eta = c1 * self.eta + c2 * zeta
        else:
            raise ValueError(f"Unknown pseudofermion refresh mode: {self.refresh_mode}")
        # EO-preconditioned monomial lives on even checkerboard.
        self.eta = self.model._project_even(eta)
        self.phi = self.model.apply_eo_schur_even_dagger(q, self.eta, normalized=True)
        self.phi = self.model._project_even(self.phi)

    def action(self, q: Array) -> Array:
        if self.phi is None:
            self.refresh(q, traj_length=1.0)
        assert self.phi is not None
        return self.model.eop_pseudofermion_action(q, self.phi)

    def _build_grad_links_ad_fn(self):
        def _pf_action_sum(u_re: Array, u_im: Array, phi: Array) -> Array:
            u = (u_re + 1j * u_im).astype(self.model.dtype)
            return jnp.sum(self.model.eop_pseudofermion_action(u, phi))

        grad_re_im = jax.grad(_pf_action_sum, argnums=(0, 1))

        def grad_links(u: Array, phi: Array) -> Array:
            g_re, g_im = grad_re_im(jnp.real(u), jnp.imag(u), phi)
            return (g_re - 1j * g_im).astype(self.model.dtype)

        return grad_links

    def action_grad_links_autodiff(self, q: Array, phi: Array) -> Array:
        if self._grad_links_ad_fn is None:
            self._grad_links_ad_fn = self._build_grad_links_ad_fn()
        return self._grad_links_ad_fn(q, phi)

    def _build_force_ad_fn(self):
        grad_links = self.action_grad_links_autodiff

        def force_fn(u: Array, phi: Array) -> Array:
            dS_dU = grad_links(u, phi)
            G = jnp.swapaxes(dS_dU, -1, -2)
            return project_su2_algebra(u @ G)

        return force_fn

    def force_autodiff(self, q: Array, phi: Array) -> Array:
        if self._force_ad_fn is None:
            self._force_ad_fn = self._build_force_ad_fn()
        return self._force_ad_fn(q, phi)

    def force_analytic(self, q: Array, phi: Array) -> Array:
        if self.model.solver_form in ("split", "eo_split"):
            x_e, y_e = self.model.solve_eop_even_split_with_intermediate(q, phi)
        else:
            x_e = self.model.solve_eop_even_normal(q, phi)
            y_e = self.model.apply_eo_schur_even(q, x_e, normalized=True)
        return self.model.eop_pseudofermion_force_from_solution(q, x_e, y_e)

    def force(self, q: Array) -> Array:
        if self.phi is None:
            self.refresh(q, traj_length=1.0)
        assert self.phi is not None
        mode = str(self.force_mode).lower()
        if mode == "autodiff":
            return self.force_autodiff(q, self.phi)
        if mode == "analytic":
            return self.force_analytic(q, self.phi)
        raise ValueError(f"Unknown pseudofermion force mode: {self.force_mode}")


@dataclass
class SU2WilsonNf2(SU2YangMills):
    mass: float = 0.0
    wilson_r: float = 1.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 500
    solver_kind: str = "cg"  # {cg, bicgstab, gmres}
    solver_form: str = "normal"  # {normal, split, eo_split}; split solves D^\dagger y=phi, D x=y
    preconditioner_kind: str = "none"  # {none, jacobi}
    gmres_restart: int = 32
    gmres_solve_method: str = "batched"
    dirac_kernel: str = "optimized"  # {optimized, reference}
    include_gauge_monomial: bool = True
    include_fermion_monomial: bool = True
    fermion_monomial_kind: str = "unpreconditioned"  # {unpreconditioned, eo_preconditioned}
    gauge_timescale: int = 0
    fermion_timescale: int = 1
    pseudofermion_refresh: str = "heatbath"
    pseudofermion_gamma: Optional[float] = None
    pseudofermion_force_mode: str = "autodiff"
    smd_gamma: float = 0.3
    auto_refresh_pseudofermions: bool = True
    jit_dirac_kernels: bool = True
    jit_solvers: bool = True
    requires_trajectory_refresh: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.solver_kind = str(self.solver_kind).lower()
        self.solver_form = str(self.solver_form).lower()
        self.preconditioner_kind = str(self.preconditioner_kind).lower()
        self.dirac_kernel = str(self.dirac_kernel).lower()
        self.fermion_monomial_kind = str(self.fermion_monomial_kind).lower()
        if self.solver_kind not in ("cg", "bicgstab", "gmres"):
            raise ValueError("solver_kind must be one of: cg, bicgstab, gmres")
        if self.solver_form not in ("normal", "split", "eo_split"):
            raise ValueError("solver_form must be one of: normal, split, eo_split")
        if self.solver_form in ("split", "eo_split") and self.solver_kind == "cg":
            raise ValueError("solver_form=split/eo_split requires a non-Hermitian solver (use solver_kind=bicgstab)")
        if self.solver_kind == "bicgstab" and bicgstab is None:
            raise ValueError("solver_kind=bicgstab requested, but jax.scipy.sparse.linalg.bicgstab is unavailable")
        if self.solver_kind == "gmres" and gmres is None:
            raise ValueError("solver_kind=gmres requested, but jax.scipy.sparse.linalg.gmres is unavailable")
        if self.preconditioner_kind not in ("none", "jacobi"):
            raise ValueError("preconditioner_kind must be one of: none, jacobi")
        if self.dirac_kernel not in ("optimized", "reference"):
            raise ValueError("dirac_kernel must be one of: optimized, reference")
        if self.fermion_monomial_kind not in ("unpreconditioned", "eo_preconditioned"):
            raise ValueError("fermion_monomial_kind must be one of: unpreconditioned, eo_preconditioned")
        self.gmres_restart = int(self.gmres_restart)
        if self.gmres_restart <= 0:
            raise ValueError("gmres_restart must be positive")

        mode = str(self.pseudofermion_refresh).lower()
        if self.pseudofermion_gamma is None:
            self.pseudofermion_gamma = float(self.smd_gamma) if mode == "ou" else 0.3
        self.dirac = WilsonDiracOperator(ndim=self.Nd, mass=self.mass, wilson_r=self.wilson_r, dtype=self.dtype)
        self._m0 = float(self.mass + self.wilson_r * self.Nd)
        self._mask_even, self._mask_odd = self._build_parity_masks()
        self._n_sites = int(np.prod(self.lattice_shape))
        self._cb_even_idx, self._cb_odd_idx = self._build_checkerboard_indices()
        self._build_checkerboard_neighbor_maps()
        self.gamma = self.dirac.gamma
        self.Ns = self.dirac.Ns
        monomials = []
        if self.include_gauge_monomial:
            monomials.append(GaugeActionMonomial(self, timescale=int(self.gauge_timescale)))
        self.fermion_monomial: Optional[object]
        if self.include_fermion_monomial:
            if self.fermion_monomial_kind == "eo_preconditioned":
                self.fermion_monomial = WilsonNf2EOPreconditionedMonomial(
                    self,
                    timescale=int(self.fermion_timescale),
                    refresh_mode=str(self.pseudofermion_refresh),
                    gamma=float(self.pseudofermion_gamma),
                    force_mode=str(self.pseudofermion_force_mode),
                )
            else:
                self.fermion_monomial = WilsonNf2PseudofermionMonomial(
                    self,
                    timescale=int(self.fermion_timescale),
                    refresh_mode=str(self.pseudofermion_refresh),
                    gamma=float(self.pseudofermion_gamma),
                    force_mode=str(self.pseudofermion_force_mode),
                )
            monomials.append(self.fermion_monomial)
        else:
            self.fermion_monomial = None
        self.hamiltonian = HamiltonianModel(tuple(monomials))
        self.requires_trajectory_refresh = any(bool(getattr(m, "stochastic", False)) for m in self.hamiltonian.monomials)
        self._configure_jit_paths()

    def _configure_jit_paths(self) -> None:
        if bool(self.jit_dirac_kernels):
            _apply_D = self.apply_D
            _apply_Ddag = self.apply_Ddag
            _apply_normal = self.apply_normal
            _apply_D_dense = self.apply_D_dense
            _apply_Ddag_dense = self.apply_Ddag_dense
            _apply_normal_dense = self.apply_normal_dense
            _apply_offdiag = self.apply_offdiag
            _apply_offdiag_dagger = self.apply_offdiag_dagger
            self.apply_D = jax.jit(lambda U, psi: _apply_D(U, psi))
            self.apply_Ddag = jax.jit(lambda U, psi: _apply_Ddag(U, psi))
            self.apply_normal = jax.jit(lambda U, psi: _apply_normal(U, psi))
            self.apply_D_dense = jax.jit(lambda U, psi: _apply_D_dense(U, psi))
            self.apply_Ddag_dense = jax.jit(lambda U, psi: _apply_Ddag_dense(U, psi))
            self.apply_normal_dense = jax.jit(lambda U, psi: _apply_normal_dense(U, psi))
            self.apply_offdiag = jax.jit(lambda U, psi: _apply_offdiag(U, psi))
            self.apply_offdiag_dagger = jax.jit(lambda U, psi: _apply_offdiag_dagger(U, psi))

        if bool(self.jit_solvers):
            _solve_direct = self.solve_direct
            _solve_dagger = self.solve_dagger
            _solve_direct_eo = self.solve_direct_eo
            _solve_dagger_eo = self.solve_dagger_eo
            _solve_eop_even_direct = self.solve_eop_even_direct
            _solve_eop_even_dagger = self.solve_eop_even_dagger
            _solve_eop_even_normal = self.solve_eop_even_normal
            _solve_normal = self.solve_normal
            _solve_split_with_intermediate = self.solve_split_with_intermediate
            _solve_split = self.solve_split
            _solve_pseudofermion = self.solve_pseudofermion
            _solve_eop_even_split_with_intermediate = self.solve_eop_even_split_with_intermediate
            _solve_eop_even_split = self.solve_eop_even_split
            _solve_eop_even = self.solve_eop_even

            self.solve_direct = jax.jit(lambda U, rhs: _solve_direct(U, rhs))
            self.solve_dagger = jax.jit(lambda U, rhs: _solve_dagger(U, rhs))
            self.solve_direct_eo = jax.jit(lambda U, rhs: _solve_direct_eo(U, rhs))
            self.solve_dagger_eo = jax.jit(lambda U, rhs: _solve_dagger_eo(U, rhs))
            self.solve_eop_even_direct = jax.jit(lambda U, rhs: _solve_eop_even_direct(U, rhs))
            self.solve_eop_even_dagger = jax.jit(lambda U, rhs: _solve_eop_even_dagger(U, rhs))
            self.solve_eop_even_normal = jax.jit(lambda U, rhs: _solve_eop_even_normal(U, rhs))
            self.solve_normal = jax.jit(lambda U, rhs: _solve_normal(U, rhs))
            self.solve_split_with_intermediate = jax.jit(lambda U, phi: _solve_split_with_intermediate(U, phi))
            self.solve_split = jax.jit(lambda U, phi: _solve_split(U, phi))
            self.solve_pseudofermion = jax.jit(lambda U, phi: _solve_pseudofermion(U, phi))
            self.solve_eop_even_split_with_intermediate = jax.jit(
                lambda U, phi: _solve_eop_even_split_with_intermediate(U, phi)
            )
            self.solve_eop_even_split = jax.jit(lambda U, phi: _solve_eop_even_split(U, phi))
            self.solve_eop_even = jax.jit(lambda U, phi: _solve_eop_even(U, phi))

    def fermion_shape(self) -> Tuple[int, ...]:
        # Spin-major for cheap gamma/projector ops and contiguous color vectors.
        return (self.Bs, *self.lattice_shape, self.Ns, 2)

    def _build_parity_masks(self) -> Tuple[Array, Array]:
        parity = np.zeros(self.lattice_shape, dtype=np.int32)
        for ax, n in enumerate(self.lattice_shape):
            shp = [1] * self.Nd
            shp[ax] = n
            parity = parity + np.arange(n, dtype=np.int32).reshape(shp)
        even = (parity % 2 == 0).astype(np.float32)
        odd = 1.0 - even
        e = jnp.asarray(even.reshape((1, *self.lattice_shape, 1, 1)))
        o = jnp.asarray(odd.reshape((1, *self.lattice_shape, 1, 1)))
        return e, o

    def _build_checkerboard_indices(self) -> Tuple[Array, Array]:
        parity = np.zeros(self.lattice_shape, dtype=np.int32)
        for ax, n in enumerate(self.lattice_shape):
            shp = [1] * self.Nd
            shp[ax] = n
            parity = parity + np.arange(n, dtype=np.int32).reshape(shp)
        flat = parity.reshape(-1)
        even_idx = np.flatnonzero((flat % 2) == 0).astype(np.int32)
        odd_idx = np.flatnonzero((flat % 2) == 1).astype(np.int32)
        return jnp.asarray(even_idx), jnp.asarray(odd_idx)

    def _shift_flat_indices_np(self, flat_idx: np.ndarray, mu: int, step: int) -> np.ndarray:
        coords = np.array(np.unravel_index(flat_idx, self.lattice_shape), dtype=np.int64).T
        coords[:, mu] = (coords[:, mu] + int(step)) % int(self.lattice_shape[mu])
        return np.ravel_multi_index(coords.T, self.lattice_shape).astype(np.int32)

    def _build_checkerboard_neighbor_maps(self) -> None:
        even_idx = np.asarray(self._cb_even_idx, dtype=np.int32)
        odd_idx = np.asarray(self._cb_odd_idx, dtype=np.int32)
        vol = int(self._n_sites)
        even_inv = np.full((vol,), -1, dtype=np.int32)
        odd_inv = np.full((vol,), -1, dtype=np.int32)
        even_inv[even_idx] = np.arange(even_idx.shape[0], dtype=np.int32)
        odd_inv[odd_idx] = np.arange(odd_idx.shape[0], dtype=np.int32)

        plus_oe = []
        minus_oe = []
        bwd_oe = []
        plus_eo = []
        minus_eo = []
        bwd_eo = []
        for mu in range(self.Nd):
            odd_plus = self._shift_flat_indices_np(odd_idx, mu, +1)
            odd_minus = self._shift_flat_indices_np(odd_idx, mu, -1)
            plus_oe.append(even_inv[odd_plus])
            minus_oe.append(even_inv[odd_minus])
            bwd_oe.append(odd_minus)

            even_plus = self._shift_flat_indices_np(even_idx, mu, +1)
            even_minus = self._shift_flat_indices_np(even_idx, mu, -1)
            plus_eo.append(odd_inv[even_plus])
            minus_eo.append(odd_inv[even_minus])
            bwd_eo.append(even_minus)

        self._cb_target_oe = jnp.asarray(odd_idx)
        self._cb_target_eo = jnp.asarray(even_idx)
        self._cb_plus_oe = jnp.asarray(np.stack(plus_oe, axis=0))
        self._cb_minus_oe = jnp.asarray(np.stack(minus_oe, axis=0))
        self._cb_bwd_oe = jnp.asarray(np.stack(bwd_oe, axis=0))
        self._cb_plus_eo = jnp.asarray(np.stack(plus_eo, axis=0))
        self._cb_minus_eo = jnp.asarray(np.stack(minus_eo, axis=0))
        self._cb_bwd_eo = jnp.asarray(np.stack(bwd_eo, axis=0))

    def _project_even(self, psi: Array) -> Array:
        if psi.ndim == self.Nd + 2:
            return psi * self._mask_even[0].astype(psi.dtype)
        return psi * self._mask_even.astype(psi.dtype)

    def _project_odd(self, psi: Array) -> Array:
        if psi.ndim == self.Nd + 2:
            return psi * self._mask_odd[0].astype(psi.dtype)
        return psi * self._mask_odd.astype(psi.dtype)

    def _pack_even_sites(self, psi: Array) -> Array:
        if psi.ndim == self.Nd + 2:
            sites = psi.reshape((self._n_sites, self.Ns, 2))
            return jnp.take(sites, self._cb_even_idx, axis=0)
        if psi.ndim == self.Nd + 3:
            sites = psi.reshape((psi.shape[0], self._n_sites, self.Ns, 2))
            return jnp.take(sites, self._cb_even_idx, axis=1)
        raise ValueError(f"Unexpected fermion rank for even pack: {psi.ndim}")

    def _pack_odd_sites(self, psi: Array) -> Array:
        if psi.ndim == self.Nd + 2:
            sites = psi.reshape((self._n_sites, self.Ns, 2))
            return jnp.take(sites, self._cb_odd_idx, axis=0)
        if psi.ndim == self.Nd + 3:
            sites = psi.reshape((psi.shape[0], self._n_sites, self.Ns, 2))
            return jnp.take(sites, self._cb_odd_idx, axis=1)
        raise ValueError(f"Unexpected fermion rank for odd pack: {psi.ndim}")

    def _unpack_even_sites(self, psi_even: Array) -> Array:
        if psi_even.ndim == 3:
            sites = jnp.zeros((self._n_sites, self.Ns, 2), dtype=psi_even.dtype)
            sites = sites.at[self._cb_even_idx].set(psi_even, indices_are_sorted=True, unique_indices=True)
            return sites.reshape((*self.lattice_shape, self.Ns, 2))
        if psi_even.ndim == 4:
            sites = jnp.zeros((psi_even.shape[0], self._n_sites, self.Ns, 2), dtype=psi_even.dtype)
            sites = sites.at[:, self._cb_even_idx].set(psi_even, indices_are_sorted=True, unique_indices=True)
            return sites.reshape((psi_even.shape[0], *self.lattice_shape, self.Ns, 2))
        raise ValueError(f"Unexpected compact even rank: {psi_even.ndim}")

    def _unpack_odd_sites(self, psi_odd: Array) -> Array:
        if psi_odd.ndim == 3:
            sites = jnp.zeros((self._n_sites, self.Ns, 2), dtype=psi_odd.dtype)
            sites = sites.at[self._cb_odd_idx].set(psi_odd, indices_are_sorted=True, unique_indices=True)
            return sites.reshape((*self.lattice_shape, self.Ns, 2))
        if psi_odd.ndim == 4:
            sites = jnp.zeros((psi_odd.shape[0], self._n_sites, self.Ns, 2), dtype=psi_odd.dtype)
            sites = sites.at[:, self._cb_odd_idx].set(psi_odd, indices_are_sorted=True, unique_indices=True)
            return sites.reshape((psi_odd.shape[0], *self.lattice_shape, self.Ns, 2))
        raise ValueError(f"Unexpected compact odd rank: {psi_odd.ndim}")

    def _apply_dslash_parity_compact_one(
        self,
        Ub: Array,
        psi_src_compact: Array,
        sign: int,
        target_block: str,
    ) -> Array:
        if target_block == "oe":
            target = self._cb_target_oe
            plus = self._cb_plus_oe
            minus = self._cb_minus_oe
            bwd = self._cb_bwd_oe
        elif target_block == "eo":
            target = self._cb_target_eo
            plus = self._cb_plus_eo
            minus = self._cb_minus_eo
            bwd = self._cb_bwd_eo
        else:
            raise ValueError(f"Unknown compact target block: {target_block}")

        out = jnp.zeros_like(psi_src_compact)
        for mu in range(self.Nd):
            U_mu = self._take_mu(Ub[None, ...], mu)[0].reshape((self._n_sites, 2, 2))
            U_fwd = jnp.take(U_mu, target, axis=0)
            U_bwd = _dagger(jnp.take(U_mu, bwd[mu], axis=0))

            psi_fwd = jnp.take(psi_src_compact, plus[mu], axis=0)
            psi_bwd = jnp.take(psi_src_compact, minus[mu], axis=0)
            fwd = self.dirac.color_mul_left(U_fwd, psi_fwd)
            bkw = self.dirac.color_mul_left(U_bwd, psi_bwd)
            fwd = self.dirac.spin_project(fwd, mu, coeff=float(+sign), use_sparse=True)
            bkw = self.dirac.spin_project(bkw, mu, coeff=float(-sign), use_sparse=True)
            out = out - 0.5 * (fwd + bkw)
        return out

    def _apply_K_oe_compact_one(self, Ub: Array, psi_even_compact: Array, daggered: bool = False) -> Array:
        sign = +1 if daggered else -1
        return self._apply_dslash_parity_compact_one(Ub, psi_even_compact, sign=sign, target_block="oe")

    def _apply_K_eo_compact_one(self, Ub: Array, psi_odd_compact: Array, daggered: bool = False) -> Array:
        sign = +1 if daggered else -1
        return self._apply_dslash_parity_compact_one(Ub, psi_odd_compact, sign=sign, target_block="eo")

    def _apply_eo_schur_even_compact_one(
        self,
        Ub: Array,
        psi_even_compact: Array,
        daggered: bool = False,
        normalized: bool = True,
    ) -> Array:
        """EO Schur matvec on compact-even vectors for one gauge configuration."""
        y_odd = self._apply_K_oe_compact_one(Ub, psi_even_compact, daggered=daggered)
        zc = self._apply_K_eo_compact_one(Ub, y_odd, daggered=daggered)
        if normalized:
            return psi_even_compact - (1.0 / (self._m0 * self._m0)) * zc
        return self._m0 * psi_even_compact - (1.0 / self._m0) * zc

    def _roll_site(self, x: Array, shift: int, direction: int) -> Array:
        return jnp.roll(x, shift=shift, axis=1 + direction)

    def apply_D(self, U: Array, psi: Array) -> Array:
        return self.apply_D_with_kernel(U, psi, kernel=self.dirac_kernel, use_sparse_gamma=True)

    def apply_D_with_kernel(self, U: Array, psi: Array, kernel: str, use_sparse_gamma: bool = True) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=-1,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
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
            kernel=self.dirac_kernel,
        )

    def apply_offdiag(self, U: Array, psi: Array) -> Array:
        # K: nearest-neighbor Dslash part (Wilson off-diagonal block).
        return self.dirac.apply_dslash(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=-1,
            use_sparse_gamma=True,
            kernel=self.dirac_kernel,
        )

    def apply_offdiag_dagger(self, U: Array, psi: Array) -> Array:
        # K^\dagger: nearest-neighbor Dslash part for D^\dagger.
        return self.dirac.apply_dslash(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=+1,
            use_sparse_gamma=True,
            kernel=self.dirac_kernel,
        )

    def apply_K_oe(self, U: Array, psi_even: Array) -> Array:
        """K_oe: even -> odd nearest-neighbor hopping block."""
        xe = self._project_even(psi_even)
        return self._project_odd(self.apply_offdiag(U, xe))

    def apply_K_eo(self, U: Array, psi_odd: Array) -> Array:
        """K_eo: odd -> even nearest-neighbor hopping block."""
        xo = self._project_odd(psi_odd)
        return self._project_even(self.apply_offdiag(U, xo))

    def apply_Kdag_oe(self, U: Array, psi_even: Array) -> Array:
        """K^\\dagger_oe: even -> odd nearest-neighbor hopping block."""
        xe = self._project_even(psi_even)
        return self._project_odd(self.apply_offdiag_dagger(U, xe))

    def apply_Kdag_eo(self, U: Array, psi_odd: Array) -> Array:
        """K^\\dagger_eo: odd -> even nearest-neighbor hopping block."""
        xo = self._project_odd(psi_odd)
        return self._project_even(self.apply_offdiag_dagger(U, xo))

    def apply_normal(self, U: Array, psi: Array) -> Array:
        return self.apply_normal_with_kernel(U, psi, kernel=self.dirac_kernel, use_sparse_gamma=True)

    def apply_normal_with_kernel(self, U: Array, psi: Array, kernel: str, use_sparse_gamma: bool = True) -> Array:
        return self.dirac.apply_normal(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
        )

    def apply_eo_schur_even(self, U: Array, psi_even: Array, normalized: bool = False) -> Array:
        # Schur complement on even sites:
        # S_e = m0 I - (1/m0) K_eo K_oe, with K = D - m0 I.
        xe = self._project_even(psi_even)
        y_odd = self.apply_K_oe(U, xe)
        z_even = self.apply_K_eo(U, y_odd)
        if normalized:
            return xe - (1.0 / (self._m0 * self._m0)) * z_even
        return self._m0 * xe - (1.0 / self._m0) * z_even

    def apply_eo_schur_even_dagger(self, U: Array, psi_even: Array, normalized: bool = False) -> Array:
        # S_e^\dagger = m0 I - (1/m0) K_eo^\dagger K_oe^\dagger.
        xe = self._project_even(psi_even)
        y_odd = self.apply_Kdag_oe(U, xe)
        z_even = self.apply_Kdag_eo(U, y_odd)
        if normalized:
            return xe - (1.0 / (self._m0 * self._m0)) * z_even
        return self._m0 * xe - (1.0 / self._m0) * z_even

    def apply_eo_schur_odd(self, U: Array, psi_odd: Array, normalized: bool = False) -> Array:
        xo = self._project_odd(psi_odd)
        y_even = self.apply_K_eo(U, xo)
        z_odd = self.apply_K_oe(U, y_even)
        if normalized:
            return xo - (1.0 / (self._m0 * self._m0)) * z_odd
        return self._m0 * xo - (1.0 / self._m0) * z_odd

    def apply_D_dense(self, U: Array, psi: Array) -> Array:
        return self.apply_D_with_kernel(U, psi, kernel=self.dirac_kernel, use_sparse_gamma=False)

    def apply_Ddag_dense(self, U: Array, psi: Array) -> Array:
        return self.dirac.apply(
            U,
            psi,
            take_mu=self._take_mu,
            roll_site=self._roll_site,
            dagger=_dagger,
            sign=+1,
            use_sparse_gamma=False,
            kernel=self.dirac_kernel,
        )

    def apply_normal_dense(self, U: Array, psi: Array) -> Array:
        return self.apply_normal_with_kernel(U, psi, kernel=self.dirac_kernel, use_sparse_gamma=False)

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
        if self.solver_form == "normal":
            x = self.solve_normal(U, phi)
            return jnp.real(jax.vmap(_vdot_field, in_axes=(0, 0))(phi, x))
        if self.solver_form in ("split", "eo_split"):
            # Split identity:
            # S = phi^\dag (D^\dag D)^-1 phi, phi = D^\dag chi  =>  S = chi^\dag chi,
            # with a single solve D^\dag chi = phi.
            if self.solver_form == "eo_split":
                chi = self.solve_dagger_eo(U, phi)
            else:
                chi = self.solve_dagger(U, phi)
            return jnp.real(jax.vmap(_vdot_field, in_axes=(0, 0))(chi, chi))
        raise ValueError(f"Unknown solver_form: {self.solver_form}")

    def eop_pseudofermion_action(self, U: Array, phi: Array) -> Array:
        phi = self._project_even(phi)
        if self.solver_form == "normal":
            x = self.solve_eop_even_normal(U, phi)
            return jnp.real(jax.vmap(_vdot_field, in_axes=(0, 0))(phi, x))
        if self.solver_form in ("split", "eo_split"):
            # Split identity on EO-preconditioned Schur operator M:
            # S = phi^\dag (M^\dag M)^-1 phi, phi = M^\dag eta => S = eta^\dag eta.
            chi = self.solve_eop_even_dagger(U, phi)
            return jnp.real(jax.vmap(_vdot_field, in_axes=(0, 0))(chi, chi))
        raise ValueError(f"Unknown solver_form: {self.solver_form}")

    @staticmethod
    def _supports_kwarg(fn, name: str) -> bool:
        try:
            return name in inspect.signature(fn).parameters
        except Exception:
            return False

    def _solver_fn(self):
        if self.solver_kind == "cg":
            return cg
        if self.solver_kind == "bicgstab":
            if bicgstab is None:
                raise RuntimeError("bicgstab is unavailable in this JAX build")
            return bicgstab
        if self.solver_kind == "gmres":
            if gmres is None:
                raise RuntimeError("gmres is unavailable in this JAX build")
            return gmres
        raise ValueError(f"Unknown solver_kind: {self.solver_kind}")

    def _preconditioner(self, operator_tag: str):
        kind = self.preconditioner_kind
        if kind == "none":
            return None
        if kind == "jacobi":
            m0 = float(self.mass + self.wilson_r * self.Nd)
            if operator_tag == "normal":
                inv = 1.0 / (m0 * m0 + 1e-20)
            else:
                inv = 1.0 / (m0 + 1e-20)
            return lambda x: (inv * x).astype(self.dtype)
        raise ValueError(f"Unknown preconditioner_kind: {kind}")

    def _solve_linear(self, A, b, operator_tag: str = "normal"):
        solver = self._solver_fn()
        kwargs = {"tol": self.cg_tol, "maxiter": self.cg_maxiter}
        if self._supports_kwarg(solver, "atol"):
            kwargs["atol"] = 0.0
        if self.solver_kind == "gmres":
            if self._supports_kwarg(solver, "restart"):
                kwargs["restart"] = int(self.gmres_restart)
            if self._supports_kwarg(solver, "solve_method"):
                kwargs["solve_method"] = str(self.gmres_solve_method)

        M = self._preconditioner(operator_tag)
        if M is not None and self._supports_kwarg(solver, "M"):
            kwargs["M"] = M
            return solver(A, b, **kwargs)

        if M is not None:
            # Generic left-preconditioned fallback when solver API has no explicit M hook.
            def A_left(x):
                return M(A(x))

            b_left = M(b)
            return solver(A_left, b_left, **kwargs)

        return solver(A, b, **kwargs)

    def solve_direct(self, U: Array, rhs: Array) -> Array:
        def solve_one(Ub, pb):
            def A(x):
                xb = x[None, ...]
                yb = self.apply_D(Ub[None, ...], xb)
                return yb[0]

            xb, _ = self._solve_linear(A, pb, operator_tag="direct")
            return xb

        return jax.vmap(solve_one, in_axes=(0, 0))(U, rhs)

    def solve_dagger(self, U: Array, rhs: Array) -> Array:
        def solve_one(Ub, pb):
            def A_dag(x):
                xb = x[None, ...]
                yb = self.apply_Ddag(Ub[None, ...], xb)
                return yb[0]

            yb, _ = self._solve_linear(A_dag, pb, operator_tag="dagger")
            return yb

        return jax.vmap(solve_one, in_axes=(0, 0))(U, rhs)

    def _solve_eo_one(self, Ub: Array, pb: Array, which: str) -> Array:
        if which == "direct":
            def K(v):
                vb = v[None, ...]
                yb = self.apply_D(Ub[None, ...], vb)
                return yb[0] - self._m0 * v

            operator_tag = "eo_direct"
        elif which == "dagger":
            def K(v):
                vb = v[None, ...]
                yb = self.apply_Ddag(Ub[None, ...], vb)
                return yb[0] - self._m0 * v

            operator_tag = "eo_dagger"
        else:
            raise ValueError(f"Unknown EO solve kind: {which}")

        b_even = self._project_even(pb)
        b_odd = self._project_odd(pb)
        rhs_even = b_even - (1.0 / self._m0) * self._project_even(K(b_odd))

        def A_even(x):
            xe = self._project_even(x)
            y_odd = self._project_odd(K(xe))
            z_even = self._project_even(K(y_odd))
            se = self._m0 * xe - (1.0 / self._m0) * z_even
            # Keep identity on odd subspace to avoid singular masked system.
            return self._project_even(se) + self._project_odd(x)

        rhs_aug = self._project_even(rhs_even)
        x_even, _ = self._solve_linear(A_even, rhs_aug, operator_tag=operator_tag)
        x_odd = (1.0 / self._m0) * (b_odd - self._project_odd(K(x_even)))
        return self._project_even(x_even) + self._project_odd(x_odd)

    def solve_direct_eo(self, U: Array, rhs: Array) -> Array:
        return jax.vmap(lambda Ub, pb: self._solve_eo_one(Ub, pb, "direct"), in_axes=(0, 0))(U, rhs)

    def solve_dagger_eo(self, U: Array, rhs: Array) -> Array:
        return jax.vmap(lambda Ub, pb: self._solve_eo_one(Ub, pb, "dagger"), in_axes=(0, 0))(U, rhs)

    def solve_eop_even_direct(self, U: Array, rhs: Array) -> Array:
        def solve_one(Ub, pb):
            pb = self._pack_even_sites(self._project_even(pb))

            def A(x):
                return self._apply_eo_schur_even_compact_one(Ub, x, daggered=False, normalized=True)

            xb, _ = self._solve_linear(A, pb, operator_tag="eop_direct")
            return self._unpack_even_sites(xb)

        return jax.vmap(solve_one, in_axes=(0, 0))(U, rhs)

    def solve_eop_even_dagger(self, U: Array, rhs: Array) -> Array:
        def solve_one(Ub, pb):
            pb = self._pack_even_sites(self._project_even(pb))

            def A(x):
                return self._apply_eo_schur_even_compact_one(Ub, x, daggered=True, normalized=True)

            xb, _ = self._solve_linear(A, pb, operator_tag="eop_dagger")
            return self._unpack_even_sites(xb)

        return jax.vmap(solve_one, in_axes=(0, 0))(U, rhs)

    def solve_eop_even_normal(self, U: Array, rhs: Array) -> Array:
        def solve_one(Ub, pb):
            pb = self._pack_even_sites(self._project_even(pb))

            def A(x):
                y = self._apply_eo_schur_even_compact_one(Ub, x, daggered=False, normalized=True)
                z = self._apply_eo_schur_even_compact_one(Ub, y, daggered=True, normalized=True)
                return z

            xb, _ = self._solve_linear(A, pb, operator_tag="eop_normal")
            return self._unpack_even_sites(xb)

        return jax.vmap(solve_one, in_axes=(0, 0))(U, rhs)

    def solve_eop_even_split_with_intermediate(self, U: Array, phi: Array) -> Tuple[Array, Array]:
        y = self.solve_eop_even_dagger(U, phi)
        x = self.solve_eop_even_direct(U, y)
        return x, y

    def solve_eop_even_split(self, U: Array, phi: Array) -> Array:
        x, _ = self.solve_eop_even_split_with_intermediate(U, phi)
        return x

    def solve_eop_even(self, U: Array, phi: Array) -> Array:
        if self.solver_form == "normal":
            return self.solve_eop_even_normal(U, phi)
        if self.solver_form in ("split", "eo_split"):
            return self.solve_eop_even_split(U, phi)
        raise ValueError(f"Unknown solver_form: {self.solver_form}")

    def solve_normal(self, U: Array, phi: Array) -> Array:
        def solve_one(Ub, pb):
            def A(x):
                xb = x[None, ...]
                yb = self.apply_normal(Ub[None, ...], xb)
                return yb[0]

            xb, _ = self._solve_linear(A, pb, operator_tag="normal")
            return xb

        return jax.vmap(solve_one, in_axes=(0, 0))(U, phi)

    def solve_split_with_intermediate(self, U: Array, phi: Array) -> Tuple[Array, Array]:
        if self.solver_form == "eo_split":
            y = self.solve_dagger_eo(U, phi)
            x = self.solve_direct_eo(U, y)
        else:
            y = self.solve_dagger(U, phi)
            x = self.solve_direct(U, y)
        return x, y

    def solve_split(self, U: Array, phi: Array) -> Array:
        x, _ = self.solve_split_with_intermediate(U, phi)
        return x

    def solve_pseudofermion(self, U: Array, phi: Array) -> Array:
        if self.solver_form == "normal":
            return self.solve_normal(U, phi)
        if self.solver_form == "split":
            return self.solve_split(U, phi)
        if self.solver_form == "eo_split":
            return self.solve_split(U, phi)
        raise ValueError(f"Unknown solver_form: {self.solver_form}")

    def normal_residual_stats(self, U: Array, phi: Array, x: Array) -> Dict[str, float]:
        """Residual diagnostics for solved (D^dag D) x = phi."""
        r = self.apply_normal(U, x) - phi
        rel = jax.vmap(lambda rb, pb: _field_norm(rb) / (_field_norm(pb) + 1e-30), in_axes=(0, 0))(r, phi)
        absn = jax.vmap(_field_norm, in_axes=0)(r)
        return {
            "mean_rel_residual": float(jnp.mean(rel)),
            "max_rel_residual": float(jnp.max(rel)),
            "mean_abs_residual": float(jnp.mean(absn)),
            "max_abs_residual": float(jnp.max(absn)),
        }

    def solver_info_stats(self, U: Array, phi: Array) -> Dict[str, float]:
        """Best-effort extraction of per-RHS iteration counts from solver info."""
        iters = []
        it1 = []
        it2 = []

        for b in range(int(U.shape[0])):
            Ub = U[b]
            pb = phi[b]

            if self.solver_form == "normal":
                def A(x):
                    xb = x[None, ...]
                    yb = self.apply_normal(Ub[None, ...], xb)
                    return yb[0]

                _, info = self._solve_linear(A, pb, operator_tag="normal")
                iters.append(_extract_cg_num_iters(info))
            else:
                if self.solver_form == "eo_split":
                    def A_dag(x):
                        xb = x[None, ...]
                        yb = self.apply_Ddag(Ub[None, ...], xb)
                        return yb[0]

                    yb, info1 = self._solve_linear(A_dag, pb, operator_tag="eo_dagger")

                    def A(x):
                        xb = x[None, ...]
                        yb2 = self.apply_D(Ub[None, ...], xb)
                        return yb2[0]

                    _, info2 = self._solve_linear(A, yb, operator_tag="eo_direct")
                else:
                    def A_dag(x):
                        xb = x[None, ...]
                        yb = self.apply_Ddag(Ub[None, ...], xb)
                        return yb[0]

                    yb, info1 = self._solve_linear(A_dag, pb, operator_tag="dagger")

                    def A(x):
                        xb = x[None, ...]
                        yb2 = self.apply_D(Ub[None, ...], xb)
                        return yb2[0]

                    _, info2 = self._solve_linear(A, yb, operator_tag="direct")
                k1 = _extract_cg_num_iters(info1)
                k2 = _extract_cg_num_iters(info2)
                it1.append(k1)
                it2.append(k2)
                if np.isfinite(k1) and np.isfinite(k2):
                    iters.append(float(k1 + k2))
                else:
                    iters.append(float("nan"))

        it = np.asarray(iters, dtype=np.float64)
        known = np.isfinite(it)
        known_frac = float(np.mean(known.astype(np.float64))) if it.size > 0 else 0.0
        mean1 = float("nan")
        mean2 = float("nan")
        if self.solver_form in ("split", "eo_split"):
            a1 = np.asarray(it1, dtype=np.float64)
            a2 = np.asarray(it2, dtype=np.float64)
            if np.any(np.isfinite(a1)):
                mean1 = float(np.nanmean(a1))
            if np.any(np.isfinite(a2)):
                mean2 = float(np.nanmean(a2))
        if np.any(known):
            known_vals = it[known]
            return {
                "iters_known_fraction": known_frac,
                "mean_solver_iters": float(np.mean(known_vals)),
                "max_solver_iters": float(np.max(known_vals)),
                "mean_solver_iters_stage1": mean1,
                "mean_solver_iters_stage2": mean2,
            }
        return {
            "iters_known_fraction": known_frac,
            "mean_solver_iters": float("nan"),
            "max_solver_iters": float("nan"),
            "mean_solver_iters_stage1": mean1,
            "mean_solver_iters_stage2": mean2,
        }

    def pseudofermion_force_from_solution(self, U: Array, X: Array, Y: Array) -> Array:
        links_force = []
        for mu in range(self.Nd):
            U_mu = self._take_mu(U, mu)
            X_xpmu = self._roll_site(X, -1, mu)
            Y_xpmu = self._roll_site(Y, -1, mu)
            UX = self.dirac.color_mul_left(U_mu, X_xpmu)
            Udag = _dagger(U_mu)

            p_fwd = self.wilson_r * self.dirac.spin_eye - self.gamma[mu]
            p_bwd = self.wilson_r * self.dirac.spin_eye + self.gamma[mu]

            c1 = jnp.einsum("...sa,st,...tc->...ac", jnp.conj(Y), p_fwd, UX)
            c2 = jnp.einsum("...sa,st,...ad,...tc->...dc", jnp.conj(Y_xpmu), p_bwd, Udag, X)
            k = jnp.swapaxes(c1 - c2, -1, -2)
            links_force.append(project_su2_algebra(k))

        if self.layout == "BMXYIJ":
            return jnp.stack(links_force, axis=1)
        return jnp.stack(links_force, axis=1 + self.Nd)

    def eop_pseudofermion_force_from_solution(self, U: Array, X_e: Array, Y_e: Array) -> Array:
        """EO-preconditioned force from Schur solutions.

        For M = I - alpha K_eo K_oe (alpha=1/m0^2):
          S = phi_e^dag (M^dag M)^-1 phi_e
          dS = -2 Re[ Y_e^dag (dM) X_e ]
             = +2 alpha Re[ Y_e^dag (dK_eo) (K_oe X_e) ]
             +2 alpha Re[ (K_eo^dag Y_e)^dag (dK_oe) X_e ].
        Each term is mapped to the standard Wilson dslash force kernel
        dS_term = -<F_term,H> with force_from_solution(X_term, Y_term),
        using Y_term = -alpha * (...) to match signs.
        """
        alpha = 1.0 / (self._m0 * self._m0)
        k_oe_x = self.apply_K_oe(U, X_e)            # odd
        kdag_oe_y = self.apply_Kdag_oe(U, Y_e)      # odd

        f1 = self.pseudofermion_force_from_solution(U, k_oe_x, -alpha * Y_e)
        f2 = self.pseudofermion_force_from_solution(U, X_e, -alpha * kdag_oe_y)
        return f1 + f2

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


def _identity_links(th: SU2WilsonNf2) -> Array:
    eye = jnp.eye(2, dtype=th.dtype)
    if th.layout == "BMXYIJ":
        shape = (th.Bs, th.Nd, *th.lattice_shape, 2, 2)
    else:
        shape = (th.Bs, *th.lattice_shape, th.Nd, 2, 2)
    return jnp.broadcast_to(eye, shape)


def _gamma5_apply(g5: Array, psi: Array) -> Array:
    # psi: [..., Ns, Nc]
    return jnp.einsum("st,...tc->...sc", g5, psi)


def test_dirac_adjoint(th: SU2WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    chi = th.random_fermion()
    lhs = _vdot_field(chi, th.apply_D(U, psi))
    rhs = _vdot_field(th.apply_Ddag(U, chi), psi)
    rel = jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12)
    return {"rel_adjoint_error": float(rel), "lhs_real": float(jnp.real(lhs)), "rhs_real": float(jnp.real(rhs))}


def test_normal_positivity(th: SU2WilsonNf2) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    npsi = th.apply_normal(U, psi)
    val = jnp.real(_vdot_field(psi, npsi))
    return {"normal_quadratic_form": float(val)}


def test_gamma_sparse_dense(th: SU2WilsonNf2) -> Dict[str, float]:
    if not th.dirac.sparse_gamma_available:
        return {"sparse_available": 0.0, "max_rel_error": float("nan")}
    psi = th.random_fermion()
    out = th.dirac.gamma_sparse_dense_error(psi)
    return {"sparse_available": out["sparse_available"], "max_rel_error": out["max_rel_error"]}


def test_perf(th: SU2WilsonNf2, n_iter: int = 10) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    active = str(th.dirac_kernel).lower()
    other = "reference" if active == "optimized" else "optimized"

    d_sparse = th.apply_D(U, psi)
    d_dense = th.apply_D_dense(U, psi)
    n_sparse = th.apply_normal(U, psi)
    n_dense = th.apply_normal_dense(U, psi)
    d_other = th.apply_D_with_kernel(U, psi, kernel=other, use_sparse_gamma=True)
    n_other = th.apply_normal_with_kernel(U, psi, kernel=other, use_sparse_gamma=True)
    d_sparse.block_until_ready()
    d_dense.block_until_ready()
    n_sparse.block_until_ready()
    n_dense.block_until_ready()
    d_other.block_until_ready()
    n_other.block_until_ready()

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

    t8 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_D_with_kernel(U, psi, kernel=other, use_sparse_gamma=True).block_until_ready()
    t9 = time.perf_counter()

    t10 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_normal_with_kernel(U, psi, kernel=other, use_sparse_gamma=True).block_until_ready()
    t11 = time.perf_counter()

    rel_d = float(jnp.linalg.norm(d_sparse - d_dense) / (jnp.linalg.norm(d_dense) + 1e-12))
    rel_n = float(jnp.linalg.norm(n_sparse - n_dense) / (jnp.linalg.norm(n_dense) + 1e-12))
    rel_d_kernel = float(jnp.linalg.norm(d_sparse - d_other) / (jnp.linalg.norm(d_other) + 1e-12))
    rel_n_kernel = float(jnp.linalg.norm(n_sparse - n_other) / (jnp.linalg.norm(n_other) + 1e-12))
    d_active_t = (t1 - t0) / max(1, n_iter)
    n_active_t = (t5 - t4) / max(1, n_iter)
    d_other_t = (t9 - t8) / max(1, n_iter)
    n_other_t = (t11 - t10) / max(1, n_iter)

    vol = int(np.prod(th.lattice_shape))
    bsz = int(th.batch_size)
    nc = 2
    flops_site_sparse = int(th.dirac.flops_per_site_matvec(nc=nc, use_sparse_gamma=True, include_diagonal=True))
    flops_site_dense = int(th.dirac.flops_per_site_matvec(nc=nc, use_sparse_gamma=False, include_diagonal=True))
    flops_d_sparse = int(flops_site_sparse * vol * bsz)
    flops_d_dense = int(flops_site_dense * vol * bsz)
    flops_n_sparse = int(2 * flops_d_sparse)
    flops_n_dense = int(2 * flops_d_dense)

    def _gflops(flops: int, seconds: float) -> float:
        return float(flops) / (float(seconds) * 1e9 + 1e-30)

    return {
        "active_kernel": 0.0 if active == "reference" else 1.0,
        "other_kernel": 0.0 if other == "reference" else 1.0,
        "sparse_enabled": float(1.0 if th.dirac.sparse_gamma_available else 0.0),
        "flops_per_site_D_sparse": float(flops_site_sparse),
        "flops_per_site_D_dense": float(flops_site_dense),
        "flops_per_call_D_sparse": float(flops_d_sparse),
        "flops_per_call_D_dense": float(flops_d_dense),
        "flops_per_call_N_sparse": float(flops_n_sparse),
        "flops_per_call_N_dense": float(flops_n_dense),
        "D_sparse_sec_per_call": d_active_t,
        "D_dense_sec_per_call": (t3 - t2) / max(1, n_iter),
        "N_sparse_sec_per_call": n_active_t,
        "N_dense_sec_per_call": (t7 - t6) / max(1, n_iter),
        "D_sparse_gflops": _gflops(flops_d_sparse, d_active_t),
        "D_dense_gflops": _gflops(flops_d_dense, (t3 - t2) / max(1, n_iter)),
        "N_sparse_gflops": _gflops(flops_n_sparse, n_active_t),
        "N_dense_gflops": _gflops(flops_n_dense, (t7 - t6) / max(1, n_iter)),
        "rel_D_sparse_vs_dense": rel_d,
        "rel_N_sparse_vs_dense": rel_n,
        "D_other_kernel_sec_per_call": d_other_t,
        "N_other_kernel_sec_per_call": n_other_t,
        "D_other_kernel_gflops": _gflops(flops_d_sparse, d_other_t),
        "N_other_kernel_gflops": _gflops(flops_n_sparse, n_other_t),
        "D_active_over_other": d_active_t / (d_other_t + 1e-16),
        "N_active_over_other": n_active_t / (n_other_t + 1e-16),
        "rel_D_active_vs_other_kernel": rel_d_kernel,
        "rel_N_active_vs_other_kernel": rel_n_kernel,
    }


def test_eo_operator(th: SU2WilsonNf2, n_iter: int = 10) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    xe = th._project_even(psi)
    xo = th._project_odd(psi)

    se = th.apply_eo_schur_even(U, xe, normalized=False)
    so = th.apply_eo_schur_odd(U, xo, normalized=False)
    se_n = th.apply_eo_schur_even(U, xe, normalized=True)
    so_n = th.apply_eo_schur_odd(U, xo, normalized=True)
    xe_c = th._pack_even_sites(xe)
    se_c = jax.vmap(
        lambda Ub, xcb: th._apply_eo_schur_even_compact_one(Ub, xcb, daggered=False, normalized=False),
        in_axes=(0, 0),
    )(U, xe_c)
    se_c_n = jax.vmap(
        lambda Ub, xcb: th._apply_eo_schur_even_compact_one(Ub, xcb, daggered=False, normalized=True),
        in_axes=(0, 0),
    )(U, xe_c)
    se_c_full = th._unpack_even_sites(se_c)
    se_c_n_full = th._unpack_even_sites(se_c_n)
    se.block_until_ready()
    so.block_until_ready()
    se_n.block_until_ready()
    so_n.block_until_ready()
    se_c_full.block_until_ready()
    se_c_n_full.block_until_ready()

    # Reference Schur construction from full D block action.
    k_oe_xe = th._project_odd(th.apply_D(U, xe))
    k_eo_k_oe_xe = th._project_even(th.apply_D(U, k_oe_xe))
    se_ref = th._m0 * xe - (1.0 / th._m0) * k_eo_k_oe_xe

    k_eo_xo = th._project_even(th.apply_D(U, xo))
    k_oe_k_eo_xo = th._project_odd(th.apply_D(U, k_eo_xo))
    so_ref = th._m0 * xo - (1.0 / th._m0) * k_oe_k_eo_xo

    rel_se = float(jnp.linalg.norm(se - se_ref) / (jnp.linalg.norm(se_ref) + 1e-12))
    rel_so = float(jnp.linalg.norm(so - so_ref) / (jnp.linalg.norm(so_ref) + 1e-12))
    rel_se_c = float(jnp.linalg.norm(se_c_full - se_ref) / (jnp.linalg.norm(se_ref) + 1e-12))

    rel_se_n = float(jnp.linalg.norm(se_n - se_ref / th._m0) / (jnp.linalg.norm(se_ref / th._m0) + 1e-12))
    rel_so_n = float(jnp.linalg.norm(so_n - so_ref / th._m0) / (jnp.linalg.norm(so_ref / th._m0) + 1e-12))
    rel_se_c_n = float(
        jnp.linalg.norm(se_c_n_full - se_ref / th._m0) / (jnp.linalg.norm(se_ref / th._m0) + 1e-12)
    )

    # Timing comparison for operator applications.
    eo_compact_apply = jax.jit(
        lambda Uin, xin: jax.vmap(
            lambda Ub, xcb: th._apply_eo_schur_even_compact_one(Ub, xcb, daggered=False, normalized=False),
            in_axes=(0, 0),
        )(Uin, xin)
    )
    eo_compact_apply(U, xe_c).block_until_ready()
    # Warm normal operator on even-field shape so timing excludes compilation.
    th.apply_normal(U, xe).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_eo_schur_even(U, xe, normalized=False).block_until_ready()
    t1 = time.perf_counter()
    t1c0 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        eo_compact_apply(U, xe_c).block_until_ready()
    t1c1 = time.perf_counter()
    t2 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th.apply_normal(U, xe).block_until_ready()
    t3 = time.perf_counter()

    return {
        "rel_even_schur_vs_ref": rel_se,
        "rel_odd_schur_vs_ref": rel_so,
        "rel_even_schur_compact_vs_ref": rel_se_c,
        "rel_even_norm_schur_vs_ref": rel_se_n,
        "rel_odd_norm_schur_vs_ref": rel_so_n,
        "rel_even_norm_schur_compact_vs_ref": rel_se_c_n,
        "eo_even_sec_per_call": (t1 - t0) / max(1, n_iter),
        "eo_even_compact_sec_per_call": (t1c1 - t1c0) / max(1, n_iter),
        "normal_even_sec_per_call": (t3 - t2) / max(1, n_iter),
        "eo_over_normal_even": ((t1 - t0) / max(1, n_iter)) / (((t3 - t2) / max(1, n_iter)) + 1e-16),
        "eo_compact_over_full_even": ((t1c1 - t1c0) / max(1, n_iter)) / (((t1 - t0) / max(1, n_iter)) + 1e-16),
    }


def test_hamiltonian_monomials(th: SU2WilsonNf2) -> Dict[str, float]:
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


def test_pseudofermion_refresh(th: SU2WilsonNf2, traj_length: float = 1.0, n_refresh: int = 8) -> Dict[str, float]:
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
    th: SU2WilsonNf2,
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


def test_dirac_conventions(th: SU2WilsonNf2) -> Dict[str, float]:
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
        # Matches apply(..., sign=-1) in Chroma convention:
        # forward coeff=-1, backward coeff=+1.
        fwd = th.dirac.spin_project(psi_xpmu, mu, coeff=-1, use_sparse=False)
        bwd = th.dirac.spin_project(psi_xmmu, mu, coeff=+1, use_sparse=False)
        D_ref = D_ref - 0.5 * (fwd + bwd)
    rel_free = float(jnp.linalg.norm(D_num - D_ref) / (jnp.linalg.norm(D_ref) + 1e-12))

    return {
        "rel_gamma5_hermiticity": g5_rel,
        "rel_freefield_stencil": rel_free,
    }


def test_pseudofermion_refresh_identity(
    th: SU2WilsonNf2,
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
        Spf = m.action(U)
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


def test_pseudofermion_force_compare(
    th: SU2WilsonNf2,
    q_scale: float = 0.05,
    n_trials: int = 2,
    n_iter_timing: int = 2,
) -> Dict[str, float]:
    if th.fermion_monomial is None:
        return {"enabled": 0.0}
    if th.fermion_monomial_kind != "unpreconditioned":
        return {"enabled": 0.0}
    m = th.fermion_monomial
    rels = []
    max_abs = []
    rel_dir = []
    mean_dir_auto = []
    mean_dir_force = []
    t_ad = []
    t_an = []
    t_an_solve = []
    t_an_kernel = []
    t_mv_normal = []
    t_mv_d = []
    t_mv_ddag = []
    solve_op_apply_equiv = []
    res_rel = []
    res_abs = []
    cg_known = []
    cg_it_mean = []
    cg_it_max = []
    solver_it_stage1 = []
    solver_it_stage2 = []

    for _ in range(max(1, int(n_trials))):
        U = th.hot_start(scale=q_scale)
        m.refresh(U, traj_length=1.0)
        assert m.phi is not None
        phi = m.phi

        if th.solver_form in ("split", "eo_split"):
            x, y = th.solve_split_with_intermediate(U, phi)
        else:
            x = th.solve_pseudofermion(U, phi)
            y = th.apply_D(U, x)
        x.block_until_ready()
        F_an = th.pseudofermion_force_from_solution(U, x, y)
        F_an.block_until_ready()

        rs = th.normal_residual_stats(U, phi, x)
        res_rel.append(rs["mean_rel_residual"])
        res_abs.append(rs["mean_abs_residual"])
        cg = th.solver_info_stats(U, phi)
        cg_known.append(cg["iters_known_fraction"])
        cg_it_mean.append(cg["mean_solver_iters"])
        cg_it_max.append(cg["max_solver_iters"])
        solver_it_stage1.append(cg["mean_solver_iters_stage1"])
        solver_it_stage2.append(cg["mean_solver_iters_stage2"])

        F_ad = m.force_autodiff(U, phi)
        F_ad.block_until_ready()

        dS_dU = m.action_grad_links_autodiff(U, phi)
        d = F_an - F_ad
        rels.append(float(jnp.linalg.norm(d) / (jnp.linalg.norm(F_ad) + 1e-12)))
        max_abs.append(float(jnp.max(jnp.abs(d))))

        # Lie-directional consistency for left variation U(eps) = exp(eps H) U.
        H = th.refresh_p()
        dU = H @ U
        dS_auto = jnp.real(jnp.sum(dS_dU * dU, axis=tuple(range(1, dU.ndim))))
        dS_force = -_algebra_inner(F_an, H)
        dS_auto_mean = jnp.mean(dS_auto)
        dS_force_mean = jnp.mean(dS_force)
        rdir = jnp.abs(dS_auto_mean - dS_force_mean) / (jnp.abs(dS_auto_mean) + 1e-12)
        rel_dir.append(float(rdir))
        mean_dir_auto.append(float(dS_auto_mean))
        mean_dir_force.append(float(dS_force_mean))

        nit = max(1, int(n_iter_timing))

        tm0 = time.perf_counter()
        for _ in range(nit):
            th.apply_normal(U, phi).block_until_ready()
        tm1 = time.perf_counter()
        mv_n = (tm1 - tm0) / nit
        t_mv_normal.append(mv_n)

        td0 = time.perf_counter()
        for _ in range(nit):
            th.apply_D(U, phi).block_until_ready()
        td1 = time.perf_counter()
        mv_d = (td1 - td0) / nit
        t_mv_d.append(mv_d)

        tdd0 = time.perf_counter()
        for _ in range(nit):
            th.apply_Ddag(U, phi).block_until_ready()
        tdd1 = time.perf_counter()
        mv_dag = (tdd1 - tdd0) / nit
        t_mv_ddag.append(mv_dag)

        t0 = time.perf_counter()
        for _ in range(nit):
            m.force_autodiff(U, phi).block_until_ready()
        t1 = time.perf_counter()
        ta = 0.0
        tk = 0.0
        for _ in range(nit):
            s0 = time.perf_counter()
            if th.solver_form in ("split", "eo_split"):
                xt, yt = th.solve_split_with_intermediate(U, phi)
            else:
                xt = th.solve_pseudofermion(U, phi)
                yt = th.apply_D(U, xt)
            xt.block_until_ready()
            s1 = time.perf_counter()
            Ft = th.pseudofermion_force_from_solution(U, xt, yt)
            Ft.block_until_ready()
            s2 = time.perf_counter()
            ta += (s1 - s0)
            tk += (s2 - s1)
        t3 = time.perf_counter()
        t_ad.append((t1 - t0) / nit)
        t_an.append((t3 - t1) / nit)
        solve_t = ta / nit
        t_an_solve.append(solve_t)
        t_an_kernel.append(tk / nit)
        if th.solver_form == "normal":
            solve_op_apply_equiv.append(solve_t / (mv_n + 1e-16))
        else:
            mv_eff = 0.5 * (mv_d + mv_dag)
            solve_op_apply_equiv.append(solve_t / (mv_eff + 1e-16))

    cg_mean = float("nan")
    cg_max = float("nan")
    cg_vals = np.asarray(cg_it_mean, dtype=np.float64)
    cg_max_vals = np.asarray(cg_it_max, dtype=np.float64)
    s1_vals = np.asarray(solver_it_stage1, dtype=np.float64)
    s2_vals = np.asarray(solver_it_stage2, dtype=np.float64)
    if np.any(np.isfinite(cg_vals)):
        cg_mean = float(np.nanmean(cg_vals))
    if np.any(np.isfinite(cg_max_vals)):
        cg_max = float(np.nanmax(cg_max_vals))
    s1_mean = float(np.nanmean(s1_vals)) if np.any(np.isfinite(s1_vals)) else float("nan")
    s2_mean = float(np.nanmean(s2_vals)) if np.any(np.isfinite(s2_vals)) else float("nan")

    return {
        "enabled": 1.0,
        "max_rel_force_diff": float(np.max(rels)),
        "mean_rel_force_diff": float(np.mean(rels)),
        "max_abs_force_diff": float(np.max(max_abs)),
        "max_rel_directional_diff": float(np.max(rel_dir)),
        "mean_rel_directional_diff": float(np.mean(rel_dir)),
        "mean_directional_autodiff": float(np.mean(mean_dir_auto)),
        "mean_directional_force": float(np.mean(mean_dir_force)),
        "mean_rel_residual": float(np.mean(res_rel)),
        "mean_abs_residual": float(np.mean(res_abs)),
        "cg_info_known_fraction": float(np.mean(cg_known)),
        "cg_info_mean_iters": cg_mean,
        "cg_info_max_iters": cg_max,
        "solver_info_known_fraction": float(np.mean(cg_known)),
        "solver_info_mean_iters": cg_mean,
        "solver_info_max_iters": cg_max,
        "solver_info_mean_iters_stage1": s1_mean,
        "solver_info_mean_iters_stage2": s2_mean,
        "autodiff_sec_per_call": float(np.mean(t_ad)),
        "analytic_sec_per_call": float(np.mean(t_an)),
        "analytic_solve_sec_per_call": float(np.mean(t_an_solve)),
        "cg_solve_sec_per_call": float(np.mean(t_an_solve)),
        "solver_solve_sec_per_call": float(np.mean(t_an_solve)),
        "analytic_force_kernel_sec_per_call": float(np.mean(t_an_kernel)),
        "normal_matvec_sec_per_call": float(np.mean(t_mv_normal)),
        "direct_matvec_sec_per_call": float(np.mean(t_mv_d)),
        "dagger_matvec_sec_per_call": float(np.mean(t_mv_ddag)),
        "solve_operator_apply_equiv": float(np.mean(solve_op_apply_equiv)),
        "analytic_over_autodiff": float(np.mean(t_an) / (np.mean(t_ad) + 1e-16)),
    }


def test_eop_force_fd(
    th: SU2WilsonNf2,
    q_scale: float = 0.05,
    n_trials: int = 2,
    eps: float = 5e-4,
    n_iter_timing: int = 2,
) -> Dict[str, float]:
    if th.fermion_monomial is None:
        return {"enabled": 0.0}
    if th.fermion_monomial_kind != "eo_preconditioned":
        return {"enabled": 0.0}
    m = th.fermion_monomial
    fd_fwd_vals = []
    fd_ctr_vals = []
    ad_vals = []
    dir_ad_vals = []
    dir_an_vals = []
    rel_fd_ad = []
    rel_fd_dir_ad = []
    rel_fd_dir_an = []
    rel_ad_vs_an_force = []
    t_action = []
    t_ad = []
    t_an = []
    for _ in range(max(1, int(n_trials))):
        U = th.hot_start(scale=q_scale)
        m.refresh(U, traj_length=1.0)
        assert m.phi is not None
        phi = m.phi
        H = th.refresh_p()
        Up = th.evolve_q(float(eps), H, U)
        Um = th.evolve_q(-float(eps), H, U)
        t0 = time.perf_counter()
        Sm = m.action(U)
        Sp = m.action(Up)
        Smn = m.action(Um)
        (Sm + Sp + Smn).block_until_ready()
        t1 = time.perf_counter()
        dS_dU = m.action_grad_links_autodiff(U, phi)
        dS_dU.block_until_ready()
        dU = H @ U
        dS_ad = jnp.mean(jnp.real(jnp.sum(dS_dU * dU, axis=tuple(range(1, dU.ndim)))))

        nit = max(1, int(n_iter_timing))
        ta0 = time.perf_counter()
        for _ in range(nit):
            m.force_autodiff(U, phi).block_until_ready()
        ta1 = time.perf_counter()
        F_ad = m.force_autodiff(U, phi)
        F_ad.block_until_ready()
        try:
            tn0 = time.perf_counter()
            for _ in range(nit):
                m.force_analytic(U, phi).block_until_ready()
            tn1 = time.perf_counter()
            F_an = m.force_analytic(U, phi)
            F_an.block_until_ready()
            has_an = True
        except Exception:
            F_an = None
            has_an = False

        fd_fwd = jnp.mean((Sp - Sm) / float(eps))
        fd_ctr = jnp.mean((Sp - Smn) / (2.0 * float(eps)))
        dir_ad = jnp.mean(-_algebra_inner(F_ad, H))

        fd_fwd_vals.append(float(fd_fwd))
        fd_ctr_vals.append(float(fd_ctr))
        ad_vals.append(float(dS_ad))
        dir_ad_vals.append(float(dir_ad))
        rel_fd_ad.append(float(jnp.abs(fd_ctr - dS_ad) / (jnp.abs(fd_ctr) + 1e-12)))
        rel_fd_dir_ad.append(float(jnp.abs(fd_ctr - dir_ad) / (jnp.abs(fd_ctr) + 1e-12)))
        t_action.append((t1 - t0) / 2.0)
        t_ad.append((ta1 - ta0) / nit)
        if has_an and F_an is not None:
            dir_an = jnp.mean(-_algebra_inner(F_an, H))
            dir_an_vals.append(float(dir_an))
            rel_fd_dir_an.append(float(jnp.abs(fd_ctr - dir_an) / (jnp.abs(fd_ctr) + 1e-12)))
            rel_ad_vs_an_force.append(float(jnp.linalg.norm(F_ad - F_an) / (jnp.linalg.norm(F_ad) + 1e-12)))
            t_an.append((tn1 - tn0) / nit)

    out = {
        "enabled": 1.0,
        "eps": float(eps),
        "mean_fd_forward": float(np.mean(fd_fwd_vals)),
        "mean_fd_central": float(np.mean(fd_ctr_vals)),
        "mean_dS_autodiff": float(np.mean(ad_vals)),
        "mean_dir_autodiff_force": float(np.mean(dir_ad_vals)),
        "max_rel_fd_central_vs_dS_autodiff": float(np.max(rel_fd_ad)),
        "mean_rel_fd_central_vs_dS_autodiff": float(np.mean(rel_fd_ad)),
        "max_rel_fd_central_vs_dir_autodiff_force": float(np.max(rel_fd_dir_ad)),
        "mean_rel_fd_central_vs_dir_autodiff_force": float(np.mean(rel_fd_dir_ad)),
        "action_sec_per_call": float(np.mean(t_action)),
        "autodiff_force_sec_per_call": float(np.mean(t_ad)),
        "analytic_available": 1.0 if len(t_an) > 0 else 0.0,
    }
    if len(t_an) > 0:
        out.update(
            {
                "analytic_force_sec_per_call": float(np.mean(t_an)),
                "mean_dir_analytic_force": float(np.mean(dir_an_vals)),
                "max_rel_fd_vs_dir_analytic_force": float(np.max(rel_fd_dir_an)),
                "mean_rel_fd_vs_dir_analytic_force": float(np.mean(rel_fd_dir_an)),
                "max_rel_force_autodiff_vs_analytic": float(np.max(rel_ad_vs_an_force)),
                "mean_rel_force_autodiff_vs_analytic": float(np.mean(rel_ad_vs_an_force)),
            }
        )
    return out


def main():
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(description="Wilson-fermion building-block checks")
    ap.add_argument("--shape", type=str, default="4,4,4,8")
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--solver-tol", type=float, default=1e-8)
    ap.add_argument("--solver-maxiter", type=int, default=500)
    ap.add_argument("--solver-kind", type=str, default="cg", choices=["cg", "bicgstab", "gmres"])
    ap.add_argument("--solver-form", type=str, default="normal", choices=["normal", "split", "eo_split"])
    ap.add_argument("--preconditioner", type=str, default="none", choices=["none", "jacobi"])
    ap.add_argument("--gmres-restart", type=int, default=32)
    ap.add_argument("--gmres-solve-method", type=str, default="batched")
    ap.add_argument("--dirac-kernel", type=str, default="optimized", choices=["optimized", "reference"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ")
    ap.add_argument("--exp-method", type=str, default="su2", choices=["expm", "su2"])
    ap.add_argument("--include-gauge-monomial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-fermion-monomial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--fermion-monomial-kind",
        type=str,
        default="unpreconditioned",
        choices=["unpreconditioned", "eo_preconditioned"],
    )
    ap.add_argument("--gauge-timescale", type=int, default=0)
    ap.add_argument("--fermion-timescale", type=int, default=1)
    ap.add_argument("--pf-refresh", type=str, default="heatbath", choices=["heatbath", "ou"])
    ap.add_argument("--pf-force-mode", type=str, default="autodiff", choices=["autodiff", "analytic"])
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
    ap.add_argument("--gauge-exp-method", type=str, default="expm", choices=["expm", "su2", "auto"])
    ap.add_argument("--pf-id-trials", type=int, default=3)
    ap.add_argument("--force-compare-trials", type=int, default=2)
    ap.add_argument("--force-compare-iters", type=int, default=2)
    ap.add_argument("--pf-fd-trials", type=int, default=2)
    ap.add_argument("--pf-fd-eps", type=float, default=5e-4)
    ap.add_argument("--pf-fd-iters", type=int, default=2, help="timing iterations for EO-preconditioned force checks")
    ap.add_argument("--auto-refresh-pseudofermions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-dirac-kernels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-solvers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--n-iter-timing", type=int, default=10)
    ap.add_argument(
        "--tests",
        type=str,
        default="all",
        help="comma-separated: gamma,adjoint,normal,perf,eo,hamiltonian,pfrefresh,pfid,gauge,conventions,forcecmp,eopforce,all",
    )
    args = ap.parse_args()

    tests = {t.strip().lower() for t in args.tests.split(",") if t.strip()}
    if "all" in tests:
        tests = {"gamma", "adjoint", "normal", "perf", "eo", "hamiltonian", "pfrefresh", "pfid", "gauge", "conventions", "forcecmp", "eopforce"}

    shape = _parse_shape(args.shape)
    pf_gamma = args.pf_gamma
    if pf_gamma is None:
        pf_gamma = float(args.smd_gamma) if str(args.pf_refresh).lower() == "ou" else 0.3

    th = SU2WilsonNf2(
        lattice_shape=shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=args.layout,
        exp_method=args.exp_method,
        mass=args.mass,
        wilson_r=args.r,
        cg_tol=float(args.solver_tol),
        cg_maxiter=int(args.solver_maxiter),
        solver_kind=str(args.solver_kind),
        solver_form=str(args.solver_form),
        preconditioner_kind=str(args.preconditioner),
        gmres_restart=int(args.gmres_restart),
        gmres_solve_method=str(args.gmres_solve_method),
        dirac_kernel=str(args.dirac_kernel),
        include_gauge_monomial=bool(args.include_gauge_monomial),
        include_fermion_monomial=bool(args.include_fermion_monomial),
        fermion_monomial_kind=str(args.fermion_monomial_kind),
        gauge_timescale=int(args.gauge_timescale),
        fermion_timescale=int(args.fermion_timescale),
        pseudofermion_refresh=str(args.pf_refresh),
        pseudofermion_gamma=float(pf_gamma),
        pseudofermion_force_mode=str(args.pf_force_mode),
        smd_gamma=float(args.smd_gamma),
        auto_refresh_pseudofermions=bool(args.auto_refresh_pseudofermions),
        jit_dirac_kernels=bool(args.jit_dirac_kernels),
        jit_solvers=bool(args.jit_solvers),
    )

    print("JAX backend:", jax.default_backend())
    print("Theory config:")
    print(f"  shape: {shape}")
    print(f"  Nd: {th.Nd}")
    print(f"  spin_dim: {th.Ns}")
    print(f"  fermion_shape: {th.fermion_shape()}")
    print(f"  monomials: {', '.join(th.hamiltonian.monomial_names())}")
    print(f"  fermion monomial kind: {th.fermion_monomial_kind}")
    print(f"  monomial timescales: gauge={th.gauge_timescale}, fermion={th.fermion_timescale}")
    print(f"  dirac kernel: {th.dirac_kernel}")
    solver_line = (
        "  solver:"
        f" kind={th.solver_kind}"
        f" form={th.solver_form}"
        f" preconditioner={th.preconditioner_kind}"
        f" tol={th.cg_tol}"
        f" maxiter={th.cg_maxiter}"
    )
    if th.solver_kind == "gmres":
        solver_line += f" gmres_restart={th.gmres_restart} gmres_solve_method={th.gmres_solve_method}"
    print(solver_line)
    print(f"  smd gamma: {float(th.smd_gamma):.6g}")
    print(f"  pseudofermion refresh: mode={th.pseudofermion_refresh}, gamma={th.pseudofermion_gamma}")
    print(f"  pseudofermion force mode: {th.pseudofermion_force_mode}")
    print(f"  auto_refresh_pseudofermions: {bool(th.auto_refresh_pseudofermions)}")
    print(f"  jit dirac kernels / solvers: {bool(th.jit_dirac_kernels)} / {bool(th.jit_solvers)}")

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
        print(f"  D FLOPs/site sparse/dense: {p['flops_per_site_D_sparse']:.0f} / {p['flops_per_site_D_dense']:.0f}")
        print(f"  D      sec/call sparse/dense: {p['D_sparse_sec_per_call']:.6e} / {p['D_dense_sec_per_call']:.6e}")
        print(f"  DdagD  sec/call sparse/dense: {p['N_sparse_sec_per_call']:.6e} / {p['N_dense_sec_per_call']:.6e}")
        print(f"  D      GFLOP/s sparse/dense:  {p['D_sparse_gflops']:.6e} / {p['D_dense_gflops']:.6e}")
        print(f"  DdagD  GFLOP/s sparse/dense:  {p['N_sparse_gflops']:.6e} / {p['N_dense_gflops']:.6e}")
        print(f"  rel diff D (sparse,dense): {p['rel_D_sparse_vs_dense']:.6e}")
        print(f"  rel diff DdagD (sparse,dense): {p['rel_N_sparse_vs_dense']:.6e}")
        active_name = "optimized" if bool(int(p["active_kernel"])) else "reference"
        other_name = "optimized" if bool(int(p["other_kernel"])) else "reference"
        print("Kernel comparison (active vs other):")
        print(f"  active/other kernels: {active_name} / {other_name}")
        print(f"  D      sec/call active/other: {p['D_sparse_sec_per_call']:.6e} / {p['D_other_kernel_sec_per_call']:.6e}")
        print(f"  DdagD  sec/call active/other: {p['N_sparse_sec_per_call']:.6e} / {p['N_other_kernel_sec_per_call']:.6e}")
        print(f"  D      GFLOP/s active/other:  {p['D_sparse_gflops']:.6e} / {p['D_other_kernel_gflops']:.6e}")
        print(f"  DdagD  GFLOP/s active/other:  {p['N_sparse_gflops']:.6e} / {p['N_other_kernel_gflops']:.6e}")
        print(f"  D      active/other ratio: {p['D_active_over_other']:.6e}")
        print(f"  DdagD  active/other ratio: {p['N_active_over_other']:.6e}")
        print(f"  rel diff D (active,other): {p['rel_D_active_vs_other_kernel']:.6e}")
        print(f"  rel diff DdagD (active,other): {p['rel_N_active_vs_other_kernel']:.6e}")

    if "eo" in tests:
        e = test_eo_operator(th, n_iter=max(1, args.n_iter_timing))
        print("Even-odd Schur operator test:")
        print(f"  rel even Schur vs ref:      {e['rel_even_schur_vs_ref']:.6e}")
        print(f"  rel odd  Schur vs ref:      {e['rel_odd_schur_vs_ref']:.6e}")
        print(f"  rel even Schur compact vs ref: {e['rel_even_schur_compact_vs_ref']:.6e}")
        print(f"  rel even normalized vs ref: {e['rel_even_norm_schur_vs_ref']:.6e}")
        print(f"  rel odd  normalized vs ref: {e['rel_odd_norm_schur_vs_ref']:.6e}")
        print(f"  rel even normalized compact vs ref: {e['rel_even_norm_schur_compact_vs_ref']:.6e}")
        print(f"  sec/call EO-even matvec:    {e['eo_even_sec_per_call']:.6e}")
        print(f"  sec/call EO-even compact:   {e['eo_even_compact_sec_per_call']:.6e}")
        print(f"  sec/call normal-even matvec:{e['normal_even_sec_per_call']:.6e}")
        print(f"  EO/normal timing ratio:     {e['eo_over_normal_even']:.6e}")
        print(f"  EO-compact/full ratio:      {e['eo_compact_over_full_even']:.6e}")

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

    if "forcecmp" in tests:
        fc = test_pseudofermion_force_compare(
            th,
            q_scale=0.05,
            n_trials=int(args.force_compare_trials),
            n_iter_timing=int(args.force_compare_iters),
        )
        if not bool(int(fc.get("enabled", 0.0))):
            print("Pseudofermion force compare: skipped (no fermion monomial)")
        else:
            print("Pseudofermion force compare (analytic vs autodiff):")
            print(f"  max rel force diff:         {fc['max_rel_force_diff']:.6e}")
            print(f"  mean rel force diff:        {fc['mean_rel_force_diff']:.6e}")
            print(f"  max abs force diff:         {fc['max_abs_force_diff']:.6e}")
            print(f"  max rel directional diff:   {fc['max_rel_directional_diff']:.6e}")
            print(f"  mean rel directional diff:  {fc['mean_rel_directional_diff']:.6e}")
            print(f"  mean dS(auto) / -<F,H>:     {fc['mean_directional_autodiff']:.6e} / {fc['mean_directional_force']:.6e}")
            print(f"  mean solve rel residual:    {fc['mean_rel_residual']:.6e}")
            print(f"  mean solve abs residual:    {fc['mean_abs_residual']:.6e}")
            known = float(fc["solver_info_known_fraction"])
            if known > 0.0:
                print(f"  solver info known fraction: {known:.6e}")
                print(f"  solver info mean/max iters: {fc['solver_info_mean_iters']:.6e} / {fc['solver_info_max_iters']:.6e}")
                if np.isfinite(fc.get("solver_info_mean_iters_stage1", float("nan"))) or np.isfinite(fc.get("solver_info_mean_iters_stage2", float("nan"))):
                    print(
                        "  solver split mean iters (stage1/stage2):"
                        f" {fc['solver_info_mean_iters_stage1']:.6e} / {fc['solver_info_mean_iters_stage2']:.6e}"
                    )
            else:
                print("  solver iterations: unavailable from solver info in this JAX build")
            print(f"  autodiff sec/call:          {fc['autodiff_sec_per_call']:.6e}")
            print(f"  analytic sec/call:          {fc['analytic_sec_per_call']:.6e}")
            print(f"  solver solve sec/call:      {fc['solver_solve_sec_per_call']:.6e}")
            if str(th.solver_form) == "normal":
                print(f"  DdagD matvec sec/call:      {fc['normal_matvec_sec_per_call']:.6e}")
            else:
                print(
                    "  D / Ddag matvec sec/call:"
                    f" {fc['direct_matvec_sec_per_call']:.6e} / {fc['dagger_matvec_sec_per_call']:.6e}"
                )
            print(f"  solve op-apply equivalent:  {fc['solve_operator_apply_equiv']:.6e}")
            print(
                "  analytic split solve/kernel:"
                f" {fc['analytic_solve_sec_per_call']:.6e} / {fc['analytic_force_kernel_sec_per_call']:.6e}"
            )
            print(f"  analytic/autodiff:          {fc['analytic_over_autodiff']:.6e}")

    if "eopforce" in tests:
        ef = test_eop_force_fd(
            th,
            q_scale=0.05,
            n_trials=int(args.pf_fd_trials),
            eps=float(args.pf_fd_eps),
            n_iter_timing=int(args.pf_fd_iters),
        )
        if not bool(int(ef.get("enabled", 0.0))):
            print("EO-preconditioned force FD check: skipped (requires --fermion-monomial-kind eo_preconditioned)")
        else:
            print("EO-preconditioned force finite-difference check:")
            print(f"  eps:                         {ef['eps']:.6e}")
            print(f"  mean fd (forward):           {ef['mean_fd_forward']:.6e}")
            print(f"  mean fd (central):           {ef['mean_fd_central']:.6e}")
            print(f"  mean dS autodiff:            {ef['mean_dS_autodiff']:.6e}")
            print(f"  mean -<F_ad,H>:              {ef['mean_dir_autodiff_force']:.6e}")
            print(
                "  max/mean rel(fd_c,dS_ad):"
                f"   {ef['max_rel_fd_central_vs_dS_autodiff']:.6e} / {ef['mean_rel_fd_central_vs_dS_autodiff']:.6e}"
            )
            print(
                "  max/mean rel(fd_c,-<F_ad,H>):"
                f" {ef['max_rel_fd_central_vs_dir_autodiff_force']:.6e} / {ef['mean_rel_fd_central_vs_dir_autodiff_force']:.6e}"
            )
            print(f"  action sec/call:             {ef['action_sec_per_call']:.6e}")
            print(f"  autodiff force sec/call:     {ef['autodiff_force_sec_per_call']:.6e}")
            if bool(int(ef.get("analytic_available", 0.0))):
                print(f"  mean -<F_an,H>:              {ef['mean_dir_analytic_force']:.6e}")
                print(
                    "  max/mean rel(fd_c,-<F_an,H>):"
                    f" {ef['max_rel_fd_vs_dir_analytic_force']:.6e} / {ef['mean_rel_fd_vs_dir_analytic_force']:.6e}"
                )
                print(
                    "  max/mean rel(F_ad,F_an):"
                    f" {ef['max_rel_force_autodiff_vs_analytic']:.6e} / {ef['mean_rel_force_autodiff_vs_analytic']:.6e}"
                )
                print(f"  analytic force sec/call:     {ef['analytic_force_sec_per_call']:.6e}")


if __name__ == "__main__":
    main()

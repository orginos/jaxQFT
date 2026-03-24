"""Reference DD determinant factorization for quenched/full U(1) Wilson Nf=2 on tiny lattices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxqft.core import HamiltonianModel
from jaxqft.core.domain_decomposition import TimeSlabDecomposition
from jaxqft.models.u1_ym import project_u1_algebra
from jaxqft.models.u1_ym_dd import U1TimeSlabDDTheory
from jaxqft.models.u1_wilson_nf2 import GaugeActionMonomial, U1WilsonNf2


Array = jax.Array


def _fermion_structure(model: U1WilsonNf2) -> Tuple[Tuple[int, ...], int, int, int, int, int]:
    lattice_shape = tuple(int(v) for v in tuple(model.lattice_shape))
    fshp = tuple(int(v) for v in tuple(model.fermion_shape()))
    bs = int(fshp[0])
    ns = int(fshp[-2])
    nc = int(fshp[-1])
    vol = int(np.prod(np.asarray(lattice_shape, dtype=np.int64)))
    nsc = int(ns * nc)
    ndof = int(vol * nsc)
    return lattice_shape, bs, ns, nc, nsc, ndof


def _submatrix(mat: Array, rows: Array, cols: Array) -> Array:
    return mat[rows[:, None], cols[None, :]]


def _dense_dirac_matrix(model: U1WilsonNf2, U: Array, *, dense_max_dof: int) -> Array:
    lattice_shape, bs, ns, nc, _nsc, ndof = _fermion_structure(model)
    max_dof = int(dense_max_dof)
    if max_dof > 0 and ndof > max_dof:
        raise ValueError(
            f"Dense DD determinant blocked: ndof={ndof} exceeds dense_max_dof={max_dof}. "
            "Increase dense_max_dof explicitly if intended."
        )
    eye = jnp.eye(ndof, dtype=model.dtype)
    rhs_cols = jnp.broadcast_to(eye[None, :, :], (bs, ndof, ndof))
    rhs_cols = jnp.moveaxis(rhs_cols, 1, 0).reshape((ndof, bs, *lattice_shape, ns, nc))
    cols = jax.vmap(lambda rhs: model.apply_D_dense(U, rhs), in_axes=0, out_axes=0)(rhs_cols)
    return jnp.transpose(cols.reshape((ndof, bs, ndof)), axes=(1, 2, 0))


def _logabsdet(mat: Array) -> Array:
    _sign, logabs = jnp.linalg.slogdet(mat)
    return jnp.real(logabs)


def _dd_factorization_data(
    model: U1WilsonNf2,
    U: Array,
    decomposition: TimeSlabDecomposition,
    *,
    dense_max_dof: int,
) -> Dict[str, Array]:
    _lattice_shape, bs, _ns, _nc, nsc, _ndof = _fermion_structure(model)
    dmat = _dense_dirac_matrix(model, U, dense_max_dof=int(dense_max_dof))
    b_idx = jnp.asarray(decomposition.boundary_component_indices(nsc), dtype=jnp.int32)
    site_comps = decomposition.interior_site_components()
    dom_idx = tuple(jnp.asarray(decomposition.component_indices(sites, nsc), dtype=jnp.int32) for sites in site_comps)
    nb = int(b_idx.size)
    eye_b = jnp.eye(nb, dtype=model.dtype)

    def per_batch(d: Array):
        d_full = d
        full_log = _logabsdet(d_full)

        d_bb = _submatrix(d, b_idx, b_idx)
        boundary_log = _logabsdet(d_bb)

        local_logs = []
        corr_sum = jnp.zeros((nb, nb), dtype=model.dtype)
        for idx in dom_idx:
            d_loc = _submatrix(d, idx, idx)
            local_logs.append(_logabsdet(d_loc))
            d_bi = _submatrix(d, b_idx, idx)
            d_ib = _submatrix(d, idx, b_idx)
            corr_sum = corr_sum + d_bi @ jnp.linalg.solve(d_loc, d_ib)

        corr = eye_b - jnp.linalg.solve(d_bb, corr_sum)
        corr_log = _logabsdet(corr)
        local_arr = jnp.stack(local_logs) if local_logs else jnp.zeros((0,), dtype=jnp.float32)
        return boundary_log, local_arr, corr_log, full_log

    boundary_log, local_logs, corr_log, full_log = jax.vmap(per_batch, in_axes=0, out_axes=0)(dmat)
    return {
        "boundary_logabsdet": boundary_log,
        "local_logabsdet": local_logs,
        "correction_logabsdet": corr_log,
        "full_logabsdet": full_log,
    }


class _DDExactDetBase:
    model: "U1WilsonNf2DDReference"
    name: str
    timescale: int
    stochastic: bool = False
    _force_ad_fn: Optional[object] = None

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        _ = q
        _ = traj_length

    def piece_action(self, q: Array) -> Array:
        raise NotImplementedError

    def action(self, q: Array) -> Array:
        return self.piece_action(q)

    def _build_force_ad_fn(self):
        def _action_sum(u_re: Array, u_im: Array) -> Array:
            u = (u_re + 1j * u_im).astype(self.model.dtype)
            return jnp.sum(self.piece_action(u))

        grad_re_im = jax.grad(_action_sum, argnums=(0, 1))

        def force_fn(u: Array) -> Array:
            g_re, g_im = grad_re_im(jnp.real(u), jnp.imag(u))
            dS_dU = (g_re - 1j * g_im).astype(self.model.dtype)
            return self.model.project_active_links(project_u1_algebra(u * dS_dU))

        return jax.jit(force_fn)

    def force(self, q: Array) -> Array:
        if self._force_ad_fn is None:
            self._force_ad_fn = self._build_force_ad_fn()
        return self._force_ad_fn(q)


@dataclass
class DDGaugeConditionalMonomial:
    model: "U1WilsonNf2DDReference"
    name: str = "dd_gauge"
    timescale: int = 0
    stochastic: bool = False

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        _ = q
        _ = traj_length

    def action(self, q: Array) -> Array:
        return self.model.dd_gauge.action(q)

    def force(self, q: Array) -> Array:
        return self.model.dd_gauge.force(q)


@dataclass
class DDLocalDomainExactDetMonomial(_DDExactDetBase):
    model: "U1WilsonNf2DDReference"
    domain_index: int = 0
    name: str = "dd_local_det"
    timescale: int = 1
    stochastic: bool = False

    def __post_init__(self) -> None:
        self.name = f"dd_local_det_{int(self.domain_index)}"

    def piece_action(self, q: Array) -> Array:
        pieces = self.model.factorization_breakdown(q)
        logs = pieces["local_logabsdet"]
        return -2.0 * logs[:, int(self.domain_index)]


@dataclass
class DDBoundaryCorrectionExactMonomial(_DDExactDetBase):
    model: "U1WilsonNf2DDReference"
    name: str = "dd_boundary_corr"
    timescale: int = 2
    stochastic: bool = False

    def piece_action(self, q: Array) -> Array:
        pieces = self.model.factorization_breakdown(q)
        return -2.0 * pieces["correction_logabsdet"]


@dataclass
class U1WilsonNf2DDReference:
    """Exact dense DD determinant factorization reference theory for tiny lattices.

    The fermion determinant is factorized as
      det D = det D_bb * (prod_i det D_i) * det R_b
    where D_bb lives on the frozen boundary, D_i are the disconnected local
    interior blocks, and R_b is the exact boundary correction matrix.

    Only the active-link dependent pieces are kept in the Hamiltonian:
      gauge_conditional + sum_i(-2 log|det D_i|) + (-2 log|det R_b|).
    """

    base: U1WilsonNf2
    boundary_slices: Tuple[int, ...]
    boundary_width: int = 1
    dense_max_dof: int = 512
    gauge_timescale: int = 0
    local_timescale: int = 1
    correction_timescale: int = 2
    include_correction_monomial: bool = True
    decomposition: TimeSlabDecomposition = field(init=False)
    dd_gauge: U1TimeSlabDDTheory = field(init=False, repr=False)
    hamiltonian: HamiltonianModel = field(init=False, repr=False)
    local_monomials: Tuple[DDLocalDomainExactDetMonomial, ...] = field(init=False, repr=False)
    correction_monomial: Optional[DDBoundaryCorrectionExactMonomial] = field(init=False, repr=False)
    requires_trajectory_refresh: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if bool(getattr(self.base, "include_fermion_monomial", False)):
            raise ValueError("U1WilsonNf2DDReference expects a gauge-only U1WilsonNf2 base (include_fermion_monomial=False)")
        self.decomposition = TimeSlabDecomposition(
            lattice_shape=tuple(int(v) for v in self.base.lattice_shape),
            boundary_slices=tuple(int(v) for v in self.boundary_slices),
            boundary_width=int(self.boundary_width),
        )
        self.boundary_slices = tuple(int(v) for v in self.decomposition.boundary_slices)
        self.boundary_width = int(self.decomposition.boundary_width)
        self.dd_gauge = U1TimeSlabDDTheory(self.base, boundary_slices=self.boundary_slices, boundary_width=self.boundary_width)

        self.local_monomials = tuple(
            DDLocalDomainExactDetMonomial(
                self,
                domain_index=i,
                timescale=int(self.local_timescale),
            )
            for i, _ in enumerate(self.decomposition.interior_site_components())
        )
        self.correction_monomial = (
            DDBoundaryCorrectionExactMonomial(self, timescale=int(self.correction_timescale))
            if bool(self.include_correction_monomial)
            else None
        )

        monomials = [DDGaugeConditionalMonomial(self, timescale=int(self.gauge_timescale))]
        monomials.extend(self.local_monomials)
        if self.correction_monomial is not None:
            monomials.append(self.correction_monomial)
        self.hamiltonian = HamiltonianModel(tuple(monomials))

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def lattice_shape(self):
        return self.base.lattice_shape

    @property
    def Nd(self):
        return self.base.Nd

    @property
    def Bs(self):
        return self.base.Bs

    def field_shape(self):
        return self.base.field_shape()

    def fermion_shape(self):
        return self.base.fermion_shape()

    def project_active_links(self, x: Array) -> Array:
        return self.dd_gauge.project_active_links(x)

    @property
    def active_mask(self):
        return self.dd_gauge.active_mask

    @property
    def frozen_mask(self):
        return self.dd_gauge.frozen_mask

    def metadata(self) -> Mapping[str, float]:
        meta = dict(self.dd_gauge.metadata())
        meta.update(
            {
                "dd_n_local_domains": float(len(self.local_monomials)),
                "dd_dense_max_dof": float(int(self.dense_max_dof)),
            }
        )
        return meta

    def factorization_breakdown(self, U: Array) -> Dict[str, Array]:
        return _dd_factorization_data(self.base, U, self.decomposition, dense_max_dof=int(self.dense_max_dof))

    def full_exact_fermion_action(self, U: Array) -> Array:
        pieces = self.factorization_breakdown(U)
        return -2.0 * pieces["full_logabsdet"]

    def boundary_constant_action(self, U: Array) -> Array:
        pieces = self.factorization_breakdown(U)
        return -2.0 * pieces["boundary_logabsdet"]

    def factorized_fermion_action(self, U: Array) -> Array:
        pieces = self.factorization_breakdown(U)
        total = -2.0 * jnp.sum(pieces["local_logabsdet"], axis=1)
        if bool(self.include_correction_monomial):
            total = total - 2.0 * pieces["correction_logabsdet"]
        return total

    def full_exact_total_action(self, U: Array) -> Array:
        return self.base.action(U) + self.full_exact_fermion_action(U)

    def conditional_exact_total_action(self, U: Array) -> Array:
        return self.dd_gauge.action(U) + self.factorized_fermion_action(U)

    def refresh_p(self) -> Array:
        return self.dd_gauge.refresh_p()

    def refresh_p_with_key(self, key: Array) -> Array:
        return self.dd_gauge.refresh_p_with_key(key)

    def kinetic(self, P: Array) -> Array:
        return self.dd_gauge.kinetic(P)

    def evolve_q(self, dt: float, P: Array, Q: Array) -> Array:
        return self.dd_gauge.evolve_q(dt, P, Q)

    def average_plaquette(self, U: Array) -> Array:
        return self.base.average_plaquette(U)

    def prepare_trajectory(self, U: Array, traj_length: float = 1.0) -> None:
        _ = U
        _ = traj_length

    def action_breakdown(self, U: Array) -> Dict[str, Array]:
        return self.hamiltonian.action_breakdown(U)

    def action(self, U: Array) -> Array:
        return self.hamiltonian.action_from_monomials(U)

    def force(self, U: Array) -> Array:
        return self.hamiltonian.force_from_monomials(U)

    refreshP = refresh_p
    evolveQ = evolve_q

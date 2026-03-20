"""O(N) sigma model with SO(N) rotation updates.

The field lives on the sphere S^{N-1} ~= SO(N)/SO(N-1), while the HMC drift is
performed by exponentiating algebra-valued momenta p^a L_a in so(N).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm


Array = jax.Array


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.core.integrators import force_gradient, leapfrog, minnorm2, minnorm4pf4
from jaxqft.core.update import SMD, hmc
from jaxqft.stats import integrated_autocorr_time


def _sinx_over_x(x: Array) -> Array:
    x2 = x * x
    return jnp.where(
        jnp.abs(x) > 1.0e-7,
        jnp.sin(x) / x,
        1.0 - x2 / 6.0 + x2 * x2 / 120.0,
    )


def _one_minus_cos_over_x2(x: Array) -> Array:
    x2 = x * x
    return jnp.where(
        jnp.abs(x) > 1.0e-5,
        (1.0 - jnp.cos(x)) / x2,
        0.5 - x2 / 24.0 + x2 * x2 / 720.0,
    )


def _parse_shape(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    else:
        vals = [int(v) for v in raw]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _correlation_length(L: int, chi_m: float, c2p: float) -> float:
    if (not np.isfinite(chi_m)) or (not np.isfinite(c2p)) or c2p <= 0.0 or chi_m <= c2p:
        return float("nan")
    return float(np.sqrt(chi_m / c2p - 1.0) / (2.0 * np.sin(np.pi / float(L))))


@dataclass
class ONSigmaModel:
    lattice_shape: Sequence[int]
    beta: float
    ncomp: int
    batch_size: int = 1
    layout: str = "B...N"
    dtype: jnp.dtype = jnp.float32
    seed: int = 0
    exp_method: str = "auto"
    jit_kernels: bool = True
    key: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lattice_shape = tuple(int(v) for v in self.lattice_shape)
        if not self.lattice_shape:
            raise ValueError("lattice_shape must be non-empty")
        self.Nd = len(self.lattice_shape)
        self.Vol = int(math.prod(self.lattice_shape))
        self.Bs = int(self.batch_size)
        self.beta = float(self.beta)
        self.ncomp = int(self.ncomp)
        if self.ncomp < 2:
            raise ValueError("ncomp must satisfy ncomp >= 2")
        self.layout = self._normalize_layout(self.layout)
        self.nalg = int(self.ncomp * (self.ncomp - 1) // 2)
        self.exp_method = self._normalize_exp_method(self.exp_method)
        self.key = jax.random.PRNGKey(int(self.seed))

        pairs = [(i, j) for i in range(self.ncomp) for j in range(i + 1, self.ncomp)]
        self._pair_i = jnp.asarray([i for i, _ in pairs], dtype=jnp.int32)
        self._pair_j = jnp.asarray([j for _, j in pairs], dtype=jnp.int32)

        if bool(self.jit_kernels):
            self._action_kernel = jax.jit(self._action_impl)
            self._force_kernel = jax.jit(self._force_impl)
            self._evolve_q_kernel = jax.jit(self._evolve_q_impl)
        else:
            self._action_kernel = self._action_impl
            self._force_kernel = self._force_impl
            self._evolve_q_kernel = self._evolve_q_impl

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        txt = str(layout).strip().upper()
        aliases = {
            "BN...": "BN...",
            "BN": "BN...",
            "B N": "BN...",
            "B...N": "B...N",
            "BXYN": "B...N",
            "AUTO": "AUTO",
        }
        if txt not in aliases:
            raise ValueError("layout must be one of: BN..., B...N, auto")
        return aliases[txt]

    def _normalize_exp_method(self, method: str) -> str:
        txt = str(method).strip().lower()
        if txt == "auto":
            return "rodrigues" if self.ncomp == 3 else "expm"
        if txt not in ("expm", "rodrigues"):
            raise ValueError("exp_method must be one of: auto, expm, rodrigues")
        if txt == "rodrigues" and self.ncomp != 3:
            raise ValueError("exp_method=rodrigues is only available for ncomp=3")
        return txt

    def _split_key(self) -> Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def field_shape(self) -> tuple[int, ...]:
        if self.layout == "BN...":
            return (self.Bs, self.ncomp, *self.lattice_shape)
        return (self.Bs, *self.lattice_shape, self.ncomp)

    def momentum_shape(self) -> tuple[int, ...]:
        if self.layout == "BN...":
            return (self.Bs, self.nalg, *self.lattice_shape)
        return (self.Bs, *self.lattice_shape, self.nalg)

    def _field_component_axis(self) -> int:
        return 1 if self.layout == "BN..." else self.Nd + 1

    def _momentum_component_axis(self) -> int:
        return 1 if self.layout == "BN..." else self.Nd + 1

    def _field_lattice_axes(self) -> tuple[int, ...]:
        if self.layout == "BN...":
            return tuple(range(2, 2 + self.Nd))
        return tuple(range(1, 1 + self.Nd))

    def _momentum_lattice_axes(self) -> tuple[int, ...]:
        if self.layout == "BN...":
            return tuple(range(2, 2 + self.Nd))
        return tuple(range(1, 1 + self.Nd))

    def _field_to_bsitecomp(self, q: Array) -> Array:
        if self.layout == "BN...":
            q = jnp.moveaxis(q, 1, -1)
        return jnp.reshape(q, (q.shape[0], self.Vol, self.ncomp))

    def _field_from_bsitecomp(self, q: Array) -> Array:
        q = jnp.reshape(q, (q.shape[0], *self.lattice_shape, self.ncomp))
        if self.layout == "BN...":
            q = jnp.moveaxis(q, -1, 1)
        return q

    def _momentum_to_bsitealg(self, p: Array) -> Array:
        if self.layout == "BN...":
            p = jnp.moveaxis(p, 1, -1)
        return jnp.reshape(p, (p.shape[0], self.Vol, self.nalg))

    def _momentum_from_bsitealg(self, p: Array) -> Array:
        p = jnp.reshape(p, (p.shape[0], *self.lattice_shape, self.nalg))
        if self.layout == "BN...":
            p = jnp.moveaxis(p, -1, 1)
        return p

    def _field_component_last(self, q: Array) -> Array:
        if self.layout == "BN...":
            return jnp.moveaxis(q, 1, -1)
        return q

    def _momentum_component_last(self, p: Array) -> Array:
        if self.layout == "BN...":
            return jnp.moveaxis(p, 1, -1)
        return p

    def _momentum_from_component_last(self, p: Array) -> Array:
        if self.layout == "BN...":
            return jnp.moveaxis(p, -1, 1)
        return p

    def _site_dot(self, a: Array, b: Array) -> Array:
        return jnp.sum(a * b, axis=self._field_component_axis())

    def _o3_pair_to_vector_last(self, p_last: Array) -> Array:
        return jnp.stack((-p_last[..., 2], p_last[..., 1], -p_last[..., 0]), axis=-1)

    def _o3_vector_to_pair_last(self, w_last: Array) -> Array:
        return jnp.stack((-w_last[..., 2], w_last[..., 1], -w_last[..., 0]), axis=-1)

    def _coeffs_to_matrix_flat(self, p_flat: Array) -> Array:
        mats = jnp.zeros((*p_flat.shape[:-1], self.ncomp, self.ncomp), dtype=p_flat.dtype)
        mats = mats.at[..., self._pair_i, self._pair_j].set(p_flat)
        mats = mats.at[..., self._pair_j, self._pair_i].set(-p_flat)
        return mats

    def _neighbor_sum(self, q: Array) -> Array:
        out = jnp.zeros_like(q)
        for axis in self._field_lattice_axes():
            out = out + jnp.roll(q, shift=1, axis=axis) + jnp.roll(q, shift=-1, axis=axis)
        return out

    def _action_impl(self, q: Array) -> Array:
        acc = self.Nd * self.Vol * jnp.ones((q.shape[0],), dtype=q.dtype)
        lattice_axes = self._field_lattice_axes()
        for axis in lattice_axes:
            term = self._site_dot(q, jnp.roll(q, shift=-1, axis=axis))
            acc = acc - jnp.sum(term, axis=tuple(range(1, term.ndim)))
        return self.beta * acc

    def _force_impl(self, q: Array) -> Array:
        nn = self._neighbor_sum(q)
        if self.ncomp == 3:
            q_last = self._field_component_last(q)
            nn_last = self._field_component_last(nn)
            omega = jnp.cross(q_last, nn_last, axis=-1)
            coeff_last = self._o3_vector_to_pair_last(omega)
            return self.beta * self._momentum_from_component_last(coeff_last)
        qf = self._field_to_bsitecomp(q)
        nf = self._field_to_bsitecomp(nn)
        coeffs = nf[..., self._pair_i] * qf[..., self._pair_j] - nf[..., self._pair_j] * qf[..., self._pair_i]
        return self.beta * self._momentum_from_bsitealg(coeffs)

    def _evolve_q_rodrigues(self, dt: float, p: Array, q: Array) -> Array:
        if self.ncomp == 3:
            p_last = self._momentum_component_last(p)
            q_last = self._field_component_last(q)
            omega = self._o3_pair_to_vector_last(p_last)
            aq = jnp.cross(omega, q_last, axis=-1)
            aaq = jnp.cross(omega, aq, axis=-1)
            w = jnp.sqrt(jnp.sum(omega * omega, axis=-1))
            dt_arr = jnp.asarray(dt, dtype=q.dtype)
            theta = dt_arr * w
            s1 = _sinx_over_x(theta)[..., None] * dt_arr
            s2 = _one_minus_cos_over_x2(theta)[..., None] * (dt_arr ** 2)
            q_new = q_last + s1 * aq + s2 * aaq
            return self._field_from_bsitecomp(jnp.reshape(q_new, (q.shape[0], self.Vol, self.ncomp)))
        p_flat = self._momentum_to_bsitealg(p)
        q_flat = self._field_to_bsitecomp(q)
        mats = self._coeffs_to_matrix_flat(p_flat)
        aq = jnp.einsum("...ij,...j->...i", mats, q_flat)
        aaq = jnp.einsum("...ij,...j->...i", mats, aq)
        w = jnp.sqrt(jnp.sum(p_flat * p_flat, axis=-1))
        theta = jnp.asarray(dt, dtype=q.dtype) * w
        s1 = _sinx_over_x(theta)[..., None] * jnp.asarray(dt, dtype=q.dtype)
        s2 = _one_minus_cos_over_x2(theta)[..., None] * (jnp.asarray(dt, dtype=q.dtype) ** 2)
        q_new = q_flat + s1 * aq + s2 * aaq
        return self._field_from_bsitecomp(q_new)

    def _evolve_q_expm(self, dt: float, p: Array, q: Array) -> Array:
        p_flat = self._momentum_to_bsitealg(p)
        q_flat = self._field_to_bsitecomp(q)
        mats = self._coeffs_to_matrix_flat(p_flat)
        mats = jnp.reshape(jnp.asarray(dt, dtype=q.dtype) * mats, (-1, self.ncomp, self.ncomp))
        qv = jnp.reshape(q_flat, (-1, self.ncomp, 1))
        rot = jax.vmap(expm)(mats)
        q_new = jnp.matmul(rot, qv)[..., 0]
        q_new = jnp.reshape(q_new, (q.shape[0], self.Vol, self.ncomp))
        return self._field_from_bsitecomp(q_new)

    def _evolve_q_impl(self, dt: float, p: Array, q: Array) -> Array:
        if self.exp_method == "rodrigues":
            return self._evolve_q_rodrigues(dt, p, q)
        return self._evolve_q_expm(dt, p, q)

    def action(self, q: Array) -> Array:
        return self._action_kernel(q)

    def force(self, q: Array) -> Array:
        return self._force_kernel(q)

    def evolve_q(self, dt: float, p: Array, q: Array) -> Array:
        return self._evolve_q_kernel(dt, p, q)

    def refresh_p(self) -> Array:
        return self.refresh_p_with_key(self._split_key())

    def refresh_p_with_key(self, key: Array) -> Array:
        return jax.random.normal(key, shape=self.momentum_shape(), dtype=self.dtype)

    def kinetic(self, p: Array) -> Array:
        return 0.5 * jnp.sum(p * p, axis=tuple(range(1, p.ndim)))

    def cold_start(self) -> Array:
        q = jnp.zeros(self.field_shape(), dtype=self.dtype)
        if self.layout == "BN...":
            q = q.at[:, 0, ...].set(1.0)
        else:
            q = q.at[..., 0].set(1.0)
        return q

    def hot_start(self) -> Array:
        q = jax.random.normal(self._split_key(), shape=self.field_shape(), dtype=self.dtype)
        return self.normalize(q)

    def normalize(self, q: Array) -> Array:
        comp_axis = self._field_component_axis()
        nrm = jnp.sqrt(jnp.sum(q * q, axis=comp_axis, keepdims=True))
        return q / jnp.maximum(nrm, jnp.asarray(1.0e-12, dtype=q.dtype))

    def average_spin(self, q: Array) -> Array:
        return jnp.mean(self._field_to_bsitecomp(q), axis=1)

    def first_momentum_structure_factor(self, q: Array, axis: int = 0) -> Array:
        ax = int(axis)
        if ax < 0 or ax >= self.Nd:
            raise ValueError(f"axis must be in [0,{self.Nd}), got {axis}")
        qv = self._field_component_last(q)
        x = np.arange(self.lattice_shape[ax], dtype=np.float64)
        phase = np.exp(2j * np.pi * x / float(self.lattice_shape[ax]))
        shape = [1] * (self.Nd + 1)
        shape[1 + ax] = self.lattice_shape[ax]
        phase = jnp.asarray(phase.reshape(shape), dtype=jnp.complex64)
        p1 = jnp.mean(qv.astype(jnp.complex64) * phase, axis=tuple(range(1, 1 + self.Nd)))
        return self.Vol * jnp.sum(jnp.real(jnp.conj(p1) * p1), axis=-1)

    def topological_charge(self, q: Array) -> Array:
        if self.Nd != 2 or self.ncomp != 3:
            raise ValueError("topological_charge is only defined for O(3) in 2D")

        def spherical_area(a: Array, b: Array, c: Array) -> Array:
            ab = jnp.sum(a * b, axis=-1)
            bc = jnp.sum(b * c, axis=-1)
            ac = jnp.sum(a * c, axis=-1)
            rho2 = 2.0 * (1.0 + ab) * (1.0 + bc) * (1.0 + ac)
            rho = jnp.sqrt(jnp.maximum(rho2, jnp.asarray(1.0e-14, dtype=q.dtype)))
            abc = jnp.sum(a * jnp.cross(b, c, axis=-1), axis=-1)
            num = 1.0 + ab + bc + ac + 1j * abc
            ratio = jnp.where(jnp.abs(num) > 1.0e-15, num / rho, jnp.ones_like(num))
            area = -2.0j * jnp.log(ratio)
            return jnp.real(area)

        ql = self._field_component_last(q)
        qx = jnp.roll(ql, shift=-1, axis=1)
        qy = jnp.roll(ql, shift=-1, axis=2)
        qxy = jnp.roll(qx, shift=-1, axis=2)
        cell = spherical_area(ql, qx, qxy) + spherical_area(ql, qxy, qy)
        return jnp.sum(cell, axis=(1, 2)) / (4.0 * np.pi)

    @classmethod
    def benchmark_layout(
        cls,
        *,
        lattice_shape: Sequence[int],
        beta: float,
        ncomp: int,
        batch_size: int = 1,
        seed: int = 0,
        exp_method: str = "auto",
        layouts: Iterable[str] = ("BN...", "B...N"),
        n_iter: int = 5,
        kernel: str = "all",
    ) -> dict[str, float]:
        kernel = str(kernel).lower()
        if kernel not in ("all", "action", "force", "evolveq"):
            raise ValueError("kernel must be one of: all, action, force, evolveq")
        out: dict[str, float] = {}
        for layout in layouts:
            th = cls(
                lattice_shape=lattice_shape,
                beta=beta,
                ncomp=ncomp,
                batch_size=batch_size,
                layout=layout,
                seed=seed,
                exp_method=exp_method,
            )
            q = th.hot_start()
            p = th.refreshP()
            if kernel in ("all", "action"):
                jax.block_until_ready(th.action(q))
            if kernel in ("all", "force"):
                jax.block_until_ready(th.force(q))
            if kernel in ("all", "evolveq"):
                jax.block_until_ready(th.evolveQ(0.1, p, q))
            tic = time.perf_counter()
            for _ in range(max(1, int(n_iter))):
                if kernel in ("all", "action"):
                    jax.block_until_ready(th.action(q))
                if kernel in ("all", "force"):
                    jax.block_until_ready(th.force(q))
                if kernel in ("all", "evolveq"):
                    jax.block_until_ready(th.evolveQ(0.1, p, q))
            toc = time.perf_counter()
            out[str(layout)] = float((toc - tic) / max(1, int(n_iter)))
        return out

    @classmethod
    def benchmark_layout_trajectory(
        cls,
        *,
        lattice_shape: Sequence[int],
        beta: float,
        ncomp: int,
        integrator: str = "minnorm2",
        nmd: int = 4,
        tau: float = 1.0,
        batch_size: int = 1,
        seed: int = 0,
        exp_method: str = "auto",
        layouts: Iterable[str] = ("BN...", "B...N"),
        n_iter: int = 3,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for layout in layouts:
            th = cls(
                lattice_shape=lattice_shape,
                beta=beta,
                ncomp=ncomp,
                batch_size=batch_size,
                layout=layout,
                seed=seed,
                exp_method=exp_method,
            )
            I = _build_integrator(str(integrator), th, int(nmd), float(tau))
            q = th.hot_start()
            p = th.refreshP()
            p1, q1 = I.integrate(p, q)
            jax.block_until_ready(p1)
            jax.block_until_ready(q1)
            tic = time.perf_counter()
            for _ in range(max(1, int(n_iter))):
                p1, q1 = I.integrate(p, q)
                jax.block_until_ready(p1)
                jax.block_until_ready(q1)
            toc = time.perf_counter()
            out[str(layout)] = float((toc - tic) / max(1, int(n_iter)))
        return out

    refreshP = refresh_p
    evolveQ = evolve_q
    hotStart = hot_start
    coldStart = cold_start


def _build_integrator(name: str, theory: ONSigmaModel, nmd: int, tau: float):
    nm = str(name).lower()
    if nm == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, int(nmd), float(tau))
    if nm == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, int(nmd), float(tau))
    if nm == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, int(nmd), float(tau))
    if nm == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, int(nmd), float(tau))
    raise ValueError(f"Unknown integrator: {name}")


def benchmark_exponentiation(theory: ONSigmaModel, dt: float = 0.1, n_iter: int = 10) -> dict[str, float]:
    if theory.ncomp != 3:
        raise ValueError("benchmark_exponentiation is only defined for O(3)")

    th_expm = ONSigmaModel(
        lattice_shape=theory.lattice_shape,
        beta=theory.beta,
        ncomp=theory.ncomp,
        batch_size=theory.Bs,
        layout=theory.layout,
        dtype=theory.dtype,
        seed=0,
        exp_method="expm",
    )
    th_rod = ONSigmaModel(
        lattice_shape=theory.lattice_shape,
        beta=theory.beta,
        ncomp=theory.ncomp,
        batch_size=theory.Bs,
        layout=theory.layout,
        dtype=theory.dtype,
        seed=0,
        exp_method="rodrigues",
    )
    q = th_rod.hot_start()
    p = th_rod.refreshP()

    q_expm = th_expm.evolveQ(float(dt), p, q)
    q_rod = th_rod.evolveQ(float(dt), p, q)
    jax.block_until_ready(q_expm)
    jax.block_until_ready(q_rod)

    t0 = time.perf_counter()
    for _ in range(max(1, int(n_iter))):
        jax.block_until_ready(th_expm.evolveQ(float(dt), p, q))
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(max(1, int(n_iter))):
        jax.block_until_ready(th_rod.evolveQ(float(dt), p, q))
    t3 = time.perf_counter()

    qx = th_expm._field_component_last(q_expm)
    qr = th_rod._field_component_last(q_rod)
    unit_expm = float(np.asarray(jnp.max(jnp.abs(jnp.sum(qx * qx, axis=-1) - 1.0))))
    unit_rod = float(np.asarray(jnp.max(jnp.abs(jnp.sum(qr * qr, axis=-1) - 1.0))))
    rel = float(np.asarray(jnp.linalg.norm(q_rod - q_expm) / (jnp.linalg.norm(q_expm) + 1e-12)))
    return {
        "expm_sec_per_call": float((t1 - t0) / max(1, int(n_iter))),
        "rodrigues_sec_per_call": float((t3 - t2) / max(1, int(n_iter))),
        "rodrigues_over_expm": float((t3 - t2) / max(1, int(n_iter)) / (((t1 - t0) / max(1, int(n_iter))) + 1e-16)),
        "rel_diff_rodrigues_vs_expm": rel,
        "unit_err_expm": unit_expm,
        "unit_err_rodrigues": unit_rod,
    }


def kernel_timing(theory: ONSigmaModel, n_iter: int = 10, dt: float = 0.1, mom_axis: int = 0) -> dict[str, float]:
    nit = max(1, int(n_iter))
    q = theory.hot_start()
    p = theory.refreshP()

    p0 = theory.refreshP()
    jax.block_until_ready(p0)
    t0 = time.perf_counter()
    for _ in range(nit):
        p0 = theory.refreshP()
        jax.block_until_ready(p0)
    t1 = time.perf_counter()

    theory.kinetic(p).block_until_ready()
    t2 = time.perf_counter()
    for _ in range(nit):
        theory.kinetic(p).block_until_ready()
    t3 = time.perf_counter()

    theory.action(q).block_until_ready()
    t4 = time.perf_counter()
    for _ in range(nit):
        theory.action(q).block_until_ready()
    t5 = time.perf_counter()

    theory.force(q).block_until_ready()
    t6 = time.perf_counter()
    for _ in range(nit):
        theory.force(q).block_until_ready()
    t7 = time.perf_counter()

    theory.evolveQ(float(dt), p, q).block_until_ready()
    t8 = time.perf_counter()
    for _ in range(nit):
        theory.evolveQ(float(dt), p, q).block_until_ready()
    t9 = time.perf_counter()

    theory.average_spin(q).block_until_ready()
    t10 = time.perf_counter()
    for _ in range(nit):
        theory.average_spin(q).block_until_ready()
    t11 = time.perf_counter()

    theory.first_momentum_structure_factor(q, axis=int(mom_axis)).block_until_ready()
    t12 = time.perf_counter()
    for _ in range(nit):
        theory.first_momentum_structure_factor(q, axis=int(mom_axis)).block_until_ready()
    t13 = time.perf_counter()

    out = {
        "refresh_p_sec_per_call": float((t1 - t0) / nit),
        "kinetic_sec_per_call": float((t3 - t2) / nit),
        "action_sec_per_call": float((t5 - t4) / nit),
        "force_sec_per_call": float((t7 - t6) / nit),
        "evolveq_sec_per_call": float((t9 - t8) / nit),
        "avg_spin_sec_per_call": float((t11 - t10) / nit),
        "c2p_sec_per_call": float((t13 - t12) / nit),
    }

    if theory.Nd == 2 and theory.ncomp == 3:
        theory.topological_charge(q).block_until_ready()
        t14 = time.perf_counter()
        for _ in range(nit):
            theory.topological_charge(q).block_until_ready()
        t15 = time.perf_counter()
        out["topology_sec_per_call"] = float((t15 - t14) / nit)

    return out


def trajectory_timing(
    theory: ONSigmaModel,
    integrators: Iterable[str] = ("leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"),
    nmd: int = 4,
    tau: float = 1.0,
    n_iter: int = 3,
) -> dict[str, float]:
    nit = max(1, int(n_iter))
    out: dict[str, float] = {}
    q = theory.hot_start()
    p = theory.refreshP()
    for name in integrators:
        key = str(name).strip().lower()
        if not key:
            continue
        I = _build_integrator(key, theory, int(nmd), float(tau))
        p1, q1 = I.integrate(p, q)
        jax.block_until_ready(p1)
        jax.block_until_ready(q1)
        t0 = time.perf_counter()
        for _ in range(nit):
            p1, q1 = I.integrate(p, q)
            jax.block_until_ready(p1)
            jax.block_until_ready(q1)
        t1 = time.perf_counter()
        out[f"{key}_sec_per_traj"] = float((t1 - t0) / nit)
    return out


def _parse_int_list(s: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(s).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected a comma-separated list of integers.")
    return tuple(vals)


def _parse_str_list(s: str) -> tuple[str, ...]:
    vals = [str(v.strip()) for v in str(s).split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected a comma-separated list of names.")
    return tuple(vals)


def _random_direction(theory: ONSigmaModel, seed: int) -> Array:
    key = jax.random.PRNGKey(int(seed))
    return jax.random.normal(key, shape=theory.momentum_shape(), dtype=theory.dtype)


def _weak_start(theory: ONSigmaModel, scale: float = 0.05, seed: int = 0) -> Array:
    q0 = theory.coldStart()
    if float(scale) <= 0.0:
        return q0
    p0 = _random_direction(theory, int(seed))
    q1 = theory.evolveQ(jnp.asarray(scale, dtype=theory.dtype), p0, q0)
    return theory.normalize(q1)


def _rel_l2(a: Array, b: Array, eps: float = 1e-30) -> float:
    return float(np.asarray(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + eps)))


def _directional_force_check(theory: ONSigmaModel, q: Array, h: Array, eps: float) -> dict[str, float]:
    sp = theory.action(theory.evolveQ(+eps, h, q))
    sm = theory.action(theory.evolveQ(-eps, h, q))
    slope = float(np.asarray(jnp.mean((sp - sm) / (2.0 * eps))))
    force_dir = float(np.asarray(-jnp.mean(jnp.sum(theory.force(q) * h, axis=tuple(range(1, h.ndim))))))
    denom = max(1.0e-12, abs(slope), abs(force_dir))
    return {
        "fd": slope,
        "force": force_dir,
        "relerr": abs(slope - force_dir) / denom,
    }


def _autodiff_directional_check(theory: ONSigmaModel, q: Array, h: Array) -> dict[str, float]:
    def scalar(eps: Array) -> Array:
        return jnp.sum(theory.action(theory.evolveQ(eps, h, q)))

    d_ad = float(np.asarray(jax.grad(scalar)(jnp.asarray(0.0, dtype=theory.dtype))))
    d_force = float(np.asarray(-jnp.sum(theory.force(q) * h)))
    denom = max(1.0e-12, abs(d_ad), abs(d_force))
    return {
        "autodiff": d_ad,
        "force": d_force,
        "relerr": abs(d_ad - d_force) / denom,
    }


def test_reversibility(
    theory: ONSigmaModel,
    integrator: str = "leapfrog",
    tau: float = 1.0,
    nmd: int = 8,
    n_samples: int = 4,
    q_scale: float = 0.05,
) -> dict[str, float]:
    ns = max(1, int(n_samples))
    I = _build_integrator(str(integrator), theory, int(nmd), float(tau))
    rel_q = []
    rel_p = []
    abs_q = []
    abs_p = []

    for k in range(ns):
        q0 = _weak_start(theory, scale=float(q_scale), seed=1000 + 2 * k)
        p0 = _random_direction(theory, seed=2000 + 2 * k)
        p1, q1 = I.integrate(p0, q0)
        p1 = jax.block_until_ready(p1)
        q1 = jax.block_until_ready(q1)
        p2, q2 = I.integrate(-p1, q1)
        p2 = jax.block_until_ready(p2)
        q2 = jax.block_until_ready(q2)

        rel_q.append(_rel_l2(q2, q0))
        rel_p.append(_rel_l2(p2, -p0))
        abs_q.append(float(np.asarray(jnp.max(jnp.abs(q2 - q0)))))
        abs_p.append(float(np.asarray(jnp.max(jnp.abs(p2 + p0)))))

    return {
        "integrator": str(integrator),
        "tau": float(tau),
        "nmd": int(nmd),
        "n_samples": int(ns),
        "mean_rel_q_roundtrip": float(np.mean(rel_q)),
        "max_rel_q_roundtrip": float(np.max(rel_q)),
        "mean_rel_p_roundtrip": float(np.mean(rel_p)),
        "max_rel_p_roundtrip": float(np.max(rel_p)),
        "mean_abs_q_roundtrip": float(np.mean(abs_q)),
        "max_abs_q_roundtrip": float(np.max(abs_q)),
        "mean_abs_p_roundtrip": float(np.mean(abs_p)),
        "max_abs_p_roundtrip": float(np.max(abs_p)),
    }


def test_reproducibility(
    theory: ONSigmaModel,
    integrator: str = "leapfrog",
    tau: float = 1.0,
    nmd: int = 8,
    n_samples: int = 4,
    q_scale: float = 0.05,
) -> dict[str, float]:
    ns = max(1, int(n_samples))
    I = _build_integrator(str(integrator), theory, int(nmd), float(tau))
    rel_q = []
    rel_p = []
    abs_q = []
    abs_p = []

    for k in range(ns):
        q0 = _weak_start(theory, scale=float(q_scale), seed=3000 + 2 * k)
        p0 = _random_direction(theory, seed=4000 + 2 * k)

        p1a, q1a = I.integrate(jnp.array(p0), jnp.array(q0))
        p1a = jax.block_until_ready(p1a)
        q1a = jax.block_until_ready(q1a)

        p1b, q1b = I.integrate(jnp.array(p0), jnp.array(q0))
        p1b = jax.block_until_ready(p1b)
        q1b = jax.block_until_ready(q1b)

        rel_q.append(_rel_l2(q1a, q1b))
        rel_p.append(_rel_l2(p1a, p1b))
        abs_q.append(float(np.asarray(jnp.max(jnp.abs(q1a - q1b)))))
        abs_p.append(float(np.asarray(jnp.max(jnp.abs(p1a - p1b)))))

    return {
        "integrator": str(integrator),
        "tau": float(tau),
        "nmd": int(nmd),
        "n_samples": int(ns),
        "mean_rel_q_repeat": float(np.mean(rel_q)),
        "max_rel_q_repeat": float(np.max(rel_q)),
        "mean_rel_p_repeat": float(np.mean(rel_p)),
        "max_rel_p_repeat": float(np.max(rel_p)),
        "mean_abs_q_repeat": float(np.mean(abs_q)),
        "max_abs_q_repeat": float(np.max(abs_q)),
        "mean_abs_p_repeat": float(np.mean(abs_p)),
        "max_abs_p_repeat": float(np.max(abs_p)),
    }


def test_epsilon2(
    theory: ONSigmaModel,
    integrator: str = "leapfrog",
    tau: float = 1.0,
    nmd_list: tuple[int, ...] = (8, 12, 16, 24, 32),
    n_samples: int = 8,
    q_scale: float = 0.05,
) -> dict[str, object]:
    nmds = [int(n) for n in nmd_list if int(n) > 0]
    if not nmds:
        raise ValueError("nmd_list must contain at least one positive integer")

    eps = np.asarray([float(tau) / float(n) for n in nmds], dtype=np.float64)
    means = []
    stds = []
    samples = []
    ns = max(1, int(n_samples))

    for k in range(ns):
        q0 = _weak_start(theory, scale=float(q_scale), seed=5000 + 2 * k)
        p0 = _random_direction(theory, seed=6000 + 2 * k)
        samples.append((q0, p0))

    for nmd in nmds:
        I = _build_integrator(str(integrator), theory, int(nmd), float(tau))
        vals = []
        for q0, p0 in samples:
            H0 = theory.kinetic(p0) + theory.action(q0)
            p1, q1 = I.integrate(p0, q0)
            H1 = theory.kinetic(p1) + theory.action(q1)
            dH = np.asarray(jax.device_get(jnp.abs(H1 - H0)), dtype=np.float64).reshape(-1)
            vals.extend(dH.tolist())
        v = np.asarray(vals, dtype=np.float64)
        means.append(float(np.mean(v)))
        stds.append(float(np.std(v, ddof=1)) if v.size > 1 else 0.0)

    means_arr = np.asarray(means, dtype=np.float64)
    floor = max(1e-16, float(np.max(means_arr)) * 1e-8)
    mask = means_arr > floor
    if int(np.sum(mask)) >= 2:
        slope, intercept = np.polyfit(np.log(eps[mask]), np.log(means_arr[mask]), 1)
        fit_n = int(np.sum(mask))
    else:
        slope = float("nan")
        intercept = float("nan")
        fit_n = int(np.sum(mask))

    return {
        "integrator": str(integrator),
        "nmd": nmds,
        "eps": eps.tolist(),
        "mean_abs_dh": means,
        "std_abs_dh": stds,
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_points": int(fit_n),
        "fit_floor": float(floor),
    }


def test_epsilon4(
    theory: ONSigmaModel,
    integrators: tuple[str, ...] = ("forcegrad", "minnorm4pf4"),
    tau: float = 1.0,
    nmd_list: tuple[int, ...] = (4, 6, 8, 12, 16),
    n_samples: int = 8,
    q_scale: float = 0.05,
) -> dict[str, object]:
    out = {}
    for name in integrators:
        out[str(name)] = test_epsilon2(
            theory,
            integrator=str(name),
            tau=float(tau),
            nmd_list=tuple(int(v) for v in nmd_list),
            n_samples=int(n_samples),
            q_scale=float(q_scale),
        )
    return out


@dataclass
class SigmaSmokeConfig:
    nmd: int = 8
    tau: float = 1.0
    warmup_no_ar: int = 2
    warmup_ar: int = 8
    nmeas: int = 24
    q_scale: float = 0.2
    iat_method: str = "gamma"
    sigma_cut: float = 4.0
    seed: int = 1234
    smd_gamma: float = 0.3
    use_fast_jit: bool = True
    reproducibility_steps: int = 3
    reproducibility_tol: float = 1e-7


def _iat_error(x: np.ndarray, method: str) -> dict[str, float]:
    iat = integrated_autocorr_time(x, method=str(method))
    tau = float(iat.get("tau_int", float("nan")))
    ess = float(iat.get("ess", float("nan")))
    win = int(iat.get("window", 0))
    ok = bool(iat.get("ok", False))
    sigma_mean = float(iat.get("sigma_mean", float("nan")))
    if (not np.isfinite(sigma_mean)) or sigma_mean <= 0.0:
        s = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        if np.isfinite(ess) and ess > 1.0 and np.isfinite(s):
            sigma_mean = s / math.sqrt(ess)
        elif np.isfinite(s) and x.size > 1:
            sigma_mean = s / math.sqrt(max(1, x.size - 1))
    return {
        "err_iat": float(sigma_mean),
        "tau_int": tau,
        "ess": ess,
        "iat_window": float(win),
        "iat_ok": float(1.0 if ok else 0.0),
    }


def _sigma_no_ar_step_hmc(q: Array, chain) -> Array:
    p0 = chain.T.refreshP()
    _, q1 = chain.I.integrate(p0, q)
    return jax.block_until_ready(q1)


def _sigma_step(q: Array, chain, update: str, *, warmup: bool = False) -> Array:
    if update == "smd":
        return chain.evolve(q, 1, warmup=bool(warmup))
    return chain.evolve(q, 1)


def _run_sigma_chain(model_factory, *, update: str, seed: int, cfg: SigmaSmokeConfig) -> dict[str, object]:
    theory = model_factory(int(seed))
    integ = force_gradient(theory.force, theory.evolveQ, int(cfg.nmd), float(cfg.tau))
    if update == "hmc":
        chain = hmc(T=theory, I=integ, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
    elif update == "smd":
        chain = SMD(
            T=theory,
            I=integ,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )
    else:
        raise ValueError(f"Unknown update type: {update}")

    q = _weak_start(theory, scale=float(cfg.q_scale), seed=int(seed) + 17)

    for _ in range(max(0, int(cfg.warmup_no_ar))):
        if update == "hmc":
            q = _sigma_no_ar_step_hmc(q, chain)
        else:
            q = _sigma_step(q, chain, update, warmup=True)

    chain.reset_acceptance()
    for _ in range(max(0, int(cfg.warmup_ar))):
        q = _sigma_step(q, chain, update, warmup=False)
    warm_acc = float(chain.calc_acceptance())

    chain.reset_acceptance()
    energy = np.empty((max(1, int(cfg.nmeas)),), dtype=np.float64)
    for k in range(energy.size):
        q = _sigma_step(q, chain, update, warmup=False)
        energy[k] = float(np.mean(np.asarray(theory.action(q), dtype=np.float64)) / float(theory.Vol))

    out = {
        "update": str(update),
        "seed": int(seed),
        "mean_energy_density": float(np.mean(energy)),
        "warmup_acceptance": warm_acc,
        "meas_acceptance": float(chain.calc_acceptance()),
    }
    out.update(_iat_error(energy, method=str(cfg.iat_method)))
    out["series"] = energy.tolist()
    return out


def _compare_smoke_runs(a: dict[str, object], b: dict[str, object], sigma_cut: float) -> dict[str, object]:
    d = abs(float(a["mean_energy_density"]) - float(b["mean_energy_density"]))
    ea = float(a.get("err_iat", float("nan")))
    eb = float(b.get("err_iat", float("nan")))
    comb = math.sqrt(max(0.0, ea * ea + eb * eb)) if np.isfinite(ea) and np.isfinite(eb) else float("nan")
    sigma = d / comb if np.isfinite(comb) and comb > 0.0 else float("inf")
    return {
        "delta": float(d),
        "sigma": float(sigma),
        "agree": bool(np.isfinite(sigma) and sigma < float(sigma_cut)),
    }


def _reproducibility_smoke(model_factory, *, update: str, seed: int, cfg: SigmaSmokeConfig) -> dict[str, object]:
    theory1 = model_factory(int(seed))
    theory2 = model_factory(int(seed))
    integ1 = force_gradient(theory1.force, theory1.evolveQ, int(cfg.nmd), float(cfg.tau))
    integ2 = force_gradient(theory2.force, theory2.evolveQ, int(cfg.nmd), float(cfg.tau))

    if update == "hmc":
        chain1 = hmc(T=theory1, I=integ1, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
        chain2 = hmc(T=theory2, I=integ2, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
    else:
        chain1 = SMD(
            T=theory1,
            I=integ1,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )
        chain2 = SMD(
            T=theory2,
            I=integ2,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )

    q1 = _weak_start(theory1, scale=float(cfg.q_scale), seed=int(seed) + 31)
    q2 = _weak_start(theory2, scale=float(cfg.q_scale), seed=int(seed) + 31)

    max_abs = 0.0
    for _ in range(max(1, int(cfg.reproducibility_steps))):
        q1 = _sigma_step(q1, chain1, update, warmup=False)
        q2 = _sigma_step(q2, chain2, update, warmup=False)
        d = float(np.asarray(jnp.max(jnp.abs(q1 - q2))))
        max_abs = max(max_abs, d)

    return {
        "max_abs_diff": float(max_abs),
        "tol": float(cfg.reproducibility_tol),
        "ok": bool(max_abs <= float(cfg.reproducibility_tol)),
    }


def run_on_mcmc_smoke_suite(model_name: str, model_factory, config: SigmaSmokeConfig | None = None) -> dict[str, object]:
    cfg = config if config is not None else SigmaSmokeConfig()
    runs = {
        "hmc": _run_sigma_chain(model_factory, update="hmc", seed=int(cfg.seed), cfg=cfg),
        "smd": _run_sigma_chain(model_factory, update="smd", seed=int(cfg.seed) + 1, cfg=cfg),
    }
    comparisons = {
        "hmc_vs_smd": _compare_smoke_runs(runs["hmc"], runs["smd"], sigma_cut=float(cfg.sigma_cut)),
    }
    reproducibility = {
        "hmc": _reproducibility_smoke(model_factory, update="hmc", seed=int(cfg.seed) + 100, cfg=cfg),
        "smd": _reproducibility_smoke(model_factory, update="smd", seed=int(cfg.seed) + 101, cfg=cfg),
    }
    ok = True
    ok = ok and all(bool(c["agree"]) for c in comparisons.values())
    ok = ok and all(bool(r["ok"]) for r in reproducibility.values())
    return {
        "model": str(model_name),
        "runs": runs,
        "comparisons": comparisons,
        "reproducibility": reproducibility,
        "pass": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser(description="O(N) sigma-model diagnostics and self-checks")
    ap.add_argument("--shape", type=str, default="32,32")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--layout", type=str, default="auto", choices=["BN...", "B...N", "auto"])
    ap.add_argument("--exp-method", type=str, default="auto", choices=["auto", "expm", "rodrigues"])
    ap.add_argument("--tests", type=str, default="selfcheck")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--selfcheck-fail", action="store_true")
    ap.add_argument("--nmd", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--layout-integrator", type=str, default="minnorm2", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--n-iter-timing", type=int, default=10)
    ap.add_argument("--timing-dt", type=float, default=0.1)
    ap.add_argument("--timing-mom-axis", type=int, default=0)
    ap.add_argument("--timing-integrators", type=str, default="leapfrog,minnorm2,forcegrad,minnorm4pf4")
    ap.add_argument("--fd-eps", type=float, default=1.0e-4)
    ap.add_argument("--eps2-integrator", type=str, default="leapfrog", choices=["leapfrog", "minnorm2"])
    ap.add_argument("--eps2-tau", type=float, default=1.0)
    ap.add_argument("--eps2-nmd-list", type=str, default="8,12,16,24,32")
    ap.add_argument("--eps2-samples", type=int, default=8)
    ap.add_argument("--eps2-qscale", type=float, default=0.05)
    ap.add_argument("--eps4-integrators", type=str, default="forcegrad,minnorm4pf4")
    ap.add_argument("--eps4-tau", type=float, default=1.0)
    ap.add_argument("--eps4-nmd-list", type=str, default="4,6,8,12,16")
    ap.add_argument("--eps4-samples", type=int, default=8)
    ap.add_argument("--eps4-qscale", type=float, default=0.05)
    ap.add_argument("--rev-integrator", type=str, default="leapfrog", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--rev-tau", type=float, default=1.0)
    ap.add_argument("--rev-nmd", type=int, default=8)
    ap.add_argument("--rev-samples", type=int, default=4)
    ap.add_argument("--rev-qscale", type=float, default=0.05)
    ap.add_argument("--repro-integrator", type=str, default="leapfrog", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--repro-tau", type=float, default=1.0)
    ap.add_argument("--repro-nmd", type=int, default=8)
    ap.add_argument("--repro-samples", type=int, default=4)
    ap.add_argument("--repro-qscale", type=float, default=0.05)
    ap.add_argument("--mcmcsmoke-seed", type=int, default=1234)
    ap.add_argument("--mcmcsmoke-noar", type=int, default=2)
    ap.add_argument("--mcmcsmoke-warmup", type=int, default=8)
    ap.add_argument("--mcmcsmoke-meas", type=int, default=24)
    ap.add_argument("--mcmcsmoke-nmd", type=int, default=8)
    ap.add_argument("--mcmcsmoke-tau", type=float, default=1.0)
    ap.add_argument("--mcmcsmoke-qscale", type=float, default=0.2)
    ap.add_argument("--mcmcsmoke-iat", type=str, default="gamma", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--mcmcsmoke-sigma-cut", type=float, default=4.0)
    ap.add_argument("--mcmcsmoke-smd-gamma", type=float, default=0.3)
    ap.add_argument("--mcmcsmoke-repro-steps", type=int, default=3)
    ap.add_argument("--mcmcsmoke-repro-tol", type=float, default=1e-7)
    args = ap.parse_args()

    tests = {t.strip().lower() for t in str(args.tests).split(",") if t.strip()}
    if args.selfcheck:
        tests = {"selfcheck"}
    if "all" in tests:
        tests = {
            "layout",
            "timing",
            "fd",
            "autodiff",
            "unit",
            "topo",
            "eps2",
            "eps4",
            "reversibility",
            "reproducibility",
            "mcmcsmoke",
            "selfcheck",
        }

    lattice_shape = _parse_shape(args.shape)
    selected_layout = args.layout
    if selected_layout == "auto" or "layout" in tests:
        bench_action = ONSigmaModel.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=float(args.beta),
            ncomp=int(args.ncomp),
            batch_size=int(args.batch_size),
            exp_method=str(args.exp_method),
            n_iter=max(1, int(args.n_iter_timing)),
            kernel="action",
        )
        bench_force = ONSigmaModel.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=float(args.beta),
            ncomp=int(args.ncomp),
            batch_size=int(args.batch_size),
            exp_method=str(args.exp_method),
            n_iter=max(1, int(args.n_iter_timing)),
            kernel="force",
        )
        bench_evolve = ONSigmaModel.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=float(args.beta),
            ncomp=int(args.ncomp),
            batch_size=int(args.batch_size),
            exp_method=str(args.exp_method),
            n_iter=max(1, int(args.n_iter_timing)),
            kernel="evolveq",
        )
        bench_traj = ONSigmaModel.benchmark_layout_trajectory(
            lattice_shape=lattice_shape,
            beta=float(args.beta),
            ncomp=int(args.ncomp),
            integrator=str(args.layout_integrator),
            nmd=int(args.nmd),
            tau=float(args.tau),
            batch_size=int(args.batch_size),
            exp_method=str(args.exp_method),
            n_iter=max(1, int(args.n_iter_timing)),
        )
        print("Layout benchmark action (sec/call):", bench_action)
        print("Layout benchmark force  (sec/call):", bench_force)
        print("Layout benchmark evolveQ(sec/call):", bench_evolve)
        print(f"Layout benchmark {args.layout_integrator} trajectory (sec/traj):", bench_traj)
        if selected_layout == "auto":
            selected_layout = min(bench_traj, key=bench_traj.get)
            print("Auto-selected layout:", selected_layout)

    th = ONSigmaModel(
        lattice_shape=lattice_shape,
        beta=float(args.beta),
        ncomp=int(args.ncomp),
        batch_size=int(args.batch_size),
        layout=selected_layout,
        exp_method=str(args.exp_method),
    )
    q = th.hotStart()
    p = th.refreshP()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(f"shape={lattice_shape} ncomp={th.ncomp} beta={th.beta} layout={th.layout} exp_method={th.exp_method}")

    failures: list[str] = []
    smoke_failed = False

    if "timing" in tests or "selfcheck" in tests:
        kt = kernel_timing(
            th,
            n_iter=max(1, int(args.n_iter_timing)),
            dt=float(args.timing_dt),
            mom_axis=int(args.timing_mom_axis),
        )
        print("Kernel timing:")
        for key, val in kt.items():
            print(f"  {key}: {val:.6e}")

        tt = trajectory_timing(
            th,
            integrators=_parse_str_list(args.timing_integrators),
            nmd=int(args.nmd),
            tau=float(args.tau),
            n_iter=max(1, int(args.n_iter_timing)),
        )
        print("Trajectory timing:")
        for key, val in tt.items():
            print(f"  {key}: {val:.6e}")

        if th.ncomp == 3:
            exb = benchmark_exponentiation(
                th,
                dt=float(args.timing_dt),
                n_iter=max(1, int(args.n_iter_timing)),
            )
            print("O(3) exponentiation benchmark:")
            for key, val in exb.items():
                print(f"  {key}: {val:.6e}")

    if "unit" in tests or "selfcheck" in tests:
        q1 = th.evolveQ(0.5, p, q)
        ql = th._field_component_last(q1)
        unit_err = float(np.asarray(jnp.max(jnp.abs(jnp.sum(ql * ql, axis=-1) - 1.0))))
        print(f"unit: max |sigma^2-1| = {unit_err:.3e}")
        if unit_err > 5.0e-5:
            failures.append("unit")

    if "fd" in tests or "selfcheck" in tests:
        h = _random_direction(th, seed=123)
        rep = _directional_force_check(th, q, h, float(args.fd_eps))
        print(
            "fd:"
            f" slope={rep['fd']:.6e}"
            f" -<F,H>={rep['force']:.6e}"
            f" relerr={rep['relerr']:.3e}"
        )
        if rep["relerr"] > 5.0e-3:
            failures.append("fd")

    if "autodiff" in tests or "selfcheck" in tests:
        h = _random_direction(th, seed=456)
        rep = _autodiff_directional_check(th, q, h)
        print(
            "autodiff:"
            f" dS/deps={rep['autodiff']:.6e}"
            f" -<F,H>={rep['force']:.6e}"
            f" relerr={rep['relerr']:.3e}"
        )
        if rep["relerr"] > 5.0e-5:
            failures.append("autodiff")

    if ("topo" in tests or "selfcheck" in tests) and th.Nd == 2 and th.ncomp == 3:
        qc = th.coldStart()
        qhot = th.hotStart()
        q_cold = float(np.asarray(jnp.mean(th.topological_charge(qc))))
        q_hot = float(np.asarray(jnp.mean(th.topological_charge(qhot))))
        print(f"topology: Q(cold)={q_cold:.6e}  Q(hot)={q_hot:.6e}")
        if abs(q_cold) > 1.0e-6:
            failures.append("topo")

    if "eps2" in tests or "selfcheck" in tests:
        eps2 = test_epsilon2(
            th,
            integrator=str(args.eps2_integrator),
            tau=float(args.eps2_tau),
            nmd_list=_parse_int_list(args.eps2_nmd_list),
            n_samples=int(args.eps2_samples),
            q_scale=float(args.eps2_qscale),
        )
        print("eps2:")
        print(f"  integrator: {eps2['integrator']}")
        print(f"  nmd: {eps2['nmd']}")
        print(f"  eps: {[f'{x:.3e}' for x in eps2['eps']]}")
        print(f"  mean |dH|: {[f'{x:.3e}' for x in eps2['mean_abs_dh']]}")
        print(f"  std  |dH|: {[f'{x:.3e}' for x in eps2['std_abs_dh']]}")
        print(f"  log-log slope (expected ~2): {float(eps2['slope']):.4f}")
        slope = float(eps2["slope"])
        if (not np.isfinite(slope)) or slope < 1.5 or slope > 2.5:
            failures.append("eps2")

    if "eps4" in tests or "selfcheck" in tests:
        eps4 = test_epsilon4(
            th,
            integrators=_parse_str_list(args.eps4_integrators),
            tau=float(args.eps4_tau),
            nmd_list=_parse_int_list(args.eps4_nmd_list),
            n_samples=int(args.eps4_samples),
            q_scale=float(args.eps4_qscale),
        )
        print("eps4:")
        for name in _parse_str_list(args.eps4_integrators):
            e = eps4[str(name)]
            slope = float(e["slope"])
            print(f"  {name}: slope={slope:.4f} eps={[f'{x:.3e}' for x in e['eps']]}")
            if (not np.isfinite(slope)) or slope < 3.0 or slope > 5.0:
                failures.append(f"eps4:{name}")

    if "reversibility" in tests or "selfcheck" in tests:
        rv = test_reversibility(
            th,
            integrator=str(args.rev_integrator),
            tau=float(args.rev_tau),
            nmd=int(args.rev_nmd),
            n_samples=int(args.rev_samples),
            q_scale=float(args.rev_qscale),
        )
        print("reversibility:")
        print(
            f"  rel q/p roundtrip (mean,max): {rv['mean_rel_q_roundtrip']:.6e} / {rv['max_rel_q_roundtrip']:.6e}"
            f" ; {rv['mean_rel_p_roundtrip']:.6e} / {rv['max_rel_p_roundtrip']:.6e}"
        )
        print(
            f"  abs q/p roundtrip (mean,max): {rv['mean_abs_q_roundtrip']:.6e} / {rv['max_abs_q_roundtrip']:.6e}"
            f" ; {rv['mean_abs_p_roundtrip']:.6e} / {rv['max_abs_p_roundtrip']:.6e}"
        )
        if rv["max_rel_q_roundtrip"] > 1.0e-4 or rv["max_rel_p_roundtrip"] > 1.0e-4:
            failures.append("reversibility")

    if "reproducibility" in tests or "selfcheck" in tests:
        rp = test_reproducibility(
            th,
            integrator=str(args.repro_integrator),
            tau=float(args.repro_tau),
            nmd=int(args.repro_nmd),
            n_samples=int(args.repro_samples),
            q_scale=float(args.repro_qscale),
        )
        print("reproducibility:")
        print(
            f"  rel q/p repeat (mean,max): {rp['mean_rel_q_repeat']:.6e} / {rp['max_rel_q_repeat']:.6e}"
            f" ; {rp['mean_rel_p_repeat']:.6e} / {rp['max_rel_p_repeat']:.6e}"
        )
        print(
            f"  abs q/p repeat (mean,max): {rp['mean_abs_q_repeat']:.6e} / {rp['max_abs_q_repeat']:.6e}"
            f" ; {rp['mean_abs_p_repeat']:.6e} / {rp['max_abs_p_repeat']:.6e}"
        )
        if rp["max_abs_q_repeat"] > 1.0e-7 or rp["max_abs_p_repeat"] > 1.0e-7:
            failures.append("reproducibility")

    if "mcmcsmoke" in tests:
        cfg = SigmaSmokeConfig(
            nmd=int(args.mcmcsmoke_nmd),
            tau=float(args.mcmcsmoke_tau),
            warmup_no_ar=int(args.mcmcsmoke_noar),
            warmup_ar=int(args.mcmcsmoke_warmup),
            nmeas=int(args.mcmcsmoke_meas),
            q_scale=float(args.mcmcsmoke_qscale),
            iat_method=str(args.mcmcsmoke_iat),
            sigma_cut=float(args.mcmcsmoke_sigma_cut),
            seed=int(args.mcmcsmoke_seed),
            smd_gamma=float(args.mcmcsmoke_smd_gamma),
            reproducibility_steps=int(args.mcmcsmoke_repro_steps),
            reproducibility_tol=float(args.mcmcsmoke_repro_tol),
        )

        def _factory(seed: int):
            return ONSigmaModel(
                lattice_shape=lattice_shape,
                beta=float(args.beta),
                ncomp=int(args.ncomp),
                batch_size=int(args.batch_size),
                layout=selected_layout,
                seed=int(seed),
                exp_method=str(args.exp_method),
            )

        s = run_on_mcmc_smoke_suite(f"O({th.ncomp})", _factory, config=cfg)
        print("mcmc smoke:")
        print(
            "  HMC:"
            f" E/V={s['runs']['hmc']['mean_energy_density']:.8f}"
            f" +/- {s['runs']['hmc']['err_iat']:.6e}"
            f" acc={s['runs']['hmc']['meas_acceptance']:.4f}"
        )
        print(
            "  SMD:"
            f" E/V={s['runs']['smd']['mean_energy_density']:.8f}"
            f" +/- {s['runs']['smd']['err_iat']:.6e}"
            f" acc={s['runs']['smd']['meas_acceptance']:.4f}"
        )
        cmp = s["comparisons"]["hmc_vs_smd"]
        print(
            "  cmp HMC vs SMD:"
            f" delta={cmp['delta']:.6e} sigma={cmp['sigma']:.3f} agree={bool(cmp['agree'])}"
        )
        rh = s["reproducibility"]["hmc"]
        rs = s["reproducibility"]["smd"]
        print(
            "  reproducibility HMC:"
            f" max_abs={rh['max_abs_diff']:.3e} tol={rh['tol']:.3e} ok={bool(rh['ok'])}"
        )
        print(
            "  reproducibility SMD:"
            f" max_abs={rs['max_abs_diff']:.3e} tol={rs['tol']:.3e} ok={bool(rs['ok'])}"
        )
        print(f"  pass={bool(s['pass'])}")
        smoke_failed = not bool(s["pass"])

    final_failures = list(failures)
    if smoke_failed:
        final_failures.append("mcmcsmoke")
    ok = not final_failures
    print("selfcheck:", "PASS" if ok else f"FAIL ({', '.join(final_failures)})")
    if bool(args.selfcheck_fail) and (not ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Pure SU(2) Yang-Mills theory in link coordinates for HMC."""

from __future__ import annotations

import argparse
import math
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxqft.LieGroups import su2 as su2_lg
from jaxqft.models.su3_ym import _dagger


Array = jax.Array


def _trace2(x: Array) -> Array:
    return x[..., 0, 0] + x[..., 1, 1]


def _mm2(a: Array, b: Array) -> Array:
    """Specialized 2x2 complex matmul; faster than generic batched @ on tiny matrices."""
    a00, a01 = a[..., 0, 0], a[..., 0, 1]
    a10, a11 = a[..., 1, 0], a[..., 1, 1]
    b00, b01 = b[..., 0, 0], b[..., 0, 1]
    b10, b11 = b[..., 1, 0], b[..., 1, 1]
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11
    return jnp.stack((jnp.stack((c00, c01), axis=-1), jnp.stack((c10, c11), axis=-1)), axis=-2)


def _project_su2_algebra_2x2(x: Array) -> Array:
    y = 0.5 * (x - _dagger(x))
    tr = (_trace2(y) / 2.0)[..., None, None]
    eye = jnp.eye(2, dtype=x.dtype)
    return y - tr * eye


def project_su2_algebra(x: Array) -> Array:
    y = 0.5 * (x - _dagger(x))
    tr = jnp.einsum("...aa->...", y)[..., None, None] / 2.0
    eye = jnp.eye(2, dtype=x.dtype)
    return y - tr * eye


def _algebra_inner(a: Array, b: Array) -> Array:
    trab = jnp.real(jnp.einsum("...ab,...ba->...", a, b))
    return -jnp.sum(trab, axis=tuple(range(1, trab.ndim)))


def _plaquette_sum_bmxyij_nd4_optimized(U: Array) -> Array:
    """Return sum_{x,mu<nu} Re Tr U_{mu,nu}(x) for BMXYIJ/4D."""
    B = U.shape[0]
    plaq_sum = jnp.zeros((B,), dtype=jnp.float32)
    U_plus = [jnp.roll(U, -1, axis=2 + d) for d in range(4)]
    for mu in range(4):
        U_mu = U[:, mu]
        for nu in range(mu + 1, 4):
            U_nu = U[:, nu]
            U_nu_xpmu = U_plus[mu][:, nu]
            U_mu_xpnu = U_plus[nu][:, mu]
            plaq = _mm2(_mm2(_mm2(U_mu, U_nu_xpmu), _dagger(U_mu_xpnu)), _dagger(U_nu))
            trp = jnp.real(_trace2(plaq))
            plaq_sum = plaq_sum + jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
    return plaq_sum


def _action_bmxyij_nd4_optimized(U: Array, beta: float) -> Array:
    return -(beta / 2.0) * _plaquette_sum_bmxyij_nd4_optimized(U)


def _force_bmxyij_nd4_optimized(U: Array, beta: float) -> Array:
    coef = -(beta / 2.0)
    U_plus = [jnp.roll(U, -1, axis=2 + d) for d in range(4)]
    U_minus = [jnp.roll(U, +1, axis=2 + d) for d in range(4)]
    U_minus_nu = [U_minus[nu][:, nu] for nu in range(4)]
    U_minus_nu_dag = [_dagger(x) for x in U_minus_nu]
    out = []
    for mu in range(4):
        U_mu = U[:, mu]
        staple = jnp.zeros_like(U_mu)
        for nu in range(4):
            if nu == mu:
                continue
            U_nu = U[:, nu]
            U_mu_xpnu = U_plus[nu][:, mu]
            U_nu_xpmu = U_plus[mu][:, nu]
            fwd = _mm2(_mm2(U_nu, U_mu_xpnu), _dagger(U_nu_xpmu))

            U_nu_m = U_minus_nu[nu]
            U_nu_m_dag = U_minus_nu_dag[nu]
            U_mu_m = U_minus[nu][:, mu]
            U_nu_m_xpmu = jnp.roll(U_nu_m, -1, axis=1 + mu)
            bwd = _mm2(_mm2(U_nu_m_dag, U_mu_m), U_nu_m_xpmu)
            staple = staple + fwd + bwd
        out.append(coef * _project_su2_algebra_2x2(_mm2(U_mu, _dagger(staple))))
    return jnp.stack(out, axis=1)


def _evolve_q_su2_fused(dt: float, p: Array, q: Array) -> Array:
    """Fast drift update for SU(2): compute exp(dt*p) @ q without materializing exp(dt*p)."""
    x = dt * p
    eps = jnp.finfo(x.real.dtype).eps
    n = (jnp.linalg.norm(x, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + eps
    return jnp.cos(n) * q + (jnp.sin(n) / n) * _mm2(x, q)


@dataclass
class SU2YangMills:
    lattice_shape: Tuple[int, ...]
    beta: float
    batch_size: int = 1
    layout: str = "BMXYIJ"
    seed: int = 0
    exp_method: str = "su2"
    dtype: jnp.dtype = jnp.complex64
    key: Array = field(init=False, repr=False)

    def __post_init__(self):
        self.lattice_shape = tuple(int(v) for v in self.lattice_shape)
        self.Nd = len(self.lattice_shape)
        self.Bs = int(self.batch_size)
        self.layout = self.layout.upper()
        self.layout = {"BM...IJ": "BMXYIJ", "B...MIJ": "BXYMIJ"}.get(self.layout, self.layout)
        if self.layout not in ("BMXYIJ", "BXYMIJ"):
            raise ValueError("layout must be one of: BMXYIJ, BXYMIJ, BM...IJ, B...MIJ")
        self.exp_method = self.exp_method.lower()
        if self.exp_method not in ("su2", "expm"):
            raise ValueError("exp_method must be one of: su2, expm")
        self.key = jax.random.PRNGKey(self.seed)
        self._set_axes()
        self._use_optimized_action = (self.Nd == 4 and self.layout == "BMXYIJ")
        self._action_is_jitted = False
        if self._use_optimized_action:
            beta = float(self.beta)
            self._action_opt = jax.jit(lambda U: _action_bmxyij_nd4_optimized(U, beta))
            self._action_is_jitted = True
        self._use_optimized_force = (self.Nd == 4 and self.layout == "BMXYIJ")
        self._force_is_jitted = False
        if self._use_optimized_force:
            beta = float(self.beta)
            self._force_opt = jax.jit(lambda U: _force_bmxyij_nd4_optimized(U, beta))
            self._force_is_jitted = True
        self._use_optimized_evolve_q = self.exp_method == "su2"
        self._evolve_q_is_jitted = False
        if self._use_optimized_evolve_q:
            self._evolve_q_opt = jax.jit(_evolve_q_su2_fused)
            self._evolve_q_is_jitted = True

    def _set_axes(self):
        if self.layout == "BMXYIJ":
            self.mu_axis = 1
            self.lat_axes = tuple(range(2, 2 + self.Nd))
        else:
            self.mu_axis = 1 + self.Nd
            self.lat_axes = tuple(range(1, 1 + self.Nd))

    def _split_key(self):
        self.key, sub = jax.random.split(self.key)
        return sub

    def field_shape(self):
        if self.layout == "BMXYIJ":
            return (self.Bs, self.Nd, *self.lattice_shape, 2, 2)
        return (self.Bs, *self.lattice_shape, self.Nd, 2, 2)

    def _take_mu(self, x: Array, mu: int) -> Array:
        return jnp.take(x, mu, axis=self.mu_axis)

    def _roll(self, x: Array, shift: int, direction: int) -> Array:
        if x.ndim == self.Nd + 3:
            ax = 1 + direction
        elif x.ndim == self.Nd + 4:
            ax = self.lat_axes[direction]
        else:
            raise ValueError(f"Unsupported tensor rank for lattice roll: ndim={x.ndim}")
        return jnp.roll(x, shift=shift, axis=ax)

    def _sample_algebra(self, scale: float = 1.0) -> Array:
        k1 = self._split_key()
        re = jax.random.normal(k1, self.field_shape(), dtype=jnp.float32)
        k2 = self._split_key()
        im = jax.random.normal(k2, self.field_shape(), dtype=jnp.float32)
        a = scale * (re + 1j * im).astype(self.dtype)
        return project_su2_algebra(a)

    def algebra_to_links(self, q: Array) -> Array:
        q = project_su2_algebra(q)
        if self.exp_method == "su2":
            return su2_lg.expo(q)
        return jax.scipy.linalg.expm(q)

    def action_reference(self, U: Array) -> Array:
        B = U.shape[0]
        S = jnp.zeros((B,), dtype=jnp.float32)
        for mu in range(self.Nd):
            U_mu = self._take_mu(U, mu)
            for nu in range(mu + 1, self.Nd):
                U_nu = self._take_mu(U, nu)
                U_nu_xpmu = self._roll(U_nu, -1, mu)
                U_mu_xpnu = self._roll(U_mu, -1, nu)
                plaq = U_mu @ U_nu_xpmu @ _dagger(U_mu_xpnu) @ _dagger(U_nu)
                trp = jnp.real(jnp.einsum("...aa->...", plaq))
                S = S - (self.beta / 2.0) * jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
        return S

    def action(self, U: Array) -> Array:
        if self._use_optimized_action:
            return self._action_opt(U)
        return self.action_reference(U)

    def action_optimized_unjitted(self, U: Array) -> Array:
        if not self._use_optimized_action:
            return self.action_reference(U)
        return _action_bmxyij_nd4_optimized(U, self.beta)

    def _staple(self, U: Array, mu: int) -> Array:
        U_mu = self._take_mu(U, mu)
        staple = jnp.zeros_like(U_mu)
        for nu in range(self.Nd):
            if nu == mu:
                continue
            U_nu = self._take_mu(U, nu)
            fwd = U_nu @ self._roll(U_mu, -1, nu) @ _dagger(self._roll(U_nu, -1, mu))
            U_nu_m = self._roll(U_nu, +1, nu)
            bwd = _dagger(U_nu_m) @ self._roll(U_mu, +1, nu) @ self._roll(U_nu_m, -1, mu)
            staple = staple + fwd + bwd
        return staple

    def force_reference(self, U: Array) -> Array:
        links_force = []
        coef = -(self.beta / 2.0)
        for mu in range(self.Nd):
            U_mu = self._take_mu(U, mu)
            staple_mu = self._staple(U, mu)
            G_mu = coef * project_su2_algebra(U_mu @ _dagger(staple_mu))
            links_force.append(G_mu)
        if self.layout == "BMXYIJ":
            return jnp.stack(links_force, axis=1)
        return jnp.stack(links_force, axis=1 + self.Nd)

    def force(self, U: Array) -> Array:
        if self._use_optimized_force:
            return self._force_opt(U)
        return self.force_reference(U)

    def force_optimized_unjitted(self, U: Array) -> Array:
        if not self._use_optimized_force:
            return self.force_reference(U)
        return _force_bmxyij_nd4_optimized(U, self.beta)

    def refresh_p(self) -> Array:
        return self._sample_algebra(scale=1.0)

    def refresh_p_with_key(self, key: Array) -> Array:
        k1, k2 = jax.random.split(key)
        re = jax.random.normal(k1, self.field_shape(), dtype=jnp.float32)
        im = jax.random.normal(k2, self.field_shape(), dtype=jnp.float32)
        a = (re + 1j * im).astype(self.dtype)
        return project_su2_algebra(a)

    def evolve_q_reference(self, dt: float, P: Array, Q: Array) -> Array:
        return self.algebra_to_links(dt * P) @ Q

    def evolve_q(self, dt: float, P: Array, Q: Array) -> Array:
        if self._use_optimized_evolve_q:
            return self._evolve_q_opt(dt, P, Q)
        return self.evolve_q_reference(dt, P, Q)

    def evolve_q_optimized_unjitted(self, dt: float, P: Array, Q: Array) -> Array:
        if not self._use_optimized_evolve_q:
            return self.evolve_q_reference(dt, P, Q)
        return _evolve_q_su2_fused(dt, P, Q)

    def kinetic(self, P: Array) -> Array:
        trp2 = jnp.real(jnp.einsum("...ab,...ba->...", P, P))
        return -0.5 * jnp.sum(trp2, axis=tuple(range(1, trp2.ndim)))

    def hot_start(self, scale: float = 0.1) -> Array:
        return self.algebra_to_links(self._sample_algebra(scale=scale))

    def average_plaquette(self, U: Array) -> Array:
        vol = int(math.prod(self.lattice_shape))
        nplanes = self.Nd * (self.Nd - 1) // 2
        if self._use_optimized_action:
            plaq_sum = _plaquette_sum_bmxyij_nd4_optimized(U)
        else:
            B = U.shape[0]
            plaq_sum = jnp.zeros((B,), dtype=jnp.float32)
            for mu in range(self.Nd):
                U_mu = self._take_mu(U, mu)
                for nu in range(mu + 1, self.Nd):
                    U_nu = self._take_mu(U, nu)
                    U_nu_xpmu = self._roll(U_nu, -1, mu)
                    U_mu_xpnu = self._roll(U_mu, -1, nu)
                    plaq = U_mu @ U_nu_xpmu @ _dagger(U_mu_xpnu) @ _dagger(U_nu)
                    trp = jnp.real(jnp.einsum("...aa->...", plaq))
                    plaq_sum = plaq_sum + jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
        return plaq_sum / (2.0 * vol * nplanes)

    @staticmethod
    def benchmark_layout(
        lattice_shape: Tuple[int, ...],
        beta: float,
        batch_size: int = 1,
        layouts: Iterable[str] = ("BMXYIJ", "BXYMIJ"),
        n_iter: int = 3,
        seed: int = 0,
        kernel: str = "action",
        exp_method: str = "su2",
    ) -> Dict[str, float]:
        kernel = kernel.lower()
        if kernel not in ("action", "force"):
            raise ValueError("kernel must be 'action' or 'force'")
        timings = {}
        for lay in layouts:
            th = SU2YangMills(
                lattice_shape=lattice_shape,
                beta=beta,
                batch_size=batch_size,
                layout=lay,
                seed=seed,
                exp_method=exp_method,
            )
            q = th.hot_start()
            if kernel == "action":
                th.action(q).block_until_ready()
            else:
                th.force(q).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                if kernel == "action":
                    th.action(q).block_until_ready()
                else:
                    th.force(q).block_until_ready()
            t1 = time.perf_counter()
            timings[lay] = (t1 - t0) / n_iter
        return timings

    refreshP = refresh_p
    evolveQ = evolve_q
    hotStart = hot_start


def benchmark_exponentiation(
    lattice_shape: Tuple[int, ...],
    beta: float,
    batch_size: int = 1,
    layout: str = "BMXYIJ",
    n_iter: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    th_expm = SU2YangMills(
        lattice_shape=lattice_shape,
        beta=beta,
        batch_size=batch_size,
        layout=layout,
        seed=seed,
        exp_method="expm",
    )
    th_su2 = SU2YangMills(
        lattice_shape=lattice_shape,
        beta=beta,
        batch_size=batch_size,
        layout=layout,
        seed=seed,
        exp_method="su2",
    )
    q = th_expm._sample_algebra(scale=0.05)

    Ue = th_expm.algebra_to_links(q)
    Us = th_su2.algebra_to_links(q)
    Ue.block_until_ready()
    Us.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th_expm.algebra_to_links(q).block_until_ready()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        th_su2.algebra_to_links(q).block_until_ready()
    t3 = time.perf_counter()

    rel = float(jnp.linalg.norm(Us - Ue) / (jnp.linalg.norm(Ue) + 1e-12))
    eye = jnp.eye(2, dtype=Ue.dtype)
    ue_u = _dagger(Ue) @ Ue
    us_u = _dagger(Us) @ Us
    unit_expm = float(jnp.mean(jnp.linalg.norm(ue_u - eye, axis=(-2, -1))))
    unit_su2 = float(jnp.mean(jnp.linalg.norm(us_u - eye, axis=(-2, -1))))
    det_expm = float(jnp.mean(jnp.abs(jnp.linalg.det(Ue) - 1.0)))
    det_su2 = float(jnp.mean(jnp.abs(jnp.linalg.det(Us) - 1.0)))

    return {
        "expm_sec_per_call": (t1 - t0) / max(1, n_iter),
        "su2_sec_per_call": (t3 - t2) / max(1, n_iter),
        "rel_diff_su2_vs_expm": rel,
        "unitarity_expm": unit_expm,
        "unitarity_su2": unit_su2,
        "det_err_expm": det_expm,
        "det_err_su2": det_su2,
    }


def force_action_fd_test(theory: SU2YangMills, eps: float = 1e-5) -> Dict[str, float]:
    U = theory.hot_start(scale=0.05)
    H = theory.refresh_p()

    sp = theory.action(theory.algebra_to_links(+eps * H) @ U)
    sm = theory.action(theory.algebra_to_links(-eps * H) @ U)
    dS_fd = (sp - sm) / (2.0 * eps)
    dS_fd_mean = jnp.mean(dS_fd)

    F = theory.force(U)
    dS_force_minus = jnp.mean(-_algebra_inner(F, H))
    rel_fd_force_minus = jnp.abs(dS_fd_mean - dS_force_minus) / (jnp.abs(dS_fd_mean) + 1e-12)
    return {
        "rel_fd_vs_force_minus": float(rel_fd_force_minus),
        "mean_fd": float(dS_fd_mean),
        "mean_force_minus": float(dS_force_minus),
    }


def action_grad_links_autodiff(theory: SU2YangMills, U: Array) -> Array:
    """Return entrywise dS/dU via real-imag autodiff on S=sum_batch action(U)."""

    grad_links_fn = getattr(theory, "_autodiff_action_grad_links_fn", None)
    if grad_links_fn is None:
        def _action_sum(u_re: Array, u_im: Array) -> Array:
            u = (u_re + 1j * u_im).astype(theory.dtype)
            return jnp.sum(theory.action(u))

        grad_re_im = jax.grad(_action_sum, argnums=(0, 1))

        def _grad_links(u: Array) -> Array:
            g_re, g_im = grad_re_im(jnp.real(u), jnp.imag(u))
            return (g_re - 1j * g_im).astype(theory.dtype)

        grad_links_fn = _grad_links
        setattr(theory, "_autodiff_action_grad_links_fn", grad_links_fn)
    return grad_links_fn(U)


def force_from_action_autodiff(theory: SU2YangMills, U: Array) -> Array:
    """Build Lie force from autodiff matrix gradient:
    dS/dU_ij = G_ji, X = U G, F = proj_su2(X).
    """

    dS_dU = action_grad_links_autodiff(theory, U)
    G = jnp.swapaxes(dS_dU, -1, -2)
    return project_su2_algebra(U @ G)


def force_action_autodiff_test(
    theory: SU2YangMills,
    q_scale: float = 0.05,
    n_trials: int = 1,
) -> Dict[str, float]:
    rel_force = []
    abs_force = []
    rel_dir = []
    mean_dir_auto = []
    mean_dir_force = []

    for _ in range(max(1, int(n_trials))):
        U = theory.hot_start(scale=q_scale)
        F = theory.force(U)

        dS_dU = action_grad_links_autodiff(theory, U)
        F_ad = project_su2_algebra(U @ jnp.swapaxes(dS_dU, -1, -2))

        dF = F_ad - F
        rel = jnp.linalg.norm(dF) / (jnp.linalg.norm(F) + 1e-12)
        rel_force.append(float(rel))
        abs_force.append(float(jnp.max(jnp.abs(dF))))

        H = theory.refresh_p()
        dU = H @ U
        dS_auto = jnp.real(jnp.sum(dS_dU * dU, axis=tuple(range(1, dU.ndim))))
        dS_force = -_algebra_inner(F, H)
        dS_auto_mean = jnp.mean(dS_auto)
        dS_force_mean = jnp.mean(dS_force)
        rdir = jnp.abs(dS_auto_mean - dS_force_mean) / (jnp.abs(dS_auto_mean) + 1e-12)
        rel_dir.append(float(rdir))
        mean_dir_auto.append(float(dS_auto_mean))
        mean_dir_force.append(float(dS_force_mean))

    return {
        "max_rel_force_diff": float(np.max(rel_force)),
        "mean_rel_force_diff": float(np.mean(rel_force)),
        "max_abs_force_diff": float(np.max(abs_force)),
        "max_rel_directional_diff": float(np.max(rel_dir)),
        "mean_rel_directional_diff": float(np.mean(rel_dir)),
        "mean_directional_autodiff": float(np.mean(mean_dir_auto)),
        "mean_directional_force": float(np.mean(mean_dir_force)),
    }


def force_action_autodiff_timing(
    theory: SU2YangMills,
    q_scale: float = 0.05,
    n_iter: int = 5,
) -> Dict[str, float]:
    U = theory.hot_start(scale=q_scale)
    nit = max(1, int(n_iter))
    _ = action_grad_links_autodiff(theory, U)

    analytic_force = theory.force
    if not bool(getattr(theory, "_force_is_jitted", False)):
        analytic_force = jax.jit(analytic_force)
    autodiff_force = jax.jit(lambda u: force_from_action_autodiff(theory, u))

    F_ana = analytic_force(U)
    F_ana.block_until_ready()
    F_ad = autodiff_force(U)
    F_ad.block_until_ready()
    rel = float(jnp.linalg.norm(F_ad - F_ana) / (jnp.linalg.norm(F_ana) + 1e-12))

    t0 = time.perf_counter()
    for _ in range(nit):
        analytic_force(U).block_until_ready()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(nit):
        autodiff_force(U).block_until_ready()
    t3 = time.perf_counter()

    ana = (t1 - t0) / nit
    ad = (t3 - t2) / nit
    return {
        "analytic_sec_per_call": float(ana),
        "autodiff_sec_per_call": float(ad),
        "autodiff_over_analytic": float(ad / (ana + 1e-16)),
        "rel_force_diff": rel,
    }


def epsilon2_test(
    theory: SU2YangMills,
    tau: float = 1.0,
    nmd_values: Iterable[int] = (4, 6, 8, 12, 16, 24, 32),
    integrator_name: str = "minnorm2",
    n_samples: int = 8,
) -> Dict[str, object]:
    from jaxqft.core.integrators import force_gradient, leapfrog, minnorm2, minnorm4pf4

    nmd_list = [int(v) for v in nmd_values if int(v) > 0]
    nmd_list = sorted(set(nmd_list))
    eps_list = []
    dh_list = []

    for nmd in nmd_list:
        name = integrator_name.lower()
        if name == "leapfrog":
            I = leapfrog(theory.force, theory.evolve_q, nmd, tau)
        elif name == "minnorm2":
            I = minnorm2(theory.force, theory.evolve_q, nmd, tau)
        elif name == "forcegrad":
            I = force_gradient(theory.force, theory.evolve_q, nmd, tau)
        elif name == "minnorm4pf4":
            I = minnorm4pf4(theory.force, theory.evolve_q, nmd, tau)
        else:
            raise ValueError(f"Unknown integrator: {integrator_name}")
        dH_samples = []
        for _ in range(max(1, int(n_samples))):
            q0 = theory.hot_start(scale=0.05)
            p0 = theory.refresh_p()
            H0 = theory.kinetic(p0) + theory.action(q0)
            p1, q1 = I.integrate(p0, q0)
            H1 = theory.kinetic(p1) + theory.action(q1)
            dH = jnp.abs(H1 - H0)
            dH_samples.append(float(jnp.mean(dH)))

        eps = tau / float(nmd)
        eps_list.append(eps)
        dh_list.append(float(np.mean(dH_samples)))

    eps_np = np.asarray(eps_list, dtype=np.float64)
    dh_np = np.asarray(dh_list, dtype=np.float64)
    floor = max(1e-14, 1e-8 * float(np.max(dh_np)))
    good = np.isfinite(dh_np) & (dh_np > floor)
    used = int(np.sum(good))
    if used >= 2:
        x = np.log(eps_np[good])
        y = np.log(dh_np[good])
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float("nan")
    return {
        "nmd": nmd_list,
        "eps": eps_list,
        "mean_abs_dH": dh_list,
        "loglog_slope": slope,
        "fit_points_used": used,
        "fit_floor": float(floor),
    }


def epsilon4_test(
    theory: SU2YangMills,
    tau: float = 1.0,
    nmd_values: Iterable[int] = (4, 6, 8, 12, 16, 24, 32),
    integrators: Iterable[str] = ("forcegrad", "minnorm4pf4"),
    n_samples: int = 8,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for name in integrators:
        key = name.strip().lower()
        if not key:
            continue
        out[key] = epsilon2_test(theory, tau=tau, nmd_values=nmd_values, integrator_name=key, n_samples=n_samples)
    return out


def kernel_timing(theory: SU2YangMills, n_iter: int = 10) -> Dict[str, float]:
    q = theory.hot_start(scale=0.2)
    theory.action(q).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        theory.action(q).block_until_ready()
    t1 = time.perf_counter()

    theory.force(q).block_until_ready()
    t2 = time.perf_counter()
    for _ in range(n_iter):
        theory.force(q).block_until_ready()
    t3 = time.perf_counter()
    return {"action_sec_per_call": (t1 - t0) / n_iter, "force_sec_per_call": (t3 - t2) / n_iter}


def _parse_nmd_list(s: str):
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _parse_list(s: str):
    return [v.strip() for v in s.split(",") if v.strip()]


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def random_site_gauge_with_method(theory: SU2YangMills, scale: float = 0.2, method: str = "expm") -> Array:
    method = method.lower()
    if method not in ("expm", "su2"):
        raise ValueError("gauge exp method must be expm or su2")
    site_shape = (theory.Bs, *theory.lattice_shape, 2, 2)
    k1 = theory._split_key()
    re = jax.random.normal(k1, site_shape, dtype=jnp.float32)
    k2 = theory._split_key()
    im = jax.random.normal(k2, site_shape, dtype=jnp.float32)
    a = scale * (re + 1j * im).astype(theory.dtype)
    a = project_su2_algebra(a)
    if method == "su2":
        return su2_lg.expo(a)
    return jax.scipy.linalg.expm(a)


def gauge_transform_links(theory: SU2YangMills, U: Array, Omega: Array) -> Array:
    links_mu = []
    for mu in range(theory.Nd):
        U_mu = theory._take_mu(U, mu)
        Om_xpmu = jnp.roll(Omega, shift=-1, axis=1 + mu)
        U_mu_p = _dagger(Omega) @ U_mu @ Om_xpmu
        links_mu.append(U_mu_p)
    if theory.layout == "BMXYIJ":
        return jnp.stack(links_mu, axis=1)
    return jnp.stack(links_mu, axis=1 + theory.Nd)


def _group_quality_metrics(X: Array) -> Dict[str, float]:
    eye = jnp.eye(2, dtype=X.dtype)
    XX = _dagger(X) @ X
    unit = float(jnp.mean(jnp.linalg.norm(XX - eye, axis=(-2, -1))))
    det_err = float(jnp.mean(jnp.abs(jnp.linalg.det(X) - 1.0)))
    return {"unitarity": unit, "det_err": det_err}


def gauge_invariance_test(
    theory: SU2YangMills,
    q_scale: float = 0.05,
    omega_scale: float = 0.05,
    n_trials: int = 3,
    gauge_exp_method: str = "expm",
) -> Dict[str, float]:
    rel_diffs = []
    om_unit = []
    om_det = []
    u_unit = []
    u_det = []
    up_unit = []
    up_det = []
    for _ in range(max(1, n_trials)):
        U = theory.hot_start(scale=q_scale)
        S0 = theory.action(U)
        Om = random_site_gauge_with_method(theory, scale=omega_scale, method=gauge_exp_method)
        Up = gauge_transform_links(theory, U, Om)
        S1 = theory.action(Up)
        rel = jnp.linalg.norm(S1 - S0) / (jnp.linalg.norm(S0) + 1e-12)
        rel_diffs.append(float(rel))
        m_om = _group_quality_metrics(Om)
        m_u = _group_quality_metrics(U)
        m_up = _group_quality_metrics(Up)
        om_unit.append(m_om["unitarity"])
        om_det.append(m_om["det_err"])
        u_unit.append(m_u["unitarity"])
        u_det.append(m_u["det_err"])
        up_unit.append(m_up["unitarity"])
        up_det.append(m_up["det_err"])
    return {
        "max_rel_action_diff": float(np.max(rel_diffs)),
        "mean_rel_action_diff": float(np.mean(rel_diffs)),
        "omega_unitarity": float(np.mean(om_unit)),
        "omega_det_err": float(np.mean(om_det)),
        "U_unitarity": float(np.mean(u_unit)),
        "U_det_err": float(np.mean(u_det)),
        "Up_unitarity": float(np.mean(up_unit)),
        "Up_det_err": float(np.mean(up_det)),
    }


def force_gauge_covariance_test(
    theory: SU2YangMills,
    q_scale: float = 0.05,
    omega_scale: float = 0.05,
    n_trials: int = 3,
    gauge_exp_method: str = "expm",
) -> Dict[str, float]:
    rels = []
    max_abs = []
    for _ in range(max(1, n_trials)):
        U = theory.hot_start(scale=q_scale)
        F = theory.force(U)
        Om = random_site_gauge_with_method(theory, scale=omega_scale, method=gauge_exp_method)
        Up = gauge_transform_links(theory, U, Om)
        Fp = theory.force(Up)

        F_cov_mu = []
        for mu in range(theory.Nd):
            F_mu = theory._take_mu(F, mu)
            F_cov_mu.append(_dagger(Om) @ F_mu @ Om)
        if theory.layout == "BMXYIJ":
            F_cov = jnp.stack(F_cov_mu, axis=1)
        else:
            F_cov = jnp.stack(F_cov_mu, axis=1 + theory.Nd)

        d = Fp - F_cov
        rel = jnp.linalg.norm(d) / (jnp.linalg.norm(F_cov) + 1e-12)
        rels.append(float(rel))
        max_abs.append(float(jnp.max(jnp.abs(d))))
    return {
        "max_rel_force_cov_diff": float(np.max(rels)),
        "mean_rel_force_cov_diff": float(np.mean(rels)),
        "max_abs_force_cov_diff": float(np.max(max_abs)),
    }


def force_impl_consistency_test(
    theory: SU2YangMills,
    q_scale: float = 0.05,
    n_trials: int = 3,
) -> Dict[str, float]:
    if not theory._use_optimized_force:
        return {
            "enabled": 0.0,
            "max_rel_force_diff": float("nan"),
            "mean_rel_force_diff": float("nan"),
            "max_abs_force_diff": float("nan"),
        }

    rels = []
    max_abs = []
    for _ in range(max(1, n_trials)):
        U = theory.hot_start(scale=q_scale)
        F_ref = theory.force_reference(U)
        F_opt = theory.force(U)
        d = F_opt - F_ref
        rel = jnp.linalg.norm(d) / (jnp.linalg.norm(F_ref) + 1e-12)
        rels.append(float(rel))
        max_abs.append(float(jnp.max(jnp.abs(d))))
    return {
        "enabled": 1.0,
        "max_rel_force_diff": float(np.max(rels)),
        "mean_rel_force_diff": float(np.mean(rels)),
        "max_abs_force_diff": float(np.max(max_abs)),
    }


def full_selfcheck(
    theory: SU2YangMills,
    lattice_shape: Tuple[int, ...],
    beta: float,
    batch_size: int,
    layout: str,
    seed: int,
    n_iter_timing: int,
    exp_rel_tol: float,
    gauge_trials: int,
    gauge_omega_scale: float,
    gauge_exp_method: str,
    fd_eps: float,
    autodiff_trials: int,
) -> Dict[str, object]:
    checks = []

    alg = float(su2_lg.check_algebra())
    checks.append(("lie_algebra", alg < 1e-5, f"rel={alg:.3e}"))

    exb = benchmark_exponentiation(
        lattice_shape=lattice_shape,
        beta=beta,
        batch_size=batch_size,
        layout=layout,
        n_iter=max(1, n_iter_timing),
        seed=seed,
    )
    exb_rel = float(exb["rel_diff_su2_vs_expm"])
    checks.append(("expo", exb_rel <= exp_rel_tol, f"rel_diff={exb_rel:.3e}, tol={exp_rel_tol:.1e}"))

    g = gauge_invariance_test(
        theory,
        q_scale=0.05,
        omega_scale=gauge_omega_scale,
        n_trials=max(1, gauge_trials),
        gauge_exp_method=gauge_exp_method,
    )
    g_rel = float(g["max_rel_action_diff"])
    checks.append(("gauge_invariance", g_rel < 1e-5, f"max_rel_action_diff={g_rel:.3e}"))

    fg = force_gauge_covariance_test(
        theory,
        q_scale=0.05,
        omega_scale=gauge_omega_scale,
        n_trials=max(1, gauge_trials),
        gauge_exp_method=gauge_exp_method,
    )
    fg_rel = float(fg["max_rel_force_cov_diff"])
    checks.append(("force_covariance", fg_rel < 1e-5, f"max_rel_force_cov_diff={fg_rel:.3e}"))

    fim = force_impl_consistency_test(
        theory,
        q_scale=0.05,
        n_trials=max(1, gauge_trials),
    )
    if bool(fim.get("enabled", 0.0)):
        fim_rel = float(fim["max_rel_force_diff"])
        checks.append(("force_impl_consistency", fim_rel < 1e-5, f"max_rel_force_diff={fim_rel:.3e}"))

    fd = force_action_fd_test(theory, eps=fd_eps)
    fd_rel = float(fd["rel_fd_vs_force_minus"])
    checks.append(("fd_force_consistency", fd_rel < 2e-2, f"rel_fd_vs_force={fd_rel:.3e}"))

    ad = force_action_autodiff_test(theory, q_scale=0.05, n_trials=max(1, autodiff_trials))
    ad_rel = float(ad["max_rel_force_diff"])
    ad_dir = float(ad["max_rel_directional_diff"])
    checks.append(
        (
            "autodiff_force_consistency",
            (ad_rel < 2e-2) and (ad_dir < 2e-2),
            f"max_rel_force={ad_rel:.3e}, max_rel_dir={ad_dir:.3e}",
        )
    )

    eps2 = epsilon2_test(theory, tau=1.0, nmd_values=(8, 16, 32), integrator_name="minnorm2")
    slope = float(eps2["loglog_slope"])
    checks.append(("eps2_scaling", (slope > 1.5) and (slope < 2.5), f"slope={slope:.3f}"))

    all_ok = all(ok for _, ok, _ in checks)
    return {"all_ok": all_ok, "checks": checks}


def main():
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(
        description="SU2 Yang-Mills diagnostics (force/action/epsilon2/layout timing)",
        allow_abbrev=False,
    )
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nd", type=int, default=4)
    ap.add_argument("--shape", type=str, default="", help="comma-separated lattice shape, e.g. 4,4,4,4")
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ", choices=["BMXYIJ", "BXYMIJ", "BM...IJ", "B...MIJ", "auto"])
    ap.add_argument("--exp-method", type=str, default="auto", choices=["expm", "su2", "auto"])
    ap.add_argument("--exp-rel-tol", type=float, default=1e-5, help="max relative diff for auto choosing su2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter-timing", type=int, default=5)
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    ap.add_argument("--gauge-trials", type=int, default=3)
    ap.add_argument("--gauge-omega-scale", type=float, default=0.05)
    ap.add_argument("--gauge-exp-method", type=str, default="expm", choices=["expm", "su2", "auto"])
    ap.add_argument("--autodiff-trials", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--nmd-list", type=str, default="8,16,32,64,128")
    ap.add_argument("--eps-samples", type=int, default=8, help="number of random trajectories per nmd point")
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["minnorm2", "leapfrog", "forcegrad", "minnorm4pf4"])
    ap.add_argument(
        "--eps4-integrators",
        type=str,
        default="forcegrad,minnorm4pf4",
        help="comma-separated 4th-order integrators to test in eps4",
    )
    ap.add_argument(
        "--tests",
        type=str,
        default="all",
        help="comma-separated subset of: layout,expo,timing,fd,autodiff,eps2,eps4,gauge,forcecov,forceimpl,selfcheck,all",
    )
    ap.add_argument("--selfcheck", action="store_true", help="run only the selfcheck suite")
    ap.add_argument("--selfcheck-fail", action="store_true", help="return nonzero exit if any selfcheck fails")
    args = ap.parse_args()

    lattice_shape = _parse_shape(args.shape) if args.shape.strip() else tuple([args.L] * args.Nd)
    if args.selfcheck:
        tests = {"selfcheck"}
    else:
        tests = {t.strip().lower() for t in args.tests.split(",") if t.strip()}
        if "all" in tests:
            tests = {"layout", "expo", "timing", "fd", "autodiff", "eps2", "eps4", "gauge", "forcecov", "forceimpl"}

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    selected_layout = args.layout
    if "layout" in tests or args.layout == "auto":
        ta = SU2YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            n_iter=args.n_iter_timing,
            seed=args.seed,
            kernel="action",
            exp_method="expm" if args.exp_method == "auto" else args.exp_method,
        )
        tf = SU2YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            n_iter=args.n_iter_timing,
            seed=args.seed,
            kernel="force",
            exp_method="expm" if args.exp_method == "auto" else args.exp_method,
        )
        print("Layout benchmark (sec/call):")
        for lay in sorted(ta):
            print(f"  {lay}: action={ta[lay]:.6e}, force={tf[lay]:.6e}")
        if args.layout == "auto":
            selected_layout = min(tf, key=tf.get)
            print("Auto-selected layout:", selected_layout)

    selected_exp_method = args.exp_method
    if "expo" in tests or args.exp_method == "auto":
        exb = benchmark_exponentiation(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            layout=selected_layout if selected_layout != "auto" else "BMXYIJ",
            n_iter=args.n_iter_timing,
            seed=args.seed,
        )
        print("Exponentiation benchmark:")
        print(f"  expm sec/call: {exb['expm_sec_per_call']:.6e}")
        print(f"  su2  sec/call: {exb['su2_sec_per_call']:.6e}")
        print(f"  rel_diff(su2,expm): {exb['rel_diff_su2_vs_expm']:.6e}")
        print(f"  unitarity expm/su2: {exb['unitarity_expm']:.6e} / {exb['unitarity_su2']:.6e}")
        print(f"  det err  expm/su2: {exb['det_err_expm']:.6e} / {exb['det_err_su2']:.6e}")
        if args.exp_method == "auto":
            if exb["rel_diff_su2_vs_expm"] <= args.exp_rel_tol and exb["su2_sec_per_call"] < exb["expm_sec_per_call"]:
                selected_exp_method = "su2"
            else:
                selected_exp_method = "expm"
            print("Auto-selected exp_method:", selected_exp_method)

    theory = SU2YangMills(
        lattice_shape=lattice_shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=selected_layout,
        seed=args.seed,
        exp_method=selected_exp_method if selected_exp_method != "auto" else "expm",
    )
    print("Theory config:")
    print(f"  lattice_shape: {lattice_shape}")
    print(f"  beta: {args.beta}")
    print(f"  batch: {args.batch}")
    print(f"  layout: {selected_layout}")
    print(f"  exp_method: {theory.exp_method}")
    print(f"  field_shape: {theory.field_shape()}")
    print(f"  mu_axis: {theory.mu_axis}")
    print(f"  lattice_axes: {theory.lat_axes}")

    if "timing" in tests:
        kt = kernel_timing(theory, n_iter=max(1, args.n_iter_timing))
        print("Kernel timing:")
        print(f"  action sec/call: {kt['action_sec_per_call']:.6e}")
        print(f"  force  sec/call: {kt['force_sec_per_call']:.6e}")

    if "fd" in tests:
        if args.fd_eps < 1e-5 and theory.dtype == jnp.complex64:
            print("WARNING: very small --fd-eps with complex64 can be dominated by roundoff.")
        fd = force_action_fd_test(theory, eps=args.fd_eps)
        print("Force/action finite-difference test:")
        print(f"  rel(fd, -<F,H>): {fd['rel_fd_vs_force_minus']:.6e}")
        print(f"  mean fd / -<F,H>: {fd['mean_fd']:.6e} / {fd['mean_force_minus']:.6e}")

    if "autodiff" in tests:
        ad = force_action_autodiff_test(
            theory,
            q_scale=0.05,
            n_trials=max(1, args.autodiff_trials),
        )
        print("Force/action autodiff Lie-derivative test:")
        print(f"  max rel force diff (autodiff vs analytic): {ad['max_rel_force_diff']:.6e}")
        print(f"  mean rel force diff (autodiff vs analytic): {ad['mean_rel_force_diff']:.6e}")
        print(f"  max abs force entry diff: {ad['max_abs_force_diff']:.6e}")
        print(f"  max rel directional diff dS(auto) vs -<F,H>: {ad['max_rel_directional_diff']:.6e}")
        print(
            "  mean directional dS(auto) / -<F,H>:"
            f" {ad['mean_directional_autodiff']:.6e} / {ad['mean_directional_force']:.6e}"
        )
        adt = force_action_autodiff_timing(
            theory,
            q_scale=0.05,
            n_iter=max(1, args.n_iter_timing),
        )
        print("Autodiff timing (post-JIT steady-state):")
        print(
            "  analytic/autodiff sec per call:"
            f" {adt['analytic_sec_per_call']:.6e} / {adt['autodiff_sec_per_call']:.6e}"
        )
        print(f"  autodiff / analytic time ratio: {adt['autodiff_over_analytic']:.3f}x")
        print(f"  rel force diff (timed kernels): {adt['rel_force_diff']:.6e}")

    if "eps2" in tests:
        nmd_values = _parse_nmd_list(args.nmd_list)
        eps = epsilon2_test(
            theory,
            tau=args.tau,
            nmd_values=nmd_values,
            integrator_name=args.integrator,
            n_samples=args.eps_samples,
        )
        print("epsilon^2 test:")
        print("  nmd:", eps["nmd"])
        print("  eps:", [f"{x:.3e}" for x in eps["eps"]])
        print("  mean |dH|:", [f"{x:.3e}" for x in eps["mean_abs_dH"]])
        print(f"  log-log slope (expected ~2 for leapfrog): {eps['loglog_slope']:.4f}")
        print(f"  fit points used: {eps['fit_points_used']} (floor={eps['fit_floor']:.3e})")

    if "eps4" in tests:
        nmd_values = _parse_nmd_list(args.nmd_list)
        integrators = _parse_list(args.eps4_integrators)
        eps4 = epsilon4_test(
            theory,
            tau=args.tau,
            nmd_values=nmd_values,
            integrators=integrators,
            n_samples=args.eps_samples,
        )
        print("epsilon^4 test:")
        for name in integrators:
            key = name.strip().lower()
            if key not in eps4:
                continue
            r = eps4[key]
            print(f"  integrator: {key}")
            print("    nmd:", r["nmd"])
            print("    eps:", [f"{x:.3e}" for x in r["eps"]])
            print("    mean |dH|:", [f"{x:.3e}" for x in r["mean_abs_dH"]])
            print(f"    log-log slope (expected ~4): {r['loglog_slope']:.4f}")
            print(f"    fit points used: {r['fit_points_used']} (floor={r['fit_floor']:.3e})")

    if "gauge" in tests:
        gauge_exp_method = args.gauge_exp_method
        if gauge_exp_method == "auto":
            gauge_exp_method = selected_exp_method if selected_exp_method != "auto" else "expm"
        g = gauge_invariance_test(
            theory,
            q_scale=0.05,
            omega_scale=args.gauge_omega_scale,
            n_trials=args.gauge_trials,
            gauge_exp_method=gauge_exp_method,
        )
        print("Gauge invariance test:")
        print(f"  max rel action diff under gauge transform: {g['max_rel_action_diff']:.6e}")
        print(f"  mean rel action diff under gauge transform: {g['mean_rel_action_diff']:.6e}")
        print(
            "  quality Omega/U/Up (unitarity):"
            f" {g['omega_unitarity']:.6e} / {g['U_unitarity']:.6e} / {g['Up_unitarity']:.6e}"
        )
        print(
            "  quality Omega/U/Up (det_err):"
            f" {g['omega_det_err']:.6e} / {g['U_det_err']:.6e} / {g['Up_det_err']:.6e}"
        )

    if "forcecov" in tests:
        gauge_exp_method = args.gauge_exp_method
        if gauge_exp_method == "auto":
            gauge_exp_method = selected_exp_method if selected_exp_method != "auto" else "expm"
        fg = force_gauge_covariance_test(
            theory,
            q_scale=0.05,
            omega_scale=args.gauge_omega_scale,
            n_trials=args.gauge_trials,
            gauge_exp_method=gauge_exp_method,
        )
        print("Force gauge-covariance test:")
        print(f"  max rel diff F(Up) vs Omega^dag F(U) Omega: {fg['max_rel_force_cov_diff']:.6e}")
        print(f"  mean rel diff F(Up) vs Omega^dag F(U) Omega: {fg['mean_rel_force_cov_diff']:.6e}")
        print(f"  max abs entrywise diff: {fg['max_abs_force_cov_diff']:.6e}")

    if "forceimpl" in tests:
        fim = force_impl_consistency_test(
            theory,
            q_scale=0.05,
            n_trials=max(1, args.gauge_trials),
        )
        if not bool(fim.get("enabled", 0.0)):
            print("Force implementation consistency test: skipped (optimized force path disabled)")
        else:
            print("Force implementation consistency test (optimized vs reference):")
            print(f"  max rel force diff: {fim['max_rel_force_diff']:.6e}")
            print(f"  mean rel force diff: {fim['mean_rel_force_diff']:.6e}")
            print(f"  max abs force diff: {fim['max_abs_force_diff']:.6e}")

    if "selfcheck" in tests:
        gauge_exp_method = args.gauge_exp_method
        if gauge_exp_method == "auto":
            gauge_exp_method = selected_exp_method if selected_exp_method != "auto" else "expm"
        report = full_selfcheck(
            theory=theory,
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            layout=selected_layout,
            seed=args.seed,
            n_iter_timing=args.n_iter_timing,
            exp_rel_tol=args.exp_rel_tol,
            gauge_trials=args.gauge_trials,
            gauge_omega_scale=args.gauge_omega_scale,
            gauge_exp_method=gauge_exp_method,
            fd_eps=args.fd_eps,
            autodiff_trials=args.autodiff_trials,
        )
        print("SU2 full selfcheck:")
        for name, ok, msg in report["checks"]:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")
        print("  overall:", "PASS" if report["all_ok"] else "FAIL")
        if args.selfcheck_fail and not report["all_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

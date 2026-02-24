"""Pure compact U(1) Yang-Mills theory (scalar links) with diagnostics."""

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


Array = jax.Array


def project_u1_algebra(x: Array) -> Array:
    # Anti-Hermitian scalar Lie algebra element: i * real.
    return 0.5 * (x - jnp.conj(x))


def _algebra_inner(a: Array, b: Array) -> Array:
    trab = jnp.real(a * b)
    return -jnp.sum(trab, axis=tuple(range(1, trab.ndim)))


def _plaquette_sum_bmxyij_nd2_optimized(U: Array) -> Array:
    """Return sum_x Re U_{0,1}(x) for BMXYIJ/2D."""
    U0 = U[:, 0]
    U1 = U[:, 1]
    U1_xp0 = jnp.roll(U1, -1, axis=1)
    U0_xp1 = jnp.roll(U0, -1, axis=2)
    plaq = U0 * U1_xp0 * jnp.conj(U0_xp1) * jnp.conj(U1)
    trp = jnp.real(plaq)
    return jnp.sum(trp, axis=tuple(range(1, trp.ndim)))


def _plaquette_sum_bmxyij_nd4_optimized(U: Array) -> Array:
    """Return sum_{x,mu<nu} Re U_{mu,nu}(x) for BMXYIJ/4D."""
    B = U.shape[0]
    plaq_sum = jnp.zeros((B,), dtype=jnp.float32)
    U_plus = [jnp.roll(U, -1, axis=2 + d) for d in range(4)]
    for mu in range(4):
        U_mu = U[:, mu]
        for nu in range(mu + 1, 4):
            U_nu = U[:, nu]
            U_nu_xpmu = U_plus[mu][:, nu]
            U_mu_xpnu = U_plus[nu][:, mu]
            plaq = U_mu * U_nu_xpmu * jnp.conj(U_mu_xpnu) * jnp.conj(U_nu)
            trp = jnp.real(plaq)
            plaq_sum = plaq_sum + jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
    return plaq_sum


def _action_bmxyij_nd4_optimized(U: Array, beta: float) -> Array:
    return -beta * _plaquette_sum_bmxyij_nd4_optimized(U)


def _action_bmxyij_nd2_optimized(U: Array, beta: float) -> Array:
    return -beta * _plaquette_sum_bmxyij_nd2_optimized(U)


def _force_bmxyij_nd2_optimized(U: Array, beta: float) -> Array:
    coef = -beta
    U_plus = [jnp.roll(U, -1, axis=2 + d) for d in range(2)]
    U_minus = [jnp.roll(U, +1, axis=2 + d) for d in range(2)]
    U_minus_nu = [U_minus[nu][:, nu] for nu in range(2)]
    out = []
    for mu in range(2):
        nu = 1 - mu
        U_mu = U[:, mu]
        U_nu = U[:, nu]
        U_mu_xpnu = U_plus[nu][:, mu]
        U_nu_xpmu = U_plus[mu][:, nu]
        fwd = U_nu * U_mu_xpnu * jnp.conj(U_nu_xpmu)

        U_nu_m = U_minus_nu[nu]
        U_mu_m = U_minus[nu][:, mu]
        U_nu_m_xpmu = jnp.roll(U_nu_m, -1, axis=1 + mu)
        bwd = jnp.conj(U_nu_m) * U_mu_m * U_nu_m_xpmu
        staple = fwd + bwd
        out.append(coef * project_u1_algebra(U_mu * jnp.conj(staple)))
    return jnp.stack(out, axis=1)


def _force_bmxyij_nd4_optimized(U: Array, beta: float) -> Array:
    coef = -beta
    U_plus = [jnp.roll(U, -1, axis=2 + d) for d in range(4)]
    U_minus = [jnp.roll(U, +1, axis=2 + d) for d in range(4)]
    U_minus_nu = [U_minus[nu][:, nu] for nu in range(4)]
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
            fwd = U_nu * U_mu_xpnu * jnp.conj(U_nu_xpmu)

            U_nu_m = U_minus_nu[nu]
            U_mu_m = U_minus[nu][:, mu]
            U_nu_m_xpmu = jnp.roll(U_nu_m, -1, axis=1 + mu)
            bwd = jnp.conj(U_nu_m) * U_mu_m * U_nu_m_xpmu
            staple = staple + fwd + bwd
        out.append(coef * project_u1_algebra(U_mu * jnp.conj(staple)))
    return jnp.stack(out, axis=1)


def _evolve_q_u1_fused(dt: float, P: Array, Q: Array) -> Array:
    # P is anti-Hermitian scalar i*phi (phi real). This avoids generic complex exp.
    phi = jnp.imag(P)
    a = dt * phi
    return (jnp.cos(a) + 1j * jnp.sin(a)) * Q


@dataclass
class U1YangMills:
    lattice_shape: Tuple[int, ...]
    beta: float
    batch_size: int = 1
    layout: str = "BMXYIJ"
    seed: int = 0
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
        self.key = jax.random.PRNGKey(self.seed)
        if self.layout == "BMXYIJ":
            self.mu_axis = 1
            self.lat_axes = tuple(range(2, 2 + self.Nd))
        else:
            self.mu_axis = 1 + self.Nd
            self.lat_axes = tuple(range(1, 1 + self.Nd))
        # For U(1), 2D specialized action/force are consistently faster.
        # 4D specialized kernels can be slower than reference on CPU due to extra
        # memory traffic from pre-rolled link fields; keep 4D opt-in via env.
        use_nd4_specialized = os.environ.get("JAXQFT_U1_ENABLE_ND4_SPECIALIZED", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        opt_nd_set = {2}
        if use_nd4_specialized:
            opt_nd_set.add(4)

        self._use_optimized_action = self.layout == "BMXYIJ" and self.Nd in opt_nd_set
        self._action_is_jitted = False
        if self._use_optimized_action:
            beta = float(self.beta)
            if self.Nd == 2:
                self._action_opt = jax.jit(lambda U: _action_bmxyij_nd2_optimized(U, beta))
            else:
                self._action_opt = jax.jit(lambda U: _action_bmxyij_nd4_optimized(U, beta))
            self._action_is_jitted = True
        self._use_optimized_force = self.layout == "BMXYIJ" and self.Nd in opt_nd_set
        self._force_is_jitted = False
        if self._use_optimized_force:
            beta = float(self.beta)
            if self.Nd == 2:
                self._force_opt = jax.jit(lambda U: _force_bmxyij_nd2_optimized(U, beta))
            else:
                self._force_opt = jax.jit(lambda U: _force_bmxyij_nd4_optimized(U, beta))
            self._force_is_jitted = True
        self._use_optimized_evolve_q = True
        self._evolve_q_is_jitted = False
        if self._use_optimized_evolve_q:
            self._evolve_q_opt = jax.jit(_evolve_q_u1_fused)
            self._evolve_q_is_jitted = True

    def _split_key(self):
        self.key, sub = jax.random.split(self.key)
        return sub

    def field_shape(self):
        if self.layout == "BMXYIJ":
            return (self.Bs, self.Nd, *self.lattice_shape)
        return (self.Bs, *self.lattice_shape, self.Nd)

    def _take_mu(self, x: Array, mu: int) -> Array:
        return jnp.take(x, mu, axis=self.mu_axis)

    def _roll(self, x: Array, shift: int, direction: int) -> Array:
        if x.ndim == self.Nd + 1:
            ax = 1 + direction
        elif x.ndim == self.Nd + 2:
            ax = self.lat_axes[direction]
        else:
            raise ValueError(f"Unsupported tensor rank for lattice roll: ndim={x.ndim}")
        return jnp.roll(x, shift=shift, axis=ax)

    def _sample_algebra(self, scale: float = 1.0) -> Array:
        k = self._split_key()
        im = jax.random.normal(k, self.field_shape(), dtype=jnp.float32)
        # Already anti-Hermitian (i * real), no projection needed.
        return (1j * scale * im).astype(self.dtype)

    def algebra_to_links(self, q: Array) -> Array:
        q = project_u1_algebra(q)
        return jnp.exp(q).astype(self.dtype)

    def action_reference(self, U: Array) -> Array:
        B = U.shape[0]
        S = jnp.zeros((B,), dtype=jnp.float32)
        for mu in range(self.Nd):
            U_mu = self._take_mu(U, mu)
            for nu in range(mu + 1, self.Nd):
                U_nu = self._take_mu(U, nu)
                U_nu_xpmu = self._roll(U_nu, -1, mu)
                U_mu_xpnu = self._roll(U_mu, -1, nu)
                plaq = U_mu * U_nu_xpmu * jnp.conj(U_mu_xpnu) * jnp.conj(U_nu)
                trp = jnp.real(plaq)
                S = S - self.beta * jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
        return S

    def action(self, U: Array) -> Array:
        if self._use_optimized_action:
            return self._action_opt(U)
        return self.action_reference(U)

    def action_optimized_unjitted(self, U: Array) -> Array:
        if self.layout != "BMXYIJ" or self.Nd not in (2, 4):
            return self.action_reference(U)
        if self.Nd == 2:
            return _action_bmxyij_nd2_optimized(U, self.beta)
        return _action_bmxyij_nd4_optimized(U, self.beta)

    def _staple(self, U: Array, mu: int) -> Array:
        U_mu = self._take_mu(U, mu)
        staple = jnp.zeros_like(U_mu)
        for nu in range(self.Nd):
            if nu == mu:
                continue
            U_nu = self._take_mu(U, nu)
            fwd = U_nu * self._roll(U_mu, -1, nu) * jnp.conj(self._roll(U_nu, -1, mu))
            U_nu_m = self._roll(U_nu, +1, nu)
            bwd = jnp.conj(U_nu_m) * self._roll(U_mu, +1, nu) * self._roll(U_nu_m, -1, mu)
            staple = staple + fwd + bwd
        return staple

    def force_reference(self, U: Array) -> Array:
        links_force = []
        coef = -self.beta
        for mu in range(self.Nd):
            U_mu = self._take_mu(U, mu)
            staple_mu = self._staple(U, mu)
            G_mu = coef * project_u1_algebra(U_mu * jnp.conj(staple_mu))
            links_force.append(G_mu)
        if self.layout == "BMXYIJ":
            return jnp.stack(links_force, axis=1)
        return jnp.stack(links_force, axis=1 + self.Nd)

    def force(self, U: Array) -> Array:
        if self._use_optimized_force:
            return self._force_opt(U)
        return self.force_reference(U)

    def force_optimized_unjitted(self, U: Array) -> Array:
        if self.layout != "BMXYIJ" or self.Nd not in (2, 4):
            return self.force_reference(U)
        if self.Nd == 2:
            return _force_bmxyij_nd2_optimized(U, self.beta)
        return _force_bmxyij_nd4_optimized(U, self.beta)

    def refresh_p(self) -> Array:
        return self._sample_algebra(scale=1.0)

    def refresh_p_with_key(self, key: Array) -> Array:
        im = jax.random.normal(key, self.field_shape(), dtype=jnp.float32)
        # Already anti-Hermitian (i * real), no projection needed.
        return (1j * im).astype(self.dtype)

    def evolve_q_reference(self, dt: float, P: Array, Q: Array) -> Array:
        return jnp.exp(dt * P) * Q

    def evolve_q(self, dt: float, P: Array, Q: Array) -> Array:
        if self._use_optimized_evolve_q:
            return self._evolve_q_opt(dt, P, Q)
        return self.evolve_q_reference(dt, P, Q)

    def evolve_q_optimized_unjitted(self, dt: float, P: Array, Q: Array) -> Array:
        if not self._use_optimized_evolve_q:
            return self.evolve_q_reference(dt, P, Q)
        return _evolve_q_u1_fused(dt, P, Q)

    def kinetic(self, P: Array) -> Array:
        trp2 = jnp.real(P * P)
        return -0.5 * jnp.sum(trp2, axis=tuple(range(1, trp2.ndim)))

    def hot_start(self, scale: float = 0.1) -> Array:
        return self.algebra_to_links(self._sample_algebra(scale=scale))

    def average_plaquette(self, U: Array) -> Array:
        vol = int(math.prod(self.lattice_shape))
        nplanes = self.Nd * (self.Nd - 1) // 2
        if self._use_optimized_action:
            if self.Nd == 2:
                plaq_sum = _plaquette_sum_bmxyij_nd2_optimized(U)
            else:
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
                    plaq = U_mu * U_nu_xpmu * jnp.conj(U_mu_xpnu) * jnp.conj(U_nu)
                    trp = jnp.real(plaq)
                    plaq_sum = plaq_sum + jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
        return plaq_sum / (vol * nplanes)

    @staticmethod
    def benchmark_layout(
        lattice_shape: Tuple[int, ...],
        beta: float,
        batch_size: int = 1,
        layouts: Iterable[str] = ("BMXYIJ", "BXYMIJ"),
        n_iter: int = 3,
        seed: int = 0,
        kernel: str = "action",
    ) -> Dict[str, float]:
        kernel = kernel.lower()
        if kernel not in ("action", "force"):
            raise ValueError("kernel must be 'action' or 'force'")
        timings = {}
        for lay in layouts:
            th = U1YangMills(
                lattice_shape=lattice_shape,
                beta=beta,
                batch_size=batch_size,
                layout=lay,
                seed=seed,
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


def force_action_fd_test(theory: U1YangMills, eps: float = 1e-5) -> Dict[str, float]:
    U = theory.hot_start(scale=0.05)
    H = theory.refresh_p()

    sp = theory.action(theory.algebra_to_links(+eps * H) * U)
    sm = theory.action(theory.algebra_to_links(-eps * H) * U)
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


def epsilon2_test(
    theory: U1YangMills,
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
    theory: U1YangMills,
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


def kernel_timing(theory: U1YangMills, n_iter: int = 10) -> Dict[str, float]:
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


def random_site_gauge(theory: U1YangMills, scale: float = 0.2) -> Array:
    site_shape = (theory.Bs, *theory.lattice_shape)
    k = theory._split_key()
    theta = scale * jax.random.normal(k, site_shape, dtype=jnp.float32)
    return jnp.exp(1j * theta).astype(theory.dtype)


def gauge_transform_links(theory: U1YangMills, U: Array, Omega: Array) -> Array:
    links_mu = []
    for mu in range(theory.Nd):
        U_mu = theory._take_mu(U, mu)
        Om_xpmu = jnp.roll(Omega, shift=-1, axis=1 + mu)
        U_mu_p = jnp.conj(Omega) * U_mu * Om_xpmu
        links_mu.append(U_mu_p)
    if theory.layout == "BMXYIJ":
        return jnp.stack(links_mu, axis=1)
    return jnp.stack(links_mu, axis=1 + theory.Nd)


def _group_quality_metrics(X: Array) -> Dict[str, float]:
    unit = float(jnp.mean(jnp.abs(jnp.abs(X) ** 2 - 1.0)))
    det_err = float(jnp.mean(jnp.abs(jnp.abs(X) - 1.0)))
    return {"unitarity": unit, "det_err": det_err}


def gauge_invariance_test(
    theory: U1YangMills,
    q_scale: float = 0.05,
    omega_scale: float = 0.05,
    n_trials: int = 3,
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
        Om = random_site_gauge(theory, scale=omega_scale)
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
    theory: U1YangMills,
    q_scale: float = 0.05,
    omega_scale: float = 0.05,
    n_trials: int = 3,
) -> Dict[str, float]:
    rels = []
    max_abs = []
    for _ in range(max(1, n_trials)):
        U = theory.hot_start(scale=q_scale)
        F = theory.force(U)
        Om = random_site_gauge(theory, scale=omega_scale)
        Up = gauge_transform_links(theory, U, Om)
        Fp = theory.force(Up)

        # U(1) adjoint action is scalar conjugation, which is identity for commuting scalars.
        F_cov_mu = []
        for mu in range(theory.Nd):
            F_mu = theory._take_mu(F, mu)
            F_cov_mu.append(jnp.conj(Om) * F_mu * Om)
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
    theory: U1YangMills,
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
    theory: U1YangMills,
    gauge_trials: int,
    gauge_omega_scale: float,
    fd_eps: float,
) -> Dict[str, object]:
    checks = []

    g = gauge_invariance_test(theory, q_scale=0.05, omega_scale=gauge_omega_scale, n_trials=max(1, gauge_trials))
    g_rel = float(g["max_rel_action_diff"])
    checks.append(("gauge_invariance", g_rel < 1e-6, f"max_rel_action_diff={g_rel:.3e}"))

    fg = force_gauge_covariance_test(theory, q_scale=0.05, omega_scale=gauge_omega_scale, n_trials=max(1, gauge_trials))
    fg_rel = float(fg["max_rel_force_cov_diff"])
    checks.append(("force_covariance", fg_rel < 1e-6, f"max_rel_force_cov_diff={fg_rel:.3e}"))

    fim = force_impl_consistency_test(
        theory,
        q_scale=0.05,
        n_trials=max(1, gauge_trials),
    )
    if bool(fim.get("enabled", 0.0)):
        fim_rel = float(fim["max_rel_force_diff"])
        checks.append(("force_impl_consistency", fim_rel < 1e-6, f"max_rel_force_diff={fim_rel:.3e}"))

    fd = force_action_fd_test(theory, eps=fd_eps)
    fd_rel = float(fd["rel_fd_vs_force_minus"])
    checks.append(("fd_force_consistency", fd_rel < 5e-3, f"rel_fd_vs_force={fd_rel:.3e}"))

    eps2 = epsilon2_test(theory, tau=1.0, nmd_values=(8, 16, 32), integrator_name="minnorm2", n_samples=16)
    slope = float(eps2["loglog_slope"])
    checks.append(("eps2_scaling", (slope > 1.5) and (slope < 2.5), f"slope={slope:.3f}"))

    for name in ("forcegrad", "minnorm4pf4"):
        eps4 = epsilon2_test(
            theory,
            tau=1.0,
            nmd_values=(4, 8, 16),
            integrator_name=name,
            n_samples=16,
        )
        slope4 = float(eps4["loglog_slope"])
        checks.append((f"eps4_scaling_{name}", (slope4 > 2.5) and (slope4 < 5.5), f"slope={slope4:.3f}"))

    all_ok = all(ok for _, ok, _ in checks)
    return {"all_ok": all_ok, "checks": checks}


def main():
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(
        description="U1 Yang-Mills diagnostics (force/action/epsilon/layout timing)",
        allow_abbrev=False,
    )
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nd", type=int, default=4)
    ap.add_argument("--shape", type=str, default="", help="comma-separated lattice shape, e.g. 4,4,4,4")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ", choices=["BMXYIJ", "BXYMIJ", "BM...IJ", "B...MIJ", "auto"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter-timing", type=int, default=5)
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    ap.add_argument("--gauge-trials", type=int, default=3)
    ap.add_argument("--gauge-omega-scale", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--nmd-list", type=str, default="8,16,32,64,128")
    ap.add_argument("--eps-samples", type=int, default=8, help="number of random trajectories per nmd point")
    ap.add_argument(
        "--integrator",
        type=str,
        default="minnorm2",
        choices=["minnorm2", "leapfrog", "forcegrad", "minnorm4pf4"],
    )
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
        help="comma-separated subset of: layout,timing,fd,eps2,eps4,gauge,forcecov,forceimpl,selfcheck,all",
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
            tests = {"layout", "timing", "fd", "eps2", "eps4", "gauge", "forcecov", "forceimpl"}

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    selected_layout = args.layout
    if "layout" in tests or args.layout == "auto":
        ta = U1YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            n_iter=args.n_iter_timing,
            seed=args.seed,
            kernel="action",
        )
        tf = U1YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=args.batch,
            n_iter=args.n_iter_timing,
            seed=args.seed,
            kernel="force",
        )
        print("Layout benchmark (sec/call):")
        for lay in sorted(ta):
            print(f"  {lay}: action={ta[lay]:.6e}, force={tf[lay]:.6e}")
        if args.layout == "auto":
            selected_layout = min(tf, key=tf.get)
            print("Auto-selected layout:", selected_layout)

    theory = U1YangMills(
        lattice_shape=lattice_shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=selected_layout,
        seed=args.seed,
    )
    print("Theory config:")
    print(f"  lattice_shape: {lattice_shape}")
    print(f"  beta: {args.beta}")
    print(f"  batch: {args.batch}")
    print(f"  layout: {selected_layout}")
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
        print(f"  log-log slope (expected ~2 for 2nd-order integrators): {eps['loglog_slope']:.4f}")
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
        g = gauge_invariance_test(
            theory,
            q_scale=0.05,
            omega_scale=args.gauge_omega_scale,
            n_trials=args.gauge_trials,
        )
        print("Gauge invariance test:")
        print(f"  max rel action diff under gauge transform: {g['max_rel_action_diff']:.6e}")
        print(f"  mean rel action diff under gauge transform: {g['mean_rel_action_diff']:.6e}")
        print(
            "  quality Omega/U/Up (unitarity):"
            f" {g['omega_unitarity']:.6e} / {g['U_unitarity']:.6e} / {g['Up_unitarity']:.6e}"
        )
        print(
            "  quality Omega/U/Up (phase_norm):"
            f" {g['omega_det_err']:.6e} / {g['U_det_err']:.6e} / {g['Up_det_err']:.6e}"
        )

    if "forcecov" in tests:
        fg = force_gauge_covariance_test(
            theory,
            q_scale=0.05,
            omega_scale=args.gauge_omega_scale,
            n_trials=args.gauge_trials,
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
        if bool(fim.get("enabled", 0.0)):
            print("Force implementation consistency test:")
            print(f"  max rel diff optimized vs reference: {fim['max_rel_force_diff']:.6e}")
            print(f"  mean rel diff optimized vs reference: {fim['mean_rel_force_diff']:.6e}")
            print(f"  max abs entrywise diff: {fim['max_abs_force_diff']:.6e}")
        else:
            print("Force implementation consistency test: skipped (optimized force not enabled for this layout/dimension)")

    if "selfcheck" in tests:
        report = full_selfcheck(
            theory=theory,
            gauge_trials=args.gauge_trials,
            gauge_omega_scale=args.gauge_omega_scale,
            fd_eps=args.fd_eps,
        )
        print("U1 full selfcheck:")
        for name, ok, msg in report["checks"]:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")
        print("  overall:", "PASS" if report["all_ok"] else "FAIL")
        if args.selfcheck_fail and not report["all_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

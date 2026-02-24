"""JAX port of HMC update utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class HMC:
    T: object
    I: object
    verbose: bool = True
    seed: int = 0
    use_fast_jit: bool = True
    AcceptReject: List[float] = field(default_factory=list)
    key: jax.Array = field(init=False, repr=False)
    _scan_cache: Dict[Tuple[int, Tuple[int, ...], str], object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.key = jax.random.PRNGKey(self.seed)

    def _split_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def calc_acceptance(self) -> float:
        return float(np.mean(self.AcceptReject)) if self.AcceptReject else 0.0

    def reset_acceptance(self) -> None:
        self.AcceptReject = []

    def _has_fast_refresh(self) -> bool:
        return hasattr(self.T, "refresh_p_with_key")

    def _evolve_python(self, q: jax.Array, N: int) -> jax.Array:
        qshape = tuple([q.shape[0]] + [1] * (q.ndim - 1))
        for k in range(int(N)):
            q0 = q
            p0 = self.T.refreshP()
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            p, q_prop = self.I.integrate(p0, q0)
            Hf = self.T.kinetic(p) + self.T.action(q_prop)
            dH = Hf - H0
            acc_prob = jnp.where(dH < 0, jnp.ones_like(dH), jnp.exp(-dH))
            R = jax.random.uniform(self._split_key(), shape=acc_prob.shape, dtype=acc_prob.dtype)
            acc_flag = R < acc_prob
            ar = jnp.where(acc_flag, 1.0, 0.0)
            self.AcceptReject.extend(np.asarray(ar).tolist())
            q = jnp.where(acc_flag.reshape(qshape), q_prop, q0)
            if self.verbose:
                print(
                    " HMC:",
                    k,
                    " dH=",
                    np.asarray(dH).tolist(),
                    " A/R=",
                    np.asarray(acc_flag).tolist(),
                    " Pacc=",
                    float(jnp.mean(ar)),
                )
        return q

    def _get_scan_kernel(self, q: jax.Array, N: int):
        cache_key = (int(N), tuple(int(v) for v in q.shape), str(q.dtype))
        if cache_key in self._scan_cache:
            return self._scan_cache[cache_key]

        qshape = tuple([q.shape[0]] + [1] * (q.ndim - 1))
        N = int(N)

        def run_scan(q_in: jax.Array, key_in: jax.Array):
            def one_step(carry, _):
                q_curr, key_curr = carry
                key_curr, kp, kr = jax.random.split(key_curr, 3)
                p0 = self.T.refresh_p_with_key(kp)
                H0 = self.T.kinetic(p0) + self.T.action(q_curr)
                p, q_prop = self.I.integrate(p0, q_curr)
                Hf = self.T.kinetic(p) + self.T.action(q_prop)
                dH = Hf - H0
                acc_prob = jnp.where(dH < 0, jnp.ones_like(dH), jnp.exp(-dH))
                R = jax.random.uniform(kr, shape=acc_prob.shape, dtype=acc_prob.dtype)
                acc_flag = R < acc_prob
                ar = jnp.where(acc_flag, 1.0, 0.0)
                q_next = jnp.where(acc_flag.reshape(qshape), q_prop, q_curr)
                return (q_next, key_curr), (ar, dH)

            (q_out, key_out), (ar_hist, dH_hist) = jax.lax.scan(one_step, (q_in, key_in), None, length=N)
            return q_out, key_out, ar_hist, dH_hist

        kernel = jax.jit(run_scan)
        self._scan_cache[cache_key] = kernel
        return kernel

    def _evolve_fast(self, q: jax.Array, N: int) -> jax.Array:
        kernel = self._get_scan_kernel(q, N)
        q_out, key_out, ar_hist, _ = kernel(q, self.key)
        self.key = key_out
        self.AcceptReject.extend(np.asarray(ar_hist).reshape(-1).tolist())
        return q_out

    def evolve(self, q: jax.Array, N: int) -> jax.Array:
        if int(N) <= 0:
            return q
        if self.use_fast_jit and (not self.verbose) and self._has_fast_refresh():
            return self._evolve_fast(q, int(N))
        return self._evolve_python(q, int(N))

    calc_Acceptance = calc_acceptance
    reset_Acceptance = reset_acceptance


hmc = HMC


@dataclass
class SMD:
    """Stochastic Molecular Dynamics / GHMC-style update.

    This follows the Chroma convention:
    - Ornstein-Uhlenbeck momentum refresh with trajectory length `I.t`
      p <- c1 * p + c2 * eta, c1=exp(-gamma*tau), c2=sqrt(1-c1^2)
    - MD proposal with integrator `I`
    - optional Metropolis accept/reject
    - on reject, restore post-OU (q,p) state.
    """

    T: object
    I: object
    gamma: float = 0.3
    accept_reject: bool = True
    verbose: bool = True
    seed: int = 0
    use_fast_jit: bool = True
    AcceptReject: List[float] = field(default_factory=list)
    key: jax.Array = field(init=False, repr=False)
    p: Optional[jax.Array] = field(default=None, init=False, repr=False)
    _scan_cache: Dict[Tuple[object, ...], object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.key = jax.random.PRNGKey(self.seed)
        self.gamma = float(self.gamma)

    def _split_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def calc_acceptance(self) -> float:
        return float(np.mean(self.AcceptReject)) if self.AcceptReject else 0.0

    def reset_acceptance(self) -> None:
        self.AcceptReject = []

    def reset_momentum(self) -> None:
        self.p = None

    def set_momentum(self, p: jax.Array) -> None:
        self.p = p

    def _has_fast_refresh(self) -> bool:
        return hasattr(self.T, "refresh_p_with_key")

    def _traj_length(self) -> float:
        return float(getattr(self.I, "t", 1.0))

    def _refresh_with_key(self, key: jax.Array) -> jax.Array:
        if hasattr(self.T, "refresh_p_with_key"):
            return self.T.refresh_p_with_key(key)
        return self.T.refreshP()

    def _init_momentum(self, q: jax.Array) -> None:
        if self.p is not None and self.p.shape == q.shape and self.p.dtype == q.dtype:
            return
        if hasattr(self.T, "refresh_p_with_key"):
            self.p = self.T.refresh_p_with_key(self._split_key())
        else:
            self.p = self.T.refreshP()

    def _ou_refresh(self, p_curr: jax.Array, eta: jax.Array) -> jax.Array:
        tau = self._traj_length()
        c1 = np.exp(-self.gamma * tau)
        c2 = np.sqrt(max(0.0, 1.0 - c1 * c1))
        return c1 * p_curr + c2 * eta

    def _evolve_python(self, q: jax.Array, N: int, warmup: bool) -> jax.Array:
        self._init_momentum(q)
        assert self.p is not None
        qshape = tuple([q.shape[0]] + [1] * (q.ndim - 1))
        pshape = tuple([self.p.shape[0]] + [1] * (self.p.ndim - 1))
        do_accept = bool(self.accept_reject and (not warmup))

        for k in range(int(N)):
            eta = self._refresh_with_key(self._split_key())
            p0 = self._ou_refresh(self.p, eta)
            q0 = q
            H0 = self.T.kinetic(p0) + self.T.action(q0)
            p_prop, q_prop = self.I.integrate(p0, q0)
            Hf = self.T.kinetic(p_prop) + self.T.action(q_prop)
            dH = Hf - H0

            if do_accept:
                acc_prob = jnp.where(dH < 0, jnp.ones_like(dH), jnp.exp(-dH))
                R = jax.random.uniform(self._split_key(), shape=acc_prob.shape, dtype=acc_prob.dtype)
                acc_flag = R < acc_prob
            else:
                acc_flag = jnp.ones_like(dH, dtype=bool)

            ar = jnp.where(acc_flag, 1.0, 0.0)
            self.AcceptReject.extend(np.asarray(ar).tolist())

            q = jnp.where(acc_flag.reshape(qshape), q_prop, q0)
            self.p = jnp.where(acc_flag.reshape(pshape), p_prop, p0)

            if self.verbose:
                print(
                    " SMD:",
                    k,
                    " dH=",
                    np.asarray(dH).tolist(),
                    " A/R=",
                    np.asarray(acc_flag).tolist(),
                    " Pacc=",
                    float(jnp.mean(ar)),
                    " warmup=",
                    bool(warmup),
                )
        return q

    def _get_scan_kernel(self, q: jax.Array, p: jax.Array, N: int, warmup: bool):
        cache_key = (
            int(N),
            bool(warmup),
            bool(self.accept_reject),
            float(self.gamma),
            float(self._traj_length()),
            tuple(int(v) for v in q.shape),
            str(q.dtype),
            tuple(int(v) for v in p.shape),
            str(p.dtype),
        )
        if cache_key in self._scan_cache:
            return self._scan_cache[cache_key]

        qshape = tuple([q.shape[0]] + [1] * (q.ndim - 1))
        pshape = tuple([p.shape[0]] + [1] * (p.ndim - 1))
        N = int(N)
        tau = float(self._traj_length())
        c1 = np.exp(-self.gamma * tau)
        c2 = np.sqrt(max(0.0, 1.0 - c1 * c1))
        do_accept = bool(self.accept_reject and (not warmup))

        def run_scan(q_in: jax.Array, p_in: jax.Array, key_in: jax.Array):
            def one_step(carry, _):
                q_curr, p_curr, key_curr = carry
                key_curr, keta, kacc = jax.random.split(key_curr, 3)
                eta = self.T.refresh_p_with_key(keta)
                p0 = c1 * p_curr + c2 * eta
                H0 = self.T.kinetic(p0) + self.T.action(q_curr)
                p_prop, q_prop = self.I.integrate(p0, q_curr)
                Hf = self.T.kinetic(p_prop) + self.T.action(q_prop)
                dH = Hf - H0

                if do_accept:
                    acc_prob = jnp.where(dH < 0, jnp.ones_like(dH), jnp.exp(-dH))
                    R = jax.random.uniform(kacc, shape=acc_prob.shape, dtype=acc_prob.dtype)
                    acc_flag = R < acc_prob
                else:
                    acc_flag = jnp.ones_like(dH, dtype=bool)

                ar = jnp.where(acc_flag, 1.0, 0.0)
                q_next = jnp.where(acc_flag.reshape(qshape), q_prop, q_curr)
                p_next = jnp.where(acc_flag.reshape(pshape), p_prop, p0)
                return (q_next, p_next, key_curr), (ar, dH)

            (q_out, p_out, key_out), (ar_hist, dH_hist) = jax.lax.scan(
                one_step,
                (q_in, p_in, key_in),
                None,
                length=N,
            )
            return q_out, p_out, key_out, ar_hist, dH_hist

        kernel = jax.jit(run_scan)
        self._scan_cache[cache_key] = kernel
        return kernel

    def _evolve_fast(self, q: jax.Array, N: int, warmup: bool) -> jax.Array:
        self._init_momentum(q)
        assert self.p is not None
        kernel = self._get_scan_kernel(q, self.p, N, warmup)
        q_out, p_out, key_out, ar_hist, _ = kernel(q, self.p, self.key)
        self.key = key_out
        self.p = p_out
        self.AcceptReject.extend(np.asarray(ar_hist).reshape(-1).tolist())
        return q_out

    def evolve(self, q: jax.Array, N: int, warmup: bool = False) -> jax.Array:
        if int(N) <= 0:
            return q
        if self.use_fast_jit and (not self.verbose) and self._has_fast_refresh():
            return self._evolve_fast(q, int(N), bool(warmup))
        return self._evolve_python(q, int(N), bool(warmup))

    calc_Acceptance = calc_acceptance
    reset_Acceptance = reset_acceptance
    reset_Momentum = reset_momentum


smd = SMD
ghmc = SMD

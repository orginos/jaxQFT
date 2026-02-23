"""JAX port of HMC update utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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

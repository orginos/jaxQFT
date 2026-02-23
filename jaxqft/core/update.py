"""JAX port of HMC update utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class HMC:
    T: object
    I: object
    verbose: bool = True
    seed: int = 0
    AcceptReject: List[float] = field(default_factory=list)
    key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.key = jax.random.PRNGKey(self.seed)

    def _split_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def calc_acceptance(self) -> float:
        return float(np.mean(self.AcceptReject)) if self.AcceptReject else 0.0

    def reset_acceptance(self) -> None:
        self.AcceptReject = []

    def evolve(self, q: jax.Array, N: int) -> jax.Array:
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

    calc_Acceptance = calc_acceptance
    reset_Acceptance = reset_acceptance


hmc = HMC


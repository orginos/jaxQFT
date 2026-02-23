"""JAX port of the torch phi^4 lattice theory helper."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass
class Phi4:
    V: Sequence[int]
    lam: float
    mass: float
    batch_size: int = 1
    dtype: jnp.dtype = jnp.float32
    seed: int = 0
    key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.V = tuple(int(v) for v in self.V)
        self.Vol = int(math.prod(self.V))
        self.Nd = len(self.V)
        self.mtil = self.mass + 2.0 * self.Nd
        self.Bs = int(self.batch_size)
        self.key = jax.random.PRNGKey(self.seed)

    def _split_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def action(self, phi: jax.Array) -> jax.Array:
        phi2 = phi * phi
        A = jnp.sum((0.5 * self.mtil + (self.lam / 24.0) * phi2) * phi2, axis=tuple(range(1, phi.ndim)))
        for mu in range(1, self.Nd + 1):
            A = A - jnp.sum(phi * jnp.roll(phi, shift=-1, axis=mu), axis=tuple(range(1, phi.ndim)))
        return A

    def force(self, phi: jax.Array) -> jax.Array:
        F = -self.mtil * phi - self.lam * phi**3 / 6.0
        for mu in range(1, self.Nd + 1):
            F = F + jnp.roll(phi, shift=1, axis=mu) + jnp.roll(phi, shift=-1, axis=mu)
        return F

    def refresh_p(self) -> jax.Array:
        return jax.random.normal(self._split_key(), shape=(self.Bs, *self.V), dtype=self.dtype)

    def evolve_q(self, dt: float, P: jax.Array, Q: jax.Array) -> jax.Array:
        return Q + dt * P

    def kinetic(self, P: jax.Array) -> jax.Array:
        return jnp.sum(P * P, axis=tuple(range(1, P.ndim))) / 2.0

    def hot_start(self) -> jax.Array:
        return jax.random.normal(self._split_key(), shape=(self.Bs, *self.V), dtype=self.dtype)

    refreshP = refresh_p
    evolveQ = evolve_q
    hotStart = hot_start


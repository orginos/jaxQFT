"""JAX port of torchQFT hmc_sampler helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from . import integrators as integ
from .update import HMC


Array = jax.Array


@dataclass
class HMCSampler:
    init_cnf: Array
    Nhmc: int = 4
    Nmd: int = 5
    integrator: str = "minnorm2"
    action: Callable[[Array], Array] = lambda x: 0.5 * jnp.sum(x * x, axis=1)
    verbose: bool = True

    def __post_init__(self) -> None:
        self.state = self.init_cnf
        self.Nb = int(self.init_cnf.shape[0])
        self.field_shape = tuple(self.init_cnf[0].shape)
        self.shape = tuple(self.init_cnf.shape)
        self.I = getattr(integ, self.integrator)(self.force, self.evolveQ, self.Nmd, 1.0)
        self.hmc = HMC(T=self, I=self.I, verbose=self.verbose)
        self.key = jax.random.PRNGKey(0)

    def _split_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def set_action(self, action: Callable[[Array], Array], Nwarm: int = 10) -> None:
        self.action = action
        self.warm_up(Nwarm)

    def force(self, x: Array) -> Array:
        grad_fn = jax.grad(lambda y: jnp.sum(self.action(y)))
        return -grad_fn(x)

    def refreshP(self) -> Array:
        return jax.random.normal(self._split_key(), shape=self.shape, dtype=self.state.dtype)

    def evolveQ(self, dt: float, P: Array, Q: Array) -> Array:
        return Q + dt * P

    def kinetic(self, P: Array) -> Array:
        return jnp.sum(P * P, axis=tuple(range(1, P.ndim))) / 2.0

    def warm_up(self, Nwarm: int) -> None:
        self.state = self.hmc.evolve(self.state, Nwarm)

    def epsilon_test(self, iN: int = 1, fN: int = 3, steps: int = 50, integrator: str = "minnorm2"):
        x, y, ey = [], [], []
        P = self.refreshP()
        Hi = self.kinetic(P) + self.action(self.state)
        for rk in np.logspace(iN, fN, steps):
            k = int(rk)
            dt = 1.0 / k
            l = getattr(integ, integrator)(self.force, self.evolveQ, k, 1.0)
            PP, QQ = l.integrate(P, self.state)
            Hf = self.kinetic(PP) + self.action(QQ)
            dH = jnp.abs(Hf - Hi)
            x.append(dt)
            y.append(float(jnp.mean(dH)))
            ey.append(float(jnp.std(dH) / jnp.sqrt(dH.shape[0] - 1)))
        return x, y, ey

    def sample(self, shape: Sequence[int]) -> Array:
        N = int(np.prod(shape))
        tt = N // self.Nb + (1 if N % self.Nb else 0)
        foo = self.hmc.evolve(self.state, self.Nhmc)
        total = foo
        for _ in range(tt - 1):
            foo = self.hmc.evolve(foo, self.Nhmc)
            total = jnp.concatenate([total, foo], axis=0)
        result = total[:N].reshape(tuple(shape) + self.field_shape)
        self.state = foo
        return result


hmc_sampler = HMCSampler


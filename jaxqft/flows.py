"""Minimal JAX flow components compatible with torchQFT conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass
class Shift2D:
    shift: Tuple[int, int]

    def __post_init__(self) -> None:
        self.ishift = (-self.shift[0], -self.shift[1])

    def __call__(self, x: Array) -> Array:
        return jnp.roll(x, shift=self.shift, axis=(-2, -1))

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return jnp.roll(x, shift=self.ishift, axis=(-2, -1)), jnp.zeros((x.shape[0],), dtype=x.dtype)


@dataclass
class AffineCoupling:
    mask: Array
    s_fn: Callable[[Array], Array]
    t_fn: Callable[[Array], Array]

    def __call__(self, x: Array) -> Array:
        x_ = x * self.mask
        s = self.s_fn(x_) * (1 - self.mask)
        t = self.t_fn(x_) * (1 - self.mask)
        return x_ + (1 - self.mask) * (x * jnp.exp(s) + t)

    def inverse(self, z: Array) -> Tuple[Array, Array]:
        z_ = z * self.mask
        s = self.s_fn(z_) * (1 - self.mask)
        t = self.t_fn(z_) * (1 - self.mask)
        x = (1 - self.mask) * (z - t) * jnp.exp(-s) + z_
        log_det = -jnp.sum(s, axis=tuple(range(1, s.ndim)))
        return x, log_det


class FlowModel:
    def __init__(self, flows: Sequence[object], prior_sample: Callable[[jax.Array, int], Array], prior_log_prob: Callable[[Array], Array]):
        self.flows = list(flows)
        self.prior_sample = prior_sample
        self.prior_log_prob = prior_log_prob

    def __call__(self, x: Array) -> Array:
        for flow in self.flows:
            x = flow(x)
        return x

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        log_det_jac = jnp.zeros((x.shape[0],), dtype=x.dtype)
        for flow in reversed(self.flows):
            x, j = flow.inverse(x)
            log_det_jac = log_det_jac + j
        return x, log_det_jac

    def log_prob(self, x: Array) -> Array:
        z, logp = self.inverse(x)
        return self.prior_log_prob(z) + logp

    def sample(self, key: jax.Array, batch_size: int) -> Array:
        z = self.prior_sample(key, batch_size)
        return self(z)


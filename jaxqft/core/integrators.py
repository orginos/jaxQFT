"""JAX port of molecular dynamics integrators used in torchQFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def simple_evolve_p(dt: float, F: Array, P: Array) -> Array:
    return P + dt * F


@dataclass
class Integrator:
    force: Callable[[Array], Array]
    evolve_q: Callable[[float, Array, Array], Array]
    Nmd: int
    t: float
    evolve_p: Callable[[float, Array, Array], Array] = simple_evolve_p

    def __post_init__(self) -> None:
        self.Nmd = int(self.Nmd)
        self.dt = float(self.t) / self.Nmd

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        raise NotImplementedError


class Leapfrog(Integrator):
    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        p = self.evolve_p(0.5 * self.dt, self.force(q), p)
        for _ in range(1, self.Nmd):
            q = self.evolve_q(self.dt, p, q)
            p = self.evolve_p(self.dt, self.force(q), p)
        q = self.evolve_q(self.dt, p, q)
        p = self.evolve_p(0.5 * self.dt, self.force(q), p)
        return p, q


class MinNorm2(Integrator):
    def __init__(self, force, evolve_q, Nmd, t, evolve_p=simple_evolve_p, lam=0.1931833275037836):
        super().__init__(force, evolve_q, Nmd, t, evolve_p)
        self.lam = float(lam)

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        for _ in range(self.Nmd):
            p = self.evolve_p(self.dt * self.lam, self.force(q), p)
            q = self.evolve_q(0.5 * self.dt, p, q)
            p = self.evolve_p(self.dt * (1.0 - 2.0 * self.lam), self.force(q), p)
            q = self.evolve_q(0.5 * self.dt, p, q)
            p = self.evolve_p(self.dt * self.lam, self.force(q), p)
        return p, q


class ForceGradient(Integrator):
    """Chroma-style STS force-gradient integrator (LCM_STS_FORCE_GRAD)."""

    def __init__(self, force, evolve_q, Nmd, t, evolve_p=simple_evolve_p, lam=1.0 / 6.0, xi=1.0 / 72.0):
        super().__init__(force, evolve_q, Nmd, t, evolve_p)
        self.lam = float(lam)
        self.xi = float(xi)

    def _fg_update(self, p: Array, q: Array, dt1: float, dt2: float) -> Tuple[Array, Array]:
        # Mirrors Chroma fg_update:
        # 1) build temporary p = dt1*F(q) from zero momentum
        # 2) drift q by unit T-flow with that temporary momentum
        # 3) update original momentum by dt2*F(q_tmp), restore original q
        p_tmp = jax.tree.map(jnp.zeros_like, p)
        p_tmp = self.evolve_p(dt1, self.force(q), p_tmp)
        q_tmp = self.evolve_q(1.0, p_tmp, q)
        p = self.evolve_p(dt2, self.force(q_tmp), p)
        return p, q

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        dt = self.dt
        lambda_dt = dt * self.lam
        dtauby2 = 0.5 * dt
        one_minus_2lambda_dt = (1.0 - 2.0 * self.lam) * dt
        two_lambda_dt = 2.0 * lambda_dt
        xi_dtdt = (2.0 * dt * dt * dt * self.xi) / one_minus_2lambda_dt

        p = self.evolve_p(lambda_dt, self.force(q), p)
        for _ in range(1, self.Nmd):
            q = self.evolve_q(dtauby2, p, q)
            p, q = self._fg_update(p, q, xi_dtdt, one_minus_2lambda_dt)
            q = self.evolve_q(dtauby2, p, q)
            p = self.evolve_p(two_lambda_dt, self.force(q), p)

        q = self.evolve_q(dtauby2, p, q)
        p, q = self._fg_update(p, q, xi_dtdt, one_minus_2lambda_dt)
        q = self.evolve_q(dtauby2, p, q)
        p = self.evolve_p(lambda_dt, self.force(q), p)
        return p, q


class MinNorm4PF4(Integrator):
    def __init__(
        self,
        force,
        evolve_q,
        Nmd,
        t,
        evolve_p=simple_evolve_p,
        rho=0.1786178958448091,
        theta=-0.06626458266981843,
        lam=0.7123418310626056,
    ):
        super().__init__(force, evolve_q, Nmd, t, evolve_p)
        self.lam = float(lam)
        self.rho = float(rho)
        self.the = float(theta)

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        for _ in range(self.Nmd):
            p = self.evolve_p(self.dt * self.rho, self.force(q), p)
            q = self.evolve_q(self.dt * self.lam, p, q)
            p = self.evolve_p(self.dt * self.the, self.force(q), p)
            q = self.evolve_q(self.dt * (1.0 - 2.0 * self.lam) / 2.0, p, q)
            p = self.evolve_p(self.dt * (1.0 - 2.0 * (self.the + self.rho)), self.force(q), p)
            q = self.evolve_q(self.dt * (1.0 - 2.0 * self.lam) / 2.0, p, q)
            p = self.evolve_p(self.dt * self.the, self.force(q), p)
            q = self.evolve_q(self.dt * self.lam, p, q)
            p = self.evolve_p(self.dt * self.rho, self.force(q), p)
        return p, q


class RRESPALeapfrog(Integrator):
    def __init__(self, fast_force, slow_force, evolve_q, Nouter, t, Ninner=4, evolve_p=simple_evolve_p):
        super().__init__(fast_force, evolve_q, Nouter, t, evolve_p)
        self.fast_force = fast_force
        self.slow_force = slow_force
        self.Ninner = int(Ninner)

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        dt_inner = self.dt / self.Ninner
        for _ in range(self.Nmd):
            p = self.evolve_p(0.5 * self.dt, self.slow_force(q), p)
            for _ in range(self.Ninner):
                p = self.evolve_p(0.5 * dt_inner, self.fast_force(q), p)
                q = self.evolve_q(dt_inner, p, q)
                p = self.evolve_p(0.5 * dt_inner, self.fast_force(q), p)
            p = self.evolve_p(0.5 * self.dt, self.slow_force(q), p)
        return p, q


class RRESPAMinNorm2(Integrator):
    def __init__(
        self,
        fast_force,
        slow_force,
        evolve_q,
        Nouter,
        t,
        Ninner=4,
        evolve_p=simple_evolve_p,
        lam=0.1931833275037836,
    ):
        super().__init__(fast_force, evolve_q, Nouter, t, evolve_p)
        self.fast_force = fast_force
        self.slow_force = slow_force
        self.Ninner = int(Ninner)
        self.lam = float(lam)

    def integrate(self, p: Array, q: Array) -> Tuple[Array, Array]:
        dt_inner = self.dt / self.Ninner
        for _ in range(self.Nmd):
            p = self.evolve_p(0.5 * self.dt, self.slow_force(q), p)
            for _ in range(self.Ninner):
                p = self.evolve_p(dt_inner * self.lam, self.fast_force(q), p)
                q = self.evolve_q(0.5 * dt_inner, p, q)
                p = self.evolve_p(dt_inner * (1.0 - 2.0 * self.lam), self.fast_force(q), p)
                q = self.evolve_q(0.5 * dt_inner, p, q)
                p = self.evolve_p(dt_inner * self.lam, self.fast_force(q), p)
            p = self.evolve_p(0.5 * self.dt, self.slow_force(q), p)
        return p, q


integrator = Integrator
leapfrog = Leapfrog
minnorm2 = MinNorm2
force_gradient = ForceGradient
minnorm4pf4 = MinNorm4PF4
rrespa_leapfrog = RRESPALeapfrog
rrespa_minnorm2 = RRESPAMinNorm2
simple_evolveP = simple_evolve_p

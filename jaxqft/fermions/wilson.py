"""Reusable Wilson-Dirac operator blocks for lattice gauge theories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import jax.numpy as jnp

from .gamma import build_euclidean_gamma


Array = jnp.ndarray


@dataclass
class WilsonDiracOperator:
    """Wilson-Dirac operator independent of gauge group choice.

    The gauge-theory model supplies geometry/layout callbacks:
    - `take_mu(U, mu)` -> link field U_mu with shape [..., Nc, Nc]
    - `roll_site(x, shift, dir)` -> nearest-neighbor shift on site axes
    - `dagger(U)` -> matrix adjoint on the last two indices
    """

    ndim: int
    mass: float = 0.0
    wilson_r: float = 1.0
    dtype: jnp.dtype = jnp.complex64
    gamma: Array | None = None

    def __post_init__(self):
        self.Nd = int(self.ndim)
        if self.Nd < 1:
            raise ValueError("ndim must be >= 1")
        self.gamma = build_euclidean_gamma(self.Nd, dtype=self.dtype) if self.gamma is None else self.gamma
        self.Ns = int(self.gamma.shape[-1])
        self.spin_eye = jnp.eye(self.Ns, dtype=self.dtype)
        self._gamma_sparse_ok = False
        self._gamma_perm = None
        self._gamma_phase = None
        self._setup_gamma_sparse()

    def _setup_gamma_sparse(self) -> None:
        perm = []
        phase = []
        ok = True
        for mu in range(self.Nd):
            g = jnp.asarray(self.gamma[mu])
            row_nnz = jnp.sum(jnp.abs(g) > 1e-7, axis=1)
            if not bool(jnp.all(row_nnz == 1)):
                ok = False
                break
            idx = jnp.argmax(jnp.abs(g), axis=1).astype(jnp.int32)
            val = g[jnp.arange(self.Ns), idx]
            perm.append(idx)
            phase.append(val)
        if ok:
            self._gamma_sparse_ok = True
            self._gamma_perm = jnp.stack(perm, axis=0)  # [Nd, Ns]
            self._gamma_phase = jnp.stack(phase, axis=0)  # [Nd, Ns]

    @property
    def sparse_gamma_available(self) -> bool:
        return bool(self._gamma_sparse_ok)

    def color_mul_left(self, U: Array, psi: Array) -> Array:
        # U: [...,Nc,Nc], psi: [...,Ns,Nc] -> [...,Ns,Nc]
        return jnp.einsum("...ab,...sb->...sa", U, psi)

    def gamma_apply(self, psi: Array, mu: int, use_sparse: bool = True) -> Array:
        # psi: [...,Ns,Nc]
        if use_sparse and self._gamma_sparse_ok:
            idx = self._gamma_perm[mu]  # [Ns]
            ph = self._gamma_phase[mu]  # [Ns]
            gathered = jnp.take(psi, idx, axis=-2)
            ph_shape = (1,) * (psi.ndim - 2) + (self.Ns, 1)
            return jnp.reshape(ph, ph_shape) * gathered
        return jnp.einsum("st,...tc->...sc", self.gamma[mu], psi)

    def spin_project(self, psi: Array, mu: int, coeff: float, use_sparse: bool = True) -> Array:
        # (r I + coeff * gamma_mu) psi
        return self.wilson_r * psi + coeff * self.gamma_apply(psi, mu, use_sparse=use_sparse)

    def apply(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        sign: int = -1,
        use_sparse_gamma: bool = True,
    ) -> Array:
        # sign=-1 -> D, sign=+1 -> D^\dagger
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (D) or +1 (Ddag)")
        out = (self.mass + self.wilson_r * self.Nd) * psi
        for mu in range(self.Nd):
            U_mu = take_mu(U, mu)  # [...,Nc,Nc]
            psi_xpmu = roll_site(psi, -1, mu)
            psi_xmmu = roll_site(psi, +1, mu)

            fwd = self.color_mul_left(U_mu, psi_xpmu)
            U_mu_xmmu = roll_site(U_mu, +1, mu)
            bwd = self.color_mul_left(dagger(U_mu_xmmu), psi_xmmu)

            fwd = self.spin_project(fwd, mu, coeff=-sign, use_sparse=use_sparse_gamma)
            bwd = self.spin_project(bwd, mu, coeff=+sign, use_sparse=use_sparse_gamma)
            out = out - 0.5 * (fwd + bwd)
        return out

    def apply_dagger(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        use_sparse_gamma: bool = True,
    ) -> Array:
        return self.apply(
            U, psi, take_mu=take_mu, roll_site=roll_site, dagger=dagger, sign=+1, use_sparse_gamma=use_sparse_gamma
        )

    def apply_normal(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        use_sparse_gamma: bool = True,
    ) -> Array:
        dpsi = self.apply(
            U, psi, take_mu=take_mu, roll_site=roll_site, dagger=dagger, sign=-1, use_sparse_gamma=use_sparse_gamma
        )
        return self.apply(
            U, dpsi, take_mu=take_mu, roll_site=roll_site, dagger=dagger, sign=+1, use_sparse_gamma=use_sparse_gamma
        )

    def gamma_sparse_dense_error(self, psi: Array) -> Dict[str, float]:
        if not self._gamma_sparse_ok:
            return {"sparse_available": 0.0, "max_rel_error": float("nan")}
        errs = []
        for mu in range(self.Nd):
            gs = self.gamma_apply(psi, mu, use_sparse=True)
            gd = self.gamma_apply(psi, mu, use_sparse=False)
            rel = jnp.linalg.norm(gs - gd) / (jnp.linalg.norm(gd) + 1e-12)
            errs.append(float(rel))
        return {"sparse_available": 1.0, "max_rel_error": float(max(errs))}


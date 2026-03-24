"""Time-slab domain-decomposed pure-gauge updates for U(1) theories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxqft.core.domain_decomposition import TimeSlabDecomposition


Array = jax.Array


@dataclass
class U1TimeSlabDDTheory:
    """Conditional pure-gauge theory with frozen time-slab boundaries.

    The wrapped theory supplies the U(1) gauge dynamics. This wrapper
    constrains MD evolution to links in the DD interior while keeping the
    boundary links fixed.
    """

    base: object
    boundary_slices: Tuple[int, ...]
    boundary_width: int = 1
    jit_wrappers: bool = True
    decomposition: TimeSlabDecomposition = field(init=False)
    requires_trajectory_refresh: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not hasattr(self.base, "field_shape"):
            raise AttributeError("base theory must expose field_shape()")
        if not hasattr(self.base, "action"):
            raise AttributeError("base theory must expose action(U)")
        if not hasattr(self.base, "force"):
            raise AttributeError("base theory must expose force(U)")
        if not hasattr(self.base, "evolve_q"):
            raise AttributeError("base theory must expose evolve_q(dt, P, Q)")
        if not hasattr(self.base, "refresh_p"):
            raise AttributeError("base theory must expose refresh_p()")
        if not hasattr(self.base, "kinetic"):
            raise AttributeError("base theory must expose kinetic(P)")
        if not hasattr(self.base, "_take_mu") or not hasattr(self.base, "_roll"):
            raise AttributeError("base theory must expose _take_mu and _roll lattice helpers")

        self.decomposition = TimeSlabDecomposition(
            lattice_shape=tuple(int(v) for v in tuple(self.base.lattice_shape)),
            boundary_slices=tuple(int(v) for v in tuple(self.boundary_slices)),
            boundary_width=int(self.boundary_width),
        )
        self.boundary_slices = tuple(int(v) for v in self.decomposition.boundary_slices)
        self.boundary_width = int(self.decomposition.boundary_width)
        self.requires_trajectory_refresh = bool(getattr(self.base, "requires_trajectory_refresh", False))

        self._active_mask_np = self.decomposition.link_active_mask(
            layout=str(self.base.layout),
            batch_size=int(self.base.Bs),
            dtype=np.float32,
        )
        self._frozen_mask_np = 1.0 - self._active_mask_np
        self._active_mask = jnp.asarray(self._active_mask_np)
        self._frozen_mask = jnp.asarray(self._frozen_mask_np)

        active_links_bm = self.decomposition.link_active_mask(layout="BMXYIJ", batch_size=1, dtype=np.bool_)[0]
        plaquette_masks: Dict[Tuple[int, int], np.ndarray] = {}
        for mu in range(int(self.base.Nd)):
            active_mu = np.asarray(active_links_bm[mu], dtype=bool)
            for nu in range(mu + 1, int(self.base.Nd)):
                active_nu = np.asarray(active_links_bm[nu], dtype=bool)
                mask = (
                    active_mu
                    | np.roll(active_nu, shift=-1, axis=mu)
                    | np.roll(active_mu, shift=-1, axis=nu)
                    | active_nu
                )
                plaquette_masks[(mu, nu)] = mask.astype(np.float32, copy=False)
        self._plaquette_masks = {k: jnp.asarray(v) for k, v in plaquette_masks.items()}
        self.active_link_fraction = float(np.mean(self._active_mask_np))
        self.frozen_link_fraction = float(np.mean(self._frozen_mask_np))
        self.active_plaquette_fraction = float(
            np.mean([float(np.mean(np.asarray(v))) for v in plaquette_masks.values()])
        )

        if bool(self.jit_wrappers):
            _action_reference = self.action_reference
            _force_reference = self.force_reference
            _evolve_q_reference = self.evolve_q_reference
            _refresh_with_key_reference = self.refresh_p_with_key_reference
            self.action = jax.jit(lambda U: _action_reference(U))
            self.force = jax.jit(lambda U: _force_reference(U))
            self.evolve_q = jax.jit(lambda dt, P, Q: _evolve_q_reference(dt, P, Q))
            self.refresh_p_with_key = jax.jit(lambda key: _refresh_with_key_reference(key))

    def field_shape(self):
        return self.base.field_shape()

    @property
    def active_mask(self) -> Array:
        return self._active_mask

    @property
    def frozen_mask(self) -> Array:
        return self._frozen_mask

    def project_active_links(self, x: Array) -> Array:
        return x * self._active_mask.astype(x.dtype)

    def project_frozen_links(self, x: Array) -> Array:
        return x * self._frozen_mask.astype(x.dtype)

    def prepare_trajectory(self, q: Array, traj_length: float = 1.0) -> None:
        fn = getattr(self.base, "prepare_trajectory", None)
        if callable(fn):
            fn(q, traj_length=float(traj_length))

    def action_reference(self, U: Array) -> Array:
        B = int(U.shape[0])
        S = jnp.zeros((B,), dtype=jnp.float32)
        for mu in range(int(self.base.Nd)):
            U_mu = self.base._take_mu(U, mu)
            for nu in range(mu + 1, int(self.base.Nd)):
                U_nu = self.base._take_mu(U, nu)
                U_nu_xpmu = self.base._roll(U_nu, -1, mu)
                U_mu_xpnu = self.base._roll(U_mu, -1, nu)
                plaq = U_mu * U_nu_xpmu * jnp.conj(U_mu_xpnu) * jnp.conj(U_nu)
                trp = jnp.real(plaq) * self._plaquette_masks[(mu, nu)].astype(plaq.real.dtype)
                S = S - float(self.base.beta) * jnp.sum(trp, axis=tuple(range(1, trp.ndim)))
        return S

    def full_action(self, U: Array) -> Array:
        return self.base.action(U)

    def force_reference(self, U: Array) -> Array:
        return self.project_active_links(self.base.force(U))

    def refresh_p_reference(self) -> Array:
        return self.project_active_links(self.base.refresh_p())

    def refresh_p_with_key_reference(self, key: Array) -> Array:
        if hasattr(self.base, "refresh_p_with_key"):
            return self.project_active_links(self.base.refresh_p_with_key(key))
        return self.refresh_p_reference()

    def refresh_p(self) -> Array:
        return self.refresh_p_reference()

    def refresh_p_with_key(self, key: Array) -> Array:
        return self.refresh_p_with_key_reference(key)

    def evolve_q_reference(self, dt: float, P: Array, Q: Array) -> Array:
        return self.base.evolve_q(dt, self.project_active_links(P), Q)

    def evolve_q(self, dt: float, P: Array, Q: Array) -> Array:
        return self.evolve_q_reference(dt, P, Q)

    def kinetic(self, P: Array) -> Array:
        return self.base.kinetic(self.project_active_links(P))

    def force(self, U: Array) -> Array:
        return self.force_reference(U)

    def action(self, U: Array) -> Array:
        return self.action_reference(U)

    def average_plaquette(self, U: Array) -> Array:
        return self.base.average_plaquette(U)

    def metadata(self) -> Mapping[str, float]:
        return {
            "dd_n_boundary_slices": float(len(self.boundary_slices)),
            "dd_boundary_width": float(self.boundary_width),
            "dd_active_link_fraction": float(self.active_link_fraction),
            "dd_frozen_link_fraction": float(self.frozen_link_fraction),
            "dd_active_plaquette_fraction": float(self.active_plaquette_fraction),
        }

    refreshP = refresh_p
    evolveQ = evolve_q

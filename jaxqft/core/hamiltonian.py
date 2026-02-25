"""Composable Hamiltonian model helpers built from action/force monomials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, runtime_checkable

import jax


Array = jax.Array


@runtime_checkable
class Monomial(Protocol):
    """Interface for one Hamiltonian contribution S_i(U)."""

    name: str
    timescale: int
    stochastic: bool

    def refresh(self, q: Array, traj_length: float = 1.0) -> None:
        """Refresh stochastic auxiliary fields (pseudofermions, noise, etc.)."""

    def action(self, q: Array) -> Array:
        """Return batched action contribution."""

    def force(self, q: Array) -> Array:
        """Return batched Lie-algebra force contribution."""


@dataclass
class HamiltonianModel:
    """Mix-in style helper that aggregates a list of monomials."""

    monomials: Sequence[Monomial] = field(default_factory=tuple)

    def set_monomials(self, monomials: Sequence[Monomial]) -> None:
        self.monomials = tuple(monomials)

    def prepare_trajectory(self, q: Array, traj_length: float = 1.0) -> None:
        for m in self.monomials:
            m.refresh(q, traj_length=float(traj_length))

    def action_from_monomials(self, q: Array) -> Array:
        if not self.monomials:
            raise ValueError("No monomials configured")
        total = None
        for m in self.monomials:
            term = m.action(q)
            total = term if total is None else (total + term)
        return total

    def force_from_monomials(self, q: Array) -> Array:
        if not self.monomials:
            raise ValueError("No monomials configured")
        total = None
        for m in self.monomials:
            term = m.force(q)
            total = term if total is None else (total + term)
        return total

    def action_breakdown(self, q: Array) -> Dict[str, Array]:
        return {m.name: m.action(q) for m in self.monomials}

    def monomial_names(self) -> List[str]:
        return [m.name for m in self.monomials]

    def monomials_by_timescale(self) -> Dict[int, List[Monomial]]:
        out: Dict[int, List[Monomial]] = {}
        for m in self.monomials:
            lvl = int(getattr(m, "timescale", 0))
            out.setdefault(lvl, []).append(m)
        return out

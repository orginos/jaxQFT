"""Fermion building blocks (gamma algebra and Wilson operators)."""

from .gamma import (
    build_euclidean_gamma,
    gamma5,
    check_gamma_algebra,
)
from .wilson import WilsonDiracOperator

__all__ = [
    "build_euclidean_gamma",
    "gamma5",
    "check_gamma_algebra",
    "WilsonDiracOperator",
]

"""Core JAX implementations for lattice QFT utilities."""

from importlib import import_module
import os
import platform

# Apple Metal JAX can fail early on some setups (e.g. PRNG key creation).
# Default to CPU unless the user has explicitly selected a JAX platform.
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

from .models import Phi4
from .core import (
    Integrator,
    Leapfrog,
    ForceGradient,
    MinNorm2,
    MinNorm4PF4,
    RRESPALeapfrog,
    RRESPAMinNorm2,
    HMC,
    HMCSampler,
)


def __getattr__(name):
    if name == "LieGroups":
        return import_module(".LieGroups", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Phi4",
    "Integrator",
    "Leapfrog",
    "ForceGradient",
    "MinNorm2",
    "MinNorm4PF4",
    "RRESPALeapfrog",
    "RRESPAMinNorm2",
    "HMC",
    "HMCSampler",
    "LieGroups",
]

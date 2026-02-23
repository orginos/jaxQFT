"""Model-specific theory implementations."""

from importlib import import_module

from .phi4 import Phi4
from .phi4_mg import (
    init_mgflow,
    mgflow_g,
    mgflow_f,
    mgflow_log_prob,
    mgflow_prior_sample,
    mgflow_prior_log_prob,
)
from .stacked_mg import (
    init_stacked_mg,
    stacked_g,
    stacked_f,
    stacked_log_prob,
    stacked_prior_sample,
)


def __getattr__(name):
    if name == "SU3YangMills":
        return import_module(".su3_ym", __name__).SU3YangMills
    if name == "SU2YangMills":
        return import_module(".su2_ym", __name__).SU2YangMills
    if name == "U1YangMills":
        return import_module(".u1_ym", __name__).U1YangMills
    if name == "SU3WilsonNf2":
        return import_module(".su3_wilson_nf2", __name__).SU3WilsonNf2
    if name == "SU2WilsonNf2":
        return import_module(".su2_wilson_nf2", __name__).SU2WilsonNf2
    if name == "U1WilsonNf2":
        return import_module(".u1_wilson_nf2", __name__).U1WilsonNf2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Phi4",
    "init_mgflow",
    "mgflow_g",
    "mgflow_f",
    "mgflow_log_prob",
    "mgflow_prior_sample",
    "mgflow_prior_log_prob",
    "init_stacked_mg",
    "stacked_g",
    "stacked_f",
    "stacked_log_prob",
    "stacked_prior_sample",
    "SU3YangMills",
    "SU2YangMills",
    "U1YangMills",
    "SU3WilsonNf2",
    "SU2WilsonNf2",
    "U1WilsonNf2",
]

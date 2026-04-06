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
from .phi4_rg_cond_flow import (
    init_rg_cond_flow,
    rg_cond_flow_g,
    rg_cond_flow_f,
    rg_cond_flow_log_prob,
    rg_cond_flow_prior_sample,
    rg_cond_flow_prior_log_prob,
)
from .phi4_rg_checkerboard_flow import (
    init_rg_checkerboard_flow,
    rg_checkerboard_flow_g,
    rg_checkerboard_flow_f,
    rg_checkerboard_flow_log_prob,
    rg_checkerboard_flow_prior_sample,
    rg_checkerboard_flow_prior_log_prob,
)
from .phi4_rg_checkerboard_inblock_flow import (
    init_rg_checkerboard_inblock_flow,
    rg_checkerboard_inblock_flow_g,
    rg_checkerboard_inblock_flow_f,
    rg_checkerboard_inblock_flow_log_prob,
    rg_checkerboard_inblock_flow_prior_sample,
    rg_checkerboard_inblock_flow_prior_log_prob,
)
from .phi4_rg_coarse_eta_flow import (
    init_rg_coarse_eta_flow,
    rg_coarse_eta_flow_g,
    rg_coarse_eta_flow_f,
    rg_coarse_eta_flow_log_prob,
    rg_coarse_eta_flow_prior_sample,
    rg_coarse_eta_flow_prior_log_prob,
)
from .stacked_mg import (
    init_stacked_mg,
    stacked_g,
    stacked_f,
    stacked_log_prob,
    stacked_prior_sample,
)


def __getattr__(name):
    if name == "ONSigmaModel":
        return import_module(".on_sigma", __name__).ONSigmaModel
    if name == "SU3YangMills":
        return import_module(".su3_ym", __name__).SU3YangMills
    if name == "SU2YangMills":
        return import_module(".su2_ym", __name__).SU2YangMills
    if name == "U1YangMills":
        return import_module(".u1_ym", __name__).U1YangMills
    if name == "U1TimeSlabDDTheory":
        return import_module(".u1_ym_dd", __name__).U1TimeSlabDDTheory
    if name == "SU3WilsonNf2":
        return import_module(".su3_wilson_nf2", __name__).SU3WilsonNf2
    if name == "SU2WilsonNf2":
        return import_module(".su2_wilson_nf2", __name__).SU2WilsonNf2
    if name == "U1WilsonNf2":
        return import_module(".u1_wilson_nf2", __name__).U1WilsonNf2
    if name == "U1WilsonNf2DDReference":
        return import_module(".u1_wilson_nf2_dd", __name__).U1WilsonNf2DDReference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ONSigmaModel",
    "Phi4",
    "init_mgflow",
    "mgflow_g",
    "mgflow_f",
    "mgflow_log_prob",
    "mgflow_prior_sample",
    "mgflow_prior_log_prob",
    "init_rg_cond_flow",
    "rg_cond_flow_g",
    "rg_cond_flow_f",
    "rg_cond_flow_log_prob",
    "rg_cond_flow_prior_sample",
    "rg_cond_flow_prior_log_prob",
    "init_rg_checkerboard_flow",
    "rg_checkerboard_flow_g",
    "rg_checkerboard_flow_f",
    "rg_checkerboard_flow_log_prob",
    "rg_checkerboard_flow_prior_sample",
    "rg_checkerboard_flow_prior_log_prob",
    "init_rg_checkerboard_inblock_flow",
    "rg_checkerboard_inblock_flow_g",
    "rg_checkerboard_inblock_flow_f",
    "rg_checkerboard_inblock_flow_log_prob",
    "rg_checkerboard_inblock_flow_prior_sample",
    "rg_checkerboard_inblock_flow_prior_log_prob",
    "init_rg_coarse_eta_flow",
    "rg_coarse_eta_flow_g",
    "rg_coarse_eta_flow_f",
    "rg_coarse_eta_flow_log_prob",
    "rg_coarse_eta_flow_prior_sample",
    "rg_coarse_eta_flow_prior_log_prob",
    "init_stacked_mg",
    "stacked_g",
    "stacked_f",
    "stacked_log_prob",
    "stacked_prior_sample",
    "SU3YangMills",
    "SU2YangMills",
    "U1YangMills",
    "U1TimeSlabDDTheory",
    "SU3WilsonNf2",
    "SU2WilsonNf2",
    "U1WilsonNf2",
    "U1WilsonNf2DDReference",
]

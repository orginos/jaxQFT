"""Stacked MGFlow utilities (JAX functional form)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from .phi4_mg import (
    init_mgflow,
    mgflow_f,
    mgflow_g,
    mgflow_log_prob_from,
    mgflow_prior_log_prob,
    mgflow_prior_sample_from,
)


Array = jax.Array


def init_stacked_mg(
    key: Array,
    stages: int,
    size: Tuple[int, int],
    n_layers: int = 3,
    width: int = 256,
    nconvs: int = 1,
    rg_type: str = "average",
    log_scale_clip: float = 5.0,
    parity: str = "none",
    fixed_bijector: bool = False,
) -> Dict:
    keys = jax.random.split(key, stages)
    stage_models = [
        init_mgflow(
            k,
            size=size,
            n_layers=n_layers,
            width=width,
            nconvs=nconvs,
            rg_type=rg_type,
            log_scale_clip=log_scale_clip,
            parity=parity,
            fixed_bijector=fixed_bijector,
        )
        for k in keys
    ]
    return {
        "cfg": {"size": size, "stage_cfgs": [m["cfg"] for m in stage_models]},
        "weights": {"stages": [m["weights"] for m in stage_models]},
    }


def stacked_prior_sample(key: Array, model_or_cfg: Dict, batch_size: int, dtype=jnp.float32, weights: Dict | None = None) -> Array:
    if weights is None:
        if "cfg" in model_or_cfg:
            cfg = model_or_cfg["cfg"]
        elif "stage_cfgs" in model_or_cfg:
            cfg = model_or_cfg
        else:
            raise ValueError("Expected model dict with 'cfg' or cfg dict with 'stage_cfgs'")
    else:
        cfg = model_or_cfg
    return mgflow_prior_sample_from(key, cfg["stage_cfgs"][0], batch_size, dtype=dtype)


def stacked_prior_log_prob(z: Array) -> Array:
    return mgflow_prior_log_prob(z)


def stacked_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    if weights is None:
        cfg = model_or_cfg["cfg"]
        weights = model_or_cfg["weights"]
    else:
        cfg = model_or_cfg
    x = z
    for scfg, sw in zip(cfg["stage_cfgs"], weights["stages"]):
        x = mgflow_g(scfg, x, sw)
    return x


def stacked_f(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Tuple[Array, Array]:
    if weights is None:
        cfg = model_or_cfg["cfg"]
        weights = model_or_cfg["weights"]
    else:
        cfg = model_or_cfg
    z = x
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    for scfg, sw in zip(reversed(cfg["stage_cfgs"]), reversed(weights["stages"])):
        z, j = mgflow_f(scfg, z, sw)
        ldj = ldj + j
    return z, ldj


def stacked_log_prob(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Array:
    z, j = stacked_f(model_or_cfg, x, weights)
    return stacked_prior_log_prob(z) + j


def stacked_diff(model_or_cfg: Dict, x: Array, action_fn, weights: Dict | None = None) -> Array:
    return stacked_log_prob(model_or_cfg, x, weights) + action_fn(x)

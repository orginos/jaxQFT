"""RG coarse-lattice fluctuation flow with learned Gaussian priors for phi^4.

This branch keeps the successful coarse-eta architecture intact and adds
independent Gaussian whitening maps:

1. At each non-terminal RG level, the local fluctuation vector ``eta in R^3``
   first passes through a learned zero-mean Gaussian map.
2. The Gaussian map can be:
   - shared across all sites at a given RG level, or
   - conditioned on a local patch of coarse fields.
3. The terminal ``2x2`` block keeps the RealNVP map and then applies an
   unconditional learned zero-mean ``4``-variate Gaussian map.

The prior is still standard normal on the full latent lattice. The Gaussian
priors are implemented as exact linear triangular maps, so they contribute a
cheap exact log-Jacobian.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .phi4_mg import init_realnvp, realnvp_f, realnvp_g
from .phi4_rg_coarse_eta_flow import (
    _eta_flow_f,
    _eta_flow_g,
    _init_local_patch_mlp,
    _local_patch_mlp_apply,
    _make_checkerboard_mask,
    _triangular_affine_f,
    _triangular_affine_g,
)
from .phi4_rg_cond_flow import (
    _coarse_condition,
    _merge_rg,
    _split_rg,
    _std_normal_log_prob,
)


Array = jax.Array


def _tri3_from_raw(log_scale_clip: float, offdiag_clip: float, raw: Array) -> Tuple[Array, Array]:
    log_diag = jnp.tanh(raw[..., :3]) * log_scale_clip
    lower = jnp.tanh(raw[..., 3:6]) * offdiag_clip
    return log_diag, lower


def _tri4_from_raw(log_scale_clip: float, offdiag_clip: float, raw: Array) -> Tuple[Array, Array]:
    log_diag = jnp.tanh(raw[..., :4]) * log_scale_clip
    lower = jnp.tanh(raw[..., 4:10]) * offdiag_clip
    return log_diag, lower


def _triangular_linear_g3(z: Array, log_diag: Array, lower: Array) -> Array:
    shift = jnp.zeros_like(log_diag)
    return _triangular_affine_g(z, log_diag, lower, shift)


def _triangular_linear_f3(x: Array, log_diag: Array, lower: Array) -> Tuple[Array, Array]:
    shift = jnp.zeros_like(log_diag)
    return _triangular_affine_f(x, log_diag, lower, shift)


def _triangular_linear_g4(z: Array, log_diag: Array, lower: Array) -> Array:
    d1 = jnp.exp(log_diag[..., 0])
    d2 = jnp.exp(log_diag[..., 1])
    d3 = jnp.exp(log_diag[..., 2])
    d4 = jnp.exp(log_diag[..., 3])
    l21 = lower[..., 0]
    l31 = lower[..., 1]
    l32 = lower[..., 2]
    l41 = lower[..., 3]
    l42 = lower[..., 4]
    l43 = lower[..., 5]

    z1 = z[..., 0]
    z2 = z[..., 1]
    z3 = z[..., 2]
    z4 = z[..., 3]

    x1 = d1 * z1
    x2 = l21 * z1 + d2 * z2
    x3 = l31 * z1 + l32 * z2 + d3 * z3
    x4 = l41 * z1 + l42 * z2 + l43 * z3 + d4 * z4
    return jnp.stack([x1, x2, x3, x4], axis=-1)


def _triangular_linear_f4(x: Array, log_diag: Array, lower: Array) -> Tuple[Array, Array]:
    e1 = jnp.exp(-log_diag[..., 0])
    e2 = jnp.exp(-log_diag[..., 1])
    e3 = jnp.exp(-log_diag[..., 2])
    e4 = jnp.exp(-log_diag[..., 3])
    l21 = lower[..., 0]
    l31 = lower[..., 1]
    l32 = lower[..., 2]
    l41 = lower[..., 3]
    l42 = lower[..., 4]
    l43 = lower[..., 5]

    z1 = x[..., 0] * e1
    z2 = (x[..., 1] - l21 * z1) * e2
    z3 = (x[..., 2] - l31 * z1 - l32 * z2) * e3
    z4 = (x[..., 3] - l41 * z1 - l42 * z2 - l43 * z3) * e4
    z = jnp.stack([z1, z2, z3, z4], axis=-1)
    ldj = -(log_diag[..., 0] + log_diag[..., 1] + log_diag[..., 2] + log_diag[..., 3])
    return z, ldj


def _eta_gaussian_context(coarse: Array, rg_mode: int) -> Array:
    return _coarse_condition(coarse, rg_mode)[..., None]


def _eta_gaussian_params(cfg: Dict, weights: Dict, coarse: Array, level: int) -> Tuple[Array, Array]:
    mode = cfg["eta_gaussian"]
    bsz, ny, nx = coarse.shape
    dtype = coarse.dtype
    if mode == "none":
        zeros = jnp.zeros((bsz, ny, nx, 3), dtype=dtype)
        return zeros, zeros
    if mode == "level":
        raw = weights["gaussian_levels"][level]["raw"]
        log_diag, lower = _tri3_from_raw(cfg["log_scale_clip"], cfg["offdiag_clip"], raw)
        log_diag = jnp.broadcast_to(log_diag[None, None, None, :], (bsz, ny, nx, 3))
        lower = jnp.broadcast_to(lower[None, None, None, :], (bsz, ny, nx, 3))
        return log_diag, lower
    if mode == "coarse_patch":
        context = _eta_gaussian_context(coarse, cfg["rg_mode"])
        conditioner_cfg = cfg["gaussian_conditioner_cfg"]
        coupling = weights["gaussian_levels"][level]
        if cfg["parity"] == "sym":
            raw_pos = _local_patch_mlp_apply(conditioner_cfg, coupling, context)
            raw_neg = _local_patch_mlp_apply(conditioner_cfg, coupling, -context)
            raw = 0.5 * (raw_pos + raw_neg)
        else:
            raw = _local_patch_mlp_apply(conditioner_cfg, coupling, context)
        return _tri3_from_raw(cfg["log_scale_clip"], cfg["offdiag_clip"], raw)
    raise ValueError(f"Unsupported eta_gaussian mode: {mode}")


def _eta_gaussian_g(cfg: Dict, weights: Dict, z: Array, coarse: Array, level: int) -> Array:
    log_diag, lower = _eta_gaussian_params(cfg, weights, coarse, level)
    return _triangular_linear_g3(z, log_diag, lower)


def _eta_gaussian_f(cfg: Dict, weights: Dict, x: Array, coarse: Array, level: int) -> Tuple[Array, Array]:
    log_diag, lower = _eta_gaussian_params(cfg, weights, coarse, level)
    z, ldj_site = _triangular_linear_f3(x, log_diag, lower)
    ldj = jnp.sum(ldj_site, axis=(1, 2))
    return z, ldj


def _terminal_gaussian_params(cfg: Dict, weights: Dict) -> Tuple[Array, Array]:
    mode = cfg["terminal_prior"]
    if mode == "std":
        zeros4 = jnp.zeros((4,), dtype=jnp.float32)
        zeros6 = jnp.zeros((6,), dtype=jnp.float32)
        return zeros4, zeros6
    if mode == "learned":
        raw = weights["terminal_gaussian"]["raw"]
        return _tri4_from_raw(cfg["log_scale_clip"], cfg["offdiag_clip"], raw)
    raise ValueError(f"Unsupported terminal prior mode: {mode}")


def _terminal_gaussian_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    flat = z.reshape((z.shape[0], 4))
    log_diag, lower = _terminal_gaussian_params(cfg, weights)
    out = _triangular_linear_g4(flat, log_diag, lower)
    return out.reshape(z.shape)


def _terminal_gaussian_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    flat = x.reshape((x.shape[0], 4))
    log_diag, lower = _terminal_gaussian_params(cfg, weights)
    z, ldj = _triangular_linear_f4(flat, log_diag, lower)
    return z.reshape(x.shape), ldj


def _terminal_flow_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    flat = z.reshape((z.shape[0], 4))
    flat = _triangular_linear_g4(flat, *_terminal_gaussian_params(cfg, weights))
    out = realnvp_g(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return out.reshape(z.shape)


def _terminal_flow_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    flat = x.reshape((x.shape[0], 4))
    z_mid, ldj_flow = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    z, ldj_gauss = _triangular_linear_f4(z_mid, *_terminal_gaussian_params(cfg, weights))
    return z.reshape(x.shape), ldj_flow + ldj_gauss


def _apply_small_output_init(weights: Dict, key: Array, scale: float, eta_gaussian: str) -> Dict:
    level_weights = []
    kcur = key
    for level in weights["levels"]:
        cycles = []
        for cycle_weights in level:
            cw = {}
            for color in ("red", "black"):
                coupling = dict(cycle_weights[color])
                coupling["l3"] = dict(coupling["l3"])
                kcur, kc = jax.random.split(kcur)
                coupling["l3"]["W"] = scale * jax.random.normal(
                    kc, coupling["l3"]["W"].shape, dtype=coupling["l3"]["W"].dtype
                )
                cw[color] = coupling
            cycles.append(cw)
        level_weights.append(cycles)

    if eta_gaussian == "coarse_patch":
        gaussian_levels = []
        for coupling in weights["gaussian_levels"]:
            gc = dict(coupling)
            gc["l3"] = dict(gc["l3"])
            kcur, kg = jax.random.split(kcur)
            gc["l3"]["W"] = scale * jax.random.normal(kg, gc["l3"]["W"].shape, dtype=gc["l3"]["W"].dtype)
            gaussian_levels.append(gc)
    else:
        gaussian_levels = list(weights["gaussian_levels"])

    terminal_flow = dict(weights["terminal_flow"])
    st = dict(terminal_flow["st"])
    nets = []
    for net in st["nets"]:
        n = dict(net)
        n["l3"] = dict(n["l3"])
        kcur, kn = jax.random.split(kcur)
        n["l3"]["W"] = scale * jax.random.normal(kn, n["l3"]["W"].shape, dtype=n["l3"]["W"].dtype)
        nets.append(n)
    st["nets"] = nets
    terminal_flow["st"] = st
    return {
        "levels": level_weights,
        "gaussian_levels": gaussian_levels,
        "terminal_flow": terminal_flow,
        "terminal_gaussian": weights["terminal_gaussian"],
    }


def init_rg_coarse_eta_gaussian_flow(
    key: Array,
    size: Tuple[int, int],
    *,
    width: int = 64,
    n_cycles: int = 2,
    radius: int = 1,
    eta_gaussian: str = "coarse_patch",
    gaussian_radius: int | None = None,
    gaussian_width: int | None = None,
    terminal_prior: str = "learned",
    rg_type: str = "average",
    log_scale_clip: float = 5.0,
    offdiag_clip: float = 2.0,
    terminal_n_layers: int = 2,
    terminal_width: int | None = None,
    output_init_scale: float = 1e-2,
    parity: str = "sym",
) -> Dict:
    h, w = size
    if h != w or (h & (h - 1)) != 0:
        raise ValueError("size must be square power-of-2")
    if h < 2:
        raise ValueError("size must be at least 2x2")
    if int(n_cycles) < 1:
        raise ValueError("n_cycles must be positive")
    if int(radius) < 0:
        raise ValueError("radius must be nonnegative")
    if eta_gaussian not in ("none", "level", "coarse_patch"):
        raise ValueError("eta_gaussian must be one of {'none','level','coarse_patch'}")
    if terminal_prior not in ("std", "learned"):
        raise ValueError("terminal_prior must be one of {'std','learned'}")
    if parity not in ("none", "sym"):
        raise ValueError("parity must be one of {'none','sym'}")

    depth = int(math.log2(h))
    rg_mode = 0 if rg_type == "average" else 1
    terminal_width = int(terminal_width if terminal_width is not None else width)
    gaussian_radius = int(radius if gaussian_radius is None else gaussian_radius)
    gaussian_width = int(width if gaussian_width is None else gaussian_width)
    if gaussian_radius < 0:
        raise ValueError("gaussian_radius must be nonnegative")

    site_red = []
    site_black = []
    for level in range(max(depth - 1, 0)):
        ny = h // (2 ** (level + 1))
        nx = w // (2 ** (level + 1))
        red = _make_checkerboard_mask(ny, nx)
        site_red.append(red)
        site_black.append(1.0 - red)

    conditioner = _init_local_patch_mlp(key, in_channels=4, hidden_dim=width, out_dim=9, radius=radius)
    conditioner_cfg = conditioner["cfg"]

    gaussian_conditioner_cfg = None
    if eta_gaussian == "coarse_patch":
        gaussian_conditioner_cfg = _init_local_patch_mlp(
            key,
            in_channels=1,
            hidden_dim=gaussian_width,
            out_dim=6,
            radius=gaussian_radius,
        )["cfg"]

    level_weights = []
    gaussian_level_weights = []
    for _ in range(max(depth - 1, 0)):
        keys = jax.random.split(key, 2 * n_cycles + 2)
        key = keys[0]
        cycles = []
        idx = 1
        for _cycle in range(n_cycles):
            red_net = _init_local_patch_mlp(keys[idx], in_channels=4, hidden_dim=width, out_dim=9, radius=radius)["weights"]
            idx += 1
            black_net = _init_local_patch_mlp(keys[idx], in_channels=4, hidden_dim=width, out_dim=9, radius=radius)["weights"]
            idx += 1
            cycles.append({"red": red_net, "black": black_net})
        level_weights.append(cycles)

        if eta_gaussian == "none":
            gaussian_level_weights.append({"raw": jnp.zeros((6,), dtype=jnp.float32)})
        elif eta_gaussian == "level":
            gaussian_level_weights.append({"raw": jnp.zeros((6,), dtype=jnp.float32)})
        else:
            gaussian_level_weights.append(
                _init_local_patch_mlp(keys[idx], in_channels=1, hidden_dim=gaussian_width, out_dim=6, radius=gaussian_radius)[
                    "weights"
                ]
            )

    key, k_terminal = jax.random.split(key)
    terminal = init_realnvp(
        k_terminal,
        n_layers=int(terminal_n_layers),
        width=terminal_width,
        log_scale_clip=log_scale_clip,
        parity=parity,
    )

    cfg = {
        "size_h": int(h),
        "size_w": int(w),
        "depth": int(depth),
        "rg_mode": int(rg_mode),
        "rg_type": str(rg_type),
        "parity": str(parity),
        "n_cycles": int(n_cycles),
        "radius": int(radius),
        "eta_gaussian": str(eta_gaussian),
        "gaussian_radius": int(gaussian_radius),
        "gaussian_width": int(gaussian_width),
        "terminal_prior": str(terminal_prior),
        "log_scale_clip": float(log_scale_clip),
        "offdiag_clip": float(offdiag_clip),
        "site_red": tuple(site_red),
        "site_black": tuple(site_black),
        "conditioner_cfg": conditioner_cfg,
        "gaussian_conditioner_cfg": gaussian_conditioner_cfg,
        "terminal_flow_cfg": terminal["cfg"],
        "arch": {
            "width": int(width),
            "n_cycles": int(n_cycles),
            "radius": int(radius),
            "eta_gaussian": str(eta_gaussian),
            "gaussian_radius": int(gaussian_radius),
            "gaussian_width": int(gaussian_width),
            "terminal_prior": str(terminal_prior),
            "rg_type": str(rg_type),
            "terminal_n_layers": int(terminal_n_layers),
            "terminal_width": int(terminal_width),
            "output_init_scale": float(output_init_scale),
            "parity": str(parity),
            "offdiag_clip": float(offdiag_clip),
        },
    }
    weights = {
        "levels": level_weights,
        "gaussian_levels": gaussian_level_weights,
        "terminal_flow": terminal["weights"],
        "terminal_gaussian": {"raw": jnp.zeros((10,), dtype=jnp.float32)},
    }
    if float(output_init_scale) > 0.0:
        key, k_patch = jax.random.split(key)
        weights = _apply_small_output_init(weights, k_patch, scale=float(output_init_scale), eta_gaussian=eta_gaussian)
    return {"cfg": cfg, "weights": weights}


def _rg_coarse_eta_gaussian_g_level(cfg: Dict, weights: Dict, z: Array, level: int) -> Array:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_g(cfg, weights, z)
    z_coarse, z_fine = _split_rg(z, cfg["rg_mode"])
    x_coarse = _rg_coarse_eta_gaussian_g_level(cfg, weights, z_coarse, level + 1)
    z_fine = _eta_gaussian_g(cfg, weights, z_fine, x_coarse, level)
    x_fine = _eta_flow_g(cfg, weights, z_fine, x_coarse, level)
    return _merge_rg(x_coarse, x_fine, cfg["rg_mode"])


def _rg_coarse_eta_gaussian_f_level(cfg: Dict, weights: Dict, x: Array, level: int) -> Tuple[Array, Array]:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_f(cfg, weights, x)
    x_coarse, x_fine = _split_rg(x, cfg["rg_mode"])
    z_coarse, ldj_coarse = _rg_coarse_eta_gaussian_f_level(cfg, weights, x_coarse, level + 1)
    z_fine, ldj_fine = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
    z_fine, ldj_gauss = _eta_gaussian_f(cfg, weights, z_fine, x_coarse, level)
    return _merge_rg(z_coarse, z_fine, cfg["rg_mode"]), ldj_coarse + ldj_fine + ldj_gauss


def rg_coarse_eta_gaussian_flow_g_from(cfg: Dict, weights: Dict, z: Array) -> Array:
    return _rg_coarse_eta_gaussian_g_level(cfg, weights, z, 0)


def rg_coarse_eta_gaussian_flow_f_from(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    return _rg_coarse_eta_gaussian_f_level(cfg, weights, x, 0)


def rg_coarse_eta_gaussian_flow_prior_sample_from(key: Array, cfg: Dict, batch_size: int, dtype=jnp.float32) -> Array:
    return jax.random.normal(key, (batch_size, cfg["size_h"], cfg["size_w"]), dtype=dtype)


def rg_coarse_eta_gaussian_flow_prior_log_prob(z: Array) -> Array:
    return _std_normal_log_prob(z, sum_axes=(1, 2))


def rg_coarse_eta_gaussian_flow_log_prob_from(cfg: Dict, weights: Dict, x: Array) -> Array:
    z, ldj = rg_coarse_eta_gaussian_flow_f_from(cfg, weights, x)
    return rg_coarse_eta_gaussian_flow_prior_log_prob(z) + ldj


def _split_model(model_or_cfg: Dict, maybe_weights: Dict | None = None):
    if maybe_weights is not None:
        return model_or_cfg, maybe_weights
    if "size_h" in model_or_cfg and "size_w" in model_or_cfg:
        return model_or_cfg, None
    if "cfg" in model_or_cfg and "weights" in model_or_cfg:
        return model_or_cfg["cfg"], model_or_cfg["weights"]
    raise ValueError("Expected either (cfg, weights) or {'cfg':..., 'weights':...}")


def rg_coarse_eta_gaussian_flow_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_gaussian_flow_g_from(cfg, w, z)


def rg_coarse_eta_gaussian_flow_f(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Tuple[Array, Array]:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_gaussian_flow_f_from(cfg, w, x)


def rg_coarse_eta_gaussian_flow_prior_sample(
    key: Array,
    model_or_cfg: Dict,
    batch_size: int,
    dtype=jnp.float32,
    weights: Dict | None = None,
) -> Array:
    cfg, _ = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_gaussian_flow_prior_sample_from(key, cfg, batch_size, dtype=dtype)


def rg_coarse_eta_gaussian_flow_log_prob(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_gaussian_flow_log_prob_from(cfg, w, x)

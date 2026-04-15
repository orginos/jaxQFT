"""RG coarse-lattice fluctuation flow for scalar phi^4.

This branch separates RG blocking from fluctuation trivialization.
At each non-terminal RG level:

1. Split the current fine field into a coarse scalar field ``c`` on the next
   lattice and a local fluctuation vector ``eta in R^3`` on each coarse site.
2. Keep ``c`` fixed throughout the level update.
3. Trivialize ``eta`` using red/black sweeps on the coarse lattice.
4. Parameterize each updated-site map with a local neighborhood MLP that sees
   the fixed coarse field and the neighboring frozen fluctuations.

The per-site eta map is a conditional lower-triangular affine transformation,
which mixes the three local fluctuation components in one shot while retaining
an exact, cheap log-Jacobian.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp

from .phi4_mg import init_realnvp, realnvp_f, realnvp_g, realnvp_g_with_ldj
from .phi4_rg_cond_flow import (
    _coarse_condition,
    _init_mlp,
    _make_local_offsets,
    _merge_rg,
    _mlp_apply,
    _roll_stack,
    _split_rg,
    _std_normal_log_prob,
)


Array = jax.Array


def _make_checkerboard_mask(ny: int, nx: int, dtype=jnp.float32) -> Array:
    yy = jnp.arange(ny, dtype=jnp.int32)[:, None]
    xx = jnp.arange(nx, dtype=jnp.int32)[None, :]
    return ((yy + xx) % 2 == 0).astype(dtype)[None, :, :, None]


def _init_local_patch_mlp(key: Array, in_channels: int, hidden_dim: int, out_dim: int, radius: int) -> Dict:
    offsets = _make_local_offsets(radius)
    return {
        "cfg": {
            "in_channels": int(in_channels),
            "radius": int(radius),
            "offsets": tuple(offsets),
        },
        "weights": _init_mlp(
            key,
            in_dim=int(in_channels * len(offsets)),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            final_scale=0.0,
        ),
    }


def _local_patch_mlp_apply(cfg: Dict, weights: Dict, x: Array) -> Array:
    bsz, ny, nx, nch = x.shape
    patches = _roll_stack(x, cfg["offsets"]).reshape((bsz, ny, nx, len(cfg["offsets"]) * nch))
    return _mlp_apply(weights, patches)


def _eta_context(eta_masked: Array, coarse: Array, rg_mode: int) -> Array:
    coarse_ctx = _coarse_condition(coarse, rg_mode)[..., None]
    return jnp.concatenate([eta_masked, coarse_ctx], axis=-1)


def _conditioner_cfg_for_level(cfg: Dict, level: int) -> Dict:
    levels = cfg.get("conditioner_cfg_levels")
    if levels is not None and len(levels) > 0:
        return levels[level]
    return cfg["conditioner_cfg"]


def _conditional_params(cfg: Dict, coupling: Dict, context: Array, level: int) -> Tuple[Array, Array, Array]:
    conditioner_cfg = _conditioner_cfg_for_level(cfg, level)
    parity = cfg["parity"]
    if parity == "none":
        raw = _local_patch_mlp_apply(conditioner_cfg, coupling, context)
        even_raw = raw[..., :6]
        shift = raw[..., 6:]
    elif parity == "sym":
        raw_pos = _local_patch_mlp_apply(conditioner_cfg, coupling, context)
        raw_neg = _local_patch_mlp_apply(conditioner_cfg, coupling, -context)
        even_raw = 0.5 * (raw_pos[..., :6] + raw_neg[..., :6])
        shift = 0.5 * (raw_pos[..., 6:] - raw_neg[..., 6:])
    else:
        raise ValueError(f"Unsupported parity mode for RG coarse-eta flow: {parity}")

    log_diag = jnp.tanh(even_raw[..., :3]) * cfg["log_scale_clip"]
    lower = jnp.tanh(even_raw[..., 3:6]) * cfg["offdiag_clip"]
    return log_diag, lower, shift


def _triangular_affine_g(eta: Array, log_diag: Array, lower: Array, shift: Array) -> Array:
    d1 = jnp.exp(log_diag[..., 0])
    d2 = jnp.exp(log_diag[..., 1])
    d3 = jnp.exp(log_diag[..., 2])
    l21 = lower[..., 0]
    l31 = lower[..., 1]
    l32 = lower[..., 2]

    z1 = eta[..., 0]
    z2 = eta[..., 1]
    z3 = eta[..., 2]

    x1 = d1 * z1 + shift[..., 0]
    x2 = l21 * z1 + d2 * z2 + shift[..., 1]
    x3 = l31 * z1 + l32 * z2 + d3 * z3 + shift[..., 2]
    return jnp.stack([x1, x2, x3], axis=-1)


def _triangular_affine_g_with_ldj(
    eta: Array, log_diag: Array, lower: Array, shift: Array
) -> Tuple[Array, Array]:
    x = _triangular_affine_g(eta, log_diag, lower, shift)
    ldj_site = log_diag[..., 0] + log_diag[..., 1] + log_diag[..., 2]
    return x, ldj_site


def _triangular_affine_f(eta: Array, log_diag: Array, lower: Array, shift: Array) -> Tuple[Array, Array]:
    e1 = jnp.exp(-log_diag[..., 0])
    e2 = jnp.exp(-log_diag[..., 1])
    e3 = jnp.exp(-log_diag[..., 2])
    l21 = lower[..., 0]
    l31 = lower[..., 1]
    l32 = lower[..., 2]

    x1 = eta[..., 0] - shift[..., 0]
    z1 = x1 * e1

    x2 = eta[..., 1] - shift[..., 1] - l21 * z1
    z2 = x2 * e2

    x3 = eta[..., 2] - shift[..., 2] - l31 * z1 - l32 * z2
    z3 = x3 * e3

    z = jnp.stack([z1, z2, z3], axis=-1)
    ldj_site = -(log_diag[..., 0] + log_diag[..., 1] + log_diag[..., 2])
    return z, ldj_site


def _site_masks(cfg: Dict, level: int) -> Tuple[Array, Array]:
    return cfg["site_red"][level], cfg["site_black"][level]


def _color_sweep_g(cfg: Dict, cycle_weights: Dict, eta: Array, coarse: Array, level: int, color: str) -> Array:
    red_mask, black_mask = _site_masks(cfg, level)
    target_mask = red_mask if color == "red" else black_mask
    eta_masked = eta * (1.0 - target_mask)
    context = _eta_context(eta_masked, coarse, cfg["rg_mode"])
    log_diag, lower, shift = _conditional_params(cfg, cycle_weights[color], context, level)
    eta_new = _triangular_affine_g(eta, log_diag, lower, shift)
    return eta_masked + target_mask * eta_new


def _color_sweep_g_with_ldj(
    cfg: Dict, cycle_weights: Dict, eta: Array, coarse: Array, level: int, color: str
) -> Tuple[Array, Array]:
    red_mask, black_mask = _site_masks(cfg, level)
    target_mask = red_mask if color == "red" else black_mask
    eta_masked = eta * (1.0 - target_mask)
    context = _eta_context(eta_masked, coarse, cfg["rg_mode"])
    log_diag, lower, shift = _conditional_params(cfg, cycle_weights[color], context, level)
    eta_new, ldj_site = _triangular_affine_g_with_ldj(eta, log_diag, lower, shift)
    eta_out = eta_masked + target_mask * eta_new
    ldj = jnp.sum(ldj_site * target_mask[..., 0], axis=(1, 2))
    return eta_out, ldj


def _color_sweep_f(
    cfg: Dict, cycle_weights: Dict, eta: Array, coarse: Array, level: int, color: str
) -> Tuple[Array, Array]:
    red_mask, black_mask = _site_masks(cfg, level)
    target_mask = red_mask if color == "red" else black_mask
    eta_masked = eta * (1.0 - target_mask)
    context = _eta_context(eta_masked, coarse, cfg["rg_mode"])
    log_diag, lower, shift = _conditional_params(cfg, cycle_weights[color], context, level)
    eta_new, ldj_site = _triangular_affine_f(eta, log_diag, lower, shift)
    eta = eta_masked + target_mask * eta_new
    ldj = jnp.sum(ldj_site * target_mask[..., 0], axis=(1, 2))
    return eta, ldj


def _eta_flow_g(cfg: Dict, weights: Dict, eta: Array, coarse: Array, level: int) -> Array:
    x = eta
    for cycle_weights in weights["levels"][level]:
        x = _color_sweep_g(cfg, cycle_weights, x, coarse, level, "red")
        x = _color_sweep_g(cfg, cycle_weights, x, coarse, level, "black")
    return x


def _eta_flow_g_with_ldj(cfg: Dict, weights: Dict, eta: Array, coarse: Array, level: int) -> Tuple[Array, Array]:
    x = eta
    ldj = jnp.zeros((eta.shape[0],), dtype=eta.dtype)
    for cycle_weights in weights["levels"][level]:
        x, ld_red = _color_sweep_g_with_ldj(cfg, cycle_weights, x, coarse, level, "red")
        x, ld_black = _color_sweep_g_with_ldj(cfg, cycle_weights, x, coarse, level, "black")
        ldj = ldj + ld_red + ld_black
    return x, ldj


def _eta_flow_f(cfg: Dict, weights: Dict, eta: Array, coarse: Array, level: int) -> Tuple[Array, Array]:
    z = eta
    ldj = jnp.zeros((eta.shape[0],), dtype=eta.dtype)
    for cycle_idx in range(len(weights["levels"][level]) - 1, -1, -1):
        cycle_weights = weights["levels"][level][cycle_idx]
        z, ld_black = _color_sweep_f(cfg, cycle_weights, z, coarse, level, "black")
        z, ld_red = _color_sweep_f(cfg, cycle_weights, z, coarse, level, "red")
        ldj = ldj + ld_black + ld_red
    return z, ldj


def _terminal_flow_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    flat = z.reshape((z.shape[0], 4))
    out = realnvp_g(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return out.reshape(z.shape)


def _terminal_flow_g_with_ldj(cfg: Dict, weights: Dict, z: Array) -> Tuple[Array, Array]:
    flat = z.reshape((z.shape[0], 4))
    out, ldj = realnvp_g_with_ldj(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return out.reshape(z.shape), ldj


def _terminal_flow_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    flat = x.reshape((x.shape[0], 4))
    z, ldj = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return z.reshape(x.shape), ldj


def _apply_small_output_init(weights: Dict, key: Array, scale: float) -> Dict:
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
    return {"levels": level_weights, "terminal_flow": terminal_flow}


def init_rg_coarse_eta_flow(
    key: Array,
    size: Tuple[int, int],
    *,
    width: int = 64,
    n_cycles: int = 2,
    radius: int = 1,
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
    if parity not in ("none", "sym"):
        raise ValueError("parity must be one of {'none','sym'}")

    depth = int(math.log2(h))
    rg_mode = 0 if rg_type == "average" else 1
    terminal_width = int(terminal_width if terminal_width is not None else width)

    site_red = []
    site_black = []
    for level in range(max(depth - 1, 0)):
        ny = h // (2 ** (level + 1))
        nx = w // (2 ** (level + 1))
        red = _make_checkerboard_mask(ny, nx)
        site_red.append(red)
        site_black.append(1.0 - red)

    conditioner = _init_local_patch_mlp(
        key,
        in_channels=4,
        hidden_dim=width,
        out_dim=9,
        radius=radius,
    )
    conditioner_cfg = conditioner["cfg"]

    level_weights = []
    for _ in range(max(depth - 1, 0)):
        keys = jax.random.split(key, 2 * n_cycles + 1)
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
        "log_scale_clip": float(log_scale_clip),
        "offdiag_clip": float(offdiag_clip),
        "site_red": tuple(site_red),
        "site_black": tuple(site_black),
        "conditioner_cfg": conditioner_cfg,
        "terminal_flow_cfg": terminal["cfg"],
        "arch": {
            "width": int(width),
            "n_cycles": int(n_cycles),
            "radius": int(radius),
            "rg_type": str(rg_type),
            "terminal_n_layers": int(terminal_n_layers),
            "terminal_width": int(terminal_width),
            "output_init_scale": float(output_init_scale),
            "parity": str(parity),
            "offdiag_clip": float(offdiag_clip),
        },
    }
    weights = {"levels": level_weights, "terminal_flow": terminal["weights"]}
    if float(output_init_scale) > 0.0:
        key, k_patch = jax.random.split(key)
        weights = _apply_small_output_init(weights, k_patch, scale=float(output_init_scale))
    return {"cfg": cfg, "weights": weights}


def _rg_coarse_eta_g_level(cfg: Dict, weights: Dict, z: Array, level: int) -> Array:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_g(cfg, weights, z)
    z_coarse, z_fine = _split_rg(z, cfg["rg_mode"])
    x_coarse = _rg_coarse_eta_g_level(cfg, weights, z_coarse, level + 1)
    x_fine = _eta_flow_g(cfg, weights, z_fine, x_coarse, level)
    return _merge_rg(x_coarse, x_fine, cfg["rg_mode"])


def _rg_coarse_eta_f_level(cfg: Dict, weights: Dict, x: Array, level: int) -> Tuple[Array, Array]:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_f(cfg, weights, x)
    x_coarse, x_fine = _split_rg(x, cfg["rg_mode"])
    z_coarse, ldj_coarse = _rg_coarse_eta_f_level(cfg, weights, x_coarse, level + 1)
    z_fine, ldj_fine = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
    return _merge_rg(z_coarse, z_fine, cfg["rg_mode"]), ldj_coarse + ldj_fine


def rg_coarse_eta_flow_g_from(cfg: Dict, weights: Dict, z: Array) -> Array:
    return _rg_coarse_eta_g_level(cfg, weights, z, 0)


def rg_coarse_eta_flow_f_from(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    return _rg_coarse_eta_f_level(cfg, weights, x, 0)


def rg_coarse_eta_flow_prior_sample_from(key: Array, cfg: Dict, batch_size: int, dtype=jnp.float32) -> Array:
    return jax.random.normal(key, (batch_size, cfg["size_h"], cfg["size_w"]), dtype=dtype)


def rg_coarse_eta_flow_prior_log_prob(z: Array) -> Array:
    return _std_normal_log_prob(z, sum_axes=(1, 2))


def rg_coarse_eta_flow_log_prob_from(cfg: Dict, weights: Dict, x: Array) -> Array:
    z, ldj = rg_coarse_eta_flow_f_from(cfg, weights, x)
    return rg_coarse_eta_flow_prior_log_prob(z) + ldj


def _split_model(model_or_cfg: Dict, maybe_weights: Dict | None = None):
    if maybe_weights is not None:
        return model_or_cfg, maybe_weights
    if "size_h" in model_or_cfg and "size_w" in model_or_cfg:
        return model_or_cfg, None
    if "cfg" in model_or_cfg and "weights" in model_or_cfg:
        return model_or_cfg["cfg"], model_or_cfg["weights"]
    raise ValueError("Expected either (cfg, weights) or {'cfg':..., 'weights':...}")


def rg_coarse_eta_flow_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_flow_g_from(cfg, w, z)


def rg_coarse_eta_flow_f(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Tuple[Array, Array]:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_flow_f_from(cfg, w, x)


def rg_coarse_eta_flow_prior_sample(
    key: Array,
    model_or_cfg: Dict,
    batch_size: int,
    dtype=jnp.float32,
    weights: Dict | None = None,
) -> Array:
    cfg, _ = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_flow_prior_sample_from(key, cfg, batch_size, dtype=dtype)


def rg_coarse_eta_flow_log_prob(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_coarse_eta_flow_log_prob_from(cfg, w, x)

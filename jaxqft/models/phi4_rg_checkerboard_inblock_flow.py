"""RG checkerboard flow with in-block eta mixing for scalar phi^4.

This is a new evolution branch beyond ``phi4_rg_checkerboard_flow``.
It keeps the red/black plus shifted-blocking cycle, but each red or black
subpass itself performs RealNVP-style mixing among the three local block
fluctuations using the familiar in-block masks ``(1,1,0)``, ``(0,1,1)``,
``(1,0,1)``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .phi4_mg import init_realnvp, realnvp_f, realnvp_g
from .phi4_rg_checkerboard_flow import (
    _init_local_mlp_conditioner,
    _local_mlp_conditioner_apply,
)
from .phi4_rg_cond_flow import (
    _coarse_condition,
    _init_local_transformer,
    _local_transformer_apply,
    _make_masks_3d,
    _merge_rg,
    _split_rg,
    _std_normal_log_prob,
)


Array = jax.Array


def _make_checkerboard_mask(ny: int, nx: int, dtype=jnp.float32) -> Array:
    yy = jnp.arange(ny, dtype=jnp.int32)[:, None]
    xx = jnp.arange(nx, dtype=jnp.int32)[None, :]
    return ((yy + xx) % 2 == 0).astype(dtype)[None, :, :, None]


def _apply_conditioner(cfg: Dict, coupling: Dict, context: Array) -> Array:
    if cfg["conditioner_type"] == "transformer":
        return _local_transformer_apply(cfg["transformer_cfg"], coupling, context)
    if cfg["conditioner_type"] == "mlp":
        return _local_mlp_conditioner_apply(coupling, context)
    raise ValueError(f"Unsupported conditioner type: {cfg['conditioner_type']}")


def _conditional_st(cfg: Dict, coupling: Dict, context: Array) -> Tuple[Array, Array]:
    parity = cfg["parity"]
    if parity == "none":
        st = _apply_conditioner(cfg, coupling, context)
        return jnp.split(st, 2, axis=-1)
    if parity == "sym":
        st_pos = _apply_conditioner(cfg, coupling, context)
        st_neg = _apply_conditioner(cfg, coupling, -context)
        d = st_pos.shape[-1] // 2
        s = 0.5 * (st_pos[..., :d] + st_neg[..., :d])
        t = 0.5 * (st_pos[..., d:] - st_neg[..., d:])
        return s, t
    raise ValueError(f"Unsupported parity mode: {parity}")


def _masked_eta_context(eta_vis: Array, coarse: Array, rg_mode: int) -> Array:
    coarse_ctx = _coarse_condition(coarse, rg_mode)[..., None]
    return jnp.concatenate([eta_vis, coarse_ctx], axis=-1)


def _apply_small_output_init_checkerboard_inblock(
    weights: Dict, key: Array, scale: float, conditioner_type: str
) -> Dict:
    level_weights = []
    kcur = key
    for level in weights["levels"]:
        passes = []
        for pass_weights in level:
            pw = {}
            for color in ("red", "black"):
                couplings = []
                for coupling0 in pass_weights[color]:
                    coupling = dict(coupling0)
                    if conditioner_type == "transformer":
                        coupling["head"] = dict(coupling["head"])
                        coupling["head"]["l3"] = dict(coupling["head"]["l3"])
                        kcur, k1 = jax.random.split(kcur)
                        coupling["head"]["l3"]["W"] = scale * jax.random.normal(
                            k1, coupling["head"]["l3"]["W"].shape, dtype=coupling["head"]["l3"]["W"].dtype
                        )
                        blocks = []
                        for blk in coupling["blocks"]:
                            b = dict(blk)
                            b["proj"] = dict(b["proj"])
                            kcur, kb1 = jax.random.split(kcur)
                            b["proj"]["W"] = scale * jax.random.normal(
                                kb1, b["proj"]["W"].shape, dtype=b["proj"]["W"].dtype
                            )
                            b["mlp"] = dict(b["mlp"])
                            b["mlp"]["l3"] = dict(b["mlp"]["l3"])
                            kcur, kb2 = jax.random.split(kcur)
                            b["mlp"]["l3"]["W"] = scale * jax.random.normal(
                                kb2, b["mlp"]["l3"]["W"].shape, dtype=b["mlp"]["l3"]["W"].dtype
                            )
                            blocks.append(b)
                        coupling["blocks"] = blocks
                    else:
                        coupling["l3"] = dict(coupling["l3"])
                        kcur, k1 = jax.random.split(kcur)
                        coupling["l3"]["W"] = scale * jax.random.normal(
                            k1, coupling["l3"]["W"].shape, dtype=coupling["l3"]["W"].dtype
                        )
                    couplings.append(coupling)
                pw[color] = couplings
            passes.append(pw)
        level_weights.append(passes)

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


def _site_masks(cfg: Dict, level: int) -> Tuple[Array, Array]:
    return cfg["site_red"][level], cfg["site_black"][level]


def _color_pass_g(cfg: Dict, pass_weights: Dict, eta: Array, coarse: Array, level: int, color: str) -> Array:
    red_mask, black_mask = _site_masks(cfg, level)
    color_mask = red_mask if color == "red" else black_mask
    other_mask = black_mask if color == "red" else red_mask
    for i, coupling in enumerate(pass_weights[color]):
        inblock_mask = cfg["inblock_masks"][i].reshape((1, 1, 1, 3))
        inblock_inv = cfg["inblock_masks_inv"][i].reshape((1, 1, 1, 3))
        visible_mask = other_mask + color_mask * inblock_mask
        update_mask = color_mask * inblock_inv
        context = _masked_eta_context(eta * visible_mask, coarse, cfg["rg_mode"])
        s, t = _conditional_st(cfg, coupling, context)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * update_mask
        t = t * update_mask
        eta = eta * (1.0 - update_mask) + update_mask * (eta * jnp.exp(s) + t)
    return eta


def _color_pass_f(
    cfg: Dict, pass_weights: Dict, eta: Array, coarse: Array, level: int, color: str
) -> Tuple[Array, Array]:
    red_mask, black_mask = _site_masks(cfg, level)
    color_mask = red_mask if color == "red" else black_mask
    other_mask = black_mask if color == "red" else red_mask
    ldj = jnp.zeros((eta.shape[0],), dtype=eta.dtype)
    for i in range(len(pass_weights[color]) - 1, -1, -1):
        coupling = pass_weights[color][i]
        inblock_mask = cfg["inblock_masks"][i].reshape((1, 1, 1, 3))
        inblock_inv = cfg["inblock_masks_inv"][i].reshape((1, 1, 1, 3))
        visible_mask = other_mask + color_mask * inblock_mask
        update_mask = color_mask * inblock_inv
        context = _masked_eta_context(eta * visible_mask, coarse, cfg["rg_mode"])
        s, t = _conditional_st(cfg, coupling, context)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * update_mask
        t = t * update_mask
        eta = eta * (1.0 - update_mask) + update_mask * (eta - t) * jnp.exp(-s)
        ldj = ldj - jnp.sum(s, axis=(1, 2, 3))
    return eta, ldj


def _checkerboard_pass_g(cfg: Dict, weights: Dict, x: Array, level: int, pass_idx: int) -> Array:
    coarse, eta = _split_rg(x, cfg["rg_mode"])
    pass_weights = weights["levels"][level][pass_idx]
    eta = _color_pass_g(cfg, pass_weights, eta, coarse, level, "red")
    eta = _color_pass_g(cfg, pass_weights, eta, coarse, level, "black")
    return _merge_rg(coarse, eta, cfg["rg_mode"])


def _checkerboard_pass_f(cfg: Dict, weights: Dict, x: Array, level: int, pass_idx: int) -> Tuple[Array, Array]:
    coarse, eta = _split_rg(x, cfg["rg_mode"])
    pass_weights = weights["levels"][level][pass_idx]
    eta, ldj_black = _color_pass_f(cfg, pass_weights, eta, coarse, level, "black")
    eta, ldj_red = _color_pass_f(cfg, pass_weights, eta, coarse, level, "red")
    return _merge_rg(coarse, eta, cfg["rg_mode"]), ldj_black + ldj_red


def _level_flow_g(cfg: Dict, weights: Dict, z: Array, level: int) -> Array:
    x = z
    for cycle in range(cfg["n_cycles"]):
        x = _checkerboard_pass_g(cfg, weights, x, level, 2 * cycle)
        x = jnp.roll(x, shift=(-1, -1), axis=(1, 2))
        x = _checkerboard_pass_g(cfg, weights, x, level, 2 * cycle + 1)
        x = jnp.roll(x, shift=(1, 1), axis=(1, 2))
    return x


def _level_flow_f(cfg: Dict, weights: Dict, x: Array, level: int) -> Tuple[Array, Array]:
    z = x
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    for cycle in range(cfg["n_cycles"] - 1, -1, -1):
        z = jnp.roll(z, shift=(-1, -1), axis=(1, 2))
        z, ld1 = _checkerboard_pass_f(cfg, weights, z, level, 2 * cycle + 1)
        z = jnp.roll(z, shift=(1, 1), axis=(1, 2))
        z, ld0 = _checkerboard_pass_f(cfg, weights, z, level, 2 * cycle)
        ldj = ldj + ld1 + ld0
    return z, ldj


def _terminal_flow_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    flat = z.reshape((z.shape[0], 4))
    out = realnvp_g(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return out.reshape(z.shape)


def _terminal_flow_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    flat = x.reshape((x.shape[0], 4))
    z, ldj = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return z.reshape(x.shape), ldj


def init_rg_checkerboard_inblock_flow(
    key: Array,
    size: Tuple[int, int],
    *,
    width: int = 64,
    n_cycles: int = 1,
    n_inner_couplings: int = 3,
    conditioner: str = "transformer",
    transformer_layers: int = 2,
    n_heads: int = 4,
    attn_radius: int = 1,
    mlp_dim: int | None = None,
    rg_type: str = "average",
    log_scale_clip: float = 5.0,
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
    if int(n_inner_couplings) < 1:
        raise ValueError("n_inner_couplings must be positive")
    if parity not in ("none", "sym"):
        raise ValueError("parity must be one of {'none','sym'}")
    if conditioner not in ("transformer", "mlp"):
        raise ValueError("conditioner must be one of {'transformer','mlp'}")

    depth = int(math.log2(h))
    rg_mode = 0 if rg_type == "average" else 1
    mlp_dim = int(mlp_dim if mlp_dim is not None else 2 * width)
    terminal_width = int(terminal_width if terminal_width is not None else width)
    inblock_masks = _make_masks_3d(n_inner_couplings)

    site_red = []
    site_black = []
    for level in range(max(depth - 1, 0)):
        field_h = h // (2**level)
        field_w = w // (2**level)
        red = _make_checkerboard_mask(field_h // 2, field_w // 2)
        site_red.append(red)
        site_black.append(1.0 - red)

    cfg = {
        "size_h": int(h),
        "size_w": int(w),
        "depth": int(depth),
        "rg_mode": int(rg_mode),
        "parity": str(parity),
        "n_cycles": int(n_cycles),
        "n_inner_couplings": int(n_inner_couplings),
        "conditioner_type": str(conditioner),
        "log_scale_clip": float(log_scale_clip),
        "site_red": tuple(site_red),
        "site_black": tuple(site_black),
        "inblock_masks": inblock_masks,
        "inblock_masks_inv": 1.0 - inblock_masks,
    }

    transformer_cfg = None
    level_weights = []
    for _ in range(max(depth - 1, 0)):
        keys = jax.random.split(key, 4 * n_cycles * n_inner_couplings + 1)
        key = keys[0]
        idx = 1
        passes = []
        for _pass in range(2 * n_cycles):
            pass_weights = {"red": [], "black": []}
            for color in ("red", "black"):
                for _inner in range(n_inner_couplings):
                    kk = keys[idx]
                    idx += 1
                    if conditioner == "transformer":
                        net = _init_local_transformer(
                            kk,
                            in_dim=4,
                            model_dim=width,
                            out_dim=6,
                            n_heads=n_heads,
                            n_blocks=transformer_layers,
                            mlp_dim=mlp_dim,
                            radius=attn_radius,
                        )
                        if transformer_cfg is None:
                            transformer_cfg = net["cfg"]
                        pass_weights[color].append(net["weights"])
                    else:
                        pass_weights[color].append(
                            _init_local_mlp_conditioner(kk, in_dim=4, hidden_dim=width, out_dim=6)
                        )
            passes.append(pass_weights)
        level_weights.append(passes)

    key, k_terminal = jax.random.split(key)
    terminal = init_realnvp(
        k_terminal,
        n_layers=int(terminal_n_layers),
        width=terminal_width,
        log_scale_clip=log_scale_clip,
        parity=parity,
    )
    cfg["transformer_cfg"] = transformer_cfg
    cfg["terminal_flow_cfg"] = terminal["cfg"]
    cfg["arch"] = {
        "width": int(width),
        "n_cycles": int(n_cycles),
        "n_inner_couplings": int(n_inner_couplings),
        "conditioner": str(conditioner),
        "transformer_layers": int(transformer_layers),
        "n_heads": int(n_heads),
        "attn_radius": int(attn_radius),
        "mlp_dim": int(mlp_dim),
        "terminal_n_layers": int(terminal_n_layers),
        "terminal_width": int(terminal_width),
        "output_init_scale": float(output_init_scale),
        "parity": str(parity),
    }
    weights = {"levels": level_weights, "terminal_flow": terminal["weights"]}
    if float(output_init_scale) > 0.0:
        key, k_patch = jax.random.split(key)
        weights = _apply_small_output_init_checkerboard_inblock(
            weights, k_patch, scale=float(output_init_scale), conditioner_type=str(conditioner)
        )
    return {"cfg": cfg, "weights": weights}


def rg_checkerboard_inblock_flow_g_from(cfg: Dict, weights: Dict, z: Array) -> Array:
    x = z
    fines = []
    for level in range(max(cfg["depth"] - 1, 0)):
        fx = _level_flow_g(cfg, weights, x, level)
        cx, ff = _split_rg(fx, cfg["rg_mode"])
        fines.append(ff)
        x = cx
    fx = _terminal_flow_g(cfg, weights, x)
    for k in range(len(fines) - 1, -1, -1):
        fx = _merge_rg(fx, fines[k], cfg["rg_mode"])
    return fx


def rg_checkerboard_inblock_flow_f_from(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    z = x
    fines = []
    for _ in range(max(cfg["depth"] - 1, 0)):
        c, f = _split_rg(z, cfg["rg_mode"])
        fines.append(f)
        z = c
    z, ld_term = _terminal_flow_f(cfg, weights, z)
    ldj = ldj + ld_term
    for level in range(len(fines) - 1, -1, -1):
        y = _merge_rg(z, fines[level], cfg["rg_mode"])
        z, ld_level = _level_flow_f(cfg, weights, y, level)
        ldj = ldj + ld_level
    return z, ldj


def rg_checkerboard_inblock_flow_prior_sample_from(
    key: Array, cfg: Dict, batch_size: int, dtype=jnp.float32
) -> Array:
    return jax.random.normal(key, (batch_size, cfg["size_h"], cfg["size_w"]), dtype=dtype)


def rg_checkerboard_inblock_flow_prior_log_prob(z: Array) -> Array:
    return _std_normal_log_prob(z, sum_axes=(1, 2))


def rg_checkerboard_inblock_flow_log_prob_from(cfg: Dict, weights: Dict, x: Array) -> Array:
    z, ldj = rg_checkerboard_inblock_flow_f_from(cfg, weights, x)
    return rg_checkerboard_inblock_flow_prior_log_prob(z) + ldj


def _split_model(model_or_cfg: Dict, maybe_weights: Dict | None = None):
    if maybe_weights is not None:
        return model_or_cfg, maybe_weights
    if "size_h" in model_or_cfg and "size_w" in model_or_cfg:
        return model_or_cfg, None
    if "cfg" in model_or_cfg and "weights" in model_or_cfg:
        return model_or_cfg["cfg"], model_or_cfg["weights"]
    raise ValueError("Expected either (cfg, weights) or {'cfg':..., 'weights':...}")


def rg_checkerboard_inblock_flow_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_checkerboard_inblock_flow_g_from(cfg, w, z)


def rg_checkerboard_inblock_flow_f(
    model_or_cfg: Dict, x: Array, weights: Dict | None = None
) -> Tuple[Array, Array]:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_checkerboard_inblock_flow_f_from(cfg, w, x)


def rg_checkerboard_inblock_flow_prior_sample(
    key: Array,
    model_or_cfg: Dict,
    batch_size: int,
    dtype=jnp.float32,
    weights: Dict | None = None,
) -> Array:
    cfg, _ = _split_model(model_or_cfg, weights)
    return rg_checkerboard_inblock_flow_prior_sample_from(key, cfg, batch_size, dtype=dtype)


def rg_checkerboard_inblock_flow_log_prob(
    model_or_cfg: Dict, x: Array, weights: Dict | None = None
) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_checkerboard_inblock_flow_log_prob_from(cfg, w, x)

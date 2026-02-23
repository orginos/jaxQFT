"""JAX MGFlow for scalar phi^4 (functional, optax-friendly)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def _std_normal_log_prob(z: Array, sum_axes: Sequence[int]) -> Array:
    return jnp.sum(-0.5 * (z * z + math.log(2.0 * math.pi)), axis=tuple(sum_axes))


def _init_linear(key: Array, in_dim: int, out_dim: int, scale: float = 0.02):
    k1, k2 = jax.random.split(key)
    W = scale * jax.random.normal(k1, (in_dim, out_dim), dtype=jnp.float32)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"W": W, "b": b}


def _linear(p: Dict[str, Array], x: Array) -> Array:
    return x @ p["W"] + p["b"]


def _init_mlp(key: Array, in_dim: int, width: int, out_dim: int):
    k1, k2, k3 = jax.random.split(key, 3)
    p1 = _init_linear(k1, in_dim, width)
    p2 = _init_linear(k2, width, width)
    p3 = _init_linear(k3, width, out_dim, scale=0.0)  # identity-friendly
    return {"l1": p1, "l2": p2, "l3": p3}


def _mlp_apply(params: Dict[str, Dict[str, Array]], x: Array) -> Array:
    x = jax.nn.silu(_linear(params["l1"], x))
    x = jax.nn.silu(_linear(params["l2"], x))
    return _linear(params["l3"], x)


def init_st_provider(key: Array, D: int, n_blocks: int, width: int) -> Dict:
    keys = jax.random.split(key, n_blocks)
    nets = [_init_mlp(k, D, width, 2 * D) for k in keys]
    return {"nets": nets}


def init_st_mode(mode: str) -> int:
    mode_map = {"none": 0, "sym": 1, "x2": 2}
    if mode not in mode_map:
        raise ValueError(f"Unsupported parity mode: {mode}")
    return int(mode_map[mode])


def st_apply(st_weights: Dict, mode_id: int, i: int, xA: Array) -> Tuple[Array, Array]:
    D = xA.shape[-1]
    net = st_weights["nets"][i]
    if mode_id == 0:
        st = _mlp_apply(net, xA)
        return jnp.split(st, 2, axis=-1)
    if mode_id == 1:
        st_pos = _mlp_apply(net, xA)
        st_neg = _mlp_apply(net, -xA)
        s = 0.5 * (st_pos[..., :D] + st_neg[..., :D])
        t = 0.5 * (st_pos[..., D:] - st_neg[..., D:])
        return s, t
    # x2 mode
    st = _mlp_apply(net, xA * xA)
    s_even, t_even = jnp.split(st, 2, axis=-1)
    sum_xA = jnp.sum(xA, axis=1, keepdims=True)
    norm_xA = jnp.sqrt(1.0 + jnp.sum(xA * xA, axis=1, keepdims=True))
    phi = sum_xA / norm_xA
    return s_even, t_even * phi


def _make_masks_4d(n_layers: int) -> Array:
    mm = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    masks = jnp.stack([mm if k % 2 == 0 else 1.0 - mm for k in range(2 * n_layers)], axis=0)
    return masks


def init_realnvp(key: Array, n_layers: int, width: int, log_scale_clip: float, parity: str) -> Dict:
    masks = _make_masks_4d(n_layers)
    mode_id = init_st_mode(parity)
    st_weights = init_st_provider(key, D=4, n_blocks=masks.shape[0], width=width)
    return {
        "cfg": {
            "masks": masks,
            "masks_inv": 1.0 - masks,
            "log_scale_clip": float(log_scale_clip),
            "mode_id": mode_id,
        },
        "weights": {"st": st_weights},
    }


def realnvp_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    x = z
    n_blocks = cfg["masks"].shape[0]
    for i in range(n_blocks):
        m = cfg["masks"][i]
        mi = cfg["masks_inv"][i]
        xA = x * m
        s, t = st_apply(weights["st"], cfg["mode_id"], i, xA)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * mi
        t = t * mi
        x = xA + mi * (x * jnp.exp(s) + t)
    return x


def realnvp_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    z = x
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    n_blocks = cfg["masks"].shape[0]
    for i in range(n_blocks - 1, -1, -1):
        m = cfg["masks"][i]
        mi = cfg["masks_inv"][i]
        zA = z * m
        s, t = st_apply(weights["st"], cfg["mode_id"], i, zA)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * mi
        t = t * mi
        z = mi * (z - t) * jnp.exp(-s) + zA
        ldj = ldj - jnp.sum(mi * s, axis=1)
    return z, ldj


def _pack2x2(z4: Array) -> Array:
    c00 = z4[:, :, 0::2, 0::2]
    c01 = z4[:, :, 0::2, 1::2]
    c10 = z4[:, :, 1::2, 0::2]
    c11 = z4[:, :, 1::2, 1::2]
    return jnp.concatenate([c00, c01, c10, c11], axis=1)


def _unpack2x2(y: Array) -> Array:
    B, _, H2, W2 = y.shape
    out = jnp.zeros((B, 1, H2 * 2, W2 * 2), dtype=y.dtype)
    out = out.at[:, :, 0::2, 0::2].set(y[:, 0:1])
    out = out.at[:, :, 0::2, 1::2].set(y[:, 1:2])
    out = out.at[:, :, 1::2, 0::2].set(y[:, 2:3])
    out = out.at[:, :, 1::2, 1::2].set(y[:, 3:4])
    return out


def _conv_apply_g(z4: Array, bij_cfg: Dict, bij_weights: Dict) -> Array:
    y = _pack2x2(z4)
    B, C, H2, W2 = y.shape
    flat = jnp.reshape(jnp.transpose(y, (0, 2, 3, 1)), (B * H2 * W2, C))
    flat = realnvp_g(bij_cfg, bij_weights, flat)
    y = jnp.transpose(jnp.reshape(flat, (B, H2, W2, C)), (0, 3, 1, 2))
    return _unpack2x2(y)


def _conv_apply_f(x4: Array, bij_cfg: Dict, bij_weights: Dict) -> Tuple[Array, Array]:
    y = _pack2x2(x4)
    B, C, H2, W2 = y.shape
    flat = jnp.reshape(jnp.transpose(y, (0, 2, 3, 1)), (B * H2 * W2, C))
    zf, ld = realnvp_f(bij_cfg, bij_weights, flat)
    y = jnp.transpose(jnp.reshape(zf, (B, H2, W2, C)), (0, 3, 1, 2))
    out = _unpack2x2(y)
    ld_per_sample = jnp.sum(jnp.reshape(ld, (B, H2, W2)), axis=(1, 2))
    return out, ld_per_sample


def init_mgflow(
    key: Array,
    size: Tuple[int, int],
    n_layers: int = 3,
    width: int = 256,
    nconvs: int = 1,
    rg_type: str = "average",
    log_scale_clip: float = 5.0,
    parity: str = "none",
    fixed_bijector: bool = False,
) -> Dict:
    H, W = size
    if H != W or (H & (H - 1)) != 0:
        raise ValueError("size must be square power-of-2")
    depth = int(math.log2(H))
    rg_mode = 0 if rg_type == "average" else 1
    cfg = {
        "size_h": int(H),
        "size_w": int(W),
        "depth": depth,
        "rg_mode": rg_mode,
        "nconvs": int(nconvs),
        "fixed_bijector": bool(fixed_bijector),
    }
    weights: Dict = {}
    bij = init_realnvp(jax.random.PRNGKey(0), n_layers=n_layers, width=width, log_scale_clip=log_scale_clip, parity=parity)
    bij_cfg = bij["cfg"]
    cfg["bij_cfg"] = bij_cfg
    if fixed_bijector:
        key, k = jax.random.split(key)
        shared = init_realnvp(k, n_layers=n_layers, width=width, log_scale_clip=log_scale_clip, parity=parity)
        weights["shared_bijector"] = shared["weights"]
        weights["cflow"] = [None for _ in range(depth)]
    else:
        cflow = []
        for _ in range(depth):
            keys = jax.random.split(key, 2 * nconvs + 1)
            key = keys[0]
            blocks = [init_realnvp(k, n_layers=n_layers, width=width, log_scale_clip=log_scale_clip, parity=parity)["weights"] for k in keys[1:]]
            cflow.append(blocks)
        weights["cflow"] = cflow
    return {"cfg": cfg, "weights": weights}


def rg_coarsen(f: Array, rg_mode: int) -> Tuple[Array, Array]:
    ff = f[:, None, :, :]
    if rg_mode == 0:
        c = 0.25 * (ff[:, :, 0::2, 0::2] + ff[:, :, 1::2, 0::2] + ff[:, :, 0::2, 1::2] + ff[:, :, 1::2, 1::2])
        up = jnp.repeat(jnp.repeat(c, 2, axis=2), 2, axis=3)
        r = ff - up
    else:  # select
        c = ff[:, :, ::2, ::2]
        up = jnp.repeat(jnp.repeat(c, 2, axis=2), 2, axis=3)
        r = ff - up
    return c[:, 0], r[:, 0]


def rg_refine(c: Array, r: Array) -> Array:
    up = jnp.repeat(jnp.repeat(c[:, None], 2, axis=2), 2, axis=3)[:, 0]
    return up + r


def _convflow_g(cfg: Dict, weights: Dict, z: Array, level: int) -> Array:
    z4 = z[:, None, :, :]
    nsteps = cfg["nconvs"]
    for k in range(nsteps):
        if cfg["fixed_bijector"]:
            bj0 = weights["shared_bijector"]
            bj1 = weights["shared_bijector"]
        else:
            bj0 = weights["cflow"][level][2 * k]
            bj1 = weights["cflow"][level][2 * k + 1]
        z4 = _conv_apply_g(z4, cfg["bij_cfg"], bj0)
        z4 = jnp.roll(z4, shift=(-1, -1), axis=(2, 3))
        z4 = _conv_apply_g(z4, cfg["bij_cfg"], bj1)
        z4 = jnp.roll(z4, shift=(1, 1), axis=(2, 3))
    return z4[:, 0]


def _convflow_f(cfg: Dict, weights: Dict, x: Array, level: int) -> Tuple[Array, Array]:
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    z4 = x[:, None, :, :]
    nsteps = cfg["nconvs"]
    for k in range(nsteps - 1, -1, -1):
        if cfg["fixed_bijector"]:
            bj0 = weights["shared_bijector"]
            bj1 = weights["shared_bijector"]
        else:
            bj0 = weights["cflow"][level][2 * k]
            bj1 = weights["cflow"][level][2 * k + 1]
        z4 = jnp.roll(z4, shift=(-1, -1), axis=(2, 3))
        z4, l1 = _conv_apply_f(z4, cfg["bij_cfg"], bj1)
        z4 = jnp.roll(z4, shift=(1, 1), axis=(2, 3))
        z4, l0 = _conv_apply_f(z4, cfg["bij_cfg"], bj0)
        ldj = ldj + l1 + l0
    return z4[:, 0], ldj


def mgflow_g_from(cfg: Dict, weights: Dict, z: Array) -> Array:
    x = z
    fines: List[Array] = []
    for level in range(cfg["depth"] - 1):
        fx = _convflow_g(cfg, weights, x, level)
        cx, ff = rg_coarsen(fx, cfg["rg_mode"])
        fines.append(ff)
        x = cx
    fx = _convflow_g(cfg, weights, x, cfg["depth"] - 1)
    for k in range(cfg["depth"] - 1, 0, -1):
        fx = rg_refine(fx, fines[k - 1])
    return fx


def mgflow_f_from(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    ldj = jnp.zeros((x.shape[0],), dtype=x.dtype)
    fines: List[Array] = []
    z = x
    for _ in range(cfg["depth"] - 1):
        c, f = rg_coarsen(z, cfg["rg_mode"])
        fines.append(f)
        z = c
    z, ldk = _convflow_f(cfg, weights, z, cfg["depth"] - 1)
    ldj = ldj + ldk
    for k in range(cfg["depth"] - 2, -1, -1):
        y = rg_refine(z, fines[k])
        z, ldk = _convflow_f(cfg, weights, y, k)
        ldj = ldj + ldk
    return z, ldj


def mgflow_prior_sample_from(key: Array, cfg: Dict, batch_size: int, dtype=jnp.float32) -> Array:
    H, W = int(cfg["size_h"]), int(cfg["size_w"])
    return jax.random.normal(key, (batch_size, H, W), dtype=dtype)


def mgflow_prior_log_prob(z: Array) -> Array:
    return _std_normal_log_prob(z, sum_axes=(1, 2))


def mgflow_log_prob_from(cfg: Dict, weights: Dict, x: Array) -> Array:
    z, ldj = mgflow_f_from(cfg, weights, x)
    return mgflow_prior_log_prob(z) + ldj


def _split_model(model_or_cfg: Dict, maybe_weights: Dict | None = None):
    if maybe_weights is not None:
        return model_or_cfg, maybe_weights
    if "size_h" in model_or_cfg and "size_w" in model_or_cfg:
        return model_or_cfg, None
    if "cfg" in model_or_cfg and "weights" in model_or_cfg:
        return model_or_cfg["cfg"], model_or_cfg["weights"]
    raise ValueError("Expected either (cfg, weights) or {'cfg':..., 'weights':...}")


def mgflow_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return mgflow_g_from(cfg, w, z)


def mgflow_f(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Tuple[Array, Array]:
    cfg, w = _split_model(model_or_cfg, weights)
    return mgflow_f_from(cfg, w, x)


def mgflow_prior_sample(key: Array, model_or_cfg: Dict, batch_size: int, dtype=jnp.float32, weights: Dict | None = None) -> Array:
    cfg, _ = _split_model(model_or_cfg, weights)
    return mgflow_prior_sample_from(key, cfg, batch_size, dtype=dtype)


def mgflow_log_prob(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return mgflow_log_prob_from(cfg, w, x)

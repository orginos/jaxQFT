"""RG-first conditional transformer flow for scalar phi^4.

This model keeps the existing ``phi4_mg`` implementation untouched and adds a
new architecture closer to the RG factorization described in the user note:

1. Split each 2x2 block into one coarse degree of freedom and three fine modes.
2. Map the three fine modes through a conditional affine coupling flow.
3. Use a local transformer only to parameterize the coupling functions ``s,t``.
4. Recurse on the coarse field.

For ``rg_type="average"`` the coarse channel is a normalized 2x2 Haar average,
proportional to the arithmetic block average. The three fine channels span the
orthogonal fluctuation subspace. For ``rg_type="select"`` the coarse field is
the selected site and the three fine channels are the remaining block entries.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp

from .phi4_mg import init_realnvp, realnvp_f, realnvp_g


Array = jax.Array

_AVG_BASIS = jnp.array(
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
    ],
    dtype=jnp.float32,
)


def _std_normal_log_prob(z: Array, sum_axes: Sequence[int]) -> Array:
    return jnp.sum(-0.5 * (z * z + math.log(2.0 * math.pi)), axis=tuple(sum_axes))


def _init_linear(key: Array, in_dim: int, out_dim: int, scale: float = 0.02) -> Dict[str, Array]:
    k_w, _ = jax.random.split(key)
    return {
        "W": scale * jax.random.normal(k_w, (in_dim, out_dim), dtype=jnp.float32),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: Dict[str, Array], x: Array) -> Array:
    return x @ params["W"] + params["b"]


def _init_layernorm(dim: int) -> Dict[str, Array]:
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _layernorm(params: Dict[str, Array], x: Array, eps: float = 1e-6) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) * jax.lax.rsqrt(var + eps)
    return xhat * params["scale"] + params["bias"]


def _init_mlp(key: Array, in_dim: int, hidden_dim: int, out_dim: int, final_scale: float = 0.0) -> Dict[str, Dict[str, Array]]:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "l1": _init_linear(k1, in_dim, hidden_dim),
        "l2": _init_linear(k2, hidden_dim, hidden_dim),
        "l3": _init_linear(k3, hidden_dim, out_dim, scale=final_scale),
    }


def _mlp_apply(params: Dict[str, Dict[str, Array]], x: Array) -> Array:
    x = jax.nn.gelu(_linear(params["l1"], x))
    x = jax.nn.gelu(_linear(params["l2"], x))
    return _linear(params["l3"], x)


def _make_local_offsets(radius: int) -> List[Tuple[int, int]]:
    return [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]


def _roll_stack(x: Array, offsets: Sequence[Tuple[int, int]]) -> Array:
    return jnp.stack([jnp.roll(x, shift=(dy, dx), axis=(1, 2)) for dy, dx in offsets], axis=3)


def _init_transformer_block(key: Array, model_dim: int, n_heads: int, mlp_dim: int, n_offsets: int) -> Dict:
    if model_dim % n_heads != 0:
        raise ValueError("model_dim must be divisible by n_heads")
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "ln1": _init_layernorm(model_dim),
        "ln2": _init_layernorm(model_dim),
        "qkv": _init_linear(k1, model_dim, 3 * model_dim),
        "proj": _init_linear(k2, model_dim, model_dim, scale=0.0),
        "rel_bias": jnp.zeros((n_heads, n_offsets), dtype=jnp.float32),
        "mlp": _init_mlp(k3, model_dim, mlp_dim, model_dim, final_scale=0.0),
        "gate": _init_linear(k4, model_dim, model_dim, scale=0.0),
    }


def _attention_apply(block: Dict, x: Array, offsets: Sequence[Tuple[int, int]], n_heads: int) -> Array:
    bsz, ny, nx, model_dim = x.shape
    head_dim = model_dim // n_heads

    qkv = _linear(block["qkv"], x).reshape(bsz, ny, nx, 3, n_heads, head_dim)
    q = qkv[:, :, :, 0]
    k = qkv[:, :, :, 1]
    v = qkv[:, :, :, 2]

    k_local = _roll_stack(k, offsets)
    v_local = _roll_stack(v, offsets)

    logits = jnp.einsum("bijhd,bijnhd->bijhn", q, k_local) / math.sqrt(head_dim)
    logits = logits + block["rel_bias"][None, None, None, :, :]
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bijhn,bijnhd->bijhd", attn, v_local).reshape(bsz, ny, nx, model_dim)
    return _linear(block["proj"], out)


def _transformer_block_apply(block: Dict, x: Array, offsets: Sequence[Tuple[int, int]], n_heads: int) -> Array:
    h = _layernorm(block["ln1"], x)
    x = x + _attention_apply(block, h, offsets, n_heads)
    h = _layernorm(block["ln2"], x)
    mlp_out = _mlp_apply(block["mlp"], h)
    gate = jax.nn.sigmoid(_linear(block["gate"], h))
    return x + gate * mlp_out


def _init_local_transformer(
    key: Array,
    in_dim: int,
    model_dim: int,
    out_dim: int,
    n_heads: int,
    n_blocks: int,
    mlp_dim: int,
    radius: int,
) -> Dict:
    offsets = _make_local_offsets(radius)
    keys = jax.random.split(key, n_blocks + 2)
    return {
        "cfg": {
            "in_dim": int(in_dim),
            "model_dim": int(model_dim),
            "out_dim": int(out_dim),
            "n_heads": int(n_heads),
            "radius": int(radius),
            "offsets": tuple(offsets),
        },
        "weights": {
            "input": _init_linear(keys[0], in_dim, model_dim),
            "blocks": [
                _init_transformer_block(k, model_dim=model_dim, n_heads=n_heads, mlp_dim=mlp_dim, n_offsets=len(offsets))
                for k in keys[1:-1]
            ],
            "head": _init_mlp(keys[-1], model_dim, mlp_dim, out_dim, final_scale=0.0),
        },
    }


def _local_transformer_apply(cfg: Dict, weights: Dict, x: Array) -> Array:
    h = _linear(weights["input"], x)
    for block in weights["blocks"]:
        h = _transformer_block_apply(block, h, offsets=cfg["offsets"], n_heads=cfg["n_heads"])
    return _mlp_apply(weights["head"], h)


def _pack_blocks(x: Array) -> Array:
    bsz, ny2, nx2 = x.shape
    ny = ny2 // 2
    nx = nx2 // 2
    return jnp.transpose(x.reshape(bsz, ny, 2, nx, 2), (0, 1, 3, 2, 4)).reshape(bsz, ny, nx, 4)


def _unpack_blocks(blocks: Array) -> Array:
    bsz, ny, nx, _ = blocks.shape
    return jnp.transpose(blocks.reshape(bsz, ny, nx, 2, 2), (0, 1, 3, 2, 4)).reshape(bsz, ny * 2, nx * 2)


def _split_rg(x: Array, rg_mode: int) -> Tuple[Array, Array]:
    blocks = _pack_blocks(x)
    if rg_mode == 0:
        coeffs = jnp.einsum("...i,ji->...j", blocks, _AVG_BASIS)
        return coeffs[..., 0], coeffs[..., 1:]
    return blocks[..., 0], blocks[..., 1:]


def _merge_rg(coarse: Array, fine: Array, rg_mode: int) -> Array:
    if rg_mode == 0:
        coeffs = jnp.concatenate([coarse[..., None], fine], axis=-1)
        blocks = jnp.einsum("...i,ij->...j", coeffs, _AVG_BASIS)
    else:
        blocks = jnp.concatenate([coarse[..., None], fine], axis=-1)
    return _unpack_blocks(blocks)


def _coarse_condition(coarse: Array, rg_mode: int) -> Array:
    if rg_mode == 0:
        return 0.5 * coarse
    return coarse


def _make_masks_3d(n_couplings: int) -> Array:
    base = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    return jnp.stack([base[k % 3] for k in range(n_couplings)], axis=0)


def _apply_small_output_init(weights: Dict, key: Array, scale: float) -> Dict:
    level_weights = []
    kcur = key
    for level in weights["levels"]:
        couplings = []
        for coupling in level:
            c = dict(coupling)
            c["head"] = dict(c["head"])
            c["head"]["l3"] = dict(c["head"]["l3"])
            kcur, k1 = jax.random.split(kcur)
            c["head"]["l3"]["W"] = scale * jax.random.normal(
                k1, c["head"]["l3"]["W"].shape, dtype=c["head"]["l3"]["W"].dtype
            )

            blocks = []
            for blk in c["blocks"]:
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
            c["blocks"] = blocks
            couplings.append(c)
        level_weights.append(couplings)

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


def _conditional_st(cfg: Dict, coupling: Dict, context: Array) -> Tuple[Array, Array]:
    transformer_cfg = cfg["transformer_cfg"]
    parity = cfg["parity"]
    if parity == "none":
        st = _local_transformer_apply(transformer_cfg, coupling, context)
        return jnp.split(st, 2, axis=-1)
    if parity == "sym":
        st_pos = _local_transformer_apply(transformer_cfg, coupling, context)
        st_neg = _local_transformer_apply(transformer_cfg, coupling, -context)
        d = st_pos.shape[-1] // 2
        s = 0.5 * (st_pos[..., :d] + st_neg[..., :d])
        t = 0.5 * (st_pos[..., d:] - st_neg[..., d:])
        return s, t
    raise ValueError(f"Unsupported parity mode for RG conditional flow: {parity}")


def _terminal_flow_g(cfg: Dict, weights: Dict, z: Array) -> Array:
    flat = z.reshape((z.shape[0], 4))
    out = realnvp_g(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return out.reshape(z.shape)


def _terminal_flow_f(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    flat = x.reshape((x.shape[0], 4))
    z, ldj = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
    return z.reshape(x.shape), ldj


def init_rg_cond_flow(
    key: Array,
    size: Tuple[int, int],
    *,
    width: int = 64,
    n_couplings: int = 4,
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
    if parity not in ("none", "sym"):
        raise ValueError("parity must be one of {'none','sym'}")

    depth = int(math.log2(h))
    rg_mode = 0 if rg_type == "average" else 1
    mlp_dim = int(mlp_dim if mlp_dim is not None else 2 * width)
    terminal_width = int(terminal_width if terminal_width is not None else width)
    masks = _make_masks_3d(n_couplings)

    cfg = {
        "size_h": int(h),
        "size_w": int(w),
        "depth": int(depth),
        "rg_mode": int(rg_mode),
        "parity": str(parity),
        "masks": masks,
        "masks_inv": 1.0 - masks,
        "log_scale_clip": float(log_scale_clip),
    }

    transformer_cfg = None
    level_weights = []
    for _ in range(depth):
        keys = jax.random.split(key, n_couplings + 1)
        key = keys[0]
        couplings = []
        for k in keys[1:]:
            trf = _init_local_transformer(
                k,
                in_dim=4,
                model_dim=width,
                out_dim=6,
                n_heads=n_heads,
                n_blocks=transformer_layers,
                mlp_dim=mlp_dim,
                radius=attn_radius,
            )
            if transformer_cfg is None:
                transformer_cfg = trf["cfg"]
            couplings.append(trf["weights"])
        level_weights.append(couplings)

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
        "n_couplings": int(n_couplings),
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
        weights = _apply_small_output_init(weights, k_patch, scale=float(output_init_scale))
    return {"cfg": cfg, "weights": weights}


def _eta_context(eta_masked: Array, coarse: Array, rg_mode: int) -> Array:
    coarse_ctx = _coarse_condition(coarse, rg_mode)[..., None]
    return jnp.concatenate([eta_masked, coarse_ctx], axis=-1)


def _eta_flow_g(cfg: Dict, weights: Dict, eta: Array, coarse: Array, level: int) -> Array:
    x = eta
    for i, coupling in enumerate(weights["levels"][level]):
        mask = cfg["masks"][i].reshape((1, 1, 1, 3))
        inv_mask = cfg["masks_inv"][i].reshape((1, 1, 1, 3))
        x_a = x * mask
        context = _eta_context(x_a, coarse, cfg["rg_mode"])
        s, t = _conditional_st(cfg, coupling, context)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * inv_mask
        t = t * inv_mask
        x = x_a + inv_mask * (x * jnp.exp(s) + t)
    return x


def _eta_flow_f(cfg: Dict, weights: Dict, eta: Array, coarse: Array, level: int) -> Tuple[Array, Array]:
    z = eta
    ldj = jnp.zeros((eta.shape[0],), dtype=eta.dtype)
    spatial_axes = (1, 2, 3)
    for i in range(len(weights["levels"][level]) - 1, -1, -1):
        coupling = weights["levels"][level][i]
        mask = cfg["masks"][i].reshape((1, 1, 1, 3))
        inv_mask = cfg["masks_inv"][i].reshape((1, 1, 1, 3))
        z_a = z * mask
        context = _eta_context(z_a, coarse, cfg["rg_mode"])
        s, t = _conditional_st(cfg, coupling, context)
        s = jnp.tanh(s) * cfg["log_scale_clip"]
        s = s * inv_mask
        t = t * inv_mask
        z = z_a + inv_mask * (z - t) * jnp.exp(-s)
        ldj = ldj - jnp.sum(s, axis=spatial_axes)
    return z, ldj


def _rg_cond_g_level(cfg: Dict, weights: Dict, z: Array, level: int) -> Array:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_g(cfg, weights, z)
    z_coarse, z_fine = _split_rg(z, cfg["rg_mode"])
    x_coarse = _rg_cond_g_level(cfg, weights, z_coarse, level + 1)
    x_fine = _eta_flow_g(cfg, weights, z_fine, x_coarse, level)
    return _merge_rg(x_coarse, x_fine, cfg["rg_mode"])


def _rg_cond_f_level(cfg: Dict, weights: Dict, x: Array, level: int) -> Tuple[Array, Array]:
    if level >= cfg["depth"] - 1:
        return _terminal_flow_f(cfg, weights, x)
    x_coarse, x_fine = _split_rg(x, cfg["rg_mode"])
    z_coarse, ldj_coarse = _rg_cond_f_level(cfg, weights, x_coarse, level + 1)
    z_fine, ldj_fine = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
    return _merge_rg(z_coarse, z_fine, cfg["rg_mode"]), ldj_coarse + ldj_fine


def rg_cond_flow_g_from(cfg: Dict, weights: Dict, z: Array) -> Array:
    return _rg_cond_g_level(cfg, weights, z, 0)


def rg_cond_flow_f_from(cfg: Dict, weights: Dict, x: Array) -> Tuple[Array, Array]:
    return _rg_cond_f_level(cfg, weights, x, 0)


def rg_cond_flow_prior_sample_from(key: Array, cfg: Dict, batch_size: int, dtype=jnp.float32) -> Array:
    return jax.random.normal(key, (batch_size, cfg["size_h"], cfg["size_w"]), dtype=dtype)


def rg_cond_flow_prior_log_prob(z: Array) -> Array:
    return _std_normal_log_prob(z, sum_axes=(1, 2))


def rg_cond_flow_log_prob_from(cfg: Dict, weights: Dict, x: Array) -> Array:
    z, ldj = rg_cond_flow_f_from(cfg, weights, x)
    return rg_cond_flow_prior_log_prob(z) + ldj


def _split_model(model_or_cfg: Dict, maybe_weights: Dict | None = None):
    if maybe_weights is not None:
        return model_or_cfg, maybe_weights
    if "size_h" in model_or_cfg and "size_w" in model_or_cfg:
        return model_or_cfg, None
    if "cfg" in model_or_cfg and "weights" in model_or_cfg:
        return model_or_cfg["cfg"], model_or_cfg["weights"]
    raise ValueError("Expected either (cfg, weights) or {'cfg':..., 'weights':...}")


def rg_cond_flow_g(model_or_cfg: Dict, z: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_cond_flow_g_from(cfg, w, z)


def rg_cond_flow_f(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Tuple[Array, Array]:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_cond_flow_f_from(cfg, w, x)


def rg_cond_flow_prior_sample(
    key: Array,
    model_or_cfg: Dict,
    batch_size: int,
    dtype=jnp.float32,
    weights: Dict | None = None,
) -> Array:
    cfg, _ = _split_model(model_or_cfg, weights)
    return rg_cond_flow_prior_sample_from(key, cfg, batch_size, dtype=dtype)


def rg_cond_flow_log_prob(model_or_cfg: Dict, x: Array, weights: Dict | None = None) -> Array:
    cfg, w = _split_model(model_or_cfg, weights)
    return rg_cond_flow_log_prob_from(cfg, w, x)

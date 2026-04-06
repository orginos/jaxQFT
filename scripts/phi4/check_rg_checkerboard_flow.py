#!/usr/bin/env python3
"""Validation and timing harness for the RG checkerboard phi^4 flow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.models.phi4 import Phi4
from jaxqft.models.phi4_rg_checkerboard_flow import (
    _checkerboard_pass_f,
    _checkerboard_pass_g,
    _conditional_st,
    _level_flow_f,
    _level_flow_g,
    _masked_eta_context,
    _site_update_f,
    _site_update_g,
    _split_rg,
    _terminal_flow_f,
    _terminal_flow_g,
    init_rg_checkerboard_flow,
    rg_checkerboard_flow_f,
    rg_checkerboard_flow_g,
    rg_checkerboard_flow_log_prob,
    rg_checkerboard_flow_prior_sample,
)


def _parse_shape(s: str) -> tuple[int, int]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if len(vals) != 2:
        raise ValueError("shape must be H,W")
    return int(vals[0]), int(vals[1])


def _sync_tree(x):
    return jax.block_until_ready(x)


def _time_call(fn, args, warmup: int, n_iter: int) -> tuple[float, object]:
    y = None
    for _ in range(max(1, warmup)):
        y = fn(*args)
        _sync_tree(y)
    t0 = time.perf_counter()
    for _ in range(max(1, n_iter)):
        y = fn(*args)
        _sync_tree(y)
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, n_iter), y


def _perturb_tree(tree, key: jax.Array, scale: float):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return tree
    keys = jax.random.split(key, len(leaves))
    out = []
    for leaf, kk in zip(leaves, keys):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            noise = jax.random.normal(kk, leaf.shape, dtype=leaf.dtype)
            out.append(leaf + scale * noise)
        else:
            out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


def run_invertibility_test(model, shape: tuple[int, int], batch_size: int, tol: float) -> dict:
    key = jax.random.PRNGKey(0)
    z = rg_checkerboard_flow_prior_sample(key, model, batch_size)
    x = rg_checkerboard_flow_g(model, z)
    zz, ldj = rg_checkerboard_flow_f(model, x)
    abs_err = jnp.max(jnp.abs(zz - z))
    rel_err = jnp.linalg.norm((zz - z).reshape(batch_size, -1), axis=1) / (
        jnp.linalg.norm(z.reshape(batch_size, -1), axis=1) + 1e-12
    )
    return {
        "shape": list(shape),
        "batch_size": int(batch_size),
        "max_abs_err": float(abs_err),
        "max_rel_err": float(jnp.max(rel_err)),
        "ldj_shape": list(ldj.shape),
        "pass": bool(float(abs_err) <= tol),
    }


def run_jacobian_test(model, shape: tuple[int, int], tol: float, perturb_scale: float) -> dict:
    cfg = model["cfg"]
    weights = model["weights"]
    key = jax.random.PRNGKey(1)
    if perturb_scale > 0.0:
        key, kp = jax.random.split(key)
        weights = _perturb_tree(weights, kp, perturb_scale)
    x = jax.random.normal(key, (1, *shape), dtype=jnp.float32)

    def flat_inverse(flat_x):
        xx = flat_x.reshape((1, *shape))
        z, _ = rg_checkerboard_flow_f(cfg, xx, weights)
        return z.reshape((-1,))

    z, ldj = rg_checkerboard_flow_f(cfg, x, weights)
    jac = jax.jacfwd(flat_inverse)(x.reshape((-1,)))
    sign, logabsdet = jnp.linalg.slogdet(jac)
    err = jnp.abs(logabsdet - ldj[0])
    return {
        "shape": list(shape),
        "perturb_scale": float(perturb_scale),
        "jacobian_sign": float(sign),
        "autodiff_logdet": float(logabsdet),
        "network_logdet": float(ldj[0]),
        "abs_err": float(err),
        "pass": bool(float(err) <= tol and float(jnp.abs(sign)) > 0.5),
    }


def run_timing_test(model, theory, batch_size: int, warmup: int, n_iter: int) -> dict:
    cfg = model["cfg"]
    weights = model["weights"]
    key = jax.random.PRNGKey(2)
    z = rg_checkerboard_flow_prior_sample(key, model, batch_size)
    x = rg_checkerboard_flow_g(model, z)
    coarse, eta = _split_rg(x, cfg["rg_mode"])
    pass_weights = weights["levels"][0][0] if weights["levels"] else None
    context = _masked_eta_context(eta * cfg["site_black"][0], coarse, cfg["rg_mode"]) if weights["levels"] else None
    terminal_x = x
    terminal_z = z
    while terminal_x.shape[1] > 2:
        terminal_x, _ = _split_rg(terminal_x, cfg["rg_mode"])
        terminal_z, _ = _split_rg(terminal_z, cfg["rg_mode"])

    split_jit = jax.jit(lambda xx: _split_rg(xx, cfg["rg_mode"]))
    trf_jit = (
        jax.jit(lambda cc: _conditional_st(cfg, pass_weights["red"], cc))
        if weights["levels"]
        else None
    )
    site_red_g_jit = (
        jax.jit(lambda ee, cc: _site_update_g(cfg, pass_weights, ee, cc, 0, "red"))
        if weights["levels"]
        else None
    )
    site_black_f_jit = (
        jax.jit(lambda ee, cc: _site_update_f(cfg, pass_weights, ee, cc, 0, "black"))
        if weights["levels"]
        else None
    )
    pass_g_jit = jax.jit(lambda xx: _checkerboard_pass_g(cfg, weights, xx, 0, 0)) if weights["levels"] else None
    pass_f_jit = jax.jit(lambda xx: _checkerboard_pass_f(cfg, weights, xx, 0, 0)) if weights["levels"] else None
    level_g_jit = jax.jit(lambda xx: _level_flow_g(cfg, weights, xx, 0)) if weights["levels"] else None
    level_f_jit = jax.jit(lambda xx: _level_flow_f(cfg, weights, xx, 0)) if weights["levels"] else None
    terminal_g_jit = jax.jit(lambda zz: _terminal_flow_g(cfg, weights, zz))
    terminal_f_jit = jax.jit(lambda xx: _terminal_flow_f(cfg, weights, xx))
    full_g_jit = jax.jit(lambda zz: rg_checkerboard_flow_g(cfg, zz, weights))
    full_f_jit = jax.jit(lambda xx: rg_checkerboard_flow_f(cfg, xx, weights))
    logp_jit = jax.jit(lambda xx: rg_checkerboard_flow_log_prob(cfg, xx, weights))

    def loss_fn(w):
        key = jax.random.PRNGKey(7)
        zz = rg_checkerboard_flow_prior_sample(key, cfg, batch_size)
        xx = rg_checkerboard_flow_g(cfg, zz, w)
        return jnp.mean(rg_checkerboard_flow_log_prob(cfg, xx, w) + theory.action(xx))

    loss_grad_jit = jax.jit(jax.value_and_grad(loss_fn))

    items = [("split_rg", split_jit, (x,))]
    if weights["levels"]:
        items.extend(
            [
                ("conditional_st", trf_jit, (context,)),
                ("site_update_red_g", site_red_g_jit, (eta, coarse)),
                ("site_update_black_f", site_black_f_jit, (eta, coarse)),
                ("checkerboard_pass_g", pass_g_jit, (x,)),
                ("checkerboard_pass_f", pass_f_jit, (x,)),
                ("level_flow_g", level_g_jit, (x,)),
                ("level_flow_f", level_f_jit, (x,)),
            ]
        )
    items.extend(
        [
            ("terminal_g", terminal_g_jit, (terminal_z,)),
            ("terminal_f", terminal_f_jit, (terminal_x,)),
            ("full_g", full_g_jit, (z,)),
            ("full_f", full_f_jit, (x,)),
            ("log_prob", logp_jit, (x,)),
            ("loss_grad", loss_grad_jit, (weights,)),
        ]
    )

    timings = {}
    for name, fn, args in items:
        t_sec, _ = _time_call(fn, args, warmup=warmup, n_iter=n_iter)
        timings[name] = float(t_sec)

    ranked = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in ranked)
    bottlenecks = [
        {"name": name, "sec": float(val), "share": float(val / max(total, 1e-12))}
        for name, val in ranked[:3]
    ]
    return {
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "n_iter": int(n_iter),
        "timings_sec": timings,
        "ranked": [{"name": name, "sec": float(val)} for name, val in ranked],
        "bottlenecks": bottlenecks,
    }


def main():
    ap = argparse.ArgumentParser(description="Check the RG checkerboard phi^4 flow.")
    ap.add_argument("--shape", type=str, default="8,8")
    ap.add_argument("--jac-shape", type=str, default="2,2")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=1)
    ap.add_argument("--conditioner", type=str, default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--transformer-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--attn-radius", type=int, default=1)
    ap.add_argument("--mlp-dim", type=int, default=0)
    ap.add_argument("--terminal-n-layers", type=int, default=2)
    ap.add_argument("--terminal-width", type=int, default=0)
    ap.add_argument("--output-init-scale", type=float, default=1e-2)
    ap.add_argument("--parity", type=str, default="sym", choices=["none", "sym"])
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=-0.5)
    ap.add_argument("--tests", type=str, default="invertibility,jacobian,timing")
    ap.add_argument("--invert-tol", type=float, default=1e-5)
    ap.add_argument("--jac-tol", type=float, default=1e-5)
    ap.add_argument("--jac-perturb-scale", type=float, default=1e-2)
    ap.add_argument("--timing-warmup", type=int, default=3)
    ap.add_argument("--timing-iters", type=int, default=10)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--selfcheck-fail", action="store_true")
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    jac_shape = _parse_shape(args.jac_shape)
    tests = {t.strip() for t in args.tests.split(",") if t.strip()}
    mlp_dim = None if args.mlp_dim <= 0 else int(args.mlp_dim)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(f"shape={shape} jac_shape={jac_shape} rg={args.rg_type}")
    print(
        "model:"
        f" width={args.width} n_cycles={args.n_cycles}"
        f" conditioner={args.conditioner}"
        f" transformer_layers={args.transformer_layers}"
        f" n_heads={args.n_heads} attn_radius={args.attn_radius}"
        f" mlp_dim={2 * args.width if mlp_dim is None else mlp_dim}"
        f" terminal_layers={args.terminal_n_layers}"
        f" terminal_width={args.width if args.terminal_width <= 0 else args.terminal_width}"
        f" output_init_scale={args.output_init_scale}"
        f" parity={args.parity}"
    )

    key = jax.random.PRNGKey(42)
    model = init_rg_checkerboard_flow(
        key,
        size=shape,
        width=args.width,
        n_cycles=args.n_cycles,
        conditioner=args.conditioner,
        transformer_layers=args.transformer_layers,
        n_heads=args.n_heads,
        attn_radius=args.attn_radius,
        mlp_dim=mlp_dim,
        rg_type=args.rg_type,
        log_scale_clip=args.log_scale_clip,
        terminal_n_layers=args.terminal_n_layers,
        terminal_width=None if args.terminal_width <= 0 else args.terminal_width,
        output_init_scale=args.output_init_scale,
        parity=args.parity,
    )
    jac_model = init_rg_checkerboard_flow(
        key,
        size=jac_shape,
        width=args.width,
        n_cycles=args.n_cycles,
        conditioner=args.conditioner,
        transformer_layers=args.transformer_layers,
        n_heads=args.n_heads,
        attn_radius=args.attn_radius,
        mlp_dim=mlp_dim,
        rg_type=args.rg_type,
        log_scale_clip=args.log_scale_clip,
        terminal_n_layers=args.terminal_n_layers,
        terminal_width=None if args.terminal_width <= 0 else args.terminal_width,
        output_init_scale=args.output_init_scale,
        parity=args.parity,
    )
    theory = Phi4(list(shape), args.lam, args.mass, batch_size=args.batch_size)

    results = {}
    overall = True

    if "invertibility" in tests:
        r = run_invertibility_test(model, shape, batch_size=args.batch_size, tol=args.invert_tol)
        results["invertibility"] = r
        overall = overall and bool(r["pass"])
        print(
            "invertibility:"
            f" max_abs_err={r['max_abs_err']:.3e}"
            f" max_rel_err={r['max_rel_err']:.3e}"
            f" pass={r['pass']}"
        )

    if "jacobian" in tests:
        r = run_jacobian_test(jac_model, jac_shape, tol=args.jac_tol, perturb_scale=args.jac_perturb_scale)
        results["jacobian"] = r
        overall = overall and bool(r["pass"])
        print(
            "jacobian:"
            f" sign={r['jacobian_sign']:.1f}"
            f" autodiff_logdet={r['autodiff_logdet']:.6f}"
            f" network_logdet={r['network_logdet']:.6f}"
            f" abs_err={r['abs_err']:.3e}"
            f" pass={r['pass']}"
        )

    if "timing" in tests:
        r = run_timing_test(model, theory, batch_size=args.batch_size, warmup=args.timing_warmup, n_iter=args.timing_iters)
        results["timing"] = r
        print("timing (sec):")
        for item in r["ranked"]:
            print(f"  {item['name']}: {item['sec']:.6e}")

    results["overall_pass"] = bool(overall)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if args.selfcheck_fail and not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the RG checkerboard in-block conditional flow for phi^4."""

from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from pathlib import Path

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
from jaxqft.models.phi4_rg_checkerboard_inblock_flow import (
    init_rg_checkerboard_inblock_flow,
    rg_checkerboard_inblock_flow_g,
    rg_checkerboard_inblock_flow_log_prob,
    rg_checkerboard_inblock_flow_prior_sample,
)


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def loss_fn(cfg, weights, key, batch_size, theory):
    z = rg_checkerboard_inblock_flow_prior_sample(key, cfg, batch_size)
    x = rg_checkerboard_inblock_flow_g(cfg, z, weights)
    return jnp.mean(rg_checkerboard_inblock_flow_log_prob(cfg, x, weights) + theory.action(x))


def save_checkpoint(path, cfg, weights, opt_state, key, losses, epoch, arch, train_cfg):
    payload = {
        "cfg": cfg,
        "weights": tree_to_numpy(weights),
        "opt_state": tree_to_numpy(opt_state),
        "rng_key": np.asarray(key),
        "losses": np.asarray(losses, dtype=np.float64),
        "epoch": int(epoch),
        "arch": arch,
        "train": train_cfg,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["weights"] = tree_to_jax(payload["weights"])
    payload["opt_state"] = tree_to_jax(payload["opt_state"])
    payload["rng_key"] = jnp.asarray(payload["rng_key"], dtype=jnp.uint32)
    payload["losses"] = payload.get("losses", np.array([], dtype=np.float64)).tolist()
    return payload


def validate(cfg, weights, theory, key, batch_size, super_batch, tag):
    diffs = []
    for _ in range(super_batch):
        key, k = jax.random.split(key)
        z = rg_checkerboard_inblock_flow_prior_sample(k, cfg, batch_size)
        x = rg_checkerboard_inblock_flow_g(cfg, z, weights)
        ds = rg_checkerboard_inblock_flow_log_prob(cfg, x, weights) + theory.action(x)
        ds = np.asarray(ds)
        ds = ds[np.isfinite(ds)]
        if ds.size:
            diffs.append(ds)
    if not diffs:
        print("Validation: no finite deltaS samples")
        return key

    diff = np.concatenate(diffs)
    m_diff = diff.mean()
    centered = diff - m_diff
    print("max  action diff:", np.max(np.abs(centered)))
    print("min  action diff:", np.min(np.abs(centered)))
    print("mean action diff:", m_diff)
    print("std  action diff:", np.std(centered))

    logw = -centered
    logw_shift = logw - np.max(logw)
    foo = np.exp(logw_shift)
    w = foo / np.mean(foo)
    print("mean re-weighting factor:", np.mean(w))
    print("std  re-weighting factor:", np.std(w))
    ess = (np.mean(foo) ** 2) / np.mean(foo * foo)
    print("ESS:", ess)

    logbins = np.logspace(np.log10(1e-3), np.log10(1e3), max(20, int(w.shape[0] / 10)))
    plt.hist(w, bins=logbins)
    plt.xscale("log")
    plt.title("Reweighting factor")
    plt.savefig(f"phi4_rg_checkerboard_inblock_rw_{tag}.pdf")
    plt.close()

    plt.hist(centered, bins=max(20, int(w.shape[0] / 10)))
    plt.title("DeltaS distribution")
    plt.savefig(f"phi4_rg_checkerboard_inblock_ds_{tag}.pdf")
    plt.close()
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="", help="resume checkpoint (.pkl)")
    ap.add_argument("--save", type=str, default="rg_checkerboard_inblock_flow_phi4_ckpt.pkl")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-batch", type=int, default=256)
    ap.add_argument("--validate-super", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=-0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=1)
    ap.add_argument("--n-inner-couplings", type=int, default=3)
    ap.add_argument("--conditioner", type=str, default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--transformer-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--attn-radius", type=int, default=1)
    ap.add_argument("--mlp-dim", type=int, default=0, help="0 -> use 2*width")
    ap.add_argument("--terminal-n-layers", type=int, default=2)
    ap.add_argument("--terminal-width", type=int, default=0, help="0 -> use width")
    ap.add_argument("--output-init-scale", type=float, default=1e-2)
    ap.add_argument("--parity", type=str, default="sym", choices=["none", "sym"])
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    default_save = "rg_checkerboard_inblock_flow_phi4_ckpt.pkl"

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    key = jax.random.PRNGKey(args.seed)
    losses = []
    start_epoch = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume)
        arch = ckpt["arch"]
        train_cfg = ckpt.get("train", {})
        key = ckpt["rng_key"]
        losses = list(ckpt["losses"])
        start_epoch = int(ckpt.get("epoch", 0))
        theory = Phi4([arch["L"], arch["L"]], arch["lam"], arch["mass"], batch_size=train_cfg.get("batch", args.batch))
        model = init_rg_checkerboard_inblock_flow(
            key,
            size=(arch["L"], arch["L"]),
            width=arch["width"],
            n_cycles=arch.get("n_cycles", 1),
            n_inner_couplings=arch.get("n_inner_couplings", 3),
            conditioner=arch.get("conditioner", "transformer"),
            transformer_layers=arch["transformer_layers"],
            n_heads=arch["n_heads"],
            attn_radius=arch["attn_radius"],
            mlp_dim=arch["mlp_dim"],
            rg_type=arch["rg_type"],
            log_scale_clip=arch["log_scale_clip"],
            terminal_n_layers=arch.get("terminal_n_layers", 2),
            terminal_width=arch.get("terminal_width", arch["width"]),
            output_init_scale=arch.get("output_init_scale", 1e-2),
            parity=arch.get("parity", "sym"),
        )
        cfg = model["cfg"]
        weights = ckpt["weights"]
        lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 3e-4))
        batch_size = int(args.batch if args.batch is not None else train_cfg.get("batch", 8))
        opt = optax.adam(lr)
        opt_state = ckpt["opt_state"]
        if args.save == default_save:
            args.save = args.resume
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        key, k_model = jax.random.split(key)
        model = init_rg_checkerboard_inblock_flow(
            k_model,
            size=(args.L, args.L),
            width=args.width,
            n_cycles=args.n_cycles,
            n_inner_couplings=args.n_inner_couplings,
            conditioner=args.conditioner,
            transformer_layers=args.transformer_layers,
            n_heads=args.n_heads,
            attn_radius=args.attn_radius,
            mlp_dim=None if args.mlp_dim <= 0 else args.mlp_dim,
            rg_type=args.rg_type,
            log_scale_clip=args.log_scale_clip,
            terminal_n_layers=args.terminal_n_layers,
            terminal_width=None if args.terminal_width <= 0 else args.terminal_width,
            output_init_scale=args.output_init_scale,
            parity=args.parity,
        )
        cfg = model["cfg"]
        weights = model["weights"]
        batch_size = args.batch
        lr = args.lr
        theory = Phi4([args.L, args.L], args.lam, args.mass, batch_size=batch_size)
        opt = optax.adam(lr)
        opt_state = opt.init(weights)

    @jax.jit
    def step(weights, opt_state, key):
        l, grads = jax.value_and_grad(loss_fn, argnums=1)(cfg, weights, key, batch_size, theory)
        updates, opt_state = opt.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, l

    arch = {
        "L": int(cfg["size_h"]),
        "lam": float(theory.lam),
        "mass": float(theory.mass),
        "width": args.width if not args.resume else int(ckpt["arch"]["width"]),
        "n_cycles": args.n_cycles if not args.resume else int(ckpt["arch"].get("n_cycles", 1)),
        "n_inner_couplings": (
            args.n_inner_couplings if not args.resume else int(ckpt["arch"].get("n_inner_couplings", 3))
        ),
        "conditioner": args.conditioner if not args.resume else str(ckpt["arch"].get("conditioner", "transformer")),
        "transformer_layers": args.transformer_layers if not args.resume else int(ckpt["arch"]["transformer_layers"]),
        "n_heads": args.n_heads if not args.resume else int(ckpt["arch"]["n_heads"]),
        "attn_radius": args.attn_radius if not args.resume else int(ckpt["arch"]["attn_radius"]),
        "mlp_dim": (
            int((2 * args.width) if args.mlp_dim <= 0 else args.mlp_dim)
            if not args.resume
            else int(ckpt["arch"]["mlp_dim"])
        ),
        "terminal_n_layers": (
            int(args.terminal_n_layers)
            if not args.resume
            else int(ckpt["arch"].get("terminal_n_layers", 2))
        ),
        "terminal_width": (
            int(args.width if args.terminal_width <= 0 else args.terminal_width)
            if not args.resume
            else int(ckpt["arch"].get("terminal_width", ckpt["arch"]["width"]))
        ),
        "output_init_scale": (
            float(args.output_init_scale)
            if not args.resume
            else float(ckpt["arch"].get("output_init_scale", 1e-2))
        ),
        "parity": args.parity if not args.resume else str(ckpt["arch"].get("parity", "sym")),
        "rg_type": args.rg_type if not args.resume else ckpt["arch"]["rg_type"],
        "log_scale_clip": float(args.log_scale_clip if not args.resume else ckpt["arch"]["log_scale_clip"]),
    }
    train_cfg = {"lr": float(lr), "batch": int(batch_size)}

    remaining_epochs = max(0, int(args.epochs) - int(start_epoch))
    print("Run config:")
    print(f"  mode: {'resume' if args.resume else 'fresh'}")
    print(f"  checkpoint_in: {args.resume if args.resume else '<none>'}")
    print(f"  checkpoint_out: {args.save}")
    print(f"  start_epoch: {start_epoch}")
    print(f"  target_epochs: {args.epochs}")
    print(f"  remaining_epochs: {remaining_epochs}")
    print(f"  batch: {batch_size}")
    print(f"  lr: {lr}")
    print(
        "  model:"
        f" L={arch['L']} mass={arch['mass']} lam={arch['lam']}"
        f" width={arch['width']} n_cycles={arch['n_cycles']}"
        f" n_inner_couplings={arch['n_inner_couplings']}"
        f" conditioner={arch['conditioner']}"
        f" transformer_layers={arch['transformer_layers']}"
        f" n_heads={arch['n_heads']} attn_radius={arch['attn_radius']}"
        f" mlp_dim={arch['mlp_dim']}"
        f" terminal_layers={arch['terminal_n_layers']}"
        f" terminal_width={arch['terminal_width']}"
        f" output_init_scale={arch['output_init_scale']}"
        f" parity={arch['parity']}"
        f" rg={arch['rg_type']}"
        f" log_scale_clip={arch['log_scale_clip']}"
    )
    print(f"  validate: {args.validate} (batch={args.validate_batch}, super={args.validate_super})")

    pbar = tqdm(range(start_epoch, int(args.epochs)), desc="Training(RG Checkerboard In-Block Flow)")
    tic = time.perf_counter()
    for ep in pbar:
        key, k = jax.random.split(key)
        weights, opt_state, lv = step(weights, opt_state, k)
        losses.append(float(lv))
        pbar.set_postfix(loss=f"{float(lv):.6f}")

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            save_checkpoint(args.save, cfg, weights, opt_state, key, losses, ep + 1, arch, train_cfg)

    toc = time.perf_counter()
    print(f"training time: {toc - tic:.2f}s")
    save_checkpoint(args.save, cfg, weights, opt_state, key, losses, int(args.epochs), arch, train_cfg)
    print(f"saved {args.save}")
    print(f"final loss {losses[-1] if losses else 'n/a'}")

    if args.validate:
        validate(cfg, weights, theory, key, args.validate_batch, args.validate_super, tag="final")


if __name__ == "__main__":
    main()

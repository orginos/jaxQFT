#!/usr/bin/env python3
"""Train a single MGFlow model for phi^4 (JAX) with resume/validation."""

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
from jaxqft.models.phi4_mg import init_mgflow, mgflow_g, mgflow_log_prob, mgflow_prior_sample


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def loss_fn(cfg, weights, key, batch_size, theory):
    z = mgflow_prior_sample(key, cfg, batch_size)
    x = mgflow_g(cfg, z, weights)
    return jnp.mean(mgflow_log_prob(cfg, x, weights) + theory.action(x))


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
        z = mgflow_prior_sample(k, cfg, batch_size)
        x = mgflow_g(cfg, z, weights)
        ds = mgflow_log_prob(cfg, x, weights) + theory.action(x)
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
    plt.savefig(f"phi4_mg_rw_{tag}.pdf")
    plt.close()

    plt.hist(centered, bins=max(20, int(w.shape[0] / 10)))
    plt.title("DeltaS distribution")
    plt.savefig(f"phi4_mg_ds_{tag}.pdf")
    plt.close()
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="", help="resume checkpoint (.pkl)")
    ap.add_argument("--save", type=str, default="mgflow_single_jax_ckpt.pkl")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-batch", type=int, default=256)
    ap.add_argument("--validate-super", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=-0.5)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--parity", type=str, default="none", choices=["none", "sym", "x2"])
    ap.add_argument("--fixed-bijector", action="store_true")
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    default_save = "mgflow_single_jax_ckpt.pkl"

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
        model = init_mgflow(
            key,
            size=(arch["L"], arch["L"]),
            n_layers=arch["n_layers"],
            width=arch["width"],
            nconvs=arch["n_convs"],
            rg_type=arch["rg_type"],
            log_scale_clip=arch["log_scale_clip"],
            parity=arch["parity"],
            fixed_bijector=arch["fixed_bijector"],
        )
        cfg = model["cfg"]
        weights = ckpt["weights"]
        # On resume, use current CLI hyperparameters as overrides.
        lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
        batch_size = int(args.batch if args.batch is not None else train_cfg.get("batch", 8))
        opt = optax.adam(lr)
        opt_state = ckpt["opt_state"]
        if args.save == default_save:
            args.save = args.resume
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        key, k_model = jax.random.split(key)
        model = init_mgflow(
            k_model,
            size=(args.L, args.L),
            n_layers=args.n_layers,
            width=args.width,
            nconvs=args.n_convs,
            rg_type=args.rg_type,
            log_scale_clip=args.log_scale_clip,
            parity=args.parity,
            fixed_bijector=args.fixed_bijector,
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
        "n_layers": args.n_layers if not args.resume else int(ckpt["arch"]["n_layers"]),
        "width": args.width if not args.resume else int(ckpt["arch"]["width"]),
        "n_convs": args.n_convs if not args.resume else int(ckpt["arch"]["n_convs"]),
        "rg_type": args.rg_type if not args.resume else ckpt["arch"]["rg_type"],
        "parity": args.parity if not args.resume else ckpt["arch"]["parity"],
        "fixed_bijector": bool(args.fixed_bijector if not args.resume else ckpt["arch"]["fixed_bijector"]),
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
    print(f"  validate: {args.validate} (batch={args.validate_batch}, super={args.validate_super})")
    print(
        "  model:"
        f" L={arch['L']} mass={arch['mass']} lam={arch['lam']}"
        f" layers={arch['n_layers']} width={arch['width']}"
        f" n_convs={arch['n_convs']} rg={arch['rg_type']}"
        f" parity={arch['parity']} fixed_bijector={arch['fixed_bijector']}"
        f" log_scale_clip={arch['log_scale_clip']}"
    )

    pbar = tqdm(range(start_epoch, args.epochs), desc="Training(MG single)")
    tic = time.perf_counter()
    for ep in pbar:
        key, k = jax.random.split(key)
        weights, opt_state, l = step(weights, opt_state, k)
        lv = float(l)
        losses.append(lv)
        pbar.set_postfix(loss=f"{lv:.6f}")

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            save_checkpoint(args.save, cfg, weights, opt_state, key, losses, ep + 1, arch, train_cfg)

        if args.validate and (ep + 1) % max(1, args.save_every if args.save_every > 0 else 100) == 0:
            tag = f"L{arch['L']}_m{arch['mass']}_l{arch['lam']}_ep{ep+1}"
            key = validate(cfg, weights, theory, key, args.validate_batch, args.validate_super, tag)

    toc = time.perf_counter()
    print(f"training time: {toc - tic:.2f}s")
    save_checkpoint(args.save, cfg, weights, opt_state, key, losses, args.epochs, arch, train_cfg)
    print("saved", args.save)
    print("final loss", losses[-1] if losses else float("nan"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train stacked MGFlow model for phi^4 (JAX) with warmup/joint/SWA/resume."""

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
from jaxqft.models.stacked_mg import init_stacked_mg, stacked_g, stacked_log_prob, stacked_prior_sample


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def loss_fn(cfg, weights, key, batch_size, theory):
    z = stacked_prior_sample(key, cfg, batch_size)
    x = stacked_g(cfg, z, weights)
    return jnp.mean(stacked_log_prob(cfg, x, weights) + theory.action(x))


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
        z = stacked_prior_sample(k, cfg, batch_size)
        x = stacked_g(cfg, z, weights)
        ds = stacked_log_prob(cfg, x, weights) + theory.action(x)
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
    plt.savefig(f"stacked_mg_rw_{tag}.pdf")
    plt.close()

    plt.hist(centered, bins=max(20, int(w.shape[0] / 10)))
    plt.title("DeltaS distribution")
    plt.savefig(f"stacked_mg_ds_{tag}.pdf")
    plt.close()
    return key


def zero_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def add_tree(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def scale_tree(a, c):
    return jax.tree_util.tree_map(lambda x: x * c, a)


def mask_grads_to_stage(grads, stage_idx):
    masked = dict(grads)
    out_stages = []
    for i, g in enumerate(grads["stages"]):
        out_stages.append(g if i == stage_idx else zero_like_tree(g))
    masked["stages"] = out_stages
    return masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="", help="resume checkpoint (.pkl)")
    ap.add_argument("--save", type=str, default="stacked_mg_jax_ckpt.pkl")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-batch", type=int, default=256)
    ap.add_argument("--validate-super", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mass", type=float, default=-0.2)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--parity", type=str, default="none", choices=["none", "sym", "x2"])
    ap.add_argument("--fixed-bijector", action="store_true")
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=400, help="fallback total epochs if warmup/joint are zero")
    ap.add_argument("--warmup", type=int, default=0, help="epochs per stage for warmup")
    ap.add_argument("--joint", type=int, default=0, help="joint-training epochs after warmup")
    ap.add_argument("--superbatch", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swa-epochs", type=int, default=0)
    ap.add_argument("--swa-start-frac", type=float, default=0.5)
    ap.add_argument("--swa-lr-mult", type=float, default=0.1)
    args = ap.parse_args()
    default_save = "stacked_mg_jax_ckpt.pkl"

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
        model = init_stacked_mg(
            key,
            stages=arch["stages"],
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
        superbatch = int(args.superbatch if args.superbatch is not None else train_cfg.get("superbatch", 1))
        warmup = int(args.warmup if args.warmup is not None else train_cfg.get("warmup", 0))
        joint = int(args.joint if args.joint is not None else train_cfg.get("joint", 0))
        stages = int(arch["stages"])
        opt = optax.adamw(lr, b1=0.9, b2=0.99)
        opt_state = ckpt["opt_state"]
        if args.save == default_save:
            args.save = args.resume
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        key, k_model = jax.random.split(key)
        model = init_stacked_mg(
            k_model,
            stages=args.stages,
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
        batch_size = int(args.batch)
        superbatch = int(args.superbatch)
        lr = float(args.lr)
        warmup = int(args.warmup)
        joint = int(args.joint)
        stages = int(args.stages)
        theory = Phi4([args.L, args.L], args.lam, args.mass, batch_size=batch_size)
        opt = optax.adamw(lr, b1=0.9, b2=0.99)
        opt_state = opt.init(weights)

    @jax.jit
    def grad_step(weights, key):
        return jax.value_and_grad(loss_fn, argnums=1)(cfg, weights, key, batch_size, theory)

    def do_epoch(weights, opt_state, key, active_stage: int | None):
        total_loss = 0.0
        total_grads = None
        for _ in range(max(1, superbatch)):
            key, km = jax.random.split(key)
            l, g = grad_step(weights, km)
            total_loss += float(l)
            if active_stage is not None:
                g = mask_grads_to_stage(g, active_stage)
            total_grads = g if total_grads is None else add_tree(total_grads, g)
        mean_grads = scale_tree(total_grads, 1.0 / max(1, superbatch))
        updates, opt_state = opt.update(mean_grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, key, total_loss / max(1, superbatch)

    phase_epochs = []
    if warmup > 0:
        for s in range(stages):
            phase_epochs += [("warmup", s)] * warmup
    if joint > 0:
        phase_epochs += [("joint", None)] * joint
    if not phase_epochs:
        phase_epochs = [("joint", None)] * int(args.epochs)

    # resume support for phase schedule
    if start_epoch > len(phase_epochs):
        start_epoch = len(phase_epochs)

    arch = {
        "L": int(cfg["size"][0]),
        "stages": int(len(cfg["stage_cfgs"])),
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
    train_cfg = {"lr": lr, "batch": batch_size, "superbatch": superbatch, "warmup": warmup, "joint": joint}

    remaining_epochs = max(0, len(phase_epochs) - int(start_epoch))
    print("Run config:")
    print(f"  mode: {'resume' if args.resume else 'fresh'}")
    print(f"  checkpoint_in: {args.resume if args.resume else '<none>'}")
    print(f"  checkpoint_out: {args.save}")
    print(f"  start_epoch: {start_epoch}")
    print(f"  schedule_total_epochs: {len(phase_epochs)}")
    print(f"  remaining_epochs: {remaining_epochs}")
    print(f"  batch: {batch_size}")
    print(f"  superbatch: {superbatch}")
    print(f"  lr: {lr}")
    print(f"  warmup_per_stage: {warmup}")
    print(f"  joint_epochs: {joint}")
    print(
        "  swa:"
        f" epochs={args.swa_epochs}"
        f" start_frac={args.swa_start_frac}"
        f" lr_mult={args.swa_lr_mult}"
    )
    print(f"  validate: {args.validate} (batch={args.validate_batch}, super={args.validate_super})")
    print(
        "  model:"
        f" L={arch['L']} stages={arch['stages']}"
        f" mass={arch['mass']} lam={arch['lam']}"
        f" layers={arch['n_layers']} width={arch['width']}"
        f" n_convs={arch['n_convs']} rg={arch['rg_type']}"
        f" parity={arch['parity']} fixed_bijector={arch['fixed_bijector']}"
        f" log_scale_clip={arch['log_scale_clip']}"
    )

    pbar = tqdm(range(start_epoch, len(phase_epochs)), desc="Training(Stacked MG)")
    tic = time.perf_counter()
    for ep in pbar:
        phase, stage_idx = phase_epochs[ep]
        active_stage = stage_idx if phase == "warmup" else None
        weights, opt_state, key, lv = do_epoch(weights, opt_state, key, active_stage)
        losses.append(float(lv))
        pbar.set_postfix(loss=f"{lv:.6f}", phase=phase, stage=stage_idx if stage_idx is not None else -1)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            save_checkpoint(args.save, cfg, weights, opt_state, key, losses, ep + 1, arch, train_cfg)

        if args.validate and (ep + 1) % max(1, args.save_every if args.save_every > 0 else 100) == 0:
            tag = f"L{arch['L']}_S{arch['stages']}_m{arch['mass']}_l{arch['lam']}_ep{ep+1}"
            key = validate(cfg, weights, theory, key, args.validate_batch, args.validate_super, tag)

    # Optional SWA-style fine-tuning and averaging
    if args.swa_epochs > 0:
        swa_lr = max(1e-12, lr * float(args.swa_lr_mult))
        swa_opt = optax.adamw(swa_lr, b1=0.9, b2=0.99)
        swa_state = swa_opt.init(weights)
        avg_weights = tree_to_jax(tree_to_numpy(weights))
        n_avg = 1
        swa_start = int(max(0, round(args.swa_start_frac * args.swa_epochs)))
        pbar_swa = tqdm(range(args.swa_epochs), desc="Training(SWA)")
        for ep in pbar_swa:
            total_loss = 0.0
            total_grads = None
            for _ in range(max(1, superbatch)):
                key, km = jax.random.split(key)
                l, g = grad_step(weights, km)
                total_loss += float(l)
                total_grads = g if total_grads is None else add_tree(total_grads, g)
            mean_grads = scale_tree(total_grads, 1.0 / max(1, superbatch))
            updates, swa_state = swa_opt.update(mean_grads, swa_state, weights)
            weights = optax.apply_updates(weights, updates)
            lv = total_loss / max(1, superbatch)
            losses.append(float(lv))
            pbar_swa.set_postfix(loss=f"{lv:.6f}")
            if ep >= swa_start:
                n_avg += 1
                a = 1.0 / n_avg
                avg_weights = jax.tree_util.tree_map(lambda m, w: (1.0 - a) * m + a * w, avg_weights, weights)
        weights = avg_weights

    toc = time.perf_counter()
    print(f"training time: {toc - tic:.2f}s")
    save_checkpoint(args.save, cfg, weights, opt_state, key, losses, len(phase_epochs), arch, train_cfg)
    print("saved", args.save)
    print("final loss", losses[-1] if losses else float("nan"))


if __name__ == "__main__":
    main()

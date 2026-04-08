#!/usr/bin/env python3
"""Train the RG coarse-eta Gaussian flow for phi^4."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

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
from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
    init_rg_coarse_eta_gaussian_flow,
    rg_coarse_eta_gaussian_flow_g,
    rg_coarse_eta_gaussian_flow_log_prob,
    rg_coarse_eta_gaussian_flow_prior_sample,
)


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def loss_fn(cfg, weights, key, batch_size, theory):
    z = rg_coarse_eta_gaussian_flow_prior_sample(key, cfg, batch_size)
    x = rg_coarse_eta_gaussian_flow_g(cfg, z, weights)
    return jnp.mean(rg_coarse_eta_gaussian_flow_log_prob(cfg, x, weights) + theory.action(x))


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
        z = rg_coarse_eta_gaussian_flow_prior_sample(k, cfg, batch_size)
        x = rg_coarse_eta_gaussian_flow_g(cfg, z, weights)
        ds = rg_coarse_eta_gaussian_flow_log_prob(cfg, x, weights) + theory.action(x)
        ds = np.asarray(ds)
        ds = ds[np.isfinite(ds)]
        if ds.size:
            diffs.append(ds)
    if not diffs:
        print("Validation: no finite deltaS samples")
        return

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
    plt.savefig(f"phi4_rg_coarse_eta_gaussian_rw_{tag}.pdf")
    plt.close()

    plt.hist(centered, bins=max(20, int(w.shape[0] / 10)))
    plt.title("DeltaS distribution")
    plt.savefig(f"phi4_rg_coarse_eta_gaussian_ds_{tag}.pdf")
    plt.close()
    return


def _maybe_int_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [int(v.strip()) for v in text.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return [int(v) for v in value]
    raise ValueError(f"Expected list-like value, got {type(value).__name__}")


def _load_toml_defaults(path: str, *, cli_resume: str = ""):
    if tomllib is None:
        raise RuntimeError("tomllib is not available in this Python")
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    mode = str(raw.get("mode", "")).strip().lower()
    io_cfg = raw.get("io", {})
    physics = raw.get("physics", {})
    model = raw.get("model", {})
    train = raw.get("train", {})
    validation = raw.get("validation", {})

    if mode not in ("", "fresh", "resume"):
        raise ValueError(f"Unsupported mode in {path}: {mode}")
    if mode == "fresh" and io_cfg.get("resume"):
        raise ValueError(f"{path}: fresh mode should not set io.resume")
    if mode == "resume" and not (io_cfg.get("resume") or cli_resume):
        raise ValueError(f"{path}: resume mode requires io.resume")

    defaults = {}

    def put(name, value):
        if value is not None:
            defaults[name] = value

    put("resume", io_cfg.get("resume"))
    put("save", io_cfg.get("save"))
    put("save_every", io_cfg.get("save_every"))

    put("L", physics.get("L"))
    put("lam", physics.get("lam"))
    put("mass", physics.get("mass"))

    for key in (
        "width",
        "n_cycles",
        "radius",
        "eta_gaussian",
        "gaussian_radius",
        "gaussian_width",
        "terminal_prior",
        "terminal_n_layers",
        "terminal_width",
        "output_init_scale",
        "parity",
        "rg_type",
        "log_scale_clip",
        "offdiag_clip",
    ):
        put(key, model.get(key))

    for key in (
        "width_levels",
        "n_cycles_levels",
        "radius_levels",
        "gaussian_radius_levels",
        "gaussian_width_levels",
    ):
        put(key, _maybe_int_list(model.get(key)))

    put("epochs", train.get("epochs"))
    put("batch", train.get("batch"))
    put("lr", train.get("lr"))
    put("seed", train.get("seed"))

    if "enabled" in validation:
        defaults["validate"] = bool(validation.get("enabled"))
    if "each_stage" in validation:
        defaults["validate_each_stage"] = bool(validation.get("each_stage"))
    put("validate_batch", validation.get("batch"))
    put("validate_super", validation.get("super"))
    return defaults


def _load_toml_raw(path: str):
    if tomllib is None:
        raise RuntimeError("tomllib is not available in this Python")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _normalize_schedule(schedule_rows):
    out = []
    prev_end = 0
    for idx, row in enumerate(schedule_rows):
        epoch_end = int(row["epoch_end"])
        batch = int(row["batch"])
        lr = float(row["lr"])
        if epoch_end <= prev_end:
            raise ValueError(f"Schedule epoch_end must increase strictly at stage {idx}")
        if batch <= 0:
            raise ValueError(f"Schedule batch must be positive at stage {idx}")
        if lr <= 0.0:
            raise ValueError(f"Schedule lr must be positive at stage {idx}")
        label = str(row.get("label", f"stage{idx+1}"))
        out.append({"epoch_end": epoch_end, "batch": batch, "lr": lr, "label": label})
        prev_end = epoch_end
    return out


def _build_schedule_from_config(raw_cfg, *, fallback_epochs, fallback_batch, fallback_lr):
    schedule_cfg = raw_cfg.get("schedule", {})
    if not schedule_cfg:
        return None

    explicit = schedule_cfg.get("stage", [])
    if explicit:
        return _normalize_schedule(explicit)

    stages = []
    ramp = schedule_cfg.get("ramp", {})
    anneal = schedule_cfg.get("anneal", {})
    current_end = 0
    last_batch = int(fallback_batch)
    last_lr = float(fallback_lr)

    if ramp:
        initial_batch = int(ramp["initial_batch"])
        epochs_per_stage = int(ramp["epochs_per_stage"])
        num_doubles = int(ramp["num_doubles"])
        ramp_lr = float(ramp.get("lr", fallback_lr))
        current_end = int(ramp.get("start_epoch", 0))
        last_batch = initial_batch
        last_lr = ramp_lr
        for idx in range(num_doubles + 1):
            current_end += epochs_per_stage
            batch = initial_batch * (2**idx)
            stages.append(
                {
                    "epoch_end": current_end,
                    "batch": batch,
                    "lr": ramp_lr,
                    "label": f"ramp_{batch}",
                }
            )
            last_batch = batch

    if anneal:
        epoch_ends = [int(v) for v in anneal.get("epoch_ends", [])]
        lrs = [float(v) for v in anneal.get("lrs", [])]
        if len(epoch_ends) != len(lrs):
            raise ValueError("schedule.anneal epoch_ends and lrs must have the same length")
        batch_value = anneal.get("batch", last_batch)
        if isinstance(batch_value, list):
            batches = [int(v) for v in batch_value]
            if len(batches) != len(epoch_ends):
                raise ValueError("schedule.anneal batch list must match epoch_ends length")
        else:
            batches = [int(batch_value) for _ in epoch_ends]
        for idx, (epoch_end, lr, batch) in enumerate(zip(epoch_ends, lrs, batches)):
            stages.append(
                {
                    "epoch_end": int(epoch_end),
                    "batch": int(batch),
                    "lr": float(lr),
                    "label": f"anneal_{idx+1}",
                }
            )

    if stages:
        return _normalize_schedule(stages)
    return None


def _build_arch_from_cfg(cfg, theory):
    arch_cfg = cfg["arch"]
    return {
        "L": int(cfg["size_h"]),
        "lam": float(theory.lam),
        "mass": float(theory.mass),
        "width": int(arch_cfg["width"]),
        "width_levels": list(int(v) for v in arch_cfg.get("width_levels", (arch_cfg["width"],))),
        "n_cycles": int(arch_cfg["n_cycles"]),
        "n_cycles_levels": list(int(v) for v in arch_cfg.get("n_cycles_levels", (arch_cfg["n_cycles"],))),
        "radius": int(arch_cfg["radius"]),
        "radius_levels": list(int(v) for v in arch_cfg.get("radius_levels", (arch_cfg["radius"],))),
        "eta_gaussian": str(arch_cfg["eta_gaussian"]),
        "gaussian_radius": int(arch_cfg["gaussian_radius"]),
        "gaussian_radius_levels": list(
            int(v) for v in arch_cfg.get("gaussian_radius_levels", (arch_cfg["gaussian_radius"],))
        ),
        "gaussian_width": int(arch_cfg["gaussian_width"]),
        "gaussian_width_levels": list(
            int(v) for v in arch_cfg.get("gaussian_width_levels", (arch_cfg["gaussian_width"],))
        ),
        "terminal_prior": str(arch_cfg["terminal_prior"]),
        "terminal_n_layers": int(arch_cfg["terminal_n_layers"]),
        "terminal_width": int(arch_cfg["terminal_width"]),
        "output_init_scale": float(arch_cfg["output_init_scale"]),
        "parity": str(arch_cfg["parity"]),
        "rg_type": str(arch_cfg["rg_type"]),
        "log_scale_clip": float(cfg["log_scale_clip"]),
        "offdiag_clip": float(cfg["offdiag_clip"]),
    }


def _schedule_to_text(values):
    vals = [int(v) for v in values]
    if not vals:
        return "[]"
    if all(v == vals[0] for v in vals):
        return str(vals[0])
    return "[" + ", ".join(str(v) for v in vals) + "]"


def _make_step(cfg, theory, batch_size, lr):
    opt = optax.adam(float(lr))

    @jax.jit
    def step(weights, opt_state, key):
        lval, grads = jax.value_and_grad(loss_fn, argnums=1)(cfg, weights, key, batch_size, theory)
        updates, opt_state = opt.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, lval

    return step, opt


def _write_nonfinite_marker(save_path: str, *, epoch: int, stage_label: str | None, loss_value: float):
    marker = {
        "error": "nonfinite_loss",
        "epoch": int(epoch),
        "stage": stage_label if stage_label is not None else "",
        "loss": float(loss_value),
    }
    marker_path = f"{save_path}.nonfinite.json"
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"wrote nonfinite marker {marker_path}")


def _check_finite_loss(*, loss_value: float, epoch: int, stage_label: str | None, save_path: str):
    if math.isfinite(float(loss_value)):
        return
    _write_nonfinite_marker(save_path, epoch=epoch, stage_label=stage_label, loss_value=float(loss_value))
    stage_text = f" stage={stage_label}" if stage_label else ""
    raise RuntimeError(f"Non-finite loss detected at epoch {epoch}{stage_text}: loss={loss_value}")


def _schedule_stage_for_epoch(schedule, epoch):
    for stage in schedule:
        if epoch < stage["epoch_end"]:
            return stage
    return schedule[-1]


def main():
    boot = argparse.ArgumentParser(add_help=False)
    boot.add_argument("--config", type=str, default="", help="TOML run configuration")
    boot.add_argument("--resume", type=str, default="", help="resume checkpoint (.pkl)")
    boot_args, _ = boot.parse_known_args()
    config_defaults = (
        _load_toml_defaults(boot_args.config, cli_resume=boot_args.resume) if boot_args.config else {}
    )
    raw_cfg = _load_toml_raw(boot_args.config) if boot_args.config else {}

    ap = argparse.ArgumentParser(parents=[boot])
    ap.add_argument("--save", type=str, default="rg_coarse_eta_gaussian_flow_phi4_ckpt.pkl")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="load the checkpoint/config, run validation, and exit without training",
    )
    ap.add_argument(
        "--validate-each-stage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when using a schedule, run validation at each stage boundary",
    )
    ap.add_argument("--validate-batch", type=int, default=256)
    ap.add_argument("--validate-super", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=-0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--width-levels", type=str, default=None, help="comma-separated per-level widths, finest->coarsest")
    ap.add_argument("--n-cycles", type=int, default=2)
    ap.add_argument(
        "--n-cycles-levels",
        type=str,
        default=None,
        help="comma-separated per-level cycle counts, finest->coarsest",
    )
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--radius-levels", type=str, default=None, help="comma-separated per-level radii, finest->coarsest")
    ap.add_argument("--eta-gaussian", type=str, default="coarse_patch", choices=["none", "level", "coarse_patch"])
    ap.add_argument("--gaussian-radius", type=int, default=-1, help="-1 -> use radius")
    ap.add_argument(
        "--gaussian-radius-levels",
        type=str,
        default=None,
        help="comma-separated per-level Gaussian radii, finest->coarsest",
    )
    ap.add_argument("--gaussian-width", type=int, default=0, help="0 -> use width")
    ap.add_argument(
        "--gaussian-width-levels",
        type=str,
        default=None,
        help="comma-separated per-level Gaussian widths, finest->coarsest",
    )
    ap.add_argument("--terminal-prior", type=str, default="learned", choices=["std", "learned"])
    ap.add_argument("--terminal-n-layers", type=int, default=2)
    ap.add_argument("--terminal-width", type=int, default=0, help="0 -> use width")
    ap.add_argument("--output-init-scale", type=float, default=1e-2)
    ap.add_argument("--parity", type=str, default="sym", choices=["none", "sym"])
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--offdiag-clip", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fail-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="abort immediately when the training loss becomes NaN/Inf",
    )
    ap.set_defaults(**config_defaults)
    args = ap.parse_args()
    default_save = "rg_coarse_eta_gaussian_flow_phi4_ckpt.pkl"

    width_levels = _maybe_int_list(args.width_levels)
    n_cycles_levels = _maybe_int_list(args.n_cycles_levels)
    radius_levels = _maybe_int_list(args.radius_levels)
    gaussian_radius_levels = _maybe_int_list(args.gaussian_radius_levels)
    gaussian_width_levels = _maybe_int_list(args.gaussian_width_levels)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    key = jax.random.PRNGKey(args.seed)
    losses = []
    start_epoch = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume)
        arch = ckpt["arch"]
        train_prev = ckpt.get("train", {})
        key = ckpt["rng_key"]
        losses = list(ckpt["losses"])
        start_epoch = int(ckpt.get("epoch", 0))
        schedule = _build_schedule_from_config(
            raw_cfg,
            fallback_epochs=int(train_prev.get("epochs", args.epochs)),
            fallback_batch=int(train_prev.get("batch", 8)),
            fallback_lr=float(train_prev.get("lr", 3e-4)),
        )
        if schedule is None:
            schedule = train_prev.get("schedule")
        if schedule:
            schedule = _normalize_schedule(schedule)
            active = _schedule_stage_for_epoch(schedule, start_epoch)
            batch_size = int(active["batch"])
            lr = float(active["lr"])
            target_epochs = int(schedule[-1]["epoch_end"])
        else:
            batch_size = int(train_prev.get("batch", 8) if args.batch is None else args.batch)
            lr = float(train_prev.get("lr", 3e-4) if args.lr is None else args.lr)
            target_epochs = int(args.epochs)
        theory = Phi4([arch["L"], arch["L"]], arch["lam"], arch["mass"], batch_size=batch_size)
        model = init_rg_coarse_eta_gaussian_flow(
            key,
            size=(arch["L"], arch["L"]),
            width=arch["width"],
            width_levels=arch.get("width_levels"),
            n_cycles=arch.get("n_cycles", 2),
            n_cycles_levels=arch.get("n_cycles_levels"),
            radius=arch.get("radius", 1),
            radius_levels=arch.get("radius_levels"),
            eta_gaussian=arch.get("eta_gaussian", "coarse_patch"),
            gaussian_radius=arch.get("gaussian_radius", arch.get("radius", 1)),
            gaussian_radius_levels=arch.get("gaussian_radius_levels"),
            gaussian_width=arch.get("gaussian_width", arch["width"]),
            gaussian_width_levels=arch.get("gaussian_width_levels"),
            terminal_prior=arch.get("terminal_prior", "learned"),
            rg_type=arch["rg_type"],
            log_scale_clip=arch["log_scale_clip"],
            offdiag_clip=arch.get("offdiag_clip", 2.0),
            terminal_n_layers=arch.get("terminal_n_layers", 2),
            terminal_width=arch.get("terminal_width", arch["width"]),
            output_init_scale=arch.get("output_init_scale", 1e-2),
            parity=arch.get("parity", "sym"),
        )
        cfg = model["cfg"]
        weights = ckpt["weights"]
        opt_state = ckpt["opt_state"]
        if args.save == default_save:
            args.save = args.resume
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    else:
        batch_size = 8 if args.batch is None else int(args.batch)
        lr = 3e-4 if args.lr is None else float(args.lr)
        schedule = _build_schedule_from_config(
            raw_cfg,
            fallback_epochs=int(args.epochs),
            fallback_batch=int(batch_size),
            fallback_lr=float(lr),
        )
        if schedule:
            batch_size = int(schedule[0]["batch"])
            lr = float(schedule[0]["lr"])
            target_epochs = int(schedule[-1]["epoch_end"])
        else:
            target_epochs = int(args.epochs)
        key, k_model = jax.random.split(key)
        gaussian_radius = None if args.gaussian_radius < 0 else args.gaussian_radius
        gaussian_width = None if args.gaussian_width <= 0 else args.gaussian_width
        model = init_rg_coarse_eta_gaussian_flow(
            k_model,
            size=(args.L, args.L),
            width=args.width,
            width_levels=width_levels,
            n_cycles=args.n_cycles,
            n_cycles_levels=n_cycles_levels,
            radius=args.radius,
            radius_levels=radius_levels,
            eta_gaussian=args.eta_gaussian,
            gaussian_radius=gaussian_radius,
            gaussian_radius_levels=gaussian_radius_levels,
            gaussian_width=gaussian_width,
            gaussian_width_levels=gaussian_width_levels,
            terminal_prior=args.terminal_prior,
            rg_type=args.rg_type,
            log_scale_clip=args.log_scale_clip,
            offdiag_clip=args.offdiag_clip,
            terminal_n_layers=args.terminal_n_layers,
            terminal_width=None if args.terminal_width <= 0 else args.terminal_width,
            output_init_scale=args.output_init_scale,
            parity=args.parity,
        )
        cfg = model["cfg"]
        weights = model["weights"]
        theory = Phi4([args.L, args.L], args.lam, args.mass, batch_size=batch_size)
        opt_state = optax.adam(lr).init(weights)

    step, _ = _make_step(cfg, theory, batch_size, lr)

    arch = _build_arch_from_cfg(cfg, theory)
    train_cfg = {
        "lr": float(lr),
        "batch": int(batch_size),
        "epochs": int(target_epochs),
        "schedule": schedule,
    }

    remaining_epochs = max(0, int(target_epochs) - int(start_epoch))
    print("Run config:")
    print(f"  mode: {'resume' if args.resume else 'fresh'}")
    print(f"  checkpoint_in: {args.resume if args.resume else '<none>'}")
    print(f"  checkpoint_out: {args.save}")
    print(f"  start_epoch: {start_epoch}")
    print(f"  target_epochs: {target_epochs}")
    print(f"  remaining_epochs: {remaining_epochs}")
    print(f"  validate_only: {args.validate_only}")
    print(f"  batch: {batch_size}")
    print(f"  lr: {lr}")
    if args.config:
        print(f"  config: {args.config}")
    if schedule:
        print("  schedule:")
        for stage in schedule:
            print(
                f"    - {stage['label']}: epoch_end={stage['epoch_end']}"
                f" batch={stage['batch']} lr={stage['lr']}"
            )
    print(
        "  model:"
        f" L={arch['L']} mass={arch['mass']} lam={arch['lam']}"
        f" width={arch['width']}"
        f" width_levels={_schedule_to_text(arch['width_levels'])}"
        f" n_cycles={arch['n_cycles']}"
        f" n_cycles_levels={_schedule_to_text(arch['n_cycles_levels'])}"
        f" radius={arch['radius']}"
        f" radius_levels={_schedule_to_text(arch['radius_levels'])}"
        f" eta_gaussian={arch['eta_gaussian']}"
        f" gaussian_radius={arch['gaussian_radius']}"
        f" gaussian_radius_levels={_schedule_to_text(arch['gaussian_radius_levels'])}"
        f" gaussian_width={arch['gaussian_width']}"
        f" gaussian_width_levels={_schedule_to_text(arch['gaussian_width_levels'])}"
        f" terminal_prior={arch['terminal_prior']}"
        f" terminal_layers={arch['terminal_n_layers']}"
        f" terminal_width={arch['terminal_width']}"
        f" output_init_scale={arch['output_init_scale']}"
        f" parity={arch['parity']}"
        f" rg={arch['rg_type']}"
        f" log_scale_clip={arch['log_scale_clip']}"
        f" offdiag_clip={arch['offdiag_clip']}"
    )
    print(
        f"  validate: {args.validate}"
        f" (batch={args.validate_batch}, super={args.validate_super}, each_stage={args.validate_each_stage})"
    )

    if args.validate_only:
        if not args.validate:
            print("validate_only requested without --validate; enabling validation")
        val_theory = Phi4([arch["L"], arch["L"]], arch["lam"], arch["mass"], batch_size=args.validate_batch)
        val_key = jax.random.fold_in(key, int(start_epoch if start_epoch > 0 else target_epochs))
        validate(cfg, weights, val_theory, val_key, args.validate_batch, args.validate_super, tag="validate_only")
        return

    tic = time.perf_counter()
    if schedule:
        prev_end = 0
        for stage in schedule:
            if stage["epoch_end"] <= start_epoch:
                prev_end = stage["epoch_end"]
                continue
            stage_begin = max(start_epoch, prev_end)
            stage_end = stage["epoch_end"]
            stage_batch = int(stage["batch"])
            stage_lr = float(stage["lr"])
            theory = Phi4([arch["L"], arch["L"]], arch["lam"], arch["mass"], batch_size=stage_batch)
            step, _ = _make_step(cfg, theory, stage_batch, stage_lr)
            train_cfg["lr"] = stage_lr
            train_cfg["batch"] = stage_batch
            pbar = tqdm(
                range(stage_begin, stage_end),
                desc=f"Training(RG Coarse Eta Gaussian Flow:{stage['label']})",
            )
            for ep in pbar:
                key, k = jax.random.split(key)
                weights, opt_state, lv = step(weights, opt_state, k)
                losses.append(float(lv))
                if args.fail_on_nonfinite:
                    _check_finite_loss(
                        loss_value=float(lv),
                        epoch=ep + 1,
                        stage_label=stage["label"],
                        save_path=args.save,
                    )
                pbar.set_postfix(loss=f"{float(lv):.6f}")
                if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                    save_checkpoint(args.save, cfg, weights, opt_state, key, losses, ep + 1, arch, train_cfg)
            if args.validate and args.validate_each_stage:
                print(f"stage validation: {stage['label']} @ epoch {stage_end}")
                val_key = jax.random.fold_in(key, int(stage_end))
                validate(
                    cfg,
                    weights,
                    theory,
                    val_key,
                    args.validate_batch,
                    args.validate_super,
                    tag=stage["label"],
                )
            prev_end = stage_end
    else:
        pbar = tqdm(range(start_epoch, int(target_epochs)), desc="Training(RG Coarse Eta Gaussian Flow)")
        for ep in pbar:
            key, k = jax.random.split(key)
            weights, opt_state, lv = step(weights, opt_state, k)
            losses.append(float(lv))
            if args.fail_on_nonfinite:
                _check_finite_loss(
                    loss_value=float(lv),
                    epoch=ep + 1,
                    stage_label=None,
                    save_path=args.save,
                )
            pbar.set_postfix(loss=f"{float(lv):.6f}")
            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                save_checkpoint(args.save, cfg, weights, opt_state, key, losses, ep + 1, arch, train_cfg)

    toc = time.perf_counter()
    print(f"training time: {toc - tic:.2f}s")
    save_checkpoint(args.save, cfg, weights, opt_state, key, losses, int(target_epochs), arch, train_cfg)
    print(f"saved {args.save}")
    print(f"final loss {losses[-1] if losses else 'n/a'}")

    if args.validate:
        final_key = jax.random.fold_in(key, int(target_epochs))
        validate(cfg, weights, theory, final_key, args.validate_batch, args.validate_super, tag="final")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug trainer for the RG coarse-eta Gaussian phi^4 flow.

This entry point is intentionally separate from the production trainer.  It
keeps the production path untouched while exposing enough diagnostics to
understand non-finite losses on hard training points.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
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


def _load_prod_trainer():
    trainer_path = Path(__file__).with_name("train_rg_coarse_eta_gaussian_flow.py")
    spec = importlib.util.spec_from_file_location("_phi4_rg_prod_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load production trainer from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prod = _load_prod_trainer()

from jaxqft.models.phi4 import Phi4
from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
    init_rg_coarse_eta_gaussian_flow,
    rg_coarse_eta_gaussian_flow_f,
    rg_coarse_eta_gaussian_flow_g_with_ldj,
    rg_coarse_eta_gaussian_flow_prior_log_prob,
    rg_coarse_eta_gaussian_flow_prior_sample,
)


def _safe_summary(arr):
    flat = jnp.ravel(arr)
    finite = jnp.isfinite(flat)
    count = jnp.sum(finite)
    safe = jnp.where(finite, flat, 0.0)
    inf = jnp.asarray(jnp.inf, dtype=flat.dtype)
    ninf = jnp.asarray(-jnp.inf, dtype=flat.dtype)
    abs_safe = jnp.where(finite, jnp.abs(flat), 0.0)
    mean = jnp.where(count > 0, jnp.sum(safe) / count, jnp.nan)
    minv = jnp.where(count > 0, jnp.min(jnp.where(finite, flat, inf)), jnp.nan)
    maxv = jnp.where(count > 0, jnp.max(jnp.where(finite, flat, ninf)), jnp.nan)
    abs_max = jnp.where(count > 0, jnp.max(abs_safe), jnp.nan)
    return {
        "size": jnp.asarray(flat.size, dtype=jnp.int32),
        "finite_count": count.astype(jnp.int32),
        "finite_fraction": jnp.mean(finite.astype(jnp.float32)),
        "mean": mean,
        "min": minv,
        "max": maxv,
        "abs_max": abs_max,
    }


def _diag_batch(cfg, weights, key, batch_size, theory, *, loss_path: str = "forward", do_inverse_check: bool = True):
    z_prior = rg_coarse_eta_gaussian_flow_prior_sample(key, cfg, batch_size)
    x, ldj_forward = rg_coarse_eta_gaussian_flow_g_with_ldj(cfg, z_prior, weights)
    prior_lp_forward = rg_coarse_eta_gaussian_flow_prior_log_prob(z_prior)
    log_q_forward = prior_lp_forward - ldj_forward
    action = theory.action(x)
    loss_vec_forward = log_q_forward + action
    diag = {
        "z_prior": _safe_summary(z_prior),
        "x": _safe_summary(x),
        "ldj_forward": _safe_summary(ldj_forward),
        "prior_log_prob_forward": _safe_summary(prior_lp_forward),
        "log_q_forward": _safe_summary(log_q_forward),
        "action": _safe_summary(action),
        "loss_vec_forward": _safe_summary(loss_vec_forward),
    }
    if do_inverse_check:
        z_inv, ldj_inverse = rg_coarse_eta_gaussian_flow_f(cfg, x, weights)
        prior_lp_inverse = rg_coarse_eta_gaussian_flow_prior_log_prob(z_inv)
        log_q_inverse = prior_lp_inverse + ldj_inverse
        loss_vec_inverse = log_q_inverse + action
        recon = z_inv - z_prior
        diag.update(
            {
                "z_inv": _safe_summary(z_inv),
                "ldj_inverse": _safe_summary(ldj_inverse),
                "prior_log_prob_inverse": _safe_summary(prior_lp_inverse),
                "log_q_inverse": _safe_summary(log_q_inverse),
                "loss_vec_inverse": _safe_summary(loss_vec_inverse),
                "z_recon_abs_max": jnp.max(jnp.abs(recon)),
                "z_recon_rms": jnp.sqrt(jnp.mean(recon * recon)),
            }
        )
    if loss_path == "forward":
        return jnp.mean(loss_vec_forward), diag
    if loss_path == "inverse":
        if not do_inverse_check:
            raise ValueError("inverse loss path requires inverse diagnostics")
        return jnp.mean(loss_vec_inverse), diag
    raise ValueError(f"Unsupported loss_path: {loss_path}")


def _make_step(cfg, theory, batch_size, lr, *, loss_path: str):
    opt = optax.adam(float(lr))
    if loss_path == "forward":
        active_loss_fn = prod.loss_fn_forward
    elif loss_path == "inverse":
        active_loss_fn = prod.loss_fn_inverse
    else:
        raise ValueError(f"Unsupported loss_path: {loss_path}")

    @jax.jit
    def step(weights, opt_state, key):
        lval, grads = jax.value_and_grad(active_loss_fn, argnums=1)(cfg, weights, key, batch_size, theory)
        updates, next_opt_state = opt.update(grads, opt_state, weights)
        next_weights = optax.apply_updates(weights, updates)
        grad_norm = optax.global_norm(grads)
        return next_weights, next_opt_state, lval, grad_norm

    return step, opt


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        arr = np.asarray(obj)
        if arr.shape == ():
            return _json_ready(arr.item())
        return arr.tolist()
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val):
            return "nan"
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_debug_marker(
    path: str,
    *,
    epoch: int,
    stage_label: str | None,
    loss_value: float,
    diag,
):
    marker = {
        "error": "nonfinite_loss_debug",
        "epoch": int(epoch),
        "stage": stage_label if stage_label is not None else "",
        "loss": _json_ready(loss_value),
        "diagnostics": _json_ready(diag),
    }
    marker_path = f"{path}.debug.nonfinite.json"
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"wrote debug nonfinite marker {marker_path}")


def _write_debug_summary(
    path: str,
    *,
    status: str,
    epoch: int,
    stage_label: str | None,
    loss_value: float,
    diag,
):
    marker = {
        "status": status,
        "epoch": int(epoch),
        "stage": stage_label if stage_label is not None else "",
        "loss": _json_ready(loss_value),
        "diagnostics": _json_ready(diag),
    }
    marker_path = f"{path}.debug.summary.json"
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"wrote debug summary {marker_path}")


def _print_diag(epoch: int, stage_label: str | None, loss_value: float, diag) -> None:
    stage_text = stage_label if stage_label else "-"
    dq = diag["log_q_forward"]
    da = diag["action"]
    dl = diag["loss_vec_forward"]
    z_inv_abs_max = diag.get("z_inv", {}).get("abs_max", jnp.nan)
    z_recon_abs_max = diag.get("z_recon_abs_max", jnp.nan)
    print(
        "diag"
        f" epoch={epoch}"
        f" stage={stage_text}"
        f" loss={loss_value}"
        f" logq_mean={float(np.asarray(dq['mean'])):.6g}"
        f" action_mean={float(np.asarray(da['mean'])):.6g}"
        f" loss_finite={float(np.asarray(dl['finite_fraction'])):.3f}"
        f" x_abs_max={float(np.asarray(diag['x']['abs_max'])):.6g}"
        f" ldj_fwd_abs_max={float(np.asarray(diag['ldj_forward']['abs_max'])):.6g}"
        f" z_inv_abs_max={float(np.asarray(z_inv_abs_max)):.6g}"
        f" z_recon_abs_max={float(np.asarray(z_recon_abs_max)):.6g}"
        f" grad_norm={float(np.asarray(diag['grad_norm'])):.6g}"
    )


def main():
    boot = argparse.ArgumentParser(add_help=False)
    boot.add_argument("--config", type=str, default="", help="TOML run configuration")
    boot.add_argument("--resume", type=str, default="", help="resume checkpoint (.pkl)")
    boot_args, _ = boot.parse_known_args()
    config_defaults = prod._load_toml_defaults(boot_args.config, cli_resume=boot_args.resume) if boot_args.config else {}
    raw_cfg = prod._load_toml_raw(boot_args.config) if boot_args.config else {}

    ap = argparse.ArgumentParser(parents=[boot])
    ap.add_argument("--save", type=str, default="rg_coarse_eta_gaussian_flow_phi4_ckpt.pkl")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=-0.5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--width-levels", type=str, default=None)
    ap.add_argument("--n-cycles", type=int, default=2)
    ap.add_argument("--n-cycles-levels", type=str, default=None)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--radius-levels", type=str, default=None)
    ap.add_argument("--eta-gaussian", type=str, default="coarse_patch", choices=["none", "level", "coarse_patch"])
    ap.add_argument("--gaussian-radius", type=int, default=-1)
    ap.add_argument("--gaussian-radius-levels", type=str, default=None)
    ap.add_argument("--gaussian-width", type=int, default=0)
    ap.add_argument("--gaussian-width-levels", type=str, default=None)
    ap.add_argument("--terminal-prior", type=str, default="learned", choices=["std", "learned"])
    ap.add_argument("--terminal-n-layers", type=int, default=2)
    ap.add_argument("--terminal-width", type=int, default=0)
    ap.add_argument("--output-init-scale", type=float, default=1e-2)
    ap.add_argument("--parity", type=str, default="sym", choices=["none", "sym"])
    ap.add_argument("--rg-type", type=str, default="average", choices=["average", "select"])
    ap.add_argument("--log-scale-clip", type=float, default=5.0)
    ap.add_argument("--offdiag-clip", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="if >0, cap the training run at this epoch even when the schedule is longer",
    )
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-path", type=str, default="forward", choices=["forward", "inverse"])
    ap.add_argument("--diag-every", type=int, default=50, help="print diagnostics every N epochs")
    ap.add_argument(
        "--inverse-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the inverse map during debug diagnostics to monitor reconstruction health",
    )
    ap.add_argument(
        "--max-z-recon-abs",
        type=float,
        default=0.0,
        help="if >0, abort when a diagnostic inverse check exceeds this reconstruction threshold",
    )
    ap.add_argument(
        "--save-last-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write a .lastfinite.pkl checkpoint when a non-finite loss is detected",
    )
    ap.set_defaults(**config_defaults)
    args = ap.parse_args()
    default_save = "rg_coarse_eta_gaussian_flow_phi4_ckpt.pkl"

    width_levels = prod._maybe_int_list(args.width_levels)
    n_cycles_levels = prod._maybe_int_list(args.n_cycles_levels)
    radius_levels = prod._maybe_int_list(args.radius_levels)
    gaussian_radius_levels = prod._maybe_int_list(args.gaussian_radius_levels)
    gaussian_width_levels = prod._maybe_int_list(args.gaussian_width_levels)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    key = jax.random.PRNGKey(args.seed)
    losses = []
    start_epoch = 0

    if args.resume:
        ckpt = prod.load_checkpoint(args.resume)
        arch = ckpt["arch"]
        train_prev = ckpt.get("train", {})
        key = ckpt["rng_key"]
        losses = list(ckpt["losses"])
        start_epoch = int(ckpt.get("epoch", 0))
        schedule = prod._build_schedule_from_config(
            raw_cfg,
            fallback_epochs=int(train_prev.get("epochs", args.epochs)),
            fallback_batch=int(train_prev.get("batch", 8)),
            fallback_lr=float(train_prev.get("lr", 3e-4)),
        )
        if schedule is None:
            schedule = train_prev.get("schedule")
        if schedule:
            schedule = prod._normalize_schedule(schedule)
            active = prod._schedule_stage_for_epoch(schedule, start_epoch)
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
        schedule = prod._build_schedule_from_config(
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

    step, _ = _make_step(cfg, theory, batch_size, lr, loss_path=args.loss_path)

    arch = prod._build_arch_from_cfg(cfg, theory)
    train_cfg = {
        "lr": float(lr),
        "batch": int(batch_size),
        "epochs": int(target_epochs),
        "schedule": schedule,
        "loss_path": str(args.loss_path),
    }

    if args.max_epochs > 0:
        target_epochs = min(int(target_epochs), int(args.max_epochs))
        if schedule:
            schedule = [dict(stage) for stage in schedule if int(stage["epoch_end"]) <= int(target_epochs)]
            if not schedule or int(schedule[-1]["epoch_end"]) != int(target_epochs):
                active_batch = int(batch_size)
                active_lr = float(lr)
                stage_label = schedule[-1]["label"] if schedule else "debug_cap"
                if stage_label.endswith("_cap"):
                    cap_label = stage_label
                else:
                    cap_label = f"{stage_label}_cap"
                schedule.append(
                    {
                        "epoch_end": int(target_epochs),
                        "batch": active_batch,
                        "lr": active_lr,
                        "label": cap_label,
                    }
                )
        train_cfg["epochs"] = int(target_epochs)
        train_cfg["schedule"] = schedule

    remaining_epochs = max(0, int(target_epochs) - int(start_epoch))
    print("Run config:")
    print(f"  mode: {'resume' if args.resume else 'fresh'}")
    print(f"  checkpoint_in: {args.resume if args.resume else '<none>'}")
    print(f"  checkpoint_out: {args.save}")
    print(f"  start_epoch: {start_epoch}")
    print(f"  target_epochs: {target_epochs}")
    print(f"  remaining_epochs: {remaining_epochs}")
    print(f"  batch: {batch_size}")
    print(f"  lr: {lr}")
    print(f"  loss_path: {args.loss_path}")
    print(f"  diag_every: {args.diag_every}")
    print(f"  inverse_check: {args.inverse_check}")
    print(f"  max_z_recon_abs: {args.max_z_recon_abs}")
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
        f" width_levels={prod._schedule_to_text(arch['width_levels'])}"
        f" n_cycles={arch['n_cycles']}"
        f" n_cycles_levels={prod._schedule_to_text(arch['n_cycles_levels'])}"
        f" radius={arch['radius']}"
        f" radius_levels={prod._schedule_to_text(arch['radius_levels'])}"
        f" eta_gaussian={arch['eta_gaussian']}"
        f" gaussian_radius={arch['gaussian_radius']}"
        f" gaussian_radius_levels={prod._schedule_to_text(arch['gaussian_radius_levels'])}"
        f" gaussian_width={arch['gaussian_width']}"
        f" gaussian_width_levels={prod._schedule_to_text(arch['gaussian_width_levels'])}"
        f" terminal_prior={arch['terminal_prior']}"
        f" terminal_layers={arch['terminal_n_layers']}"
        f" terminal_width={arch['terminal_width']}"
        f" output_init_scale={arch['output_init_scale']}"
        f" parity={arch['parity']}"
        f" rg={arch['rg_type']}"
        f" log_scale_clip={arch['log_scale_clip']}"
        f" offdiag_clip={arch['offdiag_clip']}"
    )

    tic = time.perf_counter()
    last_finite_epoch = start_epoch

    def maybe_save(epoch_value):
        if args.save_every > 0 and epoch_value % args.save_every == 0:
            prod.save_checkpoint(args.save, cfg, weights, opt_state, key, losses, epoch_value, arch, train_cfg)

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
            step, _ = _make_step(cfg, theory, stage_batch, stage_lr, loss_path=args.loss_path)
            train_cfg["lr"] = stage_lr
            train_cfg["batch"] = stage_batch
            pbar = tqdm(range(stage_begin, stage_end), desc=f"DebugTrain(RG:{stage['label']})")
            for ep in pbar:
                key, k = jax.random.split(key)
                next_weights, next_opt_state, lv, grad_norm = step(weights, opt_state, k)
                loss_value = float(np.asarray(lv))
                diag = None
                need_diag = args.diag_every > 0 and ((ep + 1) % args.diag_every == 0 or not math.isfinite(loss_value))
                if need_diag:
                    _, diag = _diag_batch(
                        cfg,
                        weights,
                        k,
                        stage_batch,
                        theory,
                        loss_path=args.loss_path,
                        do_inverse_check=args.inverse_check,
                    )
                    diag["grad_norm"] = grad_norm
                    _print_diag(ep + 1, stage["label"], loss_value, diag)
                    if args.max_z_recon_abs > 0.0 and args.inverse_check:
                        z_recon_abs = float(np.asarray(diag.get("z_recon_abs_max", jnp.nan)))
                        if math.isfinite(z_recon_abs) and z_recon_abs > float(args.max_z_recon_abs):
                            if args.save_last_finite and args.save:
                                prod.save_checkpoint(
                                    f"{args.save}.lastfinite.pkl",
                                    cfg,
                                    weights,
                                    opt_state,
                                    key,
                                    losses,
                                    last_finite_epoch,
                                    arch,
                                    train_cfg,
                                )
                            _write_debug_summary(
                                args.save,
                                status="healthcheck_abort",
                                epoch=ep + 1,
                                stage_label=stage["label"],
                                loss_value=loss_value,
                                diag=diag,
                            )
                            raise RuntimeError(
                                f"Inverse health check exceeded at epoch {ep + 1} stage={stage['label']}: "
                                f"z_recon_abs_max={z_recon_abs}"
                            )
                if not math.isfinite(loss_value):
                    if diag is None:
                        _, diag = _diag_batch(
                            cfg,
                            weights,
                            k,
                            stage_batch,
                            theory,
                            loss_path=args.loss_path,
                            do_inverse_check=args.inverse_check,
                        )
                        diag["grad_norm"] = grad_norm
                    if args.save_last_finite and args.save:
                        prod.save_checkpoint(
                            f"{args.save}.lastfinite.pkl",
                            cfg,
                            weights,
                            opt_state,
                            key,
                            losses,
                            last_finite_epoch,
                            arch,
                            train_cfg,
                        )
                    _write_debug_marker(
                        args.save,
                        epoch=ep + 1,
                        stage_label=stage["label"],
                        loss_value=loss_value,
                        diag=diag,
                    )
                    raise RuntimeError(f"Non-finite loss detected at epoch {ep + 1} stage={stage['label']}: loss={loss_value}")
                weights, opt_state = next_weights, next_opt_state
                losses.append(loss_value)
                last_finite_epoch = ep + 1
                pbar.set_postfix(loss=f"{loss_value:.6f}")
                maybe_save(ep + 1)
            prev_end = stage_end
    else:
        pbar = tqdm(range(start_epoch, int(target_epochs)), desc="DebugTrain(RG)")
        for ep in pbar:
            key, k = jax.random.split(key)
            next_weights, next_opt_state, lv, grad_norm = step(weights, opt_state, k)
            loss_value = float(np.asarray(lv))
            diag = None
            need_diag = args.diag_every > 0 and ((ep + 1) % args.diag_every == 0 or not math.isfinite(loss_value))
            if need_diag:
                _, diag = _diag_batch(
                    cfg,
                    weights,
                    k,
                    int(train_cfg["batch"]),
                    theory,
                    loss_path=args.loss_path,
                    do_inverse_check=args.inverse_check,
                )
                diag["grad_norm"] = grad_norm
                _print_diag(ep + 1, None, loss_value, diag)
                if args.max_z_recon_abs > 0.0 and args.inverse_check:
                    z_recon_abs = float(np.asarray(diag.get("z_recon_abs_max", jnp.nan)))
                    if math.isfinite(z_recon_abs) and z_recon_abs > float(args.max_z_recon_abs):
                        if args.save_last_finite and args.save:
                            prod.save_checkpoint(
                                f"{args.save}.lastfinite.pkl",
                                cfg,
                                weights,
                                opt_state,
                                key,
                                losses,
                                last_finite_epoch,
                                arch,
                                train_cfg,
                            )
                        _write_debug_summary(
                            args.save,
                            status="healthcheck_abort",
                            epoch=ep + 1,
                            stage_label=None,
                            loss_value=loss_value,
                            diag=diag,
                        )
                        raise RuntimeError(
                            f"Inverse health check exceeded at epoch {ep + 1}: z_recon_abs_max={z_recon_abs}"
                        )
            if not math.isfinite(loss_value):
                if diag is None:
                    _, diag = _diag_batch(
                        cfg,
                        weights,
                        k,
                        int(train_cfg["batch"]),
                        theory,
                        loss_path=args.loss_path,
                        do_inverse_check=args.inverse_check,
                    )
                    diag["grad_norm"] = grad_norm
                if args.save_last_finite and args.save:
                    prod.save_checkpoint(
                        f"{args.save}.lastfinite.pkl",
                        cfg,
                        weights,
                        opt_state,
                        key,
                        losses,
                        last_finite_epoch,
                        arch,
                        train_cfg,
                    )
                _write_debug_marker(args.save, epoch=ep + 1, stage_label=None, loss_value=loss_value, diag=diag)
                raise RuntimeError(f"Non-finite loss detected at epoch {ep + 1}: loss={loss_value}")
            weights, opt_state = next_weights, next_opt_state
            losses.append(loss_value)
            last_finite_epoch = ep + 1
            pbar.set_postfix(loss=f"{loss_value:.6f}")
            maybe_save(ep + 1)

    toc = time.perf_counter()
    print(f"debug training time: {toc - tic:.2f}s")
    prod.save_checkpoint(args.save, cfg, weights, opt_state, key, losses, int(target_epochs), arch, train_cfg)
    print(f"saved {args.save}")
    print(f"final loss {losses[-1] if losses else 'n/a'}")
    if losses:
        key, k_diag = jax.random.split(key)
        final_loss, final_diag = _diag_batch(
            cfg,
            weights,
            k_diag,
            int(train_cfg["batch"]),
            theory,
            loss_path=args.loss_path,
            do_inverse_check=args.inverse_check,
        )
        final_loss_value = float(np.asarray(final_loss))
        final_stage = schedule[-1]["label"] if schedule else None
        _write_debug_summary(
            args.save,
            status="completed",
            epoch=int(target_epochs),
            stage_label=final_stage,
            loss_value=final_loss_value,
            diag=final_diag,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HMC simulation for pure U(1) Yang-Mills using jaxqft.

Features:
- Arbitrary lattice shape via --shape.
- 2nd/4th order integrators with warmup nmd adaptation to target acceptance.
- Checkpoint/resume (gauge field, RNG state, acceptance history, progress).
"""

from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from pathlib import Path


def _cli_value(argv, flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _cli_bool(argv, on_flag: str, off_flag: str):
    val = None
    for a in argv:
        if a == on_flag:
            val = True
        elif a == off_flag:
            val = False
    return val


def _append_xla_flag(flag: str):
    cur = os.environ.get("XLA_FLAGS", "").split()
    if flag not in cur:
        cur.append(flag)
        os.environ["XLA_FLAGS"] = " ".join(cur).strip()


def _configure_cpu_xla_flags_from_cli_env():
    # Must run before importing jax.
    threads = _cli_value(sys.argv, "--cpu-threads")
    if threads is None:
        threads = os.environ.get("JAXQFT_CPU_THREADS")
    if threads is not None:
        n = int(threads)
        if n > 0:
            _append_xla_flag("--xla_cpu_multi_thread_eigen=true")
            _append_xla_flag(f"intra_op_parallelism_threads={n}")

    onednn = _cli_bool(sys.argv, "--cpu-onednn", "--no-cpu-onednn")
    if onednn is None:
        env = os.environ.get("JAXQFT_CPU_ONEDNN")
        if env is not None:
            onednn = env.strip().lower() not in ("0", "false", "no", "off")
    if onednn is True:
        _append_xla_flag("--xla_cpu_use_onednn=true")
    elif onednn is False:
        _append_xla_flag("--xla_cpu_use_onednn=false")


_configure_cpu_xla_flags_from_cli_env()

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np


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

from jaxqft.core.integrators import force_gradient, leapfrog, minnorm2, minnorm4pf4
from jaxqft.core.update import hmc
from jaxqft.models.u1_ym import U1YangMills
from jaxqft.stats import integrated_autocorr_time


def avg_and_err(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size - 1))


def _parse_shape(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _integrator_order(name: str) -> int:
    if name in ("forcegrad", "minnorm4pf4"):
        return 4
    return 2


def _default_target_accept(name: str) -> float:
    return 0.90 if _integrator_order(name) == 4 else 0.68


def _build_integrator(name: str, theory: U1YangMills, nmd: int, tau: float):
    if name == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, nmd, tau)
    if name == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, nmd, tau)
    if name == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, nmd, tau)
    if name == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, nmd, tau)
    raise ValueError(f"Unknown integrator: {name}")


def _save_checkpoint(path: str, q, theory, chain, state: dict, config: dict):
    payload = {
        "q": np.asarray(q),
        "theory_key": np.asarray(theory.key),
        "hmc_key": np.asarray(chain.key),
        "accept_reject": np.asarray(chain.AcceptReject, dtype=np.float32),
        "state": state,
        "config": config,
        "timestamp": time.time(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4, help="lattice size per dimension")
    ap.add_argument("--Nd", type=int, default=4, help="spacetime dimensions")
    ap.add_argument("--shape", type=str, default="", help="comma-separated lattice shape, e.g. 16,16,16,16")
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0),
        help="CPU intra-op thread hint (applied before JAX import via XLA_FLAGS); 0 keeps runtime default",
    )
    ap.add_argument(
        "--cpu-onednn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="toggle oneDNN CPU backend (applied before JAX import via XLA_FLAGS)",
    )
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--meas", type=int, default=50)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--nmd", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["minnorm2", "leapfrog", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--target-accept", type=float, default=None, help="warmup adaptation target; default depends on integrator order")
    ap.add_argument("--adapt-nmd", action=argparse.BooleanOptionalAction, default=True, help="adapt nmd during warmup")
    ap.add_argument("--adapt-interval", type=int, default=5, help="adapt every this many warmup trajectories")
    ap.add_argument("--adapt-window", type=int, default=10, help="moving window for warmup acceptance")
    ap.add_argument("--adapt-tol", type=float, default=0.03, help="target deadband")
    ap.add_argument("--nmd-min", type=int, default=1)
    ap.add_argument("--nmd-max", type=int, default=256)
    ap.add_argument("--nmd-step", type=int, default=1)
    ap.add_argument("--warmup-log-every", type=int, default=5)
    ap.add_argument("--layout", type=str, default="BMXYIJ", choices=["BMXYIJ", "BXYMIJ", "auto"])
    ap.add_argument("--layout-bench-iters", type=int, default=3)
    ap.add_argument(
        "--fast-hmc-jit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use jitted scan HMC path in jaxqft.core.update.HMC (can be slower on CPU)",
    )
    ap.add_argument("--jit-force", action=argparse.BooleanOptionalAction, default=True, help="jit U1 force kernel")
    ap.add_argument("--jit-evolve-q", action=argparse.BooleanOptionalAction, default=True, help="jit U1 link update (evolve_q)")
    ap.add_argument("--jit-action", action=argparse.BooleanOptionalAction, default=True, help="jit U1 action kernel")
    ap.add_argument("--jit-kinetic", action=argparse.BooleanOptionalAction, default=True, help="jit kinetic-energy kernel")
    ap.add_argument("--jit-refresh-key", action=argparse.BooleanOptionalAction, default=True, help="jit refresh_p_with_key kernel used by fast HMC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="u1_hmc_ckpt.pkl", help="checkpoint path")
    ap.add_argument("--resume", type=str, default="", help="resume from checkpoint")
    ap.add_argument("--save-every", type=int, default=0, help="save every N warmup trajectories / measurement steps")
    ap.add_argument(
        "--iat-method",
        type=str,
        default="ips",
        choices=["ips", "sokal", "gamma"],
        help="IAT estimator for plaquette",
    )
    ap.add_argument("--iat-max-lag", type=int, default=0, help="max lag for IAT (0 = auto)")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))

    ckpt = _load_checkpoint(args.resume) if args.resume else None

    lattice_shape = _parse_shape(args.shape) if args.shape.strip() else tuple([args.L] * args.Nd)
    beta = float(args.beta)
    batch = int(args.batch)
    tau = float(args.tau)
    skip = int(args.skip)
    integrator_name = str(args.integrator)
    nmd = int(args.nmd)
    selected_layout = str(args.layout)

    if ckpt is not None:
        c = ckpt.get("config", {})
        lattice_shape = tuple(c.get("lattice_shape", lattice_shape))
        beta = float(c.get("beta", beta))
        batch = int(c.get("batch", batch))
        tau = float(c.get("tau", tau))
        skip = int(c.get("skip", skip))
        integrator_name = str(c.get("integrator", integrator_name))
        selected_layout = str(c.get("layout", "BMXYIJ"))
        nmd = int(ckpt.get("state", {}).get("nmd", nmd))
        print(f"Resuming from {args.resume}")
        print("  using checkpoint config for shape/beta/batch/layout/integrator/tau/skip")

    if selected_layout == "auto" and ckpt is None:
        timings = U1YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=max(1, batch),
            n_iter=max(1, args.layout_bench_iters),
            seed=args.seed,
        )
        selected_layout = min(timings, key=timings.get)
        print("Layout benchmark (s/action):", timings)
        print("Selected layout:", selected_layout)

    theory = U1YangMills(
        lattice_shape=lattice_shape,
        beta=beta,
        batch_size=batch,
        layout=selected_layout,
        seed=args.seed,
    )
    # Optional explicit kernel JITs for single-device performance tuning.
    if args.jit_force:
        # Avoid nested jit if model already provides a jitted specialized force.
        if not bool(getattr(theory, "_force_is_jitted", False)):
            theory.force = jax.jit(theory.force)
    else:
        # For optimized U1 path, allow disabling the internal jitted force for A/B tests.
        if bool(getattr(theory, "_use_optimized_force", False)):
            theory.force = theory.force_optimized_unjitted
    if args.jit_evolve_q:
        # Avoid nested jit if model already provides a jitted specialized drift.
        if not bool(getattr(theory, "_evolve_q_is_jitted", False)):
            theory.evolve_q = jax.jit(theory.evolve_q)
    else:
        if bool(getattr(theory, "_use_optimized_evolve_q", False)):
            theory.evolve_q = theory.evolve_q_optimized_unjitted
    theory.evolveQ = theory.evolve_q
    if args.jit_action:
        # Avoid nested jit if model already provides a jitted specialized action.
        if not bool(getattr(theory, "_action_is_jitted", False)):
            theory.action = jax.jit(theory.action)
    else:
        if bool(getattr(theory, "_use_optimized_action", False)):
            theory.action = theory.action_optimized_unjitted
    if args.jit_kinetic:
        theory.kinetic = jax.jit(theory.kinetic)
    if args.jit_refresh_key and hasattr(theory, "refresh_p_with_key"):
        theory.refresh_p_with_key = jax.jit(theory.refresh_p_with_key)

    q = theory.hotStart(scale=0.2)

    I = _build_integrator(integrator_name, theory, nmd, tau)
    chain = hmc(T=theory, I=I, verbose=False, use_fast_jit=bool(args.fast_hmc_jit))

    warmup_done = 0
    meas_done = 0
    warmup_traj_acc = []
    meas_step_acc = []
    plaquettes = []
    actions = []

    if ckpt is not None:
        q = jax.device_put(np.asarray(ckpt["q"]))
        theory.key = jax.device_put(np.asarray(ckpt["theory_key"], dtype=np.uint32))
        chain.key = jax.device_put(np.asarray(ckpt["hmc_key"], dtype=np.uint32))
        chain.AcceptReject = list(np.asarray(ckpt.get("accept_reject", []), dtype=np.float64))
        s = ckpt.get("state", {})
        warmup_done = int(s.get("warmup_done", 0))
        meas_done = int(s.get("meas_done", 0))
        nmd = int(s.get("nmd", nmd))
        chain.I = _build_integrator(integrator_name, theory, nmd, tau)
        warmup_traj_acc = list(s.get("warmup_traj_acc", []))
        meas_step_acc = list(s.get("meas_step_acc", []))
        plaquettes = list(s.get("plaquettes", []))
        actions = list(s.get("actions", []))
        print(f"  resumed state: warmup_done={warmup_done}, meas_done={meas_done}, nmd={nmd}")

    target_accept = float(args.target_accept) if args.target_accept is not None else _default_target_accept(integrator_name)

    print("Run config:")
    print(f"  lattice_shape: {lattice_shape}")
    print(f"  beta: {beta}")
    print(f"  batch: {batch}")
    print(f"  layout: {selected_layout}")
    print(f"  integrator: {integrator_name} (nmd={nmd}, tau={tau})")
    print(f"  warmup/meas/skip: {args.warmup}/{args.meas}/{skip}")
    print(
        "  jit:"
        f" fast_hmc={bool(args.fast_hmc_jit)}"
        f" force={bool(args.jit_force)}"
        f" evolve_q={bool(args.jit_evolve_q)}"
        f" action={bool(args.jit_action)}"
        f" kinetic={bool(args.jit_kinetic)}"
        f" refresh_key={bool(args.jit_refresh_key)}"
    )
    print(
        "  warmup nmd adaptation:"
        f" enabled={args.adapt_nmd}"
        f" target={target_accept:.3f}"
        f" window={args.adapt_window}"
        f" interval={args.adapt_interval}"
        f" tol={args.adapt_tol:.3f}"
        f" step={args.nmd_step}"
        f" bounds=[{args.nmd_min},{args.nmd_max}]"
    )
    print(f"  checkpoint: save='{args.save}' resume='{args.resume or '<none>'}' every={args.save_every}")

    ckpt_config = {
        "lattice_shape": tuple(int(v) for v in lattice_shape),
        "beta": float(beta),
        "batch": int(batch),
        "layout": selected_layout,
        "integrator": integrator_name,
        "tau": float(tau),
        "skip": int(skip),
    }

    def maybe_save(reason: str):
        if not args.save:
            return
        state = {
            "nmd": int(nmd),
            "warmup_done": int(warmup_done),
            "meas_done": int(meas_done),
            "warmup_traj_acc": [float(x) for x in warmup_traj_acc],
            "meas_step_acc": [float(x) for x in meas_step_acc],
            "plaquettes": [float(x) for x in plaquettes],
            "actions": [float(x) for x in actions],
            "reason": reason,
        }
        _save_checkpoint(args.save, q, theory, chain, state, ckpt_config)

    # Warmup with optional nmd adaptation.
    warmup_start = int(warmup_done)
    warmup_t0 = time.perf_counter()
    for kw in range(warmup_start, int(args.warmup)):
        prev = len(chain.AcceptReject)
        q = chain.evolve(q, 1)
        new = np.asarray(chain.AcceptReject[prev:], dtype=np.float64)
        traj_acc = float(new.mean()) if new.size else float("nan")
        warmup_traj_acc.append(traj_acc)
        warmup_done = kw + 1

        if args.adapt_nmd and args.adapt_interval > 0 and warmup_done % int(args.adapt_interval) == 0:
            w = np.asarray(warmup_traj_acc[-max(1, int(args.adapt_window)) :], dtype=np.float64)
            acc_w = float(np.nanmean(w))
            nmd_new = int(nmd)
            if acc_w < target_accept - float(args.adapt_tol):
                nmd_new = min(int(args.nmd_max), int(nmd + args.nmd_step))
            elif acc_w > target_accept + float(args.adapt_tol):
                nmd_new = max(int(args.nmd_min), int(nmd - args.nmd_step))
            if nmd_new != nmd:
                old = nmd
                nmd = int(nmd_new)
                chain.I = _build_integrator(integrator_name, theory, nmd, tau)
                print(f"warmup adapt: k={warmup_done} acc_win={acc_w:.3f} nmd {old} -> {nmd}")

        if warmup_done % max(1, int(args.warmup_log_every)) == 0 or warmup_done == int(args.warmup):
            w = np.asarray(warmup_traj_acc[-max(1, int(args.adapt_window)) :], dtype=np.float64)
            print(
                f"warmup k={warmup_done}/{args.warmup}"
                f" nmd={nmd}"
                f" traj_acc={traj_acc:.3f}"
                f" win_acc={float(np.nanmean(w)):.3f}"
                f" global_acc={chain.calc_Acceptance():.3f}"
            )

        if args.save_every > 0 and warmup_done % int(args.save_every) == 0:
            maybe_save(reason="warmup")

    warmup_t1 = time.perf_counter()
    if args.warmup > warmup_start:
        print(f"warmup time per trajectory: {(warmup_t1 - warmup_t0) * 1e3 / max(1, args.warmup - warmup_start):.3f} ms")
    elif args.warmup > 0:
        print("warmup: already completed in checkpoint")

    meas_start = int(meas_done)
    t0 = time.perf_counter()
    for k in range(meas_start, int(args.meas)):
        plaq = np.asarray(theory.average_plaquette(q))
        act = np.asarray(theory.action(q))
        plaquettes.extend(plaq.tolist())
        actions.extend(act.tolist())

        prev = len(chain.AcceptReject)
        q = chain.evolve(q, int(skip))
        new = np.asarray(chain.AcceptReject[prev:], dtype=np.float64)
        step_acc = float(new.mean()) if new.size else float("nan")
        meas_step_acc.append(step_acc)
        meas_done = k + 1

        if k % 10 == 0:
            print(
                "k=",
                k,
                "plaquette=",
                float(plaq.mean()),
                "action=",
                float(act.mean()),
                "step_acc=",
                step_acc,
                "acc=",
                chain.calc_Acceptance(),
            )

        if args.save_every > 0 and meas_done % int(args.save_every) == 0:
            maybe_save(reason="measurement")

    t1 = time.perf_counter()
    if int(args.meas) > meas_start:
        n_traj = (int(args.meas) - meas_start) * int(skip)
        print(f"meas-update time per trajectory: {(t1 - t0) * 1e3 / max(1, n_traj):.3f} ms")
    elif args.meas > 0:
        print("measurement: already completed in checkpoint")

    mP, eP = avg_and_err(plaquettes)
    mS, eS = avg_and_err(actions)
    print("Results:")
    print("  Plaquette:", mP, "+/-", eP)
    print("  Action:", mS, "+/-", eS)
    plaq_arr = np.asarray(plaquettes, dtype=np.float64)
    iat = integrated_autocorr_time(
        plaq_arr,
        method=args.iat_method,
        max_lag=(None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)),
    )
    if bool(iat.get("ok", False)):
        nplaq = int(plaq_arr.size)
        std_plaq = float(np.std(plaq_arr, ddof=1)) if nplaq > 1 else float("nan")
        ess = float(iat["ess"])
        eP_iat = float(std_plaq / np.sqrt(max(ess, 1e-12))) if np.isfinite(std_plaq) else float("nan")
        print(
            "  Plaquette IAT:"
            f" tau_int={float(iat['tau_int']):.3f} samples"
            f" ({float(iat['tau_int']) * float(skip):.3f} trajectories),"
            f" ESS={ess:.2f}/{nplaq},"
            f" window={int(iat['window'])},"
            f" method={iat['method']}"
        )
        print(f"  Plaquette IAT-corrected error: {eP_iat}")
    else:
        print(f"  Plaquette IAT: unavailable ({iat.get('message', 'unknown error')})")
    if warmup_traj_acc:
        print("  Warmup acc (mean over traj):", float(np.nanmean(np.asarray(warmup_traj_acc))))
    if meas_step_acc:
        print("  Meas acc (mean over steps):", float(np.nanmean(np.asarray(meas_step_acc))))
    print("  Final nmd:", nmd)
    print("  Acceptance:", chain.calc_Acceptance())
    maybe_save(reason="final")


if __name__ == "__main__":
    main()

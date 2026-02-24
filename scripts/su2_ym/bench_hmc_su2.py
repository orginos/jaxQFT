#!/usr/bin/env python3
"""Component timing benchmark for a single SU(2) HMC trajectory.

This script is designed to identify dominant runtime components on realistic
lattices (default 8^4). It times one trajectory in synchronized mode so each
component contribution is observable.
"""

from __future__ import annotations

import argparse
import os
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
import jax.numpy as jnp
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
from jaxqft.models.su2_ym import SU2YangMills


def _sync(x):
    jax.block_until_ready(x)
    return x


def _parse_shape(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _build_integrator(name, force_fn, evolve_q_fn, nmd, tau):
    if name == "minnorm2":
        return minnorm2(force_fn, evolve_q_fn, nmd, tau)
    if name == "leapfrog":
        return leapfrog(force_fn, evolve_q_fn, nmd, tau)
    if name == "forcegrad":
        return force_gradient(force_fn, evolve_q_fn, nmd, tau)
    if name == "minnorm4pf4":
        return minnorm4pf4(force_fn, evolve_q_fn, nmd, tau)
    raise ValueError(f"Unknown integrator: {name}")


def _mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def _time_kernel(fn, args, n_iter: int):
    y = fn(*args)
    _sync(y)
    t0 = time.perf_counter()
    for _ in range(max(1, int(n_iter))):
        y = fn(*args)
        _sync(y)
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, int(n_iter)), y


def _rel_diff(a, b):
    da = (a - b).reshape(-1)
    num = jnp.linalg.norm(da)
    den = jnp.linalg.norm(b.reshape(-1)) + 1e-12
    return float(num / den)


def _compare_reference_kernels(theory, q, p, dt: float, n_iter: int):
    out = {}

    if hasattr(theory, "action_reference") and bool(getattr(theory, "_use_optimized_action", False)):
        if bool(getattr(theory, "_action_is_jitted", False)):
            action_opt = theory.action
        else:
            action_opt = jax.jit(theory.action_optimized_unjitted)
        action_ref = jax.jit(theory.action_reference)
        t_opt, y_opt = _time_kernel(action_opt, (q,), n_iter)
        t_ref, y_ref = _time_kernel(action_ref, (q,), n_iter)
        out["action"] = {
            "enabled": True,
            "opt_sec": float(t_opt),
            "ref_sec": float(t_ref),
            "speedup": float(t_ref / max(t_opt, 1e-12)),
            "rel_diff": _rel_diff(y_opt, y_ref),
        }
    else:
        out["action"] = {"enabled": False, "reason": "optimized action path disabled"}

    if hasattr(theory, "force_reference") and bool(getattr(theory, "_use_optimized_force", False)):
        if bool(getattr(theory, "_force_is_jitted", False)):
            force_opt = theory.force
        else:
            force_opt = jax.jit(theory.force_optimized_unjitted)
        force_ref = jax.jit(theory.force_reference)
        t_opt, y_opt = _time_kernel(force_opt, (q,), n_iter)
        t_ref, y_ref = _time_kernel(force_ref, (q,), n_iter)
        out["force"] = {
            "enabled": True,
            "opt_sec": float(t_opt),
            "ref_sec": float(t_ref),
            "speedup": float(t_ref / max(t_opt, 1e-12)),
            "rel_diff": _rel_diff(y_opt, y_ref),
        }
    else:
        out["force"] = {"enabled": False, "reason": "optimized force path disabled"}

    if hasattr(theory, "evolve_q_reference") and bool(getattr(theory, "_use_optimized_evolve_q", False)):
        evolve_opt = jax.jit(lambda pp, qq: theory.evolve_q(dt, pp, qq))
        evolve_ref = jax.jit(lambda pp, qq: theory.evolve_q_reference(dt, pp, qq))
        t_opt, y_opt = _time_kernel(evolve_opt, (p, q), n_iter)
        t_ref, y_ref = _time_kernel(evolve_ref, (p, q), n_iter)
        out["evolve_q"] = {
            "enabled": True,
            "opt_sec": float(t_opt),
            "ref_sec": float(t_ref),
            "speedup": float(t_ref / max(t_opt, 1e-12)),
            "rel_diff": _rel_diff(y_opt, y_ref),
        }
    else:
        out["evolve_q"] = {"enabled": False, "reason": "optimized evolve_q path disabled"}

    return out


def _profile_one_trajectory(q, key, theory, integrator_name: str, nmd: int, tau: float):
    stats = {}
    force_state = {"time": 0.0, "calls": 0}
    drift_state = {"time": 0.0, "calls": 0}

    traj_t0 = time.perf_counter()

    key, kp, kr = jax.random.split(key, 3)

    t0 = time.perf_counter()
    if hasattr(theory, "refresh_p_with_key"):
        p0 = theory.refresh_p_with_key(kp)
    else:
        p0 = theory.refresh_p()
    _sync(p0)
    stats["refresh_p"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    k0 = theory.kinetic(p0)
    _sync(k0)
    stats["kinetic0"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    a0 = theory.action(q)
    _sync(a0)
    stats["action0"] = time.perf_counter() - t0
    H0 = k0 + a0

    def force_wrapped(x):
        t = time.perf_counter()
        y = theory.force(x)
        _sync(y)
        force_state["time"] += time.perf_counter() - t
        force_state["calls"] += 1
        return y

    def evolve_q_wrapped(dt, p, qq):
        t = time.perf_counter()
        y = theory.evolve_q(dt, p, qq)
        _sync(y)
        drift_state["time"] += time.perf_counter() - t
        drift_state["calls"] += 1
        return y

    I = _build_integrator(integrator_name, force_wrapped, evolve_q_wrapped, nmd, tau)

    t0 = time.perf_counter()
    p1, q_prop = I.integrate(p0, q)
    _sync((p1, q_prop))
    stats["integrate_total"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    k1 = theory.kinetic(p1)
    _sync(k1)
    stats["kinetic1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    a1 = theory.action(q_prop)
    _sync(a1)
    stats["action1"] = time.perf_counter() - t0
    H1 = k1 + a1

    t0 = time.perf_counter()
    dH = H1 - H0
    acc_prob = jnp.where(dH < 0, jnp.ones_like(dH), jnp.exp(-dH))
    R = jax.random.uniform(kr, shape=acc_prob.shape, dtype=acc_prob.dtype)
    acc_flag = R < acc_prob
    qshape = tuple([q.shape[0]] + [1] * (q.ndim - 1))
    q_next = jnp.where(acc_flag.reshape(qshape), q_prop, q)
    _sync(q_next)
    stats["metropolis"] = time.perf_counter() - t0

    stats["force_total"] = force_state["time"]
    stats["force_calls"] = float(force_state["calls"])
    stats["evolve_q_total"] = drift_state["time"]
    stats["evolve_q_calls"] = float(drift_state["calls"])
    stats["integrator_overhead"] = max(
        0.0,
        stats["integrate_total"] - stats["force_total"] - stats["evolve_q_total"],
    )
    stats["total"] = time.perf_counter() - traj_t0
    stats["acc"] = float(jnp.mean(acc_flag))
    stats["mean_abs_dH"] = float(jnp.mean(jnp.abs(dH)))
    return q_next, key, stats


def main():
    ap = argparse.ArgumentParser(description="Benchmark one SU2 HMC trajectory by component timing.")
    ap.add_argument("--L", type=int, default=8, help="lattice size per dimension")
    ap.add_argument("--Nd", type=int, default=4, help="spacetime dimensions")
    ap.add_argument("--shape", type=str, default="", help="comma-separated lattice shape, e.g. 8,8,8,8")
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
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--integrator", type=str, default="forcegrad", choices=["minnorm2", "leapfrog", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--nmd", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--layout", type=str, default="auto", choices=["BMXYIJ", "BXYMIJ", "auto"])
    ap.add_argument("--layout-bench-iters", type=int, default=3)
    ap.add_argument("--exp-method", type=str, default="su2", choices=["su2", "expm"])
    ap.add_argument("--jit-force", action=argparse.BooleanOptionalAction, default=True, help="jit SU2 force kernel")
    ap.add_argument("--jit-evolve-q", action=argparse.BooleanOptionalAction, default=True, help="jit SU2 evolve_q kernel")
    ap.add_argument("--jit-action", action=argparse.BooleanOptionalAction, default=True, help="jit SU2 action kernel")
    ap.add_argument("--jit-kinetic", action=argparse.BooleanOptionalAction, default=True, help="jit kinetic kernel")
    ap.add_argument("--jit-refresh-key", action=argparse.BooleanOptionalAction, default=True, help="jit refresh_p_with_key kernel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hot-scale", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured trajectories for compilation/cache warmup")
    ap.add_argument("--repeat", type=int, default=5, help="measured trajectories")
    ap.add_argument("--print-each", action="store_true", help="print timing line per measured trajectory")
    ap.add_argument(
        "--compare-reference-kernels",
        action="store_true",
        help="compare optimized kernels against reference kernels (both jitted for fair timing)",
    )
    ap.add_argument(
        "--compare-kernel-iters",
        type=int,
        default=10,
        help="iterations per kernel in --compare-reference-kernels mode",
    )
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))

    lattice_shape = _parse_shape(args.shape) if args.shape.strip() else tuple([args.L] * args.Nd)

    selected_layout = args.layout
    if args.layout == "auto":
        ta = SU2YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=max(1, args.batch),
            n_iter=max(1, args.layout_bench_iters),
            seed=args.seed,
            kernel="action",
            exp_method=args.exp_method,
        )
        tf = SU2YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=args.beta,
            batch_size=max(1, args.batch),
            n_iter=max(1, args.layout_bench_iters),
            seed=args.seed,
            kernel="force",
            exp_method=args.exp_method,
        )
        selected_layout = min(tf, key=tf.get)
        print("Layout benchmark (sec/call):")
        for lay in sorted(ta):
            print(f"  {lay}: action={ta[lay]:.6e}, force={tf[lay]:.6e}")
        print("Selected layout:", selected_layout)

    theory = SU2YangMills(
        lattice_shape=lattice_shape,
        beta=args.beta,
        batch_size=args.batch,
        layout=selected_layout,
        seed=args.seed,
        exp_method=args.exp_method,
    )
    # Optional explicit kernel JITs used by the profile.
    if args.jit_force:
        # Avoid nested jit if model already provides a jitted specialized force.
        if not bool(getattr(theory, "_force_is_jitted", False)):
            theory.force = jax.jit(theory.force)
    else:
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

    q = theory.hot_start(scale=args.hot_scale)
    p_cmp = theory.refresh_p()
    if args.compare_reference_kernels:
        dt_cmp = float(args.tau) / max(1, int(args.nmd))
        cmp = _compare_reference_kernels(theory, q, p_cmp, dt_cmp, n_iter=max(1, args.compare_kernel_iters))
        print("Reference kernel comparison (optimized vs reference, jitted):")
        for name in ("action", "force", "evolve_q"):
            row = cmp.get(name, {})
            if not bool(row.get("enabled", False)):
                print(f"  {name:10s}: skipped ({row.get('reason', 'not available')})")
                continue
            print(
                f"  {name:10s}: opt={row['opt_sec']:.6e} s"
                f" ref={row['ref_sec']:.6e} s"
                f" speedup={row['speedup']:.2f}x"
                f" rel_diff={row['rel_diff']:.3e}"
            )
    key = jax.random.PRNGKey(args.seed + 7919)

    print("Benchmark config:")
    print(f"  lattice_shape: {lattice_shape}")
    print(f"  beta: {args.beta}")
    print(f"  batch: {args.batch}")
    print(f"  layout: {selected_layout}")
    print(f"  exp_method: {args.exp_method}")
    print(f"  integrator: {args.integrator} (nmd={args.nmd}, tau={args.tau})")
    print(
        "  jit:"
        f" force={bool(args.jit_force)}"
        f" evolve_q={bool(args.jit_evolve_q)}"
        f" action={bool(args.jit_action)}"
        f" kinetic={bool(args.jit_kinetic)}"
        f" refresh_key={bool(args.jit_refresh_key)}"
    )
    print(f"  warmup: {args.warmup}")
    print(f"  repeat: {args.repeat}")
    print("  mode: synchronized component timing (includes sync overhead by design)")

    for _ in range(max(0, args.warmup)):
        q, key, _ = _profile_one_trajectory(q, key, theory, args.integrator, args.nmd, args.tau)

    runs = []
    for i in range(max(1, args.repeat)):
        q, key, st = _profile_one_trajectory(q, key, theory, args.integrator, args.nmd, args.tau)
        runs.append(st)
        if args.print_each:
            print(
                f"run {i+1}: total={1e3*st['total']:.3f} ms"
                f" force={1e3*st['force_total']:.3f} ms"
                f" drift={1e3*st['evolve_q_total']:.3f} ms"
                f" acc={st['acc']:.3f}"
                f" |dH|={st['mean_abs_dH']:.3e}"
            )

    keys = [
        "refresh_p",
        "kinetic0",
        "action0",
        "integrate_total",
        "force_total",
        "evolve_q_total",
        "integrator_overhead",
        "kinetic1",
        "action1",
        "metropolis",
        "total",
    ]

    summary = {}
    for k in keys:
        m, s = _mean_std([r[k] for r in runs])
        summary[k] = (m, s)

    total_m = max(summary["total"][0], 1e-18)
    print("Trajectory timing breakdown (mean +/- std):")
    for k in keys:
        m, s = summary[k]
        pct = 100.0 * m / total_m
        print(f"  {k:20s}: {1e3*m:9.3f} +/- {1e3*s:9.3f} ms   ({pct:5.1f}%)")

    force_calls_m, force_calls_s = _mean_std([r["force_calls"] for r in runs])
    drift_calls_m, drift_calls_s = _mean_std([r["evolve_q_calls"] for r in runs])
    force_ms = 1e3 * summary["force_total"][0] / max(force_calls_m, 1e-12)
    drift_ms = 1e3 * summary["evolve_q_total"][0] / max(drift_calls_m, 1e-12)
    acc_m, acc_s = _mean_std([r["acc"] for r in runs])
    dh_m, dh_s = _mean_std([r["mean_abs_dH"] for r in runs])

    print("Integrator call profile:")
    print(f"  force calls/trajectory:    {force_calls_m:.2f} +/- {force_calls_s:.2f}")
    print(f"  evolve_q calls/trajectory: {drift_calls_m:.2f} +/- {drift_calls_s:.2f}")
    print(f"  mean force call time:      {force_ms:.3f} ms")
    print(f"  mean evolve_q call time:   {drift_ms:.3f} ms")
    print("HMC diagnostics:")
    print(f"  acceptance per trajectory: {acc_m:.4f} +/- {acc_s:.4f}")
    print(f"  mean |dH|:                 {dh_m:.6e} +/- {dh_s:.6e}")


if __name__ == "__main__":
    main()

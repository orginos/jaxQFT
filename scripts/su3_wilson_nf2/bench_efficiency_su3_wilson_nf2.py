#!/usr/bin/env python3
"""Efficiency benchmark for SU(3) + Nf=2 Wilson (EO-preconditioned).

Default target case:
- shape: 8,8,8,16
- beta: 5.7
- mass: 0.01
- fermion monomial: eo_preconditioned
- integrator: forcegrad, tau=1.0, nmd=6
- single timescale (gauge=0, fermion=0)

Primary metric: mean wall-clock seconds per trajectory.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict


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

# Keep CPU default on macOS unless user explicitly overrides.
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
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2
from jaxqft.models.su3_ym import SU3YangMills


def _parse_shape(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _build_integrator(name: str, theory: SU3WilsonNf2, nmd: int, tau: float):
    if name == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, nmd, tau)
    if name == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, nmd, tau)
    if name == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, nmd, tau)
    if name == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, nmd, tau)
    raise ValueError(f"Unknown integrator: {name}")


def _install_monomial_profiler(theory: SU3WilsonNf2):
    stats: Dict[str, Dict[str, float]] = {}

    def _mk_rec(name: str) -> Dict[str, float]:
        return stats.setdefault(
            name,
            {
                "force_time": 0.0,
                "force_calls": 0.0,
                "action_time": 0.0,
                "action_calls": 0.0,
            },
        )

    for m in theory.hamiltonian.monomials:
        name = str(getattr(m, "name", m.__class__.__name__))
        rec = _mk_rec(name)
        orig_force = m.force
        orig_action = m.action

        def force_wrapped(q, _orig=orig_force, _rec=rec):
            t0 = time.perf_counter()
            y = _orig(q)
            y = jax.block_until_ready(y)
            _rec["force_time"] += (time.perf_counter() - t0)
            _rec["force_calls"] += 1.0
            return y

        def action_wrapped(q, _orig=orig_action, _rec=rec):
            t0 = time.perf_counter()
            y = _orig(q)
            y = jax.block_until_ready(y)
            _rec["action_time"] += (time.perf_counter() - t0)
            _rec["action_calls"] += 1.0
            return y

        m.force = force_wrapped
        m.action = action_wrapped

    def snapshot():
        out: Dict[str, Dict[str, float]] = {}
        for k, v in stats.items():
            out[k] = dict(v)
        return out

    return snapshot


def _diff_monomial_stats(
    start: Dict[str, Dict[str, float]],
    end: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    names = sorted(set(start.keys()) | set(end.keys()))
    for n in names:
        a = start.get(n, {})
        b = end.get(n, {})
        out[n] = {
            "force_time": float(b.get("force_time", 0.0) - a.get("force_time", 0.0)),
            "force_calls": float(b.get("force_calls", 0.0) - a.get("force_calls", 0.0)),
            "action_time": float(b.get("action_time", 0.0) - a.get("action_time", 0.0)),
            "action_calls": float(b.get("action_calls", 0.0) - a.get("action_calls", 0.0)),
        }
    return out


def _print_monomial_timing(
    stats: Dict[str, Dict[str, float]],
    ntraj: int,
) -> None:
    if not stats:
        return
    tot_f = float(sum(v.get("force_time", 0.0) for v in stats.values()))
    tot_a = float(sum(v.get("action_time", 0.0) for v in stats.values()))
    print("Monomial timing (measurement segment):")
    print(f"  trajectories profiled: {int(ntraj)}")
    print(f"  total force/action time: {tot_f:.6f}s / {tot_a:.6f}s")
    for name in sorted(stats.keys()):
        s = stats[name]
        fc = float(s.get("force_calls", 0.0))
        ft = float(s.get("force_time", 0.0))
        ac = float(s.get("action_calls", 0.0))
        at = float(s.get("action_time", 0.0))
        f_ms = (1e3 * ft / fc) if fc > 0 else float("nan")
        a_ms = (1e3 * at / ac) if ac > 0 else float("nan")
        f_share = (100.0 * ft / (tot_f + 1e-30)) if tot_f > 0 else 0.0
        a_share = (100.0 * at / (tot_a + 1e-30)) if tot_a > 0 else 0.0
        f_traj = (1e3 * ft / max(1, ntraj))
        a_traj = (1e3 * at / max(1, ntraj))
        print(
            f"  {name}:"
            f" force={ft:.6f}s ({f_share:.1f}%) calls={int(fc)} ms/call={f_ms:.3f} ms/traj={f_traj:.3f};"
            f" action={at:.6f}s ({a_share:.1f}%) calls={int(ac)} ms/call={a_ms:.3f} ms/traj={a_traj:.3f}"
        )


def _run_one_traj(q, chain):
    q = chain.evolve(q, 1)
    return jax.block_until_ready(q)


def main():
    ap = argparse.ArgumentParser(description="Efficiency benchmark for SU3 Wilson Nf=2 HMC")
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0),
        help="CPU intra-op thread hint (applied before JAX import); 0 keeps runtime default",
    )
    ap.add_argument(
        "--cpu-onednn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="toggle oneDNN CPU backend (applied before JAX import)",
    )
    ap.add_argument("--shape", type=str, default="8,8,8,16")
    ap.add_argument("--beta", type=float, default=5.7)
    ap.add_argument("--mass", type=float, default=0.01)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="auto", choices=["BMXYIJ", "BXYMIJ", "auto"])
    ap.add_argument("--layout-bench-iters", type=int, default=3)
    ap.add_argument("--exp-method", type=str, default="su3", choices=["su3", "expm"])
    ap.add_argument("--hot-start-scale", type=float, default=0.2)
    ap.add_argument("--integrator", type=str, default="forcegrad", choices=["forcegrad", "minnorm4pf4", "minnorm2", "leapfrog"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--nmd", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=6, help="warmup trajectories before timed measurements")
    ap.add_argument("--meas", type=int, default=20, help="timed trajectories")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-7)
    ap.add_argument("--cg-maxiter", type=int, default=1000)
    ap.add_argument("--solver-kind", type=str, default="cg", choices=["cg", "bicgstab", "gmres"])
    ap.add_argument("--solver-form", type=str, default="normal", choices=["normal", "split", "eo_split"])
    ap.add_argument("--solver-chrono-guess", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--preconditioner", type=str, default="none", choices=["none", "jacobi"])
    ap.add_argument("--gmres-restart", type=int, default=32)
    ap.add_argument("--gmres-solve-method", type=str, default="batched")
    ap.add_argument("--pf-refresh", type=str, default="heatbath", choices=["heatbath", "ou"])
    ap.add_argument("--pf-gamma", type=float, default=0.3)
    ap.add_argument("--pf-force-mode", type=str, default="analytic", choices=["analytic", "autodiff"])
    ap.add_argument("--jit-kinetic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--jit-refresh-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--profile-monomials", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--target-sec", type=float, default=6.0, help="target mean sec/trajectory")
    ap.add_argument("--selfcheck-fail", action="store_true", help="exit nonzero when target is not met")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))

    shape = _parse_shape(args.shape)
    layout = str(args.layout)
    if layout == "auto":
        timings = SU3YangMills.benchmark_layout(
            lattice_shape=shape,
            beta=float(args.beta),
            batch_size=max(1, int(args.batch)),
            n_iter=max(1, int(args.layout_bench_iters)),
            seed=int(args.seed),
            exp_method=str(args.exp_method),
        )
        layout = min(timings, key=timings.get)
        print("Layout benchmark (s/action):", timings)
        print("Selected layout:", layout)

    theory = SU3WilsonNf2(
        lattice_shape=shape,
        beta=float(args.beta),
        batch_size=int(args.batch),
        layout=layout,
        seed=int(args.seed),
        exp_method=str(args.exp_method),
        mass=float(args.mass),
        wilson_r=float(args.r),
        cg_tol=float(args.cg_tol),
        cg_maxiter=int(args.cg_maxiter),
        solver_kind=str(args.solver_kind),
        solver_form=str(args.solver_form),
        use_solver_guess=bool(args.solver_chrono_guess),
        preconditioner_kind=str(args.preconditioner),
        gmres_restart=int(args.gmres_restart),
        gmres_solve_method=str(args.gmres_solve_method),
        include_gauge_monomial=True,
        include_fermion_monomial=True,
        fermion_monomial_kind="eo_preconditioned",
        gauge_timescale=0,
        fermion_timescale=0,
        pseudofermion_refresh=str(args.pf_refresh),
        pseudofermion_gamma=float(args.pf_gamma),
        pseudofermion_force_mode=str(args.pf_force_mode),
        smd_gamma=float(args.pf_gamma),
        auto_refresh_pseudofermions=True,
    )

    if bool(args.jit_kinetic):
        theory.kinetic = jax.jit(theory.kinetic)
    if bool(args.jit_refresh_key) and hasattr(theory, "refresh_p_with_key"):
        theory.refresh_p_with_key = jax.jit(theory.refresh_p_with_key)

    monomial_snapshot = None
    if bool(args.profile_monomials):
        monomial_snapshot = _install_monomial_profiler(theory)

    integ = _build_integrator(str(args.integrator), theory, int(args.nmd), float(args.tau))
    chain = hmc(T=theory, I=integ, verbose=False, use_fast_jit=False)

    q = theory.hotStart(scale=float(args.hot_start_scale))

    print("Benchmark config:")
    print("  lattice_shape:", shape)
    print("  beta/mass:", float(args.beta), "/", float(args.mass))
    print("  batch:", int(args.batch))
    print("  layout:", layout)
    print("  fermion monomial:", "eo_preconditioned")
    print("  timescales gauge/fermion:", "0/0 (single timescale)")
    print("  solver:", f"{args.solver_kind}/{args.solver_form}", f"tol={float(args.cg_tol):.2e}", f"maxiter={int(args.cg_maxiter)}")
    print("  integrator:", f"{args.integrator} (tau={float(args.tau)}, nmd={int(args.nmd)})")
    print("  pf:", f"refresh={args.pf_refresh}", f"gamma={float(args.pf_gamma):.3f}", f"force={args.pf_force_mode}")
    print("  warmup/meas:", f"{int(args.warmup)}/{int(args.meas)}")
    print("  target sec/traj:", float(args.target_sec))
    if bool(args.profile_monomials):
        print("  note: monomial profiling enables extra synchronizations and can increase measured wall time.")

    for k in range(max(0, int(args.warmup))):
        q = _run_one_traj(q, chain)
        if (k + 1) % max(1, int(args.log_every)) == 0:
            print(f"warmup {k+1}/{int(args.warmup)}")

    chain.reset_acceptance()
    t_series = []
    mon_start = monomial_snapshot() if callable(monomial_snapshot) else None
    for k in range(max(1, int(args.meas))):
        t0 = time.perf_counter()
        q = _run_one_traj(q, chain)
        dt = time.perf_counter() - t0
        t_series.append(float(dt))

        if (k + 1) % max(1, int(args.log_every)) == 0 or (k + 1) == int(args.meas):
            new = np.asarray(chain.AcceptReject[-int(args.batch) :], dtype=np.float64)
            step_acc = float(new.mean()) if new.size else float("nan")
            print(
                f"meas {k+1}/{int(args.meas)}"
                f" traj_sec={dt:.3f}"
                f" step_acc={step_acc:.3f}"
                f" acc={chain.calc_acceptance():.3f}"
            )

    mon_end = monomial_snapshot() if callable(monomial_snapshot) else None

    arr = np.asarray(t_series, dtype=np.float64)
    mean_sec = float(np.mean(arr))
    std_sec = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    med_sec = float(np.median(arr))
    p95_sec = float(np.quantile(arr, 0.95)) if arr.size > 0 else float("nan")
    acc = float(chain.calc_acceptance())
    pass_target = bool(mean_sec <= float(args.target_sec))

    print("Timing summary:")
    print(f"  mean sec/trajectory:   {mean_sec:.6f}")
    print(f"  median sec/trajectory: {med_sec:.6f}")
    print(f"  p95 sec/trajectory:    {p95_sec:.6f}")
    print(f"  std sec/trajectory:    {std_sec:.6f}")
    print(f"  acceptance:            {acc:.6f}")
    print(
        f"Target check: mean {mean_sec:.6f}s <= {float(args.target_sec):.6f}s -> "
        f"{'PASS' if pass_target else 'FAIL'}"
    )

    if mon_start is not None and mon_end is not None:
        _print_monomial_timing(_diff_monomial_stats(mon_start, mon_end), ntraj=int(args.meas))

    if bool(args.selfcheck_fail) and (not pass_target):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

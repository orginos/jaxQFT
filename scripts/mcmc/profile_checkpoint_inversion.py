#!/usr/bin/env python3
"""Profile inversion cost from an existing MCMC checkpoint.

This script replays a short trajectory segment from a saved checkpoint and
reports:
- trajectory wall time
- monomial force/action timing
- solver (inversion) timing inside trajectory evolution

Intended workflow:
1) finish warmup and save checkpoint with `scripts/mcmc/mcmc.py`
2) run this script on that checkpoint
3) use output JSON for cost accounting and optimization decisions
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _cli_value(argv: List[str], flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _cli_bool(argv: List[str], on_flag: str, off_flag: str):
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
from jaxqft.core.update import SMD, hmc
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2


def _cfg_get(cfg: Mapping[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _parse_shape(v: Any) -> Tuple[int, ...]:
    if isinstance(v, str):
        vals = [int(x.strip()) for x in v.split(",") if x.strip()]
    else:
        vals = [int(x) for x in list(v)]
    if not vals:
        raise ValueError("shape must be non-empty")
    return tuple(vals)


def _parse_fermion_bc(v: Any, nd: int) -> Tuple[complex, ...]:
    txt = str(v).strip().lower()
    if txt in ("", "periodic", "p"):
        return tuple([1.0 + 0.0j] * int(nd))
    if txt in ("antiperiodic-t", "anti-t", "ap-t", "apt", "chroma"):
        vals = [1.0 + 0.0j] * int(nd)
        vals[-1] = -1.0 + 0.0j
        return tuple(vals)
    toks = [t.strip() for t in str(v).split(",") if t.strip()]
    vals = [complex(t.replace("I", "j").replace("i", "j")) for t in toks]
    if len(vals) != int(nd):
        raise ValueError(f"fermion_bc must have Nd={nd} entries; got {len(vals)}")
    return tuple(vals)


def _build_integrator(name: str, theory: SU3WilsonNf2, nmd: int, tau: float):
    if name == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, nmd, tau)
    if name == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, nmd, tau)
    if name == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, nmd, tau)
    if name == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, nmd, tau)
    raise ValueError(f"Unknown integrator: {name}")


def _snapshot_nested_numeric(d: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, v in d.items():
        out[str(k)] = {str(kk): float(vv) for kk, vv in v.items()}
    return out


def _diff_nested_numeric(
    start: Mapping[str, Mapping[str, float]],
    end: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    names = sorted(set(start.keys()) | set(end.keys()))
    for n in names:
        a = start.get(n, {})
        b = end.get(n, {})
        row: Dict[str, float] = {}
        for k in sorted(set(a.keys()) | set(b.keys())):
            row[k] = float(b.get(k, 0.0) - a.get(k, 0.0))
        out[n] = row
    return out


def _install_solver_profiler(theory: SU3WilsonNf2):
    stats: Dict[str, Dict[str, float]] = {}

    def _wrap_method(name: str):
        if not hasattr(theory, name):
            return
        orig = getattr(theory, name)
        stats[name] = {"time": 0.0, "calls": 0.0}

        def wrapped(*args, _orig=orig, _name=name, **kwargs):
            t0 = time.perf_counter()
            y = _orig(*args, **kwargs)
            y = jax.block_until_ready(y)
            stats[_name]["time"] += float(time.perf_counter() - t0)
            stats[_name]["calls"] += 1.0
            return y

        setattr(theory, name, wrapped)

    for nm in (
        "solve_eop_even_normal",
        "solve_eop_even_normal_with_guess",
        "solve_eop_even_direct",
        "solve_eop_even_dagger",
        "solve_normal",
        "solve_normal_with_guess",
        "solve_direct",
        "solve_dagger",
        "solve_direct_eo",
        "solve_dagger_eo",
    ):
        _wrap_method(nm)

    return stats


def _install_monomial_profiler(theory: SU3WilsonNf2):
    stats: Dict[str, Dict[str, float]] = {}

    for m in theory.hamiltonian.monomials:
        name = str(getattr(m, "name", m.__class__.__name__))
        stats[name] = {
            "force_time": 0.0,
            "force_calls": 0.0,
            "action_time": 0.0,
            "action_calls": 0.0,
        }
        rec = stats[name]
        orig_force = m.force
        orig_action = m.action

        def force_wrapped(q, _orig=orig_force, _rec=rec):
            t0 = time.perf_counter()
            y = _orig(q)
            y = jax.block_until_ready(y)
            _rec["force_time"] += float(time.perf_counter() - t0)
            _rec["force_calls"] += 1.0
            return y

        def action_wrapped(q, _orig=orig_action, _rec=rec):
            t0 = time.perf_counter()
            y = _orig(q)
            y = jax.block_until_ready(y)
            _rec["action_time"] += float(time.perf_counter() - t0)
            _rec["action_calls"] += 1.0
            return y

        m.force = force_wrapped
        m.action = action_wrapped

    return stats


def _default_output_path(ckpt_path: Path) -> Path:
    if ckpt_path.parent.name == "ckpts" and ckpt_path.parent.parent != ckpt_path.parent:
        outdir = ckpt_path.parent.parent / "profiles"
    else:
        outdir = ckpt_path.parent / "profiles"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return outdir / f"inversion_profile_from_ckpt_{stamp}.json"


def _aggregate_solver_methods(per_traj: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    methods = sorted({k for r in per_traj for k in r.get("solver", {}).keys()})
    out: Dict[str, Dict[str, float]] = {}
    for k in methods:
        t = np.asarray([r["solver"].get(k, {}).get("time", 0.0) for r in per_traj], dtype=np.float64)
        c = np.asarray([r["solver"].get(k, {}).get("calls", 0.0) for r in per_traj], dtype=np.float64)
        out[k] = {
            "mean_sec_per_traj": float(np.mean(t)) if t.size else 0.0,
            "mean_calls_per_traj": float(np.mean(c)) if c.size else 0.0,
            "mean_ms_per_call": float(1e3 * np.mean(t / np.maximum(c, 1e-30))) if t.size else 0.0,
        }
    return out


def _aggregate_monomial_methods(per_traj: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    names = sorted({k for r in per_traj for k in r.get("monomial", {}).keys()})
    out: Dict[str, Dict[str, float]] = {}
    for k in names:
        ft = np.asarray([r["monomial"].get(k, {}).get("force_time", 0.0) for r in per_traj], dtype=np.float64)
        fc = np.asarray([r["monomial"].get(k, {}).get("force_calls", 0.0) for r in per_traj], dtype=np.float64)
        at = np.asarray([r["monomial"].get(k, {}).get("action_time", 0.0) for r in per_traj], dtype=np.float64)
        ac = np.asarray([r["monomial"].get(k, {}).get("action_calls", 0.0) for r in per_traj], dtype=np.float64)
        out[k] = {
            "mean_force_sec_per_traj": float(np.mean(ft)) if ft.size else 0.0,
            "mean_force_calls_per_traj": float(np.mean(fc)) if fc.size else 0.0,
            "mean_force_ms_per_call": float(1e3 * np.mean(ft / np.maximum(fc, 1e-30))) if ft.size else 0.0,
            "mean_action_sec_per_traj": float(np.mean(at)) if at.size else 0.0,
            "mean_action_calls_per_traj": float(np.mean(ac)) if ac.size else 0.0,
            "mean_action_ms_per_call": float(1e3 * np.mean(at / np.maximum(ac, 1e-30))) if at.size else 0.0,
        }
    return out


def _print_summary(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    print("PROFILE SUMMARY")
    print(f"  checkpoint: {s['checkpoint']}")
    print(f"  trajectories (warm/meas): {s['n_warm']} / {s['n_meas']}")
    print(f"  nmd/integrator: {s['nmd_profiled']} / {s['integrator']}")
    print(f"  mean traj sec: {s['mean_traj_sec']:.6f} (std={s['std_traj_sec']:.6f})")
    print(f"  mean solver sec: {s['mean_solver_sec']:.6f} ({100.0 * s['mean_solver_frac']:.2f}%)")
    print(
        "  mean monomial force+action sec:"
        f" {s['mean_monomial_force_action_sec']:.6f}"
        f" ({100.0 * s['mean_monomial_force_action_frac']:.2f}%)"
    )


def main():
    ap = argparse.ArgumentParser(description="Profile inversion vs trajectory cost from a checkpoint")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to mcmc.py checkpoint .pkl")
    ap.add_argument("--warmup", type=int, default=2, help="Warm trajectories before timed segment")
    ap.add_argument("--meas", type=int, default=4, help="Timed trajectories")
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
    ap.add_argument("--integrator", type=str, default="", choices=["", "leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--nmd", type=int, default=0, help="Override integrator nmd; 0 uses checkpoint/state value")
    ap.add_argument("--tau", type=float, default=0.0, help="Override trajectory length tau; 0 uses checkpoint value")
    ap.add_argument("--solver-kind", type=str, default="", choices=["", "cg", "bicgstab", "gmres"])
    ap.add_argument("--solver-form", type=str, default="", choices=["", "normal", "split", "eo_split"])
    ap.add_argument("--solver-use-guess", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--solver-guess-mode", type=str, default="", choices=["", "last", "poly", "mre"])
    ap.add_argument("--solver-guess-history", type=int, default=0, help="Override chrono guess history; <=0 uses checkpoint value")
    ap.add_argument("--solver-tol", type=float, default=0.0, help="Override solver tolerance; <=0 uses checkpoint value")
    ap.add_argument("--solver-maxiter", type=int, default=0, help="Override solver maxiter; <=0 uses checkpoint value")
    ap.add_argument("--preconditioner", type=str, default="", choices=["", "none", "jacobi"])
    ap.add_argument("--gmres-restart", type=int, default=0, help="Override GMRES restart; <=0 uses checkpoint value")
    ap.add_argument("--gmres-solve-method", type=str, default="", help="Override GMRES solve method")
    ap.add_argument("--output", type=str, default="", help="Output JSON path; default under profiles/")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    if "q" not in ckpt:
        raise ValueError("Checkpoint missing gauge field key 'q'")

    cfg = dict(ckpt.get("config", {}))
    st = dict(ckpt.get("state", {}))

    shape = _parse_shape(_cfg_get(cfg, "run.shape", "8,8,8,16"))
    batch = int(_cfg_get(cfg, "run.batch", _cfg_get(cfg, "run.batch_size", 1)))
    layout = str(_cfg_get(cfg, "run.layout", "BMXYIJ"))
    exp_method = str(_cfg_get(cfg, "run.exp_method", "su3"))
    seed = int(_cfg_get(cfg, "run.seed", 0))

    beta = float(_cfg_get(cfg, "physics.beta", 5.7))
    mass = float(_cfg_get(cfg, "physics.mass", 0.05))
    wilson_r = float(_cfg_get(cfg, "physics.r", 1.0))
    fermion_bc = _parse_fermion_bc(_cfg_get(cfg, "physics.fermion_bc", "periodic"), nd=len(shape))

    solver_kind = str(args.solver_kind or _cfg_get(cfg, "solver.kind", "cg"))
    solver_form = str(args.solver_form or _cfg_get(cfg, "solver.form", "normal"))
    solver_use_guess_cfg = bool(_cfg_get(cfg, "solver.use_solver_guess", _cfg_get(cfg, "solver.use_guess", False)))
    solver_guess_mode_cfg = str(_cfg_get(cfg, "solver.guess_mode", _cfg_get(cfg, "solver.solver_guess_mode", "last")))
    solver_guess_history_cfg = int(_cfg_get(cfg, "solver.guess_history", _cfg_get(cfg, "solver.solver_guess_history", 1)))
    solver_use_guess = bool(args.solver_use_guess) if args.solver_use_guess is not None else solver_use_guess_cfg
    solver_guess_mode = str(args.solver_guess_mode or solver_guess_mode_cfg)
    solver_guess_history = int(args.solver_guess_history) if int(args.solver_guess_history) > 0 else solver_guess_history_cfg
    solver_tol_cfg = float(_cfg_get(cfg, "solver.tol", 1e-7))
    solver_maxiter_cfg = int(_cfg_get(cfg, "solver.maxiter", 1000))
    preconditioner_cfg = str(_cfg_get(cfg, "solver.preconditioner", "none"))
    gmres_restart_cfg = int(_cfg_get(cfg, "solver.gmres_restart", 32))
    gmres_solve_method_cfg = str(_cfg_get(cfg, "solver.gmres_solve_method", "batched"))
    solver_tol = float(args.solver_tol) if float(args.solver_tol) > 0.0 else solver_tol_cfg
    solver_maxiter = int(args.solver_maxiter) if int(args.solver_maxiter) > 0 else solver_maxiter_cfg
    preconditioner = str(args.preconditioner or preconditioner_cfg)
    gmres_restart = int(args.gmres_restart) if int(args.gmres_restart) > 0 else gmres_restart_cfg
    gmres_solve_method = str(args.gmres_solve_method or gmres_solve_method_cfg)

    include_gauge = bool(_cfg_get(cfg, "monomials.include_gauge", True))
    include_fermion = bool(_cfg_get(cfg, "monomials.include_fermion", True))
    fermion_kind = str(_cfg_get(cfg, "monomials.fermion_kind", "eo_preconditioned"))
    gauge_timescale = int(_cfg_get(cfg, "monomials.gauge_timescale", 0))
    fermion_timescale = int(_cfg_get(cfg, "monomials.fermion_timescale", 1))
    pf_refresh = str(_cfg_get(cfg, "monomials.pf_refresh", "heatbath"))
    pf_force_mode = str(_cfg_get(cfg, "monomials.pf_force_mode", "analytic"))
    pf_gamma = float(_cfg_get(cfg, "monomials.pf_gamma", 0.3))

    update_name = str(_cfg_get(cfg, "update.algorithm", "hmc")).lower()
    integrator_name = str(args.integrator or _cfg_get(cfg, "update.integrator", "forcegrad")).lower()
    tau = float(args.tau) if float(args.tau) > 0.0 else float(_cfg_get(cfg, "update.tau", 1.0))
    nmd_default = int(st.get("nmd", _cfg_get(cfg, "update.nmd", 8)))
    nmd = int(args.nmd) if int(args.nmd) > 0 else int(nmd_default)
    smd_gamma = float(_cfg_get(cfg, "update.smd_gamma", 0.3))
    smd_accept_reject = bool(_cfg_get(cfg, "update.smd_accept_reject", True))

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))
    print("Checkpoint:", ckpt_path)
    print(f"Update/integrator: {update_name}/{integrator_name} tau={tau} nmd={nmd}")
    print(
        "Solver:"
        f" kind={solver_kind}"
        f" form={solver_form}"
        f" use_guess={solver_use_guess}"
        f" guess_mode={solver_guess_mode}"
        f" guess_history={solver_guess_history}"
        f" tol={solver_tol}"
        f" maxiter={solver_maxiter}"
        f" preconditioner={preconditioner}"
    )
    print(f"Warm/meas trajectories: {int(args.warmup)}/{int(args.meas)}")

    theory = SU3WilsonNf2(
        lattice_shape=shape,
        beta=beta,
        batch_size=batch,
        layout=layout,
        seed=seed,
        exp_method=exp_method,
        mass=mass,
        wilson_r=wilson_r,
        cg_tol=solver_tol,
        cg_maxiter=solver_maxiter,
        solver_kind=solver_kind,
        solver_form=solver_form,
        use_solver_guess=solver_use_guess,
        solver_guess_mode=solver_guess_mode,
        solver_guess_history=solver_guess_history,
        preconditioner_kind=preconditioner,
        gmres_restart=gmres_restart,
        gmres_solve_method=gmres_solve_method,
        fermion_bc=fermion_bc,
        include_gauge_monomial=include_gauge,
        include_fermion_monomial=include_fermion,
        fermion_monomial_kind=fermion_kind,
        gauge_timescale=gauge_timescale,
        fermion_timescale=fermion_timescale,
        pseudofermion_refresh=pf_refresh,
        pseudofermion_gamma=pf_gamma,
        pseudofermion_force_mode=pf_force_mode,
        smd_gamma=smd_gamma,
        auto_refresh_pseudofermions=True,
    )

    theory.kinetic = jax.jit(theory.kinetic)
    if hasattr(theory, "refresh_p_with_key"):
        theory.refresh_p_with_key = jax.jit(theory.refresh_p_with_key)

    I = _build_integrator(integrator_name, theory, nmd, tau)
    if update_name == "hmc":
        chain = hmc(T=theory, I=I, verbose=False, use_fast_jit=False)
    elif update_name in ("smd", "ghmc"):
        chain = SMD(
            T=theory,
            I=I,
            gamma=smd_gamma,
            accept_reject=bool(smd_accept_reject),
            verbose=False,
            use_fast_jit=False,
        )
    else:
        raise ValueError(f"Unsupported checkpoint update algorithm: {update_name}")

    q = jax.device_put(np.asarray(ckpt["q"]))
    theory_key = ckpt.get("theory_key")
    if theory_key is not None:
        theory.key = jax.device_put(np.asarray(theory_key, dtype=np.uint32))
    update_key = ckpt.get("update_key", ckpt.get("hmc_key"))
    if update_key is not None:
        chain.key = jax.device_put(np.asarray(update_key, dtype=np.uint32))
    if hasattr(chain, "p"):
        p_arr = ckpt.get("update_momentum", None)
        chain.p = None if p_arr is None else jax.device_put(np.asarray(p_arr))
    chain.AcceptReject = list(np.asarray(ckpt.get("accept_reject", []), dtype=np.float64))

    solver_stats = _install_solver_profiler(theory)
    monomial_stats = _install_monomial_profiler(theory)

    n_warm = max(0, int(args.warmup))
    n_meas = max(1, int(args.meas))

    for i in range(n_warm):
        t0 = time.perf_counter()
        q = chain.evolve(q, 1)
        q = jax.block_until_ready(q)
        dt = 1e3 * (time.perf_counter() - t0)
        print(f"warm {i + 1}/{n_warm}: {dt:.3f} ms")

    per_traj: List[Dict[str, Any]] = []
    for i in range(n_meas):
        m0 = _snapshot_nested_numeric(monomial_stats)
        s0 = _snapshot_nested_numeric(solver_stats)
        t0 = time.perf_counter()
        q = chain.evolve(q, 1)
        q = jax.block_until_ready(q)
        traj_sec = float(time.perf_counter() - t0)
        md = _diff_nested_numeric(m0, _snapshot_nested_numeric(monomial_stats))
        sd = _diff_nested_numeric(s0, _snapshot_nested_numeric(solver_stats))

        solver_sec = float(sum(v.get("time", 0.0) for v in sd.values()))
        mon_force_sec = float(sum(v.get("force_time", 0.0) for v in md.values()))
        mon_action_sec = float(sum(v.get("action_time", 0.0) for v in md.values()))
        mon_total_sec = float(mon_force_sec + mon_action_sec)
        solver_frac = float(solver_sec / (traj_sec + 1e-30))
        mon_frac = float(mon_total_sec / (traj_sec + 1e-30))
        acc_last = float(chain.AcceptReject[-1]) if chain.AcceptReject else float("nan")

        rec = {
            "traj_idx": int(i),
            "traj_sec": traj_sec,
            "solver_sec": solver_sec,
            "solver_frac": solver_frac,
            "mon_force_sec": mon_force_sec,
            "mon_action_sec": mon_action_sec,
            "mon_total_sec": mon_total_sec,
            "mon_total_frac": mon_frac,
            "accept_last": acc_last,
            "solver": sd,
            "monomial": md,
        }
        per_traj.append(rec)
        print(
            f"meas {i + 1}/{n_meas}:"
            f" traj={1e3 * traj_sec:.3f} ms"
            f" solver={1e3 * solver_sec:.3f} ms ({100.0 * solver_frac:.1f}%)"
            f" force+action={1e3 * mon_total_sec:.3f} ms"
            f" acc={acc_last:.3f}"
        )

    arr_traj = np.asarray([r["traj_sec"] for r in per_traj], dtype=np.float64)
    arr_sol = np.asarray([r["solver_sec"] for r in per_traj], dtype=np.float64)
    arr_mon = np.asarray([r["mon_total_sec"] for r in per_traj], dtype=np.float64)
    summary = {
        "checkpoint": str(ckpt_path),
        "nmd_profiled": int(nmd),
        "integrator": str(integrator_name),
        "update": str(update_name),
        "solver_kind": str(solver_kind),
        "solver_form": str(solver_form),
        "solver_use_guess": bool(solver_use_guess),
        "solver_guess_mode": str(solver_guess_mode),
        "solver_guess_history": int(solver_guess_history),
        "solver_tol": float(solver_tol),
        "solver_maxiter": int(solver_maxiter),
        "solver_preconditioner": str(preconditioner),
        "n_warm": int(n_warm),
        "n_meas": int(n_meas),
        "mean_traj_sec": float(np.mean(arr_traj)),
        "std_traj_sec": float(np.std(arr_traj, ddof=1)) if arr_traj.size > 1 else 0.0,
        "mean_solver_sec": float(np.mean(arr_sol)),
        "mean_solver_frac": float(np.mean(arr_sol / (arr_traj + 1e-30))),
        "mean_monomial_force_action_sec": float(np.mean(arr_mon)),
        "mean_monomial_force_action_frac": float(np.mean(arr_mon / (arr_traj + 1e-30))),
        "global_acceptance_after_profile": float(chain.calc_Acceptance()),
    }
    out = {
        "summary": summary,
        "solver_methods": _aggregate_solver_methods(per_traj),
        "monomial_methods": _aggregate_monomial_methods(per_traj),
        "per_traj": per_traj,
    }

    out_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    _print_summary(out)
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main()

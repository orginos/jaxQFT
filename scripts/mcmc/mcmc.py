#!/usr/bin/env python3
"""Unified MCMC driver with inline measurement support.

Current production target:
- Theory family: SU3
- Theory: SU3 Wilson Nf=2

Design points:
- Separate checkpoint state from saved gauge configurations.
- Multi-phase control: warmup(no-AR) -> warmup(AR) -> measurement.
- Inline measurement pipeline with step-local data handoff via context.
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

# Keep macOS default stable unless explicitly overridden.
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


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
from jaxqft.core.measurements import MeasurementContext, build_inline_measurements, run_inline_measurements
from jaxqft.core.update import SMD, hmc
from jaxqft.io import decode_scidac_gauge, decode_scidac_momentum, decode_scidac_pseudofermion
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2
from jaxqft.models.su3_ym import SU3YangMills
from jaxqft.stats import integrated_autocorr_time


TEMPLATE_TOML = """# jaxQFT unified MCMC control file
control_version = 1

[run]
theory_family = "su3"
theory = "su3_wilson_nf2"
seed = 0
shape = [8, 8, 8, 16]
batch = 1
layout = "BMXYIJ"  # BMXYIJ | BXYMIJ | auto
exp_method = "su3"
hot_start_scale = 0.2

[input]
init_cfg_lime = ""                    # SciDAC/ILDG cfg .lime
init_mom_lime = ""                    # SciDAC mom .lime (SMD/GHMC only)
init_pf_lime = ""                     # SciDAC pseudofermion .lime
init_pf_field_index = -1              # -1 => auto-select PF leaf
init_pf_cb_fix = "auto"               # none | auto | shiftx1
init_use_loaded_pf_first_traj = true  # skip one PF refresh after load

[physics]
beta = 5.7
mass = 0.05
r = 1.0

[solver]
kind = "cg"               # cg | bicgstab | gmres
form = "normal"           # normal | split | eo_split
tol = 1e-7
maxiter = 1000
preconditioner = "none"   # none | jacobi
gmres_restart = 32
gmres_solve_method = "batched"

[monomials]
include_gauge = true
include_fermion = true
fermion_kind = "eo_preconditioned"   # unpreconditioned | eo_preconditioned
gauge_timescale = 0
fermion_timescale = 1
pf_refresh = "heatbath"              # heatbath | ou | auto
pf_force_mode = "analytic"           # autodiff | analytic
pf_gamma = 0.3

[update]
algorithm = "hmc"          # hmc | smd | ghmc
integrator = "forcegrad"   # leapfrog | minnorm2 | forcegrad | minnorm4pf4
tau = 0.5
nmd = 12
smd_gamma = 0.3
smd_accept_reject = true

[phases]
warmup_no_ar = 0
warmup_ar = 100
measure = 1000
skip = 1
warmup_log_every = 5
measure_log_every = 10
adapt_nmd = true
target_accept = 0.90
adapt_interval = 5
adapt_window = 10
adapt_tol = 0.03
nmd_min = 1
nmd_max = 256
nmd_step = 1

[output]
checkpoint_path = "mcmc_ckpt.pkl"
checkpoint_every = 0
resume = ""
config_dir = ""              # optional directory for saved gauge configs
save_config_every = 0        # every N measurement steps (0 disables)
iat_method = "ips"           # ips | sokal | gamma
iat_max_lag = 0              # 0 => auto

[[measurements.inline]]
type = "plaquette"
name = "plaquette"
every = 1
"""


def _avg_err(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size < 2:
        return float(arr.mean()), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size - 1))


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


def _integrator_order(name: str) -> int:
    return 4 if name in ("forcegrad", "minnorm4pf4") else 2


def _default_target_accept(name: str) -> float:
    return 0.90 if _integrator_order(name) == 4 else 0.68


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


def _run_one_trajectory(q, chain, warmup: bool):
    if isinstance(chain, SMD):
        return chain.evolve(q, 1, warmup=bool(warmup))
    return chain.evolve(q, 1)


def _run_one_trajectory_hmc_no_ar(q, chain):
    if isinstance(chain, SMD):
        return chain.evolve(q, 1, warmup=True)
    if hasattr(chain, "_prepare_trajectory"):
        chain._prepare_trajectory(q)
    p0 = chain.T.refreshP()
    _, q1 = chain.I.integrate(p0, q)
    return jax.block_until_ready(q1)


def _save_checkpoint(path: str, q, theory, chain, state: Dict[str, Any], config: Dict[str, Any]):
    update_p = getattr(chain, "p", None)
    payload = {
        "q": np.asarray(q),
        "theory_key": np.asarray(theory.key),
        "update_key": np.asarray(chain.key),
        "update_momentum": (None if update_p is None else np.asarray(update_p)),
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


def _save_configuration(config_dir: str, meas_idx: int, q, metadata: Mapping[str, Any]) -> str:
    os.makedirs(config_dir, exist_ok=True)
    stem = f"cfg_{int(meas_idx):07d}"
    q_path = os.path.join(config_dir, stem + ".npy")
    m_path = os.path.join(config_dir, stem + ".json")
    np.save(q_path, np.asarray(q))
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)
    return q_path


def _load_toml(path: str) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable; Python 3.11+ is required for TOML control files")
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser(description="Unified jaxQFT MCMC driver")
    ap.add_argument("--config", type=str, default="", help="TOML control file")
    ap.add_argument("--write-template", type=str, default="", help="write template TOML and exit")
    ap.add_argument("--resume", type=str, default="", help="override resume path from control file")
    ap.add_argument("--save", type=str, default="", help="override checkpoint path from control file")
    ap.add_argument("--cpu-threads", type=int, default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0))
    ap.add_argument("--cpu-onednn", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    if args.write_template:
        out = Path(args.write_template)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(TEMPLATE_TOML, encoding="utf-8")
        print(f"Wrote template: {out}")
        return

    if not args.config:
        raise ValueError("Provide --config <control.toml> or --write-template <path>")

    cfg = _load_toml(args.config)

    theory_family = str(_cfg_get(cfg, "run.theory_family", "su3")).lower()
    theory_name = str(_cfg_get(cfg, "run.theory", "su3_wilson_nf2")).lower()
    if theory_family != "su3" or theory_name != "su3_wilson_nf2":
        raise ValueError(
            "mcmc.py currently supports run.theory_family='su3' and run.theory='su3_wilson_nf2' only"
        )

    seed = int(_cfg_get(cfg, "run.seed", 0))
    lattice_shape = _parse_shape(_cfg_get(cfg, "run.shape", [4, 4, 4, 8]))
    batch = int(_cfg_get(cfg, "run.batch", 1))
    layout = str(_cfg_get(cfg, "run.layout", "BMXYIJ"))
    exp_method = str(_cfg_get(cfg, "run.exp_method", "su3"))
    hot_start_scale = float(_cfg_get(cfg, "run.hot_start_scale", 0.2))

    init_cfg_lime = str(_cfg_get(cfg, "input.init_cfg_lime", "")).strip()
    init_mom_lime = str(_cfg_get(cfg, "input.init_mom_lime", "")).strip()
    init_pf_lime = str(_cfg_get(cfg, "input.init_pf_lime", "")).strip()
    init_pf_field_index = int(_cfg_get(cfg, "input.init_pf_field_index", -1))
    init_pf_cb_fix = str(_cfg_get(cfg, "input.init_pf_cb_fix", "auto")).strip().lower()
    init_use_loaded_pf_first_traj = bool(_cfg_get(cfg, "input.init_use_loaded_pf_first_traj", True))
    if init_pf_cb_fix not in ("none", "auto", "shiftx1"):
        raise ValueError(f"input.init_pf_cb_fix must be one of none/auto/shiftx1, got {init_pf_cb_fix!r}")

    beta = float(_cfg_get(cfg, "physics.beta", 5.8))
    mass = float(_cfg_get(cfg, "physics.mass", 0.05))
    wilson_r = float(_cfg_get(cfg, "physics.r", 1.0))

    solver_kind = str(_cfg_get(cfg, "solver.kind", "cg"))
    solver_form = str(_cfg_get(cfg, "solver.form", "normal"))
    solver_tol = float(_cfg_get(cfg, "solver.tol", 1e-8))
    solver_maxiter = int(_cfg_get(cfg, "solver.maxiter", 500))
    preconditioner = str(_cfg_get(cfg, "solver.preconditioner", "none"))
    gmres_restart = int(_cfg_get(cfg, "solver.gmres_restart", 32))
    gmres_solve_method = str(_cfg_get(cfg, "solver.gmres_solve_method", "batched"))

    include_gauge = bool(_cfg_get(cfg, "monomials.include_gauge", True))
    include_fermion = bool(_cfg_get(cfg, "monomials.include_fermion", True))
    fermion_kind = str(_cfg_get(cfg, "monomials.fermion_kind", "eo_preconditioned"))
    gauge_timescale = int(_cfg_get(cfg, "monomials.gauge_timescale", 0))
    fermion_timescale = int(_cfg_get(cfg, "monomials.fermion_timescale", 1))
    pf_refresh = str(_cfg_get(cfg, "monomials.pf_refresh", "auto"))
    pf_force_mode = str(_cfg_get(cfg, "monomials.pf_force_mode", "analytic"))
    pf_gamma = _cfg_get(cfg, "monomials.pf_gamma", None)

    update_name = str(_cfg_get(cfg, "update.algorithm", "hmc")).lower()
    if update_name == "ghmc":
        update_name = "smd"
    integrator_name = str(_cfg_get(cfg, "update.integrator", "minnorm2"))
    tau = float(_cfg_get(cfg, "update.tau", 1.0))
    nmd = int(_cfg_get(cfg, "update.nmd", 8))
    smd_gamma = float(_cfg_get(cfg, "update.smd_gamma", 0.3))
    smd_accept_reject = bool(_cfg_get(cfg, "update.smd_accept_reject", True))

    warmup_no_ar = int(_cfg_get(cfg, "phases.warmup_no_ar", 0))
    warmup_ar = int(_cfg_get(cfg, "phases.warmup_ar", 20))
    meas = int(_cfg_get(cfg, "phases.measure", 50))
    skip = int(_cfg_get(cfg, "phases.skip", 5))
    warmup_log_every = int(_cfg_get(cfg, "phases.warmup_log_every", 5))
    measure_log_every = int(_cfg_get(cfg, "phases.measure_log_every", 10))
    adapt_nmd = bool(_cfg_get(cfg, "phases.adapt_nmd", True))
    target_accept = _cfg_get(cfg, "phases.target_accept", None)
    if target_accept is None:
        target_accept = _default_target_accept(integrator_name)
    target_accept = float(target_accept)
    adapt_interval = int(_cfg_get(cfg, "phases.adapt_interval", 5))
    adapt_window = int(_cfg_get(cfg, "phases.adapt_window", 10))
    adapt_tol = float(_cfg_get(cfg, "phases.adapt_tol", 0.03))
    nmd_min = int(_cfg_get(cfg, "phases.nmd_min", 1))
    nmd_max = int(_cfg_get(cfg, "phases.nmd_max", 256))
    nmd_step = int(_cfg_get(cfg, "phases.nmd_step", 1))

    checkpoint_path = str(_cfg_get(cfg, "output.checkpoint_path", "mcmc_ckpt.pkl"))
    checkpoint_every = int(_cfg_get(cfg, "output.checkpoint_every", 0))
    resume_path = str(_cfg_get(cfg, "output.resume", ""))
    if args.resume:
        resume_path = str(args.resume)
    if args.save:
        checkpoint_path = str(args.save)
    config_dir = str(_cfg_get(cfg, "output.config_dir", ""))
    save_config_every = int(_cfg_get(cfg, "output.save_config_every", 0))
    iat_method = str(_cfg_get(cfg, "output.iat_method", "ips"))
    iat_max_lag = int(_cfg_get(cfg, "output.iat_max_lag", 0))

    inline_specs = _cfg_get(cfg, "measurements.inline", [{"type": "plaquette", "name": "plaquette", "every": 1}])
    if not isinstance(inline_specs, list):
        raise ValueError("measurements.inline must be a list of tables")
    measurements = build_inline_measurements(inline_specs)
    mctx = MeasurementContext()

    if update_name != "hmc" and warmup_no_ar > 0:
        print(f"Note: warmup_no_ar is HMC-only; using 0 for update={update_name}.")
        warmup_no_ar = 0

    if pf_refresh == "auto":
        pf_refresh = "ou" if update_name == "smd" else "heatbath"
    if pf_gamma is None:
        pf_gamma = smd_gamma if pf_refresh.lower() == "ou" else 0.3
    pf_gamma = float(pf_gamma)

    ckpt = _load_checkpoint(resume_path) if resume_path else None
    if ckpt is not None and (init_cfg_lime or init_mom_lime or init_pf_lime):
        print(
            "Note: input.init_*_lime options are ignored on resume "
            "(checkpoint state takes precedence)."
        )
        init_cfg_lime = ""
        init_mom_lime = ""
        init_pf_lime = ""

    if layout == "auto" and ckpt is None:
        timings = SU3YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=max(1, batch),
            n_iter=3,
            seed=seed,
            exp_method=exp_method,
        )
        layout = min(timings, key=timings.get)
        print("Layout benchmark (s/action):", timings)
        print("Selected layout:", layout)

    theory = SU3WilsonNf2(
        lattice_shape=lattice_shape,
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
        preconditioner_kind=preconditioner,
        gmres_restart=gmres_restart,
        gmres_solve_method=gmres_solve_method,
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
    elif update_name == "smd":
        chain = SMD(
            T=theory,
            I=I,
            gamma=smd_gamma,
            accept_reject=bool(smd_accept_reject),
            verbose=False,
            use_fast_jit=False,
        )
    else:
        raise ValueError(f"Unknown update algorithm: {update_name}")

    q = theory.hotStart(scale=hot_start_scale)

    warmup_noar_done = 0
    warmup_ar_done = 0
    meas_done = 0
    warmup_ar_accepts: List[float] = []
    meas_step_accepts: List[float] = []
    warmup_noar_traj_ms: List[float] = []
    warmup_ar_traj_ms: List[float] = []
    meas_traj_ms: List[float] = []
    inline_records: List[Dict[str, Any]] = []
    inline_history: Dict[str, List[float]] = {}
    saved_configs = 0

    if ckpt is not None:
        q = jax.device_put(np.asarray(ckpt["q"]))
        theory.key = jax.device_put(np.asarray(ckpt["theory_key"], dtype=np.uint32))
        update_key = ckpt.get("update_key", ckpt.get("hmc_key"))
        chain.key = jax.device_put(np.asarray(update_key, dtype=np.uint32))
        if hasattr(chain, "p"):
            p_arr = ckpt.get("update_momentum", None)
            chain.p = None if p_arr is None else jax.device_put(np.asarray(p_arr))
        chain.AcceptReject = list(np.asarray(ckpt.get("accept_reject", []), dtype=np.float64))
        st = ckpt.get("state", {})
        warmup_noar_done = int(st.get("warmup_noar_done", 0))
        warmup_ar_done = int(st.get("warmup_ar_done", 0))
        meas_done = int(st.get("meas_done", 0))
        nmd = int(st.get("nmd", nmd))
        chain.I = _build_integrator(integrator_name, theory, nmd, tau)
        warmup_ar_accepts = list(st.get("warmup_ar_accepts", []))
        meas_step_accepts = list(st.get("meas_step_accepts", []))
        warmup_noar_traj_ms = [float(x) for x in st.get("warmup_noar_traj_ms", [])]
        warmup_ar_traj_ms = [float(x) for x in st.get("warmup_ar_traj_ms", [])]
        meas_traj_ms = [float(x) for x in st.get("meas_traj_ms", [])]
        inline_records = list(st.get("inline_records", []))
        inline_history = {str(k): [float(vv) for vv in vv_list] for k, vv_list in st.get("inline_history", {}).items()}
        saved_configs = int(st.get("saved_configs", 0))
        mctx.state = dict(st.get("measurement_context_state", {}))
        print(f"Resumed from {resume_path} at noAR/AR/meas={warmup_noar_done}/{warmup_ar_done}/{meas_done}, nmd={nmd}")
    else:
        if init_cfg_lime:
            q_loaded = decode_scidac_gauge(init_cfg_lime, batch_size=batch)
            if tuple(q_loaded.shape) != tuple(q.shape):
                raise ValueError(
                    f"Loaded cfg shape mismatch: {tuple(q_loaded.shape)} vs expected {tuple(q.shape)}"
                )
            q = jax.device_put(np.asarray(q_loaded))

        if init_mom_lime:
            p_loaded, p_info = decode_scidac_momentum(init_mom_lime, batch_size=batch)
            if hasattr(chain, "p"):
                if tuple(p_loaded.shape) != tuple(q.shape):
                    raise ValueError(
                        f"Loaded momentum shape mismatch: {tuple(p_loaded.shape)} vs expected {tuple(q.shape)}"
                    )
                chain.p = jax.device_put(np.asarray(p_loaded, dtype=np.complex64))
                print(
                    "  loaded momenta:"
                    f" antiH={float(p_info.get('rel_antihermitian_error', float('nan'))):.3e}"
                    f" max|tr|={float(p_info.get('max_abs_trace', float('nan'))):.3e}"
                )
            else:
                print("  note: input.init_mom_lime provided, but update=hmc has no stored momentum state; ignoring")

        if init_pf_lime:
            pf_idx = None if int(init_pf_field_index) < 0 else int(init_pf_field_index)
            pf_loaded, pf_meta = decode_scidac_pseudofermion(
                init_pf_lime,
                field_index=pf_idx,
                batch_size=batch,
            )
            pf_loaded = np.asarray(pf_loaded)
            expected_pf_shape = tuple(int(v) for v in theory.fermion_shape())
            if tuple(pf_loaded.shape) != expected_pf_shape:
                raise ValueError(
                    "Loaded pseudofermion shape mismatch: "
                    f"{tuple(pf_loaded.shape)} vs expected {expected_pf_shape}"
                )

            if theory.fermion_monomial_kind == "eo_preconditioned":
                pe = np.asarray(theory._project_even(jax.device_put(pf_loaded)))
                po = np.asarray(theory._project_odd(jax.device_put(pf_loaded)))
                ne = float(np.linalg.norm(pe))
                no = float(np.linalg.norm(po))
                do_shift = False
                if init_pf_cb_fix == "shiftx1":
                    do_shift = True
                elif init_pf_cb_fix == "auto":
                    do_shift = (no > 0.0) and (ne <= 1e-12 * no)
                if do_shift:
                    pf_loaded = np.roll(pf_loaded, shift=1, axis=1)
                    pe = np.asarray(theory._project_even(jax.device_put(pf_loaded)))
                    po = np.asarray(theory._project_odd(jax.device_put(pf_loaded)))
                    ne = float(np.linalg.norm(pe))
                    no = float(np.linalg.norm(po))
                    print(
                        "  loaded pseudofermion checkerboard fix: applied x-shift(+1)"
                        f" even_norm={ne:.6e} odd_norm={no:.6e}"
                    )
                else:
                    print(
                        "  loaded pseudofermion checkerboard:"
                        f" even_norm={ne:.6e} odd_norm={no:.6e}"
                    )

            theory.set_loaded_pseudofermion(
                jax.device_put(np.asarray(pf_loaded, dtype=np.complex64)),
                skip_next_refresh=bool(init_use_loaded_pf_first_traj),
            )
            print(
                "  loaded pseudofermion:"
                f" datatype={pf_meta.datatype}"
                f" precision={pf_meta.precision}"
                f" spins={pf_meta.spins}"
                f" colors={pf_meta.colors}"
                f" recordtype={pf_meta.recordtype}"
                f" skip_first_refresh={bool(init_use_loaded_pf_first_traj)}"
            )

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))
    print("Run config:")
    print(f"  theory: {theory_family}/{theory_name}")
    print(f"  shape: {lattice_shape}")
    print(f"  beta/mass/r: {beta} / {mass} / {wilson_r}")
    print(f"  hot_start_scale: {hot_start_scale}")
    print(f"  update: {update_name}")
    print(f"  integrator: {integrator_name} (tau={tau}, nmd={nmd})")
    print(f"  warmup(no-AR/AR)/meas/skip: {warmup_no_ar}/{warmup_ar}/{meas}/{skip}")
    print(f"  monomials: {', '.join(theory.hamiltonian.monomial_names())}")
    print(f"  inline measurements: {[m.name for m in measurements]}")
    if init_cfg_lime:
        print(f"  init cfg lime: {init_cfg_lime}")
    if init_mom_lime:
        print(f"  init mom lime: {init_mom_lime}")
    if init_pf_lime:
        print(
            "  init pf lime:"
            f" path={init_pf_lime}"
            f" field_index={init_pf_field_index}"
            f" cb_fix={init_pf_cb_fix}"
            f" use_first_traj={bool(init_use_loaded_pf_first_traj)}"
        )

    ckpt_config = dict(cfg)

    def maybe_save_checkpoint(reason: str):
        nonlocal ckpt_config
        if not checkpoint_path:
            return
        state = {
            "nmd": int(nmd),
            "warmup_noar_done": int(warmup_noar_done),
            "warmup_ar_done": int(warmup_ar_done),
            "meas_done": int(meas_done),
            "warmup_ar_accepts": [float(x) for x in warmup_ar_accepts],
            "meas_step_accepts": [float(x) for x in meas_step_accepts],
            "warmup_noar_traj_ms": [float(x) for x in warmup_noar_traj_ms],
            "warmup_ar_traj_ms": [float(x) for x in warmup_ar_traj_ms],
            "meas_traj_ms": [float(x) for x in meas_traj_ms],
            "inline_records": inline_records,
            "inline_history": inline_history,
            "saved_configs": int(saved_configs),
            "measurement_context_state": dict(mctx.state),
            "reason": str(reason),
        }
        _save_checkpoint(checkpoint_path, q, theory, chain, state, ckpt_config)

    # Phase 1: optional no-AR warmup.
    t0 = time.perf_counter()
    for k in range(int(warmup_noar_done), int(warmup_no_ar)):
        ts = time.perf_counter()
        q = _run_one_trajectory_hmc_no_ar(q, chain)
        dt = 1e3 * (time.perf_counter() - ts)
        warmup_noar_traj_ms.append(float(dt))
        warmup_noar_done = k + 1
        if warmup_noar_done % max(1, warmup_log_every) == 0 or warmup_noar_done == int(warmup_no_ar):
            print(f"warmup-noar k={warmup_noar_done}/{warmup_no_ar} nmd={nmd} traj_ms={dt:.3f}")
        if checkpoint_every > 0 and warmup_noar_done % int(checkpoint_every) == 0:
            maybe_save_checkpoint("warmup_no_ar")
    if warmup_no_ar > 0 and warmup_noar_done > 0:
        print(f"warmup-noar time/trajectory: {float(np.mean(np.asarray(warmup_noar_traj_ms, dtype=np.float64))):.3f} ms")
        print(
            f"warmup-noar time/trajectory (wall): {(time.perf_counter() - t0) * 1e3 / max(1, warmup_noar_done):.3f} ms"
        )

    # Phase 2: AR warmup with optional nmd adaptation.
    t1 = time.perf_counter()
    for k in range(int(warmup_ar_done), int(warmup_ar)):
        ts = time.perf_counter()
        prev = len(chain.AcceptReject)
        q = _run_one_trajectory(q, chain, warmup=True)
        dt = 1e3 * (time.perf_counter() - ts)
        warmup_ar_traj_ms.append(float(dt))
        new = np.asarray(chain.AcceptReject[prev:], dtype=np.float64)
        traj_acc = float(new.mean()) if new.size else float("nan")
        warmup_ar_accepts.append(traj_acc)
        warmup_ar_done = k + 1

        if adapt_nmd and adapt_interval > 0 and warmup_ar_done % int(adapt_interval) == 0:
            w = np.asarray(warmup_ar_accepts[-max(1, int(adapt_window)) :], dtype=np.float64)
            acc_w = float(np.nanmean(w))
            nmd_new = int(nmd)
            if acc_w < target_accept - float(adapt_tol):
                nmd_new = min(int(nmd_max), int(nmd + nmd_step))
            elif acc_w > target_accept + float(adapt_tol):
                nmd_new = max(int(nmd_min), int(nmd - nmd_step))
            if nmd_new != nmd:
                old = nmd
                nmd = int(nmd_new)
                chain.I = _build_integrator(integrator_name, theory, nmd, tau)
                print(f"warmup-ar adapt: k={warmup_ar_done} acc_win={acc_w:.3f} nmd {old} -> {nmd}")

        if warmup_ar_done % max(1, warmup_log_every) == 0 or warmup_ar_done == int(warmup_ar):
            w = np.asarray(warmup_ar_accepts[-max(1, int(adapt_window)) :], dtype=np.float64)
            print(
                f"warmup-ar k={warmup_ar_done}/{warmup_ar}"
                f" nmd={nmd}"
                f" traj_acc={traj_acc:.3f}"
                f" traj_ms={dt:.3f}"
                f" win_acc={float(np.nanmean(w)):.3f}"
                f" global_acc={chain.calc_Acceptance():.3f}"
            )
        if checkpoint_every > 0 and warmup_ar_done % int(checkpoint_every) == 0:
            maybe_save_checkpoint("warmup_ar")
    if warmup_ar > 0 and warmup_ar_done > 0:
        print(f"warmup-ar time/trajectory: {float(np.mean(np.asarray(warmup_ar_traj_ms, dtype=np.float64))):.3f} ms")
        print(f"warmup-ar time/trajectory (wall): {(time.perf_counter() - t1) * 1e3 / max(1, warmup_ar_done):.3f} ms")

    # Phase 3: measurement.
    t2 = time.perf_counter()
    for k in range(int(meas_done), int(meas)):
        recs = run_inline_measurements(measurements, step=int(k), q=q, theory=theory, context=mctx)
        for r in recs:
            row = {
                "phase": "measure",
                "step": int(r["step"]),
                "name": str(r["name"]),
                "values": dict(r["values"]),
            }
            inline_records.append(row)
            for fk, fv in row["values"].items():
                key = f"{row['name']}.{fk}"
                inline_history.setdefault(key, []).append(float(fv))

        if save_config_every > 0 and config_dir and (k + 1) % int(save_config_every) == 0:
            _save_configuration(
                config_dir=config_dir,
                meas_idx=k + 1,
                q=q,
                metadata={
                    "step": int(k + 1),
                    "nmd": int(nmd),
                    "integrator": str(integrator_name),
                    "tau": float(tau),
                    "theory": theory_name,
                },
            )
            saved_configs += 1

        step_accs = []
        step_times = []
        for _ in range(int(skip)):
            ts = time.perf_counter()
            prev = len(chain.AcceptReject)
            q = _run_one_trajectory(q, chain, warmup=False)
            dt = 1e3 * (time.perf_counter() - ts)
            meas_traj_ms.append(float(dt))
            step_times.append(float(dt))
            new = np.asarray(chain.AcceptReject[prev:], dtype=np.float64)
            if new.size:
                step_accs.append(float(new.mean()))
        step_acc = float(np.mean(step_accs)) if step_accs else float("nan")
        meas_step_accepts.append(step_acc)
        meas_done = k + 1

        if k % max(1, measure_log_every) == 0:
            p_last = inline_history.get("plaquette.value", [])
            p_val = float(p_last[-1]) if p_last else float("nan")
            print(
                f"meas k={k}"
                f" plaquette={p_val}"
                f" step_acc={step_acc:.3f}"
                f" step_traj_ms={float(np.mean(np.asarray(step_times, dtype=np.float64))) if step_times else float('nan'):.3f}"
                f" acc={chain.calc_Acceptance():.3f}"
            )
        if checkpoint_every > 0 and meas_done % int(checkpoint_every) == 0:
            maybe_save_checkpoint("measurement")
    if meas > 0 and meas_done > 0:
        print(f"measurement update time/trajectory: {float(np.mean(np.asarray(meas_traj_ms, dtype=np.float64))):.3f} ms")
        print(
            "measurement update time/trajectory (wall):"
            f" {(time.perf_counter() - t2) * 1e3 / max(1, int(meas_done) * int(skip)):.3f} ms"
        )

    print("Results:")
    for key in sorted(inline_history.keys()):
        m, e = _avg_err(inline_history[key])
        print(f"  {key}: {m} +/- {e}")

    plaq = np.asarray(inline_history.get("plaquette.value", []), dtype=np.float64)
    if plaq.size >= 2:
        iat = integrated_autocorr_time(
            plaq,
            method=str(iat_method),
            max_lag=(None if int(iat_max_lag) <= 0 else int(iat_max_lag)),
        )
        if bool(iat.get("ok", False)):
            nplaq = int(plaq.size)
            std_plaq = float(np.std(plaq, ddof=1))
            ess = float(iat["ess"])
            e_iat = float(std_plaq / np.sqrt(max(ess, 1e-12)))
            print(
                "  plaquette IAT:"
                f" tau_int={float(iat['tau_int']):.3f} samples,"
                f" ESS={ess:.2f}/{nplaq},"
                f" window={int(iat['window'])}, method={iat['method']}"
            )
            print(f"  plaquette IAT-corrected error: {e_iat}")
        else:
            print(f"  plaquette IAT: unavailable ({iat.get('message', 'unknown')})")

    if warmup_ar_accepts:
        print(f"  warmup-ar acc (mean): {float(np.nanmean(np.asarray(warmup_ar_accepts, dtype=np.float64))):.6f}")
    if meas_step_accepts:
        print(f"  meas acc (mean): {float(np.nanmean(np.asarray(meas_step_accepts, dtype=np.float64))):.6f}")
    print(f"  final nmd: {nmd}")
    print(f"  global acceptance: {chain.calc_Acceptance():.6f}")
    print(f"  saved configs: {saved_configs}")

    maybe_save_checkpoint("final")


if __name__ == "__main__":
    main()

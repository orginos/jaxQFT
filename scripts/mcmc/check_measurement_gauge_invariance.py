#!/usr/bin/env python3
"""Gauge-invariance check for inline measurements built from an MCMC TOML card.

This driver is generic over the gauge theory family. It:

1. builds a theory from the same control file format used by `scripts/mcmc/mcmc.py`,
2. draws a gauge field `U`,
3. draws a random site gauge transformation `Omega`,
4. compares each selected inline measurement on `U` and `U^Omega`.

Timing and inversion-cache metadata are ignored by default; the physical
measurement outputs must agree within the requested tolerances.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import tomllib  # py3.11+
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires Python 3.11+ with tomllib") from exc


def _cli_value(argv: List[str], flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


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

from jaxqft.core.measurements import MeasurementContext, build_inline_measurements, run_inline_measurements
from jaxqft.models.su2_wilson_nf2 import SU2WilsonNf2
from jaxqft.models.su2_ym import SU2YangMills, gauge_transform_links as su2_gauge_transform_links
from jaxqft.models.su2_ym import random_site_gauge_with_method as su2_random_site_gauge_with_method
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2
from jaxqft.models.su3_ym import SU3YangMills, gauge_transform_links as su3_gauge_transform_links
from jaxqft.models.su3_ym import random_site_gauge_with_method as su3_random_site_gauge_with_method
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_ym import U1YangMills, gauge_transform_links as u1_gauge_transform_links
from jaxqft.models.u1_ym import random_site_gauge as u1_random_site_gauge


DEFAULT_IGNORE_RE = r"^(timing_|wall_|inv_)"


def _load_toml(path: str) -> Mapping[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("rb") as f:
        return tomllib.load(f)


def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in str(dotted).split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_shape(raw: Any) -> Tuple[int, ...]:
    if isinstance(raw, str):
        vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    elif isinstance(raw, Sequence):
        vals = [int(v) for v in raw]
    else:
        raise ValueError(f"Invalid shape spec: {raw!r}")
    if not vals:
        raise ValueError("shape must contain at least one lattice extent")
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


def _parse_measurement_names(raw: Sequence[str]) -> Optional[Tuple[str, ...]]:
    vals: List[str] = []
    for item in raw:
        vals.extend(v.strip() for v in str(item).split(",") if v.strip())
    if not vals:
        return None
    return tuple(vals)


def _resolve_layout(
    theory_family: str,
    layout: str,
    lattice_shape: Tuple[int, ...],
    beta: float,
    batch: int,
    seed: int,
    exp_method: str,
) -> str:
    lay = str(layout).upper()
    lay = {"BM...IJ": "BMXYIJ", "B...MIJ": "BXYMIJ"}.get(lay, lay)
    if lay != "AUTO":
        return lay
    if theory_family == "u1":
        timings = U1YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            n_iter=2,
            seed=seed,
        )
    elif theory_family == "su2":
        timings = SU2YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            n_iter=2,
            seed=seed,
            exp_method=exp_method,
        )
    elif theory_family == "su3":
        timings = SU3YangMills.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            n_iter=2,
            seed=seed,
            exp_method=exp_method,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported theory family for layout auto-selection: {theory_family!r}")
    return str(min(timings, key=timings.get))


def _build_theory_from_cfg(cfg: Mapping[str, Any], *, seed: int):
    theory_family = str(_cfg_get(cfg, "run.theory_family", "su3")).lower()
    theory_name = str(_cfg_get(cfg, "run.theory", "su3_wilson_nf2")).lower()
    supported = {
        ("u1", "u1_ym"),
        ("u1", "u1_wilson_nf2"),
        ("su2", "su2_ym"),
        ("su2", "su2_wilson_nf2"),
        ("su3", "su3_ym"),
        ("su3", "su3_wilson_nf2"),
    }
    if (theory_family, theory_name) not in supported:
        opts = ", ".join(sorted(f"{fam}/{th}" for fam, th in supported))
        raise ValueError(f"Unsupported theory selection: {theory_family}/{theory_name}. Supported: {opts}")

    lattice_shape = _parse_shape(_cfg_get(cfg, "run.shape", [4, 4]))
    batch = int(_cfg_get(cfg, "run.batch", 1))
    beta = float(_cfg_get(cfg, "physics.beta", 1.0))
    mass = float(_cfg_get(cfg, "physics.mass", 0.0))
    wilson_r = float(_cfg_get(cfg, "physics.r", 1.0))
    layout = _resolve_layout(
        theory_family=theory_family,
        layout=str(_cfg_get(cfg, "run.layout", "BMXYIJ")),
        lattice_shape=lattice_shape,
        beta=beta,
        batch=int(batch),
        seed=int(seed),
        exp_method=str(_cfg_get(cfg, "run.exp_method", "expm")),
    )
    exp_method = str(_cfg_get(cfg, "run.exp_method", "expm"))
    hot_start_scale = float(_cfg_get(cfg, "run.hot_start_scale", 0.2))

    if (theory_family, theory_name) == ("u1", "u1_ym"):
        theory = U1YangMills(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
        )
    elif (theory_family, theory_name) == ("u1", "u1_wilson_nf2"):
        fermion_bc = _parse_fermion_bc(
            _cfg_get(cfg, "physics.fermion_bc", "antiperiodic-t"),
            nd=len(lattice_shape),
        )
        theory = U1WilsonNf2(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
            mass=mass,
            wilson_r=wilson_r,
            cg_tol=float(_cfg_get(cfg, "solver.tol", 1e-8)),
            cg_maxiter=int(_cfg_get(cfg, "solver.maxiter", 500)),
            solver_kind=str(_cfg_get(cfg, "solver.kind", "cg")),
            solver_form=str(_cfg_get(cfg, "solver.form", "normal")),
            preconditioner_kind=str(_cfg_get(cfg, "solver.preconditioner", "none")),
            gmres_restart=int(_cfg_get(cfg, "solver.gmres_restart", 32)),
            gmres_solve_method=str(_cfg_get(cfg, "solver.gmres_solve_method", "batched")),
            fermion_bc=fermion_bc,
            include_gauge_monomial=bool(_cfg_get(cfg, "monomials.include_gauge", True)),
            include_fermion_monomial=bool(_cfg_get(cfg, "monomials.include_fermion", True)),
            fermion_monomial_kind=str(_cfg_get(cfg, "monomials.fermion_kind", "eo_preconditioned")),
            gauge_timescale=int(_cfg_get(cfg, "monomials.gauge_timescale", 0)),
            fermion_timescale=int(_cfg_get(cfg, "monomials.fermion_timescale", 1)),
            pseudofermion_refresh=str(_cfg_get(cfg, "monomials.pf_refresh", "heatbath")),
            pseudofermion_gamma=float(_cfg_get(cfg, "monomials.pf_gamma", 0.3)),
            pseudofermion_force_mode=str(_cfg_get(cfg, "monomials.pf_force_mode", "analytic")),
            smd_gamma=float(_cfg_get(cfg, "update.smd_gamma", 0.3)),
            auto_refresh_pseudofermions=True,
        )
    elif (theory_family, theory_name) == ("su2", "su2_ym"):
        theory = SU2YangMills(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
            exp_method=exp_method,
        )
    elif (theory_family, theory_name) == ("su2", "su2_wilson_nf2"):
        theory = SU2WilsonNf2(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
            exp_method=exp_method,
            mass=mass,
            wilson_r=wilson_r,
            cg_tol=float(_cfg_get(cfg, "solver.tol", 1e-8)),
            cg_maxiter=int(_cfg_get(cfg, "solver.maxiter", 500)),
            solver_kind=str(_cfg_get(cfg, "solver.kind", "cg")),
            solver_form=str(_cfg_get(cfg, "solver.form", "normal")),
            preconditioner_kind=str(_cfg_get(cfg, "solver.preconditioner", "none")),
            gmres_restart=int(_cfg_get(cfg, "solver.gmres_restart", 32)),
            gmres_solve_method=str(_cfg_get(cfg, "solver.gmres_solve_method", "batched")),
            include_gauge_monomial=bool(_cfg_get(cfg, "monomials.include_gauge", True)),
            include_fermion_monomial=bool(_cfg_get(cfg, "monomials.include_fermion", True)),
            fermion_monomial_kind=str(_cfg_get(cfg, "monomials.fermion_kind", "eo_preconditioned")),
            gauge_timescale=int(_cfg_get(cfg, "monomials.gauge_timescale", 0)),
            fermion_timescale=int(_cfg_get(cfg, "monomials.fermion_timescale", 1)),
            pseudofermion_refresh=str(_cfg_get(cfg, "monomials.pf_refresh", "heatbath")),
            pseudofermion_gamma=float(_cfg_get(cfg, "monomials.pf_gamma", 0.3)),
            pseudofermion_force_mode=str(_cfg_get(cfg, "monomials.pf_force_mode", "analytic")),
            smd_gamma=float(_cfg_get(cfg, "update.smd_gamma", 0.3)),
            auto_refresh_pseudofermions=True,
        )
    elif (theory_family, theory_name) == ("su3", "su3_ym"):
        theory = SU3YangMills(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
            exp_method=exp_method,
        )
    elif (theory_family, theory_name) == ("su3", "su3_wilson_nf2"):
        fermion_bc = _parse_fermion_bc(
            _cfg_get(cfg, "physics.fermion_bc", "periodic"),
            nd=len(lattice_shape),
        )
        theory = SU3WilsonNf2(
            lattice_shape=lattice_shape,
            beta=beta,
            batch_size=batch,
            layout=layout,
            seed=seed,
            exp_method=exp_method,
            mass=mass,
            wilson_r=wilson_r,
            cg_tol=float(_cfg_get(cfg, "solver.tol", 1e-8)),
            cg_maxiter=int(_cfg_get(cfg, "solver.maxiter", 500)),
            solver_kind=str(_cfg_get(cfg, "solver.kind", "cg")),
            solver_form=str(_cfg_get(cfg, "solver.form", "normal")),
            use_solver_guess=bool(_cfg_get(cfg, "solver.use_solver_guess", False)),
            solver_guess_mode=str(_cfg_get(cfg, "solver.guess_mode", _cfg_get(cfg, "solver.solver_guess_mode", "last"))),
            solver_guess_history=int(_cfg_get(cfg, "solver.guess_history", _cfg_get(cfg, "solver.solver_guess_history", 1))),
            preconditioner_kind=str(_cfg_get(cfg, "solver.preconditioner", "none")),
            gmres_restart=int(_cfg_get(cfg, "solver.gmres_restart", 32)),
            gmres_solve_method=str(_cfg_get(cfg, "solver.gmres_solve_method", "batched")),
            fermion_bc=fermion_bc,
            include_gauge_monomial=bool(_cfg_get(cfg, "monomials.include_gauge", True)),
            include_fermion_monomial=bool(_cfg_get(cfg, "monomials.include_fermion", True)),
            fermion_monomial_kind=str(_cfg_get(cfg, "monomials.fermion_kind", "eo_preconditioned")),
            gauge_timescale=int(_cfg_get(cfg, "monomials.gauge_timescale", 0)),
            fermion_timescale=int(_cfg_get(cfg, "monomials.fermion_timescale", 1)),
            pseudofermion_refresh=str(_cfg_get(cfg, "monomials.pf_refresh", "heatbath")),
            pseudofermion_gamma=float(_cfg_get(cfg, "monomials.pf_gamma", 0.3)),
            pseudofermion_force_mode=str(_cfg_get(cfg, "monomials.pf_force_mode", "analytic")),
            smd_gamma=float(_cfg_get(cfg, "update.smd_gamma", 0.3)),
            auto_refresh_pseudofermions=True,
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable")

    return theory_family, theory_name, theory, hot_start_scale, layout, exp_method


def _draw_random_site_gauge(theory_family: str, theory, scale: float, exp_method: str):
    if theory_family == "u1":
        return u1_random_site_gauge(theory, scale=scale)
    if theory_family == "su2":
        return su2_random_site_gauge_with_method(theory, scale=scale, method=exp_method)
    if theory_family == "su3":
        return su3_random_site_gauge_with_method(theory, scale=scale, method=exp_method)
    raise ValueError(f"Unsupported theory family: {theory_family!r}")


def _gauge_transform_links(theory_family: str, theory, q, omega):
    if theory_family == "u1":
        return u1_gauge_transform_links(theory, q, omega)
    if theory_family == "su2":
        return su2_gauge_transform_links(theory, q, omega)
    if theory_family == "su3":
        return su3_gauge_transform_links(theory, q, omega)
    raise ValueError(f"Unsupported theory family: {theory_family!r}")


def _select_measurements(cfg: Mapping[str, Any], selected: Optional[Tuple[str, ...]]):
    inline_specs = _cfg_get(cfg, "measurements.inline", [{"type": "plaquette", "name": "plaquette", "every": 1}])
    if not isinstance(inline_specs, list):
        raise ValueError("measurements.inline must be a list of tables")
    measurements = build_inline_measurements(inline_specs)
    if selected is None:
        return measurements
    wanted = {str(v) for v in selected}
    chosen = [m for m in measurements if str(m.name) in wanted]
    missing = sorted(wanted.difference(str(m.name) for m in chosen))
    if missing:
        raise ValueError(f"Requested measurement names not found in config: {', '.join(missing)}")
    return chosen


def _should_ignore_key(key: str, ignore_rx: Optional[re.Pattern[str]]) -> bool:
    return bool(ignore_rx is not None and ignore_rx.search(str(key)))


def _compare_value_maps(
    vals0: Mapping[str, Any],
    vals1: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
    ignore_rx: Optional[re.Pattern[str]],
) -> Dict[str, Any]:
    keys0 = {str(k) for k in vals0.keys() if not _should_ignore_key(str(k), ignore_rx)}
    keys1 = {str(k) for k in vals1.keys() if not _should_ignore_key(str(k), ignore_rx)}
    missing = sorted(keys0.difference(keys1))
    extra = sorted(keys1.difference(keys0))
    if missing or extra:
        raise AssertionError(
            "Measurement key mismatch after gauge transform: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    compared = 0
    max_abs_err = 0.0
    max_rel_err = 0.0
    worst_key = ""
    for key in sorted(keys0):
        z0 = complex(float(vals0[key]))
        z1 = complex(float(vals1[key]))
        abs_err = float(abs(z0 - z1))
        ref = max(abs(z0), abs(z1))
        rel_err = float(abs_err / max(ref, float(atol)))
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err == max_rel_err and abs_err == max_abs_err:
            worst_key = str(key)
        if not np.allclose(z0, z1, atol=atol, rtol=rtol, equal_nan=True):
            raise AssertionError(
                f"Gauge invariance failed for key '{key}': "
                f"v0={z0} v1={z1} abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
            )
        compared += 1
    return {
        "compared_keys": int(compared),
        "max_abs_err": float(max_abs_err),
        "max_rel_err": float(max_rel_err),
        "worst_key": str(worst_key),
    }


def _snapshot_theory_key(theory):
    if not hasattr(theory, "key"):
        return None
    return jax.device_get(getattr(theory, "key"))


def _restore_theory_key(theory, key_snapshot) -> None:
    if key_snapshot is None or not hasattr(theory, "key"):
        return
    theory.key = jnp.asarray(key_snapshot)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gauge-invariance check for inline measurements")
    ap.add_argument("--config", required=True, help="MCMC TOML card used to build the theory and measurement list")
    ap.add_argument("--measurement", action="append", default=[], help="optional measurement name(s) to test; repeat or pass comma-separated")
    ap.add_argument("--ntrials", type=int, default=1, help="number of random gauge fields / gauge transforms to test")
    ap.add_argument("--omega-scale", type=float, default=0.05, help="random site-gauge transform amplitude")
    ap.add_argument("--hot-start-scale", type=float, default=float("nan"), help="override run.hot_start_scale")
    ap.add_argument("--step", type=int, default=0, help="inline-measurement step index used for every-scheduling")
    ap.add_argument("--atol", type=float, default=5.0e-8)
    ap.add_argument("--rtol", type=float, default=2.0e-7)
    ap.add_argument("--ignore-key-regex", type=str, default=DEFAULT_IGNORE_RE)
    ap.add_argument("--json-out", type=str, default="", help="optional JSON summary path")
    ap.add_argument("--cpu-threads", type=int, default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0))
    args = ap.parse_args()

    cfg = _load_toml(str(args.config))
    selected = _parse_measurement_names(args.measurement)
    measurements = _select_measurements(cfg, selected)
    ignore_rx = re.compile(str(args.ignore_key_regex)) if str(args.ignore_key_regex).strip() else None
    ntrials = max(1, int(args.ntrials))
    base_seed = int(_cfg_get(cfg, "run.seed", 0))

    summary: Dict[str, Any] = {
        "config": str(Path(args.config).expanduser().resolve()),
        "selected_measurements": [str(m.name) for m in measurements],
        "ntrials": int(ntrials),
        "omega_scale": float(args.omega_scale),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "ignore_key_regex": str(args.ignore_key_regex),
        "trials": [],
    }

    global_max_abs = 0.0
    global_max_rel = 0.0
    compared_total = 0

    for trial in range(ntrials):
        seed = int(base_seed + trial)
        theory_family, theory_name, theory, hot_start_scale_cfg, layout, exp_method = _build_theory_from_cfg(cfg, seed=seed)
        hot_start_scale = float(hot_start_scale_cfg if not np.isfinite(args.hot_start_scale) else args.hot_start_scale)
        q0 = theory.hot_start(scale=hot_start_scale)
        omega = _draw_random_site_gauge(theory_family, theory, scale=float(args.omega_scale), exp_method=str(exp_method))
        q1 = _gauge_transform_links(theory_family, theory, q0, omega)
        measure_key = _snapshot_theory_key(theory)

        ctx0 = MeasurementContext()
        recs0 = run_inline_measurements(measurements, step=int(args.step), q=q0, theory=theory, context=ctx0)
        _restore_theory_key(theory, measure_key)
        ctx1 = MeasurementContext()
        recs1 = run_inline_measurements(measurements, step=int(args.step), q=q1, theory=theory, context=ctx1)

        if len(recs0) != len(recs1):
            raise AssertionError(f"Measurement-record count mismatch: {len(recs0)} vs {len(recs1)}")

        trial_rows: List[Dict[str, Any]] = []
        print(f"trial {trial+1}/{ntrials}: theory={theory_family}/{theory_name} layout={layout}", flush=True)
        for idx, (r0, r1) in enumerate(zip(recs0, recs1)):
            n0 = str(r0.get("name", ""))
            n1 = str(r1.get("name", ""))
            if n0 != n1:
                raise AssertionError(f"Measurement-record name mismatch at index {idx}: {n0!r} vs {n1!r}")
            cmp_row = _compare_value_maps(
                dict(r0.get("values", {})),
                dict(r1.get("values", {})),
                atol=float(args.atol),
                rtol=float(args.rtol),
                ignore_rx=ignore_rx,
            )
            compared_total += int(cmp_row["compared_keys"])
            global_max_abs = max(global_max_abs, float(cmp_row["max_abs_err"]))
            global_max_rel = max(global_max_rel, float(cmp_row["max_rel_err"]))
            row = {
                "name": str(n0),
                **cmp_row,
            }
            trial_rows.append(row)
            print(
                f"  {n0}: compared={row['compared_keys']} "
                f"max_abs_err={row['max_abs_err']:.3e} max_rel_err={row['max_rel_err']:.3e}",
                flush=True,
            )

        summary["trials"].append(
            {
                "trial": int(trial),
                "seed": int(seed),
                "theory_family": str(theory_family),
                "theory_name": str(theory_name),
                "layout": str(layout),
                "measurements": trial_rows,
            }
        )

    summary["pass"] = True
    summary["compared_keys_total"] = int(compared_total)
    summary["max_abs_err"] = float(global_max_abs)
    summary["max_rel_err"] = float(global_max_rel)

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote {out}", flush=True)

    print(
        f"PASS: compared {compared_total} measurement outputs "
        f"(max_abs_err={global_max_abs:.3e}, max_rel_err={global_max_rel:.3e})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)

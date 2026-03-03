#!/usr/bin/env python3
"""Fast HMC/SMD smoke matrix for Nf=2 Wilson models."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


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

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax


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

from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2
from jaxqft.models.su2_wilson_nf2 import SU2WilsonNf2
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.testing import SmokeConfig, run_nf2_mcmc_smoke_suite


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _mode_defaults(mode: str) -> Dict[str, float]:
    m = str(mode).lower()
    if m == "quick":
        return {
            "warmup_no_ar": 2,
            "warmup_ar": 8,
            "nmeas": 24,
            "nmd": 8,
            "tau": 1.0,
            "sigma_cut": 4.0,
            "repro_steps": 3,
        }
    if m == "long":
        return {
            "warmup_no_ar": 20,
            "warmup_ar": 100,
            "nmeas": 1000,
            "nmd": 8,
            "tau": 1.0,
            "sigma_cut": 3.0,
            "repro_steps": 5,
        }
    raise ValueError(f"Unknown mode: {mode}")


def _fmt_run(label: str, r: Dict) -> str:
    return (
        f"{label:14s} plaq={r['mean_plaquette']:.8f} +/- {r['err_iat']:.6e} "
        f"tau_int={r['tau_int']:.3f} ess={r['ess']:.1f} "
        f"acc={r['meas_acceptance']:.4f}"
    )


def _ref_threshold_defaults(mode: str) -> Dict[str, float]:
    m = str(mode).lower()
    if m == "quick":
        return {
            "max_sigma": 6.0,
            "max_abs_plaq": 8.0e-2,
            "max_acc_diff": 2.5e-1,
        }
    if m == "long":
        return {
            "max_sigma": 4.0,
            "max_abs_plaq": 3.0e-2,
            "max_acc_diff": 1.0e-1,
        }
    raise ValueError(f"Unknown mode: {mode}")


def _f64(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_reference(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _compare_reference(current_suites, reference_doc: Dict[str, Any], *, max_sigma: float, max_abs_plaq: float, max_acc_diff: float):
    ref_suites = reference_doc.get("suites", [])
    ref_by_model = {str(s.get("model")): s for s in ref_suites if isinstance(s, dict)}
    run_keys = ("hmc_ou", "smd_ou", "hmc_heatbath")

    model_results = {}
    overall_ok = True
    for cur in current_suites:
        model = str(cur.get("model"))
        ref = ref_by_model.get(model)
        if ref is None:
            model_results[model] = {
                "ok": False,
                "message": "missing model in reference",
                "runs": {},
            }
            overall_ok = False
            continue

        runs_cur = cur.get("runs", {})
        runs_ref = ref.get("runs", {})
        runs_out = {}
        model_ok = True
        for rk in run_keys:
            c = runs_cur.get(rk)
            r = runs_ref.get(rk)
            if not isinstance(c, dict) or not isinstance(r, dict):
                runs_out[rk] = {"ok": False, "message": "missing run in reference/current"}
                model_ok = False
                continue

            mc = _f64(c.get("mean_plaquette"))
            mr = _f64(r.get("mean_plaquette"))
            ec = abs(_f64(c.get("err_iat")))
            er = abs(_f64(r.get("err_iat")))
            acc_c = _f64(c.get("meas_acceptance"))
            acc_r = _f64(r.get("meas_acceptance"))

            delta = abs(mc - mr)
            ed = math.sqrt(max(0.0, ec * ec + er * er))
            sigma = delta / ed if (math.isfinite(ed) and ed > 0.0) else float("inf")
            acc_diff = abs(acc_c - acc_r)

            sigma_ok = math.isfinite(sigma) and sigma <= float(max_sigma)
            abs_ok = math.isfinite(delta) and delta <= float(max_abs_plaq)
            acc_ok = math.isfinite(acc_diff) and acc_diff <= float(max_acc_diff)
            run_ok = bool(sigma_ok and abs_ok and acc_ok)
            runs_out[rk] = {
                "ok": run_ok,
                "delta_plaquette": float(delta),
                "sigma": float(sigma),
                "acc_diff": float(acc_diff),
            }
            model_ok = model_ok and run_ok

        model_results[model] = {"ok": bool(model_ok), "runs": runs_out}
        overall_ok = overall_ok and model_ok

    return {
        "ok": bool(overall_ok),
        "thresholds": {
            "max_sigma": float(max_sigma),
            "max_abs_plaq": float(max_abs_plaq),
            "max_acc_diff": float(max_acc_diff),
        },
        "models": model_results,
    }


def main():
    ap = argparse.ArgumentParser(description="Nf=2 MCMC smoke matrix")
    ap.add_argument("--mode", type=str, default="quick", choices=["quick", "long"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shape-nd4", type=str, default="4,4,4,8")
    ap.add_argument("--shape-u1", type=str, default="8,16")
    ap.add_argument("--beta-su3", type=float, default=5.0)
    ap.add_argument("--beta-su2", type=float, default=5.0)
    ap.add_argument("--beta-u1", type=float, default=1.5)
    ap.add_argument("--mass-nd4", type=float, default=0.05)
    ap.add_argument("--mass-u1", type=float, default=0.1)
    ap.add_argument("--hot-scale", type=float, default=0.2)
    ap.add_argument("--solver-tol", type=float, default=1e-7)
    ap.add_argument("--solver-maxiter", type=int, default=1000)
    ap.add_argument("--smd-gamma", type=float, default=0.3)
    ap.add_argument("--iat-method", type=str, default="gamma", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--warmup-no-ar", type=int, default=None)
    ap.add_argument("--warmup-ar", type=int, default=None)
    ap.add_argument("--meas", type=int, default=None)
    ap.add_argument("--nmd", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--sigma-cut", type=float, default=None)
    ap.add_argument("--repro-steps", type=int, default=None)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--reference-json", type=str, default="")
    ap.add_argument("--ref-max-sigma", type=float, default=None)
    ap.add_argument("--ref-max-abs-plaq", type=float, default=None)
    ap.add_argument("--ref-max-acc-diff", type=float, default=None)
    ap.add_argument("--selfcheck-fail", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0))
    ap.add_argument("--cpu-onednn", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    d = _mode_defaults(args.mode)
    warmup_no_ar = int(d["warmup_no_ar"] if args.warmup_no_ar is None else args.warmup_no_ar)
    warmup_ar = int(d["warmup_ar"] if args.warmup_ar is None else args.warmup_ar)
    nmeas = int(d["nmeas"] if args.meas is None else args.meas)
    nmd = int(d["nmd"] if args.nmd is None else args.nmd)
    tau = float(d["tau"] if args.tau is None else args.tau)
    sigma_cut = float(d["sigma_cut"] if args.sigma_cut is None else args.sigma_cut)
    repro_steps = int(d["repro_steps"] if args.repro_steps is None else args.repro_steps)
    rt = _ref_threshold_defaults(args.mode)
    ref_max_sigma = float(rt["max_sigma"] if args.ref_max_sigma is None else args.ref_max_sigma)
    ref_max_abs_plaq = float(rt["max_abs_plaq"] if args.ref_max_abs_plaq is None else args.ref_max_abs_plaq)
    ref_max_acc_diff = float(rt["max_acc_diff"] if args.ref_max_acc_diff is None else args.ref_max_acc_diff)

    shape_nd4 = _parse_shape(args.shape_nd4)
    shape_u1 = _parse_shape(args.shape_u1)

    cfg = SmokeConfig(
        nmd=nmd,
        tau=tau,
        warmup_no_ar=warmup_no_ar,
        warmup_ar=warmup_ar,
        nmeas=nmeas,
        hot_scale=float(args.hot_scale),
        iat_method=str(args.iat_method),
        sigma_cut=sigma_cut,
        seed=int(args.seed),
        smd_gamma=float(args.smd_gamma),
        reproducibility_steps=repro_steps,
    )

    print("JAX backend:", jax.default_backend())
    print(
        "Smoke config:"
        f" mode={args.mode} nmd={cfg.nmd} tau={cfg.tau}"
        f" warmup(noAR/AR)={cfg.warmup_no_ar}/{cfg.warmup_ar}"
        f" meas={cfg.nmeas} sigma_cut={cfg.sigma_cut}"
    )

    def su3_factory(seed: int, pf_refresh: str):
        return SU3WilsonNf2(
            lattice_shape=shape_nd4,
            beta=float(args.beta_su3),
            mass=float(args.mass_nd4),
            exp_method="su3",
            batch_size=1,
            solver_kind="cg",
            solver_form="normal",
            cg_tol=float(args.solver_tol),
            cg_maxiter=int(args.solver_maxiter),
            fermion_monomial_kind="eo_preconditioned",
            pseudofermion_refresh=str(pf_refresh),
            pseudofermion_force_mode="analytic",
            smd_gamma=float(args.smd_gamma),
        )

    def su2_factory(seed: int, pf_refresh: str):
        return SU2WilsonNf2(
            lattice_shape=shape_nd4,
            beta=float(args.beta_su2),
            mass=float(args.mass_nd4),
            exp_method="su2",
            batch_size=1,
            solver_kind="cg",
            solver_form="normal",
            cg_tol=float(args.solver_tol),
            cg_maxiter=int(args.solver_maxiter),
            fermion_monomial_kind="eo_preconditioned",
            pseudofermion_refresh=str(pf_refresh),
            pseudofermion_force_mode="analytic",
            smd_gamma=float(args.smd_gamma),
        )

    def u1_factory(seed: int, pf_refresh: str):
        return U1WilsonNf2(
            lattice_shape=shape_u1,
            beta=float(args.beta_u1),
            mass=float(args.mass_u1),
            batch_size=1,
            solver_kind="cg",
            solver_form="normal",
            cg_tol=float(args.solver_tol),
            cg_maxiter=int(args.solver_maxiter),
            fermion_monomial_kind="eo_preconditioned",
            pseudofermion_refresh=str(pf_refresh),
            pseudofermion_force_mode="analytic",
            smd_gamma=float(args.smd_gamma),
        )

    suites = [
        run_nf2_mcmc_smoke_suite("SU3_Nd4", su3_factory, cfg),
        run_nf2_mcmc_smoke_suite("SU2_Nd4", su2_factory, SmokeConfig(**{**cfg.__dict__, "seed": cfg.seed + 100})),
        run_nf2_mcmc_smoke_suite("U1_Nd2", u1_factory, SmokeConfig(**{**cfg.__dict__, "seed": cfg.seed + 200})),
    ]

    overall = True
    for s in suites:
        print("=" * 88)
        print(f"Model: {s['model']}")
        print(_fmt_run("HMC+OU", s["runs"]["hmc_ou"]))
        print(_fmt_run("SMD+OU", s["runs"]["smd_ou"]))
        print(_fmt_run("HMC+HB", s["runs"]["hmc_heatbath"]))
        c1 = s["comparisons"]["hmc_ou_vs_smd_ou"]
        c2 = s["comparisons"]["hmc_ou_vs_hmc_heatbath"]
        print(f"cmp HMC+OU vs SMD+OU:   delta={c1['delta']:.6e} sigma={c1['sigma']:.3f} agree={c1['agree']}")
        print(f"cmp HMC+OU vs HMC+HB:   delta={c2['delta']:.6e} sigma={c2['sigma']:.3f} agree={c2['agree']}")
        r1 = s["reproducibility"]["hmc_ou"]
        r2 = s["reproducibility"]["smd_ou"]
        print(f"repro HMC+OU: max_abs={r1['max_abs_diff']:.3e} tol={r1['tol']:.3e} ok={r1['ok']}")
        print(f"repro SMD+OU: max_abs={r2['max_abs_diff']:.3e} tol={r2['tol']:.3e} ok={r2['ok']}")
        print(f"PASS: {bool(s['pass'])}")
        overall = overall and bool(s["pass"])

    print("=" * 88)
    print(f"OVERALL PASS: {overall}")

    ref_check = None
    if args.reference_json:
        ref_doc = _load_reference(args.reference_json)
        ref_check = _compare_reference(
            suites,
            ref_doc,
            max_sigma=ref_max_sigma,
            max_abs_plaq=ref_max_abs_plaq,
            max_acc_diff=ref_max_acc_diff,
        )
        print("=" * 88)
        print("Reference check:")
        t = ref_check["thresholds"]
        print(
            "  thresholds:"
            f" sigma<={t['max_sigma']:.3f}"
            f" |delta_plaq|<={t['max_abs_plaq']:.6e}"
            f" |delta_acc|<={t['max_acc_diff']:.6e}"
        )
        for model, mres in ref_check["models"].items():
            print(f"  {model}: ok={mres['ok']}")
            runs = mres.get("runs", {})
            for rk in ("hmc_ou", "smd_ou", "hmc_heatbath"):
                rr = runs.get(rk, {})
                if "message" in rr:
                    print(f"    {rk}: ok=False ({rr['message']})")
                    continue
                print(
                    f"    {rk}: ok={rr.get('ok', False)}"
                    f" delta={rr.get('delta_plaquette', float('nan')):.6e}"
                    f" sigma={rr.get('sigma', float('nan')):.3f}"
                    f" acc_diff={rr.get('acc_diff', float('nan')):.6e}"
                )
        print(f"REFERENCE PASS: {bool(ref_check['ok'])}")
        overall = overall and bool(ref_check["ok"])
        print("=" * 88)
        print(f"OVERALL PASS (with reference): {overall}")

    out = {
        "backend": jax.default_backend(),
        "config": cfg.__dict__,
        "suites": suites,
        "overall_pass": bool(overall),
        "reference_check": ref_check,
    }
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {p}")

    if bool(args.selfcheck_fail) and (not overall):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

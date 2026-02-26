"""Systematic Wilson benchmark matrix (U1/SU2/SU3, eager/JIT, solver variants).

This script benchmarks:
- D application
- D^dagger D application
- solve_normal

It then prints measured JIT speedups and solver rankings, and derives
optimization priorities from the measured bottlenecks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import jax
import numpy as np

# Allow running as a script without requiring `pip install -e .`.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jaxqft.models.su2_wilson_nf2 import SU2WilsonNf2
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _parse_shape_list(s: str) -> List[Tuple[int, ...]]:
    chunks = [c.strip() for c in s.split(";") if c.strip()]
    if not chunks:
        raise ValueError("shape list must contain at least one shape")
    return [_parse_shape(c) for c in chunks]


def _parse_solver_variants(s: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        if ":" not in tok:
            raise ValueError(f"invalid solver variant '{tok}', expected kind:form")
        k, f = tok.split(":", 1)
        out.append((k.strip().lower(), f.strip().lower()))
    if not out:
        raise ValueError("solver variants must not be empty")
    return out


def _parse_jit_modes(s: str) -> List[bool]:
    out: List[bool] = []
    for tok in [t.strip().lower() for t in s.split(",") if t.strip()]:
        if tok in ("jit", "true", "1", "on"):
            out.append(True)
        elif tok in ("eager", "false", "0", "off"):
            out.append(False)
        else:
            raise ValueError(f"unknown jit mode '{tok}', use eager or jit")
    if not out:
        raise ValueError("jit mode list must not be empty")
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def _bench_call(fn, args: Iterable[Any], n: int) -> float:
    y = fn(*args)
    y.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(max(1, int(n))):
        y = fn(*args)
        y.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, int(n))


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in kwargs.items() if k in fields}


def _make_model(
    model: str,
    shape: Tuple[int, ...],
    jit_enabled: bool,
    solver_kind: str,
    solver_form: str,
    args: argparse.Namespace,
):
    common = {
        "lattice_shape": shape,
        "batch_size": int(args.batch),
        "layout": str(args.layout),
        "mass": float(args.mass),
        "wilson_r": float(args.wilson_r),
        "solver_kind": str(solver_kind),
        "solver_form": str(solver_form),
        "preconditioner_kind": str(args.preconditioner_kind),
        "cg_tol": float(args.cg_tol),
        "cg_maxiter": int(args.cg_maxiter),
        "gmres_restart": int(args.gmres_restart),
        "gmres_solve_method": str(args.gmres_solve_method),
        "dirac_kernel": str(args.dirac_kernel),
        "fermion_monomial_kind": str(args.fermion_monomial_kind),
        "pseudofermion_force_mode": str(args.pf_force_mode),
        "jit_dirac_kernels": bool(jit_enabled),
        "jit_solvers": bool(jit_enabled),
    }

    if model == "u1":
        cls = U1WilsonNf2
        specific = {"beta": float(args.beta_u1)}
    elif model == "su2":
        cls = SU2WilsonNf2
        specific = {"beta": float(args.beta_su2), "exp_method": str(args.exp_method_su2)}
    elif model == "su3":
        cls = SU3WilsonNf2
        specific = {"beta": float(args.beta_su3), "exp_method": str(args.exp_method_su3)}
    else:
        raise ValueError(f"unknown model '{model}'")

    all_kwargs = {}
    all_kwargs.update(common)
    all_kwargs.update(specific)
    return cls(**_filter_kwargs(cls, all_kwargs))


def _bench_model(th, n_kernel: int, n_solve: int, do_solve: bool = True) -> Dict[str, float]:
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    row: Dict[str, float] = {}
    d_s = _bench_call(th.apply_D, (U, psi), n_kernel)
    n_s = _bench_call(th.apply_normal, (U, psi), n_kernel)
    row["D_s"] = d_s
    row["N_s"] = n_s

    nc = int(th.fermion_shape()[-1])
    vol = int(np.prod(th.lattice_shape))
    bsz = int(th.batch_size)
    flops_site_d = int(th.dirac.flops_per_site_matvec(nc=nc, use_sparse_gamma=True, include_diagonal=True))
    flops_call_d = int(flops_site_d * vol * bsz)
    flops_call_n = int(2 * flops_call_d)
    row["D_flops_per_site"] = float(flops_site_d)
    row["D_gflops"] = float(flops_call_d) / (d_s * 1e9 + 1e-30)
    row["N_gflops"] = float(flops_call_n) / (n_s * 1e9 + 1e-30)

    if do_solve and hasattr(th, "solve_normal"):
        phi = th.sample_pseudofermion(U)
        row["solve_s"] = _bench_call(th.solve_normal, (U, phi), n_solve)
        row["solve_over_N"] = row["solve_s"] / (row["N_s"] + 1e-16)
    return row


def _iter_matrix_jobs(args: argparse.Namespace):
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("models must not be empty")
    jit_modes = _parse_jit_modes(args.jit_modes)
    variants = _parse_solver_variants(args.solver_variants)

    shape_map: Dict[str, Sequence[Tuple[int, ...]]] = {
        "u1": _parse_shape_list(args.u1_shapes),
        "su2": _parse_shape_list(args.su2_shapes),
        "su3": _parse_shape_list(args.su3_shapes),
    }

    for model in models:
        if model not in shape_map:
            raise ValueError(f"unknown model '{model}' in --models")
        for shape in shape_map[model]:
            for solver_kind, solver_form in variants:
                for jit_enabled in jit_modes:
                    yield model, shape, solver_kind, solver_form, jit_enabled


def _print_rows(rows: Iterable[Dict[str, Any]]) -> None:
    print("=== Wilson Matrix ===")
    for row in rows:
        printable = {k: _fmt(v) for k, v in row.items()}
        print(printable)


def _rank_priorities(rows: Iterable[Dict[str, Any]]) -> None:
    all_rows = list(rows)
    ok = [r for r in all_rows if bool(r.get("ok", False))]
    print("=== Measured Ranking ===")
    if not ok:
        print("No successful benchmark rows.")
        return

    def _key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            r.get("model"),
            tuple(r.get("shape", ())),
            r.get("solver_kind"),
            r.get("solver_form"),
        )

    eager: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    jit: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in ok:
        if bool(r.get("jit", False)):
            jit[_key(r)] = r
        else:
            eager[_key(r)] = r

    jit_d_speedups: List[float] = []
    jit_n_speedups: List[float] = []
    jit_s_speedups: List[float] = []
    for k, re in eager.items():
        rj = jit.get(k)
        if rj is None:
            continue
        sd = float(re["D_s"]) / (float(rj["D_s"]) + 1e-16)
        sn = float(re["N_s"]) / (float(rj["N_s"]) + 1e-16)
        ss = float(re.get("solve_s", float("nan"))) / (float(rj.get("solve_s", float("nan"))) + 1e-16)
        jit_d_speedups.append(sd)
        jit_n_speedups.append(sn)
        if np.isfinite(ss):
            jit_s_speedups.append(ss)
        print(
            f"JIT speedup {k[0]} {k[1]} {k[2]}:{k[3]} "
            f"D/N/solve = {sd:.2f}x / {sn:.2f}x / {ss:.2f}x"
        )

    jit_rows = [r for r in ok if bool(r.get("jit", False))]
    print("Solver ranking (JIT rows, lower solve_s is better):")
    for model in sorted({r["model"] for r in jit_rows}):
        by_model = [r for r in jit_rows if r["model"] == model]
        for shape in sorted({tuple(r["shape"]) for r in by_model}):
            group = [r for r in by_model if tuple(r["shape"]) == tuple(shape)]
            ranked = sorted(group, key=lambda r: float(r.get("solve_s", float("inf"))))
            print(f"  {model} {shape}:")
            for r in ranked:
                print(
                    f"    {r['solver_kind']}:{r['solver_form']} "
                    f"solve={float(r.get('solve_s', float('nan'))):.6e}s "
                    f"solve/N={float(r.get('solve_over_N', float('nan'))):.2f}"
                )

    # Priority extraction from measured data.
    med_jit_d = float(np.median(jit_d_speedups)) if jit_d_speedups else float("nan")
    med_jit_n = float(np.median(jit_n_speedups)) if jit_n_speedups else float("nan")
    med_jit_s = float(np.median(jit_s_speedups)) if jit_s_speedups else float("nan")
    med_solve_over_n = float(np.median([float(r.get("solve_over_N", np.nan)) for r in jit_rows if np.isfinite(r.get("solve_over_N", np.nan))]))

    slowest_n = max(jit_rows, key=lambda r: float(r.get("N_s", -1.0)))
    slowest_s = max(jit_rows, key=lambda r: float(r.get("solve_s", -1.0)))

    print("Priority order (from current measurements):")
    print(
        "  1) Solver path and preconditioning first: "
        f"median solve/N ~= {med_solve_over_n:.2f} operator-applications."
    )
    print(
        "  2) Keep and enforce JIT for Wilson code paths: "
        f"median JIT speedup D/N/solve ~= {med_jit_d:.2f}x / {med_jit_n:.2f}x / {med_jit_s:.2f}x."
    )
    print(
        "  3) Optimize highest-cost matvec target next: "
        f"{slowest_n['model']} {tuple(slowest_n['shape'])} "
        f"{slowest_n['solver_kind']}:{slowest_n['solver_form']} N_s={float(slowest_n['N_s']):.6e}s."
    )
    print(
        "  4) Optimize highest-cost solve target next: "
        f"{slowest_s['model']} {tuple(slowest_s['shape'])} "
        f"{slowest_s['solver_kind']}:{slowest_s['solver_form']} solve_s={float(slowest_s['solve_s']):.6e}s."
    )


def main() -> None:
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(description="Systematic benchmark matrix for Wilson fermion models.")
    ap.add_argument("--models", type=str, default="u1,su2,su3")
    ap.add_argument("--u1-shapes", type=str, default="8,8;256,256")
    ap.add_argument("--su2-shapes", type=str, default="4,4,4,8")
    ap.add_argument("--su3-shapes", type=str, default="4,4,4,8")
    ap.add_argument("--jit-modes", type=str, default="eager,jit")
    ap.add_argument(
        "--solver-variants",
        type=str,
        default="cg:normal,bicgstab:normal,bicgstab:split,bicgstab:eo_split",
        help="comma-separated kind:form entries",
    )

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layout", type=str, default="BM...IJ")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--wilson-r", type=float, default=1.0)
    ap.add_argument("--beta-u1", type=float, default=1.0)
    ap.add_argument("--beta-su2", type=float, default=2.5)
    ap.add_argument("--beta-su3", type=float, default=5.8)
    ap.add_argument("--exp-method-su2", type=str, default="su2")
    ap.add_argument("--exp-method-su3", type=str, default="su3")
    ap.add_argument("--dirac-kernel", type=str, default="optimized", choices=["optimized", "reference"])
    ap.add_argument("--fermion-monomial-kind", type=str, default="unpreconditioned", choices=["unpreconditioned", "eo_preconditioned"])
    ap.add_argument("--pf-force-mode", type=str, default="autodiff", choices=["autodiff", "analytic"])
    ap.add_argument("--preconditioner-kind", type=str, default="none", choices=["none", "jacobi"])

    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--cg-maxiter", type=int, default=500)
    ap.add_argument("--gmres-restart", type=int, default=32)
    ap.add_argument("--gmres-solve-method", type=str, default="batched", choices=["batched", "incremental"])

    ap.add_argument("--kernel-iters", type=int, default=20)
    ap.add_argument("--solve-iters", type=int, default=3)
    ap.add_argument("--json-out", type=str, default="", help="optional path to save raw matrix json")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for model, shape, solver_kind, solver_form, jit_enabled in _iter_matrix_jobs(args):
        row: Dict[str, Any] = {
            "model": model,
            "shape": tuple(int(v) for v in shape),
            "solver_kind": solver_kind,
            "solver_form": solver_form,
            "jit": bool(jit_enabled),
            "ok": False,
        }
        try:
            th = _make_model(
                model=model,
                shape=shape,
                jit_enabled=jit_enabled,
                solver_kind=solver_kind,
                solver_form=solver_form,
                args=args,
            )
            row.update(_bench_model(th, n_kernel=int(args.kernel_iters), n_solve=int(args.solve_iters), do_solve=True))
            row["ok"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    print("JAX backend:", jax.default_backend())
    _print_rows(rows)
    _rank_priorities(rows)

    if args.json_out.strip():
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved json matrix to: {args.json_out}")


if __name__ == "__main__":
    main()

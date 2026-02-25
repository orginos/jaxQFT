"""Cross-model Wilson benchmark matrix (U1/SU2/SU3, eager vs JIT).

This script benchmarks D, DdagD, and (when available) solve_normal.
It is intended to quantify dispatch overhead and guide optimization ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import jax

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


def _bench_model(name: str, th, n_kernel: int, n_solve: int, do_solve: bool = True) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": name,
        "shape": tuple(int(v) for v in th.lattice_shape),
        "jit_dirac_kernels": bool(getattr(th, "jit_dirac_kernels", False)),
        "jit_solvers": bool(getattr(th, "jit_solvers", False)),
    }
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    row["D_s"] = _bench_call(th.apply_D, (U, psi), n_kernel)
    row["N_s"] = _bench_call(th.apply_normal, (U, psi), n_kernel)

    if do_solve and hasattr(th, "solve_normal"):
        phi = th.sample_pseudofermion(U)
        row["solve_s"] = _bench_call(th.solve_normal, (U, phi), n_solve)
    return row


def _bench_su2_manual_jit(shape: Tuple[int, ...], n_kernel: int) -> Dict[str, Any]:
    th = SU2WilsonNf2(
        lattice_shape=shape,
        beta=2.5,
        mass=0.05,
        wilson_r=1.0,
        batch_size=1,
        layout="BM...IJ",
        exp_method="su2",
    )
    U = th.hot_start(scale=0.05)
    psi = th.random_fermion()
    d_eager = _bench_call(th.apply_D, (U, psi), n_kernel)
    n_eager = _bench_call(th.apply_normal, (U, psi), n_kernel)
    d_jit_fn = jax.jit(lambda Uin, psin: th.apply_D(Uin, psin))
    n_jit_fn = jax.jit(lambda Uin, psin: th.apply_normal(Uin, psin))
    d_jit = _bench_call(d_jit_fn, (U, psi), n_kernel)
    n_jit = _bench_call(n_jit_fn, (U, psi), n_kernel)
    return {
        "model": "su2",
        "shape": tuple(int(v) for v in shape),
        "eager_D_s": d_eager,
        "jit_D_s": d_jit,
        "eager_N_s": n_eager,
        "jit_N_s": n_jit,
        "D_speedup": d_eager / (d_jit + 1e-16),
        "N_speedup": n_eager / (n_jit + 1e-16),
    }


def _print_rows(rows: Iterable[Dict[str, Any]]) -> None:
    print("=== Wilson Matrix ===")
    for row in rows:
        printable = {k: _fmt(v) for k, v in row.items()}
        print(printable)


def _rank_priorities(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    print("=== Priority Ranking ===")

    def _lookup(model: str, shape: Tuple[int, ...], jit: bool) -> Dict[str, Any] | None:
        for r in rows:
            if r.get("model") == model and tuple(r.get("shape", ())) == tuple(shape):
                if bool(r.get("jit_dirac_kernels", False)) == bool(jit):
                    return r
        return None

    # U1 and SU3 have explicit eager/jit variants.
    for model, shape in (("u1", (8, 8)), ("u1", (256, 256)), ("su3", (4, 4, 4, 8))):
        eager = _lookup(model, shape, False)
        fast = _lookup(model, shape, True)
        if eager is None or fast is None:
            continue
        d = eager["D_s"] / (fast["D_s"] + 1e-16)
        n = eager["N_s"] / (fast["N_s"] + 1e-16)
        s = eager.get("solve_s", float("nan")) / (fast.get("solve_s", float("nan")) + 1e-16)
        print(f"{model} {shape}: JIT speedup D/N/solve = {d:.2f}x / {n:.2f}x / {s:.2f}x")

    # SU2 manual eager-vs-jit result.
    for r in rows:
        if r.get("model") != "su2":
            continue
        print(
            f"su2 {tuple(r['shape'])}: JIT speedup D/N = "
            f"{float(r['D_speedup']):.2f}x / {float(r['N_speedup']):.2f}x"
        )

    print("Suggested order:")
    print("  1) Keep default JIT on for Wilson kernels/solvers (largest global gain).")
    print("  2) Port SU2 Wilson to SU3/U1 full monomial+EO harness with same JIT controls.")
    print("  3) Re-optimize reference-vs-optimized kernels only in JIT regime.")
    print("  4) Add solver-level instrumentation (true iteration count) and preconditioning.")


def main() -> None:
    if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    ap = argparse.ArgumentParser(description="Benchmark matrix for Wilson fermion models.")
    ap.add_argument("--u1-shape-small", type=str, default="8,8")
    ap.add_argument("--u1-shape-large", type=str, default="256,256")
    ap.add_argument("--su2-shape", type=str, default="4,4,4,8")
    ap.add_argument("--su3-shape", type=str, default="4,4,4,8")
    ap.add_argument("--kernel-iters", type=int, default=20)
    ap.add_argument("--solve-iters", type=int, default=3)
    ap.add_argument("--json-out", type=str, default="", help="optional path to save raw matrix json")
    args = ap.parse_args()

    u1_small = _parse_shape(args.u1_shape_small)
    u1_large = _parse_shape(args.u1_shape_large)
    su2_shape = _parse_shape(args.su2_shape)
    su3_shape = _parse_shape(args.su3_shape)

    rows = []

    rows.append(
        _bench_model(
            "u1",
            U1WilsonNf2(
                lattice_shape=u1_small,
                beta=1.0,
                batch_size=1,
                layout="BM...IJ",
                jit_dirac_kernels=False,
                jit_solvers=False,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )
    rows.append(
        _bench_model(
            "u1",
            U1WilsonNf2(
                lattice_shape=u1_small,
                beta=1.0,
                batch_size=1,
                layout="BM...IJ",
                jit_dirac_kernels=True,
                jit_solvers=True,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )
    rows.append(
        _bench_model(
            "u1",
            U1WilsonNf2(
                lattice_shape=u1_large,
                beta=1.0,
                batch_size=1,
                layout="BM...IJ",
                jit_dirac_kernels=False,
                jit_solvers=False,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )
    rows.append(
        _bench_model(
            "u1",
            U1WilsonNf2(
                lattice_shape=u1_large,
                beta=1.0,
                batch_size=1,
                layout="BM...IJ",
                jit_dirac_kernels=True,
                jit_solvers=True,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )

    rows.append(
        _bench_model(
            "su3",
            SU3WilsonNf2(
                lattice_shape=su3_shape,
                beta=5.8,
                batch_size=1,
                layout="BM...IJ",
                exp_method="su3",
                jit_dirac_kernels=False,
                jit_solvers=False,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )
    rows.append(
        _bench_model(
            "su3",
            SU3WilsonNf2(
                lattice_shape=su3_shape,
                beta=5.8,
                batch_size=1,
                layout="BM...IJ",
                exp_method="su3",
                jit_dirac_kernels=True,
                jit_solvers=True,
            ),
            n_kernel=args.kernel_iters,
            n_solve=args.solve_iters,
            do_solve=True,
        )
    )

    rows.append(_bench_su2_manual_jit(su2_shape, n_kernel=args.kernel_iters))

    print("JAX backend:", jax.default_backend())
    _print_rows(rows)
    _rank_priorities(rows)

    if args.json_out.strip():
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved json matrix to: {args.json_out}")


if __name__ == "__main__":
    main()

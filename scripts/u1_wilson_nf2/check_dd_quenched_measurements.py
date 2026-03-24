#!/usr/bin/env python3
"""Check exact DD Schur observables for quenched U(1) Wilson fermions on tiny lattices."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

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

from jaxqft.core.measurements import (
    MeasurementContext,
    _dd_exact_schur_inverse,
    _full_dense_inverse,
    build_inline_measurements,
    run_inline_measurements,
)
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2


def _parse_shape(s: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must be non-empty")
    return tuple(vals)


def _parse_ints(s: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return tuple(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check quenched DD Schur measurements against full dense inverse")
    ap.add_argument("--shape", type=str, default="8,8")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--mass", type=float, default=0.0)
    ap.add_argument("--boundary-slices", type=str, default="2,6")
    ap.add_argument("--boundary-width", type=int, default=1)
    ap.add_argument("--dense-max-dof", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hot-scale", type=float, default=0.05)
    ap.add_argument("--matrix-tol", type=float, default=5e-6)
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    cuts = _parse_ints(args.boundary_slices)
    theory = U1WilsonNf2(
        lattice_shape=shape,
        beta=float(args.beta),
        batch_size=1,
        seed=int(args.seed),
        mass=float(args.mass),
        include_gauge_monomial=False,
        include_fermion_monomial=False,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )
    q = theory.hot_start(scale=float(args.hot_scale))
    mctx = MeasurementContext()

    ginv, _ = _full_dense_inverse(q=q, theory=theory, context=mctx, dense_max_dof=int(args.dense_max_dof))
    sinv, decomp, info = _dd_exact_schur_inverse(
        q=q,
        theory=theory,
        context=mctx,
        boundary_slices=cuts,
        boundary_width=int(args.boundary_width),
        dense_max_dof=int(args.dense_max_dof),
    )

    nsc = int(theory.fermion_shape()[-2] * theory.fermion_shape()[-1])
    i_idx = decomp.interior_component_indices(nsc)
    ginv_interior = np.stack(
        [np.asarray(ginv[b][np.ix_(i_idx, i_idx)]) for b in range(int(ginv.shape[0]))],
        axis=0,
    )
    diff = np.asarray(ginv_interior - sinv)
    abs_err = float(np.max(np.abs(diff)))
    ref = float(np.max(np.abs(ginv_interior)))
    rel_err = abs_err / ref if ref > 0.0 else abs_err

    specs = [
        {
            "type": "pion_2pt_dd",
            "name": "pion_2pt_dd",
            "every": 1,
            "source_average": True,
            "boundary_slices": list(cuts),
            "boundary_width": int(args.boundary_width),
        },
        {
            "type": "eta_2pt_dd",
            "name": "eta_2pt_dd",
            "every": 1,
            "source_average": True,
            "boundary_slices": list(cuts),
            "boundary_width": int(args.boundary_width),
        },
    ]
    recs = run_inline_measurements(build_inline_measurements(specs), step=0, q=q, theory=theory, context=MeasurementContext())
    if len(recs) != 2:
        raise RuntimeError(f"Expected 2 DD measurement records, got {len(recs)}")

    print(f"shape={shape} cuts={cuts} bw={int(args.boundary_width)}")
    print(
        "interior Schur vs full inverse interior block:"
        f" abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
        f" dd_total_sec={float(info.get('dd_total_sec', 0.0)):.3f}"
    )
    print(f"measurement records={[str(r['name']) for r in recs]}")

    if not np.isfinite(abs_err) or abs_err > float(args.matrix_tol):
        raise SystemExit(
            f"DD Schur identity failed: abs_err={abs_err:.6e} exceeds matrix_tol={float(args.matrix_tol):.6e}"
        )


if __name__ == "__main__":
    main()

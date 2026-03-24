#!/usr/bin/env python3
"""Tiny-lattice reference DD HMC for U(1) Wilson Nf=2 with exact dense determinant factorization."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


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

from jaxqft.core.integrators import minnorm2
from jaxqft.core.update import hmc
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_wilson_nf2_dd import U1WilsonNf2DDReference


def _parse_shape(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must contain at least one dimension")
    return tuple(vals)


def _parse_ints(s: str):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return tuple(vals)


def _max_masked_abs(x, mask) -> float:
    return float(jnp.max(jnp.abs(x * mask.astype(x.dtype))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference DD HMC for U(1) Wilson Nf=2 (tiny lattices only)")
    ap.add_argument("--shape", type=str, default="4,4")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--boundary-slices", type=str, default="1,3")
    ap.add_argument("--boundary-width", type=int, default=1)
    ap.add_argument("--dense-max-dof", type=int, default=256)
    ap.add_argument("--start", type=str, default="hot", choices=["hot", "cold"])
    ap.add_argument("--hot-scale", type=float, default=0.05)
    ap.add_argument("--ntraj", type=int, default=2)
    ap.add_argument("--nmd", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    cuts = _parse_ints(args.boundary_slices)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    base = U1WilsonNf2(
        lattice_shape=shape,
        beta=float(args.beta),
        batch_size=1,
        seed=int(args.seed),
        mass=float(args.mass),
        include_gauge_monomial=True,
        include_fermion_monomial=False,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )
    ref = U1WilsonNf2DDReference(
        base=base,
        boundary_slices=cuts,
        boundary_width=int(args.boundary_width),
        dense_max_dof=int(args.dense_max_dof),
    )
    if str(args.start).lower() == "cold":
        q = jnp.ones(base.field_shape(), dtype=base.dtype)
    else:
        q = base.hot_start(scale=float(args.hot_scale))
    q0 = q
    chain = hmc(ref, minnorm2(ref.force, ref.evolve_q, int(args.nmd), float(args.tau)), verbose=False, seed=int(args.seed) + 1, use_fast_jit=False)

    t0 = time.perf_counter()
    q = chain.evolve(q, int(args.ntraj))
    t1 = time.perf_counter()

    print("DD metadata:")
    for k, v in ref.metadata().items():
        print(f"  {k}: {v}")
    print("Run summary:")
    print(f"  acceptance: {float(chain.calc_acceptance()):.4f}")
    print(f"  initial plaquette: {float(jnp.mean(base.average_plaquette(q0))):.6f}")
    print(f"  final plaquette:   {float(jnp.mean(base.average_plaquette(q))):.6f}")
    print(f"  initial conditional action: {float(jnp.mean(ref.action(q0))):.6f}")
    print(f"  final conditional action:   {float(jnp.mean(ref.action(q))):.6f}")
    print(f"  wall_sec: {float(t1 - t0):.3f}")
    print(f"  max frozen-link change: {_max_masked_abs(q - q0, ref.frozen_mask):.3e}")
    print(f"  max active-link change: {_max_masked_abs(q - q0, ref.active_mask):.3e}")


if __name__ == "__main__":
    main()

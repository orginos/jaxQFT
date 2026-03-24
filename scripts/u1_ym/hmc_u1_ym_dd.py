#!/usr/bin/env python3
"""Conditional DD HMC/SMD driver for pure U(1) Yang-Mills."""

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
from jaxqft.models.u1_ym import U1YangMills
from jaxqft.models.u1_ym_dd import U1TimeSlabDDTheory


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


def _build_integrator(name: str, theory, nmd: int, tau: float):
    key = str(name).strip().lower()
    if key == "minnorm2":
        return minnorm2(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "leapfrog":
        return leapfrog(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "forcegrad":
        return force_gradient(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolve_q, int(nmd), float(tau))
    raise ValueError(f"Unknown integrator: {name}")


def _max_masked_abs(x, mask) -> float:
    return float(jnp.max(jnp.abs(x * mask.astype(x.dtype))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Conditional DD HMC/SMD for pure U(1) Yang-Mills")
    ap.add_argument("--shape", type=str, default="8,8")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--boundary-slices", type=str, default="2,6")
    ap.add_argument("--boundary-width", type=int, default=1)
    ap.add_argument("--start", type=str, default="hot", choices=["hot", "cold"])
    ap.add_argument("--hot-scale", type=float, default=0.05)
    ap.add_argument("--ntraj", type=int, default=10)
    ap.add_argument("--nmd", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["minnorm2", "leapfrog", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--update", type=str, default="hmc", choices=["hmc", "smd", "ghmc"])
    ap.add_argument("--smd-gamma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    cuts = _parse_ints(args.boundary_slices)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    base = U1YangMills(
        lattice_shape=shape,
        beta=float(args.beta),
        batch_size=int(args.batch),
        layout="BMXYIJ",
        seed=int(args.seed),
    )
    dd = U1TimeSlabDDTheory(base, boundary_slices=cuts, boundary_width=int(args.boundary_width))
    integ = _build_integrator(args.integrator, dd, args.nmd, args.tau)

    if str(args.update).lower() == "hmc":
        chain = hmc(dd, integ, verbose=False, seed=int(args.seed) + 1, use_fast_jit=False)
        evolve = lambda q, n: chain.evolve(q, int(n))
    else:
        chain = SMD(
            dd,
            integ,
            gamma=float(args.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(args.seed) + 1,
            use_fast_jit=False,
        )
        evolve = lambda q, n: chain.evolve(q, int(n), warmup=False)

    if str(args.start).lower() == "cold":
        q = jnp.ones(base.field_shape(), dtype=base.dtype)
    else:
        q = base.hot_start(scale=float(args.hot_scale))
    q0 = q

    t0 = time.perf_counter()
    q = evolve(q, int(args.ntraj))
    t1 = time.perf_counter()

    dU = q - q0
    print("DD metadata:")
    for k, v in dd.metadata().items():
        print(f"  {k}: {v}")
    print("Run summary:")
    print(f"  update: {str(args.update).lower()}")
    print(f"  integrator: {str(args.integrator).lower()}")
    print(f"  acceptance: {float(chain.calc_acceptance()):.4f}")
    print(f"  initial plaquette: {float(jnp.mean(base.average_plaquette(q0))):.6f}")
    print(f"  final plaquette:   {float(jnp.mean(base.average_plaquette(q))):.6f}")
    print(f"  wall_sec: {float(t1 - t0):.3f}")
    print(f"  max frozen-link change: {_max_masked_abs(dU, dd.frozen_mask):.3e}")
    print(f"  max active-link change: {_max_masked_abs(dU, dd.active_mask):.3e}")


if __name__ == "__main__":
    main()

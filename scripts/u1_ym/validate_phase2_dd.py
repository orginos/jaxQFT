#!/usr/bin/env python3
"""Validation suite for phase-2 pure-gauge DD updates in 2D U(1)."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

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

from jaxqft.core.integrators import minnorm2
from jaxqft.core.update import SMD, hmc
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_ym import U1YangMills
from jaxqft.models.u1_ym_dd import U1TimeSlabDDTheory


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    wall_sec: float


def _assert_close(name: str, got, ref, *, atol: float, rtol: float) -> str:
    arr = np.asarray(got)
    ref_arr = np.asarray(ref)
    diff = np.asarray(arr - ref_arr)
    abs_err = float(np.max(np.abs(diff)))
    ref_max = float(np.max(np.abs(ref_arr)))
    rel_err = abs_err / ref_max if ref_max > 0.0 else abs_err
    if not np.allclose(arr, ref_arr, atol=float(atol), rtol=float(rtol)):
        raise AssertionError(
            f"{name} mismatch: abs_err={abs_err:.3e} rel_err={rel_err:.3e} "
            f"(atol={float(atol):.3e}, rtol={float(rtol):.3e})"
        )
    return f"abs_err={abs_err:.3e} rel_err={rel_err:.3e}"


def _algebra_inner(a, b):
    trab = jnp.real(a * b)
    return -jnp.sum(trab, axis=tuple(range(1, trab.ndim)))


def _max_masked_abs(x, mask) -> float:
    return float(jnp.max(jnp.abs(x * mask.astype(x.dtype))))


def _build_base(shape: Sequence[int], *, seed: int, beta: float = 4.0, batch_size: int = 1) -> U1YangMills:
    return U1YangMills(
        lattice_shape=tuple(int(v) for v in shape),
        beta=float(beta),
        batch_size=int(batch_size),
        layout="BMXYIJ",
        seed=int(seed),
    )


def _build_wilson_quenched(shape: Sequence[int], *, seed: int, beta: float = 4.0, batch_size: int = 1) -> U1WilsonNf2:
    return U1WilsonNf2(
        lattice_shape=tuple(int(v) for v in shape),
        beta=float(beta),
        batch_size=int(batch_size),
        seed=int(seed),
        mass=0.0,
        include_gauge_monomial=True,
        include_fermion_monomial=False,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )


def _run_test(name: str, fn: Callable[[], str], out: List[TestResult]) -> None:
    t0 = time.perf_counter()
    try:
        out.append(TestResult(name=name, passed=True, detail=str(fn()), wall_sec=float(time.perf_counter() - t0)))
    except Exception as exc:
        out.append(
            TestResult(
                name=name,
                passed=False,
                detail=f"{type(exc).__name__}: {exc}",
                wall_sec=float(time.perf_counter() - t0),
            )
        )


def _test_link_masks() -> str:
    base = _build_base((8, 8), seed=0)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    active = np.asarray(dd.active_mask)
    frozen = np.asarray(dd.frozen_mask)
    if not np.array_equal(active + frozen, np.ones_like(active)):
        raise AssertionError("active/frozen masks do not partition the links")

    time_dir = int(base.Nd - 1)
    if not np.all(active[:, 0, :, 1] == 1.0):
        raise AssertionError("Spatial links at interior-adjacent times should remain active")
    if not np.all(active[:, time_dir, :, 1] == 0.0):
        raise AssertionError("Temporal links crossing into a boundary should be frozen")
    if not np.all(active[:, time_dir, :, 3] == 1.0):
        raise AssertionError("Temporal links wholly inside the interior should be active")
    return (
        f"active_link_fraction={dd.active_link_fraction:.3f} "
        f"active_plaquette_fraction={dd.active_plaquette_fraction:.3f}"
    )


def _test_directional_fd(eps: float) -> str:
    base = _build_base((8, 8), seed=1)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    U = base.hot_start(scale=0.05)
    H = dd.refresh_p()
    Up = base.algebra_to_links(+float(eps) * H) * U
    Um = base.algebra_to_links(-float(eps) * H) * U
    dS_fd = (dd.action(Up) - dd.action(Um)) / (2.0 * float(eps))
    dS_force = -_algebra_inner(dd.force(U), H)
    rel = float(jnp.abs(jnp.mean(dS_fd) - jnp.mean(dS_force)) / (jnp.abs(jnp.mean(dS_fd)) + 1e-12))
    if rel > 5e-3:
        raise AssertionError(f"directional FD mismatch too large: rel={rel:.3e}")
    return f"rel_fd_vs_force={rel:.3e}"


def _test_action_delta_matches_full() -> str:
    base = _build_base((8, 8), seed=2)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    U0 = base.hot_start(scale=0.05)
    P = base.refresh_p()
    U1 = dd.evolve_q(0.25, P, U0)
    d_full = base.action(U1) - base.action(U0)
    d_dd = dd.action(U1) - dd.action(U0)
    return _assert_close("active_vs_full_action_delta", d_dd, d_full, atol=1e-5, rtol=1e-5)


def _test_evolve_q_boundary_invariance() -> str:
    base = _build_base((8, 8), seed=3)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    U0 = base.hot_start(scale=0.05)
    P = base.refresh_p()
    U1 = dd.evolve_q(0.3, P, U0)
    dU = U1 - U0
    frozen = _max_masked_abs(dU, dd.frozen_mask)
    active = _max_masked_abs(dU, dd.active_mask)
    if frozen > 1e-7:
        raise AssertionError(f"Frozen links changed under evolve_q: max={frozen:.3e}")
    if active < 1e-6:
        raise AssertionError(f"Active links barely moved under evolve_q: max={active:.3e}")
    return f"max_frozen_delta={frozen:.3e} max_active_delta={active:.3e}"


def _test_hmc_boundary_invariance() -> str:
    base = _build_base((8, 8), seed=4)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    U0 = base.hot_start(scale=0.05)
    chain = hmc(dd, minnorm2(dd.force, dd.evolve_q, 4, 0.5), verbose=False, seed=11, use_fast_jit=False)
    U1 = chain.evolve(U0, 3)
    dU = U1 - U0
    frozen = _max_masked_abs(dU, dd.frozen_mask)
    active = _max_masked_abs(dU, dd.active_mask)
    acc = float(chain.calc_acceptance())
    if frozen > 1e-7:
        raise AssertionError(f"Frozen links changed under DD HMC: max={frozen:.3e}")
    if active < 1e-6:
        raise AssertionError(f"Active links barely moved under DD HMC: max={active:.3e}")
    if not (0.0 <= acc <= 1.0):
        raise AssertionError(f"Invalid HMC acceptance value: {acc}")
    return f"pacc={acc:.3f} max_frozen_delta={frozen:.3e} max_active_delta={active:.3e}"


def _test_smd_boundary_invariance() -> str:
    base = _build_base((8, 8), seed=5)
    dd = U1TimeSlabDDTheory(base, boundary_slices=(2, 6), boundary_width=1)
    U0 = base.hot_start(scale=0.05)
    chain = SMD(dd, minnorm2(dd.force, dd.evolve_q, 4, 0.5), gamma=0.3, accept_reject=True, verbose=False, seed=17, use_fast_jit=False)
    U1 = chain.evolve(U0, 3, warmup=False)
    dU = U1 - U0
    frozen = _max_masked_abs(dU, dd.frozen_mask)
    active = _max_masked_abs(dU, dd.active_mask)
    acc = float(chain.calc_acceptance())
    if frozen > 1e-7:
        raise AssertionError(f"Frozen links changed under DD SMD: max={frozen:.3e}")
    if active < 1e-6:
        raise AssertionError(f"Active links barely moved under DD SMD: max={active:.3e}")
    if not (0.0 <= acc <= 1.0):
        raise AssertionError(f"Invalid SMD acceptance value: {acc}")
    return f"pacc={acc:.3f} max_frozen_delta={frozen:.3e} max_active_delta={active:.3e}"


def _test_u1_wilson_compatibility() -> str:
    ym = _build_base((6, 6), seed=6, batch_size=1)
    wq = _build_wilson_quenched((6, 6), seed=7, batch_size=1)
    q = ym.hot_start(scale=0.05)
    dd_ym = U1TimeSlabDDTheory(ym, boundary_slices=(1, 4), boundary_width=1)
    dd_wq = U1TimeSlabDDTheory(wq, boundary_slices=(1, 4), boundary_width=1)
    d1 = _assert_close("gauge_action", ym.action(q), wq.action(q), atol=1e-6, rtol=1e-6)
    d2 = _assert_close("gauge_force", ym.force(q), wq.force(q), atol=1e-6, rtol=1e-6)
    d3 = _assert_close("dd_action", dd_ym.action(q), dd_wq.action(q), atol=1e-6, rtol=1e-6)
    d4 = _assert_close("dd_force", dd_ym.force(q), dd_wq.force(q), atol=1e-6, rtol=1e-6)
    return "; ".join([d1, d2, d3, d4])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-2 DD pure-gauge validation suite")
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    args = ap.parse_args()

    results: List[TestResult] = []
    _run_test("link_masks", _test_link_masks, results)
    _run_test("directional_fd", lambda: _test_directional_fd(float(args.fd_eps)), results)
    _run_test("action_delta_matches_full", _test_action_delta_matches_full, results)
    _run_test("evolve_q_boundary_invariance", _test_evolve_q_boundary_invariance, results)
    _run_test("hmc_boundary_invariance", _test_hmc_boundary_invariance, results)
    _run_test("smd_boundary_invariance", _test_smd_boundary_invariance, results)
    _run_test("u1_wilson_compatibility", _test_u1_wilson_compatibility, results)

    nfail = sum(0 if r.passed else 1 for r in results)
    total = float(sum(r.wall_sec for r in results))
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        print(f"[{tag}] {r.name}: {r.detail} ({r.wall_sec:.3f}s)")
    print(f"summary: {len(results) - nfail}/{len(results)} passed, total_wall={total:.3f}s")
    if nfail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

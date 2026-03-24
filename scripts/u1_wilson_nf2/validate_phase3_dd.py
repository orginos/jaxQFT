#!/usr/bin/env python3
"""Validation suite for phase-3 exact DD determinant factorization in U(1) Wilson Nf=2."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

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
from jaxqft.core.update import hmc
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_wilson_nf2_dd import U1WilsonNf2DDReference


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


def _max_masked_abs(x, mask) -> float:
    return float(jnp.max(jnp.abs(x * mask.astype(x.dtype))))


def _build_base(shape: Sequence[int], *, seed: int, mass: float, beta: float = 4.0) -> U1WilsonNf2:
    return U1WilsonNf2(
        lattice_shape=tuple(int(v) for v in shape),
        beta=float(beta),
        batch_size=1,
        seed=int(seed),
        mass=float(mass),
        include_gauge_monomial=True,
        include_fermion_monomial=False,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )


def _build_ref(shape: Sequence[int], *, seed: int, mass: float, boundary_slices=(1, 3), boundary_width=1, dense_max_dof=512):
    base = _build_base(shape, seed=seed, mass=mass)
    return U1WilsonNf2DDReference(
        base=base,
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
        dense_max_dof=int(dense_max_dof),
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


def _test_factorization_identity(shape, *, seed: int, mass: float, tol: float, hot: bool) -> str:
    ref = _build_ref(shape, seed=seed, mass=mass, boundary_slices=(1, shape[-1] // 2), boundary_width=1)
    if bool(hot):
        U = ref.base.hot_start(scale=0.05)
    else:
        U = jnp.ones(ref.base.field_shape(), dtype=ref.base.dtype)
    pieces = ref.factorization_breakdown(U)
    rhs = pieces["boundary_logabsdet"] + jnp.sum(pieces["local_logabsdet"], axis=1) + pieces["correction_logabsdet"]
    return _assert_close("logdet_factorization", pieces["full_logabsdet"], rhs, atol=tol, rtol=tol)


def _test_action_breakdown_sum() -> str:
    ref = _build_ref((4, 4), seed=2, mass=0.1, boundary_slices=(1, 3), boundary_width=1)
    U = ref.base.hot_start(scale=0.05)
    br = ref.action_breakdown(U)
    total = None
    for v in br.values():
        total = v if total is None else total + v
    assert total is not None
    return _assert_close("action_breakdown_sum", total, ref.action(U), atol=1e-6, rtol=1e-6)


def _test_fermion_action_split(tol: float) -> str:
    ref = _build_ref((6, 6), seed=6, mass=0.1, boundary_slices=(1, 3), boundary_width=1)
    U = ref.base.hot_start(scale=0.05)
    split = ref.boundary_constant_action(U) + ref.factorized_fermion_action(U)
    return _assert_close("fermion_action_split", ref.full_exact_fermion_action(U), split, atol=tol, rtol=tol)


def _test_conditional_delta_matches_full(tol: float) -> str:
    ref = _build_ref((6, 6), seed=3, mass=0.1, boundary_slices=(1, 3), boundary_width=1)
    U0 = ref.base.hot_start(scale=0.05)
    P = ref.refresh_p()
    U1 = ref.evolve_q(0.2, P, U0)
    d_cond = ref.conditional_exact_total_action(U1) - ref.conditional_exact_total_action(U0)
    d_full = ref.full_exact_total_action(U1) - ref.full_exact_total_action(U0)
    return _assert_close("conditional_delta_vs_full", d_cond, d_full, atol=tol, rtol=tol)


def _test_hmc_boundary_invariance() -> str:
    ref = _build_ref((4, 4), seed=4, mass=0.2, boundary_slices=(1, 3), boundary_width=1, dense_max_dof=256)
    U0 = ref.base.hot_start(scale=0.05)
    chain = hmc(ref, minnorm2(ref.force, ref.evolve_q, 2, 0.3), verbose=False, seed=17, use_fast_jit=False)
    U1 = chain.evolve(U0, 1)
    dU = U1 - U0
    frozen = _max_masked_abs(dU, ref.frozen_mask)
    active = _max_masked_abs(dU, ref.active_mask)
    acc = float(chain.calc_acceptance())
    if frozen > 1e-7:
        raise AssertionError(f"Frozen links changed under DD HMC: max={frozen:.3e}")
    if active < 1e-7:
        raise AssertionError(f"Active links barely moved under DD HMC: max={active:.3e}")
    if not (0.0 <= acc <= 1.0):
        raise AssertionError(f"Invalid HMC acceptance value: {acc}")
    return f"pacc={acc:.3f} max_frozen_delta={frozen:.3e} max_active_delta={active:.3e}"


def _test_boundary_correction_present() -> str:
    ref = _build_ref((6, 6), seed=5, mass=0.1, boundary_slices=(1, 3), boundary_width=1)
    U = ref.base.hot_start(scale=0.05)
    pieces = ref.factorization_breakdown(U)
    corr = np.asarray(pieces["correction_logabsdet"])
    if not np.all(np.isfinite(corr)):
        raise AssertionError("Non-finite boundary correction logdet")
    return f"mean_correction_logabsdet={float(np.mean(corr)):.6e}"


def _test_boundary_constant_invariance(tol: float) -> str:
    ref = _build_ref((6, 6), seed=7, mass=0.1, boundary_slices=(1, 3), boundary_width=1)
    U0 = ref.base.hot_start(scale=0.05)
    P = ref.refresh_p()
    U1 = ref.evolve_q(0.2, P, U0)
    delta = ref.boundary_constant_action(U1) - ref.boundary_constant_action(U0)
    return _assert_close("boundary_constant_delta", delta, jnp.zeros_like(delta), atol=tol, rtol=tol)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-3 DD exact-factorization validation suite")
    ap.add_argument("--tol", type=float, default=5e-5)
    args = ap.parse_args()

    results: List[TestResult] = []
    _run_test(
        "factorization_identity_hot_6x6",
        lambda: _test_factorization_identity((6, 6), seed=0, mass=0.0, tol=float(args.tol), hot=True),
        results,
    )
    _run_test(
        "factorization_identity_cold_massive_6x6",
        lambda: _test_factorization_identity((6, 6), seed=1, mass=0.2, tol=float(args.tol), hot=False),
        results,
    )
    _run_test("action_breakdown_sum", _test_action_breakdown_sum, results)
    _run_test("fermion_action_split", lambda: _test_fermion_action_split(float(args.tol)), results)
    _run_test("conditional_delta_matches_full", lambda: _test_conditional_delta_matches_full(float(args.tol)), results)
    _run_test("hmc_boundary_invariance", _test_hmc_boundary_invariance, results)
    _run_test("boundary_correction_present", _test_boundary_correction_present, results)
    _run_test("boundary_constant_invariance", lambda: _test_boundary_constant_invariance(float(args.tol)), results)

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

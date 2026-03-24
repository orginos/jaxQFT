#!/usr/bin/env python3
"""Validation suite for phase-2 quenched multilevel DD pion measurements."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

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
from jaxqft.core.multilevel_quenched import (
    build_projector_basis,
    build_two_level_pion_geometry,
    compute_factorized_pion_blocks,
    factorized_pion_corr_from_blocks,
    split_domain_masks,
)
from jaxqft.core.integrators import minnorm2
from jaxqft.core.update import HMC
from jaxqft.models import U1TimeSlabDDTheory
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    wall_sec: float


def _make_theory(shape=(6, 8), *, seed: int = 0, mass: float = 0.1) -> U1WilsonNf2:
    return U1WilsonNf2(
        lattice_shape=tuple(int(v) for v in shape),
        beta=4.0,
        batch_size=1,
        seed=int(seed),
        mass=float(mass),
        include_fermion_monomial=False,
        include_gauge_monomial=True,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )


def _make_fixture():
    theory = _make_theory()
    q = theory.hot_start(scale=0.05)
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    geom = build_two_level_pion_geometry(
        lattice_shape=tuple(int(v) for v in theory.lattice_shape),
        boundary_slices=(2, 6),
        boundary_width=1,
        source=(0, 0),
        nsc=int(ns * nc),
        source_margin=1,
        momentum_axis=0,
    )
    return theory, q, geom


def _dd_sample(theory, q, *, seed: int, ntraj: int = 2):
    dd = U1TimeSlabDDTheory(theory, boundary_slices=(2, 6), boundary_width=1)
    chain = HMC(dd, minnorm2(dd.force, dd.evolve_q, 3, 0.3), verbose=False, seed=int(seed), use_fast_jit=False)
    return chain.evolve(q, int(ntraj))


def _splice(q_source, q_sink, geom) -> Any:
    src_mask, sink_mask = split_domain_masks(geom, batch_size=int(q_source.shape[0]), layout="BMXYIJ", dtype=np.float32)
    src_mask_j = jnp.asarray(src_mask, dtype=q_source.dtype)
    sink_mask_j = jnp.asarray(sink_mask, dtype=q_source.dtype)
    bnd_mask_j = 1.0 - src_mask_j - sink_mask_j
    return q_source * src_mask_j + q_sink * sink_mask_j + q_source * bnd_mask_j


def _assert_close(name: str, got, want, *, atol: float, rtol: float) -> str:
    got_a = np.asarray(got)
    want_a = np.asarray(want)
    diff = np.max(np.abs(got_a - want_a))
    ref = np.max(np.abs(want_a))
    rel = float(diff / ref) if ref > 0.0 else float(diff)
    if not np.allclose(got_a, want_a, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: abs_err={diff:.3e} rel_err={rel:.3e}")
    return f"abs_err={diff:.3e} rel_err={rel:.3e}"


def _test_projectors() -> str:
    theory, q, geom = _make_fixture()
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    phi_full = build_projector_basis(q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="full", nvec=0, probe_stride=2)
    phi_probe = build_projector_basis(q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="probe", nvec=0, probe_stride=2)
    phi_lap = build_projector_basis(q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="laplace", nvec=2, probe_stride=2)
    if phi_full.shape[1] != phi_full.shape[2]:
        raise AssertionError(f"full projector should be square, got {phi_full.shape}")
    if phi_probe.shape[1] >= phi_full.shape[1]:
        raise AssertionError(f"probe projector should be reduced, got {phi_probe.shape} vs {phi_full.shape}")
    gram = np.asarray(phi_lap[0] @ np.conjugate(phi_lap[0]).T)
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6, rtol=1e-6):
        raise AssertionError("laplace projector vectors are not orthonormal")
    return f"n_full={phi_full.shape[1]} n_probe={phi_probe.shape[1]} n_lap={phi_lap.shape[1]}"


def _test_domain_locality() -> str:
    theory, q0, geom = _make_fixture()
    q1 = _dd_sample(theory, q0, seed=11, ntraj=2)
    q2 = _dd_sample(theory, q0, seed=17, ntraj=3)
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    phi = build_projector_basis(q=q0, theory=theory, geometry=geom, ns=ns, nc=nc, kind="full", nvec=0, probe_stride=2)

    src0, sink0 = compute_factorized_pion_blocks(q=q0, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    src1, sink1 = compute_factorized_pion_blocks(q=q1, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    sp_src = _splice(q0, q1, geom)
    sp_sink = _splice(q2, q0, geom)
    src_sp, sink_sp = compute_factorized_pion_blocks(q=sp_src, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    src_sp2, sink_sp2 = compute_factorized_pion_blocks(q=sp_sink, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)

    d1 = _assert_close("source_locality", src_sp, src0, atol=1e-8, rtol=1e-8)
    d2 = _assert_close("sink_locality", sink_sp, sink1, atol=1e-8, rtol=1e-8)
    d3 = _assert_close("sink_locality_2", sink_sp2, sink0, atol=1e-8, rtol=1e-8)
    return f"{d1}; {d2}; {d3}"


def _test_pair_average_identity() -> str:
    theory, q0, geom = _make_fixture()
    q1 = _dd_sample(theory, q0, seed=21, ntraj=2)
    q2 = _dd_sample(theory, q0, seed=31, ntraj=3)
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    phi = build_projector_basis(q=q0, theory=theory, geometry=geom, ns=ns, nc=nc, kind="probe", nvec=0, probe_stride=2)

    src1, sink1 = compute_factorized_pion_blocks(q=q1, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    src2, sink2 = compute_factorized_pion_blocks(q=q2, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    mean_corr, mask = factorized_pion_corr_from_blocks(
        source_blocks=0.5 * (src1 + src2),
        sink_blocks=0.5 * (sink1 + sink2),
        geometry=geom,
        momenta=(0, 1),
        average_pm=True,
    )
    pair_terms = []
    for src in (src1, src2):
        for sink in (sink1, sink2):
            corr, mask_ij = factorized_pion_corr_from_blocks(
                source_blocks=src,
                sink_blocks=sink,
                geometry=geom,
                momenta=(0, 1),
                average_pm=True,
            )
            if not np.array_equal(mask, mask_ij):
                raise AssertionError("valid-mask mismatch inside pair-average identity test")
            pair_terms.append(corr)
    pair_avg = 0.25 * sum(pair_terms)
    return _assert_close("pair_average_identity", mean_corr, pair_avg, atol=1e-10, rtol=1e-10)


def _test_bias_identity() -> str:
    theory, q0, _ = _make_fixture()
    specs = [
        {
            "type": "pion_2pt_ml_dd",
            "name": "ml",
            "source": [0, 0],
            "momenta": [0, 1],
            "average_pm": True,
            "boundary_slices": [2, 6],
            "boundary_width": 1,
            "projector_kind": "full",
            "level1_ncfg": 1,
            "level1_warmup": 0,
            "level1_skip": 0,
            "level1_update": "hmc",
            "level1_integrator": "minnorm2",
            "level1_nmd": 2,
            "level1_tau": 0.2,
            "dense_max_domain_dof": 512,
            "exact_backend": "dense",
            "exact_dense_max_dof": 4096,
        }
    ]
    recs = run_inline_measurements(build_inline_measurements(specs), step=0, q=q0, theory=theory, context=MeasurementContext())
    if len(recs) != 1:
        raise AssertionError(f"Expected one multilevel record, got {len(recs)}")
    vals = recs[0]["values"]
    checked = 0
    for key, value in vals.items():
        if not key.startswith("approx_l0_p") or not key.endswith("_re"):
            continue
        stem = key[: -len("_re")]
        a = float(vals[stem + "_re"]) + 1j * float(vals[stem + "_im"])
        b = float(vals[stem.replace("approx_l0", "bias") + "_re"]) + 1j * float(vals[stem.replace("approx_l0", "bias") + "_im"])
        e = float(vals[stem.replace("approx_l0", "exact") + "_re"]) + 1j * float(vals[stem.replace("approx_l0", "exact") + "_im"])
        if abs((a + b) - e) > 1e-8:
            raise AssertionError(f"bias identity failed for {stem}")
        checked += 1
    if checked == 0:
        raise AssertionError("No approx_l0 entries found in multilevel measurement output")
    return f"checked {checked} valid times"


def _run_test(name: str, fn: Callable[[], str], results: List[TestResult]) -> None:
    t0 = time.perf_counter()
    try:
        detail = fn()
        dt = float(time.perf_counter() - t0)
        results.append(TestResult(name=name, passed=True, detail=detail, wall_sec=dt))
        print(f"[PASS] {name}: {detail} ({dt:.3f}s)")
    except Exception as exc:
        dt = float(time.perf_counter() - t0)
        results.append(TestResult(name=name, passed=False, detail=str(exc), wall_sec=dt))
        print(f"[FAIL] {name}: {exc} ({dt:.3f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-2 multilevel DD quenched pion validation suite")
    args = ap.parse_args()
    _ = args

    results: List[TestResult] = []
    t0 = time.perf_counter()
    _run_test("projectors", _test_projectors, results)
    _run_test("domain_locality", _test_domain_locality, results)
    _run_test("pair_average_identity", _test_pair_average_identity, results)
    _run_test("bias_identity", _test_bias_identity, results)
    passed = sum(1 for r in results if r.passed)
    print(f"summary: {passed}/{len(results)} passed, total_wall={float(time.perf_counter() - t0):.3f}s")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

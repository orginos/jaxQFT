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
    build_giusti_surface_projector_basis,
    build_projector_basis,
    build_two_level_pion_geometry,
    compute_factorized_pion_factors,
    compute_factorized_pion_blocks,
    compute_giusti_asymmetric_pion_factors,
    compute_giusti_asymmetric_pion_blocks,
    factorized_pion_corr_from_blocks,
    split_domain_masks,
)
from jaxqft.core.integrators import minnorm2
from jaxqft.core.update import HMC
from jaxqft.models import U1TimeSlabDDTheory
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2
from jaxqft.models.u1_ym import gauge_transform_links, random_site_gauge


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


def _splice_three(q_source, q_sink, q_boundary, geom) -> Any:
    src_mask, sink_mask = split_domain_masks(geom, batch_size=int(q_source.shape[0]), layout="BMXYIJ", dtype=np.float32)
    src_mask_j = jnp.asarray(src_mask, dtype=q_source.dtype)
    sink_mask_j = jnp.asarray(sink_mask, dtype=q_source.dtype)
    bnd_mask_j = 1.0 - src_mask_j - sink_mask_j
    return q_source * src_mask_j + q_sink * sink_mask_j + q_boundary * bnd_mask_j


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
    phi_svd = build_projector_basis(q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="svd", nvec=4, probe_stride=2)
    g_src_full = build_giusti_surface_projector_basis(
        q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="full", nvec=0, probe_stride=2, dressed_domain="source"
    )
    g_src_probe = build_giusti_surface_projector_basis(
        q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="probe", nvec=0, probe_stride=2, dressed_domain="source"
    )
    g_src_lap = build_giusti_surface_projector_basis(
        q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="laplace", nvec=2, probe_stride=2, dressed_domain="source"
    )
    g_sink_lap = build_giusti_surface_projector_basis(
        q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="laplace", nvec=2, probe_stride=2, dressed_domain="sink"
    )
    if phi_full.shape[1] != phi_full.shape[2]:
        raise AssertionError(f"full projector should be square, got {phi_full.shape}")
    if phi_probe.shape[1] >= phi_full.shape[1]:
        raise AssertionError(f"probe projector should be reduced, got {phi_probe.shape} vs {phi_full.shape}")
    if g_src_full.shape[1] != g_src_full.shape[2]:
        raise AssertionError(f"Giusti full projector should be square, got {g_src_full.shape}")
    if g_src_probe.shape[1] >= g_src_full.shape[1]:
        raise AssertionError(f"Giusti probe projector should be reduced, got {g_src_probe.shape} vs {g_src_full.shape}")
    for name, phi in (("laplace", phi_lap), ("svd", phi_svd)):
        gram = np.asarray(phi[0] @ np.conjugate(phi[0]).T)
        if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{name} projector vectors are not orthonormal")
    for name, phi in (("giusti_src_lap", g_src_lap), ("giusti_sink_lap", g_sink_lap)):
        gram = np.asarray(phi[0] @ np.conjugate(phi[0]).T)
        if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{name} projector vectors are not orthonormal")
    overlap_lookup = dict(geom.overlap_lookup)
    nsites_per_slab = [int(s.size) for s in geom.boundary_slab_sites]
    support_first = np.zeros((int(geom.overlap_sites.size) * ns * nc,), dtype=bool)
    for site in np.asarray(geom.boundary_slab_sites[0], dtype=np.int64).tolist():
        loc = int(overlap_lookup[int(site)])
        support_first[loc * ns * nc : (loc + 1) * ns * nc] = True
    support_second = np.logical_not(support_first)
    if np.max(np.abs(phi_svd[0, :2, support_second])) > 1e-10 or np.max(np.abs(phi_svd[0, 2:, support_first])) > 1e-10:
        raise AssertionError("svd projector rows are not localized on individual slabs")
    if int(g_src_full.shape[2]) != int(geom.sink_surface_sites.size) * ns * nc:
        raise AssertionError("Giusti source-dressed support does not match sink-facing surface size")
    return (
        f"n_full={phi_full.shape[1]} n_probe={phi_probe.shape[1]} "
        f"n_lap={phi_lap.shape[1]} n_svd={phi_svd.shape[1]} "
        f"g_src_full={g_src_full.shape[1]} g_src_probe={g_src_probe.shape[1]} "
        f"slab_sites={tuple(nsites_per_slab)}"
    )


def _test_domain_locality() -> str:
    theory, q0, geom = _make_fixture()
    q1 = _dd_sample(theory, q0, seed=11, ntraj=2)
    q2 = _dd_sample(theory, q0, seed=17, ntraj=3)
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    phi = build_projector_basis(q=q0, theory=theory, geometry=geom, ns=ns, nc=nc, kind="full", nvec=0, probe_stride=2)

    q_src_a = _splice_three(q0, q1, q2, geom)
    q_src_b = _splice_three(q0, q2, q2, geom)
    q_sink_a = _splice_three(q1, q0, q2, geom)
    q_sink_b = _splice_three(q2, q0, q2, geom)

    src_a, _ = compute_factorized_pion_blocks(q=q_src_a, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    src_b, _ = compute_factorized_pion_blocks(q=q_src_b, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    _, sink_c = compute_factorized_pion_blocks(q=q_sink_a, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)
    _, sink_d = compute_factorized_pion_blocks(q=q_sink_b, theory=theory, geometry=geom, phi_overlap=phi, dense_max_domain_dof=512)

    d1 = _assert_close("source_locality", src_a, src_b, atol=1e-8, rtol=1e-8)
    d2 = _assert_close("sink_locality", sink_c, sink_d, atol=1e-8, rtol=1e-8)
    return f"{d1}; {d2}"


def _test_block_contraction_identity() -> str:
    theory, q, geom = _make_fixture()
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    phi = build_projector_basis(q=q, theory=theory, geometry=geom, ns=ns, nc=nc, kind="full", nvec=0, probe_stride=2)
    src_fac, sink_fac = compute_factorized_pion_factors(
        q=q,
        theory=theory,
        geometry=geom,
        phi_overlap=phi,
        dense_max_domain_dof=512,
    )
    src_blk, sink_blk = compute_factorized_pion_blocks(
        q=q,
        theory=theory,
        geometry=geom,
        phi_overlap=phi,
        dense_max_domain_dof=512,
    )
    corr_blk, valid_mask = factorized_pion_corr_from_blocks(
        source_blocks=src_blk,
        sink_blocks=sink_blk,
        geometry=geom,
        momenta=(0, 1),
        average_pm=True,
    )
    prop = np.einsum("bsap,bpq->bsaq", sink_fac, src_fac, optimize=True)
    pair = np.sum(np.abs(prop) ** 2, axis=(2, 3))
    direct = np.zeros_like(corr_blk)
    if len(tuple(geom.decomposition.lattice_shape)) <= 1:
        lmom = 1
        x0 = 0
    else:
        lmom = int(geom.decomposition.lattice_shape[0])
        x0 = int(geom.source_coords[0]) % lmom
    for s in range(int(geom.sink_domain_sites.size)):
        dt = int(geom.sink_dt[s])
        x = int(geom.sink_momentum_coords[s]) % max(lmom, 1)
        for ip, p in enumerate((0, 1)):
            theta = (2.0 * np.pi * float(p) / float(max(lmom, 1))) * (float(x) - float(x0))
            direct[:, ip, dt] = direct[:, ip, dt] + np.cos(theta) * pair[:, s]
    expected_mask = np.zeros((int(geom.decomposition.lt),), dtype=bool)
    expected_mask[np.unique(np.asarray(geom.sink_dt, dtype=np.int64))] = True
    if not np.array_equal(np.asarray(valid_mask, dtype=bool), expected_mask):
        raise AssertionError("valid-mask mismatch in block_contraction_identity")
    return _assert_close("block_contraction_identity", corr_blk, direct, atol=1e-10, rtol=1e-10)


def _test_giusti_block_contraction_identity() -> str:
    theory, q, geom = _make_fixture()
    ns = int(theory.fermion_shape()[-2])
    nc = int(theory.fermion_shape()[-1])
    details = []
    for dressed in ("source", "sink"):
        phi = build_giusti_surface_projector_basis(
            q=q,
            theory=theory,
            geometry=geom,
            ns=ns,
            nc=nc,
            kind="full",
            nvec=0,
            probe_stride=2,
            dressed_domain=dressed,
        )
        src_fac, sink_fac = compute_giusti_asymmetric_pion_factors(
            q=q,
            theory=theory,
            geometry=geom,
            phi_surface=phi,
            dense_max_domain_dof=512,
            dressed_domain=dressed,
        )
        src_blk, sink_blk = compute_giusti_asymmetric_pion_blocks(
            q=q,
            theory=theory,
            geometry=geom,
            phi_surface=phi,
            dense_max_domain_dof=512,
            dressed_domain=dressed,
        )
        corr_blk, valid_mask = factorized_pion_corr_from_blocks(
            source_blocks=src_blk,
            sink_blocks=sink_blk,
            geometry=geom,
            momenta=(0, 1),
            average_pm=True,
        )
        prop = np.einsum("bsap,bpq->bsaq", sink_fac, src_fac, optimize=True)
        pair = np.sum(np.abs(prop) ** 2, axis=(2, 3))
        direct = np.zeros_like(corr_blk)
        if len(tuple(geom.decomposition.lattice_shape)) <= 1:
            lmom = 1
            x0 = 0
        else:
            lmom = int(geom.decomposition.lattice_shape[0])
            x0 = int(geom.source_coords[0]) % lmom
        for s in range(int(geom.sink_domain_sites.size)):
            dt = int(geom.sink_dt[s])
            x = int(geom.sink_momentum_coords[s]) % max(lmom, 1)
            for ip, p in enumerate((0, 1)):
                theta = (2.0 * np.pi * float(p) / float(max(lmom, 1))) * (float(x) - float(x0))
                direct[:, ip, dt] = direct[:, ip, dt] + np.cos(theta) * pair[:, s]
        expected_mask = np.zeros((int(geom.decomposition.lt),), dtype=bool)
        expected_mask[np.unique(np.asarray(geom.sink_dt, dtype=np.int64))] = True
        if not np.array_equal(np.asarray(valid_mask, dtype=bool), expected_mask):
            raise AssertionError(f"valid-mask mismatch in Giusti block_contraction_identity ({dressed})")
        details.append(
            _assert_close(
                f"giusti_block_contraction_identity_{dressed}",
                corr_blk,
                direct,
                atol=1e-10,
                rtol=1e-10,
            )
        )
    return "; ".join(details)


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
    checked_total = 0
    for fact_kind in ("boundary_transfer", "giusti"):
        specs = [
            {
                "type": "pion_2pt_ml_dd",
                "name": "ml",
                "source": [0, 0],
                "momenta": [0, 1],
                "average_pm": True,
                "boundary_slices": [2, 6],
                "boundary_width": 1,
                "factorization_kind": fact_kind,
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
                raise AssertionError(f"bias identity failed for {stem} ({fact_kind})")
            checked += 1
        if checked == 0:
            raise AssertionError(f"No approx_l0 entries found in multilevel measurement output ({fact_kind})")
        checked_total += checked
    return f"checked {checked_total} valid times across both factorization kinds"


def _test_measurement_gauge_invariance() -> str:
    theory, q0, _ = _make_fixture()
    compared_total = 0
    max_abs_err = 0.0
    max_rel_err = 0.0
    for fact_kind in ("boundary_transfer", "giusti"):
        specs = [
            {
                "type": "pion_2pt_ml_dd",
                "name": "ml",
                "source": [0, 0],
                "momenta": [0, 1],
                "average_pm": True,
                "boundary_slices": [2, 6],
                "boundary_width": 1,
                "factorization_kind": fact_kind,
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
        meas = build_inline_measurements(specs)
        vals0 = run_inline_measurements(meas, step=0, q=q0, theory=theory, context=MeasurementContext())[0]["values"]
        Om = random_site_gauge(theory, scale=0.05)
        q1 = gauge_transform_links(theory, q0, Om)
        vals1 = run_inline_measurements(meas, step=0, q=q1, theory=theory, context=MeasurementContext())[0]["values"]

        compare_prefixes = ("approx_l0_", "bias_", "approx_ml_", "corrected_", "exact_")
        compared = 0
        for key, value in vals0.items():
            k = str(key)
            if k.endswith("_valid_t") or "_valid_t" in k:
                other = vals1.get(k, None)
                if other is None or float(value) != float(other):
                    raise AssertionError(f"gauge invariance valid-mask mismatch for {k} ({fact_kind})")
                compared += 1
                continue
            if not any(k.startswith(pref) for pref in compare_prefixes):
                continue
            other = vals1.get(k, None)
            if other is None:
                raise AssertionError(f"Missing transformed measurement key {k} ({fact_kind})")
            z0 = complex(float(value))
            z1 = complex(float(other))
            abs_err = float(abs(z0 - z1))
            ref = max(abs(z0), abs(z1))
            rel_err = float(abs_err / ref) if ref > 0.0 else float(abs_err)
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)
            if not np.allclose(z0, z1, atol=5e-8, rtol=2e-7):
                raise AssertionError(
                    f"gauge invariance failed for {k} ({fact_kind}): "
                    f"abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
                )
            compared += 1
        if compared == 0:
            raise AssertionError(f"No multilevel correlator entries compared in gauge-invariance test ({fact_kind})")
        compared_total += compared
    return (
        f"checked {compared_total} correlator/mask entries across both factorization kinds "
        f"(max_abs_err={max_abs_err:.3e} max_rel_err={max_rel_err:.3e})"
    )


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
    _run_test("block_contraction_identity", _test_block_contraction_identity, results)
    _run_test("giusti_block_contraction_identity", _test_giusti_block_contraction_identity, results)
    _run_test("pair_average_identity", _test_pair_average_identity, results)
    _run_test("bias_identity", _test_bias_identity, results)
    _run_test("measurement_gauge_invariance", _test_measurement_gauge_invariance, results)
    passed = sum(1 for r in results if r.passed)
    print(f"summary: {passed}/{len(results)} passed, total_wall={float(time.perf_counter() - t0):.3f}s")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

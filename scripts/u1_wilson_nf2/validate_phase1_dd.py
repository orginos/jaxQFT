#!/usr/bin/env python3
"""Phase-1 validation suite for quenched DD fermionic observables in 2D U(1)."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

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

from jaxqft.core.domain_decomposition import TimeSlabDecomposition
from jaxqft.core.measurements import MeasurementContext, _dd_exact_schur_inverse, _full_dense_inverse
from jaxqft.core.measurements import build_inline_measurements, run_inline_measurements
from jaxqft.fermions import gamma5
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    wall_sec: float


@dataclass
class DDReferenceFixture:
    theory: U1WilsonNf2
    q: Any
    lattice_shape: Tuple[int, ...]
    boundary_slices: Tuple[int, ...]
    boundary_width: int
    ginv_interior: np.ndarray
    decomposition: TimeSlabDecomposition
    nsc: int


def _make_theory(
    shape: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    mass: float = 0.0,
    solver_kind: str = "cg",
) -> U1WilsonNf2:
    return U1WilsonNf2(
        lattice_shape=tuple(int(v) for v in shape),
        beta=4.0,
        batch_size=int(batch_size),
        seed=int(seed),
        mass=float(mass),
        solver_kind=str(solver_kind),
        include_gauge_monomial=False,
        include_fermion_monomial=False,
        jit_dirac_kernels=False,
        jit_solvers=False,
    )


def _cold_cfg(theory: U1WilsonNf2):
    return jnp.ones(theory.field_shape(), dtype=theory.dtype)


def _site_arrays(lattice_shape: Sequence[int], momentum_axis: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    shp = tuple(int(v) for v in lattice_shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    coords = np.asarray(np.unravel_index(np.arange(vol, dtype=np.int64), shp), dtype=np.int64).T
    t = coords[:, -1].astype(np.int64)
    lt = int(shp[-1])
    if len(shp) <= 1:
        x = np.zeros((vol,), dtype=np.int64)
        lmom = 1
    else:
        x = coords[:, int(momentum_axis)].astype(np.int64)
        lmom = int(shp[int(momentum_axis)])
    return t, x, lt, lmom


def _extract_interior_block(ginv: np.ndarray, decomposition: TimeSlabDecomposition, nsc: int) -> np.ndarray:
    i_idx = decomposition.interior_component_indices(int(nsc))
    return np.stack([np.asarray(ginv[b][np.ix_(i_idx, i_idx)]) for b in range(int(ginv.shape[0]))], axis=0)


def _assert_close(name: str, got: np.ndarray, ref: np.ndarray, *, atol: float, rtol: float) -> str:
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


def _extract_corr(values: Mapping[str, float], prefix: str, momenta: Sequence[int], lt: int) -> np.ndarray:
    moms = tuple(int(v) for v in momenta)
    out = np.zeros((len(moms), int(lt)), dtype=np.complex128)
    for ip, p in enumerate(moms):
        for t in range(int(lt)):
            re_key = f"{prefix}_p{int(p)}_t{int(t)}_re"
            im_key = f"{prefix}_p{int(p)}_t{int(t)}_im"
            if re_key not in values or im_key not in values:
                raise KeyError(f"Missing correlator key(s): {re_key}, {im_key}")
            out[ip, t] = complex(float(values[re_key]), float(values[im_key]))
    return out


def _manual_pair_correlator(
    pair_vals: np.ndarray,
    t_site: np.ndarray,
    x_site: np.ndarray,
    lt: int,
    lmom: int,
    momenta: Sequence[int],
    *,
    source_local: int | None,
    source_average: bool,
) -> np.ndarray:
    moms = tuple(int(v) for v in momenta)
    nsites = int(pair_vals.shape[0])
    corr = np.zeros((len(moms), int(lt)), dtype=np.complex128)
    if bool(source_average):
        y_iter = range(nsites)
        norm = float(nsites)
    else:
        if source_local is None:
            raise ValueError("source_local is required when source_average=False")
        y_iter = [int(source_local)]
        norm = 1.0

    for y in y_iter:
        for x in range(nsites):
            dt = int((int(t_site[x]) - int(t_site[y])) % int(lt))
            dx = float(int(x_site[x]) - int(x_site[y]))
            val = complex(pair_vals[x, y])
            if int(lmom) <= 1:
                for ip in range(len(moms)):
                    corr[ip, dt] = corr[ip, dt] + val
                continue
            for ip, p in enumerate(moms):
                ph = np.exp(1j * (2.0 * np.pi * float(p) / float(lmom)) * dx)
                corr[ip, dt] = corr[ip, dt] + val * ph

    return corr / float(norm)


def _manual_pion_from_full_inverse(
    fixture: DDReferenceFixture,
    momenta: Sequence[int],
    *,
    source: Sequence[int],
    source_average: bool,
    momentum_axis: int = 0,
) -> np.ndarray:
    shp = tuple(int(v) for v in fixture.lattice_shape)
    t_all, x_all, lt, lmom = _site_arrays(shp, momentum_axis)
    interior_sites = np.asarray(fixture.decomposition.interior_site_indices, dtype=np.int64)
    t_site = t_all[interior_sites]
    x_site = x_all[interior_sites]
    src_global = int(np.ravel_multi_index(tuple(int(v) % n for v, n in zip(source, shp)), shp))
    src_local = None if bool(source_average) else fixture.decomposition.interior_local_site(src_global)

    corr = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    for b in range(int(fixture.ginv_interior.shape[0])):
        g4 = np.asarray(fixture.ginv_interior[b]).reshape(
            int(interior_sites.size), int(fixture.nsc), int(interior_sites.size), int(fixture.nsc)
        )
        pair_vals = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        corr = corr + _manual_pair_correlator(
            pair_vals,
            t_site,
            x_site,
            lt,
            lmom,
            momenta,
            source_local=src_local,
            source_average=bool(source_average),
        )
    return corr


def _manual_eta_from_full_inverse(
    fixture: DDReferenceFixture,
    momenta: Sequence[int],
    *,
    source: Sequence[int],
    source_average: bool,
    momentum_axis: int = 0,
    n_flavor: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shp = tuple(int(v) for v in fixture.lattice_shape)
    t_all, x_all, lt, lmom = _site_arrays(shp, momentum_axis)
    interior_sites = np.asarray(fixture.decomposition.interior_site_indices, dtype=np.int64)
    t_site = t_all[interior_sites]
    x_site = x_all[interior_sites]
    src_global = int(np.ravel_multi_index(tuple(int(v) % n for v, n in zip(source, shp)), shp))
    src_local = None if bool(source_average) else fixture.decomposition.interior_local_site(src_global)

    g5 = np.asarray(gamma5(np.asarray(fixture.theory.gamma)), dtype=np.asarray(fixture.theory.gamma).dtype)
    g5c = np.kron(g5, np.eye(1, dtype=np.asarray(fixture.theory.gamma).dtype))

    conn = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    disc = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    for b in range(int(fixture.ginv_interior.shape[0])):
        g4 = np.asarray(fixture.ginv_interior[b]).reshape(
            int(interior_sites.size), int(fixture.nsc), int(interior_sites.size), int(fixture.nsc)
        )
        pair_conn = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        conn = conn + _manual_pair_correlator(
            pair_conn,
            t_site,
            x_site,
            lt,
            lmom,
            momenta,
            source_local=src_local,
            source_average=bool(source_average),
        )

        diag_blocks = g4[np.arange(int(interior_sites.size)), :, np.arange(int(interior_sites.size)), :]
        loops = np.einsum("ab,xba->x", g5c, diag_blocks, optimize=True)
        pair_disc = np.outer(loops, np.conjugate(loops))
        disc = disc + _manual_pair_correlator(
            pair_disc,
            t_site,
            x_site,
            lt,
            lmom,
            momenta,
            source_local=src_local,
            source_average=bool(source_average),
        )
    full = conn - float(int(n_flavor)) * disc
    return conn, disc, full


def _build_reference_fixture(
    shape: Sequence[int],
    *,
    boundary_slices: Sequence[int],
    boundary_width: int,
    batch_size: int,
    seed: int,
    hot_scale: float,
    mass: float = 0.0,
) -> DDReferenceFixture:
    theory = _make_theory(shape, batch_size=int(batch_size), seed=int(seed), mass=float(mass))
    q = theory.hot_start(scale=float(hot_scale))
    mctx = MeasurementContext()
    ginv, _ = _full_dense_inverse(q=q, theory=theory, context=mctx, dense_max_dof=4096)
    decomp = TimeSlabDecomposition(
        lattice_shape=tuple(int(v) for v in shape),
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
    )
    nsc = int(theory.fermion_shape()[-2] * theory.fermion_shape()[-1])
    return DDReferenceFixture(
        theory=theory,
        q=q,
        lattice_shape=tuple(int(v) for v in shape),
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
        ginv_interior=_extract_interior_block(ginv, decomp, nsc),
        decomposition=decomp,
        nsc=nsc,
    )


def _run_measurements(fixture: DDReferenceFixture, specs: List[Mapping[str, Any]]) -> Dict[str, Mapping[str, float]]:
    recs = run_inline_measurements(
        build_inline_measurements(specs),
        step=0,
        q=fixture.q,
        theory=fixture.theory,
        context=MeasurementContext(),
    )
    return {str(r["name"]): dict(r["values"]) for r in recs}


def _test_geometry_partition() -> str:
    decomp = TimeSlabDecomposition(lattice_shape=(6, 8), boundary_slices=(1, 4), boundary_width=1)
    all_sites = set(int(v) for v in decomp.boundary_site_indices) | set(int(v) for v in decomp.interior_site_indices)
    if len(all_sites) != int(decomp.vol):
        raise AssertionError("Boundary/interior partition does not cover the full lattice")
    if set(int(v) for v in decomp.boundary_site_indices) & set(int(v) for v in decomp.interior_site_indices):
        raise AssertionError("Boundary/interior partition is not disjoint")
    if int(decomp.boundary_component_indices(2).size) != 2 * int(decomp.boundary_site_indices.size):
        raise AssertionError("Boundary component indexing has the wrong size")
    return (
        f"vol={int(decomp.vol)} boundary_sites={int(decomp.boundary_site_indices.size)} "
        f"interior_sites={int(decomp.interior_site_indices.size)}"
    )


def _test_geometry_invalid_cases() -> str:
    failures = 0
    for kwargs in (
        {"lattice_shape": (8, 8), "boundary_slices": (1, 2), "boundary_width": 2},
        {"lattice_shape": (8, 8), "boundary_slices": (0, 4), "boundary_width": 4},
        {"lattice_shape": (8, 8), "boundary_slices": (1, 4), "boundary_width": 0},
    ):
        try:
            TimeSlabDecomposition(**kwargs)
        except ValueError:
            failures += 1
    if failures != 3:
        raise AssertionError(f"Expected 3 invalid geometry failures, got {failures}")
    return "overlap/full-cover/zero-width errors raised as expected"


def _test_geometry_wrap_components() -> str:
    decomp = TimeSlabDecomposition(lattice_shape=(4, 6), boundary_slices=(1, 3), boundary_width=1)
    got = {frozenset(int(v) for v in comp) for comp in decomp.interior_time_components()}
    want = {frozenset((2,)), frozenset((4, 5, 0))}
    if got != want:
        raise AssertionError(f"Unexpected periodic interior components: got={got}, want={want}")
    site_sizes = sorted(int(comp.size) for comp in decomp.interior_site_components())
    if site_sizes != [4, 12]:
        raise AssertionError(f"Unexpected periodic interior site-component sizes: got={site_sizes}, want=[4, 12]")
    return "periodic wrap-around components are grouped correctly"


def _test_schur_identity(
    shape: Sequence[int],
    *,
    boundary_slices: Sequence[int],
    boundary_width: int,
    batch_size: int,
    seed: int,
    mass: float,
    cold: bool,
    hot_scale: float,
    tol: float,
) -> str:
    theory = _make_theory(shape, batch_size=int(batch_size), seed=int(seed), mass=float(mass))
    q = _cold_cfg(theory) if bool(cold) else theory.hot_start(scale=float(hot_scale))
    mctx = MeasurementContext()
    ginv, _ = _full_dense_inverse(q=q, theory=theory, context=mctx, dense_max_dof=4096)
    sinv, decomp, _ = _dd_exact_schur_inverse(
        q=q,
        theory=theory,
        context=MeasurementContext(),
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
        dense_max_dof=4096,
    )
    nsc = int(theory.fermion_shape()[-2] * theory.fermion_shape()[-1])
    ref = _extract_interior_block(ginv, decomp, nsc)
    return _assert_close("schur_identity", sinv, ref, atol=float(tol), rtol=float(tol))


def _test_pion_measurements(fixture: DDReferenceFixture, *, tol: float) -> str:
    lt = int(fixture.lattice_shape[-1])
    moms = (0, 1)
    source = (0, 0)
    point_specs = [
        {
            "type": "dd_pion_2pt",
            "name": "pion_point",
            "source": list(source),
            "momenta": list(moms),
            "source_average": False,
            "boundary_slices": list(fixture.boundary_slices),
            "boundary_width": int(fixture.boundary_width),
        }
    ]
    avg_specs = [
        {
            "type": "pion_2pt_dd",
            "name": "pion_avg",
            "momenta": list(moms),
            "source_average": True,
            "boundary_slices": list(fixture.boundary_slices),
            "boundary_width": int(fixture.boundary_width),
        }
    ]
    point_vals = _run_measurements(fixture, point_specs)["pion_point"]
    avg_vals = _run_measurements(fixture, avg_specs)["pion_avg"]
    point_ref = _manual_pion_from_full_inverse(fixture, moms, source=source, source_average=False)
    avg_ref = _manual_pion_from_full_inverse(fixture, moms, source=source, source_average=True)
    d1 = _assert_close("pion_point", _extract_corr(point_vals, "c", moms, lt), point_ref, atol=tol, rtol=tol)
    d2 = _assert_close("pion_avg", _extract_corr(avg_vals, "c", moms, lt), avg_ref, atol=tol, rtol=tol)
    return f"{d1}; {d2}"


def _test_eta_measurements(fixture: DDReferenceFixture, *, tol: float) -> str:
    lt = int(fixture.lattice_shape[-1])
    moms = (0, 1)
    source = (0, 0)
    point_specs = [
        {
            "type": "dd_eta_2pt",
            "name": "eta_point",
            "source": list(source),
            "momenta": list(moms),
            "source_average": False,
            "boundary_slices": list(fixture.boundary_slices),
            "boundary_width": int(fixture.boundary_width),
            "n_flavor": 2,
        }
    ]
    avg_specs = [
        {
            "type": "eta_2pt_dd",
            "name": "eta_avg",
            "momenta": list(moms),
            "source_average": True,
            "boundary_slices": list(fixture.boundary_slices),
            "boundary_width": int(fixture.boundary_width),
            "n_flavor": 2,
        }
    ]
    point_vals = _run_measurements(fixture, point_specs)["eta_point"]
    avg_vals = _run_measurements(fixture, avg_specs)["eta_avg"]
    point_conn, point_disc, point_full = _manual_eta_from_full_inverse(
        fixture, moms, source=source, source_average=False, n_flavor=2
    )
    avg_conn, avg_disc, avg_full = _manual_eta_from_full_inverse(
        fixture, moms, source=source, source_average=True, n_flavor=2
    )
    d1 = _assert_close("eta_point_conn", _extract_corr(point_vals, "conn", moms, lt), point_conn, atol=tol, rtol=tol)
    d2 = _assert_close("eta_point_disc", _extract_corr(point_vals, "disc", moms, lt), point_disc, atol=tol, rtol=tol)
    d3 = _assert_close("eta_point_full", _extract_corr(point_vals, "full", moms, lt), point_full, atol=tol, rtol=tol)
    d4 = _assert_close("eta_avg_conn", _extract_corr(avg_vals, "conn", moms, lt), avg_conn, atol=tol, rtol=tol)
    d5 = _assert_close("eta_avg_disc", _extract_corr(avg_vals, "disc", moms, lt), avg_disc, atol=tol, rtol=tol)
    d6 = _assert_close("eta_avg_full", _extract_corr(avg_vals, "full", moms, lt), avg_full, atol=tol, rtol=tol)
    return "; ".join([d1, d2, d3, d4, d5, d6])


def _test_eta_output_modes(fixture: DDReferenceFixture) -> str:
    vals = _run_measurements(
        fixture,
        [
            {
                "type": "eta_2pt_dd",
                "name": "eta_disc_only",
                "momenta": [0],
                "source_average": True,
                "include_connected": False,
                "include_full": True,
                "boundary_slices": list(fixture.boundary_slices),
                "boundary_width": int(fixture.boundary_width),
            }
        ],
    )["eta_disc_only"]
    if any(k.startswith("conn_") for k in vals):
        raise AssertionError("conn_* keys should be absent when include_connected=False")
    if any(k.startswith("full_") for k in vals):
        raise AssertionError("full_* keys should be absent when include_connected=False")
    if not any(k.startswith("disc_") for k in vals):
        raise AssertionError("disc_* keys are missing")
    return "disc-only output mode behaves as expected"


def _test_dd_cache_hit(fixture: DDReferenceFixture) -> str:
    recs = run_inline_measurements(
        build_inline_measurements(
            [
                {
                    "type": "pion_2pt_dd",
                    "name": "pion_dd",
                    "source_average": True,
                    "boundary_slices": list(fixture.boundary_slices),
                    "boundary_width": int(fixture.boundary_width),
                },
                {
                    "type": "eta_2pt_dd",
                    "name": "eta_dd",
                    "source_average": True,
                    "boundary_slices": list(fixture.boundary_slices),
                    "boundary_width": int(fixture.boundary_width),
                },
            ]
        ),
        step=0,
        q=fixture.q,
        theory=fixture.theory,
        context=MeasurementContext(),
    )
    if len(recs) != 2:
        raise AssertionError(f"Expected 2 DD records, got {len(recs)}")
    hit0 = float(recs[0]["values"].get("dd_cache_hit", -1.0))
    hit1 = float(recs[1]["values"].get("dd_cache_hit", -1.0))
    if hit0 != 0.0 or hit1 != 1.0:
        raise AssertionError(f"Expected dd_cache_hit sequence 0,1 but got {hit0},{hit1}")
    return f"dd_cache_hit sequence {hit0:.0f},{hit1:.0f}"


def _test_expected_errors(fixture: DDReferenceFixture) -> str:
    failures = 0
    try:
        run_inline_measurements(
            build_inline_measurements(
                [
                    {
                        "type": "pion_2pt_dd",
                        "name": "bad_source",
                        "source": [0, int(fixture.boundary_slices[0])],
                        "source_average": False,
                        "boundary_slices": list(fixture.boundary_slices),
                        "boundary_width": int(fixture.boundary_width),
                    }
                ]
            ),
            step=0,
            q=fixture.q,
            theory=fixture.theory,
            context=MeasurementContext(),
        )
    except ValueError:
        failures += 1

    try:
        run_inline_measurements(
            build_inline_measurements(
                [
                    {
                        "type": "eta_2pt_dd",
                        "name": "small_dense_limit",
                        "source_average": True,
                        "dense_max_dof": 1,
                        "boundary_slices": list(fixture.boundary_slices),
                        "boundary_width": int(fixture.boundary_width),
                    }
                ]
            ),
            step=0,
            q=fixture.q,
            theory=fixture.theory,
            context=MeasurementContext(),
        )
    except ValueError:
        failures += 1

    if failures != 2:
        raise AssertionError(f"Expected 2 error-path checks to fail, got {failures}")
    return "boundary-source and dense-max-dof errors raised as expected"


def _test_iterative_pion_backend() -> str:
    q = _make_theory((4, 4), batch_size=1, seed=7, mass=0.1, solver_kind="cg").hot_start(scale=0.04)
    specs_dense = [
        {
            "type": "pion_2pt",
            "name": "pion_dense",
            "source": [0, 0],
            "momenta": [0, 1],
            "source_average": False,
            "propagator_backend": "dense",
            "dense_max_dof": 256,
        }
    ]
    specs_iter = [
        {
            "type": "pion_2pt",
            "name": "pion_iter",
            "source": [0, 0],
            "momenta": [0, 1],
            "source_average": False,
            "propagator_backend": "iterative",
            "dense_max_dof": 256,
        }
    ]
    dense_theory = _make_theory((4, 4), batch_size=1, seed=7, mass=0.1, solver_kind="gmres")
    vals_dense = run_inline_measurements(
        build_inline_measurements(specs_dense),
        step=0,
        q=q,
        theory=dense_theory,
        context=MeasurementContext(),
    )[0]["values"]
    gmres_theory = _make_theory((4, 4), batch_size=1, seed=7, mass=0.1, solver_kind="gmres")
    vals_gmres = run_inline_measurements(
        build_inline_measurements(specs_iter),
        step=0,
        q=q,
        theory=gmres_theory,
        context=MeasurementContext(),
    )[0]["values"]
    cg_theory = _make_theory((4, 4), batch_size=1, seed=7, mass=0.1, solver_kind="cg")
    vals_cg = run_inline_measurements(
        build_inline_measurements(specs_iter),
        step=0,
        q=q,
        theory=cg_theory,
        context=MeasurementContext(),
    )[0]["values"]
    corr_dense = _extract_corr(vals_dense, "c", (0, 1), 4)
    corr_gmres = _extract_corr(vals_gmres, "c", (0, 1), 4)
    corr_cg = _extract_corr(vals_cg, "c", (0, 1), 4)
    d1 = _assert_close("iterative_pion_gmres_vs_dense", corr_gmres, corr_dense, atol=1e-5, rtol=1e-5)
    d2 = _assert_close("iterative_pion_cg_vs_dense", corr_cg, corr_dense, atol=1e-5, rtol=1e-5)
    if float(vals_cg.get("inv_used_normal_equations", -1.0)) != 1.0:
        raise AssertionError("Expected iterative CG correlator path to report inv_used_normal_equations=1")
    return f"{d1}; {d2}"


def _run_test(name: str, fn: Callable[[], str], results: List[TestResult]) -> None:
    t0 = time.perf_counter()
    try:
        detail = str(fn())
        results.append(TestResult(name=name, passed=True, detail=detail, wall_sec=float(time.perf_counter() - t0)))
    except Exception as exc:
        results.append(
            TestResult(
                name=name,
                passed=False,
                detail=f"{type(exc).__name__}: {exc}",
                wall_sec=float(time.perf_counter() - t0),
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-1 DD validation suite")
    ap.add_argument("--schur-tol", type=float, default=5e-6)
    ap.add_argument("--meas-tol", type=float, default=1e-5)
    args = ap.parse_args()

    results: List[TestResult] = []
    fixture = _build_reference_fixture(
        (8, 8),
        boundary_slices=(2, 6),
        boundary_width=1,
        batch_size=1,
        seed=0,
        hot_scale=0.05,
        mass=0.0,
    )

    _run_test("geometry_partition", _test_geometry_partition, results)
    _run_test("geometry_invalid_cases", _test_geometry_invalid_cases, results)
    _run_test("geometry_wrap_components", _test_geometry_wrap_components, results)
    _run_test(
        "schur_identity_hot_8x8_bw1",
        lambda: _test_schur_identity(
            (8, 8),
            boundary_slices=(2, 6),
            boundary_width=1,
            batch_size=1,
            seed=0,
            mass=0.0,
            cold=False,
            hot_scale=0.05,
            tol=float(args.schur_tol),
        ),
        results,
    )
    _run_test(
        "schur_identity_hot_8x8_bw2",
        lambda: _test_schur_identity(
            (8, 8),
            boundary_slices=(1, 4),
            boundary_width=2,
            batch_size=1,
            seed=1,
            mass=0.0,
            cold=False,
            hot_scale=0.05,
            tol=float(args.schur_tol),
        ),
        results,
    )
    _run_test(
        "schur_identity_cold_massive_8x8_bw1",
        lambda: _test_schur_identity(
            (8, 8),
            boundary_slices=(2, 6),
            boundary_width=1,
            batch_size=1,
            seed=0,
            mass=0.2,
            cold=True,
            hot_scale=0.05,
            tol=float(args.schur_tol),
        ),
        results,
    )
    _run_test(
        "schur_identity_hot_batch2_6x6",
        lambda: _test_schur_identity(
            (6, 6),
            boundary_slices=(1, 4),
            boundary_width=1,
            batch_size=2,
            seed=3,
            mass=0.0,
            cold=False,
            hot_scale=0.04,
            tol=max(float(args.schur_tol), 5e-5),
        ),
        results,
    )
    _run_test("pion_measurements", lambda: _test_pion_measurements(fixture, tol=float(args.meas_tol)), results)
    _run_test("iterative_pion_backend", _test_iterative_pion_backend, results)
    _run_test("eta_measurements", lambda: _test_eta_measurements(fixture, tol=float(args.meas_tol)), results)
    _run_test("eta_output_modes", lambda: _test_eta_output_modes(fixture), results)
    _run_test("dd_cache_hit", lambda: _test_dd_cache_hit(fixture), results)
    _run_test("expected_errors", lambda: _test_expected_errors(fixture), results)

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

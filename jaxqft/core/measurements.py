"""Inline measurement framework for production MCMC drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import jax
import numpy as np

from jaxqft.fermions import gamma5

from .domain_decomposition import TimeSlabDecomposition
from .integrators import force_gradient, leapfrog, minnorm2, minnorm4pf4
from .multilevel_quenched import (
    build_giusti_surface_projector_basis,
    build_projector_basis,
    build_two_level_pion_geometry,
    compute_factorized_pion_blocks,
    compute_giusti_asymmetric_pion_blocks,
    factorized_pion_corr_from_blocks,
)
from .update import HMC, SMD


@dataclass
class MeasurementContext:
    """Mutable shared state passed through inline measurements.

    `cache` is intended for data hand-off between measurements in one step,
    while `state` can hold cross-step persistent helper data.
    """

    cache: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InlineMeasurement(Protocol):
    name: str
    every: int

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        """Run measurement and return a flat mapping of scalar values."""


@dataclass
class PlaquetteMeasurement:
    """Compute the average plaquette at the current gauge field."""

    every: int = 1
    name: str = "plaquette"

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        _ = context
        if not hasattr(theory, "average_plaquette"):
            raise AttributeError("Theory does not provide average_plaquette(q)")
        p = np.asarray(theory.average_plaquette(q), dtype=np.float64)
        return {"value": float(np.mean(p))}


def _coerce_source(source: Optional[Sequence[int]], lattice_shape: Sequence[int]) -> Tuple[int, ...]:
    shp = tuple(int(v) for v in lattice_shape)
    if not shp:
        raise ValueError("Cannot infer source coordinates: empty lattice shape")
    if source is None:
        return tuple([0] * len(shp))
    vals = tuple(int(v) for v in source)
    if len(vals) != len(shp):
        raise ValueError(
            f"Source coordinate rank mismatch: got {len(vals)} values for lattice ndim={len(shp)}"
        )
    return tuple(int(v % n) for v, n in zip(vals, shp))


def _flatten_corr(prefix: str, corr: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, v in enumerate(np.asarray(corr, dtype=np.float64).reshape(-1)):
        out[f"{prefix}_t{int(t)}"] = float(v)
    return out


def _parse_bool_from_spec(spec: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in spec:
        return bool(default)
    v = spec.get(key)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        txt = v.strip().lower()
        if txt in ("1", "true", "yes", "on"):
            return True
        if txt in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"Invalid boolean value for measurement key '{key}': {v!r}")


def _parse_bool_from_spec_aliases(spec: Mapping[str, Any], keys: Sequence[str], default: bool) -> bool:
    for key in keys:
        if key in spec:
            return _parse_bool_from_spec(spec, key, default)
    return bool(default)


def _extract_lattice_shape_from_theory(theory) -> Tuple[int, ...]:
    if hasattr(theory, "lattice_shape"):
        shp = tuple(int(v) for v in tuple(theory.lattice_shape))
        if shp:
            return shp
    fshp = tuple(int(v) for v in tuple(theory.fermion_shape()))
    if len(fshp) < 4:
        raise ValueError(f"Unexpected fermion shape rank: {fshp}")
    return tuple(fshp[1:-2])


def _fermion_structure(theory) -> Tuple[Tuple[int, ...], int, int, int, int, int]:
    lattice_shape = _extract_lattice_shape_from_theory(theory)
    fshp = tuple(int(v) for v in tuple(theory.fermion_shape()))
    if len(fshp) < 4:
        raise ValueError(f"Unexpected fermion shape rank: {fshp}")
    bs = int(fshp[0])
    ns = int(fshp[-2])
    nc = int(fshp[-1])
    vol = int(np.prod(np.asarray(lattice_shape, dtype=np.int64)))
    nsc = int(ns * nc)
    ndof = int(vol * nsc)
    return lattice_shape, bs, ns, nc, vol, ndof


def _flat_site_index(coords: Sequence[int], lattice_shape: Sequence[int]) -> int:
    shp = tuple(int(v) for v in lattice_shape)
    c = tuple(int(v) % n for v, n in zip(coords, shp))
    return int(np.ravel_multi_index(c, shp))


def _full_dense_inverse(
    q,
    theory,
    context: MeasurementContext,
    dense_max_dof: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not hasattr(theory, "apply_D"):
        raise AttributeError("Theory does not provide apply_D(U, psi); dense inversion is unavailable")
    if not hasattr(theory, "fermion_shape"):
        raise AttributeError("Theory does not provide fermion_shape(); dense inversion is unavailable")

    lattice_shape, bs, ns, nc, vol, ndof = _fermion_structure(theory)
    _ = vol
    _ = ns
    _ = nc
    max_dof = int(dense_max_dof)
    if max_dof > 0 and int(ndof) > int(max_dof):
        raise ValueError(
            f"Dense inversion blocked: ndof={ndof} exceeds dense_max_dof={max_dof}. "
            "Increase dense_max_dof explicitly if intended."
        )

    cache_key = f"dense_inv:maxdof={max_dof}"
    cached = context.cache.get(cache_key)
    if isinstance(cached, dict) and ("ginv" in cached) and ("timing" in cached):
        info = dict(cached["timing"])
        info["inv_cache_hit"] = 1.0
        info["inv_solve_total_sec_this_call"] = 0.0
        return np.asarray(cached["ginv"]), info

    ferm_shape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    fdtype = np.result_type(np.asarray(q).dtype, np.complex64)

    t0 = time.perf_counter()
    dmat = np.zeros((bs, ndof, ndof), dtype=fdtype)
    for i in range(ndof):
        rhs = np.zeros(ferm_shape, dtype=fdtype)
        rhs.reshape(bs, ndof)[:, i] = 1.0 + 0.0j
        col = np.asarray(theory.apply_D(q, jax.device_put(rhs)), dtype=fdtype).reshape(bs, ndof)
        dmat[:, :, i] = col
    t1 = time.perf_counter()

    ginv = np.zeros_like(dmat)
    for b in range(bs):
        ginv[b] = np.linalg.inv(dmat[b])
    t2 = time.perf_counter()

    build_sec = float(t1 - t0)
    invert_sec = float(t2 - t1)
    total_sec = float(t2 - t0)
    timing = {
        "inv_backend_dense": 1.0,
        "inv_dense_ndof": float(ndof),
        "inv_dense_build_sec": build_sec,
        "inv_dense_invert_sec": invert_sec,
        "inv_dense_total_sec": total_sec,
        "inv_n_solves": 1.0,
        "inv_solve_total_sec_step": total_sec,
        "inv_solve_mean_sec": total_sec,
        "inv_solve_min_sec": total_sec,
        "inv_solve_max_sec": total_sec,
        "inv_prop_build_wall_sec": total_sec,
    }
    context.cache[cache_key] = {"ginv": ginv, "timing": timing}
    info = dict(timing)
    info["inv_cache_hit"] = 0.0
    info["inv_solve_total_sec_this_call"] = float(total_sec)
    return ginv, info


def _full_dense_dirac_matrix(
    q,
    theory,
    context: MeasurementContext,
    dense_max_dof: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not hasattr(theory, "apply_D"):
        raise AttributeError("Theory does not provide apply_D(U, psi); dense DD observables are unavailable")
    if not hasattr(theory, "fermion_shape"):
        raise AttributeError("Theory does not provide fermion_shape(); dense DD observables are unavailable")

    _, bs, ns, nc, _, ndof = _fermion_structure(theory)
    _ = ns
    _ = nc
    max_dof = int(dense_max_dof)
    if max_dof > 0 and int(ndof) > int(max_dof):
        raise ValueError(
            f"Dense inversion blocked: ndof={ndof} exceeds dense_max_dof={max_dof}. "
            "Increase dense_max_dof explicitly if intended."
        )

    cache_key = f"dense_dirac:maxdof={max_dof}"
    cached = context.cache.get(cache_key)
    if isinstance(cached, dict) and ("dmat" in cached) and ("timing" in cached):
        info = dict(cached["timing"])
        info["dirac_cache_hit"] = 1.0
        return np.asarray(cached["dmat"]), info

    ferm_shape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    fdtype = np.result_type(np.asarray(q).dtype, np.complex64)
    t0 = time.perf_counter()
    dmat = np.zeros((bs, ndof, ndof), dtype=fdtype)
    for i in range(ndof):
        rhs = np.zeros(ferm_shape, dtype=fdtype)
        rhs.reshape(bs, ndof)[:, i] = 1.0 + 0.0j
        col = np.asarray(theory.apply_D(q, jax.device_put(rhs)), dtype=fdtype).reshape(bs, ndof)
        dmat[:, :, i] = col
    t1 = time.perf_counter()

    timing = {
        "dirac_dense_ndof": float(ndof),
        "dirac_dense_build_sec": float(t1 - t0),
    }
    context.cache[cache_key] = {"dmat": dmat, "timing": timing}
    info = dict(timing)
    info["dirac_cache_hit"] = 0.0
    return dmat, info


def _point_source_propagator_from_dense_inverse(
    ginv: np.ndarray,
    lattice_shape: Sequence[int],
    ns: int,
    nc: int,
    source: Sequence[int],
) -> np.ndarray:
    bs = int(ginv.shape[0])
    nsc = int(ns * nc)
    site = _flat_site_index(source, lattice_shape)
    prop = np.zeros((bs, *tuple(int(v) for v in lattice_shape), ns, nc, ns, nc), dtype=ginv.dtype)
    for src_spin in range(ns):
        for src_color in range(nc):
            col = int(site * nsc + src_spin * nc + src_color)
            sol = np.asarray(ginv[:, :, col], dtype=ginv.dtype).reshape((bs, *tuple(lattice_shape), ns, nc))
            prop[..., src_spin, src_color] = sol
    return prop


def _full_point_source_propagator(
    q,
    theory,
    context: MeasurementContext,
    source: Sequence[int],
    *,
    backend: str = "iterative",
    dense_max_dof: int = 4096,
) -> Tuple[np.ndarray, Dict[str, float]]:
    backend_key = str(backend).strip().lower()
    if backend_key in ("iter", "cg", "point"):
        backend_key = "iterative"
    if backend_key in ("direct", "exact"):
        backend_key = "dense"
    if backend_key not in ("iterative", "dense"):
        raise ValueError(f"Unsupported propagator backend: {backend!r}")

    if not hasattr(theory, "fermion_shape"):
        raise AttributeError("Theory does not provide fermion_shape() for correlator measurements")

    lattice_shape = _extract_lattice_shape_from_theory(theory)
    src = _coerce_source(source, lattice_shape)
    cache_key = f"quark_prop:{backend_key}:{','.join(str(v) for v in src)}"
    cached = context.cache.get(cache_key)
    if isinstance(cached, dict) and ("prop" in cached) and ("timing" in cached):
        info = dict(cached["timing"])
        info["inv_cache_hit"] = 1.0
        info["inv_solve_total_sec_this_call"] = 0.0
        return np.asarray(cached["prop"]), info

    ferm_shape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    bs = int(ferm_shape[0])
    ns = int(ferm_shape[-2])
    nc = int(ferm_shape[-1])

    if backend_key == "dense":
        ginv, inv_info = _full_dense_inverse(q=q, theory=theory, context=context, dense_max_dof=int(dense_max_dof))
        prop = _point_source_propagator_from_dense_inverse(
            ginv=ginv,
            lattice_shape=lattice_shape,
            ns=ns,
            nc=nc,
            source=src,
        )
        timing = dict(inv_info)
    else:
        if not hasattr(theory, "solve_direct"):
            raise AttributeError("Theory does not provide solve_direct(U, rhs) for correlator measurements")
        solver_kind = str(getattr(theory, "solver_kind", "")).strip().lower()
        fdtype = np.dtype(getattr(q, "dtype", np.complex64))
        prop = np.zeros((bs, *lattice_shape, ns, nc, ns, nc), dtype=fdtype)
        solve_times: List[float] = []

        base_index = (slice(None), *src)
        t_build0 = time.perf_counter()
        for src_spin in range(ns):
            for src_color in range(nc):
                rhs = np.zeros(ferm_shape, dtype=fdtype)
                rhs[(*base_index, src_spin, src_color)] = 1.0 + 0.0j
                ts = time.perf_counter()
                rhs_j = jax.device_put(rhs)
                if solver_kind == "cg":
                    if not hasattr(theory, "solve_normal") or not hasattr(theory, "apply_Ddag"):
                        raise AttributeError(
                            "Theory with solver.kind='cg' must provide apply_Ddag(U, rhs) and solve_normal(U, phi) "
                            "for iterative correlator measurements"
                        )
                    rhs_normal = theory.apply_Ddag(q, rhs_j)
                    sol = np.asarray(theory.solve_normal(q, rhs_normal), dtype=fdtype)
                else:
                    sol = np.asarray(theory.solve_direct(q, rhs_j), dtype=fdtype)
                solve_times.append(float(time.perf_counter() - ts))
                prop[..., src_spin, src_color] = sol

        build_dt = float(time.perf_counter() - t_build0)
        st = np.asarray(solve_times, dtype=np.float64)
        timing = {
            "inv_backend_dense": 0.0,
            "inv_n_solves": float(st.size),
            "inv_solve_total_sec_step": float(np.sum(st)) if st.size else 0.0,
            "inv_solve_mean_sec": float(np.mean(st)) if st.size else 0.0,
            "inv_solve_min_sec": float(np.min(st)) if st.size else 0.0,
            "inv_solve_max_sec": float(np.max(st)) if st.size else 0.0,
            "inv_prop_build_wall_sec": float(build_dt),
            "inv_used_normal_equations": 1.0 if solver_kind == "cg" else 0.0,
        }
    context.cache[cache_key] = {"prop": prop, "timing": timing}
    info = dict(timing)
    if "inv_cache_hit" not in info:
        info["inv_cache_hit"] = 0.0
    if "inv_solve_total_sec_this_call" not in info:
        info["inv_solve_total_sec_this_call"] = float(timing["inv_solve_total_sec_step"])
    else:
        info["inv_solve_total_sec_this_call"] = float(info["inv_solve_total_sec_this_call"])
    return prop, info


def _pion_two_point_from_propagator(prop: np.ndarray, source_t: int) -> np.ndarray:
    nd = int(prop.ndim - 5)  # B + Nd lattice + sink(spin,color) + source(spin,color)
    if nd < 1:
        raise ValueError(f"Unexpected propagator rank for pion correlator: {prop.shape}")
    time_axis = 1 + (nd - 1)
    lt = int(prop.shape[time_axis])
    corr = np.zeros((lt,), dtype=np.float64)

    for t in range(lt):
        dt = int((t - int(source_t)) % lt)
        idx = [slice(None)] * prop.ndim
        idx[time_axis] = t
        slab = np.asarray(prop[tuple(idx)])
        # C_pi(dt) = sum_x tr[S(x,0) S^\dagger(x,0)] (positive-definite).
        corr[dt] = float(np.real(np.sum(np.conjugate(slab) * slab)))
    return corr


def _coerce_momenta(momenta: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if momenta is None:
        return (0,)
    vals = tuple(int(v) for v in momenta)
    if len(vals) == 0:
        return (0,)
    return vals


def _coerce_momentum_axis(momentum_axis: int, lattice_shape: Sequence[int]) -> int:
    nd = int(len(tuple(lattice_shape)))
    if nd <= 1:
        return 0
    ax = int(momentum_axis)
    if ax < 0:
        ax += (nd - 1)
    if ax < 0 or ax >= (nd - 1):
        raise ValueError(
            f"momentum_axis must be in [0,{nd-2}] for lattice ndim={nd} (time axis is fixed to last axis)"
        )
    return ax


def _flatten_corr_momentum(
    prefix: str,
    corr: np.ndarray,
    momenta: Sequence[int],
    *,
    include_legacy_zero: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    moms = tuple(int(v) for v in momenta)
    arr = np.asarray(corr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected momentum correlator rank 2 (Nmom, Lt), got shape {arr.shape}")
    if arr.shape[0] != len(moms):
        raise ValueError(f"Momentum list size mismatch: len(momenta)={len(moms)} vs corr shape {arr.shape}")

    for ip, p in enumerate(moms):
        row = np.asarray(arr[ip])
        for t, v in enumerate(row.reshape(-1)):
            out[f"{prefix}_p{int(p)}_t{int(t)}_re"] = float(np.real(v))
            out[f"{prefix}_p{int(p)}_t{int(t)}_im"] = float(np.imag(v))
        if include_legacy_zero and int(p) == 0:
            out.update(_flatten_corr(prefix, np.real(row)))
    return out


def _flatten_corr_momentum_masked(
    prefix: str,
    corr: np.ndarray,
    momenta: Sequence[int],
    valid_mask: Sequence[bool],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    moms = tuple(int(v) for v in momenta)
    arr = np.asarray(corr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected momentum correlator rank 2 (Nmom, Lt), got shape {arr.shape}")
    if arr.shape[0] != len(moms):
        raise ValueError(f"Momentum list size mismatch: len(momenta)={len(moms)} vs corr shape {arr.shape}")
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if arr.shape[1] != mask.size:
        raise ValueError(f"Valid-mask size mismatch: corr shape {arr.shape} vs mask size {mask.size}")
    for t, ok in enumerate(mask.tolist()):
        if ok:
            out[f"{prefix}_valid_t{int(t)}"] = 1.0
    for ip, p in enumerate(moms):
        row = np.asarray(arr[ip])
        for t, ok in enumerate(mask.tolist()):
            if not ok:
                continue
            v = row[t]
            out[f"{prefix}_p{int(p)}_t{int(t)}_re"] = float(np.real(v))
            out[f"{prefix}_p{int(p)}_t{int(t)}_im"] = float(np.imag(v))
    return out


def _flatten_matrix_corr_momentum(prefix: str, corr: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    arr = np.asarray(corr)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 matrix correlator array (Nmom, Nmom, Lt), got {arr.shape}")
    for i in range(int(arr.shape[0])):
        for j in range(int(arr.shape[1])):
            row = np.asarray(arr[i, j]).reshape(-1)
            for t, v in enumerate(row):
                out[f"{prefix}_i{int(i)}_j{int(j)}_t{int(t)}_re"] = float(np.real(v))
                out[f"{prefix}_i{int(i)}_j{int(j)}_t{int(t)}_im"] = float(np.imag(v))
    return out


def _coerce_source_times(source_times: Optional[Sequence[int]], lt: int) -> Tuple[int, ...]:
    if source_times is None:
        return (0,)
    vals = tuple(int(v) % int(lt) for v in source_times)
    if len(vals) == 0:
        return (0,)
    return vals


def _parse_source_times_from_spec(spec: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    raw = None
    for key in ("source_times", "source_time", "tsrc", "t_src"):
        if key in spec:
            raw = spec.get(key)
            break
    if raw is None:
        return None
    if isinstance(raw, str):
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if not toks:
            return None
        return tuple(int(t) for t in toks)
    if isinstance(raw, Sequence):
        vals = tuple(int(v) for v in raw)
        return vals if vals else None
    return (int(raw),)


def _timeslice_site_indices_1d_spatial(lattice_shape: Sequence[int], time_slice: int) -> Tuple[np.ndarray, np.ndarray]:
    shp = tuple(int(v) for v in lattice_shape)
    if len(shp) != 2:
        raise ValueError(f"Timeslice spectroscopy helper expects a 2D lattice (Lx, Lt), got {shp}")
    lx, lt = int(shp[0]), int(shp[1])
    t = int(time_slice) % lt
    xs = np.arange(lx, dtype=np.int64)
    tt = np.full((lx,), t, dtype=np.int64)
    sites = np.asarray(np.ravel_multi_index((xs, tt), shp), dtype=np.int64)
    return xs, sites


def _timeslice_propagator_from_dense_inverse_1d_spatial(
    ginv_b: np.ndarray,
    lattice_shape: Sequence[int],
    *,
    ns: int,
    nc: int,
    source_time: int,
    sink_time: int,
) -> np.ndarray:
    shp = tuple(int(v) for v in lattice_shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    nsc = int(ns * nc)
    g4 = np.asarray(ginv_b).reshape(vol, nsc, vol, nsc)
    _, sink_sites = _timeslice_site_indices_1d_spatial(shp, sink_time)
    _, source_sites = _timeslice_site_indices_1d_spatial(shp, source_time)
    block = g4[np.ix_(sink_sites, np.arange(nsc, dtype=np.int64), source_sites, np.arange(nsc, dtype=np.int64))]
    return np.asarray(np.transpose(block, (0, 2, 1, 3)))


def _two_pion_i2_matrix_from_timeslice_propagator(
    prop_t: np.ndarray,
    momenta: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    g = np.asarray(prop_t)
    if g.ndim != 4:
        raise ValueError(f"Expected rank-4 timeslice propagator (Lx, Lx, nsc, nsc), got {g.shape}")
    lx = int(g.shape[0])
    if int(g.shape[1]) != lx:
        raise ValueError(f"Expected square spatial propagator block, got {g.shape[:2]}")
    moms = tuple(int(v) for v in momenta)
    nmom = int(len(moms))

    xs = np.arange(lx, dtype=np.float64)
    sink_phase = np.exp(2j * np.pi * np.asarray(moms, dtype=np.float64)[:, None] * xs[None, :] / float(lx))
    source_phase = np.conjugate(sink_phase)

    pion_kernel = np.einsum("xyab,xyab->xy", g, np.conjugate(g), optimize=True)
    mixed = np.einsum("px,qy,xy->pq", sink_phase, source_phase, pion_kernel, optimize=True)
    direct = mixed * np.conjugate(mixed)

    sink_pair_phase = np.exp(
        2j
        * np.pi
        * np.asarray(moms, dtype=np.float64)[:, None, None]
        * (xs[None, :, None] - xs[None, None, :])
        / float(lx)
    )
    source_pair_phase = np.conjugate(sink_pair_phase)

    left = np.einsum("xyab,Xycb->xXyac", g, np.conjugate(g), optimize=True)
    right = np.einsum("XYcd,xYad->xXYca", g, np.conjugate(g), optimize=True)
    right = np.transpose(right, (0, 1, 2, 4, 3))
    exchange_kernel = np.einsum("xXyac,xXYac->xXyY", left, right, optimize=True)
    exchange = np.einsum("pxX,qyY,xXyY->pq", sink_pair_phase, source_pair_phase, exchange_kernel, optimize=True)

    if direct.shape != (nmom, nmom) or exchange.shape != (nmom, nmom):
        raise ValueError(
            f"Unexpected two-pion matrix shapes: direct={direct.shape}, exchange={exchange.shape}, expected {(nmom, nmom)}"
        )
    return np.asarray(direct), np.asarray(exchange)


def _build_md_integrator(name: str, theory, nmd: int, tau: float):
    key = str(name).strip().lower()
    if key == "minnorm2":
        return minnorm2(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "leapfrog":
        return leapfrog(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "forcegrad":
        return force_gradient(theory.force, theory.evolve_q, int(nmd), float(tau))
    if key == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolve_q, int(nmd), float(tau))
    raise ValueError(f"Unknown integrator: {name!r}")


def _pion_two_point_momentum_from_propagator(
    prop: np.ndarray,
    source: Sequence[int],
    momenta: Sequence[int],
    momentum_axis: int,
    *,
    average_pm: bool = False,
) -> np.ndarray:
    nd = int(prop.ndim - 5)
    if nd < 1:
        raise ValueError(f"Unexpected propagator rank for pion correlator: {prop.shape}")
    moms = tuple(int(v) for v in momenta)
    time_axis = 1 + (nd - 1)
    lt = int(prop.shape[time_axis])
    corr = np.zeros((len(moms), lt), dtype=np.complex128)

    source_t = int(source[-1])
    has_spatial = nd > 1
    if has_spatial:
        lattice_shape = tuple(int(v) for v in prop.shape[1 : 1 + nd])
        mom_ax = _coerce_momentum_axis(momentum_axis, lattice_shape)
        lmom = int(prop.shape[1 + mom_ax])
        x0 = int(source[mom_ax]) % lmom
        shp_spatial = tuple(int(v) for v in prop.shape[1 : 1 + (nd - 1)])

    for t in range(lt):
        dt = int((t - source_t) % lt)
        idx = [slice(None)] * prop.ndim
        idx[time_axis] = t
        slab = np.asarray(prop[tuple(idx)])
        vals = np.sum(np.conjugate(slab) * slab, axis=tuple(range(slab.ndim - 4, slab.ndim)))
        if not has_spatial:
            s = complex(np.sum(vals))
            for ip in range(len(moms)):
                corr[ip, dt] = corr[ip, dt] + s
            continue

        for ip, p in enumerate(moms):
            theta = (2.0 * np.pi * float(p) / float(lmom)) * (np.arange(lmom) - x0)
            phase = np.cos(theta) if bool(average_pm) else np.exp(1j * theta)
            ph_shape = [1] + [1] * len(shp_spatial)
            ph_shape[1 + mom_ax] = lmom
            ph = phase.reshape(ph_shape)
            corr[ip, dt] = corr[ip, dt] + complex(np.sum(vals * ph))
    return corr


def _site_time_momentum_arrays(lattice_shape: Sequence[int], momentum_axis: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    shp = tuple(int(v) for v in lattice_shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    coords = np.asarray(np.unravel_index(np.arange(vol, dtype=np.int64), shp), dtype=np.int64).T
    t = coords[:, -1].astype(np.int64)
    lt = int(shp[-1])
    if len(shp) <= 1:
        x = np.zeros((vol,), dtype=np.int64)
        lmom = 1
    else:
        ax = _coerce_momentum_axis(momentum_axis, shp)
        x = coords[:, ax].astype(np.int64)
        lmom = int(shp[ax])
    return t, x, lt, lmom


def _accumulate_pair_correlator(
    pair_vals: np.ndarray,
    t_site: np.ndarray,
    x_site: np.ndarray,
    lt: int,
    lmom: int,
    momenta: Sequence[int],
    *,
    source_site: Optional[int],
    source_average: bool,
    average_pm: bool = False,
) -> np.ndarray:
    moms = tuple(int(v) for v in momenta)
    vol = int(pair_vals.shape[0])
    if pair_vals.shape != (vol, vol):
        raise ValueError(f"pair_vals must have shape (V,V), got {pair_vals.shape}")
    corr = np.zeros((len(moms), int(lt)), dtype=np.complex128)

    if source_average:
        for y in range(vol):
            dt = np.asarray((t_site - t_site[y]) % int(lt), dtype=np.int64)
            vals = np.asarray(pair_vals[:, y])
            if int(lmom) <= 1:
                for ip in range(len(moms)):
                    np.add.at(corr[ip], dt, vals)
                continue
            dx = np.asarray(x_site - x_site[y], dtype=np.float64)
            for ip, p in enumerate(moms):
                theta = (2.0 * np.pi * float(p) / float(lmom)) * dx
                ph = np.cos(theta) if bool(average_pm) else np.exp(1j * theta)
                np.add.at(corr[ip], dt, vals * ph)
        corr = corr / float(vol)
        return corr

    if source_site is None:
        raise ValueError("source_site is required when source_average=False")
    y0 = int(source_site) % vol
    dt = np.asarray((t_site - t_site[y0]) % int(lt), dtype=np.int64)
    vals = np.asarray(pair_vals[:, y0])
    if int(lmom) <= 1:
        for ip in range(len(moms)):
            np.add.at(corr[ip], dt, vals)
        return corr
    dx = np.asarray(x_site - x_site[y0], dtype=np.float64)
    for ip, p in enumerate(moms):
        theta = (2.0 * np.pi * float(p) / float(lmom)) * dx
        ph = np.cos(theta) if bool(average_pm) else np.exp(1j * theta)
        np.add.at(corr[ip], dt, vals * ph)
    return corr


def _pion_two_point_momentum_from_dense_inverse(
    ginv: np.ndarray,
    lattice_shape: Sequence[int],
    ns: int,
    nc: int,
    momenta: Sequence[int],
    momentum_axis: int,
    *,
    source_site: Optional[int],
    source_average: bool,
    average_pm: bool = False,
) -> np.ndarray:
    shp = tuple(int(v) for v in lattice_shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    nsc = int(ns * nc)
    t_site, x_site, lt, lmom = _site_time_momentum_arrays(shp, momentum_axis)

    corr = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    for b in range(int(ginv.shape[0])):
        g4 = np.asarray(ginv[b]).reshape(vol, nsc, vol, nsc)
        pair_vals = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        corr = corr + _accumulate_pair_correlator(
            pair_vals=pair_vals,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=source_site,
            source_average=bool(source_average),
            average_pm=bool(average_pm),
        )
    return corr


def _eta_two_point_components_from_dense_inverse(
    ginv: np.ndarray,
    lattice_shape: Sequence[int],
    ns: int,
    nc: int,
    gamma: np.ndarray,
    momenta: Sequence[int],
    momentum_axis: int,
    *,
    source_site: Optional[int],
    source_average: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    shp = tuple(int(v) for v in lattice_shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    nsc = int(ns * nc)
    t_site, x_site, lt, lmom = _site_time_momentum_arrays(shp, momentum_axis)

    gam = np.asarray(gamma)
    g5 = np.asarray(gamma5(gam), dtype=gam.dtype)
    g5c = np.kron(g5, np.eye(int(nc), dtype=gam.dtype))

    conn = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    disc = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)

    for b in range(int(ginv.shape[0])):
        g4 = np.asarray(ginv[b]).reshape(vol, nsc, vol, nsc)
        pair_conn = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        conn = conn + _accumulate_pair_correlator(
            pair_vals=pair_conn,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=source_site,
            source_average=bool(source_average),
        )

        diag_blocks = g4[np.arange(vol), :, np.arange(vol), :]
        loops = np.einsum("ab,xba->x", g5c, diag_blocks, optimize=True)
        pair_disc = np.outer(loops, np.conjugate(loops))
        disc = disc + _accumulate_pair_correlator(
            pair_vals=pair_disc,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=source_site,
            source_average=bool(source_average),
        )

    return conn, disc


def _dd_exact_schur_inverse(
    q,
    theory,
    context: MeasurementContext,
    *,
    boundary_slices: Sequence[int],
    boundary_width: int,
    dense_max_dof: int,
) -> Tuple[np.ndarray, TimeSlabDecomposition, Dict[str, float]]:
    lattice_shape, _, ns, nc, _, ndof = _fermion_structure(theory)
    nsc = int(ns * nc)
    max_dof = int(dense_max_dof)
    if max_dof > 0 and int(ndof) > int(max_dof):
        raise ValueError(
            f"Dense inversion blocked: ndof={ndof} exceeds dense_max_dof={max_dof}. "
            "Increase dense_max_dof explicitly if intended."
        )

    decomp = TimeSlabDecomposition(
        lattice_shape=tuple(int(v) for v in lattice_shape),
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
    )
    key_slices = ",".join(str(int(v)) for v in tuple(decomp.boundary_slices))
    cache_key = (
        f"dd_exact_schur:maxdof={max_dof}:bw={int(decomp.boundary_width)}:cuts={key_slices}"
    )
    cached = context.cache.get(cache_key)
    if isinstance(cached, dict) and ("sinv" in cached) and ("timing" in cached):
        info = dict(cached["timing"])
        info["dd_cache_hit"] = 1.0
        info["dd_total_sec_this_call"] = 0.0
        return np.asarray(cached["sinv"]), cached["decomp"], info

    dmat, dmat_info = _full_dense_dirac_matrix(
        q=q,
        theory=theory,
        context=context,
        dense_max_dof=int(dense_max_dof),
    )

    b_idx = decomp.boundary_component_indices(nsc)
    i_idx = decomp.interior_component_indices(nsc)
    nb = int(b_idx.size)
    ni = int(i_idx.size)

    t0 = time.perf_counter()
    sinv = np.zeros((int(dmat.shape[0]), ni, ni), dtype=dmat.dtype)
    boundary_invert_sec = 0.0
    schur_build_sec = 0.0
    schur_invert_sec = 0.0
    for b in range(int(dmat.shape[0])):
        dii = np.asarray(dmat[b][np.ix_(i_idx, i_idx)], dtype=dmat.dtype)
        if nb > 0:
            dbb = np.asarray(dmat[b][np.ix_(b_idx, b_idx)], dtype=dmat.dtype)
            dbi = np.asarray(dmat[b][np.ix_(b_idx, i_idx)], dtype=dmat.dtype)
            dib = np.asarray(dmat[b][np.ix_(i_idx, b_idx)], dtype=dmat.dtype)
            ts = time.perf_counter()
            dbb_inv = np.linalg.inv(dbb)
            boundary_invert_sec += float(time.perf_counter() - ts)
            ts = time.perf_counter()
            schur = dii - dib @ dbb_inv @ dbi
            schur_build_sec += float(time.perf_counter() - ts)
        else:
            schur = dii
        ts = time.perf_counter()
        sinv[b] = np.linalg.inv(schur)
        schur_invert_sec += float(time.perf_counter() - ts)
    t1 = time.perf_counter()

    build_sec = float(dmat_info.get("dirac_dense_build_sec", 0.0))
    total_sec = float(build_sec + boundary_invert_sec + schur_build_sec + schur_invert_sec)
    timing = {
        "dd_backend_exact_schur": 1.0,
        "dd_dense_ndof": float(ndof),
        "dd_n_boundary_sites": float(int(decomp.boundary_site_indices.size)),
        "dd_n_interior_sites": float(int(decomp.interior_site_indices.size)),
        "dd_n_boundary_slices": float(len(tuple(decomp.boundary_slices))),
        "dd_boundary_width": float(int(decomp.boundary_width)),
        "dd_dense_build_sec": build_sec,
        "dd_boundary_invert_sec": float(boundary_invert_sec),
        "dd_schur_build_sec": float(schur_build_sec),
        "dd_schur_invert_sec": float(schur_invert_sec),
        "dd_total_sec": total_sec,
        "dd_matrix_cache_hit": float(dmat_info.get("dirac_cache_hit", 0.0)),
    }
    context.cache[cache_key] = {"sinv": sinv, "decomp": decomp, "timing": timing}
    info = dict(timing)
    info["dd_cache_hit"] = 0.0
    info["dd_total_sec_this_call"] = float((t1 - t0) + (0.0 if bool(dmat_info.get("dirac_cache_hit", 0.0)) else build_sec))
    return sinv, decomp, info


def _pion_two_point_momentum_from_dd_exact_schur_inverse(
    sinv: np.ndarray,
    decomposition: TimeSlabDecomposition,
    lattice_shape: Sequence[int],
    ns: int,
    nc: int,
    momenta: Sequence[int],
    momentum_axis: int,
    *,
    source_site: Optional[int],
    source_average: bool,
    average_pm: bool = False,
) -> np.ndarray:
    shp = tuple(int(v) for v in lattice_shape)
    nsc = int(ns * nc)
    interior_sites = np.asarray(decomposition.interior_site_indices, dtype=np.int64)
    t_all, x_all, lt, lmom = _site_time_momentum_arrays(shp, momentum_axis)
    t_site = t_all[interior_sites]
    x_site = x_all[interior_sites]

    src_local = None
    if not bool(source_average):
        if source_site is None:
            raise ValueError("source_site is required when source_average=False")
        src_local = decomposition.interior_local_site(int(source_site))

    corr = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    for b in range(int(sinv.shape[0])):
        g4 = np.asarray(sinv[b]).reshape(int(interior_sites.size), nsc, int(interior_sites.size), nsc)
        pair_vals = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        corr = corr + _accumulate_pair_correlator(
            pair_vals=pair_vals,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=src_local,
            source_average=bool(source_average),
            average_pm=bool(average_pm),
        )
    return corr


def _eta_two_point_components_from_dd_exact_schur_inverse(
    sinv: np.ndarray,
    decomposition: TimeSlabDecomposition,
    lattice_shape: Sequence[int],
    ns: int,
    nc: int,
    gamma: np.ndarray,
    momenta: Sequence[int],
    momentum_axis: int,
    *,
    source_site: Optional[int],
    source_average: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    shp = tuple(int(v) for v in lattice_shape)
    nsc = int(ns * nc)
    interior_sites = np.asarray(decomposition.interior_site_indices, dtype=np.int64)
    t_all, x_all, lt, lmom = _site_time_momentum_arrays(shp, momentum_axis)
    t_site = t_all[interior_sites]
    x_site = x_all[interior_sites]

    src_local = None
    if not bool(source_average):
        if source_site is None:
            raise ValueError("source_site is required when source_average=False")
        src_local = decomposition.interior_local_site(int(source_site))

    gam = np.asarray(gamma)
    g5 = np.asarray(gamma5(gam), dtype=gam.dtype)
    g5c = np.kron(g5, np.eye(int(nc), dtype=gam.dtype))

    conn = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    disc = np.zeros((len(tuple(momenta)), lt), dtype=np.complex128)
    for b in range(int(sinv.shape[0])):
        g4 = np.asarray(sinv[b]).reshape(int(interior_sites.size), nsc, int(interior_sites.size), nsc)
        pair_conn = np.sum(np.abs(g4) ** 2, axis=(1, 3))
        conn = conn + _accumulate_pair_correlator(
            pair_vals=pair_conn,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=src_local,
            source_average=bool(source_average),
        )

        diag_blocks = g4[np.arange(int(interior_sites.size)), :, np.arange(int(interior_sites.size)), :]
        loops = np.einsum("ab,xba->x", g5c, diag_blocks, optimize=True)
        pair_disc = np.outer(loops, np.conjugate(loops))
        disc = disc + _accumulate_pair_correlator(
            pair_vals=pair_disc,
            t_site=t_site,
            x_site=x_site,
            lt=lt,
            lmom=lmom,
            momenta=momenta,
            source_site=src_local,
            source_average=bool(source_average),
        )
    return conn, disc


def _proton_two_point_from_propagator(prop: np.ndarray, source_t: int, parity_sign: int, gamma: np.ndarray) -> np.ndarray:
    nd = int(prop.ndim - 5)
    if nd != 4:
        raise ValueError(
            f"Proton correlator currently expects 4D lattices; got propagator shape {prop.shape}"
        )
    ns = int(prop.shape[-4])
    nc = int(prop.shape[-3])
    if ns != 4 or nc != 3:
        raise ValueError(
            "Proton correlator currently expects Ns=4 and Nc=3 "
            f"(got Ns={ns}, Nc={nc})"
        )

    gam = np.asarray(gamma)
    if gam.shape[0] < 4:
        raise ValueError(f"Need at least 4 gamma matrices for proton correlator, got shape {gam.shape}")
    g5 = np.asarray(gamma5(gam), dtype=gam.dtype)
    # Euclidean charge-conjugation choice for this gamma basis.
    cmat = gam[1] @ gam[3]
    cg5 = cmat @ g5
    gamma_t = gam[3]
    proj = 0.5 * (np.eye(ns, dtype=gam.dtype) + float(parity_sign) * gamma_t)

    eps = np.zeros((3, 3, 3), dtype=np.float64)
    eps[0, 1, 2] = 1.0
    eps[1, 2, 0] = 1.0
    eps[2, 0, 1] = 1.0
    eps[2, 1, 0] = -1.0
    eps[1, 0, 2] = -1.0
    eps[0, 2, 1] = -1.0

    time_axis = 1 + (nd - 1)
    lt = int(prop.shape[time_axis])
    corr = np.zeros((lt,), dtype=np.float64)
    for t in range(lt):
        dt = int((t - int(source_t)) % lt)
        idx = [slice(None)] * prop.ndim
        idx[time_axis] = t
        slab = np.asarray(prop[tuple(idx)])
        sites = slab.reshape((-1, ns, nc, ns, nc))

        direct = np.einsum(
            "abc,def,rs,uv,nraud,nsbve,npcqf->npq",
            eps,
            eps,
            cg5,
            cg5,
            sites,
            sites,
            sites,
            optimize=True,
        )
        exchange = np.einsum(
            "abc,def,rs,uv,nravd,nsbue,npcqf->npq",
            eps,
            eps,
            cg5,
            cg5,
            sites,
            sites,
            sites,
            optimize=True,
        )
        gmat = direct - exchange
        c_site = np.einsum("pq,npq->n", proj, gmat, optimize=True)
        corr[dt] = float(np.real(np.sum(c_site)))
    return corr


def _parse_source_from_spec(spec: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    if "source" not in spec:
        return None
    src = spec.get("source")
    if src is None:
        return None
    if isinstance(src, str):
        toks = [t.strip() for t in src.split(",") if t.strip()]
        if not toks:
            return None
        return tuple(int(t) for t in toks)
    if isinstance(src, Sequence):
        return tuple(int(v) for v in src)
    raise ValueError(f"Invalid source spec type: {type(src).__name__}")


def _parse_boundary_slices_from_spec(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = None
    for key in ("boundary_slices", "boundaries", "cuts", "boundary_times", "time_cuts"):
        if key in spec:
            raw = spec.get(key)
            break
    if raw is None:
        raise ValueError("DD measurement requires boundary_slices (or boundaries/cuts)")
    if isinstance(raw, str):
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if not toks:
            raise ValueError("DD measurement boundary_slices cannot be empty")
        return tuple(int(t) for t in toks)
    if isinstance(raw, Sequence):
        vals = tuple(int(v) for v in raw)
        if len(vals) == 0:
            raise ValueError("DD measurement boundary_slices cannot be empty")
        return vals
    return (int(raw),)


def _parse_momenta_from_spec(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    if "momenta" not in spec and "p" not in spec and "mom" not in spec:
        return (0,)
    raw = spec.get("momenta", spec.get("p", spec.get("mom")))
    if raw is None:
        return (0,)
    if isinstance(raw, str):
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if not toks:
            return (0,)
        return tuple(int(t) for t in toks)
    if isinstance(raw, Sequence):
        vals = [int(v) for v in raw]
        return tuple(vals) if vals else (0,)
    return (int(raw),)


def _parse_propagator_backend_from_spec(spec: Mapping[str, Any], default: str = "iterative") -> str:
    raw = str(spec.get("propagator_backend", spec.get("propagator", spec.get("inversion", default)))).strip().lower()
    if raw in ("iter", "cg", "point"):
        return "iterative"
    if raw in ("direct", "exact"):
        return "dense"
    if raw not in ("iterative", "dense", "auto"):
        raise ValueError(f"Unsupported propagator backend: {raw!r}")
    return raw


@dataclass
class PionTwoPointMeasurement:
    """Connected pseudoscalar two-point function with momentum projection."""

    every: int = 1
    name: str = "pion_2pt"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    propagator_backend: str = "iterative"  # iterative | dense | auto
    dense_max_dof: int = 4096
    source_average: bool = False
    average_pm: bool = False

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        source = _coerce_source(self.source, lattice_shape)
        moms = _coerce_momenta(self.momenta)
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        backend = str(self.propagator_backend).strip().lower()
        if backend in ("iter", "cg", "point"):
            backend = "iterative"
        if backend in ("direct", "exact"):
            backend = "dense"
        if backend == "auto":
            _, _, _, _, _, ndof = _fermion_structure(theory)
            backend = "dense" if int(ndof) <= int(self.dense_max_dof) else "iterative"
        if backend not in ("iterative", "dense"):
            raise ValueError(f"Unsupported propagator_backend: {self.propagator_backend!r}")
        if bool(self.source_average) and backend != "dense":
            raise ValueError("source_average=True requires propagator_backend=dense (or auto with dense selected)")

        if backend == "dense":
            ginv, inv_info = _full_dense_inverse(
                q=q,
                theory=theory,
                context=context,
                dense_max_dof=int(self.dense_max_dof),
            )
            t1 = time.perf_counter()
            source_site = None if bool(self.source_average) else _flat_site_index(source, lattice_shape)
            corr = _pion_two_point_momentum_from_dense_inverse(
                ginv=ginv,
                lattice_shape=lattice_shape,
                ns=int(theory.fermion_shape()[-2]),
                nc=int(theory.fermion_shape()[-1]),
                momenta=moms,
                momentum_axis=mom_axis,
                source_site=source_site,
                source_average=bool(self.source_average),
                average_pm=bool(self.average_pm),
            )
        else:
            prop, inv_info = _full_point_source_propagator(
                q=q,
                theory=theory,
                context=context,
                source=source,
                backend="iterative",
                dense_max_dof=int(self.dense_max_dof),
            )
            t1 = time.perf_counter()
            corr = _pion_two_point_momentum_from_propagator(
                prop=prop,
                source=source,
                momenta=moms,
                momentum_axis=mom_axis,
                average_pm=bool(self.average_pm),
            )
        t2 = time.perf_counter()
        out = _flatten_corr_momentum("c", corr, moms, include_legacy_zero=True)
        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
        out["average_pm"] = 1.0 if bool(self.average_pm) else 0.0
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


@dataclass
class EtaTwoPointMeasurement:
    """Flavor-singlet pseudoscalar channel with disconnected loops (dense inverse)."""

    every: int = 1
    name: str = "eta_2pt"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    dense_max_dof: int = 4096
    source_average: bool = True
    n_flavor: int = 2
    include_connected: bool = True
    include_full: bool = True

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        if not hasattr(theory, "gamma"):
            raise AttributeError("Theory does not expose gamma matrices required for eta correlator")
        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        source = _coerce_source(self.source, lattice_shape)
        moms = _coerce_momenta(self.momenta)
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        ginv, inv_info = _full_dense_inverse(
            q=q,
            theory=theory,
            context=context,
            dense_max_dof=int(self.dense_max_dof),
        )
        t1 = time.perf_counter()
        source_site = None if bool(self.source_average) else _flat_site_index(source, lattice_shape)
        conn, disc = _eta_two_point_components_from_dense_inverse(
            ginv=ginv,
            lattice_shape=lattice_shape,
            ns=int(theory.fermion_shape()[-2]),
            nc=int(theory.fermion_shape()[-1]),
            gamma=np.asarray(theory.gamma),
            momenta=moms,
            momentum_axis=mom_axis,
            source_site=source_site,
            source_average=bool(self.source_average),
        )
        t2 = time.perf_counter()

        out: Dict[str, float] = {}
        if bool(self.include_connected):
            out.update(_flatten_corr_momentum("conn", conn, moms, include_legacy_zero=False))
        out.update(_flatten_corr_momentum("disc", disc, moms, include_legacy_zero=False))
        if bool(self.include_full) and bool(self.include_connected):
            full = conn - float(int(self.n_flavor)) * disc
            out.update(_flatten_corr_momentum("full", full, moms, include_legacy_zero=True))

        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
        out["n_flavor"] = float(int(self.n_flavor))
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


@dataclass
class TwoPionI2MatrixMeasurement:
    """Two-pion I=2 correlator matrix for O_p = pi(p) pi(-p) on a 1D spatial lattice.

    This DD-independent measurement is intended for Schwinger-model spectroscopy
    on 2D lattices (Lx, Lt). It uses the dense all-to-all inverse and performs
    the spatial sums exactly on a chosen set of source time slices.
    """

    every: int = 1
    name: str = "pipi_i2_matrix"
    momenta: Tuple[int, ...] = (0, 1)
    momentum_axis: int = 0
    source_times: Optional[Tuple[int, ...]] = None
    dense_max_dof: int = 4096
    include_direct: bool = True
    include_exchange: bool = True
    include_full: bool = True

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        if len(tuple(lattice_shape)) != 2:
            raise ValueError(
                f"{self.name} currently supports only 2D lattices (Lx, Lt) for Schwinger spectroscopy; got {lattice_shape}"
            )
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        if int(mom_axis) != 0:
            raise ValueError(f"{self.name} currently requires mom_axis=0 on (Lx, Lt) lattices; got {mom_axis}")
        moms = _coerce_momenta(self.momenta)
        lt = int(tuple(lattice_shape)[-1])
        source_times = _coerce_source_times(self.source_times, lt)
        if len(source_times) == 0:
            raise ValueError(f"{self.name} requires at least one source time slice")

        ginv, inv_info = _full_dense_inverse(
            q=q,
            theory=theory,
            context=context,
            dense_max_dof=int(self.dense_max_dof),
        )
        t1 = time.perf_counter()

        nmom = int(len(moms))
        direct = np.zeros((nmom, nmom, lt), dtype=np.complex128)
        exchange = np.zeros_like(direct)
        ns = int(theory.fermion_shape()[-2])
        nc = int(theory.fermion_shape()[-1])
        bs = int(ginv.shape[0])

        for b in range(bs):
            ginv_b = np.asarray(ginv[b])
            for tsrc in source_times:
                for dt in range(lt):
                    tsnk = int((int(tsrc) + int(dt)) % lt)
                    prop_t = _timeslice_propagator_from_dense_inverse_1d_spatial(
                        ginv_b,
                        lattice_shape=lattice_shape,
                        ns=ns,
                        nc=nc,
                        source_time=int(tsrc),
                        sink_time=int(tsnk),
                    )
                    dmat, xmat = _two_pion_i2_matrix_from_timeslice_propagator(prop_t, moms)
                    direct[:, :, dt] += dmat
                    exchange[:, :, dt] += xmat

        norm = float(max(1, bs * len(source_times)))
        direct = direct / norm
        exchange = exchange / norm
        full = direct - exchange
        t2 = time.perf_counter()

        out: Dict[str, float] = {}
        for i, p in enumerate(moms):
            out[f"basis_p{i}"] = float(int(p))
        out["n_momenta"] = float(nmom)
        out["mom_axis"] = float(int(mom_axis))
        out["n_source_times"] = float(len(source_times))
        for i, tsrc in enumerate(source_times):
            out[f"source_time_{int(i)}"] = float(int(tsrc))
        if bool(self.include_direct):
            out.update(_flatten_matrix_corr_momentum("direct", direct))
        if bool(self.include_exchange):
            out.update(_flatten_matrix_corr_momentum("exchange", exchange))
        if bool(self.include_full):
            out.update(_flatten_matrix_corr_momentum("full", full))
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


@dataclass
class DDPionTwoPointMeasurement:
    """Connected pseudoscalar two-point function from the exact interior Schur complement."""

    every: int = 1
    name: str = "pion_2pt_dd"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    dense_max_dof: int = 4096
    source_average: bool = False
    average_pm: bool = False
    boundary_slices: Tuple[int, ...] = (0,)
    boundary_width: int = 1

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        source = _coerce_source(self.source, lattice_shape)
        moms = _coerce_momenta(self.momenta)
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        sinv, decomp, inv_info = _dd_exact_schur_inverse(
            q=q,
            theory=theory,
            context=context,
            boundary_slices=tuple(int(v) for v in self.boundary_slices),
            boundary_width=int(self.boundary_width),
            dense_max_dof=int(self.dense_max_dof),
        )
        t1 = time.perf_counter()
        source_site = None if bool(self.source_average) else _flat_site_index(source, lattice_shape)
        corr = _pion_two_point_momentum_from_dd_exact_schur_inverse(
            sinv=sinv,
            decomposition=decomp,
            lattice_shape=lattice_shape,
            ns=int(theory.fermion_shape()[-2]),
            nc=int(theory.fermion_shape()[-1]),
            momenta=moms,
            momentum_axis=mom_axis,
            source_site=source_site,
            source_average=bool(self.source_average),
            average_pm=bool(self.average_pm),
        )
        t2 = time.perf_counter()

        out = _flatten_corr_momentum("c", corr, moms, include_legacy_zero=True)
        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
        out["average_pm"] = 1.0 if bool(self.average_pm) else 0.0
        out["dd_n_boundary_slices"] = float(len(tuple(decomp.boundary_slices)))
        out["dd_boundary_width"] = float(int(decomp.boundary_width))
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


@dataclass
class DDEtaTwoPointMeasurement:
    """Flavor-singlet pseudoscalar channel from the exact interior Schur complement."""

    every: int = 1
    name: str = "eta_2pt_dd"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    dense_max_dof: int = 4096
    source_average: bool = True
    n_flavor: int = 2
    include_connected: bool = True
    include_full: bool = True
    boundary_slices: Tuple[int, ...] = (0,)
    boundary_width: int = 1

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        if not hasattr(theory, "gamma"):
            raise AttributeError("Theory does not expose gamma matrices required for eta correlator")
        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        source = _coerce_source(self.source, lattice_shape)
        moms = _coerce_momenta(self.momenta)
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        sinv, decomp, inv_info = _dd_exact_schur_inverse(
            q=q,
            theory=theory,
            context=context,
            boundary_slices=tuple(int(v) for v in self.boundary_slices),
            boundary_width=int(self.boundary_width),
            dense_max_dof=int(self.dense_max_dof),
        )
        t1 = time.perf_counter()
        source_site = None if bool(self.source_average) else _flat_site_index(source, lattice_shape)
        conn, disc = _eta_two_point_components_from_dd_exact_schur_inverse(
            sinv=sinv,
            decomposition=decomp,
            lattice_shape=lattice_shape,
            ns=int(theory.fermion_shape()[-2]),
            nc=int(theory.fermion_shape()[-1]),
            gamma=np.asarray(theory.gamma),
            momenta=moms,
            momentum_axis=mom_axis,
            source_site=source_site,
            source_average=bool(self.source_average),
        )
        t2 = time.perf_counter()

        out: Dict[str, float] = {}
        if bool(self.include_connected):
            out.update(_flatten_corr_momentum("conn", conn, moms, include_legacy_zero=False))
        out.update(_flatten_corr_momentum("disc", disc, moms, include_legacy_zero=False))
        if bool(self.include_full) and bool(self.include_connected):
            full = conn - float(int(self.n_flavor)) * disc
            out.update(_flatten_corr_momentum("full", full, moms, include_legacy_zero=True))

        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
        out["n_flavor"] = float(int(self.n_flavor))
        out["dd_n_boundary_slices"] = float(len(tuple(decomp.boundary_slices)))
        out["dd_boundary_width"] = float(int(decomp.boundary_width))
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


@dataclass
class DDTwoLevelFactorizedPionMeasurement:
    """Two-level quenched DD pion estimator with projector factorization and bias correction."""

    every: int = 1
    name: str = "pion_2pt_ml_dd"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    average_pm: bool = False
    boundary_slices: Tuple[int, ...] = (0,)
    boundary_width: int = 1
    source_margin: int = 1
    factorization_kind: str = "boundary_transfer"  # boundary_transfer | giusti
    projector_kind: str = "full"  # full | probe | laplace | svd
    projector_nvec: int = 0
    probe_stride: int = 2
    dense_max_domain_dof: int = 1024
    exact_backend: str = "iterative"
    exact_dense_max_dof: int = 4096
    level1_ncfg: int = 8
    level1_warmup: int = 2
    level1_skip: int = 2
    level1_update: str = "hmc"  # hmc | smd | ghmc
    level1_integrator: str = "minnorm2"
    level1_nmd: int = 4
    level1_tau: float = 0.5
    level1_gamma: float = 0.3
    level1_seed: int = 0

    def _build_level1_chain(self, dd_theory, *, seed: int):
        integ = _build_md_integrator(self.level1_integrator, dd_theory, int(self.level1_nmd), float(self.level1_tau))
        upd = str(self.level1_update).strip().lower()
        if upd == "hmc":
            chain = HMC(dd_theory, integ, verbose=False, seed=int(seed), use_fast_jit=False)

            def evolve(q_cur, nstep: int):
                return chain.evolve(q_cur, int(nstep))

            return chain, evolve
        if upd in ("smd", "ghmc"):
            chain = SMD(
                dd_theory,
                integ,
                gamma=float(self.level1_gamma),
                accept_reject=True,
                verbose=False,
                seed=int(seed),
                use_fast_jit=False,
            )

            def evolve(q_cur, nstep: int):
                return chain.evolve(q_cur, int(nstep), warmup=False)

            return chain, evolve
        raise ValueError(f"Unsupported level1_update for {self.name}: {self.level1_update!r}")

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        if bool(getattr(theory, "include_fermion_monomial", False)):
            raise ValueError(
                f"{self.name} expects a quenched gauge background (include_fermion_monomial=False), "
                "because the level-1 updates are pure-gauge conditional updates."
            )
        if not hasattr(theory, "gamma") or not hasattr(theory, "apply_D"):
            raise AttributeError(f"{self.name} requires a Wilson-fermion theory exposing gamma and apply_D")

        from jaxqft.models import U1TimeSlabDDTheory

        t0 = time.perf_counter()
        lattice_shape = _extract_lattice_shape_from_theory(theory)
        source = _coerce_source(self.source, lattice_shape)
        moms = _coerce_momenta(self.momenta)
        mom_axis = _coerce_momentum_axis(self.momentum_axis, lattice_shape)
        ns = int(theory.fermion_shape()[-2])
        nc = int(theory.fermion_shape()[-1])
        nsc = int(ns * nc)
        geom = build_two_level_pion_geometry(
            lattice_shape=lattice_shape,
            boundary_slices=tuple(int(v) for v in self.boundary_slices),
            boundary_width=int(self.boundary_width),
            source=source,
            nsc=nsc,
            source_margin=int(self.source_margin),
            momentum_axis=int(mom_axis),
        )
        fact_mode = str(self.factorization_kind).strip().lower()
        if fact_mode in ("boundary", "boundary_transfer", "transfer", "slab"):
            fact_mode = "boundary_transfer"
        elif fact_mode in ("giusti", "giusti_surface", "giusti_asym", "surface"):
            fact_mode = "giusti"
        else:
            raise ValueError(f"Unsupported factorization_kind for {self.name}: {self.factorization_kind!r}")

        # Level-0 factorized approximation and exact bias.
        if fact_mode == "boundary_transfer":
            phi = build_projector_basis(
                q=q,
                theory=theory,
                geometry=geom,
                ns=ns,
                nc=nc,
                kind=str(self.projector_kind),
                nvec=int(self.projector_nvec),
                probe_stride=int(self.probe_stride),
            )
            src0, sink0 = compute_factorized_pion_blocks(
                q=q,
                theory=theory,
                geometry=geom,
                phi_overlap=phi,
                dense_max_domain_dof=int(self.dense_max_domain_dof),
            )
            approx0_bs, valid_mask = factorized_pion_corr_from_blocks(
                source_blocks=src0,
                sink_blocks=sink0,
                geometry=geom,
                momenta=moms,
                average_pm=bool(self.average_pm),
            )
            approx0 = np.sum(np.asarray(approx0_bs), axis=0)
            n_projectors = int(phi.shape[1])
        else:
            phi_src = build_giusti_surface_projector_basis(
                q=q,
                theory=theory,
                geometry=geom,
                ns=ns,
                nc=nc,
                kind=str(self.projector_kind),
                nvec=int(self.projector_nvec),
                probe_stride=int(self.probe_stride),
                dressed_domain="source",
            )
            phi_sink = build_giusti_surface_projector_basis(
                q=q,
                theory=theory,
                geometry=geom,
                ns=ns,
                nc=nc,
                kind=str(self.projector_kind),
                nvec=int(self.projector_nvec),
                probe_stride=int(self.probe_stride),
                dressed_domain="sink",
            )
            src0_l, sink0_l = compute_giusti_asymmetric_pion_blocks(
                q=q,
                theory=theory,
                geometry=geom,
                phi_surface=phi_src,
                dense_max_domain_dof=int(self.dense_max_domain_dof),
                dressed_domain="source",
            )
            src0_r, sink0_r = compute_giusti_asymmetric_pion_blocks(
                q=q,
                theory=theory,
                geometry=geom,
                phi_surface=phi_sink,
                dense_max_domain_dof=int(self.dense_max_domain_dof),
                dressed_domain="sink",
            )
            approx0_l_bs, valid_mask = factorized_pion_corr_from_blocks(
                source_blocks=src0_l,
                sink_blocks=sink0_l,
                geometry=geom,
                momenta=moms,
                average_pm=bool(self.average_pm),
            )
            approx0_r_bs, valid_mask_r = factorized_pion_corr_from_blocks(
                source_blocks=src0_r,
                sink_blocks=sink0_r,
                geometry=geom,
                momenta=moms,
                average_pm=bool(self.average_pm),
            )
            if not np.array_equal(np.asarray(valid_mask, dtype=bool), np.asarray(valid_mask_r, dtype=bool)):
                raise RuntimeError("Internal Giusti valid-mask mismatch")
            approx0 = 0.5 * (np.sum(np.asarray(approx0_l_bs), axis=0) + np.sum(np.asarray(approx0_r_bs), axis=0))
            n_projectors = int(phi_src.shape[1] + phi_sink.shape[1])

        backend = str(self.exact_backend).strip().lower()
        if backend in ("iter", "cg", "point", "auto"):
            backend = "iterative"
        if backend in ("direct", "exact"):
            backend = "dense"
        prop, inv_info = _full_point_source_propagator(
            q=q,
            theory=theory,
            context=context,
            source=source,
            backend=backend,
            dense_max_dof=int(self.exact_dense_max_dof),
        )
        exact = _pion_two_point_momentum_from_propagator(
            prop=prop,
            source=source,
            momenta=moms,
            momentum_axis=int(mom_axis),
            average_pm=bool(self.average_pm),
        )
        bias = np.asarray(exact, dtype=np.complex128) - np.asarray(approx0, dtype=np.complex128)

        # Level-1 conditional averages.
        call_key = f"{self.name}:calls"
        call_idx = int(context.state.get(call_key, 0))
        context.state[call_key] = call_idx + 1
        dd_theory = U1TimeSlabDDTheory(
            theory,
            boundary_slices=tuple(int(v) for v in self.boundary_slices),
            boundary_width=int(self.boundary_width),
        )
        chain, evolve = self._build_level1_chain(dd_theory, seed=int(self.level1_seed) + call_idx)
        q_cur = q
        if int(self.level1_warmup) > 0:
            q_cur = evolve(q_cur, int(self.level1_warmup))

        sample_count = max(1, int(self.level1_ncfg))
        if fact_mode == "boundary_transfer":
            src_sum = None
            sink_sum = None
            for m in range(sample_count):
                if m > 0 and int(self.level1_skip) > 0:
                    q_cur = evolve(q_cur, int(self.level1_skip))
                src_m, sink_m = compute_factorized_pion_blocks(
                    q=q_cur,
                    theory=theory,
                    geometry=geom,
                    phi_overlap=phi,
                    dense_max_domain_dof=int(self.dense_max_domain_dof),
                )
                src_sum = np.asarray(src_m) if src_sum is None else (src_sum + np.asarray(src_m))
                sink_sum = np.asarray(sink_m) if sink_sum is None else (sink_sum + np.asarray(sink_m))

            src_mean = src_sum / float(sample_count)
            sink_mean = sink_sum / float(sample_count)
            approx_ml_bs, valid_mask_ml = factorized_pion_corr_from_blocks(
                source_blocks=src_mean,
                sink_blocks=sink_mean,
                geometry=geom,
                momenta=moms,
                average_pm=bool(self.average_pm),
            )
            if not np.array_equal(np.asarray(valid_mask_ml, dtype=bool), np.asarray(valid_mask, dtype=bool)):
                raise RuntimeError("Internal multilevel valid-mask mismatch")
            approx_ml = np.sum(np.asarray(approx_ml_bs), axis=0)
        else:
            src_sum_l = None
            sink_sum_l = None
            src_sum_r = None
            sink_sum_r = None
            for m in range(sample_count):
                if m > 0 and int(self.level1_skip) > 0:
                    q_cur = evolve(q_cur, int(self.level1_skip))
                src_l, sink_l = compute_giusti_asymmetric_pion_blocks(
                    q=q_cur,
                    theory=theory,
                    geometry=geom,
                    phi_surface=phi_src,
                    dense_max_domain_dof=int(self.dense_max_domain_dof),
                    dressed_domain="source",
                )
                src_r, sink_r = compute_giusti_asymmetric_pion_blocks(
                    q=q_cur,
                    theory=theory,
                    geometry=geom,
                    phi_surface=phi_sink,
                    dense_max_domain_dof=int(self.dense_max_domain_dof),
                    dressed_domain="sink",
                )
                src_sum_l = np.asarray(src_l) if src_sum_l is None else (src_sum_l + np.asarray(src_l))
                sink_sum_l = np.asarray(sink_l) if sink_sum_l is None else (sink_sum_l + np.asarray(sink_l))
                src_sum_r = np.asarray(src_r) if src_sum_r is None else (src_sum_r + np.asarray(src_r))
                sink_sum_r = np.asarray(sink_r) if sink_sum_r is None else (sink_sum_r + np.asarray(sink_r))

            approx_ml_parts = []
            for src_sum_i, sink_sum_i in (
                (src_sum_l, sink_sum_l),
                (src_sum_r, sink_sum_r),
            ):
                corr_i, valid_mask_i = factorized_pion_corr_from_blocks(
                    source_blocks=src_sum_i / float(sample_count),
                    sink_blocks=sink_sum_i / float(sample_count),
                    geometry=geom,
                    momenta=moms,
                    average_pm=bool(self.average_pm),
                )
                if not np.array_equal(np.asarray(valid_mask_i, dtype=bool), np.asarray(valid_mask, dtype=bool)):
                    raise RuntimeError("Internal Giusti multilevel valid-mask mismatch")
                approx_ml_parts.append(np.sum(np.asarray(corr_i), axis=0))
            approx_ml = 0.5 * (approx_ml_parts[0] + approx_ml_parts[1])
        corrected = approx_ml + bias
        t1 = time.perf_counter()

        out: Dict[str, float] = {}
        out.update(_flatten_corr_momentum_masked("approx_l0", approx0, moms, valid_mask))
        out.update(_flatten_corr_momentum_masked("bias", bias, moms, valid_mask))
        out.update(_flatten_corr_momentum_masked("approx_ml", approx_ml, moms, valid_mask))
        out.update(_flatten_corr_momentum_masked("corrected", corrected, moms, valid_mask))
        out.update(_flatten_corr_momentum_masked("exact", exact, moms, valid_mask))
        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["average_pm"] = 1.0 if bool(self.average_pm) else 0.0
        out["dd_n_boundary_slices"] = float(len(tuple(self.boundary_slices)))
        out["dd_boundary_width"] = float(int(self.boundary_width))
        out["dd_source_domain_index"] = float(int(geom.source_domain_index))
        out["dd_sink_domain_index"] = float(int(geom.sink_domain_index))
        out["dd_source_time"] = float(int(source[-1]))
        out["dd_n_overlap_sites"] = float(int(geom.overlap_sites.size))
        out["dd_n_projectors"] = float(int(n_projectors))
        out["dd_factorization_is_giusti"] = 1.0 if fact_mode == "giusti" else 0.0
        out["dd_surface_support_source_sites"] = float(int(geom.source_surface_sites.size))
        out["dd_surface_support_sink_sites"] = float(int(geom.sink_surface_sites.size))
        out["dd_n_level1_samples"] = float(sample_count)
        out["dd_level1_acceptance"] = float(chain.calc_acceptance())
        out.update(inv_info)
        out["wall_total_sec"] = float(t1 - t0)
        return out


@dataclass
class ProtonTwoPointMeasurement:
    """Local proton two-point function with (1 +/- gamma_t)/2 projection."""

    every: int = 1
    name: str = "proton_2pt"
    source: Optional[Tuple[int, ...]] = None
    parity_sign: int = +1
    propagator_backend: str = "iterative"  # iterative | dense | auto
    dense_max_dof: int = 4096

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        if not hasattr(theory, "gamma"):
            raise AttributeError("Theory does not expose gamma matrices required for proton correlator")
        t0 = time.perf_counter()
        source = _coerce_source(self.source, _extract_lattice_shape_from_theory(theory))
        backend = str(self.propagator_backend).strip().lower()
        if backend in ("iter", "cg", "point", "auto"):
            backend = "iterative"
        if backend in ("direct", "exact"):
            backend = "dense"
        if backend not in ("iterative", "dense"):
            raise ValueError(f"Unsupported propagator_backend for proton_2pt: {self.propagator_backend!r}")
        prop, inv_info = _full_point_source_propagator(
            q=q,
            theory=theory,
            context=context,
            source=source,
            backend=backend,
            dense_max_dof=int(self.dense_max_dof),
        )
        t1 = time.perf_counter()
        corr = _proton_two_point_from_propagator(
            prop,
            source_t=int(source[-1]),
            parity_sign=int(self.parity_sign),
            gamma=np.asarray(theory.gamma),
        )
        t2 = time.perf_counter()
        out = _flatten_corr("c", corr)
        out.update(inv_info)
        out["wall_total_sec"] = float(t2 - t0)
        out["wall_after_prop_sec"] = float(t2 - t1)
        return out


def build_inline_measurements(specs: List[Mapping[str, Any]]) -> List[InlineMeasurement]:
    out: List[InlineMeasurement] = []
    for idx, spec in enumerate(specs):
        mtype = str(spec.get("type", "")).strip().lower()
        name = str(spec.get("name", "")).strip()
        every = int(spec.get("every", 1))
        if every <= 0:
            raise ValueError(f"Measurement[{idx}] has invalid every={every}; expected >=1")
        if mtype == "plaquette":
            out.append(PlaquetteMeasurement(every=every, name=(name or "plaquette")))
            continue
        if mtype in ("pion_2pt", "pion2pt", "pion"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            backend = _parse_propagator_backend_from_spec(spec, default="iterative")
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_average = _parse_bool_from_spec(spec, "source_average", False)
            average_pm = _parse_bool_from_spec_aliases(spec, ("average_pm", "pm_average", "average_plus_minus"), False)
            out.append(
                PionTwoPointMeasurement(
                    every=every,
                    name=(name or "pion_2pt"),
                    source=source,
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    propagator_backend=str(backend),
                    dense_max_dof=int(dense_max_dof),
                    source_average=bool(source_average),
                    average_pm=bool(average_pm),
                )
            )
            continue
        if mtype in ("eta_2pt", "eta2pt", "eta"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_average = _parse_bool_from_spec(spec, "source_average", True)
            n_flavor = int(spec.get("n_flavor", spec.get("nflavor", 2)))
            include_connected = _parse_bool_from_spec(spec, "include_connected", True)
            include_full = _parse_bool_from_spec(spec, "include_full", True)
            out.append(
                EtaTwoPointMeasurement(
                    every=every,
                    name=(name or "eta_2pt"),
                    source=source,
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    dense_max_dof=int(dense_max_dof),
                    source_average=bool(source_average),
                    n_flavor=int(n_flavor),
                    include_connected=bool(include_connected),
                    include_full=bool(include_full),
                )
            )
            continue
        if mtype in ("pipi_i2_matrix", "two_pion_i2", "pipi", "pipi_i2"):
            moms = _parse_momenta_from_spec(spec)
            if len(tuple(moms)) < 1:
                raise ValueError(f"Measurement[{idx}] requires at least one momentum for {mtype}")
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_times = _parse_source_times_from_spec(spec)
            include_direct = _parse_bool_from_spec(spec, "include_direct", True)
            include_exchange = _parse_bool_from_spec_aliases(spec, ("include_exchange", "include_cross"), True)
            include_full = _parse_bool_from_spec(spec, "include_full", True)
            out.append(
                TwoPionI2MatrixMeasurement(
                    every=every,
                    name=(name or "pipi_i2_matrix"),
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    source_times=None if source_times is None else tuple(int(v) for v in source_times),
                    dense_max_dof=int(dense_max_dof),
                    include_direct=bool(include_direct),
                    include_exchange=bool(include_exchange),
                    include_full=bool(include_full),
                )
            )
            continue
        if mtype in ("pion_2pt_dd", "dd_pion_2pt", "pion2pt_dd", "dd_pion2pt", "dd_pion"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_average = _parse_bool_from_spec(spec, "source_average", False)
            average_pm = _parse_bool_from_spec_aliases(spec, ("average_pm", "pm_average", "average_plus_minus"), False)
            boundary_slices = _parse_boundary_slices_from_spec(spec)
            boundary_width = int(spec.get("boundary_width", spec.get("bw", 1)))
            out.append(
                DDPionTwoPointMeasurement(
                    every=every,
                    name=(name or "pion_2pt_dd"),
                    source=source,
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    dense_max_dof=int(dense_max_dof),
                    source_average=bool(source_average),
                    average_pm=bool(average_pm),
                    boundary_slices=tuple(int(v) for v in boundary_slices),
                    boundary_width=int(boundary_width),
                )
            )
            continue
        if mtype in ("eta_2pt_dd", "dd_eta_2pt", "eta2pt_dd", "dd_eta2pt", "dd_eta"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_average = _parse_bool_from_spec(spec, "source_average", True)
            n_flavor = int(spec.get("n_flavor", spec.get("nflavor", 2)))
            include_connected = _parse_bool_from_spec(spec, "include_connected", True)
            include_full = _parse_bool_from_spec(spec, "include_full", True)
            boundary_slices = _parse_boundary_slices_from_spec(spec)
            boundary_width = int(spec.get("boundary_width", spec.get("bw", 1)))
            out.append(
                DDEtaTwoPointMeasurement(
                    every=every,
                    name=(name or "eta_2pt_dd"),
                    source=source,
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    dense_max_dof=int(dense_max_dof),
                    source_average=bool(source_average),
                    n_flavor=int(n_flavor),
                    include_connected=bool(include_connected),
                    include_full=bool(include_full),
                    boundary_slices=tuple(int(v) for v in boundary_slices),
                    boundary_width=int(boundary_width),
                )
            )
            continue
        if mtype in ("pion_2pt_ml_dd", "dd_pion_2pt_ml", "pion2pt_ml_dd", "dd_pion2pt_ml", "pion_2pt_2lvl"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            boundary_slices = _parse_boundary_slices_from_spec(spec)
            boundary_width = int(spec.get("boundary_width", spec.get("bw", 1)))
            average_pm = _parse_bool_from_spec_aliases(spec, ("average_pm", "pm_average", "average_plus_minus"), False)
            out.append(
                DDTwoLevelFactorizedPionMeasurement(
                    every=every,
                    name=(name or "pion_2pt_ml_dd"),
                    source=source,
                    momenta=tuple(int(v) for v in moms),
                    momentum_axis=int(mom_axis),
                    average_pm=bool(average_pm),
                    boundary_slices=tuple(int(v) for v in boundary_slices),
                    boundary_width=int(boundary_width),
                    source_margin=int(spec.get("source_margin", 1)),
                    factorization_kind=str(spec.get("factorization_kind", spec.get("factorization", "boundary_transfer"))),
                    projector_kind=str(spec.get("projector_kind", spec.get("projector", "full"))),
                    projector_nvec=int(spec.get("projector_nvec", spec.get("nvec", 0))),
                    probe_stride=int(spec.get("probe_stride", spec.get("projector_stride", 2))),
                    dense_max_domain_dof=int(spec.get("dense_max_domain_dof", 1024)),
                    exact_backend=str(spec.get("exact_backend", "iterative")),
                    exact_dense_max_dof=int(spec.get("exact_dense_max_dof", spec.get("dense_max_dof", 4096))),
                    level1_ncfg=int(spec.get("level1_ncfg", spec.get("n1", 8))),
                    level1_warmup=int(spec.get("level1_warmup", 2)),
                    level1_skip=int(spec.get("level1_skip", 2)),
                    level1_update=str(spec.get("level1_update", "hmc")),
                    level1_integrator=str(spec.get("level1_integrator", "minnorm2")),
                    level1_nmd=int(spec.get("level1_nmd", 4)),
                    level1_tau=float(spec.get("level1_tau", 0.5)),
                    level1_gamma=float(spec.get("level1_gamma", 0.3)),
                    level1_seed=int(spec.get("level1_seed", 0)),
                )
            )
            continue
        if mtype in ("proton_2pt", "proton2pt", "nucleon_2pt", "nucleon2pt", "proton", "nucleon"):
            source = _parse_source_from_spec(spec)
            backend = _parse_propagator_backend_from_spec(spec, default="iterative")
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            parity_raw = str(spec.get("parity", "+")).strip().lower()
            if parity_raw in ("+", "plus", "pos", "positive", "forward"):
                parity_sign = +1
            elif parity_raw in ("-", "minus", "neg", "negative", "backward"):
                parity_sign = -1
            else:
                raise ValueError(
                    f"Measurement[{idx}] proton/nucleon parity must be +/- (got {spec.get('parity')!r})"
                )
            out.append(
                ProtonTwoPointMeasurement(
                    every=every,
                    name=(name or "proton_2pt"),
                    source=source,
                    parity_sign=parity_sign,
                    propagator_backend=str(backend),
                    dense_max_dof=int(dense_max_dof),
                )
            )
            continue
        raise ValueError(f"Unsupported inline measurement type: {mtype!r}")
    return out


def run_inline_measurements(
    measurements: List[InlineMeasurement],
    step: int,
    q,
    theory,
    context: MeasurementContext,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context.cache = {}
    for m in measurements:
        every = max(1, int(getattr(m, "every", 1)))
        if int(step) % every != 0:
            continue
        vals = dict(m.run(q, theory, context))
        context.cache[str(m.name)] = vals
        records.append(
            {
                "step": int(step),
                "name": str(m.name),
                "values": vals,
            }
        )
    return records

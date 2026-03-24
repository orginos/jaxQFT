"""Inline measurement framework for production MCMC drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import jax
import numpy as np

from jaxqft.fermions import gamma5

from .domain_decomposition import TimeSlabDecomposition


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


def _pion_two_point_momentum_from_propagator(
    prop: np.ndarray,
    source: Sequence[int],
    momenta: Sequence[int],
    momentum_axis: int,
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
            phase = np.exp(1j * (2.0 * np.pi * float(p) / float(lmom)) * (np.arange(lmom) - x0))
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
                ph = np.exp(1j * (2.0 * np.pi * float(p) / float(lmom)) * dx)
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
        ph = np.exp(1j * (2.0 * np.pi * float(p) / float(lmom)) * dx)
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
            )
        t2 = time.perf_counter()
        out = _flatten_corr_momentum("c", corr, moms, include_legacy_zero=True)
        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
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
class DDPionTwoPointMeasurement:
    """Connected pseudoscalar two-point function from the exact interior Schur complement."""

    every: int = 1
    name: str = "pion_2pt_dd"
    source: Optional[Tuple[int, ...]] = None
    momenta: Tuple[int, ...] = (0,)
    momentum_axis: int = 0
    dense_max_dof: int = 4096
    source_average: bool = False
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
        )
        t2 = time.perf_counter()

        out = _flatten_corr_momentum("c", corr, moms, include_legacy_zero=True)
        out["mom_axis"] = float(mom_axis)
        out["n_momenta"] = float(len(moms))
        out["source_average"] = 1.0 if bool(self.source_average) else 0.0
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
        if mtype in ("pion_2pt_dd", "dd_pion_2pt", "pion2pt_dd", "dd_pion2pt", "dd_pion"):
            source = _parse_source_from_spec(spec)
            moms = _parse_momenta_from_spec(spec)
            mom_axis = int(spec.get("momentum_axis", spec.get("mom_axis", 0)))
            dense_max_dof = int(spec.get("dense_max_dof", 4096))
            source_average = _parse_bool_from_spec(spec, "source_average", False)
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

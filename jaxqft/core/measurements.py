"""Inline measurement framework for production MCMC drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import jax
import numpy as np

from jaxqft.fermions import gamma5


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


def _extract_lattice_shape_from_theory(theory) -> Tuple[int, ...]:
    if hasattr(theory, "lattice_shape"):
        shp = tuple(int(v) for v in tuple(theory.lattice_shape))
        if shp:
            return shp
    fshp = tuple(int(v) for v in tuple(theory.fermion_shape()))
    if len(fshp) < 4:
        raise ValueError(f"Unexpected fermion shape rank: {fshp}")
    return tuple(fshp[1:-2])


def _full_point_source_propagator(
    q,
    theory,
    context: MeasurementContext,
    source: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not hasattr(theory, "solve_direct"):
        raise AttributeError("Theory does not provide solve_direct(U, rhs) for correlator measurements")
    if not hasattr(theory, "fermion_shape"):
        raise AttributeError("Theory does not provide fermion_shape() for correlator measurements")

    lattice_shape = _extract_lattice_shape_from_theory(theory)
    src = _coerce_source(source, lattice_shape)
    cache_key = f"quark_prop:{','.join(str(v) for v in src)}"
    cached = context.cache.get(cache_key)
    if isinstance(cached, dict) and ("prop" in cached) and ("timing" in cached):
        info = dict(cached["timing"])
        info["inv_cache_hit"] = 1.0
        info["inv_solve_total_sec_this_call"] = 0.0
        return np.asarray(cached["prop"]), info

    ferm_shape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    if len(ferm_shape) < 4:
        raise ValueError(f"Unexpected fermion shape rank: {ferm_shape}")
    bs = int(ferm_shape[0])
    ns = int(ferm_shape[-2])
    nc = int(ferm_shape[-1])
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
            sol = np.asarray(theory.solve_direct(q, jax.device_put(rhs)), dtype=fdtype)
            solve_times.append(float(time.perf_counter() - ts))
            prop[..., src_spin, src_color] = sol

    build_dt = float(time.perf_counter() - t_build0)
    st = np.asarray(solve_times, dtype=np.float64)
    timing = {
        "inv_n_solves": float(st.size),
        "inv_solve_total_sec_step": float(np.sum(st)) if st.size else 0.0,
        "inv_solve_mean_sec": float(np.mean(st)) if st.size else 0.0,
        "inv_solve_min_sec": float(np.min(st)) if st.size else 0.0,
        "inv_solve_max_sec": float(np.max(st)) if st.size else 0.0,
        "inv_prop_build_wall_sec": float(build_dt),
    }
    context.cache[cache_key] = {"prop": prop, "timing": timing}
    info = dict(timing)
    info["inv_cache_hit"] = 0.0
    info["inv_solve_total_sec_this_call"] = float(timing["inv_solve_total_sec_step"])
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


@dataclass
class PionTwoPointMeasurement:
    """Connected pseudoscalar two-point function with a point source."""

    every: int = 1
    name: str = "pion_2pt"
    source: Optional[Tuple[int, ...]] = None

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        t0 = time.perf_counter()
        source = _coerce_source(self.source, _extract_lattice_shape_from_theory(theory))
        prop, inv_info = _full_point_source_propagator(q=q, theory=theory, context=context, source=source)
        t1 = time.perf_counter()
        corr = _pion_two_point_from_propagator(prop, source_t=int(source[-1]))
        t2 = time.perf_counter()
        out = _flatten_corr("c", corr)
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

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        if not hasattr(theory, "gamma"):
            raise AttributeError("Theory does not expose gamma matrices required for proton correlator")
        t0 = time.perf_counter()
        source = _coerce_source(self.source, _extract_lattice_shape_from_theory(theory))
        prop, inv_info = _full_point_source_propagator(q=q, theory=theory, context=context, source=source)
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
            out.append(
                PionTwoPointMeasurement(
                    every=every,
                    name=(name or "pion_2pt"),
                    source=source,
                )
            )
            continue
        if mtype in ("proton_2pt", "proton2pt", "nucleon_2pt", "nucleon2pt", "proton", "nucleon"):
            source = _parse_source_from_spec(spec)
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

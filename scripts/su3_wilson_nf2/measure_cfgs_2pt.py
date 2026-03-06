#!/usr/bin/env python3
"""Measure pion/proton two-point functions on a set of gauge configurations.

This is intended for the practical workflow:
- generate ensembles externally (e.g. Chroma),
- measure 2pt observables in jaxQFT with explicit inversion timing metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Keep CPU default on macOS unless user explicitly overrides.
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
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
from jaxqft.io import decode_scidac_gauge
from jaxqft.models.su3_wilson_nf2 import SU3WilsonNf2


def _parse_shape(s: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("shape must be non-empty")
    return tuple(vals)


def _parse_source(s: str, nd: int) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if len(vals) != int(nd):
        raise ValueError(f"source must have Nd={nd} values, got {len(vals)}")
    return tuple(vals)


def _parse_fermion_bc(s: str, nd: int) -> Tuple[complex, ...]:
    txt = str(s).strip().lower()
    if txt in ("", "periodic", "p"):
        return tuple([1.0 + 0.0j] * int(nd))
    if txt in ("antiperiodic-t", "anti-t", "ap-t", "apt", "chroma"):
        vals = [1.0 + 0.0j] * int(nd)
        vals[-1] = -1.0 + 0.0j
        return tuple(vals)
    toks = [v.strip() for v in str(s).split(",") if v.strip()]
    vals = [complex(t.replace("I", "j").replace("i", "j")) for t in toks]
    if len(vals) != int(nd):
        raise ValueError(f"fermion_bc must have Nd={nd} entries; got {len(vals)}")
    return tuple(vals)


def _extract_gauge_from_pickle_payload(payload, src_path: str):
    if isinstance(payload, dict):
        for key in ("q", "U", "gauge", "links", "cfg"):
            if key in payload:
                return np.asarray(payload[key]), f"pickle-dict:{key}"
        raise ValueError(
            f"Pickle '{src_path}' has no gauge key; expected one of q/U/gauge/links/cfg."
        )
    if isinstance(payload, np.ndarray):
        return np.asarray(payload), "pickle-array"
    raise ValueError(f"Unsupported pickle payload type: {type(payload).__name__}")


def _normalize_loaded_gauge(gauge: np.ndarray, expected_shape: tuple[int, ...], nd: int):
    g = np.asarray(gauge)
    if g.ndim == nd + 3:
        g = g[None, ...]
    if g.ndim != nd + 4:
        raise ValueError(
            f"Loaded gauge rank mismatch: ndim={g.ndim}, expected {nd+4} "
            f"(shape {tuple(g.shape)} vs expected {expected_shape})"
        )

    candidates = [("as-is", g)]
    if g.shape[1] == nd:
        candidates.append(("bm-to-bxy", np.moveaxis(g, 1, 1 + nd)))
    if g.shape[1 + nd] == nd:
        candidates.append(("bxy-to-bm", np.moveaxis(g, 1 + nd, 1)))

    for tag, cand in candidates:
        if tuple(cand.shape[1:]) != tuple(expected_shape[1:]):
            continue
        if cand.shape[0] == expected_shape[0]:
            return cand.astype(np.complex64, copy=False), tag
        if cand.shape[0] == 1 and expected_shape[0] > 1:
            return np.repeat(cand, expected_shape[0], axis=0).astype(np.complex64, copy=False), f"{tag}+repeat-batch"

    tried = ", ".join(f"{tag}:{tuple(c.shape)}" for tag, c in candidates)
    raise ValueError(
        "Loaded gauge shape mismatch after layout normalization. "
        f"expected={expected_shape} got={tuple(g.shape)} tried=[{tried}]"
    )


def _load_gauge(path: str, expected_shape: tuple[int, ...], nd: int, batch_size: int):
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".lime":
        raw = decode_scidac_gauge(path, batch_size=int(batch_size))
        src = "lime"
    elif ext == ".npy":
        raw = np.load(path)
        src = "npy"
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        raw, src = _extract_gauge_from_pickle_payload(payload, path)
    else:
        raise ValueError(f"Unsupported gauge extension for '{path}'; use .lime/.npy/.pkl")

    q_arr, norm = _normalize_loaded_gauge(raw, expected_shape=expected_shape, nd=int(nd))
    return jax.device_put(q_arr), f"{src}/{norm}"


def _avg_err(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size < 2:
        return float(arr.mean()), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure pion/proton 2pt on cfg set")
    ap.add_argument("--cfg-glob", type=str, required=True, help="glob for gauge configs (.lime/.npy/.pkl)")
    ap.add_argument("--shape", type=str, default="8,8,8,16")
    ap.add_argument("--beta", type=float, default=5.6)
    ap.add_argument("--mass", type=float, default=-0.8152866242)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--fermion-bc", type=str, default="antiperiodic-t")
    ap.add_argument("--layout", type=str, default="BMXYIJ", choices=["BMXYIJ", "BXYMIJ"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--solver-kind", type=str, default="bicgstab", choices=["cg", "bicgstab", "gmres"])
    ap.add_argument("--solver-form", type=str, default="normal", choices=["normal", "split", "eo_split"])
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=1200)
    ap.add_argument("--preconditioner", type=str, default="none", choices=["none", "jacobi"])
    ap.add_argument("--gmres-restart", type=int, default=32)
    ap.add_argument("--gmres-solve-method", type=str, default="batched")
    ap.add_argument("--fermion-monomial-kind", type=str, default="eo_preconditioned", choices=["unpreconditioned", "eo_preconditioned"])
    ap.add_argument("--source", type=str, default="0,0,0,0")
    ap.add_argument("--parity", type=str, default="+", choices=["+", "-"])
    ap.add_argument("--only", type=str, default="both", choices=["both", "pion", "proton"])
    ap.add_argument("--limit", type=int, default=0, help="max number of configs (0 = all)")
    ap.add_argument("--out-json", type=str, default="runs/LQCD-example/measure_2pt_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shape = _parse_shape(args.shape)
    nd = len(shape)
    source = _parse_source(args.source, nd=nd)
    fermion_bc = _parse_fermion_bc(args.fermion_bc, nd=nd)

    cfg_paths = sorted(glob.glob(str(args.cfg_glob)))
    if args.limit > 0:
        cfg_paths = cfg_paths[: int(args.limit)]
    if not cfg_paths:
        raise ValueError(f"No files matched --cfg-glob '{args.cfg_glob}'")

    theory = SU3WilsonNf2(
        lattice_shape=shape,
        beta=float(args.beta),
        batch_size=int(args.batch),
        layout=str(args.layout),
        seed=int(args.seed),
        mass=float(args.mass),
        wilson_r=float(args.r),
        cg_tol=float(args.tol),
        cg_maxiter=int(args.maxiter),
        solver_kind=str(args.solver_kind),
        solver_form=str(args.solver_form),
        preconditioner_kind=str(args.preconditioner),
        gmres_restart=int(args.gmres_restart),
        gmres_solve_method=str(args.gmres_solve_method),
        fermion_bc=fermion_bc,
        include_gauge_monomial=False,
        include_fermion_monomial=False,
        fermion_monomial_kind=str(args.fermion_monomial_kind),
    )

    specs: List[Mapping[str, Any]] = []
    if args.only in ("both", "pion"):
        specs.append({"type": "pion_2pt", "name": "pion_2pt", "every": 1, "source": list(source)})
    if args.only in ("both", "proton"):
        specs.append(
            {
                "type": "proton_2pt",
                "name": "proton_2pt",
                "every": 1,
                "source": list(source),
                "parity": str(args.parity),
            }
        )
    measurements = build_inline_measurements(specs)
    mctx = MeasurementContext()

    expected_shape = tuple(int(v) for v in theory.field_shape())
    rows: List[Dict[str, Any]] = []
    hist: Dict[str, List[float]] = {}

    t0 = time.perf_counter()
    for i, path in enumerate(cfg_paths):
        q, src_tag = _load_gauge(path, expected_shape=expected_shape, nd=nd, batch_size=int(args.batch))
        ts = time.perf_counter()
        recs = run_inline_measurements(measurements, step=i, q=q, theory=theory, context=mctx)
        row_vals: Dict[str, float] = {}
        for r in recs:
            name = str(r["name"])
            for k, v in dict(r["values"]).items():
                key = f"{name}.{k}"
                fv = float(v)
                row_vals[key] = fv
                hist.setdefault(key, []).append(fv)
        row = {
            "index": int(i),
            "cfg_path": str(path),
            "cfg_source": str(src_tag),
            "measure_wall_sec": float(time.perf_counter() - ts),
            "values": row_vals,
        }
        rows.append(row)
        print(
            f"[{i+1}/{len(cfg_paths)}] {Path(path).name}:"
            f" wall={row['measure_wall_sec']:.3f}s"
            f" keys={len(row_vals)}"
        )

    t1 = time.perf_counter()

    summary: Dict[str, Dict[str, float]] = {}
    for key, vals in sorted(hist.items()):
        m, e = _avg_err(vals)
        summary[key] = {
            "mean": float(m),
            "err": float(e),
            "n": float(len(vals)),
        }

    inv_summary: Dict[str, Dict[str, float]] = {}
    for key, vals in sorted(hist.items()):
        if not key.endswith(".inv_solve_total_sec_this_call"):
            continue
        arr = np.asarray(vals, dtype=np.float64)
        nz = arr[arr > 0.0]
        cache_key = key[: -len("inv_solve_total_sec_this_call")] + "inv_cache_hit"
        cache_arr = np.asarray(hist.get(cache_key, []), dtype=np.float64)
        inv_summary[key] = {
            "total_sec": float(np.sum(arr)),
            "mean_build_sec": float(np.mean(nz)) if nz.size else 0.0,
            "n_eval": float(arr.size),
            "n_build": float(nz.size),
            "cache_hit_frac": float(np.mean(cache_arr)) if cache_arr.size else float("nan"),
        }

    out = {
        "meta": {
            "cfg_glob": str(args.cfg_glob),
            "n_cfg": int(len(cfg_paths)),
            "shape": list(shape),
            "beta": float(args.beta),
            "mass": float(args.mass),
            "r": float(args.r),
            "fermion_bc": [str(v) for v in fermion_bc],
            "solver_kind": str(args.solver_kind),
            "solver_form": str(args.solver_form),
            "preconditioner": str(args.preconditioner),
            "source": list(source),
            "parity": str(args.parity),
            "only": str(args.only),
            "total_wall_sec": float(t1 - t0),
        },
        "summary": summary,
        "inversion_summary": inv_summary,
        "rows": rows,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Total wall: {float(t1 - t0):.3f}s for {len(cfg_paths)} cfgs")


if __name__ == "__main__":
    main()

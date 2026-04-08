#!/usr/bin/env python3
"""Measure second-moment correlation lengths across RG levels for phi^4 flows.

This script samples the learned RG coarse-eta Gaussian flow, blocks the samples
level by level using the model's fixed RG split, and measures the second-moment
correlation length ``xi_2`` on the scalar coarse field at each level.

The measurement is based on independent model samples, so statistical errors are
estimated with a simple jackknife over sample bins.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


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

from jaxqft.models.phi4_rg_cond_flow import _split_rg
from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
    init_rg_coarse_eta_gaussian_flow,
    rg_coarse_eta_gaussian_flow_g,
    rg_coarse_eta_gaussian_flow_prior_sample,
)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["weights"] = tree_to_jax(payload["weights"])
    return payload


def _build_model_from_arch(arch: dict):
    key = jax.random.PRNGKey(0)
    model = init_rg_coarse_eta_gaussian_flow(
        key,
        size=(int(arch["L"]), int(arch["L"])),
        width=int(arch["width"]),
        width_levels=arch.get("width_levels"),
        n_cycles=int(arch.get("n_cycles", 2)),
        n_cycles_levels=arch.get("n_cycles_levels"),
        radius=int(arch.get("radius", 1)),
        radius_levels=arch.get("radius_levels"),
        eta_gaussian=str(arch.get("eta_gaussian", "coarse_patch")),
        gaussian_radius=int(arch.get("gaussian_radius", arch.get("radius", 1))),
        gaussian_radius_levels=arch.get("gaussian_radius_levels"),
        gaussian_width=int(arch.get("gaussian_width", arch["width"])),
        gaussian_width_levels=arch.get("gaussian_width_levels"),
        terminal_prior=str(arch.get("terminal_prior", "learned")),
        rg_type=str(arch["rg_type"]),
        log_scale_clip=float(arch["log_scale_clip"]),
        offdiag_clip=float(arch.get("offdiag_clip", 2.0)),
        terminal_n_layers=int(arch.get("terminal_n_layers", 2)),
        terminal_width=int(arch.get("terminal_width", arch["width"])),
        output_init_scale=float(arch.get("output_init_scale", 1e-2)),
        parity=str(arch.get("parity", "sym")),
    )
    return model["cfg"]


def _collect_level_fields(x: jax.Array, rg_mode: int):
    fields = [x]
    xx = x
    while min(int(xx.shape[1]), int(xx.shape[2])) > 2:
        xx, _ = _split_rg(xx, rg_mode)
        fields.append(xx)
    return fields


def _make_level_observable_fn(shape: tuple[int, int]):
    ny, nx = shape
    vol = float(ny * nx)
    phase_y = jnp.exp(2j * jnp.pi * jnp.arange(ny, dtype=jnp.float32) / float(ny)).reshape((1, ny, 1))
    phase_x = jnp.exp(2j * jnp.pi * jnp.arange(nx, dtype=jnp.float32) / float(nx)).reshape((1, 1, nx))

    @jax.jit
    def obs_fn(xx: jax.Array):
        m = jnp.mean(xx, axis=(1, 2))
        p1_y = jnp.mean(xx * phase_y, axis=(1, 2))
        p1_x = jnp.mean(xx * phase_x, axis=(1, 2))
        c2p_y = vol * jnp.real(jnp.conj(p1_y) * p1_y)
        c2p_x = vol * jnp.real(jnp.conj(p1_x) * p1_x)
        return m, c2p_x, c2p_y

    return obs_fn


def _xi2_from_chi_c2p(L: int, chi: float, c2p: float) -> float:
    if L <= 1 or c2p <= 0.0:
        return float("nan")
    ratio = chi / c2p - 1.0
    if ratio <= 0.0:
        return float("nan")
    return float(np.sqrt(ratio) / (2.0 * np.sin(np.pi / float(L))))


def _jk_error(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n <= 1:
        return float("nan")
    mean = float(np.mean(vals))
    return float(np.sqrt((n - 1) / n * np.sum((vals - mean) ** 2)))


def _jackknife_level(m: np.ndarray, c2p_x: np.ndarray, c2p_y: np.ndarray, L: int, bin_size: int) -> dict:
    m = np.asarray(m, dtype=np.float64)
    c2p_x = np.asarray(c2p_x, dtype=np.float64)
    c2p_y = np.asarray(c2p_y, dtype=np.float64)
    nsamples = int(m.size)
    if nsamples < 2:
        raise ValueError("Need at least 2 samples for statistics")

    m2 = m * m
    c2p = 0.5 * (c2p_x + c2p_y)
    vol = int(L * L)

    m_mean = float(np.mean(m))
    m2_mean = float(np.mean(m2))
    chi = float(vol * (m2_mean - m_mean * m_mean))
    c2p_x_mean = float(np.mean(c2p_x))
    c2p_y_mean = float(np.mean(c2p_y))
    c2p_mean = float(np.mean(c2p))
    xi2 = _xi2_from_chi_c2p(L, chi, c2p_mean)
    xi2_x = _xi2_from_chi_c2p(L, chi, c2p_x_mean)
    xi2_y = _xi2_from_chi_c2p(L, chi, c2p_y_mean)

    nbins = max(2, nsamples // max(1, int(bin_size)))
    nbins = min(nbins, nsamples)
    bins = np.array_split(np.arange(nsamples), nbins)

    m_sum = float(np.sum(m))
    m2_sum = float(np.sum(m2))
    c2p_x_sum = float(np.sum(c2p_x))
    c2p_y_sum = float(np.sum(c2p_y))

    jk_m = []
    jk_chi = []
    jk_c2p = []
    jk_c2p_x = []
    jk_c2p_y = []
    jk_xi = []
    jk_xi_x = []
    jk_xi_y = []
    for idxs in bins:
        keep = nsamples - int(idxs.size)
        if keep <= 0:
            continue
        m_i = (m_sum - float(np.sum(m[idxs]))) / keep
        m2_i = (m2_sum - float(np.sum(m2[idxs]))) / keep
        c2p_x_i = (c2p_x_sum - float(np.sum(c2p_x[idxs]))) / keep
        c2p_y_i = (c2p_y_sum - float(np.sum(c2p_y[idxs]))) / keep
        chi_i = float(vol * (m2_i - m_i * m_i))
        c2p_i = 0.5 * (c2p_x_i + c2p_y_i)
        jk_m.append(m_i)
        jk_chi.append(chi_i)
        jk_c2p.append(c2p_i)
        jk_c2p_x.append(c2p_x_i)
        jk_c2p_y.append(c2p_y_i)
        jk_xi.append(_xi2_from_chi_c2p(L, chi_i, c2p_i))
        jk_xi_x.append(_xi2_from_chi_c2p(L, chi_i, c2p_x_i))
        jk_xi_y.append(_xi2_from_chi_c2p(L, chi_i, c2p_y_i))

    return {
        "n_samples": int(nsamples),
        "n_jackknife_bins": int(len(jk_xi)),
        "bin_size": int(bin_size),
        "magnetization": {"mean": m_mean, "stderr": _jk_error(np.asarray(jk_m))},
        "chi_m": {"mean": chi, "stderr": _jk_error(np.asarray(jk_chi))},
        "C2p": {"mean": c2p_mean, "stderr": _jk_error(np.asarray(jk_c2p))},
        "C2p_x": {"mean": c2p_x_mean, "stderr": _jk_error(np.asarray(jk_c2p_x))},
        "C2p_y": {"mean": c2p_y_mean, "stderr": _jk_error(np.asarray(jk_c2p_y))},
        "xi2": {"mean": xi2, "stderr": _jk_error(np.asarray(jk_xi))},
        "xi2_x": {"mean": xi2_x, "stderr": _jk_error(np.asarray(jk_xi_x))},
        "xi2_y": {"mean": xi2_y, "stderr": _jk_error(np.asarray(jk_xi_y))},
        "xi2_over_L": {"mean": (xi2 / float(L)) if np.isfinite(xi2) else float("nan"),
                        "stderr": (_jk_error(np.asarray(jk_xi)) / float(L)) if L > 0 else float("nan")},
    }


def measure_levels(*, cfg: dict, weights: dict, nsamples: int, batch_size: int, seed: int, jk_bin_size: int):
    key = jax.random.PRNGKey(seed)
    obs_fns = {}
    level_m = None
    level_c2p_x = None
    level_c2p_y = None
    level_shapes = None

    remaining = int(nsamples)
    tic = time.perf_counter()
    while remaining > 0:
        bsz = min(int(batch_size), remaining)
        key, sub = jax.random.split(key)
        z = rg_coarse_eta_gaussian_flow_prior_sample(sub, cfg, bsz)
        x = rg_coarse_eta_gaussian_flow_g(cfg, z, weights)
        fields = _collect_level_fields(x, int(cfg["rg_mode"]))

        if level_m is None:
            nlevels = len(fields)
            level_m = [[] for _ in range(nlevels)]
            level_c2p_x = [[] for _ in range(nlevels)]
            level_c2p_y = [[] for _ in range(nlevels)]
            level_shapes = [tuple(int(v) for v in fld.shape[1:3]) for fld in fields]

        for level, fld in enumerate(fields):
            shape = tuple(int(v) for v in fld.shape[1:3])
            if shape not in obs_fns:
                obs_fns[shape] = _make_level_observable_fn(shape)
            m, c2x, c2y = obs_fns[shape](fld)
            level_m[level].append(np.asarray(m))
            level_c2p_x[level].append(np.asarray(c2x))
            level_c2p_y[level].append(np.asarray(c2y))
        remaining -= bsz
    sample_sec = time.perf_counter() - tic

    assert level_shapes is not None
    levels_out = []
    nlevels = len(level_shapes)
    for level in range(nlevels):
        m = np.concatenate(level_m[level], axis=0)
        c2x = np.concatenate(level_c2p_x[level], axis=0)
        c2y = np.concatenate(level_c2p_y[level], axis=0)
        L = int(level_shapes[level][0])
        stats = _jackknife_level(m, c2x, c2y, L, jk_bin_size)
        stats.update(
            {
                "level_from_fine": int(level),
                "distance_from_bottom": int(nlevels - 1 - level),
                "shape": [int(level_shapes[level][0]), int(level_shapes[level][1])],
                "L": int(L),
            }
        )
        levels_out.append(stats)

    for level in range(1, nlevels):
        prev = levels_out[level - 1]["xi2"]["mean"]
        cur = levels_out[level]["xi2"]["mean"]
        ratio = float(cur / prev) if np.isfinite(cur) and np.isfinite(prev) and prev != 0.0 else float("nan")
        levels_out[level]["xi2_over_prev_level"] = ratio

    return {
        "n_levels": int(nlevels),
        "levels": levels_out,
        "sample_source": {
            "source": "model",
            "n_samples": int(nsamples),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "sample_sec": float(sample_sec),
        },
    }


def _print_summary(result: dict):
    print("\nPer-level xi_2 summary:")
    print(
        f"{'level':>5}  {'L':>5}  {'xi2':>10}  {'err':>10}  {'xi2/L':>10}  {'chi':>12}  {'C2p':>12}  {'dist_bottom':>11}"
    )
    for row in result["levels"]:
        xi = row["xi2"]["mean"]
        xi_err = row["xi2"]["stderr"]
        xi_over_L = row["xi2_over_L"]["mean"]
        chi = row["chi_m"]["mean"]
        c2p = row["C2p"]["mean"]
        print(
            f"{row['level_from_fine']:5d}  {row['L']:5d}  {xi:10.5f}  {xi_err:10.5f}  "
            f"{xi_over_L:10.5f}  {chi:12.5f}  {c2p:12.5f}  {row['distance_from_bottom']:11d}"
        )


def main():
    ap = argparse.ArgumentParser(description="Measure xi_2 across RG levels for the coarse-eta Gaussian flow.")
    ap.add_argument("--resume", type=str, required=True, help="checkpoint to analyze")
    ap.add_argument("--nsamples", type=int, default=512, help="total number of independent model samples")
    ap.add_argument("--batch-size", type=int, default=64, help="sampling batch size")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for model sampling")
    ap.add_argument("--jk-bin-size", type=int, default=32, help="jackknife bin size in independent samples")
    ap.add_argument("--json-out", type=str, default="", help="optional JSON output path")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    ckpt = load_checkpoint(args.resume)
    arch = ckpt["arch"]
    cfg = _build_model_from_arch(arch)
    weights = ckpt["weights"]

    result = measure_levels(
        cfg=cfg,
        weights=weights,
        nsamples=int(args.nsamples),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        jk_bin_size=int(args.jk_bin_size),
    )
    result["checkpoint"] = str(args.resume)
    result["arch"] = {
        "L": int(arch["L"]),
        "lam": float(arch["lam"]),
        "mass": float(arch["mass"]),
        "width": int(arch["width"]),
        "width_levels": list(int(v) for v in arch.get("width_levels", [])),
        "n_cycles_levels": list(int(v) for v in arch.get("n_cycles_levels", [])),
        "radius_levels": list(int(v) for v in arch.get("radius_levels", [])),
        "eta_gaussian": str(arch.get("eta_gaussian", "unknown")),
        "terminal_prior": str(arch.get("terminal_prior", "unknown")),
    }

    _print_summary(result)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved {args.json_out}")


if __name__ == "__main__":
    main()

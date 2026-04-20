#!/usr/bin/env python3
"""Sample a trained RG coarse-eta Gaussian flow and plot field histograms."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
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


def sample_model_fields(*, cfg: dict, weights: dict, n_samples: int, batch_size: int, seed: int) -> tuple[np.ndarray, dict]:
    key = jax.random.PRNGKey(seed)
    chunks: list[np.ndarray] = []
    remaining = int(n_samples)
    tic = time.perf_counter()
    while remaining > 0:
        bsz = min(int(batch_size), remaining)
        key, sub = jax.random.split(key)
        z = rg_coarse_eta_gaussian_flow_prior_sample(sub, cfg, bsz)
        x = rg_coarse_eta_gaussian_flow_g(cfg, z, weights)
        chunks.append(np.asarray(x))
        remaining -= bsz
    elapsed = time.perf_counter() - tic
    arr = np.concatenate(chunks, axis=0)
    return arr, {
        "n_samples": int(arr.shape[0]),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "sample_sec": float(elapsed),
    }


def _auto_range(values: np.ndarray, quantile: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    q = float(np.quantile(np.abs(arr), quantile))
    q = max(q, 1e-6)
    return -q, q


def _histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> dict:
    density, edges = np.histogram(values, bins=int(bins), range=value_range, density=True)
    counts, _ = np.histogram(values, bins=int(bins), range=value_range, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "density": density.tolist(),
        "counts": counts.tolist(),
    }


def _fwhm(centers: np.ndarray, density: np.ndarray, peak_idx: int) -> float | None:
    peak = float(density[peak_idx])
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    half = 0.5 * peak
    left = peak_idx
    while left > 0 and density[left] >= half:
        left -= 1
    right = peak_idx
    while right + 1 < density.size and density[right] >= half:
        right += 1
    if right <= left:
        return None
    return float(centers[right] - centers[left])


def _double_peak_summary(hist: dict) -> dict:
    centers = np.asarray(hist["bin_centers"], dtype=np.float64)
    density = np.asarray(hist["density"], dtype=np.float64)
    if centers.size == 0:
        return {}
    neg_mask = centers < 0.0
    pos_mask = centers > 0.0
    if not np.any(neg_mask) or not np.any(pos_mask):
        return {}
    neg_indices = np.flatnonzero(neg_mask)
    pos_indices = np.flatnonzero(pos_mask)
    neg_peak_idx = int(neg_indices[np.argmax(density[neg_mask])])
    pos_peak_idx = int(pos_indices[np.argmax(density[pos_mask])])
    zero_idx = int(np.argmin(np.abs(centers)))
    neg_peak = float(centers[neg_peak_idx])
    pos_peak = float(centers[pos_peak_idx])
    neg_height = float(density[neg_peak_idx])
    pos_height = float(density[pos_peak_idx])
    valley_height = float(density[zero_idx])
    peak_height = max(neg_height, pos_height, 1e-12)
    return {
        "negative_peak": neg_peak,
        "positive_peak": pos_peak,
        "peak_separation": float(pos_peak - neg_peak),
        "negative_peak_height": neg_height,
        "positive_peak_height": pos_height,
        "valley_height_at_zero": valley_height,
        "valley_to_peak_ratio": float(valley_height / peak_height),
        "negative_peak_fwhm": _fwhm(centers, density, neg_peak_idx),
        "positive_peak_fwhm": _fwhm(centers, density, pos_peak_idx),
    }


def _basic_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def plot_histograms(
    *,
    site_values: np.ndarray,
    magnetizations: np.ndarray,
    site_hist: dict,
    mag_hist: dict,
    plot_path: str,
    checkpoint: str,
    sample_info: dict,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.hist(
        site_values,
        bins=np.asarray(site_hist["bin_edges"]),
        density=True,
        histtype="stepfilled",
        alpha=0.65,
        color="#1f77b4",
    )
    ax.set_title("Site-field histogram")
    ax.set_xlabel(r"$\phi(x)$")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(
        magnetizations,
        bins=np.asarray(mag_hist["bin_edges"]),
        density=True,
        histtype="stepfilled",
        alpha=0.65,
        color="#d62728",
    )
    ax.set_title("Magnetization histogram")
    ax.set_xlabel(r"$m = V^{-1}\sum_x \phi(x)$")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "RG coarse-eta Gaussian flow sample histograms\n"
        f"{Path(checkpoint).name}  samples={sample_info['n_samples']}  batch={sample_info['batch_size']}"
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Sample a trained phi^4 flow and plot field/magnetization histograms.")
    ap.add_argument("--resume", type=str, required=True, help="checkpoint to analyze")
    ap.add_argument("--n-samples", type=int, default=64, help="number of full-field samples to draw")
    ap.add_argument("--batch-size", type=int, default=16, help="sampling batch size")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed")
    ap.add_argument("--bins", type=int, default=160, help="number of histogram bins")
    ap.add_argument("--site-quantile", type=float, default=0.995, help="symmetric quantile range for site histogram")
    ap.add_argument(
        "--mag-quantile",
        type=float,
        default=0.995,
        help="symmetric quantile range for magnetization histogram",
    )
    ap.add_argument("--plot-out", type=str, default="phi4_flow_histograms.png", help="output plot path")
    ap.add_argument("--json-out", type=str, default="", help="optional JSON summary path")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.resume)
    cfg = ckpt["cfg"]
    weights = ckpt["weights"]

    fields, sample_info = sample_model_fields(
        cfg=cfg,
        weights=weights,
        n_samples=int(args.n_samples),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    site_values = fields.reshape(-1)
    magnetizations = np.mean(fields, axis=(1, 2, 3))

    site_range = _auto_range(site_values, quantile=float(args.site_quantile))
    mag_range = _auto_range(magnetizations, quantile=float(args.mag_quantile))
    site_hist = _histogram(site_values, bins=int(args.bins), value_range=site_range)
    mag_hist = _histogram(magnetizations, bins=int(args.bins), value_range=mag_range)

    result = {
        "checkpoint": str(args.resume),
        "sample_info": sample_info,
        "field_shape": list(fields.shape),
        "site_stats": _basic_stats(site_values),
        "magnetization_stats": _basic_stats(magnetizations),
        "site_histogram": site_hist,
        "magnetization_histogram": mag_hist,
        "site_double_peak": _double_peak_summary(site_hist),
        "magnetization_double_peak": _double_peak_summary(mag_hist),
        "notes": {
            "site_histogram_site_count": int(site_values.size),
            "magnetization_histogram_config_count": int(magnetizations.size),
            "single_batch_site_hist_ok": bool(int(args.n_samples) <= int(args.batch_size)),
            "single_batch_magnetization_hist_reliable": bool(int(args.n_samples) >= 64),
        },
    }

    plot_histograms(
        site_values=site_values,
        magnetizations=magnetizations,
        site_hist=site_hist,
        mag_hist=mag_hist,
        plot_path=args.plot_out,
        checkpoint=str(args.resume),
        sample_info=sample_info,
    )
    print(f"Wrote {args.plot_out}")
    print("Site double-peak:", json.dumps(result["site_double_peak"], indent=2))
    print("Magnetization double-peak:", json.dumps(result["magnetization_double_peak"], indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

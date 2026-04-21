#!/usr/bin/env python3
"""Level-by-level observable analysis for the RG coarse-eta Gaussian phi^4 flow.

This script draws i.i.d. samples from a trained RG coarse-eta Gaussian flow and
measures the same physics observables used in the HMC campaign on every blocked
level of the hierarchy:

- magnetization m and |m|
- m^2 and m^4
- magnetic susceptibility chi_m
- Binder ratio B4 and Binder cumulant U4
- second-moment correlation length xi2 from k=1
- generalized xi2(k) and low-momentum fits from k=1..k_max

It can also:

- compute fine-level reweighting diagnostics against the target phi^4 action
- report reweighted blocked observables using those fine-level weights
- measure locality of the learned effective action at every RG depth

The samples are independent, so no autocorrelation analysis is performed.
Errors are estimated with a blocked jackknife over the i.i.d. samples.
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
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.models.phi4 import Phi4
from jaxqft.models.phi4_mg import realnvp_f
from jaxqft.models.phi4_rg_cond_flow import _split_rg, _std_normal_log_prob
from jaxqft.models.phi4_rg_coarse_eta_flow import _eta_flow_f
from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
    _eta_gaussian_f,
    _terminal_gaussian_params,
    _triangular_linear_f4,
    init_rg_coarse_eta_gaussian_flow,
    rg_coarse_eta_gaussian_flow_g,
    rg_coarse_eta_gaussian_flow_log_prob,
    rg_coarse_eta_gaussian_flow_prior_sample,
)
from scripts.phi4.analysis.hmc_common import (
    binder_cumulant,
    binder_ratio,
    blocked_jackknife_from_series,
    generalized_second_moment_correlation_length,
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


def _fit_xi2_from_momenta(*, chi_m: float, c2pk: np.ndarray, L: int, momenta_k: np.ndarray, degree: int) -> float:
    if degree not in (1, 2):
        raise ValueError(f"unsupported fit degree {degree}")
    c2 = np.asarray(c2pk, dtype=np.float64).reshape(-1)
    ks = np.asarray(momenta_k, dtype=np.int64).reshape(-1)
    if c2.size != ks.size:
        raise ValueError("momenta_k and c2pk must have the same length")
    xs = []
    ys = []
    for k, ck in zip(ks, c2):
        if k <= 0 or k >= L or (not np.isfinite(ck)) or ck <= 0.0:
            continue
        hat_p = 2.0 * np.sin(np.pi * float(k) / float(L))
        x = hat_p * hat_p
        y = chi_m / ck - 1.0
        if x <= 0.0 or (not np.isfinite(y)) or y <= 0.0:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < degree:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if degree == 1:
        denom = float(np.dot(x, x))
        if denom <= 0.0:
            return float("nan")
        a = float(np.dot(x, y) / denom)
    else:
        A = np.column_stack([x, x * x])
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return float("nan")
        a = float(coeffs[0])
    if not np.isfinite(a) or a <= 0.0:
        return float("nan")
    return float(np.sqrt(a))


def _make_level_observable_fn(shape: tuple[int, int], k_max: int):
    ny, nx = shape
    ks = np.arange(1, min(int(k_max), ny - 1, nx - 1) + 1, dtype=np.int64)
    if ks.size == 0:
        ks = np.asarray([1], dtype=np.int64)
    vol = float(ny * nx)
    phase_y = jnp.exp(
        2j * jnp.pi * np.outer(ks.astype(np.float64), np.arange(ny, dtype=np.float64)) / float(ny)
    ).astype(jnp.complex64)
    phase_x = jnp.exp(
        2j * jnp.pi * np.outer(ks.astype(np.float64), np.arange(nx, dtype=np.float64)) / float(nx)
    ).astype(jnp.complex64)
    phase_y = phase_y.reshape((1, int(ks.size), ny, 1))
    phase_x = phase_x.reshape((1, int(ks.size), 1, nx))

    @jax.jit
    def obs_fn(xx: jax.Array):
        m = jnp.mean(xx, axis=(1, 2))
        xxc = xx.astype(jnp.complex64)
        pky = jnp.mean(xxc[:, None, :, :] * phase_y, axis=(2, 3))
        pkx = jnp.mean(xxc[:, None, :, :] * phase_x, axis=(2, 3))
        c2pk_y = vol * jnp.real(jnp.conj(pky) * pky)
        c2pk_x = vol * jnp.real(jnp.conj(pkx) * pkx)
        return m, c2pk_x, c2pk_y

    return obs_fn, ks


def _reweight_stats(delta_s: np.ndarray) -> dict:
    diff = np.asarray(delta_s, dtype=np.float64).reshape(-1)
    centered = diff - float(np.mean(diff))
    logw = -centered
    logw_shift = logw - float(np.max(logw))
    foo = np.exp(logw_shift)
    w_norm = foo / np.mean(foo)
    ess_fraction = float((np.mean(foo) ** 2) / np.mean(foo * foo))
    return {
        "n_samples": int(diff.size),
        "mean_action_diff": float(np.mean(diff)),
        "std_action_diff": float(np.std(centered)),
        "max_abs_action_diff": float(np.max(np.abs(centered))),
        "mean_reweight": float(np.mean(w_norm)),
        "std_reweight": float(np.std(w_norm)),
        "ess_fraction": ess_fraction,
        "ess_effective_samples": float(ess_fraction * diff.size),
        "max_weight": float(np.max(w_norm)),
        "q95_weight": float(np.quantile(w_norm, 0.95)),
        "q99_weight": float(np.quantile(w_norm, 0.99)),
        "weights_shifted": foo,
        "weights_normalized": w_norm,
    }


def _level_estimator_from_means(L: int, ks: np.ndarray, means: dict[str, float], *, weighted: bool) -> dict[str, float]:
    if weighted:
        w = float(means["w"])
        if not np.isfinite(w) or w <= 0.0:
            return {}
        m = float(means["wm"]) / w
        abs_m = float(means["wabs_m"]) / w
        m2 = float(means["wm2"]) / w
        m4 = float(means["wm4"]) / w
        c2x = np.asarray([float(means[f"wc2x_k{int(k)}"]) / w for k in ks], dtype=np.float64)
        c2y = np.asarray([float(means[f"wc2y_k{int(k)}"]) / w for k in ks], dtype=np.float64)
    else:
        m = float(means["m"])
        abs_m = float(means["abs_m"])
        m2 = float(means["m2"])
        m4 = float(means["m4"])
        c2x = np.asarray([float(means[f"c2x_k{int(k)}"]) for k in ks], dtype=np.float64)
        c2y = np.asarray([float(means[f"c2y_k{int(k)}"]) for k in ks], dtype=np.float64)

    vol = float(L * L)
    chi = vol * (m2 - m * m)
    c2 = 0.5 * (c2x + c2y)
    out = {
        "magnetization": m,
        "abs_magnetization": abs_m,
        "magnetization2": m2,
        "magnetization4": m4,
        "chi_m": chi,
        "binder_ratio": binder_ratio(m2, m4),
        "binder_cumulant": binder_cumulant(m2, m4),
        "C2p_x": float(c2x[0]),
        "C2p_y": float(c2y[0]),
        "C2p": float(c2[0]),
        "xi2_x": generalized_second_moment_correlation_length(L, 1, chi, float(c2x[0])),
        "xi2_y": generalized_second_moment_correlation_length(L, 1, chi, float(c2y[0])),
    }
    xi_vals = [out["xi2_x"], out["xi2_y"]]
    xi_vals = [v for v in xi_vals if np.isfinite(v)]
    out["xi2"] = float(np.mean(xi_vals)) if xi_vals else float("nan")
    out["xi2_over_L"] = float(out["xi2"] / float(L)) if np.isfinite(out["xi2"]) else float("nan")
    for kval, c2x_i, c2y_i, c2_i in zip(ks, c2x, c2y, c2):
        ik = int(kval)
        out[f"xi2_x_k{ik}"] = generalized_second_moment_correlation_length(L, ik, chi, float(c2x_i))
        out[f"xi2_y_k{ik}"] = generalized_second_moment_correlation_length(L, ik, chi, float(c2y_i))
        kv = [out[f"xi2_x_k{ik}"], out[f"xi2_y_k{ik}"]]
        kv = [v for v in kv if np.isfinite(v)]
        out[f"xi2_k{ik}"] = float(np.mean(kv)) if kv else float("nan")
    out["xi2_fit_linear"] = _fit_xi2_from_momenta(chi_m=chi, c2pk=c2, L=L, momenta_k=ks, degree=1)
    out["xi2_fit_quadratic"] = _fit_xi2_from_momenta(chi_m=chi, c2pk=c2, L=L, momenta_k=ks, degree=2)
    return out


def _assemble_level_summary(
    *,
    L: int,
    ks: np.ndarray,
    m: np.ndarray,
    c2xk: np.ndarray,
    c2yk: np.ndarray,
    jk_bin_size: int,
    rw_shifted: np.ndarray | None,
    rw_reportable: bool,
    rw_ess_fraction: float,
):
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    c2xk = np.asarray(c2xk, dtype=np.float64)
    c2yk = np.asarray(c2yk, dtype=np.float64)
    if c2xk.shape != c2yk.shape or c2xk.shape[0] != m.size or c2xk.shape[1] != ks.size:
        raise ValueError("Incompatible level observable shapes")

    m2 = m * m
    m4 = m2 * m2
    abs_m = np.abs(m)
    series = {
        "m": m,
        "abs_m": abs_m,
        "m2": m2,
        "m4": m4,
    }
    for ik, kval in enumerate(ks):
        kval = int(kval)
        series[f"c2x_k{kval}"] = c2xk[:, ik]
        series[f"c2y_k{kval}"] = c2yk[:, ik]

    unweighted_raw = blocked_jackknife_from_series(
        series,
        lambda means: _level_estimator_from_means(L, ks, means, weighted=False),
        block_size=jk_bin_size,
    )

    unweighted = {
        "n_samples": int(m.size),
        "n_jackknife_bins": int(unweighted_raw["n_blocks"]),
        "bin_size": int(jk_bin_size),
        "momenta_k": [int(v) for v in ks.tolist()],
        "magnetization": unweighted_raw["magnetization"],
        "abs_magnetization": unweighted_raw["abs_magnetization"],
        "magnetization2": unweighted_raw["magnetization2"],
        "magnetization4": unweighted_raw["magnetization4"],
        "chi_m": unweighted_raw["chi_m"],
        "binder_ratio": unweighted_raw["binder_ratio"],
        "binder_cumulant": unweighted_raw["binder_cumulant"],
        "C2p_x": unweighted_raw["C2p_x"],
        "C2p_y": unweighted_raw["C2p_y"],
        "C2p": unweighted_raw["C2p"],
        "xi2_x": unweighted_raw["xi2_x"],
        "xi2_y": unweighted_raw["xi2_y"],
        "xi2": unweighted_raw["xi2"],
        "xi2_over_L": unweighted_raw["xi2_over_L"],
        "xi2_fit_linear": unweighted_raw["xi2_fit_linear"],
        "xi2_fit_quadratic": unweighted_raw["xi2_fit_quadratic"],
        "xi2_momentum_scan": [],
    }
    for kval in ks:
        ik = int(kval)
        hat_p = 2.0 * np.sin(np.pi * float(ik) / float(L))
        unweighted["xi2_momentum_scan"].append(
            {
                "k": ik,
                "hat_p2": float(hat_p * hat_p),
                "xi2_x": unweighted_raw[f"xi2_x_k{ik}"],
                "xi2_y": unweighted_raw[f"xi2_y_k{ik}"],
                "xi2": unweighted_raw[f"xi2_k{ik}"],
            }
        )

    reweighted = None
    if rw_shifted is not None:
        rw_shifted = np.asarray(rw_shifted, dtype=np.float64).reshape(-1)
        if rw_shifted.size != m.size:
            raise ValueError("Weight array length does not match observable sample count")
        rw_series = {
            "w": rw_shifted,
            "wm": rw_shifted * m,
            "wabs_m": rw_shifted * abs_m,
            "wm2": rw_shifted * m2,
            "wm4": rw_shifted * m4,
        }
        for ik, kval in enumerate(ks):
            kval = int(kval)
            rw_series[f"wc2x_k{kval}"] = rw_shifted * c2xk[:, ik]
            rw_series[f"wc2y_k{kval}"] = rw_shifted * c2yk[:, ik]
        reweighted_raw = blocked_jackknife_from_series(
            rw_series,
            lambda means: _level_estimator_from_means(L, ks, means, weighted=True),
            block_size=jk_bin_size,
        )
        reweighted = {
            "reportable": bool(rw_reportable),
            "ess_fraction": float(rw_ess_fraction),
            "n_samples": int(m.size),
            "n_jackknife_bins": int(reweighted_raw["n_blocks"]),
            "bin_size": int(jk_bin_size),
            "momenta_k": [int(v) for v in ks.tolist()],
            "magnetization": reweighted_raw.get("magnetization"),
            "abs_magnetization": reweighted_raw.get("abs_magnetization"),
            "magnetization2": reweighted_raw.get("magnetization2"),
            "magnetization4": reweighted_raw.get("magnetization4"),
            "chi_m": reweighted_raw.get("chi_m"),
            "binder_ratio": reweighted_raw.get("binder_ratio"),
            "binder_cumulant": reweighted_raw.get("binder_cumulant"),
            "C2p_x": reweighted_raw.get("C2p_x"),
            "C2p_y": reweighted_raw.get("C2p_y"),
            "C2p": reweighted_raw.get("C2p"),
            "xi2_x": reweighted_raw.get("xi2_x"),
            "xi2_y": reweighted_raw.get("xi2_y"),
            "xi2": reweighted_raw.get("xi2"),
            "xi2_over_L": reweighted_raw.get("xi2_over_L"),
            "xi2_fit_linear": reweighted_raw.get("xi2_fit_linear"),
            "xi2_fit_quadratic": reweighted_raw.get("xi2_fit_quadratic"),
            "xi2_momentum_scan": [],
        }
        for kval in ks:
            ik = int(kval)
            hat_p = 2.0 * np.sin(np.pi * float(ik) / float(L))
            reweighted["xi2_momentum_scan"].append(
                {
                    "k": ik,
                    "hat_p2": float(hat_p * hat_p),
                    "xi2_x": reweighted_raw.get(f"xi2_x_k{ik}"),
                    "xi2_y": reweighted_raw.get(f"xi2_y_k{ik}"),
                    "xi2": reweighted_raw.get(f"xi2_k{ik}"),
                }
            )
    return unweighted, reweighted


def _periodic_shells(shape: tuple[int, int], source: tuple[int, int], metric: str) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    sy, sx = source
    yy, xx = np.indices((ny, nx))
    dy = np.abs(yy - sy)
    dx = np.abs(xx - sx)
    dy = np.minimum(dy, ny - dy)
    dx = np.minimum(dx, nx - dx)
    if metric == "manhattan":
        shells = dy + dx
        radii = np.arange(int(np.max(shells)) + 1, dtype=np.float64)
    elif metric == "chebyshev":
        shells = np.maximum(dy, dx)
        radii = np.arange(int(np.max(shells)) + 1, dtype=np.float64)
    elif metric == "euclidean2":
        shells = dy * dy + dx * dx
        radii = np.sqrt(np.arange(int(np.max(shells)) + 1, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported locality metric: {metric}")
    return shells.reshape(-1).astype(np.int32), radii


def _fit_exponential_decay(radii: np.ndarray, values: np.ndarray, counts: np.ndarray, rmin: float, rmax: float) -> dict:
    mask = (counts > 0) & np.isfinite(values) & (values > 0.0) & (radii >= rmin)
    if rmax > 0.0:
        mask &= radii <= rmax
    mask &= radii > 0.0
    if int(np.count_nonzero(mask)) < 2:
        return {"ok": False, "n_points": int(np.count_nonzero(mask)), "fit_rmin": float(rmin), "fit_rmax": float(rmax)}
    xx = radii[mask]
    yy = np.log(values[mask])
    slope, intercept = np.polyfit(xx, yy, 1)
    return {
        "ok": True,
        "n_points": int(xx.size),
        "fit_rmin": float(rmin),
        "fit_rmax": float(rmax),
        "slope": float(slope),
        "intercept": float(intercept),
        "xi_local": float(-1.0 / slope) if slope < 0.0 else float("inf"),
    }


def _default_locality_primary_rmax(shape: tuple[int, int], radii: np.ndarray) -> float:
    return float(min(float(min(shape)) / 4.0, float(np.max(radii))))


def _dyadic_locality_windows(radii: np.ndarray, shape: tuple[int, int]) -> list[tuple[float, float]]:
    max_r = float(np.max(radii))
    hard_max = min(float(min(shape)) / 2.0, max_r)
    windows = []
    r0 = 1.0
    while r0 < hard_max:
        r1 = min(2.0 * r0, hard_max)
        if r1 > r0:
            windows.append((float(r0), float(r1)))
        if r1 >= hard_max:
            break
        r0 = r1
    return windows


def _locality_weight_summaries(
    *,
    shell_sum_abs: np.ndarray,
    shell_count: np.ndarray,
    onsite: float,
    n_sources_total: int,
) -> dict:
    per_source_abs = shell_sum_abs / max(int(n_sources_total), 1)
    per_source_abs_norm = per_source_abs / max(float(onsite), 1e-30)
    tail_abs_norm = np.cumsum(per_source_abs_norm[::-1])[::-1]
    offsite_total = float(np.sum(per_source_abs_norm[1:])) if per_source_abs_norm.size > 1 else 0.0
    if offsite_total > 0.0:
        tail_fraction = tail_abs_norm / offsite_total
    else:
        tail_fraction = np.full_like(tail_abs_norm, np.nan)
    return {
        "shell_integrated_abs_response_norm": per_source_abs_norm.tolist(),
        "tail_integrated_abs_response_norm": tail_abs_norm.tolist(),
        "tail_fraction_of_offsite_weight": tail_fraction.tolist(),
        "offsite_total_abs_response_norm": offsite_total,
        "nearest_neighbor_abs_response_norm": float(per_source_abs_norm[1]) if per_source_abs_norm.size > 1 else float("nan"),
        "distance2_abs_response_norm": float(per_source_abs_norm[2]) if per_source_abs_norm.size > 2 else float("nan"),
    }


def _locality_fit_bundle(
    *,
    shape: tuple[int, int],
    radii: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    fit_rmin: float,
    fit_rmax: float,
) -> dict:
    max_r = float(np.max(radii))
    user_rmax = float(fit_rmax) if fit_rmax > 0.0 else max_r
    primary_rmax = _default_locality_primary_rmax(shape, radii)
    fits = {
        "user": _fit_exponential_decay(radii, values, counts, fit_rmin, user_rmax),
        "lquarter": _fit_exponential_decay(radii, values, counts, fit_rmin, primary_rmax),
        "full": _fit_exponential_decay(radii, values, counts, fit_rmin, max_r),
    }
    dyadic = []
    for r0, r1 in _dyadic_locality_windows(radii, shape):
        fit = _fit_exponential_decay(radii, values, counts, r0, r1)
        fit["label"] = f"[{r0:g},{r1:g}]"
        dyadic.append(fit)
    return {"fit": fits["user"], "fit_lquarter": fits["lquarter"], "fit_full": fits["full"], "dyadic_fits": dyadic}


def _level_log_prob_from(cfg: dict, weights: dict, x: jax.Array, level: int) -> jax.Array:
    if level >= cfg["depth"] - 1:
        flat = x.reshape((x.shape[0], 4))
        z_after_flow, ldj_flow = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
        z_after_gauss, ldj_gauss = _triangular_linear_f4(z_after_flow, *_terminal_gaussian_params(cfg, weights))
        z_term = z_after_gauss.reshape(x.shape)
        prior = _std_normal_log_prob(z_term, sum_axes=(1, 2))
        return prior + ldj_flow + ldj_gauss

    x_coarse, x_fine = _split_rg(x, cfg["rg_mode"])
    lp_coarse = _level_log_prob_from(cfg, weights, x_coarse, level + 1)
    z_after_flow, ldj_flow = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
    z_after_gauss, ldj_gauss = _eta_gaussian_f(cfg, weights, z_after_flow, x_coarse, level)
    prior = _std_normal_log_prob(z_after_gauss, sum_axes=(1, 2, 3))
    return lp_coarse + prior + ldj_flow + ldj_gauss


def _run_level_locality_analysis(
    *,
    cfg: dict,
    weights: dict,
    level: int,
    shape: tuple[int, int],
    samples: np.ndarray,
    metric: str,
    n_sources: int,
    seed: int,
    fit_rmin: float,
    fit_rmax: float,
) -> dict:
    if samples.ndim != 3:
        raise ValueError(f"Expected samples with shape (nsamples, H, W), got {samples.shape}")

    ny, nx = shape
    vol = ny * nx
    rng = np.random.default_rng(seed + level)

    def action_flat(flat: jax.Array) -> jax.Array:
        xx = flat.reshape((1, ny, nx))
        return -_level_log_prob_from(cfg, weights, xx, level)[0]

    score_flat = jax.grad(action_flat)

    @jax.jit
    def hvp_many(flat: jax.Array, vecs: jax.Array) -> jax.Array:
        return jax.vmap(lambda v: jax.jvp(score_flat, (flat,), (v,))[1])(vecs)

    shell_sum_abs = None
    shell_sum_sq = None
    shell_count = None
    n_sources_total = 0

    for sample in samples:
        flat = jnp.asarray(sample.reshape(-1), dtype=jnp.float32)
        source_inds = rng.choice(vol, size=min(int(n_sources), vol), replace=False)
        vecs = np.zeros((len(source_inds), vol), dtype=np.float32)
        for row, idx in enumerate(source_inds):
            vecs[row, int(idx)] = 1.0
        responses = np.asarray(hvp_many(flat, jnp.asarray(vecs, dtype=jnp.float32)))

        for row, idx in enumerate(source_inds):
            sy = int(idx) // nx
            sx = int(idx) % nx
            shells, radii = _periodic_shells(shape, (sy, sx), metric)
            vals = np.abs(responses[row]).reshape(-1)
            nshell = int(np.max(shells)) + 1
            if shell_sum_abs is None:
                shell_sum_abs = np.zeros((nshell,), dtype=np.float64)
                shell_sum_sq = np.zeros((nshell,), dtype=np.float64)
                shell_count = np.zeros((nshell,), dtype=np.int64)
                radii_out = radii
            shell_sum_abs += np.bincount(shells, weights=vals, minlength=nshell)
            shell_sum_sq += np.bincount(shells, weights=vals * vals, minlength=nshell)
            shell_count += np.bincount(shells, minlength=nshell)
            n_sources_total += 1

    assert shell_sum_abs is not None and shell_sum_sq is not None and shell_count is not None
    mean_abs = shell_sum_abs / np.maximum(shell_count, 1)
    rms = np.sqrt(shell_sum_sq / np.maximum(shell_count, 1))
    onsite = float(mean_abs[0]) if mean_abs[0] > 0.0 else 1.0
    mean_abs_norm = mean_abs / onsite
    rms_norm = rms / max(float(rms[0]), 1e-30)
    fit_bundle = _locality_fit_bundle(
        shape=shape,
        radii=radii_out,
        values=mean_abs_norm,
        counts=shell_count,
        fit_rmin=fit_rmin,
        fit_rmax=fit_rmax,
    )
    weight_summary = _locality_weight_summaries(
        shell_sum_abs=shell_sum_abs,
        shell_count=shell_count,
        onsite=onsite,
        n_sources_total=n_sources_total,
    )
    return {
        "level_from_fine": int(level),
        "shape": [int(ny), int(nx)],
        "metric": metric,
        "n_samples": int(samples.shape[0]),
        "n_sources_per_sample": int(min(int(n_sources), vol)),
        "n_sources_total": int(n_sources_total),
        "radii": radii_out.tolist(),
        "shell_count": shell_count.astype(int).tolist(),
        "mean_abs_response": mean_abs.tolist(),
        "rms_response": rms.tolist(),
        "mean_abs_response_norm": mean_abs_norm.tolist(),
        "rms_response_norm": rms_norm.tolist(),
        **fit_bundle,
        **weight_summary,
    }


def measure_flow_levels(
    *,
    cfg: dict,
    weights: dict,
    lam: float,
    mass: float,
    nsamples: int,
    batch_size: int,
    seed: int,
    jk_bin_size: int,
    k_max: int,
    reweight_ess_min: float,
    locality: bool,
    locality_nsamples: int,
    locality_nsources: int,
    locality_metrics: list[str],
    locality_fit_rmin: float,
    locality_fit_rmax: float,
) -> dict:
    key = jax.random.PRNGKey(seed)
    theory = Phi4([int(cfg["size_h"]), int(cfg["size_w"])], float(lam), float(mass), batch_size=int(batch_size))
    obs_fns: dict[tuple[int, int], tuple[callable, np.ndarray]] = {}
    level_m = None
    level_c2x = None
    level_c2y = None
    level_shapes = None
    delta_s_parts = []
    locality_fields = None

    lp_fn = jax.jit(lambda xx: rg_coarse_eta_gaussian_flow_log_prob({"cfg": cfg, "weights": weights}, xx))
    act_fn = jax.jit(lambda xx: theory.action(xx))

    remaining = int(nsamples)
    sample_tic = time.perf_counter()
    while remaining > 0:
        bsz = min(int(batch_size), remaining)
        key, sub = jax.random.split(key)
        z = rg_coarse_eta_gaussian_flow_prior_sample(sub, cfg, bsz)
        x = rg_coarse_eta_gaussian_flow_g(cfg, z, weights)
        lp = np.asarray(lp_fn(x))
        act = np.asarray(act_fn(x))
        delta_s_parts.append(lp + act)
        fields = _collect_level_fields(x, int(cfg["rg_mode"]))

        if level_m is None:
            nlevels = len(fields)
            level_m = [[] for _ in range(nlevels)]
            level_c2x = [[] for _ in range(nlevels)]
            level_c2y = [[] for _ in range(nlevels)]
            level_shapes = [tuple(int(v) for v in fld.shape[1:3]) for fld in fields]
            locality_fields = [[] for _ in range(nlevels)] if locality else None

        for level, fld in enumerate(fields):
            shape = tuple(int(v) for v in fld.shape[1:3])
            if shape not in obs_fns:
                obs_fns[shape] = _make_level_observable_fn(shape, k_max)
            obs_fn, _ = obs_fns[shape]
            m, c2pk_x, c2pk_y = obs_fn(fld)
            level_m[level].append(np.asarray(m))
            level_c2x[level].append(np.asarray(c2pk_x))
            level_c2y[level].append(np.asarray(c2pk_y))

            if locality_fields is not None:
                already = sum(chunk.shape[0] for chunk in locality_fields[level])
                need = max(0, int(locality_nsamples) - already)
                if need > 0:
                    locality_fields[level].append(np.asarray(fld[:need]))
        remaining -= bsz
    sample_sec = time.perf_counter() - sample_tic

    assert level_shapes is not None and level_m is not None and level_c2x is not None and level_c2y is not None
    delta_s = np.concatenate(delta_s_parts, axis=0)
    rw_stats = _reweight_stats(delta_s)
    rw_shifted = rw_stats.pop("weights_shifted")
    rw_stats.pop("weights_normalized")
    rw_reportable = bool(rw_stats["ess_fraction"] >= float(reweight_ess_min))

    levels_out = []
    locality_by_metric = {metric: [] for metric in locality_metrics} if locality else {}
    nlevels = len(level_shapes)
    for level in range(nlevels):
        m = np.concatenate(level_m[level], axis=0)
        c2x = np.concatenate(level_c2x[level], axis=0)
        c2y = np.concatenate(level_c2y[level], axis=0)
        L = int(level_shapes[level][0])
        ks = obs_fns[level_shapes[level]][1]
        unweighted, reweighted = _assemble_level_summary(
            L=L,
            ks=ks,
            m=m,
            c2xk=c2x,
            c2yk=c2y,
            jk_bin_size=jk_bin_size,
            rw_shifted=rw_shifted,
            rw_reportable=rw_reportable,
            rw_ess_fraction=float(rw_stats["ess_fraction"]),
        )
        row = {
            "level_from_fine": int(level),
            "distance_from_bottom": int(nlevels - 1 - level),
            "shape": [int(level_shapes[level][0]), int(level_shapes[level][1])],
            "L": int(L),
            "unweighted": unweighted,
        }
        if reweighted is not None:
            row["reweighted"] = reweighted
        levels_out.append(row)

        if locality_fields is not None:
            fields_level = np.concatenate(locality_fields[level], axis=0) if locality_fields[level] else np.empty((0, L, L))
            if fields_level.shape[0] > 0:
                for metric in locality_metrics:
                    locality_by_metric[metric].append(
                        _run_level_locality_analysis(
                            cfg=cfg,
                            weights=weights,
                            level=level,
                            shape=level_shapes[level],
                            samples=fields_level,
                            metric=metric,
                            n_sources=locality_nsources,
                            seed=seed,
                            fit_rmin=locality_fit_rmin,
                            fit_rmax=locality_fit_rmax,
                        )
                    )

    for level in range(1, nlevels):
        prev = levels_out[level - 1]["unweighted"]["xi2"]["mean"]
        cur = levels_out[level]["unweighted"]["xi2"]["mean"]
        ratio = float(cur / prev) if np.isfinite(cur) and np.isfinite(prev) and prev != 0.0 else float("nan")
        levels_out[level]["unweighted"]["xi2_over_prev_level"] = ratio
        prev_rw_row = levels_out[level - 1].get("reweighted")
        cur_rw_row = levels_out[level].get("reweighted")
        prev_rw_xi2 = prev_rw_row.get("xi2") if isinstance(prev_rw_row, dict) else None
        cur_rw_xi2 = cur_rw_row.get("xi2") if isinstance(cur_rw_row, dict) else None
        if isinstance(prev_rw_xi2, dict) and isinstance(cur_rw_xi2, dict):
            prev_rw = prev_rw_xi2.get("mean")
            cur_rw = cur_rw_xi2.get("mean")
            ratio_rw = (
                float(cur_rw / prev_rw) if np.isfinite(cur_rw) and np.isfinite(prev_rw) and prev_rw != 0.0 else float("nan")
            )
            cur_rw_row["xi2_over_prev_level"] = ratio_rw

    out = {
        "sample_source": {
            "source": "model",
            "n_samples": int(nsamples),
            "batch_size": int(batch_size),
            "super_batches": int((int(nsamples) + int(batch_size) - 1) // int(batch_size)),
            "seed": int(seed),
            "sample_sec": float(sample_sec),
        },
        "reweighting": {
            **rw_stats,
            "reportable": bool(rw_reportable),
            "reportable_min_ess_fraction": float(reweight_ess_min),
        },
        "n_levels": int(nlevels),
        "levels": levels_out,
    }
    if locality:
        primary_metric = str(locality_metrics[0])
        out["locality"] = {
            "n_samples_per_level": int(locality_nsamples),
            "n_sources": int(locality_nsources),
            "metric": primary_metric,
            "levels": locality_by_metric[primary_metric],
            "metrics": {
                metric: {
                    "n_samples_per_level": int(locality_nsamples),
                    "n_sources": int(locality_nsources),
                    "metric": metric,
                    "levels": locality_by_metric[metric],
                }
                for metric in locality_metrics
            },
        }
    return out


def _format_est(obs: dict | None, key: str) -> str:
    if not obs or key not in obs or obs[key] is None:
        return "N/A"
    row = obs[key]
    if not isinstance(row, dict):
        return "N/A"
    mean = row.get("mean", float("nan"))
    err = row.get("stderr", float("nan"))
    if not np.isfinite(mean):
        return "N/A"
    if np.isfinite(err):
        return f"{mean:.6f} +/- {err:.6f}"
    return f"{mean:.6f}"


def _parse_metric_list(spec: str) -> list[str]:
    metrics = []
    for part in str(spec).split(","):
        metric = part.strip()
        if not metric:
            continue
        if metric not in {"manhattan", "chebyshev", "euclidean2"}:
            raise ValueError(f"unsupported locality metric {metric}")
        if metric not in metrics:
            metrics.append(metric)
    if not metrics:
        raise ValueError("no locality metrics specified")
    return metrics


def _print_summary(result: dict):
    rw = result["reweighting"]
    print("\nFine-level reweighting:")
    print(
        "  "
        f"std(DeltaS)={rw['std_action_diff']:.6f}  "
        f"std(DeltaS)/L={rw['std_action_diff']/float(result['levels'][0]['L']):.6f}  "
        f"std(w)={rw['std_reweight']:.6f}  "
        f"ESS={rw['ess_fraction']:.6f}  "
        f"N_eff={rw['ess_effective_samples']:.1f}  "
        f"reweighted_reportable={rw['reportable']}"
    )

    print("\nPer-level observable summary:")
    print(
        f"{'lvl':>3}  {'L':>5}  {'xi2':>22}  {'chi':>22}  {'U4':>22}  {'xi2_rw':>22}  {'chi_rw':>22}  {'U4_rw':>22}"
    )
    for row in result["levels"]:
        rw_row = row.get("reweighted")
        if rw_row is not None and not bool(rw_row.get("reportable", False)):
            rw_row = None
        print(
            f"{row['level_from_fine']:3d}  {row['L']:5d}  "
            f"{_format_est(row['unweighted'], 'xi2'):>22}  "
            f"{_format_est(row['unweighted'], 'chi_m'):>22}  "
            f"{_format_est(row['unweighted'], 'binder_cumulant'):>22}  "
            f"{_format_est(rw_row, 'xi2'):>22}  "
            f"{_format_est(rw_row, 'chi_m'):>22}  "
            f"{_format_est(rw_row, 'binder_cumulant'):>22}"
        )

    locality = result.get("locality")
    if locality:
        metrics = locality.get("metrics") or {locality["metric"]: locality}
        for metric, payload in metrics.items():
            print(f"\nLocality by level ({metric}):")
            print(
                f"{'lvl':>3}  {'L':>5}  {'xi_L/4':>12}  {'xi_full':>12}  {'w(r>=2)':>12}  {'w(r>=L/4)':>12}"
            )
            for row in payload["levels"]:
                fit_q = row.get("fit_lquarter", {})
                fit_full = row.get("fit_full", {})
                xi_q = fit_q.get("xi_local", float("nan")) if fit_q.get("ok", False) else float("nan")
                xi_full = fit_full.get("xi_local", float("nan")) if fit_full.get("ok", False) else float("nan")
                tail = np.asarray(row.get("tail_integrated_abs_response_norm", []), dtype=np.float64)
                L = int(row["shape"][0])
                tail_r2 = float(tail[2]) if tail.size > 2 else float("nan")
                cutoff = min(int(np.floor(L / 4)), int(tail.size - 1)) if tail.size > 0 else 0
                tail_lq = float(tail[cutoff]) if cutoff > 0 else float("nan")
                print(f"{row['level_from_fine']:3d}  {L:5d}  {xi_q:12.6f}  {xi_full:12.6f}  {tail_r2:12.6e}  {tail_lq:12.6e}")


def main():
    ap = argparse.ArgumentParser(description="Measure flow observables at every RG level for the coarse-eta Gaussian flow.")
    ap.add_argument("--resume", type=str, required=True, help="checkpoint to analyze")
    ap.add_argument("--nsamples", type=int, default=4096, help="total number of i.i.d. model samples")
    ap.add_argument("--batch-size", type=int, default=256, help="model sampling batch size")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed for model sampling")
    ap.add_argument("--jk-bin-size", type=int, default=64, help="jackknife bin size in i.i.d. samples")
    ap.add_argument("--k-max", type=int, default=1, help="measure C(p_k) and xi2(k) for k=1..k_max")
    ap.add_argument("--reweight-ess-min", type=float, default=0.05, help="minimum ESS fraction required to report reweighted observables")
    ap.add_argument("--locality", action="store_true", help="also measure locality of the learned effective action at each level")
    ap.add_argument("--locality-nsamples", type=int, default=8, help="number of full fields per level for locality analysis")
    ap.add_argument("--locality-nsources", type=int, default=4, help="number of Hessian probe sources per field for locality")
    ap.add_argument(
        "--locality-metric",
        type=str,
        default="manhattan",
        help="comma-separated locality metrics from {manhattan,chebyshev,euclidean2}",
    )
    ap.add_argument("--locality-fit-rmin", type=float, default=1.0)
    ap.add_argument("--locality-fit-rmax", type=float, default=0.0)
    ap.add_argument("--json-out", type=str, default="", help="optional JSON output path")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    ckpt = load_checkpoint(args.resume)
    arch = ckpt["arch"]
    cfg = _build_model_from_arch(arch)
    weights = ckpt["weights"]
    locality_metrics = _parse_metric_list(args.locality_metric) if bool(args.locality) else []

    result = measure_flow_levels(
        cfg=cfg,
        weights=weights,
        lam=float(arch["lam"]),
        mass=float(arch["mass"]),
        nsamples=int(args.nsamples),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        jk_bin_size=int(args.jk_bin_size),
        k_max=int(args.k_max),
        reweight_ess_min=float(args.reweight_ess_min),
        locality=bool(args.locality),
        locality_nsamples=int(args.locality_nsamples),
        locality_nsources=int(args.locality_nsources),
        locality_metrics=locality_metrics,
        locality_fit_rmin=float(args.locality_fit_rmin),
        locality_fit_rmax=float(args.locality_fit_rmax),
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

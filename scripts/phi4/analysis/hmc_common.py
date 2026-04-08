"""Shared HMC analysis helpers for phi^4 observables."""

from __future__ import annotations

from typing import Callable

import numpy as np

from jaxqft.stats import integrated_autocorr_time


def second_moment_correlation_length(L: int, chi_m: float, c2p: float) -> float:
    """Return the second-moment correlation length for one lattice direction."""
    if L <= 1 or (not np.isfinite(chi_m)) or (not np.isfinite(c2p)) or c2p <= 0.0:
        return float("nan")
    ratio = chi_m / c2p - 1.0
    if ratio <= 0.0:
        return float("nan")
    return float(np.sqrt(ratio) / (2.0 * np.sin(np.pi / float(L))))


def binder_ratio(m2: float, m4: float) -> float:
    """Binder ratio B4 = <m^4> / <m^2>^2."""
    if (not np.isfinite(m2)) or (not np.isfinite(m4)) or m2 <= 0.0:
        return float("nan")
    return float(m4 / (m2 * m2))


def binder_cumulant(m2: float, m4: float) -> float:
    """Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2)."""
    b4 = binder_ratio(m2, m4)
    if not np.isfinite(b4):
        return float("nan")
    return float(1.0 - b4 / 3.0)


def _as_obs2d(obs2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D observable history, got shape {arr.shape}")
    return arr


def batched_mean_series(obs2d: np.ndarray) -> np.ndarray:
    """Return the batch-mean time series from a `(nmeas, batch)` history."""
    arr = _as_obs2d(obs2d)
    return np.asarray(np.mean(arr, axis=1), dtype=np.float64)


def summarize_batched_series(
    obs2d: np.ndarray,
    *,
    iat_method: str = "gamma",
    iat_c: float = 5.0,
    max_lag: int | None = None,
) -> dict:
    """Return mean, Gamma-analysis error, tau_int, and ESS for a batched history."""
    arr = _as_obs2d(obs2d)
    batch_mean = batched_mean_series(arr)
    grand_mean = float(np.mean(batch_mean))
    iat = integrated_autocorr_time(batch_mean, method=iat_method, c=float(iat_c), max_lag=max_lag)
    sigma_mean = float(iat.get("sigma_mean", np.nan))
    tau_int = float(iat.get("tau_int", np.nan))
    ess_total = float(iat.get("ess", np.nan)) * arr.shape[1] if np.isfinite(iat.get("ess", np.nan)) else float("nan")
    return {
        "mean": grand_mean,
        "sigma": sigma_mean,
        "tau_int": tau_int,
        "ess": ess_total,
        "window": int(iat.get("window", 0)),
        "method": str(iat_method),
        "ok": bool(iat.get("ok", True)),
        "message": str(iat.get("message", "ok")),
        "nmeas": int(arr.shape[0]),
        "batch_size": int(arr.shape[1]),
    }


def choose_block_size(
    nmeas: int,
    tau_ref: float,
    *,
    multiple: float = 2.0,
    min_blocks: int = 8,
) -> int:
    """Choose a contiguous jackknife block size from a reference IAT."""
    n = int(nmeas)
    if n <= 1:
        return 1
    if np.isfinite(tau_ref) and tau_ref > 0.0:
        block = int(np.ceil(float(multiple) * float(tau_ref)))
    else:
        block = 1
    block = max(1, block)
    nblocks = n // block
    if nblocks >= min_blocks:
        return block
    if n >= min_blocks:
        return max(1, n // min_blocks)
    return max(1, n // 2)


def blocked_jackknife(
    samples: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Contiguous blocked leave-one-out jackknife replicas."""
    arr = np.asarray(samples, dtype=np.float64)
    nmeas = int(arr.shape[0])
    bsz = max(1, int(block_size))
    nblock = nmeas // bsz
    if nblock < 2:
        raise ValueError(f"Need at least 2 blocks for jackknife; got nmeas={nmeas}, block_size={bsz}")
    ntrim = int(nblock * bsz)
    trimmed = np.asarray(arr[:ntrim], dtype=np.float64)
    blocks = trimmed.reshape((nblock, bsz) + trimmed.shape[1:]).mean(axis=1)
    block_sum = np.sum(blocks, axis=0)
    jk = np.empty_like(blocks)
    for ib in range(nblock):
        jk[ib] = (block_sum - blocks[ib]) / float(nblock - 1)
    full = trimmed.mean(axis=0)
    return full, jk, int(nblock), int(ntrim)


def jackknife_error(jk_vals: np.ndarray, full_val: np.ndarray) -> np.ndarray:
    """Jackknife standard error from leave-one-out replicas."""
    arr = np.asarray(jk_vals, dtype=np.float64)
    center = np.asarray(full_val, dtype=np.float64)
    nb = int(arr.shape[0])
    if nb <= 1:
        return np.full_like(center, np.nan, dtype=np.float64)
    dif = arr - center
    return np.sqrt((nb - 1) / float(nb) * np.sum(dif * dif, axis=0))


def blocked_jackknife_from_series(
    series: dict[str, np.ndarray],
    estimator: Callable[[dict[str, float]], float | dict[str, float]],
    *,
    block_size: int,
) -> dict:
    """Blocked jackknife for scalar or dict-valued estimators built from 1D series."""
    if not series:
        raise ValueError("need at least one input series")

    block_means: dict[str, np.ndarray] = {}
    ntrim_ref: int | None = None
    nblocks_ref: int | None = None
    for key, values in series.items():
        full, jk, nblock, ntrim = blocked_jackknife(np.asarray(values, dtype=np.float64), block_size)
        block_means[key] = jk
        if ntrim_ref is None:
            ntrim_ref = ntrim
            nblocks_ref = nblock
        else:
            if ntrim != ntrim_ref or nblock != nblocks_ref:
                raise ValueError("all series must have the same trimmed length and block count")
        if full.ndim != 0:
            raise ValueError(f"series {key!r} must be 1D")

    nblocks = int(nblocks_ref or 0)
    full_means = {k: float(np.mean(np.asarray(v[: ntrim_ref], dtype=np.float64))) for k, v in series.items()}
    full = estimator(full_means)

    loo = []
    for i in range(nblocks):
        loo_means = {k: float(block_means[k][i]) for k in block_means}
        loo.append(estimator(loo_means))

    if isinstance(full, dict):
        out: dict[str, dict | int] = {
            "block_size": int(block_size),
            "n_blocks": int(nblocks),
            "n_trimmed": int(ntrim_ref or 0),
        }
        for key in full:
            vals = np.asarray([float(v[key]) for v in loo], dtype=np.float64)
            out[key] = {
                "mean": float(full[key]),
                "stderr": float(jackknife_error(vals, np.asarray(float(full[key])))),
            }
        return out

    vals = np.asarray([float(v) for v in loo], dtype=np.float64)
    return {
        "mean": float(full),
        "stderr": float(jackknife_error(vals, np.asarray(float(full)))),
        "block_size": int(block_size),
        "n_blocks": int(nblocks),
        "n_trimmed": int(ntrim_ref or 0),
    }


def phi4_summary_from_histories(
    *,
    shape: tuple[int, int],
    magnetization: np.ndarray,
    energy_density: np.ndarray | None = None,
    c2p_x: np.ndarray,
    c2p_y: np.ndarray | None = None,
    iat_method: str = "gamma",
    iat_c: float = 5.0,
    block_size: int = 0,
) -> dict:
    """Analyze phi^4 HMC histories with Gamma errors and blocked jackknife derived errors."""
    m = _as_obs2d(magnetization)
    e = _as_obs2d(energy_density) if energy_density is not None else None
    c2x = _as_obs2d(c2p_x)
    c2y = _as_obs2d(c2p_y) if c2p_y is not None else c2x
    if c2x.shape != m.shape or c2y.shape != m.shape or (e is not None and e.shape != m.shape):
        raise ValueError("all observable histories must have the same (nmeas, batch) shape")

    vol = int(np.prod(shape))
    abs_m = np.abs(m)
    m2 = m * m
    m4 = m2 * m2
    c2 = 0.5 * (c2x + c2y)

    primitive = {
        "magnetization": summarize_batched_series(m, iat_method=iat_method, iat_c=iat_c),
        "abs_magnetization": summarize_batched_series(abs_m, iat_method=iat_method, iat_c=iat_c),
        "magnetization2": summarize_batched_series(m2, iat_method=iat_method, iat_c=iat_c),
        "magnetization4": summarize_batched_series(m4, iat_method=iat_method, iat_c=iat_c),
        "C2p_x": summarize_batched_series(c2x, iat_method=iat_method, iat_c=iat_c),
        "C2p_y": summarize_batched_series(c2y, iat_method=iat_method, iat_c=iat_c),
        "C2p": summarize_batched_series(c2, iat_method=iat_method, iat_c=iat_c),
    }
    if e is not None:
        primitive["energy_density"] = summarize_batched_series(e, iat_method=iat_method, iat_c=iat_c)

    tau_candidates = [
        primitive["magnetization2"]["tau_int"],
        primitive["C2p"]["tau_int"],
    ]
    if e is not None:
        tau_candidates.append(primitive["energy_density"]["tau_int"])
    tau_ref = max((float(t) for t in tau_candidates if np.isfinite(t)), default=float("nan"))
    block_size_eff = int(block_size) if int(block_size) > 0 else choose_block_size(int(m.shape[0]), tau_ref)

    def _derived_from_means(means: dict[str, float]) -> dict[str, float]:
        chi = float(vol) * (float(means["m2"]) - float(means["m"]) * float(means["m"]))
        b4 = binder_ratio(float(means["m2"]), float(means["m4"]))
        u4 = binder_cumulant(float(means["m2"]), float(means["m4"]))
        xi_x = second_moment_correlation_length(int(shape[0]), chi, float(means["c2x"]))
        xi_y = second_moment_correlation_length(int(shape[1]), chi, float(means["c2y"]))
        xi_vals = [v for v in (xi_x, xi_y) if np.isfinite(v)]
        xi = float(np.mean(xi_vals)) if xi_vals else float("nan")
        return {
            "chi_m": chi,
            "binder_ratio": b4,
            "binder_cumulant": u4,
            "xi2_x": xi_x,
            "xi2_y": xi_y,
            "xi2": xi,
            "xi2_over_L": float(xi / float(shape[0])) if np.isfinite(xi) else float("nan"),
        }

    derived = blocked_jackknife_from_series(
        {
            "m": batched_mean_series(m),
            "m2": batched_mean_series(m2),
            "m4": batched_mean_series(m4),
            "c2x": batched_mean_series(c2x),
            "c2y": batched_mean_series(c2y),
        },
        _derived_from_means,
        block_size=block_size_eff,
    )

    return {
        "shape": [int(v) for v in shape],
        "volume": int(vol),
        "iat_method": str(iat_method),
        "iat_c": float(iat_c),
        "block_size": int(block_size_eff),
        "tau_ref": float(tau_ref),
        "primitive": primitive,
        "derived": derived,
    }

"""Shared HMC analysis helpers for phi^4 observables."""

from __future__ import annotations

from typing import Callable

import numpy as np

from jaxqft.stats import integrated_autocorr_time


def second_moment_correlation_length(L: int, chi_m: float, c2p: float) -> float:
    """Return the second-moment correlation length for one lattice direction."""
    return generalized_second_moment_correlation_length(L=L, k=1, chi_m=chi_m, c2pk=c2p)


def generalized_second_moment_correlation_length(L: int, k: int, chi_m: float, c2pk: float) -> float:
    """Return the generalized second-moment estimator using momentum mode `k`."""
    if L <= 1 or k <= 0 or k >= L or (not np.isfinite(chi_m)) or (not np.isfinite(c2pk)) or c2pk <= 0.0:
        return float("nan")
    hat_p = 2.0 * np.sin(np.pi * float(k) / float(L))
    if hat_p <= 0.0:
        return float("nan")
    ratio = chi_m / c2pk - 1.0
    if ratio <= 0.0:
        return float("nan")
    return float(np.sqrt(ratio) / hat_p)


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


def _as_obs3d(obs: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected {name} to have shape (nmeas, batch, nk) or (nmeas, batch), got {arr.shape}")
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


def _fit_xi2_from_momenta(
    *,
    chi_m: float,
    c2pk: np.ndarray,
    L: int,
    momenta_k: np.ndarray,
    degree: int,
) -> float:
    """Fit chi/C(p_k)-1 vs hat_p_k^2 and return sqrt(a), where a is xi^2."""
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
    momenta_k: np.ndarray | None = None,
    iat_method: str = "gamma",
    iat_c: float = 5.0,
    block_size: int = 0,
) -> dict:
    """Analyze phi^4 HMC histories with Gamma errors and blocked jackknife derived errors."""
    m = _as_obs2d(magnetization)
    e = _as_obs2d(energy_density) if energy_density is not None else None
    c2xk = _as_obs3d(c2p_x, name="c2p_x")
    c2yk = _as_obs3d(c2p_y, name="c2p_y") if c2p_y is not None else c2xk
    if c2xk.shape[:2] != m.shape or c2yk.shape[:2] != m.shape or c2xk.shape != c2yk.shape or (e is not None and e.shape != m.shape):
        raise ValueError("all observable histories must have compatible shapes")

    nk = int(c2xk.shape[2])
    if momenta_k is None:
        ks = np.arange(1, nk + 1, dtype=np.int64)
    else:
        ks = np.asarray(momenta_k, dtype=np.int64).reshape(-1)
        if ks.size != nk:
            raise ValueError(f"momenta_k has length {ks.size}, but c2p histories have nk={nk}")

    vol = int(np.prod(shape))
    abs_m = np.abs(m)
    m2 = m * m
    m4 = m2 * m2
    c2x = c2xk[:, :, 0]
    c2y = c2yk[:, :, 0]
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
    if nk > 1:
        primitive["C2p_momentum_scan"] = []
        for ik, kval in enumerate(ks):
            c2x_i = c2xk[:, :, ik]
            c2y_i = c2yk[:, :, ik]
            c2_i = 0.5 * (c2x_i + c2y_i)
            primitive["C2p_momentum_scan"].append(
                {
                    "k": int(kval),
                    "C2p_x": summarize_batched_series(c2x_i, iat_method=iat_method, iat_c=iat_c),
                    "C2p_y": summarize_batched_series(c2y_i, iat_method=iat_method, iat_c=iat_c),
                    "C2p": summarize_batched_series(c2_i, iat_method=iat_method, iat_c=iat_c),
                }
            )

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
        xi_x = generalized_second_moment_correlation_length(int(shape[0]), 1, chi, float(means["c2x_k1"]))
        xi_y = generalized_second_moment_correlation_length(int(shape[1]), 1, chi, float(means["c2y_k1"]))
        xi_vals = [v for v in (xi_x, xi_y) if np.isfinite(v)]
        xi = float(np.mean(xi_vals)) if xi_vals else float("nan")
        out = {
            "chi_m": chi,
            "binder_ratio": b4,
            "binder_cumulant": u4,
            "xi2_x": xi_x,
            "xi2_y": xi_y,
            "xi2": xi,
            "xi2_over_L": float(xi / float(shape[0])) if np.isfinite(xi) else float("nan"),
        }
        c2x_vals = np.asarray([float(means[f"c2x_k{kval}"]) for kval in ks], dtype=np.float64)
        c2y_vals = np.asarray([float(means[f"c2y_k{kval}"]) for kval in ks], dtype=np.float64)
        c2_vals = 0.5 * (c2x_vals + c2y_vals)
        for kval, c2x_i, c2y_i, c2_i in zip(ks, c2x_vals, c2y_vals, c2_vals):
            out[f"xi2_x_k{int(kval)}"] = generalized_second_moment_correlation_length(int(shape[0]), int(kval), chi, float(c2x_i))
            out[f"xi2_y_k{int(kval)}"] = generalized_second_moment_correlation_length(int(shape[1]), int(kval), chi, float(c2y_i))
            xv = [out[f"xi2_x_k{int(kval)}"], out[f"xi2_y_k{int(kval)}"]]
            xv = [v for v in xv if np.isfinite(v)]
            out[f"xi2_k{int(kval)}"] = float(np.mean(xv)) if xv else float("nan")
        out["xi2_fit_linear"] = _fit_xi2_from_momenta(
            chi_m=chi,
            c2pk=c2_vals,
            L=int(shape[0]),
            momenta_k=ks,
            degree=1,
        )
        out["xi2_fit_quadratic"] = _fit_xi2_from_momenta(
            chi_m=chi,
            c2pk=c2_vals,
            L=int(shape[0]),
            momenta_k=ks,
            degree=2,
        )
        return out
    series = {
        "m": batched_mean_series(m),
        "m2": batched_mean_series(m2),
        "m4": batched_mean_series(m4),
    }
    for ik, kval in enumerate(ks):
        series[f"c2x_k{int(kval)}"] = batched_mean_series(c2xk[:, :, ik])
        series[f"c2y_k{int(kval)}"] = batched_mean_series(c2yk[:, :, ik])
    derived_raw = blocked_jackknife_from_series(series, _derived_from_means, block_size=block_size_eff)

    derived = {
        "block_size": int(derived_raw["block_size"]),
        "n_blocks": int(derived_raw["n_blocks"]),
        "n_trimmed": int(derived_raw["n_trimmed"]),
        "chi_m": derived_raw["chi_m"],
        "binder_ratio": derived_raw["binder_ratio"],
        "binder_cumulant": derived_raw["binder_cumulant"],
        "xi2_x": derived_raw["xi2_x"],
        "xi2_y": derived_raw["xi2_y"],
        "xi2": derived_raw["xi2"],
        "xi2_over_L": derived_raw["xi2_over_L"],
        "xi2_fit_linear": derived_raw["xi2_fit_linear"],
        "xi2_fit_quadratic": derived_raw["xi2_fit_quadratic"],
        "xi2_momentum_scan": [],
    }
    for kval in ks:
        ikey = int(kval)
        hat_p_x = 2.0 * np.sin(np.pi * float(ikey) / float(shape[0]))
        hat_p_y = 2.0 * np.sin(np.pi * float(ikey) / float(shape[1]))
        derived["xi2_momentum_scan"].append(
            {
                "k": ikey,
                "hat_p2_x": float(hat_p_x * hat_p_x),
                "hat_p2_y": float(hat_p_y * hat_p_y),
                "xi2_x": derived_raw[f"xi2_x_k{ikey}"],
                "xi2_y": derived_raw[f"xi2_y_k{ikey}"],
                "xi2": derived_raw[f"xi2_k{ikey}"],
            }
        )

    return {
        "shape": [int(v) for v in shape],
        "volume": int(vol),
        "momenta_k": [int(v) for v in ks.tolist()],
        "iat_method": str(iat_method),
        "iat_c": float(iat_c),
        "block_size": int(block_size_eff),
        "tau_ref": float(tau_ref),
        "primitive": primitive,
        "derived": derived,
    }


def phi4_level_unweighted_from_summary(summary: dict) -> dict:
    """Convert an HMC level summary to the flow-style per-level observable shape."""
    primitive = summary["primitive"]
    derived = summary["derived"]
    out = {
        "n_measurements": int(primitive["magnetization"]["nmeas"]),
        "batch_size": int(primitive["magnetization"]["batch_size"]),
        "momenta_k": [int(v) for v in summary["momenta_k"]],
        "magnetization": {"mean": float(primitive["magnetization"]["mean"]), "stderr": float(primitive["magnetization"]["sigma"])},
        "abs_magnetization": {
            "mean": float(primitive["abs_magnetization"]["mean"]),
            "stderr": float(primitive["abs_magnetization"]["sigma"]),
        },
        "magnetization2": {"mean": float(primitive["magnetization2"]["mean"]), "stderr": float(primitive["magnetization2"]["sigma"])},
        "magnetization4": {"mean": float(primitive["magnetization4"]["mean"]), "stderr": float(primitive["magnetization4"]["sigma"])},
        "chi_m": dict(derived["chi_m"]),
        "binder_ratio": dict(derived["binder_ratio"]),
        "binder_cumulant": dict(derived["binder_cumulant"]),
        "C2p_x": {"mean": float(primitive["C2p_x"]["mean"]), "stderr": float(primitive["C2p_x"]["sigma"])},
        "C2p_y": {"mean": float(primitive["C2p_y"]["mean"]), "stderr": float(primitive["C2p_y"]["sigma"])},
        "C2p": {"mean": float(primitive["C2p"]["mean"]), "stderr": float(primitive["C2p"]["sigma"])},
        "xi2_x": dict(derived["xi2_x"]),
        "xi2_y": dict(derived["xi2_y"]),
        "xi2": dict(derived["xi2"]),
        "xi2_over_L": dict(derived["xi2_over_L"]),
        "xi2_fit_linear": dict(derived["xi2_fit_linear"]),
        "xi2_fit_quadratic": dict(derived["xi2_fit_quadratic"]),
        "xi2_momentum_scan": [dict(row) for row in derived.get("xi2_momentum_scan", [])],
    }
    return out


def phi4_multilevel_summaries_from_payload(
    payload: dict[str, np.ndarray],
    *,
    iat_method: str = "gamma",
    iat_c: float = 5.0,
    block_size: int = 0,
) -> dict:
    """Summarize a level-0 or blocked-multilevel phi^4 HMC payload."""
    if "shape" not in payload or "magnetization" not in payload:
        raise ValueError("payload missing required level-0 history keys")

    if "blocked_level_shapes" in payload:
        shapes = np.asarray(payload["blocked_level_shapes"], dtype=np.int64)
        if shapes.ndim != 2 or shapes.shape[1] != 2:
            raise ValueError(f"blocked_level_shapes must have shape (n_levels, 2), got {shapes.shape}")
        n_levels = int(np.asarray(payload.get("blocked_n_levels", shapes.shape[0])).reshape(-1)[0])
        if n_levels != int(shapes.shape[0]):
            raise ValueError("blocked_n_levels does not match blocked_level_shapes")
    else:
        shapes = np.asarray(payload["shape"], dtype=np.int64).reshape(1, -1)
        n_levels = 1

    levels = []
    for level in range(n_levels):
        prefix = f"level{level}_"
        shape = tuple(int(v) for v in shapes[level].tolist())
        magnetization = payload.get(prefix + "magnetization", payload["magnetization"] if level == 0 else None)
        if magnetization is None:
            raise ValueError(f"missing magnetization history for level {level}")
        energy_density = payload.get(prefix + "energy_density", payload["energy_density"] if level == 0 and "energy_density" in payload else None)
        if prefix + "c2pk_x" in payload:
            c2p_x = payload[prefix + "c2pk_x"]
        elif level == 0 and "c2pk_x" in payload:
            c2p_x = payload["c2pk_x"]
        elif level == 0 and "c2p_x" in payload:
            c2p_x = payload["c2p_x"]
        else:
            raise ValueError(f"missing c2p_x history for level {level}")
        if prefix + "c2pk_y" in payload:
            c2p_y = payload[prefix + "c2pk_y"]
        elif level == 0 and "c2pk_y" in payload:
            c2p_y = payload["c2pk_y"]
        elif level == 0 and "c2p_y" in payload:
            c2p_y = payload["c2p_y"]
        else:
            raise ValueError(f"missing c2p_y history for level {level}")
        momenta_k = payload.get(prefix + "momenta_k", payload.get("momenta_k") if level == 0 else None)
        level_summary = phi4_summary_from_histories(
            shape=shape,
            magnetization=magnetization,
            energy_density=energy_density,
            c2p_x=c2p_x,
            c2p_y=c2p_y,
            momenta_k=momenta_k,
            iat_method=iat_method,
            iat_c=float(iat_c),
            block_size=int(block_size),
        )
        levels.append(
            {
                "level_from_fine": int(level),
                "L": int(shape[0]),
                "shape": [int(shape[0]), int(shape[1])],
                "summary": level_summary,
                "unweighted": phi4_level_unweighted_from_summary(level_summary),
            }
        )

    return {
        "n_levels": int(n_levels),
        "levels": levels,
    }

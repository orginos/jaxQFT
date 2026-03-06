#!/usr/bin/env python3
"""Offline analysis of pion/eta correlators from mcmc.py checkpoints.

Reads inline correlator measurements from an MCMC checkpoint produced by:
  scripts/mcmc/mcmc.py

Computes:
- mean correlators with IAT-adjusted errors,
- cosh effective masses with blocked jackknife errors,
- optional plateau estimate from a fit window.

Outputs:
- PNG figure with pion/eta correlators and effective masses,
- JSON summary with all derived arrays and metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def autocorrelation_fft(x) -> np.ndarray:
    """Return normalized autocorrelation function rho[t] for a 1D series."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    y = a - np.mean(a)
    c0 = float(np.dot(y, y))
    if not np.isfinite(c0) or c0 <= 0.0:
        return np.full((n,), np.nan, dtype=np.float64)
    f = np.fft.rfft(y, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
    return np.asarray(ac / c0, dtype=np.float64)


def integrated_autocorr_time(
    x,
    *,
    method: str = "ips",
    c: float = 5.0,
    max_lag: int | None = None,
) -> dict:
    """Estimate integrated autocorrelation time."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n < 2:
        return {
            "tau_int": float("nan"),
            "ess": float("nan"),
            "window": 0,
            "n": n,
            "method": method,
            "ok": False,
            "message": "need at least 2 samples",
        }

    rho = autocorrelation_fft(a)
    if rho.size == 0 or not np.isfinite(rho[0]):
        return {
            "tau_int": float("nan"),
            "ess": float("nan"),
            "window": 0,
            "n": n,
            "method": method,
            "ok": False,
            "message": "zero or non-finite variance",
        }

    mmax = n - 1 if max_lag is None else max(1, min(int(max_lag), n - 1))
    method = str(method).lower()

    if method == "ips":
        tau = 0.5
        w = 0
        t = 1
        while t + 1 <= mmax:
            g = float(rho[t] + rho[t + 1])
            if (not np.isfinite(g)) or (g <= 0.0):
                break
            tau += g
            w = t + 1
            t += 2
        tau = max(float(tau), 0.5)
        ess = float(n / (2.0 * tau))
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
        }

    if method == "sokal":
        tau = 0.5
        w = 1
        for _ in range(32):
            w = max(1, min(mmax, int(np.floor(c * tau))))
            r = rho[1 : w + 1]
            r = r[np.isfinite(r)]
            tau_new = float(max(0.5, 0.5 + np.sum(r)))
            if abs(tau_new - tau) <= 1e-6 * max(1.0, tau):
                tau = tau_new
                break
            tau = tau_new
        ess = float(n / (2.0 * tau))
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
        }

    if method == "gamma":
        y = a - np.mean(a)
        f = np.fft.rfft(y, n=2 * n)
        ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        norm = np.arange(n, 0, -1, dtype=np.float64)
        gamma = ac / norm
        gamma0 = float(gamma[0])
        if not np.isfinite(gamma0) or gamma0 <= 0.0:
            return {
                "tau_int": float("nan"),
                "ess": float("nan"),
                "window": 0,
                "n": n,
                "method": method,
                "ok": False,
                "message": "non-positive Gamma(0)",
            }
        rho_g = np.asarray(gamma / gamma0, dtype=np.float64)
        rho_g = np.where(np.isfinite(rho_g), rho_g, 0.0)
        tau = 0.5
        w = mmax
        for t in range(1, mmax + 1):
            tau += float(rho_g[t])
            tau = max(tau, 0.5)
            if t >= c * tau:
                w = t
                break
        tau = max(float(tau), 0.5)
        ess = float(n / (2.0 * tau))
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
        }

    raise ValueError(f"Unknown IAT method: {method}")


def cosh_meff_solve(ratio: float, t: int, t_extent: int, m_max: float = 10.0, tol: float = 1e-12) -> float:
    """Solve cosh meff equation from ratio C(t)/C(t+1) for periodic correlators."""
    if (not np.isfinite(ratio)) or ratio <= 1.0:
        return float("nan")

    u = float(t) - 0.5 * float(t_extent)
    v = u + 1.0
    if abs(v) < 1e-12:
        try:
            return float(np.arccosh(ratio))
        except Exception:
            return float("nan")

    log_ratio = float(np.log(ratio))

    def _log_cosh(z: float) -> float:
        az = abs(z)
        return az + float(np.log1p(np.exp(-2.0 * az))) - math.log(2.0)

    def f(m: float) -> float:
        return (_log_cosh(m * u) - _log_cosh(m * v)) - log_ratio

    a = 0.0
    b = float(m_max)
    fa = -log_ratio
    fb = f(b)
    if (not np.isfinite(fb)) or (fa * fb > 0.0):
        return float("nan")

    for _ in range(200):
        c = 0.5 * (a + b)
        if (b - a) < tol:
            return float(c)
        fc = f(c)
        if not np.isfinite(fc):
            b = c
            continue
        if abs(fc) < tol:
            return float(c)
        if fa * fc < 0.0:
            b = c
        else:
            a = c
            fa = fc
    return float(0.5 * (a + b))


def cosh_meff_from_corr(corr: np.ndarray) -> np.ndarray:
    """Compute cosh effective mass curve for t=0..T/2-1 from mean correlator."""
    c = np.asarray(corr, dtype=np.float64).reshape(-1)
    t_extent = int(c.size)
    if t_extent < 4:
        return np.asarray([], dtype=np.float64)
    t_half = t_extent // 2
    out = np.full((t_half,), np.nan, dtype=np.float64)
    for t in range(t_half):
        c0 = float(c[t])
        c1 = float(c[t + 1]) if (t + 1) < t_extent else float("nan")
        if (not np.isfinite(c0)) or (not np.isfinite(c1)) or c1 == 0.0:
            continue
        out[t] = cosh_meff_solve(c0 / c1, t=t, t_extent=t_extent)
    return out


def _load_inline_records_from_checkpoint(path: Path) -> List[Dict]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a dict: {type(payload).__name__}")
    state = payload.get("state", {})
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state is not a dict: {type(state).__name__}")
    records = state.get("inline_records", [])
    if not isinstance(records, list):
        raise ValueError("Checkpoint state.inline_records is not a list")
    return records


def _extract_channel_samples(
    records: Sequence[Mapping],
    *,
    measurement_name: str,
    channel: str,
    momentum: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Extract (n_meas, T) real correlator samples for one measurement/channel."""
    p = int(momentum)
    chan = str(channel)
    pat_re = re.compile(rf"^{re.escape(chan)}_p{p}_t(?P<t>\d+)_re$")
    pat_im = re.compile(rf"^{re.escape(chan)}_p{p}_t(?P<t>\d+)_im$")

    legacy_re = None
    if p == 0 and chan in ("c", "full", "conn", "disc"):
        legacy_re = re.compile(rf"^{re.escape(chan)}_t(?P<t>\d+)$")

    by_step: Dict[int, Dict[int, float]] = {}
    imag_abs_max = 0.0
    duplicate_steps = 0

    for rec in records:
        try:
            name = str(rec.get("name", ""))
        except Exception:
            continue
        if name != str(measurement_name):
            continue
        vals = rec.get("values", {})
        if not isinstance(vals, Mapping):
            continue

        tmap: Dict[int, float] = {}
        for k, v in vals.items():
            key = str(k)
            m_re = pat_re.match(key)
            if m_re is not None:
                tmap[int(m_re.group("t"))] = float(v)
                continue
            if legacy_re is not None:
                m_legacy = legacy_re.match(key)
                if m_legacy is not None:
                    tmap[int(m_legacy.group("t"))] = float(v)
                    continue
            m_im = pat_im.match(key)
            if m_im is not None:
                imag_abs_max = max(imag_abs_max, abs(float(v)))

        if not tmap:
            continue

        step = int(rec.get("step", -1))
        if step in by_step:
            duplicate_steps += 1
        by_step[step] = tmap

    if not by_step:
        raise ValueError(
            f"No correlator samples found for measurement='{measurement_name}', channel='{channel}', p={p}."
        )

    t_extent = 1 + max(max(m.keys()) for m in by_step.values() if m)
    steps_sorted = sorted(by_step.keys())
    rows: List[np.ndarray] = []
    kept_steps: List[int] = []
    dropped_incomplete = 0

    for step in steps_sorted:
        tmap = by_step[step]
        row = np.full((t_extent,), np.nan, dtype=np.float64)
        for t, val in tmap.items():
            if 0 <= int(t) < t_extent:
                row[int(t)] = float(val)
        if np.all(np.isfinite(row)):
            rows.append(row)
            kept_steps.append(int(step))
        else:
            dropped_incomplete += 1

    if not rows:
        raise ValueError(
            f"All extracted samples were incomplete for measurement='{measurement_name}', channel='{channel}', p={p}."
        )

    meta = {
        "t_extent": float(t_extent),
        "n_total_steps": float(len(steps_sorted)),
        "n_kept": float(len(rows)),
        "n_dropped_incomplete": float(dropped_incomplete),
        "n_duplicate_steps": float(duplicate_steps),
        "imag_abs_max": float(imag_abs_max),
    }
    return np.asarray(rows, dtype=np.float64), np.asarray(kept_steps, dtype=np.int64), meta


def _apply_discard_stride(samples: np.ndarray, steps: np.ndarray, discard: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    d = max(0, int(discard))
    s = max(1, int(stride))
    return np.asarray(samples[d::s], dtype=np.float64), np.asarray(steps[d::s], dtype=np.int64)


def _correlator_stats_iat(
    samples: np.ndarray,
    *,
    iat_method: str,
    iat_c: float,
    iat_max_lag: Optional[int],
) -> Dict[str, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    n, t_extent = int(x.shape[0]), int(x.shape[1])
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1) if n > 1 else np.full((t_extent,), np.nan, dtype=np.float64)
    tau = np.full((t_extent,), np.nan, dtype=np.float64)
    ess = np.full((t_extent,), np.nan, dtype=np.float64)
    err = np.full((t_extent,), np.nan, dtype=np.float64)

    for t in range(t_extent):
        iat = integrated_autocorr_time(
            x[:, t],
            method=str(iat_method),
            c=float(iat_c),
            max_lag=iat_max_lag,
        )
        tau_t = float(iat.get("tau_int", float("nan")))
        if (not np.isfinite(tau_t)) or tau_t < 0.5:
            tau_t = 0.5
        tau[t] = tau_t
        ess_t = max(1.0, float(n) / (2.0 * tau_t))
        ess[t] = ess_t
        if n > 1 and np.isfinite(std[t]):
            err[t] = float(std[t] / np.sqrt(ess_t))
    return {
        "mean": mean,
        "err": err,
        "tau": tau,
        "ess": ess,
    }


def _auto_block_size(
    sample_sets: Sequence[np.ndarray],
    *,
    iat_method: str,
    iat_c: float,
    iat_max_lag: Optional[int],
) -> Tuple[int, float]:
    n_min = min(int(s.shape[0]) for s in sample_sets if s.size > 0)
    if n_min < 2:
        return 1, float("nan")

    tau_vals: List[float] = []
    for s in sample_sets:
        if s.size == 0:
            continue
        t_extent = int(s.shape[1])
        t_stop = max(1, t_extent // 2)
        for t in range(t_stop):
            iat = integrated_autocorr_time(
                s[:, t],
                method=str(iat_method),
                c=float(iat_c),
                max_lag=iat_max_lag,
            )
            tau_t = float(iat.get("tau_int", float("nan")))
            if np.isfinite(tau_t) and tau_t >= 0.5:
                tau_vals.append(tau_t)

    tau_ref = float(max(tau_vals)) if tau_vals else 1.0
    block = max(1, int(math.ceil(2.0 * tau_ref)))

    # Keep enough blocks for jackknife stability.
    if n_min // block < 4 and n_min >= 4:
        block = max(1, n_min // 4)
    if n_min // block < 2:
        block = max(1, n_min // 2)
    return int(block), float(tau_ref)


def _block_means(samples: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    n = int(x.shape[0])
    b = max(1, int(block_size))
    n_blocks = n // b
    if n_blocks < 2:
        raise ValueError(
            f"Not enough samples for blocked jackknife: n={n}, block_size={b}, n_blocks={n_blocks}."
        )
    used = int(n_blocks * b)
    xt = x[:used]
    return np.asarray(xt.reshape(n_blocks, b, x.shape[1]).mean(axis=1), dtype=np.float64)


def _blocked_jackknife_meff(samples: np.ndarray, block_size: int) -> Dict[str, np.ndarray | float]:
    blocks = _block_means(samples, block_size=block_size)
    n_blocks = int(blocks.shape[0])
    c_mean = np.mean(blocks, axis=0)
    meff = cosh_meff_from_corr(c_mean)
    meff_loo = np.full((n_blocks, meff.size), np.nan, dtype=np.float64)
    for j in range(n_blocks):
        c_loo = (c_mean * n_blocks - blocks[j]) / float(n_blocks - 1)
        meff_loo[j] = cosh_meff_from_corr(c_loo)
    err = np.sqrt((n_blocks - 1.0) / n_blocks * np.nansum((meff_loo - meff[None, :]) ** 2, axis=0))
    return {
        "corr_mean": c_mean,
        "meff": meff,
        "meff_err": err,
        "n_blocks": float(n_blocks),
        "n_used": float(blocks.shape[0] * max(1, int(block_size))),
    }


def _parse_fit_range(txt: str) -> Optional[Tuple[int, int]]:
    s = str(txt).strip()
    if not s:
        return None
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if len(toks) != 2:
        raise ValueError("--fit-range must be 'tmin,tmax'")
    tmin = int(toks[0])
    tmax = int(toks[1])
    if tmax < tmin:
        raise ValueError("--fit-range must satisfy tmax >= tmin")
    return tmin, tmax


def _plateau_weighted_mean(
    meff: np.ndarray,
    meff_err: np.ndarray,
    fit_range: Optional[Tuple[int, int]],
) -> Dict[str, float]:
    if fit_range is None:
        return {"mass": float("nan"), "err": float("nan"), "chi2_dof": float("nan"), "npts": 0.0}

    tmin, tmax = fit_range
    t = np.arange(meff.size, dtype=np.int64)
    m = np.asarray(meff, dtype=np.float64)
    e = np.asarray(meff_err, dtype=np.float64)
    mask = (t >= int(tmin)) & (t <= int(tmax)) & np.isfinite(m)
    if np.any(np.isfinite(e)):
        mask = mask & np.isfinite(e) & (e > 0.0)
    if not np.any(mask):
        return {"mass": float("nan"), "err": float("nan"), "chi2_dof": float("nan"), "npts": 0.0}

    mm = m[mask]
    ee = e[mask]
    npts = int(mm.size)
    if np.all(np.isfinite(ee)) and np.all(ee > 0.0):
        w = 1.0 / (ee * ee)
    else:
        w = np.ones_like(mm)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return {"mass": float("nan"), "err": float("nan"), "chi2_dof": float("nan"), "npts": float(npts)}
    mass = float(np.sum(w * mm) / wsum)
    err = float(np.sqrt(1.0 / wsum))
    chi2 = float(np.sum(w * (mm - mass) ** 2))
    dof = max(1, npts - 1)
    return {
        "mass": mass,
        "err": err,
        "chi2_dof": float(chi2 / dof),
        "npts": float(npts),
    }


def _setup_matplotlib(no_gui: bool):
    if "MPLCONFIGDIR" not in os.environ:
        mpldir = Path(tempfile.gettempdir()) / f"mplconfig_{os.getuid()}"
        mpldir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpldir)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg = Path(tempfile.gettempdir()) / f"xdg_cache_{os.getuid()}"
        xdg.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg)

    if no_gui:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze pion/eta correlators from mcmc checkpoint")
    ap.add_argument("--input", type=str, required=True, help="checkpoint .pkl path from scripts/mcmc/mcmc.py")
    ap.add_argument("--outdir", type=str, default="", help="output directory (default: <checkpoint_dir>/analysis)")
    ap.add_argument("--prefix", type=str, default="u1_2pt", help="output filename prefix")
    ap.add_argument("--discard", type=int, default=0, help="discard first N measurements")
    ap.add_argument("--stride", type=int, default=1, help="keep every stride-th sample after discard")
    ap.add_argument("--mom", type=int, default=0, help="momentum index p to analyze")
    ap.add_argument("--eta-channel", type=str, default="full", choices=["full", "conn", "disc"])
    ap.add_argument("--iat-method", type=str, default="ips", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0, help="0 means automatic (n-1)")
    ap.add_argument("--block-size", type=int, default=0, help="blocked jackknife block size; <=0 selects auto")
    ap.add_argument("--fit-range", type=str, default="", help="optional plateau fit range: tmin,tmax")
    ap.add_argument("--title", type=str, default="Pion/Eta Two-Point Analysis")
    ap.add_argument("--no-gui", action="store_true", help="disable interactive display")
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {ckpt_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (ckpt_path.parent / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{args.prefix}_p{int(args.mom)}_eta{args.eta_channel}.png"
    json_path = outdir / f"{args.prefix}_p{int(args.mom)}_eta{args.eta_channel}.json"

    records = _load_inline_records_from_checkpoint(ckpt_path)
    if not records:
        raise ValueError(f"No inline_records found in checkpoint: {ckpt_path}")

    pion_samples, pion_steps, pion_meta = _extract_channel_samples(
        records,
        measurement_name="pion_2pt",
        channel="c",
        momentum=int(args.mom),
    )
    eta_samples, eta_steps, eta_meta = _extract_channel_samples(
        records,
        measurement_name="eta_2pt",
        channel=str(args.eta_channel),
        momentum=int(args.mom),
    )

    pion_samples, pion_steps = _apply_discard_stride(
        pion_samples, pion_steps, discard=int(args.discard), stride=int(args.stride)
    )
    eta_samples, eta_steps = _apply_discard_stride(
        eta_samples, eta_steps, discard=int(args.discard), stride=int(args.stride)
    )
    if pion_samples.shape[0] < 2 or eta_samples.shape[0] < 2:
        raise ValueError(
            "Need at least 2 selected samples for each channel after discard/stride "
            f"(got pion={pion_samples.shape[0]}, eta={eta_samples.shape[0]})."
        )

    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    pion_stats = _correlator_stats_iat(
        pion_samples,
        iat_method=str(args.iat_method),
        iat_c=float(args.iat_c),
        iat_max_lag=iat_max_lag,
    )
    eta_stats = _correlator_stats_iat(
        eta_samples,
        iat_method=str(args.iat_method),
        iat_c=float(args.iat_c),
        iat_max_lag=iat_max_lag,
    )

    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        tau_ref = float("nan")
    else:
        block_size, tau_ref = _auto_block_size(
            [pion_samples, eta_samples],
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
        )

    pion_jk = _blocked_jackknife_meff(pion_samples, block_size=block_size)
    eta_jk = _blocked_jackknife_meff(eta_samples, block_size=block_size)

    fit_range = _parse_fit_range(args.fit_range)
    pion_plateau = _plateau_weighted_mean(pion_jk["meff"], pion_jk["meff_err"], fit_range)
    eta_plateau = _plateau_weighted_mean(eta_jk["meff"], eta_jk["meff_err"], fit_range)

    plt = _setup_matplotlib(no_gui=bool(args.no_gui))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_pc, ax_ec = axs[0, 0], axs[0, 1]
    ax_pm, ax_em = axs[1, 0], axs[1, 1]

    t_p = np.arange(pion_stats["mean"].size, dtype=np.int64)
    t_e = np.arange(eta_stats["mean"].size, dtype=np.int64)
    ax_pc.errorbar(t_p, pion_stats["mean"], yerr=pion_stats["err"], fmt="o-", ms=3, lw=1.2, capsize=2, color="#1f77b4")
    ax_ec.errorbar(t_e, eta_stats["mean"], yerr=eta_stats["err"], fmt="o-", ms=3, lw=1.2, capsize=2, color="#d62728")
    if np.all(np.asarray(pion_stats["mean"]) > 0.0):
        ax_pc.set_yscale("log")
    if np.all(np.asarray(eta_stats["mean"]) > 0.0):
        ax_ec.set_yscale("log")
    ax_pc.set_title(f"Pion correlator C_pi(t), p={int(args.mom)}")
    ax_ec.set_title(f"Eta correlator C_eta(t) [{args.eta_channel}], p={int(args.mom)}")
    ax_pc.set_xlabel("t")
    ax_ec.set_xlabel("t")
    ax_pc.set_ylabel("C(t)")
    ax_ec.set_ylabel("C(t)")
    ax_pc.grid(True, alpha=0.25)
    ax_ec.grid(True, alpha=0.25)

    t_pm = np.arange(np.asarray(pion_jk["meff"]).size, dtype=np.int64)
    t_em = np.arange(np.asarray(eta_jk["meff"]).size, dtype=np.int64)
    ax_pm.errorbar(t_pm, pion_jk["meff"], yerr=pion_jk["meff_err"], fmt="o-", ms=3, lw=1.2, capsize=2, color="#1f77b4")
    ax_em.errorbar(t_em, eta_jk["meff"], yerr=eta_jk["meff_err"], fmt="o-", ms=3, lw=1.2, capsize=2, color="#d62728")
    ax_pm.set_title("Pion effective mass (cosh)")
    ax_em.set_title("Eta effective mass (cosh)")
    ax_pm.set_xlabel("t")
    ax_em.set_xlabel("t")
    ax_pm.set_ylabel(r"$m_{\mathrm{eff}}(t)$")
    ax_em.set_ylabel(r"$m_{\mathrm{eff}}(t)$")
    ax_pm.grid(True, alpha=0.25)
    ax_em.grid(True, alpha=0.25)

    if fit_range is not None:
        tmin, tmax = fit_range
        ax_pm.axvspan(tmin, tmax, color="#1f77b4", alpha=0.12)
        ax_em.axvspan(tmin, tmax, color="#d62728", alpha=0.12)
        if np.isfinite(float(pion_plateau["mass"])):
            ax_pm.axhline(float(pion_plateau["mass"]), color="#1f77b4", ls="--", lw=1.2)
        if np.isfinite(float(eta_plateau["mass"])):
            ax_em.axhline(float(eta_plateau["mass"]), color="#d62728", ls="--", lw=1.2)

    info_lines = [
        f"pion n={pion_samples.shape[0]} | eta n={eta_samples.shape[0]}",
        f"iat={args.iat_method} block={block_size}" + (f" (auto tau_ref~{tau_ref:.2f})" if np.isfinite(tau_ref) else ""),
        f"discard={int(args.discard)} stride={int(args.stride)}",
    ]
    fig.suptitle(str(args.title))
    fig.text(0.5, 0.01, " | ".join(info_lines), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    out = {
        "input_checkpoint": str(ckpt_path),
        "output_png": str(png_path),
        "output_json": str(json_path),
        "settings": {
            "momentum": int(args.mom),
            "eta_channel": str(args.eta_channel),
            "discard": int(args.discard),
            "stride": int(args.stride),
            "iat_method": str(args.iat_method),
            "iat_c": float(args.iat_c),
            "iat_max_lag": (None if iat_max_lag is None else int(iat_max_lag)),
            "block_size": int(block_size),
            "fit_range": (None if fit_range is None else [int(fit_range[0]), int(fit_range[1])]),
        },
        "pion": {
            "meta": pion_meta,
            "steps": np.asarray(pion_steps, dtype=np.int64).tolist(),
            "corr_mean": np.asarray(pion_stats["mean"], dtype=np.float64).tolist(),
            "corr_err_iat": np.asarray(pion_stats["err"], dtype=np.float64).tolist(),
            "corr_tau_int": np.asarray(pion_stats["tau"], dtype=np.float64).tolist(),
            "corr_ess": np.asarray(pion_stats["ess"], dtype=np.float64).tolist(),
            "meff_cosh": np.asarray(pion_jk["meff"], dtype=np.float64).tolist(),
            "meff_cosh_err": np.asarray(pion_jk["meff_err"], dtype=np.float64).tolist(),
            "meff_plateau": pion_plateau,
            "n_blocks": float(pion_jk["n_blocks"]),
            "n_used_for_jk": float(pion_jk["n_used"]),
        },
        "eta": {
            "meta": eta_meta,
            "steps": np.asarray(eta_steps, dtype=np.int64).tolist(),
            "corr_mean": np.asarray(eta_stats["mean"], dtype=np.float64).tolist(),
            "corr_err_iat": np.asarray(eta_stats["err"], dtype=np.float64).tolist(),
            "corr_tau_int": np.asarray(eta_stats["tau"], dtype=np.float64).tolist(),
            "corr_ess": np.asarray(eta_stats["ess"], dtype=np.float64).tolist(),
            "meff_cosh": np.asarray(eta_jk["meff"], dtype=np.float64).tolist(),
            "meff_cosh_err": np.asarray(eta_jk["meff_err"], dtype=np.float64).tolist(),
            "meff_plateau": eta_plateau,
            "n_blocks": float(eta_jk["n_blocks"]),
            "n_used_for_jk": float(eta_jk["n_used"]),
        },
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print("2pt Effective-Mass Summary")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  selected samples: pion={pion_samples.shape[0]} eta={eta_samples.shape[0]}")
    print(f"  momentum/channel: p={int(args.mom)} eta={args.eta_channel}")
    print(f"  iat method: {args.iat_method}  block_size={block_size}")
    if fit_range is not None:
        tmin, tmax = fit_range
        print(
            f"  pion plateau t=[{tmin},{tmax}]:"
            f" {float(pion_plateau['mass']):.6f} +/- {float(pion_plateau['err']):.6f}"
            f" (chi2/dof={float(pion_plateau['chi2_dof']):.3f})"
        )
        print(
            f"  eta  plateau t=[{tmin},{tmax}]:"
            f" {float(eta_plateau['mass']):.6f} +/- {float(eta_plateau['err']):.6f}"
            f" (chi2/dof={float(eta_plateau['chi2_dof']):.3f})"
        )
    print(f"  saved figure: {png_path}")
    print(f"  saved json:   {json_path}")

    if not args.no_gui:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

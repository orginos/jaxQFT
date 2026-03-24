#!/usr/bin/env python3
"""Correlated exponential fits and dispersion analysis from mcmc checkpoints.

Reads inline two-point correlators from a checkpoint produced by:
  scripts/mcmc/mcmc.py

For each available momentum p in a chosen measurement/channel:
1) performs correlated cosh fits C(t)=A[exp(-E t)+exp(-E (T-t))]
2) selects fit window automatically from chi2/dof
3) estimates parameter errors with blocked jackknife
4) builds and plots the dispersion relation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def autocorrelation_fft(x) -> np.ndarray:
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
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n < 2:
        return {"tau_int": float("nan"), "ok": False, "message": "need >=2 samples"}

    rho = autocorrelation_fft(a)
    if rho.size == 0 or not np.isfinite(rho[0]):
        return {"tau_int": float("nan"), "ok": False, "message": "zero/non-finite variance"}

    mmax = n - 1 if max_lag is None else max(1, min(int(max_lag), n - 1))
    method = str(method).lower()

    if method == "ips":
        tau = 0.5
        t = 1
        while t + 1 <= mmax:
            g = float(rho[t] + rho[t + 1])
            if not np.isfinite(g) or g <= 0.0:
                break
            tau += g
            t += 2
        return {"tau_int": max(0.5, float(tau)), "ok": True, "message": "ok"}

    if method == "sokal":
        tau = 0.5
        for _ in range(32):
            w = max(1, min(mmax, int(np.floor(c * tau))))
            r = rho[1 : w + 1]
            r = r[np.isfinite(r)]
            tau_new = float(max(0.5, 0.5 + np.sum(r)))
            if abs(tau_new - tau) <= 1e-6 * max(1.0, tau):
                tau = tau_new
                break
            tau = tau_new
        return {"tau_int": max(0.5, float(tau)), "ok": True, "message": "ok"}

    if method == "gamma":
        y = a - np.mean(a)
        f = np.fft.rfft(y, n=2 * n)
        ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        norm = np.arange(n, 0, -1, dtype=np.float64)
        gamma = ac / norm
        gamma0 = float(gamma[0])
        if not np.isfinite(gamma0) or gamma0 <= 0.0:
            return {"tau_int": float("nan"), "ok": False, "message": "non-positive gamma0"}
        rho_g = np.asarray(gamma / gamma0, dtype=np.float64)
        rho_g = np.where(np.isfinite(rho_g), rho_g, 0.0)
        tau = 0.5
        for t in range(1, mmax + 1):
            tau += float(rho_g[t])
            tau = max(tau, 0.5)
            if t >= c * tau:
                break
        return {"tau_int": max(0.5, float(tau)), "ok": True, "message": "ok"}

    raise ValueError(f"Unknown IAT method: {method}")


def _load_checkpoint(path: Path) -> Tuple[Dict, List[Dict]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a dict: {type(payload).__name__}")
    state = payload.get("state", {})
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state is not a dict: {type(state).__name__}")
    records = state.get("inline_records", [])
    if not isinstance(records, list):
        raise ValueError("state.inline_records is not a list")
    return payload, records


def _extract_by_momentum(
    records: Sequence[Mapping],
    *,
    measurement: str,
    channel: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    """Return momentum->(samples, steps, meta)."""
    chan = str(channel)
    pat_re = re.compile(rf"^{re.escape(chan)}_p(?P<p>-?\d+)_t(?P<t>\d+)_re$")
    pat_im = re.compile(rf"^{re.escape(chan)}_p(?P<p>-?\d+)_t(?P<t>\d+)_im$")
    pat_valid = re.compile(rf"^{re.escape(chan)}_valid_t(?P<t>\d+)$")
    legacy_re = re.compile(rf"^{re.escape(chan)}_t(?P<t>\d+)$") if chan in ("c", "full", "conn", "disc") else None

    by_p_step: Dict[int, Dict[int, Dict[int, float]]] = {}
    imag_abs_max: Dict[int, float] = {}
    valid_times_by_step: Dict[int, set[int]] = {}

    for rec in records:
        if str(rec.get("name", "")) != str(measurement):
            continue
        vals = rec.get("values", {})
        if not isinstance(vals, Mapping):
            continue
        step = int(rec.get("step", -1))
        for k, v in vals.items():
            key = str(k)
            m_valid = pat_valid.match(key)
            if m_valid is not None:
                valid_times_by_step.setdefault(step, set()).add(int(m_valid.group("t")))
                continue

            m_re = pat_re.match(key)
            if m_re is not None:
                p = int(m_re.group("p"))
                t = int(m_re.group("t"))
                by_p_step.setdefault(p, {}).setdefault(step, {})[t] = float(v)
                continue

            if legacy_re is not None:
                m_legacy = legacy_re.match(key)
                if m_legacy is not None:
                    p = 0
                    t = int(m_legacy.group("t"))
                    by_p_step.setdefault(p, {}).setdefault(step, {})[t] = float(v)
                    continue

            m_im = pat_im.match(key)
            if m_im is not None:
                p = int(m_im.group("p"))
                imag_abs_max[p] = max(float(imag_abs_max.get(p, 0.0)), abs(float(v)))

    if not by_p_step:
        raise ValueError(f"No momentum-resolved data found for measurement='{measurement}', channel='{channel}'.")

    samples_by_p: Dict[int, np.ndarray] = {}
    steps_by_p: Dict[int, np.ndarray] = {}
    meta_by_p: Dict[int, Dict[str, float]] = {}
    valid_support: Optional[List[int]] = None
    if valid_times_by_step:
        support_sets = [set(v) for v in valid_times_by_step.values() if v]
        if support_sets:
            valid_support = sorted(set.intersection(*support_sets))

    for p in sorted(by_p_step.keys()):
        by_step = by_p_step[p]
        if valid_support is not None:
            support_times = list(valid_support)
        else:
            t_extent = 1 + max(max(tm.keys()) for tm in by_step.values() if tm)
            support_times = list(range(int(t_extent)))
        rows: List[np.ndarray] = []
        kept_steps: List[int] = []
        dropped = 0
        for step in sorted(by_step.keys()):
            row = np.full((len(support_times),), np.nan, dtype=np.float64)
            vals_step = by_step[step]
            for i, t in enumerate(support_times):
                if int(t) in vals_step:
                    row[i] = float(vals_step[int(t)])
            if np.all(np.isfinite(row)):
                rows.append(row)
                kept_steps.append(int(step))
            else:
                dropped += 1
        if not rows:
            continue
        samples_by_p[p] = np.asarray(rows, dtype=np.float64)
        steps_by_p[p] = np.asarray(kept_steps, dtype=np.int64)
        meta_by_p[p] = {
            "t_extent": float(len(support_times)),
            "n_total": float(len(by_step)),
            "n_kept": float(len(rows)),
            "n_dropped_incomplete": float(dropped),
            "imag_abs_max": float(imag_abs_max.get(p, 0.0)),
            "support_times": [int(t) for t in support_times],
            "has_explicit_valid_mask": 1.0 if valid_support is not None else 0.0,
        }

    if not samples_by_p:
        raise ValueError("No complete samples remained after completeness filtering.")
    return samples_by_p, steps_by_p, meta_by_p


def _apply_discard_stride(samples: np.ndarray, steps: np.ndarray, discard: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    d = max(0, int(discard))
    s = max(1, int(stride))
    return np.asarray(samples[d::s], dtype=np.float64), np.asarray(steps[d::s], dtype=np.int64)


def _block_means(samples: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    n = int(x.shape[0])
    b = max(1, int(block_size))
    n_blocks = n // b
    if n_blocks < 2:
        raise ValueError(f"Not enough samples for blocked analysis: n={n}, block={b}, blocks={n_blocks}")
    used = int(n_blocks * b)
    xt = x[:used]
    return np.asarray(xt.reshape(n_blocks, b, x.shape[1]).mean(axis=1), dtype=np.float64)


def _auto_block_size_from_p0(
    samples_by_p: Mapping[int, np.ndarray],
    *,
    iat_method: str,
    iat_c: float,
    iat_max_lag: Optional[int],
) -> Tuple[int, float]:
    if 0 in samples_by_p:
        ref = np.asarray(samples_by_p[0], dtype=np.float64)
    else:
        p0 = sorted(samples_by_p.keys())[0]
        ref = np.asarray(samples_by_p[p0], dtype=np.float64)
    n, t_extent = ref.shape
    t_stop = max(2, t_extent // 2)
    tau_vals: List[float] = []
    for t in range(1, t_stop):
        iat = integrated_autocorr_time(
            ref[:, t],
            method=str(iat_method),
            c=float(iat_c),
            max_lag=iat_max_lag,
        )
        tau = float(iat.get("tau_int", float("nan")))
        if np.isfinite(tau) and tau >= 0.5:
            tau_vals.append(tau)
    tau_ref = float(max(tau_vals)) if tau_vals else 1.0
    block = max(1, int(math.ceil(2.0 * tau_ref)))
    if n // block < 4 and n >= 4:
        block = max(1, n // 4)
    if n // block < 2:
        block = max(1, n // 2)
    return int(block), float(tau_ref)


def _model_cosh(t: np.ndarray, t_extent: int, energy: float) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float64)
    e = float(energy)
    return np.exp(-e * tt) + np.exp(-e * (float(t_extent) - tt))


def _model_basis_matrix(t: np.ndarray, t_extent: int, energies: Sequence[float]) -> np.ndarray:
    cols = [_model_cosh(t=t, t_extent=t_extent, energy=float(e)) for e in energies]
    return np.asarray(np.stack(cols, axis=1), dtype=np.float64)


def _regularized_inverse(cov: np.ndarray, reg: float) -> Tuple[np.ndarray, float]:
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim == 0:
        c = c.reshape(1, 1)
    n = int(c.shape[0])
    tr = float(np.trace(c))
    scale = tr / float(max(1, n)) if np.isfinite(tr) and tr > 0.0 else 1.0
    eps = max(0.0, float(reg)) * scale
    c_reg = c + eps * np.eye(n, dtype=np.float64)
    inv = np.linalg.pinv(c_reg, rcond=1e-12)
    return inv, float(eps)


def _solve_amplitudes_weighted(y: np.ndarray, basis: np.ndarray, winv: np.ndarray) -> Optional[np.ndarray]:
    b = np.asarray(basis, dtype=np.float64)
    yw = np.asarray(y, dtype=np.float64)
    m = b.T @ winv @ b
    rhs = b.T @ winv @ yw
    try:
        amps = np.linalg.solve(m, rhs)
    except np.linalg.LinAlgError:
        amps = np.linalg.pinv(m, rcond=1e-12) @ rhs
    if not np.all(np.isfinite(amps)):
        return None
    return np.asarray(amps, dtype=np.float64)


def _chi2_for_energies(
    energies: Sequence[float],
    y: np.ndarray,
    t: np.ndarray,
    t_extent: int,
    winv: np.ndarray,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    e = np.asarray(energies, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return float("nan"), None, None
    if np.any(~np.isfinite(e)) or np.any(e <= 0.0):
        return float("nan"), None, None
    if e.size >= 2 and np.any(np.diff(e) <= 0.0):
        return float("nan"), None, None

    basis = _model_basis_matrix(t=t, t_extent=t_extent, energies=e)
    amps = _solve_amplitudes_weighted(y=y, basis=basis, winv=winv)
    if amps is None:
        return float("nan"), None, None
    yfit = basis @ amps
    r = y - yfit
    chi2 = float(r @ winv @ r)
    if not np.isfinite(chi2):
        return float("nan"), None, None
    return chi2, amps, yfit


def _minimize_chi2_energy(
    y: np.ndarray,
    t: np.ndarray,
    t_extent: int,
    winv: np.ndarray,
    *,
    e_min: float,
    e_max: float,
    e_grid: int,
) -> Tuple[float, float, float]:
    emin = max(1e-8, float(e_min))
    emax = max(emin * 1.001, float(e_max))
    ngrid = max(32, int(e_grid))
    grid = np.linspace(emin, emax, ngrid)
    vals = np.full((ngrid,), np.nan, dtype=np.float64)
    amps = np.full((ngrid,), np.nan, dtype=np.float64)
    for i, e in enumerate(grid):
        chi2, amp_vec, _ = _chi2_for_energies([float(e)], y, t, t_extent, winv)
        amp = float(amp_vec[0]) if (amp_vec is not None and amp_vec.size >= 1) else float("nan")
        vals[i] = chi2
        amps[i] = amp
    if not np.any(np.isfinite(vals)):
        return float("nan"), float("nan"), float("nan")
    i0 = int(np.nanargmin(vals))
    left = max(0, i0 - 1)
    right = min(ngrid - 1, i0 + 1)
    a = float(grid[left])
    b = float(grid[right])
    if b <= a:
        e_star = float(grid[i0])
        return e_star, float(vals[i0]), float(amps[i0])

    phi = 0.5 * (1.0 + np.sqrt(5.0))
    invphi = 1.0 / phi
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc_vec, ac_vec, _ = _chi2_for_energies([c], y, t, t_extent, winv)
    fd_vec, ad_vec, _ = _chi2_for_energies([d], y, t, t_extent, winv)
    fc = fc_vec
    fd = fd_vec
    ac = float(ac_vec[0]) if (ac_vec is not None and ac_vec.size >= 1) else float("nan")
    ad = float(ad_vec[0]) if (ad_vec is not None and ad_vec.size >= 1) else float("nan")
    for _ in range(96):
        if not np.isfinite(fc):
            fc = float("inf")
        if not np.isfinite(fd):
            fd = float("inf")
        if fc <= fd:
            b = d
            d = c
            fd = fc
            ad = ac
            c = b - (b - a) * invphi
            fc_vec, ac_vec, _ = _chi2_for_energies([c], y, t, t_extent, winv)
            fc = fc_vec
            ac = float(ac_vec[0]) if (ac_vec is not None and ac_vec.size >= 1) else float("nan")
        else:
            a = c
            c = d
            fc = fd
            ac = ad
            d = a + (b - a) * invphi
            fd_vec, ad_vec, _ = _chi2_for_energies([d], y, t, t_extent, winv)
            fd = fd_vec
            ad = float(ad_vec[0]) if (ad_vec is not None and ad_vec.size >= 1) else float("nan")
        if abs(b - a) < 1e-10:
            break

    if fc <= fd:
        return float(c), float(fc), float(ac)
    return float(d), float(fd), float(ad)


def _golden_search_scalar(func, a: float, b: float, n_iter: int = 72, tol: float = 1e-9) -> Tuple[float, float]:
    lo = float(min(a, b))
    hi = float(max(a, b))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        x = float(lo)
        fx = float(func(x))
        return x, fx
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    invphi = 1.0 / phi
    c = hi - (hi - lo) * invphi
    d = lo + (hi - lo) * invphi
    fc = float(func(c))
    fd = float(func(d))
    for _ in range(int(max(8, n_iter))):
        if not np.isfinite(fc):
            fc = float("inf")
        if not np.isfinite(fd):
            fd = float("inf")
        if fc <= fd:
            hi = d
            d = c
            fd = fc
            c = hi - (hi - lo) * invphi
            fc = float(func(c))
        else:
            lo = c
            c = d
            fc = fd
            d = lo + (hi - lo) * invphi
            fd = float(func(d))
        if abs(hi - lo) < float(tol):
            break
    if fc <= fd:
        return float(c), float(fc)
    return float(d), float(fd)


def _minimize_chi2_two_exp(
    y: np.ndarray,
    t: np.ndarray,
    t_extent: int,
    winv: np.ndarray,
    *,
    e_min: float,
    e_max: float,
    e_grid: int,
) -> Tuple[float, float, float, float, float]:
    emin = max(1e-8, float(e_min))
    emax = max(emin * 1.001, float(e_max))
    ngrid_1d = max(12, int(round(np.sqrt(max(32, int(e_grid))))))
    grid = np.linspace(emin, emax, ngrid_1d)
    gap = max(1e-4, 0.5 * (grid[1] - grid[0]) if grid.size > 1 else 1e-3)

    best_chi2 = float("inf")
    best_e0 = float("nan")
    best_e1 = float("nan")
    best_a0 = float("nan")
    best_a1 = float("nan")

    for i, e0 in enumerate(grid):
        for j in range(i + 1, grid.size):
            e1 = float(grid[j])
            if e1 <= e0 + gap:
                continue
            chi2, amps, _ = _chi2_for_energies([e0, e1], y, t, t_extent, winv)
            if amps is None or amps.size < 2:
                continue
            if np.isfinite(chi2) and chi2 < best_chi2:
                best_chi2 = float(chi2)
                best_e0 = float(e0)
                best_e1 = float(e1)
                best_a0 = float(amps[0])
                best_a1 = float(amps[1])

    if not np.isfinite(best_chi2):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    e0 = float(best_e0)
    e1 = float(best_e1)
    for _ in range(24):
        upper0 = min(emax - gap, e1 - gap)
        lower0 = emin
        if upper0 > lower0 + 1e-10:
            e0, _ = _golden_search_scalar(
                lambda x: _chi2_for_energies([float(x), float(e1)], y, t, t_extent, winv)[0],
                lower0,
                upper0,
                n_iter=48,
                tol=1e-8,
            )
        lower1 = max(emin + gap, e0 + gap)
        upper1 = emax
        if upper1 > lower1 + 1e-10:
            e1, _ = _golden_search_scalar(
                lambda x: _chi2_for_energies([float(e0), float(x)], y, t, t_extent, winv)[0],
                lower1,
                upper1,
                n_iter=48,
                tol=1e-8,
            )

    chi2, amps, _ = _chi2_for_energies([e0, e1], y, t, t_extent, winv)
    if amps is None or amps.size < 2 or not np.isfinite(chi2):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return float(e0), float(e1), float(chi2), float(amps[0]), float(amps[1])


def _fit_window_correlated(
    blocks: np.ndarray,
    *,
    support_times: np.ndarray,
    idx_min: int,
    idx_max: int,
    t_extent: int,
    n_exp: int,
    e_min: float,
    e_max: float,
    e_grid: int,
    cov_reg: float,
) -> Optional[Dict[str, float]]:
    b = np.asarray(blocks, dtype=np.float64)
    n_blocks = int(b.shape[0])
    times = np.asarray(support_times, dtype=np.int64).reshape(-1)
    i0 = int(idx_min)
    i1 = int(idx_max)
    if i0 < 0 or i1 < i0 or i1 >= int(times.size):
        return None
    col_idx = np.arange(i0, i1 + 1, dtype=np.int64)
    t_coords = np.asarray(times[col_idx], dtype=np.int64)
    n_pts = int(t_coords.size)
    n_params = 2 * int(max(1, n_exp))
    if n_pts <= n_params:
        return None
    yb = b[:, col_idx]
    y = np.mean(yb, axis=0)
    cov_blocks = np.cov(yb, rowvar=False, ddof=1)
    if np.ndim(cov_blocks) == 0:
        cov_blocks = np.asarray([[float(cov_blocks)]], dtype=np.float64)
    cov_mean = np.asarray(cov_blocks, dtype=np.float64) / float(n_blocks)
    winv, eps = _regularized_inverse(cov_mean, reg=float(cov_reg))

    if int(n_exp) == 1:
        e_star, chi2, amp = _minimize_chi2_energy(
            y=y,
            t=t_coords.astype(np.float64),
            t_extent=int(t_extent),
            winv=winv,
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
        )
        if not np.isfinite(e_star) or not np.isfinite(chi2) or not np.isfinite(amp):
            return None
        dof = int(n_pts - n_params)
        if dof <= 0:
            return None
        chi2_dof = float(chi2 / float(dof))
        yfit = amp * _model_cosh(t=t_coords.astype(np.float64), t_extent=int(t_extent), energy=float(e_star))
        return {
            "fit_model": "1exp",
            "n_exp": 1.0,
            "tmin": float(t_coords[0]),
            "tmax": float(t_coords[-1]),
            "npts": float(n_pts),
            "amp": float(amp),
            "energy": float(e_star),
            "chi2": float(chi2),
            "chi2_dof": float(chi2_dof),
            "dof": float(dof),
            "cov_reg_eps": float(eps),
            "yfit": np.asarray(yfit, dtype=np.float64),
            "t_idx": np.asarray(t_coords, dtype=np.int64),
            "support_idx_min": float(i0),
            "support_idx_max": float(i1),
        }

    if int(n_exp) == 2:
        e0, e1, chi2, a0, a1 = _minimize_chi2_two_exp(
            y=y,
            t=t_coords.astype(np.float64),
            t_extent=int(t_extent),
            winv=winv,
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
        )
        if not (np.isfinite(e0) and np.isfinite(e1) and np.isfinite(chi2) and np.isfinite(a0) and np.isfinite(a1)):
            return None
        dof = int(n_pts - n_params)
        if dof <= 0:
            return None
        chi2_dof = float(chi2 / float(dof))
        t_float = t_coords.astype(np.float64)
        yfit = (
            a0 * _model_cosh(t=t_float, t_extent=int(t_extent), energy=float(e0))
            + a1 * _model_cosh(t=t_float, t_extent=int(t_extent), energy=float(e1))
        )
        return {
            "fit_model": "2exp",
            "n_exp": 2.0,
            "tmin": float(t_coords[0]),
            "tmax": float(t_coords[-1]),
            "npts": float(n_pts),
            "amp": float(a0),
            "energy": float(e0),
            "amp_1": float(a1),
            "energy_1": float(e1),
            "chi2": float(chi2),
            "chi2_dof": float(chi2_dof),
            "dof": float(dof),
            "cov_reg_eps": float(eps),
            "yfit": np.asarray(yfit, dtype=np.float64),
            "t_idx": np.asarray(t_coords, dtype=np.int64),
            "support_idx_min": float(i0),
            "support_idx_max": float(i1),
        }

    raise ValueError(f"Unsupported n_exp={n_exp}; expected 1 or 2")


def _select_window(
    blocks: np.ndarray,
    *,
    support_times: np.ndarray,
    t_extent: int,
    tmin_min: int,
    tmax_max: int,
    min_points: int,
    chi2_min: float,
    chi2_max: float,
    score_window_penalty: float,
    fallback_two_exp: bool,
    two_exp_min_points: int,
    progress: bool,
    progress_every: int,
    progress_prefix: str,
    e_min: float,
    e_max: float,
    e_grid: int,
    cov_reg: float,
) -> Tuple[Optional[Dict[str, float]], List[Dict[str, float]]]:
    support = np.asarray(support_times, dtype=np.int64).reshape(-1)
    if support.size == 0:
        return None, []
    valid = np.where((support >= int(tmin_min)) & (support <= int(tmax_max)))[0]
    if valid.size == 0:
        return None, []
    idx_lo = int(valid[0])
    idx_hi = int(valid[-1])

    def _scan(n_exp: int, min_pts_local: int) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        total = 0
        for i0 in range(idx_lo, idx_hi + 1):
            n_here = int(idx_hi - (i0 + int(min_pts_local) - 1) + 1)
            if n_here > 0:
                total += n_here

        stage = f"{int(n_exp)}exp"
        t0 = time.perf_counter()
        done = 0
        if bool(progress):
            print(
                f"[{progress_prefix}] scan {stage}: windows={total} min_points={int(min_pts_local)}",
                flush=True,
            )
        for i0 in range(idx_lo, idx_hi + 1):
            for i1 in range(i0 + int(min_pts_local) - 1, idx_hi + 1):
                done += 1
                fit = _fit_window_correlated(
                    blocks,
                    support_times=support,
                    idx_min=int(i0),
                    idx_max=int(i1),
                    t_extent=int(t_extent),
                    n_exp=int(n_exp),
                    e_min=float(e_min),
                    e_max=float(e_max),
                    e_grid=int(e_grid),
                    cov_reg=float(cov_reg),
                )
                if fit is None:
                    continue
                chi2_dof = float(fit["chi2_dof"])
                score = abs(math.log(max(1e-12, chi2_dof))) + float(score_window_penalty) / float(max(1.0, fit["npts"]))
                fit["score"] = float(score)
                fit["in_chi2_band"] = 1.0 if (chi2_dof >= float(chi2_min) and chi2_dof <= float(chi2_max)) else 0.0
                out.append(fit)
                if bool(progress) and int(progress_every) > 0:
                    if done % int(progress_every) == 0 or done == total:
                        in_band = sum(1 for c in out if int(c.get("in_chi2_band", 0.0)) == 1)
                        best_now = min(out, key=lambda c: float(c["score"])) if out else None
                        best_txt = (
                            f"best chi2/dof={float(best_now['chi2_dof']):.3f} win=[{int(best_now['tmin'])},{int(best_now['tmax'])}]"
                            if best_now is not None
                            else "best n/a"
                        )
                        elapsed = float(time.perf_counter() - t0)
                        print(
                            f"[{progress_prefix}] scan {stage}: {done}/{total} done,"
                            f" candidates={len(out)} in_band={in_band}"
                            f" {best_txt} elapsed={elapsed:.1f}s",
                            flush=True,
                        )

        if bool(progress):
            elapsed = float(time.perf_counter() - t0)
            in_band = sum(1 for c in out if int(c.get("in_chi2_band", 0.0)) == 1)
            print(
                f"[{progress_prefix}] scan {stage}: complete candidates={len(out)} in_band={in_band} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return out

    one_exp = _scan(n_exp=1, min_pts_local=int(min_points))
    if not one_exp:
        if bool(progress):
            print(f"[{progress_prefix}] no valid 1exp candidates", flush=True)
        return None, []

    one_exp_in_band = [c for c in one_exp if int(c.get("in_chi2_band", 0.0)) == 1]
    if one_exp_in_band:
        best = min(
            one_exp_in_band,
            key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])),
        )
        return best, one_exp

    if not bool(fallback_two_exp):
        if bool(progress):
            print(f"[{progress_prefix}] 1exp had no in-band windows; fallback-two-exp disabled", flush=True)
        best = min(
            one_exp,
            key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])),
        )
        return best, one_exp

    if bool(progress):
        print(f"[{progress_prefix}] 1exp had no in-band windows; trying 2exp fallback", flush=True)
    two_exp = _scan(n_exp=2, min_pts_local=max(int(min_points), int(two_exp_min_points)))
    if not two_exp:
        if bool(progress):
            print(f"[{progress_prefix}] no valid 2exp candidates; using best 1exp candidate", flush=True)
        best = min(
            one_exp,
            key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])),
        )
        return best, one_exp

    two_exp_in_band = [c for c in two_exp if int(c.get("in_chi2_band", 0.0)) == 1]
    pool = two_exp_in_band if two_exp_in_band else two_exp
    best = min(
        pool,
        key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])),
    )
    best = dict(best)
    best["selected_via_two_exp_fallback"] = 1.0
    if bool(progress):
        print(
            f"[{progress_prefix}] selected 2exp window"
            f" win=[{int(best['tmin'])},{int(best['tmax'])}]"
            f" chi2/dof={float(best['chi2_dof']):.3f}",
            flush=True,
        )
    return best, (one_exp + two_exp)


def _jackknife_fit_errors(
    blocks: np.ndarray,
    *,
    support_times: np.ndarray,
    best_idx_min: int,
    best_idx_max: int,
    t_extent: int,
    n_exp: int,
    e_min: float,
    e_max: float,
    e_grid: int,
    cov_reg: float,
) -> Dict[str, float]:
    b = np.asarray(blocks, dtype=np.float64)
    n_blocks = int(b.shape[0])
    if n_blocks < 3:
        out = {"amp_err": float("nan"), "energy_err": float("nan")}
        if int(n_exp) == 2:
            out["amp_1_err"] = float("nan")
            out["energy_1_err"] = float("nan")
        return out
    amps = np.full((n_blocks,), np.nan, dtype=np.float64)
    enes = np.full((n_blocks,), np.nan, dtype=np.float64)
    amps1 = np.full((n_blocks,), np.nan, dtype=np.float64)
    enes1 = np.full((n_blocks,), np.nan, dtype=np.float64)
    for j in range(n_blocks):
        loo = np.delete(b, j, axis=0)
        fit = _fit_window_correlated(
            loo,
            support_times=np.asarray(support_times, dtype=np.int64),
            idx_min=int(best_idx_min),
            idx_max=int(best_idx_max),
            t_extent=int(t_extent),
            n_exp=int(n_exp),
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
            cov_reg=float(cov_reg),
        )
        if fit is None:
            continue
        amps[j] = float(fit["amp"])
        enes[j] = float(fit["energy"])
        if int(n_exp) == 2:
            amps1[j] = float(fit.get("amp_1", float("nan")))
            enes1[j] = float(fit.get("energy_1", float("nan")))
    mask = np.isfinite(amps) & np.isfinite(enes)
    if np.count_nonzero(mask) < 2:
        out = {"amp_err": float("nan"), "energy_err": float("nan")}
        if int(n_exp) == 2:
            out["amp_1_err"] = float("nan")
            out["energy_1_err"] = float("nan")
        return out
    amps = amps[mask]
    enes = enes[mask]
    b_eff = float(amps.size)
    amp_mean = float(np.mean(amps))
    ene_mean = float(np.mean(enes))
    amp_err = float(np.sqrt((b_eff - 1.0) / b_eff * np.sum((amps - amp_mean) ** 2)))
    ene_err = float(np.sqrt((b_eff - 1.0) / b_eff * np.sum((enes - ene_mean) ** 2)))
    out = {"amp_err": amp_err, "energy_err": ene_err}
    if int(n_exp) == 2:
        mask1 = np.isfinite(amps1) & np.isfinite(enes1)
        if np.count_nonzero(mask1) >= 2:
            a1 = amps1[mask1]
            e1 = enes1[mask1]
            b1 = float(a1.size)
            out["amp_1_err"] = float(np.sqrt((b1 - 1.0) / b1 * np.sum((a1 - np.mean(a1)) ** 2)))
            out["energy_1_err"] = float(np.sqrt((b1 - 1.0) / b1 * np.sum((e1 - np.mean(e1)) ** 2)))
        else:
            out["amp_1_err"] = float("nan")
            out["energy_1_err"] = float("nan")
    return out


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> Dict[str, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ee = np.asarray(yerr, dtype=np.float64)
    m = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee) & (ee > 0.0)
    xx = xx[m]
    yy = yy[m]
    ee = ee[m]
    n = int(xx.size)
    if n < 2:
        return {
            "ok": False,
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_err": float("nan"),
            "slope_err": float("nan"),
            "chi2_dof": float("nan"),
            "npts": float(n),
        }
    w = 1.0 / (ee * ee)
    s = float(np.sum(w))
    sx = float(np.sum(w * xx))
    sy = float(np.sum(w * yy))
    sxx = float(np.sum(w * xx * xx))
    sxy = float(np.sum(w * xx * yy))
    det = s * sxx - sx * sx
    if not np.isfinite(det) or abs(det) < 1e-16:
        return {
            "ok": False,
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_err": float("nan"),
            "slope_err": float("nan"),
            "chi2_dof": float("nan"),
            "npts": float(n),
        }
    intercept = (sxx * sy - sx * sxy) / det
    slope = (s * sxy - sx * sy) / det
    intercept_err = math.sqrt(max(0.0, sxx / det))
    slope_err = math.sqrt(max(0.0, s / det))
    resid = yy - (intercept + slope * xx)
    chi2 = float(np.sum((resid / ee) ** 2))
    dof = max(1, n - 2)
    return {
        "ok": True,
        "intercept": float(intercept),
        "slope": float(slope),
        "intercept_err": float(intercept_err),
        "slope_err": float(slope_err),
        "chi2_dof": float(chi2 / dof),
        "npts": float(n),
    }


def _derive_m_c_from_dispersion(line_fit: Mapping[str, float]) -> Dict[str, float]:
    if not bool(line_fit.get("ok", False)):
        return {
            "m": float("nan"),
            "m_err": float("nan"),
            "c": float("nan"),
            "c_err": float("nan"),
        }
    m2 = float(line_fit.get("intercept", float("nan")))
    m2_err = float(line_fit.get("intercept_err", float("nan")))
    c2 = float(line_fit.get("slope", float("nan")))
    c2_err = float(line_fit.get("slope_err", float("nan")))

    if np.isfinite(m2) and m2 > 0.0:
        m = float(np.sqrt(m2))
        m_err = float(0.5 * m2_err / m) if np.isfinite(m2_err) else float("nan")
    else:
        m = float("nan")
        m_err = float("nan")

    if np.isfinite(c2) and c2 > 0.0:
        c = float(np.sqrt(c2))
        c_err = float(0.5 * c2_err / c) if np.isfinite(c2_err) else float("nan")
    else:
        c = float("nan")
        c_err = float("nan")

    return {
        "m": float(m),
        "m_err": float(m_err),
        "c": float(c),
        "c_err": float(c_err),
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

    try:
        if no_gui:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if no_gui:
            return None
        raise

    return plt


def _temporal_extent_from_payload(payload: Mapping, meta_by_p: Mapping[int, Mapping[str, float]]) -> int:
    cfg = payload.get("config", {})
    if isinstance(cfg, Mapping):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, Mapping):
            raw_shape = run_cfg.get("shape", None)
            if isinstance(raw_shape, (list, tuple)) and raw_shape:
                return int(raw_shape[-1])
    tmax = -1
    for meta in meta_by_p.values():
        support = meta.get("support_times", [])
        if isinstance(support, (list, tuple)) and support:
            tmax = max(tmax, max(int(v) for v in support))
    return int(max(1, tmax + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Correlated 2pt fits + dispersion from checkpoint inline records")
    ap.add_argument("--input", type=str, required=True, help="checkpoint .pkl from scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", type=str, default="pion_2pt", help="inline measurement name (e.g. pion_2pt, eta_2pt)")
    ap.add_argument("--channel", type=str, default="c", help="channel prefix (e.g. c, full, conn, disc)")
    ap.add_argument("--mom-axis", type=int, default=-1, help="override momentum axis; -1 means infer from checkpoint config")
    ap.add_argument(
        "--max-p-fit",
        type=int,
        default=-1,
        help="maximum |p| included in dispersion-line fit; <0 uses all fitted momenta",
    )
    ap.add_argument("--outdir", type=str, default="", help="output directory (default: <checkpoint_dir>/analysis)")
    ap.add_argument("--prefix", type=str, default="dispersion_fit")
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iat-method", type=str, default="ips", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0, help="<=0: auto from IAT")
    ap.add_argument("--tmin-min", type=int, default=1)
    ap.add_argument("--tmax-max", type=int, default=-1, help="-1 means T//2-1")
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--chi2-min", type=float, default=0.4)
    ap.add_argument("--chi2-max", type=float, default=2.0)
    ap.add_argument("--score-window-penalty", type=float, default=0.2)
    ap.add_argument(
        "--fallback-two-exp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if no 1exp window is inside chi2 band, try 2exp correlated fits",
    )
    ap.add_argument(
        "--two-exp-min-points",
        type=int,
        default=6,
        help="minimum points for 2exp fallback windows",
    )
    ap.add_argument("--e-min", type=float, default=1e-4)
    ap.add_argument("--e-max", type=float, default=4.0)
    ap.add_argument("--e-grid", type=int, default=256)
    ap.add_argument("--cov-reg", type=float, default=1e-10, help="fractional diagonal regularization for covariance inversion")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print intermediate progress during window scans",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="progress print cadence in number of scanned windows",
    )
    ap.add_argument("--title", type=str, default="Dispersion Relation")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {ckpt_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (ckpt_path.parent / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{args.prefix}_{args.measurement}_{args.channel}.png"
    json_path = outdir / f"{args.prefix}_{args.measurement}_{args.channel}.json"
    cov_path = outdir / f"{args.prefix}_{args.measurement}_{args.channel}_covariance.npz"

    payload, records = _load_checkpoint(ckpt_path)
    if not records:
        raise ValueError(f"No inline_records found in checkpoint: {ckpt_path}")

    samples_by_p, steps_by_p, meta_by_p = _extract_by_momentum(
        records,
        measurement=str(args.measurement),
        channel=str(args.channel),
    )

    for p in list(samples_by_p.keys()):
        s, st = _apply_discard_stride(
            samples_by_p[p], steps_by_p[p], discard=int(args.discard), stride=int(args.stride)
        )
        samples_by_p[p] = s
        steps_by_p[p] = st
    samples_by_p = {p: s for p, s in samples_by_p.items() if int(s.shape[0]) >= 4}
    if not samples_by_p:
        raise ValueError("No momentum channel has >=4 samples after discard/stride.")

    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        tau_ref = float("nan")
    else:
        block_size, tau_ref = _auto_block_size_from_p0(
            samples_by_p,
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
        )

    fit_results: List[Dict[str, float]] = []
    candidate_tables: Dict[str, List[Dict[str, float]]] = {}
    covariance_payload: Dict[str, np.ndarray] = {}
    t_extent_full = _temporal_extent_from_payload(payload, meta_by_p)
    for p in sorted(samples_by_p.keys()):
        s = np.asarray(samples_by_p[p], dtype=np.float64)
        support_times = np.asarray(meta_by_p[p].get("support_times", list(range(int(s.shape[1])))), dtype=np.int64)
        if int(support_times.size) != int(s.shape[1]):
            raise ValueError(
                f"Support-size mismatch for p={int(p)}: meta has {int(support_times.size)} times, samples have {int(s.shape[1])} columns"
            )
        t_extent = int(t_extent_full)
        if bool(args.progress):
            print(
                f"[p={int(p)}] start: n_samples={int(s.shape[0])} T={t_extent}"
                f" n_support={int(support_times.size)} measurement={args.measurement}/{args.channel}",
                flush=True,
            )
        blocks = _block_means(s, block_size=block_size)
        cov_blocks = np.cov(blocks, rowvar=False, ddof=1)
        if np.ndim(cov_blocks) == 0:
            cov_blocks = np.asarray([[float(cov_blocks)]], dtype=np.float64)
        cov_mean = np.asarray(cov_blocks, dtype=np.float64) / float(int(blocks.shape[0]))
        covariance_payload[f"p{int(p)}_times"] = np.asarray(support_times, dtype=np.int64)
        covariance_payload[f"p{int(p)}_mean"] = np.asarray(np.mean(blocks, axis=0), dtype=np.float64)
        covariance_payload[f"p{int(p)}_cov_blocks"] = np.asarray(cov_blocks, dtype=np.float64)
        covariance_payload[f"p{int(p)}_cov_mean"] = np.asarray(cov_mean, dtype=np.float64)

        tmax_default = (t_extent // 2) - 1
        support_half = support_times[support_times <= int(tmax_default)]
        tmax_auto = int(support_half[-1]) if support_half.size > 0 else int(support_times[-1])
        tmax_max = int(args.tmax_max) if int(args.tmax_max) >= 0 else int(tmax_auto)
        best, candidates = _select_window(
            blocks,
            support_times=support_times,
            t_extent=t_extent,
            tmin_min=int(args.tmin_min),
            tmax_max=int(tmax_max),
            min_points=int(args.min_points),
            chi2_min=float(args.chi2_min),
            chi2_max=float(args.chi2_max),
            score_window_penalty=float(args.score_window_penalty),
            fallback_two_exp=bool(args.fallback_two_exp),
            two_exp_min_points=int(args.two_exp_min_points),
            progress=bool(args.progress),
            progress_every=int(args.progress_every),
            progress_prefix=f"p={int(p)}",
            e_min=float(args.e_min),
            e_max=float(args.e_max),
            e_grid=int(args.e_grid),
            cov_reg=float(args.cov_reg),
        )
        if best is None:
            if bool(args.progress):
                print(f"[p={int(p)}] no successful fit candidate", flush=True)
            continue
        n_exp_fit = int(best.get("n_exp", 1.0))
        jk = _jackknife_fit_errors(
            blocks,
            support_times=support_times,
            best_idx_min=int(best["support_idx_min"]),
            best_idx_max=int(best["support_idx_max"]),
            t_extent=t_extent,
            n_exp=n_exp_fit,
            e_min=float(args.e_min),
            e_max=float(args.e_max),
            e_grid=int(args.e_grid),
            cov_reg=float(args.cov_reg),
        )
        fit_results.append(
            {
                "p": float(p),
                "n_samples": float(s.shape[0]),
                "t_extent": float(t_extent),
                "fit_model": str(best.get("fit_model", "1exp")),
                "n_exp": float(n_exp_fit),
                "tmin": float(best["tmin"]),
                "tmax": float(best["tmax"]),
                "npts": float(best["npts"]),
                "support_idx_min": float(best["support_idx_min"]),
                "support_idx_max": float(best["support_idx_max"]),
                "support_times": np.asarray(support_times, dtype=np.int64).astype(int).tolist(),
                "window_times": np.asarray(best["t_idx"], dtype=np.int64).astype(int).tolist(),
                "energy": float(best["energy"]),
                "energy_err": float(jk["energy_err"]),
                "amp": float(best["amp"]),
                "amp_err": float(jk["amp_err"]),
                "chi2": float(best["chi2"]),
                "chi2_dof": float(best["chi2_dof"]),
                "dof": float(best["dof"]),
                "score": float(best["score"]),
                "in_chi2_band": float(best["in_chi2_band"]),
                "selected_via_two_exp_fallback": float(best.get("selected_via_two_exp_fallback", 0.0)),
            }
        )
        if n_exp_fit == 2:
            fit_results[-1]["energy_1"] = float(best.get("energy_1", float("nan")))
            fit_results[-1]["energy_1_err"] = float(jk.get("energy_1_err", float("nan")))
            fit_results[-1]["amp_1"] = float(best.get("amp_1", float("nan")))
            fit_results[-1]["amp_1_err"] = float(jk.get("amp_1_err", float("nan")))
        if bool(args.progress):
            print(
                f"[p={int(p)}] selected {fit_results[-1]['fit_model']}"
                f" win=[{int(fit_results[-1]['tmin'])},{int(fit_results[-1]['tmax'])}]"
                f" E={float(fit_results[-1]['energy']):.6f}"
                f" chi2/dof={float(fit_results[-1]['chi2_dof']):.3f}",
                flush=True,
            )

        compact_candidates: List[Dict[str, float]] = []
        for c in candidates:
            compact_candidates.append(
                {
                    "fit_model": str(c.get("fit_model", "1exp")),
                    "n_exp": float(c.get("n_exp", 1.0)),
                    "tmin": float(c["tmin"]),
                    "tmax": float(c["tmax"]),
                    "npts": float(c["npts"]),
                    "support_idx_min": float(c.get("support_idx_min", float("nan"))),
                    "support_idx_max": float(c.get("support_idx_max", float("nan"))),
                    "window_times": np.asarray(c.get("t_idx", []), dtype=np.int64).astype(int).tolist(),
                    "energy": float(c["energy"]),
                    "amp": float(c["amp"]),
                    "energy_1": float(c.get("energy_1", float("nan"))),
                    "amp_1": float(c.get("amp_1", float("nan"))),
                    "chi2_dof": float(c["chi2_dof"]),
                    "score": float(c["score"]),
                    "in_chi2_band": float(c["in_chi2_band"]),
                }
            )
        candidate_tables[str(p)] = compact_candidates

    if not fit_results:
        raise ValueError("No successful fits were found for any momentum.")
    fit_results = sorted(fit_results, key=lambda d: int(d["p"]))

    # Momentum conversion from checkpoint config.
    cfg = payload.get("config", {})
    shape = None
    if isinstance(cfg, dict):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, dict):
            raw_shape = run_cfg.get("shape", None)
            if isinstance(raw_shape, (list, tuple)) and raw_shape:
                shape = tuple(int(v) for v in raw_shape)
    mom_axis = int(args.mom_axis)
    if mom_axis < 0:
        mom_axis = 0
    l_axis = float(shape[mom_axis]) if (shape is not None and 0 <= mom_axis < len(shape)) else float("nan")

    p_int = np.asarray([int(r["p"]) for r in fit_results], dtype=np.int64)
    en = np.asarray([float(r["energy"]) for r in fit_results], dtype=np.float64)
    en_err = np.asarray([float(r["energy_err"]) for r in fit_results], dtype=np.float64)
    if np.isfinite(l_axis) and l_axis > 0:
        p_cont = 2.0 * np.pi * p_int.astype(np.float64) / float(l_axis)
        p_hat = 2.0 * np.sin(np.pi * p_int.astype(np.float64) / float(l_axis))
    else:
        p_cont = p_int.astype(np.float64)
        p_hat = p_int.astype(np.float64)
    p2_hat = p_hat * p_hat
    e2 = en * en
    e2_err = 2.0 * np.abs(en) * np.where(np.isfinite(en_err), en_err, np.nan)

    max_p_fit = int(args.max_p_fit)
    if max_p_fit >= 0:
        fit_mask = np.abs(p_int) <= int(max_p_fit)
    else:
        fit_mask = np.ones_like(p_int, dtype=bool)
    p2_hat_fit = p2_hat[fit_mask]
    e2_fit = e2[fit_mask]
    e2_err_fit = e2_err[fit_mask]
    p_int_fit = p_int[fit_mask]
    disp_fit = _weighted_line_fit(p2_hat_fit, e2_fit, e2_err_fit)
    disp_mc = _derive_m_c_from_dispersion(disp_fit)

    plt = _setup_matplotlib(no_gui=bool(args.no_gui))
    png_written = False
    if plt is not None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8))

        ax0.errorbar(p_int, en, yerr=en_err, fmt="o", ms=5, capsize=3, color="#1f77b4")
        ax0.set_xlabel("momentum index p")
        ax0.set_ylabel("E(p)")
        ax0.set_title(f"{args.measurement}:{args.channel} energies")
        ax0.grid(True, alpha=0.25)

        if np.any(~fit_mask):
            ax1.errorbar(
                p2_hat[~fit_mask],
                e2[~fit_mask],
                yerr=e2_err[~fit_mask],
                fmt="o",
                ms=4,
                capsize=2,
                color="#bdbdbd",
                label="excluded points",
            )
        ax1.errorbar(
            p2_hat_fit,
            e2_fit,
            yerr=e2_err_fit,
            fmt="o",
            ms=5,
            capsize=3,
            color="#d62728",
            label="fit points",
        )
        if bool(disp_fit.get("ok", False)):
            xline = np.linspace(float(np.min(p2_hat_fit)), float(np.max(p2_hat_fit)), 200)
            yline = float(disp_fit["intercept"]) + float(disp_fit["slope"]) * xline
            ax1.plot(xline, yline, "-", color="#9467bd", lw=1.4, label="weighted linear fit")
            lbl = (
                f"E^2 = m^2 + c^2 p_hat^2\n"
                f"m={float(disp_mc['m']):.4f}±{float(disp_mc['m_err']):.4f}\n"
                f"c={float(disp_mc['c']):.4f}±{float(disp_mc['c_err']):.4f}\n"
                f"chi2/dof={float(disp_fit['chi2_dof']):.3f}"
            )
            ax1.text(
                0.03,
                0.97,
                lbl,
                transform=ax1.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        ax1.set_xlabel(r"$\hat{p}^{\,2}$")
        ax1.set_ylabel(r"$E^2$")
        ax1.set_title("Dispersion relation")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        subtitle = (
            f"block={block_size}"
            + (f" (auto tau_ref~{tau_ref:.2f})" if np.isfinite(tau_ref) else "")
            + f" | n_mom={len(fit_results)}"
        )
        fig.suptitle(str(args.title))
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        png_written = True

    out = {
        "input_checkpoint": str(ckpt_path),
        "measurement": str(args.measurement),
        "channel": str(args.channel),
        "shape": (None if shape is None else list(shape)),
        "mom_axis": int(mom_axis),
        "l_axis": float(l_axis),
        "settings": {
            "discard": int(args.discard),
            "stride": int(args.stride),
            "iat_method": str(args.iat_method),
            "iat_c": float(args.iat_c),
            "iat_max_lag": (None if iat_max_lag is None else int(iat_max_lag)),
            "block_size": int(block_size),
            "tmin_min": int(args.tmin_min),
            "tmax_max": int(args.tmax_max),
            "min_points": int(args.min_points),
            "chi2_min": float(args.chi2_min),
            "chi2_max": float(args.chi2_max),
            "score_window_penalty": float(args.score_window_penalty),
            "fallback_two_exp": bool(args.fallback_two_exp),
            "two_exp_min_points": int(args.two_exp_min_points),
            "e_min": float(args.e_min),
            "e_max": float(args.e_max),
            "e_grid": int(args.e_grid),
            "cov_reg": float(args.cov_reg),
            "max_p_fit": int(args.max_p_fit),
            "progress": bool(args.progress),
            "progress_every": int(args.progress_every),
        },
        "fit_results": fit_results,
        "candidate_windows": candidate_tables,
        "dispersion": {
            "p_int": p_int.astype(int).tolist(),
            "p_cont": np.asarray(p_cont, dtype=np.float64).tolist(),
            "p_hat": np.asarray(p_hat, dtype=np.float64).tolist(),
            "p2_hat": np.asarray(p2_hat, dtype=np.float64).tolist(),
            "energy": np.asarray(en, dtype=np.float64).tolist(),
            "energy_err": np.asarray(en_err, dtype=np.float64).tolist(),
            "e2": np.asarray(e2, dtype=np.float64).tolist(),
            "e2_err": np.asarray(e2_err, dtype=np.float64).tolist(),
            "p_int_used_for_dispersion_fit": np.asarray(p_int_fit, dtype=np.int64).astype(int).tolist(),
            "linear_fit": disp_fit,
            "derived_mc": disp_mc,
        },
        "momentum_meta": {str(k): v for k, v in meta_by_p.items()},
        "output_png": (str(png_path) if bool(png_written) else ""),
        "output_json": str(json_path),
        "output_covariance_npz": str(cov_path),
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(cov_path, **covariance_payload)

    print("Dispersion Fit Summary")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  measurement/channel: {args.measurement}/{args.channel}")
    print(f"  fitted momenta: {[int(r['p']) for r in fit_results]}")
    if max_p_fit >= 0:
        print(f"  dispersion fit momentum cut: |p| <= {max_p_fit} (used {p_int_fit.astype(int).tolist()})")
    else:
        print(f"  dispersion fit momentum cut: none (used {p_int_fit.astype(int).tolist()})")
    print(f"  block size: {block_size}" + (f" (auto tau_ref~{tau_ref:.2f})" if np.isfinite(tau_ref) else ""))
    for r in fit_results:
        print(
            f"  p={int(r['p'])}:"
            f" model={r.get('fit_model','1exp')}"
            f" E={float(r['energy']):.6f} +/- {float(r['energy_err']):.6f}"
            f"  win=[{int(r['tmin'])},{int(r['tmax'])}]"
            f"  chi2/dof={float(r['chi2_dof']):.3f}"
        )
    if bool(disp_fit.get("ok", False)):
        print(
            "  dispersion linear fit:"
            f" m={float(disp_mc['m']):.6f} +/- {float(disp_mc['m_err']):.6f},"
            f" c={float(disp_mc['c']):.6f} +/- {float(disp_mc['c_err']):.6f},"
            f" chi2/dof={float(disp_fit['chi2_dof']):.3f}"
        )
    else:
        print("  dispersion linear fit: insufficient valid points")
    if bool(png_written):
        print(f"  saved figure: {png_path}")
    else:
        print("  saved figure: skipped (matplotlib unavailable)")
    print(f"  saved json:   {json_path}")
    print(f"  saved covariance: {cov_path}")

    if (plt is not None) and (not args.no_gui):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

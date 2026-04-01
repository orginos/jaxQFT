#!/usr/bin/env python3
"""Extract pion vector-current matrix elements from 2pt/3pt checkpoint data.

This script expects checkpoint inline records produced by `scripts/mcmc/mcmc.py`.

Two-point input:
  - measurement name: usually `pion_2pt`
  - channel keys: `c_p{p}_t{t}_re`, `c_p{p}_t{t}_im`

Three-point input:
  - measurement name: usually `pion_3pt_vector`
  - channel keys:
      `c3_mu{mu}_pf{pf}_pi{pi}_tsep{tsep}_tau{tau}_re`
      `c3_mu{mu}_pf{pf}_pi{pi}_tsep{tsep}_tau{tau}_im`

For one selected kinematic channel `(mu, p_i, p_f, t_sep)`, the script:
1) fits the pion 2pt functions at `p_i` and `p_f`,
2) forms `Z(p) = sqrt(2 E(p) A(p))` from `C_2(t,p) = A(p)[e^{-Et}+e^{-E(T-t)}]`,
3) constructs two effective matrix-element estimators:
     - direct overlap removal from fitted `Z(p)`,
     - the standard 2pt/3pt ratio,
4) selects a plateau window by correlated constant fits,
5) reports the bare and renormalized matrix element, and, whenever the
   kinematic prefactor is nonzero, the pion form factor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{name}' from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIT2PT = _load_script_module("fit_2pt_dispersion_mod", ROOT / "scripts" / "mcmc" / "fit_2pt_dispersion.py")


def _block_means_nd(samples: np.ndarray, block_size: int) -> np.ndarray:
    arr = np.asarray(samples)
    n = int(arr.shape[0])
    b = max(1, int(block_size))
    nblock = n // b
    if nblock < 2:
        raise ValueError(f"Need at least 2 blocks: n={n}, block_size={b}")
    used = int(nblock * b)
    trimmed = np.asarray(arr[:used])
    return np.asarray(trimmed.reshape((nblock, b) + trimmed.shape[1:]).mean(axis=1))


def _jackknife_loo_from_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks)
    nb = int(b.shape[0])
    if nb < 2:
        raise ValueError("Need at least 2 blocks for jackknife")
    total = np.sum(b, axis=0)
    out = np.empty_like(b)
    for ib in range(nb):
        out[ib] = (total - b[ib]) / float(nb - 1)
    return out


def _jackknife_err_from_replicas(jk: np.ndarray, full: np.ndarray) -> np.ndarray:
    arr = np.asarray(jk)
    center = np.asarray(full)
    nb = int(arr.shape[0])
    if nb <= 1:
        return np.full_like(center, np.nan, dtype=np.float64)
    if np.iscomplexobj(arr) or np.iscomplexobj(center):
        dif = arr - center
        return np.sqrt((nb - 1) / float(nb) * np.sum(np.abs(dif) ** 2, axis=0))
    dif = np.asarray(arr - center, dtype=np.float64)
    return np.sqrt((nb - 1) / float(nb) * np.sum(dif * dif, axis=0))


def _jackknife_covariance(jk: np.ndarray) -> np.ndarray:
    arr = np.asarray(jk, dtype=np.float64)
    nb = int(arr.shape[0])
    if nb <= 1:
        return np.full((arr.shape[-1], arr.shape[-1]), np.nan, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    dif = arr - mean
    return np.asarray((nb - 1.0) / float(nb) * (dif.T @ dif), dtype=np.float64)


def _scalar_jackknife_err(jk: np.ndarray, full: float) -> float:
    err = _jackknife_err_from_replicas(
        np.asarray(jk, dtype=np.float64)[:, None],
        np.asarray([float(full)], dtype=np.float64),
    )
    return float(np.asarray(err, dtype=np.float64).reshape(-1)[0])


def _extract_threepoint_channel(
    records: Sequence[Mapping],
    *,
    measurement: str,
    channel: str,
    mu: int,
    p_i: int,
    p_f: int,
    t_sep: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    chan = str(channel)
    pat_re = re.compile(
        rf"^{re.escape(chan)}_mu(?P<mu>\d+)_pf(?P<pf>-?\d+)_pi(?P<pi>-?\d+)_tsep(?P<tsep>\d+)_tau(?P<tau>\d+)_re$"
    )
    pat_im = re.compile(
        rf"^{re.escape(chan)}_mu(?P<mu>\d+)_pf(?P<pf>-?\d+)_pi(?P<pi>-?\d+)_tsep(?P<tsep>\d+)_tau(?P<tau>\d+)_im$"
    )

    by_step: Dict[int, Dict[int, complex]] = {}
    imag_abs_max = 0.0
    for rec in records:
        if str(rec.get("name", "")) != str(measurement):
            continue
        vals = rec.get("values", {})
        if not isinstance(vals, Mapping):
            continue
        step = int(rec.get("step", -1))
        row = by_step.setdefault(step, {})
        for k, v in vals.items():
            key = str(k)
            m_re = pat_re.match(key)
            if m_re is not None:
                if (
                    int(m_re.group("mu")) == int(mu)
                    and int(m_re.group("pf")) == int(p_f)
                    and int(m_re.group("pi")) == int(p_i)
                    and int(m_re.group("tsep")) == int(t_sep)
                ):
                    tau = int(m_re.group("tau"))
                    row[tau] = complex(float(v), row.get(tau, 0.0j).imag)
                continue
            m_im = pat_im.match(key)
            if m_im is not None:
                if (
                    int(m_im.group("mu")) == int(mu)
                    and int(m_im.group("pf")) == int(p_f)
                    and int(m_im.group("pi")) == int(p_i)
                    and int(m_im.group("tsep")) == int(t_sep)
                ):
                    tau = int(m_im.group("tau"))
                    row[tau] = complex(row.get(tau, 0.0j).real, float(v))
                    imag_abs_max = max(imag_abs_max, abs(float(v)))

    if not by_step:
        raise ValueError(
            "No matching 3pt data found for "
            f"measurement='{measurement}' channel='{channel}' mu={mu} pf={p_f} pi={p_i} tsep={t_sep}"
        )

    tau_sets = [set(v.keys()) for v in by_step.values() if v]
    if not tau_sets:
        raise ValueError("Matched 3pt records exist but contain no tau samples")
    support = sorted(set.intersection(*tau_sets))
    if not support:
        raise ValueError("No common tau support remains across 3pt samples")

    rows: List[np.ndarray] = []
    steps: List[int] = []
    dropped = 0
    for step in sorted(by_step.keys()):
        vals = by_step[step]
        row = np.asarray([vals.get(int(t), np.nan + 1j * np.nan) for t in support], dtype=np.complex128)
        if np.all(np.isfinite(np.real(row))) and np.all(np.isfinite(np.imag(row))):
            rows.append(row)
            steps.append(int(step))
        else:
            dropped += 1
    if not rows:
        raise ValueError("No complete 3pt samples remained after completeness filtering")

    return (
        np.asarray(rows, dtype=np.complex128),
        np.asarray(steps, dtype=np.int64),
        {
            "support_taus": [int(t) for t in support],
            "imag_abs_max": float(imag_abs_max),
            "n_total": float(len(by_step)),
            "n_kept": float(len(rows)),
            "n_dropped_incomplete": float(dropped),
        },
    )


def _align_common_steps(
    pion_samples_by_p: Mapping[int, np.ndarray],
    pion_steps_by_p: Mapping[int, np.ndarray],
    required_momenta: Sequence[int],
    three_samples: np.ndarray,
    three_steps: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    req = sorted({int(p) for p in required_momenta})
    common = set(int(s) for s in np.asarray(three_steps, dtype=np.int64).tolist())
    for p in req:
        if int(p) not in pion_steps_by_p:
            raise ValueError(f"Required pion momentum p={int(p)} is missing from pion_2pt records")
        common &= set(int(s) for s in np.asarray(pion_steps_by_p[int(p)], dtype=np.int64).tolist())
    steps = np.asarray(sorted(common), dtype=np.int64)
    if steps.size < 4:
        raise ValueError(f"Need at least 4 common pion/3pt samples; got {int(steps.size)}")

    three_index = {int(s): i for i, s in enumerate(np.asarray(three_steps, dtype=np.int64).tolist())}
    three_aligned = np.asarray([three_samples[three_index[int(s)]] for s in steps], dtype=np.complex128)

    pion_aligned: Dict[int, np.ndarray] = {}
    for p in req:
        st = np.asarray(pion_steps_by_p[int(p)], dtype=np.int64)
        idx = {int(s): i for i, s in enumerate(st.tolist())}
        pion_aligned[int(p)] = np.asarray([pion_samples_by_p[int(p)][idx[int(s)]] for s in steps], dtype=np.float64)
    return pion_aligned, three_aligned, steps


def _apply_common_discard_stride(
    pion_samples_by_p: Mapping[int, np.ndarray],
    three_samples: np.ndarray,
    steps: np.ndarray,
    *,
    discard: int,
    stride: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    d = max(0, int(discard))
    s = max(1, int(stride))
    keep = np.arange(int(steps.size), dtype=np.int64)[d::s]
    if keep.size < 4:
        raise ValueError(f"Need at least 4 common samples after discard/stride; got {int(keep.size)}")
    pion_out = {int(p): np.asarray(arr[keep], dtype=np.float64) for p, arr in pion_samples_by_p.items()}
    three_out = np.asarray(three_samples[keep], dtype=np.complex128)
    steps_out = np.asarray(steps[keep], dtype=np.int64)
    return pion_out, three_out, steps_out


def _auto_block_size_from_threepoint(
    samples: np.ndarray,
    *,
    support_taus: np.ndarray,
    tau_ref: int,
    iat_method: str,
    iat_c: float,
    iat_max_lag: int | None,
    component: str,
) -> Tuple[int, float]:
    taus = np.asarray(support_taus, dtype=np.int64).reshape(-1)
    if taus.size == 0:
        return 1, float("nan")
    idx = int(np.argmin(np.abs(taus - int(tau_ref))))
    series = _component_view(np.asarray(samples[:, idx], dtype=np.complex128), component)
    iat = FIT2PT.integrated_autocorr_time(
        series,
        method=str(iat_method),
        c=float(iat_c),
        max_lag=iat_max_lag,
    )
    tau = float(iat.get("tau_int", float("nan")))
    if not np.isfinite(tau) or tau < 0.5:
        tau = 1.0
    n = int(samples.shape[0])
    block = max(1, int(math.ceil(2.0 * tau)))
    if n // block < 4 and n >= 4:
        block = max(1, n // 4)
    if n // block < 2:
        block = max(1, n // 2)
    return int(block), float(tau)


def _choose_joint_block_size(
    pion_samples_by_p: Mapping[int, np.ndarray],
    three_samples: np.ndarray,
    *,
    support_taus: np.ndarray,
    tau_ref: int,
    iat_method: str,
    iat_c: float,
    iat_max_lag: int | None,
    component: str,
) -> Tuple[int, Dict[str, float]]:
    block_pi, tau_pi = FIT2PT._auto_block_size_from_p0(
        pion_samples_by_p,
        iat_method=str(iat_method),
        iat_c=float(iat_c),
        iat_max_lag=iat_max_lag,
    )
    block_3, tau_3 = _auto_block_size_from_threepoint(
        three_samples,
        support_taus=np.asarray(support_taus, dtype=np.int64),
        tau_ref=int(tau_ref),
        iat_method=str(iat_method),
        iat_c=float(iat_c),
        iat_max_lag=iat_max_lag,
        component=str(component),
    )
    block = max(int(block_pi), int(block_3))
    n = int(next(iter(pion_samples_by_p.values())).shape[0])
    while block > 1 and (n // block) < 2:
        block -= 1
    return int(max(1, block)), {
        "pion_tau_ref": float(tau_pi),
        "pion_block_auto": float(block_pi),
        "threept_tau_ref": float(tau_3),
        "threept_block_auto": float(block_3),
    }


def _parse_fit_range_text(text: str) -> Tuple[int, int]:
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    if len(toks) != 2:
        raise ValueError(f"fit range must have exactly 2 integers, got {text!r}")
    a = int(toks[0])
    b = int(toks[1])
    if a > b:
        a, b = b, a
    return int(a), int(b)


def _support_index_range(support_times: np.ndarray, tmin: int, tmax: int) -> Tuple[int, int]:
    support = np.asarray(support_times, dtype=np.int64).reshape(-1)
    idx_map = {int(t): i for i, t in enumerate(support.tolist())}
    if int(tmin) not in idx_map or int(tmax) not in idx_map:
        raise ValueError(
            f"Requested range [{int(tmin)},{int(tmax)}] is not supported by available times {support.tolist()}"
        )
    i0 = int(idx_map[int(tmin)])
    i1 = int(idx_map[int(tmax)])
    if i1 < i0:
        i0, i1 = i1, i0
    return int(i0), int(i1)


def _fit_pion_channel(
    samples: np.ndarray,
    *,
    support_times: np.ndarray,
    t_extent: int,
    block_size: int,
    fit_range: Optional[Tuple[int, int]],
    tmin_min: int,
    tmax_max: int,
    min_points: int,
    chi2_min: float,
    chi2_max: float,
    score_window_penalty: float,
    fallback_two_exp: bool,
    two_exp_min_points: int,
    e_min: float,
    e_max: float,
    e_grid: int,
    cov_reg: float,
    progress: bool,
    progress_prefix: str,
) -> Dict[str, object]:
    s = np.asarray(samples, dtype=np.float64)
    support = np.asarray(support_times, dtype=np.int64)
    blocks = FIT2PT._block_means(s, block_size=block_size)
    full_corr = np.asarray(np.mean(blocks, axis=0), dtype=np.float64)
    jk_corr = _jackknife_loo_from_blocks(blocks)

    if fit_range is not None:
        idx_min, idx_max = _support_index_range(support, int(fit_range[0]), int(fit_range[1]))
        best = FIT2PT._fit_window_correlated(
            blocks,
            support_times=support,
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            t_extent=int(t_extent),
            n_exp=1,
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
            cov_reg=float(cov_reg),
        )
        if best is None:
            raise ValueError(f"Manual pion fit window [{fit_range[0]},{fit_range[1]}] failed")
        candidates: List[Dict[str, float]] = []
    else:
        best, candidates = FIT2PT._select_window(
            blocks,
            support_times=support,
            t_extent=int(t_extent),
            tmin_min=int(tmin_min),
            tmax_max=int(tmax_max),
            min_points=int(min_points),
            chi2_min=float(chi2_min),
            chi2_max=float(chi2_max),
            score_window_penalty=float(score_window_penalty),
            fallback_two_exp=bool(fallback_two_exp),
            two_exp_min_points=int(two_exp_min_points),
            progress=bool(progress),
            progress_every=200,
            progress_prefix=str(progress_prefix),
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
            cov_reg=float(cov_reg),
        )
        if best is None:
            raise ValueError(f"No successful pion fit candidate for {progress_prefix}")

    idx_min = int(best["support_idx_min"])
    idx_max = int(best["support_idx_max"])
    n_exp = int(best.get("n_exp", 1.0))
    jk_amp = np.full((blocks.shape[0],), np.nan, dtype=np.float64)
    jk_energy = np.full((blocks.shape[0],), np.nan, dtype=np.float64)
    jk_amp_1 = np.full((blocks.shape[0],), np.nan, dtype=np.float64)
    jk_energy_1 = np.full((blocks.shape[0],), np.nan, dtype=np.float64)
    for ib in range(int(blocks.shape[0])):
        loo = np.delete(blocks, ib, axis=0)
        fit = FIT2PT._fit_window_correlated(
            loo,
            support_times=support,
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            t_extent=int(t_extent),
            n_exp=int(n_exp),
            e_min=float(e_min),
            e_max=float(e_max),
            e_grid=int(e_grid),
            cov_reg=float(cov_reg),
        )
        if fit is None:
            continue
        jk_amp[ib] = float(fit["amp"])
        jk_energy[ib] = float(fit["energy"])
        if int(n_exp) == 2:
            jk_amp_1[ib] = float(fit.get("amp_1", float("nan")))
            jk_energy_1[ib] = float(fit.get("energy_1", float("nan")))

    full_amp = float(best["amp"])
    full_energy = float(best["energy"])
    full_z = float(np.sqrt(max(0.0, 2.0 * full_energy * full_amp))) if (full_amp > 0.0 and full_energy > 0.0) else float("nan")
    jk_z = np.full_like(jk_amp, np.nan)
    mask = np.isfinite(jk_amp) & np.isfinite(jk_energy) & (jk_amp > 0.0) & (jk_energy > 0.0)
    jk_z[mask] = np.sqrt(2.0 * jk_energy[mask] * jk_amp[mask])

    out: Dict[str, object] = {
        "blocks": blocks,
        "full_corr": full_corr,
        "jk_corr": jk_corr,
        "support_times": support,
        "fit_model": str(best.get("fit_model", "1exp")),
        "n_exp": float(n_exp),
        "energy": float(full_energy),
        "amp": float(full_amp),
        "z_overlap": float(full_z),
        "energy_err": float(_scalar_jackknife_err(jk_energy, full_energy)),
        "amp_err": float(_scalar_jackknife_err(jk_amp, full_amp)),
        "z_overlap_err": float(_scalar_jackknife_err(jk_z, full_z)),
        "jk_energy": jk_energy,
        "jk_amp": jk_amp,
        "jk_z": jk_z,
        "tmin": float(best["tmin"]),
        "tmax": float(best["tmax"]),
        "chi2_dof": float(best["chi2_dof"]),
        "score": float(best.get("score", float("nan"))),
        "window_times": np.asarray(best["t_idx"], dtype=np.int64).astype(int).tolist(),
        "support_idx_min": float(idx_min),
        "support_idx_max": float(idx_max),
    }
    if int(n_exp) == 2:
        full_amp_1 = float(best.get("amp_1", float("nan")))
        full_energy_1 = float(best.get("energy_1", float("nan")))
        out["amp_1"] = float(full_amp_1)
        out["energy_1"] = float(full_energy_1)
        out["amp_1_err"] = float(_scalar_jackknife_err(jk_amp_1, full_amp_1))
        out["energy_1_err"] = float(_scalar_jackknife_err(jk_energy_1, full_energy_1))
    if candidates:
        compact_candidates: List[Dict[str, float]] = []
        for c in candidates:
            compact_candidates.append(
                {
                    "fit_model": str(c.get("fit_model", "1exp")),
                    "tmin": float(c["tmin"]),
                    "tmax": float(c["tmax"]),
                    "chi2_dof": float(c["chi2_dof"]),
                    "score": float(c["score"]),
                    "energy": float(c["energy"]),
                    "amp": float(c["amp"]),
                    "in_chi2_band": float(c["in_chi2_band"]),
                }
            )
        out["fit_candidates"] = compact_candidates
    return out


def _lookup_times(arr: np.ndarray, support_times: np.ndarray, query_times: Sequence[int]) -> np.ndarray:
    support = np.asarray(support_times, dtype=np.int64).reshape(-1)
    idx_map = {int(t): i for i, t in enumerate(support.tolist())}
    q = [idx_map.get(int(t), None) for t in query_times]
    out = np.full((len(q),), np.nan if not np.iscomplexobj(arr) else np.nan + 1j * np.nan, dtype=np.asarray(arr).dtype)
    for i, idx in enumerate(q):
        if idx is not None:
            out[i] = np.asarray(arr)[int(idx)]
    return out


def _forward_two_point_from_fit(times: Sequence[int], *, amp: float, energy: float) -> np.ndarray:
    tt = np.asarray([int(t) for t in times], dtype=np.float64)
    if not (np.isfinite(float(amp)) and np.isfinite(float(energy))):
        return np.full((tt.size,), np.nan, dtype=np.float64)
    return np.asarray(float(amp) * np.exp(-float(energy) * tt), dtype=np.float64)


def _component_view(arr: np.ndarray, component: str) -> np.ndarray:
    comp = str(component).strip().lower()
    if comp == "re":
        return np.asarray(np.real(arr), dtype=np.float64)
    if comp == "im":
        return np.asarray(np.imag(arr), dtype=np.float64)
    if comp == "abs":
        return np.asarray(np.abs(arr), dtype=np.float64)
    raise ValueError(f"Unsupported component: {component!r}")


def _resolve_analysis_component(payload: Mapping, *, mu: int, component: str) -> Tuple[str, float, str]:
    comp = str(component).strip().lower()
    if comp == "auto":
        temporal_mu = _temporal_mu_from_payload(payload)
        if int(mu) == int(temporal_mu):
            return "re", 1.0, "Re"
        return "im", -1.0, "-Im"
    if comp == "re":
        return "re", 1.0, "Re"
    if comp == "im":
        return "im", 1.0, "Im"
    if comp == "abs":
        return "abs", 1.0, "|.|"
    raise ValueError(f"Unsupported component: {component!r}")


def _analysis_component_view(arr: np.ndarray, *, payload: Mapping, mu: int, component: str) -> Tuple[np.ndarray, Dict[str, object]]:
    resolved_component, resolved_sign, resolved_label = _resolve_analysis_component(
        payload,
        mu=int(mu),
        component=str(component),
    )
    view = _component_view(arr, resolved_component)
    if resolved_sign != 1.0:
        view = np.asarray(float(resolved_sign) * np.asarray(view, dtype=np.float64), dtype=np.float64)
    meta = {
        "requested_component": str(component),
        "resolved_component": str(resolved_component),
        "resolved_sign": float(resolved_sign),
        "resolved_label": str(resolved_label),
    }
    return view, meta


def _sqrt_complex_safe(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    out = np.sqrt(arr)
    bad = ~np.isfinite(np.real(out)) | ~np.isfinite(np.imag(out))
    if np.any(bad):
        out[bad] = np.nan + 1j * np.nan
    return out


def _effective_ratio_curve(
    c3: np.ndarray,
    c2_i: np.ndarray,
    c2_f: np.ndarray,
    *,
    taus: np.ndarray,
    t_sep: int,
    support_times_i: np.ndarray,
    support_times_f: np.ndarray,
    amp_i: Optional[float] = None,
    energy_i: Optional[float] = None,
    amp_f: Optional[float] = None,
    energy_f: Optional[float] = None,
) -> np.ndarray:
    tau_list = [int(t) for t in np.asarray(taus, dtype=np.int64).tolist()]
    if (
        amp_i is not None
        and energy_i is not None
        and amp_f is not None
        and energy_f is not None
    ):
        c2f_tsep = _forward_two_point_from_fit([int(t_sep)], amp=float(amp_f), energy=float(energy_f))[0]
        c2i_tsep = _forward_two_point_from_fit([int(t_sep)], amp=float(amp_i), energy=float(energy_i))[0]
        c2i_tsep_tau = _forward_two_point_from_fit(
            [int(t_sep) - t for t in tau_list],
            amp=float(amp_i),
            energy=float(energy_i),
        )
        c2f_tsep_tau = _forward_two_point_from_fit(
            [int(t_sep) - t for t in tau_list],
            amp=float(amp_f),
            energy=float(energy_f),
        )
        c2i_tau = _forward_two_point_from_fit(tau_list, amp=float(amp_i), energy=float(energy_i))
        c2f_tau = _forward_two_point_from_fit(tau_list, amp=float(amp_f), energy=float(energy_f))
    else:
        c2f_tsep = _lookup_times(c2_f, support_times_f, [int(t_sep)])[0]
        c2i_tsep = _lookup_times(c2_i, support_times_i, [int(t_sep)])[0]
        c2i_tsep_tau = _lookup_times(c2_i, support_times_i, [int(t_sep) - t for t in tau_list])
        c2f_tsep_tau = _lookup_times(c2_f, support_times_f, [int(t_sep) - t for t in tau_list])
        c2i_tau = _lookup_times(c2_i, support_times_i, tau_list)
        c2f_tau = _lookup_times(c2_f, support_times_f, tau_list)
    pref = np.asarray(c3, dtype=np.complex128) / np.asarray(c2f_tsep, dtype=np.complex128)
    sq = _sqrt_complex_safe((c2i_tsep_tau * c2f_tau * c2f_tsep) / (c2f_tsep_tau * c2i_tau * c2i_tsep))
    out = pref * sq
    bad = ~np.isfinite(np.real(out)) | ~np.isfinite(np.imag(out))
    out[bad] = np.nan + 1j * np.nan
    return out


def _effective_matrix_element_from_ratio(
    ratio_curve: np.ndarray,
    *,
    energy_i: float,
    energy_f: float,
    zv: float,
) -> np.ndarray:
    scale = float(zv) * 2.0 * math.sqrt(max(0.0, float(energy_i) * float(energy_f)))
    return np.asarray(scale, dtype=np.float64) * np.asarray(ratio_curve, dtype=np.complex128)


def _effective_matrix_element_from_z(
    c3: np.ndarray,
    *,
    taus: np.ndarray,
    t_sep: int,
    energy_i: float,
    energy_f: float,
    z_i: float,
    z_f: float,
    zv: float,
) -> np.ndarray:
    out = np.full_like(np.asarray(c3, dtype=np.complex128), np.nan + 1j * np.nan)
    if not (np.isfinite(energy_i) and np.isfinite(energy_f) and np.isfinite(z_i) and np.isfinite(z_f)):
        return out
    if float(z_i) == 0.0 or float(z_f) == 0.0:
        return out
    pref = float(zv) * 4.0 * float(energy_i) * float(energy_f) / (float(z_i) * float(z_f))
    for it, tau in enumerate(np.asarray(taus, dtype=np.int64).tolist()):
        expo = math.exp(float(energy_f) * float(int(t_sep) - int(tau)) + float(energy_i) * float(int(tau)))
        out[it] = pref * expo * np.asarray(c3[it], dtype=np.complex128)
    return out


def _fit_constant_window_correlated(
    y: np.ndarray,
    cov: np.ndarray,
    *,
    idx_min: int,
    idx_max: int,
    support_times: np.ndarray,
    cov_reg: float,
) -> Optional[Dict[str, object]]:
    i0 = int(idx_min)
    i1 = int(idx_max)
    if i0 < 0 or i1 < i0:
        return None
    yy = np.asarray(y, dtype=np.float64)[i0 : i1 + 1]
    cc = np.asarray(cov, dtype=np.float64)[i0 : i1 + 1, i0 : i1 + 1]
    if yy.size < 2 or cc.shape[0] != yy.size:
        return None
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(cc)):
        return None
    winv, eps = FIT2PT._regularized_inverse(cc, reg=float(cov_reg))
    one = np.ones((yy.size,), dtype=np.float64)
    denom = float(one @ winv @ one)
    if not np.isfinite(denom) or abs(denom) < 1e-16:
        return None
    value = float((one @ winv @ yy) / denom)
    resid = yy - value
    chi2 = float(resid @ winv @ resid)
    dof = int(yy.size - 1)
    if dof <= 0:
        return None
    return {
        "value": float(value),
        "chi2": float(chi2),
        "chi2_dof": float(chi2 / float(dof)),
        "dof": float(dof),
        "tmin": float(np.asarray(support_times, dtype=np.int64)[i0]),
        "tmax": float(np.asarray(support_times, dtype=np.int64)[i1]),
        "npts": float(yy.size),
        "cov_reg_eps": float(eps),
        "support_idx_min": float(i0),
        "support_idx_max": float(i1),
    }


def _select_plateau_window(
    full_curve: np.ndarray,
    jk_curves: np.ndarray,
    *,
    support_times: np.ndarray,
    tmin_min: int,
    tmax_max: int,
    min_points: int,
    chi2_min: float,
    chi2_max: float,
    score_window_penalty: float,
    cov_reg: float,
    progress: bool,
    progress_prefix: str,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    support = np.asarray(support_times, dtype=np.int64).reshape(-1)
    valid = np.where((support >= int(tmin_min)) & (support <= int(tmax_max)))[0]
    if valid.size == 0:
        return None, []
    idx_lo = int(valid[0])
    idx_hi = int(valid[-1])
    cov = _jackknife_covariance(np.asarray(jk_curves, dtype=np.float64))
    candidates: List[Dict[str, object]] = []
    total = 0
    for i0 in range(idx_lo, idx_hi + 1):
        n_here = int(idx_hi - (i0 + int(min_points) - 1) + 1)
        if n_here > 0:
            total += n_here
    done = 0
    if bool(progress):
        print(
            f"[{progress_prefix}] plateau scan: windows={total} min_points={int(min_points)}",
            flush=True,
        )
    for i0 in range(idx_lo, idx_hi + 1):
        for i1 in range(i0 + int(min_points) - 1, idx_hi + 1):
            done += 1
            fit = _fit_constant_window_correlated(
                full_curve,
                cov,
                idx_min=int(i0),
                idx_max=int(i1),
                support_times=support,
                cov_reg=float(cov_reg),
            )
            if fit is None:
                continue
            chi2_dof = float(fit["chi2_dof"])
            score = abs(math.log(max(1e-12, chi2_dof))) + float(score_window_penalty) / float(max(1.0, fit["npts"]))
            fit["score"] = float(score)
            fit["in_chi2_band"] = 1.0 if (chi2_dof >= float(chi2_min) and chi2_dof <= float(chi2_max)) else 0.0
            candidates.append(fit)
            if bool(progress) and done % 200 == 0:
                in_band = sum(1 for c in candidates if int(c.get("in_chi2_band", 0.0)) == 1)
                print(
                    f"[{progress_prefix}] plateau scan: {done}/{total} done candidates={len(candidates)} in_band={in_band}",
                    flush=True,
                )
    if bool(progress):
        in_band = sum(1 for c in candidates if int(c.get("in_chi2_band", 0.0)) == 1)
        print(
            f"[{progress_prefix}] plateau scan: complete candidates={len(candidates)} in_band={in_band}",
            flush=True,
        )
    if not candidates:
        return None, []
    in_band = [c for c in candidates if int(c.get("in_chi2_band", 0.0)) == 1]
    pool = in_band if in_band else candidates
    best = min(pool, key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])))
    return dict(best), candidates


def _plateau_linear_weights(
    jk_curves: np.ndarray,
    *,
    idx_min: int,
    idx_max: int,
    cov_reg: float,
) -> np.ndarray:
    cov = _jackknife_covariance(np.asarray(jk_curves, dtype=np.float64))
    cc = np.asarray(cov, dtype=np.float64)[int(idx_min) : int(idx_max) + 1, int(idx_min) : int(idx_max) + 1]
    winv, _ = FIT2PT._regularized_inverse(cc, reg=float(cov_reg))
    one = np.ones((cc.shape[0],), dtype=np.float64)
    denom = float(one @ winv @ one)
    if not np.isfinite(denom) or abs(denom) < 1e-16:
        return np.full((cc.shape[0],), 1.0 / float(cc.shape[0]), dtype=np.float64)
    return np.asarray((winv @ one) / denom, dtype=np.float64)


def _plateau_from_weights(curve: np.ndarray, jk_curves: np.ndarray, weights: np.ndarray, idx_min: int, idx_max: int) -> Tuple[float, float]:
    full_val = float(np.dot(np.asarray(weights, dtype=np.float64), np.asarray(curve, dtype=np.float64)[int(idx_min) : int(idx_max) + 1]))
    jk_vals = np.asarray(jk_curves, dtype=np.float64)[:, int(idx_min) : int(idx_max) + 1] @ np.asarray(weights, dtype=np.float64)
    err = float(_scalar_jackknife_err(jk_vals, full_val))
    return float(full_val), float(err)


def _temporal_mu_from_payload(payload: Mapping) -> int:
    cfg = payload.get("config", {})
    if isinstance(cfg, Mapping):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, Mapping):
            raw_shape = run_cfg.get("shape", None)
            if isinstance(raw_shape, (list, tuple)) and raw_shape:
                return int(len(tuple(raw_shape)) - 1)
    return 1


def _spatial_extent_from_payload(payload: Mapping) -> int:
    cfg = payload.get("config", {})
    if isinstance(cfg, Mapping):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, Mapping):
            raw_shape = run_cfg.get("shape", None)
            if isinstance(raw_shape, (list, tuple)) and raw_shape:
                return int(raw_shape[0])
    return 1


def _temporal_extent_from_payload(payload: Mapping, support_times: Sequence[int]) -> int:
    cfg = payload.get("config", {})
    if isinstance(cfg, Mapping):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, Mapping):
            raw_shape = run_cfg.get("shape", None)
            if isinstance(raw_shape, (list, tuple)) and raw_shape:
                return int(raw_shape[-1])
    return int(max(1, max(int(v) for v in support_times) + 1))


def _continuum_momentum_1d(p: int, lx: int) -> float:
    return 2.0 * math.pi * float(int(p)) / float(int(max(1, lx)))


def _kinematic_prefactor_1d(
    payload: Mapping,
    *,
    mu: int,
    p_i: int,
    p_f: int,
    energy_i: float,
    energy_f: float,
) -> Tuple[float, str, bool]:
    temporal_mu = _temporal_mu_from_payload(payload)
    if int(mu) == int(temporal_mu):
        return float(energy_f + energy_i), "E_f+E_i", True
    lx = _spatial_extent_from_payload(payload)
    qi = _continuum_momentum_1d(int(p_i), lx)
    qf = _continuum_momentum_1d(int(p_f), lx)
    return float(qf + qi), "p_f+p_i", False


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract pion vector-current matrix elements from checkpoint inline records")
    ap.add_argument("--input", required=True, help="checkpoint .pkl from scripts/mcmc/mcmc.py")
    ap.add_argument("--pion-measurement", default="pion_2pt")
    ap.add_argument("--pion-channel", default="c")
    ap.add_argument("--threept-measurement", default="pion_3pt_vector")
    ap.add_argument("--threept-channel", default="c3")
    ap.add_argument("--mu", type=int, required=True, help="current index")
    ap.add_argument("--pi", type=int, required=True, help="source momentum quantum number")
    ap.add_argument("--pf", type=int, required=True, help="sink momentum quantum number")
    ap.add_argument("--tsep", type=int, required=True, help="source-sink separation")
    ap.add_argument(
        "--component",
        choices=("auto", "re", "im", "abs"),
        default="auto",
        help="component used for plateau fits; auto uses Re for the temporal current and -Im for spatial currents",
    )
    ap.add_argument("--zv", type=float, default=1.0, help="vector-current renormalization factor; use 1 for conserved current")
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iat-method", default="ips", choices=("ips", "sokal", "gamma"))
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0, help="override auto block size; <=0 means auto")
    ap.add_argument("--pion-fit-range", type=str, default="", help="optional manual pion fit range tmin,tmax")
    ap.add_argument("--pion-tmin-min", type=int, default=1)
    ap.add_argument("--pion-tmax-max", type=int, default=-1)
    ap.add_argument("--pion-min-points", type=int, default=4)
    ap.add_argument("--pion-chi2-min", type=float, default=0.5)
    ap.add_argument("--pion-chi2-max", type=float, default=2.0)
    ap.add_argument("--pion-score-window-penalty", type=float, default=0.25)
    ap.add_argument("--pion-fallback-two-exp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pion-two-exp-min-points", type=int, default=6)
    ap.add_argument("--e-min", type=float, default=1.0e-4)
    ap.add_argument("--e-max", type=float, default=4.0)
    ap.add_argument("--e-grid", type=int, default=256)
    ap.add_argument("--plateau-range", type=str, default="", help="optional manual plateau range tmin,tmax in insertion time")
    ap.add_argument("--tau-margin", type=int, default=1, help="exclude tau < margin and tau > tsep-margin when auto-selecting plateaux")
    ap.add_argument("--plateau-min-points", type=int, default=3)
    ap.add_argument("--plateau-chi2-min", type=float, default=0.5)
    ap.add_argument("--plateau-chi2-max", type=float, default=2.0)
    ap.add_argument("--plateau-score-window-penalty", type=float, default=0.20)
    ap.add_argument("--cov-reg", type=float, default=1.0e-10)
    ap.add_argument("--prefix", type=str, default="pion_vector_3pt")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{args.prefix}_mu{args.mu}_pf{args.pf}_pi{args.pi}_tsep{args.tsep}.png"
    json_path = outdir / f"{args.prefix}_mu{args.mu}_pf{args.pf}_pi{args.pi}_tsep{args.tsep}.json"

    payload, records = FIT2PT._load_checkpoint(ckpt_path)
    if not records:
        raise ValueError(f"No inline_records found in checkpoint: {ckpt_path}")

    pion_samples_by_p, pion_steps_by_p, pion_meta_by_p = FIT2PT._extract_by_momentum(
        records,
        measurement=str(args.pion_measurement),
        channel=str(args.pion_channel),
    )
    three_samples, three_steps, three_meta = _extract_threepoint_channel(
        records,
        measurement=str(args.threept_measurement),
        channel=str(args.threept_channel),
        mu=int(args.mu),
        p_i=int(args.pi),
        p_f=int(args.pf),
        t_sep=int(args.tsep),
    )

    pion_aligned, three_aligned, common_steps = _align_common_steps(
        pion_samples_by_p,
        pion_steps_by_p,
        required_momenta=(int(args.pi), int(args.pf)),
        three_samples=three_samples,
        three_steps=three_steps,
    )
    pion_aligned, three_aligned, common_steps = _apply_common_discard_stride(
        pion_aligned,
        three_aligned,
        common_steps,
        discard=int(args.discard),
        stride=int(args.stride),
    )
    support_taus = np.asarray(three_meta["support_taus"], dtype=np.int64)

    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    resolved_component, resolved_sign, resolved_label = _resolve_analysis_component(
        payload,
        mu=int(args.mu),
        component=str(args.component),
    )

    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        block_meta = {
            "pion_tau_ref": float("nan"),
            "pion_block_auto": float("nan"),
            "threept_tau_ref": float("nan"),
            "threept_block_auto": float("nan"),
        }
    else:
        block_size, block_meta = _choose_joint_block_size(
            pion_aligned,
            three_aligned,
            support_taus=support_taus,
            tau_ref=max(1, min(int(args.tsep) // 2, int(args.tsep) - 1)),
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
            component=str(resolved_component),
        )

    if bool(args.progress):
        print(
            f"Loaded {int(common_steps.size)} common samples for 2pt/3pt analysis; block_size={int(block_size)}",
            flush=True,
        )

    t_extent = _temporal_extent_from_payload(payload, pion_meta_by_p[int(args.pi)]["support_times"])
    pion_fit_range = _parse_fit_range_text(args.pion_fit_range) if str(args.pion_fit_range).strip() else None

    pion_results: Dict[int, Dict[str, object]] = {}
    for p in sorted({int(args.pi), int(args.pf)}):
        support_times = np.asarray(pion_meta_by_p[int(p)]["support_times"], dtype=np.int64)
        tmax_default = (int(t_extent) // 2) - 1
        support_half = support_times[support_times <= int(tmax_default)]
        tmax_auto = int(support_half[-1]) if support_half.size > 0 else int(support_times[-1])
        tmax_max = int(args.pion_tmax_max) if int(args.pion_tmax_max) >= 0 else int(tmax_auto)
        if bool(args.progress):
            print(f"[p={int(p)}] fitting pion 2pt", flush=True)
        pion_results[int(p)] = _fit_pion_channel(
            pion_aligned[int(p)],
            support_times=support_times,
            t_extent=int(t_extent),
            block_size=int(block_size),
            fit_range=pion_fit_range,
            tmin_min=int(args.pion_tmin_min),
            tmax_max=int(tmax_max),
            min_points=int(args.pion_min_points),
            chi2_min=float(args.pion_chi2_min),
            chi2_max=float(args.pion_chi2_max),
            score_window_penalty=float(args.pion_score_window_penalty),
            fallback_two_exp=bool(args.pion_fallback_two_exp),
            two_exp_min_points=int(args.pion_two_exp_min_points),
            e_min=float(args.e_min),
            e_max=float(args.e_max),
            e_grid=int(args.e_grid),
            cov_reg=float(args.cov_reg),
            progress=bool(args.progress),
            progress_prefix=f"p={int(p)}",
        )

    nb = int(next(iter(pion_results.values()))["jk_corr"].shape[0])  # type: ignore[index]
    three_blocks = _block_means_nd(three_aligned, block_size=int(block_size))
    three_full = np.asarray(np.mean(three_blocks, axis=0), dtype=np.complex128)
    three_jk = _jackknife_loo_from_blocks(three_blocks)

    p_i_res = pion_results[int(args.pi)]
    p_f_res = pion_results[int(args.pf)]
    ratio_full = _effective_ratio_curve(
        three_full,
        np.asarray(p_i_res["full_corr"], dtype=np.float64),
        np.asarray(p_f_res["full_corr"], dtype=np.float64),
        taus=support_taus,
        t_sep=int(args.tsep),
        support_times_i=np.asarray(p_i_res["support_times"], dtype=np.int64),
        support_times_f=np.asarray(p_f_res["support_times"], dtype=np.int64),
        amp_i=float(p_i_res["amp"]),
        energy_i=float(p_i_res["energy"]),
        amp_f=float(p_f_res["amp"]),
        energy_f=float(p_f_res["energy"]),
    )
    ratio_jk = np.full((nb, support_taus.size), np.nan + 1j * np.nan, dtype=np.complex128)
    mratio_jk = np.full_like(ratio_jk, np.nan + 1j * np.nan)
    mdir_jk = np.full_like(ratio_jk, np.nan + 1j * np.nan)
    for ib in range(nb):
        ratio_jk[ib] = _effective_ratio_curve(
            np.asarray(three_jk[ib], dtype=np.complex128),
            np.asarray(p_i_res["jk_corr"], dtype=np.float64)[ib],  # type: ignore[index]
            np.asarray(p_f_res["jk_corr"], dtype=np.float64)[ib],  # type: ignore[index]
            taus=support_taus,
            t_sep=int(args.tsep),
            support_times_i=np.asarray(p_i_res["support_times"], dtype=np.int64),
            support_times_f=np.asarray(p_f_res["support_times"], dtype=np.int64),
            amp_i=float(np.asarray(p_i_res["jk_amp"], dtype=np.float64)[ib]),  # type: ignore[index]
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
            amp_f=float(np.asarray(p_f_res["jk_amp"], dtype=np.float64)[ib]),  # type: ignore[index]
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
        )
        mratio_jk[ib] = _effective_matrix_element_from_ratio(
            ratio_jk[ib],
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
            zv=float(args.zv),
        )
        mdir_jk[ib] = _effective_matrix_element_from_z(
            np.asarray(three_jk[ib], dtype=np.complex128),
            taus=support_taus,
            t_sep=int(args.tsep),
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),  # type: ignore[index]
            z_i=float(np.asarray(p_i_res["jk_z"], dtype=np.float64)[ib]),  # type: ignore[index]
            z_f=float(np.asarray(p_f_res["jk_z"], dtype=np.float64)[ib]),  # type: ignore[index]
            zv=float(args.zv),
        )
    mratio_full = _effective_matrix_element_from_ratio(
        ratio_full,
        energy_i=float(p_i_res["energy"]),
        energy_f=float(p_f_res["energy"]),
        zv=float(args.zv),
    )
    mdir_full = _effective_matrix_element_from_z(
        three_full,
        taus=support_taus,
        t_sep=int(args.tsep),
        energy_i=float(p_i_res["energy"]),
        energy_f=float(p_f_res["energy"]),
        z_i=float(p_i_res["z_overlap"]),
        z_f=float(p_f_res["z_overlap"]),
        zv=float(args.zv),
    )

    ratio_comp_full = float(resolved_sign) * _component_view(mratio_full, resolved_component)
    ratio_comp_jk = float(resolved_sign) * _component_view(mratio_jk, resolved_component)
    direct_comp_full = float(resolved_sign) * _component_view(mdir_full, resolved_component)
    direct_comp_jk = float(resolved_sign) * _component_view(mdir_jk, resolved_component)

    if str(args.plateau_range).strip():
        plateau_range = _parse_fit_range_text(args.plateau_range)
        idx_min, idx_max = _support_index_range(support_taus, int(plateau_range[0]), int(plateau_range[1]))
        plateau = _fit_constant_window_correlated(
            ratio_comp_full,
            _jackknife_covariance(ratio_comp_jk),
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            support_times=support_taus,
            cov_reg=float(args.cov_reg),
        )
        if plateau is None:
            raise ValueError(f"Manual plateau range [{plateau_range[0]},{plateau_range[1]}] failed")
        plateau_candidates: List[Dict[str, object]] = []
    else:
        tmin_min = max(int(args.tau_margin), int(support_taus[0]))
        tmax_max = min(int(args.tsep) - int(args.tau_margin), int(support_taus[-1]))
        plateau, plateau_candidates = _select_plateau_window(
            ratio_comp_full,
            ratio_comp_jk,
            support_times=support_taus,
            tmin_min=int(tmin_min),
            tmax_max=int(tmax_max),
            min_points=int(args.plateau_min_points),
            chi2_min=float(args.plateau_chi2_min),
            chi2_max=float(args.plateau_chi2_max),
            score_window_penalty=float(args.plateau_score_window_penalty),
            cov_reg=float(args.cov_reg),
            progress=bool(args.progress),
            progress_prefix=f"mu={int(args.mu)} pf={int(args.pf)} pi={int(args.pi)} tsep={int(args.tsep)}",
        )
        if plateau is None:
            raise ValueError("No successful plateau candidate was found")

    weights = _plateau_linear_weights(
        ratio_comp_jk,
        idx_min=int(plateau["support_idx_min"]),
        idx_max=int(plateau["support_idx_max"]),
        cov_reg=float(args.cov_reg),
    )
    matrix_ratio, matrix_ratio_err = _plateau_from_weights(
        ratio_comp_full,
        ratio_comp_jk,
        weights,
        int(plateau["support_idx_min"]),
        int(plateau["support_idx_max"]),
    )
    matrix_direct, matrix_direct_err = _plateau_from_weights(
        direct_comp_full,
        direct_comp_jk,
        weights,
        int(plateau["support_idx_min"]),
        int(plateau["support_idx_max"]),
    )

    kin_full, kin_label, is_temporal = _kinematic_prefactor_1d(
        payload,
        mu=int(args.mu),
        p_i=int(args.pi),
        p_f=int(args.pf),
        energy_i=float(p_i_res["energy"]),
        energy_f=float(p_f_res["energy"]),
    )
    form_ratio = float("nan")
    form_ratio_err = float("nan")
    form_direct = float("nan")
    form_direct_err = float("nan")
    form_curve_full = np.full_like(ratio_comp_full, np.nan, dtype=np.float64)
    form_curve_jk = np.full_like(ratio_comp_jk, np.nan, dtype=np.float64)
    form_direct_jk = np.full_like(direct_comp_jk, np.nan, dtype=np.float64)
    if abs(float(kin_full)) > 1.0e-14:
        form_curve_full = ratio_comp_full / float(kin_full)
        for ib in range(nb):
            if is_temporal:
                kin_jk = float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]) + float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib])  # type: ignore[index]
            else:
                kin_jk = float(kin_full)
            if abs(float(kin_jk)) > 1.0e-14:
                form_curve_jk[ib] = ratio_comp_jk[ib] / kin_jk
                form_direct_jk[ib] = direct_comp_jk[ib] / kin_jk
        form_ratio, form_ratio_err = _plateau_from_weights(
            form_curve_full,
            form_curve_jk,
            weights,
            int(plateau["support_idx_min"]),
            int(plateau["support_idx_max"]),
        )
        form_direct, form_direct_err = _plateau_from_weights(
            direct_comp_full / float(kin_full),
            form_direct_jk,
            weights,
            int(plateau["support_idx_min"]),
            int(plateau["support_idx_max"]),
        )

    lx = _spatial_extent_from_payload(payload)
    qi = _continuum_momentum_1d(int(args.pi), lx)
    qf = _continuum_momentum_1d(int(args.pf), lx)
    q2 = (float(p_f_res["energy"]) - float(p_i_res["energy"])) ** 2 + (qf - qi) ** 2

    result = {
        "input": str(ckpt_path),
        "measurement_names": {
            "pion": str(args.pion_measurement),
            "threept": str(args.threept_measurement),
        },
        "kinematics": {
            "mu": int(args.mu),
            "temporal_mu": int(_temporal_mu_from_payload(payload)),
            "is_temporal": bool(is_temporal),
            "pi": int(args.pi),
            "pf": int(args.pf),
            "tsep": int(args.tsep),
            "spatial_extent": int(lx),
            "q2_cont_like": float(q2),
            "kinematic_prefactor": float(kin_full),
            "kinematic_prefactor_label": str(kin_label),
            "form_factor_available": bool(abs(float(kin_full)) > 1.0e-14),
        },
        "statistics": {
            "n_common_samples": int(common_steps.size),
            "block_size": int(block_size),
            "iat_method": str(args.iat_method),
            "discard": int(args.discard),
            "stride": int(args.stride),
            **block_meta,
        },
        "pion_fits": {
            f"p{int(p)}": {
                k: v
                for k, v in pion_results[int(p)].items()
                if k
                not in (
                    "blocks",
                    "full_corr",
                    "jk_corr",
                    "jk_energy",
                    "jk_amp",
                    "jk_z",
                    "support_times",
                )
            }
            for p in sorted(pion_results.keys())
        },
        "plateau": {
            "selected_on": "ratio",
            "component_requested": str(args.component),
            "component_resolved": str(resolved_component),
            "component_sign": float(resolved_sign),
            "component_label": str(resolved_label),
            "tmin": float(plateau["tmin"]),
            "tmax": float(plateau["tmax"]),
            "chi2_dof": float(plateau["chi2_dof"]),
            "npts": float(plateau["npts"]),
        },
        "matrix_element": {
            "zv": float(args.zv),
            "ratio_value": float(matrix_ratio),
            "ratio_err": float(matrix_ratio_err),
            "direct_value": float(matrix_direct),
            "direct_err": float(matrix_direct_err),
        },
        "form_factor": {
            "available": bool(abs(float(kin_full)) > 1.0e-14),
            "kinematic_prefactor": float(kin_full),
            "kinematic_prefactor_label": str(kin_label),
            "ratio_value": float(form_ratio),
            "ratio_err": float(form_ratio_err),
            "direct_value": float(form_direct),
            "direct_err": float(form_direct_err),
        },
        "effective_curves": {
            "taus": support_taus.astype(int).tolist(),
            "matrix_ratio_mean": np.asarray(ratio_comp_full, dtype=np.float64).tolist(),
            "matrix_ratio_err": np.sqrt(np.clip(np.diag(_jackknife_covariance(ratio_comp_jk)), 0.0, np.inf)).tolist(),
            "matrix_direct_mean": np.asarray(direct_comp_full, dtype=np.float64).tolist(),
            "matrix_direct_err": np.sqrt(np.clip(np.diag(_jackknife_covariance(direct_comp_jk)), 0.0, np.inf)).tolist(),
            "form_ratio_mean": np.asarray(form_curve_full, dtype=np.float64).tolist(),
            "form_ratio_err": np.sqrt(np.clip(np.diag(_jackknife_covariance(form_curve_jk)), 0.0, np.inf)).tolist()
            if abs(float(kin_full)) > 1.0e-14
            else [],
        },
    }
    if plateau_candidates:
        result["plateau_candidates"] = [
            {
                "tmin": float(c["tmin"]),
                "tmax": float(c["tmax"]),
                "chi2_dof": float(c["chi2_dof"]),
                "score": float(c["score"]),
                "in_chi2_band": float(c["in_chi2_band"]),
            }
            for c in plateau_candidates
        ]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    plt = _setup_matplotlib(bool(args.no_gui))
    if plt is not None:
        form_available = abs(float(kin_full)) > 1.0e-14
        nrow = 2 if form_available else 1
        fig, axes = plt.subplots(nrow, 1, figsize=(9.0, 3.6 * nrow), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        x = np.asarray(support_taus, dtype=np.int64)
        ax0 = axes[0]
        ax0.errorbar(
            x - 0.06,
            np.asarray(ratio_comp_full, dtype=np.float64),
            yerr=np.sqrt(np.clip(np.diag(_jackknife_covariance(ratio_comp_jk)), 0.0, np.inf)),
            fmt="o",
            ms=4.0,
            capsize=2.0,
            label="ratio",
        )
        ax0.errorbar(
            x + 0.06,
            np.asarray(direct_comp_full, dtype=np.float64),
            yerr=np.sqrt(np.clip(np.diag(_jackknife_covariance(direct_comp_jk)), 0.0, np.inf)),
            fmt="s",
            ms=4.0,
            capsize=2.0,
            label="direct-Z",
        )
        ax0.axvspan(float(plateau["tmin"]) - 0.5, float(plateau["tmax"]) + 0.5, color="0.92", zorder=-10)
        ax0.axhspan(
            float(matrix_ratio) - float(matrix_ratio_err),
            float(matrix_ratio) + float(matrix_ratio_err),
            color="tab:blue",
            alpha=0.15,
        )
        ax0.axhline(float(matrix_ratio), color="tab:blue", lw=1.2, label="ratio plateau")
        ax0.axhspan(
            float(matrix_direct) - float(matrix_direct_err),
            float(matrix_direct) + float(matrix_direct_err),
            color="tab:orange",
            alpha=0.15,
        )
        ax0.axhline(float(matrix_direct), color="tab:orange", lw=1.2, ls="--", label="direct-Z plateau")
        ax0.set_ylabel("matrix element")
        ax0.set_title(
            f"mu={int(args.mu)} pf={int(args.pf)} pi={int(args.pi)} tsep={int(args.tsep)} "
            f"ZV={float(args.zv):.6g}"
        )
        ax0.legend(loc="best", fontsize=9)
        if form_available:
            ax1 = axes[1]
            ax1.errorbar(
                x,
                np.asarray(form_curve_full, dtype=np.float64),
                yerr=np.sqrt(np.clip(np.diag(_jackknife_covariance(form_curve_jk)), 0.0, np.inf)),
                fmt="o",
                ms=4.0,
                capsize=2.0,
                label=f"ratio / ({kin_label})",
            )
            ax1.axvspan(float(plateau["tmin"]) - 0.5, float(plateau["tmax"]) + 0.5, color="0.92", zorder=-10)
            ax1.axhspan(
                float(form_ratio) - float(form_ratio_err),
                float(form_ratio) + float(form_ratio_err),
                color="tab:green",
                alpha=0.15,
            )
            ax1.axhline(float(form_ratio), color="tab:green", lw=1.2, label="form-factor plateau")
            ax1.set_ylabel("F")
            ax1.set_title(f"Form factor via {kin_label}")
            ax1.legend(loc="best", fontsize=9)
        axes[-1].set_xlabel("tau")
        fig.tight_layout()
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        if not bool(args.no_gui):
            plt.show()
        plt.close(fig)

    print(f"Wrote {json_path}", flush=True)
    if plt is not None:
        print(f"Wrote {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

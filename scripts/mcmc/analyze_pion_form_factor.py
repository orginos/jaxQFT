#!/usr/bin/env python3
"""Batch analysis for dense Schwinger-model pion form-factor data.

This script scans a checkpoint for `pion_2pt` and `pion_3pt_vector` inline
records, extracts all requested kinematic channels, runs the single-channel
matrix-element analysis in-process, and summarizes the results across source-
sink separations and momentum transfers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{name}' from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PV3 = _load_script_module("analyze_pion_vector_3pt_mod", ROOT / "scripts" / "mcmc" / "analyze_pion_vector_3pt.py")


def _discover_threepoint_channels(
    records: Sequence[Mapping],
    *,
    measurement: str,
    channel: str,
) -> List[Tuple[int, int, int, int]]:
    pat = re.compile(
        rf"^{re.escape(str(channel))}_mu(?P<mu>\d+)_pf(?P<pf>-?\d+)_pi(?P<pi>-?\d+)_tsep(?P<tsep>\d+)_tau(?P<tau>\d+)_(?:re|im)$"
    )
    found = set()
    for rec in records:
        if str(rec.get("name", "")) != str(measurement):
            continue
        vals = rec.get("values", {})
        if not isinstance(vals, Mapping):
            continue
        for key in vals.keys():
            m = pat.match(str(key))
            if m is None:
                continue
            found.add(
                (
                    int(m.group("mu")),
                    int(m.group("pf")),
                    int(m.group("pi")),
                    int(m.group("tsep")),
                )
            )
    return sorted(found)


def _parse_fit_range_text(text: str) -> Optional[Tuple[int, int]]:
    txt = str(text).strip()
    if not txt:
        return None
    toks = [t.strip() for t in txt.split(",") if t.strip()]
    if len(toks) != 2:
        raise ValueError(f"fit range must have exactly 2 integers, got {text!r}")
    a = int(toks[0])
    b = int(toks[1])
    if b < a:
        a, b = b, a
    return int(a), int(b)


def _filter_channels(
    chans: Sequence[Tuple[int, int, int, int]],
    *,
    mu: Optional[int],
    tsep_min: int,
    tsep_max: int,
    pair_mode: str,
    max_abs_p: int,
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    mode = str(pair_mode).strip().lower()
    for cur_mu, pf, pi, tsep in chans:
        if mu is not None and int(cur_mu) != int(mu):
            continue
        if int(tsep) < int(tsep_min) or int(tsep) > int(tsep_max):
            continue
        if max(abs(int(pf)), abs(int(pi))) > int(max_abs_p):
            continue
        if mode in ("breit", "breit_frame", "symmetric"):
            if int(pf) != -int(pi):
                continue
            if int(pf) < 0:
                continue
        elif mode in ("equal", "diagonal", "same", "q2zero"):
            if int(pf) != int(pi):
                continue
            if int(pf) < 0:
                continue
        elif mode not in ("all", "cartesian", "full"):
            raise ValueError(f"Unsupported pair-mode filter: {pair_mode!r}")
        out.append((int(cur_mu), int(pf), int(pi), int(tsep)))
    return sorted(out, key=lambda row: (row[0], row[1] - row[2], row[1], row[2], row[3]))


def _safe_jk_covariance(jk: np.ndarray) -> np.ndarray:
    arr = np.asarray(jk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.full((arr.shape[-1], arr.shape[-1]), np.nan, dtype=np.float64)
    return PV3._jackknife_covariance(arr)


def _covariance_weighted_average(full_vals: np.ndarray, jk_vals: np.ndarray, cov_reg: float) -> Tuple[float, float]:
    y = np.asarray(full_vals, dtype=np.float64).reshape(-1)
    yjk = np.asarray(jk_vals, dtype=np.float64)
    if y.size == 1:
        return float(y[0]), float(PV3._scalar_jackknife_err(yjk[:, 0], float(y[0])))
    cov = _safe_jk_covariance(yjk)
    diag_mean = float(np.nanmean(np.diag(cov))) if np.all(np.isfinite(np.diag(cov))) else 1.0
    cc = np.asarray(cov + float(cov_reg) * max(1.0, diag_mean) * np.eye(y.size), dtype=np.float64)
    try:
        winv = np.linalg.pinv(cc, rcond=1.0e-12)
    except np.linalg.LinAlgError:
        winv = np.eye(y.size, dtype=np.float64)
    one = np.ones((y.size,), dtype=np.float64)
    denom = float(one @ winv @ one)
    if not np.isfinite(denom) or abs(denom) < 1.0e-14:
        w = np.full((y.size,), 1.0 / float(y.size), dtype=np.float64)
    else:
        w = np.asarray((winv @ one) / denom, dtype=np.float64)
    full = float(w @ y)
    jk = np.asarray(yjk @ w, dtype=np.float64)
    err = float(PV3._scalar_jackknife_err(jk, full))
    return full, err


def _covariance_weighted_average_with_jk(
    full_vals: np.ndarray, jk_vals: np.ndarray, cov_reg: float
) -> Tuple[float, float, np.ndarray]:
    y = np.asarray(full_vals, dtype=np.float64).reshape(-1)
    yjk = np.asarray(jk_vals, dtype=np.float64)
    if y.size == 1:
        jk = np.asarray(yjk[:, 0], dtype=np.float64)
        return float(y[0]), float(PV3._scalar_jackknife_err(jk, float(y[0]))), jk
    cov = _safe_jk_covariance(yjk)
    diag_mean = float(np.nanmean(np.diag(cov))) if np.all(np.isfinite(np.diag(cov))) else 1.0
    cc = np.asarray(cov + float(cov_reg) * max(1.0, diag_mean) * np.eye(y.size), dtype=np.float64)
    try:
        winv = np.linalg.pinv(cc, rcond=1.0e-12)
    except np.linalg.LinAlgError:
        winv = np.eye(y.size, dtype=np.float64)
    one = np.ones((y.size,), dtype=np.float64)
    denom = float(one @ winv @ one)
    if not np.isfinite(denom) or abs(denom) < 1.0e-14:
        w = np.full((y.size,), 1.0 / float(y.size), dtype=np.float64)
    else:
        w = np.asarray((winv @ one) / denom, dtype=np.float64)
    full = float(w @ y)
    jk = np.asarray(yjk @ w, dtype=np.float64)
    err = float(PV3._scalar_jackknife_err(jk, full))
    return full, err, jk


def _extract_required_momenta(chans: Sequence[Tuple[int, int, int, int]]) -> List[int]:
    moms = sorted({int(pf) for _, pf, _, _ in chans} | {int(pi) for _, _, pi, _ in chans})
    return [int(p) for p in moms]


def _align_samples_to_steps(samples: np.ndarray, steps: Sequence[int], target_steps: Sequence[int]) -> np.ndarray:
    arr = np.asarray(samples)
    st = np.asarray(steps, dtype=np.int64)
    tgt = np.asarray(target_steps, dtype=np.int64)
    idx = {int(s): i for i, s in enumerate(st.tolist())}
    missing = [int(s) for s in tgt.tolist() if int(s) not in idx]
    if missing:
        raise ValueError(f"Target common-step set is not contained in available steps; missing={missing[:8]}")
    return np.asarray([arr[idx[int(s)]] for s in tgt], dtype=arr.dtype)


def _stack_group_jk(rows: Sequence[Mapping[str, object]], key: str, *, context: str) -> np.ndarray:
    seqs = [np.asarray(r[key], dtype=np.float64) for r in rows]
    lens = sorted({int(v.shape[0]) for v in seqs})
    if len(lens) != 1:
        desc = ", ".join(
            f"mu={int(r['mu'])},pf={int(r['pf'])},pi={int(r['pi'])},tsep={int(r['tsep'])},nb={int(np.asarray(r[key]).shape[0])}"
            for r in rows
        )
        raise ValueError(
            f"Inconsistent jackknife replica counts while grouping {context}: lengths={lens}; channels=[{desc}]. "
            "The batch analyzer should enforce a common step set and common block size; rerun with the patched analyzer."
        )
    return np.asarray(seqs, dtype=np.float64).T


def _prepare_batch_common_steps_and_block_size(
    *,
    payload: Mapping,
    records: Sequence[Mapping],
    selected: Sequence[Tuple[int, int, int, int]],
    pion_samples_by_p: Mapping[int, np.ndarray],
    pion_steps_by_p: Mapping[int, np.ndarray],
    pion_meta_by_p: Mapping[int, Mapping[str, object]],
    args,
) -> Tuple[np.ndarray, int, Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, Dict[str, object]]]]:
    required_momenta = _extract_required_momenta(selected)
    common = None
    for p in required_momenta:
        st = set(int(s) for s in np.asarray(pion_steps_by_p[int(p)], dtype=np.int64).tolist())
        common = st if common is None else (common & st)
    if common is None:
        raise ValueError("No required pion momenta were found while preparing batch common steps")

    three_cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, Dict[str, object]]] = {}
    for cur_mu, pf, pi, tsep in selected:
        three_samples, three_steps, three_meta = PV3._extract_threepoint_channel(
            records,
            measurement=str(args.threept_measurement),
            channel=str(args.threept_channel),
            mu=int(cur_mu),
            p_i=int(pi),
            p_f=int(pf),
            t_sep=int(tsep),
        )
        three_cache[(int(cur_mu), int(pf), int(pi), int(tsep))] = (three_samples, three_steps, three_meta)
        common &= set(int(s) for s in np.asarray(three_steps, dtype=np.int64).tolist())

    if not common:
        raise ValueError("No common MCMC steps remain across the selected pion and 3pt channels")

    steps = np.asarray(sorted(common), dtype=np.int64)
    keep = np.arange(int(steps.size), dtype=np.int64)[max(0, int(args.discard)) :: max(1, int(args.stride))]
    steps = np.asarray(steps[keep], dtype=np.int64)
    if steps.size < 4:
        raise ValueError(f"Need at least 4 common samples after batch discard/stride; got {int(steps.size)}")

    if int(args.block_size) > 0:
        return steps, int(args.block_size), three_cache

    aligned_pions = {
        int(p): _align_samples_to_steps(
            np.asarray(pion_samples_by_p[int(p)], dtype=np.float64),
            np.asarray(pion_steps_by_p[int(p)], dtype=np.int64),
            steps,
        )
        for p in required_momenta
    }
    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    global_block_size = 1
    for cur_mu, pf, pi, tsep in selected:
        three_samples, three_steps, three_meta = three_cache[(int(cur_mu), int(pf), int(pi), int(tsep))]
        three_aligned = _align_samples_to_steps(three_samples, three_steps, steps)
        support_taus = np.asarray(three_meta["support_taus"], dtype=np.int64)
        resolved_component, _, _ = PV3._resolve_analysis_component(
            payload,
            mu=int(cur_mu),
            component=str(args.component),
        )
        block_size, _ = PV3._choose_joint_block_size(
            {int(p): aligned_pions[int(p)] for p in sorted({int(pi), int(pf)})},
            three_aligned,
            support_taus=support_taus,
            tau_ref=max(1, min(int(tsep) // 2, int(tsep) - 1)),
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
            component=str(resolved_component),
        )
        global_block_size = max(int(global_block_size), int(block_size))
    return steps, int(global_block_size), three_cache


def _group_results_by_q2_tsep(
    results: Sequence[Mapping[str, object]],
    *,
    cov_reg: float,
    q2_tol: float = 1.0e-12,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, int], List[Mapping[str, object]]] = {}
    for row in results:
        if not bool(row.get("form_available", False)):
            continue
        q2 = float(row["q2_cont_like"])
        tsep = int(row["tsep"])
        q2_bin = int(round(q2 / float(q2_tol))) if float(q2_tol) > 0.0 else int(round(q2 * 1.0e12))
        grouped.setdefault((q2_bin, tsep), []).append(row)

    out: List[Dict[str, object]] = []
    for (_, tsep), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        q2 = float(rows[0]["q2_cont_like"])
        rr_full = np.asarray([float(r["form_ratio"]) for r in rows], dtype=np.float64)
        rr_jk = _stack_group_jk(rows, "_form_ratio_jk", context=f"Q^2={q2:.8g}, tsep={int(tsep)} ratio")
        rd_full = np.asarray([float(r["form_direct"]) for r in rows], dtype=np.float64)
        rd_jk = _stack_group_jk(rows, "_form_direct_jk", context=f"Q^2={q2:.8g}, tsep={int(tsep)} direct")
        avg_ratio, avg_ratio_err, avg_ratio_jk = _covariance_weighted_average_with_jk(
            rr_full, rr_jk, cov_reg=float(cov_reg)
        )
        avg_direct, avg_direct_err, avg_direct_jk = _covariance_weighted_average_with_jk(
            rd_full, rd_jk, cov_reg=float(cov_reg)
        )
        out.append(
            {
                "q2_cont_like": float(q2),
                "tsep": int(tsep),
                "n_channels": int(len(rows)),
                "channels": [
                    {
                        "mu": int(r["mu"]),
                        "pf": int(r["pf"]),
                        "pi": int(r["pi"]),
                        "kinematic_prefactor_label": str(r["kinematic_prefactor_label"]),
                    }
                    for r in rows
                ],
                "form_ratio": float(avg_ratio),
                "form_ratio_err": float(avg_ratio_err),
                "form_direct": float(avg_direct),
                "form_direct_err": float(avg_direct_err),
                "_form_ratio_jk": np.asarray(avg_ratio_jk, dtype=np.float64).tolist(),
                "_form_direct_jk": np.asarray(avg_direct_jk, dtype=np.float64).tolist(),
            }
        )
    return out


def _group_results_by_q2(
    grouped_q2_tsep: Sequence[Mapping[str, object]],
    *,
    cov_reg: float,
    q2_tol: float,
    tsep_max_safe: Optional[int],
    tsep_plateau_range: Optional[Tuple[int, int]],
    tsep_plateau_min_points: int,
    tsep_plateau_chi2_min: float,
    tsep_plateau_chi2_max: float,
    tsep_plateau_score_window_penalty: float,
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in grouped_q2_tsep:
        q2 = float(row["q2_cont_like"])
        q2_bin = int(round(q2 / float(q2_tol))) if float(q2_tol) > 0.0 else int(round(q2 * 1.0e12))
        grouped.setdefault(q2_bin, []).append(row)

    out: List[Dict[str, object]] = []
    for _, rows_unsorted in sorted(grouped.items(), key=lambda item: float(item[1][0]["q2_cont_like"])):
        rows = sorted(rows_unsorted, key=lambda r: int(r["tsep"]))
        rows_fit = [
            r for r in rows
            if (tsep_max_safe is None or int(r["tsep"]) <= int(tsep_max_safe))
        ]
        q2 = float(rows[0]["q2_cont_like"])
        tseps_all = np.asarray([int(r["tsep"]) for r in rows], dtype=np.int64)
        tseps = np.asarray([int(r["tsep"]) for r in rows_fit], dtype=np.int64)
        if tseps.size == 0:
            continue
        ratio_full = np.asarray([float(r["form_ratio"]) for r in rows_fit], dtype=np.float64)
        direct_full = np.asarray([float(r["form_direct"]) for r in rows_fit], dtype=np.float64)
        ratio_jk = _stack_group_jk(rows_fit, "_form_ratio_jk", context=f"late-tsep Q^2={q2:.8g} ratio")
        direct_jk = _stack_group_jk(rows_fit, "_form_direct_jk", context=f"late-tsep Q^2={q2:.8g} direct")

        plateau_candidates: List[Dict[str, object]] = []
        if tseps.size == 1:
            plateau = {
                "tmin": float(tseps[0]),
                "tmax": float(tseps[0]),
                "chi2_dof": float("nan"),
                "npts": 1.0,
                "support_idx_min": 0.0,
                "support_idx_max": 0.0,
            }
            weights = np.asarray([1.0], dtype=np.float64)
            form_ratio = float(ratio_full[0])
            form_ratio_err = float(rows[0]["form_ratio_err"])
            form_direct = float(direct_full[0])
            form_direct_err = float(rows[0]["form_direct_err"])
            ratio_jk_plateau = np.asarray(ratio_jk[:, 0], dtype=np.float64)
            direct_jk_plateau = np.asarray(direct_jk[:, 0], dtype=np.float64)
        else:
            if tsep_plateau_range is not None:
                idx_min, idx_max = PV3._support_index_range(
                    tseps, int(tsep_plateau_range[0]), int(tsep_plateau_range[1])
                )
                plateau = PV3._fit_constant_window_correlated(
                    ratio_full,
                    PV3._jackknife_covariance(ratio_jk),
                    idx_min=int(idx_min),
                    idx_max=int(idx_max),
                    support_times=tseps,
                    cov_reg=float(cov_reg),
                )
                if plateau is None:
                    raise ValueError(
                        f"Manual tsep plateau range [{tsep_plateau_range[0]},{tsep_plateau_range[1]}] failed for Q^2={q2:.8g}"
                    )
            else:
                plateau, plateau_candidates = PV3._select_plateau_window(
                    ratio_full,
                    ratio_jk,
                    support_times=tseps,
                    tmin_min=int(tseps[0]),
                    tmax_max=int(tseps[-1]),
                    min_points=int(min(max(1, int(tsep_plateau_min_points)), tseps.size)),
                    chi2_min=float(tsep_plateau_chi2_min),
                    chi2_max=float(tsep_plateau_chi2_max),
                    score_window_penalty=float(tsep_plateau_score_window_penalty),
                    cov_reg=float(cov_reg),
                    progress=False,
                    progress_prefix=f"Q2={q2:.6g}",
                )
                if plateau is None:
                    raise ValueError(f"No successful tsep plateau candidate found for Q^2={q2:.8g}")
            weights = PV3._plateau_linear_weights(
                ratio_jk,
                idx_min=int(plateau["support_idx_min"]),
                idx_max=int(plateau["support_idx_max"]),
                cov_reg=float(cov_reg),
            )
            form_ratio, form_ratio_err = PV3._plateau_from_weights(
                ratio_full,
                ratio_jk,
                weights,
                int(plateau["support_idx_min"]),
                int(plateau["support_idx_max"]),
            )
            form_direct, form_direct_err = PV3._plateau_from_weights(
                direct_full,
                direct_jk,
                weights,
                int(plateau["support_idx_min"]),
                int(plateau["support_idx_max"]),
            )
            i0 = int(plateau["support_idx_min"])
            i1 = int(plateau["support_idx_max"]) + 1
            ratio_jk_plateau = np.asarray(ratio_jk[:, i0:i1] @ weights, dtype=np.float64)
            direct_jk_plateau = np.asarray(direct_jk[:, i0:i1] @ weights, dtype=np.float64)

        out_row = {
            "q2_cont_like": float(q2),
            "interpretation": "late_tsep_constant_diagnostic",
            "note": (
                "This reduction assumes an approximately constant large-tsep window after channel-wise "
                "kinematic-factor removal. It is a diagnostic summary, not a substitute for a full excited-state fit."
            ),
            "tseps_available": tseps_all.astype(int).tolist(),
            "tseps_used": tseps.astype(int).tolist(),
            "excluded_tseps_thermal": [
                int(r["tsep"]) for r in rows if (tsep_max_safe is not None and int(r["tsep"]) > int(tsep_max_safe))
            ],
            "n_tsep": int(tseps.size),
            "n_total_channels": int(sum(int(r["n_channels"]) for r in rows)),
            "tseps": tseps.astype(int).tolist(),
            "plateau": {
                "selected_on": "form_ratio",
                "tsep_min": float(plateau["tmin"]),
                "tsep_max": float(plateau["tmax"]),
                "chi2_dof": float(plateau["chi2_dof"]),
                "npts": float(plateau["npts"]),
            },
            "form_ratio": float(form_ratio),
            "form_ratio_err": float(form_ratio_err),
            "form_direct": float(form_direct),
            "form_direct_err": float(form_direct_err),
            "per_tsep": [
                {
                    "tsep": int(r["tsep"]),
                    "n_channels": int(r["n_channels"]),
                    "form_ratio": float(r["form_ratio"]),
                    "form_ratio_err": float(r["form_ratio_err"]),
                    "form_direct": float(r["form_direct"]),
                    "form_direct_err": float(r["form_direct_err"]),
                }
                for r in rows
            ],
            "_form_ratio_jk": np.asarray(ratio_jk_plateau, dtype=np.float64).tolist(),
            "_form_direct_jk": np.asarray(direct_jk_plateau, dtype=np.float64).tolist(),
        }
        if plateau_candidates:
            out_row["plateau_candidates"] = [
                {
                    "tmin": float(c["tmin"]),
                    "tmax": float(c["tmax"]),
                    "chi2_dof": float(c["chi2_dof"]),
                    "score": float(c["score"]),
                    "in_chi2_band": float(c["in_chi2_band"]),
                }
                for c in plateau_candidates
            ]
        out.append(out_row)
    return out


def _fit_tsep_one_exp_window_correlated(
    y: np.ndarray,
    cov: np.ndarray,
    *,
    idx_min: int,
    idx_max: int,
    support_times: np.ndarray,
    gap_min: float,
    gap_max: float,
    gap_grid: int,
    cov_reg: float,
) -> Optional[Dict[str, object]]:
    i0 = int(idx_min)
    i1 = int(idx_max)
    if i0 < 0 or i1 < i0:
        return None
    yy = np.asarray(y, dtype=np.float64)[i0 : i1 + 1]
    cc = np.asarray(cov, dtype=np.float64)[i0 : i1 + 1, i0 : i1 + 1]
    tt = np.asarray(support_times, dtype=np.float64)[i0 : i1 + 1]
    if yy.size < 4 or cc.shape[0] != yy.size:
        return None
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(cc)) or not np.all(np.isfinite(tt)):
        return None
    gmin = max(1.0e-6, float(gap_min))
    gmax = max(gmin + 1.0e-6, float(gap_max))
    ngrid = max(32, int(gap_grid))
    winv, eps = PV3.FIT2PT._regularized_inverse(cc, reg=float(cov_reg))
    def _solve_at_gap(gap_val: float) -> Optional[Dict[str, object]]:
        col = np.exp(-float(gap_val) * tt)
        X = np.column_stack([np.ones_like(tt), col])
        xtwx = np.asarray(X.T @ winv @ X, dtype=np.float64)
        try:
            xtwx_inv = np.linalg.pinv(xtwx, rcond=1.0e-12)
        except np.linalg.LinAlgError:
            return None
        beta = np.asarray(xtwx_inv @ (X.T @ winv @ yy), dtype=np.float64)
        resid = np.asarray(yy - X @ beta, dtype=np.float64)
        chi2 = float(resid @ winv @ resid)
        dof = int(yy.size - 3)
        if dof <= 0 or not np.isfinite(chi2):
            return None
        return {
            "value": float(beta[0]),
            "amplitude": float(beta[1]),
            "gap": float(gap_val),
            "chi2": float(chi2),
            "chi2_dof": float(chi2 / float(dof)),
            "dof": float(dof),
            "tmin": float(tt[0]),
            "tmax": float(tt[-1]),
            "npts": float(yy.size),
            "cov_reg_eps": float(eps),
            "support_idx_min": float(i0),
            "support_idx_max": float(i1),
            "fit_curve": np.asarray(X @ beta, dtype=np.float64).tolist(),
        }

    best: Optional[Dict[str, object]] = None
    best_idx = -1
    gaps = np.linspace(gmin, gmax, ngrid, dtype=np.float64)
    coarse_rows: List[Optional[Dict[str, object]]] = []
    for ig, gap in enumerate(gaps):
        row = _solve_at_gap(float(gap))
        coarse_rows.append(row)
        if row is None:
            continue
        if best is None or float(row["chi2"]) < float(best["chi2"]):
            best = row
            best_idx = int(ig)
    if best is None:
        return None
    if 0 < best_idx < gaps.size - 1:
        y1 = coarse_rows[best_idx - 1]["chi2"] if coarse_rows[best_idx - 1] is not None else float("nan")
        y2 = coarse_rows[best_idx]["chi2"] if coarse_rows[best_idx] is not None else float("nan")
        y3 = coarse_rows[best_idx + 1]["chi2"] if coarse_rows[best_idx + 1] is not None else float("nan")
        h = float(gaps[best_idx + 1] - gaps[best_idx])
        denom = float(y1 - 2.0 * y2 + y3) if np.isfinite(y1) and np.isfinite(y2) and np.isfinite(y3) else float("nan")
        if np.isfinite(denom) and abs(denom) > 1.0e-14:
            delta = 0.5 * h * float(y1 - y3) / denom
            gap_star = float(gaps[best_idx] + delta)
            if float(gaps[best_idx - 1]) <= gap_star <= float(gaps[best_idx + 1]):
                refined = _solve_at_gap(gap_star)
                if refined is not None and float(refined["chi2"]) <= float(best["chi2"]):
                    best = refined
    return dict(best)


def _select_tsep_one_exp_window(
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
    gap_min: float,
    gap_max: float,
    gap_grid: int,
    cov_reg: float,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    support = np.asarray(support_times, dtype=np.int64).reshape(-1)
    valid = np.where((support >= int(tmin_min)) & (support <= int(tmax_max)))[0]
    if valid.size == 0:
        return None, []
    idx_lo = int(valid[0])
    idx_hi = int(valid[-1])
    cov = PV3._jackknife_covariance(np.asarray(jk_curves, dtype=np.float64))
    candidates: List[Dict[str, object]] = []
    for i0 in range(idx_lo, idx_hi + 1):
        for i1 in range(i0 + int(min_points) - 1, idx_hi + 1):
            fit = _fit_tsep_one_exp_window_correlated(
                full_curve,
                cov,
                idx_min=int(i0),
                idx_max=int(i1),
                support_times=support,
                gap_min=float(gap_min),
                gap_max=float(gap_max),
                gap_grid=int(gap_grid),
                cov_reg=float(cov_reg),
            )
            if fit is None:
                continue
            chi2_dof = float(fit["chi2_dof"])
            score = abs(math.log(max(1.0e-12, chi2_dof))) + float(score_window_penalty) / float(max(1.0, fit["npts"]))
            fit["score"] = float(score)
            fit["in_chi2_band"] = 1.0 if (chi2_dof >= float(chi2_min) and chi2_dof <= float(chi2_max)) else 0.0
            candidates.append(fit)
    if not candidates:
        return None, []
    in_band = [c for c in candidates if int(c.get("in_chi2_band", 0.0)) == 1]
    pool = in_band if in_band else candidates
    best = min(pool, key=lambda c: (float(c["score"]), -int(c["npts"]), int(c["tmin"])))
    return dict(best), candidates


def _fit_tsep_one_exp_jk(
    full_curve: np.ndarray,
    jk_curves: np.ndarray,
    *,
    support_times: np.ndarray,
    idx_min: int,
    idx_max: int,
    gap_min: float,
    gap_max: float,
    gap_grid: int,
    cov_reg: float,
    full_fit: Mapping[str, object],
) -> Dict[str, object]:
    full = _fit_tsep_one_exp_window_correlated(
        full_curve,
        PV3._jackknife_covariance(np.asarray(jk_curves, dtype=np.float64)),
        idx_min=int(idx_min),
        idx_max=int(idx_max),
        support_times=support_times,
        gap_min=float(gap_min),
        gap_max=float(gap_max),
        gap_grid=int(gap_grid),
        cov_reg=float(cov_reg),
    )
    if full is None:
        raise ValueError("full one-exp fit unexpectedly failed on selected window")
    nb = int(np.asarray(jk_curves, dtype=np.float64).shape[0])
    fit_vals = np.full((nb,), float(full["value"]), dtype=np.float64)
    fit_amps = np.full((nb,), float(full["amplitude"]), dtype=np.float64)
    fit_gaps = np.full((nb,), float(full["gap"]), dtype=np.float64)
    n_fail = 0
    for ib in range(nb):
        y = np.asarray(jk_curves[ib], dtype=np.float64)
        fit = _fit_tsep_one_exp_window_correlated(
            y,
            PV3._jackknife_covariance(np.asarray(jk_curves, dtype=np.float64)),
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            support_times=support_times,
            gap_min=float(gap_min),
            gap_max=float(gap_max),
            gap_grid=int(gap_grid),
            cov_reg=float(cov_reg),
        )
        if fit is None:
            n_fail += 1
            continue
        fit_vals[ib] = float(fit["value"])
        fit_amps[ib] = float(fit["amplitude"])
        fit_gaps[ib] = float(fit["gap"])
    return {
        "full": dict(full),
        "jk_value": fit_vals,
        "jk_amplitude": fit_amps,
        "jk_gap": fit_gaps,
        "n_fit_failures_jk": int(n_fail),
        "value_err": float(PV3._scalar_jackknife_err(fit_vals, float(full["value"]))),
        "amplitude_err": float(PV3._scalar_jackknife_err(fit_amps, float(full["amplitude"]))),
        "gap_err": float(PV3._scalar_jackknife_err(fit_gaps, float(full["gap"]))),
    }


def _fit_grouped_q2_excited_state(
    grouped_q2_tsep: Sequence[Mapping[str, object]],
    *,
    cov_reg: float,
    q2_tol: float,
    tsep_max_safe: Optional[int],
    tsep_fit_range: Optional[Tuple[int, int]],
    tsep_fit_min_points: int,
    tsep_fit_chi2_min: float,
    tsep_fit_chi2_max: float,
    tsep_fit_score_window_penalty: float,
    gap_min: float,
    gap_max: float,
    gap_grid: int,
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in grouped_q2_tsep:
        q2 = float(row["q2_cont_like"])
        q2_bin = int(round(q2 / float(q2_tol))) if float(q2_tol) > 0.0 else int(round(q2 * 1.0e12))
        grouped.setdefault(q2_bin, []).append(row)

    out: List[Dict[str, object]] = []
    for _, rows_unsorted in sorted(grouped.items(), key=lambda item: float(item[1][0]["q2_cont_like"])):
        rows = sorted(rows_unsorted, key=lambda r: int(r["tsep"]))
        rows_fit = [
            r for r in rows
            if (tsep_max_safe is None or int(r["tsep"]) <= int(tsep_max_safe))
        ]
        q2 = float(rows[0]["q2_cont_like"])
        tseps_all = np.asarray([int(r["tsep"]) for r in rows], dtype=np.int64)
        tseps = np.asarray([int(r["tsep"]) for r in rows_fit], dtype=np.int64)
        if tseps.size == 0:
            continue
        ratio_full = np.asarray([float(r["form_ratio"]) for r in rows_fit], dtype=np.float64)
        direct_full = np.asarray([float(r["form_direct"]) for r in rows_fit], dtype=np.float64)
        ratio_jk = _stack_group_jk(rows_fit, "_form_ratio_jk", context=f"excited-fit Q^2={q2:.8g} ratio")
        direct_jk = _stack_group_jk(rows_fit, "_form_direct_jk", context=f"excited-fit Q^2={q2:.8g} direct")

        if tseps.size < max(4, int(tsep_fit_min_points)):
            continue

        fit_candidates: List[Dict[str, object]] = []
        if tsep_fit_range is not None:
            idx_min, idx_max = PV3._support_index_range(
                tseps, int(tsep_fit_range[0]), int(tsep_fit_range[1])
            )
            ratio_fit = _fit_tsep_one_exp_window_correlated(
                ratio_full,
                PV3._jackknife_covariance(ratio_jk),
                idx_min=int(idx_min),
                idx_max=int(idx_max),
                support_times=tseps,
                gap_min=float(gap_min),
                gap_max=float(gap_max),
                gap_grid=int(gap_grid),
                cov_reg=float(cov_reg),
            )
            if ratio_fit is None:
                raise ValueError(
                    f"Manual tsep excited-fit range [{tsep_fit_range[0]},{tsep_fit_range[1]}] failed for Q^2={q2:.8g}"
                )
        else:
            ratio_fit, fit_candidates = _select_tsep_one_exp_window(
                ratio_full,
                ratio_jk,
                support_times=tseps,
                tmin_min=int(tseps[0]),
                tmax_max=int(tseps[-1]),
                min_points=int(min(max(4, int(tsep_fit_min_points)), tseps.size)),
                chi2_min=float(tsep_fit_chi2_min),
                chi2_max=float(tsep_fit_chi2_max),
                score_window_penalty=float(tsep_fit_score_window_penalty),
                gap_min=float(gap_min),
                gap_max=float(gap_max),
                gap_grid=int(gap_grid),
                cov_reg=float(cov_reg),
            )
            if ratio_fit is None:
                raise ValueError(f"No successful tsep excited-state fit candidate found for Q^2={q2:.8g}")

        idx_min = int(ratio_fit["support_idx_min"])
        idx_max = int(ratio_fit["support_idx_max"])
        ratio_with_jk = _fit_tsep_one_exp_jk(
            ratio_full,
            ratio_jk,
            support_times=tseps,
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            gap_min=float(gap_min),
            gap_max=float(gap_max),
            gap_grid=int(gap_grid),
            cov_reg=float(cov_reg),
            full_fit=ratio_fit,
        )
        direct_with_jk = _fit_tsep_one_exp_jk(
            direct_full,
            direct_jk,
            support_times=tseps,
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            gap_min=float(gap_min),
            gap_max=float(gap_max),
            gap_grid=int(gap_grid),
            cov_reg=float(cov_reg),
            full_fit=ratio_fit,
        )

        row = {
            "q2_cont_like": float(q2),
            "model": "F + A exp(-gap * tsep)",
            "n_tsep": int(tseps.size),
            "tseps_available": tseps_all.astype(int).tolist(),
            "tseps": tseps.astype(int).tolist(),
            "excluded_tseps_thermal": [
                int(r["tsep"]) for r in rows if (tsep_max_safe is not None and int(r["tsep"]) > int(tsep_max_safe))
            ],
            "fit_window": {
                "tsep_min": float(ratio_with_jk["full"]["tmin"]),
                "tsep_max": float(ratio_with_jk["full"]["tmax"]),
                "chi2_dof": float(ratio_with_jk["full"]["chi2_dof"]),
                "npts": float(ratio_with_jk["full"]["npts"]),
            },
            "ratio_fit": {
                "form_factor": float(ratio_with_jk["full"]["value"]),
                "form_factor_err": float(ratio_with_jk["value_err"]),
                "amplitude": float(ratio_with_jk["full"]["amplitude"]),
                "amplitude_err": float(ratio_with_jk["amplitude_err"]),
                "gap": float(ratio_with_jk["full"]["gap"]),
                "gap_err": float(ratio_with_jk["gap_err"]),
                "chi2_dof": float(ratio_with_jk["full"]["chi2_dof"]),
                "n_fit_failures_jk": int(ratio_with_jk["n_fit_failures_jk"]),
                "fit_tseps": tseps[idx_min : idx_max + 1].astype(int).tolist(),
                "fit_curve": np.asarray(ratio_with_jk["full"]["fit_curve"], dtype=np.float64).tolist(),
            },
            "direct_fit": {
                "form_factor": float(direct_with_jk["full"]["value"]),
                "form_factor_err": float(direct_with_jk["value_err"]),
                "amplitude": float(direct_with_jk["full"]["amplitude"]),
                "amplitude_err": float(direct_with_jk["amplitude_err"]),
                "gap": float(direct_with_jk["full"]["gap"]),
                "gap_err": float(direct_with_jk["gap_err"]),
                "chi2_dof": float(direct_with_jk["full"]["chi2_dof"]),
                "n_fit_failures_jk": int(direct_with_jk["n_fit_failures_jk"]),
                "fit_tseps": tseps[idx_min : idx_max + 1].astype(int).tolist(),
                "fit_curve": np.asarray(direct_with_jk["full"]["fit_curve"], dtype=np.float64).tolist(),
            },
            "per_tsep": [
                {
                    "tsep": int(r["tsep"]),
                    "n_channels": int(r["n_channels"]),
                    "form_ratio": float(r["form_ratio"]),
                    "form_ratio_err": float(r["form_ratio_err"]),
                    "form_direct": float(r["form_direct"]),
                    "form_direct_err": float(r["form_direct_err"]),
                }
                for r in rows
            ],
        }
        if fit_candidates:
            row["fit_candidates"] = [
                {
                    "tmin": float(c["tmin"]),
                    "tmax": float(c["tmax"]),
                    "chi2_dof": float(c["chi2_dof"]),
                    "gap": float(c["gap"]),
                    "score": float(c["score"]),
                    "in_chi2_band": float(c["in_chi2_band"]),
                }
                for c in fit_candidates
            ]
        out.append(row)
    return out


def _analyze_channel(
    *,
    payload: Mapping,
    records: Sequence[Mapping],
    pion_samples_by_p: Mapping[int, np.ndarray],
    pion_steps_by_p: Mapping[int, np.ndarray],
    pion_meta_by_p: Mapping[int, Mapping[str, object]],
    pion_fit_cache: Dict[Tuple[int, int, bytes], Dict[str, object]],
    common_steps_override: Optional[np.ndarray],
    block_size_override: Optional[int],
    threepoint_cache: Optional[Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, Dict[str, object]]]],
    mu: int,
    pi: int,
    pf: int,
    tsep: int,
    args,
) -> Dict[str, object]:
    cache_key_3pt = (int(mu), int(pf), int(pi), int(tsep))
    if threepoint_cache is not None and cache_key_3pt in threepoint_cache:
        three_samples, three_steps, three_meta = threepoint_cache[cache_key_3pt]
    else:
        three_samples, three_steps, three_meta = PV3._extract_threepoint_channel(
            records,
            measurement=str(args.threept_measurement),
            channel=str(args.threept_channel),
            mu=int(mu),
            p_i=int(pi),
            p_f=int(pf),
            t_sep=int(tsep),
        )
    if common_steps_override is None:
        pion_aligned, three_aligned, common_steps = PV3._align_common_steps(
            pion_samples_by_p,
            pion_steps_by_p,
            required_momenta=(int(pi), int(pf)),
            three_samples=three_samples,
            three_steps=three_steps,
        )
        pion_aligned, three_aligned, common_steps = PV3._apply_common_discard_stride(
            pion_aligned,
            three_aligned,
            common_steps,
            discard=int(args.discard),
            stride=int(args.stride),
        )
    else:
        common_steps = np.asarray(common_steps_override, dtype=np.int64)
        three_aligned = _align_samples_to_steps(three_samples, three_steps, common_steps)
        pion_aligned = {
            int(p): _align_samples_to_steps(
                np.asarray(pion_samples_by_p[int(p)], dtype=np.float64),
                np.asarray(pion_steps_by_p[int(p)], dtype=np.int64),
                common_steps,
            )
            for p in sorted({int(pi), int(pf)})
        }
    support_taus = np.asarray(three_meta["support_taus"], dtype=np.int64)
    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    resolved_component, resolved_sign, resolved_label = PV3._resolve_analysis_component(
        payload,
        mu=int(mu),
        component=str(args.component),
    )

    if block_size_override is not None:
        block_size = int(block_size_override)
        block_meta = {
            "pion_tau_ref": float("nan"),
            "pion_block_auto": float("nan"),
            "threept_tau_ref": float("nan"),
            "threept_block_auto": float("nan"),
        }
    elif int(args.block_size) > 0:
        block_size = int(args.block_size)
        block_meta = {
            "pion_tau_ref": float("nan"),
            "pion_block_auto": float("nan"),
            "threept_tau_ref": float("nan"),
            "threept_block_auto": float("nan"),
        }
    else:
        block_size, block_meta = PV3._choose_joint_block_size(
            pion_aligned,
            three_aligned,
            support_taus=support_taus,
            tau_ref=max(1, min(int(tsep) // 2, int(tsep) - 1)),
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
            component=str(resolved_component),
        )

    t_extent = PV3._temporal_extent_from_payload(payload, pion_meta_by_p[int(pi)]["support_times"])
    pion_fit_range = _parse_fit_range_text(args.pion_fit_range)
    pion_results: Dict[int, Dict[str, object]] = {}
    common_steps_key = np.asarray(common_steps, dtype=np.int64).tobytes()
    for p in sorted({int(pi), int(pf)}):
        support_times = np.asarray(pion_meta_by_p[int(p)]["support_times"], dtype=np.int64)
        tmax_default = (int(t_extent) // 2) - 1
        support_half = support_times[support_times <= int(tmax_default)]
        tmax_auto = int(support_half[-1]) if support_half.size > 0 else int(support_times[-1])
        tmax_max = int(args.pion_tmax_max) if int(args.pion_tmax_max) >= 0 else int(tmax_auto)
        cache_key = (int(p), int(block_size), common_steps_key)
        cached = pion_fit_cache.get(cache_key)
        if cached is None:
            cached = PV3._fit_pion_channel(
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
                progress=False,
                progress_prefix=f"p={int(p)}",
            )
            pion_fit_cache[cache_key] = cached
        pion_results[int(p)] = cached

    nb = int(next(iter(pion_results.values()))["jk_corr"].shape[0])
    three_blocks = PV3._block_means_nd(three_aligned, block_size=int(block_size))
    three_full = np.asarray(np.mean(three_blocks, axis=0), dtype=np.complex128)
    three_jk = PV3._jackknife_loo_from_blocks(three_blocks)

    p_i_res = pion_results[int(pi)]
    p_f_res = pion_results[int(pf)]
    ratio_full = PV3._effective_ratio_curve(
        three_full,
        np.asarray(p_i_res["full_corr"], dtype=np.float64),
        np.asarray(p_f_res["full_corr"], dtype=np.float64),
        taus=support_taus,
        t_sep=int(tsep),
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
        ratio_jk[ib] = PV3._effective_ratio_curve(
            np.asarray(three_jk[ib], dtype=np.complex128),
            np.asarray(p_i_res["jk_corr"], dtype=np.float64)[ib],
            np.asarray(p_f_res["jk_corr"], dtype=np.float64)[ib],
            taus=support_taus,
            t_sep=int(tsep),
            support_times_i=np.asarray(p_i_res["support_times"], dtype=np.int64),
            support_times_f=np.asarray(p_f_res["support_times"], dtype=np.int64),
            amp_i=float(np.asarray(p_i_res["jk_amp"], dtype=np.float64)[ib]),
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),
            amp_f=float(np.asarray(p_f_res["jk_amp"], dtype=np.float64)[ib]),
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),
        )
        mratio_jk[ib] = PV3._effective_matrix_element_from_ratio(
            ratio_jk[ib],
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),
            zv=float(args.zv),
        )
        mdir_jk[ib] = PV3._effective_matrix_element_from_z(
            np.asarray(three_jk[ib], dtype=np.complex128),
            taus=support_taus,
            t_sep=int(tsep),
            energy_i=float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]),
            energy_f=float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib]),
            z_i=float(np.asarray(p_i_res["jk_z"], dtype=np.float64)[ib]),
            z_f=float(np.asarray(p_f_res["jk_z"], dtype=np.float64)[ib]),
            zv=float(args.zv),
        )
    mratio_full = PV3._effective_matrix_element_from_ratio(
        ratio_full,
        energy_i=float(p_i_res["energy"]),
        energy_f=float(p_f_res["energy"]),
        zv=float(args.zv),
    )
    mdir_full = PV3._effective_matrix_element_from_z(
        three_full,
        taus=support_taus,
        t_sep=int(tsep),
        energy_i=float(p_i_res["energy"]),
        energy_f=float(p_f_res["energy"]),
        z_i=float(p_i_res["z_overlap"]),
        z_f=float(p_f_res["z_overlap"]),
        zv=float(args.zv),
    )

    ratio_comp_full = float(resolved_sign) * PV3._component_view(mratio_full, resolved_component)
    ratio_comp_jk = float(resolved_sign) * PV3._component_view(mratio_jk, resolved_component)
    direct_comp_full = float(resolved_sign) * PV3._component_view(mdir_full, resolved_component)
    direct_comp_jk = float(resolved_sign) * PV3._component_view(mdir_jk, resolved_component)

    plateau_range = _parse_fit_range_text(args.plateau_range)
    if plateau_range is not None:
        idx_min, idx_max = PV3._support_index_range(support_taus, int(plateau_range[0]), int(plateau_range[1]))
        plateau = PV3._fit_constant_window_correlated(
            ratio_comp_full,
            PV3._jackknife_covariance(ratio_comp_jk),
            idx_min=int(idx_min),
            idx_max=int(idx_max),
            support_times=support_taus,
            cov_reg=float(args.cov_reg),
        )
        plateau_candidates: List[Dict[str, object]] = []
        if plateau is None:
            raise ValueError(f"Manual plateau range [{plateau_range[0]},{plateau_range[1]}] failed")
    else:
        tmin_min = max(int(args.tau_margin), int(support_taus[0]))
        tmax_max = min(int(tsep) - int(args.tau_margin), int(support_taus[-1]))
        plateau, plateau_candidates = PV3._select_plateau_window(
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
            progress=False,
            progress_prefix=f"mu={int(mu)} pf={int(pf)} pi={int(pi)} tsep={int(tsep)}",
        )
        if plateau is None:
            raise ValueError("No successful plateau candidate was found")

    weights = PV3._plateau_linear_weights(
        ratio_comp_jk,
        idx_min=int(plateau["support_idx_min"]),
        idx_max=int(plateau["support_idx_max"]),
        cov_reg=float(args.cov_reg),
    )
    matrix_ratio, matrix_ratio_err = PV3._plateau_from_weights(
        ratio_comp_full,
        ratio_comp_jk,
        weights,
        int(plateau["support_idx_min"]),
        int(plateau["support_idx_max"]),
    )
    matrix_direct, matrix_direct_err = PV3._plateau_from_weights(
        direct_comp_full,
        direct_comp_jk,
        weights,
        int(plateau["support_idx_min"]),
        int(plateau["support_idx_max"]),
    )

    kin_full, kin_label, is_temporal = PV3._kinematic_prefactor_1d(
        payload,
        mu=int(mu),
        p_i=int(pi),
        p_f=int(pf),
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
    form_ratio_jk_plateau = np.full((nb,), np.nan, dtype=np.float64)
    form_direct_jk_plateau = np.full((nb,), np.nan, dtype=np.float64)
    if abs(float(kin_full)) > 1.0e-14:
        form_curve_full = ratio_comp_full / float(kin_full)
        for ib in range(nb):
            if is_temporal:
                kin_jk = float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]) + float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib])
            else:
                kin_jk = float(kin_full)
            if abs(float(kin_jk)) > 1.0e-14:
                form_curve_jk[ib] = ratio_comp_jk[ib] / kin_jk
                form_direct_jk[ib] = direct_comp_jk[ib] / kin_jk
        form_ratio, form_ratio_err = PV3._plateau_from_weights(
            form_curve_full,
            form_curve_jk,
            weights,
            int(plateau["support_idx_min"]),
            int(plateau["support_idx_max"]),
        )
        form_direct, form_direct_err = PV3._plateau_from_weights(
            direct_comp_full / float(kin_full),
            form_direct_jk,
            weights,
            int(plateau["support_idx_min"]),
            int(plateau["support_idx_max"]),
        )
        i0 = int(plateau["support_idx_min"])
        i1 = int(plateau["support_idx_max"]) + 1
        form_ratio_jk_plateau = np.asarray(form_curve_jk[:, i0:i1] @ weights, dtype=np.float64)
        form_direct_jk_plateau = np.asarray(form_direct_jk[:, i0:i1] @ weights, dtype=np.float64)

    lx = PV3._spatial_extent_from_payload(payload)
    qi = PV3._continuum_momentum_1d(int(pi), lx)
    qf = PV3._continuum_momentum_1d(int(pf), lx)
    q2 = (float(p_f_res["energy"]) - float(p_i_res["energy"])) ** 2 + (qf - qi) ** 2

    return {
        "mu": int(mu),
        "pf": int(pf),
        "pi": int(pi),
        "tsep": int(tsep),
        "q2_cont_like": float(q2),
        "is_temporal": bool(is_temporal),
        "form_available": bool(abs(float(kin_full)) > 1.0e-14),
        "kinematic_prefactor": float(kin_full),
        "kinematic_prefactor_label": str(kin_label),
        "component_requested": str(args.component),
        "component_resolved": str(resolved_component),
        "component_sign": float(resolved_sign),
        "component_label": str(resolved_label),
        "n_common_samples": int(common_steps.size),
        "block_size": int(block_size),
        "plateau_tmin": float(plateau["tmin"]),
        "plateau_tmax": float(plateau["tmax"]),
        "plateau_chi2_dof": float(plateau["chi2_dof"]),
        "matrix_ratio": float(matrix_ratio),
        "matrix_ratio_err": float(matrix_ratio_err),
        "matrix_direct": float(matrix_direct),
        "matrix_direct_err": float(matrix_direct_err),
        "form_ratio": float(form_ratio),
        "form_ratio_err": float(form_ratio_err),
        "form_direct": float(form_direct),
        "form_direct_err": float(form_direct_err),
        "pion_energy_pi": float(p_i_res["energy"]),
        "pion_energy_pi_err": float(p_i_res["energy_err"]),
        "pion_energy_pf": float(p_f_res["energy"]),
        "pion_energy_pf_err": float(p_f_res["energy_err"]),
        "block_meta": block_meta,
        "n_plateau_candidates": int(len(plateau_candidates)),
        "_form_ratio_jk": np.asarray(form_ratio_jk_plateau, dtype=np.float64).tolist(),
        "_form_direct_jk": np.asarray(form_direct_jk_plateau, dtype=np.float64).tolist(),
    }


def _setup_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch analysis for pion form-factor checkpoint data")
    ap.add_argument("--input", required=True)
    ap.add_argument("--pion-measurement", default="pion_2pt")
    ap.add_argument("--pion-channel", default="c")
    ap.add_argument("--threept-measurement", default="pion_3pt_vector")
    ap.add_argument("--threept-channel", default="c3")
    ap.add_argument("--mu", type=int, default=-1, help="current index; <0 selects the temporal current unless --all-mus is set")
    ap.add_argument("--all-mus", action="store_true", help="analyze all available current components")
    ap.add_argument(
        "--pair-mode",
        default="breit",
        choices=("breit", "all", "equal", "diagonal", "same", "q2zero"),
        help="channel filter: breit uses pf=-pi, equal/diagonal uses pf=pi, all keeps every measured pair",
    )
    ap.add_argument("--max-abs-p", type=int, default=99)
    ap.add_argument("--tsep-min", type=int, default=4)
    ap.add_argument("--tsep-max", type=int, default=16)
    ap.add_argument(
        "--tsep-auto-thermal-cut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="for reduced-Q^2 tsep fits/diagnostics, exclude tsep > T/2 - thermal_margin",
    )
    ap.add_argument(
        "--tsep-thermal-margin",
        type=int,
        default=1,
        help="thermal safety margin used with --tsep-auto-thermal-cut; default excludes tsep >= T/2 on even T",
    )
    ap.add_argument(
        "--component",
        choices=("auto", "re", "im", "abs"),
        default="auto",
        help="3pt component used in the reduced matrix element; auto uses Re for the temporal current and -Im for spatial currents",
    )
    ap.add_argument("--zv", type=float, default=1.0)
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iat-method", default="ips", choices=("ips", "sokal", "gamma"))
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0)
    ap.add_argument("--pion-fit-range", type=str, default="")
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
    ap.add_argument("--plateau-range", type=str, default="")
    ap.add_argument("--tau-margin", type=int, default=1)
    ap.add_argument("--plateau-min-points", type=int, default=3)
    ap.add_argument("--plateau-chi2-min", type=float, default=0.5)
    ap.add_argument("--plateau-chi2-max", type=float, default=2.0)
    ap.add_argument("--plateau-score-window-penalty", type=float, default=0.20)
    ap.add_argument("--q2-tol", type=float, default=1.0e-12, help="binning tolerance used when grouping channels by Q^2")
    ap.add_argument("--tsep-plateau-range", type=str, default="", help="optional manual source-sink separation plateau range tsep_min,tsep_max")
    ap.add_argument("--tsep-plateau-min-points", type=int, default=2)
    ap.add_argument("--tsep-plateau-chi2-min", type=float, default=0.5)
    ap.add_argument("--tsep-plateau-chi2-max", type=float, default=2.0)
    ap.add_argument("--tsep-plateau-score-window-penalty", type=float, default=0.15)
    ap.add_argument("--tsep-fit-range", type=str, default="", help="optional manual source-sink fit range for the one-exp excited-state fit")
    ap.add_argument("--tsep-fit-min-points", type=int, default=4)
    ap.add_argument("--tsep-fit-chi2-min", type=float, default=0.5)
    ap.add_argument("--tsep-fit-chi2-max", type=float, default=2.0)
    ap.add_argument("--tsep-fit-score-window-penalty", type=float, default=0.15)
    ap.add_argument("--tsep-fit-gap-min", type=float, default=1.0e-3)
    ap.add_argument("--tsep-fit-gap-max", type=float, default=4.0)
    ap.add_argument("--tsep-fit-gap-grid", type=int, default=256)
    ap.add_argument("--cov-reg", type=float, default=1.0e-10)
    ap.add_argument("--prefix", default="pion_form_factor")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-gui", action="store_true")
    return ap


def _charge_normalization_report(results: Sequence[Mapping[str, object]], *, q2_tol: float = 1.0e-12) -> Dict[str, object]:
    channels: List[Dict[str, object]] = []
    for row in results:
        q2 = float(row["q2_cont_like"])
        if not np.isfinite(q2) or abs(q2) > float(q2_tol):
            continue
        f0 = float(row["form_ratio"])
        ferr = float(row["form_ratio_err"])
        if not (np.isfinite(f0) and np.isfinite(ferr)):
            continue
        delta = float(f0 - 1.0)
        sigma = float(abs(delta) / ferr) if ferr > 0.0 else float("nan")
        channels.append(
            {
                "mu": int(row["mu"]),
                "pf": int(row["pf"]),
                "pi": int(row["pi"]),
                "tsep": int(row["tsep"]),
                "q2_cont_like": float(q2),
                "f_pi_q2_0": float(f0),
                "f_pi_q2_0_err": float(ferr),
                "delta_from_unity": float(delta),
                "significance_sigma": float(sigma),
            }
        )
    channels = sorted(channels, key=lambda r: (int(r["tsep"]), int(r["pf"]), int(r["pi"])))
    best = None
    if channels:
        best = min(channels, key=lambda r: float(r["f_pi_q2_0_err"]))
    return {
        "n_zero_q2_channels": int(len(channels)),
        "channels": channels,
        "best_precision_channel": best,
        "note": "Per-channel F_pi(Q^2=0) check. Different tsep values are correlated and are not averaged here.",
    }


def _default_tsep_max_safe(payload: Mapping, requested_tsep_max: int, thermal_margin: int) -> int:
    cfg = payload.get("config", {})
    raw_shape = None
    if isinstance(cfg, Mapping):
        run_cfg = cfg.get("run", {})
        if isinstance(run_cfg, Mapping):
            raw_shape = run_cfg.get("shape", None)
    if isinstance(raw_shape, (list, tuple)) and raw_shape:
        t_extent = int(raw_shape[-1])
    else:
        t_extent = int(requested_tsep_max) + 1
    return int(min(int(requested_tsep_max), max(1, int(t_extent // 2) - int(max(0, thermal_margin)))))


def main() -> int:
    ap = _setup_parser()
    args = ap.parse_args()
    ckpt_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{args.prefix}.json"
    png_path = outdir / f"{args.prefix}.png"

    payload, records = PV3.FIT2PT._load_checkpoint(ckpt_path)
    if not records:
        raise ValueError(f"No inline_records found in checkpoint: {ckpt_path}")
    pion_samples_by_p, pion_steps_by_p, pion_meta_by_p = PV3.FIT2PT._extract_by_momentum(
        records,
        measurement=str(args.pion_measurement),
        channel=str(args.pion_channel),
    )
    available = _discover_threepoint_channels(
        records,
        measurement=str(args.threept_measurement),
        channel=str(args.threept_channel),
    )
    if not available:
        raise ValueError(
            f"No 3pt channels found for measurement='{args.threept_measurement}' channel='{args.threept_channel}'"
        )

    mu = None if bool(args.all_mus) else (None if int(args.mu) < 0 else int(args.mu))
    if mu is None and (not bool(args.all_mus)):
        mu = int(PV3._temporal_mu_from_payload(payload))
    selected = _filter_channels(
        available,
        mu=None if mu is None else int(mu),
        tsep_min=int(args.tsep_min),
        tsep_max=int(args.tsep_max),
        pair_mode=str(args.pair_mode),
        max_abs_p=int(args.max_abs_p),
    )
    if not selected:
        raise ValueError("No 3pt channels remain after filtering")

    batch_common_steps, batch_block_size, threepoint_cache = _prepare_batch_common_steps_and_block_size(
        payload=payload,
        records=records,
        selected=selected,
        pion_samples_by_p=pion_samples_by_p,
        pion_steps_by_p=pion_steps_by_p,
        pion_meta_by_p=pion_meta_by_p,
        args=args,
    )
    print(
        f"[batch] common_samples={int(batch_common_steps.size)} block_size={int(batch_block_size)}",
        flush=True,
    )

    tsep_max_safe = None
    if bool(args.tsep_auto_thermal_cut):
        tsep_max_safe = _default_tsep_max_safe(
            payload,
            requested_tsep_max=int(args.tsep_max),
            thermal_margin=int(args.tsep_thermal_margin),
        )

    results: List[Dict[str, object]] = []
    pion_fit_cache: Dict[Tuple[int, int, bytes], Dict[str, object]] = {}
    for idx, (cur_mu, pf, pi, tsep) in enumerate(selected, start=1):
        print(
            f"[{idx}/{len(selected)}] mu={cur_mu} pf={pf} pi={pi} tsep={tsep}",
            flush=True,
        )
        res = _analyze_channel(
            payload=payload,
            records=records,
            pion_samples_by_p=pion_samples_by_p,
            pion_steps_by_p=pion_steps_by_p,
            pion_meta_by_p=pion_meta_by_p,
            pion_fit_cache=pion_fit_cache,
            common_steps_override=batch_common_steps,
            block_size_override=batch_block_size,
            threepoint_cache=threepoint_cache,
            mu=int(cur_mu),
            pi=int(pi),
            pf=int(pf),
            tsep=int(tsep),
            args=args,
        )
        results.append(res)

    serializable_results = [
        {k: v for k, v in row.items() if not str(k).startswith("_")}
        for row in results
    ]
    grouped_q2_tsep = _group_results_by_q2_tsep(
        results,
        cov_reg=float(args.cov_reg),
        q2_tol=float(args.q2_tol),
    )
    grouped_q2_constant_diag = _group_results_by_q2(
        grouped_q2_tsep,
        cov_reg=float(args.cov_reg),
        q2_tol=float(args.q2_tol),
        tsep_max_safe=tsep_max_safe,
        tsep_plateau_range=_parse_fit_range_text(args.tsep_plateau_range),
        tsep_plateau_min_points=int(args.tsep_plateau_min_points),
        tsep_plateau_chi2_min=float(args.tsep_plateau_chi2_min),
        tsep_plateau_chi2_max=float(args.tsep_plateau_chi2_max),
        tsep_plateau_score_window_penalty=float(args.tsep_plateau_score_window_penalty),
    )
    grouped_q2_excited_fit = _fit_grouped_q2_excited_state(
        grouped_q2_tsep,
        cov_reg=float(args.cov_reg),
        q2_tol=float(args.q2_tol),
        tsep_max_safe=tsep_max_safe,
        tsep_fit_range=_parse_fit_range_text(args.tsep_fit_range),
        tsep_fit_min_points=int(args.tsep_fit_min_points),
        tsep_fit_chi2_min=float(args.tsep_fit_chi2_min),
        tsep_fit_chi2_max=float(args.tsep_fit_chi2_max),
        tsep_fit_score_window_penalty=float(args.tsep_fit_score_window_penalty),
        gap_min=float(args.tsep_fit_gap_min),
        gap_max=float(args.tsep_fit_gap_max),
        gap_grid=int(args.tsep_fit_gap_grid),
    )

    summary = {
        "input": str(ckpt_path),
        "measurement_names": {
            "pion": str(args.pion_measurement),
            "threept": str(args.threept_measurement),
        },
        "filters": {
            "mu": None if mu is None else int(mu),
            "all_mus": bool(args.all_mus),
            "pair_mode": str(args.pair_mode),
            "max_abs_p": int(args.max_abs_p),
            "tsep_min": int(args.tsep_min),
            "tsep_max": int(args.tsep_max),
            "component": str(args.component),
            "zv": float(args.zv),
            "q2_tol": float(args.q2_tol),
            "tsep_auto_thermal_cut": bool(args.tsep_auto_thermal_cut),
            "tsep_thermal_margin": int(args.tsep_thermal_margin),
            "tsep_max_safe": None if tsep_max_safe is None else int(tsep_max_safe),
            "batch_common_samples": int(batch_common_steps.size),
            "batch_block_size": int(batch_block_size),
        },
        "analysis_notes": {
            "per_channel_methods": [
                "ratio estimator with overlap cancellation",
                "direct estimator using Z-factors from separate 2pt fits",
            ],
            "current_component_convention": (
                "By default (--component auto), the temporal current is reduced with Re[M_eff] while spatial currents "
                "are reduced with -Im[M_eff]. This matches the Euclidean phase convention used by the 2D Schwinger-model measurement."
            ),
            "q2_definition": "Q^2 = (E_f - E_i)^2 + (p_f - p_i)^2 in Euclidean conventions using fitted pion energies",
            "late_tsep_reduction": (
                "grouped_q2_constant_diag is a late-tsep constant diagnostic after grouping equal-(Q^2,tsep) channels. "
                "It should not be interpreted as a full excited-state-controlled extraction."
            ),
            "preferred_reduced_result": (
                "grouped_q2 uses a one-exponential excited-state fit in tsep to the reduced fixed-Q^2 data."
            ),
            "not_implemented": [
                "global simultaneous correlated fit of 2pt and 3pt data across all tsep",
                "multi-state simultaneous correlated fit of reduced Q^2 data beyond the one-exponential tsep model",
            ],
        },
        "n_channels": int(len(serializable_results)),
        "channels": serializable_results,
        "grouped_q2_tsep": [
            {k: v for k, v in row.items() if not str(k).startswith("_")}
            for row in grouped_q2_tsep
        ],
        "grouped_q2_constant_diag": [
            {k: v for k, v in row.items() if not str(k).startswith("_")}
            for row in grouped_q2_constant_diag
        ],
        "grouped_q2": [
            {k: v for k, v in row.items() if not str(k).startswith("_")}
            for row in grouped_q2_excited_fit
        ],
        "charge_normalization": _charge_normalization_report(serializable_results),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plt = PV3._setup_matplotlib(bool(args.no_gui))
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        vals = np.asarray(
            [r["form_ratio"] if np.isfinite(r["form_ratio"]) else r["matrix_ratio"] for r in serializable_results],
            dtype=np.float64,
        )
        errs = np.asarray(
            [r["form_ratio_err"] if np.isfinite(r["form_ratio_err"]) else r["matrix_ratio_err"] for r in serializable_results],
            dtype=np.float64,
        )
        q2s = np.asarray([r["q2_cont_like"] for r in serializable_results], dtype=np.float64)
        tseps = np.asarray([r["tsep"] for r in serializable_results], dtype=np.float64)
        sc = axes[0].scatter(q2s, vals, c=tseps, cmap="viridis", s=42)
        axes[0].errorbar(q2s, vals, yerr=errs, fmt="none", ecolor="0.45", alpha=0.8, capsize=2.0)
        axes[0].set_xlabel(r"$Q^2$")
        axes[0].set_ylabel("F_pi" if np.any(np.isfinite([r["form_ratio"] for r in serializable_results])) else "matrix element")
        axes[0].set_title("All analyzed channels")
        fig.colorbar(sc, ax=axes[0], label="tsep")
        for grp in grouped_q2_tsep:
            axes[0].errorbar(
                [float(grp["q2_cont_like"])],
                [float(grp["form_ratio"])],
                yerr=[float(grp["form_ratio_err"])],
                fmt="ks",
                ms=5.0,
                capsize=2.0,
                alpha=0.9,
            )
        for grp in grouped_q2_constant_diag:
            axes[0].errorbar(
                [float(grp["q2_cont_like"])],
                [float(grp["form_ratio"])],
                yerr=[float(grp["form_ratio_err"])],
                fmt="D",
                color="tab:red",
                mfc="white",
                mec="tab:red",
                ms=6.0,
                capsize=2.0,
                alpha=0.95,
            )
        for grp in grouped_q2_excited_fit:
            axes[0].errorbar(
                [float(grp["q2_cont_like"])],
                [float(grp["ratio_fit"]["form_factor"])],
                yerr=[float(grp["ratio_fit"]["form_factor_err"])],
                fmt="o",
                color="tab:red",
                mfc="tab:red",
                mec="tab:red",
                ms=5.5,
                capsize=2.0,
                alpha=0.95,
            )

        grouped: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for row in serializable_results:
            grouped.setdefault((int(row["pf"]), int(row["pi"])), []).append(row)
        for (pf, pi), rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda r: int(r["tsep"]))
            x = np.asarray([r["tsep"] for r in rows], dtype=np.float64)
            y = np.asarray([
                r["form_ratio"] if np.isfinite(r["form_ratio"]) else r["matrix_ratio"]
                for r in rows
            ], dtype=np.float64)
            yerr = np.asarray([
                r["form_ratio_err"] if np.isfinite(r["form_ratio_err"]) else r["matrix_ratio_err"]
                for r in rows
            ], dtype=np.float64)
            lab_mu = sorted({int(r["mu"]) for r in rows})
            axes[1].errorbar(x, y, yerr=yerr, marker="o", ms=4.0, capsize=2.0, label=f"pf={pf}, pi={pi}, mu={lab_mu}")
        for grp in grouped_q2_tsep:
            axes[1].errorbar(
                [float(grp["tsep"])],
                [float(grp["form_ratio"])],
                yerr=[float(grp["form_ratio_err"])],
                fmt="ks",
                ms=5.0,
                capsize=2.0,
                alpha=0.9,
            )
        cmap = plt.get_cmap("tab10")
        for idx, grp in enumerate(grouped_q2_constant_diag):
            color = cmap(idx % 10)
            plat = grp["plateau"]
            axes[1].hlines(
                float(grp["form_ratio"]),
                float(plat["tsep_min"]) - 0.25,
                float(plat["tsep_max"]) + 0.25,
                colors=[color],
                linestyles="--",
                linewidth=1.4,
            )
            axes[1].fill_between(
                [float(plat["tsep_min"]) - 0.25, float(plat["tsep_max"]) + 0.25],
                float(grp["form_ratio"]) - float(grp["form_ratio_err"]),
                float(grp["form_ratio"]) + float(grp["form_ratio_err"]),
                color=color,
                alpha=0.10,
            )
        for idx, grp in enumerate(grouped_q2_excited_fit):
            color = cmap(idx % 10)
            xfit = np.asarray(grp["ratio_fit"]["fit_tseps"], dtype=np.float64)
            yfit = np.asarray(grp["ratio_fit"]["fit_curve"], dtype=np.float64)
            axes[1].plot(xfit, yfit, color=color, lw=1.6, ls="-", alpha=0.9)
        axes[1].set_xlabel("tsep")
        axes[1].set_ylabel("F_pi" if np.any(np.isfinite([r["form_ratio"] for r in serializable_results])) else "matrix element")
        axes[1].set_title("Source-sink separation scan")
        axes[1].legend(loc="best", fontsize=8)
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

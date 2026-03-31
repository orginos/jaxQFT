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
        elif mode not in ("all", "cartesian", "full"):
            raise ValueError(f"Unsupported pair-mode filter: {pair_mode!r}")
        out.append((int(cur_mu), int(pf), int(pi), int(tsep)))
    return sorted(out, key=lambda row: (row[0], row[1] - row[2], row[1], row[2], row[3]))


def _analyze_channel(
    *,
    payload: Mapping,
    records: Sequence[Mapping],
    pion_samples_by_p: Mapping[int, np.ndarray],
    pion_steps_by_p: Mapping[int, np.ndarray],
    pion_meta_by_p: Mapping[int, Mapping[str, object]],
    mu: int,
    pi: int,
    pf: int,
    tsep: int,
    args,
) -> Dict[str, object]:
    three_samples, three_steps, three_meta = PV3._extract_threepoint_channel(
        records,
        measurement=str(args.threept_measurement),
        channel=str(args.threept_channel),
        mu=int(mu),
        p_i=int(pi),
        p_f=int(pf),
        t_sep=int(tsep),
    )
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
    support_taus = np.asarray(three_meta["support_taus"], dtype=np.int64)
    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    if int(args.block_size) > 0:
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
            component=str(args.component),
        )

    t_extent = PV3._temporal_extent_from_payload(payload, pion_meta_by_p[int(pi)]["support_times"])
    pion_fit_range = _parse_fit_range_text(args.pion_fit_range)
    pion_results: Dict[int, Dict[str, object]] = {}
    for p in sorted({int(pi), int(pf)}):
        support_times = np.asarray(pion_meta_by_p[int(p)]["support_times"], dtype=np.int64)
        tmax_default = (int(t_extent) // 2) - 1
        support_half = support_times[support_times <= int(tmax_default)]
        tmax_auto = int(support_half[-1]) if support_half.size > 0 else int(support_times[-1])
        tmax_max = int(args.pion_tmax_max) if int(args.pion_tmax_max) >= 0 else int(tmax_auto)
        pion_results[int(p)] = PV3._fit_pion_channel(
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

    ratio_comp_full = PV3._component_view(mratio_full, args.component)
    ratio_comp_jk = PV3._component_view(mratio_jk, args.component)
    direct_comp_full = PV3._component_view(mdir_full, args.component)
    direct_comp_jk = PV3._component_view(mdir_jk, args.component)

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

    temporal_mu = PV3._temporal_mu_from_payload(payload)
    is_temporal = int(mu) == int(temporal_mu)
    form_ratio = float("nan")
    form_ratio_err = float("nan")
    form_direct = float("nan")
    form_direct_err = float("nan")
    if is_temporal:
        kin_full = float(p_i_res["energy"]) + float(p_f_res["energy"])
        form_curve_full = np.full_like(ratio_comp_full, np.nan, dtype=np.float64)
        form_curve_jk = np.full_like(ratio_comp_jk, np.nan, dtype=np.float64)
        form_direct_jk = np.full_like(direct_comp_jk, np.nan, dtype=np.float64)
        if kin_full != 0.0:
            form_curve_full = ratio_comp_full / kin_full
        for ib in range(nb):
            kin_jk = float(np.asarray(p_i_res["jk_energy"], dtype=np.float64)[ib]) + float(np.asarray(p_f_res["jk_energy"], dtype=np.float64)[ib])
            if kin_jk != 0.0:
                form_curve_jk[ib] = ratio_comp_jk[ib] / kin_jk
                form_direct_jk[ib] = direct_comp_jk[ib] / kin_jk
        if kin_full != 0.0:
            form_ratio, form_ratio_err = PV3._plateau_from_weights(
                form_curve_full,
                form_curve_jk,
                weights,
                int(plateau["support_idx_min"]),
                int(plateau["support_idx_max"]),
            )
            form_direct, form_direct_err = PV3._plateau_from_weights(
                direct_comp_full / kin_full,
                form_direct_jk,
                weights,
                int(plateau["support_idx_min"]),
                int(plateau["support_idx_max"]),
            )

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
    }


def _setup_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch analysis for pion form-factor checkpoint data")
    ap.add_argument("--input", required=True)
    ap.add_argument("--pion-measurement", default="pion_2pt")
    ap.add_argument("--pion-channel", default="c")
    ap.add_argument("--threept-measurement", default="pion_3pt_vector")
    ap.add_argument("--threept-channel", default="c3")
    ap.add_argument("--mu", type=int, default=-1, help="current index; <0 selects the temporal current")
    ap.add_argument("--pair-mode", default="breit", choices=("breit", "all"))
    ap.add_argument("--max-abs-p", type=int, default=99)
    ap.add_argument("--tsep-min", type=int, default=4)
    ap.add_argument("--tsep-max", type=int, default=16)
    ap.add_argument("--component", choices=("re", "im", "abs"), default="re")
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

    mu = None if int(args.mu) < 0 else int(args.mu)
    if mu is None:
        mu = int(PV3._temporal_mu_from_payload(payload))
    selected = _filter_channels(
        available,
        mu=int(mu),
        tsep_min=int(args.tsep_min),
        tsep_max=int(args.tsep_max),
        pair_mode=str(args.pair_mode),
        max_abs_p=int(args.max_abs_p),
    )
    if not selected:
        raise ValueError("No 3pt channels remain after filtering")

    results: List[Dict[str, object]] = []
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
            mu=int(cur_mu),
            pi=int(pi),
            pf=int(pf),
            tsep=int(tsep),
            args=args,
        )
        results.append(res)

    summary = {
        "input": str(ckpt_path),
        "measurement_names": {
            "pion": str(args.pion_measurement),
            "threept": str(args.threept_measurement),
        },
        "filters": {
            "mu": int(mu),
            "pair_mode": str(args.pair_mode),
            "max_abs_p": int(args.max_abs_p),
            "tsep_min": int(args.tsep_min),
            "tsep_max": int(args.tsep_max),
            "component": str(args.component),
            "zv": float(args.zv),
        },
        "n_channels": int(len(results)),
        "channels": results,
        "charge_normalization": _charge_normalization_report(results),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plt = PV3._setup_matplotlib(bool(args.no_gui))
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        vals = np.asarray(
            [r["form_ratio"] if np.isfinite(r["form_ratio"]) else r["matrix_ratio"] for r in results],
            dtype=np.float64,
        )
        errs = np.asarray(
            [r["form_ratio_err"] if np.isfinite(r["form_ratio_err"]) else r["matrix_ratio_err"] for r in results],
            dtype=np.float64,
        )
        q2s = np.asarray([r["q2_cont_like"] for r in results], dtype=np.float64)
        tseps = np.asarray([r["tsep"] for r in results], dtype=np.float64)
        sc = axes[0].scatter(q2s, vals, c=tseps, cmap="viridis", s=42)
        axes[0].errorbar(q2s, vals, yerr=errs, fmt="none", ecolor="0.45", alpha=0.8, capsize=2.0)
        axes[0].set_xlabel(r"$Q^2$")
        axes[0].set_ylabel("F_pi" if np.any(np.isfinite([r["form_ratio"] for r in results])) else "matrix element")
        axes[0].set_title("All analyzed channels")
        fig.colorbar(sc, ax=axes[0], label="tsep")

        grouped: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for row in results:
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
            axes[1].errorbar(x, y, yerr=yerr, marker="o", ms=4.0, capsize=2.0, label=f"pf={pf}, pi={pi}")
        axes[1].set_xlabel("tsep")
        axes[1].set_ylabel("F_pi" if np.any(np.isfinite([r["form_ratio"] for r in results])) else "matrix element")
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

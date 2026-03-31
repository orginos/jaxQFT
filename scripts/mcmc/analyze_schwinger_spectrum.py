#!/usr/bin/env python3
"""Combined Schwinger spectroscopy analysis from mcmc checkpoint data.

This script combines the existing DD-independent analysis paths into one
correlated workflow:

1) single-pion spectrum from `pion_2pt`,
2) two-pion I=2 spectrum from `pipi_i2_matrix`,
3) correlated energy shifts Delta E_n = W_n - E_n^free using the same
   blocked-jackknife samples for both sectors.

The current free two-particle reference is the ordered noninteracting spectrum
constructed from the measured one-pion energies on the same ensemble,
    E_n^free = 2 E_pi(p_n),
where `p_n` are the basis momenta stored in the `pipi_i2_matrix` record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
PIPI = _load_script_module("analyze_pipi_i2_gevp_mod", ROOT / "scripts" / "mcmc" / "analyze_pipi_i2_gevp.py")


def _fmt_pm_text(name: str, value: float, err: float, *, value_fmt: str = ".5f", err_fmt: str = ".3g") -> str:
    if np.isfinite(float(value)) and np.isfinite(float(err)):
        return f"{name}={format(float(value), value_fmt)} +/- {format(float(err), err_fmt)}"
    if np.isfinite(float(value)):
        return f"{name}={format(float(value), value_fmt)}"
    return f"{name}=n/a"


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


def _align_common_steps(
    pion_samples_by_p: Mapping[int, np.ndarray],
    pion_steps_by_p: Mapping[int, np.ndarray],
    required_momenta: Sequence[int],
    pipi_samples: np.ndarray,
    pipi_steps: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    req = sorted({abs(int(p)) for p in required_momenta})
    common = set(int(s) for s in np.asarray(pipi_steps, dtype=np.int64).tolist())
    for p in req:
        if int(p) not in pion_steps_by_p:
            raise ValueError(f"Required pion momentum p={int(p)} is missing from pion_2pt records")
        common &= set(int(s) for s in np.asarray(pion_steps_by_p[int(p)], dtype=np.int64).tolist())
    steps = np.asarray(sorted(common), dtype=np.int64)
    if steps.size < 4:
        raise ValueError(f"Need at least 4 common pion/pipi samples; got {int(steps.size)}")

    pipi_index = {int(s): i for i, s in enumerate(np.asarray(pipi_steps, dtype=np.int64).tolist())}
    pipi_aligned = np.asarray([pipi_samples[pipi_index[int(s)]] for s in steps], dtype=np.complex128)

    pion_aligned: Dict[int, np.ndarray] = {}
    for p in req:
        st = np.asarray(pion_steps_by_p[int(p)], dtype=np.int64)
        idx = {int(s): i for i, s in enumerate(st.tolist())}
        pion_aligned[int(p)] = np.asarray([pion_samples_by_p[int(p)][idx[int(s)]] for s in steps], dtype=np.float64)
    return pion_aligned, pipi_aligned, steps


def _apply_common_discard_stride(
    pion_samples_by_p: Mapping[int, np.ndarray],
    pipi_samples: np.ndarray,
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
    pipi_out = np.asarray(pipi_samples[keep], dtype=np.complex128)
    steps_out = np.asarray(steps[keep], dtype=np.int64)
    return pion_out, pipi_out, steps_out


def _choose_joint_block_size(
    pion_samples_by_p: Mapping[int, np.ndarray],
    pipi_samples: np.ndarray,
    *,
    t0: int,
    iat_method: str,
    iat_c: float,
    iat_max_lag: int | None,
) -> Tuple[int, Dict[str, float]]:
    block_pi, tau_pi = FIT2PT._auto_block_size_from_p0(
        pion_samples_by_p,
        iat_method=str(iat_method),
        iat_c=float(iat_c),
        iat_max_lag=iat_max_lag,
    )
    block_pp, info_pp = PIPI._choose_block_size(
        np.asarray(pipi_samples, dtype=np.complex128),
        t_ref=max(1, int(t0)),
        method=str(iat_method),
    )
    block = max(int(block_pi), int(block_pp))
    n = int(next(iter(pion_samples_by_p.values())).shape[0])
    while block > 1 and (n // block) < 2:
        block -= 1
    return int(max(1, block)), {
        "pion_tau_ref": float(tau_pi),
        "pion_block_auto": float(block_pi),
        "pipi_tau_ref": float(info_pp.get("tau_int", float("nan"))),
        "pipi_block_auto": float(block_pp),
    }


def _jk_pion_window_energies(
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
) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    nb = int(b.shape[0])
    out = np.full((nb,), np.nan, dtype=np.float64)
    for ib in range(nb):
        loo = np.delete(b, ib, axis=0)
        fit = FIT2PT._fit_window_correlated(
            loo,
            support_times=np.asarray(support_times, dtype=np.int64),
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
        out[ib] = float(fit["energy"])
    return out


def _mean_over_window(arr: np.ndarray, fit_range: Tuple[int, int]) -> np.ndarray:
    a, b = int(fit_range[0]), int(fit_range[1])
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        seg = x[a : b + 1]
        seg = seg[np.isfinite(seg)]
        return np.asarray(float(np.mean(seg)) if seg.size else float("nan"), dtype=np.float64)
    out = np.full(x.shape[:-1], np.nan, dtype=np.float64)
    for idx in np.ndindex(*x.shape[:-1]):
        seg = np.asarray(x[idx + (slice(a, b + 1),)], dtype=np.float64)
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[idx] = float(np.mean(seg))
    return out


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
            f"Requested pion fit range [{int(tmin)},{int(tmax)}] is not supported by available times {support.tolist()}"
        )
    i0 = int(idx_map[int(tmin)])
    i1 = int(idx_map[int(tmax)])
    if i1 < i0:
        i0, i1 = i1, i0
    return int(i0), int(i1)


def _phat_sq_1d(p: int, lx: int) -> float:
    return float((2.0 * math.sin(math.pi * float(abs(int(p))) / float(lx))) ** 2)


def _setup_matplotlib(no_gui: bool):
    return FIT2PT._setup_matplotlib(bool(no_gui))


def main() -> int:
    ap = argparse.ArgumentParser(description="Combined Schwinger single-pion + two-pion + DeltaE analysis")
    ap.add_argument("--input", required=True, help="checkpoint .pkl from scripts/mcmc/mcmc.py")
    ap.add_argument("--pion-measurement", default="pion_2pt")
    ap.add_argument("--pion-channel", default="c")
    ap.add_argument("--pipi-measurement", default="pipi_i2_matrix")
    ap.add_argument("--pipi-channel", default="full", choices=("full", "direct", "exchange"))
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--iat-method", default="ips", choices=("ips", "sokal", "gamma"))
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0, help="<=0 chooses a common block size automatically")
    ap.add_argument("--levels", type=int, default=4, help="number of two-pion levels to analyze")
    ap.add_argument("--t0", type=int, default=1, help="GEVP reference time")
    ap.add_argument("--pipi-fit-range", type=str, default="", help="effective-energy window for two-pion levels, e.g. 2,5")
    ap.add_argument("--pion-tmin-min", type=int, default=1)
    ap.add_argument("--pion-tmax-max", type=int, default=-1)
    ap.add_argument("--pion-min-points", type=int, default=4)
    ap.add_argument("--pion-fit-range", type=str, default="", help="fixed pion fit window tmin,tmax; default is automatic")
    ap.add_argument("--pion-chi2-min", type=float, default=0.4)
    ap.add_argument("--pion-chi2-max", type=float, default=2.0)
    ap.add_argument("--pion-score-window-penalty", type=float, default=0.2)
    ap.add_argument("--pion-fallback-two-exp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pion-two-exp-min-points", type=int, default=6)
    ap.add_argument("--pion-e-min", type=float, default=1e-4)
    ap.add_argument("--pion-e-max", type=float, default=4.0)
    ap.add_argument("--pion-e-grid", type=int, default=256)
    ap.add_argument("--cov-reg", type=float, default=1e-10)
    ap.add_argument("--svd-cut", type=float, default=1.0e-10)
    ap.add_argument("--target-sigma", type=float, default=3.0, help="significance target for projected stats requirement")
    ap.add_argument("--max-p-fit", type=int, default=-1, help="maximum |p| included in the single-pion dispersion fit")
    ap.add_argument("--prefix", type=str, default="schwinger_spectrum")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {ckpt_path}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (ckpt_path.parent / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{args.prefix}.png"
    json_path = outdir / f"{args.prefix}.json"

    payload, records = FIT2PT._load_checkpoint(ckpt_path)
    if not records:
        raise ValueError(f"No inline_records found in checkpoint: {ckpt_path}")

    pion_samples_by_p_raw, pion_steps_by_p_raw, pion_meta_by_p = FIT2PT._extract_by_momentum(
        records,
        measurement=str(args.pion_measurement),
        channel=str(args.pion_channel),
    )
    basis_momenta, pipi_samples_raw, pipi_steps_raw = PIPI._extract_samples(
        records,
        measurement_name=str(args.pipi_measurement),
        channel=str(args.pipi_channel),
    )
    pipi_samples_raw = PIPI._hermitize_samples(pipi_samples_raw)

    required_momenta = [abs(int(p)) for p in basis_momenta.tolist()]
    pion_aligned, pipi_aligned, common_steps = _align_common_steps(
        pion_samples_by_p_raw,
        pion_steps_by_p_raw,
        required_momenta=required_momenta,
        pipi_samples=pipi_samples_raw,
        pipi_steps=pipi_steps_raw,
    )
    pion_aligned, pipi_aligned, common_steps = _apply_common_discard_stride(
        pion_aligned,
        pipi_aligned,
        common_steps,
        discard=int(args.discard),
        stride=int(args.stride),
    )

    iat_max_lag = None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)
    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        block_info = {"mode": "manual"}
    else:
        block_size, block_info = _choose_joint_block_size(
            pion_aligned,
            pipi_aligned,
            t0=int(args.t0),
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=iat_max_lag,
        )

    pion_blocks = {int(p): FIT2PT._block_means(arr, block_size=block_size) for p, arr in pion_aligned.items()}
    pipi_blocks = _block_means_nd(pipi_aligned, block_size=block_size)
    pipi_jk_corr = _jackknife_loo_from_blocks(pipi_blocks)
    pipi_mean_corr = np.mean(pipi_blocks, axis=0)
    n_blocks = int(pipi_blocks.shape[0])
    n_used = int(pipi_blocks.shape[0] * block_size)

    t_extent = FIT2PT._temporal_extent_from_payload(payload, pion_meta_by_p)
    lx = int(payload.get("config", {}).get("run", {}).get("shape", [len(basis_momenta), t_extent])[0]) if isinstance(payload.get("config", {}), Mapping) else int(len(basis_momenta))
    if int(args.pion_tmax_max) > 0:
        pion_tmax_max = int(args.pion_tmax_max)
    else:
        pion_tmax_max = max(1, int(t_extent // 2 - 1))

    pion_fit_results: List[Dict[str, float]] = []
    pion_energy_full_by_p: Dict[int, float] = {}
    pion_energy_jk_by_p: Dict[int, np.ndarray] = {}
    for p in sorted(pion_aligned.keys()):
        blocks = np.asarray(pion_blocks[p], dtype=np.float64)
        support_times = np.asarray(pion_meta_by_p[p].get("support_times", list(range(blocks.shape[1]))), dtype=np.int64)
        if args.pion_fit_range.strip():
            tmin_fix, tmax_fix = _parse_fit_range_text(args.pion_fit_range)
            idx_min_fix, idx_max_fix = _support_index_range(support_times, tmin_fix, tmax_fix)
            best = FIT2PT._fit_window_correlated(
                blocks,
                support_times=support_times,
                idx_min=int(idx_min_fix),
                idx_max=int(idx_max_fix),
                t_extent=int(t_extent),
                n_exp=1,
                e_min=float(args.pion_e_min),
                e_max=float(args.pion_e_max),
                e_grid=int(args.pion_e_grid),
                cov_reg=float(args.cov_reg),
            )
            if best is None:
                raise ValueError(f"Fixed pion fit range [{tmin_fix},{tmax_fix}] failed for p={int(p)}")
            best = dict(best)
            best["score"] = float("nan")
            best["in_chi2_band"] = 1.0
            candidates = [best]
        else:
            best, candidates = FIT2PT._select_window(
                blocks,
                support_times=support_times,
                t_extent=int(t_extent),
                tmin_min=int(args.pion_tmin_min),
                tmax_max=int(pion_tmax_max),
                min_points=int(args.pion_min_points),
                chi2_min=float(args.pion_chi2_min),
                chi2_max=float(args.pion_chi2_max),
                score_window_penalty=float(args.pion_score_window_penalty),
                fallback_two_exp=bool(args.pion_fallback_two_exp),
                two_exp_min_points=int(args.pion_two_exp_min_points),
                progress=False,
                progress_every=999999,
                progress_prefix=f"p={int(p)}",
                e_min=float(args.pion_e_min),
                e_max=float(args.pion_e_max),
                e_grid=int(args.pion_e_grid),
                cov_reg=float(args.cov_reg),
            )
            if best is None:
                tmin_fix = max(int(args.t0) + 1, int(args.pion_tmin_min))
                tmax_fix = min(int(pion_tmax_max), int(tmin_fix + max(3, int(args.pion_min_points) - 1)))
                idx_min_fix, idx_max_fix = _support_index_range(support_times, tmin_fix, tmax_fix)
                best = FIT2PT._fit_window_correlated(
                    blocks,
                    support_times=support_times,
                    idx_min=int(idx_min_fix),
                    idx_max=int(idx_max_fix),
                    t_extent=int(t_extent),
                    n_exp=1,
                    e_min=float(args.pion_e_min),
                    e_max=float(args.pion_e_max),
                    e_grid=int(args.pion_e_grid),
                    cov_reg=float(args.cov_reg),
                )
                if best is None:
                    raise ValueError(f"No valid correlated pion fit found for p={int(p)}")
                best = dict(best)
                best["score"] = float("nan")
                best["in_chi2_band"] = 0.0
                best["selected_manual_fallback"] = 1.0
                candidates = [best]
        idx_min = int(best["support_idx_min"])
        idx_max = int(best["support_idx_max"])
        n_exp = int(best.get("n_exp", 1.0))
        jk_en = _jk_pion_window_energies(
            blocks,
            support_times=support_times,
            idx_min=idx_min,
            idx_max=idx_max,
            t_extent=int(t_extent),
            n_exp=int(n_exp),
            e_min=float(args.pion_e_min),
            e_max=float(args.pion_e_max),
            e_grid=int(args.pion_e_grid),
            cov_reg=float(args.cov_reg),
        )
        err_map = FIT2PT._jackknife_fit_errors(
            blocks,
            support_times=support_times,
            best_idx_min=idx_min,
            best_idx_max=idx_max,
            t_extent=int(t_extent),
            n_exp=int(n_exp),
            e_min=float(args.pion_e_min),
            e_max=float(args.pion_e_max),
            e_grid=int(args.pion_e_grid),
            cov_reg=float(args.cov_reg),
        )
        row = {
            "p": float(int(p)),
            "abs_p": float(abs(int(p))),
            "energy": float(best["energy"]),
            "energy_err": float(err_map.get("energy_err", float("nan"))),
            "fit_model": str(best.get("fit_model", "1exp")),
            "tmin": float(best["tmin"]),
            "tmax": float(best["tmax"]),
            "chi2_dof": float(best["chi2_dof"]),
        }
        if n_exp == 2:
            row["energy_1"] = float(best.get("energy_1", float("nan")))
            row["energy_1_err"] = float(err_map.get("energy_1_err", float("nan")))
        pion_fit_results.append(row)
        pion_energy_full_by_p[int(abs(int(p)))] = float(best["energy"])
        pion_energy_jk_by_p[int(abs(int(p)))] = np.asarray(jk_en, dtype=np.float64)

    dispersion_pts: List[Tuple[float, float, float, int]] = []
    for row in pion_fit_results:
        p = int(row["p"])
        if int(args.max_p_fit) >= 0 and abs(p) > int(args.max_p_fit):
            continue
        dispersion_pts.append((_phat_sq_1d(p, lx), float(row["energy"]) ** 2, 2.0 * abs(float(row["energy"])) * float(row["energy_err"]), p))
    if len(dispersion_pts) >= 2:
        xx = np.asarray([v[0] for v in dispersion_pts], dtype=np.float64)
        yy = np.asarray([v[1] for v in dispersion_pts], dtype=np.float64)
        ee = np.asarray([max(v[2], 1e-12) for v in dispersion_pts], dtype=np.float64)
        dispersion_fit = FIT2PT._weighted_line_fit(xx, yy, ee)
        dispersion_mc = FIT2PT._derive_m_c_from_dispersion(dispersion_fit)
    else:
        dispersion_fit = {"ok": False}
        dispersion_mc = {"m": float("nan"), "m_err": float("nan"), "c": float("nan"), "c_err": float("nan")}

    levels_req = min(int(args.levels), int(len(basis_momenta)))
    gevp_full = PIPI._solve_gevp(
        pipi_mean_corr,
        t0=int(args.t0),
        n_levels=int(levels_req),
        svd_cut=float(args.svd_cut),
    )
    lambdas_full = np.asarray(gevp_full["lambdas"], dtype=np.float64)
    lambdas_jk = np.asarray(
        [
            PIPI._solve_gevp(
                pipi_jk_corr[ib],
                t0=int(args.t0),
                n_levels=int(levels_req),
                svd_cut=float(args.svd_cut),
            )["lambdas"]
            for ib in range(n_blocks)
        ],
        dtype=np.float64,
    )
    lambda_err = _jackknife_err_from_replicas(lambdas_jk, lambdas_full)
    eff_full = PIPI._effective_energies(lambdas_full)
    eff_jk = np.asarray([PIPI._effective_energies(lambdas_jk[ib]) for ib in range(n_blocks)], dtype=np.float64)
    eff_err = _jackknife_err_from_replicas(eff_jk, eff_full)

    if args.pipi_fit_range.strip():
        pipi_fit_range = PIPI._parse_fit_range(args.pipi_fit_range, tmin=0, tmax=max(0, eff_full.shape[1] - 1))
    else:
        tmin = min(max(0, int(args.t0) + 1), max(0, eff_full.shape[1] - 1))
        tmax = min(max(tmin, int(args.t0) + 4), max(0, eff_full.shape[1] - 1), max(0, t_extent // 2 - 2))
        pipi_fit_range = (int(tmin), int(tmax))
    w_full = np.asarray(_mean_over_window(eff_full, pipi_fit_range), dtype=np.float64).reshape(-1)
    w_jk = np.asarray(_mean_over_window(eff_jk, pipi_fit_range), dtype=np.float64)
    w_err = _jackknife_err_from_replicas(w_jk, w_full)

    basis_abs = np.asarray([abs(int(p)) for p in basis_momenta.tolist()], dtype=np.int64)
    free_full = np.full((levels_req,), np.nan, dtype=np.float64)
    free_jk = np.full((n_blocks, levels_req), np.nan, dtype=np.float64)
    for lvl in range(levels_req):
        p = int(basis_abs[lvl])
        if p not in pion_energy_full_by_p:
            raise ValueError(f"Basis momentum p={p} needed for free level {lvl} is missing from pion fits")
        free_full[lvl] = 2.0 * float(pion_energy_full_by_p[p])
        free_jk[:, lvl] = 2.0 * np.asarray(pion_energy_jk_by_p[p], dtype=np.float64)

    delta_full = np.asarray(w_full[:levels_req] - free_full[:levels_req], dtype=np.float64)
    delta_jk = np.asarray(w_jk[:, :levels_req] - free_jk[:, :levels_req], dtype=np.float64)
    delta_err = _jackknife_err_from_replicas(delta_jk, delta_full)
    significance = np.full((levels_req,), np.nan, dtype=np.float64)
    valid_sig = np.isfinite(delta_err) & (delta_err > 0.0)
    significance[valid_sig] = np.abs(delta_full[valid_sig]) / delta_err[valid_sig]

    need_factor = np.full((levels_req,), np.nan, dtype=np.float64)
    need_measure = np.full((levels_req,), np.nan, dtype=np.float64)
    target_sigma = float(args.target_sigma)
    for lvl in range(levels_req):
        sig = float(significance[lvl])
        if np.isfinite(sig) and sig > 0.0:
            fac = max(0.0, (target_sigma / sig) ** 2)
            need_factor[lvl] = float(fac)
            need_measure[lvl] = float(n_used * fac)

    summary_rows: List[Dict[str, float]] = []
    for lvl in range(levels_req):
        summary_rows.append(
            {
                "level": float(lvl),
                "basis_momentum": float(int(basis_abs[lvl])),
                "energy_interacting": float(w_full[lvl]),
                "energy_interacting_err": float(w_err[lvl]),
                "energy_free": float(free_full[lvl]),
                "energy_shift": float(delta_full[lvl]),
                "energy_shift_err": float(delta_err[lvl]),
                "significance_sigma": float(significance[lvl]),
                "projected_measurements_for_target_sigma": float(need_measure[lvl]),
                "projected_measurement_factor": float(need_factor[lvl]),
            }
        )

    plt = _setup_matplotlib(bool(args.no_gui))
    if plt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0), constrained_layout=True)
        ax0, ax1, ax2 = axes

        pvals = np.asarray([int(r["p"]) for r in pion_fit_results], dtype=np.int64)
        evals = np.asarray([float(r["energy"]) for r in pion_fit_results], dtype=np.float64)
        eerrs = np.asarray([float(r["energy_err"]) for r in pion_fit_results], dtype=np.float64)
        ax0.errorbar(pvals, evals, yerr=eerrs, fmt="o", capsize=2.5, label=r"$E_\pi(p)$")
        if bool(dispersion_fit.get("ok", False)):
            xx = np.linspace(0.0, float(np.max(np.abs(pvals))) + 0.25, 200)
            ph2 = np.asarray([_phat_sq_1d(int(round(x)), lx) for x in xx], dtype=np.float64)
            e2 = float(dispersion_fit["intercept"]) + float(dispersion_fit["slope"]) * ph2
            good = np.isfinite(e2) & (e2 > 0.0)
            ax0.plot(xx[good], np.sqrt(e2[good]), "-", lw=1.2, label="dispersion fit")
        ax0.set_xlabel("p")
        ax0.set_ylabel(r"$E_\pi$")
        ax0.set_title("Single-pion spectrum")
        ax0.legend(loc="best", fontsize=9)
        if np.isfinite(float(dispersion_mc.get("m", float("nan")))) or np.isfinite(float(dispersion_mc.get("c", float("nan")))):
            ax0.text(
                0.02,
                0.98,
                _fmt_pm_text(
                    "m_pi",
                    float(dispersion_mc.get("m", float("nan"))),
                    float(dispersion_mc.get("m_err", float("nan"))),
                    value_fmt=".5f",
                    err_fmt=".3g",
                )
                + "\n"
                + _fmt_pm_text(
                    "c",
                    float(dispersion_mc.get("c", float("nan"))),
                    float(dispersion_mc.get("c_err", float("nan"))),
                    value_fmt=".5f",
                    err_fmt=".3g",
                ),
                transform=ax0.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"),
            )

        lvl_idx = np.arange(levels_req, dtype=np.int64)
        ax1.errorbar(lvl_idx - 0.05, w_full[:levels_req], yerr=w_err[:levels_req], fmt="o", capsize=2.5, label="interacting")
        ax1.errorbar(lvl_idx + 0.05, free_full[:levels_req], yerr=np.zeros_like(free_full[:levels_req]), fmt="s", capsize=2.5, label="free")
        ax1.set_xlabel("level")
        ax1.set_ylabel("energy")
        ax1.set_title("Two-pion spectrum: interacting vs free")
        ax1.legend(loc="best", fontsize=9)

        ax2.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.6)
        ax2.errorbar(lvl_idx, delta_full[:levels_req], yerr=delta_err[:levels_req], fmt="o", capsize=2.5)
        for lvl in range(levels_req):
            sig = float(significance[lvl])
            if np.isfinite(sig):
                ax2.text(
                    float(lvl),
                    float(delta_full[lvl]),
                    f" {sig:.2f}σ",
                    fontsize=9,
                    va="bottom" if float(delta_full[lvl]) >= 0.0 else "top",
                    ha="left",
                )
        ax2.set_xlabel("level")
        ax2.set_ylabel(r"$\Delta E = W - E_{\mathrm{free}}$")
        ax2.set_title("Correlated two-pion energy shifts")

        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    result = {
        "input": str(ckpt_path),
        "common_steps_used": [int(v) for v in common_steps[:n_used].tolist()],
        "n_measurements_used": int(n_used),
        "block_size": int(block_size),
        "n_blocks": int(n_blocks),
        "block_info": block_info,
        "basis_momenta": [int(v) for v in basis_momenta.tolist()],
        "single_particle": {
            "fits": pion_fit_results,
            "dispersion_fit": dispersion_fit,
            "derived_mc": dispersion_mc,
        },
        "two_particle": {
            "t0": int(args.t0),
            "fit_range": [int(pipi_fit_range[0]), int(pipi_fit_range[1])],
            "principal_correlators": {
                "mean": np.asarray(lambdas_full, dtype=np.float64).tolist(),
                "err": np.asarray(lambda_err, dtype=np.float64).tolist(),
            },
            "effective_energies": {
                "mean": np.asarray(eff_full, dtype=np.float64).tolist(),
                "err": np.asarray(eff_err, dtype=np.float64).tolist(),
            },
            "energies": [
                {
                    "level": float(lvl),
                    "energy": float(w_full[lvl]),
                    "energy_err": float(w_err[lvl]),
                }
                for lvl in range(levels_req)
            ],
        },
        "energy_shifts": summary_rows,
        "target_sigma": float(target_sigma),
        "plot": str(png_path),
    }
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Loaded {int(common_steps.size)} common pion/pipi samples from {ckpt_path}")
    print(f"Using block_size={int(block_size)}, n_blocks={int(n_blocks)}, n_used={int(n_used)}")
    if block_info:
        print("Block-size diagnostics:")
        for k, v in block_info.items():
            print(f"  {k}: {v}")
    print("Single-pion energies:")
    for row in pion_fit_results:
        print(
            f"  p={int(row['p'])}: E={row['energy']:.8g} +/- {row['energy_err']:.3g}"
            f" [{row['fit_model']}, t={int(row['tmin'])}..{int(row['tmax'])}, chi2/dof={row['chi2_dof']:.3f}]"
        )
    print("Two-pion interacting vs free energy shifts:")
    for row in summary_rows:
        lvl = int(row["level"])
        print(
            f"  level {lvl}: W={row['energy_interacting']:.8g} +/- {row['energy_interacting_err']:.3g},"
            f" Efree={row['energy_free']:.8g},"
            f" DeltaE={row['energy_shift']:.8g} +/- {row['energy_shift_err']:.3g},"
            f" sig={row['significance_sigma']:.3g} sigma"
        )
        if np.isfinite(float(row["projected_measurements_for_target_sigma"])):
            print(
                f"    projected measurements for {target_sigma:.1f} sigma:"
                f" ~{row['projected_measurements_for_target_sigma']:.1f}"
            )
    print(f"Wrote {json_path}")
    if plt is not None:
        print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

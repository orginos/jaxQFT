#!/usr/bin/env python3
"""Scan the GEVP reference time t0 for two-pion I=2 spectroscopy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


PIPI = _load_script_module("analyze_pipi_i2_gevp_mod", ROOT / "scripts" / "mcmc" / "analyze_pipi_i2_gevp.py")


def _parse_int_list(text: str) -> Tuple[int, ...]:
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    if not toks:
        raise ValueError("expected a non-empty comma-separated integer list")
    return tuple(int(t) for t in toks)


def _default_fit_range_for_t0(t0: int, lt: int, eff_nt: int) -> Tuple[int, int]:
    tmin = min(max(0, int(t0) + 1), max(0, int(eff_nt) - 1))
    tmax = min(max(tmin, int(t0) + 4), max(0, int(eff_nt) - 1), max(0, int(lt) // 2 - 2))
    return int(tmin), int(tmax)


def _choose_common_fit_range(t0_list: Sequence[int], lt: int, eff_nt: int) -> Tuple[int, int]:
    defs = [_default_fit_range_for_t0(int(t0), int(lt), int(eff_nt)) for t0 in t0_list]
    a = max(int(x[0]) for x in defs)
    b = min(int(x[1]) for x in defs)
    if b < a:
        return defs[0]
    return int(a), int(b)


def _scan_one_t0(
    *,
    mean_corr: np.ndarray,
    jk_corr: np.ndarray,
    nblock: int,
    lt: int,
    t0: int,
    levels: int,
    svd_cut: float,
    fit_range: Tuple[int, int],
) -> Dict[str, object]:
    gevp_full = PIPI._solve_gevp(mean_corr, t0=int(t0), n_levels=int(levels), svd_cut=float(svd_cut))
    lambdas = np.asarray(gevp_full["lambdas"], dtype=np.float64)
    jk_lambdas = np.zeros((int(nblock), *lambdas.shape), dtype=np.float64)
    failures = 0
    for ib in range(int(nblock)):
        try:
            jk_lambdas[ib] = PIPI._solve_gevp(
                jk_corr[ib],
                t0=int(t0),
                n_levels=int(levels),
                svd_cut=float(svd_cut),
            )["lambdas"]
        except Exception:
            failures += 1
            jk_lambdas[ib] = np.nan

    lambda_err = PIPI._jackknife_error(jk_lambdas, lambdas)
    eff = PIPI._effective_energies(lambdas)
    eff_jk = np.asarray([PIPI._effective_energies(jk_lambdas[ib]) for ib in range(int(nblock))], dtype=np.float64)
    eff_err = PIPI._jackknife_error(eff_jk, eff)
    energies = PIPI._estimate_level_energies(eff, eff_jk, fit_range=fit_range)
    return {
        "t0": int(t0),
        "fit_range": [int(fit_range[0]), int(fit_range[1])],
        "kept_metric_modes": np.asarray(gevp_full["kept_metric_modes"]).astype(int).tolist(),
        "metric_evals": np.asarray(gevp_full["metric_evals"], dtype=np.float64).tolist(),
        "n_metric_failures_jk": int(failures),
        "principal_correlators": {
            "mean": np.asarray(lambdas, dtype=np.float64).tolist(),
            "err": np.asarray(lambda_err, dtype=np.float64).tolist(),
        },
        "effective_energies": {
            "mean": np.asarray(eff, dtype=np.float64).tolist(),
            "err": np.asarray(eff_err, dtype=np.float64).tolist(),
        },
        "energies": energies,
    }


def _level_energy_vector(entry: Dict[str, object], nlevels: int) -> Tuple[np.ndarray, np.ndarray]:
    en = np.full((int(nlevels),), np.nan, dtype=np.float64)
    er = np.full((int(nlevels),), np.nan, dtype=np.float64)
    for row in entry.get("energies", []):
        lvl = int(row.get("level", -1))
        if 0 <= lvl < int(nlevels):
            en[lvl] = float(row.get("energy", np.nan))
            er[lvl] = float(row.get("energy_err", np.nan))
    return en, er


def _recommend_t0(
    scans: Sequence[Dict[str, object]],
    *,
    stable_levels: int,
    stability_nsigma: float,
) -> Dict[str, object]:
    if len(scans) < 2:
        return {
            "recommended_t0": None,
            "reason": "need at least two t0 values to assess stability",
            "pairwise": [],
        }
    nlevels = max(1, int(stable_levels))
    pairwise: List[Dict[str, object]] = []
    recommended = None
    for left, right in zip(scans[:-1], scans[1:]):
        t0a = int(left["t0"])
        t0b = int(right["t0"])
        ea, da = _level_energy_vector(left, nlevels)
        eb, db = _level_energy_vector(right, nlevels)
        level_rows = []
        ok = True
        for lvl in range(nlevels):
            sigma = float("nan")
            if np.isfinite(ea[lvl]) and np.isfinite(eb[lvl]) and np.isfinite(da[lvl]) and np.isfinite(db[lvl]):
                den = math.sqrt(max(0.0, float(da[lvl]) ** 2 + float(db[lvl]) ** 2))
                sigma = float(abs(float(ea[lvl]) - float(eb[lvl])) / den) if den > 0.0 else float("inf")
            level_ok = bool(np.isfinite(sigma) and sigma <= float(stability_nsigma))
            level_rows.append(
                {
                    "level": int(lvl),
                    "energy_t0": float(ea[lvl]),
                    "energy_t0p1": float(eb[lvl]),
                    "sigma_shift": float(sigma),
                    "stable": bool(level_ok),
                }
            )
            ok = ok and level_ok
        pairwise.append(
            {
                "t0": int(t0a),
                "t0_next": int(t0b),
                "stable_levels": level_rows,
                "all_levels_stable": bool(ok),
            }
        )
        if recommended is None and ok:
            recommended = int(t0a)
    return {
        "recommended_t0": recommended,
        "reason": (
            f"earliest t0 whose first {int(stable_levels)} levels are stable within {float(stability_nsigma):.2f} sigma "
            "when compared to the next t0"
            if recommended is not None
            else "no adjacent t0 pair satisfied the requested stability criterion"
        ),
        "pairwise": pairwise,
    }


def _plot_t0_scan(
    *,
    scans: Sequence[Dict[str, object]],
    recommendation: Dict[str, object],
    basis: np.ndarray,
    out_path: Path,
    title_prefix: str,
    levels: int,
) -> None:
    import matplotlib.pyplot as plt

    t0s = np.asarray([int(s["t0"]) for s in scans], dtype=np.int64)
    nlev = int(levels)
    e = np.full((nlev, len(scans)), np.nan, dtype=np.float64)
    de = np.full_like(e, np.nan)
    nkeep = np.full((len(scans),), np.nan, dtype=np.float64)
    for it, scan in enumerate(scans):
        ev, dv = _level_energy_vector(scan, nlev)
        e[:, it] = ev
        de[:, it] = dv
        nkeep[it] = float(len(scan.get("kept_metric_modes", [])))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), constrained_layout=True)
    ax0, ax1 = axes
    for lvl in range(nlev):
        ax0.errorbar(
            t0s,
            e[lvl],
            yerr=de[lvl],
            fmt="o-",
            ms=4.0,
            capsize=2.0,
            label=f"level {lvl}",
        )
    rec = recommendation.get("recommended_t0", None)
    if rec is not None:
        ax0.axvline(int(rec), color="k", ls="--", lw=1.0, alpha=0.6, label=f"recommended t0={int(rec)}")
    ax0.set_xlabel(r"$t_0$")
    ax0.set_ylabel("E")
    ax0.set_title(f"{title_prefix}: t0 scan (basis {basis.astype(int).tolist()})")
    ax0.legend(loc="best", fontsize=9)

    ax1.plot(t0s, nkeep, "o-", label="kept metric modes")
    max_sigma = []
    for row in recommendation.get("pairwise", []):
        sigmas = [float(x.get("sigma_shift", np.nan)) for x in row.get("stable_levels", [])]
        finite = [s for s in sigmas if np.isfinite(s)]
        max_sigma.append(max(finite) if finite else np.nan)
    if max_sigma:
        ax1b = ax1.twinx()
        ax1b.plot(t0s[:-1], np.asarray(max_sigma, dtype=np.float64), "s--", color="tab:red", label="max level shift sigma")
        ax1b.set_ylabel("max shift / sigma", color="tab:red")
        ax1b.tick_params(axis="y", colors="tab:red")
    ax1.set_xlabel(r"$t_0$")
    ax1.set_ylabel("n kept modes")
    ax1.set_title("Metric rank / adjacent-t0 stability")
    ax1.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan GEVP t0 for two-pion I=2 correlator matrices")
    ap.add_argument("--input", required=True, help="checkpoint .pkl produced by scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", default="pipi_i2_matrix", help="inline measurement name")
    ap.add_argument("--channel", default="full", choices=("full", "direct", "exchange"), help="correlator-matrix channel")
    ap.add_argument("--t0-list", type=str, default="1,2,3,4", help="comma-separated t0 values to scan")
    ap.add_argument("--levels", type=int, default=4, help="number of levels to retain")
    ap.add_argument("--svd-cut", type=float, default=1.0e-10, help="relative cutoff for C(t0) metric eigenmodes")
    ap.add_argument("--fit-range", type=str, default="", help="common effective-energy fit window 'tmin,tmax'; default chooses the overlap of per-t0 defaults")
    ap.add_argument("--block-size", type=int, default=0, help="jackknife block size; <=0 selects 2*tau_int automatically")
    ap.add_argument("--iat-method", type=str, default="ips", choices=("ips", "sokal", "gamma"), help="IAT estimator for automatic blocking")
    ap.add_argument("--no-hermitize", action="store_true", help="skip C -> (C + C^dagger)/2 symmetrization per sample")
    ap.add_argument("--stable-levels", type=int, default=3, help="number of low-lying levels used for the t0 stability recommendation")
    ap.add_argument("--stability-nsigma", type=float, default=2.0, help="adjacent-t0 stability threshold in sigma")
    ap.add_argument("--outdir", type=str, default="", help="output directory (default: alongside checkpoint)")
    ap.add_argument("--prefix", type=str, default="", help="output filename prefix")
    ap.add_argument("--save", type=str, default="", help="explicit PNG output path")
    ap.add_argument("--json-out", type=str, default="", help="explicit JSON output path")
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else ckpt_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip() or f"{ckpt_path.stem}_{args.measurement}_{args.channel}_t0scan"
    png_path = Path(args.save).expanduser().resolve() if args.save else (outdir / f"{prefix}.png")
    json_path = Path(args.json_out).expanduser().resolve() if args.json_out else (outdir / f"{prefix}.json")

    t0_list = tuple(sorted(set(int(x) for x in _parse_int_list(args.t0_list))))
    records = PIPI._load_inline_records(ckpt_path)
    basis, samples, steps = PIPI._extract_samples(records, measurement_name=args.measurement, channel=args.channel)
    if not args.no_hermitize:
        samples = PIPI._hermitize_samples(samples)

    lt = int(samples.shape[-1])
    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        block_auto_info: Dict[str, float] = {}
    else:
        block_size, block_auto_info = PIPI._choose_block_size(samples, t_ref=max(1, min(t0_list)), method=str(args.iat_method))
    mean_corr, jk_corr, nblock, ntrim = PIPI._blocked_jackknife(samples, block_size)
    eff_nt = max(0, int(mean_corr.shape[-1]) - 1)
    if args.fit_range.strip():
        fit_range = PIPI._parse_fit_range(args.fit_range, tmin=0, tmax=max(0, eff_nt - 1))
        fit_range_mode = "manual"
    else:
        fit_range = _choose_common_fit_range(t0_list, lt, eff_nt)
        fit_range_mode = "common-default-overlap"

    scans: List[Dict[str, object]] = []
    for t0 in t0_list:
        print(f"[t0={int(t0)}] solving GEVP", flush=True)
        scan = _scan_one_t0(
            mean_corr=mean_corr,
            jk_corr=jk_corr,
            nblock=int(nblock),
            lt=int(lt),
            t0=int(t0),
            levels=int(args.levels),
            svd_cut=float(args.svd_cut),
            fit_range=fit_range,
        )
        scans.append(scan)
        en = scan["energies"]
        txt = ", ".join(
            f"l{int(row['level'])}={float(row['energy']):.6g}+/-{float(row['energy_err']):.3g}"
            for row in en[: min(3, len(en))]
        )
        print(f"[t0={int(t0)}] {txt}", flush=True)

    recommendation = _recommend_t0(
        scans,
        stable_levels=int(args.stable_levels),
        stability_nsigma=float(args.stability_nsigma),
    )
    _plot_t0_scan(
        scans=scans,
        recommendation=recommendation,
        basis=basis,
        out_path=png_path,
        title_prefix=f"{args.measurement} [{args.channel}]",
        levels=int(args.levels),
    )

    result = {
        "input": str(ckpt_path),
        "measurement": str(args.measurement),
        "channel": str(args.channel),
        "basis_momenta": basis.astype(int).tolist(),
        "steps_used": [int(s) for s in steps[:ntrim]],
        "n_measurements_total": int(samples.shape[0]),
        "n_measurements_used": int(ntrim),
        "block_size": int(block_size),
        "n_blocks": int(nblock),
        "auto_block_info": block_auto_info,
        "t0_list": [int(x) for x in t0_list],
        "fit_range": [int(fit_range[0]), int(fit_range[1])],
        "fit_range_mode": str(fit_range_mode),
        "svd_cut": float(args.svd_cut),
        "stable_levels": int(args.stable_levels),
        "stability_nsigma": float(args.stability_nsigma),
        "scans": scans,
        "recommendation": recommendation,
        "plot": str(png_path),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Basis momenta: {basis.astype(int).tolist()}", flush=True)
    print(f"Using block_size={block_size}, n_blocks={nblock}, fit_range={fit_range} ({fit_range_mode})", flush=True)
    rec = recommendation.get("recommended_t0", None)
    if rec is None:
        print("Recommended t0: none (no adjacent pair met the requested stability criterion)", flush=True)
    else:
        print(f"Recommended t0: {int(rec)}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()

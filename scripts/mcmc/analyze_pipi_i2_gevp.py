#!/usr/bin/env python3
"""Analyze DD-independent two-pion I=2 correlator matrices from mcmc checkpoints.

This script expects inline records produced by the `pipi_i2_matrix` measurement
in `jaxqft.core.measurements`. It:

- extracts the correlator matrix samples C_ij(t),
- estimates a blocking size from the IAT of C_00(t_ref) unless overridden,
- forms blocked jackknife samples,
- solves the GEVP C(t) v = lambda(t,t0) C(t0) v,
- builds principal correlators and effective energies,
- estimates level energies from a fit window on the log-ratio effective energy,
- writes a PNG summary figure and JSON payload.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.stats import integrated_autocorr_time


_MAT_KEY_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9_]+)_i(?P<i>\d+)_j(?P<j>\d+)_t(?P<t>\d+)_(?P<part>re|im)$")
_BASIS_KEY_RE = re.compile(r"^basis_p(?P<i>\d+)$")


def _parse_int_list(text: str) -> Tuple[int, ...]:
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    return tuple(int(t) for t in toks)


def _load_inline_records(path: Path) -> List[Dict]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    state = payload.get("state", {})
    records = state.get("inline_records", [])
    if not isinstance(records, list):
        raise ValueError("Checkpoint state.inline_records is not a list")
    return list(records)


def _extract_samples(
    records: Sequence[Dict],
    *,
    measurement_name: str,
    channel: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    selected = [r for r in records if str(r.get("name", "")) == str(measurement_name)]
    if not selected:
        raise ValueError(f"No inline records found for measurement '{measurement_name}'")

    first_vals = dict(selected[0].get("values", {}))
    basis_map: Dict[int, int] = {}
    nmom = int(first_vals.get("n_momenta", 0))
    for key, val in first_vals.items():
        m = _BASIS_KEY_RE.match(str(key))
        if m:
            basis_map[int(m.group("i"))] = int(round(float(val)))
    if nmom <= 0:
        nmom = max(basis_map.keys(), default=-1) + 1
    if nmom <= 0:
        raise ValueError("Could not infer momentum-basis size from measurement record")
    basis = np.asarray([int(basis_map.get(i, i)) for i in range(nmom)], dtype=np.int64)

    lt = 0
    for key in first_vals.keys():
        m = _MAT_KEY_RE.match(str(key))
        if m and m.group("prefix") == str(channel):
            lt = max(lt, int(m.group("t")) + 1)
    if lt <= 0:
        raise ValueError(f"No matrix correlator keys for channel '{channel}' in measurement '{measurement_name}'")

    samples = np.zeros((len(selected), nmom, nmom, lt), dtype=np.complex128)
    steps: List[int] = []
    for isamp, rec in enumerate(selected):
        vals = dict(rec.get("values", {}))
        steps.append(int(rec.get("step", isamp)))
        for key, val in vals.items():
            m = _MAT_KEY_RE.match(str(key))
            if not m or m.group("prefix") != str(channel):
                continue
            i = int(m.group("i"))
            j = int(m.group("j"))
            t = int(m.group("t"))
            if i >= nmom or j >= nmom or t >= lt:
                raise ValueError(f"Matrix key out of bounds: {key}")
            if m.group("part") == "re":
                samples[isamp, i, j, t] += float(val)
            else:
                samples[isamp, i, j, t] += 1j * float(val)
    return basis, samples, steps


def _hermitize_samples(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.complex128).copy()
    return 0.5 * (arr + np.conjugate(np.swapaxes(arr, 1, 2)))


def _choose_block_size(samples: np.ndarray, *, t_ref: int, method: str) -> Tuple[int, Dict[str, float]]:
    nmeas = int(samples.shape[0])
    t_use = max(0, min(int(t_ref), int(samples.shape[-1]) - 1))
    series = np.real(np.asarray(samples[:, 0, 0, t_use], dtype=np.complex128))
    info = integrated_autocorr_time(series, method=str(method))
    tau = float(info.get("tau_int", np.nan))
    if not np.isfinite(tau):
        tau = 1.0
    block = max(1, int(np.ceil(2.0 * tau)))
    block = min(block, max(1, nmeas // 4))
    while block > 1 and (nmeas // block) < 2:
        block -= 1
    return max(1, int(block)), {"t_ref": float(t_use), **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in info.items()}}


def _blocked_jackknife(samples: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    arr = np.asarray(samples)
    nmeas = int(arr.shape[0])
    bsz = max(1, int(block_size))
    nblock = nmeas // bsz
    if nblock < 2:
        raise ValueError(f"Need at least 2 blocks for jackknife; got nmeas={nmeas}, block_size={bsz}")
    ntrim = int(nblock * bsz)
    trimmed = np.asarray(arr[:ntrim])
    blocks = trimmed.reshape((nblock, bsz) + trimmed.shape[1:]).mean(axis=1)
    block_sum = np.sum(blocks, axis=0)
    jk = np.empty_like(blocks)
    for ib in range(nblock):
        jk[ib] = (block_sum - blocks[ib]) / float(nblock - 1)
    full = trimmed.mean(axis=0)
    return full, jk, int(nblock), int(ntrim)


def _jackknife_error(jk_vals: np.ndarray, full_val: np.ndarray) -> np.ndarray:
    arr = np.asarray(jk_vals)
    center = np.asarray(full_val)
    nb = int(arr.shape[0])
    if nb <= 1:
        return np.full_like(center, np.nan, dtype=np.float64)
    dif = arr - center
    return np.sqrt((nb - 1) / float(nb) * np.sum(np.abs(dif) ** 2, axis=0))


def _solve_gevp(mean_corr: np.ndarray, *, t0: int, n_levels: int, svd_cut: float) -> Dict[str, np.ndarray]:
    c = np.asarray(mean_corr, dtype=np.complex128)
    nmom, _, lt = c.shape
    if t0 < 0 or t0 >= lt:
        raise ValueError(f"t0 must satisfy 0 <= t0 < Lt={lt}; got {t0}")
    c0 = 0.5 * (c[:, :, t0] + np.conjugate(c[:, :, t0].T))
    evals0, evecs0 = np.linalg.eigh(c0)
    lam_max = float(np.max(np.real(evals0))) if evals0.size else 0.0
    keep = np.where(np.real(evals0) > float(svd_cut) * max(lam_max, 1.0e-30))[0]
    if keep.size == 0:
        raise ValueError("GEVP metric is singular at t0; no eigenmodes survived the cutoff")
    v0 = evecs0[:, keep]
    l0 = np.real(evals0[keep])
    whitener = v0 / np.sqrt(l0)[None, :]
    nkeep = int(keep.size)
    nout = min(int(n_levels), nkeep)
    lambdas = np.full((nout, lt), np.nan, dtype=np.float64)

    for t in range(lt):
        ct = 0.5 * (c[:, :, t] + np.conjugate(c[:, :, t].T))
        red = np.conjugate(whitener).T @ ct @ whitener
        red = 0.5 * (red + np.conjugate(red.T))
        ew, _ = np.linalg.eigh(red)
        ew = np.real(ew)
        ew = ew[np.argsort(ew)[::-1]]
        lambdas[:nout, t] = ew[:nout]
    return {
        "lambdas": lambdas,
        "metric_evals": np.real(evals0),
        "kept_metric_modes": np.asarray(keep, dtype=np.int64),
        "nkeep": np.asarray([nkeep], dtype=np.int64),
    }


def _effective_energies(lambdas: np.ndarray) -> np.ndarray:
    lam = np.asarray(lambdas, dtype=np.float64)
    if lam.ndim != 2:
        raise ValueError(f"Expected principal correlators with shape (Nlvl, Lt), got {lam.shape}")
    out = np.full((int(lam.shape[0]), max(0, int(lam.shape[1]) - 1)), np.nan, dtype=np.float64)
    good = (lam[:, :-1] > 0.0) & (lam[:, 1:] > 0.0)
    out[good] = np.log(lam[:, :-1][good] / lam[:, 1:][good])
    return out


def _parse_fit_range(text: str, *, tmin: int, tmax: int) -> Tuple[int, int]:
    vals = _parse_int_list(text)
    if len(vals) != 2:
        raise ValueError(f"fit range must contain exactly 2 integers, got {text!r}")
    a, b = int(vals[0]), int(vals[1])
    if a > b:
        a, b = b, a
    if a < tmin or b > tmax:
        raise ValueError(f"fit range [{a},{b}] is outside allowed meff indices [{tmin},{tmax}]")
    return a, b


def _estimate_level_energies(
    eff_full: np.ndarray,
    eff_jk: np.ndarray,
    *,
    fit_range: Tuple[int, int],
) -> List[Dict[str, float]]:
    a, b = fit_range
    out: List[Dict[str, float]] = []
    nb = int(eff_jk.shape[0])
    for lvl in range(int(eff_full.shape[0])):
        center = np.asarray(eff_full[lvl, a : b + 1], dtype=np.float64)
        finite_center = center[np.isfinite(center)]
        if finite_center.size == 0:
            out.append({"level": float(lvl), "energy": float("nan"), "energy_err": float("nan")})
            continue
        e_full = float(np.mean(finite_center))
        jk_est = np.full((nb,), np.nan, dtype=np.float64)
        for ib in range(nb):
            seg = np.asarray(eff_jk[ib, lvl, a : b + 1], dtype=np.float64)
            seg = seg[np.isfinite(seg)]
            if seg.size:
                jk_est[ib] = float(np.mean(seg))
        good = jk_est[np.isfinite(jk_est)]
        if good.size <= 1:
            e_err = float("nan")
        else:
            mean_jk = float(np.mean(good))
            e_err = float(np.sqrt((good.size - 1) / float(good.size) * np.sum((good - mean_jk) ** 2)))
        out.append(
            {
                "level": float(lvl),
                "energy": e_full,
                "energy_err": e_err,
                "fit_t_min": float(a),
                "fit_t_max": float(b),
            }
        )
    return out


def _plot_summary(
    *,
    lambdas: np.ndarray,
    lambda_err: np.ndarray,
    eff: np.ndarray,
    eff_err: np.ndarray,
    energies: Sequence[Dict[str, float]],
    fit_range: Tuple[int, int],
    t0: int,
    out_path: Path,
    title_prefix: str,
) -> None:
    nlev = int(lambdas.shape[0])
    lt = int(lambdas.shape[1])
    teff = np.arange(int(eff.shape[1]), dtype=np.int64)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), constrained_layout=True)

    ax0, ax1 = axes
    tt = np.arange(lt, dtype=np.int64)
    for lvl in range(nlev):
        ax0.errorbar(
            tt,
            lambdas[lvl],
            yerr=lambda_err[lvl],
            fmt="o-",
            ms=3.0,
            lw=1.0,
            capsize=2.0,
            label=f"level {lvl}",
        )
    ax0.set_yscale("log")
    ax0.axvline(int(t0), color="k", lw=1.0, ls="--", alpha=0.5)
    ax0.set_xlabel("t")
    ax0.set_ylabel(r"$\lambda_n(t,t_0)$")
    ax0.set_title(f"{title_prefix}: principal correlators")
    ax0.legend(loc="best", fontsize=9)

    a, b = fit_range
    for lvl in range(nlev):
        ax1.errorbar(
            teff,
            eff[lvl],
            yerr=eff_err[lvl],
            fmt="o-",
            ms=3.0,
            lw=1.0,
            capsize=2.0,
            label=f"level {lvl}",
        )
        if lvl < len(energies):
            en = energies[lvl]
            if np.isfinite(en["energy"]):
                ax1.hlines(
                    en["energy"],
                    a,
                    b + 1,
                    colors=ax1.lines[-1].get_color(),
                    linestyles="--",
                    linewidth=1.0,
                )
    ax1.axvspan(a - 0.1, b + 0.1, color="0.85", alpha=0.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$E_n^{\mathrm{eff}}(t)$")
    ax1.set_title(f"{title_prefix}: effective energies")
    ax1.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze two-pion I=2 correlator matrices from an mcmc checkpoint")
    ap.add_argument("--input", required=True, help="checkpoint .pkl produced by scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", default="pipi_i2_matrix", help="inline measurement name")
    ap.add_argument("--channel", default="full", choices=("full", "direct", "exchange"), help="correlator-matrix channel")
    ap.add_argument("--t0", type=int, default=1, help="GEVP reference timeslice")
    ap.add_argument("--levels", type=int, default=4, help="number of levels to retain")
    ap.add_argument("--svd-cut", type=float, default=1.0e-10, help="relative cutoff for C(t0) metric eigenmodes")
    ap.add_argument("--fit-range", type=str, default="", help="effective-energy fit window 'tmin,tmax'; default is t0+1..min(t0+4, Lt/2-2)")
    ap.add_argument("--block-size", type=int, default=0, help="jackknife block size; <=0 selects 2*tau_int automatically")
    ap.add_argument("--iat-method", type=str, default="ips", choices=("ips", "sokal", "gamma"), help="IAT estimator for automatic blocking")
    ap.add_argument("--no-hermitize", action="store_true", help="skip C -> (C + C^dagger)/2 symmetrization per sample")
    ap.add_argument("--outdir", type=str, default="", help="output directory (default: alongside checkpoint)")
    ap.add_argument("--prefix", type=str, default="", help="output filename prefix")
    ap.add_argument("--save", type=str, default="", help="explicit PNG output path")
    ap.add_argument("--json-out", type=str, default="", help="explicit JSON output path")
    ap.add_argument("--no-gui", action="store_true", help="suppress interactive display")
    args = ap.parse_args()

    ckpt_path = Path(args.input).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else ckpt_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip() or f"{ckpt_path.stem}_{args.measurement}_{args.channel}"
    png_path = Path(args.save).expanduser().resolve() if args.save else (outdir / f"{prefix}.png")
    json_path = Path(args.json_out).expanduser().resolve() if args.json_out else (outdir / f"{prefix}.json")

    records = _load_inline_records(ckpt_path)
    basis, samples, steps = _extract_samples(records, measurement_name=args.measurement, channel=args.channel)
    if not args.no_hermitize:
        samples = _hermitize_samples(samples)

    lt = int(samples.shape[-1])
    block_auto_info: Dict[str, float] = {}
    if int(args.block_size) > 0:
        block_size = int(args.block_size)
    else:
        block_size, block_auto_info = _choose_block_size(samples, t_ref=max(1, int(args.t0)), method=str(args.iat_method))

    mean_corr, jk_corr, nblock, ntrim = _blocked_jackknife(samples, block_size)
    gevp_full = _solve_gevp(mean_corr, t0=int(args.t0), n_levels=int(args.levels), svd_cut=float(args.svd_cut))
    lambdas = np.asarray(gevp_full["lambdas"], dtype=np.float64)
    jk_lambdas = np.zeros((nblock, *lambdas.shape), dtype=np.float64)
    for ib in range(nblock):
        jk_lambdas[ib] = _solve_gevp(
            jk_corr[ib],
            t0=int(args.t0),
            n_levels=int(args.levels),
            svd_cut=float(args.svd_cut),
        )["lambdas"]

    lambda_err = _jackknife_error(jk_lambdas, lambdas)
    eff = _effective_energies(lambdas)
    eff_jk = np.asarray([_effective_energies(jk_lambdas[ib]) for ib in range(nblock)], dtype=np.float64)
    eff_err = _jackknife_error(eff_jk, eff)

    if args.fit_range.strip():
        fit_range = _parse_fit_range(args.fit_range, tmin=0, tmax=max(0, eff.shape[1] - 1))
    else:
        tmin = min(max(0, int(args.t0) + 1), max(0, eff.shape[1] - 1))
        tmax = min(max(tmin, int(args.t0) + 4), max(0, eff.shape[1] - 1), max(0, lt // 2 - 2))
        fit_range = (tmin, tmax)
    energies = _estimate_level_energies(eff, eff_jk, fit_range=fit_range)

    _plot_summary(
        lambdas=lambdas,
        lambda_err=lambda_err,
        eff=eff,
        eff_err=eff_err,
        energies=energies,
        fit_range=fit_range,
        t0=int(args.t0),
        out_path=png_path,
        title_prefix=f"{args.measurement} [{args.channel}]",
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
        "t0": int(args.t0),
        "svd_cut": float(args.svd_cut),
        "fit_range": [int(fit_range[0]), int(fit_range[1])],
        "kept_metric_modes": np.asarray(gevp_full["kept_metric_modes"]).astype(int).tolist(),
        "principal_correlators": {
            "mean": np.asarray(lambdas, dtype=np.float64).tolist(),
            "err": np.asarray(lambda_err, dtype=np.float64).tolist(),
        },
        "effective_energies": {
            "mean": np.asarray(eff, dtype=np.float64).tolist(),
            "err": np.asarray(eff_err, dtype=np.float64).tolist(),
        },
        "energies": energies,
        "plot": str(png_path),
    }
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Loaded {samples.shape[0]} samples from {ckpt_path}")
    print(f"Basis momenta: {basis.astype(int).tolist()}")
    print(f"Using block_size={block_size}, n_blocks={nblock}, t0={int(args.t0)}, fit_range={fit_range}")
    for en in energies:
        lvl = int(en["level"])
        print(f"level {lvl}: E = {en['energy']:.8g} +/- {en['energy_err']:.3g}")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")

    if not args.no_gui:
        img = plt.imread(png_path)
        plt.figure(figsize=(9.0, 8.0))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

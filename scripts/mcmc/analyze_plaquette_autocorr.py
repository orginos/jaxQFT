#!/usr/bin/env python3
"""Analyze plaquette autocorrelation from mcmc.py output logs.

Reads plaquette measurements from either:
- mcmc stdout log lines:  meas k=<int> plaquette=<float> ...
- CSV produced by live_plot_plaquette.py: columns k,plaquette

Computes:
- normalized autocorrelation function rho(t)
- jackknife errors for rho(t) from contiguous block deletion
- integrated autocorrelation time (IAT) using ips/sokal/gamma
- jackknife error estimate for tau_int at the selected window

Also produces a figure with:
1) rho(t) with error bars
2) tau_int(W) = 0.5 + sum_{t=1}^W rho(t), with selected window/iat marked
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


MEAS_RE = re.compile(
    r"meas\s+k=(?P<k>\d+).*?\bplaquette=(?P<p>(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|nan|inf|-inf)"
)


def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
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
    x: np.ndarray,
    *,
    method: str = "ips",
    c: float = 5.0,
    max_lag: Optional[int] = None,
) -> dict:
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
            if not np.isfinite(g) or g <= 0.0:
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

    raise ValueError(f"Unknown method: {method}")


def _read_log_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ks = []
    ps = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MEAS_RE.search(line)
            if m is None:
                continue
            try:
                ks.append(int(m.group("k")))
                ps.append(float(m.group("p")))
            except Exception:
                continue
    if not ks:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    order = np.argsort(np.asarray(ks, dtype=np.int64))
    return np.asarray(ks, dtype=np.int64)[order], np.asarray(ps, dtype=np.float64)[order]


def _read_csv_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ks = []
    ps = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if "k" not in row or "plaquette" not in row:
                continue
            try:
                ks.append(int(row["k"]))
                ps.append(float(row["plaquette"]))
            except Exception:
                continue
    if not ks:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    order = np.argsort(np.asarray(ks, dtype=np.int64))
    return np.asarray(ks, dtype=np.int64)[order], np.asarray(ps, dtype=np.float64)[order]


def read_plaquette_series(path: Path, fmt: str) -> Tuple[np.ndarray, np.ndarray]:
    f = str(fmt).lower()
    if f == "auto":
        if path.suffix.lower() == ".csv":
            return _read_csv_points(path)
        return _read_log_points(path)
    if f == "csv":
        return _read_csv_points(path)
    if f == "log":
        return _read_log_points(path)
    raise ValueError(f"Unknown input format: {fmt}")


def _jackknife_rho(
    x: np.ndarray,
    max_lag: int,
    nblocks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    lag = max(1, min(int(max_lag), n - 1))

    b = max(2, int(nblocks))
    block = n // b
    while b > 2 and block < 8:
        b -= 1
        block = n // b
    if block < 4:
        return np.asarray([]), np.asarray([]), np.asarray([])

    nuse = b * block
    data = a[:nuse]
    rhos = []
    for i in range(b):
        lo = i * block
        hi = (i + 1) * block
        loo = np.concatenate((data[:lo], data[hi:]))
        rho = autocorrelation_fft(loo)
        rhos.append(np.asarray(rho[: lag + 1], dtype=np.float64))

    arr = np.asarray(rhos, dtype=np.float64)  # [B, lag+1]
    mean = np.mean(arr, axis=0)
    err = np.sqrt((b - 1.0) / b * np.sum((arr - mean[None, :]) ** 2, axis=0))
    return mean, err, arr


def _tau_curve(rho: np.ndarray, max_lag: int) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64)
    lag = min(int(max_lag), int(r.size) - 1)
    if lag < 1:
        return np.asarray([], dtype=np.float64)
    return np.asarray(0.5 + np.cumsum(r[1 : lag + 1]), dtype=np.float64)


def _summary_text(iat: dict, n: int, mean_p: float, std_p: float, tau_err: float) -> str:
    if not bool(iat.get("ok", False)):
        return (
            f"N={n}\\n<plaq>={mean_p:.6f}\\nstd={std_p:.6f}\\n"
            f"IAT unavailable: {iat.get('message', 'unknown')}"
        )
    tau = float(iat["tau_int"])
    ess = float(iat["ess"])
    w = int(iat["window"])
    err_txt = f" ± {tau_err:.3f}" if np.isfinite(tau_err) else ""
    return (
        f"N={n}\\n<plaq>={mean_p:.6f}\\nstd={std_p:.6f}\\n"
        f"tau_int={tau:.3f}{err_txt}\\nESS={ess:.1f}/{n}\\nW*={w}"
    )


def _maybe_setup_matplotlib_cache():
    if "MPLCONFIGDIR" not in os.environ:
        mpldir = Path(tempfile.gettempdir()) / f"mplconfig_{os.getuid()}"
        mpldir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpldir)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg = Path(tempfile.gettempdir()) / f"xdg_cache_{os.getuid()}"
        xdg.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg)


def _default_nblocks(n: int) -> int:
    return max(8, min(40, max(2, n // 25)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze plaquette autocorrelation with jackknife errors")
    ap.add_argument("--input", type=str, required=True, help="mcmc stdout log or CSV file")
    ap.add_argument("--format", type=str, default="auto", choices=["auto", "log", "csv"])
    ap.add_argument("--discard", type=int, default=0, help="discard first N measurements before analysis")
    ap.add_argument("--max-lag", type=int, default=200, help="maximum lag for rho(t) and tau(W)")
    ap.add_argument("--iat-method", type=str, default="ips", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--iat-c", type=float, default=5.0, help="window constant c for sokal/gamma methods")
    ap.add_argument("--nblocks", type=int, default=0, help="jackknife blocks (0=auto)")
    ap.add_argument("--title", type=str, default="Plaquette Autocorrelation Analysis")
    ap.add_argument("--save", type=str, default="", help="save figure path (png/pdf/...)")
    ap.add_argument("--json-out", type=str, default="", help="optional JSON summary output path")
    ap.add_argument("--no-gui", action="store_true", help="disable interactive display")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    ks, plaq = read_plaquette_series(in_path, fmt=args.format)
    if ks.size == 0:
        raise RuntimeError("No plaquette measurements found in input")

    discard = max(0, int(args.discard))
    if discard >= int(plaq.size):
        raise ValueError(f"--discard={discard} removes all samples (N={plaq.size})")

    ks = ks[discard:]
    plaq = plaq[discard:]
    n = int(plaq.size)
    if n < 4:
        raise RuntimeError(f"Need at least 4 samples after discard; got {n}")

    max_lag = max(1, min(int(args.max_lag), n - 1))
    rho = autocorrelation_fft(plaq)[: max_lag + 1]
    iat = integrated_autocorr_time(plaq, method=args.iat_method, c=float(args.iat_c), max_lag=max_lag)

    nblocks = _default_nblocks(n) if int(args.nblocks) <= 0 else int(args.nblocks)
    rho_jk_mean, rho_jk_err, rho_jk_loo = _jackknife_rho(plaq, max_lag=max_lag, nblocks=nblocks)

    tau_curve = _tau_curve(rho, max_lag=max_lag)
    tau_jk_mean = np.asarray([], dtype=np.float64)
    tau_jk_err = np.asarray([], dtype=np.float64)
    tau_err_at_w = float("nan")
    w_star = int(iat.get("window", 0)) if bool(iat.get("ok", False)) else 0
    if rho_jk_loo.size > 0:
        tau_loo = np.asarray([_tau_curve(r, max_lag=max_lag) for r in rho_jk_loo], dtype=np.float64)
        tau_jk_mean = np.mean(tau_loo, axis=0)
        b = float(tau_loo.shape[0])
        tau_jk_err = np.sqrt((b - 1.0) / b * np.sum((tau_loo - tau_jk_mean[None, :]) ** 2, axis=0))
        if w_star >= 1 and w_star <= int(tau_loo.shape[1]):
            tau_at_w = tau_loo[:, w_star - 1]
            tau_mean_at_w = float(np.mean(tau_at_w))
            tau_err_at_w = float(np.sqrt((b - 1.0) / b * np.sum((tau_at_w - tau_mean_at_w) ** 2)))

    mean_p = float(np.mean(plaq))
    std_p = float(np.std(plaq, ddof=1)) if n > 1 else float("nan")

    print("Plaquette Autocorr Summary")
    print(f"  input: {in_path}")
    print(f"  samples: {n} (discarded {discard})")
    print(f"  mean plaquette: {mean_p:.8f}")
    print(f"  std plaquette: {std_p:.8f}")
    print(f"  max_lag: {max_lag}")
    print(f"  jackknife blocks: {nblocks}")
    if bool(iat.get("ok", False)):
        print(
            "  IAT:"
            f" method={iat['method']}"
            f" tau_int={float(iat['tau_int']):.6f}"
            f" window={int(iat['window'])}"
            f" ESS={float(iat['ess']):.2f}/{n}"
        )
        if np.isfinite(tau_err_at_w):
            print(f"  IAT jackknife error (at W*): {tau_err_at_w:.6f}")
    else:
        print(f"  IAT: unavailable ({iat.get('message', 'unknown')})")

    _maybe_setup_matplotlib_cache()
    if args.no_gui:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False)

    lags = np.arange(0, max_lag + 1, dtype=np.int64)
    ax0.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    if rho_jk_err.size == rho.size and np.all(np.isfinite(rho_jk_err)):
        ax0.errorbar(lags, rho, yerr=rho_jk_err, fmt="o", ms=2.8, lw=1.0, capsize=2, color="#1f77b4")
    else:
        ax0.plot(lags, rho, marker="o", ms=2.8, lw=1.0, color="#1f77b4")
    ax0.set_ylabel(r"$\rho(t)$")
    ax0.set_title(str(args.title))
    ax0.grid(True, alpha=0.25)
    ax0.text(
        0.98,
        0.98,
        _summary_text(iat, n=n, mean_p=mean_p, std_p=std_p, tau_err=tau_err_at_w),
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )

    if tau_curve.size > 0:
        ws = np.arange(1, int(tau_curve.size) + 1, dtype=np.int64)
        ax1.plot(ws, tau_curve, lw=1.6, color="#d62728", label=r"$\tau_{\mathrm{int}}(W)$")
        if tau_jk_err.size == tau_curve.size and np.all(np.isfinite(tau_jk_err)):
            lo = tau_curve - tau_jk_err
            hi = tau_curve + tau_jk_err
            ax1.fill_between(ws, lo, hi, color="#d62728", alpha=0.20, linewidth=0.0, label="jackknife 1σ")
        if bool(iat.get("ok", False)):
            tau_star = float(iat["tau_int"])
            w_star = int(iat["window"])
            if w_star >= 1:
                ax1.axvline(w_star, color="#2ca02c", ls="--", lw=1.2, label=f"W*={w_star}")
            ax1.axhline(tau_star, color="#9467bd", ls="--", lw=1.2, label=rf"$\tau_{{int}}={tau_star:.3f}$")
        ax1.legend(loc="best", fontsize=9)
    ax1.set_xlabel("lag window W")
    ax1.set_ylabel(r"$\tau_{\mathrm{int}}(W)$")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()

    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  saved figure: {out}")

    if args.json_out:
        import json

        jout = Path(args.json_out).expanduser().resolve()
        jout.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(in_path),
            "discard": int(discard),
            "n_samples": int(n),
            "mean_plaquette": float(mean_p),
            "std_plaquette": float(std_p),
            "max_lag": int(max_lag),
            "nblocks": int(nblocks),
            "iat": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in iat.items()},
            "iat_jackknife_error": float(tau_err_at_w) if np.isfinite(tau_err_at_w) else None,
            "rho": np.asarray(rho, dtype=np.float64).tolist(),
            "rho_jk_err": np.asarray(rho_jk_err, dtype=np.float64).tolist() if rho_jk_err.size else [],
            "tau_curve": np.asarray(tau_curve, dtype=np.float64).tolist(),
            "tau_curve_jk_err": np.asarray(tau_jk_err, dtype=np.float64).tolist() if tau_jk_err.size else [],
            "k_min": int(np.min(ks)),
            "k_max": int(np.max(ks)),
        }
        jout.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  saved json: {jout}")

    if not args.no_gui:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

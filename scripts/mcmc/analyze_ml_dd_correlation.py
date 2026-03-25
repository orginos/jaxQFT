#!/usr/bin/env python3
"""Analyze exact-vs-truncated correlation for multilevel DD two-point observables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_2pt_dispersion import _auto_block_size_from_p0, _load_checkpoint
from inspect_2pt_correlators import _block_means_complex, _extract_complex_by_momentum


def _parse_momenta(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None or str(text).strip() == "":
        return None
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    return tuple(int(t) for t in toks) if toks else None


def _align_samples(
    steps_ref: np.ndarray,
    samples_ref: np.ndarray,
    steps_cmp: np.ndarray,
    samples_cmp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_ref = {int(step): i for i, step in enumerate(np.asarray(steps_ref, dtype=np.int64).tolist())}
    idx_cmp = {int(step): i for i, step in enumerate(np.asarray(steps_cmp, dtype=np.int64).tolist())}
    common = sorted(set(idx_ref.keys()) & set(idx_cmp.keys()))
    if len(common) < 2:
        raise ValueError("Need at least two common samples between exact and approximate channels")
    a = np.asarray([samples_ref[idx_ref[s]] for s in common], dtype=np.complex128)
    b = np.asarray([samples_cmp[idx_cmp[s]] for s in common], dtype=np.complex128)
    return np.asarray(common, dtype=np.int64), a, b


def _apply_discard_stride_complex(
    samples: np.ndarray,
    steps: np.ndarray,
    *,
    discard: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=np.complex128)
    st = np.asarray(steps, dtype=np.int64)
    d = max(0, int(discard))
    s = max(1, int(stride))
    if d >= int(arr.shape[0]):
        return arr[:0], st[:0]
    return np.asarray(arr[d::s], dtype=np.complex128), np.asarray(st[d::s], dtype=np.int64)


def _pearson_real(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.asarray(x, dtype=np.float64)
    yr = np.asarray(y, dtype=np.float64)
    if xr.size != yr.size or xr.size < 2:
        return float("nan")
    sx = float(np.std(xr, ddof=1))
    sy = float(np.std(yr, ddof=1))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze multilevel DD exact-vs-truncated correlation")
    ap.add_argument("--input", required=True, help="checkpoint pickle produced by scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", type=str, default="pion_2pt_ml_dd")
    ap.add_argument("--exact-channel", type=str, default="exact")
    ap.add_argument("--approx-channel", type=str, default="approx_l0")
    ap.add_argument("--momenta", type=str, default="", help="comma-separated subset; default uses all available")
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=0, help="<=0: auto from p=0 exact real-part IAT")
    ap.add_argument("--iat-method", type=str, default="gamma", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--eps", type=float, default=1e-15)
    ap.add_argument("--no-gui", action="store_true", help="write JSON summary only; skip plotting")
    args = ap.parse_args()

    ckpt = Path(args.input).resolve()
    _, records = _load_checkpoint(ckpt)
    exact_by_p, exact_steps_by_p, exact_meta = _extract_complex_by_momentum(
        records,
        measurement=str(args.measurement),
        channel=str(args.exact_channel),
    )
    approx_by_p, approx_steps_by_p, approx_meta = _extract_complex_by_momentum(
        records,
        measurement=str(args.measurement),
        channel=str(args.approx_channel),
    )

    selected = _parse_momenta(args.momenta)
    available = sorted(set(exact_by_p.keys()) & set(approx_by_p.keys()))
    momenta = available if selected is None else [p for p in selected if p in available]
    if not momenta:
        raise SystemExit("No common requested momenta were found in the checkpoint")

    exact_filt: Dict[int, np.ndarray] = {}
    approx_filt: Dict[int, np.ndarray] = {}
    steps_filt: Dict[int, np.ndarray] = {}
    for p in momenta:
        ex_s, ex_st = _apply_discard_stride_complex(
            exact_by_p[p],
            exact_steps_by_p[p],
            discard=int(args.discard),
            stride=int(args.stride),
        )
        ap_s, ap_st = _apply_discard_stride_complex(
            approx_by_p[p],
            approx_steps_by_p[p],
            discard=int(args.discard),
            stride=int(args.stride),
        )
        common_steps, ex_aligned, ap_aligned = _align_samples(ex_st, ex_s, ap_st, ap_s)
        exact_filt[p] = ex_aligned
        approx_filt[p] = ap_aligned
        steps_filt[p] = common_steps

    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        tau_ref = float("nan")
    else:
        real_samples_for_auto = {int(p): np.asarray(v.real, dtype=np.float64) for p, v in exact_filt.items()}
        block_size, tau_ref = _auto_block_size_from_p0(
            real_samples_for_auto,
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=(None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)),
        )

    summary: Dict[str, object] = {
        "input_checkpoint": str(ckpt),
        "measurement": str(args.measurement),
        "exact_channel": str(args.exact_channel),
        "approx_channel": str(args.approx_channel),
        "momenta": [int(p) for p in momenta],
        "block_size": int(block_size),
        "tau_ref": None if not np.isfinite(tau_ref) else float(tau_ref),
        "discard": int(args.discard),
        "stride": int(args.stride),
        "per_momentum": {},
    }

    plot_data: Dict[int, Dict[str, np.ndarray]] = {}
    for p in momenta:
        ex_blocks = _block_means_complex(exact_filt[p], block_size=int(block_size))
        ap_blocks = _block_means_complex(approx_filt[p], block_size=int(block_size))
        mean_ex = np.asarray(np.mean(ex_blocks, axis=0), dtype=np.complex128)
        mean_ap = np.asarray(np.mean(ap_blocks, axis=0), dtype=np.complex128)
        nt = int(mean_ex.size)
        support_times = np.asarray(exact_meta.get(int(p), {}).get("support_times", list(range(nt))), dtype=np.int64)
        if int(support_times.size) != nt:
            raise SystemExit(
                f"Support-size mismatch for p={int(p)}: meta has {int(support_times.size)} times, mean has {nt} entries"
            )

        corr_re = np.asarray([_pearson_real(ap_blocks[:, t].real, ex_blocks[:, t].real) for t in range(nt)], dtype=np.float64)
        corr_abs = np.asarray([_pearson_real(np.abs(ap_blocks[:, t]), np.abs(ex_blocks[:, t])) for t in range(nt)], dtype=np.float64)
        ratio_abs = np.asarray(np.abs(mean_ap) / np.maximum(np.abs(mean_ex), float(args.eps)), dtype=np.float64)
        rmse_re = np.asarray(
            [math.sqrt(float(np.mean((ap_blocks[:, t].real - ex_blocks[:, t].real) ** 2))) for t in range(nt)],
            dtype=np.float64,
        )

        plot_data[int(p)] = {
            "support_times": support_times,
            "mean_ex_abs": np.abs(mean_ex),
            "mean_ap_abs": np.abs(mean_ap),
            "corr_re": corr_re,
            "corr_abs": corr_abs,
        }
        summary["per_momentum"][str(int(p))] = {
            "n_common_samples": int(exact_filt[p].shape[0]),
            "n_blocks": int(ex_blocks.shape[0]),
            "support_times": support_times.astype(int).tolist(),
            "mean_corr_re": float(np.nanmean(corr_re)) if np.any(np.isfinite(corr_re)) else float("nan"),
            "min_corr_re": float(np.nanmin(corr_re)) if np.any(np.isfinite(corr_re)) else float("nan"),
            "mean_corr_abs": float(np.nanmean(corr_abs)) if np.any(np.isfinite(corr_abs)) else float("nan"),
            "mean_abs_ratio": float(np.nanmean(ratio_abs)) if np.any(np.isfinite(ratio_abs)) else float("nan"),
            "median_abs_ratio": float(np.nanmedian(ratio_abs)) if np.any(np.isfinite(ratio_abs)) else float("nan"),
            "max_rmse_re": float(np.nanmax(rmse_re)) if np.any(np.isfinite(rmse_re)) else float("nan"),
            "steps_first": int(steps_filt[p][0]),
            "steps_last": int(steps_filt[p][-1]),
            "exact_meta": dict(exact_meta.get(int(p), {})),
            "approx_meta": dict(approx_meta.get(int(p), {})),
        }

    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else ckpt.parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip() or ckpt.stem
    json_path = outdir / f"{prefix}_{args.measurement}_{args.approx_channel}_vs_{args.exact_channel}_correlation.json"
    fig_path = outdir / f"{prefix}_{args.measurement}_{args.approx_channel}_vs_{args.exact_channel}_correlation.png"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plot_written = False
    if not bool(args.no_gui):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise SystemExit(f"matplotlib is required for plotting unless --no-gui is used: {exc}")

        n_mom = len(momenta)
        fig, axes = plt.subplots(2, n_mom, figsize=(4.4 * n_mom, 6.8), sharex="col", squeeze=False)
        for ip, p in enumerate(momenta):
            dat = plot_data[int(p)]
            t = np.asarray(dat["support_times"], dtype=np.int64)

            ax_top = axes[0, ip]
            ax_top.semilogy(
                t,
                np.maximum(np.asarray(dat["mean_ex_abs"], dtype=np.float64), float(args.eps)),
                marker="o",
                ms=3,
                lw=1.5,
                label="|exact|",
            )
            ax_top.semilogy(
                t,
                np.maximum(np.asarray(dat["mean_ap_abs"], dtype=np.float64), float(args.eps)),
                marker="s",
                ms=3,
                lw=1.2,
                ls="--",
                label="|approx|",
            )
            ax_top.grid(alpha=0.25)
            ax_top.set_title(f"p={int(p)}")
            if ip == 0:
                ax_top.set_ylabel(r"$|C(t)|$")
            ax_top.legend(fontsize=8, loc="best")

            ax_bot = axes[1, ip]
            ax_bot.plot(t, np.asarray(dat["corr_re"], dtype=np.float64), marker="o", ms=3, lw=1.4, label="corr(Re)")
            ax_bot.plot(t, np.asarray(dat["corr_abs"], dtype=np.float64), marker="s", ms=3, lw=1.1, ls="--", label="corr(|.|)")
            ax_bot.axhline(0.9, color="0.5", lw=0.8, ls="--", alpha=0.8)
            ax_bot.axhline(0.5, color="0.7", lw=0.8, ls=":", alpha=0.8)
            ax_bot.set_ylim(-1.05, 1.05)
            ax_bot.grid(alpha=0.25)
            if ip == 0:
                ax_bot.set_ylabel("Correlation")
            ax_bot.set_xlabel("t")
            ax_bot.legend(fontsize=8, loc="best")

        fig.suptitle(
            f"{args.measurement}: {args.approx_channel} vs {args.exact_channel} | "
            f"{ckpt.name} | block={block_size}" + (f", tau_ref~{tau_ref:.2f}" if np.isfinite(tau_ref) else ""),
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        plot_written = True

    print(f"saved json:   {json_path}")
    if plot_written:
        print(f"saved figure: {fig_path}")
    print(f"block size: {block_size}" + (f" (auto tau_ref~{tau_ref:.2f})" if np.isfinite(tau_ref) else ""))
    for p in momenta:
        info = summary["per_momentum"][str(int(p))]
        print(
            f"  p={int(p)}: mean corr(Re)={float(info['mean_corr_re']):.3f}"
            f" min corr(Re)={float(info['min_corr_re']):.3f}"
            f" mean corr(|.|)={float(info['mean_corr_abs']):.3f}"
            f" median |approx|/|exact|={float(info['median_abs_ratio']):.3e}"
        )


if __name__ == "__main__":
    main()

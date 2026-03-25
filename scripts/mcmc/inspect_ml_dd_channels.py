#!/usr/bin/env python3
"""Inspect multilevel DD channels per momentum and write SVG summary plots.

For each requested momentum, this script writes a 2x2 SVG panel with:
1) exact correlator (real part, blocked mean +/- stderr)
2) corrected correlator (real part, blocked mean +/- stderr)
3) bias correlator (real part, blocked mean +/- stderr)
4) per-timeslice level-0 correlation corr(approx_l0, exact)

It also writes a JSON sidecar with the plotted numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_2pt_dispersion import _auto_block_size_from_p0, _load_checkpoint
from inspect_2pt_correlators import _block_means_complex, _extract_complex_by_momentum


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


def _stderr_from_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"Expected 2D blocked samples, got shape {b.shape}")
    n = int(b.shape[0])
    if n < 2:
        raise ValueError(f"Need at least two blocks, got {n}")
    return np.asarray(np.std(b, axis=0, ddof=1) / math.sqrt(float(n)), dtype=np.float64)


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


def _parse_momenta(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None or str(text).strip() == "":
        return None
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    return tuple(int(t) for t in toks) if toks else None


def _align_many(step_map: Mapping[str, np.ndarray], sample_map: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    common: Optional[set[int]] = None
    for key in step_map:
        s = set(int(v) for v in np.asarray(step_map[key], dtype=np.int64).tolist())
        common = s if common is None else common & s
    common_sorted = sorted(common or [])
    if len(common_sorted) < 2:
        raise ValueError("Need at least two common samples across all requested channels")
    out: Dict[str, np.ndarray] = {}
    for key in sample_map:
        idx = {int(step): i for i, step in enumerate(np.asarray(step_map[key], dtype=np.int64).tolist())}
        out[key] = np.asarray([sample_map[key][idx[s]] for s in common_sorted], dtype=np.complex128)
    return np.asarray(common_sorted, dtype=np.int64), out


def _nice_limits(ymin: float, ymax: float) -> Tuple[float, float]:
    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return (-1.0, 1.0)
    if ymin == ymax:
        pad = 1.0 if ymin == 0.0 else 0.15 * abs(ymin)
        return (ymin - pad, ymax + pad)
    pad = 0.08 * max(1e-12, ymax - ymin)
    lo = ymin - pad
    hi = ymax + pad
    if lo > 0.0 and ymin >= 0.0:
        lo = max(0.0, lo)
    if hi < 0.0 and ymax <= 0.0:
        hi = min(0.0, hi)
    return (lo, hi)


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]


def _svg_footer() -> List[str]:
    return ["</svg>"]


def _text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#111">{safe}</text>'
    )


def _polyline(points: Sequence[Tuple[float, float]], *, color: str = "#1f77b4", width: float = 2.0) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width:.2f}" points="{pts}"/>'


def _line(x1: float, y1: float, x2: float, y2: float, *, color: str = "#666", width: float = 1.0, dash: Optional[str] = None) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"{extra}/>'


def _circle(x: float, y: float, *, r: float = 2.6, color: str = "#1f77b4") -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="none"/>'


def _plot_panel(
    svg: List[str],
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    times: np.ndarray,
    y: np.ndarray,
    err: Optional[np.ndarray],
    title: str,
    y_label: str,
    color: str,
    y_limits: Tuple[float, float],
    guide_lines: Sequence[Tuple[float, str, str]] = (),
) -> None:
    left = x0 + 56.0
    right = x0 + w - 18.0
    top = y0 + 22.0
    bottom = y0 + h - 34.0
    plot_w = max(10.0, right - left)
    plot_h = max(10.0, bottom - top)
    t = np.asarray(times, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ee = None if err is None else np.asarray(err, dtype=np.float64)
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    if tmax == tmin:
        tmax = tmin + 1.0
    ymin, ymax = float(y_limits[0]), float(y_limits[1])
    if ymax == ymin:
        ymax = ymin + 1.0

    def xmap(v: float) -> float:
        return left + (float(v) - tmin) * plot_w / (tmax - tmin)

    def ymap(v: float) -> float:
        return bottom - (float(v) - ymin) * plot_h / (ymax - ymin)

    svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#d0d0d0" stroke-width="1"/>')
    svg.append(_text(x0 + 8.0, y0 + 16.0, title, size=13, weight="bold"))
    svg.append(_line(left, bottom, right, bottom, color="#333", width=1.1))
    svg.append(_line(left, top, left, bottom, color="#333", width=1.1))
    for gv, gcolor, gdash in guide_lines:
        if gv < ymin or gv > ymax:
            continue
        gy = ymap(gv)
        svg.append(_line(left, gy, right, gy, color=gcolor, width=0.9, dash=gdash))

    xticks = np.asarray(times, dtype=np.int64)
    xlabels = xticks if xticks.size <= 7 else xticks[:: max(1, xticks.size // 6)]
    if xlabels[-1] != xticks[-1]:
        xlabels = np.append(xlabels, xticks[-1])
    for tv in xlabels:
        xx = xmap(float(tv))
        svg.append(_line(xx, bottom, xx, bottom + 5.0, color="#333", width=1.0))
        svg.append(_text(xx, bottom + 18.0, str(int(tv)), size=10, anchor="middle"))

    for frac in np.linspace(0.0, 1.0, 5):
        yv = ymin + frac * (ymax - ymin)
        yy_tick = ymap(yv)
        svg.append(_line(left - 4.0, yy_tick, left, yy_tick, color="#333", width=1.0))
        svg.append(_text(left - 8.0, yy_tick + 3.5, f"{yv:.3g}", size=10, anchor="end"))

    svg.append(_text((left + right) * 0.5, y0 + h - 8.0, "t", size=11, anchor="middle"))
    svg.append(_text(x0 + 15.0, y0 + h * 0.5, y_label, size=11, anchor="middle"))

    pts: List[Tuple[float, float]] = []
    for i, tv in enumerate(t):
        xx = xmap(float(tv))
        yyv = ymap(float(yy[i]))
        pts.append((xx, yyv))
        if ee is not None and np.isfinite(ee[i]):
            ylo = ymap(float(yy[i] - ee[i]))
            yhi = ymap(float(yy[i] + ee[i]))
            svg.append(_line(xx, ylo, xx, yhi, color=color, width=1.0))
            svg.append(_line(xx - 3.0, ylo, xx + 3.0, ylo, color=color, width=1.0))
            svg.append(_line(xx - 3.0, yhi, xx + 3.0, yhi, color=color, width=1.0))
    svg.append(_polyline(pts, color=color, width=2.0))
    for xx, yyv in pts:
        svg.append(_circle(xx, yyv, r=2.8, color=color))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot multilevel DD exact/corrected/bias/correlation panels as SVG")
    ap.add_argument("--input", required=True, help="checkpoint pickle produced by scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", type=str, default="pion_2pt_ml_dd")
    ap.add_argument("--exact-channel", type=str, default="exact")
    ap.add_argument("--corrected-channel", type=str, default="corrected")
    ap.add_argument("--bias-channel", type=str, default="bias")
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
    args = ap.parse_args()

    ckpt = Path(args.input).resolve()
    _, records = _load_checkpoint(ckpt)

    exact_by_p, exact_steps_by_p, exact_meta = _extract_complex_by_momentum(records, measurement=str(args.measurement), channel=str(args.exact_channel))
    corr_by_p, corr_steps_by_p, corr_meta = _extract_complex_by_momentum(records, measurement=str(args.measurement), channel=str(args.corrected_channel))
    bias_by_p, bias_steps_by_p, bias_meta = _extract_complex_by_momentum(records, measurement=str(args.measurement), channel=str(args.bias_channel))
    appr_by_p, appr_steps_by_p, appr_meta = _extract_complex_by_momentum(records, measurement=str(args.measurement), channel=str(args.approx_channel))

    selected = _parse_momenta(args.momenta)
    available = sorted(set(exact_by_p) & set(corr_by_p) & set(bias_by_p) & set(appr_by_p))
    momenta = available if selected is None else [p for p in selected if p in available]
    if not momenta:
        raise SystemExit("No common requested momenta were found in the checkpoint")

    exact_filt: Dict[int, np.ndarray] = {}
    corr_filt: Dict[int, np.ndarray] = {}
    bias_filt: Dict[int, np.ndarray] = {}
    appr_filt: Dict[int, np.ndarray] = {}
    step_filt: Dict[int, np.ndarray] = {}
    for p in momenta:
        ex_s, ex_st = _apply_discard_stride_complex(exact_by_p[p], exact_steps_by_p[p], discard=int(args.discard), stride=int(args.stride))
        co_s, co_st = _apply_discard_stride_complex(corr_by_p[p], corr_steps_by_p[p], discard=int(args.discard), stride=int(args.stride))
        bi_s, bi_st = _apply_discard_stride_complex(bias_by_p[p], bias_steps_by_p[p], discard=int(args.discard), stride=int(args.stride))
        ap_s, ap_st = _apply_discard_stride_complex(appr_by_p[p], appr_steps_by_p[p], discard=int(args.discard), stride=int(args.stride))
        steps, aligned = _align_many(
            {"exact": ex_st, "corrected": co_st, "bias": bi_st, "approx": ap_st},
            {"exact": ex_s, "corrected": co_s, "bias": bi_s, "approx": ap_s},
        )
        exact_filt[p] = aligned["exact"]
        corr_filt[p] = aligned["corrected"]
        bias_filt[p] = aligned["bias"]
        appr_filt[p] = aligned["approx"]
        step_filt[p] = steps

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

    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else ckpt.parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip() or ckpt.stem

    summary: Dict[str, object] = {
        "input_checkpoint": str(ckpt),
        "measurement": str(args.measurement),
        "block_size": int(block_size),
        "tau_ref": None if not np.isfinite(tau_ref) else float(tau_ref),
        "momenta": [int(p) for p in momenta],
        "exact_channel": str(args.exact_channel),
        "corrected_channel": str(args.corrected_channel),
        "bias_channel": str(args.bias_channel),
        "approx_channel": str(args.approx_channel),
        "per_momentum": {},
    }

    for p in momenta:
        ex_blocks = _block_means_complex(exact_filt[p], block_size=int(block_size))
        co_blocks = _block_means_complex(corr_filt[p], block_size=int(block_size))
        bi_blocks = _block_means_complex(bias_filt[p], block_size=int(block_size))
        ap_blocks = _block_means_complex(appr_filt[p], block_size=int(block_size))
        support = np.asarray(exact_meta.get(int(p), {}).get("support_times", list(range(int(ex_blocks.shape[1])))), dtype=np.int64)
        if int(support.size) != int(ex_blocks.shape[1]):
            raise SystemExit(f"Support-size mismatch for p={int(p)}")

        exact_mean = np.asarray(np.mean(ex_blocks.real, axis=0), dtype=np.float64)
        corr_mean = np.asarray(np.mean(co_blocks.real, axis=0), dtype=np.float64)
        bias_mean = np.asarray(np.mean(bi_blocks.real, axis=0), dtype=np.float64)
        exact_err = _stderr_from_blocks(np.asarray(ex_blocks.real, dtype=np.float64))
        corr_err = _stderr_from_blocks(np.asarray(co_blocks.real, dtype=np.float64))
        bias_err = _stderr_from_blocks(np.asarray(bi_blocks.real, dtype=np.float64))
        corr_t = np.asarray([_pearson_real(ap_blocks[:, i].real, ex_blocks[:, i].real) for i in range(ex_blocks.shape[1])], dtype=np.float64)

        ch_lo = float(np.min(np.concatenate([exact_mean - exact_err, corr_mean - corr_err])))
        ch_hi = float(np.max(np.concatenate([exact_mean + exact_err, corr_mean + corr_err])))
        common_limits = _nice_limits(ch_lo, ch_hi)
        bias_limits = _nice_limits(float(np.min(bias_mean - bias_err)), float(np.max(bias_mean + bias_err)))

        width = 1180
        height = 820
        top_margin = 70.0
        gap_x = 18.0
        gap_y = 18.0
        panel_w = (width - 3 * gap_x) / 2.0
        panel_h = (height - top_margin - 3 * gap_y) / 2.0
        top_y = top_margin
        bottom_y = top_margin + gap_y + panel_h

        svg = _svg_header(width, height)
        svg.append(_text(18.0, 24.0, f"{args.measurement} summary p={int(p)}", size=18, weight="bold"))
        svg.append(_text(18.0, 42.0, f"checkpoint={ckpt.name}  block={int(block_size)}  n_blocks={int(ex_blocks.shape[0])}", size=11))
        svg.append(_text(18.0, 58.0, f"level-0 mean corr={float(np.nanmean(corr_t)):.3f}  min corr={float(np.nanmin(corr_t)):.3f}", size=11))

        _plot_panel(
            svg,
            x0=gap_x,
            y0=top_y,
            w=panel_w,
            h=panel_h,
            times=support,
            y=exact_mean,
            err=exact_err,
            title="Exact (real part)",
            y_label="C_exact",
            color="#1f77b4",
            y_limits=common_limits,
        )
        _plot_panel(
            svg,
            x0=2 * gap_x + panel_w,
            y0=top_y,
            w=panel_w,
            h=panel_h,
            times=support,
            y=corr_mean,
            err=corr_err,
            title="Corrected (real part)",
            y_label="C_corr",
            color="#d62728",
            y_limits=common_limits,
        )
        _plot_panel(
            svg,
            x0=gap_x,
            y0=bottom_y,
            w=panel_w,
            h=panel_h,
            times=support,
            y=bias_mean,
            err=bias_err,
            title="Bias (real part)",
            y_label="B",
            color="#2ca02c",
            y_limits=bias_limits,
            guide_lines=[(0.0, "#777", "4,4")],
        )
        _plot_panel(
            svg,
            x0=2 * gap_x + panel_w,
            y0=bottom_y,
            w=panel_w,
            h=panel_h,
            times=support,
            y=corr_t,
            err=None,
            title="Per-timeslice corr(approx_l0, exact)",
            y_label="rho(t)",
            color="#9467bd",
            y_limits=(-1.0, 1.0),
            guide_lines=[(0.9, "#999", "4,4"), (0.5, "#bbb", "3,5"), (0.0, "#777", "4,4")],
        )
        svg.extend(_svg_footer())

        svg_path = outdir / f"{prefix}_{args.measurement}_p{int(p)}_summary.svg"
        svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

        summary["per_momentum"][str(int(p))] = {
            "steps_first": int(step_filt[p][0]),
            "steps_last": int(step_filt[p][-1]),
            "n_common_samples": int(step_filt[p].size),
            "n_blocks": int(ex_blocks.shape[0]),
            "support_times": support.astype(int).tolist(),
            "exact_mean_re": exact_mean.tolist(),
            "exact_err_re": exact_err.tolist(),
            "corrected_mean_re": corr_mean.tolist(),
            "corrected_err_re": corr_err.tolist(),
            "bias_mean_re": bias_mean.tolist(),
            "bias_err_re": bias_err.tolist(),
            "corr_l0_exact_re": corr_t.tolist(),
            "mean_corr_l0_exact_re": float(np.nanmean(corr_t)),
            "min_corr_l0_exact_re": float(np.nanmin(corr_t)),
        }

    json_path = outdir / f"{prefix}_{args.measurement}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"saved json:   {json_path}")
    print(f"block size: {int(block_size)}" + ("" if not np.isfinite(tau_ref) else f" (auto tau_ref~{float(tau_ref):.2f})"))
    for p in momenta:
        item = summary["per_momentum"][str(int(p))]
        print(
            f"  p={int(p)}: mean corr={float(item['mean_corr_l0_exact_re']):.3f} "
            f"min corr={float(item['min_corr_l0_exact_re']):.3f} "
            f"svg={outdir / f'{prefix}_{args.measurement}_p{int(p)}_summary.svg'}"
        )


if __name__ == "__main__":
    main()

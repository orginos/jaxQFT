#!/usr/bin/env python3
"""Inspect momentum-resolved two-point correlators from an MCMC checkpoint.

Produces:
1) a semilog-y plot of the mean correlator by momentum
2) a semilog-y plot of signal-to-noise, |mean|/sigma
3) a JSON summary with basic per-momentum diagnostics
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

from fit_2pt_dispersion import _apply_discard_stride, _auto_block_size_from_p0, _block_means, _load_checkpoint


def _extract_complex_by_momentum(
    records: Sequence[Mapping],
    *,
    measurement: str,
    channel: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    import re

    chan = str(channel)
    pat_re = re.compile(rf"^{re.escape(chan)}_p(?P<p>-?\d+)_t(?P<t>\d+)_re$")
    pat_im = re.compile(rf"^{re.escape(chan)}_p(?P<p>-?\d+)_t(?P<t>\d+)_im$")
    pat_valid = re.compile(rf"^{re.escape(chan)}_valid_t(?P<t>\d+)$")
    legacy_re = re.compile(rf"^{re.escape(chan)}_t(?P<t>\d+)$") if chan in ("c", "full", "conn", "disc") else None

    by_p_step: Dict[int, Dict[int, Dict[int, complex]]] = {}
    imag_abs_max: Dict[int, float] = {}
    valid_times_by_step: Dict[int, set[int]] = {}

    for rec in records:
        if str(rec.get("name", "")) != str(measurement):
            continue
        vals = rec.get("values", {})
        if not isinstance(vals, Mapping):
            continue
        step = int(rec.get("step", -1))
        for key, raw in vals.items():
            k = str(key)
            m_valid = pat_valid.match(k)
            if m_valid is not None:
                valid_times_by_step.setdefault(step, set()).add(int(m_valid.group("t")))
                continue
            m_re = pat_re.match(k)
            if m_re is not None:
                p = int(m_re.group("p"))
                t = int(m_re.group("t"))
                entry = by_p_step.setdefault(p, {}).setdefault(step, {})
                entry[t] = complex(float(raw), float(np.imag(entry.get(t, 0.0))))
                continue

            if legacy_re is not None:
                m_legacy = legacy_re.match(k)
                if m_legacy is not None:
                    p = 0
                    t = int(m_legacy.group("t"))
                    entry = by_p_step.setdefault(p, {}).setdefault(step, {})
                    entry[t] = complex(float(raw), float(np.imag(entry.get(t, 0.0))))
                    continue

            m_im = pat_im.match(k)
            if m_im is not None:
                p = int(m_im.group("p"))
                t = int(m_im.group("t"))
                entry = by_p_step.setdefault(p, {}).setdefault(step, {})
                entry[t] = complex(float(np.real(entry.get(t, 0.0))), float(raw))
                imag_abs_max[p] = max(float(imag_abs_max.get(p, 0.0)), abs(float(raw)))

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
            row = np.full((len(support_times),), np.nan + 1j * np.nan, dtype=np.complex128)
            vals_step = by_step[step]
            for i, t in enumerate(support_times):
                if int(t) in vals_step:
                    row[i] = complex(vals_step[int(t)])
            if np.all(np.isfinite(row.real)) and np.all(np.isfinite(row.imag)):
                rows.append(row)
                kept_steps.append(int(step))
            else:
                dropped += 1
        if not rows:
            continue
        samples_by_p[p] = np.asarray(rows, dtype=np.complex128)
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


def _block_means_complex(samples: np.ndarray, block_size: int) -> np.ndarray:
    xr = _block_means(np.asarray(samples.real, dtype=np.float64), block_size=int(block_size))
    xi = _block_means(np.asarray(samples.imag, dtype=np.float64), block_size=int(block_size))
    return np.asarray(xr + 1j * xi, dtype=np.complex128)


def _stderr_from_blocks(blocks: np.ndarray) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"Expected 2D blocks, got shape {b.shape}")
    n = int(b.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 blocks, got {n}")
    return np.asarray(np.std(b, axis=0, ddof=1) / math.sqrt(float(n)), dtype=np.float64)


def _parse_momenta(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None or str(text).strip() == "":
        return None
    toks = [t.strip() for t in str(text).split(",") if t.strip()]
    return tuple(int(t) for t in toks) if toks else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect momentum-resolved 2pt correlators")
    ap.add_argument("--input", required=True, help="checkpoint pickle produced by scripts/mcmc/mcmc.py")
    ap.add_argument("--measurement", type=str, default="pion_2pt")
    ap.add_argument("--channel", type=str, default="c")
    ap.add_argument("--momenta", type=str, default="", help="comma-separated subset; default uses all available")
    ap.add_argument("--discard", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=0, help="<=0: auto from p=0 real-part IAT")
    ap.add_argument("--iat-method", type=str, default="gamma", choices=["ips", "sokal", "gamma"])
    ap.add_argument("--iat-c", type=float, default=5.0)
    ap.add_argument("--iat-max-lag", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--eps", type=float, default=1e-15, help="floor for log plots and relative errors")
    ap.add_argument(
        "--snr-thresholds",
        type=str,
        default="10,1",
        help="comma-separated S/N guide lines and cut-report thresholds",
    )
    args = ap.parse_args()

    snr_thresholds = []
    for tok in [t.strip() for t in str(args.snr_thresholds).split(",") if t.strip()]:
        val = float(tok)
        if val > 0.0:
            snr_thresholds.append(val)
    if not snr_thresholds:
        snr_thresholds = [10.0, 1.0]
    snr_thresholds = sorted(set(float(v) for v in snr_thresholds), reverse=True)

    ckpt = Path(args.input).resolve()
    payload, records = _load_checkpoint(ckpt)
    samples_by_p, steps_by_p, meta_by_p = _extract_complex_by_momentum(
        records,
        measurement=str(args.measurement),
        channel=str(args.channel),
    )

    selected = _parse_momenta(args.momenta)
    momenta = sorted(samples_by_p.keys()) if selected is None else [p for p in selected if p in samples_by_p]
    if not momenta:
        raise SystemExit("No requested momenta were found in the checkpoint")

    filt_samples: Dict[int, np.ndarray] = {}
    filt_steps: Dict[int, np.ndarray] = {}
    for p in momenta:
        s, st = _apply_discard_stride(samples_by_p[p], steps_by_p[p], discard=int(args.discard), stride=int(args.stride))
        if int(s.shape[0]) < 2:
            raise SystemExit(f"Not enough samples after discard/stride for p={p}: {s.shape[0]}")
        filt_samples[p] = np.asarray(s, dtype=np.complex128)
        filt_steps[p] = np.asarray(st, dtype=np.int64)

    if int(args.block_size) > 0:
        block_size = int(args.block_size)
        tau_ref = float("nan")
    else:
        real_samples_for_auto = {int(p): np.asarray(v.real, dtype=np.float64) for p, v in filt_samples.items()}
        block_size, tau_ref = _auto_block_size_from_p0(
            real_samples_for_auto,
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            iat_max_lag=(None if int(args.iat_max_lag) <= 0 else int(args.iat_max_lag)),
        )

    summary: Dict[str, object] = {
        "input_checkpoint": str(ckpt),
        "measurement": str(args.measurement),
        "channel": str(args.channel),
        "momenta": [int(p) for p in momenta],
        "block_size": int(block_size),
        "tau_ref": None if not np.isfinite(tau_ref) else float(tau_ref),
        "discard": int(args.discard),
        "stride": int(args.stride),
        "per_momentum": {},
    }

    per_p: Dict[int, Dict[str, np.ndarray | float | int]] = {}
    for p in momenta:
        blocks = _block_means_complex(filt_samples[p], block_size=int(block_size))
        mean = np.asarray(np.mean(blocks, axis=0), dtype=np.complex128)
        support_times = np.asarray(meta_by_p.get(int(p), {}).get("support_times", list(range(int(mean.size)))), dtype=np.int64)
        if int(support_times.size) != int(mean.size):
            raise SystemExit(
                f"Support-size mismatch for p={int(p)}: meta has {int(support_times.size)} times, mean has {int(mean.size)} entries"
            )
        err_re = _stderr_from_blocks(np.asarray(blocks.real, dtype=np.float64))
        err_im = _stderr_from_blocks(np.asarray(blocks.imag, dtype=np.float64))
        abs_re = np.abs(mean.real)
        abs_im = np.abs(mean.imag)
        snr_re = np.where(err_re > float(args.eps), abs_re / err_re, np.inf)
        imag_over_re = np.where(abs_re > float(args.eps), abs_im / abs_re, np.nan)
        sign_changes = int(np.sum((mean.real[:-1] * mean.real[1:]) < 0.0))
        first_negative = next((int(support_times[i]) for i in range(mean.size) if mean.real[i] < 0.0), None)
        snr_cutoffs = {}
        for thr in snr_thresholds:
            below = np.where(np.isfinite(snr_re) & (snr_re < float(thr)))[0]
            snr_cutoffs[str(int(thr) if float(thr).is_integer() else thr)] = (
                None if below.size == 0 else int(support_times[int(below[0])])
            )
        per_p[int(p)] = {
            "mean": mean,
            "err_re": err_re,
            "err_im": err_im,
            "snr_re": snr_re,
            "imag_over_re": imag_over_re,
            "support_times": support_times,
        }
        summary["per_momentum"][str(int(p))] = {
            "n_samples": int(filt_samples[p].shape[0]),
            "n_blocks": int(blocks.shape[0]),
            "t_extent": int(mean.size),
            "support_times": support_times.astype(int).tolist(),
            "first_negative_re_t": first_negative,
            "sign_changes_re": int(sign_changes),
            "max_abs_re": float(np.max(abs_re)),
            "max_abs_im": float(np.max(abs_im)),
            "max_imag_over_re": float(np.nanmax(imag_over_re)) if np.any(np.isfinite(imag_over_re)) else float("nan"),
            "min_snr_re": float(np.nanmin(snr_re)) if np.any(np.isfinite(snr_re)) else float("nan"),
            "max_snr_re": float(np.nanmax(snr_re)) if np.any(np.isfinite(snr_re)) else float("nan"),
            "first_snr_below": snr_cutoffs,
            "steps_first": int(filt_steps[p][0]),
            "steps_last": int(filt_steps[p][-1]),
            "meta": dict(meta_by_p.get(int(p), {})),
        }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}")

    n_mom = len(momenta)
    fig, axes = plt.subplots(2, n_mom, figsize=(4.2 * n_mom, 6.8), sharex="col", squeeze=False)
    eps = float(args.eps)
    for ip, p in enumerate(momenta):
        ax_top = axes[0, ip]
        ax_bot = axes[1, ip]
        dat = per_p[int(p)]
        mean = np.asarray(dat["mean"])
        t = np.asarray(dat["support_times"], dtype=np.int64)
        abs_re = np.maximum(np.abs(mean.real), eps)
        abs_im = np.maximum(np.abs(mean.imag), eps)
        snr_re = np.asarray(dat["snr_re"], dtype=np.float64)

        ax_top.semilogy(t, abs_re, marker="o", ms=3, lw=1.5, label="|Re C|")
        if np.any(np.abs(mean.imag) > eps):
            ax_top.semilogy(t, abs_im, marker="s", ms=2.5, lw=1.0, ls="--", label="|Im C|")
        neg = mean.real < 0.0
        if np.any(neg):
            ax_top.scatter(t[neg], abs_re[neg], marker="x", s=26, color="crimson", label="Re<0")
        ax_top.set_title(f"p={int(p)}")
        ax_top.grid(alpha=0.25)
        if ip == 0:
            ax_top.set_ylabel(r"$|C_p(t)|$")
        ax_top.legend(fontsize=8, loc="best")

        finite_s = np.isfinite(snr_re) & (snr_re > 0.0)
        ax_bot.semilogy(t[finite_s], np.maximum(snr_re[finite_s], eps), marker="o", ms=3, lw=1.5, label=r"$|{\rm Re}\,C|/\sigma_{\mathrm{Re}}$")
        for thr in snr_thresholds:
            ax_bot.axhline(float(thr), color="0.5", lw=0.9, ls="--", alpha=0.9)
            below = np.where(np.isfinite(snr_re) & (snr_re < float(thr)))[0]
            if below.size > 0:
                ax_bot.axvline(int(t[int(below[0])]), color="0.7", lw=0.8, ls=":", alpha=0.8)
        ax_bot.grid(alpha=0.25)
        if ip == 0:
            ax_bot.set_ylabel("S/N")
        ax_bot.set_xlabel("t")
        ax_bot.legend(fontsize=8, loc="best")

    block_txt = f"block={block_size}" + (f", tau_ref~{tau_ref:.2f}" if np.isfinite(tau_ref) else "")
    fig.suptitle(f"{args.measurement}/{args.channel} from {ckpt.name} | {block_txt}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else ckpt.parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip() or ckpt.stem
    fig_path = outdir / f"{prefix}_{args.measurement}_{args.channel}_inspect.png"
    json_path = outdir / f"{prefix}_{args.measurement}_{args.channel}_inspect.json"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"saved figure: {fig_path}")
    print(f"saved json:   {json_path}")
    print(f"block size: {block_size}" + (f" (auto tau_ref~{tau_ref:.2f})" if np.isfinite(tau_ref) else ""))
    for p in momenta:
        info = summary["per_momentum"][str(int(p))]
        cutoff_txt = ", ".join(
            f"S/N<{k}@t={v}" for k, v in info["first_snr_below"].items()
        )
        print(
            f"  p={int(p)}: sign_changes_re={int(info['sign_changes_re'])}"
            f" first_negative_re_t={info['first_negative_re_t']}"
            f" max|Im|/|Re|={float(info['max_imag_over_re']):.3e}"
            f" min S/N={float(info['min_snr_re']):.3e}"
            f" | {cutoff_txt}"
        )


if __name__ == "__main__":
    main()

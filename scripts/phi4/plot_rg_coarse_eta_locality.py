#!/usr/bin/env python3
"""Publication-quality locality plots for the RG coarse-eta Gaussian flow.

Reads JSON output from `analyze_rg_coarse_eta_gaussian_flow.py --tests locality`
and produces PRD-style PDF figures.

Examples:
  python scripts/phi4/plot_rg_coarse_eta_locality.py \
    --json runs/phi4/paper-2/locality/per-level-run-0/L16/locality_L16_model_manhattan.json \
    --json runs/phi4/paper-2/locality/per-level-run-0/L32/locality_L32_model_manhattan.json \
    --json runs/phi4/paper-2/locality/per-level-run-0/L64/locality_L64_model_manhattan.json \
    --json runs/phi4/paper-2/locality/per-level-run-0/L128/locality_L128_model_manhattan.json \
    --outdir docs/papers/paper-2/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRD_SINGLE = 3.375
PRD_DOUBLE = 6.75


def _set_prd_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def _load_locality(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    loc = raw["locality"]
    fit = loc["fit"]
    shape = tuple(int(v) for v in loc["shape"])
    L = int(shape[0])
    return {
        "path": path,
        "label": f"$L={L}$",
        "L": L,
        "metric": loc["metric"],
        "radii": np.asarray(loc["radii"], dtype=np.float64),
        "count": np.asarray(loc["shell_count"], dtype=np.int64),
        "mean_norm": np.asarray(loc["mean_abs_response_norm"], dtype=np.float64),
        "rms_norm": np.asarray(loc["rms_response_norm"], dtype=np.float64),
        "fit": fit,
    }


def _fit_curve(entry: dict) -> tuple[np.ndarray, np.ndarray] | None:
    fit = entry["fit"]
    if not fit.get("ok", False):
        return None
    x = entry["radii"]
    mask = x >= float(fit["fit_rmin"])
    if float(fit["fit_rmax"]) > 0.0:
        mask &= x <= float(fit["fit_rmax"])
    xfit = x[mask]
    if xfit.size == 0:
        return None
    yfit = np.exp(float(fit["intercept"]) + float(fit["slope"]) * xfit)
    return xfit, yfit


def make_overlay(entries: list[dict], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(PRD_DOUBLE, 0.62 * PRD_DOUBLE))
    colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(entries)))

    for color, entry in zip(colors, sorted(entries, key=lambda e: e["L"])):
        x = entry["radii"]
        y = entry["mean_norm"]
        mask = entry["count"] > 0
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            color=color,
            label=f"{entry['label']}, $\\xi_{{\\mathrm{{loc}}}}={entry['fit'].get('xi_local', float('nan')):.2f}$",
        )
        fit_curve = _fit_curve(entry)
        if fit_curve is not None:
            xfit, yfit = fit_curve
            ax.plot(xfit, yfit, color=color, linestyle="--", linewidth=1.0)

    ax.set_yscale("log")
    ax.set_xlabel(r"Manhattan distance $r$")
    ax.set_ylabel(r"$\langle |H_{xy}| \rangle / \langle |H_{xx}| \rangle$")
    ax.set_title("Locality of the learned action")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(5e-5, 2.0)
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()

    out = outdir / "phi4_rg_locality_decay_overlay.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_summary(entries: list[dict], outdir: Path) -> Path:
    entries = sorted(entries, key=lambda e: e["L"])
    L = np.asarray([e["L"] for e in entries], dtype=np.float64)
    xi = np.asarray([float(e["fit"].get("xi_local", np.nan)) for e in entries], dtype=np.float64)
    xi_over_L = xi / L

    fig, axes = plt.subplots(1, 2, figsize=(PRD_DOUBLE, 0.52 * PRD_DOUBLE))

    axes[0].plot(L, xi, marker="o", color="C0")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel(r"Lattice size $L$")
    axes[0].set_ylabel(r"Locality length $\xi_{\mathrm{loc}}$")
    axes[0].set_title(r"$\xi_{\mathrm{loc}}$ from exponential fit")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(L, xi_over_L, marker="s", color="C1")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel(r"Lattice size $L$")
    axes[1].set_ylabel(r"$\xi_{\mathrm{loc}}/L$")
    axes[1].set_title(r"Relative locality length")
    axes[1].grid(True, which="both", alpha=0.25)

    for ax in axes:
        ax.set_xticks(L, [str(int(v)) for v in L])

    fig.tight_layout()
    out = outdir / "phi4_rg_locality_length_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_table(entries: list[dict], outdir: Path) -> Path:
    entries = sorted(entries, key=lambda e: e["L"])
    lines = []
    lines.append("% Auto-generated by scripts/phi4/plot_rg_coarse_eta_locality.py")
    lines.append("\\begin{tabular}{rcc}")
    lines.append("\\toprule")
    lines.append("$L$ & $\\xi_{\\mathrm{loc}}$ & $\\xi_{\\mathrm{loc}}/L$ \\\\")
    lines.append("\\midrule")
    for e in entries:
        xi = float(e["fit"].get("xi_local", float("nan")))
        lines.append(f"{e['L']} & {xi:.3f} & {xi / e['L']:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out = outdir / "phi4_rg_locality_table.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot locality diagnostics for the RG coarse-eta Gaussian flow.")
    ap.add_argument("--json", action="append", required=True, help="locality JSON file; may be repeated")
    ap.add_argument("--outdir", type=str, required=True, help="output directory for PDF figures")
    args = ap.parse_args()

    _set_prd_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = [_load_locality(path) for path in args.json]

    overlay = make_overlay(entries, outdir)
    summary = make_summary(entries, outdir)
    table = make_table(entries, outdir)

    print(f"Saved: {overlay}")
    print(f"Saved: {summary}")
    print(f"Saved: {table}")


if __name__ == "__main__":
    main()

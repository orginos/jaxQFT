#!/usr/bin/env python3
"""Debug comparison plot for standalone and level-analysis locality fits.

This is intended for investigating discrepancies between the older
`analyze_rg_coarse_eta_gaussian_flow.py --tests locality` outputs and the newer
`analyze_rg_coarse_eta_gaussian_levels.py --locality` outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_standalone(path: str) -> dict:
    raw = json.load(open(path))
    loc = raw["locality"]
    return {
        "kind": "standalone",
        "radii": np.asarray(loc["radii"], dtype=np.float64),
        "mean": np.asarray(loc["mean_abs_response_norm"], dtype=np.float64),
        "count": np.asarray(loc["shell_count"], dtype=np.int64),
        "fit": dict(loc["fit"]),
    }


def _load_levels(path: str, level_index: int) -> dict:
    raw = json.load(open(path))
    loc = raw["locality"]["levels"][level_index]
    return {
        "kind": "levels",
        "radii": np.asarray(loc["radii"], dtype=np.float64),
        "mean": np.asarray(loc["mean_abs_response_norm"], dtype=np.float64),
        "count": np.asarray(loc["shell_count"], dtype=np.int64),
        "fit": dict(loc["fit"]),
    }


def _fit_curve(entry: dict) -> tuple[np.ndarray, np.ndarray] | None:
    fit = entry["fit"]
    if not fit.get("ok", False):
        return None
    x = entry["radii"]
    mask = (x >= float(fit["fit_rmin"])) & (entry["count"] > 0)
    rmax = float(fit.get("fit_rmax", 0.0))
    if rmax > 0.0:
        mask &= x <= rmax
    xfit = x[mask]
    if xfit.size == 0:
        return None
    yfit = np.exp(float(fit["intercept"]) + float(fit["slope"]) * xfit)
    return xfit, yfit


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two locality decay curves and fit windows.")
    ap.add_argument("--standalone-json", required=True)
    ap.add_argument("--levels-json", required=True)
    ap.add_argument("--level-index", type=int, default=0)
    ap.add_argument("--label-a", type=str, default="Old standalone")
    ap.add_argument("--label-b", type=str, default="New level analysis")
    ap.add_argument("--title", type=str, default="Locality fit comparison")
    ap.add_argument("--out", required=True, help="Output image/PDF path")
    args = ap.parse_args()

    a = _load_standalone(args.standalone_json)
    b = _load_levels(args.levels_json, int(args.level_index))

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    ax, ax_zoom = axes

    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"Manhattan distance $r$")
        axis.set_ylabel(r"$\langle |H_{xy}| \rangle / \langle |H_{xx}| \rangle$")

    for entry, color, label in [(a, "C0", args.label_a), (b, "C1", args.label_b)]:
        mask = entry["count"] > 0
        x = entry["radii"][mask]
        y = entry["mean"][mask]
        fit = entry["fit"]
        fit_curve = _fit_curve(entry)
        fit_label = f"{label}: " + r"$\xi_{\rm loc}$" + f"={fit.get('xi_local', float('nan')):.2f}"
        ax.plot(x, y, marker="o", ms=2.5, lw=1.0, color=color, label=fit_label)
        ax_zoom.plot(x, y, marker="o", ms=2.5, lw=1.0, color=color, label=fit_label)
        if fit_curve is not None:
            xfit, yfit = fit_curve
            ax.plot(xfit, yfit, "--", lw=1.2, color=color)
            ax_zoom.plot(xfit, yfit, "--", lw=1.2, color=color)
        ax.axvline(float(fit["fit_rmin"]), color=color, ls=":", lw=0.9)
        ax_zoom.axvline(float(fit["fit_rmin"]), color=color, ls=":", lw=0.9)
        rmax = float(fit.get("fit_rmax", 0.0))
        if rmax > 0.0:
            ax.axvline(rmax, color=color, ls=":", lw=0.9)
            ax_zoom.axvline(rmax, color=color, ls=":", lw=0.9)

    ax.set_title(args.title + " (full range)")
    ax_zoom.set_title(args.title + " (zoom)")
    ax.set_xlim(0, max(float(a["radii"][-1]), float(b["radii"][-1])))
    ax_zoom.set_xlim(0, 20)
    ax.set_ylim(3e-7, 2.0)
    ax_zoom.set_ylim(3e-5, 2.0)
    ax.legend(loc="upper right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

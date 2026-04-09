#!/usr/bin/env python3
"""Plot multiscale locality diagnostics from flow level-analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_level(path: str, metric: str, level_index: int) -> tuple[dict, dict]:
    raw = json.load(open(path))
    locality = raw["locality"]
    metrics = locality.get("metrics") or {locality["metric"]: locality}
    payload = metrics[metric]
    row = payload["levels"][level_index]
    level = raw["levels"][level_index]
    return row, level


def _fit_curve(fit: dict, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not fit.get("ok", False):
        return None
    mask = radii >= float(fit["fit_rmin"])
    rmax = float(fit.get("fit_rmax", 0.0))
    if rmax > 0.0:
        mask &= radii <= rmax
    xx = radii[mask]
    if xx.size == 0:
        return None
    yy = np.exp(float(fit["intercept"]) + float(fit["slope"]) * xx)
    return xx, yy


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot multiscale locality diagnostics for one RG level.")
    ap.add_argument("--json", required=True)
    ap.add_argument("--metric", default="manhattan")
    ap.add_argument("--level-index", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row, level = _load_level(args.json, args.metric, args.level_index)
    radii = np.asarray(row["radii"], dtype=np.float64)
    mean_norm = np.asarray(row["mean_abs_response_norm"], dtype=np.float64)
    shell_weight = np.asarray(row["shell_integrated_abs_response_norm"], dtype=np.float64)
    tail = np.asarray(row["tail_integrated_abs_response_norm"], dtype=np.float64)
    L = int(row["shape"][0])
    xi2 = float(level["unweighted"]["xi2"]["mean"])

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
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), constrained_layout=True)
    ax0, ax1 = axes

    ax0.plot(radii, mean_norm, "o-", color="C0", lw=1.0, ms=2.8, label="shell mean")
    for fit, color, name in [
        (row.get("fit_lquarter", {}), "C1", r"primary ($L/4$)"),
        (row.get("fit_full", {}), "C3", "full range"),
    ]:
        curve = _fit_curve(fit, radii)
        if curve is not None:
            xx, yy = curve
            ax0.plot(xx, yy, "--", color=color, lw=1.1, label=f"{name}: " + r"$\xi_{\rm loc}$" + f"={fit['xi_local']:.2f}")
    for fit in row.get("dyadic_fits", []):
        curve = _fit_curve(fit, radii)
        if curve is not None:
            xx, yy = curve
            ax0.plot(xx, yy, "-", color="0.6", lw=0.8, alpha=0.55)
    ax0.set_yscale("log")
    ax0.set_xlabel(rf"{args.metric} distance $r$")
    ax0.set_ylabel(r"$\langle |H_{xy}| \rangle / \langle |H_{xx}| \rangle$")
    title = args.title or f"L={L}, level {row['level_from_fine']}, " + r"$\xi_2$" + f"={xi2:.2f}"
    ax0.set_title(title)
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend(loc="upper right")

    ax1.plot(radii, shell_weight, "o-", color="C2", lw=1.0, ms=2.5, label="shell weight")
    ax1.plot(radii, tail, "-", color="C4", lw=1.2, label="cumulative tail")
    ax1.axvline(float(L) / 4.0, color="0.4", ls=":", lw=0.9, label=r"$L/4$")
    ax1.set_yscale("log")
    ax1.set_xlabel(rf"{args.metric} distance $r$")
    ax1.set_ylabel("integrated abs weight (norm.)")
    ax1.set_title("Tail-weight diagnostics")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

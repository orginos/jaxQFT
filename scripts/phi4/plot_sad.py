#!/usr/bin/env python3
"""Plot SAD vs standard estimator results.

Reads JSON output from sad_phi4.py and produces publication-quality figures:
  - Correlator C(t) with error bands
  - Cosh effective mass plateau
  - Signal-to-noise ratio comparison
  - Error improvement bar chart

Usage:
  python scripts/phi4/plot_sad.py
  python scripts/phi4/plot_sad.py --phi4-json path/to/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Free-field plots
# ---------------------------------------------------------------------------

def make_plots_free(data: dict, outdir: Path) -> list[str]:
    T, L = data["shape"]
    Thalf = T // 2
    t = np.arange(T)

    C_sad   = np.array(data["C_sad_mean"])
    C_std   = np.array(data["C_std_mean"])
    C_sad_e = np.array(data["C_sad_err"])
    C_std_e = np.array(data["C_std_err"])
    C_exact = np.array(data["C_exact"])

    mc_sad   = np.array(data["mcosh_sad"])
    mc_std   = np.array(data["mcosh_std"])
    mc_sad_e = np.array(data["mcosh_sad_err"])
    mc_std_e = np.array(data["mcosh_std_err"])
    mc_exact = np.array(data.get("mcosh_exact", [np.nan] * Thalf))

    m2, lam = data["m2"], data["lam"]
    B, N = data["batch_size"], data["nmeas"]

    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Free field: $m^2={m2}$, $\\lambda={lam}$, "
                 f"${T}\\times{L}$ lattice, $B={B}$, $N_{{\\mathrm{{meas}}}}={N}$",
                 fontsize=13)

    # -- Left: C(t) log scale --
    ax = axes[0]
    idx = slice(0, Thalf + 1)
    ax.errorbar(t[idx], C_std[idx], yerr=C_std_e[idx], fmt='s', ms=5,
                capsize=3, color='C1', label='Standard', alpha=0.8, zorder=2)
    ax.errorbar(t[idx], C_sad[idx], yerr=np.maximum(C_sad_e[idx], 1e-30), fmt='o', ms=5,
                capsize=3, color='C0', label='SAD', zorder=3)
    ax.plot(t[idx], C_exact[idx], 'k--', lw=1.5, label='Exact', zorder=1)
    ax.set_yscale('log')
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$C(t)$', fontsize=12)
    ax.set_title('Correlator $C(t)$')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, Thalf + 0.5)
    ax.grid(True, alpha=0.3)

    # -- Right: cosh effective mass --
    ax = axes[1]
    t_m = np.arange(Thalf) + 0.5
    ax.errorbar(t_m, mc_std, yerr=mc_std_e, fmt='s', ms=5,
                capsize=3, color='C1', label='Standard', alpha=0.8)
    valid = mc_sad_e > 0
    ax.errorbar(t_m[valid], mc_sad[valid], yerr=mc_sad_e[valid], fmt='o', ms=5,
                capsize=3, color='C0', label='SAD')
    if not valid.all():
        ax.plot(t_m[~valid], mc_sad[~valid], 'o', ms=5, color='C0',
                label='SAD (zero err)', zorder=4)
    ax.plot(t_m, mc_exact, 'k--', lw=1.5, label='Exact')
    ax.set_xlabel('$t + 1/2$', fontsize=12)
    ax.set_ylabel('$m_{\\mathrm{eff}}^{\\cosh}(t)$', fontsize=12)
    ax.set_title('Cosh effective mass')
    ax.legend(fontsize=10, loc='best')
    ylo = max(0, np.nanmin(mc_exact) - 0.2)
    yhi = np.nanmax(mc_exact[mc_exact < 5]) + 0.5 if np.any(mc_exact < 5) else 3.0
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.3, Thalf + 0.3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = outdir / "sad_free_correlator.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(str(p))
    print(f"  Saved: {p}")
    return saved


# ---------------------------------------------------------------------------
# phi^4 plots
# ---------------------------------------------------------------------------

def make_plots_phi4(data: dict, outdir: Path, tag: str = "") -> list[str]:
    T, L = data["shape"]
    Thalf = T // 2
    t = np.arange(T)

    C_sad   = np.array(data["C_sad_mean"])
    C_std   = np.array(data["C_std_mean"])
    C_sad_e = np.array(data["C_sad_err"])
    C_std_e = np.array(data["C_std_err"])

    mc_sad   = np.array(data["mcosh_sad"])
    mc_std   = np.array(data["mcosh_std"])
    mc_sad_e = np.array(data["mcosh_sad_err"])
    mc_std_e = np.array(data["mcosh_std_err"])

    m2, lam = data["m2"], data["lam"]
    B, N = data["batch_size"], data["nmeas"]
    nmd, tau, gamma = data["nmd"], data["tau"], data["gamma"]
    dt_val = data.get("dt", tau / nmd)

    saved = []
    sfx = f"_{tag}" if tag else ""

    # ====== Figure 1: 4-panel overview ======
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)
    fig.suptitle(f"SAD vs Standard:  $\\phi^4$, $m^2={m2}$, $\\lambda={lam}$, "
                 f"${T}\\times{L}$\n"
                 f"$B={B}$ chains, $N={N}$, SMD($\\gamma={gamma}$, "
                 f"$\\tau={tau}$, $n_{{md}}={nmd}$, $\\delta\\tau={dt_val:.3f}$)",
                 fontsize=13)

    # -- Panel 1: C(t) linear --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(t, C_std, yerr=C_std_e, fmt='s', ms=4, capsize=2,
                 color='C1', label='Standard', alpha=0.8, zorder=2)
    ax1.errorbar(t, C_sad, yerr=C_sad_e, fmt='o', ms=4, capsize=2,
                 color='C0', label='SAD', zorder=3)
    ax1.set_xlabel('$t$', fontsize=12)
    ax1.set_ylabel('$C(t)$', fontsize=12)
    ax1.set_title('Correlator $C(t)$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # -- Panel 2: C(t) log scale first half --
    ax2 = fig.add_subplot(gs[0, 1])
    idx = slice(0, Thalf + 1)
    ax2.errorbar(t[idx], C_std[idx], yerr=C_std_e[idx], fmt='s', ms=4,
                 capsize=2, color='C1', label='Standard', alpha=0.8)
    ax2.errorbar(t[idx], C_sad[idx], yerr=C_sad_e[idx], fmt='o', ms=4,
                 capsize=2, color='C0', label='SAD')
    ax2.set_yscale('log')
    ax2.set_xlabel('$t$', fontsize=12)
    ax2.set_ylabel('$C(t)$', fontsize=12)
    ax2.set_title('Correlator $C(t)$ [log scale]', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.5, Thalf + 0.5)
    ax2.grid(True, alpha=0.3)

    # -- Panel 3: Cosh effective mass --
    ax3 = fig.add_subplot(gs[1, 0])
    t_m = np.arange(Thalf) + 0.5
    valid_sad = np.isfinite(mc_sad) & (mc_sad_e > 0)
    valid_std = np.isfinite(mc_std) & (mc_std_e > 0)
    ax3.errorbar(t_m[valid_std], mc_std[valid_std], yerr=mc_std_e[valid_std],
                 fmt='s', ms=4, capsize=2, color='C1', label='Standard', alpha=0.8)
    ax3.errorbar(t_m[valid_sad], mc_sad[valid_sad], yerr=mc_sad_e[valid_sad],
                 fmt='o', ms=4, capsize=2, color='C0', label='SAD')
    # Mass estimate from plateau (middle region of valid SAD)
    plat_lo, plat_hi = max(1, Thalf // 4), 3 * Thalf // 4
    plat_mask = valid_sad.copy()
    plat_mask[:plat_lo] = False
    plat_mask[plat_hi:] = False
    if plat_mask.any():
        m_plat = np.average(mc_sad[plat_mask],
                            weights=1.0 / mc_sad_e[plat_mask] ** 2)
        ax3.axhline(m_plat, color='C0', ls=':', lw=1, alpha=0.6,
                    label=f'$m \\approx {m_plat:.4f}$')
    ax3.set_xlabel('$t + 1/2$', fontsize=12)
    ax3.set_ylabel('$m_{\\mathrm{eff}}^{\\cosh}(t)$', fontsize=12)
    ax3.set_title('Cosh effective mass', fontsize=12)
    ax3.legend(fontsize=9, loc='best')
    # Auto y-range from SAD data
    if valid_sad.any():
        yvals = mc_sad[valid_sad]
        ylo = max(0, np.nanmin(yvals) - 0.05)
        yhi = np.nanmax(yvals) + 0.05
        ax3.set_ylim(ylo, yhi)
    ax3.set_xlim(-0.3, Thalf + 0.3)
    ax3.grid(True, alpha=0.3)

    # -- Panel 4: Error ratio bar chart --
    ax4 = fig.add_subplot(gs[1, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(C_sad_e > 0, C_std_e / C_sad_e, np.nan)
    valid = np.isfinite(ratio)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, T))
    ax4.bar(t[valid], ratio[valid], color=colors[valid],
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4.axhline(1.0, color='red', ls='--', lw=1.5,
                label='Ratio = 1 (no improvement)')
    ax4.set_xlabel('$t$', fontsize=12)
    ax4.set_ylabel(r'$\sigma_{\mathrm{std}} \,/\, \sigma_{\mathrm{SAD}}$',
                   fontsize=12)
    ax4.set_title('Noise reduction factor  '
                  r'$\sigma_{\mathrm{std}}/\sigma_{\mathrm{SAD}}$', fontsize=12)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    # Value labels
    for ti, ri in zip(t[valid], ratio[valid]):
        if T <= 20 or ti % 2 == 0:
            ax4.text(ti, ri + 0.08, f'{ri:.1f}', ha='center', va='bottom',
                     fontsize=7 if T > 20 else 8)

    p = outdir / f"sad_phi4_results{sfx}.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(str(p))
    print(f"  Saved: {p}")

    # ====== Figure 2: Signal-to-noise comparison ======
    fig2, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2.suptitle(f"Signal-to-noise:  SAD vs Standard  ($m^2={m2}$, $\\lambda={lam}$, "
                  f"${T}\\times{L}$)",
                  fontsize=13)

    # Left: absolute errors
    ax_e1.semilogy(t, C_std_e, 's-', ms=4, color='C1',
                   label=r'$\sigma_{\mathrm{std}}$')
    ax_e1.semilogy(t, C_sad_e, 'o-', ms=4, color='C0',
                   label=r'$\sigma_{\mathrm{SAD}}$')
    ax_e1.set_xlabel('$t$', fontsize=12)
    ax_e1.set_ylabel(r'Error $\sigma_{C(t)}$', fontsize=12)
    ax_e1.set_title('Absolute errors', fontsize=12)
    ax_e1.legend(fontsize=10)
    ax_e1.grid(True, alpha=0.3)

    # Right: signal-to-noise = |C| / sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        sn_std = np.abs(C_std) / C_std_e
        sn_sad = np.abs(C_sad) / C_sad_e
    ax_e2.semilogy(t, sn_std, 's-', ms=4, color='C1',
                   label=r'Standard: $|C|/\sigma$')
    ax_e2.semilogy(t, sn_sad, 'o-', ms=4, color='C0',
                   label=r'SAD: $|C|/\sigma$')
    ax_e2.axhline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax_e2.set_xlabel('$t$', fontsize=12)
    ax_e2.set_ylabel(r'$|C(t)| \,/\, \sigma_{C(t)}$', fontsize=12)
    ax_e2.set_title('Signal-to-noise ratio', fontsize=12)
    ax_e2.legend(fontsize=10)
    ax_e2.grid(True, alpha=0.3)

    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = outdir / f"sad_phi4_errors{sfx}.png"
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    saved.append(str(p2))
    print(f"  Saved: {p2}")

    return saved


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_table(data: dict) -> None:
    T, L = data["shape"]
    Thalf = T // 2
    m2, lam = data["m2"], data["lam"]
    B, N = data["batch_size"], data["nmeas"]

    C_sad   = data["C_sad_mean"]
    C_std   = data["C_std_mean"]
    C_sad_e = data["C_sad_err"]
    C_std_e = data["C_std_err"]
    C_exact = data.get("C_exact")

    mc_sad   = data["mcosh_sad"]
    mc_std   = data["mcosh_std"]
    mc_sad_e = data["mcosh_sad_err"]
    mc_std_e = data["mcosh_std_err"]
    mc_exact = data.get("mcosh_exact")

    label = "FREE FIELD" if lam == 0 else f"PHI^4 (lam={lam})"
    print(f"\n{'='*100}")
    print(f"  {label}: m2={m2}, {T}x{L}, B={B}, N={N}")
    print(f"{'='*100}")

    # -- Correlator --
    print(f"\n  Correlator C(t):")
    hdr = f"  {'t':>3}  {'C_sad':>11} {'+-':>2} {'err':>9}  {'C_std':>11} {'+-':>2} {'err':>9}  {'ratio':>6}"
    if C_exact:
        hdr += f"  {'C_exact':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(T):
        r = C_std_e[i] / C_sad_e[i] if C_sad_e[i] > 0 else float('inf')
        row = (f"  {i:>3}  {C_sad[i]:>+11.5f} +- {C_sad_e[i]:>9.2e}  "
               f"{C_std[i]:>+11.5f} +- {C_std_e[i]:>9.2e}  {r:>6.1f}")
        if C_exact:
            row += f"  {C_exact[i]:>+11.5f}"
        print(row)

    # -- Cosh effective mass --
    print(f"\n  Cosh effective mass (t = 0 .. {Thalf-1}):")
    hdr2 = f"  {'t':>3}  {'mcosh_sad':>10} {'+-':>2} {'err':>8}  {'mcosh_std':>10} {'+-':>2} {'err':>8}  {'ratio':>6}"
    if mc_exact:
        hdr2 += f"  {'exact':>10}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for i in range(Thalf):
        r = mc_std_e[i] / mc_sad_e[i] if mc_sad_e[i] > 0 else float('inf')
        row = (f"  {i:>3}  {mc_sad[i]:>10.5f} +- {mc_sad_e[i]:>8.5f}  "
               f"{mc_std[i]:>10.5f} +- {mc_std_e[i]:>8.5f}  {r:>6.1f}")
        if mc_exact:
            row += f"  {mc_exact[i]:>10.5f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-json", type=str,
                    default=str(Path(__file__).parent / "sad_free.json"))
    ap.add_argument("--phi4-json", type=str, nargs="*",
                    default=[str(Path(__file__).parent / "sad_phi4.json")])
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(args.free_json).parent
    outdir.mkdir(parents=True, exist_ok=True)

    all_saved = []

    if os.path.exists(args.free_json):
        print(f"\n--- Free-field: {args.free_json} ---")
        data = load_json(args.free_json)
        print_table(data)
        all_saved.extend(make_plots_free(data, outdir))

    for jf in args.phi4_json:
        if os.path.exists(jf):
            print(f"\n--- phi^4: {jf} ---")
            data = load_json(jf)
            T, L = data["shape"]
            tag = f"{T}x{L}"
            print_table(data)
            all_saved.extend(make_plots_phi4(data, outdir, tag=tag))

    if all_saved:
        print(f"\nAll plots saved:")
        for p in all_saved:
            print(f"  {p}")


if __name__ == "__main__":
    main()

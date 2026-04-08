#!/usr/bin/env python3
"""JAX equivalent of torchQFT/hmc_phi4.py (core workflow)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import sys
from pathlib import Path
import numpy as np

# Metal backend is unstable on some Apple/JAX builds.
# On macOS only, default to CPU if no backend is explicitly selected.
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

# Allow running as `python scripts/phi4/hmc_phi4.py` from repository root.
def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.core.integrators import minnorm2
from jaxqft.models.phi4 import Phi4
from jaxqft.core.update import hmc
from scripts.phi4.analysis.hmc_common import phi4_summary_from_histories
import jax


def main():
    ap = argparse.ArgumentParser(description="HMC for phi^4 theory in 2D")
    ap.add_argument("--shape",      type=str,   default="64,64",  help="Lattice shape, e.g. 16,16")
    ap.add_argument("--lam",        type=float, default=0.5,      help="Quartic coupling lambda")
    ap.add_argument("--mass",       type=float, default=-0.205,   help="Bare mass")
    ap.add_argument("--nwarm",      type=int,   default=100,      help="Warmup trajectories")
    ap.add_argument("--nmeas",      type=int,   default=100,      help="Measurement steps")
    ap.add_argument("--nskip",      type=int,   default=10,       help="Trajectories between measurements")
    ap.add_argument("--batch-size", type=int,   default=16,       help="Batch size (independent chains)")
    ap.add_argument("--nmd",        type=int,   default=7,        help="MD steps per trajectory")
    ap.add_argument("--tau",        type=float, default=1.0,      help="Trajectory length")
    ap.add_argument("--json-out",   type=str,   default=None,     help="Save results to JSON file")
    ap.add_argument("--hist-out",   type=str,   default=None,     help="Save raw observable histories to .npz")
    ap.add_argument("--k-max",      type=int,   default=1,        help="Measure structure factors for k=1..k_max in each lattice direction")
    ap.add_argument("--iat-method", type=str,   default="gamma",  choices=["gamma", "ips", "sokal"],
                    help="Integrated autocorrelation-time estimator")
    ap.add_argument("--iat-c",      type=float, default=5.0,      help="Window parameter for gamma/sokal IAT")
    ap.add_argument("--block-size", type=int,   default=0,        help="Blocked jackknife size; <=0 selects automatically")
    args = ap.parse_args()

    lat = [int(x) for x in args.shape.split(",")]
    if len(lat) != 2:
        raise ValueError(f"phi^4 HMC script expects a 2D lattice, got shape={lat}")
    Vol = int(np.prod(lat))
    k_max = max(1, min(int(args.k_max), lat[0] // 2, lat[1] // 2))
    momenta_k = np.arange(1, k_max + 1, dtype=np.int64)
    hist_out = args.hist_out
    if hist_out is None and args.json_out:
        hist_out = str(Path(args.json_out).with_suffix(".npz"))
    if hist_out:
        Path(hist_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.json_out:
        Path(args.json_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(f"Lattice: {lat}  lam={args.lam}  mass={args.mass}  batch={args.batch_size}")
    print(f"Integrator: minnorm2  nmd={args.nmd}  tau={args.tau}")
    print(f"Momentum scan: k=1..{k_max}")
    print(f"Nwarm={args.nwarm}  Nmeas={args.nmeas}  Nskip={args.nskip}")

    sg    = Phi4(lat, args.lam, args.mass, batch_size=args.batch_size)
    phi   = sg.hotStart()
    mn2   = minnorm2(sg.force, sg.evolveQ, args.nmd, args.tau)
    chain = hmc(T=sg, I=mn2, verbose=False)

    tic = time.perf_counter()
    phi = chain.evolve(phi, args.nwarm)
    toc = time.perf_counter()
    print(f"Warmup done: {(toc - tic) * 1e6 / args.nwarm:.1f} μs/traj")

    obs_E        = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_m        = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_C2pk_x   = np.zeros((args.nmeas, args.batch_size, k_max), dtype=np.float64)
    obs_C2pk_y   = np.zeros((args.nmeas, args.batch_size, k_max), dtype=np.float64)
    phase_x      = np.exp(
        2j * np.pi * np.outer(momenta_k.astype(np.float64), np.arange(lat[0], dtype=np.float64)) / float(lat[0])
    ).reshape((k_max, lat[0], 1))
    phase_y      = np.exp(
        2j * np.pi * np.outer(momenta_k.astype(np.float64), np.arange(lat[1], dtype=np.float64)) / float(lat[1])
    ).reshape((k_max, 1, lat[1]))
    print_every  = max(1, args.nmeas // 10)

    tic = time.perf_counter()
    for k in range(args.nmeas):
        ttE      = np.asarray(sg.action(phi) / Vol)
        av_sigma = np.asarray(phi.mean(axis=(1, 2)))
        p1_x     = np.asarray((phi[:, None, :, :] * phase_x[None, :, :, :]).mean(axis=(2, 3)))
        p1_y     = np.asarray((phi[:, None, :, :] * phase_y[None, :, :, :]).mean(axis=(2, 3)))
        C2p_x_k  = np.real(np.conj(p1_x) * p1_x) * Vol
        C2p_y_k  = np.real(np.conj(p1_y) * p1_y) * Vol

        obs_E[k]       = ttE
        obs_m[k]       = av_sigma
        obs_C2pk_x[k]  = C2p_x_k
        obs_C2pk_y[k]  = C2p_y_k

        if k % print_every == 0:
            print(f"  k={k:5d}  av_phi={float(av_sigma.mean()):+.4f}"
                  f"  C2p_x(k=1)={float(C2p_x_k[:, 0].mean()):.2f}"
                  f"  C2p_y(k=1)={float(C2p_y_k[:, 0].mean()):.2f}"
                  f"  E={float(ttE.mean()):.4f}")

        phi = chain.evolve(phi, args.nskip)
    toc = time.perf_counter()
    print(f"Measurement done: {(toc - tic) * 1e6 / (args.nmeas * args.nskip):.1f} μs/traj")

    acc   = chain.calc_Acceptance()
    summary = phi4_summary_from_histories(
        shape=(int(lat[0]), int(lat[1])),
        magnetization=obs_m,
        energy_density=obs_E,
        c2p_x=obs_C2pk_x,
        c2p_y=obs_C2pk_y,
        momenta_k=momenta_k,
        iat_method=str(args.iat_method),
        iat_c=float(args.iat_c),
        block_size=int(args.block_size),
    )
    primitive = summary["primitive"]
    derived = summary["derived"]

    print("\n--- Results (HMC) ---")
    print(f"{'Observable':<10}  {'mean':>13}  {'sigma_mean':>13}  {'tau_int':>9}  {'ESS':>8}")
    for label, key in (
        ("m", "magnetization"),
        ("|m|", "abs_magnetization"),
        ("m^2", "magnetization2"),
        ("m^4", "magnetization4"),
        ("C2p_x", "C2p_x"),
        ("C2p_y", "C2p_y"),
        ("C2p", "C2p"),
        ("E/V", "energy_density"),
    ):
        row = primitive[key]
        print(f"{label:<10}  {row['mean']:>+13.6f}  {row['sigma']:>13.6f}  {row['tau_int']:>9.2f}  {row['ess']:>8.0f}")
    print(f"{'chi_m':<10}  {derived['chi_m']['mean']:>+13.6f} +/- {derived['chi_m']['stderr']:.6f}")
    print(f"{'B4':<10}  {derived['binder_ratio']['mean']:>+13.6f} +/- {derived['binder_ratio']['stderr']:.6f}")
    print(f"{'U4':<10}  {derived['binder_cumulant']['mean']:>+13.6f} +/- {derived['binder_cumulant']['stderr']:.6f}")
    print(f"{'xi2_x':<10}  {derived['xi2_x']['mean']:>+13.6f} +/- {derived['xi2_x']['stderr']:.6f}")
    print(f"{'xi2_y':<10}  {derived['xi2_y']['mean']:>+13.6f} +/- {derived['xi2_y']['stderr']:.6f}")
    print(f"{'xi2':<10}  {derived['xi2']['mean']:>+13.6f} +/- {derived['xi2']['stderr']:.6f}")
    print(f"{'xi2/L':<10}  {derived['xi2_over_L']['mean']:>+13.6f} +/- {derived['xi2_over_L']['stderr']:.6f}")
    if k_max > 1:
        print(f"{'xi2_fit_l':<10}  {derived['xi2_fit_linear']['mean']:>+13.6f} +/- {derived['xi2_fit_linear']['stderr']:.6f}")
        print(f"{'xi2_fit_q':<10}  {derived['xi2_fit_quadratic']['mean']:>+13.6f} +/- {derived['xi2_fit_quadratic']['stderr']:.6f}")
    print(f"Acceptance rate: {acc:.4f}")

    if hist_out:
        np.savez_compressed(
            hist_out,
            shape=np.asarray(lat, dtype=np.int64),
            lam=np.asarray([args.lam], dtype=np.float64),
            mass=np.asarray([args.mass], dtype=np.float64),
            nwarm=np.asarray([args.nwarm], dtype=np.int64),
            nmeas=np.asarray([args.nmeas], dtype=np.int64),
            nskip=np.asarray([args.nskip], dtype=np.int64),
            batch_size=np.asarray([args.batch_size], dtype=np.int64),
            nmd=np.asarray([args.nmd], dtype=np.int64),
            tau=np.asarray([args.tau], dtype=np.float64),
            k_max=np.asarray([k_max], dtype=np.int64),
            momenta_k=momenta_k,
            acceptance=np.asarray([acc], dtype=np.float64),
            magnetization=obs_m,
            energy_density=obs_E,
            c2p_x=obs_C2pk_x[:, :, 0],
            c2p_y=obs_C2pk_y[:, :, 0],
            c2pk_x=obs_C2pk_x,
            c2pk_y=obs_C2pk_y,
        )
        print(f"Histories saved to {hist_out}")

    if args.json_out:
        out = {
            **summary,
            "updater": "hmc",
            "lam": float(args.lam),
            "mass": float(args.mass),
            "nwarm": int(args.nwarm),
            "nmeas": int(args.nmeas),
            "nskip": int(args.nskip),
            "batch_size": int(args.batch_size),
            "nmd": int(args.nmd),
            "tau": float(args.tau),
            "acceptance": float(acc),
            "histories": str(Path(hist_out).resolve()) if hist_out else None,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()

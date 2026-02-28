#!/usr/bin/env python3
"""SMD/GHMC simulation for phi^4 theory in 2D."""

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

# Allow running as `python scripts/phi4/smd_phi4.py` from repository root.
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
from jaxqft.core.update import SMD
from jaxqft.stats.autocorr import integrated_autocorr_time
import jax


def correlation_length(L, ChiM, C2p):
    return 1 / (2 * np.sin(np.pi / L)) * np.sqrt(ChiM / C2p - 1)


def analyze(obs2d: np.ndarray) -> dict:
    """Compute mean, IAT-corrected sigma_mean, tau_int, and ESS.

    obs2d has shape (nmeas, batch_size).  Batch elements are treated as
    independent chains; IAT is estimated on the batch-mean series so that
    sigma_mean already accounts for both autocorrelations and batch averaging.
    """
    batch_mean = obs2d.mean(axis=1)          # (nmeas,)
    grand_mean = float(np.mean(batch_mean))
    iat = integrated_autocorr_time(batch_mean, method="gamma")
    sigma_mean = float(iat.get("sigma_mean", np.nan))
    tau_int    = float(iat["tau_int"])
    # ESS for the batch-mean series scaled by batch_size (independent chains)
    ess_total  = float(iat["ess"]) * obs2d.shape[1]
    return dict(mean=grand_mean, sigma=sigma_mean, tau_int=tau_int, ess=ess_total)


def main():
    ap = argparse.ArgumentParser(description="SMD/GHMC for phi^4 theory in 2D")
    ap.add_argument("--shape",            type=str,   default="64,64",  help="Lattice shape, e.g. 16,16")
    ap.add_argument("--lam",              type=float, default=0.5,      help="Quartic coupling lambda")
    ap.add_argument("--mass",             type=float, default=-0.205,   help="Bare mass")
    ap.add_argument("--nwarm",            type=int,   default=100,      help="Warmup trajectories (A/R disabled)")
    ap.add_argument("--nmeas",            type=int,   default=100,      help="Measurement steps")
    ap.add_argument("--nskip",            type=int,   default=10,       help="Trajectories between measurements")
    ap.add_argument("--batch-size",       type=int,   default=16,       help="Batch size (independent chains)")
    ap.add_argument("--nmd",              type=int,   default=7,        help="MD steps per trajectory")
    ap.add_argument("--tau",              type=float, default=1.0,      help="Trajectory length")
    ap.add_argument("--gamma",            type=float, default=0.3,      help="SMD friction coefficient")
    ap.add_argument("--no-accept-reject", action="store_true",          help="Disable Metropolis A/R (pure SMD)")
    ap.add_argument("--json-out",         type=str,   default=None,     help="Save results to JSON file")
    args = ap.parse_args()

    lat           = [int(x) for x in args.shape.split(",")]
    Vol           = int(np.prod(lat))
    accept_reject = not args.no_accept_reject

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(f"Lattice: {lat}  lam={args.lam}  mass={args.mass}  batch={args.batch_size}")
    print(f"Integrator: minnorm2  nmd={args.nmd}  tau={args.tau}")
    print(f"SMD gamma={args.gamma}  accept_reject={accept_reject}")
    print(f"Nwarm={args.nwarm}  Nmeas={args.nmeas}  Nskip={args.nskip}")

    sg    = Phi4(lat, args.lam, args.mass, batch_size=args.batch_size)
    phi   = sg.hotStart()
    mn2   = minnorm2(sg.force, sg.evolveQ, args.nmd, args.tau)
    chain = SMD(T=sg, I=mn2, gamma=args.gamma, accept_reject=accept_reject, verbose=False)

    # Warmup with A/R disabled (pure MD thermalisation)
    tic = time.perf_counter()
    phi = chain.evolve(phi, args.nwarm, warmup=True)
    toc = time.perf_counter()
    print(f"Warmup done: {(toc - tic) * 1e6 / args.nwarm:.1f} μs/traj")
    chain.reset_acceptance()

    obs_E        = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_phi      = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_chi_raw  = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_C2p      = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    phase        = np.exp(1j * np.indices(tuple(lat))[0] * 2 * np.pi / lat[0])
    print_every  = max(1, args.nmeas // 10)

    tic = time.perf_counter()
    for k in range(args.nmeas):
        ttE      = np.asarray(sg.action(phi) / Vol)
        av_sigma = np.asarray(phi.reshape(sg.Bs, Vol).mean(axis=1))
        chi_raw  = av_sigma * av_sigma * Vol
        p1_sig   = np.asarray((phi.reshape(sg.Bs, Vol) * phase.reshape(1, Vol)).mean(axis=1))
        C2p_k    = np.real(np.conj(p1_sig) * p1_sig) * Vol

        obs_E[k]       = ttE
        obs_phi[k]     = av_sigma
        obs_chi_raw[k] = chi_raw
        obs_C2p[k]     = C2p_k

        if k % print_every == 0:
            print(f"  k={k:5d}  av_phi={float(av_sigma.mean()):+.4f}"
                  f"  chi={float(chi_raw.mean()):.2f}"
                  f"  C2p={float(C2p_k.mean()):.2f}"
                  f"  E={float(ttE.mean()):.4f}")

        phi = chain.evolve(phi, args.nskip)
    toc = time.perf_counter()
    print(f"Measurement done: {(toc - tic) * 1e6 / (args.nmeas * args.nskip):.1f} μs/traj")

    # Connected susceptibility: chi_conn = <phi^2>*Vol - <phi>^2*Vol
    m_phi_est    = float(np.mean(obs_phi))
    obs_chi_conn = obs_chi_raw - m_phi_est**2 * Vol

    r_phi = analyze(obs_phi)
    r_chi = analyze(obs_chi_conn)
    r_C2p = analyze(obs_C2p)
    r_E   = analyze(obs_E)
    xi    = correlation_length(lat[0], r_chi["mean"], r_C2p["mean"])
    acc   = chain.calc_Acceptance()

    print("\n--- Results (SMD) ---")
    print(f"{'Observable':<10}  {'mean':>13}  {'sigma_mean':>13}  {'tau_int':>9}  {'ESS':>8}")
    print(f"{'av_phi':<10}  {r_phi['mean']:>+13.6f}  {r_phi['sigma']:>13.6f}  {r_phi['tau_int']:>9.2f}  {r_phi['ess']:>8.0f}")
    print(f"{'Chi_m':<10}  {r_chi['mean']:>+13.4f}  {r_chi['sigma']:>13.4f}  {r_chi['tau_int']:>9.2f}  {r_chi['ess']:>8.0f}")
    print(f"{'C2p':<10}  {r_C2p['mean']:>+13.4f}  {r_C2p['sigma']:>13.4f}  {r_C2p['tau_int']:>9.2f}  {r_C2p['ess']:>8.0f}")
    print(f"{'E/V':<10}  {r_E['mean']:>+13.6f}  {r_E['sigma']:>13.6f}  {r_E['tau_int']:>9.2f}  {r_E['ess']:>8.0f}")
    print(f"xi = {xi:.4f}")
    print(f"Acceptance rate: {acc:.4f}")

    if args.json_out:
        out = dict(
            updater="smd", shape=lat, lam=args.lam, mass=args.mass,
            nwarm=args.nwarm, nmeas=args.nmeas, nskip=args.nskip,
            batch_size=args.batch_size, nmd=args.nmd, tau=args.tau,
            gamma=args.gamma, accept_reject=accept_reject,
            phi=r_phi, chi_m=r_chi, C2p=r_C2p, E=r_E,
            xi=float(xi), acceptance=float(acc),
        )
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()

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

from jaxqft.core.integrators import force_gradient, leapfrog, minnorm2, minnorm4pf4
from jaxqft.models.phi4 import Phi4
from jaxqft.models.phi4_rg_cond_flow import _split_rg
from jaxqft.core.update import hmc
from scripts.phi4.analysis.hmc_common import phi4_multilevel_summaries_from_payload, phi4_summary_from_histories
import jax


def _build_integrator(name: str, theory: Phi4, nmd: int, tau: float):
    key = str(name).lower()
    if key == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, nmd, tau)
    if key == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, nmd, tau)
    if key == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, nmd, tau)
    if key == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, nmd, tau)
    raise ValueError(f"Unknown integrator: {name}")


def _blocked_level_shapes(shape: tuple[int, int]) -> list[tuple[int, int]]:
    shapes = [tuple(int(v) for v in shape)]
    cur = tuple(int(v) for v in shape)
    while min(cur) > 2:
        if cur[0] % 2 != 0 or cur[1] % 2 != 0:
            break
        cur = (cur[0] // 2, cur[1] // 2)
        shapes.append(cur)
    return shapes


def _collect_blocked_level_fields(x: jax.Array, rg_mode: int) -> list[jax.Array]:
    fields = [x]
    xx = x
    while min(int(xx.shape[1]), int(xx.shape[2])) > 2:
        if int(xx.shape[1]) % 2 != 0 or int(xx.shape[2]) % 2 != 0:
            break
        xx, _ = _split_rg(xx, int(rg_mode))
        fields.append(xx)
    return fields


def _phase_cache_for_shape(shape: tuple[int, int], k_max: int) -> dict:
    ny, nx = (int(shape[0]), int(shape[1]))
    nk = max(1, min(int(k_max), ny // 2, nx // 2))
    ks = np.arange(1, nk + 1, dtype=np.int64)
    phase_x = np.exp(
        2j * np.pi * np.outer(ks.astype(np.float64), np.arange(nx, dtype=np.float64)) / float(nx)
    ).reshape((nk, 1, nx))
    phase_y = np.exp(
        2j * np.pi * np.outer(ks.astype(np.float64), np.arange(ny, dtype=np.float64)) / float(ny)
    ).reshape((nk, ny, 1))
    return {
        "shape": (ny, nx),
        "volume": float(ny * nx),
        "momenta_k": ks,
        "phase_x": phase_x,
        "phase_y": phase_y,
    }


def _measure_level_observables(x: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    m = np.mean(arr, axis=(1, 2))
    x_complex = arr.astype(np.complex128)
    pkx = np.mean(x_complex[:, None, :, :] * cache["phase_x"][None, :, :, :], axis=(2, 3))
    pky = np.mean(x_complex[:, None, :, :] * cache["phase_y"][None, :, :, :], axis=(2, 3))
    c2pk_x = cache["volume"] * np.real(np.conj(pkx) * pkx)
    c2pk_y = cache["volume"] * np.real(np.conj(pky) * pky)
    return np.asarray(m, dtype=np.float64), np.asarray(c2pk_x, dtype=np.float64), np.asarray(c2pk_y, dtype=np.float64)


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
    ap.add_argument("--integrator", type=str,   default="minnorm2",
                    choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"],
                    help="Molecular-dynamics integrator")
    ap.add_argument("--seed",       type=int,   default=0,        help="PRNG seed for theory/update")
    ap.add_argument("--json-out",   type=str,   default=None,     help="Save results to JSON file")
    ap.add_argument("--hist-out",   type=str,   default=None,     help="Save raw observable histories to .npz")
    ap.add_argument("--k-max",      type=int,   default=1,        help="Measure structure factors for k=1..k_max in each lattice direction")
    ap.add_argument("--iat-method", type=str,   default="gamma",  choices=["gamma", "ips", "sokal"],
                    help="Integrated autocorrelation-time estimator")
    ap.add_argument("--iat-c",      type=float, default=5.0,      help="Window parameter for gamma/sokal IAT")
    ap.add_argument("--block-size", type=int,   default=0,        help="Blocked jackknife size; <=0 selects automatically")
    ap.add_argument("--measure-blocked-levels", action="store_true",
                    help="Also save blocked-level histories using the same 2x2 RG split as the flow analysis")
    ap.add_argument("--blocked-rg-mode", type=str, default="average", choices=["average", "select"],
                    help="RG split used for blocked-level measurements")
    args = ap.parse_args()

    lat = [int(x) for x in args.shape.split(",")]
    if len(lat) != 2:
        raise ValueError(f"phi^4 HMC script expects a 2D lattice, got shape={lat}")
    Vol = int(np.prod(lat))
    k_max = max(1, min(int(args.k_max), lat[0] // 2, lat[1] // 2))
    momenta_k = np.arange(1, k_max + 1, dtype=np.int64)
    blocked_rg_mode = 0 if str(args.blocked_rg_mode) == "average" else 1
    blocked_shapes = _blocked_level_shapes((int(lat[0]), int(lat[1])))
    blocked_caches = [_phase_cache_for_shape(shape, k_max) for shape in blocked_shapes]
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
    print(f"Integrator: {args.integrator}  nmd={args.nmd}  tau={args.tau}")
    print(f"Seed: {args.seed}")
    print(f"Momentum scan: k=1..{k_max}")
    print(f"Nwarm={args.nwarm}  Nmeas={args.nmeas}  Nskip={args.nskip}")
    if args.measure_blocked_levels:
        shape_str = ", ".join(f"{s[0]}x{s[1]}" for s in blocked_shapes)
        print(f"Blocked levels: {shape_str}  rg_mode={args.blocked_rg_mode}")

    sg    = Phi4(lat, args.lam, args.mass, batch_size=args.batch_size, seed=args.seed)
    phi   = sg.hotStart()
    integ = _build_integrator(args.integrator, sg, args.nmd, args.tau)
    chain = hmc(T=sg, I=integ, verbose=False, seed=args.seed + 1)

    tic = time.perf_counter()
    phi = chain.evolve(phi, args.nwarm)
    toc = time.perf_counter()
    warmup_sec = float(toc - tic)
    warmup_usec_per_traj = float(warmup_sec * 1e6 / max(1, args.nwarm))
    print(f"Warmup done: {warmup_usec_per_traj:.1f} μs/traj")

    obs_E        = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_m        = np.zeros((args.nmeas, args.batch_size), dtype=np.float64)
    obs_C2pk_x   = np.zeros((args.nmeas, args.batch_size, k_max), dtype=np.float64)
    obs_C2pk_y   = np.zeros((args.nmeas, args.batch_size, k_max), dtype=np.float64)
    level_m = [np.zeros((args.nmeas, args.batch_size), dtype=np.float64) for _ in blocked_shapes] if args.measure_blocked_levels else None
    level_m2 = [np.zeros((args.nmeas, args.batch_size), dtype=np.float64) for _ in blocked_shapes] if args.measure_blocked_levels else None
    level_m4 = [np.zeros((args.nmeas, args.batch_size), dtype=np.float64) for _ in blocked_shapes] if args.measure_blocked_levels else None
    level_C2pk_x = [
        np.zeros((args.nmeas, args.batch_size, int(cache["momenta_k"].size)), dtype=np.float64) for cache in blocked_caches
    ] if args.measure_blocked_levels else None
    level_C2pk_y = [
        np.zeros((args.nmeas, args.batch_size, int(cache["momenta_k"].size)), dtype=np.float64) for cache in blocked_caches
    ] if args.measure_blocked_levels else None
    print_every  = max(1, args.nmeas // 10)

    tic = time.perf_counter()
    for k in range(args.nmeas):
        ttE      = np.asarray(sg.action(phi) / Vol)
        av_sigma, C2p_x_k, C2p_y_k = _measure_level_observables(np.asarray(phi), blocked_caches[0])

        obs_E[k]       = ttE
        obs_m[k]       = av_sigma
        obs_C2pk_x[k]  = C2p_x_k
        obs_C2pk_y[k]  = C2p_y_k

        if args.measure_blocked_levels and level_m is not None:
            blocked_fields = _collect_blocked_level_fields(phi, blocked_rg_mode)
            for level, (field, cache) in enumerate(zip(blocked_fields, blocked_caches)):
                level_mean, level_c2x, level_c2y = _measure_level_observables(np.asarray(field), cache)
                level_m[level][k] = level_mean
                level_m2[level][k] = level_mean * level_mean
                level_m4[level][k] = level_m2[level][k] * level_m2[level][k]
                level_C2pk_x[level][k] = level_c2x
                level_C2pk_y[level][k] = level_c2y

        if k % print_every == 0:
            print(f"  k={k:5d}  av_phi={float(av_sigma.mean()):+.4f}"
                  f"  C2p_x(k=1)={float(C2p_x_k[:, 0].mean()):.2f}"
                  f"  C2p_y(k=1)={float(C2p_y_k[:, 0].mean()):.2f}"
                  f"  E={float(ttE.mean()):.4f}")

        phi = chain.evolve(phi, args.nskip)
    toc = time.perf_counter()
    measure_sec = float(toc - tic)
    measure_usec_per_traj = float(measure_sec * 1e6 / max(1, args.nmeas * args.nskip))
    print(f"Measurement done: {measure_usec_per_traj:.1f} μs/traj")

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
    tuning_costs = {}
    for out_key, prim_key in (
        ("m2", "magnetization2"),
        ("c2p", "C2p"),
        ("energy_density", "energy_density"),
    ):
        row = primitive.get(prim_key)
        if row is None:
            continue
        tau_int = float(row.get("tau_int", np.nan))
        tuning_costs[out_key] = {
            "tau_int": tau_int,
            "measure_usec_per_traj": float(measure_usec_per_traj),
            "batch_size": int(args.batch_size),
            "cost_usec_tau_over_batch": (
                float(measure_usec_per_traj * tau_int / float(args.batch_size))
                if np.isfinite(tau_int) and int(args.batch_size) > 0
                else float("nan")
            ),
        }

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
    print("\n--- Tuning Cost Metrics ---")
    for label, row in tuning_costs.items():
        print(
            f"{label:<14}  tau_int={row['tau_int']:.3f}  "
            f"cost=(t_traj*tau/batch)={row['cost_usec_tau_over_batch']:.3f} usec"
        )

    hist_payload = {
        "shape": np.asarray(lat, dtype=np.int64),
        "lam": np.asarray([args.lam], dtype=np.float64),
        "mass": np.asarray([args.mass], dtype=np.float64),
        "nwarm": np.asarray([args.nwarm], dtype=np.int64),
        "nmeas": np.asarray([args.nmeas], dtype=np.int64),
        "nskip": np.asarray([args.nskip], dtype=np.int64),
        "batch_size": np.asarray([args.batch_size], dtype=np.int64),
        "nmd": np.asarray([args.nmd], dtype=np.int64),
        "tau": np.asarray([args.tau], dtype=np.float64),
        "k_max": np.asarray([k_max], dtype=np.int64),
        "momenta_k": momenta_k,
        "acceptance": np.asarray([acc], dtype=np.float64),
        "magnetization": obs_m,
        "energy_density": obs_E,
        "c2p_x": obs_C2pk_x[:, :, 0],
        "c2p_y": obs_C2pk_y[:, :, 0],
        "c2pk_x": obs_C2pk_x,
        "c2pk_y": obs_C2pk_y,
    }
    if args.measure_blocked_levels and level_m is not None and level_C2pk_x is not None and level_C2pk_y is not None:
        hist_payload["blocked_n_levels"] = np.asarray([len(blocked_shapes)], dtype=np.int64)
        hist_payload["blocked_rg_mode"] = np.asarray([blocked_rg_mode], dtype=np.int64)
        hist_payload["blocked_level_shapes"] = np.asarray(blocked_shapes, dtype=np.int64)
        for level, shape in enumerate(blocked_shapes):
            prefix = f"level{level}_"
            hist_payload[prefix + "shape"] = np.asarray(shape, dtype=np.int64)
            hist_payload[prefix + "momenta_k"] = np.asarray(blocked_caches[level]["momenta_k"], dtype=np.int64)
            hist_payload[prefix + "magnetization"] = level_m[level]
            hist_payload[prefix + "magnetization2"] = level_m2[level]
            hist_payload[prefix + "magnetization4"] = level_m4[level]
            hist_payload[prefix + "c2pk_x"] = level_C2pk_x[level]
            hist_payload[prefix + "c2pk_y"] = level_C2pk_y[level]
            if level == 0:
                hist_payload[prefix + "energy_density"] = obs_E

    blocked_summary = (
        phi4_multilevel_summaries_from_payload(
            hist_payload,
            iat_method=str(args.iat_method),
            iat_c=float(args.iat_c),
            block_size=int(args.block_size),
        )
        if args.measure_blocked_levels
        else None
    )
    if blocked_summary is not None:
        print("\n--- Blocked level summary ---")
        print(f"{'lvl':>3}  {'L':>5}  {'xi2':>22}  {'U4':>22}")
        for row in blocked_summary["levels"]:
            xi = row["unweighted"]["xi2"]
            u4 = row["unweighted"]["binder_cumulant"]
            print(
                f"{row['level_from_fine']:3d}  {row['L']:5d}  "
                f"{xi['mean']:>+13.6f} +/- {xi['stderr']:<8.6f}  "
                f"{u4['mean']:>+13.6f} +/- {u4['stderr']:<8.6f}"
            )

    if hist_out:
        np.savez_compressed(hist_out, **hist_payload)
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
            "integrator": str(args.integrator),
            "seed": int(args.seed),
            "acceptance": float(acc),
            "performance": {
                "warmup_sec": float(warmup_sec),
                "measure_sec": float(measure_sec),
                "warmup_usec_per_traj": float(warmup_usec_per_traj),
                "measure_usec_per_traj": float(measure_usec_per_traj),
                "ntraj_warmup": int(args.nwarm),
                "ntraj_measure": int(args.nmeas * args.nskip),
            },
            "tuning_costs": tuning_costs,
            "histories": str(Path(hist_out).resolve()) if hist_out else None,
            "blocked_levels": blocked_summary,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()

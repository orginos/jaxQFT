#!/usr/bin/env python3
"""Analyze saved phi^4 HMC histories with Gamma-aware errors."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phi4.analysis.hmc_common import phi4_multilevel_summaries_from_payload, phi4_summary_from_histories


def _load_histories(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {"shape", "magnetization", "energy_density"}
        missing = required - keys
        if missing:
            raise ValueError(f"missing required keys in {path}: {sorted(missing)}")
        if not ({"c2p_x", "c2p_y"} <= keys or {"c2pk_x", "c2pk_y", "momenta_k"} <= keys):
            raise ValueError(f"missing momentum structure-factor histories in {path}")
        return {key: np.asarray(data[key]) for key in data.files}


def _print_summary(summary: dict, acceptance: float | None = None) -> None:
    p = summary["primitive"]
    d = summary["derived"]
    print("--- Results (reanalysis) ---")
    print(f"{'Observable':<16}  {'mean':>13}  {'sigma_mean':>13}  {'tau_int':>9}  {'ESS':>8}")
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
        if key not in p:
            continue
        row = p[key]
        print(f"{label:<16}  {row['mean']:>+13.6f}  {row['sigma']:>13.6f}  {row['tau_int']:>9.2f}  {row['ess']:>8.0f}")
    print("")
    for label, key in (
        ("chi_m", "chi_m"),
        ("binder_ratio", "binder_ratio"),
        ("binder_cumulant", "binder_cumulant"),
        ("xi2_x", "xi2_x"),
        ("xi2_y", "xi2_y"),
        ("xi2", "xi2"),
        ("xi2/L", "xi2_over_L"),
    ):
        row = d[key]
        print(f"{label:<16}  {row['mean']:>+13.6f} +/- {row['stderr']:.6f}")
    if d.get("xi2_fit_linear") is not None:
        row = d["xi2_fit_linear"]
        print(f"{'xi2_fit_lin':<16}  {row['mean']:>+13.6f} +/- {row['stderr']:.6f}")
    if d.get("xi2_fit_quadratic") is not None:
        row = d["xi2_fit_quadratic"]
        print(f"{'xi2_fit_quad':<16}  {row['mean']:>+13.6f} +/- {row['stderr']:.6f}")
    if d.get("xi2_momentum_scan"):
        print("")
        print("k-scan              xi2(mean +/- err)")
        for row in d["xi2_momentum_scan"]:
            xi = row["xi2"]
            print(f"{int(row['k']):>3d}                 {xi['mean']:+.6f} +/- {xi['stderr']:.6f}")
    if acceptance is not None and np.isfinite(acceptance):
        print(f"acceptance        {acceptance:.6f}")


def _print_level_summaries(multilevel: dict | None) -> None:
    if not multilevel or int(multilevel.get("n_levels", 0)) <= 1:
        return
    print("")
    print("Blocked levels")
    print(f"{'lvl':>3}  {'L':>5}  {'xi2':>22}  {'U4':>22}")
    for row in multilevel["levels"]:
        obs = row["unweighted"]
        xi = obs["xi2"]
        u4 = obs["binder_cumulant"]
        print(
            f"{row['level_from_fine']:3d}  {row['L']:5d}  "
            f"{xi['mean']:+13.6f} +/- {xi['stderr']:<8.6f}  "
            f"{u4['mean']:+13.6f} +/- {u4['stderr']:<8.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Reanalyze saved phi^4 HMC histories")
    ap.add_argument("histories", type=str, help="Input .npz file written by scripts/phi4/hmc_phi4.py")
    ap.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    ap.add_argument("--iat-method", type=str, default="gamma", choices=["gamma", "ips", "sokal"])
    ap.add_argument("--iat-c", type=float, default=5.0, help="Window parameter for Gamma/Sokal IAT estimators")
    ap.add_argument("--block-size", type=int, default=0, help="Blocked jackknife size; <=0 chooses automatically from tau_int")
    args = ap.parse_args()

    payload = _load_histories(args.histories)
    shape = tuple(int(v) for v in np.asarray(payload["shape"]).reshape(-1).tolist())
    if len(shape) != 2:
        raise ValueError(f"expected a 2D lattice shape, got {shape}")

    summary = phi4_summary_from_histories(
        shape=(int(shape[0]), int(shape[1])),
        magnetization=payload["magnetization"],
        energy_density=payload["energy_density"],
        c2p_x=payload["c2pk_x"] if "c2pk_x" in payload else payload["c2p_x"],
        c2p_y=payload["c2pk_y"] if "c2pk_y" in payload else payload["c2p_y"],
        momenta_k=payload["momenta_k"] if "momenta_k" in payload else None,
        iat_method=str(args.iat_method),
        iat_c=float(args.iat_c),
        block_size=int(args.block_size),
    )
    multilevel = phi4_multilevel_summaries_from_payload(
        payload,
        iat_method=str(args.iat_method),
        iat_c=float(args.iat_c),
        block_size=int(args.block_size),
    )
    meta = {}
    for key in ("lam", "mass", "nwarm", "nmeas", "nskip", "batch_size", "nmd", "tau", "acceptance"):
        if key in payload:
            val = np.asarray(payload[key]).reshape(-1)
            if val.size == 1:
                meta[key] = float(val[0]) if key in {"lam", "mass", "tau", "acceptance"} else int(round(float(val[0])))
    out = {**meta, **summary, "histories": str(Path(args.histories).resolve()), "blocked_levels": multilevel}

    _print_summary(summary, acceptance=meta.get("acceptance"))
    _print_level_summaries(multilevel)
    if args.json_out:
        Path(args.json_out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()

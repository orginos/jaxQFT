#!/usr/bin/env python3
"""Run the vendored O(3) regression suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
MANIFEST = ROOT / "docs" / "o3_regression" / "reference" / "manifest.json"
REFERENCE_ROOT = MANIFEST.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the vendored O(3) regression suite")
    ap.add_argument("--outdir", type=str, default=str(ROOT / "runs" / "oN" / "o3_regression"))
    ap.add_argument("--report-json", type=str, default="")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--layout", type=str, default="auto", choices=["BN...", "B...N", "auto"])
    ap.add_argument("--exp-method", type=str, default="auto", choices=["auto", "expm", "rodrigues"])
    ap.add_argument("--update", type=str, default="hmc", choices=["hmc", "smd", "ghmc"])
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--nwarm-scale", type=float, default=1.0)
    ap.add_argument("--nskip-scale", type=float, default=1.0)
    ap.add_argument("--nmeas-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-cut", type=float, default=4.0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--cpu-onednn", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--fail-on-mismatch", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if not MANIFEST.is_file():
        raise SystemExit(f"Missing regression manifest: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = ",".join(str(item["token"]) for item in manifest.get("cases", []) if str(item.get("token", "")).strip())
    if not cases:
        raise SystemExit(f"No regression cases listed in {MANIFEST}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "on" / "compare_o3_torch.py"),
        "--torch-root",
        str(REFERENCE_ROOT),
        "--cases",
        cases,
        "--outdir",
        str(args.outdir),
        "--layout",
        str(args.layout),
        "--exp-method",
        str(args.exp_method),
        "--update",
        str(args.update),
        "--integrator",
        str(args.integrator),
        "--tau",
        str(float(args.tau)),
        "--batch-size",
        str(int(args.batch_size)),
        "--nwarm-scale",
        str(float(args.nwarm_scale)),
        "--nskip-scale",
        str(float(args.nskip_scale)),
        "--nmeas-scale",
        str(float(args.nmeas_scale)),
        "--seed",
        str(int(args.seed)),
        "--sigma-cut",
        str(float(args.sigma_cut)),
    ]
    if str(args.report_json).strip():
        cmd.extend(["--report-json", str(args.report_json)])
    if bool(args.rerun):
        cmd.append("--rerun")
    if int(args.cpu_threads) > 0:
        cmd.extend(["--cpu-threads", str(int(args.cpu_threads))])
    if args.cpu_onednn is True:
        cmd.append("--cpu-onednn")
    if args.cpu_onednn is False:
        cmd.append("--no-cpu-onednn")
    if bool(args.fail_on_mismatch):
        cmd.append("--fail-on-mismatch")

    print("Running O(3) regression:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()

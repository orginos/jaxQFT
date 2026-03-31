#!/usr/bin/env python3
"""Run a curated inline-measurement gauge-invariance regression matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "jaxqft").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
CHECKER = ROOT / "scripts" / "mcmc" / "check_measurement_gauge_invariance.py"
CASE_DIR = ROOT / "scripts" / "mcmc" / "regression_cards" / "measurement_gauge_invariance"

CASE_SPECS: List[Dict[str, str]] = [
    {
        "name": "u1_wilson_dense",
        "config": str(CASE_DIR / "u1_wilson_dense_4x8.toml"),
        "description": "2D U(1) dense meson/two-pion/vector-current stack",
    },
    {
        "name": "su2_wilson_dense",
        "config": str(CASE_DIR / "su2_wilson_dense_2x2x2x4.toml"),
        "description": "4D SU(2) dense meson stack",
    },
    {
        "name": "su3_wilson_dense",
        "config": str(CASE_DIR / "su3_wilson_dense_2x2x2x4.toml"),
        "description": "4D SU(3) dense meson+baryon stack",
    },
]


def _parse_case_names(raw: List[str]) -> Optional[List[str]]:
    vals: List[str] = []
    for item in raw:
        vals.extend(v.strip() for v in str(item).split(",") if v.strip())
    return vals or None


def _tail(txt: str, nline: int = 20) -> str:
    lines = [ln for ln in str(txt).splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-int(nline):])


def _select_cases(names: Optional[List[str]]) -> List[Dict[str, str]]:
    if not names:
        return list(CASE_SPECS)
    wanted = {str(v) for v in names}
    chosen = [spec for spec in CASE_SPECS if spec["name"] in wanted]
    missing = sorted(wanted.difference(spec["name"] for spec in chosen))
    if missing:
        raise ValueError(f"Unknown regression case(s): {', '.join(missing)}")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description="Run curated measurement gauge-invariance regressions")
    ap.add_argument("--case", action="append", default=[], help="subset of regression cases to run; repeat or pass comma-separated")
    ap.add_argument("--ntrials", type=int, default=1)
    ap.add_argument("--omega-scale", type=float, default=0.05)
    ap.add_argument("--atol", type=float, default=1.0e-6)
    ap.add_argument("--rtol", type=float, default=1.0e-6)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--cpu-threads", type=int, default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0))
    ap.add_argument("--selfcheck-fail", action="store_true")
    args = ap.parse_args()

    cases = _select_cases(_parse_case_names(args.case))
    summary: Dict[str, Any] = {
        "root": str(ROOT),
        "checker": str(CHECKER),
        "ntrials": int(args.ntrials),
        "omega_scale": float(args.omega_scale),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "cases": [],
    }

    overall_ok = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    compared_total = 0

    with tempfile.TemporaryDirectory(prefix="gauge_invariance_regress_") as tmpdir:
        tmp = Path(tmpdir)
        for spec in cases:
            case_json = tmp / f"{spec['name']}.json"
            cmd = [
                sys.executable,
                str(CHECKER),
                "--config",
                spec["config"],
                "--ntrials",
                str(int(args.ntrials)),
                "--omega-scale",
                str(float(args.omega_scale)),
                "--atol",
                str(float(args.atol)),
                "--rtol",
                str(float(args.rtol)),
                "--json-out",
                str(case_json),
            ]
            if int(args.cpu_threads) > 0:
                cmd.extend(["--cpu-threads", str(int(args.cpu_threads))])

            print(f"[run] {spec['name']}: {spec['description']}", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            dt = float(time.perf_counter() - t0)

            case_row: Dict[str, Any] = {
                "name": spec["name"],
                "description": spec["description"],
                "config": spec["config"],
                "duration_sec": dt,
                "returncode": int(proc.returncode),
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
            }
            if case_json.exists():
                try:
                    case_doc = json.loads(case_json.read_text(encoding="utf-8"))
                    case_row["result"] = case_doc
                    case_row["pass"] = bool(case_doc.get("pass", False)) and int(proc.returncode) == 0
                    compared_total += int(case_doc.get("compared_keys_total", 0))
                    global_max_abs = max(global_max_abs, float(case_doc.get("max_abs_err", 0.0)))
                    global_max_rel = max(global_max_rel, float(case_doc.get("max_rel_err", 0.0)))
                    print(
                        f"[ok]  {spec['name']}: compared={case_doc.get('compared_keys_total', 0)} "
                        f"max_abs={float(case_doc.get('max_abs_err', 0.0)):.3e} "
                        f"max_rel={float(case_doc.get('max_rel_err', 0.0)):.3e} "
                        f"dt={dt:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    case_row["pass"] = False
                    case_row["json_error"] = str(exc)
                    print(f"[fail] {spec['name']}: could not parse case JSON ({exc})", flush=True)
            else:
                case_row["pass"] = False
                print(f"[fail] {spec['name']}: checker did not produce JSON (rc={proc.returncode})", flush=True)

            if not bool(case_row.get("pass", False)):
                overall_ok = False
                if case_row.get("stderr_tail"):
                    print(case_row["stderr_tail"], flush=True)
            summary["cases"].append(case_row)

    summary["pass"] = bool(overall_ok)
    summary["compared_keys_total"] = int(compared_total)
    summary["max_abs_err"] = float(global_max_abs)
    summary["max_rel_err"] = float(global_max_rel)

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out}", flush=True)

    if overall_ok:
        print(
            f"PASS: {len(cases)}/{len(cases)} cases passed "
            f"(compared={compared_total}, max_abs_err={global_max_abs:.3e}, max_rel_err={global_max_rel:.3e})",
            flush=True,
        )
        return 0

    print(
        f"FAIL: {sum(1 for row in summary['cases'] if row.get('pass'))}/{len(cases)} cases passed",
        flush=True,
    )
    if bool(args.selfcheck_fail):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

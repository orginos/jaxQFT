#!/usr/bin/env python3
"""Compare JAX O(3) runs against torchQFT O(3) reference summaries."""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_TORCH_ROOT = ROOT / "docs" / "o3_regression" / "reference"
FALLBACK_TORCH_ROOT = ROOT.parent / "torchQFT" / "o3_hmc_runs"
DEFAULT_CASES = ("8@1.05", "12@1.187", "24@1.375")


def _case_label(d: dict) -> str:
    return f"{int(d['Lx'])}x{int(d['Ly'])}@{float(d['beta'])}"


def _discover_torch_summaries(root: Path) -> list[tuple[dict, Path]]:
    out = []
    for path in sorted(root.glob("o3_*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.append((data, path))
    return out


def _parse_case_token(token: str) -> tuple[int, int, float]:
    txt = str(token).strip()
    if "@" not in txt:
        raise ValueError(f"case token must look like 24@1.375 or 24x24@1.375, got {token!r}")
    size_txt, beta_txt = txt.split("@", 1)
    if "x" in size_txt.lower():
        a, b = size_txt.lower().split("x", 1)
        lx = int(a)
        ly = int(b)
    else:
        lx = int(size_txt)
        ly = lx
    return lx, ly, float(beta_txt)


def _find_case(summaries: list[tuple[dict, Path]], token: str) -> tuple[dict, Path]:
    lx, ly, beta = _parse_case_token(token)
    for data, path in summaries:
        if int(data["Lx"]) == lx and int(data["Ly"]) == ly and abs(float(data["beta"]) - beta) < 1.0e-12:
            return data, path
    raise KeyError(f"Could not find torch summary for case {token}")


def _selected_cases(summaries: list[tuple[dict, Path]], raw_cases: str, max_cases: int) -> list[tuple[dict, Path]]:
    if str(raw_cases).strip():
        return [_find_case(summaries, tok.strip()) for tok in raw_cases.split(",") if tok.strip()]

    picked = []
    for tok in DEFAULT_CASES:
        try:
            picked.append(_find_case(summaries, tok))
        except KeyError:
            pass
    if picked:
        return picked

    ordered = sorted(summaries, key=lambda dp: (int(dp[0]["Lx"]) * int(dp[0]["Ly"]), float(dp[0]["beta"])))
    return ordered[: max(1, int(max_cases))]


def _build_jax_command(torch_summary: dict, args, out_json: Path) -> list[str]:
    nwarm = max(1, int(round(float(torch_summary["Nwarm"]) * float(args.nwarm_scale))))
    nskip = max(1, int(round(float(torch_summary["Nskip"]) * float(args.nskip_scale))))
    nmeas = max(1, int(round(float(torch_summary["Nmeas"]) * float(args.nmeas_scale))))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(torch_summary["batch_size"])

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "on" / "hmc_on.py"),
        "--shape",
        f"{int(torch_summary['Lx'])},{int(torch_summary['Ly'])}",
        "--beta",
        str(float(torch_summary["beta"])),
        "--ncomp",
        "3",
        "--batch-size",
        str(batch_size),
        "--layout",
        str(args.layout),
        "--exp-method",
        str(args.exp_method),
        "--start",
        "hot",
        "--update",
        str(args.update),
        "--integrator",
        str(args.integrator),
        "--nmd",
        str(int(torch_summary["Nmd"])),
        "--tau",
        str(float(args.tau)),
        "--warmup-no-ar",
        str(int(args.warmup_no_ar)),
        "--nwarm",
        str(nwarm),
        "--nmeas",
        str(nmeas),
        "--nskip",
        str(nskip),
        "--seed",
        str(int(args.seed)),
        "--json-out",
        str(out_json),
    ]
    if int(args.cpu_threads) > 0:
        cmd.extend(["--cpu-threads", str(int(args.cpu_threads))])
    if args.cpu_onednn is True:
        cmd.append("--cpu-onednn")
    if args.cpu_onednn is False:
        cmd.append("--no-cpu-onednn")
    if bool(args.benchmark_kernels):
        cmd.append("--benchmark-kernels")
        cmd.extend(["--benchmark-iter", str(int(args.benchmark_iter))])
    if str(args.benchmark_integrators).strip():
        cmd.extend(["--benchmark-integrators", str(args.benchmark_integrators)])
        cmd.extend(["--benchmark-iter", str(int(args.benchmark_iter))])
    return cmd


def _compare_field(torch_summary: dict, jax_summary: dict, name: str, err_name: str) -> dict[str, float | bool | str]:
    tval = torch_summary.get(name, None)
    jval = jax_summary.get(name, None)
    terr = torch_summary.get(err_name, None)
    jerr = jax_summary.get(err_name, None)

    if tval is None or jval is None:
        return {"field": name, "skip": True}

    t = float(tval)
    j = float(jval)
    dt = abs(j - t)
    rel = dt / max(abs(t), 1.0e-16)

    sigma = float("nan")
    if terr is not None and jerr is not None:
        te = float(terr)
        je = float(jerr)
        comb = math.sqrt(max(0.0, te * te + je * je))
        if math.isfinite(comb) and comb > 0.0:
            sigma = dt / comb

    return {
        "field": name,
        "torch": t,
        "jax": j,
        "delta": dt,
        "rel": rel,
        "sigma": sigma,
        "skip": False,
    }


def _case_ok(comparisons: list[dict], sigma_cut: float) -> bool:
    ok = True
    for comp in comparisons:
        if bool(comp.get("skip", False)):
            continue
        sigma = float(comp.get("sigma", float("nan")))
        if math.isfinite(sigma):
            ok = ok and (sigma <= float(sigma_cut))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare JAX O(3) runs against torchQFT summary.json references")
    ap.add_argument("--torch-root", type=str, default=str(DEFAULT_TORCH_ROOT))
    ap.add_argument("--cases", type=str, default="", help="Comma-separated cases like 8@1.05,12@1.187,24@1.375")
    ap.add_argument("--max-cases", type=int, default=3)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "runs" / "oN" / "torch_compare"))
    ap.add_argument("--report-json", type=str, default="")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--layout", type=str, default="auto", choices=["BN...", "B...N", "auto"])
    ap.add_argument("--exp-method", type=str, default="auto", choices=["auto", "expm", "rodrigues"])
    ap.add_argument("--update", type=str, default="hmc", choices=["hmc", "smd", "ghmc"])
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--warmup-no-ar", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0, help="Override torch batch size; <=0 uses the torch reference batch size")
    ap.add_argument("--nwarm-scale", type=float, default=1.0)
    ap.add_argument("--nskip-scale", type=float, default=1.0)
    ap.add_argument("--nmeas-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-cut", type=float, default=4.0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--cpu-onednn", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--benchmark-kernels", action="store_true")
    ap.add_argument("--benchmark-integrators", type=str, default="")
    ap.add_argument("--benchmark-iter", type=int, default=5)
    ap.add_argument("--fail-on-mismatch", action="store_true")
    args = ap.parse_args()

    torch_root = Path(args.torch_root).resolve()
    if (not torch_root.is_dir()) and (Path(args.torch_root) == DEFAULT_TORCH_ROOT):
        torch_root = FALLBACK_TORCH_ROOT.resolve()
    if not torch_root.is_dir():
        raise SystemExit(f"torch root does not exist: {torch_root}")

    summaries = _discover_torch_summaries(torch_root)
    if not summaries:
        raise SystemExit(f"No torch summaries found under {torch_root}")

    cases = _selected_cases(summaries, args.cases, args.max_cases)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    report_cases = []
    any_fail = False

    for torch_summary, torch_path in cases:
        label = _case_label(torch_summary)
        out_json = outdir / f"jax_{label.replace('@', '_b').replace('x', '_')}.json"
        if bool(args.rerun) or (not out_json.is_file()):
            cmd = _build_jax_command(torch_summary, args, out_json)
            print(f"\n[{label}] running JAX reference")
            print(" ", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(ROOT))
        else:
            print(f"\n[{label}] reusing {out_json}")

        jax_out = json.loads(out_json.read_text(encoding="utf-8"))
        compat = dict(jax_out.get("torch_compatible_summary", {}))
        comparisons = [
            _compare_field(torch_summary, compat, "xi", "xi_err"),
            _compare_field(torch_summary, compat, "chi_m", "chi_m_err"),
            _compare_field(torch_summary, compat, "c2p", "c2p_err"),
            _compare_field(torch_summary, compat, "chi_top", "chi_top_err"),
        ]
        ok = _case_ok(comparisons, sigma_cut=float(args.sigma_cut))
        any_fail = any_fail or (not ok)

        print(f"  torch summary: {torch_path}")
        for comp in comparisons:
            if bool(comp.get("skip", False)):
                continue
            sigma = float(comp["sigma"])
            sigma_txt = f"{sigma:.3f}" if math.isfinite(sigma) else "nan"
            print(
                f"  {comp['field']:>7s}:"
                f" torch={float(comp['torch']):.8g}"
                f" jax={float(comp['jax']):.8g}"
                f" delta={float(comp['delta']):.3e}"
                f" rel={float(comp['rel']):.3e}"
                f" sigma={sigma_txt}"
            )
        print(f"  pass={ok}")

        report_cases.append(
            {
                "label": label,
                "torch_summary": str(torch_path),
                "jax_json": str(out_json),
                "pass": bool(ok),
                "comparisons": comparisons,
            }
        )

    report = {
        "torch_root": str(torch_root),
        "cases": report_cases,
        "sigma_cut": float(args.sigma_cut),
        "pass": not any_fail,
    }

    report_path = Path(args.report_json).resolve() if str(args.report_json).strip() else (outdir / "compare_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport saved to {report_path}")
    print(f"overall pass={report['pass']}")

    if bool(args.fail_on_mismatch) and any_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

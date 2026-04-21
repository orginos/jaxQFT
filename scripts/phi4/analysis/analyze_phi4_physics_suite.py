#!/usr/bin/env python3
"""Unified phi^4 physics analysis over model and blocked-HMC JSON outputs.

This script is the top-level analysis driver for the paper-2 RT/locality study.
It does not rerun training or HMC. Instead, it consumes the already-produced
JSON artifacts from:

- scripts/phi4/analysis/analyze_rg_coarse_eta_gaussian_levels.py
- scripts/phi4/analysis/analyze_hmc_phi4.py (or scripts/phi4/hmc_phi4.py JSON)

and normalizes them into a single schema with:

- per-run rows that preserve seed / replica identity
- family summaries by architecture or HMC ensemble
- pairwise seed spreads by RG level
- pairwise architecture spreads by RG level
- model-vs-HMC comparison tables

The intent is to keep the RG argument explicit: seed-to-seed and
architecture-to-architecture differences are not treated as mere Monte Carlo
noise. They are tracked as level-dependent spreads in theory space.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POINT_MASS = {
    "canonical": -0.4,
    "canonical2": -0.5,
    "canonical3": -0.585,
    "canonical4": -0.70,
}


MODEL_LEVEL_RE = re.compile(r"(?P<point>[^/]+)/(?P<arch>[^/]+)/L(?P<volume>\d+)/s(?P<seed>[^/]+)/levels\.json$")
HMC_JSON_RE = re.compile(r"L(?P<volume>\d+)/m(?P<mass>[-+0-9.p]+)/r(?P<replica>[^/.]+)\.json$")


def _parse_csv_set(text: str) -> set[str]:
    if not text.strip():
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def _parse_point_mass(entries: list[str]) -> dict[str, float]:
    out = dict(DEFAULT_POINT_MASS)
    for item in entries:
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE for --point-mass, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def _normalize_mass_label(text: str) -> float:
    return float(text.replace("p", "."))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _est_mean(block: dict[str, Any], key: str) -> float | None:
    item = block.get(key)
    if isinstance(item, dict):
        return _safe_float(item.get("mean"))
    return _safe_float(item)


def _est_stderr(block: dict[str, Any], key: str) -> float | None:
    item = block.get(key)
    if isinstance(item, dict):
        return _safe_float(item.get("stderr"))
    return None


def _find_scan_entry(obs_block: dict[str, Any], *, k: int) -> dict[str, Any] | None:
    scan = obs_block.get("xi2_momentum_scan")
    if not isinstance(scan, list):
        return None
    for row in scan:
        if int(row.get("k", -1)) == int(k):
            return row
    return None


def _c2_ratio_from_scan(obs_block: dict[str, Any], *, k_num: int, k_den: int = 1) -> float | None:
    chi = _est_mean(obs_block, "chi_m")
    if chi is None or chi <= 0.0:
        return None
    row_num = _find_scan_entry(obs_block, k=k_num)
    row_den = _find_scan_entry(obs_block, k=k_den)
    if row_num is None or row_den is None:
        return None
    hat_num = _safe_float(row_num.get("hat_p2"))
    hat_den = _safe_float(row_den.get("hat_p2"))
    xi_num = _safe_float((row_num.get("xi2") or {}).get("mean"))
    xi_den = _safe_float((row_den.get("xi2") or {}).get("mean"))
    if None in (hat_num, hat_den, xi_num, xi_den):
        return None
    c2_num = chi / (1.0 + float(hat_num) * float(xi_num) * float(xi_num))
    c2_den = chi / (1.0 + float(hat_den) * float(xi_den) * float(xi_den))
    if c2_den <= 0.0:
        return None
    ratio = float(c2_num / c2_den)
    return ratio if math.isfinite(ratio) else None


def _extract_locality_fields(level_from_fine: int, locality_root: dict[str, Any], metrics: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    by_metric = locality_root.get("metrics", {}) if isinstance(locality_root, dict) else {}
    for metric in metrics:
        metric_block = by_metric.get(metric, {})
        levels = metric_block.get("levels", []) if isinstance(metric_block, dict) else []
        match = None
        for row in levels:
            if int(row.get("level_from_fine", -1)) == int(level_from_fine):
                match = row
                break
        prefix = f"loc_{metric}"
        if not isinstance(match, dict):
            out[f"{prefix}_xi_local"] = None
            out[f"{prefix}_offsite_total_abs_response_norm"] = None
            out[f"{prefix}_nearest_neighbor_abs_response_norm"] = None
            out[f"{prefix}_tail_fraction_r1"] = None
            continue
        fit = match.get("fit", {})
        out[f"{prefix}_xi_local"] = _safe_float(fit.get("xi_local"))
        out[f"{prefix}_offsite_total_abs_response_norm"] = _safe_float(match.get("offsite_total_abs_response_norm"))
        out[f"{prefix}_nearest_neighbor_abs_response_norm"] = _safe_float(match.get("nearest_neighbor_abs_response_norm"))
        tail = match.get("tail_fraction_of_offsite_weight")
        if isinstance(tail, list) and len(tail) > 1:
            out[f"{prefix}_tail_fraction_r1"] = _safe_float(tail[1])
        else:
            out[f"{prefix}_tail_fraction_r1"] = None
    return out


def _extract_observables(obs_block: dict[str, Any]) -> dict[str, float | None]:
    return {
        "chi_m": _est_mean(obs_block, "chi_m"),
        "chi_m_stderr": _est_stderr(obs_block, "chi_m"),
        "U4": _est_mean(obs_block, "binder_cumulant"),
        "U4_stderr": _est_stderr(obs_block, "binder_cumulant"),
        "xi2": _est_mean(obs_block, "xi2"),
        "xi2_stderr": _est_stderr(obs_block, "xi2"),
        "xi2_over_L": _est_mean(obs_block, "xi2_over_L"),
        "xi2_over_L_stderr": _est_stderr(obs_block, "xi2_over_L"),
        "C2_ratio_k2_k1": _c2_ratio_from_scan(obs_block, k_num=2, k_den=1),
        "C2_ratio_k3_k1": _c2_ratio_from_scan(obs_block, k_num=3, k_den=1),
    }


def _load_model_rows(path: Path, *, locality_metrics: list[str], include_reweighted: bool, include_unreportable_reweighted: bool) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    match = MODEL_LEVEL_RE.search(str(path))
    if not match:
        raise ValueError(f"could not parse model metadata from {path}")
    point = match.group("point")
    arch = match.group("arch")
    volume = int(match.group("volume"))
    seed = match.group("seed")
    rows: list[dict[str, Any]] = []
    locality_root = raw.get("locality", {})
    for level in raw.get("levels", []):
        level_from_fine = int(level["level_from_fine"])
        base = {
            "source": "model",
            "point": point,
            "mass": _safe_float(raw.get("arch", {}).get("mass")),
            "arch": arch,
            "volume": volume,
            "level_from_fine": level_from_fine,
            "blocked_L": int(level["L"]),
            "sample_kind": "seed",
            "sample_id": seed,
            "path": str(path),
        }
        row_u = {**base, "variant": "unweighted", "reportable": True}
        row_u.update(_extract_observables(level["unweighted"]))
        row_u.update(_extract_locality_fields(level_from_fine, locality_root, locality_metrics))
        rows.append(row_u)

        if include_reweighted:
            reportable = bool(level["reweighted"].get("reportable", False))
            if reportable or include_unreportable_reweighted:
                row_r = {**base, "variant": "reweighted", "reportable": reportable}
                row_r.update(_extract_observables(level["reweighted"]))
                row_r.update(_extract_locality_fields(level_from_fine, locality_root, locality_metrics))
                row_r["ess_fraction"] = _safe_float(level["reweighted"].get("ess_fraction"))
                rows.append(row_r)
    return rows


def _reverse_mass_map(point_mass: dict[str, float]) -> dict[str, str]:
    out: dict[str, str] = {}
    for point, mass in point_mass.items():
        key = f"{mass:.12g}"
        out[key] = point
    return out


def _load_hmc_rows(path: Path, *, mass_to_point: dict[str, str]) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    match = HMC_JSON_RE.search(str(path))
    if not match:
        raise ValueError(f"could not parse HMC metadata from {path}")
    volume = int(match.group("volume"))
    mass = _normalize_mass_label(match.group("mass"))
    replica = match.group("replica")
    point = mass_to_point.get(f"{mass:.12g}")
    rows: list[dict[str, Any]] = []

    blocked = raw.get("blocked_levels", {})
    levels = blocked.get("levels", []) if isinstance(blocked, dict) else []
    if levels:
        iter_levels = levels
    else:
        iter_levels = [
            {
                "level_from_fine": 0,
                "L": int(raw.get("volume", volume)),
                "shape": list(raw.get("shape", [volume, volume])),
                "unweighted": raw.get("derived", {}),
            }
        ]

    for level in iter_levels:
        obs_block = level["unweighted"]
        row = {
            "source": "hmc",
            "point": point,
            "mass": mass,
            "arch": None,
            "volume": volume,
            "level_from_fine": int(level["level_from_fine"]),
            "blocked_L": int(level["L"]),
            "sample_kind": "replica",
            "sample_id": replica,
            "variant": "unweighted",
            "reportable": True,
            "path": str(path),
        }
        row.update(_extract_observables(obs_block))
        rows.append(row)
    return rows


METRIC_KEYS = [
    "chi_m",
    "U4",
    "xi2",
    "xi2_over_L",
    "C2_ratio_k2_k1",
    "C2_ratio_k3_k1",
    "loc_manhattan_xi_local",
    "loc_manhattan_offsite_total_abs_response_norm",
    "loc_manhattan_nearest_neighbor_abs_response_norm",
    "loc_manhattan_tail_fraction_r1",
    "loc_euclidean2_xi_local",
    "loc_euclidean2_offsite_total_abs_response_norm",
    "loc_euclidean2_nearest_neighbor_abs_response_norm",
    "loc_euclidean2_tail_fraction_r1",
]


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        val = _safe_float(row.get(key))
        if val is not None:
            out.append(val)
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_stderr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def _group_family_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"],
            row.get("point"),
            row.get("arch"),
            row["volume"],
            row["level_from_fine"],
            row["variant"],
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, grp in sorted(groups.items()):
        source, point, arch, volume, level_from_fine, variant = key
        summary = {
            "source": source,
            "point": point,
            "arch": arch,
            "volume": int(volume),
            "level_from_fine": int(level_from_fine),
            "variant": variant,
            "n_runs": int(len(grp)),
            "sample_ids": sorted(str(row["sample_id"]) for row in grp),
        }
        for metric in METRIC_KEYS:
            vals = _finite_values(grp, metric)
            summary[metric] = {
                "mean": _mean(vals),
                "stderr_between": _sample_stderr(vals),
                "n_finite": int(len(vals)),
            }
        out.append(summary)
    return out


def _pairwise_abs_stats(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    diffs = [abs(a - b) for a, b in combinations(values, 2)]
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "n_pairs": int(arr.size),
        "mean_abs_diff": float(np.mean(arr)),
        "rms_abs_diff": float(np.sqrt(np.mean(arr * arr))),
        "max_abs_diff": float(np.max(arr)),
    }


def _seed_spreads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source"] != "model":
            continue
        key = (row["point"], row["arch"], row["volume"], row["level_from_fine"], row["variant"])
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, grp in sorted(groups.items()):
        point, arch, volume, level_from_fine, variant = key
        rec = {
            "point": point,
            "arch": arch,
            "volume": int(volume),
            "level_from_fine": int(level_from_fine),
            "variant": variant,
            "n_runs": int(len(grp)),
        }
        for metric in METRIC_KEYS:
            stats = _pairwise_abs_stats(_finite_values(grp, metric))
            if stats is not None:
                rec[metric] = stats
        out.append(rec)
    return out


def _architecture_spreads(family_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in family_summaries:
        if row["source"] != "model":
            continue
        key = (row["point"], row["volume"], row["level_from_fine"], row["variant"])
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, grp in sorted(groups.items()):
        point, volume, level_from_fine, variant = key
        rec = {
            "point": point,
            "volume": int(volume),
            "level_from_fine": int(level_from_fine),
            "variant": variant,
            "arches": sorted(str(row["arch"]) for row in grp),
        }
        for metric in METRIC_KEYS:
            vals = []
            for row in grp:
                mean = row[metric]["mean"]
                if mean is not None:
                    vals.append(float(mean))
            stats = _pairwise_abs_stats(vals)
            if stats is not None:
                rec[metric] = stats
        out.append(rec)
    return out


def _attach_contraction(spreads: list[dict[str, Any]], *, seed_mode: bool) -> list[dict[str, Any]]:
    baselines: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in spreads:
        if int(row["level_from_fine"]) != 0:
            continue
        if seed_mode:
            key = (row["point"], row["arch"], row["volume"], row["variant"])
        else:
            key = (row["point"], row["volume"], row["variant"])
        base = {}
        for metric in METRIC_KEYS:
            stats = row.get(metric)
            if isinstance(stats, dict):
                val = _safe_float(stats.get("mean_abs_diff"))
                if val is not None and val > 0.0:
                    base[metric] = val
        baselines[key] = base

    out = []
    for row in spreads:
        if seed_mode:
            key = (row["point"], row["arch"], row["volume"], row["variant"])
        else:
            key = (row["point"], row["volume"], row["variant"])
        base = baselines.get(key, {})
        new_row = dict(row)
        for metric in METRIC_KEYS:
            stats = new_row.get(metric)
            base_val = base.get(metric)
            if not isinstance(stats, dict) or base_val is None:
                continue
            cur = _safe_float(stats.get("mean_abs_diff"))
            if cur is None:
                continue
            stats["contraction_vs_level0"] = float(cur / base_val)
        out.append(new_row)
    return out


def _model_vs_hmc(family_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hmc_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    model_rows: list[dict[str, Any]] = []
    for row in family_summaries:
        key = (row.get("point"), row["volume"], row["level_from_fine"], row["variant"])
        if row["source"] == "hmc":
            hmc_lookup[key] = row
        elif row["source"] == "model":
            model_rows.append(row)
    out: list[dict[str, Any]] = []
    for row in model_rows:
        key = (row.get("point"), row["volume"], row["level_from_fine"], row["variant"])
        hmc = hmc_lookup.get(key)
        if hmc is None:
            continue
        rec = {
            "point": row["point"],
            "arch": row["arch"],
            "volume": row["volume"],
            "level_from_fine": row["level_from_fine"],
            "variant": row["variant"],
        }
        for metric in METRIC_KEYS:
            m_mean = _safe_float(row[metric]["mean"])
            h_mean = _safe_float(hmc[metric]["mean"])
            if m_mean is None or h_mean is None:
                continue
            m_err = _safe_float(row[metric].get("stderr_between"))
            h_err = _safe_float(hmc[metric].get("stderr_between"))
            comb = None
            z = None
            if m_err is not None and h_err is not None:
                comb = float(math.hypot(m_err, h_err))
                if comb > 0.0:
                    z = float((m_mean - h_mean) / comb)
            rec[metric] = {
                "model_mean": m_mean,
                "hmc_mean": h_mean,
                "delta": float(m_mean - h_mean),
                "rel_delta": float((m_mean - h_mean) / h_mean) if h_mean != 0.0 else None,
                "combined_stderr": comb,
                "z_score": z,
            }
        out.append(rec)
    return out


def _iter_model_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("levels.json"))


def _iter_hmc_jsons(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.name.startswith("r"))


def _filter_rows(rows: list[dict[str, Any]], *, points: set[str], arches: set[str], volumes: set[int]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if points and row.get("point") not in points:
            continue
        if arches and row.get("arch") is not None and row.get("arch") not in arches:
            continue
        if volumes and int(row["volume"]) not in volumes:
            continue
        out.append(row)
    return out


def _print_level0_summary(family_summaries: list[dict[str, Any]]) -> None:
    print("Level-0 family summaries")
    print(f"{'source':<8} {'point':<10} {'arch':<8} {'L':>4} {'n':>3} {'xi2/L':>14} {'U4':>14} {'chi_m':>14}")
    for row in family_summaries:
        if int(row["level_from_fine"]) != 0:
            continue
        xi = row["xi2_over_L"]["mean"]
        u4 = row["U4"]["mean"]
        chi = row["chi_m"]["mean"]
        arch = row["arch"] if row["arch"] is not None else "-"
        print(
            f"{row['source']:<8} {str(row.get('point')):<10} {arch:<8} {int(row['volume']):4d} {int(row['n_runs']):3d} "
            f"{xi if xi is not None else float('nan'):>14.6f} "
            f"{u4 if u4 is not None else float('nan'):>14.6f} "
            f"{chi if chi is not None else float('nan'):>14.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified phi^4 RT/locality analysis over model and HMC JSON outputs.")
    ap.add_argument("--model-root", action="append", default=[], help="Root containing model levels.json trees")
    ap.add_argument("--hmc-root", action="append", default=[], help="Root containing blocked-HMC JSON trees")
    ap.add_argument("--points", type=str, default="", help="Comma-separated point filter")
    ap.add_argument("--arches", type=str, default="", help="Comma-separated architecture filter")
    ap.add_argument("--volumes", type=str, default="", help="Comma-separated volume filter")
    ap.add_argument("--include-reweighted", action="store_true", help="Include reportable reweighted model observables")
    ap.add_argument(
        "--include-unreportable-reweighted",
        action="store_true",
        help="Include reweighted model rows even when reportable=false",
    )
    ap.add_argument("--locality-metrics", type=str, default="manhattan,euclidean2")
    ap.add_argument("--point-mass", action="append", default=[], help="Override point-to-mass map with NAME=VALUE")
    ap.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
    args = ap.parse_args()

    point_mass = _parse_point_mass(list(args.point_mass))
    mass_to_point = _reverse_mass_map(point_mass)
    points = _parse_csv_set(args.points)
    arches = _parse_csv_set(args.arches)
    volumes = {int(x) for x in _parse_csv_set(args.volumes)}
    locality_metrics = [m.strip() for m in args.locality_metrics.split(",") if m.strip()]

    rows: list[dict[str, Any]] = []
    for root_text in args.model_root:
        root = Path(root_text).expanduser().resolve()
        for path in _iter_model_jsons(root):
            rows.extend(
                _load_model_rows(
                    path,
                    locality_metrics=locality_metrics,
                    include_reweighted=bool(args.include_reweighted),
                    include_unreportable_reweighted=bool(args.include_unreportable_reweighted),
                )
            )
    for root_text in args.hmc_root:
        root = Path(root_text).expanduser().resolve()
        for path in _iter_hmc_jsons(root):
            rows.extend(_load_hmc_rows(path, mass_to_point=mass_to_point))

    rows = _filter_rows(rows, points=points, arches=arches, volumes=volumes)
    family_summaries = _group_family_summaries(rows)
    seed_spreads = _attach_contraction(_seed_spreads(rows), seed_mode=True)
    arch_spreads = _attach_contraction(_architecture_spreads(family_summaries), seed_mode=False)
    model_vs_hmc = _model_vs_hmc(family_summaries)

    out = {
        "point_mass": point_mass,
        "n_rows": int(len(rows)),
        "rows": rows,
        "family_summaries": family_summaries,
        "seed_spreads": seed_spreads,
        "architecture_spreads": arch_spreads,
        "model_vs_hmc": model_vs_hmc,
    }

    _print_level0_summary(family_summaries)
    print("")
    print(f"Loaded {len(rows)} normalized rows")
    print(f"Family summaries: {len(family_summaries)}")
    print(f"Seed spread rows: {len(seed_spreads)}")
    print(f"Architecture spread rows: {len(arch_spreads)}")
    print(f"Model-vs-HMC rows: {len(model_vs_hmc)}")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HMC diagnostics and knockout analysis for the RG coarse-eta Gaussian flow."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


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
from jaxqft.core.update import hmc
from jaxqft.models.phi4 import Phi4
from jaxqft.models.phi4_mg import realnvp_f
from jaxqft.models.phi4_rg_cond_flow import _merge_rg, _split_rg, _std_normal_log_prob
from jaxqft.models.phi4_rg_coarse_eta_flow import _eta_flow_f
from jaxqft.models.phi4_rg_coarse_eta_gaussian_flow import (
    _eta_gaussian_f,
    _terminal_gaussian_params,
    _triangular_linear_f4,
    rg_coarse_eta_gaussian_flow_log_prob,
)


def tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["weights"] = tree_to_jax(payload["weights"])
    return payload


def _delta_stats(delta_s: np.ndarray) -> dict:
    centered = delta_s - float(np.mean(delta_s))
    logw = -centered
    logw_shift = logw - float(np.max(logw))
    foo = np.exp(logw_shift)
    w = foo / np.mean(foo)
    ess = (np.mean(foo) ** 2) / np.mean(foo * foo)
    return {
        "n": int(delta_s.size),
        "mean_action_diff": float(np.mean(delta_s)),
        "std_action_diff": float(np.std(centered)),
        "max_abs_action_diff": float(np.max(np.abs(centered))),
        "std_reweight": float(np.std(w)),
        "ess": float(ess),
    }


def _moment_metrics(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    nvec, dim = arr.shape
    mean = np.mean(arr, axis=0)
    centered = arr - mean[None, :]
    cov = centered.T @ centered / max(nvec, 1)
    diag = np.diag(cov)
    std = np.sqrt(np.maximum(diag, 1e-12))
    normed = centered / std[None, :]
    skew = np.mean(normed**3, axis=0)
    kurt = np.mean(normed**4, axis=0) - 3.0
    cov_off = cov - np.diag(diag)
    return {
        "n_vectors": int(nvec),
        "dim": int(dim),
        "mean_norm": float(np.linalg.norm(mean)),
        "cov_diag": diag.tolist(),
        "cov_fro_err": float(np.linalg.norm(cov - np.eye(dim))),
        "cov_offdiag_fro": float(np.linalg.norm(cov_off)),
        "max_var_err": float(np.max(np.abs(diag - 1.0))),
        "skew_l2": float(np.linalg.norm(skew)),
        "kurt_l2": float(np.linalg.norm(kurt)),
    }


def _neighbor_metrics(field: np.ndarray) -> dict:
    field = np.asarray(field, dtype=np.float64)
    centered = field - np.mean(field, axis=(0, 1, 2), keepdims=True)
    norm = max(np.prod(centered.shape[:3]), 1)
    xshift = np.roll(centered, shift=-1, axis=2)
    yshift = np.roll(centered, shift=-1, axis=1)
    cov_x = np.einsum("bxyi,bxyj->ij", centered, xshift) / norm
    cov_y = np.einsum("bxyi,bxyj->ij", centered, yshift) / norm
    return {
        "nn_x_fro": float(np.linalg.norm(cov_x)),
        "nn_y_fro": float(np.linalg.norm(cov_y)),
        "nn_x_max_abs": float(np.max(np.abs(cov_x))),
        "nn_y_max_abs": float(np.max(np.abs(cov_y))),
    }


def _field_metrics(field: np.ndarray) -> dict:
    flat = field.reshape((-1, field.shape[-1]))
    out = _moment_metrics(flat)
    out.update(_neighbor_metrics(field))
    return out


def run_hmc_samples(
    *,
    shape: tuple[int, int],
    lam: float,
    mass: float,
    nwarm: int,
    nmeas: int,
    nskip: int,
    batch_size: int,
    nmd: int,
    tau: float,
) -> tuple[np.ndarray, dict]:
    theory = Phi4(shape, lam, mass, batch_size=batch_size)
    phi = theory.hotStart()
    integrator = minnorm2(theory.force, theory.evolveQ, nmd, tau)
    chain = hmc(T=theory, I=integrator, verbose=False)

    tic = time.perf_counter()
    phi = chain.evolve(phi, nwarm)
    warm_s = time.perf_counter() - tic

    samples = []
    tic = time.perf_counter()
    for _ in range(nmeas):
        phi = chain.evolve(phi, nskip)
        samples.append(np.asarray(phi))
    meas_s = time.perf_counter() - tic

    arr = np.concatenate(samples, axis=0)
    info = {
        "shape": list(shape),
        "lam": float(lam),
        "mass": float(mass),
        "nwarm": int(nwarm),
        "nmeas": int(nmeas),
        "nskip": int(nskip),
        "batch_size": int(batch_size),
        "nmd": int(nmd),
        "tau": float(tau),
        "n_samples": int(arr.shape[0]),
        "acceptance": float(chain.calc_Acceptance()),
        "warmup_sec": float(warm_s),
        "measure_sec": float(meas_s),
        "traj_usec": float(meas_s * 1e6 / max(1, nmeas * nskip)),
    }
    return arr, info


def _collect_level_trace(cfg: dict, weights: dict, x: jax.Array):
    n_nonterminal = max(cfg["depth"] - 1, 0)
    levels = [None] * n_nonterminal

    def rec(xx: jax.Array, level: int):
        if level >= cfg["depth"] - 1:
            flat = xx.reshape((xx.shape[0], 4))
            z_after_flow, ldj_flow = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
            z_after_gauss, ldj_gauss = _triangular_linear_f4(
                z_after_flow, *_terminal_gaussian_params(cfg, weights)
            )
            term = {
                "x": xx,
                "after_flow": z_after_flow.reshape(xx.shape),
                "after_gaussian": z_after_gauss.reshape(xx.shape),
                "ldj_flow": ldj_flow,
                "ldj_gaussian": ldj_gauss,
            }
            return term

        x_coarse, x_fine = _split_rg(xx, cfg["rg_mode"])
        term = rec(x_coarse, level + 1)
        z_after_flow, ldj_flow = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
        z_after_gauss, ldj_gauss = _eta_gaussian_f(cfg, weights, z_after_flow, x_coarse, level)
        levels[level] = {
            "coarse": x_coarse,
            "raw_eta": x_fine,
            "after_flow": z_after_flow,
            "after_gaussian": z_after_gauss,
            "ldj_flow": ldj_flow,
            "ldj_gaussian": ldj_gauss,
        }
        return term

    terminal = rec(x, 0)
    return levels, terminal


def _stack_level_buffers(buffers: list[list[np.ndarray]]) -> np.ndarray:
    if not buffers:
        raise ValueError("No buffers to stack")
    return np.concatenate(buffers, axis=0)


def run_hmc_level_diagnostic(cfg: dict, weights: dict, samples: np.ndarray, chunk_size: int) -> dict:
    level_buffers = None
    terminal_buffers = None

    for start in range(0, samples.shape[0], chunk_size):
        stop = min(samples.shape[0], start + chunk_size)
        xx = jnp.asarray(samples[start:stop], dtype=jnp.float32)
        levels, terminal = _collect_level_trace(cfg, weights, xx)

        if level_buffers is None:
            level_buffers = [
                {"raw_eta": [], "after_flow": [], "after_gaussian": [], "coarse": []}
                for _ in range(len(levels))
            ]
            terminal_buffers = {"x": [], "after_flow": [], "after_gaussian": []}

        for level, tr in enumerate(levels):
            for key in ("raw_eta", "after_flow", "after_gaussian", "coarse"):
                level_buffers[level][key].append(np.asarray(tr[key]))
        for key in ("x", "after_flow", "after_gaussian"):
            terminal_buffers[key].append(np.asarray(terminal[key]))

    assert level_buffers is not None
    assert terminal_buffers is not None

    levels_out = []
    for level, buf in enumerate(level_buffers):
        raw_eta = _stack_level_buffers(buf["raw_eta"])
        after_flow = _stack_level_buffers(buf["after_flow"])
        after_gaussian = _stack_level_buffers(buf["after_gaussian"])
        coarse = _stack_level_buffers(buf["coarse"])
        levels_out.append(
            {
                "level": int(level),
                "coarse_shape": list(raw_eta.shape[1:3]),
                "n_sites": int(raw_eta.shape[0] * raw_eta.shape[1] * raw_eta.shape[2]),
                "coarse_field": _moment_metrics(coarse.reshape((-1, 1))),
                "raw_eta": _field_metrics(raw_eta),
                "after_flow": _field_metrics(after_flow),
                "after_gaussian": _field_metrics(after_gaussian),
                "flow_cov_gain": float(
                    _field_metrics(raw_eta)["cov_fro_err"] - _field_metrics(after_flow)["cov_fro_err"]
                ),
                "gaussian_cov_gain": float(
                    _field_metrics(after_flow)["cov_fro_err"] - _field_metrics(after_gaussian)["cov_fro_err"]
                ),
            }
        )

    term_x = _stack_level_buffers(terminal_buffers["x"]).reshape((-1, 4))
    term_flow = _stack_level_buffers(terminal_buffers["after_flow"]).reshape((-1, 4))
    term_gauss = _stack_level_buffers(terminal_buffers["after_gaussian"]).reshape((-1, 4))
    terminal_out = {
        "x": _moment_metrics(term_x),
        "after_flow": _moment_metrics(term_flow),
        "after_gaussian": _moment_metrics(term_gauss),
        "flow_cov_gain": float(_moment_metrics(term_x)["cov_fro_err"] - _moment_metrics(term_flow)["cov_fro_err"]),
        "gaussian_cov_gain": float(
            _moment_metrics(term_flow)["cov_fro_err"] - _moment_metrics(term_gauss)["cov_fro_err"]
        ),
    }
    return {"levels": levels_out, "terminal": terminal_out}


def _default_controls(cfg: dict) -> dict:
    n_nonterminal = max(cfg["depth"] - 1, 0)
    return {
        "eta_flow": [True] * n_nonterminal,
        "eta_gaussian": [True] * n_nonterminal,
        "terminal_flow": True,
        "terminal_gaussian": True,
    }


def _copy_controls(ctrl: dict) -> dict:
    return {
        "eta_flow": list(ctrl["eta_flow"]),
        "eta_gaussian": list(ctrl["eta_gaussian"]),
        "terminal_flow": bool(ctrl["terminal_flow"]),
        "terminal_gaussian": bool(ctrl["terminal_gaussian"]),
    }


def _controlled_inverse(cfg: dict, weights: dict, x: jax.Array, controls: dict, level: int = 0):
    if level >= cfg["depth"] - 1:
        flat = x.reshape((x.shape[0], 4))
        if controls["terminal_flow"]:
            z_mid, ldj_flow = realnvp_f(cfg["terminal_flow_cfg"], weights["terminal_flow"], flat)
        else:
            z_mid = flat
            ldj_flow = jnp.zeros((x.shape[0],), dtype=x.dtype)
        if controls["terminal_gaussian"]:
            z_flat, ldj_gauss = _triangular_linear_f4(z_mid, *_terminal_gaussian_params(cfg, weights))
        else:
            z_flat = z_mid
            ldj_gauss = jnp.zeros((x.shape[0],), dtype=x.dtype)
        return z_flat.reshape(x.shape), ldj_flow + ldj_gauss

    x_coarse, x_fine = _split_rg(x, cfg["rg_mode"])
    z_coarse, ldj_coarse = _controlled_inverse(cfg, weights, x_coarse, controls, level + 1)

    if controls["eta_flow"][level]:
        z_tmp, ldj_flow = _eta_flow_f(cfg, weights, x_fine, x_coarse, level)
    else:
        z_tmp = x_fine
        ldj_flow = jnp.zeros((x.shape[0],), dtype=x.dtype)

    if controls["eta_gaussian"][level]:
        z_fine, ldj_gauss = _eta_gaussian_f(cfg, weights, z_tmp, x_coarse, level)
    else:
        z_fine = z_tmp
        ldj_gauss = jnp.zeros((x.shape[0],), dtype=x.dtype)

    return _merge_rg(z_coarse, z_fine, cfg["rg_mode"]), ldj_coarse + ldj_flow + ldj_gauss


def _controlled_log_prob(cfg: dict, weights: dict, x: jax.Array, controls: dict) -> jax.Array:
    z, ldj = _controlled_inverse(cfg, weights, x, controls, level=0)
    return _std_normal_log_prob(z, sum_axes=(1, 2)) + ldj


def _score_controls(
    cfg: dict,
    weights: dict,
    theory: Phi4,
    samples: np.ndarray,
    controls: dict,
    chunk_size: int,
) -> dict:
    deltas = []
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(samples.shape[0], start + chunk_size)
        xx = jnp.asarray(samples[start:stop], dtype=jnp.float32)
        logp = _controlled_log_prob(cfg, weights, xx, controls)
        ds = np.asarray(logp + theory.action(xx))
        deltas.append(ds)
    return _delta_stats(np.concatenate(deltas, axis=0))


def run_knockout_analysis(
    cfg: dict,
    weights: dict,
    theory: Phi4,
    samples: np.ndarray,
    chunk_size: int,
    include_grouped: bool,
) -> dict:
    base_controls = _default_controls(cfg)
    baseline = _score_controls(cfg, weights, theory, samples, base_controls, chunk_size)
    results = []

    for level in range(max(cfg["depth"] - 1, 0)):
        ctrl = _copy_controls(base_controls)
        ctrl["eta_flow"][level] = False
        stats = _score_controls(cfg, weights, theory, samples, ctrl, chunk_size)
        results.append(
            {
                "name": f"eta_flow_L{level}",
                "kind": "eta_flow",
                "level": int(level),
                "stats": stats,
                "delta_std_action_diff": float(stats["std_action_diff"] - baseline["std_action_diff"]),
                "delta_ess": float(stats["ess"] - baseline["ess"]),
            }
        )

        ctrl = _copy_controls(base_controls)
        ctrl["eta_gaussian"][level] = False
        stats = _score_controls(cfg, weights, theory, samples, ctrl, chunk_size)
        results.append(
            {
                "name": f"eta_gaussian_L{level}",
                "kind": "eta_gaussian",
                "level": int(level),
                "stats": stats,
                "delta_std_action_diff": float(stats["std_action_diff"] - baseline["std_action_diff"]),
                "delta_ess": float(stats["ess"] - baseline["ess"]),
            }
        )

    for name, key in (("terminal_flow", "terminal_flow"), ("terminal_gaussian", "terminal_gaussian")):
        ctrl = _copy_controls(base_controls)
        ctrl[key] = False
        stats = _score_controls(cfg, weights, theory, samples, ctrl, chunk_size)
        results.append(
            {
                "name": name,
                "kind": key,
                "level": None,
                "stats": stats,
                "delta_std_action_diff": float(stats["std_action_diff"] - baseline["std_action_diff"]),
                "delta_ess": float(stats["ess"] - baseline["ess"]),
            }
        )

    if include_grouped:
        grouped_specs = [
            ("all_eta_flow", lambda c: c["eta_flow"].__setitem__(slice(None), [False] * len(c["eta_flow"]))),
            (
                "all_eta_gaussian",
                lambda c: c["eta_gaussian"].__setitem__(slice(None), [False] * len(c["eta_gaussian"])),
            ),
            ("all_terminal", lambda c: (c.__setitem__("terminal_flow", False), c.__setitem__("terminal_gaussian", False))),
        ]
        for name, fn in grouped_specs:
            ctrl = _copy_controls(base_controls)
            fn(ctrl)
            stats = _score_controls(cfg, weights, theory, samples, ctrl, chunk_size)
            results.append(
                {
                    "name": name,
                    "kind": "group",
                    "level": None,
                    "stats": stats,
                    "delta_std_action_diff": float(stats["std_action_diff"] - baseline["std_action_diff"]),
                    "delta_ess": float(stats["ess"] - baseline["ess"]),
                }
            )

    ranked = sorted(results, key=lambda x: x["delta_std_action_diff"], reverse=True)
    return {"baseline": baseline, "results": results, "ranked": ranked}


def _print_hmc_info(info: dict):
    print("HMC target sampling:")
    print(
        f"  shape={tuple(info['shape'])} lam={info['lam']} mass={info['mass']}"
        f" batch={info['batch_size']} nwarm={info['nwarm']} nmeas={info['nmeas']} nskip={info['nskip']}"
    )
    print(
        f"  nmd={info['nmd']} tau={info['tau']} acceptance={info['acceptance']:.4f}"
        f" traj_usec={info['traj_usec']:.1f}"
    )
    print(f"  samples={info['n_samples']} warmup_sec={info['warmup_sec']:.2f} measure_sec={info['measure_sec']:.2f}")


def _print_level_diagnostic(diag: dict):
    print("\nHMC level diagnostic:")
    for level in diag["levels"]:
        raw = level["raw_eta"]
        flow = level["after_flow"]
        gauss = level["after_gaussian"]
        print(
            f"  L{level['level']} coarse={tuple(level['coarse_shape'])}"
            f" raw_cov={raw['cov_fro_err']:.4f} flow_cov={flow['cov_fro_err']:.4f} gauss_cov={gauss['cov_fro_err']:.4f}"
            f" raw_nn={0.5*(raw['nn_x_fro']+raw['nn_y_fro']):.4f}"
            f" gauss_nn={0.5*(gauss['nn_x_fro']+gauss['nn_y_fro']):.4f}"
        )
        print(
            f"     skew raw/flow/gauss = {raw['skew_l2']:.4f} / {flow['skew_l2']:.4f} / {gauss['skew_l2']:.4f},"
            f" kurt raw/flow/gauss = {raw['kurt_l2']:.4f} / {flow['kurt_l2']:.4f} / {gauss['kurt_l2']:.4f}"
        )
    term = diag["terminal"]
    print(
        f"  terminal raw_cov={term['x']['cov_fro_err']:.4f}"
        f" flow_cov={term['after_flow']['cov_fro_err']:.4f}"
        f" gauss_cov={term['after_gaussian']['cov_fro_err']:.4f}"
    )


def _print_knockout(knock: dict):
    base = knock["baseline"]
    print("\nKnockout baseline on HMC samples:")
    print(
        f"  std(ΔS)={base['std_action_diff']:.6f}"
        f" std(w)={base['std_reweight']:.6f}"
        f" ESS={base['ess']:.6f}"
    )
    print("\nKnockout ranking by Δ std(ΔS):")
    for row in knock["ranked"]:
        print(
            f"  {row['name']:<18} Δstd(ΔS)={row['delta_std_action_diff']:+.6f}"
            f" ΔESS={row['delta_ess']:+.6f}"
            f" -> std(ΔS)={row['stats']['std_action_diff']:.6f} ESS={row['stats']['ess']:.6f}"
        )


def main():
    ap = argparse.ArgumentParser(description="Analyze the RG coarse-eta Gaussian flow on HMC target samples.")
    ap.add_argument("--resume", type=str, required=True, help="checkpoint to analyze")
    ap.add_argument("--tests", type=str, default="hmc,knockout", help="comma list: hmc,knockout")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--nwarm", type=int, default=200)
    ap.add_argument("--nmeas", type=int, default=64)
    ap.add_argument("--nskip", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--nmd", type=int, default=7)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--shape", type=str, default="", help="override shape as H,W; default inferred from checkpoint")
    ap.add_argument("--include-grouped", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    tests = {t.strip() for t in args.tests.split(",") if t.strip()}
    ckpt = load_checkpoint(args.resume)
    cfg = ckpt["cfg"]
    weights = ckpt["weights"]
    arch = ckpt["arch"]

    if args.shape:
        shape = tuple(int(v.strip()) for v in args.shape.split(",") if v.strip())
        if len(shape) != 2:
            raise ValueError("shape override must be H,W")
    else:
        shape = (int(arch["L"]), int(arch["L"]))

    lam = float(arch["lam"] if args.lam is None else args.lam)
    mass = float(arch["mass"] if args.mass is None else args.mass)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(
        "Checkpoint model:"
        f" L={shape[0]} mass={mass} lam={lam}"
        f" width={arch.get('width')}"
        f" n_cycles={arch.get('n_cycles')}"
        f" radius={arch.get('radius')}"
        f" eta_gaussian={arch.get('eta_gaussian')}"
        f" terminal_prior={arch.get('terminal_prior')}"
        f" parity={arch.get('parity')}"
    )

    samples, hmc_info = run_hmc_samples(
        shape=shape,
        lam=lam,
        mass=mass,
        nwarm=args.nwarm,
        nmeas=args.nmeas,
        nskip=args.nskip,
        batch_size=args.batch_size,
        nmd=args.nmd,
        tau=args.tau,
    )
    _print_hmc_info(hmc_info)

    out = {"checkpoint": args.resume, "hmc": hmc_info}

    if "hmc" in tests:
        diag = run_hmc_level_diagnostic(cfg, weights, samples, chunk_size=args.chunk_size)
        out["hmc_level_diagnostic"] = diag
        _print_level_diagnostic(diag)

    if "knockout" in tests:
        theory = Phi4(shape, lam, mass, batch_size=args.chunk_size)
        knock = run_knockout_analysis(
            cfg,
            weights,
            theory,
            samples,
            chunk_size=args.chunk_size,
            include_grouped=args.include_grouped,
        )
        out["knockout"] = knock
        _print_knockout(knock)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved analysis to {args.json_out}")


if __name__ == "__main__":
    main()

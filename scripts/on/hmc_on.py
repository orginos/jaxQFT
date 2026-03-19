#!/usr/bin/env python3
"""Standard HMC/SMD production script for the O(N) sigma model."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _cli_value(argv, flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _cli_bool(argv, on_flag: str, off_flag: str):
    val = None
    for a in argv:
        if a == on_flag:
            val = True
        elif a == off_flag:
            val = False
    return val


def _append_xla_flag(flag: str):
    cur = os.environ.get("XLA_FLAGS", "").split()
    if flag not in cur:
        cur.append(flag)
        os.environ["XLA_FLAGS"] = " ".join(cur).strip()


def _configure_cpu_xla_flags_from_cli_env():
    threads = _cli_value(sys.argv, "--cpu-threads")
    if threads is None:
        threads = os.environ.get("JAXQFT_CPU_THREADS")
    if threads is not None:
        n = int(threads)
        if n > 0:
            _append_xla_flag("--xla_cpu_multi_thread_eigen=true")
            _append_xla_flag(f"intra_op_parallelism_threads={n}")

    onednn = _cli_bool(sys.argv, "--cpu-onednn", "--no-cpu-onednn")
    if onednn is None:
        env = os.environ.get("JAXQFT_CPU_ONEDNN")
        if env is not None:
            onednn = env.strip().lower() not in ("0", "false", "no", "off")
    if onednn is True:
        _append_xla_flag("--xla_cpu_use_onednn=true")
    elif onednn is False:
        _append_xla_flag("--xla_cpu_use_onednn=false")


_configure_cpu_xla_flags_from_cli_env()

if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax


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
from jaxqft.core.update import SMD, hmc
from jaxqft.models.on_sigma import ONSigmaModel, _correlation_length, _parse_shape
from jaxqft.stats import integrated_autocorr_time


def _build_integrator(name: str, theory: ONSigmaModel, nmd: int, tau: float):
    if name == "leapfrog":
        return leapfrog(theory.force, theory.evolveQ, int(nmd), float(tau))
    if name == "minnorm2":
        return minnorm2(theory.force, theory.evolveQ, int(nmd), float(tau))
    if name == "forcegrad":
        return force_gradient(theory.force, theory.evolveQ, int(nmd), float(tau))
    if name == "minnorm4pf4":
        return minnorm4pf4(theory.force, theory.evolveQ, int(nmd), float(tau))
    raise ValueError(f"Unknown integrator: {name}")


def _save_checkpoint(path: str, q, theory, chain, state: dict, config: dict):
    update_p = getattr(chain, "p", None)
    payload = {
        "q": np.asarray(q),
        "theory_key": np.asarray(theory.key),
        "update_key": np.asarray(chain.key),
        "update_momentum": None if update_p is None else np.asarray(update_p),
        "hmc_key": np.asarray(chain.key),
        "accept_reject": np.asarray(chain.AcceptReject, dtype=np.float32),
        "state": state,
        "config": config,
        "timestamp": time.time(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _run_one_trajectory(q, chain):
    if isinstance(chain, SMD):
        return chain.evolve(q, 1, warmup=False)
    return chain.evolve(q, 1)


def _run_one_trajectory_hmc_no_ar(q, chain):
    if isinstance(chain, SMD):
        return chain.evolve(q, 1, warmup=True)
    p0 = chain.T.refreshP()
    _, q1 = chain.I.integrate(p0, q)
    return jax.block_until_ready(q1)


def _analyze(obs2d: np.ndarray) -> dict:
    arr = np.asarray(obs2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D observable array, got shape {arr.shape}")
    batch_mean = arr.mean(axis=1)
    grand_mean = float(np.mean(batch_mean))
    iat = integrated_autocorr_time(batch_mean, method="gamma")
    sigma_mean = float(iat.get("sigma_mean", np.nan))
    tau_int = float(iat.get("tau_int", np.nan))
    ess = float(iat.get("ess", np.nan)) * float(arr.shape[1])
    return {
        "mean": grand_mean,
        "sigma": sigma_mean,
        "tau_int": tau_int,
        "ess": ess,
    }


def _compute_observables(theory: ONSigmaModel, q, mom_axis: int = 0) -> dict[str, np.ndarray]:
    q_np = np.asarray(q, dtype=np.float64)
    action = np.asarray(theory.action(q), dtype=np.float64) / float(theory.Vol)
    av_sigma = np.asarray(theory.average_spin(q), dtype=np.float64)
    chi_m = float(theory.Vol) * np.sum(av_sigma * av_sigma, axis=1)
    c2p = np.asarray(theory.first_momentum_structure_factor(q, axis=mom_axis), dtype=np.float64)
    out = {
        "E": action,
        "av_sigma": av_sigma,
        "chi_m_raw": chi_m,
        "C2p": c2p,
    }
    if theory.Nd == 2 and theory.ncomp == 3:
        out["Q"] = np.asarray(theory.topological_charge(q), dtype=np.float64)
    _ = q_np
    return out


def main():
    ap = argparse.ArgumentParser(description="Standard MCMC for the O(N) sigma model")
    ap.add_argument("--shape", type=str, default="32,32")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layout", type=str, default="auto", choices=["BN...", "B...N", "auto"])
    ap.add_argument("--exp-method", type=str, default="auto", choices=["auto", "expm", "rodrigues"])
    ap.add_argument("--start", type=str, default="hot", choices=["hot", "cold"])
    ap.add_argument("--update", type=str, default="hmc", choices=["hmc", "smd", "ghmc"])
    ap.add_argument("--integrator", type=str, default="minnorm2", choices=["leapfrog", "minnorm2", "forcegrad", "minnorm4pf4"])
    ap.add_argument("--nmd", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--smd-gamma", type=float, default=0.3)
    ap.add_argument("--warmup-no-ar", type=int, default=0, help="HMC-only initial no-AR warmup")
    ap.add_argument("--nwarm", type=int, default=100)
    ap.add_argument("--nmeas", type=int, default=200)
    ap.add_argument("--nskip", type=int, default=10)
    ap.add_argument("--mom-axis", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--cpu-threads", type=int, default=int(os.environ.get("JAXQFT_CPU_THREADS", "0") or 0))
    ap.add_argument("--cpu-onednn", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    lattice_shape = _parse_shape(args.shape)
    update_name = "smd" if str(args.update).lower() == "ghmc" else str(args.update).lower()
    selected_layout = str(args.layout)
    if selected_layout == "auto":
        bench = ONSigmaModel.benchmark_layout(
            lattice_shape=lattice_shape,
            beta=float(args.beta),
            ncomp=int(args.ncomp),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            exp_method=str(args.exp_method),
            n_iter=5,
        )
        print("Layout benchmark (s for action+force+evolveQ):", bench)
        selected_layout = min(bench, key=bench.get)
        print("Auto-selected layout:", selected_layout)

    theory = ONSigmaModel(
        lattice_shape=lattice_shape,
        beta=float(args.beta),
        ncomp=int(args.ncomp),
        batch_size=int(args.batch_size),
        layout=selected_layout,
        seed=int(args.seed),
        exp_method=str(args.exp_method),
    )
    theory.kinetic = jax.jit(theory.kinetic)
    theory.refresh_p_with_key = jax.jit(theory.refresh_p_with_key)

    chain = None
    q = None
    ckpt = _load_checkpoint(args.resume) if args.resume else None
    if ckpt is not None:
        q = jax.device_put(np.asarray(ckpt["q"]))
        theory.key = jax.device_put(np.asarray(ckpt["theory_key"], dtype=np.uint32))

    I = _build_integrator(str(args.integrator).lower(), theory, int(args.nmd), float(args.tau))
    if update_name == "hmc":
        chain = hmc(T=theory, I=I, verbose=False, seed=int(args.seed), use_fast_jit=True)
    else:
        chain = SMD(
            T=theory,
            I=I,
            gamma=float(args.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(args.seed),
            use_fast_jit=True,
        )

    warmup_noar_done = 0
    warmup_done = 0
    meas_done = 0
    obs_E = []
    obs_av_sigma = []
    obs_chi_raw = []
    obs_C2p = []
    obs_Q = []

    if ckpt is not None:
        update_key = ckpt.get("update_key", ckpt.get("hmc_key"))
        chain.key = jax.device_put(np.asarray(update_key, dtype=np.uint32))
        if hasattr(chain, "p"):
            p_arr = ckpt.get("update_momentum", None)
            chain.p = None if p_arr is None else jax.device_put(np.asarray(p_arr))
        chain.AcceptReject = list(np.asarray(ckpt.get("accept_reject", []), dtype=np.float64))
        st = dict(ckpt.get("state", {}))
        warmup_noar_done = int(st.get("warmup_noar_done", 0))
        warmup_done = int(st.get("warmup_done", 0))
        meas_done = int(st.get("meas_done", 0))
        obs_E = [np.asarray(x, dtype=np.float64) for x in st.get("obs_E", [])]
        obs_av_sigma = [np.asarray(x, dtype=np.float64) for x in st.get("obs_av_sigma", [])]
        obs_chi_raw = [np.asarray(x, dtype=np.float64) for x in st.get("obs_chi_raw", [])]
        obs_C2p = [np.asarray(x, dtype=np.float64) for x in st.get("obs_C2p", [])]
        obs_Q = [np.asarray(x, dtype=np.float64) for x in st.get("obs_Q", [])]
        print(f"Resumed from {args.resume} at noAR/warm/meas={warmup_noar_done}/{warmup_done}/{meas_done}")
    else:
        q = theory.hotStart() if str(args.start).lower() == "hot" else theory.coldStart()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "<unset>"))
    print(
        f"shape={lattice_shape} ncomp={theory.ncomp} beta={theory.beta}"
        f" batch={theory.Bs} layout={theory.layout} exp_method={theory.exp_method}"
    )
    print(
        f"update={update_name} integrator={args.integrator} tau={args.tau} nmd={args.nmd}"
        f" warmup_no_ar={args.warmup_no_ar} nwarm={args.nwarm} nmeas={args.nmeas} nskip={args.nskip}"
    )

    def maybe_save_checkpoint(reason: str):
        if not args.save:
            return
        state = {
            "warmup_noar_done": int(warmup_noar_done),
            "warmup_done": int(warmup_done),
            "meas_done": int(meas_done),
            "obs_E": [np.asarray(x) for x in obs_E],
            "obs_av_sigma": [np.asarray(x) for x in obs_av_sigma],
            "obs_chi_raw": [np.asarray(x) for x in obs_chi_raw],
            "obs_C2p": [np.asarray(x) for x in obs_C2p],
            "obs_Q": [np.asarray(x) for x in obs_Q],
            "reason": str(reason),
        }
        config = vars(args).copy()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(args.save, q, theory, chain, state, config)

    for k in range(int(warmup_noar_done), int(args.warmup_no_ar)):
        q = _run_one_trajectory_hmc_no_ar(q, chain)
        warmup_noar_done = k + 1
        if warmup_noar_done % max(1, int(args.checkpoint_every) or int(args.warmup_no_ar) or 1) == 0:
            print(f"warmup-noar k={warmup_noar_done}/{args.warmup_no_ar}")
        if int(args.checkpoint_every) > 0 and warmup_noar_done % int(args.checkpoint_every) == 0:
            maybe_save_checkpoint("warmup_no_ar")

    tic = time.perf_counter()
    for k in range(int(warmup_done), int(args.nwarm)):
        q = _run_one_trajectory(q, chain)
        warmup_done = k + 1
    if int(args.nwarm) > 0:
        print(f"warmup time/trajectory: {(time.perf_counter() - tic) * 1e3 / max(1, int(args.nwarm) - int(warmup_done == 0)):.3f} ms")
    print(f"warmup acceptance: {chain.calc_Acceptance():.6f}")
    chain.reset_Acceptance()

    tic = time.perf_counter()
    for k in range(int(meas_done), int(args.nmeas)):
        obs = _compute_observables(theory, q, mom_axis=int(args.mom_axis))
        obs_E.append(obs["E"])
        obs_av_sigma.append(obs["av_sigma"])
        obs_chi_raw.append(obs["chi_m_raw"])
        obs_C2p.append(obs["C2p"])
        if "Q" in obs:
            obs_Q.append(obs["Q"])

        if k % max(1, int(args.nmeas) // 10) == 0:
            chi_m_now = float(np.mean(obs["chi_m_raw"]))
            c2p_now = float(np.mean(obs["C2p"]))
            msg = f"meas k={k:5d} chi_m={chi_m_now:.4f} C2p={c2p_now:.4f}"
            if "Q" in obs:
                msg += f" Q={float(np.mean(obs['Q'])):.4f}"
            print(msg)

        for _ in range(int(args.nskip)):
            q = _run_one_trajectory(q, chain)
        meas_done = k + 1
        if int(args.checkpoint_every) > 0 and meas_done % int(args.checkpoint_every) == 0:
            maybe_save_checkpoint("measurement")
    print(
        "measurement time/trajectory:"
        f" {(time.perf_counter() - tic) * 1e3 / max(1, int(args.nmeas) * int(args.nskip)):.3f} ms"
    )

    obs_E_arr = np.asarray(obs_E, dtype=np.float64)
    obs_av_sigma_arr = np.asarray(obs_av_sigma, dtype=np.float64)
    obs_chi_raw_arr = np.asarray(obs_chi_raw, dtype=np.float64)
    obs_C2p_arr = np.asarray(obs_C2p, dtype=np.float64)
    av_sigma_mean = np.mean(obs_av_sigma_arr, axis=(0, 1))
    obs_chi_conn = obs_chi_raw_arr - float(theory.Vol) * float(np.dot(av_sigma_mean, av_sigma_mean))

    res_E = _analyze(obs_E_arr)
    res_chi = _analyze(obs_chi_conn)
    res_c2p = _analyze(obs_C2p_arr)
    xi = _correlation_length(int(lattice_shape[0]), float(res_chi["mean"]), float(res_c2p["mean"]))

    print("\n--- Results ---")
    print(f"E/V   : {res_E['mean']:+.6f} +/- {res_E['sigma']:.6f}  tau_int={res_E['tau_int']:.2f}  ESS={res_E['ess']:.1f}")
    print(f"Chi_m : {res_chi['mean']:+.6f} +/- {res_chi['sigma']:.6f}  tau_int={res_chi['tau_int']:.2f}  ESS={res_chi['ess']:.1f}")
    print(f"C2p   : {res_c2p['mean']:+.6f} +/- {res_c2p['sigma']:.6f}  tau_int={res_c2p['tau_int']:.2f}  ESS={res_c2p['ess']:.1f}")
    print(f"xi    : {xi:.6f}")
    print(f"acc   : {chain.calc_Acceptance():.6f}")
    print("m_sigma:", av_sigma_mean.tolist())

    out = {
        "shape": list(lattice_shape),
        "ncomp": int(theory.ncomp),
        "beta": float(theory.beta),
        "layout": str(theory.layout),
        "exp_method": str(theory.exp_method),
        "update": str(update_name),
        "integrator": str(args.integrator),
        "nmd": int(args.nmd),
        "tau": float(args.tau),
        "batch_size": int(args.batch_size),
        "warmup_no_ar": int(args.warmup_no_ar),
        "nwarm": int(args.nwarm),
        "nmeas": int(args.nmeas),
        "nskip": int(args.nskip),
        "acceptance": float(chain.calc_Acceptance()),
        "m_sigma": av_sigma_mean.tolist(),
        "E": res_E,
        "chi_m": res_chi,
        "C2p": res_c2p,
        "xi": float(xi),
    }

    if obs_Q:
        obs_Q_arr = np.asarray(obs_Q, dtype=np.float64)
        res_Q = _analyze(obs_Q_arr)
        q_all = obs_Q_arr.reshape(-1)
        chi_top = float(np.var(q_all, ddof=1) / float(theory.Vol)) if q_all.size > 1 else float("nan")
        print(f"Q     : {res_Q['mean']:+.6f} +/- {res_Q['sigma']:.6f}  tau_int={res_Q['tau_int']:.2f}  ESS={res_Q['ess']:.1f}")
        print(f"chi_t : {chi_top:.6e}")
        out["Q"] = res_Q
        out["chi_top"] = chi_top

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json_out}")

    maybe_save_checkpoint("final")


if __name__ == "__main__":
    main()

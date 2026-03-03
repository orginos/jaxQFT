"""Shared fast MCMC smoke checks for Wilson Nf=2 models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

from jaxqft.core.integrators import force_gradient
from jaxqft.core.update import HMC, SMD
from jaxqft.stats import integrated_autocorr_time


Array = jax.Array
ModelFactory = Callable[[int, str], Any]


@dataclass
class SmokeConfig:
    """Configuration for short MCMC cross-checks."""

    nmd: int = 8
    tau: float = 1.0
    warmup_no_ar: int = 2
    warmup_ar: int = 8
    nmeas: int = 24
    hot_scale: float = 0.2
    iat_method: str = "gamma"
    sigma_cut: float = 4.0
    seed: int = 1234
    smd_gamma: float = 0.3
    use_fast_jit: bool = False
    reproducibility_steps: int = 3
    reproducibility_tol: float = 1e-7


def _iat_error(x: np.ndarray, method: str) -> Dict[str, float]:
    iat = integrated_autocorr_time(x, method=str(method))
    tau = float(iat.get("tau_int", float("nan")))
    ess = float(iat.get("ess", float("nan")))
    win = int(iat.get("window", 0))
    ok = bool(iat.get("ok", False))
    msg = str(iat.get("message", ""))

    sigma_mean = float(iat.get("sigma_mean", float("nan")))
    if np.isfinite(sigma_mean) and sigma_mean > 0.0:
        err = sigma_mean
    else:
        s = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        if np.isfinite(ess) and ess > 1.0 and np.isfinite(s):
            err = s / math.sqrt(ess)
        elif np.isfinite(s) and x.size > 1:
            err = s / math.sqrt(max(1, x.size - 1))
        else:
            err = float("nan")

    return {
        "err_iat": float(err),
        "tau_int": tau,
        "ess": ess,
        "iat_window": float(win),
        "iat_ok": float(1.0 if ok else 0.0),
        "iat_message": msg,
    }


def _no_ar_step_hmc(q: Array, chain: HMC) -> Array:
    if hasattr(chain, "_prepare_trajectory"):
        chain._prepare_trajectory(q)
    p0 = chain.T.refreshP()
    _, q1 = chain.I.integrate(p0, q)
    return jax.block_until_ready(q1)


def _step(q: Array, chain: Any, update: str, *, warmup: bool = False) -> Array:
    if update == "smd":
        return chain.evolve(q, 1, warmup=bool(warmup))
    return chain.evolve(q, 1)


def _run_chain(model_factory: ModelFactory, *, update: str, pf_refresh: str, seed: int, cfg: SmokeConfig) -> Dict[str, Any]:
    theory = model_factory(int(seed), str(pf_refresh))
    integ = force_gradient(theory.force, theory.evolveQ, int(cfg.nmd), float(cfg.tau))
    if update == "hmc":
        chain = HMC(T=theory, I=integ, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
    elif update == "smd":
        chain = SMD(
            T=theory,
            I=integ,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )
    else:
        raise ValueError(f"Unknown update type: {update}")

    q = theory.hot_start(scale=float(cfg.hot_scale))

    for _ in range(max(0, int(cfg.warmup_no_ar))):
        if update == "hmc":
            q = _no_ar_step_hmc(q, chain)
        else:
            q = _step(q, chain, update, warmup=True)

    chain.reset_acceptance()
    for _ in range(max(0, int(cfg.warmup_ar))):
        q = _step(q, chain, update, warmup=False)
    warm_acc = float(chain.calc_acceptance())

    chain.reset_acceptance()
    plaq = np.empty((max(1, int(cfg.nmeas)),), dtype=np.float64)
    for k in range(plaq.size):
        q = _step(q, chain, update, warmup=False)
        plaq[k] = float(jnp.mean(theory.average_plaquette(q)))

    out = {
        "update": str(update),
        "pf_refresh": str(pf_refresh),
        "seed": int(seed),
        "mean_plaquette": float(np.mean(plaq)),
        "warmup_acceptance": warm_acc,
        "meas_acceptance": float(chain.calc_acceptance()),
    }
    out.update(_iat_error(plaq, method=str(cfg.iat_method)))
    out["series"] = plaq.tolist()
    return out


def _compare_runs(a: Dict[str, Any], b: Dict[str, Any], sigma_cut: float) -> Dict[str, Any]:
    d = abs(float(a["mean_plaquette"]) - float(b["mean_plaquette"]))
    ea = float(a.get("err_iat", float("nan")))
    eb = float(b.get("err_iat", float("nan")))
    comb = math.sqrt(max(0.0, ea * ea + eb * eb)) if np.isfinite(ea) and np.isfinite(eb) else float("nan")
    sigma = d / comb if np.isfinite(comb) and comb > 0.0 else float("inf")
    return {
        "delta": float(d),
        "sigma": float(sigma),
        "agree": bool(np.isfinite(sigma) and sigma < float(sigma_cut)),
    }


def _reproducibility_check(
    model_factory: ModelFactory,
    *,
    update: str,
    pf_refresh: str,
    seed: int,
    cfg: SmokeConfig,
) -> Dict[str, Any]:
    theory1 = model_factory(int(seed), str(pf_refresh))
    theory2 = model_factory(int(seed), str(pf_refresh))

    integ1 = force_gradient(theory1.force, theory1.evolveQ, int(cfg.nmd), float(cfg.tau))
    integ2 = force_gradient(theory2.force, theory2.evolveQ, int(cfg.nmd), float(cfg.tau))

    if update == "hmc":
        chain1 = HMC(T=theory1, I=integ1, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
        chain2 = HMC(T=theory2, I=integ2, verbose=False, seed=int(seed), use_fast_jit=bool(cfg.use_fast_jit))
    else:
        chain1 = SMD(
            T=theory1,
            I=integ1,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )
        chain2 = SMD(
            T=theory2,
            I=integ2,
            gamma=float(cfg.smd_gamma),
            accept_reject=True,
            verbose=False,
            seed=int(seed),
            use_fast_jit=bool(cfg.use_fast_jit),
        )

    # Build starts independently so both theory RNG/key states advance identically.
    q1 = theory1.hot_start(scale=float(cfg.hot_scale))
    q2 = theory2.hot_start(scale=float(cfg.hot_scale))

    max_abs = 0.0
    for _ in range(max(1, int(cfg.reproducibility_steps))):
        q1 = _step(q1, chain1, update, warmup=False)
        q2 = _step(q2, chain2, update, warmup=False)
        d = float(jnp.max(jnp.abs(q1 - q2)))
        max_abs = max(max_abs, d)

    return {
        "max_abs_diff": float(max_abs),
        "tol": float(cfg.reproducibility_tol),
        "ok": bool(max_abs <= float(cfg.reproducibility_tol)),
    }


def run_nf2_mcmc_smoke_suite(model_name: str, model_factory: ModelFactory, config: SmokeConfig | None = None) -> Dict[str, Any]:
    """Run fast HMC/SMD smoke checks for one Nf=2 model."""

    cfg = config if config is not None else SmokeConfig()
    runs = {
        "hmc_ou": _run_chain(model_factory, update="hmc", pf_refresh="ou", seed=int(cfg.seed), cfg=cfg),
        "smd_ou": _run_chain(model_factory, update="smd", pf_refresh="ou", seed=int(cfg.seed) + 1, cfg=cfg),
        "hmc_heatbath": _run_chain(
            model_factory,
            update="hmc",
            pf_refresh="heatbath",
            seed=int(cfg.seed) + 2,
            cfg=cfg,
        ),
    }

    comparisons = {
        "hmc_ou_vs_smd_ou": _compare_runs(runs["hmc_ou"], runs["smd_ou"], sigma_cut=float(cfg.sigma_cut)),
        "hmc_ou_vs_hmc_heatbath": _compare_runs(
            runs["hmc_ou"], runs["hmc_heatbath"], sigma_cut=float(cfg.sigma_cut)
        ),
    }

    reproducibility = {
        "hmc_ou": _reproducibility_check(
            model_factory,
            update="hmc",
            pf_refresh="ou",
            seed=int(cfg.seed) + 100,
            cfg=cfg,
        ),
        "smd_ou": _reproducibility_check(
            model_factory,
            update="smd",
            pf_refresh="ou",
            seed=int(cfg.seed) + 101,
            cfg=cfg,
        ),
    }

    ok = True
    ok = ok and all(bool(c["agree"]) for c in comparisons.values())
    ok = ok and all(bool(r["ok"]) for r in reproducibility.values())

    return {
        "model": str(model_name),
        "config": asdict(cfg),
        "runs": runs,
        "comparisons": comparisons,
        "reproducibility": reproducibility,
        "pass": bool(ok),
    }

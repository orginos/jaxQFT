#!/usr/bin/env python3
"""SAD (Stochastic Automatic Differentiation) for phi^4 theory in 2D.

Computes the connected two-point function C(t) via the SAD estimator
(Catumba & Ramos, arXiv:2502.15570) and compares it to the standard
noisy estimator, demonstrating the signal-to-noise improvement.

Base algorithm: SMD without accept/reject (pure OU + leapfrog dynamics).

Usage:
  # Free field test (lam=0, analytic reference available):
  python scripts/phi4/sad_phi4.py --shape 16,16 --m2 1.0 --lam 0.0

  # Interacting phi^4:
  python scripts/phi4/sad_phi4.py --shape 16,16 --m2 -0.40 --lam 2.4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
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
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
from jaxqft.models.phi4_sad import smd_sad_step, free_propagator


# ---------------------------------------------------------------------------
# Jackknife helpers
# ---------------------------------------------------------------------------

def jackknife_mean_err(chains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and jackknife error from independent chains.

    Args:
        chains: shape (B, ...) where B = number of independent chains.

    Returns:
        mean: shape (...), grand mean.
        err:  shape (...), jackknife standard error of the mean.
    """
    B = chains.shape[0]
    grand = chains.mean(axis=0)
    loo = (grand * B - chains) / (B - 1)
    err = np.sqrt((B - 1) / B * np.sum((loo - grand) ** 2, axis=0))
    return grand, err


# ---------------------------------------------------------------------------
# Cosh effective mass for periodic correlators
# ---------------------------------------------------------------------------

def cosh_meff_solve(R: float, t: int, T: int,
                    m_max: float = 10.0, tol: float = 1e-12) -> float:
    """Solve  cosh(m*(t - T/2)) / cosh(m*(t+1 - T/2)) = R  for  m >= 0.

    For a periodic correlator  C(t) ~ A cosh(m (t - T/2)),
    this gives the effective mass at timeslice t.

    Uses bisection in [0, m_max] on the log-space equation
    log(cosh(m*u)/cosh(m*v)) = log(R) to avoid overflow for large T.
    Returns NaN if no solution.
    """
    if not np.isfinite(R) or R <= 1.0:
        return np.nan

    u = t - T / 2.0
    v = u + 1.0          # = t + 1 - T/2

    # Special case t = T/2 - 1  (v = 0)
    if abs(v) < 1e-10:
        return float(np.arccosh(R))

    logR = np.log(R)

    def _log_cosh_ratio(m):
        """log(cosh(m*u) / cosh(m*v)), numerically stable for large T.

        Uses log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2).
        The difference cancels the log(2) terms, and exp(-2|x|) underflows
        gracefully to 0 for large |x|.
        """
        au = abs(m * u)
        av = abs(m * v)
        return (au - av) + np.log1p(np.exp(-2 * au)) - np.log1p(np.exp(-2 * av))

    def f(m):
        return _log_cosh_ratio(m) - logR

    # f(0) = 0 - logR < 0;  f(m_max) should be > 0 for |u| > |v|
    a, b = 0.0, m_max
    fa = -logR
    fb = f(b)
    if not np.isfinite(fb) or fa * fb > 0:
        return np.nan

    for _ in range(200):
        c = 0.5 * (a + b)
        if b - a < tol:
            return c
        fc = f(c)
        if not np.isfinite(fc):
            b = c
            continue
        if abs(fc) < tol:
            return c
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def cosh_meff_jackknife(C_chains: np.ndarray, T: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Cosh effective mass with jackknife errors from per-chain correlators.

    Args:
        C_chains: (B, T) per-chain mean correlators.
        T:        temporal extent.

    Returns:
        meff:     shape (T//2,) for t = 0 .. T/2 - 1.
        meff_err: jackknife standard error, same shape.
    """
    B = C_chains.shape[0]
    Thalf = T // 2
    grand = C_chains.mean(axis=0)

    meff = np.array([cosh_meff_solve(grand[t] / grand[t + 1], t, T)
                      for t in range(Thalf)])

    meff_loo = np.zeros((B, Thalf))
    for j in range(B):
        C_loo = (grand * B - C_chains[j]) / (B - 1)
        for t in range(Thalf):
            R = C_loo[t] / C_loo[t + 1] if C_loo[t + 1] != 0 else np.nan
            meff_loo[j, t] = cosh_meff_solve(R, t, T)

    meff_err = np.sqrt(
        (B - 1) / B * np.nansum((meff_loo - meff[np.newaxis, :]) ** 2, axis=0))
    return meff, meff_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SAD correlator for phi^4 in 2D")
    ap.add_argument("--shape",      type=str,   default="16,16",
                    help="T,L  (time x space), e.g. 16,16")
    ap.add_argument("--m2",         type=float, default=1.0,
                    help="Bare mass squared m2 (use m2>0 for free-field test)")
    ap.add_argument("--lam",        type=float, default=0.0,
                    help="Quartic coupling (0 for free field)")
    ap.add_argument("--nwarm",      type=int,   default=500,
                    help="Warmup SMD steps per chain")
    ap.add_argument("--nmeas",      type=int,   default=2000,
                    help="Measurement steps per chain")
    ap.add_argument("--batch-size", type=int,   default=8,
                    help="Number of independent chains")
    ap.add_argument("--nmd",        type=int,   default=10,
                    help="Leapfrog steps per trajectory")
    ap.add_argument("--tau",        type=float, default=1.0,
                    help="MD trajectory length")
    ap.add_argument("--gamma",      type=float, default=0.3,
                    help="SMD friction coefficient")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--json-out",   type=str,   default=None,
                    help="Save results dict to JSON file")
    args = ap.parse_args()

    T, L = [int(x) for x in args.shape.split(",")]
    B    = args.batch_size
    m2   = args.m2
    lam  = args.lam
    tau  = args.tau
    nmd  = args.nmd
    dt   = tau / nmd

    c1 = math.exp(-args.gamma * tau)
    c2 = math.sqrt(max(0.0, 1.0 - c1 * c1))

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print(f"Lattice: T={T} L={L}  m2={m2}  lam={lam}  batch={B}")
    print(f"SMD: gamma={args.gamma}  tau={tau}  nmd={nmd}  dt={dt:.4f}")
    print(f"     c1={c1:.4f}  c2={c2:.4f}")
    print(f"Nwarm={args.nwarm}  Nmeas={args.nmeas}")

    # ------------------------------------------------------------------
    # Build JIT-compiled batched step
    # ------------------------------------------------------------------
    def single_step(phi_i, pi_i, phi_tan_i, pi_tan_i, key_i):
        return smd_sad_step(phi_i, pi_i, phi_tan_i, pi_tan_i, key_i,
                            m2, lam, dt, nmd, c1, c2)

    batched_step = jax.jit(jax.vmap(single_step))

    # ------------------------------------------------------------------
    # Initialise fields and tangent state
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, sk1, sk2 = jax.random.split(key, 3)
    phi_batch = jax.random.normal(sk1, shape=(B, T, L), dtype=jnp.float32) * 0.1
    pi_batch  = jax.random.normal(sk2, shape=(B, T, L), dtype=jnp.float32)

    phi_tan_batch = jnp.zeros((B, T, L), dtype=jnp.float32)
    pi_tan_batch  = jnp.zeros((B, T, L), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    print("\nWarmup (first call triggers JIT compilation) ...")
    tic = time.perf_counter()
    for k in range(args.nwarm):
        key, *sks = jax.random.split(key, B + 1)
        keys_b = jnp.array(sks)
        phi_batch, pi_batch, phi_tan_batch, pi_tan_batch = batched_step(
            phi_batch, pi_batch, phi_tan_batch, pi_tan_batch, keys_b)
    phi_batch.block_until_ready()
    toc = time.perf_counter()
    print(f"Warmup done: {(toc-tic)*1e3:.1f} ms total  "
          f"({(toc-tic)*1e6/args.nwarm:.1f} \u03bcs/traj per chain, amortised)")

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    C_sad_sum = np.zeros((B, T), dtype=np.float64)
    C_std_sum = np.zeros((B, T), dtype=np.float64)
    phi_bar_sum = np.zeros((B, T), dtype=np.float64)
    print_every = max(1, args.nmeas // 5)

    tic = time.perf_counter()
    for k in range(args.nmeas):
        key, *sks = jax.random.split(key, B + 1)
        keys_b = jnp.array(sks)
        phi_batch, pi_batch, phi_tan_batch, pi_tan_batch = batched_step(
            phi_batch, pi_batch, phi_tan_batch, pi_tan_batch, keys_b)

        C_sad_k = np.asarray(phi_tan_batch.mean(axis=-1))
        phi_bar = np.asarray(phi_batch.mean(axis=-1))
        C_std_k = L * phi_bar * phi_bar[:, 0:1]

        C_sad_sum   += C_sad_k
        C_std_sum   += C_std_k
        phi_bar_sum += phi_bar

        if k % print_every == 0:
            sad_t1 = float(C_sad_sum[:, 1].mean() / (k + 1))
            std_t1 = float(C_std_sum[:, 1].mean() / (k + 1))
            print(f"  k={k:5d}  C_sad(t=1)={sad_t1:+.4f}  C_std(t=1)={std_t1:+.4f}")

    phi_batch.block_until_ready()
    toc = time.perf_counter()
    print(f"Measurement done: {(toc-tic)*1e6/args.nmeas:.1f} \u03bcs/step")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    N = args.nmeas
    C_sad_chains = C_sad_sum / N
    C_std_chains = C_std_sum / N

    # Connected part of std estimator
    phi_bar_mean = phi_bar_sum / N
    C_std_chains -= L * phi_bar_mean * phi_bar_mean[:, 0:1]

    C_sad_mean, C_sad_err = jackknife_mean_err(C_sad_chains)
    C_std_mean, C_std_err = jackknife_mean_err(C_std_chains)

    # Cosh effective mass  (t = 0 .. T/2 - 1)
    mcosh_sad,  mcosh_sad_err  = cosh_meff_jackknife(C_sad_chains, T)
    mcosh_std,  mcosh_std_err  = cosh_meff_jackknife(C_std_chains, T)

    # Analytic free-field reference
    C_exact = None
    mcosh_exact = None
    if lam == 0.0 and m2 > 0.0:
        C_exact = free_propagator(T, L, m2)
        Thalf = T // 2
        mcosh_exact = np.array([
            cosh_meff_solve(C_exact[t] / C_exact[t + 1], t, T)
            for t in range(Thalf)])

    # ------------------------------------------------------------------
    # Print: Correlator table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"Correlator C(t)  [T={T}, L={L}, m2={m2}, lam={lam}, "
          f"B={B}, N={N}]")
    print(f"{'='*72}")
    hdr = f"{'t':>4}  {'C_sad':>11} {'+-':>2} {'err':>9}  "
    hdr += f"{'C_std':>11} {'+-':>2} {'err':>9}"
    if C_exact is not None:
        hdr += f"  {'C_exact':>11}  {'sad/C_ex':>9}"
    print(hdr)
    print("-" * len(hdr))
    for t in range(T):
        row = (f"{t:>4}  {C_sad_mean[t]:>+11.5f}  +- {C_sad_err[t]:>9.5f}  "
               f"{C_std_mean[t]:>+11.5f}  +- {C_std_err[t]:>9.5f}")
        if C_exact is not None:
            if C_sad_err[t] > 0:
                pull = (C_sad_mean[t] - C_exact[t]) / C_sad_err[t]
                row += f"  {C_exact[t]:>+11.5f}  {pull:>+9.2f}\u03c3"
            elif C_exact[t] != 0:
                rel = (C_sad_mean[t] - C_exact[t]) / C_exact[t] * 100
                row += f"  {C_exact[t]:>+11.5f}  {rel:>+8.4f}%"
            else:
                row += f"  {C_exact[t]:>+11.5f}       ---"
        print(row)

    # ------------------------------------------------------------------
    # Print: Cosh effective mass (first half only)
    # ------------------------------------------------------------------
    Thalf = T // 2
    print(f"\n{'='*72}")
    print(f"Cosh effective mass  m_eff(t):  cosh(m(t-T/2))/cosh(m(t+1-T/2)) = C(t)/C(t+1)")
    print(f"{'='*72}")
    hdr2 = (f"{'t':>4}  {'mcosh_sad':>10} +- {'err':>8}   "
            f"{'mcosh_std':>10} +- {'err':>8}   {'ratio':>6}")
    if mcosh_exact is not None:
        hdr2 += f"   {'exact':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for t in range(Thalf):
        r = mcosh_std_err[t] / mcosh_sad_err[t] if mcosh_sad_err[t] > 0 else float('inf')
        row = (f"{t:>4}  {mcosh_sad[t]:>10.5f} +- {mcosh_sad_err[t]:>8.5f}   "
               f"{mcosh_std[t]:>10.5f} +- {mcosh_std_err[t]:>8.5f}   {r:>6.2f}")
        if mcosh_exact is not None:
            row += f"   {mcosh_exact[t]:>10.5f}"
        print(row)

    # ------------------------------------------------------------------
    # Print: Signal-to-noise improvement
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Signal-to-noise improvement:  \u03c3_std(t) / \u03c3_SAD(t)")
    print(f"{'='*72}")
    for t in range(T):
        if C_sad_err[t] > 0:
            ratio = C_std_err[t] / C_sad_err[t]
            bar = "#" * min(60, max(1, int(ratio)))
            print(f"  t={t:2d}  ratio={ratio:8.2f}  {bar}")
        else:
            print(f"  t={t:2d}  ratio=     inf  (zero SAD variance)")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    if args.json_out:
        out = dict(
            shape=[T, L], m2=m2, lam=lam,
            nwarm=args.nwarm, nmeas=N, batch_size=B,
            nmd=nmd, tau=tau, gamma=args.gamma, dt=dt,
            C_sad_mean=C_sad_mean.tolist(),
            C_sad_err=C_sad_err.tolist(),
            C_std_mean=C_std_mean.tolist(),
            C_std_err=C_std_err.tolist(),
            C_sad_chains=C_sad_chains.tolist(),
            C_std_chains=C_std_chains.tolist(),
            mcosh_sad=mcosh_sad.tolist(),
            mcosh_sad_err=mcosh_sad_err.tolist(),
            mcosh_std=mcosh_std.tolist(),
            mcosh_std_err=mcosh_std_err.tolist(),
        )
        if C_exact is not None:
            out["C_exact"] = C_exact.tolist()
        if mcosh_exact is not None:
            out["mcosh_exact"] = mcosh_exact.tolist()
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json_out}")


if __name__ == "__main__":
    main()

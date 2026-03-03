#!/usr/bin/env python3
"""Diagnose HMC vs SMD discrepancy for U(1) Nf=2 Wilson fermions.

Tests:
1. SMD with gamma=1000 (effectively HMC) - isolates OU effect
2. dH statistics for both algorithms
3. Single-trajectory reversibility check
"""
from __future__ import annotations
import os, platform, sys
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxqft.core.integrators import minnorm2
from jaxqft.core.update import HMC, SMD
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2


def avg_err(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(max(1, x.size)))


def run_and_measure(theory, chain, q, n_warmup, n_meas, skip, label):
    """Run chain and collect plaquette measurements."""
    from jaxqft.models.u1_ym import U1YangMills
    for _ in range(n_warmup):
        q = chain.evolve(q, 1)
    chain.reset_acceptance()

    plaquettes = []
    for k in range(n_meas):
        q = chain.evolve(q, skip)
        plaq = float(jnp.mean(theory.average_plaquette(q)))
        plaquettes.append(plaq)

    mp, ep = avg_err(plaquettes)
    acc = chain.calc_acceptance()
    print(f"  {label:25s}: plaq={mp:.6f}+/-{ep:.6f}  acc={acc:.3f}")
    return plaquettes, acc


def make_theory(**kw):
    defaults = dict(
        lattice_shape=(8, 8), beta=2.0, mass=0.1, batch_size=4,
        seed=42, pseudofermion_refresh="heatbath",
        pseudofermion_force_mode="autodiff",
        solver_kind="cg", solver_form="normal",
        cg_tol=1e-10, cg_maxiter=1000,
    )
    defaults.update(kw)
    return U1WilsonNf2(**defaults)


def make_chain(theory, kind, nmd=10, tau=1.0, gamma=0.3, accept_reject=True, seed=42):
    I = minnorm2(theory.force, theory.evolveQ, nmd, tau)
    if kind == "hmc":
        return HMC(T=theory, I=I, verbose=False, seed=seed, use_fast_jit=False)
    else:
        return SMD(T=theory, I=I, gamma=gamma, accept_reject=accept_reject,
                   verbose=False, seed=seed, use_fast_jit=False)


def main():
    n_warmup = 100
    n_meas = 200
    skip = 5

    print("="*60)
    print("TEST 1: SMD with gamma=1000 (should match HMC)")
    print("="*60)

    th1 = make_theory(seed=100)
    ch1 = make_chain(th1, "hmc", seed=100)
    q1 = th1.hot_start(scale=0.2)
    p1, _ = run_and_measure(th1, ch1, q1, n_warmup, n_meas, skip, "HMC")

    th2 = make_theory(seed=200)
    ch2 = make_chain(th2, "smd", gamma=1000.0, seed=200)
    q2 = th2.hot_start(scale=0.2)
    p2, _ = run_and_measure(th2, ch2, q2, n_warmup, n_meas, skip, "SMD(gamma=1000)")

    th3 = make_theory(seed=300)
    ch3 = make_chain(th3, "smd", gamma=0.3, seed=300)
    q3 = th3.hot_start(scale=0.2)
    p3, _ = run_and_measure(th3, ch3, q3, n_warmup, n_meas, skip, "SMD(gamma=0.3)")

    th4 = make_theory(seed=400)
    ch4 = make_chain(th4, "smd", gamma=0.3, accept_reject=False, seed=400)
    q4 = th4.hot_start(scale=0.2)
    p4, _ = run_and_measure(th4, ch4, q4, n_warmup, n_meas, skip, "SMD(gamma=0.3,noAR)")

    def sigma(a, b):
        m1, e1 = avg_err(a)
        m2, e2 = avg_err(b)
        return abs(m1-m2) / max(np.sqrt(e1**2+e2**2), 1e-15)

    print(f"\n  HMC vs SMD(g=1000): {sigma(p1,p2):.1f} sigma")
    print(f"  HMC vs SMD(g=0.3):  {sigma(p1,p3):.1f} sigma")
    print(f"  HMC vs SMD(noAR):   {sigma(p1,p4):.1f} sigma")

    print("\n" + "="*60)
    print("TEST 2: dH statistics per trajectory")
    print("="*60)

    for label, kind, gamma in [("HMC", "hmc", 0.3), ("SMD(0.3)", "smd", 0.3), ("SMD(1000)", "smd", 1000.0)]:
        th = make_theory(seed=500)
        I = minnorm2(th.force, th.evolveQ, 10, 1.0)
        q = th.hot_start(scale=0.2)
        # Warmup
        if kind == "hmc":
            chain = HMC(T=th, I=I, verbose=False, seed=500, use_fast_jit=False)
        else:
            chain = SMD(T=th, I=I, gamma=gamma, accept_reject=True,
                       verbose=False, seed=500, use_fast_jit=False)
        for _ in range(50):
            q = chain.evolve(q, 1)

        # Measure dH by hand
        dHs = []
        for _ in range(100):
            th.prepare_trajectory(q, traj_length=1.0)
            if kind == "hmc":
                p0 = th.refreshP()
            else:
                if chain.p is None:
                    chain._init_momentum(q)
                eta = th.refresh_p_with_key(jax.random.PRNGKey(np.random.randint(0, 2**31)))
                tau = 1.0
                c1 = np.exp(-gamma * tau)
                c2 = np.sqrt(max(0, 1 - c1*c1))
                p0 = c1 * chain.p + c2 * eta

            H0 = float(jnp.mean(th.kinetic(p0) + th.action(q)))
            p_prop, q_prop = I.integrate(p0, q)
            Hf = float(jnp.mean(th.kinetic(p_prop) + th.action(q_prop)))
            dH = Hf - H0
            dHs.append(dH)

            # Accept for next iteration
            if np.random.random() < min(1, np.exp(-max(dH, 0))):
                q = q_prop
                if kind == "smd":
                    chain.p = p_prop
            else:
                if kind == "smd":
                    chain.p = -p0

        dHs = np.array(dHs)
        print(f"  {label:15s}: <dH>={dHs.mean():.6f}  std(dH)={dHs.std():.6f}  <|dH|>={np.abs(dHs).mean():.6f}  <exp(-dH)>={np.exp(-dHs).mean():.6f}")

    print("\n" + "="*60)
    print("TEST 3: Check pseudofermion action consistency")
    print("="*60)
    # Verify that S_pf(q0, phi) + S_pf(q_prop, phi) are computed with same phi
    th = make_theory(seed=600)
    I = minnorm2(th.force, th.evolveQ, 10, 1.0)
    q = th.hot_start(scale=0.2)
    for _ in range(20):
        th.prepare_trajectory(q)
        p0 = th.refreshP()
        q0 = q
        # Record phi
        phi_before = th.fermion_monomial.phi.copy() if th.fermion_monomial.phi is not None else None
        S0 = float(jnp.mean(th.action(q0)))
        phi_after_S0 = th.fermion_monomial.phi.copy() if th.fermion_monomial.phi is not None else None

        p_prop, q_prop = I.integrate(p0, q0)
        phi_after_MD = th.fermion_monomial.phi.copy() if th.fermion_monomial.phi is not None else None

        Sf = float(jnp.mean(th.action(q_prop)))
        phi_after_Sf = th.fermion_monomial.phi.copy() if th.fermion_monomial.phi is not None else None

        # Check phi hasn't changed
        if phi_before is not None:
            d1 = float(jnp.max(jnp.abs(phi_before - phi_after_S0)))
            d2 = float(jnp.max(jnp.abs(phi_before - phi_after_MD)))
            d3 = float(jnp.max(jnp.abs(phi_before - phi_after_Sf)))
            print(f"  phi drift: after_S0={d1:.2e}  after_MD={d2:.2e}  after_Sf={d3:.2e}")

        q = q_prop  # always accept for simplicity


if __name__ == "__main__":
    main()

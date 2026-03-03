#!/usr/bin/env python3
"""Compare HMC vs SMD for U(1) + Nf=2 Wilson fermions on a small 2D lattice.

Runs both algorithms from the same starting configuration and checks
whether equilibrium observables (plaquette, action) agree statistically.
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

from jaxqft.core.integrators import leapfrog, minnorm2
from jaxqft.core.update import HMC, SMD
from jaxqft.models.u1_wilson_nf2 import U1WilsonNf2

def avg_err(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def run_chain(theory, chain, q, n_warmup, n_meas, skip, label=""):
    """Thermalize + measure plaquette/action."""
    # warmup
    for _ in range(n_warmup):
        q = chain.evolve(q, 1)
    chain.reset_acceptance()

    plaquettes = []
    actions_gauge = []
    for k in range(n_meas):
        q = chain.evolve(q, skip)
        plaq = float(jnp.mean(theory.average_plaquette(q)))
        # use parent class action (gauge only) to isolate gauge observable
        from jaxqft.models.u1_ym import U1YangMills
        s_gauge = float(jnp.mean(U1YangMills.action(theory, q)))
        plaquettes.append(plaq)
        actions_gauge.append(s_gauge)
        if k % 50 == 0:
            print(f"  [{label}] k={k}/{n_meas}  plaq={plaq:.6f}  Sg={s_gauge:.4f}  acc={chain.calc_acceptance():.3f}")

    return q, plaquettes, actions_gauge, chain.calc_acceptance()


def main():
    # Small 2D lattice for quick test
    L = 8
    lattice_shape = (L, L)
    beta = 2.0
    mass = 0.1
    batch = 4
    nmd = 10
    tau = 1.0
    seed = 42

    n_warmup = 100
    n_meas = 200
    skip = 5

    print(f"Lattice: {lattice_shape}, beta={beta}, mass={mass}, batch={batch}")
    print(f"nmd={nmd}, tau={tau}, warmup={n_warmup}, meas={n_meas}, skip={skip}")
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # --- HMC ---
    print("=== HMC ===")
    th_hmc = U1WilsonNf2(
        lattice_shape=lattice_shape,
        beta=beta,
        mass=mass,
        batch_size=batch,
        seed=seed,
        pseudofermion_refresh="heatbath",
        pseudofermion_force_mode="autodiff",
        solver_kind="cg",
        solver_form="normal",
        cg_tol=1e-10,
        cg_maxiter=1000,
    )
    I_hmc = minnorm2(th_hmc.force, th_hmc.evolveQ, nmd, tau)
    chain_hmc = HMC(T=th_hmc, I=I_hmc, verbose=False, seed=seed, use_fast_jit=False)
    q_hmc = th_hmc.hot_start(scale=0.2)

    q_hmc, plaq_hmc, sg_hmc, acc_hmc = run_chain(
        th_hmc, chain_hmc, q_hmc, n_warmup, n_meas, skip, label="HMC"
    )

    # --- SMD (with accept/reject = standard GHMC) ---
    print("\n=== SMD (accept_reject=True) ===")
    th_smd = U1WilsonNf2(
        lattice_shape=lattice_shape,
        beta=beta,
        mass=mass,
        batch_size=batch,
        seed=seed,
        pseudofermion_refresh="heatbath",
        pseudofermion_force_mode="autodiff",
        solver_kind="cg",
        solver_form="normal",
        cg_tol=1e-10,
        cg_maxiter=1000,
    )
    I_smd = minnorm2(th_smd.force, th_smd.evolveQ, nmd, tau)
    chain_smd = SMD(
        T=th_smd, I=I_smd, gamma=0.3, accept_reject=True,
        verbose=False, seed=seed, use_fast_jit=False,
    )
    q_smd = th_smd.hot_start(scale=0.2)

    q_smd, plaq_smd, sg_smd, acc_smd = run_chain(
        th_smd, chain_smd, q_smd, n_warmup, n_meas, skip, label="SMD"
    )

    # --- SMD (without accept/reject = pure SMD) ---
    print("\n=== SMD (accept_reject=False, pure SMD) ===")
    th_smd2 = U1WilsonNf2(
        lattice_shape=lattice_shape,
        beta=beta,
        mass=mass,
        batch_size=batch,
        seed=seed,
        pseudofermion_refresh="heatbath",
        pseudofermion_force_mode="autodiff",
        solver_kind="cg",
        solver_form="normal",
        cg_tol=1e-10,
        cg_maxiter=1000,
    )
    I_smd2 = minnorm2(th_smd2.force, th_smd2.evolveQ, nmd, tau)
    chain_smd2 = SMD(
        T=th_smd2, I=I_smd2, gamma=0.3, accept_reject=False,
        verbose=False, seed=seed, use_fast_jit=False,
    )
    q_smd2 = th_smd2.hot_start(scale=0.2)

    q_smd2, plaq_smd2, sg_smd2, acc_smd2 = run_chain(
        th_smd2, chain_smd2, q_smd2, n_warmup, n_meas, skip, label="SMD-noAR"
    )

    # --- Also run pure gauge (no fermions) to verify those agree ---
    print("\n=== HMC (pure gauge) ===")
    th_pg = U1WilsonNf2(
        lattice_shape=lattice_shape,
        beta=beta,
        mass=mass,
        batch_size=batch,
        seed=seed,
        include_fermion_monomial=False,
    )
    I_pg = minnorm2(th_pg.force, th_pg.evolveQ, nmd, tau)
    chain_pg = HMC(T=th_pg, I=I_pg, verbose=False, seed=seed, use_fast_jit=False)
    q_pg = th_pg.hot_start(scale=0.2)
    q_pg, plaq_pg, sg_pg, acc_pg = run_chain(
        th_pg, chain_pg, q_pg, n_warmup, n_meas, skip, label="PG-HMC"
    )

    print("\n=== SMD (pure gauge) ===")
    th_pg2 = U1WilsonNf2(
        lattice_shape=lattice_shape,
        beta=beta,
        mass=mass,
        batch_size=batch,
        seed=seed,
        include_fermion_monomial=False,
    )
    I_pg2 = minnorm2(th_pg2.force, th_pg2.evolveQ, nmd, tau)
    chain_pg2 = SMD(
        T=th_pg2, I=I_pg2, gamma=0.3, accept_reject=True,
        verbose=False, seed=seed, use_fast_jit=False,
    )
    q_pg2 = th_pg2.hot_start(scale=0.2)
    q_pg2, plaq_pg2, sg_pg2, acc_pg2 = run_chain(
        th_pg2, chain_pg2, q_pg2, n_warmup, n_meas, skip, label="PG-SMD"
    )

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    def report(name, plaq, sg, acc):
        mp, ep = avg_err(plaq)
        ms, es = avg_err(sg)
        print(f"  {name:20s}: plaq = {mp:.6f} +/- {ep:.6f}  Sg = {ms:.4f} +/- {es:.4f}  acc = {acc:.3f}")

    print("With Nf=2 Wilson fermions:")
    report("HMC", plaq_hmc, sg_hmc, acc_hmc)
    report("SMD (AR=True)", plaq_smd, sg_smd, acc_smd)
    report("SMD (AR=False)", plaq_smd2, sg_smd2, acc_smd2)

    print("\nPure gauge (no fermions):")
    report("HMC", plaq_pg, sg_pg, acc_pg)
    report("SMD", plaq_pg2, sg_pg2, acc_pg2)

    # Statistical comparison
    def compare(name1, plaq1, name2, plaq2):
        m1, e1 = avg_err(plaq1)
        m2, e2 = avg_err(plaq2)
        diff = abs(m1 - m2)
        combined_err = np.sqrt(e1**2 + e2**2)
        sigma = diff / combined_err if combined_err > 0 else float("inf")
        ok = "OK" if sigma < 3.0 else "DISAGREE"
        print(f"  {name1} vs {name2}: delta_plaq = {diff:.6f}, {sigma:.1f} sigma  [{ok}]")
        return sigma

    print("\nStatistical comparisons (plaquette):")
    s1 = compare("HMC", plaq_hmc, "SMD(AR)", plaq_smd)
    s2 = compare("HMC", plaq_hmc, "SMD(noAR)", plaq_smd2)
    s3 = compare("PG-HMC", plaq_pg, "PG-SMD", plaq_pg2)

    print()
    if s3 > 3:
        print("WARNING: Pure gauge HMC vs SMD disagree. Something is wrong with the core algorithm.")
    else:
        print("Pure gauge: HMC and SMD agree (good).")
    if s1 > 3 or s2 > 3:
        print("PROBLEM: Nf=2 HMC vs SMD disagree. Bug in fermion sector interaction with SMD.")
    else:
        print("Nf=2: HMC and SMD agree within statistics.")


if __name__ == "__main__":
    main()

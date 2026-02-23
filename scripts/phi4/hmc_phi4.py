#!/usr/bin/env python3
"""JAX equivalent of torchQFT/hmc_phi4.py (core workflow)."""

from __future__ import annotations

import os
import platform
import time
import sys
from pathlib import Path
import numpy as np

# Metal backend is unstable on some Apple/JAX builds.
# On macOS only, default to CPU if no backend is explicitly selected.
if platform.system() == "Darwin" and "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

# Allow running as `python scripts/phi4/hmc_phi4.py` from repository root.
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
from jaxqft.models.phi4 import Phi4
from jaxqft.core.update import hmc
import jax


def average(d):
    m = np.mean(d)
    e = np.std(d) / np.sqrt(len(d) - 1)
    return m, e


def correlation_length(L, ChiM, C2p):
    return 1 / (2 * np.sin(np.pi / L)) * np.sqrt(ChiM / C2p - 1)


def main():
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])

    L = 64
    lat = [L, L]
    lam = 0.5
    mas = -0.205
    Nwarm = 100
    Nmeas = 100
    Nskip = 10
    batch_size = 16

    Vol = np.prod(lat)
    sg = Phi4(lat, lam, mas, batch_size=batch_size)
    phi = sg.hotStart()
    mn2 = minnorm2(sg.force, sg.evolveQ, 7, 1.0)
    chain = hmc(T=sg, I=mn2, verbose=False)

    tic = time.perf_counter()
    phi = chain.evolve(phi, Nwarm)
    toc = time.perf_counter()
    print(f"time {(toc - tic) * 1.0e6 / Nwarm:0.4f} micro-seconds per HMC trajectory (warmup)")

    lC2p = []
    lchi_m = []
    E = []
    av_phi = []
    phase = np.exp(1j * np.indices(tuple(lat))[0] * 2 * np.pi / lat[0])

    tic = time.perf_counter()
    for k in range(Nmeas):
        ttE = np.asarray(sg.action(phi) / Vol)
        E.extend(ttE.tolist())
        av_sigma = np.asarray(phi.reshape(sg.Bs, Vol).mean(axis=1))
        av_phi.extend(av_sigma.tolist())
        chi_m = av_sigma * av_sigma * Vol
        p1_av_sig = np.asarray((phi.reshape(sg.Bs, Vol) * phase.reshape(1, Vol)).mean(axis=1))
        C2p = np.real(np.conj(p1_av_sig) * p1_av_sig) * Vol
        if k % 10 == 0:
            print(
                "k=",
                k,
                "(av_phi,chi_m, c2p, E)",
                float(av_sigma.mean()),
                float(chi_m.mean()),
                float(C2p.mean()),
                float(ttE.mean()),
            )
        lC2p.extend(C2p.tolist())
        lchi_m.extend(chi_m.tolist())
        phi = chain.evolve(phi, Nskip)
    toc = time.perf_counter()
    print(f"time {(toc - tic) * 1.0e6 / (Nmeas * Nskip):0.4f} micro-seconds per HMC trajectory (meas updates)")

    m_phi, e_phi = average(np.asarray(av_phi))
    m_chi_m, e_chi_m = average(np.asarray(lchi_m) - (m_phi**2) * Vol)
    m_C2p, e_C2p = average(np.asarray(lC2p))
    avE, eE = average(np.asarray(E))
    xi = correlation_length(lat[0], m_chi_m, m_C2p)

    print("m_phi:", m_phi, e_phi)
    print("Chi_m:", m_chi_m, e_chi_m)
    print("C2p:", m_C2p, e_C2p)
    print("E:", avE, "+/-", eE)
    print("xi:", xi)
    print("Acceptance rate:", chain.calc_Acceptance())


if __name__ == "__main__":
    main()

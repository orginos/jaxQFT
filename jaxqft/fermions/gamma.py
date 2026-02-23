"""Dimension-aware Euclidean gamma matrices."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp


def _kron(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(a, b)


def build_euclidean_gamma(ndim: int, dtype=jnp.complex64) -> jnp.ndarray:
    """Return gamma[mu] with {gamma_mu, gamma_nu} = 2 delta_{mu nu}."""
    if int(ndim) < 1:
        raise ValueError("ndim must be >= 1")
    d = int(ndim)

    s1 = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    s2 = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype)
    s3 = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    eye2 = jnp.eye(2, dtype=dtype)

    if d == 1:
        return jnp.asarray([jnp.asarray([[1.0]], dtype=dtype)])
    if d == 2:
        return jnp.asarray([s1, s2])

    gam = [s1, s2]
    cur_d = 2
    while cur_d + 1 < d:
        m = gam[0].shape[0]
        eye_m = jnp.eye(m, dtype=dtype)
        gam = [_kron(s1, g) for g in gam] + [_kron(s2, eye_m), _kron(s3, eye_m)]
        cur_d += 2

    if cur_d == d:
        return jnp.asarray(gam)

    # Odd dimension: add chirality-like matrix that anticommutes with first d-1.
    n = (d - 1) // 2
    g_last = jnp.asarray((1j) ** n, dtype=dtype) * gam[0]
    for mu in range(1, d - 1):
        g_last = g_last @ gam[mu]
    return jnp.asarray(gam + [g_last])


def gamma5(gam: jnp.ndarray) -> jnp.ndarray:
    """gamma5 for even-dimensional Euclidean Clifford algebra."""
    d = int(gam.shape[0])
    if d % 2 != 0:
        raise ValueError("gamma5 is defined here only for even ndim")
    out = jnp.asarray((1j) ** (d // 2), dtype=gam.dtype) * gam[0]
    for mu in range(1, d):
        out = out @ gam[mu]
    return out


def check_gamma_algebra(gam: jnp.ndarray) -> Dict[str, float]:
    d = int(gam.shape[0])
    n = int(gam.shape[-1])
    eye = jnp.eye(n, dtype=gam.dtype)
    max_rel = 0.0
    max_herm = 0.0
    for mu in range(d):
        gm = gam[mu]
        max_herm = max(max_herm, float(jnp.linalg.norm(gm - jnp.conj(gm.T))))
        for nu in range(d):
            anti = gm @ gam[nu] + gam[nu] @ gm
            target = 2.0 * eye if mu == nu else jnp.zeros_like(eye)
            denom = float(jnp.linalg.norm(target)) + 1e-12
            rel = float(jnp.linalg.norm(anti - target)) / denom
            max_rel = max(max_rel, rel)
    return {"max_rel_clifford_error": float(max_rel), "max_hermiticity_error": float(max_herm)}


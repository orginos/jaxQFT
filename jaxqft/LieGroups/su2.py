import numpy as np
import jax.numpy as jnp

from . import LieProjectCmplx, ad, dexpo as taylor_dexpo, expo as taylor_expo, levi_civita_tensor


tau = 0.5j * jnp.array(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0j], [-1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=jnp.complex64,
)

levi_civita = levi_civita_tensor(3).astype(tau.dtype)


def check_algebra():
    foo = jnp.einsum("aik,bkj->abij", tau, tau)
    foo = foo - jnp.swapaxes(foo, 0, 1)
    boo = jnp.einsum("abk,kij->abij", levi_civita, tau)
    return jnp.linalg.norm(foo - boo) / jnp.linalg.norm(boo)


def expo(X):
    nn = (jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + jnp.finfo(X.real.dtype).eps
    I = jnp.broadcast_to(jnp.eye(2, dtype=X.dtype), X.shape)
    return jnp.cos(nn) * I + X * (jnp.sin(nn) / nn)


def dexpo(X, Y):
    nX = (jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + jnp.finfo(X.real.dtype).eps
    adj = ad(X, Y)
    two_nX = 2.0 * nX
    sin_two_nX_over_two_nX = jnp.sin(two_nX) / two_nX
    return Y - (0.5 * (jnp.sin(nX) / nX) ** 2) * adj + 0.25 * (1.0 - sin_two_nX_over_two_nX) / (nX * nX) * ad(X, adj)


def simpsons_rule(fx, h):
    if fx.shape[-1] % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    S = fx[..., 0] + fx[..., -1]
    S = S + 4 * fx[..., 1:-1:2].sum(axis=-1)
    S = S + 2 * fx[..., 2:-2:2].sum(axis=-1)
    return (h / 3.0) * S


def dexpo_step_by_step(X, Y, N=100):
    nX = (jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + jnp.finfo(X.real.dtype).eps
    term1 = Y
    term2 = 0.5 * (jnp.sin(nX) / nX) ** 2 * ad(X, Y)
    term3 = 0.25 * (1.0 - jnp.sin(nX) / nX * jnp.cos(nX)) / (nX * nX) * ad(X, ad(X, Y))
    return term1 - term2 + term3


def check_expo_and_dexpo():
    X = LieProjectCmplx(np.random.randn(4, 2, 2, 2, 2) + 1j * np.random.randn(4, 2, 2, 2, 2))
    Y = LieProjectCmplx(np.random.randn(4, 2, 2, 2, 2) + 1j * np.random.randn(4, 2, 2, 2, 2))
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    dexpoXY = dexpo(X, Y)
    dexpoXYiter = taylor_dexpo(X, Y)
    return {
        "expo_rel": jnp.linalg.norm(expoX - t_expoX) / jnp.linalg.norm(expoX),
        "dexpo_rel": jnp.linalg.norm(dexpoXYiter - dexpoXY) / jnp.linalg.norm(dexpoXYiter),
    }


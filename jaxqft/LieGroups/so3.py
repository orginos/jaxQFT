import numpy as np
import jax.numpy as jnp

from . import LieProject, ad, dexpo as taylor_dexpo, expo as taylor_expo, levi_civita_tensor


L = jnp.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    dtype=jnp.float32,
)

levi_civita = levi_civita_tensor(3)


def check_algebra():
    foo = jnp.einsum("aik,bkj->abij", L, L)
    foo = foo - jnp.swapaxes(foo, 0, 1)
    boo = jnp.einsum("abk,kij->abij", levi_civita, L)
    return jnp.linalg.norm(foo - boo) / jnp.linalg.norm(boo)


def expo(X):
    nX = (jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + jnp.finfo(X.dtype).eps
    I = jnp.broadcast_to(jnp.eye(3, dtype=X.dtype), X.shape)
    return I + jnp.sin(nX) / nX * X + (1.0 - jnp.cos(nX)) / (nX * nX) * (X @ X)


def expo_old(X):
    nX = jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)
    I = jnp.broadcast_to(jnp.eye(3, dtype=X.dtype), X.shape)
    R = I + jnp.sin(nX) / nX * X + (1.0 - jnp.cos(nX)) / (nX * nX) * (X @ X)
    return jnp.where(nX < jnp.finfo(X.dtype).eps, I, R)


def dexpo(X, Y):
    nX = (jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)) + jnp.finfo(X.dtype).eps
    adj = ad(X, Y)
    nX2 = nX**2
    return Y - (1.0 - jnp.cos(nX)) / nX2 * adj + (1.0 - jnp.sin(nX) / nX) / nX2 * ad(X, adj)


def dexpo_old(X, Y):
    nX = jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)
    adj = ad(X, Y)
    nX2 = nX**2
    value = Y - (1.0 - jnp.cos(nX)) / nX2 * adj + (1.0 - jnp.sin(nX) / nX) / nX2 * ad(X, adj)
    return jnp.where(nX < jnp.finfo(X.dtype).eps, Y, value)


def log(R):
    Tr = jnp.einsum("...ii->...", R)[..., None, None]
    cos = 0.5 * (Tr - 1.0)
    theta = jnp.arccos(cos)
    two_sin = jnp.sqrt((3.0 - Tr) * (1.0 + Tr)) + jnp.finfo(R.dtype).eps
    return theta / two_sin * (R - jnp.swapaxes(R, -1, -2))


def check_expo_and_dexpo():
    X = LieProject(np.random.randn(4, 2, 2, 3, 3).astype(np.float32))
    Y = LieProject(np.random.randn(4, 2, 2, 3, 3).astype(np.float32))
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    dexpoXY = dexpo(X, Y)
    dexpoXYiter = taylor_dexpo(X, Y)
    return {
        "expo_rel": jnp.linalg.norm(expoX - t_expoX) / jnp.linalg.norm(expoX),
        "dexpo_rel": jnp.linalg.norm(dexpoXYiter - dexpoXY) / jnp.linalg.norm(dexpoXYiter),
    }


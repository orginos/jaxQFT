import numpy as np
import jax.numpy as jnp

from . import LieProjectCmplx, expo as taylor_expo


T = 0.5j * jnp.array(
    [
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.0j, 0.0], [-1.0j, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0j], [0.0, 0.0, 0.0], [1.0j, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0j], [0.0, 1.0j, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
    ],
    dtype=jnp.complex64,
)
T = T.at[7].set(T[7] * (1.0 / np.sqrt(3.0)))


def calc_structure():
    foo = jnp.einsum("aik,bkj->abij", T, T)
    comm = foo - jnp.swapaxes(foo, 0, 1)
    acom = foo + jnp.swapaxes(foo, 0, 1)
    f = -2.0 * jnp.einsum("aik,bcki->abc", T, comm)
    d = (2.0 * 1j) * jnp.einsum("aik,bcki->abc", T, acom)
    return f, d


f_struc, d_struc = calc_structure()


def check_algebra():
    foo = jnp.einsum("aik,bkj->abij", T, T)
    comm = foo - jnp.swapaxes(foo, 0, 1)
    acom = foo + jnp.swapaxes(foo, 0, 1)
    norm_check = jnp.linalg.norm(jnp.einsum("aij,bji->ab", T, T) + 0.5 * jnp.eye(8))
    boo_comm = jnp.einsum("abc,cij->abij", f_struc, T)
    comm_check = jnp.linalg.norm(comm - boo_comm) / jnp.linalg.norm(boo_comm)
    boo_acom = jnp.einsum("abc,cij->abij", 1j * d_struc, T)
    eye = jnp.eye(8)[:, :, None, None] * jnp.eye(3)[None, None, :, :]
    diff = acom - boo_acom + (1.0 / 3.0) * eye
    d_check = jnp.linalg.norm(diff) / 8.0
    return {"norm_check": norm_check / 8.0, "comm_check": comm_check, "d_check": d_check}


def simpsons_rule(fx, h):
    if fx.shape[-1] % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    S = fx[..., 0] + fx[..., -1]
    S = S + 4 * fx[..., 1:-1:2].sum(axis=-1)
    S = S + 2 * fx[..., 2:-2:2].sum(axis=-1)
    return (h / 3.0) * S


def expo(X):
    nX = jnp.linalg.norm(X, axis=(-1, -2), keepdims=True) / np.sqrt(2.0)
    Xhat = X / (nX + jnp.finfo(X.real.dtype).eps)
    eta = jnp.linalg.det(Xhat)[..., None, None]
    psi = (jnp.arccos(-jnp.imag(eta) * 3.0 * np.sqrt(3.0) / 2.0) / 3.0).squeeze(-1).squeeze(-1)
    z = jnp.zeros((*X.shape[:-2], 3), dtype=X.dtype)
    z = z.at[..., 0].set(2.0 / np.sqrt(3.0) * jnp.cos(psi) * 1j)
    z = z.at[..., 1].set((-jnp.sin(psi) - 1.0 / np.sqrt(3.0) * jnp.cos(psi)) * 1j)
    z = z.at[..., 2].set((jnp.sin(psi) - 1.0 / np.sqrt(3.0) * jnp.cos(psi)) * 1j)
    z = z[..., None, None]
    Xhat3 = Xhat[..., None, :, :]
    nX3 = nX[..., None, :, :]
    I = jnp.eye(3, dtype=X.dtype)
    I = jnp.broadcast_to(I, Xhat3.shape)
    num = jnp.exp(z * nX3) * ((z * z + 1.0) * I + z * Xhat3 + Xhat3 @ Xhat3)
    den = 3.0 * z * z + 1.0 + jnp.finfo(X.real.dtype).eps
    return jnp.sum(num / den, axis=-3)


def check_expo_and_dexpo():
    X = LieProjectCmplx(np.random.randn(4, 2, 2, 3, 3) + 1j * np.random.randn(4, 2, 2, 3, 3))
    expoX = expo(X)
    t_expoX = taylor_expo(X)
    return {"expo_rel": jnp.linalg.norm(expoX - t_expoX) / jnp.linalg.norm(expoX)}


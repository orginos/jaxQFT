import itertools
import math

import jax.numpy as jnp

version = "1.0"
author = "Kostas Orginos"


def levi_civita_tensor(N):
    perms = list(itertools.permutations(range(N)))
    levi_civita = jnp.zeros((N,) * N, dtype=jnp.float32)
    for p in perms:
        sign = 1.0
        for j in range(N):
            for k in range(j + 1, N):
                if p[j] > p[k]:
                    sign *= -1.0
        levi_civita = levi_civita.at[p].set(sign)
    return levi_civita


def ad(x, y):
    return x @ y - y @ x


def LieProject(X):
    return 0.5 * (X - jnp.swapaxes(X, -1, -2))


def LieProjectCmplx(X):
    N = X.shape[-1]
    Y = 0.5 * (X - jnp.conj(jnp.swapaxes(X, -1, -2)))
    T = (jnp.einsum("...aa->...", Y) / N)[..., None, None]
    I = jnp.eye(N, dtype=X.dtype)
    return Y - jnp.broadcast_to(I, X.shape) * T


def expo(X):
    return jnp.linalg.matrix_exp(X)


def dexpo(x, y, Ntaylor=20):
    r = ((-1) ** (Ntaylor % 2)) / math.factorial(Ntaylor + 1) * y
    for k in range(Ntaylor - 1, 0, -1):
        r = ad(x, r) + ((-1) ** (k % 2)) / math.factorial(k + 1) * y
    r = ad(x, r) + y
    return r


from . import so3  # noqa: E402
from . import su2  # noqa: E402
from . import su3  # noqa: E402


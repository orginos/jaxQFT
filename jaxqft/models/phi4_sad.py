"""SAD (Stochastic Automatic Differentiation) building blocks for phi^4 theory.

Reference: Catumba & Ramos, arXiv:2502.15570 (2025).

Fields have shape (T, L) for a 2D periodic lattice:
  axis 0  = time (length T)
  axis 1  = space (length L)

No batch dimension here; use jax.vmap externally for independent chains.

Action convention (matches jaxqft/models/phi4.py):

  S_J[phi] = sum_x [(mtil/2)*phi(x)^2 + (lam/24)*phi(x)^4]
           - sum_{x,mu} phi(x)*phi(x + mu_hat)
           - J * sum_x phi(x, t=0)

where  mtil = m2 + 2*Nd,  Nd = phi.ndim = 2.

With this sign convention  C(t) = d/dJ <phi(x,t)>_J|_{J=0}  is positive
and equals the connected propagator  sum_{x'} G_conn(0,t; x',0).

The leapfrog MUST use jax.lax.scan (not a Python for-loop).
Without scan, jax.jvp unrolls the full n_steps computation graph at trace
time, causing extremely slow compilation or OOM for realistic step counts.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Action and force
# ---------------------------------------------------------------------------

def action_J(phi: jax.Array, J: float | jax.Array, m2: float, lam: float) -> jax.Array:
    """phi^4 action with a linear source J at timeslice t=0.

    Args:
        phi: field array of shape (T, L).
        J:   source coupling (scalar).
        m2:  bare mass squared parameter.
        lam: quartic coupling.

    Returns:
        Scalar action value S_J[phi].
    """
    Nd = phi.ndim          # 2 for a 2D field
    mtil = m2 + 2.0 * Nd
    phi2 = phi * phi
    S = jnp.sum((0.5 * mtil + (lam / 24.0) * phi2) * phi2)
    for mu in range(Nd):
        S = S - jnp.sum(phi * jnp.roll(phi, shift=-1, axis=mu))
    S = S - J * jnp.sum(phi[0])
    return S


def force_J(phi: jax.Array, J: float | jax.Array, m2: float, lam: float) -> jax.Array:
    """Force F = -dS_J/dphi, computed via jax.grad."""
    return -jax.grad(action_J, argnums=0)(phi, J, m2, lam)


# ---------------------------------------------------------------------------
# lax.scan leapfrog  (CRITICAL: scan is mandatory for efficient jvp)
# ---------------------------------------------------------------------------

def leapfrog_scan(
    phi: jax.Array,
    pi: jax.Array,
    J: float | jax.Array,
    m2: float,
    lam: float,
    dt: float,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """BAB leapfrog integrator built on jax.lax.scan.

    Using lax.scan is mandatory when this function is called inside jax.jvp.
    A Python for-loop forces the JVP to unroll n_steps copies of the
    computation graph at trace time, which is catastrophically slow.
    With scan, JAX differentiates the loop body once and applies it n_steps
    times, keeping compilation O(1) in n_steps.

    Args:
        phi:     field, shape (T, L).
        pi:      conjugate momentum, same shape.
        J:       source value (float or scalar jax array).
        m2:      bare mass squared.
        lam:     quartic coupling.
        dt:      leapfrog step size.
        n_steps: number of BAB steps.

    Returns:
        (phi_out, pi_out) after n_steps leapfrog steps.
    """
    def step(carry, _):
        phi_c, pi_c = carry
        F = force_J(phi_c, J, m2, lam)
        pi_c = pi_c + 0.5 * dt * F
        phi_c = phi_c + dt * pi_c
        F = force_J(phi_c, J, m2, lam)
        pi_c = pi_c + 0.5 * dt * F
        return (phi_c, pi_c), None

    (phi_out, pi_out), _ = jax.lax.scan(step, (phi, pi), None, length=int(n_steps))
    return phi_out, pi_out


# ---------------------------------------------------------------------------
# SMD step with accumulated SAD tangent
# ---------------------------------------------------------------------------

def smd_sad_step(
    phi: jax.Array,
    pi: jax.Array,
    phi_tan: jax.Array,
    pi_tan: jax.Array,
    key: jax.Array,
    m2: float,
    lam: float,
    dt: float,
    n_steps: int,
    c1: float,
    c2: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One SMD step (no accept/reject) with accumulated SAD tangent propagation.

    The tangent pair (phi_tan, pi_tan) tracks the *total* derivatives
    d(phi)/dJ and d(pi)/dJ accumulated over the entire Markov chain.
    At each step the tangent is updated in two stages:

      1. OU momentum refresh:  pi_ou     = c1*pi     + c2*eta
                                pi_ou_tan = c1*pi_tan
         (eta is a fresh random draw independent of J)

      2. Leapfrog integration:
           primal:  (phi_new, pi_new)         = leapfrog(phi, pi_ou, J=0)
           tangent: (phi_new_tan, pi_new_tan) = jvp of leapfrog w.r.t.
                    (phi, pi_ou, J) at tangents (phi_tan, pi_ou_tan, 1.0)

    The tangent for J is fixed at 1.0, providing a fresh source kick at
    every step.  The (phi_tan, pi_ou_tan) tangents propagate accumulated
    information from all previous steps through the chain rule.  The OU
    damping factor c1 < 1 ensures the tangent converges geometrically.

    The observable is the spatial mean of phi_tan:
      C_sad(t) = E[ mean_x phi_tan(x,t) ]  ->  C(t)  (connected propagator)

    Args:
        phi, pi:          current primal field and momentum, shape (T, L).
        phi_tan, pi_tan:  accumulated tangent state from previous steps.
        key:              JAX PRNG key.
        m2, lam:          theory parameters.
        dt:               leapfrog step size.
        n_steps:          leapfrog steps per trajectory.
        c1, c2:           OU coefficients: c1=exp(-gamma*tau), c2=sqrt(1-c1^2).

    Returns:
        phi_new:     updated primal field.
        pi_new:      updated primal momentum.
        phi_new_tan: updated tangent for field.
        pi_new_tan:  updated tangent for momentum.
    """
    # 1. OU momentum refresh (primal and tangent)
    eta = jax.random.normal(key, phi.shape)
    pi_ou     = c1 * pi     + c2 * eta
    pi_ou_tan = c1 * pi_tan          # eta is J-independent -> d(eta)/dJ = 0

    # 2. Leapfrog with jvp through (phi, pi_ou, J)
    #    - phi_tan, pi_ou_tan propagate accumulated chain-rule information
    #    - the J-tangent = 1.0 injects the fresh source at every step
    def traj(phi_in, pi_in, J):
        return leapfrog_scan(phi_in, pi_in, J, m2, lam, dt, n_steps)

    primals  = (phi,     pi_ou,     0.0)
    tangents = (phi_tan, pi_ou_tan, 1.0)
    (phi_new, pi_new), (phi_new_tan, pi_new_tan) = jax.jvp(traj, primals, tangents)

    return phi_new, pi_new, phi_new_tan, pi_new_tan


# ---------------------------------------------------------------------------
# Analytic free-field propagator (lam=0 reference)
# ---------------------------------------------------------------------------

def free_propagator(T: int, L: int, m2: float) -> np.ndarray:
    """Exact connected correlator C(t) for the free scalar on a T x L lattice.

    Definition (matches the SAD estimator convention):
      C(t) = d/dJ <phi(x,t)>_J |_{J=0}
           = sum_{x'=0}^{L-1} G_conn(0, t; x', 0)

    For the free field the k_x = 0 projection gives:
      C(t) = (1/T) * sum_{k_t=0}^{T-1}  exp(2*pi*i*k_t*t/T) / D(k_t)
      D(k_t) = m2 + 4 * sin^2(pi * k_t / T)

    This is simply the inverse DFT of 1/D, which NumPy computes as
      C = real(ifft(1/D))
    using ifft(X)[t] = (1/N) * sum_k X[k] * exp(2*pi*i*k*t/N).

    Note: valid only for lam=0 and m2 > 0 (D(k_t) > 0 for all k_t).

    Args:
        T:   number of timeslices.
        L:   spatial extent (enters only through the source normalization;
             the result is independent of L for the free field).
        m2:  bare mass squared (must be > 0 for the free field to be stable).

    Returns:
        C_exact: real array of shape (T,).
    """
    k_t = np.arange(T, dtype=np.float64)
    D = m2 + 4.0 * np.sin(np.pi * k_t / T) ** 2
    if np.any(D <= 0):
        raise ValueError(
            f"free_propagator: D(k) <= 0 for some k (m2={m2}). "
            "The free field is unstable; use lam>0 for m2<=0."
        )
    C = np.real(np.fft.ifft(1.0 / D))
    return C

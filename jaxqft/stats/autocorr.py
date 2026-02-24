"""Autocorrelation and integrated autocorrelation time estimators."""

from __future__ import annotations

import numpy as np


def autocorrelation_fft(x) -> np.ndarray:
    """Return normalized autocorrelation function rho[t] for a 1D series."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    y = a - np.mean(a)
    c0 = float(np.dot(y, y))
    if not np.isfinite(c0) or c0 <= 0.0:
        return np.full((n,), np.nan, dtype=np.float64)

    # O(n log n) autocovariance via FFT, normalized to rho[0] = 1.
    f = np.fft.rfft(y, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
    return np.asarray(ac / c0, dtype=np.float64)


def integrated_autocorr_time(
    x,
    *,
    method: str = "ips",
    c: float = 5.0,
    max_lag: int | None = None,
) -> dict:
    """Estimate integrated autocorrelation time (IAT).

    Args:
        x: 1D time series.
        method: "ips" (initial positive sequence), "sokal" (self-consistent
            window), or "gamma" (Wolff-style automatic window on Gamma(t)).
        c: Sokal window constant (only used for method="sokal").
        max_lag: Optional cap on lag used in the estimate.

    Returns:
        Dict with keys:
            tau_int: IAT estimate in sample units.
            ess: Effective sample size n / (2 * tau_int).
            window: Last lag used by the estimator.
            n: Number of samples.
            method: Method used.
            ok: Whether estimate succeeded.
            message: Status text.
    """
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n < 2:
        return {
            "tau_int": float("nan"),
            "ess": float("nan"),
            "window": 0,
            "n": n,
            "method": method,
            "ok": False,
            "message": "need at least 2 samples",
        }

    rho = autocorrelation_fft(a)
    if rho.size == 0 or not np.isfinite(rho[0]):
        return {
            "tau_int": float("nan"),
            "ess": float("nan"),
            "window": 0,
            "n": n,
            "method": method,
            "ok": False,
            "message": "zero or non-finite variance",
        }

    mmax = n - 1 if max_lag is None else max(1, min(int(max_lag), n - 1))
    method = str(method).lower()

    if method == "ips":
        tau = 0.5
        w = 0
        # Geyer's initial positive sequence on paired sums.
        t = 1
        while t + 1 <= mmax:
            g = float(rho[t] + rho[t + 1])
            if not np.isfinite(g) or g <= 0.0:
                break
            tau += g
            w = t + 1
            t += 2
        if w == 0:
            tau = 0.5
        tau = max(float(tau), 0.5)
        ess = float(n / (2.0 * tau))
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
        }

    if method == "sokal":
        tau = 0.5
        w = 1
        # Fixed-point iteration for self-consistent window w ~= c * tau.
        for _ in range(32):
            w = max(1, min(mmax, int(np.floor(c * tau))))
            r = rho[1 : w + 1]
            r = r[np.isfinite(r)]
            tau_new = float(max(0.5, 0.5 + np.sum(r)))
            if abs(tau_new - tau) <= 1e-6 * max(1.0, tau):
                tau = tau_new
                break
            tau = tau_new
        ess = float(n / (2.0 * tau))
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
        }

    if method == "gamma":
        # Wolff-style gamma analysis with automatic windowing criterion:
        # pick first W such that W >= c * tau_int(W), where tau_int(W)
        # is built from normalized autocovariance rho(t)=Gamma(t)/Gamma(0).
        y = a - np.mean(a)
        f = np.fft.rfft(y, n=2 * n)
        ac = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        norm = np.arange(n, 0, -1, dtype=np.float64)  # unbiased Gamma(t)
        gamma = ac / norm
        gamma0 = float(gamma[0])
        if not np.isfinite(gamma0) or gamma0 <= 0.0:
            return {
                "tau_int": float("nan"),
                "ess": float("nan"),
                "window": 0,
                "n": n,
                "method": method,
                "ok": False,
                "message": "non-positive Gamma(0)",
            }
        rho_g = np.asarray(gamma / gamma0, dtype=np.float64)
        rho_g = np.where(np.isfinite(rho_g), rho_g, 0.0)

        tau = 0.5
        w = mmax
        for t in range(1, mmax + 1):
            tau += float(rho_g[t])
            tau = max(tau, 0.5)
            if t >= c * tau:
                w = t
                break

        tau = max(float(tau), 0.5)
        ess = float(n / (2.0 * tau))
        var_mean = float((gamma0 + 2.0 * np.sum(gamma[1 : w + 1])) / n)
        if not np.isfinite(var_mean) or var_mean < 0.0:
            var_mean = float("nan")
        return {
            "tau_int": tau,
            "ess": ess,
            "window": int(w),
            "n": n,
            "method": method,
            "ok": True,
            "message": "ok",
            "var_mean": var_mean,
            "sigma_mean": float(np.sqrt(var_mean)) if np.isfinite(var_mean) else float("nan"),
        }

    raise ValueError(f"Unknown IAT method: {method}")

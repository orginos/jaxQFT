"""Reusable Wilson-Dirac operator blocks for lattice gauge theories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import jax.numpy as jnp
import numpy as np

from .gamma import build_euclidean_gamma


Array = jnp.ndarray


@dataclass
class WilsonDiracOperator:
    """Wilson-Dirac operator independent of gauge group choice.

    The gauge-theory model supplies geometry/layout callbacks:
    - `take_mu(U, mu)` -> link field U_mu with shape [..., Nc, Nc] (or scalar links for U(1))
    - `roll_site(x, shift, dir)` -> nearest-neighbor shift on site axes
    - `dagger(U)` -> matrix adjoint (or complex conjugate for scalar links)
    """

    ndim: int
    mass: float = 0.0
    wilson_r: float = 1.0
    dtype: jnp.dtype = jnp.complex64
    gamma: Array | None = None
    use_half_spinor: bool = False

    def __post_init__(self):
        self.Nd = int(self.ndim)
        if self.Nd < 1:
            raise ValueError("ndim must be >= 1")
        self.gamma = build_euclidean_gamma(self.Nd, dtype=self.dtype) if self.gamma is None else self.gamma
        self.Ns = int(self.gamma.shape[-1])
        self.spin_eye = jnp.eye(self.Ns, dtype=self.dtype)
        self._gamma_sparse_ok = False
        self._gamma_perm = None
        self._gamma_phase = None
        self._half_spinor_ok = False
        self._projA = None
        self._projB = None
        self._setup_gamma_sparse()
        self._setup_half_spinor()

    def _setup_gamma_sparse(self) -> None:
        perm = []
        phase = []
        ok = True
        for mu in range(self.Nd):
            g = jnp.asarray(self.gamma[mu])
            row_nnz = jnp.sum(jnp.abs(g) > 1e-7, axis=1)
            if not bool(jnp.all(row_nnz == 1)):
                ok = False
                break
            idx = jnp.argmax(jnp.abs(g), axis=1).astype(jnp.int32)
            val = g[jnp.arange(self.Ns), idx]
            perm.append(idx)
            phase.append(val)
        if ok:
            self._gamma_sparse_ok = True
            self._gamma_perm = jnp.stack(perm, axis=0)  # [Nd, Ns]
            self._gamma_phase = jnp.stack(phase, axis=0)  # [Nd, Ns]

    @property
    def sparse_gamma_available(self) -> bool:
        return bool(self._gamma_sparse_ok)

    @property
    def half_spinor_available(self) -> bool:
        return bool(self._half_spinor_ok)

    def _setup_half_spinor(self) -> None:
        # Low-rank projector factors for common Wilson choice r=1 in Ns=4.
        self._half_spinor_ok = False
        self._projA = None
        self._projB = None
        if not bool(self.use_half_spinor):
            return
        if self.Ns != 4:
            return
        if abs(float(self.wilson_r) - 1.0) > 1e-8:
            return
        eye = np.eye(self.Ns, dtype=np.complex64)
        A_all = np.zeros((self.Nd, 2, self.Ns, 2), dtype=np.complex64)
        B_all = np.zeros((self.Nd, 2, 2, self.Ns), dtype=np.complex64)
        for mu in range(self.Nd):
            g = np.asarray(self.gamma[mu], dtype=np.complex64)
            for i, coeff in enumerate((-1.0, +1.0)):
                P = eye + coeff * g
                u, s, vh = np.linalg.svd(P, full_matrices=False)
                rank = int(np.sum(s > 1e-6))
                if rank != 2:
                    return
                A = (u[:, :2] * s[:2][None, :]).astype(np.complex64)
                B = vh[:2, :].astype(np.complex64)
                A_all[mu, i] = A
                B_all[mu, i] = B
        self._projA = jnp.asarray(A_all, dtype=self.dtype)  # [Nd,2,4,2], coeff idx:0->-1,1->+1
        self._projB = jnp.asarray(B_all, dtype=self.dtype)  # [Nd,2,2,4]
        self._half_spinor_ok = True

    def color_mul_left(self, U: Array, psi: Array) -> Array:
        # U: [...,Nc,Nc] or [...] (scalar U(1)), psi: [...,Ns,Nc] -> [...,Ns,Nc]
        if U.ndim == psi.ndim - 2:
            return U[..., None, None] * psi
        # Specialized small-Nc paths are substantially faster than generic einsum
        # on CPU (and often GPU) for Wilson matvec hot loops.
        if U.shape[-2:] == (3, 3):
            p0 = psi[..., 0]
            p1 = psi[..., 1]
            p2 = psi[..., 2]
            o0 = U[..., 0, 0, None] * p0 + U[..., 0, 1, None] * p1 + U[..., 0, 2, None] * p2
            o1 = U[..., 1, 0, None] * p0 + U[..., 1, 1, None] * p1 + U[..., 1, 2, None] * p2
            o2 = U[..., 2, 0, None] * p0 + U[..., 2, 1, None] * p1 + U[..., 2, 2, None] * p2
            return jnp.stack((o0, o1, o2), axis=-1)
        if U.shape[-2:] == (2, 2):
            p0 = psi[..., 0]
            p1 = psi[..., 1]
            o0 = U[..., 0, 0, None] * p0 + U[..., 0, 1, None] * p1
            o1 = U[..., 1, 0, None] * p0 + U[..., 1, 1, None] * p1
            return jnp.stack((o0, o1), axis=-1)
        return jnp.einsum("...ab,...sb->...sa", U, psi)

    @property
    def diagonal_mass(self) -> float:
        # Wilson diagonal block coefficient for the standard operator.
        return float(self.mass + self.wilson_r * self.Nd)

    def apply_diag(self, psi: Array, sign: int = -1) -> Array:
        """Apply the diagonal block.

        For Wilson this is sign-independent: (m + r*Nd) I.
        `sign` is kept in the signature so derived operators (e.g. clover)
        can specialize D vs D^\\dagger diagonal application without changing the API.
        """
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (D) or +1 (Ddag)")
        return self.diagonal_mass * psi

    def gamma_apply(self, psi: Array, mu: int, use_sparse: bool = True) -> Array:
        # psi: [...,Ns,Nc]
        if use_sparse and self._gamma_sparse_ok:
            idx = self._gamma_perm[mu]  # [Ns]
            ph = self._gamma_phase[mu]  # [Ns]
            gathered = jnp.take(psi, idx, axis=-2)
            ph_shape = (1,) * (psi.ndim - 2) + (self.Ns, 1)
            return jnp.reshape(ph, ph_shape) * gathered
        return jnp.einsum("st,...tc->...sc", self.gamma[mu], psi)

    def spin_project(self, psi: Array, mu: int, coeff: float, use_sparse: bool = True) -> Array:
        # (r I + coeff * gamma_mu) psi
        return self.wilson_r * psi + coeff * self.gamma_apply(psi, mu, use_sparse=use_sparse)

    def projected_hop(self, U: Array, psi: Array, mu: int, coeff: float, use_sparse: bool = True) -> Array:
        # Compute (r I + coeff*gamma_mu) [U psi] with an optional low-rank half-spinor path.
        if self._half_spinor_ok and coeff in (-1.0, 1.0):
            idx = 0 if coeff < 0 else 1
            B = self._projB[mu, idx]  # [2,4]
            A = self._projA[mu, idx]  # [4,2]
            half = jnp.einsum("as,...sc->...ac", B, psi)
            half_u = self.color_mul_left(U, half)
            return jnp.einsum("sa,...ac->...sc", A, half_u)
        upsi = self.color_mul_left(U, psi)
        return self.spin_project(upsi, mu, coeff=coeff, use_sparse=use_sparse)

    def apply(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        sign: int = -1,
        use_sparse_gamma: bool = True,
        kernel: str = "optimized",
    ) -> Array:
        # D = diag + dslash, where dslash contains only nearest-neighbor hopping.
        return self.apply_diag(psi, sign=sign) + self.apply_dslash(
            U=U,
            psi=psi,
            take_mu=take_mu,
            roll_site=roll_site,
            dagger=dagger,
            sign=sign,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
        )

    def apply_dslash(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        sign: int = -1,
        use_sparse_gamma: bool = True,
        kernel: str = "optimized",
    ) -> Array:
        k = str(kernel).lower()
        if k == "reference":
            return self._apply_dslash_reference(
                U=U,
                psi=psi,
                take_mu=take_mu,
                roll_site=roll_site,
                dagger=dagger,
                sign=sign,
                use_sparse_gamma=use_sparse_gamma,
            )
        if k == "optimized":
            return self._apply_dslash_optimized(
                U=U,
                psi=psi,
                take_mu=take_mu,
                roll_site=roll_site,
                dagger=dagger,
                sign=sign,
                use_sparse_gamma=use_sparse_gamma,
            )
        raise ValueError("kernel must be one of: optimized, reference")

    def _apply_dslash_reference(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        sign: int = -1,
        use_sparse_gamma: bool = True,
    ) -> Array:
        # sign=-1 -> D, sign=+1 -> D^\dagger
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (D) or +1 (Ddag)")
        out = jnp.zeros_like(psi)
        for mu in range(self.Nd):
            U_mu = take_mu(U, mu)  # [...,Nc,Nc]
            psi_xpmu = roll_site(psi, -1, mu)
            psi_xmmu = roll_site(psi, +1, mu)

            fwd = self.color_mul_left(U_mu, psi_xpmu)
            U_mu_xmmu = roll_site(U_mu, +1, mu)
            bwd = self.color_mul_left(dagger(U_mu_xmmu), psi_xmmu)

            # Chroma convention:
            # D' psi = U (r - isign*gamma) psi(x+mu) + U^\dag(x-mu) (r + isign*gamma) psi(x-mu)
            fwd = self.spin_project(fwd, mu, coeff=+sign, use_sparse=use_sparse_gamma)
            bwd = self.spin_project(bwd, mu, coeff=-sign, use_sparse=use_sparse_gamma)
            out = out - 0.5 * (fwd + bwd)
        return out

    def _apply_dslash_optimized(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        sign: int = -1,
        use_sparse_gamma: bool = True,
    ) -> Array:
        # sign=-1 -> D, sign=+1 -> D^\dagger
        # Optimized equivalent of _apply_reference:
        # backward hopping term uses roll( U^\dagger(x) psi(x), +mu ),
        # which removes one lattice roll per direction.
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (D) or +1 (Ddag)")
        out = jnp.zeros_like(psi)
        for mu in range(self.Nd):
            U_mu = take_mu(U, mu)  # [...,Nc,Nc]

            psi_xpmu = roll_site(psi, -1, mu)
            fwd = self.projected_hop(U_mu, psi_xpmu, mu=mu, coeff=float(+sign), use_sparse=use_sparse_gamma)

            bwd_local = self.projected_hop(dagger(U_mu), psi, mu=mu, coeff=float(-sign), use_sparse=use_sparse_gamma)
            bwd = roll_site(bwd_local, +1, mu)
            out = out - 0.5 * (fwd + bwd)
        return out

    def apply_dagger(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        use_sparse_gamma: bool = True,
        kernel: str = "optimized",
    ) -> Array:
        return self.apply(
            U,
            psi,
            take_mu=take_mu,
            roll_site=roll_site,
            dagger=dagger,
            sign=+1,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
        )

    def apply_normal(
        self,
        U: Array,
        psi: Array,
        take_mu: Callable[[Array, int], Array],
        roll_site: Callable[[Array, int, int], Array],
        dagger: Callable[[Array], Array],
        use_sparse_gamma: bool = True,
        kernel: str = "optimized",
    ) -> Array:
        dpsi = self.apply(
            U,
            psi,
            take_mu=take_mu,
            roll_site=roll_site,
            dagger=dagger,
            sign=-1,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
        )
        return self.apply(
            U,
            dpsi,
            take_mu=take_mu,
            roll_site=roll_site,
            dagger=dagger,
            sign=+1,
            use_sparse_gamma=use_sparse_gamma,
            kernel=kernel,
        )

    def gamma_sparse_dense_error(self, psi: Array) -> Dict[str, float]:
        if not self._gamma_sparse_ok:
            return {"sparse_available": 0.0, "max_rel_error": float("nan")}
        errs = []
        for mu in range(self.Nd):
            gs = self.gamma_apply(psi, mu, use_sparse=True)
            gd = self.gamma_apply(psi, mu, use_sparse=False)
            rel = jnp.linalg.norm(gs - gd) / (jnp.linalg.norm(gd) + 1e-12)
            errs.append(float(rel))
        return {"sparse_available": 1.0, "max_rel_error": float(max(errs))}

    def flops_per_site_dslash(
        self,
        nc: int,
        use_sparse_gamma: bool = True,
        complex_mul_flops: int = 6,
        complex_add_flops: int = 2,
        real_complex_mul_flops: int = 2,
    ) -> int:
        """Estimate floating-point operations per site for Wilson dslash.

        This counts arithmetic implied by the current implementation:
        color left-multiply, spin projection, and dslash accumulation.
        Site rolls/shifts and memory traffic are not counted as FLOPs.
        """
        nc = int(nc)
        ns = int(self.Ns)
        nd = int(self.Nd)

        if nc < 1:
            raise ValueError("nc must be >= 1")

        # Complex Nc x Nc matrix times Nc complex vector, repeated for each spin row.
        # Per spin row: nc outputs, each output has nc complex mul and (nc-1) complex adds.
        color_mul = ns * nc * (complex_mul_flops * nc + complex_add_flops * (nc - 1))

        if bool(use_sparse_gamma):
            # One complex phase multiply per spin/color entry.
            gamma_apply = ns * nc * complex_mul_flops
        else:
            # Dense Ns x Ns gamma multiply per color block.
            gamma_apply = nc * ns * (complex_mul_flops * ns + complex_add_flops * (ns - 1))

        # spin_project = r*psi + coeff*gamma_apply(psi)
        # One real-complex multiply for r*psi, one for coeff*gamma, then one complex add.
        spin_project = gamma_apply + ns * nc * (2 * real_complex_mul_flops + complex_add_flops)

        # For each direction: forward + backward projected hops + out accumulation.
        # out += -0.5 * (fwd + bwd)
        out_accum = ns * nc * (2 * complex_add_flops + real_complex_mul_flops)
        per_dir = 2 * (color_mul + spin_project) + out_accum
        return int(nd * per_dir)

    def flops_per_site_matvec(
        self,
        nc: int,
        use_sparse_gamma: bool = True,
        include_diagonal: bool = True,
        complex_mul_flops: int = 6,
        complex_add_flops: int = 2,
        real_complex_mul_flops: int = 2,
    ) -> int:
        """Estimate floating-point operations per site for full Wilson D matvec."""
        dslash = self.flops_per_site_dslash(
            nc=nc,
            use_sparse_gamma=use_sparse_gamma,
            complex_mul_flops=complex_mul_flops,
            complex_add_flops=complex_add_flops,
            real_complex_mul_flops=real_complex_mul_flops,
        )
        if not bool(include_diagonal):
            return int(dslash)

        ns = int(self.Ns)
        nc = int(nc)
        # apply_diag (scalar*psi) + add to dslash output.
        diag = ns * nc * (real_complex_mul_flops + complex_add_flops)
        return int(dslash + diag)

    def flops_per_matvec_call(
        self,
        nc: int,
        volume: int,
        batch: int = 1,
        use_sparse_gamma: bool = True,
        include_diagonal: bool = True,
        complex_mul_flops: int = 6,
        complex_add_flops: int = 2,
        real_complex_mul_flops: int = 2,
    ) -> int:
        """Estimate total FLOPs for one Wilson D call over full lattice+batch."""
        per_site = self.flops_per_site_matvec(
            nc=nc,
            use_sparse_gamma=use_sparse_gamma,
            include_diagonal=include_diagonal,
            complex_mul_flops=complex_mul_flops,
            complex_add_flops=complex_add_flops,
            real_complex_mul_flops=real_complex_mul_flops,
        )
        return int(per_site * int(volume) * int(batch))

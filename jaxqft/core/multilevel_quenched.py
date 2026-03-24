"""Quenched projector-factorized multilevel helpers for DD pion measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxqft.core.domain_decomposition import TimeSlabDecomposition
from jaxqft.fermions import gamma5


Array = jax.Array


@dataclass(frozen=True)
class TwoLevelPionGeometry:
    """Overlapping two-domain geometry used by the phase-2 quenched estimator."""

    decomposition: TimeSlabDecomposition
    source_site: int
    source_domain_index: int
    sink_domain_index: int
    source_domain_sites: np.ndarray
    sink_domain_sites: np.ndarray
    overlap_sites: np.ndarray
    source_block_sites: np.ndarray
    sink_block_sites: np.ndarray
    source_site_local: int
    source_coords: Tuple[int, ...]
    source_block_lookup: Mapping[int, int]
    sink_block_lookup: Mapping[int, int]
    sink_dt: np.ndarray
    sink_momentum_coords: np.ndarray
    sink_times: np.ndarray


def _site_lookup(sites: Sequence[int]) -> Dict[int, int]:
    arr = np.asarray(sites, dtype=np.int64).reshape(-1)
    return {int(v): i for i, v in enumerate(arr.tolist())}


def _all_coords_for_shape(shape: Sequence[int]) -> np.ndarray:
    shp = tuple(int(v) for v in shape)
    vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
    return np.asarray(np.unravel_index(np.arange(vol, dtype=np.int64), shp), dtype=np.int64).T


def _coords_for_sites(shape: Sequence[int], sites: Sequence[int]) -> np.ndarray:
    coords = _all_coords_for_shape(shape)
    return np.asarray(coords[np.asarray(sites, dtype=np.int64)], dtype=np.int64)


def build_two_level_pion_geometry(
    *,
    lattice_shape: Sequence[int],
    boundary_slices: Sequence[int],
    boundary_width: int,
    source: Sequence[int],
    nsc: int,
    source_margin: int = 1,
    momentum_axis: int = 0,
) -> TwoLevelPionGeometry:
    decomp = TimeSlabDecomposition(
        lattice_shape=tuple(int(v) for v in lattice_shape),
        boundary_slices=tuple(int(v) for v in boundary_slices),
        boundary_width=int(boundary_width),
    )
    comps = decomp.interior_site_components()
    if len(comps) != 2:
        raise ValueError(
            "Two-level quenched pion estimator currently expects exactly two DD interior domains; "
            f"got {len(comps)}"
        )
    shp = tuple(int(v) for v in lattice_shape)
    source_coords = tuple(int(v % n) for v, n in zip(tuple(int(v) for v in source), shp))
    source_site = int(np.ravel_multi_index(source_coords, shp))
    if source_site in set(int(v) for v in np.asarray(decomp.boundary_site_indices).tolist()):
        raise ValueError(
            "The multilevel source must lie in the DD interior, not on a frozen boundary. "
            f"source={source_coords} boundary_times={tuple(int(v) for v in decomp.boundary_times)}"
        )

    src_dom = -1
    for i, comp in enumerate(comps):
        if source_site in set(int(v) for v in np.asarray(comp).tolist()):
            src_dom = int(i)
            break
    if src_dom < 0:
        raise ValueError(f"Could not assign source site {source_coords} to a DD interior component")
    sink_dom = 1 - src_dom

    time_comps = decomp.interior_time_components()
    src_times = tuple(int(v) for v in time_comps[src_dom])
    src_t = int(source_coords[-1])
    src_pos = src_times.index(src_t)
    if int(source_margin) > 0:
        left_gap = int(src_pos)
        right_gap = int(len(src_times) - 1 - src_pos)
        if min(left_gap, right_gap) < int(source_margin):
            raise ValueError(
                "The multilevel source must sit in the bulk of the unfrozen domain. "
                f"source_time={src_t} domain_times={src_times} source_margin={int(source_margin)}"
            )

    source_domain_sites = np.asarray(comps[src_dom], dtype=np.int64)
    sink_domain_sites = np.asarray(comps[sink_dom], dtype=np.int64)
    overlap_sites = np.asarray(decomp.boundary_site_indices, dtype=np.int64)
    source_block_sites = np.concatenate([source_domain_sites, overlap_sites], axis=0)
    sink_block_sites = np.concatenate([overlap_sites, sink_domain_sites], axis=0)
    source_block_lookup = _site_lookup(source_block_sites)
    sink_block_lookup = _site_lookup(sink_block_sites)
    source_site_local = int(source_block_lookup[int(source_site)])

    sink_coords = _coords_for_sites(shp, sink_domain_sites)
    sink_dt = np.asarray((sink_coords[:, -1] - int(source_coords[-1])) % int(shp[-1]), dtype=np.int64)
    mom_ax = int(momentum_axis)
    if len(shp) <= 1:
        sink_momentum_coords = np.zeros((sink_domain_sites.size,), dtype=np.int64)
    else:
        if mom_ax < 0:
            mom_ax += (len(shp) - 1)
        if mom_ax < 0 or mom_ax >= (len(shp) - 1):
            raise ValueError(
                f"momentum_axis must be in [0,{len(shp)-2}] for lattice_shape={shp}; got {momentum_axis!r}"
            )
        sink_momentum_coords = np.asarray(sink_coords[:, mom_ax], dtype=np.int64)

    return TwoLevelPionGeometry(
        decomposition=decomp,
        source_site=int(source_site),
        source_domain_index=int(src_dom),
        sink_domain_index=int(sink_dom),
        source_domain_sites=source_domain_sites,
        sink_domain_sites=sink_domain_sites,
        overlap_sites=overlap_sites,
        source_block_sites=source_block_sites,
        sink_block_sites=sink_block_sites,
        source_site_local=int(source_site_local),
        source_coords=source_coords,
        source_block_lookup=source_block_lookup,
        sink_block_lookup=sink_block_lookup,
        sink_dt=sink_dt,
        sink_momentum_coords=sink_momentum_coords,
        sink_times=np.asarray(sink_coords[:, -1], dtype=np.int64),
    )


def _component_indices(sites: Sequence[int], nsc: int) -> np.ndarray:
    arr = np.asarray(sites, dtype=np.int64).reshape(-1)
    return (arr[:, None] * int(nsc) + np.arange(int(nsc), dtype=np.int64)[None, :]).reshape(-1)


def _build_dense_submatrix(
    *,
    q,
    theory,
    row_sites: Sequence[int],
    col_sites: Sequence[int],
    nsc: int,
    max_dof: int,
) -> np.ndarray:
    row_idx = _component_indices(row_sites, nsc)
    col_idx = _component_indices(col_sites, nsc)
    nrow = int(row_idx.size)
    ncol = int(col_idx.size)
    if int(max_dof) > 0 and max(nrow, ncol) > int(max_dof):
        raise ValueError(
            f"Dense block build blocked: max(row_dof,col_dof)=({nrow},{ncol}) exceeds dense_max_domain_dof={int(max_dof)}"
        )
    fshape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    bs = int(fshape[0])
    ndof_full = int(np.prod(np.asarray(fshape[1:], dtype=np.int64)))
    fdtype = np.result_type(np.asarray(q).dtype, np.complex128)
    mat = np.zeros((bs, nrow, ncol), dtype=fdtype)
    for j, gcol in enumerate(col_idx.tolist()):
        rhs = np.zeros(fshape, dtype=fdtype)
        rhs.reshape(bs, ndof_full)[:, int(gcol)] = 1.0 + 0.0j
        col = np.asarray(theory.apply_D(q, jax.device_put(rhs)), dtype=fdtype).reshape(bs, ndof_full)
        mat[:, :, j] = col[:, row_idx]
    return mat


def _solve_block_matrix(block: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if block.ndim != 3 or rhs.ndim != 3:
        raise ValueError(f"Expected block/rhs ranks 3, got {block.shape} and {rhs.shape}")
    if block.shape[0] != rhs.shape[0] or block.shape[1] != block.shape[2] or block.shape[1] != rhs.shape[1]:
        raise ValueError(f"Incompatible solve shapes: block={block.shape} rhs={rhs.shape}")
    out = np.zeros_like(rhs)
    for b in range(int(block.shape[0])):
        out[b] = np.linalg.solve(block[b], rhs[b])
    return out


def _gamma5_matrix(theory) -> np.ndarray:
    gam = np.asarray(getattr(theory, "gamma"))
    g5 = np.asarray(gamma5(jnp.asarray(gam)))
    return np.asarray(g5)


def _apply_spin_matrix_to_local_vectors(vecs: np.ndarray, spin_mat: np.ndarray, *, ns: int, nc: int) -> np.ndarray:
    arr = np.asarray(vecs)
    ncfg, ndof, nrhs = arr.shape
    nsite = ndof // (int(ns) * int(nc))
    tmp = arr.reshape(ncfg, nsite, int(ns), int(nc), nrhs)
    out = np.einsum("uv,bsvcr->bsucr", np.asarray(spin_mat), tmp, optimize=True)
    return out.reshape(ncfg, ndof, nrhs)


def _laplace_matrix_on_sites(q_np: np.ndarray, theory, sites: np.ndarray) -> np.ndarray:
    shp = tuple(int(v) for v in tuple(theory.lattice_shape))
    nd = int(len(shp))
    if getattr(theory, "layout", "").upper() != "BMXYIJ":
        raise ValueError("laplace projector currently requires BMXYIJ gauge layout")
    lookup = _site_lookup(sites)
    coords = _coords_for_sites(shp, sites)
    mat = np.zeros((int(q_np.shape[0]), int(sites.size), int(sites.size)), dtype=np.complex128)
    for loc, coord in enumerate(coords.tolist()):
        x = tuple(int(v) for v in coord)
        for b in range(int(q_np.shape[0])):
            diag = 0.0
            for mu in range(nd):
                xf = list(x)
                xf[mu] = (xf[mu] + 1) % shp[mu]
                gf = int(np.ravel_multi_index(tuple(xf), shp))
                jf = lookup.get(gf, None)
                if jf is not None:
                    mat[b, loc, int(jf)] -= np.asarray(q_np[(b, mu, *x)], dtype=np.complex128)
                    diag += 1.0
                xb = list(x)
                xb[mu] = (xb[mu] - 1) % shp[mu]
                gb = int(np.ravel_multi_index(tuple(xb), shp))
                jb = lookup.get(gb, None)
                if jb is not None:
                    ub = np.conjugate(np.asarray(q_np[(b, mu, *tuple(xb))], dtype=np.complex128))
                    mat[b, loc, int(jb)] -= ub
                    diag += 1.0
            mat[b, loc, loc] += diag
    return mat


def build_projector_basis(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    ns: int,
    nc: int,
    kind: str,
    nvec: int,
    probe_stride: int,
) -> np.ndarray:
    q_np = np.asarray(q)
    bs = int(q_np.shape[0])
    overlap_sites = np.asarray(geometry.overlap_sites, dtype=np.int64)
    nsite = int(overlap_sites.size)
    nsc = int(ns * nc)
    support_dof = int(nsite * nsc)
    mode = str(kind).strip().lower()
    if mode in ("full", "identity", "canonical"):
        eye = np.eye(support_dof, dtype=np.complex128)
        return np.broadcast_to(eye[None, :, :], (bs, support_dof, support_dof)).copy()
    if mode in ("probe", "probing"):
        stride = max(1, int(probe_stride))
        chosen_sites = np.arange(0, nsite, stride, dtype=np.int64)
        nprobe = int(chosen_sites.size * nsc)
        out = np.zeros((bs, nprobe, support_dof), dtype=np.complex128)
        ct = 0
        for ls in chosen_sites.tolist():
            base = int(ls) * nsc
            for sc in range(nsc):
                out[:, ct, base + sc] = 1.0
                ct += 1
        return out
    if mode in ("laplace", "distillation", "distill"):
        nkeep = int(nvec)
        if nkeep <= 0:
            raise ValueError("laplace/distillation projector requires projector_nvec > 0")
        lap = _laplace_matrix_on_sites(q_np, theory, overlap_sites)
        nkeep = min(int(nkeep), int(nsite))
        out = np.zeros((bs, nkeep * nsc, support_dof), dtype=np.complex128)
        for b in range(bs):
            evals, evecs = np.linalg.eigh(lap[b])
            order = np.argsort(np.asarray(evals).real)
            kept = np.asarray(evecs[:, order[:nkeep]], dtype=np.complex128)
            ct = 0
            for k in range(nkeep):
                vec = kept[:, k]
                for sc in range(nsc):
                    tmp = np.zeros((nsite, nsc), dtype=np.complex128)
                    tmp[:, sc] = vec
                    out[b, ct, :] = tmp.reshape(-1)
                    ct += 1
        return out
    raise ValueError(f"Unsupported projector_kind: {kind!r}")


def _embed_overlap_projectors_into_block(
    phi_overlap: np.ndarray,
    *,
    block_sites: Sequence[int],
    overlap_sites: Sequence[int],
    nsc: int,
) -> np.ndarray:
    bs = int(phi_overlap.shape[0])
    nproj = int(phi_overlap.shape[1])
    block_sites_arr = np.asarray(block_sites, dtype=np.int64)
    overlap_sites_arr = np.asarray(overlap_sites, dtype=np.int64)
    block_lookup = _site_lookup(block_sites_arr)
    block_dof = int(block_sites_arr.size * int(nsc))
    out = np.zeros((bs, block_dof, nproj), dtype=np.complex128)
    for i, site in enumerate(overlap_sites_arr.tolist()):
        block_loc = int(block_lookup[int(site)])
        src = phi_overlap[:, :, i * int(nsc) : (i + 1) * int(nsc)]
        out[:, block_loc * int(nsc) : (block_loc + 1) * int(nsc), :] = np.swapaxes(src, 1, 2)
    return out


def compute_factorized_pion_blocks(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    phi_overlap: np.ndarray,
    dense_max_domain_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lattice_shape = tuple(int(v) for v in tuple(theory.lattice_shape))
    _ = lattice_shape
    fshape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    ns = int(fshape[-2])
    nc = int(fshape[-1])
    nsc = int(ns * nc)
    g5 = _gamma5_matrix(theory)

    d_src = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.source_block_sites,
        col_sites=geometry.source_block_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    d_sink = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.sink_block_sites,
        col_sites=geometry.sink_block_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    phi_src = _embed_overlap_projectors_into_block(
        phi_overlap,
        block_sites=geometry.source_block_sites,
        overlap_sites=geometry.overlap_sites,
        nsc=nsc,
    )
    rhs_src = _apply_spin_matrix_to_local_vectors(phi_src, g5, ns=ns, nc=nc)
    sol_src = _solve_block_matrix(d_src, rhs_src)

    src_loc = int(geometry.source_site_local)
    src_spinors = np.swapaxes(
        sol_src[:, src_loc * nsc : (src_loc + 1) * nsc, :].reshape(int(sol_src.shape[0]), nsc, int(sol_src.shape[2])),
        1,
        2,
    )
    source_blocks = np.einsum("bia,bja->bij", np.conjugate(src_spinors), src_spinors, optimize=True)

    k_sink = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.sink_block_sites,
        col_sites=geometry.overlap_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    # Keep only the source-to-sink boundary couplings: rows on the overlap itself do not belong to D_{∂Γ*}.
    nov = int(geometry.overlap_sites.size)
    k_sink[:, : nov * nsc, :] = 0.0
    rhs_sink = np.einsum("bij,bjk->bik", k_sink, np.swapaxes(phi_overlap, 1, 2), optimize=True)
    sol_sink = _solve_block_matrix(d_sink, rhs_sink)

    sink_spins = []
    for site in np.asarray(geometry.sink_domain_sites, dtype=np.int64).tolist():
        loc = int(geometry.sink_block_lookup[int(site)])
        blk = sol_sink[:, loc * nsc : (loc + 1) * nsc, :]
        sink_spins.append(np.swapaxes(blk.reshape(int(sol_sink.shape[0]), nsc, int(sol_sink.shape[2])), 1, 2))
    sink_spinors = np.stack(sink_spins, axis=1)
    sink_blocks = np.einsum("bsia,bsja->bsij", np.conjugate(sink_spinors), sink_spinors, optimize=True)
    return source_blocks, sink_blocks


def factorized_pion_corr_from_blocks(
    *,
    source_blocks: np.ndarray,
    sink_blocks: np.ndarray,
    geometry: TwoLevelPionGeometry,
    momenta: Sequence[int],
    average_pm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    moms = tuple(int(v) for v in momenta)
    if len(moms) == 0:
        moms = (0,)
    bs = int(source_blocks.shape[0])
    lt = int(geometry.decomposition.lt)
    corr = np.zeros((bs, len(moms), lt), dtype=np.complex128)
    valid_mask = np.zeros((lt,), dtype=bool)
    if len(tuple(geometry.decomposition.lattice_shape)) <= 1:
        lmom = 1
        x0 = 0
    else:
        lmom = int(geometry.decomposition.lattice_shape[0])
        x0 = int(geometry.source_coords[0]) % lmom
    wick = np.einsum("bsij,bji->bs", sink_blocks, np.swapaxes(source_blocks, 1, 2), optimize=True)
    for s in range(int(geometry.sink_domain_sites.size)):
        dt = int(geometry.sink_dt[s])
        valid_mask[dt] = True
        x = int(geometry.sink_momentum_coords[s]) % max(lmom, 1)
        for ip, p in enumerate(moms):
            theta = (2.0 * np.pi * float(p) / float(max(lmom, 1))) * (float(x) - float(x0))
            phase = np.cos(theta) if bool(average_pm) else np.exp(1j * theta)
            corr[:, ip, dt] = corr[:, ip, dt] + phase * wick[:, s]
    return corr, valid_mask


def split_domain_masks(
    geometry: TwoLevelPionGeometry,
    *,
    batch_size: int,
    layout: str = "BMXYIJ",
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    shp = tuple(int(v) for v in tuple(geometry.decomposition.lattice_shape))
    nd = int(len(shp))
    source_t_mask = np.zeros((shp[-1],), dtype=bool)
    sink_t_mask = np.zeros((shp[-1],), dtype=bool)
    source_t_mask[np.unique(_coords_for_sites(shp, geometry.source_domain_sites)[:, -1])] = True
    sink_t_mask[np.unique(_coords_for_sites(shp, geometry.sink_domain_sites)[:, -1])] = True

    def _site_mask(mask_t: np.ndarray) -> np.ndarray:
        reshape = (1,) * (len(shp) - 1) + (shp[-1],)
        return np.broadcast_to(mask_t.reshape(reshape), shp).copy()

    def _link_mask(site_mask: np.ndarray) -> np.ndarray:
        by_mu = []
        for mu in range(nd):
            act = np.array(site_mask, copy=True)
            if mu == nd - 1:
                act = np.logical_and(act, np.roll(site_mask, shift=-1, axis=nd - 1))
            by_mu.append(act)
        lay = str(layout).upper()
        if lay in ("BM...IJ",):
            lay = "BMXYIJ"
        if lay != "BMXYIJ":
            raise ValueError(f"Unsupported layout for split_domain_masks: {layout!r}")
        arr_src = np.stack(by_mu, axis=0)[None, ...]
        return np.tile(arr_src, [int(batch_size)] + [1] * (arr_src.ndim - 1)).astype(dtype, copy=False)

    return _link_mask(_site_mask(source_t_mask)), _link_mask(_site_mask(sink_t_mask))

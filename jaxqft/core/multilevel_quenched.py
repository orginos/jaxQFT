"""Quenched projector-factorized multilevel helpers for DD pion measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import jax
import numpy as np

from jaxqft.core.domain_decomposition import TimeSlabDecomposition


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
    boundary_slab_sites: Tuple[np.ndarray, ...]
    source_surface_sites: np.ndarray
    sink_surface_sites: np.ndarray
    source_surface_slab_sites: Tuple[np.ndarray, ...]
    sink_surface_slab_sites: Tuple[np.ndarray, ...]
    source_site_local: int
    source_coords: Tuple[int, ...]
    source_domain_lookup: Mapping[int, int]
    overlap_lookup: Mapping[int, int]
    source_surface_lookup: Mapping[int, int]
    sink_surface_lookup: Mapping[int, int]
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


def _boundary_slab_site_groups(decomp: TimeSlabDecomposition) -> Tuple[np.ndarray, ...]:
    shp = tuple(int(v) for v in tuple(decomp.lattice_shape))
    coords = _coords_for_sites(shp, decomp.boundary_site_indices)
    groups = []
    for start in tuple(int(v) for v in decomp.boundary_slices):
        slab_times = np.asarray(
            [int((start + off) % int(decomp.lt)) for off in range(int(decomp.boundary_width))],
            dtype=np.int64,
        )
        mask = np.isin(coords[:, -1], slab_times)
        groups.append(np.asarray(decomp.boundary_site_indices[mask], dtype=np.int64))
    return tuple(groups)


def _boundary_surface_site_groups_for_domain(
    decomp: TimeSlabDecomposition,
    *,
    domain_times: Sequence[int],
) -> Tuple[np.ndarray, ...]:
    shp = tuple(int(v) for v in tuple(decomp.lattice_shape))
    coords = _coords_for_sites(shp, decomp.boundary_site_indices)
    groups = []
    dom_t = set(int(v) % int(decomp.lt) for v in tuple(domain_times))
    for start in tuple(int(v) for v in decomp.boundary_slices):
        face_times = []
        t_prev = int((start - 1) % int(decomp.lt))
        t_next = int((start + int(decomp.boundary_width)) % int(decomp.lt))
        if t_prev in dom_t:
            face_times.append(int(start))
        if t_next in dom_t:
            face_times.append(int((start + int(decomp.boundary_width) - 1) % int(decomp.lt)))
        if len(face_times) == 0:
            groups.append(np.zeros((0,), dtype=np.int64))
            continue
        mask = np.isin(coords[:, -1], np.asarray(np.unique(np.asarray(face_times, dtype=np.int64)), dtype=np.int64))
        groups.append(np.asarray(decomp.boundary_site_indices[mask], dtype=np.int64))
    return tuple(groups)


def _concat_site_groups(groups: Sequence[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(g, dtype=np.int64).reshape(-1) for g in tuple(groups) if int(np.asarray(g).size) > 0]
    if len(arrs) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(arrs, axis=0).astype(np.int64, copy=False)


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
    sink_times = tuple(int(v) for v in time_comps[sink_dom])
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
    boundary_slab_sites = _boundary_slab_site_groups(decomp)
    source_surface_slab_sites = _boundary_surface_site_groups_for_domain(decomp, domain_times=src_times)
    sink_surface_slab_sites = _boundary_surface_site_groups_for_domain(decomp, domain_times=sink_times)
    source_surface_sites = _concat_site_groups(source_surface_slab_sites)
    sink_surface_sites = _concat_site_groups(sink_surface_slab_sites)
    source_domain_lookup = _site_lookup(source_domain_sites)
    overlap_lookup = _site_lookup(overlap_sites)
    source_surface_lookup = _site_lookup(source_surface_sites)
    sink_surface_lookup = _site_lookup(sink_surface_sites)
    source_site_local = int(source_domain_lookup[int(source_site)])

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
        boundary_slab_sites=boundary_slab_sites,
        source_surface_sites=source_surface_sites,
        sink_surface_sites=sink_surface_sites,
        source_surface_slab_sites=source_surface_slab_sites,
        sink_surface_slab_sites=sink_surface_slab_sites,
        source_site_local=int(source_site_local),
        source_coords=source_coords,
        source_domain_lookup=source_domain_lookup,
        overlap_lookup=overlap_lookup,
        source_surface_lookup=source_surface_lookup,
        sink_surface_lookup=sink_surface_lookup,
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


def _embed_site_spin_vectors_into_support(
    vecs: np.ndarray,
    *,
    support_sites: Sequence[int],
    support_lookup: Mapping[int, int],
    nsc: int,
    support_dof: int,
) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 projector array, got {arr.shape}")
    bs = int(arr.shape[0])
    if int(arr.shape[2]) != int(len(tuple(support_sites))) * int(nsc):
        raise ValueError(
            f"Support projector mismatch: got {arr.shape[2]} dof for {len(tuple(support_sites))} support sites and nsc={nsc}"
        )
    out = np.zeros((bs, int(arr.shape[1]), int(support_dof)), dtype=np.complex128)
    for i, site in enumerate(np.asarray(support_sites, dtype=np.int64).tolist()):
        overlap_loc = int(support_lookup[int(site)])
        src = arr[:, :, i * int(nsc) : (i + 1) * int(nsc)]
        out[:, :, overlap_loc * int(nsc) : (overlap_loc + 1) * int(nsc)] = src
    return out


def _embed_site_spin_vectors_into_overlap(
    vecs: np.ndarray,
    *,
    slab_sites: Sequence[int],
    overlap_lookup: Mapping[int, int],
    nsc: int,
    support_dof: int,
) -> np.ndarray:
    return _embed_site_spin_vectors_into_support(
        vecs,
        support_sites=slab_sites,
        support_lookup=overlap_lookup,
        nsc=nsc,
        support_dof=support_dof,
    )


def _distribute_projector_vectors(nkeep_total: int, groups: Sequence[np.ndarray], *, label: str) -> np.ndarray:
    ng = int(len(tuple(groups)))
    if ng <= 0:
        raise ValueError(f"{label} projector requires at least one boundary slab")
    if int(nkeep_total) <= 0:
        raise ValueError(f"{label} projector requires projector_nvec > 0")
    if int(nkeep_total) % ng != 0:
        raise ValueError(
            f"{label} projector requires projector_nvec divisible by the number of slabs; "
            f"got projector_nvec={int(nkeep_total)} with {ng} slabs"
        )
    return np.full((ng,), int(nkeep_total) // ng, dtype=np.int64)


def _build_site_projector_basis(
    *,
    q,
    theory,
    support_sites: Sequence[int],
    support_groups: Sequence[np.ndarray],
    nsc: int,
    kind: str,
    nvec: int,
    probe_stride: int,
    allow_svd: bool,
    probe_by_group: bool,
) -> np.ndarray:
    q_np = np.asarray(q)
    bs = int(q_np.shape[0])
    support_sites_a = np.asarray(support_sites, dtype=np.int64).reshape(-1)
    nsite = int(support_sites_a.size)
    support_dof = int(nsite * nsc)
    mode = str(kind).strip().lower()
    if support_dof <= 0:
        raise ValueError("Projector support is empty")
    if mode in ("full", "identity", "canonical"):
        eye = np.eye(support_dof, dtype=np.complex128)
        return np.broadcast_to(eye[None, :, :], (bs, support_dof, support_dof)).copy()
    if mode in ("probe", "probing"):
        stride = max(1, int(probe_stride))
        if not bool(probe_by_group):
            chosen_sites = np.arange(0, nsite, stride, dtype=np.int64)
            nprobe = int(chosen_sites.size * nsc)
            out = np.zeros((bs, nprobe, support_dof), dtype=np.complex128)
            ct = 0
            for ls in chosen_sites.tolist():
                base = int(ls) * int(nsc)
                for sc in range(int(nsc)):
                    out[:, ct, base + sc] = 1.0
                    ct += 1
            return out
        out_rows = []
        lookup = _site_lookup(support_sites_a)
        for grp in tuple(support_groups):
            grp_a = np.asarray(grp, dtype=np.int64).reshape(-1)
            if int(grp_a.size) == 0:
                continue
            chosen_sites = grp_a[np.arange(0, int(grp_a.size), stride, dtype=np.int64)]
            rows = np.zeros((int(chosen_sites.size) * int(nsc), support_dof), dtype=np.complex128)
            ct = 0
            for site in chosen_sites.tolist():
                loc = int(lookup[int(site)])
                base = int(loc) * int(nsc)
                for sc in range(int(nsc)):
                    rows[ct, base + sc] = 1.0
                    ct += 1
            out_rows.append(rows)
        if len(out_rows) == 0:
            raise ValueError("Probe projector produced no support vectors")
        rows = np.concatenate(out_rows, axis=0)
        return np.broadcast_to(rows[None, :, :], (bs, rows.shape[0], rows.shape[1])).copy()
    if mode in ("laplace", "distillation", "distill"):
        slab_counts = _distribute_projector_vectors(int(nvec), support_groups, label="laplace/distillation")
        out = np.zeros((bs, int(np.sum(slab_counts)) * nsc, support_dof), dtype=np.complex128)
        lookup = _site_lookup(support_sites_a)
        ct0 = 0
        for grp, nkeep_grp in zip(tuple(support_groups), slab_counts.tolist()):
            grp_a = np.asarray(grp, dtype=np.int64).reshape(-1)
            lap = _laplace_matrix_on_sites(q_np, theory, grp_a)
            if int(nkeep_grp) > int(grp_a.size):
                raise ValueError(
                    f"laplace/distillation projector requested {int(nkeep_grp)} site vectors on a support group "
                    f"with {int(grp_a.size)} sites"
                )
            kept_rows = np.zeros((bs, int(nkeep_grp) * nsc, support_dof), dtype=np.complex128)
            for b in range(bs):
                evals, evecs = np.linalg.eigh(lap[b])
                order = np.argsort(np.asarray(evals).real)
                kept = np.asarray(evecs[:, order[: int(nkeep_grp)]], dtype=np.complex128)
                grp_rows = np.zeros((int(nkeep_grp) * nsc, int(grp_a.size) * nsc), dtype=np.complex128)
                ct = 0
                for k in range(int(nkeep_grp)):
                    vec = kept[:, k]
                    for sc in range(int(nsc)):
                        tmp = np.zeros((int(grp_a.size), int(nsc)), dtype=np.complex128)
                        tmp[:, sc] = vec
                        grp_rows[ct, :] = tmp.reshape(-1)
                        ct += 1
                kept_rows[b] = _embed_site_spin_vectors_into_support(
                    grp_rows[None, ...],
                    support_sites=grp_a,
                    support_lookup=lookup,
                    nsc=nsc,
                    support_dof=support_dof,
                )[0]
            out[:, ct0 : ct0 + int(nkeep_grp) * nsc, :] = kept_rows
            ct0 += int(nkeep_grp) * nsc
        return out
    if mode in ("svd", "singular", "svd_slab", "singular_slab"):
        if not bool(allow_svd):
            raise ValueError("svd/singular projectors are not supported for this factorization")
        slab_counts = _distribute_projector_vectors(int(nvec), support_groups, label="svd/singular")
        out = np.zeros((bs, int(np.sum(slab_counts)), support_dof), dtype=np.complex128)
        lookup = _site_lookup(support_sites_a)
        ct0 = 0
        for grp, nkeep_grp in zip(tuple(support_groups), slab_counts.tolist()):
            grp_a = np.asarray(grp, dtype=np.int64).reshape(-1)
            d_grp = _build_dense_submatrix(
                q=q,
                theory=theory,
                row_sites=grp_a,
                col_sites=grp_a,
                nsc=nsc,
                max_dof=int(support_dof),
            )
            if int(nkeep_grp) > int(d_grp.shape[1]):
                raise ValueError(
                    f"svd/singular projector requested {int(nkeep_grp)} vectors on a support group with only "
                    f"{int(d_grp.shape[1])} fermion dof"
                )
            kept_rows = np.zeros((bs, int(nkeep_grp), support_dof), dtype=np.complex128)
            for b in range(bs):
                gram = np.asarray(d_grp[b]).conjugate().T @ np.asarray(d_grp[b])
                evals, evecs = np.linalg.eigh(gram)
                order = np.argsort(np.asarray(evals).real)
                kept = np.asarray(evecs[:, order[: int(nkeep_grp)]], dtype=np.complex128).T
                kept_rows[b] = _embed_site_spin_vectors_into_support(
                    kept[None, ...],
                    support_sites=grp_a,
                    support_lookup=lookup,
                    nsc=nsc,
                    support_dof=support_dof,
                )[0]
            out[:, ct0 : ct0 + int(nkeep_grp), :] = kept_rows
            ct0 += int(nkeep_grp)
        return out
    raise ValueError(f"Unsupported projector_kind: {kind!r}")


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
    return _build_site_projector_basis(
        q=q,
        theory=theory,
        support_sites=np.asarray(geometry.overlap_sites, dtype=np.int64),
        support_groups=geometry.boundary_slab_sites,
        nsc=int(ns * nc),
        kind=kind,
        nvec=int(nvec),
        probe_stride=int(probe_stride),
        allow_svd=True,
        probe_by_group=False,
    )


def build_giusti_surface_projector_basis(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    ns: int,
    nc: int,
    kind: str,
    nvec: int,
    probe_stride: int,
    dressed_domain: str,
) -> np.ndarray:
    mode = str(dressed_domain).strip().lower()
    if mode in ("source", "left", "source_dressed", "a"):
        support_sites = geometry.sink_surface_sites
        support_groups = geometry.sink_surface_slab_sites
    elif mode in ("sink", "right", "sink_dressed", "c"):
        support_sites = geometry.source_surface_sites
        support_groups = geometry.source_surface_slab_sites
    else:
        raise ValueError(f"Unsupported dressed_domain for Giusti projector: {dressed_domain!r}")
    return _build_site_projector_basis(
        q=q,
        theory=theory,
        support_sites=support_sites,
        support_groups=support_groups,
        nsc=int(ns * nc),
        kind=kind,
        nvec=int(nvec),
        probe_stride=int(probe_stride),
        allow_svd=False,
        probe_by_group=True,
    )


def _local_component_indices_for_block(block_sites: Sequence[int], select_sites: Sequence[int], nsc: int) -> np.ndarray:
    lookup = _site_lookup(block_sites)
    select_local = np.asarray([int(lookup[int(site)]) for site in np.asarray(select_sites, dtype=np.int64).tolist()], dtype=np.int64)
    return _component_indices(select_local, nsc)


def _combine_block_sites(*parts: Sequence[int]) -> np.ndarray:
    arrs = [np.asarray(p, dtype=np.int64).reshape(-1) for p in parts if int(np.asarray(p).size) > 0]
    if len(arrs) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(arrs, axis=0).astype(np.int64, copy=False)


def _restrict_block_solution_to_sites(sol: np.ndarray, *, block_sites: Sequence[int], select_sites: Sequence[int], nsc: int) -> np.ndarray:
    idx = _local_component_indices_for_block(block_sites, select_sites, nsc)
    return np.asarray(sol[:, idx, :], dtype=np.complex128)


def _embed_support_columns_into_block(
    phi_cols: np.ndarray,
    *,
    support_sites: Sequence[int],
    block_sites: Sequence[int],
    nsc: int,
) -> np.ndarray:
    arr = np.asarray(phi_cols, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 embedded support columns, got {arr.shape}")
    bs = int(arr.shape[0])
    nproj = int(arr.shape[2])
    block_sites_a = np.asarray(block_sites, dtype=np.int64).reshape(-1)
    block_dof = int(block_sites_a.size) * int(nsc)
    out = np.zeros((bs, block_dof, nproj), dtype=np.complex128)
    lookup = _site_lookup(block_sites_a)
    for i, site in enumerate(np.asarray(support_sites, dtype=np.int64).tolist()):
        loc = int(lookup[int(site)])
        out[:, loc * int(nsc) : (loc + 1) * int(nsc), :] = arr[:, i * int(nsc) : (i + 1) * int(nsc), :]
    return out


def compute_factorized_pion_factors(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    phi_overlap: np.ndarray,
    dense_max_domain_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    fshape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    ns = int(fshape[-2])
    nc = int(fshape[-1])
    nsc = int(ns * nc)
    phi_rows = np.asarray(phi_overlap, dtype=np.complex128)
    phi_cols = np.swapaxes(phi_rows, 1, 2)
    nproj = int(phi_rows.shape[1])

    d_sigma_sigma = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.overlap_sites,
        col_sites=geometry.overlap_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    m_rhs = np.einsum("bij,bjk->bik", d_sigma_sigma, phi_cols, optimize=True)
    transfer = np.einsum("bij,bjk->bik", np.conjugate(phi_rows), m_rhs, optimize=True)

    d_aa = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.source_domain_sites,
        col_sites=geometry.source_domain_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    d_sigma_a = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.overlap_sites,
        col_sites=geometry.source_domain_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    rhs_src = np.zeros((int(np.asarray(q).shape[0]), d_aa.shape[1], nsc), dtype=np.complex128)
    src_loc = int(geometry.source_site_local)
    rhs_src[:, src_loc * nsc : (src_loc + 1) * nsc, :] = np.broadcast_to(
        np.eye(nsc, dtype=np.complex128)[None, :, :],
        (int(np.asarray(q).shape[0]), nsc, nsc),
    )
    sol_src = _solve_block_matrix(d_aa, rhs_src)
    boundary_src = np.einsum("bij,bjk->bik", d_sigma_a, sol_src, optimize=True)
    projected_src = np.einsum("bij,bjk->bik", np.conjugate(phi_rows), boundary_src, optimize=True)
    source_factors = _solve_block_matrix(transfer, projected_src)

    d_cc = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.sink_domain_sites,
        col_sites=geometry.sink_domain_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    d_c_sigma = _build_dense_submatrix(
        q=q,
        theory=theory,
        row_sites=geometry.sink_domain_sites,
        col_sites=geometry.overlap_sites,
        nsc=nsc,
        max_dof=int(dense_max_domain_dof),
    )
    rhs_sink = np.einsum("bij,bjk->bik", d_c_sigma, phi_cols, optimize=True)
    sol_sink = _solve_block_matrix(d_cc, rhs_sink)
    sink_factors = sol_sink.reshape(int(sol_sink.shape[0]), int(geometry.sink_domain_sites.size), nsc, nproj)
    return source_factors, sink_factors


def compute_factorized_pion_blocks(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    phi_overlap: np.ndarray,
    dense_max_domain_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source_factors, sink_factors = compute_factorized_pion_factors(
        q=q,
        theory=theory,
        geometry=geometry,
        phi_overlap=phi_overlap,
        dense_max_domain_dof=int(dense_max_domain_dof),
    )
    source_blocks = np.einsum("bia,bja->bij", source_factors, np.conjugate(source_factors), optimize=True)
    sink_blocks = np.einsum("bsap,bsaq->bspq", np.conjugate(sink_factors), sink_factors, optimize=True)
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
    # pair(y,x) = tr[(L_y S_x) (L_y S_x)^\dagger] = tr[source_blocks @ sink_blocks_y]
    # with source_blocks[p,q] = (S S^\dagger)_{p,q} and sink_blocks[s,p,q] = (L^\dagger L)_{p,q},
    # so the scalar contraction needs the sink transpose: sum_{p,q} source[p,q] sink[q,p].
    wick = np.einsum("bsij,bji->bs", sink_blocks, source_blocks, optimize=True)
    for s in range(int(geometry.sink_domain_sites.size)):
        dt = int(geometry.sink_dt[s])
        valid_mask[dt] = True
        x = int(geometry.sink_momentum_coords[s]) % max(lmom, 1)
        for ip, p in enumerate(moms):
            theta = (2.0 * np.pi * float(p) / float(max(lmom, 1))) * (float(x) - float(x0))
            phase = np.cos(theta) if bool(average_pm) else np.exp(1j * theta)
            corr[:, ip, dt] = corr[:, ip, dt] + phase * wick[:, s]
    return corr, valid_mask


def compute_giusti_asymmetric_pion_factors(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    phi_surface: np.ndarray,
    dense_max_domain_dof: int,
    dressed_domain: str,
) -> Tuple[np.ndarray, np.ndarray]:
    fshape = tuple(int(v) for v in tuple(theory.fermion_shape()))
    ns = int(fshape[-2])
    nc = int(fshape[-1])
    nsc = int(ns * nc)
    phi_rows = np.asarray(phi_surface, dtype=np.complex128)
    phi_cols = np.swapaxes(phi_rows, 1, 2)
    mode = str(dressed_domain).strip().lower()

    if mode in ("source", "left", "source_dressed", "a"):
        support_sites = np.asarray(geometry.sink_surface_sites, dtype=np.int64)
        block_sites = _combine_block_sites(geometry.source_domain_sites, geometry.overlap_sites)
        d_block = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=block_sites,
            col_sites=block_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        src_lookup = _site_lookup(block_sites)
        rhs_src = np.zeros((int(np.asarray(q).shape[0]), d_block.shape[1], nsc), dtype=np.complex128)
        src_loc = int(src_lookup[int(geometry.source_site)])
        rhs_src[:, src_loc * nsc : (src_loc + 1) * nsc, :] = np.broadcast_to(
            np.eye(nsc, dtype=np.complex128)[None, :, :],
            (int(np.asarray(q).shape[0]), nsc, nsc),
        )
        sol_src = _solve_block_matrix(d_block, rhs_src)
        support_src = _restrict_block_solution_to_sites(
            sol_src,
            block_sites=block_sites,
            select_sites=support_sites,
            nsc=nsc,
        )
        source_factors = np.einsum("bij,bjk->bik", np.conjugate(phi_rows), support_src, optimize=True)

        d_cc = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=geometry.sink_domain_sites,
            col_sites=geometry.sink_domain_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        d_c_support = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=geometry.sink_domain_sites,
            col_sites=support_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        rhs_sink = np.einsum("bij,bjk->bik", d_c_support, phi_cols, optimize=True)
        sol_sink = _solve_block_matrix(d_cc, rhs_sink)
        sink_factors = sol_sink.reshape(int(sol_sink.shape[0]), int(geometry.sink_domain_sites.size), nsc, int(phi_rows.shape[1]))
        return source_factors, sink_factors

    if mode in ("sink", "right", "sink_dressed", "c"):
        support_sites = np.asarray(geometry.source_surface_sites, dtype=np.int64)
        d_aa = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=geometry.source_domain_sites,
            col_sites=geometry.source_domain_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        d_support_a = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=support_sites,
            col_sites=geometry.source_domain_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        rhs_src = np.zeros((int(np.asarray(q).shape[0]), d_aa.shape[1], nsc), dtype=np.complex128)
        src_loc = int(geometry.source_site_local)
        rhs_src[:, src_loc * nsc : (src_loc + 1) * nsc, :] = np.broadcast_to(
            np.eye(nsc, dtype=np.complex128)[None, :, :],
            (int(np.asarray(q).shape[0]), nsc, nsc),
        )
        sol_src = _solve_block_matrix(d_aa, rhs_src)
        boundary_src = np.einsum("bij,bjk->bik", d_support_a, sol_src, optimize=True)
        source_factors = np.einsum("bij,bjk->bik", np.conjugate(phi_rows), boundary_src, optimize=True)

        block_sites = _combine_block_sites(geometry.overlap_sites, geometry.sink_domain_sites)
        d_block = _build_dense_submatrix(
            q=q,
            theory=theory,
            row_sites=block_sites,
            col_sites=block_sites,
            nsc=nsc,
            max_dof=int(dense_max_domain_dof),
        )
        rhs_block = _embed_support_columns_into_block(
            phi_cols,
            support_sites=support_sites,
            block_sites=block_sites,
            nsc=nsc,
        )
        sol_block = _solve_block_matrix(d_block, rhs_block)
        sink_rows = _restrict_block_solution_to_sites(
            sol_block,
            block_sites=block_sites,
            select_sites=geometry.sink_domain_sites,
            nsc=nsc,
        )
        sink_factors = sink_rows.reshape(int(sol_block.shape[0]), int(geometry.sink_domain_sites.size), nsc, int(phi_rows.shape[1]))
        return source_factors, sink_factors

    raise ValueError(f"Unsupported Giusti dressed_domain: {dressed_domain!r}")


def compute_giusti_asymmetric_pion_blocks(
    *,
    q,
    theory,
    geometry: TwoLevelPionGeometry,
    phi_surface: np.ndarray,
    dense_max_domain_dof: int,
    dressed_domain: str,
) -> Tuple[np.ndarray, np.ndarray]:
    source_factors, sink_factors = compute_giusti_asymmetric_pion_factors(
        q=q,
        theory=theory,
        geometry=geometry,
        phi_surface=phi_surface,
        dense_max_domain_dof=int(dense_max_domain_dof),
        dressed_domain=dressed_domain,
    )
    source_blocks = np.einsum("bia,bja->bij", source_factors, np.conjugate(source_factors), optimize=True)
    sink_blocks = np.einsum("bsap,bsaq->bspq", np.conjugate(sink_factors), sink_factors, optimize=True)
    return source_blocks, sink_blocks


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

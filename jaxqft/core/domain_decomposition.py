"""Static domain-decomposition helpers shared across measurements and updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass
class TimeSlabDecomposition:
    """Time-slab domain decomposition with the time axis fixed to the last lattice axis."""

    lattice_shape: Tuple[int, ...]
    boundary_slices: Tuple[int, ...]
    boundary_width: int = 1
    boundary_times: np.ndarray = field(init=False, repr=False)
    interior_times: np.ndarray = field(init=False, repr=False)
    boundary_site_indices: np.ndarray = field(init=False, repr=False)
    interior_site_indices: np.ndarray = field(init=False, repr=False)
    interior_lookup: np.ndarray = field(init=False, repr=False)
    vol: int = field(init=False)
    lt: int = field(init=False)

    def __post_init__(self) -> None:
        shp = tuple(int(v) for v in tuple(self.lattice_shape))
        if not shp:
            raise ValueError("TimeSlabDecomposition requires a non-empty lattice shape")

        bw = int(self.boundary_width)
        if bw <= 0:
            raise ValueError(f"boundary_width must be positive, got {self.boundary_width!r}")

        lt = int(shp[-1])
        starts = tuple(int(v) % lt for v in tuple(self.boundary_slices))
        if len(starts) == 0:
            raise ValueError("TimeSlabDecomposition requires at least one boundary slice")

        boundary_mask = np.zeros((lt,), dtype=bool)
        for s in starts:
            for off in range(bw):
                t = int((s + off) % lt)
                if boundary_mask[t]:
                    raise ValueError(
                        "Boundary slabs overlap in time. "
                        f"boundary_slices={starts}, boundary_width={bw}, Lt={lt}"
                    )
                boundary_mask[t] = True

        if bool(np.all(boundary_mask)):
            raise ValueError(
                "Boundary slabs cover the full time extent; there is no interior left. "
                f"boundary_slices={starts}, boundary_width={bw}, Lt={lt}"
            )

        vol = int(np.prod(np.asarray(shp, dtype=np.int64)))
        coords = np.asarray(np.unravel_index(np.arange(vol, dtype=np.int64), shp), dtype=np.int64).T
        t_site = coords[:, -1].astype(np.int64)

        boundary_sites = np.flatnonzero(boundary_mask[t_site]).astype(np.int64)
        interior_sites = np.flatnonzero(~boundary_mask[t_site]).astype(np.int64)
        lookup = np.full((vol,), -1, dtype=np.int64)
        lookup[interior_sites] = np.arange(interior_sites.size, dtype=np.int64)

        self.lattice_shape = shp
        self.boundary_slices = starts
        self.boundary_width = bw
        self.boundary_times = np.flatnonzero(boundary_mask).astype(np.int64)
        self.interior_times = np.flatnonzero(~boundary_mask).astype(np.int64)
        self.boundary_site_indices = boundary_sites
        self.interior_site_indices = interior_sites
        self.interior_lookup = lookup
        self.vol = vol
        self.lt = lt

    def component_indices(self, site_indices: Sequence[int], nsc: int) -> np.ndarray:
        sites = np.asarray(site_indices, dtype=np.int64).reshape(-1)
        nsc_i = int(nsc)
        if nsc_i <= 0:
            raise ValueError(f"nsc must be positive, got {nsc!r}")
        base = sites[:, None] * nsc_i + np.arange(nsc_i, dtype=np.int64)[None, :]
        return base.reshape(-1)

    def boundary_component_indices(self, nsc: int) -> np.ndarray:
        return self.component_indices(self.boundary_site_indices, nsc)

    def interior_component_indices(self, nsc: int) -> np.ndarray:
        return self.component_indices(self.interior_site_indices, nsc)

    def interior_local_site(self, global_site: int) -> int:
        site = int(global_site) % int(self.vol)
        loc = int(self.interior_lookup[site])
        if loc < 0:
            raise ValueError(
                f"Global site {site} lies on a frozen boundary and is not part of the DD interior "
                f"(boundary_times={tuple(int(v) for v in self.boundary_times)})"
            )
        return loc

    def interior_time_components(self) -> Tuple[Tuple[int, ...], ...]:
        interior_mask = np.zeros((self.lt,), dtype=bool)
        interior_mask[self.interior_times] = True
        if not bool(np.any(interior_mask)):
            return tuple()

        starts = [
            int(t)
            for t in range(int(self.lt))
            if bool(interior_mask[t]) and (not bool(interior_mask[(t - 1) % int(self.lt)]))
        ]
        comps = []
        for start in starts:
            cur = []
            t = int(start)
            while bool(interior_mask[t]):
                cur.append(int(t))
                t = int((t + 1) % int(self.lt))
                if t == int(start):
                    break
            comps.append(tuple(cur))
        return tuple(comps)

    def interior_site_components(self) -> Tuple[np.ndarray, ...]:
        comps = []
        site_t = np.asarray(np.unravel_index(self.interior_site_indices, self.lattice_shape), dtype=np.int64).T[:, -1]
        for times in self.interior_time_components():
            mask = np.isin(site_t, np.asarray(times, dtype=np.int64))
            comps.append(np.asarray(self.interior_site_indices[mask], dtype=np.int64))
        return tuple(comps)

    def site_boundary_mask(self) -> np.ndarray:
        mask_t = np.zeros((self.lt,), dtype=bool)
        mask_t[self.boundary_times] = True
        shp_t = (1,) * (len(self.lattice_shape) - 1) + (self.lt,)
        return np.broadcast_to(mask_t.reshape(shp_t), self.lattice_shape).copy()

    def site_interior_mask(self) -> np.ndarray:
        return np.logical_not(self.site_boundary_mask())

    def link_active_mask(self, *, layout: str = "BMXYIJ", batch_size: int = 1, dtype=np.float32) -> np.ndarray:
        nd = int(len(self.lattice_shape))
        site_int = self.site_interior_mask()
        by_mu = []
        for mu in range(nd):
            act = np.array(site_int, copy=True)
            if mu == nd - 1:
                act = np.logical_and(act, np.roll(site_int, shift=-1, axis=nd - 1))
            by_mu.append(act)

        lay = str(layout).upper()
        if lay in ("BM...IJ",):
            lay = "BMXYIJ"
        if lay in ("B...MIJ",):
            lay = "BXYMIJ"
        if lay == "BMXYIJ":
            arr = np.stack(by_mu, axis=0)[None, ...]
        elif lay == "BXYMIJ":
            arr = np.stack(by_mu, axis=-1)[None, ...]
        else:
            raise ValueError(f"Unsupported layout for link_active_mask: {layout!r}")

        b = int(batch_size)
        if b <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if b > 1:
            reps = [b] + [1] * (arr.ndim - 1)
            arr = np.tile(arr, reps)
        return arr.astype(dtype, copy=False)

    def link_frozen_mask(self, *, layout: str = "BMXYIJ", batch_size: int = 1, dtype=np.float32) -> np.ndarray:
        active = self.link_active_mask(layout=layout, batch_size=batch_size, dtype=np.float32)
        return (1.0 - active).astype(dtype, copy=False)

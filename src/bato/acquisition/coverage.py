"""AOI coverage fraction: how much of the AOI a footprint (or a union of
footprints) covers.

Method: sample an N x N grid of points over the AOI bounding box, keep the
points inside the AOI, and weight each by cos(latitude) so every sample
represents the same ground area. Coverage = weighted share of AOI points
that fall inside at least one footprint.

Why a grid instead of exact polygon clipping:
* it handles any simple AOI polygon (convex or not) and any footprint shape;
* the union of several slices of one satellite pass is just a logical OR;
* the error is bounded by the grid spacing along the AOI boundary. With the
  default 200 x 200 grid the error is well under 1 % for AOIs of a few km
  or more, which is ample for a coverage threshold.

Approximation: footprint edges are treated as straight lines in lon/lat.
Over the AOI's extent the difference from geodesic edges is negligible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bato.contracts.common import PolygonGeometry


def points_in_ring(lons: np.ndarray, lats: np.ndarray, ring: Sequence[tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test for a closed (lon, lat) ring."""
    inside = np.zeros(lons.shape, dtype=bool)
    xs = np.asarray([p[0] for p in ring], dtype=float)
    ys = np.asarray([p[1] for p in ring], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(xs) - 1):
            x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
            crosses = (y1 > lats) != (y2 > lats)
            x_at = (x2 - x1) * (lats - y1) / (y2 - y1) + x1
            inside ^= crosses & (lons < x_at)
    return inside


class AOICoverageGrid:
    """Precomputed sample grid for one AOI; reused for every candidate scene."""

    def __init__(self, aoi: PolygonGeometry, grid_size: int) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        ring = aoi.exterior
        lon_min, lon_max = min(p[0] for p in ring), max(p[0] for p in ring)
        lat_min, lat_max = min(p[1] for p in ring), max(p[1] for p in ring)
        # Cell centres, so no sample sits exactly on the bounding box edge.
        lon_step, lat_step = (lon_max - lon_min) / grid_size, (lat_max - lat_min) / grid_size
        lons = lon_min + lon_step * (np.arange(grid_size) + 0.5)
        lats = lat_min + lat_step * (np.arange(grid_size) + 0.5)
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        in_aoi = points_in_ring(grid_lon, grid_lat, ring)
        if not in_aoi.any():
            raise ValueError("AOI is too thin for the coverage grid; increase coverage_grid_size")
        self._lons = grid_lon[in_aoi]
        self._lats = grid_lat[in_aoi]
        self._weights = np.cos(np.radians(self._lats))
        self._total = float(self._weights.sum())

    @property
    def sample_count(self) -> int:
        return int(self._lons.size)

    def covered_mask(self, footprint: PolygonGeometry) -> np.ndarray:
        """Boolean mask over the AOI samples that lie inside ``footprint``."""
        return points_in_ring(self._lons, self._lats, footprint.exterior)

    def fraction(self, footprints: Sequence[PolygonGeometry]) -> float:
        """Area-weighted fraction of the AOI covered by the union of ``footprints``."""
        covered = np.zeros(self._lons.shape, dtype=bool)
        for footprint in footprints:
            covered |= self.covered_mask(footprint)
        return round(float(self._weights[covered].sum()) / self._total, 4)

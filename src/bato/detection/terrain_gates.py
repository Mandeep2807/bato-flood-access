"""Terrain observability gates for the SAR change baseline, from the GLO-30 mosaic (approximate geometry).

* :func:`cast_shadow` - radar shadow CAST by terrain toward the sensor (e.g. a ridge shading a valley floor):
  a DEM cell is shadowed if, looking toward the sensor along the ground azimuth, some terrain within
  ``max_distance_m`` rises above the ray at elevation angle (90 deg - incidence), i.e.
  max_d (z(d) - z0) / d > cot(incidence). Local slope alone cannot see this: the shaded floor is flat.
* :func:`sample_grid` - nearest-neighbour sampling of a DEM-grid array at the 10 m stack pixel centres.

Approximations (documented in docs/baseline_detection.md): nominal incidence for the whole scene, straight
rays at the mean ground azimuth, nearest-cell stepping, flat-earth distances inside the mosaic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from bato.inference.diagnostics import M_PER_DEG, lonlat_from_3857


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc] (NaN outside)."""
    h, w = a.shape
    out = np.full_like(a, np.nan)
    rs, re = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = a[rs + dr:re + dr, cs + dc:ce + dc]
    return out


def cast_shadow(dem: np.ndarray, transform: Sequence[float], nodata: float | None, sensor_azimuth: float,
                incidence_deg: float, max_distance_m: float) -> np.ndarray:
    """Boolean cast-shadow mask on the DEM grid (geographic transform in degrees)."""
    a, _b, _c, _d, e, f = (float(v) for v in transform[:6])
    z = dem.astype(np.float32)
    if nodata is not None and not math.isnan(nodata):
        z[dem == nodata] = np.nan
    lat_mid = f + e * z.shape[0] / 2
    dx_m = abs(a) * M_PER_DEG * math.cos(math.radians(lat_mid))
    dy_m = abs(e) * M_PER_DEG
    step = min(dx_m, dy_m)
    east, north = math.sin(math.radians(sensor_azimuth)), math.cos(math.radians(sensor_azimuth))
    limit = 1.0 / math.tan(math.radians(incidence_deg))          # cot(incidence) = tan(90 - incidence)
    horizon = np.full(z.shape, -np.inf, dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    for k in range(1, int(max_distance_m // step) + 1):
        d = k * step
        dc = int(round(d * east / dx_m))
        dr = int(round(-d * north / dy_m))                       # rows grow southward
        if (dr, dc) in seen or (dr, dc) == (0, 0):
            continue
        seen.add((dr, dc))
        dist = math.hypot(dc * dx_m, dr * dy_m)
        rise = (_shift(z, dr, dc) - z) / dist
        horizon = np.fmax(horizon, rise)
    return np.isfinite(z) & (horizon > limit)


def sample_grid(array: np.ndarray, dem_transform: Sequence[float], grid_transform: Sequence[float], row0: int,
                rows: int, width: int, fill: float | bool) -> np.ndarray:
    """``array`` (DEM grid) at the stack pixel centres of rows row0..row0+rows-1; ``fill`` outside the DEM."""
    ga, _gb, gc, _gd, ge, gf = (float(v) for v in grid_transform[:6])
    x = gc + (np.arange(width) + 0.5) * ga
    y = gf + (row0 + np.arange(rows) + 0.5) * ge
    lon, _ = lonlat_from_3857(x, np.zeros_like(x))
    _, lat = lonlat_from_3857(np.zeros_like(y), y)
    a, _b, c, _d, e, f = (float(v) for v in dem_transform[:6])
    col = np.floor((lon - c) / a).astype(np.int64)
    row = np.floor((lat - f) / e).astype(np.int64)
    h, w = array.shape
    ok = (row[:, None] >= 0) & (row[:, None] < h) & (col[None, :] >= 0) & (col[None, :] < w)
    vals = array[np.clip(row, 0, h - 1)[:, None], np.clip(col, 0, w - 1)[None, :]]
    return np.where(ok, vals, fill)

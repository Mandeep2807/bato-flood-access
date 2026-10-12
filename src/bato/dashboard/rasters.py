"""Lightweight raster products for the dashboard, derived read-only from existing BATO files.

* Step 4C evidence overlays (EPSG:3857 grid): FLOOD / PRE_EVENT_WATER and NOT_ASSESSED categories, block-reduced
  to at most ``MAX_OVERLAY_PX`` on the long side. NOT_ASSESSED areas are drawn with a diagonal hatch.
* Hillshade of the Step 4C DEM (EPSG:4326), resampled to Mercator rows so it lines up on the web map.
* SAR chips: windows of the Step 3.5 stack (VV, every date with the SAME fixed dB stretch), the Step 4C state,
  and POST - PRE2 change (fixed +/- scale, no threshold), centred on each blocked stretch.

The pure functions take arrays; the ``read_*`` helpers need rasterio. Nothing here writes into a pipeline folder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bato.impact.roads import lonlat_to_3857
from bato.inference.diagnostics import lonlat_from_3857

MAX_OVERLAY_PX = 2400
MAX_HILLSHADE_PX = 2400
CHIP_HALF_PX = 160                       # 320 x 320 px = 3.2 km at 10 m
VV_DB_RANGE = (-22.0, -2.0)              # display stretch only; identical for every date
CHANGE_DB_RANGE = 8.0                    # POST - PRE2 colour scale, +/- dB (display only)
HATCH_PERIOD = 7

FLOOD, PEW = 1, 2
GEOMETRY_CODES = (12, 13)               # layover, shadow
STEEP_CODES = (14,)
DATA_CODES = (10, 11)                   # invalid SAR, no DEM

WATER_RGB = (78, 142, 247)
PEW_RGB = (120, 150, 180)
AMBER_RGB = (226, 163, 58)
STEEP_RGB = (196, 150, 82)
DATA_RGB = (110, 124, 132)


@dataclass(frozen=True)
class Placement:
    """Where an image goes on the map: corners as lon/lat (top-left, top-right, bottom-right, bottom-left)."""

    corners: tuple[tuple[float, float], ...]
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {"corners": [[round(x, 7), round(y, 7)] for x, y in self.corners], "width": self.width,
                "height": self.height}


def placement_3857(transform: tuple[float, ...], rows: int, cols: int, row0: int = 0, col0: int = 0,
                   out_w: int | None = None, out_h: int | None = None) -> Placement:
    a, _b, c, _d, e, f = (float(v) for v in transform[:6])
    x0, x1 = c + col0 * a, c + (col0 + cols) * a
    y0, y1 = f + row0 * e, f + (row0 + rows) * e
    lon, lat = lonlat_from_3857(np.array([x0, x1]), np.array([y0, y1]))
    w, s, ea, n = float(lon[0]), float(lat[1]), float(lon[1]), float(lat[0])
    return Placement(corners=((w, n), (ea, n), (ea, s), (w, s)), width=out_w or cols, height=out_h or rows)


def block_factor(h: int, w: int, max_px: int) -> int:
    return max(1, int(math.ceil(max(h, w) / max_px)))


def block_fractions(state: np.ndarray, factor: int, groups: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Fraction of each block (factor x factor pixels) in each code group."""
    h, w = state.shape
    bh, bw = int(math.ceil(h / factor)), int(math.ceil(w / factor))
    pad = np.full((bh * factor, bw * factor), 255, dtype=np.uint8)
    pad[:h, :w] = state
    real = np.zeros(pad.shape, dtype=bool)
    real[:h, :w] = True
    count = real.reshape(bh, factor, bw, factor).sum(axis=(1, 3)).astype(np.float64)
    out = {}
    for name, codes in groups.items():
        m = np.isin(pad, np.array(codes, dtype=np.uint8))
        out[name] = m.reshape(bh, factor, bw, factor).sum(axis=(1, 3)) / np.maximum(count, 1)
    return out


def _hatch(shape: tuple[int, int], period: int = HATCH_PERIOD, width: int = 2) -> np.ndarray:
    yy, xx = np.indices(shape)
    return ((xx + yy) % period) < width


def evidence_overlays(state: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """(flood RGBA, not-assessed RGBA) block-reduced images of the Step 4C state raster."""
    fr = block_fractions(state, factor, {"flood": (FLOOD,), "pew": (PEW,), "geometry": GEOMETRY_CODES,
                                         "steep": STEEP_CODES, "data": DATA_CODES})
    h, w = fr["flood"].shape
    flood = np.zeros((h, w, 4), dtype=np.uint8)
    pew = fr["pew"] > 0.25
    flood[pew, :3], flood[pew, 3] = PEW_RGB, 120
    f = fr["flood"] > 0
    flood[f, :3] = WATER_RGB
    flood[f, 3] = np.clip(150 + 105 * np.sqrt(fr["flood"][f]), 0, 255).astype(np.uint8)
    na = np.zeros((h, w, 4), dtype=np.uint8)
    hatch = _hatch((h, w))
    for key, rgb, fill, line in (("steep", STEEP_RGB, 18, 70), ("data", DATA_RGB, 40, 110),
                                 ("geometry", AMBER_RGB, 34, 175)):
        m = fr[key] >= 0.5
        na[m, :3], na[m, 3] = rgb, fill
        mh = m & hatch
        na[mh, 3] = line
    return flood, na


def hillshade(dem: np.ndarray, transform: tuple[float, ...], nodata: float | None, azimuth: float = 315.0,
              altitude: float = 40.0, z_factor: float = 1.6) -> np.ndarray:
    """uint8 hillshade of a lon/lat DEM (light from the north-west by default); NaN / nodata -> 0."""
    a, _b, _c, _d, e, f = (float(v) for v in transform[:6])
    z = dem.astype(np.float64)
    if nodata is not None and not math.isnan(nodata):
        z[z == nodata] = np.nan
    lat = f + (np.arange(z.shape[0]) + 0.5) * e
    dx = (a * 111_320.0 * np.cos(np.radians(lat)))[:, None]
    dy = e * 110_574.0
    gy, gx = np.gradient(z)
    dzdx = gx / dx * z_factor                 # east
    dzdy = gy / dy * z_factor                 # north (e < 0, so rows downward -> south)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, -dzdy)         # azimuth of the downslope direction, clockwise from north
    zen, az = math.radians(90 - altitude), math.radians(azimuth)
    shade = math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(shade * 255, 0, 255)
    out[~np.isfinite(out)] = 0
    return out.astype(np.uint8)


def shade_rgba(shade: np.ndarray, flat: float = 181.0) -> np.ndarray:
    """Hillshade as a transparent overlay for a dark map: shadows darken, lit slopes lighten slightly, flat ground
    (shade ~ flat) stays transparent. Nodata (0) is transparent."""
    d = (shade.astype(np.float64) - flat) / 74.0
    out = np.zeros(shade.shape + (4,), dtype=np.uint8)
    lit, dark = d > 0, d < 0
    out[lit, :3] = (196, 214, 222)
    out[lit, 3] = np.clip(d[lit] * 70, 0, 70).astype(np.uint8)
    out[dark, 3] = np.clip(-d[dark] * 170, 0, 170).astype(np.uint8)
    out[shade == 0, 3] = 0
    return out


def to_mercator_rows(img: np.ndarray, lat_top: float, lat_bottom: float, max_px: int) -> np.ndarray:
    """Resample a lon/lat image (rows evenly spaced in latitude) to rows evenly spaced in Mercator y, and
    reduce so the long side is <= max_px (nearest neighbour)."""
    h, w = img.shape[:2]
    _, y = lonlat_to_3857(np.zeros(2), np.array([lat_top, lat_bottom]))
    step = block_factor(h, w, max_px)
    out_h, out_w = max(1, h // step), max(1, w // step)
    ys = y[0] + (np.arange(out_h) + 0.5) / out_h * (y[1] - y[0])
    _, lats = lonlat_from_3857(np.zeros_like(ys), ys)
    rows = np.clip(((lat_top - lats) / (lat_top - lat_bottom) * h).astype(np.int64), 0, h - 1)
    cols = np.clip(((np.arange(out_w) + 0.5) / out_w * w).astype(np.int64), 0, w - 1)
    return img[rows][:, cols]


def vv_to_gray(linear: np.ndarray) -> np.ndarray:
    """(h, w) linear sigma0 -> (h, w, 2) grey + alpha (alpha 0 where the stack has no data)."""
    lo, hi = VV_DB_RANGE
    db = 10.0 * np.log10(np.maximum(linear.astype(np.float64), 1e-6))
    g = np.clip((db - lo) / (hi - lo), 0, 1) ** 0.85 * 255
    out = np.zeros(linear.shape + (2,), dtype=np.uint8)
    out[..., 0] = g.astype(np.uint8)
    out[..., 1] = np.where(linear > 0, 255, 0).astype(np.uint8)
    return out


def change_to_rgba(post: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """POST - PRE (dB) on a fixed diverging scale: darkening -> water blue, brightening -> warm white."""
    ok = (post > 0) & (pre > 0)
    d = 10.0 * np.log10(np.maximum(post, 1e-6) / np.maximum(pre, 1e-6))
    t = np.clip(d / CHANGE_DB_RANGE, -1, 1)
    out = np.zeros(post.shape + (4,), dtype=np.uint8)
    neg, pos = (t < 0) & ok, (t >= 0) & ok
    a = np.abs(t)
    for k, v in enumerate(WATER_RGB):
        out[..., k][neg] = (40 + (v - 40) * a[neg]).astype(np.uint8)
    for k, v in enumerate((236, 228, 214)):
        out[..., k][pos] = (40 + (v - 40) * a[pos]).astype(np.uint8)
    out[..., 3] = np.where(ok, 255, 0).astype(np.uint8)
    return out


def state_chip(state: np.ndarray) -> np.ndarray:
    out = np.zeros(state.shape + (4,), dtype=np.uint8)
    f, p = state == FLOOD, state == PEW
    out[f, :3], out[f, 3] = WATER_RGB, 215
    out[p, :3], out[p, 3] = PEW_RGB, 120
    g = np.isin(state, GEOMETRY_CODES + STEEP_CODES + DATA_CODES) & _hatch(state.shape, 6, 1)
    out[g, :3], out[g, 3] = AMBER_RGB, 120
    return out


def chip_window(transform: tuple[float, ...], shape: tuple[int, int], lon: float, lat: float,
                half: int = CHIP_HALF_PX) -> tuple[int, int, int, int] | None:
    """(row0, col0, rows, cols) of a window centred on lon/lat, clipped to the raster; None if outside."""
    a, _b, c, _d, e, f = (float(v) for v in transform[:6])
    x, y = lonlat_to_3857(np.array([lon]), np.array([lat]))
    col, row = int((float(x[0]) - c) // a), int((float(y[0]) - f) // e)
    h, w = shape
    r0, c0 = max(0, row - half), max(0, col - half)
    r1, c1 = min(h, row + half), min(w, col + half)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, c0, r1 - r0, c1 - c0


# ---------------------------------------------------------------------------
# rasterio readers
# ---------------------------------------------------------------------------


def read_band(path: Path, band: int = 1, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        if window is None:
            return np.asarray(ds.read(band))
        r0, c0, rows, cols = window
        return np.asarray(ds.read(band, window=Window(c0, r0, cols, rows)))


def read_dem_crop(path: Path, west: float, south: float, east: float, north: float
                  ) -> tuple[np.ndarray, tuple[float, ...], float | None, tuple[float, float, float, float]] | None:
    """Crop of a lon/lat DEM to a lon/lat box; None if the DEM is not in EPSG:4326 or does not overlap."""
    import rasterio
    from rasterio import windows as rwin

    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            return None
        try:
            win = rwin.from_bounds(west, south, east, north, transform=ds.transform).round_offsets().round_lengths()
            win = win.intersection(rwin.Window(0, 0, ds.width, ds.height))
        except rwin.WindowError:
            return None
        if win.width <= 2 or win.height <= 2:
            return None
        dem = np.asarray(ds.read(1, window=win))
        t = ds.window_transform(win)
        b = rwin.bounds(win, ds.transform)                        # west, south, east, north
        nodata = float(ds.nodata) if ds.nodata is not None else None
        return (dem, tuple(float(v) for v in tuple(t)[:6]), nodata,
                (float(b[0]), float(b[1]), float(b[2]), float(b[3])))

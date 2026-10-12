"""EVALUATION ONLY. Small, dependency-free geometry for comparing reference polygons with the BATO grid.

* WKT POLYGON / MULTIPOLYGON parsing (the activation record's AOI ``extent``).
* Spherical polygon area and clipping against a convex polygon (EMS AOI ∩ BATO AOI).
* Pixel-centre rasterisation onto the BATO EPSG:3857 grid (even-odd rule per feature, union over features).

Coordinates are WGS84 lon/lat. They are projected with the same spherical Web-Mercator formula as the
pipeline (:func:`bato.impact.roads.lonlat_to_3857`). Nothing is reprojected from any other CRS.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np

from bato.impact.roads import EARTH_RADIUS_3857, lonlat_to_3857

Ring = list[tuple[float, float]]
Polygon = list[Ring]                    # exterior first, then holes
EARTH_RADIUS_KM = 6371.0088             # mean Earth radius (IUGG)


class GeometryError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUM_PAIR = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_wkt_polygons(wkt: str) -> list[Polygon]:
    """POLYGON or MULTIPOLYGON WKT (optionally prefixed by ``SRID=...;``) -> list of polygons."""
    text = wkt.strip()
    if ";" in text and text.upper().startswith("SRID="):
        text = text.split(";", 1)[1].strip()
    upper = text.upper()
    if upper.startswith("MULTIPOLYGON"):
        body = text[len("MULTIPOLYGON"):].strip()
        depth_split = _split_top(body, level=2)
        return [_polygon_from_text(p) for p in depth_split]
    if upper.startswith("POLYGON"):
        return [_polygon_from_text(text[len("POLYGON"):].strip())]
    raise GeometryError(f"unsupported WKT geometry: {text[:40]!r}")


def _split_top(body: str, level: int) -> list[str]:
    """Pieces enclosed at the given parenthesis depth (1-based)."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
            if depth == level:
                start = i
        elif ch == ")":
            if depth == level:
                out.append(body[start:i + 1])
            depth -= 1
    if depth != 0:
        raise GeometryError("unbalanced parentheses in WKT")
    return out


def _polygon_from_text(body: str) -> Polygon:
    rings = []
    for piece in _split_top(body, level=2):
        pts = [(float(a), float(b)) for a, b in _NUM_PAIR.findall(piece)]
        if len(pts) < 4:
            raise GeometryError("a WKT ring needs at least 4 points")
        rings.append(pts)
    if not rings:
        raise GeometryError("empty WKT polygon")
    return rings


def geojson_polygons(geom: dict | None) -> list[Polygon]:
    """Polygon / MultiPolygon / GeometryCollection -> polygons. Non-areal parts raise (callers decide)."""
    if geom is None:
        return []
    kind = geom.get("type")
    if kind == "Polygon":
        return [[[(float(p[0]), float(p[1])) for p in ring] for ring in geom["coordinates"]]]
    if kind == "MultiPolygon":
        return [[[(float(p[0]), float(p[1])) for p in ring] for ring in poly] for poly in geom["coordinates"]]
    if kind == "GeometryCollection":
        out: list[Polygon] = []
        for g in geom.get("geometries", []):
            out.extend(geojson_polygons(g))
        return out
    raise GeometryError(f"not an areal geometry: {kind}")


def geojson_lines(geom: dict | None) -> list[Ring]:
    if geom is None:
        return []
    kind = geom.get("type")
    if kind == "LineString":
        return [[(float(p[0]), float(p[1])) for p in geom["coordinates"]]]
    if kind == "MultiLineString":
        return [[(float(p[0]), float(p[1])) for p in line] for line in geom["coordinates"]]
    raise GeometryError(f"not a line geometry: {kind}")


def iter_coords(geom: dict | None) -> Iterable[tuple[float, float]]:
    if geom is None:
        return
    if geom.get("type") == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from iter_coords(g)
        return

    def walk(c: object) -> Iterable[tuple[float, float]]:
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            yield float(c[0]), float(c[1])
        elif isinstance(c, (list, tuple)):
            for x in c:
                yield from walk(x)

    yield from walk(geom.get("coordinates"))


# ---------------------------------------------------------------------------
# Area and clipping
# ---------------------------------------------------------------------------


def ring_area_km2(ring: Sequence[tuple[float, float]]) -> float:
    """Unsigned area of a lon/lat ring on a sphere (Chamberlain & Duquette 2007)."""
    n = len(ring)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[(i + 1) % n]
        total += math.radians(lon2 - lon1) * (2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return abs(total) * EARTH_RADIUS_KM ** 2 / 2.0


def polygon_area_km2(poly: Polygon) -> float:
    return max(0.0, ring_area_km2(poly[0]) - sum(ring_area_km2(h) for h in poly[1:]))


def _signed_area(ring: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                     for i in range(len(ring)))


def _open(ring: Sequence[tuple[float, float]]) -> Ring:
    pts = list(ring)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def is_convex(ring: Sequence[tuple[float, float]]) -> bool:
    pts = _open(ring)
    sign = 0
    for i in range(len(pts)):
        (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) < 1e-15:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return sign != 0


def clip_ring_convex(subject: Sequence[tuple[float, float]], clip: Sequence[tuple[float, float]]) -> Ring:
    """Sutherland-Hodgman: ``subject`` (any simple ring) clipped by a convex ``clip`` ring (lon/lat plane).

    For a concave subject the result can contain zero-width bridges along the clip edges; its area is still
    the area of the intersection, which is all it is used for.
    """
    if not is_convex(clip):
        raise GeometryError("the clipping polygon must be convex")
    cp = _open(clip)
    if _signed_area(cp) < 0:
        cp = cp[::-1]
    out = _open(subject)
    for i in range(len(cp)):
        a, b = cp[i], cp[(i + 1) % len(cp)]
        inp, out = out, []
        if not inp:
            break

        def inside(p: tuple[float, float], a: tuple[float, float] = a, b: tuple[float, float] = b) -> bool:
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def cut(p: tuple[float, float], q: tuple[float, float], a: tuple[float, float] = a,
                b: tuple[float, float] = b) -> tuple[float, float]:
            x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        prev = inp[-1]
        for cur in inp:
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
            prev = cur
    return out


def intersection_area_km2(polys: Sequence[Polygon], clip: Sequence[tuple[float, float]]) -> float:
    """Area of (union of ``polys``, assumed non-overlapping) ∩ convex ``clip``."""
    total = 0.0
    for poly in polys:
        a = ring_area_km2(clip_ring_convex(poly[0], clip))
        a -= sum(ring_area_km2(clip_ring_convex(h, clip)) for h in poly[1:])
        total += max(0.0, a)
    return total


def bbox(polys: Sequence[Polygon]) -> tuple[float, float, float, float]:
    xs = [x for p in polys for r in p for x, _ in r]
    ys = [y for p in polys for r in p for _, y in r]
    if not xs:
        raise GeometryError("no coordinates")
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Rasterisation on the BATO grid
# ---------------------------------------------------------------------------


class Grid:
    """North-up EPSG:3857 grid: rasterio transform (a, b, c, d, e, f) with b = d = 0, e < 0."""

    def __init__(self, transform: Sequence[float], width: int, height: int) -> None:
        a, b, c, d, e, f = (float(v) for v in transform[:6])
        if b != 0 or d != 0 or a <= 0 or e >= 0:
            raise GeometryError(f"grid transform {tuple(transform)} is not north-up")
        self.a, self.c, self.e, self.f = a, c, e, f
        self.width, self.height = int(width), int(height)
        self.transform = (a, b, c, d, e, f)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def centre_x(self) -> np.ndarray:
        return self.c + (np.arange(self.width) + 0.5) * self.a

    def centre_y(self) -> np.ndarray:
        return self.f + (np.arange(self.height) + 0.5) * self.e

    def pixel_area_m2(self) -> float:
        """Pixel area in Web-Mercator units (not ground)."""
        return self.a * -self.e

    def row_ground_area_m2(self) -> np.ndarray:
        """Approximate ground area of one pixel in each row: Web-Mercator area x cos^2(latitude of the row)."""
        lat = np.degrees(2 * np.arctan(np.exp(self.centre_y() / EARTH_RADIUS_3857)) - np.pi / 2)
        return np.asarray(self.pixel_area_m2() * np.cos(np.radians(lat)) ** 2)


def project_ring(ring: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(ring, dtype=float)
    x, y = lonlat_to_3857(arr[:, 0], arr[:, 1])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def rasterize_polygon(poly: Polygon, grid: Grid) -> np.ndarray:
    """Pixel centres inside one polygon (even-odd over all its rings, so holes are excluded)."""
    mask = np.zeros(grid.shape, dtype=bool)
    edges_x0, edges_y0, edges_x1, edges_y1 = [], [], [], []
    for ring in poly:
        x, y = project_ring(_open(ring))
        if len(x) < 3:
            continue
        edges_x0.append(x)
        edges_y0.append(y)
        edges_x1.append(np.roll(x, -1))
        edges_y1.append(np.roll(y, -1))
    if not edges_x0:
        return mask
    x0, y0 = np.concatenate(edges_x0), np.concatenate(edges_y0)
    x1, y1 = np.concatenate(edges_x1), np.concatenate(edges_y1)
    keep = y0 != y1                                           # horizontal edges never cross a scanline
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    if x0.size == 0:
        return mask
    ys = grid.centre_y()
    rows = np.nonzero((ys >= min(y0.min(), y1.min())) & (ys <= max(y0.max(), y1.max())))[0]
    for r in rows:
        yc = ys[r]
        hit = ((y0 <= yc) & (yc < y1)) | ((y1 <= yc) & (yc < y0))   # half-open: each vertex counted once
        if not hit.any():
            continue
        t = (yc - y0[hit]) / (y1[hit] - y0[hit])
        xs = np.sort(x0[hit] + t * (x1[hit] - x0[hit]))
        for xa, xb in zip(xs[0::2], xs[1::2], strict=False):
            c0 = max(0, math.ceil((xa - grid.c) / grid.a - 0.5))
            c1 = min(grid.width, math.ceil((xb - grid.c) / grid.a - 0.5))
            if c1 > c0:
                mask[r, c0:c1] ^= True
    return mask


def rasterize(polys: Iterable[Polygon], grid: Grid) -> np.ndarray:
    """Union of pixel-centre masks of several polygons."""
    mask = np.zeros(grid.shape, dtype=bool)
    for p in polys:
        mask |= rasterize_polygon(p, grid)
    return mask


def densify_3857(line: Sequence[tuple[float, float]], step_m: float) -> np.ndarray:
    """Points every ``step_m`` (Web-Mercator metres) along a lon/lat line, ends included; shape (n, 2).

    Web-Mercator metres exaggerate ground distance by 1/cos(lat); callers that need ground distance convert
    with :func:`mercator_scale`.
    """
    x, y = project_ring(line)
    pts = [np.array([[x[0], y[0]]])]
    for i in range(len(x) - 1):
        seg = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        n = max(1, math.ceil(seg / step_m))
        t = np.arange(1, n + 1) / n
        pts.append(np.column_stack([x[i] + t * (x[i + 1] - x[i]), y[i] + t * (y[i + 1] - y[i])]))
    return np.concatenate(pts)


def resample_ground(line: Sequence[tuple[float, float]], step_m: float) -> tuple[np.ndarray, float]:
    """Uniform samples along a lon/lat line by GROUND distance: n = max(1, round(L / step)) samples at the midpoints
    of n equal pieces; returns (points in Web-Mercator metres, ground metres represented by each sample)."""
    x, y = project_ring(line)
    lat = np.asarray([p[1] for p in line], dtype=float)
    if len(x) < 2:
        return np.column_stack([x, y]), 0.0
    seg_merc = np.hypot(np.diff(x), np.diff(y))
    seg_ground = seg_merc * np.cos(np.radians((lat[:-1] + lat[1:]) / 2))
    cum = np.concatenate([[0.0], np.cumsum(seg_ground)])
    total = float(cum[-1])
    if total <= 0:
        return np.column_stack([x[:1], y[:1]]), 0.0
    n = max(1, round(total / step_m))
    s = (np.arange(n) + 0.5) * total / n
    return np.column_stack([np.interp(s, cum, x), np.interp(s, cum, y)]), total / n


def mercator_scale(lat_deg: float) -> float:
    """Web-Mercator metres per ground metre at a latitude."""
    return 1.0 / math.cos(math.radians(lat_deg))

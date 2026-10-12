"""Minimal, dependency-free geometry helpers for the ohsome v2 GeoParquet extract.

* :func:`wkb_to_geojson` - decode the ``geom`` column (Well-Known Binary, WGS84) into a GeoJSON geometry
  dict. Supports Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon in 2D, with ISO
  (Z/M/ZM) and EWKB (Z/M/SRID flags) variants; extra dimensions are dropped.
* :func:`centroid` - the client-side centroid ohsome v2 no longer provides (v1 ``/elements/centroid``):
  area-weighted for (multi)polygons, length-weighted for (multi)lines, mean for (multi)points; computed in
  lon/lat, which is accurate for features of settlement / building size.
"""

from __future__ import annotations

import struct
from typing import Any


class WKBError(ValueError):
    """The bytes are not a supported WKB geometry."""


_TYPES = {1: "Point", 2: "LineString", 3: "Polygon", 4: "MultiPoint", 5: "MultiLineString", 6: "MultiPolygon"}


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise WKBError("truncated WKB")
        out = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return out


def _geometry(r: _Reader) -> tuple[str, Any]:
    (order,) = r.take("B")
    if order not in (0, 1):
        raise WKBError(f"bad WKB byte order {order}")
    e = "<" if order == 1 else ">"
    (code,) = r.take(e + "I")
    has_z = bool(code & 0x80000000)
    has_m = bool(code & 0x40000000)
    if code & 0x20000000:                               # EWKB SRID
        r.take(e + "I")
    code &= 0x0FFFFFFF
    base = code % 1000
    iso = code // 1000
    has_z = has_z or iso in (1, 3)
    has_m = has_m or iso in (2, 3)
    dims = 2 + int(has_z) + int(has_m)
    kind = _TYPES.get(base)
    if kind is None:
        raise WKBError(f"unsupported WKB geometry type {code}")

    def point() -> list[float]:
        vals = r.take(e + "d" * dims)
        return [float(vals[0]), float(vals[1])]

    def points() -> list[list[float]]:
        (n,) = r.take(e + "I")
        return [point() for _ in range(n)]

    if kind == "Point":
        return kind, point()
    if kind == "LineString":
        return kind, points()
    if kind == "Polygon":
        (n,) = r.take(e + "I")
        return kind, [points() for _ in range(n)]
    (n,) = r.take(e + "I")
    parts = [_geometry(r) for _ in range(n)]
    member = {"MultiPoint": "Point", "MultiLineString": "LineString", "MultiPolygon": "Polygon"}[kind]
    if any(k != member for k, _ in parts):
        raise WKBError(f"{kind} contains a non-{member} member")
    return kind, [c for _, c in parts]


def wkb_to_geojson(data: bytes) -> dict[str, Any]:
    r = _Reader(bytes(data))
    kind, coords = _geometry(r)
    if r.pos != len(r.data):
        raise WKBError("trailing bytes after the WKB geometry")
    return {"type": kind, "coordinates": coords}


def _ring_area_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if a == 0:
        return 0.0, 0.0, 0.0
    return a, cx / (6 * a), cy / (6 * a)


def _mean(points: list[list[float]]) -> tuple[float, float]:
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    """(lon, lat) centroid of a GeoJSON geometry dict (see module docstring)."""
    kind, c = geometry["type"], geometry["coordinates"]
    if kind == "Point":
        return float(c[0]), float(c[1])
    if kind == "MultiPoint":
        return _mean(c)
    if kind in ("LineString", "MultiLineString"):
        lines = [c] if kind == "LineString" else c
        total = sx = sy = 0.0
        for line in lines:
            for (x0, y0), (x1, y1) in zip(line, line[1:], strict=False):
                w = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                total += w
                sx += w * (x0 + x1) / 2
                sy += w * (y0 + y1) / 2
        return (sx / total, sy / total) if total > 0 else _mean([p for line in lines for p in line])
    if kind in ("Polygon", "MultiPolygon"):
        polys = [c] if kind == "Polygon" else c
        total = sx = sy = 0.0
        for poly in polys:
            for i, ring in enumerate(poly):
                a, x, y = _ring_area_centroid(ring)
                a = abs(a) if i == 0 else -abs(a)            # holes subtract
                total += a
                sx += a * x
                sy += a * y
        if total != 0:
            return sx / total, sy / total
        return _mean([p for poly in polys for p in poly[0]])
    raise WKBError(f"no centroid for geometry type {kind}")

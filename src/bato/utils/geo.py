"""Small, dependency-free geometry helpers for request validation.

Scope is deliberately narrow: validating AOI rings in WGS84 and estimating
their area well enough to enforce size limits. Real geospatial processing
(reprojection, rasterisation, road/flood intersection) will use shapely,
pyproj and rasterio from the milestones that need them.

Approximations (documented in docs/assumptions.md):
* Area uses a spherical Earth (mean radius 6371.0088 km). Against the WGS84
  ellipsoid this is accurate to well under 1 %, enough for AOI size limits.
* Rings that cross the antimeridian or come close to a pole are rejected.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

EARTH_MEAN_RADIUS_KM = 6371.0088
MAX_ABS_LATITUDE = 85.0

Position = tuple[float, float]  # (lon, lat), GeoJSON order


def check_lon_lat(lon: float, lat: float) -> None:
    """Raise ValueError if a coordinate is outside WGS84 bounds."""
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError(f"non-finite coordinate ({lon}, {lat})")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude {lon} outside [-180, 180]")
    if not -MAX_ABS_LATITUDE <= lat <= MAX_ABS_LATITUDE:
        raise ValueError(f"latitude {lat} outside [-{MAX_ABS_LATITUDE}, {MAX_ABS_LATITUDE}]")


def _orientation(p: Position, q: Position, r: Position) -> int:
    value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(value) < 1e-15:
        return 0
    return 1 if value > 0 else 2


def _on_segment(p: Position, q: Position, r: Position) -> bool:
    """q lies on segment pr, given p, q, r are collinear."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def segments_intersect(p1: Position, q1: Position, p2: Position, q2: Position) -> bool:
    """True if closed segments p1-q1 and p2-q2 share at least one point."""
    o1, o2 = _orientation(p1, q1, p2), _orientation(p1, q1, q2)
    o3, o4 = _orientation(p2, q2, p1), _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(p1, p2, q1)) or (o2 == 0 and _on_segment(p1, q2, q1))
            or (o3 == 0 and _on_segment(p2, p1, q2)) or (o4 == 0 and _on_segment(p2, q1, q2)))


def validate_ring(ring: Sequence[Sequence[float]]) -> list[Position]:
    """Validate a closed GeoJSON linear ring and return it as (lon, lat) tuples.

    Checks: at least 4 positions, closed, coordinates in range, at least
    3 distinct vertices, no antimeridian crossing, no self-intersection,
    non-zero area.
    """
    if len(ring) < 4:
        raise ValueError("a polygon ring needs at least 4 positions (first == last)")
    points: list[Position] = []
    for position in ring:
        if len(position) != 2:
            raise ValueError("positions must be [lon, lat] pairs")
        lon, lat = float(position[0]), float(position[1])
        check_lon_lat(lon, lat)
        points.append((lon, lat))
    if points[0] != points[-1]:
        raise ValueError("polygon ring is not closed (first position must equal last)")
    if len(set(points[:-1])) < 3:
        raise ValueError("polygon ring needs at least 3 distinct vertices")
    lons = [p[0] for p in points]
    if max(lons) - min(lons) > 180.0:
        raise ValueError("rings crossing the antimeridian are not supported")

    edges = list(pairwise(points))
    n = len(edges)
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            if adjacent:
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                raise ValueError("polygon ring self-intersects")
    if spherical_ring_area_km2(points) <= 0.0:
        raise ValueError("polygon ring has zero area")
    return points


def spherical_ring_area_km2(points: Sequence[Position]) -> float:
    """Area enclosed by a closed (lon, lat) ring on a sphere, in km^2.

    Uses the line-integral formula from Chamberlain & Duquette (2007),
    "Some algorithms for polygons on a sphere". Orientation-independent.
    """
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in pairwise(points):
        total += math.radians(lon2 - lon1) * (2.0 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return abs(total) * EARTH_MEAN_RADIUS_KM**2 / 2.0


def bbox_ring(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[Position]:
    """Closed counter-clockwise ring for a lon/lat bounding box."""
    return [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat),
            (min_lon, max_lat), (min_lon, min_lat)]


def square_around_point(lon: float, lat: float, half_size_km: float) -> list[Position]:
    """Approximately square ring centred on a point (local equirectangular
    approximation; accurate to well under 1 % for boxes of tens of km)."""
    check_lon_lat(lon, lat)
    if half_size_km <= 0:
        raise ValueError("half_size_km must be positive")
    km_per_degree = math.pi * EARTH_MEAN_RADIUS_KM / 180.0
    dlat = half_size_km / km_per_degree
    dlon = dlat / math.cos(math.radians(lat))
    ring = bbox_ring(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    for p in ring:
        check_lon_lat(*p)
    return ring

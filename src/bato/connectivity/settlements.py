"""Settlements, access destinations and road snapping from the pre-event OSM extract.

Settlements (evaluated only inside the AOI, where the Step 4C map applies):

1. **Named places** - OSM ``place=`` in ``osm.settlement_place_values`` (city/town/village/hamlet).
2. **Building clusters** - OSM building centroids linked within ``connectivity.building_cluster_eps_m``
   (single linkage). A cluster with a building within that distance of a named place (inside the AOI)
   is linked; each of its buildings is counted for the nearest place it is linked to. An unlinked
   cluster with at least ``connectivity.building_cluster_min_buildings`` buildings is an unnamed
   settlement.

``building_count`` counts mapped OSM building footprints only: a PROXY for settlement size, NOT a
population figure. Unmapped buildings are not counted.

Access destinations (any inside the analysis box = AOI + ``aoi.analysis_buffer_km``):

* ``hospital``      - ``amenity=hospital`` or ``healthcare=hospital``;
* ``town``          - ``place`` in ``osm.town_place_values`` (city/town); a town is never its own
                      destination;
* ``analysis_exit`` - a road leaving the analysis box (it continues beyond the analysed area).

A settlement is snapped to the nearest road-graph node within ``connectivity.settlement_snap_max_m`` of its
own point (place point, or cluster centroid); if there is none, from its own buildings that lie within that
distance of the point (so a snap is never more than twice that distance from the settlement). Otherwise it
has no pre-event road access. Destinations are snapped from their own point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bato.connectivity.graph import UnionFind
from bato.contracts.settlement import SettlementSource, TargetKind

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class OSMPoint:
    osm_id: str                    # e.g. "node/123", "way/45"
    lon: float
    lat: float
    tags: dict[str, str]


def parse_points(features: list[dict[str, Any]]) -> list[OSMPoint]:
    out = []
    for f in features:
        g = f["geometry"]
        if g.get("type") != "Point":
            continue
        lon, lat = float(g["coordinates"][0]), float(g["coordinates"][1])
        props = f["properties"]
        out.append(OSMPoint(osm_id=str(props["@osmId"]), lon=lon, lat=lat,
                            tags={k: str(v) for k, v in props.items() if not k.startswith("@")}))
    out.sort(key=lambda p: p.osm_id)
    return out


@dataclass(frozen=True)
class LocalProjection:
    """Equirectangular metres around a reference latitude (error well below 1 % over ~100 km)."""

    lon0: float
    lat0: float

    def xy(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        k = math.radians(1.0) * EARTH_RADIUS_M
        return np.stack([(np.asarray(lon) - self.lon0) * k * math.cos(math.radians(self.lat0)),
                         (np.asarray(lat) - self.lat0) * k], axis=-1)


class GridIndex:
    """Nearest-point queries within a fixed radius (uniform grid of that cell size)."""

    def __init__(self, xy: np.ndarray, cell: float) -> None:
        self.xy = xy
        self.cell = cell
        keys = np.floor(xy / cell).astype(np.int64) if len(xy) else np.zeros((0, 2), dtype=np.int64)
        lists: dict[tuple[int, int], list[int]] = {}
        for i, (a, b) in enumerate(keys.tolist()):
            lists.setdefault((a, b), []).append(i)
        self.cells: dict[tuple[int, int], np.ndarray] = {k: np.array(v, dtype=np.int64) for k, v in lists.items()}

    def nearest(self, pts: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """(index, distance) of the nearest indexed point within ``radius`` for each query (-1 / inf if none)."""
        best_i = np.full(len(pts), -1, dtype=np.int64)
        best_d = np.full(len(pts), np.inf)
        if not len(pts) or not self.cells:
            return best_i, best_d
        keys = np.floor(pts / self.cell).astype(np.int64)
        groups: dict[tuple[int, int], list[int]] = {}
        for qi, (a, b) in enumerate(keys.tolist()):
            groups.setdefault((a, b), []).append(qi)
        reach = int(math.ceil(radius / self.cell))
        for (a, b), qs in groups.items():
            cand = [self.cells[(a + da, b + db)] for da in range(-reach, reach + 1) for db in range(-reach, reach + 1)
                    if (a + da, b + db) in self.cells]
            if not cand:
                continue
            idx = np.sort(np.concatenate(cand))
            q = np.array(qs, dtype=np.int64)
            d = np.hypot(pts[q, None, 0] - self.xy[None, idx, 0], pts[q, None, 1] - self.xy[None, idx, 1])
            j = np.argmin(d, axis=1)
            dj = d[np.arange(len(q)), j]
            ok = dj <= radius
            best_i[q[ok]] = idx[j[ok]]
            best_d[q[ok]] = dj[ok]
        return best_i, best_d


def cluster_points(xy: np.ndarray, eps: float) -> np.ndarray:
    """Single-linkage clusters at distance ``eps`` (label = smallest member index), deterministic.

    Grid of cell size eps/sqrt(2): points sharing a cell are always within eps; neighbouring cells
    (up to 2 cells away) are merged when their closest pair is within eps.
    """
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cell = eps / math.sqrt(2.0)
    keys = np.floor(xy / cell).astype(np.int64)
    members: dict[tuple[int, int], list[int]] = {}
    for i, (a, b) in enumerate(keys.tolist()):
        members.setdefault((a, b), []).append(i)
    uf = UnionFind(n)
    arrays = {k: np.array(v, dtype=np.int64) for k, v in members.items()}
    for idx in arrays.values():
        for j in idx[1:]:
            uf.union(int(idx[0]), int(j))
    offsets = [(da, db) for da in range(-2, 3) for db in range(-2, 3) if (da, db) > (0, 0)]
    for (a, b), idx in arrays.items():
        for da, db in offsets:
            other = arrays.get((a + da, b + db))
            if other is None:
                continue
            d = np.hypot(xy[idx, None, 0] - xy[None, other, 0], xy[idx, None, 1] - xy[None, other, 1])
            if d.min() <= eps:
                uf.union(int(idx[0]), int(other[0]))
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def point_in_ring(lon: float, lat: float, ring: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-300) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class SettlementCandidate:
    settlement_id: str
    name: str | None
    source: SettlementSource
    place_type: str | None               # OSM place=* value; None for building clusters
    osm_id: str | None                   # the place feature; None for building clusters
    lon: float
    lat: float
    building_count: int                  # mapped OSM buildings (proxy, NOT population)
    building_ids: tuple[str, ...]        # first few, for the evidence trail
    anchors: np.ndarray                  # (1, 2) the settlement's own point
    is_town: bool
    snap_node: int | None = None
    snap_distance_m: float | None = None


@dataclass
class Target:
    target_id: str                        # OSM id or "exit-<node>"
    kind: TargetKind
    name: str | None
    lon: float
    lat: float
    node: int | None
    snap_distance_m: float | None
    osm_id: str | None = None


@dataclass
class SettlementSet:
    settlements: list[SettlementCandidate]
    targets: list[Target]
    unsnapped_targets: list[Target] = field(default_factory=list)
    clusters_total: int = 0
    buildings_total: int = 0


def derive(places: list[OSMPoint], hospitals: list[OSMPoint], buildings: list[OSMPoint],
           aoi_ring: tuple[tuple[float, float], ...], proj: LocalProjection, node_xy: np.ndarray,
           exit_nodes: tuple[int, ...], node_lonlat: np.ndarray, *, settlement_values: tuple[str, ...],
           town_values: tuple[str, ...], eps_m: float, min_buildings: int, snap_max_m: float) -> SettlementSet:
    """Settlements inside the AOI and destinations in the analysis box, snapped to the road graph."""
    # only places inside the AOI are settlements, and only they may claim buildings
    named = [p for p in places if p.tags.get("place") in settlement_values and point_in_ring(p.lon, p.lat, aoi_ring)]
    b_xy = proj.xy(np.array([b.lon for b in buildings]), np.array([b.lat for b in buildings])) if buildings \
        else np.zeros((0, 2))
    labels = cluster_points(b_xy, eps_m)
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        clusters.setdefault(lab, []).append(i)
    # A cluster with a building within eps of a named place is linked to the place(s) it touches; each of its
    # buildings is then assigned to the nearest of those places (a valley-long chain is split, not given whole).
    p_xy = proj.xy(np.array([p.lon for p in named]), np.array([p.lat for p in named])) if named else np.zeros((0, 2))
    linked: set[int] = set()
    place_buildings: dict[int, list[int]] = {}
    if len(named) and len(b_xy):
        near_i, _ = GridIndex(p_xy, eps_m).nearest(b_xy, eps_m)
        for lab, idx in clusters.items():
            ii = np.array(idx)
            owners = np.unique(near_i[ii][near_i[ii] >= 0])
            if not len(owners):
                continue
            linked.add(lab)
            d = np.hypot(b_xy[ii, None, 0] - p_xy[None, owners, 0], b_xy[ii, None, 1] - p_xy[None, owners, 1])
            for bi, oi in zip(ii.tolist(), owners[np.argmin(d, axis=1)].tolist(), strict=True):
                place_buildings.setdefault(int(oi), []).append(int(bi))

    node_index = GridIndex(node_xy, max(snap_max_m, 1.0))

    def snap(anchors_lonlat: np.ndarray) -> tuple[int | None, float | None]:
        q = proj.xy(anchors_lonlat[:, 0], anchors_lonlat[:, 1])
        i, d = node_index.nearest(q, snap_max_m)
        if not (i >= 0).any():
            return None, None
        k = int(np.argmin(d))
        return int(i[k]), float(d[k])

    def snap_settlement(point: tuple[float, float], members: list[int]) -> tuple[int | None, float | None]:
        """From the settlement's own point first; otherwise from its buildings within snap_max_m of that point."""
        node, dist = snap(np.array([point]))
        if node is not None or not members:
            return node, dist
        here = proj.xy(np.array([point[0]]), np.array([point[1]]))[0]
        close = [i for i in members if math.hypot(*(b_xy[i] - here)) <= snap_max_m]
        if not close:
            return None, None
        return snap(np.array([[buildings[i].lon, buildings[i].lat] for i in close]))

    out: list[SettlementCandidate] = []
    for pi, p in enumerate(named):
        bidx = sorted(place_buildings.get(pi, []))
        node, dist = snap_settlement((p.lon, p.lat), bidx)
        out.append(SettlementCandidate(
            settlement_id=f"place-{p.osm_id.replace('/', '-')}", name=p.tags.get("name"),
            source=SettlementSource.OSM_PLACE, place_type=p.tags.get("place"), osm_id=p.osm_id, lon=p.lon, lat=p.lat,
            building_count=len(bidx), building_ids=tuple(buildings[i].osm_id for i in bidx[:20]),
            anchors=np.array([[p.lon, p.lat]]), is_town=p.tags.get("place") in town_values,
            snap_node=node, snap_distance_m=dist))
    for lab in sorted(clusters):
        idx = sorted(clusters[lab])
        if lab in linked or len(idx) < min_buildings:
            continue
        lon = float(np.mean([buildings[i].lon for i in idx]))
        lat = float(np.mean([buildings[i].lat for i in idx]))
        if not point_in_ring(lon, lat, aoi_ring):
            continue
        ids = sorted(buildings[i].osm_id for i in idx)
        node, dist = snap_settlement((lon, lat), idx)
        out.append(SettlementCandidate(
            settlement_id=f"bc-{ids[0].replace('/', '-')}", name=None, source=SettlementSource.BUILDING_CLUSTER,
            place_type=None, osm_id=None, lon=lon, lat=lat, building_count=len(idx), building_ids=tuple(ids[:20]),
            anchors=np.array([[lon, lat]]), is_town=False, snap_node=node, snap_distance_m=dist))

    targets: list[Target] = []
    unsnapped: list[Target] = []
    for h in hospitals:
        node, dist = snap(np.array([[h.lon, h.lat]]))
        t = Target(target_id=h.osm_id, kind=TargetKind.HOSPITAL, name=h.tags.get("name"), lon=h.lon, lat=h.lat,
                   node=node, snap_distance_m=dist, osm_id=h.osm_id)
        (targets if node is not None else unsnapped).append(t)
    settlement_by_osm = {s.osm_id: s for s in out if s.osm_id}
    for p in places:
        if p.tags.get("place") not in town_values:
            continue
        own = settlement_by_osm.get(p.osm_id)
        node, dist = (own.snap_node, own.snap_distance_m) if own is not None else snap(np.array([[p.lon, p.lat]]))
        t = Target(target_id=p.osm_id, kind=TargetKind.TOWN, name=p.tags.get("name"), lon=p.lon, lat=p.lat,
                   node=node, snap_distance_m=dist, osm_id=p.osm_id)
        (targets if node is not None else unsnapped).append(t)
    for n in exit_nodes:
        targets.append(Target(target_id=f"exit-{n}", kind=TargetKind.ANALYSIS_EXIT, name=None,
                              lon=float(node_lonlat[n, 0]), lat=float(node_lonlat[n, 1]), node=n, snap_distance_m=0.0))
    return SettlementSet(settlements=out, targets=targets, unsnapped_targets=unsnapped,
                         clusters_total=len(clusters), buildings_total=len(buildings))

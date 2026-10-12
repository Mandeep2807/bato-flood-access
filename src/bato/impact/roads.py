"""Road topology, segmentation and the three-state road impact (OPEN / BLOCKED / UNOBSERVABLE).

Topology (shared-vertex): OSM ways that meet share a node, and ohsome returns that node's
coordinates identically in both ways (7 decimals). A vertex becomes a graph node when it ends a
way part or occurs more than once in the network (junction or self-crossing). A bridge crossing
a road shares no vertex with it, so no false junction is created. Each junction-to-junction
polyline ("edge") is cut into ``road_impact.segment_length_m`` pieces of equal length
("segments"); segments are the graph edges and the unit of every road claim.

Exit nodes: ohsome clips geometries to the request (analysis) box, so a road that leaves the box
ends exactly on its edge. Such a dead end lying on the box boundary is an exit node: the road
continues outside the analysed area.

Road state from ``baseline_state.tif`` (Step 4C), sampled along each edge every half ground pixel:

* a sample is FLOOD if any FLOOD pixel lies within ``road_impact.road_buffer_m`` of it (square
  window, radius rounded to whole pixels);
* otherwise UNOBSERVABLE if its own pixel is NOT_ASSESSED (invalid SAR, no DEM, layover, shadow,
  steep) or lies outside the raster;
* otherwise observed-open (NO_FLOOD or PRE_EVENT_WATER).

Runs are measured along the whole edge and joined across nodes (way splits, bridge ends, junctions),
so neither segment boundaries nor OSM way splits shorten them:

* BLOCKED       - the segment contains part of a run of FLOOD samples >= ``min_blocked_run_m``;
* UNOBSERVABLE  - not BLOCKED, and it contains UNOBSERVABLE samples of a run of non-open samples
                  (FLOOD or UNOBSERVABLE) >= ``min_unobservable_run_m``: a blocking run could be
                  hidden there, so the segment is neither open nor blocked;
* OPEN          - otherwise (a short FLOOD run below the blocking length is recorded as affected
                  length but does not block).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bato.connectivity.graph import UnionFind
from bato.contracts.baseline import BaselineState
from bato.contracts.common import BBox
from bato.contracts.impact import RoadState
from bato.preprocessing.terrain_validation import EARTH_RADIUS_3857

EARTH_RADIUS_M = 6_371_008.8
COORD_SCALE = 10_000_000          # 7 decimals: OSM coordinate precision
EXIT_TOLERANCE_DEG = 1e-6
OUTSIDE_RASTER = 255              # sample reason code: no SAR raster at the sample
DISTURBANCE_REASON = 201          # sample reason code (rule v2): affected by DISTURBANCE only (no FLOOD near it)
ROAD_RULE_V1, ROAD_RULE_V2 = "v1", "v2"

SAMPLE_OPEN, SAMPLE_FLOOD, SAMPLE_UNOBSERVABLE = 0, 1, 2

REASON_NAMES = {int(s): s.name for s in BaselineState if int(s) >= 10}
REASON_NAMES[OUTSIDE_RASTER] = "OUTSIDE_SAR_RASTER"


class RoadNetworkError(ValueError):
    """The road layer cannot be turned into a network."""


@dataclass(frozen=True)
class RoadWay:
    osm_id: int
    tags: dict[str, str]
    parts: tuple[tuple[tuple[float, float], ...], ...]


def _way_id(osm_id: str) -> int | None:
    kind, _, num = osm_id.partition("/")
    return int(num) if kind == "way" and num.isdigit() else None


def parse_roads(features: list[dict[str, Any]]) -> list[RoadWay]:
    """OSM highway ways from the ohsome roads layer (LineString / MultiLineString; highway areas are skipped)."""
    ways = []
    for feature in features:
        props = feature["properties"]
        osm_id = _way_id(props["@osmId"])
        if osm_id is None:
            continue
        geom = feature["geometry"]
        if geom.get("type") == "LineString":
            raw = [geom["coordinates"]]
        elif geom.get("type") == "MultiLineString":
            raw = geom["coordinates"]
        else:                                       # highway areas (area=yes squares, ...) are not roads
            continue
        parts = []
        for line in raw:
            pts: list[tuple[float, float]] = []
            for c in line:
                p = (round(float(c[0]), 7), round(float(c[1]), 7))
                if not pts or p != pts[-1]:
                    pts.append(p)
            if len(pts) >= 2:
                parts.append(tuple(pts))
        if parts:
            tags = {k: str(v) for k, v in props.items() if not k.startswith("@")}
            ways.append(RoadWay(osm_id=osm_id, tags=tags, parts=tuple(parts)))
    ways.sort(key=lambda w: w.osm_id)
    return ways


def haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _key(p: tuple[float, float]) -> tuple[int, int]:
    return (round(p[0] * COORD_SCALE), round(p[1] * COORD_SCALE))


@dataclass
class Edge:
    """A junction-to-junction polyline of one way part."""

    index: int
    way: RoadWay
    part: int
    order: int                       # edge number within the part
    coords: np.ndarray               # (m, 2) lon, lat
    cum: np.ndarray                  # cumulative length at each vertex (m)
    start: int                       # node ids
    end: int

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def point_at(self, d: np.ndarray) -> np.ndarray:
        """Positions (k, 2) at distances ``d`` along the polyline (linear interpolation)."""
        d = np.clip(d, 0.0, self.length)
        i = np.clip(np.searchsorted(self.cum, d, side="right") - 1, 0, len(self.cum) - 2)
        span = self.cum[i + 1] - self.cum[i]
        t = np.where(span > 0, (d - self.cum[i]) / np.where(span > 0, span, 1.0), 0.0)
        return self.coords[i] + (self.coords[i + 1] - self.coords[i]) * t[:, None]

    def sub_polyline(self, d0: float, d1: float) -> list[tuple[float, float]]:
        lo, hi = np.searchsorted(self.cum, d0, side="right"), np.searchsorted(self.cum, d1, side="left")
        inner = [tuple(self.coords[j]) for j in range(int(lo), int(hi))]
        a, b = self.point_at(np.array([d0, d1]))
        return [(float(a[0]), float(a[1])), *[(float(x), float(y)) for x, y in inner], (float(b[0]), float(b[1]))]


@dataclass
class Segment:
    segment_id: str
    edge: int
    piece: int                       # index within the edge
    u: int
    v: int
    d0: float
    d1: float
    coords: list[tuple[float, float]]

    @property
    def length(self) -> float:
        return self.d1 - self.d0


@dataclass
class RoadNetwork:
    ways: dict[int, RoadWay]
    node_lonlat: np.ndarray                      # (N, 2)
    edges: list[Edge]
    segments: list[Segment]
    exit_nodes: tuple[int, ...]
    junction_nodes: int                          # nodes from OSM vertices (the rest are split points)

    @property
    def node_count(self) -> int:
        return len(self.node_lonlat)


def _on_boundary(lon: float, lat: float, box: BBox) -> bool:
    t = EXIT_TOLERANCE_DEG
    inside = box.min_lon - t <= lon <= box.max_lon + t and box.min_lat - t <= lat <= box.max_lat + t
    return inside and (abs(lon - box.min_lon) <= t or abs(lon - box.max_lon) <= t
                       or abs(lat - box.min_lat) <= t or abs(lat - box.max_lat) <= t)


def build_network(ways: list[RoadWay], segment_length_m: float, analysis_bbox: BBox) -> RoadNetwork:
    """Shared-vertex topology, equal-length segments and exit nodes (deterministic)."""
    if segment_length_m <= 0:
        raise RoadNetworkError("segment_length_m must be > 0")
    counts: dict[tuple[int, int], int] = {}
    for way in ways:
        for part in way.parts:
            for j, p in enumerate(part):
                k = _key(p)
                counts[k] = counts.get(k, 0) + (2 if j in (0, len(part) - 1) else 1)
    node_ids: dict[tuple[int, int], int] = {}
    node_xy: list[tuple[float, float]] = []

    def node(p: tuple[float, float]) -> int:
        k = _key(p)
        if k not in node_ids:
            node_ids[k] = len(node_xy)
            node_xy.append(p)
        return node_ids[k]

    edges: list[Edge] = []
    for way in ways:
        for pi, part in enumerate(way.parts):
            start = 0
            order = 0
            for j in range(1, len(part)):
                if j == len(part) - 1 or counts[_key(part[j])] >= 2:
                    coords = np.array(part[start:j + 1], dtype=np.float64)
                    seg = haversine_m(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
                    cum = np.concatenate([[0.0], np.cumsum(seg)])
                    if cum[-1] > 0:
                        edges.append(Edge(index=len(edges), way=way, part=pi, order=order, coords=coords, cum=cum,
                                          start=node(part[start]), end=node(part[j])))
                        order += 1
                    start = j
    junctions = len(node_xy)
    degree: dict[int, int] = {}
    for e in edges:
        degree[e.start] = degree.get(e.start, 0) + 1
        degree[e.end] = degree.get(e.end, 0) + 1
    segments: list[Segment] = []
    for e in edges:
        pieces = max(1, math.ceil(e.length / segment_length_m - 1e-9))
        step = e.length / pieces
        cuts = [i * step for i in range(pieces + 1)]
        ids = [e.start]
        for d in cuts[1:-1]:
            p = e.point_at(np.array([d]))[0]
            ids.append(len(node_xy))
            node_xy.append((float(p[0]), float(p[1])))
        ids.append(e.end)
        for i in range(pieces):
            segments.append(Segment(segment_id=f"seg-{e.way.osm_id}-{e.part}-{e.order}-{i}", edge=e.index, piece=i,
                                    u=ids[i], v=ids[i + 1], d0=cuts[i], d1=cuts[i + 1],
                                    coords=e.sub_polyline(cuts[i], cuts[i + 1])))
    exits = tuple(sorted(n for n, deg in degree.items()
                         if deg == 1 and _on_boundary(node_xy[n][0], node_xy[n][1], analysis_bbox)))
    return RoadNetwork(ways={w.osm_id: w for w in ways}, node_lonlat=np.array(node_xy, dtype=np.float64).reshape(-1, 2),
                       edges=edges, segments=segments, exit_nodes=exits, junction_nodes=junctions)


# ---------------------------------------------------------------------------
# Sampling the Step 4C state raster
# ---------------------------------------------------------------------------


def lonlat_to_3857(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_RADIUS_3857 * np.radians(lon)
    y = EARTH_RADIUS_3857 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-window binary dilation (``radius`` pixels) with numpy shifts."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            rs, re_ = max(0, dr), h + min(0, dr)
            cs, ce = max(0, dc), w + min(0, dc)
            out[rs - dr:re_ - dr, cs - dc:ce - dc] |= mask[rs:re_, cs:ce]
    return out


@dataclass
class StateSampler:
    """Classifies points against ``baseline_state.tif`` (array + rasterio-order transform).

    Rule v1 (``debris`` is None): affected = FLOOD within the road buffer; unobservable = the sample's Step 4C
    state is NOT_ASSESSED (10-14) or the sample is outside the raster.
    Rule v2 (``debris`` = ``debris_state.tif`` on the same grid): affected = FLOOD or DISTURBANCE within the road
    buffer; unobservable ONLY where the sample is geometry-unobservable (debris state 10-13: invalid SAR, no DEM,
    layover, shadow) or outside the raster. Steep terrain is assessed by the disturbance detector, so it no
    longer makes a road sample unobservable.
    """

    state: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    radius_px: int
    debris: np.ndarray | None = None
    flood_near: np.ndarray = field(init=False)
    disturbance_near: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        a, b, _c, d, e, _f = self.transform
        if b != 0 or d != 0 or a <= 0 or e >= 0:
            raise RoadNetworkError("the state raster must be north-up with square-ish positive pixels")
        self.flood_near = dilate(self.state == int(BaselineState.FLOOD), self.radius_px)
        if self.debris is not None:
            if self.debris.shape != self.state.shape:
                raise RoadNetworkError("debris_state.tif does not match the Step 4C state raster grid")
            self.disturbance_near = dilate(self.debris == 1, self.radius_px)

    @property
    def rule(self) -> str:
        return ROAD_RULE_V1 if self.debris is None else ROAD_RULE_V2

    def pixels(self, lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) of the raster pixel holding each (k, 2) lon/lat point."""
        a, _b, c, _d, e, f = self.transform
        x, y = lonlat_to_3857(lonlat[:, 0], lonlat[:, 1])
        col = np.floor((x - c) / a).astype(np.int64)
        row = np.floor((y - f) / e).astype(np.int64)
        h, w = self.state.shape
        return row, col, (row >= 0) & (row < h) & (col >= 0) & (col < w)

    def classify(self, lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sample class, reason code) for (k, 2) lon/lat points.

        Reason: the gating state (Step 4C state in v1, debris geometry state in v2), OUTSIDE_RASTER, or
        DISTURBANCE_REASON for a v2 sample affected by DISTURBANCE only.
        """
        row, col, inside = self.pixels(lonlat)
        h, w = self.state.shape
        rr, cc = np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)
        flood = inside & self.flood_near[rr, cc]
        if self.debris is None or self.disturbance_near is None:
            reason = np.where(inside, self.state[rr, cc], OUTSIDE_RASTER).astype(np.int64)
            affected = flood
        else:
            reason = np.where(inside, self.debris[rr, cc], OUTSIDE_RASTER).astype(np.int64)
            disturbance = inside & self.disturbance_near[rr, cc]
            affected = flood | disturbance
            reason = np.where(disturbance & ~flood, DISTURBANCE_REASON, reason)
        unobs = ~affected & (reason >= 10) & (reason != DISTURBANCE_REASON)
        cls = np.where(affected, SAMPLE_FLOOD, np.where(unobs, SAMPLE_UNOBSERVABLE, SAMPLE_OPEN)).astype(np.int8)
        return cls, reason


def road_buffer_radius_px(road_buffer_m: float, pixel_size_m: float, lat_deg: float) -> int:
    """Pixels (EPSG:3857 units shrink to ground metres by cos(lat))."""
    ground = pixel_size_m * math.cos(math.radians(lat_deg))
    return max(0, int(round(road_buffer_m / ground)))


def sample_step_m(pixel_size_m: float, lat_deg: float) -> float:
    """Half a ground pixel: every pixel the road crosses is sampled at least once."""
    return 0.5 * pixel_size_m * math.cos(math.radians(lat_deg))


# ---------------------------------------------------------------------------
# Segment states
# ---------------------------------------------------------------------------


@dataclass
class SegmentAssessment:
    segment: Segment
    state: RoadState
    length_m: float
    affected_length_m: float
    longest_affected_run_m: float
    unobservable_length_m: float
    qualifying_flood_run_m: float          # longest edge-level FLOOD run this segment is part of (if blocking)
    qualifying_hidden_run_m: float         # longest edge-level non-open run with unobservable samples it is part of
    samples: int
    flood_samples: int                     # affected samples (FLOOD or, in rule v2, DISTURBANCE)
    unobservable_reasons: dict[str, int]   # REASON_NAMES -> samples
    disturbance_samples: int = 0           # rule v2: affected samples from DISTURBANCE only


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of True runs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return list(zip(np.flatnonzero(diff == 1).tolist(), np.flatnonzero(diff == -1).tolist(), strict=True))


def _longest_run(mask: np.ndarray) -> int:
    return max((e - s for s, e in _runs(mask)), default=0)


def _joined_run_lengths(network: RoadNetwork, masks: dict[int, np.ndarray], units: dict[int, float],
                        flag: dict[int, np.ndarray] | None = None
                        ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Run lengths joined across nodes: a run touching an edge end continues into every run touching the same
    node from another edge (way splits, bridge ends, junctions). Returns per edge (joined length per sample,
    whether the joined run holds a flagged sample)."""
    pieces: list[tuple[int, int, int, float, bool]] = []
    at_node: dict[int, list[int]] = {}
    for edge in network.edges:
        mask = masks[edge.index]
        n = len(mask)
        for s0, e0 in _runs(mask):
            pid = len(pieces)
            flagged = bool(flag[edge.index][s0:e0].any()) if flag is not None else False
            pieces.append((edge.index, s0, e0, (e0 - s0) * units[edge.index], flagged))
            if s0 == 0:
                at_node.setdefault(edge.start, []).append(pid)
            if e0 == n:
                at_node.setdefault(edge.end, []).append(pid)
    uf = UnionFind(len(pieces))
    for pids in at_node.values():
        for pid in pids[1:]:
            uf.union(pids[0], pid)
    total: dict[int, float] = {}
    any_flag: dict[int, bool] = {}
    for pid, (_e, _s0, _e0, length, flagged) in enumerate(pieces):
        root = uf.find(pid)
        total[root] = total.get(root, 0.0) + length
        any_flag[root] = any_flag.get(root, False) or flagged
    lengths = {e.index: np.zeros(len(masks[e.index])) for e in network.edges}
    flags = {e.index: np.zeros(len(masks[e.index]), dtype=bool) for e in network.edges}
    for pid, (e, s0, e0, _length, _flagged) in enumerate(pieces):
        root = uf.find(pid)
        lengths[e][s0:e0] = total[root]
        flags[e][s0:e0] = any_flag[root]
    return lengths, flags


def assess_network(network: RoadNetwork, sampler: StateSampler, step_m: float, min_blocked_run_m: float,
                   min_unobservable_run_m: float) -> list[SegmentAssessment]:
    """Three-state assessment of every segment (order = network.segments).

    Runs are joined across nodes (see :func:`_joined_run_lengths`), so OSM way splits, bridge ends and
    junctions do not shorten a flooded or unobservable stretch below the thresholds. At a junction the
    joined length is the total connected length (a flooded junction counts all its flooded branches).
    """
    by_edge: dict[int, list[Segment]] = {}
    for s in network.segments:
        by_edge.setdefault(s.edge, []).append(s)
    samples: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    units: dict[int, float] = {}
    for edge in network.edges:
        n = max(1, math.ceil(edge.length / step_m))
        units[edge.index] = edge.length / n
        d = (np.arange(n) + 0.5) * units[edge.index]
        cls, reason = sampler.classify(edge.point_at(d))
        samples[edge.index] = (d, cls, reason)
    flood = {e: v[1] == SAMPLE_FLOOD for e, v in samples.items()}
    unobs = {e: v[1] == SAMPLE_UNOBSERVABLE for e, v in samples.items()}
    flood_len, _ = _joined_run_lengths(network, flood, units)
    hidden_len, hidden_flag = _joined_run_lengths(network, {e: flood[e] | unobs[e] for e in flood}, units, unobs)
    out: dict[str, SegmentAssessment] = {}
    for edge in network.edges:
        d, _cls, reason = samples[edge.index]
        f_all, u_all = flood[edge.index], unobs[edge.index]
        block_run = np.where(flood_len[edge.index] + 1e-9 >= min_blocked_run_m, flood_len[edge.index], 0.0)
        hidden_ok = hidden_flag[edge.index] & (hidden_len[edge.index] + 1e-9 >= min_unobservable_run_m)
        hidden_run = np.where(hidden_ok, hidden_len[edge.index], 0.0)
        segs = by_edge[edge.index]
        piece = np.minimum((d / (edge.length / len(segs))).astype(np.int64), len(segs) - 1)
        for i, seg in enumerate(segs):
            sel = piece == i
            m = int(sel.sum())
            seg_len = seg.length
            per = seg_len / m if m else 0.0
            f, u = f_all[sel], u_all[sel]
            if m and block_run[sel].max() > 0:
                state = RoadState.BLOCKED
            elif m and (hidden_run[sel] * u).max() > 0:
                state = RoadState.UNOBSERVABLE
            else:
                state = RoadState.OPEN
            reasons: dict[str, int] = {}
            for code, cnt in zip(*np.unique(reason[sel][u], return_counts=True), strict=True):
                reasons[REASON_NAMES.get(int(code), str(int(code)))] = int(cnt)
            out[seg.segment_id] = SegmentAssessment(
                segment=seg, state=state, length_m=seg_len,
                affected_length_m=min(seg_len, int(f.sum()) * per),
                longest_affected_run_m=min(seg_len, _longest_run(f) * per),
                unobservable_length_m=min(seg_len - min(seg_len, int(f.sum()) * per), int(u.sum()) * per),
                qualifying_flood_run_m=float(block_run[sel].max()) if m else 0.0,
                qualifying_hidden_run_m=float((hidden_run[sel] * u).max()) if m else 0.0,
                samples=m, flood_samples=int(f.sum()), unobservable_reasons=reasons,
                disturbance_samples=int((f & (reason[sel] == DISTURBANCE_REASON)).sum()))
    return [out[s.segment_id] for s in network.segments]


def is_bridge(tags: dict[str, str]) -> bool:
    return tags.get("bridge", "no") not in ("no", "")

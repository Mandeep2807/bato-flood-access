"""Route traces for the dashboard, rebuilt from the PUBLISHED connectivity outputs and verified against them.

The connectivity run publishes, per settlement, the destination and the length of its shortest route on each
graph, but not the path. The dashboard redraws that path from ``road_segments.geojson`` (segment end nodes
``u``/``v``, ``length_m``, ``state``) with the same edge weights, and keeps a trace only if its length equals
the published route length (tolerance ``LENGTH_TOLERANCE_M``). A trace that cannot be reproduced is reported
as unavailable, never approximated. Equal-length alternatives may be drawn instead of the engine's choice;
they have the same length by construction.

Nothing here changes a road state, a reachability result or a settlement status.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from bato.connectivity.graph import Graph

BLOCKED, UNOBSERVABLE, OPEN = 0, 1, 2
STATE_CODE = {"blocked": BLOCKED, "unobservable": UNOBSERVABLE, "open": OPEN}
GRAPH_MASKS = ("pre_event", "post_optimistic", "post_pessimistic")
LENGTH_TOLERANCE_M = 0.5
EARTH_RADIUS_M = 6_371_008.8


@dataclass
class Network:
    graph: Graph
    state: np.ndarray            # (E,) int8 BLOCKED / UNOBSERVABLE / OPEN
    length: np.ndarray           # (E,) float64 metres
    node_lonlat: np.ndarray      # (N, 2), NaN for nodes no segment touches

    def mask(self, name: str) -> np.ndarray:
        if name == "pre_event":
            return np.ones(self.state.size, dtype=bool)
        if name == "post_optimistic":
            return self.state != BLOCKED
        if name == "post_pessimistic":
            return self.state == OPEN
        raise ValueError(f"unknown graph {name!r}")


def build_network(u: Iterable[int], v: Iterable[int], length: Iterable[float], state: Iterable[int],
                  coords: list[list[tuple[float, float]]]) -> Network:
    ua, va = np.asarray(list(u), dtype=np.int64), np.asarray(list(v), dtype=np.int64)
    la = np.asarray(list(length), dtype=np.float64)
    sa = np.asarray(list(state), dtype=np.int8)
    n = int(max(ua.max(), va.max())) + 1 if ua.size else 0
    graph = Graph.build(n, ua, va, la)
    lonlat = np.full((n, 2), np.nan)
    for i, line in enumerate(coords):
        if line:
            lonlat[ua[i]] = line[0]
            lonlat[va[i]] = line[-1]
    return Network(graph=graph, state=sa, length=la, node_lonlat=lonlat)


def shortest_tree(net: Network, allowed: np.ndarray, sources: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(distance, predecessor edge) from the nearest of ``sources`` over the allowed edges (-1: none)."""
    g = net.graph
    dist = np.full(g.n_nodes, np.inf)
    pred = np.full(g.n_nodes, -1, dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for s in sorted(set(sources)):
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    done = np.zeros(g.n_nodes, dtype=bool)
    indptr, nbr, eid, length = g.indptr, g.nbr, g.eid, g.length
    while heap:
        d, node = heapq.heappop(heap)
        if done[node]:
            continue
        done[node] = True
        for k in range(indptr[node], indptr[node + 1]):
            e = int(eid[k])
            if not allowed[e]:
                continue
            m = int(nbr[k])
            nd = d + float(length[e])
            if nd < dist[m]:
                dist[m], pred[m] = nd, e
                heapq.heappush(heap, (nd, m))
    return dist, pred


def path_edges(net: Network, pred: np.ndarray, node: int, limit: int | None = None) -> list[int]:
    """Edges from ``node`` back to the tree root (in travel order from the node)."""
    out: list[int] = []
    cap = limit if limit is not None else net.graph.n_nodes
    while pred[node] >= 0 and len(out) <= cap:
        e = int(pred[node])
        out.append(e)
        node = net.graph.other(e, node)
    return out


def haversine_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    p1, p2 = np.radians(lat), math.radians(lat0)
    dphi, dl = p2 - p1, np.radians(lon0 - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def snap_candidates(net: Network, lon: float, lat: float, snap_distance_m: float | None, k: int = 8,
                    reach_m: float = 600.0, k_near: int = 48) -> list[int]:
    """Candidate snap nodes, best first: nodes whose distance from the settlement point matches the published snap
    distance, then the nearest nodes within ``snap_distance_m + reach_m`` (the engine may snap from one of the
    settlement's buildings instead of its point). The route-length check in RouteTracer decides."""
    if snap_distance_m is None:
        return []
    ok = np.isfinite(net.node_lonlat[:, 0])
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    d = haversine_m(net.node_lonlat[idx, 0], net.node_lonlat[idx, 1], lon, lat)
    gap = np.abs(d - snap_distance_m)
    order = np.argsort(gap, kind="stable")[:k]
    tol = max(5.0, 0.02 * snap_distance_m)
    out = [int(idx[i]) for i in order if gap[i] <= tol]
    near = np.argsort(d, kind="stable")[:k_near]
    out += [int(idx[i]) for i in near if d[i] <= snap_distance_m + reach_m and int(idx[i]) not in out]
    return out


@dataclass
class Trace:
    graph: str
    target_id: str
    node: int                    # the snap node reproduced
    edges: list[int]             # segment indices, settlement -> destination
    length_m: float
    km_by_state: dict[str, float]
    unobservable_segments: int
    blocked_segments: int


def trace_metrics(net: Network, edges: list[int]) -> tuple[dict[str, float], int, int]:
    st = net.state[edges] if edges else np.zeros(0, dtype=np.int8)
    ln = net.length[edges] if edges else np.zeros(0)
    km = {name: round(float(ln[st == code].sum()) / 1000.0, 4) for name, code in STATE_CODE.items()}
    return km, int((st == UNOBSERVABLE).sum()), int((st == BLOCKED).sum())


class RouteTracer:
    """Caches one shortest-path tree per (graph, destination node)."""

    def __init__(self, net: Network) -> None:
        self.net = net
        self._trees: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._masks = {name: net.mask(name) for name in GRAPH_MASKS}

    def tree(self, graph: str, node: int) -> tuple[np.ndarray, np.ndarray]:
        key = (graph, node)
        if key not in self._trees:
            self._trees[key] = shortest_tree(self.net, self._masks[graph], [node])
        return self._trees[key]

    def trace(self, graph: str, target_id: str, target_node: int, route_length_m: float,
              candidates: list[int]) -> Trace | None:
        dist, pred = self.tree(graph, target_node)
        for node in candidates:
            if abs(float(dist[node]) - route_length_m) <= LENGTH_TOLERANCE_M:
                edges = path_edges(self.net, pred, node)
                length = float(self.net.length[edges].sum()) if edges else 0.0
                if abs(length - route_length_m) > LENGTH_TOLERANCE_M:
                    continue
                km, n_unobs, n_blocked = trace_metrics(self.net, edges)
                return Trace(graph=graph, target_id=target_id, node=node, edges=edges, length_m=round(length, 3),
                             km_by_state=km, unobservable_segments=n_unobs, blocked_segments=n_blocked)
        return None

    def nearest(self, graph: str, nodes: list[int]) -> np.ndarray:
        """Road distance from every node to the nearest of ``nodes`` on ``graph`` (inf if none)."""
        if not nodes:
            return np.full(self.net.graph.n_nodes, np.inf)
        dist, _ = shortest_tree(self.net, self._masks[graph], nodes)
        return dist


def unobservable_pockets(net: Network, segments: list[int]) -> list[list[int]]:
    """Group segments into runs that share a node (union-find), ordered by first segment index."""
    if not segments:
        return []
    parent = {e: e for e in segments}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_node: dict[int, int] = {}
    for e in segments:
        for node in (int(net.graph.u[e]), int(net.graph.v[e])):
            if node in by_node:
                ra, rb = find(e), find(by_node[node])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                by_node[node] = e
    groups: dict[int, list[int]] = {}
    for e in sorted(segments):
        groups.setdefault(find(e), []).append(e)
    return sorted(groups.values(), key=lambda g: g[0])

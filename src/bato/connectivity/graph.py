"""Deterministic graph algorithms on the segmented road network (undirected; one edge per segment).

* :func:`multi_source_dijkstra` - shortest route length from every node to its nearest access
  destination on one graph (pre-event, post-optimistic or post-pessimistic edge mask).
* :class:`UnionFind` / :func:`components` - connected components of an edge mask; used for the
  "clear one blocked segment" reconnection counts (two components merge through that segment).
* :func:`zero_one_bfs` - directed 0-1 BFS. The engine models every blocked stretch as a virtual node
  (entering it costs 1, leaving it costs 0) and passable segments as cost-0 arcs both ways, so the
  result is the minimum number of blocked stretches to clear for a node to reach any destination,
  with one such clearing path (predecessor arcs).

Ties are broken by node / edge index, so results are identical on every run.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

UNREACHABLE = -1


@dataclass(frozen=True)
class Graph:
    n_nodes: int
    u: np.ndarray                    # (E,) int64
    v: np.ndarray
    length: np.ndarray               # (E,) float64 metres
    indptr: np.ndarray               # CSR over nodes
    nbr: np.ndarray                  # neighbour node per adjacency entry
    eid: np.ndarray                  # edge index per adjacency entry

    @classmethod
    def build(cls, n_nodes: int, u: np.ndarray, v: np.ndarray, length: np.ndarray) -> Graph:
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        length = np.asarray(length, dtype=np.float64)
        if not (len(u) == len(v) == len(length)):
            raise ValueError("u, v and length must have the same length")
        if len(u) and (min(u.min(), v.min()) < 0 or max(u.max(), v.max()) >= n_nodes):
            raise ValueError("edge endpoint outside the node range")
        if (length < 0).any():
            raise ValueError("edge lengths must be >= 0")
        e = np.arange(len(u), dtype=np.int64)
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        ids = np.concatenate([e, e])
        order = np.lexsort((ids, src))
        src, dst, ids = src[order], dst[order], ids[order]
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(indptr, src + 1, 1)
        return cls(n_nodes=n_nodes, u=u, v=v, length=length, indptr=np.cumsum(indptr), nbr=dst, eid=ids)

    def other(self, edge: int, node: int) -> int:
        return int(self.v[edge]) if int(self.u[edge]) == node else int(self.u[edge])


def multi_source_dijkstra(graph: Graph, allowed: np.ndarray, sources: dict[int, int]
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(distance, source label) per node; ``sources`` maps node -> label (e.g. target index).

    Unreached nodes get distance inf and label UNREACHABLE.
    """
    dist = np.full(graph.n_nodes, np.inf)
    label = np.full(graph.n_nodes, UNREACHABLE, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []
    for node, lab in sorted(sources.items()):
        if lab < label[node] or label[node] == UNREACHABLE:
            dist[node], label[node] = 0.0, lab
    for node in sorted(sources):
        heapq.heappush(heap, (0.0, int(label[node]), node))
    done = np.zeros(graph.n_nodes, dtype=bool)
    indptr, nbr, eid, length = graph.indptr, graph.nbr, graph.eid, graph.length
    while heap:
        d, lab, node = heapq.heappop(heap)
        if done[node]:
            continue
        done[node] = True
        for k in range(indptr[node], indptr[node + 1]):
            e = eid[k]
            if not allowed[e]:
                continue
            m = int(nbr[k])
            nd = d + float(length[e])
            if nd < dist[m] or (nd == dist[m] and lab < label[m] and not done[m]):
                dist[m], label[m] = nd, lab
                heapq.heappush(heap, (nd, lab, m))
    return dist, label


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != root:
            self.parent[x], x = root, int(self.parent[x])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def components(graph: Graph, allowed: np.ndarray) -> np.ndarray:
    """Component label (smallest node id in the component) for every node."""
    uf = UnionFind(graph.n_nodes)
    for e in np.flatnonzero(allowed):
        uf.union(int(graph.u[e]), int(graph.v[e]))
    return np.array([uf.find(i) for i in range(graph.n_nodes)], dtype=np.int64)


def zero_one_bfs(n_nodes: int, arc_from: np.ndarray, arc_to: np.ndarray, arc_cost: np.ndarray,
                 sources: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Directed 0-1 BFS: minimum total arc cost (0 or 1) from any source to every node.

    Returns (cost per node, inf if unreachable; predecessor ARC index, -1 at a source).
    """
    arc_from = np.asarray(arc_from, dtype=np.int64)
    arc_to = np.asarray(arc_to, dtype=np.int64)
    arc_cost = np.asarray(arc_cost, dtype=np.int64)
    if not (len(arc_from) == len(arc_to) == len(arc_cost)) or (len(arc_cost) and not np.isin(arc_cost, (0, 1)).all()):
        raise ValueError("arcs must have matching lengths and costs 0 or 1")
    order = np.lexsort((np.arange(len(arc_from)), arc_from))
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, arc_from + 1, 1)
    indptr = np.cumsum(indptr)
    best = np.full(n_nodes, np.inf)
    pred = np.full(n_nodes, -1, dtype=np.int64)
    dq: deque[int] = deque()
    for src in sorted(set(sources)):
        best[src] = 0.0
        dq.append(src)
    while dq:
        node = dq.popleft()
        for k in range(indptr[node], indptr[node + 1]):
            a = int(order[k])
            c = int(arc_cost[a])
            m = int(arc_to[a])
            nd = best[node] + c
            if nd < best[m]:
                best[m], pred[m] = nd, a
                if c == 0:
                    dq.appendleft(m)
                else:
                    dq.append(m)
    return best, pred


def arc_path(arc_from: np.ndarray, pred: np.ndarray, node: int, limit: int) -> list[int]:
    """Arcs from the source to ``node`` (reverse walk along ``pred``), source-side first."""
    out: list[int] = []
    while pred[node] >= 0 and len(out) < limit:
        a = int(pred[node])
        out.append(a)
        node = int(arc_from[a])
    return out[::-1]

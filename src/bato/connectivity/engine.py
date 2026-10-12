"""Road connectivity before vs after the event, settlement access states, blocked-segment priorities and
evidence (Milestone 8). Deterministic; nothing is learned; nothing is validated.

Inputs (all verified by hash before use):

* the Step 4C baseline (``baseline_provenance.json`` + ``baseline_state.tif``) and, through its recorded
  hash, the Step 3.5 stack provenance (grid, AOI polygon, analysis box, the three Sentinel-1 products);
* the pre-event OSM extract (bato.osm.extract; snapshot strictly before the event).

Graphs (one edge per road segment, undirected, from the same pre-event network):

* pre-event         - every segment;
* post-optimistic   - BLOCKED removed, UNOBSERVABLE kept (best case for unobserved roads);
* post-pessimistic  - BLOCKED and UNOBSERVABLE removed (only observed-open roads).

Per settlement, a multi-source Dijkstra on each graph gives the nearest access destination and route
length; :func:`bato.contracts.settlement.classify_settlement` turns the three results into a status.
UNOBSERVABLE is never turned into OPEN or BLOCKED: it is exactly what separates CONFIRMED_CUTOFF
(optimistic graph fails) from POSSIBLE_CUTOFF (only the pessimistic graph fails).

Blocked-segment priority:

* single-stretch clearing (union-find): a blocked stretch is a connected run of BLOCKED segments;
  reopening it merges the components it touches, and every CONFIRMED_CUTOFF settlement in a merged
  component without an anchor destination (hospital / exit) regains a route if another merged component
  has one. Also counted on the pessimistic graph (route on observed-open roads);
* minimum clearing set (0-1 BFS, entering a stretch = 1, passable = 0): the fewest blocked stretches a
  cut-off settlement needs cleared, and one such set.

Every non-OPEN segment, every cut-off settlement and every segment that reconnects settlements gets an
:class:`~bato.contracts.evidence.EvidenceRecord`; settlement records chain to the evidence of the
segments that separate them.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.connectivity.graph import Graph, UnionFind, arc_path, components, multi_source_dijkstra, zero_one_bfs
from bato.connectivity.settlements import (
    LocalProjection,
    SettlementCandidate,
    SettlementSet,
    Target,
    derive,
    parse_points,
)
from bato.contracts.baseline import BaselineProvenance, BaselineState, ThresholdRecord
from bato.contracts.common import DataSource, LineStringGeometry, PointGeometry
from bato.contracts.connectivity import (
    ACCESS_STATE_BY_STATUS,
    AccessState,
    BlockedSegmentPriority,
    ConnectivityProvenance,
    ConnectivitySummary,
    FileRef,
)
from bato.contracts.debris import DebrisProvenance
from bato.contracts.evidence import (
    UNSUPPORTED_CLAIM_TERMS,
    ArtifactRef,
    EvidenceRecord,
    SourceRef,
    SubjectType,
    make_evidence_id,
)
from bato.contracts.impact import BridgeExposure, RoadSegmentImpact, RoadState
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.osm import OSMLayerName
from bato.contracts.settlement import (
    AccessStatus,
    Reachability,
    SettlementStatus,
    TargetKind,
    classify_settlement,
)
from bato.contracts.stack import StackProvenance
from bato.data_rules import assert_pipeline_input_allowed, assert_processed_output_dir
from bato.debris.disturbance import PROVENANCE_FILE as DEBRIS_PROVENANCE
from bato.debris.disturbance import TABLE_COLUMNS
from bato.debris.disturbance import output_dir as debris_output_dir
from bato.detection.baseline import DETECTION_GROUP
from bato.detection.baseline import OUTPUT_SUBDIR as BASELINE_SUBDIR
from bato.detection.baseline import PROVENANCE_FILE as BASELINE_PROVENANCE
from bato.exceptions import BatoError
from bato.impact.roads import (
    ROAD_RULE_V1,
    ROAD_RULE_V2,
    RoadNetwork,
    SegmentAssessment,
    StateSampler,
    assess_network,
    build_network,
    is_bridge,
    parse_roads,
    road_buffer_radius_px,
    sample_step_m,
)
from bato.osm.extract import OSMError, OSMExtract, load_osm_extract
from bato.preprocessing.model_input import check_stack_provenance
from bato.preprocessing.speckle import sha256_file

SEGMENTS_FILE = "road_segments.geojson"
SETTLEMENTS_FILE = "settlements.geojson"
TARGETS_FILE = "access_destinations.geojson"
PRIORITY_FILE = "blocked_segments_ranked.json"
EVIDENCE_FILE = "evidence_records.jsonl"
REPORT_FILE = "connectivity_report.md"
PROVENANCE_FILE = "connectivity_provenance.json"

METHOD_ROAD = "rule:road_state_from_baseline_v1"
METHOD_SETTLEMENT = "graph:settlement_reachability_3graph_v1"
METHOD_PRIORITY = "graph:single_segment_reconnection_unionfind_v1"
TOP_N = 25

STATE_ORDER = (RoadState.BLOCKED, RoadState.UNOBSERVABLE, RoadState.OPEN)


class ConnectivityError(BatoError):
    """Connectivity inputs are missing or inconsistent (fail closed)."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ConnectivityError(f"{path} is outside the project root {root}") from exc


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectivityJob:
    root: Path
    aoi_id: str
    candidate_id: str
    event_date: date
    baseline_provenance_path: Path
    baseline_provenance_sha256: str
    baseline: BaselineProvenance
    state_path: Path
    stack_provenance_path: Path
    stack: StackProvenance
    osm: OSMExtract
    final_dir: Path
    debris: DebrisInput | None = None            # road rule v2 input (None: rule v1)


@dataclass(frozen=True)
class DebrisInput:
    """The verified disturbance (debris) run of the same stack and Step 4C state (road rule v2)."""

    provenance_path: Path
    provenance_sha256: str
    provenance: DebrisProvenance
    state_path: Path
    table_path: Path


@dataclass
class DebrisData:
    state: np.ndarray                            # debris_state.tif
    table: dict[str, np.ndarray]                 # change_table.npz
    provenance: DebrisProvenance


def prepare_debris_input(cfg: BatoConfig, root: Path, aoi_id: str, candidate_id: str,
                         baseline: BaselineProvenance) -> DebrisInput:
    path = assert_pipeline_input_allowed(debris_output_dir(cfg, aoi_id, candidate_id) / DEBRIS_PROVENANCE,
                                         cfg.settings.paths)
    if not path.is_file():
        raise ConnectivityError(f"road rule v2 needs the disturbance run: {path} not found "
                                "(run scripts/run_debris_detection.py, or use --road-rule v1)")
    try:
        prov = load_contract(DebrisProvenance, path)
    except ContractIOError as exc:
        raise ConnectivityError(str(exc)) from exc
    if prov.aoi_id != aoi_id or prov.candidate_id != candidate_id:
        raise ConnectivityError("the disturbance run is for a different AOI or candidate")
    if prov.baseline.state_sha256 != baseline.outputs.state_sha256 or \
            prov.source.stack_sha256 != baseline.source.stack_sha256:
        raise ConnectivityError("the disturbance run was made from a different Step 4C state or stack")
    state = assert_pipeline_input_allowed(root / prov.outputs.state, cfg.settings.paths)
    table = assert_pipeline_input_allowed(root / prov.outputs.change_table, cfg.settings.paths)
    for f, sha in ((state, prov.outputs.state_sha256), (table, prov.outputs.change_table_sha256)):
        if not f.is_file() or sha256_file(f) != sha:
            raise ConnectivityError(f"{_relative(f, root)} is missing or changed since the disturbance run")
    return DebrisInput(provenance_path=path, provenance_sha256=sha256_file(path), provenance=prov, state_path=state,
                       table_path=table)


def output_dir(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> Path:
    for part in (aoi_id, candidate_id):
        if not part or any(c in part for c in "/\\:") or part in (".", ".."):
            raise ConnectivityError(f"invalid path component {part!r}")
    path = cfg.settings.paths.processed / cfg.settings.connectivity.output_subdir / aoi_id / candidate_id
    return assert_processed_output_dir(path, cfg.settings.paths)


def prepare_connectivity(cfg: BatoConfig, aoi_id: str, candidate_id: str, road_rule: str = ROAD_RULE_V1
                         ) -> ConnectivityJob:
    """Locate and verify every input (hash chain baseline -> stack provenance; OSM extract; v2: disturbance run)."""
    if road_rule not in (ROAD_RULE_V1, ROAD_RULE_V2):
        raise ConnectivityError(f"unknown road rule {road_rule!r} (v1 or v2)")
    root = cfg.config_dir.parent
    base_dir = cfg.settings.paths.interim / DETECTION_GROUP / aoi_id / candidate_id / BASELINE_SUBDIR
    prov_path = assert_pipeline_input_allowed(base_dir / BASELINE_PROVENANCE, cfg.settings.paths)
    if not prov_path.is_file():
        raise ConnectivityError(f"no Step 4C baseline: {prov_path} not found (run scripts/run_baseline_detection.py)")
    try:
        baseline = load_contract(BaselineProvenance, prov_path)
    except ContractIOError as exc:
        raise ConnectivityError(str(exc)) from exc
    if baseline.aoi_id != aoi_id or baseline.candidate_id != candidate_id:
        raise ConnectivityError("baseline provenance is for a different AOI or candidate")
    state_path = assert_pipeline_input_allowed(root / baseline.outputs.state, cfg.settings.paths)
    if not state_path.is_file() or sha256_file(state_path) != baseline.outputs.state_sha256:
        raise ConnectivityError(f"{baseline.outputs.state} is missing or changed since Step 4C (SHA-256 differs)")
    stack_path = assert_pipeline_input_allowed(root / baseline.source.provenance, cfg.settings.paths)
    if not stack_path.is_file() or sha256_file(stack_path) != baseline.source.provenance_sha256:
        raise ConnectivityError(f"{baseline.source.provenance} is missing or changed since Step 4C")
    try:
        stack = load_contract(StackProvenance, stack_path)
    except ContractIOError as exc:
        raise ConnectivityError(str(exc)) from exc
    try:
        check_stack_provenance(stack, aoi_id, candidate_id)
    except BatoError as exc:
        raise ConnectivityError(str(exc)) from exc
    assert stack.grid is not None
    event_date = date.fromisoformat(stack.event_date)
    try:
        osm = load_osm_extract(cfg, aoi_id, event_date)
    except OSMError as exc:
        raise ConnectivityError(str(exc)) from exc
    if osm.provenance.analysis_bbox != stack.grid.analysis_bbox:
        raise ConnectivityError("the OSM extract was requested for a different analysis box than the stack's "
                                "(exit nodes would be wrong); re-run scripts/fetch_osm_snapshot.py")
    debris = prepare_debris_input(cfg, root, aoi_id, candidate_id, baseline) if road_rule == ROAD_RULE_V2 else None
    return ConnectivityJob(root=root, aoi_id=aoi_id, candidate_id=candidate_id, event_date=event_date,
                           baseline_provenance_path=prov_path, baseline_provenance_sha256=sha256_file(prov_path),
                           baseline=baseline, state_path=state_path, stack_provenance_path=stack_path, stack=stack,
                           osm=osm, final_dir=output_dir(cfg, aoi_id, candidate_id), debris=debris)


def read_state_raster(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    from bato.preprocessing.sigma0_validation import _open, require_rasterio

    require_rasterio()
    with _open(path) as ds:
        state = np.asarray(ds.read(1), dtype=np.uint8)
        a, b, c, d, e, f = (float(v) for v in tuple(ds.transform)[:6])
    return state, (a, b, c, d, e, f)


def threshold_records(cfg: BatoConfig) -> tuple[ThresholdRecord, ...]:
    t = cfg.thresholds
    named = [(f"road_impact.{n}", getattr(t.road_impact, n)) for n in type(t.road_impact).model_fields]
    named += [(f"connectivity.{n}", getattr(t.connectivity, n)) for n in type(t.connectivity).model_fields]
    return tuple(ThresholdRecord(name=n, value=p.value, unit=p.unit, provisional=p.provisional,
                                 rationale=p.rationale, basis=p.basis or "inherited_bato_design") for n, p in named)


RULES = {
    "sample": "FLOOD if a FLOOD pixel lies within road_buffer_m (square window); else UNOBSERVABLE if the pixel is "
              "NOT_ASSESSED (10-14) or outside the raster; else observed-open. Samples every half ground pixel.",
    "segment_state": "BLOCKED: part of a contiguous FLOOD run >= min_blocked_run_m along the junction-to-junction "
                     "edge. UNOBSERVABLE: not BLOCKED and holds UNOBSERVABLE samples of a non-open run >= "
                     "min_unobservable_run_m. OPEN otherwise.",
    "graphs": "pre-event: all segments; post-optimistic: BLOCKED removed; post-pessimistic: BLOCKED and UNOBSERVABLE "
              "removed. Undirected (oneway ignored for access).",
    "destinations": "Reachability is decided by anchors: hospitals (amenity/healthcare=hospital) and analysis exits "
                    "(roads clipped at the analysis box = AOI + aoi.analysis_buffer_km). A town (place=city|town) is "
                    "a destination only while it is itself connected to an anchor in that graph, and never for "
                    "itself or a duplicate at its node; towns shorten routes, they never make a pocket connected.",
    "settlements": "OSM place=city|town|village|hamlet inside the AOI, plus unlinked building clusters "
                   "(building_cluster_eps_m, >= building_cluster_min_buildings) inside the AOI. Snap from the "
                   "settlement's own point (or its buildings within settlement_snap_max_m of it).",
    "status": "classify_settlement: pre unreachable -> no_pre_event_road_access; optimistic unreachable -> "
              "confirmed_cut_off; pessimistic unreachable -> possibly_cut_off; pessimistic route >= "
              "degraded_detour_ratio x pre route -> degraded; else connected.",
    "priority": "Union-find on each post graph: clearing one blocked stretch reconnects the settlements of "
                "anchor-less components it touches when another touched component has an anchor. 0-1 BFS "
                "(entering a blocked stretch = 1, else 0) gives the minimum clearing set per confirmed cut-off "
                "settlement.",
    "building_count": "Count of mapped OSM building footprints: a proxy for settlement size, NOT population.",
}
RULES_V2 = {
    "road_rule": "v2",
    "sample": "FLOOD (Step 4C) or DISTURBANCE (bato.debris) within road_buffer_m -> affected; else UNOBSERVABLE only "
              "if the pixel is geometry-unobservable (invalid SAR, no DEM, layover, shadow) or outside the raster; "
              "else observed-open. Steep terrain is assessed by the disturbance detector.",
    "disturbance": "VV and VH brightening >= z*sigma against both pre-event images, stable pre-event pair "
                   "(|d0| < z*sigma), sigma = 1.4826*MAD(d0), clusters >= detection.min_component_pixels.",
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


@dataclass
class SettlementResult:
    candidate: SettlementCandidate
    status: SettlementStatus
    access_state: AccessState
    own_target: int | None                       # index into targets (a town is not its own destination)
    min_blocked_stretches_to_clear: int | None
    clearing_stretch_ids: tuple[str, ...]
    unobservable_segments_on_clearing_path: int
    exit_reachable: dict[str, bool]


@dataclass
class ConnectivityResult:
    road_rule: str
    network: RoadNetwork
    assessments: list[SegmentAssessment]
    segment_impacts: list[RoadSegmentImpact]
    stretch_of: dict[int, int]                   # BLOCKED segment index -> stretch index
    stretches: list[list[int]]                   # segment indices per blocked stretch
    settlement_set: SettlementSet
    settlements: list[SettlementResult]
    priorities: list[BlockedSegmentPriority]     # one per stretch, ranked
    evidence: list[EvidenceRecord]
    summary: ConnectivitySummary
    warnings: list[str] = field(default_factory=list)


@dataclass
class _EvidenceContext:
    cfg_fingerprint: str
    s1_sources: tuple[SourceRef, ...]
    derived: SourceRef
    snapshot_date: date
    artifacts: tuple[ArtifactRef, ...]
    created_at: datetime
    extra_sources: tuple[SourceRef, ...] = ()


def _osm_source(osm_id: str, snapshot: date) -> SourceRef:
    return SourceRef(source=DataSource.OSM_PRE_EVENT, source_id=f"{osm_id}@{snapshot.isoformat()}",
                     reference_date=snapshot)


def _evidence(ctx: _EvidenceContext, subject_type: SubjectType, subject_id: str, method: str, claim: str,
              location: PointGeometry | LineStringGeometry, osm_ids: list[str], values: dict[str, float]
              ) -> EvidenceRecord:
    sources = (*ctx.s1_sources, ctx.derived, *ctx.extra_sources,
               *(_osm_source(o, ctx.snapshot_date) for o in sorted(set(osm_ids))))
    eid = make_evidence_id(subject_type, subject_id, method, ctx.cfg_fingerprint, [s.source_id for s in sources])
    return EvidenceRecord(evidence_id=eid, subject_type=subject_type, subject_id=subject_id, claim=claim,
                          location=location, sources=sources, method=method, method_version=bato.__version__,
                          config_fingerprint=ctx.cfg_fingerprint,
                          values={k: round(float(v), 3) for k, v in values.items()}, artifacts=ctx.artifacts,
                          created_at=ctx.created_at)


def _line(coords: list[tuple[float, float]]) -> LineStringGeometry:
    pts = [(round(x, 7), round(y, 7)) for x, y in coords]
    dedup = [pts[0]] + [p for i, p in enumerate(pts[1:], 1) if p != pts[i - 1]]
    if len(dedup) < 2:
        dedup = [pts[0], (pts[0][0] + 1e-7, pts[0][1])]
    return LineStringGeometry(coordinates=tuple(dedup))


def _point(lon: float, lat: float) -> PointGeometry:
    return PointGeometry(coordinates=(round(lon, 7), round(lat, 7)))


def _reach(dist: np.ndarray, label: np.ndarray, node: int | None, targets: list[Target]) -> Reachability:
    if node is None or not math.isfinite(dist[node]):
        return Reachability(reachable=False)
    t = targets[int(label[node])]
    return Reachability(reachable=True, target_id=t.target_id, target_kind=t.kind,
                        route_length_m=round(float(dist[node]), 3))


def _safe_name(name: str | None) -> str | None:
    """OSM names go into evidence claims; a name containing a term claims may not use is left out."""
    if name and not any(term in name.lower() for term in UNSUPPORTED_CLAIM_TERMS):
        return name
    return None


def _label(s: SettlementCandidate) -> str:
    name = _safe_name(s.name) or "unnamed building cluster"
    kind = s.place_type or "building cluster"
    return f"{name} ({kind}, {s.building_count} mapped buildings, id {s.settlement_id})"


def _target_label(targets: list[Target], target_id: str | None) -> str:
    for t in targets:
        if t.target_id == target_id:
            return f"{t.kind.value} {_safe_name(t.name) or t.target_id}"
    return str(target_id)


class _ChangeLookup:
    """Find change-table rows (DISTURBANCE and Step 4C FLOOD pixels with d1/d2/d0) by pixel."""

    def __init__(self, table: dict[str, np.ndarray], width: int) -> None:
        self.table = table
        self.width = width
        keys = table["row"].astype(np.int64) * width + table["col"].astype(np.int64)
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]

    def rows(self, flat: np.ndarray) -> np.ndarray:
        if self.keys.size == 0 or flat.size == 0:
            return np.zeros(0, dtype=np.int64)
        pos = np.searchsorted(self.keys, flat)
        ok = pos < self.keys.size
        ok[ok] = self.keys[pos[ok]] == flat[ok]
        return np.asarray(self.order[pos[ok]], dtype=np.int64)


def _segment_pixel_values(seg: Any, network: RoadNetwork, sampler: StateSampler, step_m: float,
                          debris: DebrisData | None, lookup: _ChangeLookup | None, a: SegmentAssessment
                          ) -> dict[str, float]:
    """Per-segment evidence values: Step 4C states under the road, geometry status and (v2) the d1/d2/d0 of
    the change pixels within the road buffer, with sigma and z."""
    edge = network.edges[seg.edge]
    n = max(2, int(math.ceil(seg.length / step_m)) + 1)
    rr, cc, inside = sampler.pixels(edge.point_at(np.linspace(seg.d0, seg.d1, n)))
    rr, cc = rr[inside], cc[inside]
    out: dict[str, float] = {"samples_outside_raster": float(n - int(inside.sum()))}
    for code, cnt in Counter(sampler.state[rr, cc].tolist()).items():
        name = BaselineState(code).name if code in {int(x) for x in BaselineState} else str(code)
        out[f"step4c_samples_{name}"] = float(cnt)
    if debris is None or lookup is None:
        return out
    geo = debris.state[rr, cc]
    out["geometry_observable_samples"] = float(np.isin(geo, (0, 1)).sum())
    out["geometry_unobservable_samples"] = float((geo >= 10).sum())
    r = sampler.radius_px
    h, w = sampler.state.shape
    flat = set()
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            ok = (rr + dr >= 0) & (rr + dr < h) & (cc + dc >= 0) & (cc + dc < w)
            flat.update(((rr[ok] + dr) * w + cc[ok] + dc).tolist())
    rows = lookup.rows(np.array(sorted(flat), dtype=np.int64))
    t = lookup.table
    kind = t["kind"][rows]
    dp = debris.provenance
    out.update(z=dp.z, sigma_vv_db=dp.noise[0].sigma_db, sigma_vh_db=dp.noise[1].sigma_db,
               change_pixels_disturbance=float((kind == 2).sum()), change_pixels_flood=float((kind == 1).sum()))
    if rows.size:
        pick = rows[kind == 2] if (kind == 2).any() else rows
        for col in ("d1_vv", "d2_vv", "d0_vv", "d1_vh", "d2_vh", "d0_vh"):
            vals = t[col][pick].astype(np.float64)
            out[f"{col}_db_median"] = float(np.median(vals))
            out[f"{col}_db_min"] = float(vals.min())
            out[f"{col}_db_max"] = float(vals.max())
    return out


def _join(ids: list[str] | tuple[str, ...], n: int = 5) -> str:
    return ", ".join(ids[:n]) + (f" and {len(ids) - n} more" if len(ids) > n else "")


def blocked_stretches(graph: Graph, blocked: np.ndarray) -> tuple[list[list[int]], dict[int, int]]:
    """Connected runs of BLOCKED segments (sharing a node), ordered by their first segment index."""
    uf = UnionFind(graph.n_nodes)
    for e in np.flatnonzero(blocked):
        uf.union(int(graph.u[e]), int(graph.v[e]))
    groups: dict[int, list[int]] = {}
    for e in np.flatnonzero(blocked):
        groups.setdefault(uf.find(int(graph.u[e])), []).append(int(e))
    stretches = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    return stretches, {e: i for i, g in enumerate(stretches) for e in g}


def clearing_arcs(graph: Graph, states: np.ndarray, stretches: list[list[int]]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arcs for the stretch-level 0-1 BFS: passable segments cost 0 both ways; each blocked stretch is a
    virtual node (index n_nodes + k), entered at cost 1 and left at cost 0 from any node it touches.

    Returns (from, to, cost, segment index or -1, stretch index or -1) per arc.
    """
    fr: list[int] = []
    to: list[int] = []
    cost: list[int] = []
    seg: list[int] = []
    stretch: list[int] = []
    for e in np.flatnonzero(states != 0):
        u, v = int(graph.u[e]), int(graph.v[e])
        fr += [u, v]
        to += [v, u]
        cost += [0, 0]
        seg += [int(e), int(e)]
        stretch += [-1, -1]
    for k, members in enumerate(stretches):
        hub = graph.n_nodes + k
        touched = sorted({int(graph.u[e]) for e in members} | {int(graph.v[e]) for e in members})
        for n in touched:
            fr += [n, hub]
            to += [hub, n]
            cost += [1, 0]
            seg += [-1, -1]
            stretch += [k, k]
    return (np.array(fr, dtype=np.int64), np.array(to, dtype=np.int64), np.array(cost, dtype=np.int64),
            np.array(seg, dtype=np.int64), np.array(stretch, dtype=np.int64))


def compute_connectivity(job: ConnectivityJob, cfg: BatoConfig, state: np.ndarray,
                         transform: tuple[float, float, float, float, float, float], *,
                         created_at: datetime | None = None, debris: DebrisData | None = None,
                         progress: Callable[[str], None] | None = None) -> ConnectivityResult:
    """Everything except writing files (pure given the verified inputs and the state raster).

    ``debris`` selects road rule v2 (FLOOD or DISTURBANCE affects a road; only geometry makes it unobservable).
    Graphs, routing, the decision table and the blocked-run rule are the same for both rules.
    """
    say = progress or (lambda _t: None)
    t = cfg.thresholds
    stack = job.stack
    grid = stack.grid
    assert grid is not None
    if state.shape != (grid.height, grid.width) or tuple(round(v, 6) for v in transform) != tuple(
            round(v, 6) for v in grid.transform):
        raise ConnectivityError("baseline_state.tif does not match the stack grid recorded in the stack provenance")
    osm = job.osm
    snapshot = osm.provenance.snapshot.snapshot_date
    analysis = osm.provenance.analysis_bbox
    lat_mid = (analysis.min_lat + analysis.max_lat) / 2
    lon_mid = (analysis.min_lon + analysis.max_lon) / 2

    say("building the road network ...\n")
    ways = parse_roads(osm.layers[OSMLayerName.ROADS])
    network = build_network(ways, t.road_impact.segment_length_m.value, analysis)
    say(f"  {len(ways):,} ways -> {len(network.edges):,} edges -> {len(network.segments):,} segments, "
        f"{network.node_count:,} nodes, {len(network.exit_nodes)} exit nodes\n")
    if debris is not None and debris.state.shape != state.shape:
        raise ConnectivityError("debris_state.tif does not match the stack grid")
    sampler = StateSampler(state=state, transform=transform,
                           radius_px=road_buffer_radius_px(t.road_impact.road_buffer_m.value, grid.pixel_size_m,
                                                           lat_mid),
                           debris=debris.state if debris is not None else None)
    step_m = sample_step_m(grid.pixel_size_m, lat_mid)
    say(f"road rule {sampler.rule}\n")
    assessments = assess_network(network, sampler, sample_step_m(grid.pixel_size_m, lat_mid),
                                 t.road_impact.min_blocked_run_m.value, t.road_impact.min_unobservable_run_m.value)

    say("deriving settlements and destinations ...\n")
    proj = LocalProjection(lon0=lon_mid, lat0=lat_mid)
    node_xy = proj.xy(network.node_lonlat[:, 0], network.node_lonlat[:, 1]) if network.node_count \
        else np.zeros((0, 2))
    sset = derive(parse_points(osm.layers[OSMLayerName.PLACES]), parse_points(osm.layers[OSMLayerName.HOSPITALS]),
                  parse_points(osm.layers[OSMLayerName.BUILDINGS]), stack.aoi_polygon.exterior, proj, node_xy,
                  network.exit_nodes, network.node_lonlat,
                  settlement_values=cfg.settings.osm.settlement_place_values,
                  town_values=cfg.settings.osm.town_place_values, eps_m=t.connectivity.building_cluster_eps_m.value,
                  min_buildings=int(t.connectivity.building_cluster_min_buildings.value),
                  snap_max_m=t.connectivity.settlement_snap_max_m.value)
    targets = sset.targets
    say(f"  {len(sset.settlements):,} settlements in the AOI; destinations: "
        f"{dict(sorted(Counter(x.kind.value for x in targets).items()))}\n")

    graph = Graph.build(network.node_count, np.array([s.u for s in network.segments], dtype=np.int64),
                        np.array([s.v for s in network.segments], dtype=np.int64),
                        np.array([s.length for s in network.segments], dtype=np.float64))
    states = np.array([STATE_ORDER.index(a.state) for a in assessments], dtype=np.int8)  # 0 blocked 1 unobs 2 open
    masks = {"pre_event": np.ones(len(states), dtype=bool), "post_optimistic": states != 0,
             "post_pessimistic": states == 2}
    stretches, stretch_of = blocked_stretches(graph, states == 0)

    # Reachability is decided by ANCHOR destinations (hospitals, analysis exits). A town is a destination only
    # while it is itself connected to an anchor in that graph, so towns shorten routes but never make an
    # isolated pocket look connected (and duplicate town features cannot mask each other).
    anchors = [i for i, tg in enumerate(targets)
               if tg.node is not None and tg.kind in (TargetKind.HOSPITAL, TargetKind.ANALYSIS_EXIT)]
    say("routing on the three graphs ...\n")
    comps = {name: components(graph, m) for name, m in masks.items()}
    target_count = {name: Counter(int(c[targets[i].node]) for i in anchors if targets[i].node is not None)
                    for name, c in comps.items()}

    def sources(name: str, exclude_town_nodes: frozenset[int] = frozenset()) -> dict[int, int]:
        c = comps[name]
        out: dict[int, int] = {}
        for i, tg in enumerate(targets):
            if tg.node is None:
                continue
            connected = target_count[name][int(c[tg.node])] > 0
            if tg.kind is TargetKind.TOWN and (tg.node in exclude_town_nodes or not connected):
                continue
            if tg.node not in out or i < out[tg.node]:
                out[tg.node] = i
        return out

    runs = {name: multi_source_dijkstra(graph, m, sources(name)) for name, m in masks.items()}
    exit_count = {name: Counter(int(c[tg.node]) for tg in targets
                                if tg.node is not None and tg.kind is TargetKind.ANALYSIS_EXIT)
                  for name, c in comps.items()}
    arc_from, arc_to, arc_cost, arc_seg, arc_stretch = clearing_arcs(graph, states, stretches)
    n_virtual = graph.n_nodes + len(stretches)
    anchor_nodes = sorted({n for i in anchors if (n := targets[i].node) is not None})
    best, pred = zero_one_bfs(n_virtual, arc_from, arc_to, arc_cost, anchor_nodes)

    def boundary(name: str, kinds: tuple[int, ...]) -> dict[int, list[int]]:
        """Removed segments of the given kinds that join two components, per component."""
        c = comps[name]
        out: dict[int, list[int]] = {}
        for e in np.flatnonzero(np.isin(states, kinds)):
            cu, cv = int(c[graph.u[e]]), int(c[graph.v[e]])
            if cu != cv:
                out.setdefault(cu, []).append(int(e))
                out.setdefault(cv, []).append(int(e))
        return out

    blocked_boundary = boundary("post_optimistic", (0,))
    hidden_boundary = boundary("post_pessimistic", (0, 1))
    ratio = t.connectivity.degraded_detour_ratio.value

    routed: list[tuple[SettlementCandidate, dict[str, Reachability], int | None, float, list[int]]] = []
    for s in sset.settlements:
        own_towns = [i for i, tg in enumerate(targets) if tg.kind is TargetKind.TOWN and tg.node is not None
                     and tg.node == s.snap_node] if s.snap_node is not None else []
        own = own_towns[0] if own_towns else None
        if own is not None and s.snap_node is not None:     # a town (or its duplicate) is not its own destination
            excl = frozenset({s.snap_node})
            per = {name: _reach(*multi_source_dijkstra(graph, m, sources(name, excl)), s.snap_node, targets)
                   for name, m in masks.items()}
        else:
            per = {name: _reach(*runs[name], s.snap_node, targets) for name in masks}
        k = float(best[s.snap_node]) if s.snap_node is not None else math.inf
        routed.append((s, per, own, k, []))

    # ---- road evidence and contracts
    say("writing evidence ...\n")
    stack_inputs: dict[str, Any] = {str(i.role): i for i in stack.inputs}
    s1_sources = tuple(SourceRef(source=DataSource.SENTINEL_1, source_id=i.product_name,
                                 reference_date=i.acquisition_start.date(), acquisition_start=i.acquisition_start)
                       for i in stack.inputs)
    post = stack_inputs["POST"]
    derived = SourceRef(source=DataSource.BATO_DERIVED,
                        source_id=f"step4c baseline_state.tif sha256:{job.baseline.outputs.state_sha256[:16]}",
                        reference_date=post.acquisition_start.date())
    roads_rec = next(r for r in osm.provenance.layers if r.layer is OSMLayerName.ROADS)
    artifacts: tuple[ArtifactRef, ...] = (
        ArtifactRef(path=job.baseline.outputs.state, kind="geotiff", sha256=job.baseline.outputs.state_sha256),
                 ArtifactRef(path=roads_rec.file, kind="geoparquet", sha256=roads_rec.sha256))
    extra: tuple[SourceRef, ...] = ()
    lookup: _ChangeLookup | None = None
    if debris is not None:
        dprov = debris.provenance
        extra = (SourceRef(source=DataSource.BATO_DERIVED,
                           source_id=f"disturbance debris_state.tif sha256:{dprov.outputs.state_sha256[:16]}",
                           reference_date=post.acquisition_start.date()),)
        artifacts += (ArtifactRef(path=dprov.outputs.state, kind="geotiff", sha256=dprov.outputs.state_sha256),
                      ArtifactRef(path=dprov.outputs.change_table, kind="npz",
                                  sha256=dprov.outputs.change_table_sha256))
        lookup = _ChangeLookup(debris.table, state.shape[1])
    ctx = _EvidenceContext(cfg_fingerprint=cfg.fingerprint(), s1_sources=s1_sources, derived=derived,
                           snapshot_date=snapshot, artifacts=artifacts, created_at=created_at or _utc_now(),
                           extra_sources=extra)
    dates = {r: stack_inputs[r].acquisition_start.date().isoformat() for r in ("PRE1", "PRE2", "POST")}
    evidence: list[EvidenceRecord] = []
    seg_evidence: dict[int, list[str]] = {}
    impacts: list[RoadSegmentImpact] = []
    for idx, (seg, a) in enumerate(zip(network.segments, assessments, strict=True)):
        way = network.ways[network.edges[seg.edge].way.osm_id]
        hw = way.tags.get("highway", "road")
        bridge = is_bridge(way.tags)
        geom = _line(seg.coords)
        ids: list[str] = []
        if a.state is not RoadState.OPEN:
            where = f"Road segment {seg.segment_id} (OSM way {way.osm_id}, highway={hw}{', bridge' if bridge else ''})"
            if a.state is RoadState.BLOCKED and a.disturbance_samples > 0:
                dp = debris.provenance if debris is not None else None
                sig = (f" (VV and VH brightening >= z*sigma, z={dp.z:g}, sigma VV {dp.noise[0].sigma_db:.2f} dB / VH "
                       f"{dp.noise[1].sigma_db:.2f} dB, against both PRE-1 {dates['PRE1']} and PRE-2 {dates['PRE2']}, "
                       f"stable pre-event pair; POST {dates['POST']})") if dp is not None else ""
                kinds = "radar disturbance consistent with debris" + (
                    " and new open-water-like darkening" if a.disturbance_samples < a.flood_samples else "")
                claim = (f"{where} is BLOCKED: a {a.qualifying_flood_run_m:.0f} m contiguous run of {kinds}{sig} "
                         f"covers {a.affected_length_m:.0f} m of its {a.length_m:.0f} m on the road corridor "
                         f"(+/- road buffer). Part of blocked stretch blk-{stretch_of[idx]}. Provisional rule output, "
                         "not validated.")
            elif a.state is RoadState.BLOCKED:
                claim = (f"{where} is BLOCKED: a {a.qualifying_flood_run_m:.0f} m contiguous run of new "
                         f"open-water-like darkening (Step 4C baseline rule: POST {dates['POST']} darker than "
                         f"PRE-2 {dates['PRE2']} and PRE-1 {dates['PRE1']}) covers {a.affected_length_m:.0f} m of its "
                         f"{a.length_m:.0f} m. Part of blocked stretch blk-{stretch_of[idx]}. Provisional rule output, "
                         "not validated.")
            else:
                reasons = ", ".join(f"{k} {v}" for k, v in sorted(a.unobservable_reasons.items()))
                hazard = "a blocking flood or debris" if debris is not None else "a blocking flood"
                claim = (f"{where} is UNOBSERVABLE: {a.unobservable_length_m:.0f} of {a.length_m:.0f} m could not "
                         f"be assessed by Sentinel-1 (samples: {reasons}) inside a {a.qualifying_hidden_run_m:.0f} m "
                         f"run that could hide {hazard}. Neither open nor blocked is claimed.")
            values = {"length_m": a.length_m, "affected_length_m": a.affected_length_m,
                      "longest_affected_run_m": a.longest_affected_run_m,
                      "unobservable_length_m": a.unobservable_length_m,
                      "qualifying_flood_run_m": a.qualifying_flood_run_m,
                      "qualifying_hidden_run_m": a.qualifying_hidden_run_m, "samples": a.samples,
                      "flood_samples": a.flood_samples,
                      **{f"unobservable_samples_{k}": v for k, v in a.unobservable_reasons.items()}}
            values.update(_segment_pixel_values(seg, network, sampler, step_m, debris, lookup, a))
            ev = _evidence(ctx, SubjectType.ROAD_SEGMENT, seg.segment_id, METHOD_ROAD, claim, geom,
                           [f"way/{way.osm_id}"], values)
            evidence.append(ev)
            ids = [ev.evidence_id]
        seg_evidence[idx] = ids
        exposure = (BridgeExposure.NOT_A_BRIDGE if not bridge else
                    BridgeExposure.EXPOSED if a.affected_length_m > 0 else BridgeExposure.NOT_EXPOSED)
        impacts.append(RoadSegmentImpact(
            segment_id=seg.segment_id, osm_way_id=way.osm_id, highway=hw, geometry=geom, length_m=a.length_m,
            state=a.state, affected_length_m=a.affected_length_m, longest_affected_run_m=a.longest_affected_run_m,
            unobservable_length_m=a.unobservable_length_m, bridge=exposure, evidence_ids=tuple(ids)))

    # ---- settlements
    settlements: list[SettlementResult] = []
    for s, per, own, k, arcs in routed:
        status = classify_settlement(per["pre_event"], per["post_optimistic"], per["post_pessimistic"], ratio)
        blocking: list[int] = []
        if status is AccessStatus.CONFIRMED_CUT_OFF and s.snap_node is not None:
            # every segment of each blocked stretch on the component's boundary
            comp = int(comps["post_optimistic"][s.snap_node])
            touching = sorted({stretch_of[e] for e in blocked_boundary.get(comp, [])})
            blocking = [e for k_ in touching for e in stretches[k_]]
        elif status is AccessStatus.POSSIBLY_CUT_OFF and s.snap_node is not None:
            blocking = hidden_boundary.get(int(comps["post_pessimistic"][s.snap_node]), [])
        blocking = sorted(set(blocking))
        blocking_ids = tuple(network.segments[e].segment_id for e in blocking)
        confirmed = status is AccessStatus.CONFIRMED_CUT_OFF
        if confirmed and s.snap_node is not None and math.isfinite(k):
            arcs = arc_path(arc_from, pred, s.snap_node, n_virtual + 1)
        clearing = sorted({int(arc_stretch[a]) for a in arcs if arc_stretch[a] >= 0}) if confirmed else []
        unobs_on_path = sum(1 for a in arcs if arc_seg[a] >= 0 and states[arc_seg[a]] == 1) if confirmed else 0
        ev_ids: list[str] = []
        if status in (AccessStatus.CONFIRMED_CUT_OFF, AccessStatus.POSSIBLY_CUT_OFF):
            pre = per["pre_event"]
            pre_txt = (f"before the event it reached {_target_label(targets, pre.target_id)} over "
                       f"{(pre.route_length_m or 0) / 1000:.1f} km of road")
            if confirmed:
                stretch_ids = sorted({f"blk-{stretch_of[e]}" for e in blocking})
                claim = (f"Settlement {_label(s)}: no road route to any access destination once BLOCKED segments are "
                         f"removed, even with every UNOBSERVABLE segment treated as passable; {pre_txt}. Separated by "
                         f"blocked stretch(es) {_join(stretch_ids)} ({len(blocking_ids)} segment(s) in total). "
                         f"Fewest blocked stretches to clear: {int(k) if math.isfinite(k) else 'none possible'}"
                         f"{' (' + _join([f'blk-{c}' for c in clearing]) + ')' if clearing else ''}.")
            else:
                opt = per["post_optimistic"]
                claim = (f"Settlement {_label(s)}: a route to {_target_label(targets, opt.target_id)} "
                         f"({(opt.route_length_m or 0) / 1000:.1f} km) exists only through UNOBSERVABLE road "
                         f"segments; on observed-open roads alone there is none; {pre_txt}. Blocked or unobservable "
                         f"segments adjoining its observed-open road network: {_join(list(blocking_ids))}.")
            values = {"building_count_proxy": s.building_count, "snap_distance_m": s.snap_distance_m or 0.0,
                      "pre_event_route_m": per["pre_event"].route_length_m or 0.0,
                      "separating_segments": len(blocking_ids),
                      "post_optimistic_route_m": per["post_optimistic"].route_length_m or 0.0}
            if confirmed and math.isfinite(k):
                values["min_blocked_stretches_to_clear"] = k
            osm_ids = [s.osm_id] if s.osm_id else list(s.building_ids[:5])
            ev = _evidence(ctx, SubjectType.SETTLEMENT, s.settlement_id, METHOD_SETTLEMENT, claim,
                           _point(s.lon, s.lat), osm_ids, values)
            evidence.append(ev)
            ev_ids = [ev.evidence_id] + sorted({i for e in blocking for i in seg_evidence[e]})
        contract = SettlementStatus(
            settlement_id=s.settlement_id, name=s.name, source=s.source, location=_point(s.lon, s.lat),
            building_count=s.building_count,
            snap_distance_m=round(s.snap_distance_m, 3) if s.snap_distance_m is not None else None,
            pre_event=per["pre_event"], post_optimistic=per["post_optimistic"],
            post_pessimistic=per["post_pessimistic"],
            degraded_detour_ratio=ratio, status=status, blocking_segment_ids=blocking_ids, evidence_ids=tuple(ev_ids))
        exit_ok = {name: s.snap_node is not None and exit_count[name][int(comps[name][s.snap_node])] > 0
                   for name in masks}
        settlements.append(SettlementResult(
            candidate=s, status=contract, access_state=ACCESS_STATE_BY_STATUS[status], own_target=own,
            min_blocked_stretches_to_clear=int(k) if confirmed and math.isfinite(k) else None,
            clearing_stretch_ids=tuple(f"blk-{c}" for c in clearing),
            unobservable_segments_on_clearing_path=unobs_on_path, exit_reachable=exit_ok))

    priorities = _priorities(network, graph, assessments, stretches, comps, target_count, settlements,
                             seg_evidence, ctx, evidence)
    summary = _summary(network, assessments, impacts, sset, settlements, priorities, comps, target_count, states)
    warnings = []
    if sset.unsnapped_targets:
        warnings.append(f"{len(sset.unsnapped_targets)} destination(s) farther than settlement_snap_max_m from any "
                        "road were ignored")
    if not network.exit_nodes:
        warnings.append("no road leaves the analysis box: settlements can only reach hospitals/towns")
    unobs_share = summary.road_km_by_state.get("unobservable", 0.0) / max(sum(summary.road_km_by_state.values()), 1e-9)
    if unobs_share > 0.5:
        warnings.append(f"{unobs_share:.0%} of road length is UNOBSERVABLE (Step 4C NOT_ASSESSED terrain): most "
                        "settlement conclusions are POSSIBLE_CUTOFF, not CONNECTED, by design")
    return ConnectivityResult(road_rule=sampler.rule, network=network, assessments=assessments,
                              segment_impacts=impacts,
                              stretch_of=stretch_of, stretches=stretches, settlement_set=sset, settlements=settlements,
                              priorities=priorities, evidence=evidence, summary=summary, warnings=warnings)


def _priorities(network: RoadNetwork, graph: Graph, assessments: list[SegmentAssessment], stretches: list[list[int]],
                comps: dict[str, np.ndarray], target_count: dict[str, Counter[int]],
                settlements: list[SettlementResult], seg_evidence: dict[int, list[str]], ctx: _EvidenceContext,
                evidence: list[EvidenceRecord]) -> list[BlockedSegmentPriority]:
    """Union-find answer to 'clear this blocked stretch only': which cut-off settlements regain a route."""
    cut: dict[str, dict[int, list[SettlementResult]]] = {"post_optimistic": {}, "post_pessimistic": {}}
    for r in settlements:
        node = r.candidate.snap_node
        if node is None:
            continue
        if r.access_state is AccessState.CONFIRMED_CUTOFF:
            cut["post_optimistic"].setdefault(int(comps["post_optimistic"][node]), []).append(r)
        if r.access_state in (AccessState.CONFIRMED_CUTOFF, AccessState.POSSIBLE_CUTOFF):
            cut["post_pessimistic"].setdefault(int(comps["post_pessimistic"][node]), []).append(r)
    in_set = Counter(sid for r in settlements for sid in r.clearing_stretch_ids)

    def gained(name: str, members: list[int]) -> list[SettlementResult]:
        c = comps[name]
        touched = sorted({int(c[graph.u[e]]) for e in members} | {int(c[graph.v[e]]) for e in members})
        if len(touched) < 2:
            return []
        total = sum(target_count[name][x] for x in touched)
        out = []
        if total == 0:
            return []
        out = [r for x in touched for r in cut[name].get(x, [])]
        return sorted(out, key=lambda r: r.candidate.settlement_id)

    rows: list[BlockedSegmentPriority] = []
    for k, members in enumerate(stretches):
        sid = f"blk-{k}"
        segs = [network.segments[e] for e in members]
        way_ids = sorted({network.edges[s.edge].way.osm_id for s in segs})
        tags = [network.ways[w].tags for w in way_ids]
        opt = gained("post_optimistic", members)
        pess = gained("post_pessimistic", members)
        ev_ids = sorted({i for e in members for i in seg_evidence[e]})
        if opt or pess:
            n_b = sum(r.candidate.building_count for r in opt)
            claim = (f"Clearing blocked stretch {sid} ({len(members)} segment(s) on OSM way(s) "
                     f"{', '.join(map(str, way_ids[:4]))}) alone restores a road route to an access destination for "
                     f"{len(opt)} confirmed cut-off settlement(s) with {n_b} mapped buildings, keeping UNOBSERVABLE "
                     f"roads passable; on observed-open roads only it restores {len(pess)}. Graph decision: the "
                     "stretch joins components without a destination to one with a destination.")
            coords = [p for s in segs[:1] for p in s.coords]
            ev = _evidence(ctx, SubjectType.ROAD_SEGMENT, sid, METHOD_PRIORITY, claim, _line(coords),
                           [f"way/{w}" for w in way_ids],
                           {"settlements_reconnected": len(opt), "reconnected_building_count_proxy": n_b,
                            "settlements_reconnected_on_observed_roads": len(pess), "segments": len(members)})
            evidence.append(ev)
            ev_ids = [ev.evidence_id] + ev_ids
        rows.append(BlockedSegmentPriority(
            rank=1, stretch_id=sid, segment_ids=tuple(s.segment_id for s in segs), osm_way_ids=tuple(way_ids),
            highways=tuple(sorted({tg.get("highway", "road") for tg in tags})),
            names=tuple(sorted({tg["name"] for tg in tags if tg.get("name")})),
            bridge=any(is_bridge(tg) for tg in tags), length_m=round(sum(s.length for s in segs), 3),
            affected_length_m=round(sum(assessments[e].affected_length_m for e in members), 3),
            longest_flood_run_m=round(max(assessments[e].qualifying_flood_run_m for e in members), 3),
            settlements_reconnected=len(opt), reconnected_settlement_ids=tuple(r.candidate.settlement_id for r in opt),
            reconnected_building_count=sum(r.candidate.building_count for r in opt),
            settlements_reconnected_on_observed_roads=len(pess),
            reconnected_on_observed_roads_ids=tuple(r.candidate.settlement_id for r in pess),
            in_minimum_clearing_set_of=in_set.get(sid, 0), evidence_ids=tuple(ev_ids)))
    rows.sort(key=lambda r: (-r.settlements_reconnected, -r.reconnected_building_count,
                             -r.settlements_reconnected_on_observed_roads, -r.in_minimum_clearing_set_of,
                             r.segment_ids[0]))
    return [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(rows)]


def _summary(network: RoadNetwork, assessments: list[SegmentAssessment], impacts: list[RoadSegmentImpact],
             sset: SettlementSet, settlements: list[SettlementResult], priorities: list[BlockedSegmentPriority],
             comps: dict[str, np.ndarray], target_count: dict[str, Counter[int]],
             states: np.ndarray) -> ConnectivitySummary:
    seg_counts = Counter(a.state.value for a in assessments)
    km: dict[str, float] = {}
    reason_km: dict[str, float] = {}
    for a in assessments:
        km[a.state.value] = km.get(a.state.value, 0.0) + a.length_m / 1000
        if a.samples:
            per = a.length_m / a.samples / 1000
            for k, v in a.unobservable_reasons.items():
                reason_km[k] = reason_km.get(k, 0.0) + v * per
    bridges = Counter(i.state.value for i in impacts if i.bridge is not BridgeExposure.NOT_A_BRIDGE)
    reachable = {name: sum(1 for r in settlements if getattr(r.status, name).reachable)
                 for name in ("pre_event", "post_optimistic", "post_pessimistic")}
    isolated = {}
    for name, c in comps.items():
        pockets = {int(c[r.candidate.snap_node]) for r in settlements if r.candidate.snap_node is not None}
        isolated[name] = sum(1 for p in pockets if target_count[name][p] == 0)
    return ConnectivitySummary(
        road_segments_by_state={s.value: seg_counts.get(s.value, 0) for s in STATE_ORDER},
        road_km_by_state={s.value: round(km.get(s.value, 0.0), 3) for s in STATE_ORDER},
        unobservable_km_by_reason={k: round(v, 3) for k, v in sorted(reason_km.items())},
        bridges_by_state={s.value: bridges.get(s.value, 0) for s in STATE_ORDER},
        exit_nodes=len(network.exit_nodes),
        targets_by_kind=dict(sorted(Counter(t.kind.value for t in sset.targets).items())),
        unsnapped_targets=len(sset.unsnapped_targets), settlements_total=len(settlements),
        settlements_by_source=dict(sorted(Counter(r.candidate.source.value for r in settlements).items())),
        settlements_by_access_state={a.value: sum(1 for r in settlements if r.access_state is a) for a in AccessState},
        settlements_by_status={s.value: sum(1 for r in settlements if r.status.status is s) for s in AccessStatus},
        building_count_by_access_state={a.value: sum(r.candidate.building_count for r in settlements
                                                     if r.access_state is a) for a in AccessState},
        reachable_settlements=reachable, isolated_components_with_settlements=isolated,
        blocked_segments=int((states == 0).sum()), blocked_stretches=len(priorities),
        blocked_stretches_reconnecting_alone=sum(1 for p in priorities if p.settlements_reconnected > 0))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _fc(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n",
                    encoding="utf-8")


def segment_features(result: ConnectivityResult) -> list[dict[str, Any]]:
    """Every segment; BLOCKED ones carry their stretch's 'clear this' answer (the click-a-road demo)."""
    prio = {p.stretch_id: p for p in result.priorities}
    feats = []
    for idx, (seg, a, imp) in enumerate(zip(result.network.segments, result.assessments, result.segment_impacts,
                                            strict=True)):
        way = result.network.ways[result.network.edges[seg.edge].way.osm_id]
        props = imp.model_dump(mode="json", exclude={"geometry"})
        props.update(name=way.tags.get("name"), ref=way.tags.get("ref"), u=seg.u, v=seg.v,
                     qualifying_flood_run_m=round(a.qualifying_flood_run_m, 3),
                     qualifying_hidden_run_m=round(a.qualifying_hidden_run_m, 3),
                     unobservable_samples=a.unobservable_reasons)
        if idx in result.stretch_of:
            p = prio[f"blk-{result.stretch_of[idx]}"]
            props.update(stretch_id=p.stretch_id, stretch_rank=p.rank, stretch_segment_ids=list(p.segment_ids),
                         settlements_reconnected=p.settlements_reconnected,
                         reconnected_settlement_ids=list(p.reconnected_settlement_ids),
                         reconnected_building_count=p.reconnected_building_count,
                         settlements_reconnected_on_observed_roads=p.settlements_reconnected_on_observed_roads,
                         reconnected_on_observed_roads_ids=list(p.reconnected_on_observed_roads_ids),
                         in_minimum_clearing_set_of=p.in_minimum_clearing_set_of,
                         stretch_evidence_ids=list(p.evidence_ids))
        feats.append({"type": "Feature", "geometry": imp.geometry.model_dump(mode="json"), "properties": props})
    return feats


def settlement_features(result: ConnectivityResult) -> list[dict[str, Any]]:
    feats = []
    for r in result.settlements:
        props = r.status.model_dump(mode="json", exclude={"location"})
        props.update(access_state=r.access_state.value, place_type=r.candidate.place_type, osm_id=r.candidate.osm_id,
                     building_count_note="mapped OSM buildings (proxy, not population)",
                     sample_building_ids=list(r.candidate.building_ids[:5]),
                     min_blocked_stretches_to_clear=r.min_blocked_stretches_to_clear,
                     clearing_stretch_ids=list(r.clearing_stretch_ids),
                     unobservable_segments_on_clearing_path=r.unobservable_segments_on_clearing_path,
                     exit_reachable=r.exit_reachable)
        feats.append({"type": "Feature", "geometry": r.status.location.model_dump(mode="json"), "properties": props})
    return feats


def target_features(result: ConnectivityResult) -> list[dict[str, Any]]:
    out = []
    for tg, snapped in [(x, True) for x in result.settlement_set.targets] + \
                       [(x, False) for x in result.settlement_set.unsnapped_targets]:
        out.append({"type": "Feature", "geometry": _point(tg.lon, tg.lat).model_dump(mode="json"),
                    "properties": {"target_id": tg.target_id, "kind": tg.kind.value, "name": tg.name,
                                   "snapped": snapped, "node": tg.node, "snap_distance_m": tg.snap_distance_m}})
    return out


def describe(prov: ConnectivityProvenance, result: ConnectivityResult | None = None) -> list[str]:
    s = prov.summary
    lines = [f"# Road connectivity - {prov.aoi_id} / {prov.candidate_id}", "",
             f"Event {prov.event_date} | OSM snapshot {prov.osm_snapshot.snapshot_date} (pre-event) | "
             "**NOT VALIDATED** (provisional Step 4C baseline and thresholds)", "",
             "## Roads", "", "| state | segments | km |", "|---|---|---|"]
    lines += [f"| {k.upper()} | {s.road_segments_by_state[k]:,} | {s.road_km_by_state[k]:,.1f} |"
              for k in s.road_segments_by_state]
    lines += ["", "UNOBSERVABLE sampled length by Step 4C reason (km): " +
              (", ".join(f"{k} {v:,.1f}" for k, v in s.unobservable_km_by_reason.items()) or "none"),
              f"Bridges by state: {s.bridges_by_state}", f"Exit nodes: {s.exit_nodes} | destinations: "
              f"{s.targets_by_kind} | unsnapped destinations: {s.unsnapped_targets}", "",
              "## Settlements", "", "| access state | settlements | mapped buildings (proxy, not population) |",
              "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {s.building_count_by_access_state[k]:,} |"
              for k, v in s.settlements_by_access_state.items()]
    lines += ["", f"Contract statuses: {s.settlements_by_status}", f"Sources: {s.settlements_by_source}", "",
              "## Before vs after", "", "| graph | settlements with a route | pockets with settlements but no "
              "destination |", "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {s.isolated_components_with_settlements[k]:,} |"
              for k, v in s.reachable_settlements.items()]
    lines += ["", "## Blocked stretches ranked by settlements reconnected (clearing that stretch alone)", "",
              f"{s.blocked_segments:,} BLOCKED segments in {s.blocked_stretches:,} blocked stretches; "
              f"{s.blocked_stretches_reconnecting_alone:,} stretches reconnect at least one confirmed cut-off "
              "settlement on their own.", "",
              "| rank | stretch | segments | OSM way(s) | highway | reconnected | buildings (proxy) | on observed "
              "roads | in min. clearing sets |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {p.rank} | {p.stretch_id} | {len(p.segment_ids)} | {', '.join(map(str, p.osm_way_ids[:3]))} | "
              f"{'/'.join(p.highways)}{' (bridge)' if p.bridge else ''} | {p.settlements_reconnected} | "
              f"{p.reconnected_building_count} | {p.settlements_reconnected_on_observed_roads} | "
              f"{p.in_minimum_clearing_set_of} |" for p in prov.top_blocked_segments]
    if result is not None:
        ev = {e.evidence_id: e for e in result.evidence}
        cut = sorted((r for r in result.settlements if r.access_state is AccessState.CONFIRMED_CUTOFF),
                     key=lambda r: (-r.candidate.building_count, r.candidate.settlement_id))
        poss = sorted((r for r in result.settlements if r.access_state is AccessState.POSSIBLE_CUTOFF),
                      key=lambda r: (-r.candidate.building_count, r.candidate.settlement_id))
        for title, rows, limit in (("Confirmed cut-off cards", cut, 50), ("Possible cut-off cards (largest 20)",
                                                                          poss, 20)):
            lines += ["", f"## {title}", ""]
            if not rows:
                lines.append("None.")
            for r in rows[:limit]:
                st = r.status
                own = ev.get(st.evidence_ids[0]) if st.evidence_ids else None
                lines += [f"### {_safe_name(r.candidate.name) or 'unnamed building cluster'} - {r.access_state.value}",
                          f"- settlement `{st.settlement_id}` ({r.candidate.place_type or 'building cluster'}), "
                          f"{st.building_count} mapped buildings (proxy), snapped {st.snap_distance_m} m from a road",
                          f"- before: {st.pre_event.target_kind.value if st.pre_event.target_kind else '-'} "
                          f"`{st.pre_event.target_id}` at {(st.pre_event.route_length_m or 0) / 1000:.1f} km; "
                          f"optimistic after: {'reachable' if st.post_optimistic.reachable else 'unreachable'}; "
                          f"pessimistic after: {'reachable' if st.post_pessimistic.reachable else 'unreachable'}",
                          f"- separating segments ({len(st.blocking_segment_ids)}): "
                          f"{', '.join(st.blocking_segment_ids[:8])}"
                          f"{' ...' if len(st.blocking_segment_ids) > 8 else ''}"]
                if r.min_blocked_stretches_to_clear is not None:
                    lines.append(f"- fewest blocked stretches to clear: {r.min_blocked_stretches_to_clear} "
                                 f"({', '.join(r.clearing_stretch_ids)}); UNOBSERVABLE segments on that path: "
                                 f"{r.unobservable_segments_on_clearing_path}")
                if own is not None:
                    lines.append(f"- evidence `{own.evidence_id}`: {own.claim}")
                lines.append(f"- linked segment evidence: {', '.join(st.evidence_ids[1:6]) or '-'}"
                             f"{' ...' if len(st.evidence_ids) > 6 else ''}")
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in prov.warnings] or ["- none"])
    lines += ["", "## Limitations", "",
              "- Built on the provisional Step 4C baseline: open-water flooding only; debris, landslides and "
              "damaged bridges are not detected, so OPEN means 'no flood signal on observable road', not 'passable'.",
              "- UNOBSERVABLE roads (steep / shadowed terrain, about 91 % of the scene is NOT_ASSESSED) are never "
              "called open or blocked; settlements that depend on them are POSSIBLE_CUTOFF.",
              "- OSM completeness limits everything: unmapped roads, tracks or buildings are invisible; "
              "building counts are a size proxy, not population.",
              "- Undirected graph, no road class weighting, no vehicle type; exits assume the road continues "
              "beyond the analysis box.", ""]
    return lines


def run_connectivity(job: ConnectivityJob, cfg: BatoConfig, *, overwrite: bool = False,
                     read_state: Callable[[Path], tuple[np.ndarray, tuple[float, float, float, float, float, float]]]
                     = read_state_raster, now: Callable[[], datetime] = _utc_now,
                     progress: Callable[[str], None] | None = None
                     ) -> tuple[Path, ConnectivityProvenance, ConnectivityResult]:
    """Compute and publish atomically (``<dir>.tmp-`` then rename; ``--overwrite`` keeps ``.superseded-``)."""
    if job.final_dir.exists() and not overwrite:
        raise ConnectivityError(f"output already exists: {job.final_dir} (use --overwrite; the old output is kept as "
                                "<dir>.superseded-<time>)")
    started = now()
    state, transform = read_state(job.state_path)
    if sha256_file(job.state_path) != job.baseline.outputs.state_sha256:
        raise ConnectivityError("baseline_state.tif changed during the run")
    debris_in: DebrisInput | None = getattr(job, "debris", None)
    debris = None
    if debris_in is not None:
        dstate, dtransform = read_state(debris_in.state_path)
        if tuple(round(v, 6) for v in dtransform) != tuple(round(v, 6) for v in transform):
            raise ConnectivityError("debris_state.tif and baseline_state.tif are on different grids")
        with np.load(debris_in.table_path) as npz:
            table = {c: np.asarray(npz[c]) for c in TABLE_COLUMNS}
        for f, sha in ((debris_in.state_path, debris_in.provenance.outputs.state_sha256),
                       (debris_in.table_path, debris_in.provenance.outputs.change_table_sha256)):
            if sha256_file(f) != sha:
                raise ConnectivityError(f"{f.name} changed during the run")
        debris = DebrisData(state=dstate, table=table, provenance=debris_in.provenance)
    result = compute_connectivity(job, cfg, state, transform, created_at=started, debris=debris, progress=progress)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    tmp = assert_processed_output_dir(job.final_dir.with_name(f"{job.final_dir.name}.tmp-{stamp}"),
                                      cfg.settings.paths)
    tmp.mkdir(parents=True)
    _dump(_fc(segment_features(result)), tmp / SEGMENTS_FILE)
    _dump(_fc(settlement_features(result)), tmp / SETTLEMENTS_FILE)
    _dump(_fc(target_features(result)), tmp / TARGETS_FILE)
    _dump([p.model_dump(mode="json") for p in result.priorities], tmp / PRIORITY_FILE)
    with (tmp / EVIDENCE_FILE).open("w", encoding="utf-8") as fh:
        for e in result.evidence:
            fh.write(e.model_dump_json() + "\n")
    root = job.root
    osm_prov = job.osm.provenance
    inputs = {"baseline_provenance": FileRef(path=_relative(job.baseline_provenance_path, root),
                                             sha256=job.baseline_provenance_sha256),
              "baseline_state": FileRef(path=job.baseline.outputs.state, sha256=job.baseline.outputs.state_sha256),
              "stack_provenance": FileRef(path=job.baseline.source.provenance,
                                          sha256=job.baseline.source.provenance_sha256),
              "osm_extract_provenance": FileRef(path=_relative(job.osm.provenance_path, root),
                                                sha256=job.osm.provenance_sha256)}
    inputs.update({f"osm_{r.layer.value}": FileRef(path=r.file, sha256=r.sha256) for r in osm_prov.layers})
    outputs = {name: FileRef(path=_relative(job.final_dir / name, root), sha256=sha256_file(tmp / name))
               for name in (SEGMENTS_FILE, SETTLEMENTS_FILE, TARGETS_FILE, PRIORITY_FILE, EVIDENCE_FILE)}
    if debris_in is not None:
        dp = debris_in.provenance
        inputs["debris_provenance"] = FileRef(path=_relative(debris_in.provenance_path, root),
                                              sha256=debris_in.provenance_sha256)
        inputs["debris_state"] = FileRef(path=dp.outputs.state, sha256=dp.outputs.state_sha256)
        inputs["debris_change_table"] = FileRef(path=dp.outputs.change_table, sha256=dp.outputs.change_table_sha256)
    rules = {**RULES, **(RULES_V2 if result.road_rule == ROAD_RULE_V2 else {})}
    provenance = ConnectivityProvenance(
        road_rule="v2" if result.road_rule == ROAD_RULE_V2 else "v1", bato_version=bato.__version__,
        config_fingerprint=cfg.fingerprint(),
        baseline_config_fingerprint=job.baseline.config_fingerprint, started_at=started, finished_at=now(),
        aoi_id=job.aoi_id, candidate_id=job.candidate_id, event_date=job.event_date.isoformat(),
        osm_snapshot=osm_prov.snapshot, inputs=inputs, thresholds=threshold_records(cfg), rules=rules,
        outputs=outputs, summary=result.summary, top_blocked_segments=tuple(result.priorities[:TOP_N]),
        warnings=tuple(result.warnings))
    save_contract(provenance, tmp / PROVENANCE_FILE)
    (tmp / REPORT_FILE).write_text("\n".join(describe(provenance, result)) + "\n", encoding="utf-8")
    if job.final_dir.exists():
        os.replace(job.final_dir, job.final_dir.with_name(f"{job.final_dir.name}.superseded-{stamp}"))
    os.replace(tmp, job.final_dir)
    return job.final_dir, provenance, result

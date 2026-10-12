"""Dashboard bundle: one compact JSON (+ PNG overlays and SAR chips) built read-only from the FROZEN official
connectivity run and the files it cites.

Integrity rules (the dashboard visualises existing results, it never creates new ones):

* The official run is a connectivity output with ``road_rule == "v1"``. A v2 run is refused.
* Every output file of that run must match the SHA-256 recorded in its provenance; the stack and Step 4C
  provenance it cites must match their recorded SHA-256 too.
* The summary shown is the provenance summary, and it is recomputed from ``road_segments.geojson`` and
  ``settlements.geojson``; any disagreement stops the build.
* Route traces are redrawn from the published segments and kept only when their length equals the published
  route length (bato.dashboard.routes). Other derived numbers (route km by state, road distance to the nearest
  hospital / exit, pockets of access-unverified settlements) are labelled as derived in the bundle.
* Building counts are passed through with their proxy note; no population figure exists anywhere in BATO.
* The experimental disturbance detector appears only as a methods record: its pre-registered checks are
  re-evaluated from its provenance numbers; it never changes a road state here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.dashboard import rasters as rs
from bato.dashboard.png import write_png
from bato.dashboard.routes import (
    BLOCKED,
    OPEN,
    STATE_CODE,
    UNOBSERVABLE,
    Network,
    RouteTracer,
    Trace,
    build_network,
    snap_candidates,
    unobservable_pockets,
)
from bato.data_rules import assert_pipeline_input_allowed, assert_processed_output_dir
from bato.exceptions import BatoError
from bato.impact.roads import lonlat_to_3857

BUNDLE_VERSION = "bato-dashboard-bundle-v1"
OFFICIAL_ROAD_RULE = "v1"
OUTPUT_SUBDIR = "dashboard"
BUNDLE_FILE = "bundle.json"
INDEX_FILE = "index.json"
PROVENANCE_FILE = "connectivity_provenance.json"
SEGMENTS_FILE = "road_segments.geojson"
SETTLEMENTS_FILE = "settlements.geojson"
TARGETS_FILE = "access_destinations.geojson"
PRIORITY_FILE = "blocked_segments_ranked.json"
EVIDENCE_FILE = "evidence_records.jsonl"
COORD_DECIMALS = 5
_EVIDENCE_ID = re.compile(r'"evidence_id"\s*:\s*"([^"]+)"')
KM_TOLERANCE = 0.001                        # provenance km are rounded to 3 decimals

REASON_GROUP = {"NOT_ASSESSED_LAYOVER": "radar_geometry", "NOT_ASSESSED_SHADOW": "radar_geometry",
                "NOT_ASSESSED_STEEP": "terrain_rule", "NOT_ASSESSED_INVALID_SAR": "data_gap",
                "NOT_ASSESSED_NO_DEM": "data_gap", "OUTSIDE_SAR_RASTER": "data_gap"}
REASON_GROUPS = ("radar_geometry", "terrain_rule", "data_gap", "unknown")
REASON_LABEL = {
    "radar_geometry": "Radar geometry: layover or shadow - the satellite cannot see the ground there",
    "terrain_rule": "Terrain rule: slope above the Step 4C limit - radar sees it, but the open-water flood rule "
                    "is not applied on steep ground",
    "data_gap": "Data gap: invalid SAR, no DEM, or outside the radar image",
    "unknown": "No reason recorded",
}
ROLE_CONTRIBUTES = {
    "PRE1": "Earlier pre-event reference. A flood signal must also be new against this date, which filters out "
            "changes that were already under way.",
    "PRE2": "Closest pre-event reference: the 'before' image for every change measurement.",
    "POST": "First post-event observation: the 'after' image.",
}
# Pre-registered acceptance checks of the experimental disturbance detector (docs/debris_detection.md).
DEBRIS_CONTROL_CEILING = 0.0005
DEBRIS_MIN_RATIO = 3.0
DEBRIS_MAX_FACING = 0.5


class DashboardBuildError(BatoError):
    """The official result cannot be loaded without breaking an integrity rule (nothing is written)."""


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DashboardBuildError(f"cannot read {path}: {exc}") from exc


def _need(d: Any, *keys: str | int) -> Any:
    cur = d
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError) as exc:
            raise DashboardBuildError(f"missing field {'.'.join(str(x) for x in keys)}") from exc
    return cur


def _round(v: float | None, n: int = 3) -> float | None:
    return None if v is None else round(float(v), n)


# ---------------------------------------------------------------------------
# Official run selection and verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCandidate:
    directory: Path
    provenance: dict[str, Any]
    road_rule: str
    finished_at: str


def find_connectivity_runs(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> list[RunCandidate]:
    """Every published connectivity directory for this candidate, current and superseded."""
    base = cfg.settings.paths.processed / cfg.settings.connectivity.output_subdir / aoi_id
    out: list[RunCandidate] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d.name == candidate_id or d.name.startswith(f"{candidate_id}.superseded-")):
            continue
        p = d / PROVENANCE_FILE
        if not p.is_file():
            continue
        prov = _load_json(p)
        out.append(RunCandidate(directory=d, provenance=prov, road_rule=str(prov.get("road_rule", "v1")),
                                finished_at=str(prov.get("finished_at", ""))))
    return out


def select_official_run(runs: list[RunCandidate], explicit: Path | None = None) -> tuple[RunCandidate, list[str]]:
    """The official result is road rule v1. With several v1 runs the latest finished one is used (and reported)."""
    notes: list[str] = []
    if explicit is not None:
        p = explicit / PROVENANCE_FILE
        if not p.is_file():
            raise DashboardBuildError(f"{explicit} has no {PROVENANCE_FILE}")
        prov = _load_json(p)
        run = RunCandidate(directory=explicit, provenance=prov, road_rule=str(prov.get("road_rule", "v1")),
                           finished_at=str(prov.get("finished_at", "")))
        if run.road_rule != OFFICIAL_ROAD_RULE:
            raise DashboardBuildError(f"{explicit} is a road rule {run.road_rule} run; the official result is road "
                                      f"rule {OFFICIAL_ROAD_RULE} (the experimental detector was rejected)")
        return run, notes
    official = [r for r in runs if r.road_rule == OFFICIAL_ROAD_RULE]
    skipped = [r for r in runs if r.road_rule != OFFICIAL_ROAD_RULE]
    for r in skipped:
        notes.append(f"ignored {r.directory.name}: road rule {r.road_rule} (not the official result)")
    if not official:
        raise DashboardBuildError("no road rule v1 connectivity run found (run scripts/run_connectivity.py "
                                  "--road-rule v1, or pass --connectivity-dir)")
    official.sort(key=lambda r: r.finished_at)
    chosen = official[-1]
    for r in official[:-1]:
        notes.append(f"older v1 run not used: {r.directory.name} (finished {r.finished_at})")
    return chosen, notes


def verify_run_outputs(run: RunCandidate) -> dict[str, Path]:
    outputs = _need(run.provenance, "outputs")
    files: dict[str, Path] = {}
    for name, ref in outputs.items():
        path = run.directory / Path(_need(ref, "path")).name
        if not path.is_file():
            raise DashboardBuildError(f"{path} is missing")
        if sha256_file(path) != _need(ref, "sha256"):
            raise DashboardBuildError(f"{path.name} does not match the SHA-256 in its provenance (changed after "
                                      "the run); refusing to show it")
        files[name] = path
    for need in (SEGMENTS_FILE, SETTLEMENTS_FILE, TARGETS_FILE, PRIORITY_FILE, EVIDENCE_FILE):
        if need not in files:
            raise DashboardBuildError(f"the run provenance does not list {need}")
    return files


def verify_root_file(cfg: BatoConfig, root: Path, ref: dict[str, Any], what: str) -> Path:
    path = assert_pipeline_input_allowed(root / _need(ref, "path"), cfg.settings.paths)
    if not path.is_file():
        raise DashboardBuildError(f"{what} {ref['path']} is missing")
    if sha256_file(path) != _need(ref, "sha256"):
        raise DashboardBuildError(f"{what} {ref['path']} changed since the connectivity run")
    return path


# ---------------------------------------------------------------------------
# Parsing the published layers
# ---------------------------------------------------------------------------


@dataclass
class Segments:
    ids: list[str]
    state: np.ndarray
    u: list[int]
    v: list[int]
    length: np.ndarray
    coords: list[list[tuple[float, float]]]
    props: list[dict[str, Any]]
    index: dict[str, int] = field(default_factory=dict)


def parse_segments(fc: dict[str, Any]) -> Segments:
    feats = _need(fc, "features")
    ids, state, u, v, length, coords, props = [], [], [], [], [], [], []
    for f in feats:
        p = _need(f, "properties")
        st = str(_need(p, "state"))
        if st not in STATE_CODE:
            raise DashboardBuildError(f"unknown road state {st!r}")
        ids.append(str(_need(p, "segment_id")))
        state.append(STATE_CODE[st])
        u.append(int(_need(p, "u")))
        v.append(int(_need(p, "v")))
        length.append(float(_need(p, "length_m")))
        coords.append([(float(x), float(y)) for x, y in _need(f, "geometry", "coordinates")])
        props.append(p)
    seg = Segments(ids=ids, state=np.array(state, dtype=np.int8), u=u, v=v, length=np.array(length), coords=coords,
                   props=props)
    seg.index = {s: i for i, s in enumerate(ids)}
    if len(seg.index) != len(ids):
        raise DashboardBuildError("duplicate segment ids in road_segments.geojson")
    return seg


def recompute_summary(seg: Segments, settlements: list[dict[str, Any]]) -> dict[str, Any]:
    names = {BLOCKED: "blocked", UNOBSERVABLE: "unobservable", OPEN: "open"}
    by_state = {names[c]: int((seg.state == c).sum()) for c in (BLOCKED, UNOBSERVABLE, OPEN)}
    km = {names[c]: float(seg.length[seg.state == c].sum()) / 1000.0 for c in (BLOCKED, UNOBSERVABLE, OPEN)}
    access = Counter(str(_need(s, "properties", "access_state")) for s in settlements)
    reach = {g: sum(1 for s in settlements if _need(s, "properties", g, "reachable"))
             for g in ("pre_event", "post_optimistic", "post_pessimistic")}
    return {"road_segments_by_state": by_state, "road_km_by_state": km, "settlements_by_access_state": dict(access),
            "reachable_settlements": reach, "settlements_total": len(settlements)}


def check_summary(published: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    """Disagreements between the provenance summary and the published layers (empty = consistent)."""
    bad = []
    for k, v in recomputed["road_segments_by_state"].items():
        if int(published["road_segments_by_state"].get(k, -1)) != v:
            bad.append(f"segments {k}: provenance {published['road_segments_by_state'].get(k)} vs layer {v}")
    for k, v in recomputed["road_km_by_state"].items():
        if abs(float(published["road_km_by_state"].get(k, -1.0)) - v) > KM_TOLERANCE + 1e-7 * v:
            bad.append(f"km {k}: provenance {published['road_km_by_state'].get(k)} vs layer {v:.4f}")
    pub_access = {k: int(n) for k, n in published["settlements_by_access_state"].items() if int(n)}
    if pub_access != {k: n for k, n in recomputed["settlements_by_access_state"].items() if n}:
        bad.append(f"settlements by access state: provenance {pub_access} vs layer "
                   f"{recomputed['settlements_by_access_state']}")
    for k, v in recomputed["reachable_settlements"].items():
        if int(published["reachable_settlements"].get(k, -1)) != v:
            bad.append(f"reachable {k}: provenance {published['reachable_settlements'].get(k)} vs layer {v}")
    if int(published["settlements_total"]) != recomputed["settlements_total"]:
        bad.append("settlements_total differs")
    return bad


def load_evidence(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            m = _EVIDENCE_ID.search(line)
            if m is None or m.group(1) not in wanted:
                continue
            rec = json.loads(line)
            out[rec["evidence_id"]] = rec
    missing = wanted - set(out)
    if missing:
        raise DashboardBuildError(f"{len(missing)} referenced evidence records are missing from {path.name}, "
                                  f"e.g. {sorted(missing)[:3]}")
    return out


def reason_group(samples: dict[str, Any] | None) -> tuple[str, dict[str, int]]:
    groups: Counter[str] = Counter()
    for k, n in (samples or {}).items():
        groups[REASON_GROUP.get(k, "unknown")] += int(n)
    if not groups:
        return "unknown", {}
    top = max(REASON_GROUPS, key=lambda g: (groups.get(g, 0), -REASON_GROUPS.index(g)))
    return top, dict(groups)


# ---------------------------------------------------------------------------
# Raster access (rasterio in production, arrays in tests)
# ---------------------------------------------------------------------------


@dataclass
class RasterSource:
    """Read access to the Step 4C state raster, the DEM and the stack. Any callable may be None (skipped)."""

    state: Callable[[], np.ndarray] | None = None
    state_window: Callable[[tuple[int, int, int, int]], np.ndarray] | None = None
    stack_window: Callable[[int, tuple[int, int, int, int]], np.ndarray] | None = None
    dem_crop: Callable[[float, float, float, float],
                       tuple[np.ndarray, tuple[float, ...], float | None, tuple[float, float, float, float]] | None
                       ] | None = None
    notes: list[str] = field(default_factory=list)


def rasterio_source(state_path: Path | None, stack_path: Path | None, dem_path: Path | None) -> RasterSource:
    src = RasterSource()
    try:
        import rasterio  # noqa: F401
    except ImportError:
        src.notes.append("rasterio not installed: map overlays, hillshade and SAR chips skipped")
        return src
    if state_path is not None:
        sp = state_path
        src.state = lambda: rs.read_band(sp)
        src.state_window = lambda w: rs.read_band(sp, 1, w)
    if stack_path is not None:
        kp = stack_path
        src.stack_window = lambda band, w: rs.read_band(kp, band, w)
    if dem_path is not None:
        dp = dem_path
        src.dem_crop = lambda w, s, e, n: rs.read_dem_crop(dp, w, s, e, n)
    return src


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass
class BuildInputs:
    cfg: BatoConfig
    root: Path
    run: RunCandidate
    files: dict[str, Path]
    stack: dict[str, Any]                 # stack provenance (JSON)
    stack_sha256: str
    baseline: dict[str, Any]              # Step 4C provenance (JSON)
    baseline_sha256: str
    title: str
    rasters: RasterSource = field(default_factory=RasterSource)
    debris_event: dict[str, Any] | None = None
    debris_control: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def prepare_inputs(cfg: BatoConfig, aoi_id: str, candidate_id: str, *, title: str | None = None,
                   connectivity_dir: Path | None = None, debris_control_dir: Path | None = None,
                   with_rasters: bool = True) -> BuildInputs:
    root = cfg.config_dir.parent
    runs = find_connectivity_runs(cfg, aoi_id, candidate_id)
    run, notes = select_official_run(runs, connectivity_dir)
    prov = run.provenance
    if prov.get("aoi_id") != aoi_id or prov.get("candidate_id") != candidate_id:
        raise DashboardBuildError("the connectivity run is for a different AOI or candidate")
    if prov.get("validated") is not False:
        raise DashboardBuildError("the connectivity provenance must state validated: false")
    files = verify_run_outputs(run)
    inputs = _need(prov, "inputs")
    stack_path = verify_root_file(cfg, root, _need(inputs, "stack_provenance"), "stack provenance")
    base_path = verify_root_file(cfg, root, _need(inputs, "baseline_provenance"), "Step 4C provenance")
    stack, baseline = _load_json(stack_path), _load_json(base_path)
    src = RasterSource()
    if with_rasters:
        state_p = root / _need(inputs, "baseline_state", "path")
        state_p = state_p if state_p.is_file() and sha256_file(state_p) == inputs["baseline_state"]["sha256"] else None
        stack_tif = dem = None
        out = stack.get("output") or {}
        if out.get("stack") and (root / out["stack"]).is_file():
            stack_tif = root / out["stack"]
            if sha256_file(stack_tif) != out.get("stack_sha256"):
                notes.append("stack GeoTIFF hash differs from its provenance: SAR chips skipped")
                stack_tif = None
        if baseline.get("dem") and (root / baseline["dem"]).is_file():
            dem = root / baseline["dem"]
            if baseline.get("dem_sha256") and sha256_file(dem) != baseline["dem_sha256"]:
                notes.append("DEM hash differs from the Step 4C provenance: hillshade skipped")
                dem = None
        if state_p is None:
            notes.append("Step 4C state raster missing or changed: overlays and chips skipped")
        src = rasterio_source(state_p, stack_tif, dem)
    debris_event = debris_control = None
    ev_dir = cfg.settings.paths.interim / "detection" / aoi_id / candidate_id / "debris" / "debris_provenance.json"
    if ev_dir.is_file():
        debris_event = _load_json(ev_dir)
    if debris_control_dir is not None:
        p = debris_control_dir / "debris_provenance.json"
        if not p.is_file():
            raise DashboardBuildError(f"{p} not found")
        debris_control = _load_json(p)
    return BuildInputs(cfg=cfg, root=root, run=run, files=files, stack=stack, stack_sha256=sha256_file(stack_path),
                       baseline=baseline, baseline_sha256=sha256_file(base_path), title=title or aoi_id, rasters=src,
                       debris_event=debris_event, debris_control=debris_control, notes=notes + src.notes)


def _flat(coords: list[tuple[float, float]]) -> list[float]:
    out: list[float] = []
    for x, y in coords:
        out += [round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)]
    return out


def _centroid(seg: Segments, idx: Iterable[int]) -> tuple[float, float]:
    pts = np.array([p for i in idx for p in seg.coords[i]])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _bbox(seg: Segments, idx: Iterable[int]) -> list[float]:
    pts = np.array([p for i in idx for p in seg.coords[i]])
    return [round(float(pts[:, 0].min()), 6), round(float(pts[:, 1].min()), 6), round(float(pts[:, 0].max()), 6),
            round(float(pts[:, 1].max()), 6)]


def _trace_json(t: Trace | None) -> dict[str, Any] | None:
    if t is None:
        return None
    return {"graph": t.graph, "target_id": t.target_id, "edges": t.edges, "length_m": t.length_m,
            "km_by_state": t.km_by_state, "unobservable_segments": t.unobservable_segments,
            "blocked_segments": t.blocked_segments}


def build_bundle(inp: BuildInputs, out_dir: Path, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Assemble the bundle dict and write overlays/chips into ``out_dir`` (a staging directory)."""
    say = progress or (lambda _t: None)
    prov = inp.run.provenance
    summary = _need(prov, "summary")

    say("reading the published layers ...\n")
    seg = parse_segments(_load_json(inp.files[SEGMENTS_FILE]))
    sett_fc = _need(_load_json(inp.files[SETTLEMENTS_FILE]), "features")
    targ_fc = _need(_load_json(inp.files[TARGETS_FILE]), "features")
    ranked = _load_json(inp.files[PRIORITY_FILE])
    bad = check_summary(summary, recompute_summary(seg, sett_fc))
    if bad:
        raise DashboardBuildError("the provenance summary disagrees with the published layers: " + "; ".join(bad))

    net: Network = build_network(seg.u, seg.v, seg.length, seg.state, seg.coords)
    tracer = RouteTracer(net)

    # ---- destinations
    destinations = []
    node_of: dict[str, int] = {}
    for f in targ_fc:
        p = f["properties"]
        lon, lat = _need(f, "geometry", "coordinates")
        if p.get("node") is not None:
            node_of[str(p["target_id"])] = int(p["node"])
        destinations.append({"id": p["target_id"], "kind": p["kind"], "name": p.get("name"),
                             "snapped": bool(p.get("snapped")), "lon": round(lon, 6), "lat": round(lat, 6),
                             "snap_distance_m": _round(p.get("snap_distance_m"), 1)})
    hospitals = sorted({node_of[d["id"]] for d in destinations if d["kind"] == "hospital" and d["id"] in node_of})
    exits = sorted({node_of[d["id"]] for d in destinations if d["kind"] == "analysis_exit" and d["id"] in node_of})
    say(f"  {len(seg.ids):,} segments, {len(sett_fc):,} settlements, {len(destinations):,} destinations\n")

    # ---- settlements and verified route traces
    say("redrawing route traces (verified against the published route lengths) ...\n")
    near = {(g, kind): tracer.nearest(g, nodes) for g in ("pre_event", "post_optimistic", "post_pessimistic")
            for kind, nodes in (("hospital", hospitals), ("exit", exits))}
    settlements: list[dict[str, Any]] = []
    trace_stats: Counter[str] = Counter()
    for f in sett_fc:
        p = f["properties"]
        lon, lat = (float(c) for c in _need(f, "geometry", "coordinates"))
        cands = snap_candidates(net, lon, lat, p.get("snap_distance_m"))
        traces: dict[str, Any] = {}
        snap_node: int | None = None
        for key, graphs in (("pre", ("pre_event",)), ("post", ("post_pessimistic", "post_optimistic"))):
            g = next((x for x in graphs if p[x]["reachable"]), None)
            if g is None:
                continue
            tid = p[g]["target_id"]
            if tid not in node_of:
                trace_stats["unavailable"] += 1
                continue
            order = ([snap_node] if snap_node is not None else []) + [c for c in cands if c != snap_node]
            t = tracer.trace(g, tid, node_of[tid], float(p[g]["route_length_m"]), order)
            if t is None:
                trace_stats["unavailable"] += 1
                continue
            snap_node = t.node
            trace_stats["verified"] += 1
            traces[key] = _trace_json(t)
        distances = None
        if snap_node is not None:
            distances = {f"{kind}_{g}_km": (None if not math.isfinite(near[(g, kind)][snap_node]) else
                                            round(float(near[(g, kind)][snap_node]) / 1000.0, 3))
                         for g in ("pre_event", "post_optimistic", "post_pessimistic") for kind in ("hospital", "exit")}
        blocking = [seg.index[s] for s in p.get("blocking_segment_ids", []) if s in seg.index]
        settlements.append({
            "id": p["settlement_id"], "name": p.get("name"), "place_type": p.get("place_type"),
            "source": p.get("source"), "lon": round(lon, 6), "lat": round(lat, 6),
            "access_state": p["access_state"], "status": p["status"], "buildings": int(p["building_count"]),
            "snap_distance_m": _round(p.get("snap_distance_m"), 1),
            "routes": {g: {"reachable": bool(p[g]["reachable"]), "target_id": p[g].get("target_id"),
                           "target_kind": p[g].get("target_kind"), "length_m": _round(p[g].get("route_length_m"), 1)}
                       for g in ("pre_event", "post_optimistic", "post_pessimistic")},
            "degraded_detour_ratio": p.get("degraded_detour_ratio"),
            "blocking": blocking, "evidence_ids": list(p.get("evidence_ids", [])),
            "exit_reachable": p.get("exit_reachable"), "min_blocked_stretches_to_clear":
                p.get("min_blocked_stretches_to_clear"),
            "traces": traces, "distances": distances})
    say(f"  traces verified {trace_stats['verified']:,}, unavailable {trace_stats['unavailable']:,}\n")

    # ---- blocked stretches
    stretch_of = [-1] * len(seg.ids)
    stretches = []
    for k, st in enumerate(ranked):
        idx = [seg.index[s] for s in st["segment_ids"] if s in seg.index]
        if len(idx) != len(st["segment_ids"]):
            raise DashboardBuildError(f"stretch {st['stretch_id']} names segments missing from road_segments")
        for i in idx:
            stretch_of[i] = k
        on_routes = sorted(s["id"] for s in settlements
                           if "pre" in s["traces"] and set(s["traces"]["pre"]["edges"]) & set(idx))
        lon, lat = _centroid(seg, idx)
        stretches.append({**{key: st[key] for key in (
            "rank", "stretch_id", "segment_ids", "osm_way_ids", "highways", "names", "bridge", "length_m",
            "affected_length_m", "longest_flood_run_m", "settlements_reconnected", "reconnected_settlement_ids",
            "reconnected_building_count", "settlements_reconnected_on_observed_roads",
            "reconnected_on_observed_roads_ids", "in_minimum_clearing_set_of", "evidence_ids") if key in st},
            "segments": idx, "lon": round(lon, 6), "lat": round(lat, 6), "bbox": _bbox(seg, idx),
            "on_verified_pre_event_routes": on_routes, "chip": None})

    # ---- pockets of access-unverified settlements (operations view)
    possible = [s for s in settlements if s["access_state"] == "POSSIBLE_CUTOFF"]
    pocket_edges = sorted({e for s in possible for e in s["blocking"] if seg.state[e] == UNOBSERVABLE})
    pockets: list[dict[str, Any]] = []
    for k, grp in enumerate(unobservable_pockets(net, pocket_edges)):
        gs = set(grp)
        who = sorted(s["id"] for s in possible if gs & set(s["blocking"]))
        reasons: Counter[str] = Counter()
        for e in grp:
            for r, n in reason_group(seg.props[e].get("unobservable_samples"))[1].items():
                reasons[r] += n
        lon, lat = _centroid(seg, grp)
        pockets.append({"id": f"nap-{k + 1}", "segments": grp, "length_m": round(float(seg.length[grp].sum()), 1),
                        "settlements": who, "reason_samples": dict(reasons), "lon": round(lon, 6),
                        "lat": round(lat, 6), "bbox": _bbox(seg, grp)})
    pockets.sort(key=lambda x: (-len(x["settlements"]), -x["length_m"], x["id"]))

    # ---- road arrays
    highways: list[str] = []
    hw_idx: dict[str, int] = {}
    ways: dict[str, list[Any]] = {}
    reason: list[int] = []
    reason_km: dict[str, float] = {}
    for i, p in enumerate(seg.props):
        hw = str(p.get("highway", "road"))
        if hw not in hw_idx:
            hw_idx[hw] = len(highways)
            highways.append(hw)
        ways.setdefault(str(p["osm_way_id"]), [p.get("name"), p.get("ref"), hw])
        if seg.state[i] == UNOBSERVABLE:
            g, _ = reason_group(p.get("unobservable_samples"))
            reason.append(REASON_GROUPS.index(g))
            reason_km[g] = reason_km.get(g, 0.0) + float(seg.length[i]) / 1000.0
        else:
            reason.append(-1)
    roads = {
        "count": len(seg.ids), "ids": seg.ids, "state": seg.state.tolist(), "reason": reason,
        "length": [round(float(x), 1) for x in seg.length], "way": [int(p["osm_way_id"]) for p in seg.props],
        "hw": [hw_idx[str(p.get("highway", "road"))] for p in seg.props], "highways": highways, "ways": ways,
        "bridge": [p.get("bridge", "not_a_bridge") != "not_a_bridge" for p in seg.props],
        "affected": [_round(p.get("affected_length_m"), 1) for p in seg.props],
        "unobs_len": [_round(p.get("unobservable_length_m"), 1) for p in seg.props],
        "stretch": stretch_of,
        "evidence": [(p.get("evidence_ids") or [None])[0] if seg.state[i] != OPEN else None
                     for i, p in enumerate(seg.props)],
        "samples": {str(i): p.get("unobservable_samples") for i, p in enumerate(seg.props)
                    if seg.state[i] != OPEN and p.get("unobservable_samples")},
        "coords": [_flat(c) for c in seg.coords],
    }

    # ---- evidence records referenced by the interactive objects
    wanted = {e for i in range(len(seg.ids)) if seg.state[i] == BLOCKED for e in seg.props[i].get("evidence_ids", [])}
    wanted |= {e for s in settlements if s["access_state"] in ("POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF")
               for e in s["evidence_ids"][:1]}
    wanted |= {e for st in stretches for e in st.get("evidence_ids", [])}
    say(f"reading {len(wanted):,} evidence records ...\n")
    evidence = load_evidence(inp.files[EVIDENCE_FILE], wanted)

    # ---- rasters
    overlays, chip_notes = render_rasters(inp, out_dir, stretches, say)

    total_km = float(seg.length.sum()) / 1000.0
    incident = _incident(inp, prov)
    bundle = {
        "version": BUNDLE_VERSION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bato_version": bato.__version__, "incident": incident, "timeline": _timeline(inp),
        "summary": {**summary, "total_km": round(total_km, 3),
                    "unobservable_km_by_main_reason": {g: round(reason_km.get(g, 0.0), 3) for g in REASON_GROUPS}},
        "reason_groups": [{"key": g, "label": REASON_LABEL[g]} for g in REASON_GROUPS],
        "aoi": _need(inp.stack, "aoi_polygon", "coordinates"),
        "analysis_bbox": (inp.stack.get("grid") or {}).get("analysis_bbox"),
        "roads": roads, "stretches": stretches, "settlements": settlements, "destinations": destinations,
        "pockets": pockets, "evidence": evidence, "chain": _chain(inp, prov), "methodology": _methodology(inp, prov),
        "experiment": debris_record(inp.debris_event, inp.debris_control),
        "overlays": overlays,
        "integrity": {
            "official_road_rule": OFFICIAL_ROAD_RULE, "connectivity_run": inp.run.directory.name,
            "connectivity_provenance_sha256": sha256_file(inp.run.directory / PROVENANCE_FILE),
            "outputs_verified": {n: _need(prov, "outputs", n, "sha256") for n in prov["outputs"]},
            "stack_provenance_sha256": inp.stack_sha256, "baseline_provenance_sha256": inp.baseline_sha256,
            "summary_check": "provenance summary equals the counts recomputed from the published layers",
            "route_traces": {"verified": trace_stats["verified"], "unavailable": trace_stats["unavailable"],
                             "rule": "redrawn from published segments; kept only if the length equals the "
                                     "published route length (+/- 0.5 m)"},
            "notes": inp.notes + chip_notes,
        },
    }
    return bundle


def _incident(inp: BuildInputs, prov: dict[str, Any]) -> dict[str, Any]:
    snap = _need(prov, "osm_snapshot")
    first = (inp.stack.get("inputs") or [{}])[0]
    return {"title": inp.title, "aoi_id": prov["aoi_id"], "candidate_id": prov["candidate_id"],
            "event_date": prov["event_date"], "track": inp.stack.get("track"),
            "orbit_direction": first.get("orbit_direction"), "relative_orbit": first.get("relative_orbit"),
            "osm_snapshot_date": snap.get("snapshot_date"), "osm_service": snap.get("service"),
            "road_rule": inp.run.road_rule, "validated": False,
            "validation_note": prov.get("validation_note", "NOT VALIDATED"),
            "connectivity_finished_at": prov.get("finished_at"), "warnings": list(prov.get("warnings", []))}


def _timeline(inp: BuildInputs) -> list[dict[str, Any]]:
    out = []
    for i in _need(inp.stack, "inputs"):
        out.append({"role": i["role"], "acquired": i["acquisition_start"], "product": i["product_name"],
                    "orbit_direction": i.get("orbit_direction"), "relative_orbit": i.get("relative_orbit"),
                    "days_from_event": _round(i.get("days_from_event"), 2),
                    "contributes": ROLE_CONTRIBUTES.get(i["role"], "")})
    return sorted(out, key=lambda x: x["acquired"])


def _thresholds(records: Iterable[dict[str, Any]], prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"name": r["name"], "value": r["value"], "unit": r.get("unit"), "provisional": r.get("provisional")}
            for r in records if str(r.get("name", "")).startswith(prefixes)]


def _chain(inp: BuildInputs, prov: dict[str, Any]) -> list[dict[str, Any]]:
    b, s = inp.baseline, prov["summary"]
    stats = b.get("statistics") or {}
    src = b.get("source") or {}
    rules = prov.get("rules") or {}
    return [
        {"key": "satellite", "title": "Sentinel-1 SAR", "kind": "measured",
         "status": "Three acquisitions from one relative orbit (same viewing geometry on every date)",
         "rule": "Radiometric calibration, Lee Sigma speckle filter, terrain correction to a common 10 m grid",
         "sources": [i["product_name"] for i in inp.stack.get("inputs", [])],
         "hashes": {"stack": src.get("stack_sha256"), "stack provenance": inp.stack_sha256}},
        {"key": "change", "title": "SAR change", "kind": "derived",
         "status": "POST compared with PRE-2 and PRE-1, VV and VH", "rule": b.get("rule"),
         "thresholds": _thresholds(b.get("thresholds", []), ("baseline_flood",)),
         "hashes": {"Step 4C provenance": inp.baseline_sha256}},
        {"key": "flood", "title": "Flood state", "kind": "rule output, not validated",
         "status": f"{stats.get('flood_pixels', 0):,} FLOOD pixels of {stats.get('assessed_pixels', 0):,} assessed; "
                   f"{(stats.get('not_assessed_fraction_of_all') or 0) * 100:.1f} % of the scene NOT ASSESSED",
         "rule": "Gates applied first: invalid SAR, no DEM, layover, shadow, steep slope -> NOT ASSESSED; then the "
                 "darkening rule; clusters below the minimum size removed",
         "thresholds": _thresholds(b.get("thresholds", []), ("detection", "visibility")),
         "hashes": {"Step 4C state": (b.get("outputs") or {}).get("state_sha256")}},
        {"key": "road", "title": "Road impact", "kind": "rule output, not validated",
         "status": ", ".join(f"{k} {v:,} segments" for k, v in s["road_segments_by_state"].items()),
         "rule": f"{rules.get('sample', '')} {rules.get('segment_state', '')}".strip(),
         "thresholds": _thresholds(prov.get("thresholds", []), ("road_impact",)),
         "hashes": {"road_segments.geojson": _need(prov, "outputs", SEGMENTS_FILE, "sha256")}},
        {"key": "connectivity", "title": "Connectivity", "kind": "graph computation",
         "status": ", ".join(f"{k.replace('_', ' ')} {v:,} reachable"
                             for k, v in s["reachable_settlements"].items()),
         "rule": f"{rules.get('graphs', '')} {rules.get('destinations', '')}".strip(),
         "thresholds": _thresholds(prov.get("thresholds", []), ("connectivity",)),
         "hashes": {"OSM extract provenance": _need(prov, "inputs", "osm_extract_provenance", "sha256")}},
        {"key": "settlement", "title": "Settlement status", "kind": "decision table",
         "status": ", ".join(f"{k} {v:,}" for k, v in s["settlements_by_access_state"].items()),
         "rule": rules.get("status", ""),
         "hashes": {"settlements.geojson": _need(prov, "outputs", SETTLEMENTS_FILE, "sha256")}},
    ]


def _methodology(inp: BuildInputs, prov: dict[str, Any]) -> dict[str, Any]:
    return {
        "versions": {"stack layout": inp.stack.get("layout_version"), "flood baseline": inp.baseline.get("version"),
                     "connectivity": prov.get("version"), "road rule": inp.run.road_rule,
                     "BATO at connectivity run": prov.get("bato_version")},
        "inputs_not_used": sorted(set(prov.get("inputs_not_used", [])) | set(inp.baseline.get("inputs_not_used", []))),
        "validation_notes": [prov.get("validation_note"), inp.baseline.get("validation_note")],
        "rules": prov.get("rules", {}),
        "config_fingerprints": {"connectivity": prov.get("config_fingerprint"),
                                "Step 4C": inp.baseline.get("config_fingerprint")},
    }


def debris_record(event: dict[str, Any] | None, control: dict[str, Any] | None) -> dict[str, Any]:
    """Methods record of the experimental disturbance detector, re-evaluating its pre-registered checks."""
    rec: dict[str, Any] = {"used_in_official_result": False, "event": None, "control": None, "checks": [],
                           "verdict": None}
    if event is None:
        return rec

    def stats(p: dict[str, Any]) -> dict[str, Any]:
        s = p["statistics"]
        return {"aoi_id": p["aoi_id"], "candidate_id": p["candidate_id"], "z": p.get("z"),
                "disturbance_pixels": s["disturbance_pixels"], "fraction": s["disturbance_fraction_of_assessed"],
                "facing_share": s.get("facing_sensor_share"), "edge_share": s.get("edge_pixels_share"),
                "assessed_edge_share": s.get("assessed_edge_share")}

    rec["event"] = stats(event)
    if control is None:
        rec["verdict"] = "not evaluated here (no control run supplied)"
        return rec
    rec["control"] = c = stats(control)
    e = rec["event"]
    checks = [("a", "control disturbance <= 0.05 % of assessed", c["fraction"] is not None and
               c["fraction"] <= DEBRIS_CONTROL_CEILING, f"control {c['fraction'] * 100:.4f} %"),
              ("b", "event / control >= 3", bool(c["fraction"]) and e["fraction"] / c["fraction"] >= DEBRIS_MIN_RATIO,
               f"{e['fraction'] / c['fraction']:.1f}x" if c["fraction"] else "control 0")]
    for name, r in (("event", e), ("control", c)):
        checks.append(("c", f"facing-sensor share <= 50 % ({name})",
                       r["facing_share"] is None or r["facing_share"] <= DEBRIS_MAX_FACING, f"{r['facing_share']}"))
    for name, r in (("event", e), ("control", c)):
        ok = r["edge_share"] is None or r["assessed_edge_share"] is None or r["edge_share"] <= r["assessed_edge_share"]
        checks.append(("d", f"no edge concentration ({name})", ok,
                       f"border share {r['edge_share']} vs assessed {r['assessed_edge_share']}"))
    rec["checks"] = [{"criterion": k, "test": t, "passed": bool(ok), "value": v} for k, t, ok, v in checks]
    failed = [x for x in rec["checks"] if not x["passed"]]
    rec["verdict"] = ("rejected: " + "; ".join(f"({x['criterion']}) {x['test']}" for x in failed)) if failed else \
        "all pre-registered checks passed"
    return rec


def render_rasters(inp: BuildInputs, out_dir: Path, stretches: list[dict[str, Any]],
                   say: Callable[[str], None]) -> tuple[dict[str, Any], list[str]]:
    src = inp.rasters
    overlays: dict[str, Any] = {}
    notes: list[str] = []
    grid = inp.stack.get("grid") or {}
    transform = grid.get("transform")
    if src.state is not None and transform:
        say("rendering Step 4C overlays ...\n")
        state = src.state()
        factor = rs.block_factor(state.shape[0], state.shape[1], rs.MAX_OVERLAY_PX)
        flood, na = rs.evidence_overlays(state, factor)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        write_png(out_dir / "overlays" / "flood.png", flood)
        write_png(out_dir / "overlays" / "not_assessed.png", na)
        h, w = state.shape
        place = rs.placement_3857(tuple(transform), h, w, out_w=flood.shape[1], out_h=flood.shape[0]).to_json()
        overlays["flood"] = {"file": "overlays/flood.png", **place, "block_px": factor}
        overlays["not_assessed"] = {"file": "overlays/not_assessed.png", **place, "block_px": factor}
    if src.dem_crop is not None and transform:
        scene = rs.placement_3857(tuple(transform), int(grid["height"]), int(grid["width"]))
        (w_, n_), (e_, s_) = scene.corners[0], scene.corners[2]
        crop = src.dem_crop(w_, s_, e_, n_)
        if crop is None:
            notes.append("DEM not in EPSG:4326 or outside the scene: hillshade skipped")
        else:
            say("rendering hillshade ...\n")
            dem, t, nodata, (bw, bs, be, bn) = crop
            img = rs.shade_rgba(rs.to_mercator_rows(rs.hillshade(dem, t, nodata), bn, bs, rs.MAX_HILLSHADE_PX))
            (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
            write_png(out_dir / "overlays" / "hillshade.png", img)
            overlays["hillshade"] = {"file": "overlays/hillshade.png", "corners": [[bw, bn], [be, bn], [be, bs],
                                                                                    [bw, bs]],
                                     "width": int(img.shape[1]), "height": int(img.shape[0])}
    if src.stack_window is not None and src.state_window is not None and transform:
        say(f"cutting SAR chips for {len(stretches)} blocked stretches ...\n")
        shape = (int(grid["height"]), int(grid["width"]))
        (out_dir / "chips").mkdir(parents=True, exist_ok=True)
        for st in stretches:
            x0, y0 = lonlat_to_3857(np.array([st["bbox"][0]]), np.array([st["bbox"][1]]))
            x1, y1 = lonlat_to_3857(np.array([st["bbox"][2]]), np.array([st["bbox"][3]]))
            extent_px = max(float(x1[0] - x0[0]), float(y1[0] - y0[0])) / abs(float(transform[0]))
            half = int(min(rs.CHIP_HALF_PX, max(64, extent_px * 0.9 + 40)))
            win = rs.chip_window(tuple(transform), shape, st["lon"], st["lat"], half)
            if win is None:
                continue
            files = {}
            vv = {role: src.stack_window(band, win).astype(np.float32)
                  for role, band in (("PRE1", 1), ("PRE2", 3), ("POST", 5))}
            for role, arr in vv.items():
                name = f"chips/{st['stretch_id']}_{role.lower()}_vv.png"
                write_png(out_dir / name, rs.vv_to_gray(arr))
                files[role] = name
            name = f"chips/{st['stretch_id']}_change.png"
            write_png(out_dir / name, rs.change_to_rgba(vv["POST"], vv["PRE2"]))
            files["change"] = name
            name = f"chips/{st['stretch_id']}_state.png"
            write_png(out_dir / name, rs.state_chip(src.state_window(win)))
            files["state"] = name
            r0, c0, rows, cols = win
            st["chip"] = {"files": files, **rs.placement_3857(tuple(transform), rows, cols, r0, c0).to_json(),
                          "vv_db_range": list(rs.VV_DB_RANGE), "change_db_range": rs.CHANGE_DB_RANGE}
    return overlays, notes


def publish(cfg: BatoConfig, aoi_id: str, candidate_id: str, inp: BuildInputs,
            progress: Callable[[str], None] | None = None, now: Callable[[], datetime] | None = None) -> Path:
    """Build into ``<processed>/dashboard/<aoi>/<candidate>.tmp-*`` and swap in atomically (old kept as
    ``.superseded-*``). Also refresh ``<processed>/dashboard/index.json``."""
    for part in (aoi_id, candidate_id):
        if not part or any(c in part for c in "/\\:") or part in (".", ".."):
            raise DashboardBuildError(f"invalid path component {part!r}")
    base = assert_processed_output_dir(cfg.settings.paths.processed / OUTPUT_SUBDIR, cfg.settings.paths)
    final = base / aoi_id / candidate_id
    stamp = (now or (lambda: datetime.now(timezone.utc)))().strftime("%Y%m%dT%H%M%SZ")
    tmp = final.with_name(f"{candidate_id}.tmp-{stamp}")
    tmp.mkdir(parents=True)
    bundle = build_bundle(inp, tmp, progress)
    (tmp / BUNDLE_FILE).write_text(json.dumps(bundle, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    if final.exists():
        os.replace(final, final.with_name(f"{candidate_id}.superseded-{stamp}"))
    os.replace(tmp, final)
    write_index(base)
    return final


def write_index(base: Path) -> list[dict[str, Any]]:
    entries = []
    for b in sorted(base.glob(f"*/*/{BUNDLE_FILE}")):
        if ".tmp-" in b.parent.name or ".superseded-" in b.parent.name:
            continue
        try:
            inc = json.loads(b.read_text(encoding="utf-8"))["incident"]
        except (OSError, ValueError, KeyError):
            continue
        entries.append({"path": b.parent.relative_to(base).as_posix(), "title": inc["title"],
                        "aoi_id": inc["aoi_id"], "candidate_id": inc["candidate_id"], "event_date": inc["event_date"]})
    (base / INDEX_FILE).write_text(json.dumps({"bundles": entries}, indent=1), encoding="utf-8")
    return entries

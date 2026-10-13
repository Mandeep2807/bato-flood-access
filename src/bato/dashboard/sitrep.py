"""One-page situation report (SITREP), generated deterministically from the FROZEN official result.

Read-only, like the dashboard bundle (bato.dashboard.bundle), whose integrity rules it reuses unchanged:

* the official run is the latest connectivity run with ``road_rule == "v1"`` (a v2 run is refused);
* every output of that run must match the SHA-256 in its provenance, and the stack / Step 4C provenance files it
  cites must match theirs;
* the provenance summary must equal the counts recomputed from ``road_segments.geojson`` and
  ``settlements.geojson``.

On top of those, the SITREP checks the facts it prints (``load_facts``):

* blocked segments and blocked stretches: summary == layer == ``blocked_segments_ranked.json`` (the stretches
  partition the BLOCKED segments exactly);
* settlement access states: only the four known states, summing to the total, and consistent with the
  reachability counts of the three graphs (decision table);
* Step 4C statistics: assessed + not assessed == total, the NOT_ASSESSED reasons sum to the not-assessed count,
  and the published fraction matches; every threshold provisional and ``validated: false`` (the wording says so);
* event date, AOI and candidate identical across the connectivity, stack and Step 4C provenance; one track.

Any missing or inconsistent value raises ``SitrepError``; nothing is substituted. The text is a fixed template
(``render``); only the numbers, dates and identifiers come from the result files. This is deterministic
reporting, NOT an AI component and NOT an assessment of current conditions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone, tzinfo
from pathlib import Path
from typing import Any

from bato.attribution import (
    ATTRIBUTION_COPERNICUS_DEM,
    ATTRIBUTION_OSM,
    ATTRIBUTION_SENTINEL,
    CREDIT_COPERNICUS_EMS,
)
from bato.config import BatoConfig
from bato.dashboard.bundle import (
    OFFICIAL_ROAD_RULE,
    PRIORITY_FILE,
    PROVENANCE_FILE,
    SEGMENTS_FILE,
    SETTLEMENTS_FILE,
    DashboardBuildError,
    _load_json,
    check_summary,
    parse_segments,
    prepare_inputs,
    recompute_summary,
    sha256_file,
)
from bato.dashboard.routes import BLOCKED
from bato.exceptions import BatoError

SITREP_VERSION = "bato-sitrep-v1"
DEFAULT_OUTPUT = Path("reports") / "BATO_SITREP.md"

ACCESS_STATES = ("CONNECTED", "POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF", "NO_PRE_EVENT_ACCESS")
ROAD_STATES = ("blocked", "open", "unobservable")
NOT_ASSESSED_LABEL = {
    "NOT_ASSESSED_STEEP": "steep slopes",
    "NOT_ASSESSED_LAYOVER": "layover",
    "NOT_ASSESSED_SHADOW": "shadow",
    "NOT_ASSESSED_INVALID_SAR": "invalid SAR data",
    "NOT_ASSESSED_NO_DEM": "no DEM",
}
ROLES = ("PRE1", "PRE2", "POST")
_MISSION = re.compile(r"(?<![A-Za-z0-9])S1([A-D])(?![A-Za-z0-9])")
FRACTION_TOLERANCE = 1e-6

# Fixed statements (not computed here). The EMSR927 comparison is evaluation-only: by the data rules,
# presentation code never reads data/reference, data/eval or the evaluation protocol, so this sentence restates
# the frozen protocol bato-emsr927-v1 and section 7 of reports/BATO_Technical_Report.md. It is printed ONLY for
# the run that protocol evaluated; any other run gets NO_REFERENCE_STATEMENT instead.
EMSR927_STATEMENT = (
    "**EMSR927:** **evaluation only**; no flood class, so **flood precision, recall, F1 and IoU are not "
    "calculable**. BATO missed most EMSR927-damaged road: the blocked list is incomplete.")
REFERENCE_STATEMENTS: dict[tuple[str, str], str] = {
    ("aoi-e893707c2beb", "ts-80841ae62417"): EMSR927_STATEMENT,     # Trishuli, protocol bato-emsr927-v1
}
NO_REFERENCE_STATEMENT = "**Reference comparison:** none for this run; no accuracy figure exists."
STEP4C_STATEMENT = (
    "**Step 4C:** provisional thresholds, **not validated**; open-water darkening only (misses debris, "
    "landslides, structural damage).")
NOT_ESTIMATED_STATEMENT = "**Not estimated:** population, casualties and forecasts."
STATE_DEFINITIONS = (
    "**Observed open** = no flood signal detected, passability not guaranteed. **Unobservable** = unknown, not "
    "open. **No pre-event road access** = no mapped road within {snap} m (OSM gap, not flood).")


class SitrepError(BatoError):
    """A value the SITREP needs is missing or inconsistent in the official result (nothing is written)."""


@dataclass(frozen=True)
class Pass:
    role: str
    acquired: datetime
    product_name: str


@dataclass(frozen=True)
class SitrepFacts:
    """Everything the SITREP prints that comes from the result files (all checked in ``load_facts``)."""

    aoi_id: str
    candidate_id: str
    road_rule: str
    event_date: str
    aoi_bbox: tuple[float, float, float, float]            # min_lon, min_lat, max_lon, max_lat
    analysis_buffer_km: float
    mission: str                                             # e.g. "Sentinel-1D"
    orbit_direction: str
    relative_orbit: int
    passes: tuple[Pass, Pass, Pass]                          # PRE1, PRE2, POST
    osm_snapshot_date: str
    osm_service: str
    road_segments: dict[str, int]                            # blocked / open / unobservable
    road_km: dict[str, float]                                # as published (3 decimals), checked vs the layer
    blocked_stretches: int
    settlements_total: int
    access: dict[str, int]                                   # the four ACCESS_STATES
    not_assessed_fraction: float
    not_assessed_reasons: tuple[tuple[str, int], ...]        # (state name, pixels), largest first, non-zero
    settlement_snap_max_m: float
    connectivity_run: str
    connectivity_provenance_sha256: str
    connectivity_finished_at: str

    @property
    def post(self) -> Pass:
        return self.passes[2]


# ---------------------------------------------------------------------------
# Loading and checking
# ---------------------------------------------------------------------------


def _req(d: Any, *keys: str, src: str = "") -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            where = f" from the {src}" if src else ""
            raise SitrepError(f"required field {'.'.join(keys)} is missing{where}")
        cur = cur[k]
    return cur


def _mission(names: list[str]) -> str:
    letters = set()
    for n in names:
        m = _MISSION.search(n)
        if m is None:
            raise SitrepError(f"cannot tell the Sentinel-1 unit from the product name {n!r}")
        letters.add(m.group(1))
    if len(letters) != 1:
        raise SitrepError(f"the stack mixes Sentinel-1 units {sorted(letters)}")
    return f"Sentinel-1{letters.pop()}"


def _parse_time(v: Any, what: str) -> datetime:
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SitrepError(f"{what} {v!r} is not an ISO timestamp") from exc
    if t.tzinfo is None:
        raise SitrepError(f"{what} {v!r} has no time zone")
    return t.astimezone(timezone.utc)


def _threshold_value(records: list[dict[str, Any]], name: str) -> float:
    for r in records:
        if r.get("name") == name:
            return float(_req(r, "value"))
    raise SitrepError(f"threshold {name} is not recorded in the connectivity provenance")


def load_facts(cfg: BatoConfig, aoi_id: str, candidate_id: str, *,
               connectivity_dir: Path | None = None) -> tuple[SitrepFacts, list[str]]:
    """Read and cross-check the official road-rule v1 result. Raises SitrepError on any gap or mismatch."""
    try:
        inp = prepare_inputs(cfg, aoi_id, candidate_id, connectivity_dir=connectivity_dir, with_rasters=False)
    except DashboardBuildError as exc:
        raise SitrepError(str(exc)) from exc
    prov, stack, base = inp.run.provenance, inp.stack, inp.baseline
    if inp.run.road_rule != OFFICIAL_ROAD_RULE:                       # prepare_inputs already refuses; belt
        raise SitrepError(f"road rule {inp.run.road_rule} is not the official rule {OFFICIAL_ROAD_RULE}")

    # ---- the published layers agree with the provenance summary (dashboard rule, unchanged)
    summary = _req(prov, "summary", src="connectivity provenance")
    try:
        seg = parse_segments(_load_json(inp.files[SEGMENTS_FILE]))
        settlements = _req(_load_json(inp.files[SETTLEMENTS_FILE]), "features")
        ranked = _load_json(inp.files[PRIORITY_FILE])
        bad = check_summary(summary, recompute_summary(seg, settlements))
    except (DashboardBuildError, KeyError, TypeError, ValueError) as exc:
        raise SitrepError(f"cannot read the published layers: {exc}") from exc
    if bad:
        raise SitrepError("the provenance summary disagrees with the published layers: " + "; ".join(bad))

    # ---- roads
    by_state = {k: int(_req(summary, "road_segments_by_state", k)) for k in ROAD_STATES}
    km = {k: float(_req(summary, "road_km_by_state", k)) for k in ROAD_STATES}
    blocked_ids = {seg.ids[i] for i in range(len(seg.ids)) if seg.state[i] == BLOCKED}
    if int(_req(summary, "blocked_segments")) != len(blocked_ids) or by_state["blocked"] != len(blocked_ids):
        raise SitrepError("blocked segment counts disagree between the summary and road_segments.geojson")
    if not isinstance(ranked, list):
        raise SitrepError(f"{PRIORITY_FILE} is not a list of blocked stretches")
    in_stretches: list[str] = [str(s) for st in ranked for s in _req(st, "segment_ids")]
    if len(in_stretches) != len(set(in_stretches)) or set(in_stretches) != blocked_ids:
        raise SitrepError(f"the blocked stretches in {PRIORITY_FILE} do not partition the BLOCKED segments")
    stretches = int(_req(summary, "blocked_stretches"))
    if stretches != len(ranked):
        raise SitrepError(f"summary says {stretches} blocked stretches, {PRIORITY_FILE} lists {len(ranked)}")

    # ---- settlements (decision table: pre-event / optimistic / pessimistic reachability)
    access_pub = {str(k): int(v) for k, v in _req(summary, "settlements_by_access_state").items()}
    unknown = sorted(set(access_pub) - set(ACCESS_STATES))
    if unknown:
        raise SitrepError(f"unknown settlement access states {unknown}")
    access = {k: access_pub.get(k, 0) for k in ACCESS_STATES}
    total = int(_req(summary, "settlements_total"))
    reach = {k: int(_req(summary, "reachable_settlements", k))
             for k in ("pre_event", "post_optimistic", "post_pessimistic")}
    if sum(access.values()) != total:
        raise SitrepError("settlement access states do not sum to settlements_total")
    if (reach["pre_event"] != total - access["NO_PRE_EVENT_ACCESS"]
            or reach["post_optimistic"] != access["CONNECTED"] + access["POSSIBLE_CUTOFF"]
            or reach["post_pessimistic"] != access["CONNECTED"]):
        raise SitrepError(f"access states {access} are inconsistent with the reachable counts {reach}")

    # ---- Step 4C (wording: provisional, not validated; fraction and reasons from its statistics)
    if base.get("validated") is not False or prov.get("validated") is not False:
        raise SitrepError("the Step 4C and connectivity provenance must both state validated: false")
    thresholds = _req(base, "thresholds", src="Step 4C provenance")
    if not thresholds or any(t.get("provisional") is not True for t in thresholds):
        raise SitrepError("the SITREP says 'provisional thresholds', but the Step 4C provenance records a "
                          "non-provisional threshold; update the template before reporting")
    sp = "Step 4C provenance"
    total_px, assessed_px, na_px = (int(_req(base, "statistics", k, src=sp)) for k in
                                    ("total_pixels", "assessed_pixels", "not_assessed_pixels"))
    fraction = float(_req(base, "statistics", "not_assessed_fraction_of_all", src=sp))
    counts = {str(k): int(v) for k, v in _req(base, "statistics", "state_counts", src=sp).items()}
    reasons = {k: counts.get(k, 0) for k in NOT_ASSESSED_LABEL}
    if assessed_px + na_px != total_px or total_px <= 0:
        raise SitrepError("Step 4C statistics: assessed + not assessed pixels != total pixels")
    if sum(reasons.values()) != na_px:
        raise SitrepError("Step 4C statistics: NOT_ASSESSED reason counts do not sum to not_assessed_pixels")
    if abs(na_px / total_px - fraction) > FRACTION_TOLERANCE:
        raise SitrepError("Step 4C statistics: not_assessed_fraction_of_all disagrees with the pixel counts")
    reason_list = tuple(sorted(((k, n) for k, n in reasons.items() if n > 0), key=lambda kv: (-kv[1], kv[0])))

    # ---- identity and dates across the three provenance files
    for name, p in (("stack", stack), ("Step 4C", base)):
        for key in ("aoi_id", "candidate_id", "event_date"):
            if key in p and str(p[key]) != str(_req(prov, key)):
                raise SitrepError(f"{name} provenance {key} {p[key]!r} differs from the connectivity run")
    event = str(_req(prov, "event_date"))
    try:
        event_day = date.fromisoformat(event)
    except ValueError as exc:
        raise SitrepError(f"event_date {event!r} is not an ISO date") from exc
    inputs = _req(stack, "inputs", src="stack provenance")
    if [str(_req(i, "role")) for i in inputs] != list(ROLES):
        raise SitrepError("the stack provenance must list PRE1, PRE2, POST")
    passes = tuple(Pass(role=str(i["role"]), acquired=_parse_time(_req(i, "acquisition_start"), i["role"]),
                        product_name=str(_req(i, "product_name"))) for i in inputs)
    if not passes[0].acquired < passes[1].acquired < passes[2].acquired:
        raise SitrepError("stack acquisitions are not in PRE1 < PRE2 < POST order")
    if passes[2].acquired.date() < event_day:
        raise SitrepError("the POST acquisition is before the event date")
    tracks = {(str(_req(i, "orbit_direction")), int(_req(i, "relative_orbit"))) for i in inputs}
    if len(tracks) != 1:
        raise SitrepError(f"the stack mixes tracks {sorted(tracks)}")
    (direction, rel_orbit), = tracks
    bbox = tuple(float(_req(stack, "grid", "aoi_bbox", k, src="stack provenance"))
                 for k in ("min_lon", "min_lat", "max_lon", "max_lat"))
    snap = _req(prov, "osm_snapshot")
    snapshot = str(_req(snap, "snapshot_date"))
    if snapshot >= event:
        raise SitrepError("the OSM snapshot is not before the event date")

    facts = SitrepFacts(
        aoi_id=aoi_id, candidate_id=candidate_id, road_rule=inp.run.road_rule, event_date=event,
        aoi_bbox=bbox,  # type: ignore[arg-type]
        analysis_buffer_km=float(_req(stack, "grid", "analysis_buffer_km", src="stack provenance")),
        mission=_mission([p.product_name for p in passes]), orbit_direction=direction, relative_orbit=rel_orbit,
        passes=passes, osm_snapshot_date=snapshot, osm_service=str(_req(snap, "service")),  # type: ignore[arg-type]
        road_segments=by_state, road_km=km, blocked_stretches=stretches, settlements_total=total, access=access,
        not_assessed_fraction=fraction, not_assessed_reasons=reason_list,
        settlement_snap_max_m=_threshold_value(list(_req(prov, "thresholds")), "connectivity.settlement_snap_max_m"),
        connectivity_run=inp.run.directory.name,
        connectivity_provenance_sha256=sha256_file(inp.run.directory / PROVENANCE_FILE),
        connectivity_finished_at=str(_req(prov, "finished_at")))
    return facts, list(inp.notes)


# ---------------------------------------------------------------------------
# Rendering (fixed template)
# ---------------------------------------------------------------------------


def _n(count: int, singular: str, plural: str) -> str:
    return f"{count:,} {singular if count == 1 else plural}"


def _be(count: int) -> str:
    return "is" if count == 1 else "are"


def _utc(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _local(t: datetime, tz: tzinfo | None) -> str:
    if tz is None:
        return ""
    lt = t.astimezone(tz)
    name = lt.tzname() or ""
    if name[:1] in "+-" and len(name) == 5:                       # e.g. "+0545" -> "UTC+05:45"
        name = f"UTC{name[:3]}:{name[3:]}"
    return f" ({lt.strftime('%H:%M')} {name})".rstrip()


def _bbox_text(b: tuple[float, float, float, float]) -> str:
    lon = f"{abs(b[0]):g}–{abs(b[2]):g}° {'E' if b[0] >= 0 else 'W'}"
    lat = f"{abs(b[1]):g}–{abs(b[3]):g}° {'N' if b[1] >= 0 else 'S'}"
    return f"{lon}, {lat}"


def render(facts: SitrepFacts, *, title: str, incident: str, issued_at: datetime, sitrep_number: int = 1,
           prepared_by: str | None = None, local_tz: tzinfo | None = None) -> str:
    """The one-page SITREP Markdown. Deterministic for given facts and arguments."""
    if issued_at.tzinfo is None:
        raise SitrepError("the issue time must carry a time zone")
    if not title.strip() or not incident.strip():
        raise SitrepError("title and incident description are required")
    post = facts.post
    if issued_at < post.acquired:
        raise SitrepError("the issue time is before the observation (POST) time")
    a, km, segs = facts.access, facts.road_km, facts.road_segments
    blocked, stretches = segs["blocked"], facts.blocked_stretches
    connected, unverified = a["CONNECTED"], a["POSSIBLE_CUTOFF"]
    confirmed, no_road = a["CONFIRMED_CUTOFF"], a["NO_PRE_EVENT_ACCESS"]
    age_days = (issued_at - post.acquired).days
    reasons = ", ".join(NOT_ASSESSED_LABEL[k] for k, _n_px in facts.not_assessed_reasons) or "none recorded"
    total_km = sum(km.values())
    unobs_scope = "most roads are unobservable" if km["unobservable"] > total_km / 2 else \
        "part of the road network is unobservable"
    pre1, pre2 = facts.passes[0], facts.passes[1]

    team = f"Prepared by {prepared_by.strip()} · " if prepared_by and prepared_by.strip() else ""
    reference = REFERENCE_STATEMENTS.get((facts.aoi_id, facts.candidate_id))
    ems_credit = f"Evaluation reference only: {CREDIT_COPERNICUS_EMS} (EMSR927)." if reference else ""
    lines = [
        f"# BATO SITREP {sitrep_number:02d} — {title.strip()}",
        "",
        f"**Issued** {_utc(issued_at)}{_local(issued_at, local_tz)} · **Observed** {_utc(post.acquired)} "
        f"(POST pass) · **NOT VALIDATED**",
        "",
        f"**Incident:** {incident.strip()} · **Event:** {facts.event_date} · **Area:** {_bbox_text(facts.aoi_bbox)} "
        f"(roads within a {facts.analysis_buffer_km:g} km buffer) · **Data:** {facts.mission} "
        f"{facts.orbit_direction.lower()}, orbit {facts.relative_orbit} — PRE-1 {pre1.acquired.date().isoformat()}, "
        f"PRE-2 {pre2.acquired.date().isoformat()}, **POST {post.acquired.strftime('%Y-%m-%d %H:%M:%S')} UTC**; "
        f"OSM pre-event snapshot {facts.osm_snapshot_date}. *State at the POST pass, issued "
        f"{_n(age_days, 'day', 'days')} later: not an assessment of current conditions; use only to plan field "
        f"verification.*",
        "",
        "### 1. Operational summary",
        "",
        f"Radar shows flood evidence on **{_n(blocked, 'road segment', 'road segments')} in "
        f"{_n(stretches, 'stretch', 'stretches')} ({km['blocked']:,.3f} km)**. "
        f"**{_n(confirmed, 'settlement', 'settlements')} {_be(confirmed)} confirmed cut off**; "
        f"{connected:,} {_be(connected)} connected on observed roads; "
        f"**{_n(unverified, 'settlement has', 'settlements have')} access unverified** (routes cross road the "
        f"radar could not assess). **{facts.not_assessed_fraction * 100:.1f} % of scene pixels were "
        f"not assessed** ({reasons}), so {unobs_scope}.",
        "",
        "### 2. Roads and settlement access",
        "",
        f"| Road state | Length | Settlement access ({facts.settlements_total:,}) | Count |",
        "|---|---|---|---|",
        f"| Blocked ({_n(blocked, 'segment', 'segments')}, {_n(stretches, 'stretch', 'stretches')}) | "
        f"**{km['blocked']:,.3f} km** | Connected | {connected:,} |",
        f"| Observed open | {km['open']:,.3f} km | Access unverified | **{unverified:,}** |",
        f"| Unobservable | {km['unobservable']:,.3f} km | Confirmed cutoff | **{confirmed:,}** |",
        f"| | | No pre-event road access | {no_road:,} |",
        "",
        STATE_DEFINITIONS.format(snap=f"{facts.settlement_snap_max_m:g}"),
        "",
        "### 3. Priorities for field verification",
        "",
    ]
    prio: list[str] = []
    if confirmed:
        prio.append(f"**Reach the {_n(confirmed, 'confirmed cut-off settlement', 'confirmed cut-off settlements')}**"
                    " (settlement drawer: blocked stretches to clear).")
    if stretches:
        prio.append(f"**Verify the {_n(stretches, 'blocked stretch', 'blocked stretches')}** (dashboard evidence "
                    "drawer: location, radar chips, dates).")
    else:
        prio.append("**No blocked stretch was detected**; this is not evidence that roads are clear.")
    if unverified > 1:
        prio.append(f"**Survey the {unverified:,} access-unverified settlements**, starting with segments shared "
                    "by several (settlement drawer).")
    elif unverified == 1:
        prio.append("**Survey the access-unverified settlement** (settlement drawer: separating segments).")
    prio.append("**Reconnoitre before use:** observed-open and unobservable roads are unconfirmed.")
    if no_road:
        prio.append(f"**{_n(no_road, 'settlement', 'settlements')} with no mapped road:** assess access by other "
                    "means; check OSM completeness.")
    lines += [f"{i}. {p}" for i, p in enumerate(prio, 1)]
    lines += [
        "",
        "### 4. Key uncertainty and limitations",
        "",
        f"- {STEP4C_STATEMENT}",
        f"- **Single pass:** no observation after {_utc(post.acquired)}; conditions may have changed since.",
        f"- {reference or NO_REFERENCE_STATEMENT}",
        f"- {NOT_ESTIMATED_STATEMENT}",
        "",
        "### 5. Sources and attribution",
        "",
        f"Sentinel-1 GRD (Copernicus Data Space Ecosystem), Copernicus DEM GLO-30, OpenStreetMap "
        f"({facts.osm_service}).<br>",
        # one required attribution statement per line, verbatim from bato.attribution
        *(f"{line}<br>" for line in (ATTRIBUTION_SENTINEL, ATTRIBUTION_COPERNICUS_DEM, ATTRIBUTION_OSM)),
        *([f"{ems_credit}<br>"] if ems_credit else []),
        f"<sub>{team}{SITREP_VERSION} · run {facts.aoi_id}/{facts.candidate_id} (rule {facts.road_rule}) · "
        f"deterministic, not AI-generated</sub>",
        "",
    ]
    return "\n".join(lines)


def write_report(text: str, output: Path, cfg: BatoConfig) -> Path:
    """Write atomically. Refuses pipeline-data and evaluation-only folders (the SITREP is a report)."""
    out = output.resolve()
    p = cfg.settings.paths
    for folder in (p.raw, p.interim, *p.evaluation_only_dirs()):
        f = folder.resolve()
        if out == f or f in out.parents:
            raise SitrepError(f"refusing to write the SITREP into {folder} (pipeline or evaluation data)")
    if out.suffix.lower() != ".md":
        raise SitrepError("the SITREP output must be a .md file")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    return out

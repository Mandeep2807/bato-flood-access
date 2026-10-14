"""Evidence pack for the SITREP copilot: the ONLY facts the model may use.

Built read-only from the official road-rule v1 run, through the same verified path as the SITREP generator:

1. ``bato.dashboard.sitrep.load_facts`` selects the latest road-rule v1 connectivity run (a v2 run is refused),
   verifies every output against the SHA-256 in its provenance, verifies the stack and Step 4C provenance hashes,
   recomputes the summary from the published layers and cross-checks the printed values;
2. the same run directory is re-opened with ``prepare_inputs`` (hashes verified again) to read the per-stretch
   records (``blocked_segments_ranked.json``), the settlement layer and the stretch coordinates.

EMSR927 / reference data are never read (presentation code is outside ``bato.evaluation``); the only EMSR927
sentence is the fixed, protocol-derived statement the SITREP prints for the evaluated run.

Each item has an ``id`` (cited by the model), a human-readable ``text`` carrying its numbers, the raw ``values``,
and a ``source`` naming the file and field. ``calc:*`` items are reproducible calculations from other items, with
the formula stated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bato.attribution import (
    ATTRIBUTION_COPERNICUS_DEM,
    ATTRIBUTION_OSM,
    ATTRIBUTION_SENTINEL,
    CREDIT_COPERNICUS_EMS,
)
from bato.config import BatoConfig
from bato.dashboard.bundle import (
    PRIORITY_FILE,
    SEGMENTS_FILE,
    SETTLEMENTS_FILE,
    DashboardBuildError,
    _load_json,
    parse_segments,
    prepare_inputs,
)
from bato.dashboard.sitrep import (
    NO_REFERENCE_STATEMENT,
    NOT_ASSESSED_LABEL,
    NOT_ESTIMATED_STATEMENT,
    REFERENCE_STATEMENTS,
    STEP4C_STATEMENT,
    SitrepError,
    SitrepFacts,
    load_facts,
)
from bato.exceptions import BatoError

ACCESS_LABEL = {"CONNECTED": "connected on observed roads", "POSSIBLE_CUTOFF": "access unverified",
                "CONFIRMED_CUTOFF": "confirmed cut off", "NO_PRE_EVENT_ACCESS": "no pre-event road access"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
SETTLEMENT_WORDS = ("settlement", "village", "town", "cut off", "cutoff", "isolated", "unverified", "access",
                    "बस्ती", "गाउँ", "गाउ", "पहुँच", "पहुच", "अपुष्ट")
LIST_LIMIT = 20
PROV = "connectivity_provenance.json"


class EvidenceError(BatoError):
    """The official result cannot be loaded as copilot evidence (nothing is answered)."""


@dataclass
class EvidencePack:
    aoi_id: str
    candidate_id: str
    facts: SitrepFacts
    core: list[dict[str, Any]]
    stretches: list[dict[str, Any]]
    settlements: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {it["id"]: it for it in (*self.core, *self.stretches, *self.settlements)}

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Core items and every blocked stretch always; settlements named in the question, or (for settlement
        questions) the non-connected settlements largest first, capped at LIST_LIMIT with a stated cut."""
        q = question.lower()
        items = [*self.core, *self.stretches]
        named = [s for s in self.settlements
                 if (s["values"].get("name") and len(s["values"]["name"]) >= 3 and s["values"]["name"].lower() in q)
                 or s["values"]["settlement_id"].lower() in q]
        chosen = list(named)
        if any(w in q for w in SETTLEMENT_WORDS):
            rest = [s for s in self.settlements
                    if s not in chosen and s["values"]["access_state"] != "NO_PRE_EVENT_ACCESS"]
            rest.sort(key=lambda s: (-int(s["values"]["building_count"]), s["values"]["settlement_id"]))
            shown = rest[:LIST_LIMIT]
            chosen += shown
            items.append(_item(
                "calc:settlement_list", "Settlement list shown to the copilot",
                f"The copilot sees {len(shown)} of {len(rest)} settlements with pre-event road access (largest first "
                f"by mapped-building count, a size proxy, not population). The full list is in the dashboard Access "
                f"panel.", {"shown": len(shown), "available": len(rest)},
                "calculation: settlements.geojson, sorted by building_count"))
        return items + chosen


def _item(iid: str, title: str, text: str, values: dict[str, Any], source: str) -> dict[str, Any]:
    return {"id": iid, "title": title, "text": text, "values": values, "source": source}


def _km(v: float) -> str:
    return f"{v:,.3f} km"


def build_pack(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> EvidencePack:
    """Load the official v1 result as evidence. Raises EvidenceError (never a partial pack)."""
    for part in (aoi_id, candidate_id):
        if not SAFE_ID.match(part or ""):
            raise EvidenceError(f"invalid identifier {part!r}")
    try:
        f, notes = load_facts(cfg, aoi_id, candidate_id)
        run_dir = (cfg.settings.paths.processed / cfg.settings.connectivity.output_subdir / aoi_id
                   / f.connectivity_run)
        inp = prepare_inputs(cfg, aoi_id, candidate_id, connectivity_dir=run_dir, with_rasters=False)
        ranked = _load_json(inp.files[PRIORITY_FILE])
        seg = parse_segments(_load_json(inp.files[SEGMENTS_FILE]))
        sett = _load_json(inp.files[SETTLEMENTS_FILE])["features"]
    except (SitrepError, DashboardBuildError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"the official v1 result could not be loaded: {exc}") from exc
    if inp.run.road_rule != "v1":                                   # load_facts / prepare_inputs refuse; belt
        raise EvidenceError("only the official road-rule v1 result can be used")
    return EvidencePack(aoi_id, candidate_id, f, _core(f), _stretches(ranked, seg, inp.run.directory.name),
                        _settlements(sett), notes)


def _core(f: SitrepFacts) -> list[dict[str, Any]]:
    km, seg, a = f.road_km, f.road_segments, f.access
    total_km = sum(km.values())
    pre = f.passes
    reasons = ", ".join(f"{NOT_ASSESSED_LABEL[k]} {n:,} px" for k, n in f.not_assessed_reasons)
    with_road = f.settlements_total - a["NO_PRE_EVENT_ACCESS"]
    ref = REFERENCE_STATEMENTS.get((f.aoi_id, f.candidate_id))
    items = [
        _item("run", "Official result",
              f"Official BATO result: AOI {f.aoi_id}, candidate {f.candidate_id}, road rule {f.road_rule}, "
              f"connectivity run {f.connectivity_run} (provenance SHA-256 "
              f"{f.connectivity_provenance_sha256[:12]}, finished {f.connectivity_finished_at}). "
              f"Not validated: a provisional rule output, not a measured accuracy.",
              {"road_rule": f.road_rule, "validated": False}, f"{PROV} (hash-verified outputs)"),
        _item("dates", "Event and observation dates",
              f"Event date {f.event_date}. Sentinel-1 ({f.mission}, {f.orbit_direction.lower()}, relative orbit "
              f"{f.relative_orbit}): PRE-1 {pre[0].acquired:%Y-%m-%d %H:%M:%S} UTC, PRE-2 "
              f"{pre[1].acquired:%Y-%m-%d %H:%M:%S} UTC, POST {pre[2].acquired:%Y-%m-%d %H:%M:%S} UTC. Roads and "
              f"settlements: pre-event OpenStreetMap snapshot {f.osm_snapshot_date}. The result describes the POST "
              f"pass, not current conditions.",
              {"event_date": f.event_date, "osm_snapshot_date": f.osm_snapshot_date},
              "stack provenance inputs[].acquisition_start; connectivity provenance osm_snapshot"),
        _item("roads", "Road states",
              f"Road segments by state: blocked {seg['blocked']:,} segments ({_km(km['blocked'])}), observed open "
              f"{seg['open']:,} segments ({_km(km['open'])}), unobservable {seg['unobservable']:,} segments "
              f"({_km(km['unobservable'])}). Blocked = a flood run of at least 25 m on the road. Observed open = no "
              f"flood signal detected; passability is not guaranteed. Unobservable = the radar rule could not assess "
              f"it: unknown, not open.",
              {"segments": seg, "km": km},
              f"{PROV} summary.road_segments_by_state / road_km_by_state (checked against road_segments.geojson)"),
        _item("stretches", "Blocked stretches",
              f"{f.blocked_stretches:,} blocked stretches (connected runs of blocked segments) made of "
              f"{seg['blocked']:,} blocked segments.", {"blocked_stretches": f.blocked_stretches},
              f"{PROV} summary.blocked_stretches (checked against {PRIORITY_FILE})"),
        _item("settlements", "Settlement access",
              f"{f.settlements_total:,} settlements: {a['CONNECTED']:,} connected on observed roads, "
              f"{a['POSSIBLE_CUTOFF']:,} access unverified (reachable only if unobservable roads are passable; this is "
              f"not a statement that they are cut off), {a['CONFIRMED_CUTOFF']:,} confirmed cut off, "
              f"{a['NO_PRE_EVENT_ACCESS']:,} with no pre-event road access (no mapped road within "
              f"{f.settlement_snap_max_m:g} m: an OSM gap, not a flood effect).",
              {"total": f.settlements_total, "access": a},
              f"{PROV} summary.settlements_by_access_state (checked against settlements.geojson)"),
        _item("not_assessed", "Radar observability (Step 4C)",
              f"{f.not_assessed_fraction * 100:.1f} % of scene pixels were not assessed (fraction "
              f"{f.not_assessed_fraction:.8f}). Not-assessed pixels by reason: {reasons}.",
              {"not_assessed_fraction": f.not_assessed_fraction, "reasons": dict(f.not_assessed_reasons)},
              "Step 4C provenance statistics (not_assessed_fraction_of_all, state_counts)"),
        _item("method", "Method status", STEP4C_STATEMENT.replace("**", ""), {},
              "Step 4C provenance: thresholds[].provisional, validated: false"),
        _item("limits", "Limits",
              " ".join([NOT_ESTIMATED_STATEMENT.replace("**", ""),
                        f"Single pass: no observation after {pre[2].acquired:%Y-%m-%d %H:%M} UTC.",
                        (ref or NO_REFERENCE_STATEMENT).replace("**", "")]), {},
              "fixed statements of the SITREP (frozen protocol bato-emsr927-v1 for the EMSR927 sentence)"),
        _item("priorities", "Field-verification priorities",
              "Priorities (fixed rules on the published counts, not a learned ranking): "
              + (f"reach the {a['CONFIRMED_CUTOFF']:,} confirmed cut-off settlements; "
                 if a["CONFIRMED_CUTOFF"] else "")
              + f"verify the {f.blocked_stretches:,} blocked stretches (dashboard evidence drawer); survey the "
              f"{a['POSSIBLE_CUTOFF']:,} access-unverified settlements, starting with road segments shared by several; "
              f"reconnoitre observed-open and unobservable roads before use; for the {a['NO_PRE_EVENT_ACCESS']:,} "
              f"settlements with no mapped road, assess access by other means.", {},
              "SITREP section 3 rules applied to the published counts"),
        _item("calc:road_share", "Calculation: share of road length by state",
              f"Total mapped road length = blocked + observed open + unobservable = {_km(total_km)}. Shares: blocked "
              f"{km['blocked'] / total_km * 100:.2f} %, observed open {km['open'] / total_km * 100:.1f} %, "
              f"unobservable {km['unobservable'] / total_km * 100:.1f} %; observed (blocked + open) "
              f"{_km(km['blocked'] + km['open'])} = {(km['blocked'] + km['open']) / total_km * 100:.1f} %.",
              {"total_km": round(total_km, 3)}, "calculation from the 'roads' item"),
        _item("calc:settlements_with_road", "Calculation: settlements with pre-event road access",
              f"Settlements with pre-event road access = total - no pre-event road access = {f.settlements_total:,} - "
              f"{a['NO_PRE_EVENT_ACCESS']:,} = {with_road:,}. Of these: verified on observed roads {a['CONNECTED']:,}; "
              f"access unverified {a['POSSIBLE_CUTOFF']:,}; confirmed cut off {a['CONFIRMED_CUTOFF']:,}.",
              {"with_road": with_road},
              "calculation from the 'settlements' item"),
        _item("attribution", "Required attribution",
              f"{ATTRIBUTION_SENTINEL} {ATTRIBUTION_COPERNICUS_DEM} {ATTRIBUTION_OSM}"
              + (f" Evaluation reference only: {CREDIT_COPERNICUS_EMS} (EMSR927)." if ref else ""), {},
              "bato.attribution"),
    ]
    return items


def _stretches(ranked: list[dict[str, Any]], seg: Any, run: str) -> list[dict[str, Any]]:
    out = []
    for st in ranked:
        idx = [seg.index[s] for s in st["segment_ids"] if s in seg.index]
        pts = [p for i in idx for p in seg.coords[i]]
        lon = sum(p[0] for p in pts) / len(pts) if pts else None
        lat = sum(p[1] for p in pts) / len(pts) if pts else None
        name = ", ".join(n for n in st.get("names", []) if n) or "unnamed road"
        hw = ", ".join(st.get("highways", [])) or "road"
        where = f" Centre about {lat:.4f}° N, {lon:.4f}° E (mean of its segment vertices)." if pts else ""
        out.append(_item(
            f"stretch:{st['stretch_id']}", f"Blocked stretch {st['stretch_id']} (rank {st['rank']})",
            f"Blocked stretch {st['stretch_id']}, rank {st['rank']}: {name} "
            f"({hw}{', bridge' if st.get('bridge') else ''}), "
            f"{len(st['segment_ids'])} segments, length {st['length_m']:.1f} m, flood-affected length "
            f"{st['affected_length_m']:.1f} m, longest flood run {st['longest_flood_run_m']:.1f} m; clearing it alone "
            f"reconnects {st['settlements_reconnected']} confirmed cut-off settlements "
            f"({st['settlements_reconnected_on_observed_roads']} on observed roads only).{where}",
            {"stretch_id": st["stretch_id"], "rank": st["rank"], "segments": len(st["segment_ids"]),
             "length_m": st["length_m"], "affected_length_m": st["affected_length_m"],
             "longest_flood_run_m": st["longest_flood_run_m"], "settlements_reconnected": st["settlements_reconnected"],
             "evidence_ids": list(st.get("evidence_ids", []))[:3]},
            f"{run}/{PRIORITY_FILE}; centre: calculation from {SEGMENTS_FILE}"))
    return out


def _settlements(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for feat in features:
        p = feat["properties"]
        sid, name, state = str(p["settlement_id"]), p.get("name"), str(p["access_state"])
        label = name or f"unnamed building cluster {sid}"
        out.append(_item(
            f"settlement:{sid}", f"Settlement {label}",
            f"Settlement {label} ({sid}): {ACCESS_LABEL.get(state, state)}; {int(p['building_count'])} mapped OSM "
            f"buildings (a size proxy, not population).",
            {"settlement_id": sid, "name": name, "access_state": state, "building_count": int(p["building_count"])},
            f"{SETTLEMENTS_FILE} properties (access_state, building_count)"))
    return out

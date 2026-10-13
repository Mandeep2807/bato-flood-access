"""SITREP generator (bato.dashboard.sitrep): official wording and numbers, read-only loading of the official v1
run, and refusal on missing / inconsistent values. Synthetic world = invented data (connectivity_fakes)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bato.attribution import ATTRIBUTION_COPERNICUS_DEM, ATTRIBUTION_OSM, ATTRIBUTION_SENTINEL, CREDIT_COPERNICUS_EMS
from bato.dashboard.sitrep import (
    EMSR927_STATEMENT,
    NO_REFERENCE_STATEMENT,
    STEP4C_STATEMENT,
    Pass,
    SitrepError,
    SitrepFacts,
    load_facts,
    render,
    write_report,
)

from sitrep_fakes import published_sitrep_world

UTC = timezone.utc
ISSUED = datetime(2026, 10, 13, 7, 0, tzinfo=UTC)
TITLE = "Trishuli Flood, Nepal"
INCIDENT = "flood, Bhote Koshi–Trishuli corridor (Rasuwa/Nuwakot), Nepal"


def official_facts() -> SitrepFacts:
    """The official Trishuli road-rule v1 values (technical report, Table 1/2) as the loader returns them."""
    return SitrepFacts(
        aoi_id="aoi-e893707c2beb", candidate_id="ts-80841ae62417", road_rule="v1", event_date="2026-08-26",
        aoi_bbox=(85.17, 27.97, 85.42, 28.28), analysis_buffer_km=15.0, mission="Sentinel-1D",
        orbit_direction="ASCENDING", relative_orbit=85,
        passes=(Pass("PRE1", datetime(2026, 6, 29, 12, 21, 38, tzinfo=UTC), "S1D_IW_GRDH_1SDV_PRE1"),
                Pass("PRE2", datetime(2026, 8, 16, 12, 21, 41, tzinfo=UTC), "S1D_IW_GRDH_1SDV_PRE2"),
                Pass("POST", datetime(2026, 8, 28, 12, 21, 41, tzinfo=UTC), "S1D_IW_GRDH_1SDV_POST")),
        osm_snapshot_date="2026-08-25", osm_service="ohsome-api-v2",
        road_segments={"blocked": 30, "open": 10323, "unobservable": 61499},
        road_km={"blocked": 1.227, "open": 463.714, "unobservable": 2942.161}, blocked_stretches=9,
        settlements_total=265,
        access={"CONNECTED": 1, "POSSIBLE_CUTOFF": 163, "CONFIRMED_CUTOFF": 0, "NO_PRE_EVENT_ACCESS": 101},
        not_assessed_fraction=0.91225913,
        not_assessed_reasons=(("NOT_ASSESSED_STEEP", 35544186), ("NOT_ASSESSED_LAYOVER", 4713918),
                              ("NOT_ASSESSED_SHADOW", 1027803)),
        settlement_snap_max_m=500.0, connectivity_run="ts-80841ae62417", connectivity_provenance_sha256="a" * 64,
        connectivity_finished_at="2026-10-04T06:00:00Z")


def official_text(**kw) -> str:
    args = {"title": TITLE, "incident": INCIDENT, "issued_at": ISSUED, "prepared_by": "404 Competition Not Found"}
    args.update(kw)
    return render(official_facts(), **args)


# ---------------------------------------------------------------------------
# Rendering: official values and wording
# ---------------------------------------------------------------------------


def test_official_values_and_wording():
    t = official_text()
    for s in ("**30 road segments in 9 stretches (1.227 km)**",
              "| Blocked (30 segments, 9 stretches) | **1.227 km** | Connected | 1 |",
              "| Observed open | 463.714 km | Access unverified | **163** |",
              "| Unobservable | 2,942.161 km | Confirmed cutoff | **0** |",
              "| | | No pre-event road access | 101 |",
              "Settlement access (265)",
              "**0 settlements are confirmed cut off**",
              "**163 settlements have access unverified**",
              "**91.2 % of scene pixels were not assessed** (steep slopes, layover, shadow), so most roads are "
              "unobservable",
              "**Verify the 9 blocked stretches**", "**Survey the 163 access-unverified settlements**",
              "**101 settlements with no mapped road:**",
              "**Observed open** = no flood signal detected, passability not guaranteed",
              "**Unobservable** = unknown, not open", "within 500 m (OSM gap, not flood)",
              STEP4C_STATEMENT, EMSR927_STATEMENT,
              "**Area:** 85.17–85.42° E, 27.97–28.28° N (roads within a 15 km buffer)",
              "**Data:** Sentinel-1D ascending, orbit 85 — PRE-1 2026-06-29, PRE-2 2026-08-16, "
              "**POST 2026-08-28 12:21:41 UTC**; OSM pre-event snapshot 2026-08-25",
              "**Event:** 2026-08-26", "run aoi-e893707c2beb/ts-80841ae62417 (rule v1)",
              "Prepared by 404 Competition Not Found", "deterministic, not AI-generated"):
        assert s in t, s
    assert "cut off" not in t.replace("**0 settlements are confirmed cut off**", "")   # no other cut-off claim
    assert "villages" not in t and "isolated" not in t


def test_observation_and_issue_dates_are_distinct_and_not_current_conditions():
    t = official_text()
    assert "**Issued** 2026-10-13 07:00 UTC · **Observed** 2026-08-28 12:21 UTC (POST pass) · **NOT VALIDATED**" in t
    assert "*State at the POST pass, issued 45 days later: not an assessment of current conditions" in t
    assert "**Single pass:** no observation after 2026-08-28 12:21 UTC" in t


REQUIRED_ATTRIBUTION = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
)


def _own_lines(text: str) -> list[str]:
    return [line.removesuffix("<br>") for line in text.splitlines()]


def test_required_attribution_is_verbatim():
    lines = _own_lines(official_text())
    for statement in REQUIRED_ATTRIBUTION:                    # exact text, each on its own line
        assert statement in lines, statement
    assert f"Evaluation reference only: {CREDIT_COPERNICUS_EMS} (EMSR927)." in lines
    assert ATTRIBUTION_SENTINEL == "Contains modified Copernicus Sentinel data 2026."
    assert ATTRIBUTION_COPERNICUS_DEM == (
        "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 "
        "provided under COPERNICUS by the European Union and ESA; all rights reserved.")
    assert ATTRIBUTION_OSM == "© OpenStreetMap contributors."


def test_no_invented_quantities():
    t = official_text().lower()
    assert t.count("population") == 1 and "casualties" in t and t.count("forecast") == 1     # only "not estimated"
    for word in ("accuracy", "precision of", "people", "affected population", "validated accuracy"):
        assert word not in t


def test_render_is_deterministic():
    assert official_text() == official_text()
    later = official_text(issued_at=ISSUED + timedelta(days=1))
    diff = [(a, b) for a, b in zip(official_text().splitlines(), later.splitlines(), strict=True) if a != b]
    assert len(diff) == 2 and "**Issued**" in diff[0][0] and "46 days later" in diff[1][1]


def test_local_time_label():
    assert "(12:45 UTC+05:45)" in official_text(local_tz=ZoneInfo("Asia/Kathmandu"))
    assert "(12:30 IST)" in official_text(local_tz=ZoneInfo("Asia/Kolkata"))
    assert "(" not in official_text().splitlines()[2].split("· **Observed**")[0]


@pytest.mark.parametrize("kw, msg", [
    ({"issued_at": datetime(2026, 8, 28, 12, 0, tzinfo=UTC)}, "before the observation"),
    ({"issued_at": datetime(2026, 10, 13, 7, 0)}, "time zone"),
    ({"title": "  "}, "required"),
])
def test_render_refuses_bad_arguments(kw, msg):
    with pytest.raises(SitrepError, match=msg):
        official_text(**kw)


def test_reference_statement_only_for_the_evaluated_run():
    other = replace(official_facts(), aoi_id="aoi-other")
    t = render(other, title=TITLE, incident=INCIDENT, issued_at=ISSUED)
    assert NO_REFERENCE_STATEMENT in t and "EMSR927" not in t and CREDIT_COPERNICUS_EMS not in t
    for statement in REQUIRED_ATTRIBUTION:                    # the required lines stay for any run
        assert statement in _own_lines(t), statement
    assert "Prepared by" not in t


def test_singular_and_zero_wording():
    f = replace(official_facts(), blocked_stretches=1, road_segments={"blocked": 1, "open": 5, "unobservable": 1},
                settlements_total=4,
                access={"CONNECTED": 1, "POSSIBLE_CUTOFF": 1, "CONFIRMED_CUTOFF": 1, "NO_PRE_EVENT_ACCESS": 1})
    t = render(f, title=TITLE, incident=INCIDENT, issued_at=ISSUED)
    assert "**1 road segment in 1 stretch" in t and "**1 settlement is confirmed cut off**" in t
    assert "**1 settlement has access unverified**" in t and "**Reach the 1 confirmed cut-off settlement**" in t
    assert "**Survey the access-unverified settlement**" in t
    z = render(replace(f, blocked_stretches=0, road_segments={"blocked": 0, "open": 5, "unobservable": 1}),
               title=TITLE, incident=INCIDENT, issued_at=ISSUED)
    assert "**No blocked stretch was detected**; this is not evidence that roads are clear." in z


# ---------------------------------------------------------------------------
# Loading the official v1 run (synthetic world)
# ---------------------------------------------------------------------------


def test_load_facts_reads_the_published_v1_run(tmp_path):
    w = published_sitrep_world(tmp_path)
    f, _notes = load_facts(w.cfg, "aoi-test", "ts-test")
    s = w.prov.summary
    assert f.road_rule == "v1" and f.connectivity_run == "ts-test"
    assert f.road_segments == {k: s.road_segments_by_state[k] for k in ("blocked", "open", "unobservable")}
    assert f.road_km == {k: s.road_km_by_state[k] for k in ("blocked", "open", "unobservable")}
    assert f.blocked_stretches == s.blocked_stretches == 3
    assert f.access == {"CONNECTED": 2, "POSSIBLE_CUTOFF": 1, "CONFIRMED_CUTOFF": 3, "NO_PRE_EVENT_ACCESS": 1}
    assert f.settlements_total == s.settlements_total == 7
    assert f.mission == "Sentinel-1D" and (f.orbit_direction, f.relative_orbit) == ("ASCENDING", 85)
    assert [p.role for p in f.passes] == ["PRE1", "PRE2", "POST"] and f.event_date == "2026-08-26"
    assert f.aoi_bbox == (85.015, 28.015, 85.045, 28.045) and f.analysis_buffer_km == 15.0
    assert f.not_assessed_reasons[0][0] == "NOT_ASSESSED_STEEP" and 0 < f.not_assessed_fraction < 1
    assert f.osm_snapshot_date == "2026-08-25" and f.settlement_snap_max_m == 500.0
    t = render(f, title="Synthetic", incident="test fixture", issued_at=ISSUED)
    assert "**13 road segments in 3 stretches (0.635 km)**" in t and NO_REFERENCE_STATEMENT in t


def test_uses_the_v1_run_and_refuses_an_explicit_v2_run(tmp_path):
    w = published_sitrep_world(tmp_path, extra_v2=True)
    f, notes = load_facts(w.cfg, "aoi-test", "ts-test")
    assert f.connectivity_run.startswith("ts-test.superseded-") and any("road rule v2" in n for n in notes)
    with pytest.raises(SitrepError, match="road rule v2"):
        load_facts(w.cfg, "aoi-test", "ts-test", connectivity_dir=w.out)


def test_refuses_a_changed_output_file(tmp_path):
    w = published_sitrep_world(tmp_path)
    p = w.out / "road_segments.geojson"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(SitrepError, match="SHA-256"):
        load_facts(w.cfg, "aoi-test", "ts-test")


@pytest.mark.parametrize("edit, msg", [
    (lambda s: s.update(blocked_stretches=4), "blocked stretches"),
    (lambda s: s["road_km_by_state"].update(open=1.0), "disagrees with the published layers"),
    (lambda s: s["reachable_settlements"].update(post_pessimistic=0), "disagrees|inconsistent"),
    (lambda s: s["settlements_by_access_state"].update(ISOLATED=0), "unknown settlement access states"),
])
def test_refuses_an_inconsistent_summary(tmp_path, edit, msg):
    w = published_sitrep_world(tmp_path)
    p = w.out / "connectivity_provenance.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    edit(data["summary"])
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SitrepError, match=msg):
        load_facts(w.cfg, "aoi-test", "ts-test")


def test_refuses_stretches_that_do_not_cover_the_blocked_segments(tmp_path):
    w = published_sitrep_world(tmp_path)
    p = w.out / "blocked_segments_ranked.json"
    ranked = json.loads(p.read_text(encoding="utf-8"))
    ranked[0]["segment_ids"] = ranked[0]["segment_ids"][1:]
    data = json.loads((w.out / "connectivity_provenance.json").read_text(encoding="utf-8"))
    p.write_text(json.dumps(ranked), encoding="utf-8")
    data["outputs"]["blocked_segments_ranked.json"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (w.out / "connectivity_provenance.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SitrepError, match="partition"):
        load_facts(w.cfg, "aoi-test", "ts-test")


def _drop_state_counts(b):
    del b["statistics"]["state_counts"]


def _bad_reason_counts(b):
    b["statistics"]["state_counts"]["NOT_ASSESSED_STEEP"] += 1


def _bad_fraction(b):
    b["statistics"]["not_assessed_fraction_of_all"] = 0.5


def _non_provisional(b):
    b["thresholds"][0]["provisional"] = False


@pytest.mark.parametrize("edit, msg", [
    (_drop_state_counts, "statistics.state_counts is missing"),
    (_bad_reason_counts, "reason counts do not sum"),
    (_bad_fraction, "fraction_of_all disagrees"),
    (_non_provisional, "non-provisional threshold"),
])
def test_refuses_missing_or_inconsistent_step4c_values(tmp_path, edit, msg):
    w = published_sitrep_world(tmp_path, baseline_edit=edit)
    with pytest.raises(SitrepError, match=msg):
        load_facts(w.cfg, "aoi-test", "ts-test")


def _drop_buffer(s):
    del s["grid"]["analysis_buffer_km"]


def _other_unit(s):
    s["inputs"][2]["product_name"] = "TEST-S1A-POST"


def _other_event(s):
    s["event_date"] = "2026-08-27"


@pytest.mark.parametrize("edit, msg", [
    (_drop_buffer, "grid.analysis_buffer_km is missing"),
    (_other_unit, "mixes Sentinel-1 units"),
    (_other_event, "event_date"),
])
def test_refuses_missing_or_inconsistent_stack_values(tmp_path, edit, msg):
    w = published_sitrep_world(tmp_path, stack_edit=edit)
    with pytest.raises(SitrepError, match=msg):
        load_facts(w.cfg, "aoi-test", "ts-test")


def test_refuses_when_no_v1_run_exists(tmp_path):
    w = published_sitrep_world(tmp_path)
    with pytest.raises(SitrepError, match="no road rule v1"):
        load_facts(w.cfg, "aoi-test", "ts-other")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_report_is_atomic_and_refuses_data_folders(tmp_path):
    w = published_sitrep_world(tmp_path)
    out = write_report("# x\n", tmp_path / "reports" / "BATO_SITREP.md", w.cfg)
    assert out.read_text(encoding="utf-8") == "# x\n" and not list(out.parent.glob(".*tmp-*"))
    p = w.cfg.settings.paths
    for folder in (p.raw, p.interim, *p.evaluation_only_dirs()):
        with pytest.raises(SitrepError, match="refusing"):
            write_report("# x\n", folder / "BATO_SITREP.md", w.cfg)
    with pytest.raises(SitrepError, match=".md"):
        write_report("# x\n", tmp_path / "reports" / "sitrep.txt", w.cfg)

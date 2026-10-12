"""Sentinel-1 catalogue classification and orbit grouping (TEST FIXTURE data only)."""

from __future__ import annotations

import copy
import json
import random
from datetime import date, time
from pathlib import Path

import pytest

from bato.acquisition.s1_catalogue import classify_records, group_by_track, search_window
from bato.contracts.aoi import AOI, AOIRequest, build_aoi
from bato.contracts.catalogue import CandidateRole, CandidateStatus, TrackGroup
from bato.contracts.common import BBox, OrbitDirection

from helpers import FIXED_TODAY

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cdse_s1_catalogue_TEST_FIXTURE.json"
S = CandidateStatus
EXPECTED = {
    "TEST-FIXTURE-0001": (S.ACCEPTED, CandidateRole.PRE_EVENT),
    "TEST-FIXTURE-0002": (S.ACCEPTED, CandidateRole.PRE_EVENT),
    "TEST-FIXTURE-0003": (S.ACCEPTED, CandidateRole.PRE_EVENT),
    "TEST-FIXTURE-0004": (S.REJECTED_DATE, None),            # same day, no event time
    "TEST-FIXTURE-0005": (S.ACCEPTED, CandidateRole.POST_EVENT),
    "TEST-FIXTURE-0006": (S.ACCEPTED, CandidateRole.POST_EVENT),
    "TEST-FIXTURE-0007": (S.REJECTED_PRODUCT_TYPE, None),    # SLC
    "TEST-FIXTURE-0008": (S.REJECTED_MODE, None),            # EW
    "TEST-FIXTURE-0009": (S.ACCEPTED, CandidateRole.PRE_EVENT),
    "TEST-FIXTURE-0010": (S.ACCEPTED, CandidateRole.POST_EVENT),
    "TEST-FIXTURE-0011": (S.REJECTED_POLARIZATION, None),    # HH only
    "TEST-FIXTURE-0012": (S.REJECTED_DUPLICATE, None),       # COG copy of 0006
    "TEST-FIXTURE-0013": (S.ACCEPTED, CandidateRole.POST_EVENT),  # slice 1 of a 2-slice pass
    "TEST-FIXTURE-0014": (S.ACCEPTED, CandidateRole.POST_EVENT),  # slice 2
    "TEST-FIXTURE-0015": (S.REJECTED_COVERAGE, None),        # covers half the AOI
    "TEST-FIXTURE-0016": (S.REJECTED_INVALID_METADATA, None),  # no orbitDirection
    "TEST-FIXTURE-0017": (S.REJECTED_ORBIT, None),           # relative orbit 300
    "TEST-FIXTURE-0018": (S.REJECTED_INVALID_METADATA, None),  # malformed name
    "TEST-FIXTURE-0019": (S.REJECTED_INVALID_METADATA, None),  # no ContentDate
    "TEST-FIXTURE-0020": (S.REJECTED_INVALID_METADATA, None),  # attribute contradicts name
    "TEST-FIXTURE-0021": (S.REJECTED_DATE, None),            # outside window
}


@pytest.fixture(scope="module")
def raw_records() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["value"]


def make_aoi(cfg, event_time: time | None = None) -> AOI:
    request = AOIRequest(event_date=date(2026, 8, 26), event_time_utc=event_time,
                         bbox=BBox(min_lon=85.20, min_lat=28.00, max_lon=85.40, max_lat=28.20))
    return build_aoi(request, cfg.settings, today=FIXED_TODAY)


def classify(cfg, raw, aoi: AOI | None = None):
    aoi = aoi or make_aoi(cfg)
    s1 = cfg.settings.sentinel1
    return classify_records(raw, aoi, s1, search_window(aoi, s1))


def by_id(records):
    return {r.product_id: r for r in records}


def test_fixture_is_labelled_synthetic():
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert doc["_TEST_FIXTURE"].startswith("TEST FIXTURE - SYNTHETIC")
    assert all(r["Id"].startswith("TEST-FIXTURE-") for r in doc["value"])


def test_every_record_gets_the_expected_status_and_role(cfg, raw_records):
    records = by_id(classify(cfg, raw_records))
    assert set(records) == set(EXPECTED)  # nothing dropped
    for pid, (status, role) in EXPECTED.items():
        assert (records[pid].status, records[pid].role if status is S.ACCEPTED else None) == (status, role), \
            f"{pid}: {records[pid].status} - {records[pid].reason}"


def test_rejections_explain_themselves(cfg, raw_records):
    records = by_id(classify(cfg, raw_records))
    assert "ambiguous" in records["TEST-FIXTURE-0004"].reason
    assert "IW_SLC__1S" in records["TEST-FIXTURE-0007"].reason
    assert "beam mode EW" in records["TEST-FIXTURE-0008"].reason
    assert "missing required VV+VH" in records["TEST-FIXTURE-0011"].reason
    assert records["TEST-FIXTURE-0012"].duplicate_of == records["TEST-FIXTURE-0006"].name
    assert "orbitDirection" in records["TEST-FIXTURE-0016"].reason
    assert "relative orbit 300" in records["TEST-FIXTURE-0017"].reason
    assert "naming convention" in records["TEST-FIXTURE-0018"].reason
    assert "ContentDate" in records["TEST-FIXTURE-0019"].reason
    assert "polarizations" in records["TEST-FIXTURE-0020"].reason
    assert "outside search window" in records["TEST-FIXTURE-0021"].reason


def test_coverage_is_stored_including_for_rejected_scene(cfg, raw_records):
    records = by_id(classify(cfg, raw_records))
    partial = records["TEST-FIXTURE-0015"]
    assert partial.scene_coverage == pytest.approx(0.5, abs=0.01)
    assert "minimum 80%" in partial.reason
    assert records["TEST-FIXTURE-0001"].scene_coverage == 1.0


def test_two_slices_of_one_pass_are_accepted_together(cfg, raw_records):
    records = by_id(classify(cfg, raw_records))
    south, north = records["TEST-FIXTURE-0013"], records["TEST-FIXTURE-0014"]
    assert south.pass_id == north.pass_id
    assert south.scene_coverage == pytest.approx(0.5, abs=0.02)
    assert north.scene_coverage == pytest.approx(0.6, abs=0.02)
    assert south.pass_coverage == north.pass_coverage == 1.0


def test_lone_slice_is_rejected_for_coverage(cfg, raw_records):
    alone = [r for r in raw_records if r["Id"] == "TEST-FIXTURE-0013"]
    (record,) = classify(cfg, alone)
    assert record.status is S.REJECTED_COVERAGE and record.pass_coverage == pytest.approx(0.5, abs=0.02)


def test_event_time_resolves_same_day_scene(cfg, raw_records):
    early_event = make_aoi(cfg, event_time=time(0, 0))     # scene at 00:15 -> after the event
    late_event = make_aoi(cfg, event_time=time(12, 0))     # scene at 00:15 -> before the event
    assert by_id(classify(cfg, raw_records, early_event))["TEST-FIXTURE-0004"].role is CandidateRole.POST_EVENT
    assert by_id(classify(cfg, raw_records, late_event))["TEST-FIXTURE-0004"].role is CandidateRole.PRE_EVENT


def test_days_from_event(cfg, raw_records):
    records = by_id(classify(cfg, raw_records))
    assert records["TEST-FIXTURE-0005"].days_from_event == pytest.approx(6.011, abs=0.001)
    assert records["TEST-FIXTURE-0002"].days_from_event < 0


def test_orbit_grouping_never_mixes_tracks(cfg, raw_records):
    groups = group_by_track(classify(cfg, raw_records))
    keys = [(g.orbit_direction, g.relative_orbit) for g in groups]
    assert keys == [(OrbitDirection.ASCENDING, 85), (OrbitDirection.DESCENDING, 19),
                    (OrbitDirection.DESCENDING, 121)]
    desc19 = groups[1]
    assert [n[:3] for n in desc19.pre_event] == ["S1A", "S1A", "S1C"]   # S1A and S1C share track 19
    assert len(desc19.post_event) == 2 and desc19.has_pre_and_post
    assert not groups[2].has_pre_and_post                               # post-event only


def test_same_relative_orbit_different_direction_are_separate_groups(cfg, raw_records):
    flipped = copy.deepcopy([r for r in raw_records if r["Id"] == "TEST-FIXTURE-0009"])[0]
    flipped["Id"] = "TEST-FIXTURE-0009-FLIPPED"            # a different acquisition, not a duplicate
    flipped["Name"] = flipped["Name"].replace("20260821T123005_20260821T123030",
                                              "20260821T123105_20260821T123130", 1)
    flipped["ContentDate"] = {"Start": "2026-08-21T12:31:05.000Z"}
    for attr in flipped["Attributes"]:
        if attr["Name"] == "relativeOrbitNumber":
            attr["Value"] = 19                       # ascending 19 vs descending 19
    groups = group_by_track(classify(cfg, [*raw_records, flipped]))
    keys = {(g.orbit_direction, g.relative_orbit) for g in groups}
    assert {(OrbitDirection.ASCENDING, 19), (OrbitDirection.DESCENDING, 19)} <= keys


def test_classification_is_independent_of_input_order(cfg, raw_records):
    reference = classify(cfg, raw_records)
    shuffled = list(raw_records)
    random.Random(7).shuffle(shuffled)
    assert classify(cfg, shuffled) == reference
    assert [r.sensing_start for r in reference if r.sensing_start] == \
        sorted(r.sensing_start for r in reference if r.sensing_start)


def test_required_polarisations_come_from_config(cfg, raw_records):
    vv_only = cfg.settings.sentinel1.model_copy(update={"polarizations": ("VV",)})
    aoi = make_aoi(cfg)
    records = by_id(classify_records(raw_records, aoi, vv_only, search_window(aoi, vv_only)))
    assert records["TEST-FIXTURE-0011"].status is S.REJECTED_POLARIZATION  # HH-only still fails
    hh = cfg.settings.sentinel1.model_copy(update={"polarizations": ("HH",)})
    records = by_id(classify_records(raw_records, aoi, hh, search_window(aoi, hh)))
    assert records["TEST-FIXTURE-0011"].status is S.ACCEPTED
    assert records["TEST-FIXTURE-0001"].status is S.REJECTED_POLARIZATION


def test_coverage_threshold_comes_from_config(cfg, raw_records):
    lenient = cfg.settings.sentinel1.model_copy(update={"min_aoi_coverage_fraction": 0.4})
    aoi = make_aoi(cfg)
    records = by_id(classify_records(raw_records, aoi, lenient, search_window(aoi, lenient)))
    assert records["TEST-FIXTURE-0015"].status is S.ACCEPTED


def test_search_window_uses_config(cfg):
    start, end = search_window(make_aoi(cfg), cfg.settings.sentinel1)
    assert start.isoformat() == "2026-06-27T00:00:00+00:00"   # 60 days before
    assert end.isoformat() == "2026-09-20T00:00:00+00:00"     # 24 days after, end exclusive


def test_empty_catalogue_gives_empty_result(cfg):
    assert classify(cfg, []) == ()
    assert group_by_track(()) == ()


def test_track_group_contract_rejects_mixed_tracks(cfg, raw_records):
    from bato.contracts.catalogue import CatalogueSearchResult
    aoi = make_aoi(cfg)
    records = classify(cfg, raw_records)
    start, end = search_window(aoi, cfg.settings.sentinel1)
    good = list(group_by_track(records))
    asc85 = next(r.name for r in records if r.product_id == "TEST-FIXTURE-0009")
    bad = good[1].model_copy(update={"pre_event": (*good[1].pre_event, asc85)})
    with pytest.raises(ValueError, match="wrong track"):
        CatalogueSearchResult(
            provider="test", query_filter="f", aoi_id=aoi.aoi_id, event_date=aoi.event_date,
            window_start=start, window_end=end, retrieved_at=start, returned_count=len(records), truncated=False,
            config_fingerprint=cfg.fingerprint(), records=records,
            groups=(good[0], TrackGroup(**{**bad.model_dump(exclude={"has_pre_and_post"})}), good[2]))

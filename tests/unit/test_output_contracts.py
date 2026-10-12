"""Contracts D (detection raster), E (road impact), F (settlement status),
G (evidence) and JSON (de)serialisation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from bato.contracts.common import DataSource, LineStringGeometry, PointGeometry
from bato.contracts.detection import (
    ConfidenceTier,
    DetectionMethod,
    DetectionRaster,
    ImpactClass,
    RasterBand,
    confidence_tier,
    validate_class_array,
)
from bato.contracts.evidence import (
    ArtifactRef,
    EvidenceRecord,
    SourceRef,
    SubjectType,
    make_evidence_id,
)
from bato.contracts.impact import BridgeExposure, RoadSegmentImpact, RoadState
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.settlement import (
    AccessStatus,
    Reachability,
    SettlementSource,
    SettlementStatus,
    TargetKind,
    classify_settlement,
)

from helpers import FINGERPRINT, utc

# ---- D: detection raster ----------------------------------------------------------

BANDS = (
    RasterBand(index=1, name="impact_class", dtype="uint8", description="ImpactClass codes", nodata=255),
    RasterBand(index=2, name="flood_probability", dtype="float32", description="calibrated p(flood)"),
)


def raster(**overrides) -> DetectionRaster:
    fields = dict(
        raster_id="det-test", aoi_id="aoi-test", pair_ids=("pair-test",),
        path="data/processed/aoi-test/detection.tif", crs="EPSG:32645",
        transform=(10.0, 0.0, 300000.0, 0.0, -10.0, 3100000.0), width=100, height=80, bands=BANDS,
        method=DetectionMethod.BASELINE_CHANGE, config_fingerprint=FINGERPRINT, created_at=utc(2026, 10, 1),
    )
    fields.update(overrides)
    return DetectionRaster(**fields)


def test_detection_raster_valid_and_round_trips():
    r = raster()
    assert r.pixel_size == (10.0, -10.0)
    assert DetectionRaster.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize(("overrides", "message"), [
    ({"bands": BANDS[1:]}, "band indices"),
    ({"bands": (BANDS[0].model_copy(update={"nodata": 0}),) + BANDS[1:]}, "nodata"),
    ({"method": DetectionMethod.FLOOD_MODEL}, "model_version"),
    ({"crs": "UTM45N"}, "EPSG"),
    ({"pair_ids": ()}, "pre/post pair"),
    ({"path": "/abs/detection.tif"}, "relative"),
    ({"path": "data/../secret.tif"}, "'..'"),
    ({"config_fingerprint": "abc"}, "SHA-256"),
    ({"transform": (0.0, 0.0, 300000.0, 0.0, -10.0, 3100000.0)}, "pixel sizes"),
])
def test_detection_raster_invalid(overrides, message):
    with pytest.raises(ValidationError, match=message):
        raster(**overrides)


def test_class_codes_are_stable():
    # Changing these breaks every raster already written; change deliberately.
    assert {c.name: int(c) for c in ImpactClass} == {
        "NO_CHANGE": 0, "FLOOD_WATER": 1, "DEBRIS_DISTURBANCE": 2, "PERMANENT_WATER": 3,
        "UNOBSERVABLE": 254, "OUTSIDE_AOI": 255}


def test_validate_class_array():
    validate_class_array(np.array([[0, 1], [254, 255]], dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown impact class"):
        validate_class_array(np.array([[0, 7]], dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        validate_class_array(np.array([[0, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="2-D"):
        validate_class_array(np.array([0, 1], dtype=np.uint8))


def test_confidence_tier_uses_configured_cutoffs(cfg):
    t = cfg.thresholds.confidence
    assert confidence_tier(t.tier_high_min.value, t) is ConfidenceTier.HIGH
    assert confidence_tier(t.tier_medium_min.value, t) is ConfidenceTier.MEDIUM
    assert confidence_tier(0.0, t) is ConfidenceTier.LOW
    with pytest.raises(ValueError):
        confidence_tier(1.2, t)
    with pytest.raises(ValueError):
        confidence_tier(float("nan"), t)


# ---- E: road impact ---------------------------------------------------------------

LINE = LineStringGeometry(coordinates=((85.30, 28.10), (85.3005, 28.1003)))


def road(**overrides) -> RoadSegmentImpact:
    fields = dict(segment_id="seg-1", osm_way_id=123, highway="primary", geometry=LINE, length_m=50.0,
                  state=RoadState.BLOCKED, affected_length_m=30.0, longest_affected_run_m=30.0,
                  unobservable_length_m=0.0, max_impact_probability=0.92, evidence_ids=("ev-1",))
    fields.update(overrides)
    return RoadSegmentImpact(**fields)


def test_road_impact_valid():
    r = road(unobservable_length_m=10.0)
    assert r.observable_fraction == 0.8
    assert RoadSegmentImpact.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize(("overrides", "message"), [
    ({"affected_length_m": 60.0, "longest_affected_run_m": 30.0}, "affected_length_m"),
    ({"longest_affected_run_m": 40.0}, "longest_affected_run_m"),
    ({"unobservable_length_m": 30.0}, "overlap or exceed"),
    ({"affected_length_m": 0.0, "longest_affected_run_m": 0.0}, "BLOCKED segment"),
    ({"evidence_ids": ()}, "must reference evidence"),
    ({"state": RoadState.UNOBSERVABLE, "affected_length_m": 0.0, "longest_affected_run_m": 0.0,
      "max_impact_probability": None}, "UNOBSERVABLE segment"),
    ({"max_impact_probability": 1.3}, "max_impact_probability"),
    ({"state": RoadState.OPEN, "affected_length_m": 0.0, "longest_affected_run_m": 0.0,
      "bridge": BridgeExposure.EXPOSED}, "EXPOSED"),
])
def test_road_impact_invalid(overrides, message):
    with pytest.raises(ValidationError, match=message):
        road(**overrides)


def test_bridge_exposure_cannot_express_structural_collapse():
    assert not any("collapse" in v.value or "destroy" in v.value for v in BridgeExposure)


# ---- F: settlement status -----------------------------------------------------------

def reach(length: float | None, kind: TargetKind = TargetKind.HOSPITAL) -> Reachability:
    if length is None:
        return Reachability(reachable=False)
    return Reachability(reachable=True, target_id="hosp-1", target_kind=kind, route_length_m=length)


@pytest.mark.parametrize(("pre", "opt", "pess", "expected"), [
    (None, None, None, AccessStatus.NO_PRE_EVENT_ROAD_ACCESS),
    (1000, None, None, AccessStatus.CONFIRMED_CUT_OFF),
    (1000, 1200, None, AccessStatus.POSSIBLY_CUT_OFF),
    (1000, 1000, 1600, AccessStatus.DEGRADED),
    (1000, 1000, 1400, AccessStatus.CONNECTED),
    (0, 0, 0, AccessStatus.CONNECTED),          # settlement at the target itself
    (0, 0, 500, AccessStatus.DEGRADED),          # lost its on-site target, now 500 m away
])
def test_classify_settlement_decision_table(pre, opt, pess, expected):
    assert classify_settlement(reach(pre), reach(opt), reach(pess), 1.5) is expected


def settlement(pre, opt, pess, status, **overrides) -> SettlementStatus:
    fields = dict(settlement_id="set-1", name="Example village", source=SettlementSource.OSM_PLACE,
                  location=PointGeometry(coordinates=(85.31, 28.12)), building_count=42,
                  snap_distance_m=35.0, pre_event=reach(pre), post_optimistic=reach(opt),
                  post_pessimistic=reach(pess), degraded_detour_ratio=1.5, status=status,
                  blocking_segment_ids=("seg-1",), evidence_ids=("ev-1",))
    fields.update(overrides)
    return SettlementStatus(**fields)


def test_settlement_status_valid_and_round_trips():
    s = settlement(1000, None, None, AccessStatus.CONFIRMED_CUT_OFF)
    assert SettlementStatus.model_validate_json(s.model_dump_json()) == s


def test_settlement_status_must_match_decision_table():
    with pytest.raises(ValidationError, match="contradicts"):
        settlement(1000, 1200, None, AccessStatus.CONFIRMED_CUT_OFF)


@pytest.mark.parametrize(("pre", "opt", "pess", "message"), [
    (None, 1000, 1000, "cannot make a settlement reachable"),
    (1000, None, 1200, "cannot make a settlement reachable"),
    (1000, 900, 1000, "cannot shorten"),
])
def test_settlement_monotonicity(pre, opt, pess, message):
    with pytest.raises(ValidationError, match=message):
        settlement(pre, opt, pess, AccessStatus.CONNECTED)


def test_cut_off_requires_blocking_segments_and_evidence():
    with pytest.raises(ValidationError, match="blocking segments"):
        settlement(1000, None, None, AccessStatus.CONFIRMED_CUT_OFF, blocking_segment_ids=())


def test_unsnapped_settlement_cannot_be_reachable():
    with pytest.raises(ValidationError, match="not snapped"):
        settlement(1000, 1000, 1000, AccessStatus.CONNECTED, snap_distance_m=None)


def test_reachability_details_must_match_flag():
    with pytest.raises(ValidationError, match="needs target_id"):
        Reachability(reachable=True)
    with pytest.raises(ValidationError, match="must not carry"):
        Reachability(reachable=False, route_length_m=10.0)


# ---- G: evidence ----------------------------------------------------------------------

def evidence(**overrides) -> EvidenceRecord:
    fields = dict(
        evidence_id="ev-test", subject_type=SubjectType.ROAD_SEGMENT, subject_id="seg-1",
        claim="road segment inside debris/disturbance class for a contiguous 30 m run",
        location=LINE,
        sources=(SourceRef(source=DataSource.SENTINEL_1, source_id="TEST-S1-POST",
                           reference_date=date(2026, 8, 30), acquisition_start=utc(2026, 8, 30, 0, 15)),
                 SourceRef(source=DataSource.OSM_PRE_EVENT, source_id="ohsome@2026-07-27",
                           reference_date=date(2026, 7, 27))),
        method="rule:road_blocked_run", method_version="0.1.0", config_fingerprint=FINGERPRINT,
        probability=0.92, values={"longest_run_m": 30.0, "debris_z_vv": 4.1},
        artifacts=(ArtifactRef(path="data/processed/aoi-test/chips/seg-1_post.png", kind="png_chip"),),
        created_at=utc(2026, 10, 1),
    )
    fields.update(overrides)
    return EvidenceRecord(**fields)


def test_evidence_valid_and_round_trips():
    e = evidence()
    assert EvidenceRecord.model_validate_json(e.model_dump_json()) == e


@pytest.mark.parametrize("claim", [
    "bridge collapsed", "village destroyed", "estimated population affected: 300", "2 people dead",
])
def test_evidence_rejects_unsupported_claims(claim):
    with pytest.raises(ValidationError, match="cannot support"):
        evidence(claim=claim)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"sources": ()}, "at least one source"),
    ({"probability": -0.1}, "probability"),
    ({"values": {"z": float("inf")}}, "finite"),
    ({"claim": "  "}, "claim must not be empty"),
])
def test_evidence_invalid(overrides, message):
    with pytest.raises(ValidationError, match=message):
        evidence(**overrides)


def test_evidence_cannot_cite_published_damage_maps():
    with pytest.raises(ValidationError):
        SourceRef(source="copernicus-ems", source_id="EMSR927", reference_date=date(2026, 8, 30))


def test_source_reference_date_must_match_acquisition():
    with pytest.raises(ValidationError, match="reference_date"):
        SourceRef(source=DataSource.SENTINEL_1, source_id="TEST", reference_date=date(2026, 8, 29),
                  acquisition_start=utc(2026, 8, 30, 0, 15))


def test_evidence_id_is_deterministic_and_order_independent():
    a = make_evidence_id(SubjectType.SETTLEMENT, "set-1", "rule:x", FINGERPRINT, ["b", "a"])
    b = make_evidence_id(SubjectType.SETTLEMENT, "set-1", "rule:x", FINGERPRINT, ["a", "b"])
    c = make_evidence_id(SubjectType.SETTLEMENT, "set-2", "rule:x", FINGERPRINT, ["a", "b"])
    assert a == b != c and a.startswith("ev-")


# ---- serialisation helpers -------------------------------------------------------------

def test_save_and_load_contract(tmp_path):
    e = evidence()
    path = save_contract(e, tmp_path / "nested" / "ev.json")
    assert load_contract(EvidenceRecord, path) == e


def test_load_contract_rejects_missing_or_corrupt_files(tmp_path):
    with pytest.raises(ContractIOError, match="not found"):
        load_contract(EvidenceRecord, tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"evidence_id": "ev-1"}', encoding="utf-8")
    with pytest.raises(ContractIOError, match="not a valid EvidenceRecord"):
        load_contract(EvidenceRecord, bad)

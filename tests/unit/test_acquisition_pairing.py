"""Contracts B (acquisition) and C (same-track pre/post pair), plus the
pre-event OSM snapshot rule."""

from __future__ import annotations

from datetime import date, time

import pytest
from pydantic import ValidationError

from bato.contracts.acquisition import Acquisition
from bato.contracts.common import OrbitDirection, Polarization, Sensor
from bato.contracts.osm import OSMSnapshot, select_snapshot_date
from bato.contracts.pairing import PairQualityFlag, PrePostPair

from helpers import EVENT_DATE, FOOTPRINT, s1, utc

PRE_1 = s1("TEST-S1-PRE-1", utc(2026, 8, 2, 0, 15))
PRE_2 = s1("TEST-S1-PRE-2", utc(2026, 8, 14, 0, 15))
POST = s1("TEST-S1-POST", utc(2026, 8, 30, 0, 15))   # hypothetical dates, not a real catalogue


def pair(**overrides) -> PrePostPair:
    fields = dict(
        pair_id="pair-test", event_date=EVENT_DATE, pre=(PRE_1, PRE_2), post=POST,
        orbit_direction=OrbitDirection.DESCENDING, relative_orbit=19, aoi_coverage_fraction=1.0,
        pairing_rationale="same relative orbit; earliest post image after the event",
    )
    fields.update(overrides)
    return PrePostPair(**fields)


# ---- acquisitions ---------------------------------------------------------------

def test_s1_requires_track_information():
    with pytest.raises(ValidationError, match="orbit_direction and relative_orbit"):
        Acquisition(source_id="TEST", sensor=Sensor.SENTINEL_1, acquisition_start=utc(2026, 8, 1),
                    product_type="IW_GRDH_1S", polarizations=(Polarization.VV,), footprint=FOOTPRINT)


def test_s1_relative_orbit_range():
    with pytest.raises(ValidationError, match="1..175"):
        s1("TEST", utc(2026, 8, 1), relative_orbit=176)


def test_s2_requires_cloud_cover_and_rejects_polarizations():
    with pytest.raises(ValidationError, match="cloud_cover_pct"):
        Acquisition(source_id="TEST", sensor=Sensor.SENTINEL_2, acquisition_start=utc(2026, 8, 1),
                    product_type="S2MSI2A", footprint=FOOTPRINT)
    ok = Acquisition(source_id="TEST", sensor=Sensor.SENTINEL_2, acquisition_start=utc(2026, 8, 1),
                     product_type="S2MSI2A", cloud_cover_pct=35.0, footprint=FOOTPRINT)
    assert ok.data_source.value == "sentinel-2"


def test_naive_or_non_utc_datetimes_rejected():
    from datetime import datetime, timedelta, timezone
    with pytest.raises(ValidationError, match="timezone-aware"):
        s1("TEST", datetime(2026, 8, 1, 0, 15))
    with pytest.raises(ValidationError, match="UTC"):
        s1("TEST", datetime(2026, 8, 1, 5, 45, tzinfo=timezone(timedelta(hours=5, minutes=45))))


def test_acquisition_json_round_trip():
    assert Acquisition.model_validate_json(PRE_1.model_dump_json()) == PRE_1


# ---- pairs ----------------------------------------------------------------------

def test_valid_pair_and_derived_fields():
    p = pair()
    assert p.pre_post_gap_days == 16.0
    assert p.post_delay_days == pytest.approx(4.01, abs=0.01)
    assert p.quality_flags == ()


def test_pair_json_round_trip_recomputes_derived_fields():
    p = pair()
    payload = p.model_dump_json()
    assert '"pre_post_gap_days"' in payload
    assert PrePostPair.model_validate_json(payload) == p


def test_cross_track_pair_rejected():
    other_track = s1("TEST-S1-POST-ASC", utc(2026, 8, 30, 12, 30), direction=OrbitDirection.ASCENDING,
                     relative_orbit=85)
    with pytest.raises(ValidationError, match="cross-track"):
        pair(post=other_track)


def test_same_relative_orbit_but_different_direction_rejected():
    flipped = s1("TEST-S1-POST-FLIP", utc(2026, 8, 30, 12, 30), direction=OrbitDirection.ASCENDING)
    with pytest.raises(ValidationError, match="cross-track"):
        pair(post=flipped)


def test_same_day_images_are_ambiguous_without_event_time():
    same_day = s1("TEST-S1-SAMEDAY", utc(2026, 8, 26, 0, 15))
    with pytest.raises(ValidationError, match="ambiguous"):
        pair(post=same_day)
    with pytest.raises(ValidationError, match="ambiguous"):
        pair(pre=(PRE_1, same_day))


def test_event_time_resolves_same_day_ordering():
    same_day_after = s1("TEST-S1-SAMEDAY", utc(2026, 8, 26, 12, 30))
    p = pair(post=same_day_after, event_time_utc=time(6, 0))
    assert p.post_delay_days == pytest.approx(0.271, abs=0.001)
    with pytest.raises(ValidationError, match="after the event time"):
        pair(post=same_day_after, event_time_utc=time(13, 0))


def test_pre_images_must_be_sorted_and_unique():
    with pytest.raises(ValidationError, match="oldest first"):
        pair(pre=(PRE_2, PRE_1))
    with pytest.raises(ValidationError, match="more than once"):
        pair(pre=(PRE_1, PRE_1))


def test_single_pre_image_flag_must_match():
    with pytest.raises(ValidationError, match="SINGLE_PRE_IMAGE"):
        pair(pre=(PRE_2,))
    ok = pair(pre=(PRE_2,), quality_flags=(PairQualityFlag.SINGLE_PRE_IMAGE,))
    assert ok.quality_flags == (PairQualityFlag.SINGLE_PRE_IMAGE,)


def test_pair_requires_shared_polarization_and_rationale():
    vh_only = s1("TEST-S1-POST-VH", utc(2026, 8, 30, 0, 15), pols=(Polarization.VH,))
    vv_only = s1("TEST-S1-PRE-VV", utc(2026, 8, 14, 0, 15), pols=(Polarization.VV,))
    with pytest.raises(ValidationError, match="share no polarization"):
        pair(pre=(vv_only,), post=vh_only, quality_flags=(PairQualityFlag.SINGLE_PRE_IMAGE,))
    with pytest.raises(ValidationError, match="pairing_rationale"):
        pair(pairing_rationale="   ")


# ---- OSM snapshot rule ------------------------------------------------------------

def test_snapshot_is_before_event_and_capped_by_availability():
    assert select_snapshot_date(EVENT_DATE, 1) == date(2026, 8, 25)
    assert select_snapshot_date(EVENT_DATE, 1, latest_available=date(2026, 7, 27)) == date(2026, 7, 27)
    assert select_snapshot_date(date(2021, 6, 15), 1, latest_available=date(2026, 7, 27)) == date(2021, 6, 14)


def test_snapshot_offset_must_exclude_event_day():
    with pytest.raises(ValueError, match="offset_days"):
        select_snapshot_date(EVENT_DATE, 0)


def test_post_event_snapshot_record_rejected():
    with pytest.raises(ValidationError, match="strictly before the event"):
        OSMSnapshot(event_date=EVENT_DATE, snapshot_date=EVENT_DATE, service="ohsome-api")

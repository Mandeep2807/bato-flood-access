"""AOI request validation, normalisation and event-date validation."""

from __future__ import annotations

from datetime import date, time

import pytest
from pydantic import ValidationError

from bato.contracts.aoi import AOIRequest, build_aoi, validate_event_date
from bato.contracts.common import BBox, LonLat, PolygonGeometry
from bato.exceptions import AOIValidationError, EventDateError, InvalidInputError
from bato.utils.geo import spherical_ring_area_km2

from helpers import EVENT_DATE, FIXED_TODAY

SQUARE = ((85.30, 28.10), (85.40, 28.10), (85.40, 28.20), (85.30, 28.20), (85.30, 28.10))


def build(cfg, **request):
    return build_aoi(AOIRequest(**request), cfg.settings, today=FIXED_TODAY)


# ---- geometry inputs --------------------------------------------------------

def test_polygon_request(cfg):
    aoi = build(cfg, event_date=EVENT_DATE, polygon=PolygonGeometry(coordinates=(SQUARE,)))
    assert aoi.input_kind == "polygon"
    assert 100 < aoi.area_km2 < 120          # 0.1 deg x 0.1 deg near 28 N
    assert aoi.bbox.min_lon == 85.30 and aoi.bbox.max_lat == 28.20


def test_bbox_request_becomes_closed_polygon(cfg):
    aoi = build(cfg, event_date=EVENT_DATE,
                bbox=BBox(min_lon=85.30, min_lat=28.10, max_lon=85.40, max_lat=28.20))
    ring = aoi.geometry.exterior
    assert aoi.input_kind == "bbox" and ring[0] == ring[-1] and len(ring) == 5


def test_point_request_uses_configured_box_size(cfg):
    aoi = build(cfg, event_date=EVENT_DATE, point=LonLat(lon=85.32, lat=28.15))
    side = 2 * cfg.settings.aoi.default_point_half_size_km
    assert aoi.area_km2 == pytest.approx(side * side, rel=0.01)


def test_point_request_with_custom_half_size(cfg):
    aoi = build(cfg, event_date=EVENT_DATE, point=LonLat(lon=85.32, lat=28.15), point_half_size_km=5)
    assert aoi.area_km2 == pytest.approx(100, rel=0.01)


def test_aoi_id_is_deterministic(cfg):
    a = build(cfg, event_date=EVENT_DATE, polygon=PolygonGeometry(coordinates=(SQUARE,)))
    b = build(cfg, event_date=EVENT_DATE, polygon=PolygonGeometry(coordinates=(SQUARE,)))
    c = build(cfg, event_date=date(2026, 8, 27), polygon=PolygonGeometry(coordinates=(SQUARE,)))
    assert a.aoi_id == b.aoi_id != c.aoi_id


def test_spherical_area_of_one_degree_cell_at_equator():
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    assert spherical_ring_area_km2(ring) == pytest.approx(12364, rel=0.002)


# ---- invalid requests -------------------------------------------------------

def test_exactly_one_geometry_required():
    with pytest.raises(ValidationError, match="exactly one"):
        AOIRequest(event_date=EVENT_DATE)
    with pytest.raises(ValidationError, match="exactly one"):
        AOIRequest(event_date=EVENT_DATE, point=LonLat(lon=85.3, lat=28.1),
                   bbox=BBox(min_lon=85.3, min_lat=28.1, max_lon=85.4, max_lat=28.2))


def test_half_size_only_with_point():
    with pytest.raises(ValidationError, match="only valid with a point"):
        AOIRequest(event_date=EVENT_DATE, point_half_size_km=5,
                   bbox=BBox(min_lon=85.3, min_lat=28.1, max_lon=85.4, max_lat=28.2))


def test_unknown_field_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        AOIRequest(event_date=EVENT_DATE, point=LonLat(lon=85.3, lat=28.1), radius=3)


@pytest.mark.parametrize(("ring", "message"), [
    (((85.3, 28.1), (85.4, 28.1), (85.4, 28.2), (85.3, 28.2)), "not closed"),
    (((85.3, 28.1), (85.4, 28.2), (85.4, 28.1), (85.3, 28.2), (85.3, 28.1)), "self-intersects"),
    (((85.3, 28.1), (85.4, 28.1), (85.3, 28.1)), "at least 4"),
    (((85.3, 28.1), (85.4, 28.1), (85.5, 28.1), (85.3, 28.1)), "zero area"),
    (((85.3, 95.0), (85.4, 28.1), (85.4, 28.2), (85.3, 95.0)), "latitude"),
    (((179.5, 10.0), (-179.5, 10.0), (-179.5, 11.0), (179.5, 11.0), (179.5, 10.0)), "antimeridian"),
])
def test_malformed_polygons_rejected(ring, message):
    with pytest.raises(ValidationError, match=message):
        PolygonGeometry(coordinates=(ring,))


def test_inverted_bbox_rejected():
    with pytest.raises(ValidationError, match="min_lon < max_lon"):
        BBox(min_lon=85.4, min_lat=28.1, max_lon=85.3, max_lat=28.2)


def test_aoi_too_large_rejected(cfg):
    with pytest.raises(AOIValidationError, match="exceeds the maximum"):
        build(cfg, event_date=EVENT_DATE, bbox=BBox(min_lon=84.0, min_lat=27.0, max_lon=86.0, max_lat=29.0))


def test_aoi_too_small_rejected(cfg):
    with pytest.raises(AOIValidationError, match="below the minimum"):
        build(cfg, event_date=EVENT_DATE,
              bbox=BBox(min_lon=85.300, min_lat=28.100, max_lon=85.301, max_lat=28.101))


def test_aoi_errors_are_input_errors():
    assert issubclass(AOIValidationError, InvalidInputError)
    assert issubclass(EventDateError, InvalidInputError)


# ---- event date ---------------------------------------------------------------

def test_future_event_date_rejected(cfg):
    with pytest.raises(EventDateError, match="future"):
        validate_event_date(date(2026, 10, 2), cfg.settings, today=FIXED_TODAY)


def test_event_before_sentinel1_rejected(cfg):
    with pytest.raises(EventDateError, match="Sentinel-1"):
        validate_event_date(date(2013, 6, 16), cfg.settings, today=FIXED_TODAY)


def test_valid_event_dates_accepted(cfg):
    validate_event_date(EVENT_DATE, cfg.settings, today=FIXED_TODAY)
    validate_event_date(FIXED_TODAY, cfg.settings, today=FIXED_TODAY)
    validate_event_date(cfg.settings.event.min_event_date, cfg.settings, today=FIXED_TODAY)


def test_event_time_must_be_naive_utc():
    from datetime import timezone
    with pytest.raises(ValidationError, match="naive time"):
        AOIRequest(event_date=EVENT_DATE, point=LonLat(lon=85.3, lat=28.1),
                   event_time_utc=time(6, 0, tzinfo=timezone.utc))


def test_event_datetime_combines_date_and_time(cfg):
    aoi = build(cfg, event_date=EVENT_DATE, event_time_utc=time(4, 30), point=LonLat(lon=85.3, lat=28.1))
    assert aoi.event_datetime_utc.isoformat() == "2026-08-26T04:30:00+00:00"

"""Contract A - area of interest and event request.

:class:`AOIRequest` is what a user (or a judge) submits: exactly one of a
polygon, a bounding box or a point, plus the event date. :func:`build_aoi`
applies the configuration-dependent checks (area limits, allowed dates) and
returns a normalised :class:`AOI` whose geometry is always a polygon.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from typing import Literal

from pydantic import model_validator

from bato.config import Settings
from bato.contracts.common import BBox, Contract, LonLat, PolygonGeometry
from bato.exceptions import AOIValidationError, EventDateError
from bato.utils.geo import bbox_ring, spherical_ring_area_km2, square_around_point

InputKind = Literal["polygon", "bbox", "point"]


class AOIRequest(Contract):
    """Raw request. Exactly one of ``polygon``, ``bbox`` or ``point``.

    ``event_time_utc`` is optional. Without it, satellite acquisitions on the
    event date itself are treated as ambiguous and excluded from pairing.
    """

    event_date: date
    event_time_utc: time | None = None
    polygon: PolygonGeometry | None = None
    bbox: BBox | None = None
    point: LonLat | None = None
    point_half_size_km: float | None = None
    label: str | None = None
    metadata: dict[str, str] = {}

    @model_validator(mode="after")
    def _check(self) -> AOIRequest:
        given = [k for k in ("polygon", "bbox", "point") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(f"provide exactly one of polygon, bbox or point (got {given or 'none'})")
        if self.point_half_size_km is not None:
            if self.point is None:
                raise ValueError("point_half_size_km is only valid with a point request")
            if self.point_half_size_km <= 0:
                raise ValueError("point_half_size_km must be positive")
        if self.event_time_utc is not None and self.event_time_utc.tzinfo is not None:
            raise ValueError("event_time_utc must be a naive time interpreted as UTC")
        return self

    @property
    def input_kind(self) -> InputKind:
        if self.polygon is not None:
            return "polygon"
        return "bbox" if self.bbox is not None else "point"


class AOI(Contract):
    """Validated, normalised area of interest for one pipeline run."""

    aoi_id: str
    geometry: PolygonGeometry
    bbox: BBox
    area_km2: float
    event_date: date
    event_time_utc: time | None = None
    input_kind: InputKind
    label: str | None = None
    metadata: dict[str, str] = {}

    @property
    def event_datetime_utc(self) -> datetime | None:
        """Exact event instant if the time is known, else None."""
        if self.event_time_utc is None:
            return None
        return datetime.combine(self.event_date, self.event_time_utc, tzinfo=timezone.utc)


def _aoi_id(ring: list[tuple[float, float]], event_date: date, event_time: time | None) -> str:
    """Deterministic id: the same geometry and event always give the same id."""
    payload = json.dumps(
        {"ring": [[round(x, 7), round(y, 7)] for x, y in ring],
         "event_date": event_date.isoformat(),
         "event_time_utc": event_time.isoformat() if event_time else None},
        separators=(",", ":"),
    )
    return "aoi-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def validate_event_date(event_date: date, settings: Settings, today: date | None = None) -> None:
    """Reject event dates that cannot have Sentinel-1 before/after observations.

    ``today`` is injectable so validation is deterministic in tests.
    """
    today = today or datetime.now(timezone.utc).date()
    if event_date > today:
        raise EventDateError(f"event date {event_date} is in the future (today is {today})")
    if event_date < settings.event.min_event_date:
        raise EventDateError(
            f"event date {event_date} is before {settings.event.min_event_date}, "
            "the earliest date with routine Sentinel-1 coverage"
        )


def build_aoi(request: AOIRequest, settings: Settings, today: date | None = None) -> AOI:
    """Normalise a request into an :class:`AOI` and enforce configured limits.

    Raises:
        EventDateError: the event date is outside the supported range.
        AOIValidationError: the area is too small or too large.
    """
    validate_event_date(request.event_date, settings, today)

    ring: list[tuple[float, float]]
    if request.polygon is not None:
        ring = [(float(lon), float(lat)) for lon, lat in request.polygon.exterior]
    elif request.bbox is not None:
        b = request.bbox
        ring = bbox_ring(b.min_lon, b.min_lat, b.max_lon, b.max_lat)
    elif request.point is not None:
        half = request.point_half_size_km or settings.aoi.default_point_half_size_km
        try:
            ring = square_around_point(request.point.lon, request.point.lat, half)
        except ValueError as exc:
            raise AOIValidationError(f"cannot build a box around the point: {exc}") from exc
    else:  # unreachable: AOIRequest guarantees exactly one geometry
        raise AOIValidationError("request has no geometry")

    area = spherical_ring_area_km2(ring)
    limits = settings.aoi
    if area < limits.min_area_km2:
        raise AOIValidationError(f"AOI area {area:.2f} km2 is below the minimum {limits.min_area_km2} km2")
    if area > limits.max_area_km2:
        raise AOIValidationError(
            f"AOI area {area:.1f} km2 exceeds the maximum {limits.max_area_km2} km2; "
            "choose a smaller area"
        )

    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return AOI(
        aoi_id=_aoi_id(ring, request.event_date, request.event_time_utc),
        geometry=PolygonGeometry(coordinates=(tuple(ring),)),
        bbox=BBox(min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats)),
        area_km2=round(area, 3),
        event_date=request.event_date,
        event_time_utc=request.event_time_utc,
        input_kind=request.input_kind,
        label=request.label,
        metadata=dict(request.metadata),
    )

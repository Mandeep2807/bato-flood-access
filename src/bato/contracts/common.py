"""Shared building blocks for BATO data contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from bato.utils.geo import check_lon_lat, validate_ring


class Contract(BaseModel):
    """Base for all contracts: immutable, unknown fields rejected.

    Computed (derived) fields are written on serialisation for readability,
    and dropped then recomputed on load, so JSON always round-trips and a
    stale derived value in a file can never override the real computation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls.model_computed_fields:
            data = {k: v for k, v in data.items() if k not in cls.model_computed_fields}
        return data


class DataSource(str, Enum):
    """Data sources that MAY be used as pipeline inputs, per the official PS.

    Published damage maps (Copernicus EMS, UNOSAT, ...) and post-event OSM are
    deliberately absent: they exist only as
    :class:`bato.evaluation.reference.ReferenceSource`, which pipeline code
    cannot import.
    """

    SENTINEL_1 = "sentinel-1"
    SENTINEL_2 = "sentinel-2"
    COPERNICUS_DEM = "copernicus-dem"
    OSM_PRE_EVENT = "osm-pre-event"
    KURO_SIWO = "kuro-siwo"            # training data only
    SEN1FLOODS11 = "sen1floods11"      # training data only (optional)
    BATO_DERIVED = "bato-derived"      # an intermediate artifact produced by BATO itself


class Sensor(str, Enum):
    SENTINEL_1 = "sentinel-1"
    SENTINEL_2 = "sentinel-2"


class OrbitDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class Polarization(str, Enum):
    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    if value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be in UTC (offset +00:00)")
    return value


UTCDatetime = Annotated[datetime, AfterValidator(_require_utc)]


class LonLat(Contract):
    lon: float
    lat: float

    @model_validator(mode="after")
    def _check(self) -> LonLat:
        check_lon_lat(self.lon, self.lat)
        return self


class BBox(Contract):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @model_validator(mode="after")
    def _check(self) -> BBox:
        check_lon_lat(self.min_lon, self.min_lat)
        check_lon_lat(self.max_lon, self.max_lat)
        if not (self.min_lon < self.max_lon and self.min_lat < self.max_lat):
            raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
        return self


class PointGeometry(Contract):
    """GeoJSON Point in WGS84, [lon, lat]."""

    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]

    @field_validator("coordinates")
    @classmethod
    def _check(cls, c: tuple[float, float]) -> tuple[float, float]:
        check_lon_lat(*c)
        return c


class LineStringGeometry(Contract):
    """GeoJSON LineString in WGS84."""

    type: Literal["LineString"] = "LineString"
    coordinates: tuple[tuple[float, float], ...]

    @field_validator("coordinates")
    @classmethod
    def _check(cls, c: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        if len(c) < 2:
            raise ValueError("a LineString needs at least 2 positions")
        for lon, lat in c:
            check_lon_lat(lon, lat)
        return c


class PolygonGeometry(Contract):
    """GeoJSON Polygon in WGS84 with a single exterior ring (holes not supported)."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: tuple[tuple[tuple[float, float], ...], ...]

    @field_validator("coordinates")
    @classmethod
    def _check(cls, rings: tuple[tuple[tuple[float, float], ...], ...]) -> tuple:
        if len(rings) != 1:
            raise ValueError("exactly one exterior ring is supported (no holes, no empty polygons)")
        validate_ring(rings[0])
        return rings

    @property
    def exterior(self) -> tuple[tuple[float, float], ...]:
        return self.coordinates[0]


AnyGeometry = Annotated[PointGeometry | LineStringGeometry | PolygonGeometry, Field(discriminator="type")]


# --------------------------------------------------------------------------
# Small field validators shared by several contracts
# --------------------------------------------------------------------------

_HEX = set("0123456789abcdef")


def _require_relative_posix_path(value: str) -> str:
    """Artifact paths are stored relative to the project root, POSIX-style,
    so evidence stays valid when the repository is moved or cloned."""
    if not value or value.strip() != value:
        raise ValueError("path must be a non-empty string without surrounding whitespace")
    if "\\" in value:
        raise ValueError("use forward slashes in stored paths")
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise ValueError("stored paths must be relative to the project root")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("stored paths must not contain empty, '.' or '..' segments")
    return value


def _require_sha256_hex(value: str) -> str:
    if len(value) != 64 or not set(value) <= _HEX:
        raise ValueError("expected a lowercase 64-character SHA-256 hex digest")
    return value


RelativePath = Annotated[str, AfterValidator(_require_relative_posix_path)]
Sha256Hex = Annotated[str, AfterValidator(_require_sha256_hex)]

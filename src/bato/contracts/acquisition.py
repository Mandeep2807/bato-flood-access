"""Contract B - one satellite acquisition (catalogue entry).

Describes *which* scene is used, not its pixels. Pixel data is stored as
files under ``data/raw`` / ``data/interim`` and referenced elsewhere.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from bato.contracts.common import (
    Contract,
    DataSource,
    OrbitDirection,
    Polarization,
    PolygonGeometry,
    Sensor,
    UTCDatetime,
)

S1_RELATIVE_ORBITS = 175  # Sentinel-1 repeat cycle has 175 relative orbits
S2_RELATIVE_ORBITS = 143  # Sentinel-2 has 143 relative orbits


class ProcessingStatus(str, Enum):
    CATALOGUED = "catalogued"      # found in the catalogue, nothing fetched yet
    FETCHED = "fetched"            # raw data downloaded
    PREPROCESSED = "preprocessed"  # calibrated / terrain-corrected / masked
    FAILED = "failed"


class Acquisition(Contract):
    """A single Sentinel-1 or Sentinel-2 acquisition over the AOI."""

    source_id: str                         # provider product identifier
    sensor: Sensor
    platform: str | None = None            # e.g. "S1A", "S1C", "S2B"
    acquisition_start: UTCDatetime
    product_type: str                      # e.g. "IW_GRDH_1S" or "S2MSI2A"
    orbit_direction: OrbitDirection | None = None
    relative_orbit: int | None = None
    polarizations: tuple[Polarization, ...] = ()
    cloud_cover_pct: float | None = None   # Sentinel-2 scene cloud cover
    footprint: PolygonGeometry
    processing_status: ProcessingStatus = ProcessingStatus.CATALOGUED

    @model_validator(mode="after")
    def _check(self) -> Acquisition:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.sensor is Sensor.SENTINEL_1:
            if self.orbit_direction is None or self.relative_orbit is None:
                raise ValueError("Sentinel-1 acquisitions need orbit_direction and relative_orbit")
            if not 1 <= self.relative_orbit <= S1_RELATIVE_ORBITS:
                raise ValueError(f"Sentinel-1 relative_orbit must be in 1..{S1_RELATIVE_ORBITS}")
            if not self.polarizations:
                raise ValueError("Sentinel-1 acquisitions need at least one polarization")
            if self.cloud_cover_pct is not None:
                raise ValueError("cloud_cover_pct does not apply to Sentinel-1")
        else:
            if self.cloud_cover_pct is None or not 0 <= self.cloud_cover_pct <= 100:
                raise ValueError("Sentinel-2 acquisitions need cloud_cover_pct in [0, 100]")
            if self.relative_orbit is not None and not 1 <= self.relative_orbit <= S2_RELATIVE_ORBITS:
                raise ValueError(f"Sentinel-2 relative_orbit must be in 1..{S2_RELATIVE_ORBITS}")
            if self.polarizations:
                raise ValueError("polarizations do not apply to Sentinel-2")
        if len(set(self.polarizations)) != len(self.polarizations):
            raise ValueError("duplicate polarizations")
        return self

    @property
    def data_source(self) -> DataSource:
        return DataSource.SENTINEL_1 if self.sensor is Sensor.SENTINEL_1 else DataSource.SENTINEL_2

    @property
    def track_key(self) -> tuple[OrbitDirection | None, int | None]:
        """(orbit direction, relative orbit): acquisitions are pixel-comparable
        only when this key is identical."""
        return (self.orbit_direction, self.relative_orbit)

"""OSM snapshot rule (data-rule contract).

The PS allows OpenStreetMap only "as it was before the event". The snapshot
date is therefore derived from each run's event date - never a fixed date - so
a judge-chosen historical event cannot receive OSM edits made after it.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Literal

from pydantic import model_validator

from bato.contracts.common import BBox, Contract, DataSource, RelativePath, Sha256Hex, UTCDatetime


def select_snapshot_date(event_date: date, offset_days: int, latest_available: date | None = None) -> date:
    """Snapshot date = ``event_date - offset_days``, capped at the latest date
    the OSM history service provides.

    Raises:
        ValueError: if ``offset_days`` < 1.
    """
    if offset_days < 1:
        raise ValueError("offset_days must be >= 1 so same-day (possibly post-event) edits are excluded")
    snapshot = event_date - timedelta(days=offset_days)
    if latest_available is not None and latest_available < snapshot:
        snapshot = latest_available
    return snapshot


class OSMSnapshot(Contract):
    """Record of which OSM state a run used."""

    event_date: date
    snapshot_date: date
    latest_available: date | None = None
    service: str                 # e.g. "ohsome-api"
    retrieved_feature_count: int | None = None

    @model_validator(mode="after")
    def _check(self) -> OSMSnapshot:
        if self.snapshot_date >= self.event_date:
            raise ValueError("OSM snapshot must be strictly before the event date (post-event edits are prohibited)")
        if self.latest_available is not None and self.snapshot_date > self.latest_available:
            raise ValueError("snapshot_date cannot be later than the latest available OSM date")
        if self.retrieved_feature_count is not None and self.retrieved_feature_count < 0:
            raise ValueError("retrieved_feature_count must be >= 0")
        return self


# --------------------------------------------------------------------------
# Stored pre-event extract (bato.osm.extract)
# --------------------------------------------------------------------------

OSM_EXTRACT_VERSION: Literal["bato-osm-ohsome-v2"] = "bato-osm-ohsome-v2"


class OSMLayerName(str, Enum):
    ROADS = "roads"            # highway ways (geometry, clipped to the analysis box)
    PLACES = "places"          # place=* features (client-side centroids)
    HOSPITALS = "hospitals"    # amenity=hospital / healthcare=hospital (client-side centroids)
    BUILDINGS = "buildings"    # building footprints (client-side centroids)


class OSMLayerRecord(Contract):
    layer: OSMLayerName
    endpoint: str                      # ohsome v2 endpoint: "extraction/features.parquet"
    filter: str                        # ohsome filter expression sent
    bbox: BBox                         # request "aoi" box (clipping box for geometries)
    time: str                          # request "time": the snapshot instant, e.g. "2026-08-25T00:00:00Z"
    clip: bool                         # request "clip"
    post_processing: str | None        # e.g. "client-side centroid" (v2 has no centroid endpoint)
    file: RelativePath                 # stored GeoParquet response, relative to the project root
    sha256: Sha256Hex
    feature_count: int

    @model_validator(mode="after")
    def _check(self) -> OSMLayerRecord:
        if self.feature_count < 0:
            raise ValueError("feature_count must be >= 0")
        return self


class OSMExtractProvenance(Contract):
    """Record of one pre-event OSM extract. Loaders refuse an extract whose snapshot is not strictly
    before the event, whose files changed, or whose features carry a later snapshot timestamp."""

    kind: Literal["osm_pre_event_extract"] = "osm_pre_event_extract"
    version: Literal["bato-osm-ohsome-v2"] = OSM_EXTRACT_VERSION
    data_source: DataSource = DataSource.OSM_PRE_EVENT
    bato_version: str
    aoi_id: str
    aoi_bbox: BBox
    analysis_bbox: BBox                # AOI + aoi.analysis_buffer_km (roads, places, hospitals)
    settlement_bbox: BBox              # AOI + osm.building_margin_km (buildings)
    snapshot: OSMSnapshot
    api_base_url: str
    api_version: str | None = None
    attribution: str
    retrieved_at: UTCDatetime
    layers: tuple[OSMLayerRecord, ...]

    @model_validator(mode="after")
    def _check(self) -> OSMExtractProvenance:
        names = [layer.layer for layer in self.layers]
        if sorted(n.value for n in names) != sorted(n.value for n in OSMLayerName):
            raise ValueError("an extract must contain every OSM layer exactly once")
        if self.retrieved_at.date() < self.snapshot.snapshot_date:
            raise ValueError("retrieved_at cannot be before the snapshot date")
        expected = f"{self.snapshot.snapshot_date.isoformat()}T00:00:00Z"
        if any(layer.time != expected for layer in self.layers):
            raise ValueError(f"every layer must be requested at the snapshot instant {expected}")
        return self

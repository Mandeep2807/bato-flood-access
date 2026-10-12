"""Step 2C contracts: what to fetch, where it came from, and what happened.

S1FetchPlan      which candidates / passes / slices (CDSE products) to fetch.
DEMPlan          which Copernicus DEM 1x1 degree tiles cover the AOI (+buffer).
DownloadProvenance  self-contained record written next to every downloaded file.
FetchManifest    outcome of one fetch run: per file, per pass (complete only when
                 ALL its slices are present) and per candidate (ready only when
                 all three passes are complete).

A multi-slice pass is ONE temporal observation; its slices are fetched
individually and stay separate files. Mosaicking is a preprocessing task.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import computed_field, model_validator

from bato.contracts.catalogue import CandidateRole
from bato.contracts.common import (
    Contract,
    DataSource,
    OrbitDirection,
    Polarization,
    PolygonGeometry,
    RelativePath,
    Sha256Hex,
    UTCDatetime,
)
from bato.contracts.temporal import TrackKey

# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class PlannedScene(Contract):
    """One CDSE product (one slice) to fetch."""

    product_id: str                     # CDSE product UUID
    product_name: str                   # ESA product name
    pass_id: str
    slice_index: int                    # 0-based position within the pass
    slice_count: int
    platform: str
    acquisition_start: UTCDatetime
    orbit_direction: OrbitDirection
    relative_orbit: int
    product_type: str
    polarizations: tuple[Polarization, ...]
    candidate_ids: tuple[str, ...]      # candidates that need this slice

    @model_validator(mode="after")
    def _check(self) -> PlannedScene:
        if not self.product_id.strip() or not self.product_name.strip():
            raise ValueError("product_id and product_name are required")
        if not 0 <= self.slice_index < self.slice_count:
            raise ValueError("slice_index must be in [0, slice_count)")
        if not self.candidate_ids:
            raise ValueError("a planned scene must be needed by at least one candidate")
        return self


class PlannedPass(Contract):
    pass_id: str
    track: TrackKey
    role: CandidateRole
    start: UTCDatetime
    product_ids: tuple[str, ...]        # all slices, oldest first

    @computed_field  # type: ignore[prop-decorator]
    @property
    def multi_slice(self) -> bool:
        return len(self.product_ids) > 1


class PlannedCandidate(Contract):
    candidate_id: str
    track: TrackKey
    rank: int
    pre_event_pass_ids: tuple[str, ...]
    post_event_pass_id: str


class S1FetchPlan(Contract):
    aoi_id: str
    event_date: date
    selection_digest: Sha256Hex          # SHA-256 of the TemporalSelectionResult JSON
    catalogue_digest: Sha256Hex          # must equal selection.catalogue_digest
    candidates: tuple[PlannedCandidate, ...]
    passes: tuple[PlannedPass, ...]
    scenes: tuple[PlannedScene, ...]     # unique products, fetch order

    @model_validator(mode="after")
    def _check(self) -> S1FetchPlan:
        if not self.candidates:
            raise ValueError("a fetch plan needs at least one candidate")
        ids = [s.product_id for s in self.scenes]
        if len(set(ids)) != len(ids):
            raise ValueError("each product appears once in a plan")
        passes = {p.pass_id: p for p in self.passes}
        if len(passes) != len(self.passes):
            raise ValueError("duplicate pass ids")
        by_pass: dict[str, list[str]] = {}
        for s in self.scenes:
            by_pass.setdefault(s.pass_id, []).append(s.product_id)
            p = passes.get(s.pass_id)
            if p is None:
                raise ValueError(f"scene {s.product_name} references unknown pass {s.pass_id}")
            if (s.orbit_direction, s.relative_orbit) != (p.track.orbit_direction, p.track.relative_orbit):
                raise ValueError(f"scene {s.product_name} is not on its pass's track")
        for p in self.passes:  # every slice of every planned pass is planned
            if sorted(by_pass.get(p.pass_id, [])) != sorted(p.product_ids):
                raise ValueError(f"pass {p.pass_id}: not all slices are planned")
        for c in self.candidates:
            for pid in (*c.pre_event_pass_ids, c.post_event_pass_id):
                if pid not in passes or passes[pid].track != c.track:
                    raise ValueError(f"candidate {c.candidate_id} references a pass off its track")
        return self


class DEMTile(Contract):
    tile_id: str                         # e.g. "N27E085" (lower-left corner)
    product_id: str | None = None        # None when the catalogue has no product for this cell
    product_name: str | None = None
    footprint: PolygonGeometry | None = None


class DEMPlan(Contract):
    aoi_id: str
    dataset: str
    buffer_km: float
    required_tiles: tuple[str, ...]      # every 1x1 degree cell touched by AOI + buffer
    tiles: tuple[DEMTile, ...]           # one per required cell, product or None
    query_filter: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_tiles(self) -> tuple[str, ...]:
        return tuple(t.tile_id for t in self.tiles if t.product_id is None)


# ---------------------------------------------------------------------------
# Provenance and run results
# ---------------------------------------------------------------------------


class DownloadProvenance(Contract):
    """Written as ``<file>.provenance.json`` beside every downloaded file."""

    kind: Literal["sentinel1_grd", "copernicus_dem_tile"]
    data_source: DataSource
    data_role: Literal["pipeline_input_raw"] = "pipeline_input_raw"
    product_id: str
    product_name: str
    source: str                          # e.g. "cdse-odata-download"
    source_url: str                      # no credentials, no query string
    content_length: int | None
    server_checksums: dict[str, str]     # algorithm (lower case) -> value, as published
    checksum_verified_with: str | None   # algorithm used to verify, if any
    checksum_verified: bool
    sha256: Sha256Hex                    # of the local file, always computed
    size_bytes: int
    local_path: RelativePath             # relative to the project root
    content_check: str                   # result of the structural check of the file
    downloaded_at: UTCDatetime
    attempts: int
    resumed: bool
    bato_version: str
    config_fingerprint: str
    # Sentinel-1 only
    platform: str | None = None
    acquisition_start: UTCDatetime | None = None
    orbit_direction: OrbitDirection | None = None
    relative_orbit: int | None = None
    polarizations: tuple[Polarization, ...] = ()
    product_type: str | None = None
    pass_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    # DEM only
    tile_id: str | None = None
    dem_dataset: str | None = None

    @model_validator(mode="after")
    def _check(self) -> DownloadProvenance:
        if "?" in self.source_url or "token" in self.source_url.lower():
            raise ValueError("source_url must not carry a query string or credentials")
        if self.size_bytes < 0 or self.attempts < 1:
            raise ValueError("invalid size_bytes or attempts")
        if self.checksum_verified != (self.checksum_verified_with is not None):
            raise ValueError("checksum_verified must match checksum_verified_with")
        if self.kind == "sentinel1_grd":
            if self.data_source is not DataSource.SENTINEL_1:
                raise ValueError("a Sentinel-1 file must have data_source sentinel-1")
            missing = [n for n in ("platform", "acquisition_start", "orbit_direction", "relative_orbit",
                                   "product_type") if getattr(self, n) is None]
            if missing or not self.polarizations or not self.pass_ids:
                raise ValueError(f"Sentinel-1 provenance is missing {missing or 'polarizations/pass_ids'}")
        else:
            if self.data_source is not DataSource.COPERNICUS_DEM:
                raise ValueError("a DEM tile must have data_source copernicus-dem")
            if not self.tile_id or not self.dem_dataset:
                raise ValueError("DEM provenance needs tile_id and dem_dataset")
        return self


class FetchStatus(str, Enum):
    PLANNED = "planned"                  # dry run: nothing fetched
    DOWNLOADED = "downloaded"
    SKIPPED_EXISTING = "skipped_existing"  # already present and verified
    NOT_ATTEMPTED = "not_attempted"      # beyond --max-scenes
    FAILED = "failed"


class FileFetchResult(Contract):
    kind: Literal["sentinel1_grd", "copernicus_dem_tile"]
    product_id: str
    product_name: str
    pass_id: str | None = None           # Sentinel-1 only
    tile_id: str | None = None           # DEM only
    status: FetchStatus
    size_bytes: int | None = None
    local_path: RelativePath | None = None
    provenance_path: RelativePath | None = None
    message: str = ""

    @model_validator(mode="after")
    def _check(self) -> FileFetchResult:
        present = self.status in (FetchStatus.DOWNLOADED, FetchStatus.SKIPPED_EXISTING)
        if present and (self.local_path is None or self.provenance_path is None):
            raise ValueError("a present file needs local_path and provenance_path")
        if self.status is FetchStatus.FAILED and not self.message:
            raise ValueError("a failure needs a message")
        return self

    @property
    def present(self) -> bool:
        return self.status in (FetchStatus.DOWNLOADED, FetchStatus.SKIPPED_EXISTING)


class FetchManifest(Contract):
    aoi_id: str
    dry_run: bool
    started_at: UTCDatetime
    finished_at: UTCDatetime
    selection_digest: Sha256Hex
    plan: S1FetchPlan
    dem_plan: DEMPlan | None
    files: tuple[FileFetchResult, ...]
    pass_complete: dict[str, bool]       # every slice of the pass present
    candidate_ready: dict[str, bool]     # all three passes complete

    @model_validator(mode="after")
    def _check(self) -> FetchManifest:
        present = {f.product_id for f in self.files if f.kind == "sentinel1_grd" and f.present}
        for p in self.plan.passes:
            expected = all(pid in present for pid in p.product_ids)
            if self.pass_complete.get(p.pass_id) != expected:
                raise ValueError(f"pass_complete[{p.pass_id}] contradicts the file results")
        for c in self.plan.candidates:
            expected = all(self.pass_complete[pid] for pid in (*c.pre_event_pass_ids, c.post_event_pass_id))
            if self.candidate_ready.get(c.candidate_id) != expected:
                raise ValueError(f"candidate_ready[{c.candidate_id}] contradicts pass completeness")
        return self

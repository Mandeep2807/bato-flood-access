"""Step 3.5 contracts: the co-registered Sentinel-1 temporal stack (PRE-1, PRE-2, POST).

The stack is the "pair stack" the architecture feeds to flood/debris detection: the three
terrain-corrected Step 3.4 outputs of ONE temporal-set candidate, cut to one common EPSG:3857
10 m window by whole-pixel offsets on their shared standard grid (no resampling), values copied
unchanged (linear sigma0, no dB, no normalisation, no clipping), plus a validity mask.

Layout v1 (BATO storage order, explicit band names; NOT a model channel order):
    1 PRE1_VV  2 PRE1_VH  3 PRE2_VV  4 PRE2_VH  5 POST_VV  6 POST_VH
PRE-1 = earlier pre-event pass, PRE-2 = later pre-event pass (docs/assumptions.md #13).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from bato.contracts.common import (
    BBox,
    Contract,
    DataSource,
    OrbitDirection,
    Polarization,
    PolygonGeometry,
    RelativePath,
    Sha256Hex,
    UTCDatetime,
)
from bato.contracts.preprocessing import BandStatistics, ValidationCheck

LAYOUT_VERSION = "bato-s1-stack-v1"
ROLES = ("PRE1", "PRE2", "POST")


class StackInputRecord(Contract):
    """One published Step 3.4 output used by the stack (re-hashed before and after)."""

    role: Literal["PRE1", "PRE2", "POST"]
    pass_id: str
    product_id: str
    product_name: str
    acquisition_start: UTCDatetime
    days_from_event: float                 # from 00:00 UTC of the event date (or the event time)
    orbit_direction: OrbitDirection
    relative_orbit: int
    terrain_directory: RelativePath
    terrain_provenance: RelativePath
    terrain_provenance_sha256: Sha256Hex
    files_sha256: dict[str, Sha256Hex]     # equal to the Step 3.4 provenance output_sha256 (verified)
    terrain_config_fingerprint: str
    terrain_graph_template_sha256: Sha256Hex
    source_grid: tuple[float, ...]         # DIMAP IMAGE_TO_MODEL_TRANSFORM of the Step 3.4 output
    source_size: tuple[int, int]           # width, height
    window: tuple[int, int, int, int]      # col_off, row_off, width, height read from the source


class StackGrid(Contract):
    crs: Literal["EPSG:3857"] = "EPSG:3857"
    pixel_size_m: float
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]   # rasterio order a, b, c, d, e, f (corner)
    bounds_3857: tuple[float, float, float, float]               # west, south, east, north
    footprint: PolygonGeometry                                    # lon/lat corners
    standard_grid_offsets: tuple[float, float]                    # pixel-centre position on the standard grid
    aoi_bbox: BBox
    analysis_buffer_km: float
    analysis_bbox: BBox                                           # AOI bbox + analysis buffer (before snapping)


class StackLayer(Contract):
    index: int                              # 1-based band number in the stack GeoTIFF
    name: str                               # e.g. PRE1_VV
    role: Literal["PRE1", "PRE2", "POST"]
    polarization: Polarization
    pass_id: str
    acquisition_start: UTCDatetime
    source_band_file: RelativePath
    copied_sha256: Sha256Hex                # SHA-256 of the copied float32 values (source window == stack band)


class StackValidation(Contract):
    passed: bool
    checks: tuple[ValidationCheck, ...]
    layers: tuple[BandStatistics, ...] = ()
    mask_valid_pixels: int | None = None    # pixels valid in ALL six layers
    mask_total_pixels: int | None = None
    aoi_valid_fraction: float | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> StackValidation:
        if not self.checks:
            raise ValueError("a validation needs at least one check")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must equal all(check.passed)")
        return self

    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]


class StackOutputRecord(Contract):
    directory: RelativePath
    stack: RelativePath
    stack_sha256: Sha256Hex
    mask: RelativePath
    mask_sha256: Sha256Hex
    dtype: Literal["float32"] = "float32"
    nodata: float
    mask_values: str = "1 = valid in all six layers, 0 = not"
    compression: str


class StackProvenance(Contract):
    kind: Literal["s1_temporal_stack"] = "s1_temporal_stack"
    stage: Literal["stack"] = "stack"
    status: Literal["succeeded", "failed"]
    error: str | None = None
    data_source: DataSource = DataSource.SENTINEL_1
    data_role: Literal["pipeline_intermediate"] = "pipeline_intermediate"
    layout_version: str = LAYOUT_VERSION
    value_domain: str = ("linear sigma0 (intensity) exactly as produced by Step 3.4: no dB, no normalisation, "
                         "no clipping, no resampling")
    dem_features: str = "none (DEM channel/slope for the model is an OPEN specification item)"
    bato_version: str
    config_fingerprint: str
    settings: dict[str, Any]
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    aoi_polygon: PolygonGeometry
    candidate_id: str
    track: str
    event_date: str
    event_time_utc: str | None
    inputs: tuple[StackInputRecord, ...]
    grid: StackGrid | None
    layers: tuple[StackLayer, ...] = ()
    output: StackOutputRecord | None
    validation: StackValidation | None
    open_specification_items: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> StackProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if [i.role for i in self.inputs] != list(ROLES):
            raise ValueError("inputs must be PRE1, PRE2, POST in this order")
        if self.status == "succeeded":
            if self.error is not None or self.output is None or self.validation is None or self.grid is None:
                raise ValueError("a successful run needs grid, output and validation, and no error")
            if not self.validation.passed:
                raise ValueError("a run whose validation failed cannot be 'succeeded'")
            if [layer.name for layer in self.layers] != [f"{r}_{p}" for r in ROLES for p in ("VV", "VH")]:
                raise ValueError("layers must follow layout v1")
        elif not self.error:
            raise ValueError("a failed run needs an error message")
        return self

"""Step 3.2/3.3/3.4 contracts: provenance and validation of one preprocessed Sentinel-1 pass.

Step 3.3 (Lee Sigma speckle filter) adds SpeckleProvenance / SpeckleValidation /
BandComparison, and Step 3.4 (terrain correction) TerrainProvenance / TerrainValidation /
DEMMosaicRecord, at the end of this module; earlier classes are unchanged.

PreprocessingProvenance is written as ``preprocessing_provenance.json`` inside
the stage folder (``<interim>/preprocessing/<aoi_id>/<pass_id>/radiometric/``),
for successful AND failed runs (a failed run's folder is renamed
``radiometric.failed-<UTC time>`` and is never presented as a valid output).

Statistics are DIAGNOSTIC: they describe the linear sigma0 values and carry
no acceptance thresholds. Only structural checks (bands present, readable,
float, same grid, radar geometry, linear units, precise orbit, AOI covered,
at least one valid pixel) decide whether the output is valid.
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


class BandStatistics(Contract):
    """Pixel statistics of one output band (linear sigma0, no clipping, no dB)."""

    band: str                          # DIMAP band name, e.g. Sigma0_VV
    polarization: Polarization
    file: RelativePath                 # ENVI .img read with rasterio
    dtype: str                         # as reported by rasterio
    unit: str | None                   # DIMAP PHYSICAL_UNIT (SNAP: "intensity" = linear)
    nodata_value: float | None         # DIMAP NO_DATA_VALUE when NO_DATA_VALUE_USED is true
    width: int
    height: int
    total_pixels: int
    finite_pixels: int
    nonfinite_pixels: int              # NaN or +-inf
    nodata_pixels: int                 # finite pixels equal to nodata_value
    zero_pixels: int
    negative_pixels: int
    valid_pixels: int                  # finite and not nodata; statistics below use these only
    min: float | None
    max: float | None
    mean: float | None
    median: float | None               # from the quantile subsample
    quantiles: dict[str, float]        # p01 ... p99 over valid pixels of the subsample
    quantile_sample_stride: int        # every N-th row and column
    quantile_sample_size: int          # valid pixels in the subsample
    edge_width_px: int                 # = border_limit: the strip Remove-GRD-Border-Noise may trim
    zero_pixels_edge: int              # zeros within edge_width_px of the image edge
    zero_pixels_interior: int

    @model_validator(mode="after")
    def _check(self) -> BandStatistics:
        if self.total_pixels != self.width * self.height:
            raise ValueError("total_pixels must equal width * height")
        if self.finite_pixels + self.nonfinite_pixels != self.total_pixels:
            raise ValueError("finite + nonfinite must equal total")
        if self.valid_pixels != self.finite_pixels - self.nodata_pixels:
            raise ValueError("valid = finite - nodata")
        if self.zero_pixels_edge + self.zero_pixels_interior != self.zero_pixels:
            raise ValueError("edge + interior zeros must equal all zeros")
        return self


class ValidationCheck(Contract):
    name: str
    passed: bool
    detail: str


class RadiometricValidation(Contract):
    passed: bool                       # all checks passed
    checks: tuple[ValidationCheck, ...]
    width: int | None = None
    height: int | None = None
    geometry: str | None = None        # "radar (tie-point geocoding)" expected at this stage
    rasterio_crs: str | None = None    # radar geometry: rasterio reports no CRS
    rasterio_transform: tuple[float, ...] | None = None   # (a, b, c, d, e, f); identity in radar geometry
    range_spacing_m: float | None = None
    azimuth_spacing_m: float | None = None
    footprint: PolygonGeometry | None = None              # corners of the output's lat/lon tie-point grids
    aoi_coverage: float | None = None                     # of the output footprint
    input_scene_aoi_coverage: float | None = None         # of the input scene (Step 2A catalogue)
    orbit_file: str | None = None
    processing_history: tuple[str, ...] = ()              # operators recorded by SNAP in the product
    bands: tuple[BandStatistics, ...] = ()
    warnings: tuple[str, ...] = ()
    rasterio_version: str | None = None
    gdal_version: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RadiometricValidation:
        if not self.checks:
            raise ValueError("a validation needs at least one check")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must equal all(check.passed)")
        return self

    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]


class InputProductRecord(Contract):
    product_id: str
    product_name: str
    local_path: RelativePath
    size_bytes: int
    sha256: Sha256Hex                  # from the Step 2C download provenance (file size re-checked)
    sha256_source: str
    download_provenance: RelativePath
    platform: str | None
    acquisition_start: UTCDatetime
    orbit_direction: OrbitDirection
    relative_orbit: int
    polarizations: tuple[Polarization, ...]
    safe_samples: int                  # full-scene image size from the SAFE annotation
    safe_lines: int
    safe_range_spacing_m: float
    safe_azimuth_spacing_m: float


class AOIRecord(Contract):
    aoi_id: str
    source: str                        # where the AOI geometry was read from
    catalogue_path: str
    catalogue_digest: Sha256Hex
    aoi_polygon: PolygonGeometry
    aoi_bbox: BBox
    crop_buffer_m: float
    crop_bbox: BBox                    # AOI bbox grown by crop_buffer_m
    geo_region_wkt: str                # Subset geoRegion actually used


class OperatorRecord(Contract):
    node: str
    operator: str
    parameters: dict[str, str]         # exactly as written in the resolved graph


class GraphRecord(Contract):
    template_path: str
    template_sha256: Sha256Hex
    resolved_path: RelativePath        # the graph actually executed (paths inside point to the temp folder)
    resolved_sha256: Sha256Hex
    operators: tuple[OperatorRecord, ...]
    operator_parameters_checked: bool  # every parameter name confirmed with `gpt <Operator> -h`


class SnapRecord(Contract):
    gpt_path: str                      # machine-specific executable actually used
    gpt_source: str                    # config | env:BATO_SNAP_GPT | default | PATH
    snap_version: str | None
    diagnostics: tuple[str, ...]       # head of `gpt --diag` (memory, cache, threads as configured in SNAP)
    command: tuple[str, ...]
    return_code: int | None
    runtime_s: float | None
    log_path: RelativePath | None


class OutputRecord(Contract):
    directory: RelativePath
    dim: RelativePath
    data_dir: RelativePath
    band_files: dict[str, RelativePath]


class PreprocessingProvenance(Contract):
    kind: Literal["s1_radiometric"] = "s1_radiometric"
    stage: Literal["radiometric"] = "radiometric"
    status: Literal["succeeded", "failed"]
    error: str | None = None
    data_source: DataSource = DataSource.SENTINEL_1
    data_role: Literal["pipeline_intermediate"] = "pipeline_intermediate"
    product_level: str = "linear sigma0 VV/VH, radar geometry (no speckle filter, no terrain correction, no dB)"
    bato_version: str
    config_fingerprint: str
    settings: dict[str, Any]           # preprocessing settings used (crop buffer + s1_radiometric)
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    pass_id: str
    input: InputProductRecord
    aoi: AOIRecord
    graph: GraphRecord
    snap: SnapRecord
    output: OutputRecord | None
    validation: RadiometricValidation | None

    @model_validator(mode="after")
    def _check(self) -> PreprocessingProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.status == "succeeded":
            if self.error is not None or self.output is None or self.validation is None:
                raise ValueError("a successful run needs output and validation, and no error")
            if not self.validation.passed:
                raise ValueError("a run whose validation failed cannot be 'succeeded'")
        elif not self.error:
            raise ValueError("a failed run needs an error message")
        return self


# ---------------------------------------------------------------------------
# Step 3.3: Lee Sigma speckle filter
# ---------------------------------------------------------------------------


class BandComparison(Contract):
    """Pixel-by-pixel comparison of one band before and after filtering (diagnostic,
    except ``new_nonfinite_pixels``, which is checked: filtering finite data must stay finite)."""

    band: str
    compared_pixels: int               # pixels valid (finite, not no-data) in the input
    changed_pixels: int                # of those, value differs in the output
    new_nonfinite_pixels: int          # valid in the input, NaN/inf in the output
    new_nodata_pixels: int             # valid in the input, no-data (or 0) in the output
    restored_pixels: int               # no-data in the input, valid in the output
    new_negative_pixels: int           # valid (>= 0) in the input, negative in the output
    mean_before: float | None          # over pixels valid in both
    mean_after: float | None
    mean_abs_difference: float | None
    cv_before: float | None            # coefficient of variation (std/mean) over pixels valid in both
    cv_after: float | None

    @model_validator(mode="after")
    def _check(self) -> BandComparison:
        if not 0 <= self.changed_pixels <= self.compared_pixels:
            raise ValueError("changed_pixels must be within compared_pixels")
        return self


class SpeckleValidation(Contract):
    passed: bool
    checks: tuple[ValidationCheck, ...]
    width: int | None = None
    height: int | None = None
    geometry: str | None = None
    rasterio_crs: str | None = None
    rasterio_transform: tuple[float, ...] | None = None
    range_spacing_m: float | None = None
    azimuth_spacing_m: float | None = None
    footprint: PolygonGeometry | None = None
    orbit_file: str | None = None
    processing_history: tuple[str, ...] = ()
    bands_before: tuple[BandStatistics, ...] = ()
    bands_after: tuple[BandStatistics, ...] = ()
    comparisons: tuple[BandComparison, ...] = ()
    warnings: tuple[str, ...] = ()
    rasterio_version: str | None = None
    gdal_version: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SpeckleValidation:
        if not self.checks:
            raise ValueError("a validation needs at least one check")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must equal all(check.passed)")
        return self

    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]


class SpeckleInputRecord(Contract):
    """The published Step 3.2 artifact that was filtered (never modified)."""

    stage: Literal["radiometric"] = "radiometric"
    directory: RelativePath
    dim: RelativePath
    provenance_path: RelativePath
    provenance_sha256: Sha256Hex
    files_sha256: dict[str, Sha256Hex]     # relative path -> SHA-256 (.dim + band .img/.hdr), hashed now
    product_id: str
    product_name: str
    acquisition_start: UTCDatetime
    orbit_direction: OrbitDirection
    relative_orbit: int
    radiometric_config_fingerprint: str
    radiometric_graph_sha256: Sha256Hex
    radiometric_settings: dict[str, Any]
    width: int
    height: int


class SpeckleProvenance(Contract):
    kind: Literal["s1_speckle"] = "s1_speckle"
    stage: Literal["speckle"] = "speckle"
    status: Literal["succeeded", "failed"]
    error: str | None = None
    data_source: DataSource = DataSource.SENTINEL_1
    data_role: Literal["pipeline_intermediate"] = "pipeline_intermediate"
    product_level: str = ("Lee Sigma filtered linear sigma0 VV/VH, radar geometry "
                          "(no terrain correction, no dB, no normalisation)")
    bato_version: str
    config_fingerprint: str
    settings: dict[str, Any]               # preprocessing.s1_speckle as used
    lee_sigma: dict[str, str]              # exact Speckle-Filter parameters as executed
    snap_value_sets: dict[str, Any]        # value sets confirmed with `gpt Speckle-Filter -h`
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    pass_id: str
    input: SpeckleInputRecord
    graph: GraphRecord
    snap: SnapRecord
    output: OutputRecord | None
    output_sha256: dict[str, Sha256Hex] = {}
    validation: SpeckleValidation | None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> SpeckleProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.status == "succeeded":
            if self.error is not None or self.output is None or self.validation is None or not self.output_sha256:
                raise ValueError("a successful run needs output, output hashes and validation, and no error")
            if not self.validation.passed:
                raise ValueError("a run whose validation failed cannot be 'succeeded'")
        elif not self.error:
            raise ValueError("a failed run needs an error message")
        return self


# ---------------------------------------------------------------------------
# Step 3.4: Range-Doppler terrain correction
# ---------------------------------------------------------------------------


class DEMTileRecord(Contract):
    """One Step 2C Copernicus DEM tile used for the mosaic (archive re-hashed now)."""

    tile_id: str
    product_name: str
    archive: RelativePath
    archive_sha256: Sha256Hex           # equals the Step 2C download provenance
    download_provenance: RelativePath
    raster_member: str                  # the *_DEM.tif inside the archive (or the file itself)
    width: int
    height: int
    resolution_deg: tuple[float, float]
    crs: str


class DEMMosaicRecord(Contract):
    """The external DEM given to SNAP: BATO's own Step 2C tiles, mosaicked (values unchanged)."""

    path: RelativePath
    sha256: Sha256Hex
    dataset: str                        # e.g. COP-DEM_GLO-30-DGED/2024_1
    vertical_reference: str
    crs: str
    width: int
    height: int
    bounds: tuple[float, float, float, float]       # west, south, east, north (degrees)
    resolution_deg: tuple[float, float]
    nodata: float
    margin_km: float
    footprint_bbox: tuple[float, float, float, float]   # input radar footprint the mosaic must cover
    tiles: tuple[DEMTileRecord, ...]
    missing_sea_tiles: tuple[str, ...] = ()          # required cells for which the catalogue has no tile
    nodata_pixels_in_footprint: int
    footprint_pixels: int


class TerrainValidation(Contract):
    passed: bool
    checks: tuple[ValidationCheck, ...]
    width: int | None = None
    height: int | None = None
    geometry: str | None = None
    crs_wkt: str | None = None                       # from the DIMAP header
    image_to_model_transform: tuple[float, ...] | None = None   # DIMAP: scaleX, shearY, shearX, scaleY, x0, y0
    rasterio_crs: str | None = None
    rasterio_transform: tuple[float, ...] | None = None
    pixel_size_m: tuple[float, float] | None = None
    bounds_3857: tuple[float, float, float, float] | None = None   # west, south, east, north
    footprint: PolygonGeometry | None = None          # output extent in lon/lat
    aoi_valid_fraction: dict[str, float] = {}         # band -> share of AOI samples on valid pixels
    input_scene_aoi_coverage: float | None = None
    orbit_file: str | None = None
    dem_recorded: str | None = None                   # abstracted metadata "DEM" as written by SNAP
    processing_history: tuple[str, ...] = ()
    bands_before: tuple[BandStatistics, ...] = ()     # Step 3.3 input (radar geometry)
    bands_after: tuple[BandStatistics, ...] = ()      # terrain-corrected (map geometry)
    warnings: tuple[str, ...] = ()
    rasterio_version: str | None = None
    gdal_version: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TerrainValidation:
        if not self.checks:
            raise ValueError("a validation needs at least one check")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must equal all(check.passed)")
        return self

    def failed_checks(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]


class TerrainInputRecord(Contract):
    """The published Step 3.3 artifact that was terrain corrected (never modified)."""

    stage: Literal["speckle"] = "speckle"
    directory: RelativePath
    dim: RelativePath
    provenance_path: RelativePath
    provenance_sha256: Sha256Hex
    files_sha256: dict[str, Sha256Hex]
    radiometric_provenance_path: RelativePath
    radiometric_provenance_sha256: Sha256Hex      # matches what Step 3.3 recorded (chain of custody)
    product_id: str
    product_name: str
    acquisition_start: UTCDatetime
    orbit_direction: OrbitDirection
    relative_orbit: int
    speckle_config_fingerprint: str
    speckle_graph_sha256: Sha256Hex
    speckle_settings: dict[str, Any]
    width: int
    height: int


class TerrainProvenance(Contract):
    kind: Literal["s1_terrain"] = "s1_terrain"
    stage: Literal["terrain"] = "terrain"
    status: Literal["succeeded", "failed"]
    error: str | None = None
    data_source: DataSource = DataSource.SENTINEL_1
    data_role: Literal["pipeline_intermediate"] = "pipeline_intermediate"
    product_level: str = ("Lee Sigma filtered linear sigma0 VV/VH, Range-Doppler terrain corrected with Copernicus "
                          "DEM, EPSG:3857 10 m (no dB, no normalisation)")
    bato_version: str
    config_fingerprint: str
    settings: dict[str, Any]
    terrain_correction: dict[str, str]     # exact Terrain-Correction parameters as executed
    snap_value_sets: dict[str, Any]
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    pass_id: str
    aoi_polygon: PolygonGeometry           # from the Step 3.2 provenance (used for the coverage check)
    input: TerrainInputRecord
    dem: DEMMosaicRecord | None
    graph: GraphRecord
    snap: SnapRecord
    output: OutputRecord | None
    output_sha256: dict[str, Sha256Hex] = {}
    validation: TerrainValidation | None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> TerrainProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.status == "succeeded":
            if self.error is not None or self.output is None or self.validation is None or not self.output_sha256 \
                    or self.dem is None:
                raise ValueError("a successful run needs output, output hashes, DEM record and validation, no error")
            if not self.validation.passed:
                raise ValueError("a run whose validation failed cannot be 'succeeded'")
        elif not self.error:
            raise ValueError("a failed run needs an error message")
        return self

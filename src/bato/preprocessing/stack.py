"""Step 3.5: co-registered Sentinel-1 temporal stack (PRE-1, PRE-2, POST) from published Step 3.4 outputs.

The architecture's "pair stack" (docs/architecture.md: preprocessing -> "terrain-corrected stacks";
detection <- "pair stack"). For ONE temporal-set candidate of the fetch plan (2 pre-event + 1
post-event passes on one track), the three published ``terrain/`` outputs are:

* checked: provenance succeeded/validated, files unchanged since Step 3.4 (SHA-256 against its
  provenance), same AOI / track / settings / graph, EPSG:3857, 10 m, Sigma0_VV + Sigma0_VH linear;
* ordered and checked against the event: PRE-1 < PRE-2 < event date < POST (same-day excluded,
  the fetch manifest carries no event time);
* cut to ONE window - the AOI bounding box grown by ``aoi.analysis_buffer_km`` (the road-graph
  analysis extent), snapped outward to the shared 10 m standard grid. Because every Step 3.4
  output lies on the same standard grid, each source window is a whole-pixel offset: values are
  COPIED, never resampled. Any phase / pixel-size / CRS mismatch, or a window not fully inside a
  source, fails closed;
* written as one 6-band float32 GeoTIFF (layout v1: PRE1_VV, PRE1_VH, PRE2_VV, PRE2_VH, POST_VV,
  POST_VH; band descriptions + tags) and a uint8 validity mask (1 = valid in all six layers).

Nothing is normalised, clipped, converted to dB or reordered for a model; no DEM channel is added
(model channel order, normalisation and DEM features are OPEN specification items).

Output: ``<interim>/<output_subdir>/<aoi_id>/stack/<candidate_id>/`` with the same atomic
publishing as Steps 3.2-3.4 (``.tmp-``, ``.failed-``, ``.superseded-``).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.acquisition.dem import dem_search_bbox
from bato.config import BatoConfig
from bato.contracts.catalogue import CandidateRole
from bato.contracts.common import BBox, Polarization, PolygonGeometry
from bato.contracts.fetch import FetchManifest, PlannedCandidate
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.preprocessing import BandStatistics, TerrainProvenance, ValidationCheck
from bato.contracts.stack import (
    LAYOUT_VERSION,
    ROLES,
    StackGrid,
    StackInputRecord,
    StackLayer,
    StackOutputRecord,
    StackProvenance,
    StackValidation,
)
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap
from bato.preprocessing.radiometric import (
    PROVENANCE_FILE,
    PreprocessingError,
    _display,
    _relative,
    _sibling,
    _utc_now,
    stage_dir,
)
from bato.preprocessing.sigma0_validation import QUANTILES, SIGMA0_BAND, _open, require_rasterio
from bato.preprocessing.speckle import sha256_file
from bato.preprocessing.terrain import speckle_settings
from bato.preprocessing.terrain_validation import (
    _is_3857_wkt,
    aoi_samples,
    dimap_map_geocoding,
    lonlat_to_3857,
    standard_grid_offsets,
    to_lonlat,
)

STACK_GROUP = "stack"                      # <interim>/<subdir>/<aoi_id>/stack/<candidate_id>/
INPUT_STAGE = "terrain"
STACK_FILE = "stack_sigma0.tif"
MASK_FILE = "valid_mask.tif"
POLARIZATIONS = ("VV", "VH")
NODATA = 0.0                               # Step 3.4 (SNAP Terrain-Correction) no-data value
PIXEL_SPACING_M = 10.0
BLOCK_ROWS = 512
GRID_TOLERANCE_PX = 1e-6
COVERAGE_TOLERANCE = 0.01
QUANTILE_SAMPLE_TARGET = 4_000_000
COMPRESSION = "DEFLATE, tiled 512 x 512 (float predictor 3 for the stack, predictor 2 for the mask)"
EXCLUDED_DIFF_KEYS = {"externalDEMFile"}   # per-pass path inside each Step 3.4 output
OPEN_ITEMS = (
    "Model channel order and the mapping of PRE-1/PRE-2/POST to Kuro Siwo's sec1/sec2/flood inputs "
    "(docs/kuro_siwo_preprocessing.md: sec1/sec2 ordering OPEN).",
    "Normalisation / log transform / clipping applied by the Kuro Siwo dataloader (OPEN; the stack keeps "
    "linear sigma0 unchanged).",
    "DEM channel (dem.npy source) and slope for the model (OPEN; no DEM-derived layer is added).",
    "Tile size / tiling for inference (Kuro Siwo: 224 x 224; not done here).",
    "Units of the released Kuro Siwo arrays (linear vs dB) (OPEN).",
)


class StackError(PreprocessingError):
    """The temporal stack cannot be built from the available inputs (nothing written)."""


class StackRunFailed(StackError):
    """Writing or validating the stack failed; ``<candidate>.failed-<UTC>`` and its provenance are kept."""

    def __init__(self, message: str, output_dir: Path, provenance: StackProvenance) -> None:
        super().__init__(message)
        self.output_dir = output_dir
        self.provenance = provenance


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainInput:
    role: str
    pass_id: str
    directory: Path
    provenance_path: Path
    provenance: TerrainProvenance
    dim: Path
    product: DimapProduct
    wkt: str | None
    i2m: tuple[float, ...]
    files_sha256: dict[str, str]

    def band_file(self, pol: str) -> Path:
        band = self.product.band(SIGMA0_BAND.format(pol=pol))
        assert band is not None and band.image_file is not None   # checked in check_terrain_product
        return band.image_file


def terrain_settings(cfg: BatoConfig) -> dict[str, Any]:
    """The settings a Step 3.4 provenance records (same shape), from the current config."""
    s = cfg.settings
    return {**s.preprocessing.s1_terrain.model_dump(mode="json"), "dem_dataset": s.dem.cdse_dataset,
            "polarizations": list(s.sentinel1.polarizations)}


def check_terrain_product(product: DimapProduct, wkt: str | None, i2m: tuple[float, ...] | None,
                          polarizations: Sequence[str] = POLARIZATIONS) -> tuple[float, ...]:
    """A Step 3.4 product must be exactly Sigma0_VV + Sigma0_VH, linear float32, EPSG:3857 at 10 m
    with no shear, no-data 0. Returns the image-to-map transform.

    Raises:
        StackError: anything else.
    """
    problems = []
    expected = {SIGMA0_BAND.format(pol=p) for p in polarizations}
    names = {b.name for b in product.bands}
    if names != expected:
        problems.append(f"bands {sorted(names)} (expected exactly {sorted(expected)})")
    for b in product.bands:
        if "db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled:
            problems.append(f"{b.name} is in dB / log10 scale (unit {b.unit!r})")
        if b.data_type and b.data_type != "float32":
            problems.append(f"{b.name} data type {b.data_type} (expected float32)")
        if b.nodata_value not in (None, NODATA):
            problems.append(f"{b.name} no-data value {b.nodata_value} (expected {NODATA})")
    if _is_3857_wkt(wkt) is not True:
        problems.append("CRS is not EPSG:3857 (DIMAP WKT)")
    if i2m is None:
        problems.append("no map geocoding (IMAGE_TO_MODEL_TRANSFORM)")
    else:
        sx, shear_y, shear_x, sy = i2m[0], i2m[1], i2m[2], i2m[3]
        if not (math.isclose(sx, PIXEL_SPACING_M) and math.isclose(sy, -PIXEL_SPACING_M) and shear_x == 0
                and shear_y == 0):
            problems.append(f"pixel geometry {sx} x {sy}, shear ({shear_x}, {shear_y}); expected 10 x -10, no shear")
    if problems:
        raise StackError(f"{product.path.name}: " + "; ".join(problems))
    assert i2m is not None
    return i2m


def load_terrain_input(cfg: BatoConfig, manifest: FetchManifest, pass_id: str, role: str) -> TerrainInput:
    """Read and verify one published Step 3.4 output (read-only).

    Raises:
        StackError: missing, not validated, changed since Step 3.4, other settings, or not a valid product.
        ProhibitedInputError: an input lies in an evaluation-only folder.
    """
    root = cfg.config_dir.parent
    directory = stage_dir(cfg, manifest.aoi_id, pass_id, INPUT_STAGE)
    provenance_path = assert_pipeline_input_allowed(directory / PROVENANCE_FILE, cfg.settings.paths)
    if not directory.is_dir() or not provenance_path.is_file():
        raise StackError(
            f"{role} ({pass_id}): no published Step 3.4 output in {_display(directory, root)}; run Steps 3.2-3.4 "
            f"for this pass first (preprocess_s1.py --stage radiometric, speckle_s1.py, terrain_correct_s1.py "
            f"with --pass-id {pass_id})")
    try:
        prov = load_contract(TerrainProvenance, provenance_path)
    except ContractIOError as exc:
        raise StackError(str(exc)) from exc
    if prov.status != "succeeded" or prov.output is None or prov.validation is None or not prov.validation.passed \
            or not prov.output_sha256:
        raise StackError(f"{_display(provenance_path, root)} does not describe a validated Step 3.4 output")
    if (prov.aoi_id, prov.pass_id) != (manifest.aoi_id, pass_id):
        raise StackError(f"{_display(provenance_path, root)} belongs to {prov.aoi_id}/{prov.pass_id}")
    planned = {p.pass_id: p for p in manifest.plan.passes}[pass_id]
    if prov.input.product_id not in planned.product_ids:
        raise StackError(f"{role}: Step 3.4 product {prov.input.product_id} is not the manifest product of {pass_id}")
    if prov.settings != terrain_settings(cfg) or prov.input.speckle_settings != speckle_settings(cfg):
        raise StackError(f"{role} ({pass_id}) was produced with different terrain/speckle settings than the "
                         "current configuration")
    for rel, digest in sorted(prov.output_sha256.items()):
        path = assert_pipeline_input_allowed(root / rel, cfg.settings.paths)
        if not path.is_file() or sha256_file(path) != digest:
            raise StackError(f"{role} ({pass_id}): {rel} is missing or changed since Step 3.4 published it")
    dim = assert_pipeline_input_allowed(root / prov.output.dim, cfg.settings.paths)
    try:
        product = read_dimap(dim)
        wkt, i2m = dimap_map_geocoding(dim)
    except DimapError as exc:
        raise StackError(f"{role} ({pass_id}): {exc}") from exc
    i2m = check_terrain_product(product, wkt, i2m, tuple(cfg.settings.sentinel1.polarizations))
    return TerrainInput(role=role, pass_id=pass_id, directory=directory, provenance_path=provenance_path,
                        provenance=prov, dim=dim, product=product, wkt=wkt, i2m=i2m,
                        files_sha256=dict(prov.output_sha256))


# ---------------------------------------------------------------------------
# Temporal roles
# ---------------------------------------------------------------------------


def select_candidate(manifest: FetchManifest, candidate_id: str | None) -> PlannedCandidate:
    candidates = {c.candidate_id: c for c in manifest.plan.candidates}
    if candidate_id is None:
        if len(candidates) != 1:
            raise StackError(f"the manifest has {len(candidates)} candidates; choose one with --candidate "
                             f"({sorted(candidates)})")
        return next(iter(candidates.values()))
    if candidate_id not in candidates:
        raise StackError(f"candidate {candidate_id} is not in the manifest ({sorted(candidates)})")
    return candidates[candidate_id]


def check_temporal_roles(candidate: PlannedCandidate, manifest: FetchManifest,
                         starts: dict[str, datetime], tracks: dict[str, tuple[str, int]],
                         event_date: date, event_time: time | None = None) -> None:
    """PRE-1 < PRE-2 < event < POST, all on the candidate's track, matching the plan's pass roles.

    Raises:
        StackError: wrong shape, role, order, event association or track.
    """
    if len(candidate.pre_event_pass_ids) != 2:
        raise StackError(f"candidate {candidate.candidate_id} has {len(candidate.pre_event_pass_ids)} pre-event "
                         "pass(es); the stack needs exactly 2 pre-event + 1 post-event (Kuro Siwo shape)")
    pre1, pre2 = candidate.pre_event_pass_ids
    post = candidate.post_event_pass_id
    roles = {p.pass_id: p.role for p in manifest.plan.passes}
    problems = []
    for pid, expected in ((pre1, CandidateRole.PRE_EVENT), (pre2, CandidateRole.PRE_EVENT),
                          (post, CandidateRole.POST_EVENT)):
        if roles.get(pid) is not expected:
            problems.append(f"{pid} has plan role {roles.get(pid)}, expected {expected.value}")
    if not starts[pre1] < starts[pre2] < starts[post]:
        problems.append(f"acquisitions not in PRE-1 < PRE-2 < POST order: {starts[pre1]}, {starts[pre2]}, "
                        f"{starts[post]}")
    if event_time is not None:
        event_dt = datetime.combine(event_date, event_time, tzinfo=timezone.utc)
        if not (starts[pre2] < event_dt < starts[post]):
            problems.append("pre-event passes must start before and the post-event pass after the event time")
    else:   # event time unknown: same-day images are ambiguous and excluded (as in Step 2B)
        if starts[pre2].date() >= event_date:
            problems.append(f"PRE-2 is dated {starts[pre2].date()}, not before the event date {event_date}")
        if starts[post].date() <= event_date:
            problems.append(f"POST is dated {starts[post].date()}, not after the event date {event_date}")
    expected_track = (candidate.track.orbit_direction.value, candidate.track.relative_orbit)
    for pid in (pre1, pre2, post):
        if tracks[pid] != expected_track:
            problems.append(f"{pid} is on {tracks[pid]}, not on the candidate track {expected_track}")
    if problems:
        raise StackError(f"temporal set {candidate.candidate_id}: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Common grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceGrid:
    i2m: tuple[float, ...]       # scaleX, shearY, shearX, scaleY, x0, y0 (corner)
    width: int
    height: int


def plan_stack_grid(sources: Sequence[SourceGrid], aoi_bbox: BBox, buffer_km: float,
                    spacing: float = PIXEL_SPACING_M) -> tuple[StackGrid, list[tuple[int, int, int, int]]]:
    """The common window (AOI bbox + buffer, snapped outward to the shared grid) and, per source,
    the whole-pixel window (col_off, row_off, width, height) to copy.

    Raises:
        StackError: pixel size / phase mismatch, or the window is not fully inside a source.
    """
    ref = sources[0].i2m
    for i, s in enumerate(sources):
        if not (math.isclose(s.i2m[0], spacing) and math.isclose(s.i2m[3], -spacing) and s.i2m[1] == 0
                and s.i2m[2] == 0):
            raise StackError(f"input {i}: pixel geometry {s.i2m[:4]} differs from {spacing} m north-up")
        dx, dy = (s.i2m[4] - ref[4]) / spacing, (ref[5] - s.i2m[5]) / spacing
        if abs(dx - round(dx)) > GRID_TOLERANCE_PX or abs(dy - round(dy)) > GRID_TOLERANCE_PX:
            raise StackError(f"input {i} is not on the same pixel grid as input 0 (offset {dx}, {dy} pixels); "
                             "refusing to resample")
    analysis = dem_search_bbox(aoi_bbox, buffer_km)
    (xmin, xmax), (ymin, ymax) = lonlat_to_3857(np.array([analysis.min_lon, analysis.max_lon]),
                                                np.array([analysis.min_lat, analysis.max_lat]))
    x0 = ref[4] + math.floor((float(xmin) - ref[4]) / spacing) * spacing
    y0 = ref[5] - math.floor((ref[5] - float(ymax)) / spacing) * spacing
    width = math.ceil((float(xmax) - x0) / spacing - GRID_TOLERANCE_PX)
    height = math.ceil((y0 - float(ymin)) / spacing - GRID_TOLERANCE_PX)
    windows = []
    for i, s in enumerate(sources):
        col, row = round((x0 - s.i2m[4]) / spacing), round((s.i2m[5] - y0) / spacing)
        if col < 0 or row < 0 or col + width > s.width or row + height > s.height:
            raise StackError(f"input {i}: the analysis window ({width} x {height} px at col {col}, row {row}) is not "
                             f"fully inside its {s.width} x {s.height} raster; the stack would need data it lacks")
        windows.append((col, row, width, height))
    west, north, east, south = x0, y0, x0 + width * spacing, y0 - height * spacing
    corners = [to_lonlat(west, north), to_lonlat(east, north), to_lonlat(east, south), to_lonlat(west, south)]
    ring = tuple((round(lon, 6), round(lat, 6)) for lon, lat in corners)
    grid = StackGrid(pixel_size_m=spacing, width=width, height=height,
                     transform=(spacing, 0.0, x0, 0.0, -spacing, y0), bounds_3857=(west, south, east, north),
                     footprint=PolygonGeometry(coordinates=((*ring, ring[0]),)),
                     standard_grid_offsets=standard_grid_offsets(x0, y0, spacing, spacing, spacing),
                     aoi_bbox=aoi_bbox, analysis_buffer_km=buffer_km, analysis_bbox=analysis)
    return grid, windows


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackJob:
    root: Path
    aoi_id: str
    candidate: PlannedCandidate
    event_date: date
    inputs: tuple[TerrainInput, ...]          # PRE1, PRE2, POST
    aoi_polygon: PolygonGeometry
    grid: StackGrid
    windows: tuple[tuple[int, int, int, int], ...]
    final_dir: Path


def _bbox(polygon: PolygonGeometry) -> BBox:
    lons, lats = [p[0] for p in polygon.exterior], [p[1] for p in polygon.exterior]
    return BBox(min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats))


def prepare_stack(manifest: FetchManifest, cfg: BatoConfig, candidate_id: str | None = None) -> StackJob:
    """Select the candidate, verify the three Step 3.4 outputs and plan the common grid (nothing written).

    Raises:
        StackError: any inconsistency (fail closed).
        ProhibitedInputError: an input lies in an evaluation-only folder.
    """
    if manifest.dry_run:
        raise StackError("this is a dry-run plan; use the *_fetch_manifest.json of a real download")
    candidate = select_candidate(manifest, candidate_id)
    if len(candidate.pre_event_pass_ids) != 2:
        raise StackError(f"candidate {candidate.candidate_id} has {len(candidate.pre_event_pass_ids)} pre-event "
                         "pass(es); the stack needs exactly 2 pre-event + 1 post-event (Kuro Siwo shape)")
    pass_ids = (*candidate.pre_event_pass_ids, candidate.post_event_pass_id)
    inputs = tuple(load_terrain_input(cfg, manifest, pid, role) for pid, role in zip(pass_ids, ROLES, strict=True))
    starts = {i.pass_id: i.provenance.input.acquisition_start for i in inputs}
    tracks = {i.pass_id: (i.provenance.input.orbit_direction.value, i.provenance.input.relative_orbit) for i in inputs}
    check_temporal_roles(candidate, manifest, starts, tracks, manifest.plan.event_date)

    first = inputs[0].provenance
    for i in inputs[1:]:
        p = i.provenance
        if p.aoi_polygon != first.aoi_polygon:
            raise StackError(f"{i.role} was processed for a different AOI polygon than PRE1")
        if p.graph.template_sha256 != first.graph.template_sha256:
            raise StackError(f"{i.role} was terrain corrected with a different graph template than PRE1")
        a = {k: v for k, v in first.terrain_correction.items() if k not in EXCLUDED_DIFF_KEYS}
        b = {k: v for k, v in p.terrain_correction.items() if k not in EXCLUDED_DIFF_KEYS}
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise StackError(f"{i.role} Terrain-Correction parameters differ from PRE1: {diff}")
        if p.dem is None or first.dem is None or p.dem.dataset != first.dem.dataset:
            raise StackError(f"{i.role} used a different DEM dataset than PRE1")

    grid, windows = plan_stack_grid(
        [SourceGrid(i.i2m, i.product.width, i.product.height) for i in inputs], _bbox(first.aoi_polygon),
        cfg.settings.aoi.analysis_buffer_km)
    final_dir = stage_dir(cfg, manifest.aoi_id, STACK_GROUP, candidate.candidate_id)
    return StackJob(root=cfg.config_dir.parent, aoi_id=manifest.aoi_id, candidate=candidate,
                    event_date=manifest.plan.event_date, inputs=inputs, aoi_polygon=first.aoi_polygon, grid=grid,
                    windows=tuple(windows), final_dir=final_dir)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class _Stats:
    """Block-wise statistics of one layer (values unchanged; no thresholds)."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.stride = max(1, math.ceil(math.sqrt(width * height / QUANTILE_SAMPLE_TARGET)))
        self.finite = self.nodata = self.zeros = self.negative = 0
        self.total = 0.0
        self.vmin, self.vmax = math.inf, -math.inf
        self.samples: list[np.ndarray] = []

    def add(self, row: int, data: np.ndarray) -> np.ndarray:
        finite = np.isfinite(data)
        valid = finite & (data != NODATA)
        self.finite += int(finite.sum())
        self.nodata += int((finite & (data == NODATA)).sum())
        self.zeros += int((data == 0).sum())
        self.negative += int((valid & (data < 0)).sum())
        if valid.any():
            v = data[valid].astype(np.float64)
            self.total += float(v.sum())
            self.vmin, self.vmax = min(self.vmin, float(v.min())), max(self.vmax, float(v.max()))
        rows = np.nonzero((np.arange(row, row + data.shape[0]) % self.stride) == 0)[0]
        if rows.size:
            sub, ok = data[rows, ::self.stride], valid[rows, ::self.stride]
            self.samples.append(sub[ok].astype(np.float64))
        return valid

    def result(self, name: str, pol: str, file: str) -> BandStatistics:
        total = self.width * self.height
        valid = self.finite - self.nodata
        sample = np.concatenate(self.samples) if self.samples else np.empty(0)
        q = {f"p{round(p * 100):02d}": float(v)
             for p, v in zip(QUANTILES, np.quantile(sample, QUANTILES), strict=True)} if sample.size else {}
        return BandStatistics(
            band=name, polarization=Polarization(pol), file=file, dtype="float32", unit="intensity",
            nodata_value=NODATA, width=self.width, height=self.height, total_pixels=total,
            finite_pixels=self.finite, nonfinite_pixels=total - self.finite, nodata_pixels=self.nodata,
            zero_pixels=self.zeros, negative_pixels=self.negative, valid_pixels=valid,
            min=self.vmin if valid else None, max=self.vmax if valid else None,
            mean=self.total / valid if valid else None, median=q.get("p50"), quantiles=q,
            quantile_sample_stride=self.stride, quantile_sample_size=int(sample.size), edge_width_px=0,
            zero_pixels_edge=0, zero_pixels_interior=self.zeros)


def layer_names() -> list[tuple[str, str, str]]:
    """[(name, role, polarisation)] in layout v1 order."""
    return [(f"{role}_{pol}", role, pol) for role in ROLES for pol in POLARIZATIONS]


@dataclass(frozen=True)
class StackResult:
    output_dir: Path
    provenance_path: Path
    provenance: StackProvenance


def _hash_inputs(job: StackJob) -> dict[str, str]:
    return {rel: sha256_file(job.root / rel) for i in job.inputs for rel in sorted(i.files_sha256)}


def run_stack(job: StackJob, cfg: BatoConfig, *, overwrite: bool = False,
              now: Callable[[], datetime] = _utc_now, progress: Callable[[str], None] | None = None) -> StackResult:
    """Copy the three windows into the stack, validate it, then publish atomically.

    Raises:
        StackError: output exists without ``overwrite`` (nothing written).
        StackRunFailed: writing or validation failed (failed folder + provenance kept).
        ValidationSetupError: rasterio missing (nothing written).
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.windows import Window

    require_rasterio()
    say = progress or (lambda _text: None)
    root = job.root
    final_dir = assert_interim_output_dir(job.final_dir, cfg.settings.paths)
    if final_dir.exists() and not overwrite:
        raise StackError(f"output already exists: {final_dir} (use --overwrite to supersede it; the old output is "
                         "kept as <candidate>.superseded-<time>)")
    started = now()
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", started), cfg.settings.paths)
    tmp_dir.mkdir(parents=True)
    stack_path, mask_path = tmp_dir / STACK_FILE, tmp_dir / MASK_FILE
    grid = job.grid
    w, h = grid.width, grid.height
    a, b, c, d, e, f = grid.transform
    transform = Affine(a, b, c, d, e, f)
    layers = layer_names()
    by_role = {i.role: i for i in job.inputs}
    windows = dict(zip(ROLES, job.windows, strict=True))
    checks: list[ValidationCheck] = []
    warnings: list[str] = []
    error: str | None = None
    stats = [_Stats(w, h) for _ in layers]
    hashes = [hashlib.sha256() for _ in layers]
    mask = np.ones((h, w), dtype=np.uint8)
    validation: StackValidation | None = None
    stack_sha = mask_sha = None

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))

    try:
        sources = [_open(by_role[role].band_file(pol)) for _name, role, pol in layers]
        try:
            for name, src in zip((n for n, _r, _p in layers), sources, strict=True):
                if str(src.dtypes[0]) != "float32":
                    raise StackError(f"{name}: source raster dtype {src.dtypes[0]} (expected float32)")
            say(f"writing {w} x {h} x {len(layers)} stack ...\n")
            with rasterio.open(stack_path, "w", driver="GTiff", width=w, height=h, count=len(layers),
                               dtype="float32", crs=CRS.from_epsg(3857), transform=transform, nodata=NODATA,
                               tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=3,
                               BIGTIFF="IF_SAFER") as dst:
                for row in range(0, h, BLOCK_ROWS):
                    rows = min(BLOCK_ROWS, h - row)
                    for k, ((_name, role, _pol), src) in enumerate(zip(layers, sources, strict=True)):
                        col_off, row_off, _ww, _hh = windows[role]
                        data = np.asarray(src.read(1, window=Window(col_off, row_off + row, w, rows)),
                                          dtype=np.float32)
                        valid = stats[k].add(row, data)
                        mask[row:row + rows] &= valid.astype(np.uint8)
                        hashes[k].update(data.astype("<f4").tobytes())
                        dst.write(data, k + 1, window=Window(0, row, w, rows))
                for k, (name, role, pol) in enumerate(layers):
                    i = by_role[role]
                    dst.set_band_description(k + 1, name)
                    dst.update_tags(k + 1, role=role, polarization=pol, pass_id=i.pass_id,
                                    acquisition_start=i.provenance.input.acquisition_start.isoformat(),
                                    units="linear sigma0 (intensity), not dB")
                dst.update_tags(layout=LAYOUT_VERSION, aoi_id=job.aoi_id, candidate_id=job.candidate.candidate_id,
                                event_date=job.event_date.isoformat(), nodata=str(NODATA))
        finally:
            for src in sources:
                src.close()
        with rasterio.open(mask_path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                           crs=CRS.from_epsg(3857), transform=transform, tiled=True, blockxsize=512,
                           blockysize=512, compress="deflate", predictor=2) as dst:
            dst.write(mask, 1)
            dst.set_band_description(1, "valid_all_layers")

        # ---- validation of what was written -------------------------------------------------
        say("validating the written stack ...\n")
        rel_stack = _relative(final_dir / STACK_FILE, root)
        layer_stats = [s.result(name, pol, rel_stack) for s, (name, _r, pol) in zip(stats, layers, strict=True)]
        for st in layer_stats:
            check(f"finite_{st.band}", st.nonfinite_pixels == 0,
                  f"{st.nonfinite_pixels:,} NaN/inf values (must be 0; nothing is replaced)")
            check(f"non_negative_{st.band}", st.negative_pixels == 0,
                  f"{st.negative_pixels:,} negative values (linear intensity cannot be negative; dB data would be)")
            check(f"valid_pixels_{st.band}", st.valid_pixels > 0, f"{st.valid_pixels:,} of {st.total_pixels:,} valid")
        with rasterio.open(stack_path) as ds:
            epsg = ds.crs.to_epsg() if ds.crs is not None else None
            t = tuple(float(v) for v in tuple(ds.transform)[:6])
            check("written_grid", ds.count == len(layers) and (ds.width, ds.height) == (w, h) and epsg == 3857
                  and all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(t, grid.transform, strict=True))
                  and all(str(dt) == "float32" for dt in ds.dtypes),
                  f"{ds.count} bands {ds.width} x {ds.height} EPSG:{epsg} transform {t} dtypes {set(ds.dtypes)}")
            check("band_names", tuple(ds.descriptions) == tuple(n for n, _r, _p in layers),
                  f"descriptions {tuple(ds.descriptions)}")
            for k, (name, _r, _p) in enumerate(layers):
                readback = hashlib.sha256()
                for row in range(0, h, BLOCK_ROWS):
                    rows = min(BLOCK_ROWS, h - row)
                    block = np.asarray(ds.read(k + 1, window=Window(0, row, w, rows)), dtype=np.float32)
                    readback.update(block.astype("<f4").tobytes())
                check(f"copied_unchanged_{name}", readback.hexdigest() == hashes[k].hexdigest(),
                      "stack band equals the source window bit for bit" if readback.hexdigest()
                      == hashes[k].hexdigest() else "stack band differs from the source window")
        mask_valid = int(mask.sum())
        lons, lats = aoi_samples(job.aoi_polygon, cfg.settings.sentinel1.coverage_grid_size)
        xs, ys = lonlat_to_3857(lons, lats)
        cols = np.floor((xs - c) / a).astype(np.int64)
        rws = np.floor((f - ys) / -e).astype(np.int64)
        inside = (cols >= 0) & (cols < w) & (rws >= 0) & (rws < h)
        aoi_valid = round(float(mask[rws[inside], cols[inside]].sum()) / float(lons.size), 4) if lons.size else 0.0
        needed = min(min(i.provenance.validation.aoi_valid_fraction.values(), default=1.0)
                     for i in job.inputs if i.provenance.validation is not None)
        check("aoi_coverage", aoi_valid + COVERAGE_TOLERANCE >= needed,
              f"{aoi_valid:.1%} of AOI sample points valid in all six layers; inputs covered >= {needed:.1%}")
        after = _hash_inputs(job)
        expected = {rel: digest for i in job.inputs for rel, digest in i.files_sha256.items()}
        check("inputs_unchanged", after == expected, "Step 3.4 outputs identical before and after"
              if after == expected else "a Step 3.4 output changed during the run")
        if mask_valid < w * h:
            warnings.append(f"{w * h - mask_valid:,} of {w * h:,} stack pixels are not valid in all six layers "
                            "(no-data at swath edges); see valid_mask.tif")
        validation = StackValidation(passed=all(x.passed for x in checks), checks=tuple(checks),
                                     layers=tuple(layer_stats), mask_valid_pixels=mask_valid, mask_total_pixels=w * h,
                                     aoi_valid_fraction=aoi_valid, warnings=tuple(warnings))
        if not validation.passed:
            error = "validation failed: " + "; ".join(f"{x.name}: {x.detail}" for x in validation.failed_checks())
        stack_sha, mask_sha = sha256_file(stack_path), sha256_file(mask_path)
    except StackError as exc:
        error = str(exc)

    finished = now()
    if error is None:
        target = final_dir
        if final_dir.exists():
            os.replace(final_dir, _sibling(final_dir, "superseded", finished))
    else:
        target = _sibling(final_dir, "failed", finished)
    provenance = _provenance(job, cfg, error=error, started=started, finished=finished, target=target,
                             validation=validation, stack_sha=stack_sha, mask_sha=mask_sha, hashes=hashes,
                             warnings=warnings)
    save_contract(provenance, tmp_dir / PROVENANCE_FILE)
    os.replace(tmp_dir, target)
    provenance_path = target / PROVENANCE_FILE
    if error is not None:
        raise StackRunFailed(f"{error}\nfailed output kept (NOT valid) in {target}\nprovenance: {provenance_path}",
                             target, provenance)
    return StackResult(output_dir=target, provenance_path=provenance_path, provenance=provenance)


def _provenance(job: StackJob, cfg: BatoConfig, *, error: str | None, started: datetime, finished: datetime,
                target: Path, validation: StackValidation | None, stack_sha: str | None, mask_sha: str | None,
                hashes: list[Any], warnings: list[str]) -> StackProvenance:
    root = job.root
    event_dt = datetime.combine(job.event_date, time(0, 0), tzinfo=timezone.utc)
    windows = dict(zip(ROLES, job.windows, strict=True))
    inputs = tuple(StackInputRecord(
        role=i.role, pass_id=i.pass_id, product_id=i.provenance.input.product_id,  # type: ignore[arg-type]
        product_name=i.provenance.input.product_name, acquisition_start=i.provenance.input.acquisition_start,
        days_from_event=round((i.provenance.input.acquisition_start - event_dt).total_seconds() / 86400, 3),
        orbit_direction=i.provenance.input.orbit_direction, relative_orbit=i.provenance.input.relative_orbit,
        terrain_directory=_relative(i.directory, root), terrain_provenance=_relative(i.provenance_path, root),
        terrain_provenance_sha256=sha256_file(i.provenance_path), files_sha256=i.files_sha256,
        terrain_config_fingerprint=i.provenance.config_fingerprint,
        terrain_graph_template_sha256=i.provenance.graph.template_sha256, source_grid=i.i2m,
        source_size=(i.product.width, i.product.height), window=windows[i.role]) for i in job.inputs)
    by_role = {i.role: i for i in job.inputs}
    layers: tuple[StackLayer, ...] = ()
    output = None
    if error is None and stack_sha is not None and mask_sha is not None:
        layers = tuple(StackLayer(
            index=k + 1, name=name, role=role, polarization=Polarization(pol),  # type: ignore[arg-type]
            pass_id=by_role[role].pass_id, acquisition_start=by_role[role].provenance.input.acquisition_start,
            source_band_file=_relative(by_role[role].band_file(pol), root), copied_sha256=hashes[k].hexdigest())
            for k, (name, role, pol) in enumerate(layer_names()))
        output = StackOutputRecord(
            directory=_relative(target, root), stack=_relative(target / STACK_FILE, root), stack_sha256=stack_sha,
            mask=_relative(target / MASK_FILE, root), mask_sha256=mask_sha, nodata=NODATA, compression=COMPRESSION)
    validation_out = validation
    if validation is not None and error is not None:
        validation_out = validation.model_copy(update={"layers": tuple(
            st.model_copy(update={"file": _relative(target / STACK_FILE, root)}) for st in validation.layers)})
    return StackProvenance(
        status="succeeded" if error is None else "failed", error=error, bato_version=bato.__version__,
        config_fingerprint=cfg.fingerprint(),
        settings={"aoi.analysis_buffer_km": cfg.settings.aoi.analysis_buffer_km, "layout": LAYOUT_VERSION,
                  "pixel_spacing_m": PIXEL_SPACING_M, "nodata": NODATA, "terrain": terrain_settings(cfg),
                  "speckle": speckle_settings(cfg)},
        started_at=started, finished_at=finished, aoi_id=job.aoi_id, aoi_polygon=job.aoi_polygon,
        candidate_id=job.candidate.candidate_id, track=job.candidate.track.label,
        event_date=job.event_date.isoformat(), event_time_utc=None, inputs=inputs, grid=job.grid, layers=layers,
        output=output, validation=validation_out, open_specification_items=OPEN_ITEMS, warnings=tuple(warnings))


def describe_stack(prov: StackProvenance) -> list[str]:
    lines = []
    if prov.validation is not None:
        lines += [f"{'PASS' if x.passed else 'FAIL'}  {x.name}: {x.detail}" for x in prov.validation.checks]
        for st in prov.validation.layers:
            q = st.quantiles
            lines.append(f"{st.band:8} valid {st.valid_pixels:>12,} / {st.total_pixels:,} | min {st.min} | p01 "
                         f"{q.get('p01')} | median {st.median} | p99 {q.get('p99')} | max {st.max} | mean {st.mean}")
        if prov.validation.mask_valid_pixels is not None:
            lines.append(f"valid in all layers: {prov.validation.mask_valid_pixels:,} / "
                         f"{prov.validation.mask_total_pixels:,}; AOI {prov.validation.aoi_valid_fraction:.1%}")
        lines += [f"WARN  {x}" for x in prov.validation.warnings]
    return lines

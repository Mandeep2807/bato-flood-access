"""Step 3.2: one Sentinel-1 GRD pass -> linear sigma0 VV/VH in radar geometry (SNAP gpt).

Chain (graph ``graphs/s1_radiometric.xml``, values from ``preprocessing.s1_radiometric``):

    Read -> Apply-Orbit-File -> Subset (AOI bbox + crop_buffer_m) -> ThermalNoiseRemoval
         -> Remove-GRD-Border-Noise -> Calibration (sigma0, linear) -> Write (BEAM-DIMAP)

No speckle filter, no terrain correction, no dB, no normalisation (later steps).

Inputs come only from BATO's own records: the Step 2C fetch manifest (which
file, its download provenance), the Step 2A catalogue it was planned from
(AOI polygon, matched by SHA-256 digest) and the configuration. Every path
passes the data-rule guards; outputs go only under ``paths.interim``.

Output folder: ``<interim>/<output_subdir>/<aoi_id>/<pass_id>/radiometric/``

    <pass_id>_sigma0.dim / .data/   SNAP product (Sigma0_VV, Sigma0_VH)
    graph_resolved.xml              the exact graph executed (SHA-256 in provenance)
    gpt.log                         full gpt output
    preprocessing_provenance.json   PreprocessingProvenance

Atomicity: SNAP writes into ``radiometric.tmp-<UTC>``. Only after gpt
succeeded AND validation passed is that folder renamed to ``radiometric``.
A failed run is renamed ``radiometric.failed-<UTC>`` (with a failure
provenance) and never presented as a valid output. ``overwrite`` moves an
existing output to ``radiometric.superseded-<UTC>``; nothing is deleted.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bato
from bato.acquisition.dem import aoi_bbox_from_catalogue, dem_search_bbox
from bato.acquisition.s1_fetch import digest
from bato.config import BatoConfig
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import BBox, Polarization, PolygonGeometry
from bato.contracts.fetch import DownloadProvenance, FetchManifest, PlannedPass, PlannedScene
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.preprocessing import (
    AOIRecord,
    GraphRecord,
    InputProductRecord,
    OperatorRecord,
    OutputRecord,
    PreprocessingProvenance,
    RadiometricValidation,
    SnapRecord,
)
from bato.contracts.safe import SafeProduct
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.exceptions import BatoError
from bato.preprocessing import snap
from bato.preprocessing.safe_reader import check_against_provenance, inspect_safe_zip
from bato.preprocessing.sigma0_validation import SIGMA0_BAND, require_rasterio, validate_sigma0_output
from bato.utils.geo import bbox_ring

STAGE = "radiometric"
PROVENANCE_FILE = "preprocessing_provenance.json"
RESOLVED_GRAPH_FILE = "graph_resolved.xml"
LOG_FILE = "gpt.log"
CATALOGUE_SEARCH_DIRS = ("catalogue",)        # under paths.interim
_POLYGON = re.compile(r"POLYGON\(\(([^()]*)\)\)")
# Windows without long-path support: a file path must stay below MAX_PATH (260 characters incl. the
# terminating NUL) and a directory path below 248. The deepest items SNAP writes inside <stem>.data/:
WINDOWS_MAX_FILE_PATH = 259
WINDOWS_MAX_DIR_PATH = 247
_DEEPEST_SNAP_FILE = ("vector_data", "ground_control_points.csv")
_DEEPEST_SNAP_DIR = ("tie_point_grids",)


class PreprocessingError(BatoError):
    """A pass cannot be preprocessed (bad input or refused output). Nothing was written."""


class PreprocessingRunFailed(PreprocessingError):
    """SNAP ran (or was started) but the run or its validation failed. The folder
    ``radiometric.failed-<UTC>`` and its failure provenance are kept for diagnosis."""

    def __init__(self, message: str, output_dir: Path, provenance: PreprocessingProvenance) -> None:
        super().__init__(message)
        self.output_dir = output_dir
        self.provenance = provenance


# ---------------------------------------------------------------------------
# Job preparation (no SNAP, no writing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiometricJob:
    root: Path
    aoi_id: str
    pass_id: str
    planned_pass: PlannedPass
    scene: PlannedScene
    input_path: Path
    input_provenance_path: Path
    input_provenance: DownloadProvenance
    safe: SafeProduct
    catalogue_path: Path
    catalogue_digest: str
    aoi_polygon: PolygonGeometry
    aoi_bbox: BBox
    crop_bbox: BBox
    geo_region_wkt: str
    input_scene_aoi_coverage: float | None
    final_dir: Path

    @property
    def output_stem(self) -> str:
        return f"{self.pass_id}_sigma0"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PreprocessingError(f"{path} is outside the project root {root}") from exc


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def aoi_polygon_from_catalogue(catalogue: CatalogueSearchResult) -> PolygonGeometry:
    """The AOI polygon of a Step 2A search, read back from its recorded query filter."""
    match = _POLYGON.search(catalogue.query_filter)
    if match is None:
        raise PreprocessingError("catalogue query_filter contains no AOI polygon")
    ring = tuple(tuple(float(v) for v in pair.split()) for pair in match.group(1).split(","))
    try:
        return PolygonGeometry(coordinates=(tuple((p[0], p[1]) for p in ring),))
    except ValueError as exc:
        raise PreprocessingError(f"catalogue AOI polygon is invalid: {exc}") from exc


def crop_region(aoi_bbox: BBox, buffer_m: float) -> tuple[BBox, str]:
    """AOI bounding box grown by ``buffer_m`` and its WKT for Subset geoRegion."""
    crop = dem_search_bbox(aoi_bbox, buffer_m / 1000.0)
    ring = bbox_ring(crop.min_lon, crop.min_lat, crop.max_lon, crop.max_lat)
    return crop, "POLYGON ((" + ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in ring) + "))"


def stage_dir(cfg: BatoConfig, aoi_id: str, pass_id: str, stage: str = STAGE) -> Path:
    """``<interim>/<output_subdir>/<aoi_id>/<pass_id>/<stage>`` (guarded: interim only)."""
    for part in (aoi_id, pass_id, stage):
        if not part or "/" in part or "\\" in part or part in (".", "..") or ":" in part:
            raise PreprocessingError(f"invalid path component {part!r}")
    path = cfg.settings.paths.interim / cfg.settings.preprocessing.output_subdir / aoi_id / pass_id / stage
    return assert_interim_output_dir(path, cfg.settings.paths)


def find_catalogue(manifest: FetchManifest, cfg: BatoConfig,
                   explicit: Path | None = None) -> tuple[Path, CatalogueSearchResult]:
    """The Step 2A catalogue the manifest was planned from, identified by its SHA-256 digest.

    Searched (unless ``explicit``): ``<interim>/catalogue/*.json``, then ``*.json``
    in the project root. Files that are not catalogues are skipped.

    Raises:
        PreprocessingError: not found, or the explicit file has a different digest/AOI.
    """
    root = cfg.config_dir.parent
    wanted = manifest.plan.catalogue_digest
    if explicit is not None:
        path = assert_pipeline_input_allowed(explicit, cfg.settings.paths)
        try:
            catalogue = load_contract(CatalogueSearchResult, path)
        except ContractIOError as exc:
            raise PreprocessingError(str(exc)) from exc
        if digest(catalogue) != wanted:
            raise PreprocessingError(f"{path.name} is not the catalogue this manifest was planned from "
                                     "(SHA-256 digest differs)")
        candidates = [(path, catalogue)]
    else:
        folders = [cfg.settings.paths.interim / d for d in CATALOGUE_SEARCH_DIRS] + [root]
        candidates = []
        for folder in folders:
            for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
                path = assert_pipeline_input_allowed(path, cfg.settings.paths)
                try:
                    catalogue = load_contract(CatalogueSearchResult, path)
                except ContractIOError:
                    continue
                if digest(catalogue) == wanted:
                    candidates.append((path, catalogue))
                    break
            if candidates:
                break
        if not candidates:
            raise PreprocessingError(
                "the Step 2A catalogue used for this manifest was not found in "
                + ", ".join(_display(f, root) for f in folders)
                + "; pass it with --catalogue <file>_s1.json")
    path, catalogue = candidates[0]
    if catalogue.aoi_id != manifest.aoi_id:
        raise PreprocessingError(f"catalogue AOI {catalogue.aoi_id} differs from manifest AOI {manifest.aoi_id}")
    return path, catalogue


def prepare_job(manifest: FetchManifest, pass_id: str, cfg: BatoConfig,
                catalogue_path: Path | None = None) -> RadiometricJob:
    """Resolve everything needed to process one pass, without running SNAP.

    Raises:
        PreprocessingError: dry-run manifest, unknown/incomplete/multi-slice pass,
            missing files, provenance mismatch, missing catalogue.
        ProhibitedInputError: an input lies in an evaluation-only folder.
    """
    root = cfg.config_dir.parent
    if manifest.dry_run:
        raise PreprocessingError("this is a dry-run plan; use the *_fetch_manifest.json of a real download")
    passes = {p.pass_id: p for p in manifest.plan.passes}
    if pass_id not in passes:
        raise PreprocessingError(f"pass {pass_id} is not in the manifest; known passes: {sorted(passes)}")
    planned = passes[pass_id]
    if not manifest.pass_complete.get(pass_id):
        raise PreprocessingError(f"pass {pass_id} is incomplete in the manifest (not every slice is downloaded)")
    if planned.multi_slice:
        raise PreprocessingError(
            f"pass {pass_id} has {len(planned.product_ids)} slices; slice assembly is not implemented in "
            "Step 3.2 (single-slice passes only)")
    product_id = planned.product_ids[0]
    scene = next(s for s in manifest.plan.scenes if s.product_id == product_id)
    file = next((f for f in manifest.files if f.kind == "sentinel1_grd" and f.product_id == product_id
                 and f.present), None)
    if file is None or file.local_path is None or file.provenance_path is None:
        raise PreprocessingError(f"{scene.product_name}: no downloaded file recorded in the manifest")

    input_path = assert_pipeline_input_allowed(root / file.local_path, cfg.settings.paths)
    provenance_path = assert_pipeline_input_allowed(root / file.provenance_path, cfg.settings.paths)
    if not input_path.is_file():
        raise PreprocessingError(f"input product not found: {input_path} (it is never re-downloaded here)")
    try:
        provenance = load_contract(DownloadProvenance, provenance_path)
    except ContractIOError as exc:
        raise PreprocessingError(str(exc)) from exc
    if provenance.product_id != product_id or provenance.kind != "sentinel1_grd":
        raise PreprocessingError(f"{provenance_path.name} does not describe product {product_id}")
    size = input_path.stat().st_size
    if size != provenance.size_bytes:
        raise PreprocessingError(f"{input_path.name}: size {size} differs from the download provenance "
                                 f"({provenance.size_bytes}); the file changed after download")
    safe = inspect_safe_zip(input_path, cfg.settings.paths, tuple(cfg.settings.sentinel1.polarizations))
    check_against_provenance(safe, provenance)

    catalogue_file, catalogue = find_catalogue(manifest, cfg, catalogue_path)
    aoi_polygon = aoi_polygon_from_catalogue(catalogue)
    aoi_bbox = aoi_bbox_from_catalogue(catalogue)
    crop_bbox, wkt = crop_region(aoi_bbox, cfg.settings.preprocessing.crop_buffer_m)
    record = next((r for r in catalogue.records if r.product_id == product_id), None)
    return RadiometricJob(
        root=root, aoi_id=manifest.aoi_id, pass_id=pass_id, planned_pass=planned, scene=scene,
        input_path=input_path, input_provenance_path=provenance_path, input_provenance=provenance, safe=safe,
        catalogue_path=catalogue_file, catalogue_digest=digest(catalogue), aoi_polygon=aoi_polygon,
        aoi_bbox=aoi_bbox, crop_bbox=crop_bbox, geo_region_wkt=wkt,
        input_scene_aoi_coverage=record.scene_coverage if record is not None else None,
        final_dir=stage_dir(cfg, manifest.aoi_id, pass_id),
    )


def _windows_long_paths_enabled() -> bool:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])  # type: ignore[attr-defined,unused-ignore]
    except (ImportError, OSError):
        return False


def check_output_path_length(work_dir: Path, stem: str, *, system: str | None = None,
                             long_paths_enabled: bool | None = None) -> None:
    """Refuse, before SNAP runs, an output location whose deepest files would exceed the
    Windows path limit (they could not be written or read back for validation).

    Raises:
        PreprocessingError: on Windows without long-path support, the path would be too long.
    """
    if (system or platform.system()) != "Windows":
        return
    if long_paths_enabled if long_paths_enabled is not None else _windows_long_paths_enabled():
        return
    data_dir = Path(os.path.abspath(work_dir)) / f"{stem}.data"
    deepest_file = len(str(data_dir.joinpath(*_DEEPEST_SNAP_FILE)))
    deepest_dir = len(str(data_dir.joinpath(*_DEEPEST_SNAP_DIR)))
    if deepest_file > WINDOWS_MAX_FILE_PATH or deepest_dir > WINDOWS_MAX_DIR_PATH:
        raise PreprocessingError(
            f"output paths would be up to {deepest_file} characters long (Windows limit {WINDOWS_MAX_FILE_PATH + 1} "
            f"without long-path support), e.g. {data_dir.joinpath(*_DEEPEST_SNAP_FILE)}; move the repository to a "
            "shorter folder or enable Windows long paths (LongPathsEnabled). Nothing was written.")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _gpt_path(path: Path) -> str:
    """Absolute path with forward slashes (accepted by SNAP on every platform)."""
    return path.resolve().as_posix()


def graph_values(job: RadiometricJob, cfg: BatoConfig, output_file: Path) -> dict[str, object]:
    r = cfg.settings.preprocessing.s1_radiometric
    pols = ",".join(cfg.settings.sentinel1.polarizations)
    return {
        "input_file": _gpt_path(job.input_path),
        "orbit_type": r.orbit_type,
        "orbit_poly_degree": r.orbit_poly_degree,
        "orbit_continue_on_fail": r.orbit_continue_on_fail,
        "geo_region": job.geo_region_wkt,
        "polarisations": pols,
        "remove_thermal_noise": r.remove_thermal_noise,
        "reintroduce_thermal_noise": r.reintroduce_thermal_noise,
        "border_limit": r.border_limit,
        "trim_threshold": r.trim_threshold,
        "output_in_db": r.output_in_db,
        "output_sigma0": r.output_sigma0,
        "output_file": _gpt_path(output_file),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiometricResult:
    output_dir: Path
    provenance_path: Path
    provenance: PreprocessingProvenance


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _sibling(final_dir: Path, tag: str, moment: datetime) -> Path:
    candidate = final_dir.with_name(f"{final_dir.name}.{tag}-{_stamp(moment)}")
    n = 1
    while candidate.exists():
        n += 1
        candidate = final_dir.with_name(f"{final_dir.name}.{tag}-{_stamp(moment)}-{n}")
    return candidate


def run_radiometric(job: RadiometricJob, cfg: BatoConfig, *, overwrite: bool = False,
                    gpt: snap.ResolvedGpt | None = None, runner: snap.Runner = subprocess.run,
                    popen: Callable[..., Any] = subprocess.Popen, now: Callable[[], datetime] = _utc_now,
                    progress: Callable[[str], None] | None = None,
                    environ: dict[str, str] | None = None) -> RadiometricResult:
    """Preflight, run SNAP, validate, then publish the output atomically.

    Raises:
        PreprocessingError: output exists without ``overwrite``, or (Windows) the output
            path would be too long (nothing written).
        PreprocessingRunFailed: gpt or validation failed (failed folder + provenance kept).
        snap.SnapError: SNAP missing or preflight failed (nothing written).
        ValidationSetupError: rasterio missing (checked before SNAP runs; nothing written).
    """
    settings = cfg.settings
    root = job.root
    say = progress or (lambda _text: None)
    final_dir = assert_interim_output_dir(job.final_dir, settings.paths)
    if final_dir.exists() and not overwrite:
        raise PreprocessingError(f"output already exists: {final_dir} (use --overwrite to supersede it; "
                                 "the old output is kept as radiometric.superseded-<time>)")

    # ---- preflight (nothing written yet) -------------------------------------------------
    require_rasterio()
    resolved = gpt or snap.resolve_gpt(settings.snap, environ=environ)
    say(f"SNAP gpt: {resolved.path} (from {resolved.source})\n")
    diagnostics = snap.snap_diagnostics(resolved.path, settings.snap.diag_timeout_s, runner)
    say(f"SNAP version: {diagnostics.version or 'unknown (not found in gpt --diag output)'}\n")
    template_path = snap.graph_path(settings.preprocessing.s1_radiometric.graph)
    template_text = template_path.read_text(encoding="utf-8")
    started = now()
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", started), settings.paths)
    check_output_path_length(tmp_dir, job.output_stem)
    output_file = tmp_dir / f"{job.output_stem}.dim"
    graph_xml = snap.render_graph(template_text, graph_values(job, cfg, output_file))
    say("checking operator parameters against the installed SNAP (gpt <Operator> -h) ...\n")
    snap.check_operator_parameters(resolved.path, graph_xml, settings.snap.diag_timeout_s, runner)

    # ---- run -----------------------------------------------------------------------------
    tmp_dir.mkdir(parents=True)
    graph_file = tmp_dir / RESOLVED_GRAPH_FILE
    graph_file.write_text(graph_xml, encoding="utf-8")
    log_path = tmp_dir / LOG_FILE
    run: snap.GptRun | None = None
    validation: RadiometricValidation | None = None
    error: str | None = None
    say(f"running gpt (log: {_display(log_path, root)}) ...\n")
    try:
        run = snap.run_gpt(resolved.path, graph_file, log_path, settings.snap.timeout_s, popen=popen,
                           progress=progress)
        say(f"\ngpt finished in {run.runtime_s} s; validating output with rasterio ...\n")
        validation = validate_sigma0_output(
            output_file, polarizations=[Polarization(p) for p in settings.sentinel1.polarizations],
            safe=job.safe, aoi_polygon=job.aoi_polygon, input_scene_aoi_coverage=job.input_scene_aoi_coverage,
            expect_precise_orbit="precise" in settings.preprocessing.s1_radiometric.orbit_type.lower(),
            border_limit=settings.preprocessing.s1_radiometric.border_limit,
            coverage_grid_size=settings.sentinel1.coverage_grid_size, root=root)
        if not validation.passed:
            error = "validation failed: " + "; ".join(f"{c.name}: {c.detail}" for c in validation.failed_checks())
    except snap.SnapError as exc:
        error = str(exc)

    # ---- publish ---------------------------------------------------------------------------
    finished = now()
    if error is None:
        target = final_dir
        if final_dir.exists():                  # overwrite: keep the old output
            os.replace(final_dir, _sibling(final_dir, "superseded", finished))
    else:
        target = _sibling(final_dir, "failed", finished)
    provenance = _provenance(job, cfg, status="succeeded" if error is None else "failed", error=error,
                             started=started, finished=finished, resolved=resolved, diagnostics=diagnostics,
                             template_path=template_path, template_text=template_text, graph_xml=graph_xml,
                             run=run, validation=validation, tmp_dir=tmp_dir, target=target)
    save_contract(provenance, tmp_dir / PROVENANCE_FILE)
    os.replace(tmp_dir, target)
    provenance_path = target / PROVENANCE_FILE
    if error is not None:
        raise PreprocessingRunFailed(f"{error}\nfailed output kept (NOT valid) in {target}\n"
                                     f"provenance: {provenance_path}", target, provenance)
    return RadiometricResult(output_dir=target, provenance_path=provenance_path, provenance=provenance)


def _rebase(path: Path, tmp_dir: Path, target: Path) -> Path:
    """Where a file inside the temp folder lives after the folder is renamed."""
    return target / path.relative_to(tmp_dir)


def _provenance(job: RadiometricJob, cfg: BatoConfig, *, status: str, error: str | None, started: datetime,
                finished: datetime, resolved: snap.ResolvedGpt, diagnostics: snap.SnapDiagnostics,
                template_path: Path, template_text: str, graph_xml: str, run: snap.GptRun | None,
                validation: RadiometricValidation | None, tmp_dir: Path, target: Path) -> PreprocessingProvenance:
    root = job.root
    settings = cfg.settings
    prov = job.input_provenance
    band0 = job.safe.bands[0]

    def final(path: Path) -> str:
        return _relative(_rebase(path, tmp_dir, target), root)

    validation_out = validation
    if validation is not None:   # band file paths were recorded inside the temp folder
        validation_out = validation.model_copy(update={"bands": tuple(
            b.model_copy(update={"file": final(root / b.file)}) for b in validation.bands)})
    output = None
    if status == "succeeded":
        dim = tmp_dir / f"{job.output_stem}.dim"
        data = tmp_dir / f"{job.output_stem}.data"
        output = OutputRecord(
            directory=final(tmp_dir), dim=final(dim), data_dir=final(data),
            band_files={SIGMA0_BAND.format(pol=p): final(data / f"{SIGMA0_BAND.format(pol=p)}.img")
                        for p in settings.sentinel1.polarizations})
    log_path = tmp_dir / LOG_FILE
    return PreprocessingProvenance(
        status="succeeded" if status == "succeeded" else "failed", error=error,
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(),
        settings={"crop_buffer_m": settings.preprocessing.crop_buffer_m,
                  "s1_radiometric": settings.preprocessing.s1_radiometric.model_dump(mode="json"),
                  "polarizations": list(settings.sentinel1.polarizations)},
        started_at=started, finished_at=finished, aoi_id=job.aoi_id, pass_id=job.pass_id,
        input=InputProductRecord(
            product_id=job.scene.product_id, product_name=job.scene.product_name,
            local_path=_relative(job.input_path, root), size_bytes=prov.size_bytes, sha256=prov.sha256,
            sha256_source="Step 2C download provenance (computed at download; file size re-checked now)",
            download_provenance=_relative(job.input_provenance_path, root), platform=prov.platform,
            acquisition_start=job.scene.acquisition_start, orbit_direction=job.scene.orbit_direction,
            relative_orbit=job.scene.relative_orbit, polarizations=job.scene.polarizations,
            safe_samples=band0.samples, safe_lines=band0.lines,
            safe_range_spacing_m=band0.range_pixel_spacing_m, safe_azimuth_spacing_m=band0.azimuth_pixel_spacing_m),
        aoi=AOIRecord(
            aoi_id=job.aoi_id, source="Step 2A catalogue query polygon (matched to the manifest by SHA-256 digest)",
            catalogue_path=_display(job.catalogue_path, root), catalogue_digest=job.catalogue_digest,
            aoi_polygon=job.aoi_polygon, aoi_bbox=job.aoi_bbox, crop_buffer_m=settings.preprocessing.crop_buffer_m,
            crop_bbox=job.crop_bbox, geo_region_wkt=job.geo_region_wkt),
        graph=GraphRecord(
            template_path=_display(template_path, root), template_sha256=snap.sha256_text(template_text),
            resolved_path=final(tmp_dir / RESOLVED_GRAPH_FILE), resolved_sha256=snap.sha256_text(graph_xml),
            operators=tuple(OperatorRecord(node=n, operator=o, parameters=p)
                            for n, o, p in snap.graph_operators(graph_xml)),
            operator_parameters_checked=True),
        snap=SnapRecord(
            gpt_path=str(resolved.path), gpt_source=resolved.source, snap_version=diagnostics.version,
            diagnostics=diagnostics.output_head, command=run.args if run else (str(resolved.path),
                                                                              str(tmp_dir / RESOLVED_GRAPH_FILE), "-e"),
            return_code=run.returncode if run else None, runtime_s=run.runtime_s if run else None,
            log_path=final(log_path) if log_path.exists() else None),
        output=output, validation=validation_out,
    )


def describe_validation(validation: RadiometricValidation) -> list[str]:
    """Human-readable summary lines (used by the CLI)."""
    lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}" for c in validation.checks]
    for b in validation.bands:
        q = b.quantiles
        lines.append(
            f"{b.band}: {b.width} x {b.height} {b.dtype} ({b.unit}); valid {b.valid_pixels:,}/{b.total_pixels:,} | "
            f"nodata {b.nodata_pixels:,} | zero {b.zero_pixels:,} (edge {b.zero_pixels_edge:,}, interior "
            f"{b.zero_pixels_interior:,}) | non-finite {b.nonfinite_pixels:,} | negative {b.negative_pixels:,}")
        if b.valid_pixels:
            lines.append(
                f"    min {b.min:.6g}  p01 {q.get('p01', float('nan')):.6g}  p05 {q.get('p05', float('nan')):.6g}  "
                f"p25 {q.get('p25', float('nan')):.6g}  median {b.median if b.median is not None else float('nan'):.6g}"
                f"  p75 {q.get('p75', float('nan')):.6g}  p95 {q.get('p95', float('nan')):.6g}  "
                f"p99 {q.get('p99', float('nan')):.6g}  max {b.max:.6g}  mean {b.mean:.6g}  (linear sigma0)")
    lines.extend(f"WARN  {w}" for w in validation.warnings)
    return lines


"""Step 3.4: Range-Doppler terrain correction of the published Step 3.3 output (SNAP gpt).

    Read (speckle BEAM-DIMAP) -> Terrain-Correction (Copernicus DEM, EPSG:3857, 10 m) -> Write (BEAM-DIMAP)

This is the last step of the Kuro Siwo SNAP chain (Calibration -> Speckle-Filter ->
Terrain-Correction): it moves the Lee Sigma filtered linear sigma0 from radar geometry onto the
same map grid convention as the Kuro Siwo training data. Values stay LINEAR sigma0 (no dB, no
normalisation, no radiometric terrain flattening).

Input: the PUBLISHED Step 3.3 artifact ``<interim>/<output_subdir>/<aoi_id>/<pass_id>/speckle/``
(read, hashed before and after, never modified). Its provenance chain is checked back to Step 3.2
(the Step 3.2 provenance must still have the SHA-256 that Step 3.3 recorded).

DEM: BATO's own Step 2C Copernicus GLO-30 tiles, re-hashed against their download provenance and
mosaicked into one GeoTIFF (:mod:`bato.preprocessing.dem_mosaic`), given to SNAP as an external DEM.

Output: ``<interim>/<output_subdir>/<aoi_id>/<pass_id>/terrain/``

    <pass_id>_sigma0_spk_tc.dim / .data/   Sigma0_VV, Sigma0_VH (EPSG:3857, 10 m, linear)
    dem/<pass_id>_copdem30.tif             the external DEM exactly as given to SNAP
    graph_resolved.xml, gpt.log, preprocessing_provenance.json (TerrainProvenance)

Atomic publishing as in Steps 3.2/3.3 (``terrain.tmp-``, ``terrain.failed-``, ``terrain.superseded-``).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import bato
from bato.config import BatoConfig
from bato.contracts.common import Polarization, PolygonGeometry
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.preprocessing import (
    DEMMosaicRecord,
    GraphRecord,
    OperatorRecord,
    OutputRecord,
    PreprocessingProvenance,
    SnapRecord,
    SpeckleProvenance,
    TerrainInputRecord,
    TerrainProvenance,
    TerrainValidation,
)
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.preprocessing import snap
from bato.preprocessing.dem_mosaic import DEMError, DEMSource, build_mosaic, dem_extent, select_dem_tiles
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap
from bato.preprocessing.radiometric import (
    LOG_FILE,
    PROVENANCE_FILE,
    RESOLVED_GRAPH_FILE,
    PreprocessingError,
    _display,
    _relative,
    _sibling,
    _utc_now,
    check_output_path_length,
    stage_dir,
)
from bato.preprocessing.sigma0_validation import (
    FORBIDDEN_OPERATORS,
    SIGMA0_BAND,
    require_rasterio,
    tie_point_footprint,
)
from bato.preprocessing.speckle import artifact_files, hash_artifact, radiometric_settings, sha256_file
from bato.preprocessing.terrain_validation import SPECKLE_OPERATOR, TERRAIN_OPERATOR, validate_terrain_output

STAGE = "terrain"
INPUT_STAGE = "speckle"
DEM_SUBDIR = "dem"


class TerrainRunFailed(PreprocessingError):
    """DEM preparation, gpt or validation failed; ``terrain.failed-<UTC>`` and its provenance are kept."""

    def __init__(self, message: str, output_dir: Path, provenance: TerrainProvenance) -> None:
        super().__init__(message)
        self.output_dir = output_dir
        self.provenance = provenance


@dataclass(frozen=True)
class TerrainJob:
    root: Path
    aoi_id: str
    pass_id: str
    manifest: FetchManifest
    input_dir: Path
    input_dim: Path
    input_provenance_path: Path
    input_provenance: SpeckleProvenance
    radiometric_provenance_path: Path
    radiometric_provenance: PreprocessingProvenance
    input_product: DimapProduct
    input_footprint: PolygonGeometry
    input_hashes: dict[str, str]
    dem_sources: tuple[DEMSource, ...]
    dem_sea_tiles: tuple[str, ...]
    final_dir: Path

    @property
    def output_stem(self) -> str:
        return f"{self.pass_id}_sigma0_spk_tc"

    @property
    def dem_name(self) -> str:
        return f"{self.pass_id}_copdem30.tif"


def speckle_settings(cfg: BatoConfig) -> dict[str, Any]:
    """The settings a Step 3.3 provenance records (same shape), from the current config."""
    s = cfg.settings
    return {**s.preprocessing.s1_speckle.model_dump(mode="json"), "polarizations": list(s.sentinel1.polarizations)}


def check_input_product(product: DimapProduct, polarizations: tuple[str, ...]) -> None:
    """The input must be what Step 3.3 publishes: exactly linear sigma0 VV/VH, radar geometry,
    Lee Sigma applied once, not terrain corrected and not in dB.

    Raises:
        PreprocessingError: anything else (nothing is run).
    """
    expected = {SIGMA0_BAND.format(pol=p) for p in polarizations}
    names = {b.name for b in product.bands}
    problems = []
    if names != expected:
        problems.append(f"bands {sorted(names)} (expected exactly {sorted(expected)})")
    for b in product.bands:
        if "db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled:
            problems.append(f"{b.name} is in dB / log10 scale (unit {b.unit!r}); terrain correction needs "
                            "linear sigma0")
    if product.map_geocoding or not {"latitude", "longitude"} <= set(product.tie_point_grids) \
            or product.abstracted.get("is_terrain_corrected") in ("1", "true"):
        problems.append("not in radar geometry (already map projected / terrain corrected)")
    if product.history:
        others = [op for op in product.history if op in FORBIDDEN_OPERATORS and op != SPECKLE_OPERATOR]
        if others:
            problems.append(f"already processed by {others}")
        if product.history.count(SPECKLE_OPERATOR) != 1:
            problems.append(f"Speckle-Filter recorded {product.history.count(SPECKLE_OPERATOR)} times (expected once)")
    if problems:
        raise PreprocessingError(f"{product.path.name} is not a valid Step 3.3 input: " + "; ".join(problems))


def prepare_terrain_job(manifest: FetchManifest, pass_id: str, cfg: BatoConfig) -> TerrainJob:
    """Locate and check the published Step 3.3 output and the DEM tiles (no SNAP, nothing written).

    Raises:
        PreprocessingError: dry-run manifest, unknown pass, no published speckle output, broken
            provenance chain, settings mismatch, invalid input product.
        DEMError: DEM tiles missing or changed.
        ProhibitedInputError: an input lies in an evaluation-only folder.
        ValidationSetupError: rasterio missing (needed to read the input geolocation).
    """
    root = cfg.config_dir.parent
    if manifest.dry_run:
        raise PreprocessingError("this is a dry-run plan; use the *_fetch_manifest.json of a real download")
    passes = {p.pass_id: p for p in manifest.plan.passes}
    if pass_id not in passes:
        raise PreprocessingError(f"pass {pass_id} is not in the manifest; known passes: {sorted(passes)}")
    input_dir = stage_dir(cfg, manifest.aoi_id, pass_id, INPUT_STAGE)
    provenance_path = assert_pipeline_input_allowed(input_dir / PROVENANCE_FILE, cfg.settings.paths)
    if not input_dir.is_dir() or not provenance_path.is_file():
        raise PreprocessingError(f"no published Step 3.3 output in {_display(input_dir, root)}; run it first: "
                                 f"python scripts/speckle_s1.py --manifest <manifest> --pass-id {pass_id}")
    try:
        prov = load_contract(SpeckleProvenance, provenance_path)
    except ContractIOError as exc:
        raise PreprocessingError(str(exc)) from exc
    if prov.status != "succeeded" or prov.output is None or prov.validation is None or not prov.validation.passed:
        raise PreprocessingError(f"{_display(provenance_path, root)} does not describe a validated output")
    if (prov.aoi_id, prov.pass_id) != (manifest.aoi_id, pass_id) or prov.input.product_id not in \
            passes[pass_id].product_ids:
        raise PreprocessingError(f"{_display(provenance_path, root)} belongs to {prov.aoi_id}/{prov.pass_id} "
                                 f"({prov.input.product_id}), not to this manifest pass")
    if prov.settings != speckle_settings(cfg):
        raise PreprocessingError("the Step 3.3 output was produced with different speckle settings than the current "
                                 f"configuration (recorded {prov.settings}, current {speckle_settings(cfg)})")
    if prov.input.radiometric_settings != radiometric_settings(cfg):
        raise PreprocessingError("the Step 3.2 output under this Step 3.3 output used different radiometric settings "
                                 "than the current configuration")

    rad_path = assert_pipeline_input_allowed(root / prov.input.provenance_path, cfg.settings.paths)
    if not rad_path.is_file() or sha256_file(rad_path) != prov.input.provenance_sha256:
        raise PreprocessingError(f"{_display(rad_path, root)} is missing or changed since Step 3.3 recorded it "
                                 "(provenance chain broken)")
    try:
        rad_prov = load_contract(PreprocessingProvenance, rad_path)
    except ContractIOError as exc:
        raise PreprocessingError(str(exc)) from exc

    dim = assert_pipeline_input_allowed(root / prov.output.dim, cfg.settings.paths)
    if input_dir not in dim.parents:
        raise PreprocessingError(f"{prov.output.dim} is outside {_display(input_dir, root)}")
    try:
        product = read_dimap(dim)
    except DimapError as exc:
        raise PreprocessingError(f"published Step 3.3 output is incomplete: {exc}") from exc
    check_input_product(product, tuple(cfg.settings.sentinel1.polarizations))
    require_rasterio()
    try:
        footprint = tie_point_footprint(product)
    except DimapError as exc:
        raise PreprocessingError(f"cannot read the input geolocation: {exc}") from exc
    extent = dem_extent(footprint, cfg.settings.preprocessing.s1_terrain.dem_margin_km)
    sources, sea = select_dem_tiles(manifest, extent, cfg)
    return TerrainJob(root=root, aoi_id=manifest.aoi_id, pass_id=pass_id, manifest=manifest, input_dir=input_dir,
                      input_dim=dim, input_provenance_path=provenance_path, input_provenance=prov,
                      radiometric_provenance_path=rad_path, radiometric_provenance=rad_prov, input_product=product,
                      input_footprint=footprint, input_hashes=hash_artifact(dim, root), dem_sources=tuple(sources),
                      dem_sea_tiles=tuple(sea), final_dir=stage_dir(cfg, manifest.aoi_id, pass_id, STAGE))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def graph_values(job: TerrainJob, cfg: BatoConfig, dem_file: Path, output_file: Path) -> dict[str, object]:
    t = cfg.settings.preprocessing.s1_terrain
    return {
        "input_file": job.input_dim.resolve().as_posix(),
        "source_bands": ",".join(SIGMA0_BAND.format(pol=p) for p in cfg.settings.sentinel1.polarizations),
        "dem_file": dem_file.resolve().as_posix(),
        "dem_nodata": t.dem_nodata,
        "dem_apply_egm": t.dem_apply_egm,
        "output_file": output_file.resolve().as_posix(),
    }


def terrain_parameters(graph_xml: str) -> dict[str, str]:
    for _node, operator, params in snap.graph_operators(graph_xml):
        if operator == TERRAIN_OPERATOR:
            return dict(params)
    raise snap.SnapError("the graph has no Terrain-Correction node")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainResult:
    output_dir: Path
    provenance_path: Path
    provenance: TerrainProvenance


def run_terrain(job: TerrainJob, cfg: BatoConfig, *, overwrite: bool = False, gpt: snap.ResolvedGpt | None = None,
                runner: snap.Runner = subprocess.run, popen: Callable[..., Any] = subprocess.Popen,
                now: Callable[[], datetime] = _utc_now, progress: Callable[[str], None] | None = None,
                environ: dict[str, str] | None = None) -> TerrainResult:
    """Preflight, build the DEM, run SNAP, validate, then publish atomically.

    Raises:
        PreprocessingError: output exists without ``overwrite`` or path too long (nothing written).
        TerrainRunFailed: DEM preparation, gpt or validation failed, or the input changed (failed folder kept).
        snap.SnapError: SNAP missing, or an operator / parameter / value not accepted (nothing written).
        ValidationSetupError: rasterio missing (nothing written).
    """
    settings = cfg.settings
    root = job.root
    say = progress or (lambda _text: None)
    final_dir = assert_interim_output_dir(job.final_dir, settings.paths)
    if final_dir.exists() and not overwrite:
        raise PreprocessingError(f"output already exists: {final_dir} (use --overwrite to supersede it; "
                                 "the old output is kept as terrain.superseded-<time>)")

    # ---- preflight (nothing written yet) -------------------------------------------------
    require_rasterio()
    resolved = gpt or snap.resolve_gpt(settings.snap, environ=environ)
    say(f"SNAP gpt: {resolved.path} (from {resolved.source})\n")
    diagnostics = snap.snap_diagnostics(resolved.path, settings.snap.diag_timeout_s, runner)
    say(f"SNAP version: {diagnostics.version or 'unknown (not found in gpt --diag output)'}\n")
    template_path = snap.graph_path(settings.preprocessing.s1_terrain.graph)
    template_text = template_path.read_text(encoding="utf-8")
    started = now()
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", started), settings.paths)
    check_output_path_length(tmp_dir, job.output_stem)
    output_file = tmp_dir / f"{job.output_stem}.dim"
    dem_file = tmp_dir / DEM_SUBDIR / job.dem_name
    graph_xml = snap.render_graph(template_text, graph_values(job, cfg, dem_file, output_file))
    tc_params = terrain_parameters(graph_xml)
    say("checking operator parameters and values against the installed SNAP ...\n")
    snap.check_operator_parameters(resolved.path, graph_xml, settings.snap.diag_timeout_s, runner)
    value_sets = snap.check_parameter_values(resolved.path, graph_xml, [TERRAIN_OPERATOR],
                                             settings.snap.diag_timeout_s, runner)
    warnings = [f"DEM cells without a Copernicus tile (filled with no-data): {list(job.dem_sea_tiles)}"] \
        if job.dem_sea_tiles else []

    # ---- run -----------------------------------------------------------------------------
    tmp_dir.mkdir(parents=True)
    graph_file = tmp_dir / RESOLVED_GRAPH_FILE
    graph_file.write_text(graph_xml, encoding="utf-8")
    log_path = tmp_dir / LOG_FILE
    run: snap.GptRun | None = None
    validation: TerrainValidation | None = None
    dem: DEMMosaicRecord | None = None
    error: str | None = None
    try:
        say(f"building the external DEM from {len(job.dem_sources)} Step 2C Copernicus DEM tile(s) ...\n")
        dem_file.parent.mkdir(parents=True)
        dem = build_mosaic(list(job.dem_sources), dem_extent(job.input_footprint,
                                                             settings.preprocessing.s1_terrain.dem_margin_km),
                           job.input_footprint, dem_file, nodata=settings.preprocessing.s1_terrain.dem_nodata,
                           margin_km=settings.preprocessing.s1_terrain.dem_margin_km,
                           dataset=settings.dem.cdse_dataset, sea_tiles=list(job.dem_sea_tiles), root=root)
        if dem.nodata_pixels_in_footprint:
            warnings.append(f"DEM has {dem.nodata_pixels_in_footprint:,} no-data pixels inside the input footprint "
                            f"({dem.footprint_pixels:,} pixels)")
        say(f"running gpt (log: {_display(log_path, root)}) ...\n")
        run = snap.run_gpt(resolved.path, graph_file, log_path, settings.snap.timeout_s, popen=popen,
                           progress=progress)
        say(f"\ngpt finished in {run.runtime_s} s; validating output ...\n")
        validation = validate_terrain_output(
            output_file, input_product=job.input_product,
            polarizations=[Polarization(p) for p in settings.sentinel1.polarizations],
            pixel_spacing_m=float(tc_params["pixelSpacingInMeter"]),
            aoi_polygon=job.radiometric_provenance.aoi.aoi_polygon,
            input_scene_aoi_coverage=(job.radiometric_provenance.validation.input_scene_aoi_coverage
                                      if job.radiometric_provenance.validation else None),
            coverage_grid_size=settings.sentinel1.coverage_grid_size, root=root,
            grid_origin=(float(tc_params["standardGridOriginX"]), float(tc_params["standardGridOriginY"])))
        if not validation.passed:
            error = "validation failed: " + "; ".join(f"{c.name}: {c.detail}" for c in validation.failed_checks())
    except (snap.SnapError, DEMError) as exc:
        error = str(exc)
    if dem is not None and (not dem_file.is_file() or sha256_file(dem_file) != dem.sha256):
        error = (error + "; " if error else "") + "the external DEM changed during the run"
    after = hash_artifact(job.input_dim, root) if job.input_dim.is_file() else {}
    if after != job.input_hashes:
        changed = sorted(k for k in set(after) | set(job.input_hashes) if after.get(k) != job.input_hashes.get(k))
        error = (error + "; " if error else "") + f"the Step 3.3 input changed during the run: {changed}"

    # ---- publish ---------------------------------------------------------------------------
    finished = now()
    if error is None:
        target = final_dir
        if final_dir.exists():
            os.replace(final_dir, _sibling(final_dir, "superseded", finished))
    else:
        target = _sibling(final_dir, "failed", finished)
    output_hashes = {}
    if error is None:
        for p in [*artifact_files(output_file), dem_file]:
            output_hashes[_relative(target / p.relative_to(tmp_dir), root)] = sha256_file(p)
    provenance = _provenance(job, cfg, error=error, started=started, finished=finished, resolved=resolved,
                             diagnostics=diagnostics, template_path=template_path, template_text=template_text,
                             graph_xml=graph_xml, tc_params=tc_params, value_sets=value_sets, run=run,
                             validation=validation, dem=dem, tmp_dir=tmp_dir, target=target,
                             output_hashes=output_hashes, warnings=warnings)
    save_contract(provenance, tmp_dir / PROVENANCE_FILE)
    os.replace(tmp_dir, target)
    provenance_path = target / PROVENANCE_FILE
    if error is not None:
        raise TerrainRunFailed(f"{error}\nfailed output kept (NOT valid) in {target}\nprovenance: {provenance_path}",
                               target, provenance)
    return TerrainResult(output_dir=target, provenance_path=provenance_path, provenance=provenance)


def _provenance(job: TerrainJob, cfg: BatoConfig, *, error: str | None, started: datetime, finished: datetime,
                resolved: snap.ResolvedGpt, diagnostics: snap.SnapDiagnostics, template_path: Path,
                template_text: str, graph_xml: str, tc_params: dict[str, str], value_sets: dict[str, Any],
                run: snap.GptRun | None, validation: TerrainValidation | None, dem: DEMMosaicRecord | None,
                tmp_dir: Path, target: Path, output_hashes: dict[str, str], warnings: list[str]) -> TerrainProvenance:
    root = job.root
    settings = cfg.settings
    sp = job.input_provenance

    def final(path: Path) -> str:
        return _relative(target / path.relative_to(tmp_dir), root)

    def rebase(rel: str) -> str:
        path = (root / rel).resolve()
        return final(path) if tmp_dir.resolve() in path.parents else rel

    validation_out = validation
    if validation is not None:
        validation_out = validation.model_copy(update={"bands_after": tuple(
            b.model_copy(update={"file": rebase(b.file)}) for b in validation.bands_after)})
    dem_out = dem.model_copy(update={"path": rebase(dem.path)}) if dem is not None else None
    output = None
    if error is None:
        data = tmp_dir / f"{job.output_stem}.data"
        output = OutputRecord(
            directory=final(tmp_dir), dim=final(tmp_dir / f"{job.output_stem}.dim"), data_dir=final(data),
            band_files={SIGMA0_BAND.format(pol=p): final(data / f"{SIGMA0_BAND.format(pol=p)}.img")
                        for p in settings.sentinel1.polarizations})
    log_path = tmp_dir / LOG_FILE
    assert sp.output is not None   # checked in prepare_terrain_job
    return TerrainProvenance(
        status="succeeded" if error is None else "failed", error=error,
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(),
        settings={**settings.preprocessing.s1_terrain.model_dump(mode="json"),
                  "dem_dataset": settings.dem.cdse_dataset, "polarizations": list(settings.sentinel1.polarizations)},
        terrain_correction=tc_params, snap_value_sets=value_sets, started_at=started, finished_at=finished,
        aoi_id=job.aoi_id, pass_id=job.pass_id, aoi_polygon=job.radiometric_provenance.aoi.aoi_polygon,
        input=TerrainInputRecord(
            directory=_relative(job.input_dir, root), dim=_relative(job.input_dim, root),
            provenance_path=_relative(job.input_provenance_path, root),
            provenance_sha256=sha256_file(job.input_provenance_path), files_sha256=job.input_hashes,
            radiometric_provenance_path=_relative(job.radiometric_provenance_path, root),
            radiometric_provenance_sha256=sp.input.provenance_sha256,
            product_id=sp.input.product_id, product_name=sp.input.product_name,
            acquisition_start=sp.input.acquisition_start, orbit_direction=sp.input.orbit_direction,
            relative_orbit=sp.input.relative_orbit, speckle_config_fingerprint=sp.config_fingerprint,
            speckle_graph_sha256=sp.graph.resolved_sha256, speckle_settings=sp.settings,
            width=job.input_product.width, height=job.input_product.height),
        dem=dem_out,
        graph=GraphRecord(
            template_path=_display(template_path, root), template_sha256=snap.sha256_text(template_text),
            resolved_path=final(tmp_dir / RESOLVED_GRAPH_FILE), resolved_sha256=snap.sha256_text(graph_xml),
            operators=tuple(OperatorRecord(node=n, operator=o, parameters=p)
                            for n, o, p in snap.graph_operators(graph_xml)),
            operator_parameters_checked=True),
        snap=SnapRecord(
            gpt_path=str(resolved.path), gpt_source=resolved.source, snap_version=diagnostics.version,
            diagnostics=diagnostics.output_head,
            command=run.args if run else (str(resolved.path), str(tmp_dir / RESOLVED_GRAPH_FILE), "-e"),
            return_code=run.returncode if run else None, runtime_s=run.runtime_s if run else None,
            log_path=final(log_path) if log_path.exists() else None),
        output=output, output_sha256=output_hashes, validation=validation_out, warnings=tuple(warnings),
    )


def describe_terrain_validation(validation: TerrainValidation) -> list[str]:
    """Human-readable summary (used by the CLI)."""
    lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}" for c in validation.checks]

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.6g}"

    after = {b.band: b for b in validation.bands_after}
    for before in validation.bands_before:
        a = after.get(before.band)
        if a is None:
            continue
        lines.append(f"{before.band} (linear sigma0)   radar geometry (3.3)    EPSG:3857 (3.4)")
        lines.append(f"    {'size':18} {before.width:>10} x {before.height:<10} {a.width:>8} x {a.height}")
        for label, x, y in (("valid pixels", before.valid_pixels, a.valid_pixels),
                            ("no-data", before.nodata_pixels, a.nodata_pixels),
                            ("non-finite", before.nonfinite_pixels, a.nonfinite_pixels)):
            lines.append(f"    {label:18} {x:>23,} {y:>18,}")
        for label, x2, y2 in (("min", before.min, a.min), ("p01", before.quantiles.get("p01"), a.quantiles.get("p01")),
                              ("p05", before.quantiles.get("p05"), a.quantiles.get("p05")),
                              ("p25", before.quantiles.get("p25"), a.quantiles.get("p25")),
                              ("median", before.median, a.median),
                              ("p75", before.quantiles.get("p75"), a.quantiles.get("p75")),
                              ("p95", before.quantiles.get("p95"), a.quantiles.get("p95")),
                              ("p99", before.quantiles.get("p99"), a.quantiles.get("p99")),
                              ("max", before.max, a.max), ("mean", before.mean, a.mean)):
            lines.append(f"    {label:18} {fmt(x2):>23} {fmt(y2):>18}")
    for band, fraction in validation.aoi_valid_fraction.items():
        lines.append(f"AOI on valid pixels ({band}): {fraction:.1%}")
    lines.extend(f"WARN  {w}" for w in validation.warnings)
    return lines

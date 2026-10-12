"""Step 2C: fetch Sentinel-1 GRD products (and Copernicus DEM tiles) for
selected temporal-set candidates.

    plan = build_s1_fetch_plan(selection, catalogue, cfg, tracks=..., ranks=...)
    manifest = execute_fetch(plan, dem_plan, cfg, downloader=..., dry_run=...)

* The plan uses ONLY the validated Step 2B selection and the Step 2A catalogue
  it was made from (checked by digest). CDSE product ids come from the
  catalogue; nothing is searched again and no scene id is hard-coded.
* Every slice of every selected pass is planned; slices stay separate files
  (one pass = one temporal observation; mosaicking is preprocessing's job).
  A pass is "complete" only when all its slices are present.
* Products must be IW GRDH level-1 with all configured polarisations (VV+VH):
  the Kuro Siwo-compatible input. Files are stored exactly as delivered
  (SAFE zip with calibration/noise annotations): no sigma0, no dB, no filtering.
* Files go under ``paths.raw`` only (``assert_raw_output_dir``), each with a
  ``.provenance.json`` sidecar; a run manifest goes to ``paths.raw/manifests``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from bato import __version__
from bato.acquisition.cdse_odata import ProductMetadata, fetch_product_metadata
from bato.acquisition.downloader import Downloader, DownloadResult, RemoteFile
from bato.acquisition.http import Transport
from bato.acquisition.safe_inspect import ContentCheckError, check_dem_archive, check_s1_grd_zip
from bato.config import BatoConfig
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import DataSource
from bato.contracts.fetch import (
    DEMPlan,
    DEMTile,
    DownloadProvenance,
    FetchManifest,
    FetchStatus,
    FileFetchResult,
    PlannedCandidate,
    PlannedPass,
    PlannedScene,
    S1FetchPlan,
)
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.temporal import TemporalSelectionResult
from bato.data_rules import assert_raw_output_dir
from bato.exceptions import BatoError, ProviderError
from bato.logging_utils import get_logger, stage

SOURCE = "cdse-odata-download"
_LEVEL1_STANDARD = "1S"


class FetchPlanError(BatoError):
    """The selection/catalogue cannot produce a valid fetch plan."""


def digest(model: TemporalSelectionResult | CatalogueSearchResult) -> str:
    return hashlib.sha256(model.model_dump_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def build_s1_fetch_plan(selection: TemporalSelectionResult, catalogue: CatalogueSearchResult, cfg: BatoConfig,
                        tracks: Iterable[str] | None = None, ranks: Iterable[int] = (1,)) -> S1FetchPlan:
    """Plan the products needed by the chosen candidates (default: rank 1 on every track)."""
    if digest(catalogue) != selection.catalogue_digest:
        raise FetchPlanError("the selection was not made from this catalogue (digest mismatch); "
                             "re-run select_temporal_sets.py on this catalogue")
    s1 = cfg.settings.sentinel1
    wanted_ranks = set(ranks)
    available = [t.track.label for t in selection.tracks]
    wanted_tracks = set(tracks) if tracks else set(available)
    unknown = wanted_tracks - set(available)
    if unknown:
        raise FetchPlanError(f"unknown track(s) {sorted(unknown)}; available: {available}")

    chosen = [c for t in selection.tracks if t.track.label in wanted_tracks
              for c in t.candidates if c.rank in wanted_ranks]
    if not chosen:
        raise FetchPlanError(f"no candidate with rank {sorted(wanted_ranks)} on tracks {sorted(wanted_tracks)}")

    passes = {p.pass_id: p for p in selection.passes}
    accepted = {r.name: r for r in catalogue.accepted()}
    required_type = f"{s1.acquisition_mode}_{s1.product_type}{s1.grd_resolution_class}_{_LEVEL1_STANDARD}"

    pass_order: list[str] = []
    needed_by: dict[str, set[str]] = {}
    for c in chosen:
        for pid in (*c.pre_event_pass_ids, c.post_event_pass_id):
            if pid not in pass_order:
                pass_order.append(pid)
            needed_by.setdefault(pid, set()).add(c.candidate_id)

    planned_passes, scenes = [], []
    for pid in pass_order:
        p = passes[pid]
        product_ids = []
        for index, name in enumerate(p.scene_names):
            record = accepted.get(name)
            if record is None or record.acquisition is None or not record.product_id:
                raise FetchPlanError(f"pass {pid}: slice {name} is not an accepted catalogue scene with a CDSE id")
            acq = record.acquisition
            if acq.product_type != required_type:
                raise FetchPlanError(f"{name}: product type {acq.product_type}, required {required_type}")
            missing = [pol for pol in s1.polarizations if pol not in {x.value for x in acq.polarizations}]
            if missing:
                raise FetchPlanError(f"{name}: missing required polarisation(s) {missing}")
            if acq.orbit_direction is None or acq.relative_orbit is None or \
                    (acq.orbit_direction, acq.relative_orbit) != (p.track.orbit_direction, p.track.relative_orbit):
                raise FetchPlanError(f"{name}: not on the track of pass {pid}")
            product_ids.append(record.product_id)
            scenes.append(PlannedScene(
                product_id=record.product_id, product_name=name, pass_id=pid, slice_index=index,
                slice_count=len(p.scene_names), platform=acq.platform or p.platform,
                acquisition_start=acq.acquisition_start, orbit_direction=acq.orbit_direction,
                relative_orbit=acq.relative_orbit, product_type=acq.product_type,
                polarizations=acq.polarizations, candidate_ids=tuple(sorted(needed_by[pid])),
            ))
        planned_passes.append(PlannedPass(pass_id=pid, track=p.track, role=p.role, start=p.start,
                                          product_ids=tuple(product_ids)))

    candidates = tuple(PlannedCandidate(candidate_id=c.candidate_id, track=c.track, rank=c.rank,
                                        pre_event_pass_ids=c.pre_event_pass_ids,
                                        post_event_pass_id=c.post_event_pass_id) for c in chosen)
    return S1FetchPlan(aoi_id=selection.aoi_id, event_date=selection.event_date, selection_digest=digest(selection),
                       catalogue_digest=selection.catalogue_digest, candidates=candidates,
                       passes=tuple(planned_passes), scenes=tuple(scenes))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise FetchPlanError(f"{path} is outside the project root {root}; provenance paths must be relative") from exc


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".provenance.json")


def _quarantine(path: Path) -> None:
    if path.exists():
        os.replace(path, path.with_name(path.name + ".rejected"))


class _Fetcher:
    def __init__(self, cfg: BatoConfig, downloader: Downloader, metadata_transport: Transport | None,
                 clock: Callable[[], datetime], log: logging.Logger) -> None:
        self.cfg = cfg
        self.downloader = downloader
        self.metadata_transport = metadata_transport
        self.clock = clock
        self.log = log
        self.root = cfg.config_dir.parent
        raw = cfg.settings.paths.raw
        dl = cfg.settings.download
        self.s1_dir = assert_raw_output_dir(raw / dl.s1_subdir, cfg.settings.paths)
        self.dem_dir = assert_raw_output_dir(raw / dl.dem_subdir, cfg.settings.paths)

    def _metadata(self, product_id: str, expected_name: str) -> ProductMetadata:
        meta = fetch_product_metadata(product_id, self.cfg.settings.cdse, self.metadata_transport)
        if meta.name.removesuffix(".SAFE") != expected_name.removesuffix(".SAFE"):
            raise FetchPlanError(f"CDSE product {product_id} is now named {meta.name!r}, "
                                 f"not {expected_name!r}; re-run the catalogue search")
        if meta.online is False:
            raise ProviderError(f"{expected_name} is offline (CDSE long-term archive); it must be ordered "
                                "before it can be downloaded - ordering is not supported by BATO yet")
        return meta

    def _download(self, product_id: str, name: str, dest: Path,
                  checker: Callable[[Path], str]) -> tuple[ProductMetadata, DownloadResult, str]:
        meta = self._metadata(product_id, name)
        url = f"{self.cfg.settings.download.base_url}({product_id})/$value"
        result = self.downloader.fetch(RemoteFile(url=url, label=name, expected_size=meta.content_length,
                                                  checksums=meta.checksums), dest)
        try:
            check = checker(dest)
        except ContentCheckError:
            _quarantine(dest)
            raise
        return meta, result, check

    def _provenance(self, dest: Path, meta: ProductMetadata, result: DownloadResult, check: str,
                    **fields: object) -> Path:
        sidecar = _sidecar(dest)
        if result.skipped_existing and sidecar.exists():
            try:
                previous = load_contract(DownloadProvenance, sidecar)
            except ContractIOError:
                previous = None
            if previous is not None and previous.sha256 == result.sha256 and previous.product_id == meta.product_id:
                return sidecar   # unchanged file, keep the original provenance
        record = DownloadProvenance(
            product_id=meta.product_id, product_name=meta.name, source=SOURCE,
            source_url=f"{self.cfg.settings.download.base_url}({meta.product_id})/$value",
            content_length=meta.content_length, server_checksums=meta.checksums,
            checksum_verified_with=result.verified_with, checksum_verified=result.verified_with is not None,
            sha256=result.sha256, size_bytes=result.size, local_path=_relative(dest, self.root),
            content_check=check, downloaded_at=self.clock(), attempts=result.attempts, resumed=result.resumed,
            bato_version=__version__, config_fingerprint=self.cfg.fingerprint(), **fields,  # type: ignore[arg-type]  # kind-specific fields
        )
        save_contract(record, sidecar)
        return sidecar

    def s1(self, scene: PlannedScene) -> FileFetchResult:
        dest = self.s1_dir / f"{scene.product_name}.zip"
        pols = tuple(self.cfg.settings.sentinel1.polarizations)
        meta, result, check = self._download(scene.product_id, scene.product_name, dest,
                                             lambda path: check_s1_grd_zip(path, pols))
        sidecar = self._provenance(
            dest, meta, result, check, kind="sentinel1_grd", data_source=DataSource.SENTINEL_1,
            platform=scene.platform, acquisition_start=scene.acquisition_start,
            orbit_direction=scene.orbit_direction, relative_orbit=scene.relative_orbit,
            polarizations=scene.polarizations, product_type=scene.product_type, pass_ids=(scene.pass_id,),
            candidate_ids=scene.candidate_ids)
        status = FetchStatus.SKIPPED_EXISTING if result.skipped_existing else FetchStatus.DOWNLOADED
        return FileFetchResult(kind="sentinel1_grd", product_id=scene.product_id, product_name=scene.product_name,
                               pass_id=scene.pass_id, status=status, size_bytes=result.size,
                               local_path=_relative(dest, self.root), provenance_path=_relative(sidecar, self.root),
                               message=check)

    def dem(self, tile: DEMTile, dataset: str) -> FileFetchResult:
        if tile.product_id is None or tile.product_name is None:
            raise FetchPlanError(f"DEM tile {tile.tile_id} has no catalogue product")
        dest = self.dem_dir / f"{tile.product_name}.zip"
        meta, result, check = self._download(tile.product_id, tile.product_name, dest, check_dem_archive)
        sidecar = self._provenance(dest, meta, result, check, kind="copernicus_dem_tile",
                                   data_source=DataSource.COPERNICUS_DEM, tile_id=tile.tile_id, dem_dataset=dataset)
        status = FetchStatus.SKIPPED_EXISTING if result.skipped_existing else FetchStatus.DOWNLOADED
        return FileFetchResult(kind="copernicus_dem_tile", product_id=tile.product_id, product_name=tile.product_name,
                               tile_id=tile.tile_id, status=status, size_bytes=result.size,
                               local_path=_relative(dest, self.root), provenance_path=_relative(sidecar, self.root),
                               message=check)


def _s1_result(scene: PlannedScene, status: FetchStatus, message: str = "") -> FileFetchResult:
    return FileFetchResult(kind="sentinel1_grd", product_id=scene.product_id, product_name=scene.product_name,
                           pass_id=scene.pass_id, status=status, message=message)


def _dem_result(tile: DEMTile, status: FetchStatus, message: str = "") -> FileFetchResult:
    return FileFetchResult(kind="copernicus_dem_tile", product_id=tile.product_id or "",
                           product_name=tile.product_name or "", tile_id=tile.tile_id, status=status, message=message)


def completeness(plan: S1FetchPlan, files: Iterable[FileFetchResult]) -> tuple[dict[str, bool], dict[str, bool]]:
    present = {f.product_id for f in files if f.kind == "sentinel1_grd" and f.present}
    passes = {p.pass_id: all(pid in present for pid in p.product_ids) for p in plan.passes}
    ready = {c.candidate_id: all(passes[pid] for pid in (*c.pre_event_pass_ids, c.post_event_pass_id))
             for c in plan.candidates}
    return passes, ready


def execute_fetch(plan: S1FetchPlan, dem_plan: DEMPlan | None, cfg: BatoConfig, *, dry_run: bool,
                  downloader: Downloader | None = None, metadata_transport: Transport | None = None,
                  max_scenes: int | None = None, clock: Callable[[], datetime] | None = None,
                  logger: logging.Logger | None = None) -> FetchManifest:
    """Fetch (or, with ``dry_run``, only list) every planned file and write a manifest.

    Per-file failures are recorded in the manifest (status FAILED with the
    reason) and logged; they do not stop the remaining files.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    log = logger or get_logger("acquisition")
    started = clock()
    files: list[FileFetchResult] = []
    with stage("s1_fetch", logger=log, aoi_id=plan.aoi_id, dry_run=dry_run, scenes=len(plan.scenes),
               passes=len(plan.passes), candidates=len(plan.candidates),
               dem_tiles=len(dem_plan.tiles) if dem_plan else 0, max_scenes=max_scenes) as report:
        fetcher = None
        if not dry_run:
            if downloader is None:
                raise ValueError("a downloader is required unless dry_run is set")
            fetcher = _Fetcher(cfg, downloader, metadata_transport, clock, log)
        for index, scene in enumerate(plan.scenes):
            if fetcher is None:
                files.append(_s1_result(scene, FetchStatus.PLANNED))
            elif max_scenes is not None and index >= max_scenes:
                files.append(_s1_result(scene, FetchStatus.NOT_ATTEMPTED, f"beyond --max-scenes {max_scenes}"))
            else:
                try:
                    files.append(fetcher.s1(scene))
                except BatoError as exc:  # documented: recorded per file, the run continues
                    report.warn(f"{scene.product_name}: {exc}")
                    files.append(_s1_result(scene, FetchStatus.FAILED, str(exc)))
        for tile in (dem_plan.tiles if dem_plan else ()):
            if tile.product_id is None:
                report.warn(f"DEM tile {tile.tile_id}: no product in the catalogue (open sea or not published)")
                continue
            if fetcher is None:
                files.append(_dem_result(tile, FetchStatus.PLANNED))
                continue
            try:
                files.append(fetcher.dem(tile, dem_plan.dataset if dem_plan else ""))
            except BatoError as exc:
                report.warn(f"DEM {tile.tile_id}: {exc}")
                files.append(_dem_result(tile, FetchStatus.FAILED, str(exc)))

        pass_complete, candidate_ready = completeness(plan, files)
        manifest = FetchManifest(aoi_id=plan.aoi_id, dry_run=dry_run, started_at=started, finished_at=clock(),
                                 selection_digest=plan.selection_digest, plan=plan, dem_plan=dem_plan,
                                 files=tuple(files), pass_complete=pass_complete, candidate_ready=candidate_ready)
        counts: dict[str, int] = {}
        for f in files:
            counts[f.status.value] = counts.get(f.status.value, 0) + 1
        report.record_output(files=counts, complete_passes=sum(pass_complete.values()),
                             ready_candidates=sum(candidate_ready.values()))
    return manifest


def save_manifest(manifest: FetchManifest, cfg: BatoConfig) -> Path:
    """Dry runs and real runs are saved to different names, so a dry run never
    overwrites the record of a real download."""
    folder = assert_raw_output_dir(cfg.settings.paths.raw / cfg.settings.download.manifest_subdir, cfg.settings.paths)
    suffix = "fetch_plan" if manifest.dry_run else "fetch_manifest"
    return save_contract(manifest, folder / f"{manifest.aoi_id}_{suffix}.json")

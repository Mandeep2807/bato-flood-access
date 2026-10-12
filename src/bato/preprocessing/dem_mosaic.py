"""Step 3.4: the external DEM for terrain correction, from BATO's own Step 2C Copernicus DEM tiles.

SNAP's Terrain-Correction takes ONE external DEM file, so the GLO-30 1x1 degree tiles that cover
the input footprint (+ ``preprocessing.s1_terrain.dem_margin_km``) are placed on their common
grid into a single float32 GeoTIFF. Elevation values are copied unchanged (no resampling, no
smoothing, no void filling); only the extent is cut to what the footprint needs and pixels
without a tile get ``dem_nodata``.

Inputs come only from the Step 2C fetch manifest: each tile archive passes the data-rule guard
and is re-hashed against its download provenance before use.
"""

from __future__ import annotations

import hashlib
import math
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bato.acquisition.coverage import points_in_ring
from bato.acquisition.dem import dem_search_bbox, required_tile_ids
from bato.config import BatoConfig
from bato.contracts.common import BBox, PolygonGeometry
from bato.contracts.fetch import DownloadProvenance, FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.contracts.preprocessing import DEMMosaicRecord, DEMTileRecord
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError

GRID_TOLERANCE_PX = 1e-3          # tiles must sit on one common grid
VERTICAL_REFERENCE = "EGM2008 geoid heights (Copernicus DEM specification); converted by SNAP when dem_apply_egm"


class DEMError(BatoError):
    """The DEM tiles needed for terrain correction are missing, altered or inconsistent."""


@dataclass(frozen=True)
class DEMSource:
    tile_id: str
    product_name: str
    archive: Path
    provenance_path: Path
    provenance: DownloadProvenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def polygon_bbox(polygon: PolygonGeometry) -> BBox:
    lons = [p[0] for p in polygon.exterior]
    lats = [p[1] for p in polygon.exterior]
    return BBox(min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats))


def dem_extent(footprint: PolygonGeometry, margin_km: float) -> BBox:
    """Footprint bounding box grown by ``margin_km`` on every side."""
    return dem_search_bbox(polygon_bbox(footprint), margin_km)


def select_dem_tiles(manifest: FetchManifest, extent: BBox, cfg: BatoConfig) -> tuple[list[DEMSource], list[str]]:
    """The Step 2C tiles covering ``extent``, verified against their download provenance.

    Returns (tiles, required cells for which the catalogue has no tile, e.g. open sea).

    Raises:
        DEMError: no DEM plan, a required cell not planned or not downloaded, or a tile changed.
        ProhibitedInputError: a tile lies in an evaluation-only folder.
    """
    root = cfg.config_dir.parent
    plan = manifest.dem_plan
    if plan is None:
        raise DEMError("the fetch manifest has no Copernicus DEM plan; run Step 2C with DEM tiles (without --no-dem)")
    planned = {t.tile_id: t for t in plan.tiles}
    files = {f.tile_id: f for f in manifest.files if f.kind == "copernicus_dem_tile" and f.present}
    sources: list[DEMSource] = []
    sea: list[str] = []
    for tile_id in required_tile_ids(extent):
        tile = planned.get(tile_id)
        if tile is None:
            raise DEMError(f"DEM cell {tile_id} is needed but was not planned in Step 2C (dem.buffer_km too small "
                           "for this footprint); extend the DEM plan and fetch it")
        if tile.product_id is None:
            sea.append(tile_id)
            continue
        result = files.get(tile_id)
        if result is None or result.local_path is None or result.provenance_path is None:
            raise DEMError(f"DEM tile {tile_id} ({tile.product_name}) is not downloaded; re-run the Step 2C fetch")
        archive = assert_pipeline_input_allowed(root / result.local_path, cfg.settings.paths)
        provenance_path = assert_pipeline_input_allowed(root / result.provenance_path, cfg.settings.paths)
        try:
            provenance = load_contract(DownloadProvenance, provenance_path)
        except ContractIOError as exc:
            raise DEMError(str(exc)) from exc
        if provenance.kind != "copernicus_dem_tile" or provenance.tile_id != tile_id:
            raise DEMError(f"{provenance_path.name} does not describe DEM tile {tile_id}")
        if not archive.is_file():
            raise DEMError(f"DEM tile archive not found: {archive}")
        if _sha256(archive) != provenance.sha256:
            raise DEMError(f"{archive.name}: SHA-256 differs from its download provenance; the file changed")
        sources.append(DEMSource(tile_id=tile_id, product_name=result.product_name, archive=archive,
                                 provenance_path=provenance_path, provenance=provenance))
    if not sources:
        raise DEMError(f"no Copernicus DEM tile covers {extent}")
    return sources, sea


def extract_dem_raster(source: DEMSource, workdir: Path) -> tuple[Path, str]:
    """The tile's ``*_DEM.tif`` as a plain file in ``workdir`` (archives are read, never unpacked
    by member path). Returns (file, member name)."""
    archive = source.archive
    with archive.open("rb") as fh:
        magic = fh.read(4)
    target = workdir / f"{source.tile_id}_DEM.tif"
    if magic in (b"II*\x00", b"MM\x00*"):
        target.write_bytes(archive.read_bytes())
        return target, archive.name
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            infos = [m for m in t.getmembers() if m.isfile() and m.name.lower().endswith("_dem.tif")]
            if len(infos) != 1:
                raise DEMError(f"{archive.name}: expected one *_DEM.tif, found {[m.name for m in infos]}")
            stream = t.extractfile(infos[0])
            assert stream is not None
            target.write_bytes(stream.read())
            return target, infos[0].name
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if m.lower().endswith("_dem.tif")]
            if len(members) != 1:
                raise DEMError(f"{archive.name}: expected one *_DEM.tif, found {members}")
            target.write_bytes(z.read(members[0]))
            return target, members[0]
    raise DEMError(f"{archive.name} is neither a GeoTIFF, a zip nor a tar archive")


def build_mosaic(sources: list[DEMSource], extent: BBox, footprint: PolygonGeometry, out_path: Path, *,
                 nodata: float, margin_km: float, dataset: str, sea_tiles: list[str], root: Path) -> DEMMosaicRecord:
    """Place the tiles on their common grid, cut to ``extent``, write one float32 GeoTIFF (EPSG:4326).

    Raises:
        DEMError: a tile is not geographic, resolutions differ, or tiles are not on one grid.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    workdir = out_path.parent / "tiles"
    workdir.mkdir(parents=True, exist_ok=True)
    opened = []
    records = []
    for source in sorted(sources, key=lambda s: s.tile_id):
        raster, member = extract_dem_raster(source, workdir)
        ds = rasterio.open(raster)
        opened.append((source, raster, member, ds))
    try:
        ref = opened[0][3]
        xres, yres = float(ref.transform.a), float(-ref.transform.e)
        x_ref, y_ref = float(ref.transform.c), float(ref.transform.f)
        for source, _raster, _member, ds in opened:
            if ds.crs is None or not ds.crs.is_geographic:
                raise DEMError(f"DEM tile {source.tile_id} is not in geographic coordinates ({ds.crs})")
            if not (math.isclose(ds.transform.a, xres, rel_tol=1e-9) and math.isclose(-ds.transform.e, yres,
                                                                                          rel_tol=1e-9)):
                raise DEMError(f"DEM tile {source.tile_id} has resolution {ds.transform.a}, {-ds.transform.e}; "
                               f"expected {xres}, {yres} (tiles must share one grid)")
            if ds.transform.b != 0 or ds.transform.d != 0:
                raise DEMError(f"DEM tile {source.tile_id} is rotated")
        x0 = x_ref + math.floor((extent.min_lon - x_ref) / xres) * xres
        y0 = y_ref - math.floor((y_ref - extent.max_lat) / yres) * yres
        width = math.ceil((extent.max_lon - x0) / xres - GRID_TOLERANCE_PX)
        height = math.ceil((y0 - extent.min_lat) / yres - GRID_TOLERANCE_PX)
        mosaic = np.full((height, width), nodata, dtype=np.float32)
        for source, _raster, member, ds in opened:
            col_f, row_f = (ds.transform.c - x0) / xres, (y0 - ds.transform.f) / yres
            col_off, row_off = round(col_f), round(row_f)
            if abs(col_f - col_off) > GRID_TOLERANCE_PX or abs(row_f - row_off) > GRID_TOLERANCE_PX:
                raise DEMError(f"DEM tile {source.tile_id} is not on the common grid (offset {col_f}, {row_f} px)")
            c0, c1 = max(0, col_off), min(width, col_off + ds.width)
            r0, r1 = max(0, row_off), min(height, row_off + ds.height)
            if c0 < c1 and r0 < r1:
                block = np.asarray(ds.read(1, window=Window(c0 - col_off, r0 - row_off, c1 - c0, r1 - r0)),
                                   dtype=np.float64)
                bad = ~np.isfinite(block)
                if ds.nodata is not None:
                    bad |= block == ds.nodata
                mosaic[r0:r1, c0:c1] = np.where(bad, nodata, block).astype(np.float32)
            records.append(DEMTileRecord(
                tile_id=source.tile_id, product_name=source.product_name, archive=_relative(source.archive, root),
                archive_sha256=source.provenance.sha256, download_provenance=_relative(source.provenance_path, root),
                raster_member=member, width=int(ds.width), height=int(ds.height),
                resolution_deg=(float(ds.transform.a), float(-ds.transform.e)), crs=str(ds.crs)))
    finally:
        for _s, _r, _m, ds in opened:
            ds.close()
        for _s, raster, _m, _d in opened:
            raster.unlink(missing_ok=True)
        if workdir.exists() and not any(workdir.iterdir()):
            workdir.rmdir()

    with rasterio.open(out_path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
                       crs=CRS.from_epsg(4326), transform=from_origin(x0, y0, xres, yres), nodata=nodata) as dst:
        dst.write(mosaic, 1)

    # pixels of the mosaic inside the footprint that have no elevation
    lons = x0 + (np.arange(width) + 0.5) * xres
    lats = y0 - (np.arange(height) + 0.5) * yres
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    inside = points_in_ring(grid_lon, grid_lat, footprint.exterior)
    fb = polygon_bbox(footprint)
    return DEMMosaicRecord(
        path=_relative(out_path, root), sha256=_sha256(out_path), dataset=dataset,
        vertical_reference=VERTICAL_REFERENCE, crs="EPSG:4326", width=width, height=height,
        bounds=(x0, y0 - height * yres, x0 + width * xres, y0), resolution_deg=(xres, yres), nodata=nodata,
        margin_km=margin_km, footprint_bbox=(fb.min_lon, fb.min_lat, fb.max_lon, fb.max_lat),
        tiles=tuple(records), missing_sea_tiles=tuple(sea_tiles),
        nodata_pixels_in_footprint=int((inside & (mosaic == nodata)).sum()), footprint_pixels=int(inside.sum()))

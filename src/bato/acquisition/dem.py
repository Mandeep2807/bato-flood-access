"""Copernicus DEM GLO-30 tile discovery (Step 2C).

The DEM is an allowed input (official PS). It is found in the same CDSE
catalogue as Sentinel-1, by the product attribute ``dataset`` (config
``dem.cdse_dataset``, e.g. ``COP-DEM_GLO-30-DGED/2024_1``) intersected with
the AOI expanded by ``dem.buffer_km``. GLO-30 is distributed as 1x1 degree
tiles named after their lower-left corner (``N27E085``).

Cells without a product (e.g. open sea, where GLO-30 has no tile) are reported
as missing, never invented. Downloading reuses the Sentinel-1 downloader.
"""

from __future__ import annotations

import math
import re
from typing import Any

from bato.acquisition.cdse_odata import CDSEODataClient, parse_footprint
from bato.config import BatoConfig
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import BBox, PolygonGeometry
from bato.contracts.fetch import DEMPlan, DEMTile
from bato.exceptions import MetadataError
from bato.utils.geo import bbox_ring

_KM_PER_DEG_LAT = 111.32
_TILE_IN_NAME = re.compile(r"([NS])(\d{2})_00_([EW])(\d{3})_00")


def tile_id(lat_floor: int, lon_floor: int) -> str:
    return f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}"


def aoi_bbox_from_catalogue(catalogue: CatalogueSearchResult) -> BBox:
    """The AOI box of a Step 2A search, read back from its recorded query polygon."""
    match = re.search(r"POLYGON\(\(([^()]*)\)\)", catalogue.query_filter)
    if match is None:
        raise ValueError("catalogue query_filter contains no AOI polygon")
    points = [tuple(float(v) for v in pair.split()) for pair in match.group(1).split(",")]
    lons, lats = [p[0] for p in points], [p[1] for p in points]
    return BBox(min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats))


def dem_search_bbox(aoi_bbox: BBox, buffer_km: float) -> BBox:
    """AOI bounding box grown by ``buffer_km`` on every side (clamped to valid coordinates)."""
    b = aoi_bbox
    mid_lat = math.radians((b.min_lat + b.max_lat) / 2)
    dlat = buffer_km / _KM_PER_DEG_LAT
    dlon = buffer_km / (_KM_PER_DEG_LAT * max(math.cos(mid_lat), 0.01))
    return BBox(min_lon=max(b.min_lon - dlon, -180.0), min_lat=max(b.min_lat - dlat, -85.0),
                max_lon=min(b.max_lon + dlon, 180.0), max_lat=min(b.max_lat + dlat, 85.0))


def required_tile_ids(bbox: BBox) -> tuple[str, ...]:
    """Every 1x1 degree cell touched by the box (edges exactly on a degree line excluded)."""
    eps = 1e-9
    lats = range(math.floor(bbox.min_lat), math.floor(bbox.max_lat - eps) + 1)
    lons = range(math.floor(bbox.min_lon), math.floor(bbox.max_lon - eps) + 1)
    return tuple(tile_id(lat, lon) for lat in lats for lon in lons)


def build_dem_filter(area: PolygonGeometry, dataset: str) -> str:
    if "'" in dataset:
        raise ValueError("dataset name must not contain quotes")
    coords = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in area.exterior)
    return (f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'dataset' and "
            f"att/OData.CSC.StringAttribute/Value eq '{dataset}') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({coords}))')")


def tile_of_record(record: dict[str, Any]) -> str | None:
    """Tile id from the product name (``..._N27_00_E085_00...``), else from the footprint centre."""
    name = record.get("Name")
    if isinstance(name, str) and (m := _TILE_IN_NAME.search(name)):
        lat, lon = int(m.group(2)), int(m.group(4))
        return tile_id(lat if m.group(1) == "N" else -lat, lon if m.group(3) == "E" else -lon)
    try:
        ring = parse_footprint(record).exterior
    except MetadataError:
        return None
    lons, lats = [p[0] for p in ring[:-1]], [p[1] for p in ring[:-1]]
    return tile_id(math.floor(sum(lats) / len(lats)), math.floor(sum(lons) / len(lons)))


def plan_dem(aoi_id: str, aoi_bbox: BBox, cfg: BatoConfig, client: CDSEODataClient | None = None) -> DEMPlan:
    """Search the CDSE catalogue for the DEM tiles covering the AOI + buffer.

    Raises:
        CatalogueQueryError: the catalogue could not be queried.
    """
    dem = cfg.settings.dem
    bbox = dem_search_bbox(aoi_bbox, dem.buffer_km)
    area = PolygonGeometry(coordinates=(tuple(bbox_ring(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)),))
    query_filter = build_dem_filter(area, dem.cdse_dataset)
    client = client or CDSEODataClient(cfg.settings.cdse)
    records = client.fetch_filter(query_filter).records
    by_tile: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        tid = tile_of_record(record)
        if tid is not None and isinstance(record.get("Id"), str):
            by_tile.setdefault(tid, []).append(record)
    tiles = []
    for tid in required_tile_ids(bbox):
        candidates = sorted(by_tile.get(tid, []), key=lambda r: (str(r.get("Name")), str(r["Id"])))
        if not candidates:
            tiles.append(DEMTile(tile_id=tid))
            continue
        chosen = candidates[0]
        try:
            footprint = parse_footprint(chosen)
        except MetadataError:
            footprint = None
        tiles.append(DEMTile(tile_id=tid, product_id=chosen["Id"], product_name=str(chosen.get("Name")),
                             footprint=footprint))
    return DEMPlan(aoi_id=aoi_id, dataset=dem.cdse_dataset, buffer_km=dem.buffer_km,
                   required_tiles=required_tile_ids(bbox), tiles=tuple(tiles), query_filter=query_filter)

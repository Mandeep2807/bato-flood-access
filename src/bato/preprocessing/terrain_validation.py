"""Step 3.4 output validation: terrain-corrected linear sigma0 VV/VH on the EPSG:3857 10 m grid.

Structural checks decide validity: exactly Sigma0_VV + Sigma0_VH, linear (no dB), float,
one grid for both bands; a map geocoding in EPSG:3857 with the graph's pixel spacing, aligned to
the standard grid; terrain correction recorded (once), speckle filter recorded once, no dB or
other step; orbit file unchanged; at least one valid pixel; and the AOI still covered by valid
pixels as well as the input scene covered it. Statistics before (radar geometry) and after (map
geometry) are DIAGNOSTIC: no thresholds on sigma0 values.

The DIMAP header (SNAP's own record) is the authority for the CRS and the image-to-map transform;
what rasterio/GDAL reports for the ENVI files is cross-checked and differences are warnings.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bato.acquisition.coverage import points_in_ring
from bato.contracts.common import Polarization, PolygonGeometry
from bato.contracts.preprocessing import BandStatistics, TerrainValidation, ValidationCheck
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap
from bato.preprocessing.sigma0_validation import (
    FORBIDDEN_OPERATORS,
    SIGMA0_BAND,
    _open,
    band_statistics,
    require_rasterio,
)

TERRAIN_OPERATOR = "Terrain-Correction"
SPECKLE_OPERATOR = "Speckle-Filter"
FORBIDDEN_AFTER_TERRAIN = tuple(op for op in FORBIDDEN_OPERATORS if op not in (TERRAIN_OPERATOR, SPECKLE_OPERATOR))
EARTH_RADIUS_3857 = 6378137.0
PIXEL_SIZE_RELATIVE_TOLERANCE = 1e-6
GRID_ALIGNMENT_TOLERANCE_PX = 1e-3
COVERAGE_TOLERANCE = 0.01          # resolution of the AOI sample grid


def standard_grid_offsets(x0: float, y0: float, sx: float, sy: float, spacing: float,
                          origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Position of the upper-left PIXEL CENTRE relative to the standard grid, in pixels.

    SNAP's ``alignToStandardGrid`` (CRSGeoCodingHandler) snaps the easting/northing to
    ``origin + k * spacing`` and builds ``CrsGeoCoding(..., easting, northing, ...)``, whose
    reference pixel is (0.5, 0.5): the snapped coordinate is the centre of pixel (0, 0). The
    DIMAP IMAGE_TO_MODEL_TRANSFORM (x0, y0) holds the pixel CORNER, half a pixel away. The grid is
    aligned when both returned values are whole numbers.
    """
    return (x0 + sx / 2 - origin[0]) / spacing, (y0 - sy / 2 - origin[1]) / spacing


def lonlat_to_3857(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_RADIUS_3857 * np.radians(lon)
    y = EARTH_RADIUS_3857 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    return (math.degrees(x / EARTH_RADIUS_3857),
            math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_3857)) - math.pi / 2))


def dimap_map_geocoding(dim_path: Path) -> tuple[str | None, tuple[float, ...] | None]:
    """(CRS WKT, IMAGE_TO_MODEL_TRANSFORM) as SNAP wrote them; None where absent."""
    root = ET.parse(dim_path).getroot()
    wkt = None
    for crs in root.iter("Coordinate_Reference_System"):
        element = crs.find(".//WKT")
        if element is not None and element.text and element.text.strip():
            wkt = element.text.strip()
            break
    transform = None
    element = next(root.iter("IMAGE_TO_MODEL_TRANSFORM"), None)
    if element is not None and element.text:
        try:
            values = tuple(float(v) for v in element.text.split(","))
            transform = values if len(values) == 6 else None
        except ValueError:
            transform = None
    return wkt, transform


def _is_3857_wkt(wkt: str | None) -> bool | None:
    if not wkt:
        return None
    compact = "".join(wkt.split())
    return 'AUTHORITY["EPSG","3857"]]' in compact or ("Pseudo-Mercator" in wkt and '"3857"' in compact)


def aoi_samples(aoi: PolygonGeometry, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre sample points inside the AOI (same scheme as the AOI coverage grid)."""
    ring = aoi.exterior
    lon_min, lon_max = min(p[0] for p in ring), max(p[0] for p in ring)
    lat_min, lat_max = min(p[1] for p in ring), max(p[1] for p in ring)
    lons = lon_min + (lon_max - lon_min) / grid_size * (np.arange(grid_size) + 0.5)
    lats = lat_min + (lat_max - lat_min) / grid_size * (np.arange(grid_size) + 0.5)
    glon, glat = np.meshgrid(lons, lats)
    inside = points_in_ring(glon, glat, ring)
    return glon[inside], glat[inside]


def aoi_valid_fraction(image: Path, *, x0: float, y0: float, sx: float, sy: float, width: int, height: int,
                       nodata: float | None, lons: np.ndarray, lats: np.ndarray) -> float:
    """Share of AOI sample points that fall on a valid (finite, not no-data) pixel."""
    from rasterio.windows import Window

    x, y = lonlat_to_3857(lons, lats)
    cols = np.floor((x - x0) / sx).astype(np.int64)
    rows = np.floor((y0 - y) / sy).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not inside.any():
        return 0.0
    c0, c1 = int(cols[inside].min()), int(cols[inside].max()) + 1
    r0, r1 = int(rows[inside].min()), int(rows[inside].max()) + 1
    with _open(image) as ds:
        block = np.asarray(ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)), dtype=np.float64)
    values = block[rows[inside] - r0, cols[inside] - c0]
    valid = np.isfinite(values) & ((values != nodata) if nodata is not None else True)
    return round(float(valid.sum()) / float(lons.size), 4)


def validate_terrain_output(dim_path: Path, *, input_product: DimapProduct, polarizations: Sequence[Polarization],
                            pixel_spacing_m: float, aoi_polygon: PolygonGeometry,
                            input_scene_aoi_coverage: float | None, coverage_grid_size: int,
                            root: Path, grid_origin: tuple[float, float] = (0.0, 0.0)) -> TerrainValidation:
    """Validate a Step 3.4 DIMAP output. Never raises for a bad output: every problem becomes a
    failed check (only a missing rasterio raises)."""
    rasterio_version, gdal_version = require_rasterio()
    checks: list[ValidationCheck] = []
    warnings_: list[str] = []

    def check(name: str, passed: bool, detail: str) -> bool:
        checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    def result(**kw: Any) -> TerrainValidation:
        return TerrainValidation(passed=all(c.passed for c in checks), checks=tuple(checks),
                                 warnings=tuple(warnings_), rasterio_version=rasterio_version,
                                 gdal_version=gdal_version, input_scene_aoi_coverage=input_scene_aoi_coverage, **kw)

    try:
        product = read_dimap(dim_path)
        wkt, i2m = dimap_map_geocoding(dim_path)
    except (DimapError, ET.ParseError) as exc:
        check("output_exists", False, str(exc))
        return result()
    check("output_exists", True, f"{dim_path.name} and {product.data_dir.name}/ present")

    expected = {SIGMA0_BAND.format(pol=p.value): p for p in polarizations}
    names = [b.name for b in product.bands]
    missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
    bands_ok = check("sigma0_bands", not missing and not extra,
                     f"bands {names}" + (f"; missing {missing}" if missing else "")
                     + (f"; unexpected {extra} (only the terrain-corrected sigma0 bands are allowed)" if extra else ""))
    present = [b for n in expected if (b := product.band(n)) is not None]
    db_like = [b.name for b in present
               if "db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled]
    check("linear_scale", not db_like, "units " + str({b.name: b.unit for b in present})
          + (f"; dB/log10 scaling found on {db_like}" if db_like else "; no dB, no log10 scaling"))

    # ---- map geometry ----------------------------------------------------------------------
    check("map_geometry", product.map_geocoding and i2m is not None,
          f"IMAGE_TO_MODEL_TRANSFORM {i2m}" if i2m else "no map geocoding in the DIMAP header (still radar geometry?)")
    sx = sy = x0 = y0 = None
    if i2m is not None:
        scale_x, shear_y, shear_x, scale_y, x0, y0 = i2m
        sx, sy = abs(scale_x), abs(scale_y)
        size_ok = (math.isclose(sx, pixel_spacing_m, rel_tol=PIXEL_SIZE_RELATIVE_TOLERANCE)
                   and math.isclose(sy, pixel_spacing_m, rel_tol=PIXEL_SIZE_RELATIVE_TOLERANCE)
                   and shear_x == 0 and shear_y == 0 and scale_x > 0 and scale_y < 0)
        check("pixel_size", size_ok, f"{scale_x} x {scale_y} map units, shear ({shear_x}, {shear_y}); "
              f"expected {pixel_spacing_m} x -{pixel_spacing_m}, no shear")
        fx, fy = standard_grid_offsets(x0, y0, sx, sy, pixel_spacing_m, grid_origin)
        check("standard_grid", abs(fx - round(fx)) <= GRID_ALIGNMENT_TOLERANCE_PX
              and abs(fy - round(fy)) <= GRID_ALIGNMENT_TOLERANCE_PX,
              f"upper-left corner ({x0}, {y0}); pixel centre ({x0 + sx / 2}, {y0 - sy / 2}) = ({fx:.4f}, {fy:.4f}) "
              f"pixels from the standard-grid origin {grid_origin} (SNAP puts pixel centres on whole multiples)")

    # ---- processing history and metadata -----------------------------------------------------
    history = list(product.history)
    if history:
        forbidden = [op for op in history if op in FORBIDDEN_AFTER_TERRAIN]
        check("processing_history", not forbidden and history.count(TERRAIN_OPERATOR) == 1
              and history.count(SPECKLE_OPERATOR) == 1,
              f"SNAP history {history}; Terrain-Correction {history.count(TERRAIN_OPERATOR)}x, Speckle-Filter "
              f"{history.count(SPECKLE_OPERATOR)}x" + (f"; forbidden: {forbidden}" if forbidden else ""))
    else:
        warnings_.append("the DIMAP header records no Processing_Graph history")
    flag = product.abstracted.get("is_terrain_corrected")
    check("terrain_flag", flag not in ("0", "false"), f"abstracted metadata is_terrain_corrected={flag}")
    orbit_in = input_product.abstracted.get("orbit_state_vector_file") or None
    orbit_out = product.abstracted.get("orbit_state_vector_file") or None
    check("orbit_preserved", orbit_out == orbit_in, f"orbit file {orbit_out!r} (input {orbit_in!r})")

    # ---- pixels ----------------------------------------------------------------------------
    before: list[BandStatistics] = []
    after: list[BandStatistics] = []
    rio_crs: Any = None
    rio_transform: Any = None
    readable: dict[str, bool] = {}
    for name, pol in expected.items():
        band, band_in = product.band(name), input_product.band(name)
        if band is None or band.image_file is None:
            continue
        try:
            st, dtype, crs, transform = band_statistics(band.image_file, band=name, polarization=pol, unit=band.unit,
                                                        nodata=band.nodata_value, edge_width_px=0, root=root)
        except Exception as exc:  # any read error becomes a failed check, reported with its type
            readable[name] = False
            check(f"readable_{pol.value}", False, f"{band.image_file.name}: {type(exc).__name__}: {exc}")
            continue
        readable[name] = True
        check(f"readable_{pol.value}", True, f"{band.image_file.name}: {st.width} x {st.height}, {dtype}")
        check(f"float_{pol.value}", bool(np.issubdtype(np.dtype(dtype), np.floating)),
              f"rasterio dtype {dtype}, DIMAP data type {band.data_type or '?'}")
        check(f"valid_pixels_{pol.value}", st.valid_pixels > 0,
              f"{st.valid_pixels:,} of {st.total_pixels:,} pixels finite and not no-data")
        after.append(st)
        rio_crs, rio_transform = crs, transform
        if band_in is not None and band_in.image_file is not None:
            st_in, _, _, _ = band_statistics(band_in.image_file, band=name, polarization=pol, unit=band_in.unit,
                                             nodata=band_in.nodata_value, edge_width_px=0, root=root)
            before.append(st_in)
        if st.negative_pixels:
            warnings_.append(f"{name}: {st.negative_pixels:,} negative values (reported, not clipped)")
        if st.nonfinite_pixels:
            warnings_.append(f"{name}: {st.nonfinite_pixels:,} non-finite values (counted as invalid)")

    if bands_ok and present and all(readable.get(n, False) for n in expected):
        sizes = {(s.width, s.height) for s in after}
        check("same_grid", sizes == {(product.width, product.height)},
              f"bands {sorted(sizes)}, DIMAP {product.width} x {product.height}")

    # ---- CRS: DIMAP is the authority; rasterio is cross-checked -----------------------------
    rio_epsg = None
    if rio_crs is not None:
        try:
            rio_epsg = rio_crs.to_epsg()
        except Exception:  # an unidentifiable CRS is reported below
            rio_epsg = None
    dimap_3857 = _is_3857_wkt(wkt)
    crs_ok = (dimap_3857 is True or rio_epsg == 3857) and dimap_3857 is not False \
        and (rio_crs is None or rio_epsg in (3857, None))
    check("crs_epsg_3857", crs_ok, f"DIMAP WKT {'EPSG:3857' if dimap_3857 else ('absent' if wkt is None else 'other')}"
          f"; rasterio {rio_crs if rio_crs is not None else 'no CRS'}")
    rio_tuple = tuple(float(v) for v in tuple(rio_transform)[:6]) if rio_transform is not None else None
    if rio_tuple is not None and i2m is not None and rio_crs is not None:
        expected_rio = (i2m[0], i2m[2], i2m[4], i2m[1], i2m[3], i2m[5])   # rasterio order a, b, c, d, e, f
        pairs = zip(rio_tuple, expected_rio, strict=True)
        if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6) for a, b in pairs):
            warnings_.append(f"rasterio transform {rio_tuple} differs from the DIMAP transform {expected_rio}")
    elif rio_crs is None and i2m is not None:
        warnings_.append("rasterio/GDAL reports no CRS for the ENVI band files; the DIMAP header carries it")

    # ---- extent and AOI coverage ----------------------------------------------------------------
    footprint = bounds = None
    fractions: dict[str, float] = {}
    if i2m is not None and sx and sy and x0 is not None and y0 is not None:
        west, north, east, south = x0, y0, x0 + product.width * sx, y0 - product.height * sy
        bounds = (west, south, east, north)
        corners = [to_lonlat(west, north), to_lonlat(east, north), to_lonlat(east, south), to_lonlat(west, south)]
        ring = tuple((round(lon, 6), round(lat, 6)) for lon, lat in corners)
        footprint = PolygonGeometry(coordinates=((*ring, ring[0]),))
        lons, lats = aoi_samples(aoi_polygon, coverage_grid_size)
        needed = 1.0 if input_scene_aoi_coverage is None else input_scene_aoi_coverage
        for name, pol in expected.items():
            band = product.band(name)
            if band is None or band.image_file is None or not readable.get(name):
                continue
            fractions[name] = aoi_valid_fraction(band.image_file, x0=x0, y0=y0, sx=sx, sy=sy, width=product.width,
                                                 height=product.height, nodata=band.nodata_value, lons=lons, lats=lats)
            check(f"aoi_coverage_{pol.value}", fractions[name] + COVERAGE_TOLERANCE >= needed,
                  f"{fractions[name]:.1%} of AOI sample points on valid pixels; input scene covered {needed:.1%}")

    return result(width=product.width, height=product.height,
                  geometry="map (EPSG:3857)" if crs_ok and i2m is not None else "NOT the expected map geometry",
                  crs_wkt=wkt, image_to_model_transform=i2m,
                  rasterio_crs=str(rio_crs) if rio_crs is not None else None, rasterio_transform=rio_tuple,
                  pixel_size_m=(sx, sy) if sx and sy else None, bounds_3857=bounds, footprint=footprint,
                  aoi_valid_fraction=fractions, orbit_file=orbit_out, dem_recorded=product.abstracted.get("DEM"),
                  processing_history=product.history, bands_before=tuple(before), bands_after=tuple(after))

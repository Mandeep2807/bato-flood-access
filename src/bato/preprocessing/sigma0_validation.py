"""Step 3.2 output validation: linear sigma0 VV/VH in radar geometry, read with rasterio.

Structural checks decide validity (see :class:`RadiometricValidation`); value
statistics are diagnostic only - there are NO acceptance thresholds on sigma0
values, and nothing is clipped, normalised or converted to dB.

Radar geometry at this stage means: the image grid is the SAR acquisition grid
(no map projection). rasterio/GDAL therefore reports no CRS and an identity
transform for the ENVI band files; the geolocation is carried by the latitude /
longitude tie-point grids, which are validated and used for the AOI coverage
check instead.

rasterio is imported lazily so the rest of BATO (and ordinary unit tests) does
not need it. :func:`require_rasterio` is called before SNAP runs, so a missing
rasterio never wastes a long gpt run.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bato.acquisition.coverage import AOICoverageGrid
from bato.contracts.common import Polarization, PolygonGeometry
from bato.contracts.preprocessing import BandStatistics, RadiometricValidation, ValidationCheck
from bato.contracts.safe import SafeProduct
from bato.exceptions import BatoError
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap

SIGMA0_BAND = "Sigma0_{pol}"
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
# Consistency tolerances (not scientific thresholds):
SPACING_RELATIVE_TOLERANCE = 0.01   # metadata rounding between SAFE annotation and SNAP abstracted metadata
COVERAGE_TOLERANCE = 0.01           # resolution of the AOI coverage sample grid
QUANTILE_SAMPLE_TARGET = 4_000_000  # pixels sampled (regular stride) for quantiles
BLOCK_PIXELS = 4_000_000            # pixels read per block (memory bound)
# Operators that must NOT have run at this stage (speckle, terrain, dB, normalisation-like steps).
FORBIDDEN_OPERATORS = ("Speckle-Filter", "Multi-Temporal-Speckle-Filter", "Terrain-Correction",
                       "Ellipsoid-Correction-GG", "Ellipsoid-Correction-RD", "SARSim-Terrain-Correction",
                       "Terrain-Flattening", "LinearToFromdB", "Land-Sea-Mask", "BandMaths")


class ValidationSetupError(BatoError):
    """The validation cannot run at all (e.g. rasterio is not installed)."""


def require_rasterio() -> tuple[str, str]:
    """Return (rasterio version, GDAL version).

    Raises:
        ValidationSetupError: rasterio is not importable.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ValidationSetupError("rasterio is required to validate preprocessing outputs: "
                                   "pip install rasterio (or pip install -e .[preprocessing])") from exc
    return str(rasterio.__version__), str(getattr(rasterio, "__gdal_version__", "unknown"))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _open(path: Path) -> Any:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)   # expected: radar geometry
        return rasterio.open(path)


def _read_rows(ds: Any, row: int, rows: int) -> np.ndarray:
    from rasterio.windows import Window

    return np.asarray(ds.read(1, window=Window(0, row, ds.width, rows)))


def band_statistics(path: Path, *, band: str, polarization: Polarization, unit: str | None,
                    nodata: float | None, edge_width_px: int, root: Path) -> tuple[BandStatistics, str, Any, Any]:
    """Block-wise statistics of one ENVI band. Returns (stats, dtype, crs, transform)."""
    with _open(path) as ds:
        width, height, dtype = int(ds.width), int(ds.height), str(ds.dtypes[0])
        crs, transform = ds.crs, ds.transform
        stride = max(1, math.ceil(math.sqrt(width * height / QUANTILE_SAMPLE_TARGET)))
        block_rows = max(1, BLOCK_PIXELS // max(width, 1))
        finite = nodata_n = zeros = negative = zero_edge = 0
        total_sum = 0.0
        vmin, vmax = math.inf, -math.inf
        samples: list[np.ndarray] = []
        cols = np.arange(width)
        edge_cols = (cols < edge_width_px) | (cols >= width - edge_width_px)
        for row in range(0, height, block_rows):
            rows = min(block_rows, height - row)
            data = _read_rows(ds, row, rows).astype(np.float64, copy=False)
            is_finite = np.isfinite(data)
            finite += int(is_finite.sum())
            is_zero = data == 0.0
            zeros += int(is_zero.sum())
            negative += int((data < 0).sum())
            row_idx = np.arange(row, row + rows)
            edge_rows = (row_idx < edge_width_px) | (row_idx >= height - edge_width_px)
            edge = edge_rows[:, None] | edge_cols[None, :]
            zero_edge += int((is_zero & edge).sum())
            valid = is_finite if nodata is None else is_finite & (data != nodata)
            if nodata is not None:
                nodata_n += int((is_finite & (data == nodata)).sum())
            if valid.any():
                values = data[valid]
                total_sum += float(values.sum())
                vmin, vmax = min(vmin, float(values.min())), max(vmax, float(values.max()))
            sample_rows = np.nonzero(row_idx % stride == 0)[0]
            if sample_rows.size:
                sub, sub_valid = data[sample_rows, ::stride], valid[sample_rows, ::stride]
                samples.append(sub[sub_valid])
    total = width * height
    valid_n = finite - nodata_n
    sample = np.concatenate(samples) if samples else np.empty(0)
    quantiles: dict[str, float] = {}
    if sample.size:
        values = np.quantile(sample, QUANTILES)
        quantiles = {f"p{round(q * 100):02d}": float(v) for q, v in zip(QUANTILES, values, strict=True)}
    stats = BandStatistics(
        band=band, polarization=polarization, file=_relative(path, root), dtype=dtype, unit=unit,
        nodata_value=nodata, width=width, height=height, total_pixels=total, finite_pixels=finite,
        nonfinite_pixels=total - finite, nodata_pixels=nodata_n, zero_pixels=zeros, negative_pixels=negative,
        valid_pixels=valid_n, min=vmin if valid_n else None, max=vmax if valid_n else None,
        mean=total_sum / valid_n if valid_n else None, median=quantiles.get("p50"), quantiles=quantiles,
        quantile_sample_stride=stride, quantile_sample_size=int(sample.size), edge_width_px=edge_width_px,
        zero_pixels_edge=zero_edge, zero_pixels_interior=zeros - zero_edge,
    )
    return stats, dtype, crs, transform


def tie_point_footprint(product: DimapProduct) -> PolygonGeometry:
    """Footprint from the corners of the latitude/longitude tie-point grids.

    Raises:
        DimapError: grids missing, unreadable, non-finite or out of range.
    """
    grids = {}
    for name in ("latitude", "longitude"):
        file = product.tie_point_grids.get(name)
        if file is None or not file.is_file():
            raise DimapError(f"{product.path.name}: {name} tie-point grid not found")
        with _open(file) as ds:
            grids[name] = np.asarray(ds.read(1), dtype=np.float64)
    lat, lon = grids["latitude"], grids["longitude"]
    if lat.shape != lon.shape or lat.size < 4:
        raise DimapError(f"{product.path.name}: latitude/longitude tie-point grids differ in shape or are too small")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise DimapError(f"{product.path.name}: tie-point grids contain non-finite values")
    if np.abs(lat).max() > 90 or np.abs(lon).max() > 180:
        raise DimapError(f"{product.path.name}: tie-point coordinates out of range")
    corners = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
    ring = [(round(float(lon[r, c]), 6), round(float(lat[r, c]), 6)) for r, c in corners]
    try:
        return PolygonGeometry(coordinates=((*ring, ring[0]),))
    except ValueError as exc:
        raise DimapError(f"{product.path.name}: tie-point corners do not form a valid polygon ({exc})") from exc


def validate_sigma0_output(dim_path: Path, *, polarizations: Sequence[Polarization], safe: SafeProduct,
                           aoi_polygon: PolygonGeometry, input_scene_aoi_coverage: float | None,
                           expect_precise_orbit: bool, border_limit: int, coverage_grid_size: int,
                           root: Path) -> RadiometricValidation:
    """Validate a Step 3.2 DIMAP output. Never raises for a bad output: every
    problem becomes a failed check (only a missing rasterio raises)."""
    rasterio_version, gdal_version = require_rasterio()
    checks: list[ValidationCheck] = []
    warnings_: list[str] = []

    def check(name: str, passed: bool, detail: str) -> bool:
        checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    def result(**kw: Any) -> RadiometricValidation:
        return RadiometricValidation(passed=all(c.passed for c in checks), checks=tuple(checks),
                                     warnings=tuple(warnings_), rasterio_version=rasterio_version,
                                     gdal_version=gdal_version, **kw)

    try:
        product = read_dimap(dim_path)
    except DimapError as exc:
        check("output_exists", False, str(exc))
        return result()
    check("output_exists", True, f"{dim_path.name} and {product.data_dir.name}/ present")

    # Bands: exactly linear sigma0 for the requested polarisations.
    expected = {SIGMA0_BAND.format(pol=p.value): p for p in polarizations}
    names = [b.name for b in product.bands]
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    bands_ok = check("sigma0_bands", not missing and not extra,
                     f"bands {names}" + (f"; missing {missing}" if missing else "")
                     + (f"; unexpected {extra} (only sigma0 is allowed)" if extra else ""))
    present = [product.band(n) for n in expected if product.band(n) is not None]
    db_like = [b.name for b in present if b is not None and
               ("db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled)]
    check("linear_scale", not db_like,
          "units " + str({b.name: b.unit for b in present if b is not None})
          + (f"; dB/log10 scaling found on {db_like}" if db_like else "; no dB, no log10 scaling"))

    # Geometry: radar geometry (tie-point geocoding, no map transform, no terrain-correction history).
    terrain_flag = product.abstracted.get("is_terrain_corrected")
    forbidden = [op for op in product.history if op in FORBIDDEN_OPERATORS]
    radar = (not product.map_geocoding and {"latitude", "longitude"} <= set(product.tie_point_grids)
             and terrain_flag not in ("1", "true"))
    check("radar_geometry", radar,
          ("tie-point geocoding, no map transform" if radar else
           f"map geocoding={product.map_geocoding}, tie-point grids={sorted(product.tie_point_grids)}")
          + f", is_terrain_corrected={terrain_flag}")
    check("no_downstream_operators", not forbidden,
          f"SNAP history {list(product.history) or '(not recorded)'}"
          + (f"; forbidden at this stage: {forbidden}" if forbidden else ""))
    if not product.history:
        warnings_.append("the DIMAP header records no Processing_Graph history")

    # Orbit.
    orbit_file = product.abstracted.get("orbit_state_vector_file") or None
    orbit_ok = bool(orbit_file) and (not expect_precise_orbit or "POEORB" in (orbit_file or "").upper())
    check("precise_orbit", orbit_ok, f"orbit_state_vector_file={orbit_file!r}"
          + (" (precise POEORB expected)" if expect_precise_orbit else ""))

    # Pixel spacing vs the SAFE annotation.
    rng, azi = product.abstracted_float("range_spacing"), product.abstracted_float("azimuth_spacing")
    ref = safe.bands[0]
    spacing_ok = (rng is not None and azi is not None
                  and abs(rng - ref.range_pixel_spacing_m) <= SPACING_RELATIVE_TOLERANCE * ref.range_pixel_spacing_m
                  and abs(azi - ref.azimuth_pixel_spacing_m)
                  <= SPACING_RELATIVE_TOLERANCE * ref.azimuth_pixel_spacing_m)
    check("pixel_spacing", spacing_ok,
          f"output {rng} x {azi} m (range x azimuth), SAFE annotation {ref.range_pixel_spacing_m:g} x "
          f"{ref.azimuth_pixel_spacing_m:g} m")

    # Pixels.
    stats: list[BandStatistics] = []
    crs_seen: Any = None
    transform_seen: Any = None
    for name, pol in expected.items():
        band = product.band(name)
        if band is None or band.image_file is None:
            continue
        try:
            st, dtype, crs, transform = band_statistics(
                band.image_file, band=name, polarization=pol, unit=band.unit, nodata=band.nodata_value,
                edge_width_px=border_limit, root=root)
        except Exception as exc:  # any read error becomes a failed check, reported with its type
            check(f"readable_{pol.value}", False, f"{band.image_file.name}: {type(exc).__name__}: {exc}")
            continue
        check(f"readable_{pol.value}", True, f"{band.image_file.name}: {st.width} x {st.height}, {dtype}")
        check(f"float_{pol.value}", np.issubdtype(np.dtype(dtype), np.floating),
              f"rasterio dtype {dtype}, DIMAP data type {band.data_type or '?'}")
        stats.append(st)
        crs_seen, transform_seen = crs, transform
        if band.nodata_value is None:
            warnings_.append(f"{name}: no no-data value declared; zeros are counted as valid pixels")

    if len(stats) == len(expected) and bands_ok:
        sizes = {(s.width, s.height) for s in stats}
        ref_size = (product.width, product.height)
        within = product.width <= ref.samples and product.height <= ref.lines
        check("same_grid", sizes == {ref_size} and within,
              f"bands {sorted(sizes)}, DIMAP {product.width} x {product.height}, "
              f"input scene {ref.samples} x {ref.lines}")
    for st in stats:
        check(f"valid_pixels_{st.polarization.value}", st.valid_pixels > 0,
              f"{st.valid_pixels:,} of {st.total_pixels:,} pixels finite and not no-data")
        if st.negative_pixels:
            warnings_.append(f"{st.band}: {st.negative_pixels:,} negative values (reported, not clipped)")
    if crs_seen is not None:
        warnings_.append(f"rasterio reports a CRS ({crs_seen}) for a radar-geometry band; geolocation is "
                         "taken from the tie-point grids")

    # Geolocation and AOI coverage.
    footprint = coverage = None
    try:
        footprint = tie_point_footprint(product)
    except DimapError as exc:
        check("geolocation", False, str(exc))
    else:
        check("geolocation", True, "latitude/longitude tie-point grids readable, finite, in range")
        coverage = AOICoverageGrid(aoi_polygon, coverage_grid_size).fraction([footprint])
        needed = 1.0 if input_scene_aoi_coverage is None else input_scene_aoi_coverage
        scene = (f"{input_scene_aoi_coverage:.1%}" if input_scene_aoi_coverage is not None
                 else "unknown (full coverage required)")
        check("aoi_coverage", coverage + COVERAGE_TOLERANCE >= needed,
              f"output footprint covers {coverage:.1%} of the AOI; input scene covers {scene}")

    transform_tuple = tuple(float(v) for v in tuple(transform_seen)[:6]) if transform_seen is not None else None
    return result(width=product.width, height=product.height,
                  geometry="radar (tie-point geocoding)" if radar else "NOT radar geometry",
                  rasterio_crs=str(crs_seen) if crs_seen is not None else None,
                  rasterio_transform=transform_tuple, range_spacing_m=rng, azimuth_spacing_m=azi,
                  footprint=footprint, aoi_coverage=coverage, input_scene_aoi_coverage=input_scene_aoi_coverage,
                  orbit_file=orbit_file, processing_history=product.history, bands=tuple(stats))

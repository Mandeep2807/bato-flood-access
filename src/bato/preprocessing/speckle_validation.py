"""Step 3.3 output validation: Lee Sigma filtered linear sigma0 VV/VH, still in radar geometry.

Structural checks decide validity: the same bands, grid, geolocation, pixel spacing and orbit
as the Step 3.2 input; linear units (no dB); no terrain correction / map projection / other
processing; float data; the filter actually changed the data; and no pixel that was valid in
the input became NaN/inf. Everything else (statistics before vs after, means, coefficient of
variation, no-data changes, negatives) is DIAGNOSTIC: no thresholds on sigma0 values.

Reuses the Step 3.2 readers (:mod:`bato.preprocessing.sigma0_validation`, :mod:`.dimap`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bato.contracts.common import Polarization, PolygonGeometry
from bato.contracts.preprocessing import BandComparison, BandStatistics, SpeckleValidation, ValidationCheck
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap
from bato.preprocessing.sigma0_validation import (
    BLOCK_PIXELS,
    FORBIDDEN_OPERATORS,
    SIGMA0_BAND,
    SPACING_RELATIVE_TOLERANCE,
    _open,
    _read_rows,
    band_statistics,
    require_rasterio,
    tie_point_footprint,
)

SPECKLE_OPERATOR = "Speckle-Filter"
# Operators that must not appear after Step 3.3 (the speckle filter itself is expected, once).
FORBIDDEN_AFTER_SPECKLE = tuple(op for op in FORBIDDEN_OPERATORS if op != SPECKLE_OPERATOR)
FOOTPRINT_TOLERANCE_DEG = 1e-6      # tie-point grids are copied unchanged by Speckle-Filter


def _valid(data: np.ndarray, nodata: float | None) -> np.ndarray:
    finite = np.isfinite(data)
    return finite if nodata is None else finite & (data != nodata)


def compare_band(before: Path, after: Path, *, band: str, nodata_before: float | None,
                 nodata_after: float | None) -> BandComparison:
    """Block-wise comparison of two same-sized single-band rasters."""
    with _open(before) as a, _open(after) as b:
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(f"{band}: grids differ ({a.width} x {a.height} vs {b.width} x {b.height})")
        width, height = int(a.width), int(a.height)
        block_rows = max(1, BLOCK_PIXELS // max(width, 1))
        compared = changed = new_nonfinite = new_nodata = restored = new_negative = 0
        both_n = 0
        sums = np.zeros(5)   # sum_before, sum_after, sum_abs_diff, sumsq_before, sumsq_after
        for row in range(0, height, block_rows):
            rows = min(block_rows, height - row)
            x = _read_rows(a, row, rows).astype(np.float64, copy=False)
            y = _read_rows(b, row, rows).astype(np.float64, copy=False)
            vx, vy = _valid(x, nodata_before), _valid(y, nodata_after)
            compared += int(vx.sum())
            changed += int((vx & ~(np.isfinite(y) & (x == y))).sum())
            new_nonfinite += int((vx & ~np.isfinite(y)).sum())
            new_nodata += int((vx & np.isfinite(y) & ~vy).sum())
            restored += int((~vx & vy).sum())
            new_negative += int((vx & (x >= 0) & np.isfinite(y) & (y < 0)).sum())
            both = vx & vy
            if both.any():
                xb, yb = x[both], y[both]
                both_n += int(both.sum())
                sums += (xb.sum(), yb.sum(), np.abs(yb - xb).sum(), (xb * xb).sum(), (yb * yb).sum())

    def cv(total: float, total_sq: float) -> float | None:
        if not both_n:
            return None
        mean = total / both_n
        var = max(total_sq / both_n - mean * mean, 0.0)
        return math.sqrt(var) / mean if mean else None

    return BandComparison(
        band=band, compared_pixels=compared, changed_pixels=changed, new_nonfinite_pixels=new_nonfinite,
        new_nodata_pixels=new_nodata, restored_pixels=restored, new_negative_pixels=new_negative,
        mean_before=float(sums[0] / both_n) if both_n else None, mean_after=float(sums[1] / both_n) if both_n else None,
        mean_abs_difference=float(sums[2] / both_n) if both_n else None,
        cv_before=cv(float(sums[0]), float(sums[3])), cv_after=cv(float(sums[1]), float(sums[4])))


def _same_footprint(a: PolygonGeometry, b: PolygonGeometry) -> bool:
    return len(a.exterior) == len(b.exterior) and all(
        abs(p[0] - q[0]) <= FOOTPRINT_TOLERANCE_DEG and abs(p[1] - q[1]) <= FOOTPRINT_TOLERANCE_DEG
        for p, q in zip(a.exterior, b.exterior, strict=True))


def validate_speckle_output(dim_path: Path, *, input_product: DimapProduct, polarizations: Sequence[Polarization],
                            border_limit: int, root: Path) -> SpeckleValidation:
    """Validate a Step 3.3 DIMAP output against its Step 3.2 input. Never raises for a bad
    output: every problem becomes a failed check (only a missing rasterio raises)."""
    rasterio_version, gdal_version = require_rasterio()
    checks: list[ValidationCheck] = []
    warnings_: list[str] = []

    def check(name: str, passed: bool, detail: str) -> bool:
        checks.append(ValidationCheck(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    def result(**kw: Any) -> SpeckleValidation:
        return SpeckleValidation(passed=all(c.passed for c in checks), checks=tuple(checks),
                                 warnings=tuple(warnings_), rasterio_version=rasterio_version,
                                 gdal_version=gdal_version, **kw)

    try:
        product = read_dimap(dim_path)
    except DimapError as exc:
        check("output_exists", False, str(exc))
        return result()
    check("output_exists", True, f"{dim_path.name} and {product.data_dir.name}/ present")

    expected = {SIGMA0_BAND.format(pol=p.value): p for p in polarizations}
    names = [b.name for b in product.bands]
    missing, extra = sorted(set(expected) - set(names)), sorted(set(names) - set(expected))
    bands_ok = check("sigma0_bands", not missing and not extra,
                     f"bands {names}" + (f"; missing {missing}" if missing else "")
                     + (f"; unexpected {extra} (only the filtered sigma0 bands are allowed)" if extra else ""))
    present = [b for n in expected if (b := product.band(n)) is not None]
    db_like = [b.name for b in present
               if "db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled]
    check("linear_scale", not db_like, "units " + str({b.name: b.unit for b in present})
          + (f"; dB/log10 scaling found on {db_like}" if db_like else "; no dB, no log10 scaling"))

    terrain_flag = product.abstracted.get("is_terrain_corrected")
    radar = (not product.map_geocoding and {"latitude", "longitude"} <= set(product.tie_point_grids)
             and terrain_flag not in ("1", "true"))
    check("radar_geometry", radar,
          ("tie-point geocoding, no map transform" if radar else
           f"map geocoding={product.map_geocoding}, tie-point grids={sorted(product.tie_point_grids)}")
          + f", is_terrain_corrected={terrain_flag}")
    forbidden = [op for op in product.history if op in FORBIDDEN_AFTER_SPECKLE]
    speckle_runs = product.history.count(SPECKLE_OPERATOR)
    check("no_downstream_operators", not forbidden and speckle_runs <= 1,
          f"SNAP history {list(product.history) or '(not recorded)'}"
          + (f"; forbidden at this stage: {forbidden}" if forbidden else "")
          + (f"; Speckle-Filter recorded {speckle_runs} times" if speckle_runs > 1 else ""))
    if product.history:
        check("speckle_recorded", speckle_runs == 1, f"Speckle-Filter in SNAP history: {speckle_runs} time(s)")
    else:
        warnings_.append("the DIMAP header records no Processing_Graph history")

    orbit_in = input_product.abstracted.get("orbit_state_vector_file") or None
    orbit_out = product.abstracted.get("orbit_state_vector_file") or None
    check("orbit_preserved", orbit_out == orbit_in, f"orbit file {orbit_out!r} (input {orbit_in!r})")

    rng, azi = product.abstracted_float("range_spacing"), product.abstracted_float("azimuth_spacing")
    rng_in, azi_in = input_product.abstracted_float("range_spacing"), input_product.abstracted_float("azimuth_spacing")
    spacing_ok = (None not in (rng, azi, rng_in, azi_in)
                  and abs(rng - rng_in) <= SPACING_RELATIVE_TOLERANCE * rng_in        # type: ignore[operator]
                  and abs(azi - azi_in) <= SPACING_RELATIVE_TOLERANCE * azi_in)       # type: ignore[operator]
    check("pixel_spacing", spacing_ok, f"output {rng} x {azi} m (range x azimuth), input {rng_in} x {azi_in} m")

    before: list[BandStatistics] = []
    after: list[BandStatistics] = []
    comparisons: list[BandComparison] = []
    crs_seen: Any = None
    transform_seen: Any = None
    readable: dict[str, bool] = {}
    for name, pol in expected.items():
        band, band_in = product.band(name), input_product.band(name)
        if band is None or band.image_file is None or band_in is None or band_in.image_file is None:
            continue
        try:
            st, dtype, crs, transform = band_statistics(
                band.image_file, band=name, polarization=pol, unit=band.unit, nodata=band.nodata_value,
                edge_width_px=border_limit, root=root)
        except Exception as exc:  # any read error becomes a failed check, reported with its type
            readable[name] = False
            check(f"readable_{pol.value}", False, f"{band.image_file.name}: {type(exc).__name__}: {exc}")
            continue
        readable[name] = True
        check(f"readable_{pol.value}", True, f"{band.image_file.name}: {st.width} x {st.height}, {dtype}")
        is_float = bool(np.issubdtype(np.dtype(dtype), np.floating))
        check(f"float_{pol.value}", is_float, f"rasterio dtype {dtype}, DIMAP data type {band.data_type or '?'}")
        after.append(st)
        crs_seen, transform_seen = crs, transform
        st_in, _, _, _ = band_statistics(band_in.image_file, band=name, polarization=pol, unit=band_in.unit,
                                         nodata=band_in.nodata_value, edge_width_px=border_limit, root=root)
        before.append(st_in)
        check(f"valid_pixels_{pol.value}", st.valid_pixels > 0,
              f"{st.valid_pixels:,} of {st.total_pixels:,} pixels finite and not no-data "
              f"(input {st_in.valid_pixels:,})")
        if (st.width, st.height) != (st_in.width, st_in.height) or not is_float:
            continue
        comp = compare_band(band_in.image_file, band.image_file, band=name, nodata_before=band_in.nodata_value,
                            nodata_after=band.nodata_value)
        comparisons.append(comp)
        check(f"filter_changed_{pol.value}", comp.changed_pixels > 0,
              f"{comp.changed_pixels:,} of {comp.compared_pixels:,} valid input pixels changed")
        check(f"no_new_nonfinite_{pol.value}", comp.new_nonfinite_pixels == 0,
              f"{comp.new_nonfinite_pixels:,} pixels valid in the input are NaN/inf after filtering")
        if comp.new_negative_pixels:
            warnings_.append(f"{name}: {comp.new_negative_pixels:,} pixels became negative (reported, not clipped)")
        if band.nodata_value != band_in.nodata_value:
            warnings_.append(f"{name}: no-data value changed from {band_in.nodata_value} to {band.nodata_value}")

    if bands_ok and present and all(readable.get(n, False) for n in expected):
        sizes = {(s.width, s.height) for s in after}
        check("same_grid", sizes == {(product.width, product.height)}
              and (product.width, product.height) == (input_product.width, input_product.height),
              f"bands {sorted(sizes)}, DIMAP {product.width} x {product.height}, "
              f"input {input_product.width} x {input_product.height}")
    if crs_seen is not None:
        warnings_.append(f"rasterio reports a CRS ({crs_seen}) for a radar-geometry band")

    footprint = None
    try:
        footprint = tie_point_footprint(product)
        footprint_in = tie_point_footprint(input_product)
    except DimapError as exc:
        check("geolocation_preserved", False, str(exc))
    else:
        check("geolocation_preserved", _same_footprint(footprint, footprint_in),
              f"tie-point corners {[list(p) for p in footprint.exterior[:4]]} "
              f"(input {[list(p) for p in footprint_in.exterior[:4]]})")

    transform_tuple = tuple(float(v) for v in tuple(transform_seen)[:6]) if transform_seen is not None else None
    return result(width=product.width, height=product.height,
                  geometry="radar (tie-point geocoding)" if radar else "NOT radar geometry",
                  rasterio_crs=str(crs_seen) if crs_seen is not None else None,
                  rasterio_transform=transform_tuple, range_spacing_m=rng, azimuth_spacing_m=azi,
                  footprint=footprint, orbit_file=orbit_out, processing_history=product.history,
                  bands_before=tuple(before), bands_after=tuple(after), comparisons=tuple(comparisons))

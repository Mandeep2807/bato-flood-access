"""Step 3.1 contract: what a downloaded Sentinel-1 GRD SAFE zip contains.

Produced by :func:`bato.preprocessing.safe_reader.inspect_safe_zip` from the
XML annotation files only (no pixels are read). Later preprocessing stages use
it to locate the VV/VH rasters and their calibration/noise tables, and to
check that the polarisations share one image grid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from bato.contracts.common import Contract, OrbitDirection, Polarization


class SafeBand(Contract):
    """One polarisation of a GRD product: its four member files and image geometry."""

    polarization: Polarization
    measurement_member: str            # path inside the zip, e.g. <name>.SAFE/measurement/...-vv-...tiff
    annotation_member: str
    calibration_member: str
    noise_member: str
    measurement_size_bytes: int
    samples: int                       # image width  (numberOfSamples)
    lines: int                         # image height (numberOfLines)
    range_pixel_spacing_m: float
    azimuth_pixel_spacing_m: float
    geolocation_points: int            # geolocationGridPoint count
    calibration_vectors: int           # calibrationVector count
    calibration_points_per_vector: int  # sigmaNought values in the first vector

    @model_validator(mode="after")
    def _check(self) -> SafeBand:
        if self.samples <= 0 or self.lines <= 0:
            raise ValueError("image size must be positive")
        if self.range_pixel_spacing_m <= 0 or self.azimuth_pixel_spacing_m <= 0:
            raise ValueError("pixel spacing must be positive")
        # GRD measurement rasters are uncompressed uint16 digital numbers.
        if self.measurement_size_bytes < self.samples * self.lines * 2:
            raise ValueError("measurement file is smaller than a uint16 raster of the annotated size")
        if self.geolocation_points < 4:
            raise ValueError("too few geolocation grid points")
        if self.calibration_vectors < 2 or self.calibration_points_per_vector < 2:
            raise ValueError("calibration table too small to interpolate")
        return self


class SafeProduct(Contract):
    """A Sentinel-1 GRD SAFE product as found inside its zip archive."""

    archive_name: str                  # zip file name
    product_name: str                  # top folder, e.g. S1D_IW_GRDH_..._F84B.SAFE
    product_type: Literal["GRD"]
    orbit_direction: OrbitDirection    # from the annotation <pass>
    bands: tuple[SafeBand, ...]        # sorted by polarisation

    @model_validator(mode="after")
    def _check(self) -> SafeProduct:
        if not self.bands:
            raise ValueError("a product needs at least one band")
        pols = [b.polarization for b in self.bands]
        if len(set(pols)) != len(pols):
            raise ValueError("duplicate polarisation")
        first = self.bands[0]
        grid = (first.samples, first.lines, first.range_pixel_spacing_m, first.azimuth_pixel_spacing_m)
        for band in self.bands[1:]:
            if (band.samples, band.lines, band.range_pixel_spacing_m, band.azimuth_pixel_spacing_m) != grid:
                raise ValueError(f"{band.polarization.value} is not on the same image grid as "
                                 f"{first.polarization.value}")
        return self

    def band(self, polarization: Polarization | str) -> SafeBand:
        pol = Polarization(polarization)
        for b in self.bands:
            if b.polarization is pol:
                return b
        raise KeyError(f"{self.product_name} has no {pol.value} band")

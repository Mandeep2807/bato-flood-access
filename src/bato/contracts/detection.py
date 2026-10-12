"""Contract D - detection raster.

Pixels live in a GeoTIFF on disk. This contract describes that file and fixes
the meaning of every class code, so detection, impact, evaluation and the
dashboard all read the raster the same way.

Band 1 is always ``impact_class`` (uint8, codes from :class:`ImpactClass`).
Further bands carry continuous values, e.g. ``flood_probability`` or
``debris_zscore``.
"""

from __future__ import annotations

import math
import re
from enum import Enum, IntEnum
from typing import Literal

import numpy as np
from pydantic import model_validator

from bato.config import ConfidenceThresholds
from bato.contracts.common import Contract, RelativePath, Sha256Hex, UTCDatetime


class ImpactClass(IntEnum):
    """Per-pixel class codes in band 1 of every detection raster."""

    NO_CHANGE = 0
    FLOOD_WATER = 1
    # Radar change consistent with debris or ground disturbance. Produced by
    # change statistics, NOT by a model trained on debris labels (none of the
    # permitted training datasets contains a debris class).
    DEBRIS_DISTURBANCE = 2
    PERMANENT_WATER = 3
    # Radar shadow/layover, or no valid pixel in the pair: the satellite could
    # not observe this pixel, so no claim is made about it either way.
    UNOBSERVABLE = 254
    OUTSIDE_AOI = 255  # nodata


VALID_CLASS_CODES = frozenset(int(c) for c in ImpactClass)
IMPACT_CLASS_BAND = "impact_class"


class DetectionMethod(str, Enum):
    BASELINE_CHANGE = "baseline_change"  # thresholded change detection, no learning
    FLOOD_MODEL = "flood_model"          # trained segmentation model
    FUSION = "fusion"                    # model + change rules (+ optional Sentinel-2)


class ConfidenceTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RasterBand(Contract):
    index: int                                  # 1-based, as in GDAL/rasterio
    name: str
    dtype: Literal["uint8", "float32"]
    description: str
    nodata: float | None = None


class DetectionRaster(Contract):
    """Metadata for one detection GeoTIFF."""

    raster_id: str
    aoi_id: str
    pair_ids: tuple[str, ...]                   # one per orbit track that contributed
    path: RelativePath
    sha256: Sha256Hex | None = None
    crs: str                                    # e.g. "EPSG:32645"
    transform: tuple[float, float, float, float, float, float]  # affine a, b, c, d, e, f
    width: int
    height: int
    bands: tuple[RasterBand, ...]
    method: DetectionMethod
    model_version: str | None = None
    config_fingerprint: Sha256Hex
    created_at: UTCDatetime

    @model_validator(mode="after")
    def _check(self) -> DetectionRaster:
        if not self.pair_ids:
            raise ValueError("a detection raster must name the pre/post pair(s) it was derived from")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("duplicate pair_ids")
        if not re.fullmatch(r"EPSG:\d{4,6}", self.crs):
            raise ValueError("crs must be an EPSG code such as 'EPSG:32645'")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        a, _, _, _, e, _ = self.transform
        if not all(math.isfinite(v) for v in self.transform) or a == 0 or e == 0:
            raise ValueError("transform must be finite with non-zero pixel sizes")
        if [b.index for b in self.bands] != list(range(1, len(self.bands) + 1)):
            raise ValueError("band indices must be 1..n in order")
        if len({b.name for b in self.bands}) != len(self.bands):
            raise ValueError("band names must be unique")
        first = self.bands[0] if self.bands else None
        if first is None or first.name != IMPACT_CLASS_BAND or first.dtype != "uint8":
            raise ValueError(f"band 1 must be '{IMPACT_CLASS_BAND}' with dtype uint8")
        if first.nodata != ImpactClass.OUTSIDE_AOI:
            raise ValueError("band 1 nodata must be ImpactClass.OUTSIDE_AOI (255)")
        if self.method is not DetectionMethod.BASELINE_CHANGE and not self.model_version:
            raise ValueError(f"method '{self.method.value}' requires model_version")
        return self

    @property
    def pixel_size(self) -> tuple[float, float]:
        """(x size, y size) in CRS units; y is usually negative (north-up)."""
        return (self.transform[0], self.transform[4])


def validate_class_array(classes: np.ndarray) -> None:
    """Check that an ``impact_class`` array uses only the defined codes.

    Raises:
        ValueError: wrong dtype, wrong dimensionality, or unknown class codes.
    """
    if classes.dtype != np.uint8:
        raise ValueError(f"impact_class array must be uint8, got {classes.dtype}")
    if classes.ndim != 2:
        raise ValueError(f"impact_class array must be 2-D, got {classes.ndim}-D")
    unknown = set(np.unique(classes).tolist()) - VALID_CLASS_CODES
    if unknown:
        raise ValueError(f"unknown impact class codes: {sorted(unknown)}")


def confidence_tier(probability: float, thresholds: ConfidenceThresholds) -> ConfidenceTier:
    """Map a calibrated probability to a reporting tier using configured cut-offs."""
    if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability >= thresholds.tier_high_min.value:
        return ConfidenceTier.HIGH
    if probability >= thresholds.tier_medium_min.value:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


__all__ = [
    "ConfidenceTier", "DetectionMethod", "DetectionRaster", "IMPACT_CLASS_BAND", "ImpactClass",
    "RasterBand", "VALID_CLASS_CODES", "confidence_tier", "validate_class_array",
]

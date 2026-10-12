"""Contracts for the deterministic SAR disturbance (debris) detector (bato.debris.disturbance).

State raster (``debris_state.tif``, uint8, every pixel has exactly one state):
    0  NO_DISTURBANCE          geometry-observable; no qualifying brightening (includes steep terrain)
    1  DISTURBANCE             brightening in VV AND VH vs BOTH pre-event images, stable pre-event pair,
                               not FLOOD / PRE_EVENT_WATER, in an 8-connected cluster >= min_component_pixels
    10 NOT_ASSESSED_INVALID_SAR
    11 NOT_ASSESSED_NO_DEM
    12 NOT_ASSESSED_LAYOVER
    13 NOT_ASSESSED_SHADOW     local LIA shadow or cast shadow
Codes 10-13 are taken unchanged from the Step 4C state raster (same gates, same thresholds). Step 4C's
STEEP gate (code 14) is a water-ponding plausibility gate, not a visibility gate: steep pixels are assessed
here (same track on all dates, so foreshortening is identical and cancels in the differences).

Nothing here is validated: z and the cluster size are the existing provisional thresholds, and no
reference map was used to set or check them.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import model_validator

from bato.contracts.baseline import ThresholdRecord
from bato.contracts.common import Contract, RelativePath, Sha256Hex, UTCDatetime
from bato.contracts.model_input import SourceStackRecord

DEBRIS_VERSION: Literal["bato-debris-disturbance-v1"] = "bato-debris-disturbance-v1"


class DebrisState(IntEnum):
    NO_DISTURBANCE = 0
    DISTURBANCE = 1
    NOT_ASSESSED_INVALID_SAR = 10
    NOT_ASSESSED_NO_DEM = 11
    NOT_ASSESSED_LAYOVER = 12
    NOT_ASSESSED_SHADOW = 13


DEBRIS_ASSESSED = (DebrisState.NO_DISTURBANCE, DebrisState.DISTURBANCE)
DEBRIS_NOT_ASSESSED = (DebrisState.NOT_ASSESSED_INVALID_SAR, DebrisState.NOT_ASSESSED_NO_DEM,
                       DebrisState.NOT_ASSESSED_LAYOVER, DebrisState.NOT_ASSESSED_SHADOW)
DEBRIS_STATE_CODES = {str(int(s)): s.name for s in DebrisState}


class NoiseModel(Contract):
    """Pre-event null model: robust spread of PRE2 - PRE1 (dB) over valid, geometry-observable pixels."""

    polarization: Literal["VV", "VH"]
    median_d0_db: float
    sigma_db: float                    # 1.4826 * MAD(d0)
    pixels: int

    @model_validator(mode="after")
    def _check(self) -> NoiseModel:
        if not self.sigma_db > 0 or self.pixels <= 0:
            raise ValueError("sigma must be > 0 and estimated from > 0 pixels")
        return self


class DebrisComponents(Contract):
    candidate_pixels: int
    candidate_components: int
    kept_components: int
    removed_pixels: int
    min_component_pixels: int
    largest_component_px: int


class DebrisStatistics(Contract):
    total_pixels: int
    state_counts: dict[str, int]
    assessed_pixels: int
    disturbance_pixels: int
    disturbance_fraction_of_assessed: float | None
    steep_assessed_pixels: int                 # Step 4C STEEP pixels assessed here
    darkening_diagnostic_pixels: int           # both-pol darkening (diagnostic only, never a claim)
    disturbance_by_slope_deg: dict[str, int]
    disturbance_by_geometry: dict[str, int]
    facing_sensor_share: float | None          # pre-registered criterion (c): <= 0.5
    edge_pixels_share: float | None            # disturbance share within the image border band
    assessed_edge_share: float | None          # assessed share within the same band (criterion (d) compares)
    change_db_quantiles: dict[str, dict[str, float]]
    components: DebrisComponents

    @model_validator(mode="after")
    def _check(self) -> DebrisStatistics:
        if set(self.state_counts) != {s.name for s in DebrisState}:
            raise ValueError("state_counts must list every DebrisState")
        if sum(self.state_counts.values()) != self.total_pixels:
            raise ValueError("state counts must sum to the total pixels")
        if self.assessed_pixels != sum(self.state_counts[s.name] for s in DEBRIS_ASSESSED):
            raise ValueError("assessed_pixels must be NO_DISTURBANCE + DISTURBANCE")
        if self.disturbance_pixels != self.state_counts["DISTURBANCE"]:
            raise ValueError("disturbance_pixels must match state_counts")
        if self.components.candidate_pixels - self.components.removed_pixels != self.disturbance_pixels:
            raise ValueError("disturbance pixels must equal candidates minus cluster-filtered pixels")
        if sum(self.disturbance_by_slope_deg.values()) != self.disturbance_pixels:
            raise ValueError("slope bins must account for every disturbance pixel")
        return self


class DebrisOutputs(Contract):
    directory: RelativePath
    state: RelativePath
    state_sha256: Sha256Hex
    change_table: RelativePath                 # .npz: every DISTURBANCE and Step 4C FLOOD pixel with d1/d2/d0
    change_table_sha256: Sha256Hex
    diagnostics: RelativePath


class BaselineRef(Contract):
    provenance: RelativePath
    provenance_sha256: Sha256Hex
    state: RelativePath
    state_sha256: Sha256Hex
    config_fingerprint: str


class DebrisProvenance(Contract):
    kind: Literal["sar_disturbance_detection"] = "sar_disturbance_detection"
    version: Literal["bato-debris-disturbance-v1"] = DEBRIS_VERSION
    method: Literal["change_statistics"] = "change_statistics"
    bato_version: str
    config_fingerprint: str
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    candidate_id: str
    event_date: str
    source: SourceStackRecord
    baseline: BaselineRef                      # Step 4C run supplying the geometry gates and FLOOD / water
    dem: RelativePath | None
    dem_sha256: Sha256Hex | None
    orbit_direction: str
    look_azimuth_deg: float
    z: float
    noise: tuple[NoiseModel, NoiseModel]       # VV, VH
    thresholds: tuple[ThresholdRecord, ...]
    rule: str
    state_codes: dict[str, str] = DEBRIS_STATE_CODES
    outputs: DebrisOutputs
    statistics: DebrisStatistics
    inputs_not_used: tuple[str, ...] = ("EMSR / UNOSAT / any damage map", "post-event OSM", "population data",
                                        "FloodViT outputs", "any trained model")
    validated: Literal[False] = False
    validation_note: str = ("NOT VALIDATED: provisional thresholds (detection.debris_z_min, "
                            "detection.min_component_pixels), not tuned or evaluated against any reference")
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check(self) -> DebrisProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if [n.polarization for n in self.noise] != ["VV", "VH"]:
            raise ValueError("noise models must be VV then VH")
        if self.state_codes != DEBRIS_STATE_CODES:
            raise ValueError("state codes are fixed")
        return self

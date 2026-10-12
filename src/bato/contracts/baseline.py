"""Step 4C contracts: the deterministic SAR change flood baseline (bato.detection.baseline).

State raster (``baseline_state.tif``, uint8, every pixel has exactly one state):
    0 NO_FLOOD                assessed; not flood by the rule (incl. candidates removed by the cluster filter)
    1 FLOOD                   assessed; new open-water darkening vs BOTH pre-event images, cluster kept
    2 PRE_EVENT_WATER         assessed; open water in POST that was already open-water-dark in PRE1 or PRE2
    10 NOT_ASSESSED_INVALID_SAR  stack validity mask 0 (SAR no-data in any of the six layers)
    11 NOT_ASSESSED_NO_DEM    no GLO-30 terrain at the pixel (cannot gate terrain)
    12 NOT_ASSESSED_LAYOVER   approximate local incidence angle <= visibility.layover_lia_max_deg
    13 NOT_ASSESSED_SHADOW    approximate local incidence angle >= baseline_flood.shadow_lia_min_deg, OR cast
                              shadow from terrain toward the sensor (ray test on GLO-30)
    14 NOT_ASSESSED_STEEP     GLO-30 slope >= baseline_flood.max_slope_deg
Flood raster (``baseline_flood.tif``, uint8): 1 FLOOD, 0 NO_FLOOD or PRE_EVENT_WATER, 255 NOT_ASSESSED (nodata).

NOT_ASSESSED is never NO_FLOOD: terrain or data that cannot be observed reliably makes no claim either way.
Nothing here is validated: thresholds are provisional and no reference map was used to set them.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import model_validator

from bato.contracts.common import Contract, RelativePath, Sha256Hex, UTCDatetime
from bato.contracts.model_input import SourceStackRecord

BASELINE_VERSION: Literal["bato-baseline-change-v1"] = "bato-baseline-change-v1"
FLOOD_NODATA = 255


class BaselineState(IntEnum):
    NO_FLOOD = 0
    FLOOD = 1
    PRE_EVENT_WATER = 2
    NOT_ASSESSED_INVALID_SAR = 10
    NOT_ASSESSED_NO_DEM = 11
    NOT_ASSESSED_LAYOVER = 12
    NOT_ASSESSED_SHADOW = 13
    NOT_ASSESSED_STEEP = 14


ASSESSED_STATES = (BaselineState.NO_FLOOD, BaselineState.FLOOD, BaselineState.PRE_EVENT_WATER)
NOT_ASSESSED_STATES = (BaselineState.NOT_ASSESSED_INVALID_SAR, BaselineState.NOT_ASSESSED_NO_DEM,
                       BaselineState.NOT_ASSESSED_LAYOVER, BaselineState.NOT_ASSESSED_SHADOW,
                       BaselineState.NOT_ASSESSED_STEEP)
STATE_CODES = {str(int(s)): s.name for s in BaselineState}
FLOOD_CODES = {"0": "NO_FLOOD or PRE_EVENT_WATER (assessed)", "1": "FLOOD", "255": "NOT_ASSESSED (nodata)"}


class ThresholdRecord(Contract):
    name: str                                     # dotted name in thresholds.yaml
    value: float
    unit: str
    basis: str
    provisional: bool
    rationale: str


class ComponentStats(Contract):
    candidate_pixels: int                         # passed every per-pixel test, before the cluster filter
    candidate_components: int
    kept_components: int
    removed_components: int
    removed_pixels: int
    removed_pixel_share: float | None             # of candidate pixels
    min_component_pixels: int
    largest_component_px: int
    median_kept_component_px: float | None


class BaselineStatistics(Contract):
    """Denominators: 'assessed' = FLOOD + NO_FLOOD + PRE_EVENT_WATER; 'all' = every stack pixel."""

    total_pixels: int
    assessed_pixels: int
    not_assessed_pixels: int
    state_counts: dict[str, int]                  # BaselineState name -> pixels
    flood_pixels: int
    no_flood_pixels: int
    pre_event_water_pixels: int
    flood_fraction_of_assessed: float | None
    flood_fraction_of_all: float
    not_assessed_fraction_of_all: float
    shadow_cast_only_pixels: int                  # NOT_ASSESSED_SHADOW from cast shadow alone (flat shaded floors)
    flood_by_slope_deg: dict[str, int]
    flood_by_geometry: dict[str, int]
    flood_darkened_vs_pre2: int                   # flood pixels with POST - PRE2 <= threshold in VV and VH
    flood_darkened_vs_pre1: int
    flood_change_db_quantiles: dict[str, dict[str, float]]
    components: ComponentStats
    osm_waterway_overlap: str = ("NOT AVAILABLE: pre-event OSM extraction (bato.osm) is not implemented; no other "
                                 "waterway source is used")

    @model_validator(mode="after")
    def _check(self) -> BaselineStatistics:
        if set(self.state_counts) != {s.name for s in BaselineState}:
            raise ValueError("state_counts must list every BaselineState")
        if sum(self.state_counts.values()) != self.total_pixels:
            raise ValueError("state counts must sum to the total pixels")
        if self.assessed_pixels != sum(self.state_counts[s.name] for s in ASSESSED_STATES):
            raise ValueError("assessed_pixels must be FLOOD + NO_FLOOD + PRE_EVENT_WATER")
        if self.assessed_pixels + self.not_assessed_pixels != self.total_pixels:
            raise ValueError("assessed + not assessed must equal total pixels")
        if (self.flood_pixels, self.no_flood_pixels, self.pre_event_water_pixels) != (
                self.state_counts["FLOOD"], self.state_counts["NO_FLOOD"], self.state_counts["PRE_EVENT_WATER"]):
            raise ValueError("flood / no-flood / pre-event-water counts must match state_counts")
        if sum(self.flood_by_slope_deg.values()) != self.flood_pixels:
            raise ValueError("flood_by_slope_deg must account for every flood pixel")
        if sum(self.flood_by_geometry.values()) != self.flood_pixels:
            raise ValueError("flood_by_geometry must account for every flood pixel")
        expect_all = round(self.flood_pixels / self.total_pixels, 8) if self.total_pixels else 0.0
        expect_assessed = round(self.flood_pixels / self.assessed_pixels, 8) if self.assessed_pixels else None
        if self.flood_fraction_of_all != expect_all or self.flood_fraction_of_assessed != expect_assessed:
            raise ValueError("flood fractions must use the documented denominators")
        if self.components.candidate_pixels - self.components.removed_pixels != self.flood_pixels:
            raise ValueError("flood pixels must equal candidates minus cluster-filtered pixels")
        return self


class FloodViTComparison(Contract):
    """Separate comparison only; FloodViT is never an input of the baseline."""

    floodvit_prediction: RelativePath
    floodvit_prediction_sha256: Sha256Hex
    note: str = "FloodViT is a preliminary, unvalidated layer; agreement is not evidence of correctness"
    both: int
    baseline_only: int
    floodvit_only: int
    jaccard: float | None
    floodvit_flood_in_not_assessed: int           # FloodViT flood where the baseline makes no claim


class BaselineOutputs(Contract):
    directory: RelativePath
    flood: RelativePath
    flood_sha256: Sha256Hex
    state: RelativePath
    state_sha256: Sha256Hex
    diagnostics: RelativePath


class BaselineProvenance(Contract):
    kind: Literal["sar_change_flood_baseline"] = "sar_change_flood_baseline"
    version: Literal["bato-baseline-change-v1"] = BASELINE_VERSION
    method: Literal["baseline_change"] = "baseline_change"
    bato_version: str
    config_fingerprint: str
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    candidate_id: str
    event_date: str
    source: SourceStackRecord
    dem: RelativePath | None
    dem_sha256: Sha256Hex | None
    orbit_direction: str
    look_azimuth_deg: float
    thresholds: tuple[ThresholdRecord, ...]
    rule: str
    state_codes: dict[str, str] = STATE_CODES
    flood_codes: dict[str, str] = FLOOD_CODES
    outputs: BaselineOutputs
    statistics: BaselineStatistics
    floodvit_comparison: FloodViTComparison | None
    inputs_not_used: tuple[str, ...] = ("FloodViT outputs (comparison only)", "EMSR / UNOSAT / any damage map",
                                        "post-event OSM", "population data", "JRC permanent water")
    validated: Literal[False] = False
    validation_note: str = ("NOT VALIDATED: provisional thresholds, not tuned or evaluated against any reference; "
                            "a transparent rule, not a measured accuracy")
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check(self) -> BaselineProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.state_codes != STATE_CODES or self.flood_codes != FLOOD_CODES:
            raise ValueError("state and flood codes are fixed")
        return self

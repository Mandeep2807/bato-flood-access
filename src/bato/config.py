"""Centralised configuration loading and validation.

Two files are read from the config directory:

* ``config.yaml``     - operational settings (paths, AOI limits, search windows).
* ``thresholds.yaml`` - algorithmic decision thresholds, each with a
  ``provisional`` flag, unit and rationale.

Both are validated strictly: unknown keys, wrong types and inconsistent values
raise :class:`~bato.exceptions.ConfigError`. Modules must take thresholds from
here instead of hard-coding numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bato.exceptions import ConfigError

# Source-tree location (<repo>/config). Only meaningful for a source checkout
# or editable install; a regular install places this file in site-packages.
SOURCE_TREE_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_DIR_ENV = "BATO_CONFIG_DIR"
SETTINGS_FILE = "config.yaml"
THRESHOLDS_FILE = "thresholds.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Operational settings (config.yaml)
# --------------------------------------------------------------------------


class PathSettings(_Strict):
    raw: Path
    interim: Path
    processed: Path
    reference: Path
    eval: Path
    models: Path
    reports: Path
    logs: Path

    def evaluation_only_dirs(self) -> tuple[Path, Path]:
        """Directories whose contents must never become pipeline inputs."""
        return (self.reference, self.eval)


class AOISettings(_Strict):
    min_area_km2: float
    max_area_km2: float
    default_point_half_size_km: float
    analysis_buffer_km: float

    @model_validator(mode="after")
    def _check(self) -> AOISettings:
        if not 0 < self.min_area_km2 < self.max_area_km2:
            raise ValueError("require 0 < min_area_km2 < max_area_km2")
        if self.default_point_half_size_km <= 0 or self.analysis_buffer_km < 0:
            raise ValueError("default_point_half_size_km must be > 0 and analysis_buffer_km >= 0")
        return self


class EventSettings(_Strict):
    min_event_date: date
    same_day_policy: Literal["exclude"]


class Sentinel1Settings(_Strict):
    acquisition_mode: Literal["IW"]
    product_type: Literal["GRD"]
    polarizations: tuple[Literal["VV", "VH", "HH", "HV"], ...]
    pre_window_days: int
    post_window_days: int
    min_pre_images: int
    max_pre_images: int
    max_post_delay_days: int
    min_aoi_coverage_fraction: float
    grd_resolution_class: Literal["H", "M"]
    pass_max_gap_s: float
    coverage_grid_size: int
    # A design principle, not a tunable: the only accepted value is True.
    require_same_relative_orbit: Literal[True]

    @model_validator(mode="after")
    def _check(self) -> Sentinel1Settings:
        if not self.polarizations:
            raise ValueError("at least one polarization is required")
        if self.pre_window_days <= 0 or self.post_window_days <= 0:
            raise ValueError("search windows must be positive")
        if not 1 <= self.min_pre_images <= self.max_pre_images:
            raise ValueError("require 1 <= min_pre_images <= max_pre_images")
        if self.max_post_delay_days <= 0:
            raise ValueError("max_post_delay_days must be positive")
        if not 0 < self.min_aoi_coverage_fraction <= 1:
            raise ValueError("min_aoi_coverage_fraction must be in (0, 1]")
        if self.pass_max_gap_s <= 0:
            raise ValueError("pass_max_gap_s must be positive")
        if not 20 <= self.coverage_grid_size <= 2000:
            raise ValueError("coverage_grid_size must be in [20, 2000]")
        return self


class PairingWeights(_Strict):
    post_gap: float
    pre_recency: float
    baseline_separation: float
    coverage: float
    quality: float


class PairingPenalties(_Strict):
    long_post_delay: float
    multi_slice_pass: float
    metadata_from_product_name: float
    mixed_platforms: float


class PairingSettings(_Strict):
    """PAIRING HEURISTIC parameters (Step 2B). All provisional."""

    heuristic_status: Literal["provisional"]
    min_pre_separation_days: float
    target_pre_separation_days: float
    max_candidates_per_track: int
    allow_single_pre_fallback: bool
    weights: PairingWeights
    penalties: PairingPenalties

    @model_validator(mode="after")
    def _check(self) -> PairingSettings:
        if self.min_pre_separation_days < 0:
            raise ValueError("min_pre_separation_days must be >= 0")
        if self.target_pre_separation_days <= 0 or self.target_pre_separation_days < self.min_pre_separation_days:
            raise ValueError("target_pre_separation_days must be > 0 and >= min_pre_separation_days")
        if self.max_candidates_per_track < 1:
            raise ValueError("max_candidates_per_track must be >= 1")
        weights = self.weights.model_dump()
        if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("pairing weights must be >= 0 and sum to 1")
        if any(not 0.0 <= p <= 1.0 for p in self.penalties.model_dump().values()):
            raise ValueError("pairing penalties must be in [0, 1]")
        return self


class CDSESettings(_Strict):
    odata_url: str
    token_url: str
    client_id: str
    page_size: int
    max_records: int
    timeout_s: float

    @model_validator(mode="after")
    def _check(self) -> CDSESettings:
        for name in ("odata_url", "token_url"):
            if not getattr(self, name).startswith("https://"):
                raise ValueError(f"{name} must be an https URL")
        if not 1 <= self.page_size <= 1000:
            raise ValueError("page_size must be in [1, 1000]")
        if self.max_records < self.page_size:
            raise ValueError("max_records must be >= page_size")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        return self


class Sentinel2Settings(_Strict):
    max_scene_cloud_pct: float
    post_window_days: int
    pre_composite_days: int

    @model_validator(mode="after")
    def _check(self) -> Sentinel2Settings:
        if not 0 <= self.max_scene_cloud_pct <= 100:
            raise ValueError("max_scene_cloud_pct must be in [0, 100]")
        if self.post_window_days <= 0 or self.pre_composite_days <= 0:
            raise ValueError("windows must be positive")
        return self


class DEMSettings(_Strict):
    instance: Literal["COPERNICUS_30", "COPERNICUS_90"]
    cdse_dataset: str
    buffer_km: float

    @model_validator(mode="after")
    def _check(self) -> DEMSettings:
        if not self.cdse_dataset.startswith("COP-DEM_"):
            raise ValueError("dem.cdse_dataset must name a Copernicus DEM dataset (COP-DEM_...)")
        if self.buffer_km < 0:
            raise ValueError("dem.buffer_km must be >= 0")
        return self


class DownloadSettings(_Strict):
    base_url: str
    allowed_redirect_suffixes: tuple[str, ...]
    timeout_s: float
    chunk_bytes: int
    max_attempts: int
    backoff_initial_s: float
    backoff_max_s: float
    min_free_disk_ratio: float
    verify_existing: bool
    max_scenes_per_run: int
    s1_subdir: str
    dem_subdir: str
    manifest_subdir: str

    @model_validator(mode="after")
    def _check(self) -> DownloadSettings:
        if not self.base_url.startswith("https://"):
            raise ValueError("download.base_url must be an https URL")
        suffixes = self.allowed_redirect_suffixes
        if not suffixes or any(not s or s.startswith(".") for s in suffixes):
            raise ValueError("download.allowed_redirect_suffixes must be non-empty host suffixes")
        if self.timeout_s <= 0 or self.chunk_bytes < 1024 or self.max_attempts < 1:
            raise ValueError("invalid download timeout_s / chunk_bytes / max_attempts")
        if not 0 <= self.backoff_initial_s <= self.backoff_max_s:
            raise ValueError("require 0 <= backoff_initial_s <= backoff_max_s")
        if self.min_free_disk_ratio < 1.0 or self.max_scenes_per_run < 1:
            raise ValueError("min_free_disk_ratio must be >= 1 and max_scenes_per_run >= 1")
        for name in ("s1_subdir", "dem_subdir", "manifest_subdir"):
            value = getattr(self, name)
            if not value or "/" in value or "\\" in value or value in (".", ".."):
                raise ValueError(f"download.{name} must be a plain folder name")
        return self


class OSMSettings(_Strict):
    api_base_url: str
    api_key_env: str = "OHSOME_API_KEY"
    snapshot_offset_days: int
    timeout_s: float = 600.0
    output_subdir: str = "osm"
    road_highway_values: tuple[str, ...]
    settlement_place_values: tuple[str, ...]
    town_place_values: tuple[str, ...]
    building_margin_km: float = 2.0

    @model_validator(mode="after")
    def _check(self) -> OSMSettings:
        if self.snapshot_offset_days < 1:
            raise ValueError("snapshot_offset_days must be >= 1 (no same-day OSM edits)")
        if not self.api_base_url.startswith("https://"):
            raise ValueError("osm.api_base_url must be an https URL")
        if not self.api_key_env or not self.api_key_env.replace("_", "").isalnum() or self.api_key_env[0].isdigit():
            raise ValueError("osm.api_key_env must be an environment variable NAME (never the key itself)")
        if self.timeout_s <= 0 or self.building_margin_km < 0:
            raise ValueError("osm.timeout_s must be > 0 and osm.building_margin_km >= 0")
        for name in ("road_highway_values", "settlement_place_values", "town_place_values"):
            values = getattr(self, name)
            if not values or any(not v or not v.replace("_", "").isalnum() for v in values):
                raise ValueError(f"osm.{name} must be a non-empty list of plain OSM tag values")
        if not set(self.town_place_values) <= set(self.settlement_place_values):
            raise ValueError("osm.town_place_values must be a subset of osm.settlement_place_values")
        return self


class ConnectivitySettings(_Strict):
    output_subdir: str = "connectivity"


class ModelSettings(_Strict):
    flood_model_path: Path | None = None
    flood_model_version: str | None = None


class LoggingSettings(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    jsonl_file: Path | None = None


class SnapSettings(_Strict):
    """SNAP gpt location and limits (machine-specific; not fingerprinted)."""

    gpt_path: Path | None = None
    timeout_s: float
    diag_timeout_s: float

    @model_validator(mode="after")
    def _check(self) -> SnapSettings:
        if self.timeout_s <= 0 or self.diag_timeout_s <= 0:
            raise ValueError("snap timeouts must be positive")
        return self


class S1RadiometricSettings(_Strict):
    """Step 3.2 radiometric chain (Kuro Siwo-compatible values; see docs/preprocessing.md)."""

    graph: str
    orbit_type: str
    orbit_poly_degree: int
    orbit_continue_on_fail: bool
    remove_thermal_noise: bool
    reintroduce_thermal_noise: bool
    border_limit: int
    trim_threshold: float
    # Design requirements, not tunables: linear sigma0 only.
    output_sigma0: Literal[True]
    output_in_db: Literal[False]

    @model_validator(mode="after")
    def _check(self) -> S1RadiometricSettings:
        if not self.graph.endswith(".xml") or "/" in self.graph or "\\" in self.graph:
            raise ValueError("graph must be a file name ending in .xml")
        if not 1 <= self.orbit_poly_degree <= 10:
            raise ValueError("orbit_poly_degree must be in [1, 10]")
        if self.border_limit < 0 or self.trim_threshold < 0:
            raise ValueError("border_limit and trim_threshold must be >= 0")
        return self


class S1SpeckleSettings(_Strict):
    """Step 3.3 Lee Sigma speckle filter (Kuro Siwo values; see docs/preprocessing.md).

    Values are SNAP's Speckle-Filter value-set strings. Only Lee Sigma is accepted:
    another filter would silently break Kuro Siwo compatibility.
    """

    graph: str
    filter: Literal["Lee Sigma"]
    num_looks: Literal["1", "2", "3", "4"]
    window_size: Literal["5x5", "7x7", "9x9", "11x11", "13x13", "15x15", "17x17"]
    target_window_size: Literal["3x3", "5x5"]
    sigma: Literal["0.5", "0.6", "0.7", "0.8", "0.9"]

    @model_validator(mode="after")
    def _check(self) -> S1SpeckleSettings:
        if not self.graph.endswith(".xml") or "/" in self.graph or "\\" in self.graph:
            raise ValueError("graph must be a file name ending in .xml")
        if int(self.target_window_size[0]) >= int(self.window_size.split("x")[0]):
            raise ValueError("target_window_size must be smaller than window_size")
        return self


class S1TerrainSettings(_Strict):
    """Step 3.4 Range-Doppler terrain correction (see docs/preprocessing.md).

    Projection, pixel spacing, resampling and grid alignment are fixed in the committed graph
    (Kuro Siwo values); only the external-DEM handling is configured here.
    """

    graph: str
    dem_margin_km: float
    dem_apply_egm: bool
    dem_nodata: float

    @model_validator(mode="after")
    def _check(self) -> S1TerrainSettings:
        if not self.graph.endswith(".xml") or "/" in self.graph or "\\" in self.graph:
            raise ValueError("graph must be a file name ending in .xml")
        if self.dem_margin_km < 0:
            raise ValueError("dem_margin_km must be >= 0")
        if self.dem_nodata == 0.0:
            raise ValueError("dem_nodata must not be 0.0 (a real elevation on coasts)")
        return self


class PreprocessingSettings(_Strict):
    output_subdir: str
    crop_buffer_m: float
    s1_radiometric: S1RadiometricSettings
    s1_speckle: S1SpeckleSettings
    s1_terrain: S1TerrainSettings

    @model_validator(mode="after")
    def _check(self) -> PreprocessingSettings:
        if not self.output_subdir or "/" in self.output_subdir or "\\" in self.output_subdir \
                or self.output_subdir in (".", ".."):
            raise ValueError("preprocessing.output_subdir must be a plain folder name")
        if self.crop_buffer_m < 0:
            raise ValueError("preprocessing.crop_buffer_m must be >= 0")
        return self


class Settings(_Strict):
    paths: PathSettings
    aoi: AOISettings
    event: EventSettings
    sentinel1: Sentinel1Settings
    sentinel2: Sentinel2Settings
    pairing: PairingSettings
    cdse: CDSESettings
    dem: DEMSettings
    download: DownloadSettings
    snap: SnapSettings
    preprocessing: PreprocessingSettings
    osm: OSMSettings
    connectivity: ConnectivitySettings = ConnectivitySettings()
    models: ModelSettings
    logging: LoggingSettings


# --------------------------------------------------------------------------
# Thresholds (thresholds.yaml)
# --------------------------------------------------------------------------


class Param(_Strict):
    """A tunable threshold together with its provenance."""

    value: float
    unit: str
    provisional: bool
    rationale: str
    # Where the value comes from (optional for older entries): physics / literature motivated, inherited from an
    # existing BATO design, or a provisional engineering choice. None of these means "validated".
    basis: Literal["physics_literature", "inherited_bato_design", "provisional_engineering"] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_probability(p: Param) -> bool:
    return 0.0 <= p.value <= 1.0


class DetectionThresholds(_Strict):
    flood_prob_low: Param
    flood_prob_high: Param
    debris_z_min: Param
    min_component_pixels: Param

    @model_validator(mode="after")
    def _check(self) -> DetectionThresholds:
        _require(_is_probability(self.flood_prob_low) and _is_probability(self.flood_prob_high),
                 "flood probabilities must be in [0, 1]")
        _require(self.flood_prob_low.value < self.flood_prob_high.value,
                 "flood_prob_low must be < flood_prob_high")
        _require(self.debris_z_min.value > 0, "debris_z_min must be > 0")
        _require(self.min_component_pixels.value >= 1, "min_component_pixels must be >= 1")
        return self


class VisibilityThresholds(_Strict):
    layover_lia_max_deg: Param

    @model_validator(mode="after")
    def _check(self) -> VisibilityThresholds:
        _require(0 <= self.layover_lia_max_deg.value < 90, "layover_lia_max_deg must be in [0, 90)")
        return self


class RoadImpactThresholds(_Strict):
    segment_length_m: Param
    road_buffer_m: Param
    min_blocked_run_m: Param
    min_unobservable_run_m: Param

    @model_validator(mode="after")
    def _check(self) -> RoadImpactThresholds:
        for name in ("segment_length_m", "min_blocked_run_m", "min_unobservable_run_m"):
            _require(getattr(self, name).value > 0, f"{name} must be > 0")
        _require(self.road_buffer_m.value >= 0, "road_buffer_m must be >= 0")
        _require(self.min_blocked_run_m.value <= self.segment_length_m.value,
                 "min_blocked_run_m cannot exceed segment_length_m")
        return self


class ConnectivityThresholds(_Strict):
    settlement_snap_max_m: Param
    degraded_detour_ratio: Param
    building_cluster_eps_m: Param
    building_cluster_min_buildings: Param

    @model_validator(mode="after")
    def _check(self) -> ConnectivityThresholds:
        _require(self.settlement_snap_max_m.value > 0, "settlement_snap_max_m must be > 0")
        _require(self.degraded_detour_ratio.value > 1.0, "degraded_detour_ratio must be > 1")
        _require(self.building_cluster_eps_m.value > 0, "building_cluster_eps_m must be > 0")
        _require(self.building_cluster_min_buildings.value >= 1, "building_cluster_min_buildings must be >= 1")
        return self


class ConfidenceThresholds(_Strict):
    tier_medium_min: Param
    tier_high_min: Param

    @model_validator(mode="after")
    def _check(self) -> ConfidenceThresholds:
        _require(_is_probability(self.tier_medium_min) and _is_probability(self.tier_high_min),
                 "confidence tiers must be in [0, 1]")
        _require(self.tier_medium_min.value < self.tier_high_min.value,
                 "tier_medium_min must be < tier_high_min")
        return self


class BaselineFloodThresholds(_Strict):
    """Deterministic SAR change baseline (bato.detection.baseline). Values in dB / degrees; none validated."""

    vv_water_db: Param
    vh_water_db: Param
    pre2_change_db: Param
    pre1_change_db: Param
    max_slope_deg: Param
    shadow_lia_min_deg: Param
    nominal_incidence_deg: Param
    shadow_search_m: Param

    @model_validator(mode="after")
    def _check(self) -> BaselineFloodThresholds:
        for name in ("vv_water_db", "vh_water_db"):
            _require(-40.0 <= getattr(self, name).value < 0.0, f"{name} must be in [-40, 0) dB")
        for name in ("pre2_change_db", "pre1_change_db"):
            _require(-20.0 <= getattr(self, name).value < 0.0,
                     f"{name} must be a darkening, in [-20, 0) dB")
        _require(0.0 < self.max_slope_deg.value < 90.0, "max_slope_deg must be in (0, 90)")
        _require(45.0 < self.shadow_lia_min_deg.value <= 90.0, "shadow_lia_min_deg must be in (45, 90]")
        _require(20.0 <= self.nominal_incidence_deg.value <= 50.0, "nominal_incidence_deg must be in [20, 50]")
        _require(0.0 < self.shadow_search_m.value <= 20000.0, "shadow_search_m must be in (0, 20000] m")
        for name in type(self).model_fields:
            _require(getattr(self, name).basis is not None, f"baseline_flood.{name} needs a basis")
        return self


class Thresholds(_Strict):
    detection: DetectionThresholds
    visibility: VisibilityThresholds
    baseline_flood: BaselineFloodThresholds
    road_impact: RoadImpactThresholds
    connectivity: ConnectivityThresholds
    confidence: ConfidenceThresholds

    def provisional_parameters(self) -> list[str]:
        """Dotted names of every threshold still marked provisional."""
        names: list[str] = []
        for group_name in type(self).model_fields:
            group = getattr(self, group_name)
            for param_name in type(group).model_fields:
                if getattr(group, param_name).provisional:
                    names.append(f"{group_name}.{param_name}")
        return names


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class BatoConfig(_Strict):
    settings: Settings
    thresholds: Thresholds
    config_dir: Path

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of everything that can change a result.

        Paths, logging, provider connection settings (catalogue and
        download URLs, page size, timeouts, retries) and the SNAP executable
        location/timeouts are excluded so the same configuration gives the same
        fingerprint on every machine. Recorded in evidence records for
        reproducibility.
        """
        payload = {
            "settings": self.settings.model_dump(mode="json", exclude={"paths": True, "logging": True, "cdse": True,
                                                                         "download": True, "snap": True,
                                                                         "osm": {"timeout_s", "api_base_url",
                                                                                 "api_key_env"}}),
            "thresholds": self.thresholds.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _resolve_paths(raw_settings: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Make relative paths absolute against ``base_dir`` before validation."""
    resolved = dict(raw_settings)
    paths = resolved.get("paths")
    if isinstance(paths, dict):
        resolved["paths"] = {k: str((base_dir / str(v)).resolve()) for k, v in paths.items()}
    log_cfg = resolved.get("logging")
    if isinstance(log_cfg, dict) and log_cfg.get("jsonl_file"):
        resolved["logging"] = {**log_cfg, "jsonl_file": str((base_dir / str(log_cfg["jsonl_file"])).resolve())}
    models = resolved.get("models")
    if isinstance(models, dict) and models.get("flood_model_path"):
        resolved["models"] = {**models, "flood_model_path": str((base_dir / str(models["flood_model_path"])).resolve())}
    return resolved


def resolve_config_dir(config_dir: Path | str | None = None) -> Path:
    """Pick the config directory.

    Order: explicit argument, then ``BATO_CONFIG_DIR``, then ``./config`` in
    the current working directory, then the source tree's ``config/``.
    An explicit or environment choice is used as given (and fails loudly
    later if wrong); the two fallbacks are used only if they contain
    ``config.yaml``.
    """
    chosen = config_dir or os.environ.get(CONFIG_DIR_ENV)
    if chosen:
        return Path(chosen).resolve()
    for candidate in (Path.cwd() / "config", SOURCE_TREE_CONFIG_DIR):
        if (candidate / SETTINGS_FILE).is_file():
            return candidate.resolve()
    raise ConfigError(
        "no configuration found: pass a config directory, set "
        f"{CONFIG_DIR_ENV}, or run from the repository root (which contains config/)"
    )


def load_config(config_dir: Path | str | None = None) -> BatoConfig:
    """Load and validate configuration.

    The directory is chosen by :func:`resolve_config_dir`. Relative paths
    inside ``config.yaml`` are resolved against the parent of the config
    directory (the repository root by default).
    """
    cfg_dir = resolve_config_dir(config_dir)
    settings_raw = _resolve_paths(_read_yaml(cfg_dir / SETTINGS_FILE), base_dir=cfg_dir.parent)
    thresholds_raw = _read_yaml(cfg_dir / THRESHOLDS_FILE)
    try:
        settings = Settings.model_validate(settings_raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings in {cfg_dir / SETTINGS_FILE}:\n{exc}") from exc
    try:
        thresholds = Thresholds.model_validate(thresholds_raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid thresholds in {cfg_dir / THRESHOLDS_FILE}:\n{exc}") from exc
    return BatoConfig(settings=settings, thresholds=thresholds, config_dir=cfg_dir)

"""Configuration loading, threshold loading and strict validation."""

from __future__ import annotations

import pytest

from bato.config import CONFIG_DIR_ENV, load_config
from bato.exceptions import ConfigError

from helpers import REPO_CONFIG_DIR


def test_repo_config_loads(cfg):
    assert cfg.settings.sentinel1.require_same_relative_orbit is True
    assert cfg.settings.dem.instance == "COPERNICUS_30"
    assert cfg.settings.osm.snapshot_offset_days >= 1


def test_relative_paths_resolve_against_repo_root(cfg):
    repo_root = REPO_CONFIG_DIR.parent.resolve()
    assert cfg.settings.paths.raw == repo_root / "data" / "raw"
    assert cfg.settings.paths.reference == repo_root / "data" / "reference"
    assert set(cfg.settings.paths.evaluation_only_dirs()) == {
        repo_root / "data" / "reference", repo_root / "data" / "eval"}


def test_thresholds_load_with_provenance(cfg):
    t = cfg.thresholds
    assert t.detection.flood_prob_low.value < t.detection.flood_prob_high.value
    assert t.road_impact.min_blocked_run_m.unit == "m"
    assert t.connectivity.degraded_detour_ratio.rationale


def test_every_threshold_is_flagged_provisional_until_measured(cfg):
    names = cfg.thresholds.provisional_parameters()
    assert "detection.debris_z_min" in names
    assert len(names) == 23  # all thresholds are provisional (15 foundation + 8 baseline_flood)


def test_fingerprint_is_deterministic(cfg):
    assert cfg.fingerprint() == load_config(REPO_CONFIG_DIR).fingerprint()
    assert len(cfg.fingerprint()) == 64


def test_fingerprint_changes_when_a_threshold_changes(cfg, config_copy):
    def bump(d):
        d["detection"]["debris_z_min"]["value"] = 3.5
    changed = load_config(config_copy("thresholds.yaml", bump))
    assert changed.fingerprint() != cfg.fingerprint()


def test_fingerprint_ignores_machine_specific_paths(cfg, config_copy):
    assert load_config(config_copy()).fingerprint() == cfg.fingerprint()


def test_env_var_selects_config_dir(monkeypatch, config_copy):
    target = config_copy()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(target))
    assert load_config().config_dir == target.resolve()


def test_cwd_config_is_found_without_arguments(monkeypatch, config_copy):
    target = config_copy()
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.chdir(target.parent)
    assert load_config().config_dir == target.resolve()


def test_falls_back_to_source_tree_config(monkeypatch, tmp_path):
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # no config/ here
    assert load_config().config_dir == REPO_CONFIG_DIR.resolve()


def test_no_config_anywhere_raises(monkeypatch, tmp_path):
    import bato.config as config_module
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "SOURCE_TREE_CONFIG_DIR", tmp_path / "missing")
    with pytest.raises(ConfigError, match="no configuration found"):
        load_config()


def test_missing_config_dir_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nowhere")


def test_invalid_yaml_raises(config_copy):
    target = config_copy()
    (target / "config.yaml").write_text("paths: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(target)


@pytest.mark.parametrize(("filename", "mutate", "message"), [
    ("config.yaml", lambda d: d["aoi"].update(unexpected_key=1), "Extra inputs"),
    ("config.yaml", lambda d: d["aoi"].update(min_area_km2=5000), "min_area_km2"),
    ("config.yaml", lambda d: d["sentinel1"].update(require_same_relative_orbit=False),
     "require_same_relative_orbit"),
    ("config.yaml", lambda d: d["osm"].update(snapshot_offset_days=0), "snapshot_offset_days"),
    ("config.yaml", lambda d: d["event"].update(same_day_policy="include"), "same_day_policy"),
    ("thresholds.yaml", lambda d: d["detection"]["flood_prob_low"].update(value=0.9), "flood_prob_low"),
    ("thresholds.yaml", lambda d: d["detection"].pop("debris_z_min"), "debris_z_min"),
    ("thresholds.yaml", lambda d: d["confidence"]["tier_high_min"].update(value=1.5), "confidence tiers"),
    ("thresholds.yaml", lambda d: d["road_impact"]["min_blocked_run_m"].update(value=500), "min_blocked_run_m"),
    ("thresholds.yaml", lambda d: d["connectivity"]["degraded_detour_ratio"].update(value=1.0),
     "degraded_detour_ratio"),
])
def test_invalid_values_are_rejected(config_copy, filename, mutate, message):
    with pytest.raises(ConfigError, match=message):
        load_config(config_copy(filename, mutate))

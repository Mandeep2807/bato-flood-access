"""Step 4B forensics: control-mode construction, terrain features, approximate radar geometry and the
diagnostics report on FloodViT outputs made with a FAKE pixel-local model (TEST FIXTURE, no torch)."""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest

from bato.inference import diagnostics as dg
from bato.inference import floodvit as fv
from bato.preprocessing import model_input as mi
from bato.preprocessing import snap
from bato.preprocessing.terrain_validation import lonlat_to_3857

from inference_fakes import (
    BRIGHT_BOX,
    FLOOD_BOX,
    WATER_BOX,
    candidate_id,
    change_model,
    run_mode,
    synthetic_project,
)

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
T = 224
MEAN = np.array([0.0953, 0.0264] * 3, dtype=np.float32)[:, None, None]
STD = np.array([0.0427, 0.0215] * 3, dtype=np.float32)[:, None, None]
CODES = {"PRE1_VV": 0.011, "PRE1_VH": 0.012, "PRE2_VV": 0.021, "PRE2_VH": 0.022, "POST_VV": 0.031, "POST_VH": 0.032}


def fed(mode: str) -> list[str]:
    """Which stack band ends up in each model channel (recovered through the full normalisation path)."""
    raw = np.stack([np.full((T, T), CODES[n], dtype=np.float32) for n in mi.STACK_BANDS])
    x = fv.model_input(raw, np.ones((T, T), dtype=np.uint8), mi.kuro_siwo_spec(), mode)
    return [min(CODES, key=lambda n: abs(CODES[n] - v)) for v in (x * STD + MEAN)[:, 0, 0]]


# ---- control modes ------------------------------------------------------------------------------------------


def test_no_event_construction_is_exactly_post_from_pre2_and_both_pres_from_pre1():
    assert fed("no_event") == ["PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH", "PRE1_VV", "PRE1_VH"]
    assert fv.mapping_lines(mi.kuro_siwo_spec(), "no_event") == (
        "0 POST_VV <- stack PRE2_VV", "1 POST_VH <- stack PRE2_VH", "2 PRE2_VV <- stack PRE1_VV",
        "3 PRE2_VH <- stack PRE1_VH", "4 PRE1_VV <- stack PRE1_VV", "5 PRE1_VH <- stack PRE1_VH")


def test_zero_change_and_static_post_controls():
    assert fed("event") == ["POST_VV", "POST_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"]
    assert fed("zero_change") == ["PRE2_VV", "PRE2_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"]
    assert fed("static_post") == ["POST_VV", "POST_VH"] * 3
    assert set(fv.OUTPUT_SUBDIR) == {"event", *fv.CONTROL_MODES}


# ---- terrain features ---------------------------------------------------------------------------------------


def test_running_min_matches_brute_force():
    a = np.random.default_rng(0).normal(size=(40, 50))
    for axis, radius in ((0, 3), (1, 5), (1, 0)):
        got = dg.running_min(a, radius, axis)
        want = np.empty_like(a)
        n = a.shape[axis]
        for i in range(n):
            sl = [slice(None)] * 2
            sl[axis] = slice(max(0, i - radius), min(n, i + radius + 1))
            idx = [slice(None)] * 2
            idx[axis] = i
            want[tuple(idx)] = a[tuple(sl)].min(axis=axis)
        assert np.array_equal(got, want)


def geo_transform(lat0: float = 28.2, res: float = 1 / 3600):
    return (res, 0.0, 85.2, 0.0, -res, lat0)


def test_slope_aspect_of_planes():
    tr = geo_transform()
    rows, cols = np.mgrid[0:60, 0:60]
    lat = tr[5] + (rows + 0.5) * tr[4]
    east_m = cols * tr[0] * dg.M_PER_DEG * np.cos(np.radians(lat))
    north_m = -(rows * -tr[4]) * dg.M_PER_DEG
    rising_east = dg.dem_features(np.tan(np.radians(20.0)) * east_m, tr, -32767.0)
    assert np.allclose(rising_east.slope_deg[5:-5, 5:-5], 20.0, atol=0.2)
    assert np.allclose(rising_east.aspect_deg[5:-5, 5:-5], 270.0, atol=0.5)       # faces (drops toward) west
    rising_north = dg.dem_features(np.tan(np.radians(35.0)) * north_m, tr, None)
    assert np.allclose(rising_north.slope_deg[5:-5, 5:-5], 35.0, atol=0.2)
    assert np.allclose(rising_north.aspect_deg[5:-5, 5:-5], 180.0, atol=0.5)      # faces south


def test_relative_elevation_is_height_above_the_local_minimum_and_nodata_is_nan():
    tr = geo_transform()
    cols = np.arange(200)
    dem = np.tile(np.abs(cols - 100).astype(np.float64) * 5.0, (40, 1))          # V-shaped valley, 5 m per column
    dem[0, 0] = -32767.0
    f = dg.dem_features(dem, tr, -32767.0, window_m=1000.0)
    assert f.rel_elev_m[20, 100] == 0.0 and np.isnan(f.rel_elev_m[0, 0]) and np.isnan(f.slope_deg[0, 0])
    assert 0 < f.rel_elev_m[20, 110] <= 50.0 and f.rel_elev_m[20, 170] > 50.0


def test_approximate_radar_geometry_ascending():
    look = dg.LOOK_AZIMUTH_DEG["ascending"]                                      # sensor toward 258 deg
    slope = np.array([2.0, 20.0, 45.0, 20.0, 60.0, np.nan])
    aspect = np.array([0.0, 258.0, 258.0, 78.0, 78.0, 0.0])
    assert [dg.GEOMETRY[k] for k in dg.radar_geometry(slope, aspect, look)] == [
        "flat_or_gentle", "facing_sensor", "layover_likely", "facing_away", "shadow_likely", "no_dem"]


def test_3857_inverse_round_trip():
    lon, lat = np.array([85.17, 85.42]), np.array([27.97, 28.28])
    x, y = lonlat_to_3857(lon, lat)
    back_lon, back_lat = dg.lonlat_from_3857(x, y)
    assert np.allclose(back_lon, lon, atol=1e-9) and np.allclose(back_lat, lat, atol=1e-9)


# ---- full diagnostics ---------------------------------------------------------------------------------------


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


@needs_rasterio
def test_diagnostics_report_on_all_modes(tmp_path_factory, gpt):
    project, data, mask = synthetic_project(tmp_path_factory.mktemp("d"), gpt)
    results = {m: run_mode(project, change_model, m) for m in ("event", *fv.CONTROL_MODES)}
    job = dg.prepare_diagnostics(project.cfg, project.manifest.aoi_id, candidate_id(project))
    assert [m.mode for m in job.modes] == ["event", "no_event", "zero_change", "static_post"]
    assert job.dem_path is not None and job.orbit_direction == "ascending"
    out_dir, report = dg.run_diagnostics(job, project.cfg)
    assert sorted(p.name for p in out_dir.iterdir()) == [dg.REPORT_JSON, dg.REPORT_MD]
    assert json.loads((out_dir / dg.REPORT_JSON).read_text(encoding="utf-8")) == report
    assert report["validated"] is False and "not a validation" in report["note"]
    modes = report["modes"]
    for m, r in results.items():
        assert modes[m]["flood"]["pixels"] == r.provenance.statistics.flood_pixels
        assert modes[m]["permanent_water"]["pixels"] == r.provenance.statistics.permanent_water_pixels
    event = modes["event"]
    assert event["flood"]["pixels"] == 60 * 80 and event["flood"]["shares"]["post_vs_pre2_vv_darkening_3db"] == 1.0
    assert event["flood"]["shares"]["post_vv_above_clamp"] == 0.0
    assert event["flood_components"] == {"computed": True, "components": 1, "largest_px": 4800, "median_px": 4800.0,
                                         "pixel_share_in_components": {"1 px": 0.0, "2-9 px": 0.0, "10-99 px": 0.0,
                                                                       "100-999 px": 0.0, ">= 1000 px": 1.0}} \
        if importlib.util.find_spec("scipy") else event["flood_components"]["computed"] is False
    assert modes["zero_change"]["flood"]["pixels"] == 0 and modes["static_post"]["flood"]["pixels"] == 0
    assert modes["static_post"]["permanent_water"]["pixels"] >= 60 * 80          # dark appearance -> water class
    nev = modes["no_event"]
    assert nev["flood"]["pixels"] == 40 * 40 and nev["agreement_with_event_flood"]["intersection_px"] == 0
    assert nev["flood"]["shares"]["any_vv_above_clamp"] == 1.0                     # the PRE1 bright box
    valid = int(mask.sum())
    scene = report["scene"]
    assert scene["valid_pixels"] == valid and math.isclose(scene["above_clamp_share"]["PRE1_VV"],
                                                           round(1600 / valid, 6))
    assert report["all_valid_pixels"]["pixels"] == valid
    flat = scene["flat_terrain"]                                     # the fixture DEM is gentle: all pixels are flat
    assert flat["pixels"] == valid and math.isclose(flat["above_clamp_share"]["PRE1_VV"], round(1600 / valid, 6))
    assert flat["above_clamp_share"]["POST_VV"] == 0.0 and flat["linear_quantiles"]["PRE2_VV"]["p50"] == 0.08
    shares = report["all_valid_pixels"]["shares"]
    assert math.isclose(sum(shares[f"geometry {g}"] for g in dg.GEOMETRY), 1.0, abs_tol=1e-5)
    assert math.isclose(sum(shares[f"slope {s}"] for s in dg.SLOPE_LABELS) + shares["geometry no_dem"], 1.0,
                        abs_tol=1e-5) or shares["geometry no_dem"] == 0.0
    md = (out_dir / dg.REPORT_MD).read_text(encoding="utf-8")
    assert "NOT validation" in md and "Kuro Siwo maximum possible: VV 37.9 %" in md
    assert data[0][BRIGHT_BOX].max() > 0.15 and data[4][FLOOD_BOX].max() < 0.02 and data[0][WATER_BOX].max() < 0.01


@needs_rasterio
def test_diagnostics_refuse_missing_or_changed_outputs(tmp_path_factory, gpt):
    project, _data, _mask = synthetic_project(tmp_path_factory.mktemp("r"), gpt)
    with pytest.raises(fv.InferenceError, match="no FloodViT output"):
        dg.prepare_diagnostics(project.cfg, project.manifest.aoi_id, candidate_id(project))
    result = run_mode(project, change_model, "event")
    job = dg.prepare_diagnostics(project.cfg, project.manifest.aoi_id, candidate_id(project))
    dg.run_diagnostics(job, project.cfg)
    with pytest.raises(fv.InferenceError, match="already exists"):
        dg.run_diagnostics(job, project.cfg)
    pred = result.output_dir / fv.PREDICTION_FILE
    pred.write_bytes(pred.read_bytes() + b"\0")
    with pytest.raises(fv.InferenceError, match="changed since inference"):
        dg.prepare_diagnostics(project.cfg, project.manifest.aoi_id, candidate_id(project))

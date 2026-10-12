"""Step 4C deterministic SAR change baseline: per-pixel rule, gating (NOT_ASSESSED), pre-event water,
thresholds from config, component filtering, outputs, statistics denominators, FloodViT comparison.
Synthetic TEST FIXTURE values only; no real imagery, no model."""

from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest
from pydantic import ValidationError

from bato.config import load_config
from bato.contracts.baseline import FLOOD_CODES, STATE_CODES, BaselineProvenance, BaselineState, BaselineStatistics
from bato.contracts.io import load_contract
from bato.detection import baseline as bl
from bato.detection.components import component_sizes, label_sparse
from bato.exceptions import ConfigError
from bato.preprocessing import snap

from inference_fakes import candidate_id, change_model, run_mode, synthetic_project, write_synthetic_stack

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
S = BaselineState
LOOK = 78.0                      # ascending: sensor toward 258 deg
SENSOR_AZ = 258.0


@pytest.fixture(scope="module")
def params():
    from helpers import REPO_CONFIG_DIR
    return bl.BaselineParams.from_config(load_config(REPO_CONFIG_DIR))


def lin(db: float) -> float:
    return float(10 ** (db / 10))


def pixel(pre1, pre2, post):
    """One-pixel (6, 1, 1) layout-v1 block from (VV, VH) dB pairs."""
    vals = [*pre1, *pre2, *post]
    return np.array([lin(v) for v in vals], dtype=np.float32).reshape(6, 1, 1)


def classify(raw, params, *, valid=1, slope=2.0, aspect=0.0):
    one = np.ones((1, 1))
    return bl.classify_block(raw, (one * valid).astype(np.uint8), (one * slope).astype(np.float32),
                             (one * aspect).astype(np.float32), params, LOOK)


LAND, WATER = (-9.0, -15.0), (-20.0, -27.0)


# ---- per-pixel rule -----------------------------------------------------------------------------------------


def test_new_water_relative_to_both_pre_images_is_a_flood_candidate(params):
    r = classify(pixel(LAND, LAND, WATER), params)
    assert r.state[0, 0] == S.FLOOD and r.candidate[0, 0]


def test_darkening_is_required_against_both_pre_images(params):
    post = (-16.0, -23.0)                                         # open water (below -15 / -22 dB)
    near = (-14.5, -21.5)                                         # 1.5 dB brighter than POST, not water
    assert classify(pixel(near, LAND, post), params).state[0, 0] == S.NO_FLOOD       # dark vs PRE2 only
    assert classify(pixel(LAND, near, post), params).state[0, 0] == S.NO_FLOOD       # dark vs PRE1 only
    assert classify(pixel(LAND, LAND, post), params).state[0, 0] == S.FLOOD          # dark vs both


def test_both_polarisations_must_be_dark_and_darkened(params):
    assert classify(pixel(LAND, LAND, (-20.0, -15.0)), params).state[0, 0] == S.NO_FLOOD   # VH not water
    assert classify(pixel(LAND, LAND, (-14.0, -27.0)), params).state[0, 0] == S.NO_FLOOD   # VV not water
    # water at POST but VH darkened by only 1 dB vs PRE2 -> not new
    assert classify(pixel(LAND, (-9.0, -23.0), (-20.0, -24.0)), params).state[0, 0] == S.NO_FLOOD


@pytest.mark.parametrize(("pre1", "pre2"), [(WATER, LAND), (LAND, WATER), (WATER, WATER), ((-18.0, -24.0), LAND)])
def test_pre_existing_water_is_not_flood(params, pre1, pre2):
    post = (-24.0, -31.0)                                         # darker still: passes the change test vs water
    r = classify(pixel(pre1, pre2, post), params)
    assert r.state[0, 0] == S.PRE_EVENT_WATER and not r.candidate[0, 0]


def test_receding_water_is_no_flood(params):
    assert classify(pixel(WATER, WATER, LAND), params).state[0, 0] == S.NO_FLOOD


# ---- gating: NOT_ASSESSED, never NO_FLOOD -------------------------------------------------------------------


def test_steep_terrain_is_not_assessed(params):
    side = (SENSOR_AZ + 90) % 360                                 # slope across the range direction: no layover/shadow
    for raw in (pixel(LAND, LAND, WATER), pixel(LAND, LAND, LAND)):
        assert classify(raw, params, slope=15.0, aspect=side).state[0, 0] == S.NOT_ASSESSED_STEEP
    assert classify(pixel(LAND, LAND, WATER), params, slope=14.9, aspect=side).state[0, 0] == S.FLOOD


def test_shadow_and_layover_are_not_assessed(params):
    away = LOOK                                                   # downslope along the look: faces away
    assert classify(pixel(LAND, LAND, WATER), params, slope=50.0, aspect=away).state[0, 0] == S.NOT_ASSESSED_SHADOW
    assert classify(pixel(LAND, LAND, LAND), params, slope=50.0, aspect=away).state[0, 0] == S.NOT_ASSESSED_SHADOW
    facing = SENSOR_AZ
    assert classify(pixel(LAND, LAND, WATER), params, slope=35.0, aspect=facing).state[0, 0] == S.NOT_ASSESSED_LAYOVER
    # facing away but not steep enough for shadow: steep gate instead
    assert classify(pixel(LAND, LAND, WATER), params, slope=30.0, aspect=away).state[0, 0] == S.NOT_ASSESSED_STEEP
    lia = params.incidence_deg - bl.range_slope_deg(np.array([50.0]), np.array([away]), LOOK)[0]
    assert lia >= params.shadow_lia_min_deg


def test_invalid_sar_and_missing_dem_are_not_assessed(params):
    assert classify(pixel(LAND, LAND, WATER), params, valid=0).state[0, 0] == S.NOT_ASSESSED_INVALID_SAR
    assert classify(pixel(LAND, LAND, WATER), params, slope=np.nan).state[0, 0] == S.NOT_ASSESSED_NO_DEM
    # invalid SAR wins over every other gate
    assert classify(pixel(LAND, LAND, WATER), params, valid=0, slope=60.0, aspect=LOOK).state[0, 0] == \
        S.NOT_ASSESSED_INVALID_SAR


# ---- thresholds from config ---------------------------------------------------------------------------------


def test_thresholds_come_from_config_and_are_validated(config_copy):
    cfg = load_config(config_copy())
    p = bl.BaselineParams.from_config(cfg)
    assert (p.vv_water_db, p.vh_water_db, p.pre2_change_db, p.pre1_change_db) == (-15.0, -22.0, -3.0, -3.0)
    assert (p.max_slope_deg, p.shadow_lia_min_deg, p.incidence_deg, p.shadow_search_m) == (15.0, 85.0, 39.0, 5000.0)
    assert (p.layover_lia_max_deg, p.min_component_pixels) == (5.0, 10)          # inherited
    records = {r.name: r for r in bl.threshold_records(cfg)}
    assert records["baseline_flood.vv_water_db"].basis == "physics_literature"
    assert records["detection.min_component_pixels"].basis == "inherited_bato_design"
    assert all(r.provisional and r.rationale for r in records.values())
    strict = load_config(config_copy("thresholds.yaml",
                                     lambda d: d["baseline_flood"]["vv_water_db"].update(value=-25.0)))
    raw = pixel(LAND, LAND, WATER)                                # VV -20 dB: water at -15, not at -25
    assert classify(raw, p).state[0, 0] == S.FLOOD
    assert classify(raw, bl.BaselineParams.from_config(strict)).state[0, 0] == S.NO_FLOOD


@pytest.mark.parametrize(("edit", "message"), [
    (lambda d: d["baseline_flood"]["vv_water_db"].update(value=3.0), "vv_water_db"),
    (lambda d: d["baseline_flood"]["pre1_change_db"].update(value=1.0), "pre1_change_db"),
    (lambda d: d["baseline_flood"]["max_slope_deg"].update(value=95.0), "max_slope_deg"),
    (lambda d: d["baseline_flood"]["shadow_lia_min_deg"].update(value=30.0), "shadow_lia_min_deg"),
    (lambda d: d["baseline_flood"]["nominal_incidence_deg"].pop("basis"), "needs a basis"),
    (lambda d: d["baseline_flood"].pop("vh_water_db"), "vh_water_db"),
    (lambda d: d["baseline_flood"]["shadow_search_m"].update(value=0.0), "shadow_search_m"),
])
def test_invalid_thresholds_are_rejected(config_copy, edit, message):
    with pytest.raises(ConfigError, match=message):
        load_config(config_copy("thresholds.yaml", edit))


# ---- components ---------------------------------------------------------------------------------------------


def test_component_filter_removes_small_clusters_8_connected():
    state = np.zeros((30, 30), dtype=np.uint8)
    state[2:5, 2:5] = S.FLOOD                                    # 9 px -> removed (min 10)
    state[10:12, 10:15] = S.FLOOD                                # 10 px -> kept
    for k in range(12):                                          # 12 px diagonal chain -> kept (8-connected)
        state[15 + k, 2 + k] = S.FLOOD
    state[25, 25] = S.FLOOD                                      # isolated -> removed
    state[0, 29] = S.NOT_ASSESSED_STEEP
    stats = bl.filter_components(state, 10)
    assert (stats.candidate_pixels, stats.candidate_components, stats.kept_components) == (32, 4, 2)
    assert (stats.removed_components, stats.removed_pixels, stats.removed_pixel_share) == (2, 10, round(10 / 32, 6))
    assert stats.largest_component_px == 12 and stats.median_kept_component_px == 11.0
    assert (state[2:5, 2:5] == S.NO_FLOOD).all() and state[25, 25] == S.NO_FLOOD
    assert (state[10:12, 10:15] == S.FLOOD).all() and state[0, 29] == S.NOT_ASSESSED_STEEP


def test_sparse_labelling_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((50, 70)) < rng.uniform(0.05, 0.6)
        idx, roots = label_sparse(m)
        lab, n = ndimage.label(m, structure=np.ones((3, 3), dtype=bool))
        assert len(set(roots.tolist())) == n == len(set(zip(roots.tolist(), lab.ravel()[idx].tolist(), strict=True)))
    idx, _roots, sizes = component_sizes(np.zeros((5, 5), dtype=bool))
    assert idx.size == sizes.size == 0


# ---- full runs on a Step 3.5 stack ------------------------------------------------------------------------------


H, W = 300, 470
FLOOD_BOX = (slice(100, 160), slice(300, 380))     # 60 x 80 new water
TINY_BOX = (slice(30, 33), slice(400, 403))        # 3 x 3 new water -> removed by the cluster filter
LAKE_BOX = (slice(200, 260), slice(40, 120))       # water on every date -> PRE_EVENT_WATER


def baseline_scene() -> tuple[np.ndarray, np.ndarray]:
    data = np.empty((6, H, W), dtype=np.float32)
    data[0::2], data[1::2] = lin(-9.0), lin(-15.0)
    for box in (FLOOD_BOX, TINY_BOX):
        data[4][box], data[5][box] = lin(-20.0), lin(-27.0)
    data[0::2, LAKE_BOX[0], LAKE_BOX[1]] = lin(-20.0)
    data[1::2, LAKE_BOX[0], LAKE_BOX[1]] = lin(-27.0)
    data[4, :10, :] = 0.0                                       # POST no-data strip
    return data, np.all(data != 0.0, axis=0).astype(np.uint8)


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


@pytest.fixture
def project(tmp_path_factory, gpt):
    project, _data, _mask = synthetic_project(tmp_path_factory.mktemp("b"), gpt)
    from bato.preprocessing import stack as stk
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg, overwrite=True)
    data, mask = baseline_scene()
    write_synthetic_stack(project, result, data, mask)
    return project


def read(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.tags()


@needs_rasterio
def test_full_run_states_outputs_and_denominators(project):
    job = bl.prepare_baseline(project.cfg, project.manifest.aoi_id, candidate_id(project))
    assert job.dem_path is not None and job.floodvit_prediction is None
    out, prov = bl.run_baseline(job, project.cfg)
    assert sorted(p.name for p in out.iterdir()) == sorted([bl.FLOOD_FILE, bl.STATE_FILE, bl.PROVENANCE_FILE,
                                                            bl.DIAGNOSTICS_FILE])
    assert load_contract(BaselineProvenance, out / bl.PROVENANCE_FILE) == prov and prov.validated is False
    state, state_nodata, tags = read(out / bl.STATE_FILE)
    flood, flood_nodata, _ = read(out / bl.FLOOD_FILE)
    assert state_nodata is None and flood_nodata == 255 and tags["state_13"] == "NOT_ASSESSED_SHADOW"
    assert prov.state_codes == STATE_CODES and prov.flood_codes == FLOOD_CODES
    assert set(np.unique(state).tolist()) == {0, 1, 2, 10}
    assert (state[FLOOD_BOX] == S.FLOOD).all() and (state[TINY_BOX] == S.NO_FLOOD).all()
    assert (state[LAKE_BOX] == S.PRE_EVENT_WATER).all() and (state[:10] == S.NOT_ASSESSED_INVALID_SAR).all()
    assert np.array_equal(flood == 1, state == S.FLOOD) and np.array_equal(flood == 255, state >= 10)
    assert np.array_equal(flood == 0, np.isin(state, [0, 2]))
    s = prov.statistics
    total, invalid, flood_px = H * W, 10 * W, 60 * 80
    assert (s.total_pixels, s.not_assessed_pixels, s.assessed_pixels) == (total, invalid, total - invalid)
    assert (s.flood_pixels, s.pre_event_water_pixels) == (flood_px, 60 * 80)
    assert s.flood_fraction_of_assessed == round(flood_px / (total - invalid), 8)
    assert s.flood_fraction_of_all == round(flood_px / total, 8)
    assert s.flood_darkened_vs_pre2 == s.flood_darkened_vs_pre1 == flood_px
    assert sum(s.flood_by_slope_deg.values()) == flood_px and s.flood_by_geometry["shadow_likely"] == 0
    assert (s.components.candidate_pixels, s.components.removed_pixels) == (flood_px + 9, 9)
    assert s.flood_change_db_quantiles["vv_vs_pre2"]["p50"] == pytest.approx(-11.0, abs=0.01)
    assert "NOT AVAILABLE" in s.osm_waterway_overlap
    md = (out / bl.DIAGNOSTICS_FILE).read_text(encoding="utf-8")
    assert "NOT VALIDATED" in md and "NOT_ASSESSED_INVALID_SAR" in md and "no FloodViT event output" in md
    data = json.loads((out / bl.PROVENANCE_FILE).read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="denominators"):
        BaselineStatistics.model_validate({**data["statistics"], "flood_fraction_of_all": 0.5})
    with pytest.raises(ValidationError, match="assessed"):
        BaselineStatistics.model_validate({**data["statistics"], "assessed_pixels": total})


@needs_rasterio
def test_repeated_runs_are_identical_and_floodvit_is_only_compared(project):
    job = bl.prepare_baseline(project.cfg, project.manifest.aoi_id, candidate_id(project))
    out, first = bl.run_baseline(job, project.cfg)
    state1 = read(out / bl.STATE_FILE)[0]
    run_mode(project, change_model, "event")                      # FloodViT-like output now exists (fake model)
    job2 = bl.prepare_baseline(project.cfg, project.manifest.aoi_id, candidate_id(project))
    assert job2.floodvit_prediction is not None
    out2, second = bl.run_baseline(job2, project.cfg, overwrite=True)
    state2 = read(out2 / bl.STATE_FILE)[0]
    assert np.array_equal(state1, state2)                          # detection independent of FloodViT
    assert first.statistics == second.statistics
    assert hashlib.sha256((out2 / bl.STATE_FILE).read_bytes()).hexdigest() == second.outputs.state_sha256
    c = second.floodvit_comparison
    assert c is not None and c.both + c.baseline_only == second.statistics.flood_pixels
    assert c.jaccard is not None and 0.0 <= c.jaccard <= 1.0
    assert len(list(out.parent.glob("baseline.superseded-*"))) == 1
    with pytest.raises(bl.BaselineError, match="already exists"):
        bl.run_baseline(job2, project.cfg)


# ---- cast shadow (flat ground shaded by terrain toward the sensor) ------------------------------------------


def test_cast_shadow_behind_a_ridge_matches_geometry():
    import math

    from bato.detection.terrain_gates import cast_shadow
    res = 1 / 3600
    dem = np.zeros((120, 300), dtype=np.float32)
    dem[:, 60:66] = 1000.0                                        # N-S ridge, 1 km high; sensor toward 258 deg (west)
    shadow = cast_shadow(dem, (res, 0.0, 85.0, 0.0, -res, 28.2), None, SENSOR_AZ, 39.0, 5000.0)
    behind = np.flatnonzero(shadow[60, 66:])
    dx = res * 111_320 * math.cos(math.radians(28.2 - 60 * res))
    expected = 1000.0 * math.tan(math.radians(39.0)) * abs(math.sin(math.radians(SENSOR_AZ)))   # ~792 m east-west
    assert abs((behind.max() + 1) * dx - expected) < 2 * dx
    assert not shadow[60, :60].any()                              # the sensor side is lit
    assert not cast_shadow(dem, (res, 0.0, 85.0, 0.0, -res, 28.2), None, SENSOR_AZ, 39.0, 300.0)[60, 80:].any()


def test_flat_floor_in_cast_shadow_is_not_assessed_not_water(params):
    dark_every_date = pixel(WATER, WATER, WATER)                  # shaded gorge floor looks like water on every date
    assert classify(dark_every_date, params).state[0, 0] == S.PRE_EVENT_WATER
    r = bl.classify_block(dark_every_date, np.ones((1, 1), np.uint8), np.full((1, 1), 2.0, np.float32),
                          np.zeros((1, 1), np.float32), params, LOOK, np.ones((1, 1), bool))
    assert r.state[0, 0] == S.NOT_ASSESSED_SHADOW and r.shadow_cast_only[0, 0]
    new_water = pixel(LAND, LAND, WATER)
    r = bl.classify_block(new_water, np.ones((1, 1), np.uint8), np.full((1, 1), 2.0, np.float32),
                          np.zeros((1, 1), np.float32), params, LOOK, np.ones((1, 1), bool))
    assert r.state[0, 0] == S.NOT_ASSESSED_SHADOW and not r.candidate[0, 0]


@needs_rasterio
def test_slope_bins_follow_the_threshold(project):
    cfg = load_config(project.cfg.config_dir)
    p = bl.BaselineParams.from_config(cfg)
    job = bl.prepare_baseline(cfg, project.manifest.aoi_id, candidate_id(project))
    import dataclasses
    maps = bl.compute_baseline(job, cfg, params=dataclasses.replace(p, max_slope_deg=8.0))
    assert list(maps.statistics.flood_by_slope_deg) == ["0-5 deg", "5-8 deg"]
    assert sum(maps.statistics.flood_by_slope_deg.values()) == maps.statistics.flood_pixels
    assert maps.statistics.shadow_cast_only_pixels >= 0


@needs_rasterio
def test_inconsistent_stack_values_fail_closed(project):
    from bato.preprocessing import stack as stk
    data, mask = baseline_scene()
    data[2, 150, 150] = np.nan                                    # NaN where the mask says valid
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg, overwrite=True)
    write_synthetic_stack(project, result, data, mask)
    job = bl.prepare_baseline(project.cfg, project.manifest.aoi_id, candidate_id(project))
    from bato.exceptions import BatoError
    with pytest.raises(BatoError, match="NaN/inf"):
        bl.compute_baseline(job, project.cfg)
    assert not job.final_dir.exists()

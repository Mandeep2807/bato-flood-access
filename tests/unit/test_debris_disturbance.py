"""SAR disturbance (debris) detector on a synthetic stack (TEST FIXTURE - invented backscatter, not Sentinel-1 data).

Synthetic 80 x 80 scene, every image = background dB + N(0, 0.3 dB). Patches (rows, cols, Step 4C state, change):

    A  10-14, 10-14  STEEP (14)       +6 dB on POST in VV and VH              -> DISTURBANCE
    B  10-14, 30-34  NO_FLOOD (0)     +6 dB on POST in VV only                -> rejected (one polarisation)
    C  10-14, 50-54  NO_FLOOD (0)     PRE2 +6, POST +12 in VV and VH          -> rejected (unstable pre pair)
    D  30-34, 10-14  LAYOVER (12)     +6 dB both pols                         -> NOT_ASSESSED_LAYOVER
    S  30-34, 30-34  SHADOW (13)      +6 dB both pols                         -> NOT_ASSESSED_SHADOW
    F  30-34, 50-54  FLOOD (1)        +6 dB both pols                         -> excluded (no DISTURBANCE)
    W  50-54, 10-14  PRE_EVENT_WATER  +6 dB both pols                         -> excluded
    K  50-52, 30-32  NO_FLOOD (0)     +6 dB both pols, 9 px                   -> removed by the 10 px cluster rule
    G  50-54, 50-54  NO_FLOOD (0)     -6 dB both pols                         -> darkening diagnostic only
    N  70-71, 0-9    NO_DEM (11)                                              -> NOT_ASSESSED_NO_DEM
    I  row 79        invalid SAR (stack 0, mask 0, Step 4C 10)                -> NOT_ASSESSED_INVALID_SAR
"""

from __future__ import annotations

import numpy as np
import pytest

from bato.contracts.baseline import BaselineState
from bato.contracts.debris import DebrisState, DebrisStatistics
from bato.debris.disturbance import (
    KIND_DISTURBANCE,
    KIND_FLOOD,
    DebrisError,
    classify_debris_block,
    detect,
    keep_clusters,
    noise_model,
)

H = W = 80
NOISE_DB = 0.3
Z = 3.0
MIN_PX = 10
A, B, C = (slice(10, 15), slice(10, 15)), (slice(10, 15), slice(30, 35)), (slice(10, 15), slice(50, 55))
D, S, F = (slice(30, 35), slice(10, 15)), (slice(30, 35), slice(30, 35)), (slice(30, 35), slice(50, 55))
WAT, K, G = (slice(50, 55), slice(10, 15)), (slice(50, 53), slice(30, 33)), (slice(50, 55), slice(50, 55))
N = (slice(70, 72), slice(0, 10))


def scene(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    db = np.empty((6, H, W), dtype=np.float64)
    base = {0: -12.0, 1: -19.0}                                   # VV, VH background
    for k in range(6):
        db[k] = base[k % 2] + rng.normal(0.0, NOISE_DB, (H, W))
    pre2, post = (2, 3), (4, 5)
    for patch in (A, D, S, F, WAT, K):
        for k in post:
            db[k][patch] += 6.0
    db[post[0]][B] += 6.0                                        # VV only
    for k in pre2:
        db[k][C] += 6.0
    for k in post:
        db[k][C] += 12.0
    for k in post:
        db[k][G] -= 6.0
    raw = (10.0 ** (db / 10.0)).astype(np.float32)
    raw[:, 79, :] = 0.0
    valid = np.ones((H, W), dtype=np.uint8)
    valid[79, :] = 0
    s4 = np.zeros((H, W), dtype=np.uint8)
    s4[A] = int(BaselineState.NOT_ASSESSED_STEEP)
    s4[D] = int(BaselineState.NOT_ASSESSED_LAYOVER)
    s4[S] = int(BaselineState.NOT_ASSESSED_SHADOW)
    s4[F] = int(BaselineState.FLOOD)
    s4[WAT] = int(BaselineState.PRE_EVENT_WATER)
    s4[N] = int(BaselineState.NOT_ASSESSED_NO_DEM)
    s4[79, :] = int(BaselineState.NOT_ASSESSED_INVALID_SAR)
    return raw, valid, s4


def blocks_of(raw, valid, s4, rows: int = 32):
    def factory():
        for r in range(0, H, rows):
            yield r, raw[:, r:r + rows], valid[r:r + rows], s4[r:r + rows]
    return factory


@pytest.fixture(scope="module")
def maps():
    raw, valid, s4 = scene()
    return detect(blocks_of(raw, valid, s4), (H, W), Z, MIN_PX)


def test_both_polarisation_brightening_on_steep_terrain_is_detected(maps):
    assert (maps.state[A] == int(DebrisState.DISTURBANCE)).all()
    assert maps.statistics.steep_assessed_pixels == 25


def test_only_patch_a_is_disturbance(maps):
    assert int((maps.state == int(DebrisState.DISTURBANCE)).sum()) == 25
    assert maps.statistics.disturbance_pixels == 25


def test_one_polarisation_only_is_rejected(maps):
    assert (maps.state[B] == int(DebrisState.NO_DISTURBANCE)).all()


def test_unstable_pre_event_pair_is_rejected(maps):
    # d1 = 12 dB and d2 = 6 dB pass, but |d0| = 6 dB >= z * sigma: the pixel changed before the event
    assert (maps.state[C] == int(DebrisState.NO_DISTURBANCE)).all()


def test_layover_and_shadow_are_not_assessed(maps):
    assert (maps.state[D] == int(DebrisState.NOT_ASSESSED_LAYOVER)).all()
    assert (maps.state[S] == int(DebrisState.NOT_ASSESSED_SHADOW)).all()
    assert (maps.state[N] == int(DebrisState.NOT_ASSESSED_NO_DEM)).all()
    assert (maps.state[79] == int(DebrisState.NOT_ASSESSED_INVALID_SAR)).all()


def test_flood_and_pre_event_water_are_excluded(maps):
    assert (maps.state[F] == int(DebrisState.NO_DISTURBANCE)).all()
    assert (maps.state[WAT] == int(DebrisState.NO_DISTURBANCE)).all()


def test_small_cluster_is_removed_and_counted(maps):
    assert (maps.state[K] == int(DebrisState.NO_DISTURBANCE)).all()
    c = maps.statistics.components
    assert c.candidate_pixels == 34 and c.removed_pixels == 9 and c.kept_components == 1
    assert c.min_component_pixels == MIN_PX and c.largest_component_px == 25


def test_darkening_is_a_diagnostic_never_a_claim(maps):
    assert (maps.state[G] == int(DebrisState.NO_DISTURBANCE)).all()
    assert maps.statistics.darkening_diagnostic_pixels == 25


def test_noise_model_is_the_robust_spread_of_the_pre_pair(maps):
    vv, vh = maps.noise
    assert (vv.polarization, vh.polarization) == ("VV", "VH")
    expected = NOISE_DB * np.sqrt(2.0)
    assert vv.sigma_db == pytest.approx(expected, rel=0.1) and vh.sigma_db == pytest.approx(expected, rel=0.1)
    observable = H * W - 25 - 25 - 20 - W                       # minus layover, shadow, no DEM, invalid row
    assert vv.pixels == vh.pixels == observable


def test_change_table_holds_d1_d2_d0_for_disturbance_and_flood(maps):
    t = maps.table
    dist = t["kind"] == KIND_DISTURBANCE
    flood = t["kind"] == KIND_FLOOD
    assert int(dist.sum()) == 25 and int(flood.sum()) == 25
    assert set(zip(t["row"][dist].tolist(), t["col"][dist].tolist(), strict=True)) == {
        (r, c) for r in range(10, 15) for c in range(10, 15)}
    for key in ("d1_vv", "d2_vv", "d1_vh", "d2_vh"):
        assert np.median(t[key][dist]) == pytest.approx(6.0, abs=0.5)
    for key in ("d0_vv", "d0_vh"):
        assert abs(float(np.median(t[key][dist]))) < 0.5
    assert np.isnan(t["slope_deg"]).all()                       # no terrain supplied


def test_statistics_are_consistent(maps):
    s = maps.statistics
    assert isinstance(s, DebrisStatistics)
    assert s.state_counts["NOT_ASSESSED_LAYOVER"] == 25 and s.state_counts["NOT_ASSESSED_SHADOW"] == 25
    assert s.assessed_pixels == s.state_counts["NO_DISTURBANCE"] + 25
    assert s.disturbance_fraction_of_assessed == pytest.approx(25 / s.assessed_pixels)


def test_terrain_feeds_slope_and_facing_sensor_statistics():
    raw, valid, s4 = scene()

    def terrain(row0, rows):
        return np.full((rows, W), 35.0), np.full((rows, W), -20.0)

    m = detect(blocks_of(raw, valid, s4), (H, W), Z, MIN_PX, terrain=terrain)
    assert m.statistics.disturbance_by_slope_deg["30-45 deg"] == 25
    assert m.statistics.disturbance_by_geometry["facing_away (s_r <= -10 deg)"] == 25
    assert m.statistics.facing_sensor_share == 0.0 and not m.warnings


def test_facing_sensor_majority_is_warned():
    raw, valid, s4 = scene()
    m = detect(blocks_of(raw, valid, s4), (H, W), Z, MIN_PX,
               terrain=lambda r0, rows: (np.full((rows, W), 20.0), np.full((rows, W), 15.0)))
    assert m.statistics.facing_sensor_share == 1.0
    assert any("criterion (c)" in w for w in m.warnings)


def test_block_rule_directly():
    raw, valid, s4 = scene()
    res = classify_debris_block(raw, valid, s4, NOISE_DB * np.sqrt(2.0), NOISE_DB * np.sqrt(2.0), Z)
    assert res.candidate[A].all() and res.candidate[K].all()
    assert not res.candidate[B].any() and not res.candidate[C].any()
    assert not res.candidate[F].any() and not res.candidate[WAT].any()
    assert not res.candidate[D].any() and not res.candidate[S].any()
    assert res.darkening[G].all()
    with pytest.raises(DebrisError):
        classify_debris_block(raw[:, :10], valid, s4, 1.0, 1.0, Z)


def test_noise_model_fails_closed_without_spread():
    with pytest.raises(DebrisError):
        noise_model(np.zeros(100, dtype=np.float32), "VV")
    with pytest.raises(DebrisError):
        noise_model(np.array([], dtype=np.float32), "VH")


def test_keep_clusters_is_eight_connected():
    cand = np.zeros((10, 10), dtype=bool)
    for i in range(5):
        cand[i, i] = True                                       # diagonal line: one 8-connected component
    kept, comps = keep_clusters(cand, 5)
    assert kept.sum() == 5 and comps.kept_components == 1
    kept, comps = keep_clusters(cand, 6)
    assert kept.sum() == 0 and comps.removed_pixels == 5

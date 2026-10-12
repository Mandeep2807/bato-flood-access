"""Step 2B: same-track temporal-set selection (synthetic TEST FIXTURE data only)."""

from __future__ import annotations

import random
from datetime import date, datetime, time, timezone

import pytest
from pydantic import ValidationError

from bato.acquisition.cdse_odata import CDSEODataClient
from bato.acquisition.http import HttpResponse
from bato.acquisition.s1_catalogue import search_sentinel1
from bato.acquisition.temporal_selection import (
    build_passes,
    pairing_score,
    ranking_key,
    select_temporal_sets,
    to_pre_post_pair,
)
from bato.contracts.aoi import AOIRequest, build_aoi
from bato.contracts.common import BBox, OrbitDirection
from bato.contracts.io import load_contract, save_contract
from bato.contracts.pairing import PrePostPair
from bato.contracts.temporal import SetQualityFlag, TemporalSelectionResult, TrackKey

from helpers import FIXED_TODAY, TEST_AOI_BBOX, odata_record

EVENT = date(2026, 8, 26)
RETRIEVED = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
D, A = "DESCENDING", "ASCENDING"


def t(month: int, day: int, hour: int = 0, minute: int = 15, second: int = 20) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=timezone.utc)


class _Page:
    def __init__(self, records: list[dict]) -> None:
        import json
        self.body = json.dumps({"value": records}).encode()

    def get(self, url, timeout_s):
        return HttpResponse(200, self.body)

    def post_form(self, url, form, timeout_s):  # pragma: no cover
        raise AssertionError


def catalogue(cfg, records: list[dict], event_time: time | None = None):
    lo1, la1, lo2, la2 = TEST_AOI_BBOX
    aoi = build_aoi(AOIRequest(event_date=EVENT, event_time_utc=event_time,
                               bbox=BBox(min_lon=lo1, min_lat=la1, max_lon=lo2, max_lat=la2)),
                    cfg.settings, today=FIXED_TODAY)
    client = CDSEODataClient(cfg.settings.cdse, _Page(records))
    return search_sentinel1(aoi, cfg, client=client, retrieved_at=RETRIEVED)


def select(cfg, records, event_time=None):
    return select_temporal_sets(catalogue(cfg, records, event_time), cfg)


def with_pairing(cfg, **update):
    pairing = cfg.settings.pairing.model_copy(update=update)
    return cfg.model_copy(update={"settings": cfg.settings.model_copy(update={"pairing": pairing})})


def track(result, direction: str, rel: int):
    key = TrackKey(orbit_direction=OrbitDirection(direction.lower()), relative_orbit=rel)
    return next(tr for tr in result.tracks if tr.track == key)


# A typical 12-day repeat track: 4 pre passes, 2 post passes (all synthetic).
DESC19 = [odata_record("A", D, 19, t(7, 9)), odata_record("A", D, 19, t(7, 21)),
          odata_record("A", D, 19, t(8, 2)), odata_record("A", D, 19, t(8, 14)),
          odata_record("A", D, 19, t(9, 7)), odata_record("A", D, 19, t(9, 19))]
ASC85 = [odata_record("A", A, 85, t(8, 9, 12, 30)), odata_record("A", A, 85, t(8, 21, 12, 30)),
         odata_record("A", A, 85, t(9, 2, 12, 30))]


# 1. same relative orbit required ------------------------------------------------------

def test_sets_never_mix_relative_orbits(cfg):
    records = [odata_record("A", D, 19, t(8, 2)), odata_record("A", D, 19, t(8, 14)),
               odata_record("A", D, 20, t(8, 30))]                    # post on a different orbit
    result = select(cfg, records)
    assert track(result, D, 19).status == "insufficient_post"
    assert track(result, D, 20).status == "insufficient_pre"
    assert result.all_candidates() == []


def test_result_contract_rejects_cross_track_pass(cfg):
    result = select(cfg, DESC19 + ASC85)
    asc_pass = next(p.pass_id for p in result.passes if p.track.relative_orbit == 85)
    d19 = track(result, D, 19)
    forged = d19.model_copy(update={"post_pass_ids": (*d19.post_pass_ids, asc_pass)})
    tracks = tuple(forged if tr is d19 else tr for tr in result.tracks)
    with pytest.raises(ValidationError, match="cross-track"):
        TemporalSelectionResult.model_validate({**result.model_dump(), "tracks": [x.model_dump() for x in tracks]})


# 2. orbit direction required ---------------------------------------------------------

def test_same_relative_orbit_but_other_direction_is_a_separate_track(cfg):
    records = DESC19 + [odata_record("A", A, 19, t(8, 3, 12)), odata_record("A", A, 19, t(8, 15, 12)),
                        odata_record("A", A, 19, t(9, 1, 12))]
    result = select(cfg, records)
    assert {tr.track.label for tr in result.tracks} == {"ASCENDING/19", "DESCENDING/19"}
    for tr in result.tracks:
        for c in tr.candidates:
            assert all(p.startswith(f"S1A-{tr.track.orbit_direction.value[0].upper()}019")
                       for p in (*c.pre_event_pass_ids, c.post_event_pass_id))


def test_scene_without_orbit_direction_never_enters_selection(cfg):
    bad = odata_record("A", D, 19, t(8, 26 + 4), drop=("orbitDirection",))
    result = select(cfg, DESC19 + [bad])
    names = {n for p in result.passes for n in p.scene_names}
    assert bad["Name"] not in names


# 3. same-day exclusion ---------------------------------------------------------------

def test_event_day_scene_is_never_used(cfg):
    same_day = odata_record("A", D, 19, t(8, 26))
    result = select(cfg, DESC19 + [same_day])
    assert same_day["Name"] not in {n for p in result.passes for n in p.scene_names}


# 16. event timestamp handling ---------------------------------------------------------

def test_event_time_makes_same_day_scene_usable_as_post(cfg):
    same_day = odata_record("A", D, 19, t(8, 26, 0, 15))
    result = select(cfg, DESC19 + [same_day], event_time=time(0, 0))
    best = track(result, D, 19).candidates[0]
    post = next(p for p in result.passes if p.pass_id == best.post_event_pass_id)
    assert post.scene_names == (same_day["Name"],)
    assert best.post_gap_days == pytest.approx(0.01, abs=0.01)
    late = select(cfg, DESC19 + [same_day], event_time=time(12, 0))
    assert same_day["Name"] in {n for p in late.passes if p.role.value == "pre_event" for n in p.scene_names}


# 4. post-event selection --------------------------------------------------------------

def test_closest_post_ranks_first_when_quality_is_equal(cfg):
    best = track(select(cfg, DESC19), D, 19).candidates[0]
    assert best.post_event_pass_id.endswith("20260907T001520")
    assert best.post_gap_days == pytest.approx(12.011, abs=0.001)


# 5. pre-event selection ---------------------------------------------------------------

def test_pre_selection_prefers_recent_pre_with_earlier_partner(cfg):
    result = select(cfg, DESC19)
    best = track(result, D, 19).candidates[0]
    pre1, pre2 = best.pre_event_pass_ids                     # oldest first
    assert pre2.endswith("20260814T001520")                   # most recent pre
    assert pre1 < pre2 and best.pre_separation_days >= cfg.settings.pairing.min_pre_separation_days
    assert set(best.additional_pre_pass_ids) | {pre1, pre2} == set(track(result, D, 19).pre_pass_ids)


# 6. minimum baseline separation --------------------------------------------------------

def test_pre_pairs_closer_than_minimum_are_not_generated(cfg):
    records = [odata_record("A", D, 19, t(8, 14)), odata_record("C", D, 19, t(8, 20)),   # 6 d apart
               odata_record("A", D, 19, t(9, 1))]
    result = select(cfg, records)
    tr = track(result, D, 19)
    assert all(c.shape == "1pre+1post" for c in tr.candidates)
    assert "no two pre passes are >= 12 d apart" in tr.note
    relaxed = select_temporal_sets(catalogue(cfg, records), with_pairing(cfg, min_pre_separation_days=6))
    assert track(relaxed, D, 19).candidates[0].shape == "2pre+1post"


# 7. multiple candidate sets ------------------------------------------------------------

def test_multiple_ranked_candidates_and_generated_count(cfg):
    tr = track(select(cfg, DESC19), D, 19)
    # 4 pre passes 12 d apart -> 6 pairs, all >= 12 d; x 2 posts = 12 generated
    assert tr.candidates_generated == 12
    assert len(tr.candidates) == cfg.settings.pairing.max_candidates_per_track
    assert [c.rank for c in tr.candidates] == [1, 2, 3, 4, 5]
    totals = [c.score.total for c in tr.candidates]
    assert totals == sorted(totals, reverse=True)


# 8. deterministic ordering -------------------------------------------------------------

def test_selection_is_independent_of_catalogue_order(cfg):
    records = DESC19 + ASC85
    reference = select(cfg, records)
    shuffled = list(records)
    random.Random(11).shuffle(shuffled)
    again = select(cfg, shuffled)
    assert again.model_dump_json() == reference.model_dump_json()


# 9. tie breaking -------------------------------------------------------------------------

def test_tie_breaker_is_total_and_deterministic(cfg):
    tr = track(select(cfg, DESC19), D, 19)
    a = tr.candidates[0]
    equal_score = a.score.model_copy()
    b = a.model_copy(update={"candidate_id": "ts-b", "post_gap_days": a.post_gap_days + 1, "score": equal_score})
    c = a.model_copy(update={"candidate_id": "ts-c", "pre_event_pass_ids": ("Z", "Z2"), "score": equal_score})
    ordered = sorted([c, b, a], key=ranking_key)
    assert ordered == [a, c, b]          # same score: smaller post gap first, then pass ids


# 10. coverage influence ------------------------------------------------------------------

def test_lower_coverage_post_can_lose_to_a_later_full_coverage_post(cfg):
    pre = [odata_record("A", D, 19, t(8, 2)), odata_record("A", D, 19, t(8, 14))]
    near_partial = odata_record("C", D, 19, t(9, 1), footprint="east85")      # ~85 % coverage
    later_full = odata_record("A", D, 19, t(9, 7))
    records = [*pre, near_partial, later_full]
    default = track(select(cfg, records), D, 19).candidates[0]
    assert default.post_event_pass_id.endswith("20260901T001520")              # gap wins by default
    # Isolate the coverage effect: the later post (12.01 d) would otherwise also carry the
    # LONG_POST_DELAY penalty.
    heavy_cov = with_pairing(
        cfg, weights=cfg.settings.pairing.weights.model_copy(update={"post_gap": 0.10, "coverage": 0.45}),
        penalties=cfg.settings.pairing.penalties.model_copy(update={"long_post_delay": 0.0}))
    switched = track(select_temporal_sets(catalogue(cfg, records), heavy_cov), D, 19).candidates[0]
    assert switched.post_event_pass_id.endswith("20260907T001520")
    assert switched.coverage == 1.0 and default.coverage == pytest.approx(0.85, abs=0.01)


# 11. invalid metadata --------------------------------------------------------------------

def test_rejected_catalogue_records_are_ignored(cfg):
    broken = odata_record("A", D, 19, t(8, 30))
    broken["Name"] = "S1A_NOT_A_VALID_NAME.SAFE"
    result = select(cfg, DESC19 + [broken])
    assert all("NOT_A_VALID" not in n for p in result.passes for n in p.scene_names)
    assert track(result, D, 19).status == "candidates"


# 12/13. insufficient pre / post -----------------------------------------------------------

def test_no_pre_event_pass(cfg):
    tr = track(select(cfg, [odata_record("A", D, 19, t(9, 1))]), D, 19)
    assert tr.status == "insufficient_pre" and tr.candidates == ()


def test_single_pre_fallback_is_flagged_and_configurable(cfg):
    records = [odata_record("A", D, 19, t(8, 14)), odata_record("A", D, 19, t(9, 1))]
    tr = track(select(cfg, records), D, 19)
    assert tr.status == "candidates" and tr.candidates[0].shape == "1pre+1post"
    assert SetQualityFlag.SINGLE_PRE_IMAGE in tr.candidates[0].quality_flags
    assert tr.candidates[0].score.baseline_separation_score == 0.0
    off = select_temporal_sets(catalogue(cfg, records), with_pairing(cfg, allow_single_pre_fallback=False))
    assert track(off, D, 19).status == "insufficient_pre"


def test_no_post_event_pass(cfg):
    tr = track(select(cfg, DESC19[:4]), D, 19)
    assert tr.status == "insufficient_post" and "no post-event pass" in tr.note


# 14. multiple tracks -----------------------------------------------------------------------

def test_every_viable_track_keeps_its_own_ranking(cfg):
    result = select(cfg, DESC19 + ASC85)
    labels = [tr.track.label for tr in result.tracks]
    assert labels == ["ASCENDING/85", "DESCENDING/19"]                       # deterministic track order
    assert all(tr.status == "candidates" and tr.candidates[0].rank == 1 for tr in result.tracks)


# 15. multi-slice detection -------------------------------------------------------------------

def test_two_slices_form_one_pass_and_one_temporal_observation(cfg):
    pre = [odata_record("A", D, 19, t(8, 2)), odata_record("A", D, 19, t(8, 14))]
    south = odata_record("A", D, 19, t(9, 1, 0, 20, 10), footprint="south")
    north = odata_record("A", D, 19, t(9, 1, 0, 20, 35), footprint="north")
    result = select(cfg, [*pre, south, north])
    post_passes = [p for p in result.passes if p.role.value == "post_event"]
    assert len(post_passes) == 1 and post_passes[0].multi_slice
    assert post_passes[0].scene_names == (south["Name"], north["Name"]) and post_passes[0].coverage == 1.0
    best = track(result, D, 19).candidates[0]
    assert best.multi_slice and SetQualityFlag.MULTI_SLICE_PASS in best.quality_flags
    assert track(result, D, 19).candidates_generated == 1     # slices are not separate dates


# PrePostPair enforcement + scoring formula --------------------------------------------------

def test_every_candidate_converts_to_a_valid_pre_post_pair(cfg):
    cat = catalogue(cfg, DESC19 + ASC85)
    result = select_temporal_sets(cat, cfg)
    for c in result.all_candidates():
        pair = to_pre_post_pair(c, result, cat)
        assert isinstance(pair, PrePostPair)
        assert (pair.orbit_direction, pair.relative_orbit) == (c.track.orbit_direction, c.track.relative_orbit)


def test_total_is_weighted_sum_of_components(cfg):
    w = cfg.settings.pairing.weights
    for c in select(cfg, DESC19).all_candidates():
        s = c.score
        expected = (w.post_gap * s.post_gap_score + w.pre_recency * s.pre_recency_score
                    + w.baseline_separation * s.baseline_separation_score + w.coverage * s.coverage_score
                    + w.quality * s.quality_score)
        assert s.total == pytest.approx(expected, abs=1e-5)


def test_pairing_score_components(cfg):
    s1, p = cfg.settings.sentinel1, cfg.settings.pairing
    score = pairing_score(6.0, 12.0, 24.0, 1.0, (), s1, p)
    assert score.post_gap_score == pytest.approx(1 - 6 / s1.post_window_days)
    assert score.pre_recency_score == pytest.approx(1 - 12 / s1.pre_window_days)
    assert score.baseline_separation_score == 1.0 and score.quality_score == 1.0
    penalised = pairing_score(6.0, 12.0, 24.0, 1.0, (SetQualityFlag.LONG_POST_DELAY,), s1, p)
    assert penalised.quality_score == pytest.approx(1 - p.penalties.long_post_delay)


# 17. serialization ------------------------------------------------------------------------------

def test_selection_round_trips_through_json(cfg, tmp_path):
    result = select(cfg, DESC19 + ASC85)
    assert load_contract(TemporalSelectionResult, save_contract(result, tmp_path / "sel.json")) == result


def test_passes_are_built_from_accepted_records_only(cfg):
    cat = catalogue(cfg, DESC19)
    assert sum(len(p.scene_names) for p in build_passes(cat)) == len(cat.accepted())


# 18. configuration-driven thresholds --------------------------------------------------------------

def test_max_candidates_per_track_from_config(cfg):
    tr = track(select_temporal_sets(catalogue(cfg, DESC19), with_pairing(cfg, max_candidates_per_track=2)), D, 19)
    assert len(tr.candidates) == 2 and tr.candidates_generated == 12


def test_long_post_delay_threshold_from_config(cfg):
    lenient_s1 = cfg.settings.sentinel1.model_copy(update={"max_post_delay_days": 30})
    lenient = cfg.model_copy(update={"settings": cfg.settings.model_copy(update={"sentinel1": lenient_s1})})
    strict = track(select(cfg, DESC19), D, 19).candidates
    relaxed = track(select_temporal_sets(catalogue(cfg, DESC19), lenient), D, 19).candidates
    assert any(SetQualityFlag.LONG_POST_DELAY in c.quality_flags for c in strict)
    assert not any(SetQualityFlag.LONG_POST_DELAY in c.quality_flags for c in relaxed)


@pytest.mark.parametrize(("update", "message"), [
    ({"weights": {"post_gap": 0.5, "pre_recency": 0.5, "baseline_separation": 0.5, "coverage": 0, "quality": 0}},
     "sum to 1"),
    ({"min_pre_separation_days": 30, "target_pre_separation_days": 24}, "target_pre_separation_days"),
    ({"max_candidates_per_track": 0}, "max_candidates_per_track"),
    ({"heuristic_status": "validated"}, "heuristic_status"),
])
def test_invalid_pairing_config_rejected(config_copy, update, message):
    from bato.config import load_config
    from bato.exceptions import ConfigError

    def mutate(d):
        d["pairing"].update(update)
    with pytest.raises(ConfigError, match=message):
        load_config(config_copy("config.yaml", mutate))

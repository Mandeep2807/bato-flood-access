"""Connectivity engine on the synthetic road world of connectivity_fakes (TEST FIXTURE)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bato.connectivity.engine import (
    ConnectivityError,
    compute_connectivity,
    output_dir,
    run_connectivity,
)
from bato.connectivity.graph import Graph, arc_path, components, multi_source_dijkstra, zero_one_bfs
from bato.connectivity.settlements import GridIndex, LocalProjection, cluster_points, point_in_ring
from bato.contracts.connectivity import ConnectivityProvenance
from bato.contracts.evidence import UNSUPPORTED_CLAIM_TERMS, EvidenceRecord
from bato.contracts.impact import RoadState
from bato.contracts.settlement import AccessStatus, TargetKind
from bato.exceptions import ProhibitedInputError
from bato.osm.extract import load_osm_extract

from connectivity_fakes import EVENT, NOW, fake_job, fetch, project, state_raster

# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def line_graph() -> Graph:
    # 0 -1- 1 -2- 2 -3- 3 ; 1 -10- 3 (long detour)
    return Graph.build(4, np.array([0, 1, 2, 1]), np.array([1, 2, 3, 3]), np.array([1.0, 2.0, 3.0, 10.0]))


def test_dijkstra_multi_source_nearest_label():
    g = line_graph()
    dist, label = multi_source_dijkstra(g, np.ones(4, dtype=bool), {0: 0, 3: 1})
    assert dist.tolist() == [0.0, 1.0, 3.0, 0.0] and label.tolist() == [0, 0, 0, 1]
    dist, label = multi_source_dijkstra(g, np.array([True, False, True, True]), {0: 0})
    assert dist[3] == 11.0 and dist[2] == 14.0


def test_components_with_mask():
    g = line_graph()
    c = components(g, np.array([True, False, True, False]))
    assert c.tolist() == [0, 0, 2, 2]


def test_zero_one_bfs_counts_cost_one_arcs():
    # 0 -0-> 1 -1-> 2 -0-> 3 and back arcs cost 0
    fr, to, cost = np.array([0, 1, 1, 2, 2, 3]), np.array([1, 0, 2, 1, 3, 2]), np.array([0, 0, 1, 0, 0, 0])
    best, pred = zero_one_bfs(4, fr, to, cost, [0])
    assert best.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert [int(to[a]) for a in arc_path(fr, pred, 3, 10)] == [1, 2, 3]
    with pytest.raises(ValueError):
        zero_one_bfs(2, np.array([0]), np.array([1]), np.array([2]), [0])


def test_cluster_points_single_linkage():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [500.0, 0.0], [600.0, 0.0]])
    labels = cluster_points(xy, 125.0)
    assert labels.tolist() == [0, 0, 0, 3, 3]
    assert cluster_points(xy, 99.0).tolist() == [0, 1, 2, 3, 4]


def test_grid_index_radius():
    idx = GridIndex(np.array([[0.0, 0.0], [1000.0, 0.0]]), 500.0)
    i, d = idx.nearest(np.array([[450.0, 0.0], [550.0, 10.0], [5000.0, 0.0]]), 500.0)
    assert i.tolist() == [0, 1, -1] and d[0] == pytest.approx(450.0)


def test_point_in_ring_and_projection():
    ring = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    assert point_in_ring(0.5, 0.5, ring) and not point_in_ring(1.5, 0.5, ring)
    xy = LocalProjection(85.0, 28.0).xy(np.array([85.001]), np.array([28.0]))
    assert xy[0, 0] == pytest.approx(98.2, abs=0.5)


# ---------------------------------------------------------------------------
# End to end on the synthetic world
# ---------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    cfg = project(tmp_path)
    fetch(cfg)
    extract = load_osm_extract(cfg, "aoi-test", EVENT)
    state, transform = state_raster()
    job = fake_job(cfg, extract, tmp_path, state)
    result = compute_connectivity(job, cfg, state, transform, created_at=NOW)
    return cfg, job, state, transform, result


def by_id(result):
    return {r.candidate.settlement_id: r for r in result.settlements}


def test_settlement_access_states(world):
    *_, result = world
    s = by_id(result)
    assert s["place-node-1"].access_state.value == "CONNECTED"        # town (not its own destination)
    assert s["place-node-2"].access_state.value == "CONNECTED"
    assert s["place-node-3"].access_state.value == "CONFIRMED_CUTOFF"
    assert s["bc-way-1000"].access_state.value == "CONFIRMED_CUTOFF"  # unnamed building cluster
    assert s["place-node-6"].access_state.value == "CONFIRMED_CUTOFF"
    assert s["place-node-4"].access_state.value == "POSSIBLE_CUTOFF"   # only through UNOBSERVABLE road
    assert s["place-node-5"].access_state.value == "NO_PRE_EVENT_ACCESS"
    assert "place-node-7" not in s                                     # outside the AOI
    assert s["bc-way-1000"].candidate.building_count == 6
    assert s["place-node-3"].candidate.building_count == 3            # linked buildings, proxy only
    assert s["place-node-5"].status.snap_distance_m is None


def test_town_is_not_its_own_destination(world):
    *_, result = world
    town = by_id(result)["place-node-1"]
    assert town.own_target is not None
    assert town.status.pre_event.target_id != "node/1"
    village = by_id(result)["place-node-2"]
    assert village.status.pre_event.target_id == "node/1" and village.status.pre_event.target_kind is TargetKind.TOWN


def test_road_states_and_unobservable_reasons(world):
    *_, result = world
    s = result.summary
    assert s.road_segments_by_state["blocked"] > 0 and s.road_segments_by_state["unobservable"] > 0
    assert set(s.unobservable_km_by_reason) == {"NOT_ASSESSED_STEEP"}
    assert s.exit_nodes == 2 and s.targets_by_kind == {"analysis_exit": 2, "hospital": 1, "town": 1}
    bridge = [i for i in result.segment_impacts if i.osm_way_id == 102]
    assert bridge and all(i.state is RoadState.OPEN for i in bridge)  # over pre-event water: observed open
    assert s.bridges_by_state["open"] == len(bridge)


def test_before_vs_after(world):
    *_, result = world
    s = result.summary
    assert s.reachable_settlements == {"pre_event": 6, "post_optimistic": 3, "post_pessimistic": 2}
    assert s.isolated_components_with_settlements["pre_event"] == 0
    assert s.isolated_components_with_settlements["post_optimistic"] == 2


def test_clearing_one_stretch_reconnects_b_and_d(world):
    *_, result = world
    top = result.priorities[0]
    assert top.rank == 1 and top.settlements_reconnected == 2
    assert top.reconnected_settlement_ids == ("bc-way-1000", "place-node-3")
    assert top.reconnected_building_count == 9 and len(top.segment_ids) > 1      # a stretch, not one segment
    assert top.settlements_reconnected_on_observed_roads == 2
    assert all(seg.startswith("seg-201-") for seg in top.segment_ids)


def test_two_stretches_needed_for_f_and_neither_alone_reconnects(world):
    *_, result = world
    f = by_id(result)["place-node-6"]
    assert f.min_blocked_stretches_to_clear == 2 and len(f.clearing_stretch_ids) == 2
    spur3 = [p for p in result.priorities if p.stretch_id in f.clearing_stretch_ids]
    assert len(spur3) == 2
    assert all(p.settlements_reconnected == 0 and p.in_minimum_clearing_set_of == 1 for p in spur3)
    assert by_id(result)["place-node-3"].min_blocked_stretches_to_clear == 1


def test_evidence_trail_for_every_cutoff(world):
    *_, result = world
    ev = {e.evidence_id: e for e in result.evidence}
    impacts = {i.segment_id: i for i in result.segment_impacts}
    for r in result.settlements:
        st = r.status
        if st.status in (AccessStatus.CONFIRMED_CUT_OFF, AccessStatus.POSSIBLY_CUT_OFF):
            own = ev[st.evidence_ids[0]]
            assert own.subject_id == st.settlement_id and own.method.startswith("graph:")
            assert st.blocking_segment_ids
            for sid in st.blocking_segment_ids:                              # settlement -> segment evidence
                assert impacts[sid].state is not RoadState.OPEN
                assert set(impacts[sid].evidence_ids) <= set(st.evidence_ids)
            assert {s.source.value for s in own.sources} >= {"sentinel-1", "osm-pre-event", "bato-derived"}
    for i in result.segment_impacts:
        if i.state is not RoadState.OPEN:
            e = ev[i.evidence_ids[0]]
            assert e.subject_id == i.segment_id and e.method == "rule:road_state_from_baseline_v1"
    for p in result.priorities:
        assert all(x in ev for x in p.evidence_ids)
    for e in result.evidence:
        assert not any(t in e.claim.lower() for t in UNSUPPORTED_CLAIM_TERMS)
        EvidenceRecord.model_validate(e.model_dump(mode="json"))


def test_unobservable_is_never_turned_into_open_or_blocked(world):
    cfg, job, state, transform, result = world
    c = by_id(result)["place-node-4"]
    assert c.status.post_optimistic.reachable and not c.status.post_pessimistic.reachable
    # making the steep terrain observable-open makes C connected; making it flood makes it confirmed cut off
    from connectivity_fakes import _box
    open_state = state.copy()
    _box(open_state, transform, 85.0395, 28.022, 85.0405, 28.026, 0)
    r2 = compute_connectivity(job, cfg, open_state, transform, created_at=NOW)
    assert by_id(r2)["place-node-4"].access_state.value == "CONNECTED"
    flood_state = state.copy()
    _box(flood_state, transform, 85.0395, 28.022, 85.0405, 28.026, 1)
    r3 = compute_connectivity(job, cfg, flood_state, transform, created_at=NOW)
    assert by_id(r3)["place-node-4"].access_state.value == "CONFIRMED_CUTOFF"


def test_no_flood_means_everything_with_roads_connected(world):
    cfg, job, state, transform, _ = world
    clean = np.zeros_like(state)
    r = compute_connectivity(job, cfg, clean, transform, created_at=NOW)
    assert r.summary.settlements_by_access_state == {"CONFIRMED_CUTOFF": 0, "POSSIBLE_CUTOFF": 0, "CONNECTED": 6,
                                                     "NO_PRE_EVENT_ACCESS": 1}
    assert r.priorities == [] and r.summary.road_segments_by_state["blocked"] == 0


def test_deterministic(world):
    cfg, job, state, transform, result = world
    again = compute_connectivity(job, cfg, state, transform, created_at=NOW)
    assert [e.evidence_id for e in again.evidence] == [e.evidence_id for e in result.evidence]
    assert [p.model_dump() for p in again.priorities] == [p.model_dump() for p in result.priorities]


def test_grid_mismatch_fails_closed(world):
    cfg, job, state, transform, _ = world
    with pytest.raises(ConnectivityError, match="does not match the stack grid"):
        compute_connectivity(job, cfg, state[:-1], transform)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def test_run_publishes_atomically_and_refuses_overwrite(world):
    cfg, job, state, transform, _ = world
    out, prov, result = run_connectivity(job, cfg, read_state=lambda _p: (state, transform), now=lambda: NOW)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["access_destinations.geojson", "blocked_segments_ranked.json", "connectivity_provenance.json",
                     "connectivity_report.md", "evidence_records.jsonl", "road_segments.geojson",
                     "settlements.geojson"]
    assert ConnectivityProvenance.model_validate_json((out / "connectivity_provenance.json").read_text()) == prov
    assert prov.validated is False and prov.osm_snapshot.snapshot_date.isoformat() < prov.event_date
    seg = json.loads((out / "road_segments.geojson").read_text())
    blocked = [f["properties"] for f in seg["features"] if f["properties"]["state"] == "blocked"]
    clicked = next(p for p in blocked if p["stretch_rank"] == 1)                     # the demo lookup
    assert clicked["settlements_reconnected"] == 2 and clicked["reconnected_building_count"] == 9
    settlements = json.loads((out / "settlements.geojson").read_text())["features"]
    assert all(f["properties"]["building_count_note"].endswith("not population)") for f in settlements)
    lines = (out / "evidence_records.jsonl").read_text().splitlines()
    assert len(lines) == len(result.evidence)
    report = (out / "connectivity_report.md").read_text()
    assert "Confirmed cut-off cards" in report and "NOT VALIDATED" in report
    with pytest.raises(ConnectivityError, match="already exists"):
        run_connectivity(job, cfg, read_state=lambda _p: (state, transform), now=lambda: NOW)
    run_connectivity(job, cfg, overwrite=True, read_state=lambda _p: (state, transform), now=lambda: NOW)
    assert list(out.parent.glob("ts-test.superseded-*"))


def test_state_changed_during_run_is_refused(world):
    cfg, job, state, transform, _ = world
    job.state_path.write_bytes(b"changed")
    with pytest.raises(ConnectivityError, match="changed during the run"):
        run_connectivity(job, cfg, read_state=lambda _p: (state, transform), now=lambda: NOW)


def test_outputs_never_in_evaluation_folders(tmp_path):
    cfg = project(tmp_path)
    with pytest.raises(ConnectivityError):
        output_dir(cfg, "../x", "ts")
    from bato.data_rules import assert_processed_output_dir
    with pytest.raises(ProhibitedInputError):
        assert_processed_output_dir(cfg.settings.paths.reference / "x", cfg.settings.paths)
    with pytest.raises(ProhibitedInputError):
        assert_processed_output_dir(cfg.settings.paths.interim / "x", cfg.settings.paths)


def _world_with(tmp_path, places_tf=None, buildings_tf=None):
    """The synthetic world with the places / buildings layers transformed (original list in, new list out)."""
    import connectivity_fakes as cf
    saved = (cf.places, cf.buildings)
    try:
        if places_tf is not None:
            cf.places = lambda: places_tf(saved[0]())
        if buildings_tf is not None:
            cf.buildings = lambda: buildings_tf(saved[1]())
        cfg = project(tmp_path)
        fetch(cfg)
    finally:
        cf.places, cf.buildings = saved
    extract = load_osm_extract(cfg, "aoi-test", EVENT)
    state, transform = state_raster()
    job = fake_job(cfg, extract, tmp_path, state)
    return compute_connectivity(job, cfg, state, transform, created_at=NOW)


def test_cut_off_town_does_not_make_its_pocket_connected(tmp_path):
    """B becomes a town mapped twice (node + area centroid): the flooded pocket must stay cut off."""
    import connectivity_fakes as cf

    def places(base):
        out = []
        for f in base:
            if f["properties"]["@osmId"] == "node/3":
                f["properties"]["place"] = "town"
                out.append(cf._pt("way/77", 85.0301, 28.0449, place="town", name="Village B"))
            out.append(f)
        return out

    result = _world_with(tmp_path, places_tf=places)
    s = by_id(result)
    assert s["place-node-3"].access_state.value == "CONFIRMED_CUTOFF"
    assert s["place-way-77"].access_state.value == "CONFIRMED_CUTOFF"
    assert s["bc-way-1000"].access_state.value == "CONFIRMED_CUTOFF"          # not "connected via the town"
    assert result.priorities[0].settlements_reconnected == 3                    # B, its duplicate, D
    assert result.summary.isolated_components_with_settlements["post_optimistic"] == 2


def test_cluster_next_to_a_place_outside_the_aoi_is_kept(tmp_path):
    import connectivity_fakes as cf

    result = _world_with(
        tmp_path,
        places_tf=lambda base: base + [cf._pt("node/8", 85.0332, 28.0460, place="hamlet", name="Just Outside")],
        buildings_tf=lambda base: base + [cf._pt(f"way/{2000 + i}", 85.0332, 28.0414 + 0.0009 * i) for i in range(6)])
    s = by_id(result)
    assert "place-node-8" not in s                                             # outside the AOI
    assert s["bc-way-1000"].access_state.value == "CONFIRMED_CUTOFF"
    assert s["bc-way-1000"].candidate.building_count >= 6


def test_place_snaps_from_its_own_point_not_from_a_far_building(tmp_path):
    """Hamlet E (no road within 500 m) gets no snap even when a building chain reaches a road elsewhere."""
    import connectivity_fakes as cf

    result = _world_with(tmp_path, buildings_tf=lambda base: base + [
        cf._pt(f"way/{3000 + i}", 85.02 + 0.0009 * i, 28.044) for i in range(12)])
    e = by_id(result)["place-node-5"]
    assert e.access_state.value == "NO_PRE_EVENT_ACCESS" and e.status.snap_distance_m is None


# ---------------------------------------------------------------------------
# Road rule v2: the disturbance raster wired into the road-impact input (graphs and decision table unchanged)
# ---------------------------------------------------------------------------

SPUR2_DEBRIS = (85.0395, 28.023, 85.0405, 28.025)          # inside spur 2's STEEP box


def _debris(state, transform, disturbance: bool):
    """debris_state from the Step 4C state as the detector maps it (steep/flood/water -> assessed 0), plus an
    optional DISTURBANCE patch across spur 2, and a change table for the patch and the FLOOD pixels."""
    from types import SimpleNamespace

    from bato.connectivity.engine import DebrisData
    from bato.debris.disturbance import TABLE_COLUMNS

    from connectivity_fakes import _box
    d = state.copy()
    d[np.isin(d, (1, 2, 14))] = 0
    if disturbance:
        _box(d, transform, *SPUR2_DEBRIS, 1)
    rr_d, cc_d = np.nonzero(d == 1)
    rr_f, cc_f = np.nonzero(state == 1)
    n_d, n_f = rr_d.size, rr_f.size
    table = {c: np.zeros(n_d + n_f, dtype=np.float32) for c in TABLE_COLUMNS}
    table["row"] = np.concatenate([rr_d, rr_f]).astype(np.int32)
    table["col"] = np.concatenate([cc_d, cc_f]).astype(np.int32)
    table["kind"] = np.concatenate([np.full(n_d, 2), np.full(n_f, 1)]).astype(np.uint8)
    for k, v in (("d1_vv", 6.0), ("d2_vv", 5.5), ("d0_vv", 0.2), ("d1_vh", 7.0), ("d2_vh", 6.5), ("d0_vh", -0.1)):
        table[k][:n_d] = v
    prov = SimpleNamespace(
        z=3.0, noise=(SimpleNamespace(sigma_db=0.8), SimpleNamespace(sigma_db=1.1)),
        outputs=SimpleNamespace(state="data/interim/detection/aoi-test/ts-test/debris/debris_state.tif",
                                state_sha256="d" * 64,
                                change_table="data/interim/detection/aoi-test/ts-test/debris/change_table.npz",
                                change_table_sha256="e" * 64))
    return DebrisData(state=d, table=table, provenance=prov)


def test_v2_steep_spur_without_disturbance_is_observed_open(world):
    cfg, job, state, transform, _ = world
    r = compute_connectivity(job, cfg, state, transform, created_at=NOW, debris=_debris(state, transform, False))
    assert r.road_rule == "v2"
    assert by_id(r)["place-node-4"].access_state.value == "CONNECTED"            # was POSSIBLE_CUTOFF in v1
    assert by_id(r)["place-node-3"].access_state.value == "CONFIRMED_CUTOFF"     # FLOOD still blocks
    assert r.summary.road_segments_by_state["unobservable"] == 0


def test_v2_disturbance_blocks_spur2_and_clearing_it_reconnects_c(world):
    cfg, job, state, transform, v1 = world
    r = compute_connectivity(job, cfg, state, transform, created_at=NOW, debris=_debris(state, transform, True))
    c = by_id(r)["place-node-4"]
    assert c.access_state.value == "CONFIRMED_CUTOFF" and c.min_blocked_stretches_to_clear == 1
    stretch = next(p for p in r.priorities if p.stretch_id in c.clearing_stretch_ids)
    assert stretch.settlements_reconnected == 1 and stretch.reconnected_settlement_ids == ("place-node-4",)
    # v1 results for the flood-only settlements are unchanged
    for sid in ("place-node-3", "bc-way-1000", "place-node-6", "place-node-1", "place-node-2", "place-node-5"):
        assert by_id(r)[sid].access_state == by_id(v1)[sid].access_state


def test_v2_disturbance_evidence_carries_d1_d2_d0_sigma_z_and_geometry(world):
    cfg, job, state, transform, _ = world
    r = compute_connectivity(job, cfg, state, transform, created_at=NOW, debris=_debris(state, transform, True))
    ev = {e.evidence_id: e for e in r.evidence}
    c = by_id(r)["place-node-4"]
    seg_ev = [ev[i] for i in c.status.evidence_ids if ev[i].subject_type.value == "road_segment"]
    assert seg_ev
    dist = [e for e in seg_ev if "radar disturbance consistent with debris" in e.claim]
    assert dist
    e = dist[0]
    assert "z=3" in e.claim and "sigma VV 0.80 dB" in e.claim and "stable pre-event pair" in e.claim
    v = e.values
    for key in ("z", "sigma_vv_db", "sigma_vh_db", "change_pixels_disturbance", "d1_vv_db_median", "d2_vv_db_median",
                "d0_vv_db_median", "d1_vh_db_median", "d2_vh_db_median", "d0_vh_db_median",
                "geometry_observable_samples", "step4c_samples_NOT_ASSESSED_STEEP"):
        assert key in v, key
    assert v["z"] == 3.0 and v["sigma_vv_db"] == 0.8 and v["change_pixels_disturbance"] > 0
    assert v["d1_vv_db_median"] == pytest.approx(6.0) and v["d0_vh_db_median"] == pytest.approx(-0.1)
    assert any(s.source.value == "osm-pre-event" and "way/" in s.source_id for s in e.sources)   # OSM road evidence
    assert {s.source.value for s in e.sources} >= {"sentinel-1", "bato-derived"}                 # source stack
    paths = {a.path for a in e.artifacts}
    assert "data/interim/detection/aoi-test/ts-test/debris/debris_state.tif" in paths
    assert any("disturbance debris_state.tif" in s.source_id for s in e.sources)
    for rec in r.evidence:
        assert not any(t in rec.claim.lower() for t in UNSUPPORTED_CLAIM_TERMS)
        EvidenceRecord.model_validate(rec.model_dump(mode="json"))


def test_v2_geometry_gate_keeps_the_road_unobservable(world):
    cfg, job, state, transform, _ = world
    from connectivity_fakes import _box
    d = _debris(state, transform, False)
    _box(d.state, transform, *SPUR2_DEBRIS, 12)                       # layover across spur 2
    r = compute_connectivity(job, cfg, state, transform, created_at=NOW, debris=d)
    assert by_id(r)["place-node-4"].access_state.value == "POSSIBLE_CUTOFF"
    assert set(r.summary.unobservable_km_by_reason) == {"NOT_ASSESSED_LAYOVER"}


def test_v2_debris_grid_mismatch_fails_closed(world):
    cfg, job, state, transform, _ = world
    d = _debris(state, transform, False)
    d.state = d.state[:-1]
    with pytest.raises(ConnectivityError, match="debris_state.tif does not match"):
        compute_connectivity(job, cfg, state, transform, debris=d)


def test_v2_run_records_rule_and_debris_inputs(world):
    from bato.connectivity.engine import DebrisInput
    from bato.preprocessing.speckle import sha256_file
    cfg, job, state, transform, _ = world
    d = _debris(state, transform, True)
    ddir = cfg.settings.paths.interim / "detection" / "aoi-test" / "ts-test" / "debris"
    ddir.mkdir(parents=True)
    state_path, table_path, prov_path = ddir / "debris_state.tif", ddir / "change_table.npz", ddir / "p.json"
    state_path.write_bytes(d.state.tobytes())
    np.savez_compressed(table_path, **d.table)
    prov_path.write_text("{}", encoding="utf-8")
    d.provenance.outputs.state_sha256 = sha256_file(state_path)
    d.provenance.outputs.change_table_sha256 = sha256_file(table_path)
    job.debris = DebrisInput(provenance_path=prov_path, provenance_sha256=sha256_file(prov_path),
                             provenance=d.provenance, state_path=state_path, table_path=table_path)
    rasters = {job.state_path: state, state_path: d.state}
    out, prov, result = run_connectivity(job, cfg, read_state=lambda p: (rasters[p], transform), now=lambda: NOW)
    assert prov.road_rule == "v2" and result.road_rule == "v2"
    assert {"debris_provenance", "debris_state", "debris_change_table"} <= set(prov.inputs)
    assert prov.rules["road_rule"] == "v2"
    assert by_id(result)["place-node-4"].access_state.value == "CONFIRMED_CUTOFF"
    state_path.write_bytes(b"changed")
    with pytest.raises(ConnectivityError, match="changed during the run"):
        run_connectivity(job, cfg, overwrite=True, read_state=lambda p: (rasters.get(p, d.state), transform),
                         now=lambda: NOW)


def test_v2_without_a_debris_run_is_refused(world):
    from types import SimpleNamespace

    from bato.connectivity.engine import prepare_debris_input
    cfg, job, *_ = world
    baseline = SimpleNamespace(outputs=SimpleNamespace(state_sha256="a" * 64),
                               source=SimpleNamespace(stack_sha256="b" * 64))
    with pytest.raises(ConnectivityError, match="needs the disturbance run"):
        prepare_debris_input(cfg, job.root, "aoi-test", "ts-test", baseline)

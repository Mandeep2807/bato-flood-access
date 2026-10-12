"""Dashboard bundle: official-run selection, integrity checks, verified route traces, rasters (TEST FIXTURE world)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from bato.dashboard import rasters as rs
from bato.dashboard.bundle import (
    DashboardBuildError,
    build_bundle,
    check_summary,
    debris_record,
    find_connectivity_runs,
    prepare_inputs,
    publish,
    recompute_summary,
    select_official_run,
)
from bato.dashboard.png import decode_png, encode_png
from bato.dashboard.routes import RouteTracer, build_network, unobservable_pockets

from dashboard_fakes import DEBRIS_CONTROL, DEBRIS_EVENT, inputs_for, published_world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    w = published_world(tmp_path_factory.mktemp("dash"))
    yield w
    mp.undo()


@pytest.fixture(scope="module")
def bundle(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("stage")
    return build_bundle(inputs_for(world), out), out


# ---------------------------------------------------------------- integrity


def test_bundle_numbers_are_the_provenance_numbers(world, bundle):
    b, _ = bundle
    s = world.prov.summary
    assert b["summary"]["road_km_by_state"] == s.road_km_by_state
    assert b["summary"]["road_segments_by_state"] == s.road_segments_by_state
    assert b["summary"]["settlements_by_access_state"] == s.settlements_by_access_state
    assert b["summary"]["reachable_settlements"] == s.reachable_settlements
    assert b["incident"]["road_rule"] == "v1" and b["incident"]["validated"] is False
    assert b["roads"]["count"] == sum(s.road_segments_by_state.values())
    states = b["roads"]["state"]
    assert states.count(0) == s.road_segments_by_state["blocked"]
    assert len(b["settlements"]) == s.settlements_total
    by_main = b["summary"]["unobservable_km_by_main_reason"]
    assert sum(by_main.values()) == pytest.approx(s.road_km_by_state["unobservable"], abs=0.002)


def test_every_route_trace_matches_the_published_length(world, bundle):
    b, _ = bundle
    length = b["roads"]["length"]
    n = 0
    for st in b["settlements"]:
        for key, tr in st["traces"].items():
            graph = tr["graph"]
            assert st["routes"][graph]["reachable"] and tr["target_id"] == st["routes"][graph]["target_id"]
            assert abs(tr["length_m"] - st["routes"][graph]["length_m"]) <= 0.6
            assert abs(sum(length[e] for e in tr["edges"]) - tr["length_m"]) <= 0.6 + 0.05 * len(tr["edges"])
            if key == "post" and graph == "post_pessimistic":
                assert all(b["roads"]["state"][e] == 2 for e in tr["edges"])        # observed-open roads only
            if graph == "post_optimistic":
                assert all(b["roads"]["state"][e] != 0 for e in tr["edges"])        # never through BLOCKED
            n += 1
    assert n == b["integrity"]["route_traces"]["verified"] and b["integrity"]["route_traces"]["unavailable"] == 0
    c = next(s for s in b["settlements"] if s["id"] == "place-node-4")         # village C behind steep terrain
    assert c["access_state"] == "POSSIBLE_CUTOFF" and c["traces"]["post"]["unobservable_segments"] > 0
    assert c["traces"]["post"]["graph"] == "post_optimistic"
    assert c["distances"]["hospital_pre_event_km"] is not None
    assert c["distances"]["hospital_post_pessimistic_km"] is None              # only via unobservable road


def test_stretches_keep_the_published_answer(world, bundle):
    b, _ = bundle
    top = b["stretches"][0]
    pub = world.result.priorities[0]
    assert top["stretch_id"] == pub.stretch_id and top["settlements_reconnected"] == pub.settlements_reconnected
    assert top["reconnected_settlement_ids"] == list(pub.reconnected_settlement_ids)
    assert set(top["on_verified_pre_event_routes"]) == {"bc-way-1000", "place-node-3"}
    assert all(b["roads"]["stretch"][i] == 0 for i in top["segments"])
    for st in b["stretches"]:
        assert all(e in b["evidence"] for e in st["evidence_ids"])


def test_evidence_records_for_every_clickable_claim(bundle):
    b, _ = bundle
    r = b["roads"]
    for i, s in enumerate(r["state"]):
        if s == 0:
            assert r["evidence"][i] in b["evidence"]
            assert b["evidence"][r["evidence"][i]]["subject_id"] == r["ids"][i]
        if s == 2:
            assert r["evidence"][i] is None
    for st in b["settlements"]:
        if st["access_state"] in ("POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF"):
            assert st["evidence_ids"][0] in b["evidence"]


def test_pockets_and_reason_groups(bundle):
    b, _ = bundle
    assert b["pockets"] and b["pockets"][0]["settlements"] == ["place-node-4"]
    assert b["pockets"][0]["reason_samples"] == {"terrain_rule": b["pockets"][0]["reason_samples"]["terrain_rule"]}
    unobs = [i for i, s in enumerate(b["roads"]["state"]) if s == 1]
    assert all(b["roads"]["reason"][i] == 1 for i in unobs)                    # terrain_rule (steep) in the fixture


def test_no_population_and_building_counts_are_passed_through(world, bundle):
    b, _ = bundle
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k)
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    assert not [k for k in keys(b) if "population" in k.lower()]             # no population field anywhere
    assert any("population" in x for x in b["methodology"]["inputs_not_used"])  # ...only the "not used" record
    pub = {r.candidate.settlement_id: r.status.building_count for r in world.result.settlements}
    assert {s["id"]: s["buildings"] for s in b["settlements"]} == pub


def test_experimental_detector_is_methods_only(bundle):
    b, _ = bundle
    ex = b["experiment"]
    assert ex["used_in_official_result"] is False
    assert ex["verdict"] == "rejected: (d) no edge concentration (control)"
    assert [c["passed"] for c in ex["checks"]] == [True, True, True, True, True, False]


def test_debris_record_cases():
    assert debris_record(None, None)["verdict"] is None
    assert debris_record(DEBRIS_EVENT, None)["verdict"].startswith("not evaluated")
    ok = json.loads(json.dumps(DEBRIS_CONTROL))
    ok["statistics"]["edge_pixels_share"] = 0.01
    assert debris_record(DEBRIS_EVENT, ok)["verdict"] == "all pre-registered checks passed"


def test_rasters_written(bundle):
    b, out = bundle
    for key in ("flood", "not_assessed", "hillshade"):
        ov = b["overlays"][key]
        img = decode_png((out / ov["file"]).read_bytes())
        assert img.shape[:2] == (ov["height"], ov["width"])
        (w, n), (e, s) = ov["corners"][0], ov["corners"][2]
        assert w < e and s < n
    flood = decode_png((out / b["overlays"]["flood"]["file"]).read_bytes())
    assert (flood[..., 3] > 0).any()
    chip = b["stretches"][0]["chip"]
    for f in chip["files"].values():
        assert (out / f).is_file()
    assert chip["vv_db_range"] == list(rs.VV_DB_RANGE)


# ---------------------------------------------------------------- refusal


def test_v2_run_is_never_the_official_result(tmp_path, monkeypatch):
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    monkeypatch.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    w = published_world(tmp_path, extra_v2=True)
    runs = find_connectivity_runs(w.cfg, "aoi-test", "ts-test")
    assert sorted(r.road_rule for r in runs) == ["v1", "v2"]
    chosen, notes = select_official_run(runs)
    assert chosen.road_rule == "v1" and ".superseded-" in chosen.directory.name
    assert any("road rule v2" in n for n in notes)
    with pytest.raises(DashboardBuildError, match="official result is road rule v1"):
        select_official_run(runs, w.out)
    inp = prepare_inputs(w.cfg, "aoi-test", "ts-test", with_rasters=False)
    assert inp.run.directory == chosen.directory


def test_changed_output_is_refused(world, tmp_path):
    copy = tmp_path / world.out.name
    shutil.copytree(world.out, copy)
    seg = copy / "road_segments.geojson"
    seg.write_text(seg.read_text().replace('"blocked"', '"open"', 1))
    run, _ = select_official_run([], copy)
    from bato.dashboard.bundle import verify_run_outputs
    with pytest.raises(DashboardBuildError, match="does not match the SHA-256"):
        verify_run_outputs(run)


def test_summary_disagreement_is_reported(world, bundle):
    b, _ = bundle
    fc = {"features": []}
    published = dict(world.prov.summary.model_dump(mode="json"))
    published["road_km_by_state"] = {**published["road_km_by_state"], "blocked": 99.0}
    from bato.dashboard.bundle import parse_segments
    seg = parse_segments(json.loads((world.out / "road_segments.geojson").read_text()))
    sett = json.loads((world.out / "settlements.geojson").read_text())["features"]
    assert check_summary(world.prov.summary.model_dump(mode="json"), recompute_summary(seg, sett)) == []
    bad = check_summary(published, recompute_summary(seg, sett))
    assert bad and "km blocked" in bad[0]
    assert fc["features"] == []


def test_publish_is_atomic_and_indexed(world):
    out = publish(world.cfg, "aoi-test", "ts-test", inputs_for(world, rasters=False))
    assert (out / "bundle.json").is_file()
    index = json.loads((out.parents[1] / "index.json").read_text())
    assert index["bundles"][0]["path"] == "aoi-test/ts-test"
    out2 = publish(world.cfg, "aoi-test", "ts-test", inputs_for(world, rasters=False),
                   now=lambda: __import__("datetime").datetime(2030, 1, 1))
    assert out2 == out and list(out.parent.glob("ts-test.superseded-*"))
    assert len(json.loads((out.parents[1] / "index.json").read_text())["bundles"]) == 1


# ---------------------------------------------------------------- pure helpers


def test_route_tracer_refuses_a_length_it_cannot_reproduce():
    # square 0-1-2-3-0 with unit lengths; target node 0
    net = build_network([0, 1, 2, 3], [1, 2, 3, 0], [1.0, 1.0, 1.0, 1.0], [2, 2, 1, 2],
                        [[(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)], [(0, 1), (0, 0)]])
    t = RouteTracer(net)
    tr = t.trace("pre_event", "T", 0, 2.0, [2])
    assert tr is not None and tr.length_m == 2.0 and len(tr.edges) == 2
    assert t.trace("pre_event", "T", 0, 2.7, [2]) is None
    pess = t.trace("post_pessimistic", "T", 0, 2.0, [2])                 # edge 2 (unobservable) is removed
    assert pess is not None and pess.km_by_state["unobservable"] == 0
    assert unobservable_pockets(net, [2]) == [[2]]


def test_png_roundtrip():
    for shape in ((3, 4), (3, 4, 2), (3, 4, 3), (5, 2, 4)):
        a = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
        assert np.array_equal(decode_png(encode_png(a)), a)
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2), dtype=np.float32))


def test_block_fractions_and_overlays():
    state = np.zeros((10, 10), dtype=np.uint8)
    state[:5, :5] = 1
    state[5:, 5:] = 14
    fr = rs.block_fractions(state, 5, {"f": (1,), "s": (14,)})
    assert fr["f"].tolist() == [[1.0, 0.0], [0.0, 0.0]] and fr["s"][1, 1] == 1.0
    flood, na = rs.evidence_overlays(state, 5)
    assert flood[0, 0, 3] > 0 and flood[1, 1, 3] == 0 and na[1, 1, 3] > 0 and na[0, 0, 3] == 0


def test_hillshade_lights_north_west_slopes():
    lat = np.linspace(28.1, 28.0, 50)
    lon = np.linspace(85.0, 85.1, 50)
    z = np.tile(lon * 0 + 0, (50, 1)) + (lat[:, None] - 28.0) * 20000      # rises to the north: faces south
    t = (0.1 / 50, 0.0, 85.0, 0.0, -0.1 / 50, 28.1)
    south_facing = rs.hillshade(z, t, None).mean()
    north_facing = rs.hillshade(-z, t, None).mean()
    assert north_facing > south_facing                                      # light from the north-west
    img = rs.to_mercator_rows(np.zeros((100, 300), dtype=np.uint8), 28.1, 28.0, 150)
    assert img.shape == (50, 150)
    rgba = rs.shade_rgba(np.array([[0, 181, 250, 40]], dtype=np.uint8))
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 0 and rgba[0, 2, 3] > 0 and rgba[0, 3, 3] > 0


def test_chip_helpers():
    t = (10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    from bato.inference.diagnostics import lonlat_from_3857
    lon, lat = lonlat_from_3857(np.array([55.0]), np.array([-55.0]))
    assert rs.chip_window(t, (100, 100), float(lon[0]), float(lat[0]), 3) == (2, 2, 6, 6)
    assert rs.chip_window(t, (100, 100), 50.0, 50.0, 3) is None
    g = rs.vv_to_gray(np.array([[0.0, 0.01, 1.0]], dtype=np.float32))
    assert g[0, 0, 1] == 0 and g[0, 1, 1] == 255 and g[0, 2, 0] == 255
    ch = rs.change_to_rgba(np.array([[0.01, 0.1]]), np.array([[0.1, 0.01]]))
    assert ch[0, 0, 2] > ch[0, 0, 0] and ch[0, 1, 0] > 200                # darkening blue, brightening warm

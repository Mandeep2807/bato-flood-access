"""Pre-event OSM extraction via the ohsome API v2 (fake transport and fake GeoParquet bytes; TEST FIXTURE
layers, not real OSM data). A genuine-GeoParquet round trip runs when pyarrow is installed."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from bato.contracts.osm import OSMExtractProvenance, OSMLayerName
from bato.osm.extract import (
    BUILDING_FILTER,
    ENDPOINT,
    HOSPITAL_FILTER,
    OSMError,
    api_key_from_env,
    extract_osm,
    load_osm_extract,
    plan_requests,
    rows_to_features,
)
from bato.osm.geometry import WKBError, centroid, wkb_to_geojson

from connectivity_fakes import (
    ANALYSIS,
    AOI,
    EVENT,
    NOW,
    FakeOhsome,
    fake_read_parquet_rows,
    fetch,
    project,
    to_rows,
    wkb,
)

SNAP = date(2026, 8, 25)


def _row(osm_id="node/1", edited="2026-08-24T10:00:00+00:00", valid_to="2222-01-01T00:00:00+00:00",
         geometry=None, **tags):
    feature = {"type": "Feature", "geometry": geometry or {"type": "Point", "coordinates": [85.0, 28.0]},
               "properties": {"@osmId": osm_id, **tags}}
    from connectivity_fakes import fake_parquet
    return fake_read_parquet_rows(fake_parquet(to_rows([feature], edited=edited, valid_to=valid_to)))


def test_v2_requests_are_json_with_key_header_snapshot_instant_and_unchanged_filters(tmp_path):
    cfg = project(tmp_path)
    transport = FakeOhsome()
    out, prov = fetch(cfg, transport, api_key="SECRET-KEY")
    assert prov.snapshot.snapshot_date == SNAP < EVENT                       # event - 1 day
    meta = [r for r in transport.requests if r[0].endswith("/metadata")]
    assert meta and meta[0][0] == "https://api.heigit.org/ohsome-api/v2-rc/metadata"
    posts = [(url, body, headers) for url, body, headers in transport.requests if body]
    assert len(posts) == 4
    assert {url for url, _, _ in posts} == {"https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"}
    assert all(h == {"authorization": "SECRET-KEY"} for _, _, h in posts)    # raw key, documented header
    assert {b["time"] for _, b, _ in posts} == {"2026-08-25T00:00:00Z"}
    assert all(b["clip"] is True and set(b) == {"filter", "aoi", "time", "clip"} for _, b, _ in posts)
    filters = {b["filter"] for _, b, _ in posts}
    assert HOSPITAL_FILTER in filters and BUILDING_FILTER in filters
    roads = next(b for _, b, _ in posts if b["filter"].startswith("type:way and highway in ("))
    assert "track" in roads["filter"] and "footway" not in roads["filter"] and "path" not in roads["filter"]
    assert roads["aoi"] == [ANALYSIS.min_lon, ANALYSIS.min_lat, ANALYSIS.max_lon, ANALYSIS.max_lat]
    buildings = next(b for _, b, _ in posts if b["filter"] == BUILDING_FILTER)
    assert buildings["aoi"] != roads["aoi"]                                    # AOI + building margin only
    assert {r.layer for r in prov.layers} == set(OSMLayerName)
    assert sorted(p.name for p in out.iterdir()) == ["buildings.parquet", "hospitals.parquet",
                                                    "osm_extract_provenance.json", "places.parquet", "roads.parquet"]
    assert "SECRET-KEY" not in (out / "osm_extract_provenance.json").read_text()   # key never stored
    assert out.name == "snapshot-2026-08-25" and out.parent.name == "aoi-test"


def test_api_key_comes_from_the_environment_only(cfg):
    assert cfg.settings.osm.api_key_env == "OHSOME_API_KEY"
    assert api_key_from_env(cfg, {"OHSOME_API_KEY": " abc "}) == "abc"
    with pytest.raises(OSMError, match="set the environment variable OHSOME_API_KEY"):
        api_key_from_env(cfg, {})


def test_missing_or_rejected_key_publishes_nothing(tmp_path):
    cfg = project(tmp_path)
    with pytest.raises(OSMError, match="no ohsome API key"):
        fetch(cfg, api_key="")
    t = FakeOhsome(status=403, overrides={"roads": b'{"error": "invalid API key"}'})
    with pytest.raises(OSMError, match="HTTP 403: invalid API key .check the API key in OHSOME_API_KEY"):
        fetch(cfg, t)
    assert not (cfg.settings.paths.raw / "osm" / "aoi-test" / "snapshot-2026-08-25").exists()


def test_snapshot_is_capped_at_the_latest_ohsome_date(tmp_path):
    cfg = project(tmp_path)
    t = FakeOhsome(latest="2026-08-10T20:00:00Z", edited="2026-08-01T00:00:00+00:00")
    _, prov = fetch(cfg, t)
    assert prov.snapshot.snapshot_date == date(2026, 8, 10) and prov.snapshot.latest_available == date(2026, 8, 10)
    assert {b["time"] for _, b, _ in t.requests if b} == {"2026-08-10T00:00:00Z"}


def test_feature_edited_after_the_snapshot_instant_is_refused():
    with pytest.raises(OSMError, match="post-event OSM is prohibited"):
        rows_to_features(_row(edited="2026-08-25T06:00:00+00:00"), SNAP, False, "test")
    assert len(rows_to_features(_row(edited="2026-08-24T23:59:59+00:00"), SNAP, False, "test")) == 1


def test_version_that_ended_before_the_snapshot_is_refused():
    with pytest.raises(OSMError, match="not the OSM state at the snapshot"):
        rows_to_features(_row(valid_to="2026-08-20T00:00:00+00:00"), SNAP, False, "test")


def test_post_event_rows_in_a_response_publish_nothing(tmp_path):
    cfg = project(tmp_path)
    with pytest.raises(OSMError, match="post-event OSM is prohibited"):
        fetch(cfg, FakeOhsome(edited="2026-08-27T00:00:00+00:00"))
    assert not (cfg.settings.paths.raw / "osm" / "aoi-test" / "snapshot-2026-08-25").exists()


def test_rows_become_the_feature_dicts_the_pipeline_reads():
    rows = _row("way/42", geometry={"type": "LineString", "coordinates": [[85.0, 28.0], [85.01, 28.0]]},
                highway="track", bridge="yes")
    (f,) = rows_to_features(rows, SNAP, False, "test")
    assert f["geometry"] == {"type": "LineString", "coordinates": [[85.0, 28.0], [85.01, 28.0]]}
    assert f["properties"] == {"@osmId": "way/42", "@lastEdit": "2026-08-24T10:00:00Z", "highway": "track",
                               "bridge": "yes"}


def test_centroid_layers_get_client_side_centroids():
    square = {"type": "Polygon", "coordinates": [[[85.0, 28.0], [85.002, 28.0], [85.002, 28.002], [85.0, 28.002],
                                                   [85.0, 28.0]]]}
    (f,) = rows_to_features(_row("way/7", geometry=square, building="yes"), SNAP, True, "test")
    assert f["geometry"]["type"] == "Point"
    assert f["geometry"]["coordinates"] == pytest.approx([85.001, 28.001])


def test_wkb_decoding_and_centroids():
    line = {"type": "MultiLineString", "coordinates": [[[0.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [0.0, 3.0]]]}
    assert wkb_to_geojson(wkb(line)) == line
    holed = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                                                [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]}
    x, y = centroid(wkb_to_geojson(wkb(holed)))
    assert (x, y) == pytest.approx((7 / 3, 7 / 3))                           # (16 * 2 - 4 * 1) / 12
    assert centroid({"type": "LineString", "coordinates": [[0, 0], [2, 0]]}) == (1.0, 0.0)
    big_endian_point = bytes([0]) + (1).to_bytes(4, "big") + bytes.fromhex("4055400000000000403c000000000000")
    assert wkb_to_geojson(big_endian_point) == {"type": "Point", "coordinates": [85.0, 28.0]}
    with pytest.raises(WKBError):
        wkb_to_geojson(b"\x01\x07\x00\x00\x00")


def test_http_error_publishes_nothing(tmp_path):
    cfg = project(tmp_path)
    with pytest.raises(OSMError, match="HTTP 400: bad filter"):
        fetch(cfg, FakeOhsome(status=400, overrides={"roads": b'{"error": "bad filter"}'}))
    with pytest.raises(OSMError, match="no pre-event OSM extract"):
        load_osm_extract(cfg, "aoi-test", EVENT)


def test_existing_extract_needs_overwrite_and_is_superseded(tmp_path):
    cfg = project(tmp_path)
    out, _ = fetch(cfg)
    with pytest.raises(OSMError, match="already exists"):
        fetch(cfg)
    out2, _ = fetch(cfg, overwrite=True)
    assert out2 == out and list(out.parent.glob("snapshot-2026-08-25.superseded-*"))


def test_loader_verifies_hashes_and_event(tmp_path):
    cfg = project(tmp_path)
    out, _ = fetch(cfg)
    extract = load_osm_extract(cfg, "aoi-test", EVENT)
    roads = extract.layers[OSMLayerName.ROADS]
    assert len(roads) == 7 and roads[0]["properties"]["@osmId"].startswith(("way/", "node/"))
    with pytest.raises(OSMError, match="no pre-event OSM extract for event 2026-08-20"):
        load_osm_extract(cfg, "aoi-test", date(2026, 8, 20))
    (out / "roads.parquet").write_bytes((out / "roads.parquet").read_bytes() + b" ")
    with pytest.raises(OSMError, match="changed since extraction"):
        load_osm_extract(cfg, "aoi-test", EVENT)


def test_provenance_rejects_post_event_snapshot_and_wrong_request_time(tmp_path):
    cfg = project(tmp_path)
    _, prov = fetch(cfg)
    data = prov.model_dump(mode="json")
    data["snapshot"]["snapshot_date"] = "2026-08-26"
    with pytest.raises(ValueError, match="strictly before the event"):
        OSMExtractProvenance.model_validate(data)
    data = prov.model_dump(mode="json")
    data["layers"][0]["time"] = "latest"
    with pytest.raises(ValueError, match="snapshot instant"):
        OSMExtractProvenance.model_validate(data)


def test_plan_requests_cover_every_layer(cfg):
    reqs = plan_requests(cfg, AOI, ANALYSIS)
    assert [r.layer for r in reqs] == list(OSMLayerName)
    assert all(r.endpoint == ENDPOINT == "extraction/features.parquet" for r in reqs)
    assert [r.centroids for r in reqs] == [False, True, True, True]
    assert json.dumps(reqs[0].body(SNAP))                                     # JSON-serialisable body


def test_loader_selects_the_extract_of_the_requested_event(tmp_path):
    cfg = project(tmp_path)
    fetch(cfg)                                                                 # event 08-26 -> snapshot 08-25
    extract_osm(cfg, aoi_id="aoi-test", event_date=date(2026, 8, 20), aoi_bbox=AOI, analysis_bbox=ANALYSIS,
                transport=FakeOhsome(), api_key="TEST-KEY", now=lambda: NOW)   # snapshot 08-19
    assert load_osm_extract(cfg, "aoi-test", EVENT).provenance.snapshot.snapshot_date == SNAP
    assert load_osm_extract(cfg, "aoi-test", date(2026, 8, 20)).provenance.snapshot.snapshot_date == \
        date(2026, 8, 19)


def test_genuine_geoparquet_with_the_documented_schema(tmp_path):
    """Real pyarrow path (skipped where pyarrow is not installed): documented v2 column types."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from bato.osm import extract

    ts = pa.timestamp("us", tz="UTC")
    table = pa.table({
        "osm_type": pa.array(["way"]), "osm_id": pa.array([42], pa.int64()),
        "edit_timestamp": pa.array([datetime(2026, 8, 24, 10, tzinfo=timezone.utc)], ts),
        "valid_to_timestamp": pa.array([datetime(2222, 1, 1, tzinfo=timezone.utc)], ts),
        "version": pa.array([3], pa.int32()),
        "tags": pa.array([[("highway", "track"), ("name", "X")]], pa.map_(pa.string(), pa.string())),
        "geom": pa.array([wkb({"type": "LineString", "coordinates": [[85.0, 28.0], [85.01, 28.0]]})], pa.binary()),
        "geom_type": pa.array(["LineString"]), "clipped": pa.array([False]),
    })
    path = tmp_path / "roads.parquet"
    pq.write_table(table, path)
    rows = extract.__dict__["read_parquet_rows"](path.read_bytes())          # patched: delegates to pyarrow
    (f,) = rows_to_features(rows, SNAP, False, "test")
    assert f["properties"] == {"@osmId": "way/42", "@lastEdit": "2026-08-24T10:00:00Z", "highway": "track",
                               "name": "X"}
    assert f["geometry"]["type"] == "LineString"

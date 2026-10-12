"""bato.evaluation (EMSR927 comparison): geometry, metrics, record parsing, time gaps, product reading, protocol.
All data are synthetic (tests/ems_fakes.py)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pytest

from bato.contracts.baseline import BaselineState
from bato.evaluation.ems_activation import (
    ActivationRecordError,
    EmsImage,
    EmsProduct,
    aoi_overlaps,
    check_download_url,
    parse_activation,
    plan_downloads,
)
from bato.evaluation.ems_product import ProductError, ProductZip, inspect_product, tiff_info
from bato.evaluation.geometry import (
    GeometryError,
    Grid,
    clip_ring_convex,
    intersection_area_km2,
    parse_wkt_polygons,
    polygon_area_km2,
    rasterize,
    ring_area_km2,
)
from bato.evaluation.metrics import NOT_CALCULABLE, MetricError, confusion, pixel_comparison, scores, tolerant
from bato.evaluation.protocol import ProtocolError, check_frozen, freeze, load_protocol
from bato.evaluation.roads import road_comparison
from bato.evaluation.timegap import TimeGapError, event_dates, product_gaps, readings, select_product
from bato.impact.roads import lonlat_to_3857

from ems_fakes import (
    EMS_AOI_BOX,
    activation,
    box_mask,
    box_ring,
    fc,
    feature,
    install_draft_protocol,
    product_zip,
    shapefile_bytes,
)
from helpers import REPO_CONFIG_DIR

HOST = "rapidmapping.emergency.copernicus.eu"
POST = datetime(2026, 8, 28, 12, 21, 41, tzinfo=timezone.utc)
ZONES = readings({"ems_time_zone": "undetermined", "readings": {"UTC": "+00:00", "Asia/Kathmandu": "+05:45"}})


def small_grid(lon0: float = 85.0, lat0: float = 28.05, n: int = 120) -> Grid:
    x, y = lonlat_to_3857(np.array([lon0]), np.array([lat0]))
    return Grid((10.0, 0.0, float(x[0]), 0.0, -10.0, float(y[0])), n, n)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_wkt_polygon_multipolygon_and_srid():
    p = parse_wkt_polygons("SRID=4326;POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0), (0.2 0.2, 0.4 0.2, 0.4 0.4, 0.2 0.2))")
    assert len(p) == 1 and len(p[0]) == 2
    m = parse_wkt_polygons("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))")
    assert len(m) == 2
    with pytest.raises(GeometryError):
        parse_wkt_polygons("LINESTRING (0 0, 1 1)")


def test_spherical_area_one_degree_cell_at_equator():
    a = ring_area_km2([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert a == pytest.approx(12363.7, rel=2e-3)                     # 1 deg x 1 deg at the equator


def test_convex_clip_area_and_refusal_of_concave_clip():
    half = [(0.5, -1), (2, -1), (2, 2), (0.5, 2), (0.5, -1)]
    unit = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]
    assert intersection_area_km2([unit], half) == pytest.approx(polygon_area_km2(unit) / 2, rel=2e-3)
    far = [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]
    assert intersection_area_km2([unit], far) == 0
    concave = [(0, 0), (2, 0), (1, 0.5), (2, 1), (0, 1), (0, 0)]
    with pytest.raises(GeometryError):
        clip_ring_convex(unit[0], concave)


def test_rasterize_pixel_centres_match_brute_force_and_hole_is_excluded():
    g = small_grid()
    outer = (85.002, 28.042, 85.009, 28.048)
    hole = (85.004, 28.044, 85.006, 28.046)
    poly = [[tuple(p) for p in box_ring(outer)], [tuple(p) for p in box_ring(hole)]]
    m = rasterize([poly], g)
    expect = box_mask(g, outer) & ~box_mask(g, hole)
    assert m.sum() > 1000 and np.array_equal(m, expect)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_confusion_scores_and_not_calculable():
    pred = np.array([[1, 1, 0, 0]], bool)
    ref = np.array([[1, 0, 1, 0]], bool)
    inc = np.ones_like(pred)
    c = confusion(pred, ref, inc)
    assert c == {"TP": 1, "FP": 1, "FN": 1, "TN": 1, "N": 4}
    s = scores(c)
    assert s["precision"]["value"] == 0.5 and s["iou"]["value"] == pytest.approx(1 / 3, abs=1e-6)
    z = scores(confusion(np.zeros_like(pred), np.zeros_like(ref), inc))
    assert all(z[k]["value"] == NOT_CALCULABLE and z[k]["reason"] for k in ("precision", "recall", "f1", "iou"))


def test_modes_count_not_assessed_as_negative_only_in_all_pixels_mode():
    state = np.zeros((4, 4), np.uint8)
    state[0, 0] = BaselineState.FLOOD
    state[1, 1] = BaselineState.NOT_ASSESSED_STEEP
    state[2, 2] = BaselineState.NOT_ASSESSED_LAYOVER
    ref = np.zeros((4, 4), bool)
    ref[0, 0] = ref[1, 1] = True
    area = np.ones((4, 4), bool)
    r = pixel_comparison(state, ref, area, tolerance_px=1, row_ground_area_m2=np.full(4, 100.0))
    a, b = r["modes"]["assessed_only"]["counts"], r["modes"]["all_pixels"]["counts"]
    assert a == {"TP": 1, "FP": 0, "FN": 0, "TN": 13, "N": 14}
    assert b == {"TP": 1, "FP": 0, "FN": 1, "TN": 14, "N": 16}
    assert r["reference_pixels_not_assessed"] == 1
    assert r["reference_pixels_not_assessed_by_reason"]["NOT_ASSESSED_STEEP"] == 1
    assert r["observable_fraction"]["value"] == pytest.approx(14 / 16)
    assert r["comparison_area_km2_ground_approx"] == pytest.approx(16 * 100 / 1e6, abs=1e-4)
    assert "BATO negative" in r["modes"]["all_pixels"]["not_assessed_handling"]


def test_tolerant_metrics_forgive_one_pixel_shift_only():
    pred = np.zeros((6, 6), bool)
    pred[2, 2] = True
    ref1 = np.zeros((6, 6), bool)
    ref1[2, 3] = True
    ref2 = np.zeros((6, 6), bool)
    ref2[2, 4] = True
    inc = np.ones((6, 6), bool)
    assert tolerant(pred, ref1, inc, 1)["precision_tol"]["value"] == 1.0
    assert tolerant(pred, ref2, inc, 1)["precision_tol"]["value"] == 0.0
    assert tolerant(pred, ref1, inc, 1)["iou_tol"]["value"] == NOT_CALCULABLE


def test_unknown_state_code_fails_closed():
    with pytest.raises(MetricError):
        pixel_comparison(np.full((2, 2), 7, np.uint8), np.zeros((2, 2), bool), np.ones((2, 2), bool),
                         tolerance_px=1, row_ground_area_m2=np.full(2, 100.0))


ROAD_PARAMS = {"match_distance_m": 20, "min_matched_length_m": 25, "sample_step_m": 5, "min_inside_fraction": 0.5}
PRIMARY = {"role": "primary", "label": "BLOCKED vs DESTROYED", "positive": ["destroyed"],
           "negative": ["no_visible_damage"], "excluded": ["damaged", "possibly_damaged", "not_analysed"]}
SUPPLEMENTARY = {"role": "supplementary", "label": "BLOCKED vs any damage",
                 "positive": ["destroyed", "damaged", "possibly_damaged"], "negative": ["no_visible_damage"],
                 "excluded": ["not_analysed"]}


def test_road_segments_matched_to_ems_damage_lines_under_two_comparisons():
    g = small_grid(85.0, 28.05, 400)
    area = np.ones(g.shape, bool)
    seg = lambda sid, st, lat: {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [  # noqa: E731
        [85.002, lat], [85.008, lat]]}, "properties": {"segment_id": sid, "state": st, "length_m": 588.0}}
    segments = {"features": [seg("a", "blocked", 28.045), seg("b", "open", 28.040), seg("c", "unobservable", 28.035),
                             seg("d", "open", 28.030), seg("e", "blocked", 28.025)]}
    line = lambda lat: {"type": "LineString", "coordinates": [[85.002, lat], [85.008, lat]]}  # noqa: E731
    ems = [feature(line(28.04505)), feature(line(28.0400)), feature(line(28.03505)), feature(line(28.03005)),
           feature(line(28.02505)), feature(line(28.02503))]
    grades = ["destroyed", "no_visible_damage", "destroyed", "possibly_damaged", "possibly_damaged",
              "no_visible_damage"]
    r = road_comparison(segments, ems, grades, area, g, ROAD_PARAMS,
                        {"primary_destroyed": PRIMARY, "supplementary_any_damage": SUPPLEMENTARY})
    pri, sup = r["comparisons"]["primary_destroyed"], r["comparisons"]["supplementary_any_damage"]
    # e is near both a possibly-damaged and a no-visible-damage line: excluded (ambiguous) under the primary,
    # positive under the supplementary; never silently negative.
    assert {x["segment_id"]: x["reference"] for x in pri["segments"]} == \
        {"a": "positive", "b": "negative", "c": "positive", "d": "excluded", "e": "excluded"}
    assert {x["segment_id"]: x["reference"] for x in sup["segments"]} == \
        {"a": "positive", "b": "negative", "c": "positive", "d": "positive", "e": "positive"}
    assert pri["modes"]["observed_only"]["counts"] == {"TP": 1, "FP": 0, "FN": 0, "TN": 1, "N": 2}
    assert pri["modes"]["all_segments"]["counts"] == {"TP": 1, "FP": 0, "FN": 1, "TN": 1, "N": 3}
    assert sup["modes"]["all_segments"]["counts"] == {"TP": 2, "FP": 0, "FN": 2, "TN": 1, "N": 5}
    assert pri["role"] == "primary" and sup["role"] == "supplementary"
    d = pri["descriptive_lengths_m"]
    assert d["ems_positive_features_in_area"] == 2
    assert d["bato_blocked_length_within_distance_of_ems_positive"] == pytest.approx(588, abs=8)
    assert d["ems_positive_length_within_distance_of_bato_blocked"] == pytest.approx(588, abs=8)
    assert r["bato_segments_in_area"] == {"blocked": 2, "open": 2, "unobservable": 1}


def test_short_segment_needs_its_whole_length_matched():
    g = small_grid(85.0, 28.05, 200)
    dlon = 1.0 / (111320 * np.cos(np.radians(28.045)))
    seg = {"features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [
        [85.002, 28.045], [85.002 + 15 * dlon, 28.045]]}, "properties": {"segment_id": "s", "state": "blocked"}}]}
    ems = [feature({"type": "LineString", "coordinates": [[85.0015, 28.04503], [85.0025 + 20 * dlon, 28.04503]]})]
    r = road_comparison(seg, ems, ["destroyed"], np.ones(g.shape, bool), g, ROAD_PARAMS, {"p": PRIMARY})
    assert r["comparisons"]["p"]["segments"][0]["reference"] == "positive"          # 15 m segment, all matched


# ---------------------------------------------------------------------------
# Activation record
# ---------------------------------------------------------------------------


def test_parse_activation_record_and_download_plan():
    rec = parse_activation(activation({}), "EMSR927")
    assert rec.event_time == "2026-08-25T22:00:00" and [a.number for a in rec.aois] == [1, 2, 3]
    ring = [(85.015, 28.015), (85.045, 28.015), (85.045, 28.045), (85.015, 28.045), (85.015, 28.015)]
    ov = {o.number: o for o in aoi_overlaps(rec, ring)}
    assert ov[1].overlap_km2 > 0 and ov[2].overlap_km2 == 0
    plan = {d.product.label: d for d in plan_downloads(rec, list(ov.values()), required_status="F",
                                                         allowed_host=HOST)}
    assert plan["AOI01 GRA PRODUCT v1"].download and plan["AOI01 GRA MONIT01 v1"].download
    assert not plan["AOI02 GRA PRODUCT v1"].download and "does not overlap" in plan["AOI02 GRA PRODUCT v1"].reason
    assert not plan["AOI03 GRA PRODUCT v1"].download and "not final" in plan["AOI03 GRA PRODUCT v1"].reason


def test_activation_record_missing_field_and_bad_urls_fail_closed():
    bad = activation({})
    del bad["results"][0]["aois"][0]["products"][0]["type"]
    with pytest.raises(ActivationRecordError):
        parse_activation(bad, "EMSR927")
    with pytest.raises(ActivationRecordError):
        parse_activation(activation({}), "EMSR999")
    for url in ("http://rapidmapping.emergency.copernicus.eu/backend/EMSR927/x/EMSR927_A.zip",
                "https://example.org/EMSR927_A.zip",
                "https://rapidmapping.emergency.copernicus.eu/backend/EMSR927/x/EMSR927_A.pdf",
                "https://rapidmapping.emergency.copernicus.eu/backend/EMSR111/x/EMSR111_A.zip"):
        with pytest.raises(ActivationRecordError):
            check_download_url(url, HOST, "EMSR927")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def prod(aoi: int, mon: int, ver: int, times: list[str]) -> EmsProduct:
    return EmsProduct(aoi, "x", "GRA", mon > 0, mon, True, ver, "F", None, "u",
                      tuple(EmsImage("s", t) for t in times))


def test_time_gaps_under_both_readings_and_selection_rule():
    base = prod(3, 0, 1, ["2026-08-27T04:22:00", "2026-08-27T12:09:00"])
    mon = prod(3, 1, 1, ["2026-08-27T04:47:00", "2026-08-28T08:29:00"])
    g = product_gaps(mon, POST, ZONES)
    assert g["images"][1]["gap_hours"] == {"UTC": -3.878, "Asia/Kathmandu": -9.628}
    picked, info = select_product([base, mon], POST, ZONES)
    assert picked is mon and set(info["selected_under"]) == {"UTC", "Asia/Kathmandu"}


def test_selection_that_depends_on_the_time_zone_fails_closed():
    a = prod(1, 0, 1, ["2026-08-28T10:21:41"])       # UTC: -2 h | NPT: -7.75 h
    b = prod(1, 1, 1, ["2026-08-28T16:21:41"])       # UTC: +4 h | NPT: -1.75 h
    with pytest.raises(TimeGapError):
        select_product([a, b], POST, ZONES)


def test_zone_needs_evidence_and_event_dates():
    with pytest.raises(TimeGapError):
        readings({"ems_time_zone": "UTC", "ems_time_zone_evidence": None, "readings": {"UTC": "+00:00"}})
    d = event_dates("2026-08-25T22:00:00", ZONES)
    assert d["UTC"]["nepal_date"] == "2026-08-26" and d["Asia/Kathmandu"]["nepal_date"] == "2026-08-25"


# ---------------------------------------------------------------------------
# Product zips
# ---------------------------------------------------------------------------


def write_zip(tmp_path, members: dict[str, bytes]):
    p = tmp_path / "EMSR927_AOI01_GRA_PRODUCT_v1.zip"
    with zipfile.ZipFile(p, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return p


def test_inspect_reports_layers_classes_and_time_zone_clues(tmp_path):
    p = tmp_path / "EMSR927_AOI01_GRA_PRODUCT_v1.zip"
    p.write_bytes(product_zip("EMSR927_AOI01_GRA_PRODUCT", 1))
    r = inspect_product(p)
    layers = {ly["stem"]: ly for ly in r["layers"]}
    assert set(layers) == {"areaOfInterestA", "observedEventA", "transportationL", "transportationP",
                           "transportationA", "builtUpP", "facilitiesA", "facilitiesL", "imageFootprintA",
                           "notAnalysedA"}
    ev = layers["observedEventA"]
    assert ev["wgs84_lonlat"] and ev["geometry_types"] == {"Polygon": 1}
    assert {v["value"] for v in ev["properties"]["obj_desc"]["values"]} == {"Landslide"}
    assert {v["value"] for v in layers["transportationL"]["properties"]["damage_gra"]["values"]} == \
        {"Destroyed", "Damaged", "Possibly damaged", "No visible damage", "Not Analysed"}
    assert any(c["match"].upper() == "UTC" for c in r["time_zone_clues"])
    assert any(d["path"].endswith(".pdf") for d in r["documents"])


def test_shapefile_nested_zip_and_projected_crs(tmp_path):
    shp = shapefile_bytes([[(85.0, 28.0), (85.1, 28.0), (85.1, 28.1), (85.0, 28.1), (85.0, 28.0)]],
                          [{"obj_desc": "Flooded area"}])
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for ext, data in shp.items():
            z.writestr(f"EMSR927_AOI01_GRA_PRODUCT_observedEventA_v1.{ext}", data)
    projected = fc([feature({"type": "Polygon", "coordinates": [[[950000, 3250000], [951000, 3250000],
                                                                  [951000, 3251000], [950000, 3250000]]]})],
                   crs="urn:ogc:def:crs:EPSG::32645")
    p = write_zip(tmp_path, {"vector.zip": inner.getvalue(),
                             "EMSR927_AOI01_GRA_PRODUCT_areaOfInterestA_v1.json": projected})
    pz = ProductZip(p)
    layer = pz.load_lonlat("observedEventA")
    assert layer.format == "shapefile" and layer.features[0]["properties"]["obj_desc"] == "Flooded area"
    assert layer.geometry_types() == {"Polygon": 1}
    with pytest.raises(ProductError):
        pz.load_lonlat("areaOfInterestA")


def test_tiff_header_reader():
    import struct
    entries = [(256, 3, 1, 50), (257, 3, 1, 40), (258, 3, 1, 8)]
    ifd = struct.pack("<H", len(entries)) + b"".join(struct.pack("<HHIHH", t, ty, c, v, 0)
                                                     for t, ty, c, v in entries) + struct.pack("<I", 0)
    data = b"II" + struct.pack("<HI", 42, 8) + ifd
    info = tiff_info(data)
    assert info["width"] == 50 and info["height"] == 40


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def protocol_root(tmp_path):
    import shutil
    shutil.copytree(REPO_CONFIG_DIR, tmp_path / "config")
    install_draft_protocol(tmp_path)
    return tmp_path


def test_repository_protocol_keeps_its_pre_registered_parameters():
    """Fixed before any product was seen; mappings are filled after inspection, these values never change."""
    p = load_protocol(REPO_CONFIG_DIR.parent)
    assert p.status in ("draft", "frozen") and p.protocol_id == "bato-emsr927-v1"
    assert p.raw["pixels"]["tolerance_px"] == 1
    assert p.raw["roads"] == {"match_distance_m": 20.0, "min_matched_length_m": 25.0, "sample_step_m": 5.0,
                              "min_inside_fraction": 0.5}
    assert p.raw["time"]["readings"] == {"UTC": "+00:00", "Asia/Kathmandu": "+05:45"}
    assert p.raw["reference"]["required_status_code"] == "F"
    assert p.raw["reference"]["product_selection"] == "min_worst_case_abs_gap_to_post"
    if p.status == "draft":
        with pytest.raises(ProtocolError):
            check_frozen(p, REPO_CONFIG_DIR.parent / "nowhere")


def test_repository_protocol_maps_every_inspected_emsr927_layer_and_value():
    """Mapping completed from the inspection of the official zips (2026-10-05). No flood class exists or is mapped."""
    p = load_protocol(REPO_CONFIG_DIR.parent)
    roles = {stem: p.layer_role(stem) for stem in (
        "areaOfInterestA", "observedEventA", "transportationL", "transportationP", "notAnalysedA", "builtUpP",
        "facilitiesA", "facilitiesL", "imageFootprintA", "transportationA")}
    assert roles == {"areaOfInterestA": "area_of_interest", "observedEventA": "observed_event",
                     "transportationL": "transportation", "transportationP": "bridge_context",
                     "notAnalysedA": "not_analysed", "builtUpP": "not_used", "facilitiesA": "not_used",
                     "facilitiesL": "not_used", "imageFootprintA": "not_used", "transportationA": "not_used"}
    assert "building-impact prediction layer" in p.raw["layers"]["not_used"]["builtUpP"]
    assert p.raw["observed_event"]["classes"] == {"Landslide": "reference_mass_movement"}
    assert "flood_water" not in p.raw["observed_event"]["classes"].values()
    assert p.raw["transportation"]["classes"] == {
        "Destroyed": "destroyed", "Damaged": "damaged", "Possibly damaged": "possibly_damaged",
        "No visible damage": "no_visible_damage", "Not Analysed": "not_analysed"}
    comps = p.road_comparisons()
    assert comps["primary_destroyed"]["role"] == "primary" and comps["primary_destroyed"]["positive"] == ["destroyed"]
    assert comps["supplementary_any_damage"]["positive"] == ["destroyed", "damaged", "possibly_damaged"]
    assert "not accuracy" in comps["supplementary_any_damage"]["label"]
    for value in ("Destroyed", "Possibly damaged"):
        assert p.bridge_grade(value)
    with pytest.raises(ProtocolError):
        p.road_grade("Severely damaged")                                     # unexpected value -> fail closed
    with pytest.raises(ProtocolError):
        p.bridge_grade("Damaged")                                            # not inspected for bridges


@pytest.mark.parametrize("edit,match", [
    ("      excluded: [damaged, possibly_damaged, not_analysed]\n", "exactly one of"),       # grades dropped
    ("      role: supplementary\n", "exactly one road comparison"),                       # two primaries
])
def test_road_comparisons_must_partition_grades_and_have_one_primary(tmp_path, edit, match):
    root = protocol_root(tmp_path)
    y = root / "config" / "evaluation" / "emsr927_protocol.yaml"
    text = y.read_text(encoding="utf-8")
    new = text.replace(edit, "      excluded: [not_analysed]\n" if "excluded" in edit else "      role: primary\n", 1)
    assert new != text
    y.write_text(new, encoding="utf-8")
    with pytest.raises(ProtocolError, match=match):
        load_protocol(root)


def test_mass_movement_overlap_is_descriptive_only():
    from bato.evaluation.metrics import footprint_overlap
    state = np.zeros((4, 4), np.uint8)
    state[0, :2] = BaselineState.FLOOD
    state[3, 3] = BaselineState.NOT_ASSESSED_STEEP
    ref = np.zeros((4, 4), bool)
    ref[0, 1] = ref[1, 1] = ref[3, 3] = True
    r = footprint_overlap(state, ref, np.ones((4, 4), bool), row_ground_area_m2=np.full(4, 100.0))
    assert "CLASS-MISMATCHED FOOTPRINT OVERLAP" in r["label"]
    assert r["overlap_pixels"] == 1 and r["footprint_share_marked_bato_flood"]["value"] == pytest.approx(1 / 3, 1e-5)
    assert r["footprint_pixels_by_state"]["NOT_ASSESSED_STEEP"] == 1
    keys = json.dumps(r).lower()
    for banned in ("precision", "recall", "f1", "iou", "accuracy\""):
        assert banned not in keys.replace("not flood precision, recall, f1, iou or accuracy", "")


def test_freeze_records_hashes_and_any_later_edit_is_refused(tmp_path):
    root = protocol_root(tmp_path)
    p = load_protocol(root)
    assert p.event_category("Landslide") == "reference_mass_movement"
    with pytest.raises(ProtocolError):
        p.event_category("Mudflow")                                          # unmapped -> fail closed
    with pytest.raises(ProtocolError):
        p.layer_role("hydrographyA")
    rec = freeze(p, tmp_path / "eval")
    frozen = load_protocol(root)
    assert frozen.status == "frozen" and check_frozen(frozen, tmp_path / "eval")["frozen_at"] == rec["frozen_at"]
    doc = root / "docs" / "evaluation" / "emsr927_protocol.md"
    doc.write_text(doc.read_text(encoding="utf-8") + "\nedited after freeze\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="changed after the freeze"):
        check_frozen(load_protocol(root), tmp_path / "eval")
    assert json.loads((tmp_path / "eval" / "protocol_freeze.json").read_text())["yaml"]["sha256"] == frozen.yaml_sha256


def test_protocol_rejects_unknown_category(tmp_path):
    root = protocol_root(tmp_path)
    y = root / "config" / "evaluation" / "emsr927_protocol.yaml"
    y.write_text(y.read_text(encoding="utf-8").replace("    Landslide: reference_mass_movement",
                                                       "    Landslide: flood_water_or_mud", 1), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(root)


def test_box_fixture_sanity():
    assert EMS_AOI_BOX[0] < EMS_AOI_BOX[2]


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_dense_vertices_do_not_inflate_matched_road_length():
    """100 m segment, a vertex every ~1 m; only its first ~9 m is within 20 m of the EMS line -> unmatched (< 25 m)."""
    g = small_grid(85.0, 28.05, 300)
    lat = 28.045
    dlon = 1.0 / (111320 * np.cos(np.radians(lat)))                     # ~1 m of longitude
    coords = [[85.002 + i * dlon, lat] for i in range(101)]
    seg = {"features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"segment_id": "s", "state": "blocked", "length_m": 100.0}}]}
    ems = [feature({"type": "LineString", "coordinates": [[85.002 - 30 * dlon, lat + 0.0001],
                                                          [85.002 - 8 * dlon, lat + 0.0001]]})]
    r = road_comparison(seg, ems, ["destroyed"], np.ones(g.shape, bool), g, ROAD_PARAMS, {"p": PRIMARY})
    row = r["comparisons"]["p"]["segments"][0]
    assert row["ground_length_m"] == pytest.approx(100, abs=1.5)
    assert 5 <= row["matched_m"]["positive"] <= 15 and row["reference"] == "unmatched"


def test_leakage_scan_path_check_has_no_key_exemption(tmp_path):
    import shutil

    from bato.config import load_config
    from bato.evaluation.leakage import scan
    shutil.copytree(REPO_CONFIG_DIR, tmp_path / "config")
    cfg = load_config(tmp_path / "config")
    d = cfg.settings.paths.processed / "x"
    d.mkdir(parents=True)
    (d / "a_provenance.json").write_text(json.dumps({
        "inputs_not_used": ["EMSR / UNOSAT / any damage map"],                           # statement: exempt
        "inputs": {"description_src": "data/reference/EMSR927/products/x.zip"},       # path: hit despite key
        "notes": "compared with Copernicus EMS"}))                                     # name: hit
    r = scan(cfg.settings.paths, tmp_path)
    assert r["json_files_scanned"] == 1
    assert sorted(h["json_path"] for h in r["hits"]) == ["inputs.description_src", "notes"]


def test_shapefile_record_with_two_outer_rings_is_a_multipolygon():
    from bato.evaluation.ems_product import read_shp
    a = [(85.0, 28.0), (85.0, 28.1), (85.1, 28.1), (85.1, 28.0), (85.0, 28.0)]          # clockwise = outer
    b = [(85.2, 28.0), (85.2, 28.1), (85.3, 28.1), (85.3, 28.0), (85.2, 28.0)]
    raw = shapefile_bytes([a], [{"k": "v"}])["shp"]
    import struct
    pts = b"".join(struct.pack("<2d", x, y) for x, y in a + b)
    content = struct.pack("<i4d2i", 5, 85.0, 28.0, 85.3, 28.1, 2, 10) + struct.pack("<2i", 0, 5) + pts
    rec = struct.pack(">2i", 1, len(content) // 2) + content
    (g,) = read_shp(raw[:100] + rec)
    assert g is not None and g["type"] == "MultiPolygon" and len(g["coordinates"]) == 2
    from bato.evaluation.geometry import geojson_polygons
    total = sum(polygon_area_km2(p) for p in geojson_polygons(g))
    assert total == pytest.approx(2 * polygon_area_km2([[tuple(p) for p in a]]), rel=1e-6)


def test_product_with_an_undated_image_cannot_be_selected():
    dated = prod(1, 0, 1, ["2026-08-27T05:05:00"])
    undated = prod(1, 1, 1, ["2026-08-28T12:00:00", None])  # type: ignore[list-item]
    picked, _ = select_product([dated, undated], POST, ZONES)
    assert picked is dated


def test_downloads_refuse_other_hosts():
    from bato.evaluation.acquisition import AcquisitionError, http_open
    with pytest.raises(AcquisitionError):
        http_open("https://example.org/EMSR927_x.zip")
    with pytest.raises(AcquisitionError):
        http_open("http://rapidmapping.emergency.copernicus.eu/backend/EMSR927_x.zip")


# ---------------------------------------------------------------------------
# Freeze/hash lifecycle on CRLF files and Windows text-mode writes (regression)
# ---------------------------------------------------------------------------


def _to_crlf(path):
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def test_freeze_cycle_is_stable_on_crlf_protocol_files(tmp_path):
    """A Windows checkout (git autocrlf) has CRLF files: freeze must keep the bytes' line endings and hash exactly
    what is on disk, so freeze -> reload -> check_frozen passes and a later edit still fails."""
    root = protocol_root(tmp_path)
    y = root / "config" / "evaluation" / "emsr927_protocol.yaml"
    d = root / "docs" / "evaluation" / "emsr927_protocol.md"
    _to_crlf(y)
    _to_crlf(d)
    rec = freeze(load_protocol(root), tmp_path / "eval")
    data = y.read_bytes()
    assert b"\r\nstatus: frozen\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")      # CRLF preserved
    p = load_protocol(root)
    assert rec["yaml"]["sha256"] == p.yaml_sha256 and rec["doc"]["sha256"] == p.doc_sha256
    assert check_frozen(p, tmp_path / "eval")["frozen_at"] == rec["frozen_at"]
    y.write_bytes(data.replace(b"tolerance_px: 1", b"tolerance_px: 2"))
    with pytest.raises(ProtocolError, match="changed after the freeze"):
        check_frozen(load_protocol(root), tmp_path / "eval")


def test_freeze_cycle_is_stable_under_windows_text_mode_writes(tmp_path, monkeypatch):
    """Reproduces the reported Windows failure: text-mode writes turn '\\n' into '\\r\\n'. The recorded hashes
    must still be the hashes of the files on disk."""
    import pathlib
    real = pathlib.Path.write_text

    def windows_write_text(self, data, encoding=None, errors=None, newline=None):
        return real(self, data, encoding=encoding, errors=errors, newline="\r\n")

    monkeypatch.setattr(pathlib.Path, "write_text", windows_write_text)
    root = protocol_root(tmp_path)
    for f in (root / "config" / "evaluation" / "emsr927_protocol.yaml",
              root / "docs" / "evaluation" / "emsr927_protocol.md"):
        f.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")   # text-mode rewrite: CRLF, as on Windows
    freeze(load_protocol(root), tmp_path / "eval")
    assert check_frozen(load_protocol(root), tmp_path / "eval")["protocol_id"] == "bato-emsr927-v1"


def test_freeze_refuses_files_edited_after_they_were_loaded(tmp_path):
    root = protocol_root(tmp_path)
    p = load_protocol(root)
    d = root / "docs" / "evaluation" / "emsr927_protocol.md"
    d.write_bytes(d.read_bytes() + b"\nlate edit\n")
    with pytest.raises(ProtocolError, match="changed since they were loaded"):
        freeze(p, tmp_path / "eval")
    assert load_protocol(root).status == "draft" and not (tmp_path / "eval" / "protocol_freeze.json").exists()


# ---------------------------------------------------------------------------
# Download redirect trust rule (fake HTTPS transport; no network)
# ---------------------------------------------------------------------------

OFFICIAL = "https://rapidmapping.emergency.copernicus.eu/backend/EMSR927/AOI01/GRA_PRODUCT/EMSR927_AOI01_GRA_PRODUCT_v1.zip"
S3 = "https://rapidmapping.s3.amazonaws.com/EMSR927/AOI01/GRA_PRODUCT/EMSR927_AOI01_GRA_PRODUCT_v1.zip?X-Amz-Sig=abc"


def test_official_product_url_redirect_to_trusted_s3_is_allowed_and_recorded(tmp_path):
    from functools import partial

    from bato.evaluation.acquisition import _stream_to, http_open

    from ems_fakes import FakeHTTPS
    fake = FakeHTTPS({OFFICIAL: (302, S3), S3: (200, b"PK zip bytes")})
    n, digest, transfer = _stream_to(partial(http_open, extra_handlers=(fake,)), OFFICIAL, tmp_path / "p.zip")
    assert fake.seen == [OFFICIAL, S3]
    assert (tmp_path / "p.zip").read_bytes() == b"PK zip bytes" and n == 12
    assert digest == __import__("hashlib").sha256(b"PK zip bytes").hexdigest()
    assert transfer == {"original_url": OFFICIAL, "final_url": S3,
                        "redirect_chain": [{"status": 302, "from": OFFICIAL, "to": S3}]}


@pytest.mark.parametrize("target", [
    "https://evil.example.com/EMSR927_AOI01_GRA_PRODUCT_v1.zip",                     # unrelated host
    "https://rapidmapping.s3.amazonaws.com.evil.net/EMSR927_AOI01_GRA_PRODUCT_v1.zip",  # look-alike suffix
    "https://other-bucket.s3.amazonaws.com/EMSR927_AOI01_GRA_PRODUCT_v1.zip",          # other S3 bucket
    "http://rapidmapping.s3.amazonaws.com/EMSR927_AOI01_GRA_PRODUCT_v1.zip",            # not HTTPS
    "https://rapidmapping.s3.amazonaws.com:8443/EMSR927_AOI01_GRA_PRODUCT_v1.zip",      # non-default port
    "https://user@rapidmapping.s3.amazonaws.com/EMSR927_AOI01_GRA_PRODUCT_v1.zip",      # user-info
])
def test_official_product_url_redirect_to_untrusted_target_is_rejected(target):
    from bato.evaluation.acquisition import AcquisitionError, http_open

    from ems_fakes import FakeHTTPS
    fake = FakeHTTPS({OFFICIAL: (302, target), target: (200, b"bad")})
    with pytest.raises(AcquisitionError, match="refusing redirect"):
        http_open(OFFICIAL, extra_handlers=(fake,))
    assert fake.seen == [OFFICIAL]                                   # the untrusted target is never requested


def test_direct_s3_url_not_reached_from_the_official_url_is_rejected():
    from bato.evaluation.acquisition import AcquisitionError, http_open

    from ems_fakes import FakeHTTPS
    fake = FakeHTTPS({S3: (200, b"PK")})
    with pytest.raises(AcquisitionError, match="must start at https://rapidmapping.emergency.copernicus.eu/"):
        http_open(S3, extra_handlers=(fake,))
    assert fake.seen == []                                           # refused before any request


def test_s3_redirect_is_allowed_only_from_a_product_zip_url():
    """The activation API (not a product downloadPath) may redirect only within the official host."""
    from bato.evaluation.acquisition import AcquisitionError, http_open

    from ems_fakes import API, FakeHTTPS
    fake = FakeHTTPS({API: (302, S3), S3: (200, b"{}")})
    with pytest.raises(AcquisitionError, match="refusing redirect"):
        http_open(API, extra_handlers=(fake,))
    moved = API.replace("public-activations/", "public-activations/v2/")
    ok = FakeHTTPS({API: (301, moved), moved: (200, b"{}")})
    resp = http_open(API, extra_handlers=(ok,))
    assert resp.read() == b"{}" and resp.bato_final_url == moved


def test_redirect_hops_are_capped():
    import urllib.error

    from bato.evaluation.acquisition import MAX_REDIRECTS, http_open

    from ems_fakes import FakeHTTPS
    hops = [OFFICIAL] + [f"{OFFICIAL}?hop={i}" for i in range(1, MAX_REDIRECTS + 2)]
    fake = FakeHTTPS({u: (302, hops[i + 1]) for i, u in enumerate(hops[:-1])} | {hops[-1]: (200, b"x")})
    with pytest.raises(urllib.error.HTTPError):
        http_open(OFFICIAL, extra_handlers=(fake,))


# ---------------------------------------------------------------------------
# Reference coverage: EMS positive roads that match no eligible BATO segment (regression for the FN-accounting gap)
# ---------------------------------------------------------------------------

def _seg(sid: str, st: str, lat: float, lon0: float = 85.002, lon1: float = 85.008) -> dict:
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[lon0, lat], [lon1, lat]]},
            "properties": {"segment_id": sid, "state": st}}


def _line(lat: float, lon0: float = 85.002, lon1: float = 85.008) -> dict:
    return feature({"type": "LineString", "coordinates": [[lon0, lat], [lon1, lat]]})


def _run(segs, ems, grades, n=400, comps=None):
    g = small_grid(85.0, 28.05, n)
    return road_comparison({"features": segs}, ems, grades, np.ones(g.shape, bool), g, ROAD_PARAMS,
                           comps or {"primary_destroyed": PRIMARY, "supplementary_any_damage": SUPPLEMENTARY})


def test_unmatched_destroyed_road_is_reported_not_dropped_or_counted_negative():
    """A Destroyed EMS road > 20 m from every eligible BATO segment must not vanish: the segment-level matrix is
    unchanged (it is not turned into a TN or anything else), and reference_coverage reports it as unmatched."""
    segs = [_seg("a", "blocked", 28.045), _seg("b", "open", 28.040)]
    base_ems, base_grades = [_line(28.04503), _line(28.04003)], ["destroyed", "no_visible_damage"]
    far = _line(28.030, 85.010, 85.016)                      # ~1 km south of every BATO segment, ~590 m long
    without = _run(segs, base_ems, base_grades)["comparisons"]["primary_destroyed"]
    with_far = _run(segs, base_ems + [far], base_grades + ["destroyed"])["comparisons"]["primary_destroyed"]

    # not silently counted as an ordinary negative (or anything else) in the segment-level matrix
    for mode in ("observed_only", "all_segments"):
        assert with_far["modes"][mode]["counts"] == without["modes"][mode]["counts"] == \
            {"TP": 1, "FP": 0, "FN": 0, "TN": 1, "N": 2}
    # ...but reported explicitly as unmatched reference coverage
    cov, cov0 = with_far["reference_coverage"], without["reference_coverage"]
    assert cov0["matched_positive_length_fraction"] == 1.0 and cov0["ems_positive_features_unmatched"] == 0
    assert cov["ems_positive_features_in_area"] == 2 and cov["ems_positive_features_unmatched"] == 1
    assert cov["ems_positive_length_unmatched_m"] == pytest.approx(588, abs=8)
    assert cov["unmatched_no_eligible_bato_segment_within_distance_m"] == pytest.approx(588, abs=8)
    assert cov["unmatched_near_bato_segment_without_positive_label_m"] == 0
    assert cov["matched_positive_length_fraction"] == pytest.approx(0.5, abs=0.01)
    assert cov["ems_positive_length_in_area_m"] == pytest.approx(
        cov["ems_positive_length_matched_m"] + cov["ems_positive_length_unmatched_m"], abs=0.2)
    # metrics are explicitly conditional on the matched domain, never full-network recall
    assert "NOT full-network recall" in with_far["metric_domain"]
    assert "unobservable" in with_far["modes"]["observed_only"]["interpretation"]
    # no false improvement: the matched-domain recall cannot rise by dropping the reference, and the drop is visible
    r_with = with_far["modes"]["all_segments"]["strict"]["recall"]["value"]
    r_without = without["modes"]["all_segments"]["strict"]["recall"]["value"]
    assert r_with == r_without == 1.0
    assert cov["matched_positive_length_fraction"] < cov0["matched_positive_length_fraction"]


def test_partially_matched_destroyed_geometry():
    """Half of a Destroyed line runs along an eligible blocked segment, half continues > 20 m beyond any segment."""
    segs = [_seg("a", "blocked", 28.045)]
    ems = [_line(28.04503, 85.002, 85.014)]                    # 85.002-85.008 along "a", 85.008-85.014 beyond it
    cov = _run(segs, ems, ["destroyed"])["comparisons"]["primary_destroyed"]["reference_coverage"]
    assert cov["ems_positive_features_partially_matched"] == 1 and cov["ems_positive_features_unmatched"] == 0
    assert cov["matched_positive_length_fraction"] == pytest.approx(0.5 + 20 / 1176, abs=0.02)   # + end tolerance


def test_several_grades_near_one_segment_and_coverage_of_a_short_destroyed_stretch():
    """One BATO segment with a full possibly-damaged line, a full no-visible-damage line and a 5 m destroyed stub
    19 m to the side (it reaches only ~5 + 2*sqrt(20^2 - 19^2) = ~17.5 m of the segment, < 25 m):
    primary -> excluded (excluded beats negative); supplementary -> positive. The destroyed stub lies near a segment
    WITHOUT the positive label in the primary, so it is reported as such, not dropped."""
    dlon = 1.0 / (111320 * np.cos(np.radians(28.045)))
    segs = [_seg("a", "open", 28.045)]
    ems = [_line(28.04503), _line(28.04497), _line(28.045 + 19 / 111320, 85.005, 85.005 + 5 * dlon)]
    r = _run(segs, ems, ["possibly_damaged", "no_visible_damage", "destroyed"])["comparisons"]
    assert r["primary_destroyed"]["segments"][0]["reference"] == "excluded"
    assert r["supplementary_any_damage"]["segments"][0]["reference"] == "positive"
    cov = r["primary_destroyed"]["reference_coverage"]
    assert cov["ems_positive_length_matched_m"] == 0
    assert cov["unmatched_near_bato_segment_without_positive_label_m"] == pytest.approx(5, abs=1)
    assert r["primary_destroyed"]["modes"] is None and r["primary_destroyed"]["value"] == "NOT CALCULABLE"
    sup = r["supplementary_any_damage"]
    assert sup["reference_coverage"]["matched_positive_length_fraction"] == 1.0
    assert sup["modes"]["all_segments"]["counts"] == {"TP": 0, "FP": 0, "FN": 1, "TN": 0, "N": 1}


def test_unobservable_positive_segment_counts_only_in_all_segments_coverage():
    segs = [_seg("a", "unobservable", 28.045)]
    r = _run(segs, [_line(28.04503)], ["destroyed"])["comparisons"]["primary_destroyed"]
    cov = r["reference_coverage"]
    assert cov["matched_positive_length_fraction"] == 1.0
    assert cov["matched_positive_length_fraction_observed_only"] == 0
    assert r["modes"]["observed_only"]["counts"]["N"] == 0 and r["modes"]["all_segments"]["counts"]["FN"] == 1


def test_no_overlap_between_ems_roads_and_bato_segments():
    segs = [_seg("a", "blocked", 28.045)]
    r = _run(segs, [_line(28.030), _line(28.031)], ["destroyed", "no_visible_damage"])["comparisons"]
    pri = r["primary_destroyed"]
    assert pri["modes"] is None and pri["value"] == "NOT CALCULABLE"
    assert pri["segments_by_reference_label"] == {"positive": 0, "negative": 0, "excluded": 0, "unmatched": 1}
    cov = pri["reference_coverage"]
    assert cov["ems_positive_length_matched_m"] == 0 and cov["matched_positive_length_fraction"] == 0
    assert cov["ems_positive_features_unmatched"] == 1


def test_coverage_is_not_calculable_without_positive_reference():
    r = _run([_seg("a", "open", 28.045)], [_line(28.04503)], ["no_visible_damage"])["comparisons"]
    assert r["primary_destroyed"]["reference_coverage"]["matched_positive_length_fraction"] == "NOT CALCULABLE"
    assert r["primary_destroyed"]["modes"]["all_segments"]["strict"]["recall"]["value"] == "NOT CALCULABLE"


def test_matching_rule_corridor_property_is_as_documented():
    """Documented property of the pre-registered rule (protocol G): a positive EMS stretch of length L parallel to a
    segment at small offset matches up to about L + 2 * 20 m of it, so even a ~1 m stretch close to the road labels a
    segment positive. Pinned here so any change to the rule is deliberate and visible."""
    dlon = 1.0 / (111320 * np.cos(np.radians(28.045)))
    segs = [_seg("a", "blocked", 28.045)]
    stub = _line(28.04503, 85.005, 85.005 + 1 * dlon)                         # 1 m, ~3 m off the road
    row = _run(segs, [stub], ["destroyed"])["comparisons"]["primary_destroyed"]["segments"][0]
    assert row["reference"] == "positive" and 35 <= row["matched_m"]["positive"] <= 45


# ---------------------------------------------------------------------------
# Test isolation from the live protocol state (the repository protocol may be legitimately frozen)
# ---------------------------------------------------------------------------

def _sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"])
def test_draft_fixture_resets_only_the_lifecycle_lines_of_a_frozen_protocol(eol):
    from ems_fakes import PROTOCOL_YAML_REL, draft_protocol_bytes
    live = (REPO_CONFIG_DIR.parent / PROTOCOL_YAML_REL).read_bytes().replace(b"\r\n", b"\n")
    frozen = b"\n".join(b"status: frozen" if ln.startswith(b"status:") else
                        b'frozen_at: "2026-10-05T08:41:38+00:00"' if ln.startswith(b"frozen_at:") else ln
                        for ln in live.split(b"\n")).replace(b"\n", eol)
    draft = draft_protocol_bytes(frozen)
    a, b = frozen.split(eol), draft.split(eol)
    changed = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
    assert changed == [(b"status: frozen", b"status: draft"),
                       (b'frozen_at: "2026-10-05T08:41:38+00:00"', b"frozen_at: null")]
    assert draft.count(eol) == frozen.count(eol)                              # line endings preserved


def test_draft_fixture_keeps_the_live_protocol_semantics_and_never_writes_to_the_repository(tmp_path):
    from ems_fakes import PROTOCOL_DOC_REL, PROTOCOL_YAML_REL, install_draft_protocol
    repo = REPO_CONFIG_DIR.parent
    live_y, live_d = repo / PROTOCOL_YAML_REL, repo / PROTOCOL_DOC_REL
    before = (_sha(live_y), _sha(live_d))
    root = protocol_root(tmp_path)
    p, live = load_protocol(root), load_protocol(repo)
    assert p.status == "draft" and p.raw["frozen_at"] is None
    strip = lambda r: {k: v for k, v in r.items() if k not in ("status", "frozen_at")}  # noqa: E731
    assert strip(p.raw) == strip(live.raw)                                    # same mappings, rules, thresholds
    assert p.doc_sha256 == live.doc_sha256
    assert not (tmp_path / "eval" / "protocol_freeze.json").exists()
    freeze(p, tmp_path / "eval")                                              # the real freeze, on the fixture
    assert load_protocol(root).status == "frozen"
    assert (_sha(live_y), _sha(live_d)) == before                             # live files untouched
    assert load_protocol(repo).status == live.status
    with pytest.raises(AssertionError, match="inside the repository"):
        install_draft_protocol(repo)

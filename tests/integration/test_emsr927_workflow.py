"""EMSR927 workflow end to end on synthetic data: acquire -> inspect -> (refuse unmapped) -> map -> freeze ->
evaluate. Checks the leakage and read-only guarantees. Uses the published synthetic v1 run of dashboard_fakes and
invented EMS-like products (tests/ems_fakes.py)."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from bato.data_rules import EVALUATION_PACKAGE, assert_pipeline_input_allowed
from bato.evaluation.acquisition import acquire, load_manifest
from bato.evaluation.emsr927 import EvaluationError, evaluate, freeze_protocol, inspect
from bato.evaluation.geometry import Grid
from bato.evaluation.metrics import FLOOD, confusion
from bato.evaluation.official import official_aoi
from bato.evaluation.protocol import ProtocolError, load_protocol
from bato.exceptions import ProhibitedInputError

from connectivity_fakes import AOI
from dashboard_fakes import published_world
from ems_fakes import (
    API,
    EMS_AOI_BOX,
    FLOOD_BOXES,
    LANDSLIDE_BOX,
    NOTAN_BOX,
    TRACE_BOX,
    FakeOpener,
    activation,
    add_flood_mapping,
    box_mask,
    install_draft_protocol,
    standard_zips,
)

T0 = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def snapshot(*dirs: Path) -> dict[str, str]:
    out = {}
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def world(tmp_path):
    w = published_world(tmp_path)
    install_draft_protocol(tmp_path)
    w.reader = lambda _p: (w.state, w.transform, "EPSG:3857")
    w.opener = FakeOpener(activation({}), standard_zips())
    return w


def acquire_all(w):
    paths = w.cfg.settings.paths
    return acquire(paths, official_aoi(w.cfg, "aoi-test", "ts-test"), load_protocol(w.root).raw["reference"],
                   opener=w.opener, now=lambda: T0)


def test_full_workflow_matches_brute_force_counts_and_leaves_bato_untouched(world):
    w, paths = world, world.cfg.settings.paths
    pipeline_before = snapshot(paths.raw, paths.interim, paths.processed)

    # 1. acquisition: official record saved verbatim, only overlapping final products downloaded
    m = acquire_all(w)
    assert w.opener.requests[0] == API
    assert sorted(d["label"] for d in m["downloads"]) == ["AOI01 GRA MONIT01 v1", "AOI01 GRA PRODUCT v1"]
    assert not any("AOI02" in r for r in w.opener.requests[1:])                 # no overlap -> never requested
    skipped = {s["label"]: s["reason"] for s in m["not_downloaded"]}
    assert "does not overlap" in skipped["AOI02 GRA PRODUCT v1"] and "not final" in skipped["AOI03 GRA PRODUCT v1"]
    for d in m["downloads"]:
        f = paths.reference / "EMSR927" / d["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == d["sha256"] and d["bytes"] == f.stat().st_size
        assert d["url"].startswith("https://rapidmapping.emergency.copernicus.eu/") and d["url_source"]
        with pytest.raises(ProhibitedInputError):
            assert_pipeline_input_allowed(f, paths)                            # pipeline can never read it
    act = paths.reference / "EMSR927" / "activation" / m["activation"]["file"]
    assert json.loads(act.read_text())["results"][0]["code"] == "EMSR927"
    assert m["record"]["time_zone_marker_in_record"].startswith("none")

    # 2. metrics refused while the protocol is a draft
    with pytest.raises(ProtocolError, match="draft"):
        evaluate(w.cfg, "aoi-test", "ts-test", read_state=w.reader)

    # 3. inspection: the repository mapping (completed from the official products) covers every layer and value
    rep = inspect(w.cfg, load_protocol(w.root))
    assert rep["metrics_computed"] is False
    assert rep["product_selection"]["compared"] == ["AOI01 GRA PRODUCT v1"]
    assert rep["mapping_problems_for_freeze"] == []
    geo = rep["road_geometry_check"]
    assert geo["reads_bato_road_state"] is False
    chk = geo["products"]["AOI01 GRA PRODUCT v1"]
    assert chk["share_of_ems_road_length_within_distance_of_bato_roads"]["10m"] == 1.0     # EMS lines ~3 m off
    assert chk["bato_segments_shorter_than_min_matched_length"] == 0

    # 4. freeze
    rec = freeze_protocol(w.cfg, load_protocol(w.root))
    assert load_protocol(w.root).status == "frozen" and rec["yaml"]["sha256"]

    # 5. evaluation
    out, res = evaluate(w.cfg, "aoi-test", "ts-test", read_state=w.reader)
    assert out.parent == paths.eval / "EMSR927" and (out / "metrics.json").is_file() and (out / "metrics.md").is_file()
    assert res["leakage_scan"]["hits"] == 0 and res["bato"]["files_unchanged_after_evaluation"]
    assert res["bato"]["road_rule"] == "v1" and res["primary_evaluation"]["primary_comparison"] == "primary_destroyed"
    (p,) = res["per_product"]
    assert p["product"] == "AOI01 GRA PRODUCT v1"

    # 5a. PRIMARY: roads. Expected labels follow from the fixture geometry (tests/ems_fakes.py ROADS) and the
    #     synthetic BATO states (seg-203-0-0-9..11 and 20..23 blocked): see the per-segment table below.
    roads = p["roads"]
    pri, sup = roads["comparisons"]["primary_destroyed"], roads["comparisons"]["supplementary_any_damage"]
    lab = lambda c: {r["segment_id"].rsplit("-", 1)[1]: r["reference"] for r in c["segments"]  # noqa: E731
                     if r["reference"] != "unmatched"}
    assert lab(pri) == {**{k: "positive" for k in ("0", "1", "2", "9", "10", "11")},
                        # 24 sits between a possibly-damaged and a not-analysed line: both excluded grades here,
                        # union ~39 m >= 25 m -> excluded (precedence positive > excluded > negative)
                        **{k: "excluded" for k in ("4", "5", "20", "21", "22", "23", "24", "25", "26")},
                        **{k: "negative" for k in ("13", "14", "15", "16", "17")}}
    assert lab(sup) == {**{k: "positive" for k in ("0", "1", "2", "3", "4", "5", "9", "10", "11", "20", "21", "22",
                                                   "23")},
                        **{k: "excluded" for k in ("25", "26")}, **{k: "negative" for k in ("13", "14", "15", "16",
                                                                                            "17")}}
    assert pri["modes"]["observed_only"]["counts"] == {"TP": 3, "FP": 0, "FN": 3, "TN": 5, "N": 11}
    assert pri["modes"]["observed_only"]["strict"]["precision"]["value"] == 1.0
    assert pri["modes"]["observed_only"]["strict"]["recall"]["value"] == 0.5
    assert sup["modes"]["observed_only"]["counts"] == {"TP": 7, "FP": 0, "FN": 6, "TN": 5, "N": 18}
    assert "not accuracy" in sup["label"] and sup["role"] == "supplementary"
    assert pri["descriptive_lengths_m"]["ems_positive_features_in_area"] == 2
    assert "NOT full-network recall" in pri["metric_domain"]
    for c in (pri, sup):                         # every EMS positive line here runs along a positive-labelled segment
        assert c["reference_coverage"]["matched_positive_length_fraction"] == 1.0
        assert c["reference_coverage"]["ems_positive_features_unmatched"] == 0
    assert res["pooled"]["roads"]["primary_destroyed"]["modes"]["observed_only"]["counts"] == \
        pri["modes"]["observed_only"]["counts"]

    # 5b. no flood class -> flood comparison NOT CALCULABLE; mass movement = descriptive overlap only
    assert p["comparisons"]["flood_water"]["value"] == "NOT CALCULABLE"
    assert "flood_precision_recall_f1_iou" in res["not_calculable"]
    t, h = w.transform, w.state.shape
    grid = Grid(t, h[1], h[0])
    area = box_mask(grid, EMS_AOI_BOX) & box_mask(grid, (AOI.min_lon, AOI.min_lat, AOI.max_lon, AOI.max_lat)) & \
        ~box_mask(grid, NOTAN_BOX)
    footprint = box_mask(grid, LANDSLIDE_BOX) & area
    mm = p["comparisons"]["reference_mass_movement"]
    assert mm["class_match"] is False and "CLASS-MISMATCHED FOOTPRINT OVERLAP" in mm["label"]
    assert mm["comparison_area_pixels"] == int(area.sum()) and mm["footprint_pixels"] == int(footprint.sum())
    assert mm["overlap_pixels"] == int((footprint & (w.state == FLOOD)).sum())
    assert not any(k in mm for k in ("modes", "precision", "recall", "f1", "iou"))
    assert p["pixels"]["not_analysed_removed"] == int((box_mask(grid, NOTAN_BOX) & box_mask(grid, EMS_AOI_BOX)).sum())

    # 5c. bridges: counts only
    assert p["bridges"]["features_in_area_by_grade"] == {"destroyed": 1, "possibly_damaged": 1}
    assert "does not currently generate a bridge-damage prediction layer" in p["bridges"]["note"]
    assert "bridge_precision_recall" in res["not_calculable"] and "building_precision_recall" in res["not_calculable"]
    gaps = res["time"]["gaps_ems_image_minus_bato_post_hours"][0]["images"][0]["gap_hours"]
    assert set(gaps) == {"UTC", "Asia/Kathmandu"} and gaps["UTC"] < 0

    # 6. nothing under raw / interim / processed changed; one run logged
    assert snapshot(paths.raw, paths.interim, paths.processed) == pipeline_before
    runs = (paths.eval / "EMSR927" / "runs.jsonl").read_text().splitlines()
    assert len(runs) == 1 and json.loads(runs[0])["protocol_id"] == "bato-emsr927-v1"

    # 7. a second run under the same frozen protocol reproduces the first and does not overwrite it
    first = (out / "metrics.json").read_bytes()
    evaluate(w.cfg, "aoi-test", "ts-test", read_state=w.reader)
    runs = [json.loads(x) for x in (paths.eval / "EMSR927" / "runs.jsonl").read_text().splitlines()]
    assert runs[-1]["reproduced_earlier_result"] is True and (out / "metrics.json").read_bytes() == first


def frozen_world(w):
    acquire_all(w)
    inspect(w.cfg, load_protocol(w.root))
    freeze_protocol(w.cfg, load_protocol(w.root))


def test_unexpected_class_or_damage_value_blocks_the_freeze(world):
    world.opener = FakeOpener(activation({}), standard_zips(extra_class="Mudflow", extra_road_value="Severely damaged"))
    acquire_all(world)
    rep = inspect(world.cfg, load_protocol(world.root))
    probs = rep["mapping_problems_for_freeze"]
    assert any("'Mudflow' is not mapped" in x for x in probs)
    assert any("'Severely damaged' is not mapped" in x for x in probs)
    with pytest.raises(ProtocolError, match="unmapped"):
        freeze_protocol(world.cfg, load_protocol(world.root))
    assert load_protocol(world.root).status == "draft"


def test_synthetic_flood_class_machinery_matches_brute_force_counts(world):
    """MACHINERY ONLY, on invented 'Flooded area' polygons (the real EMSR927 products have no flood class): the
    class-matched pixel comparison still reproduces brute-force counts in both modes."""
    world.opener = FakeOpener(activation({}), standard_zips(flood=True))
    add_flood_mapping(world.root)
    frozen_world(world)
    _, res = evaluate(world.cfg, "aoi-test", "ts-test", read_state=world.reader)
    (p,) = res["per_product"]
    t, h = world.transform, world.state.shape
    grid = Grid(t, h[1], h[0])
    bato_aoi = box_mask(grid, (AOI.min_lon, AOI.min_lat, AOI.max_lon, AOI.max_lat))
    flood_ref = box_mask(grid, FLOOD_BOXES[0]) | box_mask(grid, FLOOD_BOXES[1])
    area = box_mask(grid, EMS_AOI_BOX) & bato_aoi & ~box_mask(grid, NOTAN_BOX) & ~box_mask(grid, TRACE_BOX) & \
        ~box_mask(grid, LANDSLIDE_BOX)
    assessed = np.isin(world.state, (0, 1, 2))
    fl = p["comparisons"]["flood_water"]
    assert fl["class_match"] is True and fl["comparison_area_pixels"] == int(area.sum())
    assert fl["modes"]["assessed_only"]["counts"] == confusion(world.state == FLOOD, flood_ref, area & assessed)
    assert fl["modes"]["all_pixels"]["counts"] == confusion(world.state == FLOOD, flood_ref, area)
    exp_na = int((flood_ref & area & ~assessed).sum())
    assert exp_na > 0 and fl["reference_pixels_not_assessed_by_reason"]["NOT_ASSESSED_STEEP"] == exp_na


def test_leakage_hit_in_pipeline_provenance_blocks_the_evaluation(world):
    frozen_world(world)
    paths = world.cfg.settings.paths
    bad = paths.processed / "somewhere" / "leaky_provenance.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"inputs": {"labels": "data/reference/EMSR927/products/EMSR927_AOI01.zip"}}))
    with pytest.raises(EvaluationError, match="leakage"):
        evaluate(world.cfg, "aoi-test", "ts-test", read_state=world.reader)
    assert not (paths.eval / "EMSR927" / "runs.jsonl").exists()


def test_tampered_reference_zip_and_refreeze_are_refused(world):
    frozen_world(world)
    paths = world.cfg.settings.paths
    evaluate(world.cfg, "aoi-test", "ts-test", read_state=world.reader)
    m = load_manifest(paths)
    assert m is not None
    z = paths.reference / "EMSR927" / m["downloads"][0]["file"]
    original = z.read_bytes()
    z.write_bytes(original + b"x")
    with pytest.raises(Exception, match="SHA-256"):
        evaluate(world.cfg, "aoi-test", "ts-test", read_state=world.reader)
    z.write_bytes(original)
    y = world.root / "config" / "evaluation" / "emsr927_protocol.yaml"
    y.write_text(y.read_text().replace("status: frozen", "status: draft").replace(
        y.read_text().split("frozen_at: ")[1].split("\n")[0], "null"))
    with pytest.raises(ProtocolError, match="already computed"):
        freeze_protocol(world.cfg, load_protocol(world.root))


def test_pipeline_scripts_never_import_the_evaluation_package():
    """Only the three EMSR927 scripts may import bato.evaluation; every pipeline script must not."""
    allowed = {"fetch_emsr927.py", "inspect_emsr927.py", "evaluate_emsr927.py"}
    offenders = []
    for path in SCRIPTS.glob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            offenders += [f"{path.name}: {m}" for m in mods if m.startswith(EVALUATION_PACKAGE)]
    assert not offenders, offenders


def test_acquisition_through_the_official_s3_redirect_records_the_chain(world):
    """Product downloads follow the official Rapid Mapping -> rapidmapping.s3.amazonaws.com redirect (fake HTTPS
    transport, no network); the manifest keeps original URL, final URL, chain, time, bytes and SHA-256."""
    from functools import partial

    from bato.evaluation.acquisition import http_open

    from ems_fakes import FakeHTTPS

    zips = standard_zips()
    routes: dict[str, tuple[int, object]] = {API: (200, json.dumps(activation({})).encode())}
    for name, data in zips.items():
        aoi = name.split("_")[1]
        kind = "_".join(name.split("_")[2:4])
        official = f"https://rapidmapping.emergency.copernicus.eu/backend/EMSR927/{aoi}/{kind}/{name}"
        s3 = f"https://rapidmapping.s3.amazonaws.com/EMSR927/{aoi}/{kind}/{name}?X-Amz-Signature=t"
        routes[official] = (302, s3)
        routes[s3] = (200, data)
    fake = FakeHTTPS(routes)
    m = acquire(world.cfg.settings.paths, official_aoi(world.cfg, "aoi-test", "ts-test"),
                load_protocol(world.root).raw["reference"], opener=partial(http_open, extra_handlers=(fake,)),
                now=lambda: T0)
    assert m["activation"]["final_url"] == API and m["activation"]["redirect_chain"] == []
    assert len(m["downloads"]) == 2
    for d in m["downloads"]:
        assert d["url"].startswith("https://rapidmapping.emergency.copernicus.eu/backend/EMSR927/")
        assert d["final_url"].startswith("https://rapidmapping.s3.amazonaws.com/EMSR927/")
        assert d["redirect_chain"] == [{"status": 302, "from": d["url"], "to": d["final_url"]}]
        f = world.cfg.settings.paths.reference / "EMSR927" / d["file"]
        assert d["bytes"] == f.stat().st_size and d["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert d["downloaded_at_utc"] == T0.isoformat()
    saved = json.loads((world.cfg.settings.paths.reference / "EMSR927" / "acquisition_manifest.json").read_text())
    assert saved["downloads"][0]["redirect_chain"] == m["downloads"][0]["redirect_chain"]


def test_destroyed_road_far_from_bato_roads_is_reported_as_unmatched_coverage(world):
    """End to end: an extra Destroyed road ~557 m from every BATO road leaves the matched-domain matrix unchanged and
    appears in reference_coverage (and in the pooled result), so it can neither vanish nor inflate recall."""
    world.opener = FakeOpener(activation({}), standard_zips(far_destroyed=True))
    frozen_world(world)
    out, res = evaluate(world.cfg, "aoi-test", "ts-test", read_state=world.reader)
    pri = res["per_product"][0]["roads"]["comparisons"]["primary_destroyed"]
    assert pri["modes"]["observed_only"]["counts"] == {"TP": 3, "FP": 0, "FN": 3, "TN": 5, "N": 11}
    cov = pri["reference_coverage"]
    assert cov["ems_positive_features_in_area"] == 3 and cov["ems_positive_features_unmatched"] == 1
    assert cov["unmatched_no_eligible_bato_segment_within_distance_m"] == pytest.approx(393, abs=10)
    assert cov["matched_positive_length_fraction"] == pytest.approx(
        cov["ems_positive_length_matched_m"] / cov["ems_positive_length_in_area_m"], abs=1e-3)
    assert cov["matched_positive_length_fraction"] < 0.5
    pooled = res["pooled"]["roads"]["primary_destroyed"]["reference_coverage"]
    assert pooled["ems_positive_features_unmatched"] == 1
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert "Reference coverage:" in md and "recall (matched domain)" in md and "NOT full-network recall" in md

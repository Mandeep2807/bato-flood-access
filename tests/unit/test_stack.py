"""Step 3.5 temporal stack: grid planning, temporal roles, input checks, and full runs on three
Step 3.4 outputs made by the real Step 3.2-3.4 code with fake gpts. No SNAP, no real imagery.
Raster tests need rasterio."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from bato.config import load_config
from bato.contracts.common import BBox
from bato.contracts.fetch import PlannedCandidate
from bato.contracts.io import load_contract
from bato.contracts.stack import StackProvenance
from bato.contracts.temporal import TrackKey
from bato.exceptions import ProhibitedInputError
from bato.preprocessing import snap
from bato.preprocessing import stack as stk
from bato.preprocessing.dimap import read_dimap
from bato.preprocessing.terrain_validation import lonlat_to_3857

from snap_fakes import TRISHULI_POST, TRISHULI_PRE1, make_project
from stack_fakes import (
    PASSES,
    TRISHULI_PRE2,
    config_with,
    published_terrain_set,
    set_pixel,
    stage_root,
    terrain_band,
    tree_hashes,
)
from terrain_fakes import WKT_4326, write_map_dimap

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")
KS_WKT = snap.graph_operators(snap.render_graph(snap.graph_path("s1_terrain.xml").read_text(encoding="utf-8"), {
    "input_file": "a", "source_bands": "b", "dem_file": "c", "dem_nodata": -1.0, "dem_apply_egm": True,
    "output_file": "d"}))[1][2]["mapProjection"]
UTC = timezone.utc


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


@pytest.fixture
def published(tmp_path_factory, gpt):
    return published_terrain_set(tmp_path_factory.mktemp("k"), gpt)


# ---- grid planning (pure) ---------------------------------------------------------------------------

AOI = BBox(min_lon=85.30, min_lat=28.10, max_lon=85.31, max_lat=28.11)


def source(x0: float, y0: float, w: int = 400, h: int = 400, px: float = 10.0) -> stk.SourceGrid:
    return stk.SourceGrid(i2m=(px, 0.0, 0.0, -px, x0, y0), width=w, height=h)


def _origin() -> tuple[float, float]:
    x, y = lonlat_to_3857(np.array([85.29]), np.array([28.12]))
    return math.floor(x[0] / 10) * 10 - 5.0, math.ceil(y[0] / 10) * 10 + 5.0   # SNAP: centres on the grid


def test_window_is_a_whole_pixel_offset_in_every_source():
    x0, y0 = _origin()
    sources = [source(x0, y0), source(x0 + 30, y0 - 20, w=390), source(x0 - 50, y0 + 70, w=420, h=420)]
    grid, windows = stk.plan_stack_grid(sources, AOI, 0.1)
    gx, gy = grid.transform[2], grid.transform[5]
    for s, (col, row, w, h) in zip(sources, windows, strict=True):
        assert (gx - s.i2m[4]) / 10 == col and (s.i2m[5] - gy) / 10 == row and (w, h) == (grid.width, grid.height)
    fx, fy = grid.standard_grid_offsets
    assert fx == round(fx) and fy == round(fy)                     # stack keeps SNAP's standard grid
    (xmin, xmax), (ymin, ymax) = lonlat_to_3857(np.array([grid.analysis_bbox.min_lon, grid.analysis_bbox.max_lon]),
                                                np.array([grid.analysis_bbox.min_lat, grid.analysis_bbox.max_lat]))
    b = grid.bounds_3857
    assert b[0] <= xmin and b[2] >= xmax and b[1] <= ymin and b[3] >= ymax          # covers AOI + buffer
    assert b[0] > xmin - 10 and b[2] < xmax + 10                                    # snapped outward by < 1 px


@pytest.mark.parametrize("bad, match", [
    (lambda x0, y0: source(x0 + 5, y0), "not on the same pixel grid"),         # half-pixel phase shift
    (lambda x0, y0: source(x0 + 3, y0), "not on the same pixel grid"),
    (lambda x0, y0: source(x0, y0, px=20.0), "pixel geometry"),
    (lambda x0, y0: source(x0 + 2000, y0), "not fully inside"),                # window leaves the raster
    (lambda x0, y0: source(x0, y0, w=50), "not fully inside"),
])
def test_grid_mismatches_fail_closed(bad, match):
    x0, y0 = _origin()
    with pytest.raises(stk.StackError, match=match):
        stk.plan_stack_grid([source(x0, y0), bad(x0, y0), source(x0, y0)], AOI, 0.1)


def test_buffer_beyond_the_sources_fails_closed():
    x0, y0 = _origin()
    with pytest.raises(stk.StackError, match="not fully inside"):
        stk.plan_stack_grid([source(x0, y0)] * 3, AOI, 5.0)


# ---- temporal roles (pure) --------------------------------------------------------------------------

def _candidate(pre=(TRISHULI_PRE1, TRISHULI_PRE2), post=TRISHULI_POST):
    from bato.contracts.common import OrbitDirection
    return PlannedCandidate(candidate_id="ts-test", track=TrackKey(orbit_direction=OrbitDirection.ASCENDING,
                                                                   relative_orbit=85),
                            rank=1, pre_event_pass_ids=pre, post_event_pass_id=post)


@pytest.fixture
def plan_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("r")
    return make_project(load_config(config_with(root)), present_passes=()).manifest


STARTS = {TRISHULI_PRE1: datetime(2026, 6, 29, 12, 21, 38, tzinfo=UTC),
          TRISHULI_PRE2: datetime(2026, 8, 16, 12, 21, 41, tzinfo=UTC),
          TRISHULI_POST: datetime(2026, 8, 28, 12, 21, 41, tzinfo=UTC)}
TRACKS = dict.fromkeys(PASSES, ("ascending", 85))


def test_real_trishuli_roles_are_accepted(plan_manifest):
    stk.check_temporal_roles(_candidate(), plan_manifest, STARTS, TRACKS, date(2026, 8, 26))


@pytest.mark.parametrize("change, match", [
    ({"event": date(2026, 8, 16)}, "PRE-2 is dated 2026-08-16, not before the event date"),
    ({"event": date(2026, 8, 28)}, "POST is dated 2026-08-28, not after the event date"),
    ({"event": date(2026, 9, 1)}, "not after the event date"),
    ({"starts": {**STARTS, TRISHULI_PRE1: datetime(2026, 8, 20, tzinfo=UTC)}}, "not in PRE-1 < PRE-2 < POST"),
    ({"tracks": {**TRACKS, TRISHULI_POST: ("descending", 19)}}, "not on the candidate track"),
    ({"candidate": _candidate(pre=(TRISHULI_PRE1, TRISHULI_POST), post=TRISHULI_PRE2)}, "plan role"),
])
def test_wrong_roles_order_event_or_track_fail(plan_manifest, change, match):
    with pytest.raises(stk.StackError, match=match):
        stk.check_temporal_roles(change.get("candidate", _candidate()), plan_manifest, change.get("starts", STARTS),
                                 change.get("tracks", TRACKS), change.get("event", date(2026, 8, 26)))


def test_event_time_is_used_when_known(plan_manifest):
    from datetime import time
    stk.check_temporal_roles(_candidate(), plan_manifest, STARTS, TRACKS, date(2026, 8, 28), time(6, 0))
    with pytest.raises(stk.StackError, match="event time"):
        stk.check_temporal_roles(_candidate(), plan_manifest, STARTS, TRACKS, date(2026, 8, 28), time(13, 0))


def test_candidate_selection_and_shape(plan_manifest):
    assert stk.select_candidate(plan_manifest, None).candidate_id == plan_manifest.plan.candidates[0].candidate_id
    with pytest.raises(stk.StackError, match="not in the manifest"):
        stk.select_candidate(plan_manifest, "ts-nope")
    two = plan_manifest.model_copy(update={"plan": SimpleNamespace(candidates=(
        _candidate(), _candidate().model_copy(update={"candidate_id": "ts-test2"})))})
    with pytest.raises(stk.StackError, match="choose one with --candidate"):
        stk.select_candidate(two, None)


# ---- terrain product checks (pure) --------------------------------------------------------------------

def _product(tmp_path, **kw):
    bands = kw.pop("bands", {"Sigma0_VV": np.ones((4, 5), np.float32), "Sigma0_VH": np.ones((4, 5), np.float32)})
    args = dict(bands=bands, x0=9495205.0, y0=3263235.0, pixel=10.0, wkt=KS_WKT, history=["Terrain-Correction"],
                abstracted={"is_terrain_corrected": 1})
    args.update(kw)
    write_map_dimap(tmp_path / "t.dim", **args)
    from bato.preprocessing.terrain_validation import dimap_map_geocoding
    wkt, i2m = dimap_map_geocoding(tmp_path / "t.dim")
    return read_dimap(tmp_path / "t.dim"), wkt, i2m


def test_valid_terrain_product_passes(tmp_path):
    product, wkt, i2m = _product(tmp_path)
    assert stk.check_terrain_product(product, wkt, i2m)[4] == 9495205.0


@pytest.mark.parametrize("kw, match", [
    ({"unit": "intensity_db"}, "dB"),
    ({"wkt": WKT_4326}, "EPSG:3857"),
    ({"wkt": None}, "EPSG:3857"),
    ({"pixel": 20.0}, "pixel geometry"),
    ({"i2m": False}, "no map geocoding"),
    ({"nodata": -9999.0}, "no-data value"),
    ({"bands": {"Sigma0_VV": np.ones((4, 5), np.float32)}}, "expected exactly"),
    ({"bands": {"Sigma0_VV": np.ones((4, 5), np.float32), "Sigma0_VH": np.ones((4, 5), np.float32),
                "Sigma0_VV_db": np.ones((4, 5), np.float32)}}, "expected exactly"),
])
def test_invalid_terrain_products_are_refused(tmp_path, kw, match):
    product, wkt, i2m = _product(tmp_path, **kw)
    with pytest.raises(stk.StackError, match=match):
        stk.check_terrain_product(product, wkt, i2m)


# ---- full runs (rasterio) -------------------------------------------------------------------------------

@needs_rasterio
def test_stack_copies_three_passes_onto_one_grid(published):
    import rasterio
    upstream = {pid: tree_hashes(stage_root(published) / pid) for pid in PASSES}
    raw = tree_hashes(published.cfg.settings.paths.raw)
    job = stk.prepare_stack(published.manifest, published.cfg)
    assert [i.role for i in job.inputs] == ["PRE1", "PRE2", "POST"] and [i.pass_id for i in job.inputs] == list(PASSES)
    assert len({i.i2m[4] for i in job.inputs}) == 3                       # different extents, same lattice
    result = stk.run_stack(job, published.cfg)
    out = result.output_dir
    assert out == stage_root(published) / "stack" / job.candidate.candidate_id
    assert {p.name for p in out.iterdir()} == {"stack_sigma0.tif", "valid_mask.tif", "preprocessing_provenance.json"}
    prov = load_contract(StackProvenance, result.provenance_path)
    assert prov == result.provenance and prov.status == "succeeded" and prov.validation.passed
    assert [x.name for x in prov.layers] == ["PRE1_VV", "PRE1_VH", "PRE2_VV", "PRE2_VH", "POST_VV", "POST_VH"]
    assert [i.days_from_event for i in prov.inputs] == pytest.approx([-57.485, -9.485, 2.515], abs=1e-3)
    g = prov.grid
    with rasterio.open(out / "stack_sigma0.tif") as ds:
        assert ds.count == 6 and ds.crs.to_epsg() == 3857 and ds.nodata == 0.0
        assert tuple(ds.descriptions) == tuple(x.name for x in prov.layers)
        assert ds.tags(5)["role"] == "POST" and ds.tags(5)["pass_id"] == TRISHULI_POST
        stack = {x.name: ds.read(x.index) for x in prov.layers}
    for inp in prov.inputs:                                          # values copied bit for bit, no resampling
        col, row, w, h = inp.window
        for pol in ("VV", "VH"):
            src, i2m = terrain_band(published, inp.pass_id, pol)
            assert np.array_equal(stack[f"{inp.role}_{pol}"], src[row:row + h, col:col + w])
            assert (i2m[4] + col * 10, i2m[5] - row * 10) == (g.transform[2], g.transform[5])
    with rasterio.open(out / "valid_mask.tif") as ds:
        mask = ds.read(1)
    assert mask.dtype == np.uint8 and int(mask.sum()) == prov.validation.mask_valid_pixels == g.width * g.height
    assert prov.validation.aoi_valid_fraction == 1.0
    assert prov.output.stack_sha256 == hashlib.sha256((out / "stack_sigma0.tif").read_bytes()).hexdigest()
    assert {pid: tree_hashes(stage_root(published) / pid) for pid in PASSES} == upstream     # inputs untouched
    assert tree_hashes(published.cfg.settings.paths.raw) == raw
    assert all(st.unit == "intensity" and st.negative_pixels == 0 for st in prov.validation.layers)
    assert prov.open_specification_items and "DEM" in prov.dem_features


@needs_rasterio
def test_stack_is_deterministic_and_never_deletes(published):
    first = stk.run_stack(stk.prepare_stack(published.manifest, published.cfg), published.cfg)
    with pytest.raises(stk.StackError, match="already exists"):
        stk.run_stack(stk.prepare_stack(published.manifest, published.cfg), published.cfg)
    second = stk.run_stack(stk.prepare_stack(published.manifest, published.cfg), published.cfg, overwrite=True)
    assert first.provenance.output.stack_sha256 == second.provenance.output.stack_sha256
    assert first.provenance.output.mask_sha256 == second.provenance.output.mask_sha256
    assert [x.copied_sha256 for x in first.provenance.layers] == [x.copied_sha256 for x in second.provenance.layers]
    assert len(list((stage_root(published) / "stack").glob("*.superseded-*"))) == 1


@needs_rasterio
def test_missing_pass_output_fails_closed(tmp_path_factory, gpt):
    project = published_terrain_set(tmp_path_factory.mktemp("m"), gpt, skip_terrain=(TRISHULI_POST,))
    with pytest.raises(stk.StackError, match=f"POST \\({TRISHULI_POST}\\): no published Step 3.4 output.*"
                                             "terrain_correct_s1.py"):
        stk.prepare_stack(project.manifest, project.cfg)


@needs_rasterio
def test_changed_input_or_other_settings_fail_closed(published):
    other = published.cfg.model_copy(update={"settings": published.cfg.settings.model_copy(update={
        "preprocessing": published.cfg.settings.preprocessing.model_copy(update={
            "s1_terrain": published.cfg.settings.preprocessing.s1_terrain.model_copy(
                update={"dem_margin_km": 30.0})})})})
    with pytest.raises(stk.StackError, match="different terrain/speckle settings"):
        stk.prepare_stack(published.manifest, other)
    img = next((stage_root(published) / TRISHULI_PRE2 / "terrain").glob("*.data")) / "Sigma0_VH.img"
    img.write_bytes(img.read_bytes() + b"\0")
    with pytest.raises(stk.StackError, match="PRE2.*changed since Step 3.4"):
        stk.prepare_stack(published.manifest, published.cfg)


@needs_rasterio
@pytest.mark.parametrize("value, failed", [(np.nan, "finite_PRE2_VV"), (np.inf, "finite_PRE2_VV"),
                                           (-12.5, "non_negative_PRE2_VV")])
def test_nonfinite_or_negative_values_fail_closed(tmp_path_factory, gpt, value, failed):
    project = published_terrain_set(tmp_path_factory.mktemp("n"), gpt, after_run={TRISHULI_PRE2: set_pixel(value)})
    with pytest.raises(stk.StackRunFailed) as info:
        stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    prov = info.value.provenance
    assert prov.validation is not None and failed in {c.name for c in prov.validation.failed_checks()}
    assert prov.output is None and ".failed-" in info.value.output_dir.name
    assert not (stage_root(project) / "stack" / prov.candidate_id).exists()            # never published


@needs_rasterio
def test_analysis_buffer_beyond_the_inputs_fails_closed(tmp_path_factory, gpt):
    project = published_terrain_set(tmp_path_factory.mktemp("b"), gpt)
    aoi = project.cfg.settings.aoi.model_copy(update={"analysis_buffer_km": 5.0})
    cfg = project.cfg.model_copy(update={"settings": project.cfg.settings.model_copy(update={"aoi": aoi})})
    with pytest.raises(stk.StackError, match="not fully inside"):
        stk.prepare_stack(project.manifest, cfg)


def test_dry_run_and_reference_folders_are_refused(tmp_path_factory, config_copy):
    cfg = load_config(config_copy("config.yaml", lambda d: d["paths"].update(interim="data/reference/interim")))
    project = make_project(cfg, present_passes=())
    with pytest.raises(stk.StackError, match="dry-run"):
        stk.prepare_stack(project.manifest.model_copy(update={"dry_run": True}), cfg)
    with pytest.raises(ProhibitedInputError):
        stk.prepare_stack(project.manifest, cfg)

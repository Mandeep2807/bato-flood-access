"""Step 4B FloodViT inference core, without torch: deterministic FAKE models (numpy functions, TEST FIXTURE)
stand in for the network, on Step 3.5 stacks made by the real Step 3.2-3.5 code with fake gpts or replaced
by synthetic values. Covers: no padding when avoidable, invalid -> 3 / -1, class 1 vs class 2 statistics,
channel order POST/PRE2/PRE1, clamp + normalisation path, determinism, seams/overlap averaging, no-event run."""

from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from bato.contracts.inference import CLASS_CODES, FloodInferenceProvenance, InferenceStatistics, ModelRecord
from bato.contracts.io import load_contract, save_contract
from bato.exceptions import ProhibitedInputError
from bato.inference import floodvit as fv
from bato.preprocessing import model_input as mi
from bato.preprocessing import snap
from bato.preprocessing import stack as stk

from stack_fakes import published_terrain_set

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")
T = 224
MEAN = np.array([0.0953, 0.0264] * 3, dtype=np.float32)[:, None, None]
STD = np.array([0.0427, 0.0215] * 3, dtype=np.float32)[:, None, None]
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def reference_normalize(x: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(np.clip(x.astype(np.float32), np.float32(0.0), np.float32(0.15)), nan=np.float32(0.15))
    return (y - MEAN) / STD


# ---- fake models (TEST FIXTURE) -----------------------------------------------------------------------------


def change_model(x: np.ndarray) -> np.ndarray:
    """Pixel-local: flood where POST_VV dropped below PRE2_VV, permanent water where both are dark."""
    post, pre2 = x[:, 0], x[:, 2]
    logits = np.zeros((x.shape[0], 3, T, T), dtype=np.float32)
    logits[:, 1] = 3 * (-1.0 - np.maximum(post, pre2))
    logits[:, 2] = 3 * (pre2 - post - 1.0)
    return logits


def position_model(x: np.ndarray) -> np.ndarray:
    """Class-2 logit = column index inside the tile / 223 (exposes how overlapping tiles are merged)."""
    logits = np.full((x.shape[0], 3, T, T), 0.5, dtype=np.float32)
    logits[:, 2] = np.arange(T, dtype=np.float32)[None, None, :] / 223.0
    return logits


class Probe:
    """Records every model input batch; answers with constant 'no water' logits."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs.append(x.copy())
        out = np.zeros((x.shape[0], 3, T, T), dtype=np.float32)
        out[:, 0] = 1.0
        return out


# ---- stacks -------------------------------------------------------------------------------------------------


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


def stacked(root, gpt):
    project = published_terrain_set(root, gpt)
    return project, stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)


def write_synthetic_stack(project, result, data: np.ndarray, mask: np.ndarray) -> None:
    """Replace the published stack by synthetic values (TEST FIXTURE) and keep its provenance consistent."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    root = project.cfg.config_dir.parent
    prov = result.provenance
    _c, h, w = data.shape
    stack_path, mask_path = root / prov.output.stack, root / prov.output.mask
    with rasterio.open(stack_path, "w", driver="GTiff", width=w, height=h, count=6, dtype="float32",
                       crs=CRS.from_epsg(3857), transform=Affine(*prov.grid.transform), nodata=0.0) as dst:
        for k, name in enumerate(mi.STACK_BANDS):
            dst.write(data[k], k + 1)
            dst.set_band_description(k + 1, name)
    with rasterio.open(mask_path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                       crs=CRS.from_epsg(3857), transform=Affine(*prov.grid.transform)) as dst:
        dst.write(mask, 1)
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (stack_path, mask_path)}
    save_contract(prov.model_copy(update={
        "grid": prov.grid.model_copy(update={"width": w, "height": h}),
        "output": prov.output.model_copy(update={"stack_sha256": sha[stack_path], "mask_sha256": sha[mask_path]})}),
        result.provenance_path)


H, W = 300, 470
FLOOD_BOX = (slice(100, 160), slice(300, 380))     # POST_VV drops (event)
WATER_BOX = (slice(200, 260), slice(40, 120))      # dark in every layer (permanent water)
BRIGHT_BOX = (slice(20, 60), slice(20, 60))        # PRE1_VV far above the 0.15 clamp


def scene() -> tuple[np.ndarray, np.ndarray]:
    """TEST FIXTURE scene: background VV 0.08 / VH 0.02, one flood box, one water box, a 10-row POST_VV no-data strip,
    and one pixel no-data in every layer."""
    data = np.empty((6, H, W), dtype=np.float32)
    data[0::2], data[1::2] = 0.08, 0.02
    data[4][FLOOD_BOX] = 0.01
    data[0::2, WATER_BOX[0], WATER_BOX[1]] = 0.005
    data[0][BRIGHT_BOX] = 0.6
    data[4, :10, :] = 0.0
    data[:, H - 1, W - 1] = 0.0
    return data, np.all(data != 0.0, axis=0).astype(np.uint8)


def candidate_id(project) -> str:
    return stk.select_candidate(project.manifest, None).candidate_id


def checkpoint(project):
    path = project.cfg.config_dir.parent / "models" / "TEST_FIXTURE_floodvit.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"TEST FIXTURE - not a real checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, ModelRecord(checkpoint=path.as_posix(), checkpoint_sha256=digest)


def run(project, model, *, mode="event", stride=224, overwrite=False, now=lambda: NOW):
    path, record = checkpoint(project)
    job = fv.prepare_inference(project.cfg, project.manifest.aoi_id, candidate_id(project), path, mode=mode,
                               stride=stride)
    return job, fv.run_inference(job, project.cfg, model, record, batch_size=3, overwrite=overwrite, now=now)


def read(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.tags()


@pytest.fixture
def synthetic_project(tmp_path_factory, gpt):
    project, result = stacked(tmp_path_factory.mktemp("f"), gpt)
    data, mask = scene()
    write_synthetic_stack(project, result, data, mask)
    return project, data, mask


# ---- 1. no padding when avoidable ---------------------------------------------------------------------------


@needs_rasterio
def test_edge_tiles_are_shifted_not_padded(synthetic_project):
    project, data, _mask = synthetic_project
    probe = Probe()
    _job, out = run(project, probe)
    t = out.provenance.tiling
    assert (t.row_offsets, t.col_offsets) == ((0, 76), (0, 224, 246)) and t.tiles == 6 and t.padded_tiles == 0
    assert t.edge_policy == "shift_to_fit"
    seen = np.concatenate(probe.inputs)
    expected = [reference_normalize(data[[4, 5, 2, 3, 0, 1], r:r + T, c:c + T])
                for r in t.row_offsets for c in t.col_offsets]
    assert seen.shape == (6, 6, T, T)
    for got, want in zip(seen, expected, strict=True):
        assert np.array_equal(got, want)                     # every model input is real stack data, no pad value
    assert t.overlap_pixels == 148 * W + 202 * H - 148 * 202


@needs_rasterio
def test_padding_only_when_the_stack_is_smaller_than_a_tile(tmp_path_factory, gpt):
    project, _result = stacked(tmp_path_factory.mktemp("p"), gpt)          # 135 x 150 stack
    _job, out = run(project, Probe())
    assert out.provenance.tiling.padded_tiles == 1 and out.provenance.tiling.tiles == 1


# ---- 2. invalid pixels -> ignore / no-data ------------------------------------------------------------------


@needs_rasterio
def test_invalid_pixels_become_ignore_and_nodata(synthetic_project):
    project, _data, mask = synthetic_project
    _job, out = run(project, change_model)
    pred, pred_nodata, tags = read(out.output_dir / fv.PREDICTION_FILE)
    prob, prob_nodata, _ = read(out.output_dir / fv.PROBABILITY_FILE)
    assert (pred_nodata, prob_nodata) == (3, -1.0) and tags["validated"] == "false"
    assert np.array_equal(pred == 3, mask == 0) and np.array_equal(prob == -1.0, mask == 0)
    assert set(np.unique(pred[mask == 1]).tolist()) <= {0, 1, 2}
    s = out.provenance.statistics
    assert s.invalid_pixels == int((mask == 0).sum()) == 10 * W + 1 and s.valid_pixels == int(mask.sum())
    assert s.invalid_raw_argmax["2"] == 10 * W           # the model WOULD call the POST no-data strip flood
    assert any("invalid pixels written as 3" in w for w in out.provenance.warnings)


# ---- 3. class 1 and class 2 statistics are separate ---------------------------------------------------------


@needs_rasterio
def test_permanent_water_and_flood_are_counted_separately(synthetic_project):
    project, _data, mask = synthetic_project
    _job, out = run(project, change_model)
    pred, _nd, _tags = read(out.output_dir / fv.PREDICTION_FILE)
    s = out.provenance.statistics
    flood, water = 60 * 80, 60 * 80
    assert (s.flood_pixels, s.permanent_water_pixels) == (flood, water)
    assert int((pred[FLOOD_BOX] == 2).sum()) == flood and int((pred[WATER_BOX] == 1).sum()) == water
    assert s.no_water_pixels == s.valid_pixels - flood - water
    assert s.flood_fraction == round(flood / s.valid_pixels, 8) and s.permanent_water_fraction == round(
        water / s.valid_pixels, 8) and s.water_fraction == round((flood + water) / s.valid_pixels, 8)
    assert s.total_pixels == H * W and int(mask.sum()) == s.valid_pixels
    with pytest.raises(ValidationError, match="class counts"):
        InferenceStatistics.model_validate({**s.model_dump(), "flood_pixels": s.flood_pixels + 1})


# ---- 4. channel order and 5. normalisation / clamp ----------------------------------------------------------


CODES = {"PRE1_VV": 0.011, "PRE1_VH": 0.012, "PRE2_VV": 0.021, "PRE2_VH": 0.022, "POST_VV": 0.031, "POST_VH": 0.032}


def coded(mode: str) -> list[str]:
    raw = np.stack([np.full((T, T), CODES[n], dtype=np.float32) for n in mi.STACK_BANDS])
    x = fv.model_input(raw, np.ones((T, T), dtype=np.uint8), mi.kuro_siwo_spec(), mode)
    recovered = (x * STD + MEAN)[:, 0, 0]
    return [min(CODES, key=lambda n: abs(CODES[n] - v)) for v in recovered]


def test_channel_order_post_pre2_pre1():
    assert coded("event") == ["POST_VV", "POST_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"]
    assert fv.mapping_lines(mi.kuro_siwo_spec(), "event")[2] == "2 PRE2_VV <- stack PRE2_VV"
    with pytest.raises(fv.InferenceError, match="unknown inference mode"):
        fv.model_bands(mi.kuro_siwo_spec(), "reverse")


@needs_rasterio
def test_clamp_and_normalisation_path(synthetic_project):
    project, data, _mask = synthetic_project
    probe = Probe()
    run(project, probe)
    first = probe.inputs[0][0]                                    # tile (0, 0): holds the bright PRE1_VV box
    clamp_vv = (np.float32(0.15) - np.float32(0.0953)) / np.float32(0.0427)
    assert first.dtype == np.float32 and first[4][BRIGHT_BOX].max() == clamp_vv == first[4].max()
    assert np.array_equal(first, reference_normalize(data[[4, 5, 2, 3, 0, 1], :T, :T]))
    zero_vv = (np.float32(0.0) - np.float32(0.0953)) / np.float32(0.0427)
    assert first[0, 0, 5] == zero_vv                               # POST no-data enters as 0.0 (Kuro convention)


# ---- 6. determinism -----------------------------------------------------------------------------------------


@needs_rasterio
def test_repeated_run_gives_identical_results(synthetic_project):
    project, _data, _mask = synthetic_project
    _job, first = run(project, change_model)
    pred1, _a, _b = read(first.output_dir / fv.PREDICTION_FILE)
    prob1, _a, _b = read(first.output_dir / fv.PROBABILITY_FILE)
    later = datetime(2026, 10, 3, 13, tzinfo=timezone.utc)
    _job, second = run(project, change_model, overwrite=True, now=lambda: later)
    pred2, _a, _b = read(second.output_dir / fv.PREDICTION_FILE)
    prob2, _a, _b = read(second.output_dir / fv.PROBABILITY_FILE)
    assert np.array_equal(pred1, pred2) and np.array_equal(prob1, prob2)
    assert first.provenance.statistics == second.provenance.statistics
    assert len(list(first.output_dir.parent.glob("floodvit.superseded-*"))) == 1     # first run kept, not deleted


# ---- 7. seams / overlap -------------------------------------------------------------------------------------


@needs_rasterio
def test_overlapping_tiles_are_averaged_and_seams_reported(synthetic_project):
    project, _data, mask = synthetic_project
    _job, out = run(project, position_model)
    prob, _nd, _tags = read(out.output_dir / fv.PROBABILITY_FILE)

    def p2(logit2: np.ndarray) -> np.ndarray:
        e = np.exp(np.stack([np.full_like(logit2, 0.5), np.full_like(logit2, 0.5), logit2]))
        return (e[2] / e.sum(axis=0)).astype(np.float32)

    row = 150
    cols = np.arange(W)
    single = cols < 246                                             # covered by tile 0 or tile 224 only
    pos = np.where(cols < 224, cols, cols - 224).astype(np.float32)
    expected = p2(pos / 223.0)
    both = cols >= 246                                              # tiles at 224 and 246 overlap here
    expected[both] = p2(((cols[both] - 224) + (cols[both] - 246)).astype(np.float32) / 2 / 223.0)
    expected[cols >= 448] = p2((cols[cols >= 448] - 246).astype(np.float32) / 223.0)   # only the last tile
    valid_row = mask[row] == 1
    assert np.allclose(prob[row][valid_row], expected[valid_row], atol=1e-6)
    assert np.allclose(prob[row][single & valid_row], expected[single & valid_row], atol=1e-6)
    near = fv.edge_band_mask(W, (0, 224, 246), 224, 16)
    assert near[208:240].all() and near[230:262].all() and not near[:208].any() and not near[262:208 + 224].any()
    assert not fv.edge_band_mask(224, (0,), 224, 16).any()          # stack edges are not seams
    s = out.provenance.statistics
    assert s.seam_band_px == 16 and s.flood_fraction_near_tile_edges is not None


# ---- 8. no-event sanity run ---------------------------------------------------------------------------------


@needs_rasterio
def test_no_event_run_removes_the_change_signal(synthetic_project):
    project, _data, _mask = synthetic_project
    _job, event = run(project, change_model)
    job, no_event = run(project, change_model, mode="no_event")
    assert no_event.output_dir.name == "floodvit_no_event" and event.output_dir.name == "floodvit"
    assert coded("no_event") == ["PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH", "PRE1_VV", "PRE1_VH"]
    assert event.provenance.statistics.flood_pixels == 60 * 80
    pred, _nd, _tags = read(no_event.output_dir / fv.PREDICTION_FILE)
    assert int((pred[FLOOD_BOX] == 2).sum()) == 0                                   # the event change is gone
    # what remains is change BETWEEN the two pre-event images (the PRE1 bright box): a false-positive baseline
    assert no_event.provenance.statistics.flood_pixels == int((pred[BRIGHT_BOX] == 2).sum()) == 40 * 40
    assert no_event.provenance.statistics.permanent_water_pixels == 60 * 80          # water stays water
    assert no_event.provenance.mode == "no_event" and no_event.provenance.model_input_mapping[0] == \
        "0 POST_VV <- stack PRE2_VV"
    assert job.final_dir != event.output_dir


# ---- provenance and refusals --------------------------------------------------------------------------------


@needs_rasterio
def test_provenance_records_hashes_spec_and_not_validated(synthetic_project):
    project, _data, _mask = synthetic_project
    job, out = run(project, change_model)
    prov = load_contract(FloodInferenceProvenance, out.provenance_path)
    assert prov == out.provenance and prov.validated is False and "NOT VALIDATED" in prov.validation_note
    assert prov.class_codes == CLASS_CODES and "UNCALIBRATED" in prov.probability
    root = project.cfg.config_dir.parent
    assert prov.outputs.prediction_sha256 == hashlib.sha256((root / prov.outputs.prediction).read_bytes()).hexdigest()
    assert prov.source.stack_sha256 == job.model_input.provenance.output.stack_sha256
    assert prov.spec == mi.kuro_siwo_spec() and prov.spec_fingerprint == mi.spec_fingerprint(prov.spec)
    assert sorted(p.name for p in out.output_dir.iterdir()) == sorted(
        [fv.PREDICTION_FILE, fv.PROBABILITY_FILE, fv.PROVENANCE_FILE])
    lines = fv.describe_inference(prov)
    assert any("NOT VALIDATED" in x for x in lines) and any("class 1" in x for x in lines)


@needs_rasterio
@pytest.mark.parametrize(("model", "message"), [
    (lambda x: np.full((x.shape[0], 3, T, T), np.nan, dtype=np.float32), "NaN/inf"),
    (lambda x: np.zeros((x.shape[0], 2, T, T), dtype=np.float32), "output shape"),
])
def test_bad_model_output_publishes_nothing(synthetic_project, model, message):
    project, _data, _mask = synthetic_project
    path, record = checkpoint(project)
    job = fv.prepare_inference(project.cfg, project.manifest.aoi_id, candidate_id(project), path)
    with pytest.raises(fv.InferenceError, match=message):
        fv.run_inference(job, project.cfg, model, record)
    assert not job.final_dir.exists() and not list(job.final_dir.parent.glob("*.tmp-*"))


@needs_rasterio
def test_refusals(synthetic_project):
    project, _data, _mask = synthetic_project
    path, record = checkpoint(project)
    aoi, cid = project.manifest.aoi_id, candidate_id(project)
    with pytest.raises(fv.InferenceError, match="checkpoint not found"):
        fv.prepare_inference(project.cfg, aoi, cid, path.with_name("missing.pt"))
    reference = project.cfg.settings.paths.reference
    reference.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ProhibitedInputError):
        fv.prepare_inference(project.cfg, aoi, cid, reference / "x.pt")
    job = fv.prepare_inference(project.cfg, aoi, cid, path)
    with pytest.raises(fv.InferenceError, match="SHA-256"):
        fv.run_inference(job, project.cfg, change_model, record.model_copy(update={"checkpoint_sha256": "0" * 64}))
    fv.run_inference(job, project.cfg, change_model, record)
    with pytest.raises(fv.InferenceError, match="already exists"):
        fv.run_inference(job, project.cfg, change_model, record)
    stack_file = project.cfg.config_dir.parent / job.model_input.provenance.output.stack
    stack_file.write_bytes(stack_file.read_bytes() + b"\0")
    with pytest.raises(mi.ModelInputError, match="changed since Step 3.5"):
        fv.prepare_inference(project.cfg, aoi, cid, path)

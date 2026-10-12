"""Step 4A model-input contract: specification, channel order, Kuro Siwo normalisation, tiling, validity
and ignore masks, manifest provenance and rejection of unsupported/inconsistent inputs. Stacks come from the
real Step 3.2-3.5 code with fake gpts, or are synthetic (TEST FIXTURE values, never real imagery)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from bato.contracts.io import load_contract, save_contract
from bato.contracts.model_input import (
    IGNORE_LABEL,
    EvidenceStatus,
    KuroSiwoLabel,
    ModelInputManifest,
    ModelInputSpec,
    NormalizationSpec,
    TilingSpec,
    spec_fingerprint,
)
from bato.contracts.stack import StackProvenance
from bato.exceptions import ProhibitedInputError
from bato.preprocessing import model_input as mi
from bato.preprocessing import snap
from bato.preprocessing import stack as stk
from bato.preprocessing.radiometric import PROVENANCE_FILE

from snap_fakes import TRISHULI_POST
from stack_fakes import PASSES, TRISHULI_PRE2, published_terrain_set, set_pixel, stage_root, tree_hashes

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")
SPEC = mi.kuro_siwo_spec()
MODEL_ORDER = ["POST_VV", "POST_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"]
MEAN = np.array([0.0953, 0.0264] * 3, dtype=np.float32)[:, None, None]
STD = np.array([0.0427, 0.0215] * 3, dtype=np.float32)[:, None, None]
T = 224


def reference_normalize(x: np.ndarray) -> np.ndarray:
    """Independent float32 re-statement of Kuro Siwo Dataset.concat + scale_img('normalize')."""
    y = np.clip(x.astype(np.float32), np.float32(0.0), np.float32(0.15))
    y = np.nan_to_num(y, nan=np.float32(0.15))
    return (y - MEAN) / STD


# ---- specification ------------------------------------------------------------------------------------------


def test_channel_order_is_post_then_later_pre_then_earlier_pre():
    assert [c.name for c in SPEC.channels] == MODEL_ORDER
    assert [c.stack_band for c in SPEC.channels] == [5, 6, 3, 4, 1, 2]
    assert [c.kuro_siwo_input for c in SPEC.channels] == [
        "flood_vv (MS1_IVV)", "flood_vh (MS1_IVH)", "sec1_vv (SL1_IVV)", "sec1_vh (SL1_IVH)",
        "sec2_vv (SL2_IVV)", "sec2_vh (SL2_IVH)"]
    assert SPEC.tensor_shape == (6, 224, 224) and SPEC.dtype == "float32" and SPEC.tensor_layout == "C,H,W"
    assert SPEC.dem_features == "none" and mi.KURO_INPUT_BY_ROLE == {"POST": "flood", "PRE2": "sec1", "PRE1": "sec2"}


ROLE_CODES = {"PRE1_VV": 0.011, "PRE1_VH": 0.012, "PRE2_VV": 0.021, "PRE2_VH": 0.022, "POST_VV": 0.031,
              "POST_VH": 0.032}   # TEST FIXTURE: one distinct value per stack band, below the 0.15 clamp


def test_exact_six_channel_order_through_the_full_adapter():
    """Stack layout v1 in, normalised tile out: channels 0-5 must be POST_VV, POST_VH, PRE2_VV, PRE2_VH,
    PRE1_VV, PRE1_VH (PRE2 = later pre-event = Kuro sec1; PRE1 = earlier = Kuro sec2)."""
    raw = np.stack([np.full((T, T), ROLE_CODES[name], dtype=np.float32) for name in mi.STACK_BANDS])
    x, valid = mi.build_tile(raw, np.ones((T, T), dtype=np.uint8), SPEC)
    recovered = (x * STD + MEAN)[:, 0, 0]                                   # undo the z-score
    assert np.allclose(recovered, [0.031, 0.032, 0.021, 0.022, 0.011, 0.012], atol=1e-6)
    assert [min(ROLE_CODES, key=lambda n: abs(ROLE_CODES[n] - v)) for v in recovered] == MODEL_ORDER == [
        "POST_VV", "POST_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"]
    assert valid.all()


def test_kuro_hub_wrapper_reversal_gives_the_spec_order():
    """Kuro Siwo hubconf.py @ 43ec8cf: vit_decoder takes [pre2, pre1, post], each (VV, VH), and feeds
    torch.cat(list(reversed(images))). Its pre1 = sec1 = SL1 = later pre-event = BATO PRE2 (Dataset.py@43ec8cf
    L535-L561), pre2 = sec2 = BATO PRE1. Re-stated with numpy on BATO stack bands."""
    stack = np.stack([np.full((3, 3), ROLE_CODES[name], dtype=np.float32) for name in mi.STACK_BANDS])
    band = dict(zip(mi.STACK_BANDS, stack, strict=True))
    post = np.stack([band["POST_VV"], band["POST_VH"]])
    hub_pre1 = np.stack([band["PRE2_VV"], band["PRE2_VH"]])      # Kuro sec1 / SL1 (later pre-event)
    hub_pre2 = np.stack([band["PRE1_VV"], band["PRE1_VH"]])      # Kuro sec2 / SL2 (earlier pre-event)
    hub_input = [hub_pre2, hub_pre1, post]                        # input_hint "(pre2, pre1, post)"
    wrapper_tensor = np.concatenate(list(reversed(hub_input)), axis=0)
    assert np.array_equal(wrapper_tensor, mi.to_model_order(stack, SPEC))
    assert [round(float(v), 3) for v in wrapper_tensor[:, 0, 0]] == [0.031, 0.032, 0.021, 0.022, 0.011, 0.012]
    naive = np.concatenate([post, stack[0:2], stack[2:4]])          # POST, BATO PRE1, BATO PRE2 (pre names conflated)
    assert not np.array_equal(naive, wrapper_tensor)
    keys = {i.key: i for i in SPEC.items}
    for key in ("hub_wrapper_order", "hub_wrapper_scaling"):
        assert keys[key].status is EvidenceStatus.VERIFIED_FROM_CODE and "43ec8cf" in keys[key].evidence
    assert "POST, PRE2, PRE1" in keys["hub_wrapper_order"].statement


def test_post_pre1_pre2_order_is_refused():
    by_name = {c.name: c for c in SPEC.channels}
    wrong = [by_name[n].model_copy(update={"index": k}).model_dump()
             for k, n in enumerate(["POST_VV", "POST_VH", "PRE1_VV", "PRE1_VH", "PRE2_VV", "PRE2_VH"])]
    with pytest.raises(ValidationError, match="sec1 = later pre-event"):
        ModelInputSpec.model_validate({**SPEC.model_dump(), "channels": wrong})


def test_spec_refuses_any_other_channel_order_or_constants():
    swapped = (SPEC.channels[0], SPEC.channels[1], SPEC.channels[4], SPEC.channels[5], SPEC.channels[2],
               SPEC.channels[3])
    swapped = tuple(c.model_copy(update={"index": k}) for k, c in enumerate(swapped))   # BATO storage-like order
    with pytest.raises(ValidationError, match="sec1 = later pre-event"):
        ModelInputSpec.model_validate({**SPEC.model_dump(), "channels": [c.model_dump() for c in swapped]})
    bad_mean = [c.model_dump() for c in SPEC.channels]
    bad_mean[0]["mean"] = 0.1
    with pytest.raises(ValidationError, match="mean/std"):
        ModelInputSpec.model_validate({**SPEC.model_dump(), "channels": bad_mean})
    with pytest.raises(ValidationError, match="Kuro Siwo values"):
        NormalizationSpec(clamp_max=0.2)
    with pytest.raises(ValidationError, match="224"):
        TilingSpec(tile_size=256, stride=256)
    for stride in (0, 225):
        with pytest.raises(ValidationError, match="stride"):
            TilingSpec(stride=stride)


def test_every_spec_item_is_classified_and_open_items_are_reported():
    keys = {i.key: i for i in SPEC.items}
    assert all(i.evidence and i.statement for i in SPEC.items)
    assert keys["channel_order"].status is EvidenceStatus.VERIFIED_FROM_CODE
    assert keys["sec1_is_later_pre_event"].status is EvidenceStatus.VERIFIED_FROM_DATA
    assert keys["flood_is_post_event"].status is EvidenceStatus.VERIFIED_FROM_DATA
    assert keys["debris_labels"].status is EvidenceStatus.VERIFIED_FROM_CODE
    assert "NO debris" in keys["debris_labels"].statement
    assert keys["dem_channel"].status is EvidenceStatus.VERIFIED_FROM_CODE
    assert keys["tiling_policy"].status is EvidenceStatus.BATO_DESIGN_CHOICE
    unresolved = {k for k, i in keys.items() if not i.resolved}
    assert unresolved == {"released_pixel_units", "zscore_derivation", "dem_source", "radar_visibility",
                          "pretrained_checkpoints"}
    assert keys["label_names"].status is EvidenceStatus.VERIFIED_FROM_CODE
    assert "CLASS_LABELS" in keys["label_names"].evidence
    assert len(SPEC.unresolved_items) == 5 and "5 item(s) unresolved" in SPEC.kuro_siwo_compatibility
    assert "NOT AVAILABLE" in SPEC.validity.radar_visibility and SPEC.validity.ignore_label == IGNORE_LABEL == 3


def test_label_contract_has_no_debris_class():
    assert {c.name: c.value for c in KuroSiwoLabel} == {"NO_WATER": 0, "PERMANENT_WATER": 1, "FLOOD": 2, "IGNORE": 3}
    assert not any("DEBRIS" in c.name or "DAMAGE" in c.name for c in KuroSiwoLabel)


def test_spec_fingerprint_is_stable_and_sensitive():
    assert spec_fingerprint(mi.kuro_siwo_spec()) == spec_fingerprint(SPEC)
    assert spec_fingerprint(mi.kuro_siwo_spec(112)) != spec_fingerprint(SPEC)
    roundtrip = ModelInputSpec.model_validate_json(SPEC.model_dump_json())
    assert roundtrip == SPEC and spec_fingerprint(roundtrip) == spec_fingerprint(SPEC)


# ---- channel order ------------------------------------------------------------------------------------------


def test_stack_layout_is_reordered_to_model_order():
    stack = np.stack([np.full((3, 4), k + 1, dtype=np.float32) for k in range(6)])   # band k holds k+1
    out = mi.to_model_order(stack, SPEC)
    assert [int(out[k, 0, 0]) for k in range(6)] == [5, 6, 3, 4, 1, 2]
    with pytest.raises(mi.ModelInputError, match="layout-v1"):
        mi.to_model_order(stack[:4], SPEC)


def test_kuro_siwo_arrays_and_bato_stack_give_the_same_tensor():
    rng = np.random.default_rng(1)
    bato = {name: rng.uniform(0.0, 0.3, (5, 7)).astype(np.float32) for name in mi.STACK_BANDS}
    kuro = {"flood_vv": bato["POST_VV"], "flood_vh": bato["POST_VH"], "sec1_vv": bato["PRE2_VV"],
            "sec1_vh": bato["PRE2_VH"], "sec2_vv": bato["PRE1_VV"], "sec2_vh": bato["PRE1_VH"]}
    from_stack = mi.to_model_order(np.stack([bato[n] for n in mi.STACK_BANDS]), SPEC)
    assert np.array_equal(mi.from_kuro_siwo_arrays(kuro), from_stack)
    with pytest.raises(mi.ModelInputError, match="missing"):
        mi.from_kuro_siwo_arrays({k: v for k, v in kuro.items() if k != "sec2_vh"})
    with pytest.raises(mi.ModelInputError, match="same shape"):
        mi.from_kuro_siwo_arrays({**kuro, "sec1_vv": np.zeros((5, 8), dtype=np.float32)})


# ---- normalisation ------------------------------------------------------------------------------------------


def test_normalisation_follows_kuro_siwo_clamp_nanfill_zscore():
    vv = np.array([[0.0953, 0.15, 0.3], [-0.01, np.nan, np.inf]], dtype=np.float32)
    vh = np.array([[0.0264, 0.0, 1.0], [0.05, np.nan, -np.inf]], dtype=np.float32)
    x = np.stack([vv, vh] * 3)
    out = mi.normalize(x, SPEC)
    assert out.dtype == np.float32 and out.shape == x.shape and np.isfinite(out).all()
    assert np.array_equal(out, reference_normalize(x))
    f32 = np.float32
    assert out[0, 0, 0] == 0.0 and out[1, 0, 0] == 0.0                               # mean -> 0
    assert out[0, 0, 1] == out[0, 0, 2] == (f32(0.15) - f32(0.0953)) / f32(0.0427)   # 0.3 clamped to 0.15
    assert out[0, 1, 0] == (f32(0.0) - f32(0.0953)) / f32(0.0427)                    # negative clamped to 0
    assert out[0, 1, 1] == out[0, 0, 1]                                              # NaN -> 0.15
    assert out[1, 1, 1] == (f32(0.15) - f32(0.0264)) / f32(0.0215)
    assert out[0, 1, 2] == out[0, 0, 1] and out[1, 1, 2] == out[1, 0, 1]             # +inf -> 0.15, -inf -> 0
    assert np.array_equal(out[2:4], out[0:2]) and np.array_equal(out[4:6], out[0:2])  # same for every time step
    assert np.array_equal(x[0], vv, equal_nan=True)                                   # input not modified
    with pytest.raises(mi.ModelInputError, match="expected a"):
        mi.normalize(x[:2], SPEC)


# ---- tiling -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("length", "stride", "expected"), [
    (1, 224, [0]), (224, 224, [0]), (225, 224, [0, 224]), (448, 224, [0, 224]), (449, 224, [0, 224, 448]),
    (448, 112, [0, 112, 224]), (300, 112, [0, 112]), (224, 1, [0]), (226, 1, [0, 1, 2]),
])
def test_tile_offsets_boundaries(length, stride, expected):
    assert mi.tile_offsets(length, 224, stride) == expected


def test_tiles_cover_every_pixel_without_redundant_tiles():
    for length in (1, 2, 223, 224, 225, 447, 448, 449, 1000, 6180):
        for stride in (1, 7, 112, 200, 224):
            offs = mi.tile_offsets(length, 224, stride)
            covered = np.zeros(length, dtype=bool)
            for o in offs:
                covered[o:o + 224] = True
            assert covered.all() and all(o < length for o in offs)
            assert all(offs[k] + 224 < length for k in range(len(offs) - 1))       # each extra tile was needed
            assert all(b - a == stride for a, b in zip(offs, offs[1:], strict=False))
    for length, stride in ((0, 224), (10, 0), (10, 225)):
        with pytest.raises(mi.ModelInputError):
            mi.tile_offsets(length, 224, stride)


def test_plan_tiles_row_major_with_data_extent():
    tiles = mi.plan_tiles(300, 500, TilingSpec(edge_policy="pad_end"))
    assert [(t.row_off, t.col_off, t.data_height, t.data_width) for t in tiles] == [
        (0, 0, 224, 224), (0, 224, 224, 224), (0, 448, 224, 52),
        (224, 0, 76, 224), (224, 224, 76, 224), (224, 448, 76, 52)]
    assert mi.tile_transform((10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0), tiles[4]) == (
        10.0, 0.0, 1000.0 + 2240.0, 0.0, -10.0, 5000.0 - 2240.0)


@pytest.mark.parametrize(("length", "stride", "expected"), [
    (150, 224, [0]), (224, 224, [0]), (225, 224, [0, 1]), (300, 224, [0, 76]), (448, 224, [0, 224]),
    (449, 224, [0, 224, 225]), (470, 112, [0, 112, 224, 246]), (7316, 224, [*range(0, 7168, 224), 7092]),
    (6186, 224, [*range(0, 6048, 224), 5962]),
])
def test_shift_to_fit_offsets_avoid_padding(length, stride, expected):
    """Default (shift_to_fit): every tile of an axis >= 224 lies fully inside it; the Trishuli stack
    (7316 x 6186) keeps 33 x 28 tiles, none padded."""
    offs = mi.tile_offsets(length, 224, stride, "shift_to_fit")
    assert offs == expected
    if length >= 224:
        assert all(0 <= o and o + 224 <= length for o in offs)                  # no padding when avoidable
    covered = np.zeros(length, dtype=bool)
    for o in offs:
        covered[o:o + 224] = True
    assert covered.all()
    assert all(0 < offs[k + 1] - offs[k] <= stride for k in range(len(offs) - 1))   # the stride is never exceeded
    assert all(offs[k] + 224 < length for k in range(len(offs) - 1))               # each added tile was needed


def test_shift_to_fit_keeps_the_stride_for_every_length_and_stride():
    for length in range(224, 1001, 7):
        for stride in (1, 2, 7, 56, 100, 112, 113, 150, 223, 224):
            offs = mi.tile_offsets(length, 224, stride, "shift_to_fit")
            covered = np.zeros(length, dtype=bool)
            for o in offs:
                covered[o:o + 224] = True
            assert covered.all() and offs[-1] + 224 == length and all(o + 224 <= length for o in offs)
            assert all(0 < b - a <= stride for a, b in zip(offs, offs[1:], strict=False))
            assert offs[:-1] == mi.tile_offsets(length, 224, stride, "pad_end")[:-1]   # only the last tile moves


def test_shift_to_fit_is_the_default_and_pads_only_short_axes():
    assert TilingSpec().edge_policy == "shift_to_fit" == SPEC.tiling.edge_policy
    tiles = mi.plan_tiles(7316, 6186, SPEC.tiling)
    assert len(tiles) == 33 * 28 and all((t.data_height, t.data_width) == (224, 224) for t in tiles)
    assert (tiles[-1].row_off, tiles[-1].col_off) == (7092, 5962)
    short = mi.plan_tiles(150, 500, SPEC.tiling)
    assert [(t.row_off, t.col_off, t.data_height, t.data_width) for t in short] == [
        (0, 0, 150, 224), (0, 224, 150, 224), (0, 276, 150, 224)]                 # rows < 224: padding unavoidable
    with pytest.raises(mi.ModelInputError, match="edge policy"):
        mi.tile_offsets(300, 224, 224, "wrap")


# ---- validity, ignore and no-data ---------------------------------------------------------------------------


def block(h: int = 4, w: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    raw = np.random.default_rng(seed).uniform(0.01, 0.3, (6, h, w)).astype(np.float32)
    raw[4, 0, 0] = 0.0                    # POST_VV no-data, other layers observed
    raw[:, 1, 2] = 0.0                    # no-data in all layers
    valid = np.all(raw != 0.0, axis=0).astype(np.uint8)
    return raw, valid


def test_consistent_block_is_accepted():
    raw, valid = block()
    mi.check_consistency(raw, valid)
    assert valid[0, 0] == 0 and valid[1, 2] == 0 and int(valid.sum()) == 18


@pytest.mark.parametrize(("edit", "message"), [
    (lambda r, v: r.__setitem__((0, 2, 2), np.nan), "NaN/inf"),
    (lambda r, v: r.__setitem__((3, 2, 2), np.inf), "NaN/inf"),
    (lambda r, v: r.__setitem__((1, 2, 2), -0.01), "negative"),
    (lambda r, v: v.__setitem__((2, 2), 2), "other than 0/1"),
    (lambda r, v: v.__setitem__((0, 0), 1), "disagrees"),        # mask says valid where POST_VV is no-data
    (lambda r, v: v.__setitem__((3, 3), 0), "disagrees"),        # mask says invalid where all layers are valid
])
def test_inconsistent_blocks_are_rejected(edit, message):
    raw, valid = block()
    edit(raw, valid)
    with pytest.raises(mi.ModelInputError, match=message):
        mi.check_consistency(raw, valid)


def test_ignore_label_replaces_every_invalid_pixel():
    _raw, valid = block()
    prediction = np.full(valid.shape, KuroSiwoLabel.FLOOD, dtype=np.int64)
    out = mi.apply_ignore(prediction, valid)
    assert np.array_equal(out == IGNORE_LABEL, valid == 0) and (out[valid == 1] == 2).all()
    assert (prediction == 2).all()                                      # input not modified
    with pytest.raises(mi.ModelInputError, match="integer"):
        mi.apply_ignore(prediction.astype(np.float32), valid)
    with pytest.raises(mi.ModelInputError, match="shape"):
        mi.apply_ignore(prediction[:2], valid)
    with pytest.raises(mi.ModelInputError, match="0/1"):
        mi.apply_ignore(prediction, valid * 2)


def test_tile_padding_is_invalid_and_no_data_passes_through_unchanged():
    raw, valid = block(100, 150)
    padded = np.zeros((6, T, T), dtype=np.float32)
    mask = np.zeros((T, T), dtype=np.uint8)
    padded[:, :100, :150], mask[:100, :150] = raw, valid
    x, v = mi.build_tile(padded, mask, SPEC)
    assert x.shape == (6, T, T) and x.dtype == np.float32 and v.dtype == np.uint8
    assert v[100:].sum() == 0 and v[:, 150:].sum() == 0 and int(v.sum()) == int(valid.sum())
    zero = ((np.float32(0.0) - MEAN) / STD)[:, 0, 0]
    assert np.array_equal(x[:, 150, 150], zero) and np.array_equal(x[:, 1, 2], zero)   # padding / no-data -> 0.0
    assert padded[4, 0, 0] == 0.0 and padded[5, 0, 0] > 0.0                         # POST_VV no-data, POST_VH observed
    vh = np.minimum(padded[5, 0, 0], np.float32(0.15))
    assert x[0, 0, 0] == zero[0] and x[1, 0, 0] == (vh - np.float32(0.0264)) / np.float32(0.0215)  # partial kept
    assert np.array_equal(x, reference_normalize(mi.to_model_order(padded, SPEC)))
    with pytest.raises(mi.ModelInputError, match="must be"):
        mi.build_tile(raw, valid, SPEC)


# ---- full runs on Step 3.5 stacks (rasterio) ----------------------------------------------------------------


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


def stacked(root, gpt, **kw):
    project = published_terrain_set(root, gpt, **kw)
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    return project, result


def candidate_id(project) -> str:
    return stk.select_candidate(project.manifest, None).candidate_id


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
    new = prov.model_copy(update={
        "grid": prov.grid.model_copy(update={"width": w, "height": h}),
        "output": prov.output.model_copy(update={"stack_sha256": sha[stack_path], "mask_sha256": sha[mask_path]})})
    save_contract(new, result.provenance_path)


def synthetic(h: int, w: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    data = np.random.default_rng(seed).uniform(0.001, 0.4, (6, h, w)).astype(np.float32)
    data[4, :10, :] = 0.0                                    # POST_VV no-data strip
    data[:, h - 1, w - 1] = 0.0                              # one pixel no-data everywhere
    return data, np.all(data != 0.0, axis=0).astype(np.uint8)


NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@needs_rasterio
def test_manifest_from_a_step_3_5_stack(tmp_path_factory, gpt):
    import rasterio
    project, result = stacked(tmp_path_factory.mktemp("m"), gpt)
    cid = candidate_id(project)
    upstream = {pid: tree_hashes(stage_root(project) / pid) for pid in PASSES}
    stack_tree = tree_hashes(result.output_dir)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, cid)
    out = mi.run_model_input(job, project.cfg, now=lambda: NOW)
    assert out.output_dir == stage_root(project) / "model_input" / cid
    assert [p.name for p in out.output_dir.iterdir()] == ["model_input_manifest.json"]       # no tensors written
    m = load_contract(ModelInputManifest, out.manifest_path)
    assert m == out.manifest and m.spec == SPEC and m.spec_fingerprint == spec_fingerprint(SPEC)
    prov = result.provenance
    assert (m.source.stack_sha256, m.source.mask_sha256) == (prov.output.stack_sha256, prov.output.mask_sha256)
    assert m.source.provenance_sha256 == hashlib.sha256(result.provenance_path.read_bytes()).hexdigest()
    assert m.source.layout_version == "bato-s1-stack-v1"
    assert m.source.stack_config_fingerprint == prov.config_fingerprint
    assert [(p.role, p.pass_id, p.kuro_siwo_input) for p in m.source.passes] == [
        ("PRE1", PASSES[0], "sec2"), ("PRE2", TRISHULI_PRE2, "sec1"), ("POST", TRISHULI_POST, "flood")]
    assert m.config_fingerprint == project.cfg.fingerprint() and m.created_at == NOW
    g = prov.grid
    assert (m.grid.width, m.grid.height) == (g.width, g.height) == (135, 150)
    assert [(t.tile_id, t.data_height, t.data_width, t.valid_pixels) for t in m.tiles] == [
        ("r00000_c00000", 150, 135, 150 * 135)]
    assert m.tiles[0].transform == g.transform and m.valid_pixels == prov.validation.mask_valid_pixels
    assert [d.name for d in m.channel_diagnostics] == MODEL_ORDER
    with rasterio.open(result.output_dir / "stack_sigma0.tif") as ds:
        stack = np.stack([ds.read(k + 1) for k in range(6)])
    for d, c in zip(m.channel_diagnostics, SPEC.channels, strict=True):
        assert d.above_clamp_pixels == int((stack[c.stack_band - 1] > np.float32(0.15)).sum())
    assert tree_hashes(result.output_dir) == stack_tree                                       # stack untouched
    assert {pid: tree_hashes(stage_root(project) / pid) for pid in PASSES} == upstream
    lines = mi.describe_model_input(m)
    assert any("0 POST_VV=flood_vv" in x and "2 PRE2_VV=sec1_vv" in x for x in lines)


@needs_rasterio
def test_reader_tiles_equal_an_independent_computation(tmp_path_factory, gpt):
    import rasterio
    project, result = stacked(tmp_path_factory.mktemp("r"), gpt)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    out = mi.run_model_input(job, project.cfg)
    with rasterio.open(result.output_dir / "stack_sigma0.tif") as ds:
        stack = np.stack([ds.read(k + 1) for k in range(6)])
    expected = np.zeros((6, T, T), dtype=np.float32)
    expected[:, :150, :135] = stack[[4, 5, 2, 3, 0, 1]]
    with mi.ModelInputReader(out.manifest_path, project.cfg) as reader:
        tiles = list(reader)
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.x.shape == (6, T, T) and tile.x.dtype == np.float32 and tile.valid.dtype == np.uint8
    assert np.array_equal(tile.x, reference_normalize(expected))
    assert int(tile.valid.sum()) == 150 * 135 and tile.valid[150:].sum() == 0 and tile.valid[:, 135:].sum() == 0


@needs_rasterio
def test_invalid_pixels_propagate_to_the_validity_and_ignore_masks(tmp_path_factory, gpt):
    project, result = stacked(tmp_path_factory.mktemp("v"), gpt,
                              after_run={TRISHULI_POST: set_pixel(0.0, "Sigma0_VV")})   # POST_VV no-data, VH observed
    assert result.provenance.validation.mask_valid_pixels == 150 * 135 - 1
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    out = mi.run_model_input(job, project.cfg)
    assert out.manifest.valid_pixels == out.manifest.tiles[0].valid_pixels == 150 * 135 - 1
    assert any("invalid (mask 0)" in w for w in out.manifest.warnings)
    with mi.ModelInputReader(out.manifest_path, project.cfg) as reader:
        tile = reader.read("r00000_c00000")
    (r,), (c,) = np.nonzero(tile.valid[:150, :135] == 0)
    assert tile.x[0, r, c] == (np.float32(0.0) - np.float32(0.0953)) / np.float32(0.0427)   # passed through, not hidden
    assert tile.x[1, r, c] != (np.float32(0.0) - np.float32(0.0264)) / np.float32(0.0215)   # VH value kept
    labels = mi.apply_ignore(np.zeros((T, T), dtype=np.uint8), tile.valid)
    assert labels[r, c] == IGNORE_LABEL and labels[150:, :].min() == IGNORE_LABEL and labels[:150, :135].sum() == 3


@needs_rasterio
@pytest.mark.parametrize(("stride", "edge", "expected_tiles"), [
    (224, "shift_to_fit", 6),        # rows [0, 76] x cols [0, 224, 246]
    (112, "shift_to_fit", 8),        # rows [0, 76] x cols [0, 112, 224, 246]
    (224, "pad_end", 6),             # rows [0, 224] x cols [0, 224, 448]
])
def test_multi_tile_synthetic_stack(tmp_path_factory, gpt, stride, edge, expected_tiles):
    project, result = stacked(tmp_path_factory.mktemp("s"), gpt)
    data, mask = synthetic(300, 470)
    write_synthetic_stack(project, result, data, mask)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    m = mi.run_model_input(job, project.cfg, stride=stride, edge_policy=edge).manifest
    assert len(m.tiles) == expected_tiles and m.spec.tiling.stride == stride and m.spec.tiling.overlap == 224 - stride
    assert m.spec.tiling.edge_policy == edge and m.valid_pixels == int(mask.sum())
    last = [(t.row_off, t.col_off, t.data_height, t.data_width) for t in m.tiles][-1]
    if edge == "pad_end":
        assert sum(t.valid_pixels for t in m.tiles) == m.valid_pixels and last == (224, 448, 76, 22)
    else:
        assert sum(t.valid_pixels for t in m.tiles) > m.valid_pixels and last == (76, 246, 224, 224)
        assert all((t.data_height, t.data_width) == (224, 224) for t in m.tiles)   # no padded tile
    for d, c in zip(m.channel_diagnostics, m.spec.channels, strict=True):
        assert d.above_clamp_pixels == int(((data[c.stack_band - 1] > np.float32(0.15)) & (mask == 1)).sum())
    with mi.ModelInputReader(job.final_dir / mi.MANIFEST_FILE, project.cfg) as reader:
        for record in m.tiles:
            tile = reader.read(record)
            r0, c0, dh, dw = record.row_off, record.col_off, record.data_height, record.data_width
            raw = np.zeros((6, T, T), dtype=np.float32)
            raw[:, :dh, :dw] = data[:, r0:r0 + dh, c0:c0 + dw]
            assert np.array_equal(tile.x, reference_normalize(raw[[4, 5, 2, 3, 0, 1]]))
            assert np.array_equal(tile.valid[:dh, :dw], mask[r0:r0 + dh, c0:c0 + dw]) and tile.valid.sum() == \
                record.valid_pixels


# ---- rejection of unsupported / inconsistent inputs ---------------------------------------------------------


@needs_rasterio
def test_missing_or_changed_stack_is_refused(tmp_path_factory, gpt):
    project = published_terrain_set(tmp_path_factory.mktemp("x"), gpt)
    cid = candidate_id(project)
    with pytest.raises(mi.ModelInputError, match="no published Step 3.5 stack"):
        mi.prepare_model_input(project.cfg, project.manifest.aoi_id, cid)
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    stack_file = result.output_dir / "stack_sigma0.tif"
    stack_file.write_bytes(stack_file.read_bytes() + b"\0")
    with pytest.raises(mi.ModelInputError, match="changed since Step 3.5"):
        mi.prepare_model_input(project.cfg, project.manifest.aoi_id, cid)
    with pytest.raises(mi.ModelInputError, match="belongs to"):
        mi.check_stack_provenance(result.provenance, project.manifest.aoi_id, "other-candidate")


@needs_rasterio
@pytest.mark.parametrize(("update", "message"), [
    ({"layout_version": "bato-s1-stack-v2"}, "not supported"),
    ({"status": "failed", "error": "TEST FIXTURE"}, "validated stack"),
])
def test_unsupported_stack_provenance_is_refused(tmp_path_factory, gpt, update, message):
    project, result = stacked(tmp_path_factory.mktemp("p"), gpt)
    data = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    data.update(update)
    result.provenance_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mi.ModelInputError, match=message):
        mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))


@needs_rasterio
def test_stack_inside_an_evaluation_folder_is_refused(tmp_path_factory, gpt):
    project, result = stacked(tmp_path_factory.mktemp("e"), gpt)
    prov = result.provenance
    reference = project.cfg.settings.paths.reference.relative_to(project.cfg.config_dir.parent).as_posix()
    save_contract(prov.model_copy(update={"output": prov.output.model_copy(update={
        "stack": f"{reference}/stack_sigma0.tif"})}), result.provenance_path)
    with pytest.raises(ProhibitedInputError):
        mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))


@needs_rasterio
@pytest.mark.parametrize(("edit", "message"), [
    (lambda d, m: d.__setitem__((2, 5, 5), np.nan), "NaN/inf"),
    (lambda d, m: m.__setitem__((0, 0), 1), "disagrees"),            # mask 1 inside the POST_VV no-data strip
    (lambda d, m: d.__setitem__((0, 20, 20), -0.5), "negative"),
])
def test_inconsistent_stack_values_are_refused_and_nothing_is_published(tmp_path_factory, gpt, edit, message):
    project, result = stacked(tmp_path_factory.mktemp("i"), gpt)
    data, mask = synthetic(260, 230)
    edit(data, mask)
    write_synthetic_stack(project, result, data, mask)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    with pytest.raises(mi.ModelInputError, match=message):
        mi.run_model_input(job, project.cfg)
    assert not job.final_dir.exists() and not list(job.final_dir.parent.glob("*.tmp-*"))


@needs_rasterio
def test_stack_header_mismatch_is_refused(tmp_path_factory, gpt):
    project, result = stacked(tmp_path_factory.mktemp("h"), gpt)
    prov = result.provenance
    save_contract(prov.model_copy(update={"grid": prov.grid.model_copy(update={"width": prov.grid.width + 1})}),
                  result.provenance_path)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    with pytest.raises(mi.ModelInputError, match="provenance says"):
        mi.run_model_input(job, project.cfg)


@needs_rasterio
def test_existing_output_is_kept_unless_overwrite(tmp_path_factory, gpt):
    project, _result = stacked(tmp_path_factory.mktemp("o"), gpt)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    first = mi.run_model_input(job, project.cfg, now=lambda: NOW)
    with pytest.raises(mi.ModelInputError, match="already exists"):
        mi.run_model_input(job, project.cfg)
    second = mi.run_model_input(job, project.cfg, stride=112, overwrite=True,
                                now=lambda: datetime(2026, 10, 3, 13, 0, tzinfo=timezone.utc))
    kept = list(first.output_dir.parent.glob(f"{first.output_dir.name}.superseded-*"))
    assert len(kept) == 1 and load_contract(ModelInputManifest, kept[0] / mi.MANIFEST_FILE).spec.tiling.stride == 224
    assert second.manifest.spec.tiling.stride == 112


@needs_rasterio
def test_reader_refuses_changed_spec_stack_or_unknown_tiles(tmp_path_factory, gpt):
    project, result = stacked(tmp_path_factory.mktemp("q"), gpt)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    out = mi.run_model_input(job, project.cfg)
    with mi.ModelInputReader(out.manifest_path, project.cfg) as reader:
        with pytest.raises(mi.ModelInputError, match="not in this manifest"):
            reader.read("r00224_c00000")
        with pytest.raises(mi.ModelInputError, match="not in this manifest"):
            reader.read(out.manifest.tiles[0].model_copy(update={"valid_pixels": 1}))
    data = json.loads(out.manifest_path.read_text(encoding="utf-8"))
    data["spec"]["normalization"]["clamp_max"] = 0.2                       # hand-edited spec
    out.manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mi.ModelInputError, match="not a valid ModelInputManifest"):
        mi.ModelInputReader(out.manifest_path, project.cfg)
    out = mi.run_model_input(job, project.cfg, overwrite=True)
    data = json.loads(out.manifest_path.read_text(encoding="utf-8"))
    data["spec"]["items"] = data["spec"]["items"][:-1]                     # consistent but different spec
    data["spec_fingerprint"] = spec_fingerprint(ModelInputSpec.model_validate(data["spec"]))
    out.manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mi.ModelInputError, match="different model-input specification"):
        mi.ModelInputReader(out.manifest_path, project.cfg)
    out = mi.run_model_input(job, project.cfg, overwrite=True)
    mask_file = result.output_dir / "valid_mask.tif"
    mask_file.write_bytes(mask_file.read_bytes() + b"\0")
    with pytest.raises(mi.ModelInputError, match="missing or changed"):
        mi.ModelInputReader(out.manifest_path, project.cfg)


@needs_rasterio
def test_manifest_contract_rejects_inconsistent_records(tmp_path_factory, gpt):
    project, _result = stacked(tmp_path_factory.mktemp("c"), gpt)
    job = mi.prepare_model_input(project.cfg, project.manifest.aoi_id, candidate_id(project))
    m = mi.run_model_input(job, project.cfg).manifest
    data = m.model_dump(mode="json")
    with pytest.raises(ValidationError, match="spec_fingerprint"):
        ModelInputManifest.model_validate({**data, "spec_fingerprint": "0" * 64})
    with pytest.raises(ValidationError, match="valid pixels"):
        ModelInputManifest.model_validate({**data, "valid_pixels": m.valid_pixels + 1})
    passes = data["source"]["passes"]
    with pytest.raises(ValidationError, match="time order"):
        ModelInputManifest.model_validate({**data, "source": {**data["source"], "passes": [
            {**passes[0], "acquisition_start": passes[2]["acquisition_start"]}, passes[1], passes[2]]}})
    with pytest.raises(ValidationError, match="channel order"):
        ModelInputManifest.model_validate({**data, "channel_diagnostics": data["channel_diagnostics"][::-1]})
    with pytest.raises(ValidationError, match="tile_id"):
        ModelInputManifest.model_validate({**data, "tiles": [{**data["tiles"][0], "tile_id": "r1_c1"}]})


def test_stack_provenance_type_is_the_step_3_5_contract():
    assert mi.STACK_BANDS == ("PRE1_VV", "PRE1_VH", "PRE2_VV", "PRE2_VH", "POST_VV", "POST_VH")
    assert StackProvenance.model_fields["layout_version"].default == "bato-s1-stack-v1"
    assert PROVENANCE_FILE == "preprocessing_provenance.json"

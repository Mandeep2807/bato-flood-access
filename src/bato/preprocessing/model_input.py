"""Step 4A: Kuro Siwo-compatible model inputs from a published Step 3.5 temporal stack.

Nothing here trains, predicts, labels or downloads anything, and no upstream output is modified.

* :func:`kuro_siwo_spec` - the locked specification (:mod:`bato.contracts.model_input`), every element
  tagged with the evidence it rests on (Kuro Siwo code / data shipped in the Kuro Siwo repository /
  documentation only / not determinable / BATO design choice / known deviation / BATO gap).
* :func:`to_model_order` / :func:`from_kuro_siwo_arrays` - one channel order for BATO stacks and for
  Kuro Siwo samples (POST, PRE2 = Kuro sec1 = later pre-event, PRE1 = Kuro sec2 = earlier pre-event). This is
  also the order the Kuro Siwo torch.hub FloodViT wrapper (commit 43ec8cf) produces from its [pre2, pre1, post]
  input: its "pre1" is Kuro sec1 = BATO PRE2, so its [post, pre1, pre2] is POST, PRE2, PRE1.
* :func:`normalize` - Kuro Siwo ``Dataset.concat`` + ``scale_img("normalize")``: clamp [0, 0.15],
  NaN -> 0.15, ``(x - mean) / std`` in float32.
* :func:`plan_tiles` - 224 x 224 tiles from stack pixel (0, 0), ``stride`` (default 224: no overlap),
  edge tiles padded with the SAR no-data value 0.0 and marked invalid.
* :func:`check_consistency` / :func:`apply_ignore` - the validity mask travels with every tile; any
  output or label at an invalid pixel becomes the ignore label 3.
* :func:`prepare_model_input` + :func:`run_model_input` - verify the stack (provenance, hashes, header,
  mask/value consistency over every pixel) and publish ``model_input_manifest.json`` (tile index,
  spec, fingerprints, diagnostics) atomically under
  ``<interim>/<output_subdir>/<aoi_id>/model_input/<candidate_id>/``. No tensors are written.
* :class:`ModelInputReader` - re-verifies the manifest, the spec and the stack, then yields tiles.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.contracts.common import Polarization
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.model_input import (
    IGNORE_LABEL,
    KURO_MEAN,
    KURO_SAR_NODATA,
    KURO_STD,
    KURO_TILE_SIZE,
    STACK_LAYOUT,
    ChannelDiagnostics,
    EvidenceStatus,
    ModelChannel,
    ModelInputManifest,
    ModelInputSpec,
    NormalizationSpec,
    SourcePass,
    SourceStackRecord,
    SpecItem,
    StackGridRecord,
    TileRecord,
    TilingSpec,
    ValidityPolicy,
    spec_fingerprint,
)
from bato.contracts.stack import StackProvenance
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.preprocessing.radiometric import PROVENANCE_FILE, PreprocessingError, _relative, _sibling, _utc_now, stage_dir
from bato.preprocessing.sigma0_validation import _open, require_rasterio
from bato.preprocessing.speckle import sha256_file
from bato.preprocessing.stack import STACK_GROUP

MODEL_INPUT_GROUP = "model_input"           # <interim>/<subdir>/<aoi_id>/model_input/<candidate_id>/
MANIFEST_FILE = "model_input_manifest.json"
BLOCK_ROWS = 512
PIXEL_SIZE_M = 10.0
STACK_BANDS = ("PRE1_VV", "PRE1_VH", "PRE2_VV", "PRE2_VH", "POST_VV", "POST_VH")   # layout v1, 1-based order
# (role, polarisation, stack band, Kuro Siwo Dataset.__getitem__ variable (file), temporal meaning)
_CHANNELS = (
    ("POST", "VV", 5, "flood_vv", "MS1_IVV", "post-event"),
    ("POST", "VH", 6, "flood_vh", "MS1_IVH", "post-event"),
    ("PRE2", "VV", 3, "sec1_vv", "SL1_IVV", "later pre-event (closest to the event)"),
    ("PRE2", "VH", 4, "sec1_vh", "SL1_IVH", "later pre-event (closest to the event)"),
    ("PRE1", "VV", 1, "sec2_vv", "SL2_IVV", "earlier pre-event"),
    ("PRE1", "VH", 2, "sec2_vh", "SL2_IVH", "earlier pre-event"),
)
KURO_KEYS = tuple(c[3] for c in _CHANNELS)   # model order, Kuro Siwo names
KURO_INPUT_BY_ROLE = {"POST": "flood", "PRE2": "sec1", "PRE1": "sec2"}

_C, _D = EvidenceStatus.VERIFIED_FROM_CODE, EvidenceStatus.VERIFIED_FROM_DATA
_SPEC_ITEMS: tuple[tuple[str, str, EvidenceStatus, str], ...] = (
    ("channel_order", "model tensor = [POST_VV, POST_VH, PRE2_VV, PRE2_VH, PRE1_VV, PRE1_VH], i.e. Kuro Siwo "
     "flood, pre_event_1 (sec1), pre_event_2 (sec2), VV before VH", _C,
     "training/segmentation_trainer.py L61-L76 + L138-L144 (eval L323-L338 + L392-L398): image = flood, then "
     "cat pre_event, then cat pre_event_2 (set comparison, so the order of 'inputs' does not matter); "
     "dataset/Dataset.py L770-L776, returns L826-L839; concat L148-L169 (np.vstack vv, vh at L158)"),
    ("trainer_scope", "the order holds for channel-concatenating models of training/segmentation_trainer.py; the "
     "change-detection and recurrent trainers order time steps differently", _C,
     "change_detection_trainer.py L117-L136 and recurrent_trainer.py L110-L131 follow configs['inputs'] in list "
     "order (CD requires exactly 2 inputs, L19); segmentation_trainer.py L122-L137 stacks [pre_event_2, "
     "pre_event_1, post] for vivit/convlstm"),
    ("hub_wrapper_order", "the torch.hub FloodViT wrapper (vit_decoder) takes [pre2, pre1, post] and reverses it to "
     "[post, pre1, pre2]; its pre1 is sec1 (SL1, the LATER pre-event) and pre2 is sec2 (SL2), so its model order is "
     "POST, PRE2, PRE1 - identical to channel_order. NOT POST, PRE1, PRE2: BATO's PRE1 is Kuro Siwo's pre2", _C,
     "commit 43ec8cf ('Fixed image order issue'; head of PR #4, not merged into main): hubconf.py L44 '[pre2] pre1 "
     "post', L108-L113 list(reversed(norm_images)) then torch.cat, L136 input_hint, L143 img_order_forward=False; "
     "dataset/Dataset.py@43ec8cf L535-L540 SL1 -> sec1, L560-L561 'pre1_vv' = sec1, L771 pre_event_1 = concat(sec1)"),
    ("hub_wrapper_scaling", "the hub wrapper normalises with the same mean/std but applies NO clamp / NaN fill and "
     "uses other DEM/slope constants; BATO keeps the training loader's clamp [0, 0.15] + NaN -> 0.15", _C,
     "hubconf.py@43ec8cf L16-L21 (IMG_MEANS/STDS as data_config; DEM 67.0293/1765.0062, slope 2.9482/79.2493), "
     "L88 Normalize only; the loader clamps in Dataset.concat (main L164-L166)"),
    ("sec1_is_later_pre_event", "Kuro Siwo sec1 (SL1, pre_event_1) is the LATER pre-event image and sec2 (SL2, "
     "pre_event_2) the EARLIER one: BATO PRE-2 -> sec1, PRE-1 -> sec2", _D,
     "pickle/KuroV2_grid_dict.gz 31,707/31,707 and pickle/KuroV2_grid_dict_test_0_100.gz 67,490/67,490 samples "
     "have sources.SL1.source_date > sources.SL2.source_date (train is a subset of test: 67,490 unique samples); "
     "file roles Dataset.__getitem__ L669-L729, pre_event_1 = concat(sec1_*) at L774"),
    ("flood_is_post_event", "Kuro Siwo flood (MS1) is the post-event image (at day resolution)", _D,
     "MS1.source_date >= flood_date (date part) and SL1/SL2 < flood_date in 100% of both pickles (MS1 0-43 d "
     "after); 7,288 test / 2,232 train samples are same-day, which dates alone cannot order"),
    ("sar_value_domain", "the default loader applies no dB/log transform (values are used in their stored "
     "domain) and the shipped GRD graph writes linear sigma0; BATO therefore feeds linear sigma0", _C,
     "Dataset.__getitem__ L669-L729 (cv.imread IMREAD_ANYDEPTH; uint8 false: data_config.json L33); torch.log only "
     "in scale_input 'custom' (L446); grd_preprocessing.xml outputSigmaBand true (L91), outputImageScaleInDb "
     "false (L87). That the released files came from this graph is the open item released_pixel_units"),
    ("released_pixel_units", "units/value range of the released Kuro Siwo GeoTIFF/.npy pixels",
     EvidenceStatus.NOT_DETERMINABLE,
     "pixel data not inspected (not downloaded); only dtype float32 and no-data 0.0 verified from pickle metadata"),
    ("clamp", "clamp every SAR channel to [0.0, 0.15]", _C,
     "Dataset.concat L164-L165 torch.clamp(image, min=0.0, max=clamp_input); data_config.json L14 clamp_input 0.15"),
    ("nan_fill", "NaN -> 0.15 after the clamp", _C, "Dataset.concat L166 torch.nan_to_num(image, clamp_input)"),
    ("zscore", "per channel (x - mean) / std, mean VV 0.0953 VH 0.0264, std VV 0.0427 VH 0.0215, fixed constants, "
     "identical for all three time steps", _C,
     "Dataset.scale_img L193-L198 torchvision Normalize; data_config.json L15-L17; __getitem__ L811-L822"),
    ("zscore_derivation", "how the Kuro Siwo mean/std constants were computed (before/after clamp, which samples)",
     EvidenceStatus.NOT_DETERMINABLE, "hard-coded constants; only a comment in SSLDataset L968-L970"),
    ("dem_channel", "no DEM and no slope channel", _C,
     "data_config.json sets dem/slope true (L24-L25) then false (L30-L31); pyjson5 keeps the last key; "
     "main.py L66-L67 ignores --dem without --inputs"),
    ("dem_source", "source, units, resolution of Kuro Siwo MK0_DEM (only relevant if a DEM channel is ever used)",
     EvidenceStatus.NOT_DETERMINABLE, "no repository code produces MK0_DEM; grd_preprocessing.xml saveDEM false"),
    ("tile_size", "224 x 224 pixels at 10 m, EPSG:3857", _D,
     "all 67,490 unique pickle footprints are 2240 m x 2240 m with corners at 5 m mod 10 m; CRS from "
     "catalogue/catalogue.yaml L14 (CRS: 3857) and grd_preprocessing.xml L151; Dataset L780 np.zeros((224, 224)); "
     "grd_preprocessing.xml pixelSpacingInMeter 10.0 (array shape of the released files not inspected)"),
    ("no_resampling", "no resize or crop at inference (augmentations are off by default and train-only)", _C,
     "data_config.json L13 data_augmentations false; Dataset L44-L50 (augmentations built only then); "
     "Dataset.__getitem__ L792-L806 (mode == 'train' only)"),
    ("sar_nodata", "SAR no-data is 0.0 and reaches the model unmasked (normalised like any value)", _C,
     "catalogue/catalogue.py L62 _NODATA IVV/IVH 0.0; scale_img 'normalize' ignores valid_mask"),
    ("label_classes", "labels have 3 classes and the ignore value 3", _C,
     "configs/config.json L14 num_classes 3; utilities.create_loss L307-L347 ignore_index=3; catalogue.py L62 "
     "MLU no-data 3"),
    ("label_names", "0 no water, 1 permanent water, 2 flood, 3 invalid pixels (ignored)", _C,
     "segmentation_trainer.py L13 CLASS_LABELS {0: 'No water', 1: 'Permanent Waters', 2: 'Floods', 3: 'Invalid "
     "pixels'} (same in change_detection_trainer.py L15, recurrent_trainer.py L16); also the Hugging Face card"),
    ("debris_labels", "Kuro Siwo has NO debris/damage class: a Kuro Siwo-trained model maps water only", _C,
     "num_classes 3 and CLASS_LABELS; no debris/damage class or label in code, configs or pickles (the only "
     "case-insensitive 'damage' match is the MIT LICENSE text, L18)"),
    ("tiling_policy", "stride (default 224), origin at stack pixel (0, 0); edge tiles SHIFTED to end at the stack "
     "edge (no padding) whenever the axis is >= 224, so edge tiles overlap their neighbour; padding with 0.0 "
     "(marked invalid) only for an axis shorter than 224; overlapping outputs are averaged by the consumer",
     EvidenceStatus.BATO_DESIGN_CHOICE,
     "Kuro Siwo has no tiling, sliding-window or stitching code; ViT attention is global (vision_transformer.py "
     "L84-L89, every token attends to every token), so padding inside a tile changes the outputs of its real pixels"),
    ("invalid_pixel_policy", "values at invalid pixels pass through unchanged (Kuro Siwo convention); the validity "
     "mask is mandatory and every output/label at mask 0 becomes ignore (3)", EvidenceStatus.BATO_DESIGN_CHOICE,
     "Kuro Siwo feeds no-data 0.0 unmasked and ignores label 3 in losses/metrics"),
    ("radar_visibility", "no layover/shadow observability layer: the validity mask encodes SAR no-data only",
     EvidenceStatus.BATO_GAP,
     "Step 3.4 saves no layover/shadow mask (Kuro Siwo graph saveLayoverShadowMask false); bato.visibility "
     "not implemented"),
    ("pretrained_checkpoints", "input configuration of the published Kuro Siwo checkpoints (FloodViT, SNUNet)",
     EvidenceStatus.NOT_DETERMINABLE, "README links weights only; their training configs are not in the repository "
     "(the 43ec8cf hub wrapper documents the input ORDER only, see hub_wrapper_order)"),
    ("terrain_correction_dem", "BATO terrain-corrects with Copernicus GLO-30, Kuro Siwo with SRTM 1Sec HGT",
     EvidenceStatus.KNOWN_DEVIATION, "decided at Step 3.4 (docs/preprocessing.md); grd_preprocessing.xml demName"),
    ("land_sea_mask", "BATO omits Land-Sea-Mask (Kuro Siwo sets sea pixels to no-data)",
     EvidenceStatus.KNOWN_DEVIATION, "decided at Step 3.2 (docs/preprocessing.md); grd_preprocessing.xml "
     "Land-Sea-Mask landMask false, useSRTM true"),
)


class ModelInputError(PreprocessingError):
    """The stack cannot be turned into model inputs (nothing written)."""


# ---------------------------------------------------------------------------
# Specification, channel order, normalisation
# ---------------------------------------------------------------------------


def kuro_siwo_spec(stride: int = KURO_TILE_SIZE, edge_policy: str = "shift_to_fit") -> ModelInputSpec:
    """The locked v1 specification (only the stride and the edge policy are parameters)."""
    channels = tuple(ModelChannel(
        index=k, name=f"{role}_{pol}", stack_band=band, role=role,  # type: ignore[arg-type]
        polarization=Polarization(pol), kuro_siwo_input=f"{key} ({file})", temporal_meaning=meaning,
        mean=KURO_MEAN[pol], std=KURO_STD[pol]) for k, (role, pol, band, key, file, meaning) in enumerate(_CHANNELS))
    tiling = TilingSpec.model_validate({"stride": stride, "edge_policy": edge_policy})
    return ModelInputSpec(channels=channels, normalization=NormalizationSpec(), tiling=tiling,
                          validity=ValidityPolicy(),
                          items=tuple(SpecItem(key=k, statement=s, status=st, evidence=e)
                                      for k, s, st, e in _SPEC_ITEMS))


def to_model_order(stack_bands: np.ndarray, spec: ModelInputSpec) -> np.ndarray:
    """(6, H, W) in stack layout v1 (PRE1_VV ... POST_VH) -> (6, H, W) in model channel order."""
    if stack_bands.ndim != 3 or stack_bands.shape[0] != len(STACK_BANDS):
        raise ModelInputError(f"expected a (6, H, W) layout-v1 array, got shape {stack_bands.shape}")
    return stack_bands[[c.stack_band - 1 for c in spec.channels]]


def from_kuro_siwo_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Kuro Siwo sample arrays (flood_vv, flood_vh, sec1_vv, sec1_vh, sec2_vv, sec2_vh) -> (6, H, W), model order."""
    missing, extra = set(KURO_KEYS) - set(arrays), set(arrays) - set(KURO_KEYS)
    if missing or extra:
        raise ModelInputError(f"Kuro Siwo arrays must be exactly {list(KURO_KEYS)} (missing {sorted(missing)}, "
                              f"unexpected {sorted(extra)})")
    shapes = {np.shape(arrays[k]) for k in KURO_KEYS}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ModelInputError(f"Kuro Siwo arrays must be 2-D and the same shape, got {sorted(shapes)}")
    return np.stack([np.asarray(arrays[k], dtype=np.float32) for k in KURO_KEYS])


def normalize(x: np.ndarray, spec: ModelInputSpec) -> np.ndarray:
    """Kuro Siwo default input scaling of a (6, H, W) model-order array, in float32.

    Same operations and order as ``Dataset.concat`` (clamp, NaN fill) + ``scale_img("normalize")``
    (torchvision Normalize: ``(x - mean) / std`` with float32 mean/std).
    """
    if x.ndim != 3 or x.shape[0] != len(spec.channels):
        raise ModelInputError(f"expected a ({len(spec.channels)}, H, W) array, got shape {x.shape}")
    n = spec.normalization
    out = np.clip(np.asarray(x, dtype=np.float32), np.float32(n.clamp_min), np.float32(n.clamp_max))
    out = np.where(np.isnan(out), np.float32(n.nan_fill), out)
    mean = np.array([c.mean for c in spec.channels], dtype=np.float32)[:, None, None]
    std = np.array([c.std for c in spec.channels], dtype=np.float32)[:, None, None]
    return ((out - mean) / std).astype(np.float32, copy=False)


def apply_ignore(values: np.ndarray, valid: np.ndarray, ignore: int = IGNORE_LABEL) -> np.ndarray:
    """Copy of an integer label/prediction array with ``ignore`` wherever ``valid`` is 0."""
    if values.shape != valid.shape:
        raise ModelInputError(f"values {values.shape} and validity mask {valid.shape} differ in shape")
    if not np.issubdtype(values.dtype, np.integer):
        raise ModelInputError(f"labels/predictions must be integer class codes, got {values.dtype}")
    _check_mask_values(valid)
    out = values.copy()
    out[valid == 0] = ignore
    return out


# ---------------------------------------------------------------------------
# Tiling and per-tile checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileWindow:
    row_off: int
    col_off: int
    data_height: int
    data_width: int


def tile_offsets(length: int, tile: int, stride: int, edge_policy: str = "pad_end") -> list[int]:
    """Offsets along one axis, from 0 by ``stride``.

    ``pad_end``: a tile is added only while the previous one ends inside the axis; the last tile may run past
    the end (padded). ``shift_to_fit``: the same, but the last tile starts at ``length - tile`` so it ends
    exactly at the edge (no padding) - only an axis shorter than ``tile`` keeps one padded tile at 0.
    """
    if length < 1:
        raise ModelInputError(f"cannot tile an axis of length {length}")
    if not 1 <= stride <= tile:
        raise ModelInputError(f"stride {stride} must be in 1..{tile}")
    if edge_policy not in ("pad_end", "shift_to_fit"):
        raise ModelInputError(f"unknown edge policy {edge_policy!r}")
    offsets = [0]
    while offsets[-1] + tile < length:
        offsets.append(offsets[-1] + stride)
    if edge_policy == "shift_to_fit" and length >= tile and offsets[-1] + tile > length:
        offsets[-1] = length - tile                  # pull the overhanging last tile back inside the axis
    return offsets                                   # gaps stay <= stride: the shifted tile only moves closer


def plan_tiles(height: int, width: int, tiling: TilingSpec) -> list[TileWindow]:
    """Row-major tiles covering every stack pixel; ``data_*`` is the part inside the stack."""
    t, s, e = tiling.tile_size, tiling.stride, tiling.edge_policy
    return [TileWindow(r, c, min(t, height - r), min(t, width - c))
            for r in tile_offsets(height, t, s, e) for c in tile_offsets(width, t, s, e)]


def _check_mask_values(valid: np.ndarray) -> None:
    bad = ~np.isin(valid, (0, 1))
    if bad.any():
        raise ModelInputError(f"validity mask holds values other than 0/1 ({int(bad.sum())} pixels)")


def check_consistency(raw: np.ndarray, valid: np.ndarray, where: str = "") -> None:
    """Fail closed unless a (6, H, W) layout-v1 block and its mask agree with the Step 3.5 contract:
    values finite and >= 0, mask 0/1, and mask 1 exactly where all six layers are != 0.0."""
    at = f" {where}" if where else ""
    if raw.ndim != 3 or raw.shape[0] != len(STACK_BANDS) or raw.shape[1:] != valid.shape:
        raise ModelInputError(f"block{at}: shape {raw.shape} does not match mask {valid.shape}")
    _check_mask_values(valid)
    finite = np.isfinite(raw)
    if not finite.all():
        raise ModelInputError(f"block{at}: {int((~finite).sum())} NaN/inf values (the Step 3.5 stack must be finite; "
                              "Kuro Siwo's NaN -> 0.15 rule is not used to hide them)")
    if (raw < 0).any():
        raise ModelInputError(f"block{at}: {int((raw < 0).sum())} negative values (linear sigma0 cannot be negative)")
    expected = np.all(raw != KURO_SAR_NODATA, axis=0)
    mismatch = expected != (valid == 1)
    if mismatch.any():
        raise ModelInputError(f"block{at}: validity mask disagrees with the data at {int(mismatch.sum())} pixels "
                              "(mask must be 1 exactly where all six layers are != 0.0)")


@dataclass(frozen=True)
class ModelInputTile:
    record: TileRecord
    x: np.ndarray                 # (6, 224, 224) float32, model order, normalised
    valid: np.ndarray             # (224, 224) uint8, 1 = valid in all six layers, 0 = invalid or padding


def build_tile(raw: np.ndarray, valid: np.ndarray, spec: ModelInputSpec) -> tuple[np.ndarray, np.ndarray]:
    """Padded layout-v1 block + mask -> (normalised model tensor, uint8 mask), after the consistency check."""
    t = spec.tiling.tile_size
    if raw.shape != (len(STACK_BANDS), t, t):
        raise ModelInputError(f"a tile block must be (6, {t}, {t}), got {raw.shape}")
    check_consistency(raw, valid)
    return normalize(to_model_order(raw, spec), spec), valid.astype(np.uint8, copy=True)


def tile_transform(grid_transform: Sequence[float], window: TileWindow
                   ) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, e, f = (float(v) for v in grid_transform)
    return (a, b, c + window.col_off * a, d, e, f + window.row_off * e)


# ---------------------------------------------------------------------------
# Preparing a manifest from a published stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInputJob:
    root: Path
    aoi_id: str
    candidate_id: str
    stack_dir: Path
    provenance_path: Path
    provenance: StackProvenance
    provenance_sha256: str
    stack_path: Path
    mask_path: Path
    final_dir: Path


def check_stack_provenance(prov: StackProvenance, aoi_id: str, candidate_id: str) -> None:
    """Fail closed unless the provenance describes a validated layout-v1 stack of this AOI/candidate."""
    if prov.status != "succeeded" or prov.validation is None or not prov.validation.passed:
        raise ModelInputError("the Step 3.5 provenance does not describe a validated stack")
    if prov.output is None or prov.grid is None:
        raise ModelInputError("the Step 3.5 provenance has no output/grid")
    if (prov.aoi_id, prov.candidate_id) != (aoi_id, candidate_id):
        raise ModelInputError(f"the stack belongs to {prov.aoi_id}/{prov.candidate_id}, not {aoi_id}/{candidate_id}")
    if prov.layout_version != STACK_LAYOUT:
        raise ModelInputError(f"stack layout {prov.layout_version} is not supported (only {STACK_LAYOUT})")
    if tuple(x.name for x in prov.layers) != STACK_BANDS or [x.index for x in prov.layers] != [1, 2, 3, 4, 5, 6]:
        raise ModelInputError(f"stack layers {[x.name for x in prov.layers]} are not layout v1 {list(STACK_BANDS)}")
    starts = [i.acquisition_start for i in prov.inputs]
    if not starts[0] < starts[1] < starts[2]:
        raise ModelInputError("stack inputs are not in time order PRE1 < PRE2 < POST")
    if not (prov.inputs[1].days_from_event < 0 < prov.inputs[2].days_from_event):
        raise ModelInputError("PRE2 must be before and POST after the event date")
    if len({(i.orbit_direction, i.relative_orbit) for i in prov.inputs}) != 1:
        raise ModelInputError("stack inputs are not all from one track (orbit direction + relative orbit)")
    if prov.grid.crs != "EPSG:3857" or prov.grid.pixel_size_m != PIXEL_SIZE_M:
        raise ModelInputError(f"stack grid {prov.grid.crs} {prov.grid.pixel_size_m} m is not EPSG:3857 10 m")
    if prov.output.dtype != "float32" or prov.output.nodata != KURO_SAR_NODATA:
        raise ModelInputError(f"stack dtype/no-data {prov.output.dtype}/{prov.output.nodata} is not float32/0.0")


def prepare_model_input(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> ModelInputJob:
    """Locate and verify the published stack (provenance, file hashes); nothing is written.

    Raises:
        ModelInputError: missing or inconsistent stack (fail closed).
        ProhibitedInputError: an input lies in an evaluation-only folder.
    """
    root = cfg.config_dir.parent
    stack_dir = stage_dir(cfg, aoi_id, STACK_GROUP, candidate_id)
    provenance_path = assert_pipeline_input_allowed(stack_dir / PROVENANCE_FILE, cfg.settings.paths)
    if not provenance_path.is_file():
        raise ModelInputError(f"no published Step 3.5 stack: {_relative(stack_dir, root)}/{PROVENANCE_FILE} not found "
                              "(run scripts/stack_s1.py first)")
    try:
        prov = load_contract(StackProvenance, provenance_path)
    except ContractIOError as exc:
        raise ModelInputError(str(exc)) from exc
    check_stack_provenance(prov, aoi_id, candidate_id)
    assert prov.output is not None
    stack_path = assert_pipeline_input_allowed(root / prov.output.stack, cfg.settings.paths)
    mask_path = assert_pipeline_input_allowed(root / prov.output.mask, cfg.settings.paths)
    for path, expected in ((stack_path, prov.output.stack_sha256), (mask_path, prov.output.mask_sha256)):
        if not path.is_file():
            raise ModelInputError(f"{_relative(path, root)} is missing")
        if sha256_file(path) != expected:
            raise ModelInputError(f"{_relative(path, root)} changed since Step 3.5 "
                                  "(SHA-256 differs from its provenance)")
    return ModelInputJob(root=root, aoi_id=aoi_id, candidate_id=candidate_id, stack_dir=stack_dir,
                         provenance_path=provenance_path, provenance=prov,
                         provenance_sha256=sha256_file(provenance_path),
                         stack_path=stack_path, mask_path=mask_path,
                         final_dir=stage_dir(cfg, aoi_id, MODEL_INPUT_GROUP, candidate_id))


def source_stack_record(job: ModelInputJob) -> SourceStackRecord:
    """The verified source stack (paths relative to the project root, hashes, passes with Kuro Siwo roles)."""
    prov = job.provenance
    assert prov.output is not None
    passes = tuple(SourcePass(role=i.role, pass_id=i.pass_id, acquisition_start=i.acquisition_start,
                              days_from_event=i.days_from_event, kuro_siwo_input=KURO_INPUT_BY_ROLE[i.role])
                   for i in prov.inputs)
    return SourceStackRecord(
        directory=_relative(job.stack_dir, job.root), provenance=_relative(job.provenance_path, job.root),
        provenance_sha256=job.provenance_sha256, stack=prov.output.stack, stack_sha256=prov.output.stack_sha256,
        mask=prov.output.mask, mask_sha256=prov.output.mask_sha256, layout_version=prov.layout_version,
        stack_config_fingerprint=prov.config_fingerprint, track=prov.track, passes=passes)


def check_rasters(stack_ds: Any, mask_ds: Any, prov: StackProvenance) -> None:
    """Header checks of the opened stack and mask against the Step 3.5 provenance."""
    assert prov.grid is not None
    g = prov.grid
    epsg = stack_ds.crs.to_epsg() if stack_ds.crs is not None else None
    if stack_ds.count != len(STACK_BANDS) or any(str(d) != "float32" for d in stack_ds.dtypes):
        raise ModelInputError(f"stack has {stack_ds.count} bands of {set(stack_ds.dtypes)}; expected 6 x float32")
    if tuple(stack_ds.descriptions) != STACK_BANDS:
        raise ModelInputError(f"stack band names {tuple(stack_ds.descriptions)} are not layout v1 {STACK_BANDS}")
    if epsg != 3857 or (stack_ds.width, stack_ds.height) != (g.width, g.height):
        raise ModelInputError(f"stack is EPSG:{epsg} {stack_ds.width} x {stack_ds.height}; provenance says "
                              f"EPSG:3857 {g.width} x {g.height}")
    t = tuple(float(v) for v in tuple(stack_ds.transform)[:6])
    if any(abs(x - y) > 1e-6 for x, y in zip(t, g.transform, strict=True)):
        raise ModelInputError(f"stack transform {t} differs from the provenance grid {g.transform}")
    if stack_ds.nodata != KURO_SAR_NODATA:
        raise ModelInputError(f"stack no-data {stack_ds.nodata} is not 0.0")
    mt = tuple(float(v) for v in tuple(mask_ds.transform)[:6])
    if (mask_ds.count != 1 or str(mask_ds.dtypes[0]) != "uint8"
            or (mask_ds.width, mask_ds.height) != (g.width, g.height)
            or any(abs(x - y) > 1e-6 for x, y in zip(mt, g.transform, strict=True))):
        raise ModelInputError("validity mask is not one uint8 band on the stack grid")


def read_block(stack_ds: Any, mask_ds: Any, window: TileWindow, size: tuple[int, int] | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Read a (6, h, w) layout-v1 block and its mask; with ``size`` the block is padded (0.0 / mask 0)."""
    from rasterio.windows import Window

    h, w = size if size is not None else (window.data_height, window.data_width)
    raw = np.full((len(STACK_BANDS), h, w), KURO_SAR_NODATA, dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.uint8)
    win = Window(window.col_off, window.row_off, window.data_width, window.data_height)
    for k in range(len(STACK_BANDS)):
        raw[k, :window.data_height, :window.data_width] = np.asarray(stack_ds.read(k + 1, window=win), dtype=np.float32)
    valid[:window.data_height, :window.data_width] = np.asarray(mask_ds.read(1, window=win), dtype=np.uint8)
    return raw, valid


@dataclass(frozen=True)
class ModelInputResult:
    output_dir: Path
    manifest_path: Path
    manifest: ModelInputManifest


def build_manifest(job: ModelInputJob, cfg: BatoConfig, *, stride: int = KURO_TILE_SIZE,
                   edge_policy: str = "shift_to_fit", now: Callable[[], datetime] = _utc_now,
                   progress: Callable[[str], None] | None = None) -> ModelInputManifest:
    """Check every stack pixel against the contract and plan the tiles (read-only; nothing written).

    Raises:
        ModelInputError: inconsistent input.
        ValidationSetupError: rasterio missing.
    """
    require_rasterio()
    say = progress or (lambda _text: None)
    spec = kuro_siwo_spec(stride, edge_policy)
    prov = job.provenance
    assert prov.grid is not None and prov.output is not None
    w, h = prov.grid.width, prov.grid.height
    clamp_max = np.float32(spec.normalization.clamp_max)
    above = np.zeros(len(STACK_BANDS), dtype=np.int64)
    mask = np.zeros((h, w), dtype=np.uint8)
    with _open(job.stack_path) as stack_ds, _open(job.mask_path) as mask_ds:
        check_rasters(stack_ds, mask_ds, prov)
        say(f"checking {w} x {h} x 6 stack pixels against the model-input contract ...\n")
        for row in range(0, h, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, h - row)
            raw, valid = read_block(stack_ds, mask_ds, TileWindow(row, 0, rows, w))
            check_consistency(raw, valid, where=f"rows {row}-{row + rows - 1}")
            above += ((raw > clamp_max) & (valid == 1)[None]).sum(axis=(1, 2))
            mask[row:row + rows] = valid
    valid_total = int(mask.sum())
    tiles = tuple(TileRecord(
        tile_id=f"r{t.row_off:05d}_c{t.col_off:05d}", row_off=t.row_off, col_off=t.col_off, data_height=t.data_height,
        data_width=t.data_width, transform=tile_transform(prov.grid.transform, t),
        valid_pixels=int(mask[t.row_off:t.row_off + t.data_height, t.col_off:t.col_off + t.data_width].sum()))
        for t in plan_tiles(h, w, spec.tiling))
    by_band = dict(zip(STACK_BANDS, above.tolist(), strict=True))
    diagnostics = tuple(ChannelDiagnostics(
        name=c.name, valid_pixels=valid_total, above_clamp_pixels=int(by_band[c.name]),
        above_clamp_fraction=round(by_band[c.name] / valid_total, 6) if valid_total else 0.0) for c in spec.channels)
    for path, expected in ((job.stack_path, prov.output.stack_sha256), (job.mask_path, prov.output.mask_sha256)):
        if sha256_file(path) != expected:
            raise ModelInputError(f"{path.name} changed while it was being read")
    warnings = []
    if valid_total < w * h:
        warnings.append(f"{w * h - valid_total:,} of {w * h:,} stack pixels are invalid (mask 0); their model outputs "
                        "must be set to ignore (3)")
    if valid_total == 0:
        raise ModelInputError("the stack has no pixel valid in all six layers")
    warnings += [f"{d.name}: {d.above_clamp_fraction:.2%} of valid pixels exceed the Kuro Siwo clamp "
                 f"{spec.normalization.clamp_max} (saturated)" for d in diagnostics if d.above_clamp_fraction > 0]
    manifest = ModelInputManifest(
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(), created_at=now(), aoi_id=job.aoi_id,
        candidate_id=job.candidate_id, event_date=prov.event_date, source=source_stack_record(job),
        grid=StackGridRecord(pixel_size_m=prov.grid.pixel_size_m, width=w, height=h, transform=prov.grid.transform),
        spec=spec, spec_fingerprint=spec_fingerprint(spec), tiles=tiles, channel_diagnostics=diagnostics,
        valid_pixels=valid_total, warnings=tuple(warnings))
    return manifest


def run_model_input(job: ModelInputJob, cfg: BatoConfig, *, stride: int = KURO_TILE_SIZE,
                    edge_policy: str = "shift_to_fit", overwrite: bool = False,
                    now: Callable[[], datetime] = _utc_now, progress: Callable[[str], None] | None = None
                    ) -> ModelInputResult:
    """:func:`build_manifest`, then publish ``model_input_manifest.json`` atomically (tmp -> final; an existing
    output is kept as ``<candidate>.superseded-<time>`` with ``overwrite``).

    Raises:
        ModelInputError: refused or inconsistent input (nothing published).
        ValidationSetupError: rasterio missing.
    """
    final_dir = assert_interim_output_dir(job.final_dir, cfg.settings.paths)
    if final_dir.exists() and not overwrite:
        raise ModelInputError(f"output already exists: {final_dir} (use --overwrite to supersede it; the old output "
                              "is kept as <candidate>.superseded-<time>)")
    manifest = build_manifest(job, cfg, stride=stride, edge_policy=edge_policy, now=now, progress=progress)
    stamp = now()
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", stamp), cfg.settings.paths)
    tmp_dir.mkdir(parents=True)
    save_contract(manifest, tmp_dir / MANIFEST_FILE)
    if final_dir.exists():
        os.replace(final_dir, _sibling(final_dir, "superseded", stamp))
    os.replace(tmp_dir, final_dir)
    return ModelInputResult(output_dir=final_dir, manifest_path=final_dir / MANIFEST_FILE, manifest=manifest)


# ---------------------------------------------------------------------------
# Reading tiles for training/inference
# ---------------------------------------------------------------------------


class ModelInputReader:
    """Open a published manifest, re-verify spec + stack + mask, and read tiles as model tensors."""

    def __init__(self, manifest_path: Path, cfg: BatoConfig, *, verify_hashes: bool = True) -> None:
        require_rasterio()
        self.root = cfg.config_dir.parent
        path = assert_pipeline_input_allowed(manifest_path, cfg.settings.paths)
        try:
            self.manifest = load_contract(ModelInputManifest, path)
        except ContractIOError as exc:
            raise ModelInputError(str(exc)) from exc
        self.spec = kuro_siwo_spec(self.manifest.spec.tiling.stride, self.manifest.spec.tiling.edge_policy)
        if self.manifest.spec != self.spec or self.manifest.spec_fingerprint != spec_fingerprint(self.spec):
            raise ModelInputError("the manifest was made with a different model-input specification than this code "
                                  f"({self.manifest.spec.version}, fingerprint {self.manifest.spec_fingerprint[:12]})")
        src = self.manifest.source
        self.stack_path = assert_pipeline_input_allowed(self.root / src.stack, cfg.settings.paths)
        self.mask_path = assert_pipeline_input_allowed(self.root / src.mask, cfg.settings.paths)
        if verify_hashes:
            for p, expected in ((self.stack_path, src.stack_sha256), (self.mask_path, src.mask_sha256)):
                if not p.is_file() or sha256_file(p) != expected:
                    raise ModelInputError(f"{src.stack if p == self.stack_path else src.mask} is missing or changed "
                                          "since the manifest was written")
        self._tiles = {t.tile_id: t for t in self.manifest.tiles}
        self._stack = _open(self.stack_path)
        self._mask = _open(self.mask_path)

    def __enter__(self) -> ModelInputReader:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        self._stack.close()
        self._mask.close()

    def read(self, tile: TileRecord | str) -> ModelInputTile:
        record = self._tiles.get(tile if isinstance(tile, str) else tile.tile_id)
        if record is None or (not isinstance(tile, str) and tile != record):
            raise ModelInputError(f"tile {tile if isinstance(tile, str) else tile.tile_id} is not in this manifest")
        t = self.spec.tiling.tile_size
        window = TileWindow(record.row_off, record.col_off, record.data_height, record.data_width)
        raw, valid = read_block(self._stack, self._mask, window, size=(t, t))
        x, valid = build_tile(raw, valid, self.spec)
        if int(valid.sum()) != record.valid_pixels:
            raise ModelInputError(f"tile {record.tile_id}: {int(valid.sum())} valid pixels, manifest says "
                                  f"{record.valid_pixels}")
        return ModelInputTile(record=record, x=x, valid=valid)

    def __iter__(self) -> Iterator[ModelInputTile]:
        for record in self.manifest.tiles:
            yield self.read(record)


def describe_model_input(manifest: ModelInputManifest) -> list[str]:
    s = manifest.spec
    lines = [f"spec       : {s.version} (fingerprint {manifest.spec_fingerprint[:12]}), tensor {s.tensor_shape} "
             f"{s.dtype}, DEM {s.dem_features}",
             "channels   : " + ", ".join(f"{c.index} {c.name}={c.kuro_siwo_input.split()[0]}" for c in s.channels),
             f"normalise  : clamp [{s.normalization.clamp_min}, {s.normalization.clamp_max}], NaN -> "
             f"{s.normalization.nan_fill}, (x - mean) / std (Kuro Siwo constants)",
             f"tiles      : {len(manifest.tiles)} x {s.tiling.tile_size} px, stride {s.tiling.stride} (overlap "
             f"{s.tiling.overlap}), edge {s.tiling.edge_policy}; {manifest.tiles_with_valid_pixels} with valid pixels",
             f"valid      : {manifest.valid_pixels:,} of {manifest.grid.width * manifest.grid.height:,} pixels"]
    lines += [f"{d.name:8} above clamp: {d.above_clamp_pixels:,} ({d.above_clamp_fraction:.2%})"
              for d in manifest.channel_diagnostics]
    lines += [f"WARN  {x}" for x in manifest.warnings]
    return lines

# Step 4A — model-input contract (Kuro Siwo-compatible)

This document fixes how the Step 3.5 temporal stack becomes model tensors. It covers both training (on Kuro Siwo) and inference (on BATO stacks). Every element is classified by its evidence. Nothing comes from papers, README text or common practice.

- **Spec version:** `bato-model-input-v1`.
- **Code:** `src/bato/contracts/model_input.py` (contracts) and `src/bato/preprocessing/model_input.py` (spec, adapter, reader).
- **Reference implementation:** [Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo) at commit `4347ed173c4e48f5a9d578bb5fe8453706b08e5e` (2026-07-16).
  - Default configuration: `configs/config.json` (task segmentation, method unet, `num_classes` 3) and `configs/train/data_config.json` (GRD, `slc: false`).
  - Trainer: `training/segmentation_trainer.py`.
- **Independent check:** every code and data citation below was re-checked against the clone by a separate reviewer on 2026-10-03. The corrections it found are applied.

Evidence labels:

| Label | Meaning |
|---|---|
| **CODE** | Verified from Kuro Siwo code. |
| **DATA** | Verified from the Kuro Siwo data shipped in that repository: the two grid pickles, 67,490 unique samples. |
| **DOC** | Documentation only. |
| **ND** | Not determinable. |
| **BATO** | BATO design choice. Kuro Siwo has no equivalent. |
| **DEV** | Known, earlier-decided deviation. |
| **GAP** | Information BATO does not have yet. |

## 1. The tensor

| | |
|---|---|
| dtype / layout / shape | `float32`, `(C, H, W)` = `(6, 224, 224)` |
| Channels | `0 POST_VV · 1 POST_VH · 2 PRE2_VV · 3 PRE2_VH · 4 PRE1_VV · 5 PRE1_VH` |
| Stack bands (layout v1) | `5, 6, 3, 4, 1, 2` |
| Kuro Siwo names | `flood_vv, flood_vh, sec1_vv, sec1_vh, sec2_vv, sec2_vh` (files MS1, SL1, SL2) |
| Per-value operation | `clamp(x, 0.0, 0.15)` → NaN → 0.15 → `(x − mean[c]) / std[c]`, in float32 |
| mean / std | VV 0.0953 / 0.0427 · VH 0.0264 / 0.0215 (fixed constants, identical for all three time steps) |
| DEM / slope | **none** |
| Validity | `uint8` `(224, 224)`: 1 = valid in all six stack layers, 0 = invalid or padding |

**The storage order is not the model order.** Our stack is stored as PRE1, PRE2, POST. The model expects POST, then the *later* pre-event image (PRE-2 = Kuro Siwo `sec1`), then the *earlier* one (PRE-1 = `sec2`). `to_model_order` and `from_kuro_siwo_arrays` produce the identical tensor from a BATO stack and from a Kuro Siwo sample (tested).

**Name trap.** The Kuro Siwo torch.hub FloodViT wrapper (`hubconf.py` at commit `43ec8cf`, PR #4, not merged into `main`) takes images as `[pre2, pre1, post]` and feeds `torch.cat(list(reversed(images)))` = `[post, pre1, pre2]`. Its `pre1` is Kuro Siwo `sec1` (SL1, the **later** pre-event image) and `pre2` is `sec2`. In BATO names that is **POST, PRE2, PRE1**, the order above. It is **not** POST, PRE1, PRE2: BATO's PRE1 is Kuro Siwo's `pre2`. Tests reproduce the wrapper's reversal on BATO bands and refuse a POST, PRE1, PRE2 spec.

## 2. Evidence for every element

| Key | Statement | Status | Evidence |
|---|---|---|---|
| channel_order | Tensor is flood, then `pre_event_1`, then `pre_event_2`; VV before VH. | CODE | `segmentation_trainer.py` L61–76 + L138–144 (eval L323–338 + L392–398), set comparison. `Dataset.py` L770–776, returns L826–839, `concat` L158. |
| trainer_scope | The order holds for channel-concatenating segmentation models. CD and recurrent trainers follow `configs['inputs']` (CD needs exactly 2); vivit/convlstm stack `[pre2, pre1, post]`. | CODE | `change_detection_trainer.py` L19, L117–136; `recurrent_trainer.py` L110–131; `segmentation_trainer.py` L122–137. |
| hub_wrapper_order | The hub FloodViT wrapper's `[pre2, pre1, post]` → reversed `[post, pre1, pre2]` = POST, PRE2, PRE1 (same as channel_order). | CODE | `hubconf.py`@43ec8cf L44, L108–113, L136, L143; `Dataset.py`@43ec8cf L535–540 SL1 → sec1, L560–561 `pre1` = sec1, L771. |
| hub_wrapper_scaling | The hub wrapper uses the same mean/std but **no clamp / NaN fill**, and other DEM/slope constants (67.0293/1765.0062, 2.9482/79.2493). BATO keeps the training loader's clamp + NaN fill. | CODE | `hubconf.py`@43ec8cf L16–21, L88. |
| sec1_is_later_pre_event | sec1 (SL1) is the later pre-event image and sec2 (SL2) the earlier: **BATO PRE-2 → sec1, PRE-1 → sec2.** | DATA | SL1 date > SL2 date in 31,707 / 31,707 (train) and 67,490 / 67,490 (test) samples (train ⊂ test). |
| flood_is_post_event | flood (MS1) is the post-event image. | DATA | MS1 ≥ flood_date, and SL1/SL2 < flood_date, in 100 % of samples (0–43 d). Day resolution only: 7,288 test samples are same-day. |
| sar_value_domain | The loader applies no dB or log transform. The GRD graph writes linear σ⁰ → BATO feeds linear σ⁰. | CODE | `Dataset.__getitem__` L669–729 (`IMREAD_ANYDEPTH`, `uint8: false`). `torch.log` only in `custom` (L446). Graph `outputImageScaleInDb false`. |
| released_pixel_units | Units of the released pixel arrays. | **ND** | Pixels not inspected; only float32 and no-data 0.0 verified. |
| clamp | Clamp to [0, 0.15]. | CODE | `Dataset.concat` L164–165; `data_config.json` L14. |
| nan_fill | NaN → 0.15 after the clamp. | CODE | `Dataset.concat` L166. |
| zscore | `(x − mean) / std` with the constants above. | CODE | `scale_img` L193–198 (torchvision `Normalize`); `data_config.json` L15–17; applied at L811–822. |
| zscore_derivation | How those constants were computed. | **ND** | Hard-coded. |
| dem_channel | No DEM, no slope. | CODE | `data_config.json` sets them true (L24–25), then false (L30–31). pyjson5 keeps the last key. `main.py` L66–67. |
| dem_source | Source, units and resolution of `MK0_DEM`. | **ND** | Not produced by any repository code. |
| tile_size | 224 × 224 at 10 m, EPSG:3857. | DATA | All 67,490 footprints are 2240 × 2240 m with corners ≡ 5 m mod 10. `catalogue.yaml` L14. `Dataset` L780. |
| no_resampling | No resize or crop at inference. | CODE | `data_augmentations: false`; augmentation is train-only (L792–806). |
| sar_nodata | SAR no-data is 0.0 and is fed to the model unmasked. | CODE | `catalogue.py` L62; `normalize` ignores `valid_mask`. |
| label_classes | 3 classes plus ignore value 3. | CODE | `config.json` L14; `create_loss` `ignore_index=3`; MLU no-data 3. |
| label_names | 0 no water, 1 permanent water, 2 flood, 3 invalid. | CODE | `segmentation_trainer.py` L13 `CLASS_LABELS`. |
| debris_labels | **No debris/damage class.** | CODE | `num_classes` 3, `CLASS_LABELS`. The only "damage" match is the MIT licence text. |
| tiling_policy | Stride, origin, edge tiles shifted to fit (no padding unless an axis < 224), overlaps averaged (see §3). | BATO | Kuro Siwo has no tiling code; ViT attention is global (`vision_transformer.py` L84–89). |
| invalid_pixel_policy | See §4. | BATO | — |
| radar_visibility | No layover/shadow channel exists in the model input. | **GAP** | Step 3.4 saves none (Kuro Siwo neither). Step 4C computes approximate layover/shadow gates for its own state raster (`bato.detection.terrain_gates`); the separate `bato.visibility` package is empty. |
| pretrained_checkpoints | Input config of FloodViT / SNUNet weights. | **ND** | Not in the repository. |
| terrain_correction_dem | GLO-30 (BATO) vs SRTM 1Sec (Kuro Siwo). | DEV | Step 3.4 decision. |
| land_sea_mask | BATO omits Land-Sea-Mask. | DEV | Step 3.2 decision. |

Torch equivalence is reasoned from the semantics of torch 1.13.1 and torchvision 0.14.1, the versions Kuro Siwo's `requirements.txt` pins. It was not executed, because torch is not installed here. Under those semantics, `clamp` propagates NaN, `nan_to_num(x, v)` sets `nan=v`, and `Normalize` computes `(x − mean)/std` in the tensor dtype. The numpy implementation follows the same float32 operations. ±inf → 0.15 / 0.0. Re-check this once with torch installed at Step 4B.

## 3. Tiling (BATO design choice)

- **Tile size:** 224 × 224 (Kuro Siwo).
- **Origin:** stack pixel (0, 0).
- **Stride:** 1–224, default 224. `--stride 112` gives 50 % overlap.
- **Offsets along each axis:** 0, s, 2s, … A tile is added only while the previous tile ends inside the stack, so every pixel is covered and no tile is added after the stack end is reached; consecutive tiles are never more than one stride apart.
- **Edges (`shift_to_fit`, the default since the Step 4B audit):** when an axis is ≥ 224, its last tile is pulled back to start at `length − 224`. It then ends exactly at the stack edge, holds only real pixels, and overlaps its neighbour; consumers average the overlap (FloodViT inference averages logits).
  - **Why:** FloodViT's attention is global (`vision_transformer.py` L84–89), so padding inside a tile would change the outputs of its real pixels.
  - **Short axes:** only an axis shorter than 224 is still padded with 0.0 (the SAR no-data value), with validity 0.
  - **Trishuli example:** the 7,316 × 6,186 stack keeps 33 × 28 tiles. The last row of tiles starts at 7,092 and the last column at 5,962; no tile is padded.
- **`pad_end`** (the earlier policy: the last tile runs past the edge and is padded) remains selectable for comparison only.
- **Changing the policy changes the spec fingerprint.** Re-run `prepare_model_input.py --overwrite` for an existing manifest.
- **Tile transform:** the stack transform offset by whole pixels. Tile id `r<row:05d>_c<col:05d>`.

## 4. Validity, no-data and observability

- The mask is Step 3.5's `valid_mask.tif`: finite and ≠ 0.0 in **all six** layers.
- The adapter re-checks this over every pixel and refuses any disagreement. It also refuses NaN/inf and negative values: Kuro Siwo's NaN → 0.15 rule is not allowed to hide a broken stack.
- **Input values at invalid pixels** are passed through unchanged, as in Kuro Siwo files: per-layer 0.0, and partial observations are kept. They are then normalised like any value, so 0.0 VV becomes −2.232. BATO does not invent another fill value.
- **Output rule:** every prediction or label at mask 0 must become ignore = 3 (`apply_ignore`), which is `ImpactClass.UNOBSERVABLE` downstream. It is never read as water or as non-water. This matters because 0.0 backscatter looks like water to a SAR model.
- **Radar visibility (layover/shadow) is not available.** The mask encodes SAR no-data only (GAP).

## 5. Labels (none created in Step 4A)

Kuro Siwo `MK0_MLU` labels: `0` no water, `1` permanent water, `2` flood, `3` invalid/ignore (`KuroSiwoLabel`).

**There is no debris class.** A model trained on Kuro Siwo maps water only. BATO's debris/disturbance layer must therefore stay a change-statistics product (`ImpactClass.DEBRIS_DISTURBANCE`), as the detection contract already states. It cannot come from a Kuro Siwo-trained network.

## 6. Running

```bash
python scripts/prepare_model_input.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json [--stride 224] [--overwrite]
```

**What it reads.** It reads the published Step 3.5 stack only. It does not modify the stack and does not download anything.

**Checks.** Before writing anything, it checks:
- the stack provenance is succeeded and validated, layout v1, time order PRE1 < PRE2 < POST, one track, EPSG:3857 at 10 m, float32 with no-data 0;
- the stack and mask SHA-256 match the provenance, before and after the run;
- the raster header matches the provenance;
- every pixel is consistent with the mask.

**What it writes.** It writes only `data/interim/preprocessing/<aoi>/model_input/<candidate_id>/model_input_manifest.json`. The manifest holds:
- the spec and its SHA-256 fingerprint;
- the source stack, mask and provenance paths and hashes, plus the stack's config fingerprint and the current one;
- the passes with their Kuro Siwo roles;
- the grid;
- the tile index with valid-pixel counts;
- per-channel clamp-saturation diagnostics;
- warnings and unresolved items.

Publishing is atomic (`.tmp-` → final; `--overwrite` keeps `.superseded-`).

**Reading tiles.** No tensors are written. Training and inference read tiles with `ModelInputReader`. The reader re-verifies the manifest, the code's spec (fingerprint), and the stack and mask hashes. It re-checks each tile and returns `x` (normalised) and `valid`.

**Exit codes:** 0 ok; 1 refused or invalid; 2 rasterio missing.

## 7. Open before Step 4B

1. **released_pixel_units / tile shape on disk:** open one released Kuro Siwo sample (GeoTIFF or `.npy`). Check VV/VH are linear, with a median of roughly 0.02–0.1 and not negative dB, and that the arrays are 224 × 224.
2. **zscore_derivation:** not needed to reproduce the inputs, because the constants are fixed. It is recorded as unknown.
3. **dem_source:** irrelevant while no DEM channel is used.
4. **radar_visibility:** the layover/shadow mask is still unimplemented. Until it exists, outputs in steep terrain carry no visibility qualification beyond no-data.
5. **pretrained_checkpoints:** BATO trains its own model with the configuration above. Released weights must not be used unless their input config is verified.

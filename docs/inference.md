# Step 4B — FloodViT inference

**Status: model output, NOT VALIDATED.** No accuracy evidence exists for this checkpoint on BATO or Trishuli data. The probability is uncalibrated. Published damage maps (EMSR, UNOSAT, …) are never inputs.

## What runs

| Part | Where | Source |
|---|---|---|
| Network | `bato/inference/floodvit_model.py` | Kuro Siwo `models/vision_transformer.ViT`, imported from your checkout, plus the decoder of `seg_finetuner.py` / `model_utilities.py`. This is `hubconf.py@43ec8cf vit_decoder`: 205,538,499 parameters, 280 keys, `strict=True`. Unchanged from the original script. |
| Inputs | Step 4A (`bato/preprocessing/model_input.py`) | Verified stack and `valid_mask.tif` (SHA-256), channel order POST, PRE2, PRE1, clamp [0, 0.15], NaN → 0.15, z-score. |
| Tiling | Step 4A `shift_to_fit` | 224 × 224, no padding when an axis is ≥ 224. Logits of overlapping tiles are averaged. |
| Classes | Kuro Siwo `CLASS_LABELS` | `0` no water, `1` permanent water, `2` flood, `3` invalid (nodata). Argmax with no threshold, as in Kuro Siwo evaluation. |
| Probability | softmax of the averaged logits, class 2 | **Uncalibrated**; −1 = invalid. |
| Statistics | `floodvit_inference_provenance.json` | Over **valid pixels only**, class 1 and class 2 separate. Also: what the model would have said at invalid pixels (discarded), and flood fraction near internal tile edges vs tile interiors (seam check). |

## Modes

All modes run the same network on different inputs. None of them is a validation.

| Mode | POST slot | sec1 (PRE2) slot | sec2 (PRE1) slot | What it isolates |
|---|---|---|---|---|
| `event` | POST | PRE2 | PRE1 | The real input. |
| `no_event` | PRE2 | PRE1 | PRE1 | Change PRE1 → PRE2 (here 48 d, monsoon onset → peak), with **identical** pre images (never seen in Kuro Siwo training). **Confounded** by real seasonal water change, so **not** a clean false-positive baseline. |
| `zero_change` | PRE2 | PRE2 | PRE1 | Post equals sec1 exactly. Any "flood" is not driven by post-vs-sec1 change. |
| `static_post` | POST | POST | POST | No temporal information. Any "flood" or "permanent water" comes from single-date appearance: terrain, shadow, dark surfaces. |

Run `--mode all` (event plus the three controls), then `scripts/diagnose_floodvit.py`, which measures where each mode's water falls: clamp saturation, darkening, slope, valley-floor proxy, approximate radar geometry, tile edges, clusters, and agreement with the event map. The forensic reading and the pre-registered usability criteria are in [forensics/step4b_floodvit_forensics.md](forensics/step4b_floodvit_forensics.md).

## Run

```powershell
python scripts/run_floodvit_inference.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json `
  --checkpoint C:/Users/Login/Downloads/kuro_check/kurosiwo_vit_decoder_weights.pt `
  --kuro-repo C:/Users/Login/Downloads/kuro_check/KuroSiwo --mode all
python scripts/diagnose_floodvit.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json
```

**Outputs** go to `data/interim/inference/<aoi>/<candidate>/floodvit/` and `floodvit_no_event/`. Publishing is atomic, and `--overwrite` keeps the old output as `.superseded-`.

**Memory:** the logit sums take 3 × H × W × 4 bytes, about 0.55 GB for Trishuli.

**Old outputs:** the outputs of the original ad-hoc script (`flood_prediction.tif` directly under `<candidate>/`) are left untouched. They are superseded: they contain model classes at invalid pixels, padded edge tiles, and statistics over all pixels.

## Tests

- **Without torch:** `tests/unit/test_floodvit_inference.py` and `tests/integration/test_floodvit_script.py`. Fake numpy models cover:
  - no padding when avoidable;
  - invalid → 3 / −1;
  - class 1 / class 2 counts;
  - channel order;
  - the clamp path;
  - determinism;
  - overlap averaging and seams;
  - the no-event run;
  - refusals.
- **Opt-in, with torch:** `tests/integration/test_floodvit_model.py` (`BATO_KURO_REPO`, optional `BATO_FLOODVIT_CHECKPOINT`) covers parameter and key counts, strict load, output shape, and determinism of a repeated identical tile.

## Before connectivity

These are sanity checks, not validation:

1. `event` vs `no_event` flood fraction.
2. The seam ratio in the provenance.
3. Flood pixels against terrain derived from the Step 2C GLO-30 tiles: slope, and height above the nearest drainage.
4. Flood pixels against pre-event OSM waterways.
5. The share of flood pixels with a VV decrease between PRE-2 and POST.
6. The share of flood pixels at the 0.15 clamp.

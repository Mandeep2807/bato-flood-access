# Kuro Siwo: preprocessing facts to match before the preprocessing milestone

Why this matters: the flood model will be trained on Kuro Siwo and run on our own CDSE-derived Sentinel-1 data. If our preprocessing differs (calibration, filter, units, projection, normalisation), the model sees different numbers at inference: **domain shift**. Nothing here is downloaded or trained yet.

Status legend: **VERIFIED** = read from an authoritative source named in the row · **OPEN** = not yet verified; do not assume.

| Item | Kuro Siwo (GRD) | Status | Source |
|---|---|---|---|
| SAR product type | Sentinel-1 GRD (separate SLC product also provided) | VERIFIED | Paper (arXiv 2311.12056); repo README |
| Polarisation | VV and VH for every image | VERIFIED | Paper; HF dataset card (`*_vv.npy`, `*_vh.npy`) |
| Temporal structure | 2 pre-event images (`sec1`, `sec2`) + 1 post-event (`flood`), varying time gaps. **`sec1` = LATER pre-event, `sec2` = EARLIER** (BATO PRE-2 → sec1, PRE-1 → sec2); `flood` always on/after the flood date | VERIFIED (Kuro Siwo data, Step 4A) | `pickle/KuroV2_grid_dict*.gz`: SL1 date > SL2 date in 67,490/67,490 samples |
| Labels | 3 classes: no water, permanent water, flood; plus a valid-pixel mask (0 invalid / 1 valid) | VERIFIED | Paper; HF dataset card |
| Orbit/geometry of pre vs post | Whether the two pre images and the post image always share one relative orbit | **OPEN** | Not stated in material read; check `info.json` per sample |
| Orbit correction | Apply-Orbit-File, "Sentinel Precise (Auto Download)" | VERIFIED | `configs/grd_preprocessing.xml` (SNAP graph) |
| Noise removal | ThermalNoiseRemoval; Remove-GRD-Border-Noise (borderLimit 500, trimThreshold 50) | VERIFIED | SNAP graph |
| Land/sea mask | Land-Sea-Mask (useSRTM true, landMask false) | VERIFIED | SNAP graph |
| Radiometric representation | Calibration to **sigma0** (sigma band only; no gamma0/beta0) | VERIFIED | SNAP graph |
| Speckle filter | **Lee Sigma**, window 7×7, filter size 3×3 | VERIFIED | SNAP graph |
| Terrain correction | Range-Doppler Terrain-Correction, **SRTM 1Sec HGT** DEM, bilinear image and DEM resampling, no layover/shadow mask saved | VERIFIED | SNAP graph |
| Resolution / pixel spacing | 10.0 m | VERIFIED | SNAP graph; paper |
| Map projection | **WGS 84 / Pseudo-Mercator** (EPSG:3857) | VERIFIED | SNAP graph |
| dB conversion in SNAP | **No** LinearToFromdB step in the graph (output is linear sigma0) | VERIFIED | SNAP graph |
| Units of released arrays | Loader applies no dB/log (code); the released pixel values themselves were not inspected | Loader VERIFIED (code); pixels **OPEN** | `dataset/Dataset.py` L669–729; see model_input.md `released_pixel_units` |
| Normalisation / scaling for model input | Fixed z-score per channel: VV mean 0.0953 std 0.0427, VH mean 0.0264 std 0.0215, same for all time steps (derivation of the constants unknown) | VERIFIED (code) | `Dataset.scale_img` L193–198; `data_config.json` L15–17 |
| Clipping / log transform before the model | Clamp [0, 0.15], then NaN → 0.15; no log in the default path | VERIFIED (code) | `Dataset.concat` L164–166; `data_config.json` L14 |
| DEM channel | Default model uses **no DEM and no slope** (config sets them true then false; last key wins). Source/units of `MK0_DEM` / `dem.npy` not determinable | Default VERIFIED (code); source **OPEN** | `data_config.json` L24–31; `main.py` L66–67 |
| Tile size | 224 × 224 at 10 m (every GRD footprint 2240 m × 2240 m); no resize at inference | VERIFIED (Kuro Siwo data + code) | pickles; `Dataset` L780; array shape on disk not inspected |
| Nodata handling | SAR no-data 0.0, fed to the model unmasked; labels: 3 = invalid, ignored by every loss/metric; no debris class | VERIFIED (code) | `catalogue.py` L62; `create_loss` `ignore_index=3`; `segmentation_trainer.py` L13 |

## Step 2C status

Raw GRD products are fetched as complete SAFE zips. The fetch verifies that each contains the VV and VH measurement rasters plus `calibration-*.xml` (σ⁰ LUT) and `noise-*.xml`, which are the inputs needed to reproduce Kuro Siwo's chain (thermal noise removal → calibration to linear σ⁰ → Lee Sigma → terrain correction) in preprocessing. Measurement rasters are digital numbers; no conversion has happened yet.

## Step 3.2 status

Orbit, AOI subset, thermal noise removal, border noise removal and σ⁰ calibration (linear) are implemented with SNAP gpt, in the Kuro Siwo order and values; differences (orbit continueOnFail, no Land-Sea-Mask) are listed in [preprocessing.md](preprocessing.md).

Step 3.3: Lee Sigma (window 7×7, sigma 0.9, target window 3×3, 1 look) is implemented with the same Speckle-Filter node as Kuro Siwo (verified against `configs/grd_preprocessing.xml`, 2026-10-02: the node follows Calibration, and filterSizeX/Y 3, dampingFactor 2, estimateENL true, enl 1.0, anSize 50 are set but not used by Lee Sigma). Step 3.4: Range-Doppler terrain correction is implemented with the Kuro Siwo Terrain-Correction node (EPSG:3857 WKT, 10 m, bilinear, standard grid; verified against `configs/grd_preprocessing.xml`, 2026-10-02), except the DEM: BATO's Copernicus GLO-30 tiles as an external DEM instead of SRTM 1Sec. Decided 2026-10-02: keep **EPSG:3857** (consequence 4 below is answered: same grid convention as the training data).

## Step 4A status

Corroboration: the torch.hub FloodViT wrapper (`hubconf.py`, commit `43ec8cf`, PR #4) feeds `[post, pre1, pre2]` where `pre1` = `sec1` (later pre-event), i.e. BATO POST, PRE2, PRE1; it applies no 0.15 clamp (see model_input.md `hub_wrapper_scaling`).

The OPEN rows above were resolved from the Kuro Siwo code and the grid pickles shipped in its repository (commit `4347ed1`), or classified as not determinable; the locked model-input contract and every citation are in [model_input.md](model_input.md).

## Consequences for BATO (to decide in the preprocessing milestone)

1. Use **sigma0** (not gamma0 terrain-flattened) for model inputs, to match.
2. Match **Lee Sigma 7×7 / 3×3** speckle filtering. If CDSE processing only offers plain Lee, apply Lee Sigma ourselves on unfiltered data.
3. Terrain-correct at **10 m**. Our DEM must be the allowed **Copernicus DEM**, while Kuro Siwo used SRTM: a known, documented difference. Quantify it with the domain check (same scene through both pipelines, compare histograms).
4. Kuro Siwo grids are **EPSG:3857**. Our rasters may use UTM. The model is pixel-based, so projection matters only for pixel size and distortion: check that 10 m in 3857 at the training latitudes vs UTM at ~28° N does not change apparent feature size materially.
5. Resolve every **OPEN** row (units, normalisation, clipping, DEM channel) from the repository code or one downloaded sample before training. Do not train with guessed normalisation.

## Sources

- Bountos et al., 2024, *Kuro Siwo* (NeurIPS Datasets & Benchmarks), arXiv:2311.12056.
- `Orion-AI-Lab/KuroSiwo` repository, `configs/grd_preprocessing.xml` (SNAP graph, main branch, read 2026-10-01).
- Hugging Face dataset cards `orion-ai-lab/Kuro-Siwo-GeoTIFFs` / `Kuro-Siwo-Webdataset` (read 2026-10-01).

Note: the SNAP graph values were read through a text extraction of the XML. Open the XML once yourselves to confirm them before relying on them.

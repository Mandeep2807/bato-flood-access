# Sentinel-1 preprocessing

## Step 3.2 — radiometric stage (implemented)

One downloaded Sentinel-1 IW GRDH pass → **linear σ⁰ VV + VH in radar geometry**, with ESA SNAP's Graph Processing Tool (`gpt`) run as an external program.

```
Read (SAFE zip) → Apply-Orbit-File → Subset (AOI bbox + crop buffer) → ThermalNoiseRemoval
     → Remove-GRD-Border-Noise → Calibration (σ⁰, linear) → Write (BEAM-DIMAP)
```

Not done at this stage, by design: speckle filtering, terrain correction, map projection, slice assembly, dB conversion, normalisation, clipping.

### Command

```bash
python scripts/preprocess_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \
    --pass-id S1D-A085-20260629T122138 --stage radiometric
```

Options: `--catalogue <file>` (the Step 2A catalogue; found automatically by SHA-256 digest in `data/interim/catalogue/` or the project root), `--overwrite` (moves an existing output to `radiometric.superseded-<UTC>`; nothing is ever deleted), `--config-dir`.

Exit codes: 0 succeeded; 1 invalid input or refused (e.g. output exists); 2 SNAP or rasterio unavailable / preflight failed (nothing written); 3 gpt or validation failed (folder kept as `radiometric.failed-<UTC>`).

### Inputs (all from BATO's own records)

| What | Where it comes from |
|---|---|
| Product file, SHA-256, acquisition time, orbit | Step 2C fetch manifest + the product's `.provenance.json` (file size re-checked; never re-downloaded) |
| SAFE consistency | Step 3.1 `inspect_safe_zip` + `check_against_provenance`, before SNAP runs |
| AOI polygon | The Step 2A catalogue the manifest was planned from (query polygon; matched by digest) |
| Crop region | AOI bounding box grown by `preprocessing.crop_buffer_m` (PROVISIONAL 20 km = 15 km road-graph buffer + ~5 km margin for terrain displacement before terrain correction) |
| Parameters | `config/config.yaml` → `preprocessing.s1_radiometric` (fingerprinted) |

Every input path passes `assert_pipeline_input_allowed`; outputs must be under `paths.interim` (`assert_interim_output_dir`). Multi-slice passes are refused (slice assembly is a later step).

### Graph and parameters

`src/bato/preprocessing/graphs/s1_radiometric.xml` (version-controlled). `${...}` placeholders are filled from configuration; every placeholder must be filled and every value used. Before the long run, each parameter name is checked against the installed SNAP with `gpt <Operator> -h`.

| Operator | Parameters (explicit) |
|---|---|
| Apply-Orbit-File | orbitType `Sentinel Precise (Auto Download)`, polyDegree 3, continueOnFail **false** |
| Subset | geoRegion = crop polygon (WKT), subSampling 1×1, fullSwath false, copyMetadata true |
| ThermalNoiseRemoval | VV,VH; removeThermalNoise true; reIntroduceThermalNoise false; outputNoise false |
| Remove-GRD-Border-Noise | VV,VH; borderLimit 500; trimThreshold 50.0 |
| Calibration | auxFile `Product Auxiliary File`; VV,VH; outputSigmaBand true; beta/gamma outputs and creation false; outputImageInComplex false; outputImageScaleInDb **false** |
| Write | BEAM-DIMAP |

SNAP memory, cache and threads are not changed; SNAP's own settings apply and are recorded from `gpt --diag`.

### Differences from Kuro Siwo (`configs/grd_preprocessing.xml`)

| Item | Kuro Siwo | BATO Step 3.2 | Why |
|---|---|---|---|
| Orbit continueOnFail | true | false | a scene must not silently proceed without a precise orbit |
| Land-Sea-Mask | yes (SRTM) | omitted | uses SRTM, not an allowed BATO input |
| Lee Sigma, terrain correction | in the same graph | later steps (3.3/3.4) | stage-by-stage validation |
| DEM (later) | SRTM 1Sec | Copernicus DEM | PS data rules |

Orbit, subset, noise removal and σ⁰ calibration otherwise follow the Kuro Siwo order and values.

### Output

`data/interim/preprocessing/<aoi_id>/<pass_id>/radiometric/`

| File | Content |
|---|---|
| `<pass_id>_sigma0.dim` + `.data/` | `Sigma0_VV`, `Sigma0_VH` (ENVI float32, linear, unit `intensity`), tie-point grids |
| `graph_resolved.xml` | the graph actually executed (its file paths point to the temporary folder it ran in) |
| `gpt.log` | full gpt output |
| `preprocessing_provenance.json` | `PreprocessingProvenance` (see below) |

Atomicity: SNAP writes into `radiometric.tmp-<UTC>`; only after gpt succeeds **and** validation passes is it renamed to `radiometric`. Failed runs become `radiometric.failed-<UTC>` with a failure provenance.

### Validation (rasterio)

Structural checks decide validity: output present; exactly `Sigma0_VV` + `Sigma0_VH`; linear (unit not dB, no log10 scaling); readable; float dtype; same grid (and not larger than the input scene); radar geometry (tie-point geocoding, no map transform, `is_terrain_corrected` not set); no speckle/terrain/dB operator in SNAP's processing history; precise (POEORB) orbit applied; pixel spacing equal to the SAFE annotation (1 % metadata tolerance); lat/lon tie-point grids valid; AOI coverage of the output footprint ≥ the input scene's coverage (1 % grid tolerance); at least one valid pixel per band.

Diagnostics only (no thresholds): total / finite / non-finite / no-data / zero / negative / valid pixel counts; min, max, mean; p01–p99 and median from a regular subsample; zeros inside vs. outside the `border_limit` edge strip.

In radar geometry rasterio reports **no CRS and an identity transform**; this is expected. Geolocation is carried by the tie-point grids.

### Provenance

`PreprocessingProvenance` records: status/error; BATO version and config fingerprint; settings used; start/finish time; input product id/name/path/size/SHA-256 (from download provenance), acquisition time, orbit direction, relative orbit; AOI polygon, bbox, crop buffer, crop bbox and the exact geoRegion; graph template path + SHA-256, resolved graph path + SHA-256, every operator's parameters; gpt path, how it was found, SNAP version, `gpt --diag` head, command, return code, runtime, log path; output paths; full validation. No credentials are involved or stored.

### Tests

`tests/unit/test_snap.py`, `tests/unit/test_radiometric.py`, `tests/integration/test_preprocess_script.py` use a **fake gpt** (mocked `subprocess`) that writes a tiny synthetic DIMAP product: they test BATO's plumbing, not SNAP's science. Validation tests need rasterio (skipped otherwise). `tests/integration/test_snap_real.py` is opt-in (`BATO_RUN_SNAP_TESTS=1`) and checks the graph's operators/parameters against the installed SNAP without reading data.

### Known risks for the first real run

* SNAP must support the platform (S1D); otherwise the Read or orbit step fails and is reported.
* Precise orbit auto-download needs internet access from SNAP.
* Remove-GRD-Border-Noise works from the edges of the *subset*; dark pixels within `border_limit` (500 px ≈ 5 km) of the crop edge may be zeroed. The 20 km crop buffer keeps that strip outside the AOI; the edge vs. interior zero counts show it.

## Step 3.3 — Lee Sigma speckle filter (implemented)

The published Step 3.2 output → **Lee Sigma filtered linear σ⁰ VV + VH, still in radar geometry**.

```
Read (radiometric BEAM-DIMAP) → Speckle-Filter (Lee Sigma) → Write (BEAM-DIMAP)
```

Not done, by design: terrain correction, DEM, map projection, slice assembly, dB, normalisation, clipping.

### Command

```bash
python scripts/speckle_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \
    --pass-id S1D-A085-20260629T122138
```

Options: `--overwrite` (old output kept as `speckle.superseded-<UTC>`), `--config-dir`. Exit codes as Step 3.2 (0 ok; 1 invalid/refused; 2 SNAP/rasterio unavailable or SNAP does not accept a parameter/value — nothing written; 3 gpt or validation failed — `speckle.failed-<UTC>` kept).

### Input

Only the **published** `data/interim/preprocessing/<aoi>/<pass>/radiometric/` (a folder that exists only after Step 3.2 succeeded and validated). Before SNAP runs, BATO checks:

* its provenance is a succeeded, validated run for the same AOI, pass and product as the manifest;
* it was produced with the **current** radiometric settings (otherwise: re-run Step 3.2 or restore the config);
* the DIMAP holds exactly `Sigma0_VV` + `Sigma0_VH`, linear (no dB / log10), radar geometry, and no speckle filter, terrain correction or dB step in its history (no double filtering);
* every file of the input product is SHA-256 hashed; the hashes are compared again after the run (the input must not change) and recorded in provenance.

### Graph and parameters

`src/bato/preprocessing/graphs/s1_speckle.xml`. The Speckle-Filter node is **identical to the Kuro Siwo node** (`configs/grd_preprocessing.xml`), with `sourceBands` set explicitly to `Sigma0_VV,Sigma0_VH`:

| Parameter | Value | Used by Lee Sigma? |
|---|---|---|
| filter | `Lee Sigma` (config; the only accepted value) | — |
| windowSize | `7x7` (config) | yes |
| sigmaStr | `0.9` (config) | yes |
| targetWindowSizeStr | `3x3` (config) | yes (point-target window) |
| numLooksStr | `1` (config) | yes |
| filterSizeX / filterSizeY | 3 / 3 | no (Boxcar, Median, Lee, …) |
| dampingFactor | 2 | no (Frost) |
| estimateENL / enl | true / 1.0 | no (Gamma Map, …) |
| anSize | 50 | no (IDAN) |

Before the run, `gpt <Operator> -h` must list every parameter name, and for Speckle-Filter every value must be in SNAP's value set ("Value must be one of …"). A discrepancy stops the run before anything is written; no other filter is ever substituted. A value set that SNAP's help does not list is recorded as a provenance warning.

### Kuro Siwo compatibility

* Same operator, same parameter values, same input domain (linear σ⁰ straight after calibration, radar geometry, before terrain correction).
* `numLooksStr = 1` matches Kuro Siwo, although IW GRDH data has an equivalent number of looks of roughly 4–5. It is kept at 1 for training/inference consistency; changing it would change the filter's sigma ranges.
* Differences from Kuro Siwo's input (from Step 3.2): no Land-Sea-Mask before calibration, orbit `continueOnFail=false`; SNAP 12 vs. the SNAP version Kuro Siwo used (not stated).

### Output and validation

`data/interim/preprocessing/<aoi>/<pass>/speckle/`: `<pass>_sigma0_spk.dim/.data`, `graph_resolved.xml`, `gpt.log`, `preprocessing_provenance.json` (`SpeckleProvenance`). Atomic publishing as in Step 3.2.

Checks (validity): output present and complete; exactly `Sigma0_VV` + `Sigma0_VH`; linear; readable; float; same grid as each other and as the input; radar geometry; no terrain/dB/other operator after the filter, the filter recorded once; orbit file, pixel spacing and tie-point geolocation identical to the input; at least one valid pixel; the filter changed the data; no pixel that was valid in the input is NaN/inf afterwards.

Diagnostics (no thresholds): before/after valid, no-data, zero, non-finite counts; min, p01–p99, median, max, mean; changed pixels; coefficient of variation before → after; mean absolute change; new no-data, restored and new negative pixels.

### Tests

`tests/unit/test_speckle.py`, `tests/integration/test_speckle_script.py` use a **fake gpt** (a 3×3 mean, not Lee Sigma) on a Step 3.2 output made by the real Step 3.2 code with the Step 3.2 fake: they test BATO's plumbing and checks, not SNAP's filter. `tests/integration/test_snap_real_speckle.py` (opt-in, `BATO_RUN_SNAP_TESTS=1`) checks the graph's operators, parameters and Lee Sigma values against the installed SNAP.

## Step 3.4 — Range-Doppler terrain correction (implemented)

The published Step 3.3 output → **Lee Sigma filtered linear σ⁰ VV + VH, terrain corrected, on the EPSG:3857 10 m grid**: the last step of the Kuro Siwo SNAP chain (Calibration → Speckle-Filter → Terrain-Correction).

```
Read (speckle BEAM-DIMAP) → Terrain-Correction (Copernicus DEM, EPSG:3857, 10 m) → Write (BEAM-DIMAP)
```

Not done, by design: dB, normalisation, clipping, radiometric terrain flattening (γ⁰), layover/shadow mask, slice assembly, tiling.

### Command

```bash
python scripts/terrain_correct_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \
    --pass-id S1D-A085-20260629T122138
```

`--overwrite` keeps the old output as `terrain.superseded-<UTC>`. Exit codes: 0 ok; 1 invalid/refused (incl. missing or changed DEM tiles); 2 SNAP/rasterio unavailable or SNAP does not accept a parameter/value (nothing written); 3 DEM preparation, gpt or validation failed (`terrain.failed-<UTC>` kept).

### Decisions (taken 2026-10-02)

| Question | Decision | Why |
|---|---|---|
| Map projection | **EPSG:3857** (WGS 84 / Pseudo-Mercator), 10 m, aligned to the standard grid at (0, 0) — Kuro Siwo's exact WKT | Same grid convention as the training data; the standard grid makes every pass of one AOI pixel-aligned. At 28° N a 10 m 3857 pixel is ≈ 8.8 m on the ground (the same scale distortion the Kuro Siwo samples have at their latitudes). Metric work (road distances) can reproject later. |
| DEM | **BATO's Step 2C Copernicus GLO-30 tiles**, mosaicked into one GeoTIFF and passed as SNAP external DEM | Allowed input, fully under BATO provenance; no SNAP auto-download. Kuro Siwo used SRTM 1Sec (not an allowed input): a documented difference. |

### Input and DEM

* Input: only the **published** `speckle/` folder. Its provenance must be a succeeded, validated run for the manifest's AOI/pass/product, produced with the current speckle settings over a Step 3.2 output with the current radiometric settings. The Step 3.2 provenance must still have the SHA-256 Step 3.3 recorded (chain of custody). The DIMAP must be exactly `Sigma0_VV` + `Sigma0_VH`, linear, radar geometry, Speckle-Filter recorded once, not terrain corrected / dB.
* DEM: the input footprint (lat/lon tie-point corners) is grown by `dem_margin_km` (PROVISIONAL 20 km: steep terrain is imaged displaced by up to ≈ height × cot(incidence), ≈ 15 km for 8 km peaks at 30°). Every 1×1° cell it touches must be planned and downloaded in Step 2C; each archive is re-hashed against its download provenance. The `*_DEM.tif` is read from the zip/tar, the tiles are placed on their common grid (values copied unchanged; resolution or grid mismatch is refused), cut to the extent and written as float32 GeoTIFF (EPSG:4326, no-data `dem_nodata` = −32767 where a cell has no tile, e.g. open sea). The mosaic is published inside the output folder and its SHA-256 recorded; it must be unchanged after the run.
* Heights: Copernicus DEM gives EGM2008 geoid heights; `externalDEMApplyEGM: true` (Kuro Siwo value) makes SNAP convert them to ellipsoid heights with EGM96. The EGM96/EGM2008 difference is a known residual (see limitations).

### Graph

`src/bato/preprocessing/graphs/s1_terrain.xml`: the Kuro Siwo Terrain-Correction node parameter for parameter — projection WKT, `pixelSpacingInMeter` 10.0 / `pixelSpacingInDegree` 8.983152841195215E-5, `BILINEAR_INTERPOLATION` for DEM and image, `alignToStandardGrid` true at (0, 0), `nodataValueAtSea` true, only the selected source bands saved, no radiometric normalisation, no DEM / lat-lon / angle / layover-shadow bands — except the DEM (`demName` "External DEM", `externalDEMFile`, `externalDEMNoDataValue` −32767 instead of 0.0, `externalDEMApplyEGM` true) and explicit `sourceBands` `Sigma0_VV,Sigma0_VH`. Every parameter name and every value that has a SNAP value set (resampling methods, incidence-angle choices, auxFile) is checked against the installed SNAP before the run.

### Output and validation

`data/interim/preprocessing/<aoi>/<pass>/terrain/`: `<pass>_sigma0_spk_tc.dim/.data`, `dem/<pass>_copdem30.tif`, `graph_resolved.xml`, `gpt.log`, `preprocessing_provenance.json` (`TerrainProvenance`). Atomic publishing as in Steps 3.2/3.3.

Checks (validity): output present and complete; exactly `Sigma0_VV` + `Sigma0_VH`; linear (no dB / log10); readable; float; one grid for both bands; map geocoding in **EPSG:3857** (DIMAP WKT; rasterio cross-checked); pixel size equal to the graph's 10 m, no shear; **pixel centres** on the standard 10 m grid (SNAP's `alignToStandardGrid` snaps the centre of pixel (0, 0) — `CrsGeoCoding` reference pixel 0.5, 0.5 — so the DIMAP corner origin sits at x5 m, e.g. 9451025.0; a corner on a whole 10 m multiple means a half-pixel shift and fails); SNAP history with Terrain-Correction once, Speckle-Filter once and no dB/other step; `is_terrain_corrected` set; orbit file unchanged; at least one valid pixel; the AOI (Step 3.2 provenance polygon) covered by valid pixels at least as well as the input scene covered it (1 % sample-grid tolerance).

Diagnostics (no thresholds): statistics before (radar geometry) and after (map geometry): size, valid / no-data / non-finite counts, min, p01–p99, median, max, mean; AOI valid fractions; DEM no-data pixels inside the footprint.

### Tests

`tests/unit/test_terrain.py`, `tests/integration/test_terrain_script.py`: a **fake gpt** writes a small EPSG:3857 product (not a real Range-Doppler correction) on a Step 3.3 output made by the real Step 3.2/3.3 code; synthetic 0.01° DEM tiles exercise the mosaic. The fixture replaces the Step 3.2 provenance AOI with a ~1 km test AOI before Step 3.3 runs, so the fake 10 m output stays small and the provenance chain stays consistent. `tests/integration/test_snap_real_terrain.py` (opt-in, `BATO_RUN_SNAP_TESTS=1`) checks the graph against the installed SNAP.

## Step 3.5 — temporal stack PRE-1 / PRE-2 / POST (implemented)

The architecture's "pair stack" (preprocessing → *terrain-corrected stacks* → detection): the three published Step 3.4 outputs of ONE temporal-set candidate, on one common grid, in one file.

```bash
python scripts/stack_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json [--candidate ts-…] [--overwrite]
```

Output `data/interim/preprocessing/<aoi>/stack/<candidate_id>/`: `stack_sigma0.tif` (6 × float32, EPSG:3857, 10 m, no-data 0, DEFLATE-tiled), `valid_mask.tif` (uint8, 1 = valid in all six layers), `preprocessing_provenance.json` (`StackProvenance`). Atomic publishing as in Steps 3.2–3.4.

**Layout v1** (BATO storage order with explicit band descriptions and tags — *not* a model channel order): `1 PRE1_VV, 2 PRE1_VH, 3 PRE2_VV, 4 PRE2_VH, 5 POST_VV, 6 POST_VH`. PRE-1 = earlier pre-event pass, PRE-2 = later one (assumptions #13).

### What it does and does not do

* **Inputs** (read-only, fail closed): each pass's `terrain/` provenance succeeded and validated, every file re-hashed against the Step 3.4 `output_sha256`, produced with the current terrain/speckle settings, same AOI polygon, same terrain graph template and Terrain-Correction parameters (except the per-pass DEM path), same DEM dataset; DIMAP exactly `Sigma0_VV` + `Sigma0_VH`, linear (no dB / log10), float32, no-data 0, EPSG:3857 WKT, 10 m north-up.
* **Roles and event:** the candidate must be exactly 2 pre + 1 post; plan roles must match; PRE-1 < PRE-2 < POST; PRE-2 dated before and POST after the event date (same-day images refused — the manifest carries no event time); all on the candidate's track. Days from the event are recorded per input.
* **Grid:** window = AOI bounding box (Step 3.2 provenance polygon) + `aoi.analysis_buffer_km` (15 km, the road-graph analysis extent), snapped outward to the shared 10 m standard grid. Every Step 3.4 output lies on that grid, so each source window is a **whole-pixel offset** and values are **copied bit for bit** (no resampling). A phase / pixel-size mismatch, or a window not fully inside any source, fails closed.
* **Values:** linear σ⁰ exactly as Step 3.4 produced it — no dB, no normalisation, no clipping, no NaN replacement. No DEM-derived layer.

### Validation

Checks: per layer zero NaN/inf values; zero negative values (linear intensity cannot be negative — dB data would be); at least one valid pixel; the written file has 6 float32 bands, EPSG:3857, the planned transform and the layout v1 band names; every band equals its source window bit for bit (SHA-256 of the copied values vs. read-back); the AOI is valid in all six layers at least as well as the inputs covered it (1 % tolerance); the Step 3.4 outputs are unchanged after the run. Diagnostics: per-layer counts, min, p01–p99, median, max, mean; pixels valid in all layers.

### Open specification items (not decided here; recorded in every provenance)

Items 1–5 below are resolved or classified by Step 4A in [model_input.md](model_input.md); the Step 3.5 code and the strings it records are unchanged.


1. Model channel order and the mapping of PRE-1/PRE-2/POST to Kuro Siwo's `sec1`/`sec2`/`flood` (sec1/sec2 ordering OPEN in kuro_siwo_preprocessing.md).
2. Normalisation / log transform / clipping applied by the Kuro Siwo dataloader (OPEN).
3. DEM channel (`dem.npy` source) and slope (OPEN) — no DEM layer is added.
4. Tile size / tiling for inference (Kuro Siwo 224 × 224).
5. Units of the released Kuro Siwo arrays (linear vs dB) (OPEN).

### Tests

`tests/unit/test_stack.py`, `tests/integration/test_stack_script.py`: three Step 3.4 outputs made by the real Step 3.2–3.4 code with fake gpts (different extents on one lattice; test config `aoi.analysis_buffer_km` 0.1 km to keep the fake outputs small). `tests/integration/test_stack_real_data.py` is opt-in (`BATO_REAL_DATA_MANIFEST=<manifest>`): read-only checks of the real Step 3.4 outputs, and the stack plan when all three exist.

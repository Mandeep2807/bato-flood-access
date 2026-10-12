# BATO — Satellite Road-Access Intelligence

> *Bato* (बाटो) is Nepali for "road" or "path".

**Goal:** for a valley and a flood date, use Sentinel-1 radar to map where
open-water-like flooding is visible, say which roads that blocks and which
settlements may have lost road access, state how certain each conclusion is
(including where radar cannot tell), and rank the blocked road stretches whose
clearing would reconnect the most settlements.

Built for **Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from
Space**. The official problem statement (PS) is the source of truth for
requirements and data rules. This is an educational prototype, not an
operational tool.

---

## Release status

**Official submitted result (Trishuli, Nepal, event 2026-08-26):**

1. **Step 4C SAR-change flood baseline:** new open-water-like darkening on a Sentinel-1 PRE-1 / PRE-2 / POST stack
   (S1D, ascending, relative orbit 85), with terrain and radar-geometry gating to NOT_ASSESSED. Provisional
   thresholds, `validated: false`.
2. **Road rule v1 connectivity:** pre-event OSM roads → OPEN / BLOCKED / UNOBSERVABLE → three graphs →
   settlement access state (connected / access unverified / confirmed cut off / no pre-event road access) →
   ranked blocked stretches, with an evidence record for every claim.
3. **Dashboard** of that v1 run (hashes verified, summary recomputed, v2 refused).
4. **EMSR927 case-study comparison**, run afterwards as **evaluation only** under the frozen, pre-registered
   protocol `bato-emsr927-v1` ([docs/evaluation/emsr927_protocol.md](docs/evaluation/emsr927_protocol.md)).

**Not part of the official result:**
- **Road rule v2 and the SAR disturbance (debris) detector.** This was an experiment, **rejected** under its
  pre-registered criteria ([docs/debris_detection.md](docs/debris_detection.md)).
- **FloodViT inference.** Preliminary and not validated.

| Component | Status |
|---|---|
| Configuration (`config/*.yaml`, strict validation, provisional thresholds, fingerprint) | Implemented |
| Logging + per-stage reports + exception hierarchy | Implemented |
| Data contracts (AOI, acquisition, pre/post pair, detection, road impact, settlement status, evidence) | Implemented |
| Data-rule guards (evaluation-only folders, import isolation, permitted sources) | Implemented |
| Sentinel-1 catalogue discovery, same-track temporal-set selection, GRD + Copernicus DEM fetching (Steps 2A–2C) | Implemented |
| SAFE inspection, radiometric calibration (SNAP), Lee Sigma, terrain correction, temporal stack (Steps 3.1–3.5) | Implemented; single-slice passes only |
| Model-input contract (Step 4A) | Implemented ([docs/model_input.md](docs/model_input.md)) |
| SAR change flood baseline (Step 4C) | Implemented; **official flood layer**; **not validated** ([docs/baseline_detection.md](docs/baseline_detection.md)) |
| Pre-event OSM (snapshot = event − 1 day), road impact (rule v1), connectivity, settlement access, stretch ranking, evidence records | Implemented; **official**; **not validated** ([docs/connectivity.md](docs/connectivity.md)) |
| Dashboard of the official v1 result | Implemented ([docs/dashboard.md](docs/dashboard.md)) |
| EMSR927 comparison (evaluation only; acquisition, inspection, frozen protocol, metrics) | Implemented ([docs/evaluation/emsr927_protocol.md](docs/evaluation/emsr927_protocol.md)) |
| FloodViT inference (Step 4B) | Implemented; preliminary; **not validated; not official** ([docs/inference.md](docs/inference.md)) |
| SAR disturbance (debris) detector + road rule v2 | Implemented as an experiment; **rejected; not official** ([docs/debris_detection.md](docs/debris_detection.md)) |
| Slice assembly (multi-slice passes), Sentinel-2, flood-model training, fusion | **NOT IMPLEMENTED** |
| Building / bridge / facility impact prediction, situation report | **NOT IMPLEMENTED** |
| Kuro Siwo test-event and Himalayan-holdout evaluation | **NOT IMPLEMENTED** |

BATO searches the Sentinel-1 catalogue and downloads a same-track PRE-1 / PRE-2 / POST set. It calibrates and
terrain-corrects each pass with ESA SNAP and stacks them. It then produces the Step 4C flood state raster, road
states, settlement access states and the dashboard from that imagery and pre-event OSM. Every number on the
dashboard is read from a hash-verified run; none is typed in.

## The problem

When roads, bridges and phone lines fail after a flood, rescuers need to know
where the damage is and who is cut off. The PS asks for a system that takes an
area and a flood date and answers: **where did the flood hit, what was
damaged, and who is cut off** — and that works live on areas and dates chosen
by the judges, not only on the Trishuli case study.

## Pipeline (as submitted)

```
AOI + event date
 → Sentinel-1 catalogue → same-relative-orbit PRE-1 / PRE-2 / POST set → GRD + DEM download
 → SNAP calibration → Lee Sigma → terrain correction (EPSG:3857, 10 m) → temporal stack
 → Step 4C flood baseline (FLOOD / NO_FLOOD / PRE_EVENT_WATER / NOT_ASSESSED + reason)
 → pre-event OSM (event − 1 day) → road impact, rule v1 (OPEN / BLOCKED / UNOBSERVABLE)
 → pre / post-optimistic / post-pessimistic graphs → settlement access state → blocked-stretch ranking
 → evidence records → dashboard
```

Settlement access state is one of:
- `CONNECTED`;
- `POSSIBLE_CUTOFF`: **access unverified**. A route exists only through roads radar could not observe. This is
  *not* a cut-off claim.
- `CONFIRMED_CUTOFF`: no route even if every unobservable road is passable;
- `NO_PRE_EVENT_ACCESS`: no mapped road access before the event, so not a flood effect.

Details: [docs/connectivity.md](docs/connectivity.md), [docs/architecture.md](docs/architecture.md).

## Data rules (from the PS)

**Allowed as pipeline inputs:** Sentinel-1, Sentinel-2, Copernicus DEM,
OpenStreetMap *as it was before the event*, and the listed training datasets
(Kuro Siwo; optionally Sen1Floods11).

**Not allowed as inputs:** Copernicus EMS, UNOSAT or any other published damage
map, and OSM edits made after the event. These may only be used to check
results. Breaking this rule leads to disqualification.

BATO also uses **no population datasets** (WorldPop, GHSL, …) and no external
water/flood products: they are not on the allowed list. Building counts from
pre-event OSM are reported instead, and are never called population.

**EMSR927 (Copernicus EMS) is evaluation-only reference data and never a BATO
production input.** It is downloaded into `data/reference/EMSR927/`, read only by
`bato.evaluation`, and compared with the already-finished official result.
Nothing in Step 4C, road impact, connectivity or the dashboard reads it, and no
threshold was tuned on it.

How the code enforces this:
- `data/reference/` and `data/eval/` are evaluation-only;
  `bato.data_rules.assert_pipeline_input_allowed` rejects them, symlinks included.
- The EMSR927 evaluation scans every pipeline provenance / manifest for references
  to reference data before computing anything, and refuses on any hit. It also hashes
  the BATO files it reads before and after the run.
- Pipeline packages may not import `bato.evaluation` (checked by a test).
- `DataSource` lists only permitted sources, so evidence cannot cite a damage map.
- The OSM snapshot date is derived from each run's event date, never a fixed date.

## Repository layout

```
bato/
├── config/            config.yaml (settings) · thresholds.yaml (every value provisional + rationale) · test_aois.yaml
├── data/              raw · interim · processed · reference (EVAL ONLY) · eval (EVAL ONLY)
├── docs/              architecture · connectivity · baseline_detection · dashboard · limitations · evaluation/emsr927_protocol (FROZEN) · …
├── src/bato/
│   ├── config.py      strict config loading + fingerprint
│   ├── logging_utils.py  stage() context manager: inputs, outputs, timing, warnings, fallbacks, failures
│   ├── exceptions.py · data_rules.py · attribution.py
│   ├── contracts/     typed data contracts A–G + JSON io
│   ├── utils/geo.py   AOI ring validation, spherical area
│   ├── acquisition/   catalogue search (2A) · temporal-set selection (2B) · GRD/DEM fetching (2C)
│   ├── preprocessing/ SAFE inspection · SNAP calibration · Lee Sigma · terrain correction · stack
│   ├── detection/     Step 4C baseline + terrain/geometry gates (official flood layer)
│   ├── osm/ · impact/ · connectivity/   pre-event OSM · road states (rule v1 official) · graphs, access, ranking
│   ├── dashboard/     bundle builder for the official v1 run
│   ├── evaluation/    EVALUATION ONLY: EMSR927 acquisition, inspection, frozen protocol, metrics
│   ├── inference/     FloodViT (preliminary, not official) · debris/  rejected experiment
│   └── visibility/ · evidence/   empty packages (gating lives in detection/, evidence in contracts + connectivity)
├── scripts/           one script per step (see below) · *_emsr927.py = evaluation only
├── tests/             unit/ · integration/ · fixtures/
├── models/ · reports/ (no trained model; reports are generated, not committed)
├── dashboard/app/     web dashboard (static; scripts/serve_dashboard.py)
```

The package lives under `src/bato/` (not loose modules in `src/`), so it
imports as `bato` everywhere and cannot accidentally import from the working
directory.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/check_config.py     # validates config, prints fingerprint + provisional thresholds
```

### Sentinel-1 catalogue search (Step 2A)

Needs internet access to `catalogue.dataspace.copernicus.eu`; no login.

```bash
python scripts/s1_catalogue_query.py --preset trishuli_2026
python scripts/s1_catalogue_query.py --preset valencia_2024            # outside Nepal
python scripts/s1_catalogue_query.py --bbox 85.17 27.97 85.42 28.28 --event-date 2026-08-26 --save out.json
```

Prints every returned scene with platform, time, orbit direction, relative orbit,
polarisations, AOI coverage, days from the event and status (with reason if
rejected), then accepted scenes grouped by track. See
[docs/catalogue_discovery.md](docs/catalogue_discovery.md).

### Temporal-set selection (Step 2B)

```bash
python scripts/s1_catalogue_query.py --preset trishuli_2026 --save trishuli_s1.json
python scripts/select_temporal_sets.py --catalogue trishuli_s1.json --save trishuli_sets.json
```

Prints, per track, ranked PRE-1 / PRE-2 / POST sets with post gap, pre
separation, coverage, score and score components, multi-slice flag and quality
flags. Scores are a provisional PAIRING HEURISTIC. See
[docs/pair_selection.md](docs/pair_selection.md).

### Data fetching (Step 2C)

```bash
# dry run (default): plan, sizes, DEM tiles - no credentials needed
python scripts/fetch_s1_grd.py --selection trishuli_sets.json --catalogue trishuli_s1.json
# real download (needs CDSE_USERNAME / CDSE_PASSWORD in the environment)
python scripts/fetch_s1_grd.py --selection trishuli_sets.json --catalogue trishuli_s1.json --track ASCENDING/85 --download
```

Products go to `data/raw/sentinel1/`, DEM tiles to `data/raw/copernicus_dem/`,
each with a `.provenance.json`; the run manifest to `data/raw/manifests/`.
Re-running resumes partial files and skips verified ones. See
[docs/data_fetching.md](docs/data_fetching.md).

### Radiometric preprocessing (Step 3.2)

Needs ESA SNAP (gpt) and `pip install -e .[preprocessing]` (rasterio).

```bash
python scripts/preprocess_s1.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json \
    --pass-id S1D-A085-20260629T122138 --stage radiometric
```

Output: `data/interim/preprocessing/<aoi>/<pass>/radiometric/` (BEAM-DIMAP linear
σ⁰ VV/VH in radar geometry, the executed graph, gpt log and
`preprocessing_provenance.json`). See [docs/preprocessing.md](docs/preprocessing.md).

### Lee Sigma speckle filter (Step 3.3)

```bash
python scripts/speckle_s1.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --pass-id S1D-A085-20260629T122138
```

Reads the published `radiometric/` output (never modified) and writes
`data/interim/preprocessing/<aoi>/<pass>/speckle/`.

### Terrain correction (Step 3.4)

```bash
python scripts/terrain_correct_s1.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --pass-id S1D-A085-20260629T122138
```

Reads the published `speckle/` output and the Step 2C Copernicus DEM tiles (never modified) and writes
`data/interim/preprocessing/<aoi>/<pass>/terrain/` (EPSG:3857, 10 m, linear σ⁰, plus the DEM mosaic used).

### Temporal stack (Step 3.5)

```bash
python scripts/stack_s1.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json
```

Needs the `terrain/` outputs of all three passes of the candidate (PRE-1, PRE-2, POST); writes
`data/interim/preprocessing/<aoi>/stack/<candidate_id>/` (6-band linear σ⁰ GeoTIFF + validity mask).

### SAR change flood baseline (Step 4C)

```bash
python scripts/run_baseline_detection.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json
```

Writes `data/interim/detection/<aoi>/<candidate_id>/baseline/` (state + flood rasters, provenance, diagnostics). NOT_ASSESSED (steep / shadow / layover / invalid) is never reported as no flood ([docs/baseline_detection.md](docs/baseline_detection.md)).

### Pre-event OSM and road connectivity (Milestones 6–8)

```bash
python scripts/fetch_osm_snapshot.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --dry-run
python scripts/fetch_osm_snapshot.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json
python scripts/run_connectivity.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --road-rule v1   # OFFICIAL rule
```

**Always pass `--road-rule v1`.** The script's code default is `v2`, the rejected experiment, left unchanged in
code. A run without the flag is not the official result, and the dashboard refuses to build from it. On Windows:
`python scripts\run_connectivity.py --manifest data\raw\manifests\<aoi>_fetch_manifest.json --road-rule v1`.

*Experimental, rejected, not official:*
`python scripts/run_debris_detection.py --manifest …` writes `data/interim/detection/<aoi>/<candidate_id>/debris/`,
and `--road-rule v2` reads it. The detector failed its pre-registered criterion (d) on the matched no-event control
([docs/debris_detection.md](docs/debris_detection.md)). Its only use in the release is the methods panel of the
dashboard (`--debris-control-dir`).

The first writes `data/raw/osm/<aoi>/snapshot-<event − 1 day>/` (ohsome API v2 GeoParquet; needs `pyarrow` and an API key in the `OHSOME_API_KEY` environment variable, free at https://account.heigit.org/signup; `--dry-run` needs neither). The second writes `data/processed/connectivity/<aoi>/<candidate_id>/`: road segments, settlements, ranked blocked stretches, evidence records and a report. UNOBSERVABLE roads are never called open; building counts are a proxy, never population ([docs/connectivity.md](docs/connectivity.md)).

### Dashboard

```bash
python scripts/build_dashboard.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --title <name>
python scripts/serve_dashboard.py --open
```

Builds `data/processed/dashboard/<aoi>/<candidate_id>/` from the official road-rule v1 connectivity run (hashes verified, summary recomputed, v2 refused) and serves it on http://127.0.0.1:8765/ ([docs/dashboard.md](docs/dashboard.md)). It runs offline: the base map is local (DEM hillshade), and no online tile service or API key is used. The dashboard shows no EMSR927 data.

### FloodViT inference (Step 4B): preliminary, not part of the official result

```bash
python scripts/run_floodvit_inference.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --checkpoint <kurosiwo_vit_decoder_weights.pt> --kuro-repo <KuroSiwo checkout> --mode both
```

Writes `data/interim/inference/<aoi>/<candidate_id>/floodvit/` and `.../floodvit_no_event/` (prediction, uncalibrated flood probability, provenance). Needs torch + einops ([docs/inference.md](docs/inference.md)).

### Model-input manifest (Step 4A)

```bash
python scripts/prepare_model_input.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json
```

Checks the published stack against the Kuro Siwo-compatible model-input contract ([docs/model_input.md](docs/model_input.md)) and writes
`data/interim/preprocessing/<aoi>/model_input/<candidate_id>/model_input_manifest.json` (spec, hashes, 224 × 224 tile index). No tensors are written, no model is trained.

Copy `.env.example` to `.env` for downloads (Step 2C): Copernicus Data Space
credentials `CDSE_USERNAME`, `CDSE_PASSWORD` (and `CDSE_TOTP` with 2FA).
Catalogue search and dry runs do not need them. Never commit `.env`.

## Reproducing the official result (Trishuli)

Production inputs only: Sentinel-1, Copernicus DEM, pre-event OSM. Run from the repository root. Pass ids and the
candidate come from the fetch manifest.

```bash
python scripts/s1_catalogue_query.py --preset trishuli_2026 --save trishuli_s1.json
python scripts/select_temporal_sets.py --catalogue trishuli_s1.json --save trishuli_sets.json
python scripts/fetch_s1_grd.py --selection trishuli_sets.json --catalogue trishuli_s1.json --track ASCENDING/85 --download
M=data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json
# for each of the three passes (PRE-1, PRE-2, POST) listed in $M:
python scripts/preprocess_s1.py --manifest $M --pass-id <pass-id> --stage radiometric
python scripts/speckle_s1.py --manifest $M --pass-id <pass-id>
python scripts/terrain_correct_s1.py --manifest $M --pass-id <pass-id>
# then once:
python scripts/stack_s1.py --manifest $M
python scripts/run_baseline_detection.py --manifest $M
python scripts/fetch_osm_snapshot.py --manifest $M
python scripts/run_connectivity.py --manifest $M --road-rule v1        # OFFICIAL rule: always explicit
python scripts/build_dashboard.py --manifest $M --title Trishuli
python scripts/serve_dashboard.py --open
```

Windows: the same commands with `scripts\…` and `data\raw\manifests\…` paths. In particular:
`python scripts\run_connectivity.py --manifest data\raw\manifests\aoi-e893707c2beb_fetch_manifest.json --road-rule v1`.

Downloads need CDSE credentials; OSM needs `OHSOME_API_KEY`; Steps 3.2–3.4 need ESA SNAP. Re-running a published
step needs `--overwrite`, which keeps the previous output as `.superseded-<time>`.

## EMSR927 evaluation (evaluation only, separate from the pipeline)

Run this only **after** the official result exists. It never changes BATO outputs, and BATO never reads its data.

```bash
python scripts/fetch_emsr927.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json
python scripts/inspect_emsr927.py
python scripts/evaluate_emsr927.py --manifest data/reference/EMSR927/acquisition_manifest.json
```

- **What each script does:**
  - `fetch_emsr927.py` saves the official activation record and downloads only the final products that overlap
    the BATO AOI, from their official URLs. Everything goes to `data/reference/EMSR927/`, with URL, time, size and
    SHA-256 recorded.
  - `inspect_emsr927.py` describes the products and computes no metric.
  - `evaluate_emsr927.py` refuses to run unless the protocol is frozen and unchanged, the leakage scan has 0 hits,
    and every zip and BATO file matches its hash. Results go to `data/eval/EMSR927/<protocol-sha12>/metrics.{json,md}`.
- **Protocol:** `bato-emsr927-v1`, frozen 2026-10-05T08:41:38+00:00
  ([docs/evaluation/emsr927_protocol.md](docs/evaluation/emsr927_protocol.md)).
- **How to read the results:**
  - **Primary comparison:** road damage, BATO BLOCKED vs EMSR927 road DESTROYED, at segment level over the
    **matched domain** (BATO segments that receive a valid EMS reference label). Its recall is **not network-wide
    recall**; EMS damaged road outside that domain is reported separately as reference coverage.
  - **Supplementary comparison:** BATO BLOCKED vs DESTROYED + DAMAGED + POSSIBLY DAMAGED, labelled "not accuracy,
    not ground truth".
  - **Flood precision / recall / F1 / IoU are NOT CALCULABLE.** The products contain no open-water flood class:
    the observed event is a *Landslide* (6-Mass Movement), and it is **never mapped to flood**. Its overlap with
    BATO FLOOD is reported only as a descriptive, class-mismatched footprint overlap.
  - **Bridges:** reference counts as context only, because BATO has no bridge-damage prediction.
  - **Not claimed:** building, facility and settlement-isolation accuracy.
  - This is a case-study comparison on one event, not a validation.

## Tests

```bash
pytest                 # all tests
pytest tests/unit -q   # unit tests only
ruff check src tests scripts
mypy src scripts
```

The tests cover configuration, validation, contracts, serialisation, data-rule
guards, logging, and Sentinel-1 catalogue discovery against a **synthetic TEST
FIXTURE** of catalogue records (no network). Step 3.2 tests use a **fake gpt**
and tiny synthetic DIMAP products, so they show the BATO plumbing works, not that
SNAP's processing is correct; real-SNAP checks are opt-in
(`BATO_RUN_SNAP_TESTS=1`).

## Important limitations

- **Step 4C is not validated.** Its thresholds are provisional (`validated: false`), and it detects only
  open-water-like darkening, not mud, debris or structural damage.
- **Most of the Trishuli scene is NOT_ASSESSED.** About 91.2 % of the scene (41,285,907 of 45,256,776 pixels) is
  not assessed. Of those not-assessed pixels (not of the whole scene), about 86.1 % come from the 15° slope gate,
  11.4 % from layover and 2.5 % from shadow. As a result, most road length is UNOBSERVABLE and most settlements are
  **access unverified**, not cut off.
  Unobservable is never reported as "no damage".
- **EMSR927 is a cross-hazard reference.** Flood metrics are NOT CALCULABLE, the Landslide footprint is not flood,
  and road metrics are matched-domain only (see above).
- **No impact prediction for buildings, bridges or facilities.** At ~10 m resolution BATO can say a bridge is
  **exposed** to the impact zone, never that it collapsed.
- **Debris was not detected.** The debris (disturbance) experiment was rejected; FloodViT is preliminary.
- **Timing.** Sentinel-1 revisits every few days, so BATO maps the state at the POST pass and cannot warn ahead
  of a sudden event.
- **OSM completeness.** Road access depends on how complete the pre-event OSM is.

Full list: [docs/limitations.md](docs/limitations.md).

## Required attribution

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors.
- Training data (when used): Kuro Siwo — Bountos et al., 2024; Sen1Floods11 — Bonafilia et al., CVPR Workshops 2020.
- Evaluation reference only: European Union, Copernicus Emergency Management Service data.

## Licence

Source code: MIT (see `LICENSE`). Data keeps its own licence: OSM data is ODbL,
and any published modified OSM data must remain ODbL.

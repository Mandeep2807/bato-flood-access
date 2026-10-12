# BATO architecture

Layers, strictly one-directional:

```
RAW DATA → PREPROCESSING → DETECTION → IMPACT → CONNECTIVITY → EVIDENCE → PRESENTATION
```

Every arrow is a typed contract (`src/bato/contracts/`). Presentation (dashboard,
situation report) reads **only** from the evidence store; it never computes or
invents numbers.

Status words used below: **IMPLEMENTED** = in the code and covered by tests · **NOT IMPLEMENTED** = not in
the code · **not validated** = implemented, but no accuracy has been measured.

## Release status

The submitted result is: Sentinel-1 temporal stack (PRE-1, PRE-2, POST) → Step 4C SAR-change flood baseline
(provisional thresholds, `validated: false`) → pre-event OSM → road rule **v1** road impact → three-graph
connectivity and settlement access states → evidence records → dashboard. It is compared, after the fact and as
evaluation only, with Copernicus EMS EMSR927 under the frozen protocol `bato-emsr927-v1`
([evaluation/emsr927_protocol.md](evaluation/emsr927_protocol.md)).

## Milestone status

| Milestone | Status |
|---|---|
| Step 1: foundation (config, contracts, guards, logging) | IMPLEMENTED |
| Step 2A: Sentinel-1 catalogue discovery | IMPLEMENTED (live CDSE queries run by the team) |
| Step 2B: same-track temporal-set selection | IMPLEMENTED |
| Step 2C: Sentinel-1 GRD + Copernicus DEM fetching | IMPLEMENTED |
| Steps 3.1–3.5: SAFE inspection, radiometric calibration (SNAP), Lee Sigma, terrain correction, temporal stack | IMPLEMENTED (single-slice passes only; slice assembly NOT IMPLEMENTED) |
| Step 4A: model-input contract | IMPLEMENTED |
| Step 4B: FloodViT inference (Kuro Siwo checkpoint) | IMPLEMENTED, preliminary, not validated; **not part of the official result** |
| Step 4C: SAR change flood baseline | IMPLEMENTED, not validated; **official flood layer** |
| SAR disturbance (debris) detector + road rule v2 | Experiment **rejected** under its pre-registered criteria; **not part of the official result** ([debris_detection.md](debris_detection.md)) |
| Pre-event OSM, road impact (rule v1), connectivity, settlement access, stretch ranking, evidence records | IMPLEMENTED, not validated ([connectivity.md](connectivity.md)) |
| Dashboard of the official v1 result | IMPLEMENTED ([dashboard.md](dashboard.md)) |
| EMSR927 case-study comparison (evaluation only) | IMPLEMENTED; protocol frozen ([evaluation/emsr927_protocol.md](evaluation/emsr927_protocol.md)) |
| Sentinel-2, flood-model training, fusion, building / bridge / facility impact, situation report, Kuro Siwo test and Himalayan-holdout evaluation | NOT IMPLEMENTED |

## Pipeline

| # | Stage | Package | Input → Output contract | Status |
|---|---|---|---|---|
| 0 | AOI / event request | `contracts.aoi` | `AOIRequest` → `AOI` | IMPLEMENTED (validation, normalisation, limits) |
| 1 | Data preflight | `acquisition` | `AOI` → availability report | Sentinel-1 catalogue: IMPLEMENTED; S2 / OSM-date / ETA preflight: NOT IMPLEMENTED |
| 2 | Sentinel-1 acquisition | `acquisition` | selection → `S1FetchPlan` → raw SAFE zips + `DownloadProvenance` + `FetchManifest` | IMPLEMENTED |
| 3 | Same-track pairing | `acquisition` | `CatalogueSearchResult` → `TemporalSelectionResult` (PRE-1 + PRE-2 + POST per track) | IMPLEMENTED |
| 4 | SAR preprocessing | `preprocessing` | SAFE → calibrated, filtered, terrain-corrected EPSG:3857 10 m → 6-band stack | IMPLEMENTED (single-slice passes) |
| 5 | Visibility | `detection.terrain_gates` | DEM + geometry → layover / shadow / steep gates | IMPLEMENTED inside Step 4C as approximate gates (the separate `visibility` package is empty) |
| 6 | Flood detection | `detection` (Step 4C), `inference` (FloodViT) | stack → state raster (FLOOD / NO_FLOOD / PRE_EVENT_WATER / NOT_ASSESSED reasons) | Step 4C: IMPLEMENTED, official, not validated; FloodViT: preliminary, not official |
| 7 | Debris / change detection | `debris` | stack + Step 4C state → `debris_state.tif` | Experiment **rejected**; not used in the official result |
| 8 | Fusion | `detection` | detection layers (+ S2) → `DetectionRaster` | NOT IMPLEMENTED (the official result uses the Step 4C state directly) |
| 9 | OSM extraction | `osm` | AOI + event date → pre-event roads, places, hospitals, buildings | IMPLEMENTED (snapshot = event − 1 day; post-event edits refused) |
| 10 | Road / building impact | `impact` | Step 4C state + OSM → road states | Roads: IMPLEMENTED (OPEN / BLOCKED / UNOBSERVABLE, rule v1 official); buildings, bridges (structural), facilities: NOT IMPLEMENTED |
| 11 | Pre/post connectivity | `connectivity` | road states → 3 graphs | IMPLEMENTED |
| 12 | Settlement access | `connectivity` | graphs → access state (CONNECTED / POSSIBLE_CUTOFF / CONFIRMED_CUTOFF / NO_PRE_EVENT_ACCESS) | IMPLEMENTED |
| 13 | Repair priority | `connectivity` | blocked stretches → ranking (union-find + 0-1 BFS) | IMPLEMENTED |
| 14 | Evidence | `contracts.evidence` (written by `connectivity`) | every road / settlement / stretch claim → `EvidenceRecord` | IMPLEMENTED (`evidence_records.jsonl` per run; the separate `evidence` package is empty) |
| 15 | Dashboard + situation report | `dashboard/`, `bato.dashboard`, `reports/` | official v1 outputs → verified bundle → UI | Dashboard: IMPLEMENTED; situation report: NOT IMPLEMENTED |
| E | Evaluation (evaluation only) | `evaluation` | official outputs + EMSR927 reference → metrics | EMSR927 comparison: IMPLEMENTED (frozen protocol); Kuro Siwo / Himalayan holdout: NOT IMPLEMENTED |

Cross-cutting, IMPLEMENTED: configuration + fingerprint (`config.py`),
stage logging (`logging_utils.py`), exceptions, data-rule guards
(`data_rules.py`), attribution text (`attribution.py`).

## Step 2A — how catalogue discovery works

Full detail: [catalogue_discovery.md](catalogue_discovery.md).

1. **Window** from config: 60 days before to 24 days after the event (whole UTC days).
2. **Query** the CDSE OData catalogue with documented operators only: collection `SENTINEL-1`, AOI intersection, sensing-start window. Paginated, capped, no login needed.
3. **Parse** each record strictly. Metadata is cross-checked against the official ESA product name. Orbit direction and relative orbit must come from catalogue attributes and are never derived.
4. **Classify** each record, first failing rule wins: invalid metadata → duplicate → mode (IW) → product type (level-1 GRD, class H) → orbit → polarisation (VV+VH) → date (window, same-day exclusion) → coverage (pass covers ≥ 80 % of the AOI). Every record keeps its status and reason; nothing is silently dropped.
5. **Group** accepted scenes by (orbit direction, relative orbit). Groups are pairing candidates; no pair is chosen.

*(Historical note: at the end of Step 2A, pair selection, downloads and everything downstream were not yet built; see the status tables above for today.)*

## Step 2B — how temporal-set selection works

Full detail: [pair_selection.md](pair_selection.md).

1. Accepted scenes are merged into **AcquisitionPasses** (slices of one overpass = one observation, union coverage, `multi_slice` flag).
2. Per track (orbit direction + relative orbit), every post pass × every pair of pre passes ≥ `min_pre_separation_days` apart becomes a candidate (PRE-1 earlier, PRE-2 recent, POST); a flagged 1-pre fallback exists.
3. Each candidate gets a **PAIRING HEURISTIC** score with visible components, is validated as a `PrePostPair`, and is ranked with a total, deterministic tie-breaker.
4. Every viable track keeps its own top-N; no global track choice is made.

*(Historical note: at the end of Step 2B, fetching and everything downstream were not yet built; mosaicking multi-slice passes is still not implemented.)*

## Step 2C — how data fetching works

Full detail: [data_fetching.md](data_fetching.md).

1. `S1FetchPlan` from the validated 2B selection and its 2A catalogue (checked by digest): chosen candidates → passes → **every slice**, with CDSE product ids from the catalogue; IW GRDH + VV/VH enforced.
2. `DEMPlan`: Copernicus GLO-30 tiles covering AOI + `dem.buffer_km`, found by the catalogue `dataset` attribute.
3. Dry run by default; `--download` streams each product with resume, retries, redirect allow-list, size and checksum verification, atomic rename, and a disk-space check; then checks the archive structure (VV/VH measurement + calibration + noise) and writes a provenance sidecar.
4. `FetchManifest`: per-file status, per-pass completeness (all slices), per-candidate readiness.

*(Historical note: at the end of Step 2C, pixel processing and everything downstream were not yet built; see the status tables above for today.)*

## Key design decisions

**Same-track comparison only.** `PrePostPair` cannot be built from acquisitions
with different (orbit direction, relative orbit). Different tracks are fused
later at the *decision* level, never pixel by pixel. `require_same_relative_orbit`
in config accepts only `true`.

**Same-day ambiguity.** Without an event time, an acquisition on the event date
could be before or after the event, so it is excluded from both roles. With
`event_time_utc`, ordering is checked exactly.

**Unobservable is a class, not "no change".** `ImpactClass.UNOBSERVABLE` (254)
marks shadow/layover/no-data pixels. Road segments overlapping them become
`UNOBSERVABLE`, and settlements depending on them become `possibly_cut_off`.

**Three graphs, one decision table.** From the same pre-event network:
pre-event; post-optimistic (BLOCKED removed); post-pessimistic (BLOCKED and
UNOBSERVABLE removed). `classify_settlement` implements:

| Pre | Post-optimistic | Post-pessimistic | Status |
|---|---|---|---|
| unreachable | – | – | `no_pre_event_road_access` (not a flood effect) |
| reachable | unreachable | – | `confirmed_cut_off` |
| reachable | reachable | unreachable | `possibly_cut_off` |
| reachable | reachable | reachable, route ≥ ratio × pre route | `degraded` |
| reachable | reachable | reachable | `connected` |

`SettlementStatus` rejects any status that contradicts this table, and any
reachability results that break monotonicity (removing roads can neither
create a route nor shorten one).

**Targets** are hospitals, towns, and *exit nodes* where roads leave the
buffered analysis area (`aoi.analysis_buffer_km`), so the AOI edge cannot
create fake islands.

**No unsupported claims.** `BridgeExposure` has no "collapsed" value; evidence
claims mentioning collapse, casualties or population are rejected; building
counts are never called population.

**Reproducibility.** `BatoConfig.fingerprint()` hashes every setting that can
change a result (paths and logging excluded). AOI ids and evidence ids are
deterministic hashes of their inputs.

**Evaluation isolation.** Reference data lives in `data/reference` / `data/eval`.
Loaders must call `assert_pipeline_input_allowed`, and pipeline packages may
not import `bato.evaluation` (both tested). No threshold was tuned on EMSR927
(or on any reference map): the Step 4C and road thresholds are provisional values
fixed before the comparison. The EMSR927 evaluation reads the frozen official
outputs read-only, hashes them before and after, and refuses to run if any
pipeline provenance cites reference data.

## Planned fallbacks (for later milestones)

| Situation | Behaviour |
|---|---|
| No Sentinel-2 or cloudy | Sentinel-1 only; stated in the report |
| Only one pre-event image | Allowed, flagged `SINGLE_PRE_IMAGE`, confidence lowered |
| No same-track pair | No change detection; clear message, nearest available dates offered |
| Flood model unavailable or slow | Baseline change detection, recorded via `StageReport.use_fallback` |
| No hospital in range | Widen search → towns → exits; target type recorded |
| AOI too large | Rejected with a clear message (limit in config) |

Every fallback is logged through `stage()`; nothing fails silently.

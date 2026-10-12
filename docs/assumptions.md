# Assumptions and open questions

## Interpretations of the PS (ours, not stated by the organisers)

1. **Event date is a UTC date.** Acquisition times are UTC; Nepal is UTC+05:45, so a local-date event can straddle two UTC dates. Users should supply `event_time_utc` when known; otherwise same-day acquisitions are excluded.
2. **"Settlement"** = an OSM `place` feature, or a cluster of ≥ N OSM buildings (`connectivity.building_cluster_*`).
3. **"Nearest town or hospital"** = shortest road route to any OSM hospital or town. Exit nodes at the edge of the buffered analysis area also count as connections, and are labelled as such.
4. **"Cut off"** is reported in two strengths (confirmed / possibly), plus `degraded` and `no_pre_event_road_access`. The PS asks for "settlements that no longer have a road connection"; the extra categories make uncertainty explicit and avoid blaming the flood for roads that never existed.
5. **"What was damaged"** is reported as *impact/exposure* (inside the flood or debris zone), not structural damage.
6. **OSM snapshot** is chosen per run from the event date (offset ≥ 1 day). A fixed snapshot (e.g. 27 July 2026) is only valid for events after that date.
7. **Earliest event date** is 2014-10-03 (start of routine Sentinel-1A operations), a provisional configurable value.
8. **AOI limits** (1–1000 km², 20 × 20 km default box for a point) are provisional and will be revisited after runtime profiling.
9. **Catalogue source** is CDSE OData only (conservative assumption until Q2 is answered). Search needs no login.
10. **A satellite pass** = scenes on the same platform and track starting ≤ `pass_max_gap_s` (120 s) apart; coverage is judged per pass, so an AOI split across two adjacent slices is not rejected.
11. **GRD class H** (10 m) only; GRDM (40 m) is rejected as `rejected_product_type`.
12. **Attribute value formats** of the CDSE catalogue are partly undocumented. Known spellings are accepted and cross-checked against the ESA product name (see catalogue_discovery.md, OPEN items).
13. **Temporal-set shape** follows Kuro Siwo: 2 pre-event + 1 post-event images on one track. PRE-1 = earlier, PRE-2 = more recent (BATO's naming). Kuro Siwo's `sec1` is the LATER pre-event image and `sec2` the earlier (verified from the Kuro Siwo grid pickles, Step 4A): PRE-2 → sec1, PRE-1 → sec2 (docs/model_input.md).
14. **Pairing parameters** (`config.yaml → pairing`) are a provisional PAIRING HEURISTIC, not validated science: min pre separation 12 d, target 24 d, weights 0.35/0.20/0.15/0.20/0.10, penalties as documented. They are never tuned on reference maps.
15. **No global track choice** in 2B: every viable track keeps its candidates until preprocessing/model validation compares them.
16. **Days from event** are measured from 00:00 UTC of the event date when the event time is unknown.
17. **Raw products are stored as delivered** (SAFE zip). Calibration to linear σ⁰, terrain correction and everything else happen in preprocessing; this keeps the Kuro Siwo-compatible path (σ⁰ from the calibration LUT, no dB) fully open.
18. **Default fetch scope** is rank 1 per track; other ranks are opt-in. Downloads are sequential.
19. **DEM dataset** `COP-DEM_GLO-30-DGED/2024_1` and the DEM buffer (20 km) are provisional; the release name is OPEN until a live search confirms it.
20. **Download endpoint** `download.dataspace.copernicus.eu/odata/v1/Products(<id>)/$value` with redirects allowed only within `*.dataspace.copernicus.eu` (OPEN details in data_fetching.md).

## Engineering assumptions

- Area is computed on a sphere (R = 6371.0088 km): accurate to well under 1 % against the WGS84 ellipsoid, enough for size limits. Exact geodesy uses pyproj from later milestones.
- Polygons with holes or crossing the antimeridian are rejected (not needed for Himalayan valleys).
- `layover_lia_max_deg` (local incidence angle) gives only an *approximate* layover mask. Shadow comes from the provider's shadow mask (CDSE Sentinel Hub returns `shadowMask` only with radiometric terrain correction).
- Code licence MIT is a placeholder chosen by us; change it if the team prefers.

## Deviations from the requested repository layout

- `src/bato/` package instead of loose modules in `src/`: imports work identically from tests, scripts and notebooks (`import bato`), and nothing imports by accident from the working directory.
- Added `src/bato/contracts/` (all data contracts in one place), `data_rules.py`, `attribution.py`, `exceptions.py`.
- Added `tests/helpers.py` for metadata builders shared by tests.

## Open questions for the organisers (ask in writing before relying on them)

| # | Question | What depends on it |
|---|---|---|
| Q1 | Are ImageNet-pretrained backbones, or the Kuro Siwo authors' released weights, allowed? | Model milestone |
| Q2 | Must Sentinel data come from CDSE, or is Google Earth Engine acceptable? (GEE's S1 is terrain-corrected with SRTM, not Copernicus DEM.) | Acquisition milestone |
| Q3 | May we hand-label Sentinel-1/2 imagery of other past events for evaluation only? | Himalayan holdout |
| Q4 | May we use unlabeled Sentinel-1 scenes for self-training? | Optional model work |
| Q5 | Live evaluation: who runs the system, is there internet, and what are the time/AOI limits? | Live mode, `aoi.max_area_km2` |
| Q6 | If ohsome is unavailable, is Overpass with a historical date query acceptable? | OSM milestone |

Until answered, the conservative option is assumed: CDSE only, no unlabeled self-training, and no team labels used for training.

# Sentinel-1 catalogue discovery (STEP 2A)

```
AOI + event date ─► search window ─► CDSE OData search ─► parse + classify every record
                 ─► CatalogueSearchResult (all records, each with a status) ─► groups by track
```

Entry point: `bato.acquisition.s1_catalogue.search_sentinel1(aoi, cfg)`.
CLI: `python scripts/s1_catalogue_query.py --preset trishuli_2026` (or `--bbox …/--point … --event-date …`).

No pixels are downloaded. No pre/post pair is selected.

## Source and access

- **Provider:** Copernicus Data Space Ecosystem OData catalogue, `https://catalogue.dataspace.copernicus.eu/odata/v1/Products` (`config.yaml → cdse.odata_url`). No other source is used.
- **Authentication:** none for search. CDSE documents product search as public. `bato.acquisition.cdse_auth` implements the official token request (Keycloak, `client_id=cdse-public`, `grant_type=password`, optional `totp`) for the download milestone. Credentials come only from `CDSE_USERNAME` / `CDSE_PASSWORD` (/ `CDSE_TOTP`), and are never logged, put in error messages, or shown in `repr`.
- **Transport:** Python standard library (`urllib`), honouring `HTTPS_PROXY` / `SSL_CERT_FILE`. Tests inject a fake transport. Production code contains no fake responses.

## Search window

`[event_date − pre_window_days, event_date + post_window_days + 1 day)` in whole UTC days (config: 60 and 24 days).

## Server-side query (documented operators only)

```
Collection/Name eq 'SENTINEL-1'
and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((<AOI ring, counter-clockwise>))')
and ContentDate/Start ge <window start> and ContentDate/Start lt <window end>
$orderby=ContentDate/Start asc  $top=<page_size>  $expand=Attributes
```

Pagination follows `@odata.nextLink`, or `$skip` when a page is full. A hard cap (`cdse.max_records`) sets `truncated = true` and logs a warning; it never truncates silently.

Mode, product type and polarisation are deliberately **not** filtered on the server. Their attribute value formats are not fully documented, and client-side filtering keeps every excluded scene visible with a reason.

## Metadata mapping (CDSE record → `Acquisition`)

| Acquisition field | Source | If missing / inconsistent |
|---|---|---|
| `source_id` | `Name` (ESA product name) | missing or malformed name → invalid metadata |
| (CDSE id) | `Id` → `CandidateRecord.product_id` (needed for download) | missing → invalid metadata |
| `sensor` | Sentinel-1 (collection + name prefix `S1x`) | — |
| `platform` | attribute `platformSerialIdentifier` (`A` / `S1A` / `SENTINEL-1A` → `S1A`), cross-checked with name | absent → from name; disagreement → invalid |
| `acquisition_start` | `ContentDate/Start` (UTC), must match name start within 1 s | missing/unparseable/mismatch → invalid |
| `product_type` | attribute `productType` (`IW_GRDH_1S`, `IW_GRDH_1S-COG` or `GRD` accepted), cross-checked with name | absent → from name; disagreement → invalid |
| `orbit_direction` | attribute `orbitDirection` | **required** (not in the name) → invalid if missing |
| `relative_orbit` | attribute `relativeOrbitNumber` (integer) | **required** → invalid if missing; out of 1–175 → `rejected_orbit` |
| `polarizations` | attribute `polarisationChannels` (`VV&VH`, `VV VH`, `VV,VH` …), cross-checked with name code (DV/SV/DH/SH) | absent → from name; disagreement → invalid |
| `footprint` | `GeoFootprint` (GeoJSON Polygon, or single-part MultiPolygon); fallback WKT `Footprint` | missing, multi-part or self-intersecting → invalid |
| `processing_status` | `catalogued` | — |

The product name is an official ESA identifier. Using it for fields it encodes (platform, mode, type, polarisation, start time) is parsing, not guessing. Orbit direction and relative orbit are **never** derived. `CandidateRecord.metadata_sources` records whether each field came from an attribute or the name.

## Classification (first failing rule wins)

| Order | Status | Rule (configuration key) |
|---|---|---|
| 1 | `rejected_invalid_metadata` | metadata missing, malformed or inconsistent (table above) |
| 2 | `rejected_duplicate` | same platform/mode/type/polarisation/start/stop as another record; keep non-COG, then latest `PublicationDate`, then smallest id |
| 3 | `rejected_mode` | beam mode ≠ `sentinel1.acquisition_mode` (IW) |
| 4 | `rejected_product_type` | not level-1 standard `GRD` with resolution class `sentinel1.grd_resolution_class` (H) |
| 5 | `rejected_orbit` | orbit direction not ascending/descending, or relative orbit outside 1–175 |
| 6 | `rejected_polarization` | missing any of `sentinel1.polarizations` (VV, VH) |
| 7 | `rejected_date` | outside the window; or sensed on the event date with no event time (same-day policy: exclude); or exactly at the event time |
| 8 | `rejected_coverage` | footprint does not overlap the AOI, or its **pass** covers < `sentinel1.min_aoi_coverage_fraction` (0.8) |
| — | `accepted` | role `pre_event` / `post_event` set from the event date (or event time) |

With an `event_time_utc`, pre/post is decided by exact time and same-day images are usable.

**Coverage.** This is the area-weighted fraction of the AOI inside the footprint, measured on a `coverage_grid_size`² grid of sample points (cos-latitude weighted). Both the scene's own fraction and the **pass** fraction are stored. A pass is a chain of scenes on the same platform and track starting ≤ `pass_max_gap_s` apart (consecutive slices of one overpass). An AOI split across two slices is therefore accepted when the slices together cover it. Rejected scenes keep their measured fraction.

## Orbit grouping

Accepted scenes are grouped by **(orbit direction, relative orbit)**: ascending 19 and descending 19 are different groups. S1A and S1C on the same relative orbit share a group, because they fly the same ground track. Each `TrackGroup` lists pre-event and post-event scene names (oldest first) and pass ids, plus `has_pre_and_post`. Groups are candidates only. `PrePostPair` remains the final enforcement of the same-track rule, and `CatalogueSearchResult` itself rejects a group containing a scene from another track.

## Answering "why was this scene not selected?"

`result.explain("<product name, name without .SAFE, or CDSE id>")` returns the status and reason. The full result (all records) can be saved as JSON with `--save`.

## Logged per search (stage `s1_catalogue_search`)

Provider, AOI id, bbox, area, event date, window, number returned / accepted / rejected, counts per status, pages, tracks, tracks with pre+post, runtime, warnings (truncation, nothing accepted), and provider errors (logged with traceback, then re-raised). No credentials or tokens.

## OPEN items (to verify on the first live query)

1. Exact attribute value formats: `polarisationChannels`, `productType`, `platformSerialIdentifier`. The parser accepts the known spellings and cross-checks them with the name; an unexpected spelling shows up as `rejected_invalid_metadata` with the exact value in the reason.
2. Whether COG variants appear as separate records for Sentinel-1 GRD (handled by duplicate detection either way).
3. `GeoFootprint` presence for all Sentinel-1 records (WKT fallback exists).
4. Live query status: **executed by the team** (reported: Trishuli 98 returned / 14 accepted; Valencia 259 / 42). Attribute-format observations from those runs should be recorded here. Run both presets and keep the saved JSON as evidence:
   `python scripts/s1_catalogue_query.py --preset trishuli_2026 --save data/interim/catalogue/trishuli_s1.json`
   `python scripts/s1_catalogue_query.py --preset valencia_2024 --save data/interim/catalogue/valencia_s1.json`

## Known limitations

- `PrePostPair` holds a single post-event `Acquisition`. When the AOI needs two slices of one pass, pairing (Step 2B) must handle a pass-level mosaic. This is a contract change to decide in 2B.
- Footprints crossing the antimeridian (multi-part) are rejected as unsupported.
- No retries on provider errors: a failed search fails loudly and can be re-run.

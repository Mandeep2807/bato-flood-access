# Sentinel-1 GRD and Copernicus DEM fetching (STEP 2C)

```
trishuli_sets.json (2B selection) + trishuli_s1.json (2A catalogue)
  → S1FetchPlan        candidates → passes → every slice (CDSE product id from the catalogue)
  → DEMPlan            1x1° GLO-30 tiles covering AOI + dem.buffer_km (public catalogue search)
  → dry run (default)  table + <raw>/manifests/<aoi>_fetch_plan.json
  → --download         resumable, verified downloads → <raw>/sentinel1, <raw>/copernicus_dem
                       + <file>.provenance.json per file + <raw>/manifests/<aoi>_fetch_manifest.json
```

CLI: `python scripts/fetch_s1_grd.py --selection … --catalogue … [--track LABEL] [--rank N] [--max-scenes N] [--no-dem] [--offline] [--download]`

Code: `bato.acquisition.s1_fetch` (plan, execution), `downloader` (resumable HTTP), `dem` (tiles), `safe_inspect` (archive checks), `cdse_auth.TokenProvider`, `cdse_odata.fetch_product_metadata`, `http.UrllibStreamingTransport`.

Nothing in this step processes pixels. **Not implemented:** terrain correction, calibration to σ⁰, speckle filtering, dB conversion, normalisation, mosaicking, or anything downstream.

## What is fetched

- **Candidates:** by default rank 1 on every track (`--track`, `--rank` narrow or widen this). The Step 2B heuristic and its outputs are used as they are; nothing is re-ranked.
- **The plan is tied to its inputs:** it refuses to run unless SHA-256(catalogue) equals the `catalogue_digest` recorded in the selection. This stops a selection being combined with a different catalogue.
- **Every slice of every pass:** a two-slice pass yields two products with the same `pass_id`, `slice_index` 0 and 1, and `slice_count` 2.
  - Slices are stored as separate files and are never counted as separate dates.
  - In the manifest, a pass is `complete` only when **all** its slices are present, and a candidate is `ready` only when its PRE-1, PRE-2 and POST passes are all complete.
  - With `--max-scenes 1` on a two-slice pass, the pass is reported as incomplete.
- **Kuro Siwo compatibility is checked before download:** every product must be `IW_GRDH_1S` with all configured polarisations (VV + VH), or planning fails.
- **The structure is checked after download:**
  - the SAFE zip must contain `manifest.safe`, plus for VV and VH the measurement GeoTIFF, `calibration-*.xml` (the σ⁰ table) and `noise-*.xml`;
  - these are exactly what preprocessing needs to produce **linear σ⁰** the way Kuro Siwo did (SNAP Calibration → σ⁰, with no dB conversion);
  - the measurement rasters are digital numbers, not σ⁰; nothing is converted here;
  - a file that fails the check is renamed `*.rejected`, so it is never mistaken for valid data.
- Products are identified by the CDSE product id from the catalogue. Before each download, the public `Products(<id>)` record is re-read for name, size, checksum and online status. If the name changed, or the product is offline (long-term archive), that file fails with a clear message. Ordering offline products is not supported.

## Download behaviour (`downloader.py`)

| Concern | Behaviour |
|---|---|
| Endpoint | `download.base_url` + `(<id>)/$value`, header `Authorization: Bearer <token>` |
| Redirects | followed only to **https** hosts ending in `download.allowed_redirect_suffixes` (`dataspace.copernicus.eu`); the token is kept on those redirects; any other redirect is refused and not retried |
| Atomicity | written to `<file>.part`, renamed only after size and checksum checks |
| Resume | `Range: bytes=<n>-`; 206 appends after checking `Content-Range`; 200 restarts from zero; 416 handled |
| Retries | network errors, dropped connections, 429, 500/502/503/504: exponential backoff (`backoff_initial_s` … `backoff_max_s`), `Retry-After` honoured, up to `max_attempts` |
| Auth | 401 → token refreshed (at most twice); 403 / 404 / other → immediate, clear failure |
| Integrity | size = `ContentLength`; checksum verified with the first published algorithm we can compute (sha256, md5, sha3-256, blake3 if the optional `blake3` package is installed); SHA-256 of the local file always recorded |
| Idempotency | an existing file that verifies is skipped (`skipped_existing`) and keeps its original provenance; a file that does not verify is **never overwritten**; the run stops with "move it away" |
| Disk | free space must exceed remaining bytes × `min_free_disk_ratio` |
| Safety cap | plans with more than `max_scenes_per_run` products need an explicit `--max-scenes` |
| Failures | recorded per file in the manifest (`failed` + message); the run continues with other files; exit code 3 |

## Authentication

- **Dry runs and catalogue/metadata lookups** are public; no credentials are needed.
- **`--download`** needs `CDSE_USERNAME` and `CDSE_PASSWORD` (and `CDSE_TOTP` if 2FA is enabled) in the environment, e.g. `set -a; source .env; set +a`. They are used with the official token endpoint (`client_id=cdse-public`, password grant).
- **Token handling:** the token is fetched once before the first download (to fail fast) and refreshed when expired or rejected. It is never logged, written to provenance or manifests, or included in errors.

## Provenance (`<file>.provenance.json`, contract `DownloadProvenance`)

Every downloaded file records:
- **Identity:** `product_id`, `product_name`, `source` (`cdse-odata-download`), `source_url` (no query string, no credentials).
- **Integrity:** `content_length`, `server_checksums`, `checksum_verified_with` / `checksum_verified`, local `sha256`, `size_bytes`.
- **Where and when:** `local_path` (relative to the project root), `content_check`, `downloaded_at`, `attempts`, `resumed`, `bato_version`, `config_fingerprint`.
- **Data role:** `data_role = "pipeline_input_raw"`.
- **Sentinel-1 files also record:** `platform`, `acquisition_start`, `orbit_direction`, `relative_orbit`, `polarizations`, `product_type`, `pass_ids`, `candidate_ids`.
- **DEM tiles also record:** `tile_id`, `dem_dataset`.

## Data-rule guards

- Downloads may only be written under `paths.raw` and never inside `data/reference` or `data/eval` (`assert_raw_output_dir`, symlinks resolved), so a raw download can never sit among reference data.
- The reverse guard, `assert_reference_input_allowed`, makes evaluation code accept reference files **only** from the evaluation-only folders, so a raw download can never be loaded as a reference product.
- Inputs are only CDSE Sentinel-1 and Copernicus DEM products. No EMS, UNOSAT, population or other published products are used.

## Copernicus DEM (`dem.py`)

- **Search:** the public catalogue is queried with `Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'dataset' and att/OData.CSC.StringAttribute/Value eq '<dem.cdse_dataset>')` together with an intersection with the AOI box grown by `dem.buffer_km` (20 km).
- **Tiles:** required tiles are every 1×1° cell touched by that box (Trishuli: N27E084, N27E085, N28E084, N28E085; Valencia: N39W001). Products are matched to cells by the `N27_00_E085_00` pattern in the name, falling back to the footprint centre.
- **Missing cells** (e.g. open sea) are reported, never invented.
- **Download:** the same downloader writes to `<raw>/copernicus_dem` with provenance. The archive check accepts a GeoTIFF, or a zip/tar containing `*_DEM.tif`.
- The AOI box is read back from the Step 2A catalogue's recorded query polygon.

## OPEN items (verify on the first real download)

1. **Download host and redirect:** whether `download.dataspace.copernicus.eu` redirects (expected within `*.dataspace.copernicus.eu`) and whether `Range` is honoured. Both cases are handled.
2. **Checksums:** which algorithms CDSE publishes in `Checksum` for S1 GRD and DEM products. If only BLAKE3 is published, install `blake3` to verify; otherwise size is checked and SHA-256 recorded.
3. **DEM release and packaging:** whether `COP-DEM_GLO-30-DGED/2024_1` is the current release name, and how a `$value` DEM product is packaged.
4. **Account limits:** CDSE per-account download quotas and concurrency limits. Downloads are sequential.

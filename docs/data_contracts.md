# Data contracts

All contracts are immutable pydantic models (`bato.contracts.common.Contract`):
unknown fields are rejected, and validation runs on construction **and** on
loading from JSON (`bato.contracts.io.load_contract`). Computed fields are
written to JSON for readability and recomputed on load.

Conventions:
- Coordinates: WGS84, GeoJSON order `[lon, lat]`. Polygons have one exterior ring, no holes; antimeridian-crossing rings are rejected.
- Datetimes: timezone-aware **UTC** only. Event dates are UTC dates (see assumptions).
- Stored file paths: relative to the project root, POSIX style, no `..`.
- Fingerprints/hashes: lowercase SHA-256 hex.

## A. AOI request → AOI (`contracts/aoi.py`)

`AOIRequest`: `event_date`, optional `event_time_utc` (naive, read as UTC), **exactly one** of `polygon` / `bbox` / `point`, optional `point_half_size_km` (point only), `label`, `metadata`.

`build_aoi(request, settings)` → `AOI`: `aoi_id` (deterministic hash of ring + event), `geometry` (always a polygon), `bbox`, `area_km2` (spherical), `event_date`, `event_time_utc`, `input_kind`, `label`, `metadata`.

Rejected: event date in the future or before Sentinel-1 routine operations (`EventDateError`); area outside `aoi.min_area_km2..max_area_km2` (`AOIValidationError`); malformed, self-intersecting, unclosed, zero-area or out-of-range rings.

## B. Acquisition (`contracts/acquisition.py`)

`source_id`, `sensor` (S1/S2), `platform`, `acquisition_start` (UTC), `product_type`, `orbit_direction`, `relative_orbit`, `polarizations`, `cloud_cover_pct` (S2 only), `footprint`, `processing_status` (`catalogued → fetched → preprocessed | failed`).

Sentinel-1 requires direction, relative orbit (1–175) and polarizations. Sentinel-2 requires cloud cover and has no polarizations. `track_key = (direction, relative_orbit)`.

## C. Pre/post pair (`contracts/pairing.py`)

`pair_id`, `event_date`, `event_time_utc`, `pre` (1+ acquisitions, oldest first), `post`, `orbit_direction`, `relative_orbit`, `aoi_coverage_fraction`, `pairing_rationale` (required text), `quality_flags` (`single_pre_image`, `long_post_delay`, `partial_aoi_coverage`, `missing_polarization`).

Derived: `pre_post_gap_days`, `post_delay_days`.

Rejected: any acquisition on another track; non-S1 acquisitions; duplicates; same-day images when the event time is unknown; pre images not before / post not after the event; no shared polarization; `single_pre_image` flag inconsistent with the number of pre images.

## D. Detection raster (`contracts/detection.py`)

Metadata for a GeoTIFF: `raster_id`, `aoi_id`, `pair_ids`, `path`, optional `sha256`, `crs` (EPSG), `transform` (affine), `width`, `height`, `bands`, `method` (`baseline_change` / `flood_model` / `fusion`), `model_version` (required unless baseline), `config_fingerprint`, `created_at`.

Band 1 is always `impact_class`, uint8, nodata 255:

| Code | ImpactClass | Meaning |
|---|---|---|
| 0 | NO_CHANGE | observed, no qualifying change |
| 1 | FLOOD_WATER | new water |
| 2 | DEBRIS_DISTURBANCE | radar change consistent with debris/disturbance (change statistics, not a learned class) |
| 3 | PERMANENT_WATER | water before the event |
| 254 | UNOBSERVABLE | shadow/layover/no valid data: **no claim either way** |
| 255 | OUTSIDE_AOI | nodata |

Extra bands carry continuous values (e.g. `flood_probability`, `debris_zscore`). Helpers: `validate_class_array`, `confidence_tier(p, thresholds)` → low/medium/high.

## E. Road impact (`contracts/impact.py`)

`segment_id`, `osm_way_id`, `highway`, `geometry`, `length_m`, `state` (open/blocked/unobservable), `affected_length_m`, `longest_affected_run_m`, `unobservable_length_m`, `max_impact_probability`, `bridge` (`not_a_bridge` / `not_exposed` / `exposed_to_impact_zone`), `evidence_ids`. Derived: `observable_fraction`.

Rules: lengths within `[0, length_m]`; affected + unobservable ≤ length; BLOCKED needs affected length; UNOBSERVABLE needs unobservable length or a probability; non-OPEN needs evidence; EXPOSED bridges need affected length. There is no structural-collapse state.

## F. Settlement status (`contracts/settlement.py`)

`settlement_id`, `name`, `source` (`osm_place` / `building_cluster`), `location`, `building_count` (OSM buildings, **not population**), `snap_distance_m` (None = no road within the limit), `pre_event` / `post_optimistic` / `post_pessimistic` (`Reachability`: reachable, target_id, target_kind ∈ hospital/town/analysis_exit, route_length_m), `degraded_detour_ratio`, `status`, `blocking_segment_ids`, `evidence_ids`.

Rules: `status` must equal `classify_settlement(...)` (table in architecture.md); reachability and route length are monotone pre ≥ optimistic ≥ pessimistic; cut-off statuses need blocking segments and evidence.

## G. Evidence record (`contracts/evidence.py`)

`evidence_id` (use `make_evidence_id`, deterministic), `subject_type` (pair / detection_region / road_segment / building / settlement), `subject_id`, `claim`, `location` (any geometry), `sources` (`SourceRef`: permitted `DataSource`, `source_id`, `reference_date`, optional `acquisition_start`), `method`, `method_version`, `config_fingerprint`, `probability`, `values` (finite numbers), `artifacts` (`ArtifactRef`: relative path, kind, optional sha256), `created_at`.

Rules: at least one source; sources can only be permitted inputs; claims may not mention collapse, destruction, casualties or population.

## Catalogue search result (`contracts/catalogue.py`, Step 2A)

`CatalogueSearchResult`: `provider`, `query_filter`, `aoi_id`, `event_date`, `event_time_utc`, `window_start`, `window_end`, `retrieved_at`, `returned_count` (must equal the number of records: none silently dropped), `truncated`, `config_fingerprint`, `records`, `groups`. Methods: `accepted()`, `rejected()`, `status_counts()`, `explain(id)`.

`CandidateRecord`: `product_id` (CDSE Id), `name` (ESA name = `Acquisition.source_id`), `status` (`CandidateStatus`), `reason`, `sensing_start`, `acquisition` (when buildable), `role` (pre_event/post_event), `days_from_event`, `scene_coverage`, `pass_id`, `pass_coverage`, `duplicate_of`, `metadata_sources`. ACCEPTED requires acquisition, role, coverages and pass id.

`TrackGroup`: `orbit_direction`, `relative_orbit`, `pre_event`, `post_event` (accepted names, oldest first), `pass_ids`; derived `has_pre_and_post`. The result rejects groups that reference unaccepted scenes or scenes from another track.

## Temporal selection (`contracts/temporal.py`, Step 2B)

`AcquisitionPass`: `pass_id`, `platform`, `track` (`TrackKey`: orbit_direction, relative_orbit), `role`, `start`, `days_from_event`, `scene_names` (slices, oldest first; = `Acquisition.source_id`), `scene_coverages`, `coverage` (union), `metadata_from_product_name`; derived `multi_slice`. Pre passes must start before the event, post passes after.

`TemporalSetCandidate`: `candidate_id` (hash of track + pass ids), `rank` (within track), `track`, `shape` (`2pre+1post` | `1pre+1post`), `pre_event_pass_ids` (oldest first: PRE-1, PRE-2), `post_event_pass_id`, `additional_pre_pass_ids`, `post_gap_days`, `recent_pre_age_days`, `pre_separation_days`, `coverage` (weakest pass), `score` (`PairingScore`: post_gap_score, pre_recency_score, baseline_separation_score, coverage_score, quality_score, total, all in [0, 1]), `quality_flags` (`SetQualityFlag`), `multi_slice`, `selection_reason`. Shape, flags and separation must be mutually consistent; a pass may appear once per candidate.

`TrackSelection`: `track`, `status` (candidates / insufficient_pre / insufficient_post / no_valid_baseline), `note`, `pre_pass_ids`, `post_pass_ids` (every valid pass), `candidates_generated`, `candidates` (ranks 1..n).

`TemporalSelectionResult`: `heuristic`, `aoi_id`, `event_date`, `event_time_utc`, `catalogue_digest` (SHA-256 of the input catalogue JSON), `catalogue_retrieved_at`, `config_fingerprint`, `passes`, `tracks`. Rejects passes referenced from a different track ("cross-track sets are never allowed"), wrong roles, unknown ids, unsorted tracks. Scenes are referenced by name only; `Acquisition` objects stay in the catalogue result (no duplicate copies). `to_pre_post_pair(candidate, selection, catalogue)` builds the `PrePostPair`.

## Data fetching (`contracts/fetch.py`, Step 2C)

`S1FetchPlan`: `aoi_id`, `event_date`, `selection_digest`, `catalogue_digest`, `candidates` (`PlannedCandidate`), `passes` (`PlannedPass`: pass_id, track, role, start, product_ids = all slices; derived multi_slice), `scenes` (`PlannedScene`: product_id, product_name, pass_id, slice_index, slice_count, platform, acquisition_start, orbit_direction, relative_orbit, product_type, polarizations, candidate_ids). Rejects duplicate products, scenes off their pass's track, and **any planned pass with a missing slice**.

`DEMPlan`: `aoi_id`, `dataset`, `buffer_km`, `required_tiles`, `tiles` (`DEMTile`: tile_id, product_id/name or None, footprint), `query_filter`; derived `missing_tiles`.

`DownloadProvenance` (sidecar per file): see data_fetching.md. `kind` is `sentinel1_grd` (requires data_source sentinel-1 and acquisition/orbit/polarisation/pass fields) or `copernicus_dem_tile` (requires copernicus-dem, tile_id, dem_dataset); `source_url` may not carry a query string or token; `data_role` is always `pipeline_input_raw`.

`FetchManifest`: `aoi_id`, `dry_run`, `started_at`, `finished_at`, `selection_digest`, `plan`, `dem_plan`, `files` (`FileFetchResult`: kind, product, pass/tile, status planned/downloaded/skipped_existing/not_attempted/failed, size, local_path, provenance_path, message), `pass_complete`, `candidate_ready`. Completeness must agree with the file results (validated).

## Model input (`contracts/model_input.py`, Step 4A)

`ModelInputSpec` (version `bato-model-input-v1`): `channels` (`ModelChannel`: index, name, stack_band, role, polarization, kuro_siwo_input, temporal_meaning, mean, std; must be POST_VV, POST_VH, PRE2_VV, PRE2_VH, PRE1_VV, PRE1_VH with the Kuro Siwo constants), `normalization` (`NormalizationSpec`: clamp 0–0.15, NaN → 0.15, float32 z-score; constants locked), `tiling` (`TilingSpec`: tile 224 locked, stride 1–224, pad_end with 0.0, padding invalid; derived overlap), `validity` (`ValidityPolicy`), `dem_features` = none, `items` (`SpecItem`: key, statement, `EvidenceStatus`, evidence; derived resolved); derived `tensor_shape`, `unresolved_items`, `kuro_siwo_compatibility`. `spec_fingerprint()` = SHA-256 of its canonical JSON. `KuroSiwoLabel`: 0 no water, 1 permanent water, 2 flood, 3 ignore (no debris class).

`ModelInputManifest`: `bato_version`, `config_fingerprint`, `created_at`, `aoi_id`, `candidate_id`, `event_date`, `source` (`SourceStackRecord`: stack directory, provenance + SHA-256, stack + SHA-256, mask + SHA-256, layout, stack config fingerprint, track, `passes` with their Kuro Siwo role), `grid`, `spec`, `spec_fingerprint` (validated), `tiles` (`TileRecord`: tile_id, offsets, data extent, valid_pixels, transform), `channel_diagnostics` (clamp saturation), `valid_pixels` (must equal the tile sum without overlap), `warnings`; derived `tiles_with_valid_pixels`.

## Other

- `OSMSnapshot` / `select_snapshot_date` (`contracts/osm.py`): snapshot = event date − `osm.snapshot_offset_days` (≥ 1), capped at the latest available date; a snapshot on or after the event date is rejected.
- `ReferenceDataset` (`evaluation/reference.py`): evaluation-only descriptor for EMSR927 and similar. Pipeline code cannot import it.
- `StageReport` (`logging_utils.py`): name, inputs, outputs, warnings, fallback, status, duration, error.

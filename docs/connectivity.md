# Milestones 6–8: pre-event OSM, road impact, connectivity

**Status:** implemented and unit-tested on synthetic data. **Not validated**: it is built on the provisional Step 4C baseline and on provisional road and connectivity thresholds. No reference map was used anywhere.

**Official / submitted rule: road rule v1.** The submitted BATO result (dashboard, headline numbers, EMSR927 comparison) is the road-rule **v1** connectivity run: road state from the Step 4C baseline only. Road rule v2 (Step 4C FLOOD *or* SAR disturbance) was an experiment. The disturbance detector it depends on was **rejected** under its pre-registered acceptance criteria ([debris_detection.md](debris_detection.md)), so v2 is **not** part of the official result, and the dashboard refuses to build from a v2 run.

> **Reproducing the official result:** `scripts/run_connectivity.py` still defaults to `--road-rule v2` (the code default was deliberately left unchanged). Always pass the rule explicitly:
>
> ```bash
> python scripts/run_connectivity.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --road-rule v1
> ```
>
> Windows: `python scripts\run_connectivity.py --manifest data\raw\manifests\<aoi>_fetch_manifest.json --road-rule v1`

| | |
|---|---|
| OSM extraction | `src/bato/osm/extract.py`, `scripts/fetch_osm_snapshot.py` |
| Road impact | `src/bato/impact/roads.py` |
| Connectivity | `src/bato/connectivity/{settlements,graph,engine}.py`, `scripts/run_connectivity.py` |
| Contracts | `contracts/osm.py` (`OSMExtractProvenance`), `contracts/impact.py` (`RoadState`: OPEN / BLOCKED / UNOBSERVABLE), `contracts/settlement.py` (decision table), `contracts/connectivity.py`, `contracts/evidence.py` |
| Thresholds | `road_impact.*` and `connectivity.*` in `config/thresholds.yaml` (unchanged values) |
| Settings | `osm.*` and `connectivity.output_subdir` in `config/config.yaml` |

## 1. Pre-event OSM (ohsome API v2)

- **API:** ohsome API v2 at `osm.api_base_url` = `https://api.heigit.org/ohsome-api/v2-rc` ([reference](https://docs.ohsome.org/ohsome-api/v2/reference/api.html)). The old v1 (`api.ohsome.org/v1`) is retired.
- **Request:** `POST /extraction/features.parquet` with JSON body `{"filter": …, "aoi": [xmin, ymin, xmax, ymax], "time": "<snapshot>T00:00:00Z", "clip": true}`. The response is GeoParquet.
- **API key:** mandatory for every extraction request. It is sent as the raw `authorization` header and read from the environment variable named by `osm.api_key_env` (default `OHSOME_API_KEY`). It is never written to config, provenance or logs. Free keys: https://account.heigit.org/signup.
- **Snapshot date:** `event_date − osm.snapshot_offset_days` (1 day), capped at the latest date in the database (`GET /metadata` → `temporalExtent.end`), so it is always strictly before the event. Each request asks for the OSM state at 00:00 UTC of that date.
- **Boxes:** taken from the Step 3.5 stack provenance, so the road graph matches the SAR grid.
  - The *analysis box* is the AOI plus `aoi.analysis_buffer_km` (15 km). It lies inside the 20 km SAR crop.
  - The *settlement box* is the AOI plus `osm.building_margin_km` (2 km).
- **Layers:** one request each, same filters and boxes as before.

| layer | filter | box | post-processing |
|---|---|---|---|
| roads | `type:way and highway in (osm.road_highway_values)` | analysis | geometry as is |
| places | `place in (city, town, village, hamlet)` | analysis | client-side centroid |
| hospitals | `amenity=hospital or healthcare=hospital` | analysis | client-side centroid |
| buildings | `building=* and building!=no and geometry:polygon` | settlement | client-side centroid |

  Roads are the vehicle-carrying classes, including `track`; footpaths, steps and cycleways are excluded. v2 has no centroid endpoint, so centroids are computed in `bato.osm.geometry` (area-weighted for polygons). GeoParquet `geom` (WKB) is decoded there too.
- **Post-event guards:** every row's `edit_timestamp` must be at or before the snapshot instant, and its `valid_to_timestamp` must be after it (the version was the live one at the snapshot). Anything else is refused as post-event OSM. The provenance also refuses any layer requested at a time other than the snapshot instant.
- **Reading GeoParquet needs `pyarrow`** (`pip install pyarrow`, or `pip install -e .[osm]`).
- **Storage:** the GeoParquet responses (`roads.parquet`, `places.parquet`, `hospitals.parquet`, `buildings.parquet`) are stored byte-for-byte in `data/raw/osm/<aoi>/snapshot-<date>/` with `osm_extract_provenance.json` (request bodies without the key, SHA-256, counts, ohsome version and coverage date). The extract is published atomically. The loader re-verifies everything and converts the rows into the feature dicts the road and settlement code reads (`@osmId` = `<type>/<id>`, `@lastEdit`, tags).

Not used: population datasets (WorldPop, GHSL, …), damage maps, post-event OSM.

## 2. Road segments and the three road states

- **Topology:** shared vertices. Ways meeting at an OSM node share identical coordinates, so they become one graph node. A bridge crossing a road shares no vertex, so it creates no false junction.
- **Edges and segments:** an *edge* is a junction-to-junction polyline. It is cut into equal *segments* of at most `road_impact.segment_length_m` (50 m). Segments are the graph edges and the unit of every road claim.
- **Exits:** ohsome clips roads to the analysis box. A dead end lying exactly on the box edge is an **exit node**: the road continues outside.

### Road rule v1 (OFFICIAL): Step 4C FLOOD only

`baseline_state.tif` is sampled along each edge every half ground pixel. Each sample is classed as follows:

| sample | rule |
|---|---|
| FLOOD | a FLOOD pixel lies within `road_buffer_m` (10 m; square window, radius rounded to pixels) |
| UNOBSERVABLE | otherwise, its pixel is NOT_ASSESSED (invalid SAR, no DEM, layover, shadow, steep) or lies outside the raster |
| observed-open | otherwise: NO_FLOOD or PRE_EVENT_WATER |

Runs are measured along the whole edge and **joined across nodes**, so OSM way splits, bridge ends and junctions do not cut a flood run below the threshold. At a junction the joined length is the total connected length. Each segment then gets one state:

- **BLOCKED:** the segment holds part of a FLOOD run of at least `min_blocked_run_m` (25 m).
- **UNOBSERVABLE:** not BLOCKED, and the segment holds unobservable samples of a non-open run (FLOOD or UNOBSERVABLE) of at least `min_unobservable_run_m` (25 m). A blocking run could be hidden there.
- **OPEN:** otherwise. A shorter flood run is recorded as `affected_length_m` but does not block.

UNOBSERVABLE is never converted to OPEN or BLOCKED. Each UNOBSERVABLE segment records its reasons (sample counts per NOT_ASSESSED state).

### Road rule v2 (EXPERIMENTAL, REJECTED; not the official result): FLOOD or DISTURBANCE, geometry-only observability

Kept for transparency only. Its disturbance input failed its pre-registered checks ([debris_detection.md](debris_detection.md)), so no official number, dashboard view or EMSR927 comparison uses it. It is still the script's code default, which is why the official command passes `--road-rule v1` explicitly.

`--road-rule v2` also reads the disturbance run of the same stack and Step 4C state (`data/interim/detection/<aoi>/<candidate>/debris/`, [debris_detection.md](debris_detection.md)). It changes only the sample classes:

| sample | rule v2 |
|---|---|
| affected | a FLOOD (Step 4C) **or** DISTURBANCE pixel lies within `road_buffer_m` |
| UNOBSERVABLE | otherwise, its pixel is geometry-unobservable (invalid SAR, no DEM, layover, shadow: debris state 10–13) or lies outside the raster |
| observed-open | otherwise (including steep terrain, which the disturbance detector assesses) |

The 25 m blocked-run and hidden-run rules, the graphs, routing, the decision table and the stretch ranking are unchanged. A BLOCKED claim names its signal: "radar disturbance consistent with debris" (with z, sigma VV/VH, the PRE-1/PRE-2/POST dates) or open-water-like darkening. Its evidence values carry the d1/d2/d0 (VV, VH) of the change pixels within the road buffer, sigma, z, the Step 4C states under the road and the geometry status. `--road-rule v1` is the official rule (every Step 4C NOT_ASSESSED state, steep included, is unobservable); the provenance records `road_rule`, and the dashboard and the EMSR927 evaluation select only `road_rule: v1` runs.

**Bridges** keep `bridge` exposure only (`exposed_to_impact_zone` / `not_exposed`); there are no structural claims.

## 3. Settlements and access destinations

**Settlements** are evaluated inside the AOI only:

- **Named places:** OSM `place=city|town|village|hamlet`.
- **Unnamed building clusters:** single linkage at `building_cluster_eps_m` (125 m), at least `building_cluster_min_buildings` (5) buildings, and no named in-AOI place within 125 m.
- **Buildings near a named place:** a cluster with a building within 125 m of a named in-AOI place is linked. Each of its buildings counts for the nearest place it is linked to, so a valley-long chain is split, not handed whole to one place.

`building_count` counts **mapped OSM buildings, a size proxy, never population.**

**Destinations** may lie anywhere in the analysis box:

- **hospital:** `amenity=hospital` or `healthcare=hospital`. This is an *anchor*.
- **analysis_exit:** see §2. This is an *anchor*.
- **town:** `place=city|town`. A town counts only while it is itself connected to an anchor in that graph. It is never a destination for itself or for a duplicate feature at the same node.

**A settlement is reachable if and only if it can reach an anchor.** Towns only shorten the reported route; a town that is itself cut off cannot make its pocket look connected. A duplicate node-and-area town cannot mask a cut-off either.

**Snapping:** a settlement snaps to the nearest graph node within `settlement_snap_max_m` (500 m) of its own point (place point or cluster centroid). Only if there is none does it try its own buildings that lie within 500 m of that point. Destinations are snapped from their own point. A settlement with no node in range has `NO_PRE_EVENT_ACCESS` (no road), which is not a flood effect.

## 4. Before vs after: three graphs, one decision table

All three graphs come from the same pre-event network:

| graph | segments removed |
|---|---|
| pre-event | none |
| post-optimistic | BLOCKED |
| post-pessimistic | BLOCKED and UNOBSERVABLE |

A multi-source Dijkstra on each graph gives the nearest destination and route length. `classify_settlement` then decides the status:

| pre | optimistic | pessimistic | status | access state |
|---|---|---|---|---|
| unreachable | – | – | `no_pre_event_road_access` | NO_PRE_EVENT_ACCESS |
| reachable | unreachable | – | `confirmed_cut_off` | **CONFIRMED_CUTOFF** |
| reachable | reachable | unreachable | `possibly_cut_off` | **POSSIBLE_CUTOFF** |
| reachable | reachable | reachable, ≥ 1.5 × pre route | `degraded` | CONNECTED |
| reachable | reachable | reachable | `connected` | **CONNECTED** |

**Before vs after** in the summary means two things:

- settlements with a route on each graph;
- *pockets*: components that hold settlements but no destination.

## 5. Which blocked road to clear (the "click a blocked road" answer)

- **Blocked stretch:** a connected run of BLOCKED segments (adjacent segments sharing a node). One flooded spot usually covers several 50 m segments, and clearing only one of them changes nothing. So the stretch is the unit that gets cleared.
- **Clear one stretch only (union-find):** components are computed on the post-optimistic graph. Reopening a stretch merges the components it touches. Every CONFIRMED_CUTOFF settlement in a merged component without an anchor regains a route if any merged component has one. That gives `settlements_reconnected` and `reconnected_building_count` (proxy).
  - The same count on the post-pessimistic graph is `settlements_reconnected_on_observed_roads`: settlements that regain a route on observed-open roads only.
- **Fewest stretches to clear (0-1 BFS):** each blocked stretch is a virtual node, entered at cost 1 and left at cost 0. Passable segments cost 0 both ways. The search runs from all destinations.
  - For each CONFIRMED_CUTOFF settlement it gives `min_blocked_stretches_to_clear` and one clearing set.
  - Each stretch reports `in_minimum_clearing_set_of`.
  - This covers settlements that need two or more stretches cleared; no single stretch would show them.
- **Ranking:** stretches are ranked by settlements reconnected, then buildings (proxy), then observed-road reconnections, then clearing-set count. Every segment of a stretch carries the stretch's numbers in `road_segments.geojson`, so clicking any blocked segment shows how many settlements it reconnects.

## 6. Evidence trail

Each `EvidenceRecord` cites the three Sentinel-1 products (PRE-1, PRE-2, POST), the Step 4C state raster (hash), the OSM features (`way/…@<snapshot>`) and the config fingerprint. Records are written for:

- every BLOCKED and UNOBSERVABLE segment (`rule:road_state_from_baseline_v1`; under rule v2 it also cites the disturbance state raster and change table);
- every CONFIRMED and POSSIBLE cut-off settlement (`graph:settlement_reachability_3graph_v1`). Its `evidence_ids` start with its own record, followed by the records of its separating segments: settlement → graph decision → segment → flood or observability evidence;
- every stretch that reconnects settlements (`graph:single_segment_reconnection_unionfind_v1`).

Claims are checked against the banned-term list (collapse, casualties, population, …). OSM names that contain such a term are left out of the claim text.

## 7. Outputs

`data/processed/connectivity/<aoi>/<candidate>/` is published atomically; `--overwrite` keeps the old directory as `.superseded-<time>`.

| file | content |
|---|---|
| `road_segments.geojson` | every segment: state, lengths, Step 4C reasons, evidence ids; BLOCKED ones carry the stretch's reconnection numbers |
| `settlements.geojson` | access state, contract status, the three routes, separating segments, clearing set, exit reachability |
| `access_destinations.geojson` | hospitals, towns and exits, including unsnapped ones |
| `blocked_segments_ranked.json` | every blocked stretch, ranked |
| `evidence_records.jsonl` | one `EvidenceRecord` per line |
| `connectivity_provenance.json` | input and output hashes, thresholds, rules, summary, top 25 stretches, `validated: false` |
| `connectivity_report.md` | tables and evidence cards (confirmed cut-offs; largest possible cut-offs) |

## 8. Limitations

- **Open-water flooding only (official v1).** Under v1, landslides, debris flows and mud are not detected, and bridge damage is never detected. OPEN means "no open-water-like flood signal on observable road", not "passable". (The rejected v2 experiment added a disturbance signal; it is not used.)
- **UNOBSERVABLE dominates in steep terrain.** About 91.2 % of the Trishuli scene (41,285,907 of 45,256,776 pixels) is NOT_ASSESSED in Step 4C. Of those not-assessed pixels (not of the whole scene), about 86.1 % (35,544,186) come from the 15° slope gate, 11.4 % (4,713,918) from layover and 2.5 % (1,027,803) from shadow. Many settlements are therefore POSSIBLE_CUTOFF (access unverified): this means BATO cannot confirm access, **not** that they are cut off.
- **OSM completeness bounds everything.** Unmapped roads, tracks, buildings or hospitals are invisible.
- **Simplified routing.** The graph is undirected (oneway ignored), with no road-class weighting and no vehicle type. Exits assume the road continues outside the box.
- **Approximations.** Settlements snap to a single nearest node. The road-buffer window is square.

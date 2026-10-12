# Dashboard: the official result, read-only

| Part | Where |
|---|---|
| Bundle builder | `src/bato/dashboard/{bundle,routes,rasters,png}.py`, `scripts/build_dashboard.py` |
| Web app | `dashboard/app/` (static HTML/CSS/ES modules; MapLibre GL 4.7.1 and IBM Plex vendored, so it works offline) |
| Local server | `scripts/serve_dashboard.py` (standard library, binds to 127.0.0.1) |
| Tests | `tests/unit/test_dashboard_bundle.py`, `tests/integration/test_dashboard_{ui,scripts,real_data}.py` |

The dashboard shows existing results; it never computes a road state, a reachability result or a settlement status.

## 1. Build and run

```bash
python scripts/build_dashboard.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json --title Trishuli \
    --debris-control-dir data/interim/detection/aoi-f2cc742b0e25/ts-5adc0f866c2f/debris
python scripts/serve_dashboard.py --open          # http://127.0.0.1:8765/
```

The build writes `data/processed/dashboard/<aoi>/<candidate>/`:

- `bundle.json`;
- `overlays/*.png`;
- `chips/*.png`.

It also writes `data/processed/dashboard/index.json`. The bundle is published atomically, and an older one is kept as `.superseded-*`.

`rasterio` is optional. Without it there are no overlays, no hillshade and no SAR chips; everything else still works.

## 2. Integrity rules (the build refuses otherwise)

1. **Official run only.** The official run is the latest connectivity output with `road_rule: v1`, including a `.superseded-*` directory. A road-rule v2 run is refused, because the experimental disturbance detector was rejected. `--connectivity-dir` picks a v1 run explicitly.
2. **Hashes.** Every output of that run must match the SHA-256 in its provenance. The stack and Step 4C provenance it cites must match too. The Step 4C state raster, stack and DEM are used for images only after their hashes match.
3. **Summary.** The provenance summary is recomputed from `road_segments.geojson` and `settlements.geojson`:
   - segments and km by state;
   - settlements by access state;
   - reachable counts per graph.

   Any disagreement stops the build.
4. **Route traces.** The engine publishes route lengths, not paths. The dashboard redraws each path from the published segments, using `u`, `v`, `length_m` and the same three graph masks. It keeps a trace only if its length equals the published route length within 0.5 m. Otherwise the trace is reported as unavailable and nothing is drawn. Equal-length alternatives may be drawn instead of the engine's choice.
5. **Derived numbers are labelled as derived.** These are:
   - route km by road state;
   - road distance to the nearest hospital or exit on each graph;
   - groups of unobservable segments bordering access-unverified settlements.
6. **No population.** Building counts pass through with "Building count proxy — not population."
7. **Experimental detector.** It appears only under *Methodology → Experimental methods*. Its pre-registered checks (a)–(d) are re-evaluated there from its provenance numbers, so the verdict is computed, not typed.

## 3. What is on screen

- **Status modules.** Blocked, observed open and unobservable km share one network bar, with each module's own slice lit. Settlement access is a unit chart: one cell per settlement.
- **Map.** The map shows:
  - Step 4C FLOOD overlay;
  - not-assessed terrain, hatched;
  - DEM hillshade;
  - roads by state (red blocked, teal open, dashed amber unobservable);
  - settlements by access state;
  - hospitals and road exits.

  The base map is local: a dark background plus the DEM hillshade from the bundle. No online tile service is used.
  The first screen shows flood evidence, blocked and observed-open roads and settlements. Road exits and not-assessed
  terrain are off by default and one click away under Layers.
- **Modes.** Normal, Observability ("What radar can't confirm": observed roads dimmed, unobservable roads coloured by main reason), Access and Evidence.
- **Evidence drawer (blocked road).** It shows:
  - affected length and the stretch;
  - SAR chips: PRE-1, PRE-2 and POST VV on one fixed dB stretch, POST − PRE-2 change, and the Step 4C state with the road drawn in;
  - dates and the samples under the road;
  - *settlements reconnected if cleared*, including zero;
  - verified routes that used the stretch;
  - the evidence record.
- **Settlement drawer.** It shows:
  - access badge and building proxy;
  - pre-event and post-event route km;
  - unobservable route share;
  - the route strip (settlement → road network → destination, with unobservable parts hatched);
  - nearest hospital and exit by road;
  - the separating segments;
  - the evidence record.
- **Panels.**
  - Incident: mission, "What can BATO confirm?" and the incident record.
  - Map layers.
  - Access: before vs after tracks, plus a filterable settlement list.
  - Observability: radar geometry diagram and unobservable km by main reason.
  - Evidence chain: six steps with rules, thresholds and hashes.
  - Operations: lists ordered by stated quantities; there is no priority score.
  - Methodology.
- **Timeline.** PRE-1, PRE-2, the OSM snapshot, the event and POST, evenly spaced, with the real gap in days on each link.
- **Follow the evidence.** Five stages on the rank-1 blocked stretch: POST SAR chip on the map, Step 4C state, road state, access result, settlement.
- **Judge mode** (rail button or `J`). Seven presenter-driven steps (Next/Previous with buttons or arrow keys, Reset, Esc), with narration filled from the bundle and a closing card.

## 4. Wording that follows the data

"Unobservable" is the official road state. The observability views split it by main reason (the reason covering most of a segment's not-assessed samples):

- **radar geometry:** layover or shadow;
- **terrain rule:** Step 4C does not apply its open-water flood rule on steep slopes;
- **data gap:** invalid SAR, no DEM, or outside the raster.

In the frozen Trishuli v1 result most unobservable road comes from the terrain rule. The interface therefore says "What radar can't confirm", not "what radar can't see". It never attributes steep-slope roads to layover or shadow.

## 5. Limitations

- **Not validated.** Every claim carries the NOT VALIDATED note of its provenance.
- **Approximate placement.** Chips and overlays are display products: block-reduced overlays, a fixed dB stretch, and nearest-neighbour resampling for Mercator rows. They are not inputs to anything.
- **No online basemap.** CARTO's raster tiles now return images watermarked "API KEY REQUIRED" to anonymous
  requests (HTTP 200, so no error handler can catch them). The dashboard therefore uses no tile service. Without
  rasterio at build time there is no hillshade, and the base map is the plain dark background.
- **Route traces.** A trace that cannot be reproduced exactly is not drawn. Where shortest paths tie, the drawn path can differ from the engine's while having the same length.

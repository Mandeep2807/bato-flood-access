# EMSR927 comparison protocol (bato-emsr927-v1)

**Status:** DRAFT, written before any metric was computed. The official products have now been inspected and the mapping completed from them (section 11). It is frozen only after a re-run inspection reports no unmapped layer or value (section 9). The machine-readable part is `config/evaluation/emsr927_protocol.yaml`, and the code refuses to compute metrics until both files are frozen and unchanged.

**What is compared:**
- **BATO:** the official BATO result (Step 4C state raster cited by the frozen road-rule v1 connectivity run) and its published road segments.
- **Reference:** the Copernicus EMS Rapid Mapping products of activation EMSR927.

**Leakage rule.** EMSR927 is reference data. It is never a model input, training label, threshold source, feature, connectivity input or dashboard input. The products live only under `data/reference/EMSR927/`, results only under `data/eval/EMSR927/`, and only `bato.evaluation` reads them. The BATO result was produced before this protocol existed and is not re-run.

## 0. Sources

| Item | Source |
|---|---|
| Products, versions, status, image sensors/times, download URLs | Official activation record `…/dashboard-api/public-activations/?code=EMSR927`, saved raw with retrieval time and SHA-256 |
| EMS AOI geometry | The product's own `areaOfInterestA` layer. The record's AOI `extent` is used only to decide which products to download. |
| BATO AOI | `aoi_polygon` of the stack provenance cited by the official connectivity run (hash-verified). Never typed coordinates. |
| BATO grid | `grid.transform`, `width`, `height` of that stack provenance (EPSG:3857, 10 m). The Step 4C raster must have exactly this grid. |
| BATO pixel states | Step 4C `baseline_state` raster, SHA-256 checked against the connectivity provenance |
| BATO road states | `road_segments.geojson` of the official v1 run, SHA-256 checked |
| BATO POST time | `inputs[role=POST].acquisition_start` of the stack provenance (UTC, from the SAFE product) |

Every mapping below comes from the inspection of the downloaded official zips (section 11), not from any third-party copy.

**Primary evaluation: road damage (section G).** The official products contain no open-water flood class, so a flood comparison is NOT CALCULABLE. They do contain graded road-damage lines, and BATO publishes road states, so the road comparison is the primary EMSR927 evaluation. It is a case-study comparison, not ground truth for BATO's flood rule (see the mismatch notes in G).

## 1. Product selection

1. Download only products with:
   - status `F` (final);
   - a non-empty `downloadPath` on `rapidmapping.emergency.copernicus.eu`;
   - an AOI `extent` that intersects the BATO AOI.

   AOIs marked not feasible, or with no download, are listed with their reason.
2. Compare one product per EMS AOI. Each image listed for a product gives a gap |t_image − t_POST|; take the worst case over those images and choose the product with the smallest worst-case gap. A product with an image that has no time has an unknown gap and cannot be chosen. Ties go to the higher monitoring number, then the higher version. This rule is applied under each time-zone reading (section 7), and the evaluation stops if the readings disagree. The other downloaded products are listed but not scored.
3. Results are given per EMS AOI. A pooled result (sum of counts) is given only if no two compared EMS AOIs share a BATO pixel. Otherwise the pooled result is NOT CALCULABLE.

## A. Class mapping

**BATO.**
- Positive: Step 4C state 1 `FLOOD`, meaning new open-water-like darkening: VV < −15 dB after the event where it was not dark before, on assessed terrain.
- Negative: state 0 `NO_FLOOD` and state 2 `PRE_EVENT_WATER`. Water that was already present is not new flood.
- `NOT_ASSESSED`: states 10–14 (invalid SAR, no DEM, layover, shadow, slope ≥ 15°).
- Any other value stops the evaluation.

**EMS observed events** (`observedEventA`, class field `obj_desc`). Each value maps to exactly one category. A value that is not mapped stops the evaluation (fail closed).

| Category | EMSR927 values mapped (from inspection) | Use |
|---|---|---|
| `flood_water` | **none**: the products contain no open-water flood class | Class-matched pixel comparison with BATO FLOOD: **NOT CALCULABLE** |
| `reference_mass_movement` | `Landslide` (`event_type` 6-Mass Movement, `det_method` Photo-interpretation) | **CLASS-MISMATCHED FOOTPRINT OVERLAP** only (section E2) |
| `flood_trace` | none | Pixel count only, removed from every comparison area |
| `other_event` | none | Pixel count only, removed from every comparison area |

**Mismatch, stated explicitly.**
- Step 4C detects open-water-like radar darkening only. It does not detect mud, debris, deposited sediment, landslide scars, damaged buildings or damaged roads.
- The Landslide polygon is **not** renamed or treated as flood. No flood precision, recall, F1 or IoU is computed against it.
- Categories are never merged.

## B. Spatial grid

1. EMS polygons are brought onto the BATO grid (EPSG:3857, pixel-for-pixel the Step 4C raster). Its pixels are 10 m in EPSG:3857 units, which is about 10 × cos(28°) ≈ 8.8 m on the ground. Areas are reported as pixel counts and as approximate ground km², each row's pixel area scaled by cos²(latitude).
2. Polygon vertices must be WGS84 lon/lat:
   - GeoJSON: no `crs` member, or a CRS84 / EPSG:4326 one;
   - shapefile: a geographic WGS84 `.prj`.

   They are projected with the spherical Web-Mercator formula used by the pipeline. Any other CRS stops the evaluation; nothing is reprojected silently.
3. A pixel belongs to a polygon when its **centre** lies inside, using the even-odd rule over the feature's rings, so holes are excluded.
4. Pixels are never resampled and the BATO raster is never changed.
5. Consistency checks, which fail closed:
   - the product's AOI layer must cover at least one BATO AOI pixel;
   - every observed-event polygon must lie within the bounding box of the product AOI.

   These checks catch a swapped axis order or a wrong CRS.

## C. Comparison area

The comparison area for EMS AOI *k* is:

> `C_k` = { pixel centres inside EMS `areaOfInterestA` of *k* } ∩ { pixel centres inside the BATO AOI polygon } − { pixels inside an EMS `not_analysed` layer } − { pixels inside `flood_trace` or `other_event` polygons }

- The Step 4C grid also covers the analysis buffer around the BATO AOI. Pixels in that buffer are **excluded**, because the BATO AOI is the area BATO reports on.
- The overlap of each EMS AOI with the BATO AOI is reported in km² (spherical polygon area, R = 6371.0088 km) and in pixels.
- For a class-specific comparison, pixels of the *other* compared category are removed from `C_k`. For a flood comparison these are `reference_mass_movement` pixels; for the mass-movement overlap, `flood_water` pixels (none exist in EMSR927). This keeps the two categories from scoring against each other.
- `C_k` is also the area for the road and bridge comparisons: segments and features outside it are not used.

## D. Observable mask

- Assessed: Step 4C state ∈ {0, 1, 2}.
- `NOT_ASSESSED`: state ∈ {10, 11, 12, 13, 14}. These are reported by reason.
- Observable fraction of `C_k` = |assessed ∩ C_k| / |C_k|.

## E. Two modes

| Mode | Pixel set S | How NOT_ASSESSED is counted |
|---|---|---|
| `assessed_only` | C_k ∩ assessed | Excluded |
| `all_pixels` | C_k | **Counted as BATO negative (not flood).** BATO made no flood claim there, so a reference positive inside NOT_ASSESSED is a false negative in this mode. |

Inside S, with P = BATO positive and R = EMS reference positive:

- Counts:
  - TP = |P ∩ R|
  - FP = |P − R|
  - FN = |R − P|
  - TN = |S − P − R|
- Ratios:
  - precision = TP / (TP + FP)
  - recall = TP / (TP + FN)
  - F1 = 2TP / (2TP + FP + FN)
  - IoU = TP / (TP + FP + FN)
- A zero denominator gives **NOT CALCULABLE**, with the reason.
- "Accuracy" ((TP + TN) / |S|) is **not reported**. TN dominates in a mostly dry valley, so it would say nothing about mapping quality.

Also reported for each comparison:
- |R| inside `NOT_ASSESSED`, by reason;
- |R| on `PRE_EVENT_WATER` pixels;
- the observable fraction.

### E2. Mass-movement footprint: CLASS-MISMATCHED FOOTPRINT OVERLAP

"BATO radar-change footprint vs EMSR927 mass-movement footprint". With F = Step 4C FLOOD pixels and M = `reference_mass_movement` pixels, both inside `C_k`, only descriptive quantities are reported:
- |M|, |F| and |F ∩ M| (pixels, and approximate ground km² for M);
- share of the footprint marked FLOOD, |F ∩ M| / |M|;
- share of FLOOD inside the footprint, |F ∩ M| / |F|;
- share of the footprint that was assessed, and footprint pixels by Step 4C state (including each NOT_ASSESSED reason).

These are **not** precision, recall, F1, IoU or accuracy, and are never called flood metrics. The two maps describe different physical classes. A low or high overlap says where radar darkening and the mapped debris-flow footprint coincide, not whether either is correct. The overlap is contextual evidence. If it proves misleading, the report presents it as a map comparison rather than as numbers.

## F. Boundary tolerance (fixed in advance)

The tolerance is ±1 pixel: a 3 × 3 neighbourhood, 8-connected, 10 m in grid units (≈ 8.8 m on the ground). The existing BATO protocol pre-registers it as "strict and ±1-pixel boundary-tolerant".

Let D(X) be the 3 × 3 dilation of X. P and R are restricted to S before dilation, and counts are taken inside S.

- precision_tol = |P ∩ D(R)| / |P|
- recall_tol = |R ∩ D(P)| / |R|
- F1_tol = 2·precision_tol·recall_tol / (precision_tol + recall_tol)

IoU has no agreed tolerant form, so tolerant IoU is not reported. Strict metrics are the headline; tolerant metrics are supplementary. Section F applies to a class-matched pixel comparison only, which EMSR927 does not allow.

## G. Road damage comparison: PRIMARY EMSR927 evaluation

**Layers.** `transportationL` (road lines) with damage field `damage_gra`. Every value is mapped to exactly one grade; any other value stops the evaluation, before or after the freeze.

| `damage_gra` value (inspected) | Grade | Seen in |
|---|---|---|
| Destroyed | `destroyed` | AOI01 GRA (14), AOI03 GRA MONIT01 (262) |
| Damaged | `damaged` | AOI03 GRA MONIT01 (1) |
| Possibly damaged | `possibly_damaged` | AOI01 GRA (6), AOI03 GRA MONIT01 (108) |
| No visible damage | `no_visible_damage` | AOI01 GRA (11), AOI03 GRA MONIT01 (298) |
| Not Analysed | `not_analysed` | AOI03 GRA MONIT01 (1) |

The same category set is not assumed in every AOI. The check runs per compared product.

**Two pre-registered comparisons, reported separately and labelled.** Each one places every grade in exactly one of positive, negative or excluded; the code refuses a comparison that drops or merges a grade silently.

| Comparison | Role | Reference positive | Reference negative | Excluded (ambiguous, counted) |
|---|---|---|---|---|
| `primary_destroyed`: BATO BLOCKED vs EMSR927 road DESTROYED | **primary** | destroyed | no visible damage | damaged, possibly damaged, not analysed |
| `supplementary_any_damage`: BATO BLOCKED vs DESTROYED + DAMAGED + POSSIBLY DAMAGED | supplementary; not accuracy, not ground truth | destroyed, damaged, possibly damaged | no visible damage | not analysed |

In the primary comparison, damaged and possibly damaged roads are neither positive nor negative. Calling them negative would penalise BATO for flagging a road that EMS also saw as damaged.

**Matching rule (fixed before any metric).**
- **Unit.** A BATO road segment from the official v1 `road_segments.geojson` with at least 50 % of its samples inside `C_k`. Samples are spaced uniformly by ground distance (about 5 m), each standing for the same length, so dense vertices cannot inflate a length.
- **Reference label**, per comparison:
  - `need = min(25 m, segment length)`; a segment shorter than 25 m needs its whole length matched;
  - positive if ≥ `need` of the segment lies within 20 m of a positive-grade EMS line;
  - else excluded if that holds for an excluded-grade line;
  - else negative if it holds for a negative-grade line;
  - else unmatched, which is removed and counted.

  So precedence is positive > excluded > negative.
- **Distances.** 20 m = connectivity `road_buffer_m` (10 m) + 10 m for EMS/OSM line offsets. 25 m = connectivity `min_blocked_run_m`.
- **BATO label.** `BLOCKED` is positive, `OPEN` is negative.
  - `observed_only` mode: `UNOBSERVABLE` segments are excluded.
  - `all_segments` mode: `UNOBSERVABLE` segments are counted as BATO negative.
- **Metrics (segment level, matched domain).** TP, FP, FN and TN, then precision, recall and F1 (and IoU over segments, as in section E). A zero denominator gives NOT CALCULABLE. See "Evaluation unit and what the metrics mean" below.
- **Descriptive lengths.**
  - BATO blocked length in `C_k`;
  - EMS positive-grade features and length in `C_k`;
  - BATO blocked length within 20 m of EMS positive lines;
  - EMS positive length within 20 m of BATO blocked segments.

**Evaluation unit and what the metrics mean.**

- **Unit.** One BATO road segment. It is in the *matched domain* only when both hold:
  - at least 50 % of its samples lie inside `C_k`;
  - it receives a positive or negative EMS reference label under the matching rule. Segments labelled excluded or unmatched are not in the domain.
- **Counts:**
  - TP = blocked and reference-positive;
  - FP = blocked and reference-negative;
  - FN = not blocked and reference-positive;
  - TN = not blocked and reference-negative.
- **Interpretation:**
  - precision = among matched-domain segments BATO marked BLOCKED, the share EMS graded positive;
  - recall = among matched-domain segments EMS graded positive, the share BATO marked BLOCKED;
  - F1 combines the two.

  These are **conditional on the matched domain**. They are not full-network recall and not a statement about EMS-damaged roads that no BATO segment matches.
- **Modes.**
  - `observed_only`: matched-domain segments whose BATO state is BLOCKED or OPEN. UNOBSERVABLE segments are removed, because BATO made no observation there.
  - `all_segments`: every matched-domain segment, with UNOBSERVABLE counted as BATO negative. A reference-positive unobservable segment is therefore a FN.
- **Reference coverage**, reported next to every road result so EMS positive roads never disappear. EMS positive length is the length inside `C_k` of the comparison's positive grades. A positive sample is *matched* when it lies within 20 m of an eligible BATO segment that received the **positive** reference label; that is the only way it can become TP or FN. Reported:
  - EMS positive features in the area, and their total length;
  - matched length, in all segments and in `observed_only`;
  - unmatched length, split into two parts:
    - near an eligible BATO segment that did not get the positive label (for example less than 25 m alongside it);
    - no eligible BATO segment within 20 m (no BATO road there, or the BATO road lies outside the area);
  - matched fraction = matched / total (NOT CALCULABLE when the total is 0);
  - features fully, partially and entirely unmatched.

  Unmatched EMS positive length is **never** counted as TN or any other cell, and never dropped. A low matched fraction means the matched-domain recall describes only part of the EMS-damaged network.
- **Alternative unit, considered and not adopted.** True full-network FN accounting would need a common unit such as EMS road length:
  - recall = EMS positive length within 20 m of BATO BLOCKED / EMS positive length;
  - precision = BATO BLOCKED length within 20 m of EMS positive / BATO BLOCKED length.

  The two denominators measure different networks (EMS-digitised roads and OSM roads), so these would not be one consistent confusion matrix. Both lengths are reported only as descriptive figures (`descriptive_lengths_m`), never as recall or precision.
- **A property of the matching rule, stated so it is decided knowingly before the freeze.** Matched length is measured from each segment sample to the EMS line. An EMS stretch of length L lying parallel at offset d therefore covers up to about L + 2·√(20² − d²) m of the segment. For an EMS line within a few metres of the road, even a very short positive stretch reaches 25 m. In practice the 25 m minimum filters lateral near-misses and line crossings, not short EMS stretches. A test pins this behaviour, so any change to the rule must be deliberate and recorded in section 11.

**Geometry check before the freeze.** The inspection reports, for each compared product:
- the share of EMS road length within 10, 20, 30 and 50 m of the BATO road network;
- BATO segment lengths, and how many are shorter than 25 m.

It uses geometry only: no BATO road state is read. If most EMS road length lies more than 20 m from the BATO network, the 20 m distance is not suited to the data. A change would then need a reason based on that geometry, made before the freeze, and recorded in section 11.

**Mismatch, stated explicitly.**
- EMS road damage is physical damage seen in optical imagery.
- BATO `BLOCKED` is a ≥ 25 m run of Step 4C FLOOD (open-water-like radar darkening) on the road.
- A road destroyed or buried by a debris flow need not look like open water, so recall measures something BATO's rule was not built to detect.
- The comparison is a case study of agreement, not a validation of BATO's road rule.

## G2. Bridges: context only

`transportationP` (bridge and elevated-road points, `damage_gra`: Destroyed, Possibly damaged) is reported as counts by grade inside `C_k`. **BATO does not currently generate a bridge-damage prediction layer**, so no bridge precision or recall is computed.

## G3. Not used, with reasons

| Layer | Reason |
|---|---|
| `builtUpP` | EMSR927 building damage reference exists, but current BATO production output does not contain a building-impact prediction layer, so building-level validation is not applicable. |
| `facilitiesA`, `facilitiesL` | BATO has no facility-impact production output. |
| `imageFootprintA` | Source-image footprint metadata only. Image times are taken from the activation record. |
| `transportationA` | Airfield and helipad polygons, not part of the road network BATO evaluates. |

Also used: `areaOfInterestA` defines the comparison area (C), and `notAnalysedA` is removed from every comparison area (C).

The following are NOT CALCULABLE in this protocol:
- flood precision, recall, F1 and IoU: there is no EMS flood class;
- bridge precision and recall: there is no BATO bridge layer;
- building precision and recall: BATO has no building-impact output;
- settlement-isolation precision and recall: there is no reference blockage protocol or data.

## 7. Time

- The activation record gives `eventTime`, `activationTime`, the image `acquisitionTime`s and `deliveryTime` **without a time-zone marker**. They are not assumed to be UTC.
- `time.ems_time_zone` may be set to `UTC` or `Asia/Kathmandu` only when the official products or metadata state it. The quoted evidence goes in `ems_time_zone_evidence`.
- While the zone is undetermined, every gap is reported under both readings: UTC, and Asia/Kathmandu (UTC+05:45, no daylight saving).
- The BATO POST time is UTC: it is in the SAFE product name and the stack provenance.
- Reported gaps:
  - EMS image time − BATO POST, for every image of every compared product, in hours (negative when the EMS image is earlier);
  - EMS `eventTime` versus BATO `event_date` (2026-08-26, date only, from the AOI plan). The calendar date of `eventTime` is reported under each reading.

## 8. Integrity and leakage checks (run by the evaluation script)

1. **Leakage scan, before metrics.** The scan covers every provenance/manifest JSON under `data/raw`, `data/interim` and `data/processed`. Two kinds of string are hits:
   - any string that resolves into `data/reference` or `data/eval`, whatever its key;
   - any string naming EMSR\<digit\>, UNOSAT, Copernicus EMS or rapidmapping, except under keys that state what was *not* used (`inputs_not_used`, `validation_note`, `rule(s)`, `description`).

   Hits stop the evaluation.
2. **BATO inputs.** Every BATO file read is SHA-256 hashed before and after the evaluation. Any change stops the evaluation. Nothing under `data/raw`, `data/interim` or `data/processed` is written.
3. **EMS zips.** The zips must match the SHA-256 in `data/reference/EMSR927/acquisition_manifest.json`.
4. **Outputs.** Outputs go to `data/eval/EMSR927/<protocol-sha256[:12]>/`, and every run is appended to `data/eval/EMSR927/runs.jsonl`. A protocol changed after a run therefore produces a visibly different directory.
   - A repeated run under the same frozen protocol is compared with the first, ignoring timestamps. If identical it is logged as reproduced and nothing is overwritten.
   - If different, both files are kept and the run stops with an error.

## 9. Procedure

1. `python scripts/fetch_emsr927.py --manifest <fetch manifest>`: saves the activation record and downloads the selected zips, with provenance.
2. `python scripts/inspect_emsr927.py`: reports layers, CRS, geometry, fields, class values, extents, dates and time-zone clues. It computes no metric.
3. Complete the YAML mappings from the inspection report. Confirm or correct sections A, G, G2 and G3 of this document. Re-run step 2 until "Unmapped items" is 0, and review the road geometry check.
4. `python scripts/evaluate_emsr927.py --freeze-protocol`. It fails if:
   - anything inspected is unmapped;
   - the inspection report does not describe the current zips;
   - metrics already exist under the same `protocol_id`.

   Metric outputs under other protocol ids are listed in the freeze record (disclosure).
5. `python scripts/evaluate_emsr927.py`: computes the metrics.

## 10. Not claimed by this protocol

The protocol makes no claim about:
- FloodViT accuracy;
- Kuro Siwo or unseen-Himalayan performance;
- debris detection (debris-v2 was rejected);
- building, bridge or settlement-isolation precision/recall;
- flood precision, recall, F1 or IoU against EMSR927 (no flood class exists);
- general flood accuracy.

A comparison on one event, against a reference of possibly different class and time, is a case-study comparison, not a validation.

## 11. Revisions before the freeze (disclosure)

| Date (UTC) | Change | Based on | Metrics computed before? |
|---|---|---|---|
| 2026-10-05 | First draft: categories, modes, tolerance, matching distances and product rule | Activation record only | No |
| 2026-10-05 | Mapping completed from the official zips:<br>• `Landslide` → `reference_mass_movement`, no flood class;<br>• road comparison made primary, with a DESTROYED-only primary and a labelled supplementary any-damage variant;<br>• `transportationP` bridges as context only;<br>• layers not used listed with reasons;<br>• road label precedence positive > excluded > negative;<br>• whole-length rule for segments shorter than 25 m;<br>• mass-movement overlap reduced to descriptive shares, with no precision/recall/F1/IoU | Inspection of the downloaded official products (layer names, class and damage values). No BATO result was read. | No |
| 2026-10-05 | Road metrics labelled as conditional on the matched domain. A reference-coverage section was added, so EMS positive roads that match no positive-labelled BATO segment are reported, not dropped. The corridor property of the matching rule is documented. No threshold, distance or matching rule changed. | Review of the metric design (synthetic tests only) | No |

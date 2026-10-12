# Step 4C — deterministic SAR change flood baseline

**Status:** a transparent rule with **provisional thresholds**. It is **not validated**: no accuracy has been measured, and no threshold was tuned or checked against EMSR, UNOSAT or any other reference map.

**Role:** this is the flood layer BATO uses next. FloodViT stays a preliminary, unvalidated layer (see [forensics/step4b_floodvit_forensics.md](forensics/step4b_floodvit_forensics.md)) and is never an input here.

| | |
|---|---|
| Code | `src/bato/detection/baseline.py`, `src/bato/detection/components.py`, `src/bato/contracts/baseline.py` |
| Script | `scripts/run_baseline_detection.py` |
| Thresholds | `config/thresholds.yaml` → `baseline_flood` (plus inherited `detection.min_component_pixels`, `visibility.layover_lia_max_deg`) |
| Output | `data/interim/detection/<aoi>/<candidate>/baseline/` |

## Inputs

All inputs are verified by hash before use:
- **Step 3.5 stack:** PRE1, PRE2 and POST, each with VV and VH, as linear σ⁰ on the EPSG:3857 10 m grid, plus `valid_mask.tif`. The stack checks are the same as in Step 4A.
- **GLO-30 DEM mosaic** that Step 3.4 used for the POST pass. It provides slope and aspect, sampled nearest-neighbour at each 10 m pixel centre.
- **Orbit direction** of the passes, which gives the look azimuth: ascending ≈ 78°, descending ≈ 282°.

Never used: FloodViT outputs (comparison only, afterwards); EMSR, UNOSAT or any damage map; post-event OSM; population data; JRC permanent water.

## Algorithm

With dB = 10·log₁₀(σ⁰) and the thresholds below:

```
water(t)   = VV_t ≤ T_vv  ∧  VH_t ≤ T_vh                                     (open-water-like darkness)
dark(pre)  = (VV_POST − VV_pre ≤ C_pre) ∧ (VH_POST − VH_pre ≤ C_pre)          (new darkening, both pols)
candidate  = water(POST) ∧ dark(PRE2) ∧ dark(PRE1) ∧ ¬(water(PRE1) ∨ water(PRE2))
FLOOD      = candidate whose 8-connected component has ≥ N_min pixels
PRE_EVENT_WATER = water(POST) ∧ (water(PRE1) ∨ water(PRE2))
```

**Pre-existing water is excluded twice:**
1. POST must be darker than **both** pre-event images.
2. Any pixel that was already open-water-dark at PRE1 or PRE2 is `PRE_EVENT_WATER`, even if it got darker still. Calm water after wind-roughened water can pass the change test, so the change test alone is not enough.

### Gating

Gating is applied before the rule, in this order. The first hit sets the state.

| Gate | Condition | State |
|---|---|---|
| invalid SAR | `valid_mask` = 0 (no-data in any of the six layers) | `NOT_ASSESSED_INVALID_SAR` |
| no terrain | no GLO-30 slope/aspect at the pixel | `NOT_ASSESSED_NO_DEM` |
| layover-likely | LIA ≤ `visibility.layover_lia_max_deg` | `NOT_ASSESSED_LAYOVER` |
| shadow-likely | LIA ≥ `baseline_flood.shadow_lia_min_deg` **or cast shadow** (below) | `NOT_ASSESSED_SHADOW` |
| steep | slope ≥ `baseline_flood.max_slope_deg` | `NOT_ASSESSED_STEEP` |

LIA is an **approximate** local incidence angle:

```
sensor_azimuth = look_azimuth + 180°
s_r = atan( tan(slope) · cos(aspect − sensor_azimuth) )      slope component toward the sensor
LIA = θ_nominal − s_r
```

Here θ_nominal = `baseline_flood.nominal_incidence_deg`.

**Cast shadow** (`bato/detection/terrain_gates.py`) is computed on the GLO-30 grid. Looking from each cell toward the sensor (ground azimuth = look azimuth + 180°) for up to `baseline_flood.shadow_search_m`, the cell is shadowed if

```
max_d  (z(d) − z₀) / d  >  cot(θ_nominal)
```

This means terrain rises above the radar ray, which climbs at 90° − θ. It catches **flat ground shaded by nearby ridges**, such as gorge floors, which the slope gate cannot see. Those floors are dark on every date, so without the cast-shadow gate they would come out as `PRE_EVENT_WATER`/`NO_FLOOD`. The provenance reports how many pixels are `NOT_ASSESSED_SHADOW` because of cast shadow alone.

**Gate redundancy at current values.** With `max_slope_deg` = 15°, the *local* layover/shadow tests only relabel pixels that the steep gate would catch anyway: local layover needs a range slope ≥ 34°, local shadow ≤ −46°. Cast shadow is the geometry gate that reaches flat ground. Step 3.4 (locked) saved no SNAP layover/shadow mask, and the stack carries no per-pixel incidence angle. So this approximation is the "existing observability information": the GLO-30 terrain plus the orbit geometry.

## Outputs and encoding

**`baseline_state.tif`** (uint8; every pixel has exactly one state):

| Code | State | Meaning |
|---|---|---|
| 0 | NO_FLOOD | Assessed. Not flood by the rule, including candidates removed by the cluster filter. |
| 1 | FLOOD | Assessed. New open-water darkening against both pre-event images, in a kept cluster. |
| 2 | PRE_EVENT_WATER | Assessed. Water at POST that was already open-water-dark before the event. |
| 10 | NOT_ASSESSED_INVALID_SAR | SAR no-data. |
| 11 | NOT_ASSESSED_NO_DEM | No terrain to gate with. |
| 12 | NOT_ASSESSED_LAYOVER | Approximate layover. |
| 13 | NOT_ASSESSED_SHADOW | Approximate radar shadow. |
| 14 | NOT_ASSESSED_STEEP | Slope ≥ threshold. |

**`baseline_flood.tif`** (uint8): `1` FLOOD; `0` NO_FLOOD or PRE_EVENT_WATER (assessed); `255` NOT_ASSESSED (nodata).

There is **no score raster.** A thresholded rule has no calibrated probability, and an uncalibrated "score" would invite misuse. The provenance instead records the distribution of change in dB for flood pixels.

**`baseline_provenance.json`** records:
- inputs and their hashes, the DEM, and the orbit/look azimuth;
- every threshold, with its value, unit, basis, provisional flag and rationale;
- the rule, the state and flood codes, the output hashes and the statistics;
- the FloodViT comparison, if a FloodViT event output exists;
- `validated: false`.

**`baseline_diagnostics.md`** contains:
- counts per state;
- flood fraction of **assessed** pixels (denominator FLOOD + NO_FLOOD + PRE_EVENT_WATER) and of **all** pixels;
- flood by slope bin and by approximate geometry;
- flood pixels darkened against PRE2 and against PRE1 (all of them, by construction; a self-check);
- change quantiles in dB;
- connected-component statistics, including the share of candidate pixels removed by the cluster filter;
- pre-event OSM waterway overlap: **NOT AVAILABLE**. Step 4C does not read OSM (the fixed text of this provenance field was written before `bato.osm` existed and still says "not implemented"), and no other waterway source is substituted. Pre-event OSM is used later, for roads and settlements ([connectivity.md](connectivity.md)).
- the FloodViT comparison: both, baseline-only, FloodViT-only on assessed pixels, Jaccard, and FloodViT flood inside NOT_ASSESSED.

## Thresholds

| Name | Value | Basis | Why |
|---|---|---|---|
| `baseline_flood.vv_water_db` | −15 dB | physics / literature | Calm water is a specular reflector. Fixed open-water VV thresholds commonly used for Sentinel-1 lie roughly between −15 and −20 dB; −15 is the permissive end, constrained by the VH and change tests. |
| `baseline_flood.vh_water_db` | −22 dB | physics / literature | Commonly used VH thresholds lie roughly between −20 and −25 dB. This is close to the IW noise floor, so it confirms darkness rather than resolving it. |
| `baseline_flood.pre2_change_db` | −3 dB | provisional engineering | A factor-2 decrease in VV **and** VH since the most recent pre-event image. |
| `baseline_flood.pre1_change_db` | −3 dB | provisional engineering | The same against the earlier pre-event image, so the darkening is new against both. |
| `baseline_flood.max_slope_deg` | 15° | provisional engineering | Standing water is horizontal. A 30 m DEM smooths gorge floors. Same value as the Step 4B forensic criterion. |
| `baseline_flood.shadow_lia_min_deg` | 85° | physics / literature | Geometric shadow occurs at LIA ≥ 90°; the 5° margin mirrors the layover margin. |
| `baseline_flood.nominal_incidence_deg` | 39° | provisional engineering | IW incidence spans about 29–46° across the swath; no per-pixel incidence is available. |
| `baseline_flood.shadow_search_m` | 5000 m | provisional engineering | Ray length for cast shadow. A ridge of relief H shades about H·tan θ ≈ 0.8 H, so 5 km covers relief well above the AOI's. |
| `visibility.layover_lia_max_deg` | 5° | inherited BATO design | Existing visibility threshold. |
| `detection.min_component_pixels` | 10 px | inherited BATO design | Existing speckle-cluster threshold, 8-connected (≈ 0.1 ha at 10 m map spacing). |

**Change policy:** values may only be changed using training or validation data, or control scenes. Never against EMSR927 or any reference map (rule at the top of `thresholds.yaml`). Every change is recorded in the config fingerprint and in each provenance.

## Why NOT_ASSESSED exists

A dark SAR pixel on a 30° slope, in radar shadow or in layover tells us about the viewing geometry, not about water. Calling such pixels `NO_FLOOD` would claim roads there are passable when the satellite could not see them.
- `NOT_ASSESSED` is a third answer. Downstream stages must carry it as "unknown"; the road-impact contract already has `UNOBSERVABLE`.
- In steep Himalayan terrain a large NOT_ASSESSED share is expected. It is reported, not hidden.

## Why FloodViT is not the baseline

The Step 4B forensics showed that on this scene:
- **Terrain:** FloodViT's flood concentrates on steep and shadowed terrain. 64 % of it lies on slopes ≥ 30°, and shadow-likely terrain is over-represented 8.6×.
- **Timing:** a third of it appears even when POST equals PRE2, and most of it overlaps change that already existed before the event.
- **Inputs:** VV saturates the Kuro Siwo clamp on about half the scene.
- **Training domain:** it was trained on lowland floods, with no Himalayan samples.

It is therefore kept as a preliminary comparison layer only.

## Limitations and assumptions

- **Thresholds are provisional.** No accuracy is claimed. The map is a transparent rule output.
- **Open water only.** Flooded vegetation (bright double-bounce) and urban flooding are **not** detected, and neither is water under canopy.
- **Wind-roughened or turbulent water** (rapids, choppy flood water) can exceed −15 dB VV and be missed.
- **Water-like dry surfaces** (smooth tarmac, sand) are excluded only if they were also dark before the event. A new dark surface such as fresh sediment can look like new water.
- **Approximate geometry.** The look azimuth (heading ±12°), the nominal incidence and the 2-D range-slope approximation make layover/shadow approximate.
  - Cast shadow is modelled. **Layover cast onto neighbouring flat ground is not**: a valley floor overlaid by the return of a slope facing the sensor stays assessed, usually bright, so `NO_FLOOD`.
  - A SNAP layover/shadow mask would be better but needs a Step 3.4 change, and that stage is locked.
- **Fail-closed values.** NaN or negative values, or a mask that disagrees with the data, stop the run (the Step 4A consistency check, applied to every block).
- **DEM.** GLO-30 is sampled nearest-neighbour at 10 m. Narrow gorge floors can read as steep and so become NOT_ASSESSED.
- **Monsoon background.** Seasonal water appearing between PRE2 and POST that is not caused by the event is flagged as flood. The matched pre-event control (2026-08-04 / 2026-06-17 with POST′ 2026-08-16) is the planned way to measure this.
- **Waterways.** The pre-event OSM overlap is not available until `bato.osm` exists.

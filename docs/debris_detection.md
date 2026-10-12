# SAR disturbance (debris) detection — EXPERIMENT, REJECTED (not part of the official result)

> **Status: rejected.** The detector was run on the real event and on the matched no-event control and **failed its pre-registered acceptance criterion (d), "no edge concentration", on the control run**: the control's DISTURBANCE share in the 100 px border band was 0.23445, against an assessed share of 0.060758 in that band. Under the pre-registered rule a failure means the detector is unusable, so it was frozen as **not used**:
>
> - it is **not** part of the official BATO detection, road impact, connectivity, dashboard result or EMSR927 comparison;
> - road rule v2, which consumes it, is not the official rule (the official result is road rule v1, see [connectivity.md](connectivity.md));
> - the dashboard shows it only under *Methodology → Experimental methods*, where the checks are re-evaluated from the event and control provenance.
>
> Nothing was re-tuned after the result (no other z, no edge removal, no criterion change). **BATO makes no claim that debris was detected.** For the record (from the two provenance files): event 8,714 DISTURBANCE pixels = 0.0221 % of assessed; control 418 = 0.0011 % (criteria (a) and (b) pass); control facing-sensor share 0.497608 (criterion (c) passes); criterion (d) fails on the control.

| Part | Where |
|---|---|
| Detector | `src/bato/debris/disturbance.py`, `scripts/run_debris_detection.py` |
| Contract | `src/bato/contracts/debris.py` (`DebrisState`, `NoiseModel`, `DebrisStatistics`, `DebrisProvenance`) |
| Road rule v2 (consumer) | `src/bato/impact/roads.py` (`StateSampler(debris=...)`), `src/bato/connectivity/engine.py` |

This is change statistics, not a model. None of the permitted training datasets has debris labels. The sections below document what the experiment did.

## Inputs (already acquired BATO data only)

- The verified Step 3.5 stack and `valid_mask.tif`, checked against the Step 3.5 provenance.
- The Step 4C run made from that exact stack. Its stack SHA-256 must match, and its state raster and DEM hashes are re-checked.
  - Its state raster supplies the geometry gates and FLOOD / PRE_EVENT_WATER.
  - Step 4C code and thresholds are unchanged.
- The Step 4C DEM, used only for the slope and range-slope statistics in the diagnostics.

Not used: any damage or flood map (EMSR, UNOSAT), post-event OSM, population data, FloodViT outputs, or any trained model.

## Rule

The rule is applied per polarisation p ∈ {VV, VH}, to Lee Sigma-filtered σ⁰ in dB. All three images come from the same relative orbit.

```
d1 = POST − PRE1        d2 = POST − PRE2        d0 = PRE2 − PRE1
sigma_p    = 1.4826 · MAD(d0)   over valid, geometry-observable pixels (recorded in the provenance)
increase_p = (d1 ≥ z·sigma_p) ∧ (d2 ≥ z·sigma_p) ∧ (|d0| < z·sigma_p)
candidate  = increase_VV ∧ increase_VH ∧ ¬FLOOD ∧ ¬PRE_EVENT_WATER
DISTURBANCE = candidate in an 8-connected cluster of ≥ detection.min_component_pixels
```

- `z = detection.debris_z_min = 3.0` and `detection.min_component_pixels = 10` are the existing provisional thresholds. They are unchanged and were not tuned.
- The noise model is estimated from the scene's own pre-event pair, a constant-false-alarm-style null model. If MAD = 0 the run fails closed.
- Both-polarisation darkening is counted as a diagnostic (`darkening_diagnostic_pixels`) and never becomes a claim.

## Observability

| code | state | source |
|---|---|---|
| 0 | NO_DISTURBANCE | assessed, no qualifying change (includes Step 4C STEEP pixels) |
| 1 | DISTURBANCE | assessed, rule above |
| 10 | NOT_ASSESSED_INVALID_SAR | Step 4C code 10, or invalid in the mask |
| 11 | NOT_ASSESSED_NO_DEM | Step 4C code 11 |
| 12 | NOT_ASSESSED_LAYOVER | Step 4C code 12 |
| 13 | NOT_ASSESSED_SHADOW | Step 4C code 13 (local or cast shadow) |

Step 4C's STEEP gate (code 14) is a water-ponding plausibility gate, not a visibility gate. Steep pixels are therefore assessed here. All three images share one track, so foreshortening is the same on every date and cancels in d1, d2 and d0. Layover and shadow are not assessed.

## Outputs

These are written to `data/interim/detection/<aoi>/<candidate_id>/debris/` and published atomically. `--overwrite` keeps the old output as `debris.superseded-<time>`.

| file | content |
|---|---|
| `debris_state.tif` | uint8 `DebrisState`; GeoTIFF tags record sigma VV and VH |
| `change_table.npz` | every DISTURBANCE and Step 4C FLOOD pixel: `row, col, kind (1 flood, 2 disturbance), d1_vv, d2_vv, d0_vv, d1_vh, d2_vh, d0_vh, slope_deg, s_r_deg` |
| `debris_provenance.json` | inputs and hashes, Step 4C reference, noise models (median d0, sigma, pixel count), thresholds, rule, statistics, `validated: false` |
| `debris_diagnostics.md` | states, clusters, slope and geometry breakdown, pre-registered checks |

## Pre-registered checks

These are reported only and nothing is tuned. A failure means the detector is reported as unusable for that scene. **Outcome:** (d) failed on the matched control, so the detector is unusable (see the status note at the top).

1. **(a) Control ceiling:** on the matched no-event control (`aoi-f2cc742b0e25`), DISTURBANCE ≤ 0.05 % of assessed pixels.
2. **(b) Event contrast:** the real-event DISTURBANCE fraction is at least 3 × the control fraction.
3. **(c) Geometry:** no more than 50 % of DISTURBANCE pixels face the sensor (range slope ≥ 10°). A warning is written if this fails.
4. **(d) No edge concentration:** the DISTURBANCE share in the 100 px border band should not exceed the assessed share in that band.

## Road rule v2 (connectivity input; EXPERIMENTAL, REJECTED)

Not the official rule. `scripts/run_connectivity.py` still defaults to `--road-rule v2` in code, so the official result must be reproduced with `--road-rule v1` ([connectivity.md](connectivity.md)). Under v2 the script samples both rasters along each road:

- A sample is **affected** if FLOOD (Step 4C) or DISTURBANCE lies within `road_impact.road_buffer_m`.
- A sample is **unobservable** only if it is not affected and is geometry-unobservable (debris state 10–13) or outside the raster.
- Otherwise the sample is **observed open**.

The blocked-run (25 m) and hidden-run rules, the three graphs, routing, the settlement decision table and the stretch ranking are unchanged. `--road-rule v1`, the official rule, keeps every Step 4C NOT_ASSESSED state, including STEEP, unobservable.

Each BLOCKED or UNOBSERVABLE road-segment evidence record carries:

- the claim text, including z, sigma VV/VH, and the PRE-1/PRE-2/POST dates;
- values:
  - `z`, `sigma_vv_db`, `sigma_vh_db`;
  - `d1/d2/d0_{vv,vh}_db_{median,min,max}` of the change pixels inside the road buffer;
  - `change_pixels_disturbance` and `change_pixels_flood`;
  - `step4c_samples_<STATE>`;
  - `geometry_observable_samples` and `geometry_unobservable_samples`;
- sources: the Sentinel-1 products (source stack), the Step 4C state, the disturbance state, and the pre-event OSM way;
- artifact hashes for both rasters and the change table.

## Limitations

- **Rejected.** It failed pre-registered criterion (d) on the matched control and is not used anywhere in the official result.
- **Not validated.** No reference map was used to set or check anything.
- **Coverage gaps remain.** Brightening also comes from wet snow melt, new structures, crop and vegetation change, and moisture. The two-pre-image stability test and the cluster filter reduce, but do not remove, these false alarms. The control run measures the background rate.
- **Some debris will be missed.** Debris that is darker than the surface it covers is not detected, because darkening is diagnostic only.
- **Blind spots stay unobservable.** Layover and shadow slopes are reported as NOT_ASSESSED, never as no disturbance.

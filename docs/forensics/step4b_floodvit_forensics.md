# Step 4B forensics: FloodViT event vs no-event on Trishuli (2026-10-03)

**Status: 1 — TECHNICALLY CORRECT / MODEL BEHAVIOR CONCERN.**
- No code bug was found in the Step 4B inference.
- No mismatch was found between the Step 4A input contract and the Kuro Siwo training loader.
- The current flood map is kept only as a **preliminary, unvalidated model output**. It must **not** feed road/settlement connectivity until the control runs and diagnostics below have been run and pass the pre-registered criteria (§9).

Evidence labels:

| Label | Meaning |
|---|---|
| **CODE** | Verified in the BATO code / tests. |
| **KURO-DATA** | Computed from the grid pickles shipped in the Kuro Siwo repository (commit `4347ed1`, 67,490 samples). |
| **USER-RUN** | Numbers from the user's real Trishuli run (not re-measured here; the outputs were not reachable from this session). |
| **DERIVED** | Arithmetic from the above. |
| **TO MEASURE** | Produced by `scripts/diagnose_floodvit.py` on the real outputs. Nothing in this report invents those values. |

No EMSR, UNOSAT or other reference/evaluation data is used anywhere, as input or as a comparison.

## 1. Inputs of each mode (CODE)

| Model channel | event | no_event | zero_change (new) | static_post (new) |
|---|---|---|---|---|
| 0–1 POST (flood) | POST 2026-08-28 | **PRE2** 08-16 | **PRE2** 08-16 | POST |
| 2–3 PRE2 (sec1, later pre) | PRE2 08-16 | **PRE1** 06-29 | PRE2 08-16 | POST |
| 4–5 PRE1 (sec2, earlier pre) | PRE1 06-29 | PRE1 06-29 | PRE1 06-29 | POST |

`no_event` is exactly POST ← PRE2, PRE2 ← PRE1, PRE1 ← PRE1, as specified. This is tested in `tests/unit/test_floodvit_diagnostics.py::test_no_event_construction_is_exactly_post_from_pre2_and_both_pres_from_pre1`.

**Temporal signal it creates:**
- The "post" image is 2026-08-16. Both "pre" images are the *same* 2026-06-29 acquisition.
- The model therefore sees a 48-day change, from late June to mid-August, against a pre pair with no internal variability.

Two consequences:
1. **The pre pair is out of distribution (KURO-DATA).** In Kuro Siwo, sec1 ≠ sec2 in **100 %** of samples, and their gap has a median of 12 d (5th–95th percentile 12–30 d). A pair of identical pre images never occurs in training. Gap 0 also removes the model's only cue for "normal" pre-to-pre variability, so any post-vs-pre difference can look anomalous.
2. **The 48-day interval holds real seasonal surface-water change (hypothesis, TO MEASURE).** June 29 to August 16 spans monsoon onset to monsoon peak in central Nepal. Rivers widening, saturated soils and flooded paddy all add genuine new water or darkening between those dates. The event triple's post-vs-sec1 gap is only 12 days, both at monsoon peak.

**Conclusion:** no-event flood (0.578 %) exceeding event flood (0.293 %) is **possible and physically expected** for a change-sensitive model. It is **not a bug**. It does mean that `no_event`, as built, is **not a clean false-positive baseline**. The earlier description, "a sensible change model should predict (almost) no flood there", was wrong and has been corrected in `docs/inference.md`.

## 2. Is a ~2× higher no-event flood fraction a bug? (CODE + reasoning)

- **Construction:** verified (§1).
- **Masking:** valid = 45,256,776 / 45,256,776 and 0 invalid pixels (USER-RUN). The stack mask is all-valid inside this window, so invalid-pixel handling plays no role.
- **Merging:** shift-to-fit tiling, logit averaging and argmax were verified against brute force in the previous audit.
- **No bug.** The ratio reflects what the model is shown (§1), not an implementation error.

## 3. Kuro Siwo → BATO input distribution

### 3a. VV clamp saturation: a proven distribution difference (DERIVED)

- **What the constants allow.** Kuro Siwo's model-normalisation constants (VV mean 0.0953, std 0.0427) bound how many Kuro Siwo pixels can lie above the 0.15 clamp. By Cantelli's inequality, **at most 37.9 % of VV pixels** can exceed 0.15 for *any* distribution with those moments (for VH, at most 2.9 %). For a unimodal distribution the real share is much lower.
- **What Trishuli shows.** The user-reported Trishuli share is **≈ 50 % of VV** above the clamp.
- **Conclusion.** If the constants describe Kuro Siwo's pixels (the repository comment says "mean and std for all of Kuro Siwo"), half of the Trishuli VV inputs sit in a region Kuro Siwo's training data could not have filled. That half is flattened to one value (+1.28 after normalisation).

### 3b. Terrain or radiometry? (TO MEASURE)

`diagnose_floodvit.py` now reports the clamp share on **flat terrain only** (GLO-30 slope < 5°).
- **Flat-terrain VV above-clamp share ≤ 37.9 %, roughly Kuro-like:** the saturation is driven by terrain (slopes facing the sensor, foreshortening). That is **domain shift**.
- **Flat-terrain share also ≫ 37.9 %:** BATO's σ⁰ differs from Kuro Siwo's even where terrain cannot explain it. The status would change to **3 — MODEL INPUT MISMATCH**.

### 3c. Other comparisons (TO MEASURE)

Per-date VV/VH quantiles, model-normalised mean/std per channel (in-domain ≈ 0 / 1), PRE1→PRE2 and PRE2→POST change in dB, and VH−VV ratios.

## 4. Where the predicted water falls (TO MEASURE)

For the event, each control mode, and all valid pixels as the base rate, the diagnostics report the share of class 2 (flood) and class 1 (permanent water) in each of these:
- POST VV above or near the clamp, and any-date VV above the clamp;
- VV darkening ≥ 3 dB (the open-water signature) or brightening, PRE2→POST;
- slope bins;
- height above the local minimum within ~1 km (a valley-floor proxy; **not** a drainage model);
- approximate radar geometry (facing sensor / layover-likely / facing away / shadow-likely), from slope and aspect versus the ascending look direction (~78°) at a nominal 39° incidence. **This is an approximation, not a SNAP layover/shadow mask.**
- within 16 px of an internal tile edge;
- connected-component sizes;
- overlap with the event flood.

**"Near known waterways" is NOT AVAILABLE.** BATO has no pre-event OSM extraction yet. The valley-floor proxy stands in and is labelled as a proxy.

**Already suspicious (DERIVED from USER-RUN):**
- **Permanent water** (class 1) covers 4.65 % of the scene. That is ≈ 2.10 M px at 100 m² of map area, about **164 km² on the ground** at 28° N (≈ 203 km² in no-event).
- **Kuro Siwo comparison:** Kuro Siwo's tiles, all selected around lowland floods, average 2.85 % permanent water per tile, with a median of 0 (KURO-DATA).
- **Likely cause:** a Himalayan scene with more "permanent water" than Kuro Siwo's flood-site average is implausible unless large water bodies exist in the window. The likely candidates are radar shadow and very dark steep slopes, which stay dark on all dates. The diagnostics measure this with the shadow-likely / facing-away and slope shares of class 1.

## 5. Seams (DERIVED from USER-RUN)

| | flood near internal edges | interior | ratio | edge-band share of area | edge-band share of flood |
|---|---|---|---|---|---|
| event | 0.340 % | 0.276 % | **1.23** | 26.8 % | 31.1 % |
| no_event | 0.657 % | 0.549 % | **1.20** | 26.8 % | 30.5 % |

- **Size:** the edge bands hold about 20 % more flood than tile interiors. In the event run that is ≈ 7.8 k excess pixels, about 6 % of its flood.
- **Consistency:** the effect is consistent across both modes, so it is systematic, not content-driven.
- **Likely cause:** less spatial context for ViT tokens near tile borders. Merging is verified, so this is not a merging bug.
- **Severity:** moderate. It is below the 1.5 warning threshold but not negligible.
- **Test:** re-run with `--stride 112`, which gives every pixel a tile where it is ≥ 56 px from the border, then compare flood maps and seam ratios. The earlier output is kept as `floodvit.superseded-*`.

## 6. Checkpoint ↔ preprocessing compatibility

| Aspect | Status |
|---|---|
| Architecture / weights structure | Verified: 205,538,499 params, 280 keys, strict load. This shows the structure only, not the input domain. |
| Channel order, normalisation, no DEM | Verified against the Kuro Siwo training loader (Step 4A). |
| Training recipe of these exact weights | **Not determinable.** The fork release from PR #4 has no training config. |
| Released Kuro Siwo pixel units | **Not determinable here.** No sample has been measured yet. |
| VV value distribution | **Differs** (§3a): proven, assuming the constants describe Kuro Siwo. |
| Terrain domain | **Out of training domain** (KURO-DATA): no Cold/Polar climate zone; the only "Nepal" activation is AOI "Patna" (Ganges plain), in the *test* split; no training activation is named for a mountain region. |

**Strict loading proves the tensors fit, not that the model generalises to Himalayan SAR.**

## 7. Specific factors

- **Land-Sea-Mask** (Kuro Siwo has it; BATO omits it). In Kuro Siwo it only blanks sea pixels (`landMask false`). Trishuli is inland, so there is **no effect on this scene**.
- **SRTM (Kuro Siwo) vs GLO-30 (BATO) for terrain correction.** Both use `applyRadiometricNormalization false`, so the DEM affects **geocoding**, not σ⁰ values. Expect geolocation differences of tens of metres on steep slopes. This **does not explain** saturation or the class-1 excess.
- **≈ 50 % VV saturation.** Proven difference (§3a); whether it is terrain or radiometry is decided by §3b.
- **Himalayan radar geometry.** Slopes facing the sensor are foreshortened or laid over, so bright and saturated. Slopes facing away are dark or in shadow on every date. Both regimes are rare in Kuro Siwo's lowland sites. Shadow is a plausible source of false "permanent water" (TO MEASURE).
- **Orbit / temporal.** All three passes are S1D, ascending, relative orbit 85, at the same time of day, so geometry is consistent across dates. Kuro Siwo's orbit consistency is not recorded in its pickles.

## 8. Temporal spacing (KURO-DATA)

| Gap | Trishuli | Kuro Siwo median (p5–p95) | Kuro Siwo share like Trishuli |
|---|---|---|---|
| post − sec1 | 12 d | 84 d (12–228) | 23 % ≤ 12 d (common) |
| post − sec2 | 60 d | 96 d (24–240) | 11 % in 48–72 d |
| sec1 − sec2 | **48 d** | **12 d (12–30)** | **2.8 %** in 36–60 d |
| all three together | — | — | **≈ 1 %** of samples |

The post gap is typical, but a **48-day spacing between the two pre images is rare** in training. Most Kuro Siwo pre pairs are 12 days apart. This is a secondary distribution difference.

## 9. What would make the map usable: pre-registered criteria

Run:

```powershell
python scripts/run_floodvit_inference.py --manifest <manifest> --checkpoint <pt> --kuro-repo <KuroSiwo> --mode controls --overwrite
python scripts/diagnose_floodvit.py --manifest <manifest>
```

The `event` and `no_event` outputs already exist, so `--mode controls` adds `zero_change` and `static_post`.

The map stays **unusable for connectivity** if any of these hold. They are heuristics fixed before seeing the results, not validated thresholds:

| Criterion | Meaning if it fails |
|---|---|
| ≥ 50 % of event flood is also flood in `static_post` | Flood is driven by single-date appearance, not change. |
| `zero_change` flood fraction ≥ 50 % of event | Flood is not driven by the post-vs-sec1 change. |
| < 50 % of event flood shows ≥ 3 dB VV darkening PRE2→POST | Most flags lack the open-water signature. |
| > 25 % of event flood on slope ≥ 15° or ≥ 50 m above the local minimum | Physically implausible flood locations. |
| > 25 % of class 1 in shadow-likely / facing-away geometry | The permanent-water class is unreliable (it is also excluded from road impact either way). |
| Flat-terrain VV above-clamp share > 37.9 % | Input/radiometric mismatch: status changes to 3. |

### Strongest further baseline (requires two more downloads, decided by the user)

This is a real *pre-event-only* triple with the event's spacing. Run Steps 2A–3.5 with a **pseudo event date** between 2026-08-04 and 2026-08-16:
- POST′ = 2026-08-16;
- sec1′ ≈ 2026-08-04 (12 d earlier);
- sec2′ ≈ 2026-06-17 (60 d earlier, if acquired on track 85).

This is in distribution for the model: different pre images, realistic gaps. Its flood fraction is the honest false-positive baseline for the event map. It cannot rule out real monsoon water change before the event.

## 10. Decision

**Retain** the current event map **only as a preliminary, unvalidated model output** (as already labelled), and **do not use it for connectivity yet.**
- **No patch to the model:** there is nothing defensible to fix without data, and no fake fix.
- **Next steps:** run the controls, the diagnostics, and ideally the pseudo-event triple, then decide by §9.
- **If §9 fails**, the honest outcome is that the pretrained Kuro Siwo FloodViT is **not usable on Himalayan Sentinel-1 data without adaptation**. BATO would then fall back to its change-statistics baseline (`DetectionMethod.BASELINE_CHANGE`) with explicit uncertainty.

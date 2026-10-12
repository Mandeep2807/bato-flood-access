# Limitations

## Limitations of the submitted result

**What the result is.** One event (Trishuli, Nepal, August 2026), one Sentinel-1 track (S1D ascending, relative
orbit 85), one PRE-1 / PRE-2 / POST stack. The official flood layer is the Step 4C SAR-change baseline; the official
access result is the road-rule v1 connectivity run built on it. Nothing in the official result was tuned on a
reference map.

**Step 4C flood baseline**
- It is a transparent rule with **provisional** thresholds (`config/thresholds.yaml`, each with a written
  rationale). Its provenance says `validated: false`: no accuracy has been measured.
- It detects **new open-water-like radar darkening** only. Mud, debris, sediment, landslide scars and structural
  damage are not detected.
- **Most of the scene is NOT_ASSESSED.** About 91.2 % of the Trishuli scene (41,285,907 of 45,256,776 pixels) is
  NOT_ASSESSED. The breakdown below gives shares of the not-assessed pixels, not of the whole scene:
  - 15° slope gate: 35,544,186 pixels (about 86.1 %). The open-water rule is not applied on steep slopes.
  - Layover: 4,713,918 pixels (about 11.4 %).
  - Shadow: 1,027,803 pixels (about 2.5 %).
  - Invalid SAR and missing DEM: 0 pixels.

  NOT_ASSESSED is never reported as "no flood".

**Roads and settlements (road rule v1)**
- BLOCKED means a ≥ 25 m run of Step 4C FLOOD on the road; OPEN means no such signal on observable road, **not**
  "passable". Most road length is UNOBSERVABLE because of the slope gate.
- Consequently most settlements are POSSIBLE_CUTOFF (**access unverified**): BATO cannot confirm their access. This
  is **not** a statement that they are cut off. CONFIRMED_CUTOFF is used only when no route exists even if every
  unobservable road is passable.
- BATO produces **no** building-, bridge- or facility-impact prediction. Bridges carry only an
  exposed / not-exposed flag, never a damage or collapse state. Building counts are a proxy, never population.

**Experiments that are not part of the result**
- The SAR disturbance (debris) detector and road rule v2 were **rejected**: the detector failed pre-registered
  criterion (d) on the matched no-event control ([debris_detection.md](debris_detection.md)). BATO makes no debris
  claim.
- FloodViT (Kuro Siwo checkpoint) is preliminary and not validated ([inference.md](inference.md)); it is not part
  of the official result and no model accuracy is claimed. No flood model was trained.

**EMSR927 comparison (evaluation only, frozen protocol `bato-emsr927-v1`)**
- **Cross-hazard reference.** The official EMSR927 products map the event as a *Landslide* (6-Mass Movement) plus
  graded road, bridge and building damage. They contain no open-water flood class, so **flood precision / recall /
  F1 / IoU are NOT CALCULABLE**. The Landslide footprint is never mapped to flood; its overlap with BATO FLOOD is
  reported only as a descriptive, class-mismatched footprint overlap.
- **Road metrics are matched-domain.** The primary comparison (BATO BLOCKED vs EMSR927 road DESTROYED) counts BATO
  segments that receive a valid EMS reference label under the frozen 20 m / 25 m matching rule. Its recall is **not
  network-wide recall**; EMS damaged road that matches no such segment is reported separately as reference coverage.
  The supplementary any-damage comparison is labelled "not accuracy, not ground truth".
- **Class mismatch on roads.** EMS road damage is physical damage seen in optical imagery; BATO BLOCKED is open-water
  darkening on the road. A road destroyed or buried by a debris flow need not look like open water.
- **Time.** EMS imagery and the BATO POST pass were acquired on different dates, and the EMS timestamps carry no
  time-zone marker; gaps are reported under both UTC and Nepal-time readings.
- **One event.** It is a case-study comparison, not a validation, and supports no general accuracy claim. Building,
  bridge, facility and settlement-isolation precision/recall are not calculable.

**Engineering**
- Multi-slice passes are not mosaicked (single-slice passes only). Offline (long-term archive) Sentinel-1 products
  cannot be ordered by BATO.
- Catalogue footprints crossing the antimeridian are rejected.
- Pair ranking is a provisional PAIRING HEURISTIC, not validated against any outcome.
- `scripts/run_connectivity.py` defaults to the rejected `--road-rule v2`; the official result must be reproduced
  with `--road-rule v1` (README).

## General limitations of the approach

**Observation**
- Sentinel-1 revisits each track every few days. BATO describes the state at the post-event pass; it cannot warn of a sudden glacier collapse minutes ahead, and it cannot see what happened between passes.
- In steep valleys, radar shadow and layover hide parts of the terrain. Those pixels are marked unobservable and propagate as uncertainty, never as "no damage".
- Sentinel-2 is often unusable for weeks under monsoon cloud. BATO must work from Sentinel-1 alone.
- Events before routine Sentinel-1 operations (Oct 2014), or areas without a same-track pre/post pair, cannot be processed by change detection.

**Detection**
- ~10 m pixels cannot resolve narrow roads, footbridges or structural state. A bridge can be reported as *exposed to the impact zone*, never as collapsed.
- No debris detection is part of the result (the change-statistics experiment was rejected). Backscatter change from construction, vegetation, moisture or landslides can look like debris, and none of the permitted training datasets has debris labels.
- No flood model was trained. The preliminary FloodViT checkpoint comes from Kuro Siwo, which is mostly lowland floods; its mountain generalisation has not been measured.
- Terrain correction uses Copernicus DEM, while Kuro Siwo was processed with SRTM: a possible domain shift for the (non-official) FloodViT layer, not checked.
- SNAP converts the Copernicus DEM's EGM2008 geoid heights to ellipsoid heights with EGM96. The EGM96/EGM2008 difference (usually below ~1 m, a few metres in high mountains) slightly shifts terrain-corrected positions in steep terrain (a few metres, under one 10 m pixel for typical values).
- Terrain-corrected rasters are in EPSG:3857 (as Kuro Siwo): pixels are 10 m in map units, ≈ 8.8 m on the ground at 28° N. Distances must be computed after reprojection or with the scale factor.

**Infrastructure and access**
- Roads, bridges, buildings and facilities come only from pre-event OSM. Missing or misaligned OSM features lead to missed or false conclusions.
- Building counts are not population. BATO makes no population or casualty estimates.
- Road state is inferred from open-water-like darkening along the road, not observed directly. No flood depth is estimated (a 30 m DEM cannot support it in steep valleys).
- Settlements with no mapped road before the event are reported separately, not as flood-isolated.

**Use**
- Educational prototype, not an operational tool. Outputs need expert verification before any real-world use.

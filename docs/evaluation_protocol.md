# Evaluation plan (original planning document)

> **Read this first.** This is the evaluation *plan* written before any reference data was seen. What was actually done is:
>
> - **EMSR927 (implemented, frozen):** the Trishuli case-study comparison follows its own pre-registered, frozen protocol, [`docs/evaluation/emsr927_protocol.md`](evaluation/emsr927_protocol.md) (`bato-emsr927-v1`). It **supersedes every EMSR927 item below.** The official EMSR927 products contain no open-water flood class (the observed event is a *Landslide*, 6-Mass Movement), so **flood precision / recall / F1 / IoU against EMSR927 are NOT CALCULABLE** and none is reported. The primary comparison is road damage: BATO BLOCKED vs EMSR927 road DESTROYED, as **segment-level metrics over the matched domain** (not network-wide recall), with reference coverage reported separately. Building, bridge, facility and settlement-isolation precision/recall are not calculable (BATO has no corresponding output or reference protocol).
> - **Not implemented:** the Kuro Siwo test-event evaluation, the Himalayan holdout, ablations, calibration, the reference isolation list (b) and the runtime benchmark. No FloodViT or model accuracy is claimed.

The official weights (PS):

| Weight | Criterion |
|---|---|
| 30 % | Flood and damage mapping quality |
| 25 % | AI component |
| 20 % | Cut-off settlement analysis |
| 15 % | Report, understanding in Q&A, honesty about limitations |
| 10 % | Dashboard and situation report usability |

## Leakage rules (apply now)

1. Reference data (Copernicus EMS incl. **EMSR927**, UNOSAT, any published damage map, post-event OSM) lives only in `data/reference/` or `data/eval/` and is read only by `bato.evaluation`.
2. Thresholds and model choices are tuned on training/validation data (Kuro Siwo validation events), **never** on EMSR927 or other reference maps.
3. Evaluation events are excluded from training. Kuro Siwo includes a Nepal event; if it is in mountain terrain it must be held out to support any "unseen Himalayan" claim.
4. Metric scripts record the config fingerprint and model version they evaluated.

## Planned metrics (original plan; see the note at the top for what applies)

**Mapping (30 %)** *(superseded by the frozen EMSR927 protocol)*: the plan was pixel IoU, precision, recall and F1 against an EMSR927 flood class, strict and ±1-pixel tolerant, over all and over observable pixels, plus road-km and building-level precision/recall, with the EMS-vs-Sentinel-1 date gap recorded. EMSR927 has no flood class, so the pixel flood metrics are NOT CALCULABLE; the frozen protocol replaces them with the matched-domain road-damage comparison and a descriptive, class-mismatched mass-movement footprint overlap. Building metrics are not calculable. The date gap is reported.

**AI (25 %)**: Kuro Siwo official test events (flood-class F1/IoU, compared with the published baseline); Himalayan holdout (Trishuli + other mountain events, never trained on); ablations (baseline vs model, single vs temporal input, with/without slope, visibility mask, debris module); calibration (reliability diagram, ECE); runtime per km² on CPU and GPU.

**Cut-off (20 %)**:
- (a) Graph-logic unit tests on hand-built graphs, including the PS's villages A–D example.
- (b) *(not implemented)* Run the connectivity engine on road blockages derived from the EMS reference (evaluation only) to build a reference isolation list; report precision/recall of our confirmed + possible lists. No reference blockage protocol exists, so settlement-isolation precision/recall is not claimed.
- (c) Sensitivity of results to road-impact thresholds.

**Limitations (15 %)**: quantified — % of AOI unobservable, OSM completeness indicators, revisit gap — plus a failure gallery.

**Usability (10 %)**: time to first map and to full report; data dates and data age on every output.

## Where results appear

- **Dashboard:** the official v1 result only: % observable, road km by state, settlement counts by access state (connected / access unverified / confirmed cut off / no pre-event road access), data dates. The dashboard does not show EMSR927 numbers and claims no validation.
- **EMSR927 comparison:** `data/eval/EMSR927/<protocol-sha12>/metrics.{json,md}`: the matched-domain road-damage segment metrics with their reference coverage, the descriptive mass-movement overlap, bridge counts (context only) and the time gaps. No flood F1 against EMSR927 is reported anywhere: it is not calculable, because the reference has no flood class.
- **Not produced:** model F1 on unseen Himalayan events, ablations and calibration plots (not implemented).

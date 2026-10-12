# Same-track temporal-set selection (STEP 2B)

```
CatalogueSearchResult (Step 2A, accepted scenes only)
  → AcquisitionPass per satellite pass (slices of one overpass merged)
  → per track (orbit direction, relative orbit): candidate sets PRE-1 + PRE-2 + POST
  → PAIRING HEURISTIC score → PrePostPair validation → ranked top-N per track
  → TemporalSelectionResult
```

Entry point: `bato.acquisition.temporal_selection.select_temporal_sets(catalogue, cfg)`.
CLI: `python scripts/select_temporal_sets.py --catalogue <file from s1_catalogue_query.py --save> [--top N] [--save out.json]`.

No imagery is downloaded or read. No reference data (EMSR927, UNOSAT, published damage maps, post-event OSM) is used, and nothing about which pair "performs best" is known at this stage.

## Why same-track matters

A radar image depends on the viewing geometry. Two acquisitions from different tracks see every slope from a different angle (different incidence angle, different look direction, different shadow and layover), so their backscatter differs even where nothing changed. Pixel-wise change detection, and a model trained on same-track stacks like Kuro Siwo, is meaningful only when every image in the set comes from the **same orbit direction and the same relative orbit**. Images from the same track repeat the same geometry every 12 days for one satellite (6 days for S1A + S1C).

Enforcement happens at three levels:
1. Candidates are generated inside one track only.
2. `TemporalSelectionResult` rejects any track that references a pass from another track.
3. Every candidate is converted to the existing `PrePostPair` contract (`to_pre_post_pair`), which refuses cross-track acquisitions. A candidate failing this is dropped and logged, never "fixed".

Tracks are **not** compared with each other. Every viable track keeps its own ranking; which track works better is decided later by preprocessing and model validation.

## Event-date rules (reused, not re-implemented)

Pre/post roles come from Step 2A (`CandidateRecord.role`):
- **Without an event time**, pre images must be dated strictly before the event date and post images strictly after it. Event-date images are excluded.
- **With `event_time_utc`**, ordering uses the exact timestamp.

`PrePostPair._check_event_ordering` re-checks this for every candidate.

## Passes and multi-slice handling

A Sentinel-1 overpass is delivered as consecutive ~25 s slices. Step 2A gives each accepted scene a `pass_id` (same platform and track, starts ≤ `sentinel1.pass_max_gap_s` apart) and measures the coverage of the slices' union.

`AcquisitionPass` merges those slices into **one temporal observation**: `scene_names` (slices, oldest first), `scene_coverages`, union `coverage`, `multi_slice`. Slices are never counted as separate dates; for example, two slices of the 1 September pass produce one post candidate, not two.

A set that uses any multi-slice pass is flagged `MULTI_SLICE_PASS` (`multi_slice = true`). Mosaicking the slices is **deferred to preprocessing**. `PrePostPair` is unchanged; `to_pre_post_pair` represents each pass by its earliest slice for validation only.

## Candidate generation (per track)

- Pre passes are sorted oldest first; post passes by event-to-post gap.
- For every post pass, and every pair of pre passes `(earlier, recent)` with `recent − earlier ≥ pairing.min_pre_separation_days`, one **2pre+1post** candidate is created: PRE-1 = earlier, PRE-2 = recent.
- If no 2+1 set exists (only one pre pass, or no pre pair far enough apart) and `pairing.allow_single_pre_fallback` is true, **1pre+1post** candidates are offered using the most recent pre pass, flagged `SINGLE_PRE_IMAGE`. They always rank after any 2+1 set.
- Every candidate lists `additional_pre_pass_ids`: all other valid pre passes on the track, so preprocessing can still choose a different baseline. Nothing valid is discarded; `TrackSelection.pre_pass_ids` / `post_pass_ids` list every pass.
- Track status:

  | Status | Meaning |
  |---|---|
  | `candidates` | at least one set kept |
  | `insufficient_pre` | no pre pass, or one pre pass with the fallback disabled |
  | `insufficient_post` | no post pass |
  | `no_valid_baseline` | ≥ 2 pre passes but none far enough apart (fallback disabled), or all candidates failed `PrePostPair` |

- `candidates_generated` counts all valid sets; the top `pairing.max_candidates_per_track` are kept, ranked.

## PAIRING HEURISTIC (provisional, unvalidated)

Each component is in [0, 1]:

| Component | Formula | Intent |
|---|---|---|
| `post_gap_score` | `1 − gap_post / sentinel1.post_window_days` | earlier post image after the event |
| `pre_recency_score` | `1 − age_recent_pre / sentinel1.pre_window_days` | baseline close to the event (less unrelated change) |
| `baseline_separation_score` | `min(1, separation / pairing.target_pre_separation_days)`; 0 for 1-pre sets | two genuinely different baseline dates |
| `coverage_score` | `min(pass coverage of PRE-1, PRE-2, POST)` | weakest-link AOI coverage |
| `quality_score` | `∏ (1 − penalty[flag])` over flags present | penalise known problems |

```
total = 0.35·post_gap + 0.20·pre_recency + 0.15·baseline_separation + 0.20·coverage + 0.10·quality
```

Weights and penalties live in `config.yaml → pairing` and must sum to 1. Current penalties:

| Flag | Penalty | Rule |
|---|---|---|
| `long_post_delay` | 0.5 | post later than `sentinel1.max_post_delay_days` |
| `multi_slice_pass` | 0.1 | any pass needs a mosaic |
| `metadata_from_product_name` | 0.1 | a field came from the name, not a catalogue attribute |
| `mixed_platforms` | 0.0 | e.g. S1A + S1C on one track: recorded, not penalised |
| `single_pre_image` | none | handled by shape ranking |

Components and total are rounded to 6 decimals and stored on every candidate (`score`), next to a plain-language `selection_reason`.

Because coverage and quality enter the score, the closest post image is not chosen blindly. A nearer pass with lower coverage or a known problem can rank below a later, complete one (tested).

**This heuristic is an engineering choice to order candidates, not a validated scientific criterion.** The weights express preferences, not measured effects. It must not be tuned against EMSR927 or any reference map. If it is ever changed, record why.

## Deterministic tie-breaking

The sort key is:
1. shape (2+1 before 1+1)
2. −total score
3. smaller post gap
4. more recent PRE-2
5. larger separation
6. the tuple of pass ids (lexicographic)

The last element makes the order total. Passes, tracks and candidates are built from sorted data, so the output does not depend on API order, dictionary order, the clock or randomness (tested by shuffling inputs and comparing JSON). Candidate ids are hashes of track + pass ids.

## Provisional thresholds (config.yaml → `pairing`)

| Key | Value | Rationale |
|---|---|---|
| `min_pre_separation_days` | 12 | baseline spans at least one single-satellite repeat cycle |
| `target_pre_separation_days` | 24 | separation at which the score saturates |
| `max_candidates_per_track` | 5 | enough choice for later experiments |
| `allow_single_pre_fallback` | true | keep tracks with one pre pass usable for change detection |
| weights / penalties | see above | — |

Also reused: `sentinel1.max_post_delay_days` (12), `pre_window_days` (60), `post_window_days` (24). `pairing.heuristic_status` only accepts `provisional`.

## Deferred to preprocessing (Step 2C onwards)

- Mosaicking multi-slice passes, and checking that the slices' processing baselines are compatible.
- Choosing the final baseline from `additional_pre_pass_ids` if needed.
- Radiometric consistency between platforms (S1A vs S1C).
- Pixel-level validity (no-data, border noise) and the visibility mask.
- Picking which track(s) to process for the final product.

## Limitations

- Coverage is the 2A footprint-based estimate, not pixel validity.
- Scores do not know about weather, wet snow or other conditions that change backscatter between dates.
- `days_from_event` is measured from 00:00 UTC of the event date when no event time is given, so a post image at 00:15 UTC twelve days later has a gap of 12.01 days.

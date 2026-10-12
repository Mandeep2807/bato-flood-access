"""Step 2B: same-track temporal-set selection (PRE-1 + PRE-2 + POST).

    select_temporal_sets(catalogue, cfg) -> TemporalSelectionResult

Input is a Step 2A CatalogueSearchResult (live or loaded from JSON). Only
ACCEPTED scenes are used; their roles (pre/post) were assigned by the 2A
event-date rules and are re-enforced here through PrePostPair.

1. Slices of the same pass (2A ``pass_id``) become one AcquisitionPass.
2. For every track (orbit direction, relative orbit) separately:
   every post pass x every pair of pre passes at least
   ``pairing.min_pre_separation_days`` apart -> a candidate set.
   (1-pre fallback if allowed and no 2+1 set exists.)
3. Each candidate is scored by the PAIRING HEURISTIC (docs/pair_selection.md),
   validated as a PrePostPair, ranked with a deterministic tie-breaker, and
   the top ``pairing.max_candidates_per_track`` are kept per track.

No track is preferred over another: every viable track keeps its own ranking.
No imagery is read; nothing from evaluation/reference data is used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections import defaultdict

from pydantic import ValidationError

from bato.config import BatoConfig, PairingSettings, Sentinel1Settings
from bato.contracts.acquisition import Acquisition
from bato.contracts.catalogue import CandidateRecord, CandidateRole, CatalogueSearchResult
from bato.contracts.pairing import PairQualityFlag, PrePostPair
from bato.contracts.temporal import (
    HEURISTIC_NAME,
    AcquisitionPass,
    PairingScore,
    SetQualityFlag,
    SetShape,
    TemporalSelectionResult,
    TemporalSetCandidate,
    TrackKey,
    TrackSelection,
    TrackStatus,
)
from bato.exceptions import BatoError
from bato.logging_utils import get_logger, stage

_DAY_S = 86400.0
_EPS = 1e-9
_FLAG_ORDER = list(SetQualityFlag)


class SelectionError(BatoError):
    """The catalogue result is internally inconsistent (e.g. a pass spanning two tracks)."""


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def _track_of(acq: Acquisition) -> TrackKey:
    if acq.orbit_direction is None or acq.relative_orbit is None:
        raise SelectionError(f"{acq.source_id} has no track information")
    return TrackKey(orbit_direction=acq.orbit_direction, relative_orbit=acq.relative_orbit)


def build_passes(catalogue: CatalogueSearchResult) -> tuple[AcquisitionPass, ...]:
    """Group ACCEPTED scenes into passes using the 2A pass ids."""
    members: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in catalogue.accepted():
        if record.pass_id is None:  # guaranteed by CandidateRecord for ACCEPTED
            raise SelectionError(f"accepted scene {record.label} has no pass id")
        members[record.pass_id].append(record)

    passes = []
    for pass_id in sorted(members):
        records = sorted(members[pass_id], key=lambda r: (r.sensing_start.isoformat() if r.sensing_start else "",
                                                          r.name or ""))
        acqs = [r.acquisition for r in records if r.acquisition is not None]
        tracks = {_track_of(a) for a in acqs}
        platforms = {a.platform for a in acqs}
        roles = {r.role for r in records}
        if len(acqs) != len(records) or len(tracks) != 1 or len(platforms) != 1 or len(roles) != 1:
            raise SelectionError(f"pass {pass_id} mixes tracks, platforms or roles: "
                                 f"{sorted(t.label for t in tracks)}, {sorted(map(str, platforms))}")
        coverages = {round(r.pass_coverage or 0.0, 6) for r in records}
        if len(coverages) != 1:
            raise SelectionError(f"pass {pass_id} has inconsistent pass coverage values {sorted(coverages)}")
        role = roles.pop()
        if role is None:
            raise SelectionError(f"pass {pass_id} has no pre/post role")
        first = records[0]
        passes.append(AcquisitionPass(
            pass_id=pass_id, platform=str(platforms.pop()), track=tracks.pop(), role=role,
            start=acqs[0].acquisition_start, days_from_event=first.days_from_event or 0.0,
            scene_names=tuple(r.name or "" for r in records),
            scene_coverages=tuple(r.scene_coverage or 0.0 for r in records),
            coverage=coverages.pop(),
            metadata_from_product_name=any("product_name" in r.metadata_sources.values() for r in records),
        ))
    return tuple(sorted(passes, key=lambda p: (p.track.orbit_direction.value, p.track.relative_orbit,
                                               p.start.isoformat(), p.pass_id)))


# ---------------------------------------------------------------------------
# PAIRING HEURISTIC
# ---------------------------------------------------------------------------


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _flags(pre: list[AcquisitionPass], post: AcquisitionPass, s1: Sentinel1Settings) -> tuple[SetQualityFlag, ...]:
    passes = [*pre, post]
    found = set()
    if len(pre) == 1:
        found.add(SetQualityFlag.SINGLE_PRE_IMAGE)
    if post.days_from_event > s1.max_post_delay_days:
        found.add(SetQualityFlag.LONG_POST_DELAY)
    if any(p.multi_slice for p in passes):
        found.add(SetQualityFlag.MULTI_SLICE_PASS)
    if any(p.metadata_from_product_name for p in passes):
        found.add(SetQualityFlag.METADATA_FROM_PRODUCT_NAME)
    if len({p.platform for p in passes}) > 1:
        found.add(SetQualityFlag.MIXED_PLATFORMS)
    return tuple(f for f in _FLAG_ORDER if f in found)


_PENALTY_KEY = {
    SetQualityFlag.LONG_POST_DELAY: "long_post_delay",
    SetQualityFlag.MULTI_SLICE_PASS: "multi_slice_pass",
    SetQualityFlag.METADATA_FROM_PRODUCT_NAME: "metadata_from_product_name",
    SetQualityFlag.MIXED_PLATFORMS: "mixed_platforms",
}


def pairing_score(post_gap_days: float, recent_pre_age_days: float, pre_separation_days: float | None,
                  coverage: float, flags: tuple[SetQualityFlag, ...], s1: Sentinel1Settings,
                  pairing: PairingSettings) -> PairingScore:
    """PAIRING HEURISTIC (provisional, unvalidated). Formula in docs/pair_selection.md."""
    penalties = pairing.penalties.model_dump()
    quality = 1.0
    for flag in flags:  # SINGLE_PRE_IMAGE has no penalty: it is handled by shape ranking
        if flag in _PENALTY_KEY:
            quality *= 1.0 - penalties[_PENALTY_KEY[flag]]
    components = {
        "post_gap_score": _clip01(1.0 - post_gap_days / s1.post_window_days),
        "pre_recency_score": _clip01(1.0 - recent_pre_age_days / s1.pre_window_days),
        "baseline_separation_score": (0.0 if pre_separation_days is None
                                      else _clip01(pre_separation_days / pairing.target_pre_separation_days)),
        "coverage_score": _clip01(coverage),
        "quality_score": _clip01(quality),
    }
    w = pairing.weights
    total = (w.post_gap * components["post_gap_score"] + w.pre_recency * components["pre_recency_score"]
             + w.baseline_separation * components["baseline_separation_score"]
             + w.coverage * components["coverage_score"] + w.quality * components["quality_score"])
    return PairingScore(**{k: round(v, 6) for k, v in components.items()}, total=round(_clip01(total), 6))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _candidate_id(track: TrackKey, pass_ids: list[str]) -> str:
    payload = json.dumps([track.label, *pass_ids], separators=(",", ":"))
    return "ts-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _reason(shape: SetShape, pre: list[AcquisitionPass], post: AcquisitionPass, sep: float | None,
            coverage: float, flags: tuple[SetQualityFlag, ...]) -> str:
    pre_txt = " and ".join(f"{-p.days_from_event:.1f} d" for p in pre)
    sep_txt = f", {sep:.1f} d apart" if sep is not None else " (single pre-event baseline)"
    flag_txt = ", ".join(f.value for f in flags) or "none"
    return (f"{shape}: post {post.days_from_event:.1f} d after the event; pre {pre_txt} before{sep_txt}; "
            f"weakest pass covers {coverage:.0%} of the AOI; flags: {flag_txt}. Ranked by {HEURISTIC_NAME}.")


def _make_candidate(track: TrackKey, pre: list[AcquisitionPass], post: AcquisitionPass,
                    all_pre: list[AcquisitionPass], s1: Sentinel1Settings,
                    pairing: PairingSettings) -> TemporalSetCandidate:
    shape: SetShape = "2pre+1post" if len(pre) == 2 else "1pre+1post"
    sep = (pre[-1].start - pre[0].start).total_seconds() / _DAY_S if len(pre) == 2 else None
    coverage = min(p.coverage for p in (*pre, post))
    flags = _flags(pre, post, s1)
    recent_age = -pre[-1].days_from_event
    score = pairing_score(post.days_from_event, recent_age, sep, coverage, flags, s1, pairing)
    used = {p.pass_id for p in pre}
    return TemporalSetCandidate(
        candidate_id=_candidate_id(track, [*(p.pass_id for p in pre), post.pass_id]), rank=1, track=track,
        shape=shape, pre_event_pass_ids=tuple(p.pass_id for p in pre), post_event_pass_id=post.pass_id,
        additional_pre_pass_ids=tuple(p.pass_id for p in all_pre if p.pass_id not in used),
        post_gap_days=round(post.days_from_event, 3), recent_pre_age_days=round(recent_age, 3),
        pre_separation_days=round(sep, 3) if sep is not None else None, coverage=round(coverage, 4),
        score=score, quality_flags=flags, multi_slice=SetQualityFlag.MULTI_SLICE_PASS in flags,
        selection_reason=_reason(shape, pre, post, sep, coverage, flags),
    )


def ranking_key(c: TemporalSetCandidate) -> tuple[int, float, float, float, float, tuple[str, ...]]:
    """Deterministic order: 2+1 sets first, then higher total score; ties broken by
    smaller post gap, more recent pre, larger separation, then pass ids."""
    return (0 if c.shape == "2pre+1post" else 1, -c.score.total, c.post_gap_days, c.recent_pre_age_days,
            -(c.pre_separation_days or 0.0), (*c.pre_event_pass_ids, c.post_event_pass_id))


def to_pre_post_pair(candidate: TemporalSetCandidate, selection: TemporalSelectionResult,
                     catalogue: CatalogueSearchResult) -> PrePostPair:
    """Build the PrePostPair contract for a candidate (final same-track and
    event-date enforcement). Each pass is represented by its earliest slice;
    multi-slice mosaicking is deferred to preprocessing."""
    passes = {p.pass_id: p for p in selection.passes}
    acquisitions = {r.name: r.acquisition for r in catalogue.accepted() if r.acquisition is not None}

    def representative(pass_id: str) -> Acquisition:
        acq = acquisitions.get(passes[pass_id].scene_names[0])
        if acq is None:
            raise SelectionError(f"pass {pass_id} has no accepted scene in the catalogue")
        return acq

    flags = []
    if SetQualityFlag.SINGLE_PRE_IMAGE in candidate.quality_flags:
        flags.append(PairQualityFlag.SINGLE_PRE_IMAGE)
    if SetQualityFlag.LONG_POST_DELAY in candidate.quality_flags:
        flags.append(PairQualityFlag.LONG_POST_DELAY)
    return PrePostPair(
        pair_id=candidate.candidate_id, event_date=selection.event_date, event_time_utc=selection.event_time_utc,
        pre=tuple(representative(pid) for pid in candidate.pre_event_pass_ids),
        post=representative(candidate.post_event_pass_id),
        orbit_direction=candidate.track.orbit_direction, relative_orbit=candidate.track.relative_orbit,
        aoi_coverage_fraction=candidate.coverage, pairing_rationale=candidate.selection_reason,
        quality_flags=tuple(flags),
    )


def _select_track(track: TrackKey, passes: list[AcquisitionPass], s1: Sentinel1Settings,
                  pairing: PairingSettings) -> tuple[tuple[TrackStatus, str], list[TemporalSetCandidate]]:
    pres = sorted((p for p in passes if p.role is CandidateRole.PRE_EVENT), key=lambda p: (p.start, p.pass_id))
    posts = sorted((p for p in passes if p.role is CandidateRole.POST_EVENT),
                   key=lambda p: (p.days_from_event, p.pass_id))
    counts = f"{len(pres)} pre pass(es), {len(posts)} post pass(es)"
    if not posts:
        return ("insufficient_post", f"{counts}: no post-event pass on this track"), []
    if not pres:
        return ("insufficient_pre", f"{counts}: no pre-event pass on this track"), []

    candidates = []
    for post in posts:
        for earlier, recent in itertools.combinations(pres, 2):  # pres sorted oldest first
            sep = (recent.start - earlier.start).total_seconds() / _DAY_S
            if sep + _EPS >= pairing.min_pre_separation_days:
                candidates.append(_make_candidate(track, [earlier, recent], post, pres, s1, pairing))
    if candidates:
        return ("candidates", f"{counts}"), candidates

    status: TrackStatus = "insufficient_pre" if len(pres) == 1 else "no_valid_baseline"
    why = ("only one pre-event pass" if len(pres) == 1
           else f"no two pre passes are >= {pairing.min_pre_separation_days:g} d apart")
    if not pairing.allow_single_pre_fallback:
        return (status, f"{counts}: {why}; single-pre fallback disabled"), []
    for post in posts:
        candidates.append(_make_candidate(track, [pres[-1]], post, pres, s1, pairing))
    return ("candidates", f"{counts}: {why}; offering 1-pre fallback sets (SINGLE_PRE_IMAGE)"), candidates



def select_temporal_sets(catalogue: CatalogueSearchResult, cfg: BatoConfig,
                         logger: logging.Logger | None = None) -> TemporalSelectionResult:
    """Rank same-track temporal sets for every track in a catalogue result."""
    s1, pairing = cfg.settings.sentinel1, cfg.settings.pairing
    digest = hashlib.sha256(catalogue.model_dump_json().encode("utf-8")).hexdigest()
    log = logger or get_logger("acquisition")
    with stage("s1_temporal_selection", logger=log, aoi_id=catalogue.aoi_id, catalogue_digest=digest[:16],
               accepted_scenes=len(catalogue.accepted()), heuristic=HEURISTIC_NAME) as report:
        passes = build_passes(catalogue)
        by_track: dict[TrackKey, list[AcquisitionPass]] = defaultdict(list)
        for p in passes:
            by_track[p.track].append(p)

        selection_stub = TemporalSelectionResult(  # used only to validate candidates via PrePostPair
            heuristic=HEURISTIC_NAME, aoi_id=catalogue.aoi_id, event_date=catalogue.event_date,
            event_time_utc=catalogue.event_time_utc, catalogue_digest=digest,
            catalogue_retrieved_at=catalogue.retrieved_at, config_fingerprint=cfg.fingerprint(),
            passes=passes, tracks=())
        tracks = []
        for track in sorted(by_track, key=lambda t: (t.orbit_direction.value, t.relative_orbit)):
            (status, note), generated = _select_track(track, by_track[track], s1, pairing)
            valid = []
            for candidate in generated:
                try:
                    to_pre_post_pair(candidate, selection_stub, catalogue)
                except ValidationError as exc:  # documented: contract is the final gate; record, don't crash
                    report.warn(f"{track.label} candidate {candidate.candidate_id} rejected by PrePostPair: "
                                f"{exc.errors()[0]['msg']}")
                    continue
                valid.append(candidate)
            ranked = sorted(valid, key=ranking_key)[: pairing.max_candidates_per_track]
            ranked = [c.model_copy(update={"rank": i}) for i, c in enumerate(ranked, start=1)]
            if generated and not ranked:
                status, note = "no_valid_baseline", f"{note}; every candidate failed PrePostPair validation"
            members = by_track[track]
            tracks.append(TrackSelection(
                track=track, status=status, note=note,
                pre_pass_ids=tuple(p.pass_id for p in sorted(members, key=lambda p: (p.start, p.pass_id))
                                   if p.role is CandidateRole.PRE_EVENT),
                post_pass_ids=tuple(p.pass_id for p in sorted(members, key=lambda p: (p.start, p.pass_id))
                                    if p.role is CandidateRole.POST_EVENT),
                candidates_generated=len(valid), candidates=tuple(ranked),
            ))

        result = selection_stub.model_copy(update={"tracks": tuple(tracks)})
        result = TemporalSelectionResult.model_validate(result.model_dump())  # re-run cross-checks
        report.record_output(
            passes=len(passes), multi_slice_passes=sum(p.multi_slice for p in passes),
            tracks={t.track.label: f"{t.status} ({len(t.candidates)}/{t.candidates_generated})" for t in tracks})
        if not any(t.candidates for t in tracks):
            report.warn("no track has a usable temporal set; see track notes")
    return result

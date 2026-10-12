"""Step 2B contracts: acquisition passes and ranked temporal-set candidates.

Concepts
--------
AcquisitionPass
    One satellite overpass of the AOI on one track: one or more adjacent
    slices (scenes) of the same platform and pass, plus the AOI coverage of
    their union. A pass is ONE temporal observation; its slices are never
    treated as separate dates. Mosaicking the slices is deferred to
    preprocessing.

TemporalSetCandidate
    PRE-1 (earlier) + PRE-2 (more recent) + POST passes, all on one
    (orbit direction, relative orbit) track - the Kuro Siwo input shape -
    or, as a flagged fallback, 1 pre + 1 post. Passes are referenced by id;
    scenes stay in the catalogue result (no duplicate Acquisition copies).

Scores are a PAIRING HEURISTIC (deterministic, explainable, unvalidated).
:class:`~bato.contracts.pairing.PrePostPair` remains the final enforcement
of the same-track and event-date rules (see bato.acquisition.temporal_selection).
"""

from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Literal

from pydantic import computed_field, model_validator

from bato.contracts.catalogue import CandidateRole
from bato.contracts.common import Contract, OrbitDirection, Sha256Hex, UTCDatetime

HEURISTIC_NAME = "PAIRING HEURISTIC v1 (unvalidated)"
SetShape = Literal["2pre+1post", "1pre+1post"]


class TrackKey(Contract):
    orbit_direction: OrbitDirection
    relative_orbit: int

    @model_validator(mode="after")
    def _check(self) -> TrackKey:
        if not 1 <= self.relative_orbit <= 175:
            raise ValueError("relative_orbit must be in 1..175")
        return self

    @property
    def label(self) -> str:
        return f"{self.orbit_direction.value.upper()}/{self.relative_orbit}"


class AcquisitionPass(Contract):
    pass_id: str
    platform: str
    track: TrackKey
    role: CandidateRole
    start: UTCDatetime                       # earliest slice start
    days_from_event: float                   # of the earliest slice
    scene_names: tuple[str, ...]             # slices, oldest first (= Acquisition.source_id)
    scene_coverages: tuple[float, ...]       # each slice alone
    coverage: float                          # union of slices, fraction of AOI
    metadata_from_product_name: bool = False  # any slice field taken from the name, not an attribute

    @model_validator(mode="after")
    def _check(self) -> AcquisitionPass:
        if not self.scene_names:
            raise ValueError("a pass needs at least one scene")
        if len(set(self.scene_names)) != len(self.scene_names):
            raise ValueError("duplicate scene in pass")
        if len(self.scene_coverages) != len(self.scene_names):
            raise ValueError("one coverage value per scene is required")
        for value in (*self.scene_coverages, self.coverage):
            if not 0.0 <= value <= 1.0:
                raise ValueError("coverage values must be in [0, 1]")
        if self.coverage + 1e-9 < max(self.scene_coverages):
            raise ValueError("pass coverage cannot be below a single slice's coverage")
        if self.role is CandidateRole.PRE_EVENT and self.days_from_event >= 0:
            raise ValueError("a pre-event pass must start before the event")
        if self.role is CandidateRole.POST_EVENT and self.days_from_event <= 0:
            raise ValueError("a post-event pass must start after the event")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def multi_slice(self) -> bool:
        return len(self.scene_names) > 1



class SetQualityFlag(str, Enum):
    SINGLE_PRE_IMAGE = "single_pre_image"                # fallback shape, 1 pre + 1 post
    LONG_POST_DELAY = "long_post_delay"                  # post later than sentinel1.max_post_delay_days
    MULTI_SLICE_PASS = "multi_slice_pass"                # at least one pass needs a mosaic
    METADATA_FROM_PRODUCT_NAME = "metadata_from_product_name"
    MIXED_PLATFORMS = "mixed_platforms"                  # e.g. S1A + S1C on the same track


class PairingScore(Contract):
    """Components in [0, 1]; total = sum(weight * component). See docs/pair_selection.md."""

    post_gap_score: float
    pre_recency_score: float
    baseline_separation_score: float
    coverage_score: float
    quality_score: float
    total: float

    @model_validator(mode="after")
    def _check(self) -> PairingScore:
        for name, value in self.model_dump().items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return self


class TemporalSetCandidate(Contract):
    candidate_id: str
    rank: int                                  # 1 = best within its track
    track: TrackKey
    shape: SetShape
    pre_event_pass_ids: tuple[str, ...]        # oldest first: PRE-1 (earlier), PRE-2 (recent)
    post_event_pass_id: str
    additional_pre_pass_ids: tuple[str, ...]   # other valid pre passes on this track, kept for later choice
    post_gap_days: float
    recent_pre_age_days: float                 # event -> most recent pre pass
    pre_separation_days: float | None          # None for the 1-pre fallback
    coverage: float                            # weakest pass coverage in the set
    score: PairingScore
    quality_flags: tuple[SetQualityFlag, ...]
    multi_slice: bool
    selection_reason: str

    @model_validator(mode="after")
    def _check(self) -> TemporalSetCandidate:
        expected_pre = 2 if self.shape == "2pre+1post" else 1
        if len(self.pre_event_pass_ids) != expected_pre:
            raise ValueError(f"shape {self.shape} needs {expected_pre} pre-event pass(es)")
        if (self.pre_separation_days is None) != (expected_pre == 1):
            raise ValueError("pre_separation_days is required exactly for 2-pre sets")
        if (SetQualityFlag.SINGLE_PRE_IMAGE in self.quality_flags) != (expected_pre == 1):
            raise ValueError("SINGLE_PRE_IMAGE flag must match the set shape")
        if (SetQualityFlag.MULTI_SLICE_PASS in self.quality_flags) != self.multi_slice:
            raise ValueError("MULTI_SLICE_PASS flag must match multi_slice")
        used = {*self.pre_event_pass_ids, self.post_event_pass_id}
        if len(used) != expected_pre + 1 or used & set(self.additional_pre_pass_ids):
            raise ValueError("a pass may appear only once in a candidate")
        if self.post_gap_days <= 0 or self.recent_pre_age_days <= 0:
            raise ValueError("pre passes must precede and the post pass follow the event")
        if self.pre_separation_days is not None and self.pre_separation_days <= 0:
            raise ValueError("pre_separation_days must be positive")
        if self.rank < 1 or not 0.0 <= self.coverage <= 1.0 or not self.selection_reason.strip():
            raise ValueError("invalid rank, coverage or empty selection_reason")
        return self


TrackStatus = Literal["candidates", "insufficient_pre", "insufficient_post", "no_valid_baseline"]


class TrackSelection(Contract):
    track: TrackKey
    status: TrackStatus
    note: str
    pre_pass_ids: tuple[str, ...]          # every valid pre pass, oldest first
    post_pass_ids: tuple[str, ...]         # every valid post pass, earliest first
    candidates_generated: int
    candidates: tuple[TemporalSetCandidate, ...]   # top-N, ranked

    @model_validator(mode="after")
    def _check(self) -> TrackSelection:
        if [c.rank for c in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise ValueError("candidate ranks must be 1..n in order")
        if len(self.candidates) > self.candidates_generated:
            raise ValueError("cannot keep more candidates than were generated")
        if (self.status == "candidates") != bool(self.candidates):
            raise ValueError("status 'candidates' if and only if candidates exist")
        for c in self.candidates:
            if c.track != self.track:
                raise ValueError("candidate on a different track than its TrackSelection")
            if not {*c.pre_event_pass_ids, *c.additional_pre_pass_ids} <= set(self.pre_pass_ids):
                raise ValueError("candidate references an unknown pre pass")
            if c.post_event_pass_id not in self.post_pass_ids:
                raise ValueError("candidate references an unknown post pass")
        return self


class TemporalSelectionResult(Contract):
    heuristic: str
    aoi_id: str
    event_date: date
    event_time_utc: time | None = None
    catalogue_digest: Sha256Hex            # SHA-256 of the input CatalogueSearchResult JSON
    catalogue_retrieved_at: UTCDatetime
    config_fingerprint: str
    passes: tuple[AcquisitionPass, ...]
    tracks: tuple[TrackSelection, ...]

    @model_validator(mode="after")
    def _check(self) -> TemporalSelectionResult:
        by_id = {p.pass_id: p for p in self.passes}
        if len(by_id) != len(self.passes):
            raise ValueError("duplicate pass ids")
        keys = [(t.track.orbit_direction.value, t.track.relative_orbit) for t in self.tracks]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("tracks must be unique and sorted")
        for t in self.tracks:
            for pid in (*t.pre_pass_ids, *t.post_pass_ids):
                p = by_id.get(pid)
                if p is None:
                    raise ValueError(f"unknown pass {pid}")
                if p.track != t.track:
                    raise ValueError(f"pass {pid} is on {p.track.label}, not {t.track.label}: cross-track sets "
                                     "are never allowed")
            if any(by_id[pid].role is not CandidateRole.PRE_EVENT for pid in t.pre_pass_ids):
                raise ValueError("pre_pass_ids must be pre-event passes")
            if any(by_id[pid].role is not CandidateRole.POST_EVENT for pid in t.post_pass_ids):
                raise ValueError("post_pass_ids must be post-event passes")
        return self

    def all_candidates(self) -> list[TemporalSetCandidate]:
        return [c for t in self.tracks for c in t.candidates]

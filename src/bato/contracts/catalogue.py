"""Catalogue-search result: every scene the provider returned, with a status.

Nothing is silently discarded. Each returned record ends in exactly one
:class:`CandidateStatus`. Rejected records carry the first rule they failed,
a human-readable reason, and whatever metadata could be parsed, so the
system can always answer "why was this Sentinel-1 scene not selected?".

Accepted scenes are grouped by (orbit direction, relative orbit). A group is
a set of pairing CANDIDATES only; choosing the actual pre/post pair is a
later step, and :class:`~bato.contracts.pairing.PrePostPair` remains the
final enforcement of the same-track rule.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, time
from enum import Enum

from pydantic import computed_field, model_validator

from bato.contracts.acquisition import Acquisition
from bato.contracts.common import Contract, OrbitDirection, UTCDatetime


class CandidateStatus(str, Enum):
    """Checks run in this order; a record gets the FIRST one it fails."""

    REJECTED_INVALID_METADATA = "rejected_invalid_metadata"  # missing/malformed/inconsistent metadata
    REJECTED_DUPLICATE = "rejected_duplicate"                # same acquisition listed more than once
    REJECTED_MODE = "rejected_mode"                          # not the configured beam mode (IW)
    REJECTED_PRODUCT_TYPE = "rejected_product_type"          # not a level-1 GRD of the configured class
    REJECTED_ORBIT = "rejected_orbit"                        # invalid orbit direction / relative orbit
    REJECTED_POLARIZATION = "rejected_polarization"          # required polarisations not all present
    REJECTED_DATE = "rejected_date"                          # outside window, or ambiguous same-day image
    REJECTED_COVERAGE = "rejected_coverage"                  # its pass covers too little of the AOI
    ACCEPTED = "accepted"


class CandidateRole(str, Enum):
    PRE_EVENT = "pre_event"
    POST_EVENT = "post_event"


class CandidateRecord(Contract):
    product_id: str | None = None            # CDSE product UUID (needed for download later)
    name: str | None = None                  # ESA product name (= Acquisition.source_id)
    status: CandidateStatus
    reason: str
    sensing_start: UTCDatetime | None = None  # whenever a valid start time could be read
    acquisition: Acquisition | None = None   # present whenever the metadata allowed building one
    role: CandidateRole | None = None
    days_from_event: float | None = None     # from event time, or 00:00 UTC of the event date
    scene_coverage: float | None = None      # this scene alone, fraction of AOI area
    pass_id: str | None = None
    pass_coverage: float | None = None       # union of all accepted-so-far slices of the same pass
    duplicate_of: str | None = None
    metadata_sources: dict[str, str] = {}    # field -> "attribute" | "product_name"

    @model_validator(mode="after")
    def _check(self) -> CandidateRecord:
        if not self.reason.strip():
            raise ValueError("every record needs a reason")
        for name in ("scene_coverage", "pass_coverage"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.status is CandidateStatus.ACCEPTED:
            missing = [n for n in ("acquisition", "role", "scene_coverage", "pass_id", "pass_coverage")
                       if getattr(self, n) is None]
            if missing:
                raise ValueError(f"an ACCEPTED record needs {missing}")
        if self.status is CandidateStatus.REJECTED_DUPLICATE and not self.duplicate_of:
            raise ValueError("a duplicate must name the record it duplicates")
        if self.status is CandidateStatus.REJECTED_INVALID_METADATA and self.acquisition is not None:
            raise ValueError("a record with invalid metadata cannot carry an Acquisition")
        return self

    @property
    def label(self) -> str:
        return self.name or self.product_id or "<unidentified record>"


class TrackGroup(Contract):
    """Accepted scenes on one (orbit direction, relative orbit) track."""

    orbit_direction: OrbitDirection
    relative_orbit: int
    pre_event: tuple[str, ...]      # accepted names, oldest first
    post_event: tuple[str, ...]     # accepted names, oldest first
    pass_ids: tuple[str, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_pre_and_post(self) -> bool:
        return bool(self.pre_event) and bool(self.post_event)


class CatalogueSearchResult(Contract):
    provider: str
    query_filter: str
    aoi_id: str
    event_date: date
    event_time_utc: time | None = None
    window_start: UTCDatetime
    window_end: UTCDatetime
    retrieved_at: UTCDatetime
    returned_count: int
    truncated: bool
    config_fingerprint: str
    records: tuple[CandidateRecord, ...]
    groups: tuple[TrackGroup, ...]

    @model_validator(mode="after")
    def _check(self) -> CatalogueSearchResult:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if self.returned_count != len(self.records):
            raise ValueError("every returned record must appear in records (none silently dropped)")
        accepted = {r.name for r in self.records if r.status is CandidateStatus.ACCEPTED}
        for group in self.groups:
            names = set(group.pre_event) | set(group.post_event)
            if not names <= accepted:
                raise ValueError("track groups may only reference accepted records")
            for record in self.records:
                if record.name in names and record.acquisition is not None:
                    acq = record.acquisition
                    if acq.track_key != (group.orbit_direction, group.relative_orbit):
                        raise ValueError(f"{record.name} is grouped under the wrong track")
        keys = [(g.orbit_direction.value, g.relative_orbit) for g in self.groups]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("track groups must be unique and sorted")
        return self

    def accepted(self) -> list[CandidateRecord]:
        return [r for r in self.records if r.status is CandidateStatus.ACCEPTED]

    def rejected(self) -> list[CandidateRecord]:
        return [r for r in self.records if r.status is not CandidateStatus.ACCEPTED]

    def status_counts(self) -> dict[str, int]:
        counts = Counter(r.status.value for r in self.records)
        return {status.value: counts.get(status.value, 0) for status in CandidateStatus}

    def explain(self, identifier: str) -> str:
        """Why was this scene (product name, name without .SAFE, or product id) (not) selected?"""
        for record in self.records:
            names = {record.name, record.product_id}
            if record.name:
                names.add(record.name.removesuffix(".SAFE"))
            if identifier in names:
                return f"{record.label}: {record.status.value} - {record.reason}"
        return f"{identifier}: not returned by the catalogue for this AOI and window"


def _sort_key(record: CandidateRecord) -> tuple[str, str, str]:
    start = record.sensing_start.isoformat() if record.sensing_start else "~"  # unknown time sorts last
    return (start, record.name or "", record.product_id or "")


def sort_records(records: list[CandidateRecord]) -> tuple[CandidateRecord, ...]:
    """Deterministic order: acquisition time, then name, then id (unparsed records last)."""
    return tuple(sorted(records, key=_sort_key))

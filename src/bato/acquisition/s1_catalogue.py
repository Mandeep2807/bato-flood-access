"""Sentinel-1 catalogue discovery: AOI + event date -> classified candidate scenes.

    search_sentinel1(aoi, cfg) -> CatalogueSearchResult

1. Search window from config: [event_date - pre_window_days, event_date + post_window_days]
   (whole UTC days, end exclusive at the next midnight).
2. Fetch every Sentinel-1 record intersecting the AOI in the window (CDSE OData).
3. Classify each record (first failing rule wins; see CandidateStatus):
   metadata -> duplicates -> mode -> product type -> orbit -> polarisation -> date -> coverage.
4. Group accepted scenes by (orbit direction, relative orbit).

No pair is selected here. No pixels are downloaded.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any

from bato.acquisition.cdse_odata import PROVIDER, CDSEODataClient, ParsedS1Record, parse_s1_record
from bato.acquisition.coverage import AOICoverageGrid
from bato.config import BatoConfig, Sentinel1Settings
from bato.contracts.acquisition import S1_RELATIVE_ORBITS, Acquisition, ProcessingStatus
from bato.contracts.aoi import AOI
from bato.contracts.catalogue import (
    CandidateRecord,
    CandidateRole,
    CandidateStatus,
    CatalogueSearchResult,
    TrackGroup,
    sort_records,
)
from bato.contracts.common import OrbitDirection, Sensor
from bato.exceptions import MetadataError
from bato.logging_utils import get_logger, stage

S = CandidateStatus
_GRD_LEVEL, _GRD_CLASS = "1", "S"   # level-1 standard products (ESA naming)


def search_window(aoi: AOI, s1: Sentinel1Settings) -> tuple[datetime, datetime]:
    """[start, end) in UTC covering the configured pre and post windows (whole days)."""
    start_day = aoi.event_date - timedelta(days=s1.pre_window_days)
    end_day = aoi.event_date + timedelta(days=s1.post_window_days + 1)
    return (datetime.combine(start_day, time(0), tzinfo=timezone.utc),
            datetime.combine(end_day, time(0), tzinfo=timezone.utc))


def _event_reference(aoi: AOI) -> datetime:
    return aoi.event_datetime_utc or datetime.combine(aoi.event_date, time(0), tzinfo=timezone.utc)


@dataclass
class _Work:
    """Mutable per-record state during classification."""

    parsed: ParsedS1Record
    acquisition: Acquisition | None = None
    role: CandidateRole | None = None
    status: CandidateStatus | None = None
    reason: str = ""
    duplicate_of: str | None = None
    scene_coverage: float | None = None
    pass_id: str | None = None
    pass_coverage: float | None = None

    def reject(self, status: CandidateStatus, reason: str) -> None:
        self.status, self.reason = status, reason


def _identity(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    pid, name = raw.get("Id"), raw.get("Name")
    return (pid if isinstance(pid, str) and pid else None, name if isinstance(name, str) and name else None)


def _try_acquisition(p: ParsedS1Record) -> Acquisition | None:
    """Build the contract when orbit metadata is in range; otherwise None (REJECTED_ORBIT later)."""
    if p.orbit_direction not in {d.value.upper() for d in OrbitDirection}:
        return None
    if not 1 <= p.relative_orbit <= S1_RELATIVE_ORBITS:
        return None
    return Acquisition(
        source_id=p.name.name, sensor=Sensor.SENTINEL_1, platform=p.platform, acquisition_start=p.start,
        product_type=p.product_type, orbit_direction=OrbitDirection(p.orbit_direction.lower()),
        relative_orbit=p.relative_orbit, polarizations=p.polarizations, footprint=p.footprint,
        processing_status=ProcessingStatus.CATALOGUED,
    )


def _mark_duplicates(items: list[_Work]) -> None:
    """Same acquisition listed more than once (e.g. reprocessing, COG copy): keep one.

    Preference: not COG, then most recently published, then smallest product id.
    """
    by_key: dict[tuple[Any, ...], list[_Work]] = defaultdict(list)
    for w in items:
        n = w.parsed.name
        by_key[(n.platform, n.mode, w.parsed.product_type, n.polarization_code, n.start, n.stop)].append(w)
    for group in by_key.values():
        if len(group) < 2:
            continue

        def preference(w: _Work) -> tuple[bool, float, str]:
            published = w.parsed.publication_date.timestamp() if w.parsed.publication_date else 0.0
            return (w.parsed.name.is_cog, -published, w.parsed.product_id)

        keep, *others = sorted(group, key=preference)
        for w in others:
            w.duplicate_of = keep.parsed.name.name
            w.reject(S.REJECTED_DUPLICATE, f"same acquisition as {keep.parsed.name.name} (kept by preference: "
                                           "non-COG, latest publication, smallest id)")


def _check_rules(w: _Work, aoi: AOI, s1: Sentinel1Settings, window: tuple[datetime, datetime]) -> None:
    p, n = w.parsed, w.parsed.name
    if p.mode != s1.acquisition_mode:
        return w.reject(S.REJECTED_MODE, f"beam mode {p.mode}, required {s1.acquisition_mode}")
    if (n.product_kind, n.resolution, n.level, n.product_class) != (
            s1.product_type, s1.grd_resolution_class, _GRD_LEVEL, _GRD_CLASS):
        return w.reject(S.REJECTED_PRODUCT_TYPE,
                        f"product type {p.product_type}, required {s1.acquisition_mode}_"
                        f"{s1.product_type}{s1.grd_resolution_class}_{_GRD_LEVEL}{_GRD_CLASS}")
    if w.acquisition is None:
        return w.reject(S.REJECTED_ORBIT, f"invalid orbit metadata: direction {p.orbit_direction!r}, "
                                          f"relative orbit {p.relative_orbit} (valid 1..{S1_RELATIVE_ORBITS})")
    missing = [pol for pol in s1.polarizations if pol not in {x.value for x in p.polarizations}]
    if missing:
        have = "+".join(x.value for x in p.polarizations)
        return w.reject(S.REJECTED_POLARIZATION, f"has {have}, missing required {'+'.join(missing)}")

    start, end = window
    if not start <= p.start < end:
        return w.reject(S.REJECTED_DATE, f"sensed {p.start.isoformat()} outside search window "
                                         f"[{start.date()}, {end.date()})")
    event_dt = aoi.event_datetime_utc
    if event_dt is not None:
        if p.start == event_dt:
            return w.reject(S.REJECTED_DATE, "sensed exactly at the event time")
        w.role = CandidateRole.PRE_EVENT if p.start < event_dt else CandidateRole.POST_EVENT
    else:
        sensed_day = p.start.date()
        if sensed_day == aoi.event_date:
            return w.reject(S.REJECTED_DATE, "sensed on the event date and no event time given: "
                                             "before/after is ambiguous (same-day policy: exclude)")
        w.role = CandidateRole.PRE_EVENT if sensed_day < aoi.event_date else CandidateRole.POST_EVENT
    return None


def _assign_passes(items: list[_Work], gap_s: float) -> dict[str, list[_Work]]:
    """Chain scenes on the same platform and track whose starts are <= gap_s apart."""
    by_track: dict[tuple[str, str, int], list[_Work]] = defaultdict(list)
    for w in items:
        by_track[(w.parsed.platform, w.parsed.orbit_direction, w.parsed.relative_orbit)].append(w)
    passes: dict[str, list[_Work]] = {}
    for (platform, direction, rel), members in sorted(by_track.items()):
        members.sort(key=lambda w: (w.parsed.start, w.parsed.name.name))
        current_id, previous_start = "", None
        for w in members:
            if previous_start is None or (w.parsed.start - previous_start).total_seconds() > gap_s:
                current_id = f"{platform}-{direction[0]}{rel:03d}-{w.parsed.start:%Y%m%dT%H%M%S}"
                passes[current_id] = []
            passes[current_id].append(w)
            w.pass_id = current_id
            previous_start = w.parsed.start
    return passes


def classify_records(raw_records: Iterable[dict[str, Any]], aoi: AOI, s1: Sentinel1Settings,
                     window: tuple[datetime, datetime]) -> tuple[CandidateRecord, ...]:
    """Classify raw catalogue records. Pure and deterministic (input order does not matter)."""
    invalid: list[CandidateRecord] = []
    work: list[_Work] = []
    for raw in raw_records:
        try:
            parsed = parse_s1_record(raw)
        except MetadataError as exc:  # documented fallback: record kept as REJECTED_INVALID_METADATA
            pid, name = _identity(raw)
            invalid.append(CandidateRecord(product_id=pid, name=name, status=S.REJECTED_INVALID_METADATA,
                                           reason=str(exc)))
            continue
        work.append(_Work(parsed=parsed, acquisition=_try_acquisition(parsed)))

    _mark_duplicates(work)
    for w in work:
        if w.status is None:
            _check_rules(w, aoi, s1, window)

    # Coverage is measured per scene and per pass, for scenes that passed every other rule.
    eligible = [w for w in work if w.status is None]
    if eligible:
        grid = AOICoverageGrid(aoi.geometry, s1.coverage_grid_size)
        for w in eligible:
            w.scene_coverage = grid.fraction([w.parsed.footprint])
        for members in _assign_passes(eligible, s1.pass_max_gap_s).values():
            pass_cov = grid.fraction([m.parsed.footprint for m in members])
            for w in members:
                w.pass_coverage = pass_cov
                if w.scene_coverage == 0.0:
                    w.reject(S.REJECTED_COVERAGE, "footprint does not overlap the AOI")
                elif pass_cov < s1.min_aoi_coverage_fraction:
                    w.reject(S.REJECTED_COVERAGE,
                             f"pass covers {pass_cov:.1%} of the AOI (scene alone {w.scene_coverage:.1%}); "
                             f"minimum {s1.min_aoi_coverage_fraction:.0%}")
                else:
                    slices = f" with {len(members) - 1} adjacent slice(s)" if len(members) > 1 else ""
                    w.status, w.reason = S.ACCEPTED, f"pass covers {pass_cov:.1%} of the AOI{slices}"

    event_ref = _event_reference(aoi)
    records = list(invalid)
    for w in work:
        if w.status is None:  # internal invariant: every path above sets a status
            raise RuntimeError(f"unclassified catalogue record {w.parsed.name.name}")
        records.append(CandidateRecord(
            product_id=w.parsed.product_id, name=w.parsed.name.name, status=w.status, reason=w.reason,
            sensing_start=w.parsed.start, acquisition=w.acquisition, role=w.role,
            days_from_event=round((w.parsed.start - event_ref).total_seconds() / 86400, 3),
            scene_coverage=w.scene_coverage, pass_id=w.pass_id, pass_coverage=w.pass_coverage,
            duplicate_of=w.duplicate_of, metadata_sources=w.parsed.metadata_sources,
        ))
    return sort_records(records)


def group_by_track(records: Iterable[CandidateRecord]) -> tuple[TrackGroup, ...]:
    """Group ACCEPTED scenes by (orbit direction, relative orbit). Never mixes tracks."""
    groups: dict[tuple[OrbitDirection, int], list[CandidateRecord]] = defaultdict(list)
    for r in records:
        acq = r.acquisition
        if r.status is not S.ACCEPTED or acq is None:
            continue
        if acq.orbit_direction is None or acq.relative_orbit is None:  # impossible for Sentinel-1 contracts
            raise RuntimeError(f"accepted record {r.label} has no track information")
        groups[(acq.orbit_direction, acq.relative_orbit)].append(r)
    result = []
    for (direction, rel), members in sorted(groups.items(), key=lambda kv: (kv[0][0].value, kv[0][1])):
        members.sort(key=lambda r: (r.sensing_start.isoformat() if r.sensing_start else "~", r.name or ""))
        result.append(TrackGroup(
            orbit_direction=direction, relative_orbit=rel,
            pre_event=tuple(r.name for r in members if r.role is CandidateRole.PRE_EVENT and r.name),
            post_event=tuple(r.name for r in members if r.role is CandidateRole.POST_EVENT and r.name),
            pass_ids=tuple(sorted({r.pass_id for r in members if r.pass_id})),
        ))
    return tuple(result)


def search_sentinel1(aoi: AOI, cfg: BatoConfig, client: CDSEODataClient | None = None,
                     retrieved_at: datetime | None = None,
                     logger: logging.Logger | None = None) -> CatalogueSearchResult:
    """Search the CDSE catalogue and classify every returned Sentinel-1 record.

    Raises:
        CatalogueQueryError: the provider could not be queried (logged, re-raised).
    """
    s1 = cfg.settings.sentinel1
    client = client or CDSEODataClient(cfg.settings.cdse)
    window = search_window(aoi, s1)
    log = logger or get_logger("acquisition")
    with stage("s1_catalogue_search", logger=log, provider=PROVIDER, aoi_id=aoi.aoi_id,
               aoi_bbox=aoi.bbox.model_dump(), area_km2=aoi.area_km2, event_date=aoi.event_date.isoformat(),
               window=f"{window[0].isoformat()} .. {window[1].isoformat()}") as report:
        fetched = client.fetch_s1(aoi.geometry, *window)
        records = classify_records(fetched.records, aoi, s1, window)
        groups = group_by_track(records)
        result = CatalogueSearchResult(
            provider=PROVIDER, query_filter=fetched.query_filter, aoi_id=aoi.aoi_id,
            event_date=aoi.event_date, event_time_utc=aoi.event_time_utc,
            window_start=window[0], window_end=window[1],
            retrieved_at=retrieved_at or datetime.now(timezone.utc), returned_count=len(fetched.records),
            truncated=fetched.truncated, config_fingerprint=cfg.fingerprint(), records=records, groups=groups,
        )
        counts = {k: v for k, v in result.status_counts().items() if v}
        report.record_output(returned=result.returned_count, accepted=len(result.accepted()),
                             rejected=len(result.rejected()), by_status=counts, pages=fetched.pages,
                             tracks=len(groups), tracks_with_pre_and_post=sum(g.has_pre_and_post for g in groups))
        if fetched.truncated:
            report.warn(f"result truncated at max_records={cfg.settings.cdse.max_records}; "
                        "narrow the window or raise the cap")
        if not result.accepted():
            report.warn("no scene accepted; see rejection reasons in the result")
    return result

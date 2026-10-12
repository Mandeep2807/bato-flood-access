"""Contract C - a same-track Sentinel-1 pre/post-event pair.

Encodes the PS rule that change detection compares images from the same
orbit track. A pair that mixes tracks, or that uses an image whose timing
relative to the event is ambiguous, cannot be constructed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from enum import Enum

from pydantic import computed_field, model_validator

from bato.contracts.acquisition import Acquisition
from bato.contracts.common import Contract, OrbitDirection, Sensor


class PairQualityFlag(str, Enum):
    SINGLE_PRE_IMAGE = "single_pre_image"            # change baseline from one image only
    LONG_POST_DELAY = "long_post_delay"              # post image later than configured limit
    PARTIAL_AOI_COVERAGE = "partial_aoi_coverage"    # track does not cover enough of the AOI
    MISSING_POLARIZATION = "missing_polarization"    # a configured polarization is absent


class PrePostPair(Contract):
    """Pre-event stack + first post-event image from ONE relative orbit."""

    pair_id: str
    event_date: date
    event_time_utc: time | None = None
    pre: tuple[Acquisition, ...]
    post: Acquisition
    orbit_direction: OrbitDirection
    relative_orbit: int
    aoi_coverage_fraction: float
    pairing_rationale: str
    quality_flags: tuple[PairQualityFlag, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> PrePostPair:
        if not self.pre:
            raise ValueError("at least one pre-event acquisition is required")
        if not self.pairing_rationale.strip():
            raise ValueError("pairing_rationale must explain why this pair was chosen")
        if not 0.0 <= self.aoi_coverage_fraction <= 1.0:
            raise ValueError("aoi_coverage_fraction must be in [0, 1]")

        acquisitions = (*self.pre, self.post)
        for acq in acquisitions:
            if acq.sensor is not Sensor.SENTINEL_1:
                raise ValueError(f"{acq.source_id}: pairs contain Sentinel-1 acquisitions only")
            if acq.track_key != (self.orbit_direction, self.relative_orbit):
                raise ValueError(
                    f"{acq.source_id}: track {acq.track_key} differs from the pair track "
                    f"{(self.orbit_direction, self.relative_orbit)}; cross-track pixel comparison is not allowed"
                )
        ids = [a.source_id for a in acquisitions]
        if len(set(ids)) != len(ids):
            raise ValueError("an acquisition appears more than once in the pair")
        starts = [a.acquisition_start for a in self.pre]
        if starts != sorted(starts):
            raise ValueError("pre-event acquisitions must be sorted oldest first")

        self._check_event_ordering()

        if not set.intersection(*(set(a.polarizations) for a in acquisitions)):
            raise ValueError("pre and post acquisitions share no polarization")
        single = PairQualityFlag.SINGLE_PRE_IMAGE in self.quality_flags
        if (len(self.pre) == 1) != single:
            raise ValueError("SINGLE_PRE_IMAGE flag must be set if and only if there is exactly one pre image")
        return self

    def _check_event_ordering(self) -> None:
        if self.event_time_utc is not None:
            event_dt = datetime.combine(self.event_date, self.event_time_utc, tzinfo=timezone.utc)
            if any(a.acquisition_start >= event_dt for a in self.pre):
                raise ValueError("every pre-event acquisition must start before the event time")
            if self.post.acquisition_start <= event_dt:
                raise ValueError("the post-event acquisition must start after the event time")
            return
        # Event time unknown: same-day acquisitions are ambiguous and excluded.
        if any(a.acquisition_start.date() >= self.event_date for a in self.pre):
            raise ValueError(
                "pre-event acquisitions must be dated before the event date; same-day images are "
                "ambiguous without an event time"
            )
        if self.post.acquisition_start.date() <= self.event_date:
            raise ValueError(
                "the post-event acquisition must be dated after the event date; same-day images are "
                "ambiguous without an event time"
            )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pre_post_gap_days(self) -> float:
        """Days between the latest pre image and the post image."""
        return round((self.post.acquisition_start - self.pre[-1].acquisition_start).total_seconds() / 86400, 3)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def post_delay_days(self) -> float:
        """Days from the start of the event date (or event time) to the post image."""
        t = self.event_time_utc or time(0, 0)
        event_dt = datetime.combine(self.event_date, t, tzinfo=timezone.utc)
        return round((self.post.acquisition_start - event_dt).total_seconds() / 86400, 3)

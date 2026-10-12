"""Contract F - road-access status of one settlement.

Connectivity is computed on three graphs built from the same pre-event OSM
road network (engine: ``bato.connectivity``):

* pre-event          - all roads;
* post, optimistic   - BLOCKED segments removed, UNOBSERVABLE kept;
* post, pessimistic  - BLOCKED and UNOBSERVABLE segments removed.

Each graph gives the shortest route from the settlement to the nearest target
(hospital, town, or an exit at the edge of the analysis area). The final
status is a pure function of those three results - :func:`classify_settlement`
- and the contract refuses to store a status that contradicts them.

``building_count`` is a count of OSM building footprints. It is an impact
indicator, NOT a population estimate; the PS lists no population input.
"""

from __future__ import annotations

from enum import Enum
from itertools import pairwise

from pydantic import model_validator

from bato.contracts.common import Contract, PointGeometry

_ROUTE_TOLERANCE_M = 1e-6


class SettlementSource(str, Enum):
    OSM_PLACE = "osm_place"                # place=city/town/village/hamlet/... node or area
    BUILDING_CLUSTER = "building_cluster"  # unnamed cluster of OSM buildings


class TargetKind(str, Enum):
    HOSPITAL = "hospital"
    TOWN = "town"
    ANALYSIS_EXIT = "analysis_exit"        # road leaving the buffered analysis area


class AccessStatus(str, Enum):
    CONNECTED = "connected"
    DEGRADED = "degraded"                        # still reachable, but a much longer route
    POSSIBLY_CUT_OFF = "possibly_cut_off"        # depends on UNOBSERVABLE segments
    CONFIRMED_CUT_OFF = "confirmed_cut_off"      # unreachable even if every UNOBSERVABLE segment is open
    NO_PRE_EVENT_ROAD_ACCESS = "no_pre_event_road_access"  # not a flood effect


class Reachability(Contract):
    """Result of a shortest-route search on one graph."""

    reachable: bool
    target_id: str | None = None
    target_kind: TargetKind | None = None
    route_length_m: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Reachability:
        details = (self.target_id, self.target_kind, self.route_length_m)
        if self.reachable and any(v is None for v in details):
            raise ValueError("a reachable result needs target_id, target_kind and route_length_m")
        if not self.reachable and any(v is not None for v in details):
            raise ValueError("an unreachable result must not carry target or route details")
        if self.route_length_m is not None and self.route_length_m < 0:
            raise ValueError("route_length_m must be >= 0")
        return self


def classify_settlement(
    pre_event: Reachability,
    post_optimistic: Reachability,
    post_pessimistic: Reachability,
    degraded_detour_ratio: float,
) -> AccessStatus:
    """The decision table from docs/architecture.md, as code.

    DEGRADED compares the pessimistic route with the pre-event route, so a
    settlement is only called "connected" when that holds under the cautious
    assumption that UNOBSERVABLE segments are closed.
    """
    if degraded_detour_ratio <= 1.0:
        raise ValueError("degraded_detour_ratio must be > 1")
    if not pre_event.reachable:
        return AccessStatus.NO_PRE_EVENT_ROAD_ACCESS
    if not post_optimistic.reachable:
        return AccessStatus.CONFIRMED_CUT_OFF
    if not post_pessimistic.reachable:
        return AccessStatus.POSSIBLY_CUT_OFF
    before = pre_event.route_length_m or 0.0
    after = post_pessimistic.route_length_m or 0.0
    if after > before and after >= degraded_detour_ratio * before:
        return AccessStatus.DEGRADED
    return AccessStatus.CONNECTED


class SettlementStatus(Contract):
    settlement_id: str
    name: str | None = None
    source: SettlementSource
    location: PointGeometry
    building_count: int                         # OSM buildings; NOT population
    snap_distance_m: float | None = None        # None: no road within the snapping limit
    pre_event: Reachability
    post_optimistic: Reachability
    post_pessimistic: Reachability
    degraded_detour_ratio: float                # threshold used, kept for reproducibility
    status: AccessStatus
    blocking_segment_ids: tuple[str, ...] = ()  # segments that separate it from every target
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> SettlementStatus:
        if self.building_count < 0:
            raise ValueError("building_count must be >= 0")
        if self.snap_distance_m is not None and self.snap_distance_m < 0:
            raise ValueError("snap_distance_m must be >= 0")
        if self.snap_distance_m is None and self.pre_event.reachable:
            raise ValueError("a settlement not snapped to any road cannot be reachable")

        # Post-event graphs only ever lose edges, so reachability and route
        # length are monotone: pre >= optimistic >= pessimistic.
        chain = (self.pre_event, self.post_optimistic, self.post_pessimistic)
        for earlier, later in pairwise(chain):
            if later.reachable and not earlier.reachable:
                raise ValueError("removing roads cannot make a settlement reachable")
            if later.reachable and earlier.reachable:
                if (later.route_length_m or 0) + _ROUTE_TOLERANCE_M < (earlier.route_length_m or 0):
                    raise ValueError("removing roads cannot shorten the shortest route")

        expected = classify_settlement(self.pre_event, self.post_optimistic, self.post_pessimistic,
                                       self.degraded_detour_ratio)
        if self.status is not expected:
            raise ValueError(f"status {self.status.value} contradicts the reachability results "
                             f"(expected {expected.value})")
        if self.status in (AccessStatus.CONFIRMED_CUT_OFF, AccessStatus.POSSIBLY_CUT_OFF):
            if not self.blocking_segment_ids or not self.evidence_ids:
                raise ValueError("a cut-off status must name blocking segments and reference evidence")
        return self

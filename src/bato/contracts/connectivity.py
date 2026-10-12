"""Connectivity outputs (bato.connectivity.engine): access states, blocked-segment priorities, run provenance.

User-facing access states map the contract statuses of :mod:`bato.contracts.settlement`:

==================== ==============================================================
CONFIRMED_CUTOFF     ``confirmed_cut_off`` - no route even if every UNOBSERVABLE road is open
POSSIBLE_CUTOFF      ``possibly_cut_off``  - a route exists only through UNOBSERVABLE roads
CONNECTED            ``connected`` or ``degraded`` (route on observed-open roads; ``degraded`` = much longer)
NO_PRE_EVENT_ACCESS  ``no_pre_event_road_access`` - no road within the snapping distance, or no route before
                     the event either (not a flood effect)
==================== ==============================================================

``building_count`` values are counts of mapped OSM building footprints - a proxy for settlement
size, never a population figure.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import model_validator

from bato.contracts.baseline import ThresholdRecord
from bato.contracts.common import Contract, RelativePath, Sha256Hex, UTCDatetime
from bato.contracts.osm import OSMSnapshot
from bato.contracts.settlement import AccessStatus

CONNECTIVITY_VERSION: Literal["bato-connectivity-v1"] = "bato-connectivity-v1"


class AccessState(str, Enum):
    CONFIRMED_CUTOFF = "CONFIRMED_CUTOFF"
    POSSIBLE_CUTOFF = "POSSIBLE_CUTOFF"
    CONNECTED = "CONNECTED"
    NO_PRE_EVENT_ACCESS = "NO_PRE_EVENT_ACCESS"


ACCESS_STATE_BY_STATUS: dict[AccessStatus, AccessState] = {
    AccessStatus.CONFIRMED_CUT_OFF: AccessState.CONFIRMED_CUTOFF,
    AccessStatus.POSSIBLY_CUT_OFF: AccessState.POSSIBLE_CUTOFF,
    AccessStatus.CONNECTED: AccessState.CONNECTED,
    AccessStatus.DEGRADED: AccessState.CONNECTED,
    AccessStatus.NO_PRE_EVENT_ROAD_ACCESS: AccessState.NO_PRE_EVENT_ACCESS,
}


class BlockedSegmentPriority(Contract):
    """What clearing ONE blocked stretch would achieve (all other road states unchanged).

    A blocked stretch is a connected run of BLOCKED segments (adjacent segments sharing a node): one
    flooded spot usually covers several 50 m segments, and clearing only one of them changes nothing.
    Every segment of the stretch reports the stretch's numbers.
    """

    rank: int
    stretch_id: str
    segment_ids: tuple[str, ...]
    osm_way_ids: tuple[int, ...]
    highways: tuple[str, ...]
    names: tuple[str, ...] = ()
    bridge: bool
    length_m: float
    affected_length_m: float
    longest_flood_run_m: float
    # post-optimistic graph (UNOBSERVABLE kept): CONFIRMED_CUTOFF settlements that regain a route
    settlements_reconnected: int
    reconnected_settlement_ids: tuple[str, ...]
    reconnected_building_count: int          # mapped OSM buildings (proxy, not population)
    # post-pessimistic graph (UNOBSERVABLE removed): settlements that regain a route on observed-open roads
    settlements_reconnected_on_observed_roads: int
    reconnected_on_observed_roads_ids: tuple[str, ...]
    # confirmed cut-off settlements whose minimum clearing set (0-1 BFS) contains this stretch
    in_minimum_clearing_set_of: int
    evidence_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> BlockedSegmentPriority:
        if self.rank < 1:
            raise ValueError("rank starts at 1")
        if not self.segment_ids or len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError("a stretch needs one or more distinct segment ids")
        if self.settlements_reconnected != len(self.reconnected_settlement_ids):
            raise ValueError("settlements_reconnected must equal the number of reconnected settlement ids")
        if self.settlements_reconnected_on_observed_roads != len(self.reconnected_on_observed_roads_ids):
            raise ValueError("observed-road count must equal the number of ids")
        if min(self.reconnected_building_count, self.in_minimum_clearing_set_of) < 0:
            raise ValueError("counts must be >= 0")
        if not self.evidence_ids:
            raise ValueError("a blocked stretch must reference evidence")
        return self


class ConnectivitySummary(Contract):
    road_segments_by_state: dict[str, int]
    road_km_by_state: dict[str, float]
    unobservable_km_by_reason: dict[str, float]      # sampled length by Step 4C NOT_ASSESSED reason
    bridges_by_state: dict[str, int]
    exit_nodes: int
    targets_by_kind: dict[str, int]
    unsnapped_targets: int
    settlements_total: int
    settlements_by_source: dict[str, int]
    settlements_by_access_state: dict[str, int]
    settlements_by_status: dict[str, int]
    building_count_by_access_state: dict[str, int]   # proxy, not population
    reachable_settlements: dict[str, int]            # pre_event / post_optimistic / post_pessimistic
    isolated_components_with_settlements: dict[str, int]
    blocked_segments: int
    blocked_stretches: int
    blocked_stretches_reconnecting_alone: int

    @model_validator(mode="after")
    def _check(self) -> ConnectivitySummary:
        if sum(self.settlements_by_access_state.values()) != self.settlements_total:
            raise ValueError("access-state counts must sum to settlements_total")
        if sum(self.settlements_by_status.values()) != self.settlements_total:
            raise ValueError("status counts must sum to settlements_total")
        r = self.reachable_settlements
        if not r.get("pre_event", 0) >= r.get("post_optimistic", 0) >= r.get("post_pessimistic", 0):
            raise ValueError("removing roads cannot increase the number of reachable settlements")
        return self


class FileRef(Contract):
    path: RelativePath
    sha256: Sha256Hex


class ConnectivityProvenance(Contract):
    kind: Literal["road_connectivity"] = "road_connectivity"
    version: Literal["bato-connectivity-v1"] = CONNECTIVITY_VERSION
    road_rule: Literal["v1", "v2"] = "v1"        # v2: FLOOD or DISTURBANCE; only geometry makes a road unobservable
    bato_version: str
    config_fingerprint: Sha256Hex
    baseline_config_fingerprint: str
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    candidate_id: str
    event_date: str
    osm_snapshot: OSMSnapshot
    inputs: dict[str, FileRef]
    thresholds: tuple[ThresholdRecord, ...]
    rules: dict[str, str]
    outputs: dict[str, FileRef]
    summary: ConnectivitySummary
    top_blocked_segments: tuple[BlockedSegmentPriority, ...]
    inputs_not_used: tuple[str, ...] = ("population datasets (WorldPop/GHSL/...)", "EMSR / UNOSAT / any damage map",
                                        "post-event OSM", "FloodViT outputs")
    validated: Literal[False] = False
    validation_note: str = ("NOT VALIDATED: built on the provisional Step 4C baseline and provisional "
                            "road/connectivity thresholds; no reference map was used to set or check anything")
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> ConnectivityProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.osm_snapshot.snapshot_date.isoformat() >= self.event_date:
            raise ValueError("the OSM snapshot must be strictly before the event date")
        ranks = [p.rank for p in self.top_blocked_segments]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("top_blocked_segments must be ranked 1..n")
        return self

"""Contract E - impact on one road segment.

Roads from the pre-event OSM snapshot are split into short segments so an
impact can be localised. Each segment gets one of three states:

* OPEN          - observed, and no qualifying impact run.
* BLOCKED       - a qualifying contiguous run lies in flood water or debris.
* UNOBSERVABLE  - a run long enough to hide a blocking impact could not be
                  observed (terrain gates, no SAR, no DEM). Never converted to
                  OPEN or BLOCKED; propagated as "possibly cut off".

The rules that assign a state live in ``bato.impact.roads``. This contract
only enforces internal consistency.

Bridges are described by *exposure*, never by structural state: 10 m radar
can show that a bridge lies inside the impact zone, not that it collapsed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import computed_field, model_validator

from bato.contracts.common import Contract, LineStringGeometry

_LENGTH_TOLERANCE_M = 1e-6


class RoadState(str, Enum):
    OPEN = "open"
    BLOCKED = "blocked"
    UNOBSERVABLE = "unobservable"


class BridgeExposure(str, Enum):
    NOT_A_BRIDGE = "not_a_bridge"
    NOT_EXPOSED = "not_exposed"
    EXPOSED = "exposed_to_impact_zone"
    # Deliberately no "collapsed"/"destroyed" value - see module docstring.


class RoadSegmentImpact(Contract):
    segment_id: str
    osm_way_id: int
    highway: str                            # OSM highway=* value of the parent way
    geometry: LineStringGeometry
    length_m: float
    state: RoadState
    affected_length_m: float                # total length inside flood water / debris
    longest_affected_run_m: float           # longest contiguous affected run
    unobservable_length_m: float            # length in radar shadow/layover or no data
    max_impact_probability: float | None = None
    bridge: BridgeExposure = BridgeExposure.NOT_A_BRIDGE
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> RoadSegmentImpact:
        if self.osm_way_id <= 0:
            raise ValueError("osm_way_id must be positive")
        if self.length_m <= 0:
            raise ValueError("length_m must be positive")
        limit = self.length_m + _LENGTH_TOLERANCE_M
        for name in ("affected_length_m", "longest_affected_run_m", "unobservable_length_m"):
            value = getattr(self, name)
            if not 0 <= value <= limit:
                raise ValueError(f"{name} must be in [0, length_m]")
        if self.longest_affected_run_m > self.affected_length_m + _LENGTH_TOLERANCE_M:
            raise ValueError("longest_affected_run_m cannot exceed affected_length_m")
        if self.affected_length_m + self.unobservable_length_m > limit:
            raise ValueError("affected and unobservable lengths overlap or exceed length_m")
        p = self.max_impact_probability
        if p is not None and not 0.0 <= p <= 1.0:
            raise ValueError("max_impact_probability must be in [0, 1]")

        if self.state is RoadState.BLOCKED and self.affected_length_m <= 0:
            raise ValueError("a BLOCKED segment must have affected length > 0")
        if self.state is RoadState.UNOBSERVABLE and self.unobservable_length_m <= 0 and p is None:
            raise ValueError("an UNOBSERVABLE segment needs unobservable length or an impact probability")
        if self.state is not RoadState.OPEN and not self.evidence_ids:
            raise ValueError(f"a {self.state.value.upper()} segment must reference evidence")
        if self.bridge is BridgeExposure.EXPOSED and self.affected_length_m <= 0:
            raise ValueError("a bridge can only be EXPOSED if part of it is affected")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observable_fraction(self) -> float:
        return round(1.0 - self.unobservable_length_m / self.length_m, 6)

"""Contract G - evidence record.

Every important output (a blocked road, a cut-off settlement, a detected
region, a chosen image pair) must point at one or more evidence records that
say which data, dates, place, method/version, configuration, probabilities
and intermediate files support it. The dashboard and situation report will
read numbers from evidence, never compute or invent their own.

Two data-rule safeguards live here:

* ``sources`` uses :class:`~bato.contracts.common.DataSource`, which only lists
  permitted inputs, so evidence cannot cite a published damage map.
* ``claim`` text is checked against terms the system has no basis to assert
  (structural collapse, casualties, population).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from enum import Enum

from pydantic import model_validator

from bato.contracts.common import (
    AnyGeometry,
    Contract,
    DataSource,
    RelativePath,
    Sha256Hex,
    UTCDatetime,
)

# Lower-case fragments that system-generated claims must never contain.
UNSUPPORTED_CLAIM_TERMS: tuple[str, ...] = (
    "collapse", "destroyed", "washed away", "casualt", "dead", "death", "killed",
    "injur", "population", "people affected", "residents",
)


class SubjectType(str, Enum):
    PAIR = "pair"
    DETECTION_REGION = "detection_region"
    ROAD_SEGMENT = "road_segment"
    BUILDING = "building"
    SETTLEMENT = "settlement"


class SourceRef(Contract):
    """One input dataset item that supports a claim."""

    source: DataSource
    source_id: str                          # product id, OSM snapshot id, model id, ...
    reference_date: date                    # acquisition date or OSM snapshot date
    acquisition_start: UTCDatetime | None = None

    @model_validator(mode="after")
    def _check(self) -> SourceRef:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.acquisition_start is not None and self.acquisition_start.date() != self.reference_date:
            raise ValueError("reference_date must equal the UTC date of acquisition_start")
        return self


class ArtifactRef(Contract):
    """An intermediate file that a reviewer can open to check the claim."""

    path: RelativePath                      # relative to the project root
    kind: str                               # e.g. "geotiff", "geojson", "png_chip"
    sha256: Sha256Hex | None = None


class EvidenceRecord(Contract):
    evidence_id: str
    subject_type: SubjectType
    subject_id: str
    claim: str                              # short, factual, human-readable
    location: AnyGeometry
    sources: tuple[SourceRef, ...]
    method: str                             # e.g. "rule:road_blocked_run", "model:flood_unet"
    method_version: str                     # BATO version or model version
    config_fingerprint: Sha256Hex
    probability: float | None = None
    values: dict[str, float] = {}           # measured quantities, e.g. {"debris_z_vv": 4.1}
    artifacts: tuple[ArtifactRef, ...] = ()
    created_at: UTCDatetime

    @model_validator(mode="after")
    def _check(self) -> EvidenceRecord:
        for name in ("evidence_id", "subject_id", "claim", "method", "method_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not self.sources:
            raise ValueError("evidence must cite at least one source")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        bad_values = [k for k, v in self.values.items() if not math.isfinite(v)]
        if bad_values:
            raise ValueError(f"values must be finite numbers: {bad_values}")
        lowered = self.claim.lower()
        found = [t for t in UNSUPPORTED_CLAIM_TERMS if t in lowered]
        if found:
            raise ValueError(f"claim contains terms the system cannot support: {found}")
        return self


def make_evidence_id(subject_type: SubjectType, subject_id: str, method: str,
                     config_fingerprint: str, source_ids: list[str]) -> str:
    """Deterministic evidence id: rerunning the same inputs and configuration
    gives the same id, which makes runs diffable."""
    payload = json.dumps(
        {"subject_type": subject_type.value, "subject_id": subject_id, "method": method,
         "config": config_fingerprint, "sources": sorted(source_ids)},
        sort_keys=True, separators=(",", ":"),
    )
    return "ev-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

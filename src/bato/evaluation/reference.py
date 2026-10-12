"""Evaluation-only reference datasets.

Published damage maps (Copernicus EMS, e.g. activation EMSR927; UNOSAT) and
post-event OSM are allowed ONLY to check results. They are described here,
inside :mod:`bato.evaluation`, which pipeline packages are forbidden to import
(enforced by tests/unit/test_data_rules.py).

Loading and metric computation are NOT IMPLEMENTED YET.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from bato.contracts.common import Contract, RelativePath


class ReferenceSource(str, Enum):
    COPERNICUS_EMS = "copernicus-ems"
    UNOSAT = "unosat"
    OTHER_PUBLISHED_DAMAGE_MAP = "other-published-damage-map"
    OSM_POST_EVENT = "osm-post-event"
    # Team-made labels on permitted imagery, for evaluation only. Pending
    # written confirmation from the organisers (docs/assumptions.md, Q3).
    TEAM_ANNOTATED_EVAL_LABELS = "team-annotated-eval-labels"


class ReferenceDataset(Contract):
    reference_id: str          # e.g. "EMSR927"
    source: ReferenceSource
    event_name: str
    path: RelativePath         # must be under the evaluation-only folders
    attribution: str
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ReferenceDataset:
        for name in ("reference_id", "event_name", "attribution"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        return self

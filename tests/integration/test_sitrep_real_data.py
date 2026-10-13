"""OPT-IN: generate the SITREP from the REAL frozen v1 connectivity run (read-only; nothing written) and check
it against the official Trishuli values of the technical report.

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_sitrep_real_data.py -v -s
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.dashboard.sitrep import EMSR927_STATEMENT, load_facts, render
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_sitrep_matches_the_official_values(cfg):
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    candidate = stk.select_candidate(fetch, None).candidate_id
    f, notes = load_facts(cfg, fetch.aoi_id, candidate)
    print(f"\nrun {f.connectivity_run} | notes {notes}")
    assert (f.aoi_id, f.candidate_id, f.road_rule) == ("aoi-e893707c2beb", "ts-80841ae62417", "v1")
    assert f.road_segments == {"blocked": 30, "open": 10323, "unobservable": 61499}
    assert f.road_km == {"blocked": 1.227, "open": 463.714, "unobservable": 2942.161}
    assert f.blocked_stretches == 9 and f.settlements_total == 265
    assert f.access == {"CONNECTED": 1, "POSSIBLE_CUTOFF": 163, "CONFIRMED_CUTOFF": 0, "NO_PRE_EVENT_ACCESS": 101}
    assert round(f.not_assessed_fraction * 100, 1) == 91.2
    assert dict(f.not_assessed_reasons) == {"NOT_ASSESSED_STEEP": 35544186, "NOT_ASSESSED_LAYOVER": 4713918,
                                            "NOT_ASSESSED_SHADOW": 1027803}
    # The provenance keeps sub-second precision (12:21:41.762029); the SITREP shows whole seconds (truncated).
    assert f.event_date == "2026-08-26"
    assert f.post.acquired.replace(microsecond=0) == datetime(2026, 8, 28, 12, 21, 41, tzinfo=timezone.utc)
    t = render(f, title="Trishuli Flood, Nepal", incident="real-data test",
               issued_at=datetime(2026, 10, 13, 7, 0, tzinfo=timezone.utc))
    for s in ("**30 road segments in 9 stretches (1.227 km)**", "463.714 km", "2,942.161 km",
              "**163 settlements have access unverified**", "**0 settlements are confirmed cut off**",
              "| Confirmed cutoff | **0** |", "**91.2 % of scene pixels were not assessed**", EMSR927_STATEMENT,
              "**POST 2026-08-28 12:21:41 UTC**", "**Observed** 2026-08-28 12:21 UTC",
              "no observation after 2026-08-28 12:21 UTC"):
        assert s in t, s
    assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d", t), "acquisition times must be shown without fractional seconds"

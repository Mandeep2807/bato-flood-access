"""OPT-IN: build the dashboard bundle from the REAL frozen v1 connectivity run (read-only; nothing published).

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_dashboard_real_data.py -v -s

Checks the integrity rules (official v1 run, hashes, summary recomputed from the layers, verified route traces)
and prints the headline numbers. It does not judge accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.dashboard.bundle import build_bundle, prepare_inputs
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_dashboard_bundle(cfg, tmp_path):
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    inp = prepare_inputs(cfg, fetch.aoi_id, stk.select_candidate(fetch, None).candidate_id, title="real-data test")
    b = build_bundle(inp, tmp_path, progress=print)
    s = b["summary"]
    assert b["incident"]["road_rule"] == "v1" and b["incident"]["validated"] is False
    assert b["roads"]["count"] == sum(s["road_segments_by_state"].values())
    assert sum(1 for x in b["roads"]["state"] if x == 0) == s["road_segments_by_state"]["blocked"]
    tr = b["integrity"]["route_traces"]
    print(f"\nkm {s['road_km_by_state']} | settlements {s['settlements_by_access_state']} | "
          f"reachable {s['reachable_settlements']} | traces {tr} | experiment {b['experiment']['verdict']}")
    assert tr["verified"] > 0
    for st in b["settlements"]:
        for t in st["traces"].values():
            assert abs(t["length_m"] - st["routes"][t["graph"]]["length_m"]) <= 0.6

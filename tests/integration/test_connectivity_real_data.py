"""OPT-IN: connectivity on the REAL Trishuli baseline + pre-event OSM extract (read-only; nothing published).

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_connectivity_real_data.py -v -s

Needs scripts/run_baseline_detection.py and scripts/fetch_osm_snapshot.py to have run. Checks invariants and
prints the summary; it does not judge accuracy (no reference data is used).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bato.connectivity import engine
from bato.contracts.fetch import FetchManifest
from bato.contracts.impact import RoadState
from bato.contracts.io import load_contract
from bato.contracts.settlement import AccessStatus
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_connectivity_invariants(cfg):
    pytest.importorskip("rasterio")
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    job = engine.prepare_connectivity(cfg, fetch.aoi_id, stk.select_candidate(fetch, None).candidate_id)
    state, transform = engine.read_state_raster(job.state_path)
    result = engine.compute_connectivity(job, cfg, state, transform, progress=print)
    s = result.summary
    print(f"\n{s.model_dump_json(indent=2)}")
    for p in result.priorities[:10]:
        print(p.rank, p.stretch_id, len(p.segment_ids), p.osm_way_ids[:3], p.settlements_reconnected,
              p.reconnected_building_count, p.settlements_reconnected_on_observed_roads, p.in_minimum_clearing_set_of)
    assert job.osm.provenance.snapshot.snapshot_date < job.event_date
    for imp in result.segment_impacts:
        assert imp.state is RoadState.OPEN or imp.evidence_ids
    for r in result.settlements:
        if r.status.status in (AccessStatus.CONFIRMED_CUT_OFF, AccessStatus.POSSIBLY_CUT_OFF):
            assert r.status.blocking_segment_ids and r.status.evidence_ids

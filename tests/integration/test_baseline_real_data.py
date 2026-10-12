"""OPT-IN: the deterministic SAR change baseline on the REAL Trishuli stack (read-only; nothing published).

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_baseline_real_data.py -v -s

Checks the invariants of the result on real data (states, denominators, both-pre darkening of every flood
pixel, cluster filter) and prints the statistics. It does not judge accuracy: no reference data is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bato.contracts.baseline import NOT_ASSESSED_STATES, BaselineState
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.detection import baseline as bl
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_baseline_invariants(cfg):
    pytest.importorskip("rasterio")
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    job = bl.prepare_baseline(cfg, fetch.aoi_id, stk.select_candidate(fetch, None).candidate_id)
    maps = bl.compute_baseline(job, cfg, progress=print)
    s = maps.statistics
    print(f"\nDEM: {job.dem_path}\nstates: {s.state_counts}")
    print(f"flood {s.flood_pixels:,} = {s.flood_fraction_of_assessed} of assessed, {s.flood_fraction_of_all} of all; "
          f"not assessed {s.not_assessed_fraction_of_all:.2%}")
    print(f"slope {s.flood_by_slope_deg}\ngeometry {s.flood_by_geometry}\ncomponents {s.components}")
    print(f"FloodViT comparison counts: {maps.comparison_counts}")
    allowed = {int(x) for x in BaselineState}
    assert set(np.unique(maps.state).tolist()) <= allowed
    assert sum(s.state_counts.values()) == maps.state.size == s.total_pixels
    assert s.flood_darkened_vs_pre2 == s.flood_darkened_vs_pre1 == s.flood_pixels
    assert s.flood_by_geometry["shadow_likely"] == s.flood_by_geometry["layover_likely"] == 0
    assert s.components.candidate_pixels - s.components.removed_pixels == s.flood_pixels
    na = np.isin(maps.state, [int(x) for x in NOT_ASSESSED_STATES])
    assert int(na.sum()) == s.not_assessed_pixels
    assert not job.final_dir.exists() or job.final_dir.is_dir()            # nothing published by this test

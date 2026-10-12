"""OPT-IN: check the REAL Step 3.4 outputs of the Trishuli candidate (read-only, nothing written).

Skipped unless BATO_REAL_DATA_MANIFEST points to the real Step 2C fetch manifest, e.g.

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_stack_real_data.py -v -s

For every pass of the candidate that has a published terrain/ output, the stack's input checks run
(files re-hashed against the Step 3.4 provenance, EPSG:3857, 10 m, Sigma0_VV + Sigma0_VH linear
float32, no-data 0). When all three passes are present, the full stack plan (roles, event order,
common grid) is computed too. Uses the repository configuration and data folders.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.contracts.stack import ROLES
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_terrain_outputs_are_valid_stack_inputs(cfg):
    pytest.importorskip("rasterio")
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    manifest = load_contract(FetchManifest, path)
    candidate = stk.select_candidate(manifest, None)
    pass_ids = (*candidate.pre_event_pass_ids, candidate.post_event_pass_id)
    present = []
    for pid, role in zip(pass_ids, ROLES, strict=True):
        if not (stk.stage_dir(cfg, manifest.aoi_id, pid, "terrain") / "preprocessing_provenance.json").is_file():
            print(f"\n{role} {pid}: no Step 3.4 output yet")
            continue
        inp = stk.load_terrain_input(cfg, manifest, pid, role)
        print(f"\n{role} {pid}: OK - {inp.product.width} x {inp.product.height} px, origin ({inp.i2m[4]}, "
              f"{inp.i2m[5]}), {len(inp.files_sha256)} files verified")
        present.append(role)
    assert present, "no Step 3.4 output found for the candidate"
    if len(present) == 3:
        job = stk.prepare_stack(manifest, cfg, candidate.candidate_id)
        g = job.grid
        print(f"stack plan: {g.width} x {g.height} px, origin ({g.transform[2]}, {g.transform[5]}), windows "
              f"{list(job.windows)}")
        assert g.width > 0 and g.height > 0
    else:
        print(f"stack plan skipped: Step 3.4 outputs present for {present} only")

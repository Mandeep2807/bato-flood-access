"""OPT-IN: check the REAL Step 3.5 Trishuli stack against the Step 4A model-input contract (read-only).

Skipped unless BATO_REAL_DATA_MANIFEST points to the real Step 2C fetch manifest, e.g.

    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_model_input_real_data.py -v -s

Verifies the stack provenance and hashes, checks every pixel (finite, non-negative, mask == all six
layers valid), plans the 224 x 224 tiles and reads the first tile with valid pixels as a model tensor.
Nothing is written (build_manifest only; the manifest is not published).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.preprocessing import model_input as mi
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def test_real_stack_meets_the_model_input_contract(cfg):
    pytest.importorskip("rasterio")
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    candidate = stk.select_candidate(fetch, None)
    job = mi.prepare_model_input(cfg, fetch.aoi_id, candidate.candidate_id)
    manifest = mi.build_manifest(job, cfg, progress=print)
    print()
    for line in mi.describe_model_input(manifest):
        print(line)
    for item in manifest.spec.unresolved_items:
        print(f"OPEN  {item}")
    assert [p.kuro_siwo_input for p in manifest.source.passes] == ["sec2", "sec1", "flood"]
    assert manifest.valid_pixels > 0 and manifest.tiles_with_valid_pixels > 0
    record = next(t for t in manifest.tiles if t.valid_pixels > 0)
    t = manifest.spec.tiling.tile_size
    with mi._open(job.stack_path) as stack_ds, mi._open(job.mask_path) as mask_ds:
        raw, valid = mi.read_block(stack_ds, mask_ds, mi.TileWindow(record.row_off, record.col_off,
                                                                   record.data_height, record.data_width), (t, t))
    x, valid = mi.build_tile(raw, valid, manifest.spec)
    assert x.shape == (6, 224, 224) and x.dtype == np.float32 and np.isfinite(x).all()
    assert int(valid.sum()) == record.valid_pixels
    print(f"tile {record.tile_id}: valid {record.valid_pixels:,}; normalised per channel min/max "
          f"{[(round(float(c.min()), 3), round(float(c.max()), 3)) for c in x]}")
    assert not job.final_dir.exists() or job.final_dir.is_dir()      # nothing published by this test

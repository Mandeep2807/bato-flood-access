"""TEST FIXTURE for the SITREP generator: the synthetic road world of connectivity_fakes / dashboard_fakes,
published as a road-rule-v1 run, with the stack grid and Step 4C statistics fields the SITREP prints
(aoi_bbox, analysis_buffer_km, per-state pixel counts). Invented data, NOT real results.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bato.connectivity.engine import run_connectivity
from bato.contracts.baseline import BaselineState
from bato.osm.extract import load_osm_extract
from bato.preprocessing.speckle import sha256_file

from connectivity_fakes import AOI, EVENT, NOW, fake_job, fetch, grid, project, state_raster
from dashboard_fakes import baseline_json, stack_json


def full_statistics(state: np.ndarray) -> dict[str, Any]:
    counts = {s.name: int((state == int(s)).sum()) for s in BaselineState}
    assessed = counts["NO_FLOOD"] + counts["FLOOD"] + counts["PRE_EVENT_WATER"]
    total = int(state.size)
    return {"total_pixels": total, "assessed_pixels": assessed, "not_assessed_pixels": total - assessed,
            "state_counts": counts, "flood_pixels": counts["FLOOD"], "no_flood_pixels": counts["NO_FLOOD"],
            "pre_event_water_pixels": counts["PRE_EVENT_WATER"],
            "not_assessed_fraction_of_all": (total - assessed) / total}


def published_sitrep_world(tmp_path: Path, *, stack_edit=None, baseline_edit=None,
                           extra_v2: bool = False) -> SimpleNamespace:
    """Publish the synthetic v1 run. ``stack_edit`` / ``baseline_edit`` mutate those provenance dicts BEFORE the
    connectivity run, so their hashes stay consistent (to test the SITREP's own checks)."""
    cfg = project(tmp_path)
    fetch(cfg)
    extract = load_osm_extract(cfg, "aoi-test", EVENT)
    state, transform = state_raster()
    job = fake_job(cfg, extract, tmp_path, state)
    _t, (h, w) = grid()
    rel = lambda p: p.resolve().relative_to(tmp_path.resolve()).as_posix()  # noqa: E731
    stack = stack_json(cfg, h, w, transform)
    stack["grid"]["aoi_bbox"] = AOI.model_dump(mode="json")
    stack["grid"]["analysis_buffer_km"] = 15.0
    stack["aoi_id"], stack["candidate_id"] = "aoi-test", "ts-test"
    if stack_edit:
        stack_edit(stack)
    job.stack_provenance_path.write_text(json.dumps(stack), encoding="utf-8")
    job.baseline.source.provenance_sha256 = sha256_file(job.stack_provenance_path)
    base = baseline_json(state, rel(job.state_path), job.baseline.outputs.state_sha256, "c" * 64)
    base["statistics"] = full_statistics(state)
    base.update({"validated": False, "aoi_id": "aoi-test", "candidate_id": "ts-test", "event_date": EVENT.isoformat()})
    if baseline_edit:
        baseline_edit(base)
    job.baseline_provenance_path.write_text(json.dumps(base), encoding="utf-8")
    job.baseline_provenance_sha256 = sha256_file(job.baseline_provenance_path)
    out, prov, result = run_connectivity(job, cfg, read_state=lambda _p: (state, transform), now=lambda: NOW)
    if extra_v2:
        run_connectivity(job, cfg, overwrite=True, read_state=lambda _p: (state, transform),
                         now=lambda: NOW.replace(hour=7))
        p = out / "connectivity_provenance.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["road_rule"] = "v2"
        p.write_text(json.dumps(data), encoding="utf-8")
    return SimpleNamespace(cfg=cfg, root=tmp_path, out=out, prov=prov, result=result, state=state)

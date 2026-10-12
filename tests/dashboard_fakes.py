"""TEST FIXTURE: a published road-rule-v1 connectivity run on the synthetic world of connectivity_fakes, with
stand-in stack / Step 4C provenance files and in-memory rasters (invented backscatter and terrain, NOT real data).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bato.config import BatoConfig
from bato.connectivity.engine import run_connectivity
from bato.dashboard.bundle import BuildInputs, RasterSource, prepare_inputs
from bato.osm.extract import load_osm_extract
from bato.preprocessing.speckle import sha256_file

from connectivity_fakes import ANALYSIS, AOI_RING, EVENT, NOW, PIXEL, fake_job, fetch, grid, project, state_raster

DEBRIS_EVENT = {"aoi_id": "aoi-test", "candidate_id": "ts-test", "z": 3.0, "statistics": {
    "disturbance_pixels": 8714, "disturbance_fraction_of_assessed": 0.000221, "facing_sensor_share": 0.131,
    "edge_pixels_share": 0.001148, "assessed_edge_share": 0.060758}}
DEBRIS_CONTROL = {"aoi_id": "aoi-control", "candidate_id": "ts-control", "z": 3.0, "statistics": {
    "disturbance_pixels": 418, "disturbance_fraction_of_assessed": 0.000011, "facing_sensor_share": 0.497608,
    "edge_pixels_share": 0.23445, "assessed_edge_share": 0.060758}}


def stack_json(cfg: BatoConfig, h: int, w: int, transform: tuple[float, ...]) -> dict[str, Any]:
    t0 = "2026-06-29T12:21:38+00:00"
    inputs = [
        {"role": "PRE1", "product_name": "TEST-S1D-PRE1", "acquisition_start": t0, "orbit_direction": "ASCENDING",
         "relative_orbit": 85, "days_from_event": -57.5},
        {"role": "PRE2", "product_name": "TEST-S1D-PRE2", "acquisition_start": "2026-08-16T12:21:41+00:00",
         "orbit_direction": "ASCENDING", "relative_orbit": 85, "days_from_event": -9.5},
        {"role": "POST", "product_name": "TEST-S1D-POST", "acquisition_start": "2026-08-28T12:21:41+00:00",
         "orbit_direction": "ASCENDING", "relative_orbit": 85, "days_from_event": 2.5}]
    return {"kind": "s1_temporal_stack", "layout_version": "v1", "track": "ASCENDING-085", "event_date": "2026-08-26",
            "aoi_polygon": {"type": "Polygon", "coordinates": [[list(p) for p in AOI_RING]]}, "inputs": inputs,
            "grid": {"transform": list(transform), "width": w, "height": h, "pixel_size_m": PIXEL,
                     "analysis_bbox": ANALYSIS.model_dump(mode="json")},
            "output": {"stack": "data/interim/preprocessing/aoi-test/stack/ts-test/stack.tif",
                       "stack_sha256": "c" * 64}}


def baseline_json(state: np.ndarray, state_rel: str, state_sha: str, stack_sha: str) -> dict[str, Any]:
    counts = {int(k): int(v) for k, v in zip(*np.unique(state, return_counts=True), strict=True)}
    assessed = counts.get(0, 0) + counts.get(1, 0) + counts.get(2, 0)
    return {"kind": "sar_change_flood_baseline", "version": "bato-baseline-change-v1",
            "rule": "TEST: FLOOD = new open-water-like darkening against PRE-2 and PRE-1 (synthetic fixture)",
            "thresholds": [{"name": "baseline_flood.vv_water_db", "value": -15.0, "unit": "dB", "provisional": True},
                           {"name": "detection.min_component_pixels", "value": 10, "unit": "count",
                            "provisional": True}],
            "statistics": {"flood_pixels": counts.get(1, 0), "assessed_pixels": assessed,
                           "not_assessed_fraction_of_all": 1 - assessed / state.size},
            "outputs": {"state": state_rel, "state_sha256": state_sha}, "source": {"stack_sha256": stack_sha},
            "dem": None, "config_fingerprint": "b" * 64, "validation_note": "NOT VALIDATED (test fixture)",
            "inputs_not_used": ["population data"]}


def synthetic_rasters(state: np.ndarray) -> RasterSource:
    """Invented VV/VH backscatter (darker where Step 4C says FLOOD) and an invented lon/lat DEM."""
    h, w = state.shape
    rng = np.random.default_rng(5)
    yy, xx = np.indices((h, w))
    base = 0.05 * (1.2 + np.sin(xx / 37.0) * np.cos(yy / 53.0)) * rng.gamma(4, 0.25, (h, w))
    stack = np.stack([base, base * 0.25, base, base * 0.25, base, base * 0.25]).astype(np.float32)
    stack[4][state == 1] *= 0.08
    stack[5][state == 1] *= 0.12
    for k in range(6):
        stack[k][state == 2] = 0.004

    def window(arr: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, rows, cols = win
        return arr[r0:r0 + rows, c0:c0 + cols]

    def dem_crop(west: float, south: float, east: float, north: float):
        n, m = 300, 300
        lon = np.linspace(west, east, m)
        lat = np.linspace(north, south, n)
        z = 1200 + 900 * np.sin(lon[None, :] * 180) * np.cos(lat[:, None] * 140) + 300 * np.sin(lat[:, None] * 400)
        t = ((east - west) / m, 0.0, west, 0.0, -(north - south) / n, north)
        return z.astype(np.float32), t, None, (west, south, east, north)

    return RasterSource(state=lambda: state, state_window=lambda win: window(state, win),
                        stack_window=lambda band, win: window(stack[band - 1], win), dem_crop=dem_crop)


def published_world(tmp_path: Path, *, extra_v2: bool = False) -> SimpleNamespace:
    """A v1 connectivity run published under tmp_path (and optionally a later v2 run that supersedes it)."""
    cfg = project(tmp_path)
    fetch(cfg)
    extract = load_osm_extract(cfg, "aoi-test", EVENT)
    state, transform = state_raster()
    job = fake_job(cfg, extract, tmp_path, state)
    _transform, (h, w) = grid()
    rel = lambda p: p.resolve().relative_to(tmp_path.resolve()).as_posix()  # noqa: E731
    job.stack_provenance_path.write_text(json.dumps(stack_json(cfg, h, w, transform)), encoding="utf-8")
    job.baseline.source.provenance_sha256 = sha256_file(job.stack_provenance_path)
    job.baseline_provenance_path.write_text(json.dumps(baseline_json(
        state, rel(job.state_path), job.baseline.outputs.state_sha256, "c" * 64)), encoding="utf-8")
    job.baseline_provenance_sha256 = sha256_file(job.baseline_provenance_path)
    out, prov, result = run_connectivity(job, cfg, read_state=lambda _p: (state, transform), now=lambda: NOW)
    if extra_v2:
        run_connectivity(job, cfg, overwrite=True, read_state=lambda _p: (state, transform),
                         now=lambda: NOW.replace(hour=7))
        p = out / "connectivity_provenance.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["road_rule"] = "v2"
        p.write_text(json.dumps(data), encoding="utf-8")
    return SimpleNamespace(cfg=cfg, root=tmp_path, out=out, prov=prov, result=result, state=state,
                           transform=transform, job=job)


def inputs_for(world: SimpleNamespace, rasters: bool = True, control: bool = True) -> BuildInputs:
    inp = prepare_inputs(world.cfg, "aoi-test", "ts-test", title="Synthetic test world", with_rasters=False)
    if rasters:
        inp.rasters = synthetic_rasters(world.state)
    inp.debris_event = DEBRIS_EVENT
    inp.debris_control = DEBRIS_CONTROL if control else None
    return inp


"""EVALUATION ONLY. Read-only loading of the frozen official BATO result (road rule v1).

The run is selected and verified exactly as the dashboard does (:mod:`bato.dashboard.bundle`):

* the latest road-rule v1 connectivity run is used, and a v2 run is refused;
* every output must match its provenance SHA-256;
* the stack provenance, the Step 4C provenance and the Step 4C state raster it cites must match theirs too.

Every file read is hashed so the caller can prove, after the evaluation, that nothing changed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bato.config import BatoConfig
from bato.dashboard.bundle import (
    SEGMENTS_FILE,
    RunCandidate,
    find_connectivity_runs,
    select_official_run,
    sha256_file,
    verify_root_file,
    verify_run_outputs,
)
from bato.evaluation.geometry import Grid

StateReader = Callable[[Path], tuple[np.ndarray, tuple[float, ...], str]]


class OfficialResultError(ValueError):
    pass


@dataclass
class OfficialBato:
    run_dir: Path
    provenance: dict[str, Any]
    stack: dict[str, Any]
    baseline: dict[str, Any]
    grid: Grid
    aoi_ring: list[tuple[float, float]]
    post_time: datetime
    post_product: str
    event_date: date
    state: np.ndarray
    segments: dict[str, Any]
    files_read: dict[str, str] = field(default_factory=dict)      # repo-relative path -> SHA-256
    notes: list[str] = field(default_factory=list)


def rasterio_state_reader(path: Path) -> tuple[np.ndarray, tuple[float, ...], str]:
    import rasterio  # optional dependency (preprocessing extra)

    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise OfficialResultError(f"{path} has {ds.count} bands; the Step 4C state raster has one")
        t = ds.transform
        return ds.read(1), (t.a, t.b, t.c, t.d, t.e, t.f), ds.crs.to_string() if ds.crs else ""


def _utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise OfficialResultError(f"stack acquisition time {text!r} has no zone; expected UTC")
    return dt


def _official_run(cfg: BatoConfig, aoi_id: str, candidate_id: str,
                  connectivity_dir: Path | None) -> tuple[RunCandidate, list[str]]:
    run, notes = select_official_run(find_connectivity_runs(cfg, aoi_id, candidate_id), connectivity_dir)
    prov = run.provenance
    if prov.get("aoi_id") != aoi_id or prov.get("candidate_id") != candidate_id:
        raise OfficialResultError("the connectivity run is for a different AOI or candidate")
    return run, notes


def official_aoi(cfg: BatoConfig, aoi_id: str, candidate_id: str,
                 connectivity_dir: Path | None = None) -> dict[str, Any]:
    """The BATO AOI ring and POST time from the stack provenance cited by the official run (hash-verified).
    Used before any EMS product exists (to decide downloads), so it needs no raster library."""
    root = cfg.config_dir.parent
    run, _ = _official_run(cfg, aoi_id, candidate_id, connectivity_dir)
    ref = run.provenance["inputs"]["stack_provenance"]
    stack_p = verify_root_file(cfg, root, ref, "stack provenance")
    stack = json.loads(stack_p.read_text(encoding="utf-8"))
    post = [i for i in stack["inputs"] if i.get("role") == "POST"]
    return {"aoi_id": aoi_id, "candidate_id": candidate_id, "connectivity_run": run.directory.name,
            "stack_provenance": {"path": ref["path"], "sha256": ref["sha256"]},
            "aoi_ring": [[float(x), float(y)] for x, y in stack["aoi_polygon"]["coordinates"][0]],
            "post_acquisition_start": post[0]["acquisition_start"] if len(post) == 1 else None,
            "event_date": stack.get("event_date")}


def official_road_geometry(cfg: BatoConfig, aoi_id: str, candidate_id: str,
                           connectivity_dir: Path | None = None) -> dict[str, Any]:
    """GEOMETRY ONLY, for pre-freeze checks: the official v1 road segment lines (hash-verified) with every property
    except ``segment_id`` dropped, so no road state or result can influence protocol decisions; plus the stack grid."""
    root = cfg.config_dir.parent
    run, _ = _official_run(cfg, aoi_id, candidate_id, connectivity_dir)
    files = verify_run_outputs(run)
    stack_p = verify_root_file(cfg, root, run.provenance["inputs"]["stack_provenance"], "stack provenance")
    stack = json.loads(stack_p.read_text(encoding="utf-8"))
    fc = json.loads(files[SEGMENTS_FILE].read_text(encoding="utf-8"))
    lines = [[(float(x), float(y)) for x, y in f["geometry"]["coordinates"]] for f in fc["features"]]
    g = stack["grid"]
    return {"lines": lines, "grid": Grid(g["transform"], int(g["width"]), int(g["height"])),
            "aoi_ring": [(float(x), float(y)) for x, y in stack["aoi_polygon"]["coordinates"][0]],
            "segments_sha256": sha256_file(files[SEGMENTS_FILE]), "road_state_read": False}


def load_official(cfg: BatoConfig, aoi_id: str, candidate_id: str, *, read_state: StateReader = rasterio_state_reader,
                  connectivity_dir: Path | None = None) -> OfficialBato:
    root = cfg.config_dir.parent
    run, notes = _official_run(cfg, aoi_id, candidate_id, connectivity_dir)
    prov = run.provenance
    files = verify_run_outputs(run)
    files_read: dict[str, str] = {}

    def rel(p: Path) -> str:
        return p.resolve().relative_to(root.resolve()).as_posix()

    prov_path = run.directory / "connectivity_provenance.json"
    files_read[rel(prov_path)] = sha256_file(prov_path)
    inputs = prov["inputs"]
    stack_p = verify_root_file(cfg, root, inputs["stack_provenance"], "stack provenance")
    base_p = verify_root_file(cfg, root, inputs["baseline_provenance"], "Step 4C provenance")
    state_p = verify_root_file(cfg, root, inputs["baseline_state"], "Step 4C state raster")
    seg_p = files[SEGMENTS_FILE]
    for p in (stack_p, base_p, state_p, seg_p):
        files_read[rel(p)] = sha256_file(p)
    stack = json.loads(stack_p.read_text(encoding="utf-8"))
    baseline = json.loads(base_p.read_text(encoding="utf-8"))
    g = stack["grid"]
    grid = Grid(g["transform"], int(g["width"]), int(g["height"]))
    state, transform, crs = read_state(state_p)
    if tuple(state.shape) != grid.shape:
        raise OfficialResultError(f"state raster shape {state.shape} differs from the stack grid {grid.shape}")
    if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(transform, grid.transform, strict=True)):
        raise OfficialResultError(f"state raster transform {transform} differs from the stack grid {grid.transform}")
    if crs and "3857" not in crs:
        raise OfficialResultError(f"state raster CRS {crs} is not EPSG:3857")
    if not crs:
        notes.append("state raster carries no CRS; the stack grid (EPSG:3857) is used")
    ring = [(float(x), float(y)) for x, y in stack["aoi_polygon"]["coordinates"][0]]
    post = [i for i in stack["inputs"] if i.get("role") == "POST"]
    if len(post) != 1:
        raise OfficialResultError("the stack provenance must list exactly one POST input")
    segments = json.loads(seg_p.read_text(encoding="utf-8"))
    return OfficialBato(run_dir=run.directory, provenance=prov, stack=stack, baseline=baseline, grid=grid,
                        aoi_ring=ring, post_time=_utc(post[0]["acquisition_start"]),
                        post_product=str(post[0].get("product_name")),
                        event_date=date.fromisoformat(str(stack["event_date"])[:10]), state=np.asarray(state),
                        segments=segments, files_read=files_read, notes=notes)


def rehash(root: Path, files_read: dict[str, str]) -> list[str]:
    """Files whose SHA-256 changed since they were read (empty = untouched)."""
    return [p for p, h in files_read.items() if not (root / p).is_file() or sha256_file(root / p) != h]

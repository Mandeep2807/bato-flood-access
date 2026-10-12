"""TEST FIXTURES for Step 3.5: three PUBLISHED Step 3.4 outputs (PRE-1, PRE-2, POST of the real
Trishuli ASCENDING/85 candidate) made by the real Step 3.2/3.3/3.4 code with fake gpts, on one
10 m standard grid but with different extents. No SNAP, no real imagery."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from bato.config import load_config
from bato.preprocessing import radiometric as rad
from bato.preprocessing import snap
from bato.preprocessing import speckle as spk
from bato.preprocessing import terrain as ter
from bato.preprocessing.terrain_validation import dimap_map_geocoding, lonlat_to_3857

from helpers import REPO_CONFIG_DIR
from snap_fakes import TRISHULI_POST, TRISHULI_PRE1, FakeSnap, make_project
from speckle_fakes import FakeSpeckleSnap, read_envi, write_band
from terrain_fakes import TINY_AOI, FakeTerrainSnap, TerrainOutputSpec, add_dem_tiles

TRISHULI_PRE2 = "S1D-A085-20260816T122141"
PASSES = (TRISHULI_PRE1, TRISHULI_PRE2, TRISHULI_POST)
# different extents on the same 10 m lattice (the fake snaps every extent to the standard grid)
SHIFTS = {TRISHULI_PRE1: 0.0, TRISHULI_PRE2: 0.0004, TRISHULI_POST: -0.0003}
MARGIN_LON = TINY_AOI[0][0] - 0.0005          # inside the stack window (AOI + 0.1 km), outside the AOI
MARGIN_LAT = (TINY_AOI[0][1] + TINY_AOI[2][1]) / 2


def config_with(root: Path, **aoi: float) -> Path:
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    path = root / "config" / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["aoi"].update(aoi)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root / "config"


def set_pixel(value: float, band: str = "Sigma0_VV") -> Callable[[Path, Path, Path], None]:
    """FakeTerrainSnap.after_run hook: put ``value`` at a pixel inside the stack window, outside the AOI."""
    def hook(out: Path, _src: Path, _dem: Path) -> None:
        _wkt, i2m = dimap_map_geocoding(out)
        assert i2m is not None
        x, y = lonlat_to_3857(np.array([MARGIN_LON]), np.array([MARGIN_LAT]))
        col, row = int((x[0] - i2m[4]) // i2m[0]), int((i2m[5] - y[0]) // -i2m[3])
        img = out.with_suffix(".data") / f"{band}.img"
        data = read_envi(img)
        data[row, col] = value
        write_band(img, data)
    return hook


def published_terrain_set(root: Path, gpt: snap.ResolvedGpt, *, analysis_buffer_km: float = 0.1,
                          skip_terrain: tuple[str, ...] = (),
                          after_run: dict[str, Callable[[Path, Path, Path], None]] | None = None):
    """Project whose three passes went through the real Step 3.2-3.4 code (fake gpts)."""
    cfg = load_config(config_with(root, analysis_buffer_km=analysis_buffer_km))
    project = make_project(cfg, present_passes=PASSES)
    for pid in PASSES:
        fake = FakeSnap()
        result = rad.run_radiometric(rad.prepare_job(project.manifest, pid, cfg), cfg, gpt=gpt, runner=fake.run,
                                     popen=fake.popen)
        data = json.loads(result.provenance_path.read_text(encoding="utf-8"))   # TEST FIXTURE: ~1 km AOI
        data["aoi"]["aoi_polygon"]["coordinates"] = [[list(p) for p in TINY_AOI]]
        result.provenance_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        fake_spk = FakeSpeckleSnap()
        spk.run_speckle(spk.prepare_speckle_job(project.manifest, pid, cfg), cfg, gpt=gpt, runner=fake_spk.run,
                        popen=fake_spk.popen)
    add_dem_tiles(project)
    for pid in PASSES:
        if pid in skip_terrain:
            continue
        fake_tc = FakeTerrainSnap(out=TerrainOutputSpec(extent_shift_deg=SHIFTS[pid]),
                                  after_run=(after_run or {}).get(pid))
        ter.run_terrain(ter.prepare_terrain_job(project.manifest, pid, cfg), cfg, gpt=gpt, runner=fake_tc.run,
                        popen=fake_tc.popen)
    return project


def stage_root(project) -> Path:
    return project.cfg.settings.paths.interim / "preprocessing" / project.manifest.aoi_id


def tree_hashes(folder: Path) -> dict[str, str]:
    import hashlib
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob("*")) if f.is_file()}


def terrain_band(project, pass_id: str, pol: str) -> tuple[np.ndarray, tuple[float, ...]]:
    folder = stage_root(project) / pass_id / "terrain"
    dim = next(folder.glob("*.dim"))
    _wkt, i2m = dimap_map_geocoding(dim)
    assert i2m is not None
    return read_envi(dim.with_suffix(".data") / f"Sigma0_{pol}.img"), i2m


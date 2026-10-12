"""terrain_correct_s1.py paths that need no SNAP (synthetic Step 3.3 output and DEM tiles)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bato.config import load_config
from bato.preprocessing import snap

from snap_fakes import TRISHULI_PRE1, make_project
from terrain_fakes import published_speckle

REPO = Path(__file__).resolve().parents[2]


def run(project, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(REPO / "scripts" / "terrain_correct_s1.py"),
                           "--config-dir", str(project.root / "config"), "--manifest", str(project.manifest_path),
                           *args], capture_output=True, text=True, env=env, cwd=project.root)


def test_missing_step_3_3_output_is_reported(tmp_path_factory):
    root = tmp_path_factory.mktemp("c")
    shutil.copytree(REPO / "config", root / "config")
    project = make_project(load_config(root / "config"))
    r = run(project, "--pass-id", TRISHULI_PRE1)
    assert r.returncode == 1 and "no published Step 3.3 output" in r.stderr
    r = run(project, "--pass-id", "S1A-D019-20260101T000000")
    assert r.returncode == 1 and "not in the manifest" in r.stderr


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
def test_missing_snap_fails_preflight_and_writes_nothing(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    project = published_speckle(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))
    r = run(project, "--pass-id", TRISHULI_PRE1, env_extra={"BATO_SNAP_GPT": str(project.root / "no" / "gpt.exe")})
    assert r.returncode == 2 and "PREFLIGHT FAILED (nothing written)" in r.stderr and "BATO_SNAP_GPT" in r.stderr
    for text in ("Stage      : terrain", "EPSG:3857 10 m", "N27E084", "re-hashed OK", "s1_terrain.xml",
                 f"{TRISHULI_PRE1}/terrain/"):
        assert text in r.stdout
    stage = project.cfg.settings.paths.interim / "preprocessing" / project.manifest.aoi_id / TRISHULI_PRE1
    assert sorted(p.name for p in stage.iterdir()) == ["radiometric", "speckle"]

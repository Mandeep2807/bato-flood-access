"""run_baseline_detection.py end to end on a synthetic Step 3.5 stack (TEST FIXTURE)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bato.preprocessing import snap

from inference_fakes import synthetic_project

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")


def run(project, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_baseline_detection.py"), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


def test_baseline_cli(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    project, _data, _mask = synthetic_project(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))
    r = run(project)
    assert r.returncode == 0, r.stdout + r.stderr
    for text in ("deterministic, NOT VALIDATED", "## States", "NOT_ASSESSED_INVALID_SAR",
                 "baseline_flood.vv_water_db | -15 | dB | physics_literature", "no event output (no comparison)",
                 "pre-event OSM waterway overlap: NOT AVAILABLE", "RESULT     : SUCCEEDED"):
        assert text in r.stdout
    out = project.cfg.settings.paths.interim / "detection" / project.manifest.aoi_id
    assert len(list(out.rglob("baseline_state.tif"))) == 1
    r = run(project)
    assert r.returncode == 1 and "already exists" in r.stderr
    r = run(project, "--overwrite")
    assert r.returncode == 0

"""diagnose_floodvit.py end to end on FloodViT outputs made with a FAKE model (TEST FIXTURE, no torch)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bato.preprocessing import snap

from inference_fakes import change_model, run_mode, synthetic_project

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")


def run(project, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "diagnose_floodvit.py"), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


def test_diagnose_cli(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    project, _data, _mask = synthetic_project(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))
    r = run(project)
    assert r.returncode == 1 and "no FloodViT output" in r.stderr
    for mode in ("event", "zero_change", "static_post"):
        run_mode(project, change_model, mode)
    r = run(project)
    assert r.returncode == 0, r.stdout + r.stderr
    for text in ("Modes      : event, zero_change, static_post", "NOT validation", "flat terrain",
                 "Agreement with the event flood", "RESULT     : DIAGNOSTICS ONLY (no validation)"):
        assert text in r.stdout
    r = run(project)
    assert r.returncode == 1 and "already exists" in r.stderr

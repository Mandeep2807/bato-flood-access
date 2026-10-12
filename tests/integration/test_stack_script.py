"""stack_s1.py end to end on Step 3.4 outputs made by the real Step 3.2-3.4 code with fake gpts."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bato.config import load_config
from bato.preprocessing import snap

from snap_fakes import make_project
from stack_fakes import config_with, published_terrain_set

REPO = Path(__file__).resolve().parents[2]


def run(project, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "stack_s1.py"), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
def test_missing_pass_outputs_are_reported(tmp_path_factory):
    project = make_project(load_config(config_with(tmp_path_factory.mktemp("c"))))
    r = run(project)
    assert r.returncode == 1 and "no published Step 3.4 output" in r.stderr and "--pass-id" in r.stderr


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
def test_stack_cli_builds_and_publishes(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    project = published_terrain_set(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))
    r = run(project)
    assert r.returncode == 0, r.stdout + r.stderr
    for text in ("PRE1 S1D-A085-20260629T122138 (-58 d)", "PRE2 S1D-A085-20260816T122141 (-10 d)",
                 "POST S1D-A085-20260828T122141 (+2 d)", "PASS  copied_unchanged_POST_VH", "PASS  aoi_coverage",
                 "RESULT     : SUCCEEDED"):
        assert text in r.stdout
    r = run(project)
    assert r.returncode == 1 and "already exists" in r.stderr

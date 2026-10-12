"""prepare_model_input.py end to end on a Step 3.5 stack made by the real Step 3.2-3.5 code with fake gpts."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bato.contracts.io import load_contract
from bato.contracts.model_input import ModelInputManifest
from bato.preprocessing import snap
from bato.preprocessing import stack as stk

from stack_fakes import published_terrain_set, stage_root

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")


def run(project, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_model_input.py"), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


@pytest.fixture
def project(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    return published_terrain_set(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))


def test_missing_stack_is_reported(project):
    r = run(project)
    assert r.returncode == 1 and "no published Step 3.5 stack" in r.stderr and "stack_s1.py" in r.stderr


def test_cli_publishes_the_manifest(project):
    stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    r = run(project)
    assert r.returncode == 0, r.stdout + r.stderr
    for text in ("PRE1 S1D-A085-20260629T122138 (-57.5 d)", "PRE2 S1D-A085-20260816T122141 (-9.5 d)",
                 "POST S1D-A085-20260828T122141 (+2.5 d)", "(hash verified)",
                 "0 POST_VV=flood_vv, 1 POST_VH=flood_vh, 2 PRE2_VV=sec1_vv, 3 PRE2_VH=sec1_vh, 4 PRE1_VV=sec2_vv, "
                 "5 PRE1_VH=sec2_vh", "clamp [0.0, 0.15], NaN -> 0.15", "1 x 224 px, stride 224 (overlap 0)",
                 "OPEN  radar_visibility [bato_gap]", "OPEN  released_pixel_units [not_determinable]",
                 "5 item(s) unresolved", "RESULT     : SUCCEEDED"):
        assert text in r.stdout
    cid = stk.select_candidate(project.manifest, None).candidate_id
    path = stage_root(project) / "model_input" / cid / "model_input_manifest.json"
    manifest = load_contract(ModelInputManifest, path)
    assert manifest.spec.tiling.stride == 224 and len(manifest.tiles) == 1
    r = run(project)
    assert r.returncode == 1 and "already exists" in r.stderr
    r = run(project, "--overwrite", "--stride", "112")
    assert r.returncode == 0 and "stride 112 (overlap 112)" in r.stdout
    r = run(project, "--overwrite", "--stride", "0")
    assert r.returncode == 1 and "stride" in r.stderr

"""run_floodvit_inference.py refusals that need no torch and no real checkpoint (TEST FIXTURE project)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bato.preprocessing import snap
from bato.preprocessing import stack as stk

from stack_fakes import published_terrain_set

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")


def run(project, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_floodvit_inference.py"), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


@pytest.fixture
def project(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    return published_terrain_set(tmp_path_factory.mktemp("c"), snap.ResolvedGpt(gpt, "config"))


def test_missing_stack_or_checkpoint_is_refused_before_loading_torch(project, tmp_path):
    ckpt = tmp_path / "TEST_FIXTURE.pt"
    ckpt.write_bytes(b"TEST FIXTURE")
    r = run(project, "--checkpoint", str(ckpt), "--kuro-repo", str(tmp_path))
    assert r.returncode == 1 and "no published Step 3.5 stack" in r.stderr
    stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    r = run(project, "--checkpoint", str(tmp_path / "missing.pt"), "--kuro-repo", str(tmp_path))
    assert r.returncode == 1 and "checkpoint not found" in r.stderr
    r = run(project, "--checkpoint", str(ckpt), "--kuro-repo", str(tmp_path), "--stride", "0")
    assert r.returncode == 1 and "stride" in r.stderr


def test_unloadable_model_publishes_nothing(project, tmp_path):
    stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    ckpt = tmp_path / "TEST_FIXTURE.pt"
    ckpt.write_bytes(b"TEST FIXTURE")
    r = run(project, "--checkpoint", str(ckpt), "--kuro-repo", str(tmp_path / "not_kuro"), "--mode", "both")
    assert r.returncode == 2 and "MODEL UNAVAILABLE" in r.stderr          # no torch, or no Kuro Siwo checkout
    assert not (project.cfg.settings.paths.interim / "inference").exists()

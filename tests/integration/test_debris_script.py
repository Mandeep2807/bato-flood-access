"""run_debris_detection.py end to end on a synthetic Step 3.5 stack (TEST FIXTURE - invented backscatter)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bato.preprocessing import snap
from bato.preprocessing import stack as stk

from inference_fakes import H, W, scene, write_synthetic_stack
from stack_fakes import published_terrain_set

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio not installed")
DEBRIS_BOX = (slice(150, 170), slice(150, 170))          # POST +6 dB in VV and VH


def run(project, script: str, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), "--config-dir",
                           str(project.root / "config"), "--manifest", str(project.manifest_path), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


def noisy_project(root, gpt):
    project = published_terrain_set(root, gpt)
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    data, mask = scene()
    rng = np.random.default_rng(3)
    data = (data * 10.0 ** (rng.normal(0.0, 0.03, data.shape))).astype(np.float32)   # ~0.3 dB per image
    data[4:6, DEBRIS_BOX[0], DEBRIS_BOX[1]] *= np.float32(10.0 ** 0.6)
    data[:, mask == 0] = 0.0
    write_synthetic_stack(project, result, data, mask)
    return project


def test_debris_cli(tmp_path_factory, tmp_path):
    gpt = tmp_path / "gpt.exe"
    gpt.write_text("TEST FIXTURE")
    project = noisy_project(tmp_path_factory.mktemp("d"), snap.ResolvedGpt(gpt, "config"))
    r = run(project, "run_debris_detection.py")
    assert r.returncode == 1 and "no Step 4C baseline" in r.stderr              # Step 4C first
    assert run(project, "run_baseline_detection.py").returncode == 0
    r = run(project, "run_debris_detection.py")
    assert r.returncode == 0, r.stdout + r.stderr
    for text in ("deterministic, NOT VALIDATED", "sigma VV", "Pre-registered checks", "RESULT     : SUCCEEDED"):
        assert text in r.stdout
    out = project.cfg.settings.paths.interim / "detection" / project.manifest.aoi_id
    prov_path = next(out.rglob("debris_provenance.json"))
    names = sorted(p.name for p in prov_path.parent.iterdir())
    assert names == ["change_table.npz", "debris_diagnostics.md", "debris_provenance.json", "debris_state.tif"]
    prov = json.loads(prov_path.read_text())
    assert prov["validated"] is False and prov["z"] == 3.0
    assert prov["statistics"]["state_counts"]["DISTURBANCE"] > 0
    assert [n["polarization"] for n in prov["noise"]] == ["VV", "VH"] and all(n["sigma_db"] > 0 for n in prov["noise"])
    assert H * W == prov["statistics"]["total_pixels"]
    r = run(project, "run_debris_detection.py")
    assert r.returncode == 1 and "already exists" in r.stderr
    assert run(project, "run_debris_detection.py", "--overwrite").returncode == 0
    assert list(prov_path.parent.parent.glob("debris.superseded-*"))

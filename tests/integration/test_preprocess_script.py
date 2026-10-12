"""preprocess_s1.py paths that need no SNAP (synthetic SAFE zip, real catalogue metadata)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bato.config import load_config
from bato.contracts.io import save_contract

from snap_fakes import TRISHULI_POST, TRISHULI_PRE1, make_project

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    shutil.copytree(REPO / "config", tmp_path / "config")
    return make_project(load_config(tmp_path / "config"))


def run(project, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BATO_SNAP_GPT", "BATO_CONFIG_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(REPO / "scripts" / "preprocess_s1.py"),
                           "--config-dir", str(project.root / "config"), *args],
                          capture_output=True, text=True, env=env, cwd=project.root)


def test_missing_snap_fails_preflight_and_writes_nothing(project):
    r = run(project, "--manifest", str(project.manifest_path), "--pass-id", TRISHULI_PRE1, "--stage", "radiometric",
            env_extra={"BATO_SNAP_GPT": str(project.root / "no-snap" / "gpt.exe")})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "PREFLIGHT FAILED (nothing written)" in r.stderr
    assert ("BATO_SNAP_GPT is set to" in r.stderr) or ("rasterio is required" in r.stderr)
    for text in ("Product    : S1D_IW_GRDH_1SDV_20260629T122138", "AOI        : aoi-e893707c2beb",
                 "Stage      : radiometric", "s1_radiometric.xml",
                 f"data/interim/preprocessing/aoi-e893707c2beb/{TRISHULI_PRE1}/radiometric/"):
        assert text in r.stdout
    assert not (project.cfg.settings.paths.interim / "preprocessing").exists()


def test_bad_inputs_are_rejected(project):
    r = run(project, "--manifest", str(project.manifest_path), "--pass-id", TRISHULI_POST, "--stage", "radiometric")
    assert r.returncode == 1 and "incomplete" in r.stderr
    dry = save_contract(project.manifest.model_copy(update={"dry_run": True}), project.root / "plan.json")
    r = run(project, "--manifest", str(dry), "--pass-id", TRISHULI_PRE1, "--stage", "radiometric")
    assert r.returncode == 1 and "dry-run" in r.stderr
    r = run(project, "--manifest", str(project.manifest_path), "--pass-id", TRISHULI_PRE1, "--stage", "terrain")
    assert r.returncode == 2 and "invalid choice" in r.stderr                 # argparse: only 'radiometric'

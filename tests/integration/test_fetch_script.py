"""fetch_s1_grd.py paths that need no network or credentials."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "tests" / "fixtures" / "real_catalogues"


def run(tmp_path: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    # Work in a copy of the config so manifests land in tmp_path, not the repository.
    shutil.copytree(REPO / "config", tmp_path / "config", dirs_exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CDSE_")}
    env.update({"PYTHONPATH": str(REPO / "src"), **(env_extra or {})})
    return subprocess.run([sys.executable, str(REPO / "scripts" / "fetch_s1_grd.py"),
                           "--config-dir", str(tmp_path / "config"), *args],
                          capture_output=True, text=True, env=env, cwd=tmp_path)


def test_offline_dry_run_plans_every_slice(tmp_path):
    r = run(tmp_path, "--selection", str(REAL / "valencia_sets.json"), "--catalogue", str(REAL / "valencia_s1.json"),
            "--track", "DESCENDING/110", "--offline")
    assert r.returncode == 0, r.stderr
    assert "1 candidate(s), 3 pass(es), 6 Sentinel-1 product(s)" in r.stdout
    assert r.stdout.count(" 1/2 ") == 3 and r.stdout.count(" 2/2 ") == 3
    assert "DRY RUN - nothing downloaded" in r.stdout and "N39W001" in r.stdout
    assert (tmp_path / "data" / "raw" / "manifests" / "aoi-0027295ba364_fetch_plan.json").is_file()
    assert not (tmp_path / "data" / "raw" / "sentinel1").exists()


def test_large_download_needs_explicit_cap(tmp_path):
    r = run(tmp_path, "--selection", str(REAL / "valencia_sets.json"), "--catalogue", str(REAL / "valencia_s1.json"),
            "--download", "--no-dem")
    assert r.returncode == 1 and "exceed download.max_scenes_per_run" in r.stderr


def test_download_without_credentials_fails_before_any_request(tmp_path):
    r = run(tmp_path, "--selection", str(REAL / "trishuli_sets.json"), "--catalogue", str(REAL / "trishuli_s1.json"),
            "--track", "ASCENDING/85", "--max-scenes", "1", "--download", "--no-dem")
    assert r.returncode == 2 and "credentials missing" in r.stderr


def test_mismatched_inputs_are_rejected(tmp_path):
    r = run(tmp_path, "--selection", str(REAL / "trishuli_sets.json"), "--catalogue", str(REAL / "valencia_s1.json"),
            "--offline")
    assert r.returncode == 1 and "digest mismatch" in r.stderr

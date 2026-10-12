"""fetch_osm_snapshot.py / run_connectivity.py refuse cleanly without their inputs (nothing written)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from connectivity_fakes import project

REPO = Path(__file__).resolve().parents[2]


def run(script: str, root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), "--config-dir", str(root / "config"),
                           *args], capture_output=True, text=True, env=env, cwd=root)


def test_fetch_osm_requires_a_published_stack(tmp_path):
    project(tmp_path)
    r = run("fetch_osm_snapshot.py", tmp_path, "--manifest", str(tmp_path / "missing.json"), "--dry-run")
    assert r.returncode == 1 and "nothing written" in r.stderr
    assert not (tmp_path / "data" / "raw" / "osm").exists()


def test_run_connectivity_refuses_without_inputs(tmp_path):
    project(tmp_path)
    r = run("run_connectivity.py", tmp_path, "--manifest", str(tmp_path / "missing.json"))
    if importlib.util.find_spec("rasterio") is None:
        assert r.returncode == 2 and "PREFLIGHT FAILED" in r.stderr
    else:
        assert r.returncode == 1 and "nothing written" in r.stderr
    assert not (tmp_path / "data" / "processed").exists()

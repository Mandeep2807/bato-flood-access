"""build_dashboard.py / serve_dashboard.py refuse cleanly without their inputs (nothing written)."""

from __future__ import annotations

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
                           *args], capture_output=True, text=True, env=env, cwd=root, timeout=60)


def test_build_dashboard_refuses_without_inputs(tmp_path):
    project(tmp_path)
    r = run("build_dashboard.py", tmp_path, "--manifest", str(tmp_path / "missing.json"))
    assert r.returncode == 1 and "nothing written" in r.stderr
    assert not (tmp_path / "data" / "processed" / "dashboard").exists()


def test_serve_dashboard_needs_a_bundle(tmp_path):
    project(tmp_path)
    r = run("serve_dashboard.py", tmp_path)
    assert r.returncode == 1 and "build_dashboard.py" in r.stderr

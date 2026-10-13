"""scripts/build_sitrep.py: refuses cleanly without inputs (nothing written) and, on the synthetic v1 run,
writes a byte-identical SITREP for a fixed --issued-at."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bato.dashboard.sitrep import load_facts, render

from connectivity_fakes import project
from sitrep_fakes import published_sitrep_world

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_sitrep.py"
ARGS = ["--title", "Synthetic", "--incident", "test fixture", "--prepared-by", "Test team",
        "--issued-at", "2026-10-13T07:00:00Z"]


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, str(SCRIPT), "--config-dir", str(root / "config"), *args],
                          capture_output=True, text=True, env=env, cwd=root, timeout=60)


def test_refuses_without_inputs(tmp_path):
    project(tmp_path)
    r = run(tmp_path, "--manifest", str(tmp_path / "missing.json"), *ARGS)
    assert r.returncode == 1 and "nothing written" in r.stderr
    assert not (tmp_path / "reports").exists()


def test_refuses_issue_time_without_zone(tmp_path):
    project(tmp_path)
    r = run(tmp_path, "--manifest", str(tmp_path / "m.json"), "--title", "x", "--incident", "y",
            "--issued-at", "2026-10-13T07:00:00")
    assert r.returncode == 1 and "time zone" in r.stderr


def _script_module():
    spec = importlib.util.spec_from_file_location("build_sitrep_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_writes_the_sitrep_from_the_official_run(tmp_path, monkeypatch, capsys):
    w = published_sitrep_world(tmp_path)
    manifest = w.cfg.settings.paths.raw / "manifests" / "aoi-test_fetch_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    mod = _script_module()
    # The synthetic world has no Step 2C manifest; stand in for its two identifiers only.
    monkeypatch.setattr(mod, "load_contract", lambda _cls, _p: SimpleNamespace(aoi_id="aoi-test"))
    monkeypatch.setattr(mod, "select_candidate", lambda _m, _c: SimpleNamespace(candidate_id="ts-test"))
    out = tmp_path / "reports" / "BATO_SITREP.md"
    argv = ["build_sitrep.py", "--config-dir", str(tmp_path / "config"), "--manifest", str(manifest), *ARGS]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    first = out.read_bytes()
    printed = capsys.readouterr().out
    assert "road rule v1" in printed and "blocked 13 segments / 3 stretches / 0.635 km" in printed
    facts, _ = load_facts(w.cfg, "aoi-test", "ts-test")
    expected = render(facts, title="Synthetic", incident="test fixture", prepared_by="Test team",
                      issued_at=datetime(2026, 10, 13, 7, 0, tzinfo=timezone.utc))
    assert first.decode("utf-8") == expected
    assert mod.main() == 0 and out.read_bytes() == first                      # reproducible

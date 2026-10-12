"""Integration of the foundation pieces only: config -> request -> AOI ->
file -> reload, plus the config check script. No satellite processing is
exercised here because none exists yet."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from bato.contracts.aoi import AOI, AOIRequest, build_aoi
from bato.contracts.io import load_contract, save_contract
from bato.logging_utils import stage

from helpers import FIXED_TODAY

REPO = Path(__file__).resolve().parents[2]
# The script runs in a subprocess, which does not inherit pytest's pythonpath.
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def test_request_json_to_aoi_file_and_back(cfg, tmp_path):
    request_json = json.dumps({
        "event_date": "2026-08-26",
        "bbox": {"min_lon": 85.25, "min_lat": 28.05, "max_lon": 85.40, "max_lat": 28.20},
        "label": "judge-chosen test area",
    })
    with stage("aoi_validation") as report:
        aoi = build_aoi(AOIRequest.model_validate_json(request_json), cfg.settings, today=FIXED_TODAY)
        report.record_output(aoi_id=aoi.aoi_id, area_km2=aoi.area_km2)

    path = save_contract(aoi, tmp_path / aoi.aoi_id / "aoi.json")
    assert load_contract(AOI, path) == aoi
    assert report.status == "succeeded"


def test_check_config_script_runs():
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "check_config.py")],
                            capture_output=True, text=True, check=False, cwd=REPO, env=ENV)
    assert result.returncode == 0, result.stderr
    assert "fingerprint" in result.stdout and "provisional thresholds (23)" in result.stdout


def test_check_config_script_reports_invalid_config(tmp_path):
    (tmp_path / "config.yaml").write_text("paths: {}\n", encoding="utf-8")
    (tmp_path / "thresholds.yaml").write_text("{}\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "check_config.py"),
                             "--config-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False, cwd=REPO, env=ENV)
    assert result.returncode == 1 and "CONFIG INVALID" in result.stderr

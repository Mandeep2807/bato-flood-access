"""select_temporal_sets.py on a catalogue JSON built offline from the TEST FIXTURE."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from bato.acquisition.cdse_odata import CDSEODataClient
from bato.acquisition.http import HttpResponse
from bato.acquisition.s1_catalogue import search_sentinel1
from bato.contracts.aoi import AOIRequest, build_aoi
from bato.contracts.common import BBox
from bato.contracts.io import load_contract, save_contract
from bato.contracts.temporal import TemporalSelectionResult

from helpers import FIXED_TODAY

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
FIXTURE = REPO / "tests" / "fixtures" / "cdse_s1_catalogue_TEST_FIXTURE.json"


class _Fixture:
    def get(self, url, timeout_s):
        return HttpResponse(200, FIXTURE.read_bytes())

    def post_form(self, url, form, timeout_s):  # pragma: no cover
        raise AssertionError


def test_cli_ranks_sets_per_track_and_saves(cfg, tmp_path):
    aoi = build_aoi(AOIRequest(event_date=date(2026, 8, 26),
                               bbox=BBox(min_lon=85.20, min_lat=28.00, max_lon=85.40, max_lat=28.20)),
                    cfg.settings, today=FIXED_TODAY)
    catalogue = search_sentinel1(aoi, cfg, client=CDSEODataClient(cfg.settings.cdse, _Fixture()),
                                 retrieved_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    cat_path = save_contract(catalogue, tmp_path / "catalogue.json")
    out_path = tmp_path / "selection.json"
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "select_temporal_sets.py"),
                             "--catalogue", str(cat_path), "--save", str(out_path)],
                            capture_output=True, text=True, env=ENV, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert "DESCENDING/19  candidates" in result.stdout and "DESCENDING/121 insufficient_pre" in result.stdout
    assert "NOT scientifically validated" in result.stdout
    saved = load_contract(TemporalSelectionResult, out_path)
    assert {t.track.label for t in saved.tracks} == {"ASCENDING/85", "DESCENDING/19", "DESCENDING/121"}


def test_cli_missing_catalogue_file(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "select_temporal_sets.py"),
                             "--catalogue", str(tmp_path / "missing.json")],
                            capture_output=True, text=True, env=ENV, cwd=REPO)
    assert result.returncode == 1 and "not found" in result.stderr

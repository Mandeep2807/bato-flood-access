"""The live-query script, exercised WITHOUT network: preset validity, table
rendering from a TEST FIXTURE result, and input errors. A real CDSE query is
not part of the test suite (see docs/catalogue_discovery.md, live checks)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

from bato.acquisition.cdse_odata import CDSEODataClient
from bato.acquisition.http import HttpResponse
from bato.acquisition.s1_catalogue import search_sentinel1
from bato.contracts.aoi import AOIRequest, build_aoi
from bato.contracts.common import BBox

from helpers import FIXED_TODAY

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "s1_catalogue_query.py"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
FIXTURE = REPO / "tests" / "fixtures" / "cdse_s1_catalogue_TEST_FIXTURE.json"


def load_script():
    spec = importlib.util.spec_from_file_location("s1_catalogue_query", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_presets_are_valid_aois_inside_and_outside_nepal(cfg):
    presets = yaml.safe_load((REPO / "config" / "test_aois.yaml").read_text(encoding="utf-8"))
    assert {"trishuli_2026", "valencia_2024"} <= set(presets)
    centres = {}
    for name, p in presets.items():
        lo1, la1, lo2, la2 = p["bbox"]
        aoi = build_aoi(AOIRequest(event_date=p["event_date"],
                                   bbox=BBox(min_lon=lo1, min_lat=la1, max_lon=lo2, max_lat=la2)),
                        cfg.settings, today=FIXED_TODAY)
        centres[name] = ((lo1 + lo2) / 2, (la1 + la2) / 2)
        assert aoi.area_km2 <= cfg.settings.aoi.max_area_km2
    assert presets["trishuli_2026"]["event_date"] == date(2026, 8, 26)   # from the official Track B PDF
    lon, lat = centres["valencia_2024"]
    assert not (80 <= lon <= 89 and 26 <= lat <= 31)                     # genuinely outside Nepal


class _OnePage:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def get(self, url, timeout_s):
        return HttpResponse(200, self.body)

    def post_form(self, url, form, timeout_s):  # pragma: no cover
        raise AssertionError


def test_table_and_grouping_render_from_fixture_result(cfg, capsys):
    aoi = build_aoi(AOIRequest(event_date=date(2026, 8, 26),
                               bbox=BBox(min_lon=85.20, min_lat=28.00, max_lon=85.40, max_lat=28.20)),
                    cfg.settings, today=FIXED_TODAY)
    client = CDSEODataClient(cfg.settings.cdse, _OnePage(FIXTURE.read_bytes()))
    result = search_sentinel1(aoi, cfg, client=client)
    load_script().print_result(result)
    out = capsys.readouterr().out
    assert "returned 21 | accepted 9 | rejected 12" in out
    assert "descending rel  19: 3 pre, 2 post" in out and "pre+post available" in out
    assert "descending rel 121: 0 pre, 2 post" in out and "NOT pairable" in out
    assert "rejected_coverage: pass covers 50" in out
    assert json.loads(FIXTURE.read_text())["_TEST_FIXTURE"].startswith("TEST FIXTURE")


def test_script_rejects_oversized_aoi_before_any_network_call():
    result = subprocess.run([sys.executable, str(SCRIPT), "--bbox", "84", "27", "86", "29",
                             "--event-date", "2026-08-26"], capture_output=True, text=True, env=ENV, cwd=REPO)
    assert result.returncode == 1 and "exceeds the maximum" in result.stderr


def test_script_rejects_unknown_preset():
    result = subprocess.run([sys.executable, str(SCRIPT), "--preset", "atlantis"],
                            capture_output=True, text=True, env=ENV, cwd=REPO)
    assert result.returncode == 1 and "unknown preset" in result.stderr

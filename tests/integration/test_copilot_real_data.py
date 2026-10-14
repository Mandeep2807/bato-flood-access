"""OPT-IN checks of the SITREP copilot on the REAL official result.

Evidence only (no model; read-only):
    $env:BATO_REAL_DATA_MANIFEST="data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"
    python -m pytest tests/integration/test_copilot_real_data.py -v -s

Live model smoke test (calls the configured provider; costs a few requests):
    $env:BATO_LLM_LIVE_TEST="1"   (plus BATO_LLM_PROVIDER, BATO_LLM_MODEL, BATO_LLM_API_KEY)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.copilot.evidence import build_pack
from bato.copilot.service import CopilotService
from bato.dashboard.sitrep import EMSR927_STATEMENT
from bato.preprocessing import stack as stk

MANIFEST = os.environ.get("BATO_REAL_DATA_MANIFEST")
pytestmark = pytest.mark.skipif(not MANIFEST, reason="real-data test is opt-in: set BATO_REAL_DATA_MANIFEST")


def _ids(cfg):
    root = cfg.config_dir.parent
    path = Path(MANIFEST) if Path(MANIFEST).is_absolute() else root / MANIFEST
    fetch = load_contract(FetchManifest, path)
    return fetch.aoi_id, stk.select_candidate(fetch, None).candidate_id


def test_real_evidence_pack_carries_the_official_values(cfg):
    aoi, cand = _ids(cfg)
    pack = build_pack(cfg, aoi, cand)
    text = {it["id"]: it["text"] for it in pack.core}
    assert pack.facts.road_rule == "v1" and (aoi, cand) == ("aoi-e893707c2beb", "ts-80841ae62417")
    for s in ("blocked 30 segments (1.227 km)", "(463.714 km)", "(2,942.161 km)"):
        assert s in text["roads"], s
    assert text["stretches"].startswith("9 blocked stretches")
    for s in ("265 settlements", "1 connected on observed roads", "163 access unverified", "0 confirmed cut off",
              "101 with no pre-event road access"):
        assert s in text["settlements"], s
    assert text["not_assessed"].startswith("91.2 % of scene pixels were not assessed")
    assert "POST 2026-08-28 12:21:41 UTC" in text["dates"] and "Event date 2026-08-26" in text["dates"]
    assert EMSR927_STATEMENT.replace("**", "") in text["limits"]
    assert len(pack.stretches) == 9 and len(pack.settlements) == 265


@pytest.mark.skipif(os.environ.get("BATO_LLM_LIVE_TEST") != "1", reason="live model test is opt-in")
def test_live_model_answers_are_grounded(cfg):
    aoi, cand = _ids(cfg)
    svc = CopilotService(cfg)
    assert svc.status()["available"], svc.status()
    for q, lang in (("How many settlements have unverified access, and are any confirmed cut off?", "en"),
                    ("कति सडक अवरुद्ध छन्?", "ne"), ("How many people were killed?", "en")):
        r = svc.ask(aoi, cand, q, lang)
        print(f"\n[{lang}] {q}\n -> {r['status']}: {r.get('answer') or r.get('problems')}")
        assert r["status"] in ("answered", "unsupported", "withheld")
        if r["status"] == "answered":
            assert r["citations"]

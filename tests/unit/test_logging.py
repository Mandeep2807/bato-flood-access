"""Stage logging: successes, warnings, fallbacks and failures stay visible."""

from __future__ import annotations

import json
import logging

import pytest

from bato.exceptions import BatoError
from bato.logging_utils import configure_logging, get_logger, stage


@pytest.fixture
def jsonl(tmp_path):
    path = tmp_path / "logs" / "run.jsonl"
    configure_logging("INFO", path)
    yield path
    configure_logging("INFO", None)  # detach the file handler


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_successful_stage_reports_inputs_outputs_and_duration(jsonl):
    with stage("s1_pairing", aoi_id="aoi-test") as report:
        report.record_output(pair_id="pair-test")
    assert report.status == "succeeded" and report.duration_s is not None
    assert report.inputs == {"aoi_id": "aoi-test"} and report.outputs == {"pair_id": "pair-test"}
    events = [e["fields"]["event"] for e in _events(jsonl)]
    assert events == ["started", "finished"]


def test_warnings_and_fallbacks_are_recorded(jsonl):
    with stage("detection") as report:
        report.warn("only one pre-event image")
        report.use_fallback("flood model unavailable; baseline change detection used")
    assert report.warnings == ["only one pre-event image"]
    assert report.fallback.startswith("flood model unavailable")
    levels = {e["fields"]["event"]: e["level"] for e in _events(jsonl)}
    assert levels["warning"] == "WARNING" and levels["fallback"] == "WARNING"


def test_failures_are_logged_and_reraised(jsonl):
    with pytest.raises(BatoError, match="boom"):
        with stage("osm_fetch") as report:
            raise BatoError("boom")
    assert report.status == "failed" and report.error == "BatoError: boom"
    failed = [e for e in _events(jsonl) if e["fields"]["event"] == "failed"]
    assert failed and "Traceback" in failed[0]["exception"]


def test_configure_logging_does_not_duplicate_handlers(tmp_path):
    configure_logging("INFO", None)
    configure_logging("INFO", None)
    bato_handlers = [h for h in logging.getLogger("bato").handlers if getattr(h, "_bato_handler", False)]
    assert len(bato_handlers) == 1


def test_get_logger_namespaces_under_bato():
    assert get_logger("acquisition").name == "bato.acquisition"
    assert get_logger("bato.osm").name == "bato.osm"

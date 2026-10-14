"""Copilot request limits for public hosting (bato.copilot.limits) and their wiring in bato.copilot.api."""

from __future__ import annotations

import json

import pytest

from bato.copilot.api import handle
from bato.copilot.limits import CopilotLimiter

H = {"X-BATO-Copilot": "1"}


class Clock:
    def __init__(self) -> None:
        self.t, self.day = 1000.0, "2026-10-14"

    def __call__(self) -> float:
        return self.t


def limiter(clock: Clock, **kw: int) -> CopilotLimiter:
    return CopilotLimiter(clock=clock, today=lambda: clock.day, **kw)


def test_per_client_minute_window():
    c = Clock()
    lim = limiter(c, per_minute=2)
    assert lim.acquire("a", False).allowed and lim.acquire("a", False).allowed
    refused = lim.acquire("a", False)
    assert not refused.allowed and "2 per minute" in refused.message and 1 <= refused.retry_after_s <= 61
    assert lim.acquire("b", False).allowed                              # other clients are independent
    c.t += 60
    assert lim.acquire("a", False).allowed                              # the window slides


def test_daily_cap_counts_only_model_calls_and_resets():
    c = Clock()
    lim = limiter(c, per_minute=100, per_day=2)
    for _ in range(2):
        assert lim.acquire("a", True).allowed
        lim.release(True)
    refused = lim.acquire("b", True)
    assert not refused.allowed and "daily limit (2 requests)" in refused.message
    assert "deterministic English SITREP still works" in refused.message
    assert lim.acquire("b", False).allowed                              # English SITREP is not capped
    c.day = "2026-10-15"
    assert lim.acquire("b", True).allowed


def test_concurrency_cap_and_release():
    c = Clock()
    lim = limiter(c, per_minute=100, concurrency=1)
    assert lim.acquire("a", True).allowed
    busy = lim.acquire("b", True)
    assert not busy.allowed and "busy" in busy.message
    lim.release(True)
    assert lim.acquire("b", True).allowed


@pytest.mark.parametrize("env", [{"BATO_COPILOT_PER_DAY": "lots"}, {"BATO_COPILOT_PER_MINUTE": "0"}])
def test_invalid_environment_is_refused(env):
    with pytest.raises(ValueError, match="BATO_COPILOT_"):
        CopilotLimiter.from_env(env)


def test_defaults_from_environment():
    assert CopilotLimiter.from_env({}).public() == {"per_minute": 6, "per_day": 200, "concurrency": 2}
    assert CopilotLimiter.from_env({"BATO_COPILOT_PER_DAY": "50"}).per_day == 50


class FakeService:
    def __init__(self, available: bool = True):
        self.available, self.calls = available, 0

    def status(self):
        return {"available": True} if self.available else {"available": False, "reason": "BATO_LLM_PROVIDER unset"}

    def ask(self, aoi, cand, question, lang):
        self.calls += 1
        return {"status": "answered"}

    def sitrep(self, aoi, cand, lang, title):
        self.calls += 1
        return {"status": "deterministic" if lang == "en" else "translated"}


def body(**kw: str) -> bytes:
    return json.dumps({"aoi_id": "a", "candidate_id": "c", **kw}).encode()


def test_api_returns_429_with_retry_after_and_never_calls_the_service():
    svc, c = FakeService(), Clock()
    lim = limiter(c, per_minute=1)
    code, _ = handle(svc, "POST", "/api/copilot/ask", H, body(question="q"), limiter=lim, client="1.2.3.4")
    assert code == 200 and svc.calls == 1
    code, payload = handle(svc, "POST", "/api/copilot/ask", H, body(question="q"), limiter=lim, client="1.2.3.4")
    assert code == 429 and payload["status"] == "rate_limited" and payload["retry_after_s"] >= 1
    assert svc.calls == 1
    assert lim._in_flight == 0                                          # released after the admitted call


def test_api_unavailable_model_is_503_before_any_limit_is_spent():
    svc, c = FakeService(available=False), Clock()
    lim = limiter(c, per_minute=1, per_day=1)
    for _ in range(3):
        code, payload = handle(svc, "POST", "/api/copilot/ask", H, body(question="q"), limiter=lim, client="x")
        assert code == 503 and payload["status"] == "unavailable" and "BATO_LLM_PROVIDER" in payload["error"]
    code, payload = handle(svc, "POST", "/api/copilot/sitrep", H, body(lang="en"), limiter=lim, client="x")
    assert code == 200 and payload["status"] == "deterministic"        # the non-AI SITREP still works
    assert lim._used_today == 0 and svc.calls == 1


def test_api_english_sitrep_survives_a_spent_daily_cap_and_bad_lang_is_400():
    svc, c = FakeService(), Clock()
    lim = limiter(c, per_minute=100, per_day=1)
    assert handle(svc, "POST", "/api/copilot/sitrep", H, body(lang="ne"), limiter=lim, client="x")[0] == 200
    assert handle(svc, "POST", "/api/copilot/sitrep", H, body(lang="ne"), limiter=lim, client="x")[0] == 429
    assert handle(svc, "POST", "/api/copilot/sitrep", H, body(lang="en"), limiter=lim, client="x")[0] == 200
    assert handle(svc, "POST", "/api/copilot/ask", H, body(question="q", lang="fr"), limiter=lim, client="x")[0] == 400


def test_api_header_check_still_comes_first():
    svc, c = FakeService(), Clock()
    lim = limiter(c, per_minute=1)
    for _ in range(3):
        assert handle(svc, "POST", "/api/copilot/ask", {}, body(question="q"), limiter=lim, client="x")[0] == 403
    assert handle(svc, "POST", "/api/copilot/ask", H, body(question="q"), limiter=lim, client="x")[0] == 200

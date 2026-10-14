"""SITREP copilot provider transport against a local scripted HTTP server (no network, no real model):
the Gemini OpenAI-compatible request shape, retries of transient failures, readable provider errors with the key
redacted, and the output-token limit."""

from __future__ import annotations

import http.server
import json
import threading
from typing import Any

import pytest

from bato.copilot.provider import MAX_TOKENS, CopilotUnavailable, HttpChatModel, ProviderError, config_from_env

SECRET = "AIza-test-NEVER-SHOWN-123"
OK = {"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}
GEMINI_503 = [{"error": {"code": 503, "status": "UNAVAILABLE",
                         "message": "This model is currently experiencing high demand. Spikes in demand are "
                                    "usually temporary. Please try again later."}}]


class Scripted:
    """Answers each POST with the next (status, body, headers) step; "drop" closes the connection unanswered."""

    def __init__(self, steps: list[Any]):
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def __enter__(self) -> Scripted:
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                step = outer.steps.pop(0)
                if step == "drop":
                    self.close_connection = True
                    self.connection.close()
                    return
                status, payload, extra = step if len(step) == 3 else (*step, {})
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in extra.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://localhost:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def _model(base: str, sleeps: list[float], **extra: str) -> HttpChatModel:
    env = {"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "gemini-3.5-flash-lite", "BATO_LLM_API_KEY": SECRET,
           "BATO_LLM_BASE_URL": base, **extra}
    return HttpChatModel(config_from_env(env), sleep=sleeps.append)


def test_gemini_openai_compatible_request_shape():
    sleeps: list[float] = []
    with Scripted([(200, OK)]) as srv:
        # the documented Gemini base URL ends in "/v1beta/openai/" (trailing slash)
        assert _model(f"{srv.base}/v1beta/openai/", sleeps).complete("sys", "usr") == "hello"
    sent = srv.requests[0]
    assert sent["path"] == "/v1beta/openai/chat/completions"
    assert sent["headers"]["authorization"] == f"Bearer {SECRET}"
    assert sent["body"]["model"] == "gemini-3.5-flash-lite"                # short id, no "models/" prefix
    assert sent["body"]["max_tokens"] == MAX_TOKENS == 8192 and sent["body"]["temperature"] == 0
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"] and sleeps == []


def test_transient_503_is_retried_then_succeeds():
    sleeps: list[float] = []
    with Scripted([(503, GEMINI_503), (200, OK)]) as srv:
        assert _model(srv.base, sleeps).complete("s", "u") == "hello"
    assert len(srv.requests) == 2 and sleeps == [2.0]


def test_persistent_503_reports_the_provider_reason_without_the_key():
    sleeps: list[float] = []
    echo = [{"error": {**GEMINI_503[0]["error"], "message": f"overloaded (key {SECRET})"}}]
    with Scripted([(503, GEMINI_503), (503, GEMINI_503), (503, echo)]) as srv, \
            pytest.raises(ProviderError) as err:
        _model(srv.base, sleeps).complete("s", "u")
    msg = str(err.value)
    assert msg.startswith("the LLM provider returned HTTP 503 (UNAVAILABLE: overloaded (key [redacted])")
    assert msg.endswith("after 3 attempts") and SECRET not in msg
    assert len(srv.requests) == 3 and sleeps == [2.0, 5.0]


@pytest.mark.parametrize("code,status", [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (404, "NOT_FOUND")])
def test_client_errors_are_not_retried_and_say_why(code, status):
    sleeps: list[float] = []
    body = {"error": {"code": code, "status": status, "message": "models/x is not found for API version v1beta"}}
    with Scripted([(code, body)]) as srv, pytest.raises(ProviderError) as err:
        _model(srv.base, sleeps).complete("s", "u")
    assert str(err.value) == f"the LLM provider returned HTTP {code} ({status}: models/x is not found for API " \
                             f"version v1beta)"
    assert len(srv.requests) == 1 and sleeps == []


def test_retry_after_header_is_honoured_for_429():
    sleeps: list[float] = []
    with Scripted([(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}, {"Retry-After": "7"}),
                   (200, OK)]) as srv:
        assert _model(srv.base, sleeps).complete("s", "u") == "hello"
    assert sleeps == [7.0]


def test_dropped_connection_is_retried():
    sleeps: list[float] = []
    with Scripted(["drop", (200, OK)]) as srv:
        assert _model(srv.base, sleeps).complete("s", "u") == "hello"
    assert len(srv.requests) == 2 and sleeps == [2.0]


def test_empty_reply_cut_off_by_the_token_limit_is_explained():
    sleeps: list[float] = []
    cut = {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}]}
    with Scripted([(200, cut)]) as srv, pytest.raises(ProviderError, match="BATO_LLM_MAX_TOKENS"):
        _model(srv.base, sleeps, BATO_LLM_MAX_TOKENS="1024").complete("s", "u")
    assert srv.requests[0]["body"]["max_tokens"] == 1024


@pytest.mark.parametrize("value", ["lots", "100"])
def test_bad_max_tokens_makes_the_copilot_unavailable(value):
    with pytest.raises(CopilotUnavailable, match="BATO_LLM_MAX_TOKENS"):
        config_from_env({"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "m", "BATO_LLM_API_KEY": "k",
                         "BATO_LLM_MAX_TOKENS": value})

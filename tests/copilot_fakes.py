"""TEST FIXTURES for the SITREP copilot: mocked LLM providers (no network, no real model).

* ``ScriptedModel``: returns the given replies in order and records every (system, user) prompt.
* ``EchoModel``: a rule-based stand-in that answers ONLY by quoting evidence text it was given (so its numbers are
  grounded), in English or Nepali (Devanagari digits), and "translates" a SITREP by swapping a few phrases and
  digits while keeping the attribution lines verbatim.
* ``FakeLLMServer``: a local HTTP server speaking the OpenAI-compatible ``/v1/chat/completions`` and the Anthropic
  ``/v1/messages`` formats, answering with ``EchoModel`` and recording the requests (to check the API key header).
"""

from __future__ import annotations

import http.server
import json
import re
import threading
from typing import Any

from bato.attribution import REQUIRED_ATTRIBUTIONS

DEV = str.maketrans("0123456789", "०१२३४५६७८९")
NE_PHRASES = [("Operational summary", "सञ्चालन सारांश"), ("Roads and settlement access", "सडक र बस्ती पहुँच"),
              ("Priorities for field verification", "फिल्ड प्रमाणीकरणका प्राथमिकता"),
              ("Key uncertainty and limitations", "मुख्य अनिश्चितता र सीमाहरू"),
              ("Sources and attribution", "स्रोत र श्रेय"), ("Road state", "सडक अवस्था"),
              ("Observed open", "अवलोकन गरिएको खुला"), ("Unobservable", "अवलोकन गर्न नसकिएको"),
              ("Access unverified", "पहुँच अपुष्ट"), ("Confirmed cutoff", "पुष्टि भएको अवरोध")]


class ScriptedModel:
    label = "fake:scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("the scripted model was called more often than expected")
        return self.replies.pop(0)


def reply(answer: str, citations: list[str], supported: bool = True) -> str:
    return json.dumps({"supported": supported, "answer": answer, "citations": citations}, ensure_ascii=False)


def translate_like(markdown: str) -> str:
    """A deterministic pseudo-translation: Nepali headings/labels and Devanagari digits, attribution untouched."""
    out = []
    for line in markdown.splitlines():
        if any(a in line for a in REQUIRED_ATTRIBUTIONS):
            out.append(line)
            continue
        for en, ne in NE_PHRASES:
            line = line.replace(en, ne)
        out.append(line.translate(DEV))
    return "\n".join(out)


class EchoModel:
    label = "fake:echo"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if system.startswith("Translate"):
            return translate_like(user.split("\n\nYOUR PREVIOUS")[0])
        evidence = json.loads(user.split("EVIDENCE (the only facts you may use):\n", 1)[1].split("\n\nQUESTION:", 1)[0])
        by_id = {e["id"]: e for e in evidence}
        question = user.rsplit("QUESTION:\n", 1)[1].lower()
        nepali = "Nepali" in system
        if any(w in question for w in ("population", "casualt", "rain", "forecast", "जनसंख्या", "मृत्यु")):
            msg = "जनसंख्या, मृत्यु वा पूर्वानुमान BATO प्रमाणमा छैन।" if nepali else \
                "BATO evidence does not contain population, casualties or forecasts."
            return reply(msg, ["limits"], supported=False)
        stretch = next((e["id"] for e in evidence if e["id"].startswith("stretch:")), None)
        key = "settlements" if any(w in question for w in ("settlement", "बस्ती")) else \
            "dates" if any(w in question for w in ("when", "date", "कहिले")) else \
            stretch if stretch and "stretch" in question else "roads"
        sentence = re.split(r"(?<=\.)\s", by_id[key]["text"])[0]
        text = ("प्रमाण अनुसार: " + sentence.translate(DEV)) if nepali else sentence
        return reply(f"{text} [{key}]", [key])


class FakeLLMServer:
    """Context manager: ``with FakeLLMServer() as srv: srv.base_url`` (http://127.0.0.1:<port>)."""

    def __init__(self, model: Any | None = None):
        self.model = model or EchoModel()
        self.requests: list[dict[str, Any]] = []

    def __enter__(self) -> FakeLLMServer:
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
                if self.path.endswith("/chat/completions"):
                    msgs = {m["role"]: m["content"] for m in body["messages"]}
                    text = outer.model.complete(msgs["system"], msgs["user"])
                    payload: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": text}}]}
                elif self.path.endswith("/v1/messages"):
                    text = outer.model.complete(body["system"], body["messages"][0]["content"])
                    payload = {"content": [{"type": "text", "text": text}]}
                else:
                    self.send_error(404)
                    return
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

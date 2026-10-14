"""HTTP API of the copilot, served by scripts/serve_dashboard.py (same origin as the dashboard, localhost).

    GET  /api/copilot/status                       -> provider availability (never the key)
    POST /api/copilot/ask     {aoi_id, candidate_id, question, lang}
    POST /api/copilot/sitrep  {aoi_id, candidate_id, lang, title}

POST requests must carry ``X-BATO-Copilot: 1`` and a JSON body. A custom header cannot be sent cross-site without
a CORS preflight, which this server never grants, so other web pages cannot spend the configured API quota.
"""

from __future__ import annotations

import json
from typing import Any

from bato.copilot.evidence import EvidenceError
from bato.copilot.provider import CopilotUnavailable, ProviderError
from bato.copilot.service import CopilotService

MAX_BODY = 16_384
HEADER = "X-BATO-Copilot"


def handle(service: CopilotService, method: str, path: str, headers: Any, body: bytes) -> tuple[int, dict[str, Any]]:
    """Returns (HTTP status, JSON payload). Never raises for client errors."""
    if path == "/api/copilot/status" and method == "GET":
        return 200, service.status()
    if path not in ("/api/copilot/ask", "/api/copilot/sitrep"):
        return 404, {"error": "not found"}
    if method != "POST":
        return 405, {"error": "use POST"}
    if headers.get(HEADER) != "1":
        return 403, {"error": f"missing {HEADER} header"}
    if len(body) > MAX_BODY:
        return 413, {"error": "request too large"}
    try:
        req = json.loads(body.decode("utf-8") or "{}")
        aoi, cand = str(req["aoi_id"]), str(req["candidate_id"])
        lang = str(req.get("lang", "en"))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return 400, {"error": "expected JSON with aoi_id, candidate_id"}
    try:
        if path.endswith("/ask"):
            return 200, service.ask(aoi, cand, str(req.get("question", "")), lang)
        return 200, service.sitrep(aoi, cand, lang, str(req.get("title", "")))
    except CopilotUnavailable as exc:
        return 503, {"status": "unavailable", "error": f"Copilot unavailable: {exc}"}
    except ProviderError as exc:
        return 502, {"status": "provider_error", "error": str(exc)}
    except EvidenceError as exc:
        return 409, {"status": "no_evidence", "error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}

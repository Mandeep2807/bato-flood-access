"""HTTP API of the copilot, served by scripts/serve_dashboard.py (same origin as the dashboard, localhost).

    GET  /api/copilot/status                       -> provider availability (never the key)
    POST /api/copilot/ask     {aoi_id, candidate_id, question, lang}
    POST /api/copilot/sitrep  {aoi_id, candidate_id, lang, title}

POST requests must carry ``X-BATO-Copilot: 1`` and a JSON body. A custom header cannot be sent cross-site without
a CORS preflight, which this server never grants, so other web pages cannot spend the configured API quota.

With a ``limiter`` (public deployments), POSTs are also rate-limited per client and model calls are capped per day
and in flight (bato.copilot.limits); a refused request gets HTTP 429 with ``retry_after_s``. When the AI provider
is not configured, a model call is answered 503 "unavailable" before it counts against any limit.
"""

from __future__ import annotations

import json
from typing import Any

from bato.copilot.evidence import EvidenceError
from bato.copilot.limits import CopilotLimiter
from bato.copilot.provider import CopilotUnavailable, ProviderError
from bato.copilot.service import CopilotService

MAX_BODY = 16_384
HEADER = "X-BATO-Copilot"


def handle(service: CopilotService, method: str, path: str, headers: Any, body: bytes, *,
           limiter: CopilotLimiter | None = None, client: str = "local") -> tuple[int, dict[str, Any]]:
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
    if limiter is None:
        return _dispatch(service, path, req, aoi, cand, lang)
    if lang not in ("en", "ne"):
        return 400, {"error": "lang must be 'en' or 'ne'"}
    uses_model = path.endswith("/ask") or lang == "ne"
    if uses_model:
        st = service.status()
        if not st.get("available"):
            return 503, {"status": "unavailable", "error": f"Copilot unavailable: {st.get('reason', 'not configured')}"}
    decision = limiter.acquire(client, uses_model)
    if not decision.allowed:
        return 429, {"status": "rate_limited", "error": decision.message, "retry_after_s": decision.retry_after_s}
    try:
        return _dispatch(service, path, req, aoi, cand, lang)
    finally:
        limiter.release(uses_model)


def _dispatch(service: CopilotService, path: str, req: dict[str, Any], aoi: str, cand: str,
              lang: str) -> tuple[int, dict[str, Any]]:
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

"""Serve the BATO dashboard locally (standard library only).

    python scripts/serve_dashboard.py            # http://127.0.0.1:8765/
    python scripts/serve_dashboard.py --open     # and open the browser

Public hosting (e.g. Render; see docs/deployment.md):

    python scripts/serve_dashboard.py --host 0.0.0.0 --check-evidence --trust-proxy

The port defaults to the ``PORT`` environment variable when it is set (Render sets it), else 8765. ``/healthz``
reports readiness; with ``--check-evidence`` it stays 503 until the copilot evidence of every published bundle has
been rebuilt and hash-verified (status "error" and HTTP 500 if that fails). When the server listens on a
non-loopback address, copilot POSTs are rate-limited (bato.copilot.limits; ``--copilot-limits`` forces this
locally). ``--trust-proxy`` takes the client address from ``X-Forwarded-For`` (only behind a proxy that sets it).

Serves ``dashboard/app/`` at ``/`` and the built bundles (``<processed>/dashboard/``, written by
scripts/build_dashboard.py) at ``/bundles/``. Read-only; binds to localhost by default.

``/api/copilot/*`` serves the SITREP Copilot (bato.copilot.api). It needs an LLM provider configured through
environment variables (BATO_LLM_PROVIDER, BATO_LLM_MODEL, BATO_LLM_API_KEY; see docs/copilot.md); without one the
copilot reports itself unavailable and the rest of the dashboard works as before.
Exit codes: 0 stopped, 1 no bundle built yet / invalid config.
"""

from __future__ import annotations

import argparse
import gzip
import http.server
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bato.config import load_config
from bato.copilot.api import handle as copilot_handle
from bato.copilot.evidence import EvidenceError
from bato.copilot.limits import CopilotLimiter
from bato.copilot.service import CopilotService
from bato.exceptions import ConfigError

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "dashboard" / "app"
TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json",
         ".woff2": "font/woff2", ".png": "image/png", ".svg": "image/svg+xml", ".html": "text/html"}
LOOPBACK = ("127.0.0.1", "localhost", "::1")
GZIP_CACHE_MAX_BYTES = 64 << 20


class Health:
    """Readiness for /healthz: "ok", "starting" (evidence check running) or "error" (with the reason)."""

    def __init__(self) -> None:
        self.state, self.detail = "ok", ""
        self.checked: list[str] = []

    def payload(self) -> tuple[int, dict[str, Any]]:
        code = {"ok": 200, "starting": 503}.get(self.state, 500)
        return code, {"status": self.state, "detail": self.detail, "evidence_verified": self.checked}


def bundle_ids(bundles: Path) -> list[tuple[str, str]]:
    entries = json.loads((bundles / "index.json").read_text(encoding="utf-8"))["bundles"]
    return [(str(e["aoi_id"]), str(e["candidate_id"])) for e in entries]


def check_evidence(service: CopilotService, bundles: Path, health: Health) -> None:
    """Rebuild and hash-verify the copilot evidence (official v1 run) of every published bundle."""
    health.state = "starting"
    try:
        ids = bundle_ids(bundles)
        if not ids:
            raise EvidenceError("index.json lists no bundles")
        for aoi, cand in ids:
            service.pack(aoi, cand)
            health.checked.append(f"{aoi}/{cand}")
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        health.state, health.detail = "error", f"evidence check failed: {exc}"
        print(f"EVIDENCE CHECK FAILED: {exc}", file=sys.stderr, flush=True)
        return
    health.state, health.detail = "ok", "official evidence verified"
    print(f"Evidence verified: {', '.join(health.checked)}", flush=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    bundles: Path = Path(".")
    copilot: CopilotService | None = None
    limiter: CopilotLimiter | None = None
    trust_proxy: bool = False
    health: Health = Health()
    _gz: dict[Path, tuple[int, int, bytes]] = {}
    _gz_lock = threading.Lock()

    def _client(self) -> str:
        if self.trust_proxy:
            forwarded = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            if forwarded:
                return forwarded
        return str(self.client_address[0])

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        retry = payload.get("retry_after_s")
        if status == 429 and isinstance(retry, int):
            self.send_header("Retry-After", str(retry))
        self.end_headers()
        self.wfile.write(data)

    def _gzipped(self, target: Path, data: bytes) -> bytes:
        st = target.stat()
        with self._gz_lock:
            hit = self._gz.get(target)
            if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                return hit[2]
        gz = gzip.compress(data, 5)
        with self._gz_lock:
            if sum(len(v[2]) for v in self._gz.values()) + len(gz) > GZIP_CACHE_MAX_BYTES:
                self._gz.clear()
            self._gz[target] = (st.st_mtime_ns, st.st_size, gz)
        return gz

    def _copilot(self, method: str) -> None:
        if self.copilot is None:
            status, payload = 503, {"status": "unavailable", "error": "Copilot unavailable: not started"}
        else:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(n, 1 << 20)) if n > 0 else b""
            status, payload = copilot_handle(self.copilot, method, urlsplit(self.path).path, self.headers, body,
                                             limiter=self.limiter, client=self._client())
        self._json(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.startswith("/api/copilot/"):
            self._copilot("POST")
        else:
            self.send_error(405)

    def translate(self) -> Path | None:
        path = unquote(urlsplit(self.path).path)
        if path.startswith("/bundles/"):
            base, rel = self.bundles, path[len("/bundles/"):]
        else:
            base, rel = APP, path.lstrip("/") or "index.html"
        target = (base / rel).resolve()
        if base.resolve() not in target.parents and target != base.resolve():
            return None
        if target.is_dir():
            target = target / "index.html"
        return target if target.is_file() else None

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.startswith("/api/copilot/"):
            self._copilot("GET")
            return
        if urlsplit(self.path).path == "/healthz":
            self._json(*self.health.payload())
            return
        target = self.translate()
        if target is None:
            self.send_error(404)
            return
        data = target.read_bytes()
        ctype = TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        gz = target.suffix in (".json", ".js", ".css") and "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            data = self._gzipped(target, data)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the BATO dashboard (localhost by default).")
    parser.add_argument("--port", type=int, default=None, help="default: $PORT if set, else 8765")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve publicly (e.g. on Render)")
    parser.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--check-evidence", action="store_true",
                        help="verify the copilot evidence of every bundle at start-up (reported by /healthz)")
    parser.add_argument("--copilot-limits", action="store_true",
                        help="rate-limit copilot requests even on a loopback address (always on for public hosts)")
    parser.add_argument("--trust-proxy", action="store_true",
                        help="identify clients by X-Forwarded-For (only behind a proxy that sets it)")
    args = parser.parse_args()
    if args.port is None:
        try:
            args.port = int(os.environ.get("PORT") or 8765)
        except ValueError:
            print(f"INVALID PORT environment variable: {os.environ.get('PORT')!r}", file=sys.stderr)
            return 1
    try:
        cfg = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"INVALID CONFIG: {exc}", file=sys.stderr)
        return 1
    bundles = cfg.settings.paths.processed / "dashboard"
    if not (bundles / "index.json").is_file():
        print(f"No dashboard bundle yet ({bundles / 'index.json'} missing). Run scripts/build_dashboard.py first.",
              file=sys.stderr)
        return 1
    limiter = None
    if args.copilot_limits or args.host not in LOOPBACK:
        try:
            limiter = CopilotLimiter.from_env()
        except ValueError as exc:
            print(f"INVALID COPILOT LIMIT: {exc}", file=sys.stderr)
            return 1
    handler = partial(Handler)
    Handler.bundles = bundles
    Handler.copilot = CopilotService(cfg)
    Handler.limiter = limiter
    Handler.trust_proxy = args.trust_proxy
    Handler.health = Health()
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"BATO dashboard: {url}  (bundles from {bundles}; Ctrl+C to stop)", flush=True)
    st = Handler.copilot.status()
    print("SITREP Copilot: " + (f"available ({st['provider']}:{st['model']})" if st["available"]
                                 else f"unavailable - {st['reason']}"), flush=True)
    if limiter is not None:
        lim = limiter.public()
        print(f"Copilot limits: {lim['per_minute']}/min per client, {lim['per_day']} AI requests/day, "
              f"{lim['concurrency']} at once", flush=True)
    if args.check_evidence:
        Handler.health.state = "starting"
        threading.Thread(target=check_evidence, args=(Handler.copilot, bundles, Handler.health), daemon=True).start()
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}/" if args.host in ("0.0.0.0", "::") else url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

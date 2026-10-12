"""Serve the BATO dashboard locally (standard library only).

    python scripts/serve_dashboard.py            # http://127.0.0.1:8765/
    python scripts/serve_dashboard.py --open     # and open the browser

Serves ``dashboard/app/`` at ``/`` and the built bundles (``<processed>/dashboard/``, written by
scripts/build_dashboard.py) at ``/bundles/``. Read-only; binds to localhost by default.
Exit codes: 0 stopped, 1 no bundle built yet / invalid config.
"""

from __future__ import annotations

import argparse
import gzip
import http.server
import mimetypes
import sys
import webbrowser
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bato.config import load_config
from bato.exceptions import ConfigError

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "dashboard" / "app"
TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json",
         ".woff2": "font/woff2", ".png": "image/png", ".svg": "image/svg+xml", ".html": "text/html"}


class Handler(http.server.SimpleHTTPRequestHandler):
    bundles: Path = Path(".")

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
        target = self.translate()
        if target is None:
            self.send_error(404)
            return
        data = target.read_bytes()
        ctype = TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        gz = target.suffix in (".json", ".js", ".css") and "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            data = gzip.compress(data, 5)
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
    parser = argparse.ArgumentParser(description="Serve the BATO dashboard on localhost.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
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
    handler = partial(Handler)
    Handler.bundles = bundles
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"BATO dashboard: {url}  (bundles from {bundles}; Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

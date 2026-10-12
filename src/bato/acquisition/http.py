"""Minimal HTTP transport (standard library only).

Provider clients take a :class:`Transport` so tests can substitute a fake
one without network access. The production implementation,
:class:`UrllibTransport`, uses ``urllib`` and therefore honours the standard
``HTTPS_PROXY`` / ``SSL_CERT_FILE`` environment variables.

Non-2xx responses are returned, not raised; callers decide what an error
status means. Network failures raise :class:`TransportError`.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from bato import __version__

USER_AGENT = f"bato/{__version__} (hackathon prototype)"


class TransportError(Exception):
    """The request could not be completed (DNS, TLS, connection, timeout)."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes

    def text(self, limit: int | None = None) -> str:
        text = self.body.decode("utf-8", errors="replace")
        return text if limit is None else text[:limit]


class Transport(Protocol):
    def get(self, url: str, timeout_s: float) -> HttpResponse: ...

    def post_form(self, url: str, form: dict[str, str], timeout_s: float) -> HttpResponse: ...


class UrllibTransport:
    """Production transport using ``urllib.request``."""

    def get(self, url: str, timeout_s: float) -> HttpResponse:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        return self._send(request, timeout_s)

    def post_form(self, url: str, form: dict[str, str], timeout_s: float) -> HttpResponse:
        data = urllib.parse.urlencode(form).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, method="POST",
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._send(request, timeout_s)

    @staticmethod
    def _send(request: urllib.request.Request, timeout_s: float) -> HttpResponse:
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (https URLs from config)
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read() or b"")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Only the host is reported; query strings and form bodies are not.
            host = urllib.parse.urlsplit(request.full_url).netloc
            raise TransportError(f"request to {host} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Streaming downloads
# ---------------------------------------------------------------------------


class RedirectRefusedError(TransportError):
    """A redirect pointed to a host outside the allow-list. Not retried: following
    it would send the Authorization header to an untrusted host."""


@dataclass
class StreamResponse:
    status: int
    headers: dict[str, str]              # lower-case names
    final_url: str                       # after redirects
    chunks: Iterator[bytes]
    close: Callable[[], None]


class StreamingTransport(Protocol):
    def open_stream(self, url: str, headers: dict[str, str], timeout_s: float,
                    chunk_bytes: int) -> StreamResponse: ...


def host_allowed(url: str, allowed_suffixes: tuple[str, ...]) -> bool:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and any(host == s or host.endswith("." + s) for s in allowed_suffixes)


class _AllowListRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_suffixes: tuple[str, ...]) -> None:
        super().__init__()
        self.allowed_suffixes = allowed_suffixes

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        if not host_allowed(newurl, self.allowed_suffixes):
            host = urllib.parse.urlsplit(newurl).hostname
            raise RedirectRefusedError(f"refusing redirect to {host!r}: not an allowed https host")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.get_header("Authorization")
        if new is not None and auth and not new.get_header("Authorization"):
            new.add_header("Authorization", auth)  # CDSE redirects within its own domain need the token
        return new


class UrllibStreamingTransport:
    """Streams a response body in chunks; follows redirects only to allowed hosts."""

    def __init__(self, allowed_redirect_suffixes: tuple[str, ...]) -> None:
        self._opener = urllib.request.build_opener(_AllowListRedirectHandler(allowed_redirect_suffixes))

    def open_stream(self, url: str, headers: dict[str, str], timeout_s: float,
                    chunk_bytes: int) -> StreamResponse:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
        host = urllib.parse.urlsplit(url).netloc
        try:
            response = self._opener.open(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read() or b""
            return StreamResponse(status=exc.code, headers={k.lower(): v for k, v in exc.headers.items()},
                                  final_url=exc.geturl() or url, chunks=iter([body]), close=exc.close)
        except RedirectRefusedError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"request to {host} failed: {exc}") from exc

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    block = response.read(chunk_bytes)
                    if not block:
                        return
                    yield block
            except (TimeoutError, OSError) as exc:
                raise TransportError(f"connection to {host} dropped during download: {exc}") from exc

        return StreamResponse(status=response.status, headers={k.lower(): v for k, v in response.headers.items()},
                              final_url=response.geturl(), chunks=chunks(), close=response.close)

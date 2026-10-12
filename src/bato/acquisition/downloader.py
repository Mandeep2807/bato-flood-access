"""Resumable, verifying file download (one remote file -> one local file).

Behaviour
---------
* Data is written to ``<dest>.part`` and renamed to ``<dest>`` only after the
  size and (when the server publishes one we can compute) the checksum match.
  A crash never leaves a file that looks complete.
* Resume: if ``<dest>.part`` exists, ``Range: bytes=<n>-`` is sent. HTTP 206
  appends; HTTP 200 (range ignored) restarts from zero.
* Retries: network errors, dropped connections, HTTP 429 and 5xx are retried
  with exponential backoff (``Retry-After`` honoured), up to ``max_attempts``.
  HTTP 401 refreshes the token (twice at most). 403/404 and refused redirects
  fail immediately with a clear message.
* Idempotent: an existing ``<dest>`` that matches the expected size/checksum is
  not downloaded again; one that does not match is never overwritten silently.
* Disk space is checked before writing.

The access token is only ever placed in the Authorization header; it never
appears in logs, errors or provenance.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bato.acquisition.http import RedirectRefusedError, StreamingTransport, TransportError
from bato.config import DownloadSettings
from bato.exceptions import AuthenticationError, ProviderError

_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_TOKEN_REFRESHES = 2
_PREFERRED_ALGORITHMS = ("sha256", "md5", "sha3256", "blake3")


class DownloadError(ProviderError):
    """A file could not be downloaded or failed verification."""


class ChecksumMismatchError(DownloadError):
    """The downloaded bytes do not match the published checksum."""


class TokenSource(Protocol):
    def get(self) -> str: ...

    def invalidate(self) -> None: ...


@dataclass(frozen=True)
class RemoteFile:
    url: str
    label: str                              # used in messages, e.g. the product name
    expected_size: int | None
    checksums: dict[str, str]               # algorithm -> hex, as published by the server


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    sha256: str
    verified_with: str | None               # algorithm used for verification, if any
    attempts: int
    resumed: bool
    final_host: str
    skipped_existing: bool = False


def _hashers(checksums: dict[str, str]) -> dict[str, Any]:
    """Hash objects for SHA-256 (always) and every published algorithm we can compute."""
    result: dict[str, Any] = {"sha256": hashlib.sha256()}
    for algorithm in checksums:
        if algorithm in result:
            continue
        if algorithm == "md5":
            result["md5"] = hashlib.md5(usedforsecurity=False)
        elif algorithm == "sha3256":
            result["sha3256"] = hashlib.sha3_256()
        elif algorithm == "blake3":
            try:
                import blake3  # type: ignore[import-not-found, unused-ignore]  # optional dependency
            except ImportError:
                continue
            result["blake3"] = blake3.blake3()
    return result


def hash_file(path: Path, checksums: dict[str, str], chunk_bytes: int = 8 << 20) -> dict[str, str]:
    """SHA-256 plus every published algorithm we can compute, in one pass."""
    hashers = _hashers(checksums)
    with path.open("rb") as fh:
        while block := fh.read(chunk_bytes):
            for h in hashers.values():
                h.update(block)
    return {name: h.hexdigest() for name, h in hashers.items()}


def verify_hashes(computed: dict[str, str], published: dict[str, str]) -> str | None:
    """Return the algorithm that verified the file, None if none could be checked.

    Raises:
        ChecksumMismatchError: a published checksum we can compute does not match.
    """
    verified = None
    for algorithm in _PREFERRED_ALGORITHMS:
        if algorithm in published and algorithm in computed:
            if computed[algorithm] != published[algorithm].lower():
                raise ChecksumMismatchError(
                    f"{algorithm} mismatch: expected {published[algorithm]}, got {computed[algorithm]}")
            verified = verified or algorithm
    return verified


class Downloader:
    def __init__(self, settings: DownloadSettings, transport: StreamingTransport,
                 tokens: TokenSource | None, sleep: Callable[[float], None] = time.sleep,
                 disk_usage: Callable[[Path], Any] = shutil.disk_usage,
                 logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.tokens = tokens
        self.sleep = sleep
        self.disk_usage = disk_usage
        self.log = logger or logging.getLogger("bato.acquisition.download")

    # -- helpers -------------------------------------------------------------

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.strip().isdigit():
            return min(float(retry_after.strip()), self.settings.backoff_max_s)
        return min(self.settings.backoff_initial_s * (2 ** (attempt - 1)), self.settings.backoff_max_s)

    def _check_disk(self, directory: Path, remaining: int | None, label: str) -> None:
        if remaining is None or remaining <= 0:
            return
        free = self.disk_usage(directory).free
        needed = int(remaining * self.settings.min_free_disk_ratio)
        if free < needed:
            raise DownloadError(f"{label}: not enough disk space in {directory} "
                                f"({free / 1e9:.2f} GB free, {needed / 1e9:.2f} GB needed)")

    def _existing(self, remote: RemoteFile, dest: Path) -> DownloadResult | None:
        if not dest.exists():
            return None
        size = dest.stat().st_size
        if remote.expected_size is not None and size != remote.expected_size:
            raise DownloadError(f"{remote.label}: {dest} exists with {size} bytes but {remote.expected_size} "
                                "are expected; move it away and re-run (it is never overwritten silently)")
        if not self.settings.verify_existing:
            hashes = {"sha256": hash_file(dest, {})["sha256"]}
            return DownloadResult(dest, size, hashes["sha256"], None, 1, False, "", skipped_existing=True)
        hashes = hash_file(dest, remote.checksums, self.settings.chunk_bytes)
        try:
            verified = verify_hashes(hashes, remote.checksums)
        except ChecksumMismatchError as exc:
            raise DownloadError(f"{remote.label}: existing file {dest} fails verification ({exc}); "
                                "move it away and re-run") from exc
        return DownloadResult(dest, size, hashes["sha256"], verified, 1, False, "", skipped_existing=True)

    # -- main ----------------------------------------------------------------

    def fetch(self, remote: RemoteFile, dest: Path) -> DownloadResult:
        """Download ``remote`` to ``dest`` (see module docstring).

        Raises:
            DownloadError / ChecksumMismatchError / AuthenticationError with a clear message.
        """
        existing = self._existing(remote, dest)
        if existing is not None:
            self.log.info("already present, verified: %s", remote.label)
            return existing

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        resumed = False
        refreshes = 0
        last_error = "no attempt made"
        attempt = 0
        while attempt < self.settings.max_attempts:
            attempt += 1
            offset = part.stat().st_size if part.exists() else 0
            remaining = None if remote.expected_size is None else remote.expected_size - offset
            self._check_disk(dest.parent, remaining, remote.label)
            headers = {}
            if self.tokens is not None:
                headers["Authorization"] = f"Bearer {self.tokens.get()}"
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                response = self.transport.open_stream(remote.url, headers, self.settings.timeout_s,
                                                      self.settings.chunk_bytes)
            except RedirectRefusedError as exc:
                raise DownloadError(f"{remote.label}: {exc}") from exc
            except TransportError as exc:
                last_error = str(exc)
                self.log.warning("download attempt %d/%d failed for %s: %s", attempt,
                                 self.settings.max_attempts, remote.label, last_error)
                self.sleep(self._backoff(attempt, None))
                continue

            status = response.status
            if status == 401:
                response.close()
                if self.tokens is None or refreshes >= _MAX_TOKEN_REFRESHES:
                    raise AuthenticationError(f"{remote.label}: CDSE rejected the access token (HTTP 401)")
                refreshes += 1
                self.tokens.invalidate()
                attempt -= 1  # a token refresh is not a failed attempt
                continue
            if status == 416 and offset and remote.expected_size == offset:
                response.close()  # the .part file is already complete
            elif status == 416:
                response.close()
                part.unlink(missing_ok=True)
                last_error = "server rejected the resume range (HTTP 416); restarting from zero"
                continue
            elif status in _RETRY_STATUSES:
                retry_after = response.headers.get("retry-after")
                response.close()
                last_error = f"HTTP {status}"
                self.log.warning("download attempt %d/%d for %s: HTTP %d, retrying", attempt,
                                 self.settings.max_attempts, remote.label, status)
                self.sleep(self._backoff(attempt, retry_after))
                continue
            elif status == 403:
                response.close()
                raise DownloadError(f"{remote.label}: access denied (HTTP 403) - check that the CDSE account "
                                    "may download this product and that no quota is exhausted")
            elif status == 404:
                response.close()
                raise DownloadError(f"{remote.label}: not found at the download service (HTTP 404)")
            elif status not in (200, 206):
                response.close()
                raise DownloadError(f"{remote.label}: unexpected HTTP {status} from the download service")
            else:
                mode = "wb"
                if status == 206:
                    match = _RANGE.match(response.headers.get("content-range", ""))
                    if offset and match and int(match.group(1)) == offset:
                        mode, resumed = "ab", True
                    else:
                        response.close()
                        part.unlink(missing_ok=True)
                        last_error = "server returned a range that does not continue the partial file"
                        continue
                try:
                    with part.open(mode) as fh:
                        for block in response.chunks:
                            fh.write(block)
                except TransportError as exc:
                    last_error = str(exc)
                    self.log.warning("connection dropped for %s after %d bytes; will resume", remote.label,
                                     part.stat().st_size if part.exists() else 0)
                    self.sleep(self._backoff(attempt, None))
                    continue
                finally:
                    response.close()

            size = part.stat().st_size
            if remote.expected_size is not None and size < remote.expected_size:
                last_error = f"incomplete: {size} of {remote.expected_size} bytes"
                continue
            if remote.expected_size is not None and size > remote.expected_size:
                part.unlink()
                last_error = f"received {size} bytes, more than the expected {remote.expected_size}"
                continue
            hashes = hash_file(part, remote.checksums, self.settings.chunk_bytes)
            try:
                verified = verify_hashes(hashes, remote.checksums)
            except ChecksumMismatchError as exc:
                part.unlink()
                last_error = f"{exc}"
                self.log.warning("checksum mismatch for %s; discarding and retrying", remote.label)
                continue
            os.replace(part, dest)
            final_host = urllib.parse.urlsplit(response.final_url).hostname or ""
            return DownloadResult(dest, size, hashes["sha256"], verified, attempt, resumed, final_host)

        raise DownloadError(f"{remote.label}: download failed after {self.settings.max_attempts} attempts "
                            f"(last error: {last_error}); re-running resumes from the partial file")

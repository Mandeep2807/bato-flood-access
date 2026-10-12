"""Resumable downloader and streaming transport safety (all HTTP faked)."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from collections import namedtuple

import pytest

from bato.acquisition.downloader import ChecksumMismatchError, Downloader, DownloadError, RemoteFile, verify_hashes
from bato.acquisition.http import (
    RedirectRefusedError,
    StreamResponse,
    TransportError,
    _AllowListRedirectHandler,
    host_allowed,
)
from bato.exceptions import AuthenticationError

PAYLOAD = bytes(range(256)) * 400          # 102,400 bytes of synthetic content
MD5 = hashlib.md5(PAYLOAD).hexdigest()
URL = "https://download.dataspace.copernicus.eu/odata/v1/Products(abc)/$value"
DiskUsage = namedtuple("DiskUsage", "total used free")


class FakeTokens:
    def __init__(self) -> None:
        self.issued = 0

    def get(self) -> str:
        self.issued += 1
        return f"SECRET-TOKEN-{self.issued}"

    def invalidate(self) -> None:
        pass


def resp(status: int, body: bytes = b"", headers: dict | None = None, break_after: int | None = None) -> StreamResponse:
    def chunks():
        if break_after is None:
            yield body
            return
        yield body[:break_after]
        raise TransportError("connection reset by peer")
    return StreamResponse(status=status, headers=headers or {}, final_url=URL, chunks=chunks(), close=lambda: None)


class FakeStream:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def open_stream(self, url, headers, timeout_s, chunk_bytes):
        self.requests.append(dict(headers))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(headers) if callable(item) else item


@pytest.fixture
def make(cfg):
    sleeps: list[float] = []

    def _make(script, tokens=True, free=10**12):
        stream = FakeStream(script)
        dl = Downloader(cfg.settings.download, stream, FakeTokens() if tokens else None, sleep=sleeps.append,
                        disk_usage=lambda _p: DiskUsage(0, 0, free))
        return dl, stream, sleeps
    return _make


REMOTE = RemoteFile(url=URL, label="TEST-PRODUCT", expected_size=len(PAYLOAD), checksums={"md5": MD5})


def test_success_verifies_checksum_and_is_atomic(make, tmp_path, caplog):
    dl, stream, _ = make([resp(200, PAYLOAD)])
    with caplog.at_level(logging.DEBUG):
        result = dl.fetch(REMOTE, tmp_path / "p.zip")
    assert (tmp_path / "p.zip").read_bytes() == PAYLOAD and not (tmp_path / "p.zip.part").exists()
    assert result.verified_with == "md5" and result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert stream.requests[0]["Authorization"] == "Bearer SECRET-TOKEN-1"
    assert "SECRET-TOKEN" not in caplog.text


def test_retries_transient_errors_with_backoff_and_retry_after(make, tmp_path):
    dl, _, sleeps = make([TransportError("timed out"), resp(503), resp(429, headers={"retry-after": "7"}),
                          resp(200, PAYLOAD)])
    result = dl.fetch(REMOTE, tmp_path / "p.zip")
    assert result.attempts == 4 and sleeps == [5.0, 10.0, 7.0]


def test_dropped_connection_resumes_with_range(make, tmp_path):
    def second(headers):
        assert headers["Range"] == "bytes=40000-"
        return resp(206, PAYLOAD[40000:], headers={"content-range": f"bytes 40000-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"})
    dl, _, _ = make([resp(200, PAYLOAD, break_after=40000), second])
    result = dl.fetch(REMOTE, tmp_path / "p.zip")
    assert result.resumed and (tmp_path / "p.zip").read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts_from_zero(make, tmp_path):
    (tmp_path / "p.zip.part").write_bytes(PAYLOAD[:1000])
    dl, stream, _ = make([resp(200, PAYLOAD)])
    result = dl.fetch(REMOTE, tmp_path / "p.zip")
    assert stream.requests[0]["Range"] == "bytes=1000-"
    assert not result.resumed and (tmp_path / "p.zip").read_bytes() == PAYLOAD


def test_mismatched_content_range_discards_partial(make, tmp_path):
    (tmp_path / "p.zip.part").write_bytes(PAYLOAD[:1000])
    dl, _, _ = make([resp(206, PAYLOAD[500:], headers={"content-range": "bytes 500-102399/102400"}),
                     resp(200, PAYLOAD)])
    assert dl.fetch(REMOTE, tmp_path / "p.zip").size == len(PAYLOAD)


def test_401_refreshes_token_then_fails_after_two_refreshes(make, tmp_path):
    dl, stream, _ = make([resp(401), resp(200, PAYLOAD)])
    dl.fetch(REMOTE, tmp_path / "a.zip")
    assert [r["Authorization"] for r in stream.requests] == ["Bearer SECRET-TOKEN-1", "Bearer SECRET-TOKEN-2"]
    dl, _, _ = make([resp(401), resp(401), resp(401)])
    with pytest.raises(AuthenticationError, match="HTTP 401") as info:
        dl.fetch(REMOTE, tmp_path / "b.zip")
    assert "SECRET" not in str(info.value)


@pytest.mark.parametrize(("status", "message"),
                         [(403, "access denied"), (404, "not found"), (418, "unexpected HTTP 418")])
def test_non_retryable_statuses_fail_immediately(make, tmp_path, status, message):
    dl, stream, _ = make([resp(status)])
    with pytest.raises(DownloadError, match=message):
        dl.fetch(REMOTE, tmp_path / "p.zip")
    assert len(stream.requests) == 1


def test_checksum_mismatch_never_leaves_a_file(make, tmp_path, cfg):
    corrupt = bytes(len(PAYLOAD))
    dl, _, _ = make([resp(200, corrupt) for _ in range(cfg.settings.download.max_attempts)])
    with pytest.raises(DownloadError, match="md5 mismatch"):
        dl.fetch(REMOTE, tmp_path / "p.zip")
    assert not (tmp_path / "p.zip").exists() and not (tmp_path / "p.zip.part").exists()


def test_oversized_body_is_discarded(make, tmp_path):
    dl, _, _ = make([resp(200, PAYLOAD + b"extra"), resp(200, PAYLOAD)])
    assert dl.fetch(REMOTE, tmp_path / "p.zip").attempts == 2


def test_existing_valid_file_is_skipped_without_request(make, tmp_path):
    (tmp_path / "p.zip").write_bytes(PAYLOAD)
    dl, stream, _ = make([])
    result = dl.fetch(REMOTE, tmp_path / "p.zip")
    assert result.skipped_existing and result.verified_with == "md5" and stream.requests == []


def test_existing_mismatching_file_is_never_overwritten(make, tmp_path):
    (tmp_path / "p.zip").write_bytes(b"x" * len(PAYLOAD))
    dl, _, _ = make([])
    with pytest.raises(DownloadError, match="move it away"):
        dl.fetch(REMOTE, tmp_path / "p.zip")
    assert (tmp_path / "p.zip").read_bytes() == b"x" * len(PAYLOAD)


def test_insufficient_disk_space_fails_before_download(make, tmp_path):
    dl, stream, _ = make([resp(200, PAYLOAD)], free=1000)
    with pytest.raises(DownloadError, match="not enough disk space"):
        dl.fetch(REMOTE, tmp_path / "p.zip")
    assert stream.requests == []


def test_refused_redirect_is_not_retried(make, tmp_path):
    dl, stream, _ = make([RedirectRefusedError("refusing redirect to 'evil.example'")])
    with pytest.raises(DownloadError, match="evil.example"):
        dl.fetch(REMOTE, tmp_path / "p.zip")
    assert len(stream.requests) == 1


def test_unverifiable_checksum_still_checks_size(make, tmp_path):
    remote = RemoteFile(url=URL, label="X", expected_size=len(PAYLOAD), checksums={"blake3": "00" * 32})
    dl, _, _ = make([resp(200, PAYLOAD)])
    result = dl.fetch(remote, tmp_path / "p.zip")
    assert result.verified_with in (None, "blake3") and result.size == len(PAYLOAD)


def test_verify_hashes_reports_mismatch():
    with pytest.raises(ChecksumMismatchError):
        verify_hashes({"md5": "aa"}, {"md5": "bb"})
    assert verify_hashes({"sha256": "aa", "md5": "bb"}, {"md5": "BB"}) == "md5"


# ---- redirect allow-list --------------------------------------------------------------

@pytest.mark.parametrize(("url", "ok"), [
    ("https://zipper.dataspace.copernicus.eu/x", True),
    ("https://dataspace.copernicus.eu/x", True),
    ("http://zipper.dataspace.copernicus.eu/x", False),            # not https
    ("https://dataspace.copernicus.eu.evil.example/x", False),
    ("https://evildataspace.copernicus.eu/x", False),
    ("https://s3.amazonaws.com/x", False),
])
def test_host_allow_list(url, ok):
    assert host_allowed(url, ("dataspace.copernicus.eu",)) is ok


def test_redirect_handler_keeps_token_only_for_allowed_hosts():
    handler = _AllowListRedirectHandler(("dataspace.copernicus.eu",))
    request = urllib.request.Request(URL, headers={"Authorization": "Bearer T"})
    new = handler.redirect_request(request, None, 302, "Found", {}, "https://zipper.dataspace.copernicus.eu/p")
    assert new is not None and new.get_header("Authorization") == "Bearer T"
    with pytest.raises(RedirectRefusedError):
        handler.redirect_request(request, None, 302, "Found", {}, "https://elsewhere.example/p")

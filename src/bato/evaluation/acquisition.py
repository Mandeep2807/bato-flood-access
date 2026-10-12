"""EVALUATION ONLY. Acquisition of the EMSR927 reference products, with provenance.

* The official activation record is saved byte-for-byte with its retrieval time and SHA-256.
* Download URLs come from that record (``downloadPath``), never from a filename pattern.
* Only final products whose EMS AOI overlaps the BATO AOI are downloaded.
* Everything is written under ``<paths.reference>/EMSR927/``; every path is checked with
  :func:`bato.data_rules.assert_reference_input_allowed`.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from bato.attribution import CREDIT_COPERNICUS_EMS
from bato.config import PathSettings
from bato.data_rules import assert_reference_input_allowed
from bato.evaluation.ems_activation import (
    ActivationRecord,
    aoi_overlaps,
    check_download_url,
    parse_activation,
    plan_downloads,
)

CODE = "EMSR927"
MANIFEST = "acquisition_manifest.json"
USER_AGENT = "BATO-evaluation/0.1 (reference-data download for a case-study comparison)"
Opener = Callable[[str], BinaryIO]


class AcquisitionError(RuntimeError):
    pass


def reference_dir(paths: PathSettings) -> Path:
    d = paths.reference / CODE
    assert_reference_input_allowed(d, paths)
    return d


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# Download trust rule (see TrustedRedirects):
# * every fetch starts at ORIGIN_HOST, over HTTPS;
# * a redirect may go to ORIGIN_HOST, or to STORAGE_HOST only when the fetch started at an official EMSR927 product
#   zip URL (an activation-record downloadPath);
# * every hop is HTTPS on port 443, with no user-info, and with an exact host name match; at most MAX_REDIRECTS hops.
ORIGIN_HOST = "rapidmapping.emergency.copernicus.eu"
STORAGE_HOST = "rapidmapping.s3.amazonaws.com"
MAX_REDIRECTS = 5
ALLOWED_HOST = ORIGIN_HOST            # kept for callers that name the origin host


def _https_host(url: str) -> str | None:
    """The exact host of an HTTPS URL on the default port without user-info, else None."""
    u = urllib.parse.urlsplit(url)
    try:
        port = u.port
    except ValueError:
        return None
    if u.scheme != "https" or u.username is not None or u.password is not None or port not in (None, 443):
        return None
    return u.hostname


def is_official_product_url(url: str) -> bool:
    """An official EMSR927 product zip on the Rapid Mapping host (the shape of an activation-record downloadPath)."""
    if _https_host(url) != ORIGIN_HOST:
        return False
    try:
        check_download_url(url, ORIGIN_HOST, CODE)
    except ValueError:
        return False
    return True


class TrustedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows only the redirects allowed by the trust rule above, and records each hop."""

    max_redirections = MAX_REDIRECTS

    def __init__(self, origin: str) -> None:
        super().__init__()
        self.origin = origin
        self.allowed = {ORIGIN_HOST} | ({STORAGE_HOST} if is_official_product_url(origin) else set())
        self.chain: list[dict[str, Any]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if _https_host(newurl) not in self.allowed:
            raise AcquisitionError(f"refusing redirect from {req.full_url} to {newurl}: allowed targets from "
                                   f"{self.origin} are https://{' or https://'.join(sorted(self.allowed))}")
        self.chain.append({"status": int(code), "from": req.full_url, "to": newurl})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_open(url: str, *, extra_handlers: tuple[urllib.request.BaseHandler, ...] = ()) -> BinaryIO:
    """Open ``url`` under the trust rule. The response carries ``bato_final_url`` and ``bato_redirect_chain``.
    ``extra_handlers`` exists for tests (a fake HTTPS transport); production passes none."""
    if _https_host(url) != ORIGIN_HOST:
        raise AcquisitionError(f"refusing to fetch {url}: downloads must start at https://{ORIGIN_HOST}/")
    redirects = TrustedRedirects(url)
    opener = urllib.request.build_opener(redirects, *extra_handlers)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    resp = opener.open(req, timeout=120)
    final = resp.geturl()
    if _https_host(final) not in redirects.allowed:                       # defence in depth
        resp.close()
        raise AcquisitionError(f"{url} ended at an untrusted location {final}")
    resp.bato_final_url = final
    resp.bato_redirect_chain = list(redirects.chain)
    out: BinaryIO = resp
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_to(opener: Opener, url: str, dest: Path) -> tuple[int, str, dict[str, Any]]:
    """Stream to ``dest`` (via ``.part``). Returns bytes, SHA-256 and the transfer record (original URL, final URL,
    redirect chain)."""
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    n = 0
    with opener(url) as src, tmp.open("wb") as out:
        transfer = {"original_url": url, "final_url": getattr(src, "bato_final_url", url),
                    "redirect_chain": list(getattr(src, "bato_redirect_chain", []))}
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
            n += len(chunk)
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise AcquisitionError(f"{url} returned an empty body")
    os.replace(tmp, dest)
    return n, h.hexdigest(), transfer


def save_activation(paths: PathSettings, opener: Opener, api_url: str, *, now: Callable[[], datetime] = utc_now,
                    local_copy: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Retrieve (or, with ``local_copy``, register a browser-saved copy of) the activation record, verbatim."""
    d = reference_dir(paths) / "activation"
    d.mkdir(parents=True, exist_ok=True)
    stamp = now()
    dest = d / f"{CODE}_activation_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    assert_reference_input_allowed(dest, paths)
    transfer: dict[str, Any] = {"original_url": api_url, "final_url": None, "redirect_chain": []}
    if local_copy is not None:
        data = local_copy.read_bytes()
        dest.write_bytes(data)
        how = f"copied from a locally saved file ({local_copy.name}); retrieval time = when it was registered"
    else:
        _n, _digest, transfer = _stream_to(opener, api_url, dest)
        how = "downloaded by scripts/fetch_emsr927.py"
    json.loads(dest.read_text(encoding="utf-8"))           # must be JSON
    return dest, {"url": api_url, "retrieved_at_utc": stamp.isoformat(), "how": how, **transfer,
                  "file": dest.name, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}


def record_summary(rec: ActivationRecord) -> dict[str, Any]:
    return {"code": rec.code, "name": rec.name, "event_time_raw": rec.event_time,
            "activation_time_raw": rec.activation_time, "closed": rec.closed, "countries": list(rec.countries),
            "time_zone_marker_in_record": "none (times are given without Z or offset)"
            if all(t is None or not (t.endswith("Z") or "+" in t[10:]) for t in (rec.event_time, rec.activation_time))
            else "present", "top_level_keys": list(rec.top_level_keys),
            "aois": [{"number": a.number, "name": a.name, "extent_wkt_prefix": (a.extent_wkt or "")[:80],
                      "products": [p.to_json() for p in a.products]} for a in rec.aois]}


def acquire(paths: PathSettings, bato_aoi: dict[str, Any], protocol_reference: dict[str, Any], *,
            opener: Opener = http_open, now: Callable[[], datetime] = utc_now, local_activation: Path | None = None,
            dry_run: bool = False) -> dict[str, Any]:
    """Save the record, decide, download. Returns (and writes) the acquisition manifest."""
    base = reference_dir(paths)
    act_path, act_meta = save_activation(paths, opener, protocol_reference["activation_api"], now=now,
                                         local_copy=local_activation)
    record = parse_activation(json.loads(act_path.read_text(encoding="utf-8")), CODE)
    ring = [(float(x), float(y)) for x, y in bato_aoi["aoi_ring"]]
    overlaps = aoi_overlaps(record, ring)
    decisions = plan_downloads(record, overlaps, required_status=protocol_reference["required_status_code"],
                               allowed_host=protocol_reference["allowed_download_host"])
    prod_dir = base / "products"
    downloads, skipped = [], []
    old = load_manifest(paths, missing_ok=True)
    old_by_url = {d["url"]: d for d in (old or {}).get("downloads", [])}
    for dec in decisions:
        p = dec.product
        if not dec.download:
            skipped.append({"label": p.label, "aoi": p.aoi_number, "reason": dec.reason,
                            "url": p.download_path or None})
            continue
        name = check_download_url(p.download_path, protocol_reference["allowed_download_host"], CODE)
        dest = prod_dir / name
        assert_reference_input_allowed(dest, paths)
        entry = {"label": p.label, "aoi": p.aoi_number, "aoi_name": p.aoi_name, "type": p.type,
                 "monitoring": p.monitoring, "monitoring_number": p.monitoring_number, "version": p.version,
                 "status_code": p.status_code, "delivery_time_raw": p.delivery_time,
                 "images": [{"sensor": im.sensor, "acquisition_time_raw": im.acquisition_time} for im in p.images],
                 "url": p.download_path, "url_source": "activation record downloadPath", "file": f"products/{name}",
                 "reason": dec.reason}
        if dry_run:
            entry["downloaded"] = False
        elif dest.is_file() and p.download_path in old_by_url and \
                sha256_file(dest) == old_by_url[p.download_path].get("sha256"):
            prev = old_by_url[p.download_path]
            entry.update({"downloaded": True, "downloaded_at_utc": prev["downloaded_at_utc"], "bytes": prev["bytes"],
                          "sha256": prev["sha256"], "final_url": prev.get("final_url"),
                          "redirect_chain": prev.get("redirect_chain", []),
                          "note": "already present with the recorded SHA-256; not re-downloaded"})
        else:
            prod_dir.mkdir(parents=True, exist_ok=True)
            t = now()
            n, digest, transfer = _stream_to(opener, p.download_path, dest)
            entry.update({"downloaded": True, "downloaded_at_utc": t.isoformat(), "bytes": n, "sha256": digest,
                          "final_url": transfer["final_url"], "redirect_chain": transfer["redirect_chain"]})
        downloads.append(entry)
    manifest = {"kind": "emsr927_reference_acquisition", "data_role": "evaluation_reference_only",
                "warning": "REFERENCE DATA. Never a pipeline input, training label, threshold source, feature, "
                           "connectivity input or dashboard input.", "attribution": CREDIT_COPERNICUS_EMS,
                "activation": act_meta, "record": record_summary(record), "bato_aoi": bato_aoi,
                "aoi_overlaps_from_record_extent": [o.__dict__ for o in overlaps],
                "downloads": downloads, "not_downloaded": skipped, "dry_run": dry_run}
    if not dry_run:
        out = base / MANIFEST
        assert_reference_input_allowed(out, paths)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(paths: PathSettings, missing_ok: bool = False) -> dict[str, Any] | None:
    p = reference_dir(paths) / MANIFEST
    if not p.is_file():
        if missing_ok:
            return None
        raise AcquisitionError(f"{p} not found; run scripts/fetch_emsr927.py first")
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return data


def verify_downloads(paths: PathSettings, manifest: dict[str, Any]) -> list[Path]:
    """Every downloaded zip must still match its recorded SHA-256."""
    base = reference_dir(paths)
    out = []
    act = base / "activation" / manifest["activation"]["file"]
    if sha256_file(act) != manifest["activation"]["sha256"]:
        raise AcquisitionError(f"{act.name} changed since retrieval")
    for d in manifest["downloads"]:
        if not d.get("downloaded"):
            continue
        p = base / d["file"]
        assert_reference_input_allowed(p, paths)
        if not p.is_file() or sha256_file(p) != d["sha256"]:
            raise AcquisitionError(f"{d['file']} is missing or does not match its recorded SHA-256")
        out.append(p)
    return out

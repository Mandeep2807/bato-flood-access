"""Pre-event OpenStreetMap extraction via the ohsome API v2 (historical OSM, GeoParquet, API key).

The PS allows OSM only "as it was before the event". The snapshot date is derived per run:
``event_date - osm.snapshot_offset_days`` (>= 1 day), capped at the latest date the ohsome database
covers (``GET /metadata`` -> ``temporalExtent.end``; :func:`bato.contracts.osm.select_snapshot_date`).
Every request asks for the OSM state at ``<snapshot>T00:00:00Z``, so no edit made on or after the event
day can enter.

API (https://docs.ohsome.org/ohsome-api/v2/reference/api.html, OpenAPI at ``<base>/openapi.json``):
``POST <osm.api_base_url>/extraction/features.parquet`` with JSON body
``{"filter": ..., "aoi": [xmin, ymin, xmax, ymax], "time": "<snapshot>T00:00:00Z", "clip": true}`` and the
API key in the ``authorization`` header (raw key, read from the environment variable named by
``osm.api_key_env``; never stored, logged or written anywhere). The response is GeoParquet with the
columns ``osm_type``, ``osm_id``, ``edit_timestamp``, ``valid_to_timestamp``, ``version``, ``tags`` (map),
``geom`` (WKB), ``geom_type``, ``clipped`` (docs: Data Model (Extraction)).

Layers (same filters and boxes as before; one request each):

========== ====================================================== ============== =======================
layer      filter                                                 box            post-processing
========== ====================================================== ============== =======================
roads      ``type:way and highway in (<osm.road_highway_values>)`` analysis box   geometry as is
places     ``place in (<osm.settlement_place_values>)``           analysis box   client-side centroid
hospitals  ``amenity=hospital or healthcare=hospital``            analysis box   client-side centroid
buildings  ``building=* and building!=no and geometry:polygon``   AOI + margin   client-side centroid
========== ====================================================== ============== =======================

v2 has no centroid endpoint (v1 ``/elements/centroid``); centroids are computed here
(:mod:`bato.osm.geometry`). Geometries are clipped to the box (``clip: true``), so a road leaving the
analysis box ends exactly on its edge: those ends are the graph's exit nodes (bato.impact.roads).

Pre-event guards on every feature: ``edit_timestamp`` (when this version was uploaded) must be at or
before the snapshot instant, and ``valid_to_timestamp`` after it (the version was the live one at the
snapshot). Anything else is refused as post-event OSM.

The GeoParquet responses are stored byte-for-byte under ``<paths.raw>/osm/<aoi_id>/snapshot-<date>/``
with ``osm_extract_provenance.json`` (request bodies minus the key, SHA-256, feature counts, ohsome
metadata), published atomically. :func:`load_osm_extract` re-verifies all of it and converts each layer
into the feature dicts the road/settlement code reads (``geometry`` GeoJSON + ``properties`` with
``@osmId`` = ``<type>/<id>``, ``@lastEdit`` and the OSM tags). Reading GeoParquet needs ``pyarrow``.

Nothing here touches population data, damage maps or post-event OSM.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import bato
from bato.acquisition.dem import dem_search_bbox
from bato.acquisition.http import HttpResponse, TransportError
from bato.config import BatoConfig
from bato.contracts.common import BBox
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.osm import (
    OSMExtractProvenance,
    OSMLayerName,
    OSMLayerRecord,
    OSMSnapshot,
    select_snapshot_date,
)
from bato.data_rules import assert_pipeline_input_allowed, assert_raw_output_dir
from bato.exceptions import BatoError
from bato.osm.geometry import WKBError, centroid, wkb_to_geojson
from bato.preprocessing.speckle import sha256_file

ATTRIBUTION = "OpenStreetMap contributors (ODbL 1.0), historical data via the ohsome API v2 (HeiGIT)"
SERVICE = "ohsome-api-v2"
ENDPOINT = "extraction/features.parquet"
PROVENANCE_FILE = "osm_extract_provenance.json"
HOSPITAL_FILTER = "amenity=hospital or healthcare=hospital"
BUILDING_FILTER = "building=* and building!=no and geometry:polygon"
REQUIRED_COLUMNS = ("osm_type", "osm_id", "edit_timestamp", "valid_to_timestamp", "tags", "geom")
USER_AGENT = f"bato/{bato.__version__} (hackathon prototype)"


class OSMError(BatoError):
    """The OSM extract could not be fetched, stored or verified (fail closed)."""


# ---------------------------------------------------------------------------
# Transport (JSON POST with the API key header; no key ever appears in errors)
# ---------------------------------------------------------------------------


class OhsomeTransport(Protocol):
    def get(self, url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse: ...

    def post_json(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float
                  ) -> HttpResponse: ...


class UrllibOhsomeTransport:
    """Production transport (urllib; honours HTTPS_PROXY / SSL_CERT_FILE)."""

    def get(self, url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json", **headers})
        return self._send(req, timeout_s)

    def post_json(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float
                  ) -> HttpResponse:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"User-Agent": USER_AGENT, "Content-Type": "application/json",
                                              **headers})
        return self._send(req, timeout_s)

    @staticmethod
    def _send(req: urllib.request.Request, timeout_s: float) -> HttpResponse:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310 (https URL from config)
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read() or b"")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            host = urllib.parse.urlsplit(req.full_url).netloc
            raise TransportError(f"request to {host} failed: {exc}") from exc


def api_key_from_env(cfg: BatoConfig, environ: dict[str, str] | None = None) -> str:
    name = cfg.settings.osm.api_key_env
    key = (environ if environ is not None else dict(os.environ)).get(name, "").strip()
    if not key:
        raise OSMError(f"the ohsome API v2 needs an API key: set the environment variable {name} "
                       "(free key: https://account.heigit.org/signup)")
    return key


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRequest:
    layer: OSMLayerName
    filter: str
    bbox: BBox
    centroids: bool                          # replace geometries by client-side centroids
    clip: bool = True
    endpoint: str = ENDPOINT

    @property
    def file_name(self) -> str:
        return f"{self.layer.value}.parquet"

    def body(self, snapshot_date: date) -> dict[str, Any]:
        b = self.bbox
        return {"filter": self.filter, "aoi": [b.min_lon, b.min_lat, b.max_lon, b.max_lat],
                "time": snapshot_time(snapshot_date), "clip": self.clip}


def snapshot_time(snapshot_date: date) -> str:
    return f"{snapshot_date.isoformat()}T00:00:00Z"


def bbox_param(b: BBox) -> str:
    return f"{b.min_lon:.7f},{b.min_lat:.7f},{b.max_lon:.7f},{b.max_lat:.7f}"


def settlement_bbox(aoi_bbox: BBox, margin_km: float) -> BBox:
    return dem_search_bbox(aoi_bbox, margin_km) if margin_km > 0 else aoi_bbox


def plan_requests(cfg: BatoConfig, aoi_bbox: BBox, analysis_bbox: BBox) -> tuple[LayerRequest, ...]:
    osm = cfg.settings.osm
    return (
        LayerRequest(OSMLayerName.ROADS, f"type:way and highway in ({', '.join(osm.road_highway_values)})",
                     analysis_bbox, centroids=False),
        LayerRequest(OSMLayerName.PLACES, f"place in ({', '.join(osm.settlement_place_values)})", analysis_bbox,
                     centroids=True),
        LayerRequest(OSMLayerName.HOSPITALS, HOSPITAL_FILTER, analysis_bbox, centroids=True),
        LayerRequest(OSMLayerName.BUILDINGS, BUILDING_FILTER, settlement_bbox(aoi_bbox, osm.building_margin_km),
                     centroids=True),
    )


def _error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                if data.get(key):
                    return str(data[key])[:300]
    except json.JSONDecodeError:
        pass
    return text[:300]


def _http_error(what: str, response: HttpResponse, key_env: str) -> OSMError:
    hint = (f" (check the API key in {key_env})" if response.status in (401, 403) else
            " (rate limited; retry later)" if response.status == 429 else "")
    return OSMError(f"ohsome {what} returned HTTP {response.status}: {_error_message(response.body)}{hint}")


def fetch_metadata(transport: OhsomeTransport, base_url: str, api_key: str, key_env: str, timeout_s: float
                   ) -> tuple[date, str | None]:
    """Latest date covered by the ohsome database (``temporalExtent.end``) and the API version."""
    try:
        response = transport.get(f"{base_url.rstrip('/')}/metadata", {"authorization": api_key}, timeout_s)
    except TransportError as exc:
        raise OSMError(f"ohsome metadata request failed: {exc}") from exc
    if response.status != 200:
        raise _http_error("metadata", response, key_env)
    try:
        data = json.loads(response.body)
        latest = date.fromisoformat(str(data["temporalExtent"]["end"])[:10])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise OSMError(f"ohsome metadata has no readable temporalExtent.end: {exc}") from exc
    version = data.get("apiVersion")
    return latest, str(version) if version is not None else None


# ---------------------------------------------------------------------------
# GeoParquet -> verified feature dicts
# ---------------------------------------------------------------------------


def require_pyarrow() -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise OSMError("reading the ohsome GeoParquet needs pyarrow: pip install pyarrow") from exc


def read_parquet_rows(data: bytes) -> list[dict[str, Any]]:
    """Rows of an ohsome features GeoParquet (pyarrow), as Python values."""
    require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(pa.BufferReader(data))
    except Exception as exc:  # pyarrow raises several error types for corrupt input
        raise OSMError(f"response is not a readable Parquet file: {exc}") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise OSMError(f"GeoParquet lacks the documented columns {missing}")
    rows: list[dict[str, Any]] = table.select(list(REQUIRED_COLUMNS)).to_pylist()
    return rows


def _as_utc(value: Any, what: str) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        v = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
    raise OSMError(f"{what} is not a timestamp: {value!r}")


def _tags(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    items = value.items() if isinstance(value, dict) else value
    return {str(k): str(v) for k, v in items}


def rows_to_features(rows: list[dict[str, Any]], snapshot_date: date, centroids: bool, where: str
                     ) -> list[dict[str, Any]]:
    """Verify the pre-event guards and build the feature dicts the pipeline reads."""
    cutoff = datetime.combine(snapshot_date, datetime.min.time(), tzinfo=timezone.utc)
    out = []
    for i, row in enumerate(rows):
        osm_type, osm_id = row.get("osm_type"), row.get("osm_id")
        if osm_type not in ("node", "way", "relation") or not isinstance(osm_id, int) or isinstance(osm_id, bool):
            raise OSMError(f"{where}: row {i} has no valid osm_type/osm_id")
        ref = f"{osm_type}/{osm_id}"
        edited = _as_utc(row.get("edit_timestamp"), f"{where} {ref} edit_timestamp")
        valid_to = _as_utc(row.get("valid_to_timestamp"), f"{where} {ref} valid_to_timestamp")
        if edited > cutoff:
            raise OSMError(f"{where}: {ref} was edited {edited.isoformat()}, after the pre-event snapshot "
                           f"{cutoff.isoformat()} (post-event OSM is prohibited)")
        if valid_to <= cutoff:
            raise OSMError(f"{where}: {ref} version stopped being valid {valid_to.isoformat()}, before the snapshot "
                           f"{cutoff.isoformat()}: not the OSM state at the snapshot")
        geom = row.get("geom")
        if not isinstance(geom, bytes | bytearray):
            raise OSMError(f"{where}: {ref} has no WKB geometry")
        try:
            geometry = wkb_to_geojson(bytes(geom))
            if centroids:
                lon, lat = centroid(geometry)
                geometry = {"type": "Point", "coordinates": [lon, lat]}
        except WKBError as exc:
            raise OSMError(f"{where}: {ref} geometry: {exc}") from exc
        props: dict[str, Any] = {"@osmId": ref, "@lastEdit": edited.isoformat().replace("+00:00", "Z")}
        props.update(_tags(row.get("tags")))
        out.append({"type": "Feature", "geometry": geometry, "properties": props})
    return out


def fetch_layer(transport: OhsomeTransport, base_url: str, api_key: str, key_env: str, request: LayerRequest,
                snapshot_date: date, timeout_s: float) -> tuple[bytes, int]:
    url = f"{base_url.rstrip('/')}/{request.endpoint}"
    try:
        response = transport.post_json(url, request.body(snapshot_date), {"authorization": api_key}, timeout_s)
    except TransportError as exc:
        raise OSMError(f"ohsome {request.layer.value} request failed: {exc}") from exc
    if response.status != 200:
        raise _http_error(f"{request.layer.value} request", response, key_env)
    rows = read_parquet_rows(response.body)
    rows_to_features(rows, snapshot_date, request.centroids, f"ohsome {request.layer.value}")
    return response.body, len(rows)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def extract_dir(cfg: BatoConfig, aoi_id: str, snapshot_date: date) -> Path:
    if not aoi_id or any(c in aoi_id for c in "/\\:") or aoi_id in (".", ".."):
        raise OSMError(f"invalid AOI id {aoi_id!r}")
    path = cfg.settings.paths.raw / cfg.settings.osm.output_subdir / aoi_id / f"snapshot-{snapshot_date.isoformat()}"
    return assert_raw_output_dir(path, cfg.settings.paths)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise OSMError(f"{path} is outside the project root {root}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def extract_osm(cfg: BatoConfig, *, aoi_id: str, event_date: date, aoi_bbox: BBox, analysis_bbox: BBox,
                transport: OhsomeTransport, api_key: str, overwrite: bool = False,
                now: Callable[[], datetime] = _utc_now,
                progress: Callable[[str], None] | None = None) -> tuple[Path, OSMExtractProvenance]:
    """Fetch every layer at the pre-event snapshot and publish the extract atomically."""
    say = progress or (lambda _t: None)
    osm = cfg.settings.osm
    if not api_key:
        raise OSMError(f"no ohsome API key (set {osm.api_key_env})")
    root = cfg.config_dir.parent
    latest, api_version = fetch_metadata(transport, osm.api_base_url, api_key, osm.api_key_env, osm.timeout_s)
    snapshot_date = select_snapshot_date(event_date, osm.snapshot_offset_days, latest)
    snapshot = OSMSnapshot(event_date=event_date, snapshot_date=snapshot_date, latest_available=latest,
                           service=SERVICE)
    final_dir = extract_dir(cfg, aoi_id, snapshot_date)
    if final_dir.exists() and not overwrite:
        raise OSMError(f"OSM extract already exists: {final_dir} (use --overwrite; the old one is kept as "
                       "<dir>.superseded-<time>)")
    started = now()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    tmp = assert_raw_output_dir(final_dir.with_name(f"{final_dir.name}.tmp-{stamp}"), cfg.settings.paths)
    tmp.mkdir(parents=True)
    records = []
    total = 0
    for request in plan_requests(cfg, aoi_bbox, analysis_bbox):
        say(f"  {request.layer.value:9} POST {request.endpoint} time={snapshot_time(snapshot_date)} "
            f"filter={request.filter}\n")
        body, count = fetch_layer(transport, osm.api_base_url, api_key, osm.api_key_env, request, snapshot_date,
                                  osm.timeout_s)
        path = tmp / request.file_name
        path.write_bytes(body)
        total += count
        records.append(OSMLayerRecord(
            layer=request.layer, endpoint=request.endpoint, filter=request.filter, bbox=request.bbox,
            time=snapshot_time(snapshot_date), clip=request.clip,
            post_processing="client-side centroid" if request.centroids else None,
            file=_relative(final_dir / request.file_name, root), sha256=sha256_file(path), feature_count=count))
        say(f"            {count:,} features\n")
    provenance = OSMExtractProvenance(
        bato_version=bato.__version__, aoi_id=aoi_id, aoi_bbox=aoi_bbox, analysis_bbox=analysis_bbox,
        settlement_bbox=settlement_bbox(aoi_bbox, osm.building_margin_km),
        snapshot=snapshot.model_copy(update={"retrieved_feature_count": total}),
        api_base_url=osm.api_base_url, api_version=api_version, attribution=ATTRIBUTION, retrieved_at=now(),
        layers=tuple(records))
    save_contract(provenance, tmp / PROVENANCE_FILE)
    if final_dir.exists():
        os.replace(final_dir, final_dir.with_name(f"{final_dir.name}.superseded-{stamp}"))
    os.replace(tmp, final_dir)
    return final_dir, provenance


@dataclass(frozen=True)
class OSMExtract:
    directory: Path
    provenance_path: Path
    provenance_sha256: str
    provenance: OSMExtractProvenance
    layers: dict[OSMLayerName, list[dict[str, Any]]]


def load_osm_extract(cfg: BatoConfig, aoi_id: str, event_date: date) -> OSMExtract:
    """The verified pre-event extract for this AOI and event, converted to feature dicts (fail closed)."""
    root = cfg.config_dir.parent
    base = cfg.settings.paths.raw / cfg.settings.osm.output_subdir / aoi_id
    base = assert_pipeline_input_allowed(base, cfg.settings.paths)
    found = sorted(p for p in base.glob("snapshot-*") if p.is_dir() and "." not in p.name) if base.is_dir() else []
    matching: list[tuple[Path, OSMExtractProvenance]] = []
    for directory in found:
        path = assert_pipeline_input_allowed(directory / PROVENANCE_FILE, cfg.settings.paths)
        try:
            candidate = load_contract(OSMExtractProvenance, path)
        except ContractIOError as exc:
            raise OSMError(str(exc)) from exc
        if candidate.snapshot.event_date == event_date:
            matching.append((directory, candidate))
    if not matching:
        raise OSMError(f"no pre-event OSM extract for event {event_date} under "
                       f"{_relative(base, root) if base.exists() else base} (run scripts/fetch_osm_snapshot.py first)")
    directory, prov = matching[-1]                     # the latest snapshot made for this event
    provenance_path = directory / PROVENANCE_FILE
    if prov.aoi_id != aoi_id:
        raise OSMError(f"{provenance_path.name} is for AOI {prov.aoi_id}, not {aoi_id}")
    if prov.snapshot.event_date != event_date or prov.snapshot.snapshot_date >= event_date:
        raise OSMError(f"OSM extract snapshot {prov.snapshot.snapshot_date} (event {prov.snapshot.event_date}) is "
                       f"not a pre-event snapshot for event {event_date}")
    layers: dict[OSMLayerName, list[dict[str, Any]]] = {}
    for record in prov.layers:
        path = assert_pipeline_input_allowed(root / record.file, cfg.settings.paths)
        if not path.is_file():
            raise OSMError(f"{record.file} is missing")
        if sha256_file(path) != record.sha256:
            raise OSMError(f"{record.file} changed since extraction (SHA-256 differs from its provenance)")
        if record.time != snapshot_time(prov.snapshot.snapshot_date):
            raise OSMError(f"{record.file} was requested for {record.time}, not the snapshot instant")
        rows = read_parquet_rows(path.read_bytes())
        if len(rows) != record.feature_count:
            raise OSMError(f"{record.file} has {len(rows)} features, provenance says {record.feature_count}")
        layers[record.layer] = rows_to_features(rows, prov.snapshot.snapshot_date,
                                                record.post_processing is not None, record.file)
    return OSMExtract(directory=directory, provenance_path=provenance_path,
                      provenance_sha256=sha256_file(provenance_path), provenance=prov, layers=layers)

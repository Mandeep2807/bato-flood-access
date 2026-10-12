"""Copernicus Data Space Ecosystem (CDSE) OData catalogue: query + record parsing.

Server-side filtering uses only operators documented by CDSE: collection
name, sensing-start window (``ContentDate/Start``) and AOI intersection
(``OData.CSC.Intersects``). Everything else (mode, product type,
polarisation, orbit, coverage, same-day rule) is filtered client-side in
:mod:`bato.acquisition.s1_catalogue`, so every excluded scene stays visible
with a reason.

Record parsing is strict: required metadata that is missing or inconsistent
raises :class:`~bato.exceptions.MetadataError`. Nothing is guessed. The
official product name is used to cross-check attributes and to supply
platform/mode/product type/polarisation only when the attribute is absent,
because those values are encoded in the name by ESA convention. Orbit
direction and relative orbit are not in the name and are always required
from the catalogue.

ASSUMPTION (OPEN until the first live query): attribute VALUE formats
(e.g. ``polarisationChannels`` as "VV&VH" vs "VV VH") are parsed
tolerantly and cross-checked against the name; see docs/catalogue_discovery.md.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bato.acquisition.http import Transport, TransportError, UrllibTransport
from bato.acquisition.product_name import S1ProductName, parse_s1_product_name
from bato.config import CDSESettings
from bato.contracts.common import Polarization, PolygonGeometry
from bato.exceptions import CatalogueQueryError, MetadataError

PROVIDER = "cdse-odata"
COLLECTION_S1 = "SENTINEL-1"
_NAME_TIME_TOLERANCE_S = 1.0  # names carry whole seconds; ContentDate carries fractions

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _odata_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _ccw(ring: tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
    """Return the ring counter-clockwise (shoelace sign), as WKT/OData expect."""
    points = list(ring)
    signed = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points[:-1], points[1:], strict=True))
    return points if signed > 0 else points[::-1]


def build_s1_filter(aoi_polygon: PolygonGeometry, start: datetime, end: datetime) -> str:
    """OData ``$filter`` for Sentinel-1 products sensed in [start, end) intersecting the AOI."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")
    coords = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in _ccw(aoi_polygon.exterior))
    return (
        f"Collection/Name eq '{COLLECTION_S1}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({coords}))') "
        f"and ContentDate/Start ge {_odata_time(start)} "
        f"and ContentDate/Start lt {_odata_time(end)}"
    )


@dataclass(frozen=True)
class FetchResult:
    records: list[dict[str, Any]]
    truncated: bool       # True if max_records was reached and more records may exist
    query_filter: str
    pages: int


class CDSEODataClient:
    """Fetch raw catalogue records. Knows nothing about BATO filtering rules."""

    def __init__(self, settings: CDSESettings, transport: Transport | None = None) -> None:
        self.settings = settings
        self.transport = transport or UrllibTransport()

    def _first_page_url(self, query_filter: str) -> str:
        params = {
            "$filter": query_filter,
            "$orderby": "ContentDate/Start asc",
            "$top": str(self.settings.page_size),
            "$expand": "Attributes",
        }
        return f"{self.settings.odata_url}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"

    def fetch_s1(self, aoi_polygon: PolygonGeometry, start: datetime, end: datetime) -> FetchResult:
        """Fetch all Sentinel-1 records for the AOI and window (paginated).

        Raises:
            CatalogueQueryError: network failure, non-200 status, or a response
                that is not an OData product list.
        """
        return self.fetch_filter(build_s1_filter(aoi_polygon, start, end))

    def fetch_filter(self, query_filter: str) -> FetchResult:
        """Fetch all records matching an OData ``$filter`` (paginated, capped)."""
        url: str | None = self._first_page_url(query_filter)
        records: list[dict[str, Any]] = []
        pages = 0
        skip = 0
        truncated = False
        while url is not None:
            page = self._get_page(url)
            pages += 1
            items = page["value"]
            records.extend(items)
            if len(records) >= self.settings.max_records:
                more = bool(page.get("@odata.nextLink")) or len(items) == self.settings.page_size
                truncated = more
                records = records[: self.settings.max_records]
                break
            next_link = page.get("@odata.nextLink")
            if isinstance(next_link, str) and next_link:
                url = next_link
            elif len(items) == self.settings.page_size:
                skip += self.settings.page_size
                url = f"{self._first_page_url(query_filter)}&%24skip={skip}"
            else:
                url = None
        return FetchResult(records=records, truncated=truncated, query_filter=query_filter, pages=pages)

    def _get_page(self, url: str) -> dict[str, Any]:
        try:
            response = self.transport.get(url, self.settings.timeout_s)
        except TransportError as exc:
            raise CatalogueQueryError(f"CDSE catalogue unreachable: {exc}") from exc
        if response.status != 200:
            raise CatalogueQueryError(
                f"CDSE catalogue returned HTTP {response.status}: {response.text(limit=300)}")
        try:
            payload = json.loads(response.body)
        except ValueError as exc:
            raise CatalogueQueryError("CDSE catalogue response is not valid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), list):
            raise CatalogueQueryError("CDSE catalogue response has no 'value' list")
        if not all(isinstance(item, dict) for item in payload["value"]):
            raise CatalogueQueryError("CDSE catalogue 'value' contains non-object entries")
        return payload


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

_FRACTION = re.compile(r"(\.\d+)")


def parse_odata_datetime(value: Any) -> datetime:
    """Parse an OData timestamp such as ``2026-08-30T00:15:20.123Z`` to aware UTC."""
    if not isinstance(value, str) or not value:
        raise MetadataError(f"missing or non-string timestamp: {value!r}")
    text = value.strip().replace("Z", "+00:00")
    text = _FRACTION.sub(lambda m: m.group(1)[:7].ljust(7, "0"), text, count=1)  # 6 fraction digits
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MetadataError(f"unparseable timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise MetadataError(f"timestamp without timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _attributes(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("Attributes")
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise MetadataError("'Attributes' is not a list (was $expand=Attributes applied?)")
    result: dict[str, Any] = {}
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("Name"), str):
            raise MetadataError("malformed attribute entry")
        name, value = entry["Name"], entry.get("Value")
        if name in result and result[name] != value:
            raise MetadataError(f"conflicting values for attribute {name!r}")
        result[name] = value
    return result


def parse_polarisation_channels(value: Any) -> tuple[Polarization, ...]:
    """Tolerant parse of e.g. "VV&VH", "VV VH", "VV,VH", "VV+VH", "HH"."""
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"polarisationChannels is not a non-empty string: {value!r}")
    tokens = [t for t in re.split(r"[&\s,+/;]+", value.strip().upper()) if t]
    try:
        pols = tuple(Polarization(t) for t in tokens)
    except ValueError as exc:
        raise MetadataError(f"unknown polarisation in {value!r}") from exc
    if len(set(pols)) != len(pols):
        raise MetadataError(f"duplicate polarisation in {value!r}")
    return pols


def parse_footprint(record: dict[str, Any]) -> PolygonGeometry:
    geo = record.get("GeoFootprint")
    if isinstance(geo, dict):
        kind, coords = geo.get("type"), geo.get("coordinates")
        if kind == "MultiPolygon" and isinstance(coords, list) and len(coords) == 1:
            kind, coords = "Polygon", coords[0]
        if kind != "Polygon" or not isinstance(coords, list) or len(coords) != 1:
            raise MetadataError(f"unsupported footprint geometry ({kind}, {len(coords or [])} parts/rings); "
                                "multi-part footprints (e.g. antimeridian) are not supported")
        ring = tuple((float(p[0]), float(p[1])) for p in coords[0])
    elif isinstance(record.get("Footprint"), str):
        match = re.search(r"POLYGON\s*\(\((?P<body>[^()]*)\)\)", record["Footprint"], re.IGNORECASE)
        if match is None:
            raise MetadataError("Footprint WKT is not a single POLYGON")
        ring = tuple((float(x), float(y)) for x, y in (pair.split() for pair in match["body"].split(",")))
    else:
        raise MetadataError("no footprint (neither GeoFootprint nor Footprint)")
    try:
        return PolygonGeometry(coordinates=(ring,))
    except ValueError as exc:
        raise MetadataError(f"invalid footprint polygon: {exc}") from exc


@dataclass(frozen=True)
class ParsedS1Record:
    """A catalogue record whose metadata is complete and self-consistent.

    ``orbit_direction`` / ``relative_orbit`` are kept as delivered so the
    classifier can reject out-of-range values as REJECTED_ORBIT (a different
    reason from missing metadata).
    """

    product_id: str
    name: S1ProductName
    start: datetime
    stop: datetime | None
    publication_date: datetime | None
    platform: str
    mode: str
    product_type: str             # e.g. "IW_GRDH_1S" (a "-COG" suffix is removed)
    polarizations: tuple[Polarization, ...]
    orbit_direction: str          # upper-case as delivered, e.g. "DESCENDING"
    relative_orbit: int
    footprint: PolygonGeometry
    metadata_sources: dict[str, str]  # field -> "attribute" | "product_name"


def parse_s1_record(record: dict[str, Any]) -> ParsedS1Record:
    """Validate one raw OData record.

    Raises:
        MetadataError: required metadata missing, malformed or inconsistent.
    """
    product_id = record.get("Id")
    if not isinstance(product_id, str) or not product_id.strip():
        raise MetadataError("record has no 'Id'")
    raw_name = record.get("Name")
    if not isinstance(raw_name, str):
        raise MetadataError(f"record {product_id} has no 'Name'")
    name = parse_s1_product_name(raw_name)

    content = record.get("ContentDate")
    if not isinstance(content, dict):
        raise MetadataError("record has no 'ContentDate'")
    start = parse_odata_datetime(content.get("Start"))
    stop = parse_odata_datetime(content["End"]) if content.get("End") else None
    if abs((start.replace(microsecond=0) - name.start).total_seconds()) > _NAME_TIME_TOLERANCE_S:
        raise MetadataError(f"ContentDate/Start {start.isoformat()} disagrees with the product name")
    publication = parse_odata_datetime(record["PublicationDate"]) if record.get("PublicationDate") else None

    attrs = _attributes(record)
    sources: dict[str, str] = {}

    def cross_checked(field: str, attr_value: Any, from_name: Any) -> Any:
        if attr_value is None:
            sources[field] = "product_name"
            return from_name
        if attr_value != from_name:
            raise MetadataError(f"{field}: catalogue says {attr_value!r}, product name says {from_name!r}")
        sources[field] = "attribute"
        return attr_value

    # Attribute value formats are not fully documented, so known spellings are
    # normalised before the cross-check: platformSerialIdentifier "A", "S1A" or
    # "SENTINEL-1A" -> "S1A"; productType "IW_GRDH_1S", "IW_GRDH_1S-COG" or the
    # short kind "GRD" -> the full type from the name.
    serial = attrs.get("platformSerialIdentifier")
    platform_attr: str | None = None
    if serial is not None:
        serial_match = re.search(r"([A-D])$", str(serial).strip().upper())
        if serial_match is None:
            raise MetadataError(f"unrecognised platformSerialIdentifier {serial!r}")
        platform_attr = f"S1{serial_match.group(1)}"
    platform = cross_checked("platform", platform_attr, name.platform)
    mode_attr = attrs.get("operationalMode")
    mode = cross_checked("mode", str(mode_attr).strip().upper() if mode_attr is not None else None, name.mode)
    ptype_attr = attrs.get("productType")
    ptype_norm: str | None = None
    if ptype_attr is not None:
        ptype_norm = str(ptype_attr).strip().upper().removesuffix("-COG")
        if ptype_norm == name.product_kind:
            ptype_norm = name.product_type
    product_type = cross_checked("product_type", ptype_norm, name.product_type)
    pol_attr = attrs.get("polarisationChannels")
    pols_attr = parse_polarisation_channels(pol_attr) if pol_attr is not None else None
    polarizations = cross_checked("polarizations",
                                  tuple(sorted(pols_attr, key=lambda p: p.value)) if pols_attr else None,
                                  tuple(sorted(name.polarizations, key=lambda p: p.value)))

    direction = attrs.get("orbitDirection")
    if not isinstance(direction, str) or not direction.strip():
        raise MetadataError("missing catalogue attribute 'orbitDirection' (not derivable from the name)")
    rel = attrs.get("relativeOrbitNumber")
    if isinstance(rel, bool) or rel is None:
        raise MetadataError("missing catalogue attribute 'relativeOrbitNumber' (not derivable from the name)")
    try:
        relative_orbit = int(rel)
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"relativeOrbitNumber is not an integer: {rel!r}") from exc
    if str(rel).strip() != str(relative_orbit):
        raise MetadataError(f"relativeOrbitNumber is not an integer: {rel!r}")
    sources["orbit_direction"] = sources["relative_orbit"] = "attribute"

    return ParsedS1Record(
        product_id=product_id.strip(), name=name, start=start, stop=stop, publication_date=publication,
        platform=platform, mode=mode, product_type=product_type, polarizations=polarizations,
        orbit_direction=direction.strip().upper(), relative_orbit=relative_orbit,
        footprint=parse_footprint(record), metadata_sources=sources,
    )


# ---------------------------------------------------------------------------
# Single-product metadata (public; used before downloading)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductMetadata:
    product_id: str
    name: str
    content_length: int | None
    online: bool | None
    checksums: dict[str, str]          # lower-case algorithm -> lower-case value


def _parse_checksums(raw: Any) -> dict[str, str]:
    """Tolerant parse of the OData ``Checksum`` field (a list of
    {Algorithm, Value, ChecksumDate}); unknown shapes yield no checksums."""
    result: dict[str, str] = {}
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and isinstance(entry.get("Algorithm"), str) \
                    and isinstance(entry.get("Value"), str) and entry["Value"].strip():
                result[entry["Algorithm"].strip().lower().replace("-", "")] = entry["Value"].strip().lower()
    return result


def fetch_product_metadata(product_id: str, settings: CDSESettings,
                           transport: Transport | None = None) -> ProductMetadata:
    """GET ``Products(<id>)`` from the public catalogue (no login).

    Raises:
        CatalogueQueryError: unreachable, not found, or malformed response.
    """
    if not re.fullmatch(r"[0-9a-fA-F-]{8,64}", product_id):
        raise CatalogueQueryError(f"not a CDSE product id: {product_id!r}")
    transport = transport or UrllibTransport()
    url = f"{settings.odata_url}({product_id})"
    try:
        response = transport.get(url, settings.timeout_s)
    except TransportError as exc:
        raise CatalogueQueryError(f"CDSE catalogue unreachable: {exc}") from exc
    if response.status == 404:
        raise CatalogueQueryError(f"product {product_id} not found in the CDSE catalogue")
    if response.status != 200:
        raise CatalogueQueryError(f"CDSE catalogue returned HTTP {response.status} for product {product_id}")
    try:
        payload = json.loads(response.body)
    except ValueError as exc:
        raise CatalogueQueryError(f"product {product_id}: metadata is not valid JSON") from exc
    if not isinstance(payload, dict) or str(payload.get("Id", "")).lower() != product_id.lower():
        raise CatalogueQueryError(f"product {product_id}: metadata response does not describe this product")
    name = payload.get("Name")
    if not isinstance(name, str) or not name:
        raise CatalogueQueryError(f"product {product_id}: metadata has no Name")
    length = payload.get("ContentLength")
    online = payload.get("Online")
    return ProductMetadata(
        product_id=product_id, name=name,
        content_length=int(length) if isinstance(length, int) and not isinstance(length, bool) and length > 0
        else None,
        online=online if isinstance(online, bool) else None,
        checksums=_parse_checksums(payload.get("Checksum")),
    )

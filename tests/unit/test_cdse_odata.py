"""CDSE OData client: query building, pagination, provider errors, record parsing,
and the end-to-end search_sentinel1 orchestration with a fake transport."""

from __future__ import annotations

import copy
import json
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from bato.acquisition.cdse_odata import (
    CDSEODataClient,
    build_s1_filter,
    parse_odata_datetime,
    parse_polarisation_channels,
    parse_s1_record,
)
from bato.acquisition.http import HttpResponse, TransportError
from bato.acquisition.product_name import parse_s1_product_name
from bato.acquisition.s1_catalogue import search_sentinel1
from bato.contracts.aoi import AOIRequest, build_aoi
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import BBox, Polarization, PolygonGeometry
from bato.contracts.io import load_contract, save_contract
from bato.exceptions import CatalogueQueryError, MetadataError

from helpers import FIXED_TODAY

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cdse_s1_catalogue_TEST_FIXTURE.json"
UTC = timezone.utc


class FakeTransport:
    """Serves queued responses; records requested URLs. No network."""

    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, timeout_s: float) -> HttpResponse:
        self.urls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post_form(self, url, form, timeout_s):  # pragma: no cover - not used here
        raise AssertionError("catalogue search must not POST")


def page(values: list[dict], next_link: str | None = None) -> HttpResponse:
    payload: dict = {"value": values}
    if next_link:
        payload["@odata.nextLink"] = next_link
    return HttpResponse(200, json.dumps(payload).encode())


@pytest.fixture(scope="module")
def raw_records() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["value"]


@pytest.fixture
def aoi(cfg):
    request = AOIRequest(event_date=date(2026, 8, 26),
                         bbox=BBox(min_lon=85.20, min_lat=28.00, max_lon=85.40, max_lat=28.20))
    return build_aoi(request, cfg.settings, today=FIXED_TODAY)


def settings_with(cfg, **update):
    return cfg.settings.cdse.model_copy(update=update)


# ---- query -------------------------------------------------------------------------

def test_filter_uses_only_documented_operators(aoi):
    f = build_s1_filter(aoi.geometry, datetime(2026, 6, 27, tzinfo=UTC), datetime(2026, 9, 20, tzinfo=UTC))
    assert f.startswith("Collection/Name eq 'SENTINEL-1'")
    assert "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((85.200000 28.000000, " in f
    assert "ContentDate/Start ge 2026-06-27T00:00:00.000Z" in f
    assert "ContentDate/Start lt 2026-09-20T00:00:00.000Z" in f


def test_filter_polygon_is_counter_clockwise():
    clockwise = PolygonGeometry(coordinates=(((85.2, 28.0), (85.2, 28.2), (85.4, 28.2), (85.4, 28.0), (85.2, 28.0)),))
    f = build_s1_filter(clockwise, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC))
    assert "POLYGON((85.200000 28.000000, 85.400000 28.000000," in f


def test_filter_rejects_naive_or_inverted_window(aoi):
    with pytest.raises(ValueError):
        build_s1_filter(aoi.geometry, datetime(2026, 1, 1), datetime(2026, 2, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        build_s1_filter(aoi.geometry, datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))


def test_request_url_parameters(cfg, aoi):
    transport = FakeTransport([page([])])
    CDSEODataClient(cfg.settings.cdse, transport).fetch_s1(
        aoi.geometry, datetime(2026, 6, 27, tzinfo=UTC), datetime(2026, 9, 20, tzinfo=UTC))
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(transport.urls[0]).query)
    assert query["$expand"] == ["Attributes"]
    assert query["$orderby"] == ["ContentDate/Start asc"]
    assert query["$top"] == [str(cfg.settings.cdse.page_size)]
    assert query["$filter"][0].startswith("Collection/Name eq 'SENTINEL-1'")


# ---- pagination ------------------------------------------------------------------------

def window():
    return datetime(2026, 6, 27, tzinfo=UTC), datetime(2026, 9, 20, tzinfo=UTC)


def test_pagination_follows_next_link(cfg, aoi, raw_records):
    transport = FakeTransport([page(raw_records[:10], next_link="https://example.test/next"),
                               page(raw_records[10:])])
    result = CDSEODataClient(cfg.settings.cdse, transport).fetch_s1(aoi.geometry, *window())
    assert len(result.records) == len(raw_records) and result.pages == 2 and not result.truncated
    assert transport.urls[1] == "https://example.test/next"


def test_pagination_falls_back_to_skip(cfg, aoi, raw_records):
    transport = FakeTransport([page(raw_records[:5]), page(raw_records[5:8])])
    result = CDSEODataClient(settings_with(cfg, page_size=5, max_records=100), transport).fetch_s1(
        aoi.geometry, *window())
    assert len(result.records) == 8 and "%24skip=5" in transport.urls[1]


def test_max_records_truncates_visibly(cfg, aoi, raw_records):
    transport = FakeTransport([page(raw_records[:5]), page(raw_records[5:10])])
    result = CDSEODataClient(settings_with(cfg, page_size=5, max_records=5), transport).fetch_s1(
        aoi.geometry, *window())
    assert result.truncated and len(result.records) == 5


# ---- provider errors ---------------------------------------------------------------------

@pytest.mark.parametrize(("response", "message"), [
    (HttpResponse(500, b"internal error"), "HTTP 500"),
    (HttpResponse(400, b"bad filter"), "HTTP 400: bad filter"),
    (HttpResponse(200, b"<html>not json</html>"), "not valid JSON"),
    (HttpResponse(200, b'{"items": []}'), "no 'value' list"),
    (HttpResponse(200, b'{"value": [1, 2]}'), "non-object"),
    (TransportError("request to catalogue.dataspace.copernicus.eu failed: timed out"), "unreachable"),
])
def test_provider_errors_raise_catalogue_query_error(cfg, aoi, response, message):
    client = CDSEODataClient(cfg.settings.cdse, FakeTransport([response]))
    with pytest.raises(CatalogueQueryError, match=message):
        client.fetch_s1(aoi.geometry, *window())


# ---- record parsing ---------------------------------------------------------------------

def test_parse_valid_record(raw_records):
    p = parse_s1_record(raw_records[0])
    assert p.platform == "S1A" and p.mode == "IW" and p.product_type == "IW_GRDH_1S"
    assert p.polarizations == (Polarization.VH, Polarization.VV)
    assert p.orbit_direction == "DESCENDING" and p.relative_orbit == 19
    assert p.start == datetime(2026, 8, 2, 0, 15, 20, 412000, tzinfo=UTC)
    assert p.metadata_sources["platform"] == "attribute"


@pytest.mark.parametrize(("value", "expected"), [
    ("VV&VH", {"VV", "VH"}), ("VV VH", {"VV", "VH"}), ("vv,vh", {"VV", "VH"}), ("HH", {"HH"}),
])
def test_polarisation_formats(value, expected):
    assert {p.value for p in parse_polarisation_channels(value)} == expected


@pytest.mark.parametrize("value", ["", "VV&XX", "VV&VV", 7])
def test_bad_polarisation_values(value):
    with pytest.raises(MetadataError):
        parse_polarisation_channels(value)


@pytest.mark.parametrize("stamp", ["2026-08-02T00:15:20Z", "2026-08-02T00:15:20.1Z",
                                   "2026-08-02T00:15:20.123456789Z", "2026-08-02T05:45:20+05:30"])
def test_odata_datetime_variants(stamp):
    assert parse_odata_datetime(stamp).tzinfo is not None


@pytest.mark.parametrize("stamp", ["2026-08-02T00:15:20", "yesterday", None, 5])
def test_odata_datetime_rejects_bad_values(stamp):
    with pytest.raises(MetadataError):
        parse_odata_datetime(stamp)


def _mutated(raw: dict, **changes) -> dict:
    record = copy.deepcopy(raw)
    attr_changes = changes.pop("attrs", {})
    record["Attributes"] = [a for a in record["Attributes"] if attr_changes.get(a["Name"], "") is not None]
    for a in record["Attributes"]:
        if a["Name"] in attr_changes:
            a["Value"] = attr_changes[a["Name"]]
    for key, value in changes.items():
        if value is None:
            record.pop(key, None)
        else:
            record[key] = value
    return record


@pytest.mark.parametrize(("changes", "message"), [
    ({"Id": None}, "no 'Id'"),
    ({"Name": None}, "no 'Name'"),
    ({"attrs": {"relativeOrbitNumber": None}}, "relativeOrbitNumber"),
    ({"attrs": {"relativeOrbitNumber": "19.5"}}, "not an integer"),
    ({"attrs": {"relativeOrbitNumber": True}}, "relativeOrbitNumber"),
    ({"attrs": {"platformSerialIdentifier": "B"}}, "platform"),
    ({"attrs": {"platformSerialIdentifier": "X9"}}, "platformSerialIdentifier"),
    ({"attrs": {"operationalMode": "EW"}}, "mode"),
    ({"attrs": {"productType": "IW_SLC__1S"}}, "product_type"),
    ({"ContentDate": {"Start": "2026-08-02T03:00:00.000Z"}}, "disagrees with the product name"),
    ({"GeoFootprint": None}, "no footprint"),
    ({"GeoFootprint": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]],
                                                               [[[2, 2], [3, 2], [3, 3], [2, 2]]]]}},
     "multi-part"),
    ({"GeoFootprint": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}},
     "invalid footprint"),
    ({"Attributes": "nope"}, "not a list"),
])
def test_malformed_or_missing_metadata(raw_records, changes, message):
    with pytest.raises(MetadataError, match=message):
        parse_s1_record(_mutated(raw_records[0], **changes))


def test_attributes_absent_fall_back_to_name_but_orbit_still_required(raw_records):
    record = _mutated(raw_records[0], attrs={"platformSerialIdentifier": None, "operationalMode": None,
                                             "productType": None, "polarisationChannels": None})
    p = parse_s1_record(record)
    assert p.platform == "S1A" and p.metadata_sources["platform"] == "product_name"
    with pytest.raises(MetadataError, match="orbitDirection"):
        parse_s1_record(_mutated(raw_records[0], attrs={"orbitDirection": None}))


@pytest.mark.parametrize("ptype", ["GRD", "IW_GRDH_1S", "IW_GRDH_1S-COG", "iw_grdh_1s"])
def test_product_type_spellings_accepted(raw_records, ptype):
    assert parse_s1_record(_mutated(raw_records[0], attrs={"productType": ptype})).product_type == "IW_GRDH_1S"


@pytest.mark.parametrize("serial", ["A", "S1A", "SENTINEL-1A", "a"])
def test_platform_spellings_accepted(raw_records, serial):
    assert parse_s1_record(_mutated(raw_records[0], attrs={"platformSerialIdentifier": serial})).platform == "S1A"


def test_wkt_footprint_fallback(raw_records):
    wkt = "geography'SRID=4326;POLYGON((84.7 27.4,86.1 27.62,85.85 28.95,84.42 28.73,84.7 27.4))'"
    p = parse_s1_record(_mutated(raw_records[0], GeoFootprint=None, Footprint=wkt))
    assert p.footprint.exterior[1] == (86.1, 27.62)


def test_product_name_parser():
    n = parse_s1_product_name("S1C_IW_GRDH_1SDV_20260901T001519_20260901T001544_009275_00C005_1A05_COG.SAFE")
    assert (n.platform, n.product_type, n.polarization_code, n.absolute_orbit, n.is_cog) == \
        ("S1C", "IW_GRDH_1S", "DV", 9275, True)
    with pytest.raises(MetadataError, match="stop time precedes"):
        parse_s1_product_name("S1A_IW_GRDH_1SDV_20260901T001519_20260901T001500_009275_00C005_1A05")


# ---- orchestration -------------------------------------------------------------------

def test_search_sentinel1_end_to_end_with_fake_transport(cfg, aoi, raw_records, tmp_path, caplog):
    client = CDSEODataClient(cfg.settings.cdse, FakeTransport([page(raw_records)]))
    retrieved = datetime(2026, 10, 1, 6, 0, tzinfo=UTC)
    with caplog.at_level("INFO", logger="bato"):
        result = search_sentinel1(aoi, cfg, client=client, retrieved_at=retrieved)
    assert result.returned_count == 21 and len(result.accepted()) == 9 and len(result.rejected()) == 12
    assert result.status_counts()["rejected_invalid_metadata"] == 4
    assert result.explain("TEST-FIXTURE-0015").startswith(
        "S1A_IW_GRDH_1SDV_20260828T124510_20260828T124535_065380_0A1B0F_1A0F.SAFE: rejected_coverage")
    assert "not returned" in result.explain("S1A_NOT_IN_RESULT")
    finished = [r for r in caplog.records if getattr(r, "bato", {}).get("event") == "finished"]
    assert finished and finished[0].bato["outputs"]["accepted"] == 9
    # The result round-trips through JSON for inspection.
    assert load_contract(CatalogueSearchResult, save_contract(result, tmp_path / "s1.json")) == result


def test_search_sentinel1_logs_and_reraises_provider_errors(cfg, aoi, caplog):
    client = CDSEODataClient(cfg.settings.cdse, FakeTransport([HttpResponse(503, b"maintenance")]))
    with caplog.at_level("INFO", logger="bato"), pytest.raises(CatalogueQueryError, match="HTTP 503"):
        search_sentinel1(aoi, cfg, client=client)
    assert any(getattr(r, "bato", {}).get("event") == "failed" for r in caplog.records)


def test_search_warns_when_nothing_is_accepted(cfg, aoi, raw_records, caplog):
    only_bad = [r for r in raw_records if r["Id"] in {"TEST-FIXTURE-0008", "TEST-FIXTURE-0018"}]
    client = CDSEODataClient(cfg.settings.cdse, FakeTransport([page(only_bad)]))
    with caplog.at_level("INFO", logger="bato"):
        result = search_sentinel1(aoi, cfg, client=client)
    assert result.accepted() == [] and result.groups == ()
    assert any(getattr(r, "bato", {}).get("event") == "warning" for r in caplog.records)

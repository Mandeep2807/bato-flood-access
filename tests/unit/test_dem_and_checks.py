"""Copernicus DEM tile discovery/fetch, archive content checks, token provider,
and download configuration (all offline)."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bato.acquisition.cdse_auth import CDSECredentials, TokenProvider
from bato.acquisition.cdse_odata import CDSEODataClient
from bato.acquisition.dem import (
    aoi_bbox_from_catalogue,
    build_dem_filter,
    dem_search_bbox,
    plan_dem,
    required_tile_ids,
    tile_id,
    tile_of_record,
)
from bato.acquisition.downloader import Downloader
from bato.acquisition.http import HttpResponse, StreamResponse
from bato.acquisition.s1_fetch import build_s1_fetch_plan, execute_fetch
from bato.acquisition.safe_inspect import ContentCheckError, check_dem_archive, check_s1_grd_zip
from bato.config import load_config
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import BBox, PolygonGeometry
from bato.contracts.fetch import DownloadProvenance, FetchStatus
from bato.contracts.io import load_contract
from bato.contracts.temporal import TemporalSelectionResult
from bato.exceptions import AuthenticationError, ConfigError

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "real_catalogues"
DiskUsage = namedtuple("DiskUsage", "total used free")
NOW = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)


# ---- tile geometry -----------------------------------------------------------------------------

@pytest.mark.parametrize(("lat", "lon", "expected"), [(27, 85, "N27E085"), (39, -1, "N39W001"), (-1, -71, "S01W071"),
                                                     (0, 0, "N00E000")])
def test_tile_id(lat, lon, expected):
    assert tile_id(lat, lon) == expected


def test_required_tiles_for_real_aois():
    trishuli = load_contract(CatalogueSearchResult, REAL / "trishuli_s1.json")
    valencia = load_contract(CatalogueSearchResult, REAL / "valencia_s1.json")
    t_box = aoi_bbox_from_catalogue(trishuli)
    assert t_box == BBox(min_lon=85.17, min_lat=27.97, max_lon=85.42, max_lat=28.28)
    assert required_tile_ids(t_box) == ("N27E085", "N28E085")
    assert required_tile_ids(dem_search_bbox(t_box, 20.0)) == ("N27E084", "N27E085", "N28E084", "N28E085")
    assert required_tile_ids(dem_search_bbox(aoi_bbox_from_catalogue(valencia), 20.0)) == ("N39W001",)


def test_box_edge_on_degree_line_does_not_add_a_tile():
    assert required_tile_ids(BBox(min_lon=85.0, min_lat=27.5, max_lon=86.0, max_lat=28.0)) == ("N27E085",)


def test_dem_filter_uses_dataset_attribute_and_area():
    area = PolygonGeometry(coordinates=(((85.0, 27.0), (86.0, 27.0), (86.0, 28.0), (85.0, 28.0), (85.0, 27.0)),))
    f = build_dem_filter(area, "COP-DEM_GLO-30-DGED/2024_1")
    assert "att/Name eq 'dataset'" in f and "Value eq 'COP-DEM_GLO-30-DGED/2024_1'" in f
    assert "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((85.000000 27.000000" in f
    with pytest.raises(ValueError):
        build_dem_filter(area, "bad'name")


def dem_record(pid: str, name: str, lat: int, lon: int) -> dict:
    ring = [[lon, lat], [lon + 1, lat], [lon + 1, lat + 1], [lon, lat + 1], [lon, lat]]
    return {"Id": pid, "Name": name, "GeoFootprint": {"type": "Polygon", "coordinates": [ring]}}


def test_tile_of_record_from_name_or_footprint():
    assert tile_of_record({"Name": "Copernicus_DSM_10_N27_00_E085_00"}) == "N27E085"
    assert tile_of_record(dem_record("x", "UNNAMED-PRODUCT", 28, 84)) == "N28E084"
    assert tile_of_record({"Name": "nothing"}) is None


class FakeCatalogue:
    def __init__(self, records):
        self.records = records
        self.urls = []

    def get(self, url, timeout_s):
        self.urls.append(url)
        return HttpResponse(200, json.dumps({"value": self.records}).encode())

    def post_form(self, url, form, timeout_s):  # pragma: no cover
        raise AssertionError


def test_plan_dem_reports_missing_tiles(cfg):
    trishuli = load_contract(CatalogueSearchResult, REAL / "trishuli_s1.json")
    records = [dem_record("00000000-0000-0000-0000-00000000aa01", "Copernicus_DSM_10_N27_00_E085_00", 27, 85),
               dem_record("00000000-0000-0000-0000-00000000aa02", "Copernicus_DSM_10_N28_00_E085_00", 28, 85),
               dem_record("00000000-0000-0000-0000-00000000aa03", "Copernicus_DSM_10_N27_00_E084_00", 27, 84)]
    fake = FakeCatalogue(records)
    plan = plan_dem(trishuli.aoi_id, aoi_bbox_from_catalogue(trishuli), cfg, CDSEODataClient(cfg.settings.cdse, fake))
    assert plan.required_tiles == ("N27E084", "N27E085", "N27E086",
                                   "N28E084", "N28E085", "N28E086",
                                   "N29E084", "N29E085", "N29E086")
    assert plan.missing_tiles == ("N27E086", "N28E084", "N28E086", "N29E084", "N29E085", "N29E086")
    assert "COP-DEM_GLO-30-DGED" in fake.urls[0].replace("%2F", "/").replace("%20", " ")


def test_dem_tile_download_with_provenance(config_copy):
    project = load_config(config_copy())
    selection = load_contract(TemporalSelectionResult, REAL / "trishuli_sets.json")
    catalogue = load_contract(CatalogueSearchResult, REAL / "trishuli_s1.json")
    plan = build_s1_fetch_plan(selection, catalogue, project, tracks=["ASCENDING/85"])
    pid = "00000000-0000-0000-0000-00000000aa01"
    fake = FakeCatalogue([dem_record(pid, "Copernicus_DSM_10_N27_00_E085_00", 27, 85)])
    dem_plan = plan_dem(catalogue.aoi_id, aoi_bbox_from_catalogue(catalogue), project,
                        CDSEODataClient(project.settings.cdse, fake))
    tiff = b"II*\x00" + b"\x00" * 100    # TEST FIXTURE: GeoTIFF magic only

    class Fake:
        def get(self, url, timeout_s):
            return HttpResponse(200, json.dumps({"Id": pid, "Name": "Copernicus_DSM_10_N27_00_E085_00",
                                                 "ContentLength": len(tiff), "Online": True}).encode())

        def post_form(self, url, form, timeout_s):  # pragma: no cover
            raise AssertionError

        def open_stream(self, url, headers, timeout_s, chunk_bytes):
            assert pid in url
            return StreamResponse(200, {}, url, iter([tiff]), lambda: None)

    class Tokens:
        def get(self):
            return "T"

        def invalidate(self):
            pass

    downloader = Downloader(project.settings.download, Fake(), Tokens(), sleep=lambda s: None,
                            disk_usage=lambda p: DiskUsage(0, 0, 10**12))
    # max_scenes=0: no Sentinel-1 product is fetched, only the DEM tile.
    manifest = execute_fetch(plan, dem_plan, project, dry_run=False, downloader=downloader,
                             metadata_transport=Fake(), max_scenes=0, clock=lambda: NOW)
    dem_files = [f for f in manifest.files if f.kind == "copernicus_dem_tile"]
    assert [f.status for f in dem_files] == [FetchStatus.DOWNLOADED]
    prov = load_contract(DownloadProvenance, project.config_dir.parent / dem_files[0].provenance_path)
    assert prov.tile_id == "N27E085" and prov.dem_dataset == project.settings.dem.cdse_dataset
    assert prov.data_source.value == "copernicus-dem" and prov.checksum_verified is False
    assert (project.config_dir.parent / dem_files[0].local_path).parent.name == "copernicus_dem"


# ---- archive checks --------------------------------------------------------------------------------

def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_s1_zip_check(tmp_path):
    base = "S1A_X.SAFE"
    members = {f"{base}/manifest.safe": b""}
    for pol in ("vv", "vh"):
        members[f"{base}/measurement/s1a-iw-grd-{pol}-1.tiff"] = b""
        members[f"{base}/annotation/calibration/calibration-s1a-iw-grd-{pol}-1.xml"] = b""
        members[f"{base}/annotation/calibration/noise-s1a-iw-grd-{pol}-1.xml"] = b""
    assert "VV+VH" in check_s1_grd_zip(make_zip(tmp_path / "ok.zip", members), ("VV", "VH"))
    del members[f"{base}/annotation/calibration/calibration-s1a-iw-grd-vh-1.xml"]
    with pytest.raises(ContentCheckError, match="calibration VH annotation"):
        check_s1_grd_zip(make_zip(tmp_path / "bad.zip", members), ("VV", "VH"))
    (tmp_path / "html.zip").write_text("<html>error page</html>")
    with pytest.raises(ContentCheckError, match="not a valid zip"):
        check_s1_grd_zip(tmp_path / "html.zip", ("VV",))


def test_dem_archive_check_accepts_tiff_zip_tar(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"II*\x00rest")
    assert "GeoTIFF" in check_dem_archive(tmp_path / "a.tif")
    assert "_DEM.tif" in check_dem_archive(make_zip(tmp_path / "b.zip", {"x/DEM/t_DEM.tif": b""}))
    buf = io.BytesIO(b"data")
    with tarfile.open(tmp_path / "c.tar", "w") as tar:
        info = tarfile.TarInfo("x/DEM/t_DEM.tif")
        info.size = 4
        tar.addfile(info, buf)
    assert "_DEM.tif" in check_dem_archive(tmp_path / "c.tar")
    with pytest.raises(ContentCheckError, match="no \\*_DEM.tif"):
        check_dem_archive(make_zip(tmp_path / "d.zip", {"x/readme.txt": b""}))
    (tmp_path / "e.bin").write_bytes(b"garbage!")
    with pytest.raises(ContentCheckError, match="neither"):
        check_dem_archive(tmp_path / "e.bin")


# ---- token provider -----------------------------------------------------------------------------

class TokenEndpoint:
    def __init__(self):
        self.calls = 0

    def get(self, url, timeout_s):  # pragma: no cover
        raise AssertionError

    def post_form(self, url, form, timeout_s):
        self.calls += 1
        return HttpResponse(200, json.dumps({"access_token": f"tok{self.calls}", "expires_in": 600}).encode())


def test_token_provider_is_lazy_caches_and_refreshes(cfg):
    endpoint = TokenEndpoint()
    provider = TokenProvider(cfg.settings.cdse, endpoint, environ={})
    assert endpoint.calls == 0                                   # nothing read or requested yet
    with pytest.raises(AuthenticationError, match="credentials missing"):
        provider.get()
    provider = TokenProvider(cfg.settings.cdse, endpoint, credentials=CDSECredentials("u", "p"))
    assert provider.get(NOW) == "tok1" and provider.get(NOW) == "tok1"
    assert provider.get(NOW + timedelta(seconds=599)) == "tok2"  # expired (60 s margin)
    provider.invalidate()
    assert provider.get(NOW + timedelta(seconds=600)) == "tok3" and provider.refreshes == 3


# ---- configuration --------------------------------------------------------------------------------

@pytest.mark.parametrize(("section", "update", "message"), [
    ("download", {"base_url": "http://download.dataspace.copernicus.eu/odata/v1/Products"}, "https"),
    ("download", {"s1_subdir": "../escape"}, "plain folder name"),
    ("download", {"allowed_redirect_suffixes": []}, "allowed_redirect_suffixes"),
    ("download", {"min_free_disk_ratio": 0.5}, "min_free_disk_ratio"),
    ("dem", {"cdse_dataset": "SRTM_30"}, "COP-DEM"),
])
def test_invalid_fetch_config_rejected(config_copy, section, update, message):
    def mutate(d):
        d[section].update(update)
    with pytest.raises(ConfigError, match=message):
        load_config(config_copy("config.yaml", mutate))

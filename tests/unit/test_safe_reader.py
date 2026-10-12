"""Step 3.1: SAFE zip inspection (synthetic TEST FIXTURE zips; no real imagery)."""

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bato.contracts.common import DataSource, OrbitDirection, Polarization
from bato.contracts.fetch import DownloadProvenance
from bato.exceptions import ProhibitedInputError
from bato.preprocessing.safe_reader import SafeInspectionError, check_against_provenance, inspect_safe_zip

NAME = "S1D_IW_GRDH_1SDV_20260629T122138_20260629T122203_003451_006174_F84B.SAFE"
SAMPLES, LINES = 40, 30      # tiny synthetic image size (real: 25801 x 16730)


def stem(pol: str) -> str:
    n = {"vv": "001", "vh": "002"}[pol]
    return f"s1d-iw-grd-{pol}-20260629t122138-20260629t122203-003451-006174-{n}"


def annotation(pol: str, *, product_type: str = "GRD", pass_: str = "Ascending", samples: int = SAMPLES,
               spacing: str = "1.000000e+01", points: int = 6) -> str:
    grid = "".join("<geolocationGridPoint><line>0</line><pixel>0</pixel></geolocationGridPoint>" for _ in range(points))
    return (f"<product><adsHeader><productType>{product_type}</productType><polarisation>{pol.upper()}</polarisation>"
            f"</adsHeader><generalAnnotation><productInformation><pass>{pass_}</pass></productInformation>"
            f"</generalAnnotation><imageAnnotation><imageInformation><numberOfSamples>{samples}</numberOfSamples>"
            f"<numberOfLines>{LINES}</numberOfLines><rangePixelSpacing>{spacing}</rangePixelSpacing>"
            f"<azimuthPixelSpacing>{spacing}</azimuthPixelSpacing></imageInformation></imageAnnotation>"
            f"<geolocationGrid><geolocationGridPointList>{grid}</geolocationGridPointList></geolocationGrid></product>")


def calibration(values: int = 4, pixels: int | None = None) -> str:
    vec = (f"<calibrationVector><line>{{line}}</line><pixel>{' '.join(str(i * 10) for i in range(pixels or values))}"
           f"</pixel><sigmaNought>{' '.join('5.0e+02' for _ in range(values))}</sigmaNought></calibrationVector>")
    return f"<calibration><calibrationVectorList>{vec.format(line=0)}{vec.format(line=29)}</calibrationVectorList>" \
           "</calibration>"


def make_safe(path: Path, *, pols=("vv", "vh"), skip: tuple[str, ...] = (), ann_overrides: dict | None = None,
              cal_override: str | None = None, raster_bytes: int | None = None) -> Path:
    """TEST FIXTURE: a SAFE zip with the real member layout and tiny contents."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{NAME}/manifest.safe", "<xfdu/>")
        for pol in pols:
            s = stem(pol)
            files = {
                "measurement": (f"{NAME}/measurement/{s}.tiff", b"\0" * (raster_bytes or SAMPLES * LINES * 2 + 8)),
                "annotation": (f"{NAME}/annotation/{s}.xml", annotation(pol, **(ann_overrides or {}).get(pol, {}))),
                "calibration": (f"{NAME}/annotation/calibration/calibration-{s}.xml", cal_override or calibration()),
                "noise": (f"{NAME}/annotation/calibration/noise-{s}.xml", "<noise/>"),
            }
            for role, (member, data) in files.items():
                if f"{pol}:{role}" not in skip:
                    z.writestr(member, data)
        z.writestr(f"{NAME}/annotation/rfi/rfi-{stem('vv')}.xml", "<rfi/>")   # present in real products, ignored
    return path


@pytest.fixture
def raw(cfg, tmp_path):
    paths = cfg.settings.paths.model_copy(update={"raw": tmp_path / "raw", "reference": tmp_path / "reference",
                                                  "eval": tmp_path / "eval"})
    (tmp_path / "raw").mkdir()
    return paths


def inspect(path, paths):
    return inspect_safe_zip(path, paths, ("VV", "VH"))


def test_valid_product(raw):
    product = inspect(make_safe(raw.raw / f"{NAME}.zip"), raw)
    assert product.product_name == NAME and product.orbit_direction is OrbitDirection.ASCENDING
    assert [b.polarization for b in product.bands] == [Polarization.VH, Polarization.VV]
    vv = product.band("VV")
    assert (vv.samples, vv.lines, vv.range_pixel_spacing_m) == (SAMPLES, LINES, 10.0)
    assert vv.measurement_member.endswith(f"{stem('vv')}.tiff") and vv.noise_member.endswith(f"noise-{stem('vv')}.xml")
    assert (vv.calibration_vectors, vv.calibration_points_per_vector, vv.geolocation_points) == (2, 4, 6)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"pols": ("vv",)}, "exactly one VH measurement"),
    ({"skip": ("vh:calibration",)}, "calibration file missing"),
    ({"skip": ("vv:noise",)}, "noise file missing"),
    ({"skip": ("vv:annotation",)}, "annotation file missing"),
    ({"ann_overrides": {"vh": {"product_type": "SLC"}}}, "productType 'SLC'"),
    ({"ann_overrides": {"vh": {"samples": 41}}, "raster_bytes": 41 * LINES * 2}, "same image grid"),
    ({"ann_overrides": {"vh": {"pass_": "Descending"}}}, "disagree on pass direction"),
    ({"ann_overrides": {"vv": {"pass_": "Sideways"}, "vh": {"pass_": "Sideways"}}}, "unknown pass direction"),
    ({"ann_overrides": {"vv": {"spacing": "ten"}}}, "not a number"),
    ({"ann_overrides": {"vv": {"points": 2}}}, "too few geolocation"),
    ({"cal_override": calibration(values=4, pixels=5)}, "5 pixel positions but 4 sigmaNought"),
    ({"raster_bytes": 100}, "smaller than a uint16 raster"),
])
def test_invalid_products(raw, kwargs, message):
    with pytest.raises(SafeInspectionError, match=message):
        inspect(make_safe(raw.raw / "p.zip", **kwargs), raw)


def test_annotation_polarisation_must_match_file_name(raw, tmp_path):
    path = raw.raw / "swap.zip"
    with zipfile.ZipFile(make_safe(tmp_path / "src.zip")) as src, zipfile.ZipFile(path, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item.endswith(f"annotation/{stem('vh')}.xml"):
                data = annotation("vv").encode()
            dst.writestr(item, data)
    with pytest.raises(SafeInspectionError, match="annotation says VV, file name says VH"):
        inspect(path, raw)


def test_not_a_zip_and_missing_manifest(raw):
    (raw.raw / "bad.zip").write_text("<html>error</html>")
    with pytest.raises(SafeInspectionError, match="cannot open as zip"):
        inspect(raw.raw / "bad.zip", raw)
    path = raw.raw / "nomanifest.zip"
    with zipfile.ZipFile(make_safe(raw.raw / "tmp.zip")) as src, zipfile.ZipFile(path, "w") as dst:
        for item in src.namelist():
            if not item.endswith("manifest.safe"):
                dst.writestr(item, src.read(item))
    with pytest.raises(SafeInspectionError, match="manifest.safe missing"):
        inspect(path, raw)


def test_reference_folder_is_refused(raw):
    (raw.reference).mkdir()
    with pytest.raises(ProhibitedInputError):
        inspect(make_safe(raw.reference / "p.zip"), raw)


def provenance(**update) -> DownloadProvenance:
    fields = dict(
        kind="sentinel1_grd", data_source=DataSource.SENTINEL_1, product_id="00000000-0000-0000-0000-000000000001",
        product_name=NAME, source="cdse-odata-download",
        source_url="https://download.dataspace.copernicus.eu/odata/v1/Products(x)/$value", content_length=1,
        server_checksums={}, checksum_verified_with=None, checksum_verified=False, sha256="a" * 64, size_bytes=1,
        local_path=f"data/raw/sentinel1/{NAME}.zip", content_check="ok",
        downloaded_at=datetime(2026, 10, 1, tzinfo=timezone.utc), attempts=1, resumed=False, bato_version="0.1.0",
        config_fingerprint="f", platform="S1D",
        acquisition_start=datetime(2026, 6, 29, 12, 21, 38, tzinfo=timezone.utc),
        orbit_direction=OrbitDirection.ASCENDING, relative_orbit=85, polarizations=(Polarization.VH, Polarization.VV),
        product_type="IW_GRDH_1S", pass_ids=("S1D-A085-20260629T122138",))
    fields.update(update)
    return DownloadProvenance(**fields)


def test_provenance_cross_check(raw):
    product = inspect(make_safe(raw.raw / "p.zip"), raw)
    check_against_provenance(product, provenance())
    with pytest.raises(SafeInspectionError, match="provenance says"):
        check_against_provenance(product, provenance(product_name="S1D_OTHER.SAFE"))
    with pytest.raises(SafeInspectionError, match="annotation pass ascending, provenance descending"):
        check_against_provenance(product, provenance(orbit_direction=OrbitDirection.DESCENDING))
    with pytest.raises(SafeInspectionError, match="does not list"):
        check_against_provenance(product, provenance(polarizations=(Polarization.VV,)))

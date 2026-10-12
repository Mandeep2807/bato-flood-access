"""Step 3.1: read-only inspection of downloaded Sentinel-1 GRD SAFE zips.

Reads only the XML annotation files inside the zip (a few MB); the 860 MB
measurement rasters are never opened. Nothing is extracted or written.

For each required polarisation the four member files are matched by their
common file stem (``s1d-iw-grd-vv-<times>-<ids>-001``):

    measurement/<stem>.tiff
    annotation/<stem>.xml
    annotation/calibration/calibration-<stem>.xml
    annotation/calibration/noise-<stem>.xml

and the annotation is checked against the file name (product type GRD,
polarisation). :func:`check_against_provenance` then compares the product with
the Step 2C provenance sidecar written at download time.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bato.config import PathSettings
from bato.contracts.common import OrbitDirection, Polarization
from bato.contracts.fetch import DownloadProvenance
from bato.contracts.safe import SafeBand, SafeProduct
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError

_MEASUREMENT = re.compile(
    r"^(?P<top>[^/]+\.safe)/measurement/(?P<stem>s1[a-d]-[a-z0-9]+-grd-(?P<pol>[hv]{2})-[^/]+)\.tiff?$")


class SafeInspectionError(BatoError):
    """A SAFE zip is malformed or inconsistent with what BATO expects."""


def _text(root: ET.Element, path: str, member: str) -> str:
    element = root.find(path)
    if element is None or element.text is None or not element.text.strip():
        raise SafeInspectionError(f"{member}: missing <{path}>")
    return element.text.strip()


def _number(root: ET.Element, path: str, member: str, kind: type[int] | type[float]) -> float:
    raw = _text(root, path, member)
    try:
        return kind(float(raw)) if kind is int else float(raw)
    except ValueError as exc:
        raise SafeInspectionError(f"{member}: <{path}> is not a number: {raw!r}") from exc


def _parse_xml(archive: zipfile.ZipFile, member: str) -> ET.Element:
    try:
        return ET.fromstring(archive.read(member))
    except ET.ParseError as exc:
        raise SafeInspectionError(f"{member}: invalid XML ({exc})") from exc


def _band(archive: zipfile.ZipFile, names: dict[str, str], measurement_lower: str, top: str, stem: str,
          pol: str) -> tuple[SafeBand, str]:
    """Build one SafeBand; also return the annotated pass direction."""
    expected = {
        "annotation": f"{top}/annotation/{stem}.xml",
        "calibration": f"{top}/annotation/calibration/calibration-{stem}.xml",
        "noise": f"{top}/annotation/calibration/noise-{stem}.xml",
    }
    members = {}
    for role, lower in expected.items():
        if lower not in names:
            raise SafeInspectionError(f"{pol.upper()}: {role} file missing for measurement {stem}")
        members[role] = names[lower]
    measurement = names[measurement_lower]

    ann = _parse_xml(archive, members["annotation"])
    a = members["annotation"]
    product_type = _text(ann, "adsHeader/productType", a)
    if product_type != "GRD":
        raise SafeInspectionError(f"{a}: productType {product_type!r}, expected 'GRD'")
    annotated_pol = _text(ann, "adsHeader/polarisation", a).upper()
    if annotated_pol != pol.upper():
        raise SafeInspectionError(f"{a}: annotation says {annotated_pol}, file name says {pol.upper()}")
    pass_direction = _text(ann, "generalAnnotation/productInformation/pass", a)

    cal = _parse_xml(archive, members["calibration"])
    vectors = cal.findall("calibrationVectorList/calibrationVector")
    for index, vector in enumerate(vectors):
        pixels = (vector.findtext("pixel") or "").split()
        sigma = (vector.findtext("sigmaNought") or "").split()
        if not sigma or len(pixels) != len(sigma):
            raise SafeInspectionError(f"{members['calibration']}: vector {index} has {len(pixels)} pixel positions "
                                      f"but {len(sigma)} sigmaNought values")

    try:
        band = SafeBand(
            polarization=Polarization(pol.upper()), measurement_member=measurement,
            annotation_member=members["annotation"], calibration_member=members["calibration"],
            noise_member=members["noise"], measurement_size_bytes=archive.getinfo(measurement).file_size,
            samples=int(_number(ann, "imageAnnotation/imageInformation/numberOfSamples", a, int)),
            lines=int(_number(ann, "imageAnnotation/imageInformation/numberOfLines", a, int)),
            range_pixel_spacing_m=_number(ann, "imageAnnotation/imageInformation/rangePixelSpacing", a, float),
            azimuth_pixel_spacing_m=_number(ann, "imageAnnotation/imageInformation/azimuthPixelSpacing", a, float),
            geolocation_points=len(ann.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")),
            calibration_vectors=len(vectors),
            calibration_points_per_vector=len((vectors[0].findtext("sigmaNought") or "").split()) if vectors else 0,
        )
    except ValueError as exc:
        reason = exc.errors()[0]["msg"] if hasattr(exc, "errors") else str(exc)
        raise SafeInspectionError(f"{pol.upper()} band: {reason}") from exc
    return band, pass_direction


def inspect_safe_zip(path: Path, paths: PathSettings, polarizations: tuple[str, ...]) -> SafeProduct:
    """Inspect a GRD SAFE zip (annotations only).

    Raises:
        ProhibitedInputError: the file lies in an evaluation-only folder.
        SafeInspectionError: missing files, invalid XML, inconsistent metadata, or
            VV/VH not on one image grid.
    """
    resolved = assert_pipeline_input_allowed(path, paths)
    try:
        archive = zipfile.ZipFile(resolved)
    except (zipfile.BadZipFile, OSError) as exc:
        raise SafeInspectionError(f"{resolved.name}: cannot open as zip ({exc})") from exc
    with archive:
        names = {n.replace("\\", "/").lower(): n.replace("\\", "/") for n in archive.namelist()}
        measurements: dict[str, list[tuple[str, str, str]]] = {}
        tops = set()
        for lower in names:
            match = _MEASUREMENT.match(lower)
            if match:
                tops.add(match["top"])
                measurements.setdefault(match["pol"], []).append((lower, match["top"], match["stem"]))
        if len(tops) != 1:
            raise SafeInspectionError(f"{resolved.name}: expected one .SAFE folder with measurements, "
                                      f"found {len(tops)}")
        top = tops.pop()
        if f"{top}/manifest.safe" not in names:
            raise SafeInspectionError(f"{resolved.name}: manifest.safe missing")
        product_name = names[f"{top}/manifest.safe"].split("/")[0]
        bands, passes = [], set()
        for pol in sorted(p.lower() for p in polarizations):
            found = measurements.get(pol, [])
            if len(found) != 1:
                raise SafeInspectionError(f"{resolved.name}: expected exactly one {pol.upper()} measurement, "
                                          f"found {len(found)}")
            band, pass_direction = _band(archive, names, *found[0], pol)
            bands.append(band)
            passes.add(pass_direction)
        if len(passes) != 1:
            raise SafeInspectionError(f"{resolved.name}: polarisations disagree on pass direction {sorted(passes)}")
        direction = passes.pop()
        if direction.lower() not in {d.value for d in OrbitDirection}:
            raise SafeInspectionError(f"{resolved.name}: unknown pass direction {direction!r}")
        try:
            return SafeProduct(archive_name=resolved.name, product_name=product_name, product_type="GRD",
                               orbit_direction=OrbitDirection(direction.lower()), bands=tuple(bands))
        except ValueError as exc:
            reason = exc.errors()[0]["msg"] if hasattr(exc, "errors") else str(exc)
            raise SafeInspectionError(f"{resolved.name}: {reason}") from exc


def check_against_provenance(product: SafeProduct, provenance: DownloadProvenance) -> None:
    """The zip content must be the product the download provenance describes.

    Raises:
        SafeInspectionError: name, orbit direction or polarisations disagree.
    """
    problems = []
    if provenance.kind != "sentinel1_grd":
        problems.append(f"provenance kind is {provenance.kind}")
    if product.product_name.removesuffix(".SAFE") != provenance.product_name.removesuffix(".SAFE"):
        problems.append(f"zip contains {product.product_name}, provenance says {provenance.product_name}")
    if provenance.orbit_direction is not product.orbit_direction:
        problems.append(f"annotation pass {product.orbit_direction.value}, provenance "
                        f"{provenance.orbit_direction.value if provenance.orbit_direction else None}")
    missing = {b.polarization for b in product.bands} - set(provenance.polarizations)
    if missing:
        problems.append(f"provenance does not list {sorted(p.value for p in missing)}")
    if problems:
        raise SafeInspectionError(f"{product.archive_name}: " + "; ".join(problems))

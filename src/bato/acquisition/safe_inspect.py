"""Structural checks of downloaded archives (central directory only; no pixels read).

Sentinel-1 GRD (SAFE in a zip) must contain, for every required polarisation:
  measurement/<mission>-iw-grd-<pol>-....tiff            the GRD raster (DN, not yet sigma0)
  annotation/calibration/calibration-...-<pol>-....xml   sigma0 LUT  (needed for linear sigma0)
  annotation/calibration/noise-...-<pol>-....xml         thermal noise LUT
plus manifest.safe. Keeping these means preprocessing can later produce
linear-domain sigma0 exactly as Kuro Siwo did; nothing is converted here.

Copernicus DEM tile: a GeoTIFF, or a zip/tar containing a ``*_DEM.tif`` raster.
"""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

from bato.exceptions import ProviderError


class ContentCheckError(ProviderError):
    """A downloaded archive is not the product it should be."""


def _members(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            return [m.replace("\\", "/") for m in archive.namelist()]
    except zipfile.BadZipFile as exc:
        raise ContentCheckError(f"{path.name} is not a valid zip archive: {exc}") from exc


def check_s1_grd_zip(path: Path, polarizations: tuple[str, ...]) -> str:
    """Return a one-line summary, or raise ContentCheckError naming what is missing."""
    members = _members(path)
    lower = [m.lower() for m in members]
    missing = []
    if not any(m.endswith(".safe/manifest.safe") or m == "manifest.safe" for m in lower):
        missing.append("manifest.safe")
    for pol in (p.lower() for p in polarizations):
        if not any(re.search(rf"/measurement/s1[a-d]-[a-z0-9]+-grd-{pol}-[^/]*\.tiff?$", m) for m in lower):
            missing.append(f"measurement {pol.upper()} raster")
        if not any(re.search(rf"/annotation/calibration/calibration-[^/]*-{pol}-[^/]*\.xml$", m) for m in lower):
            missing.append(f"calibration {pol.upper()} annotation")
        if not any(re.search(rf"/annotation/calibration/noise-[^/]*-{pol}-[^/]*\.xml$", m) for m in lower):
            missing.append(f"noise {pol.upper()} annotation")
    if missing:
        raise ContentCheckError(f"{path.name} is missing: {', '.join(missing)}")
    pols = "+".join(p.upper() for p in polarizations)
    return f"SAFE zip OK ({len(members)} entries): manifest, {pols} measurement, calibration and noise annotations"


def check_dem_archive(path: Path) -> str:
    """DEM packaging is OPEN (zip, tar or a bare GeoTIFF); accept any of them as
    long as a DEM raster is present."""
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic in (b"II*\x00", b"MM\x00*"):
        return f"DEM GeoTIFF OK: {path.name}"
    if zipfile.is_zipfile(path):
        members = _members(path)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = archive.getnames()
    else:
        raise ContentCheckError(f"{path.name} is neither a GeoTIFF, a zip nor a tar archive")
    dems = [m for m in members if m.lower().endswith("_dem.tif")]
    if not dems:
        raise ContentCheckError(f"{path.name} contains no *_DEM.tif raster")
    return f"DEM archive OK ({len(members)} entries): {Path(dems[0]).name}"

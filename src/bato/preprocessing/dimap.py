"""Read the header (``.dim``) of a SNAP BEAM-DIMAP product. No pixels are read here.

A DIMAP product is ``<name>.dim`` (XML header) plus ``<name>.data/`` holding one
ENVI raster per band (``<band>.img`` + ``<band>.hdr``) and the tie-point grids
(``tie_point_grids/latitude.img`` ...). This module extracts what Step 3.2
validation needs:

* raster size (NCOLS/NROWS) and, per band, name, size, data type, unit,
  no-data value, log10 scaling and the ENVI file;
* tie-point grid files (latitude/longitude = radar-geometry geocoding);
* whether a map geocoding (IMAGE_TO_MODEL_TRANSFORM, written for map-projected,
  i.e. terrain-corrected products) is present;
* the ``Abstracted_Metadata`` attributes (pixel spacing, orbit file, flags);
* the operators SNAP recorded in the product's ``Processing_Graph`` history.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from bato.exceptions import BatoError


class DimapError(BatoError):
    """A DIMAP header is missing, malformed, or references missing files."""


@dataclass(frozen=True)
class DimapBand:
    index: int
    name: str
    width: int
    height: int
    data_type: str
    unit: str | None
    nodata_used: bool
    nodata_value: float | None
    log10_scaled: bool
    virtual: bool                     # defined by an expression, no file of its own
    image_file: Path | None           # ENVI .img (None for virtual bands)


@dataclass(frozen=True)
class DimapProduct:
    path: Path
    data_dir: Path
    width: int
    height: int
    bands: tuple[DimapBand, ...]
    tie_point_grids: dict[str, Path]  # name -> ENVI .img
    map_geocoding: bool               # IMAGE_TO_MODEL_TRANSFORM present
    abstracted: dict[str, str] = field(default_factory=dict)
    history: tuple[str, ...] = ()     # operators in SNAP's Processing_Graph, in order

    def band(self, name: str) -> DimapBand | None:
        return next((b for b in self.bands if b.name == name), None)

    def abstracted_float(self, name: str) -> float | None:
        raw = self.abstracted.get(name)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None


def _int(element: ET.Element, tag: str, where: str) -> int:
    text = element.findtext(tag)
    if text is None or not text.strip():
        raise DimapError(f"{where}: missing <{tag}>")
    try:
        return int(float(text))
    except ValueError as exc:
        raise DimapError(f"{where}: <{tag}> is not a number: {text!r}") from exc


def _img_for(href: str, base: Path) -> Path:
    """DIMAP references the ENVI header (.hdr); the pixels are in the sibling .img."""
    return (base / href).with_suffix(".img")


def read_dimap(path: Path) -> DimapProduct:
    """Parse a ``.dim`` header and check that every referenced band file exists.

    Raises:
        DimapError: missing/invalid header, missing ``.data`` folder or band file.
    """
    path = Path(path)
    if not path.is_file():
        raise DimapError(f"DIMAP header not found: {path}")
    data_dir = path.with_suffix(".data")
    if not data_dir.is_dir():
        raise DimapError(f"DIMAP data folder not found: {data_dir}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DimapError(f"{path.name}: invalid XML ({exc})") from exc

    dims = root.find("Raster_Dimensions")
    if dims is None:
        raise DimapError(f"{path.name}: missing <Raster_Dimensions>")
    width, height = _int(dims, "NCOLS", path.name), _int(dims, "NROWS", path.name)

    base = path.parent
    files: dict[int, Path] = {}
    for data_file in root.iter("Data_File"):
        ref = data_file.find("DATA_FILE_PATH")
        if ref is not None and ref.get("href"):
            files[_int(data_file, "BAND_INDEX", path.name)] = _img_for(ref.get("href", ""), base)

    tpg_files: dict[int, Path] = {}
    for tpg_file in root.iter("Tie_Point_Grid_File"):
        ref = tpg_file.find("TIE_POINT_GRID_FILE_PATH")
        if ref is not None and ref.get("href"):
            tpg_files[_int(tpg_file, "TIE_POINT_GRID_INDEX", path.name)] = _img_for(ref.get("href", ""), base)
    tie_point_grids: dict[str, Path] = {}
    for info in root.iter("Tie_Point_Grid_Info"):
        index = _int(info, "TIE_POINT_GRID_INDEX", path.name)
        name = (info.findtext("TIE_POINT_GRID_NAME") or "").strip()
        if name and index in tpg_files:
            tie_point_grids[name] = tpg_files[index]

    bands = []
    for info in root.iter("Spectral_Band_Info"):
        index = _int(info, "BAND_INDEX", path.name)
        name = (info.findtext("BAND_NAME") or "").strip()
        if not name:
            raise DimapError(f"{path.name}: band {index} has no name")
        expression = (info.findtext("EXPRESSION") or "").strip()
        virtual = bool(expression) or (info.findtext("VIRTUAL_BAND") or "").strip().lower() == "true"
        nodata_used = (info.findtext("NO_DATA_VALUE_USED") or "").strip().lower() == "true"
        nodata_raw = (info.findtext("NO_DATA_VALUE") or "").strip()
        unit = (info.findtext("PHYSICAL_UNIT") or "").strip() or None
        image = files.get(index)
        if not virtual and (image is None or not image.is_file()):
            raise DimapError(f"{path.name}: raster file of band {name} not found ({image})")
        bands.append(DimapBand(
            index=index, name=name,
            width=_int(info, "BAND_RASTER_WIDTH", name) if info.find("BAND_RASTER_WIDTH") is not None else width,
            height=_int(info, "BAND_RASTER_HEIGHT", name) if info.find("BAND_RASTER_HEIGHT") is not None else height,
            data_type=(info.findtext("DATA_TYPE") or "").strip(), unit=unit, nodata_used=nodata_used,
            nodata_value=float(nodata_raw) if nodata_used and nodata_raw else None,
            log10_scaled=(info.findtext("LOG10_SCALED") or "").strip().lower() == "true",
            virtual=virtual, image_file=None if virtual else image,
        ))

    abstracted: dict[str, str] = {}
    for element in root.iter("MDElem"):
        if element.get("name") == "Abstracted_Metadata":
            for attr in element.findall("MDATTR"):
                if attr.get("name"):
                    abstracted[attr.get("name", "")] = (attr.text or "").strip()
            break

    history: list[str] = []
    for element in root.iter("MDElem"):
        if element.get("name") == "Processing_Graph":
            for node in element.findall("MDElem"):
                for attr in node.findall("MDATTR"):
                    if attr.get("name") == "operator" and attr.text:
                        history.append(attr.text.strip())
            break

    return DimapProduct(
        path=path, data_dir=data_dir, width=width, height=height,
        bands=tuple(sorted(bands, key=lambda b: b.index)), tie_point_grids=tie_point_grids,
        map_geocoding=next(root.iter("IMAGE_TO_MODEL_TRANSFORM"), None) is not None,
        abstracted=abstracted, history=tuple(history),
    )

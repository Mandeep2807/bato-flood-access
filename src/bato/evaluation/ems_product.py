"""EVALUATION ONLY. Read-only access to Copernicus EMS product zips (inspection and layer loading).

Supported vector formats:

* GeoJSON (``.json`` / ``.geojson``): full read;
* ESRI shapefile (``.shp`` + ``.dbf`` + ``.prj``): full read of points, lines and polygons;
* GeoPackage: listed and described at inspection, but not loaded.

Rasters (GeoTIFF) are described from their header only. Nested zips are opened one level deep, in memory.
Nothing is extracted next to pipeline data; the zip is never modified.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import struct
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from bato.evaluation.geometry import iter_coords


class ProductError(ValueError):
    pass


LAYER_NAME = re.compile(
    r"^(?P<product>EMSR\d+_AOI\d+_[A-Z]+_(?:PRODUCT|MONIT\d+))_(?P<stem>[A-Za-z][A-Za-z0-9]*)_v(?P<ver>\d+)"
    r"\.(?P<ext>[A-Za-z0-9]+)$")
VECTOR_EXT = {"json": "geojson", "geojson": "geojson", "shp": "shapefile", "gpkg": "geopackage"}
SHAPE_PARTS = {"shp", "shx", "dbf", "prj", "cpg", "sbn", "sbx", "qix", "xml"}
TEXT_EXT = {"json", "geojson", "xml", "txt", "csv", "html", "htm", "md", "prj", "qml", "sld", "lyr", "kml"}
ZONE_CLUE = re.compile(r"\b(UTC|GMT|NPT|Nepal (?:Standard )?Time|local time)\b"
                       r"|\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})", re.IGNORECASE)
DATE_KEY = re.compile(r"date|time|acq|src|sensor|epoch", re.IGNORECASE)
WGS84_PRJ = re.compile(r'GEOGCS\["[^"]*(WGS[ _]?(19)?84)', re.IGNORECASE)
CRS84 = {"urn:ogc:def:crs:OGC:1.3:CRS84", "urn:ogc:def:crs:EPSG::4326", "EPSG:4326", "epsg:4326",
         "http://www.opengis.net/def/crs/OGC/1.3/CRS84", "http://www.opengis.net/def/crs/EPSG/0/4326"}
MAX_VALUES = 60


# ---------------------------------------------------------------------------
# Members (one level of nested zips)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    path: str                     # "outer/inner.zip!/name" for nested members
    size: int
    data_loader: Any              # () -> bytes

    @property
    def name(self) -> str:
        return PurePosixPath(self.path.split("!/")[-1]).name

    @property
    def ext(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""

    def read(self) -> bytes:
        data: bytes = self.data_loader()
        return data


def iter_members(zip_path: Path) -> Iterator[Member]:
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
    for info in infos:
        name = info.filename
        yield Member(name, info.file_size, lambda n=name: _read(zip_path, n))
        if name.lower().endswith(".zip"):
            inner = io.BytesIO(_read(zip_path, name))
            with zipfile.ZipFile(inner) as iz:
                for ii in iz.infolist():
                    if ii.is_dir():
                        continue
                    yield Member(f"{name}!/{ii.filename}", ii.file_size,
                                 lambda n=name, m=ii.filename: _read_nested(zip_path, n, m))


def _read(zip_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name)


def _read_nested(zip_path: Path, outer: str, inner: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(_read(zip_path, outer))) as iz:
        return iz.read(inner)


def layer_key(member: Member) -> tuple[str, str, int] | None:
    """(product, stem, version) from an EMS layer file name, else None."""
    m = LAYER_NAME.match(member.name)
    if m is None:
        return None
    return m.group("product"), m.group("stem"), int(m.group("ver"))


# ---------------------------------------------------------------------------
# Shapefile / DBF (pure Python)
# ---------------------------------------------------------------------------


def read_dbf(data: bytes, encoding: str = "utf-8") -> list[dict[str, Any]]:
    n_rec, header_len, rec_len = struct.unpack("<IHH", data[4:12])
    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("ascii", "replace")
        ftype = chr(data[pos + 11])
        flen, fdec = data[pos + 16], data[pos + 17]
        fields.append((name, ftype, flen, fdec))
        pos += 32
    rows = []
    for r in range(n_rec):
        off = header_len + r * rec_len
        if data[off:off + 1] == b"*":
            continue                                   # deleted record
        cur = off + 1
        row: dict[str, Any] = {}
        for name, ftype, flen, fdec in fields:
            raw = data[cur:cur + flen]
            cur += flen
            text = raw.decode(encoding, "replace").strip()
            if ftype in "NF":
                row[name] = None if not text or set(text) <= {"*"} else (float(text) if fdec or "." in text
                                                                          else int(text))
            elif ftype == "L":
                row[name] = None if text in ("", "?") else text.upper() in ("T", "Y")
            else:
                row[name] = text
        rows.append(row)
    return rows


SHAPE_TYPES = {0: "Null", 1: "Point", 3: "PolyLine", 5: "Polygon", 8: "MultiPoint", 11: "PointZ", 13: "PolyLineZ",
               15: "PolygonZ", 18: "MultiPointZ", 21: "PointM", 23: "PolyLineM", 25: "PolygonM", 28: "MultiPointM"}
# MultiPatch (31) is deliberately absent: read_shp refuses it.


def shp_header(data: bytes) -> dict[str, Any]:
    code = struct.unpack("<i", data[32:36])[0]
    xmin, ymin, xmax, ymax = struct.unpack("<4d", data[36:68])
    return {"shape_type": SHAPE_TYPES.get(code, f"code {code}"), "bbox": [xmin, ymin, xmax, ymax]}


def _shp_polygons(rings: list[list[list[float]]]) -> dict[str, Any]:
    """Shapefile rings -> (Multi)Polygon: clockwise rings are outer rings, counter-clockwise rings are holes of the
    preceding outer ring (ESRI shapefile specification)."""
    polys: list[list[list[list[float]]]] = []
    for ring in rings:
        signed = sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                     for i in range(len(ring)))
        if signed <= 0 or not polys:
            polys.append([ring])
        else:
            polys[-1].append(ring)
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def read_shp(data: bytes) -> list[dict[str, Any] | None]:
    """GeoJSON-like geometries (points, multipoints, lines, polygons; Z/M values ignored; MultiPatch refused)."""
    out: list[dict[str, Any] | None] = []
    pos = 100
    while pos + 8 <= len(data):
        _num, length = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + 2 * length]
        pos += 8 + 2 * length
        code = struct.unpack("<i", content[:4])[0]
        if code not in SHAPE_TYPES:
            raise ProductError(f"unsupported shape type {code}")
        base = code % 10 if code >= 10 else code
        if code == 0:
            out.append(None)
        elif base == 1:
            x, y = struct.unpack("<2d", content[4:20])
            out.append({"type": "Point", "coordinates": [x, y]})
        elif base == 8:
            n = struct.unpack("<i", content[36:40])[0]
            pts = [list(struct.unpack("<2d", content[40 + 16 * i:56 + 16 * i])) for i in range(n)]
            out.append({"type": "MultiPoint", "coordinates": pts})
        elif base in (3, 5):
            nparts, npts = struct.unpack("<2i", content[36:44])
            parts = list(struct.unpack(f"<{nparts}i", content[44:44 + 4 * nparts]))
            p0 = 44 + 4 * nparts
            pts = [list(struct.unpack("<2d", content[p0 + 16 * i:p0 + 16 * i + 16])) for i in range(npts)]
            rings = [pts[s:e] for s, e in zip(parts, parts[1:] + [npts], strict=True)]
            out.append(_shp_polygons(rings) if base == 5 else {"type": "MultiLineString", "coordinates": rings})
        else:
            raise ProductError(f"unsupported shape type {code}")
    return out


# ---------------------------------------------------------------------------
# GeoTIFF header (inspection only)
# ---------------------------------------------------------------------------


def tiff_info(data: bytes) -> dict[str, Any]:
    if data[:2] not in (b"II", b"MM"):
        return {"error": "not a TIFF"}
    e = "<" if data[:2] == b"II" else ">"
    magic = struct.unpack(e + "H", data[2:4])[0]
    if magic == 43:
        return {"format": "BigTIFF", "note": "header not parsed"}
    ifd = struct.unpack(e + "I", data[4:8])[0]
    n = struct.unpack(e + "H", data[ifd:ifd + 2])[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}
    fmt = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d", 16: "Q"}
    tags: dict[int, Any] = {}
    for i in range(n):
        off = ifd + 2 + 12 * i
        tag, typ, cnt = struct.unpack(e + "HHI", data[off:off + 8])
        size = sizes.get(typ, 1) * cnt
        where = off + 8 if size <= 4 else struct.unpack(e + "I", data[off + 8:off + 12])[0]
        if typ not in fmt or size > 1 << 20:
            continue
        if typ == 2:
            tags[tag] = data[where:where + cnt].rstrip(b"\x00").decode("latin-1")
        else:
            vals = struct.unpack(e + fmt[typ] * cnt, data[where:where + size])
            tags[tag] = vals
    info: dict[str, Any] = {"width": tags.get(256, [None])[0], "height": tags.get(257, [None])[0],
                            "bits_per_sample": list(tags.get(258, [])), "samples_per_pixel": tags.get(277, [1])[0]}
    if 33550 in tags:
        info["pixel_scale"] = list(tags[33550][:2])
    if 33922 in tags:
        info["tiepoint"] = list(tags[33922][:6])
    if 34735 in tags:
        keys = tags[34735]
        geo = {keys[4 + 4 * k]: keys[4 + 4 * k + 3] for k in range(keys[3]) if keys[4 + 4 * k + 1] == 0}
        if 3072 in geo:
            info["crs"] = f"EPSG:{geo[3072]}"
        elif 2048 in geo:
            info["crs"] = f"EPSG:{geo[2048]}"
    if 42113 in tags:
        info["nodata"] = tags[42113]
    return info


# ---------------------------------------------------------------------------
# Property / CRS analysis
# ---------------------------------------------------------------------------


def describe_properties(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys: dict[str, Counter[Any]] = {}
    for r in rows:
        for k, v in (r or {}).items():
            keys.setdefault(k, Counter())[json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list))
                                          else v] += 1
    out = {}
    for k, c in keys.items():
        entry: dict[str, Any] = {"distinct": len(c), "types": sorted({type(v).__name__ for v in c})}
        if len(c) <= MAX_VALUES or DATE_KEY.search(k):
            entry["values"] = [{"value": v, "count": n} for v, n in sorted(c.items(), key=lambda t: (-t[1], str(t[0])))
                               ][:MAX_VALUES]
            entry["values_truncated"] = len(c) > MAX_VALUES
        else:
            entry["sample"] = [v for v, _ in c.most_common(5)]
        out[k] = entry
    return out


def geojson_crs(fc: dict[str, Any]) -> tuple[str, bool]:
    """(description, is WGS84 lon/lat). No ``crs`` member = WGS84 lon/lat (RFC 7946)."""
    crs = fc.get("crs")
    if crs is None:
        return "none declared (RFC 7946: WGS84 lon/lat)", True
    name = (crs.get("properties") or {}).get("name") if isinstance(crs, dict) else None
    return f"declared: {name or crs}", bool(name in CRS84)


def coords_bbox(geoms: list[dict[str, Any] | None]) -> list[float] | None:
    xs, ys = [], []
    for g in geoms:
        for x, y in iter_coords(g):
            xs.append(x)
            ys.append(y)
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None


def looks_lonlat(bb: list[float] | None) -> bool:
    return bb is None or (-180 <= bb[0] <= 180 and -180 <= bb[2] <= 180 and -90 <= bb[1] <= 90 and -90 <= bb[3] <= 90)


def zone_clues(text: str, where: str, limit: int = 20) -> list[dict[str, str]]:
    out = []
    for m in ZONE_CLUE.finditer(text):
        a, b = max(0, m.start() - 60), min(len(text), m.end() + 60)
        out.append({"where": where, "match": m.group(0), "context": text[a:b].replace("\n", " ")})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    stem: str
    product: str
    version: int
    format: str
    member: str
    crs: str
    lonlat: bool
    features: list[dict[str, Any]]           # GeoJSON features (geometry + properties)

    def geometry_types(self) -> dict[str, int]:
        return dict(Counter((f.get("geometry") or {}).get("type", "None") for f in self.features))


def _load_geojson(m: Member) -> tuple[list[dict[str, Any]], str, bool]:
    fc = json.loads(m.read().decode("utf-8-sig"))
    if fc.get("type") != "FeatureCollection":
        raise ProductError(f"{m.path} is not a GeoJSON FeatureCollection")
    crs, ok = geojson_crs(fc)
    return list(fc.get("features") or []), crs, ok


def _load_shapefile(members: dict[str, Member], m: Member) -> tuple[list[dict[str, Any]], str, bool]:
    stem = m.path[:-4]
    dbf, prj, cpg = members.get(stem + ".dbf"), members.get(stem + ".prj"), members.get(stem + ".cpg")
    if dbf is None:
        raise ProductError(f"{m.path}: no .dbf")
    enc = cpg.read().decode("ascii", "replace").strip() if cpg else "utf-8"
    geoms = read_shp(m.read())
    rows = read_dbf(dbf.read(), enc if enc.lower() not in ("", "utf8") else "utf-8")
    if len(rows) != len(geoms):
        raise ProductError(f"{m.path}: {len(geoms)} shapes but {len(rows)} attribute rows")
    if prj is None:
        return [{"type": "Feature", "geometry": g, "properties": r} for g, r in zip(geoms, rows, strict=True)], \
            "no .prj", False
    wkt = prj.read().decode("latin-1")
    ok = bool(WGS84_PRJ.search(wkt)) and "PROJCS" not in wkt.upper()
    feats = [{"type": "Feature", "geometry": g, "properties": r} for g, r in zip(geoms, rows, strict=True)]
    return feats, f"prj: {wkt[:160]}", ok


class ProductZip:
    """One EMS product zip, read-only."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.members = {m.path: m for m in iter_members(path)}

    def layer_members(self) -> dict[tuple[str, str, int], list[Member]]:
        out: dict[tuple[str, str, int], list[Member]] = {}
        for m in self.members.values():
            k = layer_key(m)
            if k is not None and m.ext in VECTOR_EXT:
                out.setdefault(k, []).append(m)
        return out

    def stems(self) -> set[str]:
        return {k[1] for k in self.layer_members()}

    def load(self, stem: str) -> Layer:
        """The layer with this stem. GeoJSON is preferred over a shapefile of the same layer; several products or
        versions of one stem in one zip are refused (ambiguous)."""
        cands = {k: v for k, v in self.layer_members().items() if k[1] == stem}
        if not cands:
            raise ProductError(f"{self.path.name}: no layer {stem!r}")
        if len(cands) > 1:
            raise ProductError(f"{self.path.name}: layer {stem!r} exists for several products/versions {sorted(cands)}")
        return self.load_key(next(iter(cands)))

    def load_key(self, key: tuple[str, str, int]) -> Layer:
        product, stem, version = key
        ms = self.layer_members()[key]
        by_fmt = {VECTOR_EXT[m.ext]: m for m in ms}
        if "geojson" in by_fmt:
            m = by_fmt["geojson"]
            feats, crs, ok = _load_geojson(m)
            fmt = "geojson"
        elif "shapefile" in by_fmt:
            m = by_fmt["shapefile"]
            feats, crs, ok = _load_shapefile(self.members, m)
            fmt = "shapefile"
        else:
            raise ProductError(f"{self.path.name}: layer {stem!r} is only available as {sorted(by_fmt)}, "
                               "which this reader does not load")
        bb = coords_bbox([f.get("geometry") for f in feats])
        lonlat = ok and looks_lonlat(bb)
        return Layer(stem=stem, product=product, version=version, format=fmt, member=m.path, crs=crs,
                     lonlat=lonlat, features=feats)

    def load_lonlat(self, stem: str) -> Layer:
        layer = self.load(stem)
        if not layer.lonlat:
            raise ProductError(f"{self.path.name}: layer {stem!r} is not WGS84 lon/lat ({layer.crs}); refusing to "
                               "reproject silently")
        return layer


# ---------------------------------------------------------------------------
# Inspection (no metric)
# ---------------------------------------------------------------------------


def inspect_product(path: Path) -> dict[str, Any]:
    pz = ProductZip(path)
    members = []
    for m in sorted(pz.members.values(), key=lambda x: x.path):
        k = layer_key(m)
        members.append({"path": m.path, "bytes": m.size, "ext": m.ext,
                        "layer": None if k is None else {"product": k[0], "stem": k[1], "version": k[2]}})
    layers = []
    clues: list[dict[str, str]] = []
    for (product, stem, version), ms in sorted(pz.layer_members().items()):
        entry: dict[str, Any] = {"product": product, "stem": stem, "version": version,
                                 "formats": sorted({VECTOR_EXT[m.ext] for m in ms}),
                                 "members": sorted(m.path for m in ms)}
        try:
            if entry["formats"] == ["geopackage"]:
                entry.update(describe_geopackage(ms[0].read()))
            else:
                layer = pz.load_key((product, stem, version))
                geoms = [f.get("geometry") for f in layer.features]
                rows = [f.get("properties") or {} for f in layer.features]
                entry.update({"loaded_from": layer.format, "crs": layer.crs, "wgs84_lonlat": layer.lonlat,
                              "features": len(layer.features), "geometry_types": layer.geometry_types(),
                              "bbox": coords_bbox(geoms), "properties": describe_properties(rows)})
                for r in rows:
                    for key, v in r.items():
                        if isinstance(v, str) and DATE_KEY.search(key):
                            clues += zone_clues(v, f"{stem}.{key}", limit=5)
        except (ProductError, ValueError, KeyError, struct.error) as exc:
            entry["error"] = str(exc)
        layers.append(entry)
    rasters, docs = [], []
    for m in pz.members.values():
        if m.ext in ("tif", "tiff"):
            try:
                rasters.append({"path": m.path, **tiff_info(m.read())})
            except (struct.error, IndexError) as exc:
                rasters.append({"path": m.path, "error": str(exc)})
        elif m.ext in TEXT_EXT and m.size < 20_000_000 and layer_key(m) is None:
            text = m.read().decode("utf-8", "replace")
            clues += zone_clues(text, m.path)
            docs.append({"path": m.path, "bytes": m.size})
        elif m.ext == "pdf":
            docs.append({"path": m.path, "bytes": m.size, "note": "PDF map; not parsed"})
    return {"zip": path.name, "members": members, "layers": layers, "rasters": rasters, "documents": docs,
            "time_zone_clues": clues[:80]}


def describe_geopackage(data: bytes) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "layer.gpkg"
        p.write_bytes(data)
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            tables = con.execute("SELECT table_name, data_type, srs_id FROM gpkg_contents").fetchall()
            out: dict[str, Any] = {"loaded_from": None, "note": "GeoPackage: described only, not loaded",
                                   "tables": []}
            for name, kind, srs in tables:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
                n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                srs_row = con.execute("SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys "
                                      "WHERE srs_id = ?", (srs,)).fetchone()
                out["tables"].append({"table": name, "data_type": kind, "rows": n, "columns": cols,
                                      "crs": f"{srs_row[0]}:{srs_row[1]}" if srs_row else f"srs_id {srs}"})
            return out
        finally:
            con.close()

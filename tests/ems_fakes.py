"""TEST FIXTURE: an invented Copernicus-EMS-like activation record and product zips around the synthetic world of
connectivity_fakes. NOT real EMSR927 data; only the structure mimics the public-activations API and GRA zips."""

from __future__ import annotations

import io
import json
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bato.evaluation.geometry import Grid
from bato.preprocessing.terrain_validation import to_lonlat

from helpers import REPO_CONFIG_DIR

HOST = "https://rapidmapping.emergency.copernicus.eu/backend"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR927"
REPO_ROOT = REPO_CONFIG_DIR.parent

# Boxes (lon0, lat0, lon1, lat1) inside the synthetic BATO AOI 85.015-85.045 E, 28.015-28.045 N.
EMS_AOI_BOX = (85.025, 28.020, 85.0448, 28.0430)
FLOOD_BOXES = [(85.0290, 28.0360, 85.0303, 28.0375),    # overlaps the Step 4C FLOOD box of spur 1 (partly)
               (85.0395, 28.0230, 85.0402, 28.0250)]    # on Step 4C NOT_ASSESSED_STEEP (spur 2)
LANDSLIDE_BOX = (85.0330, 28.0310, 85.0360, 28.0330)
TRACE_BOX = (85.0262, 28.0262, 85.0270, 28.0270)
NOTAN_BOX = (85.0260, 28.0210, 85.0280, 28.0230)        # EMS "not analysed" polygon, away from the roads

# EMS road lines ~3 m east of the synthetic BATO road at lon 85.044 (segments seg-203-0-0-k, ~49 m each, k = 0 at
# lat 28.0300). BATO states there: k 9-11 and 20-23 blocked, the others open.
ROAD_X = 85.04403
ROADS = [((28.0339, 28.0353), "Destroyed"),            # covers blocked k 9-11
         ((28.0300, 28.0313), "Destroyed"),            # covers open k 0-2
         ((28.0317, 28.0326), "Damaged"),              # covers open k 4-5
         ((28.0357, 28.0379), "No visible damage"),    # covers open k 13-17
         ((28.0388, 28.0405), "Possibly damaged"),     # covers blocked k 20-23
         ((28.0409, 28.0418), "Not Analysed")]         # covers open k 25-26
BRIDGES = [((85.044, 28.0346), "Destroyed"), ((85.044, 28.0390), "Possibly damaged")]


def box_ring(b: tuple[float, float, float, float]) -> list[list[float]]:
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def feature(geom: dict[str, Any], **props: Any) -> dict[str, Any]:
    return {"type": "Feature", "geometry": geom, "properties": props}


def fc(features: list[dict[str, Any]], crs: str | None = None) -> bytes:
    d: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs:
        d["crs"] = {"type": "name", "properties": {"name": crs}}
    return json.dumps(d).encode()


FAR_DESTROYED = ((85.029, 28.025), (85.033, 28.025))   # ~390 m long, ~557 m from every synthetic BATO road


def product_zip(prefix: str, version: int, *, roads: bool = True, extra_class: str | None = None,
                flood: bool = False, extra_road_value: str | None = None, far_destroyed: bool = False) -> bytes:
    """A GRA-like zip with the layer set inspected in the official EMSR927 products: areaOfInterestA, observedEventA
    (Landslide only, unless ``flood`` adds synthetic open-water polygons for machinery tests), transportationL,
    transportationP, transportationA, builtUpP, facilitiesA, facilitiesL, imageFootprintA, notAnalysedA."""
    poly = lambda b: {"type": "Polygon", "coordinates": [box_ring(b)]}  # noqa: E731
    events = [feature(poly(LANDSLIDE_BOX), obj_desc="Landslide", event_type="6-Mass Movement",
                      det_method="Photo-interpretation")]
    if flood:
        events += [feature(poly(b), obj_desc="Flooded area", event_type="5-Flood") for b in FLOOD_BOXES]
        events.append(feature(poly(TRACE_BOX), obj_desc="Flood trace", event_type="5-Flood"))
    if extra_class:
        events.append(feature(poly((85.0300, 28.0210, 85.0305, 28.0215)), obj_desc=extra_class))
    aoi = [feature(poly(EMS_AOI_BOX), emsr_id="EMSR927", map_type="Grading", locality="Test")]
    line = lambda y0, y1: {"type": "LineString", "coordinates": [[ROAD_X, y0], [ROAD_X, y1]]}  # noqa: E731
    road_feats = [feature(line(y0, y1), damage_gra=v, obj_type="Road") for (y0, y1), v in ROADS] if roads else []
    if far_destroyed:
        road_feats.append(feature({"type": "LineString", "coordinates": [list(p) for p in FAR_DESTROYED]},
                                  damage_gra="Destroyed"))
    if extra_road_value:
        road_feats.append(feature(line(28.0250, 28.0255), damage_gra=extra_road_value))
    bridge_feats = [feature({"type": "Point", "coordinates": list(p)}, damage_gra=v) for p, v in BRIDGES]
    pt = lambda x, y, **k: feature({"type": "Point", "coordinates": [x, y]}, **k)  # noqa: E731
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        w = lambda stem, feats: z.writestr(f"{prefix}_{stem}_v{version}.json", fc(feats))  # noqa: E731
        w("areaOfInterestA", aoi)
        w("observedEventA", events)
        w("transportationL", road_feats)
        w("transportationP", bridge_feats)
        w("transportationA", [feature(poly((85.030, 28.026, 85.031, 28.027)), obj_type="Helipad")])
        w("builtUpP", [pt(85.031, 28.032, damage_gra="Destroyed")])
        w("facilitiesA", [feature(poly((85.027, 28.024, 85.028, 28.025)), obj_type="School")])
        w("facilitiesL", [feature(line(28.021, 28.022), obj_type="Power line")])
        w("imageFootprintA", [feature(poly(EMS_AOI_BOX), src_name="WorldView-3")])
        w("notAnalysedA", [feature(poly(NOTAN_BOX), reason="Cloud")])
        z.writestr(f"Maps/{prefix}_9000_map_v{version}.pdf", b"%PDF-1.4 stand-in")
        z.writestr("README.txt", "Image acquisition times are given in UTC. (synthetic)")
    return buf.getvalue()


def activation(products: dict[str, bytes]) -> dict[str, Any]:
    """Record with AOI01 (base + monitoring, overlapping), AOI02 (no overlap), AOI03 (not feasible)."""
    def prod(aoi: int, kind: str, version: int, images: list[tuple[str, str]], status: str = "F",
             path: bool = True) -> dict[str, Any]:
        name = f"EMSR927_AOI{aoi:02d}_GRA_{kind}_v{version}.zip"
        url = f"{HOST}/EMSR927/AOI{aoi:02d}/GRA_{kind}/{name}" if path else ""
        return {"id": aoi * 10 + version, "type": "GRA", "monitoring": kind != "PRODUCT",
                "monitoringNumber": 0 if kind == "PRODUCT" else int(kind[-2:]), "feasible": status != "N",
                "images": [{"sensorName": s, "acquisitionTime": t} for s, t in images], "downloadPath": url,
                "version": {"uuid": "x", "number": version, "statusCode": status, "deliveryTime": "2026-08-27T19:03"}}

    return {"count": 1, "next": None, "previous": None, "results": [{
        "code": "EMSR927", "name": "Flood in Nepal", "eventTime": "2026-08-25T22:00:00",
        "activationTime": "2026-08-26T09:53:00", "closed": True, "countries": [{"name": "Nepal"}],
        "aois": [
            {"number": 1, "name": "Testpur", "extent": "POLYGON ((85.024 28.019, 85.043 28.019, 85.043 28.041, "
                                                       "85.024 28.041, 85.024 28.019))",
             "products": [prod(1, "PRODUCT", 1, [("WorldView-3", "2026-08-27T05:05:00")]),
                          prod(1, "MONIT01", 1, [("Pleiades", "2026-09-05T04:57:00")])]},
            {"number": 2, "name": "Faraway", "extent": "MULTIPOLYGON (((84.35 27.69, 84.36 27.69, 84.36 27.70, "
                                                       "84.35 27.70, 84.35 27.69)))",
             "products": [prod(2, "PRODUCT", 1, [("Legion", "2026-08-27T05:05:00")])]},
            {"number": 3, "name": "Cloudy", "extent": "POLYGON ((85.02 28.02, 85.03 28.02, 85.03 28.03, 85.02 28.03,"
                                                      " 85.02 28.02))",
             "products": [prod(3, "PRODUCT", 1, [], status="N", path=False)]}]}]}


class FakeOpener:
    """Serves the activation JSON and product zips by URL; records every request."""

    def __init__(self, payload: dict[str, Any], zips: dict[str, bytes]) -> None:
        self.payload, self.zips, self.requests = payload, zips, []

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        if url == API:
            return io.BytesIO(json.dumps(self.payload).encode())
        name = url.rsplit("/", 1)[-1]
        if name in self.zips:
            return io.BytesIO(self.zips[name])
        raise OSError(f"404 {url}")


def standard_zips(**kw: Any) -> dict[str, bytes]:
    return {"EMSR927_AOI01_GRA_PRODUCT_v1.zip": product_zip("EMSR927_AOI01_GRA_PRODUCT", 1, **kw),
            "EMSR927_AOI01_GRA_MONIT01_v1.zip": product_zip("EMSR927_AOI01_GRA_MONIT01", 1, **kw),
            "EMSR927_AOI02_GRA_PRODUCT_v1.zip": product_zip("EMSR927_AOI02_GRA_PRODUCT", 1, **kw)}


PROTOCOL_YAML_REL = Path("config") / "evaluation" / "emsr927_protocol.yaml"
PROTOCOL_DOC_REL = Path("docs") / "evaluation" / "emsr927_protocol.md"
_LIFECYCLE = {b"status": b"status: draft", b"frozen_at": b"frozen_at: null"}


def draft_protocol_bytes(yaml_bytes: bytes) -> bytes:
    """The repository protocol YAML with ONLY its two lifecycle lines reset to an unfrozen state. Every mapping,
    comparison, threshold and comment is kept byte-for-byte (line endings included), so tests exercise the real
    protocol semantics whatever the live repository's status is (draft or legitimately frozen)."""
    out, seen = [], dict.fromkeys(_LIFECYCLE, 0)
    for line in yaml_bytes.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        key = body.split(b":", 1)[0]
        if key in _LIFECYCLE and not body.startswith((b" ", b"\t", b"#")):
            seen[key] += 1
            line = _LIFECYCLE[key] + line[len(body):]
        out.append(line)
    if any(n != 1 for n in seen.values()):
        raise AssertionError(f"protocol YAML must have exactly one top-level status and frozen_at line: {seen}")
    return b"".join(out)


def install_draft_protocol(root: Path) -> None:
    """Install an isolated DRAFT copy of the repository's EMSR927 protocol (YAML + document) under ``root``.

    The live files are only READ. The freeze record lives in the evaluation folder of the test world, so a fresh
    world has none. Refuses to write into the repository itself."""
    if root.resolve() == REPO_ROOT.resolve() or REPO_ROOT.resolve() in root.resolve().parents:
        raise AssertionError(f"refusing to install a draft protocol inside the repository ({root})")
    y, d = root / PROTOCOL_YAML_REL, root / PROTOCOL_DOC_REL
    y.parent.mkdir(parents=True, exist_ok=True)
    d.parent.mkdir(parents=True, exist_ok=True)
    y.write_bytes(draft_protocol_bytes((REPO_ROOT / PROTOCOL_YAML_REL).read_bytes()))
    shutil.copyfile(REPO_ROOT / PROTOCOL_DOC_REL, d)


def add_flood_mapping(root: Path) -> None:
    """SYNTHETIC MACHINERY TESTS ONLY: map the invented 'Flooded area' / 'Flood trace' classes of ``flood=True``
    fixtures. The real EMSR927 products have no such class and the repository protocol maps none."""
    p = root / "config" / "evaluation" / "emsr927_protocol.yaml"
    text = p.read_text(encoding="utf-8")
    old = "    Landslide: reference_mass_movement\n"
    assert old in text
    p.write_text(text.replace(old, old + "    Flooded area: flood_water\n    Flood trace: flood_trace\n"),
                 encoding="utf-8")


def centre_lonlat(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = grid.centre_x(), grid.centre_y()
    lon = np.array([to_lonlat(float(x), float(ys[0]))[0] for x in xs])
    lat = np.array([to_lonlat(float(xs[0]), float(y))[1] for y in ys])
    return np.broadcast_to(lon[None, :], grid.shape), np.broadcast_to(lat[:, None], grid.shape)


def box_mask(grid: Grid, b: tuple[float, float, float, float]) -> np.ndarray:
    """Brute-force reference: pixel centres strictly inside a lon/lat box (rectangles are exact in Mercator)."""
    lon, lat = centre_lonlat(grid)
    return (lon > b[0]) & (lon < b[2]) & (lat > b[1]) & (lat < b[3])


def shapefile_bytes(rings: list[list[tuple[float, float]]], attrs: list[dict[str, str]]) -> dict[str, bytes]:
    """Minimal polygon shapefile (.shp/.dbf/.prj) with one ring per record and string fields."""
    recs = b""
    for i, ring in enumerate(rings, 1):
        pts = b"".join(struct.pack("<2d", x, y) for x, y in ring)
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        content = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), 1, len(ring)) + struct.pack("<i", 0) \
            + pts
        recs += struct.pack(">2i", i, len(content) // 2) + content
    allx = [p[0] for r in rings for p in r]
    ally = [p[1] for r in rings for p in r]
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2) + struct.pack("<2i", 1000, 5) + \
        struct.pack("<8d", min(allx), min(ally), max(allx), max(ally), 0, 0, 0, 0)
    fields = sorted({k for a in attrs for k in a})
    flen = 30
    dbf_h = struct.pack("<4BIHH20x", 3, 126, 1, 1, len(attrs), 32 + 32 * len(fields) + 1, 1 + flen * len(fields))
    for f in fields:
        dbf_h += f.encode().ljust(11, b"\x00") + b"C" + b"\x00" * 4 + bytes([flen, 0]) + b"\x00" * 14
    dbf_h += b"\x0d"
    body = b"".join(b" " + b"".join(a.get(f, "").encode().ljust(flen) for f in fields) for a in attrs) + b"\x1a"
    prj = b'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
          b'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    return {"shp": header + recs, "dbf": dbf_h + body, "prj": prj}


class FakeHTTPS:
    """A fake HTTPS transport for urllib (no network). ``routes`` maps URL -> (status, Location or body). Plugged in
    through ``http_open(..., extra_handlers=(fake,))`` it runs ahead of the real HTTPS handler, so urllib's real
    redirect machinery (and the BATO TrustedRedirects rule) processes its 3xx responses."""

    handler_order = 100

    def __init__(self, routes: dict[str, tuple[int, Any]]) -> None:
        self.routes, self.seen = routes, []

    def add_parent(self, parent: Any) -> None:
        self.parent = parent

    def close(self) -> None:
        pass

    def __lt__(self, other: Any) -> bool:
        return self.handler_order < getattr(other, "handler_order", 500)

    def https_open(self, req: Any) -> Any:
        import email.message
        import urllib.response
        url = req.full_url
        self.seen.append(url)
        status, value = self.routes[url]
        hdrs = email.message.Message()
        if 300 <= status < 400:
            hdrs["Location"] = value
            resp = urllib.response.addinfourl(io.BytesIO(b""), hdrs, url, status)
            resp.msg = "Found"
        else:
            resp = urllib.response.addinfourl(io.BytesIO(value), hdrs, url, status)
            resp.msg = "OK"
        return resp

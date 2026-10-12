"""A small synthetic road world for the connectivity tests (TEST FIXTURE - invented coordinates, not OSM data).

Layout (analysis box 85.00-85.06 E, 28.00-28.06 N; AOI 85.015-85.045 E, 28.015-28.045 N):

* main road along 28.03 N from the west box edge to the east box edge (two exits), with a short
  bridge way (85.035-85.036, over PRE_EVENT_WATER);
* spur 1 north at 85.03 to village B; FLOOD across it at 28.036-28.038 -> B and building cluster D
  (snapped to spur 1 beyond the flood) are CONFIRMED cut off; clearing that one stretch reconnects both;
* spur 2 south at 85.04 to village C; STEEP (NOT_ASSESSED) across it -> C is POSSIBLE cut off;
* spur 3 north at 85.044 to village F with TWO separate flood stretches -> F needs 2 clearings, and
  neither stretch alone reconnects anything;
* town T and village A on the main road (connected), hospital H on the main road, village E with no
  road within 500 m (no pre-event road access).
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bato.acquisition.http import HttpResponse
from bato.config import BatoConfig, load_config
from bato.contracts.baseline import BaselineState
from bato.contracts.common import BBox, PolygonGeometry
from bato.impact.roads import lonlat_to_3857
from bato.osm.extract import extract_osm
from bato.preprocessing.speckle import sha256_file

from helpers import REPO_CONFIG_DIR

EVENT = date(2026, 8, 26)
ANALYSIS = BBox(min_lon=85.00, min_lat=28.00, max_lon=85.06, max_lat=28.06)
AOI = BBox(min_lon=85.015, min_lat=28.015, max_lon=85.045, max_lat=28.045)
AOI_RING = ((85.015, 28.015), (85.045, 28.015), (85.045, 28.045), (85.015, 28.045), (85.015, 28.015))
PIXEL = 10.0
NOW = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)


def _feature(osm_id: str, geometry: dict[str, Any], tags: dict[str, str]) -> dict[str, Any]:
    return {"type": "Feature", "geometry": geometry, "properties": {"@osmId": osm_id, **tags}}


def _line(osm_id: str, coords: list[tuple[float, float]], **tags: str) -> dict[str, Any]:
    return _feature(osm_id, {"type": "LineString", "coordinates": [list(c) for c in coords]}, tags)


def _pt(osm_id: str, lon: float, lat: float, **tags: str) -> dict[str, Any]:
    return _feature(osm_id, {"type": "Point", "coordinates": [lon, lat]}, tags)


def roads() -> list[dict[str, Any]]:
    return [
        _line("way/101", [(85.00, 28.03), (85.02, 28.03), (85.03, 28.03), (85.035, 28.03)],
              highway="primary", name="Main Road"),
        _line("way/102", [(85.035, 28.03), (85.036, 28.03)], highway="primary", bridge="yes", name="Main Road"),
        _line("way/103", [(85.036, 28.03), (85.04, 28.03), (85.044, 28.03), (85.06, 28.03)],
              highway="primary", name="Main Road"),
        _line("way/201", [(85.03, 28.03), (85.03, 28.045)], highway="track"),
        _line("way/202", [(85.04, 28.03), (85.04, 28.018)], highway="track"),
        _line("way/203", [(85.044, 28.03), (85.044, 28.044)], highway="unclassified"),
        _feature("node/9", {"type": "Point", "coordinates": [85.01, 28.01]}, {"highway": "bus_stop"}),
    ]


def places() -> list[dict[str, Any]]:
    return [
        _pt("node/1", 85.02, 28.0303, place="town", name="Town T"),
        _pt("node/2", 85.025, 28.0303, place="village", name="Village A"),
        _pt("node/3", 85.03, 28.0455 - 0.0006, place="village", name="Village B"),
        _pt("node/4", 85.04, 28.0175, place="village", name="Village C"),
        _pt("node/5", 85.02, 28.044, place="hamlet", name="Hamlet E"),
        _pt("node/6", 85.044, 28.0442, place="village", name="Village F"),
        _pt("node/7", 85.05, 28.05, place="village", name="Outside AOI"),
    ]


def hospitals() -> list[dict[str, Any]]:
    return [_pt("way/301", 85.05, 28.0302, amenity="hospital", name="Hospital H")]


def buildings() -> list[dict[str, Any]]:
    out = []
    n = 1000
    for i in range(6):                                   # cluster D (unnamed), east of spur 1
        out.append(_pt(f"way/{n}", 85.033 + 0.0002 * (i % 3), 28.041 + 0.0002 * (i // 3)))
        n += 1
    for i in range(3):                                   # buildings of village B
        out.append(_pt(f"way/{n}", 85.0302 + 0.0002 * i, 28.0449))
        n += 1
    for i in range(4):                                   # buildings of village A
        out.append(_pt(f"way/{n}", 85.025 + 0.0002 * i, 28.0306))
        n += 1
    for i in range(2):                                   # two scattered buildings (too few for a settlement)
        out.append(_pt(f"way/{n}", 85.018 + 0.003 * i, 28.020))
        n += 1
    return out


def wkb(geometry: dict[str, Any]) -> bytes:
    """Little-endian 2D WKB for Point / LineString / Polygon / MultiLineString / MultiPolygon (TEST helper)."""
    def pts(seq: list[list[float]]) -> bytes:
        return struct.pack("<I", len(seq)) + b"".join(struct.pack("<dd", float(x), float(y)) for x, y in seq)

    kind, c = geometry["type"], geometry["coordinates"]
    code = {"Point": 1, "LineString": 2, "Polygon": 3, "MultiLineString": 5, "MultiPolygon": 6}[kind]
    head = struct.pack("<BI", 1, code)
    if kind == "Point":
        return head + struct.pack("<dd", float(c[0]), float(c[1]))
    if kind == "LineString":
        return head + pts(c)
    if kind == "Polygon":
        return head + struct.pack("<I", len(c)) + b"".join(pts(r) for r in c)
    member = "LineString" if kind == "MultiLineString" else "Polygon"
    return head + struct.pack("<I", len(c)) + b"".join(wkb({"type": member, "coordinates": m}) for m in c)


FAKE_PARQUET_MAGIC = b"FAKE-GEOPARQUET\n"


def to_rows(features: list[dict[str, Any]], edited: str = "2026-08-01T00:00:00+00:00",
            valid_to: str = "2222-01-01T00:00:00+00:00") -> list[dict[str, Any]]:
    """GeoJSON-like fixture features -> rows with the documented ohsome v2 columns."""
    rows = []
    for f in features:
        props = f["properties"]
        osm_type, _, osm_id = props["@osmId"].partition("/")
        rows.append({"osm_type": osm_type, "osm_id": int(osm_id), "edit_timestamp": edited,
                     "valid_to_timestamp": valid_to, "version": 1,
                     "tags": [[k, v] for k, v in props.items() if not k.startswith("@")],
                     "geom": wkb(f["geometry"]).hex(), "geom_type": f["geometry"]["type"], "clipped": False})
    return rows


def fake_parquet(rows: list[dict[str, Any]]) -> bytes:
    """A stand-in for the GeoParquet bytes (pyarrow is not needed to run the tests)."""
    return FAKE_PARQUET_MAGIC + json.dumps(rows).encode()


def fake_read_parquet_rows(data: bytes, real: Any = None) -> list[dict[str, Any]]:
    """Decode fake_parquet() bytes like pyarrow's to_pylist() would (datetimes, (k, v) tags, WKB bytes)."""
    if not data.startswith(FAKE_PARQUET_MAGIC):
        if real is None:
            raise ValueError("not fake parquet")
        return real(data)                                       # genuine Parquet: use pyarrow
    rows = json.loads(data[len(FAKE_PARQUET_MAGIC):])
    for r in rows:
        r["edit_timestamp"] = datetime.fromisoformat(r["edit_timestamp"])
        r["valid_to_timestamp"] = datetime.fromisoformat(r["valid_to_timestamp"])
        r["tags"] = [tuple(kv) for kv in r["tags"]]
        r["geom"] = bytes.fromhex(r["geom"])
    return rows


@dataclass
class FakeOhsome:
    """Answers ohsome v2 /metadata and /extraction/features.parquet from the fixture layers; records requests."""

    latest: str = "2026-09-30T20:00:00Z"
    overrides: dict[str, bytes] = field(default_factory=dict)
    status: int = 200
    edited: str = "2026-08-01T00:00:00+00:00"
    requests: list[tuple[str, dict[str, Any], dict[str, str]]] = field(default_factory=list)

    def get(self, url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse:
        self.requests.append((url, {}, dict(headers)))
        body = {"apiVersion": "2.0.0", "attribution": {"text": "(c) OpenStreetMap contributors"},
                "temporalExtent": {"start": "2007-10-08T00:00:00Z", "end": self.latest}}
        return HttpResponse(200, json.dumps(body).encode())

    def post_json(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> HttpResponse:
        self.requests.append((url, dict(body), dict(headers)))
        if not headers.get("authorization"):
            return HttpResponse(401, b'{"error": "missing API key"}')
        f = body["filter"]
        layer = ("roads" if f.startswith("type:way and highway") else "places" if f.startswith("place in")
                 else "hospitals" if "hospital" in f else "buildings")
        if layer in self.overrides:
            return HttpResponse(self.status, self.overrides[layer])
        feats = {"roads": roads, "places": places, "hospitals": hospitals, "buildings": buildings}[layer]()
        return HttpResponse(self.status, fake_parquet(to_rows(feats, edited=self.edited)))


def project(tmp_path: Path) -> BatoConfig:
    shutil.copytree(REPO_CONFIG_DIR, tmp_path / "config", dirs_exist_ok=True)
    return load_config(tmp_path / "config")


def fetch(cfg: BatoConfig, transport: FakeOhsome | None = None, **kw: Any):
    kw.setdefault("api_key", "TEST-KEY")
    return extract_osm(cfg, aoi_id="aoi-test", event_date=EVENT, aoi_bbox=AOI, analysis_bbox=ANALYSIS,
                       transport=transport or FakeOhsome(), now=lambda: NOW, **kw)


def grid() -> tuple[tuple[float, float, float, float, float, float], tuple[int, int]]:
    x0, y1 = lonlat_to_3857(np.array([84.995]), np.array([28.065]))
    x1, y0 = lonlat_to_3857(np.array([85.065]), np.array([27.995]))
    x0f, y1f = math.floor(float(x0[0]) / PIXEL) * PIXEL, math.ceil(float(y1[0]) / PIXEL) * PIXEL
    w = int(math.ceil((float(x1[0]) - x0f) / PIXEL))
    h = int(math.ceil((y1f - float(y0[0])) / PIXEL))
    return (PIXEL, 0.0, x0f, 0.0, -PIXEL, y1f), (h, w)


def _box(state: np.ndarray, transform: tuple[float, ...], lon0: float, lat0: float, lon1: float, lat1: float,
         value: int) -> None:
    a, _b, c, _d, e, f = transform
    xs, ys = lonlat_to_3857(np.array([lon0, lon1]), np.array([lat0, lat1]))
    c0, c1 = sorted(int(math.floor((x - c) / a)) for x in xs)
    r0, r1 = sorted(int(math.floor((y - f) / e)) for y in ys)
    state[r0:r1 + 1, c0:c1 + 1] = value


def state_raster() -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    transform, (h, w) = grid()
    s = np.zeros((h, w), dtype=np.uint8)
    flood, steep, water = int(BaselineState.FLOOD), int(BaselineState.NOT_ASSESSED_STEEP), \
        int(BaselineState.PRE_EVENT_WATER)
    _box(s, transform, 85.0295, 28.036, 85.0305, 28.038, flood)          # spur 1
    _box(s, transform, 85.0395, 28.022, 85.0405, 28.026, steep)          # spur 2
    _box(s, transform, 85.0435, 28.034, 85.0445, 28.035, flood)          # spur 3, first
    _box(s, transform, 85.0435, 28.039, 85.0445, 28.040, flood)          # spur 3, second
    _box(s, transform, 85.0349, 28.0295, 85.0361, 28.0305, water)        # river under the bridge
    _box(s, transform, 85.05, 28.05, 85.06, 28.06, steep)                # terrain off the roads
    return s, transform


def fake_job(cfg: BatoConfig, extract: Any, root: Path, state: np.ndarray) -> SimpleNamespace:
    """A ConnectivityJob stand-in: the real one needs Step 3.5 / 4C files that only rasterio can write."""
    transform, (h, w) = grid()
    state_dir = cfg.settings.paths.interim / "detection" / "aoi-test" / "ts-test" / "baseline"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "baseline_state.tif"
    state_path.write_bytes(state.tobytes())            # stand-in bytes; hashed like the real file
    prov_path = state_dir / "baseline_provenance.json"
    prov_path.write_text("{}", encoding="utf-8")
    stack_prov = cfg.settings.paths.interim / "preprocessing" / "aoi-test" / "stack" / "ts-test" / "p.json"
    stack_prov.parent.mkdir(parents=True, exist_ok=True)
    stack_prov.write_text("{}", encoding="utf-8")
    t0 = datetime(2026, 6, 29, 12, 21, 38, tzinfo=timezone.utc)
    inputs = [SimpleNamespace(role="PRE1", product_name="TEST-S1D-PRE1", acquisition_start=t0),
              SimpleNamespace(role="PRE2", product_name="TEST-S1D-PRE2",
                              acquisition_start=datetime(2026, 8, 16, 12, 21, 41, tzinfo=timezone.utc)),
              SimpleNamespace(role="POST", product_name="TEST-S1D-POST",
                              acquisition_start=datetime(2026, 8, 28, 12, 21, 41, tzinfo=timezone.utc))]
    rel = lambda p: p.resolve().relative_to(root.resolve()).as_posix()  # noqa: E731
    stack = SimpleNamespace(grid=SimpleNamespace(height=h, width=w, transform=transform, pixel_size_m=PIXEL,
                                                 analysis_bbox=ANALYSIS, aoi_bbox=AOI),
                            aoi_polygon=PolygonGeometry(coordinates=(AOI_RING,)), inputs=inputs,
                            event_date=EVENT.isoformat())
    baseline = SimpleNamespace(
        outputs=SimpleNamespace(state=rel(state_path), state_sha256=sha256_file(state_path)),
        source=SimpleNamespace(provenance=rel(stack_prov), provenance_sha256=sha256_file(stack_prov)),
        config_fingerprint="b" * 64)
    final = cfg.settings.paths.processed / "connectivity" / "aoi-test" / "ts-test"
    return SimpleNamespace(root=root, aoi_id="aoi-test", candidate_id="ts-test", event_date=EVENT,
                           baseline_provenance_path=prov_path, baseline_provenance_sha256=sha256_file(prov_path),
                           baseline=baseline, state_path=state_path, stack_provenance_path=stack_prov, stack=stack,
                           osm=extract, final_dir=final)

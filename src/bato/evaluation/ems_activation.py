"""EVALUATION ONLY. The Copernicus EMS Rapid Mapping public activation record (source of truth for products).

The record is the JSON returned by ``/backend/dashboard-api/public-activations/?code=<CODE>``. Product download
URLs are taken from each product's ``downloadPath``, never built from a filename pattern. Times are kept as the
raw strings of the record: it carries no time-zone marker (see :mod:`bato.evaluation.timegap`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from bato.evaluation.geometry import GeometryError, Polygon, intersection_area_km2, parse_wkt_polygons, polygon_area_km2


class ActivationRecordError(ValueError):
    pass


STATUS_LABEL = {"F": "final", "I": "in progress", "W": "waiting", "N": "not feasible"}


@dataclass(frozen=True)
class EmsImage:
    sensor: str | None
    acquisition_time: str | None          # raw string from the record (no zone marker)


@dataclass(frozen=True)
class EmsProduct:
    aoi_number: int
    aoi_name: str
    type: str
    monitoring: bool | None
    monitoring_number: int | None
    feasible: bool | None
    version: int | None
    status_code: str | None
    delivery_time: str | None
    download_path: str
    images: tuple[EmsImage, ...]

    @property
    def filename(self) -> str | None:
        return PurePosixPath(urlparse(self.download_path).path).name if self.download_path else None

    @property
    def label(self) -> str:
        kind = f"MONIT{self.monitoring_number:02d}" if self.monitoring and self.monitoring_number else "PRODUCT"
        return f"AOI{self.aoi_number:02d} {self.type} {kind} v{self.version}"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["filename"] = self.filename
        d["label"] = self.label
        return d


@dataclass(frozen=True)
class EmsAoi:
    number: int
    name: str
    extent_wkt: str | None
    products: tuple[EmsProduct, ...] = field(default_factory=tuple)

    def polygons(self) -> list[Polygon]:
        if not self.extent_wkt:
            return []
        return parse_wkt_polygons(self.extent_wkt)


@dataclass(frozen=True)
class ActivationRecord:
    code: str
    name: str | None
    event_time: str | None
    activation_time: str | None
    closed: bool | None
    countries: tuple[str, ...]
    aois: tuple[EmsAoi, ...]
    top_level_keys: tuple[str, ...]

    def products(self) -> list[EmsProduct]:
        return [p for a in self.aois for p in a.products]


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ActivationRecordError(f"{where}: field {key!r} missing from the activation record")
    return d[key]


def _int_or_none(v: Any) -> int | None:
    return None if v is None or v == "" else int(v)


def parse_activation(payload: dict[str, Any], code: str) -> ActivationRecord:
    """Parse the public-activations response. Fails closed on a missing field or an unexpected shape."""
    results = _req(payload, "results", "response")
    matches = [r for r in results if isinstance(r, dict) and r.get("code") == code]
    if len(matches) != 1:
        raise ActivationRecordError(f"expected exactly one activation {code}, found {len(matches)}")
    act = matches[0]
    aois = []
    for i, a in enumerate(_req(act, "aois", code)):
        where = f"{code} aois[{i}]"
        number, name = int(_req(a, "number", where)), str(_req(a, "name", where))
        prods = []
        for j, p in enumerate(_req(a, "products", where)):
            pw = f"{where}.products[{j}]"
            ver = p.get("version") or {}
            images = tuple(EmsImage(sensor=im.get("sensorName"), acquisition_time=im.get("acquisitionTime"))
                           for im in (p.get("images") or []))
            prods.append(EmsProduct(
                aoi_number=number, aoi_name=name, type=str(_req(p, "type", pw)),
                monitoring=p.get("monitoring"), monitoring_number=_int_or_none(p.get("monitoringNumber")),
                feasible=p.get("feasible"), version=_int_or_none(ver.get("number")),
                status_code=ver.get("statusCode"), delivery_time=ver.get("deliveryTime"),
                download_path=str(p.get("downloadPath") or ""), images=images))
        aois.append(EmsAoi(number=number, name=name, extent_wkt=a.get("extent"), products=tuple(prods)))
    countries = tuple(str(c.get("name")) for c in (act.get("countries") or []) if isinstance(c, dict))
    return ActivationRecord(code=code, name=act.get("name"), event_time=act.get("eventTime"),
                            activation_time=act.get("activationTime"), closed=act.get("closed"), countries=countries,
                            aois=tuple(aois), top_level_keys=tuple(sorted(act)))


def check_download_url(url: str, allowed_host: str, code: str) -> str:
    """The URL must come from the record, be HTTPS on the EMS host, point to a .zip of this activation."""
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != allowed_host:
        raise ActivationRecordError(f"download URL {url!r} is not https://{allowed_host}/...")
    name = PurePosixPath(u.path).name
    if not name.lower().endswith(".zip") or not name.upper().startswith(f"{code.upper()}_"):
        raise ActivationRecordError(f"download URL {url!r} is not a {code} zip")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ActivationRecordError(f"unsafe file name {name!r}")
    return name


@dataclass(frozen=True)
class AoiOverlap:
    number: int
    name: str
    extent_area_km2: float | None
    overlap_km2: float | None
    overlap_fraction_of_ems_aoi: float | None
    note: str | None = None


def aoi_overlaps(record: ActivationRecord, bato_ring: list[tuple[float, float]]) -> list[AoiOverlap]:
    """Overlap of each EMS AOI extent (from the record) with the BATO AOI (convex ring)."""
    out = []
    for a in record.aois:
        try:
            polys = a.polygons()
        except GeometryError as exc:
            out.append(AoiOverlap(a.number, a.name, None, None, None, f"extent not parsed: {exc}"))
            continue
        if not polys:
            out.append(AoiOverlap(a.number, a.name, None, None, None, "no extent in the record"))
            continue
        area = sum(polygon_area_km2(p) for p in polys)
        inter = intersection_area_km2(polys, bato_ring)
        out.append(AoiOverlap(a.number, a.name, round(area, 4), round(inter, 4),
                              round(inter / area, 6) if area > 0 else None))
    return out


@dataclass(frozen=True)
class DownloadDecision:
    product: EmsProduct
    download: bool
    reason: str


def plan_downloads(record: ActivationRecord, overlaps: list[AoiOverlap], *, required_status: str,
                   allowed_host: str) -> list[DownloadDecision]:
    by_aoi = {o.number: o for o in overlaps}
    out = []
    for p in record.products():
        ov = by_aoi.get(p.aoi_number)
        if p.status_code != required_status:
            out.append(DownloadDecision(p, False, f"status {p.status_code} "
                                                  f"({STATUS_LABEL.get(str(p.status_code), 'unknown')}), not final"))
        elif not p.download_path:
            out.append(DownloadDecision(p, False, "no downloadPath in the activation record"))
        elif ov is None or ov.overlap_km2 is None:
            out.append(DownloadDecision(p, False, "AOI extent unavailable; overlap with BATO AOI unknown"))
        elif ov.overlap_km2 <= 0:
            out.append(DownloadDecision(p, False, "EMS AOI does not overlap the BATO AOI"))
        else:
            check_download_url(p.download_path, allowed_host, record.code)
            out.append(DownloadDecision(p, True, f"final; overlaps the BATO AOI by {ov.overlap_km2} km2"))
    return out

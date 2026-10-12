"""EVALUATION ONLY. Time gaps between EMS reference imagery and the BATO POST Sentinel-1 acquisition.

The EMS activation record gives times without a zone marker. They are not assumed to be UTC: while the protocol
leaves ``ems_time_zone`` undetermined, every gap is computed under each configured reading (UTC and
Asia/Kathmandu, UTC+05:45, by default).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from bato.evaluation.ems_activation import EmsProduct


class TimeGapError(ValueError):
    pass


def parse_offset(text: str) -> timezone:
    sign = -1 if text.startswith("-") else 1
    hh, mm = text.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def parse_ems_time(raw: str, tz: timezone) -> datetime:
    """A naive EMS timestamp read in zone ``tz``. A string that carries its own zone keeps it."""
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def has_zone(raw: str) -> bool:
    s = raw.strip()
    if s.endswith("Z"):
        return True
    return datetime.fromisoformat(s).tzinfo is not None


def readings(protocol_time: Mapping[str, Any]) -> dict[str, timezone]:
    """Zones to evaluate: the determined one only, or every configured reading."""
    configured = {name: parse_offset(off) for name, off in dict(protocol_time["readings"]).items()}
    zone = protocol_time.get("ems_time_zone", "undetermined")
    if zone == "undetermined":
        return configured
    if zone not in configured:
        raise TimeGapError(f"ems_time_zone {zone!r} is not one of the configured readings {sorted(configured)}")
    if not protocol_time.get("ems_time_zone_evidence"):
        raise TimeGapError("ems_time_zone is set but ems_time_zone_evidence is empty; quote the official source")
    return {zone: configured[zone]}


def hours(delta: timedelta) -> float:
    return round(delta.total_seconds() / 3600.0, 3)


def product_gaps(product: EmsProduct, post: datetime, zones: Mapping[str, timezone]) -> dict[str, Any]:
    """EMS image time - BATO POST, in hours (negative = EMS image earlier), per image and reading."""
    if post.tzinfo is None:
        raise TimeGapError("BATO POST time must be timezone-aware (UTC)")
    rows: list[dict[str, Any]] = []
    for im in product.images:
        if not im.acquisition_time:
            rows.append({"sensor": im.sensor, "acquisition_time_raw": None, "gap_hours": None})
            continue
        gaps = {name: hours(parse_ems_time(im.acquisition_time, tz) - post) for name, tz in zones.items()}
        rows.append({"sensor": im.sensor, "acquisition_time_raw": im.acquisition_time,
                     "zone_in_string": has_zone(im.acquisition_time), "gap_hours": gaps})
    worst: dict[str, float | None] = {}
    for name in zones:
        vals: list[float] = [abs(r["gap_hours"][name]) for r in rows if r["gap_hours"] is not None]
        undated = any(r["gap_hours"] is None for r in rows)
        worst[name] = None if undated or not vals else max(vals)       # an undated image makes the gap unknown
    return {"product": product.label, "images": rows, "worst_case_abs_gap_hours": worst}


def select_product(candidates: Sequence[EmsProduct], post: datetime,
                   zones: Mapping[str, timezone]) -> tuple[EmsProduct, dict[str, Any]]:
    """Pre-registered rule: smallest worst-case |image time - POST|; ties -> higher monitoring number, then
    version. Must give the same product under every reading, otherwise it fails closed."""
    if not candidates:
        raise TimeGapError("no candidate product")
    picks: dict[str, str] = {}
    chosen: dict[str, EmsProduct] = {}
    table = [product_gaps(p, post, zones) for p in candidates]
    for name in zones:
        scored = []
        for p, g in zip(candidates, table, strict=True):
            w = g["worst_case_abs_gap_hours"][name]
            if w is None:
                continue
            scored.append((w, -(p.monitoring_number or 0), -(p.version or 0), p))
        if not scored:
            raise TimeGapError(f"no candidate has image times (AOI{candidates[0].aoi_number:02d})")
        scored.sort(key=lambda t: t[:3])
        chosen[name] = scored[0][3]
        picks[name] = scored[0][3].label
    if len(set(picks.values())) != 1:
        raise TimeGapError(f"product selection depends on the time-zone reading: {picks}; set ems_time_zone from "
                           "official evidence before evaluating")
    first = next(iter(chosen.values()))
    return first, {"rule": "min_worst_case_abs_gap_to_post", "candidates": table, "selected": first.label,
                   "selected_under": picks}


def event_dates(event_time_raw: str | None, zones: Mapping[str, timezone]) -> dict[str, Any]:
    """Calendar date of the EMS eventTime under each reading, in UTC and in Nepal time."""
    if not event_time_raw:
        return {"event_time_raw": None}
    npt = timezone(timedelta(hours=5, minutes=45))
    out: dict[str, Any] = {"event_time_raw": event_time_raw}
    for name, tz in zones.items():
        dt = parse_ems_time(event_time_raw, tz)
        out[name] = {"utc": dt.astimezone(timezone.utc).isoformat(), "utc_date": dt.astimezone(timezone.utc).date()
                     .isoformat(), "nepal_date": dt.astimezone(npt).date().isoformat()}
    return out

"""EVALUATION ONLY. EMSR927 case-study comparison: inspection (no metrics) and the frozen-protocol evaluation.

Order enforced by the code:

1. inspect: product contents only;
2. freeze: the protocol, after checking every inspected layer and class is mapped;
3. evaluate: metrics.

The BATO result is read, never written. Its files are hashed before and after the evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bato.config import BatoConfig
from bato.data_rules import assert_reference_input_allowed
from bato.evaluation import leakage
from bato.evaluation.acquisition import CODE, load_manifest, reference_dir, sha256_file, verify_downloads
from bato.evaluation.ems_activation import EmsImage, EmsProduct, parse_activation
from bato.evaluation.ems_product import ProductZip, inspect_product
from bato.evaluation.geometry import (
    GeometryError,
    Grid,
    bbox,
    geojson_polygons,
    intersection_area_km2,
    iter_coords,
    polygon_area_km2,
    project_ring,
    rasterize,
)
from bato.evaluation.metrics import NOT_CALCULABLE, footprint_overlap, pixel_comparison, ratio, scores
from bato.evaluation.official import (
    OfficialBato,
    StateReader,
    load_official,
    official_road_geometry,
    rasterio_state_reader,
    rehash,
)
from bato.evaluation.protocol import (
    COMPARED,
    FREEZE_FILE,
    REMOVED_FROM_AREA,
    Protocol,
    ProtocolError,
    check_frozen,
    check_mapping_complete,
    freeze,
    load_protocol,
)
from bato.evaluation.roads import alignment_diagnostic, in_area, road_comparison
from bato.evaluation.timegap import event_dates, product_gaps, readings, select_product
from bato.exceptions import BatoError

RUNS_FILE = "runs.jsonl"
INSPECTION_DIR = "inspection"


class EvaluationError(RuntimeError):
    pass


def eval_dir(cfg: BatoConfig) -> Path:
    d = cfg.settings.paths.eval / CODE
    assert_reference_input_allowed(d, cfg.settings.paths)
    return d


def _product_from_manifest(d: dict[str, Any]) -> EmsProduct:
    return EmsProduct(aoi_number=int(d["aoi"]), aoi_name=str(d["aoi_name"]), type=str(d["type"]),
                      monitoring=d.get("monitoring"), monitoring_number=d.get("monitoring_number"), feasible=True,
                      version=d.get("version"), status_code=d.get("status_code"),
                      delivery_time=d.get("delivery_time_raw"), download_path=str(d["url"]),
                      images=tuple(EmsImage(i.get("sensor"), i.get("acquisition_time_raw")) for i in d["images"]))


def _post_time(manifest: dict[str, Any]) -> datetime:
    raw = manifest["bato_aoi"]["post_acquisition_start"]
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise EvaluationError("BATO POST time has no zone")
    return dt


def selected_products(manifest: dict[str, Any], protocol: Protocol, post: datetime) -> tuple[list[dict[str, Any]],
                                                                                          list[dict[str, Any]]]:
    """Per EMS AOI, the downloaded product chosen by the pre-registered rule (image times only, no content)."""
    zones = readings(protocol.raw["time"])
    by_aoi: dict[int, list[dict[str, Any]]] = {}
    for d in manifest["downloads"]:
        if d.get("downloaded"):
            by_aoi.setdefault(int(d["aoi"]), []).append(d)
    chosen, selection = [], []
    for aoi, entries in sorted(by_aoi.items()):
        prods = [_product_from_manifest(e) for e in entries]
        pick, info = select_product(prods, post, zones)
        chosen.append(next(e for e, p in zip(entries, prods, strict=True) if p is pick))
        selection.append({"aoi": aoi, **info})
    return chosen, selection


# ---------------------------------------------------------------------------
# Inspection (no metric)
# ---------------------------------------------------------------------------


def inspect(cfg: BatoConfig, protocol: Protocol) -> dict[str, Any]:
    paths = cfg.settings.paths
    manifest = load_manifest(paths)
    assert manifest is not None
    zips = verify_downloads(paths, manifest)
    base = reference_dir(paths)
    ring = [(float(x), float(y)) for x, y in manifest["bato_aoi"]["aoi_ring"]]
    post = _post_time(manifest)
    zones = readings(protocol.raw["time"])
    act = json.loads((base / "activation" / manifest["activation"]["file"]).read_text(encoding="utf-8"))
    record = parse_activation(act, CODE)
    products = []
    for z in zips:
        rep = inspect_product(z)
        rep["sha256"] = sha256_file(z)
        entry = next(d for d in manifest["downloads"] if d.get("file") == f"products/{z.name}")
        rep["label"] = entry["label"]
        rep["time_gaps"] = product_gaps(_product_from_manifest(entry), post, zones)
        pz = ProductZip(z)
        for layer in rep["layers"]:
            role = _role_or_none(protocol, layer["stem"])
            layer["protocol_role"] = role
            if layer["stem"] in (protocol.raw["layers"].get("area_of_interest") or []) and layer.get("wgs84_lonlat"):
                polys = _polygons(pz.load_key((layer["product"], layer["stem"], layer["version"])).features)
                layer["area_km2"] = round(sum(polygon_area_km2(p) for p in polys), 4)
                layer["overlap_with_bato_aoi_km2"] = round(intersection_area_km2(polys, ring), 4)
            layer["suitability_note"] = _suitability(layer, role)
        products.append(rep)
    try:
        chosen, selection = selected_products(manifest, protocol, post)
        sel: dict[str, Any] = {"compared": [c["label"] for c in chosen], "detail": selection}
    except ValueError as exc:
        sel = {"error": str(exc)}
    report = {"kind": "emsr927_inspection", "metrics_computed": False, "generated_at_utc": _now(),
              "activation": manifest["activation"], "event_time_raw": record.event_time,
              "activation_time_raw": record.activation_time, "event_dates": event_dates(record.event_time, zones),
              "bato_post_utc": post.isoformat(), "bato_event_date": manifest["bato_aoi"].get("event_date"),
              "product_selection": sel, "products": products,
              "not_downloaded": manifest.get("not_downloaded", [])}
    if "compared" in sel:
        report["road_geometry_check"] = _road_geometry_check(cfg, protocol, manifest, chosen)
        report["mapping_problems_for_freeze"] = check_mapping_complete(
            protocol, report, {Path(c["file"]).name for c in chosen})
    out = base / INSPECTION_DIR
    assert_reference_input_allowed(out, paths)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspection_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "inspection_report.md").write_text(render_inspection(report), encoding="utf-8")
    return report


def _road_geometry_check(cfg: BatoConfig, protocol: Protocol, manifest: dict[str, Any],
                         chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """Per compared product: is the pre-registered road-matching rule suited to the actual geometry? Uses EMS road
    lines (grouped by their raw damage value) and BATO road GEOMETRY only; no BATO road state is read."""
    stems = set(protocol.raw["layers"].get("transportation") or [])
    field = protocol.raw["transportation"].get("damage_field")
    try:
        geo = official_road_geometry(cfg, manifest["bato_aoi"]["aoi_id"], manifest["bato_aoi"]["candidate_id"])
    except (ValueError, KeyError, OSError, BatoError) as exc:
        return {"error": f"BATO road geometry unavailable: {exc}"}
    grid: Grid = geo["grid"]
    bato_aoi = rasterize([[geo["aoi_ring"]]], grid)
    out: dict[str, Any] = {"reads_bato_road_state": False, "segments_sha256": geo["segments_sha256"], "products": {}}
    for entry in chosen:
        pz = ProductZip(reference_dir(cfg.settings.paths) / entry["file"])
        try:
            aoi_stem = next(s for s in pz.stems() if s in (protocol.raw["layers"].get("area_of_interest") or []))
            area = rasterize(_polygons(pz.load_lonlat(aoi_stem).features), grid) & bato_aoi
            feats, values = [], []
            for stem in sorted(pz.stems() & stems):
                for f in pz.load_lonlat(stem).features:
                    feats.append(f)
                    values.append(str((f.get("properties") or {}).get(field)))
            if not feats:
                out["products"][entry["label"]] = {"value": NOT_CALCULABLE, "reason": "no mapped road layer"}
                continue
            out["products"][entry["label"]] = alignment_diagnostic(
                geo["lines"], feats, values, lambda pts, a=area: in_area(pts, a, grid), protocol.raw["roads"])
        except (ValueError, StopIteration, KeyError) as exc:
            out["products"][entry["label"]] = {"error": str(exc)}
    return out


def _role_or_none(protocol: Protocol, stem: str) -> str | None:
    try:
        return protocol.layer_role(stem)
    except ProtocolError:
        return None


def _suitability(layer: dict[str, Any], role: str | None) -> str:
    gt = layer.get("geometry_types") or {}
    areal = all(k in ("Polygon", "MultiPolygon") for k in gt) and bool(gt)
    if layer.get("error"):
        return f"not readable: {layer['error']}"
    if not layer.get("wgs84_lonlat", False):
        return "not WGS84 lon/lat: cannot be compared without reprojection (refused)"
    if role is None:
        return "unmapped in the protocol: decide its role before freezing"
    if role == "observed_event":
        return ("areal event delineation: comparable on the 10 m grid, subject to the class mapping"
                if areal else "observed-event layer is not purely areal: not comparable as pixels")
    if role == "area_of_interest":
        return "defines the EMS comparison area"
    if role == "transportation":
        return "road lines with a damage grade: section G" if not areal else "not a line layer"
    return f"role {role}"


def render_inspection(r: dict[str, Any]) -> str:
    L = ["# EMSR927 inspection report (no metrics)", "",
         f"Activation record: `{r['activation']['file']}`, SHA-256 `{r['activation']['sha256']}`, retrieved "
         f"{r['activation']['retrieved_at_utc']} UTC.", "",
         f"eventTime (raw): `{r['event_time_raw']}`; activationTime (raw): `{r['activation_time_raw']}`; "
         f"BATO POST (UTC): `{r['bato_post_utc']}`; BATO event date: `{r['bato_event_date']}`.", ""]
    for p in r["products"]:
        L += [f"## {p['label']} — `{p['zip']}`", "", f"SHA-256 `{p['sha256']}`", "",
              "| Layer | Role | Format | CRS | Features | Geometry | Note |", "|---|---|---|---|---|---|---|"]
        for ly in p["layers"]:
            L.append(f"| {ly['stem']} v{ly['version']} | {ly.get('protocol_role')} | {ly.get('loaded_from')} | "
                     f"{ly.get('crs')} | {ly.get('features')} | {ly.get('geometry_types')} | "
                     f"{ly.get('suitability_note')} |")
        L.append("")
        for ly in p["layers"]:
            for k, v in (ly.get("properties") or {}).items():
                if "values" in v and v["distinct"] <= 15:
                    vals = ", ".join(f"`{x['value']}` ({x['count']})" for x in v["values"])
                    L.append(f"- {ly['stem']}.{k}: {vals}")
        L += ["", f"Rasters: {len(p['rasters'])}; documents: {len(p['documents'])}; time-zone clues: "
                  f"{len(p['time_zone_clues'])}", ""]
        for c in p["time_zone_clues"][:10]:
            L.append(f"- `{c['where']}`: …{c['context']}…")
        L.append("")
    L += ["## Product selection (image times only)", "", "```", json.dumps(r["product_selection"], indent=1,
                                                                         default=str)[:4000], "```", ""]
    geo = r.get("road_geometry_check")
    if geo:
        L += ["## Road geometry check (geometry only; no BATO road state read)", ""]
        if "error" in geo:
            L += [f"- {geo['error']}", ""]
        for label, g in (geo.get("products") or {}).items():
            if "share_of_ems_road_length_within_distance_of_bato_roads" not in g:
                L.append(f"- {label}: {g.get('reason') or g.get('error')}")
                continue
            share = g["share_of_ems_road_length_within_distance_of_bato_roads"]
            L.append(f"- {label}: EMS road length in area by value {g['ems_road_length_in_area_m_by_grade']} m; share "
                     f"within 10/20/30/50 m of BATO roads {share};"
                     f" BATO segments in area {g['bato_segments_in_area']}, median length "
                     f"{g['bato_segment_length_m']['median']} m, shorter than {g['min_matched_length_m']} m: "
                     f"{g['bato_segments_shorter_than_min_matched_length']}")
        L.append("")
    if "mapping_problems_for_freeze" in r:
        L += ["## Mapping problems that block the freeze", ""]
        L += [f"- {x}" for x in r["mapping_problems_for_freeze"]] or ["- none"]
    return "\n".join(L) + "\n"


def _polygons(features: list[dict[str, Any]]) -> list[list[list[tuple[float, float]]]]:
    out = []
    for f in features:
        out.extend(geojson_polygons(f.get("geometry")))
    return out


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def freeze_protocol(cfg: BatoConfig, protocol: Protocol) -> dict[str, Any]:
    paths = cfg.settings.paths
    insp = reference_dir(paths) / INSPECTION_DIR / "inspection_report.json"
    if not insp.is_file():
        raise ProtocolError("no inspection report: run scripts/inspect_emsr927.py before freezing")
    report = json.loads(insp.read_text(encoding="utf-8"))
    manifest = load_manifest(paths)
    assert manifest is not None
    verify_downloads(paths, manifest)
    if {p["sha256"] for p in report["products"]} != {d["sha256"] for d in manifest["downloads"] if d.get("downloaded")}:
        raise ProtocolError("the inspection report does not describe the currently downloaded products; re-inspect")
    chosen, _ = selected_products(manifest, protocol, _post_time(manifest))
    problems = check_mapping_complete(protocol, report, {Path(c["file"]).name for c in chosen})
    if problems:
        raise ProtocolError("cannot freeze, unmapped items:\n  " + "\n  ".join(problems))
    d = eval_dir(cfg)
    runs = d / RUNS_FILE
    prior = ([json.loads(x) for x in runs.read_text(encoding="utf-8").splitlines() if x.strip()]
             if runs.is_file() else [])
    for m in sorted(d.glob("*/metrics*.json")) if d.is_dir() else []:
        meta = json.loads(m.read_text(encoding="utf-8")).get("protocol", {})
        prior.append({"protocol_id": meta.get("protocol_id"), "protocol_sha256": meta.get("combined_sha256"),
                      "finished_at_utc": None, "file": m.relative_to(d).as_posix()})
    if any(r.get("protocol_id") == protocol.protocol_id for r in prior):
        raise ProtocolError(f"metrics were already computed under {protocol.protocol_id}; a changed protocol "
                            "needs a new protocol_id")
    rec = freeze(protocol, d)
    if prior:
        rec["prior_runs_disclosed"] = prior
        (d / FREEZE_FILE).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _event_masks(protocol: Protocol, features: list[dict[str, Any]], grid: Grid) -> tuple[dict[str, np.ndarray],
                                                                                         dict[str, Any]]:
    field = protocol.raw["observed_event"]["class_field"]
    by_cat: dict[str, list[Any]] = {}
    present: dict[str, dict[str, int]] = {}
    for f in features:
        props = f.get("properties") or {}
        if field not in props:
            raise ProtocolError(f"observed-event feature without the class field {field!r}")
        value = props[field]
        cat = protocol.event_category(value)
        try:
            polys = geojson_polygons(f.get("geometry"))
        except GeometryError as exc:
            raise ProtocolError(f"observed-event feature {value!r} is not areal ({exc}); only areal events can be "
                                "compared with a pixel map") from exc
        by_cat.setdefault(cat, []).extend(polys)
        present.setdefault(cat, {}).setdefault(str(value), 0)
        present[cat][str(value)] += 1
    masks = {cat: rasterize(polys, grid) for cat, polys in by_cat.items()}
    return masks, present


def _evaluate_product(cfg: BatoConfig, protocol: Protocol, entry: dict[str, Any], bato: OfficialBato,
                      bato_aoi_mask: np.ndarray) -> dict[str, Any]:
    zpath = reference_dir(cfg.settings.paths) / entry["file"]
    pz = ProductZip(zpath)
    grid = bato.grid
    roles: dict[str, list[str]] = {}
    for stem in sorted(pz.stems()):
        roles.setdefault(protocol.layer_role(stem), []).append(stem)
    if len(roles.get("area_of_interest", [])) != 1 or len(roles.get("observed_event", [])) != 1:
        raise ProtocolError(f"{zpath.name}: needs exactly one area_of_interest and one observed_event layer, "
                            f"found {roles}")
    aoi_layer = pz.load_lonlat(roles["area_of_interest"][0])
    aoi_polys = _polygons(aoi_layer.features)
    ems_aoi = rasterize(aoi_polys, grid)
    notan = np.zeros(grid.shape, dtype=bool)
    for stem in roles.get("not_analysed", []):
        notan |= rasterize(_polygons(pz.load_lonlat(stem).features), grid)
    ev_layer = pz.load_lonlat(roles["observed_event"][0])
    masks, present = _event_masks(protocol, ev_layer.features, grid)
    removed = np.zeros(grid.shape, dtype=bool)
    for cat in REMOVED_FROM_AREA:
        removed |= masks.get(cat, np.zeros(grid.shape, dtype=bool))
    base_area = ems_aoi & bato_aoi_mask & ~notan & ~removed
    if not (ems_aoi & bato_aoi_mask).any():
        raise EvaluationError(f"{zpath.name}: the product's AOI layer covers no BATO AOI pixel although the record "
                              "extent overlaps it; check the layer CRS / axis order")
    aoi_bb = bbox(aoi_polys)
    for f in ev_layer.features:
        polys = geojson_polygons(f.get("geometry"))
        if polys:
            b = bbox(polys)
            if b[2] < aoi_bb[0] or b[0] > aoi_bb[2] or b[3] < aoi_bb[1] or b[1] > aoi_bb[3]:
                raise EvaluationError(f"{zpath.name}: an observed-event polygon lies outside the product AOI bbox; "
                                      "check the layer CRS / axis order")
    out: dict[str, Any] = {
        "product": entry["label"], "zip": Path(entry["file"]).name, "zip_sha256": entry["sha256"],
        "layers_used": {k: v for k, v in roles.items()},
        "ems_aoi_area_km2": round(sum(polygon_area_km2(p) for p in aoi_polys), 4),
        "ems_aoi_overlap_with_bato_aoi_km2": round(intersection_area_km2(aoi_polys, bato.aoi_ring), 4),
        "pixels": {"ems_aoi": int(ems_aoi.sum()), "ems_aoi_and_bato_aoi": int((ems_aoi & bato_aoi_mask).sum()),
                   "not_analysed_removed": int((ems_aoi & bato_aoi_mask & notan).sum()),
                   "flood_trace_or_other_event_removed": int((ems_aoi & bato_aoi_mask & removed).sum()),
                   "comparison_area_before_class_exclusion": int(base_area.sum()),
                   "pixel_size_note": "10 m in EPSG:3857 units, about 10 x cos(lat) m on the ground"},
        "event_classes_present": present,
        "event_pixels_in_bato_aoi_by_category": {c: int((m & ems_aoi & bato_aoi_mask).sum()) for c, m in masks.items()},
        "comparisons": {}}
    for cat, meta in COMPARED.items():
        if cat not in present:
            out["comparisons"][cat] = {**meta, "value": NOT_CALCULABLE,
                                       "reason": f"the product contains no observed-event class mapped to {cat}"}
            continue
        other = np.zeros(grid.shape, dtype=bool)
        for c2 in COMPARED:
            if c2 != cat:
                other |= masks.get(c2, np.zeros(grid.shape, dtype=bool))
        area = base_area & ~other
        if meta["class_match"]:
            res = pixel_comparison(bato.state, masks[cat], area,
                                   tolerance_px=int(protocol.raw["pixels"]["tolerance_px"]),
                                   row_ground_area_m2=grid.row_ground_area_m2())
            res["label"] = "class-matched comparison (EMS open water vs BATO open-water darkening)"
        else:
            res = footprint_overlap(bato.state, masks[cat], area, row_ground_area_m2=grid.row_ground_area_m2())
        res["other_compared_category_pixels_removed"] = int((base_area & other).sum())
        out["comparisons"][cat] = {**meta, **res}
    trans = roles.get("transportation", [])
    if not trans:
        out["roads"] = {"value": NOT_CALCULABLE, "reason": "no road-damage line layer is mapped for this product"}
    else:
        feats, grades = [], []
        field = protocol.raw["transportation"]["damage_field"]
        for stem in trans:
            for f in pz.load_lonlat(stem).features:
                props = f.get("properties") or {}
                if field not in props:
                    raise ProtocolError(f"{zpath.name}: {stem} feature without the damage field {field!r}")
                feats.append(f)
                grades.append(protocol.road_grade(props[field]))
        out["roads"] = road_comparison(bato.segments, feats, grades, base_area, grid, protocol.raw["roads"],
                                       protocol.road_comparisons())
        out["roads"]["mismatch"] = ("EMS road damage is physical damage seen in optical imagery; BATO BLOCKED is a "
                                    ">= 25 m run of Step 4C FLOOD (open-water-like radar darkening) on the road. A "
                                    "road destroyed or buried by a debris flow need not look like open water.")
    out["bridges"] = _bridge_context(protocol, pz, roles.get("bridge_context", []), base_area, grid)
    out["_areas"] = (ems_aoi & bato_aoi_mask)
    return out


def _bridge_context(protocol: Protocol, pz: ProductZip, stems: list[str], area: np.ndarray,
                    grid: Grid) -> dict[str, Any]:
    """Bridge / elevated-road reference counts by damage grade inside the comparison area. Context only."""
    note = ("CONTEXT ONLY: BATO does not currently generate a bridge-damage prediction layer, so no bridge precision "
            "or recall exists.")
    if not stems:
        return {"note": note, "value": NOT_CALCULABLE, "reason": "no bridge layer mapped for this product"}
    field = protocol.raw["bridge_context"]["damage_field"]
    counts: dict[str, int] = {}
    outside = 0
    for stem in stems:
        for f in pz.load_lonlat(stem).features:
            props = f.get("properties") or {}
            if field not in props:
                raise ProtocolError(f"{pz.path.name}: {stem} feature without the damage field {field!r}")
            grade = protocol.bridge_grade(props[field])
            pts = list(iter_coords(f.get("geometry")))
            if not pts:
                continue
            lon = sum(p[0] for p in pts) / len(pts)
            lat = sum(p[1] for p in pts) / len(pts)
            x, y = project_ring([(lon, lat)])
            c, r = int((x[0] - grid.c) // grid.a), int((y[0] - grid.f) // grid.e)
            if 0 <= r < grid.height and 0 <= c < grid.width and area[r, c]:
                counts[grade] = counts.get(grade, 0) + 1
            else:
                outside += 1
    return {"note": note, "features_in_area_by_grade": dict(sorted(counts.items())),
            "features_outside_area": outside}


def _pool(per_product: list[dict[str, Any]]) -> dict[str, Any]:
    areas = [p.pop("_areas") for p in per_product]
    if len(areas) > 1:
        overlap = np.zeros_like(areas[0], dtype=np.int16)
        for a in areas:
            overlap += a
        if (overlap > 1).any():
            return {"value": NOT_CALCULABLE, "reason": "compared EMS AOIs share BATO pixels; pooling would double "
                                                       "count"}
    pooled: dict[str, Any] = {}
    for cat, meta in COMPARED.items():
        present = [p["comparisons"][cat] for p in per_product if p["comparisons"].get(cat, {}).get("value") !=
                   NOT_CALCULABLE]
        if not present:
            pooled[cat] = {"value": NOT_CALCULABLE, "reason": f"no compared product contains a class mapped to {cat}"}
            continue
        if not meta["class_match"]:
            fp = sum(x["footprint_pixels"] for x in present)
            fl = sum(x["bato_flood_pixels"] for x in present)
            ov = sum(x["overlap_pixels"] for x in present)
            pooled[cat] = {"label": present[0]["label"], "products": len(present), "footprint_pixels": fp,
                           "bato_flood_pixels": fl, "overlap_pixels": ov,
                           "footprint_share_marked_bato_flood": ratio(ov, fp, "overlap / footprint", "empty"),
                           "bato_flood_share_inside_footprint": ratio(ov, fl, "overlap / FLOOD", "no FLOOD")}
            continue
        modes = {}
        for mode in ("assessed_only", "all_pixels"):
            c = {k: sum(x["modes"][mode]["counts"][k] for x in present) for k in ("TP", "FP", "FN", "TN", "N")}
            tol = {}
            for key in ("precision_tol", "recall_tol"):
                num = sum(x["modes"][mode]["tolerant"][key]["numerator"] for x in present)
                den = sum(x["modes"][mode]["tolerant"][key]["denominator"] for x in present)
                tol[key] = ratio(num, den, key, "zero denominator")
            modes[mode] = {"counts": c, "strict": scores(c), "tolerant": tol}
        area = sum(x["comparison_area_pixels"] for x in present)
        assessed = sum(x["assessed_pixels"] for x in present)
        pooled[cat] = {"products": len(present), "comparison_area_pixels": area,
                       "observable_fraction": ratio(assessed, area, "observable fraction", "empty area"),
                       "reference_pixels": sum(x["reference_pixels"] for x in present),
                       "reference_pixels_not_assessed": sum(x["reference_pixels_not_assessed"] for x in present),
                       "modes": modes}
    roads: dict[str, Any] = {}
    names = {n for p in per_product for n in (p["roads"].get("comparisons") or {})}
    for name in sorted(names):
        comps = [p["roads"]["comparisons"][name] for p in per_product if name in (p["roads"].get("comparisons") or {})]
        valid = [c for c in comps if c.get("modes")]
        entry: dict[str, Any] = {"role": comps[0]["role"], "label": comps[0]["label"], "products": len(comps),
                                 "metric_domain": comps[0]["metric_domain"],
                                 "segments_compared": sum(c["segments_compared"] for c in comps)}
        cov = [c["reference_coverage"] for c in comps]
        sums = {k: round(sum(x[k] for x in cov), 1) for k in cov[0] if k.endswith("_m") or k.startswith("ems_")}
        tot = sums["ems_positive_length_in_area_m"]
        entry["reference_coverage"] = {**sums, "matched_positive_length_fraction": (
            round(sums["ems_positive_length_matched_m"] / tot, 4) if tot > 0 else NOT_CALCULABLE)}
        if not valid:
            entry.update({"value": NOT_CALCULABLE, "reason": "no product produced labelled segments"})
        else:
            entry["modes"] = {}
            for mode in ("observed_only", "all_segments"):
                c = {k: sum(x["modes"][mode]["counts"][k] for x in valid) for k in ("TP", "FP", "FN", "TN", "N")}
                entry["modes"][mode] = {"counts": c, "strict": scores(c)}
            entry["descriptive_lengths_m"] = {k: round(sum(x["descriptive_lengths_m"][k] for x in comps), 1)
                                              for k in comps[0]["descriptive_lengths_m"]}
        roads[name] = entry
    pooled["roads"] = roads or {"value": NOT_CALCULABLE, "reason": "no product has a mapped road-damage layer"}
    return pooled


def evaluate(cfg: BatoConfig, aoi_id: str, candidate_id: str, *, read_state: StateReader = rasterio_state_reader,
             connectivity_dir: Path | None = None, protocol: Protocol | None = None,
             now: Callable[[], str] = _now) -> tuple[Path, dict[str, Any]]:
    root = cfg.config_dir.parent
    paths = cfg.settings.paths
    started = now()
    protocol = protocol or load_protocol(root)
    out_base = eval_dir(cfg)
    freeze_rec = check_frozen(protocol, out_base)
    leak = leakage.scan(paths, root)
    if leak["hits"]:
        raise EvaluationError(f"leakage scan found {len(leak['hits'])} hit(s); refusing to evaluate: "
                              f"{leak['hits'][:3]}")
    manifest = load_manifest(paths)
    assert manifest is not None
    zips_before = {p.name: sha256_file(p) for p in verify_downloads(paths, manifest)}
    bato = load_official(cfg, aoi_id, candidate_id, read_state=read_state, connectivity_dir=connectivity_dir)
    if [list(map(float, p)) for p in manifest["bato_aoi"]["aoi_ring"]] != [list(p) for p in bato.aoi_ring]:
        raise EvaluationError("the acquisition manifest was made for a different BATO AOI")
    act = json.loads((reference_dir(paths) / "activation" / manifest["activation"]["file"]).read_text(encoding="utf-8"))
    record = parse_activation(act, CODE)
    zones = readings(protocol.raw["time"])
    chosen, selection = selected_products(manifest, protocol, bato.post_time)
    bato_aoi_mask = rasterize([[bato.aoi_ring]], bato.grid)
    per_product = [_evaluate_product(cfg, protocol, e, bato, bato_aoi_mask) for e in chosen]
    pooled = _pool(per_product) if per_product else {"value": NOT_CALCULABLE, "reason": "no compared product"}
    gaps = [product_gaps(_product_from_manifest(e), bato.post_time, zones) for e in chosen]
    changed = rehash(root, bato.files_read)
    if changed:
        raise EvaluationError(f"BATO files changed during the evaluation: {changed}")
    zips_after = {p.name: sha256_file(p) for p in verify_downloads(paths, manifest)}
    if zips_after != zips_before:
        raise EvaluationError("reference zips changed during the evaluation")
    result = {
        "kind": "emsr927_case_study_comparison", "validated_claim": False,
        "statement": "Case-study comparison of one event against a reference that may differ in class and time. "
                     "Not a validation of BATO and not a general flood-accuracy figure.",
        "protocol": {"protocol_id": protocol.protocol_id, "yaml_sha256": protocol.yaml_sha256,
                     "doc_sha256": protocol.doc_sha256, "frozen_at": freeze_rec["frozen_at"],
                     "combined_sha256": protocol.combined_sha256},
        "leakage_scan": {"json_files_scanned": leak["json_files_scanned"], "hits": len(leak["hits"]),
                         "rule": leak["rule"]},
        "bato": {"aoi_id": aoi_id, "candidate_id": candidate_id, "connectivity_run": bato.run_dir.name,
                 "road_rule": bato.provenance.get("road_rule", "v1"), "files_read_sha256": bato.files_read,
                 "files_unchanged_after_evaluation": True, "grid": {"transform": list(bato.grid.transform),
                                                                    "width": bato.grid.width,
                                                                    "height": bato.grid.height, "crs": "EPSG:3857"},
                 "bato_aoi_pixels": int(bato_aoi_mask.sum()), "post_product": bato.post_product,
                 "post_acquisition_utc": bato.post_time.isoformat(), "event_date": bato.event_date.isoformat(),
                 "notes": bato.notes},
        "reference": {"activation_file": manifest["activation"]["file"],
                      "activation_sha256": manifest["activation"]["sha256"], "zips_sha256": zips_before,
                      "attribution": manifest.get("attribution")},
        "primary_evaluation": {"what": "road damage: BATO BLOCKED vs EMSR927 road DESTROYED (segment level)",
                               "where": "pooled.roads.<primary comparison> and per_product[*].roads.comparisons",
                               "primary_comparison": next((n for n, c in protocol.road_comparisons().items()
                                                           if c.get("role") == "primary"), None)},
        "product_selection": selection, "per_product": per_product, "pooled": pooled,
        "time": {"readings": list(zones), "ems_time_zone": protocol.raw["time"].get("ems_time_zone"),
                 "ems_time_zone_evidence": protocol.raw["time"].get("ems_time_zone_evidence"),
                 "gaps_ems_image_minus_bato_post_hours": gaps,
                 "event": {**event_dates(record.event_time, zones), "bato_event_date": bato.event_date.isoformat(),
                           "activation_time_raw": record.activation_time}},
        "not_calculable": {
            "flood_precision_recall_f1_iou": "EMSR927 products contain no open-water / flood class; the observed "
                                             "event is mass movement (Landslide), which is not mapped to flood",
            "bridge_precision_recall": "BATO does not generate a bridge-damage prediction layer; bridge reference "
                                       "features are reported as counts only",
            "building_precision_recall": "BATO has no building-impact output",
            "settlement_isolation_precision_recall": "no reference blockage data or isolation reference protocol",
            "accuracy": "not reported: (TP+TN)/N is dominated by TN and says nothing about mapping quality"},
        "not_claimed": ["FloodViT accuracy", "Kuro Siwo / unseen-Himalayan performance",
                        "debris detection performance (debris-v2 rejected)", "general flood accuracy"],
        "started_at_utc": started, "finished_at_utc": now(),
    }
    out = out_base / protocol.combined_sha256[:12]
    assert_reference_input_allowed(out, paths)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "metrics.json"
    reproduced = None
    if target.is_file():
        earlier = json.loads(target.read_text(encoding="utf-8"))
        reproduced = _comparable(earlier) == _comparable(json.loads(json.dumps(result, default=str)))
        if not reproduced:
            stamp = str(result["finished_at_utc"]).replace(":", "").replace("-", "")
            target = out / f"metrics_{stamp}.json"
    if reproduced is not True:
        target.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        (out / target.name.replace(".json", ".md")).write_text(render_metrics(result), encoding="utf-8")
    with (out_base / RUNS_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"protocol_id": protocol.protocol_id, "protocol_sha256": protocol.combined_sha256,
                             "output": f"{out.name}/{target.name}", "metrics_sha256": sha256_file(target),
                             "reproduced_earlier_result": reproduced,
                             "finished_at_utc": result["finished_at_utc"]}) + "\n")
    if reproduced is False:
        raise EvaluationError(f"this run differs from the earlier result under the same frozen protocol; both are "
                              f"kept ({target.name}) and logged in {RUNS_FILE}")
    return out, result


def _comparable(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in ("started_at_utc", "finished_at_utc")}


def _fmt(r: dict[str, Any]) -> str:
    if r.get("value") == NOT_CALCULABLE:
        return f"NOT CALCULABLE ({r.get('numerator')}/{r.get('denominator')})"
    return f"{r['value']:.4f} ({r['numerator']}/{r['denominator']})"


def render_metrics(res: dict[str, Any]) -> str:
    L = ["# EMSR927 case-study comparison", "", res["statement"], "",
         f"Protocol `{res['protocol']['protocol_id']}` frozen {res['protocol']['frozen_at']} "
         f"(sha256 `{res['protocol']['combined_sha256'][:12]}`). Leakage scan: "
         f"{res['leakage_scan']['hits']} hits in {res['leakage_scan']['json_files_scanned']} files.", ""]
    for p in res["per_product"]:
        L += [f"## {p['product']}", "",
              f"EMS AOI {p['ems_aoi_area_km2']} km², overlap with BATO AOI "
              f"{p['ems_aoi_overlap_with_bato_aoi_km2']} km²; classes present: {p['event_classes_present']}", ""]
        roads = p["roads"]
        if "comparisons" not in roads:
            L += [f"**Roads**: NOT CALCULABLE — {roads.get('reason')}", ""]
        else:
            L += [f"BATO segments in area: {roads['bato_segments_in_area']}; length (m): "
                  f"{roads['bato_length_in_area_m']}; EMS road features by grade: {roads['ems_features_by_grade']}", ""]
            for _name, c in roads["comparisons"].items():
                cv = c["reference_coverage"]
                L += [f"**Roads — {c['role'].upper()} — {c['label']}**", "",
                      f"Reference labels: {c['segments_by_reference_label']}; lengths (m): "
                      f"{c['descriptive_lengths_m']}", "",
                      f"Reference coverage: {cv['ems_positive_length_matched_m']} of "
                      f"{cv['ems_positive_length_in_area_m']} m of EMS positive road length is represented in the "
                      f"metrics (fraction {cv['matched_positive_length_fraction']}); unmatched "
                      f"{cv['ems_positive_length_unmatched_m']} m ({cv['ems_positive_features_unmatched']} features "
                      f"entirely unmatched, {cv['ems_positive_features_partially_matched']} partially).", "",
                      f"_{c['metric_domain']}_", ""]
                if not c.get("modes"):
                    L += [f"NOT CALCULABLE — {c.get('reason')}", ""]
                    continue
                L += ["| Mode (matched domain) | TP | FP | FN | TN | precision | recall (matched domain) | F1 |",
                      "|---|---|---|---|---|---|---|---|"]
                for mode, m in c["modes"].items():
                    k, sc = m["counts"], m["strict"]
                    L.append(f"| {mode} | {k['TP']} | {k['FP']} | {k['FN']} | {k['TN']} | {_fmt(sc['precision'])} | "
                             f"{_fmt(sc['recall'])} | {_fmt(sc['f1'])} |")
                L.append("")
        for cat, c in p["comparisons"].items():
            if c.get("value") == NOT_CALCULABLE:
                L += [f"**{cat}**: NOT CALCULABLE — {c['reason']}", ""]
            elif not c["class_match"]:
                L += [f"**{cat}** — {c['label']}", "",
                      f"Footprint {c['footprint_pixels']} px, BATO FLOOD {c['bato_flood_pixels']} px, overlap "
                      f"{c['overlap_pixels']} px; share of footprint marked FLOOD "
                      f"{_fmt(c['footprint_share_marked_bato_flood'])}; share of footprint assessed "
                      f"{_fmt(c['footprint_share_assessed'])}; footprint by Step 4C state "
                      f"{c['footprint_pixels_by_state']}.", ""]
            else:
                L += [f"**{cat}** — {c['label']}", "", "| Mode | TP | FP | FN | TN | precision | recall | F1 | IoU |",
                      "|---|---|---|---|---|---|---|---|---|"]
                for mode, m in c["modes"].items():
                    k, sc = m["counts"], m["strict"]
                    L.append(f"| {mode} | {k['TP']} | {k['FP']} | {k['FN']} | {k['TN']} | {_fmt(sc['precision'])} | "
                             f"{_fmt(sc['recall'])} | {_fmt(sc['f1'])} | {_fmt(sc['iou'])} |")
                L.append("")
        b = p.get("bridges", {})
        L += [f"Bridges (context only): {b.get('features_in_area_by_grade', b.get('reason'))}. {b.get('note', '')}",
              ""]
    L += ["## Not calculable", ""] + [f"- {k}: {v}" for k, v in res["not_calculable"].items()] + [""]
    return "\n".join(L) + "\n"

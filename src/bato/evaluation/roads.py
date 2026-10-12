"""EVALUATION ONLY. Road comparison (protocol section G): BATO segment states vs an EMS road-damage line layer.

The comparison runs only when the protocol maps a transportation layer with a damage-grade field. Each pre-registered
comparison (primary, supplementary) places every EMS damage grade in positive, negative or excluded.

**Unit.** A BATO segment with at least ``min_inside_fraction`` of its samples inside the comparison area.

**Reference label.** Per comparison, with precedence positive > excluded > negative > unmatched; see
:func:`road_comparison`.

Distances are ground metres. Samples are spaced uniformly by ground distance (geometry.resample_ground), each
representing the same length, and Web-Mercator search radii are scaled by 1/cos(latitude) of the segment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from bato.evaluation.geometry import Grid, densify_3857, geojson_lines, mercator_scale, resample_ground
from bato.evaluation.metrics import segment_comparison

STATE_NAMES = ("blocked", "open", "unobservable")


def _edges(lines: Sequence[Sequence[tuple[float, float]]]) -> np.ndarray:
    """All line edges in Web-Mercator metres, shape (n, 4): x0, y0, x1, y1."""
    out = []
    for line in lines:
        pts = densify_3857(line, 1e9)                      # vertices only
        if len(pts) >= 2:
            out.append(np.column_stack([pts[:-1], pts[1:]]))
    return np.concatenate(out) if out else np.zeros((0, 4))


def _min_dist(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge (Web-Mercator metres)."""
    if edges.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    px, py = points[:, 0:1], points[:, 1:2]
    x0, y0, x1, y1 = edges[:, 0][None], edges[:, 1][None], edges[:, 2][None], edges[:, 3][None]
    dx, dy = x1 - x0, y1 - y0
    den = dx * dx + dy * dy
    t = np.clip(np.where(den > 0, ((px - x0) * dx + (py - y0) * dy) / np.where(den > 0, den, 1), 0), 0, 1)
    d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    return np.asarray(d.min(axis=1))


def _near(points: np.ndarray, edges: np.ndarray, radius: float) -> np.ndarray:
    """Boolean per point: within ``radius`` of any edge (bbox prefilter, then exact distance)."""
    if edges.shape[0] == 0 or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    xmin, ymin = points.min(axis=0) - radius
    xmax, ymax = points.max(axis=0) + radius
    ex0, ex1 = np.minimum(edges[:, 0], edges[:, 2]), np.maximum(edges[:, 0], edges[:, 2])
    ey0, ey1 = np.minimum(edges[:, 1], edges[:, 3]), np.maximum(edges[:, 1], edges[:, 3])
    cand = edges[(ex1 >= xmin) & (ex0 <= xmax) & (ey1 >= ymin) & (ey0 <= ymax)]
    out = np.zeros(points.shape[0], dtype=bool)
    for s in range(0, points.shape[0], 2000):
        out[s:s + 2000] = _min_dist(points[s:s + 2000], cand) <= radius
    return out


def in_area(pts: np.ndarray, area: np.ndarray, grid: Grid) -> np.ndarray:
    cols = np.floor((pts[:, 0] - grid.c) / grid.a).astype(int)
    rws = np.floor((pts[:, 1] - grid.f) / grid.e).astype(int)
    ok = (cols >= 0) & (cols < grid.width) & (rws >= 0) & (rws < grid.height)
    inside = np.zeros(len(pts), dtype=bool)
    inside[ok] = area[rws[ok], cols[ok]]
    return inside


def _check_params(params: dict[str, Any]) -> tuple[float, float, float, float]:
    step, dist = float(params["sample_step_m"]), float(params["match_distance_m"])
    min_len, min_inside = float(params["min_matched_length_m"]), float(params["min_inside_fraction"])
    if not (math.isfinite(dist) and dist > 0 and step > 0 and min_len > 0):
        raise ValueError("match distance, sample step and minimum matched length must be positive and finite")
    return step, dist, min_len, min_inside


def _label(matched: dict[str, float], need: float) -> str:
    """Pre-registered precedence: positive > excluded > negative > unmatched."""
    for lab in ("positive", "excluded", "negative"):
        if matched[lab] >= need:
            return lab
    return "unmatched"


def road_comparison(segments_fc: dict[str, Any], ems_features: list[dict[str, Any]], grades: list[str],
                    area: np.ndarray, grid: Grid, params: dict[str, Any],
                    comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every comparison in ``comparisons`` (grade lists for positive / negative / excluded), over the same matched
    geometry. ``grades[i]`` is the protocol damage grade of ``ems_features[i]``.

    Reference label of a BATO segment, per comparison: let ``need = min(min_matched_length_m, segment length)``;
    positive if >= need of the segment lies within ``match_distance_m`` of a positive-grade EMS line; else excluded if
    that holds for an excluded-grade line (ambiguous reference, removed); else negative if it holds for a negative-grade
    line; else unmatched (removed). BATO positive = blocked; unobservable segments are excluded in ``observed_only``
    mode and counted as BATO negative in ``all_segments`` mode.
    """
    step, dist, min_len, min_inside = _check_params(params)
    lines_by_grade: dict[str, list[list[tuple[float, float]]]] = {}
    for f, g in zip(ems_features, grades, strict=True):
        lines_by_grade.setdefault(g, []).extend(geojson_lines(f.get("geometry")))
    edges = {g: _edges(v) for g, v in lines_by_grade.items()}

    # BATO segments inside the comparison area, with per-grade "near" masks on uniform ground samples
    segs: list[dict[str, Any]] = []
    for feat in segments_fc.get("features", []):
        props = feat.get("properties") or {}
        st = str(props.get("state"))
        if st not in STATE_NAMES:
            raise ValueError(f"unknown road state {st!r}")
        line = [(float(x), float(y)) for x, y in feat["geometry"]["coordinates"]]
        pts, per_sample = resample_ground(line, step)
        if per_sample <= 0:
            raise ValueError(f"segment {props.get('segment_id')!r} has zero length")
        if in_area(pts, area, grid).mean() < min_inside:
            continue
        k = mercator_scale(sum(y for _, y in line) / len(line))
        segs.append({"id": props.get("segment_id"), "state": st, "length": per_sample * len(pts), "per": per_sample,
                     "n": len(pts),
                     "near": {g: _near(pts, e, dist * k) for g, e in edges.items()}, "line": line})

    blocked = [s for s in segs if s["state"] == "blocked"]
    blocked_edges = _edges([s["line"] for s in blocked])
    out: dict[str, Any] = {
        "params": {"match_distance_m": dist, "min_matched_length_m": min_len, "sample_step_m": step,
                   "min_inside_fraction": min_inside, "label_precedence": "positive > excluded > negative > unmatched",
                   "short_segment_rule": "a segment shorter than min_matched_length_m needs its whole length matched"},
        "bato_segments_in_area": {st: sum(1 for s in segs if s["state"] == st) for st in STATE_NAMES},
        "bato_length_in_area_m": {st: round(sum(s["length"] for s in segs if s["state"] == st), 1)
                                  for st in STATE_NAMES},
        "ems_features_by_grade": dict(sorted((g, grades.count(g)) for g in set(grades))),
        "comparisons": {}}

    # EMS line length inside the area by grade, and how much of it lies near a BATO blocked segment
    ems_len: dict[str, float] = {}
    ems_near_blocked: dict[str, float] = {}
    ems_feats_in_area: dict[str, int] = {}
    ems_samples: list[dict[str, Any]] = []          # per feature: in-area samples, for reference coverage
    area_edges = _edges([s["line"] for s in segs])
    for i, (f, g) in enumerate(zip(ems_features, grades, strict=True)):
        hit = False
        for line in geojson_lines(f.get("geometry")):
            pts, per = resample_ground(line, step)
            if per <= 0:
                continue
            inside = in_area(pts, area, grid)
            if inside.any():
                hit = True
            k = mercator_scale(sum(y for _, y in line) / len(line))
            ems_len[g] = ems_len.get(g, 0.0) + float(inside.sum()) * per
            ems_near_blocked[g] = ems_near_blocked.get(g, 0.0) + float(
                (inside & _near(pts, blocked_edges, dist * k)).sum()) * per
            if inside.any():
                ems_samples.append({"feature": i, "grade": g, "pts": pts[inside], "per": per, "k": k,
                                    "near_any_bato": _near(pts[inside], area_edges, dist * k)})
        ems_feats_in_area[g] = ems_feats_in_area.get(g, 0) + int(hit)

    for name, comp in comparisons.items():
        groups = {lab: list(comp.get(lab) or []) for lab in ("positive", "negative", "excluded")}
        rows, pred, ref, obs = [], [], [], []
        tally = {"positive": 0, "negative": 0, "excluded": 0, "unmatched": 0}
        for s in segs:
            matched = {}
            for lab, gs in groups.items():
                mask = np.zeros(s["n"], dtype=bool)
                for g in gs:
                    if g in s["near"]:
                        mask = mask | s["near"][g]
                matched[lab] = float(mask.sum()) * s["per"]
            label = _label(matched, min(min_len, s["length"] - 1e-6))
            tally[label] += 1
            rows.append({"segment_id": s["id"], "state": s["state"], "ground_length_m": round(s["length"], 1),
                         "matched_m": {k2: round(v, 1) for k2, v in matched.items()}, "reference": label})
            if label in ("positive", "negative"):
                pred.append(s["state"] == "blocked")
                ref.append(label == "positive")
                obs.append(s["state"] != "unobservable")
        pos_len = sum(ems_len.get(g, 0.0) for g in groups["positive"])
        res: dict[str, Any] = {
            "role": comp["role"], "label": comp["label"], "grades": groups,
            "segments_by_reference_label": tally, "segments_compared": len(pred),
            "descriptive_lengths_m": {
                "bato_blocked_length_in_area": out["bato_length_in_area_m"]["blocked"],
                "ems_positive_features_in_area": sum(ems_feats_in_area.get(g, 0) for g in groups["positive"]),
                "ems_positive_length_in_area": round(pos_len, 1),
                "bato_blocked_length_within_distance_of_ems_positive": round(sum(
                    r["matched_m"]["positive"] for r in rows if r["state"] == "blocked"), 1),
                "ems_positive_length_within_distance_of_bato_blocked": round(sum(
                    ems_near_blocked.get(g, 0.0) for g in groups["positive"]), 1)},
            "segments": rows}
        res["reference_coverage"] = _reference_coverage(ems_samples, groups["positive"], segs, rows, dist)
        res["metric_domain"] = MATCHED_DOMAIN
        if pred:
            res.update(segment_comparison(np.array(pred), np.array(ref), np.array(obs)))
            res["modes"]["observed_only"]["interpretation"] = OBSERVED_ONLY
            res["modes"]["all_segments"]["interpretation"] = ALL_SEGMENTS
        else:
            res["modes"] = None
            res["value"] = "NOT CALCULABLE"
            res["reason"] = "no BATO segment in the comparison area received a positive or negative reference label"
        out["comparisons"][name] = res
    return out


MATCHED_DOMAIN = ("Segment-level metrics over BATO segments with valid EMS reference matches (reference label positive "
                  "or negative under the frozen matching rule). Conditional on this matched domain; NOT full-network "
                  "recall. EMS positive road length outside the domain is reported in reference_coverage, never "
                  "counted as TN and never dropped silently.")
OBSERVED_ONLY = ("matched-domain segments whose BATO state is blocked or open; unobservable segments removed "
                 "(BATO made no observation there)")
ALL_SEGMENTS = ("all matched-domain segments; unobservable segments counted as BATO negative (not blocked), so a "
                "reference-positive unobservable segment is a FN")


def _reference_coverage(samples: list[dict[str, Any]], positive_grades: list[str], segs: list[dict[str, Any]],
                        rows: list[dict[str, Any]], dist: float) -> dict[str, Any]:
    """How much EMS reference-positive road length (in the comparison area) enters the TP/FN accounting.

    A positive EMS sample is *matched* when it lies within ``match_distance_m`` of an eligible BATO segment that
    received the positive reference label (the only way it can become TP or FN). Unmatched length is split into:
    near an eligible BATO segment that did not receive the positive label (e.g. < 25 m alongside it), and no eligible
    BATO segment within the distance at all.
    """
    pos_rows = [(s, r) for s, r in zip(segs, rows, strict=True) if r["reference"] == "positive"]
    e_all = _edges([s["line"] for s, _ in pos_rows])
    e_obs = _edges([s["line"] for s, _ in pos_rows if s["state"] != "unobservable"])
    total = matched = matched_obs = near_other = 0.0
    per_feature: dict[int, list[float]] = {}
    for smp in samples:
        if smp["grade"] not in positive_grades:
            continue
        pts, per, radius = smp["pts"], smp["per"], dist * smp["k"]
        m_all = _near(pts, e_all, radius)
        m_obs = _near(pts, e_obs, radius)
        total += len(pts) * per
        matched += float(m_all.sum()) * per
        matched_obs += float(m_obs.sum()) * per
        near_other += float((~m_all & smp["near_any_bato"]).sum()) * per
        acc = per_feature.setdefault(smp["feature"], [0.0, 0.0])
        acc[0] += len(pts) * per
        acc[1] += float(m_all.sum()) * per
    unmatched = total - matched
    fully = sum(1 for t, m in per_feature.values() if t > 0 and m >= t - 1e-9)
    none = sum(1 for t, m in per_feature.values() if m <= 0)
    frac = (round(matched / total, 4) if total > 0 else "NOT CALCULABLE")
    return {
        "definition": "EMS positive-grade road length inside the comparison area, split by whether it lies within "
                      "match_distance_m of an eligible BATO segment that received the POSITIVE reference label "
                      "(i.e. whether it is represented in TP/FN)",
        "ems_positive_features_in_area": len(per_feature),
        "ems_positive_length_in_area_m": round(total, 1),
        "ems_positive_length_matched_m": round(matched, 1),
        "ems_positive_length_matched_observed_only_m": round(matched_obs, 1),
        "ems_positive_length_unmatched_m": round(unmatched, 1),
        "unmatched_near_bato_segment_without_positive_label_m": round(near_other, 1),
        "unmatched_no_eligible_bato_segment_within_distance_m": round(unmatched - near_other, 1),
        "matched_positive_length_fraction": frac,
        "matched_positive_length_fraction_observed_only": (round(matched_obs / total, 4) if total > 0
                                                            else "NOT CALCULABLE"),
        "ems_positive_features_fully_matched": fully,
        "ems_positive_features_partially_matched": len(per_feature) - fully - none,
        "ems_positive_features_unmatched": none,
    }


def alignment_diagnostic(road_lines: list[list[tuple[float, float]]], ems_features: list[dict[str, Any]],
                         grades: list[str], inside: Any, params: dict[str, Any],
                         distances: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)) -> dict[str, Any]:
    """GEOMETRY ONLY (no BATO road state is read): how far EMS road lines lie from the BATO OSM road network, and
    how many BATO segments are shorter than the minimum matched length. Used to check, before the freeze, whether
    the matching rule suits the actual geometry. ``inside(points_3857) -> bool mask`` selects the comparison area.
    """
    step, dist, min_len, _ = _check_params(params)
    net = _edges(road_lines)
    total: dict[str, float] = {}
    near: dict[str, dict[float, float]] = {}
    for f, g in zip(ems_features, grades, strict=True):
        for line in geojson_lines(f.get("geometry")):
            pts, per = resample_ground(line, step)
            if per <= 0:
                continue
            m = inside(pts)
            k = mercator_scale(sum(y for _, y in line) / len(line))
            total[g] = total.get(g, 0.0) + float(m.sum()) * per
            for d in distances:
                near.setdefault(g, {}).setdefault(d, 0.0)
                near[g][d] += float((m & _near(pts, net, d * k)).sum()) * per
    seg_len = []
    for line in road_lines:
        pts, per = resample_ground(line, step)
        if per > 0 and inside(pts).mean() >= float(params["min_inside_fraction"]):
            seg_len.append(per * len(pts))
    all_total = sum(total.values())
    share = {f"{int(d)}m": (round(sum(near[g][d] for g in near) / all_total, 4) if all_total else None)
             for d in distances}
    return {"reads_bato_road_state": False, "match_distance_m": dist, "min_matched_length_m": min_len,
            "ems_road_length_in_area_m_by_grade": {g: round(v, 1) for g, v in sorted(total.items())},
            "share_of_ems_road_length_within_distance_of_bato_roads": share,
            "share_by_grade": {g: {f"{int(d)}m": (round(near[g][d] / total[g], 4) if total[g] else None)
                                   for d in distances} for g in sorted(total)},
            "bato_segments_in_area": len(seg_len),
            "bato_segment_length_m": {"median": round(float(np.median(seg_len)), 1) if seg_len else None,
                                      "min": round(min(seg_len), 1) if seg_len else None},
            "bato_segments_shorter_than_min_matched_length": sum(1 for x in seg_len if x < min_len)}

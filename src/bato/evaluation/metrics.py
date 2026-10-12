"""EVALUATION ONLY. Pixel and segment comparison metrics (protocol bato-emsr927-v1, sections D-F).

Every ratio is returned with its numerator and denominator. A zero denominator gives ``NOT CALCULABLE`` with the
reason, never 0 or 1. "Accuracy" is not computed (TN dominates in a mostly dry valley, so it carries no
information about mapping quality).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bato.contracts.baseline import BaselineState

NOT_CALCULABLE = "NOT CALCULABLE"
ASSESSED = (int(BaselineState.NO_FLOOD), int(BaselineState.FLOOD), int(BaselineState.PRE_EVENT_WATER))
NOT_ASSESSED = {int(s): s.name for s in BaselineState if int(s) >= 10}
VALID_CODES = set(ASSESSED) | set(NOT_ASSESSED)
FLOOD = int(BaselineState.FLOOD)
PRE_EVENT_WATER = int(BaselineState.PRE_EVENT_WATER)


class MetricError(ValueError):
    pass


def ratio(num: int, den: int, name: str, why_zero: str) -> dict[str, Any]:
    if den == 0:
        return {"value": NOT_CALCULABLE, "numerator": num, "denominator": den, "reason": f"{name}: {why_zero}"}
    return {"value": round(num / den, 6), "numerator": num, "denominator": den}


def confusion(pred: np.ndarray, ref: np.ndarray, include: np.ndarray) -> dict[str, int]:
    p, r = pred & include, ref & include
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    tn = int((include & ~p & ~r).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": int(include.sum())}


def scores(c: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    return {
        "precision": ratio(tp, tp + fp, "precision = TP / (TP + FP)", "BATO marked no positive pixel/segment"),
        "recall": ratio(tp, tp + fn, "recall = TP / (TP + FN)", "the reference has no positive in this set"),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "F1 = 2TP / (2TP + FP + FN)", "neither side has a positive"),
        "iou": ratio(tp, tp + fp + fn, "IoU = TP / (TP + FP + FN)", "neither side has a positive"),
    }


def dilate3(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square (2r+1) x (2r+1) dilation (8-connected for r = 1); pixels outside the array count as empty."""
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            out[yd, xd] |= mask[ys, xs]
    return out


def tolerant(pred: np.ndarray, ref: np.ndarray, include: np.ndarray, radius: int) -> dict[str, Any]:
    """Boundary-tolerant precision / recall / F1 (protocol section F). IoU has no tolerant form here."""
    p, r = pred & include, ref & include
    dp, dr = dilate3(p, radius), dilate3(r, radius)
    prec = ratio(int((p & dr).sum()), int(p.sum()), "precision_tol = |P ∩ D(R)| / |P|", "BATO marked no positive")
    rec = ratio(int((r & dp).sum()), int(r.sum()), "recall_tol = |R ∩ D(P)| / |R|", "reference has no positive")
    if isinstance(prec["value"], float) and isinstance(rec["value"], float) and prec["value"] + rec["value"] > 0:
        f1: dict[str, Any] = {"value": round(2 * prec["value"] * rec["value"] / (prec["value"] + rec["value"]), 6),
                              "formula": "2 * precision_tol * recall_tol / (precision_tol + recall_tol)"}
    else:
        f1 = {"value": NOT_CALCULABLE, "reason": "F1_tol needs both tolerant precision and recall, not both 0"}
    return {"tolerance_px": radius, "precision_tol": prec, "recall_tol": rec, "f1_tol": f1,
            "iou_tol": {"value": NOT_CALCULABLE, "reason": "no tolerant IoU is defined in the protocol"}}


def check_states(state: np.ndarray) -> None:
    bad = set(np.unique(state).tolist()) - VALID_CODES
    if bad:
        raise MetricError(f"Step 4C raster contains unknown state codes {sorted(bad)}")


def pixel_comparison(state: np.ndarray, ref: np.ndarray, area: np.ndarray, *, tolerance_px: int,
                     row_ground_area_m2: np.ndarray) -> dict[str, Any]:
    """Both modes of protocol section E for one reference mask inside one comparison area."""
    check_states(state)
    if not (state.shape == ref.shape == area.shape):
        raise MetricError("state, reference and area masks must share the BATO grid")
    assessed = np.isin(state, ASSESSED)
    pred = state == FLOOD
    ref_in = ref & area
    na_by_reason = {name: int((ref_in & (state == code)).sum()) for code, name in NOT_ASSESSED.items()}
    out: dict[str, Any] = {
        "comparison_area_pixels": int(area.sum()),
        "comparison_area_km2_ground_approx": round(float((area.sum(axis=1) * row_ground_area_m2).sum()) / 1e6, 4),
        "assessed_pixels": int((area & assessed).sum()),
        "observable_fraction": ratio(int((area & assessed).sum()), int(area.sum()),
                                     "observable fraction = |assessed ∩ C| / |C|", "empty comparison area"),
        "reference_pixels": int(ref_in.sum()),
        "reference_pixels_not_assessed": int((ref_in & ~assessed).sum()),
        "reference_pixels_not_assessed_by_reason": na_by_reason,
        "reference_pixels_on_pre_event_water": int((ref_in & (state == PRE_EVENT_WATER)).sum()),
        "bato_flood_pixels": int((pred & area).sum()),
        "modes": {},
    }
    for mode, include in (("assessed_only", area & assessed), ("all_pixels", area)):
        c = confusion(pred, ref, include)
        out["modes"][mode] = {
            "not_assessed_handling": ("excluded" if mode == "assessed_only"
                                      else "counted as BATO negative (no flood claim was made there)"),
            "counts": c, "strict": scores(c), "tolerant": tolerant(pred, ref, include, tolerance_px)}
    return out


def segment_comparison(pred: np.ndarray, ref: np.ndarray, observable: np.ndarray) -> dict[str, Any]:
    """Road segments (protocol section G): pred = BLOCKED, ref = reference-positive, observable = not UNOBSERVABLE.
    Inputs are boolean vectors over the matched (reference-labelled) segments."""
    out: dict[str, Any] = {"modes": {}}
    for mode, include in (("observed_only", observable), ("all_segments", np.ones_like(observable))):
        c = confusion(pred[None, :], ref[None, :], include[None, :])
        out["modes"][mode] = {
            "unobservable_handling": "excluded" if mode == "observed_only" else "counted as BATO negative",
            "counts": c, "strict": scores(c)}
    return out


def footprint_overlap(state: np.ndarray, ref: np.ndarray, area: np.ndarray, *,
                      row_ground_area_m2: np.ndarray) -> dict[str, Any]:
    """CLASS-MISMATCHED FOOTPRINT OVERLAP: BATO radar-change (FLOOD) footprint vs a reference footprint of a different
    class (EMSR927 mass movement). Descriptive overlap shares only. Deliberately NOT precision, recall, F1 or IoU,
    which would present a different-class reference as if it were flood truth."""
    check_states(state)
    if not (state.shape == ref.shape == area.shape):
        raise MetricError("state, reference and area masks must share the BATO grid")
    assessed = np.isin(state, ASSESSED)
    flood = (state == FLOOD) & area
    fp = ref & area
    km2 = lambda m: round(float((m.sum(axis=1) * row_ground_area_m2).sum()) / 1e6, 4)  # noqa: E731
    return {
        "label": "CLASS-MISMATCHED FOOTPRINT OVERLAP: BATO radar-change footprint vs EMSR927 mass-movement footprint. "
                 "Not flood precision, recall, F1, IoU or accuracy.",
        "comparison_area_pixels": int(area.sum()), "comparison_area_km2_ground_approx": km2(area),
        "footprint_pixels": int(fp.sum()), "footprint_km2_ground_approx": km2(fp),
        "bato_flood_pixels": int(flood.sum()),
        "overlap_pixels": int((flood & fp).sum()),
        "footprint_share_marked_bato_flood": ratio(int((flood & fp).sum()), int(fp.sum()),
                                                   "|FLOOD ∩ footprint| / |footprint|", "empty footprint"),
        "bato_flood_share_inside_footprint": ratio(int((flood & fp).sum()), int(flood.sum()),
                                                   "|FLOOD ∩ footprint| / |FLOOD|", "no BATO FLOOD pixel"),
        "footprint_share_assessed": ratio(int((fp & assessed).sum()), int(fp.sum()),
                                          "|assessed ∩ footprint| / |footprint|", "empty footprint"),
        "footprint_pixels_by_state": {
            "NO_FLOOD": int((fp & (state == 0)).sum()), "FLOOD": int((fp & (state == FLOOD)).sum()),
            "PRE_EVENT_WATER": int((fp & (state == PRE_EVENT_WATER)).sum()),
            **{name: int((fp & (state == code)).sum()) for code, name in NOT_ASSESSED.items()}},
    }

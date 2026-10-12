"""Step 4C: deterministic SAR change flood baseline on the verified Step 3.5 stack (no model, no training).

Per pixel, with dB = 10 log10(sigma0) and thresholds from ``config/thresholds.yaml`` (``baseline_flood``):

    water(t)   = VV_t <= T_vv  AND  VH_t <= T_vh                              (open-water-like darkness)
    dark(pre)  = (VV_post - VV_pre <= C_pre)  AND  (VH_post - VH_pre <= C_pre)  (new darkening, both pols)
    candidate  = water(POST) AND dark(PRE2) AND dark(PRE1) AND NOT (water(PRE1) OR water(PRE2))
    FLOOD      = candidate in an 8-connected component of >= detection.min_component_pixels pixels

Gating (checked first, in this order; any hit makes the pixel NOT_ASSESSED, never NO_FLOOD):
    invalid SAR (valid_mask 0) -> no GLO-30 terrain -> layover-likely -> shadow-likely -> steep.
Layover/shadow use an APPROXIMATE local incidence angle from GLO-30 slope/aspect, the orbit's look direction
and a nominal incidence angle (Step 3.4 saved no SNAP layover/shadow mask):
    s_r = atan(tan(slope) * cos(aspect - sensor_azimuth))   (slope component toward the sensor)
    LIA = incidence - s_r;  layover if LIA <= visibility.layover_lia_max_deg; shadow if LIA >= shadow_lia_min_deg.
Shadow also includes CAST shadow (bato.detection.terrain_gates.cast_shadow): flat ground shaded by terrain toward
the sensor. Layover is LOCAL only (slopes facing the sensor); layover cast onto neighbouring flat ground is not
modelled. At max_slope_deg 15 the local layover/shadow tests only relabel pixels the steep gate would catch;
cast shadow is the gate that reaches flat ground.
Water already open-water-dark at PRE1 or PRE2 and still water at POST is PRE_EVENT_WATER (not flood).

FloodViT outputs are never read for the detection; they are only compared afterwards if present.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.contracts.baseline import (
    FLOOD_NODATA,
    NOT_ASSESSED_STATES,
    BaselineOutputs,
    BaselineProvenance,
    BaselineState,
    BaselineStatistics,
    ComponentStats,
    FloodViTComparison,
    ThresholdRecord,
)
from bato.contracts.inference import FloodInferenceProvenance
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.preprocessing import TerrainProvenance
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.detection.components import component_sizes
from bato.detection.terrain_gates import cast_shadow, sample_grid
from bato.exceptions import BatoError
from bato.inference.diagnostics import LOOK_AZIMUTH_DEG, DemFeatures, dem_features, sample_features
from bato.preprocessing.model_input import (
    ModelInputJob,
    check_consistency,
    check_rasters,
    prepare_model_input,
    source_stack_record,
)
from bato.preprocessing.radiometric import _relative, _sibling, _utc_now
from bato.preprocessing.sigma0_validation import _open, require_rasterio
from bato.preprocessing.speckle import sha256_file

DETECTION_GROUP = "detection"           # <interim>/detection/<aoi_id>/<candidate_id>/baseline/
OUTPUT_SUBDIR = "baseline"
FLOOD_FILE = "baseline_flood.tif"
STATE_FILE = "baseline_state.tif"
PROVENANCE_FILE = "baseline_provenance.json"
DIAGNOSTICS_FILE = "baseline_diagnostics.md"
BLOCK_ROWS = 512
DB_FLOOR = 1e-6                          # sigma0 floor before log10 (-60 dB); the stack has no zeros where valid
GEOMETRY_FLAT_DEG = 10.0                 # |s_r| below this: flat / gentle (reporting only)
RULE = ("FLOOD = water(POST) & dark_vs(PRE2) & dark_vs(PRE1) & ~(water(PRE1) | water(PRE2)), kept if its 8-connected "
        "component >= detection.min_component_pixels; water(t) = VV_t <= vv_water_db & VH_t <= vh_water_db; "
        "dark_vs(pre) = VV_post-VV_pre <= change & VH_post-VH_pre <= change; NOT_ASSESSED first: invalid SAR, no "
        "DEM, layover (approx. local LIA <= layover_lia_max_deg), shadow (approx. local LIA >= shadow_lia_min_deg "
        "OR cast shadow from terrain toward the sensor within shadow_search_m), steep (slope >= max_slope_deg)")


class BaselineError(BatoError):
    """The baseline cannot be computed or published (nothing published)."""


@dataclass(frozen=True)
class BaselineParams:
    vv_water_db: float
    vh_water_db: float
    pre2_change_db: float
    pre1_change_db: float
    max_slope_deg: float
    shadow_lia_min_deg: float
    layover_lia_max_deg: float
    incidence_deg: float
    min_component_pixels: int
    shadow_search_m: float = 5000.0

    @classmethod
    def from_config(cls, cfg: BatoConfig) -> BaselineParams:
        b, t = cfg.thresholds.baseline_flood, cfg.thresholds
        return cls(vv_water_db=b.vv_water_db.value, vh_water_db=b.vh_water_db.value,
                   pre2_change_db=b.pre2_change_db.value, pre1_change_db=b.pre1_change_db.value,
                   max_slope_deg=b.max_slope_deg.value, shadow_lia_min_deg=b.shadow_lia_min_deg.value,
                   layover_lia_max_deg=t.visibility.layover_lia_max_deg.value,
                   incidence_deg=b.nominal_incidence_deg.value,
                   min_component_pixels=int(t.detection.min_component_pixels.value),
                   shadow_search_m=b.shadow_search_m.value)


def threshold_records(cfg: BatoConfig) -> tuple[ThresholdRecord, ...]:
    t = cfg.thresholds
    named = [(f"baseline_flood.{n}", getattr(t.baseline_flood, n)) for n in type(t.baseline_flood).model_fields]
    named += [("visibility.layover_lia_max_deg", t.visibility.layover_lia_max_deg),
              ("detection.min_component_pixels", t.detection.min_component_pixels)]
    return tuple(ThresholdRecord(name=n, value=p.value, unit=p.unit, provisional=p.provisional,
                                 rationale=p.rationale, basis=p.basis or "inherited_bato_design") for n, p in named)


# ---------------------------------------------------------------------------
# Per-pixel rule
# ---------------------------------------------------------------------------


def to_db(x: np.ndarray) -> np.ndarray:
    return (np.float32(10.0) * np.log10(np.maximum(x.astype(np.float32, copy=False), np.float32(DB_FLOOR))))


def range_slope_deg(slope_deg: np.ndarray, aspect_deg: np.ndarray, look_azimuth: float) -> np.ndarray:
    """Slope component toward the sensor (positive: facing the sensor)."""
    sensor_az = (look_azimuth + 180.0) % 360.0
    return np.degrees(np.arctan(np.tan(np.radians(slope_deg)) * np.cos(np.radians(aspect_deg - sensor_az))))


@dataclass(frozen=True)
class BlockResult:
    state: np.ndarray            # uint8 BaselineState, FLOOD = candidate before the cluster filter
    candidate: np.ndarray        # bool
    s_r: np.ndarray              # float32 range slope (NaN without DEM)
    slope: np.ndarray            # float32
    change: dict[str, np.ndarray]
    shadow_cast_only: np.ndarray  # bool: NOT_ASSESSED_SHADOW only because of cast shadow


def classify_block(raw: np.ndarray, valid: np.ndarray, slope_deg: np.ndarray, aspect_deg: np.ndarray,
                   p: BaselineParams, look_azimuth: float, cast: np.ndarray | None = None) -> BlockResult:
    """Apply gating and the per-pixel rule to a (6, h, w) layout-v1 block (PRE1_VV ... POST_VH)."""
    if raw.ndim != 3 or raw.shape[0] != 6 or raw.shape[1:] != valid.shape or slope_deg.shape != valid.shape:
        raise BaselineError(f"block shapes {raw.shape} / {valid.shape} / {slope_deg.shape} do not match")
    db = to_db(raw)
    pre1_vv, pre1_vh, pre2_vv, pre2_vh, post_vv, post_vh = db

    def water(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        return (vv <= p.vv_water_db) & (vh <= p.vh_water_db)

    change = {"vv_vs_pre2": post_vv - pre2_vv, "vh_vs_pre2": post_vh - pre2_vh,
              "vv_vs_pre1": post_vv - pre1_vv, "vh_vs_pre1": post_vh - pre1_vh}
    dark2 = (change["vv_vs_pre2"] <= p.pre2_change_db) & (change["vh_vs_pre2"] <= p.pre2_change_db)
    dark1 = (change["vv_vs_pre1"] <= p.pre1_change_db) & (change["vh_vs_pre1"] <= p.pre1_change_db)
    water_post = water(post_vv, post_vh)
    pre_water = water(pre1_vv, pre1_vh) | water(pre2_vv, pre2_vh)
    s_r = range_slope_deg(slope_deg, aspect_deg, look_azimuth).astype(np.float32)
    lia = p.incidence_deg - s_r
    has_dem = np.isfinite(slope_deg) & np.isfinite(aspect_deg)
    local_shadow = lia >= p.shadow_lia_min_deg
    cast_mask = np.zeros(valid.shape, dtype=bool) if cast is None else cast.astype(bool)
    ok = valid == 1
    state = np.full(valid.shape, int(BaselineState.NO_FLOOD), dtype=np.uint8)
    gates = [(~ok, BaselineState.NOT_ASSESSED_INVALID_SAR), (~has_dem, BaselineState.NOT_ASSESSED_NO_DEM),
             (lia <= p.layover_lia_max_deg, BaselineState.NOT_ASSESSED_LAYOVER),
             (local_shadow | cast_mask, BaselineState.NOT_ASSESSED_SHADOW),
             (slope_deg >= p.max_slope_deg, BaselineState.NOT_ASSESSED_STEEP)]
    assessed = np.ones(valid.shape, dtype=bool)
    for hit, code in gates:
        sel = assessed & hit
        state[sel] = int(code)
        assessed &= ~sel
    pre_event_water = assessed & water_post & pre_water
    candidate = assessed & water_post & dark2 & dark1 & ~pre_water
    state[pre_event_water] = int(BaselineState.PRE_EVENT_WATER)
    state[candidate] = int(BaselineState.FLOOD)
    shadow_cast_only = (state == int(BaselineState.NOT_ASSESSED_SHADOW)) & cast_mask & ~local_shadow
    return BlockResult(state=state, candidate=candidate, s_r=s_r, slope=slope_deg.astype(np.float32), change=change,
                       shadow_cast_only=shadow_cast_only)


def filter_components(state: np.ndarray, min_pixels: int) -> ComponentStats:
    """Turn FLOOD candidates in components smaller than ``min_pixels`` into NO_FLOOD (in place)."""
    cand = state == int(BaselineState.FLOOD)
    idx, roots, sizes = component_sizes(cand)
    small = sizes < min_pixels
    np.put(state, idx[small], int(BaselineState.NO_FLOOD))      # in place for any memory layout
    comp_sizes = np.bincount(roots)[np.unique(roots)] if idx.size else np.zeros(0, dtype=np.int64)
    kept = comp_sizes[comp_sizes >= min_pixels]
    return ComponentStats(
        candidate_pixels=int(idx.size), candidate_components=int(comp_sizes.size), kept_components=int(kept.size),
        removed_components=int((comp_sizes < min_pixels).sum()), removed_pixels=int(small.sum()),
        removed_pixel_share=round(int(small.sum()) / idx.size, 6) if idx.size else None,
        min_component_pixels=int(min_pixels), largest_component_px=int(comp_sizes.max()) if comp_sizes.size else 0,
        median_kept_component_px=float(np.median(kept)) if kept.size else None)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineJob:
    model_input: ModelInputJob
    dem_path: Path | None
    dem_sha256: str | None
    orbit_direction: str
    floodvit_prediction: Path | None
    floodvit_sha256: str | None
    final_dir: Path


def _post_dem(job: ModelInputJob) -> tuple[Path | None, str | None]:
    post = job.provenance.inputs[2]
    path = job.root / post.terrain_provenance
    if not path.is_file():
        return None, None
    try:
        tprov = load_contract(TerrainProvenance, path)
    except ContractIOError:
        return None, None
    if tprov.dem is None:
        return None, None
    dem = job.root / tprov.dem.path
    if not dem.is_file() or sha256_file(dem) != tprov.dem.sha256:
        raise BaselineError(f"{tprov.dem.path} is missing or changed since Step 3.4")
    return dem, tprov.dem.sha256


def prepare_baseline(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> BaselineJob:
    """Verify the stack (Step 4A checks), the POST GLO-30 mosaic and, if present, the FloodViT event output."""
    job = prepare_model_input(cfg, aoi_id, candidate_id)
    prov = job.provenance
    assert prov.output is not None
    dem, dem_sha = _post_dem(job)
    fv_pred = fv_sha = None
    fv_prov = cfg.settings.paths.interim / "inference" / aoi_id / candidate_id / "floodvit" / \
        "floodvit_inference_provenance.json"
    if fv_prov.is_file():
        ip = load_contract(FloodInferenceProvenance, assert_pipeline_input_allowed(fv_prov, cfg.settings.paths))
        candidate_pred = job.root / ip.outputs.prediction
        if ip.mode == "event" and ip.source.stack_sha256 == prov.output.stack_sha256 and candidate_pred.is_file() \
                and sha256_file(candidate_pred) == ip.outputs.prediction_sha256:
            fv_pred, fv_sha = candidate_pred, ip.outputs.prediction_sha256
    final_dir = assert_interim_output_dir(
        cfg.settings.paths.interim / DETECTION_GROUP / aoi_id / candidate_id / OUTPUT_SUBDIR, cfg.settings.paths)
    return BaselineJob(model_input=job, dem_path=dem, dem_sha256=dem_sha,
                       orbit_direction=prov.inputs[2].orbit_direction.value, floodvit_prediction=fv_pred,
                       floodvit_sha256=fv_sha, final_dir=final_dir)


@dataclass(frozen=True)
class BaselineMaps:
    state: np.ndarray
    statistics: BaselineStatistics
    comparison_counts: dict[str, int] | None


SLOPE_BIN_EDGES = (0.0, 5.0, 10.0)


def compute_baseline(job: BaselineJob, cfg: BatoConfig, *, params: BaselineParams | None = None,
                     progress: Callable[[str], None] | None = None) -> BaselineMaps:
    """Run the rule over the whole stack (read-only); returns the final state raster and statistics."""
    from rasterio.windows import Window

    require_rasterio()
    p = params or BaselineParams.from_config(cfg)
    say = progress or (lambda _t: None)
    mi = job.model_input
    prov = mi.provenance
    assert prov.grid is not None
    h, w = prov.grid.height, prov.grid.width
    look = LOOK_AZIMUTH_DEG[job.orbit_direction]
    features: DemFeatures | None = None
    shadow_grid: np.ndarray | None = None
    if job.dem_path is not None:
        with _open(job.dem_path) as ds:
            dem = np.asarray(ds.read(1))
            dem_tr, dem_nodata = tuple(ds.transform)[:6], ds.nodata
        features = dem_features(dem, dem_tr, dem_nodata)
        say("computing cast shadow on the GLO-30 mosaic ...\n")
        shadow_grid = cast_shadow(dem, dem_tr, dem_nodata, (look + 180.0) % 360.0, p.incidence_deg,
                                  p.shadow_search_m)
        del dem
    state = np.empty((h, w), dtype=np.uint8)
    cand_slope: list[np.ndarray] = []
    cand_sr: list[np.ndarray] = []
    cand_idx: list[np.ndarray] = []
    cand_change: dict[str, list[np.ndarray]] = {}
    cast_only = 0
    with _open(mi.stack_path) as stack_ds, _open(mi.mask_path) as mask_ds:
        check_rasters(stack_ds, mask_ds, prov)
        for row in range(0, h, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, h - row)
            win = Window(0, row, w, rows)
            raw = np.stack([np.asarray(stack_ds.read(k + 1, window=win), dtype=np.float32) for k in range(6)])
            valid = np.asarray(mask_ds.read(1, window=win), dtype=np.uint8)
            check_consistency(raw, valid, where=f"rows {row}-{row + rows - 1}")      # fail closed on NaN / mismatch
            cast = None
            if features is not None and shadow_grid is not None:
                f = sample_features(features, prov.grid.transform, row, rows, w, look, p.incidence_deg)
                slope, aspect = f["slope_deg"], f["aspect_deg"]
                cast = sample_grid(shadow_grid, features.transform, prov.grid.transform, row, rows, w, False)
            else:
                slope = aspect = np.full((rows, w), np.nan, dtype=np.float32)
            res = classify_block(raw, valid, slope, aspect, p, look, cast)
            cast_only += int(res.shadow_cast_only.sum())
            state[row:row + rows] = res.state
            sel = res.candidate
            cand_idx.append(np.flatnonzero(sel) + row * w)
            cand_slope.append(res.slope[sel])
            cand_sr.append(res.s_r[sel])
            for k, v in res.change.items():
                cand_change.setdefault(k, []).append(v[sel])
            if row == 0 or (row // BLOCK_ROWS) % 4 == 0:
                say(f"rows {row}-{row + rows - 1} of {h}\n")
    comps = filter_components(state, p.min_component_pixels)
    idx = np.concatenate(cand_idx)
    kept = state.ravel()[idx] == int(BaselineState.FLOOD)
    slope_k, sr_k = np.concatenate(cand_slope)[kept], np.concatenate(cand_sr)[kept]
    change_k = {k: np.concatenate(v)[kept] for k, v in cand_change.items()}
    edges = [e for e in SLOPE_BIN_EDGES if e < p.max_slope_deg] + [p.max_slope_deg]
    by_slope = {f"{lo:g}-{hi:g} deg": int(((slope_k >= lo) & (slope_k < hi)).sum())
                for lo, hi in zip(edges, edges[1:], strict=False)}
    by_geom = {"flat_or_gentle (|s_r| < 10 deg)": int((np.abs(sr_k) < GEOMETRY_FLAT_DEG).sum()),
               "facing_sensor (s_r >= 10 deg)": int((sr_k >= GEOMETRY_FLAT_DEG).sum()),
               "facing_away (s_r <= -10 deg)": int((sr_k <= -GEOMETRY_FLAT_DEG).sum()),
               "layover_likely": 0, "shadow_likely": 0}
    state_counts = {s.name: int(np.count_nonzero(state == int(s))) for s in BaselineState}
    total = h * w
    not_assessed = sum(state_counts[s.name] for s in NOT_ASSESSED_STATES)
    assessed = total - not_assessed
    flood = state_counts["FLOOD"]
    dark2 = int(((change_k["vv_vs_pre2"] <= p.pre2_change_db) & (change_k["vh_vs_pre2"] <= p.pre2_change_db)).sum())
    dark1 = int(((change_k["vv_vs_pre1"] <= p.pre1_change_db) & (change_k["vh_vs_pre1"] <= p.pre1_change_db)).sum())
    quant = {k: {f"p{q:02d}": round(float(np.percentile(v, q)), 3) for q in (5, 50, 95)}
             for k, v in change_k.items()} if flood else {}
    stats = BaselineStatistics(
        total_pixels=total, assessed_pixels=assessed, not_assessed_pixels=not_assessed, state_counts=state_counts,
        flood_pixels=flood, no_flood_pixels=state_counts["NO_FLOOD"],
        pre_event_water_pixels=state_counts["PRE_EVENT_WATER"],
        flood_fraction_of_assessed=round(flood / assessed, 8) if assessed else None,
        flood_fraction_of_all=round(flood / total, 8) if total else 0.0,
        not_assessed_fraction_of_all=round(not_assessed / total, 8) if total else 0.0,
        shadow_cast_only_pixels=cast_only, flood_by_slope_deg=by_slope, flood_by_geometry=by_geom,
        flood_darkened_vs_pre2=dark2,
        flood_darkened_vs_pre1=dark1, flood_change_db_quantiles=quant, components=comps)
    comparison = None
    if job.floodvit_prediction is not None:
        comparison = compare_floodvit(state, job.floodvit_prediction, prov.grid.transform)
    return BaselineMaps(state=state, statistics=stats, comparison_counts=comparison)


def compare_floodvit(state: np.ndarray, prediction: Path, grid_transform: Any = None) -> dict[str, int]:
    """Overlap of baseline FLOOD with FloodViT class 2 (comparison only; computed after the baseline)."""
    from rasterio.windows import Window

    h, w = state.shape
    out = dict.fromkeys(("both", "baseline_only", "floodvit_only", "floodvit_flood_in_not_assessed"), 0)
    not_assessed_codes = np.array([int(s) for s in NOT_ASSESSED_STATES], dtype=np.uint8)
    with _open(prediction) as ds:
        t = tuple(float(v) for v in tuple(ds.transform)[:6])
        if (ds.width, ds.height) != (w, h) or (grid_transform is not None and any(
                abs(x - float(y)) > 1e-6 for x, y in zip(t, grid_transform, strict=True))):
            raise BaselineError("FloodViT prediction is not on the stack grid")
        for row in range(0, h, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, h - row)
            fv = np.asarray(ds.read(1, window=Window(0, row, w, rows))) == 2
            st = state[row:row + rows]
            na = np.isin(st, not_assessed_codes)
            base = st == int(BaselineState.FLOOD)
            out["both"] += int((base & fv).sum())
            out["baseline_only"] += int((base & ~fv).sum())
            out["floodvit_only"] += int((~base & fv & ~na).sum())
            out["floodvit_flood_in_not_assessed"] += int((fv & na).sum())
    return out


def _write(path: Path, array: np.ndarray, transform: Any, *, nodata: int | None, description: str,
           tags: dict[str, str]) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.windows import Window

    h, w = array.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=CRS.from_epsg(3857),
                       transform=Affine(*transform), nodata=nodata, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", BIGTIFF="IF_SAFER") as dst:
        for row in range(0, h, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, h - row)
            dst.write(array[row:row + rows], 1, window=Window(0, row, w, rows))
        dst.set_band_description(1, description)
        dst.update_tags(**tags)


def flood_raster(state: np.ndarray) -> np.ndarray:
    out = np.zeros(state.shape, dtype=np.uint8)
    out[state == int(BaselineState.FLOOD)] = 1
    out[np.isin(state, np.array([int(s) for s in NOT_ASSESSED_STATES], dtype=np.uint8))] = FLOOD_NODATA
    return out


def run_baseline(job: BaselineJob, cfg: BatoConfig, *, overwrite: bool = False,
                 now: Callable[[], datetime] = _utc_now,
                 progress: Callable[[str], None] | None = None) -> tuple[Path, BaselineProvenance]:
    """Compute, write both rasters + provenance + diagnostics, publish atomically."""
    if job.final_dir.exists() and not overwrite:
        raise BaselineError(f"output already exists: {job.final_dir} (use --overwrite; the old output is kept as "
                            "<dir>.superseded-<time>)")
    started = now()
    maps = compute_baseline(job, cfg, progress=progress)
    mi = job.model_input
    prov = mi.provenance
    assert prov.grid is not None and prov.output is not None
    for path, expected in ((mi.stack_path, prov.output.stack_sha256), (mi.mask_path, prov.output.mask_sha256)):
        if sha256_file(path) != expected:
            raise BaselineError(f"{path.name} changed during the baseline run")
    tmp = assert_interim_output_dir(_sibling(job.final_dir, "tmp", started), cfg.settings.paths)
    tmp.mkdir(parents=True)
    tags = {"method": "baseline_change", "validated": "false", "config_fingerprint": cfg.fingerprint()}
    _write(tmp / STATE_FILE, maps.state, prov.grid.transform, nodata=None, description="baseline_state",
           tags={**tags, **{f"state_{int(s)}": s.name for s in BaselineState}})
    _write(tmp / FLOOD_FILE, flood_raster(maps.state), prov.grid.transform, nodata=FLOOD_NODATA,
           description="baseline_flood", tags={**tags, "value_0": "no flood (assessed)", "value_1": "flood",
                                               "value_255": "not assessed (nodata)"})
    comparison = None
    if maps.comparison_counts is not None and job.floodvit_prediction is not None and job.floodvit_sha256:
        c = maps.comparison_counts
        union = c["both"] + c["baseline_only"] + c["floodvit_only"]
        comparison = FloodViTComparison(
            floodvit_prediction=_relative(job.floodvit_prediction, mi.root),
            floodvit_prediction_sha256=job.floodvit_sha256,
            both=c["both"], baseline_only=c["baseline_only"], floodvit_only=c["floodvit_only"],
            jaccard=round(c["both"] / union, 6) if union else None,
            floodvit_flood_in_not_assessed=c["floodvit_flood_in_not_assessed"])
    s = maps.statistics
    warnings = []
    if s.not_assessed_fraction_of_all > 0.5:
        warnings.append(f"{s.not_assessed_fraction_of_all:.1%} of the scene is NOT_ASSESSED (terrain/observability):"
                        " the map makes no claim there")
    if job.dem_path is None:
        warnings.append("no GLO-30 mosaic found: every valid pixel is NOT_ASSESSED_NO_DEM")
    finished = now()
    provenance = BaselineProvenance(
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(), started_at=started, finished_at=finished,
        aoi_id=mi.aoi_id, candidate_id=mi.candidate_id, event_date=prov.event_date, source=source_stack_record(mi),
        dem=_relative(job.dem_path, mi.root) if job.dem_path else None, dem_sha256=job.dem_sha256,
        orbit_direction=job.orbit_direction, look_azimuth_deg=LOOK_AZIMUTH_DEG[job.orbit_direction],
        thresholds=threshold_records(cfg), rule=RULE,
        outputs=BaselineOutputs(
            directory=_relative(job.final_dir, mi.root), flood=_relative(job.final_dir / FLOOD_FILE, mi.root),
            flood_sha256=sha256_file(tmp / FLOOD_FILE), state=_relative(job.final_dir / STATE_FILE, mi.root),
            state_sha256=sha256_file(tmp / STATE_FILE),
            diagnostics=_relative(job.final_dir / DIAGNOSTICS_FILE, mi.root)),
        statistics=s, floodvit_comparison=comparison, warnings=tuple(warnings))
    save_contract(provenance, tmp / PROVENANCE_FILE)
    (tmp / DIAGNOSTICS_FILE).write_text("\n".join(describe_baseline(provenance)) + "\n", encoding="utf-8")
    if job.final_dir.exists():
        os.replace(job.final_dir, _sibling(job.final_dir, "superseded", finished))
    os.replace(tmp, job.final_dir)
    return job.final_dir, provenance


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.3f}%" if d else "n/a"


def describe_baseline(prov: BaselineProvenance) -> list[str]:
    s = prov.statistics
    c = s.components
    lines = ["# Baseline SAR change flood detection (deterministic; NOT VALIDATED)", "",
             f"AOI `{prov.aoi_id}`, candidate `{prov.candidate_id}`, event {prov.event_date}, "
             f"orbit {prov.orbit_direction} (look azimuth ~{prov.look_azimuth_deg:g} deg)", "",
             f"Rule: {prov.rule}", "", "## Thresholds (all provisional)", "",
             "| name | value | unit | basis |", "|---|---|---|---|"]
    lines += [f"| {t.name} | {t.value:g} | {t.unit} | {t.basis} |" for t in prov.thresholds]
    lines += ["", "## States", "", "| state | pixels | share of all |", "|---|---|---|"]
    lines += [f"| {name} | {n:,} | {_pct(n, s.total_pixels)} |" for name, n in s.state_counts.items()]
    lines += ["", "## Summary", "",
              f"- total pixels: {s.total_pixels:,}; assessed {s.assessed_pixels:,}; not assessed "
              f"{s.not_assessed_pixels:,} ({s.not_assessed_fraction_of_all:.2%})",
              f"- flood: {s.flood_pixels:,} = {_pct(s.flood_pixels, s.assessed_pixels)} of assessed pixels, "
              f"{_pct(s.flood_pixels, s.total_pixels)} of all pixels",
              f"- NOT_ASSESSED_SHADOW from cast shadow alone (shaded flat floors): {s.shadow_cast_only_pixels:,}",
              f"- flood by slope: {s.flood_by_slope_deg}", f"- flood by geometry: {s.flood_by_geometry}",
              f"- flood darkened vs PRE2 (VV and VH): {s.flood_darkened_vs_pre2:,} / {s.flood_pixels:,}; vs PRE1: "
              f"{s.flood_darkened_vs_pre1:,} / {s.flood_pixels:,} (by construction)",
              f"- flood change quantiles (dB): {s.flood_change_db_quantiles}",
              f"- components: {c.candidate_components:,} candidate components ({c.candidate_pixels:,} px); kept "
              f"{c.kept_components:,}; removed {c.removed_components:,} ({c.removed_pixels:,} px = "
              f"{c.removed_pixel_share}); min size {c.min_component_pixels}; largest {c.largest_component_px:,} px",
              f"- pre-event OSM waterway overlap: {s.osm_waterway_overlap}"]
    lines += ["", "## FloodViT comparison (not an input; FloodViT is preliminary and unvalidated)", ""]
    if prov.floodvit_comparison is None:
        lines.append("- no FloodViT event output for this stack")
    else:
        f = prov.floodvit_comparison
        lines.append(f"- both {f.both:,}; baseline only {f.baseline_only:,}; FloodViT only (assessed) "
                     f"{f.floodvit_only:,}; Jaccard {f.jaccard}; FloodViT flood inside NOT_ASSESSED "
                     f"{f.floodvit_flood_in_not_assessed:,}")
    lines += ["", f"Status: {prov.validation_note}"]
    lines += [f"WARN  {x}" for x in prov.warnings]
    return lines

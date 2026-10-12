"""Step 4B forensics: where FloodViT's water classes fall, under which inputs (torch-free, read-only).

Inputs, all BATO-internal (no reference/evaluation data, no published flood or damage map):
* the verified Step 3.5 stack + valid_mask.tif;
* the published FloodViT outputs of any of the modes event / no_event / zero_change / static_post
  (``bato.inference.floodvit``), each checked against its provenance (stack + prediction SHA-256);
* the Copernicus GLO-30 mosaic that Step 3.4 used for the POST pass (hash-checked).

Measured per mode, for class 2 (flood), class 1 (permanent water) and for all valid pixels as the base rate:
* saturation: POST VV above the Kuro Siwo clamp 0.15, near it (>= 0.135), any-date VV above it;
* SAR change: POST vs PRE2 VV darkening (<= -3 dB) / brightening (>= +3 dB);
* terrain (GLO-30, sampled nearest-neighbour at each 10 m pixel centre): slope bins, relative elevation
  above the local minimum within ~1 km ("valley-floor proxy", NOT a drainage model), approximate radar
  geometry from slope/aspect vs the look direction (facing sensor / layover-likely / facing away /
  shadow-likely) with a NOMINAL incidence angle - an approximation, not a SNAP layover/shadow mask;
* internal tile edges (within 16 px) vs interiors;
* connected-component sizes (8-connected; needs scipy, else reported as not computed);
* agreement between the event flood and each control mode's flood.
Scene-level: per-date VV/VH quantiles, clamp-saturation shares, model-normalised mean/std per channel
(in-domain data would sit near 0 / 1), PRE1->PRE2 and PRE2->POST change quantiles in dB.

Nothing here validates the map; it shows what the model responds to.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.contracts.inference import FloodInferenceProvenance
from bato.contracts.io import ContractIOError, load_contract
from bato.contracts.preprocessing import TerrainProvenance
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.inference.floodvit import (
    INFERENCE_GROUP,
    OUTPUT_SUBDIR,
    PROVENANCE_FILE,
    SEAM_BAND_PX,
    InferenceError,
    edge_band_mask,
)
from bato.preprocessing.model_input import STACK_BANDS, ModelInputJob, prepare_model_input
from bato.preprocessing.radiometric import _relative, _sibling, _utc_now
from bato.preprocessing.sigma0_validation import _open, require_rasterio
from bato.preprocessing.speckle import sha256_file
from bato.preprocessing.terrain_validation import EARTH_RADIUS_3857

DIAGNOSTICS_VERSION = "bato-floodvit-diagnostics-v1"
DIAGNOSTICS_SUBDIR = "diagnostics"
REPORT_JSON = "floodvit_diagnostics.json"
REPORT_MD = "floodvit_diagnostics.md"
CLAMP = 0.15
NEAR_CLAMP = 0.135
CHANGE_DB = 3.0
NOMINAL_INCIDENCE_DEG = 39.0               # Sentinel-1 IW mid-swath; the true angle varies ~30-46 deg across the swath
LOOK_AZIMUTH_DEG = {"ascending": 78.0, "descending": 282.0}   # right-looking, heading ~-12 / ~192 deg (approx.)
SLOPE_BINS = (0.0, 5.0, 15.0, 30.0, 91.0)
REL_ELEV_BINS = (0.0, 10.0, 50.0, math.inf)
REL_ELEV_WINDOW_M = 1000.0
FLAT_SLOPE_DEG = 5.0
GEOMETRY = ("flat_or_gentle", "facing_sensor", "layover_likely", "facing_away", "shadow_likely", "no_dem")
BLOCK_ROWS = 512
SAMPLE_STRIDE_TARGET = 2_000_000
M_PER_DEG = 111_320.0


# ---------------------------------------------------------------------------
# Terrain from the GLO-30 mosaic
# ---------------------------------------------------------------------------


def lonlat_from_3857(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(x / EARTH_RADIUS_3857)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS_3857)) - np.pi / 2)
    return lon, lat


def running_min(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Minimum over a centred window of 2 * radius + 1 along ``axis`` (inf-padded edges)."""
    if radius <= 0:
        return a.copy()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    p = np.pad(a, pad, constant_values=np.inf)
    n = a.shape[axis]
    out = np.take(p, np.arange(0, n), axis=axis)
    for k in range(1, 2 * radius + 1):
        out = np.minimum(out, np.take(p, np.arange(k, k + n), axis=axis))
    return out


@dataclass(frozen=True)
class DemFeatures:
    transform: tuple[float, float, float, float, float, float]   # geographic (degrees), rasterio order
    slope_deg: np.ndarray
    aspect_deg: np.ndarray                                        # downslope (facing) direction, clockwise from north
    rel_elev_m: np.ndarray                                        # elevation above the minimum within ~1 km


def dem_features(dem: np.ndarray, transform: Sequence[float], nodata: float | None,
                 window_m: float = REL_ELEV_WINDOW_M) -> DemFeatures:
    a, _b, _c, _d, e, f = (float(v) for v in transform[:6])
    z = dem.astype(np.float64)
    if nodata is not None and not math.isnan(nodata):
        z[z == nodata] = np.nan
    lat = f + (np.arange(z.shape[0]) + 0.5) * e
    dx = a * M_PER_DEG * np.cos(np.radians(lat))[:, None]          # metres per column, per row
    dy_north = e * M_PER_DEG                                        # metres north per row (negative)
    dz_col = np.gradient(z, axis=1)
    dz_row = np.gradient(z, axis=0)
    dzdx, dzdn = dz_col / dx, dz_row / dy_north
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    aspect = np.degrees(np.arctan2(-dzdx, -dzdn)) % 360.0
    finite = np.where(np.isfinite(z), z, np.inf)
    ry = max(1, int(round(window_m / 2 / abs(dy_north))))
    rx = max(1, int(round(window_m / 2 / float(np.nanmedian(dx)))))
    local_min = running_min(running_min(finite, ry, 0), rx, 1)
    rel = np.where(np.isfinite(z) & np.isfinite(local_min), z - local_min, np.nan)
    return DemFeatures(transform=tuple(float(v) for v in transform[:6]),  # type: ignore[arg-type]
                       slope_deg=slope.astype(np.float32), aspect_deg=aspect.astype(np.float32),
                       rel_elev_m=rel.astype(np.float32))


def radar_geometry(slope_deg: np.ndarray, aspect_deg: np.ndarray, look_azimuth: float,
                   incidence: float = NOMINAL_INCIDENCE_DEG) -> np.ndarray:
    """Class index into GEOMETRY from the slope component in the range direction (approximate)."""
    sensor_az = (look_azimuth + 180.0) % 360.0
    s_r = np.degrees(np.arctan(np.tan(np.radians(slope_deg)) * np.cos(np.radians(aspect_deg - sensor_az))))
    cls = np.zeros(slope_deg.shape, dtype=np.uint8)
    cls[(s_r >= 10.0) & (s_r <= incidence)] = 1
    cls[s_r > incidence] = 2
    cls[(s_r <= -10.0) & (-s_r < 90.0 - incidence)] = 3
    cls[-s_r >= 90.0 - incidence] = 4
    cls[~np.isfinite(slope_deg) | ~np.isfinite(aspect_deg)] = 5
    return cls


def sample_features(features: DemFeatures, grid_transform: Sequence[float], row0: int, rows: int, width: int,
                    look_azimuth: float, incidence: float) -> dict[str, np.ndarray]:
    """Nearest-neighbour terrain features at the 10 m stack pixel centres of rows row0..row0+rows-1."""
    ga, _gb, gc, _gd, ge, gf = (float(v) for v in grid_transform[:6])
    x = gc + (np.arange(width) + 0.5) * ga
    y = gf + (row0 + np.arange(rows) + 0.5) * ge
    lon, _ = lonlat_from_3857(x, np.zeros_like(x))
    _, lat = lonlat_from_3857(np.zeros_like(y), y)
    a, _b, c, _d, e, f = features.transform
    col = np.floor((lon - c) / a).astype(np.int64)
    row = np.floor((lat - f) / e).astype(np.int64)
    h, w = features.slope_deg.shape
    ok = (row[:, None] >= 0) & (row[:, None] < h) & (col[None, :] >= 0) & (col[None, :] < w)
    rr = np.clip(row, 0, h - 1)[:, None]
    cc = np.clip(col, 0, w - 1)[None, :]
    out = {}
    for name in ("slope_deg", "aspect_deg", "rel_elev_m"):
        arr = getattr(features, name)[rr, cc]
        out[name] = np.where(ok, arr, np.nan).astype(np.float32)
    out["geometry"] = radar_geometry(out["slope_deg"], out["aspect_deg"], look_azimuth, incidence)
    return out


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


def _db(x: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(x, 1e-6))).astype(np.float32)


def bin_labels(edges: Sequence[float], unit: str) -> list[str]:
    return [f"{lo:g}-{hi:g} {unit}" if math.isfinite(hi) else f">= {lo:g} {unit}" for lo, hi in zip(edges, edges[1:],
                                                                                                    strict=False)]


SLOPE_LABELS = bin_labels(SLOPE_BINS, "deg")
REL_LABELS = bin_labels(REL_ELEV_BINS, "m")
CATEGORIES = ("post_vv_above_clamp", "post_vv_near_clamp", "any_vv_above_clamp", "post_vs_pre2_vv_darkening_3db",
              "post_vs_pre2_vv_brightening_3db", "near_internal_tile_edge", "valley_floor_proxy",
              *[f"slope {x}" for x in SLOPE_LABELS], *[f"rel_elev {x}" for x in REL_LABELS], "rel_elev unknown",
              *[f"geometry {g}" for g in GEOMETRY])


class Tally:
    """Counts of pixels per category for one selection (e.g. event flood)."""

    def __init__(self) -> None:
        self.n = 0
        self.counts = dict.fromkeys(CATEGORIES, 0)

    def add(self, sel: np.ndarray, cats: Mapping[str, np.ndarray]) -> None:
        self.n += int(sel.sum())
        for k, m in cats.items():
            self.counts[k] += int((sel & m).sum())

    def result(self) -> dict[str, Any]:
        return {"pixels": self.n, "shares": {k: (round(v / self.n, 6) if self.n else None)
                                             for k, v in self.counts.items()}}


def block_categories(raw: np.ndarray, feats: Mapping[str, np.ndarray], near_edge: np.ndarray) -> dict[str, np.ndarray]:
    pre1_vv, pre2_vv, post_vv = raw[0], raw[2], raw[4]
    change = _db(post_vv) - _db(pre2_vv)
    slope, rel, geo = feats["slope_deg"], feats["rel_elev_m"], feats["geometry"]
    cats = {"post_vv_above_clamp": post_vv > CLAMP, "post_vv_near_clamp": post_vv >= NEAR_CLAMP,
            "any_vv_above_clamp": (pre1_vv > CLAMP) | (pre2_vv > CLAMP) | (post_vv > CLAMP),
            "post_vs_pre2_vv_darkening_3db": change <= -CHANGE_DB,
            "post_vs_pre2_vv_brightening_3db": change >= CHANGE_DB, "near_internal_tile_edge": near_edge,
            "valley_floor_proxy": (slope < 5.0) & (rel < 10.0), "rel_elev unknown": ~np.isfinite(rel)}
    for lab, lo, hi in zip(SLOPE_LABELS, SLOPE_BINS, SLOPE_BINS[1:], strict=False):
        cats[f"slope {lab}"] = (slope >= lo) & (slope < hi)
    for lab, lo, hi in zip(REL_LABELS, REL_ELEV_BINS, REL_ELEV_BINS[1:], strict=False):
        cats[f"rel_elev {lab}"] = (rel >= lo) & (rel < hi)
    for k, g in enumerate(GEOMETRY):
        cats[f"geometry {g}"] = geo == k
    return cats


def component_sizes(mask: np.ndarray) -> dict[str, Any]:
    try:
        ndimage = importlib.import_module("scipy.ndimage")
    except ImportError:
        return {"computed": False, "reason": "scipy not installed"}
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0, dtype=np.int64)
    total = int(sizes.sum())

    def share(lo: int, hi: float) -> float | None:
        return round(float(sizes[(sizes >= lo) & (sizes < hi)].sum()) / total, 6) if total else None

    return {"computed": True, "components": int(n), "largest_px": int(sizes.max()) if n else 0,
            "median_px": float(np.median(sizes)) if n else None,
            "pixel_share_in_components": {"1 px": share(1, 2), "2-9 px": share(2, 10), "10-99 px": share(10, 100),
                                          "100-999 px": share(100, 1000), ">= 1000 px": share(1000, math.inf)}}


@dataclass(frozen=True)
class ModeInput:
    mode: str
    prediction: Path
    provenance: FloodInferenceProvenance


def collect(stack_ds: Any, mask_ds: Any, predictions: Mapping[str, Any], features: DemFeatures | None,
            grid_transform: Sequence[float], row_offsets: Sequence[int], col_offsets: Sequence[int], *,
            look_azimuth: float, incidence: float = NOMINAL_INCIDENCE_DEG, tile: int = 224,
            seam_band: int = SEAM_BAND_PX, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Single pass over the stack, mask, predictions (opened datasets) and terrain features."""
    from rasterio.windows import Window

    say = progress or (lambda _t: None)
    h, w = stack_ds.height, stack_ds.width
    near_rows = edge_band_mask(h, row_offsets, tile, seam_band)
    near_cols = edge_band_mask(w, col_offsets, tile, seam_band)
    stride = max(1, int(math.ceil(math.sqrt(h * w / SAMPLE_STRIDE_TARGET))))
    tallies = {(m, k): Tally() for m in predictions for k in ("flood", "permanent_water")}
    base = Tally()
    floods = {m: np.zeros((h, w), dtype=bool) for m in predictions}
    sat = np.zeros(6, dtype=np.int64)
    sums = np.zeros(6)
    sumsq = np.zeros(6)
    n_valid = 0
    samples: dict[str, list[np.ndarray]] = {n: [] for n in STACK_BANDS}
    changes: dict[str, list[np.ndarray]] = {k: [] for k in ("PRE1->PRE2 VV", "PRE2->POST VV", "PRE1->PRE2 VH",
                                                            "PRE2->POST VH", "POST VH-VV ratio", "PRE2 VH-VV ratio")}
    flat_n = 0
    flat_sat = np.zeros(6, dtype=np.int64)
    flat_samples: dict[str, list[np.ndarray]] = {n: [] for n in STACK_BANDS}
    means = np.array([0.0953, 0.0264] * 3)
    stds = np.array([0.0427, 0.0215] * 3)
    for row in range(0, h, BLOCK_ROWS):
        rows = min(BLOCK_ROWS, h - row)
        win = Window(0, row, w, rows)
        raw = np.stack([np.asarray(stack_ds.read(k + 1, window=win), dtype=np.float32) for k in range(6)])
        valid = np.asarray(mask_ds.read(1, window=win)) == 1
        if features is not None:
            feats = sample_features(features, grid_transform, row, rows, w, look_azimuth, incidence)
        else:
            nan = np.full((rows, w), np.nan, dtype=np.float32)
            feats = {"slope_deg": nan, "aspect_deg": nan, "rel_elev_m": nan,
                     "geometry": np.full((rows, w), 5, dtype=np.uint8)}
        near = near_rows[row:row + rows, None] | near_cols[None, :]
        cats = block_categories(raw, feats, near)
        base.add(valid, cats)
        for m, ds in predictions.items():
            pred = np.asarray(ds.read(1, window=win))
            tallies[(m, "flood")].add(valid & (pred == 2), cats)
            tallies[(m, "permanent_water")].add(valid & (pred == 1), cats)
            floods[m][row:row + rows] = valid & (pred == 2)
        flat = valid & (feats["slope_deg"] < FLAT_SLOPE_DEG)
        flat_n += int(flat.sum())
        flat_sat += (raw[:, flat] > CLAMP).sum(axis=1)
        v = raw[:, valid]
        n_valid += v.shape[1]
        sat += (v > CLAMP).sum(axis=1)
        z = (np.clip(v, 0.0, CLAMP) - means[:, None]) / stds[:, None]
        sums += z.sum(axis=1)
        sumsq += (z.astype(np.float64) ** 2).sum(axis=1)
        sub = (np.arange(row, row + rows) % stride == 0)
        if sub.any():
            pick = valid[sub][:, ::stride]
            r = raw[:, sub][:, :, ::stride]
            flat_pick = flat[sub][:, ::stride]
            for k, name in enumerate(STACK_BANDS):
                samples[name].append(r[k][pick])
                flat_samples[name].append(r[k][flat_pick])
            d = {name: _db(r[k][pick]) for k, name in enumerate(STACK_BANDS)}
            changes["PRE1->PRE2 VV"].append(d["PRE2_VV"] - d["PRE1_VV"])
            changes["PRE2->POST VV"].append(d["POST_VV"] - d["PRE2_VV"])
            changes["PRE1->PRE2 VH"].append(d["PRE2_VH"] - d["PRE1_VH"])
            changes["PRE2->POST VH"].append(d["POST_VH"] - d["PRE2_VH"])
            changes["POST VH-VV ratio"].append(d["POST_VH"] - d["POST_VV"])
            changes["PRE2 VH-VV ratio"].append(d["PRE2_VH"] - d["PRE2_VV"])
        if row == 0 or (row // BLOCK_ROWS) % 4 == 0:
            say(f"rows {row}-{row + rows - 1} of {h}\n")

    def quant(parts: list[np.ndarray]) -> dict[str, float] | None:
        a = np.concatenate(parts) if parts else np.zeros(0)
        if not a.size:
            return None
        return {f"p{p:02d}": round(float(np.percentile(a, p)), 5) for p in (1, 5, 25, 50, 75, 95, 99)}

    mean = sums / max(n_valid, 1)
    scene = {
        "valid_pixels": n_valid, "sample_stride": stride,
        "above_clamp_share": {name: round(int(sat[k]) / n_valid, 6) if n_valid else None
                              for k, name in enumerate(STACK_BANDS)},
        "kuro_cantelli_bound_above_clamp": {"VV": 0.379, "VH": 0.029},
        "normalised_mean": {name: round(float(mean[k]), 4) for k, name in enumerate(STACK_BANDS)},
        "normalised_std": {name: round(float(math.sqrt(max(sumsq[k] / max(n_valid, 1) - mean[k] ** 2, 0.0))), 4)
                           for k, name in enumerate(STACK_BANDS)},
        "linear_quantiles": {name: quant(parts) for name, parts in samples.items()},
        "change_db_quantiles": {name: quant(parts) for name, parts in changes.items()},
        "flat_terrain": {
            "definition": f"valid pixels with GLO-30 slope < {FLAT_SLOPE_DEG:g} deg "
                          "(terrain cannot explain brightness there)",
            "pixels": flat_n,
            "above_clamp_share": {name: round(int(flat_sat[k]) / flat_n, 6) if flat_n else None
                                  for k, name in enumerate(STACK_BANDS)},
            "linear_quantiles": {name: quant(parts) for name, parts in flat_samples.items()}},
    }
    modes: dict[str, Any] = {}
    event = floods.get("event")
    for m in predictions:
        entry = {"flood": tallies[(m, "flood")].result(), "permanent_water": tallies[(m, "permanent_water")].result(),
                 "flood_components": component_sizes(floods[m])}
        if event is not None and m != "event":
            both = int((event & floods[m]).sum())
            union = int((event | floods[m]).sum())
            entry["agreement_with_event_flood"] = {
                "intersection_px": both, "jaccard": round(both / union, 6) if union else None,
                "share_of_event_flood_also_flagged": round(both / int(event.sum()), 6) if event.sum() else None}
        modes[m] = entry
    return {"scene": scene, "all_valid_pixels": base.result(), "modes": modes}


# ---------------------------------------------------------------------------
# Job, run, publish
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsJob:
    model_input: ModelInputJob
    modes: tuple[ModeInput, ...]
    dem_path: Path | None
    dem_sha256: str | None
    orbit_direction: str
    final_dir: Path


def prepare_diagnostics(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> DiagnosticsJob:
    """Verify the stack, every published FloodViT mode output and the POST DEM mosaic (nothing written)."""
    job = prepare_model_input(cfg, aoi_id, candidate_id)
    prov = job.provenance
    assert prov.output is not None
    base = cfg.settings.paths.interim / INFERENCE_GROUP / aoi_id / candidate_id
    modes = []
    for mode, sub in OUTPUT_SUBDIR.items():
        ppath = base / sub / PROVENANCE_FILE
        if not ppath.is_file():
            continue
        try:
            ip = load_contract(FloodInferenceProvenance, assert_pipeline_input_allowed(ppath, cfg.settings.paths))
        except ContractIOError as exc:
            raise InferenceError(str(exc)) from exc
        if ip.source.stack_sha256 != prov.output.stack_sha256 or ip.mode != mode:
            raise InferenceError(f"{sub}: made from a different stack or mode")
        pred = job.root / ip.outputs.prediction
        if not pred.is_file() or sha256_file(pred) != ip.outputs.prediction_sha256:
            raise InferenceError(f"{sub}: prediction missing or changed since inference")
        modes.append(ModeInput(mode=mode, prediction=pred, provenance=ip))
    if not modes:
        raise InferenceError(f"no FloodViT output under {_relative(base, job.root)} "
                             "(run scripts/run_floodvit_inference.py)")
    post = prov.inputs[2]
    dem_path = dem_sha = None
    tprov_path = job.root / post.terrain_provenance
    if tprov_path.is_file():
        tprov = load_contract(TerrainProvenance, tprov_path)
        if tprov.dem is not None:
            candidate = job.root / tprov.dem.path
            if candidate.is_file() and sha256_file(candidate) == tprov.dem.sha256:
                dem_path, dem_sha = candidate, tprov.dem.sha256
    final_dir = assert_interim_output_dir(base / DIAGNOSTICS_SUBDIR, cfg.settings.paths)
    return DiagnosticsJob(model_input=job, modes=tuple(modes), dem_path=dem_path, dem_sha256=dem_sha,
                          orbit_direction=post.orbit_direction.value, final_dir=final_dir)


def run_diagnostics(job: DiagnosticsJob, cfg: BatoConfig, *, incidence: float = NOMINAL_INCIDENCE_DEG,
                    overwrite: bool = False, now: Callable[[], datetime] = _utc_now,
                    progress: Callable[[str], None] | None = None) -> tuple[Path, dict[str, Any]]:
    require_rasterio()
    if job.final_dir.exists() and not overwrite:
        raise InferenceError(f"output already exists: {job.final_dir} (use --overwrite)")
    mi = job.model_input
    prov = mi.provenance
    assert prov.grid is not None and prov.output is not None
    started = now()
    features = None
    if job.dem_path is not None:
        with _open(job.dem_path) as ds:
            features = dem_features(np.asarray(ds.read(1)), tuple(ds.transform)[:6], ds.nodata)
    tiling = job.modes[0].provenance.tiling
    preds = {m.mode: _open(m.prediction) for m in job.modes}
    try:
        with _open(mi.stack_path) as stack_ds, _open(mi.mask_path) as mask_ds:
            report = collect(stack_ds, mask_ds, preds, features, prov.grid.transform, tiling.row_offsets,
                             tiling.col_offsets, look_azimuth=LOOK_AZIMUTH_DEG[job.orbit_direction],
                             incidence=incidence, tile=tiling.tile_size, progress=progress)
    finally:
        for ds in preds.values():
            ds.close()
    report = {
        "kind": "floodvit_diagnostics", "version": DIAGNOSTICS_VERSION, "bato_version": bato.__version__,
        "created_at": started.isoformat(), "aoi_id": mi.aoi_id, "candidate_id": mi.candidate_id,
        "validated": False,
        "note": "diagnostics of model behaviour only; no reference or evaluation data used; not a validation",
        "inputs": {"stack": prov.output.stack, "stack_sha256": prov.output.stack_sha256,
                   "dem": _relative(job.dem_path, mi.root) if job.dem_path else None, "dem_sha256": job.dem_sha256,
                   "modes": {m.mode: {"prediction": _relative(m.prediction, mi.root),
                                      "prediction_sha256": m.provenance.outputs.prediction_sha256,
                                      "mapping": list(m.provenance.model_input_mapping)} for m in job.modes}},
        "approximations": {"incidence_deg": incidence, "look_azimuth_deg": LOOK_AZIMUTH_DEG[job.orbit_direction],
                           "orbit_direction": job.orbit_direction,
                           "terrain_sampling": "GLO-30 nearest neighbour at 10 m pixel centres",
                           "relative_elevation": f"elevation minus minimum within ~{REL_ELEV_WINDOW_M:g} m "
                                                 "(valley-floor proxy, not a drainage model)",
                           "radar_geometry": "slope component toward the sensor vs nominal incidence; NOT a SNAP "
                                             "layover/shadow mask"},
        **report}
    tmp = assert_interim_output_dir(_sibling(job.final_dir, "tmp", started), cfg.settings.paths)
    tmp.mkdir(parents=True)
    (tmp / REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (tmp / REPORT_MD).write_text("\n".join(render_markdown(report)) + "\n", encoding="utf-8")
    if job.final_dir.exists():
        os.replace(job.final_dir, _sibling(job.final_dir, "superseded", now()))
    os.replace(tmp, job.final_dir)
    return job.final_dir, report


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


KEY_ROWS = ("post_vv_above_clamp", "any_vv_above_clamp", "post_vs_pre2_vv_darkening_3db", "valley_floor_proxy",
            "slope 15-30 deg", "slope 30-91 deg", "rel_elev >= 50 m", "geometry facing_sensor",
            "geometry layover_likely", "geometry shadow_likely", "near_internal_tile_edge")


def render_markdown(report: Mapping[str, Any]) -> list[str]:
    modes = report["modes"]
    lines = ["# FloodViT diagnostics (model behaviour, NOT validation)", "",
             f"AOI `{report['aoi_id']}`, candidate `{report['candidate_id']}`, {report['created_at']}", "",
             "## Shares of each selection falling in each category", "",
             "| category | all valid | " + " | ".join(f"{m} flood | {m} perm. water" for m in modes) + " |",
             "|---|---|" + "---|---|" * len(modes)]
    lines.append("| pixels | " + f"{report['all_valid_pixels']['pixels']:,} | " + " | ".join(
        f"{modes[m]['flood']['pixels']:,} | {modes[m]['permanent_water']['pixels']:,}" for m in modes) + " |")
    for key in KEY_ROWS:
        lines.append(f"| {key} | {_pct(report['all_valid_pixels']['shares'][key])} | " + " | ".join(
            f"{_pct(modes[m]['flood']['shares'][key])} | {_pct(modes[m]['permanent_water']['shares'][key])}"
            for m in modes) + " |")
    lines += ["", "## Agreement with the event flood", ""]
    for m, e in modes.items():
        if "agreement_with_event_flood" in e:
            a = e["agreement_with_event_flood"]
            lines.append(f"- {m}: {a['intersection_px']:,} px shared, Jaccard {a['jaccard']}, "
                         f"{_pct(a['share_of_event_flood_also_flagged'])} of event flood also flagged")
    lines += ["", "## Flood clusters", ""]
    for m, e in modes.items():
        c = e["flood_components"]
        lines.append(f"- {m}: " + (f"{c['components']:,} components, largest {c['largest_px']:,} px, pixel shares "
                                   f"{c['pixel_share_in_components']}" if c.get("computed") else c.get("reason", "")))
    s = report["scene"]
    lines += ["", "## Scene", "", f"- above clamp: {s['above_clamp_share']} (Kuro Siwo maximum possible: VV 37.9 %, "
              "VH 2.9 %, from its own mean/std)",
              f"- flat terrain (slope < {FLAT_SLOPE_DEG:g} deg, {s['flat_terrain']['pixels']:,} px) above clamp: "
              f"{s['flat_terrain']['above_clamp_share']}",
              f"- flat terrain linear quantiles: {s['flat_terrain']['linear_quantiles']}",
              f"- model-normalised mean: {s['normalised_mean']}",
              f"- model-normalised std: {s['normalised_std']}", f"- change dB quantiles: {s['change_db_quantiles']}",
              "", f"Approximations: {report['approximations']}"]
    return lines

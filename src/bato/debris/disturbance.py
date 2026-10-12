"""Deterministic SAR disturbance (debris) detector on the Step 3.5 stack. Not a learned model: no permitted
training dataset has debris labels.

Rule (per polarisation p in {VV, VH}, Lee Sigma-filtered sigma0 in dB, same track on all dates):

    d1 = POST - PRE1        d2 = POST - PRE2        d0 = PRE2 - PRE1
    sigma_p = 1.4826 * MAD(d0) over valid, geometry-observable pixels (pre-event null model, recorded)
    increase_p = (d1 >= z * sigma_p) & (d2 >= z * sigma_p) & (|d0| < z * sigma_p)
    candidate  = increase_VV & increase_VH & not FLOOD & not PRE_EVENT_WATER (Step 4C state)
    DISTURBANCE = candidate in an 8-connected cluster >= detection.min_component_pixels

    z = detection.debris_z_min (3.0), min_component_pixels = detection.min_component_pixels (10): existing
    provisional thresholds, unchanged and not tuned.

Geometry gates come unchanged from the Step 4C state raster (same functions and thresholds): invalid SAR (10),
no DEM (11), layover (12), local or cast shadow (13). Step 4C's STEEP (14) is a water-ponding plausibility gate,
not a visibility gate, so steep pixels ARE assessed here: all three images share one track, so foreshortening is
identical on every date and cancels in the differences.

Scope: brightening in both polarisations, new against both pre-event images and stable before the event -
consistent with fresh debris / rough deposits. Darkening (e.g. vegetation stripped from a scar) is counted as a
diagnostic only and never becomes a claim.

Outputs: ``<interim>/detection/<aoi>/<candidate>/debris/``
    debris_state.tif          uint8 DebrisState codes (bato.contracts.debris)
    change_table.npz          every DISTURBANCE and Step 4C FLOOD pixel: row, col, kind, d1/d2/d0 VV/VH, slope, s_r
    debris_provenance.json    inputs + hashes, noise model (sigma), thresholds, statistics, validated: false
    debris_diagnostics.md
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.contracts.baseline import BaselineProvenance, BaselineState, ThresholdRecord
from bato.contracts.debris import (
    DEBRIS_ASSESSED,
    BaselineRef,
    DebrisComponents,
    DebrisOutputs,
    DebrisProvenance,
    DebrisState,
    DebrisStatistics,
    NoiseModel,
)
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.detection.baseline import (
    BLOCK_ROWS,
    DETECTION_GROUP,
    GEOMETRY_FLAT_DEG,
    _write,
    range_slope_deg,
    to_db,
)
from bato.detection.baseline import OUTPUT_SUBDIR as BASELINE_SUBDIR
from bato.detection.baseline import PROVENANCE_FILE as BASELINE_PROVENANCE
from bato.detection.components import component_sizes
from bato.exceptions import BatoError
from bato.inference.diagnostics import LOOK_AZIMUTH_DEG, DemFeatures, dem_features, sample_features
from bato.preprocessing.model_input import ModelInputJob, check_consistency, prepare_model_input, source_stack_record
from bato.preprocessing.radiometric import _relative, _sibling, _utc_now
from bato.preprocessing.sigma0_validation import _open, require_rasterio
from bato.preprocessing.speckle import sha256_file

OUTPUT_SUBDIR = "debris"
STATE_FILE = "debris_state.tif"
TABLE_FILE = "change_table.npz"
PROVENANCE_FILE = "debris_provenance.json"
DIAGNOSTICS_FILE = "debris_diagnostics.md"
MAD_TO_SIGMA = 1.4826
EDGE_BAND_PX = 100                    # 1 km border band (reporting criterion (d) only)
SLOPE_BINS = ((0, 5), (5, 15), (15, 30), (30, 45), (45, 91))
KIND_FLOOD, KIND_DISTURBANCE = 1, 2
GEOMETRY_CODES = {int(s): int(s) for s in (BaselineState.NOT_ASSESSED_INVALID_SAR, BaselineState.NOT_ASSESSED_NO_DEM,
                                           BaselineState.NOT_ASSESSED_LAYOVER, BaselineState.NOT_ASSESSED_SHADOW)}
OBSERVABLE_STEP4C = np.array([int(BaselineState.NO_FLOOD), int(BaselineState.FLOOD),
                              int(BaselineState.PRE_EVENT_WATER), int(BaselineState.NOT_ASSESSED_STEEP)],
                             dtype=np.uint8)
EXCLUDED_STEP4C = np.array([int(BaselineState.FLOOD), int(BaselineState.PRE_EVENT_WATER)], dtype=np.uint8)
RULE = ("d1=POST-PRE1, d2=POST-PRE2, d0=PRE2-PRE1 (dB, per pol); sigma=1.4826*MAD(d0) over valid geometry-observable "
        "pixels; increase=(d1>=z*sigma)&(d2>=z*sigma)&(|d0|<z*sigma); candidate=increase_VV&increase_VH&!FLOOD&"
        "!PRE_EVENT_WATER; DISTURBANCE=candidate in 8-connected cluster >= min_component_pixels. Not assessed: "
        "Step 4C invalid SAR / no DEM / layover / shadow (steep IS assessed).")
TABLE_COLUMNS = ("row", "col", "kind", "d1_vv", "d2_vv", "d0_vv", "d1_vh", "d2_vh", "d0_vh", "slope_deg", "s_r_deg")


class DebrisError(BatoError):
    """Disturbance detection inputs are missing or inconsistent (fail closed)."""


# ---------------------------------------------------------------------------
# Pure rule (testable without rasterio)
# ---------------------------------------------------------------------------


def differences(raw: np.ndarray) -> dict[str, np.ndarray]:
    """d1/d2/d0 per polarisation from a (6, h, w) layout-v1 block (PRE1_VV PRE1_VH PRE2_VV PRE2_VH POST_VV POST_VH)."""
    pre1_vv, pre1_vh, pre2_vv, pre2_vh, post_vv, post_vh = to_db(raw)
    return {"d1_vv": post_vv - pre1_vv, "d2_vv": post_vv - pre2_vv, "d0_vv": pre2_vv - pre1_vv,
            "d1_vh": post_vh - pre1_vh, "d2_vh": post_vh - pre2_vh, "d0_vh": pre2_vh - pre1_vh}


def observable(step4c_state: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid and not geometry-gated by Step 4C (STEEP counts as observable)."""
    return (valid == 1) & np.isin(step4c_state, OBSERVABLE_STEP4C)


def noise_model(d0: np.ndarray, polarization: str) -> NoiseModel:
    """Robust null model of the pre-event pair: median and 1.4826 * MAD of d0 (dB)."""
    v = np.asarray(d0, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise DebrisError(f"no valid geometry-observable pixel to estimate the {polarization} noise model")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = MAD_TO_SIGMA * mad
    if not sigma > 0:
        raise DebrisError(f"{polarization} pre-event difference has zero spread (MAD = 0); cannot set a threshold")
    return NoiseModel(polarization="VV" if polarization == "VV" else "VH", median_d0_db=round(med, 6),
                      sigma_db=round(sigma, 6), pixels=int(v.size))


@dataclass
class BlockRule:
    state: np.ndarray            # uint8: 0 assessed / 10-13 not assessed (candidates still 0 here)
    candidate: np.ndarray        # bool
    darkening: np.ndarray        # bool, diagnostic only
    d: dict[str, np.ndarray]


def classify_debris_block(raw: np.ndarray, valid: np.ndarray, step4c_state: np.ndarray, sigma_vv: float,
                          sigma_vh: float, z: float) -> BlockRule:
    if raw.ndim != 3 or raw.shape[0] != 6 or raw.shape[1:] != valid.shape or step4c_state.shape != valid.shape:
        raise DebrisError(f"block shapes {raw.shape} / {valid.shape} / {step4c_state.shape} do not match")
    d = differences(raw)
    obs = observable(step4c_state, valid)
    state = np.zeros(valid.shape, dtype=np.uint8)
    gated = ~obs
    # geometry codes exactly as Step 4C assigned them; an invalid pixel Step 4C did not mark is invalid SAR
    geo = np.where(np.isin(step4c_state, np.array(list(GEOMETRY_CODES), dtype=np.uint8)), step4c_state,
                   np.uint8(DebrisState.NOT_ASSESSED_INVALID_SAR))
    state[gated] = geo[gated]

    def change(sign: int, pol: str, sigma: float) -> np.ndarray:
        t = z * sigma
        d1, d2, d0 = d[f"d1_{pol}"], d[f"d2_{pol}"], d[f"d0_{pol}"]
        stable = np.abs(d0) < t
        if sign > 0:
            return (d1 >= t) & (d2 >= t) & stable
        return (d1 <= -t) & (d2 <= -t) & stable

    not_water = ~np.isin(step4c_state, EXCLUDED_STEP4C)
    candidate = obs & not_water & change(1, "vv", sigma_vv) & change(1, "vh", sigma_vh)
    darkening = obs & not_water & change(-1, "vv", sigma_vv) & change(-1, "vh", sigma_vh)
    return BlockRule(state=state, candidate=candidate, darkening=darkening, d=d)


def keep_clusters(candidate: np.ndarray, min_pixels: int) -> tuple[np.ndarray, DebrisComponents]:
    """8-connected clusters >= min_pixels (same labelling as the Step 4C cluster filter)."""
    idx, roots, sizes = component_sizes(candidate)
    kept = np.zeros(candidate.shape, dtype=bool)
    keep = sizes >= min_pixels
    np.put(kept, idx[keep], True)
    comps = DebrisComponents(
        candidate_pixels=int(idx.size), candidate_components=int(np.unique(roots).size) if idx.size else 0,
        kept_components=int(np.unique(roots[keep]).size) if keep.any() else 0,
        removed_pixels=int((~keep).sum()), min_component_pixels=int(min_pixels),
        largest_component_px=int(sizes.max()) if sizes.size else 0)
    return kept, comps


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebrisJob:
    model_input: ModelInputJob
    baseline_provenance_path: Path
    baseline_provenance_sha256: str
    baseline: BaselineProvenance
    step4c_state_path: Path
    dem_path: Path | None
    dem_sha256: str | None
    orbit_direction: str
    final_dir: Path


def output_dir(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> Path:
    for part in (aoi_id, candidate_id):
        if not part or any(c in part for c in "/\\:") or part in (".", ".."):
            raise DebrisError(f"invalid path component {part!r}")
    return assert_interim_output_dir(cfg.settings.paths.interim / DETECTION_GROUP / aoi_id / candidate_id /
                                     OUTPUT_SUBDIR, cfg.settings.paths)


def prepare_debris(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> DebrisJob:
    """Verify the stack (Step 4A checks) and the Step 4C run made from that exact stack."""
    try:
        mi = prepare_model_input(cfg, aoi_id, candidate_id)
    except BatoError as exc:
        raise DebrisError(str(exc)) from exc
    prov = mi.provenance
    assert prov.output is not None
    base = cfg.settings.paths.interim / DETECTION_GROUP / aoi_id / candidate_id / BASELINE_SUBDIR / BASELINE_PROVENANCE
    path = assert_pipeline_input_allowed(base, cfg.settings.paths)
    if not path.is_file():
        raise DebrisError(f"no Step 4C baseline at {base} (run scripts/run_baseline_detection.py first)")
    try:
        baseline = load_contract(BaselineProvenance, path)
    except ContractIOError as exc:
        raise DebrisError(str(exc)) from exc
    if baseline.aoi_id != aoi_id or baseline.candidate_id != candidate_id:
        raise DebrisError("the Step 4C baseline is for a different AOI or candidate")
    if baseline.source.stack_sha256 != prov.output.stack_sha256:
        raise DebrisError("the Step 4C baseline was made from a different stack (SHA-256 differs)")
    state_path = assert_pipeline_input_allowed(mi.root / baseline.outputs.state, cfg.settings.paths)
    if not state_path.is_file() or sha256_file(state_path) != baseline.outputs.state_sha256:
        raise DebrisError(f"{baseline.outputs.state} is missing or changed since Step 4C")
    dem_path = dem_sha = None
    if baseline.dem is not None:
        dem_path = assert_pipeline_input_allowed(mi.root / baseline.dem, cfg.settings.paths)
        if not dem_path.is_file() or sha256_file(dem_path) != baseline.dem_sha256:
            raise DebrisError(f"{baseline.dem} is missing or changed since Step 4C")
        dem_sha = baseline.dem_sha256
    return DebrisJob(model_input=mi, baseline_provenance_path=path, baseline_provenance_sha256=sha256_file(path),
                     baseline=baseline, step4c_state_path=state_path, dem_path=dem_path, dem_sha256=dem_sha,
                     orbit_direction=baseline.orbit_direction, final_dir=output_dir(cfg, aoi_id, candidate_id))


# ---------------------------------------------------------------------------
# Two-pass detection over blocks
# ---------------------------------------------------------------------------

Block = tuple[int, np.ndarray, np.ndarray, np.ndarray]          # row0, raw (6,h,w), valid, step4c_state
Terrain = Callable[[int, int], tuple[np.ndarray, np.ndarray] | None]   # (row0, rows) -> slope, s_r (or None)


@dataclass
class DebrisMaps:
    state: np.ndarray
    noise: tuple[NoiseModel, NoiseModel]
    statistics: DebrisStatistics
    table: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


def _quantiles(v: np.ndarray) -> dict[str, float]:
    if v.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)}


def detect(blocks: Callable[[], Iterator[Block]], shape: tuple[int, int], z: float, min_pixels: int,
           terrain: Terrain | None = None, progress: Callable[[str], None] | None = None) -> DebrisMaps:
    """Pass 1: noise model from d0 over observable pixels. Pass 2: rule. Then cluster filter and statistics."""
    say = progress or (lambda _t: None)
    h, w = shape
    buf = {"vv": np.empty(h * w, dtype=np.float32), "vh": np.empty(h * w, dtype=np.float32)}
    n = 0
    for _row0, raw, valid, s4 in blocks():
        check_consistency(raw, valid, where="debris pass 1")
        d = differences(raw)
        obs = observable(s4, valid)
        k = int(obs.sum())
        buf["vv"][n:n + k] = d["d0_vv"][obs]
        buf["vh"][n:n + k] = d["d0_vh"][obs]
        n += k
    say(f"noise model from {n:,} geometry-observable pixels\n")
    noise = (noise_model(buf["vv"][:n], "VV"), noise_model(buf["vh"][:n], "VH"))
    del buf
    state = np.zeros(shape, dtype=np.uint8)
    candidate = np.zeros(shape, dtype=bool)
    darkening = 0
    steep = 0
    rows_d: dict[str, list[np.ndarray]] = {c: [] for c in TABLE_COLUMNS}
    for row0, raw, valid, s4 in blocks():
        res = classify_debris_block(raw, valid, s4, noise[0].sigma_db, noise[1].sigma_db, z)
        rows = raw.shape[1]
        state[row0:row0 + rows] = res.state
        candidate[row0:row0 + rows] = res.candidate
        darkening += int(res.darkening.sum())
        steep += int(((s4 == int(BaselineState.NOT_ASSESSED_STEEP)) & (valid == 1)).sum())
        sel = res.candidate | (s4 == int(BaselineState.FLOOD))
        if sel.any():
            rr, cc = np.nonzero(sel)
            rows_d["row"].append(rr + row0)
            rows_d["col"].append(cc)
            rows_d["kind"].append(np.where(res.candidate[sel], KIND_DISTURBANCE, KIND_FLOOD))
            for key in ("d1_vv", "d2_vv", "d0_vv", "d1_vh", "d2_vh", "d0_vh"):
                rows_d[key].append(res.d[key][sel])
            t = terrain(row0, rows) if terrain is not None else None
            rows_d["slope_deg"].append(t[0][sel] if t is not None else np.full(int(sel.sum()), np.nan))
            rows_d["s_r_deg"].append(t[1][sel] if t is not None else np.full(int(sel.sum()), np.nan))
    kept, comps = keep_clusters(candidate, min_pixels)
    state[kept] = int(DebrisState.DISTURBANCE)
    dtypes = {"row": np.int32, "col": np.int32, "kind": np.uint8}
    table = {c: (np.concatenate(v).astype(dtypes.get(c, np.float32)) if v else
                 np.zeros(0, dtype=dtypes.get(c, np.float32))) for c, v in rows_d.items()}
    if table["row"].size:
        keep_row = (table["kind"] == KIND_FLOOD) | kept[table["row"], table["col"]]
        table = {c: v[keep_row] for c, v in table.items()}
    stats = _statistics(state, table, comps, darkening, steep)
    warnings = []
    if stats.facing_sensor_share is not None and stats.facing_sensor_share > 0.5:
        warnings.append(f"{stats.facing_sensor_share:.0%} of disturbance pixels face the sensor (criterion (c) "
                        "<= 50 % fails): possible geometry artefact")
    return DebrisMaps(state=state, noise=noise, statistics=stats, table=table, warnings=warnings)


def _statistics(state: np.ndarray, table: dict[str, np.ndarray], comps: DebrisComponents, darkening: int,
                steep: int) -> DebrisStatistics:
    counts = {s.name: int(np.count_nonzero(state == int(s))) for s in DebrisState}
    assessed = sum(counts[s.name] for s in DEBRIS_ASSESSED)
    dist = table["kind"] == KIND_DISTURBANCE
    slope, s_r = table["slope_deg"][dist], table["s_r_deg"][dist]
    by_slope = {f"{lo}-{hi} deg": int(((slope >= lo) & (slope < hi)).sum()) for lo, hi in SLOPE_BINS}
    by_slope["no slope"] = int((~np.isfinite(slope)).sum())
    by_geom = {"facing_sensor (s_r >= 10 deg)": int((s_r >= GEOMETRY_FLAT_DEG).sum()),
               "flat_or_gentle (|s_r| < 10 deg)": int((np.abs(s_r) < GEOMETRY_FLAT_DEG).sum()),
               "facing_away (s_r <= -10 deg)": int((s_r <= -GEOMETRY_FLAT_DEG).sum()),
               "no slope": int((~np.isfinite(s_r)).sum())}
    n = int(dist.sum())
    h, w = state.shape
    band = np.zeros(state.shape, dtype=bool)
    e = min(EDGE_BAND_PX, h // 2, w // 2)
    band[:e], band[-e:], band[:, :e], band[:, -e:] = True, True, True, True
    assessed_mask = np.isin(state, np.array([int(s) for s in DEBRIS_ASSESSED], dtype=np.uint8))
    edge_dist = int(band[table["row"][dist], table["col"][dist]].sum()) if n else 0
    finite_geom = n - by_geom["no slope"]
    return DebrisStatistics(
        total_pixels=int(state.size), state_counts=counts, assessed_pixels=assessed, disturbance_pixels=n,
        disturbance_fraction_of_assessed=round(n / assessed, 8) if assessed else None, steep_assessed_pixels=steep,
        darkening_diagnostic_pixels=darkening, disturbance_by_slope_deg=by_slope, disturbance_by_geometry=by_geom,
        facing_sensor_share=round(by_geom["facing_sensor (s_r >= 10 deg)"] / finite_geom, 6) if finite_geom else None,
        edge_pixels_share=round(edge_dist / n, 6) if n else None,
        assessed_edge_share=round(float((band & assessed_mask).sum()) / assessed, 6) if assessed else None,
        change_db_quantiles={k: _quantiles(table[k][dist]) for k in ("d1_vv", "d2_vv", "d0_vv", "d1_vh", "d2_vh",
                                                                      "d0_vh")},
        components=comps)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def threshold_records(cfg: BatoConfig) -> tuple[ThresholdRecord, ...]:
    t = cfg.thresholds
    named = [("detection.debris_z_min", t.detection.debris_z_min),
             ("detection.min_component_pixels", t.detection.min_component_pixels),
             ("visibility.layover_lia_max_deg (via Step 4C state)", t.visibility.layover_lia_max_deg),
             ("baseline_flood.shadow_lia_min_deg (via Step 4C state)", t.baseline_flood.shadow_lia_min_deg),
             ("baseline_flood.shadow_search_m (via Step 4C state)", t.baseline_flood.shadow_search_m)]
    return tuple(ThresholdRecord(name=n, value=p.value, unit=p.unit, provisional=p.provisional, rationale=p.rationale,
                                 basis=p.basis or "inherited_bato_design") for n, p in named)


def compute_debris(job: DebrisJob, cfg: BatoConfig, progress: Callable[[str], None] | None = None) -> DebrisMaps:
    from rasterio.windows import Window

    require_rasterio()
    mi = job.model_input
    grid = mi.provenance.grid
    assert grid is not None
    h, w = grid.height, grid.width
    look = LOOK_AZIMUTH_DEG[job.orbit_direction]
    features: DemFeatures | None = None
    if job.dem_path is not None:
        with _open(job.dem_path) as ds:
            dem = np.asarray(ds.read(1))
            features = dem_features(dem, tuple(ds.transform)[:6], ds.nodata)
        del dem

    def blocks() -> Iterator[Block]:
        with _open(mi.stack_path) as sd, _open(mi.mask_path) as md, _open(job.step4c_state_path) as bd:
            if (bd.height, bd.width) != (h, w):
                raise DebrisError("the Step 4C state raster does not match the stack grid")
            for row in range(0, h, BLOCK_ROWS):
                rows = min(BLOCK_ROWS, h - row)
                win = Window(0, row, w, rows)
                raw = np.stack([np.asarray(sd.read(k + 1, window=win), dtype=np.float32) for k in range(6)])
                yield (row, raw, np.asarray(md.read(1, window=win), dtype=np.uint8),
                       np.asarray(bd.read(1, window=win), dtype=np.uint8))

    def terrain(row0: int, rows: int) -> tuple[np.ndarray, np.ndarray] | None:
        if features is None:
            return None
        f = sample_features(features, grid.transform, row0, rows, w, look, incidence)
        s_r = range_slope_deg(f["slope_deg"], f["aspect_deg"], look)
        return f["slope_deg"], s_r

    p = cfg.thresholds.detection
    incidence = cfg.thresholds.baseline_flood.nominal_incidence_deg.value
    return detect(blocks, (h, w), p.debris_z_min.value, int(p.min_component_pixels.value), terrain, progress)


def run_debris(job: DebrisJob, cfg: BatoConfig, *, overwrite: bool = False,
               now: Callable[[], datetime] = _utc_now, progress: Callable[[str], None] | None = None
               ) -> tuple[Path, DebrisProvenance]:
    """Compute, write raster + table + provenance + diagnostics, publish atomically."""
    if job.final_dir.exists() and not overwrite:
        raise DebrisError(f"output already exists: {job.final_dir} (use --overwrite; the old output is kept as "
                          "<dir>.superseded-<time>)")
    started = now()
    maps = compute_debris(job, cfg, progress)
    mi = job.model_input
    prov = mi.provenance
    assert prov.grid is not None and prov.output is not None
    for path, expected in ((mi.stack_path, prov.output.stack_sha256), (mi.mask_path, prov.output.mask_sha256),
                           (job.step4c_state_path, job.baseline.outputs.state_sha256)):
        if sha256_file(path) != expected:
            raise DebrisError(f"{path.name} changed during the disturbance run")
    tmp = assert_interim_output_dir(_sibling(job.final_dir, "tmp", started), cfg.settings.paths)
    tmp.mkdir(parents=True)
    tags = {"method": "change_statistics", "validated": "false", "config_fingerprint": cfg.fingerprint(),
            "sigma_vv_db": str(maps.noise[0].sigma_db), "sigma_vh_db": str(maps.noise[1].sigma_db),
            **{f"state_{int(s)}": s.name for s in DebrisState}}
    _write(tmp / STATE_FILE, maps.state, prov.grid.transform, nodata=None, description="debris_state", tags=tags)
    savez: Any = np.savez_compressed                 # (numpy stubs mistype **arrays)
    savez(tmp / TABLE_FILE, **maps.table)
    root = mi.root
    b = job.baseline
    provenance = DebrisProvenance(
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(), started_at=started, finished_at=now(),
        aoi_id=mi.aoi_id, candidate_id=mi.candidate_id, event_date=prov.event_date, source=source_stack_record(mi),
        baseline=BaselineRef(provenance=_relative(job.baseline_provenance_path, root),
                             provenance_sha256=job.baseline_provenance_sha256, state=b.outputs.state,
                             state_sha256=b.outputs.state_sha256, config_fingerprint=b.config_fingerprint),
        dem=_relative(job.dem_path, root) if job.dem_path else None, dem_sha256=job.dem_sha256,
        orbit_direction=job.orbit_direction, look_azimuth_deg=LOOK_AZIMUTH_DEG[job.orbit_direction],
        z=cfg.thresholds.detection.debris_z_min.value, noise=maps.noise, thresholds=threshold_records(cfg), rule=RULE,
        outputs=DebrisOutputs(directory=_relative(job.final_dir, root),
                              state=_relative(job.final_dir / STATE_FILE, root),
                              state_sha256=sha256_file(tmp / STATE_FILE),
                              change_table=_relative(job.final_dir / TABLE_FILE, root),
                              change_table_sha256=sha256_file(tmp / TABLE_FILE),
                              diagnostics=_relative(job.final_dir / DIAGNOSTICS_FILE, root)),
        statistics=maps.statistics, warnings=tuple(maps.warnings))
    save_contract(provenance, tmp / PROVENANCE_FILE)
    (tmp / DIAGNOSTICS_FILE).write_text("\n".join(describe_debris(provenance)) + "\n", encoding="utf-8")
    if job.final_dir.exists():
        os.replace(job.final_dir, _sibling(job.final_dir, "superseded", now()))
    os.replace(tmp, job.final_dir)
    return job.final_dir, provenance


def describe_debris(prov: DebrisProvenance) -> list[str]:
    s = prov.statistics
    c = s.components
    vv, vh = prov.noise
    lines = [f"# SAR disturbance (debris) detection - {prov.aoi_id} / {prov.candidate_id} (NOT VALIDATED)", "",
             f"Rule: {prov.rule}", "",
             f"z = {prov.z:g} | sigma VV {vv.sigma_db:.4f} dB (median d0 {vv.median_d0_db:+.4f}, {vv.pixels:,} px) | "
             f"sigma VH {vh.sigma_db:.4f} dB (median d0 {vh.median_d0_db:+.4f}, {vh.pixels:,} px)",
             f"Thresholds: VV {prov.z * vv.sigma_db:.3f} dB, VH {prov.z * vh.sigma_db:.3f} dB", "",
             "## States", "", "| state | pixels |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in s.state_counts.items()]
    frac = s.disturbance_fraction_of_assessed
    lines += ["", f"Assessed {s.assessed_pixels:,} (of which Step 4C STEEP {s.steep_assessed_pixels:,}); "
              f"DISTURBANCE {s.disturbance_pixels:,} = {frac * 100 if frac is not None else float('nan'):.4f} % of "
              f"assessed", f"Darkening (diagnostic only, never a claim): {s.darkening_diagnostic_pixels:,}",
              f"Clusters: {c.candidate_pixels:,} candidate px in {c.candidate_components:,} components; kept "
              f"{c.kept_components:,} (>= {c.min_component_pixels} px), removed {c.removed_pixels:,} px; largest "
              f"{c.largest_component_px:,}", f"By slope: {s.disturbance_by_slope_deg}",
              f"By geometry: {s.disturbance_by_geometry}", "",
              "## Pre-registered checks (reporting only; nothing is tuned)", "",
              f"- (a) background ceiling (control run): disturbance <= 0.05 % of assessed -> this run "
              f"{frac * 100 if frac is not None else float('nan'):.4f} %",
              "- (b) real / control disturbance area >= 3: compare the two runs' DISTURBANCE fractions",
              f"- (c) facing-sensor share <= 50 %: {s.facing_sensor_share}",
              f"- (d) no edge concentration: disturbance share in the {EDGE_BAND_PX} px border {s.edge_pixels_share} "
              f"vs assessed share {s.assessed_edge_share}", "",
              f"Change (dB) quantiles of DISTURBANCE pixels: {s.change_db_quantiles}", "",
              "## Warnings", ""] + ([f"- {w}" for w in prov.warnings] or ["- none"])
    return lines

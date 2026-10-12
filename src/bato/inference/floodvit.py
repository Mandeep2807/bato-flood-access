"""Step 4B: FloodViT flood-segmentation inference on a verified Step 3.5 stack (torch-free core).

Everything except the network itself lives here and is tested without torch:

* the stack is verified exactly as in Step 4A (:func:`bato.preprocessing.model_input.prepare_model_input`:
  provenance, SHA-256 of stack and mask, raster header);
* tiles are the Step 4A 224 x 224 tiles with the ``shift_to_fit`` edge policy: when an axis is >= 224 the
  last tile is pulled back inside the stack, so NO tile is padded (FloodViT attention is global, so padding
  would change the outputs of the real pixels in the tile);
* each tile goes through the Step 4A path: layout-v1 block + ``valid_mask.tif`` -> consistency check ->
  channel order POST, PRE2, PRE1 (Kuro Siwo flood, sec1, sec2) -> clamp [0, 0.15] -> NaN -> 0.15 -> z-score;
* logits of overlapping tiles are averaged per pixel, then argmax (Kuro Siwo inference: argmax, no
  threshold) and softmax class 2 (UNCALIBRATED probability);
* every pixel with mask 0 is written as 3 (ignore) in the prediction and -1 in the probability, and is
  never counted as any class; statistics are over valid pixels, class 1 and class 2 separately;
* control modes (same network, other inputs; none is a validation):
  - ``no_event``: POST slot <- PRE-2, both pre slots <- PRE-1. The model sees the PRE-1 -> PRE-2 change
    (here 48 days, monsoon onset -> peak) with two IDENTICAL pre images, a combination absent from Kuro Siwo
    (sec1 != sec2 in 100 % of samples). It is confounded by real seasonal water change: NOT a clean
    false-positive baseline (see docs/forensics/step4b_floodvit_forensics.md).
  - ``zero_change``: POST slot <- PRE-2, sec1 <- PRE-2, sec2 <- PRE-1. Post equals the later pre image
    exactly: any "flood" is not driven by post-vs-sec1 change.
  - ``static_post``: all three slots <- POST. No temporal information at all: any "flood" or "permanent
    water" is driven by single-date appearance (terrain, shadow, dark surfaces), not by change.

The network is any ``LogitsFn``: (B, 6, 224, 224) float32 -> (B, 3, 224, 224) logits. The real FloodViT
is built in :mod:`bato.inference.floodvit_model` (needs torch and the Kuro Siwo repository).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import bato
from bato.config import BatoConfig
from bato.contracts.inference import (
    PREDICTION_NODATA,
    PROBABILITY_NODATA,
    FloodInferenceProvenance,
    InferenceOutputs,
    InferenceStatistics,
    ModelRecord,
    TilingRecord,
)
from bato.contracts.io import save_contract
from bato.contracts.model_input import IGNORE_LABEL, KURO_TILE_SIZE, ModelInputSpec, spec_fingerprint
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.preprocessing.model_input import (
    STACK_BANDS,
    ModelInputError,
    ModelInputJob,
    TileWindow,
    check_consistency,
    check_rasters,
    kuro_siwo_spec,
    normalize,
    plan_tiles,
    prepare_model_input,
    read_block,
    source_stack_record,
    tile_offsets,
)
from bato.preprocessing.radiometric import _relative, _sibling, _utc_now
from bato.preprocessing.sigma0_validation import _open, require_rasterio
from bato.preprocessing.speckle import sha256_file

LogitsFn = Callable[[np.ndarray], np.ndarray]
NUM_CLASSES = 3
INFERENCE_GROUP = "inference"   # <interim>/inference/<aoi_id>/<candidate_id>/<floodvit|floodvit_<control>>/
OUTPUT_SUBDIR = {"event": "floodvit", "no_event": "floodvit_no_event", "zero_change": "floodvit_zero_change",
                 "static_post": "floodvit_static_post"}
CONTROL_MODES = ("no_event", "zero_change", "static_post")
PREDICTION_FILE = "flood_prediction.tif"
PROBABILITY_FILE = "flood_probability.tif"
PROVENANCE_FILE = "floodvit_inference_provenance.json"
SEAM_BAND_PX = 16
BLOCK_ROWS = 512
# stack bands feeding model channels 0..5 (POST_VV, POST_VH, PRE2_VV, PRE2_VH, PRE1_VV, PRE1_VH) per control mode
NO_EVENT_SOURCES = ("PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH", "PRE1_VV", "PRE1_VH")
CONTROL_SOURCES = {
    "no_event": NO_EVENT_SOURCES,
    "zero_change": ("PRE2_VV", "PRE2_VH", "PRE2_VV", "PRE2_VH", "PRE1_VV", "PRE1_VH"),
    "static_post": ("POST_VV", "POST_VH", "POST_VV", "POST_VH", "POST_VV", "POST_VH"),
}


class InferenceError(ModelInputError):
    """Inference refused or failed (nothing published)."""


def model_bands(spec: ModelInputSpec, mode: str) -> list[int]:
    """0-based stack bands feeding model channels 0..5 for ``mode``."""
    if mode == "event":
        return [c.stack_band - 1 for c in spec.channels]
    if mode in CONTROL_SOURCES:
        return [STACK_BANDS.index(name) for name in CONTROL_SOURCES[mode]]
    raise InferenceError(f"unknown inference mode {mode!r} (event | {' | '.join(CONTROL_MODES)})")


def mapping_lines(spec: ModelInputSpec, mode: str) -> tuple[str, ...]:
    bands = model_bands(spec, mode)
    return tuple(f"{c.index} {c.name} <- stack {STACK_BANDS[b]}" for c, b in zip(spec.channels, bands, strict=True))


def model_input(raw: np.ndarray, valid: np.ndarray, spec: ModelInputSpec, mode: str = "event") -> np.ndarray:
    """(6, 224, 224) layout-v1 block + mask -> normalised (6, 224, 224) model tensor (Step 4A path)."""
    check_consistency(raw, valid)
    return normalize(raw[model_bands(spec, mode)], spec)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tiled prediction with overlap averaging
# ---------------------------------------------------------------------------


@dataclass
class Accumulator:
    """Per-pixel sum of logits and number of covering tiles."""

    sums: np.ndarray                            # (3, H, W) float32
    counts: np.ndarray                          # (H, W) uint16

    @classmethod
    def empty(cls, height: int, width: int) -> Accumulator:
        return cls(np.zeros((NUM_CLASSES, height, width), dtype=np.float32), np.zeros((height, width), dtype=np.uint16))

    def add(self, window: TileWindow, logits: np.ndarray) -> None:
        r, c, h, w = window.row_off, window.col_off, window.data_height, window.data_width
        self.sums[:, r:r + h, c:c + w] += logits[:, :h, :w]
        self.counts[r:r + h, c:c + w] += 1


def predict_tiles(stack_ds: Any, mask_ds: Any, height: int, width: int, spec: ModelInputSpec, logits_fn: LogitsFn,
                  *, mode: str = "event", batch_size: int = 4,
                  progress: Callable[[str], None] | None = None) -> tuple[Accumulator, list[TileWindow]]:
    """Run ``logits_fn`` over every tile (Step 4A path per tile) and accumulate logits."""
    if batch_size < 1:
        raise InferenceError("batch_size must be >= 1")
    t = spec.tiling.tile_size
    tiles = plan_tiles(height, width, spec.tiling)
    acc = Accumulator.empty(height, width)
    say = progress or (lambda _text: None)
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        x = np.stack([model_input(*read_block(stack_ds, mask_ds, w, size=(t, t)), spec, mode) for w in batch])
        out = np.asarray(logits_fn(x), dtype=np.float32)
        if out.shape != (len(batch), NUM_CLASSES, t, t):
            raise InferenceError(f"model output shape {out.shape}, expected {(len(batch), NUM_CLASSES, t, t)}")
        if not np.isfinite(out).all():
            raise InferenceError(f"model output has NaN/inf values in tiles {start}..{start + len(batch) - 1}")
        for window, logits in zip(batch, out, strict=True):
            acc.add(window, logits)
        done = start + len(batch)
        if start == 0 or (start // batch_size) % 25 == 0 or done == len(tiles):
            say(f"[{done}/{len(tiles)}] tiles\n")
    if (acc.counts == 0).any():
        raise InferenceError(f"{int((acc.counts == 0).sum())} pixels not covered by any tile")
    return acc, tiles


def edge_band_mask(length: int, offsets: Sequence[int], tile: int, band: int) -> np.ndarray:
    """1-D mask of positions within ``band`` px of an INTERNAL tile edge (stack edges excluded)."""
    edges = {o for o in offsets if o > 0} | {o + tile for o in offsets if o + tile < length}
    near = np.zeros(length, dtype=bool)
    for e in edges:
        near[max(0, e - band):min(length, e + band)] = True
    return near


@dataclass(frozen=True)
class Finalized:
    prediction: np.ndarray                      # (H, W) uint8: 0/1/2, 3 = invalid
    probability: np.ndarray                     # (H, W) float32: P(class 2), -1 = invalid
    statistics: InferenceStatistics


def finalize(acc: Accumulator, valid: np.ndarray, row_offsets: Sequence[int], col_offsets: Sequence[int],
             tile: int = KURO_TILE_SIZE, seam_band: int = SEAM_BAND_PX) -> Finalized:
    """Average logits, argmax, softmax(class 2); invalid -> 3 / -1; statistics over valid pixels only."""
    h, w = valid.shape
    if acc.counts.shape != (h, w):
        raise InferenceError("validity mask and accumulator differ in shape")
    prediction = np.empty((h, w), dtype=np.uint8)
    probability = np.empty((h, w), dtype=np.float32)
    raw_argmax = np.empty((h, w), dtype=np.uint8)
    for row in range(0, h, BLOCK_ROWS):
        rows = slice(row, min(h, row + BLOCK_ROWS))
        mean = acc.sums[:, rows] / acc.counts[rows].astype(np.float32)[None]
        raw_argmax[rows] = mean.argmax(axis=0).astype(np.uint8)
        probability[rows] = softmax(mean)[2]
    ok = valid == 1
    prediction[:] = np.where(ok, raw_argmax, np.uint8(IGNORE_LABEL))
    probability[~ok] = PROBABILITY_NODATA
    n_valid = int(ok.sum())
    counts = [int(((raw_argmax == k) & ok).sum()) for k in range(NUM_CLASSES)]
    invalid_raw = {str(k): int(((raw_argmax == k) & ~ok).sum()) for k in range(NUM_CLASSES)}
    near = edge_band_mask(h, row_offsets, tile, seam_band)[:, None] | edge_band_mask(w, col_offsets, tile,
                                                                                      seam_band)[None, :]
    flood = (prediction == 2)

    def frac(sel: np.ndarray) -> float | None:
        n = int((sel & ok).sum())
        return round(int((flood & sel & ok).sum()) / n, 8) if n else None

    stats = InferenceStatistics(
        total_pixels=h * w, valid_pixels=n_valid, invalid_pixels=h * w - n_valid, no_water_pixels=counts[0],
        permanent_water_pixels=counts[1], flood_pixels=counts[2],
        permanent_water_fraction=round(counts[1] / n_valid, 8) if n_valid else 0.0,
        flood_fraction=round(counts[2] / n_valid, 8) if n_valid else 0.0,
        water_fraction=round((counts[1] + counts[2]) / n_valid, 8) if n_valid else 0.0,
        invalid_raw_argmax=invalid_raw, seam_band_px=seam_band, flood_fraction_near_tile_edges=frac(near),
        flood_fraction_tile_interiors=frac(~near),
        flood_probability_mean_valid=round(float(probability[ok].mean()), 8) if n_valid else None)
    return Finalized(prediction=prediction, probability=probability, statistics=stats)


# ---------------------------------------------------------------------------
# Job, run, publish
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceJob:
    model_input: ModelInputJob
    spec: ModelInputSpec
    mode: str
    checkpoint: Path
    final_dir: Path


def prepare_inference(cfg: BatoConfig, aoi_id: str, candidate_id: str, checkpoint: Path, *, mode: str = "event",
                      stride: int = KURO_TILE_SIZE) -> InferenceJob:
    """Verify the stack (Step 4A) and the checkpoint path; nothing is written or loaded.

    Raises:
        InferenceError / ModelInputError: missing or inconsistent inputs.
        ProhibitedInputError: an input lies in an evaluation-only folder.
    """
    spec = kuro_siwo_spec(stride, "shift_to_fit")
    model_bands(spec, mode)
    ckpt = assert_pipeline_input_allowed(checkpoint, cfg.settings.paths)
    if not ckpt.is_file():
        raise InferenceError(f"checkpoint not found: {checkpoint}")
    job = prepare_model_input(cfg, aoi_id, candidate_id)
    for part in (aoi_id, candidate_id):
        if not part or any(ch in part for ch in "/\\:") or part in (".", ".."):
            raise InferenceError(f"invalid path component {part!r}")
    final_dir = cfg.settings.paths.interim / INFERENCE_GROUP / aoi_id / candidate_id / OUTPUT_SUBDIR[mode]
    return InferenceJob(model_input=job, spec=spec, mode=mode, checkpoint=ckpt,
                        final_dir=assert_interim_output_dir(final_dir, cfg.settings.paths))


@dataclass(frozen=True)
class InferenceResult:
    output_dir: Path
    provenance_path: Path
    provenance: FloodInferenceProvenance


def _write(path: Path, array: np.ndarray, grid_transform: Sequence[float], *, dtype: str, nodata: float,
           description: str, tags: dict[str, str]) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.windows import Window

    h, w = array.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype=dtype, crs=CRS.from_epsg(3857),
                       transform=Affine(*grid_transform), nodata=nodata, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", BIGTIFF="IF_SAFER") as dst:
        for row in range(0, h, BLOCK_ROWS):
            rows = min(BLOCK_ROWS, h - row)
            dst.write(array[row:row + rows], 1, window=Window(0, row, w, rows))
        dst.set_band_description(1, description)
        dst.update_tags(**tags)


def run_inference(job: InferenceJob, cfg: BatoConfig, logits_fn: LogitsFn, model: ModelRecord, *,
                  batch_size: int = 4, overwrite: bool = False, now: Callable[[], datetime] = _utc_now,
                  progress: Callable[[str], None] | None = None) -> InferenceResult:
    """Predict every tile, merge, mask, write both rasters + provenance, publish atomically.

    Raises:
        InferenceError / ModelInputError: refused or failed (nothing published).
    """
    require_rasterio()
    mi_job = job.model_input
    prov = mi_job.provenance
    assert prov.grid is not None and prov.output is not None
    final_dir = job.final_dir
    if final_dir.exists() and not overwrite:
        raise InferenceError(f"output already exists: {final_dir} (use --overwrite; the old output is kept as "
                             "<dir>.superseded-<time>)")
    if model.checkpoint_sha256 != sha256_file(job.checkpoint):
        raise InferenceError("checkpoint SHA-256 does not match the model record")
    started = now()
    h, w = prov.grid.height, prov.grid.width
    t = job.spec.tiling.tile_size
    with _open(mi_job.stack_path) as stack_ds, _open(mi_job.mask_path) as mask_ds:
        check_rasters(stack_ds, mask_ds, prov)
        acc, tiles = predict_tiles(stack_ds, mask_ds, h, w, job.spec, logits_fn, mode=job.mode,
                                   batch_size=batch_size, progress=progress)
        valid = np.asarray(mask_ds.read(1), dtype=np.uint8)
    rows = tile_offsets(h, t, job.spec.tiling.stride, job.spec.tiling.edge_policy)
    cols = tile_offsets(w, t, job.spec.tiling.stride, job.spec.tiling.edge_policy)
    result = finalize(acc, valid, rows, cols, tile=t)
    for path, expected in ((mi_job.stack_path, prov.output.stack_sha256), (mi_job.mask_path, prov.output.mask_sha256)):
        if sha256_file(path) != expected:
            raise InferenceError(f"{path.name} changed during inference")
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", started), cfg.settings.paths)
    tmp_dir.mkdir(parents=True)
    fp = spec_fingerprint(job.spec)
    common = {"model": model.name, "checkpoint_sha256": model.checkpoint_sha256, "spec_fingerprint": fp,
              "mode": job.mode, "validated": "false"}
    _write(tmp_dir / PREDICTION_FILE, result.prediction, prov.grid.transform, dtype="uint8", nodata=PREDICTION_NODATA,
           description="floodvit_class", tags={**common, "class_0": "no water", "class_1": "permanent water",
                                               "class_2": "flood", "class_3": "invalid (ignore)"})
    _write(tmp_dir / PROBABILITY_FILE, result.probability, prov.grid.transform, dtype="float32",
           nodata=PROBABILITY_NODATA, description="flood_probability_uncalibrated",
           tags={**common, "meaning": "softmax(mean logits)[class 2], uncalibrated, no threshold; -1 = invalid"})
    s = result.statistics
    warnings = []
    if s.invalid_pixels:
        warnings.append(f"{s.invalid_pixels:,} invalid pixels written as 3 / -1; the model would have called "
                        f"{s.invalid_raw_argmax.get('1', 0) + s.invalid_raw_argmax.get('2', 0):,} of them water")
    if s.flood_fraction_near_tile_edges is not None and s.flood_fraction_tile_interiors:
        ratio = s.flood_fraction_near_tile_edges / s.flood_fraction_tile_interiors
        if ratio > 1.5 or ratio < 1 / 1.5:
            warnings.append(f"flood fraction near internal tile edges is {ratio:.2f}x the tile interiors (seams?)")
    counts = acc.counts
    finished = now()
    outputs = InferenceOutputs(
        directory=_relative(final_dir, mi_job.root), prediction=_relative(final_dir / PREDICTION_FILE, mi_job.root),
        prediction_sha256=sha256_file(tmp_dir / PREDICTION_FILE),
        probability=_relative(final_dir / PROBABILITY_FILE, mi_job.root),
        probability_sha256=sha256_file(tmp_dir / PROBABILITY_FILE))
    provenance = FloodInferenceProvenance(
        mode=job.mode,  # type: ignore[arg-type]
        model_input_mapping=mapping_lines(job.spec, job.mode), bato_version=bato.__version__,
        config_fingerprint=cfg.fingerprint(), started_at=started, finished_at=finished, aoi_id=mi_job.aoi_id,
        candidate_id=mi_job.candidate_id, event_date=prov.event_date, source=source_stack_record(mi_job),
        spec=job.spec, spec_fingerprint=fp, model=model,
        tiling=TilingRecord(tile_size=t, stride=job.spec.tiling.stride, edge_policy=job.spec.tiling.edge_policy,
                            row_offsets=tuple(rows), col_offsets=tuple(cols), tiles=len(tiles),
                            padded_tiles=sum(1 for x in tiles if (x.data_height, x.data_width) != (t, t)),
                            overlap_pixels=int((counts > 1).sum())),
        outputs=outputs, statistics=s, warnings=tuple(warnings))
    save_contract(provenance, tmp_dir / PROVENANCE_FILE)
    if final_dir.exists():
        os.replace(final_dir, _sibling(final_dir, "superseded", finished))
    os.replace(tmp_dir, final_dir)
    return InferenceResult(output_dir=final_dir, provenance_path=final_dir / PROVENANCE_FILE, provenance=provenance)


def describe_inference(prov: FloodInferenceProvenance) -> list[str]:
    s, t = prov.statistics, prov.tiling
    lines = [f"mode       : {prov.mode} | input " + ", ".join(prov.model_input_mapping),
             f"tiles      : {t.tiles} ({len(t.row_offsets)} x {len(t.col_offsets)}) x {t.tile_size} px, stride "
             f"{t.stride}, edge {t.edge_policy}, padded tiles {t.padded_tiles}, overlap pixels {t.overlap_pixels:,}",
             f"pixels     : {s.valid_pixels:,} valid / {s.total_pixels:,} total ({s.invalid_pixels:,} invalid -> 3)",
             f"class 1    : permanent water {s.permanent_water_pixels:,} ({s.permanent_water_fraction:.4%} of valid)",
             f"class 2    : flood {s.flood_pixels:,} ({s.flood_fraction:.4%} of valid)",
             f"seams      : flood fraction within {s.seam_band_px} px of internal tile edges "
             f"{s.flood_fraction_near_tile_edges} vs interiors {s.flood_fraction_tile_interiors}",
             f"invalid    : model raw argmax at invalid pixels (discarded) {s.invalid_raw_argmax}",
             "status     : NOT VALIDATED (model output; probability uncalibrated)"]
    lines += [f"WARN  {x}" for x in prov.warnings]
    return lines

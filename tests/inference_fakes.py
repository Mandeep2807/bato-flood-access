"""TEST FIXTURES for Step 4B inference / diagnostics: a synthetic scene written over a Step 3.5 stack made by
the real Step 3.2-3.5 code with fake gpts, a pixel-local fake change model and a dummy checkpoint.
Never real imagery, never a real model."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from bato.contracts.inference import ModelRecord
from bato.contracts.io import save_contract
from bato.inference import floodvit as fv
from bato.preprocessing import model_input as mi
from bato.preprocessing import stack as stk

from stack_fakes import published_terrain_set

T = 224
H, W = 300, 470
FLOOD_BOX = (slice(100, 160), slice(300, 380))     # POST_VV drops (event)
WATER_BOX = (slice(200, 260), slice(40, 120))      # dark in every layer (permanent water)
BRIGHT_BOX = (slice(20, 60), slice(20, 60))        # PRE1_VV far above the 0.15 clamp
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def change_model(x: np.ndarray) -> np.ndarray:
    """Pixel-local fake: flood where POST_VV dropped below the sec1 VV, permanent water where both are dark."""
    post, pre2 = x[:, 0], x[:, 2]
    logits = np.zeros((x.shape[0], 3, T, T), dtype=np.float32)
    logits[:, 1] = 3 * (-1.0 - np.maximum(post, pre2))
    logits[:, 2] = 3 * (pre2 - post - 1.0)
    return logits


def scene() -> tuple[np.ndarray, np.ndarray]:
    data = np.empty((6, H, W), dtype=np.float32)
    data[0::2], data[1::2] = 0.08, 0.02
    data[4][FLOOD_BOX] = 0.01
    data[0::2, WATER_BOX[0], WATER_BOX[1]] = 0.005
    data[0][BRIGHT_BOX] = 0.6
    data[4, :10, :] = 0.0
    data[:, H - 1, W - 1] = 0.0
    return data, np.all(data != 0.0, axis=0).astype(np.uint8)


def write_synthetic_stack(project, result, data: np.ndarray, mask: np.ndarray) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    root = project.cfg.config_dir.parent
    prov = result.provenance
    _c, h, w = data.shape
    stack_path, mask_path = root / prov.output.stack, root / prov.output.mask
    with rasterio.open(stack_path, "w", driver="GTiff", width=w, height=h, count=6, dtype="float32",
                       crs=CRS.from_epsg(3857), transform=Affine(*prov.grid.transform), nodata=0.0) as dst:
        for k, name in enumerate(mi.STACK_BANDS):
            dst.write(data[k], k + 1)
            dst.set_band_description(k + 1, name)
    with rasterio.open(mask_path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                       crs=CRS.from_epsg(3857), transform=Affine(*prov.grid.transform)) as dst:
        dst.write(mask, 1)
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (stack_path, mask_path)}
    save_contract(prov.model_copy(update={
        "grid": prov.grid.model_copy(update={"width": w, "height": h}),
        "output": prov.output.model_copy(update={"stack_sha256": sha[stack_path], "mask_sha256": sha[mask_path]})}),
        result.provenance_path)


def synthetic_project(root, gpt):
    project = published_terrain_set(root, gpt)
    result = stk.run_stack(stk.prepare_stack(project.manifest, project.cfg), project.cfg)
    data, mask = scene()
    write_synthetic_stack(project, result, data, mask)
    return project, data, mask


def candidate_id(project) -> str:
    return stk.select_candidate(project.manifest, None).candidate_id


def checkpoint(project):
    path = project.cfg.config_dir.parent / "models" / "TEST_FIXTURE_floodvit.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"TEST FIXTURE - not a real checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, ModelRecord(checkpoint=path.as_posix(), checkpoint_sha256=digest)


def run_mode(project, model, mode: str):
    path, record = checkpoint(project)
    job = fv.prepare_inference(project.cfg, project.manifest.aoi_id, candidate_id(project), path, mode=mode)
    return fv.run_inference(job, project.cfg, model, record, batch_size=3, now=lambda: NOW)

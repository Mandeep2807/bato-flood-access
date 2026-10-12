"""OPT-IN: the real FloodViT network (needs torch + einops and a Kuro Siwo checkout).

    $env:BATO_KURO_REPO="C:/Users/Login/Downloads/kuro_check/KuroSiwo"
    $env:BATO_FLOODVIT_CHECKPOINT="C:/Users/Login/Downloads/kuro_check/kurosiwo_vit_decoder_weights.pt"   # optional
    python -m pytest tests/integration/test_floodvit_model.py -v

Checks the architecture (205,538,499 parameters, 280 keys), output shape/finiteness, determinism of a
repeated identical tile, and (with the checkpoint) the strict load and the same checks on real weights.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

KURO = os.environ.get("BATO_KURO_REPO")
CKPT = os.environ.get("BATO_FLOODVIT_CHECKPOINT")
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("einops") is None,
                       reason="torch/einops not installed"),
    pytest.mark.skipif(not KURO, reason="opt-in: set BATO_KURO_REPO to a Kuro Siwo checkout"),
]


def tile(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (1, 6, 224, 224)).astype(np.float32)   # TEST FIXTURE


def test_architecture_matches_the_checkpoint_layout():
    from bato.inference import floodvit_model as fm
    model = fm.build_floodvit(Path(KURO))
    keys = model.state_dict().keys()
    assert sum(p.numel() for p in model.parameters()) == fm.EXPECTED_PARAMETERS == 205_538_499
    assert len(keys) == fm.EXPECTED_KEYS == 280
    assert "model.pos_embedding" in keys and "head.deconv3.weight" in keys and not any("mlp_head" in k for k in keys)
    model.eval()
    out = fm.logits_fn(model)(tile())
    assert out.shape == (1, 3, 224, 224) and out.dtype == np.float32 and np.isfinite(out).all()


@pytest.mark.skipif(not CKPT, reason="opt-in: set BATO_FLOODVIT_CHECKPOINT")
def test_real_weights_load_strict_and_are_deterministic():
    from bato.inference import floodvit_model as fm
    model, record = fm.load_floodvit(Path(CKPT), Path(KURO), threads=4)
    assert record.parameter_count == 205_538_499 and record.state_dict_keys == 280
    run = fm.logits_fn(model)
    x = tile(1)
    a, b = run(x), run(x.copy())
    assert np.array_equal(a, b) and np.isfinite(a).all()                   # repeated identical tile -> identical
    batch = run(np.concatenate([x, tile(2)]))
    assert np.allclose(batch[:1], a, atol=1e-4)                             # batching does not change a tile's result

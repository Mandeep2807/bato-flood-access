"""Step 4B: the Kuro Siwo FloodViT network (needs torch, einops and a Kuro Siwo checkout).

The architecture is the one ``run_floodvit_inference.py`` reconstructed, unchanged: Kuro Siwo's own
``models/vision_transformer.ViT`` (imported from the given checkout, not copied) with image 224, patch 16,
6 channels, dim 1024, depth 24, heads 16, mlp_dim 2048, ``pool`` set to False (patch tokens only) and
``mlp_head`` replaced by Identity, followed by the ``Decoder`` of ``models/seg_finetuner.py`` /
``models/model_utilities.py`` (ConvT 1024->128, ReLU, Upsample x2, ConvT 128->64, ReLU, ConvT 64->3).
This equals ``hubconf.py@43ec8cf vit_decoder`` (FinetunerSegmentation, decoder=True, num_classes 3):
205,538,499 parameters, 280 state-dict keys, loaded with ``strict=True``.

torch is imported lazily (``importlib``) so the rest of BATO, its tests and type checks do not need it.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bato.contracts.inference import ModelRecord
from bato.inference.floodvit import InferenceError
from bato.preprocessing.speckle import sha256_file

EXPECTED_PARAMETERS = 205_538_499
EXPECTED_KEYS = 280
VIT_KWARGS = dict(image_size=224, patch_size=16, num_classes=1000, dim=1024, depth=24, heads=16, mlp_dim=2048,
                  pool="cls", channels=6)


def _torch() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        einops = importlib.import_module("einops")
    except ImportError as exc:
        raise InferenceError("FloodViT inference needs torch and einops (pip install torch einops)") from exc
    return torch, torch.nn, einops


def _kuro_vit(kuro_repo: Path) -> Any:
    source = Path(kuro_repo) / "models" / "vision_transformer.py"
    if not source.is_file():
        raise InferenceError(f"{source} not found: --kuro-repo must be a Kuro Siwo checkout")
    root = str(Path(kuro_repo).resolve())
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("models.vision_transformer")
    finally:
        sys.path.remove(root)
    if module.__file__ is None or Path(module.__file__).resolve() != source.resolve():
        raise InferenceError(f"imported {module.__file__}, expected {source} (another 'models' package on the path)")
    return module.ViT


def build_floodvit(kuro_repo: Path) -> Any:
    """The FloodViT module (random weights), exactly as reconstructed in run_floodvit_inference.py."""
    _torch_mod, nn, einops = _torch()
    vit_cls = _kuro_vit(kuro_repo)
    module_base: Any = nn.Module

    class Decoder(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.deconv1 = nn.ConvTranspose2d(1024, 128, 4, 2, 1)
            self.relu = nn.ReLU()
            self.up = nn.Upsample(scale_factor=2)
            self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        def forward(self, x: Any) -> Any:
            x = self.relu(self.deconv1(x))
            x = self.up(x)
            x = self.relu(self.deconv2(x))
            return self.deconv3(x)

    class FloodViT(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.model = vit_cls(**VIT_KWARGS)
            self.model.pool = False
            self.model.mlp_head = nn.Identity()
            self.head = Decoder()

        def forward(self, x: Any) -> Any:
            x = self.model(x)
            x = einops.rearrange(x, "b (h w) c -> b c h w", h=14, w=14)
            return self.head(x)

    return FloodViT()


def load_floodvit(checkpoint: Path, kuro_repo: Path, *, threads: int | None = None) -> tuple[Any, ModelRecord]:
    """Build, load ``strict=True`` (weights_only), eval mode; returns (module, ModelRecord)."""
    torch, _nn, _einops = _torch()
    if threads:
        torch.set_num_threads(threads)
    model = build_floodvit(kuro_repo)
    try:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:          # unpickling / missing or unexpected keys / shape mismatch
        raise InferenceError(f"cannot load {checkpoint} into FloodViT (strict): {exc}") from exc
    model.eval()
    params = int(sum(p.numel() for p in model.parameters()))
    if params != EXPECTED_PARAMETERS or len(state) != EXPECTED_KEYS:
        raise InferenceError(f"FloodViT has {params:,} parameters / {len(state)} keys; expected "
                             f"{EXPECTED_PARAMETERS:,} / {EXPECTED_KEYS}")
    source = Path(kuro_repo) / "models" / "vision_transformer.py"
    record = ModelRecord(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(Path(checkpoint)),
                         parameter_count=params, state_dict_keys=len(state), vit_source=source.as_posix(),
                         vit_source_sha256=sha256_file(source))
    return model, record


def logits_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """numpy (B, 6, 224, 224) float32 -> numpy (B, 3, 224, 224) float32 logits (no_grad, eval)."""
    torch, _nn, _einops = _torch()

    def run(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        return np.asarray(out.cpu().numpy(), dtype=np.float32)

    return run

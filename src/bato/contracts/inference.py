"""Step 4B contracts: provenance and statistics of one FloodViT inference run on a Step 3.5 stack.

The model is the Kuro Siwo FloodViT (ViT encoder + deconvolution decoder, 3 classes). The class codes
of the prediction raster follow Kuro Siwo (``KuroSiwoLabel``): 0 no water, 1 permanent water, 2 flood,
3 = invalid / ignore (any SAR layer no-data). The probability raster is the softmax of the averaged
logits for class 2: UNCALIBRATED, no threshold. Nothing here is validated: there is no accuracy
evidence for this checkpoint on BATO data, and ``validated`` is always False.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from bato.contracts.common import Contract, RelativePath, Sha256Hex, UTCDatetime
from bato.contracts.model_input import ModelInputSpec, SourceStackRecord, spec_fingerprint

INFERENCE_VERSION: Literal["bato-floodvit-inference-v1"] = "bato-floodvit-inference-v1"
CLASS_CODES = {"0": "no water", "1": "permanent water", "2": "flood",
               "3": "invalid / ignore (SAR no-data in any of the six layers, or padding)"}
PREDICTION_NODATA = 3
PROBABILITY_NODATA = -1.0


class ModelRecord(Contract):
    name: Literal["floodvit_vit_decoder"] = "floodvit_vit_decoder"
    architecture: str = ("Kuro Siwo ViT (image 224, patch 16, 6 channels, dim 1024, depth 24, heads 16, dim_head 64, "
                         "mlp_dim 2048, patch tokens only) + Decoder (ConvT 1024->128 /2, ReLU, Upsample x2, ConvT "
                         "128->64 /2, ReLU, ConvT 64->3 /2): hubconf.py@43ec8cf vit_decoder")
    checkpoint: str                              # as given (display only)
    checkpoint_sha256: Sha256Hex
    parameter_count: int | None = None
    state_dict_keys: int | None = None
    vit_source: str | None = None                # Kuro Siwo models/vision_transformer.py used (path)
    vit_source_sha256: Sha256Hex | None = None
    weights_origin: str = ("kurosiwo_vit_decoder_weights.pt from the hubconf.py@43ec8cf release (Kuro Siwo PR #4, "
                           "not merged into main); its training configuration is not in the repository")


class TilingRecord(Contract):
    tile_size: int
    stride: int
    edge_policy: str
    row_offsets: tuple[int, ...]
    col_offsets: tuple[int, ...]
    tiles: int
    padded_tiles: int                            # tiles that needed padding (only if an axis < 224)
    overlap_pixels: int                          # pixels covered by more than one tile
    merge: Literal["mean of logits over all covering tiles, then argmax / softmax"] = (
        "mean of logits over all covering tiles, then argmax / softmax")


class InferenceStatistics(Contract):
    """All fractions are over VALID pixels (mask 1); invalid pixels are never counted as any class."""

    total_pixels: int
    valid_pixels: int
    invalid_pixels: int
    no_water_pixels: int                         # class 0
    permanent_water_pixels: int                  # class 1
    flood_pixels: int                            # class 2
    permanent_water_fraction: float
    flood_fraction: float
    water_fraction: float                        # classes 1 + 2
    invalid_raw_argmax: dict[str, int]           # diagnostic: what the model output at invalid pixels (discarded)
    seam_band_px: int
    flood_fraction_near_tile_edges: float | None  # valid pixels within seam_band_px of an internal tile edge
    flood_fraction_tile_interiors: float | None
    flood_probability_mean_valid: float | None

    @model_validator(mode="after")
    def _check(self) -> InferenceStatistics:
        if self.valid_pixels + self.invalid_pixels != self.total_pixels:
            raise ValueError("valid + invalid must equal total pixels")
        if self.no_water_pixels + self.permanent_water_pixels + self.flood_pixels != self.valid_pixels:
            raise ValueError("class counts must sum to the valid pixels")
        if sum(self.invalid_raw_argmax.values()) != self.invalid_pixels:
            raise ValueError("invalid_raw_argmax must account for every invalid pixel")
        return self


class InferenceOutputs(Contract):
    directory: RelativePath
    prediction: RelativePath
    prediction_sha256: Sha256Hex
    probability: RelativePath
    probability_sha256: Sha256Hex
    prediction_nodata: int = PREDICTION_NODATA
    probability_nodata: float = PROBABILITY_NODATA


class FloodInferenceProvenance(Contract):
    kind: Literal["floodvit_inference"] = "floodvit_inference"
    version: Literal["bato-floodvit-inference-v1"] = INFERENCE_VERSION
    mode: Literal["event", "no_event", "zero_change", "static_post"]
    model_input_mapping: tuple[str, ...]         # e.g. "0 POST_VV <- stack POST_VV"
    bato_version: str
    config_fingerprint: str
    started_at: UTCDatetime
    finished_at: UTCDatetime
    aoi_id: str
    candidate_id: str
    event_date: str
    source: SourceStackRecord
    spec: ModelInputSpec
    spec_fingerprint: Sha256Hex
    model: ModelRecord
    tiling: TilingRecord
    outputs: InferenceOutputs
    class_codes: dict[str, str] = CLASS_CODES
    probability: str = ("softmax of the averaged logits, class 2 (flood); UNCALIBRATED, no threshold applied; "
                        "-1 at invalid pixels")
    statistics: InferenceStatistics
    validated: Literal[False] = False
    validation_note: str = ("NOT VALIDATED: no accuracy evidence exists for this checkpoint on BATO/Trishuli data; "
                            "the map is a model output, not a verified flood extent")
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> FloodInferenceProvenance:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.spec_fingerprint != spec_fingerprint(self.spec):
            raise ValueError("spec_fingerprint does not match the recorded specification")
        if len(self.model_input_mapping) != len(self.spec.channels):
            raise ValueError("model_input_mapping must list every model channel")
        if self.class_codes != CLASS_CODES:
            raise ValueError("class codes are fixed (Kuro Siwo labels + 3 = invalid)")
        return self

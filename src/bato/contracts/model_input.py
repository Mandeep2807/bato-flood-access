"""Step 4A contracts: the model-input specification and the model-input manifest.

The specification fixes how a BATO Step 3.5 temporal stack becomes model tensors so that they match
what the Kuro Siwo GRD segmentation pipeline feeds its networks. Every element carries the evidence
it rests on (:class:`EvidenceStatus`): read from the Kuro Siwo code, read from the Kuro Siwo data
shipped in that repository, documentation only, not determinable, a BATO design choice, a known
deviation, or a BATO gap. Nothing is inferred from papers or common practice.

Reference: https://github.com/Orion-AI-Lab/KuroSiwo at commit ``4347ed1`` (2026-07-16), default
configuration: ``configs/config.json`` (task segmentation, method unet) + ``configs/train/data_config.json``
(GRD, ``slc: false``), trainer ``training/segmentation_trainer.py``.

Tensor contract (v1): float32, layout (C, H, W) = (6, 224, 224), channels
    0 POST_VV  1 POST_VH  2 PRE2_VV  3 PRE2_VH  4 PRE1_VV  5 PRE1_VH
i.e. Kuro Siwo flood, sec1 (LATER pre-event), sec2 (EARLIER pre-event), VV before VH - also the output of the
Kuro Siwo torch.hub FloodViT wrapper (commit 43ec8cf), whose "pre1"/"pre2" are Kuro sec1/sec2, so BATO PRE2/PRE1
(Kuro "pre1" is NOT BATO PRE1); each value
``clamp(x, 0, 0.15)`` -> NaN to 0.15 -> ``(x - mean) / std``. No DEM channel. A uint8 validity mask
(1 = valid in all six stack layers) travels with every tile; outputs at invalid pixels must be set to
the ignore label (3), never read as water or non-water.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum, IntEnum
from typing import Literal

from pydantic import computed_field, model_validator

from bato.contracts.common import Contract, Polarization, RelativePath, Sha256Hex, UTCDatetime

MODEL_INPUT_VERSION: Literal["bato-model-input-v1"] = "bato-model-input-v1"
STACK_LAYOUT: Literal["bato-s1-stack-v1"] = "bato-s1-stack-v1"   # the only stack layout this version accepts
KURO_SIWO_REPOSITORY = "https://github.com/Orion-AI-Lab/KuroSiwo"
KURO_SIWO_COMMIT = "4347ed173c4e48f5a9d578bb5fe8453706b08e5e"
KURO_SIWO_CONFIGURATION = ("configs/config.json (task segmentation, method unet, num_classes 3) + "
                           "configs/train/data_config.json (GRD, slc false, inputs pre_event_1/pre_event_2/post_event, "
                           "channels vv/vh); trainer training/segmentation_trainer.py")

# Kuro Siwo constants (configs/train/data_config.json L14-L17; dataset/Dataset.py concat / scale_img)
KURO_CLAMP_MIN = 0.0
KURO_CLAMP_MAX = 0.15
KURO_NAN_FILL = 0.15
KURO_MEAN = {"VV": 0.0953, "VH": 0.0264}
KURO_STD = {"VV": 0.0427, "VH": 0.0215}
KURO_TILE_SIZE = 224
KURO_SAR_NODATA = 0.0
IGNORE_LABEL = 3


class KuroSiwoLabel(IntEnum):
    """Kuro Siwo ``MK0_MLU`` label values (BATO creates no labels in Step 4A).

    Verified from the Kuro Siwo code: ``training/segmentation_trainer.py`` L13 ``CLASS_LABELS = {0: "No water",
    1: "Permanent Waters", 2: "Floods", 3: "Invalid pixels"}``; ``configs/config.json`` num_classes 3;
    ``utilities/utilities.py`` create_loss ``ignore_index=3``; ``catalogue/catalogue.py`` MLU no-data 3.
    There is no debris or damage class.
    """

    NO_WATER = 0
    PERMANENT_WATER = 1
    FLOOD = 2
    IGNORE = 3


class EvidenceStatus(str, Enum):
    VERIFIED_FROM_CODE = "verified_from_kuro_siwo_code"
    VERIFIED_FROM_DATA = "verified_from_kuro_siwo_data"
    DOCUMENTATION_ONLY = "documentation_only"
    NOT_DETERMINABLE = "not_determinable"
    BATO_DESIGN_CHOICE = "bato_design_choice"       # Kuro Siwo has no equivalent; decided and recorded by BATO
    KNOWN_DEVIATION = "known_deviation"             # BATO differs from Kuro Siwo on purpose (decided earlier)
    BATO_GAP = "bato_gap"                           # information BATO does not have yet


RESOLVED_STATUSES = frozenset({EvidenceStatus.VERIFIED_FROM_CODE, EvidenceStatus.VERIFIED_FROM_DATA,
                               EvidenceStatus.BATO_DESIGN_CHOICE, EvidenceStatus.KNOWN_DEVIATION})


class SpecItem(Contract):
    key: str
    statement: str
    status: EvidenceStatus
    evidence: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved(self) -> bool:
        return self.status in RESOLVED_STATUSES


class ModelChannel(Contract):
    index: int                                   # 0-based channel in the model tensor
    name: str                                    # e.g. POST_VV
    stack_band: int                              # 1-based band in the Step 3.5 stack (layout v1)
    role: Literal["PRE1", "PRE2", "POST"]
    polarization: Polarization
    kuro_siwo_input: str                         # e.g. "flood (MS1_IVV)"
    temporal_meaning: str
    mean: float
    std: float


class NormalizationSpec(Contract):
    method: Literal["kuro_siwo_clamp_nanfill_zscore"] = "kuro_siwo_clamp_nanfill_zscore"
    clamp_min: float = KURO_CLAMP_MIN
    clamp_max: float = KURO_CLAMP_MAX
    nan_fill: float = KURO_NAN_FILL
    arithmetic: Literal["float32"] = "float32"
    order: str = ("clamp(x, clamp_min, clamp_max) -> NaN to nan_fill -> (x - mean[c]) / std[c]; "
                  "identical for all time steps")
    statistics_source: str = ("fixed Kuro Siwo constants (configs/train/data_config.json L16-L17), NOT computed "
                              "from BATO or Trishuli data; their derivation is not in the repository")

    @model_validator(mode="after")
    def _check(self) -> NormalizationSpec:
        if (self.clamp_min, self.clamp_max, self.nan_fill) != (KURO_CLAMP_MIN, KURO_CLAMP_MAX, KURO_NAN_FILL):
            raise ValueError("normalisation constants must equal the Kuro Siwo values (clamp [0, 0.15], NaN -> 0.15)")
        return self


class TilingSpec(Contract):
    tile_size: int = KURO_TILE_SIZE
    stride: int = KURO_TILE_SIZE
    origin: Literal["stack pixel (row 0, col 0)"] = "stack pixel (row 0, col 0)"
    # shift_to_fit: the last tile of an axis starts at (length - 224), so it ends exactly at the stack edge and holds
    # only real pixels (overlapping its neighbour); padding is used only for an axis shorter than 224.
    # pad_end: the last tile runs past the edge and is padded (kept for comparison; NOT used for inference: ViT
    # attention is global, so padding changes the outputs of the real pixels in the tile).
    edge_policy: Literal["shift_to_fit", "pad_end"] = "shift_to_fit"
    pad_raw_value: float = KURO_SAR_NODATA        # Kuro Siwo SAR no-data value, before normalisation
    padded_pixels_valid: Literal[False] = False
    resampling: Literal["none"] = "none"
    redundant_tiles: Literal["skipped"] = "skipped"   # a tile is added only while the previous one ends inside

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overlap(self) -> int:
        return self.tile_size - self.stride

    @model_validator(mode="after")
    def _check(self) -> TilingSpec:
        if self.tile_size != KURO_TILE_SIZE:
            raise ValueError(f"tile_size must be {KURO_TILE_SIZE} (Kuro Siwo GRD samples are 224 x 224 at 10 m)")
        if not 1 <= self.stride <= self.tile_size:
            raise ValueError(f"stride must be in 1..{self.tile_size} (no gaps between tiles)")
        if self.pad_raw_value != KURO_SAR_NODATA:
            raise ValueError("pad_raw_value must be the SAR no-data value 0.0")
        return self


class ValidityPolicy(Contract):
    mask_source: str = "Step 3.5 valid_mask.tif: 1 = finite and != 0.0 in ALL six stack layers, 0 = not"
    mask_dtype: Literal["uint8"] = "uint8"
    mask_values: str = "1 = valid (observed in all three passes, both polarisations), 0 = invalid or padding"
    input_values_at_invalid: str = ("stack values passed through unchanged (per-layer SAR no-data 0.0, as in Kuro Siwo "
                                    "files); padding = 0.0; then the same normalisation as valid pixels")
    output_rule: str = ("every prediction or label at mask 0 must be replaced by the ignore label 3 "
                        "(ImpactClass.UNOBSERVABLE downstream); it is never read as water or as non-water")
    ignore_label: int = IGNORE_LABEL
    radar_visibility: str = ("NOT AVAILABLE: no layover/shadow layer exists (Step 3.4 saved none; Kuro Siwo saves "
                             "none). The mask carries SAR no-data only, not radar visibility")


class ModelInputSpec(Contract):
    version: Literal["bato-model-input-v1"] = MODEL_INPUT_VERSION
    stack_layout: Literal["bato-s1-stack-v1"] = STACK_LAYOUT
    kuro_siwo_repository: str = KURO_SIWO_REPOSITORY
    kuro_siwo_commit: str = KURO_SIWO_COMMIT
    kuro_siwo_configuration: str = KURO_SIWO_CONFIGURATION
    dtype: Literal["float32"] = "float32"
    tensor_layout: Literal["C,H,W"] = "C,H,W"
    channels: tuple[ModelChannel, ...]
    value_domain: str = "linear sigma0 (intensity) from the Step 3.5 stack; no dB, no log"
    normalization: NormalizationSpec
    dem_features: Literal["none"] = "none"
    tiling: TilingSpec
    validity: ValidityPolicy
    items: tuple[SpecItem, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (len(self.channels), self.tiling.tile_size, self.tiling.tile_size)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved_items(self) -> tuple[str, ...]:
        return tuple(f"{i.key} [{i.status.value}]: {i.statement}" for i in self.items if not i.resolved)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kuro_siwo_compatibility(self) -> str:
        if self.unresolved_items:
            return ("compatible at code level with the default Kuro Siwo GRD segmentation pipeline; "
                    f"{len(self.unresolved_items)} item(s) unresolved")
        return "compatible with the default Kuro Siwo GRD segmentation pipeline"

    @model_validator(mode="after")
    def _check(self) -> ModelInputSpec:
        expected = [("POST", "VV", 5), ("POST", "VH", 6), ("PRE2", "VV", 3), ("PRE2", "VH", 4),
                    ("PRE1", "VV", 1), ("PRE1", "VH", 2)]
        got = [(c.role, c.polarization.value, c.stack_band) for c in self.channels]
        if got != expected or [c.index for c in self.channels] != list(range(6)):
            raise ValueError("channels must be POST_VV, POST_VH, PRE2_VV, PRE2_VH, PRE1_VV, PRE1_VH "
                             "(Kuro Siwo flood, sec1 = later pre-event, sec2 = earlier pre-event)")
        for c in self.channels:
            if c.name != f"{c.role}_{c.polarization.value}":
                raise ValueError(f"channel name {c.name} does not match its role/polarisation")
            if (c.mean, c.std) != (KURO_MEAN[c.polarization.value], KURO_STD[c.polarization.value]):
                raise ValueError(f"{c.name}: mean/std must be the Kuro Siwo {c.polarization.value} constants")
        keys = [i.key for i in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("spec item keys must be unique")
        return self


def spec_fingerprint(spec: ModelInputSpec) -> str:
    """SHA-256 of the canonical JSON of a specification (computed fields included)."""
    canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest of one model-input preparation run
# ---------------------------------------------------------------------------


class SourcePass(Contract):
    role: Literal["PRE1", "PRE2", "POST"]
    pass_id: str
    acquisition_start: UTCDatetime
    days_from_event: float
    kuro_siwo_input: str                         # flood / sec1 / sec2


class SourceStackRecord(Contract):
    directory: RelativePath
    provenance: RelativePath
    provenance_sha256: Sha256Hex
    stack: RelativePath
    stack_sha256: Sha256Hex
    mask: RelativePath
    mask_sha256: Sha256Hex
    layout_version: str
    stack_config_fingerprint: str
    track: str
    passes: tuple[SourcePass, ...]


class StackGridRecord(Contract):
    crs: Literal["EPSG:3857"] = "EPSG:3857"
    pixel_size_m: float
    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]


class TileRecord(Contract):
    tile_id: str                                 # r<row_off>_c<col_off>
    row_off: int
    col_off: int
    data_height: int                             # rows inside the stack (the rest is padding)
    data_width: int
    valid_pixels: int                            # mask == 1 inside the tile
    transform: tuple[float, float, float, float, float, float]

    @model_validator(mode="after")
    def _check(self) -> TileRecord:
        if self.tile_id != f"r{self.row_off:05d}_c{self.col_off:05d}":
            raise ValueError("tile_id must be r<row_off:05d>_c<col_off:05d>")
        if not (0 < self.data_height <= KURO_TILE_SIZE and 0 < self.data_width <= KURO_TILE_SIZE):
            raise ValueError("data extent must be 1..224")
        if not 0 <= self.valid_pixels <= self.data_height * self.data_width:
            raise ValueError("valid_pixels must be within the data extent")
        return self


class ChannelDiagnostics(Contract):
    """Diagnostic only (no thresholds): how much of each channel the Kuro Siwo clamp saturates."""

    name: str
    valid_pixels: int                            # mask == 1
    above_clamp_pixels: int                      # valid and > clamp_max (saturated to 0.15)
    above_clamp_fraction: float


class ModelInputManifest(Contract):
    kind: Literal["model_input_manifest"] = "model_input_manifest"
    stage: Literal["model_input"] = "model_input"
    bato_version: str
    config_fingerprint: str
    created_at: UTCDatetime
    aoi_id: str
    candidate_id: str
    event_date: str
    source: SourceStackRecord
    grid: StackGridRecord
    spec: ModelInputSpec
    spec_fingerprint: Sha256Hex
    tiles: tuple[TileRecord, ...]
    channel_diagnostics: tuple[ChannelDiagnostics, ...]
    valid_pixels: int                            # mask == 1 in the whole stack
    warnings: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tiles_with_valid_pixels(self) -> int:
        return sum(1 for t in self.tiles if t.valid_pixels > 0)

    @model_validator(mode="after")
    def _check(self) -> ModelInputManifest:
        if self.spec_fingerprint != spec_fingerprint(self.spec):
            raise ValueError("spec_fingerprint does not match the recorded specification")
        if self.source.layout_version != self.spec.stack_layout:
            raise ValueError(f"source stack layout {self.source.layout_version} is not {self.spec.stack_layout}")
        if [p.role for p in self.source.passes] != ["PRE1", "PRE2", "POST"]:
            raise ValueError("source passes must be PRE1, PRE2, POST")
        if not (self.source.passes[0].acquisition_start < self.source.passes[1].acquisition_start
                < self.source.passes[2].acquisition_start):
            raise ValueError("source passes must be in time order PRE1 < PRE2 < POST")
        if [d.name for d in self.channel_diagnostics] != [c.name for c in self.spec.channels]:
            raise ValueError("channel diagnostics must follow the model channel order")
        if not self.tiles:
            raise ValueError("a manifest needs at least one tile")
        counted = sum(t.valid_pixels for t in self.tiles)
        exact = self.spec.tiling.overlap == 0 and self.spec.tiling.edge_policy == "pad_end"
        if (counted != self.valid_pixels) if exact else (counted < self.valid_pixels):
            raise ValueError(f"tiles hold {counted} valid pixels, the stack {self.valid_pixels} (tiling must cover "
                             "every pixel exactly once for pad_end without overlap, otherwise at least once)")
        return self

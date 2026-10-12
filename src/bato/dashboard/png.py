"""Minimal PNG encoder (standard library + numpy only) for the dashboard's precomputed map overlays and SAR chips."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}          # channels -> PNG colour type (grey, grey+alpha, RGB, RGBA)


def encode_png(image: np.ndarray) -> bytes:
    """(h, w) or (h, w, c) uint8 array -> PNG bytes (c in 1..4)."""
    a = np.asarray(image)
    if a.dtype != np.uint8:
        raise ValueError("encode_png needs uint8 data")
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3 or a.shape[2] not in _COLOR_TYPE or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"unsupported image shape {a.shape}")
    h, w, c = a.shape
    raw = np.concatenate([np.zeros((h, 1), dtype=np.uint8), a.reshape(h, w * c)], axis=1).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, _COLOR_TYPE[c], 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) +
            chunk(b"IEND", b""))


def write_png(path: Path, image: np.ndarray) -> None:
    path.write_bytes(encode_png(image))


def decode_png(data: bytes) -> np.ndarray:
    """Decoder for the files encode_png writes (filter 0 only) - used by the tests."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, w, h, c = 8, b"", 0, 0, 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, _depth, ct = struct.unpack(">IIBB", body[:10])
            c = {v: k for k, v in _COLOR_TYPE.items()}[ct]
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * c)
    if (raw[:, 0] != 0).any():
        raise ValueError("only filter type 0 is supported")
    return raw[:, 1:].reshape(h, w, c).squeeze()

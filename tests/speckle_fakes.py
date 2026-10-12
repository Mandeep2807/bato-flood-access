"""TEST FIXTURES for Step 3.3: a fake gpt that "speckle filters" a Step 3.2 DIMAP product
(3x3 mean of valid pixels - NOT Lee Sigma; it only has to change the data plausibly) and can
produce deliberately broken outputs. No SNAP, no real imagery."""

from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bato.preprocessing.snap import graph_operators, graph_path

from snap_fakes import DIAG, FakeSnap, help_text

# Value sets as SNAP's Speckle-Filter declares them (microwave-toolbox SpeckleFilterOp).
SPECKLE_VALUE_SETS: dict[str, tuple[str, ...]] = {
    "filter": ("None", "Boxcar", "Median", "Frost", "Gamma Map", "Lee", "Refined Lee", "Lee Sigma", "IDAN"),
    "numLooksStr": ("1", "2", "3", "4"),
    "windowSize": ("5x5", "7x7", "9x9", "11x11", "13x13", "15x15", "17x17"),
    "targetWindowSizeStr": ("3x3", "5x5"),
    "sigmaStr": ("0.5", "0.6", "0.7", "0.8", "0.9"),
}


def speckle_help(params: list[str], value_sets: dict[str, tuple[str, ...]]) -> str:
    """`gpt Speckle-Filter -h` in SNAP's format (value sets as 'Value must be one of ...')."""
    lines = ["Usage:", "  gpt Speckle-Filter [options]", "", "Parameter Options:"]
    for p in params:
        lines.append(f"  -P{p}=<string>    Sets parameter '{p}' to <string>.")
        if p in value_sets:
            lines.append("                    Value must be one of " + ", ".join(f"'{v}'" for v in value_sets[p]) + ".")
            lines.append(f"                    Default value is '{value_sets[p][0]}'.")
    return "\n".join(lines) + "\n"


def read_envi(img: Path) -> np.ndarray:
    hdr = img.with_suffix(".hdr").read_text()
    w = int(re.search(r"samples = (\d+)", hdr).group(1))
    h = int(re.search(r"lines = (\d+)", hdr).group(1))
    return np.fromfile(img, dtype=">f4").reshape(h, w).astype(np.float32)


def write_band(img: Path, array: np.ndarray, dtype: str = ">f4", envi_type: int = 4) -> None:
    h, w = array.shape
    img.with_suffix(".hdr").write_text(
        f"ENVI\ndescription = {{TEST FIXTURE}}\nsamples = {w}\nlines = {h}\nbands = 1\nheader offset = 0\n"
        f"file type = ENVI Standard\ndata type = {envi_type}\ninterleave = bsq\nbyte order = 1\n"
        f"band names = {{ {img.stem} }}\n", encoding="ascii")
    array.astype(dtype).tofile(img)


def mean3(a: np.ndarray, nodata: float = 0.0) -> np.ndarray:
    """3x3 mean over valid neighbours; no-data and NaN pixels stay as they are."""
    valid = np.isfinite(a) & (a != nodata)
    v = np.where(valid, a, 0.0).astype(np.float64)
    pad_v, pad_n = np.pad(v, 1), np.pad(valid.astype(np.float64), 1)
    h, w = a.shape
    total = sum(pad_v[i:i + h, j:j + w] for i in range(3) for j in range(3))
    count = sum(pad_n[i:i + h, j:j + w] for i in range(3) for j in range(3))
    out = a.astype(np.float64).copy()
    out[valid] = total[valid] / count[valid]
    return out.astype(np.float32)


def append_history(dim_text: str, *operators: str) -> str:
    nodes = "".join(f'<MDElem name="node.x{i}"><MDATTR name="id">{op}</MDATTR>'
                    f'<MDATTR name="operator">{op}</MDATTR></MDElem>' for i, op in enumerate(operators))
    marker = '<MDElem name="Processing_Graph">'
    head, tail = dim_text.split(marker, 1)
    body, rest = _split_graph(tail)
    return head + marker + body + nodes + rest


def _split_graph(tail: str) -> tuple[str, str]:
    """Split the Processing_Graph content from what follows its closing tag."""
    depth, i = 1, 0
    while depth:
        nxt_open, nxt_close = tail.find("<MDElem", i), tail.find("</MDElem>", i)
        if nxt_open != -1 and nxt_open < nxt_close:
            depth, i = depth + 1, nxt_open + 7
        else:
            depth, i = depth - 1, nxt_close + 9
    return tail[:i - 9], tail[i - 9:]


@dataclass
class FakeSpeckleSnap(FakeSnap):
    """gpt imitation for the Step 3.3 graph (TEST FIXTURE)."""

    value_sets: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(SPECKLE_VALUE_SETS))
    filter_fn: Callable[[np.ndarray], np.ndarray] = mean3
    mutate: Callable[[Path, Path], None] | None = None     # (output .dim, input .dim) -> break something
    history: tuple[str, ...] = ("Read", "Speckle-Filter", "Write")

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        self.run_kwargs.append(kwargs)
        if args[1] == "--diag":
            return subprocess.CompletedProcess(args, 0, DIAG, "")
        operator = args[1]
        text = re.sub(r"<!--.*?-->", "", graph_path("s1_speckle.xml").read_text(encoding="utf-8"), flags=re.DOTALL)
        params = {op: list(p) for _n, op, p in graph_operators(re.sub(r"\$\{[a-z_]+\}", "x", text))}
        known = [p for p in params.get(operator, []) if (operator, p) != self.missing_param]
        if operator == "Speckle-Filter":
            return subprocess.CompletedProcess(args, 0, speckle_help(known, self.value_sets), "")
        return subprocess.CompletedProcess(args, 0, help_text(known), "")

    def produce_output(self, graph_xml: str) -> None:
        root = ET.fromstring(graph_xml)
        src = Path(root.find(".//node[@id='Read']/parameters/file").text)
        out = Path(root.find(".//node[@id='Write']/parameters/file").text)
        if not self.write_output:
            return
        shutil.copytree(src.with_suffix(".data"), out.with_suffix(".data"))
        text = src.read_text(encoding="ISO-8859-1").replace(f"{src.stem}.data/", f"{out.stem}.data/")
        out.write_text(append_history(text, *self.history), encoding="ISO-8859-1")
        for img in sorted(out.with_suffix(".data").glob("Sigma0_*.img")):
            write_band(img, self.filter_fn(read_envi(img)))
        if self.mutate is not None:
            self.mutate(out, src)


# ---- output breakers (out_dim, in_dim) -> None ----------------------------------------------------------

def set_unit_db(out: Path, _src: Path) -> None:
    out.write_text(out.read_text(encoding="ISO-8859-1").replace("<PHYSICAL_UNIT>intensity</PHYSICAL_UNIT>",
                                                              "<PHYSICAL_UNIT>intensity_db</PHYSICAL_UNIT>"),
                   encoding="ISO-8859-1")


def add_band(out: Path, _src: Path) -> None:
    data = out.with_suffix(".data")
    shutil.copy(data / "Sigma0_VV.img", data / "Gamma0_VV.img")
    shutil.copy(data / "Sigma0_VV.hdr", data / "Gamma0_VV.hdr")
    text = out.read_text(encoding="ISO-8859-1")
    text = text.replace("</Data_Access>", f'<Data_File><DATA_FILE_PATH href="{out.stem}.data/Gamma0_VV.hdr"/>'
                        "<BAND_INDEX>9</BAND_INDEX></Data_File></Data_Access>")
    text = text.replace("</Image_Interpretation>", "<Spectral_Band_Info><BAND_INDEX>9</BAND_INDEX>"
                        "<BAND_NAME>Gamma0_VV</BAND_NAME><DATA_TYPE>float32</DATA_TYPE>"
                        "<PHYSICAL_UNIT>intensity</PHYSICAL_UNIT></Spectral_Band_Info></Image_Interpretation>")
    out.write_text(text, encoding="ISO-8859-1")


def shrink_vh(out: Path, _src: Path) -> None:
    img = out.with_suffix(".data") / "Sigma0_VH.img"
    write_band(img, read_envi(img)[:, :-1])


def map_projection(out: Path, _src: Path) -> None:
    text = out.read_text(encoding="ISO-8859-1")
    out.write_text(text.replace("<Geoposition>", "<Geoposition><IMAGE_TO_MODEL_TRANSFORM>10.0,0.0,0.0,-10.0,"
                                "300000.0,3100000.0</IMAGE_TO_MODEL_TRANSFORM>", 1), encoding="ISO-8859-1")


def int_vv(out: Path, _src: Path) -> None:
    img = out.with_suffix(".data") / "Sigma0_VV.img"
    write_band(img, np.ones(read_envi(img).shape), dtype=">i2", envi_type=2)


def drop_vh_file(out: Path, _src: Path) -> None:
    (out.with_suffix(".data") / "Sigma0_VH.img").unlink()


def nan_corruption(out: Path, _src: Path) -> None:
    img = out.with_suffix(".data") / "Sigma0_VV.img"
    a = read_envi(img)
    a[20:22, 20:22] = np.nan
    write_band(img, a)


def move_geolocation(out: Path, _src: Path) -> None:
    img = out.with_suffix(".data") / "tie_point_grids" / "latitude.img"
    write_band(img, read_envi(img) + 0.01)


def change_spacing(out: Path, _src: Path) -> None:
    text = out.read_text(encoding="ISO-8859-1")
    out.write_text(re.sub(r'(<MDATTR name="range_spacing"[^>]*>)[^<]*', r"\g<1>20.0", text), encoding="ISO-8859-1")


def modify_input(_out: Path, src: Path) -> None:
    img = src.with_suffix(".data") / "Sigma0_VV.img"
    write_band(img, read_envi(img) * 2)


def change_orbit(out: Path, _src: Path) -> None:
    text = out.read_text(encoding="ISO-8859-1")
    out.write_text(re.sub(r'(<MDATTR name="orbit_state_vector_file"[^>]*>)[^<]*', r"\g<1>Sentinel Restituted X",
                          text), encoding="ISO-8859-1")

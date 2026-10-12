"""TEST FIXTURES for Step 3.2: a tiny Sentinel-1 SAFE zip, its download provenance and
fetch manifest (built from the REAL Trishuli/Valencia catalogue metadata), and a fake
SNAP gpt that writes a small BEAM-DIMAP product. No real imagery, no SNAP needed."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bato.acquisition.s1_fetch import build_s1_fetch_plan, completeness
from bato.config import BatoConfig
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.common import DataSource
from bato.contracts.fetch import DownloadProvenance, FetchManifest, FetchStatus, FileFetchResult
from bato.contracts.io import load_contract, save_contract
from bato.contracts.temporal import TemporalSelectionResult

REAL = Path(__file__).resolve().parent / "fixtures" / "real_catalogues"
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
TRISHULI_PRE1 = "S1D-A085-20260629T122138"
TRISHULI_POST = "S1D-A085-20260828T122141"
SAFE_SAMPLES, SAFE_LINES = 400, 300          # tiny synthetic scene (real: 25801 x 16730)


# ---------------------------------------------------------------------------
# SAFE zip + provenance + manifest
# ---------------------------------------------------------------------------


def safe_zip_bytes(product_name: str, *, pass_: str = "Ascending") -> bytes:
    """A SAFE zip with the real member layout and tiny contents (TEST FIXTURE)."""
    top = product_name if product_name.endswith(".SAFE") else product_name + ".SAFE"
    mission = top[:3].lower()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{top}/manifest.safe", "<xfdu/>")
        for n, pol in enumerate(("vv", "vh"), start=1):
            stem = f"{mission}-iw-grd-{pol}-20260629t122138-20260629t122203-003451-006174-00{n}"
            grid = "".join("<geolocationGridPoint><line>0</line><pixel>0</pixel></geolocationGridPoint>"
                           for _ in range(6))
            ann = (f"<product><adsHeader><productType>GRD</productType><polarisation>{pol.upper()}</polarisation>"
                   f"</adsHeader><generalAnnotation><productInformation><pass>{pass_}</pass></productInformation>"
                   f"</generalAnnotation><imageAnnotation><imageInformation>"
                   f"<numberOfSamples>{SAFE_SAMPLES}</numberOfSamples><numberOfLines>{SAFE_LINES}</numberOfLines>"
                   f"<rangePixelSpacing>1.000000e+01</rangePixelSpacing>"
                   f"<azimuthPixelSpacing>1.000000e+01</azimuthPixelSpacing></imageInformation></imageAnnotation>"
                   f"<geolocationGrid><geolocationGridPointList>{grid}</geolocationGridPointList></geolocationGrid>"
                   f"</product>")
            vec = ("<calibrationVector><line>{line}</line><pixel>0 100 200 399</pixel>"
                   "<sigmaNought>5.0e+02 5.0e+02 5.0e+02 5.0e+02</sigmaNought></calibrationVector>")
            cal = (f"<calibration><calibrationVectorList>{vec.format(line=0)}{vec.format(line=SAFE_LINES - 1)}"
                   "</calibrationVectorList></calibration>")
            z.writestr(f"{top}/measurement/{stem}.tiff", b"\0" * (SAFE_SAMPLES * SAFE_LINES * 2 + 8))
            z.writestr(f"{top}/annotation/{stem}.xml", ann)
            z.writestr(f"{top}/annotation/calibration/calibration-{stem}.xml", cal)
            z.writestr(f"{top}/annotation/calibration/noise-{stem}.xml", "<noise/>")
    return buf.getvalue()


@dataclass
class Project:
    cfg: BatoConfig
    root: Path
    manifest: FetchManifest
    manifest_path: Path
    catalogue_path: Path


def make_project(cfg: BatoConfig, *, name: str = "trishuli", track: str = "ASCENDING/85",
                 present_passes: tuple[str, ...] = (TRISHULI_PRE1,), copy_catalogue: bool = True) -> Project:
    """Real Step 2A/2B metadata -> real fetch plan; the listed passes get synthetic SAFE zips."""
    root = cfg.config_dir.parent
    selection = load_contract(TemporalSelectionResult, REAL / f"{name}_sets.json")
    catalogue = load_contract(CatalogueSearchResult, REAL / f"{name}_s1.json")
    plan = build_s1_fetch_plan(selection, catalogue, cfg, tracks=[track])
    catalogue_path = cfg.settings.paths.interim / "catalogue" / f"{name}_s1.json"
    if copy_catalogue:
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL / f"{name}_s1.json", catalogue_path)
    s1_dir = cfg.settings.paths.raw / "sentinel1"
    s1_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for scene in plan.scenes:
        if scene.pass_id not in present_passes:
            files.append(FileFetchResult(kind="sentinel1_grd", product_id=scene.product_id,
                                         product_name=scene.product_name, pass_id=scene.pass_id,
                                         status=FetchStatus.NOT_ATTEMPTED))
            continue
        data = safe_zip_bytes(scene.product_name)
        path = s1_dir / f"{scene.product_name}.zip"
        path.write_bytes(data)
        sidecar = path.with_name(path.name + ".provenance.json")
        rel = path.relative_to(root).as_posix()
        save_contract(DownloadProvenance(
            kind="sentinel1_grd", data_source=DataSource.SENTINEL_1, product_id=scene.product_id,
            product_name=scene.product_name, source="cdse-odata-download",
            source_url=f"https://download.dataspace.copernicus.eu/odata/v1/Products({scene.product_id})/$value",
            content_length=len(data), server_checksums={}, checksum_verified_with=None, checksum_verified=False,
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), local_path=rel,
            content_check="TEST FIXTURE", downloaded_at=NOW, attempts=1, resumed=False, bato_version="0.1.0",
            config_fingerprint=cfg.fingerprint(), platform=scene.platform, acquisition_start=scene.acquisition_start,
            orbit_direction=scene.orbit_direction, relative_orbit=scene.relative_orbit,
            polarizations=scene.polarizations, product_type=scene.product_type, pass_ids=(scene.pass_id,),
            candidate_ids=scene.candidate_ids), sidecar)
        files.append(FileFetchResult(kind="sentinel1_grd", product_id=scene.product_id,
                                     product_name=scene.product_name, pass_id=scene.pass_id,
                                     status=FetchStatus.DOWNLOADED, size_bytes=len(data), local_path=rel,
                                     provenance_path=sidecar.relative_to(root).as_posix()))
    pass_complete, candidate_ready = completeness(plan, files)
    manifest = FetchManifest(aoi_id=plan.aoi_id, dry_run=False, started_at=NOW, finished_at=NOW,
                             selection_digest=plan.selection_digest, plan=plan, dem_plan=None, files=tuple(files),
                             pass_complete=pass_complete, candidate_ready=candidate_ready)
    manifest_path = save_contract(manifest, cfg.settings.paths.raw / "manifests" / f"{plan.aoi_id}_fetch_manifest.json")
    return Project(cfg=cfg, root=root, manifest=manifest, manifest_path=manifest_path, catalogue_path=catalogue_path)


# ---------------------------------------------------------------------------
# Fake BEAM-DIMAP product (ENVI big-endian float32, like SNAP writes)
# ---------------------------------------------------------------------------


def write_envi(img: Path, array: np.ndarray, band_name: str) -> None:
    img.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    img.with_suffix(".hdr").write_text(
        "ENVI\ndescription = {TEST FIXTURE}\n"
        f"samples = {w}\nlines = {h}\nbands = 1\nheader offset = 0\nfile type = ENVI Standard\n"
        f"data type = 4\ninterleave = bsq\nbyte order = 1\nband names = {{ {band_name} }}\n", encoding="ascii")
    array.astype(">f4").tofile(img)


def default_bands(width: int, height: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    bands = {}
    for name, scale in (("Sigma0_VV", 0.1), ("Sigma0_VH", 0.02)):
        a = rng.gamma(1.0, scale, size=(height, width)).astype(np.float32)
        a[:, :3] = 0.0                    # border-noise strip (no-data)
        a[5, 7] = np.nan
        bands[name] = a
    return bands


@dataclass
class DimapSpec:
    width: int = 60
    height: int = 40
    bands: dict[str, np.ndarray] | None = None
    unit: str = "intensity"
    nodata: float | None = 0.0
    log10_scaled: bool = False
    corners: tuple[tuple[float, float], ...] = ((85.0, 28.4), (85.6, 28.4), (85.6, 27.9), (85.0, 27.9))
    orbit_file: str = ("Sentinel Precise S1D_OPER_AUX_POEORB_OPOD_20260719T070623_V20260628T225942_"
                       "20260630T005942.EOF.zip")
    map_geocoding: bool = False
    terrain_corrected: str = "0"
    history: tuple[str, ...] = ("Read", "Apply-Orbit-File", "Subset", "ThermalNoiseRemoval",
                                "Remove-GRD-Border-Noise", "Calibration", "Write")
    range_spacing: float = 10.0
    azimuth_spacing: float = 10.0
    skip_files: tuple[str, ...] = ()      # band names whose .img is not written (incomplete output)


def write_dimap(dim: Path, spec: DimapSpec) -> None:
    """A small BEAM-DIMAP product with the structure SNAP writes (TEST FIXTURE)."""
    data_dir = dim.with_suffix(".data")
    bands = spec.bands if spec.bands is not None else default_bands(spec.width, spec.height)
    stem = dim.stem
    data_files, band_infos = [], []
    for i, (name, array) in enumerate(bands.items()):
        if name not in spec.skip_files:
            write_envi(data_dir / f"{name}.img", array, name)
        data_files.append(f'<Data_File><DATA_FILE_PATH href="{stem}.data/{name}.hdr"/>'
                          f"<BAND_INDEX>{i}</BAND_INDEX></Data_File>")
        h, w = array.shape
        nodata = (f"<NO_DATA_VALUE_USED>true</NO_DATA_VALUE_USED><NO_DATA_VALUE>{spec.nodata}</NO_DATA_VALUE>"
                  if spec.nodata is not None else "<NO_DATA_VALUE_USED>false</NO_DATA_VALUE_USED>")
        band_infos.append(
            f"<Spectral_Band_Info><BAND_INDEX>{i}</BAND_INDEX><BAND_NAME>{name}</BAND_NAME>"
            f"<BAND_RASTER_WIDTH>{w}</BAND_RASTER_WIDTH><BAND_RASTER_HEIGHT>{h}</BAND_RASTER_HEIGHT>"
            f"<DATA_TYPE>float32</DATA_TYPE><PHYSICAL_UNIT>{spec.unit}</PHYSICAL_UNIT>"
            f"<LOG10_SCALED>{str(spec.log10_scaled).lower()}</LOG10_SCALED>{nodata}</Spectral_Band_Info>")
    # 3 x 3 tie-point grids spanning the corners (top-left, top-right, bottom-right, bottom-left)
    (tl, tr, br, bl) = spec.corners
    u, v = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    lon = (1 - v) * ((1 - u) * tl[0] + u * tr[0]) + v * ((1 - u) * bl[0] + u * br[0])
    lat = (1 - v) * ((1 - u) * tl[1] + u * tr[1]) + v * ((1 - u) * bl[1] + u * br[1])
    tpg_files, tpg_infos = [], []
    for i, (name, grid) in enumerate((("latitude", lat), ("longitude", lon))):
        write_envi(data_dir / "tie_point_grids" / f"{name}.img", grid, name)
        tpg_files.append(f'<Tie_Point_Grid_File><TIE_POINT_GRID_FILE_PATH href="{stem}.data/tie_point_grids/'
                         f'{name}.hdr"/><TIE_POINT_GRID_INDEX>{i}</TIE_POINT_GRID_INDEX></Tie_Point_Grid_File>')
        tpg_infos.append(f"<Tie_Point_Grid_Info><TIE_POINT_GRID_INDEX>{i}</TIE_POINT_GRID_INDEX>"
                         f"<TIE_POINT_GRID_NAME>{name}</TIE_POINT_GRID_NAME><NCOLS>3</NCOLS><NROWS>3</NROWS>"
                         "</Tie_Point_Grid_Info>")
    geopos = ("<Geoposition><IMAGE_TO_MODEL_TRANSFORM>10.0,0.0,0.0,-10.0,300000.0,3100000.0"
              "</IMAGE_TO_MODEL_TRANSFORM></Geoposition>" if spec.map_geocoding else
              "<Geoposition><TIE_POINT_GRID_NAME_LAT>latitude</TIE_POINT_GRID_NAME_LAT>"
              "<TIE_POINT_GRID_NAME_LON>longitude</TIE_POINT_GRID_NAME_LON></Geoposition>")
    abstracted = {"PRODUCT": stem, "range_spacing": spec.range_spacing, "azimuth_spacing": spec.azimuth_spacing,
                  "orbit_state_vector_file": spec.orbit_file, "is_terrain_corrected": spec.terrain_corrected}
    md = "".join(f'<MDATTR name="{k}" type="ascii">{v}</MDATTR>' for k, v in abstracted.items())
    history = "".join(f'<MDElem name="node.{i}"><MDATTR name="id">{op}</MDATTR>'
                      f'<MDATTR name="operator">{op}</MDATTR></MDElem>' for i, op in enumerate(spec.history))
    first = next(iter(bands.values()))
    xml = (f'<?xml version="1.0" encoding="ISO-8859-1"?><Dimap_Document name="{dim.name}">'
           f"<Dataset_Id><DATASET_NAME>{stem}</DATASET_NAME></Dataset_Id>"
           f"<Coordinate_Reference_System>{geopos}</Coordinate_Reference_System>"
           f"<Raster_Dimensions><NCOLS>{first.shape[1]}</NCOLS><NROWS>{first.shape[0]}</NROWS>"
           f"<NBANDS>{len(bands)}</NBANDS></Raster_Dimensions>"
           f"<Data_Access><DATA_FILE_FORMAT>ENVI</DATA_FILE_FORMAT>{''.join(data_files)}{''.join(tpg_files)}"
           f"</Data_Access><Tie_Point_Grids><NUM_TIE_POINT_GRIDS>2</NUM_TIE_POINT_GRIDS>{''.join(tpg_infos)}"
           f"</Tie_Point_Grids><Image_Interpretation>{''.join(band_infos)}</Image_Interpretation>"
           f'<Dataset_Sources><MDElem name="metadata"><MDElem name="Abstracted_Metadata">{md}</MDElem>'
           f'<MDElem name="Processing_Graph">{history}</MDElem></MDElem></Dataset_Sources></Dimap_Document>')
    dim.write_text(xml, encoding="ISO-8859-1")


def crop_corners(graph_xml: str) -> tuple[tuple[float, float], ...]:
    """Corners of the Subset geoRegion in a rendered graph (so the fake output covers the AOI)."""
    region = ET.fromstring(graph_xml).find(".//node[@id='Subset']/parameters/geoRegion")
    assert region is not None and region.text
    pts = [tuple(float(v) for v in p.split()) for p in re.search(r"\(\((.*)\)\)", region.text).group(1).split(",")]
    lo_lon, hi_lon = min(p[0] for p in pts), max(p[0] for p in pts)
    lo_lat, hi_lat = min(p[1] for p in pts), max(p[1] for p in pts)
    return ((lo_lon, hi_lat), (hi_lon, hi_lat), (hi_lon, lo_lat), (lo_lon, lo_lat))


# ---------------------------------------------------------------------------
# Fake gpt
# ---------------------------------------------------------------------------

DIAG = ("SNAP Release version 12.0.0\nSNAP home: C:\\Program Files\\esa-snap\nJava version: 21.0.6\n"
        "Max memory: 5.0 GB\nCache size: 512.0 MB\nTile parallelism: 4\nDefault tile size: 512 x 512 pixels\n")


def help_text(params: list[str]) -> str:
    return "Usage:\n  gpt Op [options]\n\nParameter Options:\n" + "".join(f"  -P{p}=<value>    doc\n" for p in params)


@dataclass
class FakeSnap:
    """subprocess.run / subprocess.Popen replacements imitating gpt (TEST FIXTURE)."""

    spec: DimapSpec = field(default_factory=DimapSpec)
    returncode: int = 0
    write_output: bool = True
    timeout: bool = False
    missing_param: tuple[str, str] | None = None   # (operator, parameter) the fake SNAP does not know
    calls: list[list[str]] = field(default_factory=list)
    run_kwargs: list[dict] = field(default_factory=list)
    graphs: list[str] = field(default_factory=list)
    killed: bool = False

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        self.run_kwargs.append(kwargs)
        if args[1] == "--diag":
            return subprocess.CompletedProcess(args, 0, DIAG, "")
        operator = args[1]
        from bato.preprocessing.snap import graph_operators, graph_path
        params = {op: list(p) for _n, op, p in graph_operators(_strip(graph_path("s1_radiometric.xml")))}
        known = [p for p in params.get(operator, []) if (operator, p) != self.missing_param]
        return subprocess.CompletedProcess(args, 0, help_text(known), "")

    def popen(self, args, **kwargs):
        """Like a real gpt start: nothing is written yet. Output is produced while the
        process runs (in ``wait``), so a write failure cannot masquerade as 'cannot start gpt'."""
        self.calls.append(list(args))
        self.run_kwargs.append(kwargs)
        graph_xml = Path(args[1]).read_text(encoding="utf-8")
        self.graphs.append(graph_xml)
        return _Process(self, graph_xml)

    def produce_output(self, graph_xml: str) -> None:
        out = ET.fromstring(graph_xml).find(".//node[@id='Write']/parameters/file")
        assert out is not None and out.text
        if self.write_output:
            spec = self.spec
            if spec.corners == DimapSpec().corners:
                spec = DimapSpec(**{**spec.__dict__, "corners": crop_corners(graph_xml)})
            write_dimap(Path(out.text), spec)


def _strip(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\$\{[a-z_]+\}", "x", re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL))


class _Process:
    def __init__(self, fake: FakeSnap, graph_xml: str) -> None:
        self.fake = fake
        self.graph_xml = graph_xml
        self.ran = False
        self.stdout = io.BytesIO(b"Executing processing graph\n....10%....20%....50%....100% done.\n"
                                 if fake.returncode == 0 else b"Error: [NodeId: Apply-Orbit-File] No valid orbit file "
                                 b"found for 29-JUN-2026\n")

    def wait(self, timeout=None):
        if self.fake.timeout and not self.fake.killed:
            raise subprocess.TimeoutExpired("gpt", timeout)
        if not self.ran and not self.fake.killed:
            self.ran = True
            self.fake.produce_output(self.graph_xml)
        return self.fake.returncode if not self.fake.killed else -9

    def kill(self):
        self.fake.killed = True

"""TEST FIXTURES for Step 3.4: synthetic Copernicus DEM tiles registered in a Step 2C manifest, a
published Step 3.3 output made by the real Step 3.2/3.3 code with fake gpts, and a fake gpt that
"terrain corrects" (writes a small EPSG:3857 map-geometry DIMAP product; NOT a real
Range-Doppler correction). No SNAP, no real imagery, no network."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bato.config import load_config
from bato.contracts.common import DataSource
from bato.contracts.fetch import DEMPlan, DEMTile, DownloadProvenance, FetchManifest, FetchStatus, FileFetchResult
from bato.contracts.io import save_contract
from bato.preprocessing import radiometric as rad
from bato.preprocessing import snap
from bato.preprocessing import speckle as spk
from bato.preprocessing.dimap import read_dimap
from bato.preprocessing.snap import graph_operators, graph_path
from bato.preprocessing.terrain_validation import lonlat_to_3857

from helpers import REPO_CONFIG_DIR
from snap_fakes import DIAG, NOW, TRISHULI_PRE1, FakeSnap, help_text, make_project
from speckle_fakes import FakeSpeckleSnap, write_band

TRISHULI_TILES = ("N27E084", "N27E085", "N28E084", "N28E085")
DEM_RES = 0.01                                  # synthetic tiles: 100 x 100 px per degree
# TEST FIXTURE AOI (~1 x 1 km) so the fake 10 m map output stays small; written into the Step 3.2
# provenance BEFORE Step 3.3 runs, so the provenance chain hashes stay consistent.
TINY_AOI = ((85.30, 28.10), (85.31, 28.10), (85.31, 28.11), (85.30, 28.11), (85.30, 28.10))

TC_VALUE_SETS: dict[str, tuple[str, ...]] = {
    "demResamplingMethod": ("NEAREST_NEIGHBOUR", "BILINEAR_INTERPOLATION", "CUBIC_CONVOLUTION",
                            "BISINC_5_POINT_INTERPOLATION", "BICUBIC_INTERPOLATION", "DELAUNAY_INTERPOLATION"),
    "imgResamplingMethod": ("NEAREST_NEIGHBOUR", "BILINEAR_INTERPOLATION", "CUBIC_CONVOLUTION",
                            "BISINC_5_POINT_INTERPOLATION", "BICUBIC_INTERPOLATION"),
    "incidenceAngleForSigma0": ("Use incidence angle from Ellipsoid", "Use local incidence angle from DEM",
                                "Use projected local incidence angle from DEM"),
    "incidenceAngleForGamma0": ("Use incidence angle from Ellipsoid", "Use local incidence angle from DEM",
                                "Use projected local incidence angle from DEM"),
    "auxFile": ("Latest Auxiliary File", "Product Auxiliary File", "External Auxiliary File"),
}


def tile_name(tile_id: str) -> str:
    return f"Copernicus_DSM_COG_10_{tile_id[:3]}_00_{tile_id[3:]}_00_DEM"


def elevation(tile_id: str, res: float = DEM_RES) -> np.ndarray:
    """Deterministic synthetic heights (m) for a 1 x 1 degree cell."""
    lat0 = int(tile_id[1:3]) * (1 if tile_id[0] == "N" else -1)
    lon0 = int(tile_id[4:7]) * (1 if tile_id[3] == "E" else -1)
    n = round(1 / res)
    lats = lat0 + 1 - (np.arange(n) + 0.5) * res
    lons = lon0 + (np.arange(n) + 0.5) * res
    glon, glat = np.meshgrid(lons, lats)
    return (500 + 3000 * (glat - 27) + 200 * np.sin(glon * 7)).astype(np.float32)


def write_dem_tif(path: Path, tile_id: str, res: float = DEM_RES, *, shift: float = 0.0, epsg: int = 4326) -> None:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    lat0 = int(tile_id[1:3]) * (1 if tile_id[0] == "N" else -1)
    lon0 = int(tile_id[4:7]) * (1 if tile_id[3] == "E" else -1)
    data = elevation(tile_id, res)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype="float32",
                       crs=CRS.from_epsg(epsg), transform=from_origin(lon0 + shift, lat0 + 1, res, res),
                       nodata=-32767.0) as dst:
        dst.write(data, 1)


def archive_bytes(tif: Path, name: str, kind: str = "zip") -> bytes:
    member = f"{name}/DEM/{name}.tif"
    buf = io.BytesIO()
    if kind == "zip":
        with zipfile.ZipFile(buf, "w") as z:
            z.write(tif, member)
            z.writestr(f"{name}/INFO/{name}.xml", "<TEST FIXTURE/>")
    elif kind == "tar":
        with tarfile.open(fileobj=buf, mode="w") as t:
            t.add(tif, member)
    else:
        return tif.read_bytes()
    return buf.getvalue()


def add_dem_tiles(project, tiles: tuple[str, ...] = TRISHULI_TILES, *, sea: tuple[str, ...] = (),
                  not_downloaded: tuple[str, ...] = (), kinds: dict[str, str] | None = None,
                  res: dict[str, float] | None = None, shift: dict[str, float] | None = None) -> FetchManifest:
    """Register synthetic DEM tiles in the project's Step 2C manifest (archives + provenance)."""
    cfg, root = project.cfg, project.root
    dem_dir = cfg.settings.paths.raw / "copernicus_dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    planned, files = [], []
    for tile_id in (*tiles, *sea):
        if tile_id in sea:
            planned.append(DEMTile(tile_id=tile_id))
            continue
        name = tile_name(tile_id)
        planned.append(DEMTile(tile_id=tile_id, product_id=f"TEST-DEM-{tile_id}", product_name=name))
        if tile_id in not_downloaded:
            files.append(FileFetchResult(kind="copernicus_dem_tile", product_id=f"TEST-DEM-{tile_id}",
                                         product_name=name, tile_id=tile_id, status=FetchStatus.NOT_ATTEMPTED))
            continue
        tif = dem_dir / f"{tile_id}.tmp.tif"
        write_dem_tif(tif, tile_id, (res or {}).get(tile_id, DEM_RES), shift=(shift or {}).get(tile_id, 0.0))
        data = archive_bytes(tif, name, (kinds or {}).get(tile_id, "zip"))
        tif.unlink()
        path = dem_dir / f"{name}.zip"
        path.write_bytes(data)
        sidecar = path.with_name(path.name + ".provenance.json")
        save_contract(DownloadProvenance(
            kind="copernicus_dem_tile", data_source=DataSource.COPERNICUS_DEM, product_id=f"TEST-DEM-{tile_id}",
            product_name=name, source="cdse-odata-download",
            source_url=f"https://download.dataspace.copernicus.eu/odata/v1/Products(TEST-DEM-{tile_id})/$value",
            content_length=len(data), server_checksums={}, checksum_verified_with=None, checksum_verified=False,
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), local_path=path.relative_to(root).as_posix(),
            content_check="TEST FIXTURE", downloaded_at=NOW, attempts=1, resumed=False, bato_version="0.1.0",
            config_fingerprint=cfg.fingerprint(), tile_id=tile_id, dem_dataset=cfg.settings.dem.cdse_dataset), sidecar)
        files.append(FileFetchResult(kind="copernicus_dem_tile", product_id=f"TEST-DEM-{tile_id}", product_name=name,
                                     tile_id=tile_id, status=FetchStatus.DOWNLOADED, size_bytes=len(data),
                                     local_path=path.relative_to(root).as_posix(),
                                     provenance_path=sidecar.relative_to(root).as_posix()))
    plan = DEMPlan(aoi_id=project.manifest.aoi_id, dataset=cfg.settings.dem.cdse_dataset, buffer_km=20.0,
                   required_tiles=tuple(t.tile_id for t in planned), tiles=tuple(planned), query_filter="TEST FIXTURE")
    manifest = project.manifest.model_copy(update={"dem_plan": plan, "files": (*project.manifest.files, *files)})
    save_contract(manifest, project.manifest_path)
    project.manifest = manifest
    return manifest


def published_speckle(root: Path, gpt: snap.ResolvedGpt, *, tiny_aoi: bool = True):
    """Project with a PUBLISHED Step 3.3 output (real Step 3.2/3.3 code, fake gpts) and DEM tiles."""
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    project = make_project(load_config(root / "config"))
    fake = FakeSnap()
    rad_result = rad.run_radiometric(rad.prepare_job(project.manifest, TRISHULI_PRE1, project.cfg), project.cfg,
                                     gpt=gpt, runner=fake.run, popen=fake.popen)
    if tiny_aoi:
        data = json.loads(rad_result.provenance_path.read_text(encoding="utf-8"))
        data["aoi"]["aoi_polygon"]["coordinates"] = [[list(p) for p in TINY_AOI]]
        rad_result.provenance_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    fake_spk = FakeSpeckleSnap()
    spk.run_speckle(spk.prepare_speckle_job(project.manifest, TRISHULI_PRE1, project.cfg), project.cfg, gpt=gpt,
                    runner=fake_spk.run, popen=fake_spk.popen)
    add_dem_tiles(project)
    return project


def terrain_help(params: list[str], value_sets: dict[str, tuple[str, ...]]) -> str:
    lines = ["Usage:", "  gpt Terrain-Correction [options]", "", "Parameter Options:"]
    for p in params:
        lines.append(f"  -P{p}=<string>    Sets parameter '{p}' to <string>.")
        if p in value_sets:
            lines.append("                    Value must be one of " + ", ".join(f"'{v}'" for v in value_sets[p]) + ".")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Fake map-geometry DIMAP
# ---------------------------------------------------------------------------

WKT_4326 = 'GEOGCS["WGS 84", DATUM["World Geodetic System 1984"], AUTHORITY["EPSG","4326"]]'


def write_map_dimap(dim: Path, *, bands: dict[str, np.ndarray], x0: float, y0: float, pixel: float, wkt: str | None,
                    history: list[str], abstracted: dict[str, object], unit: str = "intensity",
                    nodata: float | None = 0.0, i2m: bool = True, skip_files: tuple[str, ...] = (),
                    dtype: str = ">f4", envi_type: int = 4) -> None:
    data_dir = dim.with_suffix(".data")
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = dim.stem
    files, infos = [], []
    for i, (name, array) in enumerate(bands.items()):
        if name not in skip_files:
            write_band(data_dir / f"{name}.img", array, dtype=dtype, envi_type=envi_type)
        files.append(f'<Data_File><DATA_FILE_PATH href="{stem}.data/{name}.hdr"/><BAND_INDEX>{i}</BAND_INDEX>'
                     "</Data_File>")
        h, w = array.shape
        nd = (f"<NO_DATA_VALUE_USED>true</NO_DATA_VALUE_USED><NO_DATA_VALUE>{nodata}</NO_DATA_VALUE>"
              if nodata is not None else "<NO_DATA_VALUE_USED>false</NO_DATA_VALUE_USED>")
        infos.append(f"<Spectral_Band_Info><BAND_INDEX>{i}</BAND_INDEX><BAND_NAME>{name}</BAND_NAME>"
                     f"<BAND_RASTER_WIDTH>{w}</BAND_RASTER_WIDTH><BAND_RASTER_HEIGHT>{h}</BAND_RASTER_HEIGHT>"
                     f"<DATA_TYPE>float32</DATA_TYPE><PHYSICAL_UNIT>{unit}</PHYSICAL_UNIT>"
                     f"<LOG10_SCALED>false</LOG10_SCALED>{nd}</Spectral_Band_Info>")
    crs = f"<WKT>{wkt}</WKT>" if wkt else ""
    geopos = f"<Geoposition><IMAGE_TO_MODEL_TRANSFORM>{pixel},0.0,0.0,{-pixel},{x0},{y0}</IMAGE_TO_MODEL_TRANSFORM>" \
             "</Geoposition>" if i2m else ""
    md = "".join(f'<MDATTR name="{k}" type="ascii">{v}</MDATTR>' for k, v in abstracted.items())
    hist = "".join(f'<MDElem name="node.{i}"><MDATTR name="id">{op}</MDATTR><MDATTR name="operator">{op}</MDATTR>'
                   "</MDElem>" for i, op in enumerate(history))
    first = next(iter(bands.values()))
    xml = (f'<?xml version="1.0" encoding="ISO-8859-1"?><Dimap_Document name="{dim.name}">'
           f"<Coordinate_Reference_System>{crs}</Coordinate_Reference_System>{geopos}"
           f"<Raster_Dimensions><NCOLS>{first.shape[1]}</NCOLS><NROWS>{first.shape[0]}</NROWS>"
           f"<NBANDS>{len(bands)}</NBANDS></Raster_Dimensions>"
           f"<Data_Access><DATA_FILE_FORMAT>ENVI</DATA_FILE_FORMAT>{''.join(files)}</Data_Access>"
           f"<Image_Interpretation>{''.join(infos)}</Image_Interpretation>"
           f'<Dataset_Sources><MDElem name="metadata"><MDElem name="Abstracted_Metadata">{md}</MDElem>'
           f'<MDElem name="Processing_Graph">{hist}</MDElem></MDElem></Dataset_Sources></Dimap_Document>')
    dim.write_text(xml, encoding="ISO-8859-1")


@dataclass
class TerrainOutputSpec:
    """How the fake gpt's output deviates from a correct one (TEST FIXTURE)."""

    unit: str = "intensity"
    wkt: str | None = "AS_GRAPH"            # "AS_GRAPH" -> the graph's mapProjection
    i2m: bool = True
    pixel: float = 10.0
    x_offset_m: float = 0.0                 # shifts the origin off the standard grid (5.0 = corner on the grid)
    extent_shift_deg: float = 0.0           # moves the output away from the AOI
    extra_history: tuple[str, ...] = ("Read", "Terrain-Correction", "Write")
    terrain_flag: str = "1"
    orbit_file: str | None = None           # None -> input orbit file
    extra_band: bool = False
    shrink_vh: bool = False
    skip_files: tuple[str, ...] = ()
    integer: bool = False
    all_nodata: bool = False


@dataclass
class FakeTerrainSnap(FakeSnap):
    """gpt imitation for the Step 3.4 graph (TEST FIXTURE)."""

    value_sets: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(TC_VALUE_SETS))
    out: TerrainOutputSpec = field(default_factory=TerrainOutputSpec)
    after_run: Callable[[Path, Path, Path], None] | None = None   # (output dim, input dim, dem file)
    dem_seen: list[str] = field(default_factory=list)

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        self.run_kwargs.append(kwargs)
        if args[1] == "--diag":
            return subprocess.CompletedProcess(args, 0, DIAG, "")
        operator = args[1]
        text = re.sub(r"<!--.*?-->", "", graph_path("s1_terrain.xml").read_text(encoding="utf-8"), flags=re.DOTALL)
        params = {op: list(p) for _n, op, p in graph_operators(re.sub(r"\$\{[a-z_]+\}", "x", text))}
        known = [p for p in params.get(operator, []) if (operator, p) != self.missing_param]
        if operator == "Terrain-Correction":
            return subprocess.CompletedProcess(args, 0, terrain_help(known, self.value_sets), "")
        return subprocess.CompletedProcess(args, 0, help_text(known), "")

    def produce_output(self, graph_xml: str) -> None:
        root = ET.fromstring(graph_xml)
        src = Path(root.find(".//node[@id='Read']/parameters/file").text)
        out = Path(root.find(".//node[@id='Write']/parameters/file").text)
        dem = Path(root.find(".//node[@id='Terrain-Correction']/parameters/externalDEMFile").text)
        assert dem.is_file(), "the external DEM must exist when gpt runs"
        self.dem_seen.append(hashlib.sha256(dem.read_bytes()).hexdigest())
        if not self.write_output:
            return
        o = self.out
        product = read_dimap(src)
        median = {b.name: float(np.nanmedian(np.where(np.fromfile(b.image_file, ">f4") > 0,
                                                      np.fromfile(b.image_file, ">f4"), np.nan)))
                  for b in product.bands if b.image_file is not None}
        lons = np.array([TINY_AOI[0][0] - 0.003 + o.extent_shift_deg, TINY_AOI[1][0] + 0.003 + o.extent_shift_deg])
        lats = np.array([TINY_AOI[0][1] - 0.003, TINY_AOI[2][1] + 0.003])
        xs, ys = lonlat_to_3857(lons, lats)
        # like SNAP's CRSGeoCodingHandler + CrsGeoCoding: the CENTRE of pixel (0, 0) is snapped to the
        # 10 m standard grid; IMAGE_TO_MODEL_TRANSFORM holds the corner, half a pixel up-left of it
        x0 = float(np.floor(xs[0] / 10.0) * 10.0) - o.pixel / 2 + o.x_offset_m
        y0 = float(np.ceil(ys[1] / 10.0) * 10.0) + o.pixel / 2
        width = int(np.ceil((xs[1] - x0) / o.pixel))
        height = int(np.ceil((y0 - ys[0]) / o.pixel))
        rng = np.random.default_rng(7)
        bands = {}
        for name, med in median.items():
            a = (rng.gamma(4.0, med / 4.0, size=(height, width))).astype(np.float32)
            a[:, :5] = 0.0                          # no-data strip outside the AOI (outside the swath)
            if o.all_nodata:
                a[:] = 0.0
            if o.shrink_vh and name == "Sigma0_VH":
                a = a[:, :-1]
            bands[name] = a
        if o.extra_band:
            bands["projectedLocalIncidenceAngle"] = np.full((height, width), 30.0, np.float32)
        tc_params = {p: v for _n, op, params in graph_operators(graph_xml) if op == "Terrain-Correction"
                     for p, v in params.items()}
        wkt = tc_params["mapProjection"] if o.wkt == "AS_GRAPH" else o.wkt
        write_map_dimap(out, bands=bands, x0=x0, y0=y0, pixel=o.pixel, wkt=wkt,
                        history=[*product.history, *o.extra_history],
                        abstracted={"PRODUCT": out.stem, "range_spacing": o.pixel, "azimuth_spacing": o.pixel,
                                    "orbit_state_vector_file": o.orbit_file or product.abstracted.get(
                                        "orbit_state_vector_file", ""),
                                    "is_terrain_corrected": o.terrain_flag, "DEM": "External DEM"},
                        unit=o.unit, i2m=o.i2m, skip_files=o.skip_files,
                        dtype=">i2" if o.integer else ">f4", envi_type=2 if o.integer else 4)
        if self.after_run is not None:
            self.after_run(out, src, dem)

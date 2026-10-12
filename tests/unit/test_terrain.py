"""Step 3.4 terrain correction: graph, configuration, SNAP checks, input-stage checks, the DEM
mosaic from Step 2C tiles, and runs against a FAKE gpt on a Step 3.3 output made by the real
Step 3.2/3.3 code (fake gpts). No SNAP, no real imagery. Raster tests need rasterio."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bato.config import load_config
from bato.contracts.common import BBox, PolygonGeometry
from bato.contracts.io import load_contract
from bato.contracts.preprocessing import TerrainProvenance
from bato.exceptions import ConfigError, ProhibitedInputError
from bato.preprocessing import radiometric as rad
from bato.preprocessing import snap
from bato.preprocessing import terrain as ter
from bato.preprocessing.dem_mosaic import DEMError, build_mosaic, dem_extent, select_dem_tiles
from bato.preprocessing.dimap import read_dimap
from bato.preprocessing.terrain_validation import lonlat_to_3857, standard_grid_offsets, to_lonlat

from helpers import REPO_CONFIG_DIR
from snap_fakes import TRISHULI_PRE1, DimapSpec, make_project, write_dimap
from speckle_fakes import read_envi, write_band
from terrain_fakes import (
    TC_VALUE_SETS,
    TRISHULI_TILES,
    WKT_4326,
    FakeTerrainSnap,
    TerrainOutputSpec,
    add_dem_tiles,
    elevation,
    published_speckle,
    terrain_help,
)

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")
TEMPLATE = snap.graph_path("s1_terrain.xml").read_text(encoding="utf-8")
# Kuro Siwo configs/grd_preprocessing.xml Terrain-Correction node (read 2026-10-02), minus the DEM
# parameters BATO replaces (demName, externalDEMFile, externalDEMNoDataValue, externalDEMApplyEGM)
# and sourceBands (empty in Kuro Siwo = all bands; explicit here).
KURO_SIWO_TC = {
    "demResamplingMethod": "BILINEAR_INTERPOLATION", "imgResamplingMethod": "BILINEAR_INTERPOLATION",
    "pixelSpacingInMeter": "10.0", "pixelSpacingInDegree": "8.983152841195215E-5", "alignToStandardGrid": "true",
    "standardGridOriginX": "0.0", "standardGridOriginY": "0.0", "nodataValueAtSea": "true", "saveDEM": "false",
    "saveLatLon": "false", "saveIncidenceAngleFromEllipsoid": "false", "saveLocalIncidenceAngle": "false",
    "saveProjectedLocalIncidenceAngle": "false", "saveSelectedSourceBand": "true", "saveLayoverShadowMask": "false",
    "outputComplex": "false", "applyRadiometricNormalization": "false", "saveSigmaNought": "false",
    "saveGammaNought": "false", "saveBetaNought": "false",
    "incidenceAngleForSigma0": "Use projected local incidence angle from DEM",
    "incidenceAngleForGamma0": "Use projected local incidence angle from DEM", "auxFile": "Latest Auxiliary File",
    "externalAuxFile": "",
}
DEM_PARAMS = {"demName", "externalDEMFile", "externalDEMNoDataValue", "externalDEMApplyEGM", "sourceBands",
              "mapProjection"}


def render(**overrides: object) -> str:
    values: dict[str, object] = {"input_file": "C:/x/in.dim", "source_bands": "Sigma0_VV,Sigma0_VH",
                                 "dem_file": "C:/x/dem.tif", "dem_nodata": -32767.0, "dem_apply_egm": True,
                                 "output_file": "C:/x/out.dim"}
    return snap.render_graph(TEMPLATE, {**values, **overrides})


def tc(graph_xml: str) -> dict[str, str]:
    return next(p for _n, o, p in snap.graph_operators(graph_xml) if o == "Terrain-Correction")


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


@pytest.fixture
def published(tmp_path_factory, gpt):
    return published_speckle(tmp_path_factory.mktemp("t"), gpt)


def job_for(p):
    return ter.prepare_terrain_job(p.manifest, TRISHULI_PRE1, p.cfg)


def run(p, gpt, fake, **kw):
    return ter.run_terrain(job_for(p), p.cfg, gpt=gpt, runner=fake.run, popen=fake.popen, **kw)


def folders(p):
    parent = p.cfg.settings.paths.interim / "preprocessing" / p.manifest.aoi_id / TRISHULI_PRE1
    return sorted(x.name for x in parent.iterdir())


def tree_hashes(folder: Path) -> dict[str, str]:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob("*")) if f.is_file()}


# ---- graph, configuration, helpers (no SNAP, no rasterio) -------------------------------------

def test_graph_is_the_kuro_siwo_terrain_node_with_copernicus_dem():
    rendered = render()
    assert [o for _n, o, _p in snap.graph_operators(rendered)] == ["Read", "Terrain-Correction", "Write"]
    params = tc(rendered)
    assert {k: v for k, v in params.items() if k not in DEM_PARAMS} == KURO_SIWO_TC
    assert "Pseudo-Mercator" in params["mapProjection"] and 'AUTHORITY["EPSG","3857"]]' in params["mapProjection"]
    assert params["demName"] == "External DEM" and params["externalDEMFile"] == "C:/x/dem.tif"
    assert params["externalDEMNoDataValue"] == "-32767.0" and params["externalDEMApplyEGM"] == "true"
    assert params["sourceBands"] == "Sigma0_VV,Sigma0_VH"
    for forbidden in ("SRTM", "LinearToFromdB", "Speckle-Filter", "Terrain-Flattening", "BandMaths", "Calibration"):
        assert forbidden not in rendered


def test_graph_values_come_from_config(cfg):
    t = cfg.settings.preprocessing.s1_terrain
    assert (t.graph, t.dem_margin_km, t.dem_apply_egm, t.dem_nodata) == ("s1_terrain.xml", 20.0, True, -32767.0)


@pytest.mark.parametrize("change", [{"dem_nodata": 0.0}, {"dem_margin_km": -1.0}, {"graph": "../x.xml"},
                                    {"dem_apply_egm": "maybe"}, {"map_projection": "EPSG:32645"}])
def test_config_validation(config_copy, change):
    with pytest.raises(ConfigError):
        load_config(config_copy("config.yaml", lambda d: d["preprocessing"]["s1_terrain"].update(change)))


def test_terrain_values_checked_against_installed_snap(tmp_path):
    fake = FakeTerrainSnap()
    report = snap.check_parameter_values(tmp_path / "gpt", render(), ["Terrain-Correction"], 10, fake.run)
    assert set(report["Terrain-Correction"]["checked"]) == set(TC_VALUE_SETS)
    assert "demName" in report["Terrain-Correction"]["not_listed"]
    old = FakeTerrainSnap(value_sets={**TC_VALUE_SETS, "imgResamplingMethod": ("NEAREST_NEIGHBOUR",)})
    with pytest.raises(snap.SnapError, match="imgResamplingMethod='BILINEAR_INTERPOLATION'"):
        snap.check_parameter_values(tmp_path / "gpt", render(), ["Terrain-Correction"], 10, old.run)
    renamed = FakeTerrainSnap(missing_param=("Terrain-Correction", "externalDEMApplyEGM"))
    with pytest.raises(snap.SnapError, match="externalDEMApplyEGM"):
        snap.check_operator_parameters(tmp_path / "gpt", render(), 10, renamed.run)
    assert "Value must be one of 'NEAREST_NEIGHBOUR'" in terrain_help(["imgResamplingMethod"], old.value_sets)


# SNAP 12.0.0 real Trishuli output (2026-10-02): IMAGE_TO_MODEL_TRANSFORM 10,0,0,-10,9451025.0,3320125.0
@pytest.mark.parametrize("x0, y0, aligned", [
    (9451025.0, 3320125.0, True),          # real SNAP output: corner at x5 m, pixel centre on the 10 m grid
    (9451030.0 - 5.0, 3320120.0 + 5.0, True),
    (9451030.0, 3320130.0, False),         # corner on the grid -> centres half a pixel off SNAP's grid
    (9451028.0, 3320125.0, False),         # genuinely misaligned (3 m)
    (9451025.0, 3320125.4, False),
])
def test_standard_grid_follows_snap_pixel_centre_convention(x0, y0, aligned):
    fx, fy = standard_grid_offsets(x0, y0, 10.0, 10.0, 10.0)
    ok = abs(fx - round(fx)) <= 1e-3 and abs(fy - round(fy)) <= 1e-3
    assert ok is aligned
    if (x0, y0) == (9451025.0, 3320125.0):
        assert (fx, fy) == (945103.0, 332012.0)


def test_standard_grid_uses_the_graph_origin():
    assert standard_grid_offsets(1005.0, 2005.0, 10.0, 10.0, 10.0, origin=(0.0, 0.0)) == (101.0, 200.0)
    fx, fy = standard_grid_offsets(1005.0, 2005.0, 10.0, 10.0, 10.0, origin=(3.0, 0.0))
    assert fx == pytest.approx(100.7) and fy == 200.0


def test_mercator_helpers():
    x, y = lonlat_to_3857(np.array([180.0, 0.0]), np.array([0.0, 0.0]))
    assert x[0] == pytest.approx(20037508.342789244) and y[1] == pytest.approx(0.0, abs=1e-6)
    lon, lat = to_lonlat(*[float(v[0]) for v in lonlat_to_3857(np.array([85.3]), np.array([28.1]))])
    assert lon == pytest.approx(85.3) and lat == pytest.approx(28.1)


@pytest.mark.parametrize("spec, problem", [
    (DimapSpec(unit="intensity_db"), "dB"),
    (DimapSpec(map_geocoding=True), "radar geometry"),
    (DimapSpec(terrain_corrected="1"), "radar geometry"),
    (DimapSpec(history=("Read", "Calibration", "Write")), "Speckle-Filter recorded 0 times"),
    (DimapSpec(history=("Read", "Speckle-Filter", "Speckle-Filter", "Write")), "recorded 2 times"),
    (DimapSpec(history=("Read", "Speckle-Filter", "Terrain-Correction", "Write")), "already processed"),
    (DimapSpec(history=("Read", "Speckle-Filter", "LinearToFromdB", "Write")), "already processed"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((4, 5), np.float32)}), "expected exactly"),
])
def test_input_must_be_a_step_3_3_product(tmp_path, spec, problem):
    write_dimap(tmp_path / "in.dim", spec)
    with pytest.raises(rad.PreprocessingError, match=problem):
        ter.check_input_product(read_dimap(tmp_path / "in.dim"), ("VV", "VH"))
    ok = DimapSpec(history=("Read", "Calibration", "Write", "Read", "Speckle-Filter", "Write"))
    write_dimap(tmp_path / "ok.dim", ok)
    ter.check_input_product(read_dimap(tmp_path / "ok.dim"), ("VV", "VH"))


def test_missing_step_3_3_output_and_reference_guard(tmp_path_factory, config_copy):
    root = tmp_path_factory.mktemp("m")
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    p = make_project(load_config(root / "config"))
    with pytest.raises(rad.PreprocessingError, match="no published Step 3.3 output.*speckle_s1.py"):
        job_for(p)
    with pytest.raises(rad.PreprocessingError, match="dry-run"):
        ter.prepare_terrain_job(p.manifest.model_copy(update={"dry_run": True}), TRISHULI_PRE1, p.cfg)
    cfg = load_config(config_copy("config.yaml", lambda d: d["paths"].update(interim="data/reference/interim")))
    q = make_project(cfg)
    with pytest.raises(ProhibitedInputError):
        job_for(q)


# ---- DEM mosaic from Step 2C tiles (rasterio) ------------------------------------------------------

def _tile_project(tmp_path_factory, **kw):
    root = tmp_path_factory.mktemp("d")
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    p = make_project(load_config(root / "config"))
    add_dem_tiles(p, **kw)
    return p


FOOTPRINT = PolygonGeometry(coordinates=(((84.97, 28.46), (85.62, 28.46), (85.62, 27.79), (84.97, 27.79),
                                          (84.97, 28.46)),))


@needs_rasterio
@pytest.mark.parametrize("kind", ["zip", "tar", "tif"])
def test_mosaic_copies_tile_values_onto_one_grid(tmp_path_factory, kind):
    import rasterio
    p = _tile_project(tmp_path_factory, kinds=dict.fromkeys(TRISHULI_TILES, kind))
    extent = dem_extent(FOOTPRINT, 20.0)
    sources, sea = select_dem_tiles(p.manifest, extent, p.cfg)
    assert [s.tile_id for s in sources] == list(TRISHULI_TILES) and sea == []
    out = p.root / "work" / "dem.tif"
    out.parent.mkdir()
    rec = build_mosaic(sources, extent, FOOTPRINT, out, nodata=-32767.0, margin_km=20.0, dataset="COP-DEM_TEST",
                       sea_tiles=sea, root=p.root)
    assert rec.sha256 == hashlib.sha256(out.read_bytes()).hexdigest() and len(rec.tiles) == 4
    assert rec.nodata_pixels_in_footprint == 0 and rec.footprint_pixels > 0
    assert rec.bounds[0] <= extent.min_lon and rec.bounds[2] >= extent.max_lon
    with rasterio.open(out) as ds:
        data = ds.read(1)
        t = ds.transform
        assert ds.crs.to_epsg() == 4326 and ds.nodata == -32767.0
    # every mosaic pixel equals the tile pixel at the same location (values unchanged)
    for tile_id in TRISHULI_TILES:
        lat0, lon0 = int(tile_id[1:3]), int(tile_id[4:7])
        ref = elevation(tile_id)
        col0, row0 = round((lon0 - t.c) / t.a), round((t.f - (lat0 + 1)) / -t.e)
        r0, c0 = max(0, row0), max(0, col0)
        r1, c1 = min(data.shape[0], row0 + 100), min(data.shape[1], col0 + 100)
        assert np.array_equal(data[r0:r1, c0:c1], ref[r0 - row0:r1 - row0, c0 - col0:c1 - col0])
    assert not (p.root / "work" / "tiles").exists()                       # extracted tiles cleaned up


@needs_rasterio
def test_sea_cells_become_nodata_and_are_reported(tmp_path_factory):
    p = _tile_project(tmp_path_factory, tiles=("N27E084", "N27E085", "N28E084"), sea=("N28E085",))
    extent = dem_extent(FOOTPRINT, 20.0)
    sources, sea = select_dem_tiles(p.manifest, extent, p.cfg)
    assert sea == ["N28E085"]
    out = p.root / "dem.tif"
    rec = build_mosaic(sources, extent, FOOTPRINT, out, nodata=-32767.0, margin_km=20.0, dataset="X", sea_tiles=sea,
                       root=p.root)
    assert rec.missing_sea_tiles == ("N28E085",) and rec.nodata_pixels_in_footprint > 0


@needs_rasterio
def test_dem_tile_problems_are_refused(tmp_path_factory):
    extent = dem_extent(FOOTPRINT, 20.0)
    p = _tile_project(tmp_path_factory, not_downloaded=("N27E085",))
    with pytest.raises(DEMError, match="N27E085.*not downloaded"):
        select_dem_tiles(p.manifest, extent, p.cfg)
    p = _tile_project(tmp_path_factory, tiles=("N27E084", "N27E085", "N28E084"))
    with pytest.raises(DEMError, match="N28E085 is needed but was not planned"):
        select_dem_tiles(p.manifest, extent, p.cfg)
    p = _tile_project(tmp_path_factory)
    archive = p.root / p.manifest.files[-1].local_path
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(DEMError, match="SHA-256 differs"):
        select_dem_tiles(p.manifest, extent, p.cfg)
    with pytest.raises(DEMError, match="no Copernicus DEM plan"):
        select_dem_tiles(p.manifest.model_copy(update={"dem_plan": None}), extent, p.cfg)


@needs_rasterio
@pytest.mark.parametrize("kw, problem", [({"res": {"N28E085": 0.02}}, "resolution"),
                                          ({"shift": {"N28E085": 0.0042}}, "not on the common grid")])
def test_inconsistent_tiles_are_refused(tmp_path_factory, kw, problem):
    p = _tile_project(tmp_path_factory, **kw)
    extent = dem_extent(FOOTPRINT, 20.0)
    sources, sea = select_dem_tiles(p.manifest, extent, p.cfg)
    with pytest.raises(DEMError, match=problem):
        build_mosaic(sources, extent, FOOTPRINT, p.root / "dem.tif", nodata=-32767.0, margin_km=20.0,
                     dataset="X", sea_tiles=sea, root=p.root)


def test_dem_extent_grows_footprint():
    ext = dem_extent(FOOTPRINT, 20.0)
    assert isinstance(ext, BBox) and ext.min_lat == pytest.approx(27.79 - 20 / 111.32)


# ---- runs with a fake gpt (need rasterio) ------------------------------------------------------

@needs_rasterio
def test_job_reads_the_published_speckle_artifact_and_dem(published):
    job = job_for(published)
    assert job.input_dir.name == "speckle" and job.final_dir.name == "terrain"
    assert job.output_stem == f"{TRISHULI_PRE1}_sigma0_spk_tc" and job.dem_name == f"{TRISHULI_PRE1}_copdem30.tif"
    assert [s.tile_id for s in job.dem_sources] == list(TRISHULI_TILES) and job.dem_sea_tiles == ()
    assert job.radiometric_provenance.aoi.aoi_polygon.exterior[0] == (85.30, 28.10)


@needs_rasterio
def test_broken_provenance_chain_or_settings_are_refused(published):
    job = job_for(published)
    other = published.cfg.model_copy(update={"settings": published.cfg.settings.model_copy(update={
        "preprocessing": published.cfg.settings.preprocessing.model_copy(update={
            "s1_speckle": published.cfg.settings.preprocessing.s1_speckle.model_copy(update={"sigma": "0.8"})})})})
    with pytest.raises(rad.PreprocessingError, match="different speckle settings"):
        ter.prepare_terrain_job(published.manifest, TRISHULI_PRE1, other)
    data = json.loads(job.radiometric_provenance_path.read_text())
    data["bato_version"] = "tampered"
    job.radiometric_provenance_path.write_text(json.dumps(data))
    with pytest.raises(rad.PreprocessingError, match="provenance chain broken"):
        job_for(published)


@needs_rasterio
def test_db_input_is_refused(published):
    dim = job_for(published).input_dim
    dim.write_text(dim.read_text(encoding="ISO-8859-1").replace("<PHYSICAL_UNIT>intensity</PHYSICAL_UNIT>",
                                                                "<PHYSICAL_UNIT>intensity_db</PHYSICAL_UNIT>"),
                   encoding="ISO-8859-1")
    with pytest.raises(rad.PreprocessingError, match="dB"):
        job_for(published)


@needs_rasterio
def test_successful_run_publishes_validated_output_with_provenance(published, gpt):
    stage = published.cfg.settings.paths.interim / "preprocessing" / published.manifest.aoi_id / TRISHULI_PRE1
    previous = {name: tree_hashes(stage / name) for name in ("radiometric", "speckle")}
    raw_before = tree_hashes(published.cfg.settings.paths.raw)
    fake = FakeTerrainSnap()
    result = run(published, gpt, fake)
    assert folders(published) == ["radiometric", "speckle", "terrain"]
    out = result.output_dir
    assert {f.name for f in out.iterdir()} == {f"{TRISHULI_PRE1}_sigma0_spk_tc.dim",
                                               f"{TRISHULI_PRE1}_sigma0_spk_tc.data", "dem", "graph_resolved.xml",
                                               "gpt.log", "preprocessing_provenance.json"}
    prov = load_contract(TerrainProvenance, result.provenance_path)
    assert prov == result.provenance and prov.status == "succeeded" and prov.validation.passed
    assert {k: v for k, v in prov.terrain_correction.items() if k not in DEM_PARAMS} == KURO_SIWO_TC
    assert prov.terrain_correction["externalDEMFile"].endswith(f"/dem/{TRISHULI_PRE1}_copdem30.tif")
    # DEM: the mosaic SNAP used is the one recorded and published
    dem_path = published.root / prov.dem.path
    assert dem_path.is_file() and prov.dem.sha256 == hashlib.sha256(dem_path.read_bytes()).hexdigest()
    assert fake.dem_seen == [prov.dem.sha256] and [t.tile_id for t in prov.dem.tiles] == list(TRISHULI_TILES)
    for t in prov.dem.tiles:
        assert hashlib.sha256((published.root / t.archive).read_bytes()).hexdigest() == t.archive_sha256
    # hashes, chain of custody, nothing upstream touched
    for rel, digest in prov.output_sha256.items():
        assert hashlib.sha256((published.root / rel).read_bytes()).hexdigest() == digest and "/terrain/" in rel
    assert prov.input.files_sha256 == job_for(published).input_hashes
    assert prov.input.radiometric_provenance_sha256 == hashlib.sha256(
        (published.root / prov.input.radiometric_provenance_path).read_bytes()).hexdigest()
    assert {name: tree_hashes(stage / name) for name in ("radiometric", "speckle")} == previous
    assert tree_hashes(published.cfg.settings.paths.raw) == raw_before
    assert prov.config_fingerprint == published.cfg.fingerprint() and prov.snap.snap_version == "12.0.0"
    assert prov.aoi_polygon.exterior[0] == (85.30, 28.10)
    for rel in (prov.output.dim, prov.snap.log_path, prov.graph.resolved_path, prov.dem.path,
                *(b.file for b in prov.validation.bands_after)):
        assert (published.root / rel).exists() and ".tmp-" not in rel
    assert all(k.get("shell") is False for k in fake.run_kwargs)


@needs_rasterio
def test_validation_confirms_linear_map_geometry(published, gpt):
    v = run(published, gpt, FakeTerrainSnap()).provenance.validation
    assert {c.name for c in v.checks} >= {
        "output_exists", "sigma0_bands", "linear_scale", "map_geometry", "pixel_size", "standard_grid",
        "processing_history", "terrain_flag", "orbit_preserved", "readable_VV", "float_VH", "valid_pixels_VV",
        "same_grid", "crs_epsg_3857", "aoi_coverage_VV", "aoi_coverage_VH"}
    assert v.geometry == "map (EPSG:3857)" and v.pixel_size_m == (10.0, 10.0)
    assert v.aoi_valid_fraction == {"Sigma0_VV": 1.0, "Sigma0_VH": 1.0}
    assert all(b.unit == "intensity" and b.dtype == "float32" for b in v.bands_after)
    assert [b.band for b in v.bands_before] == [b.band for b in v.bands_after] == ["Sigma0_VV", "Sigma0_VH"]
    assert v.bounds_3857[0] % 10 == 5 and v.footprint is not None        # SNAP: centres on the 10 m grid


@needs_rasterio
def test_runs_are_deterministic(published, gpt):
    first = run(published, gpt, FakeTerrainSnap())
    second = run(published, gpt, FakeTerrainSnap(), overwrite=True)
    a, b = first.provenance, second.provenance
    assert a.dem.sha256 == b.dem.sha256 and a.graph.resolved_sha256 != "" and a.terrain_correction.keys() == \
        b.terrain_correction.keys()
    strip = {k.split("/terrain")[-1]: v for k, v in a.output_sha256.items() if not k.endswith(".dim")}
    assert strip == {k.split("/terrain")[-1]: v for k, v in b.output_sha256.items() if not k.endswith(".dim")}


@pytest.mark.parametrize("out, failed", [
    (TerrainOutputSpec(unit="intensity_db"), "linear_scale"),
    (TerrainOutputSpec(extra_band=True), "sigma0_bands"),
    (TerrainOutputSpec(shrink_vh=True), "same_grid"),
    (TerrainOutputSpec(i2m=False), "map_geometry"),
    (TerrainOutputSpec(wkt=WKT_4326), "crs_epsg_3857"),
    (TerrainOutputSpec(wkt=None), "crs_epsg_3857"),
    (TerrainOutputSpec(pixel=20.0), "pixel_size"),
    (TerrainOutputSpec(x_offset_m=3.0), "standard_grid"),
    (TerrainOutputSpec(x_offset_m=5.0), "standard_grid"),     # corner (not centre) on the grid: half-pixel shift
    (TerrainOutputSpec(extra_history=("Read", "Terrain-Correction", "LinearToFromdB", "Write")), "processing_history"),
    (TerrainOutputSpec(extra_history=("Read", "Terrain-Correction", "Terrain-Correction", "Write")),
     "processing_history"),
    (TerrainOutputSpec(extra_history=("Read", "Write")), "processing_history"),
    (TerrainOutputSpec(terrain_flag="0"), "terrain_flag"),
    (TerrainOutputSpec(orbit_file="Sentinel Restituted X"), "orbit_preserved"),
    (TerrainOutputSpec(integer=True), "float_VV"),
    (TerrainOutputSpec(skip_files=("Sigma0_VH",)), "output_exists"),
    (TerrainOutputSpec(extent_shift_deg=0.5), "aoi_coverage_VV"),
    (TerrainOutputSpec(all_nodata=True), "valid_pixels_VV"),
])
@needs_rasterio
def test_invalid_outputs_fail_and_are_never_published(published, gpt, out, failed):
    with pytest.raises(ter.TerrainRunFailed) as info:
        run(published, gpt, FakeTerrainSnap(out=out))
    prov = info.value.provenance
    assert prov.validation is not None, f"the run failed before validation: {prov.error}"
    assert prov.status == "failed" and failed in {c.name for c in prov.validation.failed_checks()}
    assert prov.output is None and prov.output_sha256 == {}
    assert info.value.output_dir.name.startswith("terrain.failed-")
    assert "terrain" not in folders(published)


@needs_rasterio
def test_input_or_dem_changed_during_the_run_fails(published, gpt):
    def touch_input(_out, src, _dem):
        img = src.with_suffix(".data") / "Sigma0_VV.img"
        write_band(img, read_envi(img) * 2)

    with pytest.raises(ter.TerrainRunFailed, match="Step 3.3 input changed during the run"):
        run(published, gpt, FakeTerrainSnap(after_run=touch_input))


@needs_rasterio
def test_dem_changed_during_the_run_fails(published, gpt):
    def touch_dem(_out, _src, dem):
        dem.write_bytes(dem.read_bytes() + b"x")

    with pytest.raises(ter.TerrainRunFailed, match="external DEM changed"):
        run(published, gpt, FakeTerrainSnap(after_run=touch_dem))


@needs_rasterio
def test_gpt_failure_and_timeout_keep_a_failed_folder(published, gpt):
    with pytest.raises(ter.TerrainRunFailed, match="exit code 1") as info:
        run(published, gpt, FakeTerrainSnap(returncode=1, write_output=False))
    assert info.value.provenance.validation is None and info.value.provenance.dem is not None
    with pytest.raises(ter.TerrainRunFailed, match="did not finish"):
        run(published, gpt, FakeTerrainSnap(timeout=True, write_output=False))
    assert "terrain" not in folders(published)


@needs_rasterio
def test_dem_build_failure_keeps_a_failed_folder(published, gpt, monkeypatch):
    def broken(*a, **k):
        raise DEMError("TEST: DEM tiles not on one grid")
    monkeypatch.setattr(ter, "build_mosaic", broken)
    fake = FakeTerrainSnap()
    with pytest.raises(ter.TerrainRunFailed, match="not on one grid"):
        run(published, gpt, fake)
    assert not any(c[1].endswith(".xml") for c in fake.calls if len(c) == 3 and c[2] == "-e")   # gpt never ran


@needs_rasterio
def test_snap_discrepancies_stop_before_anything_is_written(published, gpt):
    before = folders(published)
    with pytest.raises(snap.SnapError, match="demResamplingMethod='BILINEAR_INTERPOLATION'"):
        run(published, gpt,
            FakeTerrainSnap(value_sets={**TC_VALUE_SETS, "demResamplingMethod": ("NEAREST_NEIGHBOUR",)}))
    with pytest.raises(snap.SnapError, match="externalDEMFile"):
        run(published, gpt, FakeTerrainSnap(missing_param=("Terrain-Correction", "externalDEMFile")))
    assert folders(published) == before


@needs_rasterio
def test_existing_output_is_refused_or_superseded_never_deleted(published, gpt):
    first = run(published, gpt, FakeTerrainSnap())
    fake = FakeTerrainSnap()
    with pytest.raises(rad.PreprocessingError, match="already exists"):
        run(published, gpt, fake)
    assert fake.calls == []
    (first.output_dir / "marker.txt").write_text("old")
    second = run(published, gpt, FakeTerrainSnap(), overwrite=True)
    superseded = [n for n in folders(published) if n.startswith("terrain.superseded-")]
    assert len(superseded) == 1 and second.output_dir.name == "terrain"
    assert (second.output_dir.parent / superseded[0] / "marker.txt").read_text() == "old"

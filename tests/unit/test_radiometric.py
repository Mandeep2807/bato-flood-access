"""Step 3.2 radiometric preprocessing: job preparation on REAL Trishuli/Valencia metadata
with synthetic SAFE zips, and runs against a FAKE gpt that writes a tiny DIMAP product.
No SNAP, no real imagery, no network. Validation tests need rasterio."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from bato.config import load_config
from bato.contracts.common import Polarization
from bato.contracts.io import load_contract
from bato.contracts.preprocessing import PreprocessingProvenance
from bato.data_rules import assert_interim_output_dir
from bato.exceptions import ProhibitedInputError
from bato.preprocessing import radiometric as rad
from bato.preprocessing import snap
from bato.preprocessing.sigma0_validation import ValidationSetupError

from helpers import REPO_CONFIG_DIR
from snap_fakes import TRISHULI_POST, TRISHULI_PRE1, DimapSpec, FakeSnap, make_project, write_envi


@pytest.fixture
def project(tmp_path_factory):
    """Project under a SHORT root: SNAP-like outputs nest ~175 characters below it, and pytest's
    per-test folder names (up to 30 characters + a counter) would push them past Windows'
    260-character path limit on some machines."""
    root = tmp_path_factory.mktemp("p")
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    return make_project(load_config(root / "config"))


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "esa-snap" / "bin" / "gpt.exe"
    path.parent.mkdir(parents=True)
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


def job_for(p, pass_id=TRISHULI_PRE1, **kw):
    return rad.prepare_job(p.manifest, pass_id, p.cfg, **kw)


def run(p, gpt, fake, **kw):
    return rad.run_radiometric(job_for(p), p.cfg, gpt=gpt, runner=fake.run, popen=fake.popen, **kw)


def stage_folders(p):
    parent = p.cfg.settings.paths.interim / "preprocessing" / p.manifest.aoi_id / TRISHULI_PRE1
    return sorted(x.name for x in parent.iterdir()) if parent.exists() else []


# ---- job preparation (no rasterio, no SNAP) ---------------------------------------------------

def test_job_uses_manifest_product_and_catalogue_aoi(project):
    job = job_for(project)
    assert job.scene.product_name.startswith("S1D_IW_GRDH_1SDV_20260629T122138")
    assert job.input_path.name == job.scene.product_name + ".zip" and job.input_path.is_file()
    assert job.catalogue_path.name == "trishuli_s1.json"           # found by digest, not by name
    assert job.catalogue_digest == project.manifest.plan.catalogue_digest
    assert job.safe.orbit_direction.value == "ascending" and job.scene.relative_orbit == 85
    b, c = job.aoi_bbox, job.crop_bbox
    assert all(abs(x - y) < 1e-9 for x, y in zip(
        (min(p[0] for p in job.aoi_polygon.exterior), min(p[1] for p in job.aoi_polygon.exterior)),
        (b.min_lon, b.min_lat), strict=True))
    km_lat = (b.min_lat - c.min_lat) * 111.32
    km_lon = (b.min_lon - c.min_lon) * 111.32 * math.cos(math.radians((b.min_lat + b.max_lat) / 2))
    assert km_lat == pytest.approx(20.0, rel=1e-6) and km_lon == pytest.approx(20.0, rel=1e-6)
    assert job.geo_region_wkt.startswith("POLYGON ((") and f"{c.min_lon:.6f} {c.min_lat:.6f}" in job.geo_region_wkt
    assert job.final_dir == (project.cfg.settings.paths.interim / "preprocessing" / project.manifest.aoi_id
                             / TRISHULI_PRE1 / "radiometric").resolve()
    assert job.input_scene_aoi_coverage is not None and 0 < job.input_scene_aoi_coverage <= 1


def test_aoi_comes_from_config_buffer_not_hardcoded(config_copy):
    cfg = load_config(config_copy("config.yaml", lambda d: d["preprocessing"].update(crop_buffer_m=5000.0)))
    job = job_for(make_project(cfg))
    assert (job.aoi_bbox.min_lat - job.crop_bbox.min_lat) * 111.32 == pytest.approx(5.0, rel=1e-6)


def test_graph_values_insert_aoi_paths_and_config(project):
    job = job_for(project)
    out = job.final_dir.with_name("radiometric.tmp-X") / f"{job.output_stem}.dim"
    values = rad.graph_values(job, project.cfg, out)
    rendered = snap.render_graph(snap.graph_path("s1_radiometric.xml").read_text(), values)
    params = {op: p for _n, op, p in snap.graph_operators(rendered)}
    assert params["Subset"]["geoRegion"] == job.geo_region_wkt
    assert params["Read"]["file"] == job.input_path.as_posix() and "\\" not in params["Read"]["file"]
    assert params["Write"]["file"].endswith(f"radiometric.tmp-X/{TRISHULI_PRE1}_sigma0.dim")
    assert params["ThermalNoiseRemoval"]["selectedPolarisations"] == "VV,VH"


def test_incomplete_unknown_and_dry_run_passes_are_refused(project):
    with pytest.raises(rad.PreprocessingError, match="incomplete"):
        job_for(project, TRISHULI_POST)
    with pytest.raises(rad.PreprocessingError, match="not in the manifest"):
        job_for(project, "S1A-D019-20260101T000000")
    dry = project.manifest.model_copy(update={"dry_run": True})
    with pytest.raises(rad.PreprocessingError, match="dry-run"):
        rad.prepare_job(dry, TRISHULI_PRE1, project.cfg)


def test_multi_slice_pass_is_refused_clearly(config_copy):
    cfg = load_config(config_copy())
    first = make_project(cfg, name="valencia", track="DESCENDING/8", present_passes=())
    multi = next(p for p in first.manifest.plan.passes if p.multi_slice)
    p = make_project(cfg, name="valencia", track="DESCENDING/8", present_passes=(multi.pass_id,))
    with pytest.raises(rad.PreprocessingError, match="slice assembly is not implemented"):
        rad.prepare_job(p.manifest, multi.pass_id, cfg)


def test_changed_input_file_is_refused(project):
    job = job_for(project)
    with job.input_path.open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(rad.PreprocessingError, match="file changed after download"):
        job_for(project)


def test_catalogue_must_match_the_manifest(project, config_copy):
    project.catalogue_path.unlink()
    with pytest.raises(rad.PreprocessingError, match="--catalogue"):
        job_for(project)
    other = Path(__file__).resolve().parents[1] / "fixtures" / "real_catalogues" / "valencia_s1.json"
    with pytest.raises(rad.PreprocessingError, match="digest differs"):
        job_for(project, catalogue_path=other)


def test_reference_folders_are_refused_as_inputs_and_outputs(project):
    paths = project.cfg.settings.paths
    ref_copy = paths.reference / "trishuli_s1.json"
    ref_copy.parent.mkdir(parents=True, exist_ok=True)
    ref_copy.write_bytes(project.catalogue_path.read_bytes())
    with pytest.raises(ProhibitedInputError):
        job_for(project, catalogue_path=ref_copy)
    with pytest.raises(ProhibitedInputError):
        assert_interim_output_dir(paths.reference / "preprocessing", paths)
    with pytest.raises(ProhibitedInputError, match="inside the interim folder"):
        assert_interim_output_dir(paths.processed / "x", paths)
    with pytest.raises(rad.PreprocessingError, match="invalid path component"):
        rad.stage_dir(project.cfg, "../x", TRISHULI_PRE1)


# ---- runs with a fake gpt (validation reads with rasterio) ----------------------------------

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")


@needs_rasterio
def test_successful_run_publishes_validated_output_with_provenance(project, gpt):
    fake = FakeSnap()
    result = run(project, gpt, fake)
    assert stage_folders(project) == ["radiometric"]                    # no temp/failed folders left
    out = result.output_dir
    assert {f.name for f in out.iterdir()} == {f"{TRISHULI_PRE1}_sigma0.dim", f"{TRISHULI_PRE1}_sigma0.data",
                                               "graph_resolved.xml", "gpt.log", "preprocessing_provenance.json"}
    prov = load_contract(PreprocessingProvenance, result.provenance_path)
    assert prov == result.provenance and prov.status == "succeeded" and prov.validation.passed
    assert prov.graph.resolved_sha256 == snap.sha256_text((out / "graph_resolved.xml").read_text(encoding="utf-8"))
    assert prov.graph.template_sha256 == snap.sha256_text(snap.graph_path("s1_radiometric.xml").read_text())
    assert prov.graph.operator_parameters_checked
    ops = {o.operator: o.parameters for o in prov.graph.operators}
    assert ops["Calibration"]["outputImageScaleInDb"] == "false"
    assert ops["Subset"]["geoRegion"] == prov.aoi.geo_region_wkt
    assert prov.snap.snap_version == "12.0.0" and prov.snap.return_code == 0 and prov.snap.gpt_source == "config"
    assert prov.input.sha256 == json.loads(Path(project.root / prov.input.download_provenance).read_text())["sha256"]
    assert prov.input.acquisition_start.isoformat().startswith("2026-06-29T12:21")
    assert prov.input.orbit_direction.value == "ascending" and prov.input.relative_orbit == 85
    assert prov.config_fingerprint == project.cfg.fingerprint() and prov.aoi.crop_buffer_m == 20000.0
    for rel in (prov.output.dim, prov.output.data_dir, *prov.output.band_files.values(), prov.snap.log_path,
                prov.graph.resolved_path, *(b.file for b in prov.validation.bands)):
        assert (project.root / rel).exists() and ".tmp-" not in rel
    assert all(k.get("shell") is False for k in fake.run_kwargs)
    text = result.provenance_path.read_text()
    assert "password" not in text.lower() and "token" not in text.lower()


@needs_rasterio
def test_validation_reports_linear_statistics_without_thresholds(project, gpt):
    v = run(project, gpt, FakeSnap()).provenance.validation
    assert {c.name for c in v.checks} >= {"output_exists", "sigma0_bands", "linear_scale", "radar_geometry",
                                           "no_downstream_operators", "precise_orbit", "pixel_spacing", "same_grid",
                                           "geolocation", "aoi_coverage", "readable_VV", "float_VH"}
    assert v.geometry.startswith("radar") and v.rasterio_crs is None
    assert v.rasterio_transform == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert v.width == 60 and v.height == 40 and v.range_spacing_m == 10.0 and v.aoi_coverage == 1.0
    assert "POEORB" in v.orbit_file
    vv = next(b for b in v.bands if b.polarization is Polarization.VV)
    assert vv.dtype == "float32" and vv.unit == "intensity" and vv.total_pixels == 2400
    assert vv.nonfinite_pixels == 1 and vv.zero_pixels == vv.nodata_pixels == 3 * 40
    assert vv.valid_pixels == 2400 - 1 - 120 and 0 < vv.min <= vv.median <= vv.max
    assert list(vv.quantiles) == ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]


@pytest.mark.parametrize("spec, failed", [
    (DimapSpec(unit="intensity_db"), "linear_scale"),
    (DimapSpec(log10_scaled=True), "linear_scale"),
    (DimapSpec(map_geocoding=True), "radar_geometry"),
    (DimapSpec(history=("Read", "Calibration", "Terrain-Correction", "Write")), "no_downstream_operators"),
    (DimapSpec(orbit_file="Sentinel Restituted S1D_OPER_AUX_RESORB_OPOD_x.EOF"), "precise_orbit"),
    (DimapSpec(range_spacing=40.0), "pixel_spacing"),
    (DimapSpec(corners=((84.0, 26.0), (84.2, 26.0), (84.2, 25.8), (84.0, 25.8))), "aoi_coverage"),
    (DimapSpec(bands={"Sigma0_VV": np.zeros((40, 60), np.float32), "Sigma0_VH": np.ones((40, 60), np.float32)}),
     "valid_pixels_VV"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((40, 60), np.float32)}), "sigma0_bands"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((40, 60), np.float32), "Sigma0_VH": np.ones((40, 60), np.float32),
                      "Beta0_VV": np.ones((40, 60), np.float32)}), "sigma0_bands"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((40, 60), np.float32), "Sigma0_VH": np.ones((40, 61), np.float32)}),
     "same_grid"),
])
@needs_rasterio
def test_invalid_outputs_fail_and_are_never_published(project, gpt, spec, failed):
    fake = FakeSnap(spec=spec)
    with pytest.raises(rad.PreprocessingRunFailed) as info:
        run(project, gpt, fake)
    prov = info.value.provenance
    assert prov.validation is not None, f"the run failed before validation: {prov.error}"
    assert prov.status == "failed" and failed in {c.name for c in prov.validation.failed_checks()}
    assert info.value.output_dir.name.startswith("radiometric.failed-")
    assert stage_folders(project) == [info.value.output_dir.name]          # no final 'radiometric' folder
    assert load_contract(PreprocessingProvenance, info.value.output_dir / rad.PROVENANCE_FILE).status == "failed"


@needs_rasterio
def test_integer_band_is_not_float(project, gpt):
    class IntegerVV(FakeSnap):
        def produce_output(self, graph_xml: str) -> None:      # SNAP "writes", then VV is replaced by int16
            super().produce_output(graph_xml)
            out = graph_xml.split("<file>")[-1].split("</file>")[0]
            img = Path(out).with_suffix(".data") / "Sigma0_VV.img"
            hdr = img.with_suffix(".hdr")
            hdr.write_text(hdr.read_text().replace("data type = 4", "data type = 2"))
            np.ones((40, 60), ">i2").tofile(img)

    with pytest.raises(rad.PreprocessingRunFailed, match="float_VV"):
        run(project, gpt, IntegerVV())


@needs_rasterio
def test_incomplete_output_fails(project, gpt):
    with pytest.raises(rad.PreprocessingRunFailed, match="output_exists"):
        run(project, gpt, FakeSnap(spec=DimapSpec(skip_files=("Sigma0_VH",))))
    with pytest.raises(rad.PreprocessingRunFailed, match="output_exists"):
        run(project, gpt, FakeSnap(write_output=False))
    assert "radiometric" not in stage_folders(project)


@needs_rasterio
def test_gpt_failure_and_timeout_keep_a_failed_folder(project, gpt):
    with pytest.raises(rad.PreprocessingRunFailed, match="exit code 1") as info:
        run(project, gpt, FakeSnap(returncode=1, write_output=False))
    prov = info.value.provenance
    assert prov.validation is None and prov.output is None and prov.snap.return_code is None
    assert "No valid orbit file" in (info.value.output_dir / "gpt.log").read_text()
    with pytest.raises(rad.PreprocessingRunFailed, match="did not finish"):
        run(project, gpt, FakeSnap(timeout=True, write_output=False))
    assert "radiometric" not in stage_folders(project)


@needs_rasterio
def test_existing_output_is_refused_or_superseded_never_deleted(project, gpt):
    first = run(project, gpt, FakeSnap())
    fake = FakeSnap()
    with pytest.raises(rad.PreprocessingError, match="already exists"):
        run(project, gpt, fake)
    assert fake.calls == []                                                  # refused before SNAP
    marker = first.output_dir / "marker.txt"
    marker.write_text("old")
    second = run(project, gpt, FakeSnap(), overwrite=True)
    superseded = [n for n in stage_folders(project) if n.startswith("radiometric.superseded-")]
    assert len(superseded) == 1 and second.output_dir.name == "radiometric"
    assert (second.output_dir.parent / superseded[0] / "marker.txt").read_text() == "old"


@needs_rasterio
def test_preflight_failures_write_nothing(project, gpt, monkeypatch):
    with pytest.raises(snap.SnapError, match="trimThreshold"):
        run(project, gpt, FakeSnap(missing_param=("Remove-GRD-Border-Noise", "trimThreshold")))
    assert stage_folders(project) == []
    monkeypatch.setitem(sys.modules, "rasterio", None)
    fake = FakeSnap()
    with pytest.raises(ValidationSetupError, match="rasterio is required"):
        run(project, gpt, fake)
    assert fake.calls == [] and stage_folders(project) == []


def test_windows_path_length_is_checked_before_snap(tmp_path):
    stem = f"{TRISHULI_PRE1}_sigma0"
    deep = tmp_path / ("x" * 120) / "radiometric.tmp-20261002T120000Z"
    with pytest.raises(rad.PreprocessingError, match="Windows limit 260"):
        rad.check_output_path_length(deep, stem, system="Windows", long_paths_enabled=False)
    rad.check_output_path_length(deep, stem, system="Windows", long_paths_enabled=True)   # long paths on
    rad.check_output_path_length(deep, stem, system="Linux")                              # no such limit
    rad.check_output_path_length(Path("C:/bato/data/x"), stem, system="Windows", long_paths_enabled=False)


@needs_rasterio
def test_write_failure_during_the_run_is_not_reported_as_validation(project, gpt, monkeypatch):
    import snap_fakes

    def fail(img, array, band_name):
        raise FileNotFoundError(2, "No such file or directory", str(img))
    monkeypatch.setattr(snap_fakes, "write_envi", fail)
    with pytest.raises(FileNotFoundError):
        run(project, gpt, FakeSnap())
    assert "radiometric" not in stage_folders(project)                     # never published


@needs_rasterio
def test_band_statistics_edge_and_interior_zeros(tmp_path):
    from bato.preprocessing.sigma0_validation import band_statistics

    a = np.full((20, 30), 0.25, np.float32)
    a[0, :] = 0.0             # 30 edge zeros
    a[10, 15] = 0.0           # 1 interior zero
    a[12, 12] = np.inf
    a[13, 13] = -0.5          # negative: reported, not clipped
    write_envi(tmp_path / "b.img", a, "Sigma0_VV")
    st, dtype, crs, _ = band_statistics(tmp_path / "b.img", band="Sigma0_VV", polarization=Polarization.VV,
                                        unit="intensity", nodata=0.0, edge_width_px=2, root=tmp_path)
    assert (st.zero_pixels, st.zero_pixels_edge, st.zero_pixels_interior) == (31, 30, 1)
    assert st.nonfinite_pixels == 1 and st.negative_pixels == 1 and st.min == -0.5 and st.max == 0.25
    assert st.valid_pixels == 600 - 1 - 31 and st.mean == pytest.approx((0.25 * (st.valid_pixels - 1) - 0.5)
                                                                        / st.valid_pixels)
    assert st.quantiles["p50"] == 0.25 and st.quantile_sample_stride == 1 and crs is None

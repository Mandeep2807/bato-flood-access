"""Step 3.3 Lee Sigma speckle filter: graph, configuration, SNAP value-set checks, input checks,
and runs against a FAKE gpt on a Step 3.2 output produced by the (fake-gpt) Step 3.2 pipeline.
No SNAP, no real imagery. Tests that read rasters need rasterio."""

from __future__ import annotations

import importlib.util
import json
import shutil

import numpy as np
import pytest

from bato.config import load_config
from bato.contracts.io import load_contract
from bato.contracts.preprocessing import PreprocessingProvenance, SpeckleProvenance
from bato.exceptions import ConfigError, ProhibitedInputError
from bato.preprocessing import radiometric as rad
from bato.preprocessing import snap
from bato.preprocessing import speckle as spk
from bato.preprocessing.dimap import read_dimap

from helpers import REPO_CONFIG_DIR
from snap_fakes import TRISHULI_PRE1, DimapSpec, FakeSnap, make_project, write_dimap
from speckle_fakes import (
    SPECKLE_VALUE_SETS,
    FakeSpeckleSnap,
    add_band,
    append_history,
    change_orbit,
    change_spacing,
    drop_vh_file,
    int_vv,
    map_projection,
    modify_input,
    move_geolocation,
    nan_corruption,
    set_unit_db,
    shrink_vh,
    speckle_help,
)

needs_rasterio = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                                    reason="rasterio not installed (pip install -e .[preprocessing])")
KURO_SIWO_NODE = {"filter": "Lee Sigma", "filterSizeX": "3", "filterSizeY": "3", "dampingFactor": "2",
                  "estimateENL": "true", "enl": "1.0", "numLooksStr": "1", "windowSize": "7x7",
                  "targetWindowSizeStr": "3x3", "sigmaStr": "0.9", "anSize": "50"}
TEMPLATE = snap.graph_path("s1_speckle.xml").read_text(encoding="utf-8")


def render(**overrides: object) -> str:
    values: dict[str, object] = {"input_file": "C:/x/in.dim", "source_bands": "Sigma0_VV,Sigma0_VH",
                                 "filter": "Lee Sigma", "num_looks": "1", "window_size": "7x7",
                                 "target_window_size": "3x3", "sigma": "0.9", "output_file": "C:/x/out.dim"}
    return snap.render_graph(TEMPLATE, {**values, **overrides})


@pytest.fixture
def gpt(tmp_path):
    path = tmp_path / "gpt.exe"
    path.write_text("TEST FIXTURE")
    return snap.ResolvedGpt(path, "config")


@pytest.fixture
def cfg_root(tmp_path_factory):
    """A project under a SHORT root (SNAP-like outputs nest ~175 characters deep)."""
    root = tmp_path_factory.mktemp("s")
    shutil.copytree(REPO_CONFIG_DIR, root / "config")
    return root


@pytest.fixture
def published(cfg_root, gpt):
    """A PUBLISHED Step 3.2 output, made by the real Step 3.2 code with the fake gpt."""
    project = make_project(load_config(cfg_root / "config"))
    fake = FakeSnap()
    rad.run_radiometric(rad.prepare_job(project.manifest, TRISHULI_PRE1, project.cfg), project.cfg, gpt=gpt,
                        runner=fake.run, popen=fake.popen)
    return project


def job_for(p):
    return spk.prepare_speckle_job(p.manifest, TRISHULI_PRE1, p.cfg)


def run(p, gpt, fake, **kw):
    return spk.run_speckle(job_for(p), p.cfg, gpt=gpt, runner=fake.run, popen=fake.popen, **kw)


def folders(p):
    parent = p.cfg.settings.paths.interim / "preprocessing" / p.manifest.aoi_id / TRISHULI_PRE1
    return sorted(x.name for x in parent.iterdir())


# ---- graph and configuration (no SNAP, no rasterio) -------------------------------------------

def test_graph_is_the_kuro_siwo_lee_sigma_node():
    rendered = render()
    ops = snap.graph_operators(rendered)
    assert [o for _n, o, _p in ops] == ["Read", "Speckle-Filter", "Write"]
    params = {o: p for _n, o, p in ops}
    assert {k: v for k, v in params["Speckle-Filter"].items() if k != "sourceBands"} == KURO_SIWO_NODE
    assert params["Speckle-Filter"]["sourceBands"] == "Sigma0_VV,Sigma0_VH"
    assert params["Read"] == {"file": "C:/x/in.dim", "formatName": "BEAM-DIMAP"}
    assert params["Write"] == {"file": "C:/x/out.dim", "formatName": "BEAM-DIMAP"}
    for forbidden in ("Terrain-Correction", "LinearToFromdB", "Calibration", "Multilook", "BandMaths"):
        assert forbidden not in rendered


def test_graph_values_come_from_config(cfg):
    s = cfg.settings.preprocessing.s1_speckle
    assert (s.filter, s.num_looks, s.window_size, s.target_window_size, s.sigma) == (
        "Lee Sigma", "1", "7x7", "3x3", "0.9")


@pytest.mark.parametrize("change", [
    {"filter": "Refined Lee"},           # no silent substitution of another filter
    {"filter": "Lee"},
    {"sigma": 0.9},                      # must be SNAP's string, not a float
    {"window_size": "6x6"},
    {"num_looks": "5"},
    {"target_window_size": "7x7"},
    {"window_size": "5x5", "target_window_size": "5x5"},
    {"graph": "../evil.xml"},
])
def test_config_accepts_only_lee_sigma_with_snap_values(config_copy, change):
    with pytest.raises(ConfigError):
        load_config(config_copy("config.yaml", lambda d: d["preprocessing"]["s1_speckle"].update(change)))


def test_value_sets_are_read_from_gpt_help():
    help_ = speckle_help(["filter", "windowSize", "anSize"], SPECKLE_VALUE_SETS)
    assert snap.parameter_value_set(help_, "filter") == SPECKLE_VALUE_SETS["filter"]     # 'Gamma Map' kept whole
    assert snap.parameter_value_set(help_, "windowSize") == SPECKLE_VALUE_SETS["windowSize"]
    assert snap.parameter_value_set(help_, "anSize") is None
    assert snap.parameter_value_set(help_, "sigmaStr") is None


def test_parameter_values_checked_against_installed_snap(tmp_path):
    gpt_exe = tmp_path / "gpt.exe"
    report = snap.check_parameter_values(gpt_exe, render(), ["Speckle-Filter"], 10, FakeSpeckleSnap().run)
    checked = report["Speckle-Filter"]["checked"]
    assert set(checked) == {"filter", "numLooksStr", "windowSize", "targetWindowSizeStr", "sigmaStr"}
    assert "anSize" in report["Speckle-Filter"]["not_listed"]
    older = FakeSpeckleSnap(value_sets={**SPECKLE_VALUE_SETS, "windowSize": ("5x5", "9x9")})
    with pytest.raises(snap.SnapError, match="windowSize='7x7' is not accepted"):
        snap.check_parameter_values(gpt_exe, render(), ["Speckle-Filter"], 10, older.run)
    no_lee_sigma = FakeSpeckleSnap(value_sets={**SPECKLE_VALUE_SETS, "filter": ("Lee", "Refined Lee")})
    with pytest.raises(snap.SnapError, match="filter='Lee Sigma'"):
        snap.check_parameter_values(gpt_exe, render(), ["Speckle-Filter"], 10, no_lee_sigma.run)
    renamed = FakeSpeckleSnap(missing_param=("Speckle-Filter", "sigmaStr"))
    with pytest.raises(snap.SnapError, match="sigmaStr"):
        snap.check_operator_parameters(gpt_exe, render(), 10, renamed.run)


@pytest.mark.parametrize("spec, problem", [
    (DimapSpec(unit="intensity_db"), "dB"),
    (DimapSpec(log10_scaled=True), "dB"),
    (DimapSpec(map_geocoding=True), "radar geometry"),
    (DimapSpec(history=("Read", "Calibration", "Speckle-Filter", "Write")), "already processed"),
    (DimapSpec(history=("Read", "Calibration", "Terrain-Correction", "Write")), "already processed"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((4, 5), np.float32)}), "expected exactly"),
    (DimapSpec(bands={"Sigma0_VV": np.ones((4, 5), np.float32), "Sigma0_VH": np.ones((4, 5), np.float32),
                      "Beta0_VV": np.ones((4, 5), np.float32)}), "expected exactly"),
])
def test_input_must_be_linear_sigma0_in_radar_geometry(tmp_path, spec, problem):
    dim = tmp_path / "in.dim"
    write_dimap(dim, spec)
    with pytest.raises(rad.PreprocessingError, match=problem):
        spk.check_input_product(read_dimap(dim), ("VV", "VH"))
    write_dimap(tmp_path / "ok.dim", DimapSpec())
    spk.check_input_product(read_dimap(tmp_path / "ok.dim"), ("VV", "VH"))


def test_history_helper_appends_inside_processing_graph(tmp_path):
    write_dimap(tmp_path / "a.dim", DimapSpec())
    text = append_history((tmp_path / "a.dim").read_text(encoding="ISO-8859-1"), "Speckle-Filter")
    (tmp_path / "b.dim").write_text(text, encoding="ISO-8859-1")
    (tmp_path / "a.data").rename(tmp_path / "b.data")
    (tmp_path / "b.dim").write_text(text.replace("a.data/", "b.data/"), encoding="ISO-8859-1")
    assert read_dimap(tmp_path / "b.dim").history[-1] == "Speckle-Filter"


def test_missing_step_3_2_output_is_reported(cfg_root):
    p = make_project(load_config(cfg_root / "config"))
    with pytest.raises(rad.PreprocessingError, match="no published Step 3.2 output.*--stage radiometric"):
        job_for(p)
    with pytest.raises(rad.PreprocessingError, match="not in the manifest"):
        spk.prepare_speckle_job(p.manifest, "S1A-D019-20260101T000000", p.cfg)
    with pytest.raises(rad.PreprocessingError, match="dry-run"):
        spk.prepare_speckle_job(p.manifest.model_copy(update={"dry_run": True}), TRISHULI_PRE1, p.cfg)


def test_input_inside_an_evaluation_folder_is_refused(config_copy):
    cfg = load_config(config_copy("config.yaml", lambda d: d["paths"].update(interim="data/reference/interim")))
    p = make_project(cfg)
    with pytest.raises(ProhibitedInputError):
        job_for(p)


# ---- runs with a fake gpt (need rasterio) ------------------------------------------------------

@needs_rasterio
def test_job_reads_the_published_radiometric_artifact(published):
    job = job_for(published)
    assert job.input_dir.name == "radiometric" and job.final_dir.name == "speckle"
    assert job.input_dim.name == f"{TRISHULI_PRE1}_sigma0.dim" and job.output_stem == f"{TRISHULI_PRE1}_sigma0_spk"
    assert any(k.endswith("Sigma0_VV.img") for k in job.input_hashes) and len(job.input_hashes) >= 7
    assert all(len(v) == 64 for v in job.input_hashes.values())


@needs_rasterio
def test_db_or_already_filtered_published_input_is_refused(published):
    dim = job_for(published).input_dim
    text = dim.read_text(encoding="ISO-8859-1")
    dim.write_text(text.replace("<PHYSICAL_UNIT>intensity</PHYSICAL_UNIT>", "<PHYSICAL_UNIT>intensity_db"
                                "</PHYSICAL_UNIT>"), encoding="ISO-8859-1")
    with pytest.raises(rad.PreprocessingError, match="dB"):
        job_for(published)
    dim.write_text(append_history(text, "Speckle-Filter"), encoding="ISO-8859-1")
    with pytest.raises(rad.PreprocessingError, match="already processed"):
        job_for(published)


@needs_rasterio
def test_input_from_other_settings_or_pass_is_refused(published):
    other = published.cfg.model_copy(update={"settings": published.cfg.settings.model_copy(update={
        "preprocessing": published.cfg.settings.preprocessing.model_copy(update={"crop_buffer_m": 5000.0})})})
    with pytest.raises(rad.PreprocessingError, match="different radiometric settings"):
        spk.prepare_speckle_job(published.manifest, TRISHULI_PRE1, other)
    prov_path = job_for(published).input_provenance_path
    data = json.loads(prov_path.read_text())
    data["pass_id"] = "S1D-A085-20260816T122141"
    prov_path.write_text(json.dumps(data))
    with pytest.raises(rad.PreprocessingError, match="not to this manifest pass"):
        job_for(published)


@needs_rasterio
def test_successful_run_publishes_validated_output_with_provenance(published, gpt):
    before = folders(published)
    input_hashes = job_for(published).input_hashes
    fake = FakeSpeckleSnap()
    result = run(published, gpt, fake)
    assert folders(published) == sorted([*before, "speckle"])                    # no tmp/failed left
    out = result.output_dir
    assert {f.name for f in out.iterdir()} == {f"{TRISHULI_PRE1}_sigma0_spk.dim", f"{TRISHULI_PRE1}_sigma0_spk.data",
                                               "graph_resolved.xml", "gpt.log", "preprocessing_provenance.json"}
    prov = load_contract(SpeckleProvenance, result.provenance_path)
    assert prov == result.provenance and prov.status == "succeeded" and prov.validation.passed
    assert {k: v for k, v in prov.lee_sigma.items() if k != "sourceBands"} == KURO_SIWO_NODE
    assert prov.snap_value_sets["Speckle-Filter"]["checked"]["windowSize"] == list(SPECKLE_VALUE_SETS["windowSize"])
    assert prov.graph.resolved_sha256 == snap.sha256_text((out / "graph_resolved.xml").read_text(encoding="utf-8"))
    assert prov.graph.template_sha256 == snap.sha256_text(TEMPLATE)
    assert prov.input.files_sha256 == input_hashes == job_for(published).input_hashes   # input untouched
    radiometric = load_contract(PreprocessingProvenance, published.root / prov.input.provenance_path)
    assert prov.input.radiometric_graph_sha256 == radiometric.graph.resolved_sha256
    assert prov.input.provenance_sha256 == spk.sha256_file(published.root / prov.input.provenance_path)
    for rel, digest in prov.output_sha256.items():
        assert spk.sha256_file(published.root / rel) == digest and "/speckle/" in rel
    assert prov.config_fingerprint == published.cfg.fingerprint() and prov.snap.snap_version == "12.0.0"
    assert (prov.input.width, prov.input.height) == (prov.validation.width, prov.validation.height) == (60, 40)
    for rel in (prov.output.dim, prov.snap.log_path, prov.graph.resolved_path,
                *(b.file for b in prov.validation.bands_after)):
        assert (published.root / rel).exists() and ".tmp-" not in rel
    assert all(k.get("shell") is False for k in fake.run_kwargs)


@needs_rasterio
def test_validation_reports_before_and_after(published, gpt):
    v = run(published, gpt, FakeSpeckleSnap()).provenance.validation
    assert {c.name for c in v.checks} >= {
        "output_exists", "sigma0_bands", "linear_scale", "radar_geometry", "no_downstream_operators",
        "speckle_recorded", "orbit_preserved", "pixel_spacing", "same_grid", "geolocation_preserved",
        "readable_VV", "float_VH", "valid_pixels_VV", "filter_changed_VV", "no_new_nonfinite_VH"}
    assert v.geometry.startswith("radar") and v.rasterio_crs is None
    for before, after, comp in zip(v.bands_before, v.bands_after, v.comparisons, strict=True):
        assert before.band == after.band == comp.band
        assert before.valid_pixels == after.valid_pixels == comp.compared_pixels
        assert 0 < comp.changed_pixels <= comp.compared_pixels and comp.new_nonfinite_pixels == 0
        assert comp.cv_after < comp.cv_before                     # the fake 3x3 mean smooths (diagnostic only)
        assert before.zero_pixels == after.zero_pixels and after.nonfinite_pixels == before.nonfinite_pixels


@pytest.mark.parametrize("fake, failed", [
    (FakeSpeckleSnap(mutate=set_unit_db), "linear_scale"),
    (FakeSpeckleSnap(mutate=add_band), "sigma0_bands"),
    (FakeSpeckleSnap(mutate=shrink_vh), "same_grid"),
    (FakeSpeckleSnap(mutate=map_projection), "radar_geometry"),
    (FakeSpeckleSnap(mutate=int_vv), "float_VV"),
    (FakeSpeckleSnap(mutate=drop_vh_file), "output_exists"),
    (FakeSpeckleSnap(mutate=nan_corruption), "no_new_nonfinite_VV"),
    (FakeSpeckleSnap(mutate=move_geolocation), "geolocation_preserved"),
    (FakeSpeckleSnap(mutate=change_spacing), "pixel_spacing"),
    (FakeSpeckleSnap(mutate=change_orbit), "orbit_preserved"),
    (FakeSpeckleSnap(filter_fn=lambda a: a), "filter_changed_VV"),
    (FakeSpeckleSnap(history=("Read", "Speckle-Filter", "Terrain-Correction", "Write")), "no_downstream_operators"),
    (FakeSpeckleSnap(history=("Read", "Speckle-Filter", "LinearToFromdB", "Write")), "no_downstream_operators"),
    (FakeSpeckleSnap(history=("Read", "Write")), "speckle_recorded"),
])
@needs_rasterio
def test_invalid_outputs_fail_and_are_never_published(published, gpt, fake, failed):
    before = folders(published)
    with pytest.raises(spk.SpeckleRunFailed) as info:
        run(published, gpt, fake)
    prov = info.value.provenance
    assert prov.validation is not None, f"the run failed before validation: {prov.error}"
    assert prov.status == "failed" and failed in {c.name for c in prov.validation.failed_checks()}
    assert prov.output is None and prov.output_sha256 == {}
    assert info.value.output_dir.name.startswith("speckle.failed-")
    assert folders(published) == sorted([*before, info.value.output_dir.name])    # no 'speckle' folder
    assert load_contract(SpeckleProvenance, info.value.output_dir / rad.PROVENANCE_FILE).status == "failed"


@needs_rasterio
def test_input_changed_during_the_run_fails(published, gpt):
    with pytest.raises(spk.SpeckleRunFailed, match="input changed during the run"):
        run(published, gpt, FakeSpeckleSnap(mutate=modify_input))


@needs_rasterio
def test_gpt_failure_and_timeout_keep_a_failed_folder(published, gpt):
    with pytest.raises(spk.SpeckleRunFailed, match="exit code 1") as info:
        run(published, gpt, FakeSpeckleSnap(returncode=1, write_output=False))
    assert info.value.provenance.validation is None and info.value.provenance.error
    with pytest.raises(spk.SpeckleRunFailed, match="did not finish"):
        run(published, gpt, FakeSpeckleSnap(timeout=True, write_output=False))
    assert "speckle" not in folders(published)


@needs_rasterio
def test_snap_discrepancies_stop_before_anything_is_written(published, gpt):
    before = folders(published)
    with pytest.raises(snap.SnapError, match="windowSize='7x7'"):
        run(published, gpt, FakeSpeckleSnap(value_sets={**SPECKLE_VALUE_SETS, "windowSize": ("5x5",)}))
    with pytest.raises(snap.SnapError, match="targetWindowSizeStr"):
        run(published, gpt, FakeSpeckleSnap(missing_param=("Speckle-Filter", "targetWindowSizeStr")))
    assert folders(published) == before


@needs_rasterio
def test_unlisted_value_sets_are_recorded_as_warnings(published, gpt):
    result = run(published, gpt, FakeSpeckleSnap(value_sets={}))
    assert result.provenance.warnings and "lists no value set" in result.provenance.warnings[0]


@needs_rasterio
def test_existing_output_is_refused_or_superseded_never_deleted(published, gpt):
    first = run(published, gpt, FakeSpeckleSnap())
    fake = FakeSpeckleSnap()
    with pytest.raises(rad.PreprocessingError, match="already exists"):
        run(published, gpt, fake)
    assert fake.calls == []
    (first.output_dir / "marker.txt").write_text("old")
    second = run(published, gpt, FakeSpeckleSnap(), overwrite=True)
    superseded = [n for n in folders(published) if n.startswith("speckle.superseded-")]
    assert len(superseded) == 1 and second.output_dir.name == "speckle"
    assert (second.output_dir.parent / superseded[0] / "marker.txt").read_text() == "old"
    assert "radiometric" in folders(published)                                  # Step 3.2 output untouched

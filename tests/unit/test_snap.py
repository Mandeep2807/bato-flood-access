"""Step 3.2: SNAP gpt integration with mocked subprocess (no SNAP installation needed)."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bato.config import SnapSettings
from bato.preprocessing import snap

from snap_fakes import FakeSnap

SETTINGS = SnapSettings(gpt_path=None, timeout_s=60, diag_timeout_s=10)
TEMPLATE = snap.graph_path("s1_radiometric.xml").read_text(encoding="utf-8")
VALUES: dict[str, object] = {
    "input_file": "C:/data/raw/sentinel1/S1D_X.SAFE.zip", "orbit_type": "Sentinel Precise (Auto Download)",
    "orbit_poly_degree": 3, "orbit_continue_on_fail": False,
    "geo_region": "POLYGON ((85 27, 86 27, 86 28, 85 28, 85 27))",
    "polarisations": "VV,VH", "remove_thermal_noise": True, "reintroduce_thermal_noise": False, "border_limit": 500,
    "trim_threshold": 50.0, "output_in_db": False, "output_sigma0": True, "output_file": "C:/out/x_sigma0.dim",
}


def exe(tmp_path: Path, name: str = "gpt.exe") -> Path:
    path = tmp_path / name
    path.write_text("TEST FIXTURE")
    return path


# ---- executable resolution ---------------------------------------------------------------

def test_explicit_config_path_wins(tmp_path):
    gpt = exe(tmp_path)
    env_gpt = exe(tmp_path, "other.exe")
    found = snap.resolve_gpt(SnapSettings(gpt_path=gpt, timeout_s=1, diag_timeout_s=1),
                             environ={snap.ENV_GPT: str(env_gpt)}, candidates=[])
    assert found == snap.ResolvedGpt(gpt, "config")


def test_missing_explicit_paths_never_fall_back(tmp_path):
    with pytest.raises(snap.SnapError, match="snap.gpt_path"):
        snap.resolve_gpt(SnapSettings(gpt_path=tmp_path / "nope.exe", timeout_s=1, diag_timeout_s=1),
                         environ={}, candidates=[exe(tmp_path)])
    with pytest.raises(snap.SnapError, match=snap.ENV_GPT):
        snap.resolve_gpt(SETTINGS, environ={snap.ENV_GPT: str(tmp_path / "nope.exe")}, candidates=[exe(tmp_path)])


def test_env_then_default_then_path(tmp_path):
    gpt = exe(tmp_path)
    assert snap.resolve_gpt(SETTINGS, environ={snap.ENV_GPT: str(gpt)}, candidates=[]).source == "env:BATO_SNAP_GPT"
    default = exe(tmp_path, "default.exe")
    assert snap.resolve_gpt(SETTINGS, environ={}, candidates=[tmp_path / "x.exe", default]) == \
        snap.ResolvedGpt(default, "default")
    assert snap.resolve_gpt(SETTINGS, environ={}, candidates=[], which=lambda n: str(gpt)).source == "PATH"
    with pytest.raises(snap.SnapError, match="SNAP gpt not found"):
        snap.resolve_gpt(SETTINGS, environ={}, candidates=[], which=lambda n: None)


def test_windows_default_location_is_a_candidate_not_the_only_option():
    candidates = snap.default_gpt_candidates("Windows", Path("C:/Users/me"))
    assert any(p.as_posix().endswith("esa-snap/bin/gpt.exe") and "Program Files" in str(p) for p in candidates)
    assert Path("C:/Users/me/esa-snap/bin/gpt.exe") in candidates
    assert all(p.name == "gpt" for p in snap.default_gpt_candidates("Linux", Path("/home/me")))


# ---- diagnostics ---------------------------------------------------------------------------

def test_diagnostics_parse_version_and_use_no_shell(tmp_path):
    fake = FakeSnap()
    diag = snap.snap_diagnostics(exe(tmp_path), 10, fake.run)
    assert diag.version == "12.0.0" and any("Max memory" in line for line in diag.output_head)
    assert fake.calls[0][1:] == ["--diag"] and fake.run_kwargs[0]["shell"] is False
    assert fake.run_kwargs[0]["capture_output"] is True and fake.run_kwargs[0]["timeout"] == 10


@pytest.mark.parametrize("error, match", [(FileNotFoundError("x"), "cannot start"),
                                          (subprocess.TimeoutExpired("gpt", 10), "no answer within")])
def test_diagnostics_start_failures_are_clear(tmp_path, error, match):
    def runner(*a, **k):
        raise error
    with pytest.raises(snap.SnapError, match=match):
        snap.snap_diagnostics(exe(tmp_path), 10, runner)


def test_diagnostics_nonzero_exit(tmp_path):
    def runner(args, **k):
        return subprocess.CompletedProcess(args, 1, "", "java.lang.OutOfMemoryError")
    with pytest.raises(snap.SnapError, match="(?s)exit code 1.*OutOfMemoryError"):
        snap.snap_diagnostics(exe(tmp_path), 10, runner)


# ---- graph -------------------------------------------------------------------------------

def test_graph_contains_exactly_the_step_3_2_chain():
    rendered = snap.render_graph(TEMPLATE, VALUES)
    assert [op for _n, op, _p in snap.graph_operators(rendered)] == [
        "Read", "Apply-Orbit-File", "Subset", "ThermalNoiseRemoval", "Remove-GRD-Border-Noise", "Calibration", "Write"]
    params = {op: p for _n, op, p in snap.graph_operators(rendered)}
    assert params["Apply-Orbit-File"] == {"orbitType": "Sentinel Precise (Auto Download)", "polyDegree": "3",
                                          "continueOnFail": "false"}
    assert params["ThermalNoiseRemoval"]["removeThermalNoise"] == "true"
    assert params["ThermalNoiseRemoval"]["reIntroduceThermalNoise"] == "false"
    assert params["ThermalNoiseRemoval"]["outputNoise"] == "false"
    assert params["Remove-GRD-Border-Noise"] == {"selectedPolarisations": "VV,VH", "borderLimit": "500",
                                                 "trimThreshold": "50.0"}
    cal = params["Calibration"]
    assert cal["outputImageScaleInDb"] == "false" and cal["outputSigmaBand"] == "true"
    assert cal["outputBetaBand"] == cal["outputGammaBand"] == cal["outputImageInComplex"] == "false"
    assert cal["createBetaBand"] == cal["createGammaBand"] == "false"
    assert params["Subset"]["geoRegion"] == VALUES["geo_region"] and params["Subset"]["copyMetadata"] == "true"
    assert params["Write"] == {"file": "C:/out/x_sigma0.dim", "formatName": "BEAM-DIMAP"}
    forbidden = ("Terrain-Correction", "Speckle-Filter", "LinearToFromdB", "Land-Sea-Mask", "Multilook")
    assert not any(op in rendered for op in forbidden)


def test_render_requires_exact_placeholders_and_escapes_values():
    with pytest.raises(snap.SnapError, match="missing \\['border_limit'\\]"):
        snap.render_graph(TEMPLATE, {k: v for k, v in VALUES.items() if k != "border_limit"})
    with pytest.raises(snap.SnapError, match="unused \\['extra'\\]"):
        snap.render_graph(TEMPLATE, {**VALUES, "extra": 1})
    rendered = snap.render_graph(TEMPLATE, {**VALUES, "input_file": "C:/a&b <x>/p.zip"})
    assert ET.fromstring(rendered).find(".//node[@id='Read']/parameters/file").text == "C:/a&b <x>/p.zip"
    assert "<!--" not in rendered and "${" not in rendered


def test_hash_is_deterministic():
    a, b = snap.render_graph(TEMPLATE, VALUES), snap.render_graph(TEMPLATE, VALUES)
    assert snap.sha256_text(a) == snap.sha256_text(b) and len(snap.sha256_text(a)) == 64
    assert snap.sha256_text(snap.render_graph(TEMPLATE, {**VALUES, "border_limit": 400})) != snap.sha256_text(a)


def test_operator_parameters_are_checked_against_installed_snap(tmp_path):
    rendered = snap.render_graph(TEMPLATE, VALUES)
    fake = FakeSnap()
    checked = snap.check_operator_parameters(exe(tmp_path), rendered, 10, fake.run)
    assert set(checked) == {"Read", "Apply-Orbit-File", "Subset", "ThermalNoiseRemoval", "Remove-GRD-Border-Noise",
                            "Calibration", "Write"}
    assert all(call[2] == "-h" for call in fake.calls) and all(k["shell"] is False for k in fake.run_kwargs)
    unknown = FakeSnap(missing_param=("Remove-GRD-Border-Noise", "trimThreshold"))
    with pytest.raises(snap.SnapError, match="trimThreshold"):
        snap.check_operator_parameters(exe(tmp_path), rendered, 10, unknown.run)


def test_unknown_operator_is_reported(tmp_path):
    def runner(args, **k):
        return subprocess.CompletedProcess(args, 1, "", "Error: Unknown operator 'Calibration'")
    with pytest.raises(snap.SnapError, match="unknown to this SNAP installation"):
        snap.check_operator_parameters(exe(tmp_path), snap.render_graph(TEMPLATE, VALUES), 10, runner)


# ---- running -----------------------------------------------------------------------------

def graph_file(tmp_path: Path) -> Path:
    path = tmp_path / "graph.xml"
    path.write_text(snap.render_graph(TEMPLATE, {**VALUES, "output_file": (tmp_path / "o.dim").as_posix()}))
    return path


def test_run_streams_output_to_log_and_progress(tmp_path):
    fake, seen = FakeSnap(), []
    ticks = iter([100.0, 142.34])
    run = snap.run_gpt(exe(tmp_path), graph_file(tmp_path), tmp_path / "gpt.log", 60, popen=fake.popen,
                       clock=lambda: next(ticks), progress=seen.append)
    assert run.returncode == 0 and run.runtime_s == 42.3 and run.args[-1] == "-e"
    assert fake.run_kwargs[0]["shell"] is False and isinstance(fake.calls[0], list)
    log = (tmp_path / "gpt.log").read_text()
    assert log.startswith("$ ") and "100% done" in log and "100% done" in "".join(seen)


def test_run_failure_includes_log_tail(tmp_path):
    fake = FakeSnap(returncode=1, write_output=False)
    with pytest.raises(snap.SnapError, match="(?s)exit code 1.*No valid orbit file"):
        snap.run_gpt(exe(tmp_path), graph_file(tmp_path), tmp_path / "gpt.log", 60, popen=fake.popen)


def test_run_timeout_kills_gpt(tmp_path):
    fake = FakeSnap(timeout=True, write_output=False)
    with pytest.raises(snap.SnapError, match="did not finish within 60 s"):
        snap.run_gpt(exe(tmp_path), graph_file(tmp_path), tmp_path / "gpt.log", 60, popen=fake.popen)
    assert fake.killed


def test_run_cannot_start(tmp_path):
    def popen(*a, **k):
        raise FileNotFoundError("gpt.exe")
    with pytest.raises(snap.SnapError, match="cannot start gpt"):
        snap.run_gpt(tmp_path / "gpt.exe", graph_file(tmp_path), tmp_path / "gpt.log", 60, popen=popen)

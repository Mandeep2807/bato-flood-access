"""Data boundary of ``scripts/evaluate_emsr927.py --manifest`` (regression for "REFUSED (data rule)" on the EMSR927
evaluation manifest).

* The evaluation manifest ``data/reference/EMSR927/acquisition_manifest.json`` is accepted, under the reference guard.
* Production/pipeline entry points still refuse anything under ``data/reference/``.
* A product zip is refused everywhere.
* The zip integrity checks still run.
* Nothing from the reference folder appears in BATO production locations.

Synthetic data only (tests/ems_fakes.py, tests/dashboard_fakes.py).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pytest

from bato.data_rules import assert_pipeline_input_allowed
from bato.evaluation import emsr927 as ev
from bato.evaluation.acquisition import acquire
from bato.evaluation.cli import resolve_target
from bato.evaluation.official import official_aoi
from bato.evaluation.protocol import load_protocol
from bato.exceptions import ProhibitedInputError

from dashboard_fakes import published_world
from ems_fakes import FakeOpener, activation, install_draft_protocol, standard_zips

REPO = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def frozen(tmp_path):
    """Synthetic world: official v1 run, acquired EMS products, inspected and frozen protocol (no metric yet)."""
    w = published_world(tmp_path)
    install_draft_protocol(tmp_path)
    w.reader = lambda _p: (w.state, w.transform, "EPSG:3857")
    acquire(w.cfg.settings.paths, official_aoi(w.cfg, "aoi-test", "ts-test"), load_protocol(w.root).raw["reference"],
            opener=FakeOpener(activation({}), standard_zips()), now=lambda: T0)
    ev.inspect(w.cfg, load_protocol(w.root))
    ev.freeze_protocol(w.cfg, load_protocol(w.root))
    w.ref = w.cfg.settings.paths.reference / "EMSR927"
    w.manifest = w.ref / "acquisition_manifest.json"
    w.zip = next((w.ref / "products").glob("*.zip"))
    return w


def run_evaluate_script(w, monkeypatch, *args: str) -> int:
    """Run scripts/evaluate_emsr927.py main() in-process. The Step 4C raster reader is the only thing injected:
    rasterio is not needed for the synthetic state."""
    spec = importlib.util.spec_from_file_location("evaluate_emsr927_script", REPO / "scripts" / "evaluate_emsr927.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "evaluate", partial(ev.evaluate, read_state=w.reader))
    monkeypatch.setattr(sys, "argv", ["evaluate_emsr927.py", "--config-dir", str(w.cfg.config_dir), *args])
    return int(mod.main())


def tree(*dirs: Path) -> dict[str, str]:
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs if d.is_dir()
            for p in sorted(d.rglob("*")) if p.is_file()}


def test_evaluation_manifest_is_accepted_and_nothing_reaches_production_locations(frozen, monkeypatch, capsys):
    paths = frozen.cfg.settings.paths
    before = tree(paths.raw, paths.interim, paths.processed)
    ref_hashes = set(tree(frozen.ref).values())

    code = run_evaluate_script(frozen, monkeypatch, "--manifest", str(frozen.manifest))
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "emsr927_evaluation_manifest -> BATO aoi-test / ts-test" in out.out
    assert "REFUSED" not in out.err
    runs = (paths.eval / "EMSR927" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(runs) == 1

    after = tree(paths.raw, paths.interim, paths.processed)
    assert after == before                                               # nothing written or changed there
    assert not ref_hashes & set(after.values())                          # no reference file copied there
    assert not [p for p in after if "emsr927" in p.lower()]
    # outputs only in the evaluation folder
    assert all(Path(p).is_relative_to(paths.eval) or Path(p).is_relative_to(paths.reference)
               for p in tree(paths.eval, paths.reference))


def test_resolver_uses_the_reference_guard_only_for_the_canonical_evaluation_manifest(frozen):
    t = resolve_target(frozen.cfg, frozen.manifest)
    assert (t.aoi_id, t.candidate_id, t.manifest_kind) == ("aoi-test", "ts-test", "emsr927_evaluation_manifest")
    with pytest.raises(ValueError, match="contradicts"):
        resolve_target(frozen.cfg, frozen.manifest, candidate="ts-other")
    data = json.loads(frozen.manifest.read_text(encoding="utf-8"))
    frozen.manifest.write_text(json.dumps({**data, "kind": "something_else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an EMSR927 evaluation manifest"):
        resolve_target(frozen.cfg, frozen.manifest)


@pytest.mark.parametrize("which", ["zip", "activation", "copy_of_manifest"])
def test_other_reference_files_are_refused_as_manifest(frozen, monkeypatch, capsys, which):
    if which == "zip":
        target = frozen.zip
    elif which == "activation":
        target = next((frozen.ref / "activation").glob("*.json"))
    else:
        target = frozen.ref / "products" / "acquisition_manifest.json"
        target.write_bytes(frozen.manifest.read_bytes())
    with pytest.raises(ProhibitedInputError):
        resolve_target(frozen.cfg, target)
    assert run_evaluate_script(frozen, monkeypatch, "--manifest", str(target)) == 1
    assert "REFUSED (data rule)" in capsys.readouterr().err
    assert not (frozen.cfg.settings.paths.eval / "EMSR927" / "runs.jsonl").exists()


def test_pipeline_guard_still_refuses_reference_files(frozen):
    for p in (frozen.manifest, frozen.zip, frozen.ref):
        with pytest.raises(ProhibitedInputError):
            assert_pipeline_input_allowed(p, frozen.cfg.settings.paths)


@pytest.mark.parametrize("script", ["build_dashboard.py", "run_connectivity.py"])
@pytest.mark.parametrize("which", ["manifest", "zip"])
def test_production_scripts_refuse_reference_data_as_input(frozen, script, which):
    """Real production entry points: the EMSR927 manifest or a zip passed as their --manifest is refused."""
    if script == "run_connectivity.py" and importlib.util.find_spec("rasterio") is None:
        pytest.skip("run_connectivity.py checks for rasterio before its data guard; rasterio not installed here")
    target = frozen.manifest if which == "manifest" else frozen.zip
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--manifest", str(target),
                        "--config-dir", str(frozen.cfg.config_dir)], capture_output=True, text=True, timeout=120,
                       env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}, cwd=str(frozen.root))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSED (data rule)" in r.stderr and "evaluation-only folder" in r.stderr


def test_zip_integrity_check_still_runs_through_the_cli(frozen, monkeypatch, capsys):
    original = frozen.zip.read_bytes()
    frozen.zip.write_bytes(original + b"tampered")
    assert run_evaluate_script(frozen, monkeypatch, "--manifest", str(frozen.manifest)) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert not (frozen.cfg.settings.paths.eval / "EMSR927" / "runs.jsonl").exists()
    frozen.zip.write_bytes(original)
    assert run_evaluate_script(frozen, monkeypatch, "--manifest", str(frozen.manifest)) == 0

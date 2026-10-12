"""Data-rule guards: evaluation-only data must never reach the pipeline."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bato.contracts.common import DataSource
from bato.data_rules import EVALUATION_PACKAGE, assert_pipeline_input_allowed
from bato.exceptions import ProhibitedInputError

SRC = Path(__file__).resolve().parents[2] / "src" / "bato"
ERROR_PRIVILEGE_NOT_HELD = 1314   # Windows: "A required privilege is not held by the client"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_pipeline_code_never_imports_evaluation_package():
    offenders = []
    for path in SRC.rglob("*.py"):
        if (SRC / "evaluation") in path.parents:
            continue
        for module in _imported_modules(path):
            if module == EVALUATION_PACKAGE or module.startswith(EVALUATION_PACKAGE + "."):
                offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, "pipeline code must not import bato.evaluation:\n" + "\n".join(offenders)


def test_import_scanner_detects_violations(tmp_path):
    """Guard the guard: make sure the scanner would catch a real violation."""
    sample = tmp_path / "bad.py"
    sample.write_text("from bato.evaluation.reference import ReferenceSource\nimport bato.evaluation\n")
    assert {"bato.evaluation.reference", "bato.evaluation"} <= _imported_modules(sample)


def test_allowed_sources_exclude_prohibited_products():
    names = " ".join(f"{m.name} {m.value}" for m in DataSource).lower()
    for prohibited in ("ems", "unosat", "worldpop", "ghsl", "jrc", "post_event", "post-event"):
        assert prohibited not in names, f"DataSource must not include {prohibited!r}"


def test_reference_and_eval_folders_are_blocked(cfg, tmp_path):
    paths = cfg.settings.paths
    for folder in paths.evaluation_only_dirs():
        with pytest.raises(ProhibitedInputError):
            assert_pipeline_input_allowed(folder / "EMSR927" / "delineation.shp", paths)
        with pytest.raises(ProhibitedInputError):
            assert_pipeline_input_allowed(folder, paths)


def test_raw_inputs_are_allowed(cfg):
    paths = cfg.settings.paths
    target = paths.raw / "s1" / "chip.tif"
    assert assert_pipeline_input_allowed(target, paths) == target.resolve()


def test_symlink_into_reference_is_blocked(cfg, tmp_path):
    paths = cfg.settings.paths.model_copy(update={"reference": tmp_path / "reference",
                                                  "raw": tmp_path / "raw"})
    (tmp_path / "reference").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "reference" / "ems.geojson").write_text("{}", encoding="utf-8")
    link = tmp_path / "raw" / "looks_innocent.geojson"
    try:
        link.symlink_to(tmp_path / "reference" / "ems.geojson")
    except OSError as exc:
        # Windows: creating symlinks needs Developer Mode or admin rights (ERROR_PRIVILEGE_NOT_HELD).
        # Only that environment limitation is skipped; any other error still fails the test.
        if getattr(exc, "winerror", None) == ERROR_PRIVILEGE_NOT_HELD:
            pytest.skip("cannot create a symlink on this Windows account (WinError 1314: enable Developer "
                        "Mode or run as administrator to run this test); the guard itself is unchanged")
        raise
    with pytest.raises(ProhibitedInputError):
        assert_pipeline_input_allowed(link, paths)


def test_similarly_named_sibling_folder_is_not_confused(cfg, tmp_path):
    paths = cfg.settings.paths.model_copy(update={"reference": tmp_path / "reference"})
    sibling = tmp_path / "reference_notes" / "file.txt"
    assert assert_pipeline_input_allowed(sibling, paths) == sibling.resolve()

"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from bato.config import BatoConfig, load_config

from helpers import REPO_CONFIG_DIR


@pytest.fixture(autouse=True)
def _fake_geoparquet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let tests feed ohsome v2 responses without pyarrow: fake GeoParquet bytes (connectivity_fakes) are
    decoded directly; genuine Parquet still goes through the real pyarrow reader."""
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows

    real = extract.read_parquet_rows
    monkeypatch.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))


@pytest.fixture(scope="session")
def cfg() -> BatoConfig:
    return load_config(REPO_CONFIG_DIR)


@pytest.fixture
def config_copy(tmp_path: Path) -> Callable[..., Path]:
    """Copy the repo config to a temp dir, optionally mutating one YAML file.

    Usage: ``config_copy("thresholds.yaml", lambda d: d["detection"].pop("debris_z_min"))``
    """
    def _make(filename: str | None = None, mutate: Callable[[dict], None] | None = None) -> Path:
        target = tmp_path / "config"
        shutil.copytree(REPO_CONFIG_DIR, target, dirs_exist_ok=True)
        if filename is not None and mutate is not None:
            path = target / filename
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            mutate(data)
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return target
    return _make

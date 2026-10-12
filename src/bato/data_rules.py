"""Data-rule guards (official PS: breaking the data rules means disqualification).

Allowed pipeline inputs: Sentinel-1, Sentinel-2, Copernicus DEM, OpenStreetMap
as it was before the event, and the listed training datasets.

Not allowed as inputs: Copernicus EMS, UNOSAT or any other published damage
map, and OSM edits made after the event. They may only be used to check
results, and they live under the evaluation-only folders.

Guards in this module:

* :func:`assert_pipeline_input_allowed` - every loader in pipeline code must
  pass its input path through this before opening it.
* The import-isolation test (tests/unit/test_data_rules.py) fails if any
  pipeline package imports :mod:`bato.evaluation`.
* :class:`~bato.contracts.common.DataSource` enumerates permitted sources only.
"""

from __future__ import annotations

from pathlib import Path

from bato.config import PathSettings
from bato.exceptions import ProhibitedInputError

# Every module under src/bato except this package is pipeline code and must
# not import it. Only bato.evaluation (plus tests and scripts) may touch
# reference data.
EVALUATION_PACKAGE = "bato.evaluation"


def assert_pipeline_input_allowed(path: Path, paths: PathSettings) -> Path:
    """Return the resolved path, or raise if it lies in an evaluation-only folder.

    Symlinks are resolved first, so a link from ``data/raw`` into
    ``data/reference`` is still caught.

    Raises:
        ProhibitedInputError: the path is inside ``paths.reference`` or ``paths.eval``.
    """
    resolved = Path(path).resolve()
    for forbidden in paths.evaluation_only_dirs():
        forbidden = forbidden.resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ProhibitedInputError(
                f"{resolved} is inside the evaluation-only folder {forbidden}; reference "
                "data (e.g. Copernicus EMS) must never be a pipeline input"
            )
    return resolved


def assert_reference_input_allowed(path: Path, paths: PathSettings) -> Path:
    """The reverse guard, for evaluation code: reference/evaluation data may
    only be read from the evaluation-only folders.

    This stops raw downloads (e.g. Sentinel-1 GRD under ``data/raw``) or any
    pipeline output from being loaded by mistake as a "reference" product.

    Raises:
        ProhibitedInputError: the path is outside ``paths.reference`` and ``paths.eval``.
    """
    resolved = Path(path).resolve()
    for allowed in paths.evaluation_only_dirs():
        allowed = allowed.resolve()
        if resolved == allowed or allowed in resolved.parents:
            return resolved
    raise ProhibitedInputError(
        f"{resolved} is not inside an evaluation-only folder; reference data must come from "
        f"{paths.reference} or {paths.eval}, never from pipeline inputs or outputs"
    )


def assert_raw_output_dir(path: Path, paths: PathSettings) -> Path:
    """Downloads are pipeline inputs: they must be written under ``paths.raw``
    and never into an evaluation-only folder (symlinks resolved).

    Raises:
        ProhibitedInputError: the directory is in an evaluation-only folder or outside ``paths.raw``.
    """
    resolved = assert_pipeline_input_allowed(path, paths)
    raw = paths.raw.resolve()
    if resolved != raw and raw not in resolved.parents:
        raise ProhibitedInputError(f"download directory {resolved} must be inside the raw-data folder {raw}")
    return resolved


def assert_interim_output_dir(path: Path, paths: PathSettings) -> Path:
    """Preprocessing outputs are pipeline intermediates: they must be written
    under ``paths.interim`` and never into an evaluation-only folder (symlinks resolved).

    Raises:
        ProhibitedInputError: the directory is in an evaluation-only folder or outside ``paths.interim``.
    """
    resolved = assert_pipeline_input_allowed(path, paths)
    interim = paths.interim.resolve()
    if resolved != interim and interim not in resolved.parents:
        raise ProhibitedInputError(f"output directory {resolved} must be inside the interim folder {interim}")
    return resolved


def assert_processed_output_dir(path: Path, paths: PathSettings) -> Path:
    """Analysis outputs (impact tables, graphs, evidence) must be written under ``paths.processed``
    and never into an evaluation-only folder (symlinks resolved).

    Raises:
        ProhibitedInputError: the directory is in an evaluation-only folder or outside ``paths.processed``.
    """
    resolved = assert_pipeline_input_allowed(path, paths)
    processed = paths.processed.resolve()
    if resolved != processed and processed not in resolved.parents:
        raise ProhibitedInputError(f"output directory {resolved} must be inside the processed folder {processed}")
    return resolved

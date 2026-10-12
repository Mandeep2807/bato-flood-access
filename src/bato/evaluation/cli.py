"""EVALUATION ONLY. Resolve what ``scripts/evaluate_emsr927.py --manifest`` points at, under the matching data rule.

``--manifest`` accepts exactly two kinds of file. Each goes through ONE guard, chosen by where the file is:

* **BATO fetch manifest** (a pipeline file, e.g. ``data/raw/manifests/<aoi>_fetch_manifest.json``).
  :func:`bato.data_rules.assert_pipeline_input_allowed` is applied unchanged, so anything inside an evaluation-only
  folder is refused there.
* **EMSR927 evaluation manifest**: the canonical ``data/reference/EMSR927/acquisition_manifest.json`` written by
  ``scripts/fetch_emsr927.py``. :func:`bato.data_rules.assert_reference_input_allowed` applies. The file must be
  exactly that path and declare ``kind: emsr927_reference_acquisition`` and ``data_role: evaluation_reference_only``.
  The BATO AOI and candidate it recorded (from the hash-verified official run) select the BATO result. Any other
  file under the reference folder is refused, including a product zip.

Neither path weakens a guard. The evaluation still re-verifies every zip against the acquisition manifest SHA-256,
re-checks that the manifest's BATO AOI equals the official run's, and runs the leakage scan
(:func:`bato.evaluation.emsr927.evaluate`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bato.config import BatoConfig
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import load_contract
from bato.data_rules import assert_pipeline_input_allowed, assert_reference_input_allowed
from bato.evaluation.acquisition import MANIFEST, reference_dir
from bato.exceptions import ProhibitedInputError
from bato.preprocessing.stack import select_candidate

EVALUATION_MANIFEST_KIND = "emsr927_reference_acquisition"
EVALUATION_MANIFEST_ROLE = "evaluation_reference_only"


@dataclass(frozen=True)
class EvaluationTarget:
    aoi_id: str
    candidate_id: str
    manifest_kind: str          # "bato_fetch_manifest" | "emsr927_evaluation_manifest"
    manifest_path: Path


def _inside(path: Path, folders: tuple[Path, ...]) -> bool:
    return any(path == f.resolve() or f.resolve() in path.parents for f in folders)


def resolve_target(cfg: BatoConfig, manifest: Path, candidate: str | None = None) -> EvaluationTarget:
    """The BATO AOI and candidate to evaluate, read from a BATO fetch manifest or the EMSR927 evaluation manifest.

    Raises:
        ProhibitedInputError: a file under an evaluation-only folder that is not the canonical EMSR927 evaluation
            manifest (e.g. a product zip), or a pipeline manifest that fails the pipeline guard.
        ValueError: an evaluation manifest of the wrong kind, or a ``candidate`` that contradicts it.
    """
    paths = cfg.settings.paths
    resolved = Path(manifest).resolve()
    if not _inside(resolved, paths.evaluation_only_dirs()):
        fetch = load_contract(FetchManifest, assert_pipeline_input_allowed(manifest, paths))
        cand = select_candidate(fetch, candidate)
        return EvaluationTarget(fetch.aoi_id, cand.candidate_id, "bato_fetch_manifest", resolved)

    canonical = (reference_dir(paths) / MANIFEST).resolve()
    assert_reference_input_allowed(resolved, paths)
    if resolved != canonical:
        raise ProhibitedInputError(
            f"{resolved} is evaluation-only reference data. The only evaluation-only file --manifest accepts is the "
            f"EMSR927 evaluation manifest {canonical}; reference products (e.g. zips) are never an input here")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if data.get("kind") != EVALUATION_MANIFEST_KIND or data.get("data_role") != EVALUATION_MANIFEST_ROLE:
        raise ValueError(f"{resolved.name} is not an EMSR927 evaluation manifest (kind {data.get('kind')!r}, "
                         f"data_role {data.get('data_role')!r})")
    bato = data.get("bato_aoi") or {}
    aoi_id, cand_id = bato.get("aoi_id"), bato.get("candidate_id")
    if not aoi_id or not cand_id:
        raise ValueError(f"{resolved.name} does not record the BATO aoi_id / candidate_id it was made for")
    if candidate is not None and candidate != cand_id:
        raise ValueError(f"--candidate {candidate} contradicts the evaluation manifest (made for {cand_id})")
    return EvaluationTarget(str(aoi_id), str(cand_id), "emsr927_evaluation_manifest", resolved)

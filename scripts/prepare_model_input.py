"""Prepare Kuro Siwo-compatible model inputs from the published Step 3.5 stack (Step 4A).

Input: data/interim/preprocessing/<aoi>/stack/<candidate_id>/ (never modified, nothing downloaded).
Output: data/interim/preprocessing/<aoi>/model_input/<candidate_id>/model_input_manifest.json
    the locked model-input specification (channel order, normalisation, tiling, validity policy, the
    evidence status of every element, unresolved items), source stack hashes, the 224 x 224 tile index
    with valid-pixel counts, and clamp-saturation diagnostics. No tensors are written: training and
    inference read tiles through bato.preprocessing.model_input.ModelInputReader.

Example:
    python scripts/prepare_model_input.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused (missing or changed stack, inconsistent values, ...),
2 rasterio unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.contracts.model_input import KURO_TILE_SIZE
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.preprocessing.model_input import describe_model_input, prepare_model_input, run_model_input
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4A: Kuro Siwo-compatible model-input manifest from the "
                                                 "Step 3.5 stack.")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None, help="candidate id (default: the manifest's only candidate)")
    parser.add_argument("--stride", type=int, default=KURO_TILE_SIZE,
                        help=f"tile stride in pixels, 1..{KURO_TILE_SIZE} (default {KURO_TILE_SIZE}: no overlap)")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing manifest (kept as <candidate>.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        require_rasterio()
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        job = prepare_model_input(cfg, manifest.aoi_id, candidate.candidate_id)
        print("Stage      : model input (Step 4A) - Kuro Siwo-compatible tensors from the Step 3.5 stack")
        print(f"Candidate  : {job.candidate_id} | track {job.provenance.track} | event {job.provenance.event_date}")
        for i in job.provenance.inputs:
            print(f"  {i.role:4} {i.pass_id} ({i.days_from_event:+.1f} d)")
        print(f"Stack      : {job.stack_path.relative_to(job.root).as_posix()} (hash verified)")
        print(f"Output     : {job.final_dir.relative_to(job.root).as_posix()}/\n")
        result = run_model_input(job, cfg, stride=args.stride, overwrite=args.overwrite, progress=progress)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    m = result.manifest
    print("\nModel input:")
    for line in describe_model_input(m):
        print(f"  {line}")
    print(f"\nManifest   : {result.manifest_path.relative_to(job.root).as_posix()}")
    print(f"Kuro Siwo  : {m.spec.kuro_siwo_compatibility}")
    for item in m.spec.unresolved_items:
        print(f"  OPEN  {item}")
    print("RESULT     : SUCCEEDED (model-input manifest; no tensors written, no model trained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

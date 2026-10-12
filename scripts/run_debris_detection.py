"""Deterministic SAR disturbance (debris) detection on the Step 3.5 stack (Milestone 4). No model, no training,
no reference data. NOT VALIDATED.

Input : the verified Step 3.5 stack + valid_mask.tif and the Step 4C run made from that stack (its state raster
        supplies the geometry gates - invalid SAR, no DEM, layover, shadow - and FLOOD / PRE_EVENT_WATER).
        Thresholds: detection.debris_z_min (z) and detection.min_component_pixels, unchanged.
Output: data/interim/detection/<aoi>/<candidate_id>/debris/
    debris_state.tif        uint8: 0 no disturbance, 1 DISTURBANCE, 10 invalid, 11 no DEM, 12 layover, 13 shadow
    change_table.npz        every DISTURBANCE and Step 4C FLOOD pixel with d1 / d2 / d0 (VV, VH), slope, s_r
    debris_provenance.json, debris_diagnostics.md (sigma values, pre-registered checks)

Example:
    python scripts/run_debris_detection.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused, 2 rasterio unavailable, 3 run failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.data_rules import assert_pipeline_input_allowed
from bato.debris.disturbance import DebrisError, describe_debris, prepare_debris, run_debris
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="SAR disturbance (debris) detection (deterministic).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing output (kept as <dir>.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        require_rasterio()
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        job = prepare_debris(cfg, manifest.aoi_id, candidate.candidate_id)
        if job.final_dir.exists() and not args.overwrite:
            raise DebrisError(f"output already exists: {job.final_dir} (use --overwrite)")
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    mi = job.model_input
    print("Stage      : SAR disturbance (debris) detection - deterministic, NOT VALIDATED")
    print(f"Candidate  : {mi.candidate_id} | track {mi.provenance.track} | event {mi.provenance.event_date}")
    print(f"Step 4C    : {job.baseline.outputs.state} (geometry gates, FLOOD / PRE_EVENT_WATER)")
    try:
        out_dir, prov = run_debris(job, cfg, overwrite=args.overwrite, progress=progress)
    except BatoError as exc:
        print(f"\nFAILED (nothing published): {exc}", file=sys.stderr)
        return 3
    print()
    for line in describe_debris(prov):
        print(line)
    print(f"\nOutput     : {out_dir.relative_to(mi.root).as_posix()}/")
    print("RESULT     : SUCCEEDED (NOT VALIDATED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

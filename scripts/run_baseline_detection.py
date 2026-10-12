"""Deterministic SAR change flood baseline on the Step 3.5 stack (Step 4C). No model, no training, no
reference data. FloodViT, if present, is only compared afterwards.

Input : the verified Step 3.5 stack + valid_mask.tif, the GLO-30 mosaic Step 3.4 used for the POST pass,
        thresholds from config/thresholds.yaml (baseline_flood, detection.min_component_pixels,
        visibility.layover_lia_max_deg).
Output: data/interim/detection/<aoi>/<candidate_id>/baseline/
    baseline_state.tif     uint8 states (0 NO_FLOOD, 1 FLOOD, 2 PRE_EVENT_WATER, 10-14 NOT_ASSESSED_*)
    baseline_flood.tif     uint8 1 flood, 0 no flood (assessed), 255 not assessed (nodata)
    baseline_provenance.json, baseline_diagnostics.md

Example:
    python scripts/run_baseline_detection.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

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
from bato.detection.baseline import BaselineError, describe_baseline, prepare_baseline, run_baseline
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4C: deterministic SAR change flood baseline.")
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
        job = prepare_baseline(cfg, manifest.aoi_id, candidate.candidate_id)
        if job.final_dir.exists() and not args.overwrite:
            raise BaselineError(f"output already exists: {job.final_dir} (use --overwrite)")
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
    print("Stage      : SAR change flood baseline (Step 4C) - deterministic, NOT VALIDATED")
    print(f"Candidate  : {mi.candidate_id} | track {mi.provenance.track} | event {mi.provenance.event_date}")
    print(f"DEM        : {job.dem_path if job.dem_path else 'NOT AVAILABLE (all pixels NOT_ASSESSED_NO_DEM)'}")
    fv = "compared after detection" if job.floodvit_prediction else "no event output (no comparison)"
    print(f"FloodViT   : {fv}")
    try:
        out_dir, prov = run_baseline(job, cfg, overwrite=args.overwrite, progress=progress)
    except BatoError as exc:
        print(f"\nFAILED (nothing published): {exc}", file=sys.stderr)
        return 3
    print()
    for line in describe_baseline(prov):
        print(line)
    print(f"\nOutput     : {out_dir.relative_to(mi.root).as_posix()}/")
    print("RESULT     : SUCCEEDED (baseline map NOT VALIDATED; NOT_ASSESSED is not 'no flood')")
    return 0


if __name__ == "__main__":
    sys.exit(main())

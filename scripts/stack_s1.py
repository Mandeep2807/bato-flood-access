"""Build the co-registered Sentinel-1 temporal stack PRE-1 / PRE-2 / POST (Step 3.5).

Input: the three published Step 3.4 outputs (data/interim/preprocessing/<aoi>/<pass>/terrain/) of
one temporal-set candidate in the Step 2C fetch manifest (never modified, nothing downloaded).
Output: data/interim/preprocessing/<aoi>/stack/<candidate_id>/
    stack_sigma0.tif   6 x float32, EPSG:3857 10 m: PRE1_VV PRE1_VH PRE2_VV PRE2_VH POST_VV POST_VH
                       (linear sigma0 copied unchanged: no dB, no normalisation, no resampling)
    valid_mask.tif     uint8, 1 = valid in all six layers
    preprocessing_provenance.json

Example:
    python scripts/stack_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused (missing pass output, grid or role mismatch, ...),
2 rasterio unavailable, 3 writing or validation failed (kept as <candidate>.failed-<time>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import StackError, StackRunFailed, describe_stack, prepare_stack, run_stack


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3.5: PRE-1/PRE-2/POST temporal stack from Step 3.4 outputs.")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None, help="candidate id (default: the manifest's only candidate)")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing stack (kept as <candidate>.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        require_rasterio()
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        job = prepare_stack(manifest, cfg, args.candidate)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1

    root, g = job.root, job.grid
    print("Stage      : stack (Step 3.5) - co-registered PRE-1/PRE-2/POST, linear sigma0, EPSG:3857 10 m")
    print(f"Candidate  : {job.candidate.candidate_id} | track {job.candidate.track.label} | event {job.event_date}")
    for i in job.inputs:
        days = (i.provenance.input.acquisition_start.date() - job.event_date).days
        print(f"  {i.role:4} {i.pass_id} ({days:+d} d)  {i.dim.relative_to(root).as_posix()}  "
              f"{i.product.width} x {i.product.height} px, {len(i.files_sha256)} files verified")
    print(f"Grid       : {g.width} x {g.height} px at {g.pixel_size_m:g} m, origin ({g.transform[2]}, "
          f"{g.transform[5]}), AOI bbox + {g.analysis_buffer_km:g} km")
    print(f"Output     : {job.final_dir.relative_to(root).as_posix()}/\n")
    try:
        result = run_stack(job, cfg, overwrite=args.overwrite, progress=progress)
    except StackRunFailed as exc:
        for line in describe_stack(exc.provenance):
            print(f"  {line}")
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 3
    except StackError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 1
    prov = result.provenance
    print("\nValidation:")
    for line in describe_stack(prov):
        print(f"  {line}")
    assert prov.output is not None
    print(f"\nStack      : {prov.output.stack} (sha256 {prov.output.stack_sha256[:12]}...)")
    print(f"Mask       : {prov.output.mask}")
    print(f"Provenance : {result.provenance_path.relative_to(root).as_posix()}")
    print("Open items : " + " | ".join(prov.open_specification_items))
    print("RESULT     : SUCCEEDED (temporal stack, linear sigma0, EPSG:3857 10 m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

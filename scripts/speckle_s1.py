"""Lee Sigma speckle filter on ONE pass's published Step 3.2 output (Step 3.3, SNAP gpt).

    Read (radiometric DIMAP) -> Speckle-Filter (Lee Sigma 7x7, sigma 0.9, target 3x3, 1 look) -> BEAM-DIMAP

Radar geometry, linear sigma0: no terrain correction, no dB, no normalisation. The input is the
published data/interim/preprocessing/<aoi_id>/<pass_id>/radiometric/ output (never modified).

Example:
    python scripts/speckle_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \\
        --pass-id S1D-A085-20260629T122138

Output: data/interim/preprocessing/<aoi_id>/<pass_id>/speckle/ (+ preprocessing_provenance.json)

Exit codes: 0 ok, 1 invalid input/config/refused, 2 SNAP or rasterio unavailable / preflight failed
(including a Lee Sigma parameter or value the installed SNAP does not accept), 3 processing or
validation failed (failed output kept as speckle.failed-<time>).
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
from bato.preprocessing.radiometric import PreprocessingError
from bato.preprocessing.sigma0_validation import ValidationSetupError
from bato.preprocessing.snap import SnapError, graph_path
from bato.preprocessing.speckle import (
    STAGE,
    SpeckleRunFailed,
    describe_speckle_validation,
    prepare_speckle_job,
    run_speckle,
)


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3.3: Lee Sigma speckle filter on a Step 3.2 output.")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--pass-id", required=True, help="e.g. S1D-A085-20260629T122138")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing output (kept as speckle.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        job = prepare_speckle_job(manifest, args.pass_id, cfg)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1

    s = cfg.settings.preprocessing.s1_speckle
    root = job.root
    rp = job.input_provenance
    template = graph_path(s.graph).resolve()
    shown = template.relative_to(root).as_posix() if root.resolve() in template.parents else template.as_posix()
    print(f"Stage      : {STAGE} (Step 3.3) - Lee Sigma on linear sigma0 VV/VH, radar geometry, no dB, no TC")
    print(f"Product    : {rp.input.product_name}")
    print(f"             pass {job.pass_id} | AOI {job.aoi_id} | {rp.input.orbit_direction.value} relative orbit "
          f"{rp.input.relative_orbit} | {rp.input.acquisition_start.isoformat()}")
    print(f"Input      : {job.input_dim.relative_to(root).as_posix()} ({job.input_product.width} x "
          f"{job.input_product.height} px, {len(job.input_hashes)} files hashed)")
    print(f"Graph      : {shown}")
    print(f"Lee Sigma  : filter '{s.filter}' | window {s.window_size} | sigma {s.sigma} | target window "
          f"{s.target_window_size} | looks {s.num_looks}")
    print(f"Output     : {job.final_dir.relative_to(root).as_posix()}/\n")

    try:
        result = run_speckle(job, cfg, overwrite=args.overwrite, progress=progress)
    except (SnapError, ValidationSetupError) as exc:
        print(f"\nPREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except SpeckleRunFailed as exc:
        if exc.provenance.validation is not None:
            print("\nValidation:")
            for line in describe_speckle_validation(exc.provenance.validation):
                print(f"  {line}")
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 3
    except PreprocessingError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 1

    prov = result.provenance
    assert prov.validation is not None   # guaranteed for a successful run
    print("\nValidation:")
    for line in describe_speckle_validation(prov.validation):
        print(f"  {line}")
    for w in prov.warnings:
        print(f"  WARN  {w}")
    print(f"\nSNAP {prov.snap.snap_version or '(version unknown)'} | gpt runtime {prov.snap.runtime_s} s")
    print(f"Output     : {result.output_dir.relative_to(root).as_posix()}/")
    print(f"Provenance : {result.provenance_path.relative_to(root).as_posix()}")
    print("RESULT     : SUCCEEDED (Lee Sigma filtered linear sigma0 VV/VH, radar geometry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

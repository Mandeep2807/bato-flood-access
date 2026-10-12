"""Preprocess ONE downloaded Sentinel-1 GRD pass with SNAP gpt (Step 3.2: radiometric stage).

    Read -> Apply-Orbit-File -> Subset (AOI + preprocessing.crop_buffer_m) -> ThermalNoiseRemoval
         -> Remove-GRD-Border-Noise -> Calibration (linear sigma0 VV/VH) -> BEAM-DIMAP

Radar geometry only: no speckle filter, no terrain correction, no dB conversion.
Uses the product already downloaded by Step 2C (never re-downloads). The AOI
comes from the Step 2A catalogue the manifest was planned from (found
automatically by its SHA-256 digest under data/interim/catalogue or the project
root, or given with --catalogue).

Example:
    python scripts/preprocess_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \\
        --pass-id S1D-A085-20260629T122138 --stage radiometric

Output: data/interim/preprocessing/<aoi_id>/<pass_id>/radiometric/ (+ preprocessing_provenance.json)

SNAP gpt is found via snap.gpt_path in config/config.yaml, then the BATO_SNAP_GPT
environment variable, then the default install location, then PATH.

Exit codes: 0 ok, 1 invalid input/config/refused, 2 SNAP or rasterio unavailable / preflight failed,
3 processing or validation failed (failed output kept as radiometric.failed-<time>).
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
from bato.preprocessing.radiometric import (
    STAGE,
    PreprocessingError,
    PreprocessingRunFailed,
    describe_validation,
    prepare_job,
    run_radiometric,
)
from bato.preprocessing.sigma0_validation import ValidationSetupError
from bato.preprocessing.snap import SnapError, graph_path


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3.2: one Sentinel-1 GRD pass -> linear sigma0 (SNAP gpt).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--pass-id", required=True, help="e.g. S1D-A085-20260629T122138")
    parser.add_argument("--stage", choices=[STAGE], required=True, help="only 'radiometric' exists in Step 3.2")
    parser.add_argument("--catalogue", type=Path, default=None,
                        help="Step 2A catalogue JSON (default: found by digest in data/interim/catalogue or root)")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing output (kept as radiometric.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        job = prepare_job(manifest, args.pass_id, cfg, args.catalogue)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1

    r = cfg.settings.preprocessing.s1_radiometric
    b, c = job.aoi_bbox, job.crop_bbox
    root = job.root
    print(f"Stage      : {STAGE} (Step 3.2) - linear sigma0 VV/VH, radar geometry, no terrain correction, no dB")
    print(f"Product    : {job.scene.product_name}")
    print(f"             pass {job.pass_id} | {job.scene.orbit_direction.value} relative orbit "
          f"{job.scene.relative_orbit} | {job.scene.acquisition_start.isoformat()}")
    size_gb = job.input_provenance.size_bytes / 1e9
    print(f"Input      : {job.input_path.relative_to(root).as_posix()} ({size_gb:.2f} GB, "
          f"sha256 {job.input_provenance.sha256[:12]}... from download provenance)")
    print(f"SAFE       : {job.safe.bands[0].samples} x {job.safe.bands[0].lines} px, "
          f"{'+'.join(x.polarization.value for x in job.safe.bands)}, provenance consistent")
    print(f"AOI        : {job.aoi_id} bbox lon {b.min_lon:.4f}..{b.max_lon:.4f} lat {b.min_lat:.4f}..{b.max_lat:.4f} "
          f"(from {job.catalogue_path.name})")
    print(f"Crop       : AOI bbox + {cfg.settings.preprocessing.crop_buffer_m:g} m -> "
          f"lon {c.min_lon:.4f}..{c.max_lon:.4f} lat {c.min_lat:.4f}..{c.max_lat:.4f}; input scene AOI coverage "
          f"{job.input_scene_aoi_coverage if job.input_scene_aoi_coverage is not None else 'unknown'}")
    template = graph_path(r.graph).resolve()
    shown = template.relative_to(root).as_posix() if root.resolve() in template.parents else template.as_posix()
    print(f"Graph      : {shown}")
    print(f"Parameters : orbit '{r.orbit_type}' polyDegree {r.orbit_poly_degree} continueOnFail "
          f"{str(r.orbit_continue_on_fail).lower()} | thermal noise remove {str(r.remove_thermal_noise).lower()} "
          f"reintroduce {str(r.reintroduce_thermal_noise).lower()} | border limit {r.border_limit} trim "
          f"{r.trim_threshold:g} | sigma0 linear (dB {str(r.output_in_db).lower()})")
    print(f"Output     : {job.final_dir.relative_to(root).as_posix()}/\n")

    try:
        result = run_radiometric(job, cfg, overwrite=args.overwrite, progress=progress)
    except (SnapError, ValidationSetupError) as exc:
        print(f"\nPREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except PreprocessingRunFailed as exc:
        if exc.provenance.validation is not None:
            print("\nValidation:")
            for line in describe_validation(exc.provenance.validation):
                print(f"  {line}")
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 3
    except PreprocessingError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 1

    prov = result.provenance
    assert prov.validation is not None   # guaranteed for a successful run
    print("\nValidation:")
    for line in describe_validation(prov.validation):
        print(f"  {line}")
    print(f"\nSNAP {prov.snap.snap_version or '(version unknown)'} | gpt runtime {prov.snap.runtime_s} s")
    print(f"Output     : {result.output_dir.relative_to(root).as_posix()}/")
    print(f"Provenance : {result.provenance_path.relative_to(root).as_posix()}")
    print("RESULT     : SUCCEEDED (linear sigma0 VV/VH, radar geometry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

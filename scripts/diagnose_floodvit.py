"""Forensic diagnostics of the published FloodViT outputs (Step 4B): what the model's flood / permanent-water
classes respond to. Read-only on all inputs; no torch needed; no reference/evaluation data is used.

Input : the verified Step 3.5 stack + valid_mask.tif, every published FloodViT mode under
        data/interim/inference/<aoi>/<candidate_id>/floodvit*/ (event, no_event, zero_change, static_post),
        and the GLO-30 mosaic Step 3.4 used for the POST pass (hash-checked).
Output: data/interim/inference/<aoi>/<candidate_id>/diagnostics/floodvit_diagnostics.{json,md}

Example (run all FloodViT modes first):
    python scripts/run_floodvit_inference.py --manifest <manifest> --checkpoint <pt> --kuro-repo <KuroSiwo> --mode all
    python scripts/diagnose_floodvit.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused, 2 rasterio unavailable.
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
from bato.inference.diagnostics import NOMINAL_INCIDENCE_DEG, REPORT_MD, prepare_diagnostics, run_diagnostics
from bato.logging_utils import configure_logging
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4B forensics: FloodViT output diagnostics (no validation).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--incidence", type=float, default=NOMINAL_INCIDENCE_DEG,
                        help="nominal incidence angle for the approximate radar-geometry classes (deg)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        require_rasterio()
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        job = prepare_diagnostics(cfg, manifest.aoi_id, candidate.candidate_id)
        print(f"Modes      : {', '.join(m.mode for m in job.modes)}")
        print(f"DEM        : {job.dem_path if job.dem_path else 'NOT AVAILABLE (terrain metrics reported as no_dem)'}")
        out_dir, _report = run_diagnostics(job, cfg, incidence=args.incidence, overwrite=args.overwrite,
                                           progress=progress)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    print()
    print((out_dir / REPORT_MD).read_text(encoding="utf-8"))
    print(f"Written    : {out_dir.relative_to(job.model_input.root).as_posix()}/")
    print("RESULT     : DIAGNOSTICS ONLY (no validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

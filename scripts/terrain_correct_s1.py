"""Range-Doppler terrain correction of ONE pass's published Step 3.3 output (Step 3.4, SNAP gpt).

    Read (speckle DIMAP) -> Terrain-Correction (Copernicus DEM, EPSG:3857, 10 m, bilinear) -> BEAM-DIMAP

Linear sigma0 (no dB, no normalisation). The input is the published
data/interim/preprocessing/<aoi_id>/<pass_id>/speckle/ output (never modified); the DEM is
mosaicked from the Step 2C Copernicus DEM tiles listed in the manifest (never downloaded here).

Example:
    python scripts/terrain_correct_s1.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \\
        --pass-id S1D-A085-20260629T122138

Output: data/interim/preprocessing/<aoi_id>/<pass_id>/terrain/ (+ preprocessing_provenance.json)

Exit codes: 0 ok, 1 invalid input/config/refused (incl. missing DEM tiles), 2 SNAP or rasterio
unavailable / preflight failed (a parameter or value the installed SNAP does not accept), 3 DEM
preparation, processing or validation failed (failed output kept as terrain.failed-<time>).
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
from bato.preprocessing.terrain import (
    STAGE,
    TerrainRunFailed,
    describe_terrain_validation,
    prepare_terrain_job,
    run_terrain,
)


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3.4: terrain correction of a Step 3.3 output.")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--pass-id", required=True, help="e.g. S1D-A085-20260629T122138")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing output (kept as terrain.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        job = prepare_terrain_job(manifest, args.pass_id, cfg)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1

    t = cfg.settings.preprocessing.s1_terrain
    root = job.root
    sp = job.input_provenance
    template = graph_path(t.graph).resolve()
    shown = template.relative_to(root).as_posix() if root.resolve() in template.parents else template.as_posix()
    fp = job.input_footprint.exterior
    print(f"Stage      : {STAGE} (Step 3.4) - Range-Doppler terrain correction, EPSG:3857 10 m, linear sigma0, no dB")
    print(f"Product    : {sp.input.product_name}")
    print(f"             pass {job.pass_id} | AOI {job.aoi_id} | {sp.input.orbit_direction.value} relative orbit "
          f"{sp.input.relative_orbit} | {sp.input.acquisition_start.isoformat()}")
    print(f"Input      : {job.input_dim.relative_to(root).as_posix()} ({job.input_product.width} x "
          f"{job.input_product.height} px, {len(job.input_hashes)} files hashed)")
    print(f"Footprint  : lon {min(p[0] for p in fp):.4f}..{max(p[0] for p in fp):.4f} "
          f"lat {min(p[1] for p in fp):.4f}..{max(p[1] for p in fp):.4f}")
    print(f"DEM        : {cfg.settings.dem.cdse_dataset} tiles {[s.tile_id for s in job.dem_sources]} from Step 2C "
          f"(re-hashed OK), margin {t.dem_margin_km:g} km, EGM {str(t.dem_apply_egm).lower()}, no-data {t.dem_nodata:g}"
          + (f"; cells without a tile: {list(job.dem_sea_tiles)}" if job.dem_sea_tiles else ""))
    print(f"Graph      : {shown}")
    print(f"Output     : {job.final_dir.relative_to(root).as_posix()}/\n")

    try:
        result = run_terrain(job, cfg, overwrite=args.overwrite, progress=progress)
    except (SnapError, ValidationSetupError) as exc:
        print(f"\nPREFLIGHT FAILED (nothing written): {exc}", file=sys.stderr)
        return 2
    except TerrainRunFailed as exc:
        if exc.provenance.validation is not None:
            print("\nValidation:")
            for line in describe_terrain_validation(exc.provenance.validation):
                print(f"  {line}")
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 3
    except PreprocessingError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 1

    prov = result.provenance
    assert prov.validation is not None and prov.dem is not None   # guaranteed for a successful run
    print("\nValidation:")
    for line in describe_terrain_validation(prov.validation):
        print(f"  {line}")
    for w in prov.warnings:
        print(f"  WARN  {w}")
    print(f"\nDEM mosaic : {prov.dem.path} ({prov.dem.width} x {prov.dem.height}, sha256 {prov.dem.sha256[:12]}...)")
    print(f"SNAP {prov.snap.snap_version or '(version unknown)'} | gpt runtime {prov.snap.runtime_s} s")
    print(f"Output     : {result.output_dir.relative_to(root).as_posix()}/")
    print(f"Provenance : {result.provenance_path.relative_to(root).as_posix()}")
    print("RESULT     : SUCCEEDED (terrain-corrected linear sigma0 VV/VH, EPSG:3857 10 m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

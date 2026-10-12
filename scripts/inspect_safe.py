"""Inspect downloaded Sentinel-1 GRD SAFE zips (Step 3.1, read-only).

Reads the Step 2C fetch manifest, opens every present Sentinel-1 product's
annotation XML (not the rasters), checks VV/VH completeness and grid
consistency, and cross-checks each product with its provenance sidecar.
Nothing is written.

Example:
    python scripts/inspect_safe.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 all products OK, 1 invalid input/config, 3 at least one product failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import DownloadProvenance, FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.exceptions import BatoError, ConfigError
from bato.preprocessing.safe_reader import check_against_provenance, inspect_safe_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect downloaded Sentinel-1 GRD SAFE zips (read-only).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C fetch manifest (not the dry-run plan)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        manifest = load_contract(FetchManifest, args.manifest)
    except (ConfigError, ContractIOError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    if manifest.dry_run:
        print("INVALID INPUT: this is a dry-run plan; pass the *_fetch_manifest.json of a real download",
              file=sys.stderr)
        return 1

    root = cfg.config_dir.parent
    pols = tuple(cfg.settings.sentinel1.polarizations)
    files = [f for f in manifest.files if f.kind == "sentinel1_grd" and f.present]
    print(f"AOI {manifest.aoi_id}: {len(files)} Sentinel-1 product(s) present in the manifest\n")
    failures = 0
    for f in files:
        assert f.local_path is not None and f.provenance_path is not None  # guaranteed for present files
        try:
            product = inspect_safe_zip(root / f.local_path, cfg.settings.paths, pols)
            check_against_provenance(product, load_contract(DownloadProvenance, root / f.provenance_path))
        except (BatoError, ContractIOError) as exc:
            failures += 1
            print(f"FAILED {f.product_name}\n       {exc}\n")
            continue
        b = product.bands[0]
        print(f"OK     {product.product_name}")
        print(f"       pass {f.pass_id} | {product.orbit_direction.value} | "
              f"{'+'.join(x.polarization.value for x in product.bands)} on one grid: "
              f"{b.samples} x {b.lines} px, {b.range_pixel_spacing_m:g} x {b.azimuth_pixel_spacing_m:g} m")
        for band in product.bands:
            print(f"       {band.polarization.value}: {band.measurement_size_bytes / 1e6:,.0f} MB raster, "
                  f"{band.geolocation_points} geolocation points, {band.calibration_vectors} calibration vectors x "
                  f"{band.calibration_points_per_vector} values")
        print("       provenance: consistent\n")
    print(f"{len(files) - failures}/{len(files)} product(s) OK")
    return 3 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

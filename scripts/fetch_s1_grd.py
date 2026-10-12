"""Fetch Sentinel-1 GRD products (and Copernicus DEM tiles) for selected temporal sets (Step 2C).

DRY RUN BY DEFAULT: prints the plan (and, unless --offline, product sizes from
the public CDSE catalogue) and writes <raw>/manifests/<aoi>_fetch_plan.json.
Nothing is downloaded and no credentials are needed.

Add --download to fetch. Requires CDSE_USERNAME and CDSE_PASSWORD in the
environment (see .env.example). Files go to <raw>/sentinel1 and
<raw>/copernicus_dem, each with a .provenance.json sidecar; the run manifest goes
to <raw>/manifests/<aoi>_fetch_manifest.json. Re-running resumes partial files
and skips verified ones.

Examples:
    python scripts/fetch_s1_grd.py --selection trishuli_sets.json --catalogue trishuli_s1.json
    python scripts/fetch_s1_grd.py --selection trishuli_sets.json --catalogue trishuli_s1.json \\
        --track ASCENDING/85 --max-scenes 1 --download

Exit codes: 0 ok, 1 invalid input/config/refused, 2 provider/auth error, 3 some files failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.acquisition.cdse_auth import TokenProvider
from bato.acquisition.cdse_odata import ProductMetadata, fetch_product_metadata
from bato.acquisition.dem import aoi_bbox_from_catalogue, dem_search_bbox, plan_dem, required_tile_ids
from bato.acquisition.downloader import Downloader
from bato.acquisition.http import UrllibStreamingTransport
from bato.acquisition.s1_fetch import FetchPlanError, build_s1_fetch_plan, execute_fetch, save_manifest
from bato.config import BatoConfig, load_config
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.fetch import DEMPlan, FetchManifest, FetchStatus, S1FetchPlan
from bato.contracts.io import ContractIOError, load_contract
from bato.contracts.temporal import TemporalSelectionResult
from bato.exceptions import BatoError, ConfigError, ProviderError
from bato.logging_utils import configure_logging


def sizes(plan: S1FetchPlan, dem: DEMPlan | None, cfg: BatoConfig) -> dict[str, ProductMetadata | str]:
    ids = [s.product_id for s in plan.scenes] + [t.product_id for t in (dem.tiles if dem else ()) if t.product_id]
    result: dict[str, ProductMetadata | str] = {}
    for pid in ids:
        try:
            result[pid] = fetch_product_metadata(pid, cfg.settings.cdse)
        except ProviderError as exc:
            result[pid] = f"unavailable ({exc})"
    return result


def gb(meta: ProductMetadata | str | None) -> str:
    if isinstance(meta, ProductMetadata) and meta.content_length:
        return f"{meta.content_length / 1e9:.2f} GB"
    return "?"


def print_plan(plan: S1FetchPlan, dem: DEMPlan | None, required_dem: tuple[str, ...],
               meta: dict[str, ProductMetadata | str]) -> None:
    print(f"\nFetch plan | AOI {plan.aoi_id} | event {plan.event_date} | {len(plan.candidates)} candidate(s), "
          f"{len(plan.passes)} pass(es), {len(plan.scenes)} Sentinel-1 product(s)")
    for c in plan.candidates:
        print(f"  {c.track.label} rank {c.rank} ({c.candidate_id}): PRE-1 {c.pre_event_pass_ids[0]}, "
              f"PRE-2 {c.pre_event_pass_ids[1]}, POST {c.post_event_pass_id}")
    print(f"\n{'pass':28} {'role':10} {'slice':5} {'size':>8}  product")
    passes = {p.pass_id: p for p in plan.passes}
    total = 0
    for s in plan.scenes:
        m = meta.get(s.product_id)
        if isinstance(m, ProductMetadata) and m.content_length:
            total += m.content_length
        offline = " OFFLINE" if isinstance(m, ProductMetadata) and m.online is False else ""
        print(f"{s.pass_id:28} {passes[s.pass_id].role.value:10} {s.slice_index + 1}/{s.slice_count:<3} "
              f"{gb(m):>8}  {s.product_name}{offline}")
    unknown = sum(not isinstance(meta.get(s.product_id), ProductMetadata) for s in plan.scenes)
    if unknown == len(plan.scenes):
        print("\nSentinel-1 total: unknown (sizes not queried; GRDH products are typically 1-2 GB each)")
    else:
        print(f"\nSentinel-1 total: {total / 1e9:.2f} GB" + (f" + {unknown} unknown" if unknown else ""))
    if dem is not None:
        print(f"DEM ({dem.dataset}, AOI + {dem.buffer_km:g} km): tiles {', '.join(dem.required_tiles)}")
        for t in dem.tiles:
            print(f"  {t.tile_id}: {t.product_name or 'NO PRODUCT FOUND'} {gb(meta.get(t.product_id or ''))}")
    elif required_dem:
        print(f"DEM tiles required (not searched, --offline/--no-dem): {', '.join(required_dem)}")


def print_manifest(manifest: FetchManifest, path: Path) -> None:
    counts: dict[str, int] = {}
    for f in manifest.files:
        counts[f.status.value] = counts.get(f.status.value, 0) + 1
    print(f"\nFiles: {counts}")
    for f in manifest.files:
        if f.status is FetchStatus.FAILED:
            print(f"  FAILED {f.product_name}: {f.message}")
    print(f"Passes complete: {sum(manifest.pass_complete.values())}/{len(manifest.pass_complete)} "
          f"| candidates ready: {sum(manifest.candidate_ready.values())}/{len(manifest.candidate_ready)}")
    for pid, ok in manifest.pass_complete.items():
        if not ok:
            print(f"  incomplete pass: {pid}")
    print(f"manifest: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Sentinel-1 GRD + Copernicus DEM for selected sets.")
    parser.add_argument("--selection", type=Path, required=True, help="output of select_temporal_sets.py --save")
    parser.add_argument("--catalogue", type=Path, required=True, help="output of s1_catalogue_query.py --save")
    parser.add_argument("--track", action="append", help="e.g. ASCENDING/85 (repeatable; default: all tracks)")
    parser.add_argument("--rank", action="append", type=int, help="candidate rank(s) per track (default: 1)")
    parser.add_argument("--download", action="store_true", help="actually download (default: dry run)")
    parser.add_argument("--max-scenes", type=int, default=None, help="download at most N Sentinel-1 products")
    parser.add_argument("--no-dem", action="store_true", help="skip Copernicus DEM tiles")
    parser.add_argument("--offline", action="store_true", help="dry run without contacting CDSE")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, None)
        selection = load_contract(TemporalSelectionResult, args.selection)
        catalogue = load_contract(CatalogueSearchResult, args.catalogue)
        plan = build_s1_fetch_plan(selection, catalogue, cfg, tracks=args.track, ranks=args.rank or [1])
    except (ConfigError, ContractIOError, FetchPlanError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    if args.offline and args.download:
        print("INVALID INPUT: --offline cannot be combined with --download", file=sys.stderr)
        return 1

    aoi_box = aoi_bbox_from_catalogue(catalogue)
    required_dem = required_tile_ids(dem_search_bbox(aoi_box, cfg.settings.dem.buffer_km))
    dem_plan = None
    meta: dict[str, ProductMetadata | str] = {}
    try:
        if not args.no_dem and not args.offline:
            dem_plan = plan_dem(catalogue.aoi_id, aoi_box, cfg)
        if not args.download and not args.offline:
            meta = sizes(plan, dem_plan, cfg)
    except ProviderError as exc:
        print(f"CATALOGUE QUERY FAILED: {exc}", file=sys.stderr)
        return 2
    print_plan(plan, dem_plan, required_dem, meta)

    if not args.download:
        manifest = execute_fetch(plan, dem_plan, cfg, dry_run=True)
        print(f"\nDRY RUN - nothing downloaded. Plan saved: {save_manifest(manifest, cfg)}")
        print("Add --download (with CDSE_USERNAME / CDSE_PASSWORD set) to fetch.")
        return 0

    cap = cfg.settings.download.max_scenes_per_run
    if args.max_scenes is None and len(plan.scenes) > cap:
        print(f"REFUSED: {len(plan.scenes)} products exceed download.max_scenes_per_run={cap}; "
              "pass --max-scenes N explicitly (or narrow with --track/--rank).", file=sys.stderr)
        return 1
    tokens = TokenProvider(cfg.settings.cdse)
    try:
        tokens.get()  # fail fast on missing/wrong credentials, before any download starts
    except BatoError as exc:
        print(f"AUTHENTICATION FAILED: {exc}", file=sys.stderr)
        return 2
    downloader = Downloader(cfg.settings.download,
                            UrllibStreamingTransport(cfg.settings.download.allowed_redirect_suffixes), tokens)
    try:
        manifest = execute_fetch(plan, dem_plan, cfg, dry_run=False, downloader=downloader,
                                 max_scenes=args.max_scenes)
    except BatoError as exc:
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        return 2
    print_manifest(manifest, save_manifest(manifest, cfg))
    return 3 if any(f.status is FetchStatus.FAILED for f in manifest.files) else 0


if __name__ == "__main__":
    sys.exit(main())

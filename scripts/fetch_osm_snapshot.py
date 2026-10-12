"""Fetch the pre-event OpenStreetMap extract for one AOI from the ohsome API v2 (historical OSM, GeoParquet).

Needs an ohsome API key in the environment variable named by osm.api_key_env (default OHSOME_API_KEY;
free key: https://account.heigit.org/signup) and pyarrow. --dry-run needs neither and contacts nothing.

Snapshot date = event date - osm.snapshot_offset_days (>= 1), capped at the latest date the ohsome database
covers, and always strictly before the event. The AOI box, analysis box (AOI + aoi.analysis_buffer_km) and
event date come from the published Step 3.5 stack of the manifest's candidate, so the road graph matches the
SAR grid exactly. Layers: roads (geometry), places, hospitals, building centroids (see bato.osm.extract).

Output: data/raw/osm/<aoi_id>/snapshot-<date>/{roads,places,hospitals,buildings}.parquet
        + osm_extract_provenance.json (request parameters, SHA-256, counts). Published atomically.

Examples:
    python scripts/fetch_osm_snapshot.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json --dry-run
    python scripts/fetch_osm_snapshot.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused, 2 ohsome request failed (nothing published).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.contracts.osm import select_snapshot_date
from bato.contracts.stack import StackProvenance
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.osm.extract import (
    OSMError,
    UrllibOhsomeTransport,
    api_key_from_env,
    bbox_param,
    extract_osm,
    plan_requests,
    require_pyarrow,
)
from bato.preprocessing.model_input import check_stack_provenance
from bato.preprocessing.radiometric import PROVENANCE_FILE, stage_dir
from bato.preprocessing.stack import STACK_GROUP, select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the pre-event OSM extract (ohsome API).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the requests; contact nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="supersede an existing extract (kept as <dir>.superseded-<time>, never deleted)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, cfg.settings.logging.jsonl_file)
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        prov_path = stage_dir(cfg, manifest.aoi_id, STACK_GROUP, candidate.candidate_id) / PROVENANCE_FILE
        if not prov_path.is_file():
            raise OSMError(f"no published Step 3.5 stack provenance at {prov_path} (run scripts/stack_s1.py first)")
        stack = load_contract(StackProvenance, prov_path)
        check_stack_provenance(stack, manifest.aoi_id, candidate.candidate_id)
        assert stack.grid is not None
        event_date = date.fromisoformat(stack.event_date)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    grid = stack.grid
    osm = cfg.settings.osm
    rule_snapshot = select_snapshot_date(event_date, osm.snapshot_offset_days)
    key_set = bool(os.environ.get(osm.api_key_env, "").strip())
    print("Stage      : pre-event OSM extract (ohsome API v2, GeoParquet) - no population data, no post-event edits")
    print(f"AOI        : {manifest.aoi_id} | event {event_date} | rule snapshot {rule_snapshot} "
          "(capped at the latest ohsome date)")
    print(f"Boxes      : analysis {bbox_param(grid.analysis_bbox)} (AOI + {grid.analysis_buffer_km:g} km)")
    print(f"API key    : {osm.api_key_env} is {'SET' if key_set else 'NOT SET'} (sent as the 'authorization' header)")
    for r in plan_requests(cfg, grid.aoi_bbox, grid.analysis_bbox):
        print(f"  {r.layer.value:9} POST {osm.api_base_url}/{r.endpoint} {json.dumps(r.body(rule_snapshot))}"
              f"{'  -> client-side centroids' if r.centroids else ''}")
    if args.dry_run:
        print("\nDRY RUN - nothing contacted, nothing written.")
        return 0
    try:
        api_key = api_key_from_env(cfg)
        require_pyarrow()
    except OSMError as exc:
        print(f"INVALID SETUP (nothing contacted): {exc}", file=sys.stderr)
        return 1
    try:
        out_dir, prov = extract_osm(cfg, aoi_id=manifest.aoi_id, event_date=event_date, aoi_bbox=grid.aoi_bbox,
                                    analysis_bbox=grid.analysis_bbox, transport=UrllibOhsomeTransport(),
                                    api_key=api_key, overwrite=args.overwrite, progress=progress)
    except OSMError as exc:
        print(f"\nFAILED (nothing published): {exc}", file=sys.stderr)
        return 1 if "already exists" in str(exc) else 2
    s = prov.snapshot
    print(f"\nSnapshot   : {s.snapshot_date} (event {s.event_date}; ohsome data up to {s.latest_available})")
    for layer in prov.layers:
        print(f"  {layer.layer.value:9} {layer.feature_count:>9,} features  {layer.file}")
    print(f"Output     : {out_dir}")
    print("RESULT     : SUCCEEDED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

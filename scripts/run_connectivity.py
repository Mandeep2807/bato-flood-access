"""Road connectivity before vs after the event (Milestone 8): road states, settlement access states, blocked
stretches ranked by settlements reconnected, and evidence records. Deterministic; NOT VALIDATED.

Input : the Step 4C baseline (baseline_state.tif + provenance, hash-chained to the Step 3.5 stack provenance),
        the pre-event OSM extract (scripts/fetch_osm_snapshot.py) and, for road rule v2 (default), the disturbance
        run (scripts/run_debris_detection.py) of the same stack. Nothing is downloaded.
Output: data/processed/connectivity/<aoi>/<candidate_id>/
    road_segments.geojson         every ~50 m segment: OPEN / BLOCKED / UNOBSERVABLE, lengths, evidence ids,
                                  and for BLOCKED segments the stretch's "clear it -> N settlements reconnected"
    settlements.geojson           access state + pre / optimistic / pessimistic routes, separating segments
    access_destinations.geojson   hospitals, towns, analysis exits
    blocked_segments_ranked.json  blocked stretches ranked by settlements reconnected if cleared alone
    evidence_records.jsonl        one EvidenceRecord per claim
    connectivity_provenance.json  inputs + hashes, thresholds, rules, summary
    connectivity_report.md        summary tables and evidence cards

Example:
    python scripts/run_connectivity.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

Exit codes: 0 ok, 1 invalid input / refused, 2 rasterio unavailable, 3 run failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.config import load_config
from bato.connectivity.engine import ConnectivityError, prepare_connectivity, run_connectivity
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.logging_utils import configure_logging
from bato.preprocessing.sigma0_validation import ValidationSetupError, require_rasterio
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Road connectivity, settlement access and blocked-road priority.")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--road-rule", choices=["v1", "v2"], default="v2",
                        help="v2 (default): FLOOD or DISTURBANCE blocks; only radar geometry makes a road "
                             "UNOBSERVABLE (needs scripts/run_debris_detection.py). v1: Step 4C FLOOD only; every "
                             "NOT_ASSESSED pixel makes a road UNOBSERVABLE")
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
        job = prepare_connectivity(cfg, manifest.aoi_id, candidate.candidate_id, road_rule=args.road_rule)
        if job.final_dir.exists() and not args.overwrite:
            raise ConnectivityError(f"output already exists: {job.final_dir} (use --overwrite)")
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except ValidationSetupError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    snap = job.osm.provenance.snapshot
    print("Stage      : road connectivity (Milestone 8) - deterministic, NOT VALIDATED")
    print(f"Candidate  : {job.candidate_id} | event {job.event_date} | OSM snapshot {snap.snapshot_date} (pre-event)")
    print(f"Road rule  : {args.road_rule}" + (f" (disturbance run {job.debris.provenance.outputs.directory})"
                                               if job.debris is not None else ""))
    try:
        out_dir, prov, _result = run_connectivity(job, cfg, overwrite=args.overwrite, progress=progress)
    except BatoError as exc:
        print(f"\nFAILED (nothing published): {exc}", file=sys.stderr)
        return 3
    s = prov.summary
    print("\nRoads (segments / km):")
    for k in s.road_segments_by_state:
        print(f"  {k.upper():13} {s.road_segments_by_state[k]:>8,}  {s.road_km_by_state[k]:>9,.1f} km")
    print(f"  UNOBSERVABLE by reason (km): {s.unobservable_km_by_reason}")
    print(f"Destinations: {s.targets_by_kind} | exits {s.exit_nodes} | unsnapped {s.unsnapped_targets}")
    print("Settlements by access state (mapped buildings = proxy, not population):")
    for k, v in s.settlements_by_access_state.items():
        print(f"  {k:20} {v:>6,}  ({s.building_count_by_access_state[k]:,} buildings)")
    print(f"Reachable settlements pre / optimistic / pessimistic: {s.reachable_settlements}")
    print(f"Blocked: {s.blocked_segments:,} segments in {s.blocked_stretches:,} stretches; "
          f"{s.blocked_stretches_reconnecting_alone:,} reconnect settlements on their own")
    for p in prov.top_blocked_segments[:10]:
        print(f"  #{p.rank:<3} {p.stretch_id:10} {len(p.segment_ids):>3} seg  ways {list(p.osm_way_ids)[:3]}  "
              f"reconnects {p.settlements_reconnected} ({p.reconnected_building_count} buildings); observed-only "
              f"{p.settlements_reconnected_on_observed_roads}; in min. clearing sets {p.in_minimum_clearing_set_of}")
    for w in prov.warnings:
        print(f"WARNING    : {w}")
    print(f"\nOutput     : {out_dir.relative_to(job.root).as_posix()}/")
    print("RESULT     : SUCCEEDED (NOT VALIDATED; UNOBSERVABLE is not 'open')")
    return 0


if __name__ == "__main__":
    sys.exit(main())

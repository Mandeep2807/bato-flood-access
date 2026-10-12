"""EVALUATION ONLY. Acquire the Copernicus EMS EMSR927 reference products, with provenance.

    python scripts/fetch_emsr927.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json [--dry-run]

What it does:

1. Retrieves the official activation record and saves it byte-for-byte under
   ``data/reference/EMSR927/activation/``, with its retrieval time and SHA-256. The API URL is in
   ``config/evaluation/emsr927_protocol.yaml``.
2. Takes every product's download URL from the record (``downloadPath``); URLs are never built from a pattern.
3. Downloads only final products whose EMS AOI extent overlaps the BATO AOI. The BATO AOI is the ``aoi_polygon`` of
   the stack provenance cited by the official road-rule v1 connectivity run, hash-verified.
4. Writes ``data/reference/EMSR927/acquisition_manifest.json``. It records URL, timestamp, file name, bytes,
   SHA-256, AOI, product, version, status, and the image sensors and times.

If the EMS API cannot be reached from this machine, open the API URL in a browser, save the JSON, and pass it with
``--activation-json``. The zips are still downloaded from the URLs inside it.

Nothing is written outside data/reference/EMSR927/. No pipeline code reads that folder.

Exit codes: 0 ok, 1 refused / invalid input, 2 network or download failure.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.data_rules import assert_pipeline_input_allowed
from bato.evaluation.acquisition import AcquisitionError, acquire
from bato.evaluation.official import official_aoi
from bato.evaluation.protocol import load_protocol
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.preprocessing.stack import select_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EMSR927 reference products (evaluation only).")
    parser.add_argument("--manifest", type=Path, required=True, help="BATO Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--connectivity-dir", type=Path, default=None)
    parser.add_argument("--activation-json", type=Path, default=None,
                        help="a browser-saved copy of the official activation record (if the API is unreachable)")
    parser.add_argument("--dry-run", action="store_true", help="save the record and list decisions; download nothing")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        root = cfg.config_dir.parent
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        protocol = load_protocol(root)
        bato_aoi = official_aoi(cfg, manifest.aoi_id, candidate.candidate_id, args.connectivity_dir)
        result = acquire(cfg.settings.paths, bato_aoi, protocol.raw["reference"],
                         local_activation=args.activation_json, dry_run=args.dry_run)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(f"NETWORK FAILURE: {exc}. Save the activation JSON in a browser and pass --activation-json.",
              file=sys.stderr)
        return 2
    except (ConfigError, ContractIOError, BatoError, AcquisitionError, ValueError, KeyError, OSError) as exc:
        print(f"REFUSED / INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    a = result["activation"]
    print(f"Activation record : {a['file']}  sha256 {a['sha256']}  ({a['bytes']} bytes, "
          f"retrieved {a['retrieved_at_utc']})")
    r = result["record"]
    print(f"Activation        : {r['code']} '{r['name']}'  eventTime(raw) {r['event_time_raw']}  "
          f"activationTime(raw) {r['activation_time_raw']}  time-zone marker: {r['time_zone_marker_in_record']}")
    print("EMS AOI overlap with BATO AOI (from record extents):")
    for o in result["aoi_overlaps_from_record_extent"]:
        note = f"  ({o['note']})" if o.get("note") else ""
        print(f"  AOI{o['number']:02d} {o['name']:<12} extent {o['extent_area_km2']} km2, "
              f"overlap {o['overlap_km2']} km2{note}")
    print("Downloaded:" if not args.dry_run else "Would download:")
    for d in result["downloads"]:
        print(f"  {d['label']:<28} {d['file']}  {d.get('bytes', '-')} bytes  sha256 {d.get('sha256', '-')}")
        print(f"      url {d['url']}")
    print("Not downloaded:")
    for s in result["not_downloaded"]:
        print(f"  {s['label']:<28} {s['reason']}")
    if not args.dry_run:
        print("Manifest          : data/reference/EMSR927/acquisition_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the dashboard bundle from the FROZEN official connectivity result (road rule v1). Read-only for every
pipeline file; writes only ``<processed>/dashboard/<aoi>/<candidate_id>/`` (atomically) and its index.

    python scripts/build_dashboard.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \\
        --title Trishuli --debris-control-dir data/interim/detection/aoi-f2cc742b0e25/ts-5adc0f866c2f/debris

The official run is found automatically (the latest connectivity run with road_rule v1, including a
``.superseded-*`` directory); a v2 run is refused. ``--connectivity-dir`` selects one explicitly.
Every output hash is verified and the provenance summary is recomputed from the published layers.
rasterio (optional) adds the Step 4C overlays, the DEM hillshade and SAR chips.

Exit codes: 0 ok, 1 invalid input / integrity check failed (nothing written), 3 build failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.dashboard.bundle import BUNDLE_FILE, DashboardBuildError, prepare_inputs, publish
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.preprocessing.stack import select_candidate


def progress(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BATO dashboard bundle (official v1 result).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--title", default=None, help="display name of the incident (e.g. Trishuli)")
    parser.add_argument("--connectivity-dir", type=Path, default=None,
                        help="use this connectivity output directory (must be road rule v1)")
    parser.add_argument("--debris-control-dir", type=Path, default=None,
                        help="debris output of the matched control (methods panel only)")
    parser.add_argument("--no-rasters", action="store_true", help="skip overlays, hillshade and SAR chips")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        inp = prepare_inputs(cfg, manifest.aoi_id, candidate.candidate_id, title=args.title,
                             connectivity_dir=args.connectivity_dir, debris_control_dir=args.debris_control_dir,
                             with_rasters=not args.no_rasters)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    print(f"Official run : {inp.run.directory} (road rule {inp.run.road_rule}, finished {inp.run.finished_at})")
    for note in inp.notes:
        print(f"Note         : {note}")
    try:
        out = publish(cfg, manifest.aoi_id, candidate.candidate_id, inp, progress)
    except DashboardBuildError as exc:
        print(f"\nREFUSED (integrity check, nothing published): {exc}", file=sys.stderr)
        return 1
    except BatoError as exc:
        print(f"\nFAILED (nothing published): {exc}", file=sys.stderr)
        return 3
    b = json.loads((out / BUNDLE_FILE).read_text(encoding="utf-8"))
    s = b["summary"]
    km, n = s["road_km_by_state"], s["road_segments_by_state"]
    print("\nHeadline numbers (read from the run provenance, checked against its layers):")
    print(f"  road km      : blocked {km['blocked']:,.1f} ({n['blocked']} segments) | open {km['open']:,.1f} | "
          f"unobservable {km['unobservable']:,.1f}")
    print(f"  settlements  : {s['settlements_by_access_state']}")
    print(f"  reachable    : {s['reachable_settlements']}")
    tr = b["integrity"]["route_traces"]
    print(f"  route traces : {tr['verified']} verified, {tr['unavailable']} unavailable")
    if b["experiment"]["verdict"]:
        print(f"  experimental : {b['experiment']['verdict']}")
    print(f"\nOutput       : {out}")
    print("Next         : python scripts/serve_dashboard.py --open")
    return 0


if __name__ == "__main__":
    sys.exit(main())

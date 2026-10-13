"""Generate the one-page SITREP (reports/BATO_SITREP.md) from the FROZEN official result (road rule v1).

    python scripts/build_sitrep.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json \\
        --title "Trishuli Flood, Nepal" \\
        --incident "flood, Bhote Koshi–Trishuli corridor (Rasuwa/Nuwakot), Nepal" \\
        --prepared-by "404 Competition Not Found"

Read-only for every pipeline file. The official run is found exactly as for the dashboard (latest connectivity
run with road_rule v1; a v2 run is refused; ``--connectivity-dir`` selects one explicitly). Output hashes are
verified, the provenance summary is recomputed from the published layers, and every printed number is
cross-checked (bato.dashboard.sitrep). Any missing or inconsistent value stops the run; nothing is written.

The text is a fixed template filled with numbers from the result files: deterministic reporting, not an AI
component, and not an assessment of current conditions (the observation date is the Sentinel-1 POST pass).

``--issued-at`` fixes the issue timestamp (ISO 8601 with a zone) for a byte-identical rebuild; the default is now.
``--local-tz`` adds a local clock time next to the UTC issue time (IANA name, e.g. Asia/Kathmandu).

Exit codes: 0 ok, 1 invalid input / integrity check failed (nothing written).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bato.config import load_config
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract
from bato.dashboard.sitrep import DEFAULT_OUTPUT, SitrepError, load_facts, render, write_report
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError
from bato.preprocessing.stack import select_candidate


def parse_issued(text: str | None) -> datetime:
    if text is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("--issued-at needs a time zone, e.g. 2026-10-13T07:00:00Z")
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the BATO one-page SITREP (official v1 result).")
    parser.add_argument("--manifest", type=Path, required=True, help="Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--connectivity-dir", type=Path, default=None,
                        help="use this connectivity output directory (must be road rule v1)")
    parser.add_argument("--title", required=True, help='report title, e.g. "Trishuli Flood, Nepal"')
    parser.add_argument("--incident", required=True, help="one-line incident description (not taken from data)")
    parser.add_argument("--prepared-by", default=None, help="team name shown in the header")
    parser.add_argument("--number", type=int, default=1, help="SITREP number (default 1)")
    parser.add_argument("--issued-at", default=None, help="issue timestamp, ISO 8601 with zone (default: now)")
    parser.add_argument("--local-tz", default=None, help="IANA zone for a local issue time, e.g. Asia/Kathmandu")
    parser.add_argument("--output", type=Path, default=None, help=f"default: <repo>/{DEFAULT_OUTPUT.as_posix()}")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        issued = parse_issued(args.issued_at)
        tz = ZoneInfo(args.local_tz) if args.local_tz else None
        manifest = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, cfg.settings.paths))
        candidate = select_candidate(manifest, args.candidate)
        facts, notes = load_facts(cfg, manifest.aoi_id, candidate.candidate_id,
                                  connectivity_dir=args.connectivity_dir)
        text = render(facts, title=args.title, incident=args.incident, issued_at=issued,
                      sitrep_number=args.number, prepared_by=args.prepared_by, local_tz=tz)
        output = args.output or cfg.config_dir.parent / DEFAULT_OUTPUT
        out = write_report(text, output, cfg)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule, nothing written): {exc}", file=sys.stderr)
        return 1
    except (ValueError, ZoneInfoNotFoundError) as exc:
        print(f"INVALID ARGUMENT (nothing written): {exc}", file=sys.stderr)
        return 1
    except (SitrepError, ConfigError, ContractIOError, BatoError) as exc:
        print(f"INVALID INPUT / REFUSED (nothing written): {exc}", file=sys.stderr)
        return 1
    print(f"Official run : {facts.connectivity_run} (road rule {facts.road_rule}, finished "
          f"{facts.connectivity_finished_at}; provenance SHA-256 {facts.connectivity_provenance_sha256})")
    for note in notes:
        print(f"Note         : {note}")
    km, s, a = facts.road_km, facts.road_segments, facts.access
    print(f"Roads        : blocked {s['blocked']} segments / {facts.blocked_stretches} stretches / "
          f"{km['blocked']:.3f} km | observed open {km['open']:.3f} km | unobservable {km['unobservable']:.3f} km")
    print(f"Settlements  : {facts.settlements_total} total | connected {a['CONNECTED']} | access unverified "
          f"{a['POSSIBLE_CUTOFF']} | confirmed cutoff {a['CONFIRMED_CUTOFF']} | no pre-event road "
          f"{a['NO_PRE_EVENT_ACCESS']}")
    print(f"Observed     : {facts.post.acquired.isoformat()} (POST) | issued {issued.isoformat()}")
    print(f"Output       : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

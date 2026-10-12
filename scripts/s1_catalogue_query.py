"""Live Sentinel-1 catalogue query against CDSE (Step 2A: discovery only).

Prints every scene the catalogue returned with its status, then the accepted
scenes grouped by (orbit direction, relative orbit). Downloads nothing and
selects no pair.

Examples:
    python scripts/s1_catalogue_query.py --preset trishuli_2026
    python scripts/s1_catalogue_query.py --preset valencia_2024 --save out/valencia_s1.json
    python scripts/s1_catalogue_query.py --bbox 85.17 27.97 85.42 28.28 --event-date 2026-08-26
    python scripts/s1_catalogue_query.py --point 85.32 28.15 --event-date 2026-08-26 --event-time 04:30

Needs network access to catalogue.dataspace.copernicus.eu. No login is required
for catalogue search. Exit codes: 0 ok, 1 invalid input/config, 2 provider error.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, time
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from bato.acquisition.s1_catalogue import search_sentinel1
from bato.config import load_config
from bato.contracts.aoi import AOIRequest, build_aoi
from bato.contracts.catalogue import CandidateStatus, CatalogueSearchResult
from bato.contracts.common import BBox, LonLat
from bato.contracts.io import save_contract
from bato.exceptions import BatoError, ConfigError, InvalidInputError, ProviderError
from bato.logging_utils import configure_logging

PRESETS = Path(__file__).resolve().parents[1] / "config" / "test_aois.yaml"


def build_request(args: argparse.Namespace) -> AOIRequest:
    if args.preset:
        presets = yaml.safe_load(PRESETS.read_text(encoding="utf-8"))
        if args.preset not in presets:
            raise InvalidInputError(f"unknown preset {args.preset!r}; available: {', '.join(presets)}")
        p = presets[args.preset]
        lo1, la1, lo2, la2 = p["bbox"]
        return AOIRequest(event_date=args.event_date or p["event_date"], event_time_utc=args.event_time,
                          bbox=BBox(min_lon=lo1, min_lat=la1, max_lon=lo2, max_lat=la2), label=args.preset)
    if args.event_date is None:
        raise InvalidInputError("--event-date is required without --preset")
    if args.bbox:
        lo1, la1, lo2, la2 = args.bbox
        return AOIRequest(event_date=args.event_date, event_time_utc=args.event_time,
                          bbox=BBox(min_lon=lo1, min_lat=la1, max_lon=lo2, max_lat=la2))
    if args.point:
        return AOIRequest(event_date=args.event_date, event_time_utc=args.event_time,
                          point=LonLat(lon=args.point[0], lat=args.point[1]))
    raise InvalidInputError("give --preset, --bbox or --point")


def short(name: str | None, width: int = 72) -> str:
    text = (name or "-").removesuffix(".SAFE")
    return text if len(text) <= width else text[: width - 1] + "…"


def print_result(result: CatalogueSearchResult) -> None:
    print(f"\nprovider {result.provider} | AOI {result.aoi_id} | event {result.event_date}"
          f"{' ' + str(result.event_time_utc) + ' UTC' if result.event_time_utc else ''}")
    print(f"window {result.window_start:%Y-%m-%d} .. {result.window_end:%Y-%m-%d} (end exclusive) | "
          f"returned {result.returned_count} | accepted {len(result.accepted())} | "
          f"rejected {len(result.rejected())}{' | TRUNCATED' if result.truncated else ''}\n")
    header = f"{'scene':72} {'plat':4} {'sensed (UTC)':16} {'dir':4} {'rel':>3} {'pol':5} {'cov':>5} {'pass':>5} " \
             f"{'days':>7}  status"
    print(header)
    print("-" * len(header))
    for r in result.records:
        a = r.acquisition
        print(f"{short(r.name or r.product_id):72} "
              f"{(a.platform if a else '-'):4} "
              f"{(f'{r.sensing_start:%Y-%m-%d %H:%M}' if r.sensing_start else '-'):16} "
              f"{(a.orbit_direction.value[:4].upper() if a and a.orbit_direction else '-'):4} "
              f"{(str(a.relative_orbit) if a else '-'):>3} "
              f"{('+'.join(p.value for p in a.polarizations) if a else '-'):5} "
              f"{(f'{r.scene_coverage:.0%}' if r.scene_coverage is not None else '-'):>5} "
              f"{(f'{r.pass_coverage:.0%}' if r.pass_coverage is not None else '-'):>5} "
              f"{(f'{r.days_from_event:+.2f}' if r.days_from_event is not None else '-'):>7}  "
              f"{r.status.value}{'' if r.status is CandidateStatus.ACCEPTED else ': ' + r.reason}")

    print("\nRejections by reason:")
    for status, count in result.status_counts().items():
        if count and status != CandidateStatus.ACCEPTED.value:
            print(f"  {status:28} {count}")

    print("\nAccepted scenes by track (orbit direction, relative orbit) - candidates only, no pair selected:")
    if not result.groups:
        print("  (none)")
    for g in result.groups:
        flag = "pre+post available" if g.has_pre_and_post else "NOT pairable (missing pre or post)"
        print(f"  {g.orbit_direction.value:10} rel {g.relative_orbit:3}: {len(g.pre_event)} pre, "
              f"{len(g.post_event)} post, {len(g.pass_ids)} pass(es) - {flag}")
        for name in g.pre_event:
            print(f"      pre  {short(name)}")
        for name in g.post_event:
            print(f"      post {short(name)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel-1 catalogue discovery (CDSE).")
    parser.add_argument("--preset", help=f"named test AOI from {PRESETS.name}")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--point", nargs=2, type=float, metavar=("LON", "LAT"))
    parser.add_argument("--event-date", type=date.fromisoformat)
    parser.add_argument("--event-time", type=time.fromisoformat, help="UTC, e.g. 04:30")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--save", type=Path, help="write the full result as JSON")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, None)
        aoi = build_aoi(build_request(args), cfg.settings)
    except (ConfigError, InvalidInputError, ValueError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    try:
        result = search_sentinel1(aoi, cfg)
    except ProviderError as exc:
        print(f"CATALOGUE QUERY FAILED: {exc}", file=sys.stderr)
        return 2
    except BatoError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print_result(result)
    if args.save:
        save_contract(result, args.save)
        print(f"\nsaved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

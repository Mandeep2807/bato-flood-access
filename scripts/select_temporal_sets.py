"""Rank same-track Sentinel-1 temporal sets from a saved catalogue result (Step 2B).

Input: the JSON written by ``scripts/s1_catalogue_query.py --save``.
Nothing is downloaded; no imagery is read; no reference data is used.

Examples:
    python scripts/select_temporal_sets.py --catalogue trishuli_s1.json
    python scripts/select_temporal_sets.py --catalogue valencia_s1.json --top 3 --save valencia_sets.json

Exit codes: 0 ok, 1 invalid input/config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bato.acquisition.temporal_selection import select_temporal_sets
from bato.config import load_config
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.temporal import HEURISTIC_NAME, TemporalSelectionResult
from bato.exceptions import BatoError, ConfigError
from bato.logging_utils import configure_logging


def print_selection(result: TemporalSelectionResult, top: int | None = None) -> None:
    passes = {p.pass_id: p for p in result.passes}

    def day(pass_id: str) -> str:
        return f"{passes[pass_id].start:%Y-%m-%d}"

    print(f"\n{result.heuristic} | AOI {result.aoi_id} | event {result.event_date}"
          f"{' ' + str(result.event_time_utc) + ' UTC' if result.event_time_utc else ''}")
    print(f"catalogue sha256 {result.catalogue_digest[:16]}… | {len(result.passes)} passes "
          f"({sum(p.multi_slice for p in result.passes)} multi-slice)\n")
    print("Tracks (each viable track keeps its own ranking; no global choice is made):")
    for tr in result.tracks:
        print(f"  {tr.track.label:14} {tr.status:18} {len(tr.candidates)}/{tr.candidates_generated} kept/generated"
              f" - {tr.note}")

    header = (f"\n{'rank':>4} {'track':14} {'pre-1':10} {'pre-2':10} {'post':10} {'gap d':>6} {'sep d':>6} "
              f"{'cov':>5} {'score':>6}  {'components gap/rec/sep/cov/qual':31} {'multi':5}  flags")
    print(header)
    print("-" * (len(header) + 10))
    for tr in result.tracks:
        for c in tr.candidates[: top or None]:
            pre1 = day(c.pre_event_pass_ids[0]) if len(c.pre_event_pass_ids) == 2 else "-"
            pre2 = day(c.pre_event_pass_ids[-1])
            s = c.score
            comps = (f"{s.post_gap_score:.2f}/{s.pre_recency_score:.2f}/{s.baseline_separation_score:.2f}/"
                     f"{s.coverage_score:.2f}/{s.quality_score:.2f}")
            sep = f"{c.pre_separation_days:.1f}" if c.pre_separation_days is not None else "-"
            print(f"{c.rank:>4} {tr.track.label:14} {pre1:10} {pre2:10} {day(c.post_event_pass_id):10} "
                  f"{c.post_gap_days:>6.2f} {sep:>6} {c.coverage:>5.0%} {s.total:>6.3f}  {comps:31} "
                  f"{'yes' if c.multi_slice else 'no':5}  {', '.join(f.value for f in c.quality_flags) or '-'}")
    print(f"\nScores are a {HEURISTIC_NAME}: deterministic and explainable, NOT scientifically validated.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank same-track Sentinel-1 temporal sets (Step 2B).")
    parser.add_argument("--catalogue", type=Path, required=True, help="JSON from s1_catalogue_query.py --save")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--top", type=int, default=None, help="show only the top N per track")
    parser.add_argument("--save", type=Path, help="write the full selection result as JSON")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        configure_logging(cfg.settings.logging.level, None)
        catalogue = load_contract(CatalogueSearchResult, args.catalogue)
    except (ConfigError, ContractIOError) as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    if catalogue.config_fingerprint != cfg.fingerprint():
        print("NOTE: the catalogue was classified under a different configuration fingerprint than the "
              "current config; re-run the catalogue query if acceptance rules changed.", file=sys.stderr)
    try:
        result = select_temporal_sets(catalogue, cfg)
    except BatoError as exc:
        print(f"SELECTION FAILED: {exc}", file=sys.stderr)
        return 1
    print_selection(result, args.top)
    if args.save:
        save_contract(result, args.save)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

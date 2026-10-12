"""EVALUATION ONLY. Describe the downloaded EMSR927 products. Computes NO metric.

    python scripts/inspect_emsr927.py

The script first verifies every zip against the SHA-256 in ``data/reference/EMSR927/acquisition_manifest.json``.
It then reads the zips in place, without extracting next to pipeline data, and reports:

* members;
* layers, vector or raster;
* CRS and geometry types;
* fields and class values;
* extents, and EMS AOI overlap with the BATO AOI;
* raster headers;
* image times and time gaps to BATO POST, under each time-zone reading;
* time-zone clues found in the product;
* which product the pre-registered rule compares per AOI (decided from image times only);
* every layer or class the protocol does not map yet.

Output: ``data/reference/EMSR927/inspection/inspection_report.{json,md}``.

Exit codes: 0 ok, 1 refused / invalid input.
"""

from __future__ import annotations

import argparse
import sys

from bato.config import load_config
from bato.evaluation.acquisition import AcquisitionError
from bato.evaluation.emsr927 import inspect
from bato.evaluation.protocol import load_protocol
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect EMSR927 products (no metrics).")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        report = inspect(cfg, load_protocol(cfg.config_dir.parent))
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ConfigError, BatoError, AcquisitionError, ValueError, KeyError, OSError) as exc:
        print(f"REFUSED / INVALID INPUT: {exc}", file=sys.stderr)
        return 1
    for p in report["products"]:
        print(f"{p['label']}  ({p['zip']})")
        for ly in p["layers"]:
            print(f"  {ly['stem']:<22} role={ly.get('protocol_role')!s:<16} {ly.get('loaded_from')!s:<9} "
                  f"crs={ly.get('crs')}  n={ly.get('features')}  {ly.get('geometry_types')}")
        for c in p["time_zone_clues"][:5]:
            print(f"  time-zone clue [{c['where']}]: {c['match']}  …{c['context'][:100]}…")
    print(f"Product selection : {report['product_selection'].get('compared') or report['product_selection']}")
    for label, g in (report.get("road_geometry_check", {}).get("products") or {}).items():
        share = g.get("share_of_ems_road_length_within_distance_of_bato_roads")
        print(f"Road geometry     : {label}: share of EMS road length within 10/20/30/50 m of BATO roads {share}; "
              f"BATO segments < min matched length: {g.get('bato_segments_shorter_than_min_matched_length')}")
    probs = report.get("mapping_problems_for_freeze", [])
    print(f"Unmapped items    : {len(probs)}")
    for x in probs:
        print(f"  - {x}")
    print("Report            : data/reference/EMSR927/inspection/inspection_report.md")
    print("No metric was computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

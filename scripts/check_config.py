"""Validate BATO configuration and print a summary.

Usage:
    python scripts/check_config.py [--config-dir PATH]

Exits with status 1 and a readable message if the configuration is invalid.
"""

from __future__ import annotations

import argparse
import sys

from bato.config import load_config
from bato.exceptions import ConfigError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-dir", default=None, help="folder with config.yaml and thresholds.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"CONFIG INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"config dir   : {cfg.config_dir}")
    print(f"fingerprint  : {cfg.fingerprint()}")
    print(f"AOI limits   : {cfg.settings.aoi.min_area_km2}-{cfg.settings.aoi.max_area_km2} km2")
    print(f"S1 windows   : pre {cfg.settings.sentinel1.pre_window_days} d, "
          f"post {cfg.settings.sentinel1.post_window_days} d")
    print(f"flood model  : {cfg.settings.models.flood_model_path or 'NOT AVAILABLE YET'}")
    provisional = cfg.thresholds.provisional_parameters()
    print(f"provisional thresholds ({len(provisional)}):")
    for name in provisional:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

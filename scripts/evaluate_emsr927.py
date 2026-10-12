"""EVALUATION ONLY. Freeze the EMSR927 protocol, or compute the pre-registered comparison metrics.

    python scripts/evaluate_emsr927.py --freeze-protocol
    python scripts/evaluate_emsr927.py --manifest data/reference/EMSR927/acquisition_manifest.json
    python scripts/evaluate_emsr927.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

``--manifest`` takes either the EMSR927 evaluation manifest or the BATO fetch manifest; both select the same BATO
result. Each is checked by its own data rule (see bato.evaluation.cli); every other file under data/reference/,
including a product zip, is refused.

Freezing requires an inspection report of the current zips with every layer and class mapped, and no earlier run
under the same protocol_id.

Evaluating requires all of:

* the frozen protocol, with its hashes unchanged;
* a leakage scan with 0 hits;
* zips that match their recorded SHA-256;
* the official road-rule v1 BATO result, hash-verified.

BATO files are re-hashed afterwards to prove nothing changed. Outputs go to
``data/eval/EMSR927/<protocol-sha12>/metrics.{json,md}``, and each run is appended to
``data/eval/EMSR927/runs.jsonl``.

Needs rasterio to read the Step 4C state raster.

Exit codes: 0 ok, 1 refused (protocol, leakage, integrity), 3 failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bato.config import load_config
from bato.contracts.io import ContractIOError
from bato.evaluation.acquisition import AcquisitionError
from bato.evaluation.cli import resolve_target
from bato.evaluation.emsr927 import EvaluationError, evaluate, freeze_protocol
from bato.evaluation.protocol import ProtocolError, load_protocol
from bato.exceptions import BatoError, ConfigError, ProhibitedInputError


def main() -> int:
    parser = argparse.ArgumentParser(description="EMSR927 comparison (evaluation only).")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="data/reference/EMSR927/acquisition_manifest.json (EMSR927 evaluation manifest) or the "
                             "BATO Step 2C *_fetch_manifest.json")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--connectivity-dir", type=Path, default=None)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config_dir)
        protocol = load_protocol(cfg.config_dir.parent)
        if args.freeze_protocol:
            rec = freeze_protocol(cfg, protocol)
            print(json.dumps(rec, indent=2))
            print("Protocol frozen. Metrics may now be computed.")
            return 0
        if args.manifest is None:
            parser.error("--manifest is required to evaluate")
        target = resolve_target(cfg, args.manifest, args.candidate)
        print(f"Manifest     : {target.manifest_kind} -> BATO {target.aoi_id} / {target.candidate_id}")
        out, res = evaluate(cfg, target.aoi_id, target.candidate_id, connectivity_dir=args.connectivity_dir,
                            protocol=protocol)
    except ProhibitedInputError as exc:
        print(f"REFUSED (data rule): {exc}", file=sys.stderr)
        return 1
    except (ProtocolError, EvaluationError, AcquisitionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except (ConfigError, ContractIOError, BatoError, ValueError, KeyError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 3
    print(f"Leakage scan : {res['leakage_scan']['hits']} hits in {res['leakage_scan']['json_files_scanned']} files")
    print(f"BATO files   : unchanged ({len(res['bato']['files_read_sha256'])} hashed before and after)")
    print(f"Output       : {out}")
    print((out / "metrics.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

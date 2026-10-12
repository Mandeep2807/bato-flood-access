"""EVALUATION ONLY. Leakage scan: no BATO pipeline provenance or manifest may cite reference data.

Same file set as the Phase-1 evidence audit: every provenance / manifest JSON under the raw, interim and processed
folders is read. Any string value is a hit if it either:

* names EMSR / UNOSAT / Copernicus EMS; or
* resolves to a path inside the reference or evaluation folders.

Statements *about* reference data are exempt from the name check: values under a key that is exactly
``inputs_not_used``, ``validation_note``, ``rule``, ``rules`` or ``description``. The path check applies to every
value. This is stricter than the Phase-1 audit, which skipped those keys entirely and matched fewer names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bato.config import PathSettings

FLAG = re.compile(r"emsr\d|unosat|copernicus[ _-]?ems|rapidmapping", re.IGNORECASE)
ABOUT_KEYS = {"inputs_not_used", "validation_note", "rule", "rules", "description"}
SCAN_NAMES = re.compile(r"(provenance|manifest|_sets|_s1)\.json$")


def _strings(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}.{i}")
    elif isinstance(obj, str):
        yield path, obj


def _about_reference(jpath: str) -> bool:
    """A statement ABOUT reference data (exempt from the name check only): a path component that is exactly one of
    ABOUT_KEYS. A path into the reference/eval folders is a hit wherever it appears."""
    return any(part in ABOUT_KEYS for part in jpath.split("."))


def scan(paths: PathSettings, root: Path) -> dict[str, Any]:
    forbidden = [p.resolve() for p in paths.evaluation_only_dirs()]
    hits: list[dict[str, Any]] = []
    scanned = 0
    for base in (paths.raw, paths.interim, paths.processed):
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.json")):
            if not SCAN_NAMES.search(f.name):
                continue
            scanned += 1
            rel = f.resolve().relative_to(root.resolve()).as_posix() if root.resolve() in f.resolve().parents \
                else str(f)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                hits.append({"file": rel, "problem": f"unreadable: {exc}"})
                continue
            for jpath, s in _strings(data):
                flagged = bool(FLAG.search(s)) and not _about_reference(jpath)
                into = False
                if ("/" in s or "\\" in s) and len(s) < 400:
                    p = Path(s) if Path(s).is_absolute() else root / s
                    try:
                        rp = p.resolve()
                        into = any(rp == fb or fb in rp.parents for fb in forbidden)
                    except (OSError, ValueError):
                        pass
                if flagged or into:
                    hits.append({"file": rel, "json_path": jpath, "value": s[:200],
                                 "why": "points into reference/eval" if into else "names EMSR/UNOSAT/EMS"})
    return {"rule": "provenance/manifest JSON under raw, interim, processed; any path into reference/eval is a hit; "
                    "EMSR<digit>/UNOSAT/Copernicus EMS/rapidmapping names are hits outside inputs_not_used, "
                    "validation_note, rule(s) and description keys", "json_files_scanned": scanned, "hits": hits}

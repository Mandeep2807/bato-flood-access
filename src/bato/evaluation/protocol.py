"""EVALUATION ONLY. The pre-registered EMSR927 comparison protocol: loading, mapping checks and freezing.

The protocol is two files: ``config/evaluation/emsr927_protocol.yaml`` (machine-readable) and
``docs/evaluation/emsr927_protocol.md`` (the text). Metrics are refused unless:

* the YAML says ``status: frozen``;
* a freeze record exists whose SHA-256 values still match both files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

CATEGORIES = ("flood_water", "reference_mass_movement", "flood_trace", "other_event")
# Pixel comparisons by observed-event category. flood_water is the only class-matched one; it is computed only if an
# EMS open-water class exists and is mapped (EMSR927 has none). reference_mass_movement gives descriptive overlap
# shares only (no precision / recall / F1 / IoU).
COMPARED = {"flood_water": {"class_match": True, "role": "pixel_flood_comparison"},
            "reference_mass_movement": {"class_match": False, "role": "class_mismatched_footprint_overlap"}}
REMOVED_FROM_AREA = ("flood_trace", "other_event")
DAMAGE_GRADES = ("destroyed", "damaged", "possibly_damaged", "no_visible_damage", "not_analysed")
ROAD_LABELS = ("positive", "negative", "excluded")
ROAD_ROLES = ("primary", "supplementary")
LAYER_ROLES = ("area_of_interest", "observed_event", "transportation", "bridge_context", "not_analysed")
PROTOCOL_YAML = Path("config/evaluation/emsr927_protocol.yaml")
PROTOCOL_DOC = Path("docs/evaluation/emsr927_protocol.md")
FREEZE_FILE = "protocol_freeze.json"


class ProtocolError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Protocol:
    raw: dict[str, Any]
    yaml_path: Path
    doc_path: Path
    yaml_sha256: str
    doc_sha256: str

    @property
    def status(self) -> str:
        return str(self.raw.get("status"))

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def combined_sha256(self) -> str:
        return sha256_bytes((self.yaml_sha256 + self.doc_sha256).encode())

    def layer_role(self, stem: str) -> str:
        """Role of a layer stem; unknown stems fail closed."""
        layers = self.raw["layers"]
        for role in LAYER_ROLES:
            if stem in (layers.get(role) or []):
                return role
        if stem in (layers.get("not_used") or {}):
            return "not_used"
        raise ProtocolError(f"layer {stem!r} is not mapped in the protocol (layers.*); map it or list it under "
                            "layers.not_used with a reason before freezing")

    def event_category(self, value: Any) -> str:
        classes = self.raw["observed_event"]["classes"] or {}
        if value not in classes:
            raise ProtocolError(f"observed-event class {value!r} is not mapped in observed_event.classes")
        return str(classes[value])

    def road_grade(self, value: Any) -> str:
        """Damage grade of a road (transportation) feature value; unmapped values fail closed."""
        classes = self.raw["transportation"]["classes"] or {}
        if value not in classes:
            raise ProtocolError(f"road damage value {value!r} is not mapped in transportation.classes")
        return str(classes[value])

    def bridge_grade(self, value: Any) -> str:
        classes = self.raw["bridge_context"]["classes"] or {}
        if value not in classes:
            raise ProtocolError(f"bridge damage value {value!r} is not mapped in bridge_context.classes")
        return str(classes[value])

    def road_comparisons(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = self.raw["transportation"].get("comparisons") or {}
        return out


def load_protocol(root: Path, yaml_rel: Path = PROTOCOL_YAML, doc_rel: Path = PROTOCOL_DOC) -> Protocol:
    yp, dp = root / yaml_rel, root / doc_rel
    if not yp.is_file() or not dp.is_file():
        raise ProtocolError(f"protocol files missing: {yaml_rel} and {doc_rel} are both required")
    return _load_files(yp, dp)


def _load_files(yp: Path, dp: Path) -> Protocol:
    """Parse and hash the protocol files from their exact bytes on disk (the only hashing used anywhere)."""
    yb, db = yp.read_bytes(), dp.read_bytes()
    raw = yaml.safe_load(yb.decode("utf-8"))
    validate(raw)
    return Protocol(raw=raw, yaml_path=yp, doc_path=dp, yaml_sha256=sha256_bytes(yb), doc_sha256=sha256_bytes(db))


def validate(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise ProtocolError("protocol YAML must be a mapping")
    for key in ("protocol_id", "status", "reference", "layers", "observed_event", "transportation", "bridge_context",
                "roads", "pixels", "time"):
        if key not in raw:
            raise ProtocolError(f"protocol YAML lacks {key!r}")
    if raw["status"] not in ("draft", "frozen"):
        raise ProtocolError("status must be draft or frozen")
    for value, cat in (raw["observed_event"].get("classes") or {}).items():
        if cat not in CATEGORIES:
            raise ProtocolError(f"class {value!r} -> {cat!r}: category must be one of {CATEGORIES}")
    for section in ("transportation", "bridge_context"):
        for value, grade in (raw[section].get("classes") or {}).items():
            if grade not in DAMAGE_GRADES:
                raise ProtocolError(f"{section} value {value!r} -> {grade!r}: grade must be one of {DAMAGE_GRADES}")
        if raw[section].get("classes") and not raw[section].get("damage_field"):
            raise ProtocolError(f"{section}.classes given without {section}.damage_field")
    comps = raw["transportation"].get("comparisons") or {}
    if raw["transportation"].get("classes") and not comps:
        raise ProtocolError("transportation.classes given without transportation.comparisons")
    if comps and [c.get("role") for c in comps.values()].count("primary") != 1:
        raise ProtocolError("exactly one road comparison must have role: primary")
    for name, c in comps.items():
        if c.get("role") not in ROAD_ROLES or not str(c.get("label", "")).strip():
            raise ProtocolError(f"road comparison {name!r} needs role in {ROAD_ROLES} and a label")
        listed = [g for lab in ROAD_LABELS for g in (c.get(lab) or [])]
        if sorted(listed) != sorted(DAMAGE_GRADES):
            raise ProtocolError(f"road comparison {name!r} must place every grade {DAMAGE_GRADES} in exactly one of "
                                f"{ROAD_LABELS} (no silent merge or drop); got {listed}")
        if not c.get("positive") or not c.get("negative"):
            raise ProtocolError(f"road comparison {name!r} needs at least one positive and one negative grade")
    seen: dict[str, str] = {}
    for role in LAYER_ROLES:
        for stem in raw["layers"].get(role) or []:
            if stem in seen:
                raise ProtocolError(f"layer {stem!r} has two roles: {seen[stem]} and {role}")
            seen[stem] = role
    for stem in raw["layers"].get("not_used") or {}:
        if stem in seen:
            raise ProtocolError(f"layer {stem!r} is both {seen[stem]} and not_used")
        if not str(raw["layers"]["not_used"][stem]).strip():
            raise ProtocolError(f"layers.not_used.{stem} needs a reason")
    if int(raw["pixels"]["tolerance_px"]) < 0:
        raise ProtocolError("pixels.tolerance_px must be >= 0")


def check_mapping_complete(protocol: Protocol, inspection: dict[str, Any], compared_zips: set[str]) -> list[str]:
    """Problems that block freezing: unmapped layer stems or class values in the compared products."""
    problems = []
    raw = protocol.raw
    class_field = raw["observed_event"].get("class_field")
    for prod in inspection.get("products", []):
        if prod.get("zip") not in compared_zips:
            continue
        for layer in prod.get("layers", []):
            stem = layer["stem"]
            try:
                role = protocol.layer_role(stem)
            except ProtocolError as exc:
                problems.append(f"{prod['zip']}: {exc}")
                continue
            props = layer.get("properties") or {}
            if role == "observed_event":
                if class_field not in props:
                    problems.append(f"{prod['zip']}: {stem} has no field {class_field!r} "
                                    f"(fields: {sorted(props)})")
                    continue
                if props[class_field].get("values_truncated") or "values" not in props[class_field]:
                    problems.append(f"{prod['zip']}: too many distinct {class_field} values to map")
                    continue
                for v in props[class_field]["values"]:
                    if v["value"] not in (raw["observed_event"].get("classes") or {}):
                        problems.append(f"{prod['zip']}: {stem}.{class_field} = {v['value']!r} is not mapped")
            for sec_role, section in (("transportation", "transportation"), ("bridge_context", "bridge_context")):
                if role != sec_role:
                    continue
                field = raw[section].get("damage_field")
                if not field or field not in props:
                    problems.append(f"{prod['zip']}: {section} layer {stem} has no field {field!r} "
                                    f"(fields: {sorted(props)})")
                    continue
                if props[field].get("values_truncated") or "values" not in props[field]:
                    problems.append(f"{prod['zip']}: too many distinct {stem}.{field} values to map")
                    continue
                for v in props[field]["values"]:
                    if v["value"] not in (raw[section].get("classes") or {}):
                        problems.append(f"{prod['zip']}: {stem}.{field} = {v['value']!r} is not mapped in "
                                        f"{section}.classes")
    return problems


def freeze(protocol: Protocol, eval_dir: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Set ``status: frozen`` and ``frozen_at`` in the YAML (exact-line edits), then record both hashes.

    Everything is done on BYTES. The file's own line endings (LF or CRLF) are kept, and both recorded hashes are
    taken from the files as they are on disk after the write. This is the exact input :func:`load_protocol`
    hashes, so a freeze -> reload -> :func:`check_frozen` cycle is stable on every platform. A text-mode write
    would not be: on Windows it turns "\n" into "\r\n", so the bytes on disk would differ from the string that
    was hashed.
    """
    if protocol.status == "frozen":
        raise ProtocolError("the protocol is already frozen")
    data = protocol.yaml_path.read_bytes()
    if sha256_bytes(data) != protocol.yaml_sha256 or sha256_bytes(protocol.doc_path.read_bytes()) != \
            protocol.doc_sha256:
        raise ProtocolError("the protocol files changed since they were loaded and checked; reload before freezing")
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()
    edits = {b"status: draft": b"status: frozen", b"frozen_at: null": f'frozen_at: "{stamp}"'.encode()}
    found = dict.fromkeys(edits, 0)
    out = []
    for line in data.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        if body in edits:
            found[body] += 1
            line = edits[body] + line[len(body):]                  # keep this line's own ending (LF or CRLF)
        out.append(line)
    if any(n != 1 for n in found.values()):
        raise ProtocolError("the YAML must contain the lines 'status: draft' and 'frozen_at: null' exactly once")
    protocol.yaml_path.write_bytes(b"".join(out))
    frozen = _load_files(protocol.yaml_path, protocol.doc_path)          # hashes of the bytes now on disk
    if frozen.status != "frozen" or frozen.raw.get("frozen_at") != stamp:
        raise ProtocolError("the frozen YAML did not re-read as status frozen with the recorded frozen_at")
    record = {"protocol_id": frozen.protocol_id, "frozen_at": stamp,
              "yaml": {"path": str(PROTOCOL_YAML), "sha256": frozen.yaml_sha256},
              "doc": {"path": str(PROTOCOL_DOC), "sha256": frozen.doc_sha256}}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / FREEZE_FILE).write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    return record


def check_frozen(protocol: Protocol, eval_dir: Path) -> dict[str, Any]:
    if protocol.status != "frozen":
        raise ProtocolError("the protocol is a draft; inspect the products, complete the mappings and freeze it "
                            "(scripts/evaluate_emsr927.py --freeze-protocol) before computing any metric")
    p = eval_dir / FREEZE_FILE
    if not p.is_file():
        raise ProtocolError(f"{p} missing: the protocol was not frozen with the freeze command")
    rec: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    if rec["yaml"]["sha256"] != protocol.yaml_sha256 or rec["doc"]["sha256"] != protocol.doc_sha256:
        raise ProtocolError("the protocol files changed after the freeze; a changed protocol needs a new "
                            "protocol_id and a disclosed reason")
    if rec["protocol_id"] != protocol.protocol_id:
        raise ProtocolError("freeze record is for a different protocol_id")
    return rec

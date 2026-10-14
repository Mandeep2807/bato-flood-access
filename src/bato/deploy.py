"""Deployment data: the minimum read-only artifacts a hosted dashboard + copilot needs (scripts/deploy_data.py).

``data/`` is Git-ignored, so a fresh clone (e.g. on Render) has no results. ``pack`` collects, from a machine that
has run the pipeline, exactly what the server reads for ONE incident, and nothing else:

* the dashboard bundle ``<processed>/dashboard/<aoi>/<candidate>/`` (bundle.json + rendered PNG overlays/chips),
  plus an ``index.json`` listing only that bundle;
* the official road-rule-v1 connectivity run ``<processed>/connectivity/<aoi>/<run>/``: its provenance and every
  output the provenance lists (all SHA-256 checked again by the server);
* the stack and Step 4C provenance JSON files the run references (hash-checked by the server).

Before packing, the run is selected and cross-checked by the SITREP chain (``load_facts``: official v1 only, a v2
run is refused, every output hash and the published summary verified), and the bundle's integrity block must name
that same run and the same provenance hashes. Never packed: rasters (GeoTIFF), raw downloads, OSM snapshots,
superseded or v2 runs, the experimental-detector outputs, and anything in the evaluation-only folders (EMSR927).

``install`` extracts the archive into the repository, refusing unsafe paths, files outside data/processed and
data/interim, unknown members and any file whose SHA-256 differs from the manifest. ``verify`` starts the real
server on the installed data and checks the dashboard index, the bundle and the deterministic SITREP.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from bato.config import BatoConfig
from bato.dashboard.bundle import (
    BUNDLE_FILE,
    INDEX_FILE,
    OFFICIAL_ROAD_RULE,
    OUTPUT_SUBDIR,
    PROVENANCE_FILE,
    prepare_inputs,
    sha256_file,
)
from bato.dashboard.sitrep import SitrepError, load_facts
from bato.data_rules import assert_pipeline_input_allowed
from bato.exceptions import BatoError

DEFAULT_ARCHIVE = Path("deploy") / "bato_official_data.tar.gz"
MANIFEST = "BATO_DEPLOY_MANIFEST.json"
MANIFEST_VERSION = 1
MAX_ARCHIVE_BYTES = 95 * 1024 * 1024          # GitHub refuses files over 100 MB
ALLOWED_SUFFIXES = (".json", ".geojson", ".jsonl", ".md", ".png")


class DeployDataError(BatoError):
    """The deployment data cannot be packed, installed or verified (the message says why)."""


@dataclass(frozen=True)
class PackReport:
    archive: Path
    archive_bytes: int
    files: list[dict[str, Any]]                 # path, size, sha256 (largest first)
    connectivity_run: str


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise DeployDataError(f"{path} is outside the repository {root}; deployment data must use the "
                              "repository's data/ folders") from None


def _under(path: Path, folder: Path) -> bool:
    p, f = path.resolve(), folder.resolve()
    return p == f or f in p.parents


def collect(cfg: BatoConfig, aoi_id: str, candidate_id: str) -> tuple[dict[str, Path], dict[str, Any]]:
    """Map ``repo-relative path -> file`` for one incident, after the full official-run cross-check."""
    root = cfg.config_dir.parent
    paths = cfg.settings.paths
    try:
        facts, _notes = load_facts(cfg, aoi_id, candidate_id)
    except SitrepError as exc:
        raise DeployDataError(f"the official result does not pass the SITREP integrity checks: {exc}") from exc
    if facts.road_rule != OFFICIAL_ROAD_RULE:
        raise DeployDataError(f"road rule {facts.road_rule} is not the official rule {OFFICIAL_ROAD_RULE}")
    run_dir = paths.processed / cfg.settings.connectivity.output_subdir / aoi_id / facts.connectivity_run
    inp = prepare_inputs(cfg, aoi_id, candidate_id, connectivity_dir=run_dir, with_rasters=False)
    files: dict[str, Path] = {}
    for p in [run_dir / PROVENANCE_FILE, *inp.files.values()]:
        files[_rel(root, p)] = p
    inputs = inp.run.provenance["inputs"]
    for key in ("stack_provenance", "baseline_provenance"):
        p = root / inputs[key]["path"]
        files[_rel(root, p)] = p

    bundle_dir = paths.processed / OUTPUT_SUBDIR / aoi_id / candidate_id
    bundle_path = bundle_dir / BUNDLE_FILE
    if not bundle_path.is_file():
        raise DeployDataError(f"{bundle_path} not found: run scripts/build_dashboard.py first")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    integ, inc = bundle.get("integrity") or {}, bundle.get("incident") or {}
    expected = {
        "integrity.official_road_rule": (integ.get("official_road_rule"), OFFICIAL_ROAD_RULE),
        "incident.road_rule": (inc.get("road_rule"), OFFICIAL_ROAD_RULE),
        "incident.aoi_id": (inc.get("aoi_id"), aoi_id),
        "incident.candidate_id": (inc.get("candidate_id"), candidate_id),
        "integrity.connectivity_run": (integ.get("connectivity_run"), facts.connectivity_run),
        "integrity.connectivity_provenance_sha256": (integ.get("connectivity_provenance_sha256"),
                                                     facts.connectivity_provenance_sha256),
        "integrity.stack_provenance_sha256": (integ.get("stack_provenance_sha256"), inp.stack_sha256),
        "integrity.baseline_provenance_sha256": (integ.get("baseline_provenance_sha256"), inp.baseline_sha256),
    }
    for name, sha in (integ.get("outputs_verified") or {}).items():
        expected[f"integrity.outputs_verified.{name}"] = (sha, inp.run.provenance["outputs"][name]["sha256"])
    bad = [f"{k}: bundle has {got!r}, official run has {want!r}" for k, (got, want) in expected.items() if got != want]
    if bad:
        raise DeployDataError("the dashboard bundle was not built from the official v1 run (rebuild it with "
                              "scripts/build_dashboard.py): " + "; ".join(bad))
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file():
            files[_rel(root, p)] = p

    for rel, p in files.items():
        assert_pipeline_input_allowed(p, paths)                       # never reference / evaluation folders
        if not (_under(p, paths.processed) or _under(p, paths.interim)):
            raise DeployDataError(f"{rel} is not under data/processed or data/interim")
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            raise DeployDataError(f"{rel}: {p.suffix} files are not deployment data (allowed: {ALLOWED_SUFFIXES})")
    meta = {"aoi_id": aoi_id, "candidate_id": candidate_id, "road_rule": facts.road_rule,
            "connectivity_run": facts.connectivity_run,
            "connectivity_provenance_sha256": facts.connectivity_provenance_sha256,
            "bundle_path": f"{aoi_id}/{candidate_id}", "title": inc.get("title", aoi_id),
            "index_path": _rel(root, paths.processed / OUTPUT_SUBDIR / INDEX_FILE)}
    return files, meta


def _index_bytes(bundle_path: Path, bundle_rel: str) -> bytes:
    inc = json.loads(bundle_path.read_text(encoding="utf-8"))["incident"]
    entry = {"path": bundle_rel, "title": inc["title"], "aoi_id": inc["aoi_id"],
             "candidate_id": inc["candidate_id"], "event_date": inc["event_date"]}
    return json.dumps({"bundles": [entry]}, indent=1).encode("utf-8")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tar.addfile(info, io.BytesIO(data))


def pack(cfg: BatoConfig, aoi_id: str, candidate_id: str, out: Path,
         now: datetime | None = None) -> PackReport:
    files, meta = collect(cfg, aoi_id, candidate_id)
    index_rel = meta["index_path"]
    bundle_file = cfg.settings.paths.processed / OUTPUT_SUBDIR / aoi_id / candidate_id / BUNDLE_FILE
    payload: dict[str, bytes] = {rel: p.read_bytes() for rel, p in sorted(files.items())}
    payload[index_rel] = _index_bytes(bundle_file, meta["bundle_path"])
    entries: list[dict[str, Any]] = [{"path": rel, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                     for rel, b in payload.items()]
    manifest = {"version": MANIFEST_VERSION, "created_at": (now or datetime.now(timezone.utc)).isoformat(),
                **{k: v for k, v in meta.items() if k != "index_path"}, "index": index_rel,
                "note": "Read-only official BATO result for hosting (road rule v1, hash-verified). NOT VALIDATED.",
                "files": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tarfile.open(tmp, "w:gz", compresslevel=9) as tar:
        _add(tar, MANIFEST, json.dumps(manifest, indent=1).encode("utf-8"))
        for rel, data in payload.items():
            _add(tar, rel, data)
    size = tmp.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        tmp.unlink()
        raise DeployDataError(f"the archive would be {size / 2**20:.1f} MiB, above the "
                              f"{MAX_ARCHIVE_BYTES / 2**20:.0f} MiB limit for a file in a GitHub repository")
    os.replace(tmp, out)
    entries.sort(key=lambda e: -int(e["size"]))
    return PackReport(archive=out, archive_bytes=size, files=entries, connectivity_run=meta["connectivity_run"])


def _safe_member(name: str, allowed: tuple[str, ...]) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise DeployDataError(f"unsafe path in the archive: {name!r}")
    if not any(p.as_posix().startswith(a + "/") for a in allowed):
        raise DeployDataError(f"{name} is outside {', '.join(allowed)}")
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise DeployDataError(f"{name}: {p.suffix} files are not deployment data")
    return p


def read_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(MANIFEST)
        fh = tar.extractfile(member)
        if fh is None:
            raise DeployDataError("the archive manifest is not a regular file")
        manifest: dict[str, Any] = json.loads(fh.read().decode("utf-8"))
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("road_rule") != OFFICIAL_ROAD_RULE:
        raise DeployDataError("unsupported manifest (version or road rule)")
    return manifest


def install(cfg: BatoConfig, archive: Path, *, force: bool = False) -> dict[str, Any]:
    """Extract and hash-verify the archive into the repository. Existing identical files are kept; a different
    existing file is refused unless ``force`` (a local pipeline output is never silently overwritten)."""
    root = cfg.config_dir.parent.resolve()
    allowed = tuple(_rel(root, d) for d in (cfg.settings.paths.processed, cfg.settings.paths.interim))
    try:
        manifest = read_manifest(archive)
    except (OSError, KeyError, ValueError, tarfile.TarError) as exc:
        raise DeployDataError(f"cannot read {archive}: {exc}") from exc
    want = {e["path"]: e for e in manifest["files"]}
    if len(want) != len(manifest["files"]):
        raise DeployDataError("duplicate paths in the manifest")
    for rel in want:
        _safe_member(rel, allowed)
    with tempfile.TemporaryDirectory(prefix="bato-deploy-") as tmpd:
        staged: dict[str, Path] = {}
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name == MANIFEST:
                    continue
                if not m.isfile():
                    raise DeployDataError(f"{m.name} is not a regular file")
                if m.name not in want or m.name in staged:
                    raise DeployDataError(f"{m.name} is not listed in the manifest (or listed twice)")
                _safe_member(m.name, allowed)
                fh = tar.extractfile(m)
                data = fh.read() if fh else b""
                ref = want[m.name]
                if len(data) != ref["size"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
                    raise DeployDataError(f"{m.name} does not match its SHA-256 in the manifest")
                dest = Path(tmpd) / str(len(staged))
                dest.write_bytes(data)
                staged[m.name] = dest
        missing = sorted(set(want) - set(staged))
        if missing:
            raise DeployDataError(f"the archive is missing {missing[:5]}")
        written = kept = 0
        conflicts = [rel for rel in staged if (root / rel).is_file()
                     and sha256_file(root / rel) != want[rel]["sha256"]]
        if conflicts and not force:
            raise DeployDataError(f"{len(conflicts)} existing file(s) differ from the archive, e.g. {conflicts[0]}; "
                                  "refusing to overwrite local results (use --force on a deployment host only)")
        for rel, src in staged.items():
            dest = root / rel
            if dest.is_file() and rel not in conflicts:
                kept += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            written += 1
    return {"manifest": manifest, "written": written, "kept": kept, "overwritten": len(conflicts) if force else 0}

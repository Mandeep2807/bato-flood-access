"""Package, install and verify the read-only official BATO data for a hosted dashboard (see docs/deployment.md).

    # 1. on the machine that ran the pipeline: build the archive (official road-rule-v1 run only)
    python scripts/deploy_data.py pack --aoi aoi-e893707c2beb --candidate ts-80841ae62417

    # 2. on the host (or in a fresh clone): extract with SHA-256 checks
    python scripts/deploy_data.py install

    # 3. start the real server on that data and check dashboard + evidence + deterministic SITREP
    python scripts/deploy_data.py verify

The archive defaults to deploy/bato_official_data.tar.gz. ``verify`` never calls the LLM (BATO_LLM_* are removed
from the server it starts) and never prints a key.
Exit codes: 0 ok, 1 refused / failed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bato.config import load_config
from bato.deploy import DEFAULT_ARCHIVE, DeployDataError, install, pack, read_manifest
from bato.exceptions import BatoError

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "scripts" / "serve_dashboard.py"
REQUIRED_SITREP = ("Contains modified Copernicus Sentinel data", "© OpenStreetMap contributors.",
                   "deterministic, not AI-generated")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str, body: dict | None = None, timeout: float = 120) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json", "X-BATO-Copilot": "1"} if body is not None else {}
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def verify(config_dir: str | None, archive: Path, timeout_s: float) -> list[str]:
    """Start scripts/serve_dashboard.py as a fresh process and check it end to end. Returns the report lines."""
    manifest = read_manifest(archive)
    aoi, cand = manifest["aoi_id"], manifest["candidate_id"]
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("BATO_LLM_")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    cmd = [sys.executable, str(SERVER), "--host", "127.0.0.1", "--port", str(port), "--check-evidence",
           "--copilot-limits"] + (["--config-dir", config_dir] if config_dir else [])
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}/"
    lines: list[str] = []
    try:
        deadline, health = time.monotonic() + timeout_s, {}
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise DeployDataError(f"the server exited ({proc.returncode}):\n{out}")
            try:
                _code, raw = _get(base + "healthz", timeout=10)
                health = json.loads(raw)
                if health.get("status") in ("ok", "error"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        if health.get("status") != "ok":
            raise DeployDataError(f"/healthz did not become ok: {health or 'no answer'}")
        lines.append(f"OK  /healthz: {health['detail']} ({', '.join(health['evidence_verified'])})")

        code, raw = _get(base)
        if code != 200 or b"<html" not in raw.lower():
            raise DeployDataError(f"the dashboard page did not load (HTTP {code})")
        lines.append("OK  / dashboard page")
        code, raw = _get(base + "bundles/index.json")
        index = json.loads(raw)["bundles"] if code == 200 else []
        if [(e["aoi_id"], e["candidate_id"]) for e in index] != [(aoi, cand)]:
            raise DeployDataError(f"bundles/index.json lists {index}, expected only {aoi}/{cand}")
        lines.append(f"OK  bundles/index.json -> {aoi}/{cand}")
        code, raw = _get(base + f"bundles/{manifest['bundle_path']}/bundle.json")
        bundle = json.loads(raw) if code == 200 else {}
        integ = bundle.get("integrity") or {}
        if (integ.get("official_road_rule") != "v1"
                or integ.get("connectivity_provenance_sha256") != manifest["connectivity_provenance_sha256"]):
            raise DeployDataError("bundle.json is not the official v1 bundle named in the manifest")
        s = bundle["summary"]
        lines.append(f"OK  bundle.json: road rule v1, run {integ['connectivity_run']}, "
                     f"{s.get('blocked_segments')} blocked segments in {s.get('blocked_stretches')} stretches, "
                     f"{s.get('settlements_total')} settlements")
        code, raw = _get(base + "api/copilot/sitrep", {"aoi_id": aoi, "candidate_id": cand, "lang": "en",
                                                       "title": manifest.get("title", aoi)})
        sitrep = json.loads(raw)
        md = sitrep.get("markdown", "")
        if code != 200 or sitrep.get("status") != "deterministic" or f"run {aoi}/{cand} (rule v1)" not in md:
            raise DeployDataError(f"the deterministic SITREP failed (HTTP {code}): {sitrep.get('error', '')}")
        missing = [t for t in REQUIRED_SITREP if t not in md]
        if missing:
            raise DeployDataError(f"the SITREP lacks required text {missing}")
        headline = next((ln for ln in md.splitlines() if "road segments" in ln), "").strip()
        lines.append("OK  English SITREP (deterministic, attribution present): " + headline[:160])
        code, raw = _get(base + "api/copilot/status")
        st = json.loads(raw)
        lines.append(f"OK  copilot status endpoint answers (available={st.get('available')}; the LLM is not "
                     "called by this check)")
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pack = sub.add_parser("pack", help="build the archive from local pipeline outputs")
    p_pack.add_argument("--aoi", required=True)
    p_pack.add_argument("--candidate", required=True)
    p_inst = sub.add_parser("install", help="extract the archive into data/ with SHA-256 checks")
    p_inst.add_argument("--force", action="store_true", help="overwrite differing files (deployment hosts only)")
    p_ver = sub.add_parser("verify", help="start the server on the installed data and check it")
    p_ver.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for /healthz (default 600)")
    for p in (p_pack, p_inst, p_ver):
        p.add_argument("--archive", type=Path, default=REPO / DEFAULT_ARCHIVE)
        p.add_argument("--config-dir", default=None)
    args = parser.parse_args()
    try:
        if args.cmd == "verify":
            for line in verify(args.config_dir, args.archive, args.timeout):
                print(line)
            print("VERIFIED: the server starts and serves the official dashboard data and evidence.")
            return 0
        cfg = load_config(args.config_dir)
        if args.cmd == "pack":
            r = pack(cfg, args.aoi, args.candidate, args.archive)
            raw = sum(f["size"] for f in r.files)
            print(f"Packed {len(r.files)} files ({raw / 2**20:.2f} MiB) from official run {r.connectivity_run} "
                  f"into {r.archive} ({r.archive_bytes / 2**20:.2f} MiB compressed). Largest files:")
            for f in r.files[:8]:
                print(f"  {f['size'] / 2**20:8.2f} MiB  {f['path']}")
            return 0
        res = install(cfg, args.archive, force=args.force)
        m = res["manifest"]
        print(f"Installed {m['aoi_id']}/{m['candidate_id']} (road rule {m['road_rule']}, run "
              f"{m['connectivity_run']}): {res['written']} written, {res['kept']} already present"
              + (f", {res['overwritten']} overwritten" if res["overwritten"] else "") + "; all SHA-256 verified.")
        return 0
    except (BatoError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

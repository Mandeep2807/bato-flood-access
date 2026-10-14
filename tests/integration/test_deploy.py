"""Deployment data and public serving (bato.deploy, scripts/deploy_data.py, scripts/serve_dashboard.py).

A synthetic published world (invented data, NOT real results) is packed, then installed into a FRESH root that
holds only the repository config (what a new clone on Render has, since data/ is Git-ignored), and the real server
is started there the way Render starts it (PORT from the environment, --host 0.0.0.0, --check-evidence)."""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bato.config import load_config
from bato.dashboard.bundle import publish
from bato.deploy import MANIFEST, DeployDataError, install, pack, read_manifest

from dashboard_fakes import inputs_for
from sitrep_fakes import published_sitrep_world

REPO = Path(__file__).resolve().parents[2]
Q = {"aoi_id": "aoi-test", "candidate_id": "ts-test"}


def _world(tmp: Path, **kw):
    mp = pytest.MonkeyPatch()
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    try:
        w = published_sitrep_world(tmp, **kw)
    finally:
        mp.undo()
    publish(w.cfg, "aoi-test", "ts-test", inputs_for(w, rasters=True))
    return w


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    w = _world(tmp_path_factory.mktemp("deploy-world"))
    archive = tmp_path_factory.mktemp("deploy-out") / "bato_official_data.tar.gz"
    report = pack(w.cfg, "aoi-test", "ts-test", archive)
    return w, archive, report


def _fresh(tmp: Path) -> Path:
    """A fresh checkout as far as the server is concerned: the repository config and no data."""
    shutil.copytree(REPO / "config", tmp / "config")
    return tmp / "config"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve(config: Path, args: list[str], env_extra: dict[str, str]) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR" and not k.startswith(("BATO_LLM_",
                                                                                                   "BATO_COPILOT_"))}
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.Popen([sys.executable, str(REPO / "scripts" / "serve_dashboard.py"), "--config-dir",
                             str(config), *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def _call(url: str, body: dict | None = None) -> tuple[int, dict, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "X-BATO-Copilot": "1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), dict(exc.headers)


def _wait_health(base: str, proc: subprocess.Popen, want: str = "ok") -> tuple[int, dict]:
    for _ in range(300):
        assert proc.poll() is None, proc.stdout.read() if proc.stdout else ""
        try:
            code, h, _ = _call(base + "healthz")
            if h.get("status") == want:
                return code, h
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise AssertionError(f"/healthz never reached {want!r}")


def _stop(proc: subprocess.Popen) -> str:
    proc.terminate()
    return proc.communicate(timeout=10)[0]


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def test_pack_holds_only_the_official_read_only_artifacts(packed):
    w, archive, report = packed
    m = read_manifest(archive)
    assert (m["aoi_id"], m["candidate_id"], m["road_rule"], m["connectivity_run"]) == \
        ("aoi-test", "ts-test", "v1", "ts-test")
    names = [f["path"] for f in m["files"]]
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == sorted([MANIFEST, *names])
    assert all(n.startswith(("data/processed/", "data/interim/")) for n in names)
    assert not [n for n in names if n.endswith((".tif", ".tiff", ".parquet", ".zip", ".SAFE"))]
    assert not [n for n in names if n.startswith(("data/raw/", "data/reference/", "data/eval/"))]
    for need in ("connectivity_provenance.json", "road_segments.geojson", "settlements.geojson",
                 "access_destinations.geojson", "blocked_segments_ranked.json", "evidence_records.jsonl"):
        assert f"data/processed/connectivity/aoi-test/ts-test/{need}" in names
    assert "data/processed/dashboard/aoi-test/ts-test/bundle.json" in names
    assert any(n.startswith("data/interim/") and n.endswith("baseline_provenance.json") for n in names)
    assert report.archive_bytes == archive.stat().st_size and report.files[0]["size"] >= report.files[-1]["size"]


def test_pack_with_a_rejected_v2_run_present_ships_only_the_v1_run(tmp_path):
    w = _world(tmp_path / "w", extra_v2=True)
    archive = tmp_path / "a.tar.gz"
    pack(w.cfg, "aoi-test", "ts-test", archive)
    m = read_manifest(archive)
    run_dirs = {f["path"].split("/")[4] for f in m["files"] if f["path"].startswith("data/processed/connectivity/")}
    assert run_dirs == {m["connectivity_run"]} and ".superseded-" in m["connectivity_run"]   # v1, not the v2 dir
    prov = json.loads((w.cfg.settings.paths.processed / "connectivity" / "aoi-test" / "ts-test" /
                       "connectivity_provenance.json").read_text(encoding="utf-8"))
    assert prov["road_rule"] == "v2"                                    # the excluded current directory


def test_pack_refuses_a_bundle_not_built_from_the_official_run(tmp_path):
    w = _world(tmp_path / "w")
    b = w.cfg.settings.paths.processed / "dashboard" / "aoi-test" / "ts-test" / "bundle.json"
    data = json.loads(b.read_text(encoding="utf-8"))
    data["integrity"]["connectivity_provenance_sha256"] = "0" * 64
    b.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DeployDataError, match="not built from the official v1 run"):
        pack(w.cfg, "aoi-test", "ts-test", tmp_path / "a.tar.gz")
    assert not (tmp_path / "a.tar.gz").exists()


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _rewrite(archive: Path, out: Path, edit) -> Path:
    with tarfile.open(archive) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}  # type: ignore[union-attr]
    edit(members)
    with tarfile.open(out, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out


def test_install_rejects_a_file_that_does_not_match_the_manifest(packed, tmp_path):
    _w, archive, _r = packed
    bad = _rewrite(archive, tmp_path / "bad.tar.gz", lambda m: m.update(
        {"data/processed/connectivity/aoi-test/ts-test/settlements.geojson": b'{"type":"FeatureCollection"}'}))
    cfg = load_config(_fresh(tmp_path))
    with pytest.raises(DeployDataError, match="SHA-256"):
        install(cfg, bad)
    assert not (tmp_path / "data").exists()                              # nothing written


def test_install_rejects_paths_outside_the_data_folders(packed, tmp_path):
    _w, archive, _r = packed

    def evil(m: dict[str, bytes]) -> None:
        man = json.loads(m[MANIFEST])
        man["files"].append({"path": "data/reference/emsr927.json", "size": 2, "sha256": "x"})
        m[MANIFEST] = json.dumps(man).encode()

    cfg = load_config(_fresh(tmp_path))
    with pytest.raises(DeployDataError, match="outside"):
        install(cfg, _rewrite(archive, tmp_path / "evil.tar.gz", evil))


def test_install_never_overwrites_different_local_results(packed, tmp_path):
    _w, archive, _r = packed
    cfg = load_config(_fresh(tmp_path))
    assert install(cfg, archive)["written"] > 20
    assert install(cfg, archive)["written"] == 0                         # idempotent
    seg = tmp_path / "data/processed/connectivity/aoi-test/ts-test/road_segments.geojson"
    seg.write_text("{}", encoding="utf-8")
    with pytest.raises(DeployDataError, match="refusing to overwrite"):
        install(cfg, archive)
    assert install(cfg, archive, force=True)["overwritten"] == 1


# ---------------------------------------------------------------------------
# the server in a fresh checkout, started the way Render starts it
# ---------------------------------------------------------------------------


def test_fresh_checkout_serves_the_official_data_publicly(packed, tmp_path):
    _w, archive, _r = packed
    config = _fresh(tmp_path)
    install(load_config(config), archive)
    port = _free_port()
    proc = _serve(config, ["--host", "0.0.0.0", "--check-evidence", "--trust-proxy"],
                  {"PORT": str(port), "BATO_COPILOT_PER_MINUTE": "3"})
    base = f"http://127.0.0.1:{port}/"
    try:
        code, health = _wait_health(base, proc)
        assert code == 200 and health["evidence_verified"] == ["aoi-test/ts-test"]
        code, idx, _ = _call(base + "bundles/index.json")
        assert code == 200 and [b["path"] for b in idx["bundles"]] == ["aoi-test/ts-test"]
        code, bundle, _ = _call(base + "bundles/aoi-test/ts-test/bundle.json")
        assert code == 200 and bundle["integrity"]["official_road_rule"] == "v1"
        code, st, _ = _call(base + "api/copilot/status")
        assert code == 200 and st["available"] is False                 # no provider configured: says so
        code, r, _ = _call(base + "api/copilot/ask", {**Q, "question": "Which roads are blocked?"})
        assert code == 503 and r["status"] == "unavailable"             # clear error, no limit spent
        for _ in range(3):
            client = {"X-Forwarded-For": "203.0.113.7"}
            req = urllib.request.Request(base + "api/copilot/sitrep", method="POST", data=json.dumps(
                {**Q, "lang": "en", "title": "Synthetic"}).encode(), headers={"X-BATO-Copilot": "1", **client})
            with urllib.request.urlopen(req, timeout=30) as resp:
                en = json.loads(resp.read())
            assert en["status"] == "deterministic" and "run aoi-test/ts-test (rule v1)" in en["markdown"]
        code, r, headers = _call(base + "api/copilot/sitrep", {**Q, "lang": "en"})
        assert code == 200                                              # a different client is not limited
        req = urllib.request.Request(base + "api/copilot/sitrep", method="POST", data=json.dumps(
            {**Q, "lang": "en"}).encode(), headers={"X-BATO-Copilot": "1", "X-Forwarded-For": "203.0.113.7"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=30)
        assert err.value.code == 429 and int(err.value.headers["Retry-After"]) >= 1
        assert json.loads(err.value.read())["status"] == "rate_limited"
    finally:
        out = _stop(proc)
    assert f"0.0.0.0:{port}" in out and "Copilot limits: 3/min per client" in out
    assert "Evidence verified: aoi-test/ts-test" in out


def test_health_fails_loudly_when_the_installed_evidence_is_tampered(packed, tmp_path):
    _w, archive, _r = packed
    config = _fresh(tmp_path)
    install(load_config(config), archive)
    seg = tmp_path / "data/processed/connectivity/aoi-test/ts-test/blocked_segments_ranked.json"
    seg.write_text("[]", encoding="utf-8")
    port = _free_port()
    proc = _serve(config, ["--host", "127.0.0.1", "--port", str(port), "--check-evidence"], {})
    try:
        code, health = _wait_health(f"http://127.0.0.1:{port}/", proc, want="error")
        assert code == 500 and "does not match the SHA-256" in health["detail"]
    finally:
        out = _stop(proc)
    assert "EVIDENCE CHECK FAILED" in out


def test_local_default_is_unchanged_loopback_and_unlimited(packed, tmp_path):
    _w, archive, _r = packed
    config = _fresh(tmp_path)
    install(load_config(config), archive)
    port = _free_port()
    proc = _serve(config, ["--port", str(port)], {"PORT": "1", "BATO_COPILOT_PER_MINUTE": "1"})   # --port wins
    base = f"http://127.0.0.1:{port}/"
    try:
        _wait_health(base, proc)
        for _ in range(3):
            assert _call(base + "api/copilot/sitrep", {**Q, "lang": "en"})[0] == 200
    finally:
        out = _stop(proc)
    assert f"http://127.0.0.1:{port}/" in out and "Copilot limits" not in out


def test_cli_install_and_verify_in_a_fresh_root(packed, tmp_path):
    _w, archive, _r = packed
    config = _fresh(tmp_path)
    env = {**{k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR"},
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p),
           "BATO_LLM_API_KEY": "sk-MUST-NOT-APPEAR"}
    cli = [sys.executable, str(REPO / "scripts" / "deploy_data.py")]
    common = ["--archive", str(archive), "--config-dir", str(config)]
    r = subprocess.run([*cli, "install", *common], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "all SHA-256 verified" in r.stdout, r.stderr
    r = subprocess.run([*cli, "verify", *common, "--timeout", "60"], env=env, capture_output=True, text=True,
                       timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERIFIED" in r.stdout and "English SITREP (deterministic, attribution present)" in r.stdout
    assert "sk-MUST-NOT-APPEAR" not in r.stdout + r.stderr

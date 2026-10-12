"""BATO final release-candidate audit: READ-ONLY.

Run from the repository root (the folder with config/ and data/):

    python release_audit.py --manifest data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json

It reads the real repository: the official run, the Step 4C provenance and raster, the connectivity outputs, the
dashboard bundle, the frozen EMSR927 protocol, the metrics and the reference data. It then reports sections A-I as
PASS / FAIL / WARN / REVIEW / INFO. It never writes under data/, src/, config/, docs/, scripts/ or dashboard/. Its
only output is the report: --out, default reports/release_audit/ (git-ignored).

Options:
  --serve-smoke        start scripts/serve_dashboard.py on a free localhost port and fetch the page and the bundle
  --real-data-ui-test  also run the opt-in real-data dashboard test (tests/integration/test_dashboard_real_data.py)

Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL_ID = "bato-emsr927-v1"
EXPECTED_FROZEN_AT = "2026-10-05T08:41:38+00:00"
EXPECTED_YAML_SHA = "a02e49d8846a18602f1f1d46fe985eb3111b0d493d264883e237092bf4dae704"
EXPECTED_DOC_SHA = "872d66f31a8b8ec9fe89b7cbb2130604bef3e38462ecd6c7649dd361e95ee933"
TEXT_EXT = {".md", ".txt", ".py", ".js", ".html", ".css", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules"}
SECRET = re.compile(r"(?i)(api[_-]?key|secret|passw(or)?d|token)[A-Za-z_]*\s*[:=]\s*['\"][A-Za-z0-9_\-\.+/=]{16,}['\"]"
                    r"|(?i:^\s*(CDSE_PASSWORD|CDSE_USERNAME|OHSOME_API_KEY|CDSE_TOTP)\s*=\s*\S+)"
                    r"|eyJ[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY")
NEGATION = re.compile(r"(?i)\b(not|no|never|n't|cannot|NOT CALCULABLE|neither|without|refus|reject|instead of)\b")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.sections: dict[str, dict[str, Any]] = {}
        self.cur = ""

    def section(self, key: str, title: str) -> None:
        self.cur = key
        self.sections[key] = {"title": title, "checks": [], "facts": {}}

    def check(self, status: str, what: str, detail: Any = "") -> None:
        self.sections[self.cur]["checks"].append({"status": status, "check": what, "detail": detail})

    def fact(self, key: str, value: Any) -> None:
        self.sections[self.cur]["facts"][key] = value

    def statuses(self) -> Counter:
        return Counter(c["status"] for s in self.sections.values() for c in s["checks"])


def guarded(rep: Report, what: str, fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - an audit records every failure instead of stopping
        rep.check("FAIL", what, f"{type(exc).__name__}: {exc}")
        return None


def text_files(root: Path, roots: list[str]) -> list[Path]:
    out = []
    for r in roots:
        base = root / r
        if base.is_file():
            out.append(base)
            continue
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in TEXT_EXT and not (set(p.parts) & SKIP_DIRS) and \
                    "vendor" not in p.parts and "release_audit" not in p.parts and p.stat().st_size < 5_000_000:
                out.append(p)
    return out


def grep(files: list[Path], root: Path, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    hits = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                hits.append({"file": f.relative_to(root).as_posix(), "line": i, "text": line.strip()[:240]})
    return hits


# ---------------------------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="BATO release-candidate audit (read-only).")
    ap.add_argument("--manifest", type=Path, default=Path("data/raw/manifests/aoi-e893707c2beb_fetch_manifest.json"))
    ap.add_argument("--aoi-id", default=None)
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--serve-smoke", action="store_true")
    ap.add_argument("--real-data-ui-test", action="store_true")
    ap.add_argument("--expected-yaml-sha", default=EXPECTED_YAML_SHA)
    ap.add_argument("--expected-doc-sha", default=EXPECTED_DOC_SHA)
    ap.add_argument("--expected-frozen-at", default=EXPECTED_FROZEN_AT)
    args = ap.parse_args()

    try:
        from bato.config import load_config
        cfg = load_config(args.config_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot load BATO config: {exc}", file=sys.stderr)
        return 2
    root = cfg.config_dir.parent.resolve()
    paths = cfg.settings.paths
    out_dir = (args.out or root / "reports" / "release_audit").resolve()
    for forbidden in (paths.raw, paths.interim, paths.processed, paths.reference, paths.eval, root / "src",
                      root / "config", root / "docs", root / "dashboard", root / "scripts"):
        fb = Path(forbidden).resolve()
        if out_dir == fb or fb in out_dir.parents:
            print(f"refusing to write the report inside {fb}", file=sys.stderr)
            return 2

    aoi_id, cand_id = args.aoi_id, args.candidate_id
    if not (aoi_id and cand_id):
        from bato.contracts.fetch import FetchManifest
        from bato.contracts.io import load_contract
        from bato.data_rules import assert_pipeline_input_allowed
        from bato.preprocessing.stack import select_candidate
        m = load_contract(FetchManifest, assert_pipeline_input_allowed(args.manifest, paths))
        aoi_id, cand_id = m.aoi_id, select_candidate(m, cand_id).candidate_id

    tree_before = snapshot_protected(root, paths)
    rep = Report()
    ctx: dict[str, Any] = {"root": root, "cfg": cfg, "paths": paths, "aoi": aoi_id, "cand": cand_id}
    audit_a(rep, ctx)
    audit_b(rep, ctx)
    audit_c(rep, ctx)
    audit_d(rep, ctx, args)
    audit_e(rep, ctx)
    audit_f(rep, ctx, args)
    audit_g(rep, ctx)
    audit_h(rep, ctx)
    audit_i(rep, ctx)
    after = snapshot_protected(root, paths)
    rep.section("Z", "Audit self-check")
    rep.check("PASS" if after == tree_before else "FAIL", "audit changed no file under data/, src/, config/, docs/, "
              "scripts/, dashboard/", "" if after == tree_before else sorted(set(after.items()) ^ set(tree_before.items())))

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"generated_at_utc": stamp, "aoi_id": aoi_id, "candidate_id": cand_id, "root": str(root),
               "sections": rep.sections, "status_counts": dict(rep.statuses())}
    (out_dir / f"release_audit_{stamp}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = render(payload)
    (out_dir / f"release_audit_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nReport: {out_dir / f'release_audit_{stamp}.md'}")
    return 1 if rep.statuses().get("FAIL") else 0


def snapshot_protected(root: Path, paths: Any) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file the audit must not change (cheap; hashes are taken where it matters)."""
    out = {}
    for base in (paths.raw, paths.interim, paths.processed, paths.reference, paths.eval, root / "src",
                 root / "config", root / "docs", root / "dashboard", root / "scripts"):
        b = Path(base)
        if b.is_dir():
            for p in b.rglob("*"):
                if p.is_file() and not (set(p.parts) & SKIP_DIRS):
                    st = p.stat()
                    out[p.as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


# --------------------------------------------------------------------------------------------------------- A


def audit_a(rep: Report, ctx: dict[str, Any]) -> None:
    from bato.dashboard import bundle as B
    rep.section("A", "Production run identity")
    cfg, root, aoi, cand = ctx["cfg"], ctx["root"], ctx["aoi"], ctx["cand"]
    rep.fact("aoi_id", aoi)
    rep.fact("candidate_id", cand)
    runs = guarded(rep, "list connectivity runs", lambda: B.find_connectivity_runs(cfg, aoi, cand)) or []
    rep.fact("connectivity_runs", [{"dir": r.directory.name, "road_rule": r.road_rule, "finished_at": r.finished_at}
                                   for r in runs])
    sel = guarded(rep, "select official run (latest road-rule v1)", lambda: B.select_official_run(runs))
    if not sel:
        return
    run, notes = sel
    ctx["run"] = run
    rep.fact("official_connectivity_run", {"dir": run.directory.relative_to(root).as_posix(), "road_rule": run.road_rule,
                                           "finished_at": run.finished_at, "selection_notes": notes})
    rep.check("PASS" if run.road_rule == "v1" else "FAIL", "official run is road rule v1", run.road_rule)
    rep.check("PASS" if run.provenance.get("validated") is False else "FAIL", "connectivity provenance says "
              "validated: false", run.provenance.get("validated"))
    files = guarded(rep, "official run outputs match their provenance SHA-256", lambda: B.verify_run_outputs(run))
    if files:
        rep.check("PASS", "official run outputs match their provenance SHA-256", sorted(files))
        ctx["run_files"] = files
    inputs = run.provenance.get("inputs", {})
    for key, what in (("stack_provenance", "stack provenance"), ("baseline_provenance", "Step 4C provenance"),
                      ("baseline_state", "Step 4C state raster")):
        p = guarded(rep, f"{what} cited by the official run matches its SHA-256",
                    lambda k=key, w=what: B.verify_root_file(cfg, root, inputs[k], w))
        if p:
            rep.check("PASS", f"{what} cited by the official run matches its SHA-256", inputs[key]["path"])
            ctx[key] = p
    if "debris_provenance" in inputs:
        rep.check("FAIL", "official run does not use the rejected disturbance (debris) input", inputs["debris_provenance"])
    else:
        rep.check("PASS", "official run does not use the rejected disturbance (debris) input")
    v2 = [r.directory.name for r in runs if r.road_rule != "v1"]
    rep.check("INFO" if not v2 else "WARN", "road-rule v2 (rejected) runs present on disk", v2 or "none")
    # dashboard -> same run
    bdir = Path(ctx["paths"].processed) / "dashboard" / aoi / cand
    bundle_p = bdir / "bundle.json"
    if not bundle_p.is_file():
        rep.check("FAIL", "dashboard bundle exists", str(bundle_p))
        return
    bundle = json.loads(bundle_p.read_text(encoding="utf-8"))
    ctx["bundle"], ctx["bundle_path"] = bundle, bundle_p
    integ = bundle.get("integrity", {})
    prov_sha = B.sha256_file(run.directory / B.PROVENANCE_FILE)
    rep.fact("dashboard_bundle", {"path": bundle_p.relative_to(root).as_posix(), "built_at": bundle.get("built_at"),
                                  "connectivity_run": integ.get("connectivity_run")})
    rep.check("PASS" if integ.get("connectivity_run") == run.directory.name else "FAIL",
              "dashboard bundle was built from the official run", integ.get("connectivity_run"))
    rep.check("PASS" if integ.get("connectivity_provenance_sha256") == prov_sha else "FAIL",
              "bundle's connectivity provenance SHA-256 equals the official run's", prov_sha)
    outs = {n: r["sha256"] for n, r in run.provenance.get("outputs", {}).items()}
    rep.check("PASS" if integ.get("outputs_verified") == outs else "FAIL",
              "bundle's verified output hashes equal the official run's output hashes")
    if "baseline_provenance" in ctx:
        rep.check("PASS" if integ.get("baseline_provenance_sha256") == sha(ctx["baseline_provenance"]) else "FAIL",
                  "bundle's Step 4C provenance SHA-256 equals the file the official run cites")
    if "stack_provenance" in ctx:
        rep.check("PASS" if integ.get("stack_provenance_sha256") == sha(ctx["stack_provenance"]) else "FAIL",
                  "bundle's stack provenance SHA-256 equals the file the official run cites")
    sup = sorted(p.name for p in bdir.parent.glob(f"{cand}.superseded-*")) if bdir.parent.is_dir() else []
    rep.check("INFO", "superseded dashboard bundles on disk (never served: index skips them)", sup or "none")
    idx = Path(ctx["paths"].processed) / "dashboard" / "index.json"
    if idx.is_file():
        entries = json.loads(idx.read_text(encoding="utf-8")).get("bundles", [])
        rep.check("PASS" if any(e.get("path") == f"{aoi}/{cand}" for e in entries) else "FAIL",
                  "dashboard index lists the official bundle", entries)


# --------------------------------------------------------------------------------------------------------- B


def audit_b(rep: Report, ctx: dict[str, Any]) -> None:
    rep.section("B", "Step 4C facts")
    p = ctx.get("baseline_provenance")
    if not p:
        rep.check("FAIL", "Step 4C provenance available", "not verified in A")
        return
    prov = json.loads(p.read_text(encoding="utf-8"))
    st = prov.get("statistics", {})
    sc = st.get("state_counts", {})
    na = {k: v for k, v in sc.items() if k.startswith("NOT_ASSESSED")}
    for k in ("total_pixels", "assessed_pixels", "not_assessed_pixels", "flood_pixels", "no_flood_pixels",
              "pre_event_water_pixels", "flood_fraction_of_assessed", "flood_fraction_of_all",
              "not_assessed_fraction_of_all"):
        rep.fact(k, st.get(k))
    rep.fact("not_assessed_by_reason", na)
    rep.fact("state_counts", sc)
    rep.fact("validated", prov.get("validated"))
    rep.fact("validation_note", prov.get("validation_note"))
    rep.fact("version", prov.get("version"))
    rep.fact("thresholds", [{"name": t.get("name"), "value": t.get("value"), "provisional": t.get("provisional")}
                            for t in prov.get("thresholds", [])])
    need = ("total_pixels", "assessed_pixels", "not_assessed_pixels", "state_counts", "flood_pixels",
            "flood_fraction_of_assessed")
    missing = [k for k in need if k not in st] + ([] if "validated" in prov else ["validated"])
    rep.check("PASS" if not missing else "FAIL", "Step 4C provenance carries every audited field", missing or "all")
    if missing:
        return
    rep.check("PASS" if prov.get("validated") is False else "FAIL", "Step 4C provenance says validated: false")
    rep.check("PASS" if all(t.get("provisional") for t in prov.get("thresholds", [])) else "WARN",
              "every Step 4C threshold is marked provisional")
    ok = (st.get("assessed_pixels", 0) + st.get("not_assessed_pixels", -1) == st.get("total_pixels")
          and sum(sc.values()) == st.get("total_pixels") and sum(na.values()) == st.get("not_assessed_pixels"))
    rep.check("PASS" if ok else "FAIL", "pixel counts add up (assessed + not assessed = total; reasons sum)")
    if st.get("assessed_pixels"):
        frac = st["flood_pixels"] / st["assessed_pixels"]
        rep.check("PASS" if abs(frac - (st.get("flood_fraction_of_assessed") or -1)) < 1e-6 else "FAIL",
                  "flood fraction of assessed = flood / assessed", round(frac, 8))
    run = ctx.get("run")
    state_ref = run.provenance["inputs"]["baseline_state"] if run else {}
    outs = prov.get("outputs", {})
    rep.check("PASS" if outs.get("state_sha256") == state_ref.get("sha256") else "FAIL",
              "Step 4C state hash in its provenance equals the hash the official connectivity run used",
              {"step4c": outs.get("state_sha256"), "connectivity": state_ref.get("sha256")})
    if ctx.get("baseline_state"):
        rep.check("PASS" if sha(ctx["baseline_state"]) == outs.get("state_sha256") else "FAIL",
                  "Step 4C state raster on disk is the one recorded (not silently replaced)")
        try:
            import numpy as np
            import rasterio
            with rasterio.open(ctx["baseline_state"]) as ds:
                vals, cnt = np.unique(ds.read(1), return_counts=True)
            from bato.contracts.baseline import BaselineState
            names = {int(s): s.name for s in BaselineState}
            recount = {names.get(int(v), f"UNKNOWN_{int(v)}"): int(c) for v, c in zip(vals, cnt, strict=True)}
            rep.check("PASS" if {k: v for k, v in recount.items() if v} == {k: v for k, v in sc.items() if v}
                      else "FAIL", "raster recount equals the provenance state counts", recount)
        except ImportError:
            rep.check("WARN", "raster recount", "rasterio not installed; provenance counts not re-counted")
    others = []
    det = Path(ctx["paths"].interim) / "detection" / ctx["aoi"]
    if det.is_dir():
        others = sorted(x.relative_to(det).as_posix() for x in det.glob("*/baseline*") if x.is_dir())
    rep.check("INFO", "Step 4C output directories on disk for this AOI", others)


# --------------------------------------------------------------------------------------------------------- C


def audit_c(rep: Report, ctx: dict[str, Any]) -> None:
    from bato.dashboard import bundle as B
    rep.section("C", "Connectivity facts (official v1 run)")
    run, files = ctx.get("run"), ctx.get("run_files")
    if not run or not files:
        rep.check("FAIL", "official run verified in A")
        return
    prov = run.provenance
    s = prov.get("summary", {})
    seg = B.parse_segments(json.loads(files[B.SEGMENTS_FILE].read_text(encoding="utf-8")))
    sett = json.loads(files[B.SETTLEMENTS_FILE].read_text(encoding="utf-8"))["features"]
    re_ = B.recompute_summary(seg, sett)
    bad = B.check_summary(s, re_)
    rep.check("PASS" if not bad else "FAIL", "provenance summary equals counts recomputed from the published layers",
              bad or "consistent")
    rep.fact("road_segments_total", len(seg.ids))
    rep.fact("road_segments_by_state", re_["road_segments_by_state"])
    rep.fact("road_km_by_state_recomputed", {k: round(v, 3) for k, v in re_["road_km_by_state"].items()})
    rep.fact("road_km_by_state_provenance", s.get("road_km_by_state"))
    rep.fact("blocked_stretches", s.get("blocked_stretches"))
    rep.fact("settlements_total", re_["settlements_total"])
    rep.fact("settlements_by_access_state", re_["settlements_by_access_state"])
    rep.fact("settlements_by_status", s.get("settlements_by_status"))
    rep.fact("reachable_settlements", re_["reachable_settlements"])
    rep.fact("building_count_by_access_state (proxy, not population)", s.get("building_count_by_access_state"))
    rep.fact("osm_snapshot", prov.get("osm_snapshot"))
    acc = re_["settlements_by_access_state"]
    ctx["access"] = acc
    rep.fact("confirmed_cutoff", acc.get("CONFIRMED_CUTOFF", 0))
    rep.fact("possible_cutoff_access_unverified", acc.get("POSSIBLE_CUTOFF", 0))
    rep.fact("no_pre_event_access", acc.get("NO_PRE_EVENT_ACCESS", 0))
    rep.fact("connected", acc.get("CONNECTED", 0))
    # terminology: a count of possible cut-offs must never be called cut off
    root = ctx["root"]
    files_t = text_files(root, ["README.md", "docs", "dashboard", "reports"])
    pc, cc = acc.get("POSSIBLE_CUTOFF", 0), acc.get("CONFIRMED_CUTOFF", 0)
    pat = re.compile(r"(?i)(\d[\d,]*)\s+(villages?|settlements?|communities)\b[^.\n]{0,40}?\bcut[\s-]*off")
    flagged, seen = [], []
    for h in grep(files_t, root, pat):
        for m in pat.finditer(h["text"]):
            n = int(m.group(1).replace(",", ""))
            seen.append({**h, "number": n})
            if n == pc and n != cc and not NEGATION.search(h["text"]):
                flagged.append({**h, "number": n})
    rep.check("PASS" if not flagged else "FAIL",
              f"no text calls the {pc} access-unverified settlements 'cut off' (confirmed cut off = {cc})",
              flagged or f"{len(seen)} numeric 'cut off' mentions reviewed")
    if ctx.get("bundle"):
        bs = ctx["bundle"].get("summary", {})
        rep.check("PASS" if bs.get("settlements_by_access_state", {}) and
                  {k: v for k, v in bs["settlements_by_access_state"].items() if v} ==
                  {k: v for k, v in acc.items() if v} else "FAIL",
                  "dashboard bundle settlement counts equal the official run")


# --------------------------------------------------------------------------------------------------------- D


def audit_d(rep: Report, ctx: dict[str, Any], args: argparse.Namespace) -> None:
    rep.section("D", "EMSR927 evaluation (frozen)")
    root, paths = ctx["root"], ctx["paths"]
    from bato.evaluation import leakage
    from bato.evaluation.protocol import load_protocol
    yp, dp = root / "config/evaluation/emsr927_protocol.yaml", root / "docs/evaluation/emsr927_protocol.md"
    ysha = sha(yp) if yp.is_file() else None
    dsha = sha(dp) if dp.is_file() else None
    rep.fact("protocol_yaml_sha256", ysha)
    rep.fact("protocol_doc_sha256", dsha)
    rep.check("PASS" if ysha == args.expected_yaml_sha else "FAIL", "frozen protocol YAML SHA-256 unchanged", ysha)
    rep.check("PASS" if dsha == args.expected_doc_sha else "FAIL", "frozen protocol document SHA-256 unchanged", dsha)
    p = guarded(rep, "load protocol", lambda: load_protocol(root))
    if p:
        rep.fact("protocol_id", p.protocol_id)
        rep.fact("status", p.status)
        rep.fact("frozen_at", p.raw.get("frozen_at"))
        rep.check("PASS" if p.protocol_id == EXPECTED_PROTOCOL_ID else "FAIL", "protocol_id", p.protocol_id)
        rep.check("PASS" if p.status == "frozen" else "FAIL", "protocol status is frozen", p.status)
        rep.check("PASS" if str(p.raw.get("frozen_at")) == args.expected_frozen_at else "FAIL", "frozen_at unchanged",
                  p.raw.get("frozen_at"))
    ev = Path(paths.eval) / "EMSR927"
    fr = ev / "protocol_freeze.json"
    if fr.is_file():
        rec = json.loads(fr.read_text(encoding="utf-8"))
        rep.check("PASS" if rec["yaml"]["sha256"] == ysha and rec["doc"]["sha256"] == dsha else "FAIL",
                  "freeze record hashes equal the protocol files", rec)
        rep.check("INFO", "freeze record discloses prior runs", rec.get("prior_runs_disclosed", "none"))
    else:
        rep.check("FAIL", "freeze record exists", str(fr))
    runs_p = ev / "runs.jsonl"
    runs = [json.loads(x) for x in runs_p.read_text(encoding="utf-8").splitlines() if x.strip()] if runs_p.is_file() \
        else []
    rep.fact("evaluation_runs", runs)
    rep.check("PASS" if runs else "FAIL", "at least one logged evaluation run", len(runs))
    if p:
        mdir = ev / p.combined_sha256[:12]
        mfiles = sorted(mdir.glob("metrics*.json")) if mdir.is_dir() else []
        rep.check("PASS" if (mdir / "metrics.json").is_file() else "FAIL", "metrics.json exists for the frozen protocol",
                  str(mdir.relative_to(root)))
        rep.check("PASS" if len(mfiles) == 1 else "WARN", "a single metrics result under this protocol (a second file "
                  "means a non-reproduced re-run)", [m.name for m in mfiles])
        if (mdir / "metrics.json").is_file():
            m = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
            ctx["metrics"], ctx["metrics_path"] = m, mdir / "metrics.json"
            pr = m.get("protocol", {})
            rep.check("PASS" if pr.get("yaml_sha256") == ysha and pr.get("doc_sha256") == dsha else "FAIL",
                      "metrics.json is tied to the frozen protocol hashes", pr)
            msha = sha(mdir / "metrics.json")
            rep.check("PASS" if any(r.get("metrics_sha256") == msha for r in runs) else "FAIL",
                      "metrics.json SHA-256 is logged in runs.jsonl", msha)
            rep.check("PASS" if m.get("leakage_scan", {}).get("hits") == 0 else "FAIL",
                      "leakage scan recorded in metrics.json = 0 hits", m.get("leakage_scan"))
            changed = [f for f, h in (m.get("bato", {}).get("files_read_sha256") or {}).items()
                       if not (root / f).is_file() or sha(root / f) != h]
            rep.check("PASS" if not changed else "FAIL", "BATO production files read by the evaluation are unchanged "
                      "since (SHA-256)", changed or len(m["bato"]["files_read_sha256"]))
            if ctx.get("run"):
                rep.check("PASS" if m.get("bato", {}).get("connectivity_run") == ctx["run"].directory.name else "FAIL",
                          "evaluation used the official connectivity run", m.get("bato", {}).get("connectivity_run"))
    lk = guarded(rep, "leakage scan now", lambda: leakage.scan(paths, root))
    if lk is not None:
        rep.fact("leakage_scan_now", {"json_files_scanned": lk["json_files_scanned"], "hits": len(lk["hits"])})
        rep.check("PASS" if not lk["hits"] else "FAIL", "leakage scan now = 0 hits", lk["hits"][:10] or
                  f"{lk['json_files_scanned']} files scanned")
    from bato.evaluation.acquisition import load_manifest, verify_downloads
    am = guarded(rep, "EMSR927 acquisition manifest", lambda: load_manifest(paths))
    if am:
        z = guarded(rep, "reference zips match their recorded SHA-256", lambda: verify_downloads(paths, am))
        if z is not None:
            rep.check("PASS", "reference zips match their recorded SHA-256", [x.name for x in z])


# --------------------------------------------------------------------------------------------------------- E


def audit_e(rep: Report, ctx: dict[str, Any]) -> None:
    rep.section("E", "EMSR927 interpretation (from metrics.json and the protocol)")
    m = ctx.get("metrics")
    if not m:
        rep.check("FAIL", "metrics.json available", "see D")
        return
    pe = m.get("primary_evaluation", {})
    rep.check("PASS" if "road" in str(pe.get("what", "")).lower() and pe.get("primary_comparison") ==
              "primary_destroyed" else "FAIL", "road comparison is the primary evaluation", pe)
    facts: dict[str, Any] = {}
    flood_ok, landslide_ok, domain_ok, cov_ok, bridge_ok = True, True, True, True, True
    for p in m.get("per_product", []):
        name = p.get("product")
        fw = p.get("comparisons", {}).get("flood_water", {})
        flood_ok &= fw.get("value") == "NOT CALCULABLE"
        landslide_ok &= set(p.get("event_classes_present", {})) <= {"reference_mass_movement"}
        landslide_ok &= "Landslide" not in json.dumps(p.get("event_classes_present", {}).get("flood_water", {}))
        bridge_ok &= "does not currently generate" in str(p.get("bridges", {}).get("note", ""))
        roads = p.get("roads", {})
        pf: dict[str, Any] = {"ems_aoi_overlap_with_bato_aoi_km2": p.get("ems_aoi_overlap_with_bato_aoi_km2"),
                              "event_classes_present": p.get("event_classes_present"),
                              "bridges": p.get("bridges", {}).get("features_in_area_by_grade")}
        for cname, c in (roads.get("comparisons") or {}).items():
            domain_ok &= "NOT full-network recall" in str(c.get("metric_domain", ""))
            cov_ok &= "reference_coverage" in c
            pf[cname] = {"role": c.get("role"), "segments_by_reference_label": c.get("segments_by_reference_label"),
                         "modes": {md: {"counts": v["counts"], **{k: v["strict"][k].get("value") for k in
                                                                    ("precision", "recall", "f1")}}
                                   for md, v in (c.get("modes") or {}).items()} or c.get("reason"),
                         "reference_coverage": c.get("reference_coverage"),
                         "descriptive_lengths_m": c.get("descriptive_lengths_m")}
        mm = p.get("comparisons", {}).get("reference_mass_movement", {})
        if mm.get("value") != "NOT CALCULABLE":
            pf["mass_movement_overlap (class-mismatched)"] = {
                k: (mm.get(k, {}).get("value") if isinstance(mm.get(k), dict) else mm.get(k))
                for k in ("footprint_pixels", "bato_flood_pixels", "overlap_pixels", "footprint_share_marked_bato_flood",
                          "footprint_share_assessed")}
        facts[name] = pf
    rep.fact("per_product", facts)
    rep.fact("pooled_roads", {k: {"modes": {md: {"counts": v["counts"], **{x: v["strict"][x].get("value") for x in
                                                                         ("precision", "recall", "f1")}}
                                            for md, v in (c.get("modes") or {}).items()},
                                  "reference_coverage": c.get("reference_coverage")}
                              for k, c in (m.get("pooled", {}).get("roads") or {}).items() if isinstance(c, dict)})
    rep.fact("time_gaps", m.get("time", {}).get("gaps_ems_image_minus_bato_post_hours"))
    rep.fact("not_calculable", m.get("not_calculable"))
    rep.check("PASS" if flood_ok else "FAIL", "flood precision/recall/F1/IoU NOT CALCULABLE in every product")
    rep.check("PASS" if landslide_ok else "FAIL", "Landslide only as reference_mass_movement, never flood")
    rep.check("PASS" if domain_ok else "FAIL", "road metrics labelled matched-domain, not network-wide recall")
    rep.check("PASS" if cov_ok else "FAIL", "reference coverage reported for every road comparison")
    rep.check("PASS" if bridge_ok else "FAIL", "bridges are context only (no BATO bridge layer)")
    nc = m.get("not_calculable", {})
    need = {"flood_precision_recall_f1_iou", "bridge_precision_recall", "building_precision_recall",
            "settlement_isolation_precision_recall", "accuracy"}
    rep.check("PASS" if need <= set(nc) else "FAIL", "building / bridge / settlement / accuracy not claimed",
              sorted(need - set(nc)) or "all listed")
    sup = [c for p in m.get("per_product", []) for c in (p.get("roads", {}).get("comparisons") or {}).values()
           if c.get("role") == "supplementary"]
    rep.check("PASS" if all("not accuracy" in c.get("label", "") for c in sup) else "FAIL",
              "supplementary road comparison labelled 'not accuracy, not ground truth'")


# --------------------------------------------------------------------------------------------------------- F


def audit_f(rep: Report, ctx: dict[str, Any], args: argparse.Namespace) -> None:
    from bato.evaluation.leakage import FLAG, _about_reference, _strings
    rep.section("F", "Dashboard")
    root = ctx["root"]
    b = ctx.get("bundle")
    if b:
        hits = [(jp, s[:120]) for jp, s in _strings(b) if FLAG.search(s) and not _about_reference(jp)]
        rep.check("PASS" if not hits else "REVIEW", "bundle never presents EMSR927 / Copernicus EMS as an input or "
                  "truth source (mentions outside 'not used' statements)", hits[:15] or "none")
        urls = sorted({u for _, s in _strings(b) for u in re.findall(r"https?://[^\s\"')]+", s)})
        rep.check("PASS" if not any(re.search(r"tile|basemap|carto|arcgis|mapbox|tile\.openstreetmap", u) for u in urls)
                  else "FAIL", "bundle references no online map tiles", urls or "no URLs")
    app = root / "dashboard" / "app"
    if app.is_dir():
        app_files = [p for p in app.rglob("*") if p.is_file() and "vendor" not in p.parts
                     and p.suffix in (".js", ".html", ".css")]
        u = grep(app_files, root, re.compile(r"https?://"))
        tiles = [h for h in u if re.search(r"(?i)tile|basemap|carto|arcgis|mapbox|api[_-]?key", h["text"])]
        rep.check("PASS" if not tiles else "FAIL", "app code has no online tile / API-key dependency", tiles or
                  f"{len(u)} other URL mentions")
        emsr = grep(app_files, root, re.compile(r"(?i)emsr|copernicus ems|rapidmapping"))
        rep.check("PASS" if not emsr else "REVIEW", "app code does not mention EMSR927", emsr or "none")
    if args.serve_smoke:
        serve_smoke(rep, ctx)
    else:
        rep.check("INFO", "HTTP serve smoke test", "not run (pass --serve-smoke)")
    if args.real_data_ui_test:
        env = {**os.environ, "BATO_REAL_DATA_MANIFEST": str(args.manifest)}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/integration/test_dashboard_real_data.py",
                            "-p", "no:cacheprovider"], cwd=root, env=env, capture_output=True, text=True, timeout=900)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        rep.check("PASS" if r.returncode == 0 and "passed" in last and "skipped" not in last else "FAIL",
                  "real-data dashboard test (official bundle in a browser)", last)
    else:
        rep.check("INFO", "real-data dashboard UI test", "not run (pass --real-data-ui-test)")


def serve_smoke(rep: Report, ctx: dict[str, Any]) -> None:
    root = ctx["root"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(root / "scripts" / "serve_dashboard.py"), "--port", str(port),
                             "--config-dir", str(ctx["cfg"].config_dir)], cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        ok = False
        for _ in range(50):
            try:
                with urllib.request.urlopen(base + "/", timeout=2) as r:
                    ok = r.status == 200 and b"<html" in r.read(4096).lower()
                break
            except OSError:
                time.sleep(0.2)
        rep.check("PASS" if ok else "FAIL", "dashboard page served on localhost", base + "/")
        try:
            rel = f"/bundles/{ctx['aoi']}/{ctx['cand']}/bundle.json"
            with urllib.request.urlopen(base + rel, timeout=10) as r:
                served = json.loads(r.read())
            same = served.get("integrity") == (ctx.get("bundle") or {}).get("integrity")
            rep.check("PASS" if same else "FAIL", "served bundle is the official bundle", rel)
        except OSError as exc:
            rep.check("FAIL", "official bundle served", str(exc))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------------------------------------- G


def audit_g(rep: Report, ctx: dict[str, Any]) -> None:
    rep.section("G", "Data leakage and submission hygiene (recommendations only; nothing deleted)")
    root, paths = ctx["root"], ctx["paths"]
    ref = Path(paths.reference)
    ref_files = [p for p in ref.rglob("*") if p.is_file() and p.name.lower() != "readme.md"] if ref.is_dir() else []
    by_size: dict[int, list[Path]] = {}
    for p in ref_files:
        by_size.setdefault(p.stat().st_size, []).append(p)
    ref_hash = {sha(p): p for ps in by_size.values() for p in ps}
    copies, named = [], []
    for base in (paths.raw, paths.interim, paths.processed):
        b = Path(base)
        if not b.is_dir():
            continue
        for p in b.rglob("*"):
            if not p.is_file():
                continue
            if re.search(r"(?i)emsr|rapidmapping|copernicus[_-]?ems|unosat", p.name):
                named.append(p.relative_to(root).as_posix())
            if p.stat().st_size in by_size and sha(p) in ref_hash:
                copies.append({"file": p.relative_to(root).as_posix(),
                               "same_bytes_as": ref_hash[sha(p)].relative_to(root).as_posix()})
    rep.check("PASS" if not copies else "FAIL", "no reference file (byte-identical) inside raw/interim/processed",
              copies or f"{len(ref_files)} reference files checked")
    rep.check("PASS" if not named else "FAIL", "no EMSR/EMS/UNOSAT-named file inside raw/interim/processed",
              named or "none")
    run = ctx.get("run")
    if run:
        osm = run.provenance.get("osm_snapshot", {})
        try:
            ok = date.fromisoformat(str(osm.get("snapshot_date"))) < date.fromisoformat(str(osm.get("event_date")))
        except ValueError:
            ok = False
        rep.check("PASS" if ok else "FAIL", "official run used a PRE-event OSM snapshot", osm)
    osm_dir = Path(paths.raw) / "osm" / ctx["aoi"]
    if osm_dir.is_dir() and run:
        ev = str(run.provenance.get("event_date"))
        post = [d.name for d in osm_dir.glob("snapshot-*") if d.name.replace("snapshot-", "") >= ev]
        rep.check("PASS" if not post else "WARN", "no post-event OSM snapshot on disk for this AOI", post or "none")
    # repository files: secrets, caches, large files, tracked data
    tracked = None
    try:
        r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            tracked = [x for x in r.stdout.splitlines() if x]
    except (OSError, subprocess.TimeoutExpired):
        pass
    rep.fact("git_repository", tracked is not None)
    cand = [root / t for t in tracked] if tracked is not None else \
        [p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS)
         and "data" not in p.relative_to(root).parts[:1]]
    secrets = []
    for p in cand:
        if p.suffix.lower() in TEXT_EXT | {".env", ""} and p.is_file() and p.stat().st_size < 5_000_000:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if SECRET.search(line) and "example" not in p.name:
                    secrets.append({"file": p.relative_to(root).as_posix(), "line": i})
    rep.check("PASS" if not secrets else "FAIL", "no secrets / API keys in " + ("tracked" if tracked is not None else
              "repository") + " files", secrets[:20] or "none")
    env = root / ".env"
    if env.is_file():
        rep.check("FAIL" if tracked is not None and ".env" in tracked else "INFO", ".env present",
                  "TRACKED BY GIT" if tracked is not None and ".env" in tracked else "present locally, not tracked")
    if tracked is not None:
        bad = [t for t in tracked if re.search(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)/|\.pyc$|"
                                               r"\.part$|\.tmp$|\.DS_Store$", t)]
        data_t = [t for t in tracked if t.startswith("data/") and not t.endswith("README.md")]
        rep.check("PASS" if not bad else "WARN", "no caches / temp files tracked by git", bad[:20] or "none")
        rep.check("PASS" if not data_t else "FAIL", "no data files tracked by git (data/** is git-ignored)",
                  data_t[:20] or "none")
    big = [{"file": p.relative_to(root).as_posix(), "MB": round(p.stat().st_size / 1e6, 1)} for p in cand
           if p.is_file() and p.stat().st_size > 25_000_000]
    rep.check("PASS" if not big else "WARN", "no file over 25 MB in the submitted (non-data) tree", big or "none")
    stale = []
    for base, pat in ((Path(paths.processed) / "connectivity", "*/*.superseded-*"),
                      (Path(paths.processed) / "dashboard", "*/*.superseded-*"),
                      (Path(paths.processed) / "dashboard", "*/*.tmp-*")):
        if base.is_dir():
            stale += [p.relative_to(root).as_posix() for p in base.glob(pat)]
    rep.check("INFO", "superseded / temporary output directories (keep or archive; the official selection ignores "
              "them except older v1 runs)", stale or "none")


# --------------------------------------------------------------------------------------------------------- H


def audit_h(rep: Report, ctx: dict[str, Any]) -> None:
    rep.section("H", "Reproducibility documentation")
    root = ctx["root"]
    readme = (root / "README.md").read_text(encoding="utf-8", errors="replace") if (root / "README.md").is_file() else ""
    checks = [
        ("README gives the pipeline commands (fetch, preprocess, stack, Step 4C, OSM, connectivity)",
         all(s in readme for s in ("fetch_s1_grd.py", "stack_s1.py", "run_baseline_detection.py",
                                   "fetch_osm_snapshot.py", "run_connectivity.py"))),
        ("README tells how to reproduce the OFFICIAL connectivity run (--road-rule v1; the script default is v2)",
         bool(re.search(r"run_connectivity\.py[^\n]*--road-rule v1", readme))),
        ("README gives the dashboard commands", "build_dashboard.py" in readme and "serve_dashboard.py" in readme),
        ("README explains how to run the EMSR927 evaluation separately",
         "evaluate_emsr927.py" in readme and "fetch_emsr927.py" in readme),
        ("README separates production inputs from evaluation-only references",
         "evaluation-only" in readme.lower() and "Not allowed as inputs" in readme),
    ]
    for what, ok in checks:
        rep.check("PASS" if ok else "FAIL", what)
    stale = [("README says EMSR927 evaluation is NOT IMPLEMENTED", re.search(r"EMSR927 comparison\)\s*\|\s*\*\*NOT "
                                                                            r"IMPLEMENTED", readme)),
             ("README says no flood/damage result comes from imagery",
              re.search(r"no flood/damage result in this repository comes from imagery", readme)),
             ("README lists road rule v2 as the default without saying it was rejected",
              re.search(r"--road-rule v2 \(default\)", readme))]
    for what, m in stale:
        rep.check("FAIL" if m else "PASS", what + (" (contradicts the release)" if m else ": not present"))
    for doc, pat, what in (("docs/connectivity.md", r"Road rule v2 \(default\)", "connectivity.md presents v2 as the "
                            "default without the rejection"),
                           ("docs/debris_detection.md", r"(?i)reject", "debris_detection.md records the rejection"),
                           ("docs/architecture.md", r"connectivity, evidence, dashboard, evaluation \| NOT IMPLEMENTED",
                            "architecture.md says connectivity/dashboard/evaluation NOT IMPLEMENTED"),
                           ("docs/limitations.md", r"cannot yet process or analyse any pixels",
                            "limitations.md says BATO cannot analyse pixels"),
                           ("docs/evaluation_protocol.md", r"headline mapping F1 vs EMS",
                            "evaluation_protocol.md plans a 'headline mapping F1 vs EMS'")):
        p = root / doc
        t = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        found = bool(re.search(pat, t))
        good = found if "records the rejection" in what else not found
        rep.check("PASS" if good else "FAIL", what if not good else what + ": OK")


# --------------------------------------------------------------------------------------------------------- I


def audit_i(rep: Report, ctx: dict[str, Any]) -> None:
    rep.section("I", "Claim audit (README, docs, dashboard, reports)")
    root = ctx["root"]
    files = text_files(root, ["README.md", "docs", "dashboard", "reports"])
    pats = {
        "100 % / perfect accuracy": r"(?i)100\s?%\s*(accura|precis|recall|correct)|perfect (accuracy|detection)",
        "flood accuracy / F1 / IoU from EMSR927": r"(?i)(flood|mapping)\s+(accuracy|f1|iou|precision|recall)",
        "accuracy (any)": r"(?i)\baccura(cy|te)\b",
        "ground truth": r"(?i)ground[ -]truth",
        "validated / validation claims": r"(?i)\bvalidated\b",
        "debris detected / detection": r"(?i)debris (detected|detection)",
        "FloodViT accuracy / performance": r"(?i)floodvit[^\n]{0,60}(accura|f1|iou|performance|outperform)",
        "confirmed cut off": r"(?i)confirmed(ly)?\s+cut[\s-]*off",
    }
    table = []
    for label, pat in pats.items():
        for h in grep(files, root, re.compile(pat)):
            status = "OK (negated / qualified)" if NEGATION.search(h["text"]) else "REVIEW"
            table.append({"claim": label, "location": f"{h['file']}:{h['line']}", "status": status, "text": h["text"]})
    rep.fact("claims", table)
    n_rev = sum(1 for t in table if t["status"] == "REVIEW")
    rep.check("REVIEW" if n_rev else "PASS", "claims needing a human read (non-negated matches)", n_rev)


# ---------------------------------------------------------------------------------------------------------


def render(payload: dict[str, Any]) -> str:
    sc = payload["status_counts"]
    L = [f"# BATO release-candidate audit ({payload['generated_at_utc']})", "",
         f"AOI `{payload['aoi_id']}` / candidate `{payload['candidate_id']}`. Status counts: {sc}", ""]
    for key, s in payload["sections"].items():
        L += [f"## {key}. {s['title']}", ""]
        if s["facts"]:
            for k, v in s["facts"].items():
                if k == "claims":
                    continue
                L.append(f"- **{k}**: `{json.dumps(v, default=str)[:900]}`")
            L.append("")
        L += ["| Status | Check | Detail |", "|---|---|---|"]
        for c in s["checks"]:
            d = c["detail"] if isinstance(c["detail"], str) else json.dumps(c["detail"], default=str)
            L.append(f"| {c['status']} | {c['check']} | {d[:400].replace('|', '/')} |")
        if s["facts"].get("claims"):
            L += ["", "| Claim | Location | Status | Text |", "|---|---|---|---|"]
            for t in s["facts"]["claims"]:
                L.append(f"| {t['claim']} | {t['location']} | {t['status']} | {t['text'][:160].replace('|', '/')} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
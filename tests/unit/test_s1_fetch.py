"""Step 2C fetch planning (on REAL Trishuli/Valencia selection metadata) and
execution with mocked CDSE responses. No network, no real imagery."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from bato.acquisition.downloader import Downloader
from bato.acquisition.http import HttpResponse, StreamResponse
from bato.acquisition.s1_fetch import (
    FetchPlanError,
    build_s1_fetch_plan,
    digest,
    execute_fetch,
    save_manifest,
)
from bato.config import load_config
from bato.contracts.catalogue import CatalogueSearchResult
from bato.contracts.fetch import DownloadProvenance, FetchManifest, FetchStatus, S1FetchPlan
from bato.contracts.io import load_contract
from bato.contracts.temporal import TemporalSelectionResult
from bato.data_rules import assert_raw_output_dir, assert_reference_input_allowed
from bato.exceptions import ProhibitedInputError

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "real_catalogues"
NOW = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
DiskUsage = namedtuple("DiskUsage", "total used free")


def real(name: str):
    return (load_contract(TemporalSelectionResult, REAL / f"{name}_sets.json"),
            load_contract(CatalogueSearchResult, REAL / f"{name}_s1.json"))


# ---- planning on real metadata ------------------------------------------------------------

def test_trishuli_rank1_plan(cfg):
    selection, catalogue = real("trishuli")
    plan = build_s1_fetch_plan(selection, catalogue, cfg)
    assert [c.track.label for c in plan.candidates] == ["ASCENDING/85", "DESCENDING/19"]
    assert len(plan.passes) == 6 and len(plan.scenes) == 6 and not any(p.multi_slice for p in plan.passes)
    ids = {r.name: r.product_id for r in catalogue.accepted()}
    for scene in plan.scenes:
        assert scene.product_id == ids[scene.product_name]            # ids come from the catalogue
        assert scene.product_type == "IW_GRDH_1S" and {p.value for p in scene.polarizations} >= {"VV", "VH"}
    assert plan.catalogue_digest == selection.catalogue_digest == digest(catalogue)


def test_valencia_multi_slice_passes_are_fully_planned(cfg):
    selection, catalogue = real("valencia")
    plan = build_s1_fetch_plan(selection, catalogue, cfg)
    assert len(plan.candidates) == 4 and len(plan.passes) == 12 and len(plan.scenes) == 18
    multi = [p for p in plan.passes if p.multi_slice]
    assert len(multi) == 6 and all(len(p.product_ids) == 2 for p in multi)
    for p in multi:
        slices = sorted((s for s in plan.scenes if s.pass_id == p.pass_id), key=lambda s: s.slice_index)
        assert [s.slice_index for s in slices] == [0, 1] and all(s.slice_count == 2 for s in slices)
        assert slices[0].acquisition_start < slices[1].acquisition_start
    # every candidate still refers to exactly three passes (slices are not dates)
    assert all(len(c.pre_event_pass_ids) == 2 for c in plan.candidates)


def test_track_and_rank_selection_and_shared_passes(cfg):
    selection, catalogue = real("trishuli")
    one = build_s1_fetch_plan(selection, catalogue, cfg, tracks=["DESCENDING/19"])
    assert len(one.scenes) == 3 and {s.relative_orbit for s in one.scenes} == {19}
    both = build_s1_fetch_plan(selection, catalogue, cfg, tracks=["ASCENDING/85"], ranks=[1, 2])
    shared = [s for s in both.scenes if len(s.candidate_ids) == 2]
    assert shared and len(both.scenes) == 4            # ranks 1 and 2 share PRE-2 and POST
    with pytest.raises(FetchPlanError, match="unknown track"):
        build_s1_fetch_plan(selection, catalogue, cfg, tracks=["ASCENDING/999"])
    with pytest.raises(FetchPlanError, match="no candidate"):
        build_s1_fetch_plan(selection, catalogue, cfg, ranks=[99])


def test_plan_refuses_a_different_catalogue(cfg):
    selection, _ = real("trishuli")
    _, other = real("valencia")
    with pytest.raises(FetchPlanError, match="digest mismatch"):
        build_s1_fetch_plan(selection, other, cfg)


def test_plan_enforces_kuro_siwo_polarisations(cfg):
    selection, catalogue = real("trishuli")
    hh = cfg.settings.sentinel1.model_copy(update={"polarizations": ("HH", "HV")})
    strict = cfg.model_copy(update={"settings": cfg.settings.model_copy(update={"sentinel1": hh})})
    with pytest.raises(FetchPlanError, match="missing required polarisation"):
        build_s1_fetch_plan(selection, catalogue, strict)


def test_plan_contract_rejects_a_missing_slice(cfg):
    selection, catalogue = real("valencia")
    plan = build_s1_fetch_plan(selection, catalogue, cfg)
    multi = next(p for p in plan.passes if p.multi_slice)
    dropped = [s for s in plan.scenes if not (s.pass_id == multi.pass_id and s.slice_index == 1)]
    with pytest.raises(ValidationError, match="not all slices are planned"):
        S1FetchPlan.model_validate({**plan.model_dump(), "scenes": [s.model_dump() for s in dropped]})


# ---- execution with mocked CDSE ---------------------------------------------------------------

def fake_safe_zip(name: str, pols=("vv", "vh")) -> bytes:
    stem = name.removesuffix(".SAFE")
    mission = stem[:3].lower()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{stem}.SAFE/manifest.safe", "<xfdu/>")
        for pol in pols:
            z.writestr(f"{stem}.SAFE/measurement/{mission}-iw-grd-{pol}-x-001.tiff", b"II*\x00TEST")
            z.writestr(f"{stem}.SAFE/annotation/calibration/calibration-{mission}-iw-grd-{pol}-x-001.xml", "<c/>")
            z.writestr(f"{stem}.SAFE/annotation/calibration/noise-{mission}-iw-grd-{pol}-x-001.xml", "<n/>")
    return buf.getvalue()


class FakeCDSE:
    """Metadata (Transport.get) + download (StreamingTransport.open_stream). TEST FIXTURE content."""

    def __init__(self, products: dict[str, tuple[str, bytes]], online: bool = True, rename: dict | None = None):
        self.products = products          # id -> (name, bytes)
        self.online = online
        self.rename = rename or {}
        self.downloads: list[str] = []
        self.auth_headers: list[str] = []

    def get(self, url, timeout_s):
        pid = url[url.index("(") + 1:url.index(")")]
        name, data = self.products[pid]
        return HttpResponse(200, json.dumps({
            "Id": pid, "Name": self.rename.get(pid, name), "ContentLength": len(data), "Online": self.online,
            "Checksum": [{"Algorithm": "MD5", "Value": hashlib.md5(data).hexdigest(), "ChecksumDate": "x"}],
        }).encode())

    def post_form(self, url, form, timeout_s):  # pragma: no cover
        raise AssertionError

    def open_stream(self, url, headers, timeout_s, chunk_bytes):
        pid = url[url.index("(") + 1:url.index(")")]
        self.downloads.append(pid)
        self.auth_headers.append(headers.get("Authorization", ""))
        data = self.products[pid][1]
        return StreamResponse(200, {}, url.replace("download.", "zipper."), iter([data]), lambda: None)


class Tokens:
    def get(self):
        return "SECRET-TOKEN"

    def invalidate(self):
        pass


@pytest.fixture
def project(config_copy):
    """A temporary project (config + data folders) so downloads land in tmp_path."""
    return load_config(config_copy())


def fake_for(plan, **kwargs) -> FakeCDSE:
    return FakeCDSE({s.product_id: (s.product_name, fake_safe_zip(s.product_name)) for s in plan.scenes}, **kwargs)


def run(plan, cfg, fake, **kwargs):
    downloader = Downloader(cfg.settings.download, fake, Tokens(), sleep=lambda s: None,
                            disk_usage=lambda p: DiskUsage(0, 0, 10**12))
    return execute_fetch(plan, None, cfg, dry_run=False, downloader=downloader, metadata_transport=fake,
                         clock=lambda: NOW, **kwargs)


def valencia_desc8(cfg):
    selection, catalogue = real("valencia")
    return build_s1_fetch_plan(selection, catalogue, cfg, tracks=["DESCENDING/8"])


def test_dry_run_downloads_nothing_and_needs_no_credentials(project):
    plan = valencia_desc8(project)
    manifest = execute_fetch(plan, None, project, dry_run=True, clock=lambda: NOW)
    assert {f.status for f in manifest.files} == {FetchStatus.PLANNED}
    assert not any(manifest.pass_complete.values())
    path = save_manifest(manifest, project)
    assert path.name.endswith("_fetch_plan.json") and load_contract(FetchManifest, path) == manifest


def test_full_fetch_of_multi_slice_candidate_with_provenance(project):
    plan = valencia_desc8(project)
    fake = fake_for(plan)
    manifest = run(plan, project, fake)
    assert len(plan.scenes) == 6 and {f.status for f in manifest.files} == {FetchStatus.DOWNLOADED}
    assert all(manifest.pass_complete.values()) and all(manifest.candidate_ready.values())
    root = project.config_dir.parent
    for f in manifest.files:
        path = root / f.local_path
        assert path.is_file() and path.parent == project.settings.paths.raw / "sentinel1"
        prov = load_contract(DownloadProvenance, root / f.provenance_path)
        scene = next(s for s in plan.scenes if s.product_id == f.product_id)
        assert (prov.product_id, prov.acquisition_start, prov.orbit_direction, prov.relative_orbit) == \
            (scene.product_id, scene.acquisition_start, scene.orbit_direction, scene.relative_orbit)
        assert prov.checksum_verified_with == "md5" and prov.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert prov.source_url.endswith(f"Products({scene.product_id})/$value")
        assert "SECRET" not in (root / f.provenance_path).read_text() and prov.data_role == "pipeline_input_raw"
    assert set(fake.auth_headers) == {"Bearer SECRET-TOKEN"}
    saved = save_manifest(manifest, project)
    assert saved.name.endswith("_fetch_manifest.json") and "SECRET" not in saved.read_text()


def test_max_scenes_leaves_multi_slice_pass_incomplete(project):
    plan = valencia_desc8(project)
    manifest = run(plan, project, fake_for(plan), max_scenes=1)
    first_pass = plan.scenes[0].pass_id
    assert [f.status for f in manifest.files].count(FetchStatus.DOWNLOADED) == 1
    assert manifest.pass_complete[first_pass] is False            # 1 of 2 slices is not a pass
    assert not any(manifest.candidate_ready.values())
    assert manifest.files[1].status is FetchStatus.NOT_ATTEMPTED


def test_rerun_is_idempotent_and_keeps_provenance(project):
    plan = valencia_desc8(project)
    fake = fake_for(plan)
    first = run(plan, project, fake)
    root = project.config_dir.parent
    before = {f.product_id: (root / f.provenance_path).read_text() for f in first.files}
    second = run(plan, project, fake)
    assert {f.status for f in second.files} == {FetchStatus.SKIPPED_EXISTING}
    assert len(fake.downloads) == len(plan.scenes)                      # nothing downloaded twice
    assert {f.product_id: (root / f.provenance_path).read_text() for f in second.files} == before


def test_failures_are_recorded_and_the_run_continues(project):
    plan = valencia_desc8(project)
    fake = fake_for(plan)
    bad = plan.scenes[0]
    fake.products[bad.product_id] = (bad.product_name, fake_safe_zip(bad.product_name, pols=("vv",)))
    manifest = run(plan, project, fake)
    failed = [f for f in manifest.files if f.status is FetchStatus.FAILED]
    assert len(failed) == 1 and "measurement VH raster" in failed[0].message
    quarantined = project.settings.paths.raw / "sentinel1" / f"{bad.product_name}.zip.rejected"
    assert quarantined.exists() and not quarantined.with_suffix("").exists()
    assert sum(f.status is FetchStatus.DOWNLOADED for f in manifest.files) == 5
    assert manifest.pass_complete[bad.pass_id] is False


def test_offline_and_renamed_products_fail_clearly(project):
    plan = valencia_desc8(project)
    offline = run(plan, project, fake_for(plan, online=False))
    assert all("offline" in f.message for f in offline.files)
    first = plan.scenes[0].product_id
    renamed = run(plan, project, fake_for(plan, rename={first: "S1A_SOMETHING_ELSE.SAFE"}))
    assert "re-run the catalogue search" in next(f for f in renamed.files if f.product_id == first).message


# ---- data-rule guards ------------------------------------------------------------------------------

def test_downloads_cannot_go_to_evaluation_folders(project, config_copy):
    def raw_in_reference(d):
        d["paths"]["raw"] = "data/reference/raw"
    bad = load_config(config_copy("config.yaml", raw_in_reference))
    with pytest.raises(ProhibitedInputError):
        assert_raw_output_dir(bad.settings.paths.raw / "sentinel1", bad.settings.paths)
    plan = valencia_desc8(bad)
    with pytest.raises(ProhibitedInputError):
        run(plan, bad, fake_for(plan))
    with pytest.raises(ProhibitedInputError, match="must be inside the raw-data folder"):
        assert_raw_output_dir(project.settings.paths.processed, project.settings.paths)


def test_downloaded_data_cannot_be_loaded_as_reference(project):
    paths = project.settings.paths
    with pytest.raises(ProhibitedInputError, match="not inside an evaluation-only folder"):
        assert_reference_input_allowed(paths.raw / "sentinel1" / "S1A_x.SAFE.zip", paths)
    assert assert_reference_input_allowed(paths.reference / "EMSR927" / "x.shp", paths)

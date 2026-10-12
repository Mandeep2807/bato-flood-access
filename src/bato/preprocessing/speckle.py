"""Step 3.3: Lee Sigma speckle filter on the published Step 3.2 output (SNAP gpt).

    Read (radiometric BEAM-DIMAP) -> Speckle-Filter (Lee Sigma) -> Write (BEAM-DIMAP)

Input: the PUBLISHED Step 3.2 artifact ``<interim>/<output_subdir>/<aoi_id>/<pass_id>/radiometric/``
(only a successful, validated run is ever published there). It is read, hashed before and after the
run, and never modified. Its provenance must match the manifest pass/product and the current
radiometric configuration.

Output: ``<interim>/<output_subdir>/<aoi_id>/<pass_id>/speckle/``

    <pass_id>_sigma0_spk.dim / .data/   Sigma0_VV, Sigma0_VH (Lee Sigma, linear, radar geometry)
    graph_resolved.xml                  the exact graph executed (SHA-256 in provenance)
    gpt.log                             full gpt output
    preprocessing_provenance.json       SpeckleProvenance

Same atomic publishing as Step 3.2: SNAP writes into ``speckle.tmp-<UTC>``; only after gpt
succeeded AND validation passed is it renamed ``speckle``. Failures become
``speckle.failed-<UTC>``; ``overwrite`` keeps the old output as ``speckle.superseded-<UTC>``.

Lee Sigma is the only filter accepted (config validation + graph + SNAP value-set check before
the run); there is no fallback to another filter.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import bato
from bato.config import BatoConfig
from bato.contracts.common import Polarization
from bato.contracts.fetch import FetchManifest
from bato.contracts.io import ContractIOError, load_contract, save_contract
from bato.contracts.preprocessing import (
    GraphRecord,
    OperatorRecord,
    OutputRecord,
    PreprocessingProvenance,
    SnapRecord,
    SpeckleInputRecord,
    SpeckleProvenance,
    SpeckleValidation,
)
from bato.data_rules import assert_interim_output_dir, assert_pipeline_input_allowed
from bato.preprocessing import snap
from bato.preprocessing.dimap import DimapError, DimapProduct, read_dimap
from bato.preprocessing.radiometric import (
    LOG_FILE,
    PROVENANCE_FILE,
    RESOLVED_GRAPH_FILE,
    PreprocessingError,
    _display,
    _relative,
    _sibling,
    _utc_now,
    check_output_path_length,
    stage_dir,
)
from bato.preprocessing.sigma0_validation import FORBIDDEN_OPERATORS, SIGMA0_BAND, require_rasterio
from bato.preprocessing.speckle_validation import SPECKLE_OPERATOR, validate_speckle_output

STAGE = "speckle"
INPUT_STAGE = "radiometric"
LEE_SIGMA_PARAMETERS = ("filter", "numLooksStr", "windowSize", "targetWindowSizeStr", "sigmaStr")


class SpeckleRunFailed(PreprocessingError):
    """gpt or validation failed; ``speckle.failed-<UTC>`` and its provenance are kept."""

    def __init__(self, message: str, output_dir: Path, provenance: SpeckleProvenance) -> None:
        super().__init__(message)
        self.output_dir = output_dir
        self.provenance = provenance


# ---------------------------------------------------------------------------
# Input (the published Step 3.2 artifact)
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def artifact_files(dim: Path) -> list[Path]:
    """The .dim header and every file of its .data folder, in a stable order."""
    data = dim.with_suffix(".data")
    return [dim, *sorted((p for p in data.rglob("*") if p.is_file()), key=lambda p: p.as_posix())]


def hash_artifact(dim: Path, root: Path) -> dict[str, str]:
    return {_relative(p, root): sha256_file(p) for p in artifact_files(dim)}


@dataclass(frozen=True)
class SpeckleJob:
    root: Path
    aoi_id: str
    pass_id: str
    input_dir: Path
    input_dim: Path
    input_provenance_path: Path
    input_provenance: PreprocessingProvenance
    input_product: DimapProduct
    input_hashes: dict[str, str]
    final_dir: Path

    @property
    def output_stem(self) -> str:
        return f"{self.pass_id}_sigma0_spk"


def radiometric_settings(cfg: BatoConfig) -> dict[str, Any]:
    """The settings a Step 3.2 provenance records (same shape), from the current config."""
    s = cfg.settings
    return {"crop_buffer_m": s.preprocessing.crop_buffer_m,
            "s1_radiometric": s.preprocessing.s1_radiometric.model_dump(mode="json"),
            "polarizations": list(s.sentinel1.polarizations)}


def check_input_product(product: DimapProduct, polarizations: tuple[str, ...]) -> None:
    """The input must be what Step 3.2 publishes: exactly linear sigma0 VV/VH in radar geometry,
    not yet speckle filtered, terrain corrected or converted to dB.

    Raises:
        PreprocessingError: anything else (nothing is run).
    """
    expected = {SIGMA0_BAND.format(pol=p) for p in polarizations}
    names = {b.name for b in product.bands}
    problems = []
    if names != expected:
        problems.append(f"bands {sorted(names)} (expected exactly {sorted(expected)})")
    for b in product.bands:
        if "db" in b.name.lower() or (b.unit or "").lower().endswith("db") or b.log10_scaled:
            problems.append(f"{b.name} is in dB / log10 scale (unit {b.unit!r}); the filter needs linear sigma0")
    if product.map_geocoding or not {"latitude", "longitude"} <= set(product.tie_point_grids) \
            or product.abstracted.get("is_terrain_corrected") in ("1", "true"):
        problems.append("not in radar geometry (map geocoding or terrain-corrected flag found)")
    done = [op for op in product.history if op in FORBIDDEN_OPERATORS]
    if done:
        problems.append(f"already processed by {done} (would be filtered twice / out of order)")
    if problems:
        raise PreprocessingError(f"{product.path.name} is not a valid Step 3.2 input: " + "; ".join(problems))


def prepare_speckle_job(manifest: FetchManifest, pass_id: str, cfg: BatoConfig) -> SpeckleJob:
    """Locate and check the published Step 3.2 output for ``pass_id`` (no SNAP, nothing written).

    Raises:
        PreprocessingError: dry-run manifest, unknown pass, no published radiometric output,
            provenance mismatch, settings mismatch, or an input that is not linear sigma0.
        ProhibitedInputError: the input lies in an evaluation-only folder.
    """
    root = cfg.config_dir.parent
    if manifest.dry_run:
        raise PreprocessingError("this is a dry-run plan; use the *_fetch_manifest.json of a real download")
    passes = {p.pass_id: p for p in manifest.plan.passes}
    if pass_id not in passes:
        raise PreprocessingError(f"pass {pass_id} is not in the manifest; known passes: {sorted(passes)}")
    input_dir = stage_dir(cfg, manifest.aoi_id, pass_id, INPUT_STAGE)
    provenance_path = assert_pipeline_input_allowed(input_dir / PROVENANCE_FILE, cfg.settings.paths)
    if not input_dir.is_dir() or not provenance_path.is_file():
        raise PreprocessingError(
            f"no published Step 3.2 output in {_display(input_dir, root)}; run it first: python "
            f"scripts/preprocess_s1.py --manifest <manifest> --pass-id {pass_id} --stage radiometric")
    try:
        provenance = load_contract(PreprocessingProvenance, provenance_path)
    except ContractIOError as exc:
        raise PreprocessingError(str(exc)) from exc
    if provenance.status != "succeeded" or provenance.output is None or provenance.validation is None \
            or not provenance.validation.passed:
        raise PreprocessingError(f"{_display(provenance_path, root)} does not describe a validated output")
    planned = passes[pass_id]
    if (provenance.aoi_id, provenance.pass_id) != (manifest.aoi_id, pass_id) \
            or provenance.input.product_id not in planned.product_ids:
        raise PreprocessingError(f"{_display(provenance_path, root)} belongs to {provenance.aoi_id}/"
                                 f"{provenance.pass_id} ({provenance.input.product_id}), not to this manifest pass")
    current = radiometric_settings(cfg)
    if provenance.settings != current:
        raise PreprocessingError(
            "the Step 3.2 output was produced with different radiometric settings than the current "
            f"configuration (recorded {provenance.settings}, current {current}); re-run Step 3.2 with "
            "--overwrite or restore the configuration")

    dim = assert_pipeline_input_allowed(root / provenance.output.dim, cfg.settings.paths)
    if input_dir not in dim.parents:
        raise PreprocessingError(f"{provenance.output.dim} is outside {_display(input_dir, root)}")
    try:
        product = read_dimap(dim)
    except DimapError as exc:
        raise PreprocessingError(f"published Step 3.2 output is incomplete: {exc}") from exc
    check_input_product(product, tuple(cfg.settings.sentinel1.polarizations))
    return SpeckleJob(root=root, aoi_id=manifest.aoi_id, pass_id=pass_id, input_dir=input_dir, input_dim=dim,
                      input_provenance_path=provenance_path, input_provenance=provenance, input_product=product,
                      input_hashes=hash_artifact(dim, root), final_dir=stage_dir(cfg, manifest.aoi_id, pass_id, STAGE))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def graph_values(job: SpeckleJob, cfg: BatoConfig, output_file: Path) -> dict[str, object]:
    s = cfg.settings.preprocessing.s1_speckle
    return {
        "input_file": job.input_dim.resolve().as_posix(),
        "source_bands": ",".join(SIGMA0_BAND.format(pol=p) for p in cfg.settings.sentinel1.polarizations),
        "filter": s.filter,
        "num_looks": s.num_looks,
        "window_size": s.window_size,
        "target_window_size": s.target_window_size,
        "sigma": s.sigma,
        "output_file": output_file.resolve().as_posix(),
    }


def lee_sigma_parameters(graph_xml: str) -> dict[str, str]:
    """All Speckle-Filter parameters exactly as written in the rendered graph."""
    for _node, operator, params in snap.graph_operators(graph_xml):
        if operator == SPECKLE_OPERATOR:
            return dict(params)
    raise snap.SnapError("the graph has no Speckle-Filter node")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeckleResult:
    output_dir: Path
    provenance_path: Path
    provenance: SpeckleProvenance


def run_speckle(job: SpeckleJob, cfg: BatoConfig, *, overwrite: bool = False, gpt: snap.ResolvedGpt | None = None,
                runner: snap.Runner = subprocess.run, popen: Callable[..., Any] = subprocess.Popen,
                now: Callable[[], datetime] = _utc_now, progress: Callable[[str], None] | None = None,
                environ: dict[str, str] | None = None) -> SpeckleResult:
    """Preflight, run SNAP, validate against the input, then publish atomically.

    Raises:
        PreprocessingError: output exists without ``overwrite`` or path too long (nothing written).
        SpeckleRunFailed: gpt or validation failed, or the input changed (failed folder kept).
        snap.SnapError: SNAP missing, or an operator / parameter / value not accepted (nothing written).
        ValidationSetupError: rasterio missing (nothing written).
    """
    settings = cfg.settings
    root = job.root
    say = progress or (lambda _text: None)
    final_dir = assert_interim_output_dir(job.final_dir, settings.paths)
    if final_dir.exists() and not overwrite:
        raise PreprocessingError(f"output already exists: {final_dir} (use --overwrite to supersede it; "
                                 "the old output is kept as speckle.superseded-<time>)")

    # ---- preflight (nothing written yet) -------------------------------------------------
    require_rasterio()
    resolved = gpt or snap.resolve_gpt(settings.snap, environ=environ)
    say(f"SNAP gpt: {resolved.path} (from {resolved.source})\n")
    diagnostics = snap.snap_diagnostics(resolved.path, settings.snap.diag_timeout_s, runner)
    say(f"SNAP version: {diagnostics.version or 'unknown (not found in gpt --diag output)'}\n")
    template_path = snap.graph_path(settings.preprocessing.s1_speckle.graph)
    template_text = template_path.read_text(encoding="utf-8")
    started = now()
    tmp_dir = assert_interim_output_dir(_sibling(final_dir, "tmp", started), settings.paths)
    check_output_path_length(tmp_dir, job.output_stem)
    output_file = tmp_dir / f"{job.output_stem}.dim"
    graph_xml = snap.render_graph(template_text, graph_values(job, cfg, output_file))
    lee_sigma = lee_sigma_parameters(graph_xml)
    say("checking operator parameters and Lee Sigma values against the installed SNAP ...\n")
    snap.check_operator_parameters(resolved.path, graph_xml, settings.snap.diag_timeout_s, runner)
    value_sets = snap.check_parameter_values(resolved.path, graph_xml, [SPECKLE_OPERATOR],
                                             settings.snap.diag_timeout_s, runner)
    unverified = [p for p in LEE_SIGMA_PARAMETERS if p in value_sets[SPECKLE_OPERATOR]["not_listed"]]
    warnings = [f"`gpt Speckle-Filter -h` lists no value set for {unverified}; values passed as configured"] \
        if unverified else []

    # ---- run -----------------------------------------------------------------------------
    tmp_dir.mkdir(parents=True)
    graph_file = tmp_dir / RESOLVED_GRAPH_FILE
    graph_file.write_text(graph_xml, encoding="utf-8")
    log_path = tmp_dir / LOG_FILE
    run: snap.GptRun | None = None
    validation: SpeckleValidation | None = None
    error: str | None = None
    say(f"running gpt (log: {_display(log_path, root)}) ...\n")
    try:
        run = snap.run_gpt(resolved.path, graph_file, log_path, settings.snap.timeout_s, popen=popen,
                           progress=progress)
        say(f"\ngpt finished in {run.runtime_s} s; validating output against the input ...\n")
        validation = validate_speckle_output(
            output_file, input_product=job.input_product,
            polarizations=[Polarization(p) for p in settings.sentinel1.polarizations],
            border_limit=settings.preprocessing.s1_radiometric.border_limit, root=root)
        if not validation.passed:
            error = "validation failed: " + "; ".join(f"{c.name}: {c.detail}" for c in validation.failed_checks())
    except snap.SnapError as exc:
        error = str(exc)
    after = hash_artifact(job.input_dim, root) if job.input_dim.is_file() else {}
    if after != job.input_hashes:
        changed = sorted(k for k in set(after) | set(job.input_hashes) if after.get(k) != job.input_hashes.get(k))
        error = (error + "; " if error else "") + f"the Step 3.2 input changed during the run: {changed}"

    # ---- publish ---------------------------------------------------------------------------
    finished = now()
    if error is None:
        target = final_dir
        if final_dir.exists():
            os.replace(final_dir, _sibling(final_dir, "superseded", finished))
    else:
        target = _sibling(final_dir, "failed", finished)
    output_hashes = {_relative(target / p.relative_to(tmp_dir), root): sha256_file(p)
                     for p in artifact_files(output_file)} if error is None else {}
    provenance = _provenance(job, cfg, error=error, started=started, finished=finished, resolved=resolved,
                             diagnostics=diagnostics, template_path=template_path, template_text=template_text,
                             graph_xml=graph_xml, lee_sigma=lee_sigma, value_sets=value_sets, run=run,
                             validation=validation, tmp_dir=tmp_dir, target=target, output_hashes=output_hashes,
                             warnings=warnings)
    save_contract(provenance, tmp_dir / PROVENANCE_FILE)
    os.replace(tmp_dir, target)
    provenance_path = target / PROVENANCE_FILE
    if error is not None:
        raise SpeckleRunFailed(f"{error}\nfailed output kept (NOT valid) in {target}\nprovenance: {provenance_path}",
                               target, provenance)
    return SpeckleResult(output_dir=target, provenance_path=provenance_path, provenance=provenance)


def _provenance(job: SpeckleJob, cfg: BatoConfig, *, error: str | None, started: datetime, finished: datetime,
                resolved: snap.ResolvedGpt, diagnostics: snap.SnapDiagnostics, template_path: Path,
                template_text: str, graph_xml: str, lee_sigma: dict[str, str], value_sets: dict[str, Any],
                run: snap.GptRun | None, validation: SpeckleValidation | None, tmp_dir: Path, target: Path,
                output_hashes: dict[str, str], warnings: list[str]) -> SpeckleProvenance:
    root = job.root
    settings = cfg.settings
    rp = job.input_provenance

    def final(path: Path) -> str:
        return _relative(target / path.relative_to(tmp_dir), root)

    def rebase_file(rel: str) -> str:
        path = (root / rel).resolve()
        return final(path) if tmp_dir.resolve() in path.parents else rel

    validation_out = validation
    if validation is not None:   # output band paths were recorded inside the temp folder
        validation_out = validation.model_copy(update={"bands_after": tuple(
            b.model_copy(update={"file": rebase_file(b.file)}) for b in validation.bands_after)})
    output = None
    if error is None:
        data = tmp_dir / f"{job.output_stem}.data"
        output = OutputRecord(
            directory=final(tmp_dir), dim=final(tmp_dir / f"{job.output_stem}.dim"), data_dir=final(data),
            band_files={SIGMA0_BAND.format(pol=p): final(data / f"{SIGMA0_BAND.format(pol=p)}.img")
                        for p in settings.sentinel1.polarizations})
    log_path = tmp_dir / LOG_FILE
    assert rp.output is not None   # checked in prepare_speckle_job
    return SpeckleProvenance(
        status="succeeded" if error is None else "failed", error=error,
        bato_version=bato.__version__, config_fingerprint=cfg.fingerprint(),
        settings={**settings.preprocessing.s1_speckle.model_dump(mode="json"),
                  "polarizations": list(settings.sentinel1.polarizations)},
        lee_sigma=lee_sigma, snap_value_sets=value_sets,
        started_at=started, finished_at=finished, aoi_id=job.aoi_id, pass_id=job.pass_id,
        input=SpeckleInputRecord(
            directory=_relative(job.input_dir, root), dim=_relative(job.input_dim, root),
            provenance_path=_relative(job.input_provenance_path, root),
            provenance_sha256=sha256_file(job.input_provenance_path), files_sha256=job.input_hashes,
            product_id=rp.input.product_id, product_name=rp.input.product_name,
            acquisition_start=rp.input.acquisition_start, orbit_direction=rp.input.orbit_direction,
            relative_orbit=rp.input.relative_orbit, radiometric_config_fingerprint=rp.config_fingerprint,
            radiometric_graph_sha256=rp.graph.resolved_sha256, radiometric_settings=rp.settings,
            width=job.input_product.width, height=job.input_product.height),
        graph=GraphRecord(
            template_path=_display(template_path, root), template_sha256=snap.sha256_text(template_text),
            resolved_path=final(tmp_dir / RESOLVED_GRAPH_FILE), resolved_sha256=snap.sha256_text(graph_xml),
            operators=tuple(OperatorRecord(node=n, operator=o, parameters=p)
                            for n, o, p in snap.graph_operators(graph_xml)),
            operator_parameters_checked=True),
        snap=SnapRecord(
            gpt_path=str(resolved.path), gpt_source=resolved.source, snap_version=diagnostics.version,
            diagnostics=diagnostics.output_head,
            command=run.args if run else (str(resolved.path), str(tmp_dir / RESOLVED_GRAPH_FILE), "-e"),
            return_code=run.returncode if run else None, runtime_s=run.runtime_s if run else None,
            log_path=final(log_path) if log_path.exists() else None),
        output=output, output_sha256=output_hashes, validation=validation_out, warnings=tuple(warnings),
    )


def describe_speckle_validation(validation: SpeckleValidation) -> list[str]:
    """Human-readable summary (used by the CLI): checks, then before -> after per band."""
    lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}" for c in validation.checks]
    after = {b.band: b for b in validation.bands_after}
    comps = {c.band: c for c in validation.comparisons}

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.6g}"

    for before in validation.bands_before:
        a = after.get(before.band)
        if a is None:
            continue
        lines.append(f"{before.band} (linear sigma0)            before        after")
        for label, x, y in (("valid pixels", before.valid_pixels, a.valid_pixels),
                            ("no-data", before.nodata_pixels, a.nodata_pixels),
                            ("zero", before.zero_pixels, a.zero_pixels),
                            ("non-finite", before.nonfinite_pixels, a.nonfinite_pixels)):
            lines.append(f"    {label:24} {x:>12,} {y:>12,}")
        for label, x2, y2 in (("min", before.min, a.min), ("p01", before.quantiles.get("p01"), a.quantiles.get("p01")),
                              ("p05", before.quantiles.get("p05"), a.quantiles.get("p05")),
                              ("p25", before.quantiles.get("p25"), a.quantiles.get("p25")),
                              ("median", before.median, a.median),
                              ("p75", before.quantiles.get("p75"), a.quantiles.get("p75")),
                              ("p95", before.quantiles.get("p95"), a.quantiles.get("p95")),
                              ("p99", before.quantiles.get("p99"), a.quantiles.get("p99")),
                              ("max", before.max, a.max), ("mean", before.mean, a.mean)):
            lines.append(f"    {label:24} {fmt(x2):>12} {fmt(y2):>12}")
        c = comps.get(before.band)
        if c is not None:
            lines.append(f"    changed {c.changed_pixels:,}/{c.compared_pixels:,} | coeff. of variation "
                         f"{fmt(c.cv_before)} -> {fmt(c.cv_after)} | mean |diff| {fmt(c.mean_abs_difference)} | "
                         f"new no-data {c.new_nodata_pixels:,} | restored {c.restored_pixels:,} | "
                         f"new negative {c.new_negative_pixels:,}")
    lines.extend(f"WARN  {w}" for w in validation.warnings)
    return lines

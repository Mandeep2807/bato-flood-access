"""ESA SNAP Graph Processing Tool (gpt) integration.

gpt is an external program: BATO never uses SNAP's Python bindings, never runs
it through a shell, and never changes SNAP's own memory/cache configuration.

* :func:`resolve_gpt` - explicit config -> ``BATO_SNAP_GPT`` -> platform default -> PATH.
* :func:`snap_diagnostics` - ``gpt --diag`` (version, Java, memory) for provenance.
* :func:`render_graph` - fill a version-controlled graph template; every
  placeholder must be filled, values are XML-escaped, the result must parse.
* :func:`check_operator_parameters` - before a long run, confirm every
  parameter name used in the graph exists in the installed SNAP (``gpt <Op> -h``).
* :func:`check_parameter_values` - values of SNAP value-set parameters (e.g. Speckle-Filter
  windowSize) must be accepted by the installed SNAP.
* :func:`run_gpt` - run a graph with a timeout; full output goes to a log file
  (and to an optional progress callback while it runs).

All subprocess calls go through an injectable ``runner``/``popen`` so tests need no SNAP.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any
from xml.sax.saxutils import escape

from bato.config import SnapSettings
from bato.exceptions import BatoError

ENV_GPT = "BATO_SNAP_GPT"
GRAPH_DIR = Path(__file__).resolve().parent / "graphs"
_PLACEHOLDER = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_VERSION = re.compile(r"SNAP[^\n]*?(?:version|release)[^\d\n]*(\d+(?:\.\d+)+)", re.IGNORECASE)
_LOG_TAIL_LINES = 40

Runner = Callable[..., subprocess.CompletedProcess[Any]]


class SnapError(BatoError):
    """SNAP gpt is missing, unusable, or a gpt run failed."""


# ---------------------------------------------------------------------------
# Executable and diagnostics
# ---------------------------------------------------------------------------


def default_gpt_candidates(system: str | None = None, home: Path | None = None) -> list[Path]:
    system = system or platform.system()
    home = home or Path.home()
    if system == "Windows":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        return [program_files / "esa-snap" / "bin" / "gpt.exe", home / "esa-snap" / "bin" / "gpt.exe"]
    return [home / "esa-snap" / "bin" / "gpt", Path("/opt/esa-snap/bin/gpt"), Path("/usr/local/esa-snap/bin/gpt")]


@dataclass(frozen=True)
class ResolvedGpt:
    path: Path
    source: str          # "config" | "env:BATO_SNAP_GPT" | "default" | "PATH"


def resolve_gpt(settings: SnapSettings, environ: Mapping[str, str] | None = None,
                candidates: Sequence[Path] | None = None,
                which: Callable[[str], str | None] = shutil.which) -> ResolvedGpt:
    """Find the gpt executable. An explicit config or env path must exist (no silent fallback).

    Raises:
        SnapError: no usable gpt found.
    """
    env = os.environ if environ is None else environ
    if settings.gpt_path is not None:
        if not Path(settings.gpt_path).is_file():
            raise SnapError(f"snap.gpt_path is set to {settings.gpt_path}, but no file exists there")
        return ResolvedGpt(Path(settings.gpt_path), "config")
    if env.get(ENV_GPT):
        path = Path(env[ENV_GPT])
        if not path.is_file():
            raise SnapError(f"{ENV_GPT} is set to {path}, but no file exists there")
        return ResolvedGpt(path, f"env:{ENV_GPT}")
    for candidate in (default_gpt_candidates() if candidates is None else candidates):
        if candidate.is_file():
            return ResolvedGpt(candidate, "default")
    found = which("gpt")
    if found:
        return ResolvedGpt(Path(found), "PATH")
    raise SnapError("SNAP gpt not found: set snap.gpt_path in config/config.yaml or the "
                    f"{ENV_GPT} environment variable to the full path of gpt(.exe)")


@dataclass(frozen=True)
class SnapDiagnostics:
    version: str | None
    output_head: tuple[str, ...]     # first lines of `gpt --diag` (no secrets in it)


def snap_diagnostics(gpt: Path, timeout_s: float, runner: Runner = subprocess.run) -> SnapDiagnostics:
    """Run ``gpt --diag``.

    Raises:
        SnapError: gpt cannot be started, times out, or fails.
    """
    result = _run_short([str(gpt), "--diag"], timeout_s, runner, "gpt --diag")
    text = (result.stdout or "") + (result.stderr or "")
    match = _VERSION.search(text)
    lines = tuple(line.rstrip() for line in text.splitlines() if line.strip())[:25]
    return SnapDiagnostics(version=match.group(1) if match else None, output_head=lines)


def _run_short(args: list[str], timeout_s: float, runner: Runner, label: str) -> subprocess.CompletedProcess[Any]:
    try:
        result = runner(args, capture_output=True, text=True, timeout=timeout_s, check=False, shell=False)
    except FileNotFoundError as exc:
        raise SnapError(f"{label}: cannot start {args[0]} ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise SnapError(f"{label}: no answer within {timeout_s:g} s") from exc
    if result.returncode != 0:
        tail = "\n".join(((result.stdout or "") + (result.stderr or "")).splitlines()[-10:])
        raise SnapError(f"{label} failed with exit code {result.returncode}:\n{tail}")
    return result


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def graph_path(name: str) -> Path:
    path = GRAPH_DIR / name
    if not path.is_file():
        raise SnapError(f"graph template {name} not found in {GRAPH_DIR}")
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _xml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return escape(str(value))


def render_graph(template_text: str, values: Mapping[str, object]) -> str:
    """Fill ``${name}`` placeholders. Every placeholder must have a value, every
    value must be used, and the result must be well-formed XML.

    Raises:
        SnapError: missing/unused values or invalid XML.
    """
    body = re.sub(r"<!--.*?-->", "", template_text, flags=re.DOTALL)  # placeholders in comments don't count
    needed = {m[2:-1] for m in _PLACEHOLDER.findall(body)}
    missing, unused = needed - set(values), set(values) - needed
    if missing or unused:
        raise SnapError(f"graph placeholders mismatch: missing {sorted(missing)}, unused {sorted(unused)}")
    rendered = Template(body).substitute({k: _xml_value(v) for k, v in values.items()})
    try:
        ET.fromstring(rendered)
    except ET.ParseError as exc:
        raise SnapError(f"rendered graph is not valid XML: {exc}") from exc
    return rendered


def graph_operators(graph_xml: str) -> list[tuple[str, str, dict[str, str]]]:
    """[(node id, operator, {parameter: value})] in node order."""
    root = ET.fromstring(graph_xml)
    nodes = []
    for node in root.findall("node"):
        operator = (node.findtext("operator") or "").strip()
        params = node.find("parameters")
        values = {child.tag: (child.text or "").strip() for child in (params if params is not None else [])}
        nodes.append((node.get("id", operator), operator, values))
    return nodes


def check_operator_parameters(gpt: Path, graph_xml: str, timeout_s: float,
                              runner: Runner = subprocess.run) -> dict[str, list[str]]:
    """Confirm each parameter used in the graph is accepted by the installed SNAP.

    ``gpt <Operator> -h`` lists parameters as ``-P<name>=``. The exit code of
    a help request is not relied on (it differs between SNAP versions); the
    help text itself must list the parameters. Returns {operator: [checked parameters]}.

    Raises:
        SnapError: gpt cannot run, an operator is unknown, or a parameter name is not supported.
    """
    checked: dict[str, list[str]] = {}
    problems = []
    for _node, operator, params in graph_operators(graph_xml):
        if operator in checked:
            continue
        args = [str(gpt), operator, "-h"]
        try:
            result = runner(args, capture_output=True, text=True, timeout=timeout_s, check=False, shell=False)
        except FileNotFoundError as exc:
            raise SnapError(f"gpt {operator} -h: cannot start {gpt} ({exc})") from exc
        except subprocess.TimeoutExpired as exc:
            raise SnapError(f"gpt {operator} -h: no answer within {timeout_s:g} s") from exc
        text = (result.stdout or "") + (result.stderr or "")
        if "-P" not in text:
            tail = "\n".join(text.splitlines()[-5:])
            problems.append(f"{operator}: unknown to this SNAP installation (no parameter list from "
                            f"`gpt {operator} -h`, exit code {result.returncode}): {tail}")
        else:
            unknown = [p for p in params if f"-P{p}=" not in text]
            if unknown:
                problems.append(f"{operator}: parameter(s) {unknown} not listed by `gpt {operator} -h`")
        checked[operator] = sorted(params)
    if problems:
        raise SnapError("installed SNAP does not accept the graph as written:\n  " + "\n  ".join(problems))
    return checked


_VALUE_SET = re.compile(r"Value must be one of((?:\s*'[^']*'\s*,?)+)")


def parameter_value_set(help_text: str, parameter: str) -> tuple[str, ...] | None:
    """The allowed values SNAP lists for ``-P<parameter>=`` in ``gpt <Op> -h``
    ("Value must be one of 'a', 'b'."), or None when the help lists no value set."""
    start = help_text.find(f"-P{parameter}=")
    if start < 0:
        return None
    following = help_text.find("-P", start + 2)
    block = " ".join(help_text[start:following if following > 0 else None].split())
    match = _VALUE_SET.search(block)
    return tuple(re.findall(r"'([^']*)'", match.group(1))) if match else None


def check_parameter_values(gpt: Path, graph_xml: str, operators: Sequence[str], timeout_s: float,
                           runner: Runner = subprocess.run) -> dict[str, dict[str, Any]]:
    """For the given operators, every graph value of a parameter that has a SNAP value set must
    be in that set (e.g. Speckle-Filter windowSize '7x7'). Parameters without a listed value set
    are reported as not checkable, never assumed valid.

    Returns {operator: {"checked": {param: [allowed values]}, "not_listed": [params]}}.

    Raises:
        SnapError: gpt cannot run, or a value is not accepted by the installed SNAP.
    """
    report: dict[str, dict[str, Any]] = {}
    problems = []
    for _node, operator, params in graph_operators(graph_xml):
        if operator not in operators or operator in report:
            continue
        result = _help(gpt, operator, timeout_s, runner)
        checked: dict[str, list[str]] = {}
        not_listed: list[str] = []
        for name, value in params.items():
            allowed = parameter_value_set(result, name)
            if allowed is None:
                not_listed.append(name)
                continue
            checked[name] = list(allowed)
            if value not in allowed:
                problems.append(f"{operator}: {name}={value!r} is not accepted by this SNAP (allowed: {list(allowed)})")
        report[operator] = {"checked": checked, "not_listed": sorted(not_listed)}
    if problems:
        raise SnapError("installed SNAP does not accept the graph values:\n  " + "\n  ".join(problems))
    return report


def _help(gpt: Path, operator: str, timeout_s: float, runner: Runner) -> str:
    try:
        result = runner([str(gpt), operator, "-h"], capture_output=True, text=True, timeout=timeout_s,
                        check=False, shell=False)
    except FileNotFoundError as exc:
        raise SnapError(f"gpt {operator} -h: cannot start {gpt} ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise SnapError(f"gpt {operator} -h: no answer within {timeout_s:g} s") from exc
    text = (result.stdout or "") + (result.stderr or "")
    if "-P" not in text:
        raise SnapError(f"{operator}: unknown to this SNAP installation (no parameter list from `gpt {operator} -h`)")
    return text


# ---------------------------------------------------------------------------
# Running a graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GptRun:
    args: tuple[str, ...]
    returncode: int
    runtime_s: float
    log_path: Path


def log_tail(path: Path, lines: int = _LOG_TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log not readable)"


def run_gpt(gpt: Path, graph_file: Path, log_path: Path, timeout_s: float,
            popen: Callable[..., Any] = subprocess.Popen, clock: Callable[[], float] = time.monotonic,
            progress: Callable[[str], None] | None = None) -> GptRun:
    """Run ``gpt <graph> -e`` (no shell). stdout+stderr are copied to ``log_path``
    and, if given, to ``progress`` as they arrive (gpt prints ``....10%....``).

    Raises:
        SnapError: cannot start, timeout (the process is killed), or non-zero exit;
        the message includes the end of the log.
    """
    args = (str(gpt), str(graph_file), "-e")
    started = clock()
    with log_path.open("wb") as log:
        log.write(("$ " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n").encode("utf-8"))
        log.flush()
        try:
            process = popen(list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        except OSError as exc:
            raise SnapError(f"cannot start gpt at {gpt} ({exc})") from exc

        def copy_output() -> None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = process.stdout.read1(4096) if hasattr(process.stdout, "read1") else process.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                if progress is not None:
                    progress(decoder.decode(chunk))

        reader = threading.Thread(target=copy_output, name="gpt-output", daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            reader.join(timeout=10)
            raise SnapError(f"gpt did not finish within {timeout_s:g} s (snap.timeout_s) and was stopped; "
                            f"log: {log_path}\n{log_tail(log_path)}") from exc
        reader.join(timeout=30)
    runtime = round(clock() - started, 1)
    if returncode != 0:
        raise SnapError(f"gpt failed with exit code {returncode} after {runtime} s; log: {log_path}\n"
                        f"{log_tail(log_path)}")
    return GptRun(args=args, returncode=returncode, runtime_s=runtime, log_path=log_path)

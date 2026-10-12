"""Logging setup and per-stage reporting.

Every pipeline stage runs inside :func:`stage`, which logs when it starts,
its inputs, its outputs, its duration, any warnings, any fallback used, and
failures. Failures are logged with a traceback and then **re-raised**; this
module never swallows an exception.

Usage::

    with stage("s1_pairing", aoi_id=aoi.aoi_id) as report:
        pair = select_pair(...)
        report.record_output(pair_id=pair.pair_id)
        if pair_is_weak:
            report.warn("only one pre-event image")
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT_LOGGER_NAME = "bato"
_HANDLER_MARK = "_bato_handler"
_MAX_VALUE_CHARS = 200


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 3] + "..."


class KeyValueFormatter(logging.Formatter):
    """Human-readable console format with structured fields appended."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        fields = getattr(record, "bato", None)
        if fields:
            text += " | " + " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        return text


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line, for machine-readable run logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "bato", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", jsonl_path: Path | None = None) -> logging.Logger:
    """Attach console (and optionally JSON-lines file) handlers to the ``bato`` logger.

    Calling it again replaces BATO's handlers instead of duplicating them.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setFormatter(KeyValueFormatter())
    setattr(console, _HANDLER_MARK, True)
    logger.addHandler(console)

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        setattr(file_handler, _HANDLER_MARK, True)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``bato`` namespace."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


@dataclass
class StageReport:
    """What happened in one pipeline stage. Returned to the caller so it can
    later be written into a run manifest."""

    name: str
    inputs: dict[str, Any]
    logger: logging.Logger = field(repr=False)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fallback: str | None = None
    status: Literal["running", "succeeded", "failed"] = "running"
    duration_s: float | None = None
    error: str | None = None

    def record_output(self, **values: Any) -> None:
        self.outputs.update(values)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.logger.warning("stage warning: %s", self.name,
                            extra={"bato": {"stage": self.name, "event": "warning", "detail": message}})

    def use_fallback(self, reason: str) -> None:
        """Declare that a documented fallback replaced the primary method."""
        self.fallback = reason
        self.logger.warning("stage fallback: %s", self.name,
                            extra={"bato": {"stage": self.name, "event": "fallback", "reason": reason}})


@contextmanager
def stage(name: str, logger: logging.Logger | None = None, **inputs: Any) -> Iterator[StageReport]:
    """Run a block as a named, timed, logged pipeline stage.

    Exceptions are logged with their traceback and re-raised unchanged.
    """
    log = logger or get_logger("pipeline")
    report = StageReport(name=name, inputs=dict(inputs), logger=log)
    log.info("stage started: %s", name, extra={"bato": {"stage": name, "event": "started", **inputs}})
    started = time.perf_counter()
    try:
        yield report
    except Exception as exc:
        report.status = "failed"
        report.duration_s = round(time.perf_counter() - started, 3)
        report.error = f"{type(exc).__name__}: {exc}"
        log.error("stage failed: %s", name, exc_info=True,
                  extra={"bato": {"stage": name, "event": "failed", "duration_s": report.duration_s,
                                  "error": report.error}})
        raise
    report.status = "succeeded"
    report.duration_s = round(time.perf_counter() - started, 3)
    log.info("stage finished: %s", name,
             extra={"bato": {"stage": name, "event": "finished", "duration_s": report.duration_s,
                             "outputs": report.outputs, "warnings": len(report.warnings),
                             "fallback": report.fallback}})

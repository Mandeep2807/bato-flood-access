"""JSON (de)serialisation for contracts.

All contracts are pydantic models, so this is thin: it fixes the encoding,
the indentation, and validates on load so a corrupted or hand-edited file
fails loudly instead of flowing into the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from bato.contracts.common import Contract
from bato.exceptions import BatoError

T = TypeVar("T", bound=Contract)


class ContractIOError(BatoError):
    """A contract file could not be read or failed validation."""


def save_contract(model: Contract, path: Path) -> Path:
    """Write a contract as UTF-8 JSON (creating parent folders)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_contract(model_type: type[T], path: Path) -> T:
    """Read and validate a contract written by :func:`save_contract`."""
    if not path.is_file():
        raise ContractIOError(f"contract file not found: {path}")
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ContractIOError(f"{path} is not a valid {model_type.__name__}:\n{exc}") from exc

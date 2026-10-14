"""Deterministic checks that keep model output subordinate to BATO evidence.

* ``numbers_in``: every number written in a text, normalised (Devanagari digits -> ASCII, thousands separators
  removed, trailing zeros dropped), so ``2,942.161``, ``२,९४२.१६१`` and ``2942.1610`` compare equal.
* ``allowed_numbers``: the numbers a set of evidence items supports - the numbers in their text and values, plus
  the roundings of their decimal values that a reader would recognise (1-2 decimals; whole numbers only for values
  >= 100, so a small fraction can never "support" an arbitrary integer).
* ``ungrounded_numbers``: numbers in an answer that no cited evidence item supports. A non-empty result means the
  answer is not shown.

No model is involved here; the checks are plain string and number comparisons.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = re.compile(r"\d+(?:,\d{2,3})*(?:\.\d+)?")
_CITATION = re.compile(r"\[[^\]\n]{1,120}\]")
_LIST_MARKER = re.compile(r"^\s*(?:#{1,6}\s*)?\d{1,2}[.)]\s+", re.MULTILINE)


def canonical(token: str) -> str:
    """'2,942.1610' -> '2942.161', '08' -> '8', '0.0' -> '0'."""
    t = token.translate(DEVANAGARI_DIGITS).replace(",", "")
    try:
        d = Decimal(t)
    except InvalidOperation:
        return t
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def numbers_in(text: str) -> set[str]:
    t = text.translate(DEVANAGARI_DIGITS)
    return {canonical(m.group(0)) for m in _NUMBER.finditer(t)}


def answer_numbers(text: str) -> set[str]:
    """Numbers in a model answer, ignoring citation brackets and Markdown list / heading numbering."""
    t = _CITATION.sub(" ", text.translate(DEVANAGARI_DIGITS))
    t = _LIST_MARKER.sub(" ", t)
    return numbers_in(t)


def _value_numbers(v: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, int):
        out.add(canonical(str(v)))
    elif isinstance(v, float):
        out.add(canonical(repr(v)))
        for nd in (1, 2, 3):
            out.add(canonical(f"{v:.{nd}f}"))
        if abs(v) >= 100:
            out.add(canonical(f"{v:.0f}"))
    elif isinstance(v, str):
        out |= numbers_in(v)
    elif isinstance(v, Mapping):
        for x in v.values():
            out |= _value_numbers(x)
    elif isinstance(v, Iterable):
        for x in v:
            out |= _value_numbers(x)
    return out


def allowed_numbers(items: Iterable[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for it in items:
        out |= numbers_in(str(it.get("text", ""))) | numbers_in(str(it.get("title", "")))
        out |= _value_numbers(it.get("values", {}))
    return out


def ungrounded_numbers(answer: str, cited: Iterable[Mapping[str, Any]], question: str = "") -> list[str]:
    """Numbers in ``answer`` that the cited evidence (or the user's own question) does not contain."""
    allowed = allowed_numbers(cited) | numbers_in(question)
    return sorted(answer_numbers(answer) - allowed, key=lambda s: (len(s), s))

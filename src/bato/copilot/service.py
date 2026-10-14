"""SITREP copilot: an LLM answers questions and translates the SITREP, strictly subordinate to BATO evidence.

Questions (``ask``):
  1. the evidence pack of the official v1 run is retrieved (bato.copilot.evidence);
  2. the model gets ONLY those items and must return JSON {"supported", "answer", "citations"};
  3. deterministic checks: every citation must name a retrieved item, a supported answer must cite at least one,
     every number in the answer must occur in the cited items (or the question), and a Nepali answer must be in
     Devanagari. One corrective retry is allowed; if it still fails, the model text is WITHHELD and only the
     verified evidence is returned.

SITREPs (``sitrep``):
  * English: the deterministic SITREP of bato.dashboard.sitrep (labelled "deterministic template, no AI").
  * Nepali: the model translates that verified English SITREP. Accepted only if the set of numbers is identical to
    the English source and the three required attribution statements are still present verbatim; otherwise
    withheld (one retry), and the English SITREP remains the authoritative text.

Nothing here changes a BATO result; the model never sees anything outside the evidence pack.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from bato.attribution import REQUIRED_ATTRIBUTIONS
from bato.config import BatoConfig
from bato.copilot.evidence import EvidenceError, EvidencePack, build_pack
from bato.copilot.grounding import numbers_in, ungrounded_numbers
from bato.copilot.provider import ChatModel, CopilotUnavailable, ProviderError, config_from_env, model_from_env
from bato.dashboard.sitrep import SitrepError, render

LANGS = {"en": "English", "ne": "Nepali (नेपाली, Devanagari script)"}
MAX_QUESTION = 1000
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

SYSTEM = """You are the BATO SITREP Copilot for disaster responders. BATO maps open-water flood evidence on roads from \
Sentinel-1 radar and derives settlement road access. Rules, in order of priority:
1. Use ONLY the EVIDENCE items provided. Do not use outside knowledge about the place, the event or the world.
2. Every number you write must appear in an evidence item you cite. Do not compute new numbers; only use the \
calculations given in calc:* items. Never estimate population, casualties, damage, costs, forecasts or current \
conditions.
3. Cite the evidence ids in square brackets right after the sentence they support, e.g. [roads] or [stretch:blk-0].
4. Wording that must stay exact in meaning: "access unverified" is NOT "cut off"; "unobservable" means unknown, NOT \
open or safe; "observed open" means no flood signal was detected, NOT that the road is passable; the result is not \
validated and describes the satellite pass date, not current conditions.
5. If the evidence does not answer the question, set "supported" to false and say briefly what BATO does not \
contain. Do not guess.
6. Write the answer in {language}. Keep identifiers (e.g. blk-0, aoi-...) and units unchanged.
Return ONLY a JSON object: {{"supported": true|false, "answer": "<text>", "citations": ["<evidence id>", ...]}}"""

TRANSLATE = """Translate this BATO situation report from English into Nepali (Devanagari script) for Nepali-speaking \
responders. Strict rules:
1. Translate faithfully. Do not add, remove, round, convert or estimate any number, date, time, identifier or unit; \
copy every number exactly as written (Western or Devanagari digits are both acceptable).
2. Keep the Markdown structure (headings, table, lists) and every row of the table.
3. Keep these attribution lines EXACTLY as written, in English:
{attribution}
4. "access unverified" must not become "cut off"; "unobservable" means unknown, not open.
Return only the translated Markdown."""


class CopilotService:
    """One per dashboard server. ``model_factory`` is injectable for tests; by default it reads the environment."""

    def __init__(self, cfg: BatoConfig, model_factory: Callable[[], ChatModel] | None = None,
                 now: Callable[[], datetime] | None = None):
        self.cfg = cfg
        self.model_factory = model_factory or model_from_env
        self.now = now or (lambda: datetime.now(timezone.utc).replace(microsecond=0))
        self._packs: dict[tuple[str, str], EvidencePack] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ status and evidence
    def status(self) -> dict[str, Any]:
        try:
            if self.model_factory is model_from_env:
                info = config_from_env().public()
            else:
                info = {"provider": getattr(self.model_factory(), "label", "configured")}
            return {"available": True, **info}
        except CopilotUnavailable as exc:
            return {"available": False, "reason": str(exc)}

    def pack(self, aoi_id: str, candidate_id: str) -> EvidencePack:
        key = (aoi_id, candidate_id)
        with self._lock:
            if key not in self._packs:
                self._packs[key] = build_pack(self.cfg, aoi_id, candidate_id)
            return self._packs[key]

    # ------------------------------------------------------------------ questions
    def ask(self, aoi_id: str, candidate_id: str, question: str, lang: str = "en") -> dict[str, Any]:
        if lang not in LANGS:
            raise ValueError("lang must be 'en' or 'ne'")
        question = (question or "").strip()
        if not question or len(question) > MAX_QUESTION:
            raise ValueError(f"the question must be 1-{MAX_QUESTION} characters")
        model = self.model_factory()                                      # CopilotUnavailable if not configured
        pack = self.pack(aoi_id, candidate_id)
        items = pack.retrieve(question)
        by_id = {it["id"]: it for it in items}
        system = SYSTEM.format(language=LANGS[lang])
        user = ("EVIDENCE (the only facts you may use):\n"
                + json.dumps([{k: it[k] for k in ("id", "title", "text")} for it in items], ensure_ascii=False)
                + f"\n\nQUESTION:\n{question}")
        problems: list[str] = []
        parsed: dict[str, Any] | None = None
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                user + "\n\nYOUR PREVIOUS ANSWER WAS REJECTED by the evidence check: " + "; ".join(problems)
                + ". Answer again following the rules exactly.")
            raw = model.complete(system, prompt)
            parsed, problems = self._check(raw, by_id, question, lang)
            if not problems:
                break
        assert parsed is not None or problems
        if problems:
            return {"status": "withheld", "language": lang, "model": model.label, "answer": None,
                    "problems": problems, "evidence": [_public(it) for it in items if it["id"] in _CORE_IDS],
                    "message": "The model's answer did not pass the evidence check and is not shown. The verified "
                               "evidence is listed instead."}
        assert parsed is not None
        cited = [_public(by_id[c]) for c in parsed["citations"]]
        return {"status": "answered" if parsed["supported"] else "unsupported", "language": lang,
                "model": model.label, "answer": parsed["answer"], "citations": cited,
                "checks": {"citations_known": True, "numbers_grounded": True},
                "note": "AI-generated answer; every number was checked against the cited BATO evidence."}

    def _check(self, raw: str, by_id: dict[str, dict[str, Any]], question: str,
               lang: str) -> tuple[dict[str, Any] | None, list[str]]:
        try:
            data = json.loads(_FENCE.sub("", raw.strip()))
            supported, answer, cites = bool(data["supported"]), str(data["answer"]).strip(), data.get("citations", [])
            if not isinstance(cites, list) or not answer:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return None, ["the reply was not the required JSON object"]
        cites = list(dict.fromkeys(str(c) for c in cites))
        problems = [f"unknown evidence id {c!r}" for c in cites if c not in by_id]
        if supported and not cites:
            problems.append("a supported answer must cite at least one evidence id")
        bad = ungrounded_numbers(answer, [by_id[c] for c in cites if c in by_id], question)
        if bad:
            problems.append("numbers not found in the cited evidence: " + ", ".join(bad))
        if lang == "ne" and not _DEVANAGARI.search(answer):
            problems.append("the answer must be written in Nepali (Devanagari)")
        return {"supported": supported, "answer": answer, "citations": cites}, problems

    # ------------------------------------------------------------------ SITREPs
    def sitrep(self, aoi_id: str, candidate_id: str, lang: str, title: str) -> dict[str, Any]:
        if lang not in LANGS:
            raise ValueError("lang must be 'en' or 'ne'")
        pack = self.pack(aoi_id, candidate_id)
        title = (title or aoi_id).strip()[:80]
        try:
            english = render(pack.facts, title=title, incident=f"flood event, {title}", issued_at=self.now())
        except SitrepError as exc:
            raise EvidenceError(str(exc)) from exc
        base = {"english": english, "facts_source": pack.facts.connectivity_run}
        if lang == "en":
            return {**base, "status": "deterministic", "language": "en", "markdown": english,
                    "note": "Deterministic template filled from the verified official result (no AI)."}
        model = self.model_factory()
        source_numbers = numbers_in(english)
        problems: list[str] = []
        for attempt in range(2):
            prompt = english if attempt == 0 else (
                english + "\n\nYOUR PREVIOUS TRANSLATION WAS REJECTED: " + "; ".join(problems))
            nepali = _FENCE.sub("", model.complete(
                TRANSLATE.format(attribution="\n".join(REQUIRED_ATTRIBUTIONS)), prompt)).strip()
            problems = _translation_problems(nepali, source_numbers)
            if not problems:
                return {**base, "status": "translated", "language": "ne", "model": model.label, "markdown": nepali,
                        "note": "AI translation of the deterministic English SITREP. Every number and the attribution "
                                "were checked against the English source, which remains authoritative."}
        return {**base, "status": "withheld", "language": "ne", "model": model.label, "markdown": None,
                "problems": problems,
                "message": "The Nepali translation did not pass the check and is not shown. The English SITREP is "
                           "authoritative."}


_CORE_IDS = {"run", "dates", "roads", "stretches", "settlements", "not_assessed", "method", "limits"}


def _public(it: dict[str, Any]) -> dict[str, Any]:
    return {"id": it["id"], "title": it["title"], "text": it["text"], "source": it["source"]}


def _translation_problems(nepali: str, source_numbers: set[str]) -> list[str]:
    problems = []
    got = numbers_in(nepali)
    if got - source_numbers:
        problems.append("numbers not in the English SITREP: " + ", ".join(sorted(got - source_numbers)))
    if source_numbers - got:
        problems.append("numbers missing from the translation: " + ", ".join(sorted(source_numbers - got)))
    for line in REQUIRED_ATTRIBUTIONS:
        if line not in nepali:
            problems.append(f"attribution changed or missing: {line[:40]}...")
    if not _DEVANAGARI.search(nepali):
        problems.append("the translation is not in Devanagari")
    return problems


__all__ = ["CopilotService", "CopilotUnavailable", "EvidenceError", "ProviderError", "LANGS"]

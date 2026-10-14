"""SITREP copilot with MOCKED model providers: grounding checks, provider configuration, official-v1-only evidence,
English / Nepali answers, unsupported questions, hallucinated numbers, missing credentials and the API handler.
Synthetic world = invented data (connectivity_fakes / sitrep_fakes)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from bato.attribution import REQUIRED_ATTRIBUTIONS
from bato.copilot.api import handle
from bato.copilot.evidence import EvidenceError, build_pack
from bato.copilot.grounding import answer_numbers, canonical, numbers_in, ungrounded_numbers
from bato.copilot.provider import CopilotUnavailable, config_from_env
from bato.copilot.service import CopilotService
from bato.dashboard.sitrep import EMSR927_STATEMENT, NO_REFERENCE_STATEMENT

from copilot_fakes import EchoModel, ScriptedModel, reply, translate_like
from sitrep_fakes import published_sitrep_world

NOW = datetime(2026, 10, 14, 6, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    w = published_sitrep_world(tmp_path_factory.mktemp("copilot"))
    mp.undo()
    return w


def service(world, model) -> CopilotService:
    return CopilotService(world.cfg, model_factory=lambda: model, now=lambda: NOW)


# ---------------------------------------------------------------------------
# Grounding (no model)
# ---------------------------------------------------------------------------


def test_numbers_are_normalised_including_devanagari():
    assert canonical("2,942.1610") == "2942.161" and canonical("08") == "8" and canonical("०.६३५") == "0.635"
    assert numbers_in("२,९४२.१६१ km and 2942.161 km") == {"2942.161"}
    assert answer_numbers("1. Blocked: 13 segments [stretch:blk-0]\n## 2) Next") == {"13"}


def test_ungrounded_numbers_flags_invented_values_only():
    ev = [{"text": "blocked 13 segments (0.635 km), unobservable 2,942.161 km", "values": {"km": 2942.161}}]
    assert ungrounded_numbers("13 segments, 0.635 km and 2,942.2 km (about 2942 km)", ev) == []
    assert ungrounded_numbers("57 settlements are cut off", ev) == ["57"]
    assert ungrounded_numbers("1 km", [{"text": "", "values": {"x": 0.635}}]) == ["1"]     # no rounding to integers
    assert ungrounded_numbers("Is 50 right? No: 13.", ev, question="Are 50 blocked?") == []


# ---------------------------------------------------------------------------
# Provider configuration (environment only; never the key)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env, reason", [
    ({}, "BATO_LLM_PROVIDER"),
    ({"BATO_LLM_PROVIDER": "openai"}, "BATO_LLM_MODEL is not set"),
    ({"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "m"}, "BATO_LLM_API_KEY is not set"),
    ({"BATO_LLM_PROVIDER": "anthropic", "BATO_LLM_MODEL": "m"}, "BATO_LLM_API_KEY is not set"),
    ({"BATO_LLM_PROVIDER": "acme", "BATO_LLM_MODEL": "m", "BATO_LLM_API_KEY": "k"}, "not supported"),
    ({"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "m", "BATO_LLM_API_KEY": "k",
      "BATO_LLM_BASE_URL": "http://example.org/v1"}, "https"),
])
def test_incomplete_configuration_is_unavailable(env, reason):
    with pytest.raises(CopilotUnavailable, match=reason):
        config_from_env(env)


def test_configuration_never_exposes_the_key():
    c = config_from_env({"BATO_LLM_PROVIDER": "anthropic", "BATO_LLM_MODEL": "claude-x",
                         "BATO_LLM_API_KEY": "sk-SECRET"})
    assert c.api_key == "sk-SECRET" and "sk-SECRET" not in json.dumps(c.public())
    assert c.public() == {"provider": "anthropic", "model": "claude-x", "endpoint_host": "api.anthropic.com"}
    local = config_from_env({"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "llama3.1",
                             "BATO_LLM_BASE_URL": "http://localhost:11434/v1"})
    assert local.api_key is None                                       # a local server needs no key


def test_missing_credentials_report_unavailable_without_calling_a_model(world, monkeypatch):
    for k in ("BATO_LLM_PROVIDER", "BATO_LLM_MODEL", "BATO_LLM_API_KEY", "BATO_LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    svc = CopilotService(world.cfg)
    st = svc.status()
    assert st["available"] is False and "BATO_LLM_PROVIDER" in st["reason"]
    with pytest.raises(CopilotUnavailable):
        svc.ask("aoi-test", "ts-test", "How many roads are blocked?")
    code, payload = handle(svc, "POST", "/api/copilot/ask", {"X-BATO-Copilot": "1"},
                           json.dumps({"aoi_id": "aoi-test", "candidate_id": "ts-test", "question": "x"}).encode())
    assert code == 503 and payload["status"] == "unavailable" and "Copilot unavailable" in payload["error"]
    with pytest.raises(CopilotUnavailable):
        svc.sitrep("aoi-test", "ts-test", "ne", "Synthetic")          # Nepali needs the model ...
    en = svc.sitrep("aoi-test", "ts-test", "en", "Synthetic")         # ... the English template does not
    assert en["status"] == "deterministic" and "no AI" in en["note"]


# ---------------------------------------------------------------------------
# Evidence: official v1 only
# ---------------------------------------------------------------------------


def test_evidence_pack_is_the_verified_official_v1_result(world):
    pack = build_pack(world.cfg, "aoi-test", "ts-test")
    s = world.prov.summary
    ids = {it["id"] for it in pack.core}
    assert {"run", "dates", "roads", "stretches", "settlements", "not_assessed", "limits", "attribution"} <= ids
    roads = next(it for it in pack.core if it["id"] == "roads")
    assert roads["values"]["segments"]["blocked"] == s.road_segments_by_state["blocked"]
    assert roads["values"]["km"] == {k: s.road_km_by_state[k] for k in ("blocked", "open", "unobservable")}
    assert len(pack.stretches) == s.blocked_stretches and len(pack.settlements) == s.settlements_total
    assert all(it["source"] for it in pack.core + pack.stretches + pack.settlements)
    limits = next(it for it in pack.core if it["id"] == "limits")["text"]
    assert NO_REFERENCE_STATEMENT.replace("**", "") in limits and "EMSR927" not in limits   # not the evaluated run
    for line in REQUIRED_ATTRIBUTIONS:
        assert line in next(it for it in pack.core if it["id"] == "attribution")["text"]


def test_evidence_uses_the_v1_run_never_the_later_v2_run(tmp_path):
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp = pytest.MonkeyPatch()
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    w = published_sitrep_world(tmp_path, extra_v2=True)
    mp.undo()
    pack = build_pack(w.cfg, "aoi-test", "ts-test")
    assert pack.facts.road_rule == "v1" and pack.facts.connectivity_run.startswith("ts-test.superseded-")
    assert "road rule v1" in next(it for it in pack.core if it["id"] == "run")["text"]


def test_evidence_refuses_bad_identifiers_and_missing_runs(world):
    with pytest.raises(EvidenceError, match="invalid identifier"):
        build_pack(world.cfg, "../etc", "ts-test")
    with pytest.raises(EvidenceError, match="no road rule v1"):
        build_pack(world.cfg, "aoi-test", "ts-missing")


def test_retrieval_adds_settlements_for_settlement_questions(world):
    pack = build_pack(world.cfg, "aoi-test", "ts-test")
    plain = {it["id"] for it in pack.retrieve("Which roads are blocked?")}
    assert not any(i.startswith("settlement:") for i in plain) and all(st["id"] in plain for st in pack.stretches)
    asked = {it["id"] for it in pack.retrieve("Which settlements have unverified access?")}
    assert "calc:settlement_list" in asked and any(i.startswith("settlement:") for i in asked)
    assert "settlement:place-node-1" in {it["id"] for it in pack.retrieve("What about Town T?")}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def test_english_answer_is_grounded_and_cited(world):
    model = ScriptedModel([reply("13 road segments in 3 blocked stretches show flood evidence (0.635 km) "
                                 "[roads] [stretches].", ["roads", "stretches"])])
    r = service(world, model).ask("aoi-test", "ts-test", "How many roads are blocked?")
    assert r["status"] == "answered" and r["checks"] == {"citations_known": True, "numbers_grounded": True}
    assert [c["id"] for c in r["citations"]] == ["roads", "stretches"] and "AI-generated" in r["note"]
    system, user = model.calls[0]
    assert "English" in system and '"id": "roads"' in user and "v2" not in user.lower().replace("v2-rc", "")


def test_nepali_answer_with_devanagari_digits(world):
    model = ScriptedModel([reply("१३ सडक खण्डहरू (०.६३५ km) मा बाढीको प्रमाण छ [roads]।", ["roads"])])
    r = service(world, model).ask("aoi-test", "ts-test", "कति सडक अवरुद्ध छन्?", lang="ne")
    assert r["status"] == "answered" and r["language"] == "ne" and "१३" in r["answer"]
    assert "Nepali" in model.calls[0][0]


def test_unsupported_question_is_reported_not_invented(world):
    model = ScriptedModel([reply("BATO evidence does not contain population or casualty figures.", ["limits"],
                                 supported=False)])
    r = service(world, model).ask("aoi-test", "ts-test", "How many people died?")
    assert r["status"] == "unsupported" and "does not contain" in r["answer"]


def test_hallucinated_number_is_withheld_after_one_retry(world):
    bad = reply("57 settlements are cut off and 1,200 people are affected [settlements].", ["settlements"])
    model = ScriptedModel([bad, bad])
    r = service(world, model).ask("aoi-test", "ts-test", "How many settlements are cut off?")
    assert r["status"] == "withheld" and r["answer"] is None
    assert "57" in r["problems"][0] and "1200" in r["problems"][0]
    assert len(model.calls) == 2 and "REJECTED" in model.calls[1][1]
    assert {e["id"] for e in r["evidence"]} >= {"roads", "settlements"}       # verified evidence instead


def test_hallucination_corrected_on_retry_is_accepted(world):
    model = ScriptedModel([reply("57 settlements are cut off [settlements].", ["settlements"]),
                           reply("3 settlements are confirmed cut off [settlements].", ["settlements"])])
    r = service(world, model).ask("aoi-test", "ts-test", "How many settlements are cut off?")
    assert r["status"] == "answered" and r["answer"].startswith("3 settlements")


@pytest.mark.parametrize("raw, problem", [
    (reply("13 segments [made-up]", ["made-up"]), "unknown evidence id"),
    ("not json at all", "required JSON"),
    (reply("13 road segments are blocked.", []), "must cite"),
    (reply("13 road segments are blocked [roads].", ["roads"]), "Devanagari"),         # asked in Nepali
])
def test_invalid_answers_are_withheld(world, raw, problem):
    lang = "ne" if problem == "Devanagari" else "en"
    r = service(world, ScriptedModel([raw, raw])).ask("aoi-test", "ts-test", "Blocked roads?", lang=lang)
    assert r["status"] == "withheld" and any(problem in p for p in r["problems"])


def test_bad_requests_are_rejected(world):
    svc = service(world, ScriptedModel([]))
    for kw in ({"question": ""}, {"question": "x" * 1001}, {"question": "ok", "lang": "fr"}):
        with pytest.raises(ValueError):
            svc.ask("aoi-test", "ts-test", kw["question"], kw.get("lang", "en"))


# ---------------------------------------------------------------------------
# SITREPs
# ---------------------------------------------------------------------------


def test_english_sitrep_is_the_deterministic_template_without_a_model(world):
    def no_model():
        raise AssertionError("the English SITREP must not call the model")
    r = CopilotService(world.cfg, model_factory=no_model, now=lambda: NOW).sitrep("aoi-test", "ts-test", "en", "Synth")
    assert r["status"] == "deterministic" and r["markdown"] == r["english"]
    assert "**13 road segments in 3 stretches (0.635 km)**" in r["markdown"]
    assert all(line in r["markdown"] for line in REQUIRED_ATTRIBUTIONS)
    assert EMSR927_STATEMENT not in r["markdown"]                       # not the evaluated run


def test_nepali_sitrep_translation_is_number_checked(world):
    r = service(world, EchoModel()).sitrep("aoi-test", "ts-test", "ne", "Synth")
    assert r["status"] == "translated" and "सञ्चालन सारांश" in r["markdown"] and "१३" in r["markdown"]
    assert all(line in r["markdown"] for line in REQUIRED_ATTRIBUTIONS) and "English source" in r["note"]


def test_nepali_sitrep_with_changed_number_or_attribution_is_withheld(world):
    english = service(world, EchoModel()).sitrep("aoi-test", "ts-test", "en", "Synth")["markdown"]
    changed = translate_like(english).replace("०.६३५", "०.७३५")              # 0.635 km -> 0.735 km
    dropped = translate_like(english).replace(REQUIRED_ATTRIBUTIONS[1], "WorldDEM")
    for bad, msg in ((changed, "not in the English SITREP"), (dropped, "attribution changed")):
        r = service(world, ScriptedModel([bad, bad])).sitrep("aoi-test", "ts-test", "ne", "Synth")
        assert r["status"] == "withheld" and r["markdown"] is None and any(msg in p for p in r["problems"])


# ---------------------------------------------------------------------------
# API handler
# ---------------------------------------------------------------------------


def test_api_handler(world):
    svc = service(world, EchoModel())
    body = json.dumps({"aoi_id": "aoi-test", "candidate_id": "ts-test",
                       "question": "Which roads are blocked?"}).encode()
    hdr = {"X-BATO-Copilot": "1"}
    assert handle(svc, "GET", "/api/copilot/status", {}, b"")[0] == 200
    assert handle(svc, "POST", "/api/copilot/ask", {}, body)[0] == 403                 # no custom header
    assert handle(svc, "GET", "/api/copilot/ask", hdr, body)[0] == 405
    assert handle(svc, "POST", "/api/copilot/ask", hdr, b"{oops")[0] == 400
    assert handle(svc, "POST", "/api/copilot/ask", hdr, b"x" * 20000)[0] == 413
    assert handle(svc, "POST", "/api/copilot/nope", hdr, body)[0] == 404
    code, payload = handle(svc, "POST", "/api/copilot/ask", hdr, body)
    assert code == 200 and payload["status"] == "answered" and payload["citations"][0]["id"] == "roads"
    code, payload = handle(svc, "POST", "/api/copilot/ask", hdr,
                           json.dumps({"aoi_id": "aoi-test", "candidate_id": "nope", "question": "q"}).encode())
    assert code == 409 and payload["status"] == "no_evidence"

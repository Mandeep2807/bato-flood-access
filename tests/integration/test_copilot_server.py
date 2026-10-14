"""SITREP copilot end to end with a MOCKED provider: scripts/serve_dashboard.py runs as a subprocess, configured
only through BATO_LLM_* environment variables, and talks over HTTP to a local fake OpenAI-compatible endpoint
(copilot_fakes.FakeLLMServer). Checks the API, the key handling, the unavailable state and (if Playwright is
installed) the dashboard panel in a headless browser. Synthetic world = invented data."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bato.copilot.provider import HttpChatModel, config_from_env
from bato.dashboard.bundle import publish

from copilot_fakes import FakeLLMServer, reply
from dashboard_fakes import inputs_for
from sitrep_fakes import published_sitrep_world

REPO = Path(__file__).resolve().parents[2]
SECRET = "sk-test-NEVER-SHOWN-123"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve(world, extra_env: dict[str, str]):
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR" and not k.startswith("BATO_LLM_")}
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "serve_dashboard.py"), "--config-dir",
                             str(world.root / "config"), "--port", str(port)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    return proc, f"http://127.0.0.1:{port}/"


def _api(url: str, path: str, body: dict | None = None, header: bool = True) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", **({"X-BATO-Copilot": "1"} if header else {})}
    req = urllib.request.Request(url + "api/copilot/" + path, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    w = published_sitrep_world(tmp_path_factory.mktemp("copilot-server"))
    mp.undo()
    publish(w.cfg, "aoi-test", "ts-test", inputs_for(w, rasters=False))
    return w


@pytest.fixture(scope="module")
def configured(world):
    with FakeLLMServer() as llm:
        proc, url = _serve(world, {"BATO_LLM_PROVIDER": "openai", "BATO_LLM_MODEL": "fake-model",
                                   "BATO_LLM_API_KEY": SECRET, "BATO_LLM_BASE_URL": f"{llm.base_url}/v1"})
        yield url, llm
        proc.terminate()
        out = proc.communicate(timeout=5)[0]
    assert SECRET not in out                                           # never printed or logged


@pytest.fixture(scope="module")
def unconfigured(world):
    proc, url = _serve(world, {})
    yield url
    proc.terminate()
    proc.communicate(timeout=5)


Q = {"aoi_id": "aoi-test", "candidate_id": "ts-test"}


def test_status_and_answers_through_a_real_http_provider(configured):
    url, llm = configured
    code, st = _api(url, "status")
    assert code == 200 and st == {"available": True, "provider": "openai", "model": "fake-model",
                                  "endpoint_host": "127.0.0.1"}
    code, en = _api(url, "ask", {**Q, "question": "Which roads are blocked?", "lang": "en"})
    assert code == 200 and en["status"] == "answered" and en["citations"][0]["id"] == "roads"
    assert "13 segments" in en["answer"] and en["model"] == "openai:fake-model"
    code, ne = _api(url, "ask", {**Q, "question": "कति बस्तीहरूको पहुँच अपुष्ट छ?", "lang": "ne"})
    assert code == 200 and ne["status"] == "answered" and "प्रमाण अनुसार" in ne["answer"]
    code, un = _api(url, "ask", {**Q, "question": "What is the population affected?", "lang": "en"})
    assert code == 200 and un["status"] == "unsupported"
    sent = llm.requests[0]
    assert sent["path"] == "/v1/chat/completions" and sent["headers"]["authorization"] == f"Bearer {SECRET}"
    assert sent["body"]["temperature"] == 0 and sent["body"]["model"] == "fake-model"
    for payload in (st, en, ne, un):
        assert SECRET not in json.dumps(payload)                       # the key never reaches the browser


def test_sitreps_through_the_server(configured):
    url, _llm = configured
    code, en = _api(url, "sitrep", {**Q, "lang": "en", "title": "Synthetic"})
    assert code == 200 and en["status"] == "deterministic" and "13 road segments in 3 stretches" in en["markdown"]
    code, ne = _api(url, "sitrep", {**Q, "lang": "ne", "title": "Synthetic"})
    assert code == 200 and ne["status"] == "translated" and "सञ्चालन सारांश" in ne["markdown"]


def test_requests_without_the_custom_header_are_refused(configured):
    url, llm = configured
    before = len(llm.requests)
    code, _ = _api(url, "ask", {**Q, "question": "Which roads are blocked?"}, header=False)
    assert code == 403 and len(llm.requests) == before                 # no model call, no quota spent


def test_unconfigured_server_reports_the_copilot_unavailable(unconfigured):
    code, st = _api(unconfigured, "status")
    assert code == 200 and st["available"] is False and "BATO_LLM_PROVIDER" in st["reason"]
    code, r = _api(unconfigured, "ask", {**Q, "question": "Which roads are blocked?"})
    assert code == 503 and r["status"] == "unavailable"
    code, en = _api(unconfigured, "sitrep", {**Q, "lang": "en", "title": "Synthetic"})
    assert code == 200 and en["status"] == "deterministic"            # not labelled as AI
    with urllib.request.urlopen(unconfigured + "bundles/index.json", timeout=10) as r:
        assert r.status == 200                                         # the dashboard itself is unaffected


def test_anthropic_messages_format():
    with FakeLLMServer() as llm:
        model = HttpChatModel(config_from_env({"BATO_LLM_PROVIDER": "anthropic", "BATO_LLM_MODEL": "claude-test",
                                               "BATO_LLM_API_KEY": SECRET, "BATO_LLM_BASE_URL": llm.base_url
                                               .replace("127.0.0.1", "localhost")}))
        llm.model.complete = lambda system, user: reply("ok [roads]", ["roads"])     # type: ignore[method-assign]
        assert json.loads(model.complete("sys", "usr"))["answer"] == "ok [roads]"
        sent = llm.requests[0]
        assert sent["path"] == "/v1/messages" and sent["headers"]["x-api-key"] == SECRET
        assert sent["body"]["system"] == "sys" and sent["body"]["messages"][0]["content"] == "usr"


# ---------------------------------------------------------------------------
# Browser (optional: Playwright + Chromium)
# ---------------------------------------------------------------------------

needs_browser = pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="playwright not installed")


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--use-gl=swiftshader", "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        except Exception as exc:                                       # pragma: no cover - environment
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


def _open(browser, url):
    page = browser.new_page(viewport={"width": 1366, "height": 900}, reduced_motion="reduce")
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("window.BATO && window.BATO.ready || window.BATO_ERROR", timeout=30000)
    assert page.evaluate("window.BATO_ERROR || null") is None
    page.locator('#rail .tab[data-k="copilot"]').click()
    return page, errors


@needs_browser
def test_copilot_panel_answers_cites_and_links_to_the_map(configured, browser):
    url, _llm = configured
    page, errors = _open(browser, url)
    page.wait_for_function("document.querySelector('#cpStatus').textContent.includes('ready')", timeout=10000)
    page.locator("#cpQ").fill("Which roads are blocked?")
    page.locator("#cpAsk").click()
    page.wait_for_selector(".cp-badge.answered", timeout=20000)
    assert "13 segments" in page.locator(".cp-answer").inner_text()
    assert page.locator(".cp-cite code").first.inner_text() == "roads"
    page.locator('[data-lang="ne"]').click()
    page.locator(".cp-x").first.click()                                # a Nepali example question
    page.wait_for_function("document.querySelector('.cp-answer') && "
                           "document.querySelector('.cp-answer').textContent.includes('प्रमाण')", timeout=20000)
    page.locator("#cpSitNe").click()
    page.wait_for_selector("#cpSit .cp-md", timeout=20000)
    assert "सञ्चालन सारांश" in page.locator("#cpSit .cp-md").inner_text()
    assert "AI translation" in page.locator("#cpSit .cp-badge").text_content()
    # a cited blocked stretch opens the existing evidence drawer on the map
    page.locator('[data-lang="en"]').click()
    page.locator("#cpQ").fill("Which blocked stretch matters most?")
    page.locator("#cpAsk").click()
    page.wait_for_selector(".cp-cite .cp-show", state="attached", timeout=20000)
    assert page.locator(".cp-cite code").first.inner_text().startswith("stretch:blk-")
    page.locator(".cp-cite").first.locator("summary").click()
    page.locator(".cp-show").first.click()
    page.wait_for_function("!document.querySelector('#drawer').hidden", timeout=10000)
    assert errors == []
    page.close()


@needs_browser
def test_copilot_panel_states_when_unavailable(unconfigured, browser):
    page, errors = _open(browser, unconfigured)
    page.wait_for_function("document.querySelector('#cpStatus').textContent.includes('unavailable')", timeout=10000)
    assert page.locator("#cpAsk").is_disabled() and page.locator("#cpSitNe").is_disabled()
    page.locator("#cpSitEn").click()
    page.wait_for_selector("#cpSit .cp-md", timeout=20000)
    assert "Deterministic template · no AI" in page.locator("#cpSit .cp-badge").text_content()
    page.locator('#rail .tab[data-k="access"]').click()                # existing panels still work
    assert page.locator("#panelBody").inner_text().strip()
    assert errors == []
    page.close()

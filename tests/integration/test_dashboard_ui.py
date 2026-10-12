"""Browser smoke test of the dashboard on the synthetic world (TEST FIXTURE bundle, headless Chromium).

Skipped unless Playwright and its Chromium are installed (pip install playwright; playwright install chromium).
Online basemap tiles are blocked in the test; the app must work without them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bato.dashboard.bundle import publish

from dashboard_fakes import inputs_for, published_world

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="playwright not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    import bato.osm.extract as extract

    from connectivity_fakes import fake_read_parquet_rows
    real = extract.read_parquet_rows
    mp.setattr(extract, "read_parquet_rows", lambda data: fake_read_parquet_rows(data, real))
    world = published_world(tmp_path_factory.mktemp("ui"))
    mp.undo()
    out = publish(world.cfg, "aoi-test", "ts-test", inputs_for(world))
    bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k != "BATO_CONFIG_DIR"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "serve_dashboard.py"), "--config-dir",
                             str(world.root / "config"), "--port", str(port)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield url, bundle
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            # software WebGL (SwiftShader); recent Chromium needs --enable-unsafe-swiftshader to allow it
            b = p.chromium.launch(args=["--use-gl=swiftshader", "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        except Exception as exc:                                      # pragma: no cover - environment
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


def open_page(browser, url, size=(1366, 768)):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]}, reduced_motion="reduce")
    errors: list[str] = []
    console: list[str] = []
    failed: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("requestfailed", lambda r: None if "cartocdn" in r.url else failed.append(f"{r.url} {r.failure}"))
    page.route(re.compile("cartocdn"), lambda r: r.abort())        # the optional online basemap is never needed
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_function("window.BATO && window.BATO.ready || window.BATO_ERROR", timeout=30000)
    except Exception as exc:                                          # report where the bootstrap stopped
        state = page.evaluate("({stage: window.BATO_STAGE || null, bato: !!window.BATO, "
                              "ready: !!(window.BATO && window.BATO.ready), webgl2: "
                              "!!document.createElement('canvas').getContext('webgl2')})")
        pytest.fail(f"dashboard never became ready: {state}; page errors {errors}; console {console[-10:]}; "
                    f"failed requests {failed}; {exc}")
    error = page.evaluate("window.BATO_ERROR || null")
    assert error is None, f"dashboard bootstrap failed: {error}; page errors {errors}; console {console[-10:]}"
    return page, errors


def test_loads_with_real_bundle_numbers(served, browser):
    url, b = served
    page, errors = open_page(browser, url)
    s = b["summary"]
    txt = lambda sel: page.locator(sel).inner_text().strip()  # noqa: E731
    assert txt('[data-k="blocked"] .value b') == f"{s['road_km_by_state']['blocked']:,.1f}"
    assert txt('[data-k="open"] .value b') == f"{s['road_km_by_state']['open']:,.1f}"
    assert txt('[data-k="unobservable"] .value b') == f"{s['road_km_by_state']['unobservable']:,.1f}"
    access = txt('[data-k="access"] .access-line')
    for k in ("CONNECTED", "POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF"):
        assert str(s["settlements_by_access_state"][k]) in access
    assert page.locator(".units i").count() == s["settlements_total"]
    assert "Synthetic test world" in txt("#incident") and page.locator(".badge-nv").is_visible()
    assert page.evaluate("document.scrollingElement.scrollWidth <= window.innerWidth")
    assert errors == []
    page.close()


def test_every_panel_and_every_object_opens(served, browser):
    url, b = served
    page, errors = open_page(browser, url)
    page.evaluate("BATO.closePanel()")
    for key in ("incident", "map", "access", "observability", "evidence", "operations", "methodology"):
        page.locator(f'#rail .tab[data-k="{key}"]').click()
        assert page.locator("#panel").get_attribute("hidden") is None
        assert page.locator("#panelBody").inner_text().strip()
    for st in b["settlements"]:
        page.evaluate(f"BATO.selectSettlement({json.dumps(st['id'])})")
        assert page.locator("#drawer").get_attribute("hidden") is None
        assert "Building count proxy" in page.locator("#drawerBody").inner_text()
    for i, state in enumerate(b["roads"]["state"]):
        if state != 2 or i % 25 == 0:
            page.evaluate(f"BATO.selectSegment({i})")
            assert page.locator("#drawerHead").inner_text().lower().startswith("road impact")
    for p in b["pockets"]:
        page.evaluate(f"BATO.selectPocket({json.dumps(p['id'])})")
    for d in b["destinations"]:
        page.evaluate(f"BATO.selectDestination({json.dumps(d['id'])})")
    assert errors == []
    page.close()


def test_click_blocked_road_opens_evidence_drawer(served, browser):
    url, b = served
    page, errors = open_page(browser, url)
    page.evaluate("BATO.closePanel()")
    top = b["stretches"][0]
    i = top["segments"][len(top["segments"]) // 2]
    page.evaluate(f"BATO.mapApi.map.jumpTo({{center: [{top['lon']}, {top['lat']}], zoom: 15}})")
    page.wait_for_timeout(600)
    c = b["roads"]["coords"][i]
    js = (f"(() => {{ const p = BATO.mapApi.map.project([{(c[0] + c[-2]) / 2}, {(c[1] + c[-1]) / 2}]); "
          "const r = document.getElementById('map').getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; })()")
    xy = page.evaluate(js)
    page.mouse.click(*xy)
    page.wait_for_timeout(300)
    body = page.locator("#drawerBody").inner_text()
    assert "BLOCKED" in page.locator("#drawerHead").inner_text().upper()
    assert "reconnected if this stretch is cleared" in body
    assert "PRE-1" in body and "POST" in body
    page.locator("summary", has_text="View evidence").click()
    text = page.locator("#drawerBody").inner_text()
    shown = [b["roads"]["evidence"][k] for k in top["segments"] if b["roads"]["evidence"][k] in text]
    assert len(shown) == 1 and b["evidence"][shown[0]]["method"] in text      # the clicked segment's own record
    assert page.locator(".sar img").count() >= 5
    assert errors == []
    page.close()


def test_settlement_filter_and_operations(served, browser):
    url, b = served
    page, errors = open_page(browser, url)
    page.locator('#rail .tab[data-k="access"]').click()
    n0 = page.locator("#flist .row").count()
    page.locator("#fchips .chip", has_text="Connected").click()
    n1 = page.locator("#flist .row").count()
    assert n1 == n0 - b["summary"]["settlements_by_access_state"]["CONNECTED"]
    page.locator("#fsearch").fill("Village C")
    assert page.locator("#flist .row").count() == 1
    page.locator('#rail .tab[data-k="operations"]').click()
    assert page.locator("#ol1 .row").count() == len(b["stretches"])
    assert "no rescue priority score" in page.locator("#panelBody").inner_text()
    page.select_option("#osort", "bldg")
    page.locator("#ol1 .row").first.click()
    assert page.locator("#drawer").get_attribute("hidden") is None
    assert errors == []
    page.close()


def test_observability_mode_and_layers(served, browser):
    url, _ = served
    page, errors = open_page(browser, url)
    page.locator('[data-k="unobservable"]').click()
    assert page.evaluate("BATO.mapApi.mode") == "observability"
    assert page.locator("#modeBanner").is_visible()
    assert "not safe" in page.locator("#modeBanner").inner_text()
    page.locator('.modes [data-mode="normal"]').click()
    assert page.evaluate("BATO.mapApi.mode") == "normal" and not page.locator("#modeBanner").is_visible()
    page.locator("#layersBtn").click()
    for cb in page.locator("#layersPop input").all():
        cb.click()
        cb.click()
    page.locator(".tl-node").first.click()
    assert page.locator(".popover").is_visible()
    assert errors == []
    page.close()


def test_judge_mode_and_follow_the_evidence(served, browser):
    url, _ = served
    for size in ((1366, 768), (1920, 1080)):
        page, errors = open_page(browser, url, size)
        page.locator("#judgeTab").click()
        assert page.locator("#judge").is_visible()
        for k in range(7):
            assert page.locator("#judge .step-n").inner_text().startswith(f"{k + 1} / 7")
            if k < 6:
                page.locator('#judge [data-k="next"]').click()
        assert page.locator("#finale").is_visible()
        page.locator('#judge [data-k="reset"]').click()
        assert page.locator("#judge .step-n").inner_text().startswith("1 / 7")
        page.keyboard.press("Escape")
        assert not page.locator("#judge").is_visible()
        page.evaluate("BATO.follow.run(null, {stageMs: 250})")
        page.wait_for_timeout(2200)
        assert page.locator(".f-stage.done").count() == 5
        assert page.locator("#drawer").get_attribute("hidden") is None
        assert errors == []
        page.close()


def test_stalled_basemap_does_not_block_the_official_result(served, browser):
    """Online basemap requests that never answer (offline / dead proxy) must not delay readiness."""
    url, _ = served
    page = browser.new_page(viewport={"width": 1366, "height": 768}, reduced_motion="reduce")
    held: list = []
    page.route(re.compile("cartocdn"), lambda r: held.append(r))     # never fulfilled
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("window.BATO && window.BATO.ready || window.BATO_ERROR", timeout=30000)
    assert page.evaluate("window.BATO_ERROR || null") is None and page.evaluate("window.BATO_STAGE") == "ready"
    page.unroute_all(behavior="ignoreErrors")
    page.close()


def test_bootstrap_failure_is_reported_not_hung(served, browser):
    """An exception while the map layers are built must end in window.BATO_ERROR, not a pending promise."""
    url, _ = served
    page = browser.new_page(viewport={"width": 1366, "height": 768}, reduced_motion="reduce")
    page.route(re.compile("cartocdn"), lambda r: r.abort())
    page.add_init_script("""(() => { const iv = setInterval(() => { if (!window.maplibregl) return; clearInterval(iv);
      const add = maplibregl.Map.prototype.addSource;
      maplibregl.Map.prototype.addSource = function (id, src) {
        if (id === "roads") throw new Error("TEST: injected build failure");
        return add.call(this, id, src); }; }, 1); })();""")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("window.BATO_ERROR", timeout=30000)
    err = page.evaluate("window.BATO_ERROR")
    assert "TEST: injected build failure" in err and "map-load" in err
    assert not page.evaluate("!!(window.BATO && window.BATO.ready)")
    assert "could not load the official result" in page.locator("#loading").inner_text()
    page.close()


def test_local_base_map_only_and_no_provider_tiles(served, browser):
    """The base map is local (background + DEM hillshade): no request leaves the dashboard server, so no provider
    tile (e.g. a CARTO "API KEY REQUIRED" watermark) can ever be drawn, and the dashboard works offline."""
    url, b = served
    page = browser.new_page(viewport={"width": 1366, "height": 768}, reduced_motion="reduce")
    external: list[str] = []
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    def offline(route):                                               # everything not served locally is refused
        if route.request.url.startswith(url) or route.request.url.startswith("data:"):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    page.route("**/*", offline)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("window.BATO && window.BATO.ready || window.BATO_ERROR", timeout=30000)
    assert page.evaluate("window.BATO_ERROR || null") is None and page.evaluate("window.BATO_STAGE") == "ready"
    assert external == []
    style = page.evaluate("BATO.mapApi.map.getStyle()")
    assert [k for k, src in style["sources"].items() if src.get("tiles") or src.get("type") == "raster"] == []
    assert style["layers"][0]["id"] == "bg"
    ids = {layer["id"] for layer in style["layers"]}
    assert {"flood", "hillshade", "road-open", "road-unobs", "road-blocked", "settle"} <= ids
    assert "API KEY REQUIRED" not in page.content().upper()
    attribution = page.locator(".maplibregl-ctrl-attrib").inner_text()
    assert "OpenStreetMap" in attribution and "CARTO" not in attribution
    # first screen: evidence layers on, detail layers one click away; the limitation is still stated in full
    assert page.evaluate("BATO.mapApi.layers.flood && BATO.mapApi.layers.blocked && !BATO.mapApi.layers.exits")
    assert "rule-based" in page.locator(".badge-nv").inner_text().lower()
    assert "Not validated against any reference map" in page.locator("#panelBody").inner_text()
    assert errors == []
    page.close()

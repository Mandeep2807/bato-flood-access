// BATO dashboard bootstrap: load the official bundle, build the model, wire the UI.
import { loadIndex, loadBundle, buildModel, day, esc, bboxOf } from "./data.js";
import { createMap, DEFAULT_LAYERS } from "./map.js";
import { renderStatus } from "./ui/status.js";
import { renderTimeline } from "./ui/timeline.js";
import { SECTIONS, render as renderPanel } from "./ui/panels.js";
import { createDrawer } from "./ui/drawer.js";
import { renderControls } from "./ui/controls.js";
import { createFollow } from "./ui/follow.js";
import { createJudge } from "./ui/judge.js";
import { ICON } from "./ui/dom.js";

const $ = (id) => document.getElementById(id);

const stage = (name) => { window.BATO_STAGE = name; };

// Any error before the dashboard is ready ends the bootstrap visibly (window.BATO_ERROR), never as a silent hang.
function onBootError(ev) {
  if (window.BATO && window.BATO.ready) return;
  const err = ev.reason || ev.error || new Error(ev.message || "unknown error");
  fail(err instanceof Error ? err : new Error(String(err)));
}
window.addEventListener("error", onBootError);
window.addEventListener("unhandledrejection", onBootError);

async function boot() {
  stage("index");
  const params = new URLSearchParams(location.search);
  const index = await loadIndex();
  if (!index.bundles || !index.bundles.length) throw new Error("No dashboard bundle has been built yet. Run scripts/build_dashboard.py.");
  const pick = index.bundles.find((b) => b.path === params.get("b")) || index.bundles[0];
  stage("bundle");
  const { bundle, root } = await loadBundle(pick.path);
  const m = buildModel(bundle, root);
  const app = { m, panel: null, mode: "normal" };
  window.BATO = app;   // for the browser smoke tests and the curious

  const inc = m.b.incident;
  $("incident").innerHTML = `<b>${esc(inc.title)} event · ${day(inc.event_date)}</b><span>${esc(inc.aoi_id)} · ${esc(inc.candidate_id)} · official result (road rule ${esc(inc.road_rule)})</span>`;
  document.title = `BATO · ${inc.title} · ${day(inc.event_date)}`;

  stage("map");
  app.mapApi = createMap($("map"), m, {
    segment: (i) => app.selectSegment(i),
    settlement: (id) => app.selectSettlement(id),
    destination: (id) => app.selectDestination(id),
    empty: () => app.clearSelection(),
  });
  app.drawer = createDrawer(app);
  const controls = renderControls($("mapctl"), app);
  const banner = $("modeBanner");

  // ---------------------------------------------------------------- app actions
  app.openPanel = (key) => {
    app.panel = key;
    const sec = SECTIONS.find((s) => s.key === key);
    $("panelTitle").textContent = sec.title;
    renderPanel(key, $("panelBody"), app);
    $("panel").hidden = false;
    document.querySelectorAll("#rail .tab").forEach((t) => t.setAttribute("aria-pressed", String(t.dataset.k === key)));
  };
  app.closePanel = () => {
    app.panel = null;
    $("panel").hidden = true;
    document.querySelectorAll("#rail .tab").forEach((t) => t.setAttribute("aria-pressed", "false"));
  };
  app.setMode = (mode) => {
    app.mode = mode;
    app.mapApi.setMode(mode);
    controls.sync(mode);
    banner.hidden = mode !== "observability";
    if (mode === "observability") {
      banner.innerHTML = `<p>Unobservable is not safe. Unobservable is not damaged.</p>
        <p>Dashed roads are where the satellite rule cannot confidently decide. Observed roads are dimmed; colour shows the main reason.</p>`;
    }
    if (app.panel === "observability") {
      const b = document.getElementById("radarBtn");
      if (b) b.setAttribute("aria-pressed", String(mode === "observability"));
    }
  };
  app.setRadarMode = (on) => app.setMode(on ? "observability" : "normal");
  app.setLayer = (key, on) => { app.mapApi.setLayer(key, on); controls.sync(app.mode); };
  app.home = () => app.mapApi.home();

  app.clearSelection = () => {
    app.mapApi.selectSegments([]);
    app.mapApi.selectSettlement(null);
    app.mapApi.focus([]);
    app.mapApi.trace([]);
    app.drawer.close();
  };
  app.selectSegment = (i, opts = {}) => {
    const info = m.segInfo(i);
    const idx = info.stretch ? info.stretch.segments : [i];
    app.mapApi.selectSettlement(null);
    app.mapApi.trace([]);
    app.mapApi.selectSegments(idx);
    app.mapApi.focus(info.state === "open" ? [] : idx);
    app.drawer.segment(i);
    if (opts.fly) app.mapApi.fit(bboxOf(m, idx), { maxZoom: 15 });
  };
  app.selectStretch = (id, opts = {}) => {
    const st = m.stretchById.get(id);
    if (!st) return;
    app.selectSegment(st.segments[0], { fly: opts.fly !== false });
  };
  app.selectSettlement = (id, opts = {}) => {
    const s = m.byId.get(id);
    if (!s) return;
    if (!opts.keepFocus) { app.mapApi.selectSegments([]); app.mapApi.focus([]); }
    app.mapApi.selectSettlement(id);
    const t = s.traces.post || s.traces.pre;
    app.mapApi.trace(t ? t.edges : []);
    app.drawer.settlement(id);
    if (opts.fly) {
      if (t && t.edges.length) app.mapApi.fit(bboxOf(m, t.edges), { maxZoom: 14.5 });
      else app.mapApi.flyTo(s.lon, s.lat, 13.5);
    }
  };
  app.selectPocket = (id) => {
    const p = m.pocketById.get(id);
    if (!p) return;
    app.mapApi.selectSettlement(null);
    app.mapApi.trace([]);
    app.mapApi.selectSegments(p.segments);
    app.mapApi.focus(p.segments);
    app.drawer.pocket(id);
    app.mapApi.fit(bboxOf(m, p.segments), { maxZoom: 14.5 });
  };
  app.selectDestination = (id) => app.drawer.destination(id);
  app.focusSegments = (idx) => { app.mapApi.focus(idx); if (idx.length) app.mapApi.fit(bboxOf(m, idx), { maxZoom: 14.5 }); };
  app.reset = () => {
    app.follow.stop();
    app.clearSelection();
    app.setMode("normal");
    for (const [k, on] of Object.entries(DEFAULT_LAYERS)) app.mapApi.setLayer(k, on);
    app.home();
  };

  // ---------------------------------------------------------------- chrome
  const rail = $("rail");
  for (const sec of SECTIONS) {
    const b = document.createElement("button");
    b.className = "tab";
    b.dataset.k = sec.key;
    b.setAttribute("aria-pressed", "false");
    b.title = sec.title;
    b.innerHTML = `${ICON[sec.key]}${sec.label}`;
    b.addEventListener("click", () => (app.panel === sec.key ? app.closePanel() : app.openPanel(sec.key)));
    rail.appendChild(b);
  }
  $("panelClose").addEventListener("click", () => app.closePanel());
  renderStatus($("status"), app);
  renderTimeline($("timeline"), app);
  app.follow = createFollow(app);
  $("followBtn").addEventListener("click", () => app.follow.run());
  app.judge = createJudge(app);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("judge").hidden) { app.follow.stop(); app.clearSelection(); }
  });

  stage("map-load");
  await app.mapApi.ready;
  stage("ui");
  app.openPanel("incident");
  $("loading").classList.add("gone");
  setTimeout(() => $("loading").remove(), 500);
  app.ready = true;
  stage("ready");
  window.removeEventListener("error", onBootError);
  window.removeEventListener("unhandledrejection", onBootError);
  if (params.get("judge") === "1") app.judge.start();
}

function fail(err) {
  if (window.BATO_ERROR) return;
  console.error(err);
  const l = $("loading");
  l.classList.remove("gone");
  l.innerHTML = `<div class="error-box"><b>The dashboard could not load the official result.</b><p>${esc(err.message)}</p>
    <p class="muted">Build the bundle with <code>python scripts/build_dashboard.py --manifest …</code>, then start
    <code>python scripts/serve_dashboard.py</code> and reload.</p></div>`;
  window.BATO_ERROR = `${String(err.message)} (bootstrap stage: ${window.BATO_STAGE || "start"})`;
}

boot().catch(fail);

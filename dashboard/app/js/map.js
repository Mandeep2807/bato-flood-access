// The hero map. MapLibre GL (vendored) with the bundle's own layers on a local base map: a dark background plus the
// DEM hillshade from the bundle. No online tile service is used (offline-safe, no key, no provider error tiles).
import { COLORS, roadsGeoJSON, settlementsGeoJSON, segmentsGeoJSON, aoiBounds, esc, settlementName, roadName,
  ROAD_LABEL, ACCESS_LABEL, kmOrM } from "./data.js";

const EMPTY = { type: "FeatureCollection", features: [] };
const UNOBS_QUIET = "#d3a456";          // unobservable on the first screen: amber, but quieter than the evidence
const LOAD_TIMEOUT_MS = 20000;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const LAYERS = [
  { key: "flood", label: "Flood evidence (Step 4C)", swatch: COLORS.water },
  { key: "blocked", label: "Blocked roads", swatch: COLORS.blocked },
  { key: "open", label: "Open roads (observed)", swatch: COLORS.open },
  { key: "unobservable", label: "Unobservable roads", swatch: "hatch" },
  { key: "notassessed", label: "Not-assessed terrain", swatch: "hatch" },
  { key: "settlements", label: "Settlements", swatch: "#d9e3e7" },
  { key: "hospitals", label: "Hospitals", swatch: "#f4f6f7" },
  { key: "exits", label: "Road exits", swatch: "#9fb2bb" },
  { key: "routes", label: "Selected route", swatch: "#ffffff" },
  { key: "aoi", label: "AOI outline", swatch: "#8fa3ad" },
  { key: "terrain", label: "Terrain shading (DEM)", swatch: "#6e7c84" },
];
// First screen: flood evidence, blocked and observed-open roads, settlements. Detail layers stay one click away.
export const DEFAULT_LAYERS = Object.fromEntries(LAYERS.map((l) => [l.key, !["notassessed", "exits"].includes(l.key)]));
const MAP_BG = "#0f1a25";
const CASING = "#08111a";

export function createMap(container, m, handlers) {
  const b = m.b;
  const layers = { ...DEFAULT_LAYERS };
  let mode = "normal";
  const sel = { segs: [], settlement: null };
  const markers = { hospitals: [], exits: [], towns: [] };

  // The style holds no remote source. MapLibre's "load" waits for every source of the style (a stalled tile server
  // would block the official result), and anonymous CARTO raster tiles are now served as HTTP 200 images watermarked
  // "API KEY REQUIRED", which no error handler can catch. The base map is therefore local: this background and the
  // DEM hillshade image from the bundle.
  const style = {
    version: 8,
    sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": MAP_BG } }],
  };

  const map = new maplibregl.Map({
    container, style, bounds: aoiBounds(m), fitBoundsOptions: { padding: { top: 140, bottom: 60, left: 400, right: 60 } },
    attributionControl: { compact: false }, maxPitch: 0, dragRotate: false, pitchWithRotate: false,
  });
  map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.ScaleControl({ unit: "metric", maxWidth: 110 }), "bottom-right");
  map.on("error", (e) => console.warn("map:", e && e.error ? e.error.message : e));

  // Fail visibly instead of hanging: an exception in build() rejects (it would otherwise be swallowed as an
  // uncaught error inside MapLibre's event dispatch, leaving the promise pending), and so does a first render
  // that never completes (WebGL lost or stalled).
  const ready = new Promise((resolve, reject) => {
    const fail = (err) => { clearTimeout(timer); reject(err instanceof Error ? err : new Error(String(err))); };
    const timer = setTimeout(() => fail(new Error(`The map did not finish its first render within ${LOAD_TIMEOUT_MS / 1000} s ` +
      "(WebGL unavailable or stalled in this browser).")), LOAD_TIMEOUT_MS);
    map.getCanvas().addEventListener("webglcontextlost", () => fail(new Error("The browser lost the WebGL context while loading the map.")), { once: true });
    map.once("load", () => {
      clearTimeout(timer);
      try { build(); resolve(); } catch (err) { fail(err); }
    });
  });

  function addImage(id, ov, opacity, before) {
    if (!ov) return;
    map.addSource(id, { type: "image", url: m.root + ov.file, coordinates: ov.corners });
    map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": opacity, "raster-fade-duration": 0,
      "raster-resampling": id === "hillshade" ? "linear" : "nearest" } }, before);
  }

  function build() {
    const ov = b.overlays || {};
    addImage("hillshade", ov.hillshade, 0.7);
    addImage("flood", ov.flood, 1);
    addImage("notassessed", ov.not_assessed, 0.0);
    map.addSource("aoi", { type: "geojson", data: { type: "Feature", geometry: { type: "Polygon", coordinates: b.aoi } } });
    map.addLayer({ id: "aoi", type: "line", source: "aoi", paint: { "line-color": "#8fa3ad", "line-width": 1.1,
      "line-dasharray": [4, 3], "line-opacity": 0.7 } });

    map.addSource("roads", { type: "geojson", data: roadsGeoJSON(m), tolerance: 0.25,
      attribution: "Roads, places, hospitals: © OpenStreetMap contributors (pre-event snapshot via ohsome)" });
    const w = (a, b2, c) => ["interpolate", ["exponential", 1.6], ["zoom"], 8, a, 12, b2, 16, c];
    map.addLayer({ id: "road-casing", type: "line", source: "roads", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": CASING, "line-width": w(1.8, 3, 6.5), "line-opacity": 0.6 } });
    map.addLayer({ id: "road-open", type: "line", source: "roads", filter: ["==", ["get", "s"], 2],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#5ad1bd", "line-width": w(0.6, 1.3, 3.2), "line-opacity": 0.9 } });
    map.addLayer({ id: "road-unobs", type: "line", source: "roads", filter: ["==", ["get", "s"], 1],
      layout: { "line-join": "round" },
      paint: { "line-color": UNOBS_QUIET, "line-width": w(0.6, 1.3, 3.2), "line-opacity": 0.6,
        "line-dasharray": [2, 1.6] } });
    map.addLayer({ id: "road-blocked-glow", type: "line", source: "roads", filter: ["==", ["get", "s"], 0],
      layout: { "line-cap": "round" },
      paint: { "line-color": COLORS.blocked, "line-width": w(8, 13, 24), "line-blur": w(4, 6, 10), "line-opacity": 0.45 } });
    map.addLayer({ id: "road-blocked", type: "line", source: "roads", filter: ["==", ["get", "s"], 0],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ff5c52", "line-width": w(2.6, 4, 7.5) } });
    map.addLayer({ id: "road-select", type: "line", source: "roads", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": w(4, 6, 11), "line-blur": 1.5,
        "line-opacity": ["case", ["boolean", ["feature-state", "sel"], false], 0.85, 0] } }, "road-blocked");
    map.addLayer({ id: "road-hit", type: "line", source: "roads", paint: { "line-color": "#000", "line-opacity": 0,
      "line-width": w(8, 12, 18) } });

    map.addSource("focus", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "focus-glow", type: "line", source: "focus", layout: { "line-cap": "round" },
      paint: { "line-color": ["case", ["==", ["get", "s"], 0], COLORS.blocked, COLORS.unobservable], "line-width": w(10, 16, 26),
        "line-blur": 8, "line-opacity": 0.55 } }, "road-blocked");

    map.addSource("trace", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "trace-casing", type: "line", source: "trace", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": CASING, "line-width": w(5, 8, 13), "line-opacity": 0.9 } });
    map.addLayer({ id: "trace-solid", type: "line", source: "trace", filter: ["!=", ["get", "s"], 1],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ["match", ["get", "s"], 0, COLORS.blocked, "#7ff0dc"], "line-width": w(2.4, 4, 7) } });
    map.addLayer({ id: "trace-unobs", type: "line", source: "trace", filter: ["==", ["get", "s"], 1],
      paint: { "line-color": "#ffc35c", "line-width": w(2.4, 4, 7), "line-dasharray": [1.4, 1] } });

    map.addSource("settlements", { type: "geojson", data: settlementsGeoJSON(m) });
    const accessColor = ["match", ["get", "a"], "CONNECTED", COLORS.CONNECTED, "POSSIBLE_CUTOFF", COLORS.POSSIBLE_CUTOFF,
      "CONFIRMED_CUTOFF", COLORS.CONFIRMED_CUTOFF, COLORS.NO_PRE_EVENT_ACCESS];
    const rad = (add) => ["interpolate", ["linear"], ["zoom"], 8, ["+", 1.8 + add, ["*", 0.16, ["sqrt", ["get", "n"]]]],
      12, ["+", 3 + add, ["*", 0.3, ["sqrt", ["get", "n"]]]], 15, ["+", 5 + add, ["*", 0.6, ["sqrt", ["get", "n"]]]]];
    const radius = rad(0);
    map.addLayer({ id: "settle-halo", type: "circle", source: "settlements",
      paint: { "circle-radius": rad(7), "circle-color": "#ffffff", "circle-blur": 0.6,
        "circle-opacity": ["case", ["boolean", ["feature-state", "sel"], false], 0.55, 0] } });
    map.addLayer({ id: "settle", type: "circle", source: "settlements",
      paint: { "circle-radius": radius, "circle-color": ["case", ["==", ["get", "a"], "NO_PRE_EVENT_ACCESS"], MAP_BG,
        ["==", ["get", "a"], "POSSIBLE_CUTOFF"], "#2b2112", accessColor],
        "circle-stroke-color": accessColor, "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 9, 1, 14, 1.8],
        "circle-opacity": 1, "circle-stroke-opacity": 1 } });

    map.addSource("chip", { type: "image", url: transparentPixel(), coordinates: [[0, 1], [1, 1], [1, 0], [0, 0]] });
    map.addLayer({ id: "chip", type: "raster", source: "chip", paint: { "raster-opacity": 0, "raster-opacity-transition": { duration: 600 } } },
      "road-casing");

    addMarkers();
    wire();
    apply();
    if (!REDUCED) pulse();
  }

  function addMarkers() {
    for (const d of b.destinations) {
      const el = document.createElement("button");
      el.setAttribute("aria-label", `${d.kind}: ${d.name || d.id}`);
      let key;
      if (d.kind === "hospital") {
        el.className = "marker-h"; key = "hospitals";
        el.innerHTML = '<svg viewBox="0 0 12 12"><path d="M4.5 1h3v3.5H11v3H7.5V11h-3V7.5H1v-3h3.5z"/></svg>';
      } else if (d.kind === "analysis_exit") { el.className = "marker-wrap"; el.innerHTML = '<span class="marker-x"></span>'; key = "exits"; }
      else { el.className = "marker-wrap"; el.innerHTML = '<span class="marker-t"></span>'; key = "towns"; }
      el.addEventListener("click", (ev) => { ev.stopPropagation(); handlers.destination(d.id); });
      el.addEventListener("mouseenter", () => tag(`${d.kind === "analysis_exit" ? "Road exit" : d.kind === "hospital" ? "Hospital" : "Town"}${d.name ? " · " + esc(d.name) : ""}${d.snapped ? "" : " (no road within reach)"}`));
      el.addEventListener("mouseleave", () => tag(null));
      markers[key].push(new maplibregl.Marker({ element: el }).setLngLat([d.lon, d.lat]).addTo(map));
    }
  }

  const hoverTag = document.getElementById("hoverTag");
  let mouse = [0, 0];
  function tag(html) {
    if (!html) { hoverTag.hidden = true; return; }
    hoverTag.innerHTML = html;
    hoverTag.style.left = `${mouse[0]}px`;
    hoverTag.style.top = `${mouse[1]}px`;
    hoverTag.hidden = false;
  }

  function pick(point) {
    const box = [[point.x - 6, point.y - 6], [point.x + 6, point.y + 6]];
    const layersHit = ["settle", "road-hit"].filter((id) => map.getLayer(id));
    const feats = map.queryRenderedFeatures(box, { layers: layersHit });
    const settle = feats.find((f) => f.layer.id === "settle");
    if (settle && layers.settlements) return { kind: "settlement", id: settle.properties.id };
    const roads = feats.filter((f) => f.layer.id === "road-hit").filter((f) => visibleRoad(f.properties.s));
    if (!roads.length) return null;
    roads.sort((a, c) => a.properties.s - c.properties.s);
    return { kind: "segment", id: roads[0].properties.i };
  }
  const visibleRoad = (s) => (s === 0 && layers.blocked) || (s === 1 && layers.unobservable) || (s === 2 && layers.open);

  function wire() {
    map.on("mousemove", (e) => {
      mouse = [e.point.x, e.point.y];
      const p = pick(e.point);
      map.getCanvas().style.cursor = p ? "pointer" : "";
      if (!p) return tag(null);
      if (p.kind === "settlement") {
        const s = m.byId.get(p.id);
        tag(`<b>${esc(settlementName(s))}</b> · ${ACCESS_LABEL[s.access_state]}`);
      } else {
        const info = m.segInfo(p.id);
        tag(`<b>${ROAD_LABEL[info.state]}</b> · ${esc(roadName(info))} · ${kmOrM(info.length)}`);
      }
    });
    map.on("mouseout", () => tag(null));
    map.on("click", (e) => {
      const p = pick(e.point);
      if (!p) return handlers.empty();
      if (p.kind === "settlement") handlers.settlement(p.id);
      else handlers.segment(p.id);
    });
  }

  function pulse() {
    const t0 = performance.now();
    const step = (t) => {
      if (map.getLayer("road-blocked-glow") && layers.blocked) {
        const k = 0.32 + 0.16 * Math.sin((t - t0) / 900);
        map.setPaintProperty("road-blocked-glow", "line-opacity", mode === "observability" ? k * 0.4 : k);
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function vis(id, on) { if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); }
  function paint(id, prop, v) { if (map.getLayer(id)) map.setPaintProperty(id, prop, v); }

  function apply() {
    vis("hillshade", layers.terrain);
    vis("flood", layers.flood);
    vis("aoi", layers.aoi);
    vis("road-casing", layers.open || layers.unobservable || layers.blocked);
    vis("road-open", layers.open);
    vis("road-unobs", layers.unobservable);
    vis("road-blocked", layers.blocked);
    vis("road-blocked-glow", layers.blocked);
    vis("settle", layers.settlements);
    vis("settle-halo", layers.settlements);
    for (const id of ["trace-casing", "trace-solid", "trace-unobs"]) vis(id, layers.routes);
    for (const [key, list] of Object.entries(markers)) {
      const on = key === "towns" ? layers.hospitals || layers.exits : layers[key];
      for (const mk of list) mk.getElement().style.display = on ? "" : "none";
    }
    const obs = mode === "observability", acc = mode === "access", ev = mode === "evidence";
    paint("road-casing", "line-opacity", obs ? 0.35 : 0.6);
    paint("road-open", "line-opacity", obs ? 0.1 : acc ? 0.3 : ev ? 0.25 : 0.9);
    paint("road-unobs", "line-opacity", obs ? 0.95 : acc ? 0.45 : ev ? 0.2 : 0.6);
    paint("road-unobs", "line-color", obs ? ["match", ["get", "g"], 0, COLORS.radar_geometry, 1, COLORS.terrain_rule,
      2, COLORS.data_gap, COLORS.unknown] : UNOBS_QUIET);
    paint("road-unobs", "line-width", obs ? ["interpolate", ["exponential", 1.6], ["zoom"], 8, 1.1, 12, 2.4, 16, 4.6]
      : ["interpolate", ["exponential", 1.6], ["zoom"], 8, 0.6, 12, 1.3, 16, 3.2]);
    paint("road-blocked", "line-opacity", obs ? 0.35 : 1);
    paint("flood", "raster-opacity", obs ? 0.35 : 1);
    const showNA = layers.notassessed || obs;
    vis("notassessed", showNA);
    paint("notassessed", "raster-opacity", obs ? 0.95 : 0.7);
    // settlements: present but quiet on the first screen; full strength in the Access and Observability modes
    const quiet = ["match", ["get", "a"], "POSSIBLE_CUTOFF", 0.7, "NO_PRE_EVENT_ACCESS", 0.45, 1];
    paint("settle", "circle-opacity", obs ? ["case", ["==", ["get", "a"], "POSSIBLE_CUTOFF"], 1, 0.25] : acc ? 1 : quiet);
    paint("settle", "circle-stroke-opacity", obs ? ["case", ["==", ["get", "a"], "POSSIBLE_CUTOFF"], 1, 0.25] : acc ? 1 : quiet);
  }

  function setSegSelection(idx) {
    for (const i of sel.segs) map.setFeatureState({ source: "roads", id: i }, { sel: false });
    sel.segs = idx || [];
    for (const i of sel.segs) map.setFeatureState({ source: "roads", id: i }, { sel: true });
  }

  function setSettlementSelection(id) {
    if (sel.settlement != null) map.setFeatureState({ source: "settlements", id: sel.settlement }, { sel: false });
    sel.settlement = id == null ? null : m.settlements.findIndex((x) => x.id === id);
    if (sel.settlement != null && sel.settlement >= 0) map.setFeatureState({ source: "settlements", id: sel.settlement }, { sel: true });
  }

  return {
    map, ready,
    get mode() { return mode; },
    layers,
    setMode(next) { mode = next; apply(); },
    setLayer(key, on) { layers[key] = on; apply(); },
    selectSegments: setSegSelection,
    selectSettlement: setSettlementSelection,
    focus(idx) { map.getSource("focus")?.setData(idx && idx.length ? segmentsGeoJSON(m, idx) : EMPTY); },
    trace(edges) { map.getSource("trace")?.setData(edges && edges.length ? segmentsGeoJSON(m, edges) : EMPTY); },
    fit(bounds, opts = {}) {
      const panelOpen = !document.getElementById("panel").hidden;
      const drawerOpen = !document.getElementById("drawer").hidden;
      const pad = { top: 150, bottom: 150, left: panelOpen ? 410 : 60, right: drawerOpen ? 430 : 70 };
      map.fitBounds(bounds, { padding: pad, maxZoom: 15, duration: REDUCED ? 0 : 1400, ...opts });
    },
    flyTo(lon, lat, zoom, opts = {}) { map.flyTo({ center: [lon, lat], zoom, duration: REDUCED ? 0 : 1600, essential: true, ...opts }); },
    zoom(delta) { map.easeTo({ zoom: map.getZoom() + delta, duration: REDUCED ? 0 : 300 }); },
    home() { map.fitBounds(aoiBounds(m), { padding: { top: 140, bottom: 60, left: 400, right: 60 }, duration: REDUCED ? 0 : 1200 }); },
    chip(chip, file, opacity = 0.95) {
      const src = map.getSource("chip");
      if (!src) return;
      if (!chip) { paint("chip", "raster-opacity", 0); return; }
      src.updateImage({ url: m.root + file, coordinates: chip.corners });
      paint("chip", "raster-opacity", opacity);
    },
  };
}

function transparentPixel() {
  return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
}

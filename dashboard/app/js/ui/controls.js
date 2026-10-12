import { LAYERS } from "../map.js";
import { ICON } from "./dom.js";

export const MODES = [
  { key: "normal", label: "Normal" },
  { key: "observability", label: "Observability" },
  { key: "access", label: "Access" },
  { key: "evidence", label: "Evidence" },
];

export function renderControls(root, app) {
  root.innerHTML = `
    <div class="layers" id="layersPop" hidden></div>
    <div class="ctl-row">
      <button class="icon-btn" id="radarBtnMap" aria-pressed="false" title="Dim observed roads and show what the rules could not decide">${ICON.eyeoff}What radar can't confirm</button>
      <button class="icon-btn" id="layersBtn" aria-expanded="false" title="Layers">${ICON.layers}Layers</button>
      <button class="icon-btn" id="homeBtn" title="Back to the whole AOI" aria-label="Back to the whole AOI">${ICON.home}</button>
      <button class="icon-btn" id="zin" title="Zoom in" aria-label="Zoom in">+</button>
      <button class="icon-btn" id="zout" title="Zoom out" aria-label="Zoom out">−</button>
    </div>
    <div class="modes" role="group" aria-label="Map mode">${MODES.map((md) => `<button data-mode="${md.key}" aria-pressed="${md.key === "normal"}">${md.label}</button>`).join("")}</div>`;
  const pop = root.querySelector("#layersPop");
  const layersBtn = root.querySelector("#layersBtn");
  layersBtn.addEventListener("click", () => {
    pop.hidden = !pop.hidden;
    layersBtn.setAttribute("aria-expanded", String(!pop.hidden));
    if (!pop.hidden) fillLayers();
  });
  function fillLayers() {
    pop.innerHTML = "";
    for (const l of LAYERS) {
      const sw = l.swatch === "hatch" ? "background:var(--hatch),#2b2112" : `background:${l.swatch}`;
      const lab = document.createElement("label");
      lab.innerHTML = `<input type="checkbox" ${app.mapApi.layers[l.key] ? "checked" : ""} data-layer="${l.key}"><span class="sw" style="${sw}"></span>${l.label}`;
      lab.querySelector("input").addEventListener("change", (e) => app.setLayer(l.key, e.target.checked));
      pop.appendChild(lab);
    }
  }
  root.querySelector("#homeBtn").addEventListener("click", () => app.home());
  root.querySelector("#zin").addEventListener("click", () => app.mapApi.zoom(1));
  root.querySelector("#zout").addEventListener("click", () => app.mapApi.zoom(-1));
  root.querySelector("#radarBtnMap").addEventListener("click", (e) => app.setRadarMode(e.currentTarget.getAttribute("aria-pressed") !== "true"));
  root.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => app.setMode(b.dataset.mode)));
  return {
    sync(mode) {
      root.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
      root.querySelector("#radarBtnMap").setAttribute("aria-pressed", String(mode === "observability"));
      if (!pop.hidden) fillLayers();
    },
  };
}

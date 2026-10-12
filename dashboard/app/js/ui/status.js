import { km1, int, pct, ACCESS } from "../data.js";
import { el, ICON } from "./dom.js";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function shareBar(m, lit) {
  const s = m.share;
  const part = (k, cls) => `<i class="${cls}${lit === k ? " on" : ""}" style="flex:${Math.max(s[k], 0.002)}"></i>`;
  return `<div class="share" aria-hidden="true">${part("blocked", "b")}${part("open", "o")}${part("unobservable", "u")}</div>`;
}

export function renderStatus(root, app) {
  const m = app.m;
  const s = m.s;
  const segs = s.road_segments_by_state;
  const mainReason = Object.entries(s.unobservable_km_by_main_reason).sort((a, b) => b[1] - a[1])[0];
  const reasonText = { radar_geometry: "mostly layover / shadow", terrain_rule: "mostly steep-slope rule",
    data_gap: "mostly data gaps", unknown: "" }[mainReason ? mainReason[0] : "unknown"];
  const c = m.counts;
  const cells = [];
  for (const k of ACCESS) for (let i = 0; i < c[k]; i++) cells.push(`<i class="u-${k}"></i>`);

  root.innerHTML = "";
  const modules = [
    el(`<button class="module" data-k="blocked" title="Show blocked roads">
      <div class="head">${ICON.blocked}<span class="micro" style="color:inherit">Blocked road</span><span class="live" title="Critical state"></span></div>
      <div class="value"><b class="num" data-count="${m.km.blocked}" data-dec="1">${km1(m.km.blocked)}</b><small>km</small></div>
      <div class="desc">${int(segs.blocked)} segments in ${int(s.blocked_stretches)} stretches</div>
      ${shareBar(m, "blocked")}</button>`),
    el(`<button class="module" data-k="open" title="Observed open roads">
      <div class="head">${ICON.open}<span class="micro" style="color:inherit">Observed open</span></div>
      <div class="value"><b class="num" data-count="${m.km.open}" data-dec="1">${km1(m.km.open)}</b><small>km</small></div>
      <div class="desc">${pct(m.share.open)} of mapped road · no flood signal seen</div>
      ${shareBar(m, "open")}</button>`),
    el(`<button class="module" data-k="unobservable" title="What radar can't confirm">
      <div class="head">${ICON.unobservable}<span class="micro" style="color:inherit">Unobservable road</span></div>
      <div class="value"><b class="num" data-count="${m.km.unobservable}" data-dec="1">${km1(m.km.unobservable)}</b><small>km</small></div>
      <div class="desc">${pct(m.share.unobservable)} of mapped road · ${reasonText}</div>
      ${shareBar(m, "unobservable")}</button>`),
    el(`<button class="module" data-k="access" title="Settlement access">
      <div class="head">${ICON.settlements}<span class="micro" style="color:inherit">Settlement access</span>
        <span class="faint" style="margin-left:auto;font-size:11px">${int(s.settlements_total)} settlements</span></div>
      <div class="access-line">
        <span class="c"><b class="num" data-count="${c.CONNECTED}">${int(c.CONNECTED)}</b>connected</span>
        <span class="p"><b class="num" data-count="${c.POSSIBLE_CUTOFF}">${int(c.POSSIBLE_CUTOFF)}</b>unverified</span>
        <span class="x"><b class="num" data-count="${c.CONFIRMED_CUTOFF}">${int(c.CONFIRMED_CUTOFF)}</b>cut off</span>
        <span><b class="num" style="color:var(--text-2)" data-count="${c.NO_PRE_EVENT_ACCESS}">${int(c.NO_PRE_EVENT_ACCESS)}</b>no road</span>
      </div>
      <div class="units" aria-hidden="true" title="One cell per settlement">${cells.join("")}</div></button>`),
  ];
  for (const mod of modules) root.appendChild(mod);
  root.querySelector('[data-k="blocked"]').addEventListener("click", () => app.openPanel("operations"));
  root.querySelector('[data-k="open"]').addEventListener("click", () => app.openPanel("observability"));
  root.querySelector('[data-k="unobservable"]').addEventListener("click", () => { app.openPanel("observability"); app.setRadarMode(true); });
  root.querySelector('[data-k="access"]').addEventListener("click", () => app.openPanel("access"));
  countUp(root);
}

function countUp(root) {
  if (REDUCED) return;
  const nodes = [...root.querySelectorAll("[data-count]")];
  const t0 = performance.now();
  const dur = 1100;
  const fmt = (v, d) => (d ? km1(v) : int(Math.round(v)));
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    for (const n of nodes) n.textContent = fmt(Number(n.dataset.count) * e, n.dataset.dec);
    if (k < 1) requestAnimationFrame(step);
    else for (const n of nodes) n.textContent = fmt(Number(n.dataset.count), n.dataset.dec);
  };
  requestAnimationFrame(step);
}

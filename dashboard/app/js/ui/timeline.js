// Acquisition timeline: equal spacing between events, with the real gap in days written on each link.
import { dayShort, day, utcTime, esc } from "../data.js";
import { el } from "./dom.js";

export function renderTimeline(root, app) {
  const m = app.m;
  const inc = m.b.incident;
  const nodes = m.b.timeline.map((t) => ({ role: t.role, date: t.acquired, label: t.role.replace("PRE", "PRE-"), t }));
  nodes.push({ role: "EVENT", date: inc.event_date, label: "Event" });
  if (inc.osm_snapshot_date) nodes.push({ role: "OSM", date: inc.osm_snapshot_date, label: "OSM" });
  nodes.sort((a, b) => a.date.slice(0, 10).localeCompare(b.date.slice(0, 10)) || order(a.role) - order(b.role));
  const n = nodes.length;
  const x = (i) => `${6 + (i / (n - 1)) * 88}%`;
  root.innerHTML = '<div class="axis"></div>';
  const spanStart = nodes.findIndex((d) => d.role === "PRE2");
  const spanEnd = nodes.findIndex((d) => d.role === "POST");
  if (spanStart >= 0 && spanEnd > spanStart) {
    root.appendChild(el(`<div class="span" style="left:${x(spanStart)};width:${(spanEnd - spanStart) / (n - 1) * 88}%"></div>`));
  }
  for (let i = 1; i < n; i++) {
    const d = Math.round((Date.parse(nodes[i].date.slice(0, 10)) - Date.parse(nodes[i - 1].date.slice(0, 10))) / 86400000);
    root.appendChild(el(`<span class="faint" style="position:absolute;top:0;left:calc((${x(i - 1)} + ${x(i)}) / 2);transform:translateX(-50%);font:500 9px/1 var(--cond)">${d} d</span>`));
  }
  nodes.forEach((d, i) => {
    const b = el(`<button class="tl-node" data-role="${d.role}" style="left:${x(i)}" aria-expanded="false"
      title="${esc(d.label)} · ${day(d.date)}"><i></i><span>${d.label} ${dayShort(d.date)}</span></button>`);
    b.addEventListener("click", (ev) => { ev.stopPropagation(); toggle(b, d, app); });
    root.appendChild(b);
  });
  document.addEventListener("click", () => closePop());
}

function order(role) { return ["PRE1", "PRE2", "OSM", "EVENT", "POST"].indexOf(role); }

let pop = null;
function closePop() {
  if (pop) { pop.remove(); pop = null; }
  document.querySelectorAll(".tl-node[aria-expanded='true']").forEach((n) => n.setAttribute("aria-expanded", "false"));
}

function toggle(btn, d, app) {
  const open = btn.getAttribute("aria-expanded") === "true";
  closePop();
  if (open) return;
  const m = app.m;
  const inc = m.b.incident;
  let body;
  if (d.t) {
    body = `<h4>${d.label} · Sentinel-1</h4><div class="muted">${esc(d.t.contributes)}</div>
      <dl><dt>Acquired</dt><dd>${day(d.t.acquired)} ${utcTime(d.t.acquired)}</dd>
      <dt>Role</dt><dd>${d.role === "POST" ? "after the event" : "before the event"} (${d.t.days_from_event > 0 ? "+" : ""}${d.t.days_from_event} days)</dd>
      <dt>Orbit</dt><dd>${esc(d.t.orbit_direction || "–")}, relative orbit ${esc(d.t.relative_orbit ?? "–")}</dd>
      <dt>Product</dt><dd class="hash">${esc(d.t.product)}</dd></dl>`;
  } else if (d.role === "EVENT") {
    body = `<h4>Event · ${day(d.date)}</h4><div class="muted">The date the analysis is built around. No image is taken on it:
      the closest observations are PRE-2 before and POST after.</div>`;
  } else {
    body = `<h4>Pre-event OSM snapshot · ${day(d.date)}</h4><div class="muted">Roads, places, hospitals and buildings exactly as
      OpenStreetMap stood the day before the event (${esc(inc.osm_service || "ohsome")}). No post-event edits are used.</div>`;
  }
  pop = el(`<div class="popover" role="dialog">${body}</div>`);
  pop.addEventListener("click", (e) => e.stopPropagation());
  document.body.appendChild(pop);
  const r = btn.getBoundingClientRect();
  pop.style.top = `${r.bottom + 8}px`;
  pop.style.left = `${Math.min(window.innerWidth - 316, Math.max(8, r.left + r.width / 2 - 150))}px`;
  btn.setAttribute("aria-expanded", "true");
  btn.closest(".timeline").classList.add("active");
}

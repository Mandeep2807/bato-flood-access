// Left panel sections. Every figure is read from the bundle; copy states what each number is.
import { km1, int, pct, day, esc, kmOrM, plural, ACCESS, ACCESS_LABEL, COLORS, settlementName, destinationLabel,
  roadName, unverifiedRouteStats } from "../data.js";
import { LAYERS } from "../map.js";
import { badge } from "./dom.js";

export const SECTIONS = [
  { key: "incident", label: "Incident", title: "Incident" },
  { key: "map", label: "Map", title: "Map layers" },
  { key: "access", label: "Access", title: "Access before vs after" },
  { key: "observability", label: "Observe", title: "Radar observability" },
  { key: "evidence", label: "Evidence", title: "Evidence chain" },
  { key: "operations", label: "Ops", title: "Operations" },
  { key: "methodology", label: "Method", title: "Methodology" },
];

const MAIN_REASON_SENTENCE = {
  terrain_rule: "Most of it is unobservable because of the Step 4C terrain rule, not because the radar is blind there: " +
    "on slopes above the limit the open-water flood rule is not applied, so the road state cannot be decided.",
  radar_geometry: "Most of it lies in radar layover or shadow: the satellite cannot see the ground there.",
  data_gap: "Most of it lies where the radar image or the DEM has no valid data.",
  unknown: "",
};

export function render(key, body, app) {
  const fn = { incident, map: mapPanel, access, observability, evidence, operations, methodology }[key];
  body.innerHTML = "";
  fn(body, app);
  body.scrollTop = 0;
}

// ---------------------------------------------------------------- incident
function incident(body, app) {
  const m = app.m, s = m.s, inc = m.b.incident, km = m.km, c = m.counts;
  const observed = (km.blocked + km.open) / m.total;
  const pre = s.reachable_settlements.pre_event;
  const mainReason = topReason(m);
  const alone = s.blocked_stretches_reconnecting_alone;
  body.innerHTML = `
    <div class="section">
      <p class="lede">Map what radar can confirm. Expose what it cannot. Trace access consequences.</p>
      <p class="p">${esc(inc.title)}, event ${day(inc.event_date)}. Sentinel-1 compared the ${day(m.dates.POST?.acquired)} pass
        with ${day(m.dates.PRE2?.acquired)} and ${day(m.dates.PRE1?.acquired)} on one orbit track. Roads, places and
        destinations are OpenStreetMap as it stood on ${day(inc.osm_snapshot_date)}.</p>
    </div>
    <div class="section">
      <span class="micro">What can BATO confirm?</span>
      <div class="channels">
        <div class="channel"><span class="name">Radar observation</span><span class="level" style="color:var(--observed)">Measured</span>
          <div class="bar"><i style="width:${observed * 100}%;background:var(--observed)"></i><i style="flex:1;background:var(--hatch),#2b2112"></i></div>
          <span class="note">${pct(observed)} of mapped road length could be assessed from the backscatter change.</span></div>
        <div class="channel"><span class="name">Road impact</span><span class="level" style="color:var(--unconfirmed)">Provisional</span>
          <div class="bar"><i style="width:${m.share.blocked * 100}%;min-width:2px;background:var(--blocked)"></i><i style="width:${m.share.open * 100}%;background:var(--observed)"></i><i style="flex:1;background:var(--hatch),#2b2112"></i></div>
          <span class="note">Rule output on provisional thresholds. Not validated against any reference map.</span></div>
        <div class="channel"><span class="name">Settlement access</span><span class="level" style="color:var(--unconfirmed)">Partial</span>
          <div class="bar"><i style="width:${pre ? (c.CONNECTED / pre) * 100 : 0}%;min-width:2px;background:var(--ok)"></i><i style="width:${pre ? (c.POSSIBLE_CUTOFF / pre) * 100 : 0}%;background:var(--hatch),#2b2112"></i><i style="flex:1;background:var(--blocked);opacity:${c.CONFIRMED_CUTOFF ? 1 : 0}"></i></div>
          <span class="note">${int(c.CONNECTED)} of ${int(pre)} settlements with pre-event road access are verified on observed roads.</span></div>
      </div>
    </div>
    <div class="section ledger">
      <div class="ok"><h4>Confirmed by the data</h4><ul>
        <li>${km1(km.blocked)} km of road shows new open-water-like darkening (${plural(s.road_segments_by_state.blocked, "segment", "segments")}, ${plural(s.blocked_stretches, "stretch", "stretches")}).</li>
        <li>${km1(km.open)} km of road was observed with no flood signal.</li>
        <li>${int(c.CONNECTED)} settlement${c.CONNECTED === 1 ? "" : "s"} reach${c.CONNECTED === 1 ? "es" : ""} a destination on observed-open road.</li>
        <li>${c.CONFIRMED_CUTOFF ? `${plural(c.CONFIRMED_CUTOFF, "settlement has", "settlements have")} no route even if every unobservable road is passable.` : "No settlement is confirmed cut off."}</li>
        <li>${int(alone)} of ${plural(s.blocked_stretches, "blocked stretch", "blocked stretches")} reconnect${alone === 1 ? "s" : ""} a settlement when cleared on ${alone === 1 ? "its" : "their"} own.</li>
      </ul></div>
      <div class="un"><h4>Unverified: radar cannot establish</h4><ul>
        <li>${km1(km.unobservable)} km of road could not be assessed. ${esc(MAIN_REASON_SENTENCE[mainReason] || "")}</li>
        <li>${plural(c.POSSIBLE_CUTOFF, "settlement reaches", "settlements reach")} a destination only if unobservable road is passable.</li>
      </ul></div>
      <div class="no"><h4>Not claimed</h4><ul>
        <li>Passability. Open means no flood signal on observed road, not that a vehicle can get through.</li>
        <li>Population. Building counts are a proxy for settlement size, not people.</li>
        <li>Landslide, debris or structural damage. The experimental debris detector was rejected.</li>
      </ul></div>
    </div>
    <div class="section"><span class="micro">Incident record</span>
      <dl class="facts">
        <dt>AOI</dt><dd>${esc(inc.aoi_id)}</dd><dt>Candidate</dt><dd>${esc(inc.candidate_id)}</dd>
        <dt>Track</dt><dd>${esc(inc.orbit_direction || "")} · relative orbit ${esc(inc.relative_orbit ?? "–")}</dd>
        <dt>OSM snapshot</dt><dd>${day(inc.osm_snapshot_date)} (pre-event)</dd>
        <dt>Road rule</dt><dd>${esc(inc.road_rule)} (official, frozen)</dd>
        <dt>Validation</dt><dd>${esc(inc.validation_note)}</dd>
      </dl></div>`;
}

function topReason(m) {
  const e = Object.entries(m.s.unobservable_km_by_main_reason).sort((a, b) => b[1] - a[1])[0];
  return e && e[1] > 0 ? e[0] : "unknown";
}

// ---------------------------------------------------------------- map layers
function mapPanel(body, app) {
  const mp = app.mapApi;
  const ov = app.m.b.overlays || {};
  const missing = { flood: !ov.flood, notassessed: !ov.not_assessed, terrain: !ov.hillshade };
  body.innerHTML = `<div class="section"><span class="micro">Layers</span><div class="layers" style="width:auto;border:0;padding:0;background:none" id="pl"></div></div>
    <div class="section"><span class="micro">Reading the map</span>
      <p class="p">Solid lines were observed by the radar rule; dashed amber lines could not be assessed. Red marks a
        contiguous run of new open-water-like darkening of at least the blocking length. Blue areas are Step 4C FLOOD pixels.
        Circles are settlements: filled green connected, amber ring unverified, red confirmed cut off, hollow grey no road before the event.
        The base map is local (dark background and DEM shading); no online tile service is used.</p>
      ${Object.values(missing).some(Boolean) ? `<p class="disclaimer">Not in this bundle: ${Object.entries(missing).filter(([, v]) => v).map(([k]) => k).join(", ")} (built without rasterio).</p>` : ""}
    </div>`;
  const box = body.querySelector("#pl");
  for (const l of LAYERS) {
    const sw = l.swatch === "hatch" ? "background:var(--hatch),#2b2112" : `background:${l.swatch}`;
    const lab = document.createElement("label");
    lab.innerHTML = `<input type="checkbox" ${mp.layers[l.key] ? "checked" : ""}><span class="sw" style="${sw}"></span>${l.label}`;
    lab.querySelector("input").addEventListener("change", (e) => app.setLayer(l.key, e.target.checked));
    box.appendChild(lab);
  }
}

// ---------------------------------------------------------------- access
function access(body, app) {
  const m = app.m, s = m.s, c = m.counts;
  const reach = s.reachable_settlements;
  const order = { CONNECTED: 0, POSSIBLE_CUTOFF: 1, CONFIRMED_CUTOFF: 2 };
  const pre = m.settlements.filter((x) => x.routes.pre_event.reachable).sort((a, b) => order[a.access_state] - order[b.access_state]);
  const cells = (fn) => pre.map((x) => `<i class="${fn(x)}"></i>`).join("");
  const stats = unverifiedRouteStats(m);
  body.innerHTML = `
    <div class="section">
      <p class="p">The same pre-event road network, three ways. One cell per settlement that had road access before the event.</p>
      <div class="tracks">
        <div class="track"><span class="lab">Pre-event<small>all roads</small></span><div class="cells">${cells(() => "")}</div><span class="n">${int(reach.pre_event)}</span></div>
        <div class="track"><span class="lab">Post · optimistic<small>unobservable passable</small></span><div class="cells">${cells((x) => x.access_state === "CONNECTED" ? "" : x.access_state === "POSSIBLE_CUTOFF" ? "h" : "x")}</div><span class="n">${int(reach.post_optimistic)}</span></div>
        <div class="track"><span class="lab">Post · pessimistic<small>unobservable closed</small></span><div class="cells">${cells((x) => x.access_state === "CONNECTED" ? "" : "x")}</div><span class="n">${int(reach.post_pessimistic)}</span></div>
      </div>
      <div class="track-note">${reach.pre_event === reach.post_optimistic ? "Blocked roads alone remove no settlement's access. " : `${plural(reach.pre_event - reach.post_optimistic, "settlement loses", "settlements lose")} every route once blocked roads are removed. `}
        Most access uncertainty comes from unobservable road segments: ${int(stats.withUnobs)} of ${int(stats.traced)} verified post-event routes cross at least one.
        This does not mean that ${int(c.POSSIBLE_CUTOFF)} settlement${c.POSSIBLE_CUTOFF === 1 ? " was" : "s were"} cut off.</div>
    </div>
    <div class="section"><span class="micro">Settlements</span>
      <div class="chips" id="fchips"></div>
      <input class="search" id="fsearch" placeholder="Find a settlement by name" aria-label="Find a settlement">
      <div class="list" id="flist"></div>
      <p class="disclaimer">Mapped buildings are a proxy for settlement size, never population.</p></div>`;
  const filters = new Set(["CONNECTED", "POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF"]);
  const chipBox = body.querySelector("#fchips");
  for (const k of ACCESS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.setAttribute("aria-pressed", filters.has(k) ? "true" : "false");
    const sw = k === "POSSIBLE_CUTOFF" ? "background:var(--hatch),#2b2112" : k === "NO_PRE_EVENT_ACCESS" ? "box-shadow:inset 0 0 0 1px var(--noaccess)" : `background:${COLORS[k]}`;
    b.innerHTML = `<i style="${sw}"></i>${ACCESS_LABEL[k]} <span class="faint">${int(c[k])}</span>`;
    b.addEventListener("click", () => {
      if (filters.has(k)) filters.delete(k); else filters.add(k);
      b.setAttribute("aria-pressed", filters.has(k) ? "true" : "false");
      list();
    });
    chipBox.appendChild(b);
  }
  const search = body.querySelector("#fsearch");
  search.addEventListener("input", list);
  function list() {
    const q = search.value.trim().toLowerCase();
    const rows = m.settlements.filter((x) => filters.has(x.access_state) && (!q || settlementName(x).toLowerCase().includes(q)))
      .sort((a, b) => order2(a) - order2(b) || b.buildings - a.buildings);
    const box = body.querySelector("#flist");
    box.innerHTML = rows.slice(0, 400).map((x) => settlementRow(m, x)).join("") ||
      '<p class="disclaimer">No settlement matches. Clear the search or turn a status filter back on.</p>';
    box.querySelectorAll("[data-sid]").forEach((r) => r.addEventListener("click", () => app.selectSettlement(r.dataset.sid, { fly: true })));
  }
  const order2 = (x) => ACCESS.indexOf(x.access_state);
  list();
}

function settlementRow(m, x) {
  const t = x.traces.post;
  const r = x.routes.post_optimistic.reachable ? x.routes.post_optimistic : x.routes.pre_event;
  const sub = x.access_state === "NO_PRE_EVENT_ACCESS" ? "No road within reach before the event"
    : `${r.length_m != null ? kmOrM(r.length_m) : "–"} to ${esc(destinationLabel(m, r.target_id))}` +
      (t ? ` · ${t.unobservable_segments} unobservable segment${t.unobservable_segments === 1 ? "" : "s"} on route` : "");
  return `<button class="row" data-sid="${esc(x.id)}"><span class="t">${esc(settlementName(x))}</span>
    <span class="v">${int(x.buildings)} bldg</span><span class="s">${badge(x.access_state, ACCESS_LABEL[x.access_state])} ${sub}</span></button>`;
}

// ---------------------------------------------------------------- observability
function observability(body, app) {
  const m = app.m, s = m.s;
  const by = s.unobservable_km_by_main_reason;
  const total = Object.values(by).reduce((a, b) => a + b, 0) || 1;
  const reasons = m.b.reason_groups.filter((g) => by[g.key] > 0);
  const top = topReason(m);
  const radarOn = app.mapApi.mode === "observability";
  body.innerHTML = `
    <div class="section">
      <p class="lede">Unobservable is not open, and not blocked. It means the satellite rule cannot decide.</p>
      <div class="neq"><div class="u">Unobservable<s>state undecided</s></div><div class="o">≠ open<s>not shown as safe</s></div><div class="b">≠ blocked<s>not shown as damaged</s></div></div>
      <button class="btn accent" id="radarBtn" aria-pressed="${radarOn}">What radar can't confirm</button>
    </div>
    <div class="section radar"><span class="micro">Why a road can be unobservable</span>${radarSVG()}
      <p class="p">The radar looks sideways. Slopes facing it fold onto each other (layover); slopes facing away are never lit
        (shadow). Steep ground is seen, but Step 4C does not apply its open-water flood rule there, so no road state can be decided.</p></div>
    <div class="section"><span class="micro">Unobservable road by main reason</span>
      <div class="reasons">${reasons.map((g) => `<div class="reason"><span class="lab">${esc(labelShort(g.key))}</span><span class="km">${km1(by[g.key])} km</span>
        <div class="bar"><i style="width:${(by[g.key] / total) * 100}%;background:${COLORS[g.key]}"></i></div>
        <span class="faint" style="grid-column:1/3;font-size:11.5px">${esc(g.label)}</span></div>`).join("")}</div>
      ${MAIN_REASON_SENTENCE[top] ? `<div class="callout">${esc(MAIN_REASON_SENTENCE[top])}</div>` : ""}
      <p class="disclaimer">Main reason = the reason covering most of a segment's not-assessed samples. Sampled length by exact Step 4C reason:
        ${Object.entries(s.unobservable_km_by_reason).map(([k, v]) => `${esc(k.replace("NOT_ASSESSED_", "").toLowerCase().replace(/_/g, " "))} ${km1(v)} km`).join(" · ")}.</p>
    </div>`;
  body.querySelector("#radarBtn").addEventListener("click", (e) => {
    const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
    app.setRadarMode(on);
    e.currentTarget.setAttribute("aria-pressed", String(on));
  });
}

function labelShort(k) {
  return { radar_geometry: "Radar geometry (layover, shadow)", terrain_rule: "Terrain rule (steep slope)",
    data_gap: "Data gap", unknown: "No reason recorded" }[k];
}

function radarSVG() {
  // side view: satellite left-high, terrain profile with a sensor-facing slope (layover), a back slope (shadow),
  // a steep but visible slope (terrain rule) and an observed valley floor.
  return `<svg viewBox="0 0 340 150" role="img" aria-label="Radar viewing geometry: observed floor, layover, shadow and steep-slope zones">
    <defs><pattern id="hz" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="5" fill="#e2a33a"/></pattern>
      <pattern id="ht" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="1.4" height="6" fill="#c49652"/></pattern></defs>
    <g fill="none"><rect x="8" y="8" width="18" height="9" rx="1.5" stroke="#d9e3e7"/><path d="M4 12.5h4M26 12.5h4" stroke="#d9e3e7"/>
      <line class="ray" x1="22" y1="20" x2="96" y2="118"/><line class="ray" x1="22" y1="20" x2="150" y2="56"/><line class="ray" x1="22" y1="20" x2="250" y2="92"/>
      <line x1="22" y1="20" x2="200" y2="40" stroke="rgba(227,234,237,.12)" stroke-dasharray="2 4"/></g>
    <path d="M0 132 L60 128 L110 120 L150 52 L178 64 L212 122 L232 124 L262 82 L300 70 L340 74 L340 150 L0 150 Z" fill="#16222a"/>
    <path d="M0 132 L60 128 L110 120" stroke="#3cc4ae" stroke-width="3" fill="none"/>
    <path d="M110 120 L150 52" stroke="url(#hz)" stroke-width="7" fill="none"/>
    <path d="M150 52 L178 64 L212 122" stroke="#56636b" stroke-width="3" fill="none"/>
    <path d="M178 64 L212 122 L232 124 L236 112 Z" fill="url(#hz)" opacity=".45"/>
    <path d="M232 124 L262 82" stroke="#c49652" stroke-width="2" fill="none" opacity=".6"/>
    <path d="M232 124 L262 82" stroke="url(#ht)" stroke-width="8" fill="none"/>
    <path d="M262 82 L300 70 L340 74" stroke="#3cc4ae" stroke-width="3" fill="none"/>
    <g font-family="Plex Cond, sans-serif" font-size="9.5" fill="#98a9b2">
      <text x="40" y="146">observed</text><text x="118" y="96" fill="#e2a33a" text-anchor="end">layover</text>
      <text x="190" y="140" fill="#e2a33a">shadow</text><text x="246" y="140" fill="#c49652">steep: rule not applied</text>
      <text x="292" y="64">observed</text><text x="34" y="16" fill="#d9e3e7">Sentinel-1</text></g>
  </svg>`;
}

// ---------------------------------------------------------------- evidence chain
function evidence(body, app) {
  const m = app.m;
  const integ = m.b.integrity;
  body.innerHTML = `
    <div class="section"><p class="p">Every claim on this screen traces back through these steps to three Sentinel-1
      acquisitions and one pre-event OSM snapshot. Open a step to see its source, rule and hashes.</p>
      <button class="btn teal" id="followBtn2">Follow the evidence on the map</button></div>
    <div class="section chain" id="chain">${m.b.chain.map((c, i) => `
      <div class="link" data-k="${c.key}" aria-expanded="false" role="button" tabindex="0">
        <span class="node">${i + 1}</span>
        <span><span class="t">${esc(c.title)}</span><br><span class="k">${esc(c.kind)}</span></span>
        <div class="detail">
          <div>${esc(c.status)}</div>
          ${c.rule ? `<div class="rule">${esc(c.rule)}</div>` : ""}
          ${c.sources ? `<div>Sources: ${c.sources.map(esc).join(", ")}</div>` : ""}
          ${c.thresholds && c.thresholds.length ? `<ul>${c.thresholds.map((t) => `<li>${esc(t.name)} = ${esc(t.value)} ${esc(t.unit || "")}${t.provisional ? " (provisional)" : ""}</li>`).join("")}</ul>` : ""}
          ${Object.entries(c.hashes || {}).filter(([, v]) => v).map(([k, v]) => `<div class="hash">${esc(k)} sha256 ${esc(v)}</div>`).join("")}
        </div></div>`).join("")}</div>
    <div class="section"><span class="micro">Integrity checks run when this bundle was built</span>
      <dl class="facts">
        <dt>Official run</dt><dd>${esc(integ.connectivity_run)} (road rule ${esc(integ.official_road_rule)})</dd>
        <dt>Output hashes</dt><dd>${Object.keys(integ.outputs_verified).length} files match their provenance</dd>
        <dt>Summary</dt><dd>${esc(integ.summary_check)}</dd>
        <dt>Route traces</dt><dd>${int(integ.route_traces.verified)} verified, ${int(integ.route_traces.unavailable)} unavailable. ${esc(integ.route_traces.rule)}</dd>
      </dl>
      ${integ.notes.length ? `<p class="disclaimer">${integ.notes.map(esc).join("<br>")}</p>` : ""}</div>`;
  body.querySelectorAll(".link").forEach((l) => {
    const t = () => l.setAttribute("aria-expanded", l.getAttribute("aria-expanded") === "true" ? "false" : "true");
    l.addEventListener("click", t);
    l.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); t(); } });
  });
  body.querySelector("#followBtn2").addEventListener("click", () => app.follow());
}

// ---------------------------------------------------------------- operations
function operations(body, app) {
  const m = app.m;
  const unverified = m.settlements.filter((x) => x.access_state === "POSSIBLE_CUTOFF" || x.access_state === "CONFIRMED_CUTOFF");
  body.innerHTML = `
    <div class="section"><p class="lede">Where should a responder look first?</p>
      <p class="disclaimer">Each list is ordered by the quantity it states. BATO computes no rescue priority score.</p></div>
    <div class="section"><span class="micro">Blocked roads · by stretch rank</span><div class="list" id="ol1"></div></div>
    <div class="section"><span class="micro">Unobservable road bordering access-unverified settlements</span><div class="list" id="ol2"></div></div>
    <div class="section"><span class="micro">Settlements cut off or with unverified access</span>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px"><span class="faint" style="font-size:11.5px">Order by</span>
        <select class="sel" id="osort">
          <option value="unobs">Fewest unobservable km on route</option>
          <option value="bldg">Most mapped buildings (proxy)</option>
          <option value="hosp">Shortest road to a hospital (pre-event)</option></select></div>
      <div class="list" id="ol3"></div></div>`;
  const l1 = body.querySelector("#ol1");
  l1.innerHTML = m.b.stretches.map((st) => {
    const info = m.segInfo(st.segments[0]);
    return `<button class="row" data-st="${esc(st.stretch_id)}"><span class="t">#${st.rank} · ${esc(roadName(info))}${st.bridge ? " · bridge" : ""}</span>
      <span class="v">${int(st.affected_length_m)} m</span>
      <span class="s">Reconnects ${int(st.settlements_reconnected)} settlement${st.settlements_reconnected === 1 ? "" : "s"} if cleared · on ${int(st.on_verified_pre_event_routes.length)} verified pre-event route${st.on_verified_pre_event_routes.length === 1 ? "" : "s"}</span></button>`;
  }).join("") || '<p class="disclaimer">No blocked road in this result.</p>';
  l1.querySelectorAll("[data-st]").forEach((r) => r.addEventListener("click", () => app.selectStretch(r.dataset.st)));

  const l2 = body.querySelector("#ol2");
  l2.innerHTML = m.b.pockets.slice(0, 40).map((p) => {
    const reasons = Object.entries(p.reason_samples).sort((a, b) => b[1] - a[1]);
    return `<button class="row" data-pk="${esc(p.id)}"><span class="t">${kmOrM(p.length_m)} of unobservable road</span>
      <span class="v">${int(p.settlements.length)} sett.</span>
      <span class="s">Borders the observed roads of ${int(p.settlements.length)} access-unverified settlement${p.settlements.length === 1 ? "" : "s"}${reasons.length ? ` · ${esc(labelShort(reasons[0][0]).toLowerCase())}` : ""}</span></button>`;
  }).join("") || '<p class="disclaimer">None.</p>';
  if (m.b.pockets.length > 40) l2.insertAdjacentHTML("beforeend", `<p class="disclaimer">${m.b.pockets.length - 40} more not listed.</p>`);
  l2.querySelectorAll("[data-pk]").forEach((r) => r.addEventListener("click", () => app.selectPocket(r.dataset.pk)));

  const sortSel = body.querySelector("#osort");
  const l3 = body.querySelector("#ol3");
  const key = {
    unobs: (x) => (x.traces.post ? x.traces.post.km_by_state.unobservable : Infinity),
    bldg: (x) => -x.buildings,
    hosp: (x) => (x.distances && x.distances.hospital_pre_event_km != null ? x.distances.hospital_pre_event_km : Infinity),
  };
  function fill() {
    const k = key[sortSel.value];
    const rows = [...unverified].sort((a, b) => k(a) - k(b) || a.id.localeCompare(b.id)).slice(0, 120);
    l3.innerHTML = rows.map((x) => {
      const t = x.traces.post;
      const val = sortSel.value === "bldg" ? `${int(x.buildings)} bldg` : sortSel.value === "hosp"
        ? (x.distances && x.distances.hospital_pre_event_km != null ? `${km1(x.distances.hospital_pre_event_km)} km` : "–")
        : (t ? `${km1(t.km_by_state.unobservable)} km` : "–");
      const sub = x.access_state === "CONFIRMED_CUTOFF" ? `${badge("CONFIRMED_CUTOFF", "Confirmed cut off")} no route even with unobservable road passable`
        : t ? `Access unverified because the route crosses ${t.unobservable_segments} unobservable segment${t.unobservable_segments === 1 ? "" : "s"}`
        : "Route trace unavailable";
      return `<button class="row" data-sid="${esc(x.id)}"><span class="t">${esc(settlementName(x))}</span><span class="v">${val}</span>
        <span class="s">${sub}</span></button>`;
    }).join("") || '<p class="disclaimer">None.</p>';
    l3.querySelectorAll("[data-sid]").forEach((r) => r.addEventListener("click", () => app.selectSettlement(r.dataset.sid, { fly: true })));
  }
  sortSel.addEventListener("change", fill);
  fill();
}

// ---------------------------------------------------------------- methodology
function methodology(body, app) {
  const m = app.m;
  const meth = m.b.methodology;
  const ex = m.b.experiment;
  const steps = [
    ["Sentinel-1 SAR", "Three GRD acquisitions, one relative orbit: PRE-1, PRE-2, POST"],
    ["Radiometric preprocessing", "Orbit, thermal and border noise, calibration to linear σ⁰ (VV, VH)"],
    ["Speckle filtering", "Lee Sigma, fixed parameters"],
    ["Terrain correction", "Range-Doppler with the Copernicus DEM, EPSG:3857, 10 m"],
    ["Temporal comparison", "Common grid stack: POST against PRE-2 and PRE-1"],
    ["Deterministic flood baseline", "New open-water-like darkening; layover, shadow, steep slopes and invalid data are NOT ASSESSED"],
    ["Historical OSM", `Roads, places, hospitals, buildings as of ${day(m.b.incident.osm_snapshot_date)}`],
    ["Road connectivity", "Roads sampled against the flood state; three graphs: pre-event, optimistic, pessimistic"],
    ["Uncertainty-aware settlement access", "Decision table: connected, access unverified, confirmed cut off, no pre-event access"],
  ];
  body.innerHTML = `
    <div class="section"><div class="pipeline">${steps.map(([a, b]) => `<div><b>${a}</b><span>${esc(b)}</span></div>`).join("")}</div></div>
    <div class="section"><span class="micro">Three kinds of statement</span><div class="tri">
      <div><b style="color:var(--observed)">Measured</b>Radar backscatter on each date; mapped OSM features.</div>
      <div><b style="color:var(--text)">Inferred</b>Flood state, road state, routes and access status, by stated rules. Not validated.</div>
      <div><b style="color:var(--unconfirmed)">Unobservable</b>Where the rules cannot decide. Never shown as open or blocked.</div></div></div>
    <div class="section"><span class="micro">Versions and fingerprints</span><dl class="facts">
      ${Object.entries(meth.versions).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v ?? "–")}</dd>`).join("")}
      ${Object.entries(meth.config_fingerprints).map(([k, v]) => `<dt>config · ${esc(k)}</dt><dd class="hash">${esc(v ?? "–")}</dd>`).join("")}
      <dt>Pre-event OSM</dt><dd>snapshot ${day(m.b.incident.osm_snapshot_date)}</dd></dl></div>
    <div class="section"><span class="micro">Not used anywhere</span><p class="p">${meth.inputs_not_used.map(esc).join("; ")}.</p>
      ${meth.validation_notes.filter(Boolean).map((n) => `<p class="disclaimer">${esc(n)}</p>`).join("")}</div>
    <div class="section"><span class="micro">Experimental methods</span><div class="exp">
      An experimental SAR disturbance detector was evaluated against a matched control and rejected under its
      pre-registered validation criteria. It is not used in the official road-impact result.
      ${ex && ex.checks && ex.checks.length ? `<table>${ex.checks.map((c) => `<tr><td>(${esc(c.criterion)}) ${esc(c.test)}</td><td>${esc(c.value)}</td>
        <td class="${c.passed ? "pass" : "fail"}">${c.passed ? "pass" : "fail"}</td></tr>`).join("")}</table>
        <div style="margin-top:6px">Verdict: ${esc(ex.verdict)}</div>` : ""}
    </div></div>`;
}

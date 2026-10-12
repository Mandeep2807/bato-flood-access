// Right-hand evidence drawer: one object at a time (road segment, settlement, unobservable pocket, destination).
import { km1, int, kmOrM, day, esc, ACCESS_LABEL, ROAD_LABEL, KIND_LABEL, settlementName, destinationLabel,
  roadName, routePieces, STATE } from "../data.js";
import { badge, ICON } from "./dom.js";

const REASON_TEXT = {
  radar_geometry: "The radar could not see this ground: it lies in layover or shadow for this orbit.",
  terrain_rule: "The radar saw this ground, but the slope is above the Step 4C limit, where the open-water flood rule is not applied. The road state cannot be decided.",
  data_gap: "The radar image or the DEM has no valid data here.",
  unknown: "No not-assessed reason was recorded for this segment.",
};

export function createDrawer(app) {
  const root = document.getElementById("drawer");
  const head = document.getElementById("drawerHead");
  const body = document.getElementById("drawerBody");
  const stage = document.getElementById("stage");
  const m = app.m;

  function show(headHTML, bodyHTML) {
    head.innerHTML = headHTML;
    body.innerHTML = bodyHTML;
    body.scrollTop = 0;
    root.hidden = false;
    stage.classList.add("drawer-open");
    head.querySelector("[data-close]")?.addEventListener("click", () => app.clearSelection());
    body.querySelectorAll("[data-sid]").forEach((r) => r.addEventListener("click", () => app.selectSettlement(r.dataset.sid, { fly: true })));
    body.querySelectorAll("[data-seg]").forEach((r) => r.addEventListener("click", () => app.selectSegment(Number(r.dataset.seg), { fly: true })));
  }
  const closeBtn = `<button class="x-btn" data-close aria-label="Close">${ICON.close}</button>`;

  function close() {
    root.hidden = true;
    stage.classList.remove("drawer-open");
  }

  // ------------------------------------------------------------ road segment
  function segment(i) {
    const info = m.segInfo(i);
    const st = info.stretch;
    const ev = info.evidence ? m.b.evidence[info.evidence] : null;
    const headHTML = `<div class="row1"><span class="micro">Road impact</span>${closeBtn}</div>
      <div style="margin-top:8px">${badge(info.state, ROAD_LABEL[info.state])}${info.bridge ? ' <span class="badge" style="box-shadow:inset 0 0 0 1px var(--rule-2)">Bridge</span>' : ""}</div>
      <h3>${esc(roadName(info))}</h3>
      <div class="sub">OSM way ${esc(info.way)} · highway=${esc(info.highway)} · ${esc(info.id)}</div>`;
    let html = "";
    if (info.state === "blocked") html = blockedBody(info, st, ev);
    else if (info.state === "unobservable") html = unobsBody(info);
    else html = openBody(info);
    show(headHTML, html);
    const toggle = body.querySelector("[data-state-toggle]");
    toggle?.addEventListener("click", () => {
      const grid = body.querySelector(".sar");
      grid.classList.toggle("show-state");
      toggle.setAttribute("aria-pressed", grid.classList.contains("show-state") ? "true" : "false");
    });
  }

  function dates() {
    const d = m.dates;
    return `<div class="dates"><div><span>PRE-1</span><b>${day(d.PRE1?.acquired)}</b></div><div><span>PRE-2</span><b>${day(d.PRE2?.acquired)}</b></div>
      <div><span>POST</span><b>${day(d.POST?.acquired)}</b></div></div>`;
  }

  function blockedBody(info, st, ev) {
    const v = (ev && ev.values) || {};
    const s4 = Object.entries(v).filter(([k]) => k.startsWith("step4c_samples_"));
    const reconnected = st ? st.settlements_reconnected : 0;
    const onRoutes = st ? st.on_verified_pre_event_routes : [];
    return `
      <div class="block"><div class="kv">
        <div><span>Affected length (segment)</span><b>${int(info.affected)}<small>m</small></b></div>
        <div><span>Segment length</span><b>${int(info.length)}<small>m</small></b></div>
        ${st ? `<div><span>Blocked stretch ${esc(st.stretch_id)} · rank ${st.rank}</span><b>${int(st.affected_length_m)}<small>m of ${int(st.length_m)} m</small></b></div>
        <div><span>Longest flood run (stretch)</span><b>${int(st.longest_flood_run_m)}<small>m</small></b></div>` : ""}
        <div><span>Settlements reconnected if cleared</span><b>${int(reconnected)}</b></div>
        <div><span>Verified routes through it</span><b>${int(onRoutes.length)}</b></div>
        <div class="wide"><span>Evidence source</span><span class="txt">Sentinel-1 SAR, VV and VH, one orbit track · Step 4C flood state</span></div>
      </div></div>
      <div class="block"><span class="micro">SAR evidence · pre / post change</span>${sarBlock(st)}${dates()}</div>
      <div class="block"><span class="micro">Step 4C state under the road</span>
        <p class="p" style="margin:0">${v.samples != null ? `${int(v.flood_samples)} of ${int(v.samples)} road samples on this segment lie within the road buffer of FLOOD pixels.` : ""}
        ${s4.length ? "Samples by Step 4C state: " + s4.map(([k, n]) => `${esc(k.replace("step4c_samples_", "").toLowerCase().replace(/_/g, " "))} ${int(n)}`).join(", ") + "." : ""}</p></div>
      <div class="block"><span class="micro">Observability</span>
        <p class="p" style="margin:0">${info.samples ? `Some samples were not assessed (${Object.entries(info.samples).map(([k, n]) => `${esc(k.replace("NOT_ASSESSED_", "").toLowerCase().replace(/_/g, " "))} ${n}`).join(", ")}), but the observed flood run already reaches the blocking length.` : "Every sample on this segment was assessed by the radar rule."}</p></div>
      <div class="block"><span class="micro">Settlement impact</span>
        <div class="big-zero"><b>${int(reconnected)}</b><p>settlement${reconnected === 1 ? "" : "s"} reconnected if this stretch is cleared</p></div>
        ${reconnected === 0 ? `<div class="callout">No settlement access change demonstrated by this blocked stretch.</div>` :
          `<div class="callout red">Clearing this stretch alone restores a route for ${int(reconnected)} settlement${reconnected === 1 ? "" : "s"} (${int(st.reconnected_building_count)} mapped buildings, proxy).</div>`}
        ${onRoutes.length ? `<p class="p" style="margin-top:10px">${int(onRoutes.length)} settlement${onRoutes.length === 1 ? "" : "s"} used this stretch on the verified pre-event route:</p>
          <div class="list">${onRoutes.map((id) => detourRow(id)).join("")}</div>` :
          `<p class="disclaimer">No verified pre-event settlement route passes through this stretch.</p>`}
      </div>
      ${evidenceDetails(ev)}`;
  }

  function detourRow(id) {
    const s = m.byId.get(id);
    if (!s) return "";
    const pre = s.routes.pre_event.length_m, opt = s.routes.post_optimistic;
    const txt = opt.reachable ? `${kmOrM(pre)} before → ${kmOrM(opt.length_m)} with blocked roads removed${pre ? ` (${(opt.length_m / pre).toFixed(2)}×)` : ""}`
      : "no route once blocked roads are removed";
    return `<button class="row" data-sid="${esc(id)}"><span class="t">${esc(settlementName(s))}</span><span class="v">${badge(s.access_state, ACCESS_LABEL[s.access_state])}</span><span class="s">${txt}</span></button>`;
  }

  function sarBlock(st) {
    if (!st || !st.chip) {
      return `<p class="disclaimer">SAR image chips were not built for this bundle (the build step needs rasterio and the stack).</p>`;
    }
    const c = st.chip;
    const svg = chipRoadSVG(c, st.segments);
    const fig = (role, label) => `<figure><img src="${m.root}${c.files[role]}" alt="${label} VV backscatter">
      ${role === "POST" ? `<img class="state" src="${m.root}${c.files.state}" alt="Step 4C state">` : ""}${svg}<figcaption>${label}</figcaption></figure>`;
    return `<div class="sar">${fig("PRE1", "PRE-1")}${fig("PRE2", "PRE-2")}${fig("POST", "POST")}</div>
      <div class="sar-change sar"><figure><img src="${m.root}${c.files.change}" alt="POST minus PRE-2 change">${svg}<figcaption>POST − PRE-2</figcaption></figure>
        <figure><img src="${m.root}${c.files.state}" alt="Step 4C state" style="background:#050809">${svg}<figcaption>Step 4C state</figcaption></figure></div>
      <div class="scale"><span>−${c.change_db_range} dB darker</span><i></i><span>+${c.change_db_range} dB brighter</span></div>
      <div class="toggle-row"><button class="chip" data-state-toggle aria-pressed="false">Overlay FLOOD pixels on POST</button></div>
      <p class="disclaimer">VV backscatter, the same ${c.vv_db_range[0]} to ${c.vv_db_range[1]} dB display stretch on every date. Open water is dark.
        Window ${int(c.width * 10)} m × ${int(c.height * 10)} m around the stretch; the outlined line is the road.</p>`;
  }

  function chipRoadSVG(chip, idx) {
    const [[w, n], [e], [, s]] = [chip.corners[0], chip.corners[1], chip.corners[2]];
    const my = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
    const W = chip.width, H = chip.height;
    const x = (lon) => ((lon - w) / (e - w)) * W;
    const y = (lat) => ((my(n) - my(lat)) / (my(n) - my(s))) * H;
    const paths = idx.map((i) => m.segCoords(i).map(([lo, la], k) => `${k ? "L" : "M"}${x(lo).toFixed(1)} ${y(la).toFixed(1)}`).join(" "));
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${paths.map((d) =>
      `<path d="${d}" fill="none" stroke="#0b1115" stroke-width="7" stroke-linecap="round" vector-effect="non-scaling-stroke" opacity=".55"/>
       <path d="${d}" fill="none" stroke="#ff6a60" stroke-width="2.4" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`).join("")}</svg>`;
  }

  function unobsBody(info) {
    const users = m.postUse.get(info.i) || [];
    const reasons = info.samples ? Object.entries(info.samples) : [];
    return `
      <div class="block"><div class="kv">
        <div><span>Segment length</span><b>${int(info.length)}<small>m</small></b></div>
        <div><span>Not assessed on it</span><b>${int(info.unobsLen)}<small>m</small></b></div>
        <div class="wide"><span>Main reason</span><span class="txt">${esc(m.b.reason_groups.find((g) => g.key === info.reason)?.label || "–")}</span></div>
      </div>
      <div class="callout">${esc(REASON_TEXT[info.reason] || REASON_TEXT.unknown)} This road is not classified as safe or as damaged.</div></div>
      ${reasons.length ? `<div class="block"><span class="micro">Road samples by Step 4C reason</span><p class="p" style="margin:0">${reasons.map(([k, n]) => `${esc(k.replace("NOT_ASSESSED_", "").toLowerCase().replace(/_/g, " "))}: ${int(n)}`).join(" · ")}</p></div>` : ""}
      <div class="block"><span class="micro">Routes through it</span>
        ${users.length ? `<p class="p">${int(users.length)} settlement${users.length === 1 ? "'s" : "s'"} verified post-event route${users.length === 1 ? " uses" : "s use"} this segment, so ${users.length === 1 ? "its" : "their"} access depends on it.</p>
          <div class="list">${users.slice(0, 12).map((id) => { const s = m.byId.get(id); return `<button class="row" data-sid="${esc(id)}"><span class="t">${esc(settlementName(s))}</span><span class="v">${badge(s.access_state, ACCESS_LABEL[s.access_state])}</span></button>`; }).join("")}</div>
          ${users.length > 12 ? `<p class="disclaimer">${users.length - 12} more.</p>` : ""}` : `<p class="disclaimer">No verified settlement route uses this segment.</p>`}</div>
      <div class="block"><span class="micro">Evidence</span><p class="hash">Record ${esc(info.evidence || "–")} in evidence_records.jsonl (rule:road_state_from_baseline_v1)</p></div>`;
  }

  function openBody(info) {
    return `<div class="block"><div class="kv">
        <div><span>Segment length</span><b>${int(info.length)}<small>m</small></b></div>
        <div><span>Flood signal on it</span><b>${int(info.affected)}<small>m</small></b></div></div>
      <div class="callout teal">Observed by the radar rule with no blocking flood signal${info.affected > 0 ? " (a shorter flood run is present, below the blocking length)" : ""}.
        Open means no flood signal on observed road, not verified passability.</div></div>`;
  }

  // ------------------------------------------------------------ settlement
  function settlement(id) {
    const s = m.byId.get(id);
    const r = s.routes;
    const post = s.traces.post, pre = s.traces.pre;
    const postGraph = post ? post.graph : null;
    const postRoute = r.post_pessimistic.reachable ? r.post_pessimistic : r.post_optimistic.reachable ? r.post_optimistic : null;
    const unobsShare = post && post.length_m ? (post.km_by_state.unobservable * 1000) / post.length_m : null;
    const ev = s.evidence_ids.length ? m.b.evidence[s.evidence_ids[0]] : null;
    const explain = {
      CONNECTED: `Reaches ${esc(destinationLabel(m, r.post_pessimistic.target_id))} on observed-open roads, even with every unobservable road treated as closed.`,
      POSSIBLE_CUTOFF: `Access cannot be fully verified because part of the route lies on radar-unobservable road. With that road assumed passable it reaches ${esc(destinationLabel(m, r.post_optimistic.target_id))}.`,
      CONFIRMED_CUTOFF: "No route to any destination once blocked roads are removed, even with every unobservable road treated as passable.",
      NO_PRE_EVENT_ACCESS: "No road within reach in the pre-event OSM, so road access cannot be analysed. This is not a flood effect.",
    }[s.access_state];
    const headHTML = `<div class="row1"><span class="micro">Settlement access</span>${closeBtn}</div>
      <div style="margin-top:8px">${badge(s.access_state, ACCESS_LABEL[s.access_state])}</div>
      <h3>${esc(settlementName(s))}</h3>
      <div class="sub">${esc(s.place_type || s.source || "")} · ${esc(s.id)}</div>`;
    const dist = s.distances || {};
    const html = `
      <div class="block"><div class="kv">
        <div><span>Mapped buildings</span><b>${int(s.buildings)}</b></div>
        <div><span>Destination</span><span class="txt">${esc(destinationLabel(m, (postRoute || r.pre_event).target_id))}</span></div>
        <div class="wide faint" style="font-size:11px;margin-top:-6px">Building count proxy — not population.</div>
        <div><span>Pre-event route</span><b>${r.pre_event.reachable ? km1(r.pre_event.length_m / 1000) : "–"}<small>km</small></b></div>
        <div><span>Post-event route${postGraph === "post_optimistic" ? " (unobservable passable)" : ""}</span><b>${postRoute ? km1(postRoute.length_m / 1000) : "none"}<small>${postRoute ? "km" : ""}</small></b></div>
        <div><span>Unobservable route share</span><b>${unobsShare == null ? "–" : (unobsShare * 100).toFixed(1)}<small>${unobsShare == null ? "" : "%"}</small></b></div>
        <div><span>Unobservable segments on route</span><b>${post ? int(post.unobservable_segments) : "–"}</b></div>
      </div><div class="callout ${s.access_state === "CONNECTED" ? "green" : s.access_state === "CONFIRMED_CUTOFF" ? "red" : ""}">${explain}</div></div>
      ${post || pre ? `<div class="block"><span class="micro">Route · settlement → road network → destination</span>
        ${post ? strip(post, s, "Post-event route" + (postGraph === "post_optimistic" ? ", unobservable road assumed passable" : "")) : ""}
        ${pre && (!post || pre.edges.join() !== post.edges.join()) ? strip(pre, s, "Pre-event route") : ""}
        <p class="disclaimer">Redrawn from the published road segments; its length matches the published route length.</p></div>`
        : s.access_state !== "NO_PRE_EVENT_ACCESS" ? `<div class="block"><p class="disclaimer">Route trace unavailable: it could not be reproduced exactly from the published segments, so it is not drawn.</p></div>` : ""}
      ${s.distances ? `<div class="block"><span class="micro">Road distance to the nearest…</span><div class="kv">
        <div><span>Hospital · pre-event</span><b>${dist.hospital_pre_event_km == null ? "none" : km1(dist.hospital_pre_event_km)}<small>${dist.hospital_pre_event_km == null ? "" : "km"}</small></b></div>
        <div><span>Hospital · observed roads only</span><b>${dist.hospital_post_pessimistic_km == null ? "none" : km1(dist.hospital_post_pessimistic_km)}<small>${dist.hospital_post_pessimistic_km == null ? "" : "km"}</small></b></div>
        <div><span>Road exit · pre-event</span><b>${dist.exit_pre_event_km == null ? "none" : km1(dist.exit_pre_event_km)}<small>${dist.exit_pre_event_km == null ? "" : "km"}</small></b></div>
        <div><span>Road exit · observed roads only</span><b>${dist.exit_post_pessimistic_km == null ? "none" : km1(dist.exit_post_pessimistic_km)}<small>${dist.exit_post_pessimistic_km == null ? "" : "km"}</small></b></div>
      </div><p class="disclaimer">Derived from the published segments and road states; "observed roads only" leaves out blocked and unobservable road.</p></div>` : ""}
      ${s.blocking.length ? `<div class="block"><span class="micro">What separates it</span>
        <p class="p">${int(s.blocking.filter((e) => m.r.state[e] === 1).length)} unobservable and ${int(s.blocking.filter((e) => m.r.state[e] === 0).length)} blocked segment(s) form the edge of the observed-road area around it.</p>
        <button class="btn accent" data-focus>Show them on the map</button></div>` : ""}
      ${evidenceDetails(ev)}`;
    show(headHTML, html);
    body.querySelector("[data-focus]")?.addEventListener("click", () => app.focusSegments(s.blocking));
  }

  function strip(trace, s, label) {
    const pieces = routePieces(m, trace);
    const total = pieces.reduce((a, p) => a + p.len, 0) || 1;
    const cls = ["b", "u", "o"];
    return `<div class="route" style="margin-bottom:10px"><div class="ends"><b>${esc(settlementName(s))}</b><span>${esc(label)} · ${km1(trace.length_m / 1000)} km</span><b>${esc(destinationLabel(m, trace.target_id))}</b></div>
      <div class="strip" role="img" aria-label="${esc(label)}: ${km1(trace.km_by_state.open)} km open, ${km1(trace.km_by_state.unobservable)} km unobservable, ${km1(trace.km_by_state.blocked)} km blocked">
        ${pieces.map((p) => `<i class="${cls[p.s]}" style="flex:${p.len / total}" title="${STATE[p.s]} ${kmOrM(p.len)}"></i>`).join("")}</div>
      <div class="legend"><span style="color:var(--observed)">open ${km1(trace.km_by_state.open)} km</span><span style="color:var(--unconfirmed)">unobservable ${km1(trace.km_by_state.unobservable)} km</span>${trace.km_by_state.blocked ? `<span style="color:var(--blocked)">blocked ${km1(trace.km_by_state.blocked)} km</span>` : ""}</div></div>`;
  }

  // ------------------------------------------------------------ pocket and destination
  function pocket(id) {
    const p = m.pocketById.get(id);
    const reasons = Object.entries(p.reason_samples).sort((a, b) => b[1] - a[1]);
    show(`<div class="row1"><span class="micro">Unobservable road</span>${closeBtn}</div>
      <div style="margin-top:8px">${badge("unobservable", "Unobservable")}</div><h3>${kmOrM(p.length_m)} of unobservable road</h3>
      <div class="sub">${int(p.segments.length)} connected segments · ${esc(p.id)}</div>`,
    `<div class="block"><p class="p">These segments border the observed-road area of ${int(p.settlements.length)} access-unverified settlement${p.settlements.length === 1 ? "" : "s"}.
      A field check here is one of the checks that could verify their access; checking it alone may not be enough.</p>
      ${reasons.length ? `<p class="disclaimer">Not-assessed road samples: ${reasons.map(([k, n]) => `${esc(k.replace("_", " "))} ${int(n)}`).join(" · ")}</p>` : ""}</div>
      <div class="block"><span class="micro">Settlements</span><div class="list">${p.settlements.slice(0, 30).map((sid) => { const s = m.byId.get(sid); return `<button class="row" data-sid="${esc(sid)}"><span class="t">${esc(settlementName(s))}</span><span class="v">${int(s.buildings)} bldg</span></button>`; }).join("")}</div></div>`);
  }

  function destination(id) {
    const d = m.dest.get(id);
    const users = m.settlements.filter((s) => s.routes.post_optimistic.target_id === id || s.routes.post_pessimistic.target_id === id);
    show(`<div class="row1"><span class="micro">Access destination</span>${closeBtn}</div>
      <h3>${esc(d.name || KIND_LABEL[d.kind])}</h3><div class="sub">${esc(KIND_LABEL[d.kind])} · ${esc(d.id)}</div>`,
    `<div class="block"><div class="kv"><div><span>Joined to the road network</span><span class="txt">${d.snapped ? `yes, ${int(d.snap_distance_m)} m from a road` : "no road within reach"}</span></div>
      <div><span>Nearest destination for</span><b>${int(users.length)}<small>settlements</small></b></div></div>
      <p class="disclaimer">${d.kind === "analysis_exit" ? "A road leaving the analysis area (AOI plus buffer): the road continues outside." : d.kind === "town" ? "A town counts as a destination only while it is itself connected to a hospital or road exit." : "Hospital from the pre-event OSM snapshot."}</p></div>`);
  }

  function evidenceDetails(ev) {
    if (!ev) return "";
    const vals = Object.entries(ev.values || {});
    return `<details class="evidence"><summary>View evidence</summary><div class="in">
      <div class="claim">${esc(ev.claim)}</div>
      <div><span class="faint">Method</span> ${esc(ev.method)} · BATO ${esc(ev.method_version)}</div>
      <table>${ev.sources.map((s) => `<tr><td>${esc(s.source)}</td><td>${esc(s.source_id)} · ${esc(s.reference_date)}</td></tr>`).join("")}</table>
      ${vals.length ? `<table>${vals.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>` : ""}
      ${(ev.artifacts || []).map((a) => `<div class="hash">${esc(a.path)}<br>sha256 ${esc(a.sha256 || "–")}</div>`).join("")}
      <div class="hash">evidence ${esc(ev.evidence_id)} · config ${esc(ev.config_fingerprint)}</div></div></details>`;
  }

  return { segment, settlement, pocket, destination, close };
}

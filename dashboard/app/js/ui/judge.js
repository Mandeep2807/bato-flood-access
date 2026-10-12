// Judge mode: a presenter-driven 3-minute walkthrough. Every sentence is filled from the bundle.
import { km1, int, day, esc, kmOrM, plural, settlementName, roadName, demoPicks, unverifiedRouteStats, ACCESS_LABEL } from "../data.js";

export function createJudge(app) {
  const root = document.getElementById("judge");
  const finale = document.getElementById("finale");
  const tab = document.getElementById("judgeTab");
  let step = -1;
  let timers = [];
  const m = app.m;
  const s = m.s, km = m.km, c = m.counts, inc = m.b.incident;
  const { stretch, settlement } = demoPicks(m);
  const reason = Object.entries(s.unobservable_km_by_main_reason).sort((a, b) => b[1] - a[1])[0]?.[0];
  const reasonText = { terrain_rule: "the steep-slope rule of the flood baseline, not radar blindness",
    radar_geometry: "radar layover and shadow", data_gap: "gaps in the radar data", unknown: "unrecorded reasons" }[reason || "unknown"];
  const rs = unverifiedRouteStats(m);
  const reach = s.reachable_settlements;

  const steps = [
    { t: "Incident overview", at: "0:00", go() { app.reset(); app.openPanel("incident"); },
      p: `${esc(inc.title)}, ${day(inc.event_date)}. Three Sentinel-1 passes from one orbit, compared over the pre-event road network:
        ${km1(km.blocked)} km blocked, ${km1(km.open)} km observed open, and ${km1(km.unobservable)} km that radar cannot confirm.` },
    { t: "Map, flood evidence and road network", at: "0:15", go() {
        app.closePanel(); app.setMode("normal"); app.setLayer("flood", true);
        if (stretch) app.mapApi.fit(boundsOf(m.b.stretches.map((x) => x.bbox)), { maxZoom: 12.5 }); else app.home(); },
      p: `Blue is Step 4C flood evidence: new open-water-like darkening against both pre-event dates. Teal roads were observed
        with no flood signal, red ones carry a contiguous flood run, dashed amber ones could not be assessed.` },
    { t: "A blocked road and its evidence", at: "0:45", go() { app.closePanel(); app.setMode("normal"); if (stretch) app.selectStretch(stretch.stretch_id); },
      p: stretch ? `Rank-1 blocked stretch: ${kmOrM(stretch.affected_length_m)} on ${esc(roadName(m.segInfo(stretch.segments[0])))}. The drawer shows the
        radar pixels on all three dates and the rule that fired. Settlements reconnected if it is cleared: ${int(stretch.settlements_reconnected)}.
        BATO shows that number even when it is zero.` : "There is no blocked road in this result." },
    { t: "A settlement whose access is unverified", at: "1:15", go() { app.closePanel(); app.setMode("normal"); if (settlement) app.selectSettlement(settlement.id, { fly: true }); },
      p: !settlement ? "No settlement with a verified route trace." :
        `${esc(settlementName(settlement))}, ${int(settlement.buildings)} mapped buildings (a proxy, not population): ${ACCESS_LABEL[settlement.access_state].toLowerCase()}.
        Its post-event route crosses ${int(settlement.traces.post.unobservable_segments)} unobservable segment${settlement.traces.post.unobservable_segments === 1 ? "" : "s"}
        (${km1(settlement.traces.post.km_by_state.unobservable)} km)` +
        (settlement.access_state === "POSSIBLE_CUTOFF" ? ", so BATO calls it neither connected nor cut off." : ".") },
    { t: "What radar can't confirm", at: "1:45", go() { app.clearSelection(); app.openPanel("observability"); app.setRadarMode(true); app.home(); },
      p: `${km1(km.unobservable)} km of road is unobservable, mostly because of ${reasonText}. Unobservable is not open and not
        damaged: the satellite rule cannot decide, and the interface never hides that.` },
    { t: "Before vs after, and the evidence chain", at: "2:15", go() {
        app.clearSelection(); app.setRadarMode(false); app.openPanel("access");
        timers.push(setTimeout(() => app.openPanel("evidence"), 9000)); },
      p: `Before the event ${int(reach.pre_event)} settlements had road access. With blocked roads removed: ${int(reach.post_optimistic)}.
        With unobservable road also treated as closed: ${int(reach.post_pessimistic)}. ${int(rs.withUnobs)} of ${int(rs.traced)} verified routes
        cross unobservable road, which is where the uncertainty comes from. Every step back to the satellite is hashed and inspectable.` },
    { t: "Final statement", at: "2:45", go() { app.closePanel(); app.drawer.close(); app.home(); showFinale(); }, p: "" },
  ];

  function showFinale() {
    finale.innerHTML = `<div class="card"><h2>Map what radar can confirm. Expose what it cannot. Trace access consequences.</h2>
      <div class="lines">
        <div><i style="background:var(--blocked)"></i><span>${km1(km.blocked)} km of road blocked by new open-water signal, in ${int(s.blocked_stretches)} stretches.</span></div>
        <div><i style="background:var(--observed)"></i><span>${km1(km.open)} km observed open; ${plural(c.CONNECTED, "settlement", "settlements")} verified connected.</span></div>
        <div><i style="background:var(--hatch),#2b2112;box-shadow:inset 0 0 0 1px var(--unconfirmed)"></i><span>${km1(km.unobservable)} km unobservable; ${plural(c.POSSIBLE_CUTOFF, "settlement", "settlements")} with access unverified, ${int(c.CONFIRMED_CUTOFF)} confirmed cut off.</span></div>
        <div><i style="background:var(--blocked);opacity:.5"></i><span>${int(s.blocked_stretches_reconnecting_alone)} of ${plural(s.blocked_stretches, "blocked stretch", "blocked stretches")} reconnect${s.blocked_stretches_reconnecting_alone === 1 ? "s" : ""} a settlement when cleared on ${s.blocked_stretches_reconnecting_alone === 1 ? "its" : "their"} own.</span></div>
        <div><i style="box-shadow:inset 0 0 0 1px var(--text-3)"></i><span>Every number traces to three Sentinel-1 images, the ${day(inc.osm_snapshot_date)} OSM snapshot and hashed provenance. Not validated.</span></div>
      </div></div>`;
    finale.hidden = false;
  }

  function render() {
    const st = steps[step];
    root.innerHTML = `<div class="jtop"><span class="step-n">${step + 1} / ${steps.length}</span><h4>${st.t}</h4><span class="clock">${st.at}</span>
        <button class="x-btn" data-k="exit" aria-label="Leave judge mode"><svg viewBox="0 0 16 16"><path d="M3 3l10 10M13 3L3 13"/></svg></button></div>
      ${st.p ? `<p>${st.p}</p>` : '<p class="muted">Closing card on screen.</p>'}
      <div class="jbot"><button class="btn" data-k="prev" ${step === 0 ? "disabled" : ""}>Previous</button>
        <div class="progress">${steps.map((_, i) => `<i class="${i < step ? "done" : i === step ? "on" : ""}"></i>`).join("")}</div>
        <button class="btn" data-k="reset">Reset demo</button>
        <button class="btn teal" data-k="next">${step === steps.length - 1 ? "Finish" : "Next"}</button></div>`;
    root.querySelector('[data-k="prev"]').addEventListener("click", () => go(step - 1));
    root.querySelector('[data-k="next"]').addEventListener("click", () => (step === steps.length - 1 ? stop() : go(step + 1)));
    root.querySelector('[data-k="reset"]').addEventListener("click", () => go(0));
    root.querySelector('[data-k="exit"]').addEventListener("click", stop);
  }

  function go(i) {
    if (i < 0 || i >= steps.length) return;
    timers.forEach(clearTimeout); timers = [];
    finale.hidden = true;
    app.follow.stop();
    step = i;
    steps[i].go();
    render();
  }

  const stage = document.getElementById("stage");
  function start() { root.hidden = false; stage.classList.add("judging"); tab.setAttribute("aria-pressed", "true"); go(0); }
  function stop() {
    timers.forEach(clearTimeout); timers = [];
    root.hidden = true; finale.hidden = true; step = -1;
    stage.classList.remove("judging");
    tab.setAttribute("aria-pressed", "false");
  }

  tab.addEventListener("click", () => (root.hidden ? start() : stop()));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "j" || e.key === "J") { root.hidden ? start() : stop(); return; }
    if (root.hidden) return;
    if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); step === steps.length - 1 ? stop() : go(step + 1); }
    if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(step - 1); }
    if (e.key === "Escape") stop();
  });
  return { start, stop, go, get step() { return step; }, steps };
}

function boundsOf(boxes) {
  let w = 180, s = 90, e = -180, n = -90;
  for (const [a, b, c, d] of boxes) { w = Math.min(w, a); s = Math.min(s, b); e = Math.max(e, c); n = Math.max(n, d); }
  return [[w, s], [e, n]];
}

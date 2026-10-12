// "Follow the evidence": five linked stages on the real rank-1 blocked stretch (or the one passed in).
import { day, int, kmOrM, esc, ACCESS_LABEL, settlementName, demoPicks } from "../data.js";
import { el, ICON } from "./dom.js";

const STAGE_MS = 2600;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function createFollow(app) {
  const root = document.getElementById("followTrack");
  let timers = [];
  let running = false;

  const stage = document.getElementById("stage");
  function stop() {
    timers.forEach(clearTimeout);
    timers = [];
    running = false;
    root.hidden = true;
    stage.classList.remove("following");
    app.mapApi.chip(null);
  }

  function run(stretch, opts = {}) {
    const m = app.m;
    stop();
    const st = stretch || demoPicks(m).stretch;
    if (!st) {
      root.innerHTML = `<div class="f-stage on"><span class="micro">Follow the evidence</span><p>No blocked road in this result, so there is no flood-to-road chain to follow.</p></div>`;
      root.hidden = false;
      timers.push(setTimeout(stop, 3500));
      return;
    }
    running = true;
    const users = st.on_verified_pre_event_routes.map((id) => m.byId.get(id)).filter(Boolean);
    const target = users[0] || null;
    const post = m.dates.POST, pre2 = m.dates.PRE2;
    const stages = [
      { k: "Post-event SAR", t: st.chip ? `Sentinel-1 VV on ${day(post?.acquired)}. Smooth open water returns little energy and looks dark.` : "SAR chip not in this bundle; the flood state below comes from the same images.",
        go: () => { app.mapApi.fit([[st.bbox[0], st.bbox[1]], [st.bbox[2], st.bbox[3]]], { maxZoom: 14.2 }); if (st.chip) app.mapApi.chip(st.chip, st.chip.files.POST, 0.92); } },
      { k: "Flood evidence", t: `Step 4C: darker than PRE-2 (${day(pre2?.acquired)}) and PRE-1 in VV and VH, outside steep, layover and shadow ground.`,
        go: () => { app.setLayer("flood", true); if (st.chip) app.mapApi.chip(st.chip, st.chip.files.state, 1); } },
      { k: "Road state", t: `${kmOrM(st.affected_length_m)} of road under a contiguous flood run of ${kmOrM(st.longest_flood_run_m)}: BLOCKED (stretch rank ${st.rank}).`,
        go: () => { app.mapApi.chip(null); app.mapApi.selectSegments(st.segments); app.mapApi.focus(st.segments); } },
      { k: "Access status", t: `Settlements reconnected if cleared: ${int(st.settlements_reconnected)}. ${users.length ? `${int(users.length)} verified pre-event route${users.length === 1 ? "" : "s"} passed here.` : "No verified settlement route passed here."}`,
        go: () => { app.mapApi.trace(users.flatMap((u) => u.traces.pre ? u.traces.pre.edges : [])); } },
      { k: "Settlement", t: target ? `${esc(settlementName(target))}: ${ACCESS_LABEL[target.access_state].toLowerCase()}.` : "No settlement's access depends on this stretch.",
        go: () => { if (target) app.selectSettlement(target.id, { fly: false, keepFocus: true }); else app.selectStretch(st.stretch_id, { fly: false }); } },
    ];
    root.innerHTML = "";
    stages.forEach((s, i) => root.appendChild(el(`<div class="f-stage" data-i="${i}"><span class="micro">${i + 1} · ${s.k}</span><p>${s.t}</p><span class="bar"></span></div>`)));
    const close = el(`<button class="x-btn f-close" aria-label="Stop">${ICON.close}</button>`);
    close.addEventListener("click", stop);
    root.appendChild(close);
    root.hidden = false;
    stage.classList.add("following");
    if (!document.getElementById("judge").hidden) stage.classList.remove("following");
    app.closePanel();
    app.clearSelection();
    const dur = opts.stageMs || (REDUCED ? 1800 : STAGE_MS);
    stages.forEach((s, i) => {
      timers.push(setTimeout(() => {
        if (!running) return;
        root.querySelectorAll(".f-stage").forEach((n, j) => { n.classList.toggle("on", j === i); n.classList.toggle("done", j < i); });
        s.go();
      }, i * dur + (i === 0 ? 0 : 400)));
    });
    timers.push(setTimeout(() => {
      root.querySelectorAll(".f-stage").forEach((n) => { n.classList.remove("on"); n.classList.add("done"); });
      running = false;
    }, stages.length * dur + 600));
  }

  return { run, stop, get running() { return running; } };
}

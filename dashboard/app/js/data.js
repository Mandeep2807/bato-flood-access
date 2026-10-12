// Loading the bundle and every derived view of it. Pure functions only: nothing here invents a number;
// every value is read from the bundle (built from the frozen official outputs) or counted from it.

export const STATE = ["blocked", "unobservable", "open"];
export const ACCESS = ["CONNECTED", "POSSIBLE_CUTOFF", "CONFIRMED_CUTOFF", "NO_PRE_EVENT_ACCESS"];
export const ACCESS_LABEL = {
  CONNECTED: "Connected",
  POSSIBLE_CUTOFF: "Access unverified",
  CONFIRMED_CUTOFF: "Confirmed cut off",
  NO_PRE_EVENT_ACCESS: "No pre-event road access",
};
export const ROAD_LABEL = { blocked: "Blocked", unobservable: "Unobservable", open: "Open" };
export const COLORS = {
  blocked: "#f0534a", open: "#3cc4ae", unobservable: "#e2a33a", water: "#4e8ef7", noaccess: "#56636b",
  radar_geometry: "#f2b13f", terrain_rule: "#b88a4c", data_gap: "#7d8b93", unknown: "#7d8b93",
  CONNECTED: "#6fd08a", POSSIBLE_CUTOFF: "#e2a33a", CONFIRMED_CUTOFF: "#f0534a", NO_PRE_EVENT_ACCESS: "#56636b",
};
export const KIND_LABEL = { hospital: "Hospital", town: "Town", analysis_exit: "Road exit from the analysis area" };

export async function loadIndex(base = "bundles/") {
  const r = await fetch(base + "index.json", { cache: "no-store" });
  if (!r.ok) throw new Error(`No dashboard bundle found (${base}index.json: HTTP ${r.status}). Run scripts/build_dashboard.py.`);
  return r.json();
}

export async function loadBundle(path, base = "bundles/") {
  const root = `${base}${path}/`;
  const r = await fetch(root + "bundle.json", { cache: "no-store" });
  if (!r.ok) throw new Error(`Cannot read ${root}bundle.json (HTTP ${r.status})`);
  const b = await r.json();
  if (b.version !== "bato-dashboard-bundle-v1") throw new Error(`Unsupported bundle version ${b.version}`);
  if (b.incident.road_rule !== "v1") throw new Error("This bundle is not the official road rule v1 result");
  return { bundle: b, root };
}

// ---------------------------------------------------------------- formatting
const nf1 = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const km1 = (v) => (v == null ? "–" : nf1.format(v));
export const int = (v) => (v == null ? "–" : nf0.format(v));
export const m0 = (v) => (v == null ? "–" : `${nf0.format(v)} m`);
export const kmOrM = (meters) => (meters == null ? "–" : meters >= 1000 ? `${km1(meters / 1000)} km` : `${nf0.format(meters)} m`);
export const pct = (v, digits = 1) => (v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(digits)} %`);
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function day(iso) {
  if (!iso) return "–";
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}
export const dayShort = (iso) => day(iso).split(" ").slice(0, 2).join(" ");
export const utcTime = (iso) => (iso && iso.length > 15 ? `${iso.slice(11, 16)} UTC` : "");
export const plural = (n, one, many) => `${nf0.format(n)} ${n === 1 ? one : many}`;
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- model
export function buildModel(bundle, root) {
  const b = bundle;
  const r = b.roads;
  const s = b.summary;
  const km = s.road_km_by_state;
  const total = km.blocked + km.open + km.unobservable;
  const settlements = b.settlements;
  const byId = new Map(settlements.map((x) => [x.id, x]));
  const dest = new Map(b.destinations.map((d) => [d.id, d]));
  const stretchById = new Map(b.stretches.map((x) => [x.stretch_id, x]));
  const pocketById = new Map(b.pockets.map((x) => [x.id, x]));
  const reasonKeys = b.reason_groups.map((g) => g.key);

  // which settlements' published routes (verified traces) use each segment
  const postUse = new Map();
  for (const st of settlements) {
    const t = st.traces.post;
    if (!t) continue;
    for (const e of t.edges) {
      if (!postUse.has(e)) postUse.set(e, []);
      postUse.get(e).push(st.id);
    }
  }
  const counts = Object.fromEntries(ACCESS.map((k) => [k, s.settlements_by_access_state[k] ?? 0]));
  const timeline = b.timeline;
  const dates = Object.fromEntries(timeline.map((t) => [t.role, t]));

  return {
    b, root, r, s, km, total, counts, settlements, byId, dest, stretchById, pocketById, reasonKeys, postUse, dates,
    share: { blocked: km.blocked / total, open: km.open / total, unobservable: km.unobservable / total },
    segCoords(i) {
      const c = r.coords[i];
      const out = [];
      for (let k = 0; k < c.length; k += 2) out.push([c[k], c[k + 1]]);
      return out;
    },
    segInfo(i) {
      const way = r.ways[String(r.way[i])] || [null, null, r.highways[r.hw[i]]];
      return {
        i, id: r.ids[i], state: STATE[r.state[i]], reason: r.reason[i] >= 0 ? reasonKeys[r.reason[i]] : null,
        length: r.length[i], way: r.way[i], name: way[0], ref: way[1], highway: r.highways[r.hw[i]],
        bridge: r.bridge[i], affected: r.affected[i], unobsLen: r.unobs_len[i],
        stretch: r.stretch[i] >= 0 ? b.stretches[r.stretch[i]] : null, evidence: r.evidence[i],
        samples: r.samples[String(i)] || null,
      };
    },
  };
}

export function roadName(info) {
  const bits = [info.name, info.ref && info.ref !== info.name ? info.ref : null].filter(Boolean);
  return bits.length ? bits.join(" · ") : `Unnamed ${info.highway}`;
}

export function settlementName(st) {
  if (st.name) return st.name;
  return st.source === "building_cluster" ? "Unnamed building cluster" : `Unnamed ${st.place_type || "place"}`;
}

export function destinationLabel(m, id) {
  const d = m.dest.get(id);
  if (!d) return id || "–";
  const kind = d.kind === "analysis_exit" ? "Road exit" : d.kind === "hospital" ? "Hospital" : "Town";
  return d.name ? `${kind} · ${d.name}` : kind;
}

export function roadsGeoJSON(m) {
  const r = m.r;
  const features = new Array(r.count);
  for (let i = 0; i < r.count; i++) {
    features[i] = {
      type: "Feature", id: i,
      properties: { i, s: r.state[i], g: r.reason[i], st: r.stretch[i] },
      geometry: { type: "LineString", coordinates: m.segCoords(i) },
    };
  }
  return { type: "FeatureCollection", features };
}

export function settlementsGeoJSON(m) {
  return {
    type: "FeatureCollection",
    features: m.settlements.map((s, k) => ({
      type: "Feature", id: k,
      properties: { id: s.id, a: s.access_state, n: s.buildings, name: settlementName(s) },
      geometry: { type: "Point", coordinates: [s.lon, s.lat] },
    })),
  };
}

export function segmentsGeoJSON(m, idx, extra = {}) {
  return {
    type: "FeatureCollection",
    features: idx.map((i) => ({ type: "Feature", properties: { s: m.r.state[i], ...extra },
      geometry: { type: "LineString", coordinates: m.segCoords(i) } })),
  };
}

export function bboxOf(m, idx) {
  let w = 180, s = 90, e = -180, n = -90;
  for (const i of idx) {
    const c = m.r.coords[i];
    for (let k = 0; k < c.length; k += 2) {
      w = Math.min(w, c[k]); e = Math.max(e, c[k]); s = Math.min(s, c[k + 1]); n = Math.max(n, c[k + 1]);
    }
  }
  return [[w, s], [e, n]];
}

// route strip pieces (settlement -> destination), merged runs of equal state
export function routePieces(m, trace) {
  if (!trace) return [];
  const out = [];
  for (const e of trace.edges) {
    const st = m.r.state[e];
    const len = m.r.length[e];
    if (out.length && out[out.length - 1].s === st) out[out.length - 1].len += len;
    else out.push({ s: st, len });
  }
  return out;
}

export function aoiBounds(m) {
  const ring = m.b.aoi[0];
  let w = 180, s = 90, e = -180, n = -90;
  for (const [x, y] of ring) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
  return [[w, s], [e, n]];
}

// Deterministic demo picks (stated in the UI): the rank-1 blocked stretch; the access-unverified settlement with
// a verified route trace and the most mapped buildings (proxy).
export function demoPicks(m) {
  const stretch = m.b.stretches.find((x) => x.rank === 1) || null;
  const unverified = m.settlements.filter((x) => x.access_state === "POSSIBLE_CUTOFF" && x.traces.post);
  unverified.sort((a, b) => b.buildings - a.buildings || a.id.localeCompare(b.id));
  const settlement = unverified[0] || m.settlements.find((x) => x.traces.post) || null;
  return { stretch, settlement };
}

export function unverifiedRouteStats(m) {
  const reach = m.settlements.filter((x) => x.routes.pre_event.reachable);
  const traced = reach.filter((x) => x.traces.post);
  const withUnobs = traced.filter((x) => x.traces.post.unobservable_segments > 0);
  return { reachable: reach.length, traced: traced.length, withUnobs: withUnobs.length };
}

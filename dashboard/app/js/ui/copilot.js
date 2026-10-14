// SITREP Copilot panel. The model runs on the local BATO server (never in the browser, never with a key here):
// the server gives it only the verified official v1 evidence and rejects any answer whose numbers are not in the
// cited evidence. This panel shows the answer, its evidence, and what was checked.
import { esc } from "../data.js";

const API = "api/copilot/";
const HEADERS = { "Content-Type": "application/json", "X-BATO-Copilot": "1" };
const EXAMPLES = {
  en: ["Which roads are blocked, and where?", "How many settlements have unverified access?",
    "Why is so much road unobservable?", "When was the satellite observation made?",
    "What should field teams verify first?"],
  ne: ["कुन सडकहरू अवरुद्ध छन्, र कहाँ?", "कति बस्तीहरूको पहुँच अपुष्ट छ?", "धेरै सडक किन अवलोकन गर्न नसकिएको हो?",
    "भू-उपग्रह अवलोकन कहिले गरिएको थियो?", "फिल्ड टोलीले सबैभन्दा पहिले के जाँच गर्नुपर्छ?"],
};
const STATUS_LABEL = {
  answered: "Answered from the evidence",
  unsupported: "Not supported by BATO evidence",
  withheld: "Answer withheld: failed the evidence check",
  unavailable: "Copilot unavailable",
  provider_error: "Model provider error",
  rate_limited: "Request limit reached",
  no_evidence: "Official result not available",
};

async function call(path, body) {
  const r = await fetch(API + path, body ? { method: "POST", headers: HEADERS, body: JSON.stringify(body) } : {});
  let data = {};
  try { data = await r.json(); } catch { data = { error: `HTTP ${r.status}` }; }
  return { ok: r.ok, status: r.status, data };
}

function citationHTML(c) {
  const link = c.id.startsWith("stretch:") || c.id.startsWith("settlement:")
    ? `<button class="btn cp-show" data-id="${esc(c.id)}">Show on map</button>` : "";
  return `<details class="cp-cite"><summary><code>${esc(c.id)}</code> ${esc(c.title)}</summary>
    <p>${esc(c.text)}</p><p class="faint">Source: ${esc(c.source)}</p>${link}</details>`;
}

function answerHTML(d) {
  const st = d.status || (d.error ? "unavailable" : "answered");
  const head = `<div class="cp-badge ${esc(st)}">${esc(STATUS_LABEL[st] || st)}</div>`;
  if (d.error) return `${head}<p class="p">${esc(d.error)}</p>`;
  if (st === "withheld") {
    return `${head}<p class="p">${esc(d.message)}</p><p class="faint">Check: ${esc((d.problems || []).join("; "))}</p>
      <span class="micro">Verified evidence</span>${(d.evidence || []).map(citationHTML).join("")}`;
  }
  const lang = d.language === "ne" ? ' lang="ne"' : "";
  return `${head}<div class="cp-answer"${lang}>${esc(d.answer).replace(/\n/g, "<br>")}</div>
    <p class="faint cp-note">${esc(d.note || "")} Model: ${esc(d.model || "")}.</p>
    ${(d.citations || []).length ? `<span class="micro">Evidence cited</span>${d.citations.map(citationHTML).join("")}` : ""}`;
}

function sitrepHTML(d, lang) {
  if (d.error) return answerHTML(d);
  if (d.status === "withheld") {
    return `<div class="cp-badge withheld">Nepali translation withheld</div><p class="p">${esc(d.message)}</p>
      <p class="faint">Check: ${esc((d.problems || []).join("; "))}</p>`;
  }
  const badge = d.status === "deterministic" ? "Deterministic template · no AI" : `AI translation · ${esc(d.model)}`;
  return `<div class="cp-badge ${d.status === "deterministic" ? "deterministic" : "answered"}">${badge}</div>
    <p class="faint cp-note">${esc(d.note)}</p>
    <div class="cp-actions"><button class="btn" id="cpDownload">Download .md</button></div>
    <pre class="cp-md"${lang === "ne" ? ' lang="ne"' : ""}>${esc(d.markdown)}</pre>`;
}

export function copilotPanel(body, app) {
  const st = app.copilot || (app.copilot = { lang: "en", question: "", answer: "", sitrep: "", status: null });
  const inc = app.m.b.incident;
  body.innerHTML = `
    <div class="section cp">
      <p class="p">Ask about blocked or unobservable roads, settlement access, observation dates, uncertainty and
        field-verification priorities, in English or Nepali. A language model writes the answer using only the
        verified official result; the server rejects any number that is not in the evidence it cites.</p>
      <div class="cp-status" id="cpStatus">Checking the copilot…</div>
      <div class="chips" role="radiogroup" aria-label="Answer language">
        <button class="chip" data-lang="en" role="radio">English</button>
        <button class="chip" data-lang="ne" role="radio" lang="ne">नेपाली</button></div>
      <textarea id="cpQ" class="cp-q" rows="3" maxlength="1000" aria-label="Question"></textarea>
      <div class="cp-ex" id="cpEx"></div>
      <div class="cp-actions"><button class="btn teal" id="cpAsk">Ask</button></div>
      <div class="cp-out" id="cpOut" aria-live="polite">${st.answer}</div>
    </div>
    <div class="section"><span class="micro">Situation report</span>
      <div class="cp-actions"><button class="btn" id="cpSitEn">SITREP · English</button>
        <button class="btn" id="cpSitNe" lang="ne">SITREP · नेपाली</button></div>
      <div class="cp-out" id="cpSit">${st.sitrep}</div>
    </div>`;
  const $ = (id) => body.querySelector(`#${id}`);
  const q = $("cpQ");
  q.value = st.question;
  const setLang = (lang) => {
    st.lang = lang;
    body.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    q.placeholder = lang === "ne" ? "प्रश्न सोध्नुहोस्…" : "Ask a question about this result…";
    $("cpEx").innerHTML = EXAMPLES[lang].map((x) => `<button class="chip cp-x"${lang === "ne" ? ' lang="ne"' : ""}>${esc(x)}</button>`).join("");
    $("cpEx").querySelectorAll(".cp-x").forEach((b) => b.addEventListener("click", () => { q.value = b.textContent; ask(); }));
  };
  body.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  setLang(st.lang);

  const wire = (root) => root.querySelectorAll(".cp-show").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.id;
    if (id.startsWith("stretch:")) app.selectStretch(id.slice(8));
    else app.selectSettlement(id.slice(11), { fly: true });
  }));
  wire($("cpOut"));

  const busy = (on) => ["cpAsk", "cpSitEn", "cpSitNe"].forEach((k) => { $(k).disabled = on; });
  async function ask() {
    const question = q.value.trim();
    if (!question) return;
    st.question = question;
    busy(true);
    $("cpOut").innerHTML = `<p class="faint">Retrieving evidence and asking the model…</p>`;
    try {
      const r = await call("ask", { aoi_id: inc.aoi_id, candidate_id: inc.candidate_id, question, lang: st.lang });
      st.answer = answerHTML(r.data);
    } catch (e) {
      st.answer = answerHTML({ status: "unavailable", error: `The copilot server could not be reached (${e.message}).` });
    }
    $("cpOut").innerHTML = st.answer;
    wire($("cpOut"));
    busy(false);
  }
  async function sitrep(lang) {
    busy(true);
    $("cpSit").innerHTML = `<p class="faint">${lang === "ne" ? "Translating the verified SITREP…" : "Building the SITREP…"}</p>`;
    let d;
    try {
      d = (await call("sitrep", { aoi_id: inc.aoi_id, candidate_id: inc.candidate_id, lang, title: inc.title })).data;
    } catch (e) {
      d = { status: "unavailable", error: `The copilot server could not be reached (${e.message}).` };
    }
    st.sitrep = sitrepHTML(d, lang);
    $("cpSit").innerHTML = st.sitrep;
    const dl = $("cpDownload");
    if (dl && d.markdown) {
      dl.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([d.markdown], { type: "text/markdown;charset=utf-8" }));
        a.download = `BATO_SITREP_${lang}.md`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      });
    }
    busy(false);
  }
  $("cpAsk").addEventListener("click", ask);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); });
  $("cpSitEn").addEventListener("click", () => sitrep("en"));
  $("cpSitNe").addEventListener("click", () => sitrep("ne"));

  call("status").then((r) => {
    st.status = r.data;
    $("cpStatus").innerHTML = r.data.available
      ? `<span class="cp-dot on"></span>AI copilot ready · ${esc(r.data.provider)}${r.data.model ? ` · ${esc(r.data.model)}` : ""}`
      : `<span class="cp-dot"></span><b>Copilot unavailable.</b> ${esc(r.data.reason || "")}
         <span class="faint">The English SITREP (deterministic, no AI) still works.</span>`;
    $("cpAsk").disabled = !r.data.available;
    $("cpSitNe").disabled = !r.data.available;
  }).catch(() => { $("cpStatus").textContent = "Copilot unavailable: the copilot server could not be reached."; });
}

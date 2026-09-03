/* The run viewer. Plain DOM and hand-drawn SVG; no library.
 *
 * Everything shown comes from /api/runs/<run>/... which in turn reads only the
 * files a run persisted. The page polls the run summary and, when its
 * `version` (newest file mtime) changes, refetches and redraws the open tab,
 * keeping filters and the selection. That is how a run in progress is
 * followed.
 */

"use strict";

const POLL_MS = 5000;

const S = {
  runs: [],
  run: null,
  summary: null,
  rounds: [],
  cands: [],
  lineage: { nodes: [], edges: [] },
  grid: { k: 0, clusters: { words: {} }, tasks: [] },
  tab: "overview",
  version: null,
  filters: { task: "", diagnosis: "", origin: "", operator: "", round: "", q: "", sort: "sep" },
  lineageFilters: { task: "", onlyLineages: false },
  compare: null, // pair id of the child being compared with its parents
  selected: null, // {kind: "cand"|"pair", id}
  clips: true, // show transition clips where they exist
  jobs: null, // the clip queue for the current run
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : (x >= 0 && d === 3 ? "+" : "") + Number(x).toFixed(d));
const pct = (a, b) => (b ? Math.round((100 * a) / b) + "%" : "–");
const fileUrl = (path) => `/files/${encodeURIComponent(S.run)}/${path.split("/").map(encodeURIComponent).join("/")}`;

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// sep_min in roughly [-0.15, +0.15] onto red -> grey -> green.
function fitnessColor(v) {
  if (v === null || v === undefined) return "#cfcfcb";
  const t = Math.max(-1, Math.min(1, v / 0.15));
  const h = t < 0 ? 4 : 130;
  const s = Math.round(20 + 55 * Math.abs(t));
  const l = Math.round(78 - 30 * Math.abs(t));
  return `hsl(${h} ${s}% ${l}%)`;
}

const candById = () => new Map(S.cands.map((c) => [c.uid, c]));
const nodeById = () => new Map(S.lineage.nodes.map((n) => [n.id, n]));

function mediaHTML(c, cls = "media") {
  if (S.clips && c.animations && c.animations.length) {
    const clips = c.animations.map((a) =>
      `<figure><video src="${fileUrl(a.path)}" autoplay loop muted playsinline preload="metadata"></video><figcaption>→ ${esc(a.slot)}</figcaption></figure>`
    );
    return `<div class="${cls}">${clips.join("")}</div>`;
  }
  const items = c.media.map((m) =>
    m.kind === "audio"
      ? `<figure><audio controls preload="none" src="${fileUrl(m.path)}"></audio><figcaption>${esc(m.slot)}</figcaption></figure>`
      : `<figure><img loading="lazy" src="${fileUrl(m.path)}" alt="${esc(m.slot)}"><figcaption>${esc(m.slot)}</figcaption></figure>`
  );
  return `<div class="${cls}">${items.join("")}</div>`;
}

function promptsHTML(c) {
  const slots = c.slots.length ? c.slots : c.prompts.map((_, i) => `slot ${i}`);
  return c.prompts.map((p, i) => `<div><span class="muted">${esc(slots[i] ?? i)}:</span> ${esc(p)}</div>`).join("") +
    `<div class="muted">style: ${esc(c.style || "(none)")}</div>`;
}

// ---------------------------------------------------------------------------
// Data loading and polling
// ---------------------------------------------------------------------------

async function loadRuns() {
  S.runs = await getJSON("/api/runs");
  const sel = $("#run-select");
  sel.innerHTML = S.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.name)} · ${esc(r.proposer)} · ${r.rounds_done}/${r.rounds_planned}</option>`).join("");
  if (!S.run && S.runs.length) {
    const fromHash = location.hash.slice(1).split("/")[0];
    S.run = S.runs.some((r) => r.name === fromHash) ? fromHash : S.runs[0].name;
  }
  sel.value = S.run ?? "";
}

async function loadRun() {
  if (!S.run) return;
  const base = `/api/runs/${encodeURIComponent(S.run)}`;
  const [summary, rounds, cands] = await Promise.all([getJSON(base), getJSON(base + "/rounds"), getJSON(base + "/candidates")]);
  S.summary = summary;
  S.rounds = rounds;
  S.cands = cands;
  if (summary.has_archive) {
    [S.lineage, S.grid] = await Promise.all([getJSON(base + "/lineage"), getJSON(base + "/grid")]);
  } else {
    S.lineage = { nodes: [], edges: [] };
    S.grid = { k: 0, clusters: { words: {} }, tasks: [] };
  }
  S.version = summary.version;
  S.jobs = await getJSON(base + "/jobs").catch(() => null);
}

async function poll() {
  try {
    if (!S.run) return;
    const s = await getJSON(`/api/runs/${encodeURIComponent(S.run)}`);
    if (s.version !== S.version) {
      await loadRun();
      render();
    } else {
      S.summary = s;
      renderStatus();
    }
    $("#poll").textContent = `polled ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $("#poll").textContent = `poll failed: ${e.message}`;
  }
}

function jobStateOf(uid) {
  const j = S.jobs;
  if (!j) return null;
  if (j.running === uid) return "rendering…";
  if (j.pending.includes(uid)) return `queued (${j.pending.indexOf(uid) + 1})`;
  if (j.failed[uid]) return `failed: ${j.failed[uid]}`;
  return null;
}

function renderJobs() {
  const j = S.jobs, el = $("#jobs");
  if (!j || (!j.running && !j.pending.length && !Object.keys(j.failed).length)) { el.textContent = ""; return; }
  const parts = [];
  if (j.running) parts.push("1 rendering");
  if (j.pending.length) parts.push(`${j.pending.length} queued`);
  if (Object.keys(j.failed).length) parts.push(`${Object.keys(j.failed).length} failed`);
  el.textContent = `clips: ${parts.join(", ")}`;
}

let jobsWereBusy = false;
async function pollJobs() {
  if (!S.run) return;
  try {
    S.jobs = await getJSON(`/api/runs/${encodeURIComponent(S.run)}/jobs`);
  } catch (e) { return; }
  renderJobs();
  const busy = Boolean(S.jobs.running || S.jobs.pending.length);
  if (jobsWereBusy && !busy) await poll(); // clips landed: pick up the new files now
  jobsWereBusy = busy;
}

async function requestClip(uid) {
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(S.run)}/animate/${encodeURIComponent(uid)}`, { method: "POST" });
    if (!r.ok && r.status !== 409) throw new Error(`${r.status}`);
  } catch (e) {
    $("#jobs").textContent = `clip request failed: ${e.message}`;
  }
}

async function switchRun(name) {
  S.run = name;
  $("#run-select").value = name;
  S.selected = null;
  S.compare = null;
  location.hash = `${name}/${S.tab}`;
  await loadRun();
  render();
}

// ---------------------------------------------------------------------------
// Chrome: status line and tabs
// ---------------------------------------------------------------------------

function renderStatus() {
  const s = S.summary;
  if (!s) return;
  let txt;
  if (s.finished) txt = `finished · ${s.rounds_done} rounds`;
  else if (s.in_progress) txt = `<span class="live">live</span> · round ${s.in_progress.index}: ${s.in_progress.scored}/${s.in_progress.planned} scored`;
  else txt = `<span class="live">waiting</span> · ${s.rounds_done}/${s.rounds_planned} rounds`;
  $("#run-status").innerHTML = txt;
  $("#clips-toggle").hidden = s.track !== "image";
  renderJobs();
}

function renderTabs() {
  const tabs = [["overview", "Overview"], ["rounds", "Rounds"], ["candidates", "Candidates"]];
  if (S.summary?.has_archive) tabs.push(["lineage", "Lineage"], ["archive", "Archive"], ["compare", "Compare"]);
  if (!tabs.some(([k]) => k === S.tab)) S.tab = "overview";
  $("#tabs").innerHTML = tabs.map(([k, label]) => `<button data-tab="${k}" class="${k === S.tab ? "active" : ""}">${label}</button>`).join("");
}

function render() {
  renderStatus();
  renderTabs();
  const main = $("#main");
  if (!S.summary) { main.innerHTML = `<div class="empty">No runs under the runs directory.</div>`; return; }
  ({ overview: renderOverview, rounds: renderRounds, candidates: renderCandidates, lineage: renderLineage, archive: renderArchive, compare: renderCompare })[S.tab](main);
  renderDetail();
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

function renderOverview(main) {
  const s = S.summary;
  const stat = (k, v) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  const best = s.best;
  main.innerHTML = `
    <div class="cards">
      ${stat("proposer", esc(s.proposer))}
      ${stat("track", esc(s.track))}
      ${stat("tasks", esc(s.tasks.join(", ")) || "–")}
      ${stat("rounds", `${s.rounds_done} / ${s.rounds_planned}`)}
      ${stat("candidates", s.n_candidates)}
      ${stat("held", `${s.n_held} <span class="muted" style="font-size:13px">(${pct(s.n_held, s.n_candidates)})</span>`)}
      ${stat("best sep_min", best ? fmt(best.sep_min) : "–")}
    </div>
    ${best ? `<h2>Best candidate</h2><div class="grid" style="max-width:520px">${candCard(best)}</div>` : ""}
    <h2>Config</h2>
    <pre class="config">${esc(JSON.stringify(s.config, null, 2))}</pre>`;
  bindCards(main);
}

// ---------------------------------------------------------------------------
// Rounds: line charts and a table
// ---------------------------------------------------------------------------

const PALETTE = ["#2b5fa3", "#c2571a", "#2e7d32", "#7b3fa0", "#b5851a", "#0d8a8a", "#a3325f"];

/* series: [{name, values: [{x, y}], color}] ; opts: {y0?, y1?, width, height, yfmt} */
function lineChart(series, opts = {}) {
  const W = opts.width ?? 440, H = opts.height ?? 200, ml = 46, mr = 12, mt = 10, mb = 26;
  const xs = series.flatMap((s) => s.values.map((v) => v.x));
  const ys = series.flatMap((s) => s.values.map((v) => v.y)).filter((y) => y !== null && y !== undefined && !Number.isNaN(y));
  if (!xs.length || !ys.length) return `<div class="empty">no data yet</div>`;
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = opts.y0 ?? Math.min(...ys), y1 = opts.y1 ?? Math.max(...ys);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.08;
  if (opts.y0 === undefined) y0 -= pad;
  if (opts.y1 === undefined) y1 += pad;
  const X = (x) => ml + ((x - x0) / (x1 - x0)) * (W - ml - mr);
  const Y = (y) => mt + (1 - (y - y0) / (y1 - y0)) * (H - mt - mb);
  const yfmt = opts.yfmt ?? ((v) => v.toFixed(2));
  const yticks = 4, out = [];
  out.push(`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`);
  for (let i = 0; i <= yticks; i++) {
    const y = y0 + ((y1 - y0) * i) / yticks;
    out.push(`<line class="axis" x1="${ml}" x2="${W - mr}" y1="${Y(y)}" y2="${Y(y)}"/><text x="${ml - 4}" y="${Y(y) + 4}" text-anchor="end">${yfmt(y)}</text>`);
  }
  if (y0 < 0 && y1 > 0) out.push(`<line x1="${ml}" x2="${W - mr}" y1="${Y(0)}" y2="${Y(0)}" stroke="#999" stroke-dasharray="3 3"/>`);
  for (let x = x0; x <= x1; x++) out.push(`<text x="${X(x)}" y="${H - 8}" text-anchor="middle">${x}</text>`);
  series.forEach((s, i) => {
    const color = s.color ?? PALETTE[i % PALETTE.length];
    const pts = s.values.filter((v) => v.y !== null && v.y !== undefined && !Number.isNaN(v.y));
    out.push(`<polyline class="series" stroke="${color}" points="${pts.map((v) => `${X(v.x)},${Y(v.y)}`).join(" ")}"/>`);
    pts.forEach((v) => out.push(`<circle class="pt" cx="${X(v.x)}" cy="${Y(v.y)}" r="${v.r ?? 3}" fill="${color}"><title>${esc(s.name)} · round ${v.x}: ${esc(v.label ?? yfmt(v.y))}</title></circle>`));
  });
  out.push(`</svg>`);
  const legend = `<div class="legend">${series.map((s, i) => `<span><i style="background:${s.color ?? PALETTE[i % PALETTE.length]}"></i>${esc(s.name)}</span>`).join("")}</div>`;
  return legend + out.join("");
}

function renderRounds(main) {
  const rs = S.rounds;
  if (!rs.length) { main.innerHTML = `<div class="empty">no round scored yet</div>`; return; }
  const val = (k) => rs.map((r) => ({ x: r.index, y: r[k] }));
  const sepChart = lineChart([
    { name: "best sep_min", values: val("best") },
    { name: "mean", values: val("mean") },
    { name: "median", values: val("median") },
  ]);
  const heldChart = lineChart([{ name: "held / candidates", values: rs.map((r) => ({ x: r.index, y: r.n ? r.n_ok / r.n : null, label: `${r.n_ok}/${r.n}` })) }], { y0: 0, y1: 1, yfmt: (v) => Math.round(v * 100) + "%" });
  const charts = [["sep_min per round", sepChart], ["illusion held", heldChart]];

  const hasSearch = rs.some((r) => r.search);
  if (hasSearch) {
    const cov = (key) => rs.map((r) => ({ x: r.index, y: r.search ? Object.values(r.search.coverage ?? {}).reduce((a, c) => a + (c[key] ?? 0), 0) : null }));
    charts.push(["archive coverage (cells, all tasks)", lineChart([{ name: "filled", values: cov("filled") }, { name: "held", values: cov("held") }], { y0: 0, yfmt: (v) => v.toFixed(0) })]);
    const rho = rs.map((r) => ({ x: r.index, y: r.surrogate ? r.surrogate.rho : null, r: r.surrogate?.mode === "ucb" ? 5 : 3, label: r.surrogate ? `rho ${r.surrogate.rho.toFixed(2)} · ${r.surrogate.mode} · n=${r.surrogate.n_train}` : "" }));
    charts.push(["surrogate leave-one-out rho (big dot = drove selection)", lineChart([{ name: "rho", values: rho }], { y0: -1, y1: 1 })]);
    const ops = [...new Set(rs.flatMap((r) => Object.keys(r.search?.operators ?? {})))].sort();
    charts.push(["operator mean reward", lineChart(ops.map((op) => ({ name: op, values: rs.map((r) => ({ x: r.index, y: r.search?.operators?.[op] ?? null })) })), { y0: 0 })]);
    const rej = rs.map((r) => ({ x: r.index, y: r.search?.dedup?.n_checked ? r.search.dedup.n_rejected / r.search.dedup.n_checked : null, label: r.search?.dedup ? `${r.search.dedup.n_rejected}/${r.search.dedup.n_checked}` : "" }));
    charts.push(["near-duplicate children rejected", lineChart([{ name: "rejected / checked", values: rej }], { y0: 0, yfmt: (v) => Math.round(v * 100) + "%" })]);
  }

  const rows = rs.map((r) => `<tr>
      <td class="num">${r.index}</td><td class="num">${r.n}${r.planned > r.n ? ` <span class="muted">/ ${r.planned}</span>` : ""}</td>
      <td class="num">${r.n_ok}</td><td class="num">${fmt(r.best)}</td><td class="num">${fmt(r.mean)}</td>
      <td>${Object.entries(r.origins).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}</td>
      <td>${Object.entries(r.operators).map(([k, v]) => `${esc(k)} ${v}`).join(", ") || "–"}</td>
      <td>${r.surrogate ? `${r.surrogate.rho.toFixed(2)} ${r.surrogate.mode}` : "–"}</td>
      <td>${r.search?.dedup ? `${r.search.dedup.n_rejected}/${r.search.dedup.n_checked}` : "–"}</td>
    </tr>`).join("");
  main.innerHTML = `
    <div class="charts">${charts.map(([t, c]) => `<div class="chart"><h3>${t}</h3>${c}</div>`).join("")}</div>
    <h2>Rounds</h2>
    <div class="table-wrap"><table><thead><tr><th>round</th><th>n</th><th>held</th><th>best</th><th>mean</th><th>origins</th><th>operators</th><th>surrogate</th><th>dedup</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

function candCard(c) {
  const sel = S.selected?.kind === "cand" && S.selected.id === c.uid ? " selected" : "";
  return `<div class="card${sel}" data-uid="${esc(c.uid)}">
    ${mediaHTML(c)}
    <div class="prompts">${promptsHTML(c)}</div>
    <div class="meta">
      <span class="score" title="sep_min">${fmt(c.sep_min)}</span>
      <span title="J">J ${c.j === null ? "–" : Number(c.j).toFixed(3)}</span>
      <span class="badge ${c.ok ? "ok" : "bad"}">${esc(c.diagnosis)}</span>
      <span>${esc(c.task)}</span><span>r${c.round}</span><span>seed ${c.seed}</span>
      <span>${esc(c.operator ?? c.origin)}</span>
    </div></div>`;
}

function bindCards(root) {
  root.querySelectorAll(".card[data-uid]").forEach((el) => el.addEventListener("click", () => select("cand", el.dataset.uid)));
}

function options(values, current, blank = "all") {
  return `<option value="">${blank}</option>` + values.map((v) => `<option value="${esc(v)}" ${String(v) === String(current) ? "selected" : ""}>${esc(v)}</option>`).join("");
}

function filteredCands() {
  const f = S.filters, q = f.q.trim().toLowerCase();
  let rows = S.cands.filter((c) =>
    (!f.task || c.task === f.task) &&
    (!f.diagnosis || (f.diagnosis === "ok" ? c.ok : !c.ok)) &&
    (!f.origin || c.origin === f.origin) &&
    (!f.operator || c.operator === f.operator) &&
    (f.round === "" || String(c.round) === f.round) &&
    (!q || c.prompts.join(" ").toLowerCase().includes(q) || (c.style ?? "").toLowerCase().includes(q))
  );
  const key = { sep: (c) => -(c.sep_min ?? -Infinity), j: (c) => -(c.j ?? -1), round: (c) => -c.round, uid: (c) => c.uid }[f.sort];
  rows = rows.slice().sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
  return rows;
}

function renderCandidates(main) {
  const f = S.filters;
  const uniq = (k) => [...new Set(S.cands.map((c) => c[k]).filter((v) => v !== null && v !== undefined && v !== ""))].sort();
  const rows = filteredCands();
  main.innerHTML = `
    <div class="controls">
      <label>task <select data-f="task">${options(uniq("task"), f.task)}</select></label>
      <label>diagnosis <select data-f="diagnosis"><option value="">all</option><option value="ok" ${f.diagnosis === "ok" ? "selected" : ""}>held</option><option value="lost" ${f.diagnosis === "lost" ? "selected" : ""}>lost</option></select></label>
      <label>origin <select data-f="origin">${options(uniq("origin"), f.origin)}</select></label>
      <label>operator <select data-f="operator">${options(uniq("operator"), f.operator)}</select></label>
      <label>round <select data-f="round">${options(uniq("round"), f.round)}</select></label>
      <label>search <input data-f="q" value="${esc(f.q)}" placeholder="prompt or style"></label>
      <label>sort <select data-f="sort">${["sep", "j", "round", "uid"].map((k) => `<option value="${k}" ${f.sort === k ? "selected" : ""}>${k === "sep" ? "sep_min" : k}</option>`).join("")}</select></label>
      ${S.summary.track === "image" ? `<button class="action" id="animate-held" title="queue a transition clip for every held candidate that has none">render clips for held</button>` : ""}
      <span class="muted">${rows.length} / ${S.cands.length}</span>
    </div>
    <div class="grid">${rows.map(candCard).join("") || `<div class="empty">nothing matches</div>`}</div>`;
  main.querySelectorAll("[data-f]").forEach((el) => el.addEventListener(el.tagName === "INPUT" ? "input" : "change", () => { S.filters[el.dataset.f] = el.value; renderCandidates(main); }));
  const heldBtn = $("#animate-held", main);
  if (heldBtn) heldBtn.addEventListener("click", async () => {
    heldBtn.disabled = true;
    for (const c of S.cands) if (c.ok && !c.animations.length) await requestClip(c.uid);
    await pollJobs();
  });
  bindCards(main);
}

// ---------------------------------------------------------------------------
// Lineage: a layered DAG, one column per round the pair first appeared in
// ---------------------------------------------------------------------------

function renderLineage(main) {
  const lf = S.lineageFilters;
  const tasks = [...new Set(S.lineage.nodes.map((n) => n.task))].sort();
  let nodes = S.lineage.nodes.filter((n) => !lf.task || n.task === lf.task);
  const hasFamily = new Set(S.lineage.edges.flatMap((e) => [e.source, e.target]));
  if (lf.onlyLineages) nodes = nodes.filter((n) => hasFamily.has(n.id));
  const ids = new Set(nodes.map((n) => n.id));
  const edges = S.lineage.edges.filter((e) => ids.has(e.source) && ids.has(e.target));

  const colOf = (n) => (n.first_round === null ? 0 : n.first_round);
  const cols = new Map();
  nodes.forEach((n) => { const c = colOf(n); if (!cols.has(c)) cols.set(c, []); cols.get(c).push(n); });
  const colKeys = [...cols.keys()].sort((a, b) => a - b);
  const COLW = 200, ROWH = 24, top = 30, left = 40;
  const pos = new Map();
  let maxRows = 0;
  colKeys.forEach((c, ci) => {
    const list = cols.get(c).slice().sort((a, b) => (b.fitness ?? -9) - (a.fitness ?? -9));
    maxRows = Math.max(maxRows, list.length);
    list.forEach((n, ri) => pos.set(n.id, { x: left + ci * COLW + 40, y: top + ri * ROWH + 12 }));
  });
  const W = left + colKeys.length * COLW + 80, H = top + maxRows * ROWH + 20;
  const selId = S.selected?.kind === "pair" ? S.selected.id : null;
  const related = new Set();
  if (selId) edges.forEach((e) => { if (e.source === selId || e.target === selId) { related.add(e.source); related.add(e.target); } });

  const svg = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`];
  colKeys.forEach((c, ci) => svg.push(`<text class="col-label" x="${left + ci * COLW + 40}" y="14" text-anchor="middle">round ${c}</text>`));
  edges.forEach((e) => {
    const a = pos.get(e.source), b = pos.get(e.target);
    const hl = selId && (e.source === selId || e.target === selId) ? " hl" : "";
    const mx = (a.x + b.x) / 2;
    svg.push(`<path class="edge${hl}" d="M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}"><title>${esc(e.operator)}</title></path>`);
    if (hl) svg.push(`<text class="op-label" x="${mx}" y="${(a.y + b.y) / 2 - 3}" text-anchor="middle">${esc(e.operator)}</text>`);
  });
  nodes.forEach((n) => {
    const p = pos.get(n.id);
    const r = 4 + Math.min(4, (n.n_seeds ?? 1) - 1) * 1.5;
    const cls = ["node", n.elite ? "elite" : "", n.id === selId ? "selected" : "", n.missing ? "missing" : ""].join(" ");
    const title = `${n.prompts.join(" / ")}\nstyle: ${n.style || "(none)"}\n${n.operator ?? "bootstrap"} · fitness ${fmt(n.fitness)} · ${n.n_ok}/${n.n_seeds} held${n.elite ? " · elite" : ""}`;
    svg.push(`<circle class="${cls}" data-id="${n.id}" cx="${p.x}" cy="${p.y}" r="${r}" fill="${fitnessColor(n.fitness)}" opacity="${selId && !related.has(n.id) && n.id !== selId ? 0.35 : 1}"><title>${esc(title)}</title></circle>`);
    if (n.id === selId || related.has(n.id) || (n.elite && !selId)) {
      svg.push(`<text x="${p.x + r + 4}" y="${p.y + 4}" style="font-size:10px">${esc(n.prompts.join(" / ").slice(0, 34))}</text>`);
    }
  });
  svg.push("</svg>");
  main.innerHTML = `
    <div class="controls">
      <label>task <select data-lf="task">${options(tasks, lf.task)}</select></label>
      <label><input type="checkbox" data-lf="onlyLineages" ${lf.onlyLineages ? "checked" : ""}> only pairs with a parent or a child</label>
      <span class="muted">${nodes.length} pairs · ${edges.length} edges · colour = fitness (mean sep_min), gold ring = elite, size = seeds. Click a node.</span>
    </div>
    <div class="lineage-wrap">${svg.join("")}</div>`;
  main.querySelectorAll("[data-lf]").forEach((el) => el.addEventListener("change", () => { S.lineageFilters[el.dataset.lf] = el.type === "checkbox" ? el.checked : el.value; renderLineage(main); }));
  main.querySelectorAll("circle.node").forEach((el) => el.addEventListener("click", () => select("pair", el.dataset.id)));
}

// ---------------------------------------------------------------------------
// Archive: the MAP-Elites grid per task
// ---------------------------------------------------------------------------

function renderArchive(main) {
  const g = S.grid;
  if (!g.tasks.length) { main.innerHTML = `<div class="empty">the archive is empty</div>`; return; }
  const words = (i) => (g.clusters.words[String(i)] ?? []).join(", ");
  const parts = g.tasks.map((t) => {
    const cells = t.cells;
    if (t.dims !== 2) {
      const rows = Object.values(cells).map((c) => `<tr><td class="mono">${c.cell.join(",")}</td><td>${c.elite ? esc(c.elite.prompts.join(" / ")) : "–"}</td><td>${c.elite ? esc(c.elite.style) : ""}</td><td class="num">${c.elite ? fmt(c.elite.fitness) : "–"}</td><td class="num">${c.n_pairs}</td></tr>`).join("");
      return `<h2>${esc(t.task)} <span class="muted">(${t.dims} slots, ${Object.keys(cells).length} cells)</span></h2><table><thead><tr><th>cell</th><th>elite</th><th>style</th><th>fitness</th><th>pairs</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
    const k = g.k;
    const head = `<tr><th></th>${[...Array(k).keys()].map((j) => `<th title="${esc(words(j))}">${j}<br><span class="words">${esc(words(j).slice(0, 26))}</span></th>`).join("")}</tr>`;
    const body = [...Array(k).keys()].map((i) => `<tr><th title="${esc(words(i))}">${i} <span class="words">${esc(words(i).slice(0, 26))}</span></th>` +
      [...Array(k).keys()].map((j) => {
        const c = cells[`${i},${j}`];
        if (!c) return `<td class="empty">·</td>`;
        const e = c.elite;
        const title = e ? `${e.prompts.join(" / ")}\nstyle: ${e.style || "(none)"}\nfitness ${fmt(e.fitness)} · ${e.n_ok}/${e.n_seeds} held · ${c.n_pairs} pairs, ${c.n_evaluations} evaluations` : `${c.n_pairs} pairs, no elite`;
        return `<td style="background:${fitnessColor(e?.fitness)}" data-id="${e?.id ?? ""}" title="${esc(title)}">${e ? fmt(e.fitness) : "–"}<br><span class="muted" style="font-size:10px">${c.n_pairs}p</span></td>`;
      }).join("") + "</tr>").join("");
    const filled = Object.keys(cells).length;
    return `<div class="heat"><h2>${esc(t.task)} <span class="muted">${filled} / ${k * k} cells</span></h2>
      <div class="muted" style="font-size:12px;margin-bottom:4px">rows: cluster of the first slot · columns: cluster of the second · colour = elite fitness · Np = pairs in the cell</div>
      <table>${head}${body}</table></div>`;
  });
  main.innerHTML = parts.join("") + `<h3>Clusters</h3><table><thead><tr><th>id</th><th>size</th><th>words</th></tr></thead><tbody>${Object.keys(g.clusters.words).map((i) => `<tr><td>${i}</td><td class="num">${g.clusters.sizes?.[i] ?? ""}</td><td>${esc(words(i))}</td></tr>`).join("")}</tbody></table>`;
  main.querySelectorAll("td[data-id]").forEach((el) => { if (el.dataset.id) el.addEventListener("click", () => select("pair", el.dataset.id)); });
}

// ---------------------------------------------------------------------------
// Compare: a child pair beside its parents
// ---------------------------------------------------------------------------

function bestEval(n) {
  const byId = candById();
  const evs = n.evaluations.slice().sort((a, b) => b.sep_min - a.sep_min);
  for (const e of evs) { const c = byId.get(e.uid); if (c) return c; }
  return null;
}

function pairPane(n, other, label) {
  const c = bestEval(n);
  const prompts = n.prompts.map((p, i) => {
    const changed = other && other.prompts[i] !== p;
    return `<div class="${changed ? "changed" : ""}"><span class="muted">${esc(c?.slots?.[i] ?? i)}:</span> ${esc(p)}</div>`;
  }).join("");
  const styleChanged = other && other.style !== n.style;
  return `<div class="pane">
    <h3>${label} <span class="linkish" data-pair="${n.id}">${n.id}</span></h3>
    ${c ? mediaHTML(c, "detail-media") : `<div class="muted">no scored seed in this run</div>`}
    <div style="margin-top:6px">${prompts}<div class="${styleChanged ? "changed" : ""}"><span class="muted">style:</span> ${esc(n.style || "(none)")}</div></div>
    <div class="meta" style="margin-top:6px">fitness <b>${fmt(n.fitness)}</b> · ${n.n_ok}/${n.n_seeds} held · ${esc(n.operator ?? "bootstrap")} · round ${n.first_round}${n.elite ? ' · <span class="badge elite">elite</span>' : ""}</div>
  </div>`;
}

function renderCompare(main) {
  const byId = nodeById();
  const children = S.lineage.nodes.filter((n) => n.parents.length).sort((a, b) => (b.fitness ?? -9) - (a.fitness ?? -9));
  if (!children.length) { main.innerHTML = `<div class="empty">no evolved child yet: every pair is a bootstrap</div>`; return; }
  if (!S.compare || !byId.has(S.compare)) S.compare = children[0].id;
  const child = byId.get(S.compare);
  const parents = child.parents.map((id) => byId.get(id)).filter(Boolean);
  const gain = parents.length && child.fitness !== null ? child.fitness - Math.max(...parents.map((p) => p.fitness ?? -Infinity)) : null;
  main.innerHTML = `
    <div class="controls">
      <label>child <select id="compare-select">${children.map((n) => `<option value="${n.id}" ${n.id === child.id ? "selected" : ""}>${esc(n.operator)} · ${esc(n.prompts.join(" / "))} · ${fmt(n.fitness)}</option>`).join("")}</select></label>
      ${gain !== null ? `<span>improvement over best parent: <b class="delta ${gain >= 0 ? "up" : "down"}">${fmt(gain)}</b></span>` : ""}
    </div>
    <div class="compare">
      ${parents.map((p, i) => pairPane(p, child, parents.length > 1 ? `parent ${i + 1}` : "parent")).join("")}
      ${pairPane(child, parents[0], `child · ${esc(child.operator)}`)}
    </div>`;
  $("#compare-select").addEventListener("change", (e) => { S.compare = e.target.value; renderCompare(main); });
  main.querySelectorAll("[data-pair]").forEach((el) => el.addEventListener("click", () => select("pair", el.dataset.pair)));
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

function select(kind, id) {
  S.selected = S.selected?.kind === kind && S.selected.id === id ? null : { kind, id };
  render();
}

function matrixHTML(c) {
  if (!c.scores) return "";
  const slots = c.slots.length ? c.slots : c.prompts.map((_, i) => `view ${i}`);
  const head = `<tr><th>view \\ prompt</th>${c.prompts.map((p) => `<th>${esc(p.slice(0, 28))}</th>`).join("")}<th>p</th><th>sep</th></tr>`;
  const body = c.scores.map((row, i) => `<tr><th>${esc(slots[i])}</th>${row.map((v, j) => `<td class="num ${i === j ? "diag" : ""}">${Number(v).toFixed(3)}</td>`).join("")}<td class="num">${c.p?.[i] !== undefined ? Number(c.p[i]).toFixed(3) : "–"}</td><td class="num ${c.holds[i] ? "" : "badge bad"}">${fmt(c.sep[i])}</td></tr>`).join("");
  return `<table class="matrix">${head}${body}</table>`;
}

function clipControls(c) {
  if (S.summary.track !== "image") return "";
  if (c.animations.length) return `<div class="muted" style="font-size:12px;margin-top:4px">clips: ${c.animations.map((a) => `→ ${esc(a.slot)}`).join(", ")} · <span class="linkish" data-toggle-clips>${S.clips ? "show stills" : "show clips"}</span></div>`;
  const job = jobStateOf(c.uid);
  if (job) return `<div class="muted" style="font-size:12px;margin-top:4px">clip ${esc(job)}</div>`;
  return `<div style="margin-top:6px"><button class="action" data-animate="${esc(c.uid)}">render transition clip</button></div>`;
}

function candDetail(c) {
  const byNode = nodeById();
  const parents = c.parents.map((id) => `<span class="linkish" data-pair="${esc(id)}">${esc(byNode.get(id)?.prompts.join(" / ") ?? id)}</span>`).join(", ");
  const sur = c.surrogate && Object.keys(c.surrogate).length
    ? `<div class="muted" style="font-size:12px">surrogate: ${Object.entries(c.surrogate).map(([k, v]) => `${esc(k)} ${typeof v === "number" ? v.toFixed(3) : esc(v)}`).join(" · ")} · selection ${esc(c.selection)}</div>`
    : "";
  return `
    <h2>${esc(c.task)} <span class="mono muted">${esc(c.uid)}</span></h2>
    ${mediaHTML(c, "detail-media")}
    ${clipControls(c)}
    ${c.sample ? `<div class="muted" style="font-size:12px;margin-top:4px"><a href="${fileUrl(c.sample)}" target="_blank">sample image</a> · <a href="${fileUrl(`round_${String(c.round).padStart(3, "0")}/${c.uid}/prompt.txt`)}" target="_blank">prompt card</a></div>` : ""}
    <div style="margin-top:8px">${promptsHTML(c)}</div>
    <div class="meta" style="margin-top:6px">
      <span class="badge ${c.ok ? "ok" : "bad"}">${esc(c.diagnosis)}</span>
      sep_min <b>${fmt(c.sep_min)}</b> · J ${c.j === null ? "–" : Number(c.j).toFixed(4)} · A ${c.alignment === null ? "–" : Number(c.alignment).toFixed(3)} · C ${c.concealment === null ? "–" : Number(c.concealment).toFixed(3)}
    </div>
    <div class="muted" style="font-size:12px">round ${c.round} · seed ${c.seed} · ${esc(c.origin)}${c.operator ? ` · ${esc(c.operator)}` : ""}${c.seconds ? ` · ${Number(c.seconds).toFixed(1)} s` : ""}</div>
    ${parents ? `<div style="font-size:12px">parents: ${parents}</div>` : ""}
    ${sur}
    <h3>CLIP matrix</h3><div class="table-wrap">${matrixHTML(c)}</div>
    ${c.captions.length ? `<h3>Captions</h3>${c.captions.map((t, i) => `<div><span class="muted">${esc(c.slots[i] ?? i)}:</span> ${esc(t)}</div>`).join("")}` : ""}
    ${S.summary.has_archive ? `<div style="margin-top:10px"><span class="linkish" data-pair="${esc(c.pair)}">open this pair in the lineage</span></div>` : ""}`;
}

function pairDetail(n) {
  const byId = candById(), byNode = nodeById();
  const children = S.lineage.nodes.filter((m) => m.parents.includes(n.id));
  const evs = n.evaluations.slice().sort((a, b) => a.round_index - b.round_index).map((e) => {
    const c = byId.get(e.uid);
    return `<tr><td class="num">${e.round_index}</td><td class="num">${e.seed}</td><td>${esc(e.origin)}</td><td class="num">${fmt(e.sep_min)}</td><td>${e.ok ? '<span class="badge ok">ok</span>' : `<span class="badge bad">lost ${esc((e.lost ?? []).join(","))}</span>`}</td><td>${c ? `<span class="linkish" data-cand="${esc(e.uid)}">${esc(e.uid)}</span>` : `<span class="mono muted">${esc(e.uid)}</span>`}</td></tr>`;
  }).join("");
  const best = bestEval(n);
  const link = (m) => `<span class="linkish" data-pair="${m.id}">${esc(m.prompts.join(" / "))}${m.operator ? ` <span class="muted">(${esc(m.operator)})</span>` : ""}</span>`;
  return `
    <h2>${esc(n.task)} pair <span class="mono muted">${n.id}</span>${n.elite ? ' <span class="badge elite">elite</span>' : ""}</h2>
    ${best ? mediaHTML(best, "detail-media") : ""}
    <div style="margin-top:8px">${n.prompts.map((p, i) => `<div><span class="muted">${esc(best?.slots?.[i] ?? i)}:</span> ${esc(p)}</div>`).join("")}<div class="muted">style: ${esc(n.style || "(none)")}</div></div>
    <div class="meta" style="margin-top:6px">fitness <b>${fmt(n.fitness)}</b> · ${n.n_ok}/${n.n_seeds} held · cell ${n.cell.join(",")} · first round ${n.first_round} · ${esc(n.operator ?? "bootstrap")}</div>
    ${n.parents.length ? `<h3>Parents</h3>${n.parents.map((id) => byNode.get(id)).filter(Boolean).map(link).join("<br>")}<div style="margin-top:4px"><span class="linkish" data-compare="${n.id}">compare with parents</span></div>` : ""}
    ${children.length ? `<h3>Children (${children.length})</h3>${children.map(link).join("<br>")}` : ""}
    <h3>Evaluations</h3>
    <table><thead><tr><th>round</th><th>seed</th><th>origin</th><th>sep_min</th><th></th><th>candidate</th></tr></thead><tbody>${evs}</tbody></table>`;
}

function renderDetail() {
  const panel = $("#detail");
  const sel = S.selected;
  if (!sel) { panel.hidden = true; panel.innerHTML = ""; return; }
  let body = "";
  if (sel.kind === "cand") { const c = candById().get(sel.id); body = c ? candDetail(c) : ""; }
  else { const n = nodeById().get(sel.id); body = n ? pairDetail(n) : ""; }
  if (!body) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.innerHTML = `<button class="close" title="close">×</button>${body}`;
  $(".close", panel).addEventListener("click", () => select(sel.kind, sel.id));
  panel.querySelectorAll("[data-pair]").forEach((el) => el.addEventListener("click", () => { if (S.tab !== "lineage" && S.tab !== "archive" && S.tab !== "compare") S.tab = "lineage"; S.selected = { kind: "pair", id: el.dataset.pair }; render(); }));
  panel.querySelectorAll("[data-cand]").forEach((el) => el.addEventListener("click", () => { S.selected = { kind: "cand", id: el.dataset.cand }; render(); }));
  panel.querySelectorAll("[data-compare]").forEach((el) => el.addEventListener("click", () => { S.compare = el.dataset.compare; S.tab = "compare"; S.selected = null; render(); }));
  panel.querySelectorAll("[data-animate]").forEach((el) => el.addEventListener("click", async () => { el.disabled = true; await requestClip(el.dataset.animate); await pollJobs(); render(); }));
  panel.querySelectorAll("[data-toggle-clips]").forEach((el) => el.addEventListener("click", () => { S.clips = !S.clips; $("#clips-toggle input").checked = S.clips; render(); }));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  $("#run-select").addEventListener("change", (e) => switchRun(e.target.value));
  $("#tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (!b) return;
    S.tab = b.dataset.tab;
    location.hash = `${S.run}/${S.tab}`;
    render();
  });
  await loadRuns();
  const tab = location.hash.slice(1).split("/")[1];
  if (tab) S.tab = tab;
  await loadRun();
  render();
  setInterval(async () => { await poll(); }, POLL_MS);
  setInterval(pollJobs, 2000);
  $("#clips-toggle input").addEventListener("change", (e) => { S.clips = e.target.checked; render(); });
  setInterval(loadRuns, POLL_MS * 6);
}

boot().catch((e) => { $("#main").innerHTML = `<div class="empty">${esc(e.message)}</div>`; });

// For a console or a driver: the state and the entry points, nothing else.
window.ava = { S, render, select, switchRun, poll, pollJobs, requestClip };

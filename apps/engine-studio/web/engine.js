"use strict";

// /engine -- the full engine dashboard. Everything GET /api/engine/overview reports, refreshed every 1.5 s while
// the tab is visible. Read-only, except Reset: the same POST /api/stats/reset the chat UI's All sessions panel uses
// (a baseline kept by the Studio; the engine's own counters, cache and process are never touched).

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 1500;
const NA = "n/a";
const ui = { paused: false, includeOld: false, openPaths: new Set(), data: null, timer: null, inflight: false };

// ------------------------------------------------------------------------------------------ formatting

function esc(t) {
  return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const isNum = (v) => typeof v === "number" && isFinite(v);
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

function fmtNum(n) {
  if (!isNum(n)) return NA;
  const a = Math.abs(n);
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e8 ? 0 : 1) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}
function fmtBytes(b) {
  if (!isNum(b)) return NA;
  const a = Math.abs(b);
  if (a >= 1e9) return (b / 1e9).toFixed(1) + " GB";
  if (a >= 1e6) return (b / 1e6).toFixed(a >= 1e7 ? 0 : 1) + " MB";
  if (a >= 1e3) return (b / 1e3).toFixed(0) + " kB";
  return b + " B";
}
function fmtPct(x) { return isNum(x) ? (Math.abs(x) < 10 ? x.toFixed(1) : x.toFixed(0)) + "%" : NA; }
function fmtDur(s) {
  if (!isNum(s)) return NA;
  if (s < 90) return s.toFixed(s < 10 ? 2 : 1) + "s";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return d ? `${d}d ${h}h ${m}m` : h ? `${h}h ${m}m` : `${m}m ${sec}s`;
}
function fmtClock(t) {
  return isNum(t) ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : NA;
}
function fmtDateTime(t) {
  return isNum(t) ? new Date(t * 1000).toLocaleString([], { year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit" }) : NA;
}
function fmtRate(v, digits = 1) { return isNum(v) ? v.toFixed(digits) : NA; }
function shortSha(h) { return h ? String(h).slice(0, 12) : NA; }
function numCell(n) { return isNum(n) ? `<span title="${n.toLocaleString()}">${fmtNum(n)}</span>` : NA; }

// A generic scalar, formatted from its key's name: *bytes* -> GB/MB, *_seconds / *_s -> s, *_ms -> ms, *_at -> clock.
function fmtKeyed(key, v) {
  if (v === null || v === undefined) return NA;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (isNum(v)) {
    const k = String(key);
    if (/bytes/i.test(k)) return fmtBytes(v);
    if (/(_seconds|_s)$/.test(k)) return fmtDur(v);
    if (/_ms$/.test(k)) return v.toFixed(1) + " ms";
    if (/^(at|now|since)$|_at$/.test(k) && v > 1e9) return fmtClock(v);
    return Number.isInteger(v) ? v.toLocaleString() : String(+v.toFixed(3));
  }
  if (Array.isArray(v)) return v.every((x) => x === null || typeof x !== "object") ? v.join(", ") || "[]" : JSON.stringify(v);
  if (isObj(v)) return JSON.stringify(v);
  return String(v);
}

// ------------------------------------------------------------------------------------------ building blocks

function tile(v, l, title) {
  return `<div class="es-tile"${title ? ` title="${esc(title)}"` : ""}><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`;
}
// A bar: `used` of `total`, with an optional marker (e.g. peak) at `mark`.
function meter(label, used, total, text, mark) {
  const pct = isNum(used) && isNum(total) && total > 0 ? Math.min(100, (100 * used) / total) : 0;
  const mk = isNum(mark) && isNum(total) && total > 0
    ? `<div class="meter-mark" style="left:${Math.min(100, (100 * mark) / total)}%"></div>` : "";
  return `<div class="meter"><div class="meter-top"><span>${esc(label)}</span><span>${text}</span></div>` +
         `<div class="meter-track"><div class="meter-fill" style="width:${pct}%"></div>${mk}</div></div>`;
}
function kv(obj, opts = {}) {
  if (!isObj(obj) || !Object.keys(obj).length) return `<div class="es-empty">${esc(opts.empty || "none")}</div>`;
  const rows = Object.entries(obj).map(([k, v]) => {
    const f = opts.fmt && opts.fmt[k] ? opts.fmt[k](v) : esc(fmtKeyed(k, v));
    const exact = isNum(v) ? ` title="${esc(v)}"` : "";
    return `<dt>${esc(k)}</dt><dd${exact}${opts.mono ? ' class="mono"' : ""}>${f}</dd>`;
  });
  return `<dl class="kv">${rows.join("")}</dl>`;
}
function table(cols, rows, empty) {
  if (!rows.length) return `<div class="es-empty">${esc(empty || "none")}</div>`;
  const head = cols.map((c) => `<th${c.num ? ' class="num"' : ""}>${esc(c.h)}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map((c) => `<td${c.num ? ' class="num"' : ""}>${c.f(r)}</td>`).join("")}</tr>`).join("");
  return `<table class="dt"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
// A histogram of {label: count}: numeric labels in order, others by count (largest first).
function bars(obj, opts = {}) {
  if (!isObj(obj)) return `<div class="es-empty">${esc(opts.empty || "none")}</div>`;
  let entries = Object.entries(obj).filter(([, v]) => isNum(v));
  if (!entries.length) return `<div class="es-empty">${esc(opts.empty || "none")}</div>`;
  const numeric = entries.every(([k]) => /^-?\d+(\.\d+)?$/.test(k));
  entries.sort(numeric ? (a, b) => +a[0] - +b[0] : (a, b) => b[1] - a[1]);
  const hidden = opts.top && entries.length > opts.top ? entries.length - opts.top : 0;
  if (hidden) entries = entries.slice(0, opts.top);
  const total = Object.values(obj).filter(isNum).reduce((s, v) => s + v, 0);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  const rows = entries.map(([k, v]) =>
    `<div class="bar-row"><span class="bar-label" title="${esc(k)}">${esc((opts.prefix || "") + k)}</span>` +
    `<span class="bar-track"><span class="bar-fill" style="width:${(100 * v) / max}%"></span></span>` +
    `<span class="bar-val" title="${v.toLocaleString()}">${fmtNum(v)} <span class="muted">${fmtPct(total ? (100 * v) / total : 0)}</span></span></div>`);
  if (hidden) rows.push(`<div class="es-empty">+ ${hidden} more (see Everything else)</div>`);
  return `<div class="bars">${rows.join("")}</div>`;
}
function badge(text) {
  const t = text == null ? NA : String(text);
  const cls = t === "prefill" ? "prefill" : t === "decode" ? "decode" : /abort|error|cancel/.test(t) ? "err" : "";
  return `<span class="badge ${cls}">${esc(t)}</span>`;
}
function block(title, html) { return `<div class="sub"><h3>${esc(title)}</h3>${html}</div>`; }

// ------------------------------------------------------------------------------------------ sections

function renderHeader(d) {
  const s = d.sessions || {}, p = d.process || {}, m = d.model || {};
  $("d-dot").className = "dot " + (d.available ? "dot-on" : p.running ? "dot-loading" : "dot-off");
  $("d-model").textContent = s.model || m.id || "no model";
  const pill = (label, value, title, cls) =>
    `<span class="pill${cls ? " " + cls : ""}"${title ? ` title="${esc(title)}"` : ""}>${esc(label)} <b>${esc(value)}</b></span>`;
  const pills = [];
  pills.push(pill("up", fmtDur(s.uptime_s ?? p.uptime_s), "engine uptime (sessions.uptime_s); process uptime " + fmtDur(p.uptime_s)));
  pills.push(pill("pid", p.pid ?? NA, p.others && p.others.length ? `other engine processes: ${p.others.map((o) => o.pid).join(", ")}` : "the engine process"));
  pills.push(pill("port", d.engine_port ?? NA, d.attached ? "attached: an engine the Studio did not start" : ""));
  const bin = p.binary || {}, lib = p.metallib || {};
  pills.push(pill("bin", shortSha(bin.sha256), `${bin.path || ""}\nsha256 ${bin.sha256 || NA}`));
  if (bin.newer_than_process) pills.push(pill("!", "binary rebuilt since start", "the file on disk is newer than the running process -- its sha is NOT what runs", "warn"));
  pills.push(pill("metallib", shortSha(lib.sha256), `${lib.path || ""}\nsha256 ${lib.sha256 || NA}`));
  if (lib.newer_than_process) pills.push(pill("!", "metallib rebuilt since start", "", "warn"));
  pills.push(s.since
    ? pill("since reset", `${fmtClock(s.since)} (${fmtDur(d.at - s.since)} ago)`, "counters and recent requests count from this Reset")
    : pill("since reset", "never", "no Reset since the engine started: counters are the engine's own"));
  $("d-pills").innerHTML = pills.join("");
}

function renderLive(d) {
  const s = d.sessions || {}, a = s.aggregate || {}, sys = d.system || {};
  const raw = d.raw || {}, aw = raw.admission_wait || s.admission_wait || {}, rbq = raw.ram_budget || s.ram_budget || {};
  $("d-now").innerHTML =
    tile(`${a.active ?? NA}/${s.max_concurrent ?? NA}`, "active / max") +
    tile(a.prefilling ?? NA, "prefilling") +
    tile(a.decoding ?? NA, "decoding") +
    tile(isNum(a.decode_tokens_per_second) ? a.decode_tokens_per_second.toFixed(0) : NA, "decode tok/s now") +
    tile(numCell(a.prompt_tokens_in_flight), "prompt tokens in flight") +
    tile(s.reserved_sequences ?? NA, "reserved sequences") +
    tile(isNum(s.batch_steps) ? fmtNum(s.batch_steps) : NA, "batch steps") +
    tile(s.mtp_depth ?? NA, "MTP depth") +
    // T-0092: requests waiting for a memory reservation are in no table above; show them and what the next one costs
    tile(aw.waiting ?? NA, "waiting for admission",
         "requests the admission budget has not taken yet (they are in neither table)") +
    tile(isNum(aw.refused_after_wait) ? aw.refused_after_wait : NA, "refused after wait",
         "requests that waited the whole queue timeout and got an error") +
    tile(isNum(aw.max_wait_s) ? aw.max_wait_s.toFixed(0) + " s" : NA, "longest admission wait") +
    tile(fmtBytes(rbq.prospective_reservation_growth_bytes), "next request's price",
         "what admitting one more request costs in the admission budget (ram_budget.prospective_reservation_growth_bytes)");

  const mem = s.memory || {};
  $("d-mem").innerHTML =
    `<div class="tiles">${tile(fmtBytes(mem.active_bytes), "active")}${tile(fmtBytes(mem.peak_bytes), "peak")}` +
    `${tile(fmtBytes(mem.cache_bytes), "MLX cache")}${tile(fmtBytes(sys.ram_bytes), "RAM")}</div>` +
    meter("active of RAM (| = peak)", mem.active_bytes, sys.ram_bytes,
      `${fmtBytes(mem.active_bytes)} / ${fmtBytes(sys.ram_bytes)} · ${fmtPct(sys.ram_bytes ? (100 * mem.active_bytes) / sys.ram_bytes : null)}`,
      mem.peak_bytes) +
    (mem.at ? `<div class="hint">sampled ${fmtClock(mem.at)}</div>` : "");

  const hot = s.hot_cache || {};
  $("d-hot").innerHTML = Object.keys(hot).length
    ? meter("charged of ceiling", hot.charged_bytes, hot.ceiling_bytes,
        `${fmtBytes(hot.charged_bytes)} / ${fmtBytes(hot.ceiling_bytes)} · ${fmtPct(hot.ceiling_bytes ? (100 * hot.charged_bytes) / hot.ceiling_bytes : null)}`) +
      `<div class="tiles">${tile(hot.entries ?? NA, "entries")}${tile(hot.rungs ?? NA, "rungs")}` +
      `${tile(hot.live_rungs ?? NA, "live rungs")}${tile(hot.inflight_entries ?? NA, "in-flight entries")}` +
      `${tile(fmtBytes(hot.logical_bytes), "logical")}${tile(fmtBytes(hot.target_bytes), "target")}` +
      `${tile(hot.rejected_stores ?? NA, "rejected stores")}${tile(hot.pre_copy_evictions ?? NA, "pre-copy evictions")}</div>`
    : `<div class="es-empty">this engine build reports no hot_cache</div>`;

  const rb = s.ram_budget || {};
  $("d-ram").innerHTML = Object.keys(rb).length
    ? meter("active reservations of capacity", rb.active_reservation_bytes, rb.capacity_bytes,
        `${fmtBytes(rb.active_reservation_bytes)} / ${fmtBytes(rb.capacity_bytes)}`) +
      `<div class="tiles">${tile(fmtBytes(rb.capacity_bytes), "capacity")}${tile(rb.active_reservations ?? NA, "reservations")}` +
      `${tile(fmtBytes(rb.hot_target_bytes), "hot target")}${tile(fmtBytes(rb.hot_ceiling_bytes), "hot ceiling")}` +
      `${tile(fmtBytes(rb.live_limit_bytes), "live limit")}${tile(fmtBytes(rbq.prospective_reservation_growth_bytes), "prospective growth")}</div>`
    : `<div class="es-empty">this engine build reports no ram_budget</div>`;
}

function renderTotals(d) {
  const s = d.sessions || {}, t = s.totals || {}, raw = s.raw_totals || t;
  const keys = ["requests", "prompt_tokens", "cached_tokens", "generated_tokens", "aborted"];
  for (const k of Object.keys(raw)) if (!keys.includes(k)) keys.push(k);
  const label = { requests: "requests", prompt_tokens: "tokens in", cached_tokens: "tokens cached",
                  generated_tokens: "tokens out", aborted: "aborted" };
  const hit = (x) => (isNum(x.prompt_tokens) && x.prompt_tokens > 0 ? fmtPct(Math.min(100, (100 * (x.cached_tokens || 0)) / x.prompt_tokens)) : NA);
  const rows = keys.map((k) => ({ l: label[k] || k, a: numCell(t[k]), b: numCell(raw[k]) }));
  rows.splice(4, 0, { l: "cache hit", a: hit(t), b: hit(raw) });
  const sinceLabel = s.since ? `since reset (${fmtClock(s.since)})` : "since reset (none yet)";
  $("d-totals").innerHTML = table([
    { h: "", f: (r) => esc(r.l) },
    { h: sinceLabel, num: true, f: (r) => r.a },
    { h: `since engine start (${fmtDur(s.uptime_s)} ago)`, num: true, f: (r) => r.b },
  ], rows, "engine unavailable");
}

function renderLimits(d) {
  const rows = d.limits || [];
  $("d-limits").innerHTML = rows.length
    ? `<table class="dt limits"><thead><tr><th>limit</th><th>value</th><th>source</th><th>note</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td class="lim-name">${esc(r.name)}</td><td class="lim-val">${esc(r.value ?? NA)}</td>` +
        `<td class="lim-src mono">${esc(r.source ?? NA)}</td><td class="lim-note">${esc(r.note || "")}</td></tr>`).join("") +
      `</tbody></table>`
    : `<div class="es-empty">no limits derived</div>`;
}

const REQ_KNOWN = new Set(["id", "client", "phase", "finish_reason", "prompt_tokens", "cached_tokens", "cache_source",
  "prefilled_to", "generated", "max_tokens", "decode_tokens_per_second", "prefill_tokens_per_second", "ttft_ms", "mtp",
  "mtp_drafted", "mtp_accepted", "mtp_acceptance", "reasoning_effort", "stream", "elapsed_s", "started_at"]);

function reqCols(finished, rows) {
  const cols = [
    { h: "#", num: true, f: (r) => esc(r.id ?? NA) },
    { h: "client", f: (r) => `<span class="clip" title="${esc(r.client ?? "")}">${esc(r.client ?? NA)}</span>` },
    { h: finished ? "finish" : "phase", f: (r) => badge(finished ? (r.finish_reason ?? r.phase) : r.phase) },
    { h: "prompt", num: true, f: (r) => numCell(r.prompt_tokens) },
    { h: "cached", num: true, f: (r) => numCell(r.cached_tokens) +
        (r.cache_source && r.cache_source !== "none" ? ` <span class="muted">${esc(r.cache_source)}</span>` : "") },
    { h: "prefilled", num: true, f: (r) => {
        let bar = "";
        if (!finished && r.phase === "prefill" && r.prompt_tokens > 0) {
          const c = (100 * (r.cached_tokens || 0)) / r.prompt_tokens;
          const p = Math.max(0, (100 * ((r.prefilled_to || 0) - (r.cached_tokens || 0))) / r.prompt_tokens);
          bar = `<div class="es-bar"><div class="cached" style="width:${c}%"></div><div class="done" style="width:${p}%"></div></div>`;
        }
        return numCell(r.prefilled_to) + bar;
      } },
    { h: "out / max", num: true, f: (r) => `${numCell(r.generated)} / ${numCell(r.max_tokens)}` },
    { h: "decode tok/s", num: true, f: (r) => fmtRate(r.decode_tokens_per_second) },
    { h: "prefill tok/s", num: true, f: (r) => fmtRate(r.prefill_tokens_per_second, 0) },
    { h: "TTFT", num: true, f: (r) => (isNum(r.ttft_ms) ? (r.ttft_ms / 1000).toFixed(2) + "s" : NA) },
    { h: "MTP acc/drafted", num: true, f: (r) => {
        if (r.mtp === false) return "off";
        if (isNum(r.mtp_drafted)) {
          const acc = isNum(r.mtp_acceptance) ? r.mtp_acceptance : r.mtp_drafted ? (r.mtp_accepted || 0) / r.mtp_drafted : null;
          return `${r.mtp_accepted ?? 0}/${r.mtp_drafted} <span class="muted">${fmtPct(isNum(acc) ? 100 * acc : null)}</span>`;
        }
        return r.mtp ? "on" : NA;
      } },
    // the engine records "off" for a thinking-off request (no effort applies then); builds before it showed the
    // resolved effort there too ("low" for reasoning_effort none)
    { h: "thinking", f: (r) => r.reasoning_effort == null ? NA : r.reasoning_effort === "off" ? "off" : esc(`on · ${r.reasoning_effort}`) },
    { h: "stream", f: (r) => (r.stream == null ? NA : r.stream ? "yes" : "no") },
    { h: "elapsed", num: true, f: (r) => fmtDur(r.elapsed_s) },
    { h: "started", num: true, f: (r) => fmtClock(r.started_at) },
  ];
  // every field the engine gives: a key this page does not know yet still shows, in "other"
  if (rows.some((r) => Object.keys(r).some((k) => !REQ_KNOWN.has(k)))) {
    cols.push({ h: "other", f: (r) => Object.entries(r).filter(([k]) => !REQ_KNOWN.has(k))
      .map(([k, v]) => `<span class="muted">${esc(k)}</span>=${esc(fmtKeyed(k, v))}`).join(" ") });
  }
  return cols;
}

function renderRequests(d) {
  const s = d.sessions || {};
  const active = (s.active || []).slice().sort((x, y) => (x.id || 0) - (y.id || 0));
  const waiting = ((d.raw || {}).admission_wait || s.admission_wait || {}).waiting;
  $("d-active-n").textContent = `(${active.length}${isNum(s.max_concurrent) ? " of " + s.max_concurrent : ""}` +
    `${isNum(waiting) && waiting > 0 ? ", " + waiting + " waiting for admission" : ""})`;
  $("d-active").innerHTML = table(reqCols(false, active), active, d.available ? "no request in flight" : (d.reason || "engine unavailable"));
  const recent = (ui.includeOld ? (d.raw || {}).recent : s.recent) || [];
  $("d-recent-n").textContent = `(${recent.length}${!ui.includeOld && s.since ? " since reset" : ""})`;
  $("d-recent").innerHTML = table(reqCols(true, recent), recent, "none yet");
}

function renderScheduler(d) {
  const s = d.sessions || {};
  const out = [];
  const phases = s.scheduler_phases;
  if (isObj(phases) && Object.keys(phases).length) {
    const keys = [];
    for (const p of Object.values(phases)) if (isObj(p)) for (const k of Object.keys(p)) if (!keys.includes(k)) keys.push(k);
    const cols = [{ h: "phase", f: (r) => esc(r[0]) }].concat(keys.map((k) => ({
      h: k.replace(/_/g, " "), num: true, f: (r) => esc(fmtKeyed(k, r[1][k])) })));
    // derived: mean work and mean queue wait per job
    cols.push({ h: "work / job", num: true, f: (r) => (r[1].jobs ? ((1000 * (r[1].work_seconds || 0)) / r[1].jobs).toFixed(1) + " ms" : NA) });
    cols.push({ h: "wait / job", num: true, f: (r) => (r[1].jobs ? ((1000 * (r[1].queue_wait_seconds || 0)) / r[1].jobs).toFixed(1) + " ms" : NA) });
    out.push(block("scheduler_phases", `<div class="tbl-wrap">${table(cols, Object.entries(phases).filter(([, v]) => isObj(v)))}</div>`));
  } else {
    out.push(block("scheduler_phases", `<div class="es-empty">${phases ? "empty" : "not reported by this build"}</div>`));
  }
  const grid = [];
  grid.push(block(`batch_sizes (decode steps by rows)`, bars(s.batch_sizes, { prefix: "B=", empty: "no batched steps yet" })));
  grid.push(block("prefill_scheduling", kv(s.prefill_scheduling, { empty: "not reported" })));
  const rs = isObj(s.restack) ? s.restack : {};
  const trans = {}, rsOther = {};
  for (const [k, v] of Object.entries(rs)) (k.includes("->") ? trans : rsOther)[k] = v;
  grid.push(block("restack transitions", bars(trans, { top: 16, empty: "no membership changes" })));
  grid.push(block("restack", kv(rsOther, { empty: "not reported" })));
  grid.push(block("native_prefill_sizes", bars(s.native_prefill_sizes, { empty: "none" })));
  // P114 witnesses (a newer engine build) -- and any dict that ends in _decisions, rendered generically
  for (const [k, v] of Object.entries(s)) {
    if (/_decisions$/.test(k) && isObj(v)) grid.push(block(k, bars(v, { empty: "no decisions yet" })));
  }
  if (!("max_tokens_decisions" in s)) grid.push(block("max_tokens_decisions", `<div class="es-empty">not reported by this build (P114)</div>`));
  if (!("reasoning_effort_decisions" in s)) grid.push(block("reasoning_effort_decisions", `<div class="es-empty">not reported by this build (P114)</div>`));
  grid.push(block("pooled_private_history", kv(s.pooled_private_history, { empty: "not reported" })));
  grid.push(block("indexer_graph_construction", kv(s.indexer_graph_construction, { empty: "empty" })));
  $("d-sched").innerHTML = out.join("") + `<div class="subgrid">${grid.join("")}</div>`;
}

function renderConfig(d) {
  const p = d.process || {}, env = d.env || {}, m = d.model || {}, sys = d.system || {};
  const fileKv = (f) => (f && f.exists
    ? `<span class="mono">${esc(f.sha256 || NA)}</span><br><span class="muted">${esc(f.path)} · ${fmtBytes(f.bytes)} · ` +
      `modified ${fmtDateTime(f.mtime)}${f.newer_than_process ? " · NEWER than the running process" : ""}</span>`
    : `<span class="muted">${esc(f && f.path ? f.path + " (missing)" : NA)}</span>`);
  let proc;
  if (p.running) {
    const view = {
      pid: p.pid, "all engine pids": (p.pids || []).join(", ") || NA, subcommand: p.subcommand,
      started: fmtDateTime(p.started_at), "process uptime": fmtDur(p.uptime_s), executable: p.exe,
      "found by": p.exe_source, cwd: p.cwd, "engine port": d.engine_port, attached: d.attached ? "yes (not started by the Studio)" : "no",
    };
    const fmt = { binary: () => fileKv(p.binary), metallib: () => fileKv(p.metallib) };
    view.binary = ""; view.metallib = "";
    if (p.metallib_env) { view["metallib (env)"] = ""; fmt["metallib (env)"] = () => fileKv(p.metallib_env); }
    for (const k of Object.keys(view)) if (!fmt[k]) fmt[k] = (v) => esc(v ?? NA);
    proc = kv(view, { fmt }) + `<h3>argv</h3><pre class="argv mono">${esc(p.command || NA)}</pre>` +
      ((p.others || []).length ? `<h3>other engine processes</h3>` + p.others.map((o) =>
        `<pre class="argv mono">${esc(o.pid)}: ${esc(o.command)}</pre>`).join("") : "");
  } else {
    proc = `<div class="es-empty">${esc(p.note || "no engine process")}</div>`;
  }
  const flags = p.flags || {};
  const flagFmt = {};
  for (const k of Object.keys(flags)) flagFmt[k] = (v) => esc(v === true ? "(switch)" : v);
  const envHtml = env.available ? kv(env.vars, { mono: true, fmt: Object.fromEntries(Object.keys(env.vars).map((k) => [k, (v) => esc(v)])) }) +
      `<div class="hint">${esc(env.filter || "")}</div>`
    : `<div class="es-empty">${esc(env.note || "not available")}</div>`;
  const q = m.quantization || {};
  const modelView = Object.assign({}, m);
  const mfmt = {
    layer_types: (v) => esc(isObj(v) ? Object.entries(v).map(([k, n]) => `${n} ${k}`).join(", ") : NA),
    quantization: () => esc(`${q.bits ?? NA}-bit g${q.group_size ?? NA} ${q.mode || ""}`) +
      (isObj(q.per_module_overrides) && Object.keys(q.per_module_overrides).length
        ? `<br><span class="muted">per-module: ${esc(Object.entries(q.per_module_overrides).map(([k, n]) => `${n}× ${k}`).join(", "))}</span>` : ""),
    generation: (v) => esc(isObj(v) ? Object.entries(v).map(([k, x]) => `${k} ${x}`).join(", ") : NA),
    reasoning_levels: (v) => esc(Array.isArray(v) ? v.join(", ") : NA),
    weights_bytes: (v) => esc(`${fmtBytes(v)} in ${m.weight_files ?? NA} .safetensors files`),
    dir_bytes: (v) => esc(fmtBytes(v)),
    max_position_embeddings: (v) => esc(isNum(v) ? v.toLocaleString() + " tokens" : NA),
  };
  delete modelView.weight_files;
  const sysView = Object.assign({}, sys);
  const sfmt = {
    ram_bytes: (v) => esc(isNum(v) ? `${fmtBytes(v)} (${(v / 2 ** 30).toFixed(0)} GiB)` : NA),
    boot_time: (v) => esc(fmtDateTime(v)),
    load_avg: (v) => esc(Array.isArray(v) ? v.join(" / ") : NA),
    iogpu_wired_limit_mb: (v) => esc(v == null ? "n/a (sysctl absent)" : `${v}${v === 0 ? " (0 = macOS default)" : " MB"}`),
    iogpu_disable_wired_collector: (v) => esc(v == null ? "n/a (sysctl absent)" : String(v)),
  };
  $("d-config").innerHTML =
    block("Process", proc) +
    block(`Flags (${Object.keys(flags).length})`, kv(flags, { mono: true, fmt: flagFmt, empty: "no flags (engine process not found)" })) +
    block(`Environment (ENGINE_* / MLX*)`, envHtml) +
    block(`Model${m.id ? " · " + m.id : ""}`, m.error && !m.model_type ? `<div class="es-empty">${esc(m.error)}</div>` : kv(modelView, { fmt: mfmt })) +
    block("System", kv(sysView, { fmt: sfmt }));
}

// "Everything else": the full raw sessions payload as a collapsible tree. Which nodes are open is kept across
// refreshes (by path), and the tree is only built while its section is open.
function treeNode(key, v, path) {
  if (v !== null && typeof v === "object") {
    const arr = Array.isArray(v);
    const entries = arr ? v.map((x, i) => [i, x]) : Object.entries(v);
    const open = ui.openPaths.has(path) ? " open" : "";
    const kids = entries.map(([k, x]) => treeNode(k, x, path + "/" + k)).join("");
    return `<details class="tn" data-path="${esc(path)}"${open}><summary><span class="tk">${esc(key)}</span> ` +
           `<span class="muted">${arr ? `[${entries.length}]` : `{${entries.length}}`}</span></summary>` +
           `<div class="tc">${kids || '<div class="muted">empty</div>'}</div></details>`;
  }
  const raw = v === null ? "null" : typeof v === "string" ? JSON.stringify(v) : String(v);
  const nice = fmtKeyed(key, v);
  const plain = isNum(v) && [v.toLocaleString(), String(v), String(+v.toFixed(3))].includes(nice);
  const hint = isNum(v) && !plain ? ` <span class="muted">(${esc(nice)})</span>` : "";
  return `<div class="tl"><span class="tk">${esc(key)}</span>: <span class="tv t-${typeof v}">${esc(raw)}</span>${hint}</div>`;
}
function renderRaw(d) {
  if (!$("d-raw-box").open) return;
  const raw = d.raw || {};
  $("d-raw").innerHTML = Object.entries(raw).map(([k, v]) => treeNode(k, v, k)).join("") ||
    `<div class="es-empty">${esc(d.reason || "no payload")}</div>`;
}

// ------------------------------------------------------------------------------------------ over time (charts)
//
// The Studio server samples the engine every 1.5 s and keeps 3 h (GET /api/engine/history, chartHistory below); a Studio
// without that endpoint falls back to one sample per page refresh kept in localStorage. A gap longer than GAP_S breaks
// the line.
//   decode tok/s   the engine's aggregate.decode_tokens_per_second
//   prefill tok/s  prompt tokens actually computed per second, from the progress of each request's prefilled_to between
//                  two samples (a new request counts from its cached tokens; a request that finished in between counts
//                  up to its prompt length), so cache hits are excluded
//   concurrency    aggregate.active / decoding / prefilling
//   context        sum over active requests of prompt + generated tokens, and the largest single request

const HIST_KEY = "engine-dash-history-v1", HIST_MAX_S = 3 * 3600, GAP_S = 8;
const WINDOWS = [["5m", 300], ["15m", 900], ["1h", 3600], ["3h", 10800]];
// hover / pin: a moment in time (t), not a pixel of one canvas, so every chart marks the same moment
const charts = { samples: [], win: 900, pf: null, hover: null, pin: null, lastSave: 0, server: null };
try {
  const h = JSON.parse(localStorage.getItem(HIST_KEY) || "null");
  if (h && Array.isArray(h.samples)) charts.samples = h.samples;
  if (h && isNum(h.win)) charts.win = h.win;
} catch { /* no storage: history lives in this page only */ }

function chartSample(d) {
  const t = d.at;
  if (!isNum(t)) return;
  if (!d.available) { charts.samples.push({ t, gap: true }); charts.pf = null; return; }
  const s = d.sessions || {}, a = s.aggregate || {};
  const active = s.active || [], recent = s.recent || [];
  let pre = null;
  const pf = charts.pf;
  if (pf && t > pf.t && t - pf.t <= GAP_S) {
    let work = 0;
    for (const r of active) {
      const base = pf.prev.has(r.id) ? pf.prev.get(r.id) : (r.cached_tokens || 0);
      work += Math.max(0, (r.prefilled_to || 0) - base);
    }
    for (const r of recent) {
      if (pf.seen.has(r.id)) continue;
      const base = pf.prev.has(r.id) ? pf.prev.get(r.id) : (r.cached_tokens || 0);
      work += Math.max(0, (r.prompt_tokens || 0) - base);
    }
    pre = work / (t - pf.t);
  }
  const seen = pf ? pf.seen : new Set();
  for (const r of recent) seen.add(r.id);
  charts.pf = { t, prev: new Map(active.map((r) => [r.id, r.prefilled_to || r.cached_tokens || 0])), seen };
  let ctx = 0, ctxMax = 0;
  for (const r of active) { const c = (r.prompt_tokens || 0) + (r.generated || 0); ctx += c; ctxMax = Math.max(ctxMax, c); }
  charts.samples.push({ t, dec: isNum(a.decode_tokens_per_second) ? a.decode_tokens_per_second : null, pre,
                        act: a.active ?? null, decn: a.decoding ?? null, pren: a.prefilling ?? null, ctx, ctxMax });
  while (charts.samples.length && charts.samples[0].t < t - HIST_MAX_S) charts.samples.shift();
  if (t - charts.lastSave > 10) {
    charts.lastSave = t;
    try { localStorage.setItem(HIST_KEY, JSON.stringify({ win: charts.win, samples: charts.samples })); } catch { /* full */ }
  }
}

// The Studio server samples the engine continuously (GET /api/engine/history, same fields as chartSample); this page
// only fetches what is new since its last sample. A Studio without that endpoint falls back to sampling in the page.
async function chartHistory(d) {
  if (charts.server !== false) {
    const last = charts.server ? (charts.samples.length ? charts.samples[charts.samples.length - 1].t : 0) : 0;
    try {
      const r = await fetch(`/api/engine/history?since=${last}`, { cache: "no-store" });
      if (r.status === 404) throw new Error("no history endpoint");
      if (!r.ok) return;
      const h = await r.json();
      if (!charts.server) { charts.server = true; charts.samples = []; charts.pf = null; }   // the server's history replaces the page's
      for (const x of h.samples || []) charts.samples.push(x);
      if (h.records) { charts.records = h.records; renderRecords(); }
      const cut = (charts.samples.length ? charts.samples[charts.samples.length - 1].t : 0) - HIST_MAX_S;
      while (charts.samples.length && charts.samples[0].t < cut) charts.samples.shift();
      try { localStorage.setItem(HIST_KEY, JSON.stringify({ win: charts.win, samples: [] })); } catch { /* */ }
      return;
    } catch (e) {
      if (String(e.message).includes("no history endpoint")) charts.server = false; else return;
    }
  }
  chartSample(d);
}

// the Studio server's best value of each charted quantity since its records started (or were reset), with when, and
// next to it the median over the busy samples (value > 0) since median_since
function renderRecords() {
  const r = charts.records, box = $("d-records");
  if (!box) return;
  if (!r) { box.innerHTML = `<div class="es-empty">records need the Studio server's history</div>`; return; }
  const b = r.best || {}, med = r.median || {};
  const medSince = isNum(r.median_since) ? fmtDateTime(r.median_since) : NA;
  const cell = (k, label, fmt) => {
    const m = med[k] && isNum(med[k].v) ? med[k] : null;
    const title = (b[k] ? `best reached ${fmtDateTime(b[k].t)}` : "not reached yet") +
      (m ? `\nmedian of ${m.n.toLocaleString()} busy samples (value > 0) since ${medSince}, within 0.5%` : "\nno busy sample yet");
    return `<div class="es-tile" title="${esc(title)}"><div class="v">${b[k] ? fmt(b[k].v) : NA}` +
      `<span class="rec-med"> · med ${m ? fmt(m.v) : NA}</span></div><div class="l">${esc(label)}</div></div>`;
  };
  const when = (k) => (b[k] ? `<div class="rec-when">${fmtDateTime(b[k].t)}</div>` : "");
  const items = [["dec", "decode tok/s", (v) => v.toFixed(1)], ["pre", "prefill tok/s", (v) => v.toFixed(0)],
                 ["act", "concurrent requests", (v) => String(Math.round(v))], ["ctx", "context tokens (all active)", fmtNum],
                 ["ctxMax", "context tokens (one request)", fmtNum]];
  box.innerHTML = items.map(([k, l, f]) => `<div class="rec">${cell(k, l, f)}${when(k)}</div>`).join("");
  $("d-rec-since").textContent = (isNum(r.since) ? `best since ${fmtDateTime(r.since)}` : "") +
    (isNum(r.median_since) ? ` · median (busy samples) since ${medSince}` : "");
}

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

// series: [{key, color, width}]; the first series is the headline value
function drawChart(canvasId, nowId, series, fmt, integer) {
  const cv = $(canvasId);
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const dim = css.getPropertyValue("--text-dim").trim() || "#9a9ca3", border = css.getPropertyValue("--border").trim() || "#34363c";
  const last = charts.samples.length ? charts.samples[charts.samples.length - 1].t : Date.now() / 1000;
  const t1 = last, t0 = t1 - charts.win;
  const pts = charts.samples.filter((p) => p.t >= t0 - GAP_S);
  let vmax = 0;
  for (const p of pts) for (const s of series) if (isNum(p[s.key])) vmax = Math.max(vmax, p[s.key]);
  vmax = niceMax(integer ? Math.max(vmax, 4) : vmax * 1.05);
  if (integer) vmax = Math.ceil(vmax / 4) * 4;     // whole-number gridlines (e.g. 0 / 10 / 20 / 30 / 40, not 12.5 steps)
  const L = CH_L, R = CH_R, T = 8, B = 20, pw = W - L - R, ph = H - T - B;
  const X = (t) => L + ((t - t0) / (t1 - t0)) * pw, Y = (v) => T + ph - (v / vmax) * ph;
  g.font = "10px -apple-system, sans-serif"; g.fillStyle = dim; g.strokeStyle = border; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (vmax * i) / 4, y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(fmtNum(integer ? Math.round(v) : v), L - 5, y);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = t0 + (charts.win * i) / 4, ago = t1 - t;
    g.fillText(ago < 1 ? "now" : "-" + (ago >= 3600 ? (ago / 3600).toFixed(ago % 3600 ? 1 : 0) + "h" : Math.round(ago / 60) + "m"), X(t), H - B + 5);
  }
  g.save(); g.beginPath(); g.rect(L, T, pw, ph); g.clip();
  for (const s of series.slice().reverse()) {
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.5; g.beginPath();
    let pen = false, prevT = null;
    for (const p of pts) {
      const v = p[s.key];
      if (p.gap || !isNum(v) || (prevT !== null && p.t - prevT > GAP_S)) { pen = false; prevT = p.gap ? null : p.t; if (!isNum(v)) continue; }
      const x = X(p.t), y = Y(v);
      if (pen) g.lineTo(x, y); else g.moveTo(x, y);
      pen = true; prevT = p.t;
    }
    g.stroke();
  }
  g.restore();
  // the marked moment (hover, else a pinned click), the same t on every chart: the sample nearest it
  let shown = pts.length ? pts[pts.length - 1] : null, hoverX = null;
  const mark = charts.hover ?? charts.pin;
  if (isNum(mark) && mark >= t0 && mark <= t1 && pts.length) {
    shown = pts.reduce((b, p) => (Math.abs(p.t - mark) < Math.abs(b.t - mark) ? p : b), pts[0]);
    hoverX = X(shown.t);
    g.strokeStyle = charts.hover == null ? warmColor : dim;
    g.setLineDash(charts.hover == null ? [] : [3, 3]);
    g.beginPath(); g.moveTo(hoverX + 0.5, T); g.lineTo(hoverX + 0.5, T + ph); g.stroke(); g.setLineDash([]);
    g.fillStyle = dim;
    for (const s of series) {      // a dot where the line passes the marked moment
      if (!isNum(shown[s.key])) continue;
      g.fillStyle = s.color; g.beginPath(); g.arc(hoverX, Y(shown[s.key]), 3, 0, 2 * Math.PI); g.fill();
    }
  }
  // median of the visible window over its busy samples (value > 0), like the records' median
  const busy = series[0] ? pts.filter((p) => p.t >= t0 && !p.gap && isNum(p[series[0].key]) && p[series[0].key] > 0)
    .map((p) => p[series[0].key]).sort((a, b) => a - b) : [];
  const med = busy.length ? busy[(busy.length - 1) >> 1] : null;
  const vals = series.map((s) => `<span style="color:${s.color}">●</span> <b>${shown && isNum(shown[s.key]) ? fmt(shown[s.key]) : NA}</b>`).join(" · ");
  $(nowId).innerHTML = (hoverX !== null && shown ? `${fmtClock(shown.t)}${charts.hover == null ? " (pinned)" : ""} · ` : "") + vals +
    ` <span class="muted" title="median of the ${busy.length} busy samples (value > 0) in this window">· med ${isNum(med) ? fmt(med) : NA}</span>`;
}
const CH_L = 46, CH_R = 8, warmColor = "#e0a93a";
// the time under the cursor, from any chart (they share the window and the margins)
function chartTimeAt(cv, x) {
  const last = charts.samples.length ? charts.samples[charts.samples.length - 1].t : Date.now() / 1000;
  const pw = cv.clientWidth - CH_L - CH_R;
  return last - charts.win + (Math.min(Math.max(x - CH_L, 0), pw) / pw) * charts.win;
}

function renderCharts() {
  const box = $("d-chart-win");
  if (box && !box.childElementCount) {
    box.innerHTML = WINDOWS.map(([l, s]) => `<button type="button" data-w="${s}">${l}</button>`).join("");
    box.addEventListener("click", (e) => {
      const w = Number(e.target.dataset && e.target.dataset.w);
      if (w) {
        charts.win = w; renderCharts();
        if (charts.server) try { localStorage.setItem(HIST_KEY, JSON.stringify({ win: w, samples: [] })); } catch { /* */ }
      }
    });
    for (const id of ["d-ch-dec", "d-ch-pre", "d-ch-act", "d-ch-ctx"]) {
      const cv = $(id);
      cv.addEventListener("mousemove", (e) => { charts.hover = chartTimeAt(cv, e.offsetX); renderCharts(); });
      cv.addEventListener("mouseleave", () => { charts.hover = null; renderCharts(); });
      // a click pins the moment on all charts (a second click near it, or Esc, unpins)
      cv.addEventListener("click", (e) => {
        const t = chartTimeAt(cv, e.offsetX);
        charts.pin = isNum(charts.pin) && Math.abs(charts.pin - t) < charts.win / 100 ? null : t;
        renderCharts();
      });
    }
    window.addEventListener("resize", () => renderCharts());
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && charts.pin != null) { charts.pin = null; renderCharts(); } });
  }
  if (box) for (const b of box.children) b.classList.toggle("on", Number(b.dataset.w) === charts.win);
  const acc = "#5b8cff", ok = "#35c76a", warm = "#e0a93a", violet = "#b07cff";
  const f0 = (v) => v.toFixed(0), f1 = (v) => (v < 100 ? v.toFixed(1) : v.toFixed(0));
  drawChart("d-ch-dec", "d-cv-dec", [{ key: "dec", color: acc, width: 2 }], f1, false);
  drawChart("d-ch-pre", "d-cv-pre", [{ key: "pre", color: warm, width: 2 }], f0, false);
  drawChart("d-ch-act", "d-cv-act", [{ key: "act", color: ok, width: 2 }, { key: "decn", color: acc, width: 1 }, { key: "pren", color: warm, width: 1 }], f0, true);
  drawChart("d-ch-ctx", "d-cv-ctx", [{ key: "ctx", color: violet, width: 2 }, { key: "ctxMax", color: dimColor(), width: 1 }], (v) => fmtNum(v), true);
}
function dimColor() { return getComputedStyle(document.documentElement).getPropertyValue("--text-dim").trim() || "#9a9ca3"; }

function render(d) {
  const banner = $("d-banner");
  if (!d.available) {
    banner.textContent = `Engine sessions unavailable: ${d.reason || "unknown"}. Process, model and system facts below are still live.`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
  const parts = [renderHeader, renderLive, renderTotals, renderCharts, renderLimits, renderRequests, renderScheduler, renderConfig, renderRaw];
  for (const f of parts) {
    try { f(d); } catch (e) { console.error(f.name, e); }   // one malformed field never blanks the whole page
  }
}

// ------------------------------------------------------------------------------------------ refresh loop

async function refresh() {
  if (ui.inflight) return;
  ui.inflight = true;
  const t0 = performance.now();
  try {
    const r = await fetch("/api/engine/overview", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    ui.data = await r.json();
    try { await chartHistory(ui.data); } catch (e) { console.error("chartHistory", e); }
    render(ui.data);
    $("d-updated").textContent = `updated ${fmtClock(ui.data.at)} · ${Math.round(performance.now() - t0)} ms`;
  } catch (e) {
    const banner = $("d-banner");
    banner.textContent = `Studio server did not answer (${e.message}). Retrying…`;
    banner.classList.remove("hidden");
    $("d-dot").className = "dot dot-off";
  } finally {
    ui.inflight = false;
  }
}

function schedule() {
  clearTimeout(ui.timer);
  ui.timer = setTimeout(async () => {
    if (!document.hidden && !ui.paused) await refresh();
    schedule();
  }, REFRESH_MS);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && !ui.paused) { refresh(); schedule(); }
});
$("d-pause").addEventListener("change", (e) => {
  ui.paused = e.target.checked;
  $("d-updated").classList.toggle("paused", ui.paused);
  if (!ui.paused) refresh();
});
$("d-include-old").addEventListener("change", (e) => {
  ui.includeOld = e.target.checked;
  if (ui.data) renderRequests(ui.data);
});
$("d-rec-reset").addEventListener("click", async () => {
  if (!confirmRecordsReset()) return;
  try {
    const r = await fetch("/api/engine/records/reset", { method: "POST" });
    if (r.ok) { charts.records = await r.json(); renderRecords(); }
  } catch { /* the next refresh shows the server as unreachable */ }
});
// two clicks within 4 s: no browser dialog (the dashboard is often on a wall screen)
function confirmRecordsReset() {
  const b = $("d-rec-reset"), now = Date.now();
  if (b.dataset.armed && now - Number(b.dataset.armed) < 4000) { delete b.dataset.armed; b.textContent = "reset records"; return true; }
  b.dataset.armed = String(now); b.textContent = "click again to reset";
  setTimeout(() => { if (b.dataset.armed && Date.now() - Number(b.dataset.armed) >= 4000) { delete b.dataset.armed; b.textContent = "reset records"; } }, 4100);
  return false;
}
$("d-reset").addEventListener("click", async () => {
  try {
    const r = await fetch("/api/stats/reset", { method: "POST" });
    if (!r.ok) {
      const banner = $("d-banner");
      banner.textContent = "Reset failed: " + ((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
      banner.classList.remove("hidden");
    }
  } catch { /* the next refresh shows the server as unreachable */ }
  refresh();
});
// toggle does not bubble: capture it to remember which tree nodes are open
$("d-raw").addEventListener("toggle", (e) => {
  const path = e.target.dataset && e.target.dataset.path;
  if (path == null) return;
  if (e.target.open) ui.openPaths.add(path); else ui.openPaths.delete(path);
}, true);
$("d-raw-box").addEventListener("toggle", () => { if (ui.data) renderRaw(ui.data); });

refresh();
schedule();

"use strict";

const state = { conversationId: null, conversations: [], sending: false, attachments: [] };

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send");
const attachmentsEl = $("attachments");

function fmtBytes(b) {
  if (!b) return "—";
  const gb = b / 1e9;
  return gb >= 1 ? gb.toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB";
}
function fmtUptime(s) {
  if (!s) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}

// ---------------------------------------------------------------- conversations

async function refreshConversations() {
  const r = await fetch("/api/conversations");
  state.conversations = await r.json();
  renderConvList();
}

function renderConvList() {
  const list = $("conv-list");
  list.innerHTML = "";
  for (const c of state.conversations) {
    const row = document.createElement("div");
    row.className = "conv-item" + (c.id === state.conversationId ? " active" : "");
    const title = document.createElement("span");
    title.className = "conv-title";
    title.textContent = c.title || "New chat";
    title.onclick = () => loadConversation(c.id);
    const del = document.createElement("button");
    del.className = "conv-del";
    del.textContent = "×";
    del.onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (state.conversationId === c.id) { state.conversationId = null; messagesEl.innerHTML = ""; }
      await refreshConversations();
    };
    row.appendChild(title);
    row.appendChild(del);
    list.appendChild(row);
  }
}

async function newConversation() {
  const r = await fetch("/api/conversations", { method: "POST" });
  const conv = await r.json();
  await refreshConversations();
  await loadConversation(conv.id);
}

async function loadConversation(id) {
  state.conversationId = id;
  const r = await fetch(`/api/conversations/${id}`);
  const conv = await r.json();
  messagesEl.innerHTML = "";
  for (const m of conv.messages) renderMessage(m.role, m.content, m.reasoning_content);
  renderConvList();
  scrollToBottom();
}

function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function splitAttachmentBlocks(content) {
  // A user message this app sent may be `[Attached file: x]\n...\n[End of attachment]\n\n` blocks
  // followed by what the person actually typed -- collapse that prefix to a short caption so a
  // reloaded conversation does not render a giant wall of extracted document text as the bubble.
  const marker = "[End of attachment]\n\n";
  let count = 0, idx = -1, from = 0;
  while (true) {
    const at = content.indexOf(marker, from);
    if (at === -1) break;
    count++; idx = at + marker.length; from = idx;
  }
  if (count === 0) return { caption: null, text: content };
  return { caption: `${count} attachment${count > 1 ? "s" : ""}`, text: content.slice(idx) };
}

function renderMessage(role, content, reasoning) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  let displayContent = content;
  if (role === "user") {
    const split = splitAttachmentBlocks(content || "");
    if (split.caption) {
      const cap = document.createElement("div");
      cap.className = "attach-caption";
      cap.textContent = "📎 " + split.caption;
      wrap.appendChild(cap);
      displayContent = split.text;
    }
  }
  if (reasoning) {
    const details = document.createElement("details");
    details.className = "reasoning";
    const summary = document.createElement("summary");
    summary.textContent = "thinking";
    details.appendChild(summary);
    const body = document.createElement("div");
    body.textContent = reasoning;
    details.appendChild(body);
    wrap.appendChild(details);
  }
  const body = document.createElement("div");
  body.className = "content";
  body.textContent = displayContent || "";
  wrap.appendChild(body);
  messagesEl.appendChild(wrap);
  return wrap;
}

// -------------------------------------------------------------------- attachments

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result.slice(r.result.indexOf(",") + 1));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

function renderAttachments() {
  attachmentsEl.innerHTML = "";
  attachmentsEl.classList.toggle("hidden", state.attachments.length === 0);
  state.attachments.forEach((a, i) => {
    const chip = document.createElement("div");
    chip.className = "chip" + (a.error ? " chip-error" : "");
    const name = document.createElement("span");
    name.className = "chip-name";
    name.textContent = a.error ? `${a.label} (${a.error})` : `${a.label} · ${a.chars.toLocaleString()} chars${a.truncated ? " (truncated)" : ""}`;
    name.title = name.textContent;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "chip-remove";
    rm.textContent = "×";
    rm.onclick = () => { state.attachments.splice(i, 1); renderAttachments(); };
    chip.appendChild(name);
    chip.appendChild(rm);
    attachmentsEl.appendChild(chip);
  });
}

async function addFileAttachment(file) {
  const placeholder = { label: file.name, chars: 0, loading: true };
  state.attachments.push(placeholder);
  renderAttachments();
  try {
    const data_base64 = await fileToBase64(file);
    const r = await fetch("/api/attachments/upload", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, data_base64 }),
    });
    const body = await r.json();
    const idx = state.attachments.indexOf(placeholder);
    if (!r.ok) {
      state.attachments[idx] = { label: file.name, error: body.error || "failed to read" };
    } else {
      state.attachments[idx] = { label: file.name, text: body.text, chars: body.chars, truncated: body.truncated, kind: "file" };
    }
  } catch (e) {
    const idx = state.attachments.indexOf(placeholder);
    if (idx !== -1) state.attachments[idx] = { label: file.name, error: String(e) };
  }
  renderAttachments();
}

async function addUrlAttachment(url) {
  const placeholder = { label: url, chars: 0, loading: true };
  state.attachments.push(placeholder);
  renderAttachments();
  try {
    const r = await fetch("/api/fetch_url", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }),
    });
    const body = await r.json();
    const idx = state.attachments.indexOf(placeholder);
    if (!r.ok) {
      state.attachments[idx] = { label: url, error: body.error || "failed to fetch" };
    } else {
      const label = body.title ? `${body.title} (${url})` : url;
      state.attachments[idx] = { label, text: body.text, chars: body.chars, truncated: body.truncated, kind: "url" };
    }
  } catch (e) {
    const idx = state.attachments.indexOf(placeholder);
    if (idx !== -1) state.attachments[idx] = { label: url, error: String(e) };
  }
  renderAttachments();
}

function buildOutgoingText(userText) {
  const blocks = state.attachments.filter((a) => a.text).map((a) => {
    const header = a.kind === "url" ? `[Content fetched from ${a.label}]` : `[Attached file: ${a.label}]`;
    return `${header}\n${a.text}\n[End of attachment]`;
  });
  return blocks.length ? blocks.join("\n\n") + "\n\n" + userText : userText;
}

$("attach-btn").onclick = () => $("file-input").click();
$("file-input").addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  for (const f of files) await addFileAttachment(f);
});
$("attach-url-btn").onclick = async () => {
  const url = window.prompt("URL to fetch and attach as context:");
  if (url && url.trim()) await addUrlAttachment(url.trim());
};

// ------------------------------------------------------------------------ chat

async function sendMessage() {
  const text = inputEl.value.trim();
  const readyAttachments = state.attachments.filter((a) => a.text);
  const pendingAttachments = state.attachments.some((a) => a.loading);
  if ((!text && readyAttachments.length === 0) || state.sending || pendingAttachments) return;
  if (!state.conversationId) await newConversation();

  const outgoing = buildOutgoingText(text || "Please look at the attached content.");
  renderMessage("user", outgoing);
  inputEl.value = "";
  state.attachments = [];
  renderAttachments();
  autosize();
  scrollToBottom();

  state.sending = true;
  sendBtn.disabled = true;
  const assistantEl = renderMessage("assistant", "");
  let reasoningEl = null, reasoningBody = null, contentEl = assistantEl.querySelector(".content");
  let reasoning = "", content = "";

  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: state.conversationId, message: outgoing }),
  });
  if (!resp.ok) {
    contentEl.textContent = "(error: " + resp.status + " " + (await resp.text()) + ")";
    state.sending = false; sendBtn.disabled = false;
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      let obj;
      try { obj = JSON.parse(payload); } catch { continue; }
      if (obj.error) { contentEl.textContent = "(error: " + obj.error + ")"; continue; }
      if (obj.engine_stats && Object.keys(obj.engine_stats).length) updateLiveStats(obj.engine_stats);
      for (const ch of obj.choices || []) {
        const d = ch.delta || {};
        if (d.reasoning_content) {
          reasoning += d.reasoning_content;
          if (!reasoningEl) {
            reasoningEl = document.createElement("details");
            reasoningEl.className = "reasoning";
            reasoningEl.open = true;
            const s = document.createElement("summary");
            s.textContent = "thinking";
            reasoningEl.appendChild(s);
            reasoningBody = document.createElement("div");
            reasoningEl.appendChild(reasoningBody);
            assistantEl.insertBefore(reasoningEl, contentEl);
          }
          reasoningBody.textContent = reasoning;
        }
        if (d.content) { content += d.content; contentEl.textContent = content; }
      }
      scrollToBottom();
    }
  }
  if (reasoningEl) reasoningEl.open = false;
  state.sending = false;
  sendBtn.disabled = false;
  await refreshConversations();
  await refreshCumulativeStats();
}

function updateLiveStats(s) {
  if (s.ttft_ms != null) $("stat-ttft").textContent = "TTFT " + (s.ttft_ms / 1000).toFixed(2) + "s";
  if (s.decode_tokens_per_second != null) {
    let txt = "decode " + s.decode_tokens_per_second.toFixed(1) + " tok/s";
    if (s.mtp_rounds > 0 && s.mtp_drafted > 0) txt += ` (MTP K=${s.mtp_depth}, ${(100 * s.mtp_accepted / s.mtp_drafted).toFixed(0)}% accepted)`;
    if (s.think_guard) txt += ` · thinking closed by ${s.think_guard} guard at ${s.think_guard_at} tok`;
    $("stat-decode").textContent = txt;
  }
  if (s.prefill_tokens_per_second != null) $("stat-prefill").textContent = "prefill " + s.prefill_tokens_per_second.toFixed(0) + " tok/s";
  if (s.cache_total_tokens) {
    const pct = (100 * (s.cache_hit_tokens || 0) / s.cache_total_tokens).toFixed(0);
    $("stat-cache").textContent = `cache ${s.cache_hit_tokens}/${s.cache_total_tokens} (${pct}%)`;
  } else {
    $("stat-cache").textContent = "cache —";
  }
}

async function refreshCumulativeStats() {
  const r = await fetch("/api/stats");
  const s = await r.json();
  $("c-requests").textContent = s.requests;
  $("c-prompt").textContent = s.prompt_tokens;
  $("c-completion").textContent = s.completion_tokens;
  $("c-cache").textContent = s.cache_hit_pct + "%";
  $("c-avg-decode").textContent = s.avg_decode_tokens_per_second + " tok/s";
  $("c-peak-mem").textContent = fmtBytes(s.peak_gpu_memory_bytes);
  $("c-uptime").textContent = fmtUptime(s.uptime_s);
}

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(200, inputEl.scrollHeight) + "px";
}

// --------------------------------------------------------------------- server status / settings

let statusPoll = null;

async function refreshStatus() {
  const r = await fetch("/api/status");
  const s = await r.json();
  const dot = $("status-dot");
  dot.className = "dot " + (s.state === "running" ? "dot-on" : s.state === "starting" ? "dot-loading" : "dot-off");
  if (s.state === "starting") {
    const secs = Math.floor(s.starting_elapsed_s || 0);
    $("model-label").textContent = `loading model… ${secs}s`;
  } else if (s.state === "running") {
    $("model-label").textContent = (s.model_dir ? s.model_dir.split("/").pop() : "engine serve") + `  ·  :${s.port}`;
  } else if (s.state === "error") {
    $("model-label").textContent = "server error";
  } else {
    $("model-label").textContent = "server stopped";
  }
  fillSettingsForm(s);
  const anyStarting = s.state === "starting" || (s.tunnel && s.tunnel.state === "starting");
  if (anyStarting && !statusPoll) {
    statusPoll = setInterval(refreshStatus, 1000);
  } else if (!anyStarting && statusPoll) {
    clearInterval(statusPoll);
    statusPoll = null;
  }
  return s;
}

function fillSettingsForm(s) {
  const cfg = s.config || {};
  $("s-model").value = cfg.model_dir || "";
  $("s-port").value = cfg.engine_port || 8080;
  $("s-tokens").value = cfg.max_tokens || 262144;
  $("s-effort").value = cfg.reasoning_effort || "xhigh";
  $("s-cache").checked = !!cfg.state_cache_enabled;
  $("s-cache-step").value = cfg.state_cache_step || 512;
  $("s-mtp").value = cfg.mtp_depth == null ? 3 : cfg.mtp_depth;
  $("s-bias-max").value = cfg.think_bias_max == null ? 12 : cfg.think_bias_max;
  $("s-bias-start").value = cfg.think_bias_start == null ? 2000 : cfg.think_bias_start;
  $("s-bias-full").value = cfg.think_bias_full == null ? 8000 : cfg.think_bias_full;
  $("s-bias-deadline").value = cfg.think_bias_deadline == null ? 14000 : cfg.think_bias_deadline;
  $("s-think-budget").value = cfg.thinking_budget == null ? 0 : cfg.thinking_budget;
  $("s-loop-guard").value = cfg.loop_guard == null ? 3 : cfg.loop_guard;
  $("s-temp").value = cfg.temperature == null ? 1.0 : cfg.temperature;
  $("s-top-p").value = cfg.top_p == null ? 0.95 : cfg.top_p;
  $("s-top-k").value = cfg.top_k == null ? 20 : cfg.top_k;
  $("s-external").checked = !!cfg.external;
  if (s.state === "starting") {
    const secs = Math.floor(s.starting_elapsed_s || 0);
    $("api-url-line").textContent = `Loading model… ${secs}s (a large checkpoint can take a few minutes on a cold disk read).`;
  } else if (s.state === "error") {
    $("api-url-line").textContent = "Error: " + (s.error || "unknown");
  } else if (s.state === "running") {
    $("api-url-line").textContent = `API: ${s.api_url_local}  (chat endpoint)   ·   on this network: ${s.api_url_lan}`;
  } else {
    $("api-url-line").textContent = "Start the server to see its API URL.";
  }
  $("start-server").disabled = s.state === "starting";
  $("start-server").textContent = s.state === "starting" ? "Starting…" : "Start server";
  $("s-token").value = cfg.api_token || "";
  fillTunnelStatus(s.tunnel || {});
}

function fillTunnelStatus(t) {
  const line = $("tunnel-status-line");
  if (t.state === "running") {
    line.innerHTML = `Tunnel: running at <a href="${t.url}" target="_blank" rel="noopener">${t.url}</a>`;
  } else if (t.state === "starting") {
    line.textContent = "Tunnel: starting… (Cloudflare is assigning an address)";
  } else if (t.state === "error") {
    line.textContent = "Tunnel: error — " + (t.error || "unknown");
  } else {
    line.textContent = "Tunnel: stopped.";
  }
  $("start-tunnel").disabled = t.state === "starting" || t.state === "running";
  $("stop-tunnel").disabled = t.state !== "starting" && t.state !== "running";
}

function collectSettings() {
  return {
    model_dir: $("s-model").value.trim(),
    engine_port: parseInt($("s-port").value, 10) || 8080,
    max_tokens: parseInt($("s-tokens").value, 10) || 262144,
    reasoning_effort: $("s-effort").value,
    state_cache_enabled: $("s-cache").checked,
    state_cache_step: parseInt($("s-cache-step").value, 10) || 512,
    mtp_depth: Math.max(0, parseInt($("s-mtp").value, 10) || 0),
    think_bias_max: Math.max(0, parseFloat($("s-bias-max").value) || 0),
    think_bias_start: Math.max(0, parseInt($("s-bias-start").value, 10) || 0),
    think_bias_full: Math.max(0, parseInt($("s-bias-full").value, 10) || 0),
    think_bias_deadline: Math.max(0, parseInt($("s-bias-deadline").value, 10) || 0),
    thinking_budget: Math.max(0, parseInt($("s-think-budget").value, 10) || 0),
    loop_guard: Math.max(0, parseInt($("s-loop-guard").value, 10) || 0),
    temperature: Math.max(0, parseFloat($("s-temp").value) || 0),
    top_p: Math.min(1, Math.max(0, parseFloat($("s-top-p").value) || 0)),
    top_k: Math.max(0, parseInt($("s-top-k").value, 10) || 0),
    external: $("s-external").checked,
  };
}

// ------------------------------------------------------------------------- wiring

$("new-chat").onclick = newConversation;
$("open-settings").onclick = async () => { await refreshStatus(); $("settings-overlay").classList.remove("hidden"); };
$("close-settings").onclick = () => $("settings-overlay").classList.add("hidden");
$("save-settings").onclick = () => {
  window.__pendingCfg = collectSettings();
  $("settings-error").textContent = "Saved -- click Start server to apply (a running server keeps its current settings until restarted).";
};
$("start-server").onclick = async () => {
  $("settings-error").textContent = "";
  const cfg = window.__pendingCfg || collectSettings();
  const r = await fetch("/api/server/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) });
  if (!r.ok) { $("settings-error").textContent = (await r.json()).error || "failed to start"; return; }
  await refreshStatus();
};
$("stop-server").onclick = async () => { await fetch("/api/server/stop", { method: "POST" }); await refreshStatus(); };
$("start-tunnel").onclick = async () => {
  $("tunnel-error").textContent = "";
  const r = await fetch("/api/tunnel/start", { method: "POST" });
  if (!r.ok) { $("tunnel-error").textContent = (await r.json()).error || "failed to start tunnel"; }
  await refreshStatus();
};
$("stop-tunnel").onclick = async () => { await fetch("/api/tunnel/stop", { method: "POST" }); await refreshStatus(); };
$("copy-token").onclick = async () => {
  try { await navigator.clipboard.writeText($("s-token").value); $("copy-token").textContent = "✓"; setTimeout(() => { $("copy-token").textContent = "⧉"; }, 1200); }
  catch { $("s-token").select(); document.execCommand("copy"); }
};

// ---------------------------------------------------------------------------- traffic

async function refreshTraffic() {
  const r = await fetch("/api/traffic");
  const d = await r.json();
  $("traffic-total").textContent = `(${d.total} total)`;
  const log = $("traffic-log");
  log.innerHTML = "";
  for (const e of d.entries.slice(0, 30)) {
    const row = document.createElement("div");
    row.className = "traffic-row";
    row.innerHTML =
      `<span class="t-time">${e.ts}</span>` +
      `<span class="t-path">${e.method} ${e.path}</span>` +
      `<span class="t-status ${e.status >= 200 && e.status < 400 ? "ok" : "err"}">${e.status}</span>` +
      (e.via_tunnel ? `<span class="t-tunnel" title="via tunnel, source ${e.source}">🌐</span>` : "");
    log.appendChild(row);
  }
}

// ------------------------------------------------------------------- all sessions (engine-wide)

function esEsc(t) { const d = document.createElement("div"); d.textContent = String(t); return d.innerHTML; }
function esNum(n) { return n >= 10000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : String(n); }
function esSecs(s) { return s >= 90 ? fmtUptime(s) : s.toFixed(1) + "s"; }

function esCard(r, finished) {
  const phase = finished ? (r.finish_reason === "aborted" ? "aborted" : r.finish_reason) : r.phase;
  const badgeCls = finished ? (r.finish_reason === "aborted" ? "err" : "") : r.phase;
  const meta = [];
  meta.push(`prompt ${esNum(r.prompt_tokens)}` + (r.cached_tokens ? ` (${esNum(r.cached_tokens)} ${r.cache_source})` : ""));
  if (!finished && r.phase === "prefill") {
    meta.push(`prefilled ${esNum(r.prefilled_to)}`);
    if (r.prefill_tokens_per_second) meta.push(`${r.prefill_tokens_per_second.toFixed(0)} tok/s`);
  } else {
    meta.push(`out ${esNum(r.generated)}${finished ? "" : "/" + esNum(r.max_tokens)}`);
    if (r.decode_tokens_per_second) meta.push(`${r.decode_tokens_per_second.toFixed(1)} tok/s`);
    if (r.ttft_ms != null) meta.push(`TTFT ${(r.ttft_ms / 1000).toFixed(2)}s`);
  }
  if (r.mtp_acceptance != null) meta.push(`MTP ${(100 * r.mtp_acceptance).toFixed(0)}%`);
  else if (r.mtp) meta.push("MTP");
  meta.push(esSecs(r.elapsed_s));
  let bar = "";
  if (!finished && r.phase === "prefill" && r.prompt_tokens > 0) {
    const c = 100 * r.cached_tokens / r.prompt_tokens;
    const d = Math.max(0, 100 * (r.prefilled_to - r.cached_tokens) / r.prompt_tokens);
    bar = `<div class="es-bar"><div class="cached" style="width:${c}%"></div><div class="done" style="width:${d}%"></div></div>`;
  }
  return `<div class="es-card"><div class="top"><span class="client" title="${esEsc(r.client)}">#${r.id} ${esEsc(r.client)}</span>` +
         `<span class="badge ${badgeCls}">${esEsc(phase)}</span></div>` +
         `<div class="meta">${meta.map((m) => `<span>${esEsc(m)}</span>`).join("")}</div>${bar}</div>`;
}

async function refreshEngineSessions() {
  let d;
  try { d = await (await fetch("/api/engine/sessions")).json(); } catch { return; }
  const agg = $("es-aggregate"), act = $("es-active");
  if (!d.available) {
    $("es-sub").textContent = "";
    agg.innerHTML = "";
    act.innerHTML = `<div class="es-empty">${esEsc(d.reason || "unavailable")}</div>`;
    $("es-recent").innerHTML = ""; $("es-recent-count").textContent = "";
    return;
  }
  const a = d.aggregate, t = d.totals;
  $("es-sub").textContent = d.since
    ? `${d.model} · since reset ${new Date(d.since * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
    : `${d.model} · up ${fmtUptime(d.uptime_s)}`;
  const tile = (v, l) => `<div class="es-tile"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  agg.innerHTML =
    tile(`${a.active}/${d.max_concurrent}`, "active") +
    tile(a.prefilling, "prefilling") +
    tile(a.decoding, "decoding") +
    tile(a.decode_tokens_per_second ? a.decode_tokens_per_second.toFixed(0) : "0", "tok/s now") +
    tile(esNum(t.requests), "requests") +
    tile(t.prompt_tokens ? Math.min(100, 100 * t.cached_tokens / t.prompt_tokens).toFixed(0) + "%" : "–", "cache hit") +
    tile(esNum(t.prompt_tokens), "tokens in") +
    tile(esNum(t.cached_tokens), "tokens cached") +
    tile(esNum(t.generated_tokens), "tokens out");
  act.innerHTML = d.active.length ? d.active.map((r) => esCard(r, false)).join("") : `<div class="es-empty">no request in flight</div>`;
  $("es-recent-count").textContent = d.recent.length ? `(${d.recent.length})` : "";
  $("es-recent").innerHTML = d.recent.slice(0, 15).map((r) => esCard(r, true)).join("");
}

$("es-reset").addEventListener("click", async () => {
  try { await fetch("/api/stats/reset", { method: "POST" }); } catch { return; }
  refreshEngineSessions();
  refreshCumulativeStats();
});
$("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

(async function init() {
  await refreshConversations();
  const s = await refreshStatus();
  await refreshCumulativeStats();
  await refreshTraffic();
  if (!s.running) $("settings-overlay").classList.remove("hidden");
  setInterval(refreshStatus, 5000);
  setInterval(refreshCumulativeStats, 5000);
  setInterval(refreshTraffic, 3000);
  refreshEngineSessions();
  setInterval(() => { if (!document.hidden) refreshEngineSessions(); }, 1000);
})();

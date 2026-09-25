# Engine Studio

A local chat app and control panel for `engine serve` (this repository's OpenAI-compatible HTTP
server, `Sources/EngineCLI/Serve.swift`), built for the E9 checkpoint on a Mac Studio M3 Ultra.
Chat window with streaming replies and the reasoning shown separately, conversation history,
live per-request stats (TTFT, decode/prefill tok/s, prompt-cache hit), cumulative session stats,
a view of every request the engine is serving, and an engine dashboard.

It is a plain application built on the engine, not part of it: nothing here is compiled into
`engine`. Pure Python standard library; nothing to `pip install` (PDF attachments optionally use
`pypdf` from a local `.venv/`, see `extract.py`).

## Run it

```
swift build -c release --product engine     # once, from the repository root
tools/serve.sh                               # the engine, from the repository root (recommended)
apps/engine-studio/run.sh                    # or: python3 apps/engine-studio/server.py
```

Opens `http://127.0.0.1:7860`.

## One engine, several clients

There is exactly one `engine` process per machine -- the E9 model alone is ~195 GB of GPU memory.
So the Studio is a **client first**:

- If an engine answers on the configured port (8099 by default, the port `tools/serve.sh` uses),
  the Studio **attaches** to it (`/api/status` reports `"external": true`) and never starts a second
  one. This is the normal case: start the engine with `tools/serve.sh`, then open the Studio; the
  Studio and any other OpenAI-compatible client talk to the same server.
- It only launches an engine itself when none is alive (settings panel: point it at the model
  directory, `weights/e9` by default, and click **Start server**). It then passes its own argument
  list (hot prefix store, `--max-concurrent 8`, MTP, the thinking soft stop, the reasoning effort),
  which is NOT the full production profile of `tools/serve.sh`.
- It only stops an engine **it** started. Asking it to stop one started elsewhere returns HTTP 409.

## What each stat means

- **TTFT** -- wall time from request start to the first generated token (includes prefill).
- **decode / prefill tok/s** -- this request's own rates, read from the `engine_stats` object the
  server attaches to each response. A live number for the UI, not a benchmark.
- **cache** -- `cache_hit_tokens / cache_total_tokens` for this request: how much of the prompt
  was restored from a previously cached state instead of re-computed. 0 on a conversation's first
  turn by design (nothing to hit yet); grows on later turns as the shared prefix is reused.
- **Session stats** (right panel) -- cumulative totals since the control server started: request
  count, prompt/completion tokens, aggregate cache-hit rate, average decode tok/s, peak GPU
  memory, uptime.

## All sessions

The right-hand panel's **All sessions** block is the engine's own `GET /v1/engine/sessions`, proxied as
`/api/engine/sessions`: every request the engine is serving, whoever sent it (this app, an agent, curl), with its
phase, prefill progress, tokens out, TTFT, rates and MTP acceptance, plus the recently finished and totals since the
engine started. It shows counts and timings only, never prompt text. The block below it ("This app") is the
Studio's own chats only.

## Engine dashboard

`http://127.0.0.1:7860/engine` (the **Dashboard ↗** link in the All sessions header) is one page with everything
about the running engine, refreshed every 1.5 s while the tab is visible (a **pause** box freezes it):

- **Live**: requests in flight, memory (active / peak / MLX cache against RAM), the hot prefix cache against its
  ceiling, the RAM budget and its reservations.
- **Totals** since the last Reset and since the engine started, side by side.
- **Limits**: a derived table -- context window, the output-room rule (prompt + max_tokens + lookahead <= max_context),
  default and oversized max_tokens, reasoning effort and levels, the thinking soft stop and its deadline, concurrency,
  batching, MTP, hot cache, KV budget and bytes per token / per full-context session, prefill chunking, memory limits
  -- each with the field or flag it came from.
- **Active / recently finished requests** with every field the engine reports; **Scheduler** (phases, batch sizes,
  restack, prefill scheduling, `max_tokens_decisions` / `reasoning_effort_decisions`); **Configuration**: the process
  (PID, argv, sha256 of the binary and the metallib beside it), its parsed flags, its `ENGINE_*` / `MLX*`
  environment, the checkpoint's config facts and the machine's GPU memory sysctls.
- **Everything else**: the full raw `GET /v1/engine/sessions` payload as a collapsible tree.

It reads `GET /api/engine/overview` (read-only: it never starts, stops or POSTs to the engine). That endpoint carries
no Studio config (no API token), no environment variable but `ENGINE_*` / `MLX*` -- and of those a name that looks like
a credential (`ENGINE_API_KEY`) only as "(set, redacted)" -- and no prompt text.

## Network access

The engine itself binds `127.0.0.1` by default; `engine serve --host <non-loopback>` refuses to start without
`ENGINE_API_KEY` (clients then send `Authorization: Bearer ...`). `tools/lan_proxy.py` is a plain TCP relay that
exposes a loopback engine to private-range LAN peers without authentication.

The **"Allow connections from other devices on this network"** toggle in this app controls whether *Engine
Studio's own UI* listens on `0.0.0.0` instead of only `127.0.0.1`. `/api/*` has no authentication, so only turn it
on for a network you trust. The optional Cloudflare quick tunnel (`cloudflared`, not bundled) exposes only the
token-gated `/v1/*` passthrough; the UI and `/api/*` answer 403 through it. Whether to expose anything beyond
localhost is your decision; nothing here makes it for you.

## Layout

```
server.py       stdlib-only control/proxy server: engine serve lifecycle, chat proxy + SSE
                relay, conversation history (data/conversations/*.json), cumulative stats
extract.py      attachment / web page to plain text
web/            the single-page chat UI and the /engine dashboard (no build step, no CDN)
data/           conversations, config.json, engine_serve.log, tunnel.log -- gitignored, created on first run
run.sh          convenience launcher (starts server.py, opens the browser)
```

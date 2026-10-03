#!/usr/bin/env bash
# P106 E9/OMP profile (the production serving profile). One foreground process; no implicit disk cache.
# Profile b54 (tools/gates/gate.py LAUNCHER_PROFILE) = b53 + the P119 shared-prefix rung (ENGINE_SHARED_PREFIX_RUNG=1024)
# + the P120 admission growth window (ENGINE_ADMISSION_GROWTH_WINDOW=32768).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${ENGINE_BINARY:-$PWD/.build/release/engine}"
case "${ENGINE_SERVE_PROFILE:-throughput}" in
  throughput) ;;
  *) echo 'The OMP production launcher requires ENGINE_SERVE_PROFILE=throughput. Use engine serve directly for isolated diagnostics.' >&2; exit 2 ;;
esac
for ARG in "$@"; do
  case "$ARG" in
    --batch-min|--batch-min=*|--batch-max-rows|--batch-max-rows=*|--batch-window-ms|--batch-window-ms=*|--prefill-chunk|--prefill-chunk=*|--hot-cache-gb|--hot-cache-gb=*|--hot-keep-rungs|--hot-keep-rungs=*|--max-concurrent|--max-concurrent=*|--mtp|--mtp=*|--batch-mtp|--batch-mtp=*|--state-cache|--state-cache=*|--state-cache-step|--state-cache-step=*|--state-cache-max-gb|--state-cache-max-gb=*)
      echo 'This launcher fixes the tested OMP batch/prefill/RAM profile. Use ENGINE_DISK_CACHE=1 for optional disk storage; other profiles require a separate diagnostic engine serve invocation.' >&2
      exit 2 ;;
  esac
done
if [[ "${ENGINE_SCHEDULER:-phase}" != phase || "${ENGINE_BATCH_PREFILL:-1}" != 1 || "${ENGINE_PREFILL_ROW_PROJECTIONS:-all}" != all || "${ENGINE_SHARED_RAM_BUDGET:-1}" != 1 || "${ENGINE_MTP:-1}" != 1 || "${ENGINE_BATCH_MTP_POLICY:-auto}" != auto || "${ENGINE_H25_CANONICAL_PREFIX:-0}" != 0 || "${ENGINE_RELEASE_POOLED_PRIVATE_HISTORY:-1}" != 1 \
      || "${ENGINE_PREFILL_WIDTH_CANONICAL:-1024}" != 1024 || "${ENGINE_PREFILL_CHUNK_ALONE:-4096}" != 4096 || "${ENGINE_PREFILL_BUDGET_CUT:-1}" != 1 \
      || "${ENGINE_SHARED_PREFIX_RUNG:-1024}" != 1024 || "${ENGINE_SHARED_PREFIX_RUNG_MAX:-4}" != 4 \
      || "${ENGINE_ADMISSION_GROWTH_WINDOW:-32768}" != 32768 ]]; then
  echo 'The tested OMP profile requires phase scheduling, native prefill with all row projections, shared RAM budgeting, MTP auto, decode-origin prefix reuse, pooled release, the B40 prefill widths and the P119 shared-prefix rung at 1024 (at most 4 heads) and the P120 admission growth window 32768. Use a separate diagnostic invocation for overrides.' >&2
  exit 2
fi
# P119 (B54, tools/gates profile b54 = p119): a cold prompt whose 1024-token head was already seen cold once ends its
# first chunk at 1024; that state is kept once per distinct head (at most 4, 1/8 of the hot budget) and resumed by later
# cold prompts with the head (aider's 1321-token system prompt). Pinned, not a default: a shell value other than these
# is refused above (an unparsable one would make the binary refuse to start).
export ENGINE_SHARED_PREFIX_RUNG=1024 ENGINE_SHARED_PREFIX_RUNG_MAX=4
# P120 (B54): the hot store's share of the RAM budget follows each row's committed length (prompt + 32768, grown ahead of
# the row) instead of prompt + max_tokens; admission decisions are unchanged (they still price prompt + max_tokens).
export ENGINE_ADMISSION_GROWTH_WINDOW=32768
# Short-row batching (2026-10-02): rows join ragged batched rounds once past 64 tokens instead of past the 2048-token QSA
# budget (row-resident pool only). Measured on the real suite: 4 concurrent 118-132 tok/s vs ~90 interleaved, 8 concurrent
# 158-163 vs ~92; a batched short row attends to every position like the solo dense window, in another reduction order
# (greedy text may part at near-ties). ENGINE_BATCH_FLOOR=2048 restores the old floor.
export ENGINE_BATCH_FLOOR="${ENGINE_BATCH_FLOOR:-64}"
# --max-concurrent 16 / --batch-max-rows 16 (2026-10-02, were 8): closed-loop load of distinct real-suite prompts, 512
# tokens: 16 streams 186.6 tok/s with TTFT p50 19.7 s (8 seats + queue) -> 214.3 tok/s, TTFT 1.7 s; 12 streams 188.5 / 9.8 s ->
# 194.9 / 1.4 s; 8 streams unchanged within run-to-run spread (172.9-184.6 vs 173.4-193.0). Admission still prices memory.
# --batch-min 2 (2026-10-02, was 4): with short rows batchable, two or three concurrent chats already pay: real suite
# 2 concurrent 128 tok/s vs 97-101 interleaved, 3 concurrent 130 vs 93; a lone row never waits (rows register).
export ENGINE_SCHEDULER=phase ENGINE_BATCH_PREFILL=1 ENGINE_PREFILL_ROW_PROJECTIONS=all
export ENGINE_SHARED_RAM_BUDGET=1 ENGINE_MTP=1 ENGINE_BATCH_MTP_POLICY=auto
# P106 B40 (operator 2026-09-23): prefix reuse from the LIVE decode state too (canonical-only B27 reuse OFF: a hot
# continuation need not be bit-identical to a cold prefill); pooled private-history release kept.
# Prefill: 4096-token chunks while alone (B43; the first chunk ends at the QSA budget 2048), 1024 while shared;
# split-K-sensitive projections in 1024-row blocks, so the state is bit-identical to 1024 chunks (H50b, H57 probe at
# 16384/131072). Hot store ceiling 192 GB (raised from 128 on 2026-09-30: at 128 GB the LRU evicted still-reused prompts of a 7-prompt agent loop, 108 evictions in 45 min); the shared RAM budget only lends it what live requests do not reserve
# (H50a: 20 GB evicted live conversations, 219k tokens re-prefilled).
export ENGINE_H25_CANONICAL_PREFIX=0 ENGINE_RELEASE_POOLED_PRIVATE_HISTORY=1
export ENGINE_PREFILL_WIDTH_CANONICAL=1024 ENGINE_PREFILL_CHUNK_ALONE=4096 ENGINE_PREFILL_BUDGET_CUT=1
# H54 (B44): store-time anchor rungs -- per 8192 band of an entry's last 32k tokens, the highest inherited rung (mostly
# earlier turns' ends), charged to the hot store. They never enlarge the live ladder, so the per-row reserve is unchanged
# (H58/D52: live anchors starved the hot store at eight 258k rows).
export ENGINE_HOT_ANCHOR_WINDOW=32768
# Diagnostics belong in a separately identified direct invocation. Some switches
# test presence, so setting them to 0 would still enable them.
unset ENGINE_OWNER_ORDER_TRACE ENGINE_RESTACK_EVAL ENGINE_INDEXER_ROUTE_WITNESS ENGINE_IDX_REPLAY_DUMP \
  ENGINE_IDX_REPLAY_BUDGETS ENGINE_IDX_REPLAY_TAG ENGINE_EXPOSE_TOKEN_IDS \
  ENGINE_HOT_DEBUG ENGINE_PREFILL_PROFILE ENGINE_PROFILE_MIN_ROWS \
  ENGINE_BATCH_SYNC_PROBE ENGINE_IDX_SCORE_DEC_WITNESS ENGINE_RAGGED_WITNESS \
  ENGINE_VISION_WITNESS ENGINE_ROUTER_STATS ENGINE_DUMP_ACTS \
  ENGINE_DUMP_CALLS ENGINE_DUMP_HC0 ENGINE_DUMP_LAYERS \
  ENGINE_DUMP_ROUTER ENGINE_DUMP_ROUTER_N ENGINE_DUMP_TOP \
  ENGINE_MTP_ROUND_TRACE ENGINE_MTP_MARGIN_DUMP ENGINE_SERVE_NO_WARM \
  ENGINE_TEMPLATE_CACHE_VERIFY ENGINE_ROUND_TRACE ENGINE_ROUND_TRACE_PROBS ENGINE_ROUND_D1_DEVICE
# P106 B32 (H42): ENGINE_TEMPLATE_CACHE_MB defaults to 256 inside the binary; VERIFY is a
# diagnostic double-check (HTTP 500 on mismatch) and must never run in production.
export ENGINE_CACHE_LIMIT_GB="${ENGINE_CACHE_LIMIT_GB:-32}"
# A directory setting alone never turns disk storage on.
DISK_ARGS=()
case "${ENGINE_DISK_CACHE:-0}" in
  0) ;;
  1) DISK_ARGS=(--state-cache "${ENGINE_STATE_CACHE_DIR:-$HOME/.cache/engine-prefix}" --state-cache-max-gb "${ENGINE_STATE_CACHE_MAX_GB:-192}") ;;
  *) echo 'ENGINE_DISK_CACHE must be 0 or 1.' >&2; exit 2 ;;
esac
# macOS 27: keep the measured unwired default; an explicit operator value wins.
# This launcher makes no system-wide sysctl changes.
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 27 ]]; then
  export ENGINE_WIRED_LIMIT_GB="${ENGINE_WIRED_LIMIT_GB:-0}"
fi
[[ -x "$BIN" ]] || { echo 'Build engine first.' >&2; exit 2; }
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd -P)"
export MLXFAST_MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$BIN_DIR/mlx.metallib}"
[[ -f "$MLXFAST_MLX_METALLIB" ]] || { echo 'Install the matching metallib beside engine first.' >&2; exit 2; }
if pgrep -x engine >/dev/null; then
  echo 'An engine process is already running; drain and stop it before starting another.' >&2
  exit 95
fi
# P121: macOS's GPU wired collector (on by default, back on after every reboot) unwires the model within about a minute and
# decode then decays request by request to single-digit tok/s (measured here on macOS 27.0, 2026-09-21; a third party's
# llm_context_benchmarks run looked exactly like it). Only a warning: changing it needs sudo, and production must still start.
if [[ "$(/usr/sbin/sysctl -n iogpu.disable_wired_collector 2>/dev/null || echo unknown)" != 1 ]]; then
  echo 'WARNING: iogpu.disable_wired_collector is not 1 -- macOS will unwire the model and decode will slow to single-digit tok/s. Run: sudo sysctl iogpu.disable_wired_collector=1, or once for every boot: sudo tools/install_wired_collector_boot.sh. Without sudo, ENGINE_WIRED_LIMIT_GB=200 avoids the decode stalls but prefill stays about 25 % slower (P123).' >&2
fi
# 2026-10-02 (operator: 1M context): up to 4x the checkpoint context (1048576 tokens), validated by needle tests at
# 260k/500k/1M (retrieval past the trained 262k exact); past 262144 tokens a lone prefill uses 1024-token chunks.
export ENGINE_CONTEXT_EXTENSION=1
# 2026-10-02 (T-0042, D-0007): copy mode -- after a round whose last 6 tokens repeat an earlier span of the prompt or the
# output (the most recent 64k tokens), verify up to 12 looked-up tokens instead of MTP drafts (single-stream rounds).
# Measured: 10 code edits 120.2 -> 150.7 tok/s, real suite 91.4 -> 90.7, graded quality and the copy check unchanged
# (C-0094, C-0107); the index is bounded because building it over 1M tokens stalls the model thread ~0.4 s (C-0111).
export ENGINE_DRAFT_COPY="${ENGINE_DRAFT_COPY:-1}" ENGINE_DRAFT_COPY_MIN="${ENGINE_DRAFT_COPY_MIN:-6}" ENGINE_DRAFT_COPY_L="${ENGINE_DRAFT_COPY_L:-12}"
# 2026-10-02 (T-0043/T-0045, D-0009): the firmer think soft stop P1 (bias 16, ramp 1000 -> 4000, deadline 8000), acting only
# right after a token that ends a line until the deadline (--think-bias-gate line), so a stop never lands mid-sentence. On
# the 38 graded coding tasks at xhigh: 38/38 in 850 s, no thinking cut mid-sentence, against the old 12 / 2000 -> 8000 /
# 14000 ramp's 35/38 in 1,646 s (C-0103, C-0118).
# 2026-10-03 (T-0048): decode time share -- with a prefill waiting, decode may use up to half of the last prefill dispatch's
# time, the model thread waiting up to 5 ms for a decoding row's next step; a long-context chunk (~1.3 s at 1M) then leaves
# room for chats while a short prompt's chunk leaves none (a fixed burst of 16 had starved short-context prefill: N=8
# 158.1 vs 169.7 tok/s). A request the KV budget cannot take waits for room instead of a 503 (ENGINE_ADMISSION_WAIT, code
# default), and copy mode stops at 64k-token contexts (C-0122).
# 2026-10-03 (T-0051): the admission bound prices 2 copies of each row's history instead of 3 -- row-resident KV keeps
# one, written in place. Thirteen 128k rows peaked at ~7.3 GB each against a 15 GB price, so a 16-session 128k burst left
# three requests waiting 1500 s (then 503) with 180 GB free; at 2x all 16 fit (~10.5 GB priced per row).
export ENGINE_DECODE_AFFINITY_MS="${ENGINE_DECODE_AFFINITY_MS:-5}" ENGINE_DECODE_SHARE="${ENGINE_DECODE_SHARE:-0.5}"
export ENGINE_ADMISSION_HISTORY_FACTOR="${ENGINE_ADMISSION_HISTORY_FACTOR:-2}"
exec "$BIN" serve --model "${ENGINE_MODEL:-$PWD/weights/e9}" --max-context "${ENGINE_MAX_CONTEXT:-1048576}" \
  --host "${ENGINE_HOST:-127.0.0.1}" --port "${ENGINE_PORT:-8099}" \
  --tokens 0 --max-tokens-clamp 1 --mtp 3 --batch-mtp 3 --think-bias-max 16 --think-bias-start 1000 \
  --think-bias-full 4000 --think-bias-deadline 8000 --think-bias-gate line --max-concurrent 16 \
  --reasoning-effort xhigh --hot-cache-gb 192 --hot-keep-rungs 2 \
  --batch-min 2 --batch-max-rows 16 --batch-window-ms 25 --prefill-chunk 0 --state-cache-step 512 \
  --queue-max 32 --queue-timeout-s 1500 ${DISK_ARGS[@]+"${DISK_ARGS[@]}"} "$@"

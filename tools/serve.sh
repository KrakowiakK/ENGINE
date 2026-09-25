#!/usr/bin/env bash
# P106 E9/OMP profile (the production serving profile). One foreground process; no implicit disk cache.
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
      || "${ENGINE_PREFILL_WIDTH_CANONICAL:-1024}" != 1024 || "${ENGINE_PREFILL_CHUNK_ALONE:-4096}" != 4096 || "${ENGINE_PREFILL_BUDGET_CUT:-1}" != 1 ]]; then
  echo 'The tested OMP profile requires phase scheduling, native prefill with all row projections, shared RAM budgeting, MTP auto, decode-origin prefix reuse, pooled release and the B40 prefill widths. Use a separate diagnostic invocation for overrides.' >&2
  exit 2
fi
export ENGINE_SCHEDULER=phase ENGINE_BATCH_PREFILL=1 ENGINE_PREFILL_ROW_PROJECTIONS=all
export ENGINE_SHARED_RAM_BUDGET=1 ENGINE_MTP=1 ENGINE_BATCH_MTP_POLICY=auto
# P106 B40 (operator 2026-09-23): prefix reuse from the LIVE decode state too (canonical-only B27 reuse OFF: a hot
# continuation need not be bit-identical to a cold prefill); pooled private-history release kept.
# Prefill: 4096-token chunks while alone (B43; the first chunk ends at the QSA budget 2048), 1024 while shared;
# split-K-sensitive projections in 1024-row blocks, so the state is bit-identical to 1024 chunks (H50b, H57 probe at
# 16384/131072). Hot store ceiling 128 GB: the shared RAM budget only lends it what live requests do not reserve
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
exec "$BIN" serve --model "${ENGINE_MODEL:-$PWD/weights/e9}" \
  --host "${ENGINE_HOST:-127.0.0.1}" --port "${ENGINE_PORT:-8099}" \
  --tokens 0 --max-tokens-clamp 1 --mtp 3 --batch-mtp 3 --think-bias-max 12 --think-bias-start 2000 \
  --think-bias-full 8000 --think-bias-deadline 14000 --max-concurrent 8 \
  --reasoning-effort xhigh --hot-cache-gb 128 --hot-keep-rungs 2 \
  --batch-min 4 --batch-max-rows 8 --batch-window-ms 25 --prefill-chunk 0 --state-cache-step 512 ${DISK_ARGS[@]+"${DISK_ARGS[@]}"} "$@"

# Changelog

Numbers are measured on a Mac Studio M3 Ultra (512 GB) with the E9 checkpoint, unless a line says otherwise.

## 2026-10-03

Everything since the B56 release (2026-09-28).

### Fixed

- **Autoreleased objects on the model thread.** The model thread lives as long as the server and never drained an
  autorelease pool, so objects autoreleased during its jobs were never released. Every model-thread job now runs in
  its own autorelease pool. A burst of 4 concurrent requests still leaves 52-93 CoreFoundation strings (2.5-4.5 KB)
  behind after warm-up; their source is not traced yet. Leak checks (identical traffic cycles, then the hot store and MLX's buffer cache flushed on an idle
  server) found no growth of GPU memory: a one-time +128 MB on the first batched cycle, then the floor moved by tens to
  hundreds of kilobytes per cycle of ~50 requests, with a flat buffer count and flat file and thread counts. Memory
  that grows between such checks is the hot prefix cache filling, which its budget bounds.
- **Host cost that grew with history.** The hot prefix store walked every entry, rung and array on each decode step,
  so single-stream decode fell ~6% after a busy spell until a restart; its totals are now memoised and its lookups
  and evictions no longer scale with the store. The template cache uses an O(1) LRU instead of a list scanned on
  every hit.

### Added and changed

- **Contexts up to 1M tokens** (`--max-context 1048576` in `tools/serve.sh`): a 1,017,845-token prompt prefills at
  855 tok/s (19.9 min), and answers decode at 59 tok/s at temperature 1.0 (78.5 tok/s greedy on a 981k-token
  recall document).
- **16 concurrent requests** (was 8): 16 streams 214.3 tok/s aggregate (was 186.6) with TTFT p50 1.7 s (was 19.7 s);
  short rows join batched rounds (`--batch-min 2`).
- **Admission waits instead of failing.** A request the KV budget cannot hold yet waits for room
  (`ENGINE_ADMISSION_WAIT`, on by default, bounded by `--queue-timeout-s`) instead of an immediate 503.
- **Admission sized for row-resident KV.** The bound now prices 2 copies of a row's history instead of 3
  (`ENGINE_ADMISSION_HISTORY_FACTOR`, 1-3, default 3; `tools/serve.sh` sets 2): row-resident KV keeps one copy,
  written in place. 16 parallel 128k-token sessions now run together (peak 342 GB of MLX memory, at least 137 GB
  free); before, 13 ran and 3 waited for 1500 s and got a 503, with 180 GB of memory still free.
- **Chats keep flowing during long prefills.** Decode may use up to half of the last long prefill dispatch's time
  (`ENGINE_DECODE_AFFINITY_MS=5`, `ENGINE_DECODE_SHARE=0.5`): short chats decode at ~29 tok/s while a 1M-token prompt
  prefills (was ~2.5 tok/s).
- **Thinking soft stop with a line gate** (`--think-bias-*` in `tools/serve.sh`): overlong reasoning is closed at a
  line break, never mid-sentence. On 38 graded coding tasks at reasoning effort xhigh: 38/38 in 850 s (was 35/38 in
  1,646 s).
- **Copy mode for edits** (`ENGINE_DRAFT_COPY=1`, contexts up to 64k tokens): when the answer repeats the context,
  the verify block is looked up in the context instead of drafted; code edits decode ~25% faster (120 -> 151 tok/s)
  with the same graded results.
- **Reasoning controls:** a real reasoning-off switch, `max_answer_tokens`, and `finish_phase` saying whether a
  length limit cut the reasoning or the answer.
- **Hot store:** 192 GB ceiling, rungs shared by entries with a common prefix, unused rungs trimmed before whole
  entries are evicted, a long-document spine of resume points.
- **Diagnostics:** `POST /v1/engine/flush` (`ENGINE_ADMIN_FLUSH=1`, off by default) evicts the hot store and frees
  MLX's buffer cache on an idle server, for leak checks; `ENGINE_GPU_LEDGER=1` reports GPU busy and idle time per
  window of command buffers.

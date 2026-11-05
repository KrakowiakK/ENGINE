# Changelog

Numbers are measured on a Mac Studio M3 Ultra (512 GB) with the E9 checkpoint, unless a line says otherwise.

## 2026-11-05

**What changed for throughput since 2026-10-12, measured on the same machine** (fresh servers, each release through its
own launcher; details and charts in the README's Performance section): a closed loop of 32 concurrent sessions
(~3,000-character prompts, ~1,030 tokens out, greedy) gets **286.5 tok/s with `tools/serve_e9_docs.sh` against 186.8**
for the 2026-10-12 release (**+53%**; median request 109.7 s against 171.5 s) and 216.5 tok/s (+16%) with
`tools/serve.sh` at 32 seats. With `llm_context_benchmarks` at 32 parallel sessions (greedy), all 32 finish 6-11%
sooner, aggregate prefill is 20-113% higher and the longest wait for a first token is 17-53% shorter. A single request
runs at the same speed as before. At temperature > 0 a batched request is now sampled instead of decoded greedily
(see Fixed), which lowers multi-session decode rates at temperature 1.0 against the old release's greedy output.

Everything since the 2026-10-12 release. `tools/serve.sh` stays the coding profile (16 decode seats by default); the
new `tools/serve_e9_docs.sh` runs `tools/serve.sh` with the throughput profile for many concurrent sessions (32 seats,
documents and page images). Most new paths are off in the binary and switched on by `tools/serve_e9_docs.sh`; each
bullet names the variable that turns it off again.

### Added and changed

**Concurrency and batched decode**

- **Expert-grouped GEMM for batched verify rounds** (on by default in `tools/serve.sh`, `MLX_GATHER_RHS_MIN_RATIO=0.5`):
  the routed-expert gathers take MLX's expert-grouped GEMM (`gather_qmm_rhs`) from 26 rows instead of MLX's default 205.
  At E9's batched-verify widths it reads each expert once for all its rows and is faster than the sorted gather: an MTP
  round is 4.1 / 7.6 / 9.0 / 10.1 / 14.7% faster at 12 / 16 / 20 / 24 / 32 rows and copy-draft rounds of 136-176 rows
  13-17% faster. With 32 sessions on `tools/serve_e9_docs.sh`: +10.0% throughput on the closed loop above and +7.2% on
  an extraction workload, teacher-forced output check unchanged (no committed token more than 0.5 nat below the
  argmax). Single-request decode rounds stay below the switch; short prompts' prefill switches too, which changes
  rounding (greedy outputs of short prompts can differ in later tokens). Rollback: `MLX_GATHER_RHS_MIN_RATIO=4`.
- **Up to 32 decode seats.** `ENGINE_SEATS` (1-64) sets `--max-concurrent` and `--batch-max-rows` together in
  `tools/serve.sh` (default 16, as before; the wait queue grows with it); `tools/serve_e9_docs.sh` sets 32. On a
  client-like closed loop (32 sessions, ~3,000-character prompts, up to 1,300 tokens out, a re-typing workload) 32
  sessions gave 260.95 tok/s total, 8.1% more than 20 sessions (241.40 tok/s). The extra throughput costs latency: on
  an extraction workload 32 sessions gave 7.8% more than 20 at a 51% longer median request (78 s against 52 s).
- **Faster batched verify rounds** (on by default): the GDN state roll-back runs as one kernel for all rows, the
  PLE roll-back runs as gathers, the verify-block attention gather over row-resident KV runs one dispatch
  per 8 rows, and the sparse-attention indexer handles all rows that gained blocks in one pass (one score and one top-k
  dispatch per 16 rows). A 16-row MTP round went from 193.05 to 176.65 ms with every committed token identical.
  `ENGINE_GDN_ROW_REPLAY=1`, `ENGINE_RR_VERIFY_MULTI=0` and `ENGINE_IDX_ROWS_BATCHED=0` restore the per-row paths.
- **Expert-run MoE gather for decode** (on by default): for sorted decode-sized expert gathers, one threadgroup applies
  each 8-bit weight tile it loads to a run of 2 consecutive same-expert pairs, with MLX `qmv_fast`'s exact arithmetic.
  Batched verify rounds -3.0% / -4.0% / -5.2% / -6.6% at 8 / 12 / 16 / 32 rows, committed tokens identical.
  `ENGINE_MOE_RUNS=0` restores the stock gather.
- **Fused verify-width kernels** (with `tools/serve_e9_docs.sh`): 8-bit projections and the LM head at verify widths
  without MLX `qmm`'s 32-row tile padding (`ENGINE_VERIFY_PROJ=1`; teacher-forced verify step -0.6 to -4.8% at 40-56 and
  68-80 rows, unchanged at multiples of 32), fused hyper-connection kernels up to 48 rows
  (`ENGINE_HC_VERIFY_FUSED=1`; -0.8 to -2.6% at 32-48 rows) and a tiled simdgroup-MMA sparse-attention verify gather
  (`ENGINE_QSA_VERIFY_TILED=1`; verify rounds -2.3 to -2.8% at 12-32 rows). Page-transcription accuracy was unchanged
  with each. Set any of them to `0` to turn it off.
- **GDN roll-back in place** (with `tools/serve_e9_docs.sh`, `ENGINE_GDN_ROLLBACK_INPLACE=1`): the verify roll-back
  writes over the forward's state buffer through donating custom kernels in the vendored MLX, so a fully accepted row
  costs no state traffic. Committed streams identical; decode throughput +1.2% / +3.0% on two closed-loop workloads.
  `ENGINE_DONATE_STATS=1` logs donations and every copy. Rollback: `ENGINE_GDN_ROLLBACK_INPLACE=0`.
- **Fixed batched draft depth.** `ENGINE_BATCH_MTP_POLICY=cycle:2` drafts 2 tokens in every batched round instead of the
  adaptive policy, which alternated between 2 and 3 on extraction output; `tools/serve.sh` accepts it and
  `tools/serve_e9_docs.sh` sets it: 20 sessions +1.65% throughput, fixed 3 loses 1.2%. Single-stream rounds keep 3
  drafts. `ENGINE_BATCH_MTP_K` (1-8, default 3) caps the adaptive policy's depth.
- **Batched rounds are no longer split** (with `tools/serve_e9_docs.sh`): control jobs (`ENGINE_CONTROL_PASSES_DECODE=1`)
  and decode closures such as a new row's first round (`ENGINE_DECODE_CLOSURE_PASSES=1`) run before the queued decode
  steps instead of cutting a round in two. 20 sessions: +4.5% and then +1.9% throughput.
- **Less run-time kernel compilation** (with `tools/serve_e9_docs.sh`): five prefill kernel families take the chunk's
  row count as an input (`ENGINE_RUNTIME_ROWS=1`; compile time after a fresh 32-request burst 3.4 s -> 0.33 s, outputs
  byte-identical) and the tiled verify gather reads its block count at run time (`ENGINE_QSA_TILED_RUNTIME_K=1`; 20
  sessions +2.3%).
- **Buffer-cache reuse tolerance.** `ENGINE_CACHE_REUSE_TOL=f` lets a buffer request of 1 MB or more reuse a cached MLX
  buffer up to (1+f) times its size (default 0, MLX's own rule; `tools/serve_e9_docs.sh` sets 0.25): cache misses -93%,
  decode +1.5% at 20 sessions. `ENGINE_ALLOC_STATS=1` reports allocator hits and misses every 10 s.

**Copy drafts in batched rounds**

- **Batched copy (n-gram) drafts** (with `tools/serve_e9_docs.sh`, `ENGINE_BATCH_COPY=1 ENGINE_BATCH_COPY_PACKED=1`):
  in a batched round each greedy row may draft from a lookup over its own prompt and output (a match of at least 6
  tokens, up to 7 drafts) instead of the MTP head, and rows of different draft lengths are verified together in one
  packed forward with a per-row depth. A cost gate takes a round's lookups only when they pay at the packed verify's
  cost. Head to head on the same binary, closed loop, re-typing workload: +1.1% at 32 sessions (260.95 against 258.0
  tok/s) and +1.3% at 20 sessions; on extraction output, which repeats little of its prompt, throughput was unchanged
  (the gate refused about 60% of copy rounds there). Rollback: `ENGINE_BATCH_COPY=0`.
- **Copy drafts keep the output.** A teacher-forced scan of 32 outputs (22,399 tokens) at 32 sessions found no committed
  token more than 1 nat below the model's argmax; a 20-minute soak (32 text sessions plus two 16-page document waves,
  282 requests) ended with 0 errors and flat memory (132.8-133.1 GB resident). Taking every lookup without the gate
  (`ENGINE_BATCH_COPY_ALWAYS=1`, diagnostic) loses 3.3%. `ENGINE_BATCH_COPY_COST_TABLE` (off by default) prices rounds
  from measured cost tables instead of the linear fit; it measured no gain.
- **Copy-mode entry rule.** `ENGINE_DRAFT_COPY_MIN` above 6 requires the 6-gram match to extend backwards that far
  (`tools/serve.sh` keeps 6; `tools/serve_e9_docs.sh` sets 10): on 40 chat prompts +2.45% pooled (95% interval +0.25 to
  +4.48%), with code-edit style copies keeping their gain.
- **Copy census per client** (with `tools/serve_e9_docs.sh`, `ENGINE_COPY_CENSUS=1`): after each text response a
  background job counts how much of its output a copy lookup would have drafted, and how many MTP and copy rounds,
  drafts and acceptances it had, per client label (`x-engine-client` header, else `User-Agent`), in
  `/v1/engine/sessions` under `copy_census`. Counts and labels only, no text or token ids;
  `ENGINE_COPY_CENSUS_FILE=<path>` keeps the totals across restarts.

**Prefill and scheduling (with `tools/serve_e9_docs.sh`)**

- **Arrival order and mixed rounds.** `ENGINE_PREFILL_ORDER=arrival` prefills the oldest request first instead of
  round-robin, and `ENGINE_MIXED_ROUND=1` carries the batched verify rows inside prefill chunks (one forward; chunk rows
  bit-identical). Bursts of 7 x 8.8k-token prompts: mean time to first token 78.9 -> 43.5 s, generated tok/s +3.3%.
  Rollback: `ENGINE_PREFILL_ORDER=fifo ENGINE_MIXED_ROUND=0`.
- **Packed prefill.** `ENGINE_PREFILL_PACK_ROWS=8192` packs several requests' queued prompt chunks into one forward
  (bit-identical per prompt), with the decoders' steps riding along: on a 32-request text burst the whole burst ended
  3.4-4.4% sooner and the longest time to first token fell 12-18%; the first answer starts ~7 s in instead of ~2 s.
- **Prefill kernels and chunking.** A GDN prefill recurrence with 4 value rows per simdgroup (`ENGINE_GDN_ROWS=1`;
  bit-identical, prefill +4.2% at 8192-wide forwards), 8192-token chunks for a prompt that prefills alone with the
  sparse gather kept at that width (`ENGINE_PREFILL_CHUNK_ALONE=8192 ENGINE_QSA_GATHER_MAX_S=8192`; lone time to first
  token -1.7 to -2.7%, answers identical), a prompt finished in one chunk across the sparse-attention budget cut when
  at most 4,096 tokens remain (`ENGINE_PREFILL_MERGE_TAIL`), and `ENGINE_HOT_RESERVE=0` for clients that send
  independent prompts.
- **Short shared-prefix rung.** `ENGINE_SHARED_PREFIX_RUNG=191` with `ENGINE_SHARED_PREFIX_RUNG_SHORT=1` keeps the state
  after a common 191-token head (system prompts of ~192-255 tokens); `tools/serve.sh` accepts it only together. Lone time
  to first token about -10%, answer scores unchanged; not bit-identical to a cold prefill. Rollback:
  `ENGINE_SHARED_PREFIX_RUNG=1024 ENGINE_SHARED_PREFIX_RUNG_SHORT=0`.
- **Decode share 0.1** (`ENGINE_DECODE_SHARE`, `tools/serve.sh` keeps 0.5): waiting requests start sooner and decode
  rounds run fuller: 20 sessions 176.3 -> 179.05 tok/s, mean time to first token -7%.

**Images and documents**

- **E9 image rows join batched decode.** Image requests on E9 now decode in batched rounds (they ran serially): 16
  concurrent A4 pages at 11.94 pages/min with the document profile, against 7.6 serially on the same pages, accuracy
  unchanged.
- **Faster vision tower** (with `tools/serve_e9_docs.sh`): the tower runs in bf16 (`ENGINE_VISION_BF16=1`; float32 pixels
  had promoted it to float32; tower -21.7% per page, 16-page waves +5.6% pages/min), the rotary tables are built once
  per image and the rotation compiled (`ENGINE_VISION_ROTARY=compiled`; features bit-identical, tower -3.5%), and its
  attention runs at its native head dimension 72 instead of padded to 80 (`MLX_SDPA_FULL_D72=1`; bit-identical, tower
  -4.5%). Accuracy unchanged.
- **Page waves start sooner.** An image request takes its arrival place before its vision tower runs
  (`ENGINE_VISION_ARRIVAL_STAMP=1`; 16 pages at once: first page ~20.7 -> ~4.2 s), and up to 2 younger requests' towers
  may run first so their page chunks pack (`ENGINE_PACK_TOWER_LOOKAHEAD=2`; +4.4% pages/min, each wave's first page
  ~7.5 s instead of ~3.5 s).

**Requests and operations**

- **Programming requests default to a lower reasoning effort** (with `tools/serve_e9_docs.sh`,
  `ENGINE_REASONING_EFFORT_CODING=low`; accepts `low`, `medium` or `xhigh`, unset by default). A request that names no
  `reasoning_effort` and no `enable_thinking`, and carries tool definitions, a fenced code block or at least 3 lines that
  look like source code, gets that effort instead of the server default (xhigh); an explicit effort always wins. Decode
  speed is unchanged (the 9 other single-stream suite prompts read a median 92.3 tok/s before and after); a code
  explanation prompt now runs at low effort. Unset the variable to roll back.
- **Graceful stop** (with `tools/serve_e9_docs.sh`, `ENGINE_GRACEFUL_STOP=1`): on SIGTERM the server answers new
  requests with `503` and `Retry-After`, lets admitted ones finish for up to `ENGINE_DRAIN_SECONDS` (120) and exits with
  status 143. Tested live: the request in flight finished, a new one got 503, the server then exited on its own. Off,
  SIGTERM stops the server at once, as before.
- **Committed token ids.** A non-streaming request with `"engine_return_token_ids": true` gets the committed output token
  ids in each choice's `engine_token_ids`, for scoring what was generated rather than a re-tokenization of its text.
- **Engine Studio.** The engine page has four live charts (decode tok/s, prefill tok/s, concurrency, context tokens)
  over 5 min to 3 h with one shared hover/pinned moment, history sampled by the Studio server every 1.5 s and kept for
  3 h, "Best so far" records with a median beside each, and prefill tok/s computed over a 6 s window (it was a
  per-chunk sawtooth).
- **Qwen3.5-397B-A17B on the native port.** A `qwen3_5_*` text config selects a plain Qwen3.5 mode of the native model
  (no hyper-connections or sparse indexer, the Qwen3.5 MTP head), with batched serving and image requests in batched
  decode. On the M3 Ultra, 8-bit, the 10 real-suite prompts decode at a median 48.44 tok/s single-stream with MTP K=4.
  `ENGINE_GENERIC_QWEN35=1` keeps the vendored model.

### Fixed

- **Sampled requests in batched rounds sampled again.** A request with `temperature` above 0 and no `top_k` (or a
  `top_k` at least the vocabulary size) decoded greedily whenever it shared a batched round; it now samples the full
  vocabulary (with `top_p`) as a lone request does.
- **`ENGINE_MTP=0` turned MTP on.** Any value did; MTP now needs `ENGINE_MTP=1`.
- **Connection resets at simultaneous starts.** The listen backlog was 16, so 32 clients connecting at the same moment
  had connections reset (75 failed attempts in the first 20 ms of a 32-session start); it is now 512 (the kernel caps it
  at `kern.ipc.somaxconn`, 128 on macOS), and 32 simultaneous requests all completed.
- **Short-rung prefill state.** With the 191-token shared-prefix rung, a resumed chunk's convolution state sometimes
  held stale values (3-6 bad answers per 600 under concurrency); the fused prefill convolution now writes the next
  state itself (`ENGINE_GDN_CONV_PREFILL_TAIL=1`, set by `tools/serve_e9_docs.sh`; bit-identical on cold prompts): 0 of
  1,200.
- **Plain Qwen3.5 image rows.** A hot-cache-restored image row joining a batched round crashed the server on a model
  without the sparse indexer.
- **MLX fixes from upstream.** Two more ml-explore/mlx fixes are ported onto the vendored MLX: the conv padding fill
  value is kept alive (#4660) and `fence_wait` spin-waits after every earlier encoder (#4689).

## 2026-10-12

### Fixed

- **MLX fixes from upstream.** Five fixes from ml-explore/mlx are ported onto the vendored MLX: a Metal fence that
  could wait on a later signal of the same fence and deadlock (#4552), concurrent lookups in the Metal kernel cache
  (#4043), fence tracking for donated dynamic-slice offsets (#4099), the activation row stride of sorted `gather_mm`
  (#3960) and the batched row count of `qvm_split_k` (#4673). The greedy output is identical to the previous build, and
  decode is unchanged (real suite 91.15 tok/s; 8 and 16 concurrent sessions 170.1 and 188.7 tok/s).
- **Thinking-off requests are reported as such.** A request sent with `reasoning_effort` `none` / `off` / `disabled` or
  `enable_thinking: false` now shows `off` as its effort in `/v1/engine/sessions`; it showed `low` or the server
  default although the template applies no effort when thinking is off (a one-line prompt renders to the same 15
  tokens whatever the effort).
- **Engine Studio.** The engine page no longer warns "binary rebuilt since start" for a binary deployed and started in
  the same second (the process start read from `ps etime` is rounded to the second); the "Recently finished" table
  shows thinking `off` or `on · <effort>`; the Limits panel sits above Configuration.

## 2026-10-09

### Added and changed

- **Image input on `/v1/chat/completions`.** OpenAI `image_url` parts (also `input_image`, the Anthropic base64 `source`
  and Hugging Face `image` keys, with `data:` or http(s) URLs) now reach the checkpoint's vision tower; before, the
  server kept only the text parts and answered as if no image had been sent. Images are sized before admission, the
  tower runs before the first response byte (a failure is still an HTTP error), and the prompt is prefilled with the
  image features spliced onto the template's placeholders and 3-axis mRoPE positions. Malformed image parts get a
  `400`. On a test server the answers read rendered text, colours and layout, the second of two images and a
  multi-turn follow-up correctly, past the 2,048-token sparse-attention threshold too, and the greedy output was
  identical to the `engine generate --chat-file` image path. The tower took 80 ms for a 512x384 image and 3.1 s at the
  4,194,304-pixel cap, at most 10.6 GB above the active memory.
- **The vision tower matches the reference.** Compared stage by stage with the mlx-vlm Qwen3-VL/Qwen3.5 tower on the E9
  weights and the Hugging Face processor: the tower's MLP used the sigmoid approximation of GELU instead of the
  checkpoint's `gelu_pytorch_tanh` (features at cosine 0.79-0.93 from the reference on identical pixels), and image
  preprocessing skipped the sRGB tone curve, so the pixels reached the tower in linear light. Both are fixed, and images
  are resized with PIL's bicubic (antialiased, 8-bit per pass) as the reference processor does: the pixels now equal the
  reference's (99.7-100% of the 8-bit values identical), and the features are within 1-2% (relative L2) of the reference
  tower computed in float32. On a fine-print test the previous build misread a reference code that now reads right.
- **Image requests use the hot prefix cache and MTP.** An image's rows are keyed by the image bytes' SHA-256, so a
  stored state is resumed only for the same image at the same place; a per-image cache of tower outputs
  (`ENGINE_VISION_FEATURE_CACHE_MB`, 2048) skips the tower for an image seen before. Image answers decoded at
  77-84 tok/s (44 without MTP), and the later turns of a screenshot conversation resumed 793-877 tokens with the first
  token in about 0.1 s.
- **Speculative sampling.** A sampled request (`temperature` above 0, `top_k` 1-1024, no logit processor) drafts by
  sampling the MTP head under its own temperature / top-k / top-p and accepts with probability min(1, p/q), resampling
  a rejected position from norm(max(0, p - q)). At `temperature 1.0, top_p 0.95, top_k 20` (the checkpoint's
  recommendation, sent by Engine Studio) the real-suite prompts decoded at 86.8 and 89.3 tok/s (median of two runs)
  against 73.0 with argmax drafts on the same binary; 400 drafted and 400 serial samples of one prompt agreed at every
  one of 6 positions (chi-square p >= 0.26). The first token of a sampled answer under MTP is now sampled; it was the
  argmax. `ENGINE_SPEC_SAMPLING=0` restores argmax drafts; greedy requests are unchanged.

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

### Open

- After repeated batched traffic the host heap still grows by about 0.3 MB per batched request even with the hot
  cache and MLX's buffer cache flushed: newly compiled Metal kernels for new shapes, a bounded index-range cache and
  untyped allocations. Whether it levels off has not been established yet; GPU memory does not grow.
- A burst of many cold requests gets its first tokens only after the whole burst has prefilled (see the README's
  16-session table and known limitations).

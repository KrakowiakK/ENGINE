# Changelog

Numbers are measured on a Mac Studio M3 Ultra (512 GB) with the E9 checkpoint, unless a line says otherwise.

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

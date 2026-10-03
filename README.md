# ENGINE

ENGINE is an OpenAI-compatible inference server written in Swift on
[MLX](https://github.com/ml-explore/mlx-swift) for one model on Apple silicon: **E9**, an 8-bit MLX
quantisation of `Qwen/Qwen3.8-Flash-Next` (model_type `qwen4_exp`). It was built and tuned for a single
machine, a Mac Studio with M3 Ultra and 512 GB of unified memory, where it serves an agentic coding workload
(many long, concurrent, tool-calling sessions with shared prefixes).

What it does:

- **Continuous batching** of up to 16 concurrent sessions: requests join and leave a running batch; decode and
  prefill are scheduled in phases, compatible prefills are batched natively, and while a long prompt prefills the
  decoding rows keep getting up to half of its time (`ENGINE_DECODE_SHARE=0.5`, `ENGINE_DECODE_AFFINITY_MS=5`).
- **MTP speculative decoding** in the production launcher's profile (`engine serve` itself defaults to `--mtp 0`,
  no MTP), using the model's own multi-token-prediction head: K = 3 for a lone request; batched rows draft up to
  K = 3, with the depth (or a plain step) chosen adaptively from measured acceptance and round cost. Some requests
  are not armed for drafting and decode without MTP: requests that ask for `logprobs`; requests with a hard
  thinking guard (`thinking_budget` or `loop_guard` above 0, per request or as a server flag), because the chat
  template opens the thinking block on every request; and requests resumed from a cache entry saved without MTP
  state. A forced tool-call prefix switches a request to serial decoding at that point.
- **Hot prefix cache in GPU memory**: a new turn can resume from the previous turn's state instead of re-prefilling
  the conversation, as long as that state is still in the cache. It shares one RAM budget with the live requests
  (ceiling 192 GB) and only gets what live reservations leave. An optional disk cache can be enabled.
- **1,048,576-token context** (`--max-context 1048576` in `tools/serve.sh`, four times the checkpoint's
  `max_position_embeddings` of 262,144). In our tests at about 1M tokens, facts were recalled verbatim, but the order of
  two far-apart facts could come out swapped (the same answer without any engine feature involved: a model limit).
- **Admission control** reserves memory before a request starts. A request that does not fit yet waits for room (up
  to the queue timeout) instead of failing at once, and only then answers `503` + `Retry-After`. The bound prices two
  copies of each row's history (`ENGINE_ADMISSION_HISTORY_FACTOR=2` in `tools/serve.sh`; the binary's default is 3):
  rows keep their own KV, written in place.
- **Memory guard** for a Mac that also runs other things: the engine never plans for more than the memory that was free
  when it started (and refuses to start, with the numbers, if the weights cannot fit), and it follows macOS's
  memory-pressure signal plus the free memory every 2 s. Under pressure the hot prefix cache first shrinks to half, then
  empties, and new requests wait (up to the queue timeout, then `503`) while running ones finish; it recovers on its own. The level is shown in
  `/v1/engine/sessions` (`memory_guard`) and logged. `ENGINE_MEMORY_GUARD=0` turns it off.
- **Thinking soft stop**: instead of a hard thinking budget, a bias on `</think>` ramps up to +16 logits between
  1,000 and 4,000 thinking tokens, and until a deadline of 8,000 thinking tokens it acts only right after a token that
  ends a line (`--think-bias-gate line`), so the model closes its reasoning at a line break, never mid-sentence. After
  4,000 tokens the bias keeps climbing, quadratically, to +56 logits at the deadline, where it acts at every token. That
  exceeds every `</think>` gap measured (about 24 logits), so the block closes in practice (DERIVED, not a measured
  guarantee). On 38 graded coding tasks at `xhigh` this profile passed 38/38 in 850 s, against 35/38 in 1,646 s for the
  previous one. A smaller `max_tokens` ends the reply first.
- **Copy mode** (`ENGINE_DRAFT_COPY=1` in `tools/serve.sh`, contexts up to 64k tokens): when the answer repeats text
  from the context (code edits), a lone request verifies blocks of up to 12 tokens looked up in the context instead of
  drafted ones; code edits decoded ~25% faster (120 -> 151 tok/s) with the same graded results.
- **Reasoning effort levels** from the checkpoint's chat template: `xhigh` (default), `medium`, `low`.
- Custom Metal kernels for the model's hyper-connection chain, MoE experts, gated-delta-net layers and
  sparse-attention gather, in `Sources/Qwen4Exp`, on a modified fork of mlx-swift / mlx-swift-lm (`Vendor/`).

**Units.** GB in this README means 10^9 bytes, as in the engine's code and logs. Apple's nominal memory sizes
(512 GB, 256 GB) are binary: a 512 GB Mac has 512 GiB, about 549.8 GB, and a 256 GB Mac about 274.9 GB.

## Hardware and software requirements

- **Tested only on a Mac Studio M3 Ultra with 512 GB**. Developed on macOS 26 and 27; the current build and every
  number below ran on macOS 27.2. The E9 weights alone take about 195 GB of GPU memory, and the admission budget,
  the hot cache ceiling and the launcher's defaults are sized for 512 GB.
- **Memory floor (DERIVED from the code, not tested):** `engine serve` sets its KV budget to
  0.90 × physical memory − (max(32, `ENGINE_CACHE_LIMIT_GB`) + 48) GB − the resident model (about 195 GB), all in
  decimal GB, and refuses to start when that is not positive ("KV budget exceeds memory remaining ..."). It
  therefore needs roughly 306 GB (about 285 GiB) or more of physical memory before any request is admitted
  (0.9 × memory − 80 GB − 195 GB > 0). In practice that means the 512 GB configuration: a 256 GB Mac (about
  274.9 GB) is refused at startup (budget about −28 GB). Anything in between is untested. At 512 GB (about
  549.8 GB) the same formula gives about 220 GB of KV budget. Without the shared RAM budget (for example an engine
  started by Engine Studio) the hot-cache ceiling is subtracted too, which raises the floor.
- Memory: the weights take 194.9 GB. A single 262,144-token prefill (chunk 4096, `engine nll`, build B44) peaked at
  252.6 GB (OBS-ENG-204), and the B50 production build peaked at 292.8 GB in the release gate's concurrent
  long-context soak test (SOAK900) and at 344.9 GB in the full-context (258k-token) gate run, where the hot prefix
  cache held about 127 GB. On 2026-10-03, 16 parallel sessions peaked at 366.9 GB of MLX memory at 32k tokens each
  (free memory never below 114 GB). These are observed peaks, not an upper bound. `tools/serve.sh` caps the MLX buffer
  cache at 32 GB.
- Xcode with the Swift 6.3 toolchain (`swift-tools-version: 6.3`), Xcode's Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) and `cmake` (`brew install cmake`) for the Metal library.
- Network access for the first build: SwiftPM fetches the Swift package dependencies (for example
  `swift-transformers` and `swift-jinja`) from GitHub.
- About 195 GB of disk for the weights.
- **Required for the documented performance: `sudo sysctl iogpu.disable_wired_collector=1`** (it resets on every
  reboot; `tools/serve.sh` warns when it is not set). Without it macOS unwires the model's memory: on this machine
  (macOS 27.0, 2026-09-21) wired memory fell from about 269 GB to about 8 GB within a minute of the server starting and
  decode decayed request by request to about 7 tok/s. A controlled toggle on 2026-09-27 (same build, same session,
  llm_context_benchmarks at 0.5k-128k): with the collector on, prefill 352-941 tok/s and single runs stalling at
  19-29 tok/s of generation; with it off, prefill 1087-1267 tok/s and generation 61-93 tok/s. If you cannot use sudo,
  `ENGINE_WIRED_LIMIT_GB=190 tools/serve.sh` wires only the model's weights: in the same test generation stayed at
  59-86 tok/s but prefill reached only 764-990 tok/s (one run); a sweep of that limit (200-260 GB) found no value
  without a loss. To keep the setting across reboots, `sudo tools/install_wired_collector_boot.sh` installs a small
  LaunchDaemon that applies it at every boot (`--uninstall` removes it).
  `tools/serve.sh` never changes sysctls itself. On macOS 27 it runs without an MLX wired limit
  (`ENGINE_WIRED_LIMIT_GB=0`, overridable).

## Build

```sh
swift build -c release --product engine    # .build/release/engine
tools/build-mlx-metallib.sh                 # .build/release/mlx.metallib, next to the binary
```

MLX loads `mlx.metallib` from the directory of the running executable, so the library must sit next to `engine`
and must be rebuilt after any change to the vendored Metal sources. The metallib build compiles the vendored MLX
kernels with CMake (build tree in `.build-worker/`). A fresh build of this export on the M3 Ultra (2026-09-25,
macOS 27.2) took about 5 minutes for the release build, including fetching the dependencies, and about 70 seconds
for the metallib.

## Get the weights

The model is **not** in this repository. Download E9 from Hugging Face into `weights/e9` (the default path of the
launcher and of Engine Studio):

```sh
pip install -U huggingface_hub
hf download aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit --local-dir weights/e9
```

Model card: <https://huggingface.co/aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit>. The weights are licensed under the
Qwen Community License 1.0 of the base model `Qwen/Qwen3.8-Flash-Next`. Read that license before using them.
To keep the weights elsewhere, symlink `weights/e9` or set `ENGINE_MODEL=/path/to/e9`.

## Run

```sh
tools/serve.sh                     # serves http://127.0.0.1:8099/v1
curl http://127.0.0.1:8099/v1/models
```

`tools/serve.sh` is the production launcher. It runs one foreground process and pins the tested profile: MTP K = 3
(batched rows up to K = 3, adaptively), up to 16 concurrent requests with up to 32 more waiting in a first-in-first-out
queue (at most 1,500 s each), batching from 2 ready rows up to 16 rows with a 25 ms gather window,
phase scheduling with native batched prefill and the decode time share, a shared RAM budget with a 192 GB hot-cache
ceiling, copy mode, the thinking soft stop, `reasoning_effort` default `xhigh`, a 1,048,576-token context, no default
`max_tokens` cap, and oversized `max_tokens` clamped. It
refuses overrides of the batching, prefill, hot-cache, concurrency, MTP and state-cache flags and of the scheduler
environment. Other trailing arguments (for example reasoning effort, the soft-stop settings, the `max_tokens`
policy or the request queue's `--queue-max` / `--queue-timeout-s`) are passed through and override the defaults, as do `ENGINE_CACHE_LIMIT_GB` and `ENGINE_WIRED_LIMIT_GB`.
Run `.build/release/engine serve ...` directly for anything else. It also refuses to start while another `engine`
process is running: run one engine per machine.

Loading ~195 GB can take from seconds (warm page cache) to several minutes (cold disk). The server answers
`GET /v1/models` once it is ready.

Useful environment variables for the launcher:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_MODEL` | `./weights/e9` | checkpoint directory |
| `ENGINE_HOST`, `ENGINE_PORT` | `127.0.0.1`, `8099` | bind address; a non-loopback host requires `ENGINE_API_KEY` |
| `ENGINE_API_KEY` | unset | when set, every request needs `Authorization: Bearer <key>` |
| `ENGINE_DISK_CACHE` | `0` | `1` adds the persistent prefix cache (`ENGINE_STATE_CACHE_DIR`, default `~/.cache/engine-prefix`; `ENGINE_STATE_CACHE_MAX_GB`, default 192) |
| `ENGINE_BINARY` | `.build/release/engine` | binary to run (its `mlx.metallib` must sit beside it) |

`tools/lan_proxy.py --listen <LAN_IP>:8099` relays the loopback server to private-range LAN peers without
restarting it, and without authentication.

## API

- `POST /v1/chat/completions`: OpenAI chat completions, streaming (SSE) and non-streaming. Supported fields:
  `messages` (text only: the text parts of `content` are concatenated, and image and other non-text content parts
  are silently dropped, with no error; the server has no image or video input), `max_tokens` /
  `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repetition_penalty` (alias `repeat_penalty`), `logit_bias`, `seed`, `n` (1-8), `stop`, `stream`,
  `stream_options.include_usage`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` (best effort:
  an instruction, not constrained decoding), `logprobs` / `top_logprobs`, `reasoning_effort`. Extensions:
  `thinking_budget` and `loop_guard` (hard guards, off by default; a request with either one active decodes
  without MTP), `think_bias_max` and `think_bias_deadline`
  (per-request overrides of the soft stop's peak bias and deadline), `preserve_thinking`. `enable_thinking` and
  `chat_template_kwargs` are not forwarded to the chat template; use `reasoning_effort: low` for brief thinking.
  The reasoning is returned separately as `reasoning_content`. `usage` carries prompt/generation rates, and
  `engine_stats` carries per-request timings, cache hits and MTP acceptance.
- `GET /v1/models`: the served model.
- `GET /v1/engine/sessions`: live state of the engine: requests in flight (phase, prefill progress, tokens,
  TTFT, rates), recent requests, totals, memory, RAM budget and reservations, hot cache, scheduler and batching
  counters, and the runtime configuration. Counts and timings only, never prompt text.

Request defaults (production launcher):

- **No `max_tokens`**: the reply may use the rest of the context (1,048,576 − prompt − speculative lookahead).
- **Oversized `max_tokens`**: clamped to the room left instead of refused. A prompt that leaves no room at all is
  refused with HTTP 400.
- **`reasoning_effort`**: default `xhigh`. `high`, `max` and `maximum` map to `xhigh`, and `minimal`, `none` and `off`
  map to `low`. `medium` and `low` are used as given, and anything else gets the default.
- **Sampling**: `temperature` defaults to 0 (greedy) when omitted. The checkpoint's own generation config
  recommends `temperature 1.0, top_p 0.95, top_k 20`, and Engine Studio sends those values.
  Order (Hugging Face / vLLM V0): `logit_bias` (-100..100), `repetition_penalty` (prompt and output tokens; 1 = off),
  `presence_penalty` / `frequency_penalty` (-2..2, output tokens, OpenAI's formula), then temperature, `top_k`,
  `top_p`, `min_p` (0..1, relative to the most probable token). The penalties and `logit_bias` also change a greedy
  request's argmax; `min_p` does nothing to a greedy request. Out-of-range values are refused with `400`. A request
  that uses a penalty or `logit_bias` (or `min_p` with sampling) decodes serially without MTP, like `logprobs`, so
  it is slower; default values (0, 1, empty) keep the fast batched path. Logprobs are those of the raw logits.
- Limits: HTTP headers 64 KiB, body 32 MiB. A request that finds all 16 slots busy waits in a first-in-first-out
  queue and starts as soon as a slot frees (`tools/serve.sh`: `--queue-max 32 --queue-timeout-s 1500`); a request
  whose KV reservation does not fit yet waits for memory the same way (`ENGINE_ADMISSION_WAIT`, on by default). A full
  queue or a wait past the limit answers `503` with `Retry-After: 1`. Run directly without `--queue-max` (the binary's
  default is 0; its default `--queue-timeout-s` is 1800), `engine serve` answers the 17th concurrent request with the
  `503` at once. The live counters are in `GET /v1/engine/sessions` under `queue` and `admission_wait`.

## Engine Studio and the dashboard

`apps/engine-studio` is a local chat app and control panel, written in pure Python stdlib with no build step:

```sh
apps/engine-studio/run.sh          # http://127.0.0.1:7860
```

It attaches to an engine already serving on port 8099 and never starts a second one. It cannot attach to an engine
that requires `ENGINE_API_KEY`: Studio sends no key, so the engine answers 401 and Studio reports it as alive but
not answering. It starts an engine itself only when none is running. An engine launched by Studio uses Studio's own
argument list, not the gated production profile. Compared with `tools/serve.sh` it has a 48 GB hot cache instead of
192 GB, the disk cache on and loop guard 3, and it lacks native batched prefill (`ENGINE_BATCH_PREFILL` /
`ENGINE_PREFILL_ROW_PROJECTIONS` unset), the shared RAM budget, the pooled private-history release and the
width-canonical prefill (`ENGINE_PREFILL_WIDTH_CANONICAL`). Phase scheduling and batching from 4 rows come from the
binary's defaults. MTP does not: `engine serve` defaults to `--mtp 0`. Studio passes `--mtp 3` (its `mtp_depth`
setting) and sets `ENGINE_MTP=1`; `--batch-mtp`, which Studio does not pass, then defaults to the same K = 3, and the
batched policy is `auto` unless `ENGINE_BATCH_MTP_POLICY` is set in Studio's environment. Because of loop guard 3,
however, a request decodes without MTP unless it sends `loop_guard: 0` itself, and Studio's own chat does not send
that field; with Studio's defaults, chat requests therefore run without MTP (DERIVED from the code, not measured).
The measured numbers below and the "loop guard off by default" statement do not apply to such an engine. Start the
engine with `tools/serve.sh` first.
`http://127.0.0.1:7860/engine` is a read-only dashboard for the running engine: memory, hot cache, RAM budget,
derived limits, active and recent requests, scheduler counters, the process's flags and environment, and the raw
sessions payload. See [apps/engine-studio/README.md](apps/engine-studio/README.md).

## Performance

Measured on 2026-10-03 on one Mac Studio with M3 Ultra (80-core GPU) and 512 GB, macOS 27.2,
`iogpu.disable_wired_collector=1`, behind `tools/serve.sh`, with the prose files of the `llm_context_benchmarks`
client (commit 5574d0b): a novel cut to the given length in cl100k tokens, a unique cache-busting prefix per request
(every prompt is cold) and the request "Please provide a summary of the above text.", at most 512 generated tokens.
The single-stream figures ran on the production build just before this export; this export adds the model-thread
autorelease pool and the admission factor, and its quality gates measured the same speed (real suite 90.89 against
90.90 tok/s; 16 concurrent streams 189.7 against 191.2 tok/s). Generated tokens include the reasoning tokens (default
`reasoning_effort` `xhigh`). Treat the figures as indicative, not as guarantees.

**One request at a time** (`openai_benchmark.py`): server-reported prefill and generation rates, time to first token
end to end. At the client's default `temperature` 1.0 two runs per point up to 256k (each figure the better of the
two), one run at 512k and 1000k; greedy (`temperature` 0) one run per point:

| Context (cl100k) | 0.5k | 8k | 32k | 128k | 256k | 512k | 1000k |
|---|---|---|---|---|---|---|---|
| Prompt tokens (E9) | 554 | 8,281 | 32,759 | 129,264 | 260,636 | 521,241 | 1,017,845 |
| Prefill, tok/s | 826 | 1,212 | 1,261 | 1,249 | 1,223 | 1,027 | 855 |
| Time to first token, s | 0.70 | 6.90 | 26.1 | 104.0 | 214.1 | 509.3 | 1,193.5 |
| Generation at temperature 1.0, tok/s | 83.6 | 61.3 | 62.8 | 59.8 | 54.1 | 63.7 | 59.4 |
| Generation greedy, tok/s | 85.4 | 79.0 | 76.6 | 71.8 | 68.3 | - | - |

Greedy generation is faster because a draft token is accepted only when it matches the token the target picks, and a
sampled pick matches the draft less often than the argmax does. A 1M-token prompt takes about 20 minutes to prefill
cold; a later turn of the same conversation resumes from the hot prefix cache as long as its state is still there.

**16 parallel sessions** (the project's concurrent client, the same prose files and settings, `temperature` 1.0, one
run per point, all 16 requests released together): aggregate prefill = all prompt tokens ÷ the slowest session's time
to first token; generation = the sum over sessions of their server-reported decode rates.

| Context per session (cl100k) | 0.5k | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| Prompt tokens per session (E9) | 552 | 2,172 | 8,280 | 32,758 | 65,265 | 129,262 |
| Aggregate prefill, tok/s | 631 | 844 | 1,194 | 1,223 | 1,205 | 1,138 |
| Aggregate generation, tok/s | 272 | 260 | 236 | 218 | 204 | 189 |
| Generation per session (median), tok/s | 16.8 | 15.8 | 14.6 | 13.7 | 12.8 | 11.9 |
| Time to first token (median), s | 13.8 | 41.2 | 109.5 | 427.3 | 865.3 | 1,815.5 |
| Peak MLX memory, GB | 288 | 305 | 333 | 390 | 367 | 342 |

All 16 requests were admitted together at every size, this export's build (at 128k: 176 GB of the 219.5 GB KV
budget reserved, free memory never below 137 GB).

The requests of a burst do not start decoding one by one: up to 4k tokens one of them decodes at once and the other
15 get their first token together once every prefill in the burst has finished; from 8k tokens on all 16 wait for the
whole burst (a scheduling property, noted below).

A later turn of the same conversation can avoid paying the prefill again for the part already seen, but only if the
previous turn's state is still in the hot prefix cache: the engine then restores it from GPU memory and prefills only
the tokens after the cached point. The cache is bounded by its 192 GB ceiling and by what live requests' RAM
reservations leave, and it evicts entries to make room; a turn whose entry was evicted pays the cold prefill again.
Hit rates under load are not part of the measurements above.

## Known limitations

- Continuous batching makes outputs schedule-dependent: batch composition depends on arrival order, and the batched
  and solo numeric paths differ. Identical requests can therefore produce different tokens even at `temperature` 0.
  For strict reproducibility, run a diagnostic `engine serve` without batching and without the hot cache, with a
  fixed prefill width (`--batch-min 0 --hot-cache-gb 0 --prefill-chunk 1024`; otherwise the adaptive prefill uses
  4096-token chunks while a request is alone and 1024 while others are active, so prefill numerics may depend on
  concurrency), and also without MTP (omit `--mtp`, or pass `--mtp 0`, the binary's default) if the output must be
  bit-identical to serial greedy decoding.
  Sending requests one at a time removes only the batching dependence; the hot cache and MTP still apply.
- A continuation resumed from the hot cache is not guaranteed to be token-identical to a cold prefill of the same
  prompt. This is a design trade-off: the launcher reuses the live decode state as the prefix, which avoids
  re-prefilling but is not bit-identical to a cold prefill.
- When many cold requests arrive together, the ones whose prefill finishes early wait for the whole burst's prefill
  before their first token (see the 16-session table); a single request or a trickle of requests is not affected.
- Only E9 (`qwen4_exp`) is supported by the serving path. Other models are untested.
- The HTTP server is text-only. Image and video input exist only in the `engine generate` development command
  (`--image`, `--video`); that path was last tested with E9 on ENGINE builds of 2026-08-31 and 2026-09-01, not on B50 or B52.
- The other `engine` subcommands (`bench`, `generate`, `batchprobe`, `kernelbench`, `tokbench`, `nll`,
  `logits`, `ngram-ids`) are development tools. Their default input paths (`bench/`, `runs/`) point to research
  files that are not part of this repository, so pass explicit `--prompt` / `--file` arguments. Some `kernelbench`
  cases read routing dumps (`runs/rt.i32.<tag>`) that no flag overrides and that are not shipped here, so they
  cannot run as-is; others default to tensor dumps under `runs/flashnext/`, which are not shipped either.

## Tests

```sh
tools/run_tests.sh --filter RequestDefaultsTests
# next to a running engine:
ENGINE_TESTS_ALLOW_WITH_ENGINE=1 tools/run_tests.sh --filter RequestDefaultsTests
```

`tools/run_tests.sh` copies `.build/release/mlx.metallib` into the test bundle and refuses a run that executed no
tests. It assumes the full suite may need about 270 GB of memory (an unmeasured estimate; the only measured memory
growth was the XCTest autorelease accumulation in `IncrementalDetokenizerTests`, which this repository fixes). While
an `engine` process is running and less than 300 GB are free, it refuses every run, whatever the filter, unless
`ENGINE_TESTS_ALLOW_WITH_ENGINE=1` is set. Do not run the whole suite next to a serving engine; set that variable
only together with a light `--filter`. `IncrementalDetokenizerTests` took about 6 minutes in a debug build when
this export was tested (2026-09-25, macOS 27.2). XCTest records a call stack for every error the standard library
throws internally while repairing invalid UTF-8, which is also why the test wraps each window in an
`autoreleasepool`. One of its tests uses the checkpoint's tokenizer from
`weights/e9` (or `ENGINE_TEST_MODEL`) and is skipped if the checkpoint is absent.

## Layout

```
Sources/EngineCLI           the `engine` executable: serve (HTTP server, scheduler), bench and diagnostics
Sources/EngineServeSupport  HTTP transport, admission budget, request defaults, model thread, sampler
Sources/Qwen4Exp            the qwen4_exp (E9) model port, its Metal kernels, the prefix stores
Vendor/                     modified forks of mlx-swift and mlx-swift-lm
Tests/PrefixStoreTests      unit tests
tools/                      serve.sh, build-mlx-metallib.sh, run_tests.sh, lan_proxy.py
apps/engine-studio          chat app, control panel and /engine dashboard
```

The custom Metal kernels are defined in `Sources/Qwen4Exp/Qwen4ExpKernels.swift` and, for the n-gram row gather,
in `Sources/Qwen4Exp/Qwen4Exp.swift`. Their bodies are spread over three places: most come from
`Sources/Qwen4Exp/Qwen4ExpLabKernelSources.swift`, the rest are written inline in `Qwen4ExpKernels.swift` (some
built by helper functions there) or in `Qwen4Exp.swift`. The header of `Qwen4ExpLabKernelSources.swift` says it is
generated from a kernel lab (`Kernels/lab`, `tools/gen_lab_kernels.py`) that is not part of this export, so edit
that Swift file directly.

Source comments also refer to other research-tree paths that are not shipped here, such as `lab/` (including the
reference implementations named in the header of `Sources/Qwen4Exp/Qwen4Exp.swift`, which says they are kept under
`lab/reference/`), `bench/` and `runs/`.

Code comments cite identifiers such as `P106`, `H50a`, `B44` or `OBS-ENG-167`. These refer to the project's internal
research log (plans, hypotheses, builds, observations), which is not included in this export.

## License

MIT, see [LICENSE](LICENSE). The MIT license covers the code in this repository that is not otherwise marked:
`Sources/`, `Tests/`, `tools/` and `apps/`. Parts of the repository started from Layr Labs' MIT-licensed mlxfast
challenge starter (`Layr-Labs/qwen-3.8-mtp-challenge`); its notice is kept in
[LICENSE.mlxfast-challenge](LICENSE.mlxfast-challenge). The vendored libraries under `Vendor/` keep their own
licenses (`Vendor/mlx-swift/LICENSE`, `Vendor/mlx-swift-lm/LICENSE` and the license files inside
`Vendor/mlx-swift/Source/Cmlx/`). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No model weights are distributed in this repository. The E9 checkpoint is published separately on Hugging Face under
the Qwen Community License 1.0 of its base model; this MIT license does not apply to it. Serving E9 to third parties
(for example through Engine Studio's tunnel or `tools/lan_proxy.py`) is subject to condition 2 of that license: if
you or your affiliates run a Model-as-a-Service or AI Work Assistant business (an AI product mainly for coding or
office productivity), you need a separate license from Qwen before any commercial use.

# ENGINE

ENGINE is an OpenAI-compatible inference server written in Swift on
[MLX](https://github.com/ml-explore/mlx-swift) for one model on Apple silicon: **E9**, an 8-bit MLX
quantisation of `Qwen/Qwen3.8-Flash-Next` (model_type `qwen4_exp`). It was built and tuned for a single
machine, a Mac Studio with M3 Ultra and 512 GB of unified memory, where it serves an agentic coding workload
(many long, concurrent, tool-calling sessions with shared prefixes).

What it does:

- **Continuous batching** of up to 8 concurrent sessions: requests join and leave a running batch; decode and
  prefill are scheduled in phases, and compatible prefills are batched natively.
- **MTP speculative decoding** in the production launcher's profile (`engine serve` itself defaults to `--mtp 0`,
  no MTP), using the model's own multi-token-prediction head: K = 3 for a lone request; batched rows draft up to
  K = 3, with the depth (or a plain step) chosen adaptively from measured acceptance and round cost. Some requests
  are not armed for drafting and decode without MTP: requests that ask for `logprobs`; requests with a hard
  thinking guard (`thinking_budget` or `loop_guard` above 0, per request or as a server flag), because the chat
  template opens the thinking block on every request; and requests resumed from a cache entry saved without MTP
  state. A forced tool-call prefix switches a request to serial decoding at that point.
- **Hot prefix cache in GPU memory**: a new turn can resume from the previous turn's state instead of re-prefilling
  the conversation, as long as that state is still in the cache. It shares one RAM budget with the live requests
  (ceiling 128 GB) and only gets what live reservations leave: with the default rest-of-context reservation, about
  22 GB at 8 concurrent requests (DERIVED from the admission formula, not measured under load). An optional disk
  cache can be enabled.
- **262,144-token context** (the checkpoint's `max_position_embeddings`), with admission control that reserves
  memory before a request starts and answers `503` + `Retry-After` instead of running out of memory.
- **Thinking soft stop**: instead of a hard thinking budget, a bias on `</think>` ramps up to +12 logits between
  2,000 and 8,000 thinking tokens so the model closes its reasoning at a sentence boundary. After 8,000 tokens the bias keeps
  climbing, quadratically, to +52 logits at a deadline of 14,000 thinking tokens. That exceeds every `</think>` gap
  measured (about 24 logits), so the block closes in practice (DERIVED, not a measured guarantee). A smaller
  `max_tokens` ends the reply first.
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
  cache held about 127 GB. These are observed peaks, not an upper bound. `tools/serve.sh` caps the MLX buffer cache at 32 GB.
- Xcode with the Swift 6.3 toolchain (`swift-tools-version: 6.3`), Xcode's Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) and `cmake` (`brew install cmake`) for the Metal library.
- Network access for the first build: SwiftPM fetches the Swift package dependencies (for example
  `swift-transformers` and `swift-jinja`) from GitHub.
- About 195 GB of disk for the weights.
- **Recommended: `sudo sysctl iogpu.disable_wired_collector=1`** (it resets on every reboot). Every measurement in
  this project since 2026-09-21 assumed it. In the first GPU run after a reboot into macOS 27.2 it was back at 0: a
  31,744-token prefill took 69.7 s instead of 29.5 s and many decode rounds stalled for about 2.3 s. With it set
  to 1 again, the same prefill took 29.7 s. (This is consistent with the collector as the cause but confounded
  with a reboot, an OS update and a build change; no within-window toggle was run.)
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
(batched rows up to K = 3, adaptively), up to 8 concurrent requests, batching from 4 ready rows up to 8 rows with a 25 ms gather window,
phase scheduling with native batched prefill, a shared RAM budget with a 128 GB hot-cache ceiling, the thinking
soft stop, `reasoning_effort` default `xhigh`, no default `max_tokens` cap, and oversized `max_tokens` clamped. It
refuses overrides of the batching, prefill, hot-cache, concurrency, MTP and state-cache flags and of the scheduler
environment. Other trailing arguments (for example reasoning effort, the soft-stop settings or the `max_tokens`
policy) are passed through and override the defaults, as do `ENGINE_CACHE_LIMIT_GB` and `ENGINE_WIRED_LIMIT_GB`.
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
  `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `n` (1-8), `stop`, `stream`,
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

- **No `max_tokens`**: the reply may use the rest of the context (262,144 − prompt − speculative lookahead).
- **Oversized `max_tokens`**: clamped to the room left instead of refused. A prompt that leaves no room at all is
  refused with HTTP 400.
- **`reasoning_effort`**: default `xhigh`. `high`, `max` and `maximum` map to `xhigh`, and `minimal`, `none` and `off`
  map to `low`. `medium` and `low` are used as given, and anything else gets the default.
- **Sampling**: `temperature` defaults to 0 (greedy) when omitted. The checkpoint's own generation config
  recommends `temperature 1.0, top_p 0.95, top_k 20`, and Engine Studio sends those values.
- Limits: HTTP headers 64 KiB, body 32 MiB. Over 8 concurrent requests, or without memory for the reservation,
  the answer is `503` with `Retry-After: 1`.

## Engine Studio and the dashboard

`apps/engine-studio` is a local chat app and control panel, written in pure Python stdlib with no build step:

```sh
apps/engine-studio/run.sh          # http://127.0.0.1:7860
```

It attaches to an engine already serving on port 8099 and never starts a second one. It cannot attach to an engine
that requires `ENGINE_API_KEY`: Studio sends no key, so the engine answers 401 and Studio reports it as alive but
not answering. It starts an engine itself only when none is running. An engine launched by Studio uses Studio's own
argument list, not the gated production profile. Compared with `tools/serve.sh` it has a 48 GB hot cache instead of
128 GB, the disk cache on and loop guard 3, and it lacks native batched prefill (`ENGINE_BATCH_PREFILL` /
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

All numbers below were **measured on build B50** (the production build before this export) on one Mac Studio with
M3 Ultra (80-core GPU) and 512 GB, macOS 27.2, `iogpu.disable_wired_collector=1`. This export is build B52, which
changed only the request defaults above; B50 served with a default `reasoning_effort` of `medium` and a default
`max_tokens` of 32,000. None of the clients below set `reasoning_effort`, so every figure ran at `medium`, and
every client set `max_tokens` explicitly. The figures were not re-measured on B52. Generated tokens include the
reasoning tokens. Treat the figures as indicative, not as guarantees.

**One request at a time**: B50 behind the production launcher `tools/serve.sh` (MTP on), measured with the
`llm_context_benchmarks` client (commit 5574d0b). Each request is a cold prompt of the given length (the client
busts the cache per prompt), 256 generated tokens, `temperature` 1.0 (the client's default). Two runs per point;
each figure is the better of the two runs for that metric, so the three figures in one column can come from
different runs. Generation differed by up to 11 % between the two runs; at 0.5k, prefill and time to first token
differed by about 25 %. Client-side timing: prefill = prompt
tokens ÷ the client's prompt time, generation = (generated tokens − 1) ÷ the time after the first token, time to
first token end to end:

| Context | 0.5k | 8k | 32k | 64k | 256k |
|---|---|---|---|---|---|
| Generation, tok/s | 102.7 | 91.3 | 88.6 | 80.4 | 77.0 |
| Prefill, tok/s | 765 | 1,185 | 1,255 | 1,251 | 1,196 |
| Time to first token, s | 0.67 | 6.95 | 26.1 | 52.1 | 218.0 |

The 0.5k generation figure rests on a delayed first token.

**N concurrent sessions**: B50 behind `tools/serve.sh`, measured with the project's own concurrent client. N
requests are released together, each a cold prompt of the given length (a unique prefix per request defeats the
cache) followed by a summarisation question, 256 generated tokens each, greedy (`temperature` 0), one run per
point. Summed decode = the sum over sessions of (generated tokens ÷ time after the first token), client-side:

| Context per session | 2k | 8k | 32k | 64k | 256k |
|---|---|---|---|---|---|
| N = 8, tok/s | 226.0 | 221.7 | 206.4 | - | - |
| N = 4, tok/s | 147.8 | 152.4 | 146.4 | 151.5 | - |
| N = 1, tok/s | 89.4 | 90.5 | 92.5 | 79.5 | 79.7 |

In the same runs, aggregate prefill of the concurrent cold prompts (all prompt tokens ÷ the slowest session's time
to first token) was 1,143 tok/s (N = 8, 8k), 1,159 tok/s (N = 8, 32k) and 864 tok/s (N = 4, 2k).

**Full context, cold**: the server-reported time to first token for a 258,089-token prompt (a novel plus one
question, greedy, 64 output tokens) was 210.80 s and 210.49 s in two runs on B50. These come from the project's
release-gate harness, which starts a fresh `engine serve` for each run with its own copy of the serving flags and
sends that one request; it is not `tools/serve.sh` under a production workload.

A later turn of the same conversation can avoid paying this again for the part already seen, but only if the
previous turn's state is still in the hot prefix cache: the engine then restores it from GPU memory and prefills
only the tokens after the cached point. The cache is bounded by its 128 GB ceiling and by what live requests' RAM
reservations leave (about 22 GB at 8 concurrent rest-of-context requests, DERIVED), and it evicts entries to make
room; a turn whose entry was evicted pays the cold prefill again. Hit rates under load are not part of the
measurements above.

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

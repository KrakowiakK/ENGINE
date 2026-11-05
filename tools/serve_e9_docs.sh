#!/usr/bin/env bash
# E9 (Qwen3.8-Flash-Next) for bulk document pages (T-0098, 2026-10-20): the production E9 launcher (tools/serve.sh) plus
# two prefill settings measured on 16 concurrent A4 page images (pdf_load, same pages, binary a094a0c, image rows in
# batched decode):
#   ENGINE_PREFILL_CHUNK_SHARED=4096  prefill chunks while other requests run (production 1024; the width-canonical
#                                     projections keep the state bit-identical): 11.47 -> ~12.0 pages/min
#   MLX_GATHER_QMM_NARROW_BELOW=64    the routed-expert gather's 16-row tile below 64 rows per expert: 2k-token text
#                                     prefill 1.79 -> 1.71 s; 8/8 text answers identical, page accuracy unchanged
# Together 11.47 -> 11.94 pages/min on the same pages (serial image decode before a094a0c: 7.6).
# The OMP coding profile (tools/serve.sh alone) is unchanged: longer shared chunks lengthen chat decode stalls.
set -euo pipefail
cd "$(dirname "$0")/.."
export ENGINE_PREFILL_CHUNK_SHARED="${ENGINE_PREFILL_CHUNK_SHARED:-4096}"
export MLX_GATHER_QMM_NARROW_BELOW="${MLX_GATHER_QMM_NARROW_BELOW:-64}"
# T-0109 (2026-10-29): 32 decode seats (tools/serve.sh ENGINE_SEATS). Closed loop on this profile: decode-heavy load 32
# streams 157.5 vs 16 streams 151.4 tok/s (mean batch 17.3 vs 11.7), client-shaped 3.1k-token prompts 112.2 vs 111.9; a
# round grows almost in proportion to its rows (experts read at ~90% of the bandwidth roof), so the gain is small, but
# requests past 16 decode instead of queueing (the client keeps up to ~32 in flight at night).
export ENGINE_SEATS="${ENGINE_SEATS:-32}"
# T-0111 (2026-10-29): no 1-token hot reserve chunk (ENGINE_HOT_RESERVE=0). It serves multi-turn clients whose next prompt
# ends one token short; this client sends independent prompts (0.2% exact repeats), and the extra forward cost 4.7% of its
# prefill work. Serve A/B (8 streams, ~800-token prompts, 40 tokens out): 143/147 vs 137/143 requests per 180 s; Polish
# transcription edits on 64 pages (seeds 20-27): equal except the known knife-edge page of seed 23.
export ENGINE_HOT_RESERVE="${ENGINE_HOT_RESERVE:-0}"
# T-0111: a prompt finishes in ONE chunk across the 2048 QSA budget cut when <= 4096 tokens remain (the client's pages and
# rung-resumed prompts were [1024,2048) + a small tail: two forwards with few rows per expert). Lossless (the rows below the
# budget see all their blocks), not bit-identical. Measured: pages 12.2 -> 12.5 /min, Polish edits on 64 pages 233 (vs 274 /
# 312 for the cut schedule's two runs), text prefill +2%.
export ENGINE_PREFILL_MERGE_TAIL="${ENGINE_PREFILL_MERGE_TAIL:-4096}"
# T-0119 (2026-10-30): batched-verify 8-bit projections without MLX qmm's 32-row tile padding (Qwen4ExpVerifyProj.swift).
# Teacher-forced verify step -0.6..-4.8% at 40-56 and 68-80 rows, unchanged at multiples of 32 (C-0357); wide projections
# bit-identical to MLX qmm, narrow ones closer to the exact product; serve A/B within noise, PDF accuracy identical (C-0358).
export ENGINE_VERIFY_PROJ="${ENGINE_VERIFY_PROJ:-1}"
# T-0121 (2026-10-30): the fused verify-width hyper-connection kernels (Qwen4ExpHCVerify.swift) at <= 48 rows on top of
# the projections: teacher-forced step -0.8..-2.6% at 32-48 rows, +0.2..0.8% at 52-64 (hence the 48-row limit) (C-0370);
# closer to the exact float32 result than the qmm path (C-0340); PDF transcription accuracy unchanged over 3 seeds x 2 (C-0371).
export ENGINE_HC_VERIFY_FUSED="${ENGINE_HC_VERIFY_FUSED:-1}"
# T-0127 (2026-10-30): the tiled (simdgroup-MMA) QSA verify gather (Qwen4ExpQSATiled.swift): x1.6-1.8 on the gather,
# verify rounds -2.3..-2.8% at 12-32 rows (C-0381, C-0382); ~1 ulp from the shipped kernel; PDF accuracy unchanged
# over 3 seeds x 2 (C-0383).
export ENGINE_QSA_VERIFY_TILED="${ENGINE_QSA_VERIFY_TILED:-1}"
# T-0141..T-0145 (2026-10-30, operator: each request's latency matters, Q-0006): ENGINE_PREFILL_ORDER=arrival serves the
# prefill of the oldest request first (control jobs pass queued prefill, C-0419) instead of FIFO round-robin, which gave all
# 7 prompts of a live 9k-token burst the same 73 s TTFT (C-0402, C-0407); ENGINE_MIXED_ROUND=1 carries the batched verify
# rows inside the prefill chunks (one forward, chunk rows bit-identical, verify rows at rounding level with equal NLL:
# C-0420, C-0421, C-0423; full step quorum only, C-0425) so early decoders do not run small expert-rereading rounds. W3 vs
# fifo, same window (C-0428): 1 lead + 7 x 8.8k bursts at max_tokens 5200 TTFT 78.9 -> 43.5 s, completion -4.9%, tok/s
# +3.3%; short-output bursts TTFT -22%, completion -20%; PDF page TTFT -24%, accuracy unchanged; a whole wave ends ~5%
# later (tail, C-0427). Rollback: ENGINE_PREFILL_ORDER=fifo ENGINE_MIXED_ROUND=0.
export ENGINE_PREFILL_ORDER="${ENGINE_PREFILL_ORDER:-arrival}"
export ENGINE_MIXED_ROUND="${ENGINE_MIXED_ROUND:-1}"
# T-0151 (2026-10-31): the GDN verify roll-back IN PLACE over the forward's state buffer (ENGINE MLX fork: "__donate<k>"
# custom kernels), so a fully accepted row costs no state traffic. The roll-back is state traffic (C-0466); committed
# streams identical (C-0471); serve ABBA: decode tok/s of work +1.2% (correct) / +3.0% (extract), PDF accuracy unchanged,
# buffer donated every time (C-0473). vLLM-style K+1 state slots measured slower (C-0470). ENGINE_DONATE_STATS=1 logs
# the donations every 4096 and EVERY copy (a copy = another holder of the state buffer = the gain is gone).
# Rollback: ENGINE_GDN_ROLLBACK_INPLACE=0.
export ENGINE_GDN_ROLLBACK_INPLACE="${ENGINE_GDN_ROLLBACK_INPLACE:-1}"
export ENGINE_DONATE_STATS="${ENGINE_DONATE_STATS:-1}"
# T-0155 (2026-10-31): 8192-wide prefill chunks for a prompt that prefills ALONE, with the sparse QSA gather kept for 8192
# rows (ENGINE_QSA_GATHER_MAX_S, default 4096: above it attention fell back to dense SDPA, C-0468). Forward +2.1..2.6%
# (C-0472); serve ABBA: lone TTFT per token -1.7..-2.7% at 8.8k/19k/34k, greedy answers identical, bursts and PDF within
# noise (C-0480, C-0481). Shared chunks stay 4096 (8192 shared starved decode in bursts, C-0465).
# Rollback: ENGINE_PREFILL_CHUNK_ALONE=4096 ENGINE_QSA_GATHER_MAX_S=4096.
export ENGINE_PREFILL_CHUNK_ALONE="${ENGINE_PREFILL_CHUNK_ALONE:-8192}"
export ENGINE_QSA_GATHER_MAX_S="${ENGINE_QSA_GATHER_MAX_S:-8192}"
# T-0156 (2026-10-31): a 191-token shared-prefix rung for the client's ~192-255-token common head (C-0449; the 1024 rung
# never hit), with the fused prefill conv writing its own next conv state (ENGINE_GDN_CONV_PREFILL_TAIL=1). Without that
# fix a rung-resumed chunk's conv state held garbage now and then (3-6 per 600 answers under concurrency, C-0488..C-0493);
# with it 0 of 1200 (C-0494). Serve ABBA with the fix (W24, C-0496): lone TTFT about -10%, extract client tok/s +4-5%, answer
# scores and PDF accuracy unchanged. The rung is NOT bit-identical to a cold prefill (other kernels for the 191-row head).
# Rollback: ENGINE_SHARED_PREFIX_RUNG=1024 ENGINE_SHARED_PREFIX_RUNG_SHORT=0 (keep the tail fix: it is bit-identical cold).
export ENGINE_GDN_CONV_PREFILL_TAIL="${ENGINE_GDN_CONV_PREFILL_TAIL:-1}"
export ENGINE_SHARED_PREFIX_RUNG="${ENGINE_SHARED_PREFIX_RUNG:-191}"
export ENGINE_SHARED_PREFIX_RUNG_SHORT="${ENGINE_SHARED_PREFIX_RUNG_SHORT:-1}"
# T-0176 (2026-11-02): copy-mode lookups only after a match of >= 10 tokens (the 6-gram index's match extended backwards;
# needs a binary from 76e6ec9 on -- an older one never sees a match above 6 and so never copies). Chat: 40 prompts pooled
# +2.45% (95% interval +0.25..+4.48%), 27/40 faster (C-0569); copy-heavy edits keep the whole copy gain, +0.94% over 10
# identical texts (C-0568). A 6-token match on chat text is often coincidental (16 of 69 suite lookups accepted nothing).
export ENGINE_DRAFT_COPY_MIN="${ENGINE_DRAFT_COPY_MIN:-10}"
# T-0178 (2026-11-02, D-0057): an image request takes its arrival stamp before its vision tower, so a wave's towers no
# longer all run before the first page's prefill. 16 pages at once, client-timed: first page ~20.7 -> ~4.2 s, mean TTFT
# -14.8%, mean completion -2%, last page +3 s, pages/min -1.9%, accuracy unchanged (C-0579). Needs bin-5987448 or later.
export ENGINE_VISION_ARRIVAL_STAMP="${ENGINE_VISION_ARRIVAL_STAMP:-1}"
# T-0185 (2026-11-02, D-0060): a chunk mixes with the decoders' steps even after an arrival-order continuation wait (30 of
# 32 burst chunks mixed instead of 17-27), and a mixed ride no longer uses up the decoders' separate share. Text32 burst vs
# the previous production: completion mean -1.9%, TTFT mean +3-4%; PDF 16 pages: pages/min +3.8%, TTFT mean -7%
# (C-0590). Needs bin-cd43a31 or later. Deeper drafts in mixed forwards (ENGINE_MIXED_MTP_K) do not pay (C-0592).
export ENGINE_MIXED_AFTER_CONTINUE="${ENGINE_MIXED_AFTER_CONTINUE:-1}"
export ENGINE_MIXED_DECODE_SHARE_KEEP="${ENGINE_MIXED_DECODE_SHARE_KEEP:-1}"
# T-0186 (2026-11-02, D-0062): packed prefill -- the owner packs other requests' queued prompt chunks with the picked one
# (up to 8192 rows, i.e. 3-4 of the client's ~2k-token prompts) into one forward, bit-identical per prompt (canonical
# blocking per segment), and the decoders' steps ride it. Text32 burst vs K1, 6 passes in two windows: span -3.4..-4.4%,
# TTFT max -12..-18%, output +3.6..+4.6%, mean completion neutral, first answer ~7 s instead of ~2 s; PDF16 +1.2..+1.7%
# pages/min, accuracy unchanged (C-0597, C-0598). Needs bin-5d89493 or later.
export ENGINE_PREFILL_PACK_ROWS="${ENGINE_PREFILL_PACK_ROWS:-8192}"
# T-0189 (2026-11-02, D-0064): the vision tower in its own dtype -- the preprocessor's float32 pixels had promoted the whole
# bf16 tower to fp32. Tower -21.7% per page; PDF16 +5.6% pages/min, TTFT -8.4%, last page -5.5%, accuracy and worst pages
# unchanged (C-0604). Needs bin-277c57d or later.
export ENGINE_VISION_BF16="${ENGINE_VISION_BF16:-1}"
# T-0190 (2026-11-02, D-0066): decode share 0.25 of the last prefill dispatch's work (serve.sh default 0.5). Until bdb6c53 a
# per-request setup closure counted as the last prefill dispatch and shrank the share to ~0 during waves -- an accidental
# decode throttle; with honest accounting 0.5 cost PDF waves 1.4 %, 0.25 gives PDF +0.9 % and text TTFT -6 % (C-0609).
# T-0209 (2026-11-04): 0.1 instead of 0.25 -- waiting requests start sooner and the decode rounds run fuller: 20-stream
# closed loop +1.56 % server tok/s (176.3 -> 179.05, every 0.1 run above every 0.25 run), TTFT mean -7 % (C-0678).
export ENGINE_DECODE_SHARE="${ENGINE_DECODE_SHARE:-0.1}"
# T-0191 (2026-11-02, D-0068): the vision rotary's cos/sin tables once per image and the rotation as one compiled graph --
# bitwise identical features, tower -3.5 %, PDF16 +1.4 % pages/min (C-0612). Needs bin-0551148 or later.
export ENGINE_VISION_ROTARY="${ENGINE_VISION_ROTARY:-compiled}"
# T-0191 (2026-11-03, D-0070): the vision tower's attention at its native head_dim 72 (a bd72 steel attention instantiation)
# instead of padding to 80 -- bitwise identical features, tower -4.5 %, PDF16 +1.0 % pages/min (C-0617). Needs bin-265624e
# (its metallib carries the bd72 kernel). The Swift tower and the MLX host read the same variable.
export MLX_SDPA_FULL_D72="${MLX_SDPA_FULL_D72:-1}"
# T-0192 (2026-11-03, D-0072): with packing, up to 2 younger image requests' towers run before an older page chunk so page
# chunks pack (production packed none in PDF waves): PDF16 +4.4 % pages/min, per-page completion -2.7 %, TTFT max -8 %,
# accuracy unchanged; each wave's first page ~7.5 s instead of ~3.5 s (C-0619, W94 C-0621). Needs bin-820d6aa or later.
export ENGINE_PACK_TOWER_LOOKAHEAD="${ENGINE_PACK_TOWER_LOOKAHEAD:-2}"
# T-0194 (2026-11-03): the GDN prefill recurrence with 4 value rows per simdgroup -- bit-identical y and state, prefill
# +4.2 % at 8192-wide forwards and +2.9 % at 2048 (C-0630). Needs bin-c323672 or later.
export ENGINE_GDN_ROWS="${ENGINE_GDN_ROWS:-1}"
# T-0198 (2026-11-03): the five prefill kernel families whose template carried the chunk's row count take it as an input,
# so a new prompt length compiles nothing on the model thread: JIT after a fresh text32 burst 3.4 s -> 0.33 s, outputs
# byte-identical, prefill speed unchanged (C-0640, C-0641). Needs bin-4b1d4f0 or later.
export ENGINE_RUNTIME_ROWS="${ENGINE_RUNTIME_ROWS:-1}"
# T-0200 (2026-11-03): control (exclusive) jobs -- ~7 per request plus every sessions poll -- pass the queued batched decode
# steps instead of ending the gathering, so a round is no longer split into two: 20-stream closed loop +4.5 % server tok/s
# (162.0 -> 169.35), mean step B 13.6 -> 16.7 (C-0652). Needs bin-29e4a4a or later.
export ENGINE_CONTROL_PASSES_DECODE="${ENGINE_CONTROL_PASSES_DECODE:-1}"
# T-0202 (2026-11-03): the tiled QSA verify gather reads K (selected blocks per row) at run time instead of compiling a
# variant per context length (654 compiles, ~3.3 s of model-thread JIT per 170 s on a fresh server): 20-stream closed
# loop +2.3 % server tok/s (168.3 -> 172.25), verify step unchanged in ABBA batchprobe (C-0656; C-0536 refuted).
export ENGINE_QSA_TILED_RUNTIME_K="${ENGINE_QSA_TILED_RUNTIME_K:-1}"
# T-0204 (2026-11-03): a buffer request of >= 1 MB may reuse a cached MLX buffer up to 1.25x its size, so sizes that follow
# context lengths stop missing the cache: 20-stream closed loop cache misses -93 %, newBuffer time -88 %, decode tok/s of
# work +1.5 % (every T run above every A run), server tok/s +1.1 % (within noise) (C-0661, C-0662; D-0078).
export ENGINE_CACHE_REUSE_TOL="${ENGINE_CACHE_REUSE_TOL:-0.25}"
# T-0207 (2026-11-04): decode closures (a new row's serial round, a rung capture, a rearm) pass the queued batched steps
# instead of ending the gathering, as control jobs do since T-0200: 20-stream closed loop +1.9 % server tok/s
# (173.8 -> 177.0, both ABBA pairs), decode tok/s of work +2.2 %, mean step B 15.7 -> 16.6 (C-0673). Needs bin-bb817a7.
export ENGINE_DECODE_CLOSURE_PASSES="${ENGINE_DECODE_CLOSURE_PASSES:-1}"
# T-0208 (2026-11-04): batched rounds draft a fixed K=2 instead of the auto policy, which on the client's extract load
# alternated K=2/K=3 (it measures acceptance at the K it runs) and sampled plain steps: 20 streams +1.65 % server tok/s
# (175.3 -> 178.2, both halves), decode work +2.85 %; fixed K=3 loses (-1.2 %); 8 streams tie, 4 streams +0.7 % (C-0675,
# C-0677). Single-stream serial rounds keep K=3 (--mtp 3).
export ENGINE_BATCH_MTP_POLICY="${ENGINE_BATCH_MTP_POLICY:-cycle:2}"
# T-0217 (2026-11-04): content-free copy census -- after each text response, a background count of how much of its output a
# copy-mode lookup would have drafted (/v1/engine/sessions copy_census; counts only). Decides whether batched rounds get copy
# drafts (C-0684: ~0 % on extract-style output, +13..23 % on output that re-types its prompt). Needs the T-0217 binary.
export ENGINE_COPY_CENSUS="${ENGINE_COPY_CENSUS:-1}"
# T-0221/T-0223: the census keeps totals per client label (x-engine-client ?? user-agent) and persists them (counts and
# labels only) to this file, restored at start, so a restart no longer loses a day of counts (C-0692, C-0695). Needs a
# binary from 816f3b2 on; an older one ignores the variable.
export ENGINE_COPY_CENSUS_FILE="${ENGINE_COPY_CENSUS_FILE:-$HOME/.cache/engine/copy_census.json}"
mkdir -p "$(dirname "$ENGINE_COPY_CENSUS_FILE")"
# T-0259 (C-0742): graceful stop -- SIGTERM refuses new requests (503 + Retry-After), lets admitted ones finish (up to
# ENGINE_DRAIN_SECONDS, 120) and exits 143, so windows and restarts no longer cut a client request. Needs bin-ccfe5c7+.
export ENGINE_GRACEFUL_STOP="${ENGINE_GRACEFUL_STOP:-1}"
# T-0247 (D-0090, C-0750): batched copy drafts through the packed per-row-depth verify -- per row a lookup over prompt +
# output (match >= 6, drafts up to 7), taken only when the round's cost gate says it pays; greedy rows, never in a mixed
# step. Client-like correct load +2.75%, extract +0.2% (P304); output as faithful as copy off (teacher-forced scan).
# Needs bin-62d15e5+ (an older binary ignores both variables).
export ENGINE_BATCH_COPY="${ENGINE_BATCH_COPY:-1}"
export ENGINE_BATCH_COPY_PACKED="${ENGINE_BATCH_COPY_PACKED:-1}"
# T-0294 (Q-0001, operator 2026-11-05 C-0794): programming requests that name no reasoning_effort default to low (tools,
# fenced code or >= 3 code lines; explicit efforts win). Needs bin-4eb54e1+ (an older binary ignores the variable).
export ENGINE_REASONING_EFFORT_CODING="${ENGINE_REASONING_EFFORT_CODING:-low}"
exec tools/serve.sh "$@"

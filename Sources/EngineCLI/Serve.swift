// engine serve -- an OpenAI-chat-completions-compatible HTTP server.
//
//   engine serve --model DIR [--port 8080] [--tokens 2048]
//                 [--state-cache DIR [--state-cache-step 512] [--state-cache-max-gb 64]]
//
// Concurrent HTTP connections submit model work to one owner thread. Decode steps
// can be continuously batched; prefills yield at chunk boundaries.
//
// Supports: streaming (true per-token SSE, not a single buffered frame),
// reasoning_content (this checkpoint's chat template opens a <think> block in
// its own generation prompt; ENGINE splits it from the visible answer rather
// than concatenating both into `content`), tool calling (this checkpoint's
// template uses an XML <tool_call><function=NAME><parameter=P>V</parameter>
// ...  form, NOT the common JSON <tool_call>{...}</tool_call> form -- ENGINE
// parses THIS checkpoint's actual format, not a generic guess), tool_choice
// "required"/named-function forcing via forced-prefix injection at the point
// the model would otherwise start its visible answer, logprobs/top_logprobs,
// n>1 choices, seed, and client stop sequences.
import Foundation
import CryptoKit
import Darwin
import MLX
import MLXLMCommon
import EngineServeSupport
import MLXLLM
import MLXNN
import MLXHuggingFace
import Tokenizers
import Jinja
import Qwen4Exp

private let h8BoundaryProbe = ProcessInfo.processInfo.environment["ENGINE_H8_BOUNDARY_PROBE"] == "1"
private let h8BoundarySplit = ProcessInfo.processInfo.environment["ENGINE_H8_BOUNDARY_SPLIT"] == "1"
private let h25CanonicalPrefix = ProcessInfo.processInfo.environment["ENGINE_H25_CANONICAL_PREFIX"] == "1"
private let h25CanonicalWidth = 1024
/// P106 H50: a B1 chunk wider than 1024 computes the split-K-sensitive projections in 1024-row blocks
/// (Qwen4ExpPrefillWidth), so its state at a 1024 boundary is the all-1024 trajectory's state (H50b probe);
/// only then may a wide chunk advance the canonical trajectory. Read once: the knob is process-wide.
private let h50WidthCanonical = Qwen4ExpPrefillWidth.canonicalRows == h25CanonicalWidth
private let h50MaxCanonicalChunk = 2048   // H50b: the widths measured state-identical (1024, 2048); 3072/4096 are not
private let h26PrefillMetadata = ProcessInfo.processInfo.environment["ENGINE_H26_PREFILL_METADATA"] == "1"
private let h25Metadata = h8BoundaryProbe || h26PrefillMetadata
private let h26PrefillIntervalCapacity = 64
nonisolated(unsafe) private var h26NativeGroupSerial = 0       // model owner only; actual native callbacks

/// Owner-published plain memory counters; polling never inserts a model-queue barrier.
private final class ServeMemoryWitness: @unchecked Sendable {
    private let lock = NSLock()
    private var value: [String: Double] = [:]
    private var prefill: [String: Int] = [:]
    private var indexer: [String: Int] = [:]
    private var restack: [String: Double] = [:]
    private var cacheLayout: [String: Any] = [:]
    private var ramBudget: [String: Double] = [:]
    private var hotCache: [String: Int] = [:]
    private var pooledPrivateHistory: [String: Any] = [:]
    func publish() {
        precondition(modelThreadShared.isOwner)
        let next = ["at": Date().timeIntervalSince1970, "active_bytes": Double(Memory.activeMemory),
                    "cache_bytes": Double(Memory.cacheMemory), "peak_bytes": Double(Memory.peakMemory)]
        let batches = Dictionary(uniqueKeysWithValues: nativePrefillSizes.map { ("B\($0.key)", $0.value) })
        let routes = Qwen4ExpIndexerRouteWitness.graphConstructionSnapshot()
        var restacks = ["host_graph_ms": batchRestackMs, "stack_completion_inclusive_ms": batchRestackCompletionMs,
                        "completion_barrier_enabled": batchRestackEval ? 1.0 : 0.0,
                        "logical_output_bytes": Double(batchRestackLogicalBytes), "membership_changes": Double(batchRestacks),
                        "row_resident_pool_installs": Double(rowResidentPoolInstalls),
                        "row_resident_head_clones": Double(rowResidentHeadClones),
                        "row_resident_alias_repairs": Double(rowResidentAliasRepairs),
                        "row_resident_markers_registered": Double(Qwen4ExpBatch.rowResidentRegistered)]
        for (transition, count) in batchRestackTransitions { restacks[transition] = Double(count) }
        var layout: [String: Any]? = nil
        // Only the owner inspects MLX metadata. The warm-up is too short to create
        // pooled indexer keys, so capture their first materialized decode pool too.
        // P106 H48: a row-resident pool's attention layers are markers; describe the first member's own caches
        if cacheLayout.isEmpty, let cache = batchPoolShared.cache,
           let first = cache.first(where: { $0 is CacheList }) as? CacheList,
           case let list = Qwen4ExpModel.rowResidentRows(first)?.first ?? first,
           let kv = list[0] as? KVCacheSimple, let idx = list[1] as? ArraysCache, idx[1] != nil {
            let describe: (MLXArray) -> [String: Any] = { ["shape": $0.shape, "dtype": String(describing: $0.dtype), "nbytes": $0.nbytes] }
            layout = ["offset": kv.offset, "kv": kv.state.map(describe), "kv_capacity": [kv.rawKeys, kv.rawValues].compactMap { $0 }.map(describe), "indexer": idx.state.map(describe),
                      "row_resident": batchPoolShared.rowResident,
                      "at": Date().timeIntervalSince1970, "scope": batchPoolShared.rowResident
                        ? "first materialized row-resident decode pool: member 0's own trunk caches; tensor metadata"
                        : "first materialized trunk decode pool; tensor metadata"]
        }
        let budget: [String: Double]
        var hot: [String: Int]
        if serveSharedRAMBudget {
            budget = (serveAdmission?.snapshot() ?? [:]).merging(serveRAMReclaimWitness) { _, new in new }
            hot = hotStoreShared?.snapshot() ?? [:]
        } else { budget = [:]; hot = [:] }
        hot.merge(seqRegistry.hotRungSnapshot()) { _, new in new }
        // Read owner-only counters here; HTTP receives only the locked plain snapshot.
        let privateHistory: [String: Any] = ["observed": true,
            "rows_installed": pooledPrivateHistoryRowsInstalled,
            "pool_installs": pooledPrivateHistoryPoolInstalls]
        lock.lock(); value = next; prefill = batches; indexer = routes; restack = restacks
        ramBudget = budget; hotCache = hot; pooledPrivateHistory = privateHistory
        if let layout { cacheLayout = layout }
        lock.unlock()
    }
    func snapshot() -> [String: Double] { lock.lock(); defer { lock.unlock() }; return value }
    func prefillSnapshot() -> [String: Int] { lock.lock(); defer { lock.unlock() }; return prefill }
    func indexerSnapshot() -> [String: Int] { lock.lock(); defer { lock.unlock() }; return indexer }
    func restackSnapshot() -> [String: Double] { lock.lock(); defer { lock.unlock() }; return restack }
    func cacheLayoutSnapshot() -> [String: Any] { lock.lock(); defer { lock.unlock() }; return cacheLayout }
    func ramBudgetSnapshot() -> [String: Double] { lock.lock(); defer { lock.unlock() }; return ramBudget }
    func hotCacheSnapshot() -> [String: Int] { lock.lock(); defer { lock.unlock() }; return hotCache }
    func pooledPrivateHistorySnapshot() -> [String: Any] { lock.lock(); defer { lock.unlock() }; return pooledPrivateHistory }
}
private let serveMemoryWitness = ServeMemoryWitness()
nonisolated(unsafe) private var nativePrefillSizes: [Int: Int] = [:] // owner only
/// Research opt-in: private trunk placeholders while a distinct standing pool owns history.
nonisolated(unsafe) private var releasePooledPrivateHistory = ProcessInfo.processInfo.environment["ENGINE_RELEASE_POOLED_PRIVATE_HISTORY"] == "1"
// Owner only. Counts installations, including re-installations for the same request; not bytes freed.
nonisolated(unsafe) private var pooledPrivateHistoryRowsInstalled = 0, pooledPrivateHistoryPoolInstalls = 0
/// P106 H48 -- ROW-RESIDENT KV (ENGINE_ROW_RESIDENT_KV, default 1; 0 restores the stacked pool). The
/// batch pool stops copying members' growing history: every member keeps its attention K/V and
/// indexer buffers and the ragged step writes them in place, so a membership change -- a join, a
/// leave, a member that misses one dispatch -- re-stacks only the fixed-size GDN/PLE/n-gram state and
/// the draft heads' Slast (~115 MB per row), where the stacked pool re-copied ~8 GB per row per layer
/// set at 262k (H45: 0.9-1.6 s per change). Owner-thread state; a diagnostic may flip it between arms
/// while no pool stands. Ineligible environments (masked/ablation selection, ragged SDPA, replay
/// dump) keep the stacked pool (`Qwen4ExpModel.rowResidentEligible`).
nonisolated(unsafe) var rowResidentKV = (ProcessInfo.processInfo.environment["ENGINE_ROW_RESIDENT_KV"] ?? "1") != "0"
nonisolated(unsafe) var rowResidentPoolInstalls = 0, rowResidentHeadClones = 0, rowResidentAliasRepairs = 0
nonisolated(unsafe) private var hotRungCaptures = 0, hotRungPruned = 0, hotRungInvalid = 0
nonisolated(unsafe) private var hotRungPooledSkips = 0, hotRungHighWaterPerSequence = 0
/// P106 B41: decode rungs of batched row-resident members, captured from the pool in place (ENGINE_POOLED_HOT_RUNGS=0 skips, as B40).
private let pooledHotRungs = ProcessInfo.processInfo.environment["ENGINE_POOLED_HOT_RUNGS"] != "0"
/// P106 B41: prompt-prefix rung spacing for a prefill from position 0 (0 = none, as B40).
private let prefixRungStep = Int(ProcessInfo.processInfo.environment["ENGINE_PREFIX_RUNG_STEP"] ?? "8192") ?? 8192
nonisolated(unsafe) private var hotRungPooledCaptures = 0

func serve(_ o: Options) async throws {
    // A client that disconnects mid-request (a timeout, a cancelled stream) makes the eventual
    // write() to its socket fail with EPIPE. The default SIGPIPE disposition is to terminate the
    // WHOLE PROCESS for that -- one impatient client would take the server down for everyone else.
    // writeAll already treats a failed write as "stop sending", so ignoring the signal is enough.
    signal(SIGPIPE, SIG_IGN)
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("serve requires --model DIR") }
    let port = try o.int("--port", 8080)
    let host = o.values["--host"] ?? "127.0.0.1"
    guard (1...65535).contains(port) else { throw EngineError.invalid("port must be 1...65535") }
    var bindAddress = in_addr()
    guard inet_pton(AF_INET, host, &bindAddress) == 1 else { throw EngineError.invalid("--host requires an IPv4 address") }
    serveBearerToken = ProcessInfo.processInfo.environment["ENGINE_API_KEY"]
    guard host.hasPrefix("127.") || !(serveBearerToken ?? "").isEmpty else {
        throw EngineError.invalid("non-loopback --host requires ENGINE_API_KEY")
    }
    let defaultMaxTokens = try o.nonNegativeInt("--tokens", 2048)   // 0 is documented (P114); `int` refuses it (D46)
    // P114: `--tokens 0` = a request that names no max_tokens may use the rest of the context; `--max-tokens-clamp 1`
    // cuts a max_tokens larger than that room to the room instead of refusing it (RequestDefaults.swift).
    serveMaxTokensClamp = try o.nonNegativeInt("--max-tokens-clamp", 0) != 0
    // P058: an OPTIONAL, opt-in prompt/prefix cache for the server, reusing main.swift's
    // StateCache trunk-only export/import (proven by P024/OBS-ENG-084) rather than a new
    // mechanism. `--state-cache-step` doubles as the prefill chunk width while caching is
    // active (see below) so an ORDINARY chat turn, not just a huge document, crosses interior
    // chunk boundaries and gets rungs written; 512 is a chat-sized default, not the 32768
    // `generate` uses for whole-document reuse. This does not change default `serve` behaviour
    // (state cache is off unless `--state-cache DIR` is given) and touches no gate path.
    // P077 soft stop: a ramped bias on `</think>` instead of a hard cut. Measured on the p3 chain:
    // thinking 19050 -> 7989 tokens with a CORRECT and more complete answer, and the model closes at
    // its own sentence boundary rather than being truncated mid-derivation.
    let thinkBiasMax = Float(o.values["--think-bias-max"] ?? "") ?? 0
    let thinkBiasStart = try o.int("--think-bias-start", 2000)
    let thinkBiasFull = try o.int("--think-bias-full", 8000)
    let thinkBiasDeadline = try o.nonNegativeInt("--think-bias-deadline", 0)
    let stateCacheDir = o.values["--state-cache"]
    let stateCacheStep = try o.int("--state-cache-step", 512)
    // P086: the store is shared and content-keyed, so it is bounded by a cap and an LRU, not by the
    // life of a session. 0 disables the cap and restores the old unbounded behaviour.
    let stateCacheMaxGB = Double(o.values["--state-cache-max-gb"] ?? "") ?? (StateCache.maxBytes / 1e9)
    // P087: how many requests may be in flight at once. Each carries its own KV cache, so this is a
    // memory bound as much as a fairness one; over it the server refuses with 503 + Retry-After.
    let maxConcurrent = try o.int("--max-concurrent", 8)
    guard (1...64).contains(maxConcurrent) else { throw EngineError.invalid("--max-concurrent must be 1...64") }
    serveMaxConcurrent = maxConcurrent
    // P116: when every slot is taken, wait in a bounded FIFO queue (RequestGate.swift) instead of an immediate 503;
    // 0 keeps the old immediate refusal.
    serveQueueMax = try o.nonNegativeInt("--queue-max", 0)
    serveQueueTimeout = Double(try o.nonNegativeInt("--queue-timeout-s", 1800))
    _ = liveSessions.startedAt                  // a lazy global: touch it so the sessions view's uptime starts here
    // P088/P089/P092: how many concurrent, batch-eligible requests before their decode steps are
    // BATCHED into one forward. 0 = off. DEFAULT 4, and the number is the measured crossover:
    //
    //   - a request that is alone keeps MTP (63 tok/s at 32k against 52 serial; K=3 is the peak on
    //     E9, OBS-ENG-167). Nothing about a solo request changes when this is > 0.
    //   - below four eligible rows, interleaving MTP requests already delivers ~88-97 tok/s
    //     aggregate, and a batched step of two or three is SLOWER than that (OBS-ENG-162).
    //   - at four the batched step passes it (93), at eight it is 1.4x (130), so the fleet case --
    //     subagents fanning out -- is where this earns its keep.
    //
    // What it costs, so it is chosen and not discovered: a batched step is a different numeric
    // path (OBS-ENG-158/165) -- four identical greedy requests are 0/4 byte-identical to the solo
    // answer when batched, 4/4 when interleaved; the argmax differs on ~2% of steps at B=4. The
    // operator chose throughput for the fleet case with that on the table (P092). And a request
    // that once joined a group stays off MTP for the rest of THAT request (re-priming the draft
    // state would cost a full prefill); each agent turn is a new request, so the loss is bounded
    // to the tail of one turn.
    let batchMinRows = max(0, Int(o.values["--batch-min"] ?? "") ?? 4)
    // P088: how long a leader lingers for peers. MEASURED: at 2 ms the members never re-converge
    // after their host-side work (detokenise + SSE), so groups were B=2 out of four active requests
    // and the pool re-stacked on 599 of 600 steps -- all cost, no width. The window has to exceed
    // the spread in host time, not the step time.
    let batchWindowMs = Double(o.values["--batch-window-ms"] ?? "") ?? 25
    let batchMaxRows = max(batchMinRows, Int(o.values["--batch-max-rows"] ?? "") ?? 8)
    // P067: `--mtp K` decodes with the checkpoint's MTP head in speculative rounds (the same round
    // as `generate --mtp K`, speculativeLoop in main.swift), K drafts per round. Needs ENGINE_MTP=1 at
    // load so the head is built. Requests that ask for logprobs decode serially (a round carries no
    // per-token logits); a forced tool-call prefix switches the request to serial at that point.
    let mtpDepth = try o.nonNegativeInt("--mtp", 0)
    // P099: batched MTP -- rows served together draft K tokens each and verify them in one [B, K+1] block
    // (`--batch-mtp K`, 0 = plain batched steps as before P099). ENGINE_BATCH_MTP_POLICY: auto (draft while the
    // measured acceptance covers the round's cost against a plain step, sampled in situ), always, never.
    let batchMTPDepth = max(0, Int(o.values["--batch-mtp"] ?? "") ?? (mtpDepth > 0 ? mtpDepth : 0))
    batchMTPDepthShared = batchMTPDepth
    // P090: two levers an agentic client cannot reach through the OpenAI schema, both measured
    // against OMP driving this server (runs/p090).
    //
    // `--reasoning-effort` is this checkpoint's own vendor extension. The template's default is
    // 'xhigh', which has a documented non-termination hazard, so the server's default stays
    // 'medium'; an agent loop pays for every thinking token in latency, and 'low' is one flag away.
    // It is a SERVER setting because it lands at the very head of the system message: changing it
    // per request would invalidate the whole prefix cache.
    //
    // (`reasoning_content` that a client sends BACK is already dropped by `parseChatRequest`, which
    // never copies it into the templated message. OMP round-trips ours faithfully -- 88000
    // characters, 46% of everything on the wire, by request 65 of one task -- and it costs upload
    // bandwidth but not a single prefill token. Measured in runs/p090.)
    let serveReasoningEffort = o.values["--reasoning-effort"] ?? "medium"
    guard ReasoningEffortPolicy.levels.contains(serveReasoningEffort) else {
        throw EngineError.invalid("--reasoning-effort must be one of \(ReasoningEffortPolicy.levels)")
    }
    // P068: thinking guard. `--think-budget N` closes the reasoning after N generated thinking tokens;
    // `--loop-guard N` closes it when a 32-token window of the reasoning recurs N times (the model
    // re-verifying the same table over and over). Both inject the checkpoint's own wrap-up phrase and
    // `</think>` so the answer still gets written. Per request: `thinking_budget`, `loop_guard`.
    let thinkBudget = try o.nonNegativeInt("--think-budget", 0)
    let loopGuard = try o.nonNegativeInt("--loop-guard", 0)
    // P093: the HOT prefix store -- finished sequences kept in GPU memory and resumable at any rung
    // (see HotPrefixStore.swift for what the disk store could not do). GB of unified memory it may
    // hold; 0 disables it. 48 GB is ~15 conversations of 100k tokens.
    let hotCacheGB = Double(o.values["--hot-cache-gb"] ?? "") ?? 48
    // Candidate only: --hot-cache-gb is an idle ceiling in shared mode. The fixed
    // hot20 full-soak profile remains unchanged unless this flag is explicitly set.
    serveSharedRAMBudget = ProcessInfo.processInfo.environment["ENGINE_SHARED_RAM_BUDGET"] == "1"
    // P093: the prefill chunk width, decoupled from the rung spacing (it used to BE the rung
    // spacing whenever the disk cache was on: 512). MEASURED cold, 16k prompt: 512 -> 788 tok/s,
    // 1024 -> 981, 2048 -> 1122, 4096 -> 1222 (runs/p093/chunk_ab.txt) -- the MoE sweeps most of
    // its 512 experts per chunk whatever the chunk holds. 0 = ADAPTIVE: 4096 while the request is
    // alone, 1024 while others are active (a chunk holds the model thread, so the width is the
    // decode stall every other row pays: ~1 s at 1024). Chunk ends are aligned to multiples of the
    // rung spacing so a prefill that resumes from a hot rung at an arbitrary length still writes
    // disk rungs where the next session can find them.
    let prefillChunk = try o.nonNegativeInt("--prefill-chunk", 0)
    // P093: this checkpoint's chat template renders the `reasoning_content` of EVERY earlier
    // assistant turn back into the prompt (`preserve_thinking` defaults to true in
    // chat_template.jinja); the server used to drop it silently, which is a deviation from the
    // checkpoint's own contract the client cannot see. `template` hands it through and lets the
    // template decide (a request may set `preserve_thinking: false` to keep only the current
    // turn's tool loop); `none` restores the old behaviour.
    let preserveThinking = o.values["--preserve-thinking"] ?? "template"
    guard ["template", "none"].contains(preserveThinking) else { throw EngineError.invalid("--preserve-thinking must be template or none") }

    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    let wiredConfiguration = applyWiredLimit()
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    engineStopIdSet = engineStopIds(dir.path)
    let configObject = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
    let textConfig = configObject["text_config"] as? [String: Any] ?? configObject
    let nativeContext = textConfig["max_position_embeddings"] as? Int ?? 0
    let contextLimit = try o.int("--max-context", nativeContext)
    guard contextLimit > 0, nativeContext == 0 || contextLimit <= nativeContext else {
        throw EngineError.invalid("--max-context must be positive and no larger than the checkpoint context")
    }
    guard defaultMaxTokens >= 0, defaultMaxTokens < contextLimit, stateCacheStep > 0,
          hotCacheGB.isFinite, hotCacheGB >= 0, stateCacheMaxGB.isFinite, stateCacheMaxGB >= 0,
          batchWindowMs.isFinite, batchWindowMs >= 0, sharedChunk > 0, aloneChunk > 0,
          mtpDepth <= 16, batchMTPDepth <= 16 else { throw EngineError.invalid("invalid serving limits") }
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    let layers = textConfig["num_hidden_layers"] as? Int ?? 1
    let fullLayers = (textConfig["layer_types"] as? [String])?.filter { $0 == "full_attention" }.count ?? layers
    let kvHeads = textConfig["num_key_value_heads"] as? Int ?? (textConfig["num_attention_heads"] as? Int ?? 1)
    let headDim = textConfig["head_dim"] as? Int ?? ((textConfig["hidden_size"] as? Int ?? 4096) / max(1, textConfig["num_attention_heads"] as? Int ?? 1))
    let qwenConfig = (context.model as? Qwen4ExpModel)?.configuration.text
    let bytesPerToken: Double
    if let qwenConfig {
        bytesPerToken = AdmissionBudget.perTokenBytes(
            attentionLayers: fullLayers + (mtpDepth > 0 ? 1 : 0), kvHeads: kvHeads, headDim: headDim,
            indexerHeadDim: qwenConfig.indexerHeadDim, indexerCompressRatio: qwenConfig.indexerCompressRatio)
    } else {
        // Preserve the prior generic-family margin until that cache layout is measured.
        bytesPerToken = Double(fullLayers + (mtpDepth > 0 ? 1 : 0)) * (Double(kvHeads) * Double(headDim) * 4 + 256)
    }
    // Cold load is only a diagnostic witness. Lazy resident storage must be bound
    // before admission is calculated (see the owner-thread preparation below).
    let loadedModelBytes = Memory.activeMemory
    serveRuntime = ["os": ProcessInfo.processInfo.operatingSystemVersionString,
                    "program_identity": try StateCache.programIdentity(), "host": host,
                    "max_context": contextLimit, "model_loaded_bytes": loadedModelBytes,
                    "kv_bytes_per_token": bytesPerToken, "wired_configuration": wiredConfiguration,
                    "hot_cache_gb": hotCacheGB, "prefill_chunk": prefillChunk,
                    "prefill_chunk_shared": sharedChunk, "prefill_chunk_alone": aloneChunk, "prefill_budget_cut": h57BudgetCut, "batch_min": batchMinRows,
                    "batch_mtp_policy": batchMTPPolicy,
                    "scheduler": ProcessInfo.processInfo.environment["ENGINE_SCHEDULER"] == "fifo" ? "fifo" : "phase",
                    "scheduler_decode_burst": 4]
    let modelId = dir.lastPathComponent
    // P089 U10: from this line on, EVERY MLX call in this process happens on the model thread, and
    // every MLXArray a request owns is created and destroyed there. Loading above is single
    // threaded, so it needs no owner; serving is not, and two crashes proved a gate is not enough.
    let mtModel = context.model
    let mtRatio = (mtModel as? Qwen4ExpModel)?.configuration.text.indexerCompressRatio ?? 4
    let batchPrefill = ProcessInfo.processInfo.environment["ENGINE_BATCH_PREFILL"] == "1"
    if batchPrefill, ProcessInfo.processInfo.environment["ENGINE_PREFILL_ROW_PROJECTIONS"] != "all" {
        throw EngineError.invalid("batched prefill requires ENGINE_PREFILL_ROW_PROJECTIONS=all for the exact projection geometry")
    }
    guard !(h8BoundaryProbe && h26PrefillMetadata) else {
        throw EngineError.invalid("H26 metadata and the H8 B1-only trace are mutually exclusive")
    }
    guard !(h25Metadata || h25CanonicalPrefix) || !h8BoundarySplit else {
        throw EngineError.invalid("H25 requires ENGINE_H8_BOUNDARY_SPLIT OFF")
    }
    if h25Metadata || h25CanonicalPrefix {
        guard maxConcurrent == 8, batchMinRows == 4, batchMaxRows == 8, batchWindowMs == 25,
              batchPrefill, stateCacheDir == nil, hotCacheGB == 20,
              // H50: C1024, or adaptive (4096 alone / 1024 shared) with 1024-row width-canonical projections
              prefillChunk == 1024 || (prefillChunk == 0 && sharedChunk == 1024 && h50WidthCanonical
                                       && aloneChunk % 1024 == 0 && aloneChunk <= h50MaxCanonicalChunk),
              stateCacheStep == 512, (Int(o.values["--hot-keep-rungs"] ?? "") ?? 2) == 2,
              mtpDepth == 3, batchMTPDepth == 3, batchMTPPolicy == "auto",
              serveSharedRAMBudget, hotCoalesce,
              ProcessInfo.processInfo.environment["ENGINE_SCHEDULER"] != "fifo",
              let diagnosticModel = mtModel as? Qwen4ExpModel, diagnosticModel.mtp != nil else {
            throw EngineError.invalid("H25 policy/probe requires Qwen MTP3 auto, phase, CB4/max8/window25, native ON/all, RAM20/shared, coalesce ON, disk OFF, C1024 (or C0 + shared1024 + alone<=2048 + ENGINE_PREFILL_WIDTH_CANONICAL=1024), rung512/keep2")
        }
    }
    serveRuntime["h8_boundary_probe"] = h8BoundaryProbe
    serveRuntime["default_max_tokens"] = defaultMaxTokens
    serveRuntime["queue_max"] = serveQueueMax
    serveRuntime["queue_timeout_s"] = Int(serveQueueTimeout)
    serveRuntime["max_tokens_clamp"] = serveMaxTokensClamp
    serveRuntime["reasoning_effort_default"] = serveReasoningEffort
    serveRuntime["h8_cold_split_positions"] = [Int]()
    serveRuntime["h25_canonical_prefix"] = h25CanonicalPrefix
    serveRuntime["h25_canonical_width"] = h25CanonicalWidth
    serveRuntime["prefill_width_canonical_rows"] = Qwen4ExpPrefillWidth.canonicalRows
    serveRuntime["h26_prefill_metadata"] = h26PrefillMetadata
    serveRuntime["h26_prefill_interval_capacity"] = h26PrefillIntervalCapacity
    serveRuntime["batch_prefill"] = batchPrefill
    serveRuntime["release_pooled_private_history"] = releasePooledPrivateHistory
    // P106 H48: requested AND applicable -- an ineligible diagnostic environment keeps the stacked pool
    serveRuntime["row_resident_kv"] = rowResidentKV && Qwen4ExpModel.rowResidentEligible
    serveRuntime["prefill_batch_max"] = batchPrefill ? 4 : 1
    serveRuntime["prefill_batch_token_budget"] = 4096
    serveRuntime["prefill_batch_decode_token_budget"] = 1024
    serveRuntime["prefill_batch_chunk_widths"] = [512, 1024]
    serveRuntime["prefill_row_projections"] = ProcessInfo.processInfo.environment["ENGINE_PREFILL_ROW_PROJECTIONS"] ?? "none"
    // P106 B32 (H42): rendered-segment id cache. render_ms ~1.5% of applyChatTemplate, encode
    // ~97% (H40/H41); caching segment ids removes the repeat encode for every previously seen
    // segment with byte-identical results. 0 disables; VERIFY double-checks ids per request
    // and returns HTTP 500 on mismatch (diagnostic only; unset by the production launcher).
    let templateCacheMB = Int(ProcessInfo.processInfo.environment["ENGINE_TEMPLATE_CACHE_MB"] ?? "256") ?? 256
    templateCacheVerify = ProcessInfo.processInfo.environment["ENGINE_TEMPLATE_CACHE_VERIFY"] == "1"
    if templateCacheMB > 0 {
        templateRenderShared = try ChatTemplateRender(modelDir: dir)
        templateCacheShared = try TemplateTokenCache(capacityBytes: templateCacheMB * 1024 * 1024,
                                                     tokenizer: context.tokenizer, modelDir: dir)
        FileHandle.standardError.write(String(format: "engine serve: template cache %d MB%s\n", templateCacheMB,
                                              templateCacheVerify ? " (verify)" : "").data(using: .utf8)!)
    }
    serveRuntime["template_cache_mb"] = templateCacheMB
    serveRuntime["template_cache_verify"] = templateCacheVerify
    modelThreadShared = ModelThread(minBatch: max(1, batchMinRows), gatherWindow: batchWindowMs / 1000,
                                    maxBatch: batchMaxRows, phaseScheduling: ProcessInfo.processInfo.environment["ENGINE_SCHEDULER"] != "fifo",
                                    prefillMaxBatch: batchPrefill ? 4 : 1, prefillTokenBudget: 4096, snapshot: { req, token in
        let row = seqRegistry[req.key].row
        let pooledSpec = batchPoolShared.specPool != nil && batchPoolShared.memberIds.contains(ObjectIdentifier(row))
        return StepResult(token: token, queued: row.takeQueued(), hasSpec: row.spec != nil || pooledSpec)
    }, ownerTrace: serveOwnerOrderTrace.enabled ? { serveOwnerOrderTrace.observe($0) } : nil) { reqs in
        defer { serveMemoryWitness.publish() }
        guard let qm = mtModel as? Qwen4ExpModel else { return reqs.map { _ in -1 } }
        return runBatchedStep(reqs, model: mtModel, qm: qm, ratio: mtRatio)
    }
    hotStoreShared = HotPrefixStore(capBytes: Int(hotCacheGB * 1e9), rungStep: stateCacheStep,
        keepDecodeRungs: Int(o.values["--hot-keep-rungs"] ?? "") ?? 2, strictBudget: serveSharedRAMBudget)
    hotStoreShared.debug = ProcessInfo.processInfo.environment["ENGINE_HOT_DEBUG"] != nil
    // P106 H54: keep the prefix-step rungs of an entry's last 32k tokens (ENGINE_HOT_ANCHOR_WINDOW, 0 = legacy retention)
    hotStoreShared.anchorStep = prefixRungStep
    hotStoreShared.anchorWindow = Int(ProcessInfo.processInfo.environment["ENGINE_HOT_ANCHOR_WINDOW"] ?? "") ?? 32768
    serveRuntime["hot_anchor_window"] = hotStoreShared.anchorWindow
    hotStoreShared.evictAffinity = ProcessInfo.processInfo.environment["ENGINE_HOT_EVICT"] != "lru"
    if hotStoreShared.enabled {
        FileHandle.standardError.write(String(format: "engine serve: hot prefix store cap %.0f GB, rung every %d tokens\n", hotCacheGB, stateCacheStep).data(using: .utf8)!)
    }
    var stateCache: StateCacheConfig? = nil
    if let sd = stateCacheDir {
        let scDir = URL(fileURLWithPath: sd)
        try FileManager.default.createDirectory(at: scDir, withIntermediateDirectories: true)
        let witness = (context.model as? Qwen4ExpModel)?.fusionWitness() ?? ""
        FileHandle.standardError.write(Data("engine: verifying full weight content for persistent cache identity…\n".utf8))
        stateCache = StateCacheConfig(dir: scDir, identity: try StateCache.identity(dir: dir, witness: witness),
                                      step: stateCacheStep, writeExact: false, maxBytes: stateCacheMaxGB * 1e9)
        serveRuntime["state_cache_identity"] = stateCache?.identity
        FileHandle.standardError.write(String(format: "engine serve: prefix cache at %@, rung %d, blocks %d tokens, cap %.0f GB\n",
                                              scDir.path, stateCacheStep, StateCache.blockRows, stateCacheMaxGB).data(using: .utf8)!)
        FileHandle.standardError.write("engine serve: state cache \(scDir.path) step \(stateCacheStep)\n".data(using: .utf8)!)
    }

    let serverFD = socket(AF_INET, SOCK_STREAM, 0)
    guard serverFD >= 0 else { throw EngineError.invalid("serve: socket() failed (errno \(errno))") }
    var reuse: Int32 = 1
    setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(port).bigEndian
    addr.sin_addr = bindAddress
    let bindOk = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(serverFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindOk == 0 else { throw EngineError.invalid("serve: bind() failed on port \(port) (errno \(errno))") }
    guard listen(serverFD, 16) == 0 else { throw EngineError.invalid("serve: listen() failed (errno \(errno))") }
    // P067: warm the kernels this server will dispatch BEFORE the first client arrives. The first
    // use of every kernel shape pays MLX's JIT compile (cross-process cached by Metal; OBS-ENG-140/141:
    // a cold verify width ran 4x slower than warm on its first block). One short throwaway prefill,
    // four serial steps, and -- with MTP -- the prime and two rounds at the served depth.
    let servingStorage = modelThreadShared.exclusive {
        // Must precede warm-up and every served cache, including NO_WARM mode.
        if mtModel is Qwen4ExpModel { Qwen4ExpCacheCapacity.configureGrowthLimit(contextLimit) }
        return (mtModel as? Qwen4ExpModel)?.prepareServingStorage() ?? [:]
    }
    serveRuntime["serving_storage"] = servingStorage
    if ProcessInfo.processInfo.environment["ENGINE_SERVE_NO_WARM"] == nil {
        let tw = Date()
        modelThreadShared.exclusive { serveWarmUp(context: context, mtpDepth: mtpDepth); serveMemoryWitness.publish() }
        FileHandle.standardError.write(String(format: "engine serve: warm-up done in %.1f s (serial%@)\n", Date().timeIntervalSince(tw), mtpDepth > 0 ? " + MTP K=\(mtpDepth)" : "").data(using: .utf8)!)
    }
    let residentModelBytes = modelThreadShared.exclusive {
        serveMemoryWitness.publish()
        return Memory.activeMemory
    }
    // Conservative reservation from the materialized resident model, including
    // storage prepared even in NO_WARM diagnostics. Triple KV covers private,
    // pooled and simultaneous restack buffers; the other margins are unchanged.
    let allocatorGB = Double(ProcessInfo.processInfo.environment["ENGINE_CACHE_LIMIT_GB"] ?? "32") ?? 32
    // Keep physical, allocator and workspace margins identical. Shared mode leases
    // the remainder between the SAME active bound and reclaimable cache storage.
    serveRAMLiveLimitBytes = Double(ProcessInfo.processInfo.physicalMemory) * 0.90
        - (max(32, allocatorGB) + 48) * 1e9
    let available = serveRAMLiveLimitBytes - Double(residentModelBytes)
        - (serveSharedRAMBudget ? 0 : hotCacheGB * 1e9)
    let requestedBudget = Double(o.values["--kv-budget-gb"] ?? "").map { $0 * 1e9 } ?? available
    guard requestedBudget.isFinite, requestedBudget > 0, requestedBudget <= available else {
        close(serverFD)
        throw EngineError.invalid("KV budget exceeds memory remaining after resident model/cache/workspace reserves")
    }
    var historyCapacity: (@Sendable (Int) -> Double)? = nil
    var fixedBytesPerSequence = 384e6
    if let a = qwenConfig {
        let kvStep = Qwen4ExpCacheCapacity.kvStep, indexerStep = Qwen4ExpCacheCapacity.indexerStep
        let configuredChunk = prefillChunk > 0 ? prefillChunk
            : (Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "") ?? 0)
        // chunkEnd only rounds down/splits. Include native chunks, warm-up and verify.
        let forwardRows = max(1024, max(configuredChunk > 0 ? configuredChunk : max(aloneChunk, sharedChunk),
                                       max(mtpDepth, batchMTPDepth) + 1))
        historyCapacity = { length in
            AdmissionBudget.historyCapacityBytes(length: length,
                attentionLayers: fullLayers + (mtpDepth > 0 ? 1 : 0), kvHeads: kvHeads, headDim: headDim,
                indexerHeadDim: a.indexerHeadDim, indexerCompressRatio: a.indexerCompressRatio,
                kvStep: kvStep, indexerStep: indexerStep, maxForwardRows: forwardRows, capacityGrowthLimit: contextLimit)
        }
        let stableState = Qwen4ExpCacheCapacity.fixedStateBytes(a, mtp: mtpDepth > 0)
        let rungBytes = Qwen4ExpCacheCapacity.hotRungBytes(a, maxForwardRows: forwardRows,
                                                        allocationSlack: max(16_384, 3 * Int(getpagesize())))
        let liveRungReserve = hotStoreShared.enabled ? Double(hotStoreShared.retainedRungLimit + 1) * rungBytes : 0
        fixedBytesPerSequence = max(fixedBytesPerSequence, 3 * stableState + liveRungReserve)
        serveRuntime["hot_live_rung_limit"] = hotStoreShared.retainedRungLimit
        serveRuntime["hot_rung_bytes_upper_bound_derived"] = rungBytes
        serveRuntime["hot_live_rung_reserve_bytes_derived"] = liveRungReserve
        serveRuntime["history_capacity_growth_limit"] = contextLimit
        serveRuntime["history_kv_step"] = kvStep
        serveRuntime["history_indexer_step"] = indexerStep
        serveRuntime["history_max_forward_rows"] = forwardRows
        serveRuntime["stable_state_bytes_per_row_derived"] = stableState
    }
    serveAdmission = AdmissionBudget(maxContext: contextLimit, capacityBytes: requestedBudget,
        bytesPerToken: bytesPerToken, fixedBytesPerSequence: fixedBytesPerSequence,
        historyCapacityBytes: historyCapacity, hotCacheCeilingBytes: serveSharedRAMBudget ? hotCacheGB * 1e9 : nil)
    serveRuntime["shared_ram_budget"] = serveSharedRAMBudget
    serveRuntime["ram_live_limit_bytes"] = serveRAMLiveLimitBytes
    modelThreadShared.exclusive {
        if serveSharedRAMBudget {
            hotStoreShared.permitAllocation = { bytes in
                precondition(modelThreadShared.isOwner)
                return Double(Memory.activeMemory) + Double(bytes) <= serveRAMLiveLimitBytes
            }
            _ = hotStoreShared.setBudgetBytes(Int(serveAdmission.snapshot()["hot_target_bytes"]!))
        }
        serveMemoryWitness.publish()
    }
    serveRuntime["fixed_bytes_per_sequence"] = fixedBytesPerSequence
    serveRuntime["full_context_history_capacity_bytes_per_row_derived"] = historyCapacity?(contextLimit)
        ?? AdmissionBudget.historyCapacityLengths(length: contextLimit, kvStep: 1024).kv * bytesPerToken
    serveRuntime["kv_budget_bytes"] = requestedBudget
    serveRuntime["model_resident_bytes"] = residentModelBytes
    serveRuntime["indexer_route_graph_construction_witness"] = Qwen4ExpIndexerRouteWitness.enabled
    serveRuntime["restack_completion_barrier"] = batchRestackEval
    serveRuntime["kernel_warmup"] = ProcessInfo.processInfo.environment["ENGINE_SERVE_NO_WARM"] == nil
    FileHandle.standardError.write("engine serve: model \(modelId) ready\(mtpDepth > 0 ? " (MTP K=\(mtpDepth))" : ""), listening on \(host):\(port)\n".data(using: .utf8)!)

    // P087/P089: one thread per connection, doing HTTP, the chat template, detokenising and the
    // socket -- and nothing else. Every model call and every MLXArray belongs to the model thread.
    let connectionCache = stateCache
    let connections = ActiveCount()
    let connectionLimit = max(16, maxConcurrent * 2 + 8) + serveQueueMax   // a waiting request holds its connection
    if let data = try? JSONSerialization.data(withJSONObject: serveRuntime, options: [.sortedKeys]), let line = String(data: data, encoding: .utf8) {
        FileHandle.standardError.write(Data("engine runtime: \(line)\n".utf8))
    }
    while true {
        let clientFD = accept(serverFD, nil, nil)
        guard clientFD >= 0 else { continue }
        HTTPTransport.configure(clientFD)
        guard connections.enter(max: connectionLimit) else { close(clientFD); continue }
        // The capacity gate is applied inside the handler, once the path is known: a GET (the model list, the
        // sessions view a dashboard polls every second) must neither count as a request -- `activeRequests.current`
        // picks the prefill chunk width -- nor be refused with a 503 while the server is full.
        let worker = Thread {
            handleServeConnection(clientFD, context: context, modelId: modelId, defaultMaxTokens: defaultMaxTokens, stateCache: connectionCache, mtpDepth: mtpDepth, thinkBudget: thinkBudget, loopGuard: loopGuard, thinkBiasMax: thinkBiasMax, thinkBiasStart: thinkBiasStart, thinkBiasFull: thinkBiasFull, thinkBiasDeadline: thinkBiasDeadline, batchMinRows: batchMinRows, serveReasoningEffort: serveReasoningEffort, prefillChunk: prefillChunk, keepReasoning: preserveThinking != "none")
            close(clientFD)
            connections.leave()
        }
        worker.stackSize = 4 << 20
        worker.start()
    }
}

nonisolated(unsafe) var serveMaxConcurrent = 8
nonisolated(unsafe) var serveQueueMax = 0
nonisolated(unsafe) var serveQueueTimeout = 1800.0
nonisolated(unsafe) var serveAdmission: AdmissionBudget!
nonisolated(unsafe) private var serveSharedRAMBudget = false
nonisolated(unsafe) private var serveRAMLiveLimitBytes = 0.0
nonisolated(unsafe) private var serveRAMReclaimWitness: [String: Double] = [:] // owner only
nonisolated(unsafe) var serveBearerToken: String?
nonisolated(unsafe) var serveRuntime: [String: Any] = [:]
nonisolated(unsafe) var serveMaxTokensClamp = false
let serveMaxTokensWitness = DecisionWitness()
let serveEffortWitness = DecisionWitness()

// MARK: - HTTP plumbing

@discardableResult
private func writeAll(_ fd: Int32, _ data: Data) -> Bool { HTTPTransport.writeAll(fd, data) }

private func writeHTTPResponse(_ fd: Int32, status: Int, statusText: String, contentType: String, body: Data) {
    let retry = status == 503 ? "Retry-After: 1\r\n" : ""
    let head = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\n\(retry)Connection: close\r\n\r\n"
    var out = Data(head.utf8)
    out.append(body)
    writeAll(fd, out)
}

private func jsonErrorBody(_ message: String, type: String = "invalid_request_error") -> Data {
    (try? JSONSerialization.data(withJSONObject: ["error": ["message": message, "type": type]])) ?? Data()
}

// MARK: - JSON <-> Sendable bridging (Tokenizers' Message = [String: any Sendable])

private func toSendable(_ value: Any) -> any Sendable {
    if let v = value as? Bool { return v }
    if let v = value as? Int { return v }
    if let v = value as? Double { return v }
    if let v = value as? String { return v }
    if let v = value as? [Any] { return v.map(toSendable) }
    if let v = value as? [String: Any] { return toSendableDict(v) }
    return "\(value)"
}
private func toSendableDict(_ d: [String: Any]) -> [String: any Sendable] { d.mapValues(toSendable) }

// MARK: - request model

private enum ToolChoiceMode { case auto, none, required, named(String) }

private struct ChatRequest {
    var messages: [Message]
    var maxTokens: Int
    var requestedMaxTokens: Int?
    var temperature: Float
    var topP: Float
    var topK: Int
    var thinkingBudget: Int?
    var loopGuard: Int?
    var thinkBiasMax: Float?
    var thinkBiasDeadline: Int?
    var n: Int
    var stream: Bool
    var streamIncludeUsage: Bool
    var stopStrings: [String]
    var seed: UInt64?
    var reasoningEffort: String
    var preserveThinking: Bool?
    var logprobsRequested: Bool
    var topLogprobsCount: Int
    var rawTools: [[String: Any]]
    var toolChoice: ToolChoiceMode
    var parallelToolCalls: Bool
    var jsonInstruction: String?
}

private func parseChatRequest(_ obj: [String: Any], defaultMaxTokens: Int, defaultEffort: String = "medium", keepReasoning: Bool = true) -> ChatRequest? {
    guard let rawMessages = obj["messages"] as? [[String: Any]], !rawMessages.isEmpty else { return nil }
    var messages: [Message] = []
    for m in rawMessages {
        guard let role = m["role"] as? String else { continue }
        var msg: [String: any Sendable] = ["role": role]
        if let s = m["content"] as? String {
            msg["content"] = s
        } else if let parts = m["content"] as? [[String: Any]] {
            msg["content"] = parts.compactMap { $0["text"] as? String }.joined()
        } else {
            msg["content"] = ""
        }
        if let tcs = m["tool_calls"] as? [[String: Any]] {
            // The wire format carries `function.arguments` as a JSON STRING; this checkpoint's own
            // chat_template.jinja iterates it as a mapping (`tool_call.arguments|items`), so replay
            // turns need it decoded back to an object before it reaches the template.
            let converted: [[String: Any]] = tcs.map { tc in
                var tc = tc
                if var fn = tc["function"] as? [String: Any], let argsStr = fn["arguments"] as? String,
                   let d = argsStr.data(using: .utf8), let parsed = try? JSONSerialization.jsonObject(with: d) {
                    fn["arguments"] = parsed
                    tc["function"] = fn
                }
                return tc
            }
            // P106 B41 (D49): keep the arguments' member order (the order the model generated them in, which the
            // streamed fragments and the buffered string both carry) all the way into the template.
            msg["tool_calls"] = zip(tcs, converted).map { raw, tc -> [String: any Sendable] in
                var d = toSendableDict(tc)
                if var fn = d["function"] as? [String: any Sendable],
                   let argsStr = (raw["function"] as? [String: Any])?["arguments"] as? String,
                   let ordered = orderedJinjaValue(json: argsStr), case .object = ordered {
                    fn["arguments"] = ordered
                    d["function"] = fn
                }
                return d
            }
        }
        if let tcid = m["tool_call_id"] as? String { msg["tool_call_id"] = tcid }
        if let name = m["name"] as? String { msg["name"] = name }
        // P093: the reasoning a client sends back is part of the prompt under this checkpoint's
        // template (its `preserve_thinking` defaults to true). Dropped only under
        // `--preserve-thinking none`.
        if keepReasoning, role == "assistant", let r = m["reasoning_content"] as? String, !r.isEmpty { msg["reasoning_content"] = r }
        messages.append(msg)
    }
    guard !messages.isEmpty else { return nil }

    let requestedMaxTokens = (obj["max_tokens"] as? Int) ?? (obj["max_completion_tokens"] as? Int)
    let maxTokens = requestedMaxTokens ?? defaultMaxTokens   // serve replaces it after tokenising (MaxTokensPolicy)
    let temp = (obj["temperature"] as? Double).map(Float.init) ?? 0
    let topP = (obj["top_p"] as? Double).map(Float.init) ?? 1
    let topK = max(0, (obj["top_k"] as? Int) ?? 0)   // P068: the checkpoint asks for top_k 20 with sampling
    let thinkingBudget = obj["thinking_budget"] as? Int
    let loopGuardReq = obj["loop_guard"] as? Int
    let thinkBiasReq = (obj["think_bias_max"] as? Double).map { Float($0) }
    let thinkBiasDeadlineReq = obj["think_bias_deadline"] as? Int
    let n = (obj["n"] as? Int) ?? 1
    let stream = (obj["stream"] as? Bool) ?? false
    let streamIncludeUsage = ((obj["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false
    var stopStrings: [String] = []
    if let s = obj["stop"] as? String { stopStrings = [s] } else if let arr = obj["stop"] as? [String] { stopStrings = arr }
    let seed = (obj["seed"] as? Int).map { UInt64(bitPattern: Int64($0)) }
    // The template's OWN default is 'xhigh'. Its old hazard -- an open-ended question running thousands of tokens
    // without closing </think> -- is what the P078 soft stop (--think-bias-*) closes, so since P114 production
    // defaults to xhigh (operator 2026-09-25: every default at its maximum). Other spellings map to the nearest
    // level (ReasoningEffortPolicy); a client's explicit level always wins.
    let reasoningEffort = ReasoningEffortPolicy.resolve(obj["reasoning_effort"] as? String, serverDefault: defaultEffort)
    let preserveThinking = obj["preserve_thinking"] as? Bool
    let logprobsRequested = (obj["logprobs"] as? Bool) ?? false
    let topLogprobsCount = (obj["top_logprobs"] as? Int) ?? 0

    let rawTools = (obj["tools"] as? [[String: Any]]) ?? []
    var toolChoice: ToolChoiceMode = .auto
    if let tc = obj["tool_choice"] as? String {
        switch tc {
        case "none": toolChoice = .none
        case "required": toolChoice = .required
        default: toolChoice = .auto
        }
    } else if let tc = obj["tool_choice"] as? [String: Any], let fn = tc["function"] as? [String: Any],
              let name = fn["name"] as? String {
        toolChoice = .named(name)
    }
    let parallelToolCalls = (obj["parallel_tool_calls"] as? Bool) ?? true

    // No grammar-constrained decoding here, so `response_format` is honored the only way a plain
    // autoregressive model can honor it: a strong trailing instruction, not a hard guarantee. Per the
    // "silently ignoring a requested parameter is a MUST failure" rule, the alternative -- accepting
    // the field and doing nothing -- would be worse than this best-effort attempt.
    var jsonInstruction: String? = nil
    if let rf = obj["response_format"] as? [String: Any], let ty = rf["type"] as? String, ty != "text" {
        let schema: [String: Any]? = (rf["json_schema"] as? [String: Any])?["schema"] as? [String: Any]
        if let schema, let d = try? JSONSerialization.data(withJSONObject: schema), let s = String(data: d, encoding: .utf8) {
            jsonInstruction = "\n\nRespond with ONLY valid JSON matching this JSON Schema, with no markdown code fences and no text outside the JSON: \(s)"
        } else {
            jsonInstruction = "\n\nRespond with ONLY valid JSON, with no markdown code fences and no text outside the JSON."
        }
    }
    if let instruction = jsonInstruction, var last = messages.popLast() {
        let existing = (last["content"] as? String) ?? ""
        last["content"] = existing + instruction
        messages.append(last)
    }

    return ChatRequest(
        messages: messages, maxTokens: maxTokens, requestedMaxTokens: requestedMaxTokens, temperature: temp, topP: topP, topK: topK, thinkingBudget: thinkingBudget, loopGuard: loopGuardReq, thinkBiasMax: thinkBiasReq, thinkBiasDeadline: thinkBiasDeadlineReq, n: n, stream: stream,
        streamIncludeUsage: streamIncludeUsage, stopStrings: stopStrings, seed: seed, reasoningEffort: reasoningEffort,
        preserveThinking: preserveThinking, logprobsRequested: logprobsRequested, topLogprobsCount: topLogprobsCount, rawTools: rawTools, toolChoice: toolChoice,
        parallelToolCalls: parallelToolCalls, jsonInstruction: jsonInstruction)
}

// MARK: - tool-call parsing (THIS checkpoint's XML form, from chat_template.jinja)

private func toolParamTypeMap(from tools: [[String: Any]]) -> [String: [String: String]] {
    var out: [String: [String: String]] = [:]
    for t in tools {
        guard let fn = t["function"] as? [String: Any], let name = fn["name"] as? String,
              let params = fn["parameters"] as? [String: Any], let props = params["properties"] as? [String: Any]
        else { continue }
        var m: [String: String] = [:]
        for (k, v) in props {
            if let d = v as? [String: Any], let ty = d["type"] as? String { m[k] = ty }
        }
        out[name] = m
    }
    return out
}

/// One key:value pair as a JSON fragment, e.g. `"city":"Warsaw"` -- built via JSONSerialization
/// (correct escaping for any type) and stripped of its own wrapping braces, so streamed argument
/// fragments can be concatenated by the client into one valid object, exactly like a real
/// streaming tool call: `{` + `"a":1` + `,"b":2` + `}`.
private func jsonKeyValueFragment(_ key: String, _ value: Any) -> String {
    guard let d = try? JSONSerialization.data(withJSONObject: [key: value]), var s = String(data: d, encoding: .utf8) else { return "" }
    if s.hasPrefix("{") { s.removeFirst() }
    if s.hasSuffix("}") { s.removeLast() }
    return s
}

private func coerceParamValue(_ raw: String, type: String?) -> Any {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    // P106 B41 (D49): a string value is returned EXACTLY as generated. The template renders
    // `<parameter=P>\n` + value + `\n</parameter>` and both parsers already strip exactly those two
    // newlines; trimming more cut a file's final newline and made the replayed call differ from the
    // tokens the model generated (the next turn re-prefilled everything after it).
    switch type {
    case "integer": return Int(trimmed) ?? trimmed
    case "number": return Double(trimmed) ?? trimmed
    case "boolean": return Bool(trimmed) ?? (trimmed.lowercased() == "true")
    case "array", "object":
        if let d = trimmed.data(using: .utf8), let parsed = try? JSONSerialization.jsonObject(with: d) { return parsed }
        return trimmed
    default: return raw
    }
}

/// A JSON value as a Jinja value with every object's member order kept and scalars typed from their JSON
/// spelling (`1` an int, `true` a boolean -- the Any bridge turned 0/1 into booleans).
func orderedJinjaValue(json raw: String) -> Jinja.Value? {
    let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let first = t.first else { return nil }
    switch first {
    case "{":
        guard let members = jsonObjectMembersInOrder(t) else { return nil }
        var od = OrderedDictionary<String, Jinja.Value>()
        for m in members { guard let v = orderedJinjaValue(json: m.raw) else { return nil }; od[m.key] = v }
        return .object(od)
    case "[":
        guard let parsed = try? JSONSerialization.jsonObject(with: Data(t.utf8)) as? [Any] else { return nil }
        return try? Jinja.Value(any: parsed.map { v -> Any? in v is NSNull ? nil : v })
    case "\"":
        return ((try? JSONSerialization.jsonObject(with: Data(t.utf8), options: .fragmentsAllowed)) as? String).map { .string($0) }
    default:
        if t == "true" { return .boolean(true) }
        if t == "false" { return .boolean(false) }
        if t == "null" { return .null }
        if let v = Int(t) { return .int(v) }
        return Double(t).map { .double($0) }
    }
}

/// Parses `<tool_call>\n<function=NAME>\n<parameter=P>V</parameter>...\n</function>\n</tool_call>`
/// blocks -- the exact shape this checkpoint's own chat_template.jinja instructs the model to emit
/// (read from models/qwen3.8-flash-next-mix1v/chat_template.jinja), not the JSON `{"name":...}` form
/// other Qwen releases use. Returns OpenAI-shaped tool_calls (arguments as a JSON STRING, per the
/// wire spec) plus the text before the first call.
private func extractToolCalls(from text: String, toolParamTypes: [String: [String: String]]) -> (calls: [[String: Any]], remainder: String) {
    guard let callRegex = try? NSRegularExpression(
        pattern: "<tool_call>\\s*<function=([^>]+)>([\\s\\S]*?)</function>\\s*</tool_call>", options: [])
    else { return ([], text) }
    let ns = text as NSString
    let matches = callRegex.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length))
    guard !matches.isEmpty else { return ([], text) }
    let paramRegex = try? NSRegularExpression(pattern: "<parameter=([^>]+)>\\n?([\\s\\S]*?)\\n?</parameter>", options: [])

    var calls: [[String: Any]] = []
    for m in matches {
        let name = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespaces)
        let body = ns.substring(with: m.range(at: 2))
        // P106 B41 (D49): members in generation order, as the streamed fragments already are (a dictionary +
        // JSONSerialization emitted them in hash order).
        var order: [String] = []
        var values: [String: Any] = [:]                  // a repeated parameter keeps its first position, last value
        if let paramRegex {
            let bns = body as NSString
            for pm in paramRegex.matches(in: body, options: [], range: NSRange(location: 0, length: bns.length)) {
                let pname = bns.substring(with: pm.range(at: 1)).trimmingCharacters(in: .whitespaces)
                let praw = bns.substring(with: pm.range(at: 2))
                if values[pname] == nil { order.append(pname) }
                values[pname] = coerceParamValue(praw, type: toolParamTypes[name]?[pname])
            }
        }
        let argsString = "{" + order.map { jsonKeyValueFragment($0, values[$0]!) }.joined(separator: ",") + "}"
        calls.append([
            "id": "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24),
            "type": "function",
            "function": ["name": name, "arguments": argsString],
        ])
    }
    let remainder = text.range(of: "<tool_call>").map { String(text[text.startIndex ..< $0.lowerBound]) } ?? ""
    return (calls, remainder.trimmingCharacters(in: .whitespacesAndNewlines))
}

// MARK: - logprobs

private func topKLogprobs(_ logSoftmax: MLXArray, k: Int) -> [(id: Int, logprob: Float)] {
    let V = logSoftmax.dim(-1)
    let kk = min(max(1, k), V)
    let part = argPartition(-logSoftmax, kth: kk - 1, axis: -1)[.ellipsis, 0 ..< kk]
    var vals = takeAlong(logSoftmax, part, axis: -1)
    let ord = argSort(-vals, axis: -1)
    vals = takeAlong(vals, ord, axis: -1)
    let ids = takeAlong(part, ord, axis: -1)
    eval(vals, ids)
    var result: [(Int, Float)] = []
    for i in 0 ..< kk { result.append((ids[0, i].item(Int.self), vals[0, i].item(Float.self))) }
    return result
}

// P087: `FairGate` / `ActiveCount` live in EngineServeSupport so they can be unit-tested --
// a Swift executable target cannot be imported by a test target.

/// P088 -- one request's slot in a batched decode step.
final class BatchRow {
    var caches: [KVCache]           // the request's own cache; the pool holds the rows while grouped
    var pending: Int = 0            // the token to forward next
    var length: Int = 0             // committed context length, i.e. this row's offset
    var nextToken: Int = -1         // what the batched forward sampled for this row
    /// Supplied by the request: apply its own soft-stop bias to its own logits row and sample with
    /// its own sampler. It runs on the MODEL thread while the request's own thread is blocked in
    /// `ModelThread.step`, so the two never touch it at once.
    var step: ((MLXArray) -> Int)? = nil
    /// P095 U3-A: the pieces of `step` the batched path needs separately -- the bias this row wants
    /// on `</think>` right now (0 = none), whether its sampler is greedy, and the host side effect of
    /// a sampled token -- so one argmax over the whole [B, V] block and ONE host read serve the
    /// group, instead of B slices and B round-trips.
    var biasNow: (() -> Float)? = nil
    var greedy = true
    var commit: ((Int) -> Void)? = nil
    /// P095 U3-K: the row's sampler parameters and its next draw key, for the batched sampled path
    var samplerParams: (temp: Float, topP: Float, topK: Int) = (0, 1, 0)
    var drawKey: (() -> MLXArray?)? = nil
    // P099: the row's draft state while batched, and the round's extra committed tokens for the connection thread
    var spec: BatchRowSpec? = nil
    var queued: [Int] = []
    var stopIds: Set<Int> = []
    var thinkOpenNow: (() -> Bool)? = nil
    func takeQueued() -> [Int] { let q = queued; queued = []; return q }
    /// P099 diagnostics: the row's last events (model thread + its own connection thread; racy reads are fine for a log)
    var trail: [String] = []
    func note(_ e: String) { trail.append(e); if trail.count > 14 { trail.removeFirst() } }
    /// ONE row per request, not per step: the pool's membership is object identity, and a fresh row
    /// each step would look like a membership change and re-stack the whole batch every token.
    init(caches: [KVCache]) { self.caches = caches }
}

/// P089 U10 -- EVERY MLX-TYPED VALUE ONE REQUEST OWNS.
///
/// Created, mutated and destroyed on the model thread; a connection thread holds only the integer
/// handle into `seqRegistry`. That is the whole ownership rule, in one object: if a field is here,
/// no connection thread can ever be the one that frees it.
final class SeqState {
    var sampler: EngineSampler
    var cache: [KVCache]
    var xs: MLXArray                       // the prompt; released the moment prefill is done
    let prefillIDs: [Int]
    var prefillForward: (MLXArray, MLXArray)? = nil // owner-only native batch result
    var prefillForwardBatchSize = 0
    var h26ForwardGroupID = 0, h26ForwardSlot = -1, h26ForwardDecoderCount = -1
    var h26Intervals: [[String: Int]] = []
    var h26IntervalsSeen = 0, h26DroppedIntervals = 0
    var h26B1Intervals = 0, h26NativeIntervals = 0
    func h26BeforePrefill() -> (Int, Int, Int) {
        guard h26PrefillMetadata else { return (-1, -1, 0) }
        precondition(modelThreadShared.isOwner)
        return (canonicalTrajectory.map { $0.valid ? 1 : 0 } ?? -1,
                canonicalTrajectory?.boundary ?? -1,
                hotRungs.contains { $0.length == canonicalTarget && $0.canonical != nil } ? 1 : 0)
    }
    /// A per-row consumed interval, recorded after existing eval and policy bookkeeping.
    /// Native callback membership is captured by its producer, not inferred from this body.
    func h26RecordPrefill(_ before: (Int, Int, Int), from: Int, to: Int, batch: Int,
                         nativeGroup: Int, nativeSlot: Int, nativeDecoders: Int, bodyDecoders: Int,
                         origin: HotPrefixStore.CanonicalPrefillOrigin?) {
        guard h26PrefillMetadata else { return }
        precondition(modelThreadShared.isOwner)
        let index = h26IntervalsSeen; h26IntervalsSeen += 1
        if batch == 1 { h26B1Intervals += 1 } else { h26NativeIntervals += 1 }
        guard h26Intervals.count < h26PrefillIntervalCapacity else { h26DroppedIntervals += 1; return }
        let after = h26BeforePrefill()
        h26Intervals.append([
            "index": index, "from": from, "to": to, "actual_batch": batch,
            "native_group_id": nativeGroup, "native_slot": nativeSlot,
            "native_callback_decoder_count": nativeDecoders, "body_decoder_count": bodyDecoders,
            "trajectory_valid_before": before.0, "trajectory_boundary_before": before.1,
            "trajectory_valid_after": after.0, "trajectory_boundary_after": after.1,
            "certificate_length": origin == nil ? 0 : to,
            "certificate_mtp_next_token": origin?.mtpNextToken ?? -1,
            "target_retained_before": before.2, "target_retained_after": after.2,
        ])
    }
    func h26IntervalSummary(key: Int) -> [String: Int] {
        precondition(modelThreadShared.isOwner)
        return ["schema_version": 1, "sequence_key": key, "interval_capacity": h26PrefillIntervalCapacity,
                "intervals_seen": h26IntervalsSeen, "intervals_stored": h26Intervals.count,
                "dropped_intervals": h26DroppedIntervals, "b1_intervals": h26B1Intervals,
                "native_intervals": h26NativeIntervals]
    }
    // All diagnostic fields are plain data, populated only for the bounded probe.
    var h8PrefillCalls: [[String: Int]] = []
    var h25Witness: [String: Int] = [:]
    var h25SelectedEntry: [String: Int] = [:]
    var h25SelectedEntryRungs: [[String: Int]] = []
    var h25SelectedRung: [String: Int] = [:]
    var h25CarriedRung: [String: Int] = [:]
    var h25CapturedRung: [String: Int] = [:]
    var h25Candidates: [[String: Int]] = []
    var canonicalTrajectory: HotPrefixStore.CanonicalPrefillTrajectory? = nil
    private var canonicalCarrySlot: Int? = nil
    var canonicalTarget: Int { ((prefillIDs.count - 1) / h25CanonicalWidth) * h25CanonicalWidth }
    func h25Count(_ key: String, _ amount: Int = 1) {
        if h25Metadata { h25Witness[key, default: 0] += amount }
    }
    func h25Lookup(_ ids: [Int], minLength: Int = 1, mtp: Bool, quiet: Bool = false,
                   kind: Int) -> HotPrefixStore.Hit? {
        precondition(modelThreadShared.isOwner)
        let observe: (([String: Int]) -> Void)? = h25Metadata && h25CanonicalPrefix ? { [self] candidate in
            if h25Candidates.count < 512 {
                var row = candidate; row["lookup_kind"] = kind; h25Candidates.append(row)
            } else { h25Count("candidate_overflow") }
        } : nil
        return hotStoreShared.lookup(ids, minLength: minLength, preferMTP: mtp, quiet: quiet,
                                     canonicalWidth: h25CanonicalPrefix ? h25CanonicalWidth : nil,
                                     diagnostic: observe)
    }
    /// Reuse only the selected compact fixed-head value. No Entry/rows survive this call.
    /// The required strict budget imports through contiguous, creating distinct mutable
    /// MLXArray wrappers; struct copying alone would NOT establish that separation.
    /// Before target/tail capture the only carried rung may advance via coalescing.
    func h25Adopt(_ hit: HotPrefixStore.Hit) {
        guard h25CanonicalPrefix else { return }
        guard let origin = hit.selectedRung.canonical,
              canonicalTrajectory?.resume(length: hit.length, origin: origin) == true else {
            h25Count("carry_refusals"); canonicalTrajectory?.invalidate(); invalidateHotRungs(); return
        }
        if let slot = canonicalCarrySlot {
            guard slot == 0, hotRungs.count == 1, hit.length >= hotRungs[slot].length else {
                h25Count("carry_refusals"); canonicalTrajectory?.invalidate(); invalidateHotRungs(); return
            }
            hotRungs[slot] = hit.selectedRung
            h25Count("carry_replacements")
        } else {
            guard hotRungs.isEmpty else {
                h25Count("carry_refusals"); canonicalTrajectory?.invalidate(); invalidateHotRungs(); return
            }
            hotRungs.append(hit.selectedRung); canonicalCarrySlot = 0
        }
        h25Count("carry_adoptions")
        refreshHotRungBytes()
        if h25Metadata {
            h25Witness["carried_length"] = hit.length
            h25CarriedRung = hit.selectedRung.metadata
        }
    }
    /// Called after the existing eval; one persistent target snapshot, not one per chunk.
    func h25AfterPrefill(from: Int, to: Int, batch: Int, mtp: Bool, model: Qwen4ExpModel) -> HotPrefixStore.CanonicalPrefillOrigin? {
        guard h25CanonicalPrefix else { return nil }
        let interior = to < prefillIDs.count
        let origin = canonicalTrajectory?.advance(from: from, to: to, batch: batch, interior: interior,
                                                  mtpNextToken: mtp && interior ? prefillIDs[to] : nil)
        if batch > 1 { h25Count("native_chunks") }
        else if (to - from) % h25CanonicalWidth != 0 || !interior { h25Count("irregular_chunks") }
        else if origin != nil { h25Count("regular_b1_chunks") }
        guard let origin else { return nil }
        if to == canonicalTarget && !hotRungs.contains(where: { $0.length == to && $0.canonical != nil }) {
            if let last = hotRungs.last, last.length == to {
                // The ordinary reserve capture already owns this exact evaluated state.
                hotRungs[hotRungs.count - 1] = last.withCanonical(origin)
                h25Count("target_relabels")
            } else {
                captureHotRung(length: to, model: model, canonical: origin)
                h25Count("target_captures")
            }
            if let rung = hotRungs.last, rung.length == to, rung.canonical == origin {
                h25Count("target_certifications")
                if h25Metadata { h25Witness["target_length"] = to; h25CapturedRung = rung.metadata }
            }
        }
        return origin
    }
    func h8Offsets(_ mtp: KVCache?, suffix: String) -> [String: Int] {
        guard h8BoundaryProbe else { return [:] }
        precondition(modelThreadShared.isOwner)
        var values: [String: Int] = [:]
        for (index, item) in cache.enumerated() {
            if let list = item as? CacheList, let kv = list[0] as? KVCacheSimple {
                values["trunk_L\(index)_\(suffix)"] = kv.offset
            }
        }
        values["mtp_\(suffix)"] = ((mtp as? CacheList)?[0] as? KVCacheSimple)?.offset ?? -1
        return values
    }
    func h8RecordPrefill(_ before: [String: Int], from: Int, to: Int, mtp: KVCache?, headTokens: [Int] = []) {
        guard h8BoundaryProbe else { return }
        precondition(modelThreadShared.isOwner)
        var row = before.merging(h8Offsets(mtp, suffix: "after")) { _, new in new }
        row.merge(["call_index": h8PrefillCalls.count, "from": from, "to": to,
                   "input_batch": 1, "input_sequence": to - from, "mtp": mtp == nil ? 0 : 1,
                   "after_existing_eval": 1, "is_last_prompt_call": to == prefillIDs.count ? 1 : 0,
                   "mtp_input_count": headTokens.count, "mtp_input_first": headTokens.first ?? -1,
                   "mtp_input_last": headTokens.last ?? -1, "hot_rungs_after": hotRungs.count]) { _, new in new }
        h8PrefillCalls.append(row)
    }
    var lastLogits = MLXArray(0)
    var prevLogits = MLXArray(0)
    var pending = MLXArray(0)
    /// H60: a serial spec dropped or replaced while it holds a queued chain is settled here too, so no path can leave the
    /// K-1 chain rows in a head that someone else (retiredMTPCache, the hot store) still reads.
    var spec: ServeSpecState? = nil {
        didSet {
            if var o = oldValue, o.pre != nil,
               spec.map({ ($0.mtpCache as AnyObject) !== (o.mtpCache as AnyObject) }) ?? true { o.settle() }
        }
    }
    // scratch carried between prefill chunks of the MTP prime, so a long prefill can still yield
    // the model thread between chunks instead of owning it for minutes
    var primeMTPCache: KVCache? = nil
    var primeMixed = MLXArray(0)
    var primeS = MLXArray(0)
    // P093: the hot store's rungs for this sequence (prefill end, then every rung step of decode),
    // and the MTP cache a batch handover retired -- still valid to its own length, so the stored
    // entry can arm MTP up to there.
    var hotRungs: [HotPrefixStore.Rung] = []
    private(set) var hotRungLogicalBytes = 0, hotRungRollbackLogicalBytes = 0
    private var hotRungsValid = true
    var retiredMTPCache: KVCache? = nil
    let row: BatchRow
    init(sampler: EngineSampler, cache: [KVCache], xs: MLXArray, ids: [Int]) {
        self.sampler = sampler; self.cache = cache; self.xs = xs
        self.prefillIDs = ids
        self.row = BatchRow(caches: cache)
        if h25CanonicalPrefix {
            canonicalTrajectory = HotPrefixStore.CanonicalPrefillTrajectory(
                width: h25CanonicalWidth, maxChunk: h50WidthCanonical ? h50MaxCanonicalChunk : h25CanonicalWidth)
        }
        if h25Metadata {
            h25Witness = [
                "enabled": h25CanonicalPrefix ? 1 : 0, "width": h25CanonicalWidth, "target": canonicalTarget,
                "initial_entries": -1, "initial_lookup_attempts": 0, "initial_hit_length": 0, "initial_hit_has_mtp": 0,
                "materialize_attempts": 0, "materialize_successes": 0,
                "restore_after_eval_trunk_offset": -1, "restore_after_eval_mtp_offset": -1,
                "carry_adoptions": 0, "carry_replacements": 0, "carry_refusals": 0, "carried_length": 0,
                "target_certifications": 0, "target_captures": 0, "target_relabels": 0, "target_length": 0,
                "regular_b1_chunks": 0, "native_chunks": 0, "irregular_chunks": 0, "coalesced_imports": 0,
                "candidate_overflow": 0, "final_store_committed": 0,
            ]
        }
    }
    func captureHotRung(length: Int, model: Qwen4ExpModel, canonical: HotPrefixStore.CanonicalPrefillOrigin? = nil,
                        from source: [KVCache]? = nil) {
        precondition(modelThreadShared.isOwner)
        guard hotRungsValid else { return }
        if let last = hotRungs.last, length < last.length {
            invalidateHotRungs(); return
        }
        let next = hotRungs + [hotStoreShared.captureRung(source ?? cache, length: length, model: model, canonical: canonical)]
        guard let retained = hotStoreShared.retainedLiveRungs(next) else {
            invalidateHotRungs(); return
        }
        hotRungCaptures += 1
        hotRungPruned += next.count - retained.count
        hotRungs = retained
        refreshHotRungBytes()
    }
    private func refreshHotRungBytes() {
        let retained = hotRungs
        hotRungLogicalBytes = retained.reduce(0) { $0 + $1.bytes }
        hotRungRollbackLogicalBytes = retained.reduce(0) { total, rung in
            total + rung.head.filter { $0.key.hasSuffix(".a4") || $0.key.hasSuffix(".a5") }
                .values.reduce(0) { $0 + $1.nbytes }
        }
        hotRungHighWaterPerSequence = max(hotRungHighWaterPerSequence, hotRungs.count)
    }
    func canStoreHotRungs(validTo: Int) -> Bool {
        precondition(modelThreadShared.isOwner)
        guard hotRungsValid else { return false }
        guard hotStoreShared.retainedLiveRungs(hotRungs, validTo: validTo) != nil else {
            invalidateHotRungs(); return false
        }
        return true
    }
    private func invalidateHotRungs() {
        hotRungsValid = false; hotRungs.removeAll(); hotRungInvalid += 1
        hotRungLogicalBytes = 0; hotRungRollbackLogicalBytes = 0
        FileHandle.standardError.write(Data("engine: nonmonotone hot-rung history; finished cache store disabled for this request\n".utf8))
    }
}

/// Equal-offset, equal-width chunks only. MTP priming, cache publication and cancellation
/// remain in each row's existing body. Duplicate prefixes use the coalescing path.
private func runNativePrefill(_ keys: [Int], from: Int, to: Int, model: Qwen4ExpModel) {
    precondition(modelThreadShared.isOwner)
    var states: [SeqState] = []
    for key in keys {
        let state = seqRegistry[key]
        if states.contains(where: { $0.prefillIDs.prefix(to).elementsEqual(state.prefillIDs.prefix(to)) }) { continue }
        states.append(state)
    }
    guard states.count >= 2 else { return }
    let caches = from == 0 ? model.newCache(parameters: nil) : model.stackCaches(states.map { $0.cache })
    let xs = concatenated(states.map { $0.xs[0..., from ..< to] }, axis: 0)
    if h26PrefillMetadata { h26NativeGroupSerial += 1 }
    let groupID = h26PrefillMetadata ? h26NativeGroupSerial : 0
    // Actual registered decode lifetimes at this callback, not the earlier selection.
    // The owner executes this callback with ModelThread's condition lock released.
    let callbackDecoders = h26PrefillMetadata ? modelThreadShared.decoderCount : -1
    let (logits, hidden) = model.forwardHidden(xs, cache: caches)
    // The MTP body only needs hidden/cache for an interior chunk. Final logits are
    // evaluated by the per-row body when used; do not force an unused vocabulary head.
    eval(hidden, caches.flatMap { $0.state })
    for (row, state) in states.enumerated() {
        state.cache = model.unstackRow(caches, slot: row, length: to, pooledBlocks: to / model.configuration.text.indexerCompressRatio)
        state.row.caches = state.cache
        state.prefillForward = (logits[row ..< row + 1], hidden[row ..< row + 1])
        state.prefillForwardBatchSize = states.count
        if h26PrefillMetadata {
            state.h26ForwardGroupID = groupID; state.h26ForwardSlot = row
            state.h26ForwardDecoderCount = callbackDecoders
        }
        eval(state.cache.flatMap { $0.state })
    }
    nativePrefillSizes[states.count, default: 0] += 1
}

/// Model-thread-only. No lock, because there is exactly one thread: the preconditions say so out
/// loud, so a future caller that forgets fails immediately instead of corrupting a dictionary.
final class SeqRegistry {
    private var byId: [Int: SeqState] = [:]
    private var nextId = 1
    func create(_ make: () -> SeqState) -> Int {
        precondition(modelThreadShared.isOwner, "SeqState may only be created on the model thread")
        let id = nextId; nextId += 1; byId[id] = make()
        serveOwnerOrderTrace.record("sequence_start", ["key": id, "prompt_tokens": byId[id]!.prefillIDs.count])
        return id
    }
    subscript(id: Int) -> SeqState {
        precondition(modelThreadShared.isOwner, "SeqState may only be touched on the model thread")
        return byId[id]!
    }
    func destroy(_ id: Int) {
        precondition(modelThreadShared.isOwner, "SeqState may only be destroyed on the model thread")
        byId[id] = nil
        serveOwnerOrderTrace.record("sequence_end", ["key": id])
    }
    func traceKeys() -> [Int] { precondition(modelThreadShared.isOwner); return byId.keys.sorted() }
    func traceKey(_ row: BatchRow) -> Int? { precondition(modelThreadShared.isOwner); return byId.first { $0.value.row === row }?.key }
    var count: Int { byId.count }
    func hotRungSnapshot() -> [String: Int] {
        precondition(modelThreadShared.isOwner)
        return ["live_rungs": byId.values.reduce(0) { $0 + $1.hotRungs.count },
                "live_rung_logical_bytes": byId.values.reduce(0) { $0 + $1.hotRungLogicalBytes },
                "live_rung_rollback_logical_bytes": byId.values.reduce(0) { $0 + $1.hotRungRollbackLogicalBytes },
                "live_rung_captures": hotRungCaptures, "live_rung_pruned": hotRungPruned,
                "live_rung_invalid": hotRungInvalid, "live_rung_pooled_skips": hotRungPooledSkips,
                "live_rung_pooled_captures": hotRungPooledCaptures,
                "live_rung_high_water_per_sequence": hotRungHighWaterPerSequence]
    }
}
nonisolated(unsafe) let seqRegistry = SeqRegistry()

/// Ordering only: bounded plain owner data, serialized once after the last probe sequence.
/// It deliberately adds no clock, tensor evaluation, live endpoint or per-event I/O.
private final class ServeOwnerOrderTrace {
    let path = ProcessInfo.processInfo.environment["ENGINE_OWNER_ORDER_TRACE"]
    var enabled: Bool { path != nil }
    private let buffer = ModelOwnerTraceBuffer<[String: Any]>()
    private var dispatch = 0, eventSequence = 0, generation = 0
    private var queueOverflow = 0, maxRegistryCount = 0
    private var invalid = false, flushed = false
    func keys(_ rows: [BatchRow]) -> [Int] {
        rows.map { row in
            guard let key = seqRegistry.traceKey(row) else { invalid = true; return -1 }
            return key
        }
    }
    func observe(_ event: ModelOwnerTraceEvent) {
        guard enabled, !flushed else { return }
        dispatch = event.dispatchSequence
        if event.event == "dispatch" {
            if event.queueTotal > event.queue.count { queueOverflow += 1 }
            if !event.identityValid { invalid = true }
        }
        record(event.event, event.json)
    }
    func record(_ event: String, _ fields: @autoclosure () -> [String: Any] = [:]) {
        guard enabled, !flushed else { return }
        precondition(modelThreadShared.isOwner)
        let registry = seqRegistry.traceKeys()
        maxRegistryCount = max(maxRegistryCount, registry.count)
        eventSequence += 1
        buffer.append({
            var d = fields()
            d["event"] = event; d["event_seq"] = eventSequence; d["dispatch_seq"] = dispatch
            d["registry_keys"] = registry; d["pool_generation"] = generation
            d["pool_keys"] = keys(batchPoolShared.rows); d["pool_lengths"] = batchPoolShared.lengths
            return d
        }())
    }
    func poolTransition(_ event: String, beforeKeys: [Int], beforeLengths: [Int]) {
        guard enabled, !flushed else { return }
        let before = generation; generation += 1
        record(event, ["generation_before": before, "generation_after": generation,
                       "before_keys": beforeKeys, "before_lengths": beforeLengths,
                       "after_keys": keys(batchPoolShared.rows), "after_lengths": batchPoolShared.lengths])
    }
    func rung(key: Int, requested: Int, threshold: Int, emitted: Int, verified: Int,
              outcome: String, have: Int?) {
        guard enabled, !flushed else { return }
        record("rung", ["key": key, "requested_length": requested, "next_threshold": threshold,
                        "emitted": emitted, "verified_queue_count": verified, "outcome": outcome,
                        "have": have as Any? ?? NSNull(),
                        "retained_lengths": seqRegistry[key].hotRungs.map(\.length)])
    }
    func batchResult(_ reqs: [StepRequest], K: Int, accepted: [Int: Int]) {
        guard enabled, !flushed else { return }
        let lengths = reqs.map { req -> Int in
            let row = seqRegistry[req.key].row
            if let slot = batchPoolShared.memberIds.firstIndex(of: ObjectIdentifier(row)) {
                return batchPoolShared.lengths[slot]
            }
            return (row.caches.first { $0 is CacheList } as? CacheList).map { ($0[0] as! KVCacheSimple).offset } ?? -1
        }
        record("batch_result", ["keys": reqs.map(\.key), "K": K,
               "accepted": reqs.map { accepted[$0.key] ?? 0 }, "committed_lengths": lengths,
               "queued_counts": reqs.map { seqRegistry[$0.key].row.queued.count }])
    }
    func flushIfIdle() {
        guard let path, !flushed, maxRegistryCount > 0, seqRegistry.count == 0 else { return }
        precondition(modelThreadShared.isOwner)
        let finalKeys = keys(batchPoolShared.rows)
        let result: [String: Any] = ["schema": "p106-owner-order-v1", "pid": ProcessInfo.processInfo.processIdentifier,
            "complete": !invalid && buffer.dropped == 0 && queueOverflow == 0, "instrument_invalid": invalid,
            "dropped_events": buffer.dropped, "queue_overflow": queueOverflow, "event_count": buffer.records.count,
            "max_registry_count": maxRegistryCount, "early_flush": maxRegistryCount < 8,
            "final_registry_count": seqRegistry.count, "final_pool_keys": finalKeys,
            "final_pool_generation": generation, "events": buffer.records]
        do {
            let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
            try data.write(to: URL(fileURLWithPath: path), options: [.atomic])
        } catch {
            FileHandle.standardError.write(Data("engine: owner-order trace write failed: \(error)\n".utf8))
        }
        flushed = true; buffer.removeAll()
    }
}
nonisolated(unsafe) private let serveOwnerOrderTrace = ServeOwnerOrderTrace()

/// The pooled cache the current group decodes into. Stacking costs a full copy of every row's KV, so
/// it is done when the MEMBERSHIP changes and not per step -- membership changes when a request
/// finishes or a new one joins, which is thousands of steps apart.
nonisolated(unsafe) var batchMTPDepthShared = 0
nonisolated(unsafe) var batchMTPPolicy: String = ProcessInfo.processInfo.environment["ENGINE_BATCH_MTP_POLICY"] ?? "auto"
nonisolated(unsafe) var batchMTPRounds = 0, batchMTPDrafted = 0, batchMTPAccepted = 0, batchMTPPlainSteps = 0
nonisolated(unsafe) var batchGroupsWithoutSpec = 0, batchSpecDropped = 0, batchRoundsSkipped = 0

final class BatchPool {
    var memberIds: [ObjectIdentifier] = []
    var cache: [KVCache]? = nil
    /// P106 H48: this pool was installed row-resident (its attention layers are registered markers)
    var rowResident = false
    var rows: [BatchRow] = []
    // P099: the group's draft state and the per-cycle policy (the ds4 rule made explicit: draft while the measured
    // acceptance covers the round's cost against a plain step; both costs sampled on the engine)
    var specPool: BatchSpecPool? = nil
    var specRows: [Int] = []               // the group slots that carry draft state (the pool covers exactly these)
    var policy = BatchDraftPolicy()
    func draftDepth(maxK: Int) -> Int {
        specPool == nil ? 0 : policy.depth(maxK: maxK, mode: batchMTPPolicy)
    }
    func noteRound(K: Int, ms: Double, acc: Double) { policy.noteRound(k: K, ms: ms, acceptance: acc) }
    func notePlain(ms: Double) { policy.notePlain(ms: ms) }
    /// THE POOL'S OWN length per row, advanced by the forward it ran -- never the member's copy.
    /// A member updates `row.length` before it submits its step, so a member that misses a round
    /// still carries the length it had BEFORE the pool advanced it. Unstacking on that stale number
    /// drops the last position, and the row then regenerates a token it had already emitted: the
    /// doubled words ("the Bennet the Bennet") that this cost a debugging round to find.
    var lengths: [Int] = []
    /// A member is about to leave the batched path (a forced prefix, MTP, or it is finishing) and
    /// needs its own cache back, which only exists inside the pool. Dissolving for everyone is the
    /// honest move: the next step re-stacks whoever is still there.
    func releaseIfMember(_ row: BatchRow, _ qm: Qwen4ExpModel, ratio: Int) {
        guard memberIds.contains(ObjectIdentifier(row)) else { return }
        dissolve(qm, ratio: ratio)
    }
    /// Dissolve: hand every row back the single-sequence cache it would have had alone.
    func dissolve(_ qm: Qwen4ExpModel, ratio: Int) {
        guard let pool = cache else { return }
        let traceKeys = serveOwnerOrderTrace.enabled ? serveOwnerOrderTrace.keys(rows) : []
        let traceLengths = serveOwnerOrderTrace.enabled ? lengths : []
        if rowResident {
            // P106 H48: the members' attention caches ARE the pool; hand them back at the committed
            // length (the pool's, never the member's -- see `lengths`) with fresh fixed-state slices.
            for (i, m) in rows.enumerated() {
                let n = i < lengths.count ? lengths[i] : m.length
                let (caches, aliased) = qm.unstackRowResident(pool, slot: i, length: n, own: m.caches)
                if !aliased { rowResidentAliasRepairs += 1 }
                m.caches = caches
                m.length = n
            }
            if let sp = specPool {
                for (j, i) in specRows.enumerated() {
                    let n = sp.headLens[j]
                    rows[i].spec = BatchRowSpec(mtpCache: qm.unstackHeadCacheRowResident(sp.headCache, slot: j, length: n),
                                                headLen: n, d1: sp.d1[j], Slast: sp.Slast[j ..< (j + 1)])
                }
                Qwen4ExpModel.releaseRowResident([sp.headCache])
            }
            Qwen4ExpModel.releaseRowResident(pool)
        } else {
            for (i, m) in rows.enumerated() {
                let n = i < lengths.count ? lengths[i] : m.length
                m.caches = qm.unstackRow(pool, slot: i, length: n, pooledBlocks: n / ratio)
                m.length = n
            }
            if let sp = specPool { for (j, i) in specRows.enumerated() { rows[i].spec = unstackSpecPool(qm, sp, slot: j, ratio: ratio) } }
        }
        cache = nil; rowResident = false; memberIds = []; rows = []; lengths = []; specPool = nil; specRows = []
        policy = BatchDraftPolicy()
        serveOwnerOrderTrace.poolTransition("pool_dissolve", beforeKeys: traceKeys, beforeLengths: traceLengths)
    }
}
nonisolated(unsafe) var modelThreadShared: ModelThread! = nil
nonisolated(unsafe) let batchPoolShared = BatchPool()
nonisolated(unsafe) var batchRestacks = 0
/// Host graph construction only. Completion is optional and separately reported;
/// the old P100 timer ended BEFORE eval and could not bound GPU copy cost.
nonisolated(unsafe) var batchRestackMs = 0.0
nonisolated(unsafe) var batchRestackCompletionMs = 0.0
nonisolated(unsafe) var batchRestackLogicalBytes = 0
nonisolated(unsafe) var batchRestackTransitions: [String: Int] = [:]
let batchRestackEval = ProcessInfo.processInfo.environment["ENGINE_RESTACK_EVAL"] != nil
/// P093: rows whose cache length disagreed with the length they submitted -- must stay 0.
nonisolated(unsafe) var batchLengthMismatches = 0
/// P093: decode rungs refused because the caches did not hold the counted length -- must stay 0.
nonisolated(unsafe) var hotRungMismatches = 0
/// P093: model-thread-only, like the registry.
nonisolated(unsafe) var hotStoreShared: HotPrefixStore! = nil
// P106 B32 (H42): segment id cache for chat-template encoding; nil when
// ENGINE_TEMPLATE_CACHE_MB=0 (the original applyChatTemplate call is used unchanged).
nonisolated(unsafe) var templateCacheShared: TemplateTokenCache? = nil
nonisolated(unsafe) var templateRenderShared: ChatTemplateRender? = nil
nonisolated(unsafe) var templateCacheVerify = false

/// P102: ENGINE_PREFILL_CHUNK_SHARED (default 1024) -- the prefill chunk width while other requests are in flight.
let sharedChunk: Int = Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK_SHARED"] ?? "") ?? 1024
/// P106 H50b: the adaptive width while a request prefills alone (P093: 4096). With 1024-row width-canonical
/// projections a 2048 chunk leaves the state bit-identical to 1024 chunks (probe at 16384 and 131072); 3072 and
/// 4096 still differ from layer 4 on (a width-dependent op in the first attention layer, not yet identified).
let aloneChunk: Int = Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK_ALONE"] ?? "") ?? 4096
/// P106 H57: ENGINE_PREFILL_BUDGET_CUT (default ON; 0 = off) -- with width-canonical projections, a prefill chunk never
/// straddles the QSA indexer budget, so alone chunks wider than 2048 stay state-identical to 1024 chunks.
let h57BudgetCut: Bool = ProcessInfo.processInfo.environment["ENGINE_PREFILL_BUDGET_CUT"] != "0"
/// P100: coalesced prefills -- an in-flight prefill stores its state at every chunk end and re-checks the hot store before
/// every chunk, so N identical prompts arriving together (a subagent fan-out) cost one prefill. ENGINE_HOT_COALESCE=0 restores today's behaviour.
let hotCoalesce: Bool = ProcessInfo.processInfo.environment["ENGINE_HOT_COALESCE"] != "0"
/// ENGINE_TOOLCHOICE_NONE_DROPS_TOOLS=1 restores the pre-P098 render (no tool schemas under tool_choice: none).
let toolChoiceNoneDropsTools: Bool = ProcessInfo.processInfo.environment["ENGINE_TOOLCHOICE_NONE_DROPS_TOOLS"] != nil

func batchStatsLine() -> String {
    "\(modelThreadShared.statsLine()), restacks \(batchRestacks) (\(String(format: "%.0f", batchRestackMs)) ms), length mismatches \(batchLengthMismatches), rung mismatches \(hotRungMismatches), live sequences \(seqRegistry.count); batched MTP rounds \(batchMTPRounds) drafted \(batchMTPDrafted) accepted \(batchMTPAccepted) plain-with-head \(batchMTPPlainSteps) skipped-by-policy \(batchRoundsSkipped) groups-without-spec \(batchGroupsWithoutSpec) specs-dropped \(batchSpecDropped); \(hotStoreShared.statsLine())"
}

/// One decode step for a whole group, run BY the model thread. `reqs` is what the requests queued;
/// the rows, the caches and the pool are resolved here, because this is the only thread allowed to
/// touch any of them.
/// P095: ENGINE_BATCH_SYNC_PROBE=1 -- how much of a batched step is the forward + eval, and how much the
/// per-row sampling loop after it (B host round-trips). Host-side stamps, reported every 200 steps.
let batchSyncProbe: Bool = ProcessInfo.processInfo.environment["ENGINE_BATCH_SYNC_PROBE"] != nil
/// P095 U3-A: ENGINE_BATCH_SAMPLE=rows restores the per-row sampling loop (B host round-trips).
let batchedSampling: Bool = (ProcessInfo.processInfo.environment["ENGINE_BATCH_SAMPLE"] ?? "block") != "rows"
nonisolated(unsafe) var batchSyncEvalMs = 0.0, batchSyncLoopMs = 0.0
nonisolated(unsafe) var batchSyncSteps = 0
func runBatchedStep(_ reqs: [StepRequest], model: any LanguageModel, qm: Qwen4ExpModel, ratio: Int) -> [Int] {
    if roundTraceShared.enabled {
        roundTraceShared.stamp("s_entry"); roundTraceShared.set("rows", reqs.count)
        let ph = modelThreadShared.phaseStats()
        roundTraceShared.set("dispatches", ph.values.reduce(0) { $0 + $1.dispatches })
        roundTraceShared.set("prefill_dispatches", ph["prefill"]?.dispatches ?? 0)
    }
    defer { if roundTraceShared.enabled { roundTraceShared.stamp("s_exit"); roundTraceShared.flush() } }
    var traceK = 0
    var traceAccepted: [Int: Int] = [:]
    defer { serveOwnerOrderTrace.batchResult(reqs, K: traceK, accepted: traceAccepted) }
    var incoming: [BatchRow] = []
    incoming.reserveCapacity(reqs.count)
    for r in reqs {
        let s = seqRegistry[r.key]
        s.row.pending = r.pending
        s.row.length = r.length
        // P093 (D36): `row.caches` is the AUTHORITY, never `s.cache`. When another member leaves,
        // `dissolve` hands every remaining row its freshly unstacked cache in `row.caches`, but the
        // owner's `SeqState.cache` still names the objects from BEFORE the pool was stacked. This
        // line used to read `s.row.caches = s.cache`, which reverted every survivor to that stale
        // state on its next step and re-stacked the pool from it: the tokens generated while pooled
        // were simply gone from the KV. On the 8-session 8k fleet that failed 25 of 48 turns.
        s.cache = s.row.caches
        // A row INSIDE the standing pool keeps its own caches at the pre-stack length by design (the
        // pool holds its state); the check is for rows the next line may (re)stack or step alone.
        let have = (s.row.caches.first { $0 is CacheList } as? CacheList).map { ($0[0] as! KVCacheSimple).offset } ?? -1
        if have != r.length, !batchPoolShared.memberIds.contains(ObjectIdentifier(s.row)) {
            batchLengthMismatches += 1
            if batchLengthMismatches <= 20 {
                FileHandle.standardError.write("engine: BATCH ROW LENGTH MISMATCH seq \(r.key): cache holds \(have), request claims \(r.length) (queued \(s.row.queued.count), spec \(s.row.spec == nil ? "none" : "headLen \(s.row.spec!.headLen)"), serial spec \(s.spec != nil)) trail: \(s.row.trail.joined(separator: " | "))\n".data(using: .utf8)!)
            }
        }
        incoming.append(s.row)
    }
    // The pool's slots are positional, but the queue hands the group back in arrival order, which
    // varies from step to step. Comparing the order rather than the SET made the same four members
    // look like a membership change and re-stack the whole batch on 156 of 200 steps. Re-order to
    // the pool's slots instead: same set, same slots, no copy.
    var group = incoming
    if Set(incoming.map { ObjectIdentifier($0) }) == Set(batchPoolShared.memberIds),
       incoming.count == batchPoolShared.memberIds.count {
        let byId = Dictionary(uniqueKeysWithValues: incoming.map { (ObjectIdentifier($0), $0) })
        group = batchPoolShared.memberIds.compactMap { byId[$0] }
    }
    let ids = group.map { ObjectIdentifier($0) }
    let prevMembers = Set(batchPoolShared.memberIds)
    if batchPoolShared.memberIds != ids {
        // membership changed: give the previous members their own caches back, then stack the new set.
        // A group of one with no pool standing is not a re-stack -- it is the ordinary
        // single-sequence step, and counting it as one made the instrument report 4118 re-stacks
        // that never happened.
        if batchPoolShared.cache != nil || group.count > 1 {
            let left = batchPoolShared.rows.filter { !ids.contains(ObjectIdentifier($0)) }.count
            for m in batchPoolShared.rows { m.note("dissolve B\(batchPoolShared.rows.count)->\(group.count) left \(left) len \(m.length)") }
            batchRestackTransitions["B\(batchPoolShared.rows.count)->B\(group.count)", default: 0] += 1
            let rs0 = Date()
            batchPoolShared.dissolve(qm, ratio: ratio)
            batchRestacks += 1
            batchRestackMs += Date().timeIntervalSince(rs0) * 1000
        }
        for m in group where !prevMembers.contains(ObjectIdentifier(m)) { m.note("join B\(group.count) len \(m.length) spec \(m.spec.map { String($0.headLen) } ?? "-")") }
        if group.count > 1 {
            let st0 = Date()
            defer {
                batchRestackMs += Date().timeIntervalSince(st0) * 1000
                var arrays = batchPoolShared.cache!.flatMap { $0.state }
                if let spec = batchPoolShared.specPool { arrays += spec.headCache.state + [spec.Slast] }
                batchRestackLogicalBytes += arrays.reduce(0) { $0 + $1.nbytes }
                if batchRestackEval {
                    eval(arrays)
                    batchRestackCompletionMs += Date().timeIntervalSince(st0) * 1000
                }
            }
            let rowResidentPool = rowResidentKV && Qwen4ExpModel.rowResidentEligible
            batchPoolShared.cache = rowResidentPool ? qm.stackCachesRowResident(group.map { $0.caches })
                                                    : qm.stackCaches(group.map { $0.caches })
            batchPoolShared.rowResident = rowResidentPool
            if rowResidentPool { rowResidentPoolInstalls += 1 }
            batchPoolShared.memberIds = ids
            batchPoolShared.rows = group
            batchPoolShared.lengths = group.map { $0.length }
            serveOwnerOrderTrace.poolTransition("pool_install", beforeKeys: [], beforeLengths: [])
            // `raggedPooled` -- how many blocks of row b are already pooled -- belongs to a
            // PARTICULAR set of rows. After a re-stack row b is a different sequence.
            qm.resetRaggedState()
            // P099: the rows' draft state, stacked -- only when every row has one (a row that never drafted, or whose
            // head fell out of step, cannot join a round; the whole group then decodes plain, as before P099)
            batchPoolShared.specRows = group.indices.filter { group[$0].spec != nil && group[$0].spec!.headLen == group[$0].length }
            for (i, m) in group.enumerated() where m.spec != nil && !batchPoolShared.specRows.contains(i) {
                batchSpecDropped += 1
                if batchSpecDropped <= 20 { FileHandle.standardError.write("engine: P099 spec dropped: headLen \(m.spec!.headLen) length \(m.length) trail: \(m.trail.joined(separator: " | "))\n".data(using: .utf8)!) }
                m.spec = nil; m.note("drop")
            }
            if batchMTPDepthShared > 0, qm.mtp != nil, !batchPoolShared.specRows.isEmpty {
                if rowResidentPool {
                    // A head cache that is still its request's retired handover owner is copied before the
                    // pool writes heads in place: the stacked pool never wrote it, and teardown may store it.
                    let keyOf = Dictionary(uniqueKeysWithValues: reqs.map { (ObjectIdentifier(seqRegistry[$0.key].row), $0.key) })
                    for i in batchPoolShared.specRows {
                        guard let key = keyOf[ObjectIdentifier(group[i])], let retired = seqRegistry[key].retiredMTPCache,
                              (retired as AnyObject) === (group[i].spec!.mtpCache as AnyObject) else { continue }
                        group[i].spec!.mtpCache = Qwen4ExpModel.ownedAttentionCopy(retired as! CacheList)
                        rowResidentHeadClones += 1
                    }
                    batchPoolShared.specPool = stackSpecPoolRowResident(qm, batchPoolShared.specRows.map { group[$0].spec! })
                } else {
                    batchPoolShared.specPool = stackSpecPool(qm, batchPoolShared.specRows.map { group[$0].spec! })
                }
            } else {
                batchPoolShared.specPool = nil
            }
            if batchPoolShared.specRows.count < group.count {
                batchGroupsWithoutSpec += 1
                if batchGroupsWithoutSpec <= 30 {
                    let desc = group.enumerated().map { (i, m) in "[\(i): spec \(m.spec == nil ? "none" : "headLen \(m.spec!.headLen)") length \(m.length) prev \(prevMembers.contains(ObjectIdentifier(m)))]" }.joined(separator: " ")
                    FileHandle.standardError.write("engine: P099 group without full draft state: \(desc)\n".data(using: .utf8)!)
                }
            }
            if releasePooledPrivateHistory && !rowResidentPool {
                // (Row-resident pools own no copy: the members' caches are the live history.)
                // The new trunk/spec graphs own their input arrays. Retire private trunk history
                // in BOTH aliases for this group only; departing rows keep their unstacked state.
                // Lazy graph ownership is sufficient here; no additional eval barrier is needed.
                for request in reqs {
                    let state = seqRegistry[request.key]
                    let privateCache = Qwen4ExpModel.privateTrunkHistoryPlaceholder(state.row.caches)
                    state.row.caches = privateCache
                    state.cache = privateCache
                    pooledPrivateHistoryRowsInstalled += 1
                }
                pooledPrivateHistoryPoolInstalls += 1
            }
        }
    }
    if modelThreadShared.stepCount % 200 == 0 {
        FileHandle.standardError.write("engine: \(batchStatsLine())\n".data(using: .utf8)!)
        if batchSyncProbe, batchSyncSteps > 0 {
            FileHandle.standardError.write(String(format: "engine: batch sync probe -- %d batched steps: forward+eval %.2f ms/step, sampling loop %.2f ms/step\n",
                                                  batchSyncSteps, batchSyncEvalMs / Double(batchSyncSteps), batchSyncLoopMs / Double(batchSyncSteps)).data(using: .utf8)!)
        }
    }
    if group.count > 1, let pool = batchPoolShared.cache {
        let lengths = group.map { $0.length }
        let p0 = group[0].samplerParams
        let sameSampled = batchedSampling && !group[0].greedy
            && group.allSatisfy { !$0.greedy && $0.samplerParams == p0 }
        let allGreedy = group.allSatisfy { $0.greedy }
        let fullSpec = batchPoolShared.specPool != nil && batchPoolShared.specRows.count == group.count
        let K = (fullSpec && (allGreedy || sameSampled)) ? batchPoolShared.draftDepth(maxK: batchMTPDepthShared) : 0
        if serveOwnerOrderTrace.enabled {
            traceK = K
            serveOwnerOrderTrace.record("batch_policy", ["keys": serveOwnerOrderTrace.keys(group), "K": K,
                "full_spec": fullSpec, "policy": batchMTPPolicy])
        }
        if fullSpec, K == 0 { batchRoundsSkipped += 1 }
        if K > 0, let sp = batchPoolShared.specPool {
            // P099: one batched MTP round -- K drafts per row, one [B, K+1] verify block, per-row acceptance and roll-back
            let r0 = Date()
            let rowsIn = group.map { m in BatchRoundRow(greedy: m.greedy, bias: m.biasNow?() ?? 0, thinkOpen: m.thinkOpenNow?() ?? false, stopIds: m.stopIds, drawKey: m.drawKey) }
            let out = runBatchedMTPRound(qm, caches: pool, spec: sp, pending: group.map { $0.pending }, lengths: lengths, K: K, rows: rowsIn,
                                         samplerParams: sameSampled ? (p0.temp, p0.topK, p0.topP) : nil)
            var acc = 0
            for (i, m) in group.enumerated() {
                let toks = out[i].tokens
                m.nextToken = toks[0]; m.queued = Array(toks.dropFirst())
                for t in toks { m.commit?(t) }
                batchPoolShared.lengths[i] = m.length + out[i].accepted + 1
                m.note("round K\(K) a\(out[i].accepted) len \(m.length)")
                acc += out[i].accepted
                if serveOwnerOrderTrace.enabled, let key = seqRegistry.traceKey(m) { traceAccepted[key] = out[i].accepted }
            }
            let ms = Date().timeIntervalSince(r0) * 1000
            batchPoolShared.noteRound(K: K, ms: ms, acc: Double(acc) / Double(max(1, K * group.count)))
            batchMTPRounds += 1; batchMTPDrafted += K * group.count; batchMTPAccepted += acc
            var outTok: [Int] = []
            for r in reqs { let s = seqRegistry[r.key]; s.cache = s.row.caches; outTok.append(s.row.nextToken) }
            return outTok
        }
        let y = MLXArray(group.map { Int32($0.pending) })
        let t0 = batchSyncProbe ? Date() : nil
        // H60 review: a chain queued by the previous round runs before this plain step on the GPU; finish it before the
        // policy's plain-step timer starts, so the sample measures this plain step's own cost (forward + sampling +
        // queuing the keep-alive -- whose d1 now stays on the device: H60 ABBA, plain batched step -5.5 % at R4 and R8)
        if let pq = batchPoolShared.specPool?.prequeued { eval(pq.drafts.last!) }
        let p0t = Date()
        let (lgFull, vh) = Qwen4ExpBatch.with(lengths) { qm.forwardHidden(y[0..., .newAxis], cache: pool) }
        let lg = lgFull[0..., -1, 0...]
        let t1: Date?
        if sameSampled, let toksArr = sampleTopKBlock(
                { () -> MLXArray in
                    var l = lg
                    let biases = group.map { $0.biasNow?() ?? 0 }
                    if biases.contains(where: { $0 > 0 }) {
                        let col = l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
                        l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(biases).reshaped([biases.count, 1]).asType(l.dtype)
                    }
                    return l
                }(),
                temp: p0.temp, topK: p0.topK, topP: p0.topP,
                draw: { b in group[b].drawKey?() }) {
            // P095 U3-K: sampled rows with one parameter set -- the O(V) work once over the block,
            // each row's k-wide draw with its own key, ONE host read. Token-identical to the loop.
            let toks = toksArr.asArray(Int32.self)
            t1 = batchSyncProbe ? Date() : nil
            for (i, m) in group.enumerated() {
                m.nextToken = Int(toks[i]); m.commit?(m.nextToken)
                batchPoolShared.lengths[i] = m.length + 1
            }
        } else if batchedSampling, group.allSatisfy({ $0.greedy }) {
            // P095 U3-A: one argmax over the block, one host read. Per row this is the same argmax
            // over the same values as the per-row slice (first index on a tie in both), so the
            // tokens are identical; what changes is B-1 host round-trips and the eval barrier.
            var l = lg
            let biases = group.map { $0.biasNow?() ?? 0 }
            if biases.contains(where: { $0 > 0 }) {
                let col = l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
                l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(biases).reshaped([biases.count, 1]).asType(l.dtype)
            }
            let toks = l.argMax(axis: -1).asArray(Int32.self)
            t1 = batchSyncProbe ? Date() : nil
            for (i, m) in group.enumerated() {
                m.nextToken = Int(toks[i]); m.commit?(m.nextToken)
                batchPoolShared.lengths[i] = m.length + 1        // the pool, not the member, owns this
            }
        } else {
            eval(lg)
            t1 = batchSyncProbe ? Date() : nil
            for (i, m) in group.enumerated() {
                m.nextToken = m.step!(lg[i ..< (i + 1)])
                batchPoolShared.lengths[i] = m.length + 1        // the pool, not the member, owns this
            }
        }
        if let t0, let t1 {
            // P095 sync probe: the step's GPU wall (build + eval) against the B-row sampling loop that
            // follows it, each row a host round-trip. Host stamps only; printed with the stats line.
            batchSyncEvalMs += t1.timeIntervalSince(t0) * 1000
            batchSyncLoopMs += Date().timeIntervalSince(t1) * 1000
            batchSyncSteps += 1
        }
        if let sp = batchPoolShared.specPool {
            // P099: keep the draft state of the rows that have one in step with the plain token they just committed
            let sr = batchPoolShared.specRows
            let hs = sr.count == group.count ? vh : concatenated(sr.map { vh[$0 ..< ($0 + 1)] }, axis: 0)
            batchedHeadKeepAlive(qm, spec: sp, hidden: hs, next: sr.map { group[$0].nextToken })
            if sr.count == group.count { batchPoolShared.notePlain(ms: Date().timeIntervalSince(p0t) * 1000) }
            batchMTPPlainSteps += 1
        }
    } else {
        // a group of one is the ordinary single-sequence step, on the request's own cache
        let m = group[0]
        m.note("lone len \(m.length) spec \(m.spec.map { String($0.headLen) } ?? "-")")
        if batchMTPDepthShared > 0, m.spec != nil, m.spec!.headLen == m.length {
            // P099: the row's draft state stays in step with a lone step too (one head position, serial arithmetic)
            let (lgFull, vh) = qm.forwardHidden(MLXArray([Int32(m.pending)])[.newAxis], cache: m.caches)
            m.nextToken = m.step!(lgFull[0..., -1, 0...])
            rowHeadKeepAlive(qm, spec: &m.spec!, hidden: vh, next: m.nextToken)
        } else {
            if m.spec != nil { m.spec = nil; batchSpecDropped += 1; m.note("drop-lone") }
            let lg = model(MLXArray([Int32(m.pending)])[.newAxis], cache: m.caches)[0..., -1, 0...]
            m.nextToken = m.step!(lg)
        }
    }
    // Hand every row's cache back to its sequence: the pool may have dissolved under it.
    var out: [Int] = []
    out.reserveCapacity(reqs.count)
    for r in reqs {
        let s = seqRegistry[r.key]
        s.cache = s.row.caches
        out.append(s.row.nextToken)
    }
    return out
}



/// P092: the prefill and decode rates ride inside `usage` as well as in `engine_stats`. The
/// llm_context_benchmarks harness (and LM Studio / mlx-serve clients modelled on it) reads
/// `usage.prompt_tps` / `usage.generation_tps` to separate decode from the wall clock in its
/// concurrent-request benchmark; without them it divides total tokens by a wall that includes
/// the cold prefill and reports THAT as a generation rate. OpenAI clients ignore extra keys.
private func usageObject(promptTokens: Int, completionTokens: Int, stats: [String: Any]) -> [String: Any] {
    var u: [String: Any] = ["prompt_tokens": promptTokens, "completion_tokens": completionTokens,
                            "total_tokens": promptTokens + completionTokens]
    if let p = stats["prefill_tokens_per_second"] { u["prompt_tps"] = p }
    if let g = stats["decode_tokens_per_second"] { u["generation_tps"] = g }
    return u
}

// MARK: - main handler

private func handleServeConnection(_ fd: Int32, context: ModelContext, modelId: String, defaultMaxTokens: Int, stateCache: StateCacheConfig?, mtpDepth: Int = 0, thinkBudget: Int = 0, loopGuard: Int = 0, thinkBiasMax: Float = 0, thinkBiasStart: Int = 2000, thinkBiasFull: Int = 8000, thinkBiasDeadline: Int = 0, batchMinRows: Int = 0, serveReasoningEffort: String = "medium", prefillChunk: Int = 0, keepReasoning: Bool = true) {
    let req: HTTPTransport.Request
    do { req = try HTTPTransport.readRequest(fd) }
    catch {
        let status = (error as? HTTPTransport.Failure) == .tooLarge ? 413 : ((error as? HTTPTransport.Failure) == .timeout ? 408 : 400)
        writeHTTPResponse(fd, status: status, statusText: "Request Rejected", contentType: "application/json", body: jsonErrorBody("invalid, oversized, incomplete or timed-out HTTP request"))
        return
    }
    if let key = serveBearerToken, !key.isEmpty, req.headers["authorization"] != "Bearer " + key {
        writeHTTPResponse(fd, status: 401, statusText: "Unauthorized", contentType: "application/json", body: jsonErrorBody("invalid bearer token"))
        return
    }
    var writeFailed = false
    var responseStarted = false
    var lastHeartbeat = 0.0
    func cancelled() -> Bool {
        if writeFailed || HTTPTransport.peerClosed(fd) { return true }
        let now = ProcessInfo.processInfo.systemUptime
        if responseStarted, now - lastHeartbeat >= 0.25 {
            lastHeartbeat = now
            if !HTTPTransport.heartbeat(fd) { writeFailed = true; return true }
        }
        return false
    }

    if req.method == "GET", req.path == "/v1/models" {
        let obj: [String: Any] = ["object": "list", "data": [["id": modelId, "object": "model", "created": 0, "owned_by": "engine"]]]
        writeHTTPResponse(fd, status: 200, statusText: "OK", contentType: "application/json",
                           body: (try? JSONSerialization.data(withJSONObject: obj)) ?? Data())
        return
    }
    if req.method == "GET", req.path == "/v1/engine/sessions" {
        // P106 B34 (D42): the live counters go on a LOCAL view. `serveRuntime` is embedded in
        // chat responses on other threads; mutating it here let a response serialize a torn or
        // empty dictionary. It must never be written after startup.
        var runtimeView = serveRuntime
        if let tc = templateCacheShared {
            let s = tc.stats()
            runtimeView["template_cache_entries"] = s.entries
            runtimeView["template_cache_bytes"] = s.bytes
            runtimeView["template_cache_hits"] = s.hits
            runtimeView["template_cache_misses"] = s.misses
            runtimeView["template_cache_inflight_waits"] = s.inflightWaits
            runtimeView["template_cache_evictions"] = s.evictions
            runtimeView["template_cache_segment_hits"] = s.segmentHits
            runtimeView["template_cache_segment_misses"] = s.segmentMisses
        }
        let phases = modelThreadShared.phaseStats().mapValues { s -> [String: Any] in
            ["jobs": s.jobs, "dispatches": s.dispatches, "cancelled_jobs": s.cancelledJobs,
             "queue_wait_seconds": s.queueWaitSeconds, "max_queue_wait_seconds": s.maxQueueWaitSeconds,
             "work_seconds": s.workSeconds, "max_work_seconds": s.maxWorkSeconds, "queued": s.queued]
        }
        let ps = modelThreadShared.prefillBatchStats()
        let prefillScheduling: [String: Any] = ["gathered_groups": ps.gatheredGroups,
            "callback_batches": ps.executedBatches, "filtered_jobs": ps.filteredJobs,
            "reordered_dispatches": ps.reorderedDispatches, "forced_oldest_dispatches": ps.forcedOldestDispatches]
        let obj = liveSessions.snapshot(extra: ["model": modelId, "max_concurrent": serveMaxConcurrent, "mtp_depth": mtpDepth, "runtime": runtimeView, "reserved_sequences": serveAdmission.active,
            "scheduler_phases": phases, "prefill_scheduling": prefillScheduling,
            "queue": activeRequests.snapshot(), "max_tokens_decisions": serveMaxTokensWitness.snapshot(), "reasoning_effort_decisions": serveEffortWitness.snapshot(),
            "cache_layout": serveMemoryWitness.cacheLayoutSnapshot(),
            "pooled_private_history": serveMemoryWitness.pooledPrivateHistorySnapshot(),
            "ram_budget": serveMemoryWitness.ramBudgetSnapshot(), "hot_cache": serveMemoryWitness.hotCacheSnapshot(),
            "indexer_graph_construction": serveMemoryWitness.indexerSnapshot(), "restack": serveMemoryWitness.restackSnapshot(), "memory": serveMemoryWitness.snapshot(), "native_prefill_sizes": serveMemoryWitness.prefillSnapshot(),
            "batch_steps": modelThreadShared.stepCount, "batch_sizes": Dictionary(uniqueKeysWithValues: modelThreadShared.sizeHistogram.map { (String($0.key), $0.value) })])
        writeHTTPResponse(fd, status: 200, statusText: "OK", contentType: "application/json",
                           body: (try? JSONSerialization.data(withJSONObject: obj)) ?? Data())
        return
    }
    guard req.path == "/v1/chat/completions" else {
        writeHTTPResponse(fd, status: 404, statusText: "Not Found", contentType: "application/json",
                           body: jsonErrorBody("unknown endpoint \(req.path)", type: "not_found"))
        return
    }
    guard req.method == "POST" else {
        writeHTTPResponse(fd, status: 405, statusText: "Method Not Allowed", contentType: "application/json",
                           body: jsonErrorBody("method not allowed"))
        return
    }
    // P116: wait in a bounded FIFO queue for a free slot (--queue-max), or with no queue refuse at once. A client that
    // got an immediate 503 backed off exponentially (aider: up to 4096 s) while slots sat free. The queue stays bounded:
    // past --queue-max waiting requests, or --queue-timeout-s of waiting, the answer is still a 503.
    let gate: (outcome: RequestGate.Outcome, waited: Double) = serveQueueMax > 0
        ? activeRequests.acquire(max: serveMaxConcurrent, maxWaiting: serveQueueMax, timeout: serveQueueTimeout,
                                 gone: { HTTPTransport.peerClosed(fd) })
        : (activeRequests.enter(max: serveMaxConcurrent) ? .admitted : .queueFull, 0)
    guard gate.outcome == .admitted else {
        if gate.outcome == .clientGone { return }
        writeHTTPResponse(fd, status: 503, statusText: "Service Unavailable", contentType: "application/json",
                           body: jsonErrorBody("server at capacity (\(serveMaxConcurrent) running, \(activeRequests.waiting) waiting); retry shortly", type: "overloaded"))
        return
    }
    var releaseReservation: (() -> Void)?
    var endSession: (() -> Void)?
    var requestResourcesFinished = false
    func finishRequestResources() {
        guard !requestResourcesFinished else { return }
        // Choice-scope defers destroy every SeqState on the owner first.
        // Keep an already registered session visible while owner admission
        // release waits: a disconnected client can finish locally before us.
        // Its disappearance, like successful terminal output, must follow both
        // reservation release and request-limiter release.
        releaseReservation?()
        activeRequests.leave()
        endSession?()
        requestResourcesFinished = true
    }
    defer { finishRequestResources() }
    guard let obj = try? JSONSerialization.jsonObject(with: req.body) as? [String: Any] else {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                           body: jsonErrorBody("request body is not valid JSON"))
        return
    }
    guard var chatReq = parseChatRequest(obj, defaultMaxTokens: defaultMaxTokens, defaultEffort: serveReasoningEffort, keepReasoning: keepReasoning) else {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                           body: jsonErrorBody("'messages' is required"))
        return
    }

    let toolParamTypes = toolParamTypeMap(from: chatReq.rawTools)
    var toolsForTemplate: [Message]? = nil
    // P098 (OBS-ENG-188 (6)): the tool schemas stay in the render under `tool_choice: none`. They sit at the head of
    // the system message (13.5k tokens under OMP), so dropping them re-renders the whole prompt from token 3 and no
    // stored prefix matches -- OMP's compaction handoff, sent with `tool_choice: none` at the largest context of a
    // session, read 12-31 s of TTFT for that reason (hot debug: "common 3; hit 0"). `none` still means no tool call is
    // parsed or streamed; the model just sees the same prompt it saw the turn before.
    if !chatReq.rawTools.isEmpty, !(toolChoiceNoneDropsTools && { if case .none = chatReq.toolChoice { return true } else { return false } }()) {
        toolsForTemplate = chatReq.rawTools.map(toSendableDict)
    }

    let ids: [Int]
    do {
        var extra: [String: any Sendable] = ["reasoning_effort": chatReq.reasoningEffort]
        if let pt = chatReq.preserveThinking { extra["preserve_thinking"] = pt }
        if let cache = templateCacheShared, let renderer = templateRenderShared {
            let rendered = try renderer.render(messages: chatReq.messages, tools: toolsForTemplate, extra: extra)
            let (cachedIds, _) = cache.encode(rendered: rendered)
            ids = cachedIds
            if templateCacheVerify {
                let reference = try context.tokenizer.applyChatTemplate(
                    messages: chatReq.messages, tools: toolsForTemplate, additionalContext: extra)
                if reference != ids {
                    var idx = min(reference.count, ids.count)
                    var cachedValue = -1, referenceValue = -1
                    for i in 0..<min(reference.count, ids.count) where reference[i] != ids[i] {
                        idx = i; cachedValue = ids[i]; referenceValue = reference[i]; break
                    }
                    let client = req.headers["x-engine-client"] ?? req.headers["user-agent"] ?? "unknown"
                    FileHandle.standardError.write(String(format: "engine: template-cache MISMATCH client=%@ index=%d cached=%d reference=%d\n",
                                                          client, idx, cachedValue, referenceValue).data(using: .utf8)!)
                    writeHTTPResponse(fd, status: 500, statusText: "Internal Server Error", contentType: "application/json",
                                       body: jsonErrorBody("template cache ids differ from reference", type: "template_cache_mismatch"))
                    return
                }
            }
        } else {
            ids = try context.tokenizer.applyChatTemplate(
                messages: chatReq.messages, tools: toolsForTemplate, additionalContext: extra)
        }
    } catch {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                           body: jsonErrorBody("chat template failed: \(error)"))
        return
    }

    let lookahead = max(mtpDepth, batchMTPDepthShared) + 1
    let tokensDecision = MaxTokensPolicy.effective(requested: chatReq.requestedMaxTokens, defaultTokens: defaultMaxTokens,
                                                   room: serveAdmission.maxContext - ids.count - lookahead,
                                                   clamp: serveMaxTokensClamp)
    chatReq.maxTokens = tokensDecision.tokens
    serveMaxTokensWitness.record(tokensDecision.reason)
    let effortAsked = (obj["reasoning_effort"] as? String).map { String($0.prefix(16)) } ?? "(default)"
    serveEffortWitness.record("\(effortAsked)->\(chatReq.reasoningEffort)")
    if h8BoundaryProbe && (ids.isEmpty || ids.count > 8192 || !(1...32).contains(chatReq.maxTokens)
            || chatReq.n != 1 || chatReq.temperature != 0 || chatReq.logprobsRequested
            || chatReq.stream || !chatReq.rawTools.isEmpty
            || (chatReq.thinkingBudget ?? thinkBudget) > 0 || (chatReq.loopGuard ?? loopGuard) > 0) {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                          body: jsonErrorBody("H25 probe accepts only bounded nonstream greedy text requests <=8192/max32, without tools/logprobs/hard guards"))
        return
    }
    if h26PrefillMetadata && (ids.isEmpty || ids.count > 8192 || !(1...512).contains(chatReq.maxTokens)
            || chatReq.n != 1 || chatReq.temperature != 0 || chatReq.logprobsRequested
            || !chatReq.rawTools.isEmpty
            || (chatReq.thinkingBudget ?? thinkBudget) > 0 || (chatReq.loopGuard ?? loopGuard) > 0) {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                          body: jsonErrorBody("H26 metadata accepts only bounded greedy text requests <=8192/max512, without tools/logprobs/hard guards"))
        return
    }
    guard serveAdmission.valid(prompt: ids.count, output: chatReq.maxTokens, choices: chatReq.n, lookahead: lookahead) else {
        writeHTTPResponse(fd, status: 400, statusText: "Bad Request", contentType: "application/json",
                          body: jsonErrorBody("prompt (\(ids.count)) + max_tokens (\(chatReq.maxTokens)) + lookahead (\(lookahead)) must fit max_context (\(serveAdmission.maxContext)); n must be 1...\(serveAdmission.maxChoices)"))
        return
    }
    let pendingReservation: Int?
    if serveSharedRAMBudget {
        pendingReservation = modelThreadShared.exclusive {
            let result = serveAdmission.reserve(length: ids.count + chatReq.maxTokens + lookahead) { target, growth in
                precondition(modelThreadShared.isOwner)
                let before = Double(Memory.activeMemory)
                let reclaimed = hotStoreShared.setBudgetBytes(Int(target))
                let after = Double(Memory.activeMemory)
                // Deleted entries may still have pending backend references. Refuse
                // instead of crediting presumed release. This additional guard is
                // conservative: it reserves full workspace even if some is live now.
                let fits = reclaimed && after + growth <= serveRAMLiveLimitBytes
                serveRAMReclaimWitness = ["reclaim_before_active_bytes": before, "reclaim_after_active_bytes": after,
                    "prospective_reservation_growth_bytes": growth, "reclaim_accepted": fits ? 1 : 0,
                    "live_limit_bytes": serveRAMLiveLimitBytes]
                return fits
            }
            serveRAMReclaimWitness["last_admission_committed"] = result == nil ? 0 : 1
            if result == nil {
                _ = hotStoreShared.setBudgetBytes(Int(serveAdmission.snapshot()["hot_target_bytes"]!))
            }
            serveMemoryWitness.publish()
            return result
        }
    } else { pendingReservation = serveAdmission.reserve(length: ids.count + chatReq.maxTokens + lookahead) }
    guard let reservation = pendingReservation else {
        writeHTTPResponse(fd, status: 503, statusText: "Service Unavailable", contentType: "application/json", body: jsonErrorBody("KV memory budget exhausted; retry after active requests finish", type: "overloaded"))
        return
    }
    releaseReservation = {
        if serveSharedRAMBudget {
            // Every per-choice SeqState teardown has completed on the owner before
            // the handler releases its active reservation credit.
            modelThreadShared.exclusive {
                serveAdmission.release(reservation)
                _ = hotStoreShared.setBudgetBytes(Int(serveAdmission.snapshot()["hot_target_bytes"]!))
                serveMemoryWitness.publish()
            }
        } else { serveAdmission.release(reservation) }
    }
    guard !cancelled() else { return }
    let stateCache = stateCache?.indexed(for: ids)

    // This checkpoint's generation prompt opens <think> itself (see main.swift's engineThinkOpen
    // derivation) -- the token never appears in the GENERATED ids, only the prompt's.
    let sessionId = liveSessions.begin(client: req.headers["x-engine-client"] ?? req.headers["user-agent"] ?? "unknown",
                                       promptTokens: ids.count, maxTokens: chatReq.maxTokens, stream: chatReq.stream,
                                       reasoningEffort: chatReq.reasoningEffort)
    endSession = { liveSessions.end(sessionId) }

    let thinkOpenInitially = ids.lastIndex(of: engineThinkOpenId).map { open in
        (ids.lastIndex(of: engineThinkCloseId) ?? -1) < open
    } ?? false

    var forcedPrefixIdsTemplate: [Int] = []
    switch chatReq.toolChoice {
    case .required:
        // "required" with exactly one tool available is unambiguous -- force that name outright.
        // Forcing only the bare "<function=" prefix and leaving the name to the model was measured
        // to fail under a tight token budget: the model wrote "call" as a placeholder name and then
        // dumped a raw JSON blob instead of proper <parameter> tags (caught via llmprobe's streamed
        // tool-argument reassembly check). With one candidate there is no real choice to leave open.
        if chatReq.rawTools.count == 1, let fn = chatReq.rawTools[0]["function"] as? [String: Any], let name = fn["name"] as? String {
            forcedPrefixIdsTemplate = context.tokenizer.encode(text: "<tool_call>\n<function=\(name)>\n", addSpecialTokens: false)
        } else {
            forcedPrefixIdsTemplate = context.tokenizer.encode(text: "<tool_call>\n<function=", addSpecialTokens: false)
        }
    case .named(let name):
        forcedPrefixIdsTemplate = context.tokenizer.encode(text: "<tool_call>\n<function=\(name)>\n", addSpecialTokens: false)
    default: break
    }

    let created = Int(Date().timeIntervalSince1970)
    let completionId = "chatcmpl-\(UUID().uuidString)"
    var sseHeaderSent = false
    func sendSSEHeaderIfNeeded() {
        guard !sseHeaderSent else { return }
        sseHeaderSent = true
        if !writeAll(fd, Data("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".utf8)) { writeFailed = true }
    }
    func sendSSEChunk(_ obj: [String: Any]) {
        guard let d = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: d, encoding: .utf8) else { return }
        if !writeAll(fd, Data("data: \(s)\n\n".utf8)) { writeFailed = true }
    }
    // Headers are sent after validation/admission. Newlines are legal JSON whitespace
    // and empty SSE lines; periodic nonblocking heartbeats distinguish a full disconnect
    // from a client that merely shut down its request-writing half of TCP.
    if chatReq.stream { sendSSEHeaderIfNeeded() }
    else if !writeAll(fd, Data("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n".utf8)) { writeFailed = true }
    responseStarted = true
    var isNoneToolChoice = false
    if case .none = chatReq.toolChoice { isNoneToolChoice = true }
    let toolStreamingActive = chatReq.stream && !chatReq.rawTools.isEmpty && !isNoneToolChoice

    let model = context.model
    var choicesOut: [[String: Any]] = []
    let totalPromptTokens = ids.count
    var totalCompletionTokens = 0

    var lastRequestStats: [String: Any] = [:]
    var finalStreamChunk: [String: Any]?
    for choiceIndex in 0 ..< chatReq.n {
        guard !cancelled() else { return }
        let tReq0 = Date()
        let mt = modelThreadShared!
        let qm = model as? Qwen4ExpModel
        // P058: with a state cache configured, the RUNG SPACING doubles as the prefill chunk
        // width -- otherwise an ordinary chat-length prompt (well under 4096 tokens) is one
        // chunk with no interior boundary and nothing is ever cached. The existing invariant
        // (rung spacing a multiple of the chunk width) is trivially satisfied this way.
        let fixedChunk = prefillChunk > 0 ? prefillChunk : (Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "") ?? 0)
        // P102: the shared-regime chunk width is a knob (P093 measured 1024 -> 957/981 tok/s, 2048 -> 1090/1122, 4096 -> 1160/1222;
        // the price of a wider chunk is the stall every decoding row pays per chunk)
        func chunkWidth() -> Int { fixedChunk > 0 ? fixedChunk : (activeRequests.current > 1 ? sharedChunk : aloneChunk) }
        // P093: chunk ends land on multiples of the rung spacing whatever length the prefill resumed
        // from, so the disk rungs stay where another session will look for them.
        let rungAlign = max(1, stateCache?.step ?? 1)
        // P093: the last chunk is split so a hot rung lands `hotReserve` token(s) before the end of
        // the prompt. A client that does NOT echo reasoning renders the previous turn as
        // `<think>\n\n</think>`, whose `\n\n` is one token where the generation prompt had `\n` --
        // so the next prompt's common prefix ends exactly one token short of this prompt, and a rung
        // at the exact prompt length never serves it (measured: every turn of a non-echoing client
        // fell back to the disk rung, up to 511 tokens back). One token: the split costs one
        // decode-sized forward (~20 ms); eight cost ~70 ms, the MoE touching up to 80 experts.
        let hotReserve = hotStoreShared.enabled ? 1 : 0
        // H50: the last interior 1024 boundary, where the canonical rung is certified (SeqState.canonicalTarget).
        let canonicalTarget = ((ids.count - 1) / h25CanonicalWidth) * h25CanonicalWidth
        // H57: the QSA indexer picks dense (no mask) or sparse (array mask) attention per CALL, from offset + S against its
        // budget (2048). A chunk that straddles the budget runs its first rows sparse where a 1024/2048 schedule runs them
        // dense: a different kernel, so the state differs from layer 4 on. Ending a chunk AT the budget keeps every wider
        // schedule state-identical to 1024 chunks (probe: 2048 then 4096-wide, 16384 and 131072 tokens, trunk and head).
        let qsaBudgetCut = (h50WidthCanonical && h57BudgetCut) ? (qm?.configuration.text.indexerBudget ?? 0) : 0
        func chunkEnd(_ from: Int) -> Int {
            // H57: a chunk ends on a multiple of its own width (after the budget cut at 2048 a 4096 schedule runs
            // 2048, 4096, 8192, 12288, ...), so the prefix-step rungs (every 8192) and the H54 anchors are still captured.
            let w = max(1, chunkWidth())
            let aligned = (((w > rungAlign ? ((from + w) / w) * w : from + w)) / rungAlign) * rungAlign
            var end = min(ids.count, max(aligned, from + 1))
            if qsaBudgetCut > 0, from < qsaBudgetCut, end > qsaBudgetCut { end = qsaBudgetCut }
            // H50: a wide chunk must end ON the target rather than step over it, or no certified rung is captured.
            if h25CanonicalPrefix, h50WidthCanonical, from < canonicalTarget, end > canonicalTarget { end = canonicalTarget }
            if hotReserve > 0, end == ids.count, from < ids.count - hotReserve { return ids.count - hotReserve }
            return end
        }
        // P089 U10: the sampler, the cache and the prompt are MLX-typed, so they are built on the
        // model thread and this thread keeps only `seqId`. A seeded request is reproducible because
        // the KEY is per request (P087); the old global `MLXRandom.seed` here was both redundant and
        // a cross-request side effect, so it is gone.
        let seqId: Int = mt.exclusive {
            guard !cancelled() else { return -1 }
            var sampler = EngineSampler(temp: chatReq.temperature, topP: chatReq.topP, topK: chatReq.topK)
            if !sampler.isGreedy {
                let seed: UInt64 = chatReq.seed.map { UInt64(truncatingIfNeeded: $0) } ?? UInt64.random(in: 0 ..< UInt64(UInt32.max))
                sampler.key = MLXRandom.key(seed)
            }
            let cache = model.newCache(parameters: nil)
            let xs = MLXArray(ids.map { Int32($0) })[.newAxis]
            return seqRegistry.create { SeqState(sampler: sampler, cache: cache, xs: xs, ids: ids) }
        }
        guard seqId >= 0 else { return }
        // Teardown is also MLX work: leaving the pool and dropping every array this request owns
        // both happen on the model thread, which is what makes the ownership rule complete.
        // P093: set at the end of decode to EXACTLY the tokens the caches hold, so the teardown can
        // hand the finished sequence to the hot store; nil (an error path) stores nothing.
        var hotTokens: [Int]? = nil
        defer {
            let stored: ([String: Int], [[String: Int]], Int) = mt.exclusive {
                var storedEntry: [String: Int] = [:]
                var storedRungs: [[String: Int]] = []
                var storeCommitted = 0
                // The helper's local SeqState must be released before publishing idle memory.
                func releaseSequence() {
                    let s = seqRegistry[seqId]
                    if let qm {
                        batchPoolShared.releaseIfMember(s.row, qm, ratio: qm.configuration.text.indexerCompressRatio)
                        s.cache = s.row.caches
                        if let toks = hotTokens, hotStoreShared.enabled {
                            // P099: a row that was batched keeps its LIVE head cache on the row (the retired one stops at the handover)
                            s.spec?.settle()         // H60: the stored head holds exactly the committed rows
                            let mtpc = s.spec?.mtpCache ?? s.row.spec?.mtpCache ?? s.retiredMTPCache
                            let trunkLen = (s.cache.first { $0 is CacheList } as? CacheList)
                                .map { ($0[0] as! KVCacheSimple).offset } ?? -1
                            if s.canStoreHotRungs(validTo: min(toks.count, trunkLen)) {
                                let storesBefore = hotStoreShared.stores
                                hotStoreShared.store(tokens: toks, caches: s.cache, mtp: mtpc, rungs: s.hotRungs, model: qm)
                                if h25Metadata, hotStoreShared.stores == storesBefore + 1,
                                   let metadata = hotStoreShared.completedMetadata(tokens: toks) {
                                    storedEntry = metadata.0; storedRungs = metadata.1; storeCommitted = 1
                                }
                            }
                        }
                        // P103: this request's in-flight (coalescing) entries go with it, on EVERY exit -- an error or a
                        // disconnected client used to leave them in the store for good, pinning its whole KV buffer.
                        if hotStoreShared.enabled { hotStoreShared.reapInFlight(owner: seqId) }
                    }
                    seqRegistry.destroy(seqId)
                }
                releaseSequence()
                serveOwnerOrderTrace.flushIfIdle()
                serveMemoryWitness.publish()
                return (storedEntry, storedRungs, storeCommitted)
            }
            if h25Metadata, var witness = lastRequestStats["h25_canonical"] as? [String: Any] {
                witness["stored_entry"] = stored.0; witness["stored_rungs"] = stored.1
                var counters = witness["counters"] as? [String: Int] ?? [:]
                counters["final_store_committed"] = stored.2
                witness["counters"] = counters; lastRequestStats["h25_canonical"] = witness
            }
        }
        var gc0 = 0
        var cacheHitTokens = 0
        var cacheHitSource = "none"
        // The exact-length rung (M == ids.count) is never taken: it would answer with no
        // prefill and no fresh logits, and this path has nowhere to restore them from -- so a
        // hit is accepted only strictly short of the prompt, guaranteeing the loop below always
        // runs a real final chunk.
        // P093: the hot store first -- exact lengths, no disk, and the reasoning of the previous turn
        // already in the state. The disk store is consulted only when it can beat the hot rung.
        // A hard guard may inject tokens immediately after any emitted token. Until
        // speculative commits can stop at that exact dynamic boundary, keep these
        // requests on plain decode; they remain eligible for continuous batching.
        let guardBudget = chatReq.thinkingBudget ?? thinkBudget
        let guardRepeats = chatReq.loopGuard ?? loopGuard
        let hardThinkingGuard = thinkOpenInitially && (guardBudget > 0 || guardRepeats > 0)
        let mtpEligible = mtpDepth > 0 && (qm?.mtp != nil) && !chatReq.logprobsRequested
        let wantMTP = mtpEligible && !hardThinkingGuard
        let mtpPolicyReason = mtpEligible ? (hardThinkingGuard ? "hard_guard" : "eligible") : "ineligible"
        if let qm, hotStoreShared.enabled {
            let hot: Int = mt.prefill {
                guard !cancelled() else { return 0 }
                let st = seqRegistry[seqId]
                if h25Metadata { st.h25Witness["initial_entries"] = hotStoreShared.count; st.h25Count("initial_lookup_attempts") }
                guard let h = st.h25Lookup(ids, mtp: wantMTP, kind: 0) else { return 0 }
                if h25Metadata {
                    st.h25Witness["initial_hit_length"] = h.length
                    st.h25Witness["initial_hit_has_mtp"] = h.hasMTP ? 1 : 0
                    st.h25SelectedEntry = h.entryMetadata; st.h25SelectedEntryRungs = h.entryRungMetadata
                    st.h25SelectedRung = h.selectedRung.metadata
                }
                var mtpCache: KVCache? = nil
                if wantMTP, h.hasMTP, let mtp = qm.mtp { mtpCache = mtp.newCache() }
                st.h25Count("materialize_attempts")
                guard hotStoreShared.materialize(h, into: st.cache, mtp: mtpCache, model: qm) else { return 0 }
                st.h25Count("materialize_successes")
                if let mtpCache { st.primeMTPCache = mtpCache; eval(mtpCache.state) }
                eval(st.cache.flatMap { $0.state })
                st.h25Adopt(h)
                if h25Metadata {
                    st.h25Witness["restore_after_eval_trunk_offset"] = (st.cache.first { $0 is CacheList } as? CacheList)
                        .map { ($0[0] as! KVCacheSimple).offset } ?? -1
                    st.h25Witness["restore_after_eval_mtp_offset"] = ((mtpCache as? CacheList)?[0] as? KVCacheSimple)?.offset ?? -1
                }
                // A hit at the reserve point skips the chunk that normally captures
                // it. Carry that state into this request's ladder before advancing:
                // its final store replaces the donor, and a rung at prompt length
                // cannot serve the next identical prompt (lookup is strictly shorter).
                if !h25CanonicalPrefix, hotReserve > 0, h.length == ids.count - hotReserve {
                    st.captureHotRung(length: h.length, model: qm)
                }
                return h.length
            }
            if hot > 0 { gc0 = hot; cacheHitTokens = hot; cacheHitSource = "hot" }
            if hotStoreShared.debug {
                // where does each stored sequence part from this prompt? (ENGINE_HOT_DEBUG)
                // D50: `lastDebug` is shared; another request's lookup may have replaced it since ours (the B40 H50d
                // crash: a longer prompt's common prefix sliced past this prompt). Read it only if it is still ours.
                let diag: [(Int, [Int], Int, Int, String, String)] = mt.prefill {
                    guard hotStoreShared.lastDebugQuery == ids else { return [] }
                    return hotStoreShared.lastDebug.map { d in
                        let lo = min(max(0, d.common - 6), ids.count, d.tokens.count)
                        let a = Array(d.tokens[lo ..< min(d.tokens.count, max(lo, d.common + 6))])
                        let b = Array(ids[lo ..< min(ids.count, max(lo, d.common + 6))])
                        return (d.tokens.count, d.rungs, d.common, d.mtpValidTo, context.tokenizer.decode(tokenIds: a), context.tokenizer.decode(tokenIds: b))
                    }
                }
                for (len, rungs, common, mtpTo, sa, sb) in diag {
                    FileHandle.standardError.write("engine: hot debug: prompt \(ids.count), entry \(len) rungs \(rungs) mtp<=\(mtpTo), common \(common); hit \(hot)\n  entry: \(sa.debugDescription)\n  prompt: \(sb.debugDescription)\n".data(using: .utf8)!)
                }
            }
        }
        if let sc = stateCache, let qm, let (u, M) = StateCache.lookup(sc, ids, minCount: gc0 + 1, maxCount: ids.count - 1) {
            // the load itself returns MLXArrays, so it happens on the model thread too. ONE load: the
            // MTP cache rides in the same entry, and loading it a second time for the arming below
            // used to double the disk read on every hit.
            let hit: Bool = mt.prefill {
                guard !cancelled() else { return false }
                guard let d = StateCache.load(sc, ids, M, u) else { return false }
                let st = seqRegistry[seqId]
                qm.importCaches(st.cache, prefix: "trunk.", from: d)
                st.primeMTPCache = nil
                if wantMTP, let mtp = qm.mtp, d.keys.contains(where: { $0.hasPrefix("mtp.") }) {
                    let c = mtp.newCache()
                    qm.importCaches([c], prefix: "mtp.", from: d)
                    eval(c.state)
                    st.primeMTPCache = c
                }
                eval(st.cache.flatMap { $0.state })
                return true
            }
            if hit { gc0 = M; cacheHitTokens = M; cacheHitSource = "disk" }
        }
        liveSessions.update(sessionId) { [gc0, cacheHitTokens, cacheHitSource] e in
            e.phase = "prefill"; e.cachedTokens = cacheHitTokens; e.cacheSource = cacheHitSource; e.prefilledTo = gc0
        }
        let tPrefill0 = Date()
        // P067: speculative decode when the server runs --mtp K, the head is loaded, the request wants
        // no logprobs, and (with a state cache) the hit entry carries the MTP cache -- entries written
        // by a serial server do not, and such a request decodes serially.
        //
        // P088 POLICY, and it is measured rather than assumed. MTP and batching both spend their
        // win on the same thing -- amortising one read of the expert weights -- so they compete.
        // Solo MTP runs 103 tok/s; batched decode passes that only at B = 4 (114 equal-length,
        // 93 ragged). So a request that joins a crowd forgoes drafting and joins the batch; a
        // request that is alone keeps MTP, which is the single-client experience.
        //
        // P089 CORRECTION. That policy used to disarm MTP HERE, on the CROWD SIZE ALONE. On the
        // P078 agentic fleet -- four sessions whose whole conversation sits within a few hundred
        // tokens of the 2048 indexer budget -- the four rows were almost never batchable at the
        // same moment, so all four lost drafting and gained nothing: 4600 batched steps of which
        // 4118 were a group of ONE, and 32.8 tok/s against interleaving's 84.5. Drafting is now
        // armed for every request and handed over PER STEP, at the point where this row is really
        // about to join a group (see the handover in the decode loop).
        var specActive = false
        if wantMTP, let qm {
            let armed: Bool = mt.prefill {
                guard !cancelled() else { return false }
                guard let mtp = qm.mtp else { return false }
                let st = seqRegistry[seqId]
                if cacheHitTokens > 0 {
                    // an entry written by a SERIAL server (or stored past a batch handover) carries
                    // no MTP cache at this length; such a request cannot resume a draft state and
                    // decodes serially
                    return st.primeMTPCache != nil
                }
                st.primeMTPCache = mtp.newCache()
                return true
            }
            specActive = armed
            mt.prefill { seqRegistry[seqId].row.note("arm \(armed) hit \(cacheHitSource) \(cacheHitTokens) of \(ids.count)") }
        }
        let mtpArmed = specActive           // initial per-choice arming, before handover/invalidation
        liveSessions.update(sessionId) { [specActive] e in e.mtp = specActive }
        let prefillBatchEnabled = (serveRuntime["batch_prefill"] as? Bool) == true
        func prefillJob<T: Sendable>(from: Int, to: Int, mtp: Bool, _ body: () -> T) -> T {
            // Other tail/resume geometries keep their validated solo path.
            guard prefillBatchEnabled, let qm, [512, 1024].contains(to - from) else { return mt.prefill(body) }
            return mt.prefill(key: seqId, position: from, group: "qwen:\(from):\(to):\(mtp)", tokens: to - from, prepare: {
                guard !cancelled() else { return false }
                if hotCoalesce, hotStoreShared.enabled,
                   let hit = seqRegistry[seqId].h25Lookup(ids, minLength: from + 1, mtp: mtp, quiet: true, kind: 1),
                   !mtp || hit.hasMTP { return false }
                return true
            }, batch: { keys in runNativePrefill(keys, from: from, to: to, model: qm) }, body)
        }
        if specActive, let qm {
            var n0 = 0
            var c0 = gc0
            while c0 < ids.count {
                guard !cancelled() else { return }
                let c1 = chunkEnd(c0)
                let isLast = (c1 == ids.count)
                // one chunk per job: a 128k prime yields the model thread between chunks instead of
                // owning it for two minutes (P087's fairness property, kept)
                let (tok, jump): (Int, Int) = prefillJob(from: c0, to: c1, mtp: true) {
                    defer { serveMemoryWitness.publish() }
                    guard !cancelled() else { return (0, -2) }
                    let st = seqRegistry[seqId]
                    guard let mtp = qm.mtp, let mtpCache = st.primeMTPCache else { return (0, -1) }
                    // P100: another request (a sibling subagent) may have prefilled this prefix further since our lookup
                    if st.prefillForward == nil, hotCoalesce, hotStoreShared.enabled, c0 > 0 || hotStoreShared.coalescedStores > 0,
                       let h = st.h25Lookup(ids, minLength: c0 + 1, mtp: true, quiet: true, kind: 2), h.hasMTP, h.length > c0,
                       hotStoreShared.materialize(h, into: st.cache, mtp: mtpCache, model: qm) {
                        // P103: the imported rows are LAZY slices of the donor's buffers. The pre-existing hot-hit path
                        // evals them at the hit; this one did not, so a sibling's cache carried the donor's unevaluated
                        // prefill graph across model-thread jobs -- unbounded retention for no gain.
                        eval(st.cache.flatMap { $0.state }); eval(mtpCache.state)
                        if c0 > 0 { hotStoreShared.coalescedIntoNonFresh += 1 }
                        st.h25Count("coalesced_imports"); st.h25Adopt(h)
                        if !h25CanonicalPrefix, hotReserve > 0, h.length == ids.count - hotReserve { st.captureHotRung(length: h.length, model: qm) }
                        return (0, h.length)
                    }
                    let embed = qm.model.embedTokens
                    let h8Before = st.h8Offsets(mtpCache, suffix: "before")
                    if h8BoundaryProbe { precondition(st.prefillForward == nil, "H25 B1 diagnostic cannot witness native-prefill call inputs") }
                    let actualBatch = st.prefillForward == nil ? 1 : st.prefillForwardBatchSize
                    let h26Before = st.h26BeforePrefill()
                    let nativeGroup = st.h26ForwardGroupID, nativeSlot = st.h26ForwardSlot
                    let nativeDecoders = st.h26ForwardDecoderCount
                    let bodyDecoders = h26PrefillMetadata ? mt.decoderCount : -1
                    let (lg, h) = st.prefillForward ?? qm.forwardHidden(st.xs[0..., c0 ..< c1], cache: st.cache)
                    st.prefillForward = nil; st.prefillForwardBatchSize = 0
                    st.h26ForwardGroupID = 0; st.h26ForwardSlot = -1; st.h26ForwardDecoderCount = -1
                    var toks = Array(ids[(c0 + 1) ..< min(c1 + 1, ids.count)])
                    var nn = 0
                    if isLast {
                        st.lastLogits = lg
                        nn = lg[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                        toks.append(nn)
                    }
                    let (mixed, S) = mtp(hidden: h, tokens: MLXArray(toks.map { Int32($0) })[.newAxis], embed: embed, cache: mtpCache)
                    st.primeMixed = mixed; st.primeS = S
                    if hotReserve > 0, c1 == ids.count - hotReserve { st.captureHotRung(length: c1, model: qm) }
                    // P106 B41: a rung every `prefixRungStep` tokens of a prompt prefilled from the start: the first two
                    // of a conversation are retained for its whole life, so another conversation sharing the prefix (a
                    // client's system prompt and tool list -- OMP subagents share 11.4k tokens with the main agent)
                    // resumes there instead of from zero.
                    else if !h25CanonicalPrefix, prefixRungStep > 0, c1 % prefixRungStep == 0, !isLast {
                        st.captureHotRung(length: c1, model: qm)
                    }
                    if let sc = stateCache, !isLast, c1 % sc.step == 0 {
                        eval(st.cache.flatMap { $0.state }); eval(mtpCache.state)
                        var d: [String: MLXArray] = ["T": MLXArray(Int32(c1))]
                        qm.exportCaches(st.cache, prefix: "trunk.", into: &d)
                        qm.exportCaches([mtpCache], prefix: "mtp.", into: &d)
                        StateCache.write(sc, ids, c1, d)
                    }
                    // A scheduling boundary must also bound submitted GPU work. Otherwise
                    // several lazy chunks can queue ahead of a newly ready decoder.
                    eval(S, st.cache.flatMap { $0.state }, mtpCache.state)
                    let origin = st.h25AfterPrefill(from: c0, to: c1, batch: actualBatch, mtp: true, model: qm)
                    st.h26RecordPrefill(h26Before, from: c0, to: c1, batch: actualBatch,
                                        nativeGroup: nativeGroup, nativeSlot: nativeSlot, nativeDecoders: nativeDecoders,
                                        bodyDecoders: bodyDecoders, origin: origin)
                    st.h8RecordPrefill(h8Before, from: c0, to: c1, mtp: mtpCache, headTokens: toks)
                    if hotCoalesce, hotStoreShared.enabled, !isLast, c1 < ids.count - hotReserve {
                        hotStoreShared.store(tokens: Array(ids[0 ..< c1]), caches: st.cache, mtp: mtpCache, rungs: [], model: qm, inFlight: true, owner: seqId, finalCanonical: origin)
                    }
                    return (nn, -1)
                }
                guard jump != -2 else { return }
                if jump >= 0 { c0 = jump; liveSessions.update(sessionId) { [c0] e in e.prefilledTo = c0 }; continue }
                if isLast { n0 = tok }
                c0 = c1
                liveSessions.update(sessionId) { [c0] e in e.prefilledTo = c0 }
            }
            gc0 = ids.count
            mt.prefill {
                guard !cancelled() else { return }
                let st = seqRegistry[seqId]
                guard let mtp = qm.mtp, let mtpCache = st.primeMTPCache else { return }
                let d1 = qm.draftToken(st.primeMixed[0..., -1, 0...])
                let Slast = st.primeS[0..., (st.primeS.dim(1) - 1)..., 0...]
                eval(d1, Slast)
                st.spec = ServeSpecState(model: qm, mtp: mtp, embed: qm.model.embedTokens, mtpCache: mtpCache,
                                         n0: n0, d1: d1, Slast: Slast, K: mtpDepth)
                st.primeMixed = MLXArray(0); st.primeS = MLXArray(0); st.primeMTPCache = nil
            }
        }
        while gc0 < ids.count {
            guard !cancelled() else { return }
            let gc1 = chunkEnd(gc0)
            // P087: a long prefill yields the model thread between chunks instead of owning it for
            // minutes. At 128k that is the difference between a 114 s freeze for every other client
            // and a 114 s request that the others interleave with.
            let jump: Int = prefillJob(from: gc0, to: gc1, mtp: false) {
                defer { serveMemoryWitness.publish() }
                guard !cancelled() else { return -2 }
                let st = seqRegistry[seqId]
                if st.prefillForward == nil, hotCoalesce, hotStoreShared.enabled, let qm, gc0 > 0 || hotStoreShared.coalescedStores > 0,
                   let h = st.h25Lookup(ids, minLength: gc0 + 1, mtp: false, quiet: true, kind: 3), h.length > gc0,
                   hotStoreShared.materialize(h, into: st.cache, mtp: nil, model: qm) {
                    eval(st.cache.flatMap { $0.state })                        // P103: as above
                    if gc0 > 0 { hotStoreShared.coalescedIntoNonFresh += 1 }
                    st.h25Count("coalesced_imports"); st.h25Adopt(h)
                    if !h25CanonicalPrefix, hotReserve > 0, h.length == ids.count - hotReserve { st.captureHotRung(length: h.length, model: qm) }
                    return h.length
                }
                let h8Before = st.h8Offsets(nil, suffix: "before")
                if h8BoundaryProbe { precondition(st.prefillForward == nil, "H25 B1 diagnostic cannot witness native-prefill call inputs") }
                let actualBatch = st.prefillForward == nil ? 1 : st.prefillForwardBatchSize
                let h26Before = st.h26BeforePrefill()
                let nativeGroup = st.h26ForwardGroupID, nativeSlot = st.h26ForwardSlot
                let nativeDecoders = st.h26ForwardDecoderCount
                let bodyDecoders = h26PrefillMetadata ? mt.decoderCount : -1
                st.lastLogits = st.prefillForward?.0 ?? model(st.xs[0..., gc0 ..< gc1], cache: st.cache)
                st.prefillForward = nil; st.prefillForwardBatchSize = 0
                st.h26ForwardGroupID = 0; st.h26ForwardSlot = -1; st.h26ForwardDecoderCount = -1
                if hotReserve > 0, let qm, gc1 == ids.count - hotReserve { st.captureHotRung(length: gc1, model: qm) }
                if let sc = stateCache, let qm, gc1 < ids.count, gc1 % sc.step == 0 {
                    eval(st.cache.flatMap { $0.state })
                    var d: [String: MLXArray] = ["T": MLXArray(Int32(gc1))]
                    qm.exportCaches(st.cache, prefix: "trunk.", into: &d)
                    StateCache.write(sc, ids, gc1, d)
                }
                eval(st.lastLogits, st.cache.flatMap { $0.state })
                let origin = qm.flatMap { st.h25AfterPrefill(from: gc0, to: gc1, batch: actualBatch, mtp: false, model: $0) }
                st.h26RecordPrefill(h26Before, from: gc0, to: gc1, batch: actualBatch,
                                    nativeGroup: nativeGroup, nativeSlot: nativeSlot, nativeDecoders: nativeDecoders,
                                    bodyDecoders: bodyDecoders, origin: origin)
                st.h8RecordPrefill(h8Before, from: gc0, to: gc1, mtp: nil)
                if hotCoalesce, hotStoreShared.enabled, let qm, gc1 < ids.count - hotReserve {
                    hotStoreShared.store(tokens: Array(ids[0 ..< gc1]), caches: st.cache, mtp: nil, rungs: [], model: qm, inFlight: true, owner: seqId, finalCanonical: origin)
                }
                return -1
            }
            guard jump != -2 else { return }
            gc0 = jump >= 0 ? jump : gc1
            liveSessions.update(sessionId) { [gc0] e in e.prefilledTo = gc0 }
        }
        // with MTP the first token was already taken greedily during the prime (n0); with a sampler at
        // temperature > 0 the prime's token is the argmax -- the same choice `generate --mtp` makes
        let firstTok: Int = mt.prefill {
            guard !cancelled() else { return -1 }
            let st = seqRegistry[seqId]
            // P093: the prefill-end rung -- the state at exactly the prompt, before any token is
            // forwarded. The next turn of a client that does not echo reasoning resumes here.
            if let qm, hotStoreShared.enabled { st.captureHotRung(length: ids.count, model: qm) }
            st.prevLogits = st.lastLogits[0..., -1, 0...]
            let y = st.spec.map { MLXArray([Int32($0.n0)]) } ?? st.sampler.sample(st.prevLogits)
            eval(y)
            st.pending = y
            st.lastLogits = MLXArray(0)
            st.xs = MLXArray(0)               // the prompt is done with; a 256k one is worth freeing
            return y.item(Int.self)
        }
        guard firstTok >= 0 else { return }
        // measured AFTER the first token is materialised: the graph is lazy, and timing it before the
        // eval reported the build, not the prefill (the serial path used to print 240k tok/s)
        let prefillSeconds = max(Date().timeIntervalSince(tPrefill0), 1e-6)
        let ttftSeconds = Date().timeIntervalSince(tReq0)
        let tDecode0 = Date()
        liveSessions.update(sessionId) { e in e.phase = "decode"; e.prefilledTo = e.promptTokens; if e.firstTokenAt == nil { e.firstTokenAt = tDecode0 } }

        mt.enterDecode(key: seqId)
        defer { mt.leaveDecode(key: seqId) }

        var out: [Int] = []
        var forceQueue: [Int] = (!thinkOpenInitially && !forcedPrefixIdsTemplate.isEmpty) ? forcedPrefixIdsTemplate : []
        var toolCallInjected = !forceQueue.isEmpty
        var toolBoundaryPlainSteps = 0
        var closedThink = false

        // Incremental tool-call streaming: this checkpoint writes tool calls as XML
        // (<tool_call><function=NAME><parameter=P>V</parameter>...), not raw JSON, so there is no
        // JSON to forward token-by-token. Instead: announce a call the moment its name closes, then
        // emit one `function.arguments` fragment per <parameter> as it closes (each fragment valid
        // to concatenate: `{"a":1` then `,"b":2` then `}`), which is what a real streamed tool call
        // reassembles from on the client side.
        var xmlStream = XMLToolStream()
        var toolIndex = -1
        var toolName = ""
        var toolParameters = 0
        func streamToolCallProgress(_ delta: String) {
            guard toolStreamingActive else { return }
            for event in xmlStream.append(delta) {
                var tool: [String: Any]
                switch event {
                case .open(let name):
                    toolIndex += 1; toolName = name; toolParameters = 0
                    tool = ["index": toolIndex, "id": "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24),
                            "type": "function", "function": ["name": name, "arguments": ""]]
                case .parameter(let name, let raw):
                    let value = coerceParamValue(raw, type: (toolParamTypes[toolName] ?? [:])[name])
                    let fragment = (toolParameters == 0 ? "{" : ",") + jsonKeyValueFragment(name, value)
                    toolParameters += 1
                    tool = ["index": toolIndex, "function": ["arguments": fragment]]
                case .close:
                    tool = ["index": toolIndex, "function": ["arguments": toolParameters == 0 ? "{}" : "}"]]
                }
                if chatReq.parallelToolCalls || toolIndex == 0 {
                    sendSSEChunk(["id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId,
                                  "choices": [["index": choiceIndex, "finish_reason": NSNull(), "delta": ["tool_calls": [tool]]]]])
                }
            }
        }
        var reasoningSplit = DelimiterStream(["</think>"])
        var toolSplit = DelimiterStream(toolStreamingActive ? ["<tool_call>"] : [])
        var stopSplit = DelimiterStream(chatReq.stopStrings)
        var reasoningPrevText = ""
        var contentPrevText = ""
        var reasoningFinal: String? = nil
        var logprobsContent: [[String: Any]] = []
        var finishReason = "length"

        // P077 soft stop ramp (per connection; a request may override the ceiling)
        let softBias = EngineThinkBias(start: thinkBiasStart, full: thinkBiasFull,
                                       maxBias: chatReq.thinkBiasMax ?? thinkBiasMax,
                                       deadline: chatReq.thinkBiasDeadline ?? thinkBiasDeadline)
        // P068 thinking guard state; effective settings were also used before
        // MTP cache import/priming to select the safe request policy.
        var guardWindows: [[Int]: Int] = [:]
        var guardFired: String? = nil
        var guardAt = 0
        let guardCloseIds: [Int] = (guardBudget > 0 || guardRepeats > 0) && thinkOpenInitially
            ? context.tokenizer.encode(text: "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n", addSpecialTokens: false) : []
        // P078: both decode paths run one token ahead of what has been APPENDED, so `closedThink`
        // (which is derived from the decoded text of `out`) lags the real state by a token and the
        // ramp could close the block twice -- a stray `</think>` at the head of `content`. This flag
        // is set the moment a close is KNOWN on the host: for a speculative round, when the round
        // returns its tokens; for the serial step, when the sampled id is read back.
        var closeCommitted = false
        var specQueue: [Int] = []          // verified tokens of the current round not yet emitted
        var specStats = (rounds: 0, drafted: 0, accepted: 0)
        // P093: O(1) detokenisation per token (was a whole-output decode per step: 108 -> 83 tok/s
        // over a 13k-token response, measured). Same text, byte for byte -- see the type's tests.
        var detok = IncrementalDetokenizer(window: 32) { context.tokenizer.decode(tokenIds: $0) }
        // P093: which tokens the caches hold, for the hot store. `stopForwarded` records that the
        // stop id was pushed through the model (the serial and batched steps forward the pending
        // token before they can see it is a stop; the speculative path checks first and never does).
        var stopForwarded = false
        var stopTok = -1
        let stopIdSet: Set<Int> = engineStopIdSet.union(context.tokenizer.eosTokenId.map { [$0] } ?? [])
        // A forced tool prefix changes the continuation at </think>. End a
        // speculative commit there too, leaving the delimiter as the unforwarded
        // bonus. It is a scheduling boundary, not a terminal response token.
        let speculativeStopIds = forcedPrefixIdsTemplate.isEmpty
            ? stopIdSet : stopIdSet.union([engineThinkCloseId])
        var nextHotRung = ids.count + hotStoreShared.rungStep
        // P088/P089 batching. Eligibility is decided per step: no logprobs (a batched step hands back
        // a token, not a distribution), no forced prefix, and a context past the indexer budget,
        // which is the only region where `raggedDecode` is defined at all. A DRAFTING row is a
        // candidate too -- it hands its draft state over the moment a group is really forming.
        let batchRatio = qm?.configuration.text.indexerCompressRatio ?? 4
        let batchFloor = qm?.configuration.text.indexerBudget ?? 2048
        var pendingTok = firstTok                    // the pending token as a plain Int, always known
        var pendingStale = false                     // `seq.pending` lags on the batched path
        // P089: the model thread lingers for peers only when at least `--batch-min` rows COULD join,
        // and only the request knows whether its own row could. So each one registers while it is
        // batchable and unregisters the moment it is not -- including at the end, hence the `defer`.
        var registered = false
        func setRegistered(_ on: Bool) {
            guard on != registered else { return }
            registered = on
            if on { mt.enter(key: seqId) } else { mt.leave(key: seqId) }
        }
        defer { setRegistered(false) }
        // The row's own decode policy -- its soft-stop bias and its sampler -- set ONCE, because it
        // reads boxed locals at call time. It runs on the model thread while this thread is blocked
        // inside `step`, so the two never touch `out` or `closeCommitted` at once.
        // Only plain snapshots leave the owner. A pool dissolve may mutate any row
        // while its connection drains previously committed tokens.
        var hasRowSpec = false
        mt.decode {
            seqRegistry[seqId].row.step = { lg in
                var l = lg
                if softBias.active, thinkOpenInitially, !closedThink, !closeCommitted {
                    let b = softBias.bias(out.count)
                    if b > 0 {
                        let col = l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
                        l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(b).asType(l.dtype)
                    }
                }
                let tok = seqRegistry[seqId].sampler.sample(l).item(Int.self)
                if tok == engineThinkCloseId { closeCommitted = true }
                return tok
            }
            seqRegistry[seqId].row.biasNow = { (softBias.active && thinkOpenInitially && !closedThink && !closeCommitted) ? softBias.bias(out.count) : 0 }
            seqRegistry[seqId].row.greedy = seqRegistry[seqId].sampler.isGreedy
            seqRegistry[seqId].row.commit = { tok in if tok == engineThinkCloseId { closeCommitted = true } }
            let sp = seqRegistry[seqId].sampler
            seqRegistry[seqId].row.samplerParams = (sp.temp, sp.topP, sp.topK)
            seqRegistry[seqId].row.drawKey = { seqRegistry[seqId].sampler.drawKey() }
            seqRegistry[seqId].row.stopIds = speculativeStopIds
            seqRegistry[seqId].row.thinkOpenNow = { thinkOpenInitially && !closedThink && !closeCommitted }
        }
        while out.count < chatReq.maxTokens {
            guard !cancelled() else { liveSessions.update(sessionId) { $0.finishReason = "cancelled" }; return }
            // P093: a decode rung for the hot store every `rungStep` committed tokens. The caches
            // hold ids + out + the verified-but-unemitted queue (the last queue element is the
            // round's bonus, NOT yet forwarded, and the pending token stands in for it -- so the
            // count is exactly ids + out + queue). Skipped while the row is pooled: its state lives
            // in the pool then, and the final rung is captured after the pool releases it.
            let inCache = ids.count + out.count + specQueue.count
            if hotStoreShared.enabled, let qm, inCache >= nextHotRung {
                mt.decode(traceLabel: "hot_rung", traceKey: seqId) {
                    let st = seqRegistry[seqId]
                    guard !batchPoolShared.memberIds.contains(ObjectIdentifier(st.row)) else {
                        let slot = batchPoolShared.memberIds.firstIndex(of: ObjectIdentifier(st.row))!
                        // P106 B41: a row-resident member is captured from the pool in place (its rows are its own
                        // buffers, its fixed state is slot `slot` of the stacked arrays), when the pool's committed
                        // length is exactly the counted one. Stacked pools and any disagreement still skip.
                        if pooledHotRungs, batchPoolShared.rowResident, let pool = batchPoolShared.cache,
                           slot < batchPoolShared.lengths.count, batchPoolShared.lengths[slot] == inCache {
                            let traceCaptures = hotRungCaptures
                            st.captureHotRung(length: inCache, model: qm, from: Qwen4ExpModel.rowResidentView(pool, slot: slot))
                            if hotRungCaptures > traceCaptures { hotRungPooledCaptures += 1 }
                            serveOwnerOrderTrace.rung(key: seqId, requested: inCache, threshold: nextHotRung,
                                emitted: out.count, verified: specQueue.count,
                                outcome: hotRungCaptures > traceCaptures ? "pooled_captured" : "capture_rejected", have: inCache)
                            return
                        }
                        hotRungPooledSkips += 1
                        serveOwnerOrderTrace.rung(key: seqId, requested: inCache, threshold: nextHotRung,
                            emitted: out.count, verified: specQueue.count, outcome: "pooled_skip", have: batchPoolShared.lengths[slot])
                        return
                    }
                    st.cache = st.row.caches                 // fresh after any dissolve (D36)
                    // the rung is only right if the caches hold exactly `inCache` tokens; a
                    // disagreement means the bookkeeping above is wrong, and the rung is not taken
                    let have = (st.cache.first { $0 is CacheList } as? CacheList).map { ($0[0] as! KVCacheSimple).offset } ?? -1
                    guard have == inCache else {
                        hotRungMismatches += 1
                        serveOwnerOrderTrace.rung(key: seqId, requested: inCache, threshold: nextHotRung,
                            emitted: out.count, verified: specQueue.count, outcome: "length_mismatch", have: have)
                        if hotRungMismatches <= 20 { FileHandle.standardError.write("engine: HOT RUNG LENGTH MISMATCH seq \(seqId): cache holds \(have), loop counts \(inCache)\n".data(using: .utf8)!) }
                        return
                    }
                    let traceCaptures = hotRungCaptures
                    st.captureHotRung(length: inCache, model: qm)
                    serveOwnerOrderTrace.rung(key: seqId, requested: inCache, threshold: nextHotRung,
                        emitted: out.count, verified: specQueue.count,
                        outcome: hotRungCaptures > traceCaptures ? "captured" : "capture_rejected", have: have)
                }
                nextHotRung = inCache + hotStoreShared.rungStep
            }
            // Batchability of THIS row, recomputed every step because the context grows past the
            // indexer budget mid-request. It deliberately ignores drafting: a drafting row still
            // counts as a candidate, because the next lines are where it stops drafting.
            // The delimiter was left unforwarded by the speculative stop set.
            // Forward it once through the plain path before the host injects the
            // tool prefix: a new MTP round here would commit past the boundary.
            let toolBoundary = pendingTok == engineThinkCloseId
                && !toolCallInjected && !forcedPrefixIdsTemplate.isEmpty
            let rowBatchable = batchMinRows > 0 && forceQueue.isEmpty && !toolBoundary && !chatReq.logprobsRequested
                && (ids.count + out.count) > batchFloor && qm != nil
            setRegistered(rowBatchable)
            // The MTP -> batch handover, at the only safe point: with the round's verified tokens all
            // emitted, the pending token is the one not yet in the cache -- exactly the serial
            // invariant the batched step assumes. Mid-round the cache is ahead of it, and dropping
            // the draft state there would silently replay positions.
            if rowBatchable, specActive, specQueue.isEmpty, mt.steppableCount >= batchMinRows {
                hasRowSpec = mt.decode(traceLabel: "handover", traceKey: seqId) {
                    let st = seqRegistry[seqId]
                    st.spec?.settle()                // H60: a queued serial chain never crosses into the batch
                    st.retiredMTPCache = st.spec?.mtpCache
                    // P099: the draft state moves to the row (the head cache holds exactly the committed length here)
                    if batchMTPDepthShared > 0, let sp = st.spec, (ids.count + out.count) == ((sp.mtpCache as! CacheList)[0] as! KVCacheSimple).offset {
                        st.row.spec = BatchRowSpec(mtpCache: sp.mtpCache, headLen: ids.count + out.count, d1: sp.d1.item(Int.self), Slast: sp.Slast)
                        st.row.note("handover len \(ids.count + out.count)")
                    } else {
                        st.row.note("handover NOSPEC len \(ids.count + out.count) head \(st.spec.map { String(($0.mtpCache as! CacheList)[0].offset) } ?? "-")")
                    }
                    st.spec = nil
                    return st.row.spec != nil
                }
                specActive = false
            }
            let canBatch = rowBatchable && !specActive
            let t: Int
            var stepLogprobEntry: [String: Any]? = nil
            var forwardedNow = false
            if canBatch {
                // `pendingTok` is the only form of the pending token this thread may hold, and it is
                // the authority on every path -- the MLXArray form lives on the model thread and is
                // rebuilt from this number when a serial step needs it. Keeping the Int authoritative
                // is what makes the three paths interchangeable step by step.
                t = pendingTok
                if stopIdSet.contains(t) {
                    // P099: a pending stop is not stepped (a batched round would forward it as the block's first row
                    // and draft past it; the plain step wasted a row on it) -- the stop check below ends the request
                } else if !specQueue.isEmpty {
                    // P099: a batched round committed more than one token for this row; emit them before stepping again
                    pendingTok = specQueue.removeFirst()
                } else {
                    if batchMTPDepthShared > 0, hasRowSpec, mt.steppableCount < batchMinRows {
                        // P099: below the batch minimum the owner never lingers for company and the rows fragment into
                        // B1/B2 plain steps -- a row with live draft state goes back to the lone request's MTP round
                        // BEFORE it steps (trunk at ids+out, head at the same length, the pending token not yet in the cache)
                        let rearmed: Bool = mt.decode(traceLabel: "prearm", traceKey: seqId) {
                            let st = seqRegistry[seqId]
                            guard let qm, let mtp = qm.mtp, let rs = st.row.spec, rs.headLen == ids.count + out.count, !batchPoolShared.memberIds.contains(ObjectIdentifier(st.row)) else {
                                st.row.note("prearm FAIL head \(st.row.spec.map { String($0.headLen) } ?? "-") len \(ids.count + out.count) member \(batchPoolShared.memberIds.contains(ObjectIdentifier(st.row)))"); return false }
                            st.cache = st.row.caches                 // fresh after any dissolve (D36): the round runs on THESE
                            st.spec = ServeSpecState(model: qm, mtp: mtp, embed: qm.model.embedTokens, mtpCache: rs.mtpCache, n0: pendingTok,
                                                     d1: MLXArray([Int32(rs.d1)]), Slast: rs.Slast, K: mtpDepth)
                            st.row.spec = nil
                            st.row.note("prearm len \(ids.count + out.count)")
                            return true
                        }
                        if rearmed { specActive = true; hasRowSpec = false; continue }
                    }
                    let step = mt.stepResult(key: seqId, pending: t, length: ids.count + out.count, cancelled: cancelled)
                    guard !step.cancelled else { return }
                    pendingTok = step.token
                    hasRowSpec = step.hasSpec
                    forwardedNow = true
                    if batchMTPDepthShared > 0 {
                        let q = step.queued
                        if !q.isEmpty { specQueue = q; if q.contains(engineThinkCloseId) { closeCommitted = true } }
                        else if hasRowSpec, mt.steppableCount < batchMinRows {   // alone: no batch to fragment
                            // P099: alone again with a live draft state -- back to the lone request's MTP round
                            let rearmed: Bool = mt.decode(traceLabel: "rearm", traceKey: seqId) {
                                let st = seqRegistry[seqId]
                                guard let qm, let mtp = qm.mtp, let rs = st.row.spec, rs.headLen == ids.count + out.count + 1, !batchPoolShared.memberIds.contains(ObjectIdentifier(st.row)) else {
                                    st.row.note("rearm FAIL head \(st.row.spec.map { String($0.headLen) } ?? "-") len \(ids.count + out.count) member \(batchPoolShared.memberIds.contains(ObjectIdentifier(st.row)))"); return false }
                                st.row.note("rearm len \(ids.count + out.count)")
                                st.cache = st.row.caches                 // fresh after any dissolve (D36): the serial rounds run on THESE, not the pre-stack copies
                                st.spec = ServeSpecState(model: qm, mtp: mtp, embed: qm.model.embedTokens, mtpCache: rs.mtpCache, n0: pendingTok,
                                                         d1: MLXArray([Int32(rs.d1)]), Slast: rs.Slast, K: mtpDepth)
                                st.row.spec = nil
                                return true
                            }
                            if rearmed { specActive = true; hasRowSpec = false }
                        }
                    }
                }
                pendingStale = true
            } else if specActive, forceQueue.isEmpty, !toolBoundary {
                // P067 speculative path. The pending token is always the last emitted one, not yet
                // forwarded -- the serial loop's invariant too, so switching to serial is free.
                // P093: a pending stop is checked BEFORE a round would forward it as the round's
                // first row -- that round was pure waste, and it left the stop and a few drafts
                // after it in the caches, which is what kept the final state from being exact.
                if specQueue.isEmpty, !stopIdSet.contains(pendingTok) {
                    let r: (tokens: [Int], drafted: Int, accepted: Int) = mt.decode(traceLabel: "serial_mtp", traceKey: seqId) {
                        defer { serveMemoryWitness.publish() }
                        guard !cancelled() else { return ([], -1, 0) }
                        let st = seqRegistry[seqId]
                        guard var sp = st.spec else { return ([], 0, 0) }
                        let rr = serveSpecRound(&sp, cache: st.cache, sampler: &st.sampler, bias: softBias,
                                                generated: out.count,
                                                thinkOpen: thinkOpenInitially && !closedThink && !closeCommitted,
                                                stopIds: speculativeStopIds)
                        st.spec = sp
                        return rr
                    }
                    guard r.drafted >= 0 else { return }
                    specQueue = r.tokens; specStats.rounds += 1; specStats.drafted += r.drafted; specStats.accepted += r.accepted
                    if specQueue.contains(engineThinkCloseId) { closeCommitted = true }
                    if specQueue.isEmpty { specActive = false; continue }     // no draft state: serial from here
                }
                t = pendingTok
                if !specQueue.isEmpty { pendingTok = specQueue.removeFirst() }
                pendingStale = true
            } else {
                let r: (t: Int, next: Int, lp: Data?) = mt.decode(traceLabel: "serial_plain", traceKey: seqId) {
                    defer { serveMemoryWitness.publish() }
                    guard !cancelled() else { return (-1, -1, nil) }
                    let st = seqRegistry[seqId]
                    if let qm { batchPoolShared.releaseIfMember(st.row, qm, ratio: batchRatio); st.cache = st.row.caches }
                    if pendingStale { st.pending = MLXArray([Int32(pendingTok)]) }
                    st.spec?.settle()                   // H60 review: the head may alias retiredMTPCache (stored at release)
                    st.spec = nil                       // forced prefix or ineligible: serial from here
                    st.row.spec = nil                   // no stale pre-boundary head may rearm later
                    forwardedNow = true
                    var logits = model(st.pending[.newAxis], cache: st.cache)[0..., -1, 0...]
                    // P077 soft stop: while the reasoning is open, add a ramped bias to `</think>` so
                    // the model closes at ITS OWN next sentence boundary instead of being truncated
                    // mid-derivation.
                    if softBias.active, thinkOpenInitially, !closedThink, !closeCommitted, forceQueue.isEmpty {
                        let b = softBias.bias(out.count)
                        if b > 0 {
                            let col = logits[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
                            logits[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(b).asType(logits.dtype)
                        }
                    }
                    let nextArr = forceQueue.isEmpty ? st.sampler.sample(logits) : MLXArray([Int32(forceQueue.removeFirst())])
                    asyncEval(nextArr)
                    let nextTok = nextArr.item(Int.self)
                    if softBias.active, !closeCommitted, nextTok == engineThinkCloseId { closeCommitted = true }
                    let tt = pendingTok
                    let stop = engineStopIdSet.contains(tt) || (context.tokenizer.eosTokenId.map { tt == $0 } ?? false)
                    var entry: [String: Any]? = nil
                    if chatReq.logprobsRequested, !stop {
                        let lp = st.prevLogits.asType(.float32)
                        let logSoftmax = lp - lp.logSumExp(axis: -1, keepDims: true)
                        let tokenLogprob = logSoftmax[0, tt].item(Float.self)
                        var e: [String: Any] = ["token": context.tokenizer.convertIdToToken(tt) ?? "", "logprob": tokenLogprob, "bytes": NSNull()]
                        if chatReq.topLogprobsCount > 0 {
                            e["top_logprobs"] = topKLogprobs(logSoftmax, k: chatReq.topLogprobsCount).map {
                                ["token": context.tokenizer.convertIdToToken($0.id) ?? "", "logprob": $0.logprob, "bytes": NSNull()] as [String: Any]
                            }
                        }
                        entry = e
                    }
                    st.prevLogits = logits
                    st.pending = nextArr
                    return (tt, nextTok, entry.flatMap { try? JSONSerialization.data(withJSONObject: $0) })
                }
                guard r.t >= 0 else { return }
                t = r.t
                pendingTok = r.next
                pendingStale = false
                stepLogprobEntry = r.lp.flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
                specActive = false
                hasRowSpec = false
                if toolBoundary { toolBoundaryPlainSteps += 1 }
                specQueue.removeAll()
            }
            let isStopId = stopIdSet.contains(t)

            // The stop token itself (<|im_end|> etc.) must never reach `content`/`reasoning_content` --
            // decode(tokenIds:) keeps special tokens as literal text (needed so <think>/</think> stay
            // visible for the reasoning split below), so it has to be excluded here, before decoding,
            // rather than trimmed off the string afterwards.
            if isStopId {
                finishReason = "stop"
                stopForwarded = forwardedNow                          // serial / batched steps forwarded it
                stopTok = t
                break
            }
            if let e = stepLogprobEntry { logprobsContent.append(e) }

            // Everything below is host-side: no MLX, no model thread held.
            out.append(t)
            liveSessions.tokenEmitted(sessionId)
            let textDelta = detok.appendDelta(t)
            if thinkOpenInitially, !closedThink, guardFired == nil, forceQueue.isEmpty, !guardCloseIds.isEmpty {
                if guardBudget > 0, out.count >= guardBudget { guardFired = "budget" }
                else if guardRepeats > 0, out.count >= 32 {
                    let w = Array(out.suffix(32))
                    let c = (guardWindows[w] ?? 0) + 1
                    guardWindows[w] = c
                    if c >= guardRepeats { guardFired = "loop" }
                }
                if guardFired != nil { guardAt = out.count; forceQueue = guardCloseIds }
            }
            var reasoningDelta = ""
            var contentInput = textDelta
            if thinkOpenInitially, !closedThink {
                let split = reasoningSplit.append(textDelta)
                reasoningDelta = split.text
                reasoningPrevText.append(reasoningDelta)
                contentInput = split.remainder
                if reasoningSplit.matched != nil {
                    closedThink = true
                    reasoningFinal = reasoningPrevText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if forceQueue.isEmpty, !toolCallInjected, !forcedPrefixIdsTemplate.isEmpty {
                        forceQueue = forcedPrefixIdsTemplate; toolCallInjected = true
                    }
                }
            }
            // Detect a stop BEFORE emitting any part of it, including across token boundaries.
            let safeContent = stopSplit.append(contentInput).text
            contentPrevText.append(safeContent)
            let plain = toolSplit.append(safeContent)
            if chatReq.stream {
                sendSSEHeaderIfNeeded()
                for (channel, delta) in [("reasoning_content", reasoningDelta), ("content", plain.text)] where !delta.isEmpty {
                    var choice: [String: Any] = ["index": choiceIndex, "delta": [channel: delta], "finish_reason": NSNull()]
                    if let e = stepLogprobEntry { choice["logprobs"] = ["content": [e]] }
                    sendSSEChunk(["id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId, "choices": [choice]])
                }
                if toolSplit.matched != nil { streamToolCallProgress(plain.remainder) }
            }
            if stopSplit.matched != nil { finishReason = "stop"; break }
        }
        // Flush incomplete UTF-8 before the delimiter buffers, including literal U+FFFD.
        let finalText = detok.finishDelta()
        // A length/EOS boundary may leave a partial delimiter; it is ordinary text.
        if thinkOpenInitially, !closedThink {
            let tail = reasoningSplit.append(finalText).text + reasoningSplit.finish()
            reasoningPrevText.append(tail)
            if chatReq.stream, !tail.isEmpty {
                sendSSEHeaderIfNeeded()
                sendSSEChunk(["id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId,
                              "choices": [["index": choiceIndex, "delta": ["reasoning_content": tail], "finish_reason": NSNull()]]])
            }
        } else {
            let tail = stopSplit.append(finalText).text + stopSplit.finish()
            if stopSplit.matched != nil { finishReason = "stop" }
            contentPrevText.append(tail)
            let plain = toolSplit.append(tail)
            let finalDelta = plain.text + toolSplit.finish()
            if chatReq.stream {
                if !finalDelta.isEmpty {
                    sendSSEHeaderIfNeeded()
                    sendSSEChunk(["id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId,
                                  "choices": [["index": choiceIndex, "delta": ["content": finalDelta], "finish_reason": NSNull()]]])
                }
                if toolSplit.matched != nil { streamToolCallProgress(plain.remainder) }
            }
        }

        // P093: exactly what the caches hold now (see `inCache` above): the emitted tokens, plus the
        // forwarded stop on the serial/batched paths, plus a speculative round's verified-but-unemitted
        // tokens on a length or stop-string exit.
        // P099: a batched round that forwarded a pending stop also committed tokens after it (the queue; its last
        // element is the round's bonus, never forwarded) -- the claim counts the stop AND those.
        let queuedInCache: [Int] = specQueue.isEmpty ? [] : [pendingTok] + specQueue.dropLast()
        if stopForwarded { hotTokens = ids + out + [stopTok] + queuedInCache }
        else { hotTokens = ids + out + queuedInCache }
        let decodeSeconds = max(Date().timeIntervalSince(tDecode0), 1e-6)
        lastRequestStats = [
            "ttft_ms": ttftSeconds * 1000,
            "runtime": serveRuntime,
            "batch_stats": mt.exclusive { batchStatsLine() },
            "prefill_tokens_per_second": Double(max(1, ids.count - cacheHitTokens)) / prefillSeconds,
            "decode_tokens_per_second": Double(max(1, out.count)) / decodeSeconds,
            "cache_hit_tokens": cacheHitTokens,
            "cache_hit_source": cacheHitSource,
            "cache_total_tokens": ids.count,
            "peak_gpu_memory_bytes": mt.exclusive { Memory.peakMemory },
            "mtp_policy_reason": mtpPolicyReason,
            "mtp_requested": mtpDepth > 0,
            "mtp_eligible": wantMTP,
            "mtp_armed": mtpArmed,
            "tool_boundary_plain_steps": toolBoundaryPlainSteps,
            "mtp_depth": specStats.rounds > 0 ? mtpDepth : 0,
            "mtp_rounds": specStats.rounds,
            "mtp_drafted": specStats.drafted,
            "mtp_accepted": specStats.accepted,
            "think_guard": guardFired ?? "",
            "think_guard_at": guardAt,
        ]
        if ProcessInfo.processInfo.environment["ENGINE_EXPOSE_TOKEN_IDS"] == "1" { lastRequestStats["token_ids"] = out }
        if h25Metadata {
            let diagnostic = mt.exclusive {
                let st = seqRegistry[seqId]
                return (st.h8PrefillCalls, st.h25Witness, st.h25SelectedEntry, st.h25SelectedEntryRungs,
                        st.h25SelectedRung, st.h25CarriedRung, st.h25CapturedRung, st.h25Candidates,
                        st.hotRungs.map { $0.metadata }, st.h26IntervalSummary(key: seqId), st.h26Intervals)
            }
            if h8BoundaryProbe {
                lastRequestStats["h8_prefill_prompt_ids"] = ids
                lastRequestStats["h8_prefill_calls"] = diagnostic.0
                lastRequestStats["h8_prefill_witness_scope"] = "actual B1 model-call boundaries recorded after existing consumer eval; not kernel dispatch counts or timing"
            }
            if h26PrefillMetadata {
                lastRequestStats["h26_prompt_ids"] = ids
                var routes: [String: Any] = diagnostic.9.mapValues { $0 as Any }
                routes["intervals"] = diagnostic.10
                lastRequestStats["h26_prefill"] = routes
            }
            lastRequestStats["h25_canonical"] = [
                "counters": diagnostic.1, "selected_entry": diagnostic.2, "selected_entry_rungs": diagnostic.3,
                "selected_rung": diagnostic.4, "carried_rung": diagnostic.5, "captured_rung": diagnostic.6,
                "lookup_candidates": diagnostic.7, "rungs_before_store": diagnostic.8,
                "stored_entry": [String: Int](), "stored_rungs": [[String: Int]](),
            ] as [String: Any]
        }
        if thinkOpenInitially, !closedThink { reasoningFinal = reasoningPrevText.trimmingCharacters(in: .whitespacesAndNewlines) }
        var finalContent = contentPrevText.trimmingCharacters(in: .whitespacesAndNewlines)
        totalCompletionTokens += out.count

        var toolCalls: [[String: Any]] = []
        if case .none = chatReq.toolChoice {} else if !chatReq.rawTools.isEmpty {
            let (calls, remainder) = extractToolCalls(from: finalContent, toolParamTypes: toolParamTypes)
            if !calls.isEmpty {
                toolCalls = chatReq.parallelToolCalls ? calls : Array(calls.prefix(1))
                finalContent = remainder
                finishReason = "tool_calls"
            }
        }

        liveSessions.update(sessionId) { [finishReason, specStats] e in
            e.finishReason = finishReason; e.mtpDrafted += specStats.drafted; e.mtpAccepted += specStats.accepted
        }
        if chatReq.stream {
            // Tool calls were already streamed incrementally above (streamToolCallProgress) when
            // toolStreamingActive; repeating the full array here would duplicate every argument.
            let terminalChunk: [String: Any] = [
                "id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId,
                "choices": [["index": choiceIndex, "delta": [String: Any](), "finish_reason": finishReason]],
            ]
            if choiceIndex == chatReq.n - 1 { finalStreamChunk = terminalChunk }
            else { sendSSEChunk(terminalChunk) }
        } else {
            var message: [String: Any] = ["role": "assistant"]
            message["content"] = (toolCalls.isEmpty && finalContent.isEmpty) ? NSNull() as Any : finalContent
            if let r = reasoningFinal, !r.isEmpty { message["reasoning_content"] = r }
            if !toolCalls.isEmpty { message["tool_calls"] = toolCalls }
            var choice: [String: Any] = ["index": choiceIndex, "message": message, "finish_reason": finishReason]
            if chatReq.logprobsRequested { choice["logprobs"] = ["content": logprobsContent] }
            choicesOut.append(choice)
        }
    }

    // The final choice's MLX teardown has now run. Publish completion only after
    // both reservation and request capacity have been released, including for
    // clients that stop reading at finish_reason instead of waiting for [DONE].
    finishRequestResources()
    if chatReq.stream {
        sendSSEHeaderIfNeeded()
        if let finalStreamChunk { sendSSEChunk(finalStreamChunk) }
        if chatReq.streamIncludeUsage {
            sendSSEChunk([
                "id": completionId, "object": "chat.completion.chunk", "created": created, "model": modelId, "choices": [],
                "usage": usageObject(promptTokens: totalPromptTokens, completionTokens: totalCompletionTokens, stats: lastRequestStats),
                "engine_stats": lastRequestStats,
            ])
        }
        writeAll(fd, Data("data: [DONE]\n\n".utf8))
        return
    }

    let respObj: [String: Any] = [
        "id": completionId, "object": "chat.completion", "created": created, "model": modelId,
        "choices": choicesOut,
        "usage": usageObject(promptTokens: totalPromptTokens, completionTokens: totalCompletionTokens, stats: lastRequestStats),
        "engine_stats": lastRequestStats,
    ]
    writeAll(fd, (try? JSONSerialization.data(withJSONObject: respObj)) ?? Data())
}


// MARK: - P067: speculative rounds for the server (the `generate --mtp K` round, without its diagnostics)

/// Everything one request's speculative decode carries between rounds.
struct ServeSpecState {
    let model: Qwen4ExpModel
    let mtp: Qwen4ExpMTP
    let embed: Embedding
    let mtpCache: KVCache
    var n0: Int              // the last emitted token; the next round forwards it first
    var d1: MLXArray         // the head's first draft for the round
    var Slast: MLXArray      // the head's stream at the last committed position
    let K: Int
    /// P106 H60: the next round's drafts d2..dK, queued behind this round's re-prime (the batched round's pre-queue for
    /// the B1 path). The chain appended K-1 rows to the head cache past the committed length; `settle()` removes them
    /// with the round's own cleanup before anything else reads the head (handover, store), so every other reader sees
    /// exactly B44's state. ENGINE_SERIAL_PREQUEUE_DRAFTS=0 restores drafting at round entry.
    var pre: [MLXArray]? = nil
    /// H60 review: a queued chain that is never consumed (the request's last round) must not have grown a head buffer,
    /// or the stored head's capacity (charged by the hot store) would differ from B44's. Queue only when the K-1 rows fit.
    func chainFitsWithoutGrowth() -> Bool {
        let kv = (mtpCache as! CacheList)[0] as! KVCacheSimple
        let idxc = (mtpCache as! CacheList)[1] as! ArraysCache
        let reach = kv.offset + K - 1
        guard let keys = kv.rawKeys, keys.dim(2) >= reach else { return false }
        if let raw = idxc[0], raw.dim(1) < reach { return false }
        guard let pooled = idxc[1] else { return false }
        return pooled.dim(1) >= reach / model.configuration.text.indexerCompressRatio
    }
    mutating func settle() {
        guard let pre else { return }
        self.pre = nil
        let kv = (mtpCache as! CacheList)[0] as! KVCacheSimple
        _ = kv.trim(pre.count)
        let idxc = (mtpCache as! CacheList)[1] as! ArraysCache
        let T = kv.offset
        if !Qwen4ExpModel.indexerPooledBuffer, let p = idxc[1] { let r = model.configuration.text.indexerCompressRatio; if p.dim(1) > T / r { idxc[1] = p[0..., 0 ..< (T / r), 0...] } }
    }
}
let serialPrequeueDrafts: Bool = ProcessInfo.processInfo.environment["ENGINE_SERIAL_PREQUEUE_DRAFTS"] != "0"

/// One speculative round: K-1 further drafts from the head, one verify forward of [n0]+drafts on the
/// trunk, accept the longest agreeing prefix plus the bonus token, roll the trunk tape back over the
/// rejected rows, re-prime the head on the accepted rows. Mirrors speculativeLoop in main.swift.
/// Returns the newly verified tokens in order; the state is updated in place.
func serveSpecRound(_ sp: inout ServeSpecState, cache: [KVCache], sampler: inout EngineSampler,
                    bias: EngineThinkBias = EngineThinkBias(), generated: Int = 0, thinkOpen: Bool = false,
                    stopIds: Set<Int> = []) -> (tokens: [Int], drafted: Int, accepted: Int) {
    let model = sp.model, mtp = sp.mtp, embed = sp.embed, K = sp.K
    var draftArrs: [MLXArray] = [sp.d1]
    if let pre = sp.pre, pre.count == K - 1 {
        draftArrs += pre                                    // H60: queued by the previous round
    } else {
        sp.settle()
        var Slast = sp.Slast
        for _ in 1 ..< K {
            let (m, s2) = mtp(hidden: Slast, tokens: draftArrs.last![.newAxis], embed: embed, cache: sp.mtpCache)
            draftArrs.append(model.draftToken(m[0..., -1, 0...]))
            Slast = s2
        }
        asyncEval(draftArrs.last!)
    }
    sp.pre = nil
    let T = (cache.first { $0 is CacheList } as! CacheList)[0].offset      // committed length before the round
    let block = concatenated([MLXArray([Int32(sp.n0)])] + draftArrs, axis: 0)[.newAxis]
    let (vl, vh) = model.forwardHidden(block, cache: cache)
    // P078: the soft stop must ride the SPECULATIVE path too -- row j of the verify block continues a
    // prefix of `generated + j` tokens, so it gets the bias serial decoding would apply there. Without
    // this the server's ramp was inert whenever MTP was on, which is every agentic deployment.
    var vlb = vl[0]
    if thinkOpen, bias.active {
        let rows = vlb.dim(0)
        var bs = [Float](repeating: 0, count: rows)
        for j in 0 ..< rows { bs[j] = bias.bias(generated + j) }
        if bs.contains(where: { $0 > 0 }) {
            let col = vlb[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
            vlb[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(bs).reshaped([rows, 1]).asType(vlb.dtype)
        }
    }
    let predsArr = (sampler.isGreedy ? vlb.argMax(axis: -1) : sampler.sample(vlb.reshaped(-1, vlb.dim(-1)))).asType(.int32)
    let both = concatenated([predsArr, concatenated(draftArrs, axis: 0)], axis: 0)
    let bothHost = both.asArray(Int32.self).map { Int($0) }
    let preds = Array(bothHost.prefix(K + 1)), drafts = Array(bothHost.suffix(K))
    var a = 0
    while a < K && preds[a] == drafts[a] { a += 1 }
    // a committed `</think>` ends the round: the rows after it were biased as if still thinking --
    // P094: only when a bias actually reached those rows; at a zero ramp the cut is one wasted round
    if thinkOpen, bias.active,
       let ci = (0 ..< a).first(where: { drafts[$0] == engineThinkCloseId }),
       ((ci + 1) ... K).contains(where: { bias.bias(generated + $0) > 0 }) { a = ci }
    // P093: so does a committed stop id -- the tokens after it would never be emitted, and rolling
    // them back here (they become the bonus, never forwarded) is what makes the final cache state
    // exactly the emitted sequence, which the hot prefix store depends on.
    if let si = (0 ..< a).first(where: { stopIds.contains(drafts[$0]) }) { a = si }
    let bonus = preds[a]
    let newTokens = Array(drafts.prefix(a)) + [bonus]
    // P106 H60 (H56 for the serial round): the head re-prime reads only the head cache and the verify stream, so it is
    // queued FIRST and the trunk roll-back graph (36 GDN tape replays) is built while the GPU runs it -- the same
    // operations on the same inputs, only their construction order changes. ENGINE_SERIAL_REPRIME_FIRST=0 restores it.
    if !serialReprimeFirst, a < K { model.rollback(cache, blockRows: K + 1, keep: a + 1) }
    let mtpKV = (sp.mtpCache as! CacheList)[0] as! KVCacheSimple
    let extra = mtpKV.offset - T
    if extra > 0 { _ = mtpKV.trim(extra) }
    let idxc = (sp.mtpCache as! CacheList)[1] as! ArraysCache
    if !Qwen4ExpModel.indexerPooledBuffer, let p = idxc[1] { let r = model.configuration.text.indexerCompressRatio; if p.dim(1) > T / r { idxc[1] = p[0..., 0 ..< (T / r), 0...] } }
    let nextTokens = MLXArray(newTokens.map { Int32($0) })[.newAxis]
    let (m2, s3) = mtp(hidden: vh[0..., 0 ..< (a + 1), 0...], tokens: nextTokens, embed: embed, cache: sp.mtpCache)
    sp.d1 = model.draftToken(m2[0..., -1, 0...])
    asyncEval(sp.d1)
    if serialReprimeFirst, a < K { model.rollback(cache, blockRows: K + 1, keep: a + 1) }
    sp.Slast = s3[0..., (s3.dim(1) - 1)..., 0...]
    sp.n0 = bonus
    if serialPrequeueDrafts, serialReprimeFirst, K > 1, sp.chainFitsWithoutGrowth() {
        var next: [MLXArray] = [sp.d1]
        var Sl = sp.Slast
        for _ in 1 ..< K {
            let (m, s2) = mtp(hidden: Sl, tokens: next.last![.newAxis], embed: embed, cache: sp.mtpCache)
            next.append(model.draftToken(m[0..., -1, 0...]))
            Sl = s2
        }
        asyncEval(next.last!)
        sp.pre = Array(next.dropFirst())
    }
    return (newTokens, K, a)
}
let serialReprimeFirst: Bool = ProcessInfo.processInfo.environment["ENGINE_SERIAL_REPRIME_FIRST"] != "0"

/// P067: throwaway prefill + serial steps (+ prime and two rounds with MTP) so the first client never
/// pays a kernel compile. Constant token ids, nothing kept.
func serveWarmUp(context: ModelContext, mtpDepth: Int) {
    let model = context.model
    // 1024 tokens: the prefill kernels are shape-templated, and a 64-token warm left the first real
    // chat prompt paying ~2.5 s of compile (P067 A/B: serial TTFT 3.5 s at 1k on the first request).
    let ids = (0 ..< 1024).map { Int32(100 + ($0 &* 7919) % 20_000) }
    let cache = model.newCache(parameters: nil)
    let xs = MLXArray(ids)[.newAxis]
    var logits = model(xs, cache: cache)[0..., -1, 0...]
    var pending = logits.argMax(axis: -1)
    for _ in 0 ..< 4 {
        logits = model(pending[.newAxis], cache: cache)[0..., -1, 0...]
        pending = logits.argMax(axis: -1)
    }
    eval(pending)
    if let list = cache.first(where: { $0 is CacheList }) as? CacheList,
       let kv = list[0] as? KVCacheSimple, let indexer = list[1] as? ArraysCache {
        let describe: (MLXArray) -> [String: Any] = { ["shape": $0.shape, "dtype": String(describing: $0.dtype), "nbytes": $0.nbytes] }
        serveRuntime["warm_cache_layout"] = ["offset": kv.offset, "kv": kv.state.map(describe),
            "kv_capacity": [kv.rawKeys, kv.rawValues].compactMap { $0 }.map(describe),
            "indexer": indexer.state.map(describe)]
    }
    guard mtpDepth > 0, let qm = model as? Qwen4ExpModel, let mtp = qm.mtp else { return }
    let cache2 = qm.newCache(parameters: nil)
    let mtpCache = mtp.newCache()
    let (lg, h) = qm.forwardHidden(xs, cache: cache2)
    let n0 = lg[0..., -1, 0...].argMax(axis: -1).item(Int.self)
    let toks = Array(ids.dropFirst().map { Int($0) }) + [n0]
    let (mixed, S) = mtp(hidden: h, tokens: MLXArray(toks.map { Int32($0) })[.newAxis], embed: qm.model.embedTokens, cache: mtpCache)
    var sp = ServeSpecState(model: qm, mtp: mtp, embed: qm.model.embedTokens, mtpCache: mtpCache, n0: n0,
                            d1: qm.draftToken(mixed[0..., -1, 0...]), Slast: S[0..., (S.dim(1) - 1)..., 0...], K: mtpDepth)
    eval(sp.d1, sp.Slast)
    var sampler = EngineSampler(temp: 0, topP: 1, topK: 0)
    for _ in 0 ..< 2 { _ = serveSpecRound(&sp, cache: cache2, sampler: &sampler) }
    eval(sp.d1)
}

// H20 diagnostic appendix to Serve.swift, not a standalone target.
// Calls the actual owner registry/pool/runBatchedStep. No HTTP/gather or production policy change.
func h20FixedServeSchedule(_ o: Options, model qm: Qwen4ExpModel, corpus: [Int],
                          corpusSHA256: String) throws {
    let B = try o.int("--batch", 8)
    let prompt = try o.int("--prompt-tokens", 8192)
    let chunk = try o.int("--prefill-step", 1024)
    let dispatchCount = try o.int("--decode", 128)
    let K = try o.int("--mtp", 3)
    let env = ProcessInfo.processInfo.environment
    guard B == 8, prompt == 8192, chunk == 1024, dispatchCount == 128, K == 3,
          qm.mtp != nil, qm.configuration.text.indexerCompressRatio == 4,
          (env["ENGINE_BATCH_MTP_POLICY"] == "always" && batchMTPPolicy == "always")
            || (BatchDraftPolicy.cycle(batchMTPPolicy) != nil && env["ENGINE_BATCH_MTP_POLICY"] == batchMTPPolicy),
          ["0", "1"].contains(env["ENGINE_RELEASE_POOLED_PRIVATE_HISTORY"] ?? ""),
          ["0", "1"].contains(env["ENGINE_ROW_RESIDENT_KV"] ?? ""),
          env["ENGINE_RESTACK_EVAL"] == nil, env["ENGINE_INDEXER_ROUTE_WITNESS"] == nil,
          env["ENGINE_OWNER_ORDER_TRACE"] == nil, env["ENGINE_IDX_REPLAY_DUMP"] == nil,
          env["ENGINE_IDX_REPLAY_BUDGETS"] == nil,
          !batchRestackEval, !Qwen4ExpIndexerRouteWitness.enabled,
          Qwen4ExpCacheCapacity.kvStep == 256, Qwen4ExpCacheCapacity.indexerStep == 1024 else {
        throw EngineError.invalid("H20 requires B8/prompt8192/C1024/128dispatches/MTP3/always (or a cycle:K,... policy), explicit release0|1 and row-resident0|1, E9 cache steps256/1024, diagnostic traces/eval OFF")
    }
    let lengths = (0..<B).map { prompt + 4 * $0 }
    let starts = (0..<B).map { chunk * $0 }
    guard corpus.count >= starts.last! + lengths.last! else { throw EngineError.invalid("H20 corpus expansion incomplete") }
    let rows = (0..<B).map { Array(corpus[starts[$0]..<(starts[$0] + lengths[$0])]) }
    let owner = ModelThread(minBatch: 4, gatherWindow: 0.025, maxBatch: 8) { _ in
        preconditionFailure("H20 uses explicit owner calls, never the scheduler runBatch callback")
    }
    modelThreadShared = owner
    defer { owner.shutdown() }
    // ModelThread.exclusive is the existing serial owner API (there is no perform API).
    let result: (String, String?) = owner.exclusive {
        do { return (try h20FixedServeScheduleOnOwner(qm, rows: rows, starts: starts,
                                                     chunk: chunk, corpusSHA256: corpusSHA256), nil) }
        catch { return ("", String(describing: error)) }
    }
    if let error = result.1 { throw EngineError.invalid(error) }
    print(result.0)
}

private func h20FixedServeScheduleOnOwner(_ qm: Qwen4ExpModel, rows: [[Int]], starts: [Int],
                                        chunk: Int, corpusSHA256: String) throws -> String {
    func need(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError.invalid("H20 INSTRUMENTFAIL: " + message) }
    }
    try need(modelThreadShared.isOwner && seqRegistry.count == 0 && batchPoolShared.cache == nil, "owner/fresh process")
    let requested = ProcessInfo.processInfo.environment["ENGINE_RELEASE_POOLED_PRIVATE_HISTORY"] == "1"
    try need(requested == releasePooledPrivateHistory, "release request/readback")
    let ratio = qm.configuration.text.indexerCompressRatio
    Qwen4ExpCacheCapacity.configureGrowthLimit(262144)
    batchMTPDepthShared = 3
    // Direct calls keep ModelThread.stepCount=0; existing stats logging therefore executes.
    // Initialize its real dependency instead of altering runBatchedStep's logging branch.
    hotStoreShared = HotPrefixStore(capBytes: 20_000_000_000, rungStep: 512)
    var keys: [Int] = []
    defer {
        batchPoolShared.dissolve(qm, ratio: ratio)
        for key in keys { seqRegistry[key].cache = seqRegistry[key].row.caches; seqRegistry.destroy(key) }
        hotStoreShared = nil
        Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
    }
    func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func tokenDigest(_ tokens: [Int]) -> String {
        tokens.map { Int32($0).littleEndian }.withUnsafeBytes { digest(Data($0)) }
    }
    func logicalArray(_ name: String, _ array: MLXArray) -> [String: Any] {
        // Hash the logical view's exact stored dtype/bytes; never capacity padding or raw values in JSON.
        let data = array.asData(access: .copy).data
        return ["name": name, "shape": array.shape, "dtype": String(describing: array.dtype),
                "logical_bytes": array.nbytes, "sha256": digest(data)]
    }
    func cacheDigest(_ caches: [KVCache], expectedLength: Int) throws -> [[String: Any]] {
        try caches.enumerated().map { layer, cache in
            var arrays: [[String: Any]] = []
            if let list = cache as? CacheList {
                let kv = list[0] as! KVCacheSimple, idx = list[1] as! ArraysCache
                try need(kv.offset == expectedLength && kv.state.count == 2 && idx[0] != nil && idx[1] != nil && idx[2] == nil,
                         "logical text attention state at layer \(layer)")
                try need(idx[0]!.dim(1) >= expectedLength && idx[1]!.dim(1) >= expectedLength / ratio, "logical indexer extent")
                arrays.append(logicalArray("key", kv.state[0]))
                arrays.append(logicalArray("value", kv.state[1]))
                arrays.append(logicalArray("indexer_raw", idx[0]![0..., 0..<expectedLength, 0...]))
                arrays.append(logicalArray("indexer_pooled", idx[1]![0..., 0..<(expectedLength / ratio), 0...]))
                return ["layer": layer, "kind": "attention", "cache_offset": cache.offset,
                        "kv_offset": kv.offset, "indexer_offset": idx.offset, "arrays": arrays]
            }
            guard let a = cache as? ArraysCache else { throw EngineError.invalid("H20 unknown cache type") }
            // Only stable fixed state is a value oracle. Verify/prefill scratch is not logically retained
            // decode state; record its layout/presence without hashing/evaluating or claiming identity of padding.
            for slot in 0..<4 { if let x = a[slot] { arrays.append(logicalArray("slot\(slot)", x)) } }
            var transient: [[String: Any]] = []
            func layout(_ name: String, _ x: MLXArray) {
                transient.append(["name": name, "shape": x.shape, "dtype": String(describing: x.dtype)])
            }
            for slot in 4..<6 { if let x = a[slot] { layout("slot\(slot)", x) } }
            if let pair = a.rollbackState { layout("rollback_conv", pair.0); layout("rollback_ssm", pair.1) }
            for (i, pair) in a.rollbackCheckpoints.enumerated() {
                layout("checkpoint\(i)_conv", pair.0); layout("checkpoint\(i)_ssm", pair.1)
            }
            var tape: [String: Int] = [:]
            if let t = a.prefixReplayTape {
                tape = ["row_count": t.rowCount, "conv_state_rows": t.convStateRows]
                for (name, x) in [("conv", t.convInput), ("q", t.q), ("k", t.k), ("v", t.v),
                                  ("a", t.a), ("b", t.b), ("g", t.g), ("beta", t.beta)] { layout("tape_" + name, x) }
                if let x = t.ssmPre { layout("tape_ssm_pre", x) }
                if let x = t.mask { layout("tape_mask", x) }
            }
            try need(!arrays.isEmpty, "empty fixed state at layer \(layer)")
            return ["layer": layer, "kind": "fixed", "cache_offset": a.offset,
                    "rollback_checkpoints": a.rollbackCheckpoints.count, "tape": tape, "transient_layout_only": transient, "arrays": arrays]
        }
    }
    func kvOffset(_ caches: [KVCache]) -> Int {
        let first = caches.first { $0 is CacheList } as! CacheList
        return (first[0] as! KVCacheSimple).offset
    }
    func hasPrivateHistory(_ caches: [KVCache]) -> Bool {
        let attention = caches.compactMap { $0 as? CacheList }
        return !attention.isEmpty && attention.allSatisfy {
            let kv = $0[0] as! KVCacheSimple, idx = $0[1] as! ArraysCache
            return kv.rawKeys != nil && kv.rawValues != nil && idx[0] != nil && idx[1] != nil
        }
    }
    func isPlaceholder(_ caches: [KVCache]) -> Bool {
        let attention = caches.compactMap { $0 as? CacheList }
        return !attention.isEmpty && attention.allSatisfy {
            let kv = $0[0] as! KVCacheSimple, idx = $0[1] as! ArraysCache
            return kv.rawKeys == nil && kv.rawValues == nil && idx[0] == nil && idx[1] == nil && idx[2] == nil
        }
    }
    func sameCacheObjects(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { ($0.0 as AnyObject) === ($0.1 as AnyObject) }
    }
    func memoryGuard() throws {
        try need(Double(Memory.activeMemory + Memory.cacheMemory) + 48e9 <= Double(ProcessInfo.processInfo.physicalMemory) * 0.90,
                 "90% physical +48GB workspace safety bound")
    }
    let storageBefore = Memory.activeMemory
    let storage = qm.prepareServingStorage()
    let storageAfter = Memory.activeMemory
    try memoryGuard()
    var pending: [Int] = [], committed = rows.map(\.count), streams: [[Int]] = []
    var retiredHeadOwners: [ObjectIdentifier] = []
    var seed: [[String: Any]] = []
    // Same fresh causal prefill/prime geometry in each A1/B/A2 process. No restored/cold H8 comparison.
    for row in 0..<8 {
        FileHandle.standardError.write(Data("H20 seed row=\(row) length=\(rows[row].count) begin\n".utf8))
        qm.resetRaggedState(); qm.resetHeadRaggedState()
        let cache = qm.newCache(parameters: nil)
        let prime = primeRowSpec(qm, ids: rows[row], cache: cache, chunk: chunk)
        let key = seqRegistry.create { SeqState(sampler: EngineSampler(), cache: cache, xs: MLXArray(0), ids: rows[row]) }
        keys.append(key)
        let s = seqRegistry[key]
        s.row.spec = prime.spec
        s.retiredMTPCache = prime.spec.mtpCache // mirrors real post-handover retained head owner; H19 never changes it
        retiredHeadOwners.append(ObjectIdentifier(prime.spec.mtpCache as AnyObject))
        s.row.length = committed[row]; s.row.pending = prime.n0
        s.row.step = { $0.argMax(axis: -1).item(Int.self) }
        s.row.greedy = true; s.row.biasNow = { 0 }; s.row.thinkOpenNow = { false }; s.row.stopIds = []
        pending.append(prime.n0); streams.append([prime.n0])
        seed.append(["row": row, "length": committed[row], "pending": prime.n0,
            "head_length": prime.spec.headLen, "d1": prime.spec.d1,
            "trunk": try cacheDigest(cache, expectedLength: committed[row]),
            "head": try cacheDigest([prime.spec.mtpCache], expectedLength: committed[row]),
            "slast": logicalArray("Slast", prime.spec.Slast)])
        try memoryGuard()
        FileHandle.standardError.write(Data("H20 seed row=\(row) ready\n".utf8))
    }
    let keyToRow = Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($0.element, $0.offset) })
    func poolRows() throws -> [Int] {
        try batchPoolShared.rows.map { row in
            guard let key = seqRegistry.traceKey(row), let id = keyToRow[key] else { throw EngineError.invalid("H20 unknown pool row") }
            return id
        }
    }
    func orderedMembers(_ dispatch: Int) -> [Int] {
        let base: [Int]
        switch dispatch {
        case 0..<32, 64..<96, 104..<128: base = Array(0..<8)
        case 32..<48: base = Array(0..<7)
        case 48..<64: base = Array(0..<4)
        default: base = [0] // dispatch96...103: real plain B1 + head keep-alive
        }
        let shift = dispatch % base.count
        let rotated = Array(base[shift...]) + Array(base[..<shift])
        return dispatch % 2 == 0 ? rotated : Array(rotated.reversed())
    }
    let schedule = (0..<128).map(orderedMembers)
    let scheduleData = try JSONSerialization.data(withJSONObject: schedule, options: [.sortedKeys])
    let installsBefore = (rows: pooledPrivateHistoryRowsInstalled, pools: pooledPrivateHistoryPoolInstalls)
    let rrBefore = (repairs: rowResidentAliasRepairs, clones: rowResidentHeadClones)
    var records: [[String: Any]] = []
    var staleLengthTransitions = 0, retiredPreservedChecks = 0, placeholderChecks = 0, privateChecks = 0, rowResidentChecks = 0
    let rowResidentMode = rowResidentKV && Qwen4ExpModel.rowResidentEligible
    var selectedRows = 0, mtpDispatches = 0, plainDispatches = 0, acceptedPositive = 0, rollbackRows = 0
    var otherKDispatches = 0, multiRowPlainDispatches = 0
    let prequeueBefore = (taken: roundPrequeueTaken, discarded: roundPrequeueDiscarded)
    // No checkpoint/hash/eval between first stack and final dissolve. This is deliberate lazy-ownership coverage.
    for dispatch in 0..<128 {
        let members = schedule[dispatch]
        let beforePoolRows = try poolRows(), beforePoolLengths = batchPoolShared.lengths
        let beforeRowLengths = keys.map { seqRegistry[$0].row.length }
        let lengthIn = members.map { committed[$0] }, pendingIn = members.map { pending[$0] }
        let changed = Set(beforePoolRows) != Set(members)
        if changed && !beforePoolRows.isEmpty {
            if zip(beforePoolRows, beforePoolLengths).contains(where: { beforeRowLengths[$0.0] != $0.1 }) { staleLengthTransitions += 1 }
        }
        let roundsBefore = batchMTPRounds, plainHeadsBefore = batchMTPPlainSteps, draftedBefore = batchMTPDrafted
        let requests = members.map { StepRequest(key: keys[$0], pending: pending[$0], length: committed[$0]) }
        let first = runBatchedStep(requests, model: qm, qm: qm, ratio: ratio)
        try need(first.count == members.count, "returned token count")
        let afterPoolRows = try poolRows()
        let afterPoolLengths = batchPoolShared.lengths
        // H60 audit: the K the round really drafted (drafted / rows), not a constant -- `always` still records 3, as B44 did
        let executedK = batchMTPRounds == roundsBefore + 1 ? (batchMTPDrafted - draftedBefore) / max(1, members.count) : 0
        if executedK > 0 && executedK != 3 { otherKDispatches += 1 }
        if executedK == 0 && members.count > 1 { multiRowPlainDispatches += 1 }
        // H60: under a `cycle:` gate policy a multi-row dispatch may be a plain step or a K2 round on purpose (the pre-queue
        // discard gate); the fixed branch is asserted only for `always`. A K2 round still satisfies tokens <= executedK + 1.
        if BatchDraftPolicy.cycle(batchMTPPolicy) == nil {
            try need(executedK == (members.count > 1 ? 3 : 0), "fixed always-K3/plain-B1 branch")
        }
        if executedK > 0 { mtpDispatches += 1 } else { plainDispatches += 1 }
        var outputs: [[Int]] = [], queues: [[Int]] = [], headLens: [Int] = [], drafts: [Int] = [], accepted: [Int] = []
        for (i, row) in members.enumerated() {
            let state = seqRegistry[keys[row]]
            let queued = state.row.takeQueued() // exactly the owner snapshot drain used by ModelThread
            let tokens = [first[i]] + queued
            try need(!tokens.isEmpty && tokens.allSatisfy { $0 >= 0 } && tokens.count <= executedK + 1, "invalid committed token stream")
            let a = executedK == 0 ? 0 : tokens.count - 1
            if executedK > 0 { if a > 0 { acceptedPositive += 1 }; if a < executedK { rollbackRows += 1 } }
            accepted.append(a); outputs.append(tokens); queues.append(queued)
            streams[row] += tokens; pending[row] = tokens.last!; committed[row] += tokens.count
            if let slot = afterPoolRows.firstIndex(of: row), let headSlot = batchPoolShared.specRows.firstIndex(of: slot), let sp = batchPoolShared.specPool {
                try need(afterPoolLengths[slot] == committed[row] && sp.headLens[headSlot] == committed[row], "pool/head committed authority")
                headLens.append(sp.headLens[headSlot]); drafts.append(sp.d1[headSlot])
            } else {
                guard let sp = state.row.spec else { throw EngineError.invalid("H20 missing private head") }
                try need(members.count == 1 && kvOffset(state.row.caches) == committed[row] && sp.headLen == committed[row], "plain private/head authority")
                headLens.append(sp.headLen); drafts.append(sp.d1)
            }
            try need(sameCacheObjects(state.cache, state.row.caches), "post-dispatch D36 alias sync")
            if afterPoolRows.contains(row) && requested && !batchPoolShared.rowResident {
                try need(isPlaceholder(state.cache) && isPlaceholder(state.row.caches), "both aliases must be private placeholders")
                placeholderChecks += 1
            } else if afterPoolRows.contains(row) && batchPoolShared.rowResident {
                // P106 H48: a row-resident member's own caches ARE the pool's live history -- by object identity,
                // layer by layer, and its draft head is the one registered behind the spec pool's marker
                try need(hasPrivateHistory(state.cache) && hasPrivateHistory(state.row.caches), "row-resident member history missing")
                let slot = afterPoolRows.firstIndex(of: row)!
                for (l, c) in batchPoolShared.cache!.enumerated() {
                    guard let rowsL = Qwen4ExpModel.rowResidentRows(c) else { continue }
                    try need(rowsL[slot] === (state.row.caches[l] as AnyObject), "row-resident registry/member identity layer \(l)")
                }
                if let sp = batchPoolShared.specPool, let headSlot = batchPoolShared.specRows.firstIndex(of: slot) {
                    try need(Qwen4ExpModel.rowResidentRows(sp.headCache)?[headSlot] === (state.row.spec?.mtpCache as AnyObject?),
                             "row-resident head registry/member identity")
                    try need(!((state.row.spec!.mtpCache as AnyObject) === (state.retiredMTPCache as AnyObject?)),
                             "row-resident pool writes the retired head")
                }
                rowResidentChecks += 1
            } else {
                try need(hasPrivateHistory(state.cache) && hasPrivateHistory(state.row.caches), "usable private cache missing")
                privateChecks += 1
            }
        }
        // Suspended rows are real departed owners, retained without artificial teardown or advancing their state.
        for row in 0..<8 {
            let state = seqRegistry[keys[row]]
            try need(state.retiredMTPCache != nil && ObjectIdentifier(state.retiredMTPCache! as AnyObject) == retiredHeadOwners[row], "retired MTP owner changed")
            retiredPreservedChecks += 1
            if !afterPoolRows.contains(row) {
                try need(hasPrivateHistory(state.row.caches) && kvOffset(state.row.caches) == committed[row], "departed private cache/length")
            }
        }
        selectedRows += members.count
        records.append(["dispatch": dispatch, "member_rows": members, "member_keys": requests.map(\.key),
            "pending_in": pendingIn, "lengths_in": lengthIn, "row_lengths_before": beforeRowLengths,
            "pool_rows_before": beforePoolRows, "pool_lengths_before": beforePoolLengths,
            "pool_rows_after": afterPoolRows, "pool_lengths_after": afterPoolLengths,
            "K": executedK, "returned_tokens": first, "drained_queued": queues,
            "committed_tokens": outputs, "accepted": accepted, "head_lengths_after": headLens, "d1_after": drafts,
            "all_committed_lengths_after": committed, "all_pending_after": pending,
            "plain_head_counter_delta": batchMTPPlainSteps - plainHeadsBefore])
        try memoryGuard()
        if (dispatch + 1) % 16 == 0 {
            FileHandle.standardError.write(Data("H20 dispatches=\(dispatch + 1)/128 selected_rows=\(selectedRows)\n".utf8))
        }
    }
    // Actual dissolve restores row.caches using pool-authoritative lengths. SeqState's stale/placeholder alias
    // is synchronized in the same direction as serving teardown; final hashes are taken only afterward.
    batchPoolShared.dissolve(qm, ratio: ratio)
    var finalRows: [[String: Any]] = []
    for row in 0..<8 {
        let s = seqRegistry[keys[row]]
        s.cache = s.row.caches
        guard let sp = s.row.spec, let retired = s.retiredMTPCache else { throw EngineError.invalid("H20 final head owner missing") }
        try need(hasPrivateHistory(s.cache) && !isPlaceholder(s.cache) && sameCacheObjects(s.cache, s.row.caches), "final alias placeholder/sync")
        try need(kvOffset(s.cache) == committed[row] && sp.headLen == committed[row], "final committed length")
        finalRows.append(["row": row, "length": committed[row], "pending": pending[row], "head_length": sp.headLen,
            "d1": sp.d1, "aliases_synchronized": true, "private_history_present": true, "placeholder": false,
            "trunk": try cacheDigest(s.cache, expectedLength: committed[row]),
            "head": try cacheDigest([sp.mtpCache], expectedLength: committed[row]),
            "slast": logicalArray("Slast", sp.Slast),
            "retired_head": try cacheDigest([retired], expectedLength: rows[row].count)])
    }
    let rowsInstalled = pooledPrivateHistoryRowsInstalled - installsBefore.rows
    let poolsInstalled = pooledPrivateHistoryPoolInstalls - installsBefore.pools
    if BatchDraftPolicy.cycle(batchMTPPolicy) == nil {
        try need(records.count == 128 && selectedRows == 888 && mtpDispatches == 120 && plainDispatches == 8, "empty/incomplete fixed coverage")
    } else {
        // H60 audit (GQ12): under a cycle policy the coverage that matters is the pre-queue's DISCARD paths -- multi-row plain
        // steps and a K other than 3 must both occur (B49 kept the always-only 120/8 check and would have ERRORed)
        try need(records.count == 128 && selectedRows == 888 && mtpDispatches + plainDispatches == 128
                 && multiRowPlainDispatches > 0 && otherKDispatches > 0, "cycle coverage: multi-row plain steps and a K change after rounds")
    }
    try need(staleLengthTransitions >= 4, "D36 stale member-length coverage missing")
    try need(!rowResidentMode || (rowResidentAliasRepairs - rrBefore.repairs == 0 && rowResidentHeadClones - rrBefore.clones == 8),
             "row-resident alias repairs 0 and one retired-head copy per row")
    let expectInstalls = requested && !rowResidentMode
    try need(rowsInstalled == (expectInstalls ? 35 : 0) && poolsInstalled == (expectInstalls ? 5 : 0), "release actual installation coverage")
    try need(rowResidentMode ? (rowResidentChecks > 0 && placeholderChecks == 0) : rowResidentChecks == 0, "row-resident mode coverage")
    try need(batchLengthMismatches == 0 && hotRungMismatches == 0 && batchSpecDropped == 0, "serving invariant counters")
    for key in keys { seqRegistry.destroy(key) }
    keys.removeAll()
    try need(seqRegistry.count == 0 && batchPoolShared.cache == nil && batchPoolShared.rows.isEmpty, "final owner registry/pool cleanup")
    let identity: [String: Any] = ["corpus_sha256": corpusSHA256, "row_token_sha256": rows.map(tokenDigest),
        "row_token_counts": rows.map(\.count), "row_start_offsets": starts, "chunk": chunk,
        "schedule_sha256": digest(scheduleData), "growth_limit": 262144, "kv_step": 256, "indexer_step": 1024,
        "mtp_depth": 3, "mtp_policy": batchMTPPolicy, "sampling": "greedy", "bias": 0, "stop_ids": [Int]()]
    let comparison: [String: Any] = ["schedule": schedule, "seed_checkpoint": seed, "dispatches": records,
        "full_ids_by_row": streams, "final_lengths": committed, "final_pending": pending,
        "final_checkpoint": finalRows, "dispatch_count": records.count, "selected_row_calls": selectedRows,
        "mtp_dispatches": mtpDispatches, "plain_dispatches": plainDispatches,
        "accepted_positive_rows": acceptedPositive, "rollback_rows": rollbackRows,
        "final_registry_count": seqRegistry.count, "final_pool_rows": try poolRows()]
    let execution: [String: Any] = ["rows_installed": rowsInstalled, "pool_installs": poolsInstalled,
        "placeholder_checks": placeholderChecks, "private_history_checks": privateChecks,
        "row_resident": rowResidentMode, "row_resident_checks": rowResidentChecks,
        "row_resident_pool_installs": rowResidentPoolInstalls, "row_resident_head_clones": rowResidentHeadClones,
        "row_resident_alias_repairs": rowResidentAliasRepairs, "row_resident_markers_left": Qwen4ExpBatch.rowResidentRegistered,
        "retired_owner_checks": retiredPreservedChecks, "stale_length_transitions": staleLengthTransitions,
        "prequeue_taken": roundPrequeueTaken - prequeueBefore.taken, "prequeue_discarded": roundPrequeueDiscarded - prequeueBefore.discarded,
        "multi_row_plain_dispatches": multiRowPlainDispatches, "other_k_dispatches": otherKDispatches,
        "length_mismatches": batchLengthMismatches, "rung_mismatches": hotRungMismatches, "specs_dropped": batchSpecDropped,
        "restack_eval": batchRestackEval, "direct_owner_dispatches": records.count,
        "model_thread_scheduled_steps": modelThreadShared.stepCount,
        "serving_storage": storage, "active_before_storage": storageBefore, "active_after_storage": storageAfter]
    try need(Qwen4ExpBatch.rowResidentRegistered == 0, "row-resident markers released")
    let report: [String: Any] = ["instrument": "h20-fixed-serve-schedule-v1", "status": "COMPLETE",
        "release_requested": requested, "release_applied": releasePooledPrivateHistory, "owner_execution": modelThreadShared.isOwner,
        "row_resident_requested": ProcessInfo.processInfo.environment["ENGINE_ROW_RESIDENT_KV"] == "1", "row_resident_applied": rowResidentMode,
        "scope": "fresh A1OFF/BON/A2OFF diagnostic build; same fixed real Serve batch calls, MTP always3 and B1 plain; logical K/V/indexer, stable fixed0-3 and MTP head/Slast value hashes; transient4-5/tapes layout-only; no HTTP/auto-policy/TPS/full-context transfer",
        "input_identity": identity, "comparison": comparison, "execution": execution]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

// H35 isolated diagnostic. The production callback is byte-identical to B27.
func h35SharedBoundary(_ o: Options, model qm: Qwen4ExpModel, corpus: [Int],
                      corpusSHA256: String) throws {
    let prompt = try o.int("--prompt-tokens", 261600)
    let chunk = try o.int("--prefill-step", 1024)
    let rowCount = try o.int("--ownership-shared-rows", 8)
    let env = ProcessInfo.processInfo.environment
    guard try o.int("--batch", 8) == rowCount, (2...8).contains(rowCount),
          [8192, 261600].contains(prompt), chunk == 1024,
          try o.int("--decode", 128) == 128, try o.int("--mtp", 3) == 3,
          let mtp = qm.mtp, qm.configuration.text.indexerCompressRatio == 4,
          env["ENGINE_BATCH_MTP_POLICY"] == "always", batchMTPPolicy == "always",
          env["ENGINE_RELEASE_POOLED_PRIVATE_HISTORY"] == "0", !releasePooledPrivateHistory,
          env["ENGINE_ROW_RESIDENT_KV"] == "0", !rowResidentKV,   // H35 asserts the stacked pool's aliasing
          env["ENGINE_RESTACK_EVAL"] == nil, env["ENGINE_INDEXER_ROUTE_WITNESS"] == nil,
          env["ENGINE_OWNER_ORDER_TRACE"] == nil, env["ENGINE_IDX_REPLAY_DUMP"] == nil,
          env["ENGINE_IDX_REPLAY_BUDGETS"] == nil, !batchRestackEval,
          !Qwen4ExpIndexerRouteWitness.enabled, Qwen4ExpCacheCapacity.kvStep == 256,
          Qwen4ExpCacheCapacity.indexerStep == 1024, corpus.count >= prompt + 1 else {
        throw EngineError.invalid("H35 requires explicit diagnostic geometry and trace-free always3/release0/row-resident0 E9 profile")
    }
    let owner = ModelThread(minBatch: 4, gatherWindow: 0.025, maxBatch: 8) { _ in
        preconditionFailure("H35 invokes the actual callback directly on its owner")
    }
    modelThreadShared = owner
    defer { owner.shutdown() }
    let result: (String, String?) = owner.exclusive {
        do { return (try h35SharedBoundaryOnOwner(qm, mtp: mtp, tokens: corpus,
                     prompt: prompt, chunk: chunk, corpusSHA256: corpusSHA256, rowCount: rowCount), nil) }
        catch { return ("", String(describing: error)) }
    }
    if let error = result.1 { throw EngineError.invalid(error) }
    print(result.0)
}

private func h35SharedBoundaryOnOwner(_ qm: Qwen4ExpModel, mtp: Qwen4ExpMTP,
        tokens: [Int], prompt: Int, chunk: Int, corpusSHA256: String, rowCount R: Int) throws -> String {
    func need(_ ok: Bool, _ why: String) throws { if !ok { throw EngineError.invalid("H35 INSTRUMENTFAIL: " + why) } }
    try need(modelThreadShared.isOwner && seqRegistry.count == 0 && batchPoolShared.cache == nil, "fresh owner")
    let previousRelease = releasePooledPrivateHistory
    defer { releasePooledPrivateHistory = previousRelease; Qwen4ExpCacheCapacity.configureGrowthLimit(nil) }
    Qwen4ExpCacheCapacity.configureGrowthLimit(262144)
    let lengths = (0..<R).map { prompt - 4 * $0 }
    let sharedLength = ((lengths.min()! - 1) / chunk) * chunk
    let rows = lengths.map { Array(tokens.prefix($0)) }
    let physicalLimit = Double(ProcessInfo.processInfo.physicalMemory) * 0.90
    func memoryGuard() throws {
        try need(Double(Memory.activeMemory + Memory.cacheMemory) + 48e9 <= physicalLimit, "90% physical plus48GB workspace")
    }
    func log(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
    let storage = qm.prepareServingStorage()
    try memoryGuard()
    var copiedBytes = 0, copiedArrays = 0
    func detached(_ a: MLXArray) -> MLXArray {
        autoreleasepool {
            let bytes = a.asData(access: .copy)
            let copy = MLXArray(data: bytes)
            eval(copy)
            let sourcePointer = a.asData(access: .noCopy).data.withUnsafeBytes { Int(bitPattern: $0.baseAddress!) }
            let copyData = copy.asData(access: .noCopy)
            let copyPointer = copyData.data.withUnsafeBytes { Int(bitPattern: $0.baseAddress!) }
            precondition(sourcePointer != copyPointer, "diagnostic clone aliases source")
            precondition(copyData.data == bytes.data, "diagnostic clone altered stored bits")
            copiedBytes += a.nbytes; copiedArrays += 1
            return copy
        }
    }
    // Snapshot raw extents explicitly: ordinary export truncates KV/raw keys and
    // keeps pooled capacity, which would mix layouts. All offsets are plain metadata.
    func capture(_ cache: [KVCache], head: KVCache) -> [String: MLXArray] {
        var d: [String: MLXArray] = [:]
        for (prefix, cs) in [("trunk.", cache), ("mtp.", [head])] {
            qm.exportCaches(cs, prefix: prefix, into: &d)
            for (i, c) in cs.enumerated() {
                guard let list = c as? CacheList, let kv = list[0] as? KVCacheSimple,
                      let idx = list[1] as? ArraysCache else { continue }
                d["\(prefix)L\(i).k"] = kv.rawKeys!; d["\(prefix)L\(i).v"] = kv.rawValues!
                d["\(prefix)L\(i).i0"] = idx[0]!
                d["\(prefix)L\(i).i1"] = idx[1]!
            }
        }
        eval(Array(d.values))
        return d
    }
    func clone(_ d: [String: MLXArray], offset: Int) throws -> ([KVCache], KVCache) {
        let copied = Dictionary(uniqueKeysWithValues: d.keys.sorted().map { ($0, detached(d[$0]!)) })
        let cache = qm.newCache(parameters: nil), head = mtp.newCache()
        for (prefix, cs) in [("trunk.", cache), ("mtp.", [head])] {
            qm.importCaches(cs, prefix: prefix, from: copied)
            for (i, c) in cs.enumerated() {
                guard let list = c as? CacheList, let kv = list[0] as? KVCacheSimple else { continue }
                kv.setBuffers(keys: copied["\(prefix)L\(i).k"]!, values: copied["\(prefix)L\(i).v"]!, offset: offset)
            }
        }
        try memoryGuard()
        return (cache, head)
    }
    // Every common-prefix chunk primes with the actual next token, including the
    // last chunk. Only a completed row primes its final hidden state with n0.
    func advance(_ cache: [KVCache], _ head: KVCache, from: Int, to: Int,
                 final: Bool, label: String) throws -> (Int, BatchRowSpec?) {
        var at = from, n0 = 0
        var lastMixed: MLXArray?, lastHidden: MLXArray?
        while at < to {
            let end = min(to, at + chunk)
            let input = MLXArray(tokens[at ..< end].map { Int32($0) }).reshaped([1, end - at])
            let (logits, hidden) = qm.forwardHidden(input, cache: cache)
            var shifted = Array(tokens[(at + 1) ..< (end + 1)])
            if final && end == to {
                n0 = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                shifted[shifted.count - 1] = n0
            }
            let (mixed, h) = mtp(hidden: hidden,
                tokens: MLXArray(shifted.map { Int32($0) }).reshaped([1, end - at]),
                embed: qm.model.embedTokens, cache: head)
            eval(mixed, h, cache.flatMap { $0.state }, head.state)
            lastMixed = mixed; lastHidden = h; at = end
            try memoryGuard()
            if at % 16384 == 0 || at == to { log("mtp-capacity \(label) prefilled=\(at) active=\(Memory.activeMemory)") }
        }
        guard final, let mixed = lastMixed, let hidden = lastHidden else { return (n0, nil) }
        let draft = qm.draftToken(mixed[0..., -1, 0...])
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        eval(draft, last)
        return (n0, BatchRowSpec(mtpCache: head, headLen: to, d1: draft.item(Int.self), Slast: last))
    }

    func makeSharedSeed() throws -> [String: MLXArray] {
        qm.resetRaggedState(); qm.resetHeadRaggedState()
        let cache = qm.newCache(parameters: nil), head = mtp.newCache()
        _ = try advance(cache, head, from: 0, to: sharedLength, final: false, label: "H35_shared_seed")
        return capture(cache, head: head)
    }
    let seed = try makeSharedSeed()
    let seedBytes = seed.values.reduce(0) { $0 + $1.nbytes }
    var reports: [[String: Any]] = []
    for (name, release) in [("A1", false), ("B", true), ("A2", false)] {
        try need(seqRegistry.count == 0 && batchPoolShared.cache == nil, "between-arm cleanup")
        releasePooledPrivateHistory = release
        log("H35 arm=\(name) release=\(release) seed=\(sharedLength) begin")
        let beforeBytes = copiedBytes, beforeArrays = copiedArrays
        let report = try h35SharedArmOnOwner(qm, rows: rows, rowCount: R, starts: Array(repeating: 0, count: R),
                chunk: chunk, corpusSHA256: corpusSHA256, requested: release) { row in
            let (cache, head) = try clone(seed, offset: sharedLength)
            let (n0, spec) = try advance(cache, head, from: sharedLength, to: lengths[row],
                                          final: true, label: "H35_\(name)_row\(row)")
            guard let spec = spec else { throw EngineError.invalid("H35 missing primed head") }
            return (cache, (n0: n0, spec: spec))
        }
        reports.append(["name": name, "release": release,
            "copied_bytes": copiedBytes - beforeBytes, "copied_arrays": copiedArrays - beforeArrays,
            "report": try JSONSerialization.jsonObject(with: Data(report.utf8))])
        try memoryGuard()
        log("H35 arm=\(name) done active=\(Memory.activeMemory)")
    }
    let report: [String: Any] = ["instrument": "h35-shared-boundary-v1", "status": "COMPLETE",
        "prompt": prompt, "rows": R, "shared_seed_length": sharedLength, "shared_seed_logical_bytes": seedBytes,
        "cold_shared_prefills": 1, "arms": reports, "serving_storage": storage,
        "copied_bytes": copiedBytes, "copied_arrays": copiedArrays,
        "scope": "near-limit real serving callback identity; one real causal RAM seed; no TPS/HTTP/deployment claim"]
    let bytes = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    return String(decoding: bytes, as: UTF8.self)
}

private func h35SharedArmOnOwner(_ qm: Qwen4ExpModel, rows: [[Int]], rowCount R: Int, starts: [Int],
                                        chunk: Int, corpusSHA256: String, requested: Bool,
                                        primeRow: (Int) throws -> ([KVCache], (n0: Int, spec: BatchRowSpec))) throws -> String {
    func need(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError.invalid("H35 INSTRUMENTFAIL: " + message) }
    }
    try need(modelThreadShared.isOwner && seqRegistry.count == 0 && batchPoolShared.cache == nil, "owner/fresh process")
    try need(requested == releasePooledPrivateHistory, "release request/readback")
    let ratio = qm.configuration.text.indexerCompressRatio
    Qwen4ExpCacheCapacity.configureGrowthLimit(262144)
    batchMTPDepthShared = 3
    // Direct calls keep ModelThread.stepCount=0; existing stats logging therefore executes.
    // Initialize its real dependency instead of altering runBatchedStep's logging branch.
    hotStoreShared = HotPrefixStore(capBytes: 20_000_000_000, rungStep: 512)
    var keys: [Int] = []
    defer {
        batchPoolShared.dissolve(qm, ratio: ratio)
        for key in keys { seqRegistry[key].cache = seqRegistry[key].row.caches; seqRegistry.destroy(key) }
        hotStoreShared = nil
        Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
    }
    func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func tokenDigest(_ tokens: [Int]) -> String {
        tokens.map { Int32($0).littleEndian }.withUnsafeBytes { digest(Data($0)) }
    }
    func logicalArray(_ name: String, _ array: MLXArray) -> [String: Any] {
        // Hash the logical view's exact stored dtype/bytes; never capacity padding or raw values in JSON.
        let data = array.asData(access: .copy).data
        return ["name": name, "shape": array.shape, "dtype": String(describing: array.dtype),
                "logical_bytes": array.nbytes, "sha256": digest(data)]
    }
    var capacityWitnesses: [[String: Int]] = []
    func cacheDigest(_ caches: [KVCache], expectedLength: Int) throws -> [[String: Any]] {
        try caches.enumerated().map { layer, cache in
            var arrays: [[String: Any]] = []
            if let list = cache as? CacheList {
                let kv = list[0] as! KVCacheSimple, idx = list[1] as! ArraysCache
                try need(kv.offset == expectedLength && kv.state.count == 2 && idx[0] != nil && idx[1] != nil && idx[2] == nil,
                         "logical text attention state at layer \(layer)")
                try need(idx[0]!.dim(1) >= expectedLength && idx[1]!.dim(1) >= expectedLength / ratio, "logical indexer extent")
                let kcap = kv.rawKeys!.dim(2), vcap = kv.rawValues!.dim(2)
                let rawcap = idx[0]!.dim(1), poolcap = idx[1]!.dim(1)
                try need(kcap <= 262144 && vcap <= 262144 && rawcap <= 262144 && poolcap <= 65536,
                         "actual history capacity exceeds native ceiling")
                capacityWitnesses.append(["layer": layer, "length": expectedLength,
                    "key_capacity": kcap, "value_capacity": vcap, "raw_capacity": rawcap, "pool_capacity": poolcap])
                arrays.append(logicalArray("key", kv.state[0]))
                arrays.append(logicalArray("value", kv.state[1]))
                arrays.append(logicalArray("indexer_raw", idx[0]![0..., 0..<expectedLength, 0...]))
                arrays.append(logicalArray("indexer_pooled", idx[1]![0..., 0..<(expectedLength / ratio), 0...]))
                return ["layer": layer, "kind": "attention", "cache_offset": cache.offset,
                        "kv_offset": kv.offset, "indexer_offset": idx.offset, "arrays": arrays]
            }
            guard let a = cache as? ArraysCache else { throw EngineError.invalid("H35 unknown cache type") }
            // Only stable fixed state is a value oracle. Verify/prefill scratch is not logically retained
            // decode state; record its layout/presence without hashing/evaluating or claiming identity of padding.
            for slot in 0..<4 { if let x = a[slot] { arrays.append(logicalArray("slot\(slot)", x)) } }
            var transient: [[String: Any]] = []
            func layout(_ name: String, _ x: MLXArray) {
                transient.append(["name": name, "shape": x.shape, "dtype": String(describing: x.dtype)])
            }
            for slot in 4..<6 { if let x = a[slot] { layout("slot\(slot)", x) } }
            if let pair = a.rollbackState { layout("rollback_conv", pair.0); layout("rollback_ssm", pair.1) }
            for (i, pair) in a.rollbackCheckpoints.enumerated() {
                layout("checkpoint\(i)_conv", pair.0); layout("checkpoint\(i)_ssm", pair.1)
            }
            var tape: [String: Int] = [:]
            if let t = a.prefixReplayTape {
                tape = ["row_count": t.rowCount, "conv_state_rows": t.convStateRows]
                for (name, x) in [("conv", t.convInput), ("q", t.q), ("k", t.k), ("v", t.v),
                                  ("a", t.a), ("b", t.b), ("g", t.g), ("beta", t.beta)] { layout("tape_" + name, x) }
                if let x = t.ssmPre { layout("tape_ssm_pre", x) }
                if let x = t.mask { layout("tape_mask", x) }
            }
            try need(!arrays.isEmpty, "empty fixed state at layer \(layer)")
            return ["layer": layer, "kind": "fixed", "cache_offset": a.offset,
                    "rollback_checkpoints": a.rollbackCheckpoints.count, "tape": tape, "transient_layout_only": transient, "arrays": arrays]
        }
    }
    func kvOffset(_ caches: [KVCache]) -> Int {
        let first = caches.first { $0 is CacheList } as! CacheList
        return (first[0] as! KVCacheSimple).offset
    }
    func hasPrivateHistory(_ caches: [KVCache]) -> Bool {
        let attention = caches.compactMap { $0 as? CacheList }
        return !attention.isEmpty && attention.allSatisfy {
            let kv = $0[0] as! KVCacheSimple, idx = $0[1] as! ArraysCache
            return kv.rawKeys != nil && kv.rawValues != nil && idx[0] != nil && idx[1] != nil
        }
    }
    func isPlaceholder(_ caches: [KVCache]) -> Bool {
        let attention = caches.compactMap { $0 as? CacheList }
        return !attention.isEmpty && attention.allSatisfy {
            let kv = $0[0] as! KVCacheSimple, idx = $0[1] as! ArraysCache
            return kv.rawKeys == nil && kv.rawValues == nil && idx[0] == nil && idx[1] == nil && idx[2] == nil
        }
    }
    func sameCacheObjects(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { ($0.0 as AnyObject) === ($0.1 as AnyObject) }
    }
    func memoryGuard() throws {
        try need(Double(Memory.activeMemory + Memory.cacheMemory) + 48e9 <= Double(ProcessInfo.processInfo.physicalMemory) * 0.90,
                 "90% physical +48GB workspace safety bound")
    }
    let storageBefore = Memory.activeMemory
    let storage = qm.prepareServingStorage()
    let storageAfter = Memory.activeMemory
    try memoryGuard()
    var pending: [Int] = [], committed = rows.map(\.count), streams: [[Int]] = []
    var retiredHeadOwners: [ObjectIdentifier] = []
    var seed: [[String: Any]] = []
    // Same fresh causal prefill/prime geometry in each A1/B/A2 process. No restored/cold H8 comparison.
    for row in 0..<R {
        FileHandle.standardError.write(Data("H35 seed row=\(row) length=\(rows[row].count) begin\n".utf8))
        qm.resetRaggedState(); qm.resetHeadRaggedState()
        let (cache, prime) = try primeRow(row)
        let key = seqRegistry.create { SeqState(sampler: EngineSampler(), cache: cache, xs: MLXArray(0), ids: rows[row]) }
        keys.append(key)
        let s = seqRegistry[key]
        s.row.spec = prime.spec
        s.retiredMTPCache = prime.spec.mtpCache // mirrors real post-handover retained head owner; H19 never changes it
        retiredHeadOwners.append(ObjectIdentifier(prime.spec.mtpCache as AnyObject))
        s.row.length = committed[row]; s.row.pending = prime.n0
        s.row.step = { $0.argMax(axis: -1).item(Int.self) }
        s.row.greedy = true; s.row.biasNow = { 0 }; s.row.thinkOpenNow = { false }; s.row.stopIds = []
        pending.append(prime.n0); streams.append([prime.n0])
        seed.append(["row": row, "length": committed[row], "pending": prime.n0,
            "head_length": prime.spec.headLen, "d1": prime.spec.d1,
            "trunk": try cacheDigest(cache, expectedLength: committed[row]),
            "head": try cacheDigest([prime.spec.mtpCache], expectedLength: committed[row]),
            "slast": logicalArray("Slast", prime.spec.Slast)])
        try memoryGuard()
        FileHandle.standardError.write(Data("H35 seed row=\(row) ready\n".utf8))
    }
    let keyToRow = Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($0.element, $0.offset) })
    func poolRows() throws -> [Int] {
        try batchPoolShared.rows.map { row in
            guard let key = seqRegistry.traceKey(row), let id = keyToRow[key] else { throw EngineError.invalid("H35 unknown pool row") }
            return id
        }
    }
    func orderedMembers(_ dispatch: Int) -> [Int] {
        let base: [Int]
        switch dispatch {
        case 0..<32, 64..<96, 104..<128: base = Array(0..<R)
        case 32..<48: base = Array(0..<(R - 1))
        case 48..<64: base = Array(0..<max(2, R / 2))
        default: base = [0] // dispatch96...103: real plain B1 + head keep-alive
        }
        let shift = dispatch % base.count
        let rotated = Array(base[shift...]) + Array(base[..<shift])
        return dispatch % 2 == 0 ? rotated : Array(rotated.reversed())
    }
    let schedule = (0..<128).map(orderedMembers)
    let scheduleData = try JSONSerialization.data(withJSONObject: schedule, options: [.sortedKeys])
    let installsBefore = (rows: pooledPrivateHistoryRowsInstalled, pools: pooledPrivateHistoryPoolInstalls)
    var records: [[String: Any]] = []
    var staleLengthTransitions = 0, retiredPreservedChecks = 0, placeholderChecks = 0, privateChecks = 0
    var selectedRows = 0, mtpDispatches = 0, plainDispatches = 0, acceptedPositive = 0, rollbackRows = 0
    // No checkpoint/hash/eval between first stack and final dissolve. This is deliberate lazy-ownership coverage.
    for dispatch in 0..<128 {
        let members = schedule[dispatch]
        let beforePoolRows = try poolRows(), beforePoolLengths = batchPoolShared.lengths
        let beforeRowLengths = keys.map { seqRegistry[$0].row.length }
        let lengthIn = members.map { committed[$0] }, pendingIn = members.map { pending[$0] }
        let changed = Set(beforePoolRows) != Set(members)
        if changed && !beforePoolRows.isEmpty {
            if zip(beforePoolRows, beforePoolLengths).contains(where: { beforeRowLengths[$0.0] != $0.1 }) { staleLengthTransitions += 1 }
        }
        let roundsBefore = batchMTPRounds, plainHeadsBefore = batchMTPPlainSteps
        try need(lengthIn.allSatisfy { $0 + 4 <= 262144 }, "executed MTP horizon exceeds native context")
        let requests = members.map { StepRequest(key: keys[$0], pending: pending[$0], length: committed[$0]) }
        let first = runBatchedStep(requests, model: qm, qm: qm, ratio: ratio)
        try need(first.count == members.count, "returned token count")
        let afterPoolRows = try poolRows()
        let afterPoolLengths = batchPoolShared.lengths
        let executedK = batchMTPRounds == roundsBefore + 1 ? 3 : 0
        try need(executedK == (members.count > 1 ? 3 : 0), "fixed always-K3/plain-B1 branch")
        if executedK == 3 { mtpDispatches += 1 } else { plainDispatches += 1 }
        var outputs: [[Int]] = [], queues: [[Int]] = [], headLens: [Int] = [], drafts: [Int] = [], accepted: [Int] = []
        for (i, row) in members.enumerated() {
            let state = seqRegistry[keys[row]]
            let queued = state.row.takeQueued() // exactly the owner snapshot drain used by ModelThread
            let tokens = [first[i]] + queued
            try need(!tokens.isEmpty && tokens.allSatisfy { $0 >= 0 } && tokens.count <= executedK + 1, "invalid committed token stream")
            let a = executedK == 0 ? 0 : tokens.count - 1
            if executedK > 0 { if a > 0 { acceptedPositive += 1 }; if a < executedK { rollbackRows += 1 } }
            accepted.append(a); outputs.append(tokens); queues.append(queued)
            streams[row] += tokens; pending[row] = tokens.last!; committed[row] += tokens.count
            if let slot = afterPoolRows.firstIndex(of: row), let headSlot = batchPoolShared.specRows.firstIndex(of: slot), let sp = batchPoolShared.specPool {
                try need(afterPoolLengths[slot] == committed[row] && sp.headLens[headSlot] == committed[row], "pool/head committed authority")
                headLens.append(sp.headLens[headSlot]); drafts.append(sp.d1[headSlot])
            } else {
                guard let sp = state.row.spec else { throw EngineError.invalid("H35 missing private head") }
                try need(members.count == 1 && kvOffset(state.row.caches) == committed[row] && sp.headLen == committed[row], "plain private/head authority")
                headLens.append(sp.headLen); drafts.append(sp.d1)
            }
            try need(sameCacheObjects(state.cache, state.row.caches), "post-dispatch D36 alias sync")
            if afterPoolRows.contains(row) && requested {
                try need(isPlaceholder(state.cache) && isPlaceholder(state.row.caches), "both aliases must be private placeholders")
                placeholderChecks += 1
            } else {
                try need(hasPrivateHistory(state.cache) && hasPrivateHistory(state.row.caches), "usable private cache missing")
                privateChecks += 1
            }
        }
        // Suspended rows are real departed owners, retained without artificial teardown or advancing their state.
        for row in 0..<R {
            let state = seqRegistry[keys[row]]
            try need(state.retiredMTPCache != nil && ObjectIdentifier(state.retiredMTPCache! as AnyObject) == retiredHeadOwners[row], "retired MTP owner changed")
            retiredPreservedChecks += 1
            if !afterPoolRows.contains(row) {
                try need(hasPrivateHistory(state.row.caches) && kvOffset(state.row.caches) == committed[row], "departed private cache/length")
            }
        }
        selectedRows += members.count
        records.append(["dispatch": dispatch, "member_rows": members, "member_keys": members.map { $0 + 1 },
            "pending_in": pendingIn, "lengths_in": lengthIn, "row_lengths_before": beforeRowLengths,
            "pool_rows_before": beforePoolRows, "pool_lengths_before": beforePoolLengths,
            "pool_rows_after": afterPoolRows, "pool_lengths_after": afterPoolLengths,
            "K": executedK, "returned_tokens": first, "drained_queued": queues,
            "committed_tokens": outputs, "accepted": accepted, "head_lengths_after": headLens, "d1_after": drafts,
            "all_committed_lengths_after": committed, "all_pending_after": pending,
            "plain_head_counter_delta": batchMTPPlainSteps - plainHeadsBefore])
        try memoryGuard()
        if (dispatch + 1) % 16 == 0 {
            FileHandle.standardError.write(Data("H35 dispatches=\(dispatch + 1)/128 selected_rows=\(selectedRows)\n".utf8))
        }
    }
    // Actual dissolve restores row.caches using pool-authoritative lengths. SeqState's stale/placeholder alias
    // is synchronized in the same direction as serving teardown; final hashes are taken only afterward.
    batchPoolShared.dissolve(qm, ratio: ratio)
    var finalRows: [[String: Any]] = []
    for row in 0..<R {
        let s = seqRegistry[keys[row]]
        s.cache = s.row.caches
        guard let sp = s.row.spec, let retired = s.retiredMTPCache else { throw EngineError.invalid("H35 final head owner missing") }
        try need(hasPrivateHistory(s.cache) && !isPlaceholder(s.cache) && sameCacheObjects(s.cache, s.row.caches), "final alias placeholder/sync")
        try need(kvOffset(s.cache) == committed[row] && sp.headLen == committed[row], "final committed length")
        finalRows.append(["row": row, "length": committed[row], "pending": pending[row], "head_length": sp.headLen,
            "d1": sp.d1, "aliases_synchronized": true, "private_history_present": true, "placeholder": false,
            "trunk": try cacheDigest(s.cache, expectedLength: committed[row]),
            "head": try cacheDigest([sp.mtpCache], expectedLength: committed[row]),
            "slast": logicalArray("Slast", sp.Slast),
            "retired_head": try cacheDigest([retired], expectedLength: rows[row].count)])
    }
    let rowsInstalled = pooledPrivateHistoryRowsInstalled - installsBefore.rows
    let poolsInstalled = pooledPrivateHistoryPoolInstalls - installsBefore.pools
    let expectedSelected = schedule.reduce(0) { $0 + $1.count }
    let expectedMTP = schedule.filter { $0.count > 1 }.count
    let expectedPlain = 128 - expectedMTP
    try need(records.count == 128 && selectedRows == expectedSelected && mtpDispatches == expectedMTP && plainDispatches == expectedPlain, "empty/incomplete fixed coverage")
    try need(staleLengthTransitions >= 1, "D36 stale member-length coverage missing")
    try need(requested ? (rowsInstalled > 0 && poolsInstalled > 0) : (rowsInstalled == 0 && poolsInstalled == 0), "release actual installation coverage")
    try need(batchLengthMismatches == 0 && hotRungMismatches == 0 && batchSpecDropped == 0, "serving invariant counters")
    let keysForEvidence = keys
    for key in keys { seqRegistry.destroy(key) }
    keys.removeAll()
    try need(seqRegistry.count == 0 && batchPoolShared.cache == nil && batchPoolShared.rows.isEmpty, "final owner registry/pool cleanup")
    let identity: [String: Any] = ["corpus_sha256": corpusSHA256, "row_token_sha256": rows.map(tokenDigest),
        "row_token_counts": rows.map(\.count), "row_start_offsets": starts, "chunk": chunk, "rows": R,
        "schedule_sha256": digest(scheduleData), "growth_limit": 262144, "kv_step": 256, "indexer_step": 1024,
        "mtp_depth": 3, "mtp_policy": "always", "sampling": "greedy", "bias": 0, "stop_ids": [Int]()]
    let comparison: [String: Any] = ["schedule": schedule, "seed_checkpoint": seed, "dispatches": records,
        "full_ids_by_row": streams, "final_lengths": committed, "final_pending": pending,
        "final_checkpoint": finalRows, "dispatch_count": records.count, "selected_row_calls": selectedRows,
        "mtp_dispatches": mtpDispatches, "plain_dispatches": plainDispatches,
        "accepted_positive_rows": acceptedPositive, "rollback_rows": rollbackRows,
        "final_registry_count": seqRegistry.count, "final_pool_rows": try poolRows()]
    let execution: [String: Any] = ["rows_installed": rowsInstalled, "pool_installs": poolsInstalled,
        "placeholder_checks": placeholderChecks, "private_history_checks": privateChecks,
        "retired_owner_checks": retiredPreservedChecks, "stale_length_transitions": staleLengthTransitions,
        "length_mismatches": batchLengthMismatches, "rung_mismatches": hotRungMismatches, "specs_dropped": batchSpecDropped,
        "restack_eval": batchRestackEval, "direct_owner_dispatches": records.count,
        "model_thread_scheduled_steps": modelThreadShared.stepCount, "sequence_keys": keysForEvidence, "capacity_witnesses": capacityWitnesses,
        "serving_storage": storage, "active_before_storage": storageBefore, "active_after_storage": storageAfter]
    let report: [String: Any] = ["instrument": "h35-shared-boundary-arm-v1", "status": "COMPLETE",
        "release_requested": requested, "release_applied": releasePooledPrivateHistory, "owner_execution": modelThreadShared.isOwner,
        "scope": "one causal shared RAM seed, sequential A1OFF/BON/A2OFF on owner; same fixed real Serve batch calls, MTP always3 and B1 plain; logical K/V/indexer, stable fixed0-3 and MTP head/Slast value hashes; transient4-5/tapes layout-only; no HTTP/auto-policy/TPS claim",
        "input_identity": identity, "comparison": comparison, "execution": execution]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

// H40 pre-admission diagnostic: tokenize a wire payload exactly as the serve path does
// (JSONSerialization -> parseChatRequest -> applyChatTemplate) on N barrier-synchronized
// threads, without loading model weights. Shared mode passes one Tokenizer to every
// thread, mirroring production's shared context.tokenizer; per-thread loads N instances.
private struct TokCallRecord {
    var thread: Int
    var repeatIndex: Int
    var jsonMs: Double
    var parseMs: Double
    var templateMs: Double
    var totalMs: Double
    var nIds: Int
    var idsSHA256: String
    var error: String?
}

private final class TokBenchShared: @unchecked Sendable {
    var records: [[TokCallRecord]]
    let lock = NSLock()
    init(threads: Int) { records = Array(repeating: [], count: threads) }
}

private final class TokLoadBox: @unchecked Sendable {
    var tokenizers: [any MLXLMCommon.Tokenizer] = []
    var error: Error?
}

// Synchronous entry point: the async tokenizer load is bridged with a detached
// task so the barrier/join semaphore waits stay legal outside async contexts.
func tokBench(_ o: Options) throws {
    let modelPath = o.string("--model", "weights/e9")
    let payloadPath = o.string("--payload", "")
    let concurrency = try o.int("--concurrency", 1)
    let repeatCount = try o.int("--repeat", 3)
    let instances = o.string("--instances", "shared")
    let idsOut = o.string("--ids-out", "")
    let cacheCheck = o.string("--cache-check", "")
    if !cacheCheck.isEmpty {
        guard FileManager.default.fileExists(atPath: cacheCheck) else {
            throw EngineError.invalid("--cache-check spec not found")
        }
        let dir0 = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let loadStart0 = DispatchTime.now().uptimeNanoseconds
        let box0 = TokLoadBox()
        let loaded0 = DispatchSemaphore(value: 0)
        Task.detached {
            do { box0.tokenizers = [try await #huggingFaceTokenizerLoader().load(from: dir0)] }
            catch { box0.error = error }
            loaded0.signal()
        }
        loaded0.wait()
        if let e = box0.error { throw e }
        let loadMs0 = Double(DispatchTime.now().uptimeNanoseconds - loadStart0) / 1e6
        FileHandle.standardError.write(String(format: "engine tokbench: tokenizer loaded in %.1f ms (cache-check), no weights\n", loadMs0).data(using: .utf8)!)
        try tokBenchCacheCheck(specPath: cacheCheck, tokenizer: box0.tokenizers[0], modelDir: dir0, tokenizerLoadMs: loadMs0)
        return
    }
    guard !payloadPath.isEmpty else { throw EngineError.invalid("tokbench requires --payload PATH") }
    guard instances == "shared" || instances == "per-thread" else {
        throw EngineError.invalid("--instances must be shared or per-thread")
    }
    guard (1...64).contains(concurrency), (1...1000).contains(repeatCount) else {
        throw EngineError.invalid("--concurrency/--repeat out of range")
    }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let box = TokLoadBox()
    let loaded = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            if instances == "per-thread" {
                for _ in 0..<concurrency {
                    box.tokenizers.append(try await #huggingFaceTokenizerLoader().load(from: dir))
                }
            } else {
                let one = try await #huggingFaceTokenizerLoader().load(from: dir)
                box.tokenizers = Array(repeating: one, count: concurrency)
            }
        } catch { box.error = error }
        loaded.signal()
    }
    loaded.wait()
    if let e = box.error { throw e }
    let tokenizers = box.tokenizers
    let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e6
    FileHandle.standardError.write(String(format: "engine tokbench: tokenizer(s) loaded in %.1f ms (%@), no weights\n", loadMs, instances).data(using: .utf8)!)
    let body = try Data(contentsOf: URL(fileURLWithPath: payloadPath))
    let payloadSHA = SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined()
    let shared = TokBenchShared(threads: concurrency)
    let ready = DispatchSemaphore(value: 0)
    let start = DispatchSemaphore(value: 0)
    let done = DispatchSemaphore(value: 0)
    for t in 0..<concurrency {
        let tokenizer = tokenizers[t]
        Thread {
            ready.signal()
            start.wait()
            var mine: [TokCallRecord] = []
            mine.reserveCapacity(repeatCount)
            for r in 0..<repeatCount {
                let t0 = DispatchTime.now().uptimeNanoseconds
                var rec = TokCallRecord(thread: t, repeatIndex: r, jsonMs: -1, parseMs: -1,
                                        templateMs: -1, totalMs: -1, nIds: -1, idsSHA256: "", error: nil)
                do {
                    guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                        throw EngineError.invalid("payload is not a JSON object")
                    }
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    guard let chatReq = parseChatRequest(obj, defaultMaxTokens: 32000,
                                                         defaultEffort: "medium", keepReasoning: true) else {
                        throw EngineError.invalid("payload has no messages")
                    }
                    let t2 = DispatchTime.now().uptimeNanoseconds
                    var toolsForTemplate: [Message]? = nil
                    if !chatReq.rawTools.isEmpty, !(toolChoiceNoneDropsTools && { if case .none = chatReq.toolChoice { return true } else { return false } }()) {
                        toolsForTemplate = chatReq.rawTools.map(toSendableDict)
                    }
                    var extra: [String: any Sendable] = ["reasoning_effort": chatReq.reasoningEffort]
                    if let pt = chatReq.preserveThinking { extra["preserve_thinking"] = pt }
                    let ids = try tokenizer.applyChatTemplate(messages: chatReq.messages,
                                                              tools: toolsForTemplate, additionalContext: extra)
                    let t3 = DispatchTime.now().uptimeNanoseconds
                    let idsData = try JSONSerialization.data(withJSONObject: ids)
                    rec.jsonMs = Double(t1 - t0) / 1e6
                    rec.parseMs = Double(t2 - t1) / 1e6
                    rec.templateMs = Double(t3 - t2) / 1e6
                    rec.totalMs = Double(t3 - t0) / 1e6
                    rec.nIds = ids.count
                    rec.idsSHA256 = SHA256.hash(data: idsData).map { String(format: "%02x", $0) }.joined()

                    if r == 0 && t == 0 && !idsOut.isEmpty {
                        try? idsData.write(to: URL(fileURLWithPath: idsOut))
                    }
                } catch {
                    rec.totalMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                    rec.error = String(describing: error)
                }
                mine.append(rec)
            }
            shared.lock.lock(); shared.records[t] = mine; shared.lock.unlock()
            done.signal()
        }.start()
    }
    for _ in 0..<concurrency { ready.wait() }
    for _ in 0..<concurrency { start.signal() }
    for _ in 0..<concurrency { done.wait() }
    var all: [TokCallRecord] = []
    for t in 0..<concurrency { all += shared.records[t] }
    if let failure = all.compactMap({ $0.error }).first { throw EngineError.invalid("tokbench call failed: \(failure)") }
    func stats(_ key: KeyPath<TokCallRecord, Double>) -> [String: Double] {
        let v = all.map { $0[keyPath: key] }.sorted()
        return ["min_ms": v.first ?? 0, "median_ms": v[v.count / 2], "max_ms": v.last ?? 0]
    }
    let threadsJSON: [[String: Any]] = shared.records.map { recs in
        ["thread": recs.first?.thread ?? -1,
         "calls": recs.map { ["repeat": $0.repeatIndex, "json_ms": $0.jsonMs, "parse_ms": $0.parseMs,
                              "apply_template_ms": $0.templateMs, "total_ms": $0.totalMs,
                              "n_ids": $0.nIds, "ids_sha256": $0.idsSHA256] }]
    }
    let report: [String: Any] = [
        "instrument": "h40-tokbench-v1", "status": "COMPLETE", "payload_sha256": payloadSHA,
        "prompt_tokens": all.first?.nIds ?? 0, "concurrency": concurrency, "instances": instances,
        "repeats": repeatCount, "tokenizer_load_ms": loadMs,
        "ids_identical_across_calls": all.allSatisfy { $0.idsSHA256 == all.first?.idsSHA256 },
        "ids_sha256": all.first?.idsSHA256 ?? "",
        "json": stats(\.jsonMs), "parse": stats(\.parseMs),
        "apply_template": stats(\.templateMs), "total": stats(\.totalMs),
        "threads": threadsJSON,
        "scope": "tokenizer-only host stage timings; no weights/GPU; not a TPS or serving claim"]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

// H42-UNIT: gate the B32 segment id cache. One JSON result, five sub-checks:
//  (a) every payload: cached ids == one-shot applyChatTemplate ids;
//  (b) per-OMP24-worker multi-turn: misses == segments new vs that worker's earlier waves;
//  (c) 8 concurrent encodes of the same payload through one cache: 1 miss, 7 waits, <=1.5 s;
//  (d) capacity 1_000_000 B (below one payload's ~1.03 MB): ids exact, evictions > 0;
//  (e) disabled path ids identical and a fresh cache's counters stay zero.
private struct TokPrepared {
    var name: String
    var oneShotIds: [Int]
    var rendered: String
    var renderMs: Double
    var oneShotMs: Double
}

private func tokPrepare(name: String, path: String, tokenizer: any MLXLMCommon.Tokenizer,
                        renderer: ChatTemplateRender) throws -> TokPrepared {
    let body = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
        throw EngineError.invalid("\(name): payload is not a JSON object")
    }
    guard let chatReq = parseChatRequest(obj, defaultMaxTokens: 32000,
                                         defaultEffort: "medium", keepReasoning: true) else {
        throw EngineError.invalid("\(name): payload has no messages")
    }
    var toolsForTemplate: [Message]? = nil
    if !chatReq.rawTools.isEmpty, !(toolChoiceNoneDropsTools && { if case .none = chatReq.toolChoice { return true } else { return false } }()) {
        toolsForTemplate = chatReq.rawTools.map(toSendableDict)
    }
    var extra: [String: any Sendable] = ["reasoning_effort": chatReq.reasoningEffort]
    if let pt = chatReq.preserveThinking { extra["preserve_thinking"] = pt }
    let r0 = DispatchTime.now().uptimeNanoseconds
    let rendered = try renderer.render(messages: chatReq.messages, tools: toolsForTemplate, extra: extra)
    let r1 = DispatchTime.now().uptimeNanoseconds
    let ids = try tokenizer.applyChatTemplate(messages: chatReq.messages,
                                              tools: toolsForTemplate, additionalContext: extra)
    let r2 = DispatchTime.now().uptimeNanoseconds
    return TokPrepared(name: name, oneShotIds: ids, rendered: rendered,
                       renderMs: Double(r1 - r0) / 1e6, oneShotMs: Double(r2 - r1) / 1e6)
}

private final class TokCacheCheckBox: @unchecked Sendable {
    var ids: [[Int]] = []
    var error: String?
}

private func tokIdsEqualDetail(_ a: [Int], _ b: [Int]) -> (Bool, Int, Int, Int) {
    for i in 0..<min(a.count, b.count) where a[i] != b[i] { return (false, i, a[i], b[i]) }
    if a.count == b.count { return (true, -1, -1, -1) }
    return (false, min(a.count, b.count), -1, -1)
}

func tokBenchCacheCheck(specPath: String, tokenizer: any MLXLMCommon.Tokenizer,
                        modelDir: URL, tokenizerLoadMs: Double) throws {
    let specData = try Data(contentsOf: URL(fileURLWithPath: specPath))
    guard let spec = try JSONSerialization.jsonObject(with: specData) as? [String: Any],
          let singles = spec["singles"] as? [String: String],
          let workers = spec["workers"] as? [String: [String]],
          let concurrentPath = spec["concurrent_payload"] as? String else {
        throw EngineError.invalid("--cache-check spec must have singles, workers, concurrent_payload")
    }
    // Optional {"name": "<name>.cpu-ids.json"} fixtures: assert ids equal the frozen CPU
    // reference arrays, not just the in-process one-shot call.
    var fixtureIds: [String: [Int]] = [:]
    for (name, path) in spec["fixtures"] as? [String: String] ?? [:] {
        if let arr = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [Int] {
            fixtureIds[name] = arr
        } else if let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
                  let arr = (obj["ids"] ?? obj["token_ids"]) as? [Int] {
            fixtureIds[name] = arr
        }
    }
    let renderer = try ChatTemplateRender(modelDir: modelDir)
    var failures: [String] = []
    var report: [String: Any] = [:]

    // Prepare all payloads once (render + one-shot reference ids).
    var prepared: [String: TokPrepared] = [:]
    for (name, path) in singles { prepared[name] = try tokPrepare(name: name, path: path, tokenizer: tokenizer, renderer: renderer) }
    for (worker, paths) in workers {
        for (i, p) in paths.enumerated() {
            prepared["\(worker)/\(i)"] = try tokPrepare(name: "\(worker)/\(i)", path: p, tokenizer: tokenizer, renderer: renderer)
        }
    }
    prepared["concurrent"] = try tokPrepare(name: "concurrent", path: concurrentPath, tokenizer: tokenizer, renderer: renderer)

    // (a) all payloads through one cache; ids must equal the one-shot ids.
    do {
        let cache = try TemplateTokenCache(capacityBytes: 256 * 1024 * 1024, tokenizer: tokenizer, modelDir: modelDir)
        var rows: [[String: Any]] = []
        for name in prepared.keys.sorted() {
            let p = prepared[name]!
            let (ids, d) = cache.encode(rendered: p.rendered)
            let (eq, idx, cv, rv) = tokIdsEqualDetail(ids, p.oneShotIds)
            if !eq { failures.append("a:\(name):first_diff index=\(idx) cached=\(cv) reference=\(rv) counts \(ids.count)vs\(p.oneShotIds.count)") }
            var fixtureEq = true
            if let fix = fixtureIds[name] {
                fixtureEq = ids == fix
                if !fixtureEq { failures.append("a:\(name):fixture cpu-ids differ") }
            }
            rows.append(["name": name, "tokens": ids.count, "exact": eq, "fixture_exact": fixtureEq, "sections": d.sections,
                         "text_segments": d.textSegments, "cached_segments": d.cachedSegments,
                         "encoded_segments": d.encodedSegments, "waited": d.waited,
                         "render_ms": p.renderMs, "one_shot_ms": p.oneShotMs])
        }
        report["a_exactness"] = ["pass": rows.allSatisfy { ($0["exact"] as? Bool == true) && ($0["fixture_exact"] as? Bool == true) },
                                 "payloads": rows, "cache_stats": tokCacheStatsDict(cache)]
    }

    // (b) per-worker multi-turn: misses must equal segments new vs the worker's earlier waves.
    do {
        var rows: [[String: Any]] = []
        for worker in workers.keys.sorted() {
            let paths = workers[worker] ?? []
            let cache = try TemplateTokenCache(capacityBytes: 256 * 1024 * 1024, tokenizer: tokenizer, modelDir: modelDir)
            var seenKeys = Set<String>()
            for (wave, _) in paths.enumerated() {
                let p = prepared["\(worker)/\(wave)"]!
                let (_, keys) = cache.sectionKeys(rendered: p.rendered)
                let thisKeys = Set(keys.compactMap { $0 })
                let expectedNew = thisKeys.subtracting(seenKeys)
                let (ids, d) = cache.encode(rendered: p.rendered)
                let (eq, idx, cv, rv) = tokIdsEqualDetail(ids, p.oneShotIds)
                if !eq { failures.append("b:\(worker)/\(wave):first_diff index=\(idx) cached=\(cv) reference=\(rv)") }
                let observedNew = Set(d.missedKeys)
                if observedNew != expectedNew {
                    failures.append("b:\(worker)/\(wave):missed-set differs (observed \(observedNew.count) expected \(expectedNew.count))")
                }
                rows.append(["worker": worker, "wave": wave, "tokens": ids.count, "exact": eq,
                             "sections": d.sections, "text_segments": d.textSegments,
                             "cached_segments": d.cachedSegments, "encoded_segments": d.encodedSegments,
                             "expected_new_segments": expectedNew.count, "observed_new_segments": observedNew.count,
                             "missed_set_matches_expected": observedNew == expectedNew])
                seenKeys.formUnion(thisKeys)
            }
        }
        report["b_multiturn"] = ["pass": rows.allSatisfy { ($0["exact"] as? Bool == true) && ($0["missed_set_matches_expected"] as? Bool == true) },
                                 "requests": rows]
    }

    // (c) eight concurrent encodes of the same payload through one fresh cache.
    do {
        let p = prepared["concurrent"]!
        let cache = try TemplateTokenCache(capacityBytes: 256 * 1024 * 1024, tokenizer: tokenizer, modelDir: modelDir)
        let n = 8
        let box = TokCacheCheckBox(); box.ids = Array(repeating: [], count: n)
        let ready = DispatchSemaphore(value: 0), start = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
        let errLock = NSLock()
        for t in 0..<n {
            Thread {
                ready.signal(); start.wait()
                let (ids, _) = cache.encode(rendered: p.rendered)
                box.ids[t] = ids
                if ids != p.oneShotIds { errLock.lock(); box.error = "thread \(t) ids differ"; errLock.unlock() }
                done.signal()
            }.start()
        }
        for _ in 0..<n { ready.wait() }
        let t0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<n { start.signal() }
        for _ in 0..<n { done.wait() }
        let wallMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let s = cache.stats()
        let allEq = box.error == nil && box.ids.allSatisfy { $0 == p.oneShotIds }
        let ok = allEq && s.misses == 1 && s.inflightWaits == n - 1 && wallMs <= 1500
        if !ok { failures.append("c: misses=\(s.misses) waits=\(s.inflightWaits) wall_ms=\(wallMs) ids_equal=\(allEq) err=\(box.error ?? "nil")") }
        report["c_concurrent"] = ["pass": ok, "threads": n, "wall_ms": wallMs, "ids_equal": allEq,
                                  "misses": s.misses, "inflight_waits": s.inflightWaits,
                                  "cache_stats": tokCacheStatsDict(cache)]
    }

    // (d) capacity below one payload's bytes: ids stay exact, evictions > 0.
    do {
        let cache = try TemplateTokenCache(capacityBytes: 1_000_000, tokenizer: tokenizer, modelDir: modelDir)
        var rows: [[String: Any]] = []
        var allEq = true
        for name in ["E", "F", "F0"].sorted() {
            guard let p = prepared[name] else { continue }
            let (ids, _) = cache.encode(rendered: p.rendered)
            let (eq, idx, cv, rv) = tokIdsEqualDetail(ids, p.oneShotIds)
            if !eq { allEq = false; failures.append("d:\(name):first_diff index=\(idx) cached=\(cv) reference=\(rv)") }
            rows.append(["name": name, "tokens": ids.count, "exact": eq])
        }
        let s = cache.stats()
        let ok = allEq && s.evictions > 0
        if !ok { failures.append("d: exact=\(allEq) evictions=\(s.evictions) bytes=\(s.bytes)") }
        report["d_capacity"] = ["pass": ok, "capacity_bytes": 1_000_000, "encodes": rows,
                                "evictions": s.evictions, "cache_stats": tokCacheStatsDict(cache)]
    }

    // (e) disabled path: one-shot ids only; a fresh cache stays at zero.
    do {
        let p = prepared["concurrent"]!
        let cache = try TemplateTokenCache(capacityBytes: 256 * 1024 * 1024, tokenizer: tokenizer, modelDir: modelDir)
        let ids = p.oneShotIds  // disabled path is the original applyChatTemplate call itself
        let s = cache.stats()
        let zero = s.entries == 0 && s.bytes == 0 && s.hits == 0 && s.misses == 0 && s.inflightWaits == 0 && s.evictions == 0
        let ok = ids == p.oneShotIds && zero
        if !ok { failures.append("e: zero_counters=\(zero)") }
        report["e_disabled"] = ["pass": ok, "tokens": ids.count, "counters_zero": zero]
    }

    report["instrument"] = "h42-tokbench-cache-check-v1"
    report["status"] = failures.isEmpty ? "PASS_H42_CACHE_CHECK" : "FAIL"
    report["failures"] = failures
    report["tokenizer_load_ms"] = tokenizerLoadMs
    report["scope"] = "tokenizer-only host cache gate; no weights/GPU; not a TPS or serving claim"
    let out = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    FileHandle.standardOutput.write(out)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    if !failures.isEmpty { throw EngineError.invalid("cache-check failed: \(failures.joined(separator: "; "))") }
}

private func tokCacheStatsDict(_ cache: TemplateTokenCache) -> [String: Any] {
    let s = cache.stats()
    return ["entries": s.entries, "bytes": s.bytes, "hits": s.hits, "misses": s.misses,
            "inflight_waits": s.inflightWaits, "evictions": s.evictions,
            "segment_hits": s.segmentHits, "segment_misses": s.segmentMisses]
}


// H43 isolated diagnostic. The production callback is byte-identical to B32; this only
// wall-times it. One cold shared seed (H35/H38 priming verbatim), then per (R, policy):
// clone R rows from the seed, prime to L-4r, register exactly as h35SharedArmOnOwner,
// run W+N fixed-membership dispatches of the real runBatchedStep timed by
// mach_absolute_time. Policies: always-K3 (production draft depth) and plain-with-head
// (mode "never" -> K=0; the pool still carries spec state, so the plain step keeps the
// head alive exactly as production's plain-with-head branch does).
func h43SharedTiming(_ o: Options, model qm: Qwen4ExpModel, corpus: [Int],
                     corpusSHA256: String) throws {
    let prompt = try o.int("--prompt-tokens", 261600)
    let chunk = try o.int("--prefill-step", 1024)
    let maxRows = try o.int("--ownership-shared-rows", 8)
    let warmup = try o.int("--timing-warmup", 8)
    let dispatches = try o.int("--timing-dispatches", 64)
    let schedule = o.string("--timing-schedule", "fixed")
    // P106 H48: paired pool-mode arms in one process ("0,1,1,0"), a subset of the fixed-membership
    // row counts / policies, and an optional final logical-state digest per row
    let rrkvArms = o.string("--timing-rrkv-arms", "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    let rowList = o.string("--timing-rows", "8,4,1").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let policyFilter = Set(o.string("--timing-policies", "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
    // `Options.int` refuses 0; these flags take 0 legitimately (equal row lengths, digest/suffix off)
    let finalDigest = o.string("--timing-final-digest", "0") == "1"
    guard let rowSpacing = Int(o.string("--timing-row-spacing", "4")) else { throw EngineError.invalid("--timing-row-spacing must be an integer") }
    let distinctSuffix = o.string("--timing-distinct-suffix", "0") == "1"
    let env = ProcessInfo.processInfo.environment
    guard rrkvArms.allSatisfy({ $0 == "0" || $0 == "1" }), !rowList.isEmpty, rowList.allSatisfy({ (1...8).contains($0) }),
          (0...16).contains(rowSpacing), rowSpacing % 4 == 0 || distinctSuffix,
          policyFilter.isSubset(of: ["", "always_k3", "plain"]),
          try o.int("--batch", 8) == maxRows, maxRows == 8, (8192...261600).contains(prompt),
          chunk == 1024, warmup >= 0, warmup + dispatches <= 512, dispatches > 0,
          ["fixed", "churn"].contains(schedule), schedule != "churn" || dispatches == 112,
          let mtp = qm.mtp, qm.configuration.text.indexerCompressRatio == 4,
          env["ENGINE_BATCH_MTP_POLICY"] == "always", batchMTPPolicy == "always",
          env["ENGINE_RELEASE_POOLED_PRIVATE_HISTORY"] == "1", releasePooledPrivateHistory,
          env["ENGINE_H25_CANONICAL_PREFIX"] == "1", h25CanonicalPrefix,
          env["ENGINE_RESTACK_EVAL"] == nil, env["ENGINE_INDEXER_ROUTE_WITNESS"] == nil,
          env["ENGINE_OWNER_ORDER_TRACE"] == nil, env["ENGINE_IDX_REPLAY_DUMP"] == nil,
          env["ENGINE_IDX_REPLAY_BUDGETS"] == nil, !batchRestackEval,
          !Qwen4ExpIndexerRouteWitness.enabled, Qwen4ExpCacheCapacity.kvStep == 256,
          Qwen4ExpCacheCapacity.indexerStep == 1024, corpus.count >= prompt + 1 else {
        throw EngineError.invalid("H43 requires --batch 8 --ownership-shared-rows 8, 8192<=prompt<=261600, C1024, MTP3 always, canonical-prefix/release 1, trace-free E9 profile")
    }
    let owner = ModelThread(minBatch: 4, gatherWindow: 0.025, maxBatch: 8) { _ in
        preconditionFailure("H43 invokes the actual callback directly on its owner")
    }
    modelThreadShared = owner
    defer { owner.shutdown() }
    let result: (String, String?) = owner.exclusive {
        do { return (try h43SharedTimingOnOwner(qm, mtp: mtp, tokens: corpus, prompt: prompt,
                     chunk: chunk, corpusSHA256: corpusSHA256, maxRows: maxRows,
                     warmup: warmup, dispatches: dispatches, schedule: schedule,
                     rrkvArms: rrkvArms.map { $0 == "1" }, rowList: rowList,
                     policyFilter: policyFilter.subtracting([""]), finalDigest: finalDigest,
                     rowSpacing: rowSpacing, distinctSuffix: distinctSuffix), nil) }
        catch { return ("", String(describing: error)) }
    }
    if let error = result.1 { throw EngineError.invalid(error) }
    print(result.0)
}

private func h43SharedTimingOnOwner(_ qm: Qwen4ExpModel, mtp: Qwen4ExpMTP,
        tokens: [Int], prompt L: Int, chunk: Int, corpusSHA256: String,
        maxRows Rmax: Int, warmup W: Int, dispatches N: Int, schedule: String,
        rrkvArms: [Bool] = [], rowList: [Int] = [8, 4, 1], policyFilter: Set<String> = [],
        finalDigest: Bool = false, rowSpacing: Int = 4, distinctSuffix: Bool = false) throws -> String {
    func need(_ ok: Bool, _ why: String) throws { if !ok { throw EngineError.invalid("H43 INSTRUMENTFAIL: " + why) } }
    try need(modelThreadShared.isOwner && seqRegistry.count == 0 && batchPoolShared.cache == nil, "fresh owner")
    defer {
        batchMTPPolicy = "always"; batchMTPDepthShared = 0
        Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
    }
    Qwen4ExpCacheCapacity.configureGrowthLimit(262144)
    let ratio = qm.configuration.text.indexerCompressRatio
    let physicalLimit = Double(ProcessInfo.processInfo.physicalMemory) * 0.90
    func memoryGuard() throws {
        try need(Double(Memory.activeMemory + Memory.cacheMemory) + 48e9 <= physicalLimit, "90% physical plus48GB workspace")
    }
    func log(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
    var tb = mach_timebase_info_data_t()
    mach_timebase_info(&tb)
    let ticksToMs = Double(tb.numer) / Double(tb.denom) / 1e6
    let sharedLength = ((L - 1 - 4 * (Rmax - 1)) / chunk) * chunk
    try need(sharedLength >= chunk && sharedLength < L - 4 * (Rmax - 1), "seed geometry")
    var copiedBytes = 0, copiedArrays = 0
    func detached(_ a: MLXArray) -> MLXArray {
        autoreleasepool {
            let bytes = a.asData(access: .copy)
            let copy = MLXArray(data: bytes)
            eval(copy)
            let sourcePointer = a.asData(access: .noCopy).data.withUnsafeBytes { Int(bitPattern: $0.baseAddress!) }
            let copyData = copy.asData(access: .noCopy)
            let copyPointer = copyData.data.withUnsafeBytes { Int(bitPattern: $0.baseAddress!) }
            precondition(sourcePointer != copyPointer, "diagnostic clone aliases source")
            precondition(copyData.data == bytes.data, "diagnostic clone altered stored bits")
            copiedBytes += a.nbytes; copiedArrays += 1
            return copy
        }
    }
    func capture(_ cache: [KVCache], head: KVCache) -> [String: MLXArray] {
        var d: [String: MLXArray] = [:]
        for (prefix, cs) in [("trunk.", cache), ("mtp.", [head])] {
            qm.exportCaches(cs, prefix: prefix, into: &d)
            for (i, c) in cs.enumerated() {
                guard let list = c as? CacheList, let kv = list[0] as? KVCacheSimple,
                      let idx = list[1] as? ArraysCache else { continue }
                d["\(prefix)L\(i).k"] = kv.rawKeys!; d["\(prefix)L\(i).v"] = kv.rawValues!
                d["\(prefix)L\(i).i0"] = idx[0]!
                d["\(prefix)L\(i).i1"] = idx[1]!
            }
        }
        eval(Array(d.values))
        return d
    }
    func clone(_ d: [String: MLXArray], offset: Int) throws -> ([KVCache], KVCache) {
        let copied = Dictionary(uniqueKeysWithValues: d.keys.sorted().map { ($0, detached(d[$0]!)) })
        let cache = qm.newCache(parameters: nil), head = mtp.newCache()
        for (prefix, cs) in [("trunk.", cache), ("mtp.", [head])] {
            qm.importCaches(cs, prefix: prefix, from: copied)
            for (i, c) in cs.enumerated() {
                guard let list = c as? CacheList, let kv = list[0] as? KVCacheSimple else { continue }
                kv.setBuffers(keys: copied["\(prefix)L\(i).k"]!, values: copied["\(prefix)L\(i).v"]!, offset: offset)
            }
        }
        try memoryGuard()
        return (cache, head)
    }
    func advance(_ cache: [KVCache], _ head: KVCache, from: Int, to: Int,
                 final: Bool, label: String, source: [Int]? = nil) throws -> (Int, BatchRowSpec?) {
        let src = source ?? tokens
        var at = from, n0 = 0
        var lastMixed: MLXArray?, lastHidden: MLXArray?
        while at < to {
            let end = min(to, at + chunk)
            let input = MLXArray(src[at ..< end].map { Int32($0) }).reshaped([1, end - at])
            let (logits, hidden) = qm.forwardHidden(input, cache: cache)
            var shifted = Array(src[(at + 1) ..< (end + 1)])
            if final && end == to {
                n0 = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                shifted[shifted.count - 1] = n0
            }
            let (mixed, h) = mtp(hidden: hidden,
                tokens: MLXArray(shifted.map { Int32($0) }).reshaped([1, end - at]),
                embed: qm.model.embedTokens, cache: head)
            eval(mixed, h, cache.flatMap { $0.state }, head.state)
            lastMixed = mixed; lastHidden = h; at = end
            try memoryGuard()
            if at % 16384 == 0 || at == to { log("mtp-capacity \(label) prefilled=\(at) active=\(Memory.activeMemory)") }
        }
        guard final, let mixed = lastMixed, let hidden = lastHidden else { return (n0, nil) }
        let draft = qm.draftToken(mixed[0..., -1, 0...])
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        eval(draft, last)
        return (n0, BatchRowSpec(mtpCache: head, headLen: to, d1: draft.item(Int.self), Slast: last))
    }
    func makeSharedSeed() throws -> [String: MLXArray] {
        qm.resetRaggedState(); qm.resetHeadRaggedState()
        let cache = qm.newCache(parameters: nil), head = mtp.newCache()
        _ = try advance(cache, head, from: 0, to: sharedLength, final: false, label: "H43_shared_seed")
        return capture(cache, head: head)
    }
    let storage = qm.prepareServingStorage()
    try memoryGuard()
    // Direct calls keep ModelThread.stepCount=0; the stats line inside runBatchedStep
    // therefore executes every dispatch, so its real dependency must exist.
    hotStoreShared = HotPrefixStore(capBytes: 20_000_000_000, rungStep: 512)
    defer { hotStoreShared = nil }
    let seedT0 = mach_absolute_time()
    let seed = try makeSharedSeed()
    let seedMs = Double(mach_absolute_time() - seedT0) * ticksToMs
    let seedBytes = seed.values.reduce(0) { $0 + $1.nbytes }
    log("H43 seed=\(sharedLength) wall_ms=\(seedMs) active=\(Memory.activeMemory)")
    // P106 H48: every arm below runs from the SAME seed in this process. `rrkvArms` empty = one arm in
    // the process's current pool mode (the H43/H45 report shapes, unchanged); non-empty = one arm per
    // entry with `rowResidentKV` set for it (flipped only while no pool stands), wrapped in an
    // `h48-row-resident-ab-v1` report. Per-dispatch committed token ids and (optionally) a SHA-256 of
    // every row's final logical state make each arm a value witness of the others at identical
    // composition (the H47 G2 lesson: compare only at identical batch composition).
    let originalMode = rowResidentKV
    defer { rowResidentKV = originalMode }
    func rowDigest(_ caches: [KVCache], head: BatchRowSpec?, length: Int) throws -> String {
        var h = SHA256()
        func add(_ name: String, _ a: MLXArray) {
            h.update(data: Data(name.utf8))
            h.update(data: Data("\(a.shape)|\(a.dtype)".utf8))
            h.update(data: a.asData(access: .copy).data)
        }
        func addCaches(_ cs: [KVCache], prefix: String, len: Int) throws {
            for (i, c) in cs.enumerated() {
                if let list = c as? CacheList {
                    let kv = list[0] as! KVCacheSimple, idx = list[1] as! ArraysCache
                    try need(kv.offset == len && idx[0] != nil && idx[1] != nil && idx[0]!.dim(1) >= len && idx[1]!.dim(1) >= len / ratio,
                             "final logical extent \(prefix)L\(i) offset \(kv.offset) len \(len)")
                    add("\(prefix)L\(i).k", kv.state[0]); add("\(prefix)L\(i).v", kv.state[1])
                    add("\(prefix)L\(i).i0", idx[0]![0..., 0 ..< len, 0...])
                    add("\(prefix)L\(i).i1", idx[1]![0..., 0 ..< (len / ratio), 0...])
                } else if let a = c as? ArraysCache {
                    for s in 0 ..< 4 { if let x = a[s] { add("\(prefix)L\(i).s\(s)", x) } }
                }
            }
        }
        try addCaches(caches, prefix: "trunk.", len: length)
        if let head {
            try addCaches([head.mtpCache], prefix: "mtp.", len: head.headLen)
            add("slast", head.Slast)
            h.update(data: Data("d1=\(head.d1) headLen=\(head.headLen)".utf8))
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }
    /// Row r's token stream: the shared seed prefix, then either the corpus continuation (the H43/H45
    /// geometry) or -- `distinctSuffix` -- a row-specific span copied from earlier in the corpus, so rows of
    /// EQUAL length still differ in content (a swapped or aliased row would change their tokens).
    func rowSource(_ row: Int, length: Int) -> [Int] {
        guard distinctSuffix else { return tokens }
        let n = length - sharedLength + 1
        let from = 1 + (row * 9973) % max(1, tokens.count - n - 1)
        precondition(from + n <= tokens.count, "distinct suffix inside the corpus")
        return Array(tokens[0 ..< sharedLength]) + Array(tokens[from ..< (from + n)])
    }
    func primeRows(_ lengths: [Int], label: String, keys: inout [Int], committed: inout [Int], pending: inout [Int]) throws {
        for row in 0..<lengths.count {
            qm.resetRaggedState(); qm.resetHeadRaggedState()
            let (cache, head) = try clone(seed, offset: sharedLength)
            let source = rowSource(row, length: lengths[row])
            let (n0, spec) = try advance(cache, head, from: sharedLength, to: lengths[row],
                                         final: true, label: "\(label)_row\(row)", source: source)
            guard let spec else { throw EngineError.invalid("H43 missing primed head") }
            let ids = Array(source.prefix(lengths[row]))
            let key = seqRegistry.create { SeqState(sampler: EngineSampler(), cache: cache, xs: MLXArray(0), ids: ids) }
            keys.append(key)
            let s = seqRegistry[key]
            s.row.spec = spec
            s.retiredMTPCache = spec.mtpCache
            s.row.length = lengths[row]; s.row.pending = n0
            s.row.step = { $0.argMax(axis: -1).item(Int.self) }
            s.row.greedy = true; s.row.biasNow = { 0 }; s.row.thinkOpenNow = { false }; s.row.stopIds = []
            committed.append(lengths[row]); pending.append(n0)
            try memoryGuard()
        }
    }
    /// Dissolve, digest (optional), destroy. Returns per-row final records.
    func teardown(_ keys: [Int], committed: [Int], pending: [Int]) throws -> [[String: Any]] {
        batchPoolShared.dissolve(qm, ratio: ratio)
        var finals: [[String: Any]] = []
        for (row, key) in keys.enumerated() {
            let s = seqRegistry[key]
            s.cache = s.row.caches
            var rec: [String: Any] = ["row": row, "length": committed[row], "pending": pending[row]]
            if finalDigest {
                let d0 = mach_absolute_time()
                rec["state_sha256"] = try rowDigest(s.cache, head: s.row.spec, length: committed[row])
                rec["digest_ms"] = Double(mach_absolute_time() - d0) * ticksToMs
            }
            finals.append(rec)
        }
        for key in keys { seqRegistry.destroy(key) }
        batchMTPPolicy = "always"; batchMTPDepthShared = 0
        try memoryGuard()
        try need(seqRegistry.count == 0 && batchPoolShared.rows.isEmpty && Qwen4ExpBatch.rowResidentRegistered == 0, "arm teardown")
        return finals
    }
    func churnReport() throws -> [String: Any] {
        let invariantBefore = (batchLengthMismatches, hotRungMismatches, batchSpecDropped)
        // H45: one R=8 always-K3 arm; membership churns mid-run so every transition pays
        // dissolve + re-stack inside the timed dispatch. Retained rows keep their registry
        // entry and unstacked state, exactly as H20's departed rows do.
        let R = Rmax
        let lengths = (0..<R).map { L - rowSpacing * $0 }
        let membership: [[Int]] = (0..<N).map { d in
            switch d {
            case 16..<32: return Array(0..<7)
            case 48..<64: return Array(0..<4)
            case 80..<96: return Array(1..<8)
            default: return Array(0..<8)
            }
        }
        var keys: [Int] = []
        var committed = [Int](), pending = [Int]()
        var timed: [[String: Any]] = []
        var warmupMs: [Double] = []
        var status = "COMPLETE", failure: String? = nil
        let restacksBefore = batchRestacks
        do {
            batchMTPDepthShared = 3
            batchMTPPolicy = "always"
            try primeRows(lengths, label: "H45_churn", keys: &keys, committed: &committed, pending: &pending)
            log("H45 churn R=\(R) primed; W+N=\(W + N) dispatches row_resident=\(rowResidentKV)")
            var prevMembers = Array(0..<R)
            for d in 0..<(W + N) {
                let members = d < W ? Array(0..<R) : membership[d - W]
                let requests = members.map { StepRequest(key: keys[$0], pending: pending[$0], length: committed[$0]) }
                let roundsBefore = batchMTPRounds, rsBefore = batchRestacks, bytesBefore = batchRestackLogicalBytes
                let t0 = mach_absolute_time()
                let first = runBatchedStep(requests, model: qm, qm: qm, ratio: ratio)
                let ms = Double(mach_absolute_time() - t0) * ticksToMs
                let executedK = batchMTPRounds == roundsBefore + 1 ? 3 : 0
                try need(executedK == 3, "churn arm executed K (K \(executedK))")
                var perRowTokens: [Int] = [], perRowAccepted: [Int] = [], committedTokens: [[Int]] = []
                for (i, row) in members.enumerated() {
                    let state = seqRegistry[keys[row]]
                    let queued = state.row.takeQueued()
                    let toks = [first[i]] + queued
                    try need(!toks.isEmpty && toks.allSatisfy { $0 >= 0 } && toks.count <= executedK + 1,
                             "invalid committed token stream")
                    perRowTokens.append(toks.count)
                    perRowAccepted.append(toks.count - 1)
                    committedTokens.append(toks)
                    pending[row] = toks.last!; committed[row] += toks.count
                }
                for row in 0..<R where !members.contains(row) {
                    let state = seqRegistry[keys[row]]
                    _ = state.row.takeQueued()
                    try need(state.row.length == committed[row], "retained row \(row) length drift")
                }
                if d >= W {
                    let membershipChanged = members != prevMembers
                    var rec: [String: Any] = ["dispatch": d - W, "ms": ms, "K": executedK,
                        "members": members, "membership_changed": membershipChanged,
                        "tokens_per_row": perRowTokens, "accepted_per_row": perRowAccepted,
                        "committed_tokens": committedTokens,
                        "active_memory": Memory.activeMemory,
                        "restacks_delta": batchRestacks - rsBefore,
                        "restack_logical_bytes_delta": batchRestackLogicalBytes - bytesBefore]
                    if membershipChanged {
                        let departed = prevMembers.filter { !members.contains($0) }
                        var departedTrunkBytes = 0, departedSpecBytes = 0
                        for row in departed {
                            let state = seqRegistry[keys[row]]
                            departedTrunkBytes += state.row.caches.flatMap { $0.state }.reduce(0) { $0 + $1.nbytes }
                            if let spec = state.row.spec {
                                departedSpecBytes += spec.mtpCache.state.reduce(0) { $0 + $1.nbytes } + spec.Slast.nbytes
                            }
                        }
                        // The lazy stack copy already ran inside the timed dispatch; this eval
                        // prices whatever is left (expected ~0) and sizes the stacked pool.
                        var stackArrays = batchPoolShared.cache?.flatMap { $0.state } ?? []
                        if let sp = batchPoolShared.specPool { stackArrays += sp.headCache.state + [sp.Slast] }
                        let stackBytes = stackArrays.reduce(0) { $0 + $1.nbytes }
                        let e0 = mach_absolute_time()
                        eval(stackArrays)
                        rec["transition"] = "B\(prevMembers.count)->B\(members.count)"
                        rec["departed_rows"] = departed
                        rec["departed_trunk_bytes"] = departedTrunkBytes
                        rec["departed_spec_bytes"] = departedSpecBytes
                        rec["stack_bytes"] = stackBytes
                        rec["stack_eval_ms"] = Double(mach_absolute_time() - e0) * ticksToMs
                    }
                    timed.append(rec)
                    prevMembers = members
                } else {
                    warmupMs.append(ms)
                }
                try memoryGuard()
            }
        } catch {
            failure = String(describing: error)
            status = (failure?.contains("48GB") == true || failure?.contains("workspace") == true)
                ? "INSTRUMENT_BLOCKED_R\(R)" : "FAILED"
        }
        let finals = try teardown(keys, committed: committed, pending: pending)
        if let failure, status == "FAILED" { throw EngineError.invalid("H45 churn: \(failure)") }
        try need(batchLengthMismatches == invariantBefore.0 && hotRungMismatches == invariantBefore.1 &&
                 batchSpecDropped == invariantBefore.2, "serving invariant counters")
        var report: [String: Any] = ["instrument": "h45-churn-timing-v1", "status": status,
            "prompt_tokens": L, "chunk": chunk, "max_rows": Rmax,
            "shared_seed_length": sharedLength, "shared_seed_logical_bytes": seedBytes,
            "seed_wall_ms": seedMs, "cold_shared_prefills": 1,
            "corpus_sha256": corpusSHA256, "warmup": W, "dispatches_requested": N,
            "owner_execution": modelThreadShared.isOwner, "schedule": "churn",
            "policy": "always_k3", "row_lengths": lengths,
            "membership_schedule": membership,
            "copied_bytes": copiedBytes, "copied_arrays": copiedArrays,
            "serving_storage": storage,
            "restacks_total": batchRestacks - restacksBefore]
        if let failure { report["failure"] = failure }
        report["release_requested"] = true
        report["release_applied"] = releasePooledPrivateHistory
        report["row_resident"] = rowResidentKV && Qwen4ExpModel.rowResidentEligible
        report["invariants"] = ["length_mismatches": batchLengthMismatches, "rung_mismatches": hotRungMismatches,
                                "specs_dropped": batchSpecDropped]
        report["warmup_ms"] = warmupMs
        report["per_dispatch"] = timed
        report["final_rows"] = finals
        report["scope"] = "wall-time census of runBatchedStep under membership churn; per-dispatch ms + restack bytes + post-dispatch stack eval; no TPS/HTTP/deployment claim"
        return report
    }
    func fixedReport() throws -> [String: Any] {
        let invariantBefore = (batchLengthMismatches, hotRungMismatches, batchSpecDropped)
        var arms: [[String: Any]] = []
        let policies = [("always_k3", "always"), ("plain", "never")].filter { policyFilter.isEmpty || policyFilter.contains($0.0) }
        for R in rowList {
            let lengths = (0..<R).map { L - rowSpacing * $0 }
            for (policyName, policyMode) in policies {
                var keys: [Int] = []
                var committed = [Int](), pending = [Int]()
                var status = "COMPLETE", failure: String? = nil
                var timed: [[String: Any]] = []
                var warmupMs: [Double] = []
                let restacksBefore = batchRestacks
                do {
                    batchMTPDepthShared = 3
                    batchMTPPolicy = policyMode
                    try primeRows(lengths, label: "H43_R\(R)_\(policyName)", keys: &keys, committed: &committed, pending: &pending)
                    log("H43 R=\(R) policy=\(policyName) primed; W+N=\(W + N) dispatches row_resident=\(rowResidentKV)")
                    for d in 0..<(W + N) {
                        let requests = (0..<R).map { StepRequest(key: keys[$0], pending: pending[$0], length: committed[$0]) }
                        let roundsBefore = batchMTPRounds
                        let t0 = mach_absolute_time()
                        let first = runBatchedStep(requests, model: qm, qm: qm, ratio: ratio)
                        let ms = Double(mach_absolute_time() - t0) * ticksToMs
                        let executedK = batchMTPRounds == roundsBefore + 1 ? 3 : 0
                        try need(executedK == ((policyMode == "always" && R > 1) ? 3 : 0),
                                 "policy arm executed K (mode \(policyMode) R \(R) K \(executedK))")
                        var perRowTokens: [Int] = [], perRowAccepted: [Int] = [], committedTokens: [[Int]] = []
                        for (i, row) in (0..<R).enumerated() {
                            let state = seqRegistry[keys[row]]
                            let queued = state.row.takeQueued()
                            let toks = [first[i]] + queued
                            try need(!toks.isEmpty && toks.allSatisfy { $0 >= 0 } && toks.count <= executedK + 1,
                                     "invalid committed token stream")
                            perRowTokens.append(toks.count)
                            perRowAccepted.append(executedK == 0 ? 0 : toks.count - 1)
                            committedTokens.append(toks)
                            pending[row] = toks.last!; committed[row] += toks.count
                        }
                        if d >= W {
                            timed.append(["dispatch": d - W, "ms": ms, "K": executedK,
                                "tokens_per_row": perRowTokens, "accepted_per_row": perRowAccepted,
                                "committed_tokens": committedTokens,
                                "active_memory": Memory.activeMemory])
                        } else {
                            warmupMs.append(ms)
                        }
                        try memoryGuard()
                    }
                } catch {
                    failure = String(describing: error)
                    status = (failure?.contains("48GB") == true || failure?.contains("workspace") == true)
                        ? "INSTRUMENT_BLOCKED_R\(R)" : "FAILED"
                }
                let finals = try teardown(keys, committed: committed, pending: pending)
                if let failure, status == "FAILED" { throw EngineError.invalid("H43 R\(R) \(policyName): \(failure)") }
                let mss = (timed.map { $0["ms"] as! Double }).sorted()
                var summary: [String: Any] = ["status": status]
                if let failure { summary["failure"] = failure }
                if !mss.isEmpty {
                    let totalTokens = timed.flatMap { $0["tokens_per_row"] as! [Int] }.reduce(0, +)
                    let totalMs = mss.reduce(0, +)
                    let accepted = timed.flatMap { $0["accepted_per_row"] as! [Int] }
                    summary["median_ms"] = mss[mss.count / 2]
                    summary["p10_ms"] = mss[mss.count / 10]
                    summary["p90_ms"] = mss[mss.count * 9 / 10]
                    summary["mean_ms"] = totalMs / Double(mss.count)
                    summary["tokens_per_dispatch_mean"] = Double(totalTokens) / Double(mss.count)
                    summary["tok_per_s"] = Double(totalTokens) / (totalMs / 1000.0)
                    summary["accepted_per_row_mean"] = Double(accepted.reduce(0, +)) / Double(accepted.count)
                    summary["warmup_ms"] = warmupMs
                    summary["restacks"] = batchRestacks - restacksBefore
                }
                arms.append(["row_count": R, "row_lengths": lengths, "policy": policyName,
                             "policy_mode": policyMode, "warmup": W, "dispatches": timed.count,
                             "summary": summary, "per_dispatch": timed, "final_rows": finals])
                log("H43 R=\(R) policy=\(policyName) status=\(status) active=\(Memory.activeMemory)")
            }
        }
        try need(batchLengthMismatches == invariantBefore.0 && hotRungMismatches == invariantBefore.1 &&
                 batchSpecDropped == invariantBefore.2, "serving invariant counters")
        var report: [String: Any] = ["instrument": "h43-shared-timing-v1", "status": "COMPLETE",
            "prompt_tokens": L, "chunk": chunk, "max_rows": Rmax,
            "shared_seed_length": sharedLength, "shared_seed_logical_bytes": seedBytes,
            "seed_wall_ms": seedMs, "cold_shared_prefills": 1,
            "corpus_sha256": corpusSHA256, "warmup": W, "dispatches_requested": N,
            "owner_execution": modelThreadShared.isOwner,
            "row_order": rowList, "policy_order": policies.map { $0.0 },
            "copied_bytes": copiedBytes, "copied_arrays": copiedArrays,
            "serving_storage": storage]
        report["release_requested"] = true
        report["release_applied"] = releasePooledPrivateHistory
        report["row_resident"] = rowResidentKV && Qwen4ExpModel.rowResidentEligible
        report["r1_note"] = "B1 takes the lone-step branch; both policy labels execute the identical armed keep-alive step there (no pool, no verify block)"
        report["invariants"] = ["length_mismatches": batchLengthMismatches, "rung_mismatches": hotRungMismatches,
                                "specs_dropped": batchSpecDropped]
        report["scope"] = "wall-time census of the real runBatchedStep at fixed membership on H35/H38-primed rows; mach_absolute_time per dispatch; no TPS/HTTP/deployment claim"
        report["arms"] = arms
        return report
    }
    let single: [String: Any]
    if rrkvArms.isEmpty {
        single = schedule == "churn" ? try churnReport() : try fixedReport()
        let data = try JSONSerialization.data(withJSONObject: single, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
    var armReports: [[String: Any]] = []
    for (a, mode) in rrkvArms.enumerated() {
        try need(batchPoolShared.cache == nil && seqRegistry.count == 0 && Qwen4ExpBatch.rowResidentRegistered == 0, "arm boundary")
        rowResidentKV = mode
        // every arm starts from the same allocator state: no cached buffers carried over from the previous arm
        Memory.clearCache()
        let boundary: [String: Any] = ["active_bytes": Memory.activeMemory, "cache_bytes": Memory.cacheMemory]
        let installsBefore = rowResidentPoolInstalls, clonesBefore = rowResidentHeadClones, repairsBefore = rowResidentAliasRepairs
        let copiedBefore = (copiedBytes, copiedArrays)
        let a0 = mach_absolute_time()
        log("H48 arm \(a) row_resident=\(mode) begin")
        var arm: [String: Any] = ["arm": a, "row_resident_requested": mode,
                                  "row_resident_applied": mode && Qwen4ExpModel.rowResidentEligible]
        arm["report"] = schedule == "churn" ? try churnReport() : try fixedReport()
        arm["row_resident_pool_installs"] = rowResidentPoolInstalls - installsBefore
        arm["row_resident_head_clones"] = rowResidentHeadClones - clonesBefore
        arm["row_resident_alias_repairs"] = rowResidentAliasRepairs - repairsBefore
        arm["arm_wall_ms"] = Double(mach_absolute_time() - a0) * ticksToMs
        arm["boundary_memory"] = boundary
        arm["copied_bytes"] = copiedBytes - copiedBefore.0
        arm["copied_arrays"] = copiedArrays - copiedBefore.1
        armReports.append(arm)
    }
    let report: [String: Any] = ["instrument": "h48-row-resident-ab-v1", "status": "COMPLETE",
        "schedule": schedule, "prompt_tokens": L, "arms_requested": rrkvArms,
        "shared_seed_length": sharedLength, "seed_wall_ms": seedMs, "corpus_sha256": corpusSHA256,
        "final_digest": finalDigest, "row_spacing": rowSpacing, "distinct_suffix": distinctSuffix, "arms": armReports,
        "scope": "paired same-process same-seed arms of the pool mode; per-dispatch wall ms and committed token ids at identical membership; optional final logical-state SHA-256 per row; no TPS/HTTP/deployment claim"]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

import Foundation

/// Inference speed sweep over input length, output length, and batch size,
/// plus a batched-correctness check against batch = 1.
func runBenchmark(engine: MetalEngine, st: Safetensors, config: ModelConfig, tokenizer: Tokenizer) {
    fputs("loading weights...\n", stderr)
    let weights = ModelWeights(engine: engine, st: st, config: config)

    // ---- correctness: batch B with identical prompts must equal batch 1 ----
    let promptIds = tokenizer.encode("The capital of France is")
    func greedy(_ m: BatchModel, _ ids: [Int], _ n: Int) -> [[Int]] {
        m.reset()
        let B = m.batch
        var next = [Int](repeating: 0, count: B)
        for (p, t) in ids.enumerated() { next = m.forward(tokens: Array(repeating: t, count: B), pos: p) }
        var seqs = next.map { [$0] }
        var pos = ids.count
        for _ in 1..<n {
            next = m.forward(tokens: seqs.map { $0.last! }, pos: pos)
            pos += 1
            for b in 0..<B { seqs[b].append(next[b]) }
        }
        return seqs
    }
    print("== correctness (batch > 1 must match batch 1, identical prompts) ==")
    let ref1 = greedy(BatchModel(engine: engine, weights: weights, batch: 1, maxT: 64), promptIds, 8)[0]
    print("  batch 1: \(ref1)")
    for B in [2, 4] {
        let seqs = greedy(BatchModel(engine: engine, weights: weights, batch: B, maxT: 64), promptIds, 8)
        let ok = seqs.allSatisfy { $0 == ref1 }
        print("  batch \(B): \(seqs[0]) \(ok ? "OK" : "MISMATCH")")
    }

    // ---- sweep ----
    let corpus = tokenizer.encode("The history of the Roman Empire is a long story of rise and fall, ambition and ruin. " +
                                  "In the beginning the city was small, and its neighbors were many. ")
    func prompt(_ n: Int) -> [Int] { (0..<n).map { corpus[$0 % corpus.count] } }

    let inputLens = [16, 64, 128, 256]
    let outputLens = [32, 128]
    let batches = [1, 2, 4]

    print("\n== sweep (tokens/s; prefill = prompt processing, decode = generation) ==")
    print("batch    in     out |  prefill t/s   decode t/s | prefill ms   decode ms/tok")
    for B in batches {
        for inLen in inputLens {
            for outLen in outputLens {
                let maxT = inLen + outLen + 16
                let m = BatchModel(engine: engine, weights: weights, batch: B, maxT: maxT)
                m.reset()
                let ids = prompt(inLen)

                // prefill
                let t0 = Date()
                var next = [Int](repeating: 0, count: B)
                for (p, t) in ids.enumerated() { next = m.forward(tokens: Array(repeating: t, count: B), pos: p) }
                let prefillT = Date().timeIntervalSince(t0)

                // decode
                let t1 = Date()
                var pos = inLen
                for _ in 0..<outLen {
                    next = m.forward(tokens: next, pos: pos)
                    pos += 1
                }
                let decodeT = Date().timeIntervalSince(t1)

                let prefillTps = Double(inLen * B) / prefillT
                let decodeTps = Double(outLen * B) / decodeT
                print(String(format: "%5d %6d %7d | %12.1f %12.1f | %10.2f %10.2f",
                             B, inLen, outLen, prefillTps, decodeTps,
                             prefillT * 1000 / Double(inLen), decodeT * 1000 / Double(outLen)))
            }
        }
    }
}

/// Chunked-prefill speed test.
func runPrefillBench(engine: MetalEngine, st: Safetensors, config: ModelConfig, tokenizer: Tokenizer) {
    fputs("loading weights (+ fp16 GEMM weights)...\n", stderr)
    let weights = ModelWeights(engine: engine, st: st, config: config, buildBF16: true)

    let optimized = ProcessInfo.processInfo.environment["HUSKY_PREFILL_REFERENCE"] != "1"
    let repeats = max(1, Int(ProcessInfo.processInfo.environment["HUSKY_REPEATS"] ?? "3") ?? 3)
    let check = tokenizer.encode("The capital of France is")
    let chunkSizes = (ProcessInfo.processInfo.environment["HUSKY_CHUNKS"] ?? "64,128,256").split(separator: ",").map { Int($0)! }
    let lengths = (ProcessInfo.processInfo.environment["HUSKY_LENGTHS"] ?? "64,128,256,512").split(separator: ",").map { Int($0)! }

    // correctness: chunked prefill must produce the same next token as token-by-token
    print("== chunked prefill correctness ==")
    for C in chunkSizes {
        let m = PrefillModel(engine: engine, weights: weights, chunk: C, maxT: 2048, optimized: optimized)
        let t = m.prefill(check)
        precondition(t == 11751, "prefill golden token mismatch")
        print("  chunk \(C): next token \(t) OK")
    }

    let corpus = tokenizer.encode("The history of the Roman Empire is a long story of rise and fall, ambition and ruin. " +
                                  "In the beginning the city was small, and its neighbors were many. ")
    func prompt(_ n: Int) -> [Int] { (0..<n).map { corpus[$0 % corpus.count] } }

    if PrefillModel.prof { print("DIAGNOSTIC MODE: serialized dispatches; TPS is not representative.") }
    print("\n== chunked prefill throughput (median of \(repeats), optimized=\(optimized)) ==")
    print("chunk   len | prefill t/s | total ms | ms/token")
    for C in chunkSizes {
        for n in lengths {
            let m = PrefillModel(engine: engine, weights: weights, chunk: C, maxT: n + 16, optimized: optimized)
            let ids = prompt(n)
            _ = m.prefill(ids) // compile/cache/clock warmup, excluded
            var samples: [Double] = []
            for _ in 0..<repeats {
                let t0 = DispatchTime.now().uptimeNanoseconds
                _ = m.prefill(ids)
                samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            }
            let dt = samples.sorted()[samples.count / 2]
            print(String(format: "%5d %6d | %11.1f | %8.1f | %8.3f",
                         C, n, Double(n) / dt, dt * 1000, dt * 1000 / Double(n)))
        }
    }
}

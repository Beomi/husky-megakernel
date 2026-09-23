import Foundation

setvbuf(stdout, nil, _IONBF, 0)

func argValue(_ name: String, default def: String? = nil) -> String? {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    return def
}

let modelDir = argValue("--model", default: "/tmp/woof")!
let prompt = argValue("--prompt", default: "The capital of France is")!
let idsArg = argValue("--ids")
let maxTokens = Int(argValue("--max-tokens", default: "16")!)!
let maxSeq = Int(argValue("--max-seq", default: "4096")!)!
let useMegakernel = CommandLine.arguments.contains("--megakernel")
let prefillChunk = Int(argValue("--prefill-chunk", default: "256")!)!
let ngroups = Int(argValue("--groups", default: "1")!)!

do {
    let t0 = Date()
    if CommandLine.arguments.contains("--check-kernels") {
        runKernelChecks(engine: try MetalEngine())
        exit(0)
    }
    if CommandLine.arguments.contains("--bench-hardware") {
        try runHardwareBench(engine: MetalEngine())
        exit(0)
    }
    let config = try ModelConfig.load(path: "\(modelDir)/config.json")
    let st = try Safetensors(path: "\(modelDir)/model.safetensors")
    let engine = try MetalEngine()
    fputs("device: \(engine.device.name), maxBufferLength \(engine.device.maxBufferLength / (1024*1024)) MB\n", stderr)
    let tokenizer = try Tokenizer(path: "\(modelDir)/tokenizer.json")

    if CommandLine.arguments.contains("--bench-megakernel") {
        runMegakernelBench(engine: engine, st: st, config: config, tokenizer: tokenizer)
        exit(0)
    }
    if CommandLine.arguments.contains("--check-prefill") {
        runPrefillChecks(engine: engine, st: st, config: config, tokenizer: tokenizer)
        exit(0)
    }
    if CommandLine.arguments.contains("--bench-prefill") {
        runPrefillBench(engine: engine, st: st, config: config, tokenizer: tokenizer)
        exit(0)
    }
    if CommandLine.arguments.contains("--bench") {
        runBenchmark(engine: engine, st: st, config: config, tokenizer: tokenizer)
        exit(0)
    }

    let promptIds: [Int]
    if let idsArg {
        promptIds = idsArg.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces))! }
    } else {
        promptIds = tokenizer.encode(prompt)
    }
    precondition(!promptIds.isEmpty && promptIds.count <= maxSeq && maxTokens > 0 && prefillChunk >= 0)
    precondition(promptIds.allSatisfy { $0 >= 0 && $0 < config.vocabSize })
    fputs("prompt ids: \(promptIds)\n", stderr)

    var generated: [Int] = []
    let genStart = Date()
    var next = 0

    if useMegakernel {
        if ngroups > 1 {
            fputs("WARNING: --groups > 1 requires a device-wide barrier with acquire/release memory\n" +
                  "         ordering, which Metal/MSL does not expose on this toolchain. Cross-threadgroup\n" +
                  "         handoffs are racy and results WILL be wrong. Use --groups 1 for correct output.\n", stderr)
        }
        let mk = MegakernelModel(engine: engine, st: st, config: config, ngroups: ngroups, maxT: maxSeq)
        mk.reset()
        fputs("loaded megakernel (groups=\(ngroups)) in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s\n", stderr)
        for (p, tok) in promptIds.enumerated() { next = mk.forward(token: tok, pos: p) }
        var pos = promptIds.count
        generated.append(next)
        while generated.count < maxTokens && !tokenizer.eosIds.contains(next) && pos < maxSeq {
            next = mk.forward(token: next, pos: pos)
            pos += 1
            generated.append(next)
        }
        fputs(String(format: "megakernel GPU: %.3f ms/token, %.2f tokens/s (%d steps)\n", mk.gpuTime * 1000 / Double(mk.steps), Double(mk.steps) / mk.gpuTime, mk.steps), stderr)
    } else {
        let usePrefill = prefillChunk > 0 && (promptIds.count >= 64 || CommandLine.arguments.contains("--prefill-chunk"))
        let weights = ModelWeights(engine: engine, st: st, config: config, buildBF16: usePrefill)
        let model = Model(engine: engine, weights: weights, maxT: maxSeq)
        fputs("loaded multi-kernel in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s\n", stderr)
        let prefillStart = DispatchTime.now().uptimeNanoseconds
        if usePrefill {
            let prefill = PrefillModel(engine: engine, weights: weights, chunk: prefillChunk, maxT: maxSeq)
            next = prefill.prefill(promptIds)
            prefill.handoff(to: model, tokenCount: promptIds.count)
        } else {
            for (p, tok) in promptIds.enumerated() { next = model.forward(token: tok, pos: p) }
        }
        let prefillSeconds = Double(DispatchTime.now().uptimeNanoseconds - prefillStart) / 1e9
        fputs(String(format: "prefill: %d tokens, %.3f s, %.1f tokens/s (weights already loaded)\n", promptIds.count, prefillSeconds, Double(promptIds.count) / prefillSeconds), stderr)
        var pos = promptIds.count
        generated.append(next)
        while generated.count < maxTokens && !tokenizer.eosIds.contains(next) && pos < maxSeq {
            next = model.forward(token: next, pos: pos)
            pos += 1
            generated.append(next)
        }
    }
    let dt = Date().timeIntervalSince(genStart)

    print("generated ids: \(generated)")
    print("output: \(tokenizer.decode(generated))")
    fputs("\(generated.count) tokens end-to-end (weight loading + prefill + decode) in \(String(format: "%.3f", dt))s = \(String(format: "%.1f", Double(generated.count) / dt)) tok/s\n", stderr)
    if Model.timing {
        let n = Double(generated.count + promptIds.count)
        fputs("host encode: \(String(format: "%.2f", Model.encodeTime*1000/n)) ms/token, gpu: \(String(format: "%.2f", Model.gpuTime*1000/n)) ms/token\n", stderr)
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}

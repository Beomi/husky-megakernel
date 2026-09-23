import Foundation
import Metal

/// Chunked prefill: processes `chunk` prompt tokens at once as a GEMM so the
/// weights are reused across each 64-token GEMM tile.
final class PrefillModel {
    static let prof = ProcessInfo.processInfo.environment["HUSKY_PREFILL_PROF"] != nil
    let engine: MetalEngine
    let w: ModelWeights
    let config: ModelConfig
    let optimized: Bool
    let chunk: Int
    let maxT: Int

    let h, heads, kvHeads, headDim, qProjOut, oProjIn, inter: Int
    let kHeads, vHeads, kDim, vDim, keyDim, valueDim, convDim, vocab: Int

    let hidden, xn, qkv, z, aBuf, bBuf, gBuf, betaBuf, convOut: MTLBuffer
    let qLin, kLin, vLin, qn, kn, linOut, qBuf, gateBuf, kBuf, vBuf, attnOut, mlpGate: MTLBuffer
    let logits, partialVal, partialIdx: MTLBuffer
    let mlpUp, projOut: MTLBuffer
    let tokenIn: MTLBuffer

    var convStates: [MTLBuffer?]
    var recStates: [MTLBuffer?]
    var kCaches: [MTLBuffer?]
    var vCaches: [MTLBuffer?]

    init(engine: MetalEngine, weights: ModelWeights, chunk: Int, maxT: Int = 4096, optimized: Bool = true) {
        self.engine = engine; self.w = weights; self.config = weights.config
        precondition(chunk > 0 && maxT > 0)
        self.optimized = optimized
        self.chunk = chunk; self.maxT = maxT
        let c = weights.config
        self.h = c.hiddenSize; self.heads = c.numAttentionHeads; self.kvHeads = c.numKeyValueHeads
        self.headDim = c.headDim; self.qProjOut = c.numAttentionHeads * c.headDim * 2
        self.oProjIn = c.numAttentionHeads * c.headDim; self.inter = c.intermediateSize
        self.kHeads = c.linearNumKeyHeads; self.vHeads = c.linearNumValueHeads
        self.kDim = c.linearKeyHeadDim; self.vDim = c.linearValueHeadDim
        self.keyDim = self.kDim * self.kHeads; self.valueDim = self.vDim * self.vHeads
        self.convDim = self.keyDim * 2 + self.valueDim; self.vocab = c.vocabSize
        let C = chunk
        hidden = engine.emptyBf16(C * h); xn = engine.emptyBf16(C * h)
        qkv = engine.emptyBf16(C * max(qProjOut, convDim)); z = engine.emptyBf16(C * valueDim)
        aBuf = engine.emptyBf16(C * vHeads); bBuf = engine.emptyBf16(C * vHeads)
        gBuf = engine.emptyBf16(C * vHeads); betaBuf = engine.emptyBf16(C * vHeads)
        convOut = engine.emptyBf16(C * convDim)
        qLin = engine.emptyBf16(C * keyDim); kLin = engine.emptyBf16(C * keyDim); vLin = engine.emptyBf16(C * valueDim)
        qn = engine.emptyBf16(C * keyDim); kn = engine.emptyBf16(C * keyDim); linOut = engine.emptyBf16(C * valueDim)
        qBuf = engine.emptyBf16(C * oProjIn); gateBuf = engine.emptyBf16(C * oProjIn)
        kBuf = engine.emptyBf16(C * kvHeads * headDim); vBuf = engine.emptyBf16(C * kvHeads * headDim)
        attnOut = engine.emptyBf16(C * oProjIn); mlpGate = engine.emptyBf16(C * inter)
        mlpUp = engine.emptyBf16(C * inter); projOut = engine.emptyBf16(C * h)
        logits = engine.emptyBf16(vocab)
        partialVal = engine.empty(256); partialIdx = engine.empty(256)
        tokenIn = engine.empty(C)
        convStates = Array(repeating: nil, count: c.numHiddenLayers)
        recStates = Array(repeating: nil, count: c.numHiddenLayers)
        kCaches = Array(repeating: nil, count: c.numHiddenLayers)
        vCaches = Array(repeating: nil, count: c.numHiddenLayers)
        for i in 0..<c.numHiddenLayers {
            if c.layerTypes[i] == "linear_attention" {
                convStates[i] = engine.emptyBf16Zeros(convDim * 3)
                recStates[i] = engine.emptyBf16Zeros(vHeads * vDim * kDim)
            } else {
                kCaches[i] = engine.emptyBf16Zeros(kvHeads * maxT * headDim)
                vCaches[i] = engine.emptyBf16Zeros(kvHeads * maxT * headDim)
            }
        }
    }

    func reset() {
        for i in 0..<config.numHiddenLayers {
            if let s = convStates[i] { memset(s.contents(), 0, s.length) }
            if let s = recStates[i] { memset(s.contents(), 0, s.length) }
            if let s = kCaches[i] { memset(s.contents(), 0, s.length) }
            if let s = vCaches[i] { memset(s.contents(), 0, s.length) }
        }
    }

    private func mv(_ e: Encoder, _ t: QuantTensor, _ x: MTLBuffer, _ y: MTLBuffer, _ C: Int) {
        if let ht = t.halfT {
            e.run2D("gemm_mma", [.buf(x), .buf(ht), .buf(y),
                                 .u32(UInt32(C)), .u32(UInt32(t.outN)), .u32(UInt32(t.inN))],
                    grid: ((C + 63) / 64, (t.outN + 31) / 32), threadgroup: 256)
            return
        }
        let groups = max(1, (t.outN + 3) / 4)
        e.runRows("qmatvec_prefill", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(y),
                                      .u32(UInt32(t.outN)), .u32(UInt32(t.inN)), .u32(UInt32(C)), .u32(UInt32(groups * 4))],
                  rows: groups, threadgroup: 128)
    }
    /// Process tokens[pos..<pos+C] (chunk at offset `start`, length C) and, if
    /// `last`, return the greedy next token.
    func forwardChunk(_ tokens: ArraySlice<Int>, start: Int, C: Int, pos: Int, last: Bool) -> Int? {
        precondition(C > 0 && C <= chunk && tokens.count == C && pos >= 0 && pos + C <= maxT)
        let eps = config.rmsNormEps, theta = config.ropeTheta
        let ropeDim = UInt32(config.rotaryDim)
        let scaleQ = 1.0 / Float(Double(kDim).squareRoot())
        let scaleAttn = 1.0 / Float(Double(headDim).squareRoot())
        tokenIn.contents().bindMemory(to: UInt32.self, capacity: C)
            .update(from: tokens.map { UInt32($0) }, count: C)

        let cb = engine.queue.makeCommandBuffer()!
        let e = Encoder(engine, cb, profile: PrefillModel.prof)
        e.run("embed_lookup_b", [.buf(w.embed.weight), .buf(w.embed.scales), .buf(w.embed.biases),
                                 .buf(tokenIn), .buf(hidden), .u32(UInt32(h)), .u32(UInt32(C))],
              grid: C * h, threadgroup: 256)

        for i in 0..<config.numHiddenLayers {
            e.runRows("rmsnorm", [.buf(hidden), .buf(w.inputNorms[i]), .buf(xn),
                                  .u32(UInt32(h)), .f32(eps), .u32(0)], rows: C, threadgroup: 256)
            switch w.layers[i] {
            case .linear(let lw):
                mv(e, lw.inProjQKV, xn, qkv, C)
                mv(e, lw.inProjZ, xn, z, C)
                mv(e, lw.inProjA, xn, aBuf, C)
                mv(e, lw.inProjB, xn, bBuf, C)
                e.run("conv_chunk", [.buf(qkv), .buf(convStates[i]!), .buf(lw.convW), .buf(convOut),
                                     .u32(UInt32(convDim)), .u32(UInt32(C))], grid: convDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(qLin), .u32(UInt32(keyDim)), .u32(0), .u32(UInt32(convDim))], grid: C * keyDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(kLin), .u32(UInt32(keyDim)), .u32(UInt32(keyDim)), .u32(UInt32(convDim))], grid: C * keyDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(vLin), .u32(UInt32(valueDim)), .u32(UInt32(2 * keyDim)), .u32(UInt32(convDim))], grid: C * valueDim, threadgroup: 256)
                e.runRows("l2norm_scale", [.buf(qLin), .buf(qn), .u32(UInt32(kDim)), .f32(scaleQ)], rows: C * kHeads, threadgroup: 128)
                e.runRows("l2norm_scale", [.buf(kLin), .buf(kn), .u32(UInt32(kDim)), .f32(1.0)], rows: C * kHeads, threadgroup: 128)
                e.run("g_beta_b", [.buf(aBuf), .buf(bBuf), .buf(lw.aLog), .buf(lw.dtBias), .buf(gBuf), .buf(betaBuf),
                                   .u32(UInt32(vHeads))], grid: C * vHeads, threadgroup: vHeads)
                e.runRows(optimized && kDim == 128 ? "gdelta_chunk_register" : "gdelta_chunk", [.buf(qn), .buf(kn), .buf(vLin), .buf(gBuf), .buf(betaBuf), .buf(recStates[i]!), .buf(linOut),
                                           .u32(UInt32(vHeads)), .u32(UInt32(kHeads)), .u32(UInt32(kDim)), .u32(UInt32(vDim)), .u32(UInt32(C))],
                          rows: vHeads, threadgroup: 128)
                e.runRows("rmsnorm_gated", [.buf(linOut), .buf(lw.normW), .buf(z), .buf(linOut),
                                            .u32(UInt32(vDim)), .f32(eps)], rows: C * (valueDim / vDim), threadgroup: 128)
                mv(e, lw.outProj, linOut, projOut, C)
                e.run("add_inplace", [.buf(hidden), .buf(projOut), .u32(UInt32(C * h))], grid: C * h, threadgroup: 256)
            case .full(let fw):
                mv(e, fw.qProj, xn, qkv, C)
                e.run("split_q_gate_b", [.buf(qkv), .buf(qBuf), .buf(gateBuf),
                                         .u32(UInt32(heads)), .u32(UInt32(headDim)), .u32(UInt32(C))], grid: C * oProjIn, threadgroup: 256)
                mv(e, fw.kProj, xn, kBuf, C)
                mv(e, fw.vProj, xn, vBuf, C)
                e.runRows("rmsnorm", [.buf(qBuf), .buf(fw.qNorm), .buf(qBuf), .u32(UInt32(headDim)), .f32(eps), .u32(0)], rows: C * heads, threadgroup: 256)
                e.runRows("rmsnorm", [.buf(kBuf), .buf(fw.kNorm), .buf(kBuf), .u32(UInt32(headDim)), .f32(eps), .u32(0)], rows: C * kvHeads, threadgroup: 256)
                e.run("apply_rope_chunk", [.buf(qBuf), .u32(UInt32(heads)), .u32(UInt32(headDim)), .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)], grid: C * heads * config.rotaryDim / 2, threadgroup: 128)
                e.run("apply_rope_chunk", [.buf(kBuf), .u32(UInt32(kvHeads)), .u32(UInt32(headDim)), .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)], grid: C * kvHeads * config.rotaryDim / 2, threadgroup: 128)
                e.run("write_kv_chunk", [.buf(kBuf), .buf(vBuf), .buf(kCaches[i]!), .buf(vCaches[i]!),
                                         .u32(UInt32(kvHeads)), .u32(UInt32(headDim)), .u32(UInt32(maxT)), .u32(UInt32(pos))], grid: C * kvHeads * headDim, threadgroup: 256)
                e.runRows(optimized && headDim == 256 ? "attention_chunk_simd" : "attention_chunk", [.buf(qBuf), .buf(kCaches[i]!), .buf(vCaches[i]!), .buf(attnOut),
                                              .u32(UInt32(heads)), .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                              .u32(UInt32(pos)), .f32(scaleAttn), .u32(UInt32(maxT))],
                          rows: C * heads, threadgroup: 256)
                e.run("mul_sigmoid_gate", [.buf(attnOut), .buf(gateBuf), .u32(UInt32(C * oProjIn))], grid: C * oProjIn, threadgroup: 256)
                mv(e, fw.oProj, attnOut, projOut, C)
                e.run("add_inplace", [.buf(hidden), .buf(projOut), .u32(UInt32(C * h))], grid: C * h, threadgroup: 256)
            }
            e.runRows("rmsnorm", [.buf(hidden), .buf(w.postAttnNorms[i]), .buf(xn), .u32(UInt32(h)), .f32(eps), .u32(0)], rows: C, threadgroup: 256)
            mv(e, w.mlpGate[i], xn, mlpGate, C)
            mv(e, w.mlpUp[i], xn, mlpUp, C)
            e.run("silu_mul", [.buf(mlpGate), .buf(mlpUp), .buf(mlpGate), .u32(UInt32(C * inter))], grid: C * inter, threadgroup: 256)
            mv(e, w.mlpDown[i], mlpGate, projOut, C)
            e.run("add_inplace", [.buf(hidden), .buf(projOut), .u32(UInt32(C * h))], grid: C * h, threadgroup: 256)
        }

        e.runRows("rmsnorm", [.buf(hidden), .buf(w.finalNorm), .buf(xn), .u32(UInt32(h)), .f32(eps), .u32(0)], rows: C, threadgroup: 256)

        var tok: Int? = nil
        if last {
            // logits for the last token only
            let xOff = (C - 1) * h * 2
            let lg = max(1, (vocab + 3) / 4)
            e.runRows("qmatvec_warp", [.buf(w.embed.weight), .buf(w.embed.scales), .buf(w.embed.biases),
                                       .bufOffset(xn, xOff), .buf(logits),
                                       .u32(UInt32(vocab)), .u32(UInt32(h)), .u32(UInt32(lg * 4))],
                      rows: lg, threadgroup: 128)
            e.runRows("argmax_partial", [.buf(logits), .buf(partialVal), .buf(partialIdx), .u32(UInt32(vocab))],
                      rows: 256, threadgroup: 256)
        }
        e.end(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.status == .completed, "prefill failed: \(String(describing: cb.error))")
        if PrefillModel.prof { e.reportProfile() }
        if last {
            let pv = partialVal.contents().bindMemory(to: Float.self, capacity: 256)
            let pi = partialIdx.contents().bindMemory(to: UInt32.self, capacity: 256)
            var best: Float = -.infinity; var bi = 0
            for k in 0..<256 where pv[k] > best { best = pv[k]; bi = Int(pi[k]) }
            tok = bi
        }
        return tok
    }

    /// Continue decode from the state produced by prefill, without replaying tokens.
    func handoff(to model: Model, tokenCount: Int) {
        precondition(model.w === w && model.maxT == maxT)
        precondition(tokenCount > 0 && tokenCount <= maxT)
        model.convStates = convStates; model.recStates = recStates
        model.kCaches = kCaches; model.vCaches = vCaches
        model.cacheLen = tokenCount
    }

    /// Prefill an entire prompt in chunks; returns the greedy next token.
    func prefill(_ tokens: [Int]) -> Int {
        precondition(!tokens.isEmpty && tokens.count <= maxT)
        precondition(tokens.allSatisfy { $0 >= 0 && $0 < vocab })
        reset()
        var pos = 0
        var i = 0
        var tok = 0
        while i < tokens.count {
            let C = min(chunk, tokens.count - i)
            let last = (i + C) >= tokens.count
            if let t = forwardChunk(tokens[i..<(i + C)], start: i, C: C, pos: pos, last: last) { tok = t }
            pos += C; i += C
        }
        return tok
    }
}

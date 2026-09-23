import Foundation
import Metal

/// Batched forward pass: `batch` independent sequences processed together so the
/// weight stream is read once per output row for all batch elements.
final class BatchModel {
    let engine: MetalEngine
    let w: ModelWeights
    let config: ModelConfig
    let batch: Int
    let maxT: Int

    let h, heads, kvHeads, headDim, qProjOut, oProjIn, inter: Int
    let kHeads, vHeads, kDim, vDim, keyDim, valueDim, convDim, vocab: Int

    // scratch (all [batch][dim])
    let hidden, xn, qkv, z, aBuf, bBuf, gBuf, betaBuf, convOut: MTLBuffer
    let qLin, kLin, vLin, qn, kn, linOut, qBuf, gateBuf, kBuf, vBuf, attnOut: MTLBuffer
    let mlpGate, logits: MTLBuffer
    let partialVal, partialIdx: MTLBuffer
    let tokenIn, tokenOut: MTLBuffer

    var convStates: [MTLBuffer?]
    var recStates: [MTLBuffer?]
    var kCaches: [MTLBuffer?]
    var vCaches: [MTLBuffer?]

    init(engine: MetalEngine, weights: ModelWeights, batch: Int, maxT: Int = 4096) {
        self.engine = engine
        self.w = weights
        self.config = weights.config
        self.batch = batch
        self.maxT = maxT
        let c = weights.config
        self.h = c.hiddenSize; self.heads = c.numAttentionHeads; self.kvHeads = c.numKeyValueHeads
        self.headDim = c.headDim; self.qProjOut = c.numAttentionHeads * c.headDim * 2
        self.oProjIn = c.numAttentionHeads * c.headDim; self.inter = c.intermediateSize
        self.kHeads = c.linearNumKeyHeads; self.vHeads = c.linearNumValueHeads
        self.kDim = c.linearKeyHeadDim; self.vDim = c.linearValueHeadDim
        self.keyDim = self.kDim * self.kHeads; self.valueDim = self.vDim * self.vHeads
        self.convDim = self.keyDim * 2 + self.valueDim; self.vocab = c.vocabSize
        let B = batch

        hidden = engine.emptyBf16(B * h)
        xn = engine.emptyBf16(B * h)
        qkv = engine.emptyBf16(B * max(qProjOut, convDim))
        z = engine.emptyBf16(B * valueDim)
        aBuf = engine.emptyBf16(B * vHeads); bBuf = engine.emptyBf16(B * vHeads)
        gBuf = engine.emptyBf16(B * vHeads); betaBuf = engine.emptyBf16(B * vHeads)
        convOut = engine.emptyBf16(B * convDim)
        qLin = engine.emptyBf16(B * keyDim); kLin = engine.emptyBf16(B * keyDim); vLin = engine.emptyBf16(B * valueDim)
        qn = engine.emptyBf16(B * keyDim); kn = engine.emptyBf16(B * keyDim)
        linOut = engine.emptyBf16(B * valueDim)
        qBuf = engine.emptyBf16(B * oProjIn); gateBuf = engine.emptyBf16(B * oProjIn)
        kBuf = engine.emptyBf16(B * kvHeads * headDim); vBuf = engine.emptyBf16(B * kvHeads * headDim)
        attnOut = engine.emptyBf16(B * oProjIn)
        mlpGate = engine.emptyBf16(B * inter)
        logits = engine.emptyBf16(B * vocab)
        partialVal = engine.empty(256); partialIdx = engine.empty(256)
        tokenIn = engine.empty(B); tokenOut = engine.empty(B)

        convStates = Array(repeating: nil, count: c.numHiddenLayers)
        recStates = Array(repeating: nil, count: c.numHiddenLayers)
        kCaches = Array(repeating: nil, count: c.numHiddenLayers)
        vCaches = Array(repeating: nil, count: c.numHiddenLayers)
        for i in 0..<c.numHiddenLayers {
            if c.layerTypes[i] == "linear_attention" {
                convStates[i] = engine.emptyBf16Zeros(B * convDim * 3)
                recStates[i] = engine.emptyBf16Zeros(B * vHeads * vDim * kDim)
            } else {
                kCaches[i] = engine.emptyBf16Zeros(B * kvHeads * maxT * headDim)
                vCaches[i] = engine.emptyBf16Zeros(B * kvHeads * maxT * headDim)
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

    private func mv(_ e: Encoder, _ t: QuantTensor, _ x: MTLBuffer, _ y: MTLBuffer) {
        let groups = max(1, (t.outN + 3) / 4)
        e.runRows("qmatvec_b", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(y),
                                .u32(UInt32(t.outN)), .u32(UInt32(t.inN)),
                                .u32(UInt32(batch)), .u32(UInt32(groups * 4))],
                  rows: groups, threadgroup: 128)
    }
    private func mvAdd(_ e: Encoder, _ t: QuantTensor, _ x: MTLBuffer, _ r: MTLBuffer, _ y: MTLBuffer) {
        let groups = max(1, (t.outN + 3) / 4)
        e.runRows("qmatvec_add_b", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(r), .buf(y),
                                    .u32(UInt32(t.outN)), .u32(UInt32(t.inN)),
                                    .u32(UInt32(batch)), .u32(UInt32(groups * 4))],
                  rows: groups, threadgroup: 128)
    }

    func forward(tokens: [Int], pos: Int) -> [Int] {
        let B = batch
        let eps = config.rmsNormEps, theta = config.ropeTheta
        let ropeDim = UInt32(config.rotaryDim)
        let scaleQ = 1.0 / Float(Double(kDim).squareRoot())
        let scaleAttn = 1.0 / Float(Double(headDim).squareRoot())
        precondition(tokens.count == B)
        tokenIn.contents().bindMemory(to: UInt32.self, capacity: B)
            .update(from: tokens.map { UInt32($0) }, count: B)

        let cb = engine.queue.makeCommandBuffer()!
        let e = Encoder(engine, cb)

        e.run("embed_lookup_b", [.buf(w.embed.weight), .buf(w.embed.scales), .buf(w.embed.biases),
                                 .buf(tokenIn), .buf(hidden), .u32(UInt32(h)), .u32(UInt32(B))],
              grid: B * h, threadgroup: 256)

        for i in 0..<config.numHiddenLayers {
            e.runRows("rmsnorm", [.buf(hidden), .buf(w.inputNorms[i]), .buf(xn),
                                  .u32(UInt32(h)), .f32(eps), .u32(0)], rows: B, threadgroup: 256)
            switch w.layers[i] {
            case .linear(let lw):
                mv(e, lw.inProjQKV, xn, qkv)
                mv(e, lw.inProjZ, xn, z)
                mv(e, lw.inProjA, xn, aBuf)
                mv(e, lw.inProjB, xn, bBuf)
                e.run("conv_step_b", [.buf(qkv), .buf(convStates[i]!), .buf(lw.convW), .buf(convOut),
                                      .u32(UInt32(convDim))], grid: B * convDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(qLin), .u32(UInt32(keyDim)), .u32(0), .u32(UInt32(convDim))],
                      grid: B * keyDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(kLin), .u32(UInt32(keyDim)), .u32(UInt32(keyDim)), .u32(UInt32(convDim))],
                      grid: B * keyDim, threadgroup: 256)
                e.run("copy_range_b", [.buf(convOut), .buf(vLin), .u32(UInt32(valueDim)), .u32(UInt32(2 * keyDim)), .u32(UInt32(convDim))],
                      grid: B * valueDim, threadgroup: 256)
                e.runRows("l2norm_scale", [.buf(qLin), .buf(qn), .u32(UInt32(kDim)), .f32(scaleQ)],
                          rows: B * kHeads, threadgroup: 128)
                e.runRows("l2norm_scale", [.buf(kLin), .buf(kn), .u32(UInt32(kDim)), .f32(1.0)],
                          rows: B * kHeads, threadgroup: 128)
                e.run("g_beta_b", [.buf(aBuf), .buf(bBuf), .buf(lw.aLog), .buf(lw.dtBias), .buf(gBuf), .buf(betaBuf),
                                   .u32(UInt32(vHeads))], grid: B * vHeads, threadgroup: 64)
                e.runRows("gdelta_step_b", [.buf(qn), .buf(kn), .buf(vLin), .buf(gBuf), .buf(betaBuf),
                                            .buf(recStates[i]!), .buf(linOut),
                                            .u32(UInt32(vHeads)), .u32(UInt32(kHeads)),
                                            .u32(UInt32(kDim)), .u32(UInt32(vDim))],
                          rows: B * vHeads, threadgroup: 128)
                e.runRows("rmsnorm_gated", [.buf(linOut), .buf(lw.normW), .buf(z), .buf(linOut),
                                            .u32(UInt32(vDim)), .f32(eps)],
                          rows: B * (valueDim / vDim), threadgroup: 128)
                mvAdd(e, lw.outProj, linOut, hidden, hidden)
            case .full(let fw):
                mv(e, fw.qProj, xn, qkv)
                e.run("split_q_gate_b", [.buf(qkv), .buf(qBuf), .buf(gateBuf),
                                         .u32(UInt32(heads)), .u32(UInt32(headDim)), .u32(UInt32(B))],
                      grid: B * oProjIn, threadgroup: 256)
                mv(e, fw.kProj, xn, kBuf)
                mv(e, fw.vProj, xn, vBuf)
                e.runRows("rmsnorm", [.buf(qBuf), .buf(fw.qNorm), .buf(qBuf),
                                      .u32(UInt32(headDim)), .f32(eps), .u32(0)], rows: B * heads, threadgroup: 256)
                e.runRows("rmsnorm", [.buf(kBuf), .buf(fw.kNorm), .buf(kBuf),
                                      .u32(UInt32(headDim)), .f32(eps), .u32(0)], rows: B * kvHeads, threadgroup: 256)
                e.run("apply_rope_b", [.buf(qBuf), .u32(UInt32(heads)), .u32(UInt32(headDim)),
                                       .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)],
                      grid: B * heads * config.rotaryDim / 2, threadgroup: 128)
                e.run("apply_rope_b", [.buf(kBuf), .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                       .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)],
                      grid: B * kvHeads * config.rotaryDim / 2, threadgroup: 128)
                e.run("write_kv_b", [.buf(kBuf), .buf(vBuf), .buf(kCaches[i]!), .buf(vCaches[i]!),
                                     .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                     .u32(UInt32(maxT)), .u32(UInt32(pos))],
                      grid: B * kvHeads * headDim, threadgroup: 256)
                e.runRows("attention_decode_b", [.buf(qBuf), .buf(kCaches[i]!), .buf(vCaches[i]!), .buf(attnOut),
                                                 .u32(UInt32(heads)), .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                                 .u32(UInt32(pos + 1)), .f32(scaleAttn), .u32(UInt32(maxT))],
                          rows: B * heads, threadgroup: 256)
                e.run("mul_sigmoid_gate", [.buf(attnOut), .buf(gateBuf), .u32(UInt32(B * oProjIn))],
                      grid: B * oProjIn, threadgroup: 256)
                mvAdd(e, fw.oProj, attnOut, hidden, hidden)
            }

            e.runRows("rmsnorm", [.buf(hidden), .buf(w.postAttnNorms[i]), .buf(xn),
                                  .u32(UInt32(h)), .f32(eps), .u32(0)], rows: B, threadgroup: 256)
            e.runRows("qmatvec_pair_silu_b",
                      [.buf(w.mlpGate[i].weight), .buf(w.mlpGate[i].scales), .buf(w.mlpGate[i].biases),
                       .buf(w.mlpUp[i].weight), .buf(w.mlpUp[i].scales), .buf(w.mlpUp[i].biases),
                       .buf(xn), .buf(mlpGate), .u32(UInt32(inter)), .u32(UInt32(h)), .u32(UInt32(B))],
                      rows: (inter + 3) / 4, threadgroup: 128)
            mvAdd(e, w.mlpDown[i], mlpGate, hidden, hidden)
        }

        e.runRows("rmsnorm", [.buf(hidden), .buf(w.finalNorm), .buf(xn),
                              .u32(UInt32(h)), .f32(eps), .u32(0)], rows: B, threadgroup: 256)
        mv(e, w.embed, xn, logits)
        e.runRows("argmax_b", [.buf(logits), .buf(tokenOut), .u32(UInt32(vocab))],
                  rows: B, threadgroup: 256)
        e.end()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fputs("batch error: \(err)\n", stderr) }
        let p = tokenOut.contents().bindMemory(to: UInt32.self, capacity: B)
        return (0..<B).map { Int(p[$0]) }
    }
}

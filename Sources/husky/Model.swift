import Foundation
import Metal

final class Model {
    static let timing = ProcessInfo.processInfo.environment["HUSKY_TIMING"] != nil
    static let tgSize = Int(ProcessInfo.processInfo.environment["HUSKY_TG"] ?? "128") ?? 128
    static let targetWarps = Int(ProcessInfo.processInfo.environment["HUSKY_SPLIT"] ?? "0") ?? 0
    static var encodeTime: TimeInterval = 0
    static var gpuTime: TimeInterval = 0
    let engine: MetalEngine
    let w: ModelWeights
    let config: ModelConfig
    let maxT: Int

    // scratch
    let hidden: MTLBuffer
    let xn: MTLBuffer
    let qkv: MTLBuffer
    let z: MTLBuffer
    let aBuf: MTLBuffer
    let bBuf: MTLBuffer
    let gBuf: MTLBuffer
    let betaBuf: MTLBuffer
    let convOut: MTLBuffer
    let qLin: MTLBuffer
    let kLin: MTLBuffer
    let vLin: MTLBuffer
    let qn: MTLBuffer
    let kn: MTLBuffer
    let linOut: MTLBuffer
    let qBuf: MTLBuffer
    let gateBuf: MTLBuffer
    let kBuf: MTLBuffer
    let vBuf: MTLBuffer
    let attnOut: MTLBuffer
    let mixBuf: MTLBuffer
    let mlpGateBuf: MTLBuffer
    let mlpUpBuf: MTLBuffer
    let mlpDownBuf: MTLBuffer
    let logits: MTLBuffer
    let partialVal: MTLBuffer
    let partialIdx: MTLBuffer
    let splitPartial: MTLBuffer

    // states
    var convStates: [MTLBuffer?]
    var recStates: [MTLBuffer?]
    var kCaches: [MTLBuffer?]
    var vCaches: [MTLBuffer?]
    var cacheLen: Int = 0

    let dumpEnabled: Bool
    var dumpBuffers: [MTLBuffer] = []
    var tapBuffers: [MTLBuffer] = []
    var dumps: [[Float]] = []
    var dumpLogits: [Float] = []
    var taps: [[Float]] = []

    let h: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let qProjOut: Int
    let oProjIn: Int
    let inter: Int
    let kHeads: Int
    let vHeads: Int
    let kDim: Int
    let vDim: Int
    let keyDim: Int
    let valueDim: Int
    let convDim: Int
    let vocab: Int

    init(engine: MetalEngine, weights: ModelWeights, maxT: Int = 4096) {
        self.engine = engine
        self.w = weights
        self.config = weights.config
        self.maxT = maxT
        let c = weights.config
        self.h = c.hiddenSize
        self.heads = c.numAttentionHeads
        self.kvHeads = c.numKeyValueHeads
        self.headDim = c.headDim
        self.qProjOut = c.numAttentionHeads * c.headDim * 2
        self.oProjIn = c.numAttentionHeads * c.headDim
        self.inter = c.intermediateSize
        self.kHeads = c.linearNumKeyHeads
        self.vHeads = c.linearNumValueHeads
        self.kDim = c.linearKeyHeadDim
        self.vDim = c.linearValueHeadDim
        self.keyDim = c.linearKeyHeadDim * c.linearNumKeyHeads
        self.valueDim = c.linearValueHeadDim * c.linearNumValueHeads
        self.convDim = self.keyDim * 2 + self.valueDim
        self.vocab = c.vocabSize

        hidden = engine.emptyBf16(h)
        xn = engine.emptyBf16(h)
        qkv = engine.emptyBf16(max(qProjOut, convDim))
        z = engine.emptyBf16(valueDim)
        aBuf = engine.emptyBf16(vHeads)
        bBuf = engine.emptyBf16(vHeads)
        gBuf = engine.emptyBf16(vHeads)
        betaBuf = engine.emptyBf16(vHeads)
        convOut = engine.emptyBf16(convDim)
        qLin = engine.emptyBf16(keyDim)
        kLin = engine.emptyBf16(keyDim)
        vLin = engine.emptyBf16(valueDim)
        qn = engine.emptyBf16(keyDim)
        kn = engine.emptyBf16(keyDim)
        linOut = engine.emptyBf16(valueDim)
        qBuf = engine.emptyBf16(oProjIn)
        gateBuf = engine.emptyBf16(oProjIn)
        kBuf = engine.emptyBf16(kvHeads * headDim)
        vBuf = engine.emptyBf16(kvHeads * headDim)
        attnOut = engine.emptyBf16(oProjIn)
        mixBuf = engine.emptyBf16(h)
        mlpGateBuf = engine.emptyBf16(inter)
        mlpUpBuf = engine.emptyBf16(inter)
        mlpDownBuf = engine.emptyBf16(h)
        logits = engine.emptyBf16(vocab)
        partialVal = engine.empty(256)
        partialIdx = engine.empty(256)
        splitPartial = engine.empty(1 << 21)

        self.dumpEnabled = ProcessInfo.processInfo.environment["HUSKY_DUMP"] != nil
        if dumpEnabled {
            for _ in 0..<(c.numHiddenLayers + 1) { dumpBuffers.append(engine.emptyBf16(h)) }
            for _ in 0..<8 { tapBuffers.append(engine.emptyBf16(max(h, oProjIn))) }
        }
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

    private func readBf16(_ b: MTLBuffer, _ count: Int) -> [Float] {
        let p = b.contents().bindMemory(to: UInt16.self, capacity: count)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = Float(bitPattern: UInt32(p[i]) << 16) }
        return out
    }

    func reset() {
        cacheLen = 0
        for i in 0..<config.numHiddenLayers {
            if config.layerTypes[i] == "linear_attention" {
                memset(convStates[i]!.contents(), 0, convStates[i]!.length)
                memset(recStates[i]!.contents(), 0, recStates[i]!.length)
            } else {
                memset(kCaches[i]!.contents(), 0, kCaches[i]!.length)
                memset(vCaches[i]!.contents(), 0, vCaches[i]!.length)
            }
        }
    }

    private func qmatvec(_ e: Encoder, _ t: QuantTensor, _ x: MTLBuffer, _ y: MTLBuffer) {
        let inWords = t.inN / 8
        // Keep each slice >= 128 words so all 32 lanes stay busy (32 uint4).
        let minWPS = 128
        let maxSplits = max(1, inWords / minWPS)
        var nSplits = max(1, min(maxSplits, (Model.targetWarps + t.outN - 1) / t.outN))
        var wps0 = ((inWords + nSplits - 1) / nSplits + 7) & ~7
        if wps0 < minWPS { wps0 = minWPS }
        nSplits = (inWords + wps0 - 1) / wps0
        if nSplits <= 1 {
            let wpg = Model.tgSize / 32
            let groups = max(1, (t.outN + wpg - 1) / wpg)
            e.runRows("qmatvec_warp", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(y),
                                       .u32(UInt32(t.outN)), .u32(UInt32(t.inN)), .u32(UInt32(groups * wpg))],
                      rows: groups, threadgroup: Model.tgSize)
            return
        }
        let wpg = Model.tgSize / 32
        let totalWarps = t.outN * nSplits
        let groups = max(1, (totalWarps + wpg - 1) / wpg)
        e.runRows("qmatvec_split", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(splitPartial),
                                    .u32(UInt32(t.outN)), .u32(UInt32(t.inN)), .u32(UInt32(wps0)), .u32(UInt32(nSplits))],
                  rows: groups, threadgroup: Model.tgSize)
        e.run("split_reduce", [.buf(splitPartial), .buf(y), .u32(UInt32(t.outN)), .u32(UInt32(nSplits))],
              grid: t.outN, threadgroup: 256)
    }

    private func qmatvecPairSilu(_ e: Encoder, _ a: QuantTensor, _ b: QuantTensor, _ y: MTLBuffer, _ x: MTLBuffer) {
        let wpg = Model.tgSize / 32
        let groups = max(1, (a.outN + wpg - 1) / wpg)
        e.runRows("qmatvec_pair_silu",
                  [.buf(a.weight), .buf(a.scales), .buf(a.biases),
                   .buf(b.weight), .buf(b.scales), .buf(b.biases), .buf(x), .buf(y),
                   .u32(UInt32(a.outN)), .u32(UInt32(a.inN))],
                  rows: groups, threadgroup: Model.tgSize)
    }

    private func qmatvecAdd(_ e: Encoder, _ t: QuantTensor, _ x: MTLBuffer, _ r: MTLBuffer, _ y: MTLBuffer) {
        let wpg = Model.tgSize / 32
        let groups = max(1, (t.outN + wpg - 1) / wpg)
        e.runRows("qmatvec_warp_add", [.buf(t.weight), .buf(t.scales), .buf(t.biases), .buf(x), .buf(r), .buf(y),
                                       .u32(UInt32(t.outN)), .u32(UInt32(t.inN)), .u32(UInt32(groups * wpg))],
                  rows: groups, threadgroup: Model.tgSize)
    }

    private func qmatvecPair(_ e: Encoder, _ a: QuantTensor, _ ya: MTLBuffer, _ b: QuantTensor, _ yb: MTLBuffer, _ x: MTLBuffer) {
        let wpg = Model.tgSize / 32
        let total = a.outN + b.outN
        let groups = max(1, (total + wpg - 1) / wpg)
        e.runRows("qmatvec_pair",
                  [.buf(a.weight), .buf(a.scales), .buf(a.biases),
                   .buf(b.weight), .buf(b.scales), .buf(b.biases),
                   .buf(x), .buf(ya), .buf(yb),
                   .u32(UInt32(a.outN)), .u32(UInt32(b.outN)), .u32(UInt32(a.inN))],
                  rows: groups, threadgroup: Model.tgSize)
    }

    /// Runs one token through the model and returns the greedy next token id.
    func forward(token: Int, pos: Int) -> Int {
        let eps = config.rmsNormEps
        let theta = config.ropeTheta
        let ropeDim = UInt32(config.rotaryDim)
        let scaleQ = 1.0 / Float(Double(kDim).squareRoot())
        let scaleAttn = 1.0 / Float(Double(headDim).squareRoot())

        let forwardStart = Date()
        let cb = engine.queue.makeCommandBuffer()!
        let e = Encoder(engine, cb)

        e.run("embed_lookup", [.buf(w.embed.weight), .buf(w.embed.scales), .buf(w.embed.biases),
                               .buf(hidden), .u32(UInt32(token)), .u32(UInt32(h))],
              grid: h, threadgroup: 256)
        if dumpEnabled {
            e.run("copy_range", [.buf(hidden), .buf(dumpBuffers[0]), .u32(UInt32(h)), .u32(0)],
                  grid: h, threadgroup: 256)
        }

        for i in 0..<config.numHiddenLayers {
            e.runRows("rmsnorm", [.buf(hidden), .buf(w.inputNorms[i]), .buf(xn),
                                  .u32(UInt32(h)), .f32(eps), .u32(0)],
                      rows: 1, threadgroup: 256)
            switch w.layers[i] {
            case .linear(let lw):
                qmatvec(e, lw.inProjQKV, xn, qkv)
                qmatvec(e, lw.inProjZ, xn, z)
                qmatvec(e, lw.inProjA, xn, aBuf)
                qmatvec(e, lw.inProjB, xn, bBuf)
                e.run("conv_step", [.buf(qkv), .buf(convStates[i]!), .buf(lw.convW), .buf(convOut),
                                    .u32(UInt32(convDim))],
                      grid: convDim, threadgroup: 256)
                // split convOut -> qLin, kLin, vLin
                e.run("copy_range", [.buf(convOut), .buf(qLin), .u32(UInt32(keyDim)), .u32(0)],
                      grid: keyDim, threadgroup: 256)
                e.run("copy_range", [.buf(convOut), .buf(kLin), .u32(UInt32(keyDim)), .u32(UInt32(keyDim))],
                      grid: keyDim, threadgroup: 256)
                e.run("copy_range", [.buf(convOut), .buf(vLin), .u32(UInt32(valueDim)), .u32(UInt32(2 * keyDim))],
                      grid: valueDim, threadgroup: 256)

                e.runRows("l2norm_scale", [.buf(qLin), .buf(qn), .u32(UInt32(kDim)), .f32(scaleQ)],
                          rows: kHeads, threadgroup: 128)
                e.runRows("l2norm_scale", [.buf(kLin), .buf(kn), .u32(UInt32(kDim)), .f32(1.0)],
                          rows: kHeads, threadgroup: 128)
                e.run("g_beta", [.buf(aBuf), .buf(bBuf), .buf(lw.aLog), .buf(lw.dtBias),
                                 .buf(gBuf), .buf(betaBuf), .u32(UInt32(vHeads))],
                      grid: vHeads, threadgroup: 32)
                e.runRows("gdelta_step", [.buf(qn), .buf(kn), .buf(vLin), .buf(gBuf), .buf(betaBuf),
                                          .buf(recStates[i]!), .buf(linOut),
                                          .u32(UInt32(vHeads)), .u32(UInt32(kHeads)),
                                          .u32(UInt32(kDim)), .u32(UInt32(vDim))],
                          rows: vHeads, threadgroup: 128)
                e.runRows("rmsnorm_gated", [.buf(linOut), .buf(lw.normW), .buf(z), .buf(linOut),
                                            .u32(UInt32(vDim)), .f32(eps)],
                          rows: valueDim / vDim, threadgroup: 128)
                qmatvecAdd(e, lw.outProj, linOut, hidden, hidden)
            case .full(let fw):
                qmatvec(e, fw.qProj, xn, qkv)
                e.run("split_q_gate", [.buf(qkv), .buf(qBuf), .buf(gateBuf),
                                       .u32(UInt32(heads)), .u32(UInt32(headDim))],
                      grid: oProjIn, threadgroup: 256)
                qmatvec(e, fw.kProj, xn, kBuf)
                qmatvec(e, fw.vProj, xn, vBuf)
                if dumpEnabled && i == 3 {
                    e.run("copy_range", [.buf(qBuf), .buf(tapBuffers[0]), .u32(UInt32(oProjIn)), .u32(0)], grid: oProjIn, threadgroup: 256)
                }
                e.runRows("rmsnorm", [.buf(qBuf), .buf(fw.qNorm), .buf(qBuf),
                                      .u32(UInt32(headDim)), .f32(eps), .u32(0)],
                          rows: heads, threadgroup: 256)
                e.runRows("rmsnorm", [.buf(kBuf), .buf(fw.kNorm), .buf(kBuf),
                                      .u32(UInt32(headDim)), .f32(eps), .u32(0)],
                          rows: kvHeads, threadgroup: 256)
                if dumpEnabled && i == 3 {
                    e.run("copy_range", [.buf(qBuf), .buf(tapBuffers[1]), .u32(UInt32(oProjIn)), .u32(0)], grid: oProjIn, threadgroup: 256)
                    e.run("copy_range", [.buf(kBuf), .buf(tapBuffers[2]), .u32(UInt32(kvHeads * headDim)), .u32(0)], grid: kvHeads * headDim, threadgroup: 256)
                    e.run("copy_range", [.buf(vBuf), .buf(tapBuffers[4]), .u32(UInt32(kvHeads * headDim)), .u32(0)], grid: kvHeads * headDim, threadgroup: 256)
                }
                e.run("apply_rope", [.buf(qBuf), .u32(UInt32(heads)), .u32(UInt32(headDim)),
                                     .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)],
                      grid: heads * config.rotaryDim / 2, threadgroup: 128)
                e.run("apply_rope", [.buf(kBuf), .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                     .u32(ropeDim), .u32(UInt32(pos)), .f32(theta)],
                      grid: kvHeads * config.rotaryDim / 2, threadgroup: 128)
                if dumpEnabled && i == 3 {
                    e.run("copy_range", [.buf(qBuf), .buf(tapBuffers[3]), .u32(UInt32(oProjIn)), .u32(0)], grid: oProjIn, threadgroup: 256)
                }
                e.run("write_kv", [.buf(kBuf), .buf(vBuf), .buf(kCaches[i]!), .buf(vCaches[i]!),
                                   .u32(UInt32(kvHeads)), .u32(UInt32(headDim)),
                                   .u32(UInt32(maxT)), .u32(UInt32(pos))],
                      grid: kvHeads * headDim, threadgroup: 256)
                e.runRows("attention_decode", [.buf(qBuf), .buf(kCaches[i]!), .buf(vCaches[i]!),
                                               .buf(attnOut), .u32(UInt32(heads)), .u32(UInt32(kvHeads)),
                                               .u32(UInt32(headDim)), .u32(UInt32(pos + 1)), .f32(scaleAttn),
                                               .u32(UInt32(maxT))],
                          rows: heads, threadgroup: 256)
                if dumpEnabled && i == 3 {
                    e.run("copy_range", [.buf(attnOut), .buf(tapBuffers[5]), .u32(UInt32(oProjIn)), .u32(0)], grid: oProjIn, threadgroup: 256)
                }
                e.run("mul_sigmoid_gate", [.buf(attnOut), .buf(gateBuf), .u32(UInt32(oProjIn))],
                      grid: oProjIn, threadgroup: 256)
                qmatvecAdd(e, fw.oProj, attnOut, hidden, hidden)
            }

            e.runRows("rmsnorm", [.buf(hidden), .buf(w.postAttnNorms[i]), .buf(xn),
                                  .u32(UInt32(h)), .f32(eps), .u32(0)],
                      rows: 1, threadgroup: 256)
            qmatvecPairSilu(e, w.mlpGate[i], w.mlpUp[i], mlpGateBuf, xn)
            qmatvecAdd(e, w.mlpDown[i], mlpGateBuf, hidden, hidden)
            if dumpEnabled {
                e.run("copy_range", [.buf(hidden), .buf(dumpBuffers[i + 1]), .u32(UInt32(h)), .u32(0)],
                      grid: h, threadgroup: 256)
            }
        }

        e.runRows("rmsnorm", [.buf(hidden), .buf(w.finalNorm), .buf(xn),
                              .u32(UInt32(h)), .f32(eps), .u32(0)],
                  rows: 1, threadgroup: 256)
        qmatvec(e, w.embed, xn, logits)
        e.runRows("argmax_partial", [.buf(logits), .buf(partialVal), .buf(partialIdx),
                                     .u32(UInt32(vocab))],
                  rows: 256, threadgroup: 256)
        e.end()
        let encodeDone = Date()
        cb.commit()
        cb.waitUntilCompleted()
        let gpuDone = Date()
        if Model.timing {
            Model.encodeTime += encodeDone.timeIntervalSince(forwardStart)
            Model.gpuTime += gpuDone.timeIntervalSince(encodeDone)
        }

        let pv = partialVal.contents().bindMemory(to: Float.self, capacity: 256)
        let pi = partialIdx.contents().bindMemory(to: UInt32.self, capacity: 256)
        var best: Float = -.infinity
        var bi = 0
        for k in 0..<256 where pv[k] > best { best = pv[k]; bi = Int(pi[k]) }
        if dumpEnabled {
            dumps = dumpBuffers.map { readBf16($0, h) }
            dumpLogits = readBf16(logits, vocab)
            let tapCount = max(h, oProjIn)
            taps = tapBuffers.map { readBf16($0, tapCount) }
        }
        cacheLen = pos + 1
        return bi
    }
}

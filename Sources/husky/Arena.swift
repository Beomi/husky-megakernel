import Foundation
import Metal

/// Field layout inside the per-layer offset table (stride = 32 u32).
enum LayerField {
    static let stride = 32
    static let inNorm = 0, postNorm = 1
    static let gateW = 2, gateS = 3, gateB = 4
    static let upW = 5, upS = 6, upB = 7
    static let downW = 8, downS = 9, downB = 10
    static let q1W = 11, q1S = 12, q1B = 13   // linear: in_proj_qkv | full: q_proj
    static let q2W = 14, q2S = 15, q2B = 16   // linear: in_proj_z   | full: k_proj
    static let q3W = 17, q3S = 18, q3B = 19   // linear: in_proj_a   | full: v_proj
    static let q4W = 20, q4S = 21, q4B = 22   // linear: in_proj_b   | full: o_proj
    static let q5W = 23, q5S = 24, q5B = 25   // linear: out_proj
    static let x1 = 26, x2 = 27, x3 = 28, x4 = 29
    static let type = 30
    // linear: x1=convW x2=A_log x3=dt_bias x4=norm.weight
    // full:   x1=q_norm x2=k_norm
}

/// Flat configuration indices shared with the Metal kernel.
enum CfgIndex {
    static let hiddenSize = 0, headDim = 1, numHeads = 2, numKvHeads = 3
    static let qProjOut = 4, oProjIn = 5, inter = 6
    static let kHeads = 7, vHeads = 8, kDim = 9, vDim = 10
    static let keyDim = 11, valueDim = 12, convDim = 13, vocab = 14
    static let maxT = 15, numLayers = 16, pos = 17, ngroups = 18
    static let sHidden = 19, sXn = 20, sQkv = 21, sZ = 22, sAbg = 23
    static let sConvOut = 24, sQLin = 25, sKLin = 26, sVLin = 27, sQn = 28, sKn = 29
    static let sLinOut = 30, sQBuf = 31, sGateBuf = 32, sKBuf = 33, sVBuf = 34
    static let sAttnOut = 35, sMix = 36, sMlpGate = 37, sMlpUp = 38, sMlpDown = 39
    static let sLogits = 40, sPartialVal = 41, sPartialIdx = 42, sTokenOut = 43
    static let sBar = 44, sConvState = 45, sRecState = 46, sKCache = 47, sVCache = 48
    static let sEmbedW = 49, sEmbedS = 50, sEmbedB = 51, sFinalNorm = 52, sTokenIn = 53, sPos = 54
    static let count = 55
}

final class PackedArena {
    let arena: MTLBuffer
    let layerTable: MTLBuffer
    let cfg: [UInt32]
    let config: ModelConfig
    let totalSize: Int

    init(engine: MetalEngine, st: Safetensors, config: ModelConfig, ngroups: Int, maxT: Int) {
        self.config = config
        let prefix = "language_model.model."
        let hidden = config.hiddenSize
        let qProjOut = config.numAttentionHeads * config.headDim * 2
        let oProjIn = config.numAttentionHeads * config.headDim
        let inter = config.intermediateSize
        let kHeads = config.linearNumKeyHeads, vHeads = config.linearNumValueHeads
        let kDim = config.linearKeyHeadDim, vDim = config.linearValueHeadDim
        let keyDim = kDim * kHeads, valueDim = vDim * vHeads
        let convDim = keyDim * 2 + valueDim
        let vocab = config.vocabSize
        let kvHeads = config.numKeyValueHeads

        // ---- first pass: collect tensor offsets ----
        var cursor = 0
        func align16() { cursor = (cursor + 15) & ~15 }
        var off: [String: UInt32] = [:]
        var size = 0

        func placeQuant(_ base: String) {
            for (suf, _) in [(".weight", 4), (".scales", 2), (".biases", 2)] {
                let name = base + suf
                align16()
                off[name] = UInt32(cursor)
                cursor += st.info(name).length
            }
        }
        func placeBf16(_ name: String) {
            align16()
            off[name] = UInt32(cursor)
            cursor += st.info(name).length
        }

        placeQuant(prefix + "embed_tokens")
        placeBf16(prefix + "norm.weight")
        for i in 0..<config.numHiddenLayers {
            let lp = "\(prefix)layers.\(i)."
            placeBf16(lp + "input_layernorm.weight")
            placeBf16(lp + "post_attention_layernorm.weight")
            placeQuant(lp + "mlp.gate_proj")
            placeQuant(lp + "mlp.up_proj")
            placeQuant(lp + "mlp.down_proj")
            if config.layerTypes[i] == "linear_attention" {
                let la = lp + "linear_attn."
                placeQuant(la + "in_proj_qkv")
                placeQuant(la + "in_proj_z")
                placeQuant(la + "in_proj_a")
                placeQuant(la + "in_proj_b")
                placeQuant(la + "out_proj")
                placeBf16(la + "conv1d.weight")
                placeBf16(la + "A_log")
                placeBf16(la + "dt_bias")
                placeBf16(la + "norm.weight")
            } else {
                let sa = lp + "self_attn."
                placeQuant(sa + "q_proj")
                placeQuant(sa + "k_proj")
                placeQuant(sa + "v_proj")
                placeQuant(sa + "o_proj")
                placeBf16(sa + "q_norm.weight")
                placeBf16(sa + "k_norm.weight")
            }
        }
        size = cursor

        // ---- scratch / state layout (byte offsets) ----
        var scratch: [Int: UInt32] = [:]
        func alloc(_ count: Int, _ elem: Int) -> UInt32 {
            align16()
            let o = UInt32(cursor)
            cursor += count * elem
            return o
        }
        let sHidden = alloc(hidden, 2)
        let sXn = alloc(hidden, 2)
        let sQkv = alloc(max(qProjOut, convDim), 2)
        let sZ = alloc(valueDim, 2)
        let sAbg = alloc(4 * vHeads, 2)
        let sConvOut = alloc(convDim, 2)
        let sQLin = alloc(keyDim, 2)
        let sKLin = alloc(keyDim, 2)
        let sVLin = alloc(valueDim, 2)
        let sQn = alloc(keyDim, 2)
        let sKn = alloc(keyDim, 2)
        let sLinOut = alloc(valueDim, 2)
        let sQBuf = alloc(oProjIn, 2)
        let sGateBuf = alloc(oProjIn, 2)
        let sKBuf = alloc(kvHeads * config.headDim, 2)
        let sVBuf = alloc(kvHeads * config.headDim, 2)
        let sAttnOut = alloc(oProjIn, 2)
        let sMix = alloc(hidden, 2)
        let sMlpGate = alloc(inter, 2)
        let sMlpUp = alloc(inter, 2)
        let sMlpDown = alloc(hidden, 2)
        let sLogits = alloc(vocab, 2)
        let sPartialVal = alloc(ngroups, 4)
        let sPartialIdx = alloc(ngroups, 4)
        let sTokenOut = alloc(1, 4)
        let sTokenIn = alloc(1, 4)
        let sPos = alloc(1, 4)
        let sBar = alloc(1, 4)
        let sConvState = alloc(config.numHiddenLayers * convDim * 3, 2)
        let sRecState = alloc(config.numHiddenLayers * vHeads * vDim * kDim, 2)
        let sKCache = alloc(config.numHiddenLayers * kvHeads * maxT * config.headDim, 2)
        let sVCache = alloc(config.numHiddenLayers * kvHeads * maxT * config.headDim, 2)
        scratch = [
            CfgIndex.sHidden: sHidden, CfgIndex.sXn: sXn, CfgIndex.sQkv: sQkv, CfgIndex.sZ: sZ,
            CfgIndex.sAbg: sAbg, CfgIndex.sConvOut: sConvOut, CfgIndex.sQLin: sQLin, CfgIndex.sKLin: sKLin,
            CfgIndex.sVLin: sVLin, CfgIndex.sQn: sQn, CfgIndex.sKn: sKn, CfgIndex.sLinOut: sLinOut,
            CfgIndex.sQBuf: sQBuf, CfgIndex.sGateBuf: sGateBuf, CfgIndex.sKBuf: sKBuf, CfgIndex.sVBuf: sVBuf,
            CfgIndex.sAttnOut: sAttnOut, CfgIndex.sMix: sMix, CfgIndex.sMlpGate: sMlpGate, CfgIndex.sMlpUp: sMlpUp,
            CfgIndex.sMlpDown: sMlpDown, CfgIndex.sLogits: sLogits, CfgIndex.sPartialVal: sPartialVal,
            CfgIndex.sPartialIdx: sPartialIdx, CfgIndex.sTokenOut: sTokenOut, CfgIndex.sBar: sBar,
            CfgIndex.sConvState: sConvState, CfgIndex.sRecState: sRecState, CfgIndex.sKCache: sKCache,
            CfgIndex.sVCache: sVCache, CfgIndex.sTokenIn: sTokenIn, CfgIndex.sPos: sPos,
        ]

        guard cursor <= Int(engine.device.maxBufferLength) else {
            fatalError("arena \(cursor) exceeds maxBufferLength \(engine.device.maxBufferLength)")
        }
        let buf = engine.device.makeBuffer(length: cursor, options: .storageModeShared)!
        memset(buf.contents(), 0, cursor)
        // ---- second pass: copy weights ----
        func copyQuant(_ base: String) {
            for suf in [".weight", ".scales", ".biases"] {
                let name = base + suf
                st.copyTensor(name, into: buf.contents().advanced(by: Int(off[name]!)))
            }
        }
        copyQuant(prefix + "embed_tokens")
        st.copyTensor(prefix + "norm.weight", into: buf.contents().advanced(by: Int(off[prefix + "norm.weight"]!)))
        for i in 0..<config.numHiddenLayers {
            let lp = "\(prefix)layers.\(i)."
            for n in [lp + "input_layernorm.weight", lp + "post_attention_layernorm.weight"] {
                st.copyTensor(n, into: buf.contents().advanced(by: Int(off[n]!)))
            }
            copyQuant(lp + "mlp.gate_proj"); copyQuant(lp + "mlp.up_proj"); copyQuant(lp + "mlp.down_proj")
            if config.layerTypes[i] == "linear_attention" {
                let la = lp + "linear_attn."
                copyQuant(la + "in_proj_qkv"); copyQuant(la + "in_proj_z")
                copyQuant(la + "in_proj_a"); copyQuant(la + "in_proj_b"); copyQuant(la + "out_proj")
                for n in [la + "conv1d.weight", la + "A_log", la + "dt_bias", la + "norm.weight"] {
                    st.copyTensor(n, into: buf.contents().advanced(by: Int(off[n]!)))
                }
            } else {
                let sa = lp + "self_attn."
                copyQuant(sa + "q_proj"); copyQuant(sa + "k_proj"); copyQuant(sa + "v_proj"); copyQuant(sa + "o_proj")
                for n in [sa + "q_norm.weight", sa + "k_norm.weight"] {
                    st.copyTensor(n, into: buf.contents().advanced(by: Int(off[n]!)))
                }
            }
        }
        self.arena = buf
        self.totalSize = cursor

        // ---- layer offset table ----
        var table = [UInt32](repeating: 0, count: config.numHiddenLayers * LayerField.stride)
        func put(_ layer: Int, _ field: Int, _ value: UInt32) { table[layer * LayerField.stride + field] = value }
        for i in 0..<config.numHiddenLayers {
            let lp = "\(prefix)layers.\(i)."
            put(i, LayerField.inNorm, off[lp + "input_layernorm.weight"]!)
            put(i, LayerField.postNorm, off[lp + "post_attention_layernorm.weight"]!)
            func q(_ base: String, _ wf: Int, _ sf: Int, _ bf: Int) {
                put(i, wf, off[base + ".weight"]!); put(i, sf, off[base + ".scales"]!); put(i, bf, off[base + ".biases"]!)
            }
            q(lp + "mlp.gate_proj", LayerField.gateW, LayerField.gateS, LayerField.gateB)
            q(lp + "mlp.up_proj", LayerField.upW, LayerField.upS, LayerField.upB)
            q(lp + "mlp.down_proj", LayerField.downW, LayerField.downS, LayerField.downB)
            if config.layerTypes[i] == "linear_attention" {
                let la = lp + "linear_attn."
                q(la + "in_proj_qkv", LayerField.q1W, LayerField.q1S, LayerField.q1B)
                q(la + "in_proj_z", LayerField.q2W, LayerField.q2S, LayerField.q2B)
                q(la + "in_proj_a", LayerField.q3W, LayerField.q3S, LayerField.q3B)
                q(la + "in_proj_b", LayerField.q4W, LayerField.q4S, LayerField.q4B)
                q(la + "out_proj", LayerField.q5W, LayerField.q5S, LayerField.q5B)
                put(i, LayerField.x1, off[la + "conv1d.weight"]!)
                put(i, LayerField.x2, off[la + "A_log"]!)
                put(i, LayerField.x3, off[la + "dt_bias"]!)
                put(i, LayerField.x4, off[la + "norm.weight"]!)
                put(i, LayerField.type, 1)
            } else {
                let sa = lp + "self_attn."
                q(sa + "q_proj", LayerField.q1W, LayerField.q1S, LayerField.q1B)
                q(sa + "k_proj", LayerField.q2W, LayerField.q2S, LayerField.q2B)
                q(sa + "v_proj", LayerField.q3W, LayerField.q3S, LayerField.q3B)
                q(sa + "o_proj", LayerField.q4W, LayerField.q4S, LayerField.q4B)
                put(i, LayerField.x1, off[sa + "q_norm.weight"]!)
                put(i, LayerField.x2, off[sa + "k_norm.weight"]!)
                put(i, LayerField.type, 0)
            }
        }
        self.layerTable = engine.buffer(table)

        var c = [UInt32](repeating: 0, count: CfgIndex.count)
        c[CfgIndex.hiddenSize] = UInt32(hidden); c[CfgIndex.headDim] = UInt32(config.headDim)
        c[CfgIndex.numHeads] = UInt32(config.numAttentionHeads); c[CfgIndex.numKvHeads] = UInt32(kvHeads)
        c[CfgIndex.qProjOut] = UInt32(qProjOut); c[CfgIndex.oProjIn] = UInt32(oProjIn); c[CfgIndex.inter] = UInt32(inter)
        c[CfgIndex.kHeads] = UInt32(kHeads); c[CfgIndex.vHeads] = UInt32(vHeads)
        c[CfgIndex.kDim] = UInt32(kDim); c[CfgIndex.vDim] = UInt32(vDim)
        c[CfgIndex.keyDim] = UInt32(keyDim); c[CfgIndex.valueDim] = UInt32(valueDim)
        c[CfgIndex.convDim] = UInt32(convDim); c[CfgIndex.vocab] = UInt32(vocab)
        c[CfgIndex.maxT] = UInt32(maxT); c[CfgIndex.numLayers] = UInt32(config.numHiddenLayers)
        c[CfgIndex.ngroups] = UInt32(ngroups)
        for (k, v) in scratch { c[k] = v }
        c[CfgIndex.sEmbedW] = off[prefix + "embed_tokens.weight"]!
        c[CfgIndex.sEmbedS] = off[prefix + "embed_tokens.scales"]!
        c[CfgIndex.sEmbedB] = off[prefix + "embed_tokens.biases"]!
        c[CfgIndex.sFinalNorm] = off[prefix + "norm.weight"]!
        self.cfg = c
    }

    func setToken(_ token: Int, pos: Int) {
        let base = arena.contents()
        base.advanced(by: Int(cfg[CfgIndex.sTokenIn])).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(token)
        base.advanced(by: Int(cfg[CfgIndex.sPos])).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(pos)
    }

    func resetBar() {
        memset(arena.contents().advanced(by: Int(cfg[CfgIndex.sBar])), 0, 4)
    }

    /// Zero all recurrent/conv/KV state (states occupy the tail of the arena).
    func resetStates() {
        let start = Int(cfg[CfgIndex.sConvState])
        memset(arena.contents().advanced(by: start), 0, totalSize - start)
    }
}

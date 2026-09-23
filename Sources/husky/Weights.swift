import Foundation
import Metal

final class QuantTensor {
    let weight: MTLBuffer   // U32 [outN, inN/8]
    let scales: MTLBuffer   // F32 [outN, inN/64]
    let biases: MTLBuffer   // F32 [outN, inN/64]
    let outN: Int
    let inN: Int
    let transposed: Bool
    var halfT: MTLBuffer? = nil   // fp16 packed [N/32][K/32][32][32] for prefill

    init(weight: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer, outN: Int, inN: Int, transposed: Bool = false) {
        self.weight = weight; self.scales = scales; self.biases = biases
        self.outN = outN; self.inN = inN
        self.transposed = transposed
    }
}

enum LayerWeights {
    case linear(LinearLayerWeights)
    case full(FullLayerWeights)
}

struct LinearLayerWeights {
    let inProjQKV: QuantTensor
    let inProjZ: QuantTensor
    let inProjA: QuantTensor
    let inProjB: QuantTensor
    let outProj: QuantTensor
    let convW: MTLBuffer     // f32 [convDim*4]
    let aLog: MTLBuffer      // f32 [numVHeads]
    let dtBias: MTLBuffer    // f32 [numVHeads]
    let normW: MTLBuffer     // f32 [headVDim]
}

struct FullLayerWeights {
    let qProj: QuantTensor
    let kProj: QuantTensor
    let vProj: QuantTensor
    let oProj: QuantTensor
    let qNorm: MTLBuffer
    let kNorm: MTLBuffer
}

final class ModelWeights {
    let config: ModelConfig
    let embed: QuantTensor
    let finalNorm: MTLBuffer
    let layers: [LayerWeights]
    let inputNorms: [MTLBuffer]
    let postAttnNorms: [MTLBuffer]
    let mlpGate: [QuantTensor]
    let mlpUp: [QuantTensor]
    let mlpDown: [QuantTensor]

    let buildBF16: Bool

    init(engine: MetalEngine, st: Safetensors, config: ModelConfig, buildBF16: Bool = false) {
        self.config = config
        self.buildBF16 = buildBF16
        let prefix = "language_model.model."

        func quant(_ base: String, prefill: Bool = true) -> QuantTensor {
            let w = st.info(base + ".weight")
            let outN = w.shape[0], inN = w.shape[1] * 8
            let q = QuantTensor(
                weight: engine.privateBuffer(st.u32(base + ".weight")),
                scales: engine.privateBuffer(st.bf16(base + ".scales")),
                biases: engine.privateBuffer(st.bf16(base + ".biases")),
                outN: outN, inN: inN)
            if buildBF16 && prefill {
                precondition(outN % 32 == 0 && inN % 32 == 0, "packed GEMM requires multiples of 32")
                let dst = engine.device.makeBuffer(length: outN * inN * 2, options: .storageModePrivate)!
                let cb = engine.queue.makeCommandBuffer()!
                let e = Encoder(engine, cb)
                e.run("dequant_pack", [.buf(q.weight), .buf(q.scales), .buf(q.biases), .buf(dst),
                                            .u32(UInt32(outN)), .u32(UInt32(inN))],
                      grid: outN * inN, threadgroup: 256)
                e.end(); cb.commit(); cb.waitUntilCompleted()
                precondition(cb.status == .completed, "weight packing failed: \(String(describing: cb.error))")
                q.halfT = dst
            }
            return q
        }
        func fbuf(_ base: String) -> MTLBuffer { engine.privateBuffer(st.bf16(base)) }

        self.embed = quant(prefix + "embed_tokens", prefill: false)
        self.finalNorm = fbuf(prefix + "norm.weight")

        var layers: [LayerWeights] = []
        var inputNorms: [MTLBuffer] = []
        var postAttnNorms: [MTLBuffer] = []
        var gs: [QuantTensor] = []; var us: [QuantTensor] = []; var ds: [QuantTensor] = []

        for i in 0..<config.numHiddenLayers {
            let lp = "\(prefix)layers.\(i)."
            inputNorms.append(fbuf(lp + "input_layernorm.weight"))
            postAttnNorms.append(fbuf(lp + "post_attention_layernorm.weight"))
            gs.append(quant(lp + "mlp.gate_proj"))
            us.append(quant(lp + "mlp.up_proj"))
            ds.append(quant(lp + "mlp.down_proj"))

            if config.layerTypes[i] == "linear_attention" {
                let la = lp + "linear_attn."
                layers.append(.linear(LinearLayerWeights(
                    inProjQKV: quant(la + "in_proj_qkv"),
                    inProjZ: quant(la + "in_proj_z"),
                    inProjA: quant(la + "in_proj_a"),
                    inProjB: quant(la + "in_proj_b"),
                    outProj: quant(la + "out_proj"),
                    convW: fbuf(la + "conv1d.weight"),
                    aLog: fbuf(la + "A_log"),
                    dtBias: fbuf(la + "dt_bias"),
                    normW: fbuf(la + "norm.weight"))))
            } else {
                let sa = lp + "self_attn."
                layers.append(.full(FullLayerWeights(
                    qProj: quant(sa + "q_proj"),
                    kProj: quant(sa + "k_proj"),
                    vProj: quant(sa + "v_proj"),
                    oProj: quant(sa + "o_proj"),
                    qNorm: fbuf(sa + "q_norm.weight"),
                    kNorm: fbuf(sa + "k_norm.weight"))))
            }
        }
        self.layers = layers
        self.inputNorms = inputNorms
        self.postAttnNorms = postAttnNorms
        self.mlpGate = gs; self.mlpUp = us; self.mlpDown = ds
    }
}

import Foundation

struct ModelConfig {
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var vocabSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var layerTypes: [String]

    // linear attention (gated delta net)
    var linearConvKernelDim: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearNumKeyHeads: Int
    var linearNumValueHeads: Int

    var quantGroupSize: Int
    var quantBits: Int

    var fullAttentionLayers: [Int] { layerTypes.enumerated().filter { $0.element == "full_attention" }.map { $0.offset } }
    var linearAttentionLayers: [Int] { layerTypes.enumerated().filter { $0.element == "linear_attention" }.map { $0.offset } }

    var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    static func load(path: String) throws -> ModelConfig {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let t = root["text_config"] as! [String: Any]
        func i(_ k: String) -> Int { (t[k] as? NSNumber)?.intValue ?? 0 }
        func f(_ k: String) -> Float { (t[k] as? NSNumber)?.floatValue ?? 0 }
        let q = (t["quantization_config"] as? [String: Any]) ?? [:]
        var layerTypes = (t["layer_types"] as? [String]) ?? []
        if layerTypes.isEmpty {
            let n = i("num_hidden_layers")
            let interval = i("full_attention_interval")
            layerTypes = (0..<n).map { (($0 + 1) % interval == 0) ? "full_attention" : "linear_attention" }
        }
        return ModelConfig(
            hiddenSize: i("hidden_size"),
            intermediateSize: i("intermediate_size"),
            numHiddenLayers: i("num_hidden_layers"),
            numAttentionHeads: i("num_attention_heads"),
            numKeyValueHeads: i("num_key_value_heads"),
            headDim: i("head_dim"),
            vocabSize: i("vocab_size"),
            rmsNormEps: f("rms_norm_eps"),
            ropeTheta: ((t["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber)?.floatValue ?? 10000,
            partialRotaryFactor: ((t["rope_parameters"] as? [String: Any])?["partial_rotary_factor"] as? NSNumber)?.floatValue ?? 1.0,
            layerTypes: layerTypes,
            linearConvKernelDim: i("linear_conv_kernel_dim"),
            linearKeyHeadDim: i("linear_key_head_dim"),
            linearValueHeadDim: i("linear_value_head_dim"),
            linearNumKeyHeads: i("linear_num_key_heads"),
            linearNumValueHeads: i("linear_num_value_heads"),
            quantGroupSize: (q["group_size"] as? NSNumber)?.intValue ?? 64,
            quantBits: (q["bits"] as? NSNumber)?.intValue ?? 4
        )
    }
}

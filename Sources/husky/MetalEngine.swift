import Foundation
import Metal

final class MetalEngine {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "husky", code: 1, userInfo: [NSLocalizedDescriptionKey: "no Metal device"])
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "husky", code: 2, userInfo: [NSLocalizedDescriptionKey: "no command queue"])
        }
        self.queue = queue
        guard let url = Bundle.module.url(forResource: "kernels", withExtension: "metal") else {
            throw NSError(domain: "husky", code: 3, userInfo: [NSLocalizedDescriptionKey: "kernels.metal not found"])
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        self.library = try device.makeLibrary(source: src, options: opts)

        // megakernel is a separate translation unit
        if let mkURL = Bundle.module.url(forResource: "megakernel", withExtension: "metal") {
            let mkSrc = try String(contentsOf: mkURL, encoding: .utf8)
            let mkLib = try device.makeLibrary(source: mkSrc, options: opts)
            let constants = MTLFunctionConstantValues()
            var threads = UInt32(MegakernelModel.TGS)
            constants.setConstantValue(&threads, type: .uint, index: 0)
            let function = try mkLib.makeFunction(name: "husky_step", constantValues: constants)
            pipelines["husky_step"] = try device.makeComputePipelineState(function: function)
        }

        // chunked-prefill kernels
        if let pkURL = Bundle.module.url(forResource: "prefillkernel", withExtension: "metal") {
            let pkSrc = try String(contentsOf: pkURL, encoding: .utf8)
            let pkLib = try device.makeLibrary(source: pkSrc, options: opts)
            for name in ["qmatvec_prefill", "qmatvec_pair_silu_prefill", "qmatvec_add_prefill",
                         "conv_chunk", "gdelta_chunk", "gdelta_chunk_register", "attention_chunk_simd", "apply_rope_chunk", "write_kv_chunk", "attention_chunk",
                         "dequant_transpose", "dequant_pack", "gemm_mma"] {
                pipelines[name] = try device.makeComputePipelineState(function: pkLib.makeFunction(name: name)!)
            }
        }

        // batched kernels live in a second translation unit
        if let bkURL = Bundle.module.url(forResource: "batchkernel", withExtension: "metal") {
            let bkSrc = try String(contentsOf: bkURL, encoding: .utf8)
            let bkLib = try device.makeLibrary(source: bkSrc, options: opts)
            for name in ["qmatvec_b", "qmatvec_pair_silu_b", "qmatvec_add_b", "embed_lookup_b",
                         "conv_step_b", "g_beta_b", "gdelta_step_b", "write_kv_b",
                         "attention_decode_b", "apply_rope_b", "split_q_gate_b", "argmax_b", "copy_range_b"] {
                pipelines[name] = try device.makeComputePipelineState(function: bkLib.makeFunction(name: name)!)
            }
        }

        for name in ["qmatvec_warp", "qmatvec_warp_add", "qmatvec_pair_silu", "embed_lookup", "rmsnorm", "rmsnorm_gated", "apply_rope",
                     "conv_step", "l2norm_scale", "g_beta", "gdelta_step", "attention_decode",
                     "write_kv",
                     "mul_sigmoid_gate", "silu_mul", "add_inplace", "copy_range", "split_q_gate", "argmax_partial"] {
            let fn = library.makeFunction(name: name)!
            pipelines[name] = try device.makeComputePipelineState(function: fn)
        }
    }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        pipelines[name]!
    }

    func buffer<Element>(_ values: [Element]) -> MTLBuffer {
        let len = values.count * MemoryLayout<Element>.stride
        return values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: max(len, 4), options: .storageModeShared)!
        }
    }

    /// GPU-private buffer initialized from CPU data (weights are read-only).
    func privateBuffer<Element>(_ values: [Element]) -> MTLBuffer {
        let len = max(values.count * MemoryLayout<Element>.stride, 4)
        let staged = values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: len, options: .storageModeShared)!
        }
        let dst = device.makeBuffer(length: len, options: .storageModePrivate)!
        let cb = queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: staged, sourceOffset: 0, to: dst, destinationOffset: 0, size: len)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return dst
    }

    func empty(_ count: Int) -> MTLBuffer {
        device.makeBuffer(length: max(count * 4, 4), options: .storageModeShared)!
    }

    func emptyZeros(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: max(count * 4, 4), options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }

    /// bfloat16 activation buffer (2 bytes/element).
    func emptyBf16(_ count: Int) -> MTLBuffer {
        device.makeBuffer(length: max(count * 2, 4), options: .storageModeShared)!
    }

    func emptyBf16Zeros(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: max(count * 2, 4), options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }
}

enum Arg {
    case buf(MTLBuffer)
    case bufOffset(MTLBuffer, Int)   // byte offset
    case u32(UInt32)
    case f32(Float)
}

/// A command encoder wrapper for dispatching a sequence of kernels.
final class Encoder {
    var enc: MTLComputeCommandEncoder
    let originalEncoder: MTLComputeCommandEncoder
    let engine: MetalEngine
    let profile: Bool
    var samples: [(String, MTLCommandBuffer)] = []

    init(_ engine: MetalEngine, _ cb: MTLCommandBuffer, profile: Bool = false) {
        self.engine = engine
        self.profile = profile
        self.originalEncoder = cb.makeComputeCommandEncoder()!
        self.enc = originalEncoder
    }

    private func beginSample(_ name: String) {
        if profile {
            let cb = engine.queue.makeCommandBuffer()!
            cb.label = name
            enc = cb.makeComputeCommandEncoder()!
            samples.append((name, cb))
        }
    }

    private func endSample() {
        if profile { enc.endEncoding(); samples.last!.1.commit(); samples.last!.1.waitUntilCompleted() }
    }

    // Separate command buffers perturb scheduling: diagnostics only, never TPS.
    func reportProfile() {
        var totals: [String: Double] = [:]
        for (name, cb) in samples {
            precondition(cb.status == .completed, "profile command failed: \(String(describing: cb.error))")
            totals[name, default: 0] += cb.gpuEndTime - cb.gpuStartTime
        }
        for (name, seconds) in totals.sorted(by: { $0.value > $1.value }) {
            print(String(format: "profile %-28@ %9.3f ms", name as NSString, seconds * 1000))
        }
    }

    private func bind(_ args: [Arg], pipeline: MTLComputePipelineState) {
        enc.setComputePipelineState(pipeline)
        for (i, a) in args.enumerated() {
            switch a {
            case .buf(let b): enc.setBuffer(b, offset: 0, index: i)
            case .bufOffset(let b, let o): enc.setBuffer(b, offset: o, index: i)
            case .u32(var v): enc.setBytes(&v, length: 4, index: i)
            case .f32(var v): enc.setBytes(&v, length: 4, index: i)
            }
        }
    }

    /// Grid-stride dispatch: `grid` total threads, grouped in `threadgroup`-sized groups.
    func run(_ name: String, _ args: [Arg], grid: Int, threadgroup: Int) {
        beginSample(name)
        let p = engine.pipeline(name)
        bind(args, pipeline: p)
        let tg = min(threadgroup, p.maxTotalThreadsPerThreadgroup)
        let groups = (grid + tg - 1) / tg
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        endSample()
    }

    /// 2D threadgroup grid.
    func run2D(_ name: String, _ args: [Arg], grid: (Int, Int), threadgroup: Int) {
        beginSample(name)
        let p = engine.pipeline(name)
        bind(args, pipeline: p)
        enc.dispatchThreadgroups(MTLSize(width: grid.0, height: grid.1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1))
        endSample()
    }

    /// One threadgroup per row.
    func runRows(_ name: String, _ args: [Arg], rows: Int, threadgroup: Int) {
        beginSample(name)
        let p = engine.pipeline(name)
        bind(args, pipeline: p)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1))
        endSample()
    }

    func end() { originalEncoder.endEncoding() }
}

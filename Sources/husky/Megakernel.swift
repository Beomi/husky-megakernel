import Foundation
import Metal

/// Host driver for the single persistent megakernel. One dispatch runs all
/// layers of the model for one token; generation is one dispatch per token.
final class MegakernelModel {
    let engine: MetalEngine
    let packed: PackedArena
    let ngroups: Int
    let cfgBuf: MTLBuffer
    let tokenOutBuf: MTLBuffer
    static let TGS = Int(ProcessInfo.processInfo.environment["HUSKY_MEGAKERNEL_TG"] ?? "1024") ?? 1024
    var gpuTime: Double = 0
    var steps = 0

    init(engine: MetalEngine, st: Safetensors, config: ModelConfig, ngroups: Int = 1, maxT: Int = 4096) {
        precondition([256, 512, 1024].contains(Self.TGS))
        self.engine = engine
        self.ngroups = ngroups
        self.packed = PackedArena(engine: engine, st: st, config: config, ngroups: ngroups, maxT: maxT)
        self.cfgBuf = engine.buffer(packed.cfg)
        self.tokenOutBuf = engine.empty(1)
    }

    func reset() {
        packed.resetStates()
        packed.resetBar()
    }

    func forward(token: Int, pos: Int) -> Int {
        packed.setToken(token, pos: pos)
        packed.resetBar()
        let cb = engine.queue.makeCommandBuffer()!
        let e = Encoder(engine, cb)
        e.runRows("husky_step",
                  [.buf(packed.arena), .buf(packed.layerTable), .buf(cfgBuf), .buf(tokenOutBuf)],
                  rows: ngroups, threadgroup: Self.TGS)
        e.end()
        cb.commit()
        cb.waitUntilCompleted()
        precondition(cb.status == .completed, "megakernel failed: \(String(describing: cb.error))")
        gpuTime += cb.gpuEndTime - cb.gpuStartTime
        steps += 1
        return Int(tokenOutBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}

func runMegakernelBench(engine: MetalEngine, st: Safetensors, config: ModelConfig, tokenizer: Tokenizer) {
    let model = MegakernelModel(engine: engine, st: st, config: config, maxT: 64)
    let prompt = tokenizer.encode("2+2=")
    let golden = [19,3709,19,10,19,28,23,3709,23,10,19,28]
    var samples: [Double] = []
    for trial in 0..<4 {
        model.reset()
        let before = model.gpuTime
        var next = 0
        for (pos, token) in prompt.enumerated() { next = model.forward(token: token, pos: pos) }
        var generated = [next]
        for pos in prompt.count..<(prompt.count + golden.count - 1) {
            next = model.forward(token: next, pos: pos); generated.append(next)
        }
        precondition(generated == golden, "megakernel golden mismatch: \(generated)")
        if trial > 0 { samples.append((model.gpuTime - before) / Double(prompt.count + golden.count - 1)) }
    }
    let dt = samples.sorted()[samples.count/2]
    print(String(format: "megakernel groups=1 threads=%d: %.3f ms/token, %.3f TPS (GPU median of 3 warmed 15-step runs); 12/12 golden tokens match", MegakernelModel.TGS, dt*1000, 1/dt))
}

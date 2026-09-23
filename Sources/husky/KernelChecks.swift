import Foundation
import Metal

func bfValues(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let p = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
    return (0..<count).map { Float(bitPattern: UInt32(p[$0]) << 16) }
}

func checkClose(_ label: String, _ actual: [Float], _ expected: [Float], tolerance: Float) {
    precondition(actual.count == expected.count)
    var err: Double = 0, norm: Double = 0, maxError: Float = 0
    for (a, b) in zip(actual, expected) {
        precondition(a.isFinite && b.isFinite, "\(label): non-finite value")
        err += Double(a-b)*Double(a-b); norm += Double(b)*Double(b)
        maxError = max(maxError, abs(a-b))
    }
    let relative = sqrt(err / max(norm, 1e-20))
    if relative > Double(tolerance) { fatalError("\(label): relative RMSE \(relative) > \(tolerance)") }
    print(String(format: "%@: relative RMSE %.6f, max abs %.6f", label, relative, maxError))
}

/// Numerical checks cover every output and state element, not only argmax.
func runKernelChecks(engine: MetalEngine) {
    func data(_ n: Int, _ seed: Int, _ scale: Float = 1) -> MTLBuffer {
        engine.buffer((0..<n).map { f32ToBf16Bits(Float((($0*37+seed*19)%127)-63)/64*scale) })
    }
    func run(_ name: String, _ args: [Arg], rows: Int, threads: Int) {
        let cb = engine.queue.makeCommandBuffer()!, e = Encoder(engine, cb)
        e.runRows(name, args, rows: rows, threadgroup: threads)
        e.end(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.status == .completed, "kernel check failed: \(String(describing: cb.error))")
    }
    for c in [1, 5, 65, 257] {
        let vh=4, kh=2, kd=128, vd=8
        let q=data(c*kh*kd,1,0.05), k=data(c*kh*kd,2,0.05), v=data(c*vh*vd,3)
        let g=engine.buffer([UInt16](repeating: f32ToBf16Bits(0.95),count:c*vh))
        let beta=engine.buffer([UInt16](repeating:f32ToBf16Bits(0.3),count:c*vh))
        let s0=data(vh*vd*kd,4,0.01), s1=data(vh*vd*kd,4,0.01)
        let o0=engine.emptyBf16(c*vh*vd), o1=engine.emptyBf16(c*vh*vd)
        func args(_ state: MTLBuffer, _ out: MTLBuffer) -> [Arg] {
            [.buf(q),.buf(k),.buf(v),.buf(g),.buf(beta),.buf(state),.buf(out),.u32(UInt32(vh)),.u32(UInt32(kh)),.u32(UInt32(kd)),.u32(UInt32(vd)),.u32(UInt32(c))]
        }
        run("gdelta_chunk",args(s0,o0),rows:vh,threads:128)
        run("gdelta_chunk_register",args(s1,o1),rows:vh,threads:128)
        checkClose("delta output C=\(c)",bfValues(o1,count:c*vh*vd),bfValues(o0,count:c*vh*vd),tolerance:0.004)
        checkClose("delta state C=\(c)",bfValues(s1,count:vh*vd*kd),bfValues(s0,count:vh*vd*kd),tolerance:0.004)
        let heads=4, kv=2, dim=256, pos=7, maxT=c+pos
        let aq=data(c*heads*dim,5), ak=data(maxT*kv*dim,6), av=data(maxT*kv*dim,7)
        let ao0=engine.emptyBf16(c*heads*dim), ao1=engine.emptyBf16(c*heads*dim)
        func attnArgs(_ out: MTLBuffer) -> [Arg] {
            [.buf(aq),.buf(ak),.buf(av),.buf(out),.u32(UInt32(heads)),.u32(UInt32(kv)),.u32(UInt32(dim)),.u32(UInt32(pos)),.f32(1/16),.u32(UInt32(maxT))]
        }
        run("attention_chunk",attnArgs(ao0),rows:c*heads,threads:256)
        run("attention_chunk_simd",attnArgs(ao1),rows:c*heads,threads:256)
        checkClose("attention C=\(c), prefix=7",bfValues(ao1,count:c*heads*dim),bfValues(ao0,count:c*heads*dim),tolerance:0.004)
    }
    for m in [1, 5, 63, 64, 65, 128] {
        let n=64, k=96
        let x=data(m*k,11,0.2), y=engine.emptyBf16(m*n)
        let xv=bfValues(x,count:m*k)
        var packed=[Float16](repeating:0,count:n*k)
        var dense=[Float](repeating:0,count:n*k)
        for col in 0..<n { for inner in 0..<k {
            let value=Float16(Float((col*13+inner*7)%101-50)/257)
            dense[inner*n+col]=Float(value)
            packed[(col/32)*(k/32)*1024+(inner/32)*1024+(inner%32)*32+col%32]=value
        } }
        let b=engine.buffer(packed)
        let cb=engine.queue.makeCommandBuffer()!, e=Encoder(engine,cb)
        e.run2D("gemm_mma",[.buf(x),.buf(b),.buf(y),.u32(UInt32(m)),.u32(UInt32(n)),.u32(UInt32(k))],grid:((m+63)/64,n/32),threadgroup:256)
        e.end(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.status == .completed)
        var expected=[Float](repeating:0,count:m*n)
        for row in 0..<m { for col in 0..<n {
            var sum:Double=0
            for inner in 0..<k { sum += Double(Float(Float16(xv[row*k+inner])))*Double(dense[inner*n+col]) }
            expected[row*n+col]=Float(bitPattern:UInt32(f32ToBf16Bits(Float(sum)))<<16)
        } }
        checkClose("packed GEMM M=\(m)",bfValues(y,count:m*n),expected,tolerance:0.004)
    }
    print("Kernel numerical checks passed.")
}

func runPrefillChecks(engine: MetalEngine, st: Safetensors, config: ModelConfig, tokenizer: Tokenizer) {
    let w = ModelWeights(engine: engine, st: st, config: config, buildBF16: true)
    let corpus = tokenizer.encode("The history of the Roman Empire is a long story of rise and fall, ambition and ruin. ")
    for (n, chunk) in [(1,64), (5,64), (63,64), (64,64), (65,64), (129,64), (257,64), (256,256), (512,256)] {
        let ids = (0..<n).map { corpus[$0 % corpus.count] }
        let ref = PrefillModel(engine: engine, weights: w, chunk: chunk, maxT: 1024, optimized: false)
        let opt = PrefillModel(engine: engine, weights: w, chunk: chunk, maxT: 1024)
        let r = ref.prefill(ids), a = opt.prefill(ids)
        precondition(r == a, "argmax mismatch at length \(n): \(r) vs \(a)")
        checkClose("prefill logits length=\(n)",bfValues(opt.logits,count:config.vocabSize),bfValues(ref.logits,count:config.vocabSize),tolerance:0)
        // Cross-check handoff with continued single-token prefill. This exercises
        // conv, recurrent and KV caches, including a partially filled last chunk.
        let decoder = Model(engine: engine, weights: w, maxT: 1024)
        opt.handoff(to: decoder, tokenCount: n)
        let next = decoder.forward(token:a,pos:n)
        _ = opt.prefill(ids)
        let expected = opt.forwardChunk([a][...],start:0,C:1,pos:n,last:true)!
        precondition(next == expected, "decode handoff mismatch at length \(n)")
        print("length \(n): argmax and decode handoff OK")
    }
    let ids = tokenizer.encode("2+2=")
    let prefill = PrefillModel(engine: engine, weights:w,chunk:3,maxT:64)
    let decoder = Model(engine:engine,weights:w,maxT:64)
    var token = prefill.prefill(ids)
    prefill.handoff(to:decoder,tokenCount:ids.count)
    var generated=[token]
    for p in ids.count..<(ids.count+11) { token=decoder.forward(token:token,pos:p); generated.append(token) }
    let expected=[19,3709,19,10,19,28,23,3709,23,10,19,28]
    precondition(generated == expected, "golden continuation mismatch: \(generated)")
    print("2+2= chunk=3, 12-token continuation: exact match")
    print("Prefill integration checks passed.")
}

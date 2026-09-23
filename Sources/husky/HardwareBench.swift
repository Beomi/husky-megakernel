import Foundation
import Metal

/// Measured sustained rates, not vendor peak FLOPS. No model weights required.
func runHardwareBench(engine: MetalEngine) throws {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void bandwidth(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
    kernel void peak(device const half* a [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint& iterations [[buffer(2)]], uint g [[threadgroup_position_in_grid]],
                     uint s [[simdgroup_index_in_threadgroup]]) {
        simdgroup_float8x8 c0(0), c1(0), c2(0), c3(0);
        for (uint i=0; i<iterations; ++i) {
            simdgroup_half8x8 x, y;
            simdgroup_load(x, a + (i & 255u)*128u, 8);
            simdgroup_load(y, a + (i & 255u)*128u + 64u, 8);
            simdgroup_multiply_accumulate(c0,x,y,c0);
            simdgroup_multiply_accumulate(c1,y,x,c1);
            simdgroup_multiply_accumulate(c2,x,x,c2);
            simdgroup_multiply_accumulate(c3,y,y,c3);
        }
        uint base = (g*4u+s)*256u;
        simdgroup_store(c0,out+base,8); simdgroup_store(c1,out+base+64,8);
        simdgroup_store(c2,out+base+128,8); simdgroup_store(c3,out+base+192,8);
    }
    """
    let lib = try engine.device.makeLibrary(source: source, options: nil)
    let copy = try engine.device.makeComputePipelineState(function: lib.makeFunction(name: "bandwidth")!)
    let peak = try engine.device.makeComputePipelineState(function: lib.makeFunction(name: "peak")!)
    func time(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        body(enc); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        precondition(cb.status == .completed, "GPU benchmark failed: \(String(describing: cb.error))")
        return cb.gpuEndTime - cb.gpuStartTime
    }
    func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
    print("device: \(engine.device.name); threadgroup memory: \(engine.device.maxThreadgroupMemoryLength) bytes")
    let bytes = 256 * 1024 * 1024
    let a = engine.device.makeBuffer(length: bytes, options: .storageModeShared)!
    let b = engine.device.makeBuffer(length: bytes, options: .storageModeShared)!
    memset(a.contents(), 0x3c, bytes)
    var copyTimes: [Double] = []
    for trial in 0..<8 {
        let dt = time { e in
            e.setComputePipelineState(copy); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
            e.dispatchThreads(MTLSize(width: bytes / 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if trial >= 2 { copyTimes.append(dt) }
    }
    precondition(memcmp(a.contents(), b.contents(), bytes) == 0)
    let bandwidth = Double(bytes * 2) / median(copyTimes) / 1e9
    print(String(format: "256 MiB copy, read+write: %.1f GB/s (median of 6 warmed runs)", bandwidth))
    // Distinct, non-constant operands prevent identical accumulator elimination.
    let values = (0..<(256*128)).map { Float16(($0 % 17 + 1)) / 512 }
    let input = engine.buffer(values)
    var best = 0.0
    for groups in [1024, 4096, 8192] {
        let output = engine.empty(groups * 4 * 256)
        var durations: [Double] = []
        for trial in 0..<7 {
            let dt = time { e in
                var iterations: UInt32 = 256
                e.setComputePipelineState(peak); e.setBuffer(input, offset: 0, index: 0); e.setBuffer(output, offset: 0, index: 1)
                e.setBytes(&iterations, length: 4, index: 2)
                e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            if trial >= 2 { durations.append(dt) }
        }
        // Check all four chains against CPU matrix multiplication for one SIMD group.
        let actual = output.contents().bindMemory(to: Float.self, capacity: groups * 1024)
        var maxError: Float = 0
        for chain in 0..<4 { for r in 0..<8 { for c in 0..<8 {
            var expected: Float = 0
            for i in 0..<256 { for k in 0..<8 {
                let xb = (chain == 1 || chain == 3) ? 64 : 0
                let yb = (chain == 0 || chain == 3) ? 64 : 0
                expected += Float(values[i*128+xb+r*8+k]) * Float(values[i*128+yb+k*8+c])
            } }
            maxError = max(maxError, abs(actual[chain*64+r*8+c] - expected))
        } } }
        precondition(maxError < 0.001, "MMA validation failed: \(maxError)")
        let tmac = Double(groups) * 4 * 256 * 4 * 512 / median(durations) / 1e12
        best = max(best, tmac)
        print(String(format: "MMA groups=%d: %.3f ms, %.2f TMAC/s = %.2f TFLOP/s; max error %.6f", groups, median(durations)*1000, tmac, tmac*2, maxError))
    }
    print(String(format: "Measured compute-only projection bound (Woof, 3.569090560 GMAC/token): %.0f tokens/s", best * 1e12 / 3_569_090_560))
    print("This excludes staging, recurrence, attention and output logits; it is not an end-to-end TPS guarantee.")
}

#include <metal_stdlib>
using namespace metal;

// bfloat16 helpers (Metal does not expose bf16 on all toolchains).
inline ushort f2bf_bits(float f) {
    uint x = as_type<uint>(f);
    uint lsb = (x >> 16) & 1u;
    uint r = (x + 0x7FFFu + lsb) >> 16;
    return ushort(r);
}
struct bf16 {
    ushort bits;
    bf16() : bits(0) {}
    bf16(float f) : bits(f2bf_bits(f)) {}
};
inline float bf2f(bf16 x) { return as_type<float>(uint(x.bits) << 16); }

// Activations and state are bfloat16 (matching the model dtype); all arithmetic
// accumulates in float32 and is rounded back to bfloat16 on store, mirroring the
// reference (MLX) engine.

// ---------------------------------------------------------------------------
// Quantized (MLX affine, 4-bit, group 64) matrix-vector product.
// W: uint32[outN * inN/8], S/B: bf16[outN * inN/64], X: bf16[inN] -> Y[outN]
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Coalesced quantized matvec: one warp per output row, 128-bit weight loads,
// warp-level reduction. Grid = ceil(outN / warpsPerThreadgroup).
// ---------------------------------------------------------------------------
kernel void qmatvec_warp(
    device const uint*       W   [[buffer(0)]],
    device const bf16*       S   [[buffer(1)]],
    device const bf16*       B   [[buffer(2)]],
    device const bf16*       X   [[buffer(3)]],
    device       bf16*       Y   [[buffer(4)]],
    constant     uint&       outN [[buffer(5)]],
    constant     uint&       inN  [[buffer(6)]],
    constant     uint&       totalWarps [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u;
    uint nGroups = inN >> 6u;
    uint nVec = inWords >> 2u;

    for (uint warp = warp0; warp < outN; warp += totalWarps) {
        device const uint* Wrow = W + (ulong)warp * inWords;
        device const bf16* Srow = S + (ulong)warp * nGroups;
        device const bf16* Brow = B + (ulong)warp * nGroups;
        float ac[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint v = lane; v < nVec; v += 32u) {
            uint word0 = v << 2u;
            uint g = word0 >> 3u;
            float sc = bf2f(Srow[g]);
            float bi = bf2f(Brow[g]);
            uint base = word0 << 3u;
            uint4 wv = *(device const uint4*)(Wrow + word0);
            #pragma unroll
            for (uint t = 0; t < 4u; ++t) {
                uint ww = (t == 0u) ? wv.x : (t == 1u) ? wv.y : (t == 2u) ? wv.z : wv.w;
                uint b8 = base + (t << 3u);
                #pragma unroll
                for (uint k = 0; k < 8u; ++k) {
                    float q = float((ww >> (4u * k)) & 0xFu);
                    ac[t] += (q * sc + bi) * bf2f(X[b8 + k]);
                }
            }
        }
        float acc = simd_sum(ac[0] + ac[1] + ac[2] + ac[3]);
        if (lane == 0u) Y[warp] = bf16(acc);
    }
}

// Two weight matrices, one shared X, one warp per output row: Y1 = A1*X, Y2 = A2*X.

// ---------------------------------------------------------------------------
// Weight transposition at load: src [rows, cols] -> dst [cols, rows].
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Matvec against transposed weights: thread-per-output, coalesced reads across
// consecutive outputs, no cross-thread reduction.
// W_T: [in/8, out], S_T/B_T: [in/64, out].
// ---------------------------------------------------------------------------

// One thread per output row with 128-bit weight loads (high per-thread MLP).

// Two output rows per warp: X is loaded once and reused for both rows,
// doubling per-warp work and memory-level parallelism.

// Warp-per-row matvec writing Y = matvec + residual (folds the residual add).
kernel void qmatvec_warp_add(
    device const uint* W [[buffer(0)]], device const bf16* S [[buffer(1)]], device const bf16* B [[buffer(2)]],
    device const bf16* X [[buffer(3)]], device const bf16* R [[buffer(4)]], device bf16* Y [[buffer(5)]],
    constant uint& outN [[buffer(6)]], constant uint& inN [[buffer(7)]], constant uint& totalWarps [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint warp = warp0; warp < outN; warp += totalWarps) {
        device const uint* Wrow = W + (ulong)warp * inWords;
        device const bf16* Srow = S + (ulong)warp * nGroups;
        device const bf16* Brow = B + (ulong)warp * nGroups;
        float acc = 0.0f;
        for (uint v = lane; v < nVec; v += 32u) {
            uint word0 = v << 2u;
            uint g = word0 >> 3u;
            float sc = bf2f(Srow[g]);
            float bi = bf2f(Brow[g]);
            uint b8 = word0 << 3u;
            uint4 w = *(device const uint4*)(Wrow + word0);
            #pragma unroll
            for (uint t = 0; t < 4u; ++t) {
                uint ww = (t == 0u) ? w.x : (t == 1u) ? w.y : (t == 2u) ? w.z : w.w;
                uint bb = b8 + (t << 3u);
                #pragma unroll
                for (uint k = 0; k < 8u; ++k) {
                    float q = float((ww >> (4u * k)) & 0xFu);
                    acc += (q * sc + bi) * bf2f(X[bb + k]);
                }
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) Y[warp] = bf16(acc + bf2f(R[warp]));
    }
}

// gate/up pair writing silu(gate)*up (folds SwiGLU).
kernel void qmatvec_pair_silu(
    device const uint* W1 [[buffer(0)]], device const bf16* S1 [[buffer(1)]], device const bf16* B1 [[buffer(2)]],
    device const uint* W2 [[buffer(3)]], device const bf16* S2 [[buffer(4)]], device const bf16* B2 [[buffer(5)]],
    device const bf16* X  [[buffer(6)]], device bf16* Y [[buffer(7)]],
    constant uint& outN [[buffer(8)]], constant uint& inN [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp = tg * (tgSize >> 5u) + (tid >> 5u);
    if (warp >= outN) return;
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    device const uint* A1 = W1 + (ulong)warp * inWords;
    device const uint* A2 = W2 + (ulong)warp * inWords;
    device const bf16* s1 = S1 + (ulong)warp * nGroups;
    device const bf16* s2 = S2 + (ulong)warp * nGroups;
    device const bf16* b1 = B1 + (ulong)warp * nGroups;
    device const bf16* b2 = B2 + (ulong)warp * nGroups;
    float acc1 = 0.f, acc2 = 0.f;
    for (uint v = lane; v < nVec; v += 32u) {
        uint word0 = v << 2u;
        uint g = word0 >> 3u;
        float sc1 = bf2f(s1[g]), bi1 = bf2f(b1[g]);
        float sc2 = bf2f(s2[g]), bi2 = bf2f(b2[g]);
        uint b8 = word0 << 3u;
        uint4 wa = *(device const uint4*)(A1 + word0);
        uint4 wb = *(device const uint4*)(A2 + word0);
        #pragma unroll
        for (uint t = 0; t < 4u; ++t) {
            uint a = (t == 0u) ? wa.x : (t == 1u) ? wa.y : (t == 2u) ? wa.z : wa.w;
            uint b = (t == 0u) ? wb.x : (t == 1u) ? wb.y : (t == 2u) ? wb.z : wb.w;
            uint bb = b8 + (t << 3u);
            #pragma unroll
            for (uint k = 0; k < 8u; ++k) {
                float x = bf2f(X[bb + k]);
                acc1 += (float((a >> (4u * k)) & 0xFu) * sc1 + bi1) * x;
                acc2 += (float((b >> (4u * k)) & 0xFu) * sc2 + bi2) * x;
            }
        }
    }
    acc1 = simd_sum(acc1); acc2 = simd_sum(acc2);
    if (lane == 0u) { float g = acc1; Y[warp] = bf16((g / (1.0f + exp(-g))) * acc2); }
}

// ---------------------------------------------------------------------------
// Two independent matvecs with a shared X in one dispatch (concatenated warp
// grid). Removes a dependency bubble between the two.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 8 adjacent output rows per warp: the warp streams a contiguous 8-row block
// (8*inWords words) with 8 register accumulators, so each warp reads a long
// contiguous burst instead of a short row.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Split-K matvec: one warp per (row, input-slice), so small matrices still
// launch many warps (latency hiding). Partials go to partial[row*nSplits + s],
// then split_reduce sums them. wordsPerSplit is a multiple of 8.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Four output rows per warp, 8 lanes per row. Each 8-lane group streams one
// row with 128-byte coalesced transactions and ~(inN/32/8) independent loads
// per lane (deep pipeline). Reduction via 3 xor-shuffles within the 8-lane group.
// ---------------------------------------------------------------------------

// Gather + dequantize one embedding row.
kernel void embed_lookup(
    device const uint*       W   [[buffer(0)]],
    device const bf16* S   [[buffer(1)]],
    device const bf16* B   [[buffer(2)]],
    device       bf16* Y   [[buffer(3)]],
    constant     uint&       rowIndex [[buffer(4)]],
    constant     uint&       inN [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= inN) return;
    const uint inWords = inN / 8u;
    const uint nGroups = inN / 64u;
    const uint row = rowIndex * inWords;
    const uint srow = rowIndex * nGroups;
    uint j = gid >> 3u;
    uint k = gid & 7u;
    uint g = gid >> 6u;
    uint w = W[row + j];
    float sc = bf2f(S[srow + g]);
    float bi = bf2f(B[srow + g]);
    Y[gid] = bf16(float((w >> (4u * k)) & 0xFu) * sc + bi);
}

// ---------------------------------------------------------------------------
// RMSNorm over rows of length n. addOne: (1 + weight) when set.
// ---------------------------------------------------------------------------
kernel void rmsnorm(
    device const bf16* X [[buffer(0)]],
    device const bf16* W [[buffer(1)]],
    device       bf16* Y [[buffer(2)]],
    constant     uint&       n [[buffer(3)]],
    constant     float&      eps [[buffer(4)]],
    constant     uint&       addOne [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    const uint off = tgid * n;
    float local = 0.0f;
    for (uint i = tid; i < n; i += tgsize) { float v = bf2f(X[off + i]); local += v * v; }
    red[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tgsize) {
        float w = bf2f(W[i]);
        if (addOne != 0u) w = 1.0f + w;
        Y[off + i] = bf16(bf2f(X[off + i]) * inv * w);
    }
}

// Gated RMSNorm: Y = rms(X)*W*silu(Z)
kernel void rmsnorm_gated(
    device const bf16* X [[buffer(0)]],
    device const bf16* W [[buffer(1)]],
    device const bf16* Z [[buffer(2)]],
    device       bf16* Y [[buffer(3)]],
    constant     uint&       n [[buffer(4)]],
    constant     float&      eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[128];
    const uint off = tgid * n;
    float local = 0.0f;
    for (uint i = tid; i < n; i += tgsize) { float v = bf2f(X[off + i]); local += v * v; }
    red[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tgsize) {
        float z = bf2f(Z[off + i]);
        float silu = z / (1.0f + exp(-z));
        Y[off + i] = bf16(bf2f(X[off + i]) * inv * bf2f(W[i]) * silu);
    }
}

// ---------------------------------------------------------------------------
// Partial RoPE over per-token tensors. One thread per (head, i), i < rotaryDim/2.
// ---------------------------------------------------------------------------
kernel void apply_rope(
    device bf16* X [[buffer(0)]],
    constant uint& numHeads [[buffer(1)]],
    constant uint& headDim [[buffer(2)]],
    constant uint& rotaryDim [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfDim = rotaryDim / 2u;
    if (gid >= numHeads * halfDim) return;
    uint head = gid / halfDim;
    uint i = gid % halfDim;
    uint base = head * headDim;
    float a = bf2f(X[base + i]);
    float b = bf2f(X[base + i + halfDim]);
    float invFreq = pow(theta, -float(2u * i) / float(rotaryDim));
    float freq = float(pos) * invFreq;
    float c = cos(freq);
    float s = sin(freq);
    X[base + i]            = bf16(a * c - b * s);
    X[base + i + halfDim]  = bf16(b * c + a * s);
}

// Causal depthwise conv step with resident bf16 state [convDim][3].
kernel void conv_step(
    device const bf16* X [[buffer(0)]],
    device bf16* state [[buffer(1)]],
    device const bf16* W [[buffer(2)]],
    device bf16* Y [[buffer(3)]],
    constant uint& convDim [[buffer(4)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= convDim) return;
    float s0 = bf2f(state[c * 3u + 0u]);
    float s1 = bf2f(state[c * 3u + 1u]);
    float s2 = bf2f(state[c * 3u + 2u]);
    float x  = bf2f(X[c]);
    float acc = bf2f(W[c * 4u + 0u]) * s0 + bf2f(W[c * 4u + 1u]) * s1
              + bf2f(W[c * 4u + 2u]) * s2 + bf2f(W[c * 4u + 3u]) * x;
    Y[c] = bf16(acc / (1.0f + exp(-acc)));
    state[c * 3u + 0u] = bf16(s1);
    state[c * 3u + 1u] = bf16(s2);
    state[c * 3u + 2u] = bf16(x);
}

// Per-head L2 norm + scale. One threadgroup per vector of length D.
kernel void l2norm_scale(
    device const bf16* X [[buffer(0)]],
    device bf16* Y [[buffer(1)]],
    constant uint& D [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[128];
    const uint off = tgid * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tgsize) { float v = bf2f(X[off + i]); local += v * v; }
    red[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = scale * rsqrt(red[0] + 1e-6f);
    for (uint i = tid; i < D; i += tgsize) Y[off + i] = bf16(bf2f(X[off + i]) * inv);
}

// g (already exponentiated decay) and beta for the gated delta rule.
kernel void g_beta(
    device const bf16* A [[buffer(0)]],
    device const bf16* Bv [[buffer(1)]],
    device const bf16* A_log [[buffer(2)]],
    device const bf16* dt [[buffer(3)]],
    device bf16* G [[buffer(4)]],
    device bf16* Beta [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= H) return;
    float a = bf2f(A[i]) + bf2f(dt[i]);
    float sp = (a > 20.0f) ? a : log(1.0f + exp(a));
    G[i] = bf16(exp(-exp(bf2f(A_log[i])) * sp));
    Beta[i] = bf16(1.0f / (1.0f + exp(-bf2f(Bv[i]))));
}

// One recurrent step of the gated delta rule. State layout [numVHeads][vDim][kDim].
kernel void gdelta_step(
    device const bf16* QN [[buffer(0)]],
    device const bf16* KN [[buffer(1)]],
    device const bf16* Vt [[buffer(2)]],
    device const bf16* G [[buffer(3)]],
    device const bf16* Beta [[buffer(4)]],
    device bf16* State [[buffer(5)]],
    device bf16* Out [[buffer(6)]],
    constant uint& numVHeads [[buffer(7)]],
    constant uint& numKHeads [[buffer(8)]],
    constant uint& kDim [[buffer(9)]],
    constant uint& vDim [[buffer(10)]],
    uint h [[threadgroup_position_in_grid]],
    uint v [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    uint khead = h / (numVHeads / numKHeads);
    uint kbase = khead * kDim;
    uint vbase = h * vDim;
    float g = bf2f(G[h]);
    float beta = bf2f(Beta[h]);

    threadgroup float ksh[128];
    threadgroup float qsh[128];
    if (v < kDim) {
        ksh[v] = bf2f(KN[kbase + v]);
        qsh[v] = bf2f(QN[kbase + v]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device bf16* S = State + (ulong)h * vDim * kDim + (ulong)v * kDim;
    float kv = 0.0f;
    for (uint k = 0; k < kDim; ++k) kv += bf2f(S[k]) * ksh[k];
    float delta = (bf2f(Vt[vbase + v]) - kv) * beta;
    float out = 0.0f;
    for (uint k = 0; k < kDim; ++k) {
        float s = bf2f(S[k]) * g + ksh[k] * delta;
        S[k] = bf16(s);
        out += s * qsh[k];
    }
    Out[vbase + v] = bf16(out);
}

// Write this token's k/v into the per-head cache at `pos`.
kernel void write_kv(
    device const bf16* K [[buffer(0)]],
    device const bf16* V [[buffer(1)]],
    device bf16* KC [[buffer(2)]],
    device bf16* VC [[buffer(3)]],
    constant uint& numKvHeads [[buffer(4)]],
    constant uint& headDim [[buffer(5)]],
    constant uint& maxT [[buffer(6)]],
    constant uint& pos [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint h = gid / headDim;
    uint d = gid % headDim;
    uint idx = (h * maxT + pos) * headDim + d;
    KC[idx] = K[gid];
    VC[idx] = V[gid];
}

// Single-token decode attention with GQA. One threadgroup per query head.
kernel void attention_decode(
    device const bf16* Q [[buffer(0)]],
    device const bf16* K [[buffer(1)]],
    device const bf16* V [[buffer(2)]],
    device bf16* O [[buffer(3)]],
    constant uint& numHeads [[buffer(4)]],
    constant uint& numKvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& maxT [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    uint group = numHeads / numKvHeads;
    uint kvHead = head / group;
    uint qbase = head * headDim;
    float q = bf2f(Q[qbase + d]);
    device const bf16* Kb = K + (ulong)kvHead * maxT * headDim;
    device const bf16* Vb = V + (ulong)kvHead * maxT * headDim;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint t = 0; t < T; ++t) {
        float partial = q * bf2f(Kb[(ulong)t * headDim + d]);
        red[d] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) {
            if (d < s) red[d] += red[d + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score = red[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float newM = max(m, score);
        float corr = exp(m - newM);
        float p = exp(score - newM);
        acc = acc * corr + p * bf2f(Vb[(ulong)t * headDim + d]);
        l = l * corr + p;
        m = newM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    O[qbase + d] = bf16(acc / l);
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------
kernel void mul_sigmoid_gate(
    device bf16* X [[buffer(0)]],
    device const bf16* Gate [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = bf2f(Gate[gid]);
    X[gid] = bf16(bf2f(X[gid]) / (1.0f + exp(-g)));
}

kernel void silu_mul(
    device const bf16* Gate [[buffer(0)]],
    device const bf16* Up [[buffer(1)]],
    device bf16* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = bf2f(Gate[gid]);
    Y[gid] = bf16((g / (1.0f + exp(-g))) * bf2f(Up[gid]));
}

kernel void copy_range(
    device const bf16* Src [[buffer(0)]],
    device bf16* Dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& srcOffset [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Dst[gid] = Src[srcOffset + gid];
}

kernel void add_inplace(
    device bf16* X [[buffer(0)]],
    device const bf16* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = bf16(bf2f(X[gid]) + bf2f(Y[gid]));
}

kernel void split_q_gate(
    device const bf16* X [[buffer(0)]],
    device bf16* Q [[buffer(1)]],
    device bf16* Gate [[buffer(2)]],
    constant uint& numHeads [[buffer(3)]],
    constant uint& headDim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint per = numHeads * headDim;
    if (gid >= per) return;
    uint head = gid / headDim;
    uint d = gid % headDim;
    uint base = head * headDim * 2u;
    Q[gid] = X[base + d];
    Gate[gid] = X[base + headDim + d];
}

// argmax over bf16 logits, 256 block partial pass.
kernel void argmax_partial(
    device const bf16* X [[buffer(0)]],
    device float* outVal [[buffer(1)]],
    device uint* outIdx [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float rv[256];
    threadgroup uint ri[256];
    uint chunk = (n + 255u) / 256u;
    uint start = tg * chunk;
    uint end = min(start + chunk, n);
    float best = -INFINITY;
    uint bi = start;
    for (uint i = start + tid; i < end; i += tgsize) {
        float v = bf2f(X[i]);
        if (v > best) { best = v; bi = i; }
    }
    rv[tid] = best; ri[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (rv[tid + s] > rv[tid]) { rv[tid] = rv[tid + s]; ri[tid] = ri[tid + s]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { outVal[tg] = rv[0]; outIdx[tg] = ri[0]; }
}

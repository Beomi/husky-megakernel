#include <metal_stdlib>
using namespace metal;

// Chunked-prefill kernels. Activations are [C][dim] where C is the number of
// prompt tokens in the chunk. Weights are read once per output row and reused
// across the chunk's tokens; dequantization is amortized over the token block.

inline ushort f2bf_bits_p(float f) {
    uint x = as_type<uint>(f);
    uint lsb = (x >> 16) & 1u;
    uint r = (x + 0x7FFFu + lsb) >> 16;
    return ushort(r);
}
struct bf16p {
    ushort bits;
    bf16p() : bits(0) {}
    bf16p(float f) : bits(f2bf_bits_p(f)) {}
};
inline float bf2f_p(bf16p x) { return as_type<float>(uint(x.bits) << 16); }

constant uint TB = 8;   // token block per warp

// GEMM: Y[t][o] = sum_i W[o][i] * X[t][i], one warp per output row streaming
// over token blocks. Dequant once per (row, uint4), reused for TB tokens.
kernel void qmatvec_prefill(
    device const uint* W [[buffer(0)]], device const bf16p* S [[buffer(1)]], device const bf16p* Bs [[buffer(2)]],
    device const bf16p* X [[buffer(3)]], device bf16p* Y [[buffer(4)]],
    constant uint& outN [[buffer(5)]], constant uint& inN [[buffer(6)]],
    constant uint& C [[buffer(7)]], constant uint& totalWarps [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp0; o < outN; o += totalWarps) {
        device const uint* Wrow = W + (ulong)o * inWords;
        device const bf16p* Srow = S + (ulong)o * nGroups;
        device const bf16p* Brow = Bs + (ulong)o * nGroups;
        for (uint t0 = 0; t0 < C; t0 += TB) {
            float acc[TB];
            #pragma unroll
            for (uint j = 0; j < TB; ++j) acc[j] = 0.0f;
            for (uint v = lane; v < nVec; v += 32u) {
                uint word0 = v << 2u;
                uint g = word0 >> 3u;
                float sc = bf2f_p(Srow[g]);
                float bi = bf2f_p(Brow[g]);
                uint b8 = word0 << 3u;
                uint4 w = *(device const uint4*)(Wrow + word0);
                float wf[32];
                #pragma unroll
                for (uint t = 0; t < 4u; ++t) {
                    uint ww = (t == 0u) ? w.x : (t == 1u) ? w.y : (t == 2u) ? w.z : w.w;
                    #pragma unroll
                    for (uint k = 0; k < 8u; ++k)
                        wf[(t << 3u) + k] = float((ww >> (4u * k)) & 0xFu) * sc + bi;
                }
                #pragma unroll
                for (uint j = 0; j < TB; ++j) {
                    uint tt = t0 + j;
                    if (tt < C) {
                        device const bf16p* Xr = X + (ulong)tt * inN + b8;
                        #pragma unroll
                        for (uint k = 0; k < 32u; ++k) acc[j] += wf[k] * bf2f_p(Xr[k]);
                    }
                }
            }
            #pragma unroll
            for (uint j = 0; j < TB; ++j) {
                float a = simd_sum(acc[j]);
                if (lane == 0u && (t0 + j) < C) Y[(ulong)(t0 + j) * outN + o] = bf16p(a);
            }
        }
    }
}

// gate/up + SwiGLU chunk GEMM -> Y[t][o] = silu(gate)*up.
kernel void qmatvec_pair_silu_prefill(
    device const uint* W1 [[buffer(0)]], device const bf16p* S1 [[buffer(1)]], device const bf16p* B1 [[buffer(2)]],
    device const uint* W2 [[buffer(3)]], device const bf16p* S2 [[buffer(4)]], device const bf16p* B2 [[buffer(5)]],
    device const bf16p* X [[buffer(6)]], device bf16p* Y [[buffer(7)]],
    constant uint& outN [[buffer(8)]], constant uint& inN [[buffer(9)]], constant uint& C [[buffer(10)]],
    constant uint& totalWarps [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp0; o < outN; o += totalWarps) {
        device const uint* A1 = W1 + (ulong)o * inWords;
        device const uint* A2 = W2 + (ulong)o * inWords;
        device const bf16p* s1 = S1 + (ulong)o * nGroups;
        device const bf16p* s2 = S2 + (ulong)o * nGroups;
        device const bf16p* b1 = B1 + (ulong)o * nGroups;
        device const bf16p* b2 = B2 + (ulong)o * nGroups;
        for (uint t0 = 0; t0 < C; t0 += TB) {
            float a1[TB], a2[TB];
            #pragma unroll
            for (uint j = 0; j < TB; ++j) { a1[j] = 0.f; a2[j] = 0.f; }
            for (uint v = lane; v < nVec; v += 32u) {
                uint word0 = v << 2u;
                uint g = word0 >> 3u;
                float sc1 = bf2f_p(s1[g]), bi1 = bf2f_p(b1[g]);
                float sc2 = bf2f_p(s2[g]), bi2 = bf2f_p(b2[g]);
                uint b8 = word0 << 3u;
                uint4 wa = *(device const uint4*)(A1 + word0);
                uint4 wb = *(device const uint4*)(A2 + word0);
                float w1f[32], w2f[32];
                #pragma unroll
                for (uint t = 0; t < 4u; ++t) {
                    uint ua = (t == 0u) ? wa.x : (t == 1u) ? wa.y : (t == 2u) ? wa.z : wa.w;
                    uint ub = (t == 0u) ? wb.x : (t == 1u) ? wb.y : (t == 2u) ? wb.z : wb.w;
                    #pragma unroll
                    for (uint k = 0; k < 8u; ++k) {
                        w1f[(t << 3u) + k] = float((ua >> (4u * k)) & 0xFu) * sc1 + bi1;
                        w2f[(t << 3u) + k] = float((ub >> (4u * k)) & 0xFu) * sc2 + bi2;
                    }
                }
                #pragma unroll
                for (uint j = 0; j < TB; ++j) {
                    uint t = t0 + j;
                    if (t < C) {
                        device const bf16p* Xr = X + (ulong)t * inN + b8;
                        #pragma unroll
                        for (uint k = 0; k < 32u; ++k) { float x = bf2f_p(Xr[k]); a1[j] += w1f[k] * x; a2[j] += w2f[k] * x; }
                    }
                }
            }
            #pragma unroll
            for (uint j = 0; j < TB; ++j) {
                float g1 = simd_sum(a1[j]);
                float g2 = simd_sum(a2[j]);
                if (lane == 0u && (t0 + j) < C) Y[(ulong)(t0 + j) * outN + o] = bf16p((g1 / (1.0f + exp(-g1))) * g2);
            }
        }
    }
}

// GEMM + residual: Y[t][o] = matvec + R[t][o].
kernel void qmatvec_add_prefill(
    device const uint* W [[buffer(0)]], device const bf16p* S [[buffer(1)]], device const bf16p* Bs [[buffer(2)]],
    device const bf16p* X [[buffer(3)]], device const bf16p* R [[buffer(4)]], device bf16p* Y [[buffer(5)]],
    constant uint& outN [[buffer(6)]], constant uint& inN [[buffer(7)]], constant uint& C [[buffer(8)]],
    constant uint& totalWarps [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp0; o < outN; o += totalWarps) {
        device const uint* Wrow = W + (ulong)o * inWords;
        device const bf16p* Srow = S + (ulong)o * nGroups;
        device const bf16p* Brow = Bs + (ulong)o * nGroups;
        for (uint t0 = 0; t0 < C; t0 += TB) {
            float acc[TB];
            #pragma unroll
            for (uint j = 0; j < TB; ++j) acc[j] = 0.0f;
            for (uint v = lane; v < nVec; v += 32u) {
                uint word0 = v << 2u;
                uint g = word0 >> 3u;
                float sc = bf2f_p(Srow[g]);
                float bi = bf2f_p(Brow[g]);
                uint b8 = word0 << 3u;
                uint4 w = *(device const uint4*)(Wrow + word0);
                float wf[32];
                #pragma unroll
                for (uint t = 0; t < 4u; ++t) {
                    uint ww = (t == 0u) ? w.x : (t == 1u) ? w.y : (t == 2u) ? w.z : w.w;
                    #pragma unroll
                    for (uint k = 0; k < 8u; ++k) wf[(t << 3u) + k] = float((ww >> (4u * k)) & 0xFu) * sc + bi;
                }
                #pragma unroll
                for (uint j = 0; j < TB; ++j) {
                    uint t = t0 + j;
                    if (t < C) {
                        device const bf16p* Xr = X + (ulong)t * inN + b8;
                        #pragma unroll
                        for (uint k = 0; k < 32u; ++k) acc[j] += wf[k] * bf2f_p(Xr[k]);
                    }
                }
            }
            #pragma unroll
            for (uint j = 0; j < TB; ++j) {
                float a = simd_sum(acc[j]);
                uint t = t0 + j;
                if (lane == 0u && t < C) Y[(ulong)t * outN + o] = bf16p(a + bf2f_p(R[(ulong)t * outN + o]));
            }
        }
    }
}

// Causal depthwise conv over the chunk, one thread per channel.
kernel void conv_chunk(
    device const bf16p* X [[buffer(0)]], device bf16p* state [[buffer(1)]], device const bf16p* W [[buffer(2)]],
    device bf16p* Y [[buffer(3)]], constant uint& convDim [[buffer(4)]], constant uint& C [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= convDim) return;
    float s0 = bf2f_p(state[c * 3u + 0u]), s1 = bf2f_p(state[c * 3u + 1u]), s2 = bf2f_p(state[c * 3u + 2u]);
    float w0 = bf2f_p(W[c * 4u + 0u]), w1 = bf2f_p(W[c * 4u + 1u]);
    float w2 = bf2f_p(W[c * 4u + 2u]), w3 = bf2f_p(W[c * 4u + 3u]);
    for (uint t = 0; t < C; ++t) {
        float x = bf2f_p(X[(ulong)t * convDim + c]);
        float acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        Y[(ulong)t * convDim + c] = bf16p(acc / (1.0f + exp(-acc)));
        s0 = s1; s1 = s2; s2 = x;
    }
    state[c * 3u + 0u] = bf16p(s0); state[c * 3u + 1u] = bf16p(s1); state[c * 3u + 2u] = bf16p(s2);
}

// Gated-delta recurrence over the chunk, one threadgroup per value head.
kernel void gdelta_chunk(
    device const bf16p* QN [[buffer(0)]], device const bf16p* KN [[buffer(1)]], device const bf16p* Vt [[buffer(2)]],
    device const bf16p* G [[buffer(3)]], device const bf16p* Beta [[buffer(4)]], device bf16p* State [[buffer(5)]],
    device bf16p* Out [[buffer(6)]], constant uint& numVHeads [[buffer(7)]], constant uint& numKHeads [[buffer(8)]],
    constant uint& kDim [[buffer(9)]], constant uint& vDim [[buffer(10)]], constant uint& C [[buffer(11)]],
    uint h [[threadgroup_position_in_grid]], uint v [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]])
{
    if (h >= numVHeads) return;
    uint khead = h / (numVHeads / numKHeads);
    uint keyDim = kDim * numKHeads, valueDim = vDim * numVHeads;
    uint kbase = khead * kDim;
    device bf16p* S = State + (ulong)h * vDim * kDim + (ulong)v * kDim;
    threadgroup float ksh[128];
    threadgroup float qsh[128];
    for (uint t = 0; t < C; ++t) {
        float g = bf2f_p(G[(ulong)t * numVHeads + h]);
        float beta = bf2f_p(Beta[(ulong)t * numVHeads + h]);
        if (v < kDim) {
            ksh[v] = bf2f_p(KN[(ulong)t * keyDim + kbase + v]);
            qsh[v] = bf2f_p(QN[(ulong)t * keyDim + kbase + v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (v < vDim) {
            float kv = 0.f;
            for (uint k = 0; k < kDim; ++k) kv += bf2f_p(S[k]) * ksh[k];
            float delta = (bf2f_p(Vt[(ulong)t * valueDim + h * vDim + v]) - kv) * beta;
            float out = 0.f;
            for (uint k = 0; k < kDim; ++k) {
                float s = bf2f_p(S[k]) * g + ksh[k] * delta;
                S[k] = bf16p(s);
                out += s * qsh[k];
            }
            Out[(ulong)t * valueDim + h * vDim + v] = bf16p(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Keep each value row in registers across the chunk. Preserve the reference
// summation order and per-step bf16 state rounding (important for recurrence).
kernel void gdelta_chunk_register(
    device const bf16p* QN [[buffer(0)]], device const bf16p* KN [[buffer(1)]], device const bf16p* Vt [[buffer(2)]],
    device const bf16p* G [[buffer(3)]], device const bf16p* Beta [[buffer(4)]], device bf16p* State [[buffer(5)]],
    device bf16p* Out [[buffer(6)]], constant uint& numVHeads [[buffer(7)]], constant uint& numKHeads [[buffer(8)]],
    constant uint& kDim [[buffer(9)]], constant uint& vDim [[buffer(10)]], constant uint& C [[buffer(11)]],
    uint h [[threadgroup_position_in_grid]], uint v [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]])
{
    #pragma clang fp reassociate(off)

    if (h >= numVHeads) return;
    uint khead = h / (numVHeads / numKHeads);
    uint keyDim = kDim * numKHeads, valueDim = vDim * numVHeads;
    uint kbase = khead * kDim;
    device bf16p* S = State + (ulong)h * vDim * kDim + (ulong)v * kDim;
    threadgroup float ksh[128];
    threadgroup float qsh[128];
    float state[128];
    #pragma unroll
    for (uint k = 0; k < 128; ++k) state[k] = v < vDim ? bf2f_p(S[k]) : 0.f;
    for (uint t = 0; t < C; ++t) {
        float g = bf2f_p(G[(ulong)t * numVHeads + h]);
        float beta = bf2f_p(Beta[(ulong)t * numVHeads + h]);
        if (v < kDim) {
            ksh[v] = bf2f_p(KN[(ulong)t * keyDim + kbase + v]);
            qsh[v] = bf2f_p(QN[(ulong)t * keyDim + kbase + v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (v < vDim) {
            float kv = 0.f;
            #pragma unroll
            for (uint k = 0; k < 128; ++k) kv += state[k] * ksh[k];
            float delta = (bf2f_p(Vt[(ulong)t * valueDim + h * vDim + v]) - kv) * beta;
            float out = 0.f;
            #pragma unroll
            for (uint k = 0; k < 128; ++k) {
                float s = state[k] * g + ksh[k] * delta;
                state[k] = bf2f_p(bf16p(s));
                out += s * qsh[k];
            }
            Out[(ulong)t * valueDim + h * vDim + v] = bf16p(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (v < vDim) {
        #pragma unroll
        for (uint k = 0; k < 128; ++k) S[k] = bf16p(state[k]);
    }
}

// Per-token RoPE over the chunk. grid = C * numHeads * (rotaryDim/2).
kernel void apply_rope_chunk(
    device bf16p* X [[buffer(0)]], constant uint& numHeads [[buffer(1)]], constant uint& headDim [[buffer(2)]],
    constant uint& rotaryDim [[buffer(3)]], constant uint& basePos [[buffer(4)]], constant float& theta [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfDim = rotaryDim / 2u;
    uint per = numHeads * halfDim;
    uint t = gid / per, r = gid % per;
    uint head = r / halfDim, j = r % halfDim;
    uint base = (ulong)t * numHeads * headDim + head * headDim;
    float va = bf2f_p(X[base + j]);
    float vb = bf2f_p(X[base + j + halfDim]);
    float invFreq = pow(theta, -float(2u * j) / float(rotaryDim));
    float freq = float(basePos + t) * invFreq;
    float c = cos(freq), s = sin(freq);
    X[base + j] = bf16p(va * c - vb * s);
    X[base + j + halfDim] = bf16p(vb * c + va * s);
}

// Write C tokens of K/V into the cache at positions basePos..basePos+C-1.
kernel void write_kv_chunk(
    device const bf16p* K [[buffer(0)]], device const bf16p* V [[buffer(1)]], device bf16p* KC [[buffer(2)]],
    device bf16p* VC [[buffer(3)]], constant uint& numKvHeads [[buffer(4)]], constant uint& headDim [[buffer(5)]],
    constant uint& maxT [[buffer(6)]], constant uint& basePos [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    uint per = numKvHeads * headDim;
    uint t = gid / per, r = gid % per;
    uint h = r / headDim, d = r % headDim;
    uint idx = (ulong)h * maxT * headDim + (ulong)(basePos + t) * headDim + d;
    KC[idx] = K[(ulong)t * per + r];
    VC[idx] = V[(ulong)t * per + r];
}

// Causal attention over the chunk. One threadgroup per (token, head).
kernel void attention_chunk(
    device const bf16p* Q [[buffer(0)]], device const bf16p* K [[buffer(1)]], device const bf16p* V [[buffer(2)]],
    device bf16p* O [[buffer(3)]], constant uint& numHeads [[buffer(4)]], constant uint& numKvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]], constant uint& basePos [[buffer(7)]], constant float& scale [[buffer(8)]],
    constant uint& maxT [[buffer(9)]], uint row [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    uint t = row / numHeads, head = row % numHeads;
    uint group = numHeads / numKvHeads;
    uint kvHead = head / group;
    uint oProjIn = numHeads * headDim;
    uint qbase = (ulong)t * oProjIn + head * headDim;
    uint T = basePos + t + 1u;   // causal: past + current chunk up to t
    float q = (d < headDim) ? bf2f_p(Q[qbase + d]) : 0.0f;
    device const bf16p* Kb = K + (ulong)kvHead * maxT * headDim;
    device const bf16p* Vb = V + (ulong)kvHead * maxT * headDim;
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (uint kt = 0; kt < T; ++kt) {
        float partial = (d < headDim) ? q * bf2f_p(Kb[(ulong)kt * headDim + d]) : 0.0f;
        red[d] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float score = red[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float newM = max(m, score);
        float corr = exp(m - newM);
        float p = exp(score - newM);
        if (d < headDim) acc = acc * corr + p * bf2f_p(Vb[(ulong)kt * headDim + d]);
        l = l * corr + p;
        m = newM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (d < headDim) O[qbase + d] = bf16p(acc / l);
}

// Preserve the reference dot-product tree, but finish its reduction in one
// SIMD group instead of eight threadgroup-wide barrier rounds.
kernel void attention_chunk_simd(
    device const bf16p* Q [[buffer(0)]], device const bf16p* K [[buffer(1)]], device const bf16p* V [[buffer(2)]],
    device bf16p* O [[buffer(3)]], constant uint& numHeads [[buffer(4)]], constant uint& numKvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]], constant uint& basePos [[buffer(7)]], constant float& scale [[buffer(8)]],
    constant uint& maxT [[buffer(9)]], uint row [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    uint t = row / numHeads, head = row % numHeads;
    uint group = numHeads / numKvHeads;
    uint kvHead = head / group;
    uint oProjIn = numHeads * headDim;
    uint qbase = (ulong)t * oProjIn + head * headDim;
    uint T = basePos + t + 1u;   // causal: past + current chunk up to t
    float q = (d < headDim) ? bf2f_p(Q[qbase + d]) : 0.0f;
    device const bf16p* Kb = K + (ulong)kvHead * maxT * headDim;
    device const bf16p* Vb = V + (ulong)kvHead * maxT * headDim;
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (uint kt = 0; kt < T; ++kt) {
        float partial = (d < headDim) ? q * bf2f_p(Kb[(ulong)kt * headDim + d]) : 0.0f;
        red[d] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < 32u) {
            #pragma clang fp reassociate(off)
            float sum = ((red[d] + red[d+128]) + (red[d+64] + red[d+192]))
                      + ((red[d+32] + red[d+160]) + (red[d+96] + red[d+224]));
            #pragma unroll
            for (uint shift=16; shift>0; shift>>=1) sum += simd_shuffle_down(sum, shift);
            if (d == 0u) red[0] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = red[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float newM = max(m, score);
        float corr = exp(m - newM);
        float p = exp(score - newM);
        if (d < headDim) acc = acc * corr + p * bf2f_p(Vb[(ulong)kt * headDim + d]);
        l = l * corr + p;
        m = newM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (d < headDim) O[qbase + d] = bf16p(acc / l);
}

// ---------------------------------------------------------------------------
// Dequantize a 4-bit tensor and transpose:  OUT[i*outN + o] = dequant(W[o][i])
// Output is fp16 (matrix-unit operand). One-time at load.
// ---------------------------------------------------------------------------
kernel void dequant_transpose(
    device const uint* W [[buffer(0)]], device const bf16p* S [[buffer(1)]], device const bf16p* Bs [[buffer(2)]],
    device half* OUT [[buffer(3)]], constant uint& outN [[buffer(4)]], constant uint& inN [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outN * inN) return;
    uint i = gid / outN, o = gid % outN;
    uint inWords = inN >> 3u, nGroups = inN >> 6u;
    uint w = W[(ulong)o * inWords + (i >> 3u)];
    float sc = bf2f_p(S[(ulong)o * nGroups + (i >> 6u)]);
    float bi = bf2f_p(Bs[(ulong)o * nGroups + (i >> 6u)]);
    OUT[gid] = half(float((w >> (4u * (i & 7u))) & 0xFu) * sc + bi);
}

// ---------------------------------------------------------------------------
constant uint BM = 64, BN = 32, BK = 32;

// Dequantize 4-bit weights to fp16 in a TILED layout:
//   OUT[ nTile ][ kTile ][ r ][ c ]  = dequant(W[ nTile*BN + c ][ kTile*BK + r ])
// so each (nTile, kTile) [BK][BN] block is contiguous for coalesced GEMM loads.
// ---------------------------------------------------------------------------
kernel void dequant_pack(
    device const uint* W [[buffer(0)]], device const bf16p* S [[buffer(1)]], device const bf16p* Bias [[buffer(2)]],
    device half* OUT [[buffer(3)]], constant uint& outN [[buffer(4)]], constant uint& inN [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outN * inN) return;
    uint c = gid % BN, t1 = gid / BN;
    uint r = t1 % BK, t2 = t1 / BK;
    uint kTiles = inN / BK;
    uint kTile = t2 % kTiles, nTile = t2 / kTiles;
    uint n = nTile * BN + c, k = kTile * BK + r;
    uint inWords = inN >> 3u, nGroups = inN >> 6u;
    uint w = W[(ulong)n * inWords + (k >> 3u)];
    float sc = bf2f_p(S[(ulong)n * nGroups + (k >> 6u)]);
    float bi = bf2f_p(Bias[(ulong)n * nGroups + (k >> 6u)]);
    OUT[gid] = half(float((w >> (4u * (k & 7u))) & 0xFu) * sc + bi);
}

// ---------------------------------------------------------------------------
// fp16 GEMM on the matrix units:  Y[M][N] = X[M][K] * B[K][N]
// X / Y are bf16 (activations), B is fp16 in the packed layout produced by dequant_pack.
// Threadgroup tile BM x BN, K streamed in BK chunks through threadgroup memory.
// ---------------------------------------------------------------------------

inline uint4 prefetchA(device const bf16p* X, uint m0, uint k0, uint tid, uint M, uint K) {
    uint base = tid * 8u, i = base / BK, j = base % BK;
    uint gm = m0 + i, gk = k0 + j;
    if (gm < M && gk < K) return *(device const uint4*)(X + (ulong)gm * K + gk);
    return uint4(0u);
}
inline uint4 prefetchBpack(device const half* B, uint base, uint tid) {
    if (tid < (BK * BN) / 8u) return *(device const uint4*)(B + base + tid * 8u);
    return uint4(0u);
}
inline void commitA(threadgroup half* As, uint tid, uint4 v) {
    ushort4 lo = as_type<ushort4>(v.xy), hi = as_type<ushort4>(v.zw);
    *(threadgroup half4*)(As + tid * 8u) = half4(as_type<float4>(uint4(lo) << 16));
    *(threadgroup half4*)(As + tid * 8u + 4u) = half4(as_type<float4>(uint4(hi) << 16));
}
inline void commitB(threadgroup half* Bs, uint tid, uint4 v) {
    *(threadgroup uint4*)(Bs + tid * 8u) = v;
}

kernel void gemm_mma(
    device const bf16p* X [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device bf16p*       Y [[buffer(2)]],
    constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]])
{
    threadgroup half As[2][BM * BK];
    threadgroup half Bs[2][BK * BN];
    threadgroup float Cs[BM * BN];
    uint tid = tid2.x;
    uint m0 = tgid.x * BM, n0 = tgid.y * BN;
    uint sg = tid >> 5u;
    bool bLoader = (tid < (BK * BN) / 8u);

    simdgroup_matrix<float, 8, 8> c[4];
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) c[j] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint kTiles = K / BK;
    uint baseN = (n0 / BN) * kTiles;   // nTile * kTiles
    uint4 regA = prefetchA(X, m0, 0u, tid, M, K);
    uint4 regB = bLoader ? prefetchBpack(B, (ulong)baseN * (BK * BN), tid) : uint4(0u);
    uint cur = 0u;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        commitA(As[cur], tid, regA);
        if (bLoader) commitB(Bs[cur], tid, regB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < K) {
            regA = prefetchA(X, m0, k0 + BK, tid, M, K);
            if (bLoader) regB = prefetchBpack(B, (ulong)(baseN + (k0 + BK) / BK) * (BK * BN), tid);
        }
        for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a, b[4];
            simdgroup_load(a, As[cur] + (sg * 8u) * BK + kk, BK);
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs[cur] + kk * BN + j * 8u, BN);
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) simdgroup_multiply_accumulate(c[j], a, b[j], c[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur ^= 1u;
    }

    #pragma unroll
    for (uint j = 0; j < 4u; ++j)
        simdgroup_store(c[j], Cs + (sg * 8u) * BN + j * 8u, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < BM * BN; idx += 256u) {
        uint i = idx / BN, j = idx % BN;
        uint gm = m0 + i, gn = n0 + j;
        if (gm < M && gn < N) Y[(ulong)gm * N + gn] = bf16p(Cs[idx]);
    }
}

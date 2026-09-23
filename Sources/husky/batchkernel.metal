#include <metal_stdlib>
using namespace metal;

// Batched (~B independent sequences) kernels. Activations are laid out
// [B][dim] (batch-major, contiguous). Weights are shared and read once per
// output row for all B batch elements — the point of batching.

inline ushort f2bf_bits_b(float f) {
    uint x = as_type<uint>(f);
    uint lsb = (x >> 16) & 1u;
    uint r = (x + 0x7FFFu + lsb) >> 16;
    return ushort(r);
}
struct bf16b {
    ushort bits;
    bf16b() : bits(0) {}
    bf16b(float f) : bits(f2bf_bits_b(f)) {}
};
inline float bf2f_b(bf16b x) { return as_type<float>(uint(x.bits) << 16); }

constant uint MAXB = 8;

// One warp per output row; reads the weight row once and applies it to all B
// activation rows. X: [B][inN], Y: [B][outN].
kernel void qmatvec_b(
    device const uint* W [[buffer(0)]], device const bf16b* S [[buffer(1)]], device const bf16b* Bs [[buffer(2)]],
    device const bf16b* X [[buffer(3)]], device bf16b* Y [[buffer(4)]],
    constant uint& outN [[buffer(5)]], constant uint& inN [[buffer(6)]],
    constant uint& nb [[buffer(7)]], constant uint& totalWarps [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp0; o < outN; o += totalWarps) {
        device const uint* Wrow = W + (ulong)o * inWords;
        device const bf16b* Srow = S + (ulong)o * nGroups;
        device const bf16b* Brow = Bs + (ulong)o * nGroups;
        float acc[MAXB];
        for (uint b = 0; b < MAXB; ++b) acc[b] = 0.0f;
        for (uint v = lane; v < nVec; v += 32u) {
            uint word0 = v << 2u;
            uint g = word0 >> 3u;
            float sc = bf2f_b(Srow[g]);
            float bi = bf2f_b(Brow[g]);
            uint b8 = word0 << 3u;
            uint4 w = *(device const uint4*)(Wrow + word0);
            #pragma unroll
            for (uint t = 0; t < 4u; ++t) {
                uint ww = (t == 0u) ? w.x : (t == 1u) ? w.y : (t == 2u) ? w.z : w.w;
                uint bb = b8 + (t << 3u);
                for (uint b = 0; b < nb; ++b) {
                    device const bf16b* Xb = X + (ulong)b * inN;
                    #pragma unroll
                    for (uint k = 0; k < 8u; ++k) {
                        float q = float((ww >> (4u * k)) & 0xFu);
                        acc[b] += (q * sc + bi) * bf2f_b(Xb[bb + k]);
                    }
                }
            }
        }
        for (uint b = 0; b < nb; ++b) {
            float a = simd_sum(acc[b]);
            if (lane == 0u) Y[(ulong)b * outN + o] = bf16b(a);
        }
    }
}

// gate+up pair with SwiGLU, batched. X [B][inN], Y [B][outN].
kernel void qmatvec_pair_silu_b(
    device const uint* W1 [[buffer(0)]], device const bf16b* S1 [[buffer(1)]], device const bf16b* B1 [[buffer(2)]],
    device const uint* W2 [[buffer(3)]], device const bf16b* S2 [[buffer(4)]], device const bf16b* B2 [[buffer(5)]],
    device const bf16b* X [[buffer(6)]], device bf16b* Y [[buffer(7)]],
    constant uint& outN [[buffer(8)]], constant uint& inN [[buffer(9)]], constant uint& nb [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint o = tg * (tgSize >> 5u) + (tid >> 5u);
    if (o >= outN) return;
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    device const uint* A1 = W1 + (ulong)o * inWords;
    device const uint* A2 = W2 + (ulong)o * inWords;
    device const bf16b* s1 = S1 + (ulong)o * nGroups;
    device const bf16b* s2 = S2 + (ulong)o * nGroups;
    device const bf16b* b1 = B1 + (ulong)o * nGroups;
    device const bf16b* b2 = B2 + (ulong)o * nGroups;
    float a1[MAXB]; float a2[MAXB];
    for (uint b = 0; b < MAXB; ++b) { a1[b] = 0.f; a2[b] = 0.f; }
    for (uint v = lane; v < nVec; v += 32u) {
        uint word0 = v << 2u;
        uint g = word0 >> 3u;
        float sc1 = bf2f_b(s1[g]), bi1 = bf2f_b(b1[g]);
        float sc2 = bf2f_b(s2[g]), bi2 = bf2f_b(b2[g]);
        uint b8 = word0 << 3u;
        uint4 wa = *(device const uint4*)(A1 + word0);
        uint4 wb = *(device const uint4*)(A2 + word0);
        #pragma unroll
        for (uint t = 0; t < 4u; ++t) {
            uint ua = (t == 0u) ? wa.x : (t == 1u) ? wa.y : (t == 2u) ? wa.z : wa.w;
            uint ub = (t == 0u) ? wb.x : (t == 1u) ? wb.y : (t == 2u) ? wb.z : wb.w;
            uint bb = b8 + (t << 3u);
            for (uint b = 0; b < nb; ++b) {
                device const bf16b* Xb = X + (ulong)b * inN;
                #pragma unroll
                for (uint k = 0; k < 8u; ++k) {
                    float x = bf2f_b(Xb[bb + k]);
                    a1[b] += (float((ua >> (4u * k)) & 0xFu) * sc1 + bi1) * x;
                    a2[b] += (float((ub >> (4u * k)) & 0xFu) * sc2 + bi2) * x;
                }
            }
        }
    }
    for (uint b = 0; b < nb; ++b) {
        float g1 = simd_sum(a1[b]);
        float g2 = simd_sum(a2[b]);
        if (lane == 0u) Y[(ulong)b * outN + o] = bf16b((g1 / (1.0f + exp(-g1))) * g2);
    }
}

// matvec + residual, batched. Y[b] = matvec + R[b].
kernel void qmatvec_add_b(
    device const uint* W [[buffer(0)]], device const bf16b* S [[buffer(1)]], device const bf16b* Bs [[buffer(2)]],
    device const bf16b* X [[buffer(3)]], device const bf16b* R [[buffer(4)]], device bf16b* Y [[buffer(5)]],
    constant uint& outN [[buffer(6)]], constant uint& inN [[buffer(7)]],
    constant uint& nb [[buffer(8)]], constant uint& totalWarps [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    uint lane = tid & 31u;
    uint warp0 = tg * (tgSize >> 5u) + (tid >> 5u);
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp0; o < outN; o += totalWarps) {
        device const uint* Wrow = W + (ulong)o * inWords;
        device const bf16b* Srow = S + (ulong)o * nGroups;
        device const bf16b* Brow = Bs + (ulong)o * nGroups;
        float acc[MAXB];
        for (uint b = 0; b < MAXB; ++b) acc[b] = 0.0f;
        for (uint v = lane; v < nVec; v += 32u) {
            uint word0 = v << 2u;
            uint g = word0 >> 3u;
            float sc = bf2f_b(Srow[g]);
            float bi = bf2f_b(Brow[g]);
            uint b8 = word0 << 3u;
            uint4 w = *(device const uint4*)(Wrow + word0);
            #pragma unroll
            for (uint t = 0; t < 4u; ++t) {
                uint ww = (t == 0u) ? w.x : (t == 1u) ? w.y : (t == 2u) ? w.z : w.w;
                uint bb = b8 + (t << 3u);
                for (uint b = 0; b < nb; ++b) {
                    device const bf16b* Xb = X + (ulong)b * inN;
                    #pragma unroll
                    for (uint k = 0; k < 8u; ++k) {
                        float q = float((ww >> (4u * k)) & 0xFu);
                        acc[b] += (q * sc + bi) * bf2f_b(Xb[bb + k]);
                    }
                }
            }
        }
        for (uint b = 0; b < nb; ++b) {
            float a = simd_sum(acc[b]);
            if (lane == 0u) Y[(ulong)b * outN + o] = bf16b(a + bf2f_b(R[(ulong)b * outN + o]));
        }
    }
}

// embed lookup for B tokens. tokenBuf: uint[B]. Y: [B][hidden].
kernel void embed_lookup_b(
    device const uint* W [[buffer(0)]], device const bf16b* S [[buffer(1)]], device const bf16b* Bs [[buffer(2)]],
    device const uint* tokenBuf [[buffer(3)]], device bf16b* Y [[buffer(4)]],
    constant uint& inN [[buffer(5)]], constant uint& nb [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid / inN, i = gid % inN;
    if (b >= nb) return;
    uint rowIndex = tokenBuf[b];
    uint inWords = inN >> 3u, nGroups = inN >> 6u;
    uint row = rowIndex * inWords, srow = rowIndex * nGroups;
    uint j = i >> 3u, k = i & 7u, g = i >> 6u;
    uint w = W[row + j];
    Y[(ulong)b * inN + i] = bf16b(float((w >> (4u * k)) & 0xFu) * bf2f_b(S[srow + g]) + bf2f_b(Bs[srow + g]));
}

// conv step, batched. grid = B * convDim.
kernel void conv_step_b(
    device const bf16b* X [[buffer(0)]], device bf16b* state [[buffer(1)]], device const bf16b* W [[buffer(2)]],
    device bf16b* Y [[buffer(3)]], constant uint& convDim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid / convDim, c = gid % convDim;
    device bf16b* St = state + (ulong)b * convDim * 3u + c * 3u;
    device const bf16b* Xb = X + (ulong)b * convDim;
    device bf16b* Yb = Y + (ulong)b * convDim;
    float s0 = bf2f_b(St[0]), s1 = bf2f_b(St[1]), s2 = bf2f_b(St[2]);
    float x = bf2f_b(Xb[c]);
    float acc = bf2f_b(W[c * 4u + 0u]) * s0 + bf2f_b(W[c * 4u + 1u]) * s1
              + bf2f_b(W[c * 4u + 2u]) * s2 + bf2f_b(W[c * 4u + 3u]) * x;
    Yb[c] = bf16b(acc / (1.0f + exp(-acc)));
    St[0] = bf16b(s1); St[1] = bf16b(s2); St[2] = bf16b(x);
}

// g/beta, batched. grid = B * vHeads.
kernel void g_beta_b(
    device const bf16b* A [[buffer(0)]], device const bf16b* Bv [[buffer(1)]], device const bf16b* Alog [[buffer(2)]],
    device const bf16b* Dt [[buffer(3)]], device bf16b* G [[buffer(4)]], device bf16b* Beta [[buffer(5)]],
    constant uint& H [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    uint b = gid / H, h = gid % H;
    float x = bf2f_b(A[(ulong)b * H + h]) + bf2f_b(Dt[h]);
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    G[(ulong)b * H + h] = bf16b(exp(-exp(bf2f_b(Alog[h])) * sp));
    Beta[(ulong)b * H + h] = bf16b(1.0f / (1.0f + exp(-bf2f_b(Bv[(ulong)b * H + h]))));
}

// gated delta step, batched. rows = B * numVHeads.
kernel void gdelta_step_b(
    device const bf16b* QN [[buffer(0)]], device const bf16b* KN [[buffer(1)]], device const bf16b* Vt [[buffer(2)]],
    device const bf16b* G [[buffer(3)]], device const bf16b* Beta [[buffer(4)]], device bf16b* State [[buffer(5)]],
    device bf16b* Out [[buffer(6)]], constant uint& numVHeads [[buffer(7)]], constant uint& numKHeads [[buffer(8)]],
    constant uint& kDim [[buffer(9)]], constant uint& vDim [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]], uint v [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]])
{
    uint b = row / numVHeads, h = row % numVHeads;
    uint khead = h / (numVHeads / numKHeads);
    uint keyDim = kDim * numKHeads, valueDim = vDim * numVHeads;
    uint kbase = (ulong)b * keyDim + khead * kDim;
    uint vbase = (ulong)b * valueDim + h * vDim;
    float g = bf2f_b(G[(ulong)b * numVHeads + h]);
    float beta = bf2f_b(Beta[(ulong)b * numVHeads + h]);
    threadgroup float ksh[128];
    threadgroup float qsh[128];
    if (v < kDim) { ksh[v] = bf2f_b(KN[kbase + v]); qsh[v] = bf2f_b(QN[kbase + v]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (v < vDim) {
        device bf16b* S = State + ((ulong)b * numVHeads + h) * vDim * kDim + (ulong)v * kDim;
        float kv = 0.f;
        for (uint k = 0; k < kDim; ++k) kv += bf2f_b(S[k]) * ksh[k];
        float delta = (bf2f_b(Vt[vbase + v]) - kv) * beta;
        float out = 0.f;
        for (uint k = 0; k < kDim; ++k) {
            float s = bf2f_b(S[k]) * g + ksh[k] * delta;
            S[k] = bf16b(s);
            out += s * qsh[k];
        }
        Out[vbase + v] = bf16b(out);
    }
}

// KV write, batched. grid = B * numKvHeads * headDim.
kernel void write_kv_b(
    device const bf16b* K [[buffer(0)]], device const bf16b* V [[buffer(1)]], device bf16b* KC [[buffer(2)]],
    device bf16b* VC [[buffer(3)]], constant uint& numKvHeads [[buffer(4)]], constant uint& headDim [[buffer(5)]],
    constant uint& maxT [[buffer(6)]], constant uint& pos [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    uint per = numKvHeads * headDim;
    uint b = gid / per, r = gid % per;
    uint h = r / headDim, d = r % headDim;
    uint idx = ((ulong)b * numKvHeads + h) * maxT * headDim + (ulong)pos * headDim + d;
    KC[idx] = K[(ulong)b * per + r];
    VC[idx] = V[(ulong)b * per + r];
}

// batched single-token attention. rows = B * numHeads, one threadgroup per row.
kernel void attention_decode_b(
    device const bf16b* Q [[buffer(0)]], device const bf16b* K [[buffer(1)]], device const bf16b* V [[buffer(2)]],
    device bf16b* O [[buffer(3)]], constant uint& numHeads [[buffer(4)]], constant uint& numKvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]], constant uint& T [[buffer(7)]], constant float& scale [[buffer(8)]],
    constant uint& maxT [[buffer(9)]], uint row [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    uint b = row / numHeads, head = row % numHeads;
    uint group = numHeads / numKvHeads;
    uint kvHead = head / group;
    uint oProjIn = numHeads * headDim;
    uint qbase = (ulong)b * oProjIn + head * headDim;
    float q = (d < headDim) ? bf2f_b(Q[qbase + d]) : 0.0f;
    device const bf16b* Kb = K + ((ulong)b * numKvHeads + kvHead) * maxT * headDim;
    device const bf16b* Vb = V + ((ulong)b * numKvHeads + kvHead) * maxT * headDim;
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (uint t = 0; t < T; ++t) {
        float partial = (d < headDim) ? q * bf2f_b(Kb[(ulong)t * headDim + d]) : 0.0f;
        red[d] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float score = red[0] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float newM = max(m, score);
        float corr = exp(m - newM);
        float p = exp(score - newM);
        if (d < headDim) acc = acc * corr + p * bf2f_b(Vb[(ulong)t * headDim + d]);
        l = l * corr + p;
        m = newM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (d < headDim) O[qbase + d] = bf16b(acc / l);
}

// partial RoPE, batched. grid = B * numHeads * (rotaryDim/2).
kernel void apply_rope_b(
    device bf16b* X [[buffer(0)]], constant uint& numHeads [[buffer(1)]], constant uint& headDim [[buffer(2)]],
    constant uint& rotaryDim [[buffer(3)]], constant uint& pos [[buffer(4)]], constant float& theta [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfDim = rotaryDim / 2u;
    uint per = numHeads * halfDim;
    uint b = gid / per, r = gid % per;
    uint head = r / halfDim, j = r % halfDim;
    uint base = (ulong)b * numHeads * headDim + head * headDim;
    float va = bf2f_b(X[base + j]);
    float vb = bf2f_b(X[base + j + halfDim]);
    float invFreq = pow(theta, -float(2u * j) / float(rotaryDim));
    float freq = float(pos) * invFreq;
    float c = cos(freq), s = sin(freq);
    X[base + j] = bf16b(va * c - vb * s);
    X[base + j + halfDim] = bf16b(vb * c + va * s);
}

// split q/gate, batched. grid = B * oProjIn.
kernel void split_q_gate_b(
    device const bf16b* X [[buffer(0)]], device bf16b* Q [[buffer(1)]], device bf16b* G [[buffer(2)]],
    constant uint& numHeads [[buffer(3)]], constant uint& headDim [[buffer(4)]], constant uint& nb [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint per = numHeads * headDim;
    uint b = gid / per, i = gid % per;
    if (b >= nb) return;
    uint head = i / headDim, d = i % headDim;
    uint base = (ulong)b * per * 2u + head * headDim * 2u;
    Q[(ulong)b * per + i] = X[base + d];
    G[(ulong)b * per + i] = X[base + headDim + d];
}

// batched argmax: one threadgroup per batch element over vocab logits.
kernel void argmax_b(
    device const bf16b* X [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& vocab [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float rv[256];
    threadgroup uint ri[256];
    device const bf16b* Xb = X + (ulong)b * vocab;
    float best = -INFINITY; uint bi = 0u;
    for (uint i = tid; i < vocab; i += tgsize) { float v = bf2f_b(Xb[i]); if (v > best) { best = v; bi = i; } }
    rv[tid] = best; ri[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) { if (rv[tid + s] > rv[tid]) { rv[tid] = rv[tid + s]; ri[tid] = ri[tid + s]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[b] = ri[0];
}

// batched channel slice: dst[b][i] = src[b][srcOffset+i], i<n, src row stride chStride.
kernel void copy_range_b(
    device const bf16b* Src [[buffer(0)]], device bf16b* Dst [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& srcOffset [[buffer(3)]], constant uint& chStride [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint b = gid / n, i = gid % n;
    Dst[(ulong)b * n + i] = Src[(ulong)b * chStride + srcOffset + i];
}

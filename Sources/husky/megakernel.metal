#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// bfloat16 helpers
// ---------------------------------------------------------------------------
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

constant uint TGS [[function_constant(0)]];

// cfg indices (must match Arena.swift CfgIndex)
constant uint C_hiddenSize=0, C_headDim=1, C_numHeads=2, C_numKvHeads=3;
constant uint C_qProjOut=4, C_oProjIn=5, C_inter=6;
constant uint C_kHeads=7, C_vHeads=8, C_kDim=9, C_vDim=10;
constant uint C_keyDim=11, C_valueDim=12, C_convDim=13, C_vocab=14;
constant uint C_maxT=15, C_numLayers=16, C_pos=17, C_ngroups=18;
constant uint C_sHidden=19, C_sXn=20, C_sQkv=21, C_sZ=22, C_sAbg=23;
constant uint C_sConvOut=24, C_sQLin=25, C_sKLin=26, C_sVLin=27, C_sQn=28, C_sKn=29;
constant uint C_sLinOut=30, C_sQBuf=31, C_sGateBuf=32, C_sKBuf=33, C_sVBuf=34;
constant uint C_sAttnOut=35, C_sMix=36, C_sMlpGate=37, C_sMlpUp=38, C_sMlpDown=39;
constant uint C_sLogits=40, C_sPartialVal=41, C_sPartialIdx=42, C_sTokenOut=43;
constant uint C_sBar=44, C_sConvState=45, C_sRecState=46, C_sKCache=47, C_sVCache=48;
constant uint C_sEmbedW=49, C_sEmbedS=50, C_sEmbedB=51, C_sFinalNorm=52, C_sTokenIn=53;
constant uint C_sPos=54;

// layer table fields (stride 32)
constant uint F_inNorm=0, F_postNorm=1;
constant uint F_gateW=2, F_gateS=3, F_gateB=4, F_upW=5, F_upS=6, F_upB=7, F_downW=8, F_downS=9, F_downB=10;
constant uint F_q1W=11, F_q1S=12, F_q1B=13, F_q2W=14, F_q2S=15, F_q2B=16;
constant uint F_q3W=17, F_q3S=18, F_q3B=19, F_q4W=20, F_q4S=21, F_q4B=22;
constant uint F_q5W=23, F_q5S=24, F_q5B=25, F_x1=26, F_x2=27, F_x3=28, F_x4=29, F_type=30;
constant uint LSTRIDE=32;

// ---------------------------------------------------------------------------
// device-wide barrier over a persistent grid
// ---------------------------------------------------------------------------
inline void gbar(device atomic_uint* bar, uint ngroups, thread uint& phase, uint tid) {
    if (ngroups == 1u) {
        threadgroup_barrier(mem_flags::mem_device);
        return;
    }
    // Flush this group's device-memory writes before arriving.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
        uint target = (phase + 1u) * ngroups;
        while (atomic_load_explicit(bar, memory_order_relaxed) < target) { }
    }
    // Ensure all threads wait, then re-fence device memory to pick up peers' writes.
    threadgroup_barrier(mem_flags::mem_device);
    phase += 1u;
}

// ---------------------------------------------------------------------------
// stages
// ---------------------------------------------------------------------------
inline void qmatvec(device char* a, uint wOff, uint sOff, uint bOff, uint xOff, uint yOff,
                    uint outN, uint inN, uint gtid, uint ngt) {
    device const uint* W = (device const uint*)(a + wOff);
    device const bf16* S = (device const bf16*)(a + sOff);
    device const bf16* Bp = (device const bf16*)(a + bOff);
    device const bf16* X = (device const bf16*)(a + xOff);
    device bf16* Y = (device bf16*)(a + yOff);
    uint lane = gtid & 31u, warp = gtid >> 5u, warps = ngt >> 5u;
    uint inWords = inN >> 3u, nGroups = inN >> 6u, nVec = inWords >> 2u;
    for (uint o = warp; o < outN; o += warps) {
        float ac[4] = {0.f, 0.f, 0.f, 0.f};
        for (uint v = lane; v < nVec; v += 32u) {
            uint word = v * 4u, group = word >> 3u;
            float sc = bf2f(S[o * nGroups + group]), bi = bf2f(Bp[o * nGroups + group]);
            uint4 packed = *(device const uint4*)(W + o * inWords + word);
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                uint w = packed[j];
                #pragma unroll
                for (uint k = 0; k < 8u; ++k)
                    ac[j] += (float((w >> (4u*k)) & 15u)*sc+bi)*bf2f(X[word*8u+j*8u+k]);
            }
        }
        float sum = simd_sum(ac[0]+ac[1]+ac[2]+ac[3]);
        if (lane == 0u) Y[o] = bf16(sum);
    }
}

inline void embed_lookup(device char* a, uint wOff, uint sOff, uint bOff, uint yOff,
                         uint rowIndex, uint inN, uint gtid, uint ngt) {
    device const uint* W = (device const uint*)(a + wOff);
    device const bf16* S = (device const bf16*)(a + sOff);
    device const bf16* Bp = (device const bf16*)(a + bOff);
    device bf16* Y = (device bf16*)(a + yOff);
    uint inWords = inN >> 3u, nGroups = inN >> 6u;
    uint row = rowIndex * inWords, srow = rowIndex * nGroups;
    for (uint i = gtid; i < inN; i += ngt) {
        uint j = i >> 3u, k = i & 7u, g = i >> 6u;
        uint w = W[row + j];
        Y[i] = bf16(float((w >> (4u * k)) & 0xFu) * bf2f(S[srow + g]) + bf2f(Bp[srow + g]));
    }
}

inline void rmsnorm_rows(device char* a, uint xOff, uint wOff, uint yOff, uint n, uint nRows,
                         uint addOne, float eps, uint tg, uint tid, uint tgsize, uint ngroups,
                         threadgroup float* red) {
    device const bf16* X = (device const bf16*)(a + xOff);
    device const bf16* Wp = (device const bf16*)(a + wOff);
    device bf16* Y = (device bf16*)(a + yOff);
    for (uint row = tg; row < nRows; row += ngroups) {
        uint o = row * n;
        float local = 0.f;
        for (uint i = tid; i < n; i += tgsize) { float v = bf2f(X[o + i]); local += v * v; }
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(red[0] / float(n) + eps);
        for (uint i = tid; i < n; i += tgsize) {
            float w = bf2f(Wp[i]);
            if (addOne != 0u) w = 1.0f + w;
            Y[o + i] = bf16(bf2f(X[o + i]) * inv * w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void rmsnorm_gated_rows(device char* a, uint xOff, uint wOff, uint zOff, uint yOff,
                               uint n, uint nRows, float eps, uint tg, uint tid, uint tgsize, uint ngroups,
                               threadgroup float* red) {
    device const bf16* X = (device const bf16*)(a + xOff);
    device const bf16* Wp = (device const bf16*)(a + wOff);
    device const bf16* Z = (device const bf16*)(a + zOff);
    device bf16* Y = (device bf16*)(a + yOff);
    for (uint row = tg; row < nRows; row += ngroups) {
        uint o = row * n;
        float local = 0.f;
        for (uint i = tid; i < n; i += tgsize) { float v = bf2f(X[o + i]); local += v * v; }
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(red[0] / float(n) + eps);
        for (uint i = tid; i < n; i += tgsize) {
            float z = bf2f(Z[o + i]);
            float silu = z / (1.0f + exp(-z));
            Y[o + i] = bf16(bf2f(X[o + i]) * inv * bf2f(Wp[i]) * silu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void l2norm_rows(device char* a, uint xOff, uint yOff, uint D, float scale, uint nRows,
                        uint tg, uint tid, uint tgsize, uint ngroups, threadgroup float* red) {
    device const bf16* X = (device const bf16*)(a + xOff);
    device bf16* Y = (device bf16*)(a + yOff);
    for (uint row = tg; row < nRows; row += ngroups) {
        uint o = row * D;
        float local = 0.f;
        for (uint i = tid; i < D; i += tgsize) { float v = bf2f(X[o + i]); local += v * v; }
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgsize >> 1; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = scale * rsqrt(red[0] + 1e-6f);
        for (uint i = tid; i < D; i += tgsize) Y[o + i] = bf16(bf2f(X[o + i]) * inv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void conv_step(device char* a, uint xOff, uint stateOff, uint wOff, uint yOff,
                      uint convDim, uint gtid, uint ngt) {
    device const bf16* X = (device const bf16*)(a + xOff);
    device bf16* St = (device bf16*)(a + stateOff);
    device const bf16* W = (device const bf16*)(a + wOff);
    device bf16* Y = (device bf16*)(a + yOff);
    for (uint c = gtid; c < convDim; c += ngt) {
        float s0 = bf2f(St[c * 3u + 0u]), s1 = bf2f(St[c * 3u + 1u]), s2 = bf2f(St[c * 3u + 2u]);
        float x = bf2f(X[c]);
        float acc = bf2f(W[c * 4u + 0u]) * s0 + bf2f(W[c * 4u + 1u]) * s1
                  + bf2f(W[c * 4u + 2u]) * s2 + bf2f(W[c * 4u + 3u]) * x;
        Y[c] = bf16(acc / (1.0f + exp(-acc)));
        St[c * 3u + 0u] = bf16(s1); St[c * 3u + 1u] = bf16(s2); St[c * 3u + 2u] = bf16(x);
    }
}

inline void copy_range(device char* a, uint srcOff, uint dstOff, uint n, uint srcOffset,
                       uint gtid, uint ngt) {
    device const bf16* S = (device const bf16*)(a + srcOff);
    device bf16* D = (device bf16*)(a + dstOff);
    for (uint i = gtid; i < n; i += ngt) D[i] = S[srcOffset + i];
}

inline void add_inplace(device char* a, uint xOff, uint yOff, uint n, uint gtid, uint ngt) {
    device bf16* X = (device bf16*)(a + xOff);
    device const bf16* Y = (device const bf16*)(a + yOff);
    for (uint i = gtid; i < n; i += ngt) X[i] = bf16(bf2f(X[i]) + bf2f(Y[i]));
}

inline void silu_mul(device char* a, uint gOff, uint uOff, uint yOff, uint n, uint gtid, uint ngt) {
    device const bf16* G = (device const bf16*)(a + gOff);
    device const bf16* U = (device const bf16*)(a + uOff);
    device bf16* Y = (device bf16*)(a + yOff);
    for (uint i = gtid; i < n; i += ngt) { float g = bf2f(G[i]); Y[i] = bf16((g / (1.0f + exp(-g))) * bf2f(U[i])); }
}

inline void mul_sigmoid_gate(device char* a, uint xOff, uint gOff, uint n, uint gtid, uint ngt) {
    device bf16* X = (device bf16*)(a + xOff);
    device const bf16* G = (device const bf16*)(a + gOff);
    for (uint i = gtid; i < n; i += ngt) X[i] = bf16(bf2f(X[i]) / (1.0f + exp(-bf2f(G[i]))));
}

inline void split_q_gate(device char* a, uint xOff, uint qOff, uint gOff, uint numHeads, uint headDim,
                         uint gtid, uint ngt) {
    device const bf16* X = (device const bf16*)(a + xOff);
    device bf16* Q = (device bf16*)(a + qOff);
    device bf16* G = (device bf16*)(a + gOff);
    uint per = numHeads * headDim;
    for (uint i = gtid; i < per; i += ngt) {
        uint head = i / headDim, d = i % headDim;
        uint base = head * headDim * 2u;
        Q[i] = X[base + d];
        G[i] = X[base + headDim + d];
    }
}

inline void apply_rope(device char* a, uint xOff, uint numHeads, uint headDim, uint rotaryDim,
                       uint pos, float theta, uint gtid, uint ngt) {
    device bf16* X = (device bf16*)(a + xOff);
    uint halfDim = rotaryDim / 2u;
    for (uint i = gtid; i < numHeads * halfDim; i += ngt) {
        uint head = i / halfDim, j = i % halfDim;
        uint base = head * headDim;
        float va = bf2f(X[base + j]);
        float vb = bf2f(X[base + j + halfDim]);
        float invFreq = pow(theta, -float(2u * j) / float(rotaryDim));
        float freq = float(pos) * invFreq;
        float c = cos(freq), s = sin(freq);
        X[base + j] = bf16(va * c - vb * s);
        X[base + j + halfDim] = bf16(vb * c + va * s);
    }
}

inline void write_kv(device char* a, uint kOff, uint vOff, uint kcOff, uint vcOff,
                     uint numKvHeads, uint headDim, uint maxT, uint pos, uint gtid, uint ngt) {
    device const bf16* K = (device const bf16*)(a + kOff);
    device const bf16* V = (device const bf16*)(a + vOff);
    device bf16* KC = (device bf16*)(a + kcOff);
    device bf16* VC = (device bf16*)(a + vcOff);
    for (uint i = gtid; i < numKvHeads * headDim; i += ngt) {
        uint h = i / headDim, d = i % headDim;
        uint idx = (h * maxT + pos) * headDim + d;
        KC[idx] = K[i];
        VC[idx] = V[i];
    }
}

inline void g_beta(device char* a, uint aOff, uint bOff, uint alogOff, uint dtOff,
                   uint gOff, uint betaOff, uint H, uint tg, uint tid) {
    if (tg != 0u || tid >= H) return;
    device const bf16* A = (device const bf16*)(a + aOff);
    device const bf16* Bv = (device const bf16*)(a + bOff);
    device const bf16* Alog = (device const bf16*)(a + alogOff);
    device const bf16* Dt = (device const bf16*)(a + dtOff);
    device bf16* G = (device bf16*)(a + gOff);
    device bf16* Beta = (device bf16*)(a + betaOff);
    float x = bf2f(A[tid]) + bf2f(Dt[tid]);
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    G[tid] = bf16(exp(-exp(bf2f(Alog[tid])) * sp));
    Beta[tid] = bf16(1.0f / (1.0f + exp(-bf2f(Bv[tid]))));
}

inline void gdelta(device char* a, uint qnOff, uint knOff, uint vtOff, uint gOff, uint betaOff,
                   uint stateOff, uint outOff, uint numVHeads, uint numKHeads, uint kDim, uint vDim,
                   uint tg, uint tid, uint tgsize, uint ngroups,
                   threadgroup float* ksh, threadgroup float* qsh) {
    device const bf16* QN = (device const bf16*)(a + qnOff);
    device const bf16* KN = (device const bf16*)(a + knOff);
    device const bf16* Vt = (device const bf16*)(a + vtOff);
    device const bf16* G = (device const bf16*)(a + gOff);
    device const bf16* Beta = (device const bf16*)(a + betaOff);
    device bf16* State = (device bf16*)(a + stateOff);
    device bf16* Out = (device bf16*)(a + outOff);
    for (uint h = tg; h < numVHeads; h += ngroups) {
        uint khead = h / (numVHeads / numKHeads);
        uint kbase = khead * kDim;
        uint vbase = h * vDim;
        float g = bf2f(G[h]);
        float beta = bf2f(Beta[h]);
        if (tid < kDim) { ksh[tid] = bf2f(KN[kbase + tid]); qsh[tid] = bf2f(QN[kbase + tid]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < vDim) {
            device bf16* S = State + (ulong)h * vDim * kDim + (ulong)tid * kDim;
            float kv = 0.f;
            for (uint k = 0; k < kDim; ++k) kv += bf2f(S[k]) * ksh[k];
            float delta = (bf2f(Vt[vbase + tid]) - kv) * beta;
            float out = 0.f;
            for (uint k = 0; k < kDim; ++k) {
                float s = bf2f(S[k]) * g + ksh[k] * delta;
                S[k] = bf16(s);
                out += s * qsh[k];
            }
            Out[vbase + tid] = bf16(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void attention(device char* a, uint qOff, uint kcOff, uint vcOff, uint oOff,
                      uint numHeads, uint numKvHeads, uint headDim, uint T, float scale, uint maxT,
                      uint tg, uint tid, uint tgsize, uint ngroups, threadgroup float* red) {
    device const bf16* Q = (device const bf16*)(a + qOff);
    device const bf16* K = (device const bf16*)(a + kcOff);
    device const bf16* V = (device const bf16*)(a + vcOff);
    device bf16* O = (device bf16*)(a + oOff);
    uint group = numHeads / numKvHeads;
    for (uint head = tg; head < numHeads; head += ngroups) {
        uint kvHead = head / group;
        uint qbase = head * headDim;
        float q = (tid < headDim) ? bf2f(Q[qbase + tid]) : 0.0f;
        device const bf16* Kb = K + (ulong)kvHead * maxT * headDim;
        device const bf16* Vb = V + (ulong)kvHead * maxT * headDim;
        float m = -INFINITY, l = 0.f, acc = 0.f;
        for (uint t = 0; t < T; ++t) {
            float partial = (tid < headDim) ? q * bf2f(Kb[(ulong)t * headDim + tid]) : 0.0f;
            red[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgsize >> 1; s > 0; s >>= 1) {
                if (tid < s) red[tid] += red[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float score = red[0] * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float newM = max(m, score);
            float corr = exp(m - newM);
            float p = exp(score - newM);
            if (tid < headDim) acc = acc * corr + p * bf2f(Vb[(ulong)t * headDim + tid]);
            l = l * corr + p;
            m = newM;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < headDim) O[qbase + tid] = bf16(acc / l);
    }
}

// ---------------------------------------------------------------------------
// persistent megakernel: one dispatch runs all layers for one token
// ---------------------------------------------------------------------------
kernel void husky_step(
    device char* arena [[buffer(0)]],
    device const uint* lt [[buffer(1)]],
    constant uint* cfg [[buffer(2)]],
    device uint* tokenOut [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint ngroups = cfg[C_ngroups];
    uint gtid = tg * TGS + tid;
    uint ngt = ngroups * TGS;
    uint phase = 0u;
    device atomic_uint* bar = (device atomic_uint*)(arena + cfg[C_sBar]);
    threadgroup float tgf[1024];
    threadgroup uint tgu[256];

    uint hidden = cfg[C_hiddenSize], headDim = cfg[C_headDim], numHeads = cfg[C_numHeads];
    uint numKvHeads = cfg[C_numKvHeads], oProjIn = cfg[C_oProjIn], inter = cfg[C_inter];
    uint kHeads = cfg[C_kHeads], vHeads = cfg[C_vHeads], kDim = cfg[C_kDim], vDim = cfg[C_vDim];
    uint keyDim = cfg[C_keyDim], valueDim = cfg[C_valueDim], convDim = cfg[C_convDim];
    uint vocab = cfg[C_vocab], maxT = cfg[C_maxT], numLayers = cfg[C_numLayers];
    uint pos = *(device uint*)(arena + cfg[C_sPos]);
    uint sAbg = cfg[C_sAbg];
    uint tokenIn = *(device uint*)(arena + cfg[C_sTokenIn]);
    float eps = 1e-6f;
    float theta = 10000000.0f;
    uint rotaryDim = uint(float(headDim) * 0.25f);

    uint convStride = convDim * 3u;
    uint recStride = vHeads * vDim * kDim;
    uint cacheStride = numKvHeads * maxT * headDim;

    gbar(bar, ngroups, phase, tid);
    embed_lookup(arena, cfg[C_sEmbedW], cfg[C_sEmbedS], cfg[C_sEmbedB], cfg[C_sHidden],
                 tokenIn, hidden, gtid, ngt);

    for (uint li = 0; li < numLayers; ++li) {
        device const uint* L = lt + li * LSTRIDE;
        gbar(bar, ngroups, phase, tid);
        rmsnorm_rows(arena, cfg[C_sHidden], L[F_inNorm], cfg[C_sXn], hidden, 1u, 0u, eps, tg, tid, TGS, ngroups, tgf);
        gbar(bar, ngroups, phase, tid);

        if (L[F_type] == 1u) {
            // ---- linear attention layer ----
            qmatvec(arena, L[F_q1W], L[F_q1S], L[F_q1B], cfg[C_sXn], cfg[C_sQkv], convDim, hidden, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            qmatvec(arena, L[F_q2W], L[F_q2S], L[F_q2B], cfg[C_sXn], cfg[C_sZ], valueDim, hidden, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            qmatvec(arena, L[F_q3W], L[F_q3S], L[F_q3B], cfg[C_sXn], sAbg + 0u * vHeads * 2u, vHeads, hidden, gtid, ngt);
            qmatvec(arena, L[F_q4W], L[F_q4S], L[F_q4B], cfg[C_sXn], sAbg + 1u * vHeads * 2u, vHeads, hidden, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            conv_step(arena, cfg[C_sQkv], cfg[C_sConvState] + li * convStride * 2u, L[F_x1],
                      cfg[C_sConvOut], convDim, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            copy_range(arena, cfg[C_sConvOut], cfg[C_sQLin], keyDim, 0u, gtid, ngt);
            copy_range(arena, cfg[C_sConvOut], cfg[C_sKLin], keyDim, keyDim, gtid, ngt);
            copy_range(arena, cfg[C_sConvOut], cfg[C_sVLin], valueDim, 2u * keyDim, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            l2norm_rows(arena, cfg[C_sQLin], cfg[C_sQn], kDim, rsqrt(float(kDim)), kHeads, tg, tid, TGS, ngroups, tgf);
            l2norm_rows(arena, cfg[C_sKLin], cfg[C_sKn], kDim, 1.0f, kHeads, tg, tid, TGS, ngroups, tgf);
            gbar(bar, ngroups, phase, tid);
            g_beta(arena, sAbg + 0u * vHeads * 2u, sAbg + 1u * vHeads * 2u, L[F_x2], L[F_x3],
                   sAbg + 2u * vHeads * 2u, sAbg + 3u * vHeads * 2u, vHeads, tg, tid);
            gbar(bar, ngroups, phase, tid);
            gdelta(arena, cfg[C_sQn], cfg[C_sKn], cfg[C_sVLin], sAbg + 2u * vHeads * 2u,
                   sAbg + 3u * vHeads * 2u, cfg[C_sRecState] + li * recStride * 2u, cfg[C_sLinOut],
                   vHeads, kHeads, kDim, vDim, tg, tid, TGS, ngroups, tgf, tgf + 128);
            gbar(bar, ngroups, phase, tid);
            rmsnorm_gated_rows(arena, cfg[C_sLinOut], L[F_x4], cfg[C_sZ], cfg[C_sLinOut],
                               vDim, valueDim / vDim, eps, tg, tid, TGS, ngroups, tgf);
            gbar(bar, ngroups, phase, tid);
            qmatvec(arena, L[F_q5W], L[F_q5S], L[F_q5B], cfg[C_sLinOut], cfg[C_sMix], hidden, valueDim, gtid, ngt);
        } else {
            // ---- full attention layer ----
            qmatvec(arena, L[F_q1W], L[F_q1S], L[F_q1B], cfg[C_sXn], cfg[C_sQkv], numHeads * headDim * 2u, hidden, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            split_q_gate(arena, cfg[C_sQkv], cfg[C_sQBuf], cfg[C_sGateBuf], numHeads, headDim, gtid, ngt);
            qmatvec(arena, L[F_q2W], L[F_q2S], L[F_q2B], cfg[C_sXn], cfg[C_sKBuf], numKvHeads * headDim, hidden, gtid, ngt);
            qmatvec(arena, L[F_q3W], L[F_q3S], L[F_q3B], cfg[C_sXn], cfg[C_sVBuf], numKvHeads * headDim, hidden, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            rmsnorm_rows(arena, cfg[C_sQBuf], L[F_x1], cfg[C_sQBuf], headDim, numHeads, 0u, eps, tg, tid, TGS, ngroups, tgf);
            rmsnorm_rows(arena, cfg[C_sKBuf], L[F_x2], cfg[C_sKBuf], headDim, numKvHeads, 0u, eps, tg, tid, TGS, ngroups, tgf);
            gbar(bar, ngroups, phase, tid);
            apply_rope(arena, cfg[C_sQBuf], numHeads, headDim, rotaryDim, pos, theta, gtid, ngt);
            apply_rope(arena, cfg[C_sKBuf], numKvHeads, headDim, rotaryDim, pos, theta, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            write_kv(arena, cfg[C_sKBuf], cfg[C_sVBuf], cfg[C_sKCache] + li * cacheStride * 2u,
                     cfg[C_sVCache] + li * cacheStride * 2u, numKvHeads, headDim, maxT, pos, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            attention(arena, cfg[C_sQBuf], cfg[C_sKCache] + li * cacheStride * 2u,
                      cfg[C_sVCache] + li * cacheStride * 2u, cfg[C_sAttnOut],
                      numHeads, numKvHeads, headDim, pos + 1u, rsqrt(float(headDim)), maxT, tg, tid, TGS, ngroups, tgf);
            gbar(bar, ngroups, phase, tid);
            mul_sigmoid_gate(arena, cfg[C_sAttnOut], cfg[C_sGateBuf], oProjIn, gtid, ngt);
            gbar(bar, ngroups, phase, tid);
            qmatvec(arena, L[F_q4W], L[F_q4S], L[F_q4B], cfg[C_sAttnOut], cfg[C_sMix], hidden, oProjIn, gtid, ngt);
        }

        gbar(bar, ngroups, phase, tid);
        add_inplace(arena, cfg[C_sHidden], cfg[C_sMix], hidden, gtid, ngt);
        gbar(bar, ngroups, phase, tid);
        rmsnorm_rows(arena, cfg[C_sHidden], L[F_postNorm], cfg[C_sXn], hidden, 1u, 0u, eps, tg, tid, TGS, ngroups, tgf);
        gbar(bar, ngroups, phase, tid);
        qmatvec(arena, L[F_gateW], L[F_gateS], L[F_gateB], cfg[C_sXn], cfg[C_sMlpGate], inter, hidden, gtid, ngt);
        qmatvec(arena, L[F_upW], L[F_upS], L[F_upB], cfg[C_sXn], cfg[C_sMlpUp], inter, hidden, gtid, ngt);
        gbar(bar, ngroups, phase, tid);
        silu_mul(arena, cfg[C_sMlpGate], cfg[C_sMlpUp], cfg[C_sMlpGate], inter, gtid, ngt);
        gbar(bar, ngroups, phase, tid);
        qmatvec(arena, L[F_downW], L[F_downS], L[F_downB], cfg[C_sMlpGate], cfg[C_sMlpDown], hidden, inter, gtid, ngt);
        gbar(bar, ngroups, phase, tid);
        add_inplace(arena, cfg[C_sHidden], cfg[C_sMlpDown], hidden, gtid, ngt);
    }

    gbar(bar, ngroups, phase, tid);
    rmsnorm_rows(arena, cfg[C_sHidden], cfg[C_sFinalNorm], cfg[C_sXn], hidden, 1u, 0u, eps, tg, tid, TGS, ngroups, tgf);
    gbar(bar, ngroups, phase, tid);
    qmatvec(arena, cfg[C_sEmbedW], cfg[C_sEmbedS], cfg[C_sEmbedB], cfg[C_sXn], cfg[C_sLogits], vocab, hidden, gtid, ngt);

    // in-kernel argmax
    gbar(bar, ngroups, phase, tid);
    device bf16* logits = (device bf16*)(arena + cfg[C_sLogits]);
    threadgroup float rv[1024];
    threadgroup uint ri[1024];
    float best = -INFINITY;
    uint bi = 0u;
    for (uint i = gtid; i < vocab; i += ngt) { float v = bf2f(logits[i]); if (v > best) { best = v; bi = i; } }
    rv[tid] = best; ri[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TGS >> 1; s > 0; s >>= 1) {
        if (tid < s) { if (rv[tid + s] > rv[tid]) { rv[tid] = rv[tid + s]; ri[tid] = ri[tid + s]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        device float* pv = (device float*)(arena + cfg[C_sPartialVal]);
        device uint* pi = (device uint*)(arena + cfg[C_sPartialIdx]);
        pv[tg] = rv[0]; pi[tg] = ri[0];
    }
    gbar(bar, ngroups, phase, tid);
    if (tg == 0u && tid == 0u) {
        device float* pv = (device float*)(arena + cfg[C_sPartialVal]);
        device uint* pi = (device uint*)(arena + cfg[C_sPartialIdx]);
        float bv = -INFINITY; uint bidx = 0u;
        for (uint g = 0; g < ngroups; ++g) if (pv[g] > bv) { bv = pv[g]; bidx = pi[g]; }
        tokenOut[0] = bidx;
    }
}

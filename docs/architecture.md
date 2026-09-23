# Architecture

## The model: Woof (Qwen3.5 hybrid, text-only)

`ConwayResearch/Underdog-Woof-4B-1.1` is a 4-bit affine-quantized hybrid
transformer. Config (from `config.json`):

| field | value |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` (text-only used here) |
| layers | 32 |
| layer types | 24 × `linear_attention`, 8 × `full_attention` (every 4th) |
| hidden size | 2560 |
| intermediate size | 9216 |
| attention heads | 16 q, 4 kv, `head_dim` 256 |
| RoPE | partial, factor 0.25 → 64 rotated dims; theta 1e7 (mrope config, text-only degenerates to standard) |
| linear attn | 16 key heads × 128, 32 value heads × 128, depthwise conv kernel 4 |
| vocab | 248320, tied embeddings |
| quantization | 4-bit affine, group size 64 (MLX format) |
| file size / params | 2.37 GB / 4.0 B |

Tensor names are prefixed `language_model.model.` (the text tower of the
conditional-generation wrapper). Quantized matrices are stored as three tensors:
`weight` (`U32`, packed 4-bit), `scales` (`BF16`), `biases` (`BF16`) — the MLX
affine layout (`[out, in/8]` words, `[out, in/64]` scales/biases).

### Per-layer computation (matches `transformers` `qwen3_5`)

**Full attention layer** (`Qwen3_5Attention`):
- `q_proj` outputs `num_heads * head_dim * 2`; split into `query` and `gate`.
- `q_norm` / `k_norm` are RMSNorm over `head_dim` (weights used **directly** —
  see the 1-centered norm note below).
- partial RoPE on the first 64 dims of each head; GQA with 4 kv heads.
- output `attn_output * sigmoid(gate)` then `o_proj`.

**Linear attention layer** (`Qwen3_5GatedDeltaNet`, gated delta net):
- `in_proj_qkv` → depthwise causal conv (kernel 4) with a resident 3-token state → split q/k/v.
- `in_proj_z` → gate; `in_proj_a`/`in_proj_b` → per-head `g`/`beta`.
- q and k are L2-normalized per key head (q additionally scaled by `1/sqrt(dim)`).
- gated delta recurrence with a resident `[v_heads][v_dim][k_dim]` state:
  `S = S*exp(g); kv = S·k; delta = (v - kv)*beta; S += k⊗delta; out = S·q`.
- gated RMSNorm (`rmsnorm(x)*silu(z)`) then `out_proj`.

**MLP** (all layers): SwiGLU — `down(silu(gate(x)) * up(x))`.

**Decoder layer**: `h = x + mixer(norm(x)); out = h + mlp(norm(h))`.

> Norm weights: `Qwen3_5RMSNorm` is defined as `(1 + weight)` in `transformers`,
> but this checkpoint (converted by `mlx-lm`) stores **1-centered** weights and
> MLX consumes them directly. We consume them directly too. Getting this wrong was
> one of the first bugs (see `porting-log.md`).

## The engine

Two forward paths share the same weight/dequant math:

### 1. Multi-kernel path (`Model.swift` + `kernels.metal`) — production

One command buffer per token, ~580 compute dispatches back-to-back. Per-token
graph:

```
embed_lookup(token) -> hidden
for each layer:
    rmsnorm(input_layernorm) -> xn
    [linear] qkv,z,a,b matvecs; conv_step; split; l2norm(q,k); g_beta; gdelta;
             gated rmsnorm; out_proj(+residual)
    [full]   q,k,v matvecs; split_q_gate; rmsnorm(q,k); rope; write_kv;
             attention_decode; sigmoid-gate; o_proj(+residual)
    rmsnorm(post_attention_layernorm) -> xn
    gate/up matvec (fused, SwiGLU applied in-kernel) -> down matvec(+residual)
rmsnorm(final) -> xn
logits matvec (tied embed) -> argmax
```

State is kept resident across tokens: conv state and recurrent state per linear
layer; K/V caches per full-attention layer. Only token ids/positions are written
from the host.

### 2. Megakernel path (`Arena.swift`, `Megakernel.swift`, `megakernel.metal`)

Every weight, scratch buffer, activation, recurrent/KV state, and even the
barrier counter and output token live in **one packed `MTLBuffer`** plus a small
per-layer offset table. A single persistent kernel (`husky_step`) runs the whole
model for one token with in-kernel argmax — one dispatch per token instead of
~580. See `megakernel-and-memory-model.md` for why it is currently limited to a
single threadgroup.

### Weight layout & memory

- Weights are uploaded to **GPU-private** (`storageModePrivate`) buffers via a
  one-time blit from CPU memory. This measured ~10% faster than
  `storageModeShared` for the matvec access pattern.
- Scales/biases are kept bf16 (as stored), weights as packed `U32`.
- All activations and state are **bf16**, arithmetic accumulates in `f32`, and
  results are rounded back to bf16 — mirroring MLX's numeric behavior.

### Kernels (`kernels.metal`)

| kernel | role |
|---|---|
| `qmatvec_warp` | coalesced quantized matvec, one warp per output row, `uint4` weight loads, `simd_sum` reduction |
| `qmatvec_warp_add` | same, plus residual add (folds the residual into the projection) |
| `qmatvec_pair_silu` | gate+up projections in one dispatch with SwiGLU applied in-kernel |
| `embed_lookup` | gather + dequantize one embedding row |
| `rmsnorm` / `rmsnorm_gated` | row RMSNorm (`addOne` flag) / gated RMSNorm |
| `apply_rope` | partial RoPE, computed in-kernel from position |
| `conv_step` | depthwise causal conv with resident state |
| `l2norm_scale` | per-head L2 norm + scale |
| `g_beta` | `g`/`beta` for the delta rule |
| `gdelta_step` | one recurrent gated-delta step per value head |
| `attention_decode` | single-token GQA attention over the KV cache |
| `write_kv`, `split_q_gate`, `mul_sigmoid_gate`, `silu_mul`, `copy_range`, `add_inplace`, `argmax_partial` | support ops |

### Tokenizer

`Tokenizer.swift` implements the Qwen byte-level BPE directly from
`tokenizer.json`: GPT-2 byte↔unicode mapping, the Qwen pre-tokenizer regex, merge
ranks, and special-token handling. No external dependency.

### Toolchain notes

- MSL is compiled at runtime (`device.makeLibrary(source:)`) with
  `MTLLanguageVersion.version3_0`; the `.metal` files ship as package resources.
- `bfloat16_t` is not exposed, so a tiny `bf16` struct with `bf2f`/`f2bf` helpers
  is defined in each MSL source.
- No SwiftPM dependencies.

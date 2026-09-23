# Porting log — implementation, trials, and errors

This is the honest record of building the engine: every wrong turn, how it was
found, and what fixed it. Numbers are from an M4 Pro / 128 GB, macOS 27 SDK,
Swift 6.4.

## 0. Ground truth first

Before writing any Metal, the reference was established:

- `transformers` 5.3.0 ships `qwen3_5`, and `mlx_lm` 0.31.0 ships `qwen3_5` — both
  can load the checkpoint. Since the file is `bf16` + MLX affine quant, `mlx_lm`
  is the natural golden reference.
- Golden greedy output for `"The capital of France is"`:
  `[11751, 13, 198, 32, 13, 2912, ...]`, prompt ids `[760, 6511, 314, 9338, 369]`.
- A small `mlx_lm` script also dumps **per-layer activations** for a fixed token
  prefix; this was the single most valuable debugging tool (see §4).

## 1. Architecture misread #1 — the 1-centered norm

`Qwen3_5RMSNorm.forward` computes `output * (1.0 + weight)` and initializes
`weight` to zeros. So the first implementation passed `addOne = 1` to the RMSNorm
kernel. Output was garbage.

The failure was found by dumping the embedding and first layers and noticing the
norm output was ~2× too large. Inspecting the checkpoint:

```
input_layernorm.weight  mean 1.14  min 0.98  max 2.66
```

The stored weights are already 1-centered. Critically, `mlx-lm`'s `sanitize()`
only adds 1.0 to norm weights when the checkpoint has MTP weights or an
unsanitized conv1d — neither is true here — so **MLX uses the weights directly**.
Fix: `addOne = 0` everywhere.

**Lesson:** "Qwen3.5 norm = 1+w" is true of the `transformers` module, but the
checkpoint's prevailing convention is set by the converter (`mlx-lm`). Follow the
reference that produced the numbers.

## 2. Architecture misread #2 — KV cache stride

After the norm fix, the model matched MLX for 2 generated tokens, then diverged at
the third. The per-layer dump showed the divergence began exactly at the first
full-attention layer (layer 3). The attention kernel indexed the KV cache as:

```metal
device const bf16* Kb = K + kvHead * T * headDim;   // WRONG
```

but the cache is laid out `[kvHead][maxT][headDim]`, and tokens are written at
`(h*maxT + pos)*headDim`. The head stride is `maxT`, not the current sequence
length `T`. For `kvHead > 0` the reads landed in the wrong head's region. Fix:
pass `maxT` and use `kvHead * maxT * headDim`.

**Lesson:** the first 1–2 tokens matched by coincidence (`"Paris"` is a very
likely completion); a 2-token agreement is not validation. The per-layer diff
localized it immediately.

## 3. Numerics — f32 vs bf16

Even after both fixes, open-ended prompts diverged after ~5 tokens while
`2+2=` matched. Layer diffs were smooth and small (1e-4 → 2e-2 over 3 layers),
i.e. not a logic bug but accumulation order. The engine was computing in `f32`
while MLX computes in `bf16` (the checkpoint dtype).

Fix: switch **all** activations and state to `bf16`, with `f32` accumulation and
bf16 rounding on store, mirroring MLX. After this the embedding matched MLX
bit-exactly and per-layer diffs were ~1 ULP.

Remaining divergence is inherent: MLX's tiled reductions sum in a different order
than our sequential ones, so hand-written kernels cannot be bit-identical.
Result: deterministic prompts (`2+2=`) match fully; open-ended prompts match a
prefix then diverge when two continuations are near-tied. This is the same class
of behavior the blog describes ("greedy output compared token by token") and is
why we kept the MLX-dump harness permanently.

## 4. The debugging harness

`HUSKY_DUMP=1 HUSKY_DUMP_DIR=dir` makes `Model.forward` copy the post-embedding
hidden state and the hidden state after every decoder layer into per-layer
buffers, plus the final logits (bf16→f32). A Python script mirrors the reference
(`mlx_lm`) for the same token prefix and writes the same files. A one-line diff of
max-abs per layer pinpoints the first diverging layer.

This turned "the model is wrong" into "layer N is wrong" every time:
- norm bug → visible at layer 0.
- KV stride bug → first appears at the first full-attention layer.
- precision → smooth growth across all layers.

`tap_*` dumps (q/k before & after norm/RoPE, v, attention output) were added
later for the attention layer.

## 5. Metal toolchain fights

| problem | error | fix |
|---|---|---|
| `bfloat16_t` unknown | `unknown type name 'bfloat16_t'` (no language version enables it) | define `struct bf16 { ushort bits; ... }` with `bf2f`/`f2bf` free functions |
| `threadgroup` locals in helpers | `variables in the threadgroup address space cannot be declared in a non-qualified function` | declare scratch in the kernel and pass pointers into the helpers |
| `half` keyword | `cannot combine with previous 'type-name'` | rename local `half` → `halfDim` |
| fences | `use of undeclared identifier 'memory_order_acq_rel'` | MSL only exposes `memory_order_relaxed` (see `megakernel-and-memory-model.md`) |

## 6. The megakernel and the grid-barrier wall

A single persistent kernel (`husky_step`) was built that runs all 32 layers with
in-kernel argmax, fed by one packed arena. With `--groups 1` (a single
threadgroup) it is **correct** but uses one GPU core (~3.4 tok/s). Attempts to use
many threadgroups with a device-wide barrier produced wrong results. This was
investigated to the root and is written up separately in
`megakernel-and-memory-model.md`. Summary:

- a shared RMW counter barrier is logically unsound (a group's own increment
  satisfies its own wait) — this was the first false lead;
- even a correct all-to-all sentinel barrier leaves **stale reads** because normal
  device stores are not coherent across threadgroups on this GPU;
- only atomic-accessed locations are coherent;
- MSL here exposes no acquire/release ordering or fences, so a correct grid-wide
  barrier is impossible on this toolchain.

This matches the blog's own note that the megakernel "measured honestly: identical
output, and no faster".

## 7. What "success" looked like

- Loads the real 4B/4-bit checkpoint, generates coherent text, ties embeddings.
- `2+2=` generated bit-identically to MLX (12/12 tokens).
- `The capital of France is` → ` Paris.` (same first tokens as MLX).
- ~50–57 tok/s on M4 Pro after the performance work (`performance.md`).

## 8. Validation checklist used throughout

1. Tokenizer: prompt encodes to the same ids as the reference.
2. Embedding row: dequant matches `mx.dequantize` exactly (bit-for-bit).
3. Per-layer hidden states vs MLX dumps (max-abs and relative).
4. Final logits argmax vs MLX at every generated step.
5. Full greedy sequence comparison on several prompts (deterministic ones must
   match exactly).

# Performance — the optimization journey and measured limits

Hardware: **Apple M4 Pro** (20-core GPU class), 128 GB. Measurement harness:
`HUSKY_TIMING=1` reports host-encode time and GPU time per token; throughput is
generated tokens over wall time (includes prefill, so short runs look slower).

## Where the time goes

Greedy token generation is memory-bound: each token streams the whole model.

```
weights:        2.36 GB
scales+biases:  0.26 GB
total:          2.62 GB per token
```

A calibrated raw-read benchmark put the achievable GPU read bandwidth at
**~244 GB/s** (best of several patterns), so the absolute floor is
`2.62 / 0.244 ≈ 10.7 ms/token`. Everything else is overhead.

## Timeline

| stage | GPU ms/token | tok/s (128 tok) |
|---|---|---|
| initial correct engine (thread-per-output qmatvec, f32→bf16) | 28.1 | 34 |
| warp-per-output-row matvec, `uint4` loads, `simd_sum` | 14.0 | 57 |
| `storageModePrivate` weights | 13.7 | ~55 |
| residual-add folded into projections; SwiGLU folded into gate/up | 13.6 | ~55 |

Host-encode is ~0.25–0.5 ms/token; the GPU is the critical path. Dispatch
overhead measured ~1 µs each (see below), so the engine is **not** launch-bound.

## The win: coalesced warp-per-row matvec

The original `qmatvec` assigned one **thread** per output row. Each thread walked
its own 1280-byte row, so the 32 threads of a warp touched 32 different cache
lines per step — uncoalesced. Redesign:

- **one warp per output row** — lanes stride across the row, so every step is a
  contiguous 512-byte coalesced transaction;
- **`uint4` (128-bit) weight loads** — 4× fewer load instructions;
- **`simd_sum`** warp reduction instead of a per-thread full-row sum.

This alone took 28.1 → 14.0 ms.

## Experiments that did NOT help (and why)

All were built, measured, and reverted/removed:

| experiment | result | reason |
|---|---|---|
| vectorized `uint4` X loads | neutral | X is 5 KB and L1-resident |
| 2 rows/warp | 14.1 ms | same bytes, fewer warps |
| 4 rows/warp (8 lanes/row) | 14.6 ms | fewer warps, 128-B transactions |
| 8 adjacent rows/warp (10 KB block) | 15.6 ms | long burst but 8× fewer warps |
| threadgroup 32/64/128/256 | neutral | occupancy already fine |
| multiple accumulators | neutral | not FMA-latency-bound |
| gate+up fused (one dispatch) | neutral | not dispatch-bound |
| **transposed weights + thread-per-output** | 43 ms | `outN`≈9k ⇒ ~9k threads only; too few to hide latency; strided stream |
| **split-K (more warps)** | 28 ms | slices shorter than 32 `uint4` idle most lanes; even full-lane slices stayed 28 ms |

## Calibration microbenchmarks

These are the measurements that explain the ceiling:

| benchmark | GB/s |
|---|---|
| contiguous grid-stride read, `storageModePrivate` | 238–244 |
| row-per-warp, `storageModePrivate`, 2304–589k warps (2 GB) | 226–245 |
| row-per-warp, `storageModeShared`, 2304–9216 warps | **211–216** |
| row-per-warp, `storageModeShared`, 147k warps | 236 |
| 2 GB split into 1 / 16 / 64 / 256 / 700 chained dispatches | 234 / 233 / 224 / 218 / **202** |
| our real weight stream, math stripped (`storageModeShared`) | **~188** |
| empty dispatch overhead | ~1.1–3.5 µs |

Two durable findings:

1. **`storageModeShared` costs ~10%** versus private for this pattern (~216 vs
   ~238 at our warp counts; worse at low warp counts). → weights moved to private.
2. **Chaining dispatches costs ~14%** by 700 dispatches (234 → 202 GB/s) —
   inherent to a sequential decoder that issues ~580 dispatches/token.

## Why 9 ms (and 11 ms) are not reachable here

- 9 ms ⇒ `2.36 / 0.009 ≈ 262 GB/s` (weights alone), or `2.62/0.009 ≈ 291 GB/s`
  with scales/biases — **above the measured 244 GB/s peak**.
- 11 ms ⇒ `2.62 / 0.011 ≈ 238 GB/s` sustained — that is 98% of peak, across
  ~580 short dependency-chained dispatches. Not attainable.
- The realistic floor for this engine is ~12 ms (`2.62 / 0.216 ≈ 12.1 ms` at the
  private row-pattern bandwidth our warp counts reach). We are at ~13.6 ms.

The measured "more warps ⇒ more bandwidth" effect cannot be exploited: a
9216-row matrix has at most 9216 warps one-row-per-warp, and giving each warp more
rows lengthens its stream but cuts the warp count — the two effects cancel on this
GPU. This is the fundamental short-row limitation.

## What is left on the table (~0.5–1.5 ms)

- ~580 dispatches × ~1–3 µs launch/tail = ~0.6–1.7 ms. Folding the remaining
  small dispatches (conv's split-copy, the two L2-norm calls, attention's output
  gate) is worth maybe ~0.3–0.5 ms. (Two folds so far — residual add and
  SwiGLU — recovered ~0.13 ms.)
- Scales/biases are ~1.1 ms of irreducible traffic.
- Bigger algorithmic levers (not matvec): prompt-lookup speculative decoding
  (fewer forward passes) and an 8-bit KV cache (longer contexts). These buy more
  tok/s than further matvec grinding.

## Accuracy impact

Optimizations changed **speed only**; dequant math and accumulation order in the
matvec are unchanged (warp-per-row vs thread-per-row sum the same products in f32,
just in a different order), so the correctness results are identical before and
after:

- `2+2=` → `[19, 3709, 19, 10, 19, 28, 23, 3709, 23, 10, 19, 28]` — 12/12 exact vs MLX.
- `The capital of France is` → ` Paris.` (matches MLX through token 5, then the
  usual near-tie divergence).

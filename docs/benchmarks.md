# Historical token-by-token benchmark — input/output length × batch

These measurements describe `--bench`, which intentionally uses the batched
single-token path. Normal generation now has chunked prefill; see
[prefill.md](prefill.md) for its current 635–669 TPS measurements.

Harness: `husky --bench` (`Sources/husky/Bench.swift`). Hardware: **Apple M4 Pro**.
Weights resident; one command buffer per token; greedy decode.

- **prefill** = processing the prompt, **one token per forward pass** (no chunked
  prefill yet), so prefill throughput ≈ decode throughput by construction.
- **decode** = generating tokens after the prompt.
- Throughput is aggregate across the batch (tokens/s); latency columns are
  per-step wall time for the whole batch.

## Correctness first

Batched runs with identical prompts reproduce batch 1 exactly:

```
batch 1: [11751, 13, 198, 32, 13, 220, 97901, 198]
batch 2: [11751, 13, 198, 32, 13, 220, 97901, 198]  OK
batch 4: [11751, 13, 198, 32, 13, 220, 97901, 198]  OK
```

## Results

| batch | in | out | prefill t/s | decode t/s | prefill ms/tok | decode ms/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 32 | 57.8 | 57.2 | 17.30 | 17.47 |
| 1 | 16 | 128 | 57.9 | 55.2 | 17.28 | 18.12 |
| 1 | 64 | 32 | 57.3 | 54.7 | 17.46 | 18.28 |
| 1 | 64 | 128 | 56.8 | 53.4 | 17.60 | 18.73 |
| 1 | 128 | 32 | 55.9 | 53.1 | 17.89 | 18.84 |
| 1 | 128 | 128 | 56.0 | 51.2 | 17.85 | 19.54 |
| 1 | 256 | 32 | 53.5 | 48.5 | 18.70 | 20.62 |
| 1 | 256 | 128 | 53.5 | 46.5 | 18.69 | 21.52 |
| 2 | 16 | 32 | 68.6 | 67.9 | 29.14 | 29.45 |
| 2 | 16 | 128 | 68.0 | 66.6 | 29.42 | 30.03 |
| 2 | 64 | 32 | 68.0 | 67.5 | 29.40 | 29.63 |
| 2 | 64 | 128 | 68.5 | 63.3 | 29.18 | 31.58 |
| 2 | 128 | 32 | 61.3 | 59.0 | 32.64 | 33.90 |
| 2 | 128 | 128 | 60.9 | 57.8 | 32.83 | 34.57 |
| 2 | 256 | 32 | 59.4 | 55.8 | 33.67 | 35.82 |
| 2 | 256 | 128 | 61.8 | 57.4 | 32.37 | 34.84 |
| 4 | 16 | 32 | 72.0 | 69.9 | 55.55 | 57.23 |
| 4 | 16 | 128 | 69.8 | 71.2 | 57.29 | 56.20 |
| 4 | 64 | 32 | 72.1 | 71.1 | 55.51 | 56.23 |
| 4 | 64 | 128 | 72.0 | 70.4 | 55.52 | 56.78 |
| 4 | 128 | 32 | 71.6 | 70.3 | 55.90 | 56.90 |
| 4 | 128 | 128 | 70.9 | 69.5 | 56.38 | 57.55 |
| 4 | 256 | 32 | 70.1 | 68.3 | 57.04 | 58.54 |
| 4 | 256 | 128 | 69.0 | 64.3 | 57.95 | 62.21 |

## Interpretation

**Longer context costs throughput.** At batch 1, decode drops from ~57 tok/s
(16-token prompt) to ~46 tok/s (256-token prompt, 128 generated). The extra cost
is full-attention over a growing KV cache; the linear-attention layers keep a
constant-size state and do not grow with context.

**Batching helps only modestly (~1.2×), not ~4×.** Aggregate decode throughput
goes 57 → 67 → 70 tok/s for batch 1 → 2 → 4, while per-step latency goes
17 → 30 → 56 ms. Ideal batching on a purely weight-bandwidth-bound engine would
approach ~4× (the weights are read once for all sequences). Getting ~1.2× means
the engine is **not** purely memory-bound once batched:

- The 4-bit dequant matvec does per-weight work (nibble unpack, `q*scale+bias`,
  FMA) that scales with batch, while weight traffic does not. At batch 1 that
  work hides behind memory latency; at batch ≥2 the arithmetic becomes the
  bottleneck and caps the gain.
- The non-matvec kernels (attention, gated-delta, norms, elementwise) scale
  linearly with batch and add latency that does not amortize.

So on this GPU the matvec is closer to **compute/issue-bound than
bandwidth-bound** at the margin, which is consistent with the single-stream
measurement: ~13.6 ms/token at ~216 GB/s, i.e. the arithmetic is not free.

**Prefill has no dedicated fast path.** Prefill is done one token at a time
(the same kernels as decode), so prompt processing is not accelerated by
parallelism over prompt tokens. A chunked/batched prefill would raise prefill
throughput substantially and is the obvious next optimization for
latency-sensitive use.

## Reproduce

```sh
swift build -c release
.build/release/husky --model /tmp/woof --bench
```

The sweep arrays (`inputLens`, `outputLens`, `batches`) are at the bottom of
`Sources/husky/Bench.swift`; a batch of 8 is supported by `MAXB` in
`batchkernel.metal`.

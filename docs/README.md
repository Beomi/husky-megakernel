# Husky — a native Metal inference engine for Underdog's Woof

This repository contains a from-scratch **Swift + Metal** implementation of a
model-specific inference engine ("MSI") for **Woof**, Underdog's 4B 4-bit hybrid
model, running on Apple silicon. It was built in response to the Husky
announcement ([blog](https://husky.underdog.ai), [tweet](https://x.com/0xsigil/status/2102165862065328538)).

It loads the real `ConwayResearch/Underdog-Woof-4B-1.1` weights, tokenizes with a
byte-level BPE implementation, runs the full forward pass on the GPU, and
generates text greedily — with no Python and no MLX at inference time.

## Status

| | |
|---|---|
| Correctness | Matches the MLX reference token-for-token on deterministic prompts (`2+2=` → 12/12 exact). Open-ended prompts share a prefix then diverge due to float accumulation order (see `performance.md`). |
| Throughput | Prefill **~635–669 tok/s** for 256–512-token prompts on an M4 Pro; decode historically ~50–57 tok/s. See `prefill.md` for warmed comparisons and hardware measurements. |
| Coverage | 24 Gated-DeltaNet (linear attention) layers + 8 GQA full-attention layers, 4-bit affine dequant in-kernel, 8/16-bit states, tied embeddings. |
| Also included | An optimized single-threadgroup **megakernel** prototype, reproducible hardware/kernel benchmarks, and numerical checks. Multi-threadgroup execution still lacks a correct cross-group handoff. |

## Requirements

- Apple silicon Mac, macOS 14+
- Xcode / Swift 6.x toolchain
- The Woof weights, e.g. downloaded to `/tmp/woof`:
  ```sh
  hf download ConwayResearch/Underdog-Woof-4B-1.1 --local-dir /tmp/woof
  ```

## Build & run

```sh
swift build -c release

# greedy generation
.build/release/husky --model /tmp/woof --prompt "The capital of France is" --max-tokens 16

# feed explicit token ids (useful for A/B testing against a reference)
.build/release/husky --model /tmp/woof --ids "760,6511,314,9338,369" --max-tokens 8

# the single-dispatch megakernel path
.build/release/husky --model /tmp/woof --megakernel --groups 1 --prompt "2+2=" --max-tokens 8

# inference speed sweep over input/output length and batch (see docs/benchmarks.md)
.build/release/husky --model /tmp/woof --bench

# hardware and correctness checks
.build/release/husky --bench-hardware
.build/release/husky --check-kernels
.build/release/husky --model /tmp/woof --check-prefill
.build/release/husky --model /tmp/woof --bench-megakernel

# chunked-prefill speed test (see docs/prefill.md)
.build/release/husky --model /tmp/woof --bench-prefill
```

Normal generation uses 256-token prefill chunks for prompts of at least 64 tokens,
then continues decode with the same caches. `--prefill-chunk N` explicitly selects
a chunk size; `--prefill-chunk 0` selects token-by-token prefill. fp16 preparation
is included in model loading and adds 7.14 GB of resident weights.

Environment knobs:

| var | meaning |
|---|---|
| `HUSKY_CHUNKS=64,128,256`, `HUSKY_LENGTHS=64,128,256,512` | prefill benchmark sweep |
| `HUSKY_REPEATS=3` | timed repetitions after warmup; report median |
| `HUSKY_PREFILL_REFERENCE=1` | benchmark original recurrence/attention kernels |
| `HUSKY_PREFILL_PROF=1` | isolated command-buffer stage timing; perturbs throughput |
| `HUSKY_MEGAKERNEL_TG=1024` | megakernel threads (256/512/1024); 1024 was fastest |
| `HUSKY_TIMING=1` | print host-encode vs GPU time per token |
| `HUSKY_TG=n` | threads per threadgroup for matvec (default 128) |
| `HUSKY_DUMP=1` + `HUSKY_DUMP_DIR=dir` | dump per-layer activations for diffing against a reference |
| `HUSKY_SPLIT=n` | split-K matvec target warps (default 0 = off; see performance notes) |

## Source layout

```
Package.swift
Sources/husky/
  Config.swift         Qwen3.5 text config parsing
  Safetensors.swift     mmap safetensors reader + bf16/raw conversion
  Tokenizer.swift       byte-level BPE from tokenizer.json (no dependencies)
  Weights.swift         quantized tensor loading (GPU-private buffers)
  MetalEngine.swift     device/queue, runtime MSL compilation, pipeline cache
  Model.swift           the multi-kernel forward pass (batch = 1, production path)
  BatchModel.swift      batched forward pass (batch = 1/2/4/8)
  PrefillModel.swift    chunked prefill (GEMM over a chunk of prompt tokens)
  Bench.swift           warmed speed sweeps (throughput and prefill)
  HardwareBench.swift   checked MMA throughput and GPU copy bandwidth
  KernelChecks.swift    numerical kernel checks and prefill/decode handoff
  Arena.swift           packed weight+scratch+state arena for the megakernel
  Megakernel.swift      host driver for the single-dispatch kernel
  kernels.metal         the production kernels
  batchkernel.metal     batched kernels (weights read once per row for all B)
  prefillkernel.metal   chunked-prefill GEMM + conv/delta/attention chunk kernels
  megakernel.metal      the persistent single-dispatch kernel
  main.swift            CLI
docs/
  README.md                      (this file)
  architecture.md                model + engine design
  porting-log.md                 implementation + debugging trials & errors
  performance.md                 optimization journey and measured limits
  megakernel-and-memory-model.md the single-dispatch attempt and Metal's memory model
  benchmarks.md                  input/output length × batch speed results
  prefill.md                     current prefill measurements and empirical roofline
```

## Reference used for validation

Apple's `mlx_lm` (0.31.0) supports `qwen3_5` and loads the same weights, so it was
used as the golden reference (greedy decode, per-layer activation dumps).

- prompt `"The capital of France is"` → `[760, 6511, 314, 9338, 369]`
- MLX greedy → `[11751, 13, 198, 32, 13, 2912, 198, 33, 13, 3439, 198, 15666]` (`" Paris.\nA. True\nB. False\nAnswer"`)
- this engine → same first 5 tokens, then diverges on the open-ended continuation.
- `"2+2="` → `[19, 3709, 19, 10, 19, 28, 23, 3709, 23, 10, 19, 28]` — **identical, all 12**.

See `performance.md` for the full accuracy/throughput analysis.

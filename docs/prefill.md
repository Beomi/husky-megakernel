# Metal prefill status and throughput ceiling

Measured on 2026-09-23, Apple M4 Pro (20 GPU cores, 48 GB unified memory).
The current path uses packed fp16 GEMMs with bf16 activations and fp32
accumulation. Quantized weights remain available for decode and final logits.

## Results

Weights are resident, one untimed warmup precedes each case, and timings are the
median of three wall-clock runs. Reset, all prompt chunks, final logits and argmax
are included; model loading and weight packing are excluded. Same-binary kernel
reference selection uses `HUSKY_PREFILL_REFERENCE=1` (the original recurrent and
attention kernels, with the same GEMM and weights as the optimized path).

| Prompt tokens | Chunk | Reference TPS | Optimized TPS | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 256 | 522.7 | 662.0 | 1.27× |
| 256 | 256 | 509.6 | 668.9 | 1.31× |
| 512 | 256 | 462.3 | 635.3 | 1.37× |

The untouched initial checkout measured 494.4 TPS at 256 tokens and 450.4 TPS at
512 tokens in single runs. Those are less robust than the warmed comparison above.
Earlier documentation's ~490 TPS plateau was real, but its conclusion that only
GEMM tuning could improve throughput was not supported by stage measurements.

## Changes

- **Register-resident recurrence:** `gdelta_chunk_register` keeps each value
  row's 128 state elements in registers for the whole chunk. The previous kernel
  read and wrote device state every token. Per-step bf16 rounding is retained.
  Reassociation is disabled here to preserve the reference summation order.
- **Attention reduction:** `attention_chunk_simd` preserves the original
  256-element reduction tree but finishes it with SIMD shuffles. It replaces
  eight threadgroup reduction barriers with one. Reassociation is disabled only
  inside the reduction; applying it to the softmax update changed results.
- **GEMM staging:** vector shared-memory stores replace scalar stores. This by
  itself gave little improvement; recurrence and attention are the main gains.
- **Memory:** skip the unused fp16 embedding/output-head expansion, saving
  1,271,398,400 bytes (1.27 GB). Transformer fp16 projections occupy
  7,138,181,120 bytes (7.14 GB) in addition to quantized weights and caches.
- **Actual generation:** prompts of at least 64 tokens now use chunked prefill,
  then hand the convolution, recurrent and KV buffers directly to the decode
  model. `--prefill-chunk 256` explicitly enables it for shorter prompts;
  `--prefill-chunk 0` selects token-by-token prefill. Short prompts keep the
  quantized path by default to avoid fp16 packing and underfilled GEMM tiles.
- **Measurement:** working per-dispatch command-buffer timing replaces the
  crashing counter-buffer experiment. Profiling serializes submissions and is
  diagnostic only; its wall time is not a normal throughput measurement.
- **Checks:** release builds retain preconditions (removed `-Ounchecked`).
  The g/beta dispatch uses an exact token/head grid, including odd chunk sizes.

An initial SIMD-parallel recurrent reduction was faster but drifted on long
prompts; it was rejected. Checking only the next token would have missed this.

## Stage measurements

Isolated command-buffer GPU timing for a 256-token chunk (all 32 layers):

| Stage | Before, ms | After, ms |
|---|---:|---:|
| GEMMs | 321.3 | 319.5 |
| Recurrent updates | 117.3 | 28.4 |
| Full attention | 53.1 | 21.8 |

The major gains come from recurrence and attention. These serialized diagnostic
timings explain the bottlenecks but should not be substituted for normal wall-time
benchmarks. An additional experiment aliasing GEMM accumulator scratch with the
A staging buffer reduced shared memory from 20 to 12 KiB but regressed throughput;
it was reverted.

## Hardware and the TPS calculation

Apple specifies **273 GB/s** memory bandwidth for this M4 Pro configuration
([Apple specifications](https://support.apple.com/en-ie/121553)). Metal reports a
32 KiB threadgroup-memory limit. The checked local microbenchmarks measured:

| Workload | Sustained measurement |
|---|---:|
| 256 MiB GPU copy, counting reads + writes | 215.8 GB/s |
| fp16 matrix operands, fp32 accumulation, four distinct chains | 3.44–3.56 TMAC/s |
| Same matrix rate in FLOPs (2 FLOPs/MAC) | 6.88–7.12 TFLOP/s |

The MMA benchmark consumes changing device-buffer operands, stores all four
results, and checks them against CPU matrix multiplication. It sweeps grid size
and reports warmed medians. These are **measured sustained rates**, not proof of
an absolute silicon maximum. Apple does not give an applicable MMA peak in the
cited specifications. The old notes' 9.5–14 TMAC/s figures were not reproduced by
this checked benchmark, and neither the old 830 TPS hard cap nor a >2000 TPS
ceiling should be treated as established hardware facts.

Count the actual transformer projection shapes, excluding the tied embedding:

```text
P = sum(outN * inN for transformer projections) = 3,569,090,560 MAC/token
H = vocabSize * hiddenSize                     =   635,699,200 MAC/prompt
projection FLOPs/token                         = 2 * P = 7.138181120 GFLOP
compute-only TPS estimate                     = 3.56e12 / P ≈ 997 TPS
```

The output head is evaluated once, at the last prompt token; charging it to every
prefill token incorrectly gives ~4.2 GMAC/token. For prompt length L, its amortized
cost is H/L. Causal full attention adds approximately
`8 layers * 4096 head-elements * (L+1)` MAC/token; the recurrent updates add about
38 MMAC/token plus rounding, normalization, nonlinearities and data movement.
They lower the end-to-end rate further. A useful compute-only estimate on this
machine is therefore **roughly 1,000 TPS**, with the current measured kernel path
reaching about **635–669 TPS** for 256–512-token prompts. This is an empirical
roofline estimate, not a guaranteed or universal maximum.

For 256 tokens, an ideal single read of 7.138 GB fp16 weights gives a bandwidth
bound of `273e9 * 256 / 7.138e9 ≈ 9,790 TPS`. The actual 64-row GEMM tile loads
weights four times per 256-token chunk before considering cache reuse, giving
~2,448 TPS using advertised bandwidth, or ~1,935 TPS using the measured copy
rate. Both exceed the compute estimate. Cache behavior and mixed read/write
traffic mean these bandwidth calculations are optimistic models, not timings.

## Correctness and reproduction

`--check-prefill` compares every final-logit element with the retained reference
kernels at prompt lengths 1, 5, 63, 64, 65, 129, 257, 256 and 512, including
64- and 256-token chunks. The tested logits are **bit-identical**. It also checks
prefill-to-decode handoff and the 12-token `2+2=` golden continuation with chunk 3.
This validates equivalence to the existing engine, not universal MLX equivalence.

`--check-kernels` checks recurrent outputs and final states, attention with a
nonzero cache prefix, and packed GEMM against a CPU reference, including partial
64-row tiles. It also passes with Metal shader validation enabled.

```sh
swift build -c release
.build/release/husky --bench-hardware
MTL_SHADER_VALIDATION=1 .build/release/husky --check-kernels
.build/release/husky --model /tmp/woof --check-prefill
HUSKY_CHUNKS=256 HUSKY_LENGTHS=128,256,512 .build/release/husky --bench-prefill
HUSKY_PREFILL_REFERENCE=1 HUSKY_CHUNKS=256 HUSKY_LENGTHS=128,256,512 .build/release/husky --bench-prefill
HUSKY_PREFILL_PROF=1 HUSKY_REPEATS=1 HUSKY_CHUNKS=256 HUSKY_LENGTHS=256 .build/release/husky --bench-prefill
```

`HUSKY_REPEATS` defaults to 3. The model-free hardware and kernel checks do not
require `/tmp/woof`. Raw runs are in `docs/measurements/2026-09-23/`.

# Husky

A native Swift + Metal inference engine for
`ConwayResearch/Underdog-Woof-4B-1.1`, a 4-bit hybrid language model. Husky runs
greedy text generation on Apple silicon, with chunked prefill and quantized
decode. Inference requires neither Python nor MLX.

## Requirements

- An Apple silicon Mac running macOS 14 or later.
- An installed Xcode/Swift toolchain with Metal support.
- The Woof model files, downloaded to a local directory.

Fast prefill prepares an additional **7.14 GB of fp16 weights**, alongside the
quantized weights, activations, and caches. The benchmarks below used a Mac with
48 GB of unified memory.

## Quick start

From the repository root, build the executable:

```sh
swift build -c release
```

If the weights are not already available, download them using the Hugging Face
CLI (`hf` must be installed separately):

```sh
hf download ConwayResearch/Underdog-Woof-4B-1.1 --local-dir /tmp/woof
```

Generate text:

```sh
.build/release/husky \
  --model /tmp/woof \
  --prompt "Explain how transformer language models work." \
  --max-tokens 128
```

The executable prints generated token IDs and text, plus timing information.
`--max-tokens` controls the number of generated tokens. The final end-to-end
timing includes weight loading, prefill, and decode; the separate prefill timing
excludes weight loading.

To use a longer prompt saved in `prompt.txt`:

```sh
.build/release/husky \
  --model /tmp/woof \
  --prompt "$(cat prompt.txt)" \
  --max-tokens 256
```

## Prefill and generation options

Normal generation automatically uses **256-token prefill chunks for prompts of
at least 64 tokens**, then passes the populated caches directly to decode.
Shorter prompts use token-by-token prefill to avoid weight preparation and
underfilled matrix tiles.

| Option | Behavior |
|---|---|
| `--model PATH` | Model directory; defaults to `/tmp/woof`. |
| `--prompt TEXT` | Text to continue. |
| `--ids "760,6511,314,9338,369"` | Explicit input token IDs; overrides `--prompt`. |
| `--max-tokens N` | Maximum generated tokens; defaults to 16. |
| `--max-seq N` | Cache capacity in tokens; defaults to 4096. Allow room for the prompt and continuation. |
| `--prefill-chunk N` | Explicitly enable chunked prefill with chunk size N, including for short prompts. |
| `--prefill-chunk 0` | Use token-by-token prefill. |

The normal generation path is recommended. The separate `--megakernel` option
is an experimental single-dispatch implementation and remains much slower.
Only `--groups 1` is correct; multiple groups have unresolved synchronization
issues.

## Measured performance

Apple M4 Pro, 20 GPU cores, 48 GB memory; measured on September 23, 2026.
Prefill figures are medians of three warmed runs with weights resident, using
256-token chunks. Loading and weight preparation are excluded.

| Prompt length | Before optimization | Current prefill |
|---:|---:|---:|
| 128 tokens | 523 tokens/s | 662 tokens/s |
| 256 tokens | 510 tokens/s | 669 tokens/s |
| 512 tokens | 462 tokens/s | 635 tokens/s |

The checked matrix benchmark measured up to 3.56 TMAC/s. The model's transformer
projections require 3.569 GMAC per prompt token, giving an estimated
**~1,000 tokens/s projection-only ceiling** at that measured rate. Attention,
recurrence, staging, and output logits reduce end-to-end throughput. This is an
empirical estimate, not a proven absolute hardware maximum.

See the [prefill report](docs/prefill.md) for calculations, validation, and
[raw measurements](docs/measurements/2026-09-23/README.md).

## Benchmarks and checks

```sh
# Hardware throughput; no model files required
.build/release/husky --bench-hardware

# Prefill throughput sweep
.build/release/husky --model /tmp/woof --bench-prefill

# Numerical kernel checks; no model files required
MTL_SHADER_VALIDATION=1 .build/release/husky --check-kernels

# Prefill logits, chunk boundaries, and decode continuation
.build/release/husky --model /tmp/woof --check-prefill

# Experimental single-threadgroup megakernel benchmark
.build/release/husky --model /tmp/woof --bench-megakernel
```

To narrow the prefill sweep:

```sh
HUSKY_CHUNKS=256 HUSKY_LENGTHS=128,256,512 HUSKY_REPEATS=3 \
  .build/release/husky --model /tmp/woof --bench-prefill
```

Add `HUSKY_PREFILL_REFERENCE=1` to compare against the retained original
recurrence and attention kernels. `HUSKY_PREFILL_PROF=1` prints isolated stage
timings, but serializes submissions and is unsuitable for normal TPS comparisons.

## Documentation

- [Engine overview and source layout](docs/README.md)
- [Architecture](docs/architecture.md)
- [Prefill optimization and hardware estimates](docs/prefill.md)
- [Decode performance and reference accuracy](docs/performance.md)
- [Megakernel design and synchronization limits](docs/megakernel-and-memory-model.md)
- [Historical batched token-by-token benchmarks](docs/benchmarks.md)

The optimized prefill matches the retained reference logits exactly on the
documented integration cases. The `2+2=` continuation matches the 12-token golden
sequence. Open-ended generation can diverge from MLX because of floating-point
accumulation differences; see the accuracy notes above.

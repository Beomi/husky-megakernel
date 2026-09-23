# The single-dispatch megakernel and Metal's memory model

## Current measurement (2026-09-23)

The correct single-threadgroup path was optimized with a warp-per-output
quantized matvec (coalesced 128-bit reads, four independent accumulators) and a
single device-memory threadgroup barrier at stage boundaries, avoiding the
unnecessary atomic counter when `ngroups == 1`.

| Version | GPU ms/token | GPU TPS |
|---|---:|---:|
| Original | 224.642 | 4.452 |
| Optimized | 213.796 | 4.677 |

This is **5.1% higher throughput**, measured as the median of three warmed
15-step runs (four prompt tokens plus eleven continuation forwards). All twelve
`2+2=` output tokens match the golden sequence. Host weight loading is excluded.
`husky --bench-megakernel` reproduces the measurement. Trials with
`HUSKY_MEGAKERNEL_TG=256` and `512` were slower; 1024 remains the default.

This prototype still runs on one threadgroup for correct results. The change
makes no claim to solve the cross-threadgroup synchronization problem described
below. Chunked prefill is a separate multi-dispatch path, currently **635–669
TPS**; it is enabled in normal generation for prompts of at least 64 tokens.
See [prefill.md](prefill.md) for hardware calculations and validation.

The following sections are the historical design and synchronization experiments.


The blog's direction is a "whole-step megakernel": one persistent dispatch that
runs all 32 layers against a static tile schedule, meeting at **device-side
barriers** between stages. The point is to remove host/launch overhead and let the
next step's weight loads overlap the current step's tail.

This document records the megakernel prototype in this repo
(`Arena.swift`, `Megakernel.swift`, `megakernel.metal`) and, more importantly, the
hardware/toolchain experiments that show why a correct all-core version is not
possible on this Apple GPU with this Metal toolchain.

## What was built

**One packed arena.** Every weight, scratch activation, per-layer recurrent/conv
state, KV cache, the barrier counter, and the output token live in a single
`MTLBuffer`, addressed by byte offsets. A per-layer offset table (stride 32 u32)
points at each layer's tensors. A flat `cfg` array carries dimensions and scratch
offsets. The kernel reads the input token and position from the arena.

**One kernel.** `husky_step` is a persistent kernel that runs, for one token:
embedding lookup → 32 layers (gated-delta or full attention + MLP) → final norm →
the 248k-way logits matvec → **in-kernel argmax** → writes the next token id.
Stage boundaries are a `gbar()` device-wide barrier. Generation is one dispatch
per token (one `husky_step` per generated token, since decoding is sequential).

**Result:** with a single threadgroup (`--groups 1`) it is **correct** — coherent
output, `2+2=` matches MLX, `The capital of France is` → ` Paris.` — but it runs on
one GPU core (~3.4 tok/s). To use all cores, multiple threadgroups need a correct
grid-wide barrier. That is where it fails.

## The barrier experiments

All experiments used a minimal two-kernel probe: a producer writes a value to
device memory, a grid-wide barrier runs, a consumer reads it; the host checks the
consumer's read over hundreds of trials.

| experiment | failures |
|---|---|
| shared RMW counter barrier (`atomic_fetch_add`, spin until `count == nGroups`) | 189/200 |
| same, data accessed through `volatile` | 194/200 |
| same, consumer address depends on the atomic load | 190/200 |
| producer writes data via `atomic_store`, consumer reads via `atomic_load` | **0/200** |
| producer-only sentinel (separate buffer) | 0/300 |
| all-to-all sentinel barrier, 8 groups, each verifies every peer | **stale reads on 5–7 cells, every trial** |

### Two false leads

1. **Shared RMW counter is logically unsound.** A group increments the counter and
   spins until `count == nGroups`; but its *own* increment can be the one that
   reaches `nGroups`, so it proceeds without ever observing another group's write.
   The ~190/200 failures are as much a logic bug as a memory-ordering bug.
2. **"Atomic store in the same cache line" looked promising, then was invalid.** A
   probe where an atomic store made a nearby normal write visible returned 0/200 —
   but both threadgroups were writing the sentinel, so the consumer satisfied its
   own wait. Re-running with a producer-only sentinel removed the "magic"; the
   all-to-all version then exposed stale reads.

### The decisive result

In the all-to-all test, the kernel's own verification read stale peer values
(`out = 105..107`) on nearly every trial, **while the host read the correct final
values after completion**. That is the signature of non-coherent normal stores:
the writes land eventually, but a peer threadgroup reading after the barrier does
not necessarily see them. Only locations accessed through **atomics** are coherent
across threadgroups.

### Why it can't be fixed with fences

The MSL `<metal_types>` memory-order enum gates `acquire`, `release`, and
`acq_rel` behind `__HAVE_PARTIAL_ORDER_ATOMIC__`, and `seq_cst` plus
`atomic_thread_fence` behind `__HAVE_ATOMIC_FENCE__`. On this toolchain **none of
those macros are defined** for any `MTLLanguageVersion` (2.4, 3.0, 3.1, 3.2 all
fail to compile a source that references them), and defining
`__HAVE_PARTIAL_ORDER_ATOMIC__` manually does not unlock the identifiers
(the header is force-included). So:

- `memory_order_relaxed` is the only available ordering;
- `atomic_thread_fence`, `__threadfence`, and `mem_fence` do not exist;
- `threadgroup_barrier(mem_flags::mem_device)` synchronizes *within* a
  threadgroup only, and does not make one group's normal stores visible to
  another.

Without acquire/release or a device fence there is no way to publish a
threadgroup's activation writes to its peers, so a correct grid-wide barrier — the
prerequisite for an all-core single dispatch — cannot be built here.

## Why this is consistent with the source material

The blog already flagged this: *"We built the first piece, a layer's whole MLP as
one persistent dispatch with a grid barrier, and measured it honestly: identical
output, and no faster. On this GPU, dispatches placed back to back in one encoder
already run with almost no gap, so a megakernel only pays where it can overlap the
next stage's weight loads with the current stage's tail."*

Our own measurements agree that dispatch overhead is small (~1 µs/dispatch; host
encode ~0.25 ms/token), so a megakernel's upside is limited. The prototype is kept
in the tree as a correctness demonstration (one dispatch runs the whole model) and
as a record of the toolchain limitation. `--groups > 1` still runs but prints a
warning that results are racy.

## If this were to be pursued further

Options, in rough order of feasibility:

1. **Atomic data path.** Make all cross-threadgroup activation traffic go through
   `atomic_load`/`atomic_store` (e.g. activations stored as packed `uint`).
   Correct, but likely slower and a large rewrite.
2. **Tile-by-tile dual dispatch.** Keep parallelism but fuse independent stages
   into single dispatches and use `MTLComputeCommandEncoder.memoryBarrier` between
   dependent ones. Correct and simple, but the win is small given dispatch cost.
3. **A toolchain/OS where `__HAVE_PARTIAL_ORDER_ATOMIC__` is defined**, which would
   allow a real device-scope acquire/release barrier.

# Local measurements, 2026-09-23

Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory. Model:
`ConwayResearch/Underdog-Woof-4B-1.1`, local `/tmp/woof`.

Commands and interpretation: [prefill](../../prefill.md) and
[megakernel](../../megakernel-and-memory-model.md).

`prefill-original.txt` is the untouched source's single-run sweep.
The reference and optimized warmed files use the final binary, one warmup and
three timed runs per case; the reference selects the original recurrent and
attention kernels with `HUSKY_PREFILL_REFERENCE=1`.

The original megakernel was measured from a saved source copy using the same
benchmark harness and normal release optimization. Each entry is a median of
three warmed 15-forward runs, using GPU command-buffer timestamps.

Stage files use serialized command buffers. Their TPS is perturbed by profiling;
use the warmed prefill files for end-to-end throughput.

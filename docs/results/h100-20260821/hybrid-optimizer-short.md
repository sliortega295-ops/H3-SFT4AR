# H100 hybrid optimizer short gate — 2026-08-21

## Outcome

The first Nsight-guided candidate with enough memory headroom to finish a
window completed five warm-up and five measured steps at 43.2914 seconds/step.
Against the two native 30-step baselines, this is 1.2718x-1.2827x throughput,
or a 21.37%-22.04% reduction in step time.

This is the first large acceleration signal in this experiment, but the current
configuration is not ready for a long run because its observed HBM margin is
only about 568 MiB and DeepSpeed performed two allocator cache flushes.

## Successful candidate

The source, data, model, world size, BF16 mode, full-DiT training, gradient
checkpointing, optimizer order, and measurement contract match the baseline.
The candidate changes only the ZeRO/allocator substrate:

- ZeRO stage 3 remains enabled.
- 50% of optimizer subgroups are assigned to CPU offload; the rest use GPU
  AdamW through DeepSpeed's hybrid optimizer path.
- optimizer `sub_group_size`: 100M elements;
- AllGather and ReduceScatter bucket sizes: 50M elements;
- parameter prefetch bucket: 50M elements;
- `PYTORCH_ALLOC_CONF=expandable_segments:True`;
- five warm-up steps followed by five measured steps.

The exact committed configs are
[`accelerate_config_zero3_hybrid_optimizer_050.yaml`](../../../experiments/accelerate_config_zero3_hybrid_optimizer_050.yaml)
and
[`deepspeed_zero3_hybrid_optimizer_050.json`](../../../experiments/deepspeed_zero3_hybrid_optimizer_050.json).

## Native timing result

| quantity | value |
|---|---:|
| max-rank elapsed time | 216.457152847 s |
| mean-rank elapsed time | 216.457143522 s |
| measured steps / rank | 5 |
| seconds / step | 43.291430569 s |
| global samples / second | 0.1847940781 |

A normalized copy of the runner output is committed as
[`hybrid-optimizer-050-benchmark.json`](hybrid-optimizer-050-benchmark.json).
Its values are unchanged; the committed copy adds a trailing newline.

| reference | seconds / step | throughput speedup | step-time reduction |
|---|---:|---:|---:|
| baseline trial 1 | 55.0566 | 1.2717667x | 21.3692% |
| baseline trial 2 | 55.5311 | 1.2827273x | 22.0411% |
| baseline Nsight window | 55.2803 | 1.2769344x | 21.6874% |

The first allocator cache flush occurred after the first warm-up step; the
second occurred after measured step four. The measured result already includes
the latter event.

## Memory feasibility gates

Three more aggressive candidates were rejected before timing claims:

1. Full GPU-resident optimizer without expandable segments failed during the
   first forward AllGather while requesting 222 MiB with only 153 MiB free.
2. Full GPU-resident optimizer with expandable segments completed one step in
   39.96 seconds, then failed in the next forward while requesting 318 MiB with
   78.91 GiB already in use. The 39.96-second value is a feasibility diagnostic,
   not a benchmark result.
3. A 25% offload candidate with 500M communication buckets failed in its first
   forward while requesting 294 MiB with 78.82 GiB in use.

The successful 50% candidate was observed at up to 80,991 MiB on an 81,559 MiB
device during periodic `nvidia-smi` samples. This is not an automatically
sampled peak, but it is sufficient to show that the present margin is unsafe
for a production-length run.

## Decision

The baseline bottleneck diagnosis is validated by intervention: moving part of
the CPU-offloaded optimizer work onto H100 and reducing the memory-heavy
communication buckets produced a large end-to-end improvement. The next
iteration should preserve this path while adding several GiB of HBM margin,
then repeat a paired 5-10 step A/B. It should not return to data-loader tuning
or profile the old no-speedup candidates.

Remote artifact integrity:

- `benchmark.json`: 476 bytes, SHA-256
  `cf781766299e851e9b5bcc150ed9a9093d8180e829e4e5e91226cd4f85fa2ae8`
- committed normalized benchmark JSON: SHA-256
  `818091c1207dbddd0cefd9997d668cb9bb67bedce21b7cbc1582e4c8f7b5ed64`
- `train.log`: 34,883 bytes, SHA-256
  `f945ca43871813ed25e0696dc3c58e5cc5370e99be9530451692640409518c68`

# H100 Ref2VA A/B results — 2026-08-21

## Status

This snapshot contains seven complete 30-step 8-GPU H100 benchmark windows.
Those initial variants do not establish a speedup: their measured effects are
small and change across trials. A later Nsight-guided 5-step window found a
large preliminary hybrid-optimizer signal; it is reported separately because
its window is shorter and its HBM margin is very small.

The second-trial `optimized-zero` run was intentionally stopped after 5 warm-up
steps and 12 measured steps when the experiment owner requested shorter windows.
The stop request was issued after the 10th measured step; two more steps completed
while the owned launcher and accelerator processes were being terminated.
The current runner writes `benchmark.json` only after the configured window
finishes, so that partial run is recorded as `PARTIAL_NOT_REPORTED` and is not
used below.

A baseline-only Nsight Systems capture was subsequently authorized to locate
the acceleration bound despite the small BF16 self-repeat drift. It covers two
steps after five warm-up steps and is diagnostic-only; see
[`nsight-baseline.md`](nsight-baseline.md).

## Bound contract

- H3-SFT4AR source commit: `bdc3cd8fa2a71f01f4c70bf86d091668712baf87`
- Pinned DiffSynth-Studio commit: `6343deda06b3e09efc9b1ce23c135c35a341d143`
- Hardware: 8 x NVIDIA H100 80GB HBM3
- World size: 8; one sample per rank and optimizer step
- Complete windows: 5 warm-up steps followed by 30 measured steps
- Primary metric: `30 * world_size / max_rank_elapsed_seconds`
- Window includes data wait, forward, backward, optimizer, scheduler,
  `zero_grad`, and logging; it has one synchronization before and after the
  full window and no per-phase synchronization.
- Same Ref2VA DiT checkpoint, preprocessed CSV/cache set, seed, learning rate,
  full-DiT training, gradient-checkpoint boundary, and optimizer update order.

## Results

| trial | mode | max-rank seconds | seconds / step | global samples / second | vs same-trial baseline |
|---:|---|---:|---:|---:|---:|
| 1 | baseline | 1651.6982 | 55.0566 | 0.145305 | 1.0000x |
| 1 | optimized-core | 1671.2456 | 55.7082 | 0.143605 | 0.9883x |
| 1 | optimized-data | 1674.8382 | 55.8279 | 0.143297 | 0.9862x |
| 1 | optimized-zero | 1664.5509 | 55.4850 | 0.144183 | 0.9923x |
| 2 | baseline | 1665.9319 | 55.5311 | 0.144064 | 1.0000x |
| 2 | optimized-core | 1671.7591 | 55.7253 | 0.143561 | 0.9965x |
| 2 | optimized-data | 1653.8933 | 55.1298 | 0.145112 | 1.0073x |
| 2 | optimized-zero | — | — | — | `PARTIAL_NOT_REPORTED` |

`optimized-core` is slower in both complete trials. `optimized-data` changes
from -1.38% in trial 1 to +0.73% in trial 2. These data therefore do not support
a stable acceleration claim. The observed differences are also small enough
that additional short repetitions should report dispersion, not only a point
estimate.

## Nsight-guided short follow-up

The baseline profile localized 87.46% of the step to backward/ZeRO update and
showed 38.3% of aggregate GPU-kernel duration in NCCL AllGather and
ReduceScatter. A memory-bounded hybrid ZeRO-3 candidate then completed five
warm-up and five measured steps:

| mode | measured steps | seconds / step | global samples / second | throughput vs baseline trial 1 | throughput vs baseline trial 2 |
|---|---:|---:|---:|---:|---:|
| baseline + 50% optimizer offload + 50M buckets | 5 | 43.2914 | 0.184794 | 1.2718x | 1.2827x |

This corresponds to 21.37%-22.04% lower step time, or 27.18%-28.27% higher
throughput, than the two 30-step baselines. It is a strong acceleration signal,
not yet a final production claim: the current window is short, no fresh paired
baseline was run immediately before it, two allocator cache flushes occurred,
and sampled HBM use reached 80,991 of 81,559 MiB. See
[`hybrid-optimizer-short.md`](hybrid-optimizer-short.md) for the exact contract
and failed feasibility gates.

## Preparation and validation performed on the H100 node

- `scripts/run_checks.sh` passed after reconstructing both source trees and
  checking the 17 optimized-path Git blob hashes.
- An 8-rank CUDA/NCCL/BF16 distributed smoke test passed.
- All 13 official Ref2VA safetensor shards and their config/index were verified
  against a SHA-256 manifest.
- The CSV referenced 24/24 available preprocessed cache files. A fixed-seed
  dataset comparison matched baseline and optimized chunk/reference video,
  audio, and text tensors for all eight usable rows.
- Raw videos, a text encoder, and video/audio VAEs were not loaded because this
  benchmark uses preprocessed embeddings and latents.

## Limits of this snapshot

- The repository's current `run_checks.sh` is a reconstruction/static hash gate;
  it does not reproduce the README's historical `7 passed` semantic test claim.
- The benchmark runner does not emit per-step loss, gradient norm, or parameter
  checksums. This snapshot therefore does not claim GPU numerical equivalence.
- Per-process CPU RSS and an automatically sampled GPU-memory peak were not
  captured by the committed runner.
- The Nsight trace supports phase and aggregate kernel-duration attribution,
  but it is not a throughput result and contains no hardware-counter claims.

The raw machine-readable reports are stored in `trial-1/*/benchmark.json` and
`trial-2/*/benchmark.json`.

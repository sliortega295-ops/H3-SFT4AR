# H100 Ref2VA A/B results — 2026-08-21

## Status

This snapshot contains seven complete 8-GPU H100 benchmark windows. It does not
claim a speedup: the measured effects are small and change across trials.

The second-trial `optimized-zero` run was intentionally stopped after 5 warm-up
steps and 10 measured steps when the experiment owner requested shorter windows.
The current runner writes `benchmark.json` only after the configured window
finishes, so that partial run is recorded as `PARTIAL_NOT_REPORTED` and is not
used below.

Nsight profiling was skipped because no repeatable positive throughput signal
had been established.

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
- No Nsight trace or performance-bound attribution is included.

The raw machine-readable reports are stored in `trial-1/*/benchmark.json` and
`trial-2/*/benchmark.json`.

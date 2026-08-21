# H100 baseline correctness gate — 2026-08-21

## Decision

The exact baseline self-repeat gate failed. The optimized-zero correctness run
and Nsight capture were therefore not run. This is a fail-closed correctness
result, not a performance attribution.

## Three-step baseline repeat (v4)

Two fresh 8-rank baseline processes used the same three samples and chunks,
seed, checkpoint, BF16 + ZeRO-3 configuration, and optimizer settings.

- All 24 rank/step sample identities matched.
- At step 0, all eight rank losses and all 16 video/audio prediction sampled
  hashes matched exactly.
- The strict comparator reported 719 mismatched fields after the first combined
  DeepSpeed backward/update boundary.
- By step 1, every rank's loss and prediction hashes differed. The maximum loss
  difference was 0.0171595812 (1.5301% relative).
- By step 2, the maximum loss difference was 0.0300844908 (2.0570% relative).

The v4 diagnostic used `accelerator.backward(loss)`. Under Accelerate 1.12 with
DeepSpeed, that call executes both `engine.backward()` and `engine.step()`.
Consequently, the v4 fields named `parameter_before`, `gradient`, and
`parameter_after` do not isolate backward from update. They only localize the
first divergence to the combined backward/update boundary.

## Corrected one-step boundary diagnostic

The runner now samples the same three largest trainable local FP32 shards at
four explicit points:

1. step start;
2. after `engine.backward()` and before `engine.step()`;
3. the corresponding gradient before `engine.step()`;
4. after `engine.step()`.

The gate compares two fresh one-step baseline processes before it permits an
optimized-zero launch. The optimized run is marked `not_run` when the baseline
self-repeat fails.

The corrected one-step run completed with the following exact counts:

- sample identities: 8/8 equal;
- loss values: 8/8 equal;
- video/audio prediction sampled hashes: 16/16 equal;
- parameter-at-step-start sampled hashes: 24/24 equal;
- parameter-before-update sampled hashes: 24/24 equal;
- gradient-before-update sampled hashes: 0/24 equal;
- parameter-after-update sampled hashes: 0/24 equal.

The strict comparator reported 108 mismatched fields. This establishes that the
first observed divergence is in backward/gradient generation, before the
optimizer update. The current evidence does not separate PyTorch checkpoint
recomputation/SDPA backward from ZeRO-3 gradient reduction; both remain in the
candidate boundary.

## Runtime and source audit

- PyTorch 2.9.1+cu128; host CUDA toolkit warning reports 12.4.
- DeepSpeed 0.19.5; Accelerate 1.12.0.
- BF16, ZeRO stage 3, gradient accumulation 1, optimizer CPU offload, no
  parameter offload.
- External FlashAttention packages are unavailable. DiffSynth selects its
  `torch` attention path, which calls PyTorch scaled-dot-product attention and
  leaves backend selection to PyTorch.
- Gradient checkpointing is enabled. DeepSpeed activation checkpointing is not
  configured, so DiffSynth uses PyTorch non-reentrant checkpointing.
- Seeds are set per rank, but deterministic algorithms are disabled;
  `CUBLAS_WORKSPACE_CONFIG`, fixed NCCL algorithm/protocol, and a forced SDPA
  backend are not configured.

## What the benchmark currently supports

The native 30-step windows remain the only performance measurements:

- baseline: 55.0566 and 55.5311 seconds/step;
- optimized-core: 55.7082 and 55.7253 seconds/step;
- optimized-data: 55.8279 and 55.1298 seconds/step;
- optimized-zero: 55.4850 seconds/step in the one complete trial.

These runs do not show a stable speedup. They rule out the existing scalar-sync,
RoPE reuse, packing/cache, and data-worker changes as material end-to-end wins
under this contract. They do not yet prove which kernels dominate. The leading
source-backed hypothesis is that the full 50-block DiT forward/backward,
checkpoint recomputation, and ZeRO-3 communication/update path dominate the
step, but that attribution requires a valid baseline-only timeline.

Nsight remains blocked until the baseline correctness policy is resolved. A
timeline, if later authorized, is diagnostic-only and must not replace the
native throughput numbers.

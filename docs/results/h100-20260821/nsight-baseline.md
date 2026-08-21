# H100 baseline Nsight diagnosis — 2026-08-21

## Outcome

The baseline is not data-bound. In the captured steady-state window, 87.46% of
the NVTX train-step duration is inside the combined backward/ZeRO update path.
That path is almost evenly split between DeepSpeed backward and the ZeRO-3
step. At the GPU-kernel level, NCCL collectives and FlashAttention together
account for about 77.4% of aggregate kernel duration.

This is a bottleneck diagnosis, not a speedup result. The native 30-step
windows in [`README.md`](README.md) remain the throughput source of truth.
Exact extracted values are also recorded in
[`nsight-baseline-summary.json`](nsight-baseline-summary.json).

## Capture contract

- Mode: unmodified `baseline`
- Hardware: 8 x NVIDIA H100 80GB HBM3
- Warm-up: 5 optimizer steps
- Captured window: 2 optimizer steps
- Nsight Systems: 2024.1.1.0
- Trace domains: `cuda,nvtx,osrt,cublas,cudnn`
- CPU sampling and context-switch tracing: disabled
- Capture range: CUDA profiler API, started after warm-up
- Correctness capture: disabled

Nsight Systems 2024.1 on this node does not accept `nccl` as a trace domain.
NCCL device kernels are nevertheless present in the CUDA trace. The capture
range started and ended normally and the report parses successfully. The
distributed launcher later hung during teardown, so the runner did not emit a
`benchmark.json`; after validating the report, only the owned residual ranks
were terminated.

## NVTX phase summary

The table reports the average over 16 NVTX instances: 8 ranks times 2 captured
steps. Nested ranges must not be added together.

| NVTX range | average / rank-step | share of `H3/train_step` |
|---|---:|---:|
| `H3/train_step` | 55.2803 s | 100.00% |
| `H3/forward` | 6.9322 s | 12.54% |
| `H3/backward` | 48.3477 s | 87.46% |
| `DeepSpeedEngine.backward` | 25.1312 s | 45.46% |
| `DeepSpeedZeroOptimizer_Stage3.step` | 23.2159 s | 42.00% |
| `H3/data_wait` | 0.0328 s | 0.06% |

`H3/optimizer` is effectively zero because Accelerate with DeepSpeed executes
both `engine.backward()` and `engine.step()` inside `accelerator.backward()`;
the later optimizer call is a no-op under this integration.

DeepSpeed's nested ranges also show frequent parameter materialization and
partition movement. Their durations overlap parent ranges and are evidence of
where ZeRO work occurs, not additional wall time:

- `PartitionedParameterCoordinator.fetch_sub_module`: 36,480 calls and 117.920
  aggregate seconds, or 7.370 aggregate seconds per rank-step.
- `_reassign_or_swap_out_partitioned_parameters`: 80 calls and 107.602
  aggregate seconds, or 6.725 aggregate seconds per rank-step.
- `unscale_and_clip_grads`: 80 calls and 26.093 aggregate seconds, or 1.631
  aggregate seconds per rank-step.

## CUDA kernel summary

These percentages are shares of summed GPU-kernel durations across all eight
devices, not shares of train-step wall time. Concurrent kernels may overlap.

| kernel family | aggregate kernel-duration share |
|---|---:|
| NCCL ring AllGather | 28.7% |
| NCCL ring BF16 ReduceScatter | 9.6% |
| FlashAttention backward | 23.4% |
| FlashAttention forward | 15.7% |
| top four combined | 77.4% |

The FlashAttention forward kernels can include gradient-checkpoint
recomputation; this summary groups by kernel name and does not assign every
instance to an original-forward or recompute phase.

CUDA memory-operation durations are also dominated by transfers: host-to-device
is 61.2% (20.172 aggregate seconds), device-to-host is 28.4% (9.348 aggregate
seconds), and device-to-device is 10.3% (3.388 aggregate seconds). Because the
dataset wait is only 0.033 seconds per rank-step and the run uses ZeRO-3 with
optimizer CPU offload, these transfers support the optimizer/offload diagnosis;
they are not evidence of an input-pipeline bottleneck.

## Acceleration priority

1. Target ZeRO-3 materialization, collective communication, and CPU-offloaded
   optimizer movement first. This is the clearest non-model-compute bound.
2. Target backward attention/checkpoint recomputation second. Selective
   checkpointing should be evaluated against peak memory before changing it.
3. Do not spend the next iteration on data-loader tuning under this contract;
   its measured share is negligible.

Every candidate still needs a short native 5-10 step A/B confirmation after
warm-up. Nsight is used to choose the candidate, not to claim its speedup.

The first follow-up is documented in
[`hybrid-optimizer-short.md`](hybrid-optimizer-short.md). A memory-bounded 50%
optimizer-offload candidate completed a native 5-step window at 43.2914
seconds/step, versus 55.0566 and 55.5311 seconds/step for the prior baselines.

## Artifact integrity

- Report: `results/nsys-baseline-20260821-short-v1/baseline.nsys-rep`
- Report size: 131,115,827 bytes
- SHA-256:
  `3b871fdbed582a331b1bef9fc1a2f44bc8eb3d9e0628b9db08c87607dd361ae2`
- Exported SQLite size: 355,188,736 bytes

The raw binary artifacts stay on the H100 node and are not committed to Git.

# MiniMax-H3 Ref2VA 语义等价加速实验

## 0. 第一次使用

```bash
git clone https://github.com/sliortega295-ops/H3-SFT4AR.git
cd H3-SFT4AR
cp config.example.env config.env
# 编辑 CSV_PATH 与 DIT_MODEL_PATH
bash scripts/run_checks.sh
bash run_ab.sh
```

`bootstrap.sh` 会自动从固定 DiffSynth commit 重建完整 baseline，并从同一个 baseline 生成完整 optimized；不需要手工复制 DiffSynth 依赖。

若训练集群不能访问 GitHub，可在 `config.env` 中设置：

```bash
DIFFSYNTH_SOURCE=/path/to/local/DiffSynth-Studio
```

该 checkout 只需要包含固定 commit `6343deda06b3e09efc9b1ce23c135c35a341d143`。

## 1. 主 A/B：累计消融

默认 `bash run_ab.sh` 依次运行：

| mode | ZeRO | workers/data cache | 目的 |
|---|---|---|---|
| `baseline` | 原配置 | 原始 workers=0 | 原 Ref2VA；只加入测量 instrumentation |
| `optimized-core` | 原配置 | 全关 | 只测 host-sync、timestep、packed metadata、RoPE 热路径调度 |
| `optimized-data` | 原配置 | mmap/pin/prefetch/cache 开 | 再测数据与 H2D overlap |
| `optimized-zero` | overlap 候选 | 同 optimized-data | 再测 ZeRO-3 通信调度 |

四组都保持完整 Ref2VA/target token、attention 语义、loss、full-DiT trainable 参数、gradient-checkpoint 边界、optimizer 与更新顺序不变。

默认：5 step warm-up + 30 step measurement，测量窗口内不保存 checkpoint。主指标写入各目录的 `benchmark.json`：

```text
samples_per_second_global = measured_steps * world_size / max_rank_elapsed
```

即以最慢 rank 为准。

结果最终汇总到：

```text
results/ab-*/summary.md
```

## 2. 正式性能数据

至少重复 3 次，最好 5 次：

```bash
for trial in 1 2 3; do
  RESULT_ROOT="$PWD/results/trial-${trial}" bash run_ab.sh
done
```

报告中位数，并同时保存 max-rank s/step、global samples/s、GPU peak、CPU RSS。初始化、首次 JIT/kernel、首次 mmap page fault 不进入 measurement window。

## 3. DataLoader sweep

多 worker 可能改变各 worker 的 RNG 消费顺序，因此这一组只作为吞吐消融，不用于逐-step bitwise 结论：

```bash
for workers in 0 1 2 4; do
  RESULT_ROOT="$PWD/results/workers-${workers}" \
  DATA_WORKERS="${workers}" \
  bash scripts/run_variant.sh optimized-data
done
```

重点看 `H3/data_wait`、共享存储吞吐、CPU RSS/page faults 与最慢 rank step time。8 ranks × 4 workers 会产生 32 readers，不一定更快。

## 4. ZeRO-3 prefetch/overlap sweep

```bash
RESULT_ROOT="$PWD/results/zero-sweep" \
WARMUP_STEPS=5 MEASURE_STEPS=30 \
bash scripts/run_zero_sweep.sh
```

会比较 original、50M、160M、320M prefetch window。接受某个配置至少需要：多次中位数更快、无 OOM、max-rank 不恶化，并在 Nsight 中看到 exposed all-gather 确实缩短。

## 5. Nsight 找 bound

```bash
PROFILE_MODE=baseline bash scripts/profile_nsys.sh
PROFILE_MODE=optimized-core bash scripts/profile_nsys.sh
PROFILE_MODE=optimized-data bash scripts/profile_nsys.sh
PROFILE_MODE=optimized-zero bash scripts/profile_nsys.sh
```

建议先各采 3 step warm-up + 5 step trace。主要判断：

| timeline 现象 | 说明 |
|---|---|
| step 开头 GPU 空白、`H3/data_wait` 长 | I/O/H2D bound |
| block 间仍有 D2H / host sync | 仍存在 Python/CUDA 同步 |
| block 前 exposed `ncclAllGather` | ZeRO-3 参数 gather bound |
| backward 后 GPU 空、CPU 满载 | CPU optimizer/NUMA/内存带宽 bound |
| attention/MLP 连续满载且 NCCL 已隐藏 | 接近 compute floor，纯调度剩余空间小 |

可用 `nsys stats --report cuda_gpu_kern_sum,nvtx_sum,osrt_sum <report>.nsys-rep` 做汇总。

## 6. 数值等价门槛

仓库静态/CPU 门槛：

```bash
bash scripts/run_checks.sh
```

它会重建完整树，并校验 optimized 的 17 个变化文件是否与已验证版本的 Git blob hash 一致。

正式 GPU 性能结论前，再做一个短的 workers=0、原 ZeRO 配置运行：固定 checkpoint、CSV、world size、seed、CUDA/PyTorch/DeepSpeed/NCCL，记录每 step chunk ID/timestep/loss/grad norm，并对若干固定参数 shard 在 optimizer step 后做 checksum。`optimized-zero` 单独评估，不把潜在 collective 浮点规约顺序差异混到 core-equivalence 结论里。

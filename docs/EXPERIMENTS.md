# 实验执行计划

## 0. 代码与 CPU 等价门槛

```bash
bash scripts/run_checks.sh
```

必须全部通过。测试覆盖：

- 同 seed 的 chunk ID、target/reference latent 与文本逐元素一致；
- mmap 与普通 `torch.load` 内容一致；
- FL2VA/Ref2VA packed layout 逐元素一致；
- varlen attention 输出与梯度一致；
- 0–999 timestep 的 float32 舍入一致；
- 缩小版真实 H3 `model_fn → forward → loss → backward` 逐元素一致；
- baseline 测量补丁可干净应用，原始 launch script 未被覆盖。

## 1. 主实验：先隔离核心调度收益

```bash
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/Ref2VA/transformer/model*.safetensors' \
SEED=20260820 \
WARMUP_STEPS=5 \
MEASURE_STEPS=30 \
MODES='baseline optimized-core optimized-cache' \
bash scripts/run_ab.sh
```

三个模式的合同：

| mode | ZeRO 配置 | workers | mmap/text/layout/pin | 目的 |
|---|---|---:|---|---|
| baseline | 原配置 | 0 | 原实现 | 控制组；只临时打计时 patch |
| optimized-core | 原配置 | 0 | 全关 | 只测 host-sync、RoPE、timestep 热路径 |
| optimized-cache | 原配置 | 0 | 开 | 测 immutable cache 与 H2D 路径，不引入 worker RNG 变化 |

两边始终使用同一模型、CSV、full DiT、BF16、ZeRO-3、CPU optimizer offload、gradient checkpoint、LR、batch、seed 和更新顺序。测量期统一不写 checkpoint。

`benchmark.json` 的主指标是：

```text
samples_per_second_global = measured_steps × world_size / max_rank_elapsed
```

以最慢 rank 为准，而不是主 rank 平均值。

## 2. 重复实验

至少重复 3 次，最好 5 次。每次更换独立结果目录：

```bash
for trial in 1 2 3; do
  RESULT_ROOT="$PWD/results/trial-${trial}" \
  CSV_PATH=/path/to/preprocessed.csv \
  DIT_MODEL_PATH='/path/to/model*.safetensors' \
  MODES='baseline optimized-core optimized-cache' \
  bash scripts/run_ab.sh
done
```

论文表格报告中位数，并给出最小/最大或标准差。不要把模型初始化、首次 mmap page fault、首次 kernel/JIT warm-up 混入窗口。

## 3. DataLoader worker sweep

多 worker 可能改变各 worker 的 RNG 消费顺序，因此它是吞吐消融，不作为逐 step bitwise A/B：

```bash
for workers in 0 1 2 4; do
  RESULT_ROOT="$PWD/results/workers-${workers}" \
  DATA_WORKERS="${workers}" \
  CSV_PATH=/path/to/preprocessed.csv \
  DIT_MODEL_PATH='/path/to/model*.safetensors' \
  bash scripts/run_benchmark_variant.sh optimized-data
done
```

同时记录：

- `H3/data_wait`；
- 最慢 rank step time；
- CPU RSS、page fault、共享文件系统读吞吐；
- pinned-memory 使用。

8 ranks × 4 workers 会形成 32 个 reader，可能比 1–2 workers 更慢。

## 4. ZeRO-3 overlap 与 prefetch sweep

先跑默认候选：

```bash
RESULT_ROOT="$PWD/results/zero-default" \
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/model*.safetensors' \
bash scripts/run_benchmark_variant.sh optimized-zero
```

然后用封装脚本只改变 ZeRO 配置；该脚本强制 `workers=0`，并把原配置、50M、160M、320M 四组写入同一结果目录。`zero-original → zero-prefetch-50m` 测的是整套 overlap/pinned/prefetch 候选；`50M → 160M → 320M` 才只改变 `stage3_prefetch_bucket_size`：

```bash
RESULT_ROOT="$PWD/results/zero-sweep" \
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/model*.safetensors' \
WARMUP_STEPS=5 MEASURE_STEPS=30 \
bash scripts/run_zero_sweep.sh
```

接受某个配置的条件：

```text
3+ 次中位数更快
+ 没有 OOM
+ max-rank 不恶化
+ peak GPU/host memory 可接受
+ Nsight 中 exposed all-gather 确实缩短
```

不能根据单次结果或仅根据 bucket 更大就判定更优。

## 5. Nsight Systems 找真实 bound

```bash
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/model*.safetensors' \
PROFILE_MODE=optimized-core \
WARMUP_STEPS=3 PROFILE_STEPS=5 \
bash scripts/profile_nsys.sh
```

建议分别 profile：

```text
baseline
optimized-core
optimized-data
optimized-zero
```

baseline 使用临时 instrumentation patch，因此原始 baseline 源码仍保持不变。

看 timeline：

| 现象 | bound / 下一步 |
|---|---|
| step 开头 GPU 空白、`H3/data_wait` 长 | I/O/page fault/H2D；扫 workers、mmap、pin |
| 50 blocks 中穿插 D2H/host sync | 仍有 `.item()`/Python scalar；继续定位调用栈 |
| block 前 exposed `ncclAllGather` | ZeRO-3 parameter gather；扫 prefetch window |
| backward 后 GPU 空白、CPU 满载 | CPU optimizer/NUMA/host bandwidth |
| attention/MLP 连续满载，NCCL 已覆盖 | 接近 compute floor，调度上限小 |
| 保存 step 的 `logging_checkpoint` 巨大 | 单独报告 checkpoint，不混稳态吞吐 |

可导出汇总：

```bash
nsys stats --report cuda_gpu_kern_sum,nvtx_sum,osrt_sum report.nsys-rep
```

NCCL report 名称取决于 Nsight 版本；若不支持，直接从 timeline 观察 NCCL kernel/collective。

## 6. 真实 GPU 数值等价

在性能结论前做一个 10-step 短跑：

1. 固定 canonical checkpoint、CSV、world size、seed、CUDA/PyTorch/DeepSpeed/NCCL；
2. workers=0，使用相同原 ZeRO 配置；
3. 记录每 step 的 row/chunk ID、video/audio timestep、loss、grad norm；
4. 在 optimizer step 后记录若干固定参数 shard 的 checksum；
5. baseline 与 optimized-core/cache 比较绝对/相对误差；
6. ZeRO overlap/prefetch 另成一组，不把可能的浮点规约顺序差异混入 core-equivalence 结论。

## 7. 建议结果表

| Variant | max-rank s/step | global samples/s | speedup | GPU peak | CPU RSS | data wait | exposed NCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | | | 1.00× | | | | |
| + host-sync/RoPE | | | | | | | |
| + layout/data cache | | | | | | | |
| + workers/prefetch | | | | | | | |
| + ZeRO schedule | | | | | | | |

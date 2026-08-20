# H3-SFT4AR：MiniMax-H3 Ref2VA 语义等价调度优化

这个仓库同时保存两份代码：

```text
baseline/DiffSynth-Studio/   用户上传的原始 H3 chunk-by-chunk Ref2VA SFT 代码
optimized/DiffSynth-Studio/  不减少 token、不改 loss/更新顺序的系统优化版
```

优化目标不是修改 Ref2VA 算法，而是减少训练中的 host synchronization、重复静态计算、重复 metadata 构造和数据等待，并提供可测的 ZeRO-3 预取调度候选。

> 当前完成了代码审查、CPU 单元测试、缩小版 H3 端到端前后向逐元素对照和静态检查。此运行环境没有 MiniMax-H3 权重、预处理数据与 8×A100，因此仓库**不预先宣称任何加速比**；真实性能需要按 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 在目标集群测量。

## 保持不变的训练语义

- 同样的 CSV 过滤和 `torch.randint(actual_num_chunks)` 随机 chunk 采样；
- 当前完整 chunk：15 个 video latent、93 个 audio latent；
- `chunk_id > 0` 时，上一完整 chunk 继续作为 Ref2VA reference；
- 相同的 token 排列、position ID、token tag、padding 和 attention；
- 相同的噪声/timestep 分布、FlowMatch loss、full-DiT trainable 参数；
- 相同的 BF16、gradient checkpoint、AdamW、ZeRO-3、CPU optimizer offload 与更新顺序。

语义边界详见 [`docs/SEMANTIC_CONTRACT.md`](docs/SEMANTIC_CONTRACT.md)。

## 实现的优化

1. **消除 repeated-block host sync**：`cu_seqlens` 在 packing 时同步保留 Python bounds，不再让每个 attention block 对 CUDA tensor 调 `.tolist()`。
2. **RoPE 表只算一次**：同一 forward 的 position grid 不变，`cos/sin` 在进入 50 个主 block 前计算一次并复用；数值与原逐层计算一致。
3. **消除 CUDA scalar 读取**：timestep、augmentation max 和 `max_seqlen` 不再通过 Python `float/int/max` 等待 GPU。
4. **复用静态 packed layout**：相同 CSV row、相同 reference 状态和 shape 复用 immutable position/index/tag metadata。
5. **数据/H2D 流水**：可控 worker、prefetch、persistent worker、pinned memory、non-blocking placement、`torch.load(mmap=True)` 和有界 text LRU。
6. **可测量运行时**：固定 warm-up/measurement window、max-rank throughput、NVTX 和 CUDA profiler capture。
7. **ZeRO-3 调度候选**：单独提供 optimizer pinned memory、通信 overlap 与 prefetch-window sweep；不会把未实测候选写成默认结论。

逐文件解释见 [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)。

## 目录

```text
baseline/                    原始 runtime code snapshot
optimized/                   优化版 runtime code
patches/                     baseline → optimized 的可应用补丁
experiments/                 baseline 测量补丁与固定 seed hook
scripts/                     等价检查、A/B、汇总和 Nsight 脚本
tests/                       CPU 语义等价测试
data/manifest.example.csv    不含真实路径的 CSV 格式示例
BASELINE_SHA256SUMS.txt       baseline 文件完整性清单
```

模型权重、latent cache、原视频/音频、真实 CSV 和训练 checkpoint 均未提交。

## 先跑等价检查

```bash
bash scripts/run_checks.sh
```

其中最强的一项会分别加载 baseline 和 optimized 的真实 H3 DiT/pipeline 代码，在两个独立 Python 进程中跑缩小版：

```text
dataset sampling → Ref2VA packing → model_fn → DiT forward → loss → backward
```

当前结果是 common tensors、模型输出、loss 和参数梯度逐元素完全一致；优化版只额外携带 immutable 调度 metadata。

## 严格 A/B

默认运行三个模式：

```text
baseline         原代码 + 仅用于计时的 instrumentation patch
optimized-core   只开 host-sync/RoPE/timestep 热路径优化；worker=0、缓存关闭
optimized-cache  同一原 ZeRO 配置与 worker=0，再开 mmap/text/layout/H2D 优化
```

```bash
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/Ref2VA/transformer/model*.safetensors' \
WARMUP_STEPS=5 MEASURE_STEPS=30 \
MODES='baseline optimized-core optimized-cache' \
bash scripts/run_ab.sh
```

结果目录会生成各模式的 `benchmark.json`、日志、运行合同和汇总表。多 worker 与 ZeRO 调度作为后续独立消融，避免把不同 RNG 消费顺序或通信配置混进核心 A/B。

## 直接运行优化版

```bash
cd optimized/DiffSynth-Studio
CSV_PATH=/path/to/preprocessed.csv \
DIT_MODEL_PATH='/path/to/Ref2VA/transformer/model*.safetensors' \
bash examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV-Optimized.sh
```

原始脚本仍保留为：

```text
examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV.sh
```

详细实验矩阵、Nsight 判据和 ZeRO sweep 命令见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 来源

代码基于用户上传的 DiffSynth-Studio code-only snapshot，上游 commit 为 `6343deda06b3e09efc9b1ce23c135c35a341d143`。更完整的来源与排除项见 [`SOURCE_SNAPSHOT.md`](SOURCE_SNAPSHOT.md)。仓库沿用上游 Apache-2.0 License。

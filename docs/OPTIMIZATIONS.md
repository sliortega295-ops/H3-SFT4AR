# 优化版逐项说明

## 1. 每个 attention block 的 CUDA→host 同步

原 `_sdpa_varlen_attention()` 在 `cu_seqlens` 已位于 CUDA 后执行：

```python
bounds = cu_seqlens.tolist()
```

Python 必须等待 tensor 可读。H3 的 50 个主 block 以及 gradient-checkpoint recompute 会反复经过该路径。

优化版在 packed builder 已知 `used`/`seq_len` 时同时保存：

```python
cu_seqlens_bounds = (0, used, seq_len)
```

DiT 进入 repeated blocks 前解析一次，后续 block 直接复用 host integer tuple。旧调用若没有该字段，最多在一次 DiT forward 入口同步一次，不再每层同步。

涉及文件：

```text
diffsynth/pipelines/minimax_h3_audio_video.py
diffsynth/models/minimax_h3_dit.py
```

## 2. 50 层重复计算同一 RoPE `cos/sin`

同一次 DiT forward 中，`img_position_ids` 和 `rope_freqs` 对所有主 block 相同。原代码在每层 Q、K 进入 attention 时再次执行相同的 `torch.cos`/`torch.sin`，checkpoint backward 还会重算。

优化版在 block loop 之前计算一次：

```python
rope_cos = torch.cos(rope_freqs).to(hidden_dtype).unsqueeze(1)
rope_sin = torch.sin(rope_freqs).to(hidden_dtype).unsqueeze(1)
```

所有 block 只读取同一结果。计算公式、dtype 和广播形状保持不变，缩小版完整前后向测试逐元素一致。

## 3. timestep 与长度的 Python scalar sync

原路径包含：

```python
t_video = 1.0 - float(timestep_video) / 1000
max(t_video, imgvid_cond_noise_aug)
int(cu[1])
```

输入在 CUDA 时，这些 Python scalar 转换会暴露同步点。

优化版：

- 在 device 上用 float64 复现原 Python-double 舍入，再转 float32；
- 使用 `torch.maximum`；
- packing 时直接保存 Python `max_seqlen`；
- refiner bounds 由已知 Python shape 构造。

## 4. static packed-layout cache

以下 metadata 对同一 CSV row 和 reference 状态是静态的：

```text
img_pos / audio_pos / text_pos
img_position_ids
cu_seqlens / max_seqlen
token_tags / padding
```

优化版按：

```text
row id + has_reference + target/ref shapes + device
```

使用每 rank 有界 LRU，默认 64。缓存不包含 latent、noise、activation、loss 或梯度；模型不会修改缓存对象。设为 0 可关闭：

```bash
PACKED_LAYOUT_CACHE_SIZE=0
```

## 5. text tag 不再 GPU→CPU→GPU

原 builder 在 CPU layout 上写 tag，若 Accelerate 已把 batch 放到 CUDA，则先执行 `text_token_tags.cpu()`，随后又把完整 layout 搬回 CUDA。

优化版先把 layout 放到目标 device，再用 device-local `index_copy_` 写 text tags。

## 6. 数据和 H2D 流水

优化版增加：

- top-level、可 pickle 的 batch-size-one collate；
- `num_workers`、`persistent_workers`、`prefetch_factor`；
- DataLoader `pin_memory`；
- Accelerate 支持时启用 non-blocking placement；
- 递归 transfer 保持原 device→dtype 顺序，并允许 pinned H2D non-blocking；
- video/audio cache 使用 `torch.load(mmap=True)`，只在访问选中 slice 时触发页面读取；
- 小 text cache 使用每 worker 有界 LRU。

没有做 video/audio 全文件 Python LRU，以免 `8 ranks × N workers` 各自长期持有大 latent 文件。共享存储上的 worker 数必须实测，不能默认越多越快。

## 7. benchmark 与 NVTX

训练 runner 增加：

```text
H3/data_wait
H3/forward
H3/backward
H3/optimizer
H3/logging_checkpoint
```

benchmark window 仅在首尾同步，内部不为 phase timing 插入 barrier；最终吞吐使用最慢 rank 的窗口时间。正式训练默认保存频率仍与 baseline 一致，只有 benchmark 命令关闭写盘。

## 8. 可选 ZeRO-3 调度候选

`deepspeed_zero3_overlap.json` 保持 ZeRO-3、BF16、CPU optimizer offload、microbatch 和更新方式不变，并显式配置：

- contiguous gradients 与 communication overlap；
- CPU optimizer pinned memory；
- Stage-3 parameter prefetch；
- 显式使用 500M-element reduce/all-gather bucket，避免把 DeepSpeed 版本默认值当作实验常量。

`zero3_sweeps/` 只扫描 `stage3_prefetch_bucket_size`：50M、160M、320M elements。它们是候选，不是结论；更大窗口可能增加 live parameters、显存和 pinned-memory 压力。

## 9. 为什么保留 `find_unused_parameters`

H3 RoPE 模块注册了 `inv_freq` parameter，但当前 forward 重新构造频率，没有读取该 parameter。优化版没有擅自删除 `--find_unused_parameters`，避免在目标 DeepSpeed/DDP 环境中引入 unused-parameter 行为变化。

## 预期在哪里看到收益

- CPU launch timeline 中不再有每层 `.tolist()`/scalar read 造成的空洞；
- RoPE trig kernel 数量从随 block 数增长变为每个 forward 一次；
- repeated prompt/layout 不再每 step 重建 NumPy/CPU grid 并搬到设备；
- 数据慢时，`H3/data_wait` 和 step 开头 GPU idle 缩短；
- communication-bound 时，合适的 prefetch window 减少 exposed `ncclAllGather`。

若 Nsight 显示 attention/MLP kernel 已连续占满且 NCCL/data 都被覆盖，则当前已经接近纯计算 floor，调度优化不会产生很大端到端收益。

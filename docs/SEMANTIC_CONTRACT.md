# 语义等价合同

## 必须保持不变

优化版不得减少 reference/target token，不得改变 chunk 边界、reference 来源、noise、timestep 分布、position ID、token tag、attention、loss、gradient accumulation、optimizer 或参数更新顺序。

一次训练样本仍为：

```text
prompt embedding
+ 上一完整 chunk 的 clean video/audio latent（chunk_id > 0）
+ 当前完整 chunk 的 noisy video/audio latent
→ H3 DiT
→ 仅对当前 target video/audio 计算 FlowMatch SFT loss
```

生产脚本仍默认每 100 step 保存一次，与原脚本一致。benchmark 中的 `--skip_final_save` 和极大 `save_steps` 只用于隔离稳态训练吞吐，不用于正式训练。

## 允许改变

只改变不参与训练问题定义的执行方式：

- CPU 文件读取、mmap 和有界只读缓存；
- DataLoader worker、prefetch、pinned memory 与 H2D 发起时机；
- immutable packed metadata 的复用；
- host scalar/segment metadata 的保存位置；
- 同一输入上重复纯函数结果（RoPE `cos/sin`）的复用；
- ZeRO-3 参数预取窗口与通信—计算 overlap；
- benchmark/profiling 标记和测量期间的 checkpoint I/O 隔离。

## 数值边界

原代码通过 Python double 计算 `1 - timestep/1000`，再写入 float32 tensor。优化版在设备上使用 float64 完成同一运算后转 float32；测试遍历 0–999 的 timestep，逐元素完全一致。

CPU 缩小版 H3 测试分别执行 baseline 与 optimized 的真实：dataset、packed builder、`model_fn_minimax_h3`、DiT forward、loss 和 backward，当前输出与所有可见参数梯度均逐元素完全一致。

单独启用 ZeRO/NCCL overlap 或 bucket sweep 后，数学操作不变，但分布式浮点规约的执行顺序可能由 DeepSpeed/NCCL 改变，因此不承诺跨配置 bitwise identity。此时必须同时报告 loss/梯度/参数误差，而不能只写“语义相同”。

## 明确没有做

- 减少 reference 帧数、空间分辨率或 token；
- sparse attention、token pruning、sequence shortening；
- 改 attention mask 以实现 reference KV cache；
- LoRA 替代 full fine-tuning；
- 冻结 AdaLN、RoPE 或任何参数；
- 改 batch、loss 权重、timestep sampling 或 optimizer；
- 改 gradient checkpoint 边界。

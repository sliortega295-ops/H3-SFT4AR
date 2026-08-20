# ZeRO-3 prefetch sweep

These configs keep the training graph, precision, optimizer placement and update
order unchanged. Only `stage3_prefetch_bucket_size` changes.

- `50m`: 50,000,000 parameter elements.
- `160m`: 160,000,000 parameter elements.
- `320m`: 320,000,000 parameter elements.

Run one config at a time from the DiffSynth-Studio root:

```bash
ACCELERATE_CONFIG=examples/minimax_h3/model_training/full/zero3_sweeps/accelerate_zero3_prefetch_160m.yaml \
CSV_PATH=/path/to/manifest.csv \
DIT_MODEL_PATH='/path/to/model*.safetensors' \
bash examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV-Optimized.sh
```

Compare at least three runs and inspect exposed `ncclAllGather`, max-rank step
time, peak GPU memory and host pinned memory. A larger window can be slower or
OOM if it increases live parameters without hiding additional communication.

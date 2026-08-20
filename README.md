# H3-SFT4AR — one-command MiniMax-H3 Ref2VA A/B

This repo compares the original chunk-by-chunk MiniMax-H3 Ref2VA SFT path with semantic-preserving scheduling optimizations.

## Run

```bash
git clone https://github.com/sliortega295-ops/H3-SFT4AR.git
cd H3-SFT4AR
cp config.example.env config.env
# edit CSV_PATH and DIT_MODEL_PATH
bash run_ab.sh
```

That single command reconstructs **complete runnable source trees** under `.work/`, launches all requested modes, and writes a speedup table to `results/.../summary.md`.

## Source reconstruction

The uploaded code snapshot was based on `modelscope/DiffSynth-Studio` commit:

`6343deda06b3e09efc9b1ce23c135c35a341d143`

`baseline/DiffSynth-Studio/` stores the user's Ref2VA delta on top of that pinned upstream commit. `scripts/bootstrap.sh` reconstructs the complete baseline dependency tree from the pinned commit and this delta.

The complete optimized tree is then derived from **that exact baseline**. Experiment-facing dataset/launch/ZeRO files are stored under `optimized/DiffSynth-Studio/examples/`; the five core runtime changes are stored in `patches/optimized-core/` and applied in order. This patch set is the source of truth for the optimized runtime. It avoids storing two full DiffSynth copies while guaranteeing both variants share the same dependency closure.

If a cluster cannot access GitHub, set `DIFFSYNTH_SOURCE` in `config.env` to an existing DiffSynth git checkout containing the pinned commit.

## Comparison modes

- `baseline`: original uploaded Ref2VA source + timing-only instrumentation.
- `optimized-core`: same workers/cache/ZeRO; only hot-path scheduling/sync/RoPE optimizations.
- `optimized-data`: adds mmap/cache/pinned prefetch/non-blocking H2D.
- `optimized-zero`: additionally enables the candidate ZeRO-3 overlap config.

All modes keep full Ref2VA/target tokens, attention semantics, loss, trainable parameters, gradient-checkpoint boundary, and optimizer update order unchanged.

## Validation

The reconstructed optimized tree has been checked against the previously validated optimized source for all 17 changed paths: all Git blob hashes match. The semantic test suite reports `7 passed`, and a tiny real H3 path (`dataset -> packing -> DiT -> loss -> backward`) is bitwise identical between baseline and optimized.

A real 8-GPU MiniMax-H3 throughput run is intentionally not claimed here because this environment does not contain the H3 weights/latent dataset. Run the A/B command above on the target cluster.

## Profile the bound

```bash
PROFILE_MODE=baseline bash scripts/profile_nsys.sh
PROFILE_MODE=optimized-core bash scripts/profile_nsys.sh
bash scripts/run_zero_sweep.sh
```

See `docs/SEMANTIC_CONTRACT.md` and `docs/EXPERIMENTS.md` for the measurement contract and experiment interpretation.

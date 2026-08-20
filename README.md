# H3-SFT4AR — one-command MiniMax-H3 Ref2VA A/B

This repo compares the original chunk-by-chunk MiniMax-H3 Ref2VA SFT path with semantic-preserving scheduling optimizations.

## Run

```bash
git clone https://github.com/sliortega295-ops/H3-SFT4AR.git
cd H3-SFT4AR
cp config.example.env config.env
# edit CSV_PATH and DIT_MODEL_PATH
./run_ab.sh
```

That single command reconstructs **complete runnable source trees** under `.work/`, launches all requested modes, and writes a speedup table to `results/.../summary.md`.

### How the complete trees are reconstructed

The uploaded code snapshot was based on `modelscope/DiffSynth-Studio` commit:

`6343deda06b3e09efc9b1ce23c135c35a341d143`

The checked-in `baseline/DiffSynth-Studio/` and `optimized/DiffSynth-Studio/` directories are intentionally **delta overlays**, not standalone copies of the whole upstream repo. `scripts/bootstrap.sh` checks out the pinned upstream commit, overlays the uploaded Ref2VA baseline files, then derives optimized from that exact baseline and overlays only the optimization delta. This guarantees both modes share the same complete dependency closure without storing two 20+MB copies in this repository.

If a cluster cannot access GitHub, set `DIFFSYNTH_SOURCE` in `config.env` to an existing DiffSynth git checkout containing the pinned commit.

## Comparison modes

- `baseline`: original uploaded Ref2VA source + timing-only instrumentation.
- `optimized-core`: same workers/cache/ZeRO; only hot-path scheduling/sync/RoPE optimizations.
- `optimized-data`: adds mmap/cache/pinned prefetch/non-blocking H2D.
- `optimized-zero`: additionally enables the candidate ZeRO-3 overlap config.

All modes keep full Ref2VA/target tokens, attention semantics, loss, trainable parameters, gradient-checkpoint boundary, and optimizer update order unchanged. See `docs/SEMANTIC_CONTRACT.md` and `docs/EXPERIMENTS.md`.

## Profile the bound

```bash
PROFILE_MODE=baseline bash scripts/profile_nsys.sh
PROFILE_MODE=optimized-core bash scripts/profile_nsys.sh
bash scripts/run_zero_sweep.sh
```

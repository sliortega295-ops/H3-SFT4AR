# Optimized source layout

The committed `optimized/DiffSynth-Studio/examples/` tree is the readable experiment-facing overlay (dataset, launchers, ZeRO configs).

The five core runtime modifications are intentionally kept as the exact verified patches in `../patches/optimized-core/`:

1. logger/final-save benchmark control
2. training runner + NVTX/benchmark/data scheduling
3. non-blocking recursive H2D transfer
4. MiniMax-H3 DiT host-sync/RoPE reuse
5. MiniMax-H3 packed Ref2VA metadata/timestep scheduling

Run `bash ../scripts/bootstrap.sh` from the repository root to materialize the complete optimized source at `.work/optimized/`. Do not treat partial convenience files under this directory as a standalone DiffSynth checkout; `.work/optimized/` is the canonical generated optimized tree.

# Uploaded source snapshot

The uploaded code-only snapshot reported:

- upstream: `https://github.com/modelscope/DiffSynth-Studio.git`
- branch: `main`
- HEAD: `6343deda06b3e09efc9b1ce23c135c35a341d143`

The user's Ref2VA training changes were local additions/edits on top of that commit. `baseline/DiffSynth-Studio/` stores the runtime-relevant baseline delta. `scripts/bootstrap.sh` archives the exact pinned upstream commit and applies that baseline delta to reconstruct the complete original training tree.

The optimized tree is always derived from that reconstructed baseline. Experiment-facing files under `optimized/DiffSynth-Studio/examples/` are overlaid, and the verified five core runtime patches under `patches/optimized-core/` are applied in order. `experiments/optimized_expected_git_hashes.txt` pins all 17 resulting changed blobs for reproducibility checks.

Model weights, latent caches, videos, and the real preprocessed CSV are not stored in this repository; they are supplied through `config.env`.

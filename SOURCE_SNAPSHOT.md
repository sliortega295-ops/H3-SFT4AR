# Uploaded source snapshot

The uploaded code-only snapshot reported:

- upstream: `https://github.com/modelscope/DiffSynth-Studio.git`
- branch: `main`
- HEAD: `6343deda06b3e09efc9b1ce23c135c35a341d143`

The user's Ref2VA runtime changes were local additions/edits on top of that commit. The repository keeps those paths under `baseline/DiffSynth-Studio/` as an overlay. `scripts/bootstrap.sh` applies that overlay to the pinned upstream commit to build a full runnable baseline, then applies the checked-in optimized overlay to build the full runnable optimized tree.

Model weights, latent caches, videos, and the real preprocessed CSV are not stored in this repository; they are supplied through `config.env`.

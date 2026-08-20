# Source snapshot

- Uploaded archive: `DiffSynth-Studio-minimax-h3-code-only-20260820.tar.gz`
- Upstream repository recorded in the archive: `modelscope/DiffSynth-Studio`
- Upstream commit recorded in the archive: `6343deda06b3e09efc9b1ce23c135c35a341d143`
- Snapshot date: 2026-08-20
- Uploaded archive SHA-256: `9dbeae7d8a9bf9fc2e9056e5aa1d37436432d1b758337c2416c2d6fac554e739`

`baseline/DiffSynth-Studio` is a source-preserved subset of the uploaded code-only tree. Every file present under this baseline was copied byte-for-byte from the upload; in particular, the user-added H3 files and the original ZeRO-3 configuration are untouched. `BASELINE_SHA256SUMS.txt` records the public baseline tree for later verification.

The public repository intentionally omits:

- real training CSV manifests, because they contain private filesystem paths and captions;
- model weights, latent caches, media and checkpoints;
- generated `__pycache__`, build metadata (`diffsynth.egg-info`) and upstream documentation unrelated to executing the uploaded H3 trainer.

A path-free schema example is provided at `data/manifest.example.csv`.

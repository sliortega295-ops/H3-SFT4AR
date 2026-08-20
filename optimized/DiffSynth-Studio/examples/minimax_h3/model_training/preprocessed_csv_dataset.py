from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch


H3_FRAME_RATE = 24
H3_AUDIO_LATENT_RATE = 40
DEFAULT_CHUNK_NUM_FRAMES = 56
DEFAULT_VIDEO_CHUNK_LATENTS = 15
DEFAULT_AUDIO_CHUNK_LATENTS = round(
    DEFAULT_CHUNK_NUM_FRAMES / H3_FRAME_RATE * H3_AUDIO_LATENT_RATE
)


def _load_torch_cache(path, *, mmap=False):
    """Load a trusted preprocessing cache with compatibility fallbacks.

    ``mmap=True`` changes only host-side storage scheduling. The serialized
    tensor values and the chunk sampling rule remain identical to the baseline.
    """

    attempts = []
    if mmap:
        attempts.append({"map_location": "cpu", "weights_only": False, "mmap": True})
    attempts.extend(
        [
            {"map_location": "cpu", "weights_only": False},
            {"map_location": "cpu"},
        ]
    )
    last_error = None
    for kwargs in attempts:
        try:
            return torch.load(path, **kwargs)
        except (TypeError, RuntimeError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to load cache {path}.")


def _cache_tensor(cache, key, path):
    if torch.is_tensor(cache):
        tensor = cache
    elif isinstance(cache, dict) and key in cache:
        tensor = cache[key]
    else:
        raise ValueError(f"Cache {path} does not contain tensor field {key!r}.")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Cache field {key!r} in {path} is not a tensor.")
    return tensor


class MiniMaxH3PreprocessedCSVDataset(torch.utils.data.Dataset):
    """Read H3 caches and sample one random target/reference chunk per item.

    Large video/audio cache files may be memory-mapped. The much smaller text
    cache is retained in a bounded per-worker LRU because the same prompts are
    reused by ``dataset_repeat``. No video/audio tensor is retained across items.
    """

    load_from_cache = False

    def __init__(
        self,
        metadata_path,
        dataset_base_path=None,
        repeat=1,
        chunk_num_frames=DEFAULT_CHUNK_NUM_FRAMES,
        video_chunk_latents=DEFAULT_VIDEO_CHUNK_LATENTS,
        audio_chunk_latents=DEFAULT_AUDIO_CHUNK_LATENTS,
        video_latents_column="h3_video_latents",
        audio_latents_column="h3_audio_latents",
        text_emb_column="h3_text_emb",
        status_column="h3_status",
        cache_mmap=True,
        text_cache_size=64,
    ):
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.base_path = (
            Path(dataset_base_path).expanduser().resolve()
            if dataset_base_path
            else self.metadata_path.parent
        )
        self.repeat = int(repeat)
        self.chunk_num_frames = int(chunk_num_frames)
        self.video_chunk_latents = int(video_chunk_latents)
        self.audio_chunk_latents = int(audio_chunk_latents)
        self.cache_mmap = bool(cache_mmap)
        self.text_cache_size = int(text_cache_size)
        self._text_cache = OrderedDict()
        if self.repeat < 1:
            raise ValueError("dataset_repeat must be at least 1.")
        if min(self.chunk_num_frames, self.video_chunk_latents, self.audio_chunk_latents) < 1:
            raise ValueError("Chunk sizes must be positive.")
        if self.text_cache_size < 0:
            raise ValueError("text_cache_size must be non-negative.")

        dataframe = pd.read_csv(self.metadata_path)
        required_columns = {
            video_latents_column,
            audio_latents_column,
            text_emb_column,
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"CSV {self.metadata_path} is missing columns: {sorted(missing_columns)}"
            )

        records = []
        for row_id, row in dataframe.iterrows():
            status = str(row.get(status_column, "ok"))
            if status not in ("ok", "skipped_existing"):
                continue
            try:
                video_path = self._resolve_path(row[video_latents_column])
                audio_path = self._resolve_path(row[audio_latents_column])
                text_path = self._resolve_path(row[text_emb_column])
                num_frames = int(row.get("h3_num_frames", 0))
                video_latent_t = int(row.get("h3_video_latent_t", 0))
                audio_latent_t = int(row.get("h3_audio_latent_t", 0))
            except (TypeError, ValueError):
                continue

            num_chunks = min(
                num_frames // self.chunk_num_frames,
                video_latent_t // self.video_chunk_latents,
                audio_latent_t // self.audio_chunk_latents,
            )
            if num_chunks < 1:
                continue
            records.append(
                {
                    "row_id": int(row_id),
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "text_path": text_path,
                    "num_chunks": num_chunks,
                }
            )

        if not records:
            raise ValueError(
                "No usable rows remain after filtering status and chunk lengths in "
                f"{self.metadata_path}."
            )
        self.records = records
        print(
            f"Loaded {len(self.records)} usable rows from {self.metadata_path}; "
            f"each access samples one random {self.chunk_num_frames}-frame/"
            f"{self.video_chunk_latents}-video-latent chunk. "
            f"mmap={self.cache_mmap}, text_cache_size={self.text_cache_size}."
        )

    def _resolve_path(self, value):
        if pd.isna(value) or not str(value).strip():
            raise ValueError("empty cache path")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.base_path / path
        return path.resolve()

    def _load_text_cache(self, path):
        if self.text_cache_size == 0:
            return _load_torch_cache(path, mmap=self.cache_mmap)
        key = str(path)
        cached = self._text_cache.pop(key, None)
        if cached is None:
            cached = _load_torch_cache(path, mmap=self.cache_mmap)
        self._text_cache[key] = cached
        while len(self._text_cache) > self.text_cache_size:
            self._text_cache.popitem(last=False)
        return cached

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        record = self.records[index % len(self.records)]
        video_cache = _load_torch_cache(record["video_path"], mmap=self.cache_mmap)
        audio_cache = _load_torch_cache(record["audio_path"], mmap=self.cache_mmap)
        text_cache = self._load_text_cache(record["text_path"])

        video_latents = _cache_tensor(video_cache, "latents", record["video_path"])
        audio_latents = _cache_tensor(audio_cache, "latents", record["audio_path"])
        prompt_embeds = _cache_tensor(
            text_cache, "prompt_embeds", record["text_path"]
        )
        text_token_tags = _cache_tensor(
            text_cache, "text_token_tags", record["text_path"]
        )

        if video_latents.ndim != 5 or video_latents.shape[0] != 1:
            raise ValueError(
                f"Expected video latents [1,C,T,H,W], got {tuple(video_latents.shape)} "
                f"from {record['video_path']}."
            )
        if audio_latents.ndim != 3:
            raise ValueError(
                f"Expected audio latents [C,D,T], got {tuple(audio_latents.shape)} "
                f"from {record['audio_path']}."
            )
        if prompt_embeds.ndim != 2 or text_token_tags.numel() != prompt_embeds.shape[0]:
            raise ValueError(
                "prompt_embeds/text_token_tags have incompatible shapes in "
                f"{record['text_path']}: {tuple(prompt_embeds.shape)} and "
                f"{tuple(text_token_tags.shape)}."
            )

        actual_num_chunks = min(
            record["num_chunks"],
            video_latents.shape[2] // self.video_chunk_latents,
            audio_latents.shape[-1] // self.audio_chunk_latents,
        )
        if actual_num_chunks < 1:
            raise ValueError(
                f"Caches for CSV row {record['row_id']} are shorter than manifest metadata."
            )

        # Preserve the exact baseline RNG call and chunk distribution.
        chunk_id = int(torch.randint(actual_num_chunks, (1,)).item())
        video_start = chunk_id * self.video_chunk_latents
        audio_start = chunk_id * self.audio_chunk_latents
        video_end = video_start + self.video_chunk_latents
        audio_end = audio_start + self.audio_chunk_latents
        has_reference = chunk_id > 0

        sample = {
            "video_latents": video_latents[:, :, video_start:video_end].contiguous(),
            "audio_latents": audio_latents[..., audio_start:audio_end].contiguous(),
            "prompt_embeds": prompt_embeds.contiguous(),
            "text_token_tags": text_token_tags.reshape(-1).to(torch.long).contiguous(),
            "has_reference": has_reference,
            "chunk_id": chunk_id,
            "row_id": record["row_id"],
            # Static packed metadata depends on this row's text tags and whether
            # a previous chunk is present, not on which later chunk was sampled.
            "layout_cache_key": f"row={record['row_id']}:ref={int(has_reference)}",
        }
        if has_reference:
            ref_video_start = video_start - self.video_chunk_latents
            ref_audio_start = audio_start - self.audio_chunk_latents
            sample["ref_video_latents"] = video_latents[
                :, :, ref_video_start:video_start
            ].contiguous()
            sample["ref_audio_latents"] = audio_latents[
                ..., ref_audio_start:audio_start
            ].contiguous()
        return sample

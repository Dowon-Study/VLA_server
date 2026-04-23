"""
LIBERO dataset loader for OpenVLA fine-tuning.

Reads the lerobot v3 parquet + mp4 format and returns
(pixel_values, input_ids, attention_mask, labels) for each timestep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Prompt format (OpenVLA paper §3.2)
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def _make_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Video frame reader  (lazy — reads one frame at a time via decord)
# ──────────────────────────────────────────────────────────────────────────────

class _VideoReader:
    """Thin wrapper around decord VideoReader with per-path caching."""

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def get_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        try:
            import decord
            decord.bridge.set_bridge("native")
        except ImportError:
            raise ImportError("decord is required: pip install decord")

        if video_path not in self._cache:
            import decord
            self._cache[video_path] = decord.VideoReader(video_path, ctx=decord.cpu(0))
        vr = self._cache[video_path]
        frame = vr[frame_idx].asnumpy()   # H×W×3, uint8, RGB
        return frame

    def clear(self):
        self._cache.clear()


_VIDEO_READER = _VideoReader()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset index builder
# ──────────────────────────────────────────────────────────────────────────────

def _load_tasks(meta_dir: Path) -> Dict[int, str]:
    """Return {task_index: instruction_string}."""
    path = meta_dir / "tasks.jsonl"
    if not path.exists():
        # Fallback: episodes.jsonl may contain task info
        path = meta_dir / "episodes.jsonl"
    tasks: Dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task_index" in obj and "task" in obj:
                tasks[int(obj["task_index"])] = obj["task"]
            elif "index" in obj and "task" in obj:
                tasks[int(obj["index"])] = obj["task"]
    return tasks


def _build_index(dataset_root: Path, camera_key: str) -> List[dict]:
    """
    Scan all parquet files and build a flat list of per-frame records.

    Each record:
        episode_idx  : int
        frame_idx    : int   (within episode)
        task_idx     : int
        action       : np.ndarray (7,)
        video_path   : str
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required: pip install pyarrow")

    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    data_pattern = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_pattern = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    chunks_size = info.get("chunks_size", 1000)

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    index = []
    for pf in parquet_files:
        table = pq.read_table(str(pf), columns=["episode_index", "frame_index", "task_index", "action"])
        ep_indices = table.column("episode_index").to_pylist()
        fr_indices = table.column("frame_index").to_pylist()
        task_indices = table.column("task_index").to_pylist()
        actions = table.column("action").to_pylist()

        for ep_idx, fr_idx, task_idx, action in zip(ep_indices, fr_indices, task_indices, actions):
            ep_idx = int(ep_idx)
            fr_idx = int(fr_idx)
            task_idx = int(task_idx)
            action = np.array(action, dtype=np.float32)

            chunk_i = ep_idx // chunks_size
            file_i = ep_idx % chunks_size
            video_path = str(dataset_root / video_pattern.format(
                video_key=camera_key,
                chunk_index=chunk_i,
                file_index=file_i,
            ))
            index.append({
                "episode_idx": ep_idx,
                "frame_idx": fr_idx,
                "task_idx": task_idx,
                "action": action,
                "video_path": video_path,
            })

    return index


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class LIBERODataset(Dataset):
    """
    PyTorch Dataset for OpenVLA fine-tuning on LIBERO lerobot-format data.

    Returns per-timestep dicts:
        pixel_values    : (3, 224, 224) float32 tensor (normalized by processor)
        input_ids       : (seq_len,) long tensor
        attention_mask  : (seq_len,) long tensor
        labels          : (seq_len,) long tensor  — -100 for prompt, action IDs for output
    """

    def __init__(
        self,
        dataset_root: str,
        processor,
        action_tokenizer,
        camera_key: str = "observation.images.image",
        image_size: int = 224,
        norm_mode: str = "minmax",
    ):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.image_size = image_size

        print(f"[Dataset] Loading index from {self.dataset_root} ...")
        self.index = _build_index(self.dataset_root, camera_key)
        print(f"[Dataset] {len(self.index)} frames across all episodes.")

        self.tasks = _load_tasks(self.dataset_root / "meta")
        if not self.tasks:
            raise RuntimeError("tasks.jsonl / episodes.jsonl not found or empty. "
                               "Run merge_with_hf_libero.py to build the merged dataset.")

        # Load normalization bounds
        from .action_tokenizer import ActionTokenizer
        bounds = ActionTokenizer.load_stats(str(self.dataset_root), norm_mode)
        self.action_lo = bounds["lo"]
        self.action_hi = bounds["hi"]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec = self.index[idx]
        task_idx = rec["task_idx"]
        action = rec["action"]            # np.ndarray (7,)
        video_path = rec["video_path"]
        frame_idx = rec["frame_idx"]

        # ── language instruction ──────────────────────────────────────
        instruction = self.tasks.get(task_idx, "complete the task")
        prompt = _make_prompt(instruction)

        # ── image ─────────────────────────────────────────────────────
        frame_np = _VIDEO_READER.get_frame(video_path, frame_idx)  # H×W×3 uint8
        image = Image.fromarray(frame_np).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # ── tokenize action ───────────────────────────────────────────
        action_token_ids = self.action_tokenizer.encode_full(
            action, self.action_lo, self.action_hi
        )  # list[int], length = action_dim

        # ── build model input ─────────────────────────────────────────
        # processor handles image embedding + prompt tokenization
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)         # (prompt_len,)
        pixel_values = encoding["pixel_values"].squeeze(0)   # (C, H, W)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Append action token IDs to input_ids
        action_ids = torch.tensor(action_token_ids, dtype=torch.long)
        full_input_ids = torch.cat([input_ids, action_ids], dim=0)
        full_attention_mask = torch.cat(
            [attention_mask, torch.ones(len(action_ids), dtype=torch.long)], dim=0
        )

        # Labels: -100 for prompt tokens, actual token IDs for action tokens
        labels = torch.full_like(full_input_ids, fill_value=-100)
        labels[len(input_ids):] = action_ids

        return {
            "pixel_values": pixel_values,
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────────────────────────────────────

class DataCollatorForOpenVLA:
    """Left-pads sequences so all samples in a batch have the same length."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, features: List[dict]) -> dict:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        pixel_values_batch = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Left-pad (OpenVLA uses left-padding consistent with Llama-2)
            input_ids_batch.append(
                torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=torch.long), f["input_ids"]])
            )
            attention_mask_batch.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long), f["attention_mask"]])
            )
            labels_batch.append(
                torch.cat([torch.full((pad_len,), -100, dtype=torch.long), f["labels"]])
            )
            pixel_values_batch.append(f["pixel_values"])

        return {
            "input_ids": torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attention_mask_batch),
            "labels": torch.stack(labels_batch),
            "pixel_values": torch.stack(pixel_values_batch),
        }

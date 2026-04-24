"""
LIBERO dataset loader for OpenVLA fine-tuning.

Reads lerobot v3 parquet format (images stored as bytes inside parquet).
MP4 video files are NOT required.

Meta files required:
  meta/info.json
  meta/stats.json
  meta/tasks.parquet  (or tasks.jsonl as fallback)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Prompt format (OpenVLA paper §3.2)
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut: "


def _make_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Parquet image cache
# parquet 파일별로 이미지 bytes를 캐시 (frame position 기반 접근)
# ──────────────────────────────────────────────────────────────────────────────

class _ParquetImageCache:
    """
    LRU cache for image bytes read from lerobot v3 parquet files.

    parquet 1개 파일 = 에피소드 1개 (수백 프레임).
    get_image() 호출 시 해당 파일 전체를 로드해 캐시한 뒤
    row 위치로 직접 접근합니다.
    """

    def __init__(self, max_files: int = 32):
        self._max   = max_files
        self._cache: Dict[str, list] = {}   # path -> list of image bytes (per row)
        self._order: List[str] = []         # LRU order

    def get_image(self, parquet_path: str, row_in_file: int, col: str) -> np.ndarray:
        if parquet_path not in self._cache:
            self._load(parquet_path, col)
        img_bytes = self._cache[parquet_path][row_in_file]
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    def _load(self, parquet_path: str, col: str):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=[col])
        rows  = table.column(col).to_pylist()  # list of {"bytes": ..., "path": ...}
        self._cache[parquet_path] = [r["bytes"] for r in rows]
        self._order.append(parquet_path)
        # LRU eviction
        while len(self._order) > self._max:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)


_PARQUET_IMG_CACHE = _ParquetImageCache(max_files=32)


# ──────────────────────────────────────────────────────────────────────────────
# Task loader  (tasks.parquet 우선, tasks.jsonl fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _load_tasks(meta_dir: Path) -> Dict[int, str]:
    """Return {task_index: instruction_string}."""

    # 1) tasks.parquet (lerobot v3 기본 포맷)
    pq_path = meta_dir / "tasks.parquet"
    if pq_path.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(str(pq_path)).to_pydict()
            # 컬럼명: task_index + task  OR  task_index + __index_level_0__
            descs = t.get("task", t.get("__index_level_0__", []))
            idxs  = t.get("task_index", list(range(len(descs))))
            tasks = {int(i): str(d) for i, d in zip(idxs, descs)}
            if tasks:
                return tasks
        except Exception as e:
            print(f"[Dataset] tasks.parquet 로드 실패: {e}")

    # 2) tasks.jsonl / episodes.jsonl (fallback)
    for fname in ("tasks.jsonl", "episodes.jsonl"):
        path = meta_dir / fname
        if not path.exists():
            continue
        tasks: Dict[int, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task_index" in obj and "task" in obj:
                tasks[int(obj["task_index"])] = obj["task"]
            elif "index" in obj and "task" in obj:
                tasks[int(obj["index"])] = obj["task"]
        if tasks:
            return tasks

    return {}


# ──────────────────────────────────────────────────────────────────────────────
# No-op filter
# ──────────────────────────────────────────────────────────────────────────────

def _is_noop_action(
    action: np.ndarray,
    prev_action: Optional[np.ndarray],
    threshold: float,
) -> bool:
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return (
        np.linalg.norm(action[:-1]) < threshold
        and np.isclose(action[-1], prev_action[-1], atol=threshold)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Index builder  (parquet 기반)
# ──────────────────────────────────────────────────────────────────────────────

def _build_index(
    dataset_root: Path,
    camera_key: str,
    skip_noop_actions: bool = True,
    noop_threshold: float = 1e-4,
) -> List[dict]:
    """
    모든 parquet 파일을 스캔해 프레임 단위 인덱스를 만듭니다.

    각 레코드:
        episode_idx  : int
        frame_idx    : int   (에피소드 내 프레임 번호)
        task_idx     : int
        action       : np.ndarray (7,)
        parquet_path : str   (이미지를 읽을 parquet 파일 경로)
        row_in_file  : int   (parquet 파일 내 행 위치)
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required: pip install pyarrow")

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"parquet 파일 없음: {data_dir}")

    index: List[dict] = []
    prev_kept_action_by_episode: Dict[int, np.ndarray] = {}
    skipped_noops = 0

    for pf in parquet_files:
        table = pq.read_table(
            str(pf),
            columns=["episode_index", "frame_index", "task_index", "action"],
        )
        ep_indices   = table.column("episode_index").to_pylist()
        fr_indices   = table.column("frame_index").to_pylist()
        task_indices = table.column("task_index").to_pylist()
        actions      = table.column("action").to_pylist()

        for row_i, (ep_idx, fr_idx, task_idx, action) in enumerate(
            zip(ep_indices, fr_indices, task_indices, actions)
        ):
            ep_idx   = int(ep_idx)
            fr_idx   = int(fr_idx)
            task_idx = int(task_idx)
            action   = np.array(action, dtype=np.float32)

            prev_action = prev_kept_action_by_episode.get(ep_idx)
            if skip_noop_actions and _is_noop_action(action, prev_action, noop_threshold):
                skipped_noops += 1
                continue
            prev_kept_action_by_episode[ep_idx] = action

            index.append({
                "episode_idx":  ep_idx,
                "frame_idx":    fr_idx,
                "task_idx":     task_idx,
                "action":       action,
                "parquet_path": str(pf),   # 이미지 bytes 위치
                "row_in_file":  row_i,     # 파일 내 행 위치
            })

    if skip_noop_actions:
        print(f"[Dataset] No-op {skipped_noops}개 스킵.")
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class LIBERODataset(Dataset):
    """
    PyTorch Dataset for OpenVLA fine-tuning on LIBERO lerobot-format data.

    이미지를 parquet bytes에서 직접 읽으므로 MP4 파일 불필요.

    Returns per-timestep dicts:
        pixel_values    : (3, 224, 224) float32 tensor (processor 정규화 적용)
        input_ids       : (seq_len,) long tensor
        attention_mask  : (seq_len,) long tensor
        labels          : (seq_len,) long tensor  — prompt 부분 -100, action 토큰만 학습
    """

    def __init__(
        self,
        dataset_root: str,
        processor,
        action_tokenizer,
        camera_key: str = "observation.images.image",
        image_size: int = 224,
        norm_mode: str = "quantile",
        image_aug: bool = True,
        random_crop_scale: float = 0.9,
        color_jitter: float = 0.2,
        hue_jitter: float = 0.05,
        skip_noop_actions: bool = True,
        noop_threshold: float = 1e-4,
    ):
        self.dataset_root     = Path(dataset_root)
        self.processor        = processor
        self.action_tokenizer = action_tokenizer
        self.camera_key       = camera_key
        self.image_size       = image_size
        self.image_aug        = image_aug
        self.image_transform  = None

        if image_aug:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(random_crop_scale, random_crop_scale),
                    ratio=(1.0, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=hue_jitter,
                ),
            ])

        print(f"[Dataset] 인덱스 빌드 중: {self.dataset_root}")
        self.index = _build_index(
            self.dataset_root,
            camera_key,
            skip_noop_actions=skip_noop_actions,
            noop_threshold=noop_threshold,
        )
        print(f"[Dataset] 총 {len(self.index)} 프레임 로드 완료.")

        self.tasks = _load_tasks(self.dataset_root / "meta")
        if not self.tasks:
            raise RuntimeError(
                "tasks.parquet / tasks.jsonl을 찾을 수 없습니다. "
                "meta/tasks.parquet 파일이 dataset_root/meta/ 에 있는지 확인하세요."
            )
        print(f"[Dataset] {len(self.tasks)}개 태스크 로드 완료.")

        from .action_tokenizer import ActionTokenizer
        bounds = ActionTokenizer.load_stats(str(self.dataset_root), norm_mode)
        self.action_lo = bounds["lo"]
        self.action_hi = bounds["hi"]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec          = self.index[idx]
        task_idx     = rec["task_idx"]
        action       = rec["action"]           # np.ndarray (7,)
        parquet_path = rec["parquet_path"]
        row_in_file  = rec["row_in_file"]

        # ── 언어 명령 ────────────────────────────────────────────────
        instruction = self.tasks.get(task_idx, "complete the task")
        prompt = _make_prompt(instruction)

        # ── 이미지 (parquet bytes → PIL Image) ───────────────────────
        frame_np = _PARQUET_IMG_CACHE.get_image(parquet_path, row_in_file, self.camera_key)
        image = Image.fromarray(frame_np).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.image_transform is not None:
            image = self.image_transform(image)

        # ── 액션 토크나이즈 ──────────────────────────────────────────
        action_token_ids = self.action_tokenizer.encode_full(
            action, self.action_lo, self.action_hi
        )  # list[int], length = action_dim

        # ── 모델 입력 조립 ───────────────────────────────────────────
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        action_ids   = torch.tensor(action_token_ids, dtype=torch.long)
        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("OpenVLA tokenizer에 eos_token_id가 없습니다.")
        eos_id = torch.tensor([eos_token_id], dtype=torch.long)

        full_input_ids = torch.cat([input_ids, action_ids, eos_id], dim=0)
        full_attention_mask = torch.cat(
            [attention_mask, torch.ones(len(action_ids) + 1, dtype=torch.long)], dim=0
        )

        # prompt 부분은 -100 (loss 계산 제외), action + EOS만 학습
        labels = torch.full_like(full_input_ids, fill_value=-100)
        labels[len(input_ids):] = torch.cat([action_ids, eos_id], dim=0)

        return {
            "pixel_values":   pixel_values,
            "input_ids":      full_input_ids,
            "attention_mask": full_attention_mask,
            "labels":         labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────────────────────────────────────

class DataCollatorForOpenVLA:
    """Right-pads sequences to match the official OpenVLA fine-tuning collator."""

    def __init__(self, tokenizer):
        self.pad_token_id    = tokenizer.pad_token_id or 0
        self.model_max_length = tokenizer.model_max_length

    def __call__(self, features: List[dict]) -> dict:
        max_len = min(
            max(f["input_ids"].shape[0] for f in features),
            self.model_max_length,
        )

        input_ids_batch      = []
        attention_mask_batch = []
        labels_batch         = []
        pixel_values_batch   = []

        for f in features:
            input_ids      = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels         = f["labels"][:max_len]
            seq_len        = input_ids.shape[0]
            pad_len        = max_len - seq_len

            input_ids_batch.append(
                torch.cat([input_ids,
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask_batch.append(
                torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_batch.append(
                torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            pixel_values_batch.append(f["pixel_values"])

        return {
            "input_ids":      torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attention_mask_batch),
            "labels":         torch.stack(labels_batch),
            "pixel_values":   torch.stack(pixel_values_batch),
        }

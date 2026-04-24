from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


class ChunkedLeRobotDataset(torch.utils.data.Dataset):
    """Dataset reader for local LeRobot-style chunk/file parquet layouts.

    The local LIBERO datasets under `/media/idna/Data/Server/Libeoro_dataset` store frames in files like
    `data/chunk-000/file-000.parquet` instead of the upstream per-episode parquet layout. openpi only needs
    random access to rows and future action chunks, so we build that directly from the chunked parquet files.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        del force_cache_sync, download_videos, video_backend  # Unused for local-only datasets.

        self.repo_id = repo_id
        self.root = Path(root) if root else lerobot_dataset.HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset.CODEBASE_VERSION

        self.meta = lerobot_dataset.LeRobotDatasetMetadata(self.repo_id, self.root, self.revision)
        self._tasks = self.meta.tasks
        self._file_cache: dict[str, list[dict]] = {}

        self.delta_indices = None
        if self.delta_timestamps is not None:
            lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)

        self._rows = self._index_rows()
        self.episode_data_index = self._build_episode_data_index()

    @property
    def fps(self) -> int:
        return self.meta.fps

    def _index_rows(self) -> list[dict]:
        data_dir = self.root / "data"
        parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
        indexed_rows = []
        for parquet_file in parquet_files:
            table = pq.read_table(
                parquet_file,
                columns=["index", "episode_index", "task_index", "timestamp"],
            )
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
            for local_row, (global_index, episode_index, task_index, timestamp) in enumerate(
                zip(indices, episode_indices, task_indices, timestamps, strict=True)
            ):
                indexed_rows.append(
                    {
                        "global_index": int(global_index),
                        "episode_index": int(episode_index),
                        "task_index": int(task_index),
                        "timestamp": float(timestamp),
                        "file_path": str(parquet_file),
                        "local_row": local_row,
                    }
                )

        indexed_rows.sort(key=lambda item: item["global_index"])
        return indexed_rows

    def _build_episode_data_index(self) -> dict[int, tuple[int, int]]:
        bounds: dict[int, list[int]] = defaultdict(list)
        for global_row, item in enumerate(self._rows):
            bounds[item["episode_index"]].append(global_row)
        return {
            episode_index: (indices[0], indices[-1] + 1)
            for episode_index, indices in bounds.items()
        }

    def _load_file_rows(self, file_path: str) -> list[dict]:
        if file_path not in self._file_cache:
            self._file_cache[file_path] = pq.read_table(file_path).to_pylist()
        return self._file_cache[file_path]

    def _get_row(self, global_row: int) -> dict:
        row_info = self._rows[global_row]
        file_rows = self._load_file_rows(row_info["file_path"])
        return file_rows[row_info["local_row"]]

    @staticmethod
    def _decode_image(image_struct: dict) -> np.ndarray:
        image = Image.open(BytesIO(image_struct["bytes"]))
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._get_row(idx)
        episode_index = int(row["episode_index"])
        ep_start, ep_end = self.episode_data_index[episode_index]

        item = {
            "observation": {
                "images": {
                    "image": self._decode_image(row["observation.images.image"]),
                    "image2": self._decode_image(row["observation.images.image2"]),
                },
                "state": np.asarray(row["observation.state"], dtype=np.float32),
            },
            "timestamp": np.asarray(row["timestamp"], dtype=np.float32),
            "frame_index": np.asarray(row["frame_index"], dtype=np.int64),
            "episode_index": np.asarray(row["episode_index"], dtype=np.int64),
            "index": np.asarray(row["index"], dtype=np.int64),
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
        }

        if self.delta_indices is not None and "action" in self.delta_indices:
            query_indices = [max(ep_start, min(ep_end - 1, idx + delta)) for delta in self.delta_indices["action"]]
            padding = [(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in self.delta_indices["action"]]
            actions = [np.asarray(self._get_row(query_idx)["action"], dtype=np.float32) for query_idx in query_indices]
            item["action"] = np.stack(actions, axis=0)
            item["action_is_pad"] = np.asarray(padding, dtype=bool)
        else:
            item["action"] = np.asarray(row["action"], dtype=np.float32)

        if self.image_transforms is not None:
            for key in ("image", "image2"):
                item["observation"]["images"][key] = self.image_transforms(item["observation"]["images"][key])

        return item

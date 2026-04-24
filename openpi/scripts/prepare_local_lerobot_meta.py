#!/usr/bin/env python3
"""Hydrate local LeRobot-v3 metadata so openpi/lerobot can train on it offline.

This script is meant for local dataset roots such as:
  - /media/idna/Data/Server/Libeoro_dataset/HuggingFaceVLA_libero_lerobot
  - /media/idna/Data/Server/Libeoro_dataset/dataset_gt_prepared
  - /media/idna/Data/Server/Libeoro_dataset/libero_gt_aug_balanced

It creates:
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/episodes_stats.jsonl

from the existing parquet files and task metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.utils import EPISODES_PATH
from lerobot.common.datasets.utils import EPISODES_STATS_PATH
from lerobot.common.datasets.utils import TASKS_PATH
from lerobot.common.datasets.utils import write_episode
from lerobot.common.datasets.utils import write_episode_stats


def _load_tasks(meta_dir: Path) -> dict[int, str]:
    tasks_parquet = meta_dir / "tasks.parquet"
    if not tasks_parquet.exists():
        raise FileNotFoundError(f"Missing tasks.parquet: {tasks_parquet}")

    table = pq.read_table(tasks_parquet)
    tasks: dict[int, str] = {}
    columns = set(table.column_names)

    if {"task_index", "task"} <= columns:
        task_indices = table["task_index"].to_pylist()
        task_names = table["task"].to_pylist()
        tasks = {int(idx): str(name) for idx, name in zip(task_indices, task_names, strict=True)}
    elif {"task_index", "__index_level_0__"} <= columns:
        task_indices = table["task_index"].to_pylist()
        task_names = table["__index_level_0__"].to_pylist()
        tasks = {int(idx): str(name) for idx, name in zip(task_indices, task_names, strict=True)}
    elif "task" in columns:
        task_names = table["task"].to_pylist()
        tasks = {idx: str(name) for idx, name in enumerate(task_names)}
    else:
        raise ValueError(f"Unsupported tasks.parquet schema: {table.column_names}")

    return dict(sorted(tasks.items()))


def _write_tasks_jsonl(meta_dir: Path, tasks: dict[int, str]) -> None:
    tasks_path = meta_dir / Path(TASKS_PATH).name
    with tasks_path.open("w", encoding="utf-8") as f:
        for task_index, task in tasks.items():
            f.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")


def _to_dense_matrix(column) -> np.ndarray:
    values = column.to_pylist()
    return np.asarray(values, dtype=np.float32)


def build_episode_metadata(dataset_root: Path, tasks: dict[int, str]) -> tuple[dict[int, dict], dict[int, dict]]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    episode_lengths: dict[int, int] = defaultdict(int)
    episode_tasks: dict[int, set[str]] = defaultdict(set)
    state_buffers: dict[int, list[np.ndarray]] = defaultdict(list)
    action_buffers: dict[int, list[np.ndarray]] = defaultdict(list)

    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    required_columns = ["episode_index", "task_index", "observation.state", "action"]

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=required_columns)
        episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        states = _to_dense_matrix(table["observation.state"])
        actions = _to_dense_matrix(table["action"])

        unique_episode_indices = np.unique(episode_indices)
        for episode_index in unique_episode_indices:
            mask = episode_indices == episode_index
            ep_task_indices = np.unique(task_indices[mask]).tolist()
            for task_index in ep_task_indices:
                episode_tasks[int(episode_index)].add(tasks[int(task_index)])

            episode_lengths[int(episode_index)] += int(mask.sum())
            state_buffers[int(episode_index)].append(states[mask])
            action_buffers[int(episode_index)].append(actions[mask])

    episodes: dict[int, dict] = {}
    episodes_stats: dict[int, dict] = {}

    for episode_index in sorted(episode_lengths):
        states = np.concatenate(state_buffers[episode_index], axis=0)
        actions = np.concatenate(action_buffers[episode_index], axis=0)

        episodes[episode_index] = {
            "episode_index": episode_index,
            "tasks": sorted(episode_tasks[episode_index]),
            "length": episode_lengths[episode_index],
        }
        episodes_stats[episode_index] = {
            "observation.state": get_feature_stats(states, axis=0, keepdims=False),
            "action": get_feature_stats(actions, axis=0, keepdims=False),
        }

    return episodes, episodes_stats


def write_metadata(dataset_root: Path, tasks: dict[int, str], episodes: dict[int, dict], episodes_stats: dict[int, dict]) -> None:
    meta_dir = dataset_root / "meta"
    _write_tasks_jsonl(meta_dir, tasks)

    episodes_path = meta_dir / Path(EPISODES_PATH).name
    episodes_stats_path = meta_dir / Path(EPISODES_STATS_PATH).name
    episodes_path.unlink(missing_ok=True)
    episodes_stats_path.unlink(missing_ok=True)

    for episode_index in sorted(episodes):
        write_episode(episodes[episode_index], dataset_root)
        write_episode_stats(episode_index, episodes_stats[episode_index], dataset_root)

    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = len(tasks)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing meta directory: {meta_dir}")

    tasks = _load_tasks(meta_dir)
    episodes, episodes_stats = build_episode_metadata(dataset_root, tasks)
    write_metadata(dataset_root, tasks, episodes, episodes_stats)

    print(f"Prepared local LeRobot metadata for: {dataset_root}")
    print(f"  tasks: {len(tasks)}")
    print(f"  episodes: {len(episodes)}")


if __name__ == "__main__":
    main()

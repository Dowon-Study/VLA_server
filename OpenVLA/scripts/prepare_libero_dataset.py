#!/usr/bin/env python3
"""Prepare LeRobot-format LIBERO datasets for OpenVLA fine-tuning.

Two workflows are supported:

1) `hydrate`: create a trainable GT-only dataset root from a source that is
   missing `meta/stats.json` and/or `meta/tasks.parquet`.
2) `merge`: combine multiple LeRobot dataset roots into one merged root for
   experiments such as GT + augmented training.

The training code expects a single dataset root containing:
  - meta/info.json
  - meta/stats.json
  - meta/tasks.parquet (or tasks.jsonl fallback)
  - data/**/*.parquet

Images are read directly from parquet bytes, so videos are not required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


@dataclass
class DatasetSummary:
    total_frames: int
    total_episodes: int
    total_tasks: int
    features: dict
    fps: float
    robot_type: str
    chunks_size: int
    action_stats: dict


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_tasks_table(path: Path) -> pa.Table:
    if path.is_dir():
        meta_dir = path / "meta"
        parquet_path = meta_dir / "tasks.parquet"
        if parquet_path.exists():
            return _read_tasks_table(parquet_path)
        for fname in ("tasks.jsonl", "episodes.jsonl"):
            jsonl_path = meta_dir / fname
            if jsonl_path.exists():
                return _read_tasks_table(jsonl_path)
        raise FileNotFoundError(f"tasks metadata not found under {meta_dir}")

    if path.suffix == ".parquet":
        table = pq.read_table(path)
        data = table.to_pydict()
        descs = data.get("task", data.get("__index_level_0__", []))
        idxs = data.get("task_index", list(range(len(descs))))
        return pa.table({
            "task_index": pa.array([int(i) for i in idxs], type=pa.int64()),
            "task": pa.array([str(d) for d in descs], type=pa.string()),
        })

    if path.suffix == ".jsonl":
        task_index: List[int] = []
        task_desc: List[str] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("task_index", obj.get("index"))
            desc = obj.get("task")
            if idx is None or desc is None:
                continue
            task_index.append(int(idx))
            task_desc.append(str(desc))
        if not task_index:
            raise ValueError(f"no tasks found in {path}")
        return pa.table({
            "task_index": pa.array(task_index, type=pa.int64()),
            "task": pa.array(task_desc, type=pa.string()),
        })

    raise ValueError(f"unsupported task metadata source: {path}")


def _write_tasks_table(table: pa.Table, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = table.sort_by([("task_index", "ascending")])
    pq.write_table(table, out_path)


def _iter_parquet_files(root: Path) -> List[Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {root / 'data'}")
    return files


def _compute_action_stats(parquet_files: Sequence[Path]) -> tuple[dict, int, List[int], List[int]]:
    all_actions: List[np.ndarray] = []
    total_frames = 0
    episode_ids: List[int] = []
    task_ids: List[int] = []

    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index", "task_index", "action"])
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]
        all_actions.append(actions)
        total_frames += actions.shape[0]
        episode_ids.extend(int(x) for x in table.column("episode_index").to_pylist())
        task_ids.extend(int(x) for x in table.column("task_index").to_pylist())

    stacked = np.concatenate(all_actions, axis=0)
    stats = {
        "min": stacked.min(axis=0).tolist(),
        "max": stacked.max(axis=0).tolist(),
        "mean": stacked.mean(axis=0).tolist(),
        "std": stacked.std(axis=0).tolist(),
        "count": [int(stacked.shape[0])],
        "q01": np.quantile(stacked, 0.01, axis=0).tolist(),
        "q10": np.quantile(stacked, 0.10, axis=0).tolist(),
        "q50": np.quantile(stacked, 0.50, axis=0).tolist(),
        "q90": np.quantile(stacked, 0.90, axis=0).tolist(),
        "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
    }
    return stats, total_frames, sorted(set(episode_ids)), sorted(set(task_ids))


def _build_summary(source_root: Path, parquet_files: Sequence[Path], task_table: pa.Table) -> DatasetSummary:
    info = _load_json(source_root / "meta" / "info.json")
    action_stats, total_frames, episode_ids, task_ids = _compute_action_stats(parquet_files)
    return DatasetSummary(
        total_frames=total_frames,
        total_episodes=len(episode_ids),
        total_tasks=max(len(task_ids), task_table.num_rows),
        features=info.get("features", {}),
        fps=float(info.get("fps", 10.0)),
        robot_type=str(info.get("robot_type", "panda")),
        chunks_size=int(info.get("chunks_size", 1000)),
        action_stats=action_stats,
    )


def _build_info_payload(summary: DatasetSummary, source_note: str) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": summary.robot_type,
        "total_episodes": summary.total_episodes,
        "total_frames": summary.total_frames,
        "total_tasks": summary.total_tasks,
        "chunks_size": summary.chunks_size,
        "fps": summary.fps,
        "splits": {"train": f"0:{summary.total_episodes}"},
        "data_path": DEFAULT_DATA_PATH,
        "source": source_note,
        "features": summary.features,
    }


def _maybe_linktree(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        os.symlink(src, dst, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def _hydrate_dataset(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    task_source = Path(args.task_source).resolve()

    parquet_files = _iter_parquet_files(source_root)
    task_table = _read_tasks_table(task_source)
    summary = _build_summary(source_root, parquet_files, task_table)

    output_root.mkdir(parents=True, exist_ok=True)
    _maybe_linktree(source_root / "data", output_root / "data", symlink=not args.copy_data)
    if (source_root / "sim_states").exists():
        _maybe_linktree(source_root / "sim_states", output_root / "sim_states", symlink=not args.copy_data)

    _write_tasks_table(task_table, output_root / "meta" / "tasks.parquet")
    _write_json(output_root / "meta" / "stats.json", {"action": summary.action_stats})
    _write_json(
        output_root / "meta" / "info.json",
        _build_info_payload(summary, source_note=f"hydrated from {source_root}"),
    )

    manifest = {
        "mode": "hydrate",
        "source_root": str(source_root),
        "task_source": str(task_source),
        "total_frames": summary.total_frames,
        "total_episodes": summary.total_episodes,
    }
    _write_json(output_root / "meta" / "prepare_manifest.json", manifest)


def _parse_source_specs(values: Sequence[str]) -> List[tuple[Path, str]]:
    specs: List[tuple[Path, str]] = []
    for idx, value in enumerate(values):
        if "::" in value:
            root_str, name = value.split("::", 1)
        else:
            root_str, name = value, f"source_{idx}"
        specs.append((Path(root_str).resolve(), name))
    return specs


def _select_episode_ids(root: Path, max_episodes: int | None, seed: int) -> List[int]:
    return _select_episode_ids_with_caps(root, max_episodes=max_episodes, max_frames=None, seed=seed)


def _select_episode_ids_with_caps(
    root: Path,
    max_episodes: int | None,
    max_frames: int | None,
    seed: int,
) -> List[int]:
    files = _iter_parquet_files(root)
    frame_counts: Dict[int, int] = {}
    for pf in files:
        table = pq.read_table(pf, columns=["episode_index"])
        for ep in table.column("episode_index").to_pylist():
            ep = int(ep)
            frame_counts[ep] = frame_counts.get(ep, 0) + 1

    unique_ids = np.array(sorted(frame_counts.keys()), dtype=np.int64)
    if max_episodes is None and max_frames is None:
        return unique_ids.tolist()
    if max_episodes is not None and max_episodes >= unique_ids.shape[0] and max_frames is None:
        return unique_ids.tolist()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    selected: List[int] = []
    total_frames = 0
    for ep in shuffled:
        ep = int(ep)
        if max_episodes is not None and len(selected) >= max_episodes:
            break
        next_frames = total_frames + frame_counts[ep]
        if max_frames is not None and selected and next_frames > max_frames:
            break
        selected.append(ep)
        total_frames = next_frames
        if max_frames is not None and total_frames >= max_frames:
            break

    if not selected:
        raise ValueError(f"selection caps too strict for {root}: max_episodes={max_episodes}, max_frames={max_frames}")
    return sorted(selected)


def _replace_column(table: pa.Table, name: str, values: Iterable[int]) -> pa.Table:
    arr = pa.array(list(values), type=pa.int64())
    idx = table.schema.get_field_index(name)
    if idx >= 0:
        table = table.remove_column(idx)
        table = table.add_column(idx, name, arr)
    else:
        table = table.append_column(name, arr)
    return table


def _write_parquet_shard(table: pa.Table, output_root: Path, shard_idx: int) -> None:
    chunk_idx = shard_idx // 1000
    file_idx = shard_idx % 1000
    out_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"file-{file_idx:03d}.parquet"
    pq.write_table(table, out_path)


def _merge_datasets(args: argparse.Namespace) -> None:
    source_specs = _parse_source_specs(args.source_root)
    if not source_specs:
        raise ValueError("at least one --source_root is required")

    max_episodes = args.max_episodes or []
    if max_episodes and len(max_episodes) != len(source_specs):
        raise ValueError("--max_episodes must be omitted or match the number of --source_root entries")
    max_frames = args.max_frames or []
    if max_frames and len(max_frames) != len(source_specs):
        raise ValueError("--max_frames must be omitted or match the number of --source_root entries")

    task_table = _read_tasks_table(Path(args.task_source).resolve())
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output_root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    info_template = _load_json(source_specs[0][0] / "meta" / "info.json")
    features = info_template.get("features", {})
    fps = float(info_template.get("fps", 10.0))
    robot_type = str(info_template.get("robot_type", "panda"))
    chunks_size = int(info_template.get("chunks_size", 1000))

    global_episode_offset = 0
    global_index_offset = 0
    shard_idx = 0
    total_frames = 0
    total_episodes = 0
    action_rows: List[np.ndarray] = []
    manifest_sources: List[dict] = []

    for spec_idx, (source_root, source_name) in enumerate(source_specs):
        selected_eps = _select_episode_ids_with_caps(
            source_root,
            max_episodes=max_episodes[spec_idx] if max_episodes else None,
            max_frames=max_frames[spec_idx] if max_frames else None,
            seed=args.seed + spec_idx,
        )
        ep_map = {old_ep: global_episode_offset + new_ep for new_ep, old_ep in enumerate(selected_eps)}
        manifest_sources.append({
            "name": source_name,
            "root": str(source_root),
            "selected_episodes": len(selected_eps),
            "max_episodes": None if not max_episodes else max_episodes[spec_idx],
            "max_frames": None if not max_frames else max_frames[spec_idx],
        })

        for pf in _iter_parquet_files(source_root):
            table = pq.read_table(pf)
            mask = pc.is_in(table["episode_index"], value_set=pa.array(selected_eps, type=pa.int64()))
            filtered = table.filter(mask)
            if filtered.num_rows == 0:
                continue

            old_eps = filtered["episode_index"].to_pylist()
            new_eps = [ep_map[int(ep)] for ep in old_eps]
            filtered = _replace_column(filtered, "episode_index", new_eps)

            if filtered.schema.get_field_index("index") >= 0:
                new_index = range(global_index_offset, global_index_offset + filtered.num_rows)
                filtered = _replace_column(filtered, "index", new_index)
                global_index_offset += filtered.num_rows

            _write_parquet_shard(filtered, output_root, shard_idx)
            shard_idx += 1
            total_frames += filtered.num_rows
            action_rows.append(np.asarray(filtered["action"].to_pylist(), dtype=np.float32))

        global_episode_offset += len(selected_eps)
        total_episodes += len(selected_eps)

    merged_actions = np.concatenate(action_rows, axis=0)
    action_stats = {
        "min": merged_actions.min(axis=0).tolist(),
        "max": merged_actions.max(axis=0).tolist(),
        "mean": merged_actions.mean(axis=0).tolist(),
        "std": merged_actions.std(axis=0).tolist(),
        "count": [int(merged_actions.shape[0])],
        "q01": np.quantile(merged_actions, 0.01, axis=0).tolist(),
        "q10": np.quantile(merged_actions, 0.10, axis=0).tolist(),
        "q50": np.quantile(merged_actions, 0.50, axis=0).tolist(),
        "q90": np.quantile(merged_actions, 0.90, axis=0).tolist(),
        "q99": np.quantile(merged_actions, 0.99, axis=0).tolist(),
    }

    _write_tasks_table(task_table, output_root / "meta" / "tasks.parquet")
    _write_json(output_root / "meta" / "stats.json", {"action": action_stats})
    _write_json(
        output_root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": task_table.num_rows,
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": DEFAULT_DATA_PATH,
            "source": "merged from multiple dataset roots",
            "features": features,
        },
    )
    _write_json(
        output_root / "meta" / "prepare_manifest.json",
        {
            "mode": "merge",
            "sources": manifest_sources,
            "seed": args.seed,
            "total_frames": total_frames,
            "total_episodes": total_episodes,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LIBERO datasets for OpenVLA full fine-tuning")
    sub = parser.add_subparsers(dest="command", required=True)

    hydrate = sub.add_parser("hydrate", help="create a trainable GT-only dataset root")
    hydrate.add_argument("--source_root", required=True, help="dataset root to hydrate")
    hydrate.add_argument("--output_root", required=True, help="output hydrated dataset root")
    hydrate.add_argument(
        "--task_source",
        required=True,
        help="dataset root or task metadata file used to populate tasks.parquet",
    )
    hydrate.add_argument(
        "--copy_data",
        action="store_true",
        help="copy parquet data instead of creating symlinks",
    )

    merge = sub.add_parser("merge", help="merge multiple dataset roots into one trainable root")
    merge.add_argument(
        "--source_root",
        action="append",
        required=True,
        help="dataset root to merge; optional alias format: /path/to/root::name",
    )
    merge.add_argument(
        "--max_episodes",
        action="append",
        type=int,
        help="optional per-source episode cap; repeat once per --source_root",
    )
    merge.add_argument(
        "--max_frames",
        action="append",
        type=int,
        help="optional per-source frame cap; repeat once per --source_root",
    )
    merge.add_argument("--output_root", required=True, help="output merged dataset root")
    merge.add_argument(
        "--task_source",
        required=True,
        help="dataset root or task metadata file used to populate tasks.parquet",
    )
    merge.add_argument("--seed", type=int, default=42, help="random seed for per-source episode sampling")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "hydrate":
        _hydrate_dataset(args)
    elif args.command == "merge":
        _merge_datasets(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge one or more LeRobot v3 datasets into one v3 dataset.

The data produced by the current CR3/O6 collector is already LeRobot v3 on a
per-session basis.  This tool combines those session directories while
rewriting episode, frame, task, data-file, and video-file references.

Examples:

    # Discover v3 datasets directly below ``ceshi``.
    python scripts/merge_lerobot_v3.py \
        --input-root /home/je/code/lerobot/dataset/ceshi \
        --output /home/je/code/lerobot/dataset/cr5_o6_20260912_merged

    # Validate and print what would be merged without writing files.
    python scripts/merge_lerobot_v3.py \
        --input-root /home/je/code/lerobot/dataset/ceshi \
        --dry-run

The script refuses to overwrite an existing output directory.  The source
datasets are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("merge_lerobot_v3")
CHUNK_RE = re.compile(r"chunk-(\d+)$")
FILE_RE = re.compile(r"file-(\d+)$")
STATS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def numbered_path_key(path: Path) -> tuple[int, int, str]:
    """Sort chunk/file paths numerically, including non-standard names safely."""
    chunk_match = CHUNK_RE.match(path.parent.name)
    file_match = FILE_RE.match(path.stem)
    chunk = int(chunk_match.group(1)) if chunk_match else -1
    file = int(file_match.group(1)) if file_match else -1
    return chunk, file, str(path)


def parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("chunk-*/file-*.parquet"), key=numbered_path_key)
    if not files:
        raise RuntimeError(f"No parquet files found under {directory}")
    return files


def video_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.glob("chunk-*/*")
        if path.is_file() and path.stem.startswith("file-")
    ]
    return sorted(files, key=numbered_path_key)


def discover_sources(input_root: Path) -> list[Path]:
    """Accept either one dataset directory or a directory of datasets."""
    if (input_root / "meta" / "info.json").is_file():
        return [input_root]

    sources = sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir() and (path / "meta" / "info.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not sources:
        raise RuntimeError(
            f"No LeRobot datasets found directly below {input_root}. "
            "Each source must contain meta/info.json."
        )
    return sources


def normalize_task_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def read_source(source: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    info = load_json(source / "meta" / "info.json")
    data = pd.concat(
        [pd.read_parquet(path) for path in parquet_files(source / "data")],
        ignore_index=True,
    )
    episodes = pd.concat(
        [pd.read_parquet(path) for path in parquet_files(source / "meta" / "episodes")],
        ignore_index=True,
    )

    expected_frames = int(info["total_frames"])
    expected_episodes = int(info["total_episodes"])
    if len(data) != expected_frames:
        raise RuntimeError(f"{source}: info says {expected_frames} frames, found {len(data)}")
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"{source}: info says {expected_episodes} episodes, found {len(episodes)}"
        )
    return info, data, episodes


def validate_schema(sources: list[Path], infos: list[dict[str, Any]], dataframes: list[pd.DataFrame]) -> None:
    reference = infos[0]
    if reference.get("codebase_version") != "v3.0":
        raise RuntimeError(f"{sources[0]} is not LeRobot v3 (codebase_version is not v3.0)")

    reference_features = reference["features"]
    reference_columns = list(dataframes[0].columns)
    for source, info, data in zip(sources, infos, dataframes, strict=True):
        if info.get("codebase_version") != "v3.0":
            raise RuntimeError(f"{source} is not LeRobot v3")
        if info.get("fps") != reference.get("fps"):
            raise RuntimeError(f"{source}: fps differs from the first source")
        if info.get("robot_type") != reference.get("robot_type"):
            raise RuntimeError(f"{source}: robot_type differs from the first source")
        if info["features"] != reference_features:
            raise RuntimeError(f"{source}: feature schema differs from the first source")
        if list(data.columns) != reference_columns:
            raise RuntimeError(f"{source}: data parquet columns differ from the first source")


def task_texts_for_source(data: pd.DataFrame, episodes: pd.DataFrame, source: Path) -> dict[int, str]:
    """Read task text from tasks.parquet or recover it from episode metadata.

    Some collector versions wrote meta/tasks.parquet with only task_index and
    stored the actual task string in the episodes ``tasks`` column.  Both
    layouts are accepted.
    """
    task_path = source / "meta" / "tasks.parquet"
    task_table = pd.read_parquet(task_path) if task_path.is_file() else pd.DataFrame()
    result: dict[int, str] = {}

    if "task_index" not in task_table.columns:
        if not task_table.empty:
            raise RuntimeError(f"{task_path} has no task_index column")
    elif "task" in task_table.columns:
        for row in task_table.itertuples(index=False):
            result[int(row.task_index)] = str(row.task)

    data_by_episode = {
        int(episode_index): group
        for episode_index, group in data.groupby("episode_index", sort=False)
    }
    if "tasks" in episodes.columns:
        for episode in episodes.itertuples(index=False):
            old_episode = int(episode.episode_index)
            task_indices = sorted(
                int(value) for value in data_by_episode[old_episode]["task_index"].unique()
            )
            texts = normalize_task_values(episode.tasks)
            if len(task_indices) == len(texts):
                result.update(zip(task_indices, texts, strict=True))
            elif len(task_indices) == 1 and texts:
                result[task_indices[0]] = texts[0]
            elif texts:
                raise RuntimeError(
                    f"{source}, episode {old_episode}: cannot match task indices {task_indices} "
                    f"to task texts {texts}"
                )

    required = {int(value) for value in data["task_index"].unique()}
    missing = sorted(required - result.keys())
    if missing:
        raise RuntimeError(f"{source}: no task text found for task indices {missing}")
    return result


def array_matrix(values: pd.Series) -> np.ndarray:
    items = list(values)
    if not items:
        raise RuntimeError("Cannot calculate statistics for an empty series")
    try:
        array = np.asarray(items)
        if array.dtype == object:
            array = np.stack([np.asarray(item) for item in items])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Feature values are not numeric arrays") from exc
    if not np.issubdtype(array.dtype, np.number):
        raise RuntimeError("Feature values are not numeric")
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    else:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.float64, copy=False)


def calculate_stats(values: pd.Series) -> dict[str, list[Any]]:
    array = array_matrix(values)
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])] * array.shape[1],
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def update_episode_stats(row: dict[str, Any], episode_data: pd.DataFrame) -> None:
    for column in episode_data.columns:
        try:
            stats = calculate_stats(episode_data[column])
        except RuntimeError:
            continue
        for stat in STATS:
            field = f"stats/{column}/{stat}"
            if field in row:
                row[field] = stats[stat]


def rewrite_episode_index(value: Any, new_episode: int) -> Any:
    if isinstance(value, dict):
        return {
            key: rewrite_episode_index(item, new_episode)
            if key == "episode_index"
            else rewrite_episode_index(item, new_episode)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_episode_index(item, new_episode) for item in value]
    return value


def merge(input_root: Path, output: Path, dry_run: bool = False) -> None:
    sources = discover_sources(input_root)
    LOGGER.info("Found %d source dataset(s)", len(sources))
    for source in sources:
        LOGGER.info("  %s", source)

    infos: list[dict[str, Any]] = []
    dataframes: list[pd.DataFrame] = []
    episodeframes: list[pd.DataFrame] = []
    source_tasks: list[dict[int, str]] = []
    for source in sources:
        info, data, episodes = read_source(source)
        infos.append(info)
        dataframes.append(data)
        episodeframes.append(episodes)
        source_tasks.append(task_texts_for_source(data, episodes, source))

    validate_schema(sources, infos, dataframes)
    reference_info = infos[0]
    video_keys = [
        name.removeprefix("observation.images.")
        for name, feature in reference_info["features"].items()
        if name.startswith("observation.images.") and feature.get("dtype") == "video"
    ]
    for source in sources:
        for video_key in video_keys:
            if not video_files(source / "videos" / f"observation.images.{video_key}"):
                raise RuntimeError(f"{source}: no video files for {video_key}")

    merged_task_indices: dict[str, int] = {}
    source_task_maps: list[dict[int, int]] = []
    for task_map in source_tasks:
        local_to_global: dict[int, int] = {}
        for local_index, text in sorted(task_map.items()):
            if text not in merged_task_indices:
                merged_task_indices[text] = len(merged_task_indices)
            local_to_global[local_index] = merged_task_indices[text]
        source_task_maps.append(local_to_global)

    merged_data: list[pd.DataFrame] = []
    merged_episodes: list[dict[str, Any]] = []
    video_file_maps: list[dict[str, dict[tuple[int, int], int]]] = []
    video_offsets = {video_key: 0 for video_key in video_keys}
    frame_offset = 0
    episode_offset = 0

    for source_index, (source, data, episodes) in enumerate(
        zip(sources, dataframes, episodeframes, strict=True)
    ):
        data = data.copy()
        old_indices = data["index"].astype(np.int64).tolist()
        if len(set(old_indices)) != len(old_indices):
            raise RuntimeError(f"{source}: data index contains duplicates")
        old_index_end = max(old_indices) + 1
        index_map = {
            old: frame_offset + position
            for position, old in enumerate(old_indices)
        }
        episode_map = {
            int(old): episode_offset + position
            for position, old in enumerate(episodes["episode_index"].tolist())
        }

        data["index"] = np.arange(frame_offset, frame_offset + len(data), dtype=np.int64)
        data["episode_index"] = data["episode_index"].map(episode_map)
        data["task_index"] = data["task_index"].map(source_task_maps[source_index])
        if data[["episode_index", "task_index"]].isna().any().any():
            raise RuntimeError(f"{source}: episode_index or task_index could not be remapped")
        data["episode_index"] = data["episode_index"].astype(np.int64)
        data["task_index"] = data["task_index"].astype(np.int64)
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].astype(np.float32)
        if "frame_index" in data:
            data["frame_index"] = data["frame_index"].astype(np.int64)
        merged_data.append(data)

        source_video_map: dict[str, dict[tuple[int, int], int]] = {}
        for video_key in video_keys:
            source_video_map[video_key] = {}
            video_dir = source / "videos" / f"observation.images.{video_key}"
            for local_index, path in enumerate(video_files(video_dir)):
                chunk_match = CHUNK_RE.match(path.parent.name)
                file_match = FILE_RE.match(path.stem)
                if chunk_match is None or file_match is None:
                    raise RuntimeError(f"Unexpected video path: {path}")
                source_video_map[video_key][
                    (int(chunk_match.group(1)), int(file_match.group(1)))
                ] = video_offsets[video_key] + local_index
            video_offsets[video_key] += len(source_video_map[video_key])
        video_file_maps.append(source_video_map)

        for episode in episodes.to_dict(orient="records"):
            old_episode = int(episode["episode_index"])
            new_episode = episode_map[old_episode]
            row = copy.deepcopy(episode)
            row["episode_index"] = new_episode
            row["data/chunk_index"] = 0
            row["data/file_index"] = 0
            old_from_index = int(episode["dataset_from_index"])
            old_to_index = int(episode["dataset_to_index"])
            try:
                row["dataset_from_index"] = index_map[old_from_index]
            except KeyError as exc:
                raise RuntimeError(
                    f"{source}, episode {old_episode}: dataset_from_index "
                    f"{old_from_index} is not present in data"
                ) from exc
            # LeRobot stores dataset_to_index as an exclusive boundary.  The
            # final episode therefore commonly points one past the last row.
            if old_to_index == old_index_end:
                row["dataset_to_index"] = frame_offset + len(data)
            else:
                try:
                    row["dataset_to_index"] = index_map[old_to_index]
                except KeyError as exc:
                    raise RuntimeError(
                        f"{source}, episode {old_episode}: dataset_to_index "
                        f"{old_to_index} is not present or a valid exclusive boundary"
                    ) from exc
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0

            episode_data = merged_data[-1][merged_data[-1]["episode_index"] == new_episode]
            update_episode_stats(row, episode_data)

            task_indices = sorted(int(value) for value in episode_data["task_index"].unique())
            row["tasks"] = [
                next(text for text, index in merged_task_indices.items() if index == task_index)
                for task_index in task_indices
            ]
            for video_key in video_keys:
                prefix = f"videos/observation.images.{video_key}"
                old_reference = (
                    int(episode[f"{prefix}/chunk_index"]),
                    int(episode[f"{prefix}/file_index"]),
                )
                try:
                    new_file_index = source_video_map[video_key][old_reference]
                except KeyError as exc:
                    raise RuntimeError(
                        f"{source}, episode {old_episode}: video reference {old_reference} "
                        f"for {video_key} does not exist"
                    ) from exc
                row[f"{prefix}/chunk_index"] = 0
                row[f"{prefix}/file_index"] = new_file_index
            merged_episodes.append(row)

        frame_offset += len(data)
        episode_offset += len(episodes)

    all_data = pd.concat(merged_data, ignore_index=True)
    all_episodes = pd.DataFrame(merged_episodes).sort_values("episode_index").reset_index(drop=True)
    LOGGER.info(
        "Validated merge: %d episodes, %d frames, %d tasks",
        len(all_episodes),
        len(all_data),
        len(merged_task_indices),
    )
    if dry_run:
        return

    if output.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {output}")

    output.mkdir(parents=True)
    (output / "data" / "chunk-000").mkdir(parents=True)
    (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output / "meta" / "collection").mkdir(parents=True)
    for video_key in video_keys:
        (output / "videos" / f"observation.images.{video_key}" / "chunk-000").mkdir(
            parents=True
        )

    all_data.to_parquet(output / "data" / "chunk-000" / "file-000.parquet", index=False)
    all_episodes.to_parquet(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    pd.DataFrame(
        [
            {"task_index": index, "task": task}
            for task, index in sorted(merged_task_indices.items(), key=lambda item: item[1])
        ]
    ).to_parquet(output / "meta" / "tasks.parquet", index=False)

    for source_index, source in enumerate(sources):
        for video_key in video_keys:
            source_dir = source / "videos" / f"observation.images.{video_key}"
            target_dir = output / "videos" / f"observation.images.{video_key}" / "chunk-000"
            for path in video_files(source_dir):
                new_index = video_file_maps[source_index][video_key][
                    (int(CHUNK_RE.match(path.parent.name).group(1)), int(FILE_RE.match(path.stem).group(1)))
                ]
                shutil.copy2(path, target_dir / f"file-{new_index:03d}{path.suffix}")

        source_episodes = episodeframes[source_index]
        episode_map = {
            int(old): sum(len(frame) for frame in episodeframes[:source_index]) + position
            for position, old in enumerate(source_episodes["episode_index"].tolist())
        }
        for old_episode, new_episode in episode_map.items():
            source_json = source / "meta" / "collection" / f"episode_{old_episode:06d}.json"
            if source_json.is_file():
                save_json(
                    output / "meta" / "collection" / f"episode_{new_episode:06d}.json",
                    rewrite_episode_index(load_json(source_json), new_episode),
                )

    output_info = copy.deepcopy(reference_info)
    output_info.update(
        {
            "total_episodes": len(all_episodes),
            "total_frames": len(all_data),
            "total_tasks": len(merged_task_indices),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": f"0:{len(all_episodes)}"},
        }
    )
    save_json(output / "meta" / "info.json", output_info)

    output_stats = copy.deepcopy(load_json(sources[0] / "meta" / "stats.json"))
    for column in all_data.columns:
        output_stats[column] = calculate_stats(all_data[column])
    save_json(output / "meta" / "stats.json", output_stats)
    LOGGER.info("Wrote merged dataset to %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Dataset or directory containing datasets")
    parser.add_argument("--output", type=Path, help="Output dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing output")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output = args.output or args.input_root.with_name(f"{args.input_root.name}_lerobot_v3")
    merge(args.input_root, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

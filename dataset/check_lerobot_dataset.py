#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================

DATASET_ROOT = Path(
    "/home/je/code/lerobot/dataset/cr5_o6_20260915_172625_722385566"
)


# ============================================================
# 工具函数
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_finite_array(name, arr):
    arr = np.asarray(arr)

    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()

    if nan_count == 0 and inf_count == 0:
        print(f"  [OK] {name}: no NaN / Inf")
        return True

    print(f"  [ERROR] {name}: NaN={nan_count}, Inf={inf_count}")
    return False


# ============================================================
# 1. 检查目录结构
# ============================================================

section("1. Dataset directory structure")

required_paths = [
    "data",
    "meta",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes",
    "meta/collection",
    "videos",
]

all_ok = True

for p in required_paths:
    path = DATASET_ROOT / p

    if path.exists():
        print(f"  [OK]      {p}")
    else:
        print(f"  [MISSING] {p}")
        all_ok = False


# ============================================================
# 2. 读取 info.json
# ============================================================

section("2. info.json")

info_path = DATASET_ROOT / "meta/info.json"

with open(info_path, "r") as f:
    info = json.load(f)

print("  codebase_version :", info.get("codebase_version"))
print("  robot_type       :", info.get("robot_type"))
print("  fps              :", info.get("fps"))
print("  total_episodes   :", info.get("total_episodes"))
print("  total_frames     :", info.get("total_frames"))
print("  total_tasks      :", info.get("total_tasks"))
print("  chunks_size      :", info.get("chunks_size"))

if info.get("codebase_version") == "v3.0":
    print("  [OK] LeRobot V3")
else:
    print("  [WARN] codebase_version is not v3.0")


# ============================================================
# 3. 检查 features
# ============================================================

section("3. Dataset features")

features = info.get("features", {})

for name, feature in features.items():

    dtype = feature.get("dtype")
    shape = feature.get("shape")
    names = feature.get("names")

    print(f"\n  {name}")
    print(f"    dtype : {dtype}")
    print(f"    shape : {shape}")

    if names:
        for i, n in enumerate(names):
            print(f"      [{i:02d}] {n}")


# ============================================================
# 4. 检查 parquet
# ============================================================

section("4. Parquet data")

data_dir = DATASET_ROOT / "data"

parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

print("  parquet files:", len(parquet_files))

if not parquet_files:
    raise RuntimeError("No parquet files found.")

for p in parquet_files:
    print("   ", p.relative_to(DATASET_ROOT))

# 为了完整检查，这里读取所有 parquet
dfs = []

for p in parquet_files:
    df_part = pd.read_parquet(p)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

print("\n  combined shape:", df.shape)
print("  columns:")

for c in df.columns:
    print("   ", c)


# ============================================================
# 5. state/action shape
# ============================================================

section("5. State / Action shape")

for key in ["observation.state", "action"]:

    if key not in df.columns:
        print(f"  [ERROR] missing column: {key}")
        all_ok = False
        continue

    sample = df[key].iloc[0]

    print(f"\n  {key}")
    print("    python type :", type(sample))
    print("    shape       :", getattr(sample, "shape", None))

    expected_shape = tuple(
        features[key]["shape"]
    )

    actual_shape = tuple(np.asarray(sample).shape)

    if actual_shape == expected_shape:
        print(f"    [OK] matches info.json shape {expected_shape}")
    else:
        print(
            f"    [ERROR] expected {expected_shape}, "
            f"got {actual_shape}"
        )
        all_ok = False


# ============================================================
# 6. 检查 state/action NaN / Inf
# ============================================================

section("6. State / Action NaN / Inf")

for key in ["observation.state", "action"]:

    values = np.stack(df[key].to_numpy())

    print(f"\n  {key}")
    print("    shape:", values.shape)

    check_finite_array(key, values)

    print("    global min:", np.min(values))
    print("    global max:", np.max(values))


# ============================================================
# 7. 检查 stats.json
# ============================================================

section("7. stats.json")

stats_path = DATASET_ROOT / "meta/stats.json"

with open(stats_path, "r") as f:
    stats = json.load(f)

print("  stats keys:")

for k in stats:
    print("   ", k)


for key in ["observation.state", "action"]:

    print(f"\n  ---- {key} ----")

    if key not in stats:
        print(f"  [ERROR] {key} missing from stats.json")
        all_ok = False
        continue

    s = stats[key]

    mean = np.asarray(s["mean"], dtype=np.float64)
    std = np.asarray(s["std"], dtype=np.float64)
    min_v = np.asarray(s["min"], dtype=np.float64)
    max_v = np.asarray(s["max"], dtype=np.float64)

    print("  dimension:", len(mean))

    print("  std:")
    for i, v in enumerate(std):
        print(f"    [{i:02d}] {v:.10g}")

    zero_std = np.where(std == 0)[0]

    if len(zero_std) == 0:
        print("  [OK] no std=0 dimensions")
    else:
        print(
            "  [WARN] std=0 dimensions:",
            zero_std.tolist()
        )

        names = features[key].get("names")

        if names:
            for i in zero_std:
                print(
                    f"         [{i:02d}] {names[i]}"
                )

    check_finite_array(f"{key}.mean", mean)
    check_finite_array(f"{key}.std", std)
    check_finite_array(f"{key}.min", min_v)
    check_finite_array(f"{key}.max", max_v)

    # 检查 max < min
    if np.any(max_v < min_v):
        print("  [ERROR] max < min detected")
        all_ok = False
    else:
        print("  [OK] min/max relationship")


# ============================================================
# 8. 比较 parquet 实际统计值和 stats.json
# ============================================================

section("8. Compare parquet statistics with stats.json")

for key in ["observation.state", "action"]:

    values = np.stack(df[key].to_numpy()).astype(np.float64)

    actual_min = values.min(axis=0)
    actual_max = values.max(axis=0)
    actual_mean = values.mean(axis=0)
    actual_std = values.std(axis=0)

    stats_min = np.asarray(stats[key]["min"], dtype=np.float64)
    stats_max = np.asarray(stats[key]["max"], dtype=np.float64)
    stats_mean = np.asarray(stats[key]["mean"], dtype=np.float64)
    stats_std = np.asarray(stats[key]["std"], dtype=np.float64)

    names = features[key].get("names")

    print(f"\n  {key}")

    print(
        f"  {'dim':>3} "
        f"{'name':<35} "
        f"{'actual_mean':>12} "
        f"{'stats_mean':>12} "
        f"{'actual_std':>12} "
        f"{'stats_std':>12}"
    )

    for i in range(len(actual_mean)):

        name = names[i] if names else f"dim_{i}"

        print(
            f"  {i:3d} "
            f"{name:<35} "
            f"{actual_mean[i]:12.5f} "
            f"{stats_mean[i]:12.5f} "
            f"{actual_std[i]:12.5f} "
            f"{stats_std[i]:12.5f}"
        )


# ============================================================
# 9. 检查 episode
# ============================================================

section("9. Episode structure")

episode = df["episode_index"].to_numpy()

unique_episodes = np.unique(episode)

print("  episode count:", len(unique_episodes))
print("  first episode:", unique_episodes[0])
print("  last episode :", unique_episodes[-1])

expected_episodes = info["total_episodes"]

if len(unique_episodes) == expected_episodes:
    print("  [OK] episode count matches info.json")
else:
    print(
        f"  [ERROR] expected {expected_episodes}, "
        f"got {len(unique_episodes)}"
    )
    all_ok = False


expected_indices = np.arange(expected_episodes)

if np.array_equal(unique_episodes, expected_indices):
    print("  [OK] episode indices are continuous")
else:
    print("  [WARN] episode indices are not continuous")


# 每个 episode 的 frame 数
episode_counts = df.groupby("episode_index").size()

print("\n  episode frame statistics:")
print("    min:", episode_counts.min())
print("    max:", episode_counts.max())
print("    mean:", episode_counts.mean())

print("\n  first 10 episodes:")

for ep, count in episode_counts.head(10).items():
    print(f"    episode {ep:03d}: {count} frames")


# ============================================================
# 10. 检查 frame_index
# ============================================================

section("10. Frame index continuity")

bad_episodes = []

for ep, group in df.groupby("episode_index"):

    frames = group["frame_index"].to_numpy()

    expected = np.arange(len(frames))

    if not np.array_equal(frames, expected):
        bad_episodes.append(ep)

if not bad_episodes:
    print("  [OK] frame_index is continuous inside every episode")
else:
    print(
        "  [WARN] non-continuous frame_index episodes:",
        bad_episodes
    )


# ============================================================
# 11. 检查 timestamp
# ============================================================

section("11. Timestamp")

timestamp = df["timestamp"].to_numpy()

print("  global min:", timestamp.min())
print("  global max:", timestamp.max())

timestamp_issues = []

for ep, group in df.groupby("episode_index"):

    t = group["timestamp"].to_numpy()

    if np.any(np.diff(t) < 0):
        timestamp_issues.append(ep)

if not timestamp_issues:
    print("  [OK] timestamp is monotonically increasing")
else:
    print(
        "  [WARN] timestamp decreases in episodes:",
        timestamp_issues
    )


# ============================================================
# 12. 检查 20 FPS
# ============================================================

section("12. FPS consistency")

fps = info.get("fps")

print("  dataset fps:", fps)

if fps == 20:
    print("  [OK] dataset fps = 20")
else:
    print("  [WARN] dataset fps != 20")


# ============================================================
# 13. 检查视频
# ============================================================

section("13. Video files")

video_keys = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]

for key in video_keys:

    video_dir = DATASET_ROOT / "videos" / key

    files = sorted(video_dir.glob("chunk-*/file-*.mp4"))

    print(f"\n  {key}")
    print("    files:", len(files))

    if not files:
        print("    [ERROR] no mp4 files")
        all_ok = False
        continue

    total_size = sum(p.stat().st_size for p in files)

    print(
        "    total size:",
        f"{total_size / 1024 / 1024:.2f} MB"
    )

    for p in files:
        print(
            "     ",
            p.relative_to(DATASET_ROOT),
            f"{p.stat().st_size / 1024 / 1024:.2f} MB"
        )


# ============================================================
# 14. Absolute action vs Delta action
# ============================================================

section("14. Absolute action vs Delta action")

state = np.stack(
    df["observation.state"].to_numpy()
).astype(np.float64)

action = np.stack(
    df["action"].to_numpy()
).astype(np.float64)


# state[t+1] - state[t]
state_delta = state[1:] - state[:-1]

# action[t] - state[t]
action_minus_state = action[:-1] - state[:-1]

names = features["action"].get("names")

print(
    "\n  Compare:"
    "\n    A = action[t] - state[t]"
    "\n    B = state[t+1] - state[t]"
)

print(
    f"\n  {'dim':>3} "
    f"{'name':<35} "
    f"{'mean|A|':>12} "
    f"{'mean|B|':>12}"
)

for i in range(action.shape[1]):

    a = np.mean(np.abs(action_minus_state[:, i]))
    b = np.mean(np.abs(state_delta[:, i]))

    name = names[i] if names else f"dim_{i}"

    print(
        f"  {i:3d} "
        f"{name:<35} "
        f"{a:12.6f} "
        f"{b:12.6f}"
    )


# 判断 action 是否接近 state
mean_abs_action_state = np.mean(np.abs(action_minus_state))
mean_abs_delta = np.mean(np.abs(state_delta))

print("\n  Overall:")
print("    mean |action - state| :", mean_abs_action_state)
print("    mean |state[t+1]-state[t]| :", mean_abs_delta)

if mean_abs_action_state < mean_abs_delta * 0.5:
    print(
        "  [INFO] action appears closer to absolute/target "
        "state than to delta."
    )
else:
    print(
        "  [INFO] action is not obviously identical to absolute state."
    )


# ============================================================
# 15. 检查 state/action 是否存在严重错位
# ============================================================

section("15. State / Action alignment")

diff = np.abs(action - state)

print("  mean |action-state| per dimension:")

for i in range(diff.shape[1]):

    name = names[i] if names else f"dim_{i}"

    print(
        f"    [{i:02d}] {name:<35} "
        f"{diff[:, i].mean():.6f}"
    )


# ============================================================
# 16. 检查常数维
# ============================================================

section("16. Constant dimensions")

for key in ["observation.state", "action"]:

    values = np.stack(df[key].to_numpy()).astype(np.float64)

    std = values.std(axis=0)

    names = features[key].get("names")

    print(f"\n  {key}")

    constant_dims = np.where(std == 0)[0]

    if len(constant_dims) == 0:
        print("    [OK] no constant dimensions")
    else:
        print(
            "    [WARN] constant dimensions:",
            constant_dims.tolist()
        )

        for i in constant_dims:
            print(
                f"      [{i:02d}] "
                f"{names[i] if names else ''} = "
                f"{values[0, i]}"
            )


# ============================================================
# 17. 检查 normalization 后是否产生 NaN
# ============================================================

section("17. Simulated normalization check")

print(
    "  This is only a mathematical check."
    "\n  It does NOT modify the dataset."
)

for key in ["observation.state", "action"]:

    values = np.stack(df[key].to_numpy()).astype(np.float64)

    mean = np.asarray(
        stats[key]["mean"],
        dtype=np.float64
    )

    std = np.asarray(
        stats[key]["std"],
        dtype=np.float64
    )

    # 使用一个安全的 std，避免除以 0
    safe_std = np.where(std == 0, 1.0, std)

    normalized = (values - mean) / safe_std

    print(f"\n  {key}")

    print("    normalized min:", normalized.min())
    print("    normalized max:", normalized.max())

    if np.isnan(normalized).any():
        print("    [ERROR] normalized data contains NaN")
        all_ok = False
    elif np.isinf(normalized).any():
        print("    [ERROR] normalized data contains Inf")
        all_ok = False
    else:
        print("    [OK] normalized data contains no NaN / Inf")


# ============================================================
# 18. 最终总结
# ============================================================

section("FINAL SUMMARY")

if all_ok:
    print("  [PASS] Basic dataset integrity checks passed.")
else:
    print("  [WARN] Some checks reported errors.")

print("\n  Important observations:")

# robot type
print(
    f"  - robot_type = {info.get('robot_type')}"
)

# state/action
print(
    f"  - observation.state = {info['features']['observation.state']['shape']}"
)

print(
    f"  - action = {info['features']['action']['shape']}"
)

# zero std
for key in ["observation.state", "action"]:

    std = np.asarray(
        stats[key]["std"],
        dtype=np.float64
    )

    zero_std = np.where(std == 0)[0]

    if len(zero_std):
        names = features[key].get("names")

        print(
            f"  - {key} has std=0 dimensions: "
            + ", ".join(
                f"{i}:{names[i] if names else ''}"
                for i in zero_std
            )
        )

print(
    "\n  No training was started."
)
print(
    "  No dataset files were modified."
)

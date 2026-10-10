#!/usr/bin/env python3

"""Deploy an OpenPI policy trained with the CR3+O6 joint-target dataset.

Dataset schema:

* state: ``cr3.q1..q6.rad`` followed by six O6 positions (12 values)
* action: ``cr3.target_q1..q6.rad`` followed by six O6 commands (12 values)

The CR3 actions are absolute joint targets in radians.  The ROS shared-memory
bridge accepts joint targets in degrees, so this file performs exactly that
conversion before sending command 7.  It deliberately does not reuse the
TCP-delta action path from ``deploy_dobot_openpi.py``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the standalone OpenPI-compatible hardware bridge and camera monitor.
# Importing it also configures the OpenPI source paths before OpenPI imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_dobot_openpi as _base

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


CHECKPOINT_ROOT = Path(
    os.environ.get(
        "OPENPI_CHECKPOINT",
        "/home/je/code/lerobot/dataset/cr5_o6_20260909_200040_349965059_result",
    )
)
CHECKPOINT_STEP = os.environ.get("OPENPI_CHECKPOINT_STEP")

DATASET_ROOT = Path("/home/je/code/lerobot/dataset/cr5_o6_20260909_200040_349965059")
DATASET_STATS_PATH = DATASET_ROOT / "meta" / "stats.json"
ASSET_ID = DATASET_ROOT.name
TASK_PROMPT = "Pick up the motor and place it on the right side with the protruding side facing left."

STATE_NAMES = (
    "cr3.q1.rad",
    "cr3.q2.rad",
    "cr3.q3.rad",
    "cr3.q4.rad",
    "cr3.q5.rad",
    "cr3.q6.rad",
    "o6.thumb_flex.position",
    "o6.thumb_yaw.position",
    "o6.index_flex.position",
    "o6.middle_flex.position",
    "o6.ring_flex.position",
    "o6.little_flex.position",
)
ACTION_NAMES = (
    "cr3.target_q1.rad",
    "cr3.target_q2.rad",
    "cr3.target_q3.rad",
    "cr3.target_q4.rad",
    "cr3.target_q5.rad",
    "cr3.target_q6.rad",
    "o6.thumb_flex.command",
    "o6.thumb_yaw.command",
    "o6.index_flex.command",
    "o6.middle_flex.command",
    "o6.ring_flex.command",
    "o6.little_flex.command",
)

CONTROL_FPS = 20.0
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 500
REPLAN_STEPS = 1
DRY_RUN = False
SHOW_CAMERA_WINDOW = True
CAMERA_WINDOW_NAME = "Dobot CR3 + O6 | OpenPI joint target"

# Safety envelope from the new dataset's observed action min/max.  Predictions
# outside the collection envelope are clipped before conversion to degrees.
JOINT_MIN_RAD = np.array(
    [-0.4524068, -1.3138773, -0.7545326, -0.2983725, -1.1004673, -2.1733513],
    dtype=np.float64,
)
JOINT_MAX_RAD = np.array(
    [0.6747624, 0.2520110, 1.2331995, 2.5263002, 0.9260743, 2.4551303],
    dtype=np.float64,
)
ENABLE_DATASET_JOINT_LIMITS = True

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def _latest_checkpoint(root: Path) -> Path:
    if CHECKPOINT_STEP:
        path = root / CHECKPOINT_STEP
        if not (path / "params").is_dir():
            raise FileNotFoundError(f"OpenPI params directory not found: {path / 'params'}")
        return path
    if (root / "params").is_dir():
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "params").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No OpenPI checkpoint found at {root}. Set OPENPI_CHECKPOINT to the trained model directory."
        )
    return max(candidates, key=lambda path: int(path.name))


def _load_dataset_norm_stats() -> dict[str, _normalize.NormStats]:
    """Convert LeRobot v3 stats.json to OpenPI's state/actions format."""
    if not DATASET_STATS_PATH.is_file():
        raise FileNotFoundError(f"Dataset stats not found: {DATASET_STATS_PATH}")
    raw = json.loads(DATASET_STATS_PATH.read_text())

    def make_stats(key: str) -> _normalize.NormStats:
        values = raw[key]
        return _normalize.NormStats(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
            q01=np.asarray(values["q01"], dtype=np.float32),
            q99=np.asarray(values["q99"], dtype=np.float32),
        )

    stats = {
        "state": make_stats("observation.state"),
        "actions": make_stats("action"),
    }
    if stats["state"].mean.shape != (12,) or stats["actions"].mean.shape != (12,):
        raise ValueError(f"Expected 12-dimensional state/actions stats, got {stats}")
    logger.info("Loaded LeRobot dataset stats: state=(12,), actions=(12,)")
    return stats


def _as_rgb_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 camera image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class Cr3JointInputs(_transforms.DataTransformFn):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        return {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _as_rgb_uint8(image["base_0_rgb"]),
                "left_wrist_0_rgb": _as_rgb_uint8(image["left_wrist_0_rgb"]),
                "right_wrist_0_rgb": _as_rgb_uint8(image["right_wrist_0_rgb"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data.get("prompt", TASK_PROMPT),
        }


@dataclasses.dataclass(frozen=True)
class Cr3JointOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(data["actions"][..., :12])}


@dataclasses.dataclass(frozen=True)
class Cr3JointDataConfig(_config.DataConfigFactory):
    repo_id: str = ASSET_ID
    assets: _config.AssetsConfig = dataclasses.field(
        default_factory=lambda: _config.AssetsConfig(asset_id=ASSET_ID)
    )

    def create(
        self,
        assets_dirs: Path,
        model_config: _model.BaseModelConfig,
    ) -> _config.DataConfig:
        del assets_dirs
        return _config.DataConfig(
            repo_id=self.repo_id,
            asset_id=ASSET_ID,
            data_transforms=_transforms.Group(
                inputs=[Cr3JointInputs()],
                outputs=[Cr3JointOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory(default_prompt=TASK_PROMPT)(model_config),
            use_quantile_norm=True,
            action_sequence_keys=("actions",),
        )


def _make_train_config() -> _config.TrainConfig:
    return _config.TrainConfig(
        name="cr5_o6_20260909_joint_openpi",
        exp_name="deployment",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, dtype="bfloat16"),
        data=Cr3JointDataConfig(),
        assets_base_dir=str(DATASET_ROOT),
        checkpoint_base_dir=str(CHECKPOINT_ROOT.parent),
        wandb_enabled=False,
    )


class Cr3JointHardware(_base.DobotHardware):
    """Reuse the bridge/cameras, but send absolute CR3 joint targets."""

    def send_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action shape (12,), got {action.shape}")

        target_rad = action[:6].copy()
        if ENABLE_DATASET_JOINT_LIMITS:
            target_rad = np.clip(target_rad, JOINT_MIN_RAD, JOINT_MAX_RAD)

        # ROS command 7 expects CR5/CR3 joint angles in degrees and reserves
        # the seventh value for the external axis.
        target_deg = np.concatenate((np.rad2deg(target_rad), np.array([0.0])))
        self.arm.send_joint_command(target_deg)

        hand_command = np.clip(action[6:], 0, 255).astype(int).tolist()
        self.hand.set_joint_positions(hand_command)
        self.action_count += 1
        if self.action_count == 1 or self.action_count % 20 == 0:
            logger.info(
                "Joint action #%d: target_rad=%s target_deg=%s hand=%s",
                self.action_count,
                np.array2string(target_rad, precision=5),
                np.array2string(target_deg, precision=3),
                hand_command,
            )


def _make_policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray([observation[name] for name in STATE_NAMES], dtype=np.float32)
    if state.shape != (12,):
        raise ValueError(f"Expected 12-dimensional CR3/O6 state, got {state.shape}")
    return {
        "state": state,
        "image": {
            "base_0_rgb": observation["base_0_rgb"],
            "left_wrist_0_rgb": observation["left_wrist_0_rgb"],
            "right_wrist_0_rgb": observation["right_wrist_0_rgb"],
        },
        "prompt": TASK_PROMPT,
    }


def main() -> None:
    _base._check_hardware_dependencies()
    checkpoint = _latest_checkpoint(CHECKPOINT_ROOT)
    norm_stats = _load_dataset_norm_stats()
    logger.info("Loading OpenPI CR3 joint checkpoint: %s", checkpoint)
    logger.info("Dataset: %s", DATASET_ROOT)
    logger.info("Task prompt: %s", TASK_PROMPT)

    policy = policy_config.create_trained_policy(
        _make_train_config(),
        checkpoint,
        default_prompt=TASK_PROMPT,
        norm_stats=norm_stats,
    )
    logger.info("OpenPI policy loaded. JAX devices: %s", __import__("jax").devices())

    robot = Cr3JointHardware()
    display_enabled = False
    try:
        logger.info("Connecting CR3 + O6...")
        robot.connect()
        logger.info("Robot connected.")

        if SHOW_CAMERA_WINDOW:
            try:
                import cv2

                cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
                display_enabled = True
            except Exception as exc:
                logger.warning("Camera window unavailable; continuing without it: %s", exc)

        period = 1.0 / CONTROL_FPS
        stop_requested = False
        for episode in range(MAX_EPISODES):
            action_plan: deque[np.ndarray] = deque()
            logger.info("========== Episode %d / %d ==========", episode + 1, MAX_EPISODES)

            for step in range(MAX_STEPS_PER_EPISODE):
                loop_start = time.perf_counter()
                observation = robot.get_observation()
                if not action_plan:
                    infer_start = time.perf_counter()
                    result = policy.infer(_make_policy_observation(observation))
                    action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != 12:
                        raise ValueError(f"Expected action chunk [N, 12], got {action_chunk.shape}")
                    action_plan.extend(action_chunk[:REPLAN_STEPS])
                    logger.info(
                        "OpenPI inference %.1f ms, action[0]=%s",
                        (time.perf_counter() - infer_start) * 1000.0,
                        np.array2string(action_chunk[0], precision=5),
                    )

                action = np.asarray(action_plan.popleft(), dtype=np.float64)
                if DRY_RUN:
                    logger.info("DRY_RUN=True; action was not sent: %s", action.tolist())
                else:
                    robot.send_action(action)

                if display_enabled:
                    try:
                        if not _base._show_camera_and_pose(observation, episode, step):
                            stop_requested = True
                            break
                    except Exception as exc:
                        logger.warning("Camera display failed; disabling it: %s", exc)
                        display_enabled = False

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if stop_requested:
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        if display_enabled:
            import cv2

            cv2.destroyWindow(CAMERA_WINDOW_NAME)
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()

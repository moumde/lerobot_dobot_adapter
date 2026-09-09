#!/usr/bin/env python3

import time
import logging

import cv2
import numpy as np
import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import (
    build_inference_frame,
    make_robot_action,
)

from lerobot.robots.dobot_cr5_o6.dobot_cr5_o6 import DobotCR5O6
from lerobot.robots.dobot_cr5_o6.config_dobot_cr5_o6 import DobotCR5O6RobotConfig


# ================================================================
# 配置
# ================================================================

MAX_EPISODES = 5

# 如果你的 ACT 是持续运行任务，建议先不要限制得太小
MAX_STEPS_PER_EPISODE = 500

# ACT 推理频率
CONTROL_FPS = 20.0

# 是否真正发送机器人控制命令
# 第一次验证时建议 False
DRY_RUN = False

# Keep this enabled for normal deployment.  Set to False only for a
# deliberately controlled test; the workspace guard below is independent.
ENABLE_ACTION_LIMITS = True

# ----------------
# RealSense
# ----------------

LEFT_CAMERA_SERIAL = "317222074617"
RIGHT_CAMERA_SERIAL = "254622075848"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Runtime monitor.  The third pane, base_0_rgb, is the configured ROI from
# the right camera and is the same image that is sent to the policy.
SHOW_CAMERA_WINDOW = True
CAMERA_WINDOW_NAME = "Dobot CR5 + O6 | cameras and pose"
# Each policy view is 224x224.  The 2x2 monitor is therefore rendered at
# 896x896 instead of the native 448x448 size.
CAMERA_DISPLAY_SCALE = 4.0


# ================================================================
# 日志
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    # Some LeRobot/third-party imports may configure the root logger before
    # this module is loaded.  Without force=True, basicConfig() then becomes
    # a no-op and the robot adapter's INFO logs stay hidden.
    force=True,
)

logger = logging.getLogger(__name__)


def _camera_image_to_bgr(image) -> np.ndarray:
    """Convert a LeRobot RGB observation to a displayable BGR image."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 camera image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)


def _draw_camera_and_pose_window(
    observation: dict,
    episode_idx: int,
    step_idx: int,
) -> bool:
    """Show the three policy views and the latest robot feedback.

    Returns False when the operator requests shutdown with Q or Escape.
    """
    view_names = (
        ("left_wrist_0_rgb", "LEFT WRIST"),
        ("right_wrist_0_rgb", "RIGHT WRIST"),
        ("base_0_rgb", "BASE ROI (from right)"),
    )
    views = []
    for key, title in view_names:
        if key not in observation:
            raise KeyError(f"Missing camera observation: {key}")
        image = _camera_image_to_bgr(observation[key])
        views.append((title, image))

    pane_height = max(image.shape[0] for _, image in views)
    pane_width = max(image.shape[1] for _, image in views)
    canvas = np.zeros((pane_height * 2, pane_width * 2, 3), dtype=np.uint8)

    placements = ((0, 0), (pane_width, 0), (0, pane_height))
    for (title, image), (x, y) in zip(views, placements, strict=True):
        if image.shape[:2] != (pane_height, pane_width):
            image = cv2.resize(image, (pane_width, pane_height))
        canvas[y : y + pane_height, x : x + pane_width] = image
        cv2.rectangle(
            canvas,
            (x, y),
            (x + pane_width - 1, y + pane_height - 1),
            (0, 220, 255),
            1,
        )
        cv2.putText(
            canvas,
            title,
            (x + 6, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    info_x = pane_width
    info_y = pane_height
    info = canvas[info_y:, info_x:]
    info[:] = (24, 24, 24)

    tcp_xyz_mm = [
        float(observation[f"cr5.tcp.{axis}.m"]) * 1000.0
        for axis in ("x", "y", "z")
    ]
    tcp_rpy_deg = np.rad2deg(
        [
            float(observation[f"cr5.tcp.{axis}.rad"])
            for axis in ("roll", "pitch", "yaw")
        ]
    )
    joint_deg = np.rad2deg(
        [float(observation[f"cr5.j{index}.rad"]) for index in range(1, 7)]
    )
    lines = [
        f"Episode {episode_idx + 1}  Step {step_idx}",
        "Feedback TCP",
        f"XYZ mm: {tcp_xyz_mm[0]:7.1f} {tcp_xyz_mm[1]:7.1f} {tcp_xyz_mm[2]:7.1f}",
        f"RPY deg: {tcp_rpy_deg[0]:7.1f} {tcp_rpy_deg[1]:7.1f} {tcp_rpy_deg[2]:7.1f}",
        "CR5 joints deg",
        " ".join(f"{value:6.1f}" for value in joint_deg[:3]),
        " ".join(f"{value:6.1f}" for value in joint_deg[3:]),
        "Q / ESC: stop",
    ]
    line_height = max(18, pane_height // 12)
    for line_index, line in enumerate(lines):
        cv2.putText(
            info,
            line,
            (8, 22 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    if CAMERA_DISPLAY_SCALE <= 0:
        raise ValueError("CAMERA_DISPLAY_SCALE must be positive")
    display_canvas = cv2.resize(
        canvas,
        dsize=None,
        fx=CAMERA_DISPLAY_SCALE,
        fy=CAMERA_DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imshow(CAMERA_WINDOW_NAME, display_canvas)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)


# ================================================================
# main
# ================================================================

def main():
    # ============================================================
    # 1. Device
    # ============================================================

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ============================================================
    # 2. ACT model
    # ============================================================

    model_id = "/home/je/code/lerobot/dataset/cr5_o6_motor_recognition_v2/checkpoints/last/pretrained_model"

    print(f"Loading ACT model from: {model_id}")

    model = ACTPolicy.from_pretrained(model_id)

    model.eval()

    # ============================================================
    # 3. Dataset metadata
    # ============================================================

    dataset_id = "cr5_o6"

    print(
        f"Loading dataset metadata: {dataset_id}"
    )

    dataset_metadata = LeRobotDatasetMetadata(
        dataset_id,
        root="/home/je/code/lerobot/dataset/cr5_o6_motor_recognition_merged",
    )

    # ============================================================
    # 4. Preprocess / Postprocess
    # ============================================================

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=dataset_metadata.stats,
    )

    print("Pre/post processors created.")

    # ============================================================
    # 5. Dobot CR5 + O6 configuration
    # ============================================================

    cameras_config = {

        "left_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=LEFT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),

        "right_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=RIGHT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
    }

    robot_cfg = DobotCR5O6RobotConfig(
        ip="192.168.2.14",
        cameras=cameras_config,
        enable_action_limits=ENABLE_ACTION_LIMITS,
        # Temporary dataset-envelope guard. Replace with calibrated physical
        # workspace limits before using this on a different installation.
        workspace_min_m=(0.30, -0.20, 0.05),
        workspace_max_m=(0.60, 0.02, 0.45),
    )

    # ============================================================
    # 6. 创建机器人
    # ============================================================

    robot = DobotCR5O6(robot_cfg)

    # ============================================================
    # 7. 连接
    # ============================================================

    print("Connecting Dobot CR5 + O6...")

    try:
        robot.connect()
    except Exception:
        # Do not leave cameras, the hand, or the shared-memory attachment to
        # __del__ when connection fails before the main-loop finally block.
        robot.disconnect()
        raise

    print("Robot connected.")

    display_enabled = False
    if SHOW_CAMERA_WINDOW:
        try:
            cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
            display_enabled = True
        except cv2.error as exc:
            logger.warning(
                "Camera window is unavailable; continuing without display: %s",
                exc,
            )

    # ============================================================
    # 8. 打印 observation / action features
    # ============================================================

    print("Robot observation features:")
    for key, value in robot.observation_features.items():
        print(f"  {key}: {value}")

    print("Robot action features:")
    for key, value in robot.action_features.items():
        print(f"  {key}: {value}")

    print("Dataset features:")
    for key, value in dataset_metadata.features.items():
        print(f"  {key}: {value}")

    # ============================================================
    # 9. 主循环
    # ============================================================

    period = 1.0 / CONTROL_FPS
    stop_requested = False

    try:

        for episode_idx in range(MAX_EPISODES):

            print(
                f"========== Episode {episode_idx + 1} "
                f"/ {MAX_EPISODES} =========="
            )

            for step_idx in range(MAX_STEPS_PER_EPISODE):

                loop_start = time.perf_counter()

                # =================================================
                # 9.1 获取机器人 observation
                # =================================================

                obs = robot.get_observation()

                # =================================================
                # 9.2 打印第一次 observation
                # =================================================

                if step_idx == 0:

                    print(
                        "========== First Observation =========="
                    )

                    for key, value in obs.items():

                        if hasattr(value, "shape"):
                            print(
                                f"{key}: "
                                f"shape={value.shape}, "
                                f"dtype={value.dtype}"
                            )
                        else:
                            print(
                                f"{key}: {type(value)}"
                            )

                    print(
                        "========================================"
                    )

                # =================================================
                # 9.3 构造 ACT inference frame
                # =================================================

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                # =================================================
                # 9.4 Preprocess
                # =================================================

                obs_processed = preprocess(obs_frame)

                # =================================================
                # 9.5 ACT inference
                # =================================================

                with torch.inference_mode():

                    action = model.select_action(
                        obs_processed
                    )

                # =================================================
                # 9.6 Postprocess
                # =================================================

                action = postprocess(action)

                # =================================================
                # 9.7 转成机器人 action
                # =================================================

                action = make_robot_action(
                    action,
                    dataset_metadata.features,
                )

                # =================================================
                # 9.8 打印 action
                # =================================================

                if step_idx % 10 == 0:

                    print(
                        f"step={step_idx}"
                    )

                    for key, value in action.items():

                        print(
                            f"  {key}: {value}"
                        )

                # =================================================
                # 9.9 发送机器人控制
                # =================================================

                if DRY_RUN:

                    print(
                        "DRY_RUN=True -> "
                        "action NOT sent to robot"
                    )

                else:

                    robot.send_action(action)

                if display_enabled:
                    try:
                        if not _draw_camera_and_pose_window(
                            obs,
                            episode_idx,
                            step_idx,
                        ):
                            print("Camera window requested shutdown.")
                            stop_requested = True
                            break
                    except cv2.error as exc:
                        logger.warning(
                            "Camera display failed; disabling the window: %s",
                            exc,
                        )
                        display_enabled = False

                # =================================================
                # 9.10 控制频率
                # =================================================

                elapsed = time.perf_counter() - loop_start

                sleep_time = period - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # else:
                #     print(
                #         f"Control loop overrun: "
                #         f"{elapsed * 1000:.1f} ms"
                #     )

            print(
                "Episode finished! "
                "Starting new episode..."
            )

            if stop_requested:
                break

    except KeyboardInterrupt:

        print(
            "Keyboard interrupt received."
        )

    finally:

        if display_enabled:
            cv2.destroyWindow(CAMERA_WINDOW_NAME)

        print(
            "Disconnecting robot..."
        )

        robot.disconnect()

        print(
            "Robot disconnected."
        )


# ================================================================
# entry
# ================================================================

if __name__ == "__main__":
    main()

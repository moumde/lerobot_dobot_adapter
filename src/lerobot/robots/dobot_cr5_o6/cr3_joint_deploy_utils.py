"""Shared CR3/O6 deployment helpers for LeRobot ACT and PI05 policies."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.dobot_cr5_o6.config_dobot_cr5_o6 import DobotCR5O6RobotConfig
from lerobot.robots.dobot_cr5_o6.dobot_cr5_o6 import (
    CR3_JOINT_STATE_NAMES,
    O6_STATE_NAMES,
    DobotCR5O6,
)

DATASET_ROOT = Path("/home/je/code/lerobot/dataset/cr5_o6_20250917")

LEFT_CAMERA_SERIAL = "317222074617"
RIGHT_CAMERA_SERIAL = "254622075848"
SIDE_CAMERA_SERIAL = "262422074985"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

TASK = "Pick up the motor and place it on the right side with the protruding side facing left."
ROBOT_TYPE = "dobot_cr3_o6"


def make_robot() -> DobotCR5O6:
    cameras = {
        "left_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=LEFT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
        "base_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=RIGHT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
        # third camera
        "right_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=SIDE_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
    }
    return DobotCR5O6(
        DobotCR5O6RobotConfig(
            ip="192.168.2.14",
            cameras=cameras,
            control_mode="joint_target",
            enable_action_limits=True,
        )
    )


def show_camera_and_pose(observation: dict, window_name: str, scale: float, episode: int, step: int) -> bool:
    views = []
    for key, title in (
        ("left_wrist_0_rgb", "LEFT WRIST"),
        ("right_wrist_0_rgb", "RIGHT WRIST ROI"),
        ("base_0_rgb", "BASE (RIGHT FULL)"),
    ):
        image = np.asarray(observation[key])
        image = np.clip(image, 0, 255).astype(np.uint8)
        views.append((title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))

    pane_height = max(image.shape[0] for _, image in views)
    pane_width = max(image.shape[1] for _, image in views)
    canvas = np.zeros((pane_height * 2, pane_width * 2, 3), dtype=np.uint8)
    for (title, image), (x, y) in zip(
        views,
        ((0, 0), (pane_width, 0), (0, pane_height)),
        strict=True,
    ):
        if image.shape[:2] != (pane_height, pane_width):
            image = cv2.resize(image, (pane_width, pane_height))
        canvas[y : y + pane_height, x : x + pane_width] = image
        cv2.putText(canvas, title, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    info = canvas[pane_height:, pane_width:]
    info[:] = (24, 24, 24)
    joint_values = [observation[name] for name in CR3_JOINT_STATE_NAMES[:6]]
    joints_deg = np.rad2deg(joint_values)
    lines = [
        f"Episode {episode + 1}  Step {step}",
        "CR3 feedback joints (deg)",
        " ".join(f"{v:7.2f}" for v in joints_deg[:3]),
        " ".join(f"{v:7.2f}" for v in joints_deg[3:]),
        f"O6: {[int(observation[name]) for name in O6_STATE_NAMES]}",
        "Q / ESC: stop",
    ]
    for index, line in enumerate(lines):
        cv2.putText(info, line, (10, 30 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)

    cv2.imshow(window_name, cv2.resize(canvas, dsize=None, fx=scale, fy=scale))
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), ord("Q"), 27)

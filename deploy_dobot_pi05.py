#!/usr/bin/env python3

"""Deploy the PI05 policy trained for Dobot CR5 + LinkerHand O6.

This is intentionally separate from deploy_dobot.py, which is the ACT
deployment entry point.  PI05 additionally needs a task string because its
preprocessor builds a language prompt containing the normalized robot state.
"""

import logging
import time

import cv2
import numpy as np
import torch

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.robots.dobot_cr5_o6.config_dobot_cr5_o6 import DobotCR5O6RobotConfig
from lerobot.robots.dobot_cr5_o6.dobot_cr5_o6 import DobotCR5O6


# -----------------------------------------------------------------------------
# Deployment configuration
# -----------------------------------------------------------------------------

MODEL_ROOT = "/home/je/code/lerobot/dataset/cr5_o6_motor_recognition_pi05_expert"
CHECKPOINT = "050000"
MODEL_PATH = f"{MODEL_ROOT}/checkpoints/{CHECKPOINT}/pretrained_model"

# The PI05 training config records this as the dataset root.  The model output
# directory contains checkpoints/processors, but does not contain meta/info.json.
DATASET_ROOT = "/home/je/code/lerobot/dataset/cr5_o6_motor_recognition_merged"
DATASET_ID = "cr5_o6"

# This must match the task used during training.  PI05 uses it to construct
# the language prompt: "Task: ..., State: ...; Action:"
TASK = "Pick up the motor and place it on the right side with the protruding side facing left."
ROBOT_TYPE = "dobot_cr5_o6"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 500
CONTROL_FPS = 20.0
DRY_RUN = False

# The robot adapter keeps the same safety policy as the ACT deployment.
ENABLE_ACTION_LIMITS = True
WORKSPACE_MIN_M = (0.30, -0.20, 0.05)
WORKSPACE_MAX_M = (0.60, 0.02, 0.45)

# -----------------------------------------------------------------------------
# Cameras and runtime monitor
# -----------------------------------------------------------------------------

LEFT_CAMERA_SERIAL = "317222074617"
RIGHT_CAMERA_SERIAL = "254622075848"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

SHOW_CAMERA_WINDOW = True
CAMERA_WINDOW_NAME = "Dobot CR5 + O6 | PI05 cameras and pose"
CAMERA_DISPLAY_SCALE = 2.0


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def _camera_image_to_bgr(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 camera image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)


def _show_camera_and_pose(observation: dict, episode_idx: int, step_idx: int) -> bool:
    """Display the three policy views and current feedback pose."""
    view_specs = (
        ("left_wrist_0_rgb", "LEFT WRIST"),
        ("right_wrist_0_rgb", "RIGHT WRIST"),
        ("base_0_rgb", "BASE ROI (from right)"),
    )
    views = []
    for key, title in view_specs:
        views.append((title, _camera_image_to_bgr(observation[key])))

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

    info = canvas[pane_height:, pane_width:]
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
    for line_index, line in enumerate(lines):
        cv2.putText(
            info,
            line,
            (8, 22 + line_index * max(18, pane_height // 12)),
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


def _load_pi05(device: torch.device) -> PI05Policy:
    """Load the selected local PI05 checkpoint on the requested device."""
    logger.info("Loading PI05 model from: %s", MODEL_PATH)

    # Read the checkpoint config first so a CPU-only machine does not try to
    # allocate the model on the training machine's recorded CUDA device.
    model_config = PI05Config.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model_config.device = str(device)
    if device.type != "cuda":
        model_config.use_amp = False

    model = PI05Policy.from_pretrained(
        MODEL_PATH,
        config=model_config,
        local_files_only=True,
        strict=True,
    )
    model.eval()
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = _load_pi05(device)

    print(f"Loading dataset metadata: {DATASET_ID}")
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID, root=DATASET_ROOT)

    # Load the processor definitions and normalization tensors saved beside
    # this PI05 checkpoint.  This is important for PI05's tokenizer prompt.
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        pretrained_path=MODEL_PATH,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    cameras = {
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
        cameras=cameras,
        enable_action_limits=ENABLE_ACTION_LIMITS,
        workspace_min_m=WORKSPACE_MIN_M,
        workspace_max_m=WORKSPACE_MAX_M,
    )
    robot = DobotCR5O6(robot_cfg)

    print("Connecting Dobot CR5 + O6...")
    try:
        robot.connect()
    except Exception:
        robot.disconnect()
        raise
    print("Robot connected.")

    display_enabled = False
    if SHOW_CAMERA_WINDOW:
        try:
            cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
            display_enabled = True
        except cv2.error as exc:
            logger.warning("Camera window unavailable; continuing without it: %s", exc)

    print(f"PI05 task: {TASK}")
    print(f"PI05 checkpoint: {MODEL_PATH}")
    print("Robot observation features:")
    for key, value in robot.observation_features.items():
        print(f"  {key}: {value}")

    period = 1.0 / CONTROL_FPS
    stop_requested = False

    try:
        for episode_idx in range(MAX_EPISODES):
            model.reset()
            print(f"========== Episode {episode_idx + 1} / {MAX_EPISODES} ==========")

            for step_idx in range(MAX_STEPS_PER_EPISODE):
                loop_start = time.perf_counter()
                observation = robot.get_observation()

                inference_frame = build_inference_frame(
                    observation=observation,
                    ds_features=dataset_metadata.features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )
                processed_observation = preprocess(inference_frame)

                with torch.inference_mode():
                    policy_action = model.select_action(processed_observation)

                policy_action = postprocess(policy_action)
                robot_action = make_robot_action(
                    policy_action,
                    dataset_metadata.features,
                )

                if step_idx % 10 == 0:
                    logger.info("PI05 step=%d action=%s", step_idx, robot_action)

                if DRY_RUN:
                    logger.info("DRY_RUN=True; action was not sent")
                else:
                    robot.send_action(robot_action)

                if display_enabled:
                    try:
                        if not _show_camera_and_pose(observation, episode_idx, step_idx):
                            stop_requested = True
                            break
                    except cv2.error as exc:
                        logger.warning("Camera display failed; disabling it: %s", exc)
                        display_enabled = False

                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print("Episode finished!")
            if stop_requested:
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    finally:
        if display_enabled:
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
        print("Disconnecting robot...")
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()

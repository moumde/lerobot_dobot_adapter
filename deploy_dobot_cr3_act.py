#!/usr/bin/env python3

"""Deploy an ACT checkpoint trained on cr5_o6_20260909_200040_349965059."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action

from cr3_joint_deploy_utils import (
    DATASET_ROOT,
    ROBOT_TYPE,
    TASK,
    make_robot,
    show_camera_and_pose,
)


MODEL_PATH = Path(
    os.environ.get(
        "LEROBOT_MODEL_PATH",
        "/home/je/code/lerobot/dataset/cr5_o6_20250917_act_result/checkpoints/last/pretrained_model",
    )
)
DATASET_ID = DATASET_ROOT.name
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 500
CONTROL_FPS = 20.0
DRY_RUN = False
SHOW_CAMERA_WINDOW = True
CAMERA_WINDOW_NAME = "Dobot CR3 + O6 | ACT joint target"
CAMERA_DISPLAY_SCALE = 3.0

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ACT checkpoint not found: {MODEL_PATH}. Set LEROBOT_MODEL_PATH to the trained checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Loading ACT checkpoint: %s", MODEL_PATH)
    model = ACTPolicy.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model.config.device = str(device)
    model.to(device)
    model.eval()

    metadata = LeRobotDatasetMetadata(DATASET_ID, root=DATASET_ROOT)
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=metadata.stats,
        pretrained_path=MODEL_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    robot = make_robot()
    display_enabled = False

    try:
        logger.info("Connecting CR3 + O6...")
        robot.connect()
        logger.info("Robot connected. Dataset features: %s", metadata.features.keys())

        if SHOW_CAMERA_WINDOW:
            try:
                cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
                display_enabled = True
            except cv2.error as exc:
                logger.warning("Camera window unavailable: %s", exc)

        period = 1.0 / CONTROL_FPS
        stop_requested = False
        for episode in range(MAX_EPISODES):
            model.reset()
            logger.info("========== Episode %d / %d ==========", episode + 1, MAX_EPISODES)

            for step in range(MAX_STEPS_PER_EPISODE):
                loop_start = time.perf_counter()
                raw_observation = robot.get_observation()
                frame = build_inference_frame(
                    observation=raw_observation,
                    ds_features=metadata.features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )
                with torch.inference_mode():
                    action = model.select_action(preprocess(frame))
                action = make_robot_action(postprocess(action), metadata.features)

                if step % 10 == 0:
                    logger.info("ACT step=%d action=%s", step, action)
                if not DRY_RUN:
                    robot.send_action(action)
                else:
                    logger.info("DRY_RUN=True; action was not sent")

                if display_enabled:
                    try:
                        if not show_camera_and_pose(
                            raw_observation,
                            CAMERA_WINDOW_NAME,
                            CAMERA_DISPLAY_SCALE,
                            episode,
                            step,
                        ):
                            stop_requested = True
                            break
                    except cv2.error as exc:
                        logger.warning("Camera display failed: %s", exc)
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
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()

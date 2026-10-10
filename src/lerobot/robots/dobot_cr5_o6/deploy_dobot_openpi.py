#!/usr/bin/env python3

"""Deploy the OpenPI JAX checkpoint on the Dobot CR5 + LinkerHand O6.

This entry point is intentionally separate from ``deploy_dobot.py`` and
``deploy_dobot_pi05.py``.  The checkpoint in
``cr3_motor_vertical_bf16_bs64_20260904`` is an OpenPI Orbax/JAX checkpoint,
not a LeRobot ``model.safetensors`` checkpoint, so it must be loaded through
OpenPI's native policy API.

Run this file with the Python environment used by OpenPI.  The script adds
the OpenPI and LinkerHand source trees to ``sys.path`` so it can be started
directly from the LeRobot checkout.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


# The OpenPI checkout is not installed as a wheel in the local environment.
# Keep this deployment file self-contained when launched from the repository.
LEROBOT_SRC = Path(__file__).resolve().parents[3]
OPENPI_SRC = Path("/home/je/code/openpi/src")
OPENPI_CLIENT_SRC = Path("/home/je/code/openpi/packages/openpi-client/src")
LINKERHAND_SRC = Path("/home/je/code/linkerhand-python-sdk/LinkerHand")
# OpenPI's environment contains an older LeRobot package under
# ``lerobot.common``.  Do not let this checkout shadow it during import.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != LEROBOT_SRC]
for _source_path in (OPENPI_SRC, OPENPI_CLIENT_SRC, LINKERHAND_SRC):
    if str(_source_path) not in sys.path:
        sys.path.insert(0, str(_source_path))

import cv2
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera


# -----------------------------------------------------------------------------
# Deployment configuration
# -----------------------------------------------------------------------------

CHECKPOINT_ROOT = Path(
    "/home/je/code/lerobot/dataset/cr3_motor_vertical_bf16_bs64_20260904"
)
ASSET_ID = "dobot_cr5_o6_motor_clean_50"

# None selects the numerically latest checkpoint containing ``params``.
CHECKPOINT_STEP: str | None = None

# This must be the same instruction used while training the OpenPI checkpoint.
# It matches the task in cr5_o6_motor_recognition_merged/meta/tasks.parquet.
TASK_PROMPT = (
    "Pick up the motor and place it on the right side with the protruding side facing left."
)

CR5_JOINT_STATE_NAMES = tuple(f"cr5.j{index}.rad" for index in range(1, 7))
CR5_TCP_STATE_NAMES = (
    "cr5.tcp.x.m",
    "cr5.tcp.y.m",
    "cr5.tcp.z.m",
    "cr5.tcp.roll.rad",
    "cr5.tcp.pitch.rad",
    "cr5.tcp.yaw.rad",
)
O6_STATE_NAMES = (
    "o6.thumb_flex.position",
    "o6.thumb_yaw.position",
    "o6.index_flex.position",
    "o6.middle_flex.position",
    "o6.ring_flex.position",
    "o6.little_flex.position",
)
ACTION_NAMES = (
    "cr5.delta_tcp.x.m",
    "cr5.delta_tcp.y.m",
    "cr5.delta_tcp.z.m",
    "cr5.delta_tcp.roll.rad",
    "cr5.delta_tcp.pitch.rad",
    "cr5.delta_tcp.yaw.rad",
    "o6.thumb_flex.command",
    "o6.thumb_yaw.command",
    "o6.index_flex.command",
    "o6.middle_flex.command",
    "o6.ring_flex.command",
    "o6.little_flex.command",
)

# Keep these aligned with the existing Dobot adapter.  Change them here only
# after confirming the physical reset pose for this OpenPI experiment.
RESET_ROBOT_JOINTS_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
RESET_HAND_POSITIONS = (157, 95, 175, 0, 0, 0)
RESET_DURATION_SECONDS = 3.0

LEFT_CAMERA_SERIAL = "317222074617"
RIGHT_CAMERA_SERIAL = "254622075848"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

CONTROL_FPS = 20.0
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 500

# OpenPI predicts a 50-step action chunk.  Start with one action per policy
# inference while validating the model/robot mapping.  This is slower but
# avoids executing stale actions if the hardware feedback changes quickly.
REPLAN_STEPS = 1

DRY_RUN = False
SHOW_CAMERA_WINDOW = True
CAMERA_WINDOW_NAME = "Dobot CR5 + O6 | OpenPI cameras and pose"
CAMERA_DISPLAY_SCALE = 2.0

# Keep the adapter's safety limits enabled during the first hardware test.
ENABLE_ACTION_LIMITS = True
WORKSPACE_MIN_M = (0.30, -0.20, 0.05)
WORKSPACE_MAX_M = (0.60, 0.02, 0.45)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


SHM_NAME = "cr5_shared_state"
STATE_SIZE = 128
CMD_SIZE = 256
RESP_OFFSET = STATE_SIZE + CMD_SIZE
CMD_OFFSET = STATE_SIZE

CMD_POWER_ON = 1
CMD_POWER_OFF = 2
CMD_OPEN_SERVOJ = 4
CMD_CLOSE_SERVOJ = 5
CMD_SEND_TCP_COMMAND = 6
CMD_SEND_JOINT_COMMAND = 7


class _DobotCR5SharedMemory:
    """Small Python 3.10-compatible client for the existing ROS bridge."""

    def __init__(self):
        import struct
        from multiprocessing import resource_tracker, shared_memory

        self._struct = struct
        self._resource_tracker = resource_tracker
        self._shared_memory = shared_memory
        self._shm = None
        self.connected = False
        self.servoj_open = False
        self.servo_powered = False
        self.command_count = {}

    def connect(self):
        try:
            self._shm = self._shared_memory.SharedMemory(name=SHM_NAME, create=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Shared memory not found. Start cr5_ros_node.py before this deployment script."
            ) from exc
        # This process attaches to the ROS-owned segment and must not unlink it.
        self._resource_tracker.unregister(self._shm._name, "shared_memory")
        self.connected = True
        logger.info("Connected to CR5 ROS bridge through shared memory.")

    def _read_state(self):
        if self._shm is None or self._shm.buf[0] == 0:
            raise RuntimeError("CR5 state is not available yet.")
        joints = self._struct.unpack_from("6d", self._shm.buf, 1)
        tcp = self._struct.unpack_from("6d", self._shm.buf, 49)
        return joints, tcp

    def wait_for_state(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._read_state()
                return True
            except RuntimeError:
                time.sleep(0.01)
        return False

    def get_joint_state(self):
        joints, _ = self._read_state()
        return np.deg2rad(np.asarray(joints, dtype=np.float64))

    def get_tcp_state(self):
        _, tcp = self._read_state()
        result = np.asarray(tcp, dtype=np.float64)
        result[:3] /= 1000.0
        return result

    def _send_command(self, command_id, params=()):
        if not self.connected or self._shm is None:
            raise RuntimeError("CR5 shared-memory bridge is not connected.")
        params = list(params)
        if len(params) > 18:
            raise ValueError("The ROS bridge accepts at most 18 command parameters.")

        count = self.command_count.get(command_id, 0) + 1
        self.command_count[command_id] = count
        if command_id not in (CMD_SEND_TCP_COMMAND, CMD_SEND_JOINT_COMMAND) or count == 1 or count % 20 == 0:
            logger.info("Sending CR5 command id=%d count=%d params=%s", command_id, count, params)

        self._shm.buf[RESP_OFFSET + 132] = 0
        padded = np.zeros(18, dtype=np.float64)
        padded[: len(params)] = params
        self._struct.pack_into("i", self._shm.buf, CMD_OFFSET, command_id)
        self._struct.pack_into("18d", self._shm.buf, CMD_OFFSET + 4, *padded)
        self._shm.buf[CMD_OFFSET + 148] = 1

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and self._shm.buf[RESP_OFFSET + 132] != 1:
            time.sleep(0.001)
        if self._shm.buf[RESP_OFFSET + 132] != 1:
            raise TimeoutError("Timeout waiting for ROS bridge response.")

        code = self._struct.unpack_from("i", self._shm.buf, RESP_OFFSET)[0]
        message = bytes(self._shm.buf[RESP_OFFSET + 4 : RESP_OFFSET + 132]).rstrip(b"\0").decode(
            "utf-8", errors="replace"
        )
        self._shm.buf[RESP_OFFSET + 132] = 0
        if code != 0:
            raise RuntimeError(f"ROS bridge rejected command: {message}")

    def power_on(self):
        self._send_command(CMD_POWER_ON)
        self.servo_powered = True

    def power_off(self):
        self._send_command(CMD_POWER_OFF)
        self.servo_powered = False

    def open_servoj(self):
        self._send_command(CMD_OPEN_SERVOJ)
        self.servoj_open = True

    def close_servoj(self):
        self._send_command(CMD_CLOSE_SERVOJ)
        self.servoj_open = False

    def send_tcp_command(self, pose):
        self._send_command(CMD_SEND_TCP_COMMAND, np.asarray(pose, dtype=np.float64).tolist())

    def send_joint_command(self, joints_deg):
        self._send_command(CMD_SEND_JOINT_COMMAND, np.asarray(joints_deg, dtype=np.float64).tolist())

    def disconnect(self):
        if not self.connected:
            return
        try:
            if self.servoj_open:
                self.close_servoj()
            if self.servo_powered:
                self.power_off()
        finally:
            if self._shm is not None:
                self._shm.close()
                self._shm = None
            self.connected = False
            self.servoj_open = False
            self.servo_powered = False


class DobotHardware:
    """Standalone hardware wrapper used because OpenPI currently runs Python 3.11."""

    def __init__(self):
        try:
            from core.rs485.linker_hand_o6_rs485 import LinkerHandO6RS485
        except ImportError as exc:
            raise RuntimeError(
                "The OpenPI environment is missing the O6 SDK or pymodbus. "
                "Add /home/je/code/linkerhand-python-sdk/LinkerHand to PYTHONPATH "
                "and install the SDK dependencies before starting deployment."
            ) from exc

        self.arm = _DobotCR5SharedMemory()
        self.cameras = {
            "left_wrist_0_rgb": IntelRealSenseCamera(
                IntelRealSenseCameraConfig(
                    serial_number=int(LEFT_CAMERA_SERIAL),
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                )
            ),
            "right_wrist_0_rgb": IntelRealSenseCamera(
                IntelRealSenseCameraConfig(
                    serial_number=int(RIGHT_CAMERA_SERIAL),
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                )
            ),
        }
        self.hand = LinkerHandO6RS485(
            hand_id=39,
            modbus_port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
            baudrate=115200,
        )
        self.action_count = 0

    @staticmethod
    def _resize_rgb(image, height=224, width=224):
        image = _as_rgb_uint8(image)
        if image.shape[:2] == (height, width):
            return np.ascontiguousarray(image)
        ratio = max(image.shape[1] / width, image.shape[0] / height)
        new_size = (max(1, int(image.shape[1] / ratio)), max(1, int(image.shape[0] / ratio)))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA if ratio >= 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        top = (height - resized.shape[0]) // 2
        left = (width - resized.shape[1]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        return canvas

    def connect(self):
        self.arm.connect()
        for name, camera in self.cameras.items():
            logger.info("Connecting camera %s", name)
            camera.connect()
        if not self.arm.wait_for_state():
            raise RuntimeError("CR5 state did not arrive from ROS bridge.")
        self.arm.power_on()
        self.arm.open_servoj()
        self.reset_pose()

    def reset_pose(self):
        deadline = time.monotonic() + RESET_DURATION_SECONDS
        while time.monotonic() < deadline:
            self.arm.send_joint_command(RESET_ROBOT_JOINTS_DEG)
            time.sleep(0.05)
        self.hand.set_joint_positions(list(RESET_HAND_POSITIONS))
        time.sleep(1.0)
        logger.info("Reset pose sent: joints=%s hand=%s", RESET_ROBOT_JOINTS_DEG.tolist(), RESET_HAND_POSITIONS)

    def get_observation(self):
        joint = self.arm.get_joint_state()
        tcp = self.arm.get_tcp_state()
        hand = np.asarray(self.hand.get_state(), dtype=np.float32)
        left = self._resize_rgb(self.cameras["left_wrist_0_rgb"].read())
        right = self._resize_rgb(self.cameras["right_wrist_0_rgb"].read())
        x1, y1, x2, y2 = 0.37, 0.56, 0.55, 0.79
        height, width = right.shape[:2]
        base = self._resize_rgb(right[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width)])
        state = np.concatenate((joint, tcp, hand)).astype(np.float32)
        names = CR5_JOINT_STATE_NAMES + CR5_TCP_STATE_NAMES + O6_STATE_NAMES
        observation = {name: float(value) for name, value in zip(names, state, strict=True)}
        observation.update({"base_0_rgb": base, "left_wrist_0_rgb": left, "right_wrist_0_rgb": right})
        return observation

    def send_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (12,):
            raise ValueError(f"Expected 12 action values, got {action.shape}")
        delta = action[:6].copy()
        delta[:3] = np.clip(delta[:3], -0.003, 0.003)
        delta[3:] = np.clip(delta[3:], -0.005, 0.005)
        current_tcp = self.arm.get_tcp_state()
        target_xyz = np.clip(current_tcp[:3] + delta[:3], WORKSPACE_MIN_M, WORKSPACE_MAX_M)
        target = current_tcp.copy()
        target[:3] = target_xyz * 1000.0
        target[3:] += delta[3:]
        self.arm.send_tcp_command(target)
        self.hand.set_joint_positions(np.clip(action[6:], 0, 255).astype(int).tolist())
        self.action_count += 1
        if self.action_count == 1 or self.action_count % 20 == 0:
            logger.info("Action #%d dispatched: raw=%s target_tcp_ros=%s", self.action_count, action.tolist(), target.tolist())

    def disconnect(self):
        try:
            if self.arm.connected and self.arm.servoj_open:
                self.reset_pose()
        except Exception as exc:
            logger.warning("Reset on disconnect failed: %s", exc)
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception as exc:
                logger.warning("Camera disconnect failed: %s", exc)
        try:
            self.hand.close()
        except Exception as exc:
            logger.warning("O6 disconnect failed: %s", exc)
        try:
            self.arm.disconnect()
        except Exception as exc:
            logger.warning("CR5 disconnect failed: %s", exc)


def _latest_checkpoint(root: Path) -> Path:
    """Return the latest numeric OpenPI checkpoint under ``root``."""
    if CHECKPOINT_STEP is not None:
        checkpoint = root / CHECKPOINT_STEP
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"OpenPI params directory not found: {checkpoint / 'params'}")
        return checkpoint

    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "params").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No numeric OpenPI checkpoints found under {root}")
    return max(candidates, key=lambda path: int(path.name))


def _load_checkpoint_norm_stats(checkpoint: Path) -> dict[str, Any]:
    """Load norm stats from either standard OpenPI or ``assets/local`` layout."""
    candidates = (
        checkpoint / "assets" / ASSET_ID,
        checkpoint / "assets" / "local" / ASSET_ID,
    )
    for directory in candidates:
        if (directory / "norm_stats.json").is_file():
            logger.info("Loading norm stats from: %s", directory / "norm_stats.json")
            return _normalize.load(directory)
    searched = ", ".join(str(directory / "norm_stats.json") for directory in candidates)
    raise FileNotFoundError(f"OpenPI norm stats not found. Searched: {searched}")


def _as_rgb_uint8(image: Any) -> np.ndarray:
    """Convert a camera image to the HWC uint8 format expected by OpenPI."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D camera image, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB camera image, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        # LeRobot/OpenPI datasets may store images either in [0, 1] or [0, 255].
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class DobotInputs(_transforms.DataTransformFn):
    """Map the live Dobot observation to OpenPI's three-camera schema."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]
        base = _as_rgb_uint8(image["base_0_rgb"])
        left = _as_rgb_uint8(image["left_wrist_0_rgb"])
        right = _as_rgb_uint8(image["right_wrist_0_rgb"])
        return {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data.get("prompt", TASK_PROMPT),
        }


@dataclasses.dataclass(frozen=True)
class DobotOutputs(_transforms.DataTransformFn):
    """Remove OpenPI's padding dimensions from the predicted action chunk."""

    action_dim: int = len(ACTION_NAMES)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class DobotDataConfig(_config.DataConfigFactory):
    """Inference-only data config matching the custom OpenPI training schema."""

    repo_id: str = "cr5_o6_motor_recognition_merged"
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
                inputs=[DobotInputs()],
                outputs=[DobotOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory(
                default_prompt=TASK_PROMPT,
            )(model_config),
            use_quantile_norm=True,
            action_sequence_keys=("actions",),
        )


def _make_train_config() -> _config.TrainConfig:
    # OpenPI PI05 uses a 32-dimensional internal action/state space.  The
    # checkpoint's norm_stats contains 12 real dimensions; PadStatesAndActions
    # and DobotOutputs handle the 18-state/12-action custom robot layout.
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        dtype="bfloat16",
    )
    return _config.TrainConfig(
        name="cr3_motor_vertical_openpi",
        exp_name="deployment",
        model=model_config,
        data=DobotDataConfig(),
        assets_base_dir=str(CHECKPOINT_ROOT),
        checkpoint_base_dir=str(CHECKPOINT_ROOT.parent),
        wandb_enabled=False,
    )


def _make_policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    state_names = CR5_JOINT_STATE_NAMES + CR5_TCP_STATE_NAMES + O6_STATE_NAMES
    state = np.asarray([observation[name] for name in state_names], dtype=np.float32)
    if state.shape != (18,):
        raise ValueError(f"Expected 18 state values, got {state.shape}")
    return {
        "state": state,
        "image": {
            "base_0_rgb": observation["base_0_rgb"],
            "left_wrist_0_rgb": observation["left_wrist_0_rgb"],
            "right_wrist_0_rgb": observation["right_wrist_0_rgb"],
        },
        "prompt": TASK_PROMPT,
    }


def _check_hardware_dependencies() -> None:
    try:
        import pyrealsense2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The OpenPI environment is missing pyrealsense2. Install the RealSense Python package "
            "in that environment before deployment."
        ) from exc
    try:
        import pymodbus  # noqa: F401
        import inspect
        from pymodbus.client import ModbusSerialClient

        if "slave" not in inspect.signature(ModbusSerialClient.write_registers).parameters:
            raise RuntimeError(
                "The installed pymodbus API uses device_id instead of slave. "
                "The LinkerHand O6 SDK requires pymodbus==3.5.1."
            )
    except ImportError as exc:
        raise RuntimeError(
            "The OpenPI environment is missing pymodbus, required by the O6 RS485 SDK."
        ) from exc


def _camera_image_to_bgr(image: Any) -> np.ndarray:
    return cv2.cvtColor(_as_rgb_uint8(image), cv2.COLOR_RGB2BGR)


def _show_camera_and_pose(observation: dict[str, Any], episode: int, step: int) -> bool:
    views = [
        ("LEFT WRIST", _camera_image_to_bgr(observation["left_wrist_0_rgb"])),
        ("RIGHT WRIST", _camera_image_to_bgr(observation["right_wrist_0_rgb"])),
        ("BASE ROI", _camera_image_to_bgr(observation["base_0_rgb"])),
    ]
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
        cv2.putText(
            canvas,
            title,
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
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
        f"Episode {episode + 1}  Step {step}",
        "Feedback TCP (mm / deg)",
        f"XYZ: {tcp_xyz_mm[0]:7.1f} {tcp_xyz_mm[1]:7.1f} {tcp_xyz_mm[2]:7.1f}",
        f"RPY: {tcp_rpy_deg[0]:7.1f} {tcp_rpy_deg[1]:7.1f} {tcp_rpy_deg[2]:7.1f}",
        "CR5 joints (deg)",
        " ".join(f"{value:6.1f}" for value in joint_deg[:3]),
        " ".join(f"{value:6.1f}" for value in joint_deg[3:]),
        "Q / ESC: stop",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            info,
            line,
            (10, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    display = cv2.resize(
        canvas,
        dsize=None,
        fx=CAMERA_DISPLAY_SCALE,
        fy=CAMERA_DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imshow(CAMERA_WINDOW_NAME, display)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), ord("Q"), 27)


def main() -> None:
    _check_hardware_dependencies()
    checkpoint = _latest_checkpoint(CHECKPOINT_ROOT)
    logger.info("Loading OpenPI checkpoint: %s", checkpoint)
    logger.info("Task prompt: %s", TASK_PROMPT)

    train_config = _make_train_config()
    norm_stats = _load_checkpoint_norm_stats(checkpoint)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=TASK_PROMPT,
        norm_stats=norm_stats,
    )
    logger.info("OpenPI policy loaded. JAX devices: %s", __import__("jax").devices())

    if not ENABLE_ACTION_LIMITS:
        raise ValueError("The standalone OpenPI deployment currently requires action limits enabled.")
    robot = DobotHardware()

    display_enabled = False
    try:
        logger.info("Connecting Dobot CR5 + O6...")
        robot.connect()
        logger.info("Robot connected.")

        if SHOW_CAMERA_WINDOW:
            try:
                cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
                display_enabled = True
            except cv2.error as exc:
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
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != len(ACTION_NAMES):
                        raise ValueError(
                            f"Expected action chunk [N, {len(ACTION_NAMES)}], got {action_chunk.shape}"
                        )
                    action_plan.extend(action_chunk[:REPLAN_STEPS])
                    logger.info(
                        "OpenPI inference %.1f ms, action[0]=%s",
                        (time.perf_counter() - infer_start) * 1000.0,
                        np.array2string(action_chunk[0], precision=5, suppress_small=False),
                    )

                action_vector = np.asarray(action_plan.popleft(), dtype=np.float64)
                if DRY_RUN:
                    logger.info("DRY_RUN=True; action was not sent: %s", action_vector.tolist())
                else:
                    robot.send_action(action_vector)

                if display_enabled:
                    try:
                        if not _show_camera_and_pose(observation, episode, step):
                            stop_requested = True
                            break
                    except cv2.error as exc:
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
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()

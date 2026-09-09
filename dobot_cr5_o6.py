#!/usr/bin/env python3

import logging
import time
from functools import cached_property
from typing import Any

import struct
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from multiprocessing.synchronize import Semaphore
from multiprocessing import Lock

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation

from ..robot import Robot

from .config_dobot_cr5_o6 import DobotCR5O6RobotConfig

logger = logging.getLogger(__name__)
# Keep adapter INFO logs enabled even when the embedding application leaves
# the package logger at its default level.  The deployment entry point owns
# the terminal handler and formatting.
logger.setLevel(logging.INFO)


# ================================================================
# Dataset feature names
# ================================================================

CR5_JOINT_STATE_NAMES = tuple(
    f"cr5.j{index}.rad"
    for index in range(1, 7)
)

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

# Pose used when starting and leaving a deployment run.  The CR5 joint
# interface expects degrees and includes one external-axis value; the O6
# interface expects six register values in the range [0, 255].
RESET_ROBOT_JOINTS_DEG = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
RESET_HAND_POSITIONS = (157, 95, 175, 0, 0, 0)
RESET_COMMAND_FPS = 20.0
RESET_DURATION_SECONDS = 3.0


# ================================================================
# ROS2 adapter
# ================================================================

# Shared memory configuration (must match node)
SHM_NAME = "cr5_shared_state"
STATE_SIZE = 128
CMD_SIZE = 256
RESP_SIZE = 256
SHM_SIZE = 1024

STATE_OFFSET = 0
CMD_OFFSET = STATE_SIZE
RESP_OFFSET = STATE_SIZE + CMD_SIZE

# Command IDs
CMD_POWER_ON = 1
CMD_POWER_OFF = 2
CMD_CLEAR_ERROR = 3
CMD_OPEN_SERVOJ = 4
CMD_CLOSE_SERVOJ = 5
CMD_SEND_TCP_COMMAND = 6
CMD_SEND_JOINT_COMMAND = 7

COMMAND_NAMES = {
    CMD_POWER_ON: "POWER_ON",
    CMD_POWER_OFF: "POWER_OFF",
    CMD_CLEAR_ERROR: "CLEAR_ERROR",
    CMD_OPEN_SERVOJ: "OPEN_SERVOJ",
    CMD_CLOSE_SERVOJ: "CLOSE_SERVOJ",
    CMD_SEND_TCP_COMMAND: "SEND_TCP_COMMAND",
    CMD_SEND_JOINT_COMMAND: "SEND_JOINT_COMMAND",
}


class _DobotCR5Adapter:
    """
    Adapter between LeRobot and the ROS2 Humble CR5 driver via shared memory.

    This class runs in Python 3.12 and communicates with a separate ROS node
    (Python 3.10) using shared memory and semaphores.
    """

    def __init__(
        self,
        ip: str,
        ports: tuple[int, ...],
        node_name: str = "lerobot_dobot_cr5_o6",
    ):
        self.ip = ip
        self.ports = ports
        self.node_name = node_name

        self._shm = None
        self._cmd_sem = None
        self._resp_sem = None
        self._state_lock = None

        self._connected = False
        self._servo_powered = False
        self._servoj_open = False
        self._command_counts: dict[int, int] = {}

    # ------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------
    def connect(self):
        try:
            self._shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        except FileNotFoundError:
            raise RuntimeError(
                "Shared memory not found. Please start the CR5 ROS node first "
                "(run 'python cr5_ros_node.py' in the ROS2 environment)."
            )

        # This process only attaches to the shared memory created by
        # cr5_ros_node.py; it must never unlink the ROS node's segment when
        # Python's resource_tracker exits after an exception.
        resource_tracker.unregister(self._shm._name, "shared_memory")
        self._connected = True
        logger.info("Connected to CR5 ROS node via shared memory.")

    def disconnect(self):
        if not self._connected:
            return
        # Only send cleanup commands for operations that this adapter actually
        # completed.  This prevents a late response to a failed power-on from
        # being mistaken for the response to close_servoj/power_off.
        if self._servoj_open:
            try:
                self.close_servoj()
            except Exception as e:
                logger.warning("Failed to close ServoJ: %s", e)
        if self._servo_powered:
            try:
                self.power_off()
            except Exception as e:
                logger.warning("Failed to power off CR5: %s", e)
        # 关闭共享内存句柄
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        self._connected = False
        self._servo_powered = False
        self._servoj_open = False
        logger.info("Dobot CR5 ROS2 adapter disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_servoj_open(self) -> bool:
        return self._servoj_open

    # ------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------
    def _read_state(self):
        """读取状态区，返回 (joints_deg, tcp_pose)"""
        if self._shm.buf[STATE_OFFSET] == 0:
            raise RuntimeError("No state data available yet.")
        joints = struct.unpack_from("6d", self._shm.buf, STATE_OFFSET + 1)
        tcp = struct.unpack_from("6d", self._shm.buf, STATE_OFFSET + 49)
        return joints, tcp

    def get_joint_state(self) -> np.ndarray:
        joints_deg, _ = self._read_state()
        return np.deg2rad(np.asarray(joints_deg, dtype=np.float64))

    def get_tcp_state(self) -> np.ndarray:
        _, tcp = self._read_state()
        tcp_state = np.asarray(tcp, dtype=np.float64)

        # The NRC ROS interface publishes Cartesian position in mm, while
        # the LeRobot dataset contract uses metres for x/y/z and radians for
        # roll/pitch/yaw.
        tcp_state[:3] /= 1000.0
        return tcp_state

    def wait_for_state(self, timeout: float = 2.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                self._read_state()
                return True
            except RuntimeError:
                time.sleep(0.01)
        return False

    # ------------------------------------------------------------
    # Command sending
    # ------------------------------------------------------------
    def _send_command(self, cmd_id: int, params: list[float] | None = None) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to CR5 ROS node.")
        if params is None:
            params = []
        if len(params) > 18:
            raise ValueError("Too many parameters for command (max 18).")

        command_name = COMMAND_NAMES.get(cmd_id, f"UNKNOWN({cmd_id})")
        command_count = self._command_counts.get(cmd_id, 0) + 1
        self._command_counts[cmd_id] = command_count
        should_log = (
            cmd_id not in (CMD_SEND_TCP_COMMAND, CMD_SEND_JOINT_COMMAND)
            or command_count == 1
            or command_count % 20 == 0
        )
        if should_log:
            logger.info(
                "Sending ROS command %s #%d: params=%s",
                command_name,
                command_count,
                [round(float(value), 6) for value in params],
            )

        # Discard a response left by a previous command before issuing a new
        # request.  The ROS side writes the response asynchronously.
        self._shm.buf[RESP_OFFSET + 132] = 0

        # 填充参数数组
        param_array = np.zeros(18, dtype=np.float64)
        param_array[:len(params)] = params

        # 写入命令
        struct.pack_into("i", self._shm.buf, CMD_OFFSET, cmd_id)
        struct.pack_into("18d", self._shm.buf, CMD_OFFSET + 4, *param_array)
        self._shm.buf[CMD_OFFSET + 148] = 1  # 命令标志

        # 等待响应标志（轮询，带超时）
        start = time.monotonic()
        timeout = 5.0
        while time.monotonic() - start < timeout:
            if self._shm.buf[RESP_OFFSET + 132] == 1:
                break
            time.sleep(0.001)  # 1ms 轮询
        else:
            logger.error(
                "ROS command %s timed out after %.1f s",
                command_name,
                timeout,
            )
            raise TimeoutError("Timeout waiting for ROS node response.")

        # 读取响应
        code = struct.unpack_from("i", self._shm.buf, RESP_OFFSET)[0]
        msg_bytes = bytes(self._shm.buf[RESP_OFFSET + 4 : RESP_OFFSET + 4 + 128]).rstrip(b"\x00")
        message = msg_bytes.decode("utf-8", errors="replace")
        # 清除响应标志
        self._shm.buf[RESP_OFFSET + 132] = 0

        if code != 0:
            logger.error(
                "ROS command %s rejected: %s",
                command_name,
                message,
            )
            raise RuntimeError(f"ROS node command failed: {message}")

        if should_log:
            logger.info(
                "ROS command %s acknowledged by shared-memory bridge",
                command_name,
            )

    # ------------------------------------------------------------
    # Public API (same as before)
    # ------------------------------------------------------------
    def power_on(self):
        logger.info("Powering on CR5 servo...")
        self._send_command(CMD_POWER_ON)
        self._servo_powered = True

    def power_off(self):
        logger.info("Powering off CR5 servo...")
        self._send_command(CMD_POWER_OFF)
        self._servo_powered = False

    def clear_error(self):
        self._send_command(CMD_CLEAR_ERROR)

    def open_servoj(self, vmax, amax, jmax):
        vmax = list(vmax)
        amax = list(amax)
        jmax = list(jmax)
        if vmax or amax or jmax:
            raise ValueError(
                "Custom ServoJ constraints are not supported by the current "
                "shared-memory layout; leave all three lists empty so the "
                "ROS node uses its 7-axis defaults."
            )
        self._send_command(CMD_OPEN_SERVOJ)
        self._servoj_open = True

    def close_servoj(self):
        self._send_command(CMD_CLOSE_SERVOJ)
        self._servoj_open = False

    def send_tcp_command(self, pose_mm: np.ndarray):
        pose = np.asarray(pose_mm, dtype=np.float64).reshape(-1)
        if len(pose) != 6:
            raise ValueError("TCP pose must contain 6 values")
        self._send_command(CMD_SEND_TCP_COMMAND, pose.tolist())

    def send_joint_command(self, joints_deg: np.ndarray):
        joints = np.asarray(joints_deg, dtype=np.float64).reshape(-1)
        if len(joints) != 7:
            raise ValueError("CR5 joint command must contain 7 values")
        self._send_command(CMD_SEND_JOINT_COMMAND, joints.tolist())



# ================================================================
# Main LeRobot robot
# ================================================================

class DobotCR5O6(Robot):
    """
    LeRobot interface for:

        Dobot CR5
        + LinkerHand O6
        + left RealSense
        + right RealSense
        + right-camera ROI as base camera
    """

    config_class = DobotCR5O6RobotConfig
    name = "dobot_cr5_o6"

    def __init__(
        self,
        config: DobotCR5O6RobotConfig,
    ):
        super().__init__(config)

        self.config = config
        self._action_count = 0
        self._unbounded_action_warning_emitted = False

        # --------------------------------------------------------
        # CR5
        # --------------------------------------------------------

        self.arm = _DobotCR5Adapter(
            ip=config.ip,
            ports=config.ports,
            node_name=config.ros2_node_name,
        )

        # --------------------------------------------------------
        # O6
        # --------------------------------------------------------

        from core.rs485.linker_hand_o6_rs485 import (
            LinkerHandO6RS485,
        )

        self.hand = LinkerHandO6RS485(
            hand_id=config.hand_id,
            modbus_port=config.modbus_port,
            baudrate=config.baudrate,
        )

        # --------------------------------------------------------
        # Cameras
        # --------------------------------------------------------

        self.cameras = make_cameras_from_configs(
            config.cameras
        )

    # ============================================================
    # Camera processing
    # ============================================================

    @staticmethod
    def resize_rgb_for_openpi(
        image: np.ndarray,
        height: int = 224,
        width: int = 224,
    ) -> np.ndarray:
        """
        Match the preprocessing used during the original dataset
        collection.

        Preserve aspect ratio and pad with black.

        Example:

            640x480
                ↓
            224x168
                ↓
            centered inside 224x224
        """

        array = np.asarray(image)

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"RGB image must have HxWx3 shape, "
                f"got {array.shape}"
            )

        if array.dtype != np.uint8:
            if np.issubdtype(
                array.dtype,
                np.floating,
            ):
                maximum = (
                    float(np.nanmax(array))
                    if array.size
                    else 0.0
                )

                if maximum <= 1.0:
                    array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        source_height, source_width = array.shape[:2]

        if (
            source_height <= 0
            or source_width <= 0
        ):
            raise ValueError(
                "RGB image is empty"
            )

        if (
            source_height == height
            and source_width == width
        ):
            return np.ascontiguousarray(array)

        ratio = max(
            source_width / width,
            source_height / height,
        )

        resized_height = max(
            1,
            int(source_height / ratio),
        )

        resized_width = max(
            1,
            int(source_width / ratio),
        )

        resized = cv2.resize(
            array,
            (
                resized_width,
                resized_height,
            ),
            interpolation=(
                cv2.INTER_AREA
                if ratio >= 1.0
                else cv2.INTER_LINEAR
            ),
        )

        canvas = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        top = (
            height - resized_height
        ) // 2

        left = (
            width - resized_width
        ) // 2

        canvas[
            top : top + resized_height,
            left : left + resized_width,
        ] = resized

        return np.ascontiguousarray(
            canvas
        )

    def _get_camera_rgb(
        self,
        cam,
    ) -> np.ndarray:
        image = cam.read_latest()

        return self.resize_rgb_for_openpi(
            image,
            self.config.camera_height,
            self.config.camera_width,
        )

    def _get_base_from_right(
        self,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extract base camera from the resized right image.

        roi_norm:
            [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = (
            self.config.roi_norm
        )

        height, width = (
            right_image.shape[:2]
        )

        left = int(x1 * width)
        top = int(y1 * height)

        right = int(x2 * width)
        bottom = int(y2 * height)

        left = max(
            0,
            min(left, width),
        )
        right = max(
            0,
            min(right, width),
        )

        top = max(
            0,
            min(top, height),
        )
        bottom = max(
            0,
            min(bottom, height),
        )

        if right <= left or bottom <= top:
            raise ValueError(
                f"Invalid ROI: "
                f"{self.config.roi_norm}"
            )

        roi = right_image[
            top:bottom,
            left:right,
        ]

        return self.resize_rgb_for_openpi(
            roi,
            self.config.camera_height,
            self.config.camera_width,
        )

    # ============================================================
    # LeRobot feature definitions
    # ============================================================

    @cached_property
    def observation_features(
        self,
    ) -> dict[str, Any]:
        # Robot features describe the raw keys returned by get_observation().
        # LeRobot adds the ``observation.`` / ``observation.images.`` prefixes
        # when it builds dataset features.  Keeping those prefixes out here is
        # important because build_dataset_frame() uses the raw keys to assemble
        # the vector expected by the policy.
        return {
            **{
                name: float
                for name in CR5_JOINT_STATE_NAMES
                + CR5_TCP_STATE_NAMES
                + O6_STATE_NAMES
            },
            "base_0_rgb": (
                self.config.camera_height,
                self.config.camera_width,
                3,
            ),
            "left_wrist_0_rgb": (
                self.config.camera_height,
                self.config.camera_width,
                3,
            ),
            "right_wrist_0_rgb": (
                self.config.camera_height,
                self.config.camera_width,
                3,
            ),
        }

    @cached_property
    def action_features(
        self,
    ) -> dict[str, Any]:
        # As with observation_features, this describes the raw action keys
        # accepted by send_action().  Dataset construction later aggregates
        # them into the single ``action`` vector.
        return {name: float for name in ACTION_NAMES}

    # ============================================================
    # Connection
    # ============================================================

    @property
    def is_connected(self) -> bool:

        cameras_connected = all(
            cam.is_connected
            for cam in self.cameras.values()
        )

        return (
            self.arm.is_connected
            and cameras_connected
            and getattr(
                self.hand,
                "connected",
                False,
            )
        )

    def connect(self):
        logger.info(
            "Connecting Dobot CR5 + O6 + cameras..."
        )

        # CR5
        self.arm.connect()

        # Cameras
        for name, cam in self.cameras.items():
            logger.info(
                "Connecting camera '%s'...",
                name,
            )
            cam.connect()

        # O6 is connected during construction.
        if not getattr(
            self.hand,
            "connected",
            False,
        ):
            raise RuntimeError(
                "O6 hand is not connected."
            )

        # Wait for CR5 state
        if not self.arm.wait_for_state(
            timeout=5.0
        ):
            raise RuntimeError(
                "CR5 state topics did not arrive."
            )

        # --------------------------------------------------------
        # Safety:
        # Don't automatically power the robot unless explicitly
        # requested by config.
        # --------------------------------------------------------

        if self.config.auto_power_on:
            self.arm.power_on()

        if self.config.auto_open_servoj:
            self.arm.open_servoj(
                self.config.servo_vmax,
                self.config.servo_amax,
                self.config.servo_jmax,
            )

        # Put both devices in a known pose before starting inference.
        self.move_to_reset_position()

        logger.info(
            "DobotCR5O6 connected."
        )

    def move_to_reset_position(
        self,
        duration: float = RESET_DURATION_SECONDS,
    ) -> None:
        """Move the CR5 and O6 to the configured deployment reset pose.

        ServoJ is a streaming interface, so repeat the target command during
        the short settling period instead of publishing only one point.
        """
        if not self.arm.is_connected:
            raise RuntimeError("Cannot reset pose before CR5 is connected.")
        if not self.arm.is_servoj_open:
            raise RuntimeError("Cannot reset pose before ServoJ is open.")

        period = 1.0 / RESET_COMMAND_FPS
        deadline = time.monotonic() + duration
        logger.info(
            "Starting CR5 reset stream for %.1f s at %.1f Hz: joints_deg=%s",
            duration,
            RESET_COMMAND_FPS,
            RESET_ROBOT_JOINTS_DEG.tolist(),
        )
        while time.monotonic() < deadline:
            self.arm.send_joint_command(RESET_ROBOT_JOINTS_DEG)
            time.sleep(period)

        logger.info(
            "Sending O6 reset command: positions=%s",
            list(RESET_HAND_POSITIONS),
        )
        self.hand.set_joint_positions(list(RESET_HAND_POSITIONS))
        logger.info(
            "Reset pose commands completed: CR5 joints=%s, O6 positions=%s",
            RESET_ROBOT_JOINTS_DEG.tolist(),
            list(RESET_HAND_POSITIONS),
        )

        time.sleep(1)

    # ============================================================
    # Observation
    # ============================================================

    def get_observation(
        self,
    ) -> RobotObservation:

        # --------------------------------------------------------
        # CR5
        # --------------------------------------------------------

        joint_state = (
            self.arm.get_joint_state()
        )

        tcp_state = (
            self.arm.get_tcp_state()
        )

        # --------------------------------------------------------
        # O6
        # --------------------------------------------------------

        hand_state = np.asarray(
            self.hand.get_state(),
            dtype=np.float32,
        )

        if hand_state.shape != (6,):
            raise RuntimeError(
                f"O6 state shape must be (6,), "
                f"got {hand_state.shape}"
            )

        # --------------------------------------------------------
        # Combined state = 18
        # --------------------------------------------------------

        state = np.concatenate(
            [
                joint_state,
                tcp_state,
                hand_state,
            ]
        ).astype(
            np.float32
        )

        # --------------------------------------------------------
        # Cameras
        # --------------------------------------------------------

        if "left_wrist_0_rgb" not in self.cameras:
            raise RuntimeError(
                "Camera 'left_wrist_0_rgb' is not configured."
            )

        if "right_wrist_0_rgb" not in self.cameras:
            raise RuntimeError(
                "Camera 'right_wrist_0_rgb' is not configured."
            )

        left_image = self._get_camera_rgb(
            self.cameras["left_wrist_0_rgb"]
        )

        right_image = self._get_camera_rgb(
            self.cameras["right_wrist_0_rgb"]
        )

        base_image = (
            self._get_base_from_right(
                right_image
            )
        )

        state_names = (
            CR5_JOINT_STATE_NAMES
            + CR5_TCP_STATE_NAMES
            + O6_STATE_NAMES
        )

        # Return raw, named hardware values.  The dataset/policy-facing
        # ``observation.state`` vector is constructed centrally by
        # build_dataset_frame(), just like for the built-in LeRobot robots.
        observation: RobotObservation = {
            name: float(value)
            for name, value in zip(state_names, state, strict=True)
        }
        observation.update(
            {
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
                "base_0_rgb": base_image,
            }
        )
        return observation

    # ============================================================
    # Action
    # ============================================================

    def send_action(
        self,
        action: RobotAction,
    ) -> RobotAction:

        if "action" in action:
            action_values = action["action"]
        else:
            missing = [name for name in ACTION_NAMES if name not in action]
            if missing:
                raise ValueError(f"Missing action features: {missing}")
            action_values = [action[name] for name in ACTION_NAMES]

        action_array = np.asarray(action_values, dtype=np.float64).reshape(-1)

        if action_array.shape != (12,):
            raise ValueError(
                f"Expected action shape (12,), "
                f"got {action_array.shape}"
            )

        # --------------------------------------------------------
        # CR5 delta TCP
        #
        # Dataset:
        #
        #   x/y/z       : meters
        #   roll/pitch/yaw : radians
        #
        # ROS ServoL:
        #
        #   x/y/z       : mm
        #   roll/pitch/yaw : rad
        # --------------------------------------------------------

        raw_delta_tcp = action_array[:6].copy()
        delta_tcp = raw_delta_tcp.copy()

        if self.config.enable_action_limits:
            xyz_limit = np.asarray(
                self.config.xyz_delta_limit_m,
                dtype=np.float64,
            )
            rpy_limit = np.asarray(
                self.config.rpy_delta_limit_rad,
                dtype=np.float64,
            )
            if xyz_limit.shape != (3,) or np.any(xyz_limit <= 0):
                raise ValueError(
                    "xyz_delta_limit_m must contain three positive values"
                )
            if rpy_limit.shape != (3,) or np.any(rpy_limit <= 0):
                raise ValueError(
                    "rpy_delta_limit_rad must contain three positive values"
                )

            delta_tcp[:3] = np.clip(
                delta_tcp[:3],
                -xyz_limit,
                xyz_limit,
            )
            delta_tcp[3:6] = np.clip(
                delta_tcp[3:6],
                -rpy_limit,
                rpy_limit,
            )
        elif not self._unbounded_action_warning_emitted:
            logger.warning(
                "Per-step TCP action limits are disabled; workspace limits "
                "are still applied if configured."
            )
            self._unbounded_action_warning_emitted = True

        current_tcp_m = self.arm.get_tcp_state()

        workspace_min = self.config.workspace_min_m
        workspace_max = self.config.workspace_max_m
        if (workspace_min is None) != (workspace_max is None):
            raise ValueError(
                "workspace_min_m and workspace_max_m must be set together"
            )
        if workspace_min is not None and workspace_max is not None:
            lower = np.asarray(workspace_min, dtype=np.float64)
            upper = np.asarray(workspace_max, dtype=np.float64)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or np.any(lower >= upper)
            ):
                raise ValueError(
                    "workspace limits must be three-element lower/upper bounds"
                )
            target_xyz_m = np.clip(
                current_tcp_m[:3] + delta_tcp[:3],
                lower,
                upper,
            )
            delta_tcp[:3] = target_xyz_m - current_tcp_m[:3]

        # Build the target in the units expected by ServoL: mm for position
        # and radians for orientation.
        target_tcp = current_tcp_m.copy()
        target_tcp[:3] = (current_tcp_m[:3] + delta_tcp[:3]) * 1000.0
        target_tcp[3:6] += delta_tcp[3:6]

        self._action_count += 1
        should_log_action = (
            self._action_count == 1
            or self._action_count % 20 == 0
        )
        if should_log_action:
            logger.info(
                "Policy action #%d: current_tcp_m=%s, "
                "raw_delta_tcp=%s, clipped_delta_tcp=%s, "
                "target_tcp_ros=%s, hand=%s",
                self._action_count,
                [round(float(value), 6) for value in current_tcp_m],
                [round(float(value), 6) for value in raw_delta_tcp],
                [round(float(value), 6) for value in delta_tcp],
                [round(float(value), 6) for value in target_tcp],
                [int(np.clip(value, 0, 255)) for value in action_array[6:12]],
            )

        self.arm.send_tcp_command(target_tcp)

        # --------------------------------------------------------
        # O6
        # --------------------------------------------------------

        hand_command = action_array[6:12]

        hand_command = np.clip(
            hand_command,
            0,
            255,
        ).astype(int).tolist()

        self.hand.set_joint_positions(
            hand_command
        )

        if should_log_action:
            logger.info(
                "Policy action #%d completed: TCP and O6 commands dispatched",
                self._action_count,
            )

        return {
            name: float(value)
            for name, value in zip(ACTION_NAMES, action_array, strict=True)
        }

    # ============================================================
    # Disconnect
    # ============================================================

    def disconnect(self):

        logger.info(
            "Disconnecting DobotCR5O6..."
        )

        # Return to the known pose while the arm and hand are still connected.
        # Cleanup must continue even if the reset command is rejected during
        # an abnormal shutdown.
        if self.arm.is_connected and self.arm.is_servoj_open:
            try:
                self.move_to_reset_position()
            except Exception as e:
                logger.warning(
                    "Reset pose on disconnect failed: %s",
                    e,
                )

        try:
            for cam in self.cameras.values():
                cam.disconnect()
        except Exception as e:
            logger.warning(
                "Camera disconnect failed: %s",
                e,
            )

        try:
            self.hand.close()
        except Exception as e:
            logger.warning(
                "O6 disconnect failed: %s",
                e,
            )

        try:
            self.arm.disconnect()
        except Exception as e:
            logger.warning(
                "CR5 disconnect failed: %s",
                e,
            )

        logger.info(
            "DobotCR5O6 disconnected."
        )

    def configure(self):
        return super().configure()

    def calibrate(self):
        return super().calibrate()

    @property
    def is_calibrated(self):
        return super().is_calibrated

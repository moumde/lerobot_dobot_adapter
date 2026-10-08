#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class DobotCR5O6Config:
    """Configuration for Dobot CR5 + LinkerHand O6 + RealSense cameras."""

    # ================================================================
    # Dobot CR5
    # ================================================================

    ip: str

    # The same physical arm can be deployed with two dataset action
    # conventions.  Keep TCP-delta as the default for existing datasets.
    # ``joint_target`` is used by the new CR3/O6 dataset and sends absolute
    # joint targets (radians in the dataset, converted to degrees for ROS).
    control_mode: Literal["tcp_delta", "joint_target"] = "tcp_delta"

    # ROS2 driver uses 6001 and 7000 by default.
    # Keep 6000 here as well because some SDK configurations use it.
    ports: tuple[int, ...] = (6000, 6001, 7000)

    # ================================================================
    # Cameras
    # ================================================================

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # OpenPI-compatible image size
    camera_height: int = 224
    camera_width: int = 224

    # ROI extracted from the resized right camera.  In joint_target mode the
    # full right image is exposed as base_0_rgb and this ROI as
    # right_wrist_0_rgb, matching the CR3/O6 dataset.
    #
    # [x1, y1, x2, y2]
    #
    # normalized coordinates.
    roi_norm: tuple[float, float, float, float] = (
        0.37,
        0.56,
        0.55,
        0.79,
    )

    # Per-command TCP safety limits.  These match the limits used by the
    # original data-collection controller and prevent one policy output from
    # commanding a large Cartesian jump.
    xyz_delta_limit_m: tuple[float, float, float] = (
        0.003,
        0.003,
        0.003,
    )
    rpy_delta_limit_rad: tuple[float, float, float] = (
        0.005,
        0.005,
        0.005,
    )

    # Keep the per-step limits enabled during normal deployment.  This can
    # be disabled for a controlled experiment, but the workspace limits
    # below remain a separate safety layer.
    enable_action_limits: bool = True

    # Optional Cartesian workspace limits in dataset units (m).  Keep these
    # unset until calibrated for the physical installation.
    workspace_min_m: tuple[float, float, float] | None = None
    workspace_max_m: tuple[float, float, float] | None = None

    # Safety envelope for absolute joint targets in radians.  These bounds
    # match the observed action envelope of the new CR3/O6 dataset.  They are
    # only used when control_mode == "joint_target".
    joint_target_min_rad: tuple[float, ...] = (
        -0.4524068,
        -1.3138773,
        -0.7545326,
        -0.2983725,
        -1.1004673,
        -2.1733513,
    )
    joint_target_max_rad: tuple[float, ...] = (
        0.6747624,
        0.2520110,
        1.2331995,
        2.5263002,
        0.9260743,
        2.4551303,
    )

    # ================================================================
    # LinkerHand O6
    # ================================================================

    hand_id: int = 39

    modbus_port: str = (
        "/dev/serial/by-id/"
        "usb-1a86_USB_Serial-if00-port0"
    )

    baudrate: int = 115200

    # ================================================================
    # ROS2
    # ================================================================

    ros2_node_name: str = "lerobot_dobot_cr5_o6"

    # Whether to automatically power on / open servo on connect.
    #
    # For safety I recommend False.
    auto_power_on: bool = True
    auto_open_servoj: bool = True

    # ServoJ constraints.
    # Currently we primarily use ServoL for delta-TCP actions.
    # Empty values tell the ROS node to use its 7-axis defaults.  The current
    # shared-memory command area can carry only 18 doubles, so it cannot carry
    # three custom 7-element constraint vectors (21 doubles).
    servo_vmax: tuple[float, ...] = ()
    servo_amax: tuple[float, ...] = ()
    servo_jmax: tuple[float, ...] = ()


@RobotConfig.register_subclass("dobot_cr5_o6")
@dataclass
class DobotCR5O6RobotConfig(RobotConfig, DobotCR5O6Config):
    pass

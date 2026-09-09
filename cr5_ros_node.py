#!/usr/bin/env python3
"""
Dobot CR5 ROS2 node with shared memory IPC for LeRobot (Python 3.10).
Uses shared memory flags for synchronization (no named semaphores).
"""

import struct
import time
import sys
import signal
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nrc_msgs.msg import ServoJPos, ServoLPos
from nrc_msgs.srv import OpenServoJ
from std_srvs.srv import Trigger

from multiprocessing import shared_memory

# Shared memory configuration
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


class CR5ROSNode(Node):
    def __init__(self, shm):
        super().__init__("cr5_ros_node")
        self.shm = shm

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            "/nrc_driver/current_position",
            self.joint_callback,
            10,
        )
        self.tcp_sub = self.create_subscription(
            PoseStamped,
            "/nrc_driver/current_position_tcp",
            self.tcp_callback,
            10,
        )

        # Publishers
        self.servoj_pub = self.create_publisher(ServoJPos, "/nrc_driver/servoj_command", 10)
        self.servol_pub = self.create_publisher(ServoLPos, "/nrc_driver/servol_command", 10)

        # Service clients
        self.poweron_client = self.create_client(Trigger, "/nrc_driver/set_servo_poweron")
        self.poweroff_client = self.create_client(Trigger, "/nrc_driver/set_servo_poweroff")
        self.clear_error_client = self.create_client(Trigger, "/nrc_driver/clear_error")
        self.open_servoj_client = self.create_client(OpenServoJ, "/nrc_driver/open_servoj")
        self.close_servoj_client = self.create_client(Trigger, "/nrc_driver/close_servoj")

        self.joint_valid = False
        self.tcp_valid = False
        self.command_in_progress = False
        self.command_counts = {}

        # Timer to poll command flag
        self.cmd_timer = self.create_timer(0.001, self.check_command_flag)

    def joint_callback(self, msg):
        if len(msg.position) < 6:
            return
        # Write joints in degrees (raw values)
        data = struct.pack("6d", *msg.position[:6])
        self.shm.buf[STATE_OFFSET + 1 : STATE_OFFSET + 1 + 48] = data
        self.joint_valid = True
        self.update_valid_flag()

    def tcp_callback(self, msg):
        pose = msg.pose
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)
        data = struct.pack("6d", x, y, z, roll, pitch, yaw)
        self.shm.buf[STATE_OFFSET + 49 : STATE_OFFSET + 49 + 48] = data
        self.tcp_valid = True
        self.update_valid_flag()

    def update_valid_flag(self):
        if self.joint_valid and self.tcp_valid:
            self.shm.buf[STATE_OFFSET] = 1
        else:
            self.shm.buf[STATE_OFFSET] = 0

    @staticmethod
    def quaternion_to_euler(x, y, z, w):
        # 标准四元数转欧拉角，返回弧度
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def check_command_flag(self):
        """轮询命令标志，若有命令则处理"""
        if self.command_in_progress or self.shm.buf[CMD_OFFSET + 148] == 0:
            return
        # 读取命令
        cmd_id = struct.unpack_from("i", self.shm.buf, CMD_OFFSET)[0]
        params = struct.unpack_from("18d", self.shm.buf, CMD_OFFSET + 4)
        # 清除命令标志
        self.shm.buf[CMD_OFFSET + 148] = 0

        count = self.command_counts.get(cmd_id, 0) + 1
        self.command_counts[cmd_id] = count
        if (
            cmd_id not in (CMD_SEND_TCP_COMMAND, CMD_SEND_JOINT_COMMAND)
            or count == 1
            or count % 20 == 0
        ):
            if cmd_id == CMD_SEND_JOINT_COMMAND:
                logged_params = [round(float(value), 6) for value in params[0:7]]
            elif cmd_id == CMD_SEND_TCP_COMMAND:
                logged_params = [round(float(value), 6) for value in params[0:6]]
            else:
                logged_params = [round(float(value), 6) for value in params]
            self.get_logger().info(
                f"Received command {COMMAND_NAMES.get(cmd_id, cmd_id)} "
                f"(id={cmd_id}) count={count}, "
                f"params={logged_params}"
            )

        self.command_in_progress = True
        try:
            if cmd_id == CMD_POWER_ON:
                self.call_trigger_async(self.poweron_client)
                return
            elif cmd_id == CMD_POWER_OFF:
                self.call_trigger_async(self.poweroff_client)
                return
            elif cmd_id == CMD_CLEAR_ERROR:
                self.call_trigger_async(self.clear_error_client)
                return
            elif cmd_id == CMD_OPEN_SERVOJ:
                # An all-zero parameter block means that the caller supplied
                # no constraints.  Let the ROS driver use its configured
                # 7-axis defaults rather than sending invalid 6-element
                # vectors through the legacy shared-memory layout.
                if all(value == 0.0 for value in params):
                    self.call_open_servoj_async([], [], [])
                else:
                    raise RuntimeError(
                        "Custom ServoJ constraints are not supported; "
                        "leave the constraint lists empty."
                    )
                return
            elif cmd_id == CMD_CLOSE_SERVOJ:
                self.call_trigger_async(self.close_servoj_client)
                return
            elif cmd_id == CMD_SEND_TCP_COMMAND:
                pose_mm = list(params[0:6])
                self.publish_tcp_command(pose_mm)
            elif cmd_id == CMD_SEND_JOINT_COMMAND:
                joints_deg = list(params[0:7])
                self.publish_joint_command(joints_deg)
            else:
                raise RuntimeError(f"Unknown command ID: {cmd_id}")
            self.write_response(True, "")
        except Exception as e:
            self.write_response(False, str(e))

    def write_response(self, success, message):
        """Write one command response and release the command gate."""
        code = 0 if success else 1
        msg_bytes = message.encode("utf-8")[:128].ljust(128, b"\x00")
        struct.pack_into("i", self.shm.buf, RESP_OFFSET, code)
        self.shm.buf[RESP_OFFSET + 4 : RESP_OFFSET + 4 + 128] = msg_bytes
        self.shm.buf[RESP_OFFSET + 132] = 1
        self.command_in_progress = False

    def handle_service_future(self, future):
        """Finish an asynchronous ROS service command."""
        try:
            response = future.result()
            if response is None:
                raise RuntimeError("ROS service returned no response")
            if hasattr(response, "success") and not response.success:
                message = getattr(response, "message", "ROS service failed")
                # The NRC driver reports an already-open ServoJ session as a
                # failure, although this is a valid state when deploy_dobot.py
                # is restarted without restarting the ROS driver.
                if message != "ServoJ tracking already active":
                    raise RuntimeError(message)
            self.write_response(True, getattr(response, "message", ""))
        except Exception as e:
            self.write_response(False, str(e))

    def call_trigger_async(self, client):
        if not client.service_is_ready():
            raise RuntimeError(f"Service unavailable: {client.srv_name}")
        request = client.srv_type.Request()
        future = client.call_async(request)
        future.add_done_callback(self.handle_service_future)

    def call_open_servoj_async(self, vmax, amax, jmax):
        if not self.open_servoj_client.service_is_ready():
            raise RuntimeError("Service unavailable: /nrc_driver/open_servoj")
        request = OpenServoJ.Request()
        request.vmax = list(vmax)
        request.amax = list(amax)
        request.jmax = list(jmax)
        future = self.open_servoj_client.call_async(request)
        future.add_done_callback(self.handle_service_future)

    def publish_tcp_command(self, pose_mm):
        msg = ServoLPos()
        msg.pose = [
            float(pose_mm[0]), float(pose_mm[1]), float(pose_mm[2]),
            float(pose_mm[3]), float(pose_mm[4]), float(pose_mm[5]), 0.0
        ]
        self.servol_pub.publish(msg)
        pose_log = [round(float(value), 6) for value in pose_mm]
        self.get_logger().debug(
            f"Published ServoL command pose_mm={pose_log}"
        )

    def publish_joint_command(self, joints_deg):
        msg = ServoJPos()
        if len(joints_deg) != 7:
            raise ValueError("CR5 joint command must contain 7 values")
        msg.q = [float(value) for value in joints_deg]
        self.servoj_pub.publish(msg)
        joints_log = [round(float(value), 6) for value in joints_deg]
        self.get_logger().info(
            f"Published ServoJ command q_deg={joints_log}"
        )


def main():
    rclpy.init()

    # 创建共享内存
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    except FileExistsError:
        print(f"Shared memory '{SHM_NAME}' already exists. Is the node already running?")
        sys.exit(1)

    # 初始化共享内存（全零）
    shm.buf[:] = b"\x00" * SHM_SIZE

    node = CR5ROSNode(shm)

    def shutdown_handler(signum, frame):
        print("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()
        # 清理共享内存
        shm.close()
        shm.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("CR5 ROS node started. Waiting for commands...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()

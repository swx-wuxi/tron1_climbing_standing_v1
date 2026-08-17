"""Read and display joint angles from a physical LimX robot.

This utility is intentionally read-only: it subscribes to RobotState and never
publishes RobotCmd.  Stop every motion controller before moving the robot by
hand, and only move joints when the robot is in a safe, backdrivable state.
"""

# 验证关节角度是否正确，只读。
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
from scipy.spatial.transform import Rotation as R


class JointStateMonitor:
    """Keep the newest robot state received by the SDK callback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._imu = None
        self._received_at = 0.0

    def callback(self, robot_state):
        # Copy the SDK-owned arrays because the callback object may be reused.
        state = {
            "stamp": robot_state.stamp,
            "q": list(robot_state.q),
            "dq": list(robot_state.dq),
            "tau": list(robot_state.tau),
            "motor_names": list(getattr(robot_state, "motor_names", [])),
        }
        with self._lock:
            self._state = state
            self._received_at = time.monotonic()

    def imu_callback(self, imu_data):
        # SDK quaternion order is [w, x, y, z]; SciPy expects [x, y, z, w].
        quat_wxyz = list(imu_data.quat)
        imu = {
            "stamp": imu_data.stamp,
            "gyro": list(imu_data.gyro),
            "quat": quat_wxyz,
            "euler_deg": None,
        }
        if len(quat_wxyz) == 4:
            quat_xyzw = [
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ]
            try:
                imu["euler_deg"] = R.from_quat(quat_xyzw).as_euler(
                    "xyz", degrees=True
                ).tolist()
            except ValueError:
                pass
        with self._lock:
            self._imu = imu

    def snapshot(self):
        with self._lock:
            return self._state, self._imu, self._received_at


def load_configured_joint_names(robot_type):
    """Load the policy joint order as a fallback for older robot firmware."""
    if not robot_type:
        return []

    config_path = (
        Path(__file__).resolve().parent
        / "controllers"
        / "model"
        / robot_type
        / "params.yaml"
    )
    if not config_path.is_file():
        print(
            f"Warning: no model configuration found for {robot_type}: {config_path}",
            file=sys.stderr,
        )
        return []

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        return list(config["PointfootCfg"]["joint_names"])
    except (ImportError, KeyError, TypeError, OSError) as exc:
        print(f"Warning: cannot read joint names from {config_path}: {exc}", file=sys.stderr)
        return []


def choose_joint_names(state, configured_names):
    count = len(state["q"])
    state_names = state["motor_names"]

    if len(state_names) == count and all(state_names):
        return state_names
    if len(configured_names) == count:
        return configured_names
    return [f"joint_{index}" for index in range(count)]


def format_state(state, imu, configured_names, connection_age):
    names = choose_joint_names(state, configured_names)
    packet_status = "OK" if connection_age < 1.0 else "STALE - check robot connection"
    lines = [
        "Joint angles (Ctrl+C to quit)",
        f"Robot timestamp: {state['stamp']}",
        f"State: {packet_status}    Last packet: {connection_age * 1000:.0f} ms ago",
    ]

    if imu is not None and imu["euler_deg"] is not None:
        roll, pitch, yaw = imu["euler_deg"]
        gyro = imu["gyro"]
        lines.extend([
            (
                f"IMU Euler xyz [deg]: roll={roll:8.3f}  "
                f"pitch={pitch:8.3f}  yaw={yaw:8.3f}"
            ),
            f"IMU gyro [rad/s]: {gyro}",
        ])
    else:
        lines.append("IMU: waiting for data ...")

    lines.extend([
        "",
        f"{'No.':>3}  {'Joint':<22} {'rad':>11} {'deg':>11} {'rad/s':>11}",
        "-" * 64,
    ])

    for index, angle in enumerate(state["q"]):
        velocity = state["dq"][index] if index < len(state["dq"]) else float("nan")
        lines.append(
            f"{index:>3}  {names[index]:<22} {angle:>11.6f} "
            f"{math.degrees(angle):>11.3f} {velocity:>11.6f}"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connect to a physical LimX robot and display joint angles (read-only)."
    )
    parser.add_argument("robot_ip", help="Physical robot IP address, for example 192.168.1.2")
    parser.add_argument(
        "--robot-type",
        default=os.getenv("ROBOT_TYPE"),
        help="Model name used for fallback joint labels, e.g. PF_TRON1B (default: ROBOT_TYPE)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Terminal refresh frequency in Hz (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first state packet (default: 5)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Allow a localhost simulator connection",
    )
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main():
    args = parse_args()
    if args.robot_ip in {"127.0.0.1", "localhost", "::1"} and not args.simulation:
        print(
            "Error: localhost requires --simulation; otherwise provide a physical robot IP.",
            file=sys.stderr,
        )
        return 2

    configured_names = load_configured_joint_names(args.robot_type)
    robot = Robot(RobotType.PointFoot)

    print(f"Connecting to robot at {args.robot_ip} ...")
    if not robot.init(args.robot_ip):
        print("Error: robot connection failed. Check the IP address and network.", file=sys.stderr)
        return 1

    monitor = JointStateMonitor()
    # Keep the bound callback alive for as long as the subscription is active.
    state_callback = monitor.callback
    imu_callback = monitor.imu_callback
    robot.subscribeRobotState(state_callback)
    robot.subscribeImuData(imu_callback)
    print("Connected. Waiting for RobotState packets ...")

    deadline = time.monotonic() + args.timeout
    while monitor.snapshot()[0] is None:
        if time.monotonic() >= deadline:
            print(
                f"Error: no RobotState packet received within {args.timeout:g} seconds.",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.02)

    refresh_period = 1.0 / args.rate
    try:
        while True:
            state, imu, received_at = monitor.snapshot()
            connection_age = time.monotonic() - received_at
            # ANSI clear-screen works in current Linux and Windows terminals.
            print(
                "\033[2J\033[H"
                + format_state(state, imu, configured_names, connection_age),
                end="",
                flush=True,
            )
            time.sleep(refresh_period)
    except KeyboardInterrupt:
        print("\nStopped. No motor commands were sent.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

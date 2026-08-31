#!/usr/bin/env python3
"""Passive MuJoCo/real TRON1 zero-gap monitor.

The process subscribes to RobotState for measured wheel velocity and receives
read-only policy/controller snapshots over a local Unix datagram socket.  It
never constructs or publishes RobotCmd.
"""

import argparse
import atexit
import csv
import math
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent / "tron1-rl-deploy-python"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


DEFAULT_ROBOT_TYPE = "WF_TRON1A"
DEFAULT_RL_TYPE = "isaacgym"
PACKET = struct.Struct("<4sQ6d")
FIELDS = (
    "timestamp_utc",
    "mode",
    "pitch_rate",
    "projected_gravity_x",
    "raw_policy_wheel_action_left",
    "raw_policy_wheel_action_right",
    "sent_wheel_dq_left_rad_s",
    "sent_wheel_dq_right_rad_s",
    "measured_wheel_dq_left_rad_s",
    "measured_wheel_dq_right_rad_s",
)
WHEEL_INDICES = (3, 7)


class RobotStateReceiver:
    """Thread-safe holder for the latest SDK RobotState wheel feedback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._wheel_dq = None
        self._received_at = 0.0

    def callback(self, robot_state):
        dq = list(robot_state.dq)
        if len(dq) <= WHEEL_INDICES[1]:
            return
        wheel_dq = (float(dq[WHEEL_INDICES[0]]), float(dq[WHEEL_INDICES[1]]))
        with self._lock:
            self._wheel_dq = wheel_dq
            self._received_at = time.monotonic()

    def snapshot(self):
        with self._lock:
            return self._wheel_dq, self._received_at


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same core policy/control signals in MuJoCo and on "
            f"hardware (default: {DEFAULT_ROBOT_TYPE}, {DEFAULT_RL_TYPE})."
        )
    )
    parser.add_argument("--mode", choices=("real", "sim"), required=True)
    parser.add_argument(
        "--robot-ip",
        help=(
            "SDK address; sim defaults to 127.0.0.1, real defaults to "
            "$ROBOT_IP or 192.168.1.2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "zero_gap_logs"),
        help="CSV directory (default: ./zero_gap_logs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds to wait for controller and RobotState data (default: 10)",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than zero")
    if args.robot_ip is None:
        args.robot_ip = (
            "127.0.0.1"
            if args.mode == "sim"
            else os.environ.get("ROBOT_IP", "192.168.1.2")
        )
    return args


def drain_telemetry(sock, latest):
    """Drain queued policy snapshots and return the newest valid one."""
    while True:
        try:
            payload = sock.recv(PACKET.size)
        except BlockingIOError:
            return latest
        if len(payload) != PACKET.size:
            continue
        magic, timestamp_ns, *values = PACKET.unpack(payload)
        if magic != b"ZGM1" or not all(math.isfinite(value) for value in values):
            continue
        latest = (timestamp_ns, values, time.monotonic())


def make_row(mode, telemetry, measured_wheel_dq):
    timestamp_ns, values, _ = telemetry
    timestamp = datetime.fromtimestamp(
        timestamp_ns / 1.0e9,
        tz=timezone.utc,
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    numeric = [*values, *measured_wheel_dq]
    formatted = [f"{value:.9f}" for value in numeric]
    return dict(zip(FIELDS, [timestamp, mode, *formatted]))


def print_row(row):
    print(
        f"{row['timestamp_utc']} mode={row['mode']} "
        f"pitch_rate={row['pitch_rate']} (policy input) "
        f"projected_gravity_x={row['projected_gravity_x']} "
        f"raw_policy_wheel_action=[{row['raw_policy_wheel_action_left']}, "
        f"{row['raw_policy_wheel_action_right']}] "
        f"sent_wheel_dq=[{row['sent_wheel_dq_left_rad_s']}, "
        f"{row['sent_wheel_dq_right_rad_s']}] rad/s "
        f"measured_wheel_dq=[{row['measured_wheel_dq_left_rad_s']}, "
        f"{row['measured_wheel_dq_right_rad_s']}] rad/s",
        flush=True,
    )


def main():
    args = parse_args()

    telemetry_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    telemetry_socket.setblocking(False)
    telemetry_path = f"/tmp/tron1_zero_gap_{args.mode}.sock"
    if os.path.exists(telemetry_path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            probe.sendto(b"PROBE", telemetry_path)
        except OSError:
            os.unlink(telemetry_path)
        else:
            print(
                f"Error: another {args.mode} zero-gap monitor is already running.",
                file=sys.stderr,
            )
            return 1
        finally:
            probe.close()
    telemetry_socket.bind(telemetry_path)

    def cleanup_socket():
        telemetry_socket.close()
        try:
            os.unlink(telemetry_path)
        except FileNotFoundError:
            pass

    atexit.register(cleanup_socket)

    robot = Robot(RobotType.PointFoot)
    if not robot.init(args.robot_ip):
        print(f"Error: cannot connect to SDK at {args.robot_ip}", file=sys.stderr)
        return 1

    receiver = RobotStateReceiver()
    state_callback = receiver.callback
    robot.subscribeRobotState(state_callback)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = output_dir / f"zero_gap_{args.mode}_{session}.csv"

    latest_telemetry = None
    deadline = time.monotonic() + args.timeout
    print(
        f"Waiting for {args.mode} controller telemetry and RobotState at "
        f"{args.robot_ip} ...",
        flush=True,
    )
    while True:
        latest_telemetry = drain_telemetry(telemetry_socket, latest_telemetry)
        measured_wheel_dq, _ = receiver.snapshot()
        if latest_telemetry is not None and measured_wheel_dq is not None:
            break
        if time.monotonic() >= deadline:
            print(
                "Error: timed out waiting for WALK policy telemetry and RobotState; "
                "make sure the matching controller is running.",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.01)

    print(f"CSV: {csv_path}", flush=True)
    next_sample = time.monotonic()
    try:
        with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                latest_telemetry = drain_telemetry(
                    telemetry_socket,
                    latest_telemetry,
                )
                now = time.monotonic()
                if now >= next_sample:
                    measured_wheel_dq, state_received_at = receiver.snapshot()
                    telemetry_age = now - latest_telemetry[2]
                    state_age = now - state_received_at
                    if telemetry_age > 2.0 or state_age > 2.0:
                        print(
                            "Warning: controller telemetry or RobotState is stale; "
                            "waiting for fresh data.",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        row = make_row(args.mode, latest_telemetry, measured_wheel_dq)
                        writer.writerow(row)
                        csv_file.flush()
                        print_row(row)
                    next_sample += 1.0
                    if next_sample <= now:
                        next_sample = now + 1.0
                time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\nStopped. CSV saved to: {csv_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

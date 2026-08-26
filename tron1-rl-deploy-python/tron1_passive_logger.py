#!/usr/bin/env python3
"""Passive TRON1 real-robot data logger.

This program only subscribes to RobotState, ImuData, and SensorJoy.  It never
constructs or publishes RobotCmd, so it can run beside the robot's existing
remote-controller process without becoming a second motor controller.

Samples are taken from the latest received packets at 1 Hz by default and are
written to CSV in one-second batches.  The raw SDK quaternion convention is
assumed to be [w, x, y, z], matching the current TRON1 deployment code.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import signal
import threading
import time
from copy import deepcopy

import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


DEFAULT_SAMPLE_RATE_HZ = 1.0
DEFAULT_FLUSH_INTERVAL_S = 1.0
JOY_AXIS_FIELDS = 8
JOY_BUTTON_FIELDS = 16


def finite_float(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def padded(values, size):
    result = [finite_float(value) for value in list(values)[:size]]
    result.extend([float("nan")] * (size - len(result)))
    return result


def safe_column_name(value):
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return cleaned or "unnamed"


def imu_orientation(quat_wxyz):
    """Return projected gravity, roll, and pitch using deployment conventions."""
    quat = padded(quat_wxyz, 4)
    if not all(math.isfinite(value) for value in quat):
        return quat, float("nan"), [float("nan")] * 3, float("nan"), float("nan")

    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-8:
        return quat, norm, [float("nan")] * 3, float("nan"), float("nan")

    w, x, y, z = (value / norm for value in (w, x, y, z))
    normalized = [w, x, y, z]

    # R(q)^T @ [0, 0, -1], identical to
    # scipy Rotation.from_quat([x,y,z,w]).inv().apply([0,0,-1]).
    gravity = [
        2.0 * (w * y - x * z),
        -2.0 * (y * z + w * x),
        2.0 * (x * x + y * y) - 1.0,
    ]
    roll = math.atan2(-gravity[1], -gravity[2])
    pitch = math.atan2(gravity[0], -gravity[2])
    return normalized, norm, gravity, math.degrees(roll), math.degrees(pitch)


class PassiveReceiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.imu = None
        self.joy = None
        self.state_seq = 0
        self.imu_seq = 0
        self.joy_seq = 0
        self.previous_state_received = None
        self.previous_imu_received = None
        self.previous_joy_received = None

    @staticmethod
    def packet_gap_ms(previous, current):
        if previous is None:
            return float("nan")
        return (current - previous) * 1000.0

    def state_callback(self, state):
        received = time.monotonic()
        value = {
            "stamp": integer(state.stamp),
            "q": [finite_float(item) for item in list(state.q)],
            "dq": [finite_float(item) for item in list(state.dq)],
            "tau": [finite_float(item) for item in list(state.tau)],
            "motor_names": [str(item) for item in list(state.motor_names)],
            "received": received,
        }
        with self.lock:
            self.state_seq += 1
            value["seq"] = self.state_seq
            value["gap_ms"] = self.packet_gap_ms(
                self.previous_state_received, received
            )
            self.previous_state_received = received
            self.state = value

    def imu_callback(self, imu):
        received = time.monotonic()
        value = {
            "stamp": integer(imu.stamp),
            "acc": padded(imu.acc, 3),
            "gyro": padded(imu.gyro, 3),
            "quat": padded(imu.quat, 4),
            "received": received,
        }
        with self.lock:
            self.imu_seq += 1
            value["seq"] = self.imu_seq
            value["gap_ms"] = self.packet_gap_ms(
                self.previous_imu_received, received
            )
            self.previous_imu_received = received
            self.imu = value

    def joy_callback(self, joy):
        received = time.monotonic()
        value = {
            "stamp": integer(joy.stamp),
            "axes": [finite_float(item) for item in list(joy.axes)],
            "buttons": [integer(item) for item in list(joy.buttons)],
            "received": received,
        }
        with self.lock:
            self.joy_seq += 1
            value["seq"] = self.joy_seq
            value["gap_ms"] = self.packet_gap_ms(
                self.previous_joy_received, received
            )
            self.previous_joy_received = received
            self.joy = value

    def ready(self):
        with self.lock:
            return self.state is not None and self.imu is not None

    def snapshot(self):
        with self.lock:
            return deepcopy(self.state), deepcopy(self.imu), deepcopy(self.joy)


def find_wheel_indices(motor_names, joint_count):
    left = None
    right = None
    for index, name in enumerate(motor_names):
        lowered = name.lower()
        if "wheel" not in lowered:
            continue
        if "left" in lowered or "_l" in lowered or lowered.startswith("l_"):
            left = index
        elif "right" in lowered or "_r" in lowered or lowered.startswith("r_"):
            right = index

    if left is not None and right is not None:
        return left, right, "motor_names"
    if joint_count == 8:
        return 3, 7, "standard_TRON1_indices"
    return None, None, "unresolved"


def motor_field(prefix, index, motor_names):
    if index < len(motor_names):
        return f"{prefix}_{index}_{safe_column_name(motor_names[index])}"
    return f"{prefix}_{index}"


def build_fields(joint_count, motor_names):
    fields = [
        "session", "sample_index", "wall_time_ns", "monotonic_s", "elapsed_s",
        "sample_dt_ms", "state_seq", "imu_seq", "joy_seq",
        "state_new", "imu_new", "joy_new",
        "state_stamp_ns", "imu_stamp_ns", "joy_stamp_ns",
        "state_age_ms", "imu_age_ms", "joy_age_ms",
        "state_packet_gap_ms", "imu_packet_gap_ms", "joy_packet_gap_ms",
        "imu_state_stamp_skew_ms",
        "quat_w", "quat_x", "quat_y", "quat_z", "quat_norm_raw",
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
        "gravity_x", "gravity_y", "gravity_z", "roll_deg", "pitch_deg",
        "wheel_left_index", "wheel_right_index",
        "wheel_q_left", "wheel_q_right", "wheel_q_common", "wheel_q_diff",
        "wheel_dq_left", "wheel_dq_right", "wheel_dq_common", "wheel_dq_diff",
        "wheel_tau_left", "wheel_tau_right", "wheel_tau_common", "wheel_tau_diff",
        "joy_axes_count", "joy_buttons_count",
    ]
    fields.extend(f"joy_axis_{index}" for index in range(JOY_AXIS_FIELDS))
    fields.extend(f"joy_button_{index}" for index in range(JOY_BUTTON_FIELDS))
    fields.extend(motor_field("q", index, motor_names) for index in range(joint_count))
    fields.extend(motor_field("dq", index, motor_names) for index in range(joint_count))
    fields.extend(motor_field("tau", index, motor_names) for index in range(joint_count))
    return fields


def value_at(values, index):
    if index is None or index < 0 or index >= len(values):
        return float("nan")
    return finite_float(values[index])


def mean_pair(left, right):
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return 0.5 * (left + right)


def diff_pair(left, right):
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return 0.5 * (left - right)


def packet_age_ms(packet, now):
    if packet is None:
        return float("nan")
    return (now - packet["received"]) * 1000.0


def make_row(
    session,
    sample_index,
    start_time,
    now,
    sample_dt_ms,
    state,
    imu,
    joy,
    previous_sequences,
    joint_count,
    motor_names,
    wheel_left,
    wheel_right,
):
    quat, quat_norm, gravity, roll_deg, pitch_deg = imu_orientation(imu["quat"])
    q = padded(state["q"], joint_count)
    dq = padded(state["dq"], joint_count)
    tau = padded(state["tau"], joint_count)
    axes = [] if joy is None else joy["axes"]
    buttons = [] if joy is None else joy["buttons"]

    q_left, q_right = value_at(q, wheel_left), value_at(q, wheel_right)
    dq_left, dq_right = value_at(dq, wheel_left), value_at(dq, wheel_right)
    tau_left, tau_right = value_at(tau, wheel_left), value_at(tau, wheel_right)

    state_stamp = state["stamp"]
    imu_stamp = imu["stamp"]
    joy_stamp = 0 if joy is None else joy["stamp"]
    stamp_skew_ms = float("nan")
    if state_stamp > 0 and imu_stamp > 0:
        stamp_skew_ms = (imu_stamp - state_stamp) / 1.0e6

    joy_seq = 0 if joy is None else joy["seq"]
    row = {
        "session": session,
        "sample_index": sample_index,
        "wall_time_ns": time.time_ns(),
        "monotonic_s": now,
        "elapsed_s": now - start_time,
        "sample_dt_ms": sample_dt_ms,
        "state_seq": state["seq"],
        "imu_seq": imu["seq"],
        "joy_seq": joy_seq,
        "state_new": int(state["seq"] != previous_sequences[0]),
        "imu_new": int(imu["seq"] != previous_sequences[1]),
        "joy_new": int(joy_seq != previous_sequences[2]),
        "state_stamp_ns": state_stamp,
        "imu_stamp_ns": imu_stamp,
        "joy_stamp_ns": joy_stamp,
        "state_age_ms": packet_age_ms(state, now),
        "imu_age_ms": packet_age_ms(imu, now),
        "joy_age_ms": packet_age_ms(joy, now),
        "state_packet_gap_ms": state["gap_ms"],
        "imu_packet_gap_ms": imu["gap_ms"],
        "joy_packet_gap_ms": float("nan") if joy is None else joy["gap_ms"],
        "imu_state_stamp_skew_ms": stamp_skew_ms,
        "quat_w": quat[0], "quat_x": quat[1], "quat_y": quat[2], "quat_z": quat[3],
        "quat_norm_raw": quat_norm,
        "acc_x": imu["acc"][0], "acc_y": imu["acc"][1], "acc_z": imu["acc"][2],
        "gyro_x": imu["gyro"][0], "gyro_y": imu["gyro"][1], "gyro_z": imu["gyro"][2],
        "gravity_x": gravity[0], "gravity_y": gravity[1], "gravity_z": gravity[2],
        "roll_deg": roll_deg, "pitch_deg": pitch_deg,
        "wheel_left_index": "" if wheel_left is None else wheel_left,
        "wheel_right_index": "" if wheel_right is None else wheel_right,
        "wheel_q_left": q_left, "wheel_q_right": q_right,
        "wheel_q_common": mean_pair(q_left, q_right),
        "wheel_q_diff": diff_pair(q_left, q_right),
        "wheel_dq_left": dq_left, "wheel_dq_right": dq_right,
        "wheel_dq_common": mean_pair(dq_left, dq_right),
        "wheel_dq_diff": diff_pair(dq_left, dq_right),
        "wheel_tau_left": tau_left, "wheel_tau_right": tau_right,
        "wheel_tau_common": mean_pair(tau_left, tau_right),
        "wheel_tau_diff": diff_pair(tau_left, tau_right),
        "joy_axes_count": len(axes),
        "joy_buttons_count": len(buttons),
    }

    for index in range(JOY_AXIS_FIELDS):
        row[f"joy_axis_{index}"] = value_at(axes, index)
    for index in range(JOY_BUTTON_FIELDS):
        row[f"joy_button_{index}"] = integer(buttons[index]) if index < len(buttons) else ""
    for index in range(joint_count):
        row[motor_field("q", index, motor_names)] = q[index]
        row[motor_field("dq", index, motor_names)] = dq[index]
        row[motor_field("tau", index, motor_names)] = tau[index]
    return row, (state["seq"], imu["seq"], joy_seq)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only TRON1 IMU, joint, wheel, timing, and joystick CSV logger."
    )
    parser.add_argument("robot_ip", help="robot IP used by the working LIMX SDK connection")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="capture duration in seconds; use 0 to run until Ctrl+C (default: 60)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="CSV sample rate in Hz (default: 1)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=DEFAULT_FLUSH_INTERVAL_S,
        help="seconds between batched disk writes (default: 1)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="sensor wait timeout")
    parser.add_argument("--output", help="CSV path; default is a new timestamped file")
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration cannot be negative")
    if args.sample_rate <= 0.0 or args.flush_interval <= 0.0 or args.timeout <= 0.0:
        parser.error("sample rate, flush interval, and timeout must be positive")
    return args


def default_output_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "passive_logs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"tron1_passive_{stamp}.csv")


def configure_status_logger(csv_path):
    status_path = os.path.splitext(csv_path)[0] + ".status.log"
    logger = logging.getLogger("tron1.passive_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(status_path, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger, status_path


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.output) if args.output else default_output_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        raise FileExistsError(f"refusing to overwrite existing CSV: {csv_path}")

    logger, status_path = configure_status_logger(csv_path)
    logger.info("PASSIVE LOGGER START | no RobotCmd is constructed or published")
    logger.info("CSV=%s | status_log=%s", csv_path, status_path)
    logger.info(
        "robot_ip=%s | sample_rate=%.3fHz | flush_interval=%.3fs | duration=%.3fs",
        args.robot_ip,
        args.sample_rate,
        args.flush_interval,
        args.duration,
    )

    robot = Robot(RobotType.PointFoot)
    if not robot.init(args.robot_ip):
        logger.error("Robot connection failed")
        return 1

    receiver = PassiveReceiver()
    # Local references deliberately keep callbacks alive for the subscription lifetime.
    state_callback = receiver.state_callback
    imu_callback = receiver.imu_callback
    joy_callback = receiver.joy_callback
    robot.subscribeRobotState(state_callback)
    robot.subscribeImuData(imu_callback)
    robot.subscribeSensorJoy(joy_callback)

    deadline = time.monotonic() + args.timeout
    while not receiver.ready():
        if time.monotonic() >= deadline:
            logger.error("Timed out waiting for RobotState and ImuData")
            return 1
        time.sleep(0.02)

    initial_state, _, _ = receiver.snapshot()
    sdk_motor_names = [str(item) for item in list(robot.getMotorNames())]
    state_motor_names = initial_state.get("motor_names", [])
    if state_motor_names and any(name.strip() for name in state_motor_names):
        motor_names = state_motor_names
    else:
        motor_names = sdk_motor_names
    joint_count = max(
        integer(robot.getMotorNumber()),
        len(initial_state["q"]),
        len(initial_state["dq"]),
        len(initial_state["tau"]),
        len(motor_names),
    )
    motor_names = motor_names + [
        f"motor_{index}" for index in range(len(motor_names), joint_count)
    ]
    wheel_left, wheel_right, wheel_source = find_wheel_indices(
        motor_names, joint_count
    )
    fields = build_fields(joint_count, motor_names)
    session = time.strftime("%Y%m%d_%H%M%S")
    logger.info("motor_names=%s", json.dumps(motor_names, ensure_ascii=False))
    logger.info(
        "wheel_mapping | left=%s | right=%s | source=%s",
        wheel_left,
        wheel_right,
        wheel_source,
    )

    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sample_period = 1.0 / args.sample_rate
    start_time = time.monotonic()
    next_sample = start_time
    last_sample = None
    last_flush = start_time
    sample_index = 0
    previous_sequences = (0, 0, 0)
    buffer = []

    with open(csv_path, "x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        csv_file.flush()
        logger.info("CAPTURE START")

        while not stop_requested.is_set():
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= args.duration:
                break
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.01))
                continue

            state, imu, joy = receiver.snapshot()
            if state is not None and imu is not None:
                sample_dt_ms = (
                    float("nan") if last_sample is None else (now - last_sample) * 1000.0
                )
                row, previous_sequences = make_row(
                    session,
                    sample_index,
                    start_time,
                    now,
                    sample_dt_ms,
                    state,
                    imu,
                    joy,
                    previous_sequences,
                    joint_count,
                    motor_names,
                    wheel_left,
                    wheel_right,
                )
                buffer.append(row)
                sample_index += 1
                last_sample = now

            if now - last_flush >= args.flush_interval:
                if buffer:
                    writer.writerows(buffer)
                    buffer.clear()
                    csv_file.flush()
                last_flush = now

            next_sample += sample_period
            if next_sample < now - sample_period:
                next_sample = now + sample_period

        if buffer:
            writer.writerows(buffer)
            csv_file.flush()

    logger.info("CAPTURE STOP | rows=%d | no motor commands were sent", sample_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

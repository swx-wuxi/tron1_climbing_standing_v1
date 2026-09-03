#!/usr/bin/env python3
"""Passive high-rate MuJoCo/real TRON1 actuator-data CSV monitor.

The controller sends synchronized final-command, RobotState, IMU, and FSM
snapshots over a local Unix datagram socket.  This process writes an immutable
raw CSV, a separate position-error CSV, and one configuration row.  It never
constructs or publishes RobotCmd.
"""

import argparse
import atexit
import csv
import os
import socket
import struct
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path


DEFAULT_ROBOT_TYPE = "WF_TRON1A"
DEFAULT_RL_TYPE = "isaacgym"
JOINT_NAMES = (
    "abad_l",
    "hip_l",
    "knee_l",
    "wheel_l",
    "abad_r",
    "hip_r",
    "knee_r",
    "wheel_r",
)
RAW_GROUPS = (
    ("q_des", "rad"),
    ("dq_des", "rad_s"),
    ("q", "rad"),
    ("dq", "rad_s"),
    ("tau_feedback", "nm"),
)
RAW_NUMERIC_FIELDS = tuple(
    f"{joint}_{quantity}_{unit}"
    for quantity, unit in RAW_GROUPS
    for joint in JOINT_NAMES
) + (
    "pitch_deg",
    "gyro_y_rad_s",
)
RAW_FIELDS = (
    "timestamp_ns",
    *RAW_NUMERIC_FIELDS,
    "fsm",
)
LEG_INDICES = (0, 1, 2, 4, 5, 6)
PROCESSED_FIELDS = (
    "timestamp_ns",
    *(
        f"{JOINT_NAMES[index]}_position_error_rad"
        for index in LEG_INDICES
    ),
)
CONFIG_GROUPS = (
    ("tau_ff", "nm"),
    ("kp", "nm_per_rad"),
    ("kd", "nm_s_per_rad"),
)
CONFIG_NUMERIC_FIELDS = tuple(
    f"{joint}_{quantity}_{unit}"
    for quantity, unit in CONFIG_GROUPS
    for joint in JOINT_NAMES
)
CONFIG_FIELDS = (
    "timestamp_ns",
    "mode",
    "fsm_at_log_start",
    *CONFIG_NUMERIC_FIELDS,
)
PACKET = struct.Struct("<4sQ32s66d")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record matching high-rate actuator command/response signals in "
            f"MuJoCo and on hardware (default: {DEFAULT_ROBOT_TYPE}, "
            f"{DEFAULT_RL_TYPE})."
        )
    )
    parser.add_argument("--mode", choices=("real", "sim"), required=True)
    parser.add_argument(
        "--robot-ip",
        help=(
            "accepted for command compatibility only; the controller owns "
            "the SDK connection and sends RobotState in each telemetry packet"
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
        help="seconds to wait for controller telemetry (default: 10)",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than zero")
    return args


def drain_telemetry(sock):
    """Return every complete controller snapshot currently queued."""
    samples = []
    while True:
        try:
            payload = sock.recv(PACKET.size)
        except BlockingIOError:
            return samples
        if len(payload) != PACKET.size:
            continue
        magic, timestamp_ns, fsm_bytes, *values = PACKET.unpack(payload)
        if magic != b"ACT1":
            continue
        fsm = fsm_bytes.split(b"\0", 1)[0].decode(
            "ascii", errors="replace"
        )
        samples.append(
            (
                timestamp_ns,
                fsm,
                values,
                time.monotonic(),
            )
        )


def format_values(values):
    return [f"{value:.9f}" for value in values]


def make_raw_row(telemetry):
    timestamp_ns, fsm, values, _ = telemetry
    raw_values = values[:len(RAW_NUMERIC_FIELDS)]
    return dict(zip(
        RAW_FIELDS,
        [str(timestamp_ns), *format_values(raw_values), fsm],
    ))


def make_processed_row(telemetry):
    timestamp_ns, _, values, _ = telemetry
    q_des = values[0:8]
    q_actual = values[16:24]
    position_error = [
        q_des[index] - q_actual[index]
        for index in LEG_INDICES
    ]
    return dict(zip(
        PROCESSED_FIELDS,
        [str(timestamp_ns), *format_values(position_error)],
    ))


def make_config_row(mode, telemetry):
    timestamp_ns, fsm, values, _ = telemetry
    config_values = values[len(RAW_NUMERIC_FIELDS):]
    return dict(zip(
        CONFIG_FIELDS,
        [str(timestamp_ns), mode, fsm, *format_values(config_values)],
    ))


def print_row(row):
    print(
        f"timestamp_ns={row['timestamp_ns']} fsm={row['fsm']} "
        f"wheel_dq_des=[{row['wheel_l_dq_des_rad_s']}, "
        f"{row['wheel_r_dq_des_rad_s']}] rad/s "
        f"wheel_dq=[{row['wheel_l_dq_rad_s']}, "
        f"{row['wheel_r_dq_rad_s']}] rad/s "
        f"pitch={row['pitch_deg']} deg gyro_y={row['gyro_y_rad_s']} rad/s",
        flush=True,
    )


def main():
    args = parse_args()

    telemetry_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    telemetry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
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

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_csv_path = output_dir / f"actuator_raw_{args.mode}_{session}.csv"
    processed_csv_path = (
        output_dir / f"actuator_position_error_{args.mode}_{session}.csv"
    )
    config_csv_path = (
        output_dir / f"actuator_config_{args.mode}_{session}.csv"
    )

    pending_telemetry = []
    deadline = time.monotonic() + args.timeout
    print(
        f"Waiting for {args.mode} controller telemetry at {telemetry_path} ...",
        flush=True,
    )
    while True:
        pending_telemetry.extend(drain_telemetry(telemetry_socket))
        if pending_telemetry:
            break
        if time.monotonic() >= deadline:
            print(
                "Error: timed out waiting for controller telemetry; "
                "make sure the matching controller is running.",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.01)

    with config_csv_path.open("x", newline="", encoding="utf-8") as config_file:
        config_writer = csv.DictWriter(config_file, fieldnames=CONFIG_FIELDS)
        config_writer.writeheader()
        config_writer.writerow(make_config_row(args.mode, pending_telemetry[0]))

    print(f"Raw CSV: {raw_csv_path}", flush=True)
    print(f"Position-error CSV: {processed_csv_path}", flush=True)
    print(f"Configuration CSV: {config_csv_path}", flush=True)
    print("Recording every controller snapshot at nominal 500 Hz; terminal output is 1 Hz.")
    next_flush = time.monotonic() + 1.0
    next_print = time.monotonic()
    next_stale_warning = time.monotonic()
    latest_row = None
    last_telemetry_received_at = pending_telemetry[-1][3]
    try:
        with ExitStack() as stack:
            raw_file = stack.enter_context(
                raw_csv_path.open("x", newline="", encoding="utf-8")
            )
            processed_file = stack.enter_context(
                processed_csv_path.open(
                    "x", newline="", encoding="utf-8"
                )
            )
            raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
            processed_writer = csv.DictWriter(
                processed_file, fieldnames=PROCESSED_FIELDS
            )
            raw_writer.writeheader()
            processed_writer.writeheader()
            while True:
                if pending_telemetry:
                    telemetry_batch = pending_telemetry
                    pending_telemetry = []
                else:
                    telemetry_batch = drain_telemetry(telemetry_socket)

                for telemetry in telemetry_batch:
                    latest_row = make_raw_row(telemetry)
                    raw_writer.writerow(latest_row)
                    processed_writer.writerow(make_processed_row(telemetry))
                if telemetry_batch:
                    last_telemetry_received_at = telemetry_batch[-1][3]

                now = time.monotonic()
                telemetry_stale = (
                    last_telemetry_received_at == 0.0
                    or now - last_telemetry_received_at > 2.0
                )
                if telemetry_stale and now >= next_stale_warning:
                    print(
                        "Warning: controller telemetry is stale; "
                        "waiting for fresh data.",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_stale_warning = now + 1.0

                if latest_row is not None and now >= next_print:
                    print_row(latest_row)
                    next_print = now + 1.0

                if now >= next_flush:
                    raw_file.flush()
                    processed_file.flush()
                    next_flush = now + 1.0

                time.sleep(0.001)
    except KeyboardInterrupt:
        print(f"\nStopped. Raw CSV saved to: {raw_csv_path}")
        print(f"Position-error CSV saved to: {processed_csv_path}")
        print(f"Configuration CSV saved to: {config_csv_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

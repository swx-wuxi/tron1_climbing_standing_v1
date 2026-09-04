#!/usr/bin/env python3
"""Passively record the exact 28-D WheelfootController policy observation.

The controller copies its final scaled/clipped ``self.observations`` to a
local Unix datagram socket once per policy update.  This monitor only receives
that copy and writes CSV; it never constructs or publishes RobotCmd.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from datetime import datetime
import os
from pathlib import Path
import socket
import struct
import sys
import time


OBSERVATION_SIZE = 28
PACKET = struct.Struct("<4sQ32s28d")
MAGIC = b"OBS1"


def policy_joint_orders(rl_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the joint orders used inside compute_observation()."""
    if rl_type == "isaaclab":
        leg_order = (
            "abad_l", "abad_r", "hip_l", "hip_r", "knee_l", "knee_r",
        )
        full_order = (
            "abad_l", "abad_r", "hip_l", "hip_r",
            "knee_l", "knee_r", "wheel_l", "wheel_r",
        )
    else:
        leg_order = (
            "abad_l", "hip_l", "knee_l", "abad_r", "hip_r", "knee_r",
        )
        full_order = (
            "abad_l", "hip_l", "knee_l", "wheel_l",
            "abad_r", "hip_r", "knee_r", "wheel_r",
        )
    return leg_order, full_order


def observation_fields(rl_type: str) -> tuple[str, ...]:
    leg_order, full_order = policy_joint_orders(rl_type)
    semantic_names = (
        "base_ang_vel_x_scaled",
        "base_ang_vel_y_scaled",
        "base_ang_vel_z_scaled",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
        *(f"{joint}_position_offset_scaled" for joint in leg_order),
        *(f"{joint}_velocity_scaled" for joint in full_order),
        *(f"last_action_{joint}" for joint in full_order),
    )
    if len(semantic_names) != OBSERVATION_SIZE:
        raise RuntimeError("internal observation layout is not 28-D")
    return tuple(
        f"obs_{index:02d}_{name}"
        for index, name in enumerate(semantic_names)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("real", "sim"),
        required=True,
        help="select the matching real or simulator controller telemetry",
    )
    parser.add_argument(
        "--rl-type",
        choices=("isaacgym", "isaaclab"),
        default=os.environ.get("RL_TYPE", "isaacgym"),
        help="names the policy joint order in the CSV header",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output_observation",
        help="CSV directory (default: ./output_observation)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds to wait for controller observation telemetry",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional recording duration in seconds; default runs until Ctrl-C",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than zero")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    return args


def drain(sock: socket.socket) -> list[tuple[int, str, tuple[float, ...]]]:
    samples = []
    while True:
        try:
            payload = sock.recv(PACKET.size)
        except BlockingIOError:
            return samples
        if len(payload) != PACKET.size:
            continue
        magic, timestamp_ns, fsm_bytes, *values = PACKET.unpack(payload)
        if magic != MAGIC:
            continue
        fsm = fsm_bytes.split(b"\0", 1)[0].decode("ascii", errors="replace")
        samples.append((timestamp_ns, fsm, tuple(values)))


def bind_monitor_socket(mode: str) -> tuple[socket.socket, str]:
    telemetry_path = f"/tmp/tron1_policy_observation_{mode}.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        # Some restricted/containerized environments forbid enlarging the
        # receive buffer; the platform default is still sufficient at 50 Hz.
        pass
    sock.setblocking(False)

    if os.path.exists(telemetry_path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            probe.sendto(b"PROBE", telemetry_path)
        except OSError:
            os.unlink(telemetry_path)
        else:
            probe.close()
            sock.close()
            raise RuntimeError(
                f"another {mode} observation monitor is already running"
            )
        finally:
            probe.close()
    sock.bind(telemetry_path)
    return sock, telemetry_path


def main() -> int:
    args = parse_args()
    try:
        telemetry_socket, telemetry_path = bind_monitor_socket(args.mode)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    def cleanup() -> None:
        telemetry_socket.close()
        try:
            os.unlink(telemetry_path)
        except FileNotFoundError:
            pass

    atexit.register(cleanup)
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = (
        args.output_dir.expanduser().resolve()
        / f"policy_observation_{args.mode}_{session}.csv"
    )
    fields = ("timestamp_ns", "mode", "fsm", "rl_type", *observation_fields(args.rl_type))

    print(
        f"Waiting for exact 28-D {args.mode} policy observations at "
        f"{telemetry_path} ...",
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    pending = []
    while not pending:
        pending = drain(telemetry_socket)
        if time.monotonic() >= deadline:
            print(
                "Error: timed out. Start the matching WheelfootController and "
                "make sure it has entered WALK policy mode.",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.005)

    sample_count = 0
    started_at = time.monotonic()
    next_flush = started_at + 1.0
    next_print = started_at
    print(f"Observation CSV: {csv_path}", flush=True)
    print("Recording exact scaled/clipped self.observations; press Ctrl-C to stop.")
    try:
        with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            while True:
                batch = pending if pending else drain(telemetry_socket)
                pending = []
                for timestamp_ns, fsm, values in batch:
                    row = {
                        "timestamp_ns": str(timestamp_ns),
                        "mode": args.mode,
                        "fsm": fsm,
                        "rl_type": args.rl_type,
                    }
                    row.update(
                        (field, f"{value:.9f}")
                        for field, value in zip(fields[4:], values)
                    )
                    writer.writerow(row)
                    sample_count += 1

                now = time.monotonic()
                if batch and now >= next_print:
                    _, fsm, values = batch[-1]
                    print(
                        f"samples={sample_count} fsm={fsm} "
                        f"gyro_y_scaled={values[1]:.6f} "
                        f"gravity_x={values[3]:.6f} "
                        f"obs_range=[{min(values):.6f}, {max(values):.6f}]",
                        flush=True,
                    )
                    next_print = now + 1.0
                if now >= next_flush:
                    csv_file.flush()
                    next_flush = now + 1.0
                if args.duration is not None and now - started_at >= args.duration:
                    break
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - started_at
    print(
        f"Stopped: {sample_count} observations in {elapsed:.2f} s. CSV: {csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

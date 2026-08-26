#!/usr/bin/env python3
"""Read-only IMU zero/frame diagnostic for TRON1 deployment.

This program never publishes RobotCmd.  It reproduces the quaternion conversion,
projected-gravity calculation, roll, and pitch convention used by the current
WheelfootController, then reports static offsets or checks gyro axis/sign while
the unpowered/safely supported robot is moved by hand.
"""

import argparse
import csv
import math
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=float)


def imu_values(quat_wxyz, gyro_xyz):
    """Return values using exactly the deployment controller's convention."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    gyro_xyz = np.asarray(gyro_xyz, dtype=float)
    if quat_wxyz.shape != (4,) or gyro_xyz.shape != (3,):
        raise ValueError("unexpected IMU array size")
    if not np.all(np.isfinite(quat_wxyz)) or not np.all(np.isfinite(gyro_xyz)):
        raise ValueError("non-finite IMU value")

    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm < 1.0e-8:
        raise ValueError("zero quaternion")
    quat_wxyz = quat_wxyz / quat_norm

    # LIMX SDK: [w, x, y, z]. SciPy/current controller: [x, y, z, w].
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    orientation = R.from_quat(quat_xyzw)
    projected_gravity = orientation.inv().apply(GRAVITY_WORLD)

    # Same formulas as get_full_orientation_state()/get_pitch_state().
    roll = math.atan2(-projected_gravity[1], -projected_gravity[2])
    pitch = math.atan2(projected_gravity[0], -projected_gravity[2])
    return {
        "quat_wxyz": quat_wxyz,
        "quat_xyzw": quat_xyzw,
        "rotation": orientation,
        "gravity": projected_gravity,
        "roll": roll,
        "pitch": pitch,
        "gyro": gyro_xyz,
        "quat_norm_before_normalization": quat_norm,
    }


class Monitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_state = None
        self._latest_imu = None
        self._samples = []

    def state_callback(self, state):
        value = {
            "stamp": int(state.stamp),
            "q": np.asarray(state.q, dtype=float).copy(),
            "dq": np.asarray(state.dq, dtype=float).copy(),
            "received": time.monotonic(),
        }
        with self._lock:
            self._latest_state = value

    def imu_callback(self, imu):
        received = time.monotonic()
        try:
            value = imu_values(list(imu.quat), list(imu.gyro))
        except ValueError:
            return
        value["stamp"] = int(imu.stamp)
        value["received"] = received

        with self._lock:
            state = self._latest_state
            value["state_stamp"] = None if state is None else state["stamp"]
            value["q"] = None if state is None else state["q"].copy()
            value["dq"] = None if state is None else state["dq"].copy()
            self._latest_imu = value
            self._samples.append(value)

    def ready(self):
        with self._lock:
            return self._latest_imu is not None and self._latest_state is not None

    def latest(self):
        with self._lock:
            return self._latest_imu

    def clear_samples(self):
        with self._lock:
            self._samples = []

    def samples(self):
        with self._lock:
            return list(self._samples)


def array_stats(samples, key):
    values = np.asarray([sample[key] for sample in samples], dtype=float)
    return np.mean(values, axis=0), np.std(values, axis=0), np.ptp(values, axis=0)


def scalar_stats_deg(samples, key):
    values = np.degrees(np.asarray([sample[key] for sample in samples], dtype=float))
    return float(np.mean(values)), float(np.std(values)), float(np.ptp(values))


def print_live(sample):
    gravity = sample["gravity"]
    gyro = sample["gyro"]
    quat = sample["quat_wxyz"]
    stamp_skew = "n/a"
    if sample["state_stamp"] is not None and sample["stamp"] > 0:
        stamp_skew = f"{(sample['stamp'] - sample['state_stamp']) / 1.0e6:+.2f} ms"
    print(
        f"roll={math.degrees(sample['roll']):+8.3f} deg  "
        f"pitch={math.degrees(sample['pitch']):+8.3f} deg  "
        f"gyro=[{gyro[0]:+7.4f}, {gyro[1]:+7.4f}, {gyro[2]:+7.4f}] rad/s\n"
        f"gravity=[{gravity[0]:+8.5f}, {gravity[1]:+8.5f}, {gravity[2]:+8.5f}]  "
        f"quat_wxyz=[{quat[0]:+8.5f}, {quat[1]:+8.5f}, "
        f"{quat[2]:+8.5f}, {quat[3]:+8.5f}]  stamp_skew={stamp_skew}"
    )


def static_report(samples, reference_roll_deg, reference_pitch_deg):
    roll_mean, roll_std, roll_p2p = scalar_stats_deg(samples, "roll")
    pitch_mean, pitch_std, pitch_p2p = scalar_stats_deg(samples, "pitch")
    gravity_mean, gravity_std, gravity_p2p = array_stats(samples, "gravity")
    gyro_mean, gyro_std, gyro_p2p = array_stats(samples, "gyro")
    quat_norms = np.asarray(
        [sample["quat_norm_before_normalization"] for sample in samples]
    )

    print("\n========== STATIC IMU REPORT ==========")
    print(f"samples: {len(samples)}")
    print(
        f"controller roll : mean={roll_mean:+.4f} deg  "
        f"std={roll_std:.4f}  peak-to-peak={roll_p2p:.4f}"
    )
    print(
        f"controller pitch: mean={pitch_mean:+.4f} deg  "
        f"std={pitch_std:.4f}  peak-to-peak={pitch_p2p:.4f}"
    )
    print(
        "projected gravity mean/std/p2p:\n"
        f"  mean {np.array2string(gravity_mean, precision=6, sign='+')}\n"
        f"  std  {np.array2string(gravity_std, precision=6)}\n"
        f"  p2p  {np.array2string(gravity_p2p, precision=6)}"
    )
    print(
        "gyro [rad/s] mean/std/p2p:\n"
        f"  mean {np.array2string(gyro_mean, precision=6, sign='+')}\n"
        f"  std  {np.array2string(gyro_std, precision=6)}\n"
        f"  p2p  {np.array2string(gyro_p2p, precision=6)}"
    )
    print(
        f"raw quaternion norm: mean={np.mean(quat_norms):.7f}, "
        f"min={np.min(quat_norms):.7f}, max={np.max(quat_norms):.7f}"
    )

    roll_bias = roll_mean - reference_roll_deg
    pitch_bias = pitch_mean - reference_pitch_deg
    print("\nReference comparison (external measurement is authoritative):")
    print(
        f"  roll bias  = measured - reference = {roll_bias:+.4f} deg"
    )
    print(
        f"  pitch bias = measured - reference = {pitch_bias:+.4f} deg"
    )
    print(
        "  Approximate projected-gravity frame correction: "
        f"roll {roll_bias:+.4f} deg, pitch {pitch_bias:+.4f} deg."
    )
    print(
        "  Do not copy these numbers into params.yaml until the motion sign test "
        "also passes and the base reference angle has been measured independently."
    )
    print("=======================================")


def motion_report(samples):
    if len(samples) < 10:
        print("Not enough motion samples.", file=sys.stderr)
        return

    measured_rates = []
    gyros = []
    for previous, current in zip(samples[:-1], samples[1:]):
        dt = current["received"] - previous["received"]
        if dt <= 1.0e-5 or dt > 0.1:
            continue
        # Relative orientation is expressed in the previous body frame and can
        # therefore be compared directly with a body-frame gyro for slow motion.
        relative = previous["rotation"].inv() * current["rotation"]
        measured_rates.append(relative.as_rotvec() / dt)
        gyros.append(0.5 * (previous["gyro"] + current["gyro"]))

    measured_rates = np.asarray(measured_rates)
    gyros = np.asarray(gyros)
    active = np.linalg.norm(measured_rates, axis=1) > math.radians(2.0)
    measured_rates = measured_rates[active]
    gyros = gyros[active]
    if len(measured_rates) < 10:
        print(
            "Not enough deliberate motion. Repeat with several slow nose-up/"
            "nose-down movements of roughly 5-10 degrees.",
            file=sys.stderr,
        )
        return

    labels = ("x/roll", "y/pitch", "z/yaw")
    print("\n========== GYRO AXIS/SIGN REPORT ==========")
    print(f"active motion intervals: {len(measured_rates)}")
    print("Rows: orientation-derived body rate; columns: raw gyro x/y/z correlation")
    correlation = np.full((3, 3), np.nan)
    for row in range(3):
        for column in range(3):
            if np.std(measured_rates[:, row]) > 1.0e-4 and np.std(gyros[:, column]) > 1.0e-4:
                correlation[row, column] = np.corrcoef(
                    measured_rates[:, row], gyros[:, column]
                )[0, 1]
    print(np.array2string(correlation, precision=3, suppress_small=True))

    for axis, label in enumerate(labels):
        row = correlation[axis]
        if np.all(np.isnan(row)):
            continue
        best = int(np.nanargmax(np.abs(row)))
        sign = "+" if row[best] >= 0.0 else "-"
        print(
            f"derived {label:7s} best matches gyro {'xyz'[best]} with {sign} sign "
            f"(corr={row[best]:+.3f}); expected gyro {'xyz'[axis]} with + sign"
        )

    direct_error = gyros - measured_rates
    print(
        "direct gyro - orientation-rate mean [rad/s]: "
        + np.array2string(np.mean(direct_error, axis=0), precision=5, sign="+")
    )
    print(
        "PASS expectation: strong positive diagonal correlation; during nose-up, "
        "controller pitch and gyro_y should both increase."
    )
    print("===========================================")


def write_csv(path, samples):
    joint_count = max(
        (0 if sample["q"] is None else len(sample["q"]) for sample in samples),
        default=0,
    )
    fields = [
        "received_monotonic_s", "imu_stamp_ns", "state_stamp_ns",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "gyro_x", "gyro_y", "gyro_z",
        "gravity_x", "gravity_y", "gravity_z", "roll_deg", "pitch_deg",
    ]
    fields += [f"q_{index}" for index in range(joint_count)]
    fields += [f"dq_{index}" for index in range(joint_count)]
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {
                "received_monotonic_s": sample["received"],
                "imu_stamp_ns": sample["stamp"],
                "state_stamp_ns": sample["state_stamp"],
                "quat_w": sample["quat_wxyz"][0],
                "quat_x": sample["quat_wxyz"][1],
                "quat_y": sample["quat_wxyz"][2],
                "quat_z": sample["quat_wxyz"][3],
                "gyro_x": sample["gyro"][0],
                "gyro_y": sample["gyro"][1],
                "gyro_z": sample["gyro"][2],
                "gravity_x": sample["gravity"][0],
                "gravity_y": sample["gravity"][1],
                "gravity_z": sample["gravity"][2],
                "roll_deg": math.degrees(sample["roll"]),
                "pitch_deg": math.degrees(sample["pitch"]),
            }
            if sample["q"] is not None:
                row.update({f"q_{i}": value for i, value in enumerate(sample["q"])})
                row.update({f"dq_{i}": value for i, value in enumerate(sample["dq"])})
            writer.writerow(row)
    print(f"Raw diagnostic CSV written to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only TRON1 IMU zero, projected-gravity, and gyro-frame diagnostic."
    )
    parser.add_argument("robot_ip", help="physical robot IP, e.g. 192.168.1.2")
    parser.add_argument(
        "--mode", choices=("static", "motion", "live"), default="static",
        help="static zero statistics, guided motion axis/sign check, or live display",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="capture seconds")
    parser.add_argument("--display-rate", type=float, default=5.0, help="live print rate")
    parser.add_argument("--timeout", type=float, default=5.0, help="packet timeout seconds")
    parser.add_argument("--reference-roll-deg", type=float, default=0.0)
    parser.add_argument("--reference-pitch-deg", type=float, default=0.0)
    parser.add_argument("--csv", help="optional output CSV path; no file is written by default")
    parser.add_argument(
        "--simulation", action="store_true", help="permit localhost MuJoCo connection"
    )
    args = parser.parse_args()
    if args.duration <= 0.0 or args.display_rate <= 0.0 or args.timeout <= 0.0:
        parser.error("duration, display-rate, and timeout must be positive")
    if args.robot_ip in {"127.0.0.1", "localhost", "::1"} and not args.simulation:
        parser.error("localhost requires --simulation")
    return args


def main():
    args = parse_args()
    print("READ-ONLY MODE: this program subscribes to sensors and never sends RobotCmd.")
    print("Stop every motion controller and mechanically secure the robot before testing.")

    robot = Robot(RobotType.PointFoot)
    print(f"Connecting to {args.robot_ip} ...")
    if not robot.init(args.robot_ip):
        print("Robot connection failed.", file=sys.stderr)
        return 1

    monitor = Monitor()
    # Keep bound callbacks alive throughout the subscription lifetime.
    state_callback = monitor.state_callback
    imu_callback = monitor.imu_callback
    robot.subscribeRobotState(state_callback)
    robot.subscribeImuData(imu_callback)

    deadline = time.monotonic() + args.timeout
    while not monitor.ready():
        if time.monotonic() >= deadline:
            print("Timed out waiting for RobotState and IMU packets.", file=sys.stderr)
            return 1
        time.sleep(0.02)

    if args.mode == "static":
        print(
            f"Keep the independently measured base pose completely still for "
            f"{args.duration:g} seconds."
        )
    elif args.mode == "motion":
        print(
            f"For {args.duration:g} seconds, slowly repeat nose-up/nose-down by "
            "5-10 degrees. If safe, also make a small left/right roll movement."
        )
    else:
        print("Live display; press Ctrl+C to stop.")

    monitor.clear_samples()
    start = time.monotonic()
    next_display = start
    display_period = 1.0 / args.display_rate
    try:
        while args.mode == "live" or time.monotonic() - start < args.duration:
            now = time.monotonic()
            if now >= next_display:
                sample = monitor.latest()
                if sample is not None:
                    print_live(sample)
                next_display = now + display_period
            time.sleep(min(0.01, display_period))
    except KeyboardInterrupt:
        pass

    samples = monitor.samples()
    if not samples:
        print("No valid IMU samples captured.", file=sys.stderr)
        return 1
    if args.mode == "static":
        static_report(samples, args.reference_roll_deg, args.reference_pitch_deg)
    elif args.mode == "motion":
        motion_report(samples)
    if args.csv:
        write_csv(args.csv, samples)
    print("Finished. No motor commands were sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train reference-style TRON1 leg and wheel actuator torque networks.

This follows the timing contract used by the ANYmal supplementary deployment:
three timestamp-matched samples are presented oldest-to-current and the label
is measured torque at the current sample.  Legs and wheels are trained as two
separate shared models because their tracking errors have different meanings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from actuator_dataset import ArrayDataset, Normalization, _find_timestamp, _timestamp_seconds
from actuator_model import make_model


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "input_raw_realdata/Raw_v1.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
HISTORY_OFFSETS_S = np.asarray((0.0, 0.0075, 0.015), dtype=np.float64)
TIMESTAMP_TOLERANCE_S = 0.0025
TRAIN_FRACTION = 0.80
MINIMUM_DT_S = 0.00025
MAXIMUM_GAP_S = 0.005
HIDDEN_SIZES = (64, 64)
BATCH_SIZE = 1024
EPOCHS = 200
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
EARLY_STOPPING_PATIENCE = 30
SEED = 42


@dataclass(frozen=True)
class JointSpec:
    csv_prefix: str
    simulator_name: str
    group: str
    tracking_error_type: str


JOINT_SPECS = (
    JointSpec("abad_l", "abad_L_Joint", "legs", "position"),
    JointSpec("hip_l", "hip_L_Joint", "legs", "position"),
    JointSpec("knee_l", "knee_L_Joint", "legs", "position"),
    JointSpec("wheel_l", "wheel_L_Joint", "wheels", "velocity"),
    JointSpec("abad_r", "abad_R_Joint", "legs", "position"),
    JointSpec("hip_r", "hip_R_Joint", "legs", "position"),
    JointSpec("knee_r", "knee_R_Joint", "legs", "position"),
    JointSpec("wheel_r", "wheel_R_Joint", "wheels", "velocity"),
)


@dataclass
class JointData:
    timestamp_s: np.ndarray
    actual_dq: np.ndarray
    tracking_error: np.ndarray
    torque: np.ndarray
    specs: tuple[JointSpec, ...]
    median_dt_s: float
    source_rows: int


@dataclass
class Examples:
    inputs: np.ndarray
    targets: np.ndarray
    joint_ids: np.ndarray
    timestamps_s: np.ndarray
    match_errors_s: np.ndarray
    rejected_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--group", choices=("both", "legs", "wheels"), default="both")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_data(csv_path: Path) -> JointData:
    frame = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    timestamp_column = _find_timestamp(list(frame.columns))
    required = [timestamp_column]
    for spec in JOINT_SPECS:
        required.extend((
            f"{spec.csv_prefix}_dq_rad_s",
            f"{spec.csv_prefix}_tau_feedback_nm",
            f"{spec.csv_prefix}_{'q_des_rad' if spec.group == 'legs' else 'dq_des_rad_s'}",
        ))
        if spec.group == "legs":
            required.append(f"{spec.csv_prefix}_q_rad")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required actuator fields: {missing}")

    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    numeric = numeric.loc[valid].reset_index(drop=True)
    timestamp_s = _timestamp_seconds(
        numeric[timestamp_column].to_numpy(dtype=np.float64), timestamp_column
    )
    monotonic = np.r_[True, np.diff(timestamp_s) > 0.0]
    numeric = numeric.loc[monotonic].reset_index(drop=True)
    timestamp_s = timestamp_s[monotonic]

    actual_dq = []
    errors = []
    torque = []
    for spec in JOINT_SPECS:
        prefix = spec.csv_prefix
        dq = numeric[f"{prefix}_dq_rad_s"].to_numpy(dtype=np.float32)
        if spec.group == "legs":
            desired = numeric[f"{prefix}_q_des_rad"].to_numpy(dtype=np.float32)
            actual = numeric[f"{prefix}_q_rad"].to_numpy(dtype=np.float32)
        else:
            desired = numeric[f"{prefix}_dq_des_rad_s"].to_numpy(dtype=np.float32)
            actual = dq
        actual_dq.append(dq)
        errors.append(desired - actual)
        torque.append(numeric[f"{prefix}_tau_feedback_nm"].to_numpy(dtype=np.float32))

    return JointData(
        timestamp_s=timestamp_s,
        actual_dq=np.stack(actual_dq, axis=1),
        tracking_error=np.stack(errors, axis=1),
        torque=np.stack(torque, axis=1),
        specs=JOINT_SPECS,
        median_dt_s=float(np.median(np.diff(timestamp_s))),
        source_rows=len(frame),
    )


def build_examples(data: JointData, group: str, subset: str) -> Examples:
    row_count = len(data.timestamp_s)
    split = int(row_count * TRAIN_FRACTION)
    start, stop = (0, split) if subset == "train" else (split, row_count)
    segment_time = data.timestamp_s[start:stop]
    transition_ok = (np.diff(data.timestamp_s) >= MINIMUM_DT_S) & (
        np.diff(data.timestamp_s) <= MAXIMUM_GAP_S
    )
    first = max(
        start,
        int(np.searchsorted(data.timestamp_s, data.timestamp_s[start] + HISTORY_OFFSETS_S[-1])),
    )
    selected = [index for index, spec in enumerate(data.specs) if spec.group == group]
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    joint_ids: list[int] = []
    timestamps: list[float] = []
    errors: list[np.ndarray] = []
    rejected = 0

    def nearest(requested: float) -> int:
        insertion = int(np.searchsorted(segment_time, requested))
        candidates = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(segment_time):
            candidates.append(insertion)
        local = min(candidates, key=lambda i: (abs(segment_time[i] - requested), segment_time[i]))
        return start + local

    for joint_id in selected:
        for current in range(first, stop):
            requested = data.timestamp_s[current] - HISTORY_OFFSETS_S
            indices_new_to_old = np.asarray(
                [current if age == 0.0 else nearest(float(t)) for age, t in zip(HISTORY_OFFSETS_S, requested)],
                dtype=np.int64,
            )
            matched = data.timestamp_s[indices_new_to_old]
            match_error = matched - requested
            causal = indices_new_to_old[0] == current and np.all(np.diff(indices_new_to_old) < 0)
            continuous = np.all(transition_ok[indices_new_to_old[-1]:current])
            if not causal or not continuous or np.any(np.abs(match_error) > TIMESTAMP_TOLERANCE_S):
                rejected += 1
                continue
            indices_old_to_new = indices_new_to_old[::-1]
            inputs.append(np.concatenate((
                data.actual_dq[indices_old_to_new, joint_id],
                data.tracking_error[indices_old_to_new, joint_id],
            )).astype(np.float32))
            targets.append(float(data.torque[current, joint_id]))
            joint_ids.append(joint_id)
            timestamps.append(float(data.timestamp_s[current]))
            errors.append(match_error[::-1])

    if not inputs:
        raise ValueError(f"No valid {subset} examples for {group}")
    return Examples(
        inputs=np.stack(inputs),
        targets=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        joint_ids=np.asarray(joint_ids, dtype=np.int64),
        timestamps_s=np.asarray(timestamps),
        match_errors_s=np.stack(errors),
        rejected_windows=rejected,
    )


def fit_normalization(examples: Examples) -> Normalization:
    return Normalization(
        examples.inputs.mean(axis=0, dtype=np.float64).astype(np.float32),
        np.maximum(examples.inputs.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32),
        examples.targets.mean(axis=0, dtype=np.float64).astype(np.float32),
        np.maximum(examples.targets.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32),
    )


@torch.no_grad()
def predict(model, inputs, normalization, device, batch_size):
    model.eval()
    result = []
    for start in range(0, len(inputs), batch_size):
        x = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        result.append(model(x).cpu().numpy())
    return np.concatenate(result) * normalization.target_std + normalization.target_mean


def train_group(data: JointData, group: str, args, device: torch.device) -> None:
    train = build_examples(data, group, "train")
    val = build_examples(data, group, "val")
    norm = fit_normalization(train)
    train_x = ((train.inputs - norm.input_mean) / norm.input_std).astype(np.float32)
    train_y = ((train.targets - norm.target_mean) / norm.target_std).astype(np.float32)
    val_x = ((val.inputs - norm.input_mean) / norm.input_std).astype(np.float32)
    val_y = ((val.targets - norm.target_mean) / norm.target_std).astype(np.float32)
    train_loader = DataLoader(ArrayDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)
    model = make_model(train_x.shape[1], HIDDEN_SIZES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    output = args.output_dir / f"best_reference_{group}_actuator.pt"
    best = float("inf")
    stale = 0

    print(f"\n[{group}] examples train={len(train_x)}, val={len(val_x)}, rejected={train.rejected_windows + val.rejected_windows}")
    print(f"[{group}] history=-15/-7.5/0 ms oldest-to-current; target=current measured torque")
    print(f"[{group}] timestamp match max={np.abs(np.r_[train.match_errors_s.ravel(), val.match_errors_s.ravel()]).max()*1000:.4f} ms")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        total = count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_loss = criterion(model(x), y)
                total += float(batch_loss) * len(x)
                count += len(x)
        val_loss = total / count
        if epoch == 1 or epoch % 10 == 0:
            physical = predict(model, val_x, norm, device, args.batch_size)
            residual = physical - val.targets
            print(f"[{group}] epoch={epoch:03d} val_loss={val_loss:.7f} MAE={np.abs(residual).mean():.5f} Nm RMSE={np.sqrt(np.mean(residual**2)):.5f} Nm")
        if val_loss < best - 1e-8:
            best, stale = val_loss, 0
            group_specs = [spec for spec in data.specs if spec.group == group]
            torch.save({
                "format_version": 3,
                "model_state_dict": model.state_dict(),
                "input_dim": train_x.shape[1],
                "hidden_sizes": list(HIDDEN_SIZES),
                "history_length": len(HISTORY_OFFSETS_S),
                "history_version": "reference_v3",
                "history_offsets_s": HISTORY_OFFSETS_S.tolist(),
                "history_order": "oldest_to_current",
                "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
                "feature_order": ["dq_actual_oldest_to_current", "tracking_error_oldest_to_current"],
                "tracking_error_type": group_specs[0].tracking_error_type,
                "actuator_group": group,
                "joint_names": [spec.simulator_name for spec in group_specs],
                "target_mode": "torque",
                "target_row_offset": 0,
                "target_offset_s": 0.0,
                "run_every_simulation_step": True,
                "input_mean": torch.from_numpy(norm.input_mean),
                "input_std": torch.from_numpy(norm.input_std),
                "target_mean": torch.from_numpy(norm.target_mean),
                "target_std": torch.from_numpy(norm.target_std),
                "train_fraction": TRAIN_FRACTION,
                "minimum_dt_s": MINIMUM_DT_S,
                "maximum_gap_s": MAXIMUM_GAP_S,
                "source_csv": str(args.csv.expanduser().resolve()),
                "median_dt_s": data.median_dt_s,
                "best_val_loss": best,
                "seed": SEED,
            }, output)
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                break

    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    physical = predict(model, val_x, norm, device, args.batch_size)
    residual = physical - val.targets
    print(f"[{group}] best MAE={np.abs(residual).mean():.5f} Nm RMSE={np.sqrt(np.mean(residual**2)):.5f} Nm")
    for joint_id in np.unique(val.joint_ids):
        mask = val.joint_ids == joint_id
        joint_residual = residual[mask]
        name = data.specs[int(joint_id)].simulator_name
        print(f"  {name}: MAE={np.abs(joint_residual).mean():.5f}, RMSE={np.sqrt(np.mean(joint_residual**2)):.5f} Nm")
    print(f"[{group}] checkpoint: {output}")


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.csv)
    print(f"rows source={data.source_rows}, valid={len(data.timestamp_s)}, median_dt={data.median_dt_s*1000:.4f} ms, device={device}")
    groups = ("legs", "wheels") if args.group == "both" else (args.group,)
    for group in groups:
        train_group(data, group, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

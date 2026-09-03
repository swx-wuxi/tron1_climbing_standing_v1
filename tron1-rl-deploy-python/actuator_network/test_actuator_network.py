#!/usr/bin/env python3
"""Evaluate a saved wheel actuator network and show minimal inference usage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tron1_actuator_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from actuator_dataset import build_examples, load_wheel_csv
from actuator_model import make_model


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "outputs" / "best_wheel_actuator.pt"
DEFAULT_PLOT = SCRIPT_DIR / "outputs" / "test_predicted_vs_actual.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def load_checkpoint_model(
    checkpoint_path: str | Path,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    model = make_model(checkpoint["input_dim"], checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    checkpoint: dict,
    desired_dq_history: np.ndarray,
    actual_dq_history: np.ndarray,
) -> float:
    """Minimal inference example for one wheel.

    Histories must be oldest-to-current and have checkpoint history_length
    values.  A torque checkpoint returns Nm; a next_velocity checkpoint returns
    the next measured wheel velocity in rad/s.
    """

    desired = np.asarray(desired_dq_history, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual_dq_history, dtype=np.float32).reshape(-1)
    history_length = int(checkpoint["history_length"])
    if len(desired) != history_length or len(actual) != history_length:
        raise ValueError(f"Both histories must contain {history_length} values")
    features = np.concatenate((actual, desired - actual)).astype(np.float32)
    input_mean = checkpoint["input_mean"].numpy()
    input_std = checkpoint["input_std"].numpy()
    normalized = (features - input_mean) / input_std
    output_normalized = model(torch.from_numpy(normalized).unsqueeze(0)).item()
    target_mean = float(checkpoint["target_mean"].item())
    target_std = float(checkpoint["target_std"].item())
    return output_normalized * target_std + target_mean


def main() -> int:
    args = parse_args()
    model, checkpoint = load_checkpoint_model(args.checkpoint)
    csv_path = args.csv or Path(checkpoint["source_csv"])
    session = load_wheel_csv(csv_path, target_mode=checkpoint["target_mode"])
    examples = build_examples(
        session,
        history_length=int(checkpoint["history_length"]),
        train_fraction=float(checkpoint["train_fraction"]),
        subset="val",
        minimum_dt_s=float(checkpoint["minimum_dt_s"]),
        maximum_gap_s=float(checkpoint["maximum_gap_s"]),
    )

    input_mean = checkpoint["input_mean"].numpy()
    input_std = checkpoint["input_std"].numpy()
    target_mean = float(checkpoint["target_mean"].item())
    target_std = float(checkpoint["target_std"].item())
    normalized = (examples.inputs - input_mean) / input_std
    with torch.no_grad():
        predicted = model(torch.from_numpy(normalized.astype(np.float32))).numpy()
    predicted = predicted * target_std + target_mean
    residual = predicted[:, 0] - examples.targets[:, 0]
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    unit = "Nm" if checkpoint["target_mode"] == "torque" else "rad/s"
    print(f"Validation samples: {len(examples.inputs)}")
    print(f"MAE={mae:.6f} {unit}, RMSE={rmse:.6f} {unit}")
    for side_id, side_name in ((0, "left"), (1, "right")):
        mask = examples.side_ids == side_id
        side_residual = residual[mask]
        print(
            f"{side_name}: MAE={np.mean(np.abs(side_residual)):.6f} {unit}, "
            f"RMSE={np.sqrt(np.mean(side_residual**2)):.6f} {unit}"
        )

    history_length = int(checkpoint["history_length"])
    sample = examples.inputs[0]
    actual_history = sample[:history_length]
    error_history = sample[history_length:]
    desired_history = actual_history + error_history
    one_prediction = predict_one(
        model, checkpoint, desired_history, actual_history
    )
    print(
        f"Single-window inference: predicted={one_prediction:.6f} {unit}, "
        f"actual={examples.targets[0, 0]:.6f} {unit}"
    )

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    points = min(5000, len(predicted))
    plt.figure(figsize=(13, 4))
    plt.plot(examples.targets[:points, 0], label="actual", linewidth=1.0)
    plt.plot(predicted[:points, 0], label="predicted", linewidth=1.0)
    plt.xlabel("validation sample")
    plt.ylabel(unit)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    plt.close()
    print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


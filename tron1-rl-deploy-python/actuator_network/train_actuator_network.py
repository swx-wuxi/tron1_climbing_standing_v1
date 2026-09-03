#!/usr/bin/env python3
"""Train the first history-based TRON1 wheel actuator network."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tron1_actuator_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from actuator_dataset import (
    ArrayDataset,
    WheelExamples,
    build_examples,
    fit_normalization,
    load_wheel_csv,
    normalize_examples,
)
from actuator_model import make_model


# All first-version experiment settings are intentionally centralized here.
DEFAULT_CSV = Path(
    "/home/air/Desktop/sim2real_data/0902data/"
    "actuator_raw_real_20241023_161025_043119.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET_MODE = "auto"  # auto selects torque when both wheel torque fields are usable
HISTORY_LENGTH = 8
TRAIN_FRACTION = 0.80
MINIMUM_DT_MS = 0.25
MAXIMUM_GAP_MS = 5.0
HIDDEN_SIZES = (64, 64)
BATCH_SIZE = 1024
EPOCHS = 200
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
EARLY_STOPPING_PATIENCE = 30
SEED = 42
PLOT_POINTS_PER_SIDE = 2500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--target-mode",
        choices=("auto", "torque", "next_velocity"),
        default=TARGET_MODE,
    )
    parser.add_argument("--history", type=int, default=HISTORY_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def predict_physical(
    model: nn.Module,
    normalized_inputs: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    tensor = torch.from_numpy(normalized_inputs.astype(np.float32))
    for start in range(0, len(tensor), batch_size):
        output = model(tensor[start : start + batch_size].to(device))
        predictions.append(output.cpu().numpy())
    normalized = np.concatenate(predictions, axis=0)
    return normalized * target_std + target_mean


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = predicted.reshape(-1) - actual.reshape(-1)
    return float(np.mean(np.abs(residual))), float(np.sqrt(np.mean(residual**2)))


def print_side_metrics(
    examples: WheelExamples, predictions: np.ndarray, target_label: str
) -> None:
    for side_id, side_name in ((0, "left"), (1, "right")):
        mask = examples.side_ids == side_id
        mae, rmse = metrics(examples.targets[mask], predictions[mask])
        print(f"  {side_name:>5s}: MAE={mae:.6f}, RMSE={rmse:.6f} {target_label}")


def save_prediction_plot(
    path: Path,
    examples: WheelExamples,
    predictions: np.ndarray,
    target_label: str,
    points_per_side: int,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    for axis, (side_id, side_name) in zip(axes, ((0, "left"), (1, "right"))):
        indices = np.flatnonzero(examples.side_ids == side_id)[:points_per_side]
        time = examples.timestamps_s[indices]
        time = time - time[0]
        axis.plot(time, examples.targets[indices, 0], label="actual", linewidth=1.0)
        axis.plot(time, predictions[indices, 0], label="predicted", linewidth=1.0)
        axis.set_title(f"{side_name.capitalize()} wheel validation")
        axis.set_ylabel(target_label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("time within plotted validation segment [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    set_reproducible_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = load_wheel_csv(args.csv, target_mode=args.target_mode)
    train_examples = build_examples(
        session,
        history_length=args.history,
        train_fraction=TRAIN_FRACTION,
        subset="train",
        minimum_dt_s=MINIMUM_DT_MS / 1000.0,
        maximum_gap_s=MAXIMUM_GAP_MS / 1000.0,
    )
    val_examples = build_examples(
        session,
        history_length=args.history,
        train_fraction=TRAIN_FRACTION,
        subset="val",
        minimum_dt_s=MINIMUM_DT_MS / 1000.0,
        maximum_gap_s=MAXIMUM_GAP_MS / 1000.0,
    )
    normalization = fit_normalization(train_examples)
    train_x, train_y = normalize_examples(train_examples, normalization)
    val_x, val_y = normalize_examples(val_examples, normalization)

    print("CSV field mapping:")
    for key, value in asdict(session.columns).items():
        print(f"  {key}: {value}")
    print(
        f"Rows: source={session.source_rows}, valid={session.valid_rows}, "
        f"median_dt={session.median_dt_s * 1000.0:.4f} ms"
    )
    print(
        f"Target={session.target_mode}, history={args.history} frames "
        f"(~{(args.history - 1) * session.median_dt_s * 1000.0:.2f} ms), "
        f"input_dim={train_x.shape[1]}"
    )
    print(
        f"Examples: train={len(train_x)} (rejected={train_examples.rejected_windows}), "
        f"val={len(val_x)} (rejected={val_examples.rejected_windows})"
    )
    print(f"Device: {device}")

    train_loader = DataLoader(
        ArrayDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ArrayDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = make_model(train_x.shape[1], HIDDEN_SIZES).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    checkpoint_path = args.output_dir / "best_wheel_actuator.pt"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(inputs)
            train_count += len(inputs)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                loss = criterion(model(inputs), targets)
                val_loss_sum += float(loss.item()) * len(inputs)
                val_count += len(inputs)
        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count

        val_predictions = predict_physical(
            model,
            val_x,
            normalization.target_mean,
            normalization.target_std,
            device,
            args.batch_size,
        )
        val_mae, val_rmse = metrics(val_examples.targets, val_predictions)
        unit = "Nm" if session.target_mode == "torque" else "rad/s"
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.7f} "
            f"val_loss={val_loss:.7f} val_MAE={val_mae:.6f} {unit} "
            f"val_RMSE={val_rmse:.6f} {unit}"
        )

        if val_loss < best_val_loss - 1.0e-8:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_x.shape[1],
                    "hidden_sizes": list(HIDDEN_SIZES),
                    "history_length": args.history,
                    "feature_order": [
                        "dq_actual_oldest_to_current",
                        "dq_des_minus_dq_actual_oldest_to_current",
                    ],
                    "target_mode": session.target_mode,
                    "input_mean": torch.from_numpy(normalization.input_mean),
                    "input_std": torch.from_numpy(normalization.input_std),
                    "target_mean": torch.from_numpy(normalization.target_mean),
                    "target_std": torch.from_numpy(normalization.target_std),
                    "train_fraction": TRAIN_FRACTION,
                    "minimum_dt_s": MINIMUM_DT_MS / 1000.0,
                    "maximum_gap_s": MAXIMUM_GAP_MS / 1000.0,
                    "csv_columns": asdict(session.columns),
                    "source_csv": str(Path(args.csv).expanduser().resolve()),
                    "median_dt_s": session.median_dt_s,
                    "best_val_loss": best_val_loss,
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_predictions = predict_physical(
        model,
        val_x,
        normalization.target_mean,
        normalization.target_std,
        device,
        args.batch_size,
    )
    unit = "Nm" if session.target_mode == "torque" else "rad/s"
    val_mae, val_rmse = metrics(val_examples.targets, val_predictions)
    print(f"Best validation: MAE={val_mae:.6f}, RMSE={val_rmse:.6f} {unit}")
    print_side_metrics(val_examples, val_predictions, unit)

    plot_path = args.output_dir / "validation_predicted_vs_actual.png"
    save_prediction_plot(
        plot_path,
        val_examples,
        val_predictions,
        unit,
        PLOT_POINTS_PER_SIDE,
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Validation plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


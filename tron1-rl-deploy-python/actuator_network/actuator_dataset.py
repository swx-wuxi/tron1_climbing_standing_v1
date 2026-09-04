"""CSV loading and history-window construction for the TRON1 wheel actuator net."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TargetMode = Literal["auto", "torque", "next_velocity"]
HistoryOrder = Literal["oldest_to_current", "current_to_oldest"]


@dataclass(frozen=True)
class WheelColumns:
    timestamp: str
    left_dq_des: str
    right_dq_des: str
    left_dq: str
    right_dq: str
    left_tau: str | None = None
    right_tau: str | None = None


@dataclass
class WheelSession:
    timestamp_s: np.ndarray
    dq_des: np.ndarray
    dq: np.ndarray
    tau: np.ndarray | None
    columns: WheelColumns
    target_mode: Literal["torque", "next_velocity"]
    source_rows: int
    valid_rows: int
    median_dt_s: float


@dataclass
class WheelExamples:
    inputs: np.ndarray
    targets: np.ndarray
    timestamps_s: np.ndarray
    history_timestamps_s: np.ndarray
    history_match_errors_s: np.ndarray | None
    side_ids: np.ndarray
    rejected_windows: int


@dataclass(frozen=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


class ArrayDataset(Dataset):
    """Small torch Dataset backed by already-normalized float32 arrays."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_exact(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalized_name(column): column for column in columns}
    for candidate in candidates:
        match = normalized.get(_normalized_name(candidate))
        if match is not None:
            return match
    return None


def _find_timestamp(columns: list[str]) -> str:
    exact = _find_exact(
        columns,
        ("timestamp_ns", "stamp_ns", "time_ns", "timestamp", "time"),
    )
    if exact is not None:
        return exact
    for column in columns:
        name = _normalized_name(column)
        if "timestamp" in name or name in {"stamp", "time"}:
            return column
    raise ValueError("Could not identify a timestamp column in the CSV header")


def _has_side(name: str, side: Literal["left", "right"]) -> bool:
    tokens = set(name.split("_"))
    if side == "left":
        return "left" in tokens or "wheel_l" in name or "l_wheel" in name
    return "right" in tokens or "wheel_r" in name or "r_wheel" in name


def _find_wheel_signal(
    columns: list[str], side: Literal["left", "right"], signal: str
) -> str | None:
    short = "l" if side == "left" else "r"
    exact_candidates = {
        "dq_des": (
            f"wheel_{short}_dq_des_rad_s",
            f"wheel_{side}_dq_des_rad_s",
            f"{side}_wheel_dq_des_rad_s",
            f"wheel_{short}_desired_velocity",
            f"wheel_{side}_desired_velocity",
        ),
        "dq": (
            f"wheel_{short}_dq_rad_s",
            f"wheel_{side}_dq_rad_s",
            f"{side}_wheel_dq_rad_s",
            f"wheel_{short}_actual_velocity",
            f"wheel_{side}_actual_velocity",
        ),
        "tau": (
            f"wheel_{short}_tau_feedback_nm",
            f"wheel_{side}_tau_feedback_nm",
            f"{side}_wheel_tau_feedback_nm",
            f"wheel_{short}_torque_nm",
            f"wheel_{side}_torque_nm",
        ),
    }
    exact = _find_exact(columns, exact_candidates[signal])
    if exact is not None:
        return exact

    matches: list[str] = []
    for column in columns:
        name = _normalized_name(column)
        if "wheel" not in name or not _has_side(name, side):
            continue
        is_velocity = "dq" in name or "velocity" in name or "vel" in name
        is_desired = "des" in name or "target" in name or "command" in name
        is_torque = "tau" in name or "torque" in name
        if signal == "dq_des" and is_velocity and is_desired and not is_torque:
            matches.append(column)
        elif signal == "dq" and is_velocity and not is_desired and not is_torque:
            matches.append(column)
        elif signal == "tau" and is_torque:
            matches.append(column)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {side} wheel {signal} columns: {matches}. "
            "Rename the desired column or add an exact candidate."
        )
    return None


def identify_wheel_columns(columns: list[str]) -> WheelColumns:
    result = WheelColumns(
        timestamp=_find_timestamp(columns),
        left_dq_des=_find_wheel_signal(columns, "left", "dq_des") or "",
        right_dq_des=_find_wheel_signal(columns, "right", "dq_des") or "",
        left_dq=_find_wheel_signal(columns, "left", "dq") or "",
        right_dq=_find_wheel_signal(columns, "right", "dq") or "",
        left_tau=_find_wheel_signal(columns, "left", "tau"),
        right_tau=_find_wheel_signal(columns, "right", "tau"),
    )
    missing = [
        key
        for key, value in asdict(result).items()
        if key not in {"left_tau", "right_tau"} and not value
    ]
    if missing:
        raise ValueError(f"Missing required wheel CSV fields: {missing}")
    return result


def _timestamp_seconds(values: np.ndarray, column_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    name = _normalized_name(column_name)
    if name.endswith("_ns") or np.nanmedian(np.abs(values)) > 1.0e14:
        scale = 1.0e9
    elif name.endswith("_us"):
        scale = 1.0e6
    elif name.endswith("_ms"):
        scale = 1.0e3
    else:
        scale = 1.0
    return (values - values[0]) / scale


def load_wheel_csv(
    csv_path: str | Path,
    target_mode: TargetMode = "auto",
    minimum_torque_coverage: float = 0.95,
) -> WheelSession:
    """Load the valid prefix/body of a wheel actuator CSV.

    Invalid or NUL-padded tail rows become non-numeric and are discarded.  Rows
    with non-increasing timestamps are also discarded while preserving order.
    """

    csv_path = Path(csv_path).expanduser().resolve()
    frame = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    source_rows = len(frame)
    columns = identify_wheel_columns(list(frame.columns))

    base_names = [
        columns.timestamp,
        columns.left_dq_des,
        columns.right_dq_des,
        columns.left_dq,
        columns.right_dq,
    ]
    numeric = frame[base_names].apply(pd.to_numeric, errors="coerce")
    base_valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)

    torque_available = columns.left_tau is not None and columns.right_tau is not None
    torque_numeric: pd.DataFrame | None = None
    torque_valid = np.zeros(len(frame), dtype=bool)
    if torque_available:
        torque_numeric = frame[[columns.left_tau, columns.right_tau]].apply(
            pd.to_numeric, errors="coerce"
        )
        torque_valid = np.isfinite(
            torque_numeric.to_numpy(dtype=np.float64)
        ).all(axis=1)

    if target_mode == "torque" and not torque_available:
        raise ValueError("target_mode=torque, but both wheel torque columns were not found")
    if target_mode == "auto":
        coverage = float(np.mean(torque_valid[base_valid])) if np.any(base_valid) else 0.0
        enough_variation = False
        if torque_available and np.any(base_valid & torque_valid):
            tau_values = torque_numeric.loc[base_valid & torque_valid].to_numpy(
                dtype=np.float64
            )
            enough_variation = bool(np.all(np.std(tau_values, axis=0) > 1.0e-4))
        selected_mode: Literal["torque", "next_velocity"] = (
            "torque"
            if coverage >= minimum_torque_coverage and enough_variation
            else "next_velocity"
        )
    else:
        selected_mode = target_mode

    valid = base_valid & torque_valid if selected_mode == "torque" else base_valid
    numeric = numeric.loc[valid].reset_index(drop=True)
    if torque_numeric is not None:
        torque_numeric = torque_numeric.loc[valid].reset_index(drop=True)

    raw_timestamp = numeric[columns.timestamp].to_numpy(dtype=np.float64)
    timestamp_s = _timestamp_seconds(raw_timestamp, columns.timestamp)
    monotonic = np.ones(len(timestamp_s), dtype=bool)
    last = -np.inf
    for index, timestamp in enumerate(timestamp_s):
        if timestamp <= last:
            monotonic[index] = False
        else:
            last = timestamp
    numeric = numeric.loc[monotonic].reset_index(drop=True)
    timestamp_s = timestamp_s[monotonic]
    if torque_numeric is not None:
        torque_numeric = torque_numeric.loc[monotonic].reset_index(drop=True)

    if len(timestamp_s) < 3:
        raise ValueError("Too few valid timestamped rows after CSV filtering")
    dt = np.diff(timestamp_s)
    median_dt_s = float(np.median(dt))

    dq_des = numeric[[columns.left_dq_des, columns.right_dq_des]].to_numpy(
        dtype=np.float32
    )
    dq = numeric[[columns.left_dq, columns.right_dq]].to_numpy(dtype=np.float32)
    tau = None
    if selected_mode == "torque":
        assert torque_numeric is not None
        tau = torque_numeric.to_numpy(dtype=np.float32)

    return WheelSession(
        timestamp_s=timestamp_s.astype(np.float64),
        dq_des=dq_des,
        dq=dq,
        tau=tau,
        columns=columns,
        target_mode=selected_mode,
        source_rows=source_rows,
        valid_rows=len(timestamp_s),
        median_dt_s=median_dt_s,
    )


def build_examples(
    session: WheelSession,
    history_length: int,
    train_fraction: float,
    subset: Literal["train", "val"],
    minimum_dt_s: float,
    maximum_gap_s: float,
    history_offsets_s: tuple[float, ...] | None = None,
    history_order: HistoryOrder = "oldest_to_current",
    timestamp_tolerance_s: float | None = None,
) -> WheelExamples:
    """Build left/right examples after a continuous-in-time row split.

    With ``history_offsets_s=None`` this retains the V1 behavior: consecutive
    rows are returned oldest-to-current.  Otherwise offsets are non-negative
    ages relative to the target row (0 means the target timestamp), and each
    age is matched against the real timestamps rather than a row stride.
    """

    if not 2 <= history_length <= 100:
        raise ValueError("history_length must be between 2 and 100")
    if not 0.5 < train_fraction < 0.95:
        raise ValueError("train_fraction must be between 0.5 and 0.95")
    if not 0.0 <= minimum_dt_s < maximum_gap_s:
        raise ValueError("Expected 0 <= minimum_dt_s < maximum_gap_s")
    if history_order not in {"oldest_to_current", "current_to_oldest"}:
        raise ValueError(f"Unsupported history_order: {history_order}")

    timestamp_history = history_offsets_s is not None
    if timestamp_history:
        offsets = np.asarray(history_offsets_s, dtype=np.float64)
        if offsets.ndim != 1 or len(offsets) != history_length:
            raise ValueError(
                "history_offsets_s must contain exactly history_length values"
            )
        if not np.all(np.isfinite(offsets)) or np.any(offsets < 0.0):
            raise ValueError("history_offsets_s must be finite and non-negative")
        if not np.isclose(offsets[0], 0.0, atol=1.0e-12):
            raise ValueError("history_offsets_s must start with the current (0 s) age")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("history_offsets_s ages must be strictly increasing")
        if timestamp_tolerance_s is None or timestamp_tolerance_s <= 0.0:
            raise ValueError(
                "timestamp_tolerance_s must be positive for timestamp histories"
            )
    else:
        offsets = np.empty(0, dtype=np.float64)
        if history_order != "oldest_to_current":
            raise ValueError("V1 contiguous history must remain oldest_to_current")

    row_count = len(session.timestamp_s)
    split_row = int(row_count * train_fraction)
    segment_start, segment_stop = (
        (0, split_row) if subset == "train" else (split_row, row_count)
    )
    transition_dt = np.diff(session.timestamp_s)
    transition_ok = (transition_dt >= minimum_dt_s) & (
        transition_dt <= maximum_gap_s
    )

    inputs: list[np.ndarray] = []
    targets: list[float] = []
    timestamps: list[float] = []
    history_timestamps: list[np.ndarray] = []
    history_match_errors: list[np.ndarray] = []
    side_ids: list[int] = []
    rejected = 0

    segment_timestamps = session.timestamp_s[segment_start:segment_stop]

    def nearest_segment_row(requested_timestamp: float) -> int:
        """Return the closest row, preferring the earlier row on an exact tie."""

        insertion = int(np.searchsorted(segment_timestamps, requested_timestamp))
        candidates: list[int] = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(segment_timestamps):
            candidates.append(insertion)
        local_index = min(
            candidates,
            key=lambda index: (
                abs(segment_timestamps[index] - requested_timestamp),
                segment_timestamps[index],
            ),
        )
        return segment_start + local_index

    for side_id in (0, 1):
        desired = session.dq_des[:, side_id]
        actual = session.dq[:, side_id]
        velocity_error = desired - actual
        last_history_end = segment_stop - (1 if session.target_mode == "torque" else 2)
        first_history_end = segment_start + history_length - 1
        if timestamp_history:
            first_history_end = int(
                np.searchsorted(
                    session.timestamp_s,
                    session.timestamp_s[segment_start] + offsets[-1],
                    side="left",
                )
            )
            first_history_end = max(segment_start, first_history_end)

        for history_end in range(first_history_end, last_history_end + 1):
            if timestamp_history:
                current_timestamp = float(session.timestamp_s[history_end])
                requested_current_to_oldest = current_timestamp - offsets
                indices_current_to_oldest = np.asarray(
                    [
                        history_end
                        if age == 0.0
                        else nearest_segment_row(float(requested))
                        for age, requested in zip(
                            offsets, requested_current_to_oldest
                        )
                    ],
                    dtype=np.int64,
                )
                matched_current_to_oldest = session.timestamp_s[
                    indices_current_to_oldest
                ]
                errors_current_to_oldest = (
                    matched_current_to_oldest - requested_current_to_oldest
                )
                indices_are_causal = (
                    indices_current_to_oldest[0] == history_end
                    and np.all(np.diff(indices_current_to_oldest) < 0)
                )
                within_tolerance = np.all(
                    np.abs(errors_current_to_oldest)
                    <= float(timestamp_tolerance_s) + 1.0e-12
                )
                oldest_row = int(indices_current_to_oldest[-1])
                transition_stop = (
                    history_end
                    if session.target_mode == "torque"
                    else history_end + 1
                )
                continuous = np.all(transition_ok[oldest_row:transition_stop])
                if not (indices_are_causal and within_tolerance and continuous):
                    rejected += 1
                    continue

                if history_order == "current_to_oldest":
                    history_indices = indices_current_to_oldest
                    matched_timestamps = matched_current_to_oldest
                    match_errors = errors_current_to_oldest
                else:
                    history_indices = indices_current_to_oldest[::-1]
                    matched_timestamps = matched_current_to_oldest[::-1]
                    match_errors = errors_current_to_oldest[::-1]
            else:
                history_start = history_end - history_length + 1
                transition_stop = (
                    history_end
                    if session.target_mode == "torque"
                    else history_end + 1
                )
                if not np.all(transition_ok[history_start:transition_stop]):
                    rejected += 1
                    continue
                history_indices = np.arange(history_start, history_end + 1)
                matched_timestamps = session.timestamp_s[history_indices]
                match_errors = None

            # Match the official actuator-net semantics: measured velocity
            # history first, followed by controller tracking-error history.
            features = np.concatenate(
                (
                    actual[history_indices],
                    velocity_error[history_indices],
                )
            )
            if session.target_mode == "torque":
                assert session.tau is not None
                # The measured-torque label and the 0 ms history sample are
                # deliberately anchored to the same timestamped CSV row.
                target = float(session.tau[history_end, side_id])
                target_row = history_end
            else:
                target = float(actual[history_end + 1])
                target_row = history_end + 1
            inputs.append(features.astype(np.float32, copy=False))
            targets.append(target)
            timestamps.append(float(session.timestamp_s[target_row]))
            history_timestamps.append(matched_timestamps.astype(np.float64, copy=False))
            if match_errors is not None:
                history_match_errors.append(
                    match_errors.astype(np.float64, copy=False)
                )
            side_ids.append(side_id)

    if not inputs:
        raise ValueError(f"No valid {subset} windows were produced")
    return WheelExamples(
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        history_timestamps_s=np.stack(history_timestamps).astype(np.float64),
        history_match_errors_s=(
            np.stack(history_match_errors).astype(np.float64)
            if timestamp_history
            else None
        ),
        side_ids=np.asarray(side_ids, dtype=np.int64),
        rejected_windows=rejected,
    )


def fit_normalization(examples: WheelExamples, epsilon: float = 1.0e-6) -> Normalization:
    input_mean = examples.inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = examples.inputs.std(axis=0, dtype=np.float64).astype(np.float32)
    target_mean = examples.targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std = examples.targets.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(input_std, epsilon)
    target_std = np.maximum(target_std, epsilon)
    return Normalization(input_mean, input_std, target_mean, target_std)


def normalize_examples(
    examples: WheelExamples, normalization: Normalization
) -> tuple[np.ndarray, np.ndarray]:
    inputs = (examples.inputs - normalization.input_mean) / normalization.input_std
    targets = (examples.targets - normalization.target_mean) / normalization.target_std
    return inputs.astype(np.float32), targets.astype(np.float32)

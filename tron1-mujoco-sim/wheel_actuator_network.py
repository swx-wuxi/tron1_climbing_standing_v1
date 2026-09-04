"""Experimental wheel actuator-network adapter for the MuJoCo simulator."""

from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path

import numpy as np
import torch


class WheelActuatorNetwork:
    """Run one shared torque network for a checkpoint-defined actuator group.

    Old V1/V2 checkpoints contain no joint metadata and therefore retain the
    original wheel-only behaviour.  Reference-style checkpoints explicitly
    list their joints and whether the second signal is velocity or position
    tracking error.
    """

    WHEEL_NAMES = ("wheel_L_Joint", "wheel_R_Joint")

    def __init__(
        self,
        checkpoint_path: str | Path,
        joint_names: list[str],
        simulation_dt: float,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"wheel actuator checkpoint does not exist: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.actuator_group = str(checkpoint.get("actuator_group", "wheels"))
        self.joint_names = tuple(checkpoint.get("joint_names", self.WHEEL_NAMES))
        self.tracking_error_type = str(
            checkpoint.get("tracking_error_type", "velocity")
        )
        if self.tracking_error_type not in {"velocity", "position"}:
            raise ValueError(
                "unsupported actuator tracking_error_type: "
                f"{self.tracking_error_type}"
            )

        missing_names = [name for name in self.joint_names if name not in joint_names]
        if missing_names:
            raise ValueError(
                "actuator network checkpoint requires joints: "
                + ", ".join(missing_names)
            )
        self.joint_indices = {joint_names.index(name) for name in self.joint_names}
        # Compatibility for existing tests and users of the wheel-only class.
        self.wheel_indices = self.joint_indices
        if checkpoint.get("target_mode") != "torque":
            raise ValueError(
                "MuJoCo wheel actuator integration requires a torque checkpoint"
            )

        model_path = self.checkpoint_path.parent.parent / "actuator_model.py"
        spec = importlib.util.spec_from_file_location(
            "tron1_wheel_actuator_model", model_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load actuator model definition: {model_path}")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        self.model = model_module.make_model(
            int(checkpoint["input_dim"]), checkpoint["hidden_sizes"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.history_length = int(checkpoint["history_length"])
        if int(checkpoint["input_dim"]) != 2 * self.history_length:
            raise ValueError(
                "wheel actuator checkpoint input_dim must equal "
                "2 * history_length"
            )
        self.input_mean = checkpoint["input_mean"].detach().cpu().numpy()
        self.input_std = checkpoint["input_std"].detach().cpu().numpy()
        self.target_mean = float(checkpoint["target_mean"].item())
        self.target_std = float(checkpoint["target_std"].item())

        training_dt = float(checkpoint["median_dt_s"])
        self.simulation_dt = float(simulation_dt)
        run_every_simulation_step = bool(
            checkpoint.get("run_every_simulation_step", False)
        )
        self.update_interval = (
            1
            if run_every_simulation_step
            else max(1, int(round(training_dt / simulation_dt)))
        )
        effective_dt = self.update_interval * simulation_dt
        if (
            not run_every_simulation_step
            and abs(effective_dt - training_dt) > 0.25 * training_dt
        ):
            raise ValueError(
                "simulation timestep cannot reproduce actuator training rate: "
                f"training_dt={training_dt:.6f}s, simulation_dt={simulation_dt:.6f}s"
            )

        checkpoint_offsets = checkpoint.get("history_offsets_s")
        if checkpoint_offsets is None:
            # Backward-compatible V1 timing: consecutive samples at the
            # checkpoint's training rate, presented oldest-to-current.
            self.history_offsets_s = (
                np.arange(self.history_length, dtype=np.float64) * training_dt
            )
            self.history_order = checkpoint.get(
                "history_order", "oldest_to_current"
            )
            self.timestamp_tolerance_s = 0.51 * self.simulation_dt
        else:
            self.history_offsets_s = np.asarray(
                checkpoint_offsets, dtype=np.float64
            )
            self.history_order = checkpoint.get("history_order")
            self.timestamp_tolerance_s = float(
                checkpoint.get("timestamp_tolerance_s", 0.51 * simulation_dt)
            )

        if (
            self.history_offsets_s.shape != (self.history_length,)
            or not np.isclose(self.history_offsets_s[0], 0.0, atol=1.0e-12)
            or np.any(np.diff(self.history_offsets_s) <= 0.0)
        ):
            raise ValueError(
                "checkpoint history offsets must start at 0 and increase"
            )
        if self.history_order not in {
            "current_to_oldest",
            "oldest_to_current",
        }:
            raise ValueError(
                f"unsupported checkpoint history_order: {self.history_order}"
            )
        if self.timestamp_tolerance_s <= 0.0:
            raise ValueError("checkpoint timestamp tolerance must be positive")

        self.maximum_gap_s = float(
            checkpoint.get("maximum_gap_s", 2.5 * self.simulation_dt)
        )
        self.maximum_history_age_s = float(self.history_offsets_s[-1])
        buffer_length = (
            int(
                np.ceil(
                    (
                        self.maximum_history_age_s
                        + self.timestamp_tolerance_s
                    )
                    / self.simulation_dt
                )
            )
            + 2
        )
        self.sample_histories = {
            index: deque(maxlen=buffer_length) for index in self.joint_indices
        }
        self.last_torques = {index: 0.0 for index in self.joint_indices}
        self.prediction_valid = {index: False for index in self.joint_indices}

        display_offsets = (
            -self.history_offsets_s
            if self.history_order == "current_to_oldest"
            else -self.history_offsets_s[::-1]
        )
        offsets_ms = ", ".join(
            "0" if np.isclose(offset, 0.0) else f"{offset * 1000.0:g}"
            for offset in display_offsets
        )

        print(
            f"*** EXPERIMENTAL {self.actuator_group} actuator network LOADED "
            "(inactive): "
            f"{self.checkpoint_path} "
            f"(history=[{offsets_ms}] ms {self.history_order}, update every "
            f"{self.update_interval} MuJoCo steps) ***"
        )

    def handles(self, joint_index: int) -> bool:
        return joint_index in self.joint_indices

    def reset(self, clear_history: bool = False) -> None:
        """Reset held predictions, optionally discarding timestamped history."""
        if clear_history:
            for history in self.sample_histories.values():
                history.clear()
        for joint_index in self.last_torques:
            self.last_torques[joint_index] = 0.0
            self.prediction_valid[joint_index] = False

    def observe(
        self,
        joint_index: int,
        simulation_time_s: float,
        desired_value: float,
        actual_dq: float,
        actual_position: float | None = None,
    ) -> None:
        """Record one simulation-step sample, even while the net is inactive."""
        if joint_index not in self.joint_indices:
            raise ValueError(f"joint index {joint_index} is not handled")

        if self.tracking_error_type == "velocity":
            tracking_error = float(desired_value) - float(actual_dq)
        else:
            if actual_position is None:
                raise ValueError("position-error actuator observation needs q_actual")
            tracking_error = float(desired_value) - float(actual_position)

        sample = (
            float(simulation_time_s),
            float(actual_dq),
            tracking_error,
        )
        history = self.sample_histories[joint_index]
        if history and sample[0] <= history[-1][0]:
            if np.isclose(
                sample[0], history[-1][0], rtol=0.0, atol=1.0e-12
            ):
                history[-1] = sample
                return
            history.clear()
            self.prediction_valid[joint_index] = False
        elif history and sample[0] - history[-1][0] > self.maximum_gap_s:
            history.clear()
            self.prediction_valid[joint_index] = False
        history.append(sample)

    def _history_features(self, joint_index: int) -> np.ndarray | None:
        history = self.sample_histories[joint_index]
        if not history:
            return None

        samples = np.asarray(history, dtype=np.float64)
        timestamps = samples[:, 0]
        current_time = timestamps[-1]
        if (
            current_time - timestamps[0]
            < self.maximum_history_age_s - 1.0e-12
        ):
            return None

        requested = current_time - self.history_offsets_s
        indices_current_to_oldest = np.asarray(
            [int(np.argmin(np.abs(timestamps - timestamp))) for timestamp in requested],
            dtype=np.int64,
        )
        matched = timestamps[indices_current_to_oldest]
        if np.any(
            np.abs(matched - requested)
            > self.timestamp_tolerance_s + 1.0e-12
        ):
            return None
        if np.any(np.diff(indices_current_to_oldest) >= 0):
            return None

        if self.history_order == "current_to_oldest":
            feature_indices = indices_current_to_oldest
        else:
            feature_indices = indices_current_to_oldest[::-1]

        actual_dq = samples[feature_indices, 1].astype(np.float32)
        tracking_error = samples[feature_indices, 2].astype(np.float32)
        return np.concatenate((actual_dq, tracking_error)).astype(np.float32)

    def histories_ready(self) -> bool:
        """Return whether every wheel has a complete checkpoint history."""
        return all(
            self._history_features(joint_index) is not None
            for joint_index in self.joint_indices
        )

    @torch.no_grad()
    def torque(
        self,
        joint_index: int,
        frame_count: int,
        fallback_torque: float,
    ) -> float:
        """Return a checkpoint-rate prediction, or baseline torque during warm-up."""
        if joint_index not in self.joint_indices:
            raise ValueError(f"joint index {joint_index} is not handled")
        features = self._history_features(joint_index)
        if features is None:
            self.prediction_valid[joint_index] = False
            self.last_torques[joint_index] = float(fallback_torque)
            return float(fallback_torque)
        if frame_count % self.update_interval != 0:
            return (
                self.last_torques[joint_index]
                if self.prediction_valid[joint_index]
                else float(fallback_torque)
            )

        normalized = (features - self.input_mean) / self.input_std
        normalized_torque = self.model(
            torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        ).item()
        torque = normalized_torque * self.target_std + self.target_mean
        self.last_torques[joint_index] = float(torque)
        self.prediction_valid[joint_index] = True
        return self.last_torques[joint_index]

"""Experimental wheel actuator-network adapter for the MuJoCo simulator."""

from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path

import numpy as np
import torch


class WheelActuatorNetwork:
    """Run a shared torque network for the left and right wheel actuators."""

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

        missing_names = [name for name in self.WHEEL_NAMES if name not in joint_names]
        if missing_names:
            raise ValueError(
                "wheel actuator network requires wheel joints: "
                + ", ".join(missing_names)
            )
        self.wheel_indices = {
            joint_names.index(name) for name in self.WHEEL_NAMES
        }

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
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
        self.input_mean = checkpoint["input_mean"].detach().cpu().numpy()
        self.input_std = checkpoint["input_std"].detach().cpu().numpy()
        self.target_mean = float(checkpoint["target_mean"].item())
        self.target_std = float(checkpoint["target_std"].item())

        training_dt = float(checkpoint["median_dt_s"])
        self.update_interval = max(1, int(round(training_dt / simulation_dt)))
        effective_dt = self.update_interval * simulation_dt
        if abs(effective_dt - training_dt) > 0.25 * training_dt:
            raise ValueError(
                "simulation timestep cannot reproduce actuator training rate: "
                f"training_dt={training_dt:.6f}s, simulation_dt={simulation_dt:.6f}s"
            )

        self.desired_histories = {
            index: deque(maxlen=self.history_length) for index in self.wheel_indices
        }
        self.actual_histories = {
            index: deque(maxlen=self.history_length) for index in self.wheel_indices
        }
        self.last_torques = {index: 0.0 for index in self.wheel_indices}

        print(
            "*** EXPERIMENTAL wheel actuator network ENABLED: "
            f"{self.checkpoint_path} "
            f"(history={self.history_length}, update every "
            f"{self.update_interval} MuJoCo steps) ***"
        )

    def handles(self, joint_index: int) -> bool:
        return joint_index in self.wheel_indices

    @torch.no_grad()
    def torque(
        self,
        joint_index: int,
        frame_count: int,
        desired_dq: float,
        actual_dq: float,
    ) -> float:
        """Return a new 500 Hz prediction or hold the preceding prediction."""
        if joint_index not in self.wheel_indices:
            raise ValueError(f"joint index {joint_index} is not a wheel")
        if frame_count % self.update_interval != 0:
            return self.last_torques[joint_index]

        desired_history = self.desired_histories[joint_index]
        actual_history = self.actual_histories[joint_index]
        if not desired_history:
            desired_history.extend([float(desired_dq)] * self.history_length)
            actual_history.extend([float(actual_dq)] * self.history_length)
        else:
            desired_history.append(float(desired_dq))
            actual_history.append(float(actual_dq))

        desired = np.asarray(desired_history, dtype=np.float32)
        actual = np.asarray(actual_history, dtype=np.float32)
        features = np.concatenate((actual, desired - actual))
        normalized = (features - self.input_mean) / self.input_std
        normalized_torque = self.model(
            torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        ).item()
        torque = normalized_torque * self.target_std + self.target_mean
        self.last_torques[joint_index] = float(torque)
        return self.last_torques[joint_index]

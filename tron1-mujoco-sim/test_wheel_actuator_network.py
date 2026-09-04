"""Regression tests for timestamped actuator-network inference history."""

from __future__ import annotations

from collections import deque
import unittest

import numpy as np

from wheel_actuator_network import WheelActuatorNetwork


def make_adapter() -> WheelActuatorNetwork:
    adapter = WheelActuatorNetwork.__new__(WheelActuatorNetwork)
    adapter.wheel_indices = {0, 1}
    adapter.joint_indices = adapter.wheel_indices
    adapter.tracking_error_type = "velocity"
    adapter.history_offsets_s = np.asarray(
        (0.0, 0.010, 0.020, 0.030, 0.040), dtype=np.float64
    )
    adapter.history_order = "current_to_oldest"
    adapter.timestamp_tolerance_s = 0.0025
    adapter.maximum_gap_s = 0.005
    adapter.maximum_history_age_s = 0.040
    adapter.sample_histories = {
        0: deque(maxlen=45),
        1: deque(maxlen=45),
    }
    adapter.last_torques = {0: 0.0, 1: 0.0}
    adapter.prediction_valid = {0: False, 1: False}
    return adapter


class TimestampedInferenceHistoryTests(unittest.TestCase):
    def test_v2_uses_0_to_minus_40_ms_current_to_oldest(self) -> None:
        adapter = make_adapter()
        for step in range(40):
            adapter.observe(0, step * 0.001, 100.0 + step, float(step))
        self.assertIsNone(adapter._history_features(0))

        adapter.observe(0, 0.040, 140.0, 40.0)
        features = adapter._history_features(0)
        assert features is not None
        np.testing.assert_array_equal(
            features[:5],
            np.asarray((40.0, 30.0, 20.0, 10.0, 0.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            features[5:], np.full(5, 100.0, dtype=np.float32)
        )

    def test_observing_while_inactive_makes_both_wheels_ready(self) -> None:
        adapter = make_adapter()
        for step in range(41):
            timestamp = step * 0.001
            adapter.observe(0, timestamp, float(step), float(step))
            adapter.observe(1, timestamp, float(step), float(step))
        self.assertTrue(adapter.histories_ready())

    def test_toggle_reset_preserves_history_unless_explicitly_cleared(self) -> None:
        adapter = make_adapter()
        for step in range(41):
            adapter.observe(0, step * 0.001, float(step), float(step))
        adapter.prediction_valid[0] = True

        adapter.reset(clear_history=False)
        self.assertIsNotNone(adapter._history_features(0))
        self.assertFalse(adapter.prediction_valid[0])

        adapter.reset(clear_history=True)
        self.assertIsNone(adapter._history_features(0))

    def test_oldest_to_current_checkpoint_order_is_supported(self) -> None:
        adapter = make_adapter()
        adapter.history_order = "oldest_to_current"
        for step in range(41):
            adapter.observe(0, step * 0.001, 100.0 + step, float(step))

        features = adapter._history_features(0)
        assert features is not None
        np.testing.assert_array_equal(
            features[:5],
            np.asarray((0.0, 10.0, 20.0, 30.0, 40.0), dtype=np.float32),
        )

    def test_backward_simulation_time_discards_stale_history(self) -> None:
        adapter = make_adapter()
        adapter.observe(0, 100.0, 1.0, 1.0)
        adapter.observe(0, 99.999, 2.0, 2.0)
        self.assertEqual(list(adapter.sample_histories[0]), [(99.999, 2.0, 0.0)])

    def test_position_tracking_error_uses_q_des_minus_q_actual(self) -> None:
        adapter = make_adapter()
        adapter.tracking_error_type = "position"
        for step in range(41):
            adapter.observe(0, step * 0.001, 1.5, float(step), 1.0)
        features = adapter._history_features(0)
        assert features is not None
        np.testing.assert_array_equal(
            features[5:], np.full(5, 0.5, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()

"""Timing regression tests for wheel actuator-network history construction."""

from __future__ import annotations

import unittest

import numpy as np

from actuator_dataset import WheelColumns, WheelSession, build_examples


V2_OFFSETS_S = (0.0, 0.010, 0.020, 0.030, 0.040)


def make_session(timestamps_s: np.ndarray) -> WheelSession:
    rows = np.arange(len(timestamps_s), dtype=np.float32)
    dq = np.column_stack((rows, rows + 1000.0)).astype(np.float32)
    dq_des = dq + np.asarray((100.0, 200.0), dtype=np.float32)
    tau = np.column_stack((rows * 10.0, rows * 10.0 + 1.0)).astype(np.float32)
    return WheelSession(
        timestamp_s=np.asarray(timestamps_s, dtype=np.float64),
        dq_des=dq_des,
        dq=dq,
        tau=tau,
        columns=WheelColumns(
            timestamp="timestamp_ns",
            left_dq_des="wheel_l_dq_des_rad_s",
            right_dq_des="wheel_r_dq_des_rad_s",
            left_dq="wheel_l_dq_rad_s",
            right_dq="wheel_r_dq_rad_s",
            left_tau="wheel_l_tau_feedback_nm",
            right_tau="wheel_r_tau_feedback_nm",
        ),
        target_mode="torque",
        source_rows=len(timestamps_s),
        valid_rows=len(timestamps_s),
        median_dt_s=float(np.median(np.diff(timestamps_s))),
    )


class TimestampHistoryTests(unittest.TestCase):
    def test_v2_uses_real_time_offsets_and_current_torque_target(self) -> None:
        session = make_session(np.arange(151, dtype=np.float64) * 0.002)
        examples = build_examples(
            session,
            history_length=5,
            train_fraction=0.8,
            subset="train",
            minimum_dt_s=0.00025,
            maximum_gap_s=0.005,
            history_offsets_s=V2_OFFSETS_S,
            history_order="current_to_oldest",
            timestamp_tolerance_s=0.0025,
        )

        # The first anchor is row 20 (40 ms).  V2 follows the reference
        # current-to-past convention: rows 20, 15, 10, 5, 0.
        np.testing.assert_array_equal(
            examples.inputs[0, :5],
            np.asarray((20.0, 15.0, 10.0, 5.0, 0.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            examples.inputs[0, 5:],
            np.full(5, 100.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            examples.history_timestamps_s[0],
            np.asarray((0.040, 0.030, 0.020, 0.010, 0.0)),
            atol=1.0e-12,
        )
        self.assertEqual(examples.timestamps_s[0], 0.040)
        self.assertEqual(examples.targets[0, 0], 200.0)
        assert examples.history_match_errors_s is not None
        np.testing.assert_allclose(
            examples.history_match_errors_s[0], 0.0, atol=1.0e-12
        )

    def test_v2_matches_nearest_timestamp_and_reports_error(self) -> None:
        timestamps = np.arange(151, dtype=np.float64) * 0.002
        timestamps[15] += 0.0004
        session = make_session(timestamps)
        examples = build_examples(
            session,
            history_length=5,
            train_fraction=0.8,
            subset="train",
            minimum_dt_s=0.00025,
            maximum_gap_s=0.005,
            history_offsets_s=V2_OFFSETS_S,
            history_order="current_to_oldest",
            timestamp_tolerance_s=0.0025,
        )

        self.assertEqual(examples.inputs[0, 1], 15.0)
        assert examples.history_match_errors_s is not None
        self.assertAlmostEqual(examples.history_match_errors_s[0, 1], 0.0004)
        self.assertEqual(examples.timestamps_s[0], examples.history_timestamps_s[0, 0])

    def test_v2_does_not_cross_train_validation_boundary(self) -> None:
        session = make_session(np.arange(151, dtype=np.float64) * 0.002)
        examples = build_examples(
            session,
            history_length=5,
            train_fraction=0.8,
            subset="val",
            minimum_dt_s=0.00025,
            maximum_gap_s=0.005,
            history_offsets_s=V2_OFFSETS_S,
            history_order="current_to_oldest",
            timestamp_tolerance_s=0.0025,
        )

        split_timestamp = session.timestamp_s[int(len(session.timestamp_s) * 0.8)]
        self.assertTrue(np.all(examples.history_timestamps_s >= split_timestamp))

    def test_v1_consecutive_row_behavior_is_unchanged(self) -> None:
        session = make_session(np.arange(40, dtype=np.float64) * 0.002)
        examples = build_examples(
            session,
            history_length=8,
            train_fraction=0.8,
            subset="train",
            minimum_dt_s=0.00025,
            maximum_gap_s=0.005,
        )

        np.testing.assert_array_equal(
            examples.inputs[0, :8], np.arange(8, dtype=np.float32)
        )
        self.assertEqual(examples.targets[0, 0], 70.0)
        self.assertIsNone(examples.history_match_errors_s)


if __name__ == "__main__":
    unittest.main()

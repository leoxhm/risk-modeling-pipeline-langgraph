"""Tests for model discrimination, calibration and uncertainty metrics."""

from __future__ import annotations

import unittest

from modeling.evaluation import (
    bootstrap_metric_intervals,
    build_calibration_table,
    calculate_brier_score,
    calculate_pr_auc,
    evaluate_binary_model,
)


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        self.scores = [0.05, 0.10, 0.85, 0.20, 0.70, 0.90, 0.15, 0.65, 0.30, 0.80]

    def test_extended_metrics_are_returned(self) -> None:
        result = evaluate_binary_model(self.labels, self.scores)
        self.assertIsNotNone(result.pr_auc)
        self.assertIsNotNone(result.brier_score)
        self.assertIsNotNone(result.calibration)
        self.assertEqual(result.lift.height, 10)
        self.assertIn("cumulative_bad_capture", result.lift.columns)
        self.assertAlmostEqual(float(result.lift["cumulative_sample_rate"][-1]), 1.0)
        self.assertAlmostEqual(float(result.lift["cumulative_bad_capture"][-1]), 1.0)

    def test_pr_auc_brier_calibration_and_bootstrap(self) -> None:
        self.assertGreater(calculate_pr_auc(self.labels, self.scores) or 0.0, 0.5)
        brier = calculate_brier_score(self.labels, self.scores)
        self.assertIsNotNone(brier)
        self.assertGreaterEqual(brier or 0.0, 0.0)
        self.assertLessEqual(brier or 0.0, 1.0)
        calibration = build_calibration_table(self.labels, self.scores, bins=5)
        self.assertEqual(calibration.height, 5)
        intervals = bootstrap_metric_intervals(self.labels, self.scores, rounds=30, seed=7)
        self.assertEqual(set(intervals["metric"].to_list()), {"auc", "ks", "pr_auc", "brier_score"})


if __name__ == "__main__":
    unittest.main()

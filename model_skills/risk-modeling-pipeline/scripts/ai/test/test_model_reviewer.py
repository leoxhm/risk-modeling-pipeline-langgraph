"""Coverage for evidence-backed model diagnostics."""

from __future__ import annotations

import unittest

import polars as pl

from ai.model_reviewer import build_model_review


class ModelReviewerTest(unittest.TestCase):
    def test_review_covers_gaps_drift_lift_months_and_recommendations(self) -> None:
        metrics = pl.DataFrame(
            {
                "dataset": ["train", "test", "oot"],
                "sample_count": [1000, 300, 300],
                "bad_count": [100, 30, 30],
                "auc": [0.90, 0.70, 0.58],
                "ks": [0.70, 0.50, 0.35],
                "score_psi_vs_train": [0.0, 0.30, 0.12],
            }
        )
        history = pl.DataFrame(
            {
                "train_auc": [0.65, 0.90],
                "test_auc": [0.64, 0.70],
            }
        )
        lift = pl.DataFrame(
            {
                "dataset": ["test"] * 5,
                "bucket": [1, 2, 3, 4, 5],
                "bad_rate": [0.30, 0.20, 0.25, 0.10, 0.15],
                "lift": [1.5, 1.0, 1.2, 0.5, 0.4],
            }
        )
        monthly = pl.DataFrame(
            {
                "dataset": ["oot"] * 3,
                "sample_count": [50, 50, 50],
                "bad_count": [10, 10, 10],
                "auc": [0.80, 0.60, 0.75],
                "ks": [0.50, 0.20, 0.45],
                "bad_rate": [0.10, 0.20, 0.12],
            }
        )
        importance = pl.DataFrame(
            {"feature": ["leaky_feature", "other"], "gain_importance": [80.0, 20.0]}
        )

        result = build_model_review(
            metrics,
            best_iteration=8,
            feature_importance=importance,
            training_history=history,
            lift_detail=lift,
            monthly_performance=monthly,
        )
        codes = {finding["code"] for finding in result["findings"]}
        self.assertIn("TRAIN_TEST_AUC_GAP_CRITICAL", codes)
        self.assertIn("TRAIN_TEST_KS_GAP_CRITICAL", codes)
        self.assertIn("OOT_AUC_DROP", codes)
        self.assertIn("TEST_SCORE_PSI_HIGH", codes)
        self.assertIn("LIFT_ORDERING_WEAK", codes)
        self.assertIn("MONTHLY_PERFORMANCE_UNSTABLE", codes)
        self.assertIn("FEATURE_IMPORTANCE_CONCENTRATED", codes)
        self.assertTrue(result["recommendations"])
        self.assertIn("evidence", result)
        self.assertIn("thresholds", result["evidence"])
        self.assertEqual(result["diagnosis"], result["findings"])
        self.assertTrue(result["requires_user_confirmation"])
        self.assertEqual(result["status"], "blocked")

    def test_review_flags_underfit_when_both_splits_are_weak(self) -> None:
        metrics = pl.DataFrame(
            {
                "dataset": ["train", "test", "oot"],
                "sample_count": [1000, 300, 300],
                "bad_count": [100, 30, 30],
                "auc": [0.54, 0.53, 0.52],
                "ks": [0.08, 0.07, 0.06],
            }
        )
        result = build_model_review(
            metrics,
            best_iteration=50,
            feature_importance=pl.DataFrame(
                {"feature": ["f1"], "gain_importance": [1.0]}
            ),
        )
        codes = {finding["code"] for finding in result["findings"]}
        self.assertIn("MODEL_UNDERFIT_CRITICAL", codes)
        self.assertTrue(any(item["category"] == "underfit" for item in result["recommendations"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from modeling.config import LightGbmConfig, TuningSearchSpace
from modeling.llm_tuning import (
    _bounded_config,
    _csv_safe_value,
    _history_context,
    _parse_json,
    _validate_metric,
)


class LlmTuningGuardrailTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LightGbmConfig(
            "binary", "auc", 0.03, 31, -1, 30, 0.8, 0.8, 1, 0.0, 1.0, 0.0, 1000, 100, "ks"
        )
        self.bounds = TuningSearchSpace(
            (0.01, 0.08), (7, 63), (3, 8), (30, 300), (0.6, 1.0),
            (0.6, 1.0), (1, 10), (0.0001, 20.0), (0.0001, 20.0), (0.0, 1.0)
        )

    def test_json_fence_is_supported(self) -> None:
        self.assertEqual(_parse_json('```json {"parameters":{"num_leaves":32}}```')["parameters"]["num_leaves"], 32)

    def test_out_of_bound_proposal_is_rejected(self) -> None:
        config, error = _bounded_config(self.config, {"num_leaves": 100}, self.bounds)
        self.assertIsNone(config)
        self.assertIn("硬边界", error or "")

    def test_cross_parameter_constraint_is_rejected(self) -> None:
        config, error = _bounded_config(self.config, {"max_depth": 3, "num_leaves": 16}, self.bounds)
        self.assertIsNone(config)
        self.assertIn("num_leaves", error or "")

    def test_nested_history_values_are_csv_safe(self) -> None:
        self.assertEqual(_csv_safe_value(["gap too large"]), '["gap too large"]')
        self.assertEqual(_csv_safe_value({"ks": 0.42}), '{"ks": 0.42}')

    def test_validate_metric_prefers_explicit_alias(self) -> None:
        self.assertEqual(_validate_metric({"test_ks": 0.31, "validate_ks": 0.32}, "ks"), 0.32)
        self.assertEqual(_validate_metric({"test_ks": 0.31}, "ks"), 0.31)

    def test_history_context_keeps_rejection_reason(self) -> None:
        context = _history_context([{"round": 1, "status": "rejected_metric", "rejection_reason": "提升不足"}])
        self.assertEqual(context[0]["rejection_reason"], "提升不足")


if __name__ == "__main__":
    unittest.main()

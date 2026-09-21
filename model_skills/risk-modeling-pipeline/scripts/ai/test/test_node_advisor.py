import tempfile
import unittest
from pathlib import Path

from ai.node_advisor import _validate_feature_patch, _validate_sample_patch, advise, write_recommendation


class NodeAdvisorTest(unittest.TestCase):
    def test_sample_patch_is_allow_listed_and_bounded(self):
        patch, errors = _validate_sample_patch(
            {"diagnostics": {"monthly_bad_rate_change_warning": 0.05}, "treatment": {"class_imbalance_action": "class_weight"}}
        )
        self.assertFalse(errors)
        self.assertEqual(patch["treatment"]["class_imbalance_action"], "class_weight")
        _, errors = _validate_sample_patch({"treatment": {"unknown": "drop"}})
        self.assertTrue(errors)

    def test_feature_patch_is_allow_listed(self):
        patch, errors = _validate_feature_patch({"selection": {"min_iv": 0.05, "min_month_samples": 50}})
        self.assertFalse(errors)
        self.assertEqual(patch["selection"]["min_month_samples"], 50)

    def test_disabled_advisor_is_non_blocking_and_redacts_secret(self):
        advice = advise("sample-diagnosis", current_parameters={"llm": {"enabled": False}}, evidence={})
        self.assertEqual(advice["status"], "disabled")
        with tempfile.TemporaryDirectory() as directory:
            path = write_recommendation(
                Path(directory) / "recommendation.yaml",
                node_id="sample-diagnosis",
                config_path=Path(directory) / "config.yaml",
                current_parameters={"llm": {"api_key": "secret"}},
                advice=advice,
                evidence={},
            )
            self.assertNotIn("secret", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

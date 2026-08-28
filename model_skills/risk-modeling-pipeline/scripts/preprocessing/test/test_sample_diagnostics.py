"""Checks for raw sample diagnostics and confirmed treatment actions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import polars as pl

from data.contract import DataContract, DataContractError, validate_contract
from preprocessing.sample_config import load_sample_config
from preprocessing.sample_diagnostics import (
    apply_sample_treatment,
    build_sample_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class SampleDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = DataContract(
            input_path=None,
            id_cols=("id",),
            date_col="date",
            target_col="target",
            good_label=0,
            bad_label=1,
            exclude_cols=(),
        )
        self.data = pl.DataFrame(
            {
                "id": [1, 1, 2, 3, 4, 5],
                "date": ["202501", "202501", "202503", "202503", "202504", "202504"],
                "target": [0, 0, 1, None, 0, 0],
                "all_null": [None, None, None, None, None, None],
                "feature": [1.0, 2.0, None, None, 5.0, 6.0],
            }
        )
        self.config = load_sample_config(
            PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "sample_config.yaml"
        )

    def test_diagnoses_issues_before_strict_contract_validation(self) -> None:
        permissive = validate_contract(
            self.data,
            self.contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )
        report = build_sample_diagnostics(self.data, permissive.contract, self.config)
        codes = {finding["code"] for finding in report["findings"]}

        self.assertIn("MISSING_TARGET", codes)
        self.assertIn("DUPLICATE_ID", codes)
        self.assertIn("ALL_NULL_FEATURES", codes)
        self.assertIn("MISSING_MONTHS", codes)
        self.assertEqual(report["metrics"]["duplicate_key_count"], 1)
        self.assertEqual(report["metrics"]["missing_target_count"], 1)

    def test_applies_only_configured_treatments(self) -> None:
        with self.assertRaises(DataContractError):
            apply_sample_treatment(self.data, self.contract, self.config)

        treatment = replace(
            self.config.treatment,
            missing_target_action="drop",
            incomplete_latest_month_action="keep",
        )
        result = apply_sample_treatment(
            self.data,
            self.contract,
            replace(self.config, treatment=treatment),
        )

        self.assertEqual(result.removed_missing_target_count, 1)
        self.assertEqual(result.removed_duplicate_count, 1)
        self.assertEqual(result.removed_all_null_features, ("all_null",))
        self.assertNotIn("all_null", result.data.columns)
        self.assertEqual(result.data.height, 4)


if __name__ == "__main__":
    unittest.main()

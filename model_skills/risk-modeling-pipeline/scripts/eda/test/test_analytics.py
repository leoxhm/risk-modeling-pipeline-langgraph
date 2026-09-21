"""Integration test for deterministic EDA calculations."""

from __future__ import annotations

from pathlib import Path
import unittest

from data.contract import load_contract, validate_contract
from data.loader import load_table
from preprocessing.cleaning import preprocess_data

from ..analytics import build_eda_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_PATH = PROJECT_ROOT / "data.csv"
CONTRACT_PATH = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"


class EdaAnalyticsIntegrationTest(unittest.TestCase):
    def test_build_eda_analysis(self) -> None:
        raw_data = load_table(TEST_DATA_PATH).data
        contract = validate_contract(raw_data, load_contract(CONTRACT_PATH))
        cleaned = preprocess_data(raw_data, contract)
        analysis = build_eda_analysis(cleaned.data, contract)

        self.assertEqual(analysis.univariate_overview.height, 19)
        self.assertGreater(analysis.binning_detail.height, 19)
        missing_ks = analysis.binning_detail.filter(
            analysis.binning_detail["bin"] == "MISSING"
        ).get_column("ks")
        self.assertTrue(all(value is None for value in missing_ks))
        self.assertLess(analysis.univariate_overview.get_column("ks").max(), 0.5)
        self.assertEqual(analysis.monthly_sample.get_column("sample_count").sum(), 5_000)
        baseline_month = analysis.monthly_psi.get_column("baseline_month").min()
        baseline_psi = analysis.monthly_psi.filter(
            analysis.monthly_psi["event_month"] == baseline_month
        ).get_column("psi").max()
        self.assertEqual(baseline_psi, 0.0)
        non_baseline_psi = analysis.monthly_psi.filter(
            analysis.monthly_psi["event_month"] != baseline_month
        ).get_column("psi").max()
        self.assertGreater(non_baseline_psi, 0.0)
        self.assertEqual(analysis.correlation_pairs.height, 171)
        self.assertEqual(analysis.monthly_discrimination.height, 19 * 10)
        self.assertGreaterEqual(analysis.ks_bucket.height, 19 * 2)
        self.assertIn("mean", analysis.univariate_overview.columns)
        self.assertIn("std", analysis.univariate_overview.columns)
        self.assertIn("psi", analysis.monthly_sample.columns)
        self.assertIn("cumulative_iv", analysis.binning_detail.columns)
        self.assertIn("cumulative_ks", analysis.binning_detail.columns)
        self.assertEqual(analysis.psi_summary.height, 19)
        self.assertEqual(analysis.correlation_matrix.height, 19)


if __name__ == "__main__":
    unittest.main()

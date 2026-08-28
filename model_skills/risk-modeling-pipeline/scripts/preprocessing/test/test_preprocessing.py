"""Integration test for the exploration and cleaning workflow."""

from __future__ import annotations

from pathlib import Path
import unittest

import polars as pl

from data.contract import load_contract, validate_contract
from data.loader import load_table
from preprocessing.cleaning import preprocess_data
from preprocessing.exploration import explore_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_PATH = PROJECT_ROOT / "data.csv"
CONTRACT_PATH = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"


class PreprocessingIntegrationTest(unittest.TestCase):
    def test_explore_and_preprocess_sample_csv(self) -> None:
        raw_data = load_table(TEST_DATA_PATH).data
        contract = validate_contract(raw_data, load_contract(CONTRACT_PATH))

        exploration = explore_dataset(raw_data, contract)
        result = preprocess_data(raw_data, contract)

        print("\n探索摘要：")
        for key, value in exploration.summary.items():
            print(f"  {key}: {value}")
        print("\n标签分布：")
        print(exploration.target_distribution)
        print("\n清理结果：")
        print(f"  cleaned_rows: {result.data.height}")
        print(f"  cleaned_columns: {result.data.width}")
        print(f"  retained_features: {len(result.feature_cols)}")
        print(f"  removed_duplicate_count: {result.removed_duplicate_count}")
        print(f"  invalid_date_count: {result.invalid_date_count}")
        print("\n字段决策：")
        print(result.column_decisions)

        self.assertEqual(exploration.summary["row_count"], 5_000)
        self.assertEqual(exploration.summary["candidate_feature_count"], 19)
        self.assertEqual(exploration.target_distribution.get_column("sample_count").sum(), 5_000)
        self.assertEqual(result.data.height, 5_000)
        self.assertEqual(result.data.get_column("event_date").dtype, pl.Date)
        self.assertEqual(result.data.get_column("event_month").dtype, pl.String)
        self.assertEqual(len(result.feature_cols), 19)
        self.assertEqual(result.removed_duplicate_count, 0)
        self.assertEqual(result.invalid_date_count, 0)
        self.assertEqual(result.column_decisions.filter(pl.col("decision") == "retained").height, 19)


if __name__ == "__main__":
    unittest.main()

"""Integration test for the configured chronological data split."""

from __future__ import annotations

from pathlib import Path
import unittest

from data.contract import load_contract, validate_contract
from data.loader import load_table
from preprocessing.cleaning import preprocess_data

from ..config import load_model_config
from ..split import split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class ModelSplitIntegrationTest(unittest.TestCase):
    def test_split_sample_data(self) -> None:
        raw_data = load_table(PROJECT_ROOT / "data.csv").data
        contract = validate_contract(raw_data, load_contract(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"))
        cleaned = preprocess_data(raw_data, contract)
        config = load_model_config(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml")
        result = split_dataset(cleaned.data, contract, config.split)

        self.assertEqual(result.oot_month_values, ("202511", "202512"))
        self.assertEqual(result.train.height + result.test.height + result.oot.height, cleaned.data.height)
        self.assertEqual(result.oot.height, 520)
        self.assertAlmostEqual(result.test.height / (result.train.height + result.test.height), 0.2, places=2)
        self.assertFalse(set(result.train.get_column("map_key")) & set(result.test.get_column("map_key")))
        self.assertFalse(set(result.train.get_column("map_key")) & set(result.oot.get_column("map_key")))
        self.assertFalse(set(result.test.get_column("map_key")) & set(result.oot.get_column("map_key")))
        for split_data in (result.train, result.test, result.oot):
            self.assertEqual(set(split_data.get_column("y_flag").unique()), {0, 1})
        self.assertEqual(result.summary.get_column("sample_count").sum(), cleaned.data.height)


if __name__ == "__main__":
    unittest.main()

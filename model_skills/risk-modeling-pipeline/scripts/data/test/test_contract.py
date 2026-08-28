"""Integration test for the confirmed schema and test dataset."""

from __future__ import annotations

from pathlib import Path
import unittest

from ..contract import load_contract, validate_contract
from ..loader import load_table


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_PATH = PROJECT_ROOT / "data.csv"
CONTRACT_PATH = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"


class DataContractIntegrationTest(unittest.TestCase):
    def test_validate_sample_contract(self) -> None:
        contract = load_contract(CONTRACT_PATH)
        result = validate_contract(load_table(TEST_DATA_PATH).data, contract)

        self.assertEqual(contract.id_cols, ("map_key",))
        self.assertEqual(contract.date_col, "clean_date")
        self.assertEqual(contract.target_col, "y_flag")
        self.assertEqual(len(result.feature_cols), 19)
        self.assertEqual(result.feature_cols[0], "flag_1")
        self.assertEqual(result.feature_cols[-1], "flag_19")
        self.assertEqual(result.id_unique_rate, 1.0)
        self.assertEqual(result.date_parse_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

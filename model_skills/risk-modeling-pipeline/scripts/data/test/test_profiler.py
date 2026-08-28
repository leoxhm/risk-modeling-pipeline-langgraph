"""Integration tests for deterministic column profiling."""

from __future__ import annotations

from pathlib import Path
import unittest

from ..loader import load_table
from ..profiler import profile_columns


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_PATH = PROJECT_ROOT / "data.csv"


class ProfileColumnsIntegrationTest(unittest.TestCase):
    def test_profile_sample_csv(self) -> None:
        profile = profile_columns(load_table(TEST_DATA_PATH).data)

        display_columns = [
            "column_name",
            "data_type",
            "missing_rate",
            "unique_rate",
            "date_parse_rate",
            "candidate_role",
            "role_reason",
        ]
        print("\n字段画像结果：")
        print(profile.select(display_columns))

        self.assertEqual(profile.height, 22)
        self.assertIn("candidate_role", profile.columns)
        self.assertIn("role_reason", profile.columns)

        records = {row["column_name"]: row for row in profile.to_dicts()}
        self.assertEqual(records["map_key"]["candidate_role"], "candidate_id")
        self.assertEqual(records["clean_date"]["candidate_role"], "candidate_date")
        self.assertEqual(records["y_flag"]["candidate_role"], "candidate_target")
        self.assertEqual(records["flag_1"]["candidate_role"], "candidate_feature")
        self.assertEqual(records["clean_date"]["date_parse_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

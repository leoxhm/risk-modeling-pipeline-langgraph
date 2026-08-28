import tempfile
import unittest
from pathlib import Path

import polars as pl

from data.contract import DataContract, apply_role_overrides
from workflow.node_config import ensure_node_configs, populate_data_read_roles


class NodeConfigTest(unittest.TestCase):
    def test_template_is_populated_during_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data-read.yaml"
            path.write_text(
                """node_id: data-read
version: 1
parameters:
  id_col_nm: null
  dt_col_nm: null
  label_col_nm: null
  encoding: null
  sheet_name: 0
  infer_schema_length: 10000
  allow_id_duplicates: false
  allow_target_issues: true
""",
                encoding="utf-8",
            )
            config = ensure_node_configs(directory, ["data-read"])["data-read"]
            fallback = DataContract(
                input_path=None,
                id_cols=("customer_id",),
                date_col="observation_date",
                target_col="target",
                good_label=0,
                bad_label=1,
                exclude_cols=(),
            )
            data = pl.DataFrame(
                {
                    "customer_id": [1, 2],
                    "observation_date": [20240101, 20240102],
                    "target": [0, 1],
                    "feature": [0.1, 0.2],
                }
            )
            self.assertTrue(populate_data_read_roles(config, data, fallback))
            refreshed = ensure_node_configs(directory, ["data-read"])["data-read"]
            self.assertEqual(refreshed.parameters["id_col_nm"], "customer_id")
            self.assertEqual(refreshed.parameters["dt_col_nm"], "observation_date")
            self.assertEqual(refreshed.parameters["label_col_nm"], "target")

    def test_data_read_role_parameters_are_loaded_and_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data-read.yaml"
            path.write_text(
                """node_id: data-read
version: 1
parameters:
  id_col_nm: customer_id
  dt_col_nm: observation_dt
  label_col_nm: target
  encoding: null
  sheet_name: 0
  infer_schema_length: 10000
  allow_id_duplicates: false
  allow_target_issues: true
""",
                encoding="utf-8",
            )
            configs = ensure_node_configs(directory, ["data-read"])
            contract = DataContract(
                input_path=None,
                id_cols=("old_id",),
                date_col="old_date",
                target_col="old_target",
                good_label=0,
                bad_label=1,
                exclude_cols=(),
            )
            effective = apply_role_overrides(contract, configs["data-read"].parameters)
            self.assertEqual(effective.id_cols, ("customer_id",))
            self.assertEqual(effective.date_col, "observation_dt")
            self.assertEqual(effective.target_col, "target")


if __name__ == "__main__":
    unittest.main()

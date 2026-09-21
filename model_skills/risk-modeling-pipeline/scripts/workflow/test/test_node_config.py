import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from data.contract import DataContract, apply_role_overrides
from workflow.node_config import ensure_node_configs, populate_data_read_roles
from workflow.nodes.common import approval_path


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

    def test_template_copy_and_role_population_preserve_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data-read.yaml"
            path.write_text(
                """# keep this explanation\nnode_id: data-read\nversion: 1\nparameters:\n  # ID explanation\n  id_col_nm: null\n  # date explanation\n  dt_col_nm: null\n  # target explanation\n  label_col_nm: null\n  encoding: null\n  sheet_name: 0\n  infer_schema_length: 10000\n  allow_id_duplicates: false\n  allow_target_issues: true\n""",
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
            output = path.read_text(encoding="utf-8")
            self.assertIn("# keep this explanation", output)
            self.assertIn("# ID explanation", output)
            self.assertIn("# date explanation", output)
            self.assertIn("id_col_nm: \"customer_id\"", output)

    def test_legacy_commentless_config_is_migrated_when_template_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "data-read.yaml").write_text(
                """node_id: data-read
version: 1
parameters:
  id_col_nm: map_key
  dt_col_nm: clean_date
  label_col_nm: y_flag
  encoding: null
  sheet_name: 0
  infer_schema_length: 10000
  allow_id_duplicates: false
  allow_target_issues: true
""",
                encoding="utf-8",
            )
            template_dir = root / "assets"
            template_dir.mkdir()
            (template_dir / "data_read.template.yaml").write_text(
                """# template comment
node_id: data-read
version: 1
parameters:
  # ID comment
  id_col_nm: null
  # date comment
  dt_col_nm: null
  # target comment
  label_col_nm: null
  encoding: null
  sheet_name: 0
  infer_schema_length: 10000
  allow_id_duplicates: false
  allow_target_issues: true
""",
                encoding="utf-8",
            )
            config = ensure_node_configs(config_dir, ["data-read"], template_dir=template_dir)["data-read"]
            output = config.path.read_text(encoding="utf-8")
            self.assertIn("# template comment", output)
            self.assertIn("# ID comment", output)
            self.assertIn("id_col_nm: \"map_key\"", output)

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

    def test_approval_path_prefers_dedicated_directory_and_reads_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = SimpleNamespace(
                approval_dir=root / "configs" / "approvals",
                node_config_dir=root / "configs" / "node_configs",
            )
            current = context.approval_dir / "data-read.approval.json"
            legacy = context.node_config_dir / "data-read.approval.json"
            context.approval_dir.mkdir(parents=True)
            context.node_config_dir.mkdir(parents=True)

            self.assertEqual(approval_path(context, "data-read", for_write=True), current)
            legacy.write_text("{}", encoding="utf-8")
            self.assertEqual(approval_path(context, "data-read"), legacy)
            current.write_text("{}", encoding="utf-8")
            self.assertEqual(approval_path(context, "data-read"), current)


if __name__ == "__main__":
    unittest.main()

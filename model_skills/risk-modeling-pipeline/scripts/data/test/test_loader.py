"""Integration test for the local example CSV and the Polars loader."""

from __future__ import annotations

from pathlib import Path
import unittest

import polars as pl

from ..loader import DataLoadResult, DataLoaderError, discover_data_file, load_table


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_PATH = PROJECT_ROOT / "data.csv"


class LoadTableIntegrationTest(unittest.TestCase):
    """Verify that the loader reads the repository's sample CSV unchanged."""

    def test_load_sample_csv(self) -> None:
        result = load_table(TEST_DATA_PATH)

        self.assertIsInstance(result, DataLoadResult)
        self.assertIsInstance(result.data, pl.DataFrame)
        self.assertEqual(result.source_format, "csv")
        self.assertEqual(result.path, TEST_DATA_PATH.resolve())
        self.assertEqual(result.row_count, 5_000)
        self.assertEqual(result.column_count, 22)
        self.assertEqual(result.data.columns[:3], ["map_key", "clean_date", "y_flag"])
        self.assertEqual(result.data.schema["map_key"], pl.Int64)
        self.assertEqual(result.data.schema["y_flag"], pl.Int64)

    def test_discover_unique_root_level_data_file(self) -> None:
        with self.subTest("repository root has one supported source for this test"):
            self.assertEqual(discover_data_file(PROJECT_ROOT), TEST_DATA_PATH.resolve())

    def test_discovery_requires_explicit_path_when_ambiguous(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.csv").write_text("x\n1\n", encoding="utf-8")
            (root / "b.parquet").write_bytes(b"not a parquet file")
            with self.assertRaisesRegex(DataLoaderError, "Multiple data files"):
                discover_data_file(root)


if __name__ == "__main__":
    unittest.main()

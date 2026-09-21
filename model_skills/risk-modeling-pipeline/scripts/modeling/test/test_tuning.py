"""Small-budget integration test for Train-only Optuna tuning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import polars as pl

from data.contract import load_contract, validate_contract
from data.loader import load_table
from modeling.config import load_model_config
from modeling.feature_selection import select_features
from modeling.split import split_dataset
from modeling.tuning import _rolling_fold_ids, tune_lightgbm
from preprocessing.cleaning import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TuningIntegrationTest(unittest.TestCase):
    def test_rolling_folds_are_expanding_and_chronological(self) -> None:
        data = pl.DataFrame(
            {
                "__tuning_row_id": list(range(12)),
                "event_month": [
                    "202501", "202501", "202502", "202502", "202503", "202503",
                    "202504", "202504", "202505", "202505", "202506", "202506",
                ],
            }
        )
        folds = _rolling_fold_ids(
            data,
            month_col="event_month",
            folds=4,
            validation_months=1,
            gap_months=0,
            min_train_months=2,
        )
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0][2], ("202501", "202502"))
        self.assertEqual(folds[0][3], ("202503",))
        self.assertEqual(folds[-1][2], ("202501", "202502", "202503", "202504", "202505"))
        self.assertEqual(folds[-1][3], ("202506",))
        self.assertTrue(set(folds[0][0]).isdisjoint(folds[0][1]))

    def test_two_trial_tuning_uses_train_only_folds(self) -> None:
        data = load_table(PROJECT_ROOT / "data.csv").data
        contract = validate_contract(data, load_contract(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"))
        config = load_model_config(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml")
        cleaning = preprocess_data(data, contract)
        cleaned_contract = replace(contract, feature_cols=cleaning.feature_cols)
        split = split_dataset(cleaning.data, cleaned_contract, config.split)
        selection = select_features(split.train, cleaned_contract, config.feature_selection)
        small_budget = replace(
            config.tuning,
            n_trials=2,
            timeout_seconds=120,
            startup_trials=1,
            min_bad_samples_per_fold=10,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = tune_lightgbm(
                split.train,
                cleaned_contract,
                selection.feature_cols,
                config.model,
                small_budget,
                temporary_directory,
            )
            self.assertTrue(result.study_path.is_file())
            self.assertTrue((Path(temporary_directory) / "tuning_trials.csv").is_file())
            self.assertTrue((Path(temporary_directory) / "best_params.yaml").is_file())
            # Retrying the same output directory must create a fresh Optuna
            # study instead of failing on the previous study name.
            retry = tune_lightgbm(
                split.train,
                cleaned_contract,
                selection.feature_cols,
                config.model,
                small_budget,
                temporary_directory,
            )
            self.assertTrue(retry.study_path.samefile(result.study_path))
        self.assertEqual(result.trials.height, 2)
        self.assertEqual(result.fold_metrics.height, 2 * small_budget.cv_folds)
        self.assertIn("mean_best_iteration", result.trials.columns)
        self.assertIn("min_best_iteration", result.trials.columns)
        self.assertIn("max_best_iteration", result.trials.columns)
        self.assertIn("cv_strategy", result.trials.columns)
        self.assertTrue((result.trials["cv_strategy"] == "rolling").all())
        self.assertIn("validation_month_start", result.fold_metrics.columns)
        self.assertTrue((result.trials["mean_best_iteration"] > 0).all())
        self.assertGreaterEqual(result.best_config.num_leaves, small_budget.search_space.num_leaves[0])


if __name__ == "__main__":
    unittest.main()

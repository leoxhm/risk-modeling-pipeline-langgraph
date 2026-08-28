"""Small-budget integration test for Train-only Optuna tuning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from data.contract import load_contract, validate_contract
from data.loader import load_table
from modeling.config import load_model_config
from modeling.feature_selection import select_features
from modeling.split import split_dataset
from modeling.tuning import tune_lightgbm
from preprocessing.cleaning import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TuningIntegrationTest(unittest.TestCase):
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
        self.assertEqual(result.trials.height, 2)
        self.assertEqual(result.fold_metrics.height, 2 * small_budget.cv_folds)
        self.assertGreaterEqual(result.best_config.num_leaves, small_budget.search_space.num_leaves[0])


if __name__ == "__main__":
    unittest.main()

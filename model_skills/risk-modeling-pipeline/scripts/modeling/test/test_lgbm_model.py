"""Checks for leakage-safe class weighting in LightGBM training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import polars as pl

from data.contract import (
    DataContract,
    ValidatedDataContract,
    load_contract,
    validate_contract,
)
from data.loader import load_table
from modeling.config import load_model_config
from modeling.lgbm_model import train_lightgbm
from modeling.split import split_dataset
from preprocessing.cleaning import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class LightGbmClassWeightTest(unittest.TestCase):
    def test_class_weight_uses_train_distribution_only(self) -> None:
        data = load_table(PROJECT_ROOT / "data.csv").data
        contract = validate_contract(
            data, load_contract(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml")
        )
        cleaned = preprocess_data(data, contract)
        config = load_model_config(PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml")
        split = split_dataset(cleaned.data, contract, config.split)
        short_config = replace(
            config.model, num_boost_round=10, early_stopping_rounds=3
        )

        trained = train_lightgbm(
            split.train,
            split.test,
            contract,
            cleaned.feature_cols,
            short_config,
            seed=config.split.random_seed,
            balance_classes=True,
        )

        target = split.train.get_column(contract.contract.target_col)
        bad_count = int((target == contract.contract.bad_label).sum())
        good_count = int((target == contract.contract.good_label).sum())
        self.assertAlmostEqual(
            float(trained.booster.params["scale_pos_weight"]), good_count / bad_count
        )

    def test_native_categorical_feature_can_be_trained(self) -> None:
        config = load_model_config(
            PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml"
        )
        short_config = replace(
            config.model,
            num_boost_round=10,
            early_stopping_rounds=3,
            min_data_in_leaf=2,
        )
        raw_contract = DataContract(
            input_path=None,
            id_cols=("id",),
            date_col="date",
            target_col="target",
            good_label=0,
            bad_label=1,
            exclude_cols=(),
        )
        contract = ValidatedDataContract(
            contract=raw_contract,
            feature_cols=("amount", "channel"),
            id_unique_rate=1.0,
            date_parse_rate=1.0,
        )
        train = pl.DataFrame(
            {
                "id": list(range(80)),
                "date": [20250101] * 80,
                "target": [index % 2 for index in range(80)],
                "amount": [float(index % 7) for index in range(80)],
                "channel": [index % 3 for index in range(80)],
            }
        ).with_columns(pl.col("channel").cast(pl.Int32))
        test = train.tail(20)

        trained = train_lightgbm(
            train.head(60),
            test,
            contract,
            contract.feature_cols,
            short_config,
            seed=42,
            categorical_features=("channel",),
        )

        self.assertGreater(trained.best_iteration, 0)


if __name__ == "__main__":
    unittest.main()

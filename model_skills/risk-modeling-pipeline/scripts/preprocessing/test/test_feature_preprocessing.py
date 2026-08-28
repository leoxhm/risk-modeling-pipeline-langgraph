"""Tests for Train-fitted feature typing and categorical transformations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import polars as pl

from data.contract import DataContract, ValidatedDataContract
from modeling.config import load_model_config
from preprocessing.feature_preprocessing import (
    fit_feature_preprocessor,
    transform_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class FeaturePreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        raw_contract = DataContract(
            input_path=None,
            id_cols=("id",),
            date_col="date",
            target_col="target",
            good_label=0,
            bad_label=1,
            exclude_cols=(),
        )
        self.contract = ValidatedDataContract(
            contract=raw_contract,
            feature_cols=("amount", "channel", "near_id", "notes"),
            id_unique_rate=1.0,
            date_parse_rate=1.0,
        )
        self.config = load_model_config(
            PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml"
        ).feature_preprocessing

    def test_plan_is_train_fitted_and_handles_unseen_categories(self) -> None:
        train = pl.DataFrame(
            {
                "id": list(range(60)),
                "date": [20250101] * 60,
                "target": [index % 2 for index in range(60)],
                "amount": [float(index) for index in range(59)] + [float("inf")],
                "channel": ["app"] * 30 + ["branch"] * 30,
                "near_id": [f"code-{index}" for index in range(60)],
                "notes": [("long text " * 10) + str(index) for index in range(60)],
                "event_month": ["202501"] * 60,
            }
        )
        test = pl.DataFrame(
            {
                "id": [100, 101],
                "date": [20250201, 20250201],
                "target": [0, 1],
                "amount": [1.0, 2.0],
                "channel": ["app", "new-channel"],
                "near_id": ["new-100", "new-101"],
                "notes": ["new note", "new note"],
                "event_month": ["202502", "202502"],
            }
        )

        plan = fit_feature_preprocessor(train, self.contract, self.config)
        transformed_train = transform_features(train, plan)
        transformed_test = transform_features(test, plan)

        self.assertEqual(plan.categorical_features, ("channel",))
        self.assertEqual(set(plan.dropped_features), {"near_id", "notes"})
        self.assertEqual(transformed_train.get_column("channel").dtype, pl.Int32)
        self.assertEqual(transformed_train.get_column("amount").null_count(), 1)
        self.assertEqual(transformed_test.get_column("channel").null_count(), 1)
        self.assertNotIn("new-channel", plan.categorical_encodings["channel"].value_to_code)

    def test_drop_mode_is_available_as_conservative_fallback(self) -> None:
        train = pl.DataFrame(
            {
                "id": list(range(40)),
                "date": [20250101] * 40,
                "target": [index % 2 for index in range(40)],
                "amount": list(range(40)),
                "channel": ["app"] * 20 + ["branch"] * 20,
                "near_id": [f"code-{index}" for index in range(40)],
                "notes": ["short"] * 40,
                "event_month": ["202501"] * 40,
            }
        )
        categorical = replace(self.config.categorical, strategy="drop")
        config = replace(self.config, categorical=categorical)

        plan = fit_feature_preprocessor(train, self.contract, config)

        self.assertNotIn("channel", plan.retained_features)
        decision = plan.decisions.filter(pl.col("feature") == "channel").row(
            0, named=True
        )
        self.assertEqual(decision["action"], "drop_categorical")


if __name__ == "__main__":
    unittest.main()

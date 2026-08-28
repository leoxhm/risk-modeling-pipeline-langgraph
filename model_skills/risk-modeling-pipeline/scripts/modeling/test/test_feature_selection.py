"""Tests for layered Train-only feature selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import polars as pl

from data.contract import DataContract, ValidatedDataContract
from modeling.config import load_model_config
from modeling.feature_selection import select_features


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class FeatureSelectionTest(unittest.TestCase):
    def test_stability_and_iv_ordered_correlation_rules(self) -> None:
        months = ["202501", "202502", "202503", "202504"]
        rows: list[dict[str, object]] = []
        for month_index, month in enumerate(months):
            for index in range(60):
                target = index % 2
                strong = target * 2.0 + (index % 7) * 0.01
                weak = strong + (index % 5) * 0.08
                rows.append(
                    {
                        "id": month_index * 60 + index,
                        "date": int(month + "01"),
                        "target": target,
                        "weak": weak,
                        "strong": strong,
                        "unstable": month_index * 100.0 + target,
                        "constant": 1.0,
                        "channel": "bad" if target else "good",
                        "event_month": month,
                    }
                )
        data = pl.DataFrame(rows)
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
            feature_cols=("weak", "strong", "unstable", "constant", "channel"),
            id_unique_rate=1.0,
            date_parse_rate=1.0,
        )
        base = load_model_config(
            PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml"
        ).feature_selection
        config = replace(
            base,
            min_iv=0.0,
            max_correlation=0.80,
            max_psi=0.01,
            max_unstable_month_ratio=0.20,
            min_month_samples=1,
            stability_action="drop",
        )

        result = select_features(data, contract, config)
        decisions = {
            row["feature"]: row for row in result.decisions.to_dicts()
        }
        correlated_pair = [decisions["weak"], decisions["strong"]]
        retained = next(
            row for row in correlated_pair if row["decision"].startswith("retained")
        )
        excluded = next(
            row for row in correlated_pair if row["decision"] == "excluded_correlation"
        )

        self.assertGreaterEqual(retained["iv"], excluded["iv"])
        self.assertEqual(decisions["constant"]["decision"], "excluded_constant")
        self.assertEqual(decisions["unstable"]["decision"], "excluded_unstable")
        self.assertIn("channel", result.feature_cols)
        self.assertEqual(decisions["weak"]["correlation_method"], "spearman")


if __name__ == "__main__":
    unittest.main()

"""Leakage-safe train, test, and out-of-time data splitting."""

from __future__ import annotations

from dataclasses import dataclass
import random

import polars as pl

from data.contract import DataContractError, ValidatedDataContract
from logger import get_logger

from .config import SplitConfig


logger = get_logger(__name__)


@dataclass(frozen=True)
class DataSplitResult:
    """Disjoint Train/Test/OOT samples and an auditable split summary."""

    train: pl.DataFrame
    test: pl.DataFrame
    oot: pl.DataFrame
    test_month_values: tuple[str, ...]
    oot_month_values: tuple[str, ...]
    summary: pl.DataFrame


def _stratified_test_row_ids(data: pl.DataFrame, target_col: str, config: SplitConfig) -> set[int]:
    randomizer = random.Random(config.random_seed)
    test_row_ids: set[int] = set()
    for target_value in data.get_column(target_col).unique().to_list():
        row_ids = data.filter(pl.col(target_col) == target_value).get_column("__split_row_id").to_list()
        randomizer.shuffle(row_ids)
        requested_count = round(len(row_ids) * config.test_ratio)
        if len(row_ids) >= 2:
            requested_count = max(1, min(requested_count, len(row_ids) - 1))
        else:
            requested_count = 0
        test_row_ids.update(row_ids[:requested_count])
    return test_row_ids


def _split_summary(
    name: str,
    data: pl.DataFrame,
    target_col: str,
) -> dict[str, object]:
    sample_count = data.height
    bad_count = int(data.get_column(target_col).sum()) if sample_count else 0
    return {
        "dataset": name,
        "sample_count": sample_count,
        "bad_count": bad_count,
        "bad_rate": bad_count / sample_count if sample_count else None,
        "month_start": data.get_column("event_month").min() if sample_count else None,
        "month_end": data.get_column("event_month").max() if sample_count else None,
    }


def split_dataset(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    config: SplitConfig,
) -> DataSplitResult:
    """Reserve latest months for OOT and preceding months for Test.

    event_month must originate from preprocessing. The newest oot_months
    observed values are never used in feature selection or parameter tuning.
    The default time strategy makes Test a contiguous block immediately before
    OOT. The random_stratified strategy remains available for legacy runs.
    """
    required_columns = {"event_month", contract.contract.target_col}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise DataContractError(
            "Model splitting requires preprocessed data with columns: "
            f"{sorted(required_columns)}; missing={sorted(missing_columns)}"
        )
    month_values = sorted(
        str(value) for value in data.get_column("event_month").drop_nulls().unique().to_list()
    )
    if len(month_values) <= config.oot_months:
        raise DataContractError(
            f"Need more than {config.oot_months} distinct months to create development and OOT samples; "
            f"found {len(month_values)}"
        )
    oot_month_values = tuple(month_values[-config.oot_months :])
    indexed = data.with_row_index("__split_row_id")
    is_oot = pl.col("event_month").cast(pl.String).is_in(oot_month_values)
    oot = indexed.filter(is_oot).drop("__split_row_id")
    development = indexed.filter(~is_oot)
    if oot.is_empty() or development.is_empty():
        raise DataContractError("OOT or development sample is empty after chronological split")

    target_col = contract.contract.target_col
    if config.strategy == "time":
        development_months = month_values[: len(month_values) - config.oot_months]
        if len(development_months) <= config.test_months:
            raise DataContractError(
                "Not enough development months for chronological Train/Test split: "
                f"found={len(development_months)}, test_months={config.test_months}"
            )
        test_month_values = tuple(development_months[-config.test_months :])
        is_test = pl.col("event_month").cast(pl.String).is_in(test_month_values)
        test = development.filter(is_test).drop("__split_row_id")
        train = development.filter(~is_test).drop("__split_row_id")
    else:
        test_month_values = ()
        test_row_ids = _stratified_test_row_ids(development, target_col, config)
        test = development.filter(pl.col("__split_row_id").is_in(test_row_ids)).drop("__split_row_id")
        train = development.filter(~pl.col("__split_row_id").is_in(test_row_ids)).drop("__split_row_id")
    if train.is_empty() or test.is_empty():
        raise DataContractError("Train or test sample is empty after development split")

    summary = pl.DataFrame(
        [
            _split_summary("train", train, target_col),
            _split_summary("test", test, target_col),
            _split_summary("oot", oot, target_col),
        ]
    )
    logger.info(
        "Dataset split completed: train=%d, test=%d, oot=%d, oot_months=%s",
        train.height,
        test.height,
        oot.height,
        list(oot_month_values),
    )
    return DataSplitResult(train, test, oot, test_month_values, oot_month_values, summary)

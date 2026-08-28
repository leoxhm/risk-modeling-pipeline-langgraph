"""Deterministic preprocessing for data with a validated field contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from data.contract import DataContractError, ValidatedDataContract
from logger import get_logger


logger = get_logger(__name__)
_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m", "%Y-%m")
DuplicateStrategy = Literal["keep_first", "keep_last", "error"]


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned dataset, retained feature list, and auditable processing decisions."""

    data: pl.DataFrame
    feature_cols: tuple[str, ...]
    column_decisions: pl.DataFrame
    removed_duplicate_count: int
    invalid_date_count: int


def _parsed_date_expression(date_col: str) -> pl.Expr:
    normalized = pl.col(date_col).cast(pl.String).str.strip_chars()
    return pl.coalesce(
        [normalized.str.strptime(pl.Date, format=date_format, strict=False) for date_format in _DATE_FORMATS]
    )


def _top_value_rate(series: pl.Series) -> float:
    values = series.drop_nulls().to_list()
    if not values:
        return 1.0
    return Counter(str(value) for value in values).most_common(1)[0][1] / len(values)


def _deduplicate(
    data: pl.DataFrame,
    id_cols: tuple[str, ...],
    strategy: DuplicateStrategy,
) -> tuple[pl.DataFrame, int]:
    unique_count = data.select(list(id_cols)).unique().height
    duplicate_count = data.height - unique_count
    if duplicate_count == 0:
        return data, 0
    if strategy == "error":
        raise DataContractError(
            f"Detected {duplicate_count} duplicate rows for configured ID columns: {list(id_cols)}"
        )
    keep = "first" if strategy == "keep_first" else "last"
    deduplicated = data.unique(subset=list(id_cols), keep=keep, maintain_order=True)
    logger.warning("Removed %d duplicate rows using strategy=%s", duplicate_count, strategy)
    return deduplicated, duplicate_count


def preprocess_data(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    *,
    missing_rate_threshold: float = 0.8,
    constant_rate_threshold: float = 0.8,
    duplicate_strategy: DuplicateStrategy = "keep_first",
    filter_features: bool = True,
) -> CleaningResult:
    """Normalize date fields, remove duplicate IDs, and filter low-quality features.

    Preserve all role columns and add ``event_date`` and ``event_month``. Drop only
    manually excluded or automatically excluded candidate feature columns.
    """
    if not 0 <= missing_rate_threshold <= 1:
        raise ValueError("missing_rate_threshold must be between 0 and 1")
    if not 0 <= constant_rate_threshold <= 1:
        raise ValueError("constant_rate_threshold must be between 0 and 1")
    if {"event_date", "event_month"} & set(data.columns):
        raise DataContractError("Input already contains reserved columns: event_date or event_month")

    raw_contract = contract.contract
    logger.info("Preprocessing dataset: rows=%d, candidate_features=%d", data.height, len(contract.feature_cols))
    date_expr = _parsed_date_expression(raw_contract.date_col)
    invalid_date_count = data.select(
        (pl.col(raw_contract.date_col).is_not_null() & date_expr.is_null()).sum()
    ).item()
    if invalid_date_count:
        raise DataContractError(
            f"Cannot preprocess: {invalid_date_count} non-null values in '{raw_contract.date_col}' cannot be parsed."
        )

    dated_data = data.with_columns(
        date_expr.alias("event_date"),
        date_expr.dt.strftime("%Y%m").alias("event_month"),
    )
    deduplicated_data, removed_duplicate_count = _deduplicate(
        dated_data, raw_contract.id_cols, duplicate_strategy
    )

    excluded_columns: set[str] = set(raw_contract.exclude_cols)
    decisions: list[dict[str, Any]] = [
        {
            "column_name": column,
            "decision": "excluded_manual",
            "reason": "Listed in schema.exclude_cols",
            "missing_rate": None,
            "constant_rate": None,
        }
        for column in raw_contract.exclude_cols
    ]
    retained_features: list[str] = []
    for column in contract.feature_cols:
        series = deduplicated_data.get_column(column)
        missing_rate = series.null_count() / deduplicated_data.height if deduplicated_data.height else 0.0
        constant_rate = _top_value_rate(series)
        if not filter_features:
            decision, reason = "deferred_train_fit", "Feature decision deferred until the Train split"
            retained_features.append(column)
        elif missing_rate > missing_rate_threshold:
            decision, reason = "excluded_missing", f"Missing rate exceeds {missing_rate_threshold:.0%}"
            excluded_columns.add(column)
        elif constant_rate > constant_rate_threshold:
            decision, reason = "excluded_constant", f"Constant rate exceeds {constant_rate_threshold:.0%}"
            excluded_columns.add(column)
        else:
            decision, reason = "retained", "Passed missing-rate and constant-rate checks"
            retained_features.append(column)
        decisions.append(
            {
                "column_name": column,
                "decision": decision,
                "reason": reason,
                "missing_rate": missing_rate,
                "constant_rate": constant_rate,
            }
        )

    cleaned_data = deduplicated_data.drop(sorted(excluded_columns))
    result = CleaningResult(
        data=cleaned_data,
        feature_cols=tuple(retained_features),
        column_decisions=pl.DataFrame(decisions),
        removed_duplicate_count=removed_duplicate_count,
        invalid_date_count=invalid_date_count,
    )
    logger.info(
        "Preprocessing completed: rows=%d, retained_features=%d, excluded_features=%d",
        result.data.height,
        len(result.feature_cols),
        len(excluded_columns),
    )
    return result

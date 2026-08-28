"""Dataset-level exploration built on the data profiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from data.contract import ValidatedDataContract
from data.profiler import profile_columns
from logger import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class ExplorationResult:
    """Structured exploration outputs for one validated input dataset."""

    summary: dict[str, Any]
    column_profile: pl.DataFrame
    target_distribution: pl.DataFrame


def explore_dataset(data: pl.DataFrame, contract: ValidatedDataContract) -> ExplorationResult:
    """Create reusable raw-data exploration outputs without changing the data."""
    target_col = contract.contract.target_col
    logger.info("Exploring validated dataset: rows=%d, columns=%d", data.height, data.width)

    target_distribution = (
        data.group_by(target_col)
        .len()
        .rename({"len": "sample_count"})
        .with_columns((pl.col("sample_count") / data.height).alias("sample_rate"))
        .sort(target_col)
    )
    bad_count = (
        data.filter(pl.col(target_col) == contract.contract.bad_label).height
    )
    summary = {
        "row_count": data.height,
        "column_count": data.width,
        "candidate_feature_count": len(contract.feature_cols),
        "id_unique_rate": contract.id_unique_rate,
        "date_parse_rate": contract.date_parse_rate,
        "bad_sample_count": bad_count,
        "bad_sample_rate": bad_count / data.height if data.height else 0.0,
    }
    result = ExplorationResult(
        summary=summary,
        column_profile=profile_columns(data),
        target_distribution=target_distribution,
    )
    logger.info("Exploration completed: bad_sample_rate=%.2f%%", summary["bad_sample_rate"] * 100)
    return result

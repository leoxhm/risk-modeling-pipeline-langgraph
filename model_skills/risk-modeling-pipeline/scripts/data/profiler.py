"""Column profiling and deterministic field-role suggestions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
import re
from typing import Any

import polars as pl

from logger import get_logger


logger = get_logger(__name__)

_TARGET_NAMES = {
    "y",
    "y_flag",
    "label",
    "target",
    "bad_flag",
    "is_bad",
    "default_flag",
}
_ID_NAME_PATTERN = re.compile(r"(^|_)(id|key|no|number)(_|$)")
_DATE_NAME_PATTERN = re.compile(r"(date|time|month|yearmonth|ym|日期|时间|年月)")
_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m", "%Y-%m")


def _to_display(value: Any) -> str:
    """Produce a stable text representation for report columns."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return str(value)


def _date_parse_rate(values: list[Any]) -> float | None:
    """Return the share of non-null values matching supported date formats."""
    if not values:
        return None

    parsed_count = 0
    for value in values:
        text = _to_display(value).strip()
        if any(_try_parse_date(text, date_format) for date_format in _DATE_FORMATS):
            parsed_count += 1
    return parsed_count / len(values)


def _try_parse_date(value: str, date_format: str) -> bool:
    try:
        datetime.strptime(value, date_format)
        return True
    except ValueError:
        return False


def _suggest_role(
    column_name: str,
    non_null_count: int,
    unique_rate: float,
    date_parse_rate: float | None,
) -> tuple[str, str]:
    """Return a conservative role candidate and the deterministic reason."""
    normalized_name = column_name.lower().strip()
    if normalized_name in _TARGET_NAMES:
        return "candidate_target", "字段名符合预设标签命名规则"

    if _DATE_NAME_PATTERN.search(normalized_name) and (date_parse_rate or 0) >= 0.8:
        return "candidate_date", "字段名包含日期语义且日期可解析率不低于 80%"

    if _ID_NAME_PATTERN.search(normalized_name) and non_null_count > 0 and unique_rate >= 0.95:
        return "candidate_id", "字段名符合主键命名规则且非空唯一率不低于 95%"

    return "candidate_feature", "未命中标签、日期或主键候选规则"


def _numeric_summary(series: pl.Series) -> tuple[float | None, float | None, float | None]:
    if not series.dtype.is_numeric():
        return None, None, None
    return series.min(), series.max(), series.mean()


def profile_columns(data: pl.DataFrame, *, sample_size: int = 3) -> pl.DataFrame:
    """Profile every column and suggest a field role using deterministic rules.

    The function does not alter ``data``. Candidate roles are advisory only and
    must be confirmed in the data contract before any model training begins.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    logger.info("Profiling dataset: rows=%d, columns=%d", data.height, data.width)
    rows: list[dict[str, Any]] = []
    for column_name in data.columns:
        series = data.get_column(column_name)
        non_null = series.drop_nulls()
        non_null_count = len(non_null)
        missing_count = series.null_count()
        unique_count = non_null.n_unique()
        unique_rate = unique_count / non_null_count if non_null_count else 0.0
        values = non_null.to_list()
        date_rate = _date_parse_rate(values) if _DATE_NAME_PATTERN.search(column_name.lower()) else None
        role, role_reason = _suggest_role(
            column_name, non_null_count, unique_rate, date_rate
        )
        numeric_min, numeric_max, numeric_mean = _numeric_summary(non_null)

        counts = Counter(_to_display(value) for value in values)
        top_value, top_value_count = counts.most_common(1)[0] if counts else (None, 0)
        rows.append(
            {
                "column_name": column_name,
                "data_type": str(series.dtype),
                "row_count": data.height,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_rate": missing_count / data.height if data.height else 0.0,
                "unique_count": unique_count,
                "unique_rate": unique_rate,
                "sample_values": json.dumps(
                    [_to_display(value) for value in values[:sample_size]],
                    ensure_ascii=False,
                ),
                "top_value": top_value,
                "top_value_count": top_value_count,
                "top_value_rate": top_value_count / non_null_count if non_null_count else 0.0,
                "numeric_min": numeric_min,
                "numeric_max": numeric_max,
                "numeric_mean": numeric_mean,
                "date_parse_rate": date_rate,
                "candidate_role": role,
                "role_reason": role_reason,
            }
        )

    result = pl.DataFrame(rows)
    logger.info("Profiling completed: %d columns", result.height)
    return result

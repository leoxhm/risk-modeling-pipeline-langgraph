"""Train-fitted feature typing and deterministic LightGBM transformations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import polars as pl

from data.contract import ValidatedDataContract
from logger import get_logger

if TYPE_CHECKING:
    from modeling.config import FeaturePreprocessingConfig


logger = get_logger(__name__)


@dataclass(frozen=True)
class CategoricalEncoding:
    """Frozen Train-derived encoding for one categorical feature."""

    value_to_code: dict[str, int]
    other_code: int
    unknown_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_to_code": self.value_to_code,
            "other_code": self.other_code,
            "unknown_action": self.unknown_action,
        }


@dataclass(frozen=True)
class FeaturePreprocessingPlan:
    """Auditable transformations learned without inspecting Test or OOT."""

    retained_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    categorical_encodings: dict[str, CategoricalEncoding]
    decisions: pl.DataFrame
    invalid_to_null: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_scope": "train_only",
            "retained_features": list(self.retained_features),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "dropped_features": list(self.dropped_features),
            "invalid_to_null": self.invalid_to_null,
            "categorical_encodings": {
                feature: encoding.as_dict()
                for feature, encoding in self.categorical_encodings.items()
            },
        }


def _string_values(series: pl.Series) -> list[str]:
    return [str(value) for value in series.drop_nulls().to_list()]


def _average_string_length(values: list[str]) -> float:
    return sum(len(value) for value in values) / len(values) if values else 0.0


def _is_string_like(dtype: pl.DataType) -> bool:
    return dtype in {pl.String, pl.Categorical} or str(dtype).startswith("Enum")


def _categorical_encoding(
    values: list[str], config: FeaturePreprocessingConfig
) -> CategoricalEncoding:
    counts = Counter(values)
    minimum_count = max(
        config.categorical.rare_min_count,
        math.ceil(len(values) * config.categorical.rare_min_rate),
    )
    frequent = [
        value
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_count
    ][: max(1, config.categorical.max_categories - 2)]
    if not frequent and counts:
        frequent = [counts.most_common(1)[0][0]]
    frequent_codes = {value: index for index, value in enumerate(frequent)}
    other_code = len(frequent_codes)
    value_to_code = {
        value: frequent_codes.get(value, other_code) for value in counts
    }
    return CategoricalEncoding(
        value_to_code=value_to_code,
        other_code=other_code,
        unknown_action=config.categorical.unknown_action,
    )


def fit_feature_preprocessor(
    train_data: pl.DataFrame,
    contract: ValidatedDataContract,
    config: FeaturePreprocessingConfig,
) -> FeaturePreprocessingPlan:
    """Infer feature types and learn category mappings using Train only."""
    retained: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[str] = []
    encodings: dict[str, CategoricalEncoding] = {}
    decision_rows: list[dict[str, Any]] = []

    for feature in contract.feature_cols:
        series = train_data.get_column(feature)
        non_null_count = len(series.drop_nulls())
        unique_count = series.drop_nulls().n_unique()
        unique_rate = unique_count / non_null_count if non_null_count else 0.0
        missing_rate = series.null_count() / train_data.height if train_data.height else 0.0
        category_count: int | None = None
        encoded_category_count: int | None = None

        if series.dtype.is_numeric() or series.dtype == pl.Boolean:
            inferred_type = "boolean" if series.dtype == pl.Boolean else "numeric"
            action = "retain_numeric"
            reason = "Numeric/boolean feature; missing values use LightGBM native handling"
            retained.append(feature)
            numeric.append(feature)
        elif _is_string_like(series.dtype):
            values = _string_values(series)
            category_count = len(set(values))
            average_length = _average_string_length(values)
            is_text = (
                average_length >= config.text.min_average_length
                and unique_rate >= 0.50
            )
            if is_text:
                inferred_type = "text"
                action = "drop_text"
                reason = "Free-text-like feature is excluded by configured policy"
                dropped.append(feature)
            elif unique_rate >= config.categorical.near_unique_rate:
                inferred_type = "categorical"
                action = "drop_near_unique"
                reason = (
                    "Categorical unique rate exceeds "
                    f"{config.categorical.near_unique_rate:.0%}"
                )
                dropped.append(feature)
            elif config.categorical.strategy == "drop":
                inferred_type = "categorical"
                action = "drop_categorical"
                reason = "Categorical features are excluded by configured policy"
                dropped.append(feature)
            elif (
                category_count > config.categorical.max_categories
                and config.categorical.high_cardinality_action == "drop"
            ):
                inferred_type = "categorical"
                action = "drop_high_cardinality"
                reason = (
                    f"Category count exceeds {config.categorical.max_categories}"
                )
                dropped.append(feature)
            else:
                inferred_type = "categorical"
                action = "encode_lightgbm_native"
                reason = "Train-fitted integer encoding for LightGBM categorical splits"
                encoding = _categorical_encoding(values, config)
                encodings[feature] = encoding
                encoded_category_count = len(set(encoding.value_to_code.values()))
                retained.append(feature)
                categorical.append(feature)
        elif series.dtype.is_temporal():
            inferred_type = "temporal"
            action = "drop_temporal"
            reason = "Unconfigured temporal feature; derive approved observation-safe features first"
            dropped.append(feature)
        else:
            inferred_type = "unsupported"
            action = "drop_unsupported"
            reason = f"Unsupported modeling dtype: {series.dtype}"
            dropped.append(feature)

        decision_rows.append(
            {
                "feature": feature,
                "original_dtype": str(series.dtype),
                "inferred_type": inferred_type,
                "action": action,
                "reason": reason,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "unique_rate": unique_rate,
                "category_count": category_count,
                "encoded_category_count": encoded_category_count,
            }
        )

    if not retained:
        raise ValueError(
            "No trainable features remain after Train-only type preprocessing"
        )
    logger.info(
        "Feature preprocessor fitted on Train: retained=%d, categorical=%d, dropped=%d",
        len(retained),
        len(categorical),
        len(dropped),
    )
    return FeaturePreprocessingPlan(
        retained_features=tuple(retained),
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        dropped_features=tuple(dropped),
        categorical_encodings=encodings,
        decisions=pl.DataFrame(decision_rows),
        invalid_to_null=config.numeric.invalid_to_null,
    )


def transform_features(
    data: pl.DataFrame, plan: FeaturePreprocessingPlan
) -> pl.DataFrame:
    """Apply a frozen Train-derived plan without learning from the input split."""
    result = data.drop(
        [feature for feature in plan.dropped_features if feature in data.columns]
    )
    numeric_expressions: list[pl.Expr] = []
    for feature in plan.numeric_features:
        numeric_value = pl.col(feature).cast(pl.Float64, strict=False)
        if plan.invalid_to_null:
            numeric_value = pl.when(numeric_value.is_infinite()).then(None).otherwise(
                numeric_value
            )
        numeric_expressions.append(numeric_value.alias(feature))
    if numeric_expressions:
        result = result.with_columns(numeric_expressions)

    encoded_series: list[pl.Series] = []
    for feature in plan.categorical_features:
        encoding = plan.categorical_encodings[feature]
        values: list[int | None] = []
        for value in result.get_column(feature).to_list():
            if value is None:
                values.append(None)
                continue
            key = str(value)
            if key in encoding.value_to_code:
                values.append(encoding.value_to_code[key])
            elif encoding.unknown_action == "other":
                values.append(encoding.other_code)
            else:
                values.append(None)
        encoded_series.append(pl.Series(feature, values, dtype=pl.Int32))
    if encoded_series:
        result = result.with_columns(encoded_series)
    return result

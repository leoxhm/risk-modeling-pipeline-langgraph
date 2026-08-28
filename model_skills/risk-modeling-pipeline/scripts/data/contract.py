"""Load and validate the user-confirmed data contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from logger import get_logger


logger = get_logger(__name__)
_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m", "%Y-%m")


class DataContractError(ValueError):
    """Raised when contract configuration and input data are inconsistent."""


@dataclass(frozen=True)
class DataContract:
    """User-confirmed roles and label mapping for one data source."""

    input_path: str | None
    id_cols: tuple[str, ...]
    date_col: str
    target_col: str
    good_label: Any
    bad_label: Any
    exclude_cols: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedDataContract:
    """A validated contract enriched with the derived candidate feature list."""

    contract: DataContract
    feature_cols: tuple[str, ...]
    id_unique_rate: float
    date_parse_rate: float


def apply_role_overrides(
    contract: DataContract, parameters: dict[str, Any]
) -> DataContract:
    """Apply optional role names from the data-read node configuration.

    Null values fall back to ``data_contract.yaml``. Non-null values are
    explicit, user-confirmed overrides and continue through normal contract
    validation, treatment, EDA and modeling.
    """

    id_value = parameters.get("id_col_nm")
    if id_value is None:
        id_cols = contract.id_cols
    elif isinstance(id_value, str) and id_value.strip():
        id_cols = (id_value.strip(),)
    elif isinstance(id_value, (list, tuple)) and id_value and all(
        isinstance(item, str) and item.strip() for item in id_value
    ):
        id_cols = tuple(item.strip() for item in id_value)
    else:
        raise DataContractError(
            "data-read.parameters.id_col_nm must be a non-empty string, "
            "a list of non-empty strings, or null"
        )

    def _optional_name(key: str, fallback: str) -> str:
        value = parameters.get(key)
        if value is None:
            return fallback
        if not isinstance(value, str) or not value.strip():
            raise DataContractError(
                f"data-read.parameters.{key} must be a non-empty string or null"
            )
        return value.strip()

    return replace(
        contract,
        id_cols=id_cols,
        date_col=_optional_name("dt_col_nm", contract.date_col),
        target_col=_optional_name("label_col_nm", contract.target_col),
    )


def load_contract(path: str | Path) -> DataContract:
    """Load one YAML data contract without inspecting the underlying data."""
    contract_path = Path(path).expanduser().resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"Data contract does not exist: {contract_path}")

    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        data_config = raw.get("data", {})
        schema = raw["schema"]
        target = schema["target"]
        contract = DataContract(
            input_path=data_config.get("input_path"),
            id_cols=tuple(schema["id_cols"]),
            date_col=schema["date_col"],
            target_col=target["column"],
            good_label=target["good_label"],
            bad_label=target["bad_label"],
            exclude_cols=tuple(schema.get("exclude_cols", [])),
        )
    except (KeyError, TypeError) as exc:
        raise DataContractError(
            "Invalid data contract. Required fields: schema.id_cols, schema.date_col, "
            "schema.target.column, schema.target.good_label, schema.target.bad_label."
        ) from exc

    if not contract.id_cols:
        raise DataContractError("schema.id_cols must contain at least one column")
    if contract.good_label == contract.bad_label:
        raise DataContractError("good_label and bad_label must be different")
    logger.info("Loaded data contract: %s", contract_path)
    return contract


def _parse_date(value: Any) -> bool:
    text = str(value).strip()
    for date_format in _DATE_FORMATS:
        try:
            datetime.strptime(text, date_format)
            return True
        except ValueError:
            continue
    return False


def _ensure_existing_columns(data: pl.DataFrame, columns: set[str]) -> None:
    missing_columns = sorted(columns - set(data.columns))
    if missing_columns:
        raise DataContractError(
            f"Contract columns are absent from the data: {missing_columns}"
        )


def validate_contract(
    data: pl.DataFrame,
    contract: DataContract,
    *,
    allow_id_duplicates: bool = False,
    allow_target_issues: bool = False,
) -> ValidatedDataContract:
    """Validate role assignments and derive candidate features from a raw table."""
    logger.info(
        "Validating data contract against rows=%d, columns=%d", data.height, data.width
    )
    role_columns = set(contract.id_cols) | {contract.date_col, contract.target_col}
    _ensure_existing_columns(data, role_columns | set(contract.exclude_cols))

    if len(role_columns) != len(contract.id_cols) + 2:
        raise DataContractError("ID, date, and target columns must not overlap")
    if set(contract.exclude_cols) & role_columns:
        raise DataContractError(
            "exclude_cols must not contain ID, date, or target columns"
        )

    id_unique_count = data.select(list(contract.id_cols)).unique().height
    id_unique_rate = id_unique_count / data.height if data.height else 0.0
    if id_unique_rate < 0.95 and not allow_id_duplicates:
        raise DataContractError(
            f"Configured ID columns have a combined unique rate of {id_unique_rate:.2%}, below 95%."
        )

    date_values = data.get_column(contract.date_col).drop_nulls().to_list()
    if not date_values:
        raise DataContractError(
            f"Date column '{contract.date_col}' has no non-null values"
        )
    date_parse_rate = sum(_parse_date(value) for value in date_values) / len(
        date_values
    )
    if date_parse_rate < 0.8:
        raise DataContractError(
            f"Date column '{contract.date_col}' parse rate is {date_parse_rate:.2%}, below 80%."
        )

    target_values = set(
        data.get_column(contract.target_col).drop_nulls().unique().to_list()
    )
    expected_labels = {contract.good_label, contract.bad_label}
    if target_values != expected_labels and not allow_target_issues:
        raise DataContractError(
            f"Target column '{contract.target_col}' values are {sorted(target_values)!r}; "
            f"expected exactly {sorted(expected_labels)!r}."
        )
    if (
        data.get_column(contract.target_col).null_count() > 0
        and not allow_target_issues
    ):
        raise DataContractError(
            f"Target column '{contract.target_col}' contains null values"
        )

    feature_cols = tuple(
        column
        for column in data.columns
        if column not in role_columns and column not in contract.exclude_cols
    )
    if not feature_cols:
        raise DataContractError(
            "No candidate feature columns remain after applying the contract"
        )

    result = ValidatedDataContract(
        contract=contract,
        feature_cols=feature_cols,
        id_unique_rate=id_unique_rate,
        date_parse_rate=date_parse_rate,
    )
    logger.info(
        "Contract validated: features=%d, id_unique_rate=%.2f%%, date_parse_rate=%.2f%%",
        len(result.feature_cols),
        result.id_unique_rate * 100,
        result.date_parse_rate * 100,
    )
    return result

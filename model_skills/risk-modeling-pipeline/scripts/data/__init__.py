"""Data access primitives for the risk-modeling engine."""

from .contract import (
    DataContract,
    DataContractError,
    ValidatedDataContract,
    apply_role_overrides,
    load_contract,
    validate_contract,
)
from .loader import DataLoadResult, DataLoaderError, discover_data_file, load_table
from .profiler import profile_columns

__all__ = [
    "DataContract",
    "DataContractError",
    "DataLoadResult",
    "DataLoaderError",
    "ValidatedDataContract",
    "apply_role_overrides",
    "load_contract",
    "load_table",
    "discover_data_file",
    "profile_columns",
    "validate_contract",
]

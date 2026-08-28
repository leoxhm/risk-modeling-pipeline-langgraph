"""User-editable, hash-bound configuration for each workflow node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
import polars as pl

from logger import get_logger
from data.contract import DataContract
from data.profiler import profile_columns


logger = get_logger(__name__)


NODE_CONFIG_FILENAMES = {
    "data-read": "data-read.yaml",
    "sample-diagnosis": "sample-diagnosis.yaml",
    "eda-analysis": "eda-analysis.yaml",
    "feature-processing": "feature-processing.yaml",
    "model-config": "model-config.yaml",
    "training-tuning": "training-tuning.yaml",
    "model-review": "model-review.yaml",
    "report-delivery": "report-delivery.yaml",
}

NODE_CONFIG_TEMPLATE_FILENAMES = {
    "data-read": "data_read.template.yaml",
    "sample-diagnosis": "sample_diagnosis.template.yaml",
    "eda-analysis": "eda_analysis.template.yaml",
    "feature-processing": "feature_processing.template.yaml",
    "model-config": "model_config_node.template.yaml",
    "training-tuning": "training_tuning.template.yaml",
    "model-review": "model_review.template.yaml",
    "report-delivery": "report_delivery.template.yaml",
}

# Defaults are intentionally conservative. They are copied into the user's
# workspace only when a node config is missing, then included in the approval
# hash so the user can edit and confirm them before execution.
DEFAULT_NODE_CONFIGS: dict[str, dict[str, Any]] = {
    "data-read": {
        "version": 1,
        "parameters": {
            # Optional role overrides; null falls back to data_contract.yaml.
            "id_col_nm": None,
            "dt_col_nm": None,
            "label_col_nm": None,
            "encoding": None,
            "sheet_name": 0,
            "infer_schema_length": 10000,
            "allow_id_duplicates": False,
            "allow_target_issues": True,
        },
    },
    "sample-diagnosis": {
        "version": 1,
        "parameters": {
            "require_data_read_confirmation": True,
            "diagnostics": {
                "imbalance_warning_minority_rate": 0.10,
                "imbalance_critical_minority_rate": 0.01,
                "latest_month_min_volume_ratio": 0.50,
                "monthly_bad_rate_change_warning": 0.03,
                "high_missing_row_rate": 0.80,
            },
            "treatment": {
                "duplicate_action": "error",
                "missing_target_action": "error",
                "all_null_feature_action": "drop",
                "high_missing_row_action": "keep",
                "incomplete_latest_month_action": "error",
                "class_imbalance_action": "class_weight",
            },
        },
    },
    "eda-analysis": {
        "version": 1,
        "parameters": {
            "month_col": "event_month",
            "psi_base_month": None,
            "bin_count": 10,
            "ks_method": "quantile",
            "ks_bucket": 10,
            "binning_method": "quantile",
            "max_categories": 20,
            "correlation_method": "pearson",
            "metrics_backend": "toad",
            "missing_rate_threshold": 0.8,
            "constant_rate_threshold": 0.8,
            "duplicate_strategy": "keep_first",
            "generate_plots": True,
            "plot_top_n": 50,
            "generate_report": True,
        },
    },
    "feature-processing": {
        "version": 1,
        "parameters": {
            "fit_scope": "train_only",
            "selection": {
                "max_missing_rate": 0.80,
                "min_iv": 0.02,
                "max_correlation": 0.80,
                "max_psi": 0.25,
                "max_unstable_month_ratio": 0.30,
                "min_month_samples": 100,
                "stability_action": "review",
                "correlation_method": "spearman",
                "max_dominant_rate_warning": 0.95,
            },
            "preprocessing": {
                "numeric": {"invalid_to_null": True, "missing_strategy": "native"},
                "categorical": {
                    "strategy": "lightgbm_native",
                    "rare_min_count": 20,
                    "rare_min_rate": 0.001,
                    "unknown_action": "missing",
                    "near_unique_rate": 0.98,
                    "max_categories": 100,
                    "high_cardinality_action": "drop",
                },
                "text": {"action": "drop", "min_average_length": 64},
            },
            "output": {"write_processed_data": True},
        },
    },
    "model-config": {"version": 1, "parameters": {"config_path": "configs/model_config.yaml"}},
    "training-tuning": {
        "version": 1,
        "parameters": {"mode": "baseline", "max_trials": 20, "objective_metric": "auc"},
    },
    "model-review": {
        "version": 1,
        "parameters": {"auc_gap_warning": 0.05, "psi_warning": 0.25},
    },
    "report-delivery": {
        "version": 1,
        "parameters": {"write_xlsx": True, "write_model_files": True},
    },
}


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    version: int
    parameters: dict[str, Any]
    path: Path


def populate_data_read_roles(
    config: NodeConfig, data: pl.DataFrame, fallback_contract: DataContract
) -> bool:
    """Fill missing role fields in a data-read template during preparation.

    Only null values are filled. Contract roles are preferred when their
    columns exist; otherwise conservative profiler candidates are used. A host
    such as OpenCode may replace the suggestions before approval.
    """

    parameters = dict(config.parameters)
    columns = set(data.columns)
    profile = profile_columns(data)

    def _candidate(role: str) -> str | None:
        rows = profile.filter(pl.col("candidate_role") == role)
        return rows.get_column("column_name")[0] if rows.height else None

    changed = False
    if parameters.get("id_col_nm") is None:
        if fallback_contract.id_cols and set(fallback_contract.id_cols).issubset(columns):
            suggested_id: str | list[str] = (
                fallback_contract.id_cols[0]
                if len(fallback_contract.id_cols) == 1
                else list(fallback_contract.id_cols)
            )
        else:
            suggested_id = _candidate("candidate_id")
        if suggested_id is not None:
            parameters["id_col_nm"] = suggested_id
            changed = True

    for key, fallback, candidate_role in (
        ("dt_col_nm", fallback_contract.date_col, "candidate_date"),
        ("label_col_nm", fallback_contract.target_col, "candidate_target"),
    ):
        if parameters.get(key) is None:
            suggested = fallback if fallback in columns else _candidate(candidate_role)
            if suggested is not None:
                parameters[key] = suggested
                changed = True

    if not changed:
        return False

    body = yaml.safe_load(config.path.read_text(encoding="utf-8")) or {}
    body["parameters"] = parameters
    config.path.write_text(
        yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    logger.info("Populated data-read role suggestions in %s", config.path)
    return True


def _validate(node_id: str, parameters: dict[str, Any]) -> None:
    if node_id == "data-read":
        id_value = parameters.get("id_col_nm")
        if id_value is not None and not (
            (isinstance(id_value, str) and id_value.strip())
            or (
                isinstance(id_value, (list, tuple))
                and bool(id_value)
                and all(isinstance(item, str) and item.strip() for item in id_value)
            )
        ):
            raise ValueError(
                "data-read.parameters.id_col_nm must be a non-empty string, "
                "a non-empty list of strings, or null"
            )
        for key in ("dt_col_nm", "label_col_nm"):
            value = parameters.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"data-read.parameters.{key} must be a non-empty string or null"
                )
        if parameters.get("encoding") not in {None, "utf-8", "utf-8-sig", "gbk"}:
            raise ValueError("data-read.parameters.encoding must be utf-8, utf-8-sig, gbk, or null")
        if not isinstance(parameters.get("infer_schema_length"), int) or parameters["infer_schema_length"] <= 0:
            raise ValueError("data-read.parameters.infer_schema_length must be a positive integer")
    if node_id == "sample-diagnosis":
        # Accept the pre-v1 EDA-shaped template so existing workspaces can be
        # upgraded without silently losing their file. New files keep both
        # diagnostic thresholds and treatment actions in this node YAML.
        legacy_shape = "diagnostics" not in parameters and "treatment" not in parameters
        if not legacy_shape and not isinstance(parameters.get("require_data_read_confirmation"), bool):
            raise ValueError("sample-diagnosis.parameters.require_data_read_confirmation must be boolean")
    if node_id == "feature-processing":
        selection = parameters.get("selection")
        preprocessing = parameters.get("preprocessing")
        if not isinstance(selection, dict) or not isinstance(preprocessing, dict):
            raise ValueError("feature-processing requires selection and preprocessing mappings")
        for key in ("max_missing_rate", "min_iv", "max_correlation", "max_psi", "max_unstable_month_ratio", "max_dominant_rate_warning"):
            if not isinstance(selection.get(key), (int, float)) or not 0 <= float(selection[key]) <= 1:
                raise ValueError(f"feature-processing.selection.{key} must be between 0 and 1")
        if not isinstance(selection.get("min_month_samples"), int) or selection["min_month_samples"] < 1:
            raise ValueError("feature-processing.selection.min_month_samples must be positive")
        if selection.get("stability_action") not in {"review", "drop"}:
            raise ValueError("feature-processing.selection.stability_action must be review or drop")
        if selection.get("correlation_method") not in {"pearson", "spearman"}:
            raise ValueError("feature-processing.selection.correlation_method must be pearson or spearman")
        numeric = preprocessing.get("numeric")
        categorical = preprocessing.get("categorical")
        text = preprocessing.get("text")
        if not all(isinstance(item, dict) for item in (numeric, categorical, text)):
            raise ValueError("feature-processing.preprocessing requires numeric, categorical and text mappings")
        if numeric.get("missing_strategy") != "native":
            raise ValueError("feature-processing numeric.missing_strategy must be native")
        if categorical.get("strategy") not in {"drop", "lightgbm_native"}:
            raise ValueError("feature-processing categorical.strategy must be drop or lightgbm_native")
        if categorical.get("unknown_action") not in {"missing", "other"}:
            raise ValueError("feature-processing categorical.unknown_action must be missing or other")
        if categorical.get("high_cardinality_action") not in {"drop", "native"}:
            raise ValueError("feature-processing categorical.high_cardinality_action must be drop or native")
        if text.get("action") != "drop":
            raise ValueError("feature-processing text.action currently supports only drop")
    if node_id == "eda-analysis":
        if parameters.get("correlation_method") not in {"pearson", "spearman"}:
            raise ValueError("EDA node parameters.correlation_method must be pearson or spearman")
        if parameters.get("metrics_backend") not in {"toad", "legacy"}:
            raise ValueError("EDA node parameters.metrics_backend must be toad or legacy")
        for key in ("bin_count", "max_categories"):
            if not isinstance(parameters.get(key), int) or parameters[key] < 2:
                raise ValueError(f"eda-analysis.parameters.{key} must be an integer >= 2")
        if parameters.get("ks_method") not in {"quantile", "step"}:
            raise ValueError("eda-analysis.parameters.ks_method must be quantile or step")
        if not isinstance(parameters.get("ks_bucket"), int) or parameters["ks_bucket"] < 2:
            raise ValueError("eda-analysis.parameters.ks_bucket must be an integer >= 2")
        if parameters.get("binning_method") not in {"quantile", "chi"}:
            raise ValueError("eda-analysis.parameters.binning_method must be quantile or chi")
        if parameters.get("duplicate_strategy") not in {"keep_first", "keep_last", "error"}:
            raise ValueError("eda-analysis.parameters.duplicate_strategy is invalid")
        for key in ("missing_rate_threshold", "constant_rate_threshold"):
            value = parameters.get(key)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"eda-analysis.parameters.{key} must be between 0 and 1")
        if not isinstance(parameters.get("plot_top_n"), int) or parameters["plot_top_n"] < 0:
            raise ValueError("eda-analysis.parameters.plot_top_n must be >= 0")


def ensure_node_configs(
    config_dir: str | Path,
    node_ids: Iterable[str],
    *,
    template_dir: str | Path | None = None,
) -> dict[str, NodeConfig]:
    """Load node configs and create editable defaults for missing files."""
    directory = Path(config_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configs: dict[str, NodeConfig] = {}
    for node_id in node_ids:
        if node_id not in DEFAULT_NODE_CONFIGS:
            raise ValueError(f"Unsupported workflow node: {node_id}")
        path = directory / NODE_CONFIG_FILENAMES[node_id]
        if not path.is_file():
            template_path = (
                Path(template_dir).expanduser().resolve()
                / NODE_CONFIG_TEMPLATE_FILENAMES[node_id]
                if template_dir is not None
                else None
            )
            if template_path is not None and template_path.is_file():
                body = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
            else:
                body = {"node_id": node_id, **DEFAULT_NODE_CONFIGS[node_id]}
            path.write_text(
                yaml.safe_dump(body, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if raw.get("node_id") != node_id:
            raise ValueError(f"Node config {path} must declare node_id={node_id!r}")
        version = raw.get("version")
        parameters = raw.get("parameters")
        if not isinstance(version, int) or version < 1 or not isinstance(parameters, dict):
            raise ValueError(f"Invalid node config structure: {path}")
        _validate(node_id, parameters)
        configs[node_id] = NodeConfig(node_id, version, parameters, path)
        logger.info("Loaded node config: node_id=%s, path=%s", node_id, path)
    return configs

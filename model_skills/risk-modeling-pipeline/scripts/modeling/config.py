"""Validated configuration for deterministic model-development runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from logger import get_logger


logger = get_logger(__name__)


class ModelConfigError(ValueError):
    """Raised when the model configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class SplitConfig:
    """Rules for reserving OOT data and sampling the development test set."""

    oot_months: int
    test_ratio: float
    random_seed: int


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Initial deterministic feature-quality thresholds."""

    min_iv: float
    max_correlation: float
    max_missing_rate: float
    max_dominant_rate_warning: float
    max_psi: float
    max_unstable_month_ratio: float
    min_month_samples: int
    stability_action: str
    correlation_method: str


@dataclass(frozen=True)
class NumericPreprocessingConfig:
    """Train-fitted numerical feature handling."""

    invalid_to_null: bool
    missing_strategy: str


@dataclass(frozen=True)
class CategoricalPreprocessingConfig:
    """Train-fitted categorical encoding and cardinality safeguards."""

    strategy: str
    rare_min_count: int
    rare_min_rate: float
    unknown_action: str
    near_unique_rate: float
    max_categories: int
    high_cardinality_action: str


@dataclass(frozen=True)
class TextPreprocessingConfig:
    """Conservative handling for free-text-like fields."""

    action: str
    min_average_length: int


@dataclass(frozen=True)
class FeaturePreprocessingConfig:
    """Feature type policies fitted exclusively from Train."""

    numeric: NumericPreprocessingConfig
    categorical: CategoricalPreprocessingConfig
    text: TextPreprocessingConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Select a deterministic baseline run or an Optuna tuning run."""

    mode: str


@dataclass(frozen=True)
class TuningSearchSpace:
    """Bounded LightGBM search ranges used by the TPE sampler."""

    learning_rate: tuple[float, float]
    num_leaves: tuple[int, int]
    max_depth: tuple[int, int]
    min_data_in_leaf: tuple[int, int]
    feature_fraction: tuple[float, float]
    bagging_fraction: tuple[float, float]
    bagging_freq: tuple[int, int]
    lambda_l1: tuple[float, float]
    lambda_l2: tuple[float, float]
    min_gain_to_split: tuple[float, float]


@dataclass(frozen=True)
class TuningConfig:
    """Optuna budget, validation safeguards, and objective penalties."""

    sampler: str
    n_trials: int
    timeout_seconds: int
    startup_trials: int
    cv_folds: int
    random_seed: int
    min_bad_samples_per_fold: int
    auc_gap_penalty: float
    fold_std_penalty: float
    objective_metric: str
    search_space: TuningSearchSpace


@dataclass(frozen=True)
class LightGbmConfig:
    """LightGBM parameters; model fitting is implemented in a later module."""

    objective: str
    metric: str
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    min_gain_to_split: float
    num_boost_round: int
    early_stopping_rounds: int
    early_stopping_metric: str


@dataclass(frozen=True)
class ModelConfig:
    """All configuration sections needed by the modeling pipeline."""

    split: SplitConfig
    feature_preprocessing: FeaturePreprocessingConfig
    feature_selection: FeatureSelectionConfig
    training: TrainingConfig
    tuning: TuningConfig
    model: LightGbmConfig


def _require_mapping(raw: dict, section: str) -> dict:
    value = raw.get(section)
    if not isinstance(value, dict):
        raise ModelConfigError(f"Configuration section '{section}' must be a mapping")
    return value


def _range(raw: dict, name: str, cast: type) -> tuple:
    value = raw[name]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ModelConfigError(f"tuning.search_space.{name} must contain [low, high]")
    low, high = cast(value[0]), cast(value[1])
    if low > high:
        raise ModelConfigError(f"tuning.search_space.{name} low must not exceed high")
    return low, high


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate YAML model configuration before any data is split."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        split_raw = _require_mapping(raw, "split")
        preprocessing_raw = raw.get("feature_preprocessing", {})
        numeric_raw = preprocessing_raw.get("numeric", {})
        categorical_raw = preprocessing_raw.get("categorical", {})
        text_raw = preprocessing_raw.get("text", {})
        selection_raw = _require_mapping(raw, "feature_selection")
        training_raw = raw.get("training", {"mode": "baseline"})
        tuning_raw = raw.get("tuning", {})
        search_raw = tuning_raw.get("search_space", {})
        model_raw = _require_mapping(raw, "model")
        default_search = {
            "learning_rate": [0.01, 0.08],
            "num_leaves": [7, 63],
            "max_depth": [3, 8],
            "min_data_in_leaf": [30, 300],
            "feature_fraction": [0.6, 1.0],
            "bagging_fraction": [0.6, 1.0],
            "bagging_freq": [1, 10],
            "lambda_l1": [0.0001, 20.0],
            "lambda_l2": [0.0001, 20.0],
            "min_gain_to_split": [0.0, 1.0],
        }
        merged_search = {**default_search, **search_raw}
        config = ModelConfig(
            split=SplitConfig(
                oot_months=int(split_raw["oot_months"]),
                test_ratio=float(split_raw["test_ratio"]),
                random_seed=int(split_raw["random_seed"]),
            ),
            feature_preprocessing=FeaturePreprocessingConfig(
                numeric=NumericPreprocessingConfig(
                    invalid_to_null=bool(numeric_raw.get("invalid_to_null", True)),
                    missing_strategy=str(numeric_raw.get("missing_strategy", "native")),
                ),
                categorical=CategoricalPreprocessingConfig(
                    strategy=str(categorical_raw.get("strategy", "lightgbm_native")),
                    rare_min_count=int(categorical_raw.get("rare_min_count", 20)),
                    rare_min_rate=float(categorical_raw.get("rare_min_rate", 0.001)),
                    unknown_action=str(categorical_raw.get("unknown_action", "missing")),
                    near_unique_rate=float(categorical_raw.get("near_unique_rate", 0.98)),
                    max_categories=int(categorical_raw.get("max_categories", 100)),
                    high_cardinality_action=str(
                        categorical_raw.get("high_cardinality_action", "drop")
                    ),
                ),
                text=TextPreprocessingConfig(
                    action=str(text_raw.get("action", "drop")),
                    min_average_length=int(text_raw.get("min_average_length", 64)),
                ),
            ),
            feature_selection=FeatureSelectionConfig(
                min_iv=float(selection_raw["min_iv"]),
                max_correlation=float(selection_raw["max_correlation"]),
                max_missing_rate=float(selection_raw["max_missing_rate"]),
                max_dominant_rate_warning=float(
                    selection_raw.get("max_dominant_rate_warning", 0.95)
                ),
                max_psi=float(selection_raw.get("max_psi", 0.25)),
                max_unstable_month_ratio=float(
                    selection_raw.get("max_unstable_month_ratio", 0.30)
                ),
                min_month_samples=int(selection_raw.get("min_month_samples", 100)),
                stability_action=str(selection_raw.get("stability_action", "review")),
                correlation_method=str(
                    selection_raw.get("correlation_method", "spearman")
                ),
            ),
            training=TrainingConfig(mode=str(training_raw.get("mode", "baseline"))),
            tuning=TuningConfig(
                sampler=str(tuning_raw.get("sampler", "tpe")),
                n_trials=int(tuning_raw.get("n_trials", 50)),
                timeout_seconds=int(tuning_raw.get("timeout_seconds", 900)),
                startup_trials=int(tuning_raw.get("startup_trials", 10)),
                cv_folds=int(tuning_raw.get("cv_folds", 3)),
                random_seed=int(tuning_raw.get("random_seed", split_raw["random_seed"])),
                min_bad_samples_per_fold=int(tuning_raw.get("min_bad_samples_per_fold", 20)),
                auc_gap_penalty=float(tuning_raw.get("auc_gap_penalty", 0.5)),
                fold_std_penalty=float(tuning_raw.get("fold_std_penalty", 0.25)),
                objective_metric=str(tuning_raw.get("objective_metric", "ks")),
                search_space=TuningSearchSpace(
                    learning_rate=_range(merged_search, "learning_rate", float),
                    num_leaves=_range(merged_search, "num_leaves", int),
                    max_depth=_range(merged_search, "max_depth", int),
                    min_data_in_leaf=_range(merged_search, "min_data_in_leaf", int),
                    feature_fraction=_range(merged_search, "feature_fraction", float),
                    bagging_fraction=_range(merged_search, "bagging_fraction", float),
                    bagging_freq=_range(merged_search, "bagging_freq", int),
                    lambda_l1=_range(merged_search, "lambda_l1", float),
                    lambda_l2=_range(merged_search, "lambda_l2", float),
                    min_gain_to_split=_range(merged_search, "min_gain_to_split", float),
                ),
            ),
            model=LightGbmConfig(
                objective=str(model_raw["objective"]),
                metric=str(model_raw["metric"]),
                learning_rate=float(model_raw["learning_rate"]),
                num_leaves=int(model_raw["num_leaves"]),
                max_depth=int(model_raw["max_depth"]),
                min_data_in_leaf=int(model_raw.get("min_data_in_leaf", 20)),
                feature_fraction=float(model_raw["feature_fraction"]),
                bagging_fraction=float(model_raw["bagging_fraction"]),
                bagging_freq=int(model_raw["bagging_freq"]),
                lambda_l1=float(model_raw.get("lambda_l1", 0.0)),
                lambda_l2=float(model_raw["lambda_l2"]),
                min_gain_to_split=float(model_raw.get("min_gain_to_split", 0.0)),
                num_boost_round=int(model_raw["num_boost_round"]),
                early_stopping_rounds=int(model_raw["early_stopping_rounds"]),
                early_stopping_metric=str(model_raw.get("early_stopping_metric", "ks")),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelConfigError(f"Invalid model configuration: {config_path}") from exc

    if config.split.oot_months < 1:
        raise ModelConfigError("split.oot_months must be at least 1")
    if not 0 < config.split.test_ratio < 1:
        raise ModelConfigError("split.test_ratio must be between 0 and 1")
    for name, value in (
        ("feature_preprocessing.categorical.rare_min_rate", config.feature_preprocessing.categorical.rare_min_rate),
        ("feature_preprocessing.categorical.near_unique_rate", config.feature_preprocessing.categorical.near_unique_rate),
        ("feature_selection.min_iv", config.feature_selection.min_iv),
        ("feature_selection.max_correlation", config.feature_selection.max_correlation),
        ("feature_selection.max_missing_rate", config.feature_selection.max_missing_rate),
        ("feature_selection.max_dominant_rate_warning", config.feature_selection.max_dominant_rate_warning),
        ("feature_selection.max_psi", config.feature_selection.max_psi),
        ("feature_selection.max_unstable_month_ratio", config.feature_selection.max_unstable_month_ratio),
        ("model.feature_fraction", config.model.feature_fraction),
        ("model.bagging_fraction", config.model.bagging_fraction),
    ):
        if not 0 <= value <= 1:
            raise ModelConfigError(f"{name} must be between 0 and 1")
    preprocessing = config.feature_preprocessing
    if preprocessing.numeric.missing_strategy != "native":
        raise ModelConfigError("feature_preprocessing.numeric.missing_strategy currently supports only 'native'")
    if preprocessing.categorical.strategy not in {"drop", "lightgbm_native"}:
        raise ModelConfigError("feature_preprocessing.categorical.strategy must be 'drop' or 'lightgbm_native'")
    if preprocessing.categorical.unknown_action not in {"missing", "other"}:
        raise ModelConfigError("feature_preprocessing.categorical.unknown_action must be 'missing' or 'other'")
    if preprocessing.categorical.high_cardinality_action not in {"drop", "native"}:
        raise ModelConfigError("feature_preprocessing.categorical.high_cardinality_action must be 'drop' or 'native'")
    if preprocessing.text.action != "drop":
        raise ModelConfigError("feature_preprocessing.text.action currently supports only 'drop'")
    if preprocessing.categorical.rare_min_count < 1 or preprocessing.categorical.max_categories < 2:
        raise ModelConfigError("Categorical count thresholds are invalid")
    if preprocessing.text.min_average_length < 1:
        raise ModelConfigError("feature_preprocessing.text.min_average_length must be positive")
    if config.feature_selection.min_month_samples < 1:
        raise ModelConfigError("feature_selection.min_month_samples must be positive")
    if config.feature_selection.stability_action not in {"review", "drop"}:
        raise ModelConfigError("feature_selection.stability_action must be 'review' or 'drop'")
    if config.feature_selection.correlation_method not in {"pearson", "spearman"}:
        raise ModelConfigError("feature_selection.correlation_method must be 'pearson' or 'spearman'")
    if config.model.objective != "binary":
        raise ModelConfigError("model.objective must be 'binary' for this workflow")
    if config.training.mode not in {"baseline", "tuning"}:
        raise ModelConfigError("training.mode must be 'baseline' or 'tuning'")
    if config.tuning.sampler != "tpe":
        raise ModelConfigError("tuning.sampler currently supports only 'tpe'")
    if config.tuning.objective_metric not in {"auc", "ks"}:
        raise ModelConfigError("tuning.objective_metric must be 'auc' or 'ks'")
    if config.tuning.n_trials < 1 or config.tuning.timeout_seconds < 1:
        raise ModelConfigError("Tuning trial and timeout budgets must be positive")
    if config.tuning.cv_folds < 2:
        raise ModelConfigError("tuning.cv_folds must be at least 2")
    if config.tuning.startup_trials < 0 or config.tuning.min_bad_samples_per_fold < 1:
        raise ModelConfigError("Tuning startup trials and minimum bad samples are invalid")
    if config.model.num_boost_round < 1 or config.model.early_stopping_rounds < 1:
        raise ModelConfigError("LightGBM iteration counts must be positive")
    if config.model.early_stopping_metric not in {"auc", "ks"}:
        raise ModelConfigError("model.early_stopping_metric must be 'auc' or 'ks'")

    logger.info("Loaded model configuration: %s", config_path)
    return config

"""LightGBM training and scoring, isolated from orchestration logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
import polars as pl

from data.contract import ValidatedDataContract
from logger import get_logger

from .config import LightGbmConfig
from .evaluation import calculate_auc_ks


logger = get_logger(__name__)


@dataclass(frozen=True)
class LightGbmTrainingResult:
    """Trained booster and train-derived feature importance."""

    booster: lgb.Booster
    feature_importance: pl.DataFrame
    training_history: pl.DataFrame
    best_iteration: int


def _feature_matrix(data: pl.DataFrame, features: Iterable[str]) -> np.ndarray:
    return (
        data.select(list(features))
        .cast(pl.Float64)
        .to_numpy()
        .astype(float, copy=False)
    )


def train_lightgbm(
    train: pl.DataFrame,
    test: pl.DataFrame,
    contract: ValidatedDataContract,
    features: tuple[str, ...],
    config: LightGbmConfig,
    *,
    seed: int,
    balance_classes: bool = False,
    categorical_features: tuple[str, ...] = (),
) -> LightGbmTrainingResult:
    """Train with Test early stopping; OOT data is intentionally not accepted."""
    target_col = contract.contract.target_col
    params = {
        "objective": config.objective,
        # KS is not a native LightGBM objective.  We record both AUC and KS
        # through a custom evaluation callback and let the configured first
        # metric control early stopping.
        "metric": "None",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "min_gain_to_split": config.min_gain_to_split,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": -1,
    }
    if balance_classes:
        bad_count = int(
            (train.get_column(target_col) == contract.contract.bad_label).sum()
        )
        good_count = int(
            (train.get_column(target_col) == contract.contract.good_label).sum()
        )
        if not bad_count or not good_count:
            raise ValueError("Class weighting requires both target classes in Train")
        params["scale_pos_weight"] = good_count / bad_count
    train_set = lgb.Dataset(
        _feature_matrix(train, features),
        label=train.get_column(target_col).to_numpy(),
        feature_name=list(features),
        categorical_feature=list(categorical_features),
    )
    test_set = lgb.Dataset(
        _feature_matrix(test, features),
        label=test.get_column(target_col).to_numpy(),
        reference=train_set,
        feature_name=list(features),
        categorical_feature=list(categorical_features),
    )
    logger.info(
        "Training LightGBM: train=%d, test=%d, features=%d",
        train.height,
        test.height,
        len(features),
    )
    evaluation_history: dict[str, dict[str, list[float]]] = {}

    def custom_eval(predictions: np.ndarray, dataset: lgb.Dataset) -> list[tuple[str, float, bool]]:
        auc, ks = calculate_auc_ks(dataset.get_label().astype(int), predictions)
        metrics = {"auc": float(auc or 0.0), "ks": float(ks or 0.0)}
        ordered_names = (
            [config.early_stopping_metric, "auc" if config.early_stopping_metric == "ks" else "ks"]
        )
        return [(name, metrics[name], True) for name in ordered_names]

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, test_set],
        valid_names=["train", "test"],
        feval=custom_eval,
        callbacks=[
            lgb.record_evaluation(evaluation_history),
            lgb.early_stopping(
                config.early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
        ],
    )
    best_iteration = booster.best_iteration or config.num_boost_round
    importance = pl.DataFrame(
        {
            "feature": list(features),
            "gain_importance": booster.feature_importance(
                importance_type="gain"
            ).tolist(),
            "split_importance": booster.feature_importance(
                importance_type="split"
            ).tolist(),
        }
    ).sort("gain_importance", descending=True)
    train_auc = evaluation_history["train"]["auc"]
    test_auc = evaluation_history["test"]["auc"]
    train_ks = evaluation_history["train"]["ks"]
    test_ks = evaluation_history["test"]["ks"]
    training_history = pl.DataFrame(
        {
            "iteration": range(1, len(train_auc) + 1),
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_ks": train_ks,
            "test_ks": test_ks,
        }
    ).with_columns(
        (pl.col("train_auc") - pl.col("test_auc")).alias("auc_gap"),
        (pl.col("train_ks") - pl.col("test_ks")).alias("ks_gap"),
        (pl.col("iteration") == best_iteration).alias("is_best_iteration"),
    )
    logger.info("LightGBM training completed: best_iteration=%d", best_iteration)
    return LightGbmTrainingResult(booster, importance, training_history, best_iteration)


def predict_bad_probability(
    booster: lgb.Booster,
    data: pl.DataFrame,
    features: tuple[str, ...],
    best_iteration: int,
) -> np.ndarray:
    """Return a bad-event probability for each record."""
    return booster.predict(
        _feature_matrix(data, features), num_iteration=best_iteration
    )

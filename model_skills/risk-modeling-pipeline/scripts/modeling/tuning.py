"""Optuna TPE tuning over fixed Train-only stratified folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Any

import polars as pl
import yaml

from data.contract import ValidatedDataContract
from logger import get_logger

from .config import LightGbmConfig, TuningConfig
from .evaluation import evaluate_binary_model
from .lgbm_model import predict_bad_probability, train_lightgbm


logger = get_logger(__name__)


@dataclass(frozen=True)
class TuningResult:
    """Best parameters and complete auditable tuning history."""

    best_config: LightGbmConfig
    best_objective_value: float
    best_trial_number: int
    trials: pl.DataFrame
    fold_metrics: pl.DataFrame
    study_path: Path


def _stratified_fold_ids(labels: list[int], folds: int, seed: int) -> list[list[int]]:
    randomizer = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for row_id, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(row_id)
    result = [[] for _ in range(folds)]
    for row_ids in by_label.values():
        randomizer.shuffle(row_ids)
        for index, row_id in enumerate(row_ids):
            result[index % folds].append(row_id)
    for row_ids in result:
        row_ids.sort()
    return result


def _trial_config(
    trial: Any, base: LightGbmConfig, tuning: TuningConfig
) -> LightGbmConfig:
    space = tuning.search_space
    max_depth = trial.suggest_int("max_depth", *space.max_depth)
    leaf_high = min(space.num_leaves[1], 2**max_depth)
    leaf_low = min(space.num_leaves[0], leaf_high)
    return replace(
        base,
        learning_rate=trial.suggest_float(
            "learning_rate", *space.learning_rate, log=True
        ),
        max_depth=max_depth,
        num_leaves=trial.suggest_int("num_leaves", leaf_low, leaf_high),
        min_data_in_leaf=trial.suggest_int(
            "min_data_in_leaf", *space.min_data_in_leaf, log=True
        ),
        feature_fraction=trial.suggest_float(
            "feature_fraction", *space.feature_fraction
        ),
        bagging_fraction=trial.suggest_float(
            "bagging_fraction", *space.bagging_fraction
        ),
        bagging_freq=trial.suggest_int("bagging_freq", *space.bagging_freq),
        lambda_l1=trial.suggest_float("lambda_l1", *space.lambda_l1, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", *space.lambda_l2, log=True),
        min_gain_to_split=trial.suggest_float(
            "min_gain_to_split", *space.min_gain_to_split
        ),
    )


def _trial_table(study: Any) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_value": trial.value,
            "duration_seconds": trial.duration.total_seconds()
            if trial.duration
            else None,
        }
        row.update({f"param_{name}": value for name, value in trial.params.items()})
        for key in (
            "mean_train_auc",
            "mean_validation_auc",
            "mean_validation_ks",
            "mean_train_ks",
            "auc_gap",
            "ks_gap",
            "fold_auc_std",
            "fold_ks_std",
            "objective_metric",
        ):
            row[key] = trial.user_attrs.get(key)
        rows.append(row)
    return (
        pl.DataFrame(rows) if rows else pl.DataFrame(schema={"trial_number": pl.Int64})
    )


def tune_lightgbm(
    train_data: pl.DataFrame,
    contract: ValidatedDataContract,
    features: tuple[str, ...],
    base_config: LightGbmConfig,
    tuning: TuningConfig,
    output_dir: str | Path,
    *,
    balance_classes: bool = False,
    categorical_features: tuple[str, ...] = (),
) -> TuningResult:
    """Tune on Train only; Test and OOT are intentionally not accepted."""

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required when training.mode=tuning") from exc

    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    target_col = contract.contract.target_col
    labels = [int(value) for value in train_data.get_column(target_col).to_list()]
    fold_ids = _stratified_fold_ids(labels, tuning.cv_folds, tuning.random_seed)
    fold_bad_counts = [sum(labels[index] for index in indices) for indices in fold_ids]
    if min(fold_bad_counts, default=0) < tuning.min_bad_samples_per_fold:
        raise ValueError(
            "Insufficient bad samples for tuning folds: "
            f"counts={fold_bad_counts}, required={tuning.min_bad_samples_per_fold}"
        )

    indexed = train_data.with_row_index("__tuning_row_id")
    fold_rows: list[dict[str, Any]] = []
    study_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{study_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(
        seed=tuning.random_seed,
        n_startup_trials=tuning.startup_trials,
    )
    study = optuna.create_study(
        study_name="risk_modeling_lightgbm",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        # A tuning directory is an immutable experiment unit. Reusing an
        # existing study would silently append trials while losing matching
        # fold details from the earlier process.
        load_if_exists=False,
    )

    def objective(trial: Any) -> float:
        trial_config = _trial_config(trial, base_config, tuning)
        train_aucs: list[float] = []
        validation_aucs: list[float] = []
        train_ks_values: list[float] = []
        validation_ks_values: list[float] = []
        for fold_number, validation_ids in enumerate(fold_ids, start=1):
            validation = indexed.filter(
                pl.col("__tuning_row_id").is_in(validation_ids)
            ).drop("__tuning_row_id")
            fold_train = indexed.filter(
                ~pl.col("__tuning_row_id").is_in(validation_ids)
            ).drop("__tuning_row_id")
            trained = train_lightgbm(
                fold_train,
                validation,
                contract,
                features,
                trial_config,
                seed=tuning.random_seed + trial.number * 100 + fold_number,
                balance_classes=balance_classes,
                categorical_features=categorical_features,
            )
            train_scores = predict_bad_probability(
                trained.booster, fold_train, features, trained.best_iteration
            )
            validation_scores = predict_bad_probability(
                trained.booster, validation, features, trained.best_iteration
            )
            train_eval = evaluate_binary_model(
                fold_train.get_column(target_col).to_list(), train_scores.tolist()
            )
            validation_eval = evaluate_binary_model(
                validation.get_column(target_col).to_list(), validation_scores.tolist()
            )
            if (
                train_eval.auc is None
                or validation_eval.auc is None
                or validation_eval.ks is None
            ):
                raise ValueError(f"Fold {fold_number} lacks both target classes")
            train_aucs.append(float(train_eval.auc))
            validation_aucs.append(float(validation_eval.auc))
            train_ks_values.append(float(train_eval.ks))
            validation_ks_values.append(float(validation_eval.ks))
            fold_rows.append(
                {
                    "trial_number": trial.number,
                    "fold": fold_number,
                    "train_sample_count": fold_train.height,
                    "validation_sample_count": validation.height,
                    "validation_bad_count": int(
                        validation.get_column(target_col).sum()
                    ),
                    "train_auc": train_eval.auc,
                    "validation_auc": validation_eval.auc,
                    "validation_ks": validation_eval.ks,
                    "train_ks": train_eval.ks,
                    "ks_gap": float(train_eval.ks) - float(validation_eval.ks),
                    "auc_gap": float(train_eval.auc) - float(validation_eval.auc),
                    "best_iteration": trained.best_iteration,
                }
            )

        mean_train_auc = mean(train_aucs)
        mean_validation_auc = mean(validation_aucs)
        mean_validation_ks = mean(validation_ks_values)
        auc_gap = mean_train_auc - mean_validation_auc
        mean_train_ks = mean(train_ks_values)
        ks_gap = mean_train_ks - mean_validation_ks
        fold_auc_std = pstdev(validation_aucs) if len(validation_aucs) > 1 else 0.0
        fold_ks_std = pstdev(validation_ks_values) if len(validation_ks_values) > 1 else 0.0
        if tuning.objective_metric == "ks":
            objective_value = mean_validation_ks
            objective_gap = ks_gap
            objective_std = fold_ks_std
        else:
            objective_value = mean_validation_auc
            objective_gap = auc_gap
            objective_std = fold_auc_std
        score = (
            objective_value
            - tuning.auc_gap_penalty * max(0.0, objective_gap)
            - tuning.fold_std_penalty * objective_std
        )
        for key, value in {
            "mean_train_auc": mean_train_auc,
            "mean_validation_auc": mean_validation_auc,
            "mean_validation_ks": mean_validation_ks,
            "mean_train_ks": mean_train_ks,
            "auc_gap": auc_gap,
            "ks_gap": ks_gap,
            "fold_auc_std": fold_auc_std,
            "fold_ks_std": fold_ks_std,
            "objective_metric": tuning.objective_metric,
        }.items():
            trial.set_user_attr(key, value)
        return score

    logger.info(
        "Starting Optuna tuning: trials=%d, folds=%d, features=%d",
        tuning.n_trials,
        tuning.cv_folds,
        len(features),
    )
    study.optimize(
        objective,
        n_trials=tuning.n_trials,
        timeout=tuning.timeout_seconds,
        gc_after_trial=True,
    )
    if not study.best_trials:
        raise RuntimeError("Optuna completed without a successful trial")
    best_trial = study.best_trial
    best_config_values = asdict(base_config)
    best_config_values.update(best_trial.params)
    best_config = LightGbmConfig(**best_config_values)
    trials = _trial_table(study)
    folds = (
        pl.DataFrame(fold_rows)
        if fold_rows
        else pl.DataFrame(schema={"trial_number": pl.Int64})
    )
    trials.write_csv(run_dir / "tuning_trials.csv")
    folds.write_csv(run_dir / "cv_fold_metrics.csv")
    (run_dir / "best_params.json").write_text(
        json.dumps(asdict(best_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "best_params.yaml").write_text(
        yaml.safe_dump(asdict(best_config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(
        "Optuna tuning completed: best_trial=%d, objective=%.6f",
        best_trial.number,
        best_trial.value,
    )
    return TuningResult(
        best_config=best_config,
        best_objective_value=float(best_trial.value),
        best_trial_number=int(best_trial.number),
        trials=trials,
        fold_metrics=folds,
        study_path=study_path,
    )

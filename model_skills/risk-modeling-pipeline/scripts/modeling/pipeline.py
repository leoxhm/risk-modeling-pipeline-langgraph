"""End-to-end LightGBM modeling workflow and model-report export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from contextlib import nullcontext
from collections import Counter
import json
import math
import random
from pathlib import Path
import shutil
from typing import Any

import polars as pl

from ai.model_reviewer import build_model_review
from data.contract import ValidatedDataContract
from logger import get_logger
from preprocessing.cleaning import preprocess_data
from preprocessing.feature_preprocessing import (
    fit_feature_preprocessor,
    transform_features,
)
from progress import ProgressReporter
from reporting.xlsx_report import write_model_report
from reporting.model_html_report import write_model_html_report

from .config import ModelConfig
from .evaluation import (
    bootstrap_metric_intervals,
    evaluate_binary_model,
    population_stability_index,
)
from .export import export_model_artifacts
from .feature_selection import select_features
from .lgbm_model import LightGbmTrainingResult, predict_bad_probability, train_lightgbm
from .llm_tuning import run_llm_tuning
from .split import split_dataset
from .tuning import TuningResult, tune_lightgbm


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelRunResult:
    """Key artifacts created by a single reproducible modeling run."""

    output_dir: Path
    report_path: Path
    selected_features: tuple[str, ...]
    metrics: pl.DataFrame
    split_summary: pl.DataFrame
    html_report_path: Path | None = None


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    return value


def _payload_table(data: pl.DataFrame) -> dict[str, Any]:
    return {
        "headers": data.columns,
        "rows": [[_json_value(value) for value in row] for row in data.rows()],
    }


def _write_csv(data: pl.DataFrame, output_dir: Path, name: str) -> None:
    data.write_csv(output_dir / f"{name}.csv")


def _score_split(
    name: str,
    data: pl.DataFrame,
    training: LightGbmTrainingResult,
    features: tuple[str, ...],
    target_col: str,
) -> tuple[pl.DataFrame, Any]:
    probabilities = predict_bad_probability(
        training.booster, data, features, training.best_iteration
    )
    scored = data.with_columns(
        pl.Series("bad_probability", probabilities),
        pl.lit(name).alias("dataset"),
    )
    evaluation = evaluate_binary_model(
        scored.get_column(target_col).to_list(), probabilities.tolist()
    )
    return scored, evaluation


def _binary_labels(data: pl.DataFrame, target_col: str) -> list[int]:
    values = data.get_column(target_col).to_list()
    return [int(value) for value in values]


def _roc_curve(labels: list[int], scores: list[float]) -> pl.DataFrame:
    """Return an exact ROC path, sorted from lowest to highest FPR."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return pl.DataFrame({"fpr": [], "tpr": [], "threshold": []})
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    rows = [{"fpr": 0.0, "tpr": 0.0, "threshold": None}]
    tp = fp = 0
    previous: float | None = None
    for score, label in ordered:
        if previous is not None and score != previous:
            rows.append({"fpr": fp / negatives, "tpr": tp / positives, "threshold": previous})
        if label:
            tp += 1
        else:
            fp += 1
        previous = score
    rows.append({"fpr": fp / negatives, "tpr": tp / positives, "threshold": previous})
    return pl.DataFrame(rows)


def _ks_curve(labels: list[int], scores: list[float]) -> pl.DataFrame:
    """Return cumulative good/bad rates and KS along risk-ranked samples."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return pl.DataFrame({"population_rate": [], "cumulative_bad_rate": [], "cumulative_good_rate": [], "ks": []})
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    rows = []
    for index, (_, label) in enumerate(ordered, start=1):
        tp += int(label)
        fp += 1 - int(label)
        bad_rate, good_rate = tp / positives, fp / negatives
        rows.append({"population_rate": index / len(ordered), "cumulative_bad_rate": bad_rate, "cumulative_good_rate": good_rate, "ks": abs(bad_rate - good_rate)})
    return pl.DataFrame(rows)


def _feature_score_values(data: pl.DataFrame, feature: str, target_col: str) -> list[float]:
    series = data.get_column(feature)
    target = _binary_labels(data, target_col)
    if series.dtype.is_numeric() and series.dtype != pl.Boolean:
        values = [float(value) if value is not None and math.isfinite(float(value)) else 0.0 for value in series.to_list()]
        return values
    groups: dict[str, list[int]] = {}
    for value, label in zip(series.to_list(), target):
        groups.setdefault("<MISSING>" if value is None else str(value), []).append(label)
    rates = {key: (sum(values) / len(values) if values else 0.0) for key, values in groups.items()}
    return [rates.get("<MISSING>" if value is None else str(value), 0.0) for value in series.to_list()]


def _feature_entropy(data: pl.DataFrame, feature: str) -> float | None:
    values = ["<MISSING>" if value is None else str(value) for value in data.get_column(feature).to_list()]
    if not values:
        return None
    total = len(values)
    return float(-sum((count / total) * math.log2(count / total) for count in Counter(values).values()))


def _variable_information(
    split_data: dict[str, pl.DataFrame],
    contract: ValidatedDataContract,
    features: tuple[str, ...],
) -> pl.DataFrame:
    """Combine importance, IV/missingness and univariate Gini/entropy by split."""
    from eda.analytics import build_eda_analysis

    target_col = contract.contract.target_col
    by_feature: dict[str, dict[str, Any]] = {feature: {"feature": feature} for feature in features}
    for dataset, frame in split_data.items():
        try:
            analysis = build_eda_analysis(frame, replace(contract, feature_cols=features), bin_count=10, ks_bucket=10, metrics_backend="toad")
            overview = {row["feature"]: row for row in analysis.univariate_overview.to_dicts()}
        except Exception as exc:  # report gaps explicitly; training should not fail
            logger.warning("Unable to calculate %s variable information: %s", dataset, exc)
            overview = {}
        labels = _binary_labels(frame, target_col)
        for feature in features:
            row = by_feature[feature]
            source = overview.get(feature, {})
            row[f"{dataset}_iv"] = source.get("iv")
            row[f"{dataset}_missing_rate"] = float(frame.get_column(feature).null_count() / frame.height) if frame.height else None
            row[f"{dataset}_entropy"] = _feature_entropy(frame, feature)
            try:
                evaluation = evaluate_binary_model(labels, _feature_score_values(frame, feature, target_col), metrics_backend="toad")
                row[f"{dataset}_gini"] = 2 * float(evaluation.auc) - 1 if evaluation.auc is not None else None
            except (TypeError, ValueError, ZeroDivisionError):
                row[f"{dataset}_gini"] = None
    result = pl.DataFrame(list(by_feature.values()))
    # The pipeline names the development validation split ``test``. Keep an
    # explicit validation alias in the report so users asking for
    # train/test/validation IV can read the table without guessing terminology.
    return result.with_columns(
        pl.col("test_iv").alias("validation_iv"),
        pl.col("test_missing_rate").alias("validation_missing_rate"),
        pl.col("test_entropy").alias("validation_entropy"),
        pl.col("test_gini").alias("validation_gini"),
    )


def _monthly_performance(
    scored: pl.DataFrame, target_col: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in sorted(scored.get_column("event_month").unique().to_list()):
        month_data = scored.filter(pl.col("event_month") == month)
        result = evaluate_binary_model(
            month_data.get_column(target_col).to_list(),
            month_data.get_column("bad_probability").to_list(),
        )
        rows.append(
            {
                "dataset": month_data.get_column("dataset")[0],
                "event_month": month,
                "sample_count": month_data.height,
                "bad_count": int(month_data.get_column(target_col).sum()),
                "bad_rate": float(month_data.get_column(target_col).mean()),
                "auc": result.auc,
                "ks": result.ks,
                "pr_auc": result.pr_auc,
                "brier_score": result.brier_score,
            }
        )
    return rows


def _detect_group_columns(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    features: tuple[str, ...],
    *,
    max_dimensions: int = 5,
) -> tuple[str, ...]:
    """Find safe low-cardinality dimensions for subgroup diagnostics."""
    raw = contract.contract
    excluded = set(raw.id_cols) | {raw.date_col, raw.target_col, "event_date", "event_month"}
    name_hints = ("channel", "region", "province", "city", "segment", "group", "客群", "渠道", "地区")
    # Group dimensions may be excluded from model features (for example,
    # channel/region is often used only for monitoring). Inspect the full
    # scored frame, while keeping model features first as a deterministic
    # fallback when no explicit dimension is available.
    candidates = list(dict.fromkeys((*features, *data.columns)))
    candidates.sort(
        key=lambda column: (
            0 if any(hint in column.lower() for hint in name_hints) else 1,
            0 if data.get_column(column).dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean) else 1,
            column,
        )
    )
    selected: list[str] = []
    for column in candidates:
        if column in excluded or column not in data.columns:
            continue
        series = data.get_column(column)
        unique_count = series.n_unique()
        if unique_count < 2 or unique_count > 20:
            continue
        if series.dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean) or any(
            hint in column.lower() for hint in name_hints
        ):
            selected.append(column)
        if len(selected) >= max_dimensions:
            break
    return tuple(selected)


def _group_performance(
    scored_by_name: dict[str, pl.DataFrame],
    contract: ValidatedDataContract,
    features: tuple[str, ...],
) -> pl.DataFrame:
    """Calculate model performance by month and detected low-cardinality groups."""
    target_col = contract.contract.target_col
    dimensions = ("event_month",) + _detect_group_columns(
        scored_by_name["train"], contract, features
    )
    rows: list[dict[str, Any]] = []
    for dataset, frame in scored_by_name.items():
        for dimension in dimensions:
            if dimension not in frame.columns:
                continue
            values = frame.get_column(dimension).unique().sort().to_list()
            for value in values:
                subset = (
                    frame.filter(pl.col(dimension).is_null())
                    if value is None
                    else frame.filter(pl.col(dimension) == value)
                )
                if subset.height < 20:
                    continue
                evaluation = evaluate_binary_model(
                    subset.get_column(target_col).to_list(),
                    subset.get_column("bad_probability").to_list(),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "dimension": dimension,
                        "group_value": "<MISSING>" if value is None else str(value),
                        "sample_count": subset.height,
                        "bad_count": int(subset.get_column(target_col).sum()),
                        "bad_rate": float(subset.get_column(target_col).mean()),
                        "auc": evaluation.auc,
                        "ks": evaluation.ks,
                        "pr_auc": evaluation.pr_auc,
                        "brier_score": evaluation.brier_score,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "dataset": pl.String,
            "dimension": pl.String,
            "group_value": pl.String,
            "sample_count": pl.Int64,
            "bad_count": pl.Int64,
            "bad_rate": pl.Float64,
            "auc": pl.Float64,
            "ks": pl.Float64,
            "pr_auc": pl.Float64,
            "brier_score": pl.Float64,
        }
    )


def _permutation_importance_by_split(
    scored_by_name: dict[str, pl.DataFrame],
    training: LightGbmTrainingResult,
    features: tuple[str, ...],
    target_col: str,
    *,
    seed: int = 42,
) -> pl.DataFrame:
    """Estimate feature importance drift by AUC loss after deterministic shuffling."""
    rows: list[dict[str, Any]] = [{"feature": feature} for feature in features]
    for dataset, frame in scored_by_name.items():
        labels = frame.get_column(target_col).to_list()
        baseline = evaluate_binary_model(
            labels, frame.get_column("bad_probability").to_list()
        ).auc
        for index, feature in enumerate(features):
            values = frame.get_column(feature).to_list()
            shuffled = list(values)
            random.Random(seed + index).shuffle(shuffled)
            permuted = frame.with_columns(pl.Series(feature, shuffled))
            permuted_scores = predict_bad_probability(
                training.booster, permuted, features, training.best_iteration
            )
            permuted_auc = evaluate_binary_model(
                labels, permuted_scores.tolist()
            ).auc
            rows[index][f"{dataset}_permutation_importance"] = (
                max(0.0, float(baseline) - float(permuted_auc))
                if baseline is not None and permuted_auc is not None
                else None
            )
    return pl.DataFrame(rows)


def _seed_stability(
    split_data: dict[str, pl.DataFrame],
    final_config: Any,
    contract: ValidatedDataContract,
    features: tuple[str, ...],
    target_col: str,
    *,
    base_seed: int,
    balance_classes: bool,
    categorical_features: tuple[str, ...],
    seed_count: int = 3,
) -> pl.DataFrame:
    """Retrain the accepted configuration under several seeds and report spread."""
    rows: list[dict[str, Any]] = []
    for offset in range(seed_count):
        seed = base_seed + offset * 1009
        trained = train_lightgbm(
            split_data["train"],
            split_data["test"],
            contract,
            features,
            final_config,
            seed=seed,
            balance_classes=balance_classes,
            categorical_features=categorical_features,
        )
        for dataset, frame in split_data.items():
            scores = predict_bad_probability(
                trained.booster, frame, features, trained.best_iteration
            )
            evaluation = evaluate_binary_model(
                frame.get_column(target_col).to_list(), scores.tolist()
            )
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "best_iteration": trained.best_iteration,
                    "auc": evaluation.auc,
                    "ks": evaluation.ks,
                    "pr_auc": evaluation.pr_auc,
                    "brier_score": evaluation.brier_score,
                }
            )
    return pl.DataFrame(rows)


def _candidate_row(
    name: str,
    training: LightGbmTrainingResult,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: tuple[str, ...],
    target_col: str,
    parameters: Any | None = None,
    oot: pl.DataFrame | None = None,
) -> dict[str, object]:
    train_scores = predict_bad_probability(
        training.booster, train, features, training.best_iteration
    )
    test_scores = predict_bad_probability(
        training.booster, test, features, training.best_iteration
    )
    train_eval = evaluate_binary_model(
        train.get_column(target_col).to_list(), train_scores.tolist()
    )
    test_eval = evaluate_binary_model(
        test.get_column(target_col).to_list(), test_scores.tolist()
    )
    oot_eval = None
    if oot is not None:
        oot_scores = predict_bad_probability(
            training.booster, oot, features, training.best_iteration
        )
        oot_eval = evaluate_binary_model(
            oot.get_column(target_col).to_list(), oot_scores.tolist()
        )
    train_auc = float(train_eval.auc) if train_eval.auc is not None else None
    test_auc = float(test_eval.auc) if test_eval.auc is not None else None
    return {
        "candidate": name,
        "best_iteration": training.best_iteration,
        "train_auc": train_auc,
        "train_ks": train_eval.ks,
        "test_auc": test_auc,
        "test_ks": test_eval.ks,
        # ``test_*`` is kept for backward-compatible reports; in the
        # modeling workflow this split is the independent Validate holdout.
        "validate_auc": test_auc,
        "validate_ks": test_eval.ks,
        "oot_auc": oot_eval.auc if oot_eval is not None else None,
        "oot_ks": oot_eval.ks if oot_eval is not None else None,
        "parameters": json.dumps(asdict(parameters), ensure_ascii=False) if parameters is not None else None,
        "auc_gap": train_auc - test_auc
        if train_auc is not None and test_auc is not None
        else None,
    }


def run_model_pipeline(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    config: ModelConfig,
    output_dir: str | Path,
    *,
    node_executable: str = "node",
    config_path: str | Path | None = None,
    progress: ProgressReporter | None = None,
    balance_classes: bool = False,
    pmml_converter: str | Path | None = None,
    model_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    generate_report: bool = True,
    llm_tuning: dict[str, Any] | None = None,
) -> ModelRunResult:
    """Run cleaning, split, selection, LightGBM training, evaluation, and reporting."""
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = (
        Path(model_dir).expanduser().resolve() if model_dir is not None else run_dir
    )
    report_output_dir = (
        Path(report_dir).expanduser().resolve() if report_dir is not None else run_dir
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting model pipeline: %s", run_dir)
    cleaning_stage = (
        progress.track(
            "cleaning",
            running_summary="正在准备建模数据",
            success_summary="建模数据清理完成",
            artifacts=[run_dir / "cleaned_data.parquet"],
        )
        if progress
        else nullcontext()
    )
    with cleaning_stage:
        cleaning = preprocess_data(data, contract, filter_features=False)
        cleaned_contract = replace(contract, feature_cols=cleaning.feature_cols)
        cleaning.data.write_parquet(run_dir / "cleaned_data.parquet")

    selection_stage = (
        progress.track(
            "feature-selection",
            running_summary="正在切分样本并使用 Train 数据筛选特征",
            success_summary="数据切分和特征筛选完成",
            artifacts=[
                run_dir / "split_summary.csv",
                run_dir / "feature_preprocessing.csv",
                run_dir / "feature_preprocessing.json",
                run_dir / "feature_selection.csv",
            ],
        )
        if progress
        else nullcontext()
    )
    with selection_stage:
        raw_split = split_dataset(cleaning.data, cleaned_contract, config.split)
        preprocessing_plan = fit_feature_preprocessor(
            raw_split.train,
            cleaned_contract,
            config.feature_preprocessing,
        )
        preprocessed_contract = replace(
            cleaned_contract,
            feature_cols=preprocessing_plan.retained_features,
        )
        selection = select_features(
            raw_split.train,
            preprocessed_contract,
            config.feature_selection,
        )
        split = replace(
            raw_split,
            train=transform_features(raw_split.train, preprocessing_plan),
            test=transform_features(raw_split.test, preprocessing_plan),
            oot=transform_features(raw_split.oot, preprocessing_plan),
        )
        _write_csv(split.summary, run_dir, "split_summary")
        _write_csv(
            preprocessing_plan.decisions,
            run_dir,
            "feature_preprocessing",
        )
        (run_dir / "feature_preprocessing.json").write_text(
            json.dumps(preprocessing_plan.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(selection.decisions, run_dir, "feature_selection")
        pl.concat(
            [
                split.train.with_columns(pl.lit("train").alias("dataset")),
                split.test.with_columns(pl.lit("test").alias("dataset")),
                split.oot.with_columns(pl.lit("oot").alias("dataset")),
            ]
        ).write_parquet(run_dir / "model_matrix.parquet")

    target_col = preprocessed_contract.contract.target_col
    categorical_features = tuple(
        feature
        for feature in selection.feature_cols
        if feature in preprocessing_plan.categorical_features
    )
    tuning_result: TuningResult | None = None
    # The tuning backend is intentionally exclusive.  ``method=optuna``
    # runs the deterministic search, ``method=llm`` starts LLM refinement from
    # the baseline, and ``method=none`` trains only the baseline model.
    tuning_method = getattr(config.tuning, "method", "llm")
    if tuning_method == "optuna":
        tuning_stage = (
            progress.track(
                "tuning",
                running_summary="正在运行 Optuna Train-only 交叉验证",
                success_summary="Optuna 自动调参完成",
                artifacts=[run_dir / "tuning"],
            )
            if progress
            else nullcontext()
        )
        with tuning_stage:
            tuning_result = tune_lightgbm(
                split.train,
                cleaned_contract,
                selection.feature_cols,
                config.model,
                config.tuning,
                run_dir / "tuning",
                balance_classes=balance_classes,
                categorical_features=categorical_features,
                progress=progress,
            )
    elif progress:
        progress.emit(
            "tuning",
            "skipped",
            summary=(
                "当前调参方式为 LLM，跳过 Optuna；"
                "LLM 将从 Baseline 参数开始迭代"
                if tuning_method == "llm"
                else "当前配置不使用自动调参，跳过 Optuna 和 LLM"
            ),
        )

    training_stage = (
        progress.track(
            "training",
            running_summary="正在训练基线模型和最终 LightGBM 模型",
            success_summary="LightGBM 模型训练完成",
            artifacts=[
                model_output_dir / "lightgbm_model.txt",
                model_output_dir / "lightgbm_model.pkl",
                model_output_dir / "model_bundle.pkl",
                model_output_dir / "pmml_export_status.json",
                run_dir / "training_history.csv",
            ],
        )
        if progress
        else nullcontext()
    )
    candidate_rows: list[dict[str, object]] = []
    llm_history_table = pl.DataFrame(schema={"round": pl.Int64})
    llm_summary_path: Path | None = None
    selected_candidate = "baseline"
    acceptance_reason = "未启用调参，使用 Baseline 模型"
    acceptance_improvement: float | None = None
    with training_stage:
        baseline_training = train_lightgbm(
            split.train,
            split.test,
            cleaned_contract,
            selection.feature_cols,
            config.model,
            seed=config.split.random_seed,
            balance_classes=balance_classes,
            categorical_features=categorical_features,
        )
        tuned_training = (
            train_lightgbm(
                split.train,
                split.test,
                cleaned_contract,
                selection.feature_cols,
                tuning_result.best_config,
                seed=config.split.random_seed,
                balance_classes=balance_classes,
                categorical_features=categorical_features,
            )
            if tuning_result is not None
            else None
        )

        candidate_rows.append(
            _candidate_row(
                "baseline",
                baseline_training,
                split.train,
                split.test,
                selection.feature_cols,
                target_col,
                config.model,
                split.oot,
            )
        )
        if progress is not None:
            baseline_row = candidate_rows[0]
            baseline_metric = baseline_row.get(
                f"test_{config.training.acceptance_metric}"
            )
            progress.emit(
                "tuning",
                "running",
                    summary="Baseline 模型已完成 Validate 集评估，并完成 OOT 校验",
                experiment={
                    "kind": "training",
                    "phase": "baseline_evaluation",
                    "candidate": "baseline",
                    "status": "baseline",
                    "parameters": asdict(config.model),
                    "metrics": {
                        "acceptance_metric": config.training.acceptance_metric,
                        "train_ks": baseline_row.get("train_ks"),
                        "test_auc": baseline_row.get("test_auc"),
                        "test_ks": baseline_row.get("test_ks"),
                        "validate_auc": baseline_row.get("validate_auc"),
                        "validate_ks": baseline_row.get("validate_ks"),
                        "oot_auc": baseline_row.get("oot_auc"),
                        "oot_ks": baseline_row.get("oot_ks"),
                    },
                    "objective": baseline_metric,
                    "best_so_far": True,
                    "accepted": True,
                    "reason": "作为后续 Optuna/LLM 候选的比较基线",
                },
            )
        training = baseline_training
        final_model_config = config.model
        metric_column = (
            "test_ks" if config.training.acceptance_metric == "ks" else "test_auc"
        )
        if tuned_training is not None:
            candidate_rows.append(
                _candidate_row(
                    "optuna_tuned",
                    tuned_training,
                    split.train,
                    split.test,
                    selection.feature_cols,
                    target_col,
                    tuning_result.best_config,
                    split.oot,
                )
            )
            baseline_score = candidate_rows[0].get(metric_column)
            tuned_score = candidate_rows[1].get(metric_column)
            if baseline_score is None or tuned_score is None:
                acceptance_reason = (
                    f"无法根据 {config.training.acceptance_metric.upper()} 比较，"
                    "保留 Baseline 模型"
                )
            else:
                acceptance_improvement = float(tuned_score) - float(baseline_score)
                if acceptance_improvement >= config.training.min_improvement:
                    selected_candidate = "optuna_tuned"
                    training = tuned_training
                    final_model_config = tuning_result.best_config
                    acceptance_reason = (
                        f"调参模型 {config.training.acceptance_metric.upper()} 提升 "
                        f"{acceptance_improvement:.6f}，达到最低提升要求"
                    )
                else:
                    acceptance_reason = (
                        f"调参模型 {config.training.acceptance_metric.upper()} 提升 "
                        f"{acceptance_improvement:.6f}，低于最低要求，保留 Baseline"
                    )
            if progress is not None:
                tuned_row = candidate_rows[-1]
                best_trial_rows = tuning_result.trials.filter(
                    pl.col("trial_number") == tuning_result.best_trial_number
                )
                best_trial_metrics = (
                    best_trial_rows.row(0, named=True)
                    if best_trial_rows.height
                    else {}
                )
                progress.emit(
                    "tuning",
                    "running",
                    summary="Optuna 最优参数已完成 Validate 集和 OOT 集最终评估",
                    experiment={
                        "kind": "optuna",
                        "phase": "final_evaluation",
                        "candidate": "optuna_tuned",
                        "trial": tuning_result.best_trial_number + 1,
                        "total_trials": config.tuning.n_trials,
                        "status": "accepted" if selected_candidate == "optuna_tuned" else "rejected_metric",
                        "parameters": asdict(tuning_result.best_config),
                        "metrics": {
                            "objective_metric": config.tuning.objective_metric,
                            "acceptance_metric": config.training.acceptance_metric,
                            "mean_validation_auc": best_trial_metrics.get("mean_validation_auc"),
                            "mean_validation_ks": best_trial_metrics.get("mean_validation_ks"),
                            "test_auc": tuned_row.get("test_auc"),
                            "test_ks": tuned_row.get("test_ks"),
                            "validate_auc": tuned_row.get("validate_auc"),
                            "validate_ks": tuned_row.get("validate_ks"),
                            "train_ks": tuned_row.get("train_ks"),
                            "oot_ks": tuned_row.get("oot_ks"),
                            "improvement": acceptance_improvement,
                            "improvement_vs_baseline": acceptance_improvement,
                            "oot_improvement_vs_baseline": (
                                float(tuned_row["oot_ks"]) - float(candidate_rows[0]["oot_ks"])
                                if isinstance(tuned_row.get("oot_ks"), (int, float))
                                and isinstance(candidate_rows[0].get("oot_ks"), (int, float))
                                else None
                            ),
                        },
                        "objective": tuning_result.best_objective_value,
                        "best_so_far": True,
                        "accepted": selected_candidate == "optuna_tuned",
                        "reason": acceptance_reason,
                    },
                )
        # Optional LLM refinement.  It is mutually exclusive with Optuna:
        # when ``method=llm`` it starts from the baseline (because the Optuna
        # branch above was skipped), and when ``method=optuna`` it is ignored.
        # The LLM never replaces the
        # deterministic evaluator: every proposal is hard-bound, retrained on
        # the frozen Train/Test split, and accepted only when the configured
        # Test metric improves by the minimum amount.  Provider failures are
        # captured by the tuner and cannot fail this training run.
        llm_enabled = (
            tuning_method == "llm"
            and isinstance(llm_tuning, dict)
            and bool(llm_tuning.get("enabled", False))
        )
        if llm_enabled:
            llm_dir = run_dir / "llm-tuning"
            llm_stage = (
                progress.track(
                    "llm-tuning",
                    running_summary="正在以当前最优参数为起点进行 LLM 迭代调参",
                    success_summary="LLM 调参迭代完成，已记录每轮效果对比",
                    artifacts=[llm_dir],
                )
                if progress
                else nullcontext()
            )
            with llm_stage:
                previous_candidate = selected_candidate
                previous_config = final_model_config
                previous_training = training
                refined_config, refined_training, llm_history, llm_history_table, llm_candidates = run_llm_tuning(
                    split.train,
                    split.test,
                    cleaned_contract,
                    selection.feature_cols,
                    previous_config,
                    previous_training,
                    config.tuning.search_space,
                    llm_tuning,
                    llm_dir,
                    oot=split.oot,
                    baseline_score=(
                        float(candidate_rows[0].get(metric_column))
                        if candidate_rows and isinstance(candidate_rows[0].get(metric_column), (int, float))
                        else None
                    ),
                    baseline_oot_ks=(
                        float(candidate_rows[0].get("oot_ks"))
                        if candidate_rows and isinstance(candidate_rows[0].get("oot_ks"), (int, float))
                        else None
                    ),
                    objective_metric=config.training.acceptance_metric,
                    min_improvement=float(llm_tuning.get("min_improvement", config.training.min_improvement)),
                    seed=config.split.random_seed,
                    balance_classes=balance_classes,
                    categorical_features=categorical_features,
                    progress=progress,
                )
                llm_summary_path = llm_dir / "llm_tuning_summary.json"
                # Keep every successfully retrained LLM proposal in the
                # comparison table, including candidates rejected by the
                # minimum-improvement gate.
                for round_number, candidate_config, candidate_training, candidate_record in llm_candidates:
                    comparison_row = _candidate_row(
                        f"llm_round_{round_number}",
                        candidate_training,
                        split.train,
                        split.test,
                        selection.feature_cols,
                        target_col,
                        candidate_config,
                        split.oot,
                    )
                    comparison_row.update(
                        {
                            "llm_round": round_number,
                            "llm_status": candidate_record.get("status"),
                            "llm_reason": candidate_record.get("reason"),
                            "llm_rejection_reason": candidate_record.get("rejection_reason"),
                            "llm_parameters": json.dumps(asdict(candidate_config), ensure_ascii=False),
                        }
                    )
                    candidate_rows.append(comparison_row)
                accepted_rounds = [row for row in llm_history if row.get("accepted")]
                if accepted_rounds:
                    selected_candidate = "llm_tuned"
                    training = refined_training
                    final_model_config = refined_config
                    candidate_rows.append(
                        _candidate_row(
                            "llm_tuned",
                            refined_training,
                            split.train,
                            split.test,
                            selection.feature_cols,
                            target_col,
                            refined_config,
                            split.oot,
                        )
                    )
                    last = accepted_rounds[-1]
                    acceptance_improvement = (
                        float(last["candidate_score"]) - float(candidate_rows[0][metric_column])
                        if last.get("candidate_score") is not None and candidate_rows[0].get(metric_column) is not None
                        else acceptance_improvement
                    )
                    acceptance_reason = (
                        f"LLM 调参在 {len(accepted_rounds)} 轮中接受最后一轮，"
                        f"{config.training.acceptance_metric.upper()} 相对当前最优继续提升"
                    )
                else:
                    # Keep the exact Optuna/Baseline winner when no LLM round
                    # passes the acceptance gate.
                    selected_candidate = previous_candidate
                    acceptance_reason = (
                        "LLM 调参已执行，但没有候选达到接受条件，保留当前基线模型"
                    )
        elif progress and tuning_method == "optuna" and isinstance(llm_tuning, dict) and bool(llm_tuning.get("enabled", False)):
            progress.emit(
                "llm-tuning",
                "skipped",
                summary="当前调参方式为 Optuna，已跳过 LLM 调参；两种方式互斥",
            )
        metric_column = (
            "test_ks" if config.training.acceptance_metric == "ks" else "test_auc"
        )
        baseline_score = candidate_rows[0].get(metric_column)
        for row in candidate_rows:
            row["selected_candidate"] = row["candidate"] == selected_candidate
            row["acceptance_metric"] = config.training.acceptance_metric
            row["min_improvement"] = config.training.min_improvement
            row["acceptance_score"] = row.get(metric_column)
            if row["candidate"] == "baseline":
                row["improvement_vs_baseline"] = 0.0
            elif row.get(metric_column) is not None and baseline_score is not None:
                row["improvement_vs_baseline"] = float(row[metric_column]) - float(baseline_score)
            else:
                row["improvement_vs_baseline"] = None
            baseline_oot = candidate_rows[0].get("oot_ks")
            if row["candidate"] == "baseline":
                row["oot_improvement_vs_baseline"] = 0.0
            elif isinstance(row.get("oot_ks"), (int, float)) and isinstance(baseline_oot, (int, float)):
                row["oot_improvement_vs_baseline"] = float(row["oot_ks"]) - float(baseline_oot)
            else:
                row["oot_improvement_vs_baseline"] = None
        model_artifacts = export_model_artifacts(
            training.booster,
            model_output_dir,
            best_iteration=training.best_iteration,
            feature_cols=selection.feature_cols,
            preprocessing_plan=preprocessing_plan,
            model_config=final_model_config,
            contract=preprocessed_contract,
            pmml_converter=pmml_converter,
        )
        _write_csv(training.training_history, run_dir, "training_history")

    candidate_comparison = pl.DataFrame(candidate_rows)
    train_scored, train_evaluation = _score_split(
        "train", split.train, training, selection.feature_cols, target_col
    )
    test_scored, test_evaluation = _score_split(
        "test", split.test, training, selection.feature_cols, target_col
    )
    oot_scored, oot_evaluation = _score_split(
        "oot", split.oot, training, selection.feature_cols, target_col
    )
    scored_data = pl.concat([train_scored, test_scored, oot_scored]).sort(
        list(cleaned_contract.contract.id_cols)
    )

    evaluations = {
        "train": train_evaluation,
        "test": test_evaluation,
        "oot": oot_evaluation,
    }
    scored_by_name = {"train": train_scored, "test": test_scored, "oot": oot_scored}
    train_scores = train_scored.get_column("bad_probability").to_list()
    metric_rows: list[dict[str, object]] = []
    lift_frames: list[pl.DataFrame] = []
    monthly_rows: list[dict[str, object]] = []
    for name in ("train", "test", "oot"):
        scored = scored_by_name[name]
        evaluation = evaluations[name]
        metric_rows.append(
            {
                "dataset": name,
                "sample_count": scored.height,
                "bad_count": int(scored.get_column(target_col).sum()),
                "bad_rate": float(scored.get_column(target_col).mean()),
                "auc": evaluation.auc,
                "ks": evaluation.ks,
                "pr_auc": evaluation.pr_auc,
                "brier_score": evaluation.brier_score,
                "metrics_backend": evaluation.backend,
                "score_psi_vs_train": population_stability_index(
                    train_scores, scored.get_column("bad_probability").to_list()
                ),
            }
        )
        if evaluation.lift.height:
            lift_frames.append(
                evaluation.lift.with_columns(pl.lit(name).alias("dataset"))
            )
        monthly_rows.extend(_monthly_performance(scored, target_col))
    metrics = pl.DataFrame(metric_rows)
    lift_detail = pl.concat(lift_frames).select(
        [
            "dataset",
            "bucket",
            "score_min",
            "score_max",
            "sample_count",
            "bad_count",
            "bad_rate",
            "lift",
            "cumulative_sample_rate",
            "cumulative_bad_capture",
        ]
    )
    monthly_performance = pl.DataFrame(monthly_rows).sort(["dataset", "event_month"])
    roc_frames = []
    for name, frame in scored_by_name.items():
        roc = _roc_curve(
            _binary_labels(frame, target_col),
            frame.get_column("bad_probability").to_list(),
        )
        if roc.height:
            roc_frames.append(roc.with_columns(pl.lit(name).alias("dataset")))
    curve_roc = pl.concat(roc_frames).select(
        ["dataset", "fpr", "tpr", "threshold"]
    ) if roc_frames else pl.DataFrame(
        schema={"dataset": pl.String, "fpr": pl.Float64, "tpr": pl.Float64, "threshold": pl.Float64}
    )
    curve_ks = _ks_curve(_binary_labels(split.train, target_col), train_scores)
    calibration_frames = []
    bootstrap_frames = []
    for name, evaluation in evaluations.items():
        if evaluation.calibration is not None and evaluation.calibration.height:
            calibration_frames.append(
                evaluation.calibration.with_columns(pl.lit(name).alias("dataset"))
            )
        bootstrap = bootstrap_metric_intervals(
            _binary_labels(scored_by_name[name], target_col),
            scored_by_name[name].get_column("bad_probability").to_list(),
        )
        if bootstrap.height:
            bootstrap_frames.append(bootstrap.with_columns(pl.lit(name).alias("dataset")))
    calibration = pl.concat(calibration_frames).select(
        ["dataset", "bucket", "sample_count", "mean_predicted", "observed_bad_rate", "absolute_error"]
    ) if calibration_frames else pl.DataFrame()
    bootstrap_intervals = pl.concat(bootstrap_frames).select(
        ["dataset", "metric", "estimate", "lower", "upper", "rounds"]
    ) if bootstrap_frames else pl.DataFrame()
    importance = training.feature_importance
    total_gain = float(importance.get_column("gain_importance").sum() or 0.0) if importance.height else 0.0
    importance = importance.with_columns(
        (pl.col("gain_importance") / total_gain if total_gain else pl.lit(0.0)).alias("gain_share")
    ).with_columns(pl.col("gain_share").cum_sum().alias("cumulative_gain_share"))
    variable_information = _variable_information(
        {"train": split.train, "test": split.test, "oot": split.oot},
        preprocessed_contract,
        selection.feature_cols,
    ).join(
        importance.select(["feature", "gain_importance", "split_importance", "gain_share", "cumulative_gain_share"]),
        on="feature",
        how="left",
    ).sort("gain_importance", descending=True)
    importance_by_split = _permutation_importance_by_split(
        scored_by_name,
        training,
        selection.feature_cols,
        target_col,
    ).join(
        importance.select(["feature", "gain_importance", "split_importance", "gain_share"]),
        on="feature",
        how="left",
    )
    seed_stability = _seed_stability(
        {"train": split.train, "test": split.test, "oot": split.oot},
        final_model_config,
        preprocessed_contract,
        selection.feature_cols,
        target_col,
        base_seed=config.split.random_seed,
        balance_classes=balance_classes,
        categorical_features=categorical_features,
    )
    group_performance = _group_performance(
        scored_by_name,
        preprocessed_contract,
        selection.feature_cols,
    )
    stability_rows = metrics.select(
        ["dataset", "sample_count", "bad_count", "bad_rate", "auc", "ks", "pr_auc", "brier_score", "score_psi_vs_train", "metrics_backend"]
    )
    review_path = run_dir / "ai_model_review.json"
    review_stage = (
        progress.track(
            "review",
            running_summary="正在审查泛化差距、稳定性和模型风险",
            success_summary="模型审查完成",
            artifacts=[review_path],
        )
        if progress
        else nullcontext()
    )
    with review_stage:
        model_review = build_model_review(
            metrics,
            best_iteration=training.best_iteration,
            feature_importance=training.feature_importance,
            training_history=training.training_history,
            lift_detail=lift_detail,
            monthly_performance=monthly_performance,
            split_summary=split.summary,
        )
        review_path.write_text(
            json.dumps(model_review, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    scored_data.write_parquet(run_dir / "scored_data.parquet")
    _write_csv(importance, run_dir, "feature_importance")
    _write_csv(metrics, run_dir, "metrics_by_split")
    _write_csv(lift_detail, run_dir, "lift_detail")
    _write_csv(monthly_performance, run_dir, "monthly_performance")
    _write_csv(candidate_comparison, run_dir, "candidate_comparison")
    _write_csv(curve_roc, run_dir, "roc_curve")
    _write_csv(curve_ks, run_dir, "ks_curve")
    _write_csv(calibration, run_dir, "calibration_curve")
    _write_csv(bootstrap_intervals, run_dir, "bootstrap_intervals")
    _write_csv(seed_stability, run_dir, "seed_stability")
    _write_csv(group_performance, run_dir, "group_performance")
    _write_csv(importance_by_split, run_dir, "feature_importance_by_split")
    _write_csv(variable_information, run_dir, "variable_information")
    _write_csv(stability_rows, run_dir, "model_stability")
    candidate_comparison_json = run_dir / "candidate_comparison.json"
    candidate_comparison_json.write_text(
        json.dumps(candidate_comparison.to_dicts(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if llm_history_table.height or llm_summary_path is not None:
        _write_csv(llm_history_table, run_dir, "llm_tuning_history")
    if config_path is not None:
        shutil.copy2(Path(config_path), run_dir / "model_config.yaml")

    review_messages = [finding["message"] for finding in model_review["findings"]]
    summary = {
        "model_type": "LightGBM",
        "training_mode": config.training.mode,
        "tuning_method": tuning_method,
        "selected_candidate": selected_candidate,
        "acceptance_metric": config.training.acceptance_metric,
        "acceptance_min_improvement": config.training.min_improvement,
        "acceptance_improvement_vs_baseline": acceptance_improvement,
        "acceptance_reason": acceptance_reason,
        "class_imbalance_action": "class_weight" if balance_classes else "none",
        "best_iteration": training.best_iteration,
        "best_trial_number": tuning_result.best_trial_number if tuning_result else None,
        "best_cv_objective": tuning_result.best_objective_value
        if tuning_result
        else None,
        "tuning_trials_artifact": str(run_dir / "tuning" / "tuning_trials.csv")
        if tuning_result
        else None,
        "cv_fold_metrics_artifact": str(run_dir / "tuning" / "cv_fold_metrics.csv")
        if tuning_result
        else None,
        "tuning_cv_strategy": config.tuning.cv_strategy
        if tuning_method == "optuna"
        else None,
        "tuning_cv_folds": (
            int(tuning_result.fold_metrics.get_column("fold").n_unique())
            if tuning_result is not None and "fold" in tuning_result.fold_metrics.columns
            else (config.tuning.cv_folds if tuning_method == "optuna" else None)
        ),
        "tuning_validation_months": config.tuning.validation_months
        if tuning_method == "optuna"
        else None,
        "tuning_gap_months": config.tuning.gap_months
        if tuning_method == "optuna"
        else None,
        "final_model_parameters": asdict(final_model_config),
        "selected_feature_count": len(selection.feature_cols),
        "selected_features": list(selection.feature_cols),
        "categorical_feature_count": len(categorical_features),
        "categorical_features": list(categorical_features),
        "split_strategy": config.split.strategy,
        "test_month_values": list(split.test_month_values),
        "oot_month_values": list(split.oot_month_values),
        "review_status": model_review["status"],
        "review_finding_count": len(model_review["findings"]),
        "review_blocker_count": sum(
            finding["severity"] == "blocker" for finding in model_review["findings"]
        ),
        "review_warning_count": sum(
            finding["severity"] == "warning" for finding in model_review["findings"]
        ),
        "review_recommendation_count": len(model_review["recommendations"]),
        "review_artifact": review_path.name,
        "llm_tuning": {
            "enabled": bool(
                tuning_method == "llm"
                and llm_tuning
                and llm_tuning.get("enabled", False)
            ),
            "method": tuning_method,
            "history_artifact": str(run_dir / "llm-tuning" / "llm_tuning_history.json")
            if llm_summary_path is not None
            else None,
            "summary_artifact": str(llm_summary_path) if llm_summary_path else None,
            "accepted_rounds": int(
                sum(bool(row.get("accepted")) for row in llm_history_table.to_dicts())
            ) if llm_history_table.height else 0,
        },
        "candidate_comparison_artifact": str(candidate_comparison_json),
        "evaluation_artifacts": {
            "roc_curve": str(run_dir / "roc_curve.csv"),
            "calibration_curve": str(run_dir / "calibration_curve.csv"),
            "bootstrap_intervals": str(run_dir / "bootstrap_intervals.csv"),
            "seed_stability": str(run_dir / "seed_stability.csv"),
            "group_performance": str(run_dir / "group_performance.csv"),
            "feature_importance_by_split": str(run_dir / "feature_importance_by_split.csv"),
        },
        "report_path": str(report_output_dir / "model_report.xlsx"),
        "html_report_path": str(report_output_dir / "model_report.html") if generate_report else None,
        "model_artifacts": model_artifacts,
        "warning": "；".join(review_messages)
        if review_messages
        else "自动审查未发现阈值型异常，仍需人工独立验证。",
    }
    (run_dir / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "title": "LightGBM 风控模型报告",
        "model_review": {
            "status": model_review["status"],
            "finding_count": len(model_review["findings"]),
            "blocker_count": sum(
                finding["severity"] == "blocker" for finding in model_review["findings"]
            ),
            "warning_count": sum(
                finding["severity"] == "warning" for finding in model_review["findings"]
            ),
            "recommendation_count": len(model_review["recommendations"]),
        },
        "tables": {
            "1.模型概览": {
                "headers": ["指标", "值"],
                "rows": [
                    ["模型类型", summary["model_type"]],
                    ["训练模式", summary["training_mode"]],
                    ["最终采用模型", summary["selected_candidate"]],
                    ["模型接受指标", summary["acceptance_metric"]],
                    ["最低提升要求", summary["acceptance_min_improvement"]],
                    ["相对基线提升", summary["acceptance_improvement_vs_baseline"]],
                    ["模型选择原因", summary["acceptance_reason"]],
                    [
                        "调参目标",
                        config.tuning.objective_metric if config.training.mode == "tuning" else "未调参",
                    ],
                    ["早停指标", final_model_config.early_stopping_metric],
                    ["类别不平衡处理", summary["class_imbalance_action"]],
                    ["最佳迭代轮数", summary["best_iteration"]],
                    ["最佳调参 Trial", summary["best_trial_number"]],
                    ["Optuna 验证策略", summary["tuning_cv_strategy"] or "未调参"],
                    ["Optuna 验证折数", summary["tuning_cv_folds"] or "未调参"],
                    ["滚动验证窗口", summary["tuning_validation_months"] or "未调参"],
                    ["训练-验证间隔", summary["tuning_gap_months"] if summary["tuning_gap_months"] is not None else "未调参"],
                    ["最佳参数", json.dumps(summary["final_model_parameters"], ensure_ascii=False)],
                    ["入模特征数", summary["selected_feature_count"]],
                    ["类别特征数", summary["categorical_feature_count"]],
                    ["切分策略", summary["split_strategy"]],
                    ["Test 月份", ", ".join(summary["test_month_values"]) or "随机抽样"],
                    ["OOT 月份", ", ".join(summary["oot_month_values"])],
                    ["提示", summary["warning"]],
                ],
            },
            "2.数据切分": _payload_table(split.summary),
            "3.特征筛选": _payload_table(selection.decisions),
            "4.特征重要性": _payload_table(importance),
            "5.分箱明细": _payload_table(
                selection.binning_detail.filter(
                    pl.col("feature").is_in(list(selection.feature_cols))
                )
            ),
            "5.训练过程": _payload_table(training.training_history),
            "6.效果指标": _payload_table(metrics),
            "7.Lift明细": _payload_table(lift_detail),
            "8.月度表现": _payload_table(monthly_performance),
            "10.候选模型": _payload_table(candidate_comparison),
            "9.模型稳定性": _payload_table(stability_rows),
            "15.入模变量信息": _payload_table(variable_information),
            "16.ROC曲线": _payload_table(curve_roc),
            "17.KS曲线": _payload_table(curve_ks),
            "18.校准曲线": _payload_table(calibration),
            "19.Bootstrap置信区间": _payload_table(bootstrap_intervals),
            "20.随机种子稳定性": _payload_table(seed_stability),
            "21.分群效果": _payload_table(group_performance),
            "22.跨数据集特征重要性": _payload_table(importance_by_split),
            "10.LLM调参迭代": _payload_table(llm_history_table),
            "11.调参试验": _payload_table(
                tuning_result.trials
                if tuning_result is not None
                else pl.DataFrame(
                    {
                        "trial_number": [],
                        "state": [],
                        "objective_value": [],
                    },
                    schema={
                        "trial_number": pl.Int64,
                        "state": pl.String,
                        "objective_value": pl.Float64,
                    },
                )
            ),
            "12.CV折表现": _payload_table(
                tuning_result.fold_metrics
                if tuning_result is not None
                else pl.DataFrame(
                    {"trial_number": [], "fold": []},
                    schema={"trial_number": pl.Int64, "fold": pl.Int64},
                )
            ),
            "13.AI模型审查": {
                "headers": [
                    "code",
                    "severity",
                    "message",
                    "evidence",
                    "recommendation",
                ],
                "rows": [
                    [
                        finding["code"],
                        finding["severity"],
                        finding["message"],
                        json.dumps(finding.get("evidence", {}), ensure_ascii=False),
                        finding.get("recommendation"),
                    ]
                    for finding in model_review["findings"]
                ],
            },
            "13.诊断建议": {
                "headers": ["类别", "优先级", "来源编码", "改进建议"],
                "rows": [
                    [
                        item["category"],
                        item["priority"],
                        ", ".join(item["source_codes"]),
                        item["action"],
                    ]
                    for item in model_review["recommendations"]
                ],
            },
            "14.特征预处理": _payload_table(preprocessing_plan.decisions),
        },
    }
    payload_path = run_dir / "model_report_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report_path = report_output_dir / "model_report.xlsx"
    html_report_path = report_output_dir / "model_report.html"
    if generate_report:
        report_stage = (
            progress.track(
                "model-report",
                running_summary="正在生成模型 Excel 报告",
                success_summary="建模报告已生成",
                artifacts=[report_path, html_report_path],
            )
            if progress
            else nullcontext()
        )
        with report_stage:
            write_model_report(payload_path, report_path)
            write_model_html_report(payload_path, html_report_path)
    elif progress:
        progress.emit("model-report", "skipped", summary="未选择报告交付节点")
    logger.info("Model pipeline completed: %s", report_path)
    return ModelRunResult(
        run_dir, report_path, selection.feature_cols, metrics, split.summary, html_report_path if generate_report else None
    )

"""End-to-end LightGBM modeling workflow and model-report export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from contextlib import nullcontext
import json
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

from .config import ModelConfig
from .evaluation import evaluate_binary_model, population_stability_index
from .export import export_model_artifacts
from .feature_selection import select_features
from .lgbm_model import LightGbmTrainingResult, predict_bad_probability, train_lightgbm
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
            }
        )
    return rows


def _candidate_row(
    name: str,
    training: LightGbmTrainingResult,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: tuple[str, ...],
    target_col: str,
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
    train_auc = float(train_eval.auc) if train_eval.auc is not None else None
    test_auc = float(test_eval.auc) if test_eval.auc is not None else None
    return {
        "candidate": name,
        "best_iteration": training.best_iteration,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "test_ks": test_eval.ks,
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
    final_model_config = config.model
    if config.training.mode == "tuning":
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
            )
        final_model_config = tuning_result.best_config
    elif progress:
        progress.emit(
            "tuning", "skipped", summary="当前配置使用 baseline 模式，跳过自动调参"
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
        training = (
            train_lightgbm(
                split.train,
                split.test,
                cleaned_contract,
                selection.feature_cols,
                final_model_config,
                seed=config.split.random_seed,
                balance_classes=balance_classes,
                categorical_features=categorical_features,
            )
            if tuning_result is not None
            else baseline_training
        )
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

    candidate_rows = [
        _candidate_row(
            "baseline",
            baseline_training,
            split.train,
            split.test,
            selection.feature_cols,
            target_col,
        )
    ]
    if tuning_result is not None:
        candidate_rows.append(
            _candidate_row(
                "optuna_tuned",
                training,
                split.train,
                split.test,
                selection.feature_cols,
                target_col,
            )
        )
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
        ]
    )
    monthly_performance = pl.DataFrame(monthly_rows).sort(["dataset", "event_month"])
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
    _write_csv(training.feature_importance, run_dir, "feature_importance")
    _write_csv(metrics, run_dir, "metrics_by_split")
    _write_csv(lift_detail, run_dir, "lift_detail")
    _write_csv(monthly_performance, run_dir, "monthly_performance")
    _write_csv(candidate_comparison, run_dir, "candidate_comparison")
    if config_path is not None:
        shutil.copy2(Path(config_path), run_dir / "model_config.yaml")

    review_messages = [finding["message"] for finding in model_review["findings"]]
    summary = {
        "model_type": "LightGBM",
        "training_mode": config.training.mode,
        "class_imbalance_action": "class_weight" if balance_classes else "none",
        "best_iteration": training.best_iteration,
        "best_trial_number": tuning_result.best_trial_number if tuning_result else None,
        "best_cv_objective": tuning_result.best_objective_value
        if tuning_result
        else None,
        "final_model_parameters": asdict(final_model_config),
        "selected_feature_count": len(selection.feature_cols),
        "selected_features": list(selection.feature_cols),
        "categorical_feature_count": len(categorical_features),
        "categorical_features": list(categorical_features),
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
                    [
                        "调参目标",
                        config.tuning.objective_metric if config.training.mode == "tuning" else "未调参",
                    ],
                    ["早停指标", final_model_config.early_stopping_metric],
                    ["类别不平衡处理", summary["class_imbalance_action"]],
                    ["最佳迭代轮数", summary["best_iteration"]],
                    ["最佳调参 Trial", summary["best_trial_number"]],
                    ["入模特征数", summary["selected_feature_count"]],
                    ["类别特征数", summary["categorical_feature_count"]],
                    ["OOT 月份", ", ".join(summary["oot_month_values"])],
                    ["提示", summary["warning"]],
                ],
            },
            "2.数据切分": _payload_table(split.summary),
            "3.特征筛选": _payload_table(selection.decisions),
            "4.特征重要性": _payload_table(training.feature_importance),
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
    if generate_report:
        report_stage = (
            progress.track(
                "model-report",
                running_summary="正在生成模型 Excel 报告",
                success_summary="建模报告已生成",
                artifacts=[report_path],
            )
            if progress
            else nullcontext()
        )
        with report_stage:
            write_model_report(payload_path, report_path)
    elif progress:
        progress.emit("model-report", "skipped", summary="未选择报告交付节点")
    logger.info("Model pipeline completed: %s", report_path)
    return ModelRunResult(
        run_dir, report_path, selection.feature_cols, metrics, split.summary
    )

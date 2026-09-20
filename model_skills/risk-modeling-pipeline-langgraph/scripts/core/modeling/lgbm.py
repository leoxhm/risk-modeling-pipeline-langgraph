"""独立的 LightGBM 训练和 Train/Validate/OOT 评估实现。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from core.data_read.contract import DataContract
from .evaluation import auc_ks, classification_curves, decile_analysis, pr_auc, score_distribution, score_psi


@dataclass(frozen=True)
class TrainingResult:
    """保存一次 LightGBM 拟合、最终指标和逐 boosting 轮次历史。"""

    model: Any
    parameters: dict[str, Any]
    # 每个分区除了标量指标，还包含曲线点、十分位明细、分数分布和月度指标。
    metrics: dict[str, dict[str, Any]]
    fit_history: dict[str, Any]
    best_iteration: int


def _month_key(value: Any) -> str | None:
    """把日期字段转换成 YYYYMM，用于复用 sample-split 清单。"""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y%m")
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return digits[:6] if len(digits) >= 6 else None


def split_from_manifest(frame: pl.DataFrame, contract: DataContract, manifest: dict[str, Any]) -> dict[str, pl.DataFrame]:
    """按已确认的月份清单从缓存数据帧切出 Train/Validate/OOT。"""
    if not contract.date_col or contract.date_col not in frame.columns:
        raise ValueError("模型训练需要已确认的日期字段")
    months = [_month_key(value) for value in frame.get_column(contract.date_col).to_list()]
    result: dict[str, pl.DataFrame] = {}
    for name, key in (("train", "train_months"), ("validate", "validate_months"), ("oot", "oot_months")):
        allowed = set(str(value) for value in manifest.get(key, []))
        mask = [month in allowed for month in months]
        result[name] = frame.filter(pl.Series(f"_{name}", mask))
    return result


def numeric_features(frame: pl.DataFrame, features: list[str]) -> tuple[str, ...]:
    """只保留可直接输入当前 LightGBM 训练器的数值字段。"""
    return tuple(feature for feature in features if feature in frame.columns and frame.schema[feature].is_numeric() and frame.schema[feature] != pl.Boolean)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """构造数值矩阵并把无穷值统一为 NaN，交给 LightGBM 原生处理。"""
    if not features:
        raise ValueError("没有可用于 LightGBM 的数值特征")
    values = frame.select(list(features)).cast(pl.Float64).to_numpy()
    return np.where(np.isfinite(values), values, np.nan).astype(float, copy=False)


def _dataset_metrics(
    frame: pl.DataFrame,
    model: Any,
    features: tuple[str, ...],
    target_col: str,
    bad_label: Any,
    best_iteration: int,
    reference_scores: np.ndarray | None = None,
    date_col: str | None = None,
) -> dict[str, Any]:
    """计算一个分区的 AUC、PR-AUC、KS、Gini 和 Score PSI。"""
    if not frame.height:
        return {"auc": None, "pr_auc": None, "gini": None, "ks": None, "bad_rate": None, "sample_count": 0, "score_unique_count": 0, "score_psi_vs_train": None, "roc_curve": [], "pr_curve": [], "deciles": [], "score_distribution": [], "monthly": []}
    scores = model.predict_proba(_matrix(frame, features), num_iteration=best_iteration)[:, 1]
    labels = [int(value == bad_label) for value in frame.get_column(target_col).to_list()]
    auc, ks = auc_ks(labels, scores.tolist())
    curves = classification_curves(labels, scores.tolist())
    monthly: list[dict[str, Any]] = []
    if date_col and date_col in frame.columns:
        month_values: dict[str, list[int]] = {}
        month_scores: dict[str, list[float]] = {}
        for value, label, score in zip(frame.get_column(date_col).to_list(), labels, scores.tolist()):
            month = _month_key(value) or "UNKNOWN"
            month_values.setdefault(month, []).append(label)
            month_scores.setdefault(month, []).append(float(score))
        for month in sorted(month_values):
            month_labels = month_values[month]
            month_score_values = month_scores[month]
            month_auc, month_ks = auc_ks(month_labels, month_score_values)
            monthly.append(
                {
                    "month": month,
                    "sample_count": len(month_labels),
                    "bad_count": sum(month_labels),
                    "bad_rate": sum(month_labels) / len(month_labels) if month_labels else None,
                    "auc": month_auc,
                    "ks": month_ks,
                    "pr_auc": pr_auc(month_labels, month_score_values),
                }
            )
    return {
        "auc": auc,
        "pr_auc": pr_auc(labels, scores.tolist()),
        "gini": (2.0 * auc - 1.0) if auc is not None else None,
        "ks": ks,
        "bad_rate": sum(labels) / len(labels) if labels else None,
        "sample_count": frame.height,
        # 用于识别“模型没有产生有效排序”的退化 trial；分数全相同时 KS 必须为 0。
        "score_unique_count": int(np.unique(np.round(scores, 12)).size),
        "score_psi_vs_train": score_psi(reference_scores, scores) if reference_scores is not None else None,
        "roc_curve": curves["roc_curve"],
        "pr_curve": curves["pr_curve"],
        "deciles": decile_analysis(labels, scores.tolist()),
        "score_distribution": score_distribution(labels, scores.tolist()),
        "monthly": monthly,
    }


def train_lightgbm(train: pl.DataFrame, validate: pl.DataFrame, oot: pl.DataFrame, contract: DataContract, features: list[str], parameters: dict[str, Any]) -> TrainingResult:
    """用一组参数训练 LightGBM，并返回拟合历史与三分区指标。"""
    target_col = contract.target_col
    if not target_col or target_col not in train.columns:
        raise ValueError("目标字段未确认，无法训练 LightGBM")
    usable = numeric_features(train, features)
    if not usable:
        raise ValueError("特征筛选后没有可用于 LightGBM 的数值字段")
    params = {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_split_gain": 0.0,
        "n_estimators": 300,
        "random_state": 42,
        "verbosity": -1,
    }
    params.update(parameters or {})
    model = lgb.LGBMClassifier(objective="binary", **params)
    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks = [lgb.record_evaluation(evals_result), lgb.log_evaluation(0)]
    if validate.height:
        callbacks.append(lgb.early_stopping(50, verbose=False))
    train_labels = np.asarray([int(value == contract.bad_label) for value in train.get_column(target_col).to_list()], dtype=np.int32)
    validate_labels = np.asarray([int(value == contract.bad_label) for value in validate.get_column(target_col).to_list()], dtype=np.int32)
    train_matrix = _matrix(train, usable)
    validate_matrix = _matrix(validate, usable) if validate.height else None
    # LightGBM 4.7 推荐 eval_X/eval_y，避免旧版 eval_set 的弃用警告。
    model.fit(
        train_matrix,
        train_labels,
        eval_X=(train_matrix, validate_matrix) if validate.height else (train_matrix,),
        eval_y=(train_labels, validate_labels) if validate.height else (train_labels,),
        callbacks=callbacks,
    )
    # 新接口默认使用 valid_0/valid_1；统一改成监控页面和模型报告使用的名称。
    if "valid_0" in evals_result:
        evals_result["train"] = evals_result.pop("valid_0")
    if "valid_1" in evals_result:
        evals_result["validate"] = evals_result.pop("valid_1")
    best_iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
    train_scores = model.predict_proba(_matrix(train, usable), num_iteration=best_iteration)[:, 1] if train.height else np.asarray([], dtype=float)
    metrics = {
        "train": _dataset_metrics(train, model, usable, target_col, contract.bad_label, best_iteration, reference_scores=None, date_col=contract.date_col),
        "validate": _dataset_metrics(validate, model, usable, target_col, contract.bad_label, best_iteration, reference_scores=train_scores, date_col=contract.date_col),
        "oot": _dataset_metrics(oot, model, usable, target_col, contract.bad_label, best_iteration, reference_scores=train_scores, date_col=contract.date_col),
    }
    metrics["train"]["score_psi_vs_train"] = 0.0
    return TrainingResult(model=model, parameters={**params, "features": list(usable)}, metrics=metrics, fit_history=evals_result, best_iteration=best_iteration)

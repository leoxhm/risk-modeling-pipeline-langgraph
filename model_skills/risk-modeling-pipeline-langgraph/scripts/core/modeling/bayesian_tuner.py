"""使用 Optuna/TPE 对 LightGBM 进行可持久化的贝叶斯优化。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import optuna
import polars as pl

from core.data_read.contract import DataContract

from .lgbm import TrainingResult, train_lightgbm
from .monitor import TrainingMonitor
from .evaluation import model_quality_components, model_quality_score


_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.20),
    "num_leaves": (7, 128),
    "max_depth": (-1, 12),
    "min_child_samples": (10, 500),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (0.0, 50.0),
    "reg_lambda": (0.0, 50.0),
    "min_split_gain": (0.0, 2.0),
    "n_estimators": (50, 1000),
}
_INTEGER_FIELDS = {"num_leaves", "max_depth", "min_child_samples", "n_estimators"}


def _bounds(settings: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """合并默认搜索边界和 YAML 中的合法边界。"""
    result = dict(_DEFAULT_BOUNDS)
    raw = settings.get("bounds") or {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            if name not in result or not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            try:
                low, high = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                continue
            if low <= high:
                result[str(name)] = (low, high)
    return result


def _sample_parameters(trial: optuna.Trial, bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """从每个字段的硬边界中采样一组可直接传给 LightGBM 的参数。"""
    values: dict[str, Any] = {}
    for name, (low, high) in bounds.items():
        if name in _INTEGER_FIELDS:
            values[name] = trial.suggest_int(name, int(low), int(high))
        else:
            values[name] = trial.suggest_float(name, low, high)
    max_depth = int(values.get("max_depth", -1))
    if max_depth > 0:
        values["num_leaves"] = min(int(values["num_leaves"]), 2**max_depth)
    return values


def _objective_value(metrics: dict[str, Any], metric: str) -> float | None:
    """从分区指标中提取 Optuna 目标；默认使用综合模型质量分。"""
    if metric in {"quality_score", "model_quality_score", "model_quality"}:
        return model_quality_score(metrics)
    if "_" not in metric:
        return None
    dataset, name = metric.split("_", 1)
    value = (metrics.get(dataset) or {}).get(name)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metrics_with_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    """把综合质量分和差距组成项写进监控事件，便于面板解释每轮取舍。"""
    result = dict(metrics)
    result["model_quality_score"] = model_quality_score(metrics)
    result["quality_components"] = model_quality_components(metrics)
    return result


def _is_degenerate_candidate(metrics: dict[str, Any], dataset: str = "validate") -> bool:
    """识别没有产生有效排序的模型（例如所有预测概率完全相同）。"""
    values = metrics.get(dataset) or {}
    try:
        unique_count = int(values.get("score_unique_count", 2))
    except (TypeError, ValueError):
        unique_count = 2
    return unique_count <= 1


def run_bayesian_tuning(
    *,
    train: pl.DataFrame,
    validate: pl.DataFrame,
    oot: pl.DataFrame,
    contract: DataContract,
    features: list[str],
    baseline_parameters: dict[str, Any],
    settings: dict[str, Any],
    monitor: TrainingMonitor,
    model_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行 baseline + Optuna trials，并保存每个 trial 的真实训练结果。"""
    max_trials = max(1, int(settings.get("max_trials", 50)))
    metric = str(settings.get("metric", "quality_score")).strip().lower()
    # 未知指标回退到综合质量分，避免所有 trial 因目标为空而被静默标记为 pruned。
    if metric not in {"quality_score", "model_quality_score", "model_quality", "validate_ks", "validate_auc"}:
        metric = "quality_score"
    direction = str(settings.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        direction = "maximize"
    bounds = _bounds(settings)

    baseline = train_lightgbm(train, validate, oot, contract, features, baseline_parameters)
    baseline_score = _objective_value(baseline.metrics, metric)
    monitor.record_iteration(
        iteration=0,
        parameters=baseline.parameters,
        metrics=_metrics_with_quality(baseline.metrics),
        fit_history=baseline.fit_history,
        accepted=True,
        reason="baseline",
    )
    best = baseline
    best_score = baseline_score

    sampler_name = str(settings.get("sampler", "tpe")).lower()
    # 详细 trial 已持久化到 JSONL，终端不再重复打印 Optuna 的每轮 INFO。
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler_name == "random":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=int(settings.get("random_seed", 42)))
    else:
        sampler = optuna.samplers.TPESampler(seed=int(settings.get("random_seed", 42)), multivariate=True)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=f"risk-modeling-{monitor.run_id}")
    history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        """采样、训练、评估并把当前 trial 追加到监控日志。"""
        nonlocal best, best_score
        proposal = _sample_parameters(trial, bounds)
        try:
            candidate = train_lightgbm(train, validate, oot, contract, features, {**baseline_parameters, **proposal})
            score = _objective_value(candidate.metrics, metric)
            if score is None:
                raise optuna.TrialPruned(f"目标指标 {metric} 不可计算")
            if _is_degenerate_candidate(candidate.metrics):
                # 记录退化 trial，但不允许它以错误的 KS=1 抢走最佳模型。
                reason = "退化 trial：Validate 预测分数只有一个取值，已排除"
                logged_metrics = _metrics_with_quality(candidate.metrics)
                monitor.record_iteration(iteration=trial.number + 1, parameters=candidate.parameters, metrics=logged_metrics, fit_history=candidate.fit_history, status="degenerate", accepted=False, reason=reason)
                history.append({"trial": trial.number, "status": "degenerate", "parameters": candidate.parameters, "metrics": logged_metrics, "objective": None, "accepted": False, "reason": reason})
                trial.set_user_attr("metrics", candidate.metrics)
                trial.set_user_attr("accepted", False)
                return -1.0 if direction == "maximize" else 1.0
            improved = best_score is None or (score > best_score if direction == "maximize" else score < best_score)
            reason = f"{metric} {'刷新最佳' if improved else '未超过当前最佳'}"
            monitor.record_iteration(
                iteration=trial.number + 1,
                parameters=candidate.parameters,
                metrics=_metrics_with_quality(candidate.metrics),
                fit_history=candidate.fit_history,
                accepted=improved,
                reason=reason,
            )
            history.append({
                "trial": trial.number,
                "status": "accepted" if improved else "complete",
                "parameters": candidate.parameters,
                "metrics": _metrics_with_quality(candidate.metrics),
                "objective": score,
                "accepted": improved,
                "reason": reason,
            })
            trial.set_user_attr("metrics", candidate.metrics)
            trial.set_user_attr("accepted", improved)
            if improved:
                best, best_score = candidate, score
            # Optuna 已按 direction 比较目标值，这里直接返回原始指标。
            return score
        except optuna.TrialPruned:
            monitor.record_iteration(iteration=trial.number + 1, parameters=proposal, metrics={}, status="pruned", accepted=False, reason=f"目标指标 {metric} 不可计算")
            history.append({"trial": trial.number, "status": "pruned", "parameters": proposal, "metrics": {}, "accepted": False, "reason": f"目标指标 {metric} 不可计算"})
            raise
        except Exception as exc:  # 单个 trial 失败不应中断整个搜索
            reason = f"训练失败：{type(exc).__name__}: {exc}"
            monitor.record_iteration(iteration=trial.number + 1, parameters=proposal, metrics={}, status="failed", accepted=False, reason=reason)
            history.append({"trial": trial.number, "status": "failed", "parameters": proposal, "metrics": {}, "accepted": False, "reason": reason})
            raise optuna.TrialPruned(reason)

    study.optimize(objective, n_trials=max_trials, catch=(Exception,))
    monitor.finish(status="completed", best_iteration=0 if best_score is None else max((int(item.get("trial", -1)) + 1 for item in history if item.get("accepted")), default=0))

    model_path: str | None = None
    if model_output_path is not None:
        destination = Path(model_output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(best.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_path = str(destination)

    # Optuna 的 study 不包含 baseline；只有真正刷新 baseline 的 trial 才能称为最佳 trial。
    accepted_trials = [item for item in history if item.get("accepted") is True]
    best_trial_number = accepted_trials[-1].get("trial") if accepted_trials else None
    return {
        "status": "success",
        "method": "bayesian",
        "metric": metric,
        "direction": direction,
        "requested_trials": max_trials,
        "completed_trials": len(history),
        "best_trial": best_trial_number,
        "best_source": "trial" if best_trial_number is not None else "baseline",
        "best_parameters": best.parameters,
        "best_metrics": _metrics_with_quality(best.metrics),
        "history": history,
        "monitor": str(monitor.events_path),
        "model_path": model_path,
    }

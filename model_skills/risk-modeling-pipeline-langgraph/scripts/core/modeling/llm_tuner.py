"""LightGBM 的 LLM 迭代调参闭环：请求 JSON 参数、校验、重训并记录结果。"""

from __future__ import annotations

import json
import os
import pickle
import re
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import polars as pl

from core.data_read.contract import DataContract
from .lgbm import TrainingResult, train_lightgbm
from .monitor import TrainingMonitor
from .evaluation import model_quality_components, model_quality_score


_TUNABLE_FIELDS = {"learning_rate", "num_leaves", "max_depth", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain", "n_estimators"}
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {"learning_rate": (0.005, 0.20), "num_leaves": (7, 128), "max_depth": (-1, 12), "min_child_samples": (10, 500), "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0), "reg_alpha": (0.0, 50.0), "reg_lambda": (0.0, 50.0), "min_split_gain": (0.0, 2.0), "n_estimators": (50, 1000)}


def _objective_value(metrics: dict[str, Any], metric: str) -> float | None:
    """读取 LLM 调参目标；默认使用 Train/Validate/OOT 综合质量分。"""
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
    """将质量分和差距拆解项写入监控日志，供前端直接解释。"""
    result = dict(metrics)
    result["model_quality_score"] = model_quality_score(metrics)
    result["quality_components"] = model_quality_components(metrics)
    return result


def _is_degenerate_candidate(metrics: dict[str, Any], dataset: str = "validate") -> bool:
    """识别所有预测概率相同的退化候选，避免其产生虚假的排序指标。"""
    values = metrics.get(dataset) or {}
    try:
        return int(values.get("score_unique_count", 2)) <= 1
    except (TypeError, ValueError):
        return False


def _parse_json(text: str) -> dict[str, Any]:
    """解析纯 JSON 或被 Markdown 代码块包裹的 JSON。"""
    candidate = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S | re.I)
    if match:
        candidate = match.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("LLM 返回必须是 JSON 对象")
    return value


def _bounded_patch(current: dict[str, Any], proposal: dict[str, Any], bounds: dict[str, tuple[float, float]]) -> tuple[dict[str, Any] | None, str | None]:
    """只接受白名单和硬边界内的参数，不自动替 LLM 修正非法值。"""
    unknown = sorted(set(proposal) - _TUNABLE_FIELDS)
    if unknown:
        return None, f"包含不允许调整的参数：{', '.join(unknown)}"
    result = dict(current)
    integer_fields = {"num_leaves", "max_depth", "min_child_samples", "n_estimators"}
    for name, raw in proposal.items():
        try:
            value = int(raw) if name in integer_fields else float(raw)
        except (TypeError, ValueError):
            return None, f"{name} 不是有效数值"
        low, high = bounds.get(name, _DEFAULT_BOUNDS[name])
        if value < low or value > high:
            return None, f"{name}={value} 超出硬边界 [{low}, {high}]"
        result[name] = value
    if result.get("max_depth", -1) > 0 and result.get("num_leaves", 31) > 2 ** int(result["max_depth"]):
        return None, "num_leaves 不能大于 2**max_depth"
    return result, None


def _request_proposal(*, current: dict[str, Any], metrics: dict[str, Any], history: list[dict[str, Any]], round_number: int, settings: dict[str, Any], bounds: dict[str, tuple[float, float]]) -> tuple[dict[str, Any] | None, str | None]:
    """调用 OpenAI 兼容接口，并要求只返回 parameters/reason JSON。"""
    api_key = settings.get("api_key") or os.getenv(str(settings.get("api_key_env", "LLM_TUNING_API_KEY")))
    base_url = str(settings.get("base_url") or os.getenv("LLM_TUNING_BASE_URL", "")).rstrip("/")
    model = settings.get("model") or os.getenv("LLM_TUNING_MODEL")
    if not api_key or not base_url or not model:
        return None, "未配置 api_key、base_url 或 model；请在 llm-optimization.yaml 或环境变量中配置"
    prompt = {
        "round": round_number,
        "current_parameters": current,
        "metrics": metrics,
        "previous_rounds": history[-8:],
        "hard_bounds": bounds,
        "objective": "maximize model_quality_score：优先 OOT KS，同时要求 Train/Validate/OOT KS 接近，避免过拟合和时间退化",
        "output_format": {"parameters": {"learning_rate": 0.03}, "reason": "简短中文原因"},
    }
    request_payload = {"model": model, "temperature": float(settings.get("temperature", 0.0)), "messages": [{"role": "system", "content": "你是风控 LightGBM 调参顾问，只输出合法 JSON，不要 Markdown。"}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]}
    # Qwen3 系列可关闭思考链，避免每轮请求等待很久；其他兼容服务可在 YAML 中保持 null。
    if settings.get("enable_thinking") is not None:
        request_payload["enable_thinking"] = bool(settings.get("enable_thinking"))
    elif str(model).lower().startswith("qwen3"):
        # 兼容已有模板：Qwen3 默认关闭思考链，避免旧配置仍然触发长响应。
        request_payload["enable_thinking"] = False
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = Request(f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
    verify_ssl = settings.get("verify_ssl", True)
    ca_bundle = settings.get("ca_bundle") or os.getenv("LLM_TUNING_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    context = ssl._create_unverified_context() if verify_ssl is False else ssl.create_default_context(cafile=str(Path(ca_bundle).expanduser()) if ca_bundle else None)
    configured_timeout = max(1, int(settings.get("timeout_seconds", 60)))
    # 防止配置误填 300/600 秒时把整个节点长时间阻塞；需要更长等待可显式提高上限。
    timeout_cap = max(1, int(settings.get("max_timeout_seconds", 90)))
    timeout_seconds = min(configured_timeout, timeout_cap)
    configured_retries = max(0, int(settings.get("max_retries_per_round", 1)))
    retry_cap = max(0, int(settings.get("max_retries_cap", 1)))
    retries = min(configured_retries, retry_cap)
    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"].get("content", "")
            parsed = _parse_json(content)
            proposal = parsed.get("parameters", parsed.get("params"))
            if not isinstance(proposal, dict) or not proposal:
                raise ValueError("LLM 未返回非空 parameters")
            return {"parameters": proposal, "reason": str(parsed.get("reason", "未提供原因"))}, None
        except Exception as exc:  # 网络、证书和解析错误都要进入监控日志
            last_error = exc
    return None, f"LLM 请求失败：{type(last_error).__name__}: {last_error}"


def run_llm_tuning(*, train: pl.DataFrame, validate: pl.DataFrame, oot: pl.DataFrame, contract: DataContract, features: list[str], baseline_parameters: dict[str, Any], settings: dict[str, Any], monitor: TrainingMonitor, model_output_path: str | Path | None = None) -> dict[str, Any]:
    """执行 baseline + N 轮 LLM 提议/重训，并把每轮写入 JSONL。"""
    max_rounds = max(1, int(settings.get("max_rounds", 10)))
    metric = str(settings.get("metric", "quality_score")).strip().lower()
    if metric not in {"quality_score", "model_quality_score", "model_quality", "validate_ks", "validate_auc"}:
        metric = "quality_score"
    bounds = {**_DEFAULT_BOUNDS, **{str(key): (float(value[0]), float(value[1])) for key, value in (settings.get("hard_bounds") or {}).items() if isinstance(value, (list, tuple)) and len(value) == 2}}
    current_parameters = dict(baseline_parameters)
    current = train_lightgbm(train, validate, oot, contract, features, current_parameters)
    history: list[dict[str, Any]] = []
    configuration_error: str | None = None
    last_request_error: str | None = None
    request_failures = 0
    monitor.record_iteration(iteration=0, parameters=current.parameters, metrics=_metrics_with_quality(current.metrics), fit_history=current.fit_history, accepted=True, reason="baseline")
    best = current
    best_score = _objective_value(best.metrics, metric)
    started_at = time.monotonic()
    max_total_seconds = max(0, int(settings.get("max_total_seconds", 0)))
    for round_number in range(1, max_rounds + 1):
        if max_total_seconds and time.monotonic() - started_at >= max_total_seconds:
            reason = f"达到 LLM 调参总时长上限 {max_total_seconds} 秒，已保留当前最佳模型"
            history.append({"round": round_number, "status": "stopped_timeout", "metrics": current.metrics, "current_parameters": current.parameters, "accepted": False, "reason": reason})
            monitor.update_progress(iteration=round_number, status="stopped_timeout", message=reason)
            last_request_error = reason
            break
        monitor.update_progress(iteration=round_number, status="requesting_llm", message=f"正在请求第 {round_number}/{max_rounds} 轮参数")
        proposal, error = _request_proposal(current=current.parameters, metrics=_metrics_with_quality(current.metrics), history=history, round_number=round_number, settings=settings, bounds=bounds)
        if error:
            request_failures += 1
            last_request_error = error
            if "未配置" in error:
                configuration_error = error
            row = {"round": round_number, "status": "skipped", "reason": error, "metrics": _metrics_with_quality(current.metrics), "current_parameters": current.parameters, "accepted": False}
            history.append(row)
            monitor.record_iteration(iteration=round_number, parameters=current.parameters, metrics=_metrics_with_quality(current.metrics), status="skipped", accepted=False, reason=error)
            if "未配置" in error:
                break
            continue
        monitor.update_progress(iteration=round_number, status="training_candidate", message=f"正在训练第 {round_number}/{max_rounds} 轮候选模型")
        candidate_parameters, bounds_error = _bounded_patch(current.parameters, proposal.get("parameters", {}), bounds)
        if bounds_error or candidate_parameters is None:
            row = {"round": round_number, "status": "rejected_bounds", "reason": bounds_error, "metrics": current.metrics, "current_parameters": current.parameters, "proposed_parameters": proposal.get("parameters"), "accepted": False}
            history.append(row)
            monitor.record_iteration(iteration=round_number, parameters=current.parameters, metrics=current.metrics, status="rejected_bounds", accepted=False, reason=bounds_error)
            continue
        candidate = train_lightgbm(train, validate, oot, contract, features, candidate_parameters)
        if _is_degenerate_candidate(candidate.metrics):
            reason = "退化候选：Validate 预测分数只有一个取值，已拒绝"
            logged_metrics = _metrics_with_quality(candidate.metrics)
            monitor.record_iteration(iteration=round_number, parameters=candidate.parameters, metrics=logged_metrics, fit_history=candidate.fit_history, status="rejected_degenerate", accepted=False, reason=reason)
            history.append({"round": round_number, "status": "rejected_degenerate", "current_parameters": best.parameters, "proposed_parameters": proposal.get("parameters"), "metrics": logged_metrics, "objective": None, "improvement": None, "accepted": False, "reason": reason})
            continue
        candidate_score = _objective_value(candidate.metrics, metric)
        candidate_oot = (candidate.metrics.get("oot") or {}).get("ks")
        current_oot = (best.metrics.get("oot") or {}).get("ks")
        improvement = None if candidate_score is None or best_score is None else float(candidate_score) - float(best_score)
        oot_regressed = candidate_oot is not None and current_oot is not None and float(candidate_oot) < float(current_oot) - float(settings.get("max_oot_ks_regression", 0.02))
        accepted = improvement is not None and improvement >= float(settings.get("min_improvement", 0.001)) and not oot_regressed
        reason = proposal.get("reason", "") if accepted else (f"{metric} 未达到最小提升或 OOT KS 下降超过阈值")
        logged_metrics = _metrics_with_quality(candidate.metrics)
        monitor.record_iteration(iteration=round_number, parameters=candidate.parameters, metrics=logged_metrics, fit_history=candidate.fit_history, accepted=accepted, reason=reason)
        history.append({"round": round_number, "status": "accepted" if accepted else "rejected_metric", "current_parameters": best.parameters, "proposed_parameters": proposal.get("parameters"), "metrics": logged_metrics, "objective": candidate_score, "improvement": improvement, "accepted": accepted, "reason": reason})
        if accepted:
            best = candidate
            best_score = candidate_score
            # 下一轮必须基于上一轮已接受的参数继续提议，而不是始终回到 baseline。
            current = candidate
    monitor_status = "completed" if not request_failures else "partial"
    monitor.finish(status=monitor_status, best_iteration=0 if best_score is None else max((int(item.get("round", 0)) for item in history if item.get("accepted")), default=0))
    model_path: str | None = None
    if model_output_path is not None:
        destination = Path(model_output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 只保存最终最佳模型，不把模型对象放进 LangGraph checkpoint。
        with destination.open("wb") as handle:
            pickle.dump(best.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_path = str(destination)
    final_status = "needs_configuration" if configuration_error else ("partial" if request_failures else "success")
    return {"status": final_status, "method": "llm", "metric": metric, "best_parameters": best.parameters, "best_metrics": _metrics_with_quality(best.metrics), "history": history, "monitor": str(monitor.events_path), "model_path": model_path, "error": configuration_error or last_request_error}

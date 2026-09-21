"""Legacy bounded LLM recommendations for pre-modeling workflow nodes.

This module is retained for advanced integrations only; the user-facing
workflow no longer calls it. The advisor is deliberately advisory: it receives aggregated EDA evidence,
validates every proposed value against a small allow-list, and writes a
separate YAML draft.  It never changes the user's confirmed configuration or
applies a treatment automatically.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import ssl
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

from logger import get_logger


logger = get_logger(__name__)


def _parse_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _settings(parameters: dict[str, Any]) -> dict[str, Any]:
    ai = parameters.get("ai")
    value = parameters.get("llm")
    if value is None and isinstance(ai, dict):
        value = ai.get("llm", {})
    return value if isinstance(value, dict) else {}


def _ssl_context(settings: dict[str, Any]) -> ssl.SSLContext:
    import ssl as _ssl

    verify_ssl = settings.get("verify_ssl", True)
    ca_bundle = settings.get("ca_bundle") or os.getenv("LLM_TUNING_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if verify_ssl is False:
        logger.warning("节点参数建议已显式关闭 TLS 证书校验；仅建议用于隔离内网测试")
        return _ssl._create_unverified_context()
    if ca_bundle:
        return _ssl.create_default_context(cafile=str(Path(str(ca_bundle)).expanduser()))
    return _ssl.create_default_context()


def _request(settings: dict[str, Any], evidence: dict[str, Any], instructions: str) -> tuple[dict[str, Any] | None, str | None]:
    if not bool(settings.get("enabled", False)):
        return None, "未启用节点参数建议（llm.enabled=false）"
    api_key = settings.get("api_key") or os.getenv(str(settings.get("api_key_env", "LLM_TUNING_API_KEY")))
    base_url = str(settings.get("base_url") or os.getenv("LLM_TUNING_BASE_URL", "")).rstrip("/")
    model = settings.get("model") or os.getenv("LLM_TUNING_MODEL")
    if not api_key or not base_url or not model:
        return None, "未配置 api_key、base_url 或 model，跳过节点参数建议"
    payload = {
        "model": model,
        "temperature": float(settings.get("temperature", 0.0)),
        "messages": [
            {"role": "system", "content": "你是风控建模参数顾问，只输出合法 JSON，不要 markdown。"},
            {"role": "user", "content": json.dumps({"evidence": evidence, "instructions": instructions}, ensure_ascii=False)},
        ],
    }
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=int(settings.get("timeout_seconds", 60)), context=_ssl_context(settings)) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
        return _parse_json(content), None
    except (OSError, URLError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        message = f"LLM 节点参数建议请求失败: {type(exc).__name__}: {exc}"
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            message += "；请设置 llm.ca_bundle，或仅在隔离内网测试中显式设置 verify_ssl: false"
        return None, message


def _number(value: Any, integer: bool = False) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _validate_sample_patch(patch: Any) -> tuple[dict[str, Any], list[str]]:
    allowed = {
        "diagnostics": {
            "imbalance_warning_minority_rate": (0.0001, 0.4999),
            "imbalance_critical_minority_rate": (0.00001, 0.4999),
            "latest_month_min_volume_ratio": (0.01, 1.0),
            "monthly_bad_rate_change_warning": (0.0001, 1.0),
            "high_missing_row_rate": (0.01, 1.0),
        },
        "treatment": {
            "duplicate_action": {"error", "keep_first", "keep_last"},
            "missing_target_action": {"error", "drop"},
            "all_null_feature_action": {"error", "drop"},
            "high_missing_row_action": {"keep", "drop"},
            "incomplete_latest_month_action": {"error", "keep", "exclude"},
            "class_imbalance_action": {"none", "class_weight"},
        },
    }
    if not isinstance(patch, dict):
        return {}, ["建议不是对象，已忽略"]
    result: dict[str, Any] = {}
    errors: list[str] = []
    for section, fields in allowed.items():
        values = patch.get(section, {})
        if not isinstance(values, dict):
            errors.append(f"{section} 必须是对象")
            continue
        valid: dict[str, Any] = {}
        for key, value in values.items():
            if key not in fields:
                errors.append(f"不允许调整参数: {section}.{key}")
                continue
            rule = fields[key]
            if isinstance(rule, set):
                if value not in rule:
                    errors.append(f"{section}.{key}={value} 不在允许值内")
                else:
                    valid[key] = value
            else:
                number = _number(value)
                if number is None or not rule[0] <= float(number) <= rule[1]:
                    errors.append(f"{section}.{key} 超出硬边界 [{rule[0]}, {rule[1]}]")
                else:
                    valid[key] = float(number)
        if valid:
            result[section] = valid
    warning = result.get("diagnostics", {}).get("imbalance_warning_minority_rate")
    critical = result.get("diagnostics", {}).get("imbalance_critical_minority_rate")
    if warning is not None and critical is not None and critical > warning:
        errors.append("imbalance_critical_minority_rate 不能大于 warning")
        result["diagnostics"].pop("imbalance_critical_minority_rate", None)
    return result, errors


def _validate_feature_patch(patch: Any) -> tuple[dict[str, Any], list[str]]:
    allowed_selection = {
        "max_missing_rate": (0.0, 1.0), "min_iv": (0.0, 1.0),
        "max_correlation": (0.0, 1.0), "max_psi": (0.0, 1.0),
        "max_unstable_month_ratio": (0.0, 1.0), "min_month_samples": (1, 10_000_000),
        "max_dominant_rate_warning": (0.0, 1.0),
    }
    allowed_enums = {
        "stability_action": {"review", "drop"},
        "correlation_method": {"pearson", "spearman"},
    }
    allowed_preprocessing = {
        "numeric": {"invalid_to_null", "missing_strategy"},
        "categorical": {"strategy", "rare_min_count", "rare_min_rate", "unknown_action", "near_unique_rate", "max_categories", "high_cardinality_action"},
        "text": {"action", "min_average_length"},
    }
    if not isinstance(patch, dict):
        return {}, ["建议不是对象，已忽略"]
    result: dict[str, Any] = {}
    errors: list[str] = []
    selection = patch.get("selection", {})
    if isinstance(selection, dict):
        valid: dict[str, Any] = {}
        for key, value in selection.items():
            if key in allowed_selection:
                if key == "min_month_samples":
                    try:
                        if isinstance(value, bool) or float(value).is_integer() is False:
                            errors.append(f"selection.{key} 必须是整数")
                            continue
                    except (TypeError, ValueError):
                        errors.append(f"selection.{key} 必须是整数")
                        continue
                number = _number(value, integer=key == "min_month_samples")
                low, high = allowed_selection[key]
                if number is None or not low <= float(number) <= high or (key == "min_month_samples" and float(number).is_integer() is False):
                    errors.append(f"selection.{key} 超出硬边界或类型错误")
                else:
                    valid[key] = int(number) if key == "min_month_samples" else float(number)
            elif key in allowed_enums:
                if value not in allowed_enums[key]:
                    errors.append(f"selection.{key}={value} 不在允许值内")
                else:
                    valid[key] = value
            else:
                errors.append(f"不允许调整参数: selection.{key}")
        if valid:
            result["selection"] = valid
    elif selection:
        errors.append("selection 必须是对象")
    preprocessing = patch.get("preprocessing", {})
    if isinstance(preprocessing, dict):
        valid_pre: dict[str, Any] = {}
        enum_rules = {
            "strategy": {"drop", "lightgbm_native"}, "unknown_action": {"missing", "other"},
            "high_cardinality_action": {"drop", "native"}, "missing_strategy": {"native"}, "action": {"drop"},
        }
        for group, values in preprocessing.items():
            if group not in allowed_preprocessing or not isinstance(values, dict):
                errors.append(f"不允许调整预处理分组: preprocessing.{group}")
                continue
            valid_group: dict[str, Any] = {}
            for key, value in values.items():
                if key not in allowed_preprocessing[group]:
                    errors.append(f"不允许调整参数: preprocessing.{group}.{key}")
                    continue
                if key in enum_rules:
                    if value not in enum_rules[key]:
                        errors.append(f"preprocessing.{group}.{key}={value} 不在允许值内")
                    else:
                        valid_group[key] = value
                else:
                    integer = key in {"rare_min_count", "max_categories", "min_average_length"}
                    if integer:
                        try:
                            if isinstance(value, bool) or float(value).is_integer() is False:
                                errors.append(f"preprocessing.{group}.{key} 必须是整数")
                                continue
                        except (TypeError, ValueError):
                            errors.append(f"preprocessing.{group}.{key} 必须是整数")
                            continue
                    number = _number(value, integer=integer)
                    low, high = (1, 1_000_000) if integer else (0.0, 1.0)
                    if number is None or not low <= float(number) <= high:
                        errors.append(f"preprocessing.{group}.{key} 超出硬边界")
                    else:
                        valid_group[key] = int(number) if integer else float(number)
            if valid_group:
                valid_pre[group] = valid_group
        if valid_pre:
            result["preprocessing"] = valid_pre
    elif preprocessing:
        errors.append("preprocessing 必须是对象")
    return result, errors


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for section, values in patch.items():
        if isinstance(values, dict) and isinstance(result.get(section), dict):
            result[section].update(values)
        else:
            result[section] = values
    return result


def _redact(value: Any) -> Any:
    """Prevent API keys from being copied into review artifacts."""
    if isinstance(value, dict):
        return {key: ("***REDACTED***" if key in {"api_key", "token", "secret"} else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def advise(node_id: str, *, current_parameters: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a validated proposal; all failures are non-blocking."""
    settings = _settings(current_parameters)
    if node_id == "sample-diagnosis":
        instructions = (
            "根据 EDA 汇总和样本诊断证据，建议 diagnostics 阈值和 treatment 策略。"
            "只返回 {proposal:{diagnostics:{},treatment:{}},reason,confidence}；"
            "不要修改数据，不要建议模型训练参数。"
        )
    elif node_id == "feature-processing":
        instructions = (
            "根据 EDA 的缺失率、IV、KS、PSI、相关性和变量类型统计，建议 selection 阈值与 preprocessing。"
            "只返回 {proposal:{selection:{},preprocessing:{}},reason,confidence}；"
            "只可调整给定参数，不要返回字段名或原始数据。"
        )
    else:
        return {"status": "unsupported", "proposal": {}, "reason": f"不支持节点 {node_id}"}
    response, error = _request(settings, evidence, instructions)
    if error:
        return {"status": "disabled" if not settings.get("enabled", False) else "unavailable", "proposal": {}, "reason": error}
    raw_patch = response.get("proposal", response.get("parameters", {})) if response else {}
    patch, validation_errors = (_validate_sample_patch(raw_patch) if node_id == "sample-diagnosis" else _validate_feature_patch(raw_patch))
    if node_id == "sample-diagnosis" and patch:
        merged_diagnostics = _merge(current_parameters.get("diagnostics", {}), patch.get("diagnostics", {}))
        critical = merged_diagnostics.get("imbalance_critical_minority_rate")
        warning = merged_diagnostics.get("imbalance_warning_minority_rate")
        if critical is not None and warning is not None and float(critical) > float(warning):
            validation_errors.append("合并当前配置后 critical 不能大于 warning，已忽略诊断阈值建议")
            patch.pop("diagnostics", None)
    result = {
        "status": "ready" if patch else "rejected",
        "proposal": patch,
        "reason": str((response or {}).get("reason", "未提供原因")),
        "confidence": (response or {}).get("confidence"),
    }
    if validation_errors:
        result["validation_errors"] = validation_errors
    return result


def write_recommendation(path: str | Path, *, node_id: str, config_path: Path, current_parameters: dict[str, Any], advice: dict[str, Any], evidence: dict[str, Any]) -> Path:
    """Write a reviewable YAML draft without mutating the confirmed config."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proposed = _merge(current_parameters, advice.get("proposal", {}))
    body = {
        "node_id": node_id,
        "version": 1,
        "parameters": _redact(proposed),
    }
    content = (
        "# 这是大模型基于 EDA 生成的建议草案。请人工检查后复制需要的参数到原配置，"
        "再执行 --confirm-config；本文件不会自动生效。\n"
        + yaml.safe_dump(body, allow_unicode=True, sort_keys=False)
    )
    destination.write_text(content, encoding="utf-8")
    return destination


__all__ = ["advise", "write_recommendation"]

"""Optional LLM-guided parameter refinement after Optuna.

The deterministic Optuna run remains the source of truth.  This module only
proposes bounded LightGBM parameter patches, retrains them on the frozen
Train/Test split, and accepts a proposal when the configured Test metric
improves by the required amount.  Any provider or parsing failure is recorded
and leaves the current best model untouched.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import ssl
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import polars as pl

from data.contract import ValidatedDataContract
from logger import get_logger
from progress import ProgressReporter

from .config import LightGbmConfig, TuningSearchSpace
from .evaluation import evaluate_binary_model
from .lgbm_model import LightGbmTrainingResult, predict_bad_probability, train_lightgbm


logger = get_logger(__name__)

_TUNABLE_FIELDS = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "min_gain_to_split",
)


def _csv_safe_value(value: Any) -> Any:
    """Convert nested history values to CSV-safe scalar values.

    Polars intentionally rejects list/dict columns when writing CSV.  LLM
    tuning history is kept as structured JSON, but its CSV companion is a
    flat display/export artifact, so nested values must be encoded as JSON
    strings before constructing the DataFrame.  This also covers fields
    added in future (for example ``guardrail_reasons``) instead of relying on
    a hard-coded list of known nested keys.
    """
    if isinstance(value, (dict, list, tuple, set)):
        normalized = list(value) if isinstance(value, set) else value
        return json.dumps(normalized, ensure_ascii=False, default=str)
    return value


def _metric(
    training: LightGbmTrainingResult,
    data: pl.DataFrame,
    features: tuple[str, ...],
    target_col: str,
    metric: str,
) -> float | None:
    scores = predict_bad_probability(training.booster, data, features, training.best_iteration)
    evaluation = evaluate_binary_model(data.get_column(target_col).to_list(), scores.tolist())
    value = evaluation.ks if metric == "ks" else evaluation.auc
    return float(value) if value is not None else None


def _stopping_options(settings: dict[str, Any]) -> dict[str, Any]:
    """Read configurable LLM stopping targets and safety guardrails."""
    stopping = settings.get("stopping") if isinstance(settings.get("stopping"), dict) else {}
    targets = stopping.get("targets") if isinstance(stopping.get("targets"), dict) else {}
    guardrails = stopping.get("guardrails") if isinstance(stopping.get("guardrails"), dict) else {}

    def number(section: dict[str, Any], name: str, default: float) -> float:
        value = section.get(name, settings.get(name, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def integer(section: dict[str, Any], name: str, default: int) -> int:
        value = section.get(name, settings.get(name, default))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    def boolean(section: dict[str, Any], name: str, default: bool) -> bool:
        value = section.get(name, settings.get(name, default))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    return {
        "plateau_rounds": integer(stopping, "plateau_rounds", 2),
        "plateau_min_improvement": number(stopping, "plateau_min_improvement", 0.003),
        "target_tolerance": number(stopping, "target_tolerance", 0.01),
        "targets": {
            "validate_ks": number(targets, "validate_ks", 0.30),
            "validate_auc": number(targets, "validate_auc", 0.75),
            "oot_ks": number(targets, "oot_ks", 0.25),
            "oot_auc": number(targets, "oot_auc", 0.70),
        },
        "guardrails": {
            "max_train_validate_ks_gap": number(guardrails, "max_train_validate_ks_gap", 0.10),
            "allow_gap_improvement": boolean(guardrails, "allow_gap_improvement", True),
            "min_gap_reduction": number(guardrails, "min_gap_reduction", 0.02),
            "max_validate_ks_regression": number(
                guardrails, "max_validate_ks_regression", 0.005
            ),
            "max_oot_ks_regression": number(guardrails, "max_oot_ks_regression", 0.01),
            "max_validate_oot_ks_drop": number(guardrails, "max_validate_oot_ks_drop", 0.08),
            "max_oot_degradation_vs_baseline": number(
                guardrails, "max_oot_degradation_vs_baseline", 0.03
            ),
            "min_oot_bad_count": integer(guardrails, "min_oot_bad_count", 30),
        },
    }


def _provider_options(settings: dict[str, Any]) -> dict[str, float | int]:
    """Return bounded retry settings for transient LLM provider failures."""
    def integer(name: str, default: int, minimum: int = 0) -> int:
        try:
            return max(minimum, int(settings.get(name, default)))
        except (TypeError, ValueError):
            return default

    def number(name: str, default: float, minimum: float = 0.0) -> float:
        try:
            return max(minimum, float(settings.get(name, default)))
        except (TypeError, ValueError):
            return default

    return {
        # Number of additional attempts for one logical LLM round.
        "request_retries": integer("request_retries", 2),
        # A misconfigured/unavailable provider should not cause an endless run.
        "max_consecutive_provider_errors": integer("max_consecutive_provider_errors", 3, 1),
        "retry_backoff_seconds": number("retry_backoff_seconds", 2.0),
    }


def _retryable_provider_error(message: str | None) -> bool:
    """Return whether retrying can plausibly recover the provider failure."""
    if not message:
        return False
    non_retryable = ("未配置 api_key", "LLM TLS 证书配置无效")
    return not any(token in message for token in non_retryable)


def _validate_metric(metrics: dict[str, Any], metric: str) -> float | None:
    """Read the user-facing Validate metric, with a legacy test_* fallback.

    The pipeline historically called the temporal holdout ``test``.  Reports
    and the monitoring UI call the same split ``Validate``.  Keeping this
    alias in one place prevents LLM tuning from accidentally optimizing a
    different dataset than the one shown to the user.
    """
    value = metrics.get(f"validate_{metric}")
    if value is None:
        value = metrics.get(f"test_{metric}")
    return float(value) if value is not None else None


def _history_context(history: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Return a compact, JSON-safe view of prior LLM rounds for the prompt."""
    context: list[dict[str, Any]] = []
    for row in history[-limit:]:
        context.append(
            {
                "round": row.get("round"),
                "status": row.get("status"),
                "current_parameters": row.get("current_parameters"),
                "proposed_parameters": row.get("proposed_parameters"),
                "metrics": row.get("metrics"),
                "improvement_vs_current_best": row.get("improvement_vs_current_best"),
                "improvement_vs_baseline": row.get("improvement_vs_baseline"),
                "oot_improvement_vs_baseline": row.get("oot_improvement_vs_baseline"),
                "rejection_reason": row.get("rejection_reason"),
            }
        )
    return context


def _auto_ca_bundle() -> str | None:
    """Find a common local CA bundle when Python's compiled path is missing."""
    default_cafile = ssl.get_default_verify_paths().cafile
    if default_cafile and Path(default_cafile).expanduser().is_file():
        return None  # Let OpenSSL use its configured default.
    candidates = (
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    )
    return next((path for path in candidates if Path(path).is_file()), None)


def _target_evaluation(
    metrics: dict[str, Any],
    stopping: dict[str, Any],
    oot_bad_count: int,
) -> tuple[bool, bool, str]:
    """Return (target_met, target_close, explanation). OOT is gated by sample size."""
    targets = stopping["targets"]
    min_oot_bad_count = stopping["guardrails"]["min_oot_bad_count"]
    tolerance = stopping["target_tolerance"]
    checks = [
        ("Validate KS", _validate_metric(metrics, "ks"), targets["validate_ks"]),
        ("Validate AUC", _validate_metric(metrics, "auc"), targets["validate_auc"]),
    ]
    oot_eligible = oot_bad_count >= min_oot_bad_count
    if oot_eligible:
        checks.extend(
            [
                ("OOT KS", metrics.get("oot_ks"), targets["oot_ks"]),
                ("OOT AUC", metrics.get("oot_auc"), targets["oot_auc"]),
            ]
        )
    else:
        oot_note = f"OOT 坏样本 {oot_bad_count} < {min_oot_bad_count}，OOT 目标仅作提示"
    missing = [name for name, value, _ in checks if value is None]
    failed = [name for name, value, target in checks if value is not None and float(value) < target]
    close_failed = [
        name for name, value, target in checks if value is not None and float(value) < target - tolerance
    ]
    if missing:
        detail = f"缺少指标：{', '.join(missing)}"
    elif failed:
        detail = f"未达标：{', '.join(failed)}"
    else:
        detail = "全部目标已达成"
    if not oot_eligible:
        detail = f"{detail}；{oot_note}"
    return not missing and not failed, not missing and not close_failed, detail


def _guardrail_reasons(
    metrics: dict[str, Any],
    stopping: dict[str, Any],
    baseline_oot_ks: float | None,
    oot_bad_count: int,
    *,
    allow_gap_improvement: bool = False,
) -> list[str]:
    """Return hard rejection reasons; OOT rules apply only with enough bad cases."""
    reasons: list[str] = []
    guardrails = stopping["guardrails"]
    train_ks = metrics.get("train_ks")
    validate_ks = _validate_metric(metrics, "ks")
    if train_ks is not None and validate_ks is not None:
        gap = float(train_ks) - float(validate_ks)
        if gap > guardrails["max_train_validate_ks_gap"] and not allow_gap_improvement:
            reasons.append(
                f"Train/Validate KS gap={gap:.4f} > {guardrails['max_train_validate_ks_gap']:.4f}"
            )
    if oot_bad_count < guardrails["min_oot_bad_count"]:
        return reasons
    oot_ks = metrics.get("oot_ks")
    if validate_ks is not None and oot_ks is not None:
        drop = float(validate_ks) - float(oot_ks)
        if drop > guardrails["max_validate_oot_ks_drop"]:
            reasons.append(
                f"Validate/OOT KS drop={drop:.4f} > {guardrails['max_validate_oot_ks_drop']:.4f}"
            )
    if baseline_oot_ks is not None and oot_ks is not None:
        degradation = float(baseline_oot_ks) - float(oot_ks)
        if degradation > guardrails["max_oot_degradation_vs_baseline"]:
            reasons.append(
                f"OOT KS 相对 Baseline 下降={degradation:.4f} > {guardrails['max_oot_degradation_vs_baseline']:.4f}"
            )
    return reasons


def _generalization_improvement(
    candidate_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    stopping: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    """Allow a small Validate trade-off when the Train–Validate gap closes materially."""
    guardrails = stopping["guardrails"]
    if not guardrails.get("allow_gap_improvement", True):
        return False, {}
    candidate_gap = candidate_metrics.get("train_validate_ks_gap")
    reference_gap = reference_metrics.get("train_validate_ks_gap")
    candidate_validate = _validate_metric(candidate_metrics, "ks")
    reference_validate = _validate_metric(reference_metrics, "ks")
    if None in {candidate_gap, reference_gap, candidate_validate, reference_validate}:
        return False, {}
    gap_reduction = float(reference_gap) - float(candidate_gap)
    validate_delta = float(candidate_validate) - float(reference_validate)
    if gap_reduction < guardrails["min_gap_reduction"]:
        return False, {"gap_reduction": gap_reduction, "validate_delta": validate_delta}
    if validate_delta < -guardrails["max_validate_ks_regression"]:
        return False, {"gap_reduction": gap_reduction, "validate_delta": validate_delta}
    candidate_oot = candidate_metrics.get("oot_ks")
    reference_oot = reference_metrics.get("oot_ks")
    oot_delta = None
    if candidate_oot is not None and reference_oot is not None:
        oot_delta = float(candidate_oot) - float(reference_oot)
        if oot_delta < -guardrails["max_oot_ks_regression"]:
            return False, {
                "gap_reduction": gap_reduction,
                "validate_delta": validate_delta,
                "oot_delta": oot_delta,
            }
    detail = {"gap_reduction": gap_reduction, "validate_delta": validate_delta}
    if oot_delta is not None:
        detail["oot_delta"] = oot_delta
    return True, detail


def _parse_json(text: str) -> dict[str, Any]:
    """Parse strict JSON or a JSON object wrapped in a markdown fence."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("LLM proposal must be a JSON object")
    return value


def _bounded_config(
    current: LightGbmConfig,
    proposal: dict[str, Any],
    search_space: TuningSearchSpace,
) -> tuple[LightGbmConfig | None, str | None]:
    """Validate a proposal; invalid values are rejected, never clamped."""
    unknown = sorted(set(proposal) - set(_TUNABLE_FIELDS))
    if unknown:
        return None, f"包含不允许调整的参数: {', '.join(unknown)}"
    values = asdict(current)
    bounds = asdict(search_space)
    for name, value in proposal.items():
        if name not in bounds:
            continue
        low, high = bounds[name]
        try:
            if name in {"num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"}:
                if isinstance(value, bool) or int(value) != value:
                    return None, f"{name} 必须是整数"
                value = int(value)
            else:
                value = float(value)
        except (TypeError, ValueError):
            return None, f"{name} 不是有效数值"
        if not low <= value <= high:
            return None, f"{name}={value} 超出硬边界 [{low}, {high}]"
        values[name] = value
    if values["max_depth"] > 0 and values["num_leaves"] > 2 ** values["max_depth"]:
        return None, "num_leaves 不能大于 2**max_depth"
    return LightGbmConfig(**values), None


def _request_proposal(
    *,
    current: LightGbmConfig,
    metrics: dict[str, Any],
    search_space: TuningSearchSpace,
    round_number: int,
    settings: dict[str, Any],
    objective_metric: str,
    min_improvement: float,
    baseline_score: float | None = None,
    baseline_oot_ks: float | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call an OpenAI-compatible endpoint when explicitly configured."""
    import os

    api_key = settings.get("api_key") or None
    api_key_env = str(settings.get("api_key_env", "LLM_TUNING_API_KEY"))
    if not api_key:
        api_key = os.getenv(api_key_env)
    base_url = str(settings.get("base_url") or os.getenv("LLM_TUNING_BASE_URL", "")).rstrip("/")
    model = settings.get("model") or os.getenv("LLM_TUNING_MODEL")
    if not api_key or not base_url or not model:
        return None, "未配置 api_key/LLM_TUNING_API_KEY、base_url 或 model，跳过 LLM 调参"

    # Keep certificate verification enabled by default.  Internal deployments
    # can provide their CA chain explicitly when the Python runtime does not
    # know the organization's root certificate; disabling verification is an
    # explicit last-resort opt-in only.
    verify_ssl = settings.get("verify_ssl", True)
    ca_bundle = settings.get("ca_bundle") or os.getenv("LLM_TUNING_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    try:
        if verify_ssl is False:
            logger.warning("LLM 调参已显式关闭 TLS 证书校验；仅建议用于隔离的内网测试环境")
            ssl_context = ssl._create_unverified_context()
        elif ca_bundle:
            ssl_context = ssl.create_default_context(cafile=str(Path(str(ca_bundle)).expanduser()))
        else:
            detected_ca = _auto_ca_bundle()
            ssl_context = (
                ssl.create_default_context(cafile=detected_ca)
                if detected_ca
                else ssl.create_default_context()
            )
    except (OSError, ssl.SSLError, ValueError) as exc:
        return None, f"LLM TLS 证书配置无效: {type(exc).__name__}: {exc}；请设置 tuning.llm.ca_bundle 指向 CA 文件"
    prompt = {
        "round": round_number,
        "current_parameters": asdict(current),
        "metrics": metrics,
        "current_best_score": _validate_metric(metrics, objective_metric),
        "baseline_validate_score": baseline_score,
        "baseline_oot_ks": baseline_oot_ks,
        "objective_metric": objective_metric,
        "min_improvement_to_accept": min_improvement,
        "oot_used_for_decision": False,
        "stopping_policy": settings.get("stopping", {}),
        "hard_bounds": asdict(search_space),
        "previous_rounds": _history_context(history or []),
        "instructions": (
            "你是风控 LightGBM 调参顾问。只能返回 JSON，不要 markdown。"
            "请基于 current_parameters、metrics 和 previous_rounds 诊断当前模型，再提出小步修改；"
            "优先修复历史轮次中被拒绝的原因，不要重复已经尝试过的参数组合。"
            "新方案至少比当前 Validate 指标提升 min_improvement_to_accept 才有意义。"
            "只允许修改 hard_bounds 中的字段；给出小步、可解释的参数 patch。"
            "格式: {\"parameters\": {字段: 新值}, \"reason\": \"原因\"}。"
        ),
    }
    request_url = f"{base_url}/chat/completions"
    request_body = json.dumps(
        {
            "model": model,
            "temperature": float(settings.get("temperature", 0.0)),
            "messages": [
                {"role": "system", "content": "只输出合法 JSON。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        },
        ensure_ascii=False,
    )
    request = Request(
        request_url,
        data=request_body.encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(
            request,
            timeout=int(settings.get("timeout_seconds", 60)),
            context=ssl_context,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        result = _parse_json(content)
        proposal = result.get("parameters", result.get("params"))
        if not isinstance(proposal, dict) or not proposal:
            raise ValueError("LLM response has no non-empty parameters object")
        return {"parameters": proposal, "reason": str(result.get("reason", "未提供原因"))}, None
    except (OSError, URLError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # macOS curl uses the system Keychain, while the Python OpenSSL build
        # often has an independent/empty CA bundle. If the only failure is
        # certificate verification, retry once through the system curl binary
        # (without -k), preserving normal TLS validation and the same payload.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            try:
                curl = subprocess.run(
                    [
                        "curl",
                        "--silent",
                        "--show-error",
                        "--fail-with-body",
                        "--max-time",
                        str(int(settings.get("timeout_seconds", 60))),
                        request_url,
                        "-H",
                        "Content-Type: application/json",
                        "-H",
                        f"Authorization: Bearer {api_key}",
                        "--data",
                        request_body,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=int(settings.get("timeout_seconds", 60)) + 5,
                    check=False,
                )
                if curl.returncode == 0 and curl.stdout.strip():
                    payload = json.loads(curl.stdout)
                    content = payload["choices"][0]["message"].get("content")
                    result = _parse_json(content)
                    proposal = result.get("parameters", result.get("params"))
                    if isinstance(proposal, dict) and proposal:
                        logger.info("LLM 调参已通过系统 curl 完成请求（Python CA bundle 未包含服务端证书）")
                        return {"parameters": proposal, "reason": str(result.get("reason", "未提供原因"))}, None
                curl_error = (curl.stderr or curl.stdout or "curl 返回空响应").strip()
            except (FileNotFoundError, OSError, subprocess.SubprocessError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as curl_exc:
                curl_error = f"curl 回退失败: {type(curl_exc).__name__}: {curl_exc}"
            message = (
                f"LLM 调参请求失败: {type(exc).__name__}: {exc}；"
                f"系统 curl 回退也未成功: {curl_error}；"
                "请设置 tuning.llm.ca_bundle，或仅在隔离内网测试中显式设置 verify_ssl: false"
            )
            return None, message
        message = f"LLM 调参请求失败: {type(exc).__name__}: {exc}"
        return None, message


def run_llm_tuning(
    train: pl.DataFrame,
    test: pl.DataFrame,
    contract: ValidatedDataContract,
    features: tuple[str, ...],
    current_config: LightGbmConfig,
    current_training: LightGbmTrainingResult,
    search_space: TuningSearchSpace,
    settings: dict[str, Any] | None,
    output_dir: str | Path,
    *,
    oot: pl.DataFrame | None = None,
    baseline_score: float | None = None,
    baseline_oot_ks: float | None = None,
    objective_metric: str,
    min_improvement: float,
    seed: int,
    balance_classes: bool = False,
    categorical_features: tuple[str, ...] = (),
    progress: ProgressReporter | None = None,
) -> tuple[
    LightGbmConfig,
    LightGbmTrainingResult,
    list[dict[str, Any]],
    pl.DataFrame,
    list[tuple[int, LightGbmConfig, LightGbmTrainingResult, dict[str, Any]]],
]:
    """Iterate from the current best config and persist a complete audit trail."""
    options = settings if isinstance(settings, dict) else {}
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    max_rounds = max(0, int(options.get("max_rounds", 10)))
    # Consume the configured budget by default. Target/plateau conditions are
    # still recorded, but do not stop the loop when this switch is enabled.
    run_all_rounds = bool(options.get("run_all_rounds", True))
    stopping = _stopping_options(options)
    provider_options = _provider_options(options)
    target_col = contract.contract.target_col
    best_config, best_training = current_config, current_training
    # ``test`` is the historical name for the independent Validate holdout.
    # It remains frozen throughout LLM tuning and is never replaced by OOT.
    best_score = _metric(best_training, test, features, target_col, objective_metric)
    oot_bad_count = (
        int((oot.get_column(target_col) == contract.contract.bad_label).sum())
        if oot is not None
        else 0
    )
    history: list[dict[str, Any]] = []
    candidate_results: list[tuple[int, LightGbmConfig, LightGbmTrainingResult, dict[str, Any]]] = []
    history_path = run_dir / "llm_tuning_history.json"

    def persist_history() -> None:
        """Keep a recoverable audit trail even if a long run is interrupted."""
        temporary_path = history_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(history_path)

    def emit_round(round_number: int, row: dict[str, Any], status: str) -> None:
        if progress is None:
            return
        metrics = dict(row.get("metrics") or {})
        metrics.setdefault("acceptance_metric", objective_metric)
        if row.get("candidate_score") is not None:
            metrics["candidate_score"] = row["candidate_score"]
        if row.get("improvement") is not None:
            metrics["improvement"] = row["improvement"]
        if row.get("improvement_vs_current_best") is not None:
            metrics["improvement_vs_current_best"] = row["improvement_vs_current_best"]
        if row.get("improvement_vs_baseline") is not None:
            metrics["improvement_vs_baseline"] = row["improvement_vs_baseline"]
        if row.get("oot_improvement_vs_baseline") is not None:
            metrics["oot_improvement_vs_baseline"] = row["oot_improvement_vs_baseline"]
        progress.emit(
            "llm-tuning",
            "running",
            summary=f"LLM 调参第 {round_number}/{max_rounds} 轮：{status}",
            experiment={
                "kind": "llm",
                "phase": "post_tuning",
                "candidate": f"llm_round_{round_number}",
                "round": round_number,
                "max_rounds": max_rounds,
                "status": status,
                "request_attempts": row.get("request_attempts", 1),
                "parameters": row.get("proposed_parameters") or row.get("current_parameters"),
                "metrics": metrics,
                "objective": row.get("candidate_score") if row.get("candidate_score") is not None else row.get("current_score"),
                "accepted": bool(row.get("accepted")),
                "reason": row.get("reason"),
                "rejection_reason": row.get("rejection_reason"),
                "stop_reason": row.get("stop_reason"),
            },
        )

    def training_metrics(training: LightGbmTrainingResult) -> dict[str, Any]:
        current_metrics: dict[str, Any] = {
            "train_auc": _metric(training, train, features, target_col, "auc"),
            "train_ks": _metric(training, train, features, target_col, "ks"),
            "test_ks": _metric(training, test, features, target_col, "ks"),
            "test_auc": _metric(training, test, features, target_col, "auc"),
        }
        # ``test`` is the legacy field name; expose explicit Validate aliases
        # so the prompt, event stream and UI use the same terminology.
        current_metrics["validate_ks"] = current_metrics["test_ks"]
        current_metrics["validate_auc"] = current_metrics["test_auc"]
        if oot is not None:
            current_metrics["oot_ks"] = _metric(training, oot, features, target_col, "ks")
            current_metrics["oot_auc"] = _metric(training, oot, features, target_col, "auc")
        if current_metrics.get("train_ks") is not None and current_metrics.get("test_ks") is not None:
            current_metrics["train_validate_ks_gap"] = (
                current_metrics["train_ks"] - current_metrics["test_ks"]
            )
        return current_metrics

    current_metrics = training_metrics(best_training)
    target_met, target_close, target_detail = _target_evaluation(
        current_metrics, stopping, oot_bad_count
    )
    stop_reason = "max_rounds"
    stagnant_rounds = 0
    consecutive_provider_errors = 0
    if target_met:
        stop_reason = "target_met_before_llm" if not run_all_rounds else "max_rounds"
        row = {
            "round": 0,
            "status": "target_met",
            "objective_metric": objective_metric,
            "metrics": current_metrics,
            "current_score": best_score,
            "current_parameters": asdict(best_config),
            "proposed_parameters": None,
            "reason": f"调参前已达到目标：{target_detail}",
            "candidate_score": best_score,
            "improvement": 0.0,
            "improvement_vs_current_best": 0.0,
            "improvement_vs_baseline": 0.0 if baseline_score is not None else None,
            "oot_improvement_vs_baseline": (
                current_metrics.get("oot_ks") - baseline_oot_ks
                if current_metrics.get("oot_ks") is not None and baseline_oot_ks is not None
                else None
            ),
            "accepted": True,
            "rejection_reason": None,
        }
        history.append(row)
        persist_history()
        emit_round(0, row, "target_met_before_llm")

    for round_number in range(1, max_rounds + 1) if (run_all_rounds or not target_met) else ():
        current_metrics = training_metrics(best_training)
        if progress is not None:
            progress.emit(
                "llm-tuning",
                "running",
                summary=f"LLM 调参第 {round_number}/{max_rounds} 轮：正在请求参数建议",
            )
        proposal_response: dict[str, Any] | None = None
        request_error: str | None = None
        request_attempts = 0
        retry_count = int(provider_options["request_retries"])
        for attempt in range(retry_count + 1):
            request_attempts = attempt + 1
            proposal_response, request_error = _request_proposal(
                current=best_config,
                metrics=current_metrics,
                search_space=search_space,
                round_number=round_number,
                settings=options,
                objective_metric=objective_metric,
                min_improvement=min_improvement,
                baseline_score=baseline_score,
                baseline_oot_ks=baseline_oot_ks,
                history=history,
            )
            if request_error is None:
                break
            if not _retryable_provider_error(request_error) or attempt >= retry_count:
                break
            backoff = float(provider_options["retry_backoff_seconds"])
            if backoff > 0:
                time.sleep(backoff * (attempt + 1))
        row: dict[str, Any] = {
            "round": round_number,
            "status": "skipped" if request_error else "proposed",
            "request_attempts": request_attempts,
            "objective_metric": objective_metric,
            "metrics": current_metrics,
            "current_score": best_score,
            "current_parameters": asdict(best_config),
            "proposed_parameters": None,
            "reason": request_error,
            "candidate_score": None,
            "improvement": None,
            "improvement_vs_current_best": None,
            "improvement_vs_baseline": None,
            "oot_improvement_vs_baseline": None,
            "accepted": False,
            "rejection_reason": request_error,
        }
        if request_error:
            # A transient provider failure should not discard the remaining
            # LLM budget. Retry within this round, then record the failed round
            # and continue. Only repeated failures reach the terminal guard.
            consecutive_provider_errors += 1
            row["stop_reason"] = "provider_error_retry_exhausted"
            history.append(row)
            persist_history()
            emit_round(round_number, row, "skipped_retry_exhausted")
            if (
                not _retryable_provider_error(request_error)
                or consecutive_provider_errors >= int(provider_options["max_consecutive_provider_errors"])
            ):
                stop_reason = "provider_error"
                row["stop_reason"] = stop_reason
                break
            continue
        consecutive_provider_errors = 0
        proposal = proposal_response or {}
        row["proposed_parameters"] = proposal.get("parameters")
        row["reason"] = proposal.get("reason")
        candidate_config, bounds_error = _bounded_config(
            best_config, proposal.get("parameters", {}), search_space
        )
        if bounds_error or candidate_config is None:
            row["status"] = "rejected_bounds"
            row["rejection_reason"] = bounds_error
            history.append(row)
            persist_history()
            emit_round(round_number, row, "rejected_bounds")
            continue
        if asdict(candidate_config) == asdict(best_config):
            row["status"] = "rejected_no_change"
            row["rejection_reason"] = "LLM 返回的参数与当前最优参数完全相同"
            history.append(row)
            persist_history()
            stagnant_rounds += 1
            emit_round(round_number, row, "rejected_no_change")
            if stagnant_rounds >= stopping["plateau_rounds"] and not run_all_rounds:
                stop_reason = "plateau"
                row["stop_reason"] = stop_reason
                break
            continue
        try:
            candidate_training = train_lightgbm(
                train, test, contract, features, candidate_config,
                # Keep the seed fixed so a round is compared on parameters,
                # not on a different random draw from LightGBM bagging.
                seed=seed,
                balance_classes=balance_classes,
                categorical_features=categorical_features,
            )
            candidate_metrics = {
                "train_ks": _metric(candidate_training, train, features, target_col, "ks"),
                "test_ks": _metric(candidate_training, test, features, target_col, "ks"),
                "test_auc": _metric(candidate_training, test, features, target_col, "auc"),
            }
            candidate_metrics["validate_ks"] = candidate_metrics["test_ks"]
            candidate_metrics["validate_auc"] = candidate_metrics["test_auc"]
            if candidate_metrics["train_ks"] is not None and candidate_metrics["test_ks"] is not None:
                candidate_metrics["train_validate_ks_gap"] = (
                    candidate_metrics["train_ks"] - candidate_metrics["test_ks"]
                )
            if oot is not None:
                candidate_metrics["oot_ks"] = _metric(
                    candidate_training, oot, features, target_col, "ks"
                )
                candidate_metrics["oot_auc"] = _metric(
                    candidate_training, oot, features, target_col, "auc"
                )
            row["metrics"] = candidate_metrics
            candidate_score = _metric(candidate_training, test, features, target_col, objective_metric)
            row["candidate_score"] = candidate_score
            row["improvement_vs_current_best"] = (
                candidate_score - best_score
                if candidate_score is not None and best_score is not None
                else None
            )
            # ``improvement`` is retained for backward compatibility with
            # existing history files; new events expose the unambiguous
            # current-best and baseline deltas separately.
            row["improvement"] = row["improvement_vs_current_best"]
            row["improvement_vs_baseline"] = (
                candidate_score - baseline_score
                if candidate_score is not None and baseline_score is not None
                else None
            )
            if oot is not None and baseline_oot_ks is not None:
                candidate_oot_ks = candidate_metrics.get("oot_ks")
                row["oot_improvement_vs_baseline"] = (
                    float(candidate_oot_ks) - baseline_oot_ks
                    if candidate_oot_ks is not None
                    else None
                )
            gap_improved, gap_detail = _generalization_improvement(
                candidate_metrics, current_metrics, stopping
            )
            row["generalization_improvement"] = gap_detail
            row["accepted_by_generalization"] = False
            guardrail_reasons = _guardrail_reasons(
                candidate_metrics,
                stopping,
                baseline_oot_ks,
                oot_bad_count,
                allow_gap_improvement=gap_improved,
            )
            row["guardrail_reasons"] = guardrail_reasons
            if guardrail_reasons:
                row["status"] = "rejected_guardrail"
                row["rejection_reason"] = "；".join(guardrail_reasons)
            elif candidate_score is not None and best_score is not None and candidate_score - best_score >= min_improvement:
                best_config, best_training, best_score = candidate_config, candidate_training, candidate_score
                row["status"] = "accepted"
                row["accepted"] = True
                row["rejection_reason"] = None
            elif gap_improved:
                best_config, best_training, best_score = candidate_config, candidate_training, candidate_score
                row["status"] = "accepted_generalization"
                row["accepted"] = True
                row["accepted_by_generalization"] = True
                row["rejection_reason"] = None
            else:
                row["status"] = "rejected_metric"
                row["rejection_reason"] = f"提升未达到最低要求 {min_improvement}"
            candidate_results.append((round_number, candidate_config, candidate_training, row.copy()))
            stop_after_round = False
            if row["status"] == "accepted_generalization":
                # A material gap reduction is progress even if Validate KS is
                # nearly flat; do not count it toward plateau stopping.
                stagnant_rounds = 0
            elif row["status"] == "accepted":
                gain = row.get("improvement_vs_current_best")
                stagnant_rounds = (
                    stagnant_rounds + 1
                    if gain is None or float(gain) < stopping["plateau_min_improvement"]
                    else 0
                )
                target_met, target_close, target_detail = _target_evaluation(
                    candidate_metrics, stopping, oot_bad_count
                )
                if target_met and not run_all_rounds:
                    stop_reason = "target_met"
                    row["stop_reason"] = stop_reason
                    row["reason"] = f"已达到目标：{target_detail}"
                    stop_after_round = True
                elif target_close and not run_all_rounds:
                    stop_reason = "target_close"
                    row["stop_reason"] = stop_reason
                    row["reason"] = f"已接近目标（容差 {stopping['target_tolerance']:.4f}）：{target_detail}"
                    stop_after_round = True
            else:
                stagnant_rounds += 1
            if stagnant_rounds >= stopping["plateau_rounds"] and not stop_after_round and not run_all_rounds:
                stop_reason = "plateau"
                row["stop_reason"] = stop_reason
                row["reason"] = f"连续 {stagnant_rounds} 轮未形成可接受提升，停止调参"
                stop_after_round = True
            history.append(row)
            persist_history()
            emit_round(round_number, row, str(row["status"]))
            if stop_after_round:
                break
        except Exception as exc:  # candidate failure must not break the run
            row["status"] = "candidate_error"
            row["rejection_reason"] = f"候选训练失败: {type(exc).__name__}: {exc}"
            history.append(row)
            persist_history()
            emit_round(round_number, row, "candidate_error")
    final_metrics = training_metrics(best_training)
    _, _, target_detail = _target_evaluation(final_metrics, stopping, oot_bad_count)
    persist_history()
    flat_rows = []
    for item in history:
        flat = {
            key: _csv_safe_value(value)
            for key, value in item.items()
            if key not in {"current_parameters", "proposed_parameters", "metrics"}
        }
        flat["current_parameters"] = json.dumps(item.get("current_parameters"), ensure_ascii=False)
        flat["proposed_parameters"] = json.dumps(item.get("proposed_parameters"), ensure_ascii=False)
        flat["metrics"] = json.dumps(item.get("metrics"), ensure_ascii=False)
        flat_rows.append(flat)
    history_table = pl.DataFrame(flat_rows) if flat_rows else pl.DataFrame(schema={"round": pl.Int64})
    history_table.write_csv(run_dir / "llm_tuning_history.csv")
    (run_dir / "llm_tuning_summary.json").write_text(
        json.dumps(
            {
                # Round 0 is the baseline snapshot, not an LLM proposal round.
                "enabled": bool(options.get("enabled", False)),
                "max_rounds": max_rounds,
                "run_all_rounds": run_all_rounds,
                "rounds_completed": sum(
                    1 for row in history if int(row.get("round", 0) or 0) > 0
                ),
                "accepted_rounds": sum(
                    bool(row.get("accepted"))
                    for row in history
                    if int(row.get("round", 0) or 0) > 0
                ),
                "stop_reason": stop_reason,
                "provider_policy": provider_options,
                "request_attempts_total": sum(
                    int(row.get("request_attempts", 0) or 0)
                    for row in history
                    if int(row.get("round", 0) or 0) > 0
                ),
                "target_detail": target_detail,
                "oot_bad_count": oot_bad_count,
                "stopping": stopping,
                "selected_parameters": asdict(best_config),
                "selected_score": best_score,
                "history": str(history_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_config, best_training, history, history_table, candidate_results

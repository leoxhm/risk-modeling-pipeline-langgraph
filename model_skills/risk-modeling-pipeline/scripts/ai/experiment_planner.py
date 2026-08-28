"""Generate a reviewable experiment plan from deterministic evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from modeling.config import ModelConfig


def build_experiment_plan(
    planned_mode: str,
    preflight_report: dict[str, Any],
    model_config: ModelConfig | None,
) -> dict[str, Any]:
    """Recommend an execution shape without modifying the configured plan."""

    blocker_codes = [
        finding["code"]
        for finding in preflight_report["findings"]
        if finding["severity"] == "blocker"
    ]
    if planned_mode == "eda":
        return {
            "status": "needs_confirmation" if blocker_codes else "ready_for_confirmation",
            "planned_mode": planned_mode,
            "recommended_training_mode": "not_applicable",
            "blocker_codes": blocker_codes,
            "reason": "EDA 可以在确认字段语义后运行，不包含模型训练。",
            "requires_user_confirmation": True,
        }

    assert model_config is not None
    requested_training_mode = model_config.training.mode
    recommended_training_mode = requested_training_mode
    if blocker_codes and requested_training_mode == "tuning":
        recommended_training_mode = "baseline"
    return {
        "status": "blocked" if blocker_codes else "ready_for_confirmation",
        "planned_mode": planned_mode,
        "requested_training_mode": requested_training_mode,
        "recommended_training_mode": recommended_training_mode,
        "blocker_codes": blocker_codes,
        "validation": {
            "oot_months": model_config.split.oot_months,
            "test_ratio": model_config.split.test_ratio,
            "tuning_cv_folds": model_config.tuning.cv_folds,
            "oot_usage": "final_validation_only",
        },
        "tuning": {
            "framework": "optuna",
            "sampler": model_config.tuning.sampler,
            "n_trials": model_config.tuning.n_trials,
            "timeout_seconds": model_config.tuning.timeout_seconds,
            "selection_metric": "penalized_mean_cv_auc",
        },
        "feature_policy": {
            "preprocessing": asdict(model_config.feature_preprocessing),
            "selection": asdict(model_config.feature_selection),
            "fit_scope": "train_only",
            "test_oot_usage": "frozen_transform_only",
        },
        "reason": (
            "存在阻断项，默认不建议执行自动调参。"
            if blocker_codes
            else "样本检查未发现阻断项，可在用户确认后执行配置的实验方案。"
        ),
        "requires_user_confirmation": True,
    }

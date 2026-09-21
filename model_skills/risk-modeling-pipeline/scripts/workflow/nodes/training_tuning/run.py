"""Execute model training only after the model-config approval gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import yaml

from ..common import (
    NodeContext,
    approval_path as node_approval_path,
    write_json,
    write_node_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the training-tuning workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--pmml-converter")
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="仅基于上一次模型的最佳参数执行 LLM 后置调参，不重新运行 Optuna",
    )
    return parser


def _require_model_config(context: NodeContext) -> tuple[dict[str, Any], Path]:
    approval_path = node_approval_path(context, "model-config")
    if not approval_path.is_file():
        raise RuntimeError("model-config has not been confirmed; run model-config with --confirm-config first.")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise RuntimeError(
            "model-config result has not been confirmed; run workflow confirm model-config first."
        )
    canonical = Path(approval.get("canonical_model_config", ""))
    if not canonical.is_absolute():
        canonical = (context.project_root / canonical).resolve()
    if not canonical.is_file():
        raise RuntimeError(f"Confirmed model configuration cannot be found: {canonical}")
    import hashlib

    if approval.get("canonical_sha256") != hashlib.sha256(canonical.read_bytes()).hexdigest():
        raise RuntimeError("configs/model_config.yaml changed after model-config confirmation; rerun model-config.")
    return approval, canonical


def _sample_config(parameters: dict[str, Any]):
    from preprocessing.sample_config import (
        SampleConfig,
        SampleDiagnosticThresholds,
        SampleTreatmentConfig,
    )

    treatment = parameters["sample_treatment"]
    return SampleConfig(
        diagnostics=SampleDiagnosticThresholds(
            iv_weak_threshold=0.02,
            psi_warning_threshold=0.25,
            psi_stable_threshold=0.10,
            bad_rate_min=0.01,
            bad_rate_max=0.20,
            feature_quality_pass_ratio=0.80,
            imbalance_warning_minority_rate=0.10,
            imbalance_critical_minority_rate=0.01,
            latest_month_min_volume_ratio=0.50,
            monthly_bad_rate_change_warning=0.03,
            high_missing_row_rate=0.80,
        ),
        treatment=SampleTreatmentConfig(
            duplicate_action=str(treatment["duplicate_action"]),
            missing_target_action=str(treatment["missing_target_action"]),
            all_null_feature_action=str(treatment["all_null_feature_action"]),
            high_missing_row_action=str(treatment["high_missing_row_action"]),
            incomplete_latest_month_action=str(treatment["incomplete_latest_month_action"]),
            class_imbalance_action=str(treatment["class_imbalance_action"]),
        ),
    )


def _review_digest(path: Path) -> dict[str, Any]:
    """Build a compact, deterministic review summary for the UI response."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unknown", "finding_count": 0, "findings": [], "recommendations": []}
    findings = payload.get("findings") if isinstance(payload, dict) else []
    recommendations = payload.get("recommendations") if isinstance(payload, dict) else []
    findings = findings if isinstance(findings, list) else []
    recommendations = recommendations if isinstance(recommendations, list) else []
    return {
        "status": payload.get("status", "unknown"),
        "finding_count": len(findings),
        "severity_counts": {
            severity: sum(
                isinstance(item, dict) and item.get("severity") == severity
                for item in findings
            )
            for severity in ("blocker", "warning", "info")
        },
        "findings": [
            str(item["message"])
            for item in findings
            if isinstance(item, dict) and item.get("message")
        ][:4],
        "recommendations": [
            str(item["action"])
            for item in recommendations
            if isinstance(item, dict) and item.get("action")
        ][:3],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = NodeContext.from_args(args, "training-tuning")
    _require_model_config(context)
    from data.contract import validate_contract
    from modeling.config import load_model_config
    from modeling.pipeline import run_model_pipeline
    from preprocessing.sample_diagnostics import apply_sample_treatment

    approval, canonical_path = _require_model_config(context)
    model_config = load_model_config(canonical_path)
    # model-config publishes a YAML snapshot. Keep the training executor on
    # the same parser as the configuration gate instead of attempting to parse
    # YAML as JSON (which caused JSONDecodeError immediately after approval).
    canonical_body = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(canonical_body, dict):
        raise RuntimeError(f"Confirmed model configuration must be a YAML mapping: {canonical_path}")
    llm_settings = (canonical_body.get("tuning", {}) or {}).get("llm", {})
    if getattr(args, "llm_only", False):
        if not isinstance(llm_settings, dict) or not llm_settings.get("enabled", False):
            raise RuntimeError(
                "LLM 后置调参未启用，请先在 model-config.yaml 中设置 tuning.llm.enabled=true 并重新确认。"
            )
        previous_summary = (
            context.project_root / "outputs" / "training-tuning" / "run" / "model_summary.json"
        )
        if not previous_summary.is_file():
            raise RuntimeError(
                f"找不到上一次模型摘要: {previous_summary}；请先完成 model-config 训练。"
            )
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        previous_parameters = previous.get("final_model_parameters")
        if not isinstance(previous_parameters, dict):
            raise RuntimeError("上一次模型摘要缺少 final_model_parameters，无法执行 LLM 后置调参。")
        # Rebuild a baseline-only ModelConfig using the previously selected
        # parameters. run_model_pipeline will then perform only the bounded
        # LLM refinement layer and write a separate outputs/llm-tuning run.
        from modeling.config import LightGbmConfig

        model_config = replace(
            model_config,
            model=LightGbmConfig(**previous_parameters),
            training=replace(model_config.training, mode="baseline"),
            tuning=replace(model_config.tuning, method="llm"),
        )
    with context.progress.track(
        "training",
        running_summary="正在按已确认配置读取并处理训练样本",
        success_summary="训练数据准备完成，开始 LightGBM 训练与调参",
    ):
        loaded = context.reload_data()
        validated = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )
        treatment = apply_sample_treatment(
            loaded.data,
            validated.contract,
            _sample_config(canonical_body),
        )
        treated_contract = validate_contract(treatment.data, validated.contract)

    model_dir = context.output_dir / "models"
    report_dir = context.output_dir / "reports"
    result = run_model_pipeline(
        treatment.data,
        treated_contract,
        model_config,
        context.output_dir / "run",
        config_path=canonical_path,
        progress=context.progress,
        balance_classes=canonical_body["sample_treatment"]["class_imbalance_action"] == "class_weight",
        pmml_converter=args.pmml_converter,
        model_dir=model_dir,
        report_dir=report_dir,
        generate_report=True,
        llm_tuning=llm_settings,
    )
    model_summary_path = result.output_dir / "model_summary.json"
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    review_path = result.output_dir / "ai_model_review.json"
    review_digest = _review_digest(review_path)
    metric_rows = result.metrics.to_dicts()
    test_row = next((row for row in metric_rows if row.get("dataset") == "test"), {})
    oot_row = next((row for row in metric_rows if row.get("dataset") == "oot"), {})
    selected_candidate = model_summary.get("selected_candidate", "baseline")
    acceptance_metric = model_summary.get("acceptance_metric", "ks")
    llm_info = model_summary.get("llm_tuning") or {}
    candidate_comparison_artifact = model_summary.get("candidate_comparison_artifact")
    llm_note = (
        f"LLM 调参接受 {llm_info.get('accepted_rounds', 0)} 轮，详见迭代记录。"
        if llm_info.get("enabled")
        else "LLM 后置调参未启用。"
    )
    node_summary_text = (
        f"LightGBM 训练与调参完成，最终采用 {selected_candidate}；"
        f"Validate {acceptance_metric.upper()}={test_row.get(acceptance_metric)}, "
        f"OOT {acceptance_metric.upper()}={oot_row.get(acceptance_metric)}；{llm_note}\n\n"
        f"模型审查：{review_digest['status']}，发现 {review_digest['finding_count']} 项。"
        + ("\n" + "\n".join(f"- {item}" for item in review_digest["findings"]) if review_digest["findings"] else "\n- 未发现阈值型异常。")
        + ("\n\n改进建议：\n" + "\n".join(f"- {item}" for item in review_digest["recommendations"]) if review_digest["recommendations"] else "")
    )
    summary = write_json(
        context.output_dir / "training_summary.json",
        {
            "node_id": "training-tuning",
            "status": "success",
            "model_config": str(canonical_path),
            "model_config_approval": str(node_approval_path(context, "model-config")),
            "training_mode": model_config.training.mode,
            "selected_features": list(result.selected_features),
            "selected_candidate": selected_candidate,
            "acceptance_metric": model_summary.get("acceptance_metric"),
            "acceptance_min_improvement": model_summary.get("acceptance_min_improvement"),
            "acceptance_improvement_vs_baseline": model_summary.get(
                "acceptance_improvement_vs_baseline"
            ),
            "acceptance_reason": model_summary.get("acceptance_reason"),
            "llm_tuning": model_summary.get("llm_tuning"),
            "report": str(result.report_path),
            "html_report": str(result.html_report_path) if result.html_report_path else None,
            "models_dir": str(model_dir),
            "metrics": metric_rows,
            "split_strategy": model_summary.get("split_strategy"),
            "test_month_values": model_summary.get("test_month_values", []),
            "oot_month_values": model_summary.get("oot_month_values", []),
            "tuning_cv_strategy": model_summary.get("tuning_cv_strategy"),
            "tuning_cv_folds": model_summary.get("tuning_cv_folds"),
            "tuning_validation_months": model_summary.get("tuning_validation_months"),
            "tuning_gap_months": model_summary.get("tuning_gap_months"),
            "tuning_trials": model_summary.get("tuning_trials_artifact"),
            "cv_fold_metrics": model_summary.get("cv_fold_metrics_artifact"),
            "evaluation_artifacts": model_summary.get("evaluation_artifacts", {}),
            "model_review": review_digest,
            "split_summary": result.split_summary.to_dicts(),
        },
    )
    node_summary = write_node_summary(
        context.output_dir,
        "training-tuning",
        node_summary_text,
        {
            "selected_candidate": selected_candidate,
            "acceptance_metric": acceptance_metric,
            "test": test_row,
            "oot": oot_row,
            "report": str(result.report_path),
            "html_report": str(result.html_report_path) if result.html_report_path else None,
            "models_dir": str(model_dir),
            "llm_tuning": llm_info,
            "llm_history": llm_info.get("history_artifact"),
            "candidate_comparison": candidate_comparison_artifact,
            "tuning_cv_strategy": model_summary.get("tuning_cv_strategy"),
            "tuning_cv_folds": model_summary.get("tuning_cv_folds"),
            "tuning_trials": model_summary.get("tuning_trials_artifact"),
            "cv_fold_metrics": model_summary.get("cv_fold_metrics_artifact"),
            "evaluation_artifacts": model_summary.get("evaluation_artifacts", {}),
            "model_review": review_digest,
        },
    )
    context.progress.emit(
        "training-tuning",
        "success",
        summary=node_summary_text,
        artifacts=[
            summary,
            node_summary,
            model_summary_path,
            *([Path(candidate_comparison_artifact)] if candidate_comparison_artifact else []),
            result.report_path,
            *([result.html_report_path] if result.html_report_path else []),
            model_dir,
            report_dir,
            *([Path(llm_info["history_artifact"])] if llm_info.get("history_artifact") else []),
            *([Path(model_summary["tuning_trials_artifact"])] if model_summary.get("tuning_trials_artifact") else []),
            *([Path(model_summary["cv_fold_metrics_artifact"])] if model_summary.get("cv_fold_metrics_artifact") else []),
            *[Path(path) for path in model_summary.get("evaluation_artifacts", {}).values()],
        ],
    )
    return {
        "status": "success",
        "node_id": "training-tuning",
        "summary": str(summary),
        "node_summary": str(node_summary),
        "report": str(result.report_path),
        "html_report": str(result.html_report_path) if result.html_report_path else None,
        "review": str(review_path),
        "review_summary": review_digest,
        "review_status": model_summary.get("review_status"),
        "models_dir": str(model_dir),
        "selected_candidate": selected_candidate,
        "acceptance_reason": model_summary.get("acceptance_reason"),
        "llm_tuning": llm_info,
        "candidate_comparison": candidate_comparison_artifact,
        "selected_features": list(result.selected_features),
        "metrics": result.metrics.to_dicts(),
        "approval": str(node_approval_path(context, "model-config")),
        "display_files": [
            str(result.report_path),
            *([str(result.html_report_path)] if result.html_report_path else []),
            str(result.output_dir / "variable_information.csv"),
            str(result.output_dir / "model_stability.csv"),
            *([str(candidate_comparison_artifact)] if candidate_comparison_artifact else []),
            *([str(llm_info["history_artifact"])] if llm_info.get("history_artifact") else []),
            *([str(model_summary["tuning_trials_artifact"])] if model_summary.get("tuning_trials_artifact") else []),
            *([str(model_summary["cv_fold_metrics_artifact"])] if model_summary.get("cv_fold_metrics_artifact") else []),
            *list(model_summary.get("evaluation_artifacts", {}).values()),
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except Exception as exc:
        # Do not leave a previous successful training_summary.json in place
        # after a retry fails (for example because a stale Optuna study or a
        # changed input blocks the run). A machine-readable failed summary is
        # less likely to be mistaken for the current run's result.
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else Path(args.project_root).expanduser().resolve() / "outputs" / "training-tuning"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "training_summary.json",
            {
                "node_id": "training-tuning",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Model configuration and training node.

The user-facing ``model-config`` node is a composite boundary. Its first call
creates or loads the concise YAML, publishes an auditable canonical snapshot and
pauses on a single decision-table confirmation. A confirmed call starts the
deterministic ``training-tuning`` executor in-process. Ordinary users receive a
concise decision table without having to understand every LightGBM/Optuna
parameter. ``--confirm-config`` remains accepted for compatibility with older
callers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..common import NodeContext, approval_path as node_approval_path, write_json
from progress import ProgressReporter


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge user-facing core settings over safe engine defaults."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _effective_parameters(context: NodeContext) -> dict[str, Any]:
    """Expand the concise model YAML before canonical validation/training.

    Feature preprocessing and selection are inherited from the confirmed
    feature-processing YAML. Any omitted low-level LightGBM/Optuna fields are
    filled from the engine's conservative defaults, keeping the confirmation
    file readable without changing the actual training contract.
    """
    from workflow.node_config import DEFAULT_NODE_CONFIGS

    defaults = copy.deepcopy(DEFAULT_NODE_CONFIGS["model-config"]["parameters"])
    feature_path = context.node_config_dir / "feature-processing.yaml"
    if feature_path.is_file():
        try:
            feature_body = yaml.safe_load(feature_path.read_text(encoding="utf-8")) or {}
            feature_parameters = feature_body.get("parameters", {})
            if isinstance(feature_parameters, dict):
                if isinstance(feature_parameters.get("preprocessing"), dict):
                    defaults["feature_preprocessing"] = copy.deepcopy(feature_parameters["preprocessing"])
                if isinstance(feature_parameters.get("selection"), dict):
                    defaults["feature_selection"] = copy.deepcopy(feature_parameters["selection"])
        except (OSError, yaml.YAMLError):
            pass
    return _deep_merge(defaults, context.config.parameters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the model-config workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--confirm-config", action="store_true")
    parser.add_argument(
        "--confirm-result",
        action="store_true",
        help="Confirm the completed training/review result",
    )
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_approval(context: NodeContext, node_id: str) -> dict[str, Any]:
    path = node_approval_path(context, node_id)
    if not path.is_file():
        raise RuntimeError(
            f"{node_id}.yaml has not been completed. Run the {node_id} node first."
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise RuntimeError(
            f"{node_id} result has not been confirmed; run workflow confirm {node_id} first."
        )
    config_path = context.node_config_dir / f"{node_id}.yaml"
    if approval.get("config_sha256") != _hash(config_path):
        raise RuntimeError(f"{config_path.name} changed after confirmation; rerun {node_id}.")
    return approval


def _canonical_config(parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract the modeling sections plus the auditable treatment policy."""
    sections = (
        "split",
        "feature_preprocessing",
        "feature_selection",
        "training",
        "tuning",
        "model",
    )
    canonical = {section: copy.deepcopy(parameters[section]) for section in sections}
    # ``load_model_config`` intentionally ignores this extra section, while
    # downstream training/audit code can read the confirmed policy from the
    # same canonical YAML instead of relying on a second config file.
    canonical["sample_treatment"] = copy.deepcopy(parameters["sample_treatment"])
    return {"sample_treatment": canonical.pop("sample_treatment"), **canonical}


def _write_yaml(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _build_model_config_summary(
    *,
    parameters: dict[str, Any],
) -> str:
    """Build the small user-facing decision table without exposing YAML."""
    split = parameters.get("split", {}) or {}
    tuning = parameters.get("tuning", {}) or {}
    model = parameters.get("model", {}) or {}
    strategy = str(split.get("strategy", "time")).strip().lower()
    oot_months = int(split.get("oot_months", 2))
    validate_months = int(split.get("test_months", 2))
    if strategy == "time":
        split_choice = f"按时间切分；Train=早期月份，Validate=之后 {validate_months} 个月，OOT=最近 {oot_months} 个月"
        split_reason = "按时间顺序模拟上线，避免随机切分把未来信息带入训练；OOT 仅作最终稳定性验证"
    else:
        split_choice = f"{strategy}；Validate 比例 {float(split.get('test_ratio', 0.2)):.0%}"
        split_reason = "兼容旧版随机切分；如数据有明确时间顺序，建议改用 time 策略"
    method = str(tuning.get("method", "llm")).strip().lower()
    method_label = {
        "llm": "LLM 逐轮调参",
        "optuna": "Optuna 自动调参",
        "none": "仅 Baseline",
    }.get(method, method or "未设置")
    model_name = "LightGBM（当前唯一模型）"
    table_rows = [
        ("样本切分", "时间划分", split_choice, split_reason),
        ("调参设置", "调参方式", method_label, "三种方式互斥；本次只执行当前选择"),
        ("模型选择", "基础模型", model_name, "当前版本仅支持 LightGBM"),
    ]
    lines = [
        "## 模型配置确认摘要",
        "",
        "| 类别 | 决策项 | 当前选择 | 说明 |",
        "|---|---|---|---|",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in table_rows
    )
    lines.append("\n完整参数保存在工作区 model-config.yaml，仅用于内部校验和复现，不在聊天区展开。")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = NodeContext.from_args(args, "model-config")
    # The first invocation only publishes the compact decision table.  The
    # host must call ``workflow confirm model-config`` after the user accepts
    # that table; only then is training/tuning started. Repeated first
    # invocations are idempotent and never retrain the model.
    confirm_requested = bool(
        getattr(args, "confirm_result", False) or getattr(args, "confirm_config", False)
    )
    result_confirmed = confirm_requested
    # Model configuration is downstream of confirmed roles and the feature
    # policy. It is intentionally independent from the legacy batch runner.
    _require_approval(context, "data-read")
    _require_approval(context, "feature-processing")

    from data.contract import validate_contract
    from modeling.config import load_model_config

    loaded = context.reload_data()
    validated = validate_contract(
        loaded.data,
        context.effective_contract,
        allow_id_duplicates=True,
        allow_target_issues=True,
    )

    config_path = context.config.path
    config_hash = _hash(config_path)
    parameters = _effective_parameters(context)
    config_summary_text = _build_model_config_summary(parameters=parameters)
    decision_summary_path = context.output_dir / "model_config_decision_summary.md"
    decision_summary_path.parent.mkdir(parents=True, exist_ok=True)
    decision_summary_path.write_text(config_summary_text + "\n", encoding="utf-8")
    llm_parameters = parameters["tuning"].get("llm") or {}
    tuning_method = str(parameters["tuning"].get("method", "llm")).strip().lower()
    canonical_path = context.project_root / "configs" / "model_config.yaml"
    # A result confirmation should acknowledge the completed run rather than
    # retraining the model. The first pass stores the compact result metadata
    # in the pending approval; a matching hash lets this branch finalize it.
    existing_approval_path = node_approval_path(context, "model-config")
    existing_approval: dict[str, Any] = {}
    if existing_approval_path.is_file():
        try:
            existing_approval = json.loads(existing_approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_approval = {}

    # A repeated initial command must not append another progress event or
    # launch another training process. The previous waiting event already
    # contains the exact table to display.
    if (
        not confirm_requested
        and existing_approval.get("status") == "awaiting_user_confirmation"
        and existing_approval.get("training_completed") is not True
        and existing_approval.get("config_sha256") == config_hash
    ):
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "model-config",
            "summary": "模型决策摘要已生成，等待用户确认后开始训练。",
            "decision_summary": str(decision_summary_path),
            "approval": str(existing_approval_path),
            "display_files": [str(decision_summary_path)],
        }

    context.progress.emit(
        "model-config",
        "running",
        summary=(
            "已确认建模决策，正在进入训练与调参"
            if confirm_requested
            else "正在读取已确认的数据和特征结果，生成建模决策摘要"
        ),
    )
    if result_confirmed and existing_approval_path.is_file():
        try:
            existing_approval = json.loads(existing_approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_approval = {}
        if (
            existing_approval.get("status") == "awaiting_user_confirmation"
            and existing_approval.get("config_sha256") == config_hash
            and existing_approval.get("training_completed") is True
        ):
            approved = dict(existing_approval)
            approved["status"] = "approved"
            approved["result_confirmed"] = True
            approval = write_json(existing_approval_path, approved)
            result_summary = str(existing_approval.get("result_summary") or "模型训练结果已确认。")
            stored_artifacts = [Path(str(item)) for item in existing_approval.get("artifacts", []) if item]
            context.progress.emit(
                "model-config",
                "success",
                summary=result_summary,
                artifacts=[approval, *[item for item in stored_artifacts if item.is_file()]],
            )
            return {
                "status": "success",
                "node_id": "model-config",
                # The full confirmation table is emitted once via the success
                # progress event; avoid duplicating it in the JSON payload.
                "summary": "模型配置与训练结果已确认。",
                "model_review": existing_approval.get("review_path"),
                "review_status": existing_approval.get("review_status"),
                "review_summary": existing_approval.get("review_summary", {}),
                "training": existing_approval.get("training_result", {}),
                "approval": str(approval),
                "decision_summary": str(context.output_dir / "model_config_decision_summary.md"),
                "display_files": [str(item) for item in stored_artifacts if item.is_file()],
            }
    summary_path = write_json(
        context.output_dir / "model_config_summary.json",
        {
            "node_id": "model-config",
            "status": "approved" if confirm_requested else "awaiting_user_confirmation",
            "data_path": str(context.data_path),
            "row_count": loaded.row_count,
            "feature_count": len(validated.feature_cols),
            "feature_cols": list(validated.feature_cols),
            "sections": [
                "sample_treatment",
                "split",
                "feature_preprocessing",
                "feature_selection",
                "training",
                "tuning",
                "model",
            ],
            "sample_treatment": parameters["sample_treatment"],
            "training_mode": parameters["training"].get("mode"),
            "tuning_method": tuning_method,
            "acceptance_metric": parameters["training"].get("acceptance_metric", "ks"),
            "min_improvement": parameters["training"].get("min_improvement", 0.0),
            "tuning": {
                "method": tuning_method,
                "sampler": parameters["tuning"].get("sampler"),
                "objective_metric": parameters["tuning"].get("objective_metric"),
                "n_trials": parameters["tuning"].get("n_trials"),
                "timeout_seconds": parameters["tuning"].get("timeout_seconds"),
                "cv_strategy": parameters["tuning"].get("cv_strategy", "rolling"),
                "cv_folds": parameters["tuning"].get("cv_folds", 3),
                "validation_months": parameters["tuning"].get("validation_months", 1),
                "gap_months": parameters["tuning"].get("gap_months", 0),
                "min_train_months": parameters["tuning"].get("min_train_months", 3),
                "llm": {
                    "enabled": bool(
                        tuning_method == "llm" and llm_parameters.get("enabled", False)
                    ),
                    "max_rounds": llm_parameters.get("max_rounds", 10),
                    "run_all_rounds": bool(llm_parameters.get("run_all_rounds", True)),
                    "min_improvement": llm_parameters.get("min_improvement", 0.005),
                    "stopping": llm_parameters.get("stopping", {}),
                    "model": llm_parameters.get("model"),
                    "base_url": llm_parameters.get("base_url"),
                    "api_key_configured": bool(llm_parameters.get("api_key")),
                    "verify_ssl": bool(llm_parameters.get("verify_ssl", True)),
                    "ca_bundle": llm_parameters.get("ca_bundle"),
                },
            },
            "node_config": str(config_path),
            "node_config_sha256": config_hash,
            "decision_summary": str(decision_summary_path),
        },
    )
    confirmation_path = write_json(
        context.output_dir / "model_config_confirmation.json",
        {
            "node_id": "model-config",
            "status": "approved" if confirm_requested else "awaiting_user_confirmation",
            "user_confirmation_required": not confirm_requested,
            "message": (
                "已确认建模决策摘要，可以开始训练与调参。"
                if confirm_requested
                else "建模决策摘要已生成，请确认后开始训练与调参。"
            ),
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "canonical_model_config": str(canonical_path),
            "summary": str(summary_path),
            "display_files": [str(decision_summary_path)],
        },
    )
    # Validate the concise node YAML using the exact parser consumed by the
    # training pipeline, then publish one canonical snapshot for training.
    canonical = _canonical_config(parameters)
    snapshot_path = _write_yaml(context.output_dir / "model_config_snapshot.yaml", canonical)
    load_model_config(snapshot_path)
    _write_yaml(canonical_path, canonical)
    canonical_hash = _hash(canonical_path)
    if not confirm_requested:
        pending_approval = write_json(
            node_approval_path(context, "model-config", for_write=True),
            {
                "node_id": "model-config",
                "status": "awaiting_user_confirmation",
                "config": str(config_path),
                "config_sha256": config_hash,
                "canonical_model_config": str(canonical_path),
                "canonical_sha256": canonical_hash,
                "training_completed": False,
            },
        )
        context.progress.emit(
            "model-config",
            "waiting_confirmation",
            summary=(config_summary_text + "\n\n请确认以上建模决策后开始训练；如需调整，请直接提出文字修改要求。"),
            artifacts=[config_path, decision_summary_path, summary_path, confirmation_path, snapshot_path, canonical_path, pending_approval],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "model-config",
            "config": str(config_path),
            "summary": str(summary_path),
            "confirmation": str(confirmation_path),
            "snapshot": str(snapshot_path),
            "canonical_model_config": str(canonical_path),
            "approval": str(pending_approval),
            "display_files": [str(decision_summary_path)],
        }
    approval_path = write_json(
        node_approval_path(context, "model-config", for_write=True),
        {
            "node_id": "model-config",
            "status": "approved",
            "config": str(config_path),
            "config_sha256": config_hash,
            "canonical_model_config": str(canonical_path),
            "canonical_sha256": canonical_hash,
            "sample_treatment": parameters["sample_treatment"],
        },
    )
    context.progress.emit(
        "model-config",
        "running",
        summary="建模配置已生成，正在自动进入模型训练与调参",
        artifacts=[config_path, snapshot_path, canonical_path, approval_path],
    )

    # ``model-config`` is presented as one selectable UI node.  Run the
    # training executor directly after the approval gate instead of asking the
    # LLM to issue another shell command.  The executor still has its own
    # approval/hash checks and writes its normal artifacts under the dedicated
    # training-tuning output directory.
    from workflow.nodes.training_tuning.run import run as run_training

    training_args = argparse.Namespace(
        project_root=str(context.project_root),
        engine_root=str(args.engine_root) if getattr(args, "engine_root", None) else None,
        data=str(args.data) if getattr(args, "data", None) else None,
        data_contract=str(args.data_contract) if getattr(args, "data_contract", None) else None,
        node_config_dir=str(args.node_config_dir) if getattr(args, "node_config_dir", None) else None,
        output_dir=str(context.project_root / "outputs" / "training-tuning"),
        run_id="training-tuning",
        pmml_converter=None,
    )
    try:
        training_result = run_training(training_args)
    except BaseException as exc:
        # model-config invokes training-tuning in-process.  If the composite
        # executor fails after its data-preparation context has completed, no
        # outer runner catches the exception; explicitly close both state
        # streams so the UI does not leave a stale `running` node behind.
        failure_summary = f"模型训练与调参失败：{type(exc).__name__}: {exc}"
        context.progress.emit("model-config", "failed", summary=failure_summary)
        ProgressReporter(
            context.project_root / "outputs" / "training-tuning",
            run_id="training-tuning",
        ).emit("training-tuning", "failed", summary=failure_summary)
        raise
    training_summary = training_result.get("summary")
    selected_candidate = training_result.get("selected_candidate", "baseline")
    review_status = training_result.get("review_status") or "completed"
    review_path = training_result.get("review")
    review_summary = training_result.get("review_summary") or {}
    report_path = training_result.get("report")
    html_report_path = training_result.get("html_report")
    if not result_confirmed:
        pending_approval = write_json(
            node_approval_path(context, "model-config", for_write=True),
            {
                "node_id": "model-config",
                "status": "awaiting_user_confirmation",
                "config": str(config_path),
                "config_sha256": config_hash,
                "canonical_model_config": str(canonical_path),
                "canonical_sha256": canonical_hash,
                "sample_treatment": parameters["sample_treatment"],
                "training_completed": True,
                "result_summary": (
                    config_summary_text
                    + "\n\n"
                    + f"模型训练与调参已完成（当前候选 {selected_candidate}），模型审查状态：{review_status}。"
                ),
                "selected_candidate": selected_candidate,
                "review_status": review_status,
                "review_summary": review_summary,
                "review_path": review_path,
                "training_result": training_result,
                "artifacts": [
                    str(decision_summary_path),
                    str(Path(training_summary)) if training_summary else None,
                    str(Path(review_path)) if review_path else None,
                    str(Path(report_path)) if report_path else None,
                    str(Path(html_report_path)) if html_report_path else None,
                ],
            },
        )
        result_summary = (
            config_summary_text
            + "\n\n"
            + f"模型训练与调参已完成（当前候选 {selected_candidate}），模型审查状态：{review_status}。"
            + "\n请确认以上模型结果后再结束本次流程；如需调整，请直接提出文字修改要求。"
        )
        context.progress.emit(
            "model-config",
            "waiting_confirmation",
            summary=result_summary,
            artifacts=[
                pending_approval,
                decision_summary_path,
                *([Path(training_summary)] if training_summary else []),
                *([Path(review_path)] if review_path else []),
                *([Path(report_path)] if report_path else []),
                *([Path(html_report_path)] if html_report_path else []),
            ],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "model-config",
            # The full Markdown table is already carried by the single
            # waiting_confirmation progress event. Keep the JSON result short
            # so the host LLM does not render the same table twice.
            "summary": "模型训练与调参已完成，等待用户确认模型结果。",
            "model_review": review_path,
            "review_status": review_status,
            "review_summary": review_summary,
            "training": training_result,
            "approval": str(pending_approval),
            "decision_summary": str(decision_summary_path),
            "display_files": [
                str(decision_summary_path),
                *([str(training_summary)] if training_summary else []),
                *([str(html_report_path)] if html_report_path else []),
                *([str(report_path)] if report_path else []),
            ],
        }
    context.progress.emit(
        "model-config",
        "success",
        summary=(
            f"模型配置与训练完成（采用 {selected_candidate}）；"
            f"模型审查已完成（状态：{review_status}，发现 {review_summary.get('finding_count', 0)} 项）"
            + (
                "\n关键审查结论：\n"
                + "\n".join(f"- {item}" for item in review_summary.get("findings", []))
                if review_summary.get("findings")
                else "\n关键审查结论：未发现阈值型异常。"
            )
            + (
                "\n改进建议：\n"
                + "\n".join(f"- {item}" for item in review_summary.get("recommendations", []))
                if review_summary.get("recommendations")
                else ""
            )
        ),
        artifacts=[
            config_path,
            snapshot_path,
            canonical_path,
            approval_path,
            decision_summary_path,
            *([Path(training_summary)] if training_summary else []),
            *([Path(review_path)] if review_path else []),
            *([Path(report_path)] if report_path else []),
            *([Path(html_report_path)] if html_report_path else []),
        ],
    )
    return {
        "status": "success",
        "node_id": "model-config",
        "config": str(config_path),
        "canonical_model_config": str(canonical_path),
        "snapshot": str(snapshot_path),
        "approval": str(approval_path),
        "decision_summary": str(decision_summary_path),
        "training": training_result,
        "summary": (
            f"模型配置已生成并完成训练与调参，最终采用 {selected_candidate}；"
            f"模型审查状态：{review_status}；训练摘要：{training_summary}"
        ),
        "model_review": review_path,
        "review_status": review_status,
        "review_summary": review_summary,
        "sample_treatment": parameters["sample_treatment"],
        "display_files": [
            *([str(training_summary)] if training_summary else []),
            *([str(html_report_path)] if html_report_path else []),
            *([str(report_path)] if report_path else []),
        ],
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

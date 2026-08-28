#!/usr/bin/env python3
"""Command-line adapter for the repository's risk-modeling engine."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REQUIRED_PACKAGES = (
    "polars",
    "fastexcel",
    "yaml",
    "numpy",
    "scipy",
    "lightgbm",
    "optuna",
    "toad",
    "xlsxwriter",
)
REQUIRED_PROJECT_FILES = (
    "ai/approval.py",
    "ai/pipeline.py",
    "data/loader.py",
    "data/contract.py",
    "eda/pipeline.py",
    "modeling/pipeline.py",
    "preprocessing/feature_preprocessing.py",
    "preprocessing/sample_config.py",
    "preprocessing/sample_diagnostics.py",
    "workflow/definition.py",
    "workflow/__main__.py",
    "workflow/nodes/common.py",
    "workflow/nodes/data_read/run.py",
    "workflow/nodes/sample_diagnosis/run.py",
    "workflow/nodes/eda_analysis/run.py",
    "progress.py",
)

MAIN_NODE_ORDER = (
    "data-read",
    "eda-analysis",
    "sample-diagnosis",
    "feature-processing",
    "model-config",
    "training-tuning",
    "model-review",
    "report-delivery",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run risk-modeling EDA and LightGBM workflows"
    )
    parser.add_argument(
        "--mode",
        choices=("check", "prepare", "approve", "eda", "model", "all"),
        required=True,
    )
    parser.add_argument("--planned-mode", choices=("eda", "model", "all"))
    parser.add_argument(
        "--steps",
        help=(
            "Comma-separated user-facing node IDs to execute. When omitted, "
            "the selected mode keeps its legacy full behavior."
        ),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Built-in model_skills engine root; defaults to the installed Skill root",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-contract", type=Path)
    parser.add_argument("--sample-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--node-config-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirmation-request", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--confirmed-by")
    parser.add_argument(
        "--confirm-node",
        action="append",
        default=[],
        help="Explicitly confirm one selected node's YAML configuration (repeatable)",
    )
    parser.add_argument("--acknowledge-finding", action="append", default=[])
    parser.add_argument(
        "--node-executable",
        default=shutil.which("node") or "node",
        help="Legacy compatibility option; current XlsxWriter exporter does not use Node",
    )
    parser.add_argument(
        "--pmml-converter",
        type=Path,
        help="Optional JPMML-LightGBM executable JAR used to create a PMML artifact",
    )
    parser.add_argument(
        "--run-id", help="Stable identifier used by progress events across commands"
    )
    return parser


def _parse_steps(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    selected = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(selected) - set(MAIN_NODE_ORDER))
    if unknown:
        raise ValueError(
            "Unknown workflow node(s): " + ", ".join(unknown) +
            ". Valid IDs: " + ", ".join(MAIN_NODE_ORDER)
        )
    if not selected:
        raise ValueError("--steps must contain at least one workflow node ID")
    if "data-read" not in selected:
        selected = ("data-read", *selected)
    return tuple(node for node in MAIN_NODE_ORDER if node in selected)


def _resolved(path: Path, base: Path) -> Path:
    return (
        path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()
    )


def _scripts_root(engine_root: Path) -> Path:
    """Resolve the merged Python package directory for repo and installed layouts."""
    candidates = (
        engine_root / "scripts",
        engine_root / "risk-modeling-pipeline" / "scripts",
    )
    for candidate in candidates:
        if (candidate / "workflow" / "runner.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate workflow scripts under engine root: " + str(engine_root)
    )


def _environment_status(engine_root: Path) -> dict[str, Any]:
    scripts_root = _scripts_root(engine_root)
    package_status = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES
    }
    file_status = {
        name: (scripts_root / name).is_file() for name in REQUIRED_PROJECT_FILES
    }
    return {
        "engine_root": str(engine_root),
        "scripts_root": str(scripts_root),
        "packages": package_status,
        "project_files": file_status,
        "ready": all(package_status.values()) and all(file_status.values()),
    }


def _default_output(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / f"skill_run_{timestamp}"


def main() -> int:
    args = _parser().parse_args()
    selected_steps = _parse_steps(args.steps)
    selected_step_set = set(selected_steps or MAIN_NODE_ORDER)
    wants_eda_stage = selected_steps is None or bool(
        selected_step_set & {"eda-analysis", "sample-diagnosis", "feature-processing"}
    )
    wants_model_stage = selected_steps is None or bool(
        selected_step_set & {"model-config", "training-tuning", "model-review"}
    )
    if selected_steps is None:
        effective_mode = args.mode
    elif wants_eda_stage and wants_model_stage:
        effective_mode = "all"
    elif wants_model_stage:
        effective_mode = "model"
    else:
        effective_mode = "eda"
    project_root = args.project_root.expanduser().resolve()
    engine_root = args.engine_root.expanduser().resolve()
    status = _environment_status(engine_root)
    status["project_root"] = str(project_root)
    status["engine_root"] = str(engine_root)
    if args.mode == "check":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["ready"] else 1
    if not status["ready"]:
        print(json.dumps(status, ensure_ascii=False, indent=2), file=sys.stderr)
        raise RuntimeError(
            "Environment check failed; install dependencies or correct --engine-root"
        )

    scripts_root = _scripts_root(engine_root)
    sys.path.insert(0, str(scripts_root))
    from workflow.nodes.common import ensure_workspace_configs, resolve_engine_assets

    assets_dir = resolve_engine_assets(args)
    ensure_workspace_configs(
        project_root,
        assets_dir,
        set(selected_steps or MAIN_NODE_ORDER),
    )
    from ai.approval import create_approval_manifest, validate_approval_manifest
    from ai.pipeline import prepare_modeling_request
    from data.contract import apply_role_overrides, load_contract, validate_contract
    from data.loader import discover_data_file, load_table
    from eda.pipeline import run_eda
    from modeling.config import load_model_config
    from modeling.pipeline import run_model_pipeline
    from preprocessing.sample_config import load_sample_config
    from preprocessing.sample_diagnostics import apply_sample_treatment
    from progress import ProgressReporter
    from workflow.node_config import ensure_node_configs, populate_data_read_roles

    if args.mode == "approve":
        if args.confirmation_request is None or not args.confirmed_by:
            raise ValueError(
                "approve requires --confirmation-request and --confirmed-by"
            )
        request_path = _resolved(args.confirmation_request, project_root)
        progress = ProgressReporter(request_path.parent, run_id=args.run_id)
        approval_target = (
            _resolved(args.approval_file, project_root)
            if args.approval_file
            else request_path.with_name("approval_manifest.json")
        )
        with progress.track(
            "approval",
            running_summary="正在校验用户确认和阻断项确认",
            success_summary=f"用户 {args.confirmed_by} 已确认建模配置",
            artifacts=[approval_target],
        ):
            approval_path = create_approval_manifest(
                request_path,
                confirmed_by=args.confirmed_by,
                acknowledged_findings=set(args.acknowledge_finding),
                confirmed_nodes=set(args.confirm_node),
                output_path=approval_target,
            )
        print(
            json.dumps(
                {"status": "approved", "approval_file": str(approval_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    contract_path = _resolved(
        args.data_contract or Path("configs/data_contract.yaml"), project_root
    )
    raw_contract = load_contract(contract_path)
    if args.data:
        data_path = _resolved(args.data, project_root)
    elif raw_contract.input_path:
        data_path = _resolved(Path(raw_contract.input_path), contract_path.parent)
    else:
        data_path = discover_data_file(project_root)
    planned_mode = args.planned_mode if args.mode == "prepare" else effective_mode
    if args.mode == "prepare" and selected_steps is not None:
        if args.planned_mode not in {None, "all", effective_mode}:
            raise ValueError(
                f"--steps selects {effective_mode!r}, but --planned-mode is {args.planned_mode!r}"
            )
        planned_mode = effective_mode
    wants_model_config = wants_model_stage
    model_config_path = None
    model_config = None
    if planned_mode in ("model", "all") and wants_model_config:
        model_config_path = _resolved(
            args.model_config or Path("configs/model_config.yaml"), project_root
        )
        model_config = load_model_config(model_config_path)
    node_config_dir = _resolved(
        args.node_config_dir or Path("configs/node_configs"), project_root
    )
    node_config_ids = selected_steps or MAIN_NODE_ORDER
    node_configs = ensure_node_configs(
        node_config_dir, node_config_ids, template_dir=assets_dir
    )
    # Standalone v1 stores sample thresholds and treatment actions in the
    # sample-diagnosis node YAML. Keep accepting an explicitly supplied or
    # legacy sample_config.yaml for batch compatibility.
    if args.sample_config:
        sample_config_path = _resolved(args.sample_config, project_root)
    elif (
        node_configs.get("sample-diagnosis") is not None
        and "diagnostics" in node_configs["sample-diagnosis"].parameters
        and "treatment" in node_configs["sample-diagnosis"].parameters
    ):
        sample_config_path = node_configs["sample-diagnosis"].path
    else:
        sample_config_path = _resolved(Path("configs/sample_config.yaml"), project_root)
    sample_config = load_sample_config(sample_config_path)
    data_read_parameters = node_configs["data-read"].parameters
    # Role names may be supplied by data-read.yaml. Null values fall back to
    # data_contract.yaml; the effective contract is used consistently by all
    # downstream stages.
    effective_contract = apply_role_overrides(raw_contract, data_read_parameters)

    def load_configured_data():
        options: dict[str, Any] = {}
        if data_path.suffix.lower() == ".csv":
            if data_read_parameters.get("encoding") is not None:
                options["encoding"] = data_read_parameters["encoding"]
            options["infer_schema_length"] = data_read_parameters.get(
                "infer_schema_length", 10000
            )
        elif data_path.suffix.lower() in {".xls", ".xlsx"}:
            options["sheet_name"] = data_read_parameters.get("sheet_name", 0)
        return load_table(data_path, **options)

    output_dir = (
        _resolved(args.output_dir, project_root)
        if args.output_dir
        else _default_output(project_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep a run directory easy to scan: executable artifacts are grouped into
    # models, reports, and artifacts.  The internal pipeline still receives a
    # dedicated artifact directory so its intermediate CSV/Parquet/JSON files
    # never sit beside the deliverables.
    execution_mode = args.mode in ("eda", "model", "all")
    artifacts_dir = output_dir / "artifacts" if execution_mode else output_dir
    reports_dir = output_dir / "reports" if execution_mode else output_dir
    models_dir = output_dir / "models" if execution_mode else output_dir
    for directory in (artifacts_dir, reports_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(artifacts_dir, run_id=args.run_id)

    if args.mode in ("eda", "model", "all"):
        if args.approval_file is None:
            raise ValueError(
                "Execution requires --approval-file. Run prepare, obtain explicit user confirmation, then run approve."
            )
        approval_path = _resolved(args.approval_file, project_root)
        validate_approval_manifest(
            approval_path,
            planned_mode=effective_mode,
            data_path=data_path,
            contract_path=contract_path,
            sample_config_path=sample_config_path,
            model_config_path=model_config_path,
            require_model_config=wants_model_config,
            selected_steps=selected_steps,
            node_config_paths={
                node_id: config.path for node_id, config in node_configs.items()
            },
        )
        progress.emit(
            "approval",
            "success",
            summary="批准清单哈希校验通过",
            artifacts=[approval_path],
        )

    if args.mode == "prepare":
        with progress.track(
            "loader",
            running_summary="正在读取数据并校验数据契约",
            success_summary="原始数据读取完成，节点参数建议已生成",
        ):
            data = load_configured_data().data
            # Bootstrap read: role fields can still be null in the template.
            # Fill only null values, then reload the YAML so the effective
            # roles used below are exactly what the user can review/edit.
            if populate_data_read_roles(
                node_configs["data-read"], data, raw_contract
            ):
                node_configs = ensure_node_configs(node_config_dir, node_config_ids)
                data_read_parameters = node_configs["data-read"].parameters
                effective_contract = apply_role_overrides(
                    raw_contract, data_read_parameters
                )
            contract = validate_contract(
                data,
                effective_contract,
                allow_id_duplicates=bool(
                    data_read_parameters.get("allow_id_duplicates", True)
                ),
                allow_target_issues=bool(
                    data_read_parameters.get("allow_target_issues", True)
                ),
            )
    else:
        with progress.track(
            "loader",
            running_summary="正在读取数据并校验数据契约",
            success_summary="数据读取和格式校验完成",
        ):
            raw_data = load_configured_data().data
            validate_contract(
                raw_data,
                effective_contract,
                allow_id_duplicates=bool(
                    data_read_parameters.get("allow_id_duplicates", True)
                ),
                allow_target_issues=bool(
                    data_read_parameters.get("allow_target_issues", True)
                ),
            )
        try:
            treatment = apply_sample_treatment(raw_data, effective_contract, sample_config)
        except Exception as exc:
            progress.emit(
                "cleaning",
                "failed",
                summary=f"样本处理配置无法执行：{str(exc) or exc.__class__.__name__}",
            )
            raise
        data = treatment.data
        contract = validate_contract(data, effective_contract)
        treatment_path = artifacts_dir / "sample_treatment_result.json"
        treatment_path.write_text(
            json.dumps(treatment.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.copy2(sample_config_path, artifacts_dir / "sample_config.yaml")

    if args.mode == "prepare":
        if planned_mode is None:
            raise ValueError("prepare requires --planned-mode")
        prepared = prepare_modeling_request(
            data,
            contract,
            planned_mode=planned_mode,
            data_path=data_path,
            contract_path=contract_path,
            model_config=model_config,
            model_config_path=model_config_path,
            sample_config=sample_config,
            sample_config_path=sample_config_path,
            output_dir=output_dir,
            progress=progress,
            selected_steps=selected_steps,
            node_configs=node_configs,
        )
        request = json.loads(
            prepared.confirmation_request_path.read_text(encoding="utf-8")
        )
        progress.emit(
            "approval",
            "waiting_confirmation",
            summary="等待用户确认字段角色、泄漏排除、特征处理、稳定性、数据切分和训练参数",
            artifacts=[prepared.confirmation_request_path],
        )
        print(
            json.dumps(
                {
                    "status": "awaiting_user_confirmation",
                    "planned_mode": planned_mode,
                    "confirmation_request": str(prepared.confirmation_request_path),
                    "blocker_codes": request["blocker_codes"],
                    "review_artifacts": request["review_artifacts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    result: dict[str, Any] = {
        "mode": effective_mode,
        "input": str(data_path),
        "output_dir": str(output_dir),
    }
    if args.mode != "prepare":
        result["sample_treatment"] = str(treatment_path)
    run_eda_stage = effective_mode in ("eda", "all") and wants_eda_stage
    run_model_stage = effective_mode in ("model", "all") and wants_model_stage
    # In the standalone flow EDA and sample diagnosis are separate. Keep the
    # historical fallback only for callers that explicitly use the legacy
    # batch runner with sample-diagnosis but no EDA node config.
    eda_config = node_configs.get("eda-analysis") or node_configs.get("sample-diagnosis")
    eda_parameters = eda_config.parameters if eda_config is not None else {}
    generate_eda_report = (
        (
            selected_steps is None
            or bool(selected_step_set & {"eda-analysis", "sample-diagnosis"})
        )
        and bool(eda_parameters.get("generate_report", True))
    )
    generate_model_report = selected_steps is None or "report-delivery" in selected_step_set
    if run_eda_stage:
        eda_result = run_eda(
            data,
            contract,
            artifacts_dir / "eda",
            node_executable=args.node_executable,
            progress=progress,
            report_dir=reports_dir,
            generate_report=generate_eda_report,
            analysis_config=eda_parameters,
        )
        if eda_result.report_path.is_file():
            result["eda_report"] = str(eda_result.report_path)
    if run_model_stage:
        assert model_config_path is not None and model_config is not None
        model_result = run_model_pipeline(
            data,
            contract,
            model_config,
            artifacts_dir / "model",
            node_executable=args.node_executable,
            config_path=model_config_path,
            progress=progress,
            balance_classes=sample_config.treatment.class_imbalance_action
            == "class_weight",
            pmml_converter=(
                _resolved(args.pmml_converter, project_root)
                if args.pmml_converter
                else None
            ),
            model_dir=models_dir,
            report_dir=reports_dir,
            generate_report=generate_model_report,
        )
        if model_result.report_path.is_file():
            result["model_report"] = str(model_result.report_path)
        result["selected_feature_count"] = len(model_result.selected_features)
        result["metrics"] = model_result.metrics.to_dicts()
        model_summary_path = model_result.output_dir / "model_summary.json"
        if model_summary_path.is_file():
            model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
            result["model_artifacts"] = model_summary.get("model_artifacts", {})
            review_path = model_result.output_dir / (
                model_summary.get("review_artifact") or "ai_model_review.json"
            )
            review_body = (
                json.loads(review_path.read_text(encoding="utf-8"))
                if review_path.is_file()
                else {}
            )
            result["model_review"] = {
                "status": model_summary.get("review_status"),
                "finding_count": model_summary.get("review_finding_count", 0),
                "blocker_count": model_summary.get("review_blocker_count", 0),
                "warning_count": model_summary.get("review_warning_count", 0),
                "recommendation_count": model_summary.get(
                    "review_recommendation_count", 0
                ),
                "diagnosis": review_body.get("diagnosis", []),
                "recommendations": review_body.get("recommendations", []),
                "artifact": str(review_path),
            }
    if selected_steps is not None:
        result["selected_steps"] = list(selected_steps)
        result["reports_generated"] = {
            "eda": generate_eda_report,
            "model": generate_model_report,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

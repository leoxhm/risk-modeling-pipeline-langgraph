"""Prepare AI advisory artifacts and a hash-bound confirmation request."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from data.contract import ValidatedDataContract
from data.profiler import profile_columns
from modeling.config import ModelConfig
from modeling.feature_selection import select_features
from modeling.split import split_dataset
from preprocessing.cleaning import preprocess_data
from preprocessing.feature_preprocessing import fit_feature_preprocessor
from preprocessing.sample_config import SampleConfig
from preprocessing.sample_diagnostics import build_sample_diagnostics
from progress import ProgressReporter
from workflow.node_config import NodeConfig

from .approval import canonical_sha256, file_sha256
from .experiment_planner import build_experiment_plan
from .preflight_advisor import build_preflight_report
from .schema_assistant import build_schema_proposal


@dataclass(frozen=True)
class PreparationResult:
    output_dir: Path
    confirmation_request_path: Path
    preflight_path: Path
    schema_proposal_path: Path
    experiment_plan_path: Path
    sample_diagnostics_path: Path
    feature_preprocessing_path: Path | None
    feature_selection_path: Path | None
    model_config_proposal_path: Path | None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_modeling_request(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    *,
    planned_mode: str,
    data_path: str | Path,
    contract_path: str | Path,
    model_config: ModelConfig | None,
    model_config_path: str | Path | None,
    sample_config: SampleConfig,
    sample_config_path: str | Path,
    output_dir: str | Path,
    progress: ProgressReporter | None = None,
    selected_steps: tuple[str, ...] | None = None,
    node_configs: dict[str, NodeConfig] | None = None,
) -> PreparationResult:
    """Create read-only diagnostics; do not create an approval or run a model."""

    if planned_mode not in {"eda", "model", "all"}:
        raise ValueError("planned_mode must be eda, model, or all")
    if planned_mode in {"model", "all"} and (
        model_config is None or model_config_path is None
    ):
        raise ValueError(
            "Model preparation requires model_config and model_config_path"
        )

    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = run_dir / "ai_preflight.json"
    schema_path = run_dir / "ai_schema_proposal.json"
    experiment_path = run_dir / "ai_experiment_plan.json"
    profile_path = run_dir / "column_profile.csv"
    sample_diagnostics_path = run_dir / "sample_diagnostics.json"
    feature_preprocessing_path = run_dir / "feature_preprocessing.csv"
    feature_selection_path = run_dir / "feature_selection.csv"
    feature_proposal_path = run_dir / "feature_proposal.json"
    model_config_proposal_path = run_dir / "model_config_proposal.yaml"

    profile_stage = (
        progress.track(
            "profiler",
            running_summary="正在计算字段类型、缺失率和唯一率",
            success_summary="字段画像已生成",
            artifacts=[profile_path],
        )
        if progress
        else nullcontext()
    )
    with profile_stage:
        profile = profile_columns(data)
        profile.write_csv(profile_path)

    sample_diagnostics = build_sample_diagnostics(
        data, contract.contract, sample_config
    )
    _write_json(sample_diagnostics_path, sample_diagnostics)

    schema_stage = (
        progress.track(
            "schema-review",
            running_summary="正在生成字段角色和泄漏风险建议",
            success_summary="字段语义审查已生成",
            artifacts=[schema_path],
        )
        if progress
        else nullcontext()
    )
    with schema_stage:
        schema_proposal = build_schema_proposal(profile, contract)
        _write_json(schema_path, schema_proposal)

    preflight_stage = (
        progress.track(
            "preflight",
            running_summary="正在检查样本、时间窗口和阻断项",
            success_summary="入模诊断已生成",
            artifacts=[sample_diagnostics_path, preflight_path],
        )
        if progress
        else nullcontext()
    )
    with preflight_stage:
        preflight = build_preflight_report(
            data,
            contract,
            profile,
            model_config,
            sample_diagnostics,
        )
        _write_json(preflight_path, preflight)

    if planned_mode in {"model", "all"} and model_config is not None:
        experiment_stage = (
            progress.track(
                "experiment-plan",
                running_summary="正在规划数据切分、验证和调参方案",
                success_summary="实验规划已生成",
                artifacts=[experiment_path],
            )
            if progress
            else nullcontext()
        )
        with experiment_stage:
            experiment_plan = build_experiment_plan(planned_mode, preflight, model_config)
            _write_json(experiment_path, experiment_plan)
    elif progress:
        progress.emit("experiment-plan", "skipped", summary="当前节点选择不包含模型配置")

    feature_preview: dict[str, Any] | None = None
    if model_config is not None:
        model_config_proposal_path.write_text(
            yaml.safe_dump(asdict(model_config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        feature_stage = (
            progress.track(
                "feature-selection",
                running_summary="正在使用 Train-only 规则生成特征预处理和筛选预览",
                success_summary="特征预处理和筛选预览已生成",
                artifacts=[
                    feature_preprocessing_path,
                    feature_selection_path,
                    feature_proposal_path,
                ],
            )
            if progress
            else nullcontext()
        )
        with feature_stage:
            # Advisory and read-only. Approved execution repeats the fit after
            # confirmed sample treatment, so this cannot change the model run.
            cleaning = preprocess_data(data, contract, filter_features=False)
            cleaned_contract = replace(contract, feature_cols=cleaning.feature_cols)
            split = split_dataset(cleaning.data, cleaned_contract, model_config.split)
            preprocessing_plan = fit_feature_preprocessor(
                split.train, cleaned_contract, model_config.feature_preprocessing
            )
            preprocessed_contract = replace(
                cleaned_contract, feature_cols=preprocessing_plan.retained_features
            )
            selection = select_features(
                split.train, preprocessed_contract, model_config.feature_selection
            )
            preprocessing_plan.decisions.write_csv(feature_preprocessing_path)
            selection.decisions.write_csv(feature_selection_path)
            feature_preview = {
                "status": "preview_requires_sample_confirmation",
                "sample_treatment_applied": False,
                "fit_scope": "train_only",
                "train_rows": split.train.height,
                "test_rows": split.test.height,
                "oot_rows": split.oot.height,
                "retained_features": list(selection.feature_cols),
                "dropped_by_type": list(preprocessing_plan.dropped_features),
                "preprocessing_decisions": preprocessing_plan.decisions.to_dicts(),
                "selection_decisions": selection.decisions.to_dicts(),
            }
            _write_json(feature_proposal_path, feature_preview)

    artifacts = {
        "data": {
            "path": str(Path(data_path).expanduser().resolve()),
            "sha256": file_sha256(data_path),
        },
        "data_contract": {
            "path": str(Path(contract_path).expanduser().resolve()),
            "sha256": file_sha256(contract_path),
        },
    }
    if model_config_path is not None:
        artifacts["model_config"] = {
            "path": str(Path(model_config_path).expanduser().resolve()),
            "sha256": file_sha256(model_config_path),
        }
    artifacts["sample_config"] = {
        "path": str(Path(sample_config_path).expanduser().resolve()),
        "sha256": file_sha256(sample_config_path),
    }
    if node_configs:
        artifacts["node_configs"] = {
            node_id: {
                "path": str(config.path),
                "sha256": file_sha256(config.path),
            }
            for node_id, config in node_configs.items()
        }
    blocker_codes = sorted(
        {
            finding["code"]
            for report in (preflight, schema_proposal)
            for finding in report.get("findings", [])
            if finding.get("severity") == "blocker"
        }
    )
    request_body = {
        "schema_version": 1,
        "status": "awaiting_user_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_mode": planned_mode,
        "selected_steps": list(selected_steps) if selected_steps is not None else None,
        "artifacts": artifacts,
        "blocker_codes": blocker_codes,
        "confirmed_roles": {
            "id_cols": list(contract.contract.id_cols),
            "date_col": contract.contract.date_col,
            "target_col": contract.contract.target_col,
            "good_label": contract.contract.good_label,
            "bad_label": contract.contract.bad_label,
            "exclude_cols": list(contract.contract.exclude_cols),
            "feature_count": len(contract.feature_cols),
        },
        "sample_treatment": sample_diagnostics["proposed_treatment"],
        "sample_findings": sample_diagnostics["findings"],
        "feature_policy": (
            {
                "preprocessing": asdict(model_config.feature_preprocessing),
                "selection": asdict(model_config.feature_selection),
                "fit_scope": "train_only",
            }
            if model_config is not None
            else None
        ),
        "review_artifacts": {
            "sample_diagnostics": str(sample_diagnostics_path),
            "preflight": str(preflight_path),
            "schema_proposal": str(schema_path),
            "column_profile": str(profile_path),
        },
    }
    if experiment_path.is_file():
        request_body["review_artifacts"]["experiment_plan"] = str(experiment_path)
    if node_configs:
        request_body["review_artifacts"]["node_configs"] = {
            node_id: str(config.path) for node_id, config in node_configs.items()
        }
    if feature_preview is not None:
        request_body["review_artifacts"].update(
            {
                "feature_proposal": str(feature_proposal_path),
                "feature_preprocessing": str(feature_preprocessing_path),
            "feature_selection": str(feature_selection_path),
            "model_config_proposal": str(model_config_proposal_path),
            }
        )
        request_body["feature_preview"] = {
            "fit_scope": feature_preview["fit_scope"],
            "train_rows": feature_preview["train_rows"],
            "test_rows": feature_preview["test_rows"],
            "oot_rows": feature_preview["oot_rows"],
            "retained_feature_count": len(feature_preview["retained_features"]),
            "dropped_by_type": feature_preview["dropped_by_type"],
        }
    request_body["request_id"] = canonical_sha256(request_body)
    request_path = run_dir / "confirmation_request.json"
    _write_json(request_path, request_body)
    return PreparationResult(
        run_dir,
        request_path,
        preflight_path,
        schema_path,
        experiment_path,
        sample_diagnostics_path,
        feature_preprocessing_path if feature_preview is not None else None,
        feature_selection_path if feature_preview is not None else None,
        model_config_proposal_path if model_config is not None else None,
    )

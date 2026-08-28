"""Feature statistics, user-confirmed selection, and preprocessing node."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from ..common import NodeContext, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the feature-processing workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--confirm-config", action="store_true")
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_parameters(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parameters = raw.get("parameters", raw)
    if not isinstance(parameters, dict):
        raise ValueError(f"Invalid feature-processing parameters: {path}")
    return parameters


def _config_objects(parameters: dict[str, Any]):
    from modeling.config import (
        CategoricalPreprocessingConfig,
        FeaturePreprocessingConfig,
        FeatureSelectionConfig,
        NumericPreprocessingConfig,
        TextPreprocessingConfig,
    )

    selection = parameters.get("selection", {})
    preprocessing = parameters.get("preprocessing", {})
    numeric = preprocessing.get("numeric", {})
    categorical = preprocessing.get("categorical", {})
    text = preprocessing.get("text", {})
    selection_config = FeatureSelectionConfig(
        min_iv=float(selection.get("min_iv", 0.02)),
        max_correlation=float(selection.get("max_correlation", 0.80)),
        max_missing_rate=float(selection.get("max_missing_rate", 0.80)),
        max_dominant_rate_warning=float(selection.get("max_dominant_rate_warning", 0.95)),
        max_psi=float(selection.get("max_psi", 0.25)),
        max_unstable_month_ratio=float(selection.get("max_unstable_month_ratio", 0.30)),
        min_month_samples=int(selection.get("min_month_samples", 100)),
        stability_action=str(selection.get("stability_action", "review")),
        correlation_method=str(selection.get("correlation_method", "spearman")),
    )
    preprocessing_config = FeaturePreprocessingConfig(
        numeric=NumericPreprocessingConfig(
            invalid_to_null=bool(numeric.get("invalid_to_null", True)),
            missing_strategy=str(numeric.get("missing_strategy", "native")),
        ),
        categorical=CategoricalPreprocessingConfig(
            strategy=str(categorical.get("strategy", "lightgbm_native")),
            rare_min_count=int(categorical.get("rare_min_count", 20)),
            rare_min_rate=float(categorical.get("rare_min_rate", 0.001)),
            unknown_action=str(categorical.get("unknown_action", "missing")),
            near_unique_rate=float(categorical.get("near_unique_rate", 0.98)),
            max_categories=int(categorical.get("max_categories", 100)),
            high_cardinality_action=str(categorical.get("high_cardinality_action", "drop")),
        ),
        text=TextPreprocessingConfig(
            action=str(text.get("action", "drop")),
            min_average_length=int(text.get("min_average_length", 64)),
        ),
    )
    return selection_config, preprocessing_config


def _require_approval(context: NodeContext, node_id: str) -> tuple[dict[str, Any], Path]:
    path = context.node_config_dir / f"{node_id}.approval.json"
    if not path.is_file():
        raise RuntimeError(
            f"{node_id}.yaml has not been confirmed. Complete the {node_id} node with --confirm-config first."
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    config_path = context.node_config_dir / f"{node_id}.yaml"
    if approval.get("config_sha256") != _hash(config_path):
        raise RuntimeError(f"{config_path.name} changed after confirmation; rerun {node_id}.")
    return approval, config_path


def _load_eda_parameters(context: NodeContext) -> dict[str, Any]:
    path = context.node_config_dir / "eda-analysis.yaml"
    if not path.is_file():
        return {}
    return _read_parameters(path)


def _prepare_data(context: NodeContext, data: pl.DataFrame, validated):
    from preprocessing.cleaning import preprocess_data

    # Add event_month/event_date without dropping candidate features. This is
    # needed by the stability/PSI calculation and is not a modeling split.
    cleaned = preprocess_data(
        data,
        validated,
        missing_rate_threshold=1.0,
        constant_rate_threshold=1.0,
        duplicate_strategy="error",
        filter_features=False,
    )
    return cleaned.data, replace(validated, feature_cols=validated.feature_cols)


def _apply_confirmed_sample_policy(context: NodeContext, data: pl.DataFrame, validated):
    approval_path = context.node_config_dir / "sample-diagnosis.approval.json"
    if not approval_path.is_file():
        return data, validated, None
    from data.contract import validate_contract
    from preprocessing.sample_config import load_sample_config
    from preprocessing.sample_diagnostics import apply_sample_treatment

    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    policy_path = Path(approval.get("policy_config") or approval.get("sample_config", ""))
    if not policy_path.is_absolute():
        policy_path = (context.project_root / policy_path).resolve()
    if not policy_path.is_file():
        raise RuntimeError("Confirmed sample policy configuration cannot be found.")
    sample_config = load_sample_config(policy_path)
    treatment = apply_sample_treatment(data, validated.contract, sample_config)
    strict = validate_contract(
        treatment.data,
        validated.contract,
        allow_id_duplicates=False,
        allow_target_issues=False,
    )
    return treatment.data, strict, {"path": str(policy_path), "sha256": _hash(policy_path), "policy": asdict(sample_config.treatment)}


def _distribution_summary(stats: pl.DataFrame) -> dict[str, Any]:
    """Return compact ranges for the UI while retaining the full CSV evidence."""
    result: dict[str, Any] = {}
    for column in ("missing_rate", "iv", "ks", "max_monthly_psi"):
        if column not in stats.columns or stats.height == 0:
            result[column] = {"min": None, "max": None, "mean": None}
            continue
        values = stats.get_column(column).drop_nulls().cast(pl.Float64)
        result[column] = {
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
        }
    return result


def _write_markdown_table(data: pl.DataFrame, path: Path, title: str) -> Path:
    """Write a human-readable evidence artifact for OpenCode/user review."""
    headers = data.columns
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data.rows():
        values = ["" if value is None else str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_selection_summary(decisions: pl.DataFrame, dropped_by_type: list[str], path: Path) -> Path:
    """Write an explicit retained/excluded/review list for human confirmation."""
    rows = decisions.to_dicts()
    excluded = [row for row in rows if str(row.get("decision", "")).startswith("excluded")]
    review = [row for row in rows if row.get("decision") == "retained_review"]
    retained = [row for row in rows if row.get("decision") == "retained"]
    lines = ["# 特征筛选结果（确认后生成）", "", f"- 保留字段：{len(retained)}", f"- 需复核字段：{len(review)}", f"- 按质量/IV/相关性/稳定性剔除：{len(excluded)}", f"- 按变量类型预处理剔除：{len(dropped_by_type)}", ""]
    for title, items in (("保留字段", retained), ("需复核字段", review), ("剔除字段", excluded)):
        lines.extend([f"## {title}", "", "| 字段 | 决策 | 原因 |", "| --- | --- | --- |"])
        if not items:
            lines.append("| （无） | - | - |")
        for row in items:
            reason = str(row.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {row.get('feature', '')} | {row.get('decision', '')} | {reason} |")
        lines.append("")
    if dropped_by_type:
        lines.extend(["## 变量类型预处理剔除", "", "| 字段 | 原因 |", "| --- | --- |"])
        for feature in dropped_by_type:
            decision = next((row for row in rows if row.get("feature") == feature), {})
            lines.append(f"| {feature} | {decision.get('reason', '变量类型策略剔除')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "feature-processing")
    from data.contract import validate_contract
    from eda.analytics import build_eda_analysis
    from modeling.feature_selection import select_features
    from preprocessing.feature_preprocessing import fit_feature_preprocessor, transform_features
    from reporting.feature_report import write_feature_report

    _require_approval(context, "data-read")
    _require_approval(context, "eda-analysis")
    with context.progress.track("loader", running_summary="正在按已确认配置重新读取数据", success_summary="特征处理数据读取完成"):
        loaded = context.reload_data()
        validated = validate_contract(loaded.data, context.effective_contract, allow_id_duplicates=True, allow_target_issues=True)
        working_data, working_contract = _prepare_data(context, loaded.data, validated)

    eda_params = _load_eda_parameters(context)
    analysis = build_eda_analysis(
        working_data,
        working_contract,
        month_col=str(eda_params.get("month_col", "event_month")),
        bin_count=int(eda_params.get("bin_count", 10)),
        ks_bucket=int(eda_params.get("ks_bucket", eda_params.get("bin_count", 10))),
        ks_method=str(eda_params.get("ks_method", "quantile")),
        baseline_month=eda_params.get("psi_base_month"),
        max_categories=int(eda_params.get("max_categories", 20)),
        correlation_method=str(eda_params.get("correlation_method", "pearson")),
        metrics_backend=str(eda_params.get("metrics_backend", "toad")),
    )
    overview = analysis.univariate_overview
    psi_summary = (
        analysis.monthly_psi.group_by("feature")
        .agg(pl.col("psi").max().alias("max_monthly_psi"))
        if analysis.monthly_psi.height
        else pl.DataFrame({"feature": [], "max_monthly_psi": []})
    )
    type_summary = pl.DataFrame(
        {
            "feature": list(working_contract.feature_cols),
            "variable_type": [str(working_data.get_column(feature).dtype) for feature in working_contract.feature_cols],
        }
    )
    stats = type_summary.join(overview, on="feature", how="left").join(psi_summary, on="feature", how="left")
    stats_path = context.output_dir / "feature_statistics.csv"
    stats.write_csv(stats_path)
    statistics_preview = _write_markdown_table(
        stats,
        context.output_dir / "feature_statistics.md",
        "字段级特征统计（请确认）",
    )
    correlation_path = context.output_dir / "feature_correlations.csv"
    analysis.correlation_pairs.write_csv(correlation_path)
    feature_report_path = write_feature_report(
        stats,
        analysis.correlation_pairs,
        context.output_dir / "feature_processing_report.html",
    )
    summary_path = write_json(
        context.output_dir / "feature_statistics_summary.json",
        {
            "node_id": "feature-processing",
            "row_count": working_data.height,
            "feature_count": len(working_contract.feature_cols),
            "statistics": str(stats_path),
            "statistics_preview": str(statistics_preview),
            "correlations": str(correlation_path),
            "html_report": str(feature_report_path),
            "metrics": {"missing_rate": True, "iv": True, "ks": True, "stability_psi": True, "correlation": True},
            "distribution": _distribution_summary(stats),
            "correlation_pair_count": analysis.correlation_pairs.height,
        },
    )
    config_path = context.config.path
    confirmation = write_json(
        context.output_dir / "feature_processing_confirmation.json",
        {
            "node_id": "feature-processing",
            "status": "confirmed" if args.confirm_config else "awaiting_user_confirmation",
            "message": "请确认 feature-processing.yaml 中的筛选阈值和不同变量类型的预处理方法。确认前只统计证据，不生成处理数据；确认后才生成 processed_data.parquet。",
            "config_path": str(config_path),
            "config_sha256": _hash(config_path),
            "data_read_config_sha256": _hash(context.data_read_config.path),
            "eda_config_sha256": _hash(context.node_config_dir / "eda-analysis.yaml"),
            "statistics": str(stats_path),
            "correlations": str(correlation_path),
            "distribution": _distribution_summary(stats),
            "correlation_pair_count": analysis.correlation_pairs.height,
            "display_files": [str(config_path)],
            "parameters": context.config.parameters,
        },
    )
    if not args.confirm_config:
        context.progress.emit("feature-processing", "waiting_confirmation", summary="等待用户确认特征筛选阈值和预处理方法", artifacts=[stats_path, statistics_preview, correlation_path, feature_report_path, summary_path, confirmation])
        return {"status": "awaiting_user_confirmation", "node_id": "feature-processing", "statistics": str(stats_path), "statistics_preview": str(statistics_preview), "correlations": str(correlation_path), "html_report": str(feature_report_path), "config": str(config_path), "distribution": _distribution_summary(stats), "display_files": [str(config_path), str(feature_report_path)], "confirmation": str(confirmation)}

    selected_config, preprocessing_config = _config_objects(context.config.parameters)
    model_data, model_contract, sample_policy = _apply_confirmed_sample_policy(context, loaded.data, validated)
    model_data, model_contract = _prepare_data(context, model_data, model_contract)
    selection = select_features(model_data, model_contract, selected_config)
    selected_contract = replace(model_contract, feature_cols=selection.feature_cols)
    plan = fit_feature_preprocessor(model_data, selected_contract, preprocessing_config)
    processed = transform_features(model_data, plan)
    output_params = context.config.parameters.get("output", {})
    processed_path = context.output_dir / "processed_data.parquet"
    if bool(output_params.get("write_processed_data", True)):
        processed.write_parquet(processed_path)
    selection_path = context.output_dir / "feature_selection.csv"
    selection.decisions.write_csv(selection_path)
    preprocessing_path = write_json(context.output_dir / "feature_preprocessing_plan.json", plan.as_dict())
    selection_summary = _write_selection_summary(
        selection.decisions,
        list(plan.dropped_features),
        context.output_dir / "feature_selection_summary.md",
    )
    selection_summary_json = write_json(
        context.output_dir / "feature_selection_summary.json",
        {
            "retained": list(selection.feature_cols),
            "excluded_by_selection": [
                row["feature"]
                for row in selection.decisions.to_dicts()
                if str(row.get("decision", "")).startswith("excluded")
            ],
            "retained_review": [
                row["feature"]
                for row in selection.decisions.to_dicts()
                if row.get("decision") == "retained_review"
            ],
            "excluded_by_type_preprocessing": list(plan.dropped_features),
            "summary_artifact": str(selection_summary),
        },
    )
    manifest = write_json(
        context.output_dir / "feature_processing_manifest.json",
        {
            "node_id": "feature-processing",
            "status": "success",
            "config": str(config_path),
            "config_sha256": _hash(config_path),
            "statistics": str(stats_path),
            "statistics_preview": str(statistics_preview),
            "html_report": str(feature_report_path),
            "feature_selection": str(selection_path),
            "feature_selection_summary": str(selection_summary),
            "feature_selection_summary_json": str(selection_summary_json),
            "preprocessing_plan": str(preprocessing_path),
            "processed_data": str(processed_path) if processed_path.is_file() else None,
            "sample_policy": sample_policy,
            "retained_features": list(selection.feature_cols),
        },
    )
    approval = write_json(
        context.node_config_dir / "feature-processing.approval.json",
        {"node_id": "feature-processing", "status": "approved", "config": str(config_path), "config_sha256": _hash(config_path), "manifest": str(manifest), "processed_data": str(processed_path) if processed_path.is_file() else None},
    )
    context.progress.emit("feature-processing", "success", summary="特征筛选与预处理完成，已生成处理数据和字段去留清单", artifacts=[stats_path, selection_path, selection_summary, selection_summary_json, preprocessing_path, feature_report_path, manifest, approval, *([processed_path] if processed_path.is_file() else [])])
    return {"status": "success", "node_id": "feature-processing", "statistics": str(stats_path), "feature_selection": str(selection_path), "feature_selection_summary": str(selection_summary), "feature_selection_summary_json": str(selection_summary_json), "preprocessing_plan": str(preprocessing_path), "html_report": str(feature_report_path), "processed_data": str(processed_path) if processed_path.is_file() else None, "manifest": str(manifest), "approval": str(approval), "display_files": [str(config_path), str(feature_report_path)]}


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

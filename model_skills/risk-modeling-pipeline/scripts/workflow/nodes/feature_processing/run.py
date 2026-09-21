"""Feature statistics, preprocessing, and selection node.

The node first computes feature evidence and recommendations, then pauses for
result confirmation. Only the confirmed pass applies the YAML and writes the
processed data. ``--confirm-config`` remains a backwards-compatible alias for
the explicit confirmation pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from ..common import (
    NodeContext,
    approval_path as node_approval_path,
    write_json,
    write_node_summary,
)


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
    parser.add_argument(
        "--confirm-result",
        action="store_true",
        help="Confirm the completed feature-processing result",
    )
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
    path = node_approval_path(context, node_id)
    if not path.is_file():
        if node_id == "eda-analysis":
            raise RuntimeError("EDA has not been completed. Run the eda-analysis node first.")
        raise RuntimeError(
            f"{node_id}.yaml has not been confirmed. Complete the {node_id} node with --confirm-config first."
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise RuntimeError(
            f"{node_id} has not been result-confirmed; run workflow confirm {node_id} first."
        )
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
    sample_approval_path = node_approval_path(context, "sample-diagnosis")
    if not sample_approval_path.is_file():
        return data, validated, None
    from data.contract import validate_contract
    from preprocessing.sample_config import load_sample_config
    from preprocessing.sample_diagnostics import apply_sample_treatment

    approval = json.loads(sample_approval_path.read_text(encoding="utf-8"))
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
    lines = ["# 特征筛选结果", "", f"- 保留字段：{len(retained)}", f"- 需复核字段：{len(review)}", f"- 按质量/IV/相关性/稳定性剔除：{len(excluded)}", f"- 按变量类型预处理剔除：{len(dropped_by_type)}", ""]
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


def _build_processing_summary(
    *,
    working_data: pl.DataFrame,
    processed: pl.DataFrame,
    decisions: pl.DataFrame,
    dropped_by_type: list[str],
    parameters: dict[str, Any],
    stats: pl.DataFrame,
) -> str:
    """Build the compact result table shown after feature processing."""
    rows = decisions.to_dicts()
    excluded = [row for row in rows if str(row.get("decision", "")).startswith("excluded")]
    review = [row for row in rows if row.get("decision") == "retained_review"]
    retained = [row for row in rows if row.get("decision") == "retained"]
    def compact(items: list[dict[str, Any]], limit: int = 8) -> str:
        values = [
            f"`{item.get('feature', '—')}`（{item.get('reason', '未记录原因')}）"
            for item in items[:limit]
        ]
        if len(items) > limit:
            values.append(f"…等 {len(items)} 个")
        return "；".join(values) if values else "无"

    def feature_names(items: list[dict[str, Any]], limit: int = 10) -> str:
        values = [f"`{item.get('feature', '—')}`" for item in items[:limit]]
        if len(items) > limit:
            values.append(f"…等 {len(items)} 个")
        return "、".join(values) if values else "无"

    selection = parameters.get("selection", {})
    preprocessing = parameters.get("preprocessing", {})
    numeric = preprocessing.get("numeric", {})
    categorical = preprocessing.get("categorical", {})
    text = preprocessing.get("text", {})
    # Type preprocessing decisions are kept separately because those fields
    # can be dropped after the quality/IV/correlation selection pass.
    type_drop_names = [f"`{feature}`" for feature in dropped_by_type[:8]]
    if len(dropped_by_type) > 8:
        type_drop_names.append(f"…等 {len(dropped_by_type)} 个")
    type_drop_text = "、".join(type_drop_names) if type_drop_names else "无"
    max_missing = float(selection.get("max_missing_rate", 0.80))
    quality_distribution = _distribution_summary(stats)

    def metric_range(name: str, digits: int = 4) -> str:
        value = quality_distribution.get(name, {})
        if value.get("min") is None:
            return "—"
        return f"{value['min']:.{digits}f}–{value['max']:.{digits}f}（均值 {value['mean']:.{digits}f}）"

    table_rows = [
        ("数据概况", "输入规模", f"{working_data.height:,} 行 × {len(rows):,} 个候选特征", "按已确认 data-read 角色重新读取；本节点只做特征处理"),
        ("字段去留", "直接保留", f"{len(retained):,} 个", feature_names(retained) or "无"),
        ("字段去留", "待复核（暂保留）", f"{len(review):,} 个", f"{feature_names(review)}；存在稳定性或集中度提示，建议人工复核"),
        ("字段去留", "规则剔除", f"{len(excluded):,} 个", f"{compact(excluded)}；过多时仅展示前 8 个，完整清单见 feature_selection_summary.md"),
        ("字段去留", "变量类型剔除", f"{len(dropped_by_type):,} 个", f"{type_drop_text}；按文本、近似主键、高基数或不支持类型策略处理"),
        ("处理结果", "输出数据", f"{processed.height:,} 行 × {processed.width:,} 列", "已生成 processed_data.parquet；预处理规则按当前 YAML 执行"),
        ("筛选阈值", "缺失率", f"≤ {max_missing:.0%}", "超过阈值的字段进入剔除或复核，取决于当前策略"),
        ("筛选阈值", "信息价值（IV）", f"≥ {float(selection.get('min_iv', 0.02)):.2f}", "低于阈值表示单变量区分度弱"),
        ("筛选阈值", "最大月度 PSI", f"≤ {float(selection.get('max_psi', 0.25)):.2f}", "超过阈值默认标记复核；稳定性 action={}".format(selection.get("stability_action", "review"))),
        ("筛选阈值", "相关系数绝对值", f"≤ {float(selection.get('max_correlation', 0.80)):.2f}", "超过阈值的高相关字段按 IV/缺失率等证据择优"),
        ("实际指标", "IV 范围", metric_range("iv"), "完整字段级值见 feature_statistics.csv"),
        ("实际指标", "KS 范围", metric_range("ks"), "完整字段级值见 feature_statistics.csv"),
        ("实际指标", "最大月度 PSI 范围", metric_range("max_monthly_psi"), "完整字段级值见 feature_statistics.csv"),
        ("预处理策略", "数值变量", f"invalid_to_null={bool(numeric.get('invalid_to_null', True))}；missing_strategy={numeric.get('missing_strategy', 'native')}", "无穷值转空值，缺失由 LightGBM 原生处理"),
        ("预处理策略", "类别变量", str(categorical.get("strategy", "lightgbm_native")), "按当前 YAML 的类别编码、稀有类别和高基数策略处理"),
        ("预处理策略", "文本变量", str(text.get("action", "drop")), "默认删除文本字段，避免未经确认的语义建模"),
    ]
    lines = ["## 特征预处理与筛选完成", "", "| 类别 | 字段/指标 | 当前结果 | 说明 |", "|---|---|---|---|"]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in table_rows
    )
    lines.extend(["", "完整剔除原因、字段指标和预处理计划请查看 feature_selection_summary.md、feature_statistics.csv 和 feature_processing_report.html。"])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "feature-processing")
    # Feature processing uses the current workspace YAML immediately. The
    # user receives the computed result first; only the result (not the YAML)
    # must be confirmed before the next node starts.
    confirmed = bool(getattr(args, "confirm_result", False) or getattr(args, "confirm_config", False))
    config_path = context.config.path
    config_hash = _hash(config_path)
    prior_approval_path = node_approval_path(context, "feature-processing")
    prior_approval: dict[str, Any] = {}
    if prior_approval_path.is_file():
        try:
            prior_approval = json.loads(prior_approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_approval = {}
    # Confirmation/retry commands are idempotent. Once a result has already
    # been approved, return its existing artifacts instead of recomputing the
    # node and making the host produce another feedback card.
    if (
        confirmed
        and prior_approval.get("status") == "approved"
        and prior_approval.get("config_sha256") == config_hash
    ):
        return {
            "status": "success",
            "node_id": "feature-processing",
            "summary": "特征处理结果已确认，可进入下一个已选节点。",
            "approval": str(prior_approval_path),
            "processed_data": prior_approval.get("processed_data"),
            "manifest": prior_approval.get("manifest"),
            "display_files": [
                str(item)
                for item in (
                    prior_approval.get("manifest"),
                    prior_approval.get("processed_data"),
                )
                if item and Path(str(item)).is_file()
            ],
        }
    if (
        not confirmed
        and prior_approval.get("status") == "awaiting_user_confirmation"
        and prior_approval.get("config_sha256") == config_hash
    ):
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "feature-processing",
            "summary": "特征处理结果已生成，等待用户确认后进入模型配置。",
            "approval": str(prior_approval_path),
            "display_files": [
                str(item)
                for item in (
                    prior_approval.get("manifest"),
                    prior_approval.get("processed_data"),
                )
                if item and Path(str(item)).is_file()
            ],
        }
    from data.contract import validate_contract
    from eda.analytics import build_eda_analysis
    from modeling.feature_selection import select_features
    from preprocessing.feature_preprocessing import fit_feature_preprocessor, transform_features
    from reporting.feature_report import write_feature_report

    _require_approval(context, "data-read")
    _require_approval(context, "eda-analysis")
    with context.progress.track("loader", running_summary="正在按当前配置重新读取数据", success_summary="特征处理数据读取完成"):
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
        binning_method=str(eda_params.get("binning_method", "chi")),
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
        "字段级特征统计",
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
    # Do not call a separate node-parameter advisor here. The host LLM turns
    # deterministic evidence into a concise summary and textual suggestions;
    # no recommendation YAML is generated and no extra model service is needed.
    llm_advice = {
        "status": "disabled",
        "proposal": {},
        "reason": "节点只返回总结和文字化参数建议，由宿主基于确定性证据生成",
    }
    recommendation_path = context.output_dir / "feature-processing.llm-recommended.yaml"
    auto_apply = True
    confirmation = write_json(
        context.output_dir / "feature_processing_confirmation.json",
        {
            "node_id": "feature-processing",
            "status": "approved" if confirmed else "awaiting_user_confirmation",
            "user_confirmation_required": not confirmed,
            "message": (
                "已按当前 feature-processing.yaml（或默认值）执行并生成 processed_data.parquet。"
                if confirmed
                else "特征统计、筛选证据和大模型建议已生成。"
            ),
            "config_path": str(config_path),
            "config_sha256": _hash(config_path),
            "data_read_config_sha256": _hash(context.data_read_config.path),
            "eda_config_sha256": _hash(context.node_config_dir / "eda-analysis.yaml"),
            "statistics": str(stats_path),
            "correlations": str(correlation_path),
            "distribution": _distribution_summary(stats),
            "correlation_pair_count": analysis.correlation_pairs.height,
            "display_files": [str(config_path), str(feature_report_path), str(statistics_preview)],
            "parameters": context.config.parameters,
            "llm_advice": llm_advice,
            "llm_recommendation": str(recommendation_path) if recommendation_path.is_file() else None,
        },
    )
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
            "status": "approved" if confirmed else "awaiting_user_confirmation",
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
        node_approval_path(context, "feature-processing", for_write=True),
        {"node_id": "feature-processing", "status": "approved" if confirmed else "awaiting_user_confirmation", "config": str(config_path), "config_sha256": _hash(config_path), "manifest": str(manifest), "processed_data": str(processed_path) if processed_path.is_file() else None},
    )
    retained_count = len(selection.feature_cols)
    excluded_count = len(selection.decisions.filter(pl.col("decision").str.starts_with("excluded")))
    dropped_type_count = len(plan.dropped_features)
    summary_text = _build_processing_summary(
        working_data=model_data,
        processed=processed,
        decisions=selection.decisions,
        dropped_by_type=list(plan.dropped_features),
        parameters=context.config.parameters,
        stats=stats,
    )
    node_summary = write_node_summary(
        context.output_dir,
        "feature-processing",
        summary_text,
        {
            "retained_count": retained_count,
            "excluded_count": excluded_count,
            "dropped_by_type_count": dropped_type_count,
            "selection_summary": str(selection_summary),
            "processed_data": str(processed_path) if processed_path.is_file() else None,
            "recommendations": [
                (
                    "建议复核剔除字段及其缺失率、IV、相关性和稳定性证据；如需调整，编辑 feature-processing.yaml 后重新运行本节点。"
                    if excluded_count or dropped_type_count
                    else "当前筛选规则未剔除字段，可继续执行模型配置与训练。"
                )
            ],
        },
    )
    artifacts = [stats_path, statistics_preview, selection_path, selection_summary, selection_summary_json, preprocessing_path, feature_report_path, manifest, approval, node_summary, *([processed_path] if processed_path.is_file() else [])]
    if not confirmed:
        context.progress.emit(
            "feature-processing",
            "waiting_confirmation",
            summary=summary_text + "\n\n请确认以上特征处理结果后再进入下一个节点；如需调整，请直接提出文字修改要求。",
            artifacts=artifacts,
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "feature-processing",
            "statistics": str(stats_path),
            "statistics_preview": str(statistics_preview),
            "feature_selection": str(selection_path),
            "feature_selection_summary": str(selection_summary),
            "feature_selection_summary_json": str(selection_summary_json),
            "preprocessing_plan": str(preprocessing_path),
            "html_report": str(feature_report_path),
            "processed_data": str(processed_path) if processed_path.is_file() else None,
            "manifest": str(manifest),
            "approval": str(approval),
            "node_summary": str(node_summary),
            "display_files": [str(feature_report_path), str(selection_summary), str(statistics_preview)],
            "auto_applied": auto_apply,
            "llm_advice": llm_advice,
        }
    context.progress.emit("feature-processing", "success", summary=summary_text, artifacts=artifacts)
    return {"status": "success", "node_id": "feature-processing", "statistics": str(stats_path), "statistics_preview": str(statistics_preview), "feature_selection": str(selection_path), "feature_selection_summary": str(selection_summary), "feature_selection_summary_json": str(selection_summary_json), "preprocessing_plan": str(preprocessing_path), "html_report": str(feature_report_path), "processed_data": str(processed_path) if processed_path.is_file() else None, "manifest": str(manifest), "approval": str(approval), "node_summary": str(node_summary), "display_files": [str(feature_report_path), str(selection_summary), str(statistics_preview)], "auto_applied": auto_apply, "llm_advice": llm_advice}


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

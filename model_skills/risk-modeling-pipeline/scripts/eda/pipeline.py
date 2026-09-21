"""End-to-end data EDA workflow and report artifact export."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from data.contract import ValidatedDataContract
from logger import get_logger
from preprocessing.cleaning import preprocess_data
from preprocessing.exploration import explore_dataset
from progress import ProgressReporter
from reporting.xlsx_report import write_eda_report
from reporting.eda_html_report import write_eda_html_report

from .analytics import EdaAnalysisResult, build_eda_analysis


logger = get_logger(__name__)
_SUMMARY_LABELS = {
    "row_count": "原始样本数",
    "column_count": "原始字段数",
    "feature_count": "候选特征数",
    "candidate_feature_count": "候选特征数",
    "id_unique_rate": "主键唯一率",
    "date_parse_rate": "日期解析率",
    "bad_sample_count": "原始坏样本数",
    "target_bad_rate": "原始坏样本率",
    "bad_sample_rate": "原始坏样本率",
    "cleaned_row_count": "清洗后样本数",
    "cleaned_column_count": "清洗后字段数",
    "retained_feature_count": "保留特征数",
    "removed_duplicate_count": "去重样本数",
    "invalid_date_count": "无效日期数",
}


@dataclass(frozen=True)
class EdaRunResult:
    """Paths and in-memory results from one completed data EDA run."""

    output_dir: Path
    report_path: Path
    html_report_path: Path
    cleaned_data_path: Path
    analysis: EdaAnalysisResult
    conclusion_path: Path


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _table_payload(data: pl.DataFrame) -> dict[str, Any]:
    return {
        "headers": data.columns,
        "rows": [[_json_value(value) for value in row] for row in data.rows()],
    }


def _write_csv(data: pl.DataFrame, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.csv"
    data.write_csv(path)
    return path


def _build_conclusion(
    summary: Mapping[str, Any], analysis: EdaAnalysisResult
) -> list[tuple[str, str]]:
    """Create a concise, deterministic narrative for the end of the EDA report."""
    rows: list[tuple[str, str]] = []
    rows.append(
        (
            "数据规模",
            f"原始数据 {summary.get('row_count', 0):,} 行、{summary.get('column_count', 0):,} 列；"
            f"清洗后保留 {summary.get('cleaned_row_count', 0):,} 行、"
            f"{summary.get('retained_feature_count', 0):,} 个候选特征。",
        )
    )
    rows.append(
        (
            "数据质量",
            f"去重 {summary.get('removed_duplicate_count', 0):,} 行、"
            f"无效日期 {summary.get('invalid_date_count', 0):,} 行；"
            f"主键唯一率 {float(summary.get('id_unique_rate', 0) or 0):.2%}，"
            f"日期解析率 {float(summary.get('date_parse_rate', 0) or 0):.2%}。",
        )
    )
    overview = analysis.univariate_overview
    if overview.height:
        iv_row = overview.sort("iv", descending=True).row(0, named=True)
        ks_row = overview.sort("ks", descending=True).row(0, named=True)
        rows.append(
            (
                "单变量区分度",
                f"IV 最高字段为 {iv_row.get('feature')}（{float(iv_row.get('iv') or 0):.4f}），"
                f"KS 最高字段为 {ks_row.get('feature')}（{float(ks_row.get('ks') or 0):.4f}）。",
            )
        )
    psi = analysis.monthly_psi
    max_psi = float(psi.get_column("psi").max() or 0) if psi.height and "psi" in psi.columns else 0.0
    psi_features = (
        psi.filter(pl.col("psi") >= 0.25).get_column("feature").n_unique()
        if psi.height and "psi" in psi.columns and "feature" in psi.columns
        else 0
    )
    rows.append(
        (
            "稳定性",
            f"最大月度 PSI 为 {max_psi:.4f}，达到 0.25 预警阈值的特征数为 {psi_features}；"
            + ("建议优先复核漂移字段。" if psi_features else "暂未发现明显月度漂移。"),
        )
    )
    monthly = analysis.monthly_sample
    if monthly.height and "event_month" in monthly.columns:
        latest = monthly.sort("event_month").row(-1, named=True)
        rows.append(
            (
                "月度分布",
                f"最新月份 {latest.get('event_month')} 共 {int(latest.get('sample_count') or 0):,} 条，"
                f"坏样本率 {float(latest.get('bad_rate') or 0):.2%}；小样本月份的区分度和 PSI 需谨慎解读。",
            )
        )
    rows.append(("后续建议", "以上结果用于数据理解和建模前复核；字段筛选、样本处理和模型训练将在后续节点单独确认。"))
    return rows


def run_eda(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    output_dir: str | Path,
    *,
    node_executable: str = "node",
    progress: ProgressReporter | None = None,
    report_dir: str | Path | None = None,
    generate_report: bool = True,
    analysis_config: Mapping[str, Any] | None = None,
    emit_analysis_progress: bool = True,
) -> EdaRunResult:
    """Clean confirmed data, calculate EDA tables, and export Excel + HTML reports.

    The workbook is generated by the portable XlsxWriter backend; no Node.js
    or Codex-specific report runtime is required.
    """
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir = (
        Path(report_dir).expanduser().resolve() if report_dir is not None else run_dir
    )
    report_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting EDA run: %s", run_dir)
    cleaned_data_path = run_dir / "cleaned_data.parquet"
    payload_path = run_dir / "eda_report_payload.json"
    report_path = report_output_dir / "data_eda_report.xlsx"
    html_report_path = report_output_dir / "data_eda_report.html"

    cleaning_stage = (
        progress.track(
            "eda-cleaning",
            running_summary="正在处理重复样本、无效日期和低质量字段",
            success_summary="数据清理完成",
            artifacts=[cleaned_data_path, run_dir / "column_decisions.csv"],
        )
        if progress
        else nullcontext()
    )
    with cleaning_stage:
        options = dict(analysis_config or {})
        cleaning = preprocess_data(
            data,
            contract,
            missing_rate_threshold=float(options.get("missing_rate_threshold", 0.8)),
            constant_rate_threshold=float(options.get("constant_rate_threshold", 0.8)),
            duplicate_strategy=str(options.get("duplicate_strategy", "keep_first")),
        )
        cleaning.data.write_parquet(cleaned_data_path)
        _write_csv(cleaning.column_decisions, run_dir, "column_decisions")

    eda_stage = (
        progress.track(
            "eda",
            running_summary="正在计算分布、分箱、相关性和月度稳定性",
            success_summary="探索性分析计算完成",
            artifacts=[run_dir / "eda_summary.json", payload_path],
        )
        if progress and emit_analysis_progress
        else nullcontext()
    )
    with eda_stage:
        raw_exploration = explore_dataset(data, contract)
        cleaned_contract = replace(contract, feature_cols=cleaning.feature_cols)
        analysis = build_eda_analysis(
            cleaning.data,
            cleaned_contract,
            month_col=str(options.get("month_col", "event_month")),
            bin_count=int(options.get("bin_count", 10)),
            ks_bucket=int(options.get("ks_bucket", options.get("bin_count", 10))),
            ks_method=str(options.get("ks_method", "quantile")),
            binning_method=str(options.get("binning_method", "quantile")),
            baseline_month=options.get("psi_base_month"),
            max_categories=int(options.get("max_categories", 20)),
            correlation_method=str(options.get("correlation_method", "pearson")),
            metrics_backend=str(options.get("metrics_backend", "toad")),
        )
        _write_csv(raw_exploration.column_profile, run_dir, "column_profile")
        _write_csv(analysis.univariate_overview, run_dir, "univariate_overview")
        _write_csv(analysis.binning_detail, run_dir, "binning_detail")
        _write_csv(analysis.monthly_sample, run_dir, "monthly_sample")
        _write_csv(analysis.monthly_psi, run_dir, "monthly_psi")
        _write_csv(analysis.psi_summary, run_dir, "psi_summary")
        _write_csv(analysis.monthly_discrimination, run_dir, "monthly_discrimination")
        _write_csv(analysis.ks_bucket, run_dir, "ks_bucket")
        _write_csv(analysis.correlation_pairs, run_dir, "correlation_pairs")
        _write_csv(analysis.correlation_matrix, run_dir, "correlation_matrix")

        summary = {
            **raw_exploration.summary,
            "cleaned_row_count": cleaning.data.height,
            "cleaned_column_count": cleaning.data.width,
            "retained_feature_count": len(cleaning.feature_cols),
            "removed_duplicate_count": cleaning.removed_duplicate_count,
            "invalid_date_count": cleaning.invalid_date_count,
            "binning_method": str(options.get("binning_method", "quantile")),
            "metrics_backend": str(options.get("metrics_backend", "toad")),
            "psi_baseline_month": analysis.monthly_psi.get_column("baseline_month").min()
            if analysis.monthly_psi.height and "baseline_month" in analysis.monthly_psi.columns
            else options.get("psi_base_month"),
        }
        conclusion_rows = _build_conclusion(summary, analysis)
        conclusion_path = run_dir / "eda_conclusion.md"
        conclusion_path.write_text(
            "# EDA 结果概述\n\n"
            + "\n".join(f"- **{title}**：{text}" for title, text in conclusion_rows)
            + "\n",
            encoding="utf-8",
        )
        summary["conclusion_path"] = str(conclusion_path)
        summary["conclusion"] = " ".join(f"{title}：{text}" for title, text in conclusion_rows)
        (run_dir / "eda_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload = {
            "title": "风控数据 EDA 报告",
            "tables": {
                "1.数据概况": {
                    "headers": ["指标", "值"],
                    "rows": [
                        [_SUMMARY_LABELS.get(key, key), _json_value(value)]
                        for key, value in summary.items()
                    ],
                },
                "2.字段画像": _table_payload(raw_exploration.column_profile),
                "3.特征质量": _table_payload(cleaning.column_decisions),
                "4.单变量概览": _table_payload(analysis.univariate_overview),
                "5.分箱明细": _table_payload(analysis.binning_detail),
                "6.月度样本": _table_payload(analysis.monthly_sample),
                "7.稳定性PSI": _table_payload(analysis.monthly_psi),
                "7b.变量 PSI 汇总": _table_payload(analysis.psi_summary),
                "8.相关性": _table_payload(analysis.correlation_pairs),
                "8b.相关性矩阵": _table_payload(analysis.correlation_matrix),
                "9.月度区分度": _table_payload(analysis.monthly_discrimination),
                "10.KS十分位": _table_payload(analysis.ks_bucket),
                "11.结果概述": {
                    "headers": ["结论类别", "自动概述"],
                    "rows": [[title, text] for title, text in conclusion_rows],
                },
            },
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if generate_report:
        logger.info("Generating EDA workbook: %s", report_path)
        report_stage = (
            progress.track(
                "eda-report",
                running_summary="正在生成 EDA Excel 报告",
                success_summary="EDA 报告已生成",
                artifacts=[report_path, html_report_path],
            )
            if progress
            else nullcontext()
        )
        with report_stage:
            write_eda_report(payload_path, report_path)
            write_eda_html_report(payload_path, html_report_path)
    elif progress:
        progress.emit("eda-report", "skipped", summary="未选择报告交付节点")
    logger.info("EDA run completed: %s", report_path)
    return EdaRunResult(
        run_dir, report_path, html_report_path, cleaned_data_path, analysis, conclusion_path
    )

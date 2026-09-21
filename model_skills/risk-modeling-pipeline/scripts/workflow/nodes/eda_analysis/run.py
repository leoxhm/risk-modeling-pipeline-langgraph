"""Run the EDA analysis node directly after data-read confirmation.

The node deliberately reloads the source data in a fresh process. It reads
the approved data-read configuration for roles/loader options and its own EDA
configuration for binning, stability and report settings. EDA runs with the
current configuration and writes the full HTML/Excel/CSV report. The report is
the user-facing EDA deliverable; the workflow auto-approves this evidence and
continues to sample diagnosis when that node is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ..common import NodeContext, approval_path, write_json, write_node_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the EDA analysis workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root", help="Accepted for Skill command compatibility")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--confirm-config",
        action="store_true",
        help="Backward-compatible alias for --confirm-result",
    )
    parser.add_argument(
        "--confirm-result",
        action="store_true",
        help="Confirm the completed EDA result and allow the next node",
    )
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_parameters(parameters: dict) -> dict:
    """Apply safe defaults so a direct EDA run is not blocked by optional YAML typos."""
    def integer(name: str, default: int, minimum: int = 2) -> int:
        try:
            return max(minimum, int(parameters.get(name, default)))
        except (TypeError, ValueError):
            return default

    def fraction(name: str, default: float) -> float:
        try:
            return min(1.0, max(0.0, float(parameters.get(name, default))))
        except (TypeError, ValueError):
            return default

    backend = parameters.get("metrics_backend", "toad")
    if backend not in {"toad", "legacy"}:
        backend = "toad"
    correlation = parameters.get("correlation_method", "pearson")
    if correlation not in {"pearson", "spearman"}:
        correlation = "pearson"
    ks_method = parameters.get("ks_method", "quantile")
    if ks_method not in {"quantile", "step"}:
        ks_method = "quantile"
    binning_method = parameters.get("binning_method", "chi")
    if binning_method not in {"quantile", "chi"}:
        binning_method = "chi"
    duplicate_strategy = parameters.get("duplicate_strategy", "keep_first")
    if duplicate_strategy not in {"keep_first", "keep_last", "error"}:
        duplicate_strategy = "keep_first"
    return {
        "month_col": str(parameters.get("month_col", "event_month")),
        "psi_base_month": parameters.get("psi_base_month"),
        "bin_count": integer("bin_count", 10),
        "ks_method": ks_method,
        "binning_method": binning_method,
        "ks_bucket": integer("ks_bucket", integer("bin_count", 10)),
        "max_categories": integer("max_categories", 20),
        "correlation_method": correlation,
        "metrics_backend": backend,
        "missing_rate_threshold": fraction("missing_rate_threshold", 0.8),
        "constant_rate_threshold": fraction("constant_rate_threshold", 0.8),
        "duplicate_strategy": duplicate_strategy,
        "generate_report": bool(parameters.get("generate_report", True)),
    }


def _approve_cached_result(
    context: NodeContext,
    *,
    eda_config_path: Path,
    data_read_config_path: Path,
    data_read_config_hash: str,
) -> dict | None:
    """Approve an already-generated EDA result without running EDA again.

    ``workflow confirm eda-analysis`` is a state transition, not a request to
    regenerate the report. Reusing the pending artifacts avoids duplicate EDA
    reports and keeps confirmation fast and deterministic.
    """
    marker_path = approval_path(context, "eda-analysis")
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("status") != "awaiting_user_confirmation":
        return None
    if marker.get("config_sha256") != _hash(eda_config_path):
        return None
    if marker.get("data_read_config_sha256") != data_read_config_hash:
        return None
    try:
        manifest_path = Path(str(marker["manifest_path"])).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        node_summary_path = context.output_dir / "node_summary.json"
        node_summary = json.loads(node_summary_path.read_text(encoding="utf-8"))
        report_path = Path(str(manifest["report_path"])).expanduser().resolve()
        html_report_path = Path(str(manifest["html_report_path"])).expanduser().resolve()
        conclusion_path = Path(str(manifest["conclusion_path"])).expanduser().resolve()
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    required = (manifest_path, node_summary_path, report_path, html_report_path, conclusion_path)
    if not all(path.is_file() for path in required):
        return None

    marker["status"] = "approved"
    marker["confirmed_without_rerun"] = True
    marker["config_path"] = str(eda_config_path)
    marker["data_read_config_path"] = str(data_read_config_path)
    write_json(marker_path, marker)
    manifest["status"] = "success"
    write_json(manifest_path, manifest)
    if node_summary.get("status") != "success":
        node_summary["status"] = "success"
        write_json(node_summary_path, node_summary)
    confirmation_path = context.output_dir / "eda_confirmation.json"
    if confirmation_path.is_file():
        try:
            confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
            confirmation["status"] = "approved"
            confirmation["user_confirmation_required"] = False
            confirmation["message"] = "EDA 结果已确认，可进入下一个节点。"
            write_json(confirmation_path, confirmation)
        except (OSError, json.JSONDecodeError):
            pass

    # Even when upgrading an older waiting marker, keep the conversational
    # payload report-only; the long EDA narrative remains in the artifacts.
    details = node_summary.get("details") if isinstance(node_summary.get("details"), dict) else {}
    row_count = details.get("row_count") if isinstance(details, dict) else None
    feature_count = details.get("feature_count") if isinstance(details, dict) else None
    summary = (
        f"EDA 报告已生成：分析 {int(row_count):,} 行、{int(feature_count):,} 个特征；"
        "未进行样本切分、字段删除或模型训练，现进入样本诊断。"
        if isinstance(row_count, (int, float)) and isinstance(feature_count, (int, float))
        else "EDA 报告已生成，未进行样本切分、字段删除或模型训练，现进入样本诊断。"
    )
    artifacts = [manifest_path, node_summary_path, conclusion_path, report_path, html_report_path, marker_path]
    context.progress.emit("eda-analysis", "success", summary=summary, artifacts=artifacts)
    return {
        "status": "success",
        "node_id": "eda-analysis",
        "report": str(report_path),
        "html_report": str(html_report_path),
        "conclusion": str(conclusion_path),
        "manifest": str(manifest_path),
        "approval": str(marker_path),
        "node_summary": str(node_summary_path),
        "display_files": [str(html_report_path), str(report_path), str(conclusion_path)],
    }


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(
        args, "eda-analysis", skip_config_validation={"eda-analysis"}
    )
    # EDA is report-only in the conversational workflow. Keep accepting the
    # historical flags for CLI compatibility, but do not create a second
    # confirmation gate between data-read and sample diagnosis.
    confirmed = True
    from data.contract import validate_contract

    data_read_approval = approval_path(context, "data-read")
    if not data_read_approval.is_file():
        raise RuntimeError(
            "data-read.yaml has not been explicitly confirmed. Run the data-read "
            "node with --confirm-config before starting EDA."
        )
    approval = json.loads(data_read_approval.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise RuntimeError(
            "data-read result has not been confirmed; run workflow confirm data-read first."
        )
    current_data_read_hash = _hash(context.data_read_config.path)
    if approval.get("config_sha256") != current_data_read_hash:
        raise RuntimeError(
            "data-read.yaml changed after confirmation. Re-run data-read with "
            "--confirm-config before starting EDA."
        )

    eda_config_path = context.config.path
    data_read_config_path = context.data_read_config.path
    parameters = _normalized_parameters(context.config.parameters)
    if confirmed:
        cached = _approve_cached_result(
            context,
            eda_config_path=eda_config_path,
            data_read_config_path=data_read_config_path,
            data_read_config_hash=current_data_read_hash,
        )
        if cached is not None:
            return cached

    # This is intentionally a fresh read.  No dataframe is inherited from the
    # data-read node, so the EDA evidence always reflects the current YAML.
    with context.progress.track(
        "loader",
        running_summary="正在按已确认的数据配置重新读取数据",
        success_summary="数据重新读取完成，开始 EDA 分析",
    ):
        loaded = context.reload_data()
        validated = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )

    from eda.pipeline import run_eda

    # The standalone node owns the user-facing EDA event. Do not pass the
    # reporter into run_eda: that legacy helper emits both ``eda`` and
    # ``eda-report`` events, which canonicalize to the same UI node and make
    # OpenCode display the EDA result twice.
    context.progress.emit(
        "eda",
        "running",
        summary="正在计算字段画像、分箱、IV/KS、PSI、相关性和月度分析",
    )
    try:
        result = run_eda(
            loaded.data,
            validated,
            context.output_dir,
            progress=None,
            report_dir=context.output_dir / "reports",
            generate_report=bool(parameters.get("generate_report", True)),
            analysis_config=parameters,
            emit_analysis_progress=False,
        )
    except BaseException as exc:
        context.progress.emit("eda", "failed", summary=str(exc) or exc.__class__.__name__)
        raise

    confirmation = write_json(
        context.output_dir / "eda_confirmation.json",
        {
            "node_id": "eda-analysis",
            "status": "approved",
            "user_confirmation_required": False,
            "message": "EDA 报告已生成；如需调整分析参数，请提出文字修改要求。",
            "project_root": str(context.project_root),
            "data_path": str(context.data_path),
            "data_read_config_path": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "eda_config_path": str(eda_config_path),
            "eda_config_sha256": _hash(eda_config_path),
            "row_count": loaded.row_count,
            "column_count": loaded.column_count,
            "feature_count": len(validated.feature_cols),
            "parameters": parameters,
            "display_files": [str(result.html_report_path), str(result.report_path), str(result.conclusion_path)],
        },
    )

    manifest = write_json(
        context.output_dir / "eda_manifest.json",
        {
            "node_id": "eda-analysis",
            "status": "success",
            "project_root": str(context.project_root),
            "data_path": str(context.data_path),
            "data_read_config": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "eda_config": str(eda_config_path),
            "eda_config_sha256": _hash(eda_config_path),
            "report_path": str(result.report_path),
            "html_report_path": str(result.html_report_path),
            "conclusion_path": str(result.conclusion_path),
            "cleaned_data_path": str(result.cleaned_data_path),
            "output_dir": str(context.output_dir),
        },
    )
    # Write a marker on every pass. Downstream nodes accept it only when its
    # status is ``approved``, so an old approval cannot bypass this gate.
    approval = write_json(
        approval_path(context, "eda-analysis", for_write=True),
        {
            "node_id": "eda-analysis",
            "status": "approved",
            "config_path": str(eda_config_path),
            "config_sha256": _hash(eda_config_path),
            "data_read_config_path": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "manifest_path": str(manifest),
        },
    )
    overview = result.analysis.univariate_overview
    top_iv = overview.sort("iv", descending=True).row(0, named=True) if overview.height and "iv" in overview.columns else {}
    top_ks = overview.sort("ks", descending=True).row(0, named=True) if overview.height and "ks" in overview.columns else {}
    max_psi = (
        float(result.analysis.monthly_psi.get_column("psi").max() or 0.0)
        if result.analysis.monthly_psi.height and "psi" in result.analysis.monthly_psi.columns
        else 0.0
    )
    overview = result.analysis.univariate_overview
    psi_summary = result.analysis.psi_summary
    correlation_pairs = result.analysis.correlation_pairs
    monthly_sample = result.analysis.monthly_sample
    monthly_discrimination = result.analysis.monthly_discrimination

    def _count_at_least(frame: pl.DataFrame, column: str, threshold: float) -> int:
        if frame.height == 0 or column not in frame.columns:
            return 0
        return frame.filter(pl.col(column).cast(pl.Float64, strict=False) >= threshold).height

    weak_iv_count = _count_at_least(overview, "iv", 0.02)
    weak_iv_count = overview.height - weak_iv_count if overview.height else 0
    high_corr_count = _count_at_least(correlation_pairs, "abs_correlation", 0.70)
    high_psi_feature_count = _count_at_least(psi_summary, "max_psi", 0.25)
    latest_month = monthly_sample.sort("event_month").row(-1, named=True) if monthly_sample.height else {}
    target = loaded.data.get_column(validated.contract.target_col)
    labeled_mask = target.is_in([validated.contract.good_label, validated.contract.bad_label])
    labeled_count = int(labeled_mask.sum())
    overall_bad_rate = (
        float(((target == validated.contract.bad_label) & labeled_mask).sum()) / labeled_count
        if labeled_count else 0.0
    )
    latest_month_text = (
        f"最新月 {latest_month.get('event_month')} 样本 {int(latest_month.get('sample_count') or 0):,} 条，"
        f"坏账率 {float(latest_month.get('bad_rate') or 0):.2%}。"
        if latest_month else "未形成可用的月度样本统计。"
    )
    monthly_ks_text = ""
    if monthly_discrimination.height and "ks" in monthly_discrimination.columns:
        ks_values = monthly_discrimination.get_column("ks").drop_nulls().cast(pl.Float64)
        if len(ks_values):
            monthly_ks_text = f"月度 KS 范围 {float(ks_values.min()):.4f}–{float(ks_values.max()):.4f}。"
    psi_by_feature = {
        str(row.get("feature")): float(row.get("max_psi") or 0.0)
        for row in psi_summary.to_dicts()
    }
    iv_by_feature = {
        str(row.get("feature")): float(row.get("iv") or 0.0)
        for row in overview.to_dicts()
    }

    def _cell(value: object) -> str:
        if value is None:
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def _metric(row: dict, key: str, digits: int = 4) -> str:
        value = row.get(key)
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return _cell(value)

    def _pct(row: dict, key: str) -> str:
        value = row.get(key)
        if value is None:
            return "—"
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return _cell(value)

    def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
        lines = [f"### {title}", "", "| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|")
        if rows:
            lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
        else:
            lines.append("| 暂无可用统计 | " + " | ".join("—" for _ in headers[1:]) + " |")
        return "\n".join(lines)

    # Keep the chat response small and deterministic. Full detail remains in
    # the report/CSV artifacts; these five tables are the EDA decision surface.
    overview_tables = _table(
        "样本与分析概况",
        ["指标", "值"],
        [
            ["样本量", f"{loaded.row_count:,} 行"],
            ["候选特征", f"{len(validated.feature_cols)} 个"],
            ["整体坏样本率", f"{overall_bad_rate:.2%}"],
            ["观察月份", f"{monthly_sample.height} 个"],
            ["最大月度样本 PSI", _metric(monthly_sample.sort("psi", descending=True).row(0, named=True), "psi") if monthly_sample.height and "psi" in monthly_sample.columns else "—"],
        ],
    )
    top_iv_rows = [
        [
            str(row.get("feature")),
            _metric(row, "iv"),
            _metric(row, "ks"),
            _pct(row, "missing_rate"),
        ]
        for row in overview.sort("iv", descending=True).head(5).to_dicts()
    ] if overview.height else []
    iv_table = _table("重点特征（按 IV 排名）", ["变量", "IV", "KS", "缺失率"], top_iv_rows)

    top_psi_rows = [
        [
            str(row.get("feature")),
            _metric(row, "average_psi"),
            _metric(row, "max_psi"),
            _metric({"value": iv_by_feature.get(str(row.get("feature")))}, "value"),
        ]
        for row in psi_summary.sort("max_psi", descending=True).head(5).to_dicts()
    ] if psi_summary.height else []
    psi_table = _table("稳定性重点特征（按最大 PSI 排名）", ["变量", "平均 PSI", "最大 PSI", "IV"], top_psi_rows)

    month_rows = monthly_sample.sort("event_month").to_dicts() if monthly_sample.height else []
    month_note = ""
    if len(month_rows) > 8:
        month_rows = month_rows[-8:]
        month_note = f"（仅展示最近 8 个月，共 {monthly_sample.height} 个月）"
    monthly_table = _table(
        f"按月样本与坏样本占比{month_note}",
        ["月份", "样本数", "坏样本数", "坏账率", "样本 PSI"],
        [[str(row.get("event_month")), _cell(row.get("sample_count")), _cell(row.get("bad_count")), _pct(row, "bad_rate"), _metric(row, "psi")] for row in month_rows],
    )

    correlation_rows = [
        [str(row.get("feature_1")), str(row.get("feature_2")), _metric(row, "correlation"), _metric(row, "abs_correlation")]
        for row in correlation_pairs.sort("abs_correlation", descending=True).head(5).to_dicts()
    ] if correlation_pairs.height else []
    correlation_table = _table("高相关字段（按绝对相关系数排名）", ["变量 1", "变量 2", "相关系数", "|相关系数|"], correlation_rows)
    summary_tables = "\n\n".join([overview_tables, iv_table, psi_table, monthly_table, correlation_table])
    summary_text = (
        "## EDA 分析完成\n\n"
        f"已分析 {loaded.row_count:,} 行、{len(validated.feature_cols)} 个特征，"
        f"整体坏样本率 {overall_bad_rate:.2%}，报告已生成。\n\n"
        "### 关键建模发现\n\n"
        f"- 区分度：最高 IV 为 **{top_iv.get('feature', '无')}**（{float(top_iv.get('iv') or 0):.4f}），"
        f"最高 KS 为 **{top_ks.get('feature', '无')}**（{float(top_ks.get('ks') or 0):.4f}）；"
        f"IV<0.02 的弱区分度特征 {weak_iv_count} 个。\n"
        f"- 稳定性：最大月度 PSI 为 **{max_psi:.4f}**，其中 {high_psi_feature_count} 个特征达到 PSI 预警水平；"
        "这会增加 OOT 退化风险，建议先核对月份口径和变量分布，不建议直接批量删除。\n"
        f"- 冗余性：|相关系数|≥0.70 的字段对有 {high_corr_count} 组；后续应按 IV、缺失率和 PSI 择优，避免重复信息同时入模。\n"
        f"- 时间稳定性：{monthly_ks_text or '月度区分度统计有限。'} {latest_month_text}\n\n"
        f"{summary_tables}\n"
        "### 对后续建模的建议\n\n"
        "1. 特征筛选阶段优先复核高 PSI、高相关和低区分度字段，先保留证据再决定去留。\n"
        "2. 最新月份样本量过少时，不建议用该月单独判断模型好坏；时间切分和 OOT 设计需单独确认。\n"
        "3. EDA 不自动修改数据，样本处理、类别不平衡和字段去留在后续节点确认。"
    )
    node_summary = write_node_summary(
        context.output_dir,
        "eda-analysis",
        summary_text,
        {
            "row_count": loaded.row_count,
            "feature_count": len(validated.feature_cols),
            "top_iv": top_iv,
            "top_ks": top_ks,
            "max_monthly_psi": max_psi,
            "weak_iv_feature_count": weak_iv_count,
            "high_psi_feature_count": high_psi_feature_count,
            "high_correlation_pair_count": high_corr_count,
            "latest_month": latest_month,
            "top_features": overview.sort("iv", descending=True).head(5).to_dicts() if overview.height else [],
            "report": str(result.report_path),
            "html_report": str(result.html_report_path),
            "conclusion": str(result.conclusion_path),
            "recommendations": [
                "优先复核弱区分度、高 PSI 和高相关字段，再确认特征筛选阈值。",
                "结合最新月份样本量和月度 KS 波动，确认时间窗口及后续稳定性处理。",
            ],
        },
        status="success",
    )
    artifacts = [
        manifest,
        node_summary,
        result.conclusion_path,
        result.report_path,
        result.html_report_path,
    ]
    if approval is not None:
        artifacts.append(approval)
    # Do not stream the long EDA Markdown summary into chat.  It remains in
    # node_summary.json and the report artifacts for audit and inspection.
    report_only_summary = (
        f"EDA 报告已生成：分析 {loaded.row_count:,} 行、{len(validated.feature_cols)} 个特征；"
        "未进行样本切分、字段删除或模型训练，现进入样本诊断。"
    )
    context.progress.emit(
        "eda-analysis",
        "success",
        summary=report_only_summary,
        artifacts=artifacts,
    )
    return {
        "status": "success",
        "node_id": "eda-analysis",
        "report": str(result.report_path),
        "html_report": str(result.html_report_path),
        "conclusion": str(result.conclusion_path),
        "manifest": str(manifest),
        "approval": str(approval) if approval is not None else None,
        "node_summary": str(node_summary),
        "display_files": [str(result.html_report_path), str(result.report_path), str(result.conclusion_path)],
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

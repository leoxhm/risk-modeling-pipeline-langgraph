"""Standalone sample diagnosis node using the default diagnostic policy.

The node uses the Skill's default diagnostic parameters, summarizes the raw
sample evidence, and never modifies data. It returns a concise summary and
textual recommendations; sample-treatment choices remain a later modeling
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ..common import NodeContext, approval_path, resolve_path, write_json, write_node_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the sample-diagnosis workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root", help="Accepted for Skill command compatibility")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--sample-config")
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
        help="Confirm the completed diagnosis result and allow the next node",
    )
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_eda_csv(output_dir: Path, name: str) -> pl.DataFrame:
    """Read one EDA evidence table without making the diagnosis depend on it."""
    path = output_dir / f"{name}.csv"
    if not path.is_file():
        return pl.DataFrame()
    try:
        return pl.read_csv(path, ignore_errors=True)
    except (OSError, pl.exceptions.PolarsError):
        return pl.DataFrame()


def _eda_output_dir(approval: dict[str, object], project_root: Path) -> Path:
    """Resolve the output directory recorded by the approved EDA manifest."""
    manifest_value = approval.get("manifest_path")
    if isinstance(manifest_value, str):
        manifest_path = Path(manifest_value).expanduser().resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output_value = manifest.get("output_dir")
            if isinstance(output_value, str):
                return Path(output_value).expanduser().resolve()
        except (OSError, json.JSONDecodeError):
            pass
    return (project_root / "outputs" / "eda-analysis").resolve()


def _ratio_status(value: float | None, *, pass_ratio: float) -> str:
    if value is None:
        return "提示"
    if value >= pass_ratio:
        return "通过"
    if value >= pass_ratio / 2:
        return "提示"
    return "警告"


def _build_modeling_risk_table(
    *,
    diagnostics: dict[str, object],
    overview: pl.DataFrame,
    psi_summary: pl.DataFrame,
    monthly: pl.DataFrame,
    thresholds: object,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Build one compact, explainable risk table for the modeling hand-off."""
    metrics = diagnostics.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    total_features = overview.height
    iv_values = (
        overview.get_column("iv").cast(pl.Float64, strict=False).drop_nulls()
        if total_features and "iv" in overview.columns
        else pl.Series([], dtype=pl.Float64)
    )
    psi_values = (
        psi_summary.get_column("max_psi").cast(pl.Float64, strict=False).drop_nulls()
        if psi_summary.height and "max_psi" in psi_summary.columns
        else pl.Series([], dtype=pl.Float64)
    )
    missing_values = (
        overview.get_column("missing_rate").cast(pl.Float64, strict=False).drop_nulls()
        if total_features and "missing_rate" in overview.columns
        else pl.Series([], dtype=pl.Float64)
    )
    iv_pass = int((iv_values >= thresholds.iv_weak_threshold).sum()) if len(iv_values) else 0
    psi_effective = int((psi_values < thresholds.psi_warning_threshold).sum()) if len(psi_values) else 0
    psi_stable = int((psi_values < thresholds.psi_stable_threshold).sum()) if len(psi_values) else 0
    missing_pass = int((missing_values <= 0.80).sum()) if len(missing_values) else 0
    iv_ratio = iv_pass / len(iv_values) if len(iv_values) else None
    psi_ratio = psi_effective / len(psi_values) if len(psi_values) else None
    psi_stable_ratio = psi_stable / len(psi_values) if len(psi_values) else None
    missing_ratio = missing_pass / len(missing_values) if len(missing_values) else None
    iv_rows = overview.to_dicts() if total_features and "iv" in overview.columns else []
    iv_min_row = min(iv_rows, key=lambda row: float(row.get("iv") or 0), default={})
    iv_max_row = max(iv_rows, key=lambda row: float(row.get("iv") or 0), default={})
    psi_rows = psi_summary.to_dicts() if psi_summary.height and "max_psi" in psi_summary.columns else []
    psi_max_row = max(psi_rows, key=lambda row: float(row.get("max_psi") or 0), default={})
    bad_rate = metrics.get("bad_rate")
    bad_rate = float(bad_rate) if isinstance(bad_rate, (int, float)) else None
    bad_rate_ok = bad_rate is not None and thresholds.bad_rate_min <= bad_rate <= thresholds.bad_rate_max
    findings = diagnostics.get("findings", [])
    finding_codes = {
        str(item.get("code")): item
        for item in findings
        if isinstance(item, dict)
    }
    duplicate_count = int(metrics.get("duplicate_key_count") or 0)
    missing_target_count = int(metrics.get("missing_target_count") or 0)
    monthly_issue = any(
        code in finding_codes
        for code in ("MISSING_MONTHS", "MONTHLY_BAD_RATE_SHIFT", "LATEST_MONTH_INCOMPLETE")
    )
    imbalance_issue = "CLASS_IMBALANCE_CRITICAL" in finding_codes or "CLASS_IMBALANCE_WARNING" in finding_codes
    rows: list[dict[str, str]] = []

    def add(item: str, current: str, status: str, explanation: str, advice: str) -> None:
        rows.append(
            {
                "检查项": item,
                "当前结果": current,
                "判定": status,
                "阈值解释": explanation,
                "建模建议": advice,
            }
        )

    add(
        "标签与主键质量",
        f"重复主键 {duplicate_count:,} 组；缺失标签 {missing_target_count:,} 条",
        "通过" if duplicate_count == 0 and missing_target_count == 0 else "警告",
        "重复主键和缺失标签均应为 0；本节点只提示，不自动删除或改写。",
        "在模型配置阶段明确去重和缺失标签处理策略。",
    )
    add(
        "整体坏样本率",
        "—" if bad_rate is None else f"{bad_rate:.2%}",
        "通过" if bad_rate_ok else "警告" if bad_rate is not None else "提示",
        f"通用参考区间 {thresholds.bad_rate_min:.0%}–{thresholds.bad_rate_max:.0%}；业务口径可覆盖。",
        "过低会导致坏样本不足，过高需核对标签口径和抽样策略。",
    )
    add(
        "IV 达标特征占比",
        (
            f"{iv_pass:,}/{len(iv_values):,}（{iv_ratio:.1%}）；"
            f"最高 {iv_max_row.get('feature', '—')}={float(iv_max_row.get('iv') or 0):.4f}，"
            f"最低 {iv_min_row.get('feature', '—')}={float(iv_min_row.get('iv') or 0):.4f}"
        ) if iv_ratio is not None else "EDA 明细不可用",
        _ratio_status(iv_ratio, pass_ratio=thresholds.feature_quality_pass_ratio),
        f"IV ≥ {thresholds.iv_weak_threshold:.2f} 视为有基本区分度；达标占比 ≥ {thresholds.feature_quality_pass_ratio:.0%} 为通过。",
        "优先复核 IV 较低字段，不建议仅凭单一指标批量删除。",
    )
    add(
        "PSI 有效特征占比",
        (
            f"{psi_effective:,}/{len(psi_values):,}（{psi_ratio:.1%}）；"
            f"最大 {psi_max_row.get('feature', '—')}={float(psi_max_row.get('max_psi') or 0):.4f}"
        ) if psi_ratio is not None else "EDA 明细不可用",
        _ratio_status(psi_ratio, pass_ratio=thresholds.feature_quality_pass_ratio),
        f"最大 PSI < {thresholds.psi_warning_threshold:.2f} 视为未显著漂移；达标占比 ≥ {thresholds.feature_quality_pass_ratio:.0%} 为通过。",
        "对 PSI ≥ 阈值的字段核对月份口径、数据源变更和 OOT 风险。",
    )
    add(
        "PSI 稳定特征占比",
        f"{psi_stable:,}/{len(psi_values):,}（{psi_stable_ratio:.1%}）" if psi_stable_ratio is not None else "EDA 明细不可用",
        _ratio_status(psi_stable_ratio, pass_ratio=thresholds.feature_quality_pass_ratio),
        f"最大 PSI < {thresholds.psi_stable_threshold:.2f} 为稳定；{thresholds.psi_stable_threshold:.2f}–{thresholds.psi_warning_threshold:.2f} 为观察区间。",
        "稳定性不足时，后续筛选需结合业务解释和时间切分复核。",
    )
    add(
        "缺失率达标特征占比",
        f"{missing_pass:,}/{len(missing_values):,}（{missing_ratio:.1%}）" if missing_ratio is not None else "EDA 明细不可用",
        _ratio_status(missing_ratio, pass_ratio=thresholds.feature_quality_pass_ratio),
        "单字段缺失率 ≤ 80% 作为通用质量线；达标占比 ≥ 80% 为通过。",
        "高缺失字段需结合业务含义决定保留、填补或剔除。",
    )
    add(
        "类别不平衡",
        f"少数类占比 {float(metrics.get('minority_rate') or 0):.2%}",
        "警告" if imbalance_issue else "通过",
        f"少数类占比 < {thresholds.imbalance_warning_minority_rate:.0%} 提示不平衡，< {thresholds.imbalance_critical_minority_rate:.0%} 为严重区间。",
        "仅在 Train 上评估 class_weight/采样方案，Test/OOT 保持自然分布。",
    )
    add(
        "时间完整性与月度波动",
        f"缺失月份 {int(metrics.get('missing_month_count') or 0):,} 个；" + ("存在异常" if monthly_issue else "未发现异常"),
        "警告" if monthly_issue else "通过",
        f"相邻月坏账率变化 ≥ {thresholds.monthly_bad_rate_change_warning:.1%}、月份断层或最新月低于近期中位数 {thresholds.latest_month_min_volume_ratio:.0%} 会提示。",
        "先核对数据截止时间和标签成熟度，再确定时间窗口与 OOT 月份。",
    )
    risk = {
        "iv_pass_ratio": iv_ratio,
        "psi_effective_ratio": psi_ratio,
        "psi_stable_ratio": psi_stable_ratio,
        "missing_rate_pass_ratio": missing_ratio,
        "bad_rate": bad_rate,
        "bad_rate_range": [thresholds.bad_rate_min, thresholds.bad_rate_max],
        "feature_count": total_features,
        "risk_table": rows,
    }
    return rows, risk


def _resolve_sample_config(args: argparse.Namespace, context: NodeContext) -> Path:
    """Resolve the single node config, with compatibility for old workspaces."""
    if args.sample_config:
        return resolve_path(args.sample_config, context.project_root)
    if "diagnostics" in context.config.parameters and "treatment" in context.config.parameters:
        return context.config.path
    legacy_path = context.project_root / "configs/sample_config.yaml"
    if legacy_path.is_file():
        return legacy_path
    return context.config.path


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "sample-diagnosis")
    # Sample diagnosis always uses Skill defaults. It has no YAML form, but
    # its result must be explicitly confirmed before the next node starts.
    confirmed = bool(getattr(args, "confirm_result", False) or getattr(args, "confirm_config", False))
    from data.contract import validate_contract
    from preprocessing.sample_config import load_sample_config
    from preprocessing.sample_diagnostics import build_sample_diagnostics

    data_read_approval = approval_path(context, "data-read")
    if not data_read_approval.is_file():
        raise RuntimeError(
            "data-read.yaml has not been confirmed. Complete the data-read node before starting sample diagnosis."
        )
    approval = json.loads(data_read_approval.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise RuntimeError(
            "data-read result has not been confirmed; run workflow confirm data-read first."
        )
    current_read_hash = _hash(context.data_read_config.path)
    if approval.get("config_sha256") != current_read_hash:
        raise RuntimeError(
            "data-read.yaml changed after confirmation. Re-run data-read with --confirm-config before starting sample diagnosis."
        )

    eda_approval_path = approval_path(context, "eda-analysis")
    if not eda_approval_path.is_file():
        raise RuntimeError(
            "EDA has not been completed. Run the eda-analysis node before starting sample diagnosis."
        )
    eda_approval = json.loads(eda_approval_path.read_text(encoding="utf-8"))
    if eda_approval.get("status") != "approved":
        raise RuntimeError(
            "EDA result has not been confirmed; run workflow confirm eda-analysis first."
        )
    eda_config_path = context.node_config_dir / "eda-analysis.yaml"
    if not eda_config_path.is_file():
        raise RuntimeError("eda-analysis.yaml is missing. Re-run the eda-analysis node before starting sample diagnosis.")
    eda_config_hash = _hash(eda_config_path)
    if eda_approval.get("config_sha256") != eda_config_hash:
        raise RuntimeError(
            "eda-analysis.yaml changed after EDA completed. Re-run eda-analysis before starting sample diagnosis."
        )

    # This node intentionally ignores user-edited treatment parameters. It
    # always loads the Skill template defaults and only produces diagnostics.
    from ..common import resolve_engine_assets

    defaults_path = resolve_engine_assets(args) / "sample_diagnosis.template.yaml"
    sample_config = load_sample_config(defaults_path)
    sample_config_path = _resolve_sample_config(args, context)
    with context.progress.track(
        "loader",
        running_summary="正在按已确认的数据配置重新读取数据",
        success_summary="样本诊断数据读取完成",
    ):
        loaded = context.reload_data()
        validated = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )

    diagnostics = build_sample_diagnostics(loaded.data, validated.contract, sample_config)
    # EDA is an evidence-producing node.  Reuse its persisted aggregates here
    # so the hand-off to modeling is a single, explainable risk table rather
    # than another copy of the full EDA narrative.
    eda_dir = _eda_output_dir(eda_approval, context.project_root)
    eda_overview = _read_eda_csv(eda_dir, "univariate_overview")
    eda_psi_summary = _read_eda_csv(eda_dir, "psi_summary")
    eda_monthly = _read_eda_csv(eda_dir, "monthly_sample")
    risk_rows, modeling_risk = _build_modeling_risk_table(
        diagnostics=diagnostics,
        overview=eda_overview,
        psi_summary=eda_psi_summary,
        monthly=eda_monthly,
        thresholds=sample_config.diagnostics,
    )
    diagnostics["modeling_risk"] = modeling_risk
    risk_table_path = write_json(
        context.output_dir / "modeling_risk_summary.json",
        {
            "node_id": "sample-diagnosis",
            "threshold_source": str(defaults_path),
            "rows": risk_rows,
            **modeling_risk,
        },
    )
    diagnostics_path = write_json(
        context.output_dir / "sample_diagnostics.json",
        {"node_id": "sample-diagnosis", **diagnostics},
    )
    finding_count = len(diagnostics.get("findings", []))
    diagnostic_metrics = diagnostics.get("metrics", {})
    manifest = write_json(
        context.output_dir / "sample_diagnosis_manifest.json",
        {
            "node_id": "sample-diagnosis",
            "status": "success" if confirmed else "awaiting_user_confirmation",
            "defaults_used": True,
            "defaults_source": str(defaults_path),
            "data_path": str(context.data_path),
            "data_read_config": str(context.data_read_config.path),
            "data_read_config_sha256": current_read_hash,
            "eda_config": str(eda_config_path),
            "eda_config_sha256": eda_config_hash,
            "node_config": str(context.config.path),
            "node_config_sha256": _hash(context.config.path),
            "workspace_sample_config": str(sample_config_path),
            "workspace_sample_config_sha256": _hash(sample_config_path),
            "diagnostics": str(diagnostics_path),
            "modeling_risk_summary": str(risk_table_path),
            "finding_count": finding_count,
            "treatment_policy_applied": False,
        },
    )
    # The risk table is the only conversational payload.  Detailed findings
    # remain in sample_diagnostics.json and the EDA report for inspection.
    def _risk_table_markdown(rows: list[dict[str, str]]) -> str:
        lines = [
            "## 样本诊断与建模风险提示",
            "",
            f"已检查 {diagnostic_metrics.get('row_count', loaded.data.height):,} 条样本；本节点使用 Skill 默认阈值，不修改原始数据。",
            "",
            "| 检查项 | 当前结果 | 判定 | 阈值解释 | 建模建议 |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    str(row.get(key, "—")).replace("|", "\\|").replace("\n", " ")
                    for key in ("检查项", "当前结果", "判定", "阈值解释", "建模建议")
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "判定仅为建模前提示，不是自动阻断；详细字段明细请查看 EDA 报告。",
            ]
        )
        return "\n".join(lines)

    summary = _risk_table_markdown(risk_rows)
    node_summary = write_node_summary(
        context.output_dir,
        "sample-diagnosis",
        summary,
        {
            "finding_count": finding_count,
            "duplicate_key_count": diagnostic_metrics.get("duplicate_key_count", 0),
            "missing_target_count": diagnostic_metrics.get("missing_target_count", 0),
            "minority_rate": diagnostic_metrics.get("minority_rate"),
            "defaults_used": True,
            "treatment_policy_applied": False,
            "modeling_risk": modeling_risk,
            "recommendations": [
                "优先处理风险表中的警告项，再在模型配置阶段确认样本处理和类别不平衡策略。",
                "IV/PSI/坏样本率阈值是通用参考值，必要时结合业务口径调整。",
            ],
        },
        status="success" if confirmed else "awaiting_user_confirmation",
    )
    confirmation_path = write_json(
        context.output_dir / "sample_diagnosis_confirmation.json",
        {
            "node_id": "sample-diagnosis",
            "status": "approved" if confirmed else "awaiting_user_confirmation",
            "user_confirmation_required": not confirmed,
            "message": (
                "样本诊断结果已确认，可进入下一个节点。"
                if confirmed
                else "样本诊断已完成，请查看摘要并确认诊断结果后再进入下一个节点；本节点不修改样本。"
            ),
            "diagnostics": str(diagnostics_path),
            "modeling_risk_summary": str(risk_table_path),
            "node_summary": str(node_summary),
        },
    )
    if not confirmed:
        context.progress.emit(
            "sample-diagnosis",
            "waiting_confirmation",
            summary=summary + "\n\n请确认以上样本诊断结果后再进入下一个节点。",
            artifacts=[diagnostics_path, risk_table_path, node_summary, manifest, confirmation_path],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "sample-diagnosis",
            "diagnostics": str(diagnostics_path),
            "modeling_risk": str(risk_table_path),
            "node_summary": str(node_summary),
            "manifest": str(manifest),
            "confirmation": str(confirmation_path),
            "display_files": [str(node_summary), str(risk_table_path), str(diagnostics_path)],
        }
    context.progress.emit(
        "sample-diagnosis",
        "success",
        summary=summary,
        artifacts=[diagnostics_path, risk_table_path, node_summary, manifest, confirmation_path],
    )
    return {
        "status": "success",
        "node_id": "sample-diagnosis",
        "diagnostics": str(diagnostics_path),
        "modeling_risk": str(risk_table_path),
        "node_summary": str(node_summary),
        "manifest": str(manifest),
        "confirmation": str(confirmation_path),
        "display_files": [str(node_summary), str(risk_table_path), str(diagnostics_path)],
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

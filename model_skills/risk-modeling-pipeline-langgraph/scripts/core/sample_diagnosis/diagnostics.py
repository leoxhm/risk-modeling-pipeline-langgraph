"""Derive concise, evidence-based sample diagnostics from EDA tables."""

from __future__ import annotations

from statistics import median
from typing import Any

import polars as pl


def _rows(frame: pl.DataFrame, *, limit: int = 10) -> list[dict[str, Any]]:
    """将 DataFrame 转为适合 JSON/Markdown 展示的有限行列表。"""
    return frame.head(limit).to_dicts() if frame.height else []


def _fmt(value: Any, digits: int = 5, *, percent: bool = False) -> str:
    """格式化诊断表中的数值，避免 NaN/None 破坏输出。

    指标（IV、KS、PSI、相关系数等）按原始比例保留五位小数；只有
    缺失率、坏账率等明确的比例字段才转换为百分比，避免把 IV 误显示成百分数。
    """
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}%}" if percent else f"{number:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """将二维值列表渲染为 Markdown 表格。"""
    if not rows:
        return "暂无"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _severity_label(value: str) -> str:
    """把内部严重程度转换成面向建模人员的中文标签。"""
    return {"blocker": "阻断", "warning": "警告", "info": "提示"}.get(value, value)


def _feature_values(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    limit: int = 3,
    percent: bool = False,
) -> str:
    """把 Top 特征压缩成一行证据，避免对话摘要膨胀。"""
    values: list[str] = []
    for row in rows[:limit]:
        feature = row.get("feature")
        value = row.get(value_key)
        if feature is None:
            continue
        values.append(f"{feature}={_fmt(value, percent=percent)}")
    return "、".join(values) if values else "暂无"


def build_sample_diagnosis(
    *,
    overview: pl.DataFrame,
    monthly: pl.DataFrame,
    psi_summary: pl.DataFrame,
    correlation_pairs: pl.DataFrame,
    monthly_discrimination: pl.DataFrame,
    duplicate_groups: int = 0,
    missing_labels: int = 0,
    total_rows: int = 0,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """从 EDA 结果生成样本诊断发现、Top10 清单和建模建议。"""
    limits = {
        "missing_warning": 0.50,
        "missing_delete": 0.80,
        "iv_delete": 0.02,
        "iv_leakage": 0.50,
        "psi_warning": 0.25,
        "corr_warning": 0.70,
        "small_month_ratio": 0.30,
        "monthly_ks_std": 0.10,
    }
    limits.update(thresholds or {})
    findings: list[dict[str, Any]] = []
    recommendations: list[str] = []

    if missing_labels:
        findings.append({"name": "缺失标签", "value": missing_labels, "threshold": "必须为 0", "severity": "blocker", "recommendation": "剔除或补齐缺失标签后再建模"})
        recommendations.append("先处理缺失标签，避免训练样本口径不一致。")
    if duplicate_groups:
        findings.append({"name": "重复主键组", "value": duplicate_groups, "threshold": "默认必须为 0", "severity": "warning", "recommendation": "确认一客多行是否符合业务定义"})
        recommendations.append("确认重复主键是有效多笔记录还是重复导入。")

    # 先初始化为空表，保证极端情况下（无特征或无月度标签）摘要也能正常返回。
    low_iv = high_iv = high_missing = missing_warning = pl.DataFrame()
    if overview.height:
        low_iv = overview.filter(pl.col("iv").is_not_null() & (pl.col("iv") < limits["iv_delete"]))
        high_iv = overview.filter(pl.col("iv").is_not_null() & (pl.col("iv") >= limits["iv_leakage"]))
        high_missing = overview.filter(pl.col("missing_rate") >= limits["missing_delete"])
        missing_warning = overview.filter(pl.col("missing_rate") >= limits["missing_warning"])
        if low_iv.height:
            findings.append({"name": "弱区分度特征", "value": low_iv.height, "threshold": f"IV < {limits['iv_delete']}", "severity": "warning", "recommendation": "作为删除候选，结合业务解释性复核"})
            recommendations.append(f"复核 IV<{limits['iv_delete']} 的 {low_iv.height} 个特征，可优先删除。")
        if high_iv.height:
            findings.append({"name": "疑似高区分度/泄漏特征", "value": high_iv.height, "threshold": f"IV ≥ {limits['iv_leakage']}", "severity": "warning", "recommendation": "检查是否使用了结果发生后的字段"})
            recommendations.append("检查 IV 过高特征是否存在时间穿越或结果字段泄漏。")
        if high_missing.height:
            findings.append({"name": "高缺失特征", "value": high_missing.height, "threshold": f"缺失率 ≥ {limits['missing_delete']:.0%}", "severity": "warning", "recommendation": "删除候选或设计专门缺失分箱"})
        elif missing_warning.height:
            findings.append({"name": "较高缺失特征", "value": missing_warning.height, "threshold": f"缺失率 ≥ {limits['missing_warning']:.0%}", "severity": "info", "recommendation": "确认缺失机制并选择填补或缺失分箱"})

    if psi_summary.height:
        high_psi = psi_summary.filter(pl.col("max_psi") >= limits["psi_warning"]).sort("max_psi", descending=True)
        stable_psi = psi_summary.filter(pl.col("max_psi") < 0.10).sort("max_psi")
        if high_psi.height:
            findings.append({"name": "高 PSI 特征", "value": high_psi.height, "threshold": f"最大 PSI ≥ {limits['psi_warning']}", "severity": "warning", "recommendation": "复核时间口径、数据来源和变量漂移，不建议直接批量删除"})
            recommendations.append(f"重点复核 PSI≥{limits['psi_warning']} 的 {high_psi.height} 个特征。")
        psi_summary_meta = {
            "high_count": high_psi.height,
            "stable_count": stable_psi.height,
            "high_top10": _rows(high_psi),
            "stable_top10": _rows(stable_psi),
        }
    else:
        psi_summary_meta = {"high_count": 0, "stable_count": 0, "high_top10": [], "stable_top10": []}

    high_corr = correlation_pairs.filter(pl.col("abs_correlation") >= limits["corr_warning"]) if correlation_pairs.height else correlation_pairs
    if high_corr.height:
        findings.append({"name": "高相关字段对", "value": high_corr.height, "threshold": f"|r| ≥ {limits['corr_warning']}", "severity": "info", "recommendation": "按 IV、PSI、缺失率和业务解释性择优"})
        recommendations.append(f"对 {high_corr.height} 组高相关字段进行择优，避免重复信息入模。")

    monthly_meta: dict[str, Any] = {"low_bad_rate_top10": [], "high_bad_rate_top10": [], "small_sample_top10": [], "zero_bad_months": 0}
    small_months = pl.DataFrame()
    zero_bad = 0
    if monthly.height and "sample_count" in monthly.columns:
        counts = [float(value) for value in monthly.get_column("sample_count").to_list() if value is not None]
        typical = median(counts) if counts else 0.0
        small_months = monthly.filter(pl.col("sample_count") < typical * limits["small_month_ratio"]).sort("sample_count") if typical else pl.DataFrame()
        low_bad = monthly.sort("bad_rate").head(10) if "bad_rate" in monthly.columns else pl.DataFrame()
        high_bad = monthly.sort("bad_rate", descending=True).head(10) if "bad_rate" in monthly.columns else pl.DataFrame()
        zero_bad = monthly.filter(pl.col("bad_count") == 0).height if "bad_count" in monthly.columns else 0
        monthly_meta = {"low_bad_rate_top10": _rows(low_bad), "high_bad_rate_top10": _rows(high_bad), "small_sample_top10": _rows(small_months), "zero_bad_months": zero_bad, "median_sample_count": typical}
        if small_months.height:
            findings.append({"name": "小样本月份", "value": small_months.height, "threshold": f"样本量 < 月中位数的 {limits['small_month_ratio']:.0%}", "severity": "warning", "recommendation": "核对是否为不完整月份，再决定时间窗口"})
            recommendations.append("最新或小样本月份不要单独用于判断模型效果。")
        if zero_bad:
            findings.append({"name": "无坏样本月份", "value": zero_bad, "threshold": "坏样本数必须结合样本量判断", "severity": "warning", "recommendation": "检查标签延迟、月份完整性和业务规则变化"})

    volatile_features: list[dict[str, Any]] = []
    if monthly_discrimination.height and "feature" in monthly_discrimination.columns and "ks" in monthly_discrimination.columns:
        for feature, group in monthly_discrimination.group_by("feature", maintain_order=True):
            values = [float(value) for value in group.get_column("ks").to_list() if value is not None]
            if len(values) >= 2:
                avg = sum(values) / len(values)
                std = (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5
                volatile_features.append({"feature": feature[0] if isinstance(feature, tuple) else feature, "monthly_ks_mean": avg, "monthly_ks_std": std})
        volatile_features.sort(key=lambda row: row["monthly_ks_std"], reverse=True)
        volatile_count = sum(row["monthly_ks_std"] >= limits["monthly_ks_std"] for row in volatile_features)
        if volatile_count:
            findings.append({"name": "月度区分度波动", "value": volatile_count, "threshold": f"月度 KS 标准差 ≥ {limits['monthly_ks_std']}", "severity": "warning", "recommendation": "检查变量在不同月份的排序稳定性"})

    severity_order = {"blocker": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda item: severity_order.get(item["severity"], 9))
    status = "blocker" if any(item["severity"] == "blocker" for item in findings) else ("warning" if findings else "normal")
    low_iv_candidates = overview.filter(pl.col("iv").is_not_null() & (pl.col("iv") < limits["iv_delete"])) if overview.height and "iv" in overview.columns else pl.DataFrame()
    low_iv_rows = _rows(low_iv_candidates.sort("iv").head(10)) if low_iv_candidates.height else (_rows(overview.filter(pl.col("iv").is_not_null()).sort("iv").head(10)) if overview.height and "iv" in overview.columns else [])
    # max_psi 位于独立的 psi_summary 表中；补回到低 IV Top10，避免报告显示为空。
    psi_by_feature = {
        str(row.get("feature")): row.get("max_psi")
        for row in (psi_summary.to_dicts() if psi_summary.height else [])
    }
    for row in low_iv_rows:
        if row.get("max_psi") is None:
            row["max_psi"] = psi_by_feature.get(str(row.get("feature")))
    high_missing_candidates = overview.filter(pl.col("missing_rate") >= limits["missing_warning"]) if overview.height and "missing_rate" in overview.columns else pl.DataFrame()
    high_missing_rows = _rows(high_missing_candidates.sort("missing_rate", descending=True).head(10)) if high_missing_candidates.height else (_rows(overview.sort("missing_rate", descending=True).head(10)) if overview.height and "missing_rate" in overview.columns else [])
    # 为对话摘要补充“证据 + 动作”，让用户知道为什么需要关注以及下一步怎么处理。
    zero_bad_rows = (
        _rows(monthly.filter(pl.col("bad_count") == 0).sort("sample_count"))
        if monthly.height and "bad_count" in monthly.columns
        else []
    )
    corr_top_rows = _rows(
        high_corr.sort("abs_correlation", descending=True) if high_corr.height else high_corr,
        limit=3,
    )
    corr_top = ", ".join(
        f"{row.get('feature_1')}↔{row.get('feature_2')}({_fmt(row.get('correlation'))})"
        for row in corr_top_rows
    ) or "暂无"
    small_month_count = int(small_months.height) if "small_months" in locals() else 0
    evidence_by_name: dict[str, str] = {
        "缺失标签": f"{missing_labels} 条标签为空",
        "重复主键组": f"{duplicate_groups} 组主键重复",
        "弱区分度特征": f"{low_iv_candidates.height} 个特征 IV<{limits['iv_delete']}；最低 IV 特征：{_feature_values(low_iv_rows, 'iv')}",
        "疑似高区分度/泄漏特征": f"{high_iv.height} 个特征 IV≥{limits['iv_leakage']}",
        "高缺失特征": f"{high_missing_candidates.height} 个特征缺失率≥{limits['missing_delete']:.0%}；最高缺失：{_feature_values(high_missing_rows, 'missing_rate', percent=True)}",
        "较高缺失特征": f"{missing_warning.height} 个特征缺失率≥{limits['missing_warning']:.0%}；最高缺失：{_feature_values(high_missing_rows, 'missing_rate', percent=True)}",
        "高 PSI 特征": f"{psi_summary_meta['high_count']} 个特征最大 PSI≥{limits['psi_warning']}；最高 PSI：{_feature_values(psi_summary_meta['high_top10'], 'max_psi')}",
        "高相关字段对": f"{high_corr.height} 组 |r|≥{limits['corr_warning']}；典型字段对：{corr_top}",
        "小样本月份": f"{small_month_count} 个月样本量低于月中位数的 {limits['small_month_ratio']:.0%}；月份：{', '.join(str(row.get('event_month')) for row in monthly_meta.get('small_sample_top10', [])[:3]) or '暂无'}",
        "无坏样本月份": f"{zero_bad} 个月坏样本数为 0；月份：{', '.join(str(row.get('event_month')) for row in zero_bad_rows[:3]) or '暂无'}",
        "月度区分度波动": f"{sum(row['monthly_ks_std'] >= limits['monthly_ks_std'] for row in volatile_features)} 个特征月度 KS 标准差≥{limits['monthly_ks_std']}；波动较大：{_feature_values(volatile_features, 'monthly_ks_std')}",
    }
    for item in findings:
        item["severity_label"] = _severity_label(item["severity"])
        item["evidence"] = evidence_by_name.get(item["name"], str(item.get("value", "—")))

    summary_lines = ["## EDA 与样本诊断", "", f"诊断状态：**{_severity_label(status)}**，共发现 **{len(findings)}** 项需要关注的问题。", ""]
    summary_lines.append("| 诊断项 | 当前结果 | 判断标准 | 风险 |")
    summary_lines.append("|---|---:|---|---|")
    summary_lines.extend(f"| {item['name']} | {item['value']} | {item['threshold']} | {item['severity_label']} |" for item in findings)
    low_iv_title = "弱区分度特征（IV<阈值）" if low_iv_candidates.height else "区分度最低特征 Top10（未发现低于阈值的特征）"
    missing_title = "高缺失特征 Top10" if high_missing_candidates.height else "缺失率最高特征 Top10（未达到高缺失阈值）"
    summary_lines.extend(["", f"### {low_iv_title}", _table(["特征", "信息价值（IV）", "KS", "最大 PSI"], [[row.get("feature"), _fmt(row.get("iv")), _fmt(row.get("ks")), _fmt(row.get("max_psi"))] for row in low_iv_rows]), "", "### 高 PSI 特征 Top10", _table(["特征", "平均 PSI", "最大 PSI"], [[row.get("feature"), _fmt(row.get("average_psi")), _fmt(row.get("max_psi"))] for row in psi_summary_meta["high_top10"]]), "", "### PSI 最稳定特征 Top10", _table(["特征", "平均 PSI", "最大 PSI"], [[row.get("feature"), _fmt(row.get("average_psi")), _fmt(row.get("max_psi"))] for row in psi_summary_meta["stable_top10"]]), "", f"### {missing_title}", _table(["特征", "缺失率", "信息价值（IV）", "KS"], [[row.get("feature"), _fmt(row.get("missing_rate"), percent=True), _fmt(row.get("iv")), _fmt(row.get("ks"))] for row in high_missing_rows]), "", "### 坏账率最低月份 Top10", _table(["月份", "样本数", "坏样本数", "坏账率"], [[row.get("event_month"), row.get("sample_count"), row.get("bad_count"), _fmt(row.get("bad_rate"), percent=True)] for row in monthly_meta["low_bad_rate_top10"]]), "", "### 高相关字段 Top10", _table(["特征1", "特征2", "相关系数"], [[row.get("feature_1"), row.get("feature_2"), _fmt(row.get("correlation"))] for row in _rows(high_corr.sort("abs_correlation", descending=True) if high_corr.height else high_corr)]), "", "### 小样本月份 Top10", _table(["月份", "样本数", "坏样本数", "坏账率"], [[row.get("event_month"), row.get("sample_count"), row.get("bad_count"), _fmt(row.get("bad_rate"), percent=True)] for row in monthly_meta["small_sample_top10"]]), "", "### 建模建议"])
    summary_lines.extend(f"- {recommendation}" for recommendation in (recommendations or ["当前未发现明显样本风险，可进入特征处理阶段。"]))
    # 对话只展示核心证据；完整 Top10 和诊断明细仍保留在 summary_markdown/HTML 报告中。
    feature_count = overview.height
    iv_values = [float(value) for value in overview.get_column("iv").to_list() if value is not None] if overview.height and "iv" in overview.columns else []
    ks_values = [float(value) for value in overview.get_column("ks").to_list() if value is not None] if overview.height and "ks" in overview.columns else []
    missing_values = [(str(row.get("feature")), float(row.get("missing_rate"))) for row in _rows(overview.sort("missing_rate", descending=True).head(1)) if row.get("missing_rate") is not None] if overview.height and "missing_rate" in overview.columns else []
    psi_values = [(str(row.get("feature")), float(row.get("max_psi"))) for row in _rows(psi_summary.sort("max_psi", descending=True).head(1)) if row.get("max_psi") is not None] if psi_summary.height else []
    overall_bad_rate = None
    if monthly.height and "sample_count" in monthly.columns and "bad_count" in monthly.columns:
        total_month_samples = sum(float(value or 0) for value in monthly.get_column("sample_count").to_list())
        total_month_bad = sum(float(value or 0) for value in monthly.get_column("bad_count").to_list())
        overall_bad_rate = total_month_bad / total_month_samples if total_month_samples else None
    brief_lines = [
        "## EDA 与样本诊断反馈",
        "",
        f"已分析 **{total_rows:,}** 条样本、**{feature_count:,}** 个特征；整体坏样本率 **{_fmt(overall_bad_rate, percent=True)}**。",
        "本节点只提供建模证据，不自动删除样本、删除字段或改变标签。",
        "",
        "### 指标摘要",
        "",
        "| 建模关注项 | 当前结果 | 判断参考 |",
        "|---|---:|---|",
    ]
    brief_lines.append(f"| 区分度 | IV最高 {_fmt(max(iv_values) if iv_values else None)}；KS最高 {_fmt(max(ks_values) if ks_values else None)} | IV< {limits['iv_delete']} 的特征共 {low_iv_candidates.height} 个，作为删除候选复核 |")
    brief_lines.append(f"| 稳定性 | PSI≥{limits['psi_warning']} 共 {psi_summary_meta['high_count']} 个 | {('最高 PSI 为 ' + psi_values[0][0] + '（' + _fmt(psi_values[0][1]) + '），优先核对月份和数据来源' if psi_values else '暂无 PSI 结果')} |")
    brief_lines.append(f"| 缺失 | 最高缺失率 {(_fmt(missing_values[0][1], percent=True) if missing_values else '—')} | 高缺失字段先确认缺失机制，不自动删除 |")
    brief_lines.append(f"| 冗余 | 高相关字段对 {high_corr.height} 组 | 按 IV、PSI、缺失率择优，避免重复信息 |")
    brief_lines.append(f"| 时间样本 | 小样本月份 {len(monthly_meta.get('small_sample_top10', []))} 个；无坏样本月份 {monthly_meta.get('zero_bad_months', 0)} 个 | 小样本/无坏样本月份不单独判断模型效果 |")
    brief_lines.extend(["", f"### 风险与对应调整建议（共 {len(findings)} 项）", ""])
    if findings:
        brief_lines.extend(["| 风险等级 | 问题 | 证据 | 建议调整 |", "|---|---|---|---|"])
        brief_lines.extend(
            f"| {item['severity_label']} | {item['name']} | {item['evidence']} | {item['recommendation']} |"
            for item in findings
        )
    else:
        brief_lines.append("未发现超过当前规则阈值的明显风险；仍需结合业务规则和时间切分方案复核。")
    brief_lines.extend(["", "### 建模前建议", ""])
    if recommendations:
        # 去重后最多展示 6 条，保留针对性但避免重复刷屏。
        unique_recommendations = list(dict.fromkeys(recommendations))
        brief_lines.extend(f"{index}. {item}" for index, item in enumerate(unique_recommendations[:6], start=1))
    else:
        brief_lines.append("当前没有额外的样本处理建议。")
    brief_lines.extend(
        [
            "",
            "### 下一步",
            "请确认以上风险的处理原则（例如是否排除不完整月份、是否在特征筛选阶段处理高 PSI/高相关字段）。确认后进入 `sample-split + feature-screening`；本节点不会自动执行这些调整。",
        ]
    )
    return {"status": status, "findings": findings, "recommendations": recommendations, "low_iv_top10": low_iv_rows, "high_missing_top10": high_missing_rows, "psi": psi_summary_meta, "monthly": monthly_meta, "volatile_features_top10": volatile_features[:10], "summary_markdown": "\n".join(summary_lines), "brief_summary_markdown": "\n".join(brief_lines)}

"""Linux/OpenCode-compatible Excel reports built with XlsxWriter.

The report payloads are already JSON serializable and contain the deterministic
results produced by the pipeline.  This module only handles presentation: it
does not recalculate IV, KS, PSI, model metrics, or selection decisions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import xlsxwriter

from logger import get_logger


logger = get_logger(__name__)

COLORS = {
    "navy": "#17365D",
    "blue": "#2F75B5",
    "orange": "#ED7D31",
    "light_blue": "#D9EAF7",
    "pale_blue": "#EEF5FB",
    "text": "#203040",
    "muted": "#66788A",
    "line": "#D9E2EA",
    "green": "#E2F0D9",
    "green_text": "#2E7D32",
    "yellow": "#FFF2CC",
    "yellow_text": "#9C6500",
    "red": "#FCE4D6",
    "red_text": "#C00000",
    "gray": "#F2F2F2",
    "white": "#FFFFFF",
}

HEADER_LABELS = {
    "column_name": "字段名",
    "data_type": "数据类型",
    "row_count": "样本数",
    "non_null_count": "非空数",
    "missing_count": "缺失数",
    "missing_rate": "缺失率",
    "unique_count": "唯一值数",
    "unique_rate": "唯一率",
    "sample_values": "样例值",
    "top_value": "众数",
    "top_value_count": "众数样本数",
    "top_value_rate": "众数占比",
    "numeric_min": "最小值",
    "numeric_max": "最大值",
    "numeric_mean": "均值",
    "date_parse_rate": "日期解析率",
    "candidate_role": "候选角色",
    "role_reason": "判断依据",
    "decision": "处理决定",
    "reason": "原因",
    "constant_rate": "最高频值占比",
    "feature": "特征",
    "metrics_backend": "指标后端",
    "iv": "IV",
    "ks": "KS",
    "auc_adjusted": "调整后AUC",
    "auc": "AUC",
    "bin_count": "分箱数",
    "bin_order": "分箱序号",
    "bin": "分箱区间",
    "sample_count": "样本数",
    "sample_rate": "样本占比",
    "bad_count": "坏样本数",
    "good_count": "好样本数",
    "bad_rate": "坏样本率",
    "bad_distribution": "坏样本分布",
    "good_distribution": "好样本分布",
    "woe": "WOE",
    "iv_bin": "分箱IV",
    "event_month": "月份",
    "baseline_month": "基准月",
    "psi": "PSI",
    "feature_1": "特征一",
    "feature_2": "特征二",
    "method": "方法",
    "correlation": "相关系数",
    "abs_correlation": "绝对相关系数",
    "pair_count": "有效样本对数",
    "dataset": "样本集",
    "month_start": "开始月份",
    "month_end": "结束月份",
    "gain_importance": "Gain重要性",
    "split_importance": "分裂次数",
    "gain_share": "Gain占比",
    "cumulative_gain_share": "累计Gain占比",
    "iteration": "迭代轮次",
    "train_auc": "Train AUC",
    "test_auc": "Test AUC",
    "train_ks": "Train KS",
    "auc_gap": "AUC差距",
    "ks_gap": "KS差距",
    "is_best_iteration": "最佳轮次",
    "score_psi_vs_train": "相对Train分数PSI",
    "reliability": "可靠性",
    "interpretation": "解读",
    "bucket": "风险十分位",
    "score_min": "最低风险概率",
    "score_max": "最高风险概率",
    "lift": "Lift",
    "lift_10": "前10% Lift",
    "cumulative_sample_rate": "累计样本占比",
    "cumulative_bad_capture": "累计坏样本捕获率",
    "cumulative_good_capture": "累计好样本捕获率",
    "cumulative_lift": "累计Lift",
    "best_test_auc_marker": "最佳轮次Test AUC",
    "candidate": "候选模型",
    "test_ks": "Test KS",
    "trial_number": "Trial编号",
    "state": "运行状态",
    "objective_value": "目标函数值",
    "duration_seconds": "耗时（秒）",
    "fold": "CV折",
    "validation_sample_count": "验证样本数",
    "validation_bad_count": "验证坏样本数",
    "validation_auc": "验证AUC",
    "validation_ks": "验证KS",
    "code": "审查编码",
    "severity": "严重等级",
    "message": "审查结论",
    "evidence": "证据",
    "recommendation": "改进建议",
    "original_dtype": "原始类型",
    "inferred_type": "识别类型",
    "action": "预处理动作",
    "category_count": "原始类别数",
    "encoded_category_count": "编码后类别数",
    "dominant_rate": "最高频值占比",
    "max_monthly_psi": "最大月度PSI",
    "unstable_month_ratio": "不稳定月份占比",
    "eligible_month_count": "有效月份数",
    "correlated_with": "高相关保留变量",
    "correlation_method": "相关性方法",
}

VALUE_LABELS = {
    "train": "Train",
    "test": "Test",
    "oot": "OOT",
    "retained": "保留",
    "excluded": "排除",
    "candidate_id": "主键候选",
    "candidate_date": "日期候选",
    "candidate_target": "目标候选",
    "candidate_feature": "特征候选",
}


def _label(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return VALUE_LABELS.get(value, value)


def _col(index: int) -> str:
    value = index
    result = ""
    while True:
        result = chr(65 + value % 26) + result
        value = value // 26 - 1
        if value < 0:
            return result


def _table_headers(table: dict[str, Any]) -> list[str]:
    return [str(value) for value in table.get("headers", [])]


def _format_for(workbook: xlsxwriter.Workbook, header: str) -> xlsxwriter.format.Format:
    text = header.lower()
    if "rate" in text or header in {
        "sample_rate",
        "bad_distribution",
        "good_distribution",
        "gain_share",
        "cumulative_gain_share",
        "cumulative_sample_rate",
        "cumulative_bad_capture",
        "cumulative_good_capture",
    }:
        return workbook._formats["percent"]
    if any(
        token in text
        for token in ("auc", "ks", "psi", "iv", "woe", "correlation", "lift", "gap")
    ):
        return workbook._formats["decimal"]
    if "count" in text or header in {
        "bin_order",
        "bin_count",
        "iteration",
        "bucket",
        "split_importance",
    }:
        return workbook._formats["integer"]
    if header == "duration_seconds":
        return workbook._formats["seconds"]
    return workbook._formats["body"]


def _init_formats(workbook: xlsxwriter.Workbook) -> None:
    workbook._formats = {
        "title": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": COLORS["white"],
                "bg_color": COLORS["navy"],
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "subtitle": workbook.add_format(
            {
                "font_color": COLORS["muted"],
                "font_size": 10,
                "bg_color": COLORS["pale_blue"],
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": COLORS["navy"],
                "bg_color": COLORS["light_blue"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "body": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "body_wrap": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "valign": "vcenter",
                "text_wrap": True,
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "percent": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "num_format": "0.00%",
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "decimal": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "num_format": "0.0000",
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "integer": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "num_format": "#,##0",
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "seconds": workbook.add_format(
            {
                "font_color": COLORS["text"],
                "font_size": 10,
                "num_format": "0.000",
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#E8EEF3",
            }
        ),
        "section": workbook.add_format(
            {
                "bold": True,
                "font_color": COLORS["navy"],
                "bg_color": COLORS["light_blue"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "legend": workbook.add_format(
            {
                "bold": True,
                "font_size": 9,
                "font_color": COLORS["text"],
                "bg_color": COLORS["gray"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "card_label": workbook.add_format(
            {
                "bold": True,
                "font_color": COLORS["muted"],
                "bg_color": COLORS["white"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "card_value": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": COLORS["navy"],
                "bg_color": COLORS["white"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "card_bad": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": COLORS["red_text"],
                "bg_color": COLORS["red"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "card_good": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": COLORS["green_text"],
                "bg_color": COLORS["green"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "note": workbook.add_format(
            {
                "font_color": COLORS["muted"],
                "font_size": 9,
                "bg_color": COLORS["gray"],
                "text_wrap": True,
                "valign": "vcenter",
            }
        ),
        "red_cond": workbook.add_format(
            {"bg_color": COLORS["red"], "font_color": COLORS["red_text"], "bold": True}
        ),
        "yellow_cond": workbook.add_format(
            {"bg_color": COLORS["yellow"], "font_color": COLORS["yellow_text"]}
        ),
        "green_cond": workbook.add_format(
            {"bg_color": COLORS["green"], "font_color": COLORS["green_text"]}
        ),
    }


def _title(
    ws: xlsxwriter.worksheet.Worksheet,
    workbook: xlsxwriter.Workbook,
    title: str,
    subtitle: str,
    end_col: int,
) -> None:
    ws.hide_gridlines(2)
    ws.merge_range(0, 0, 0, max(end_col, 11), title, workbook._formats["title"])
    ws.set_row(0, 30)
    ws.merge_range(1, 0, 1, max(end_col, 11), subtitle, workbook._formats["subtitle"])
    ws.set_row(1, 21)


def _legend(ws, workbook: xlsxwriter.Workbook, end_col: int) -> None:
    items = [
        ("图例", COLORS["navy"]),
        ("IV / KS：越高表示区分能力越强", COLORS["pale_blue"]),
        ("缺失率 > 80%：高风险字段", COLORS["red"]),
        ("唯一率接近 100%：可能是 ID", COLORS["yellow"]),
        ("PSI < 0.10 稳定；0.10～0.25 关注；≥ 0.25 漂移", COLORS["green"]),
    ]
    # Tables have different widths.  Partition the available columns instead
    # of using fixed ranges (which overlap on narrow/wide EDA sheets).
    total_columns = max(end_col + 1, 1)
    span_width = max(1, math.ceil(total_columns / len(items)))
    spans = [
        (index * span_width, min(total_columns - 1, (index + 1) * span_width - 1))
        for index in range(len(items))
    ]
    for (text, color), (start, end) in zip(items, spans):
        if start > end:
            continue
        fmt = workbook.add_format(
            {
                "bold": True,
                "font_size": 9,
                "font_color": COLORS["white"]
                if color == COLORS["navy"]
                else COLORS["text"],
                "bg_color": color,
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        )
        ws.merge_range(2, start, 2, end, text, fmt)
    ws.set_row(2, 30)


def _width(header: str) -> int:
    label = HEADER_LABELS.get(header, header)
    if header in {
        "reason",
        "role_reason",
        "interpretation",
        "message",
        "evidence",
        "recommendation",
    }:
        return 34
    if header == "sample_values":
        return 27
    if header in {"feature", "column_name", "feature_1", "feature_2", "bin"}:
        return 16
    return min(max(len(label) * 2 + 3, 11), 20)


def _write_table(
    workbook: xlsxwriter.Workbook,
    sheet_name: str,
    title: str,
    subtitle: str,
    table: dict[str, Any],
    *,
    legend: bool = False,
    tab_color: str | None = None,
) -> dict[str, Any]:
    headers = _table_headers(table)
    rows = [[_label(value) for value in row] for row in table.get("rows", [])]
    ws = workbook.add_worksheet(sheet_name[:31])
    if tab_color:
        ws.set_tab_color(tab_color)
    last_col = max(len(headers) - 1, 0)
    _title(ws, workbook, title, subtitle, last_col)
    if legend:
        _legend(ws, workbook, last_col)
    header_row = 3
    data_row = 4
    ws.write_row(
        header_row,
        0,
        [HEADER_LABELS.get(header, header) for header in headers],
        workbook._formats["header"],
    )
    for row_offset, row in enumerate(rows):
        for col, value in enumerate(row):
            header = headers[col]
            fmt = _format_for(workbook, header)
            if header in {
                "reason",
                "role_reason",
                "interpretation",
                "message",
                "evidence",
                "recommendation",
            }:
                fmt = workbook._formats["body_wrap"]
            ws.write(data_row + row_offset, col, value, fmt)
    if rows:
        ws.add_table(
            header_row,
            0,
            data_row + len(rows) - 1,
            last_col,
            {
                "style": "Table Style Medium 2",
                "columns": [
                    {"header": HEADER_LABELS.get(header, header)} for header in headers
                ],
            },
        )
    for col, header in enumerate(headers):
        ws.set_column(col, col, _width(header))
    ws.freeze_panes(data_row, 1 if len(headers) > 1 else 0)
    return {
        "sheet": ws,
        "workbook": workbook,
        "headers": headers,
        "rows": rows,
        "header_row": header_row,
        "data_row": data_row,
        "last_col": last_col,
    }


def _index(headers: list[str], name: str) -> int:
    try:
        return headers.index(name)
    except ValueError:
        return -1


def _conditional_quality(table: dict[str, Any]) -> None:
    ws, workbook, headers, start, end = (
        table["sheet"],
        table["workbook"],
        table["headers"],
        table["data_row"],
        table["data_row"] + len(table["rows"]) - 1,
    )
    if end < start:
        return
    missing = _index(headers, "missing_rate")
    if missing >= 0:
        ws.conditional_format(
            start,
            missing,
            end,
            missing,
            {
                "type": "cell",
                "criteria": ">=",
                "value": 0.8,
                "format": workbook._formats["red_cond"],
            },
        )
        ws.conditional_format(
            start,
            missing,
            end,
            missing,
            {
                "type": "cell",
                "criteria": "between",
                "minimum": 0.5,
                "maximum": 0.799999,
                "format": workbook._formats["yellow_cond"],
            },
        )
    unique = _index(headers, "unique_rate")
    if unique >= 0:
        ws.conditional_format(
            start,
            unique,
            end,
            unique,
            {
                "type": "cell",
                "criteria": ">=",
                "value": 0.98,
                "format": workbook._formats["yellow_cond"],
            },
        )
    for name in ("iv", "ks", "objective_value"):
        index = _index(headers, name)
        if index >= 0:
            ws.conditional_format(
                start,
                index,
                end,
                index,
                {"type": "data_bar", "bar_color": COLORS["blue"]},
            )


def _merge_field_tables(
    profile: dict[str, Any], univariate: dict[str, Any]
) -> dict[str, Any]:
    headers = list(profile["headers"])
    for header in univariate["headers"]:
        if header != "feature" and header not in headers:
            headers.append(header)
    pidx = _index(profile["headers"], "column_name")
    uidx = _index(univariate["headers"], "feature")
    by_feature = {str(row[uidx]): row for row in univariate["rows"]}
    rows: list[list[Any]] = []
    for profile_row in profile["rows"]:
        feature_row = by_feature.get(str(profile_row[pidx]))
        values = []
        for header in headers:
            if header in profile["headers"]:
                values.append(profile_row[_index(profile["headers"], header)])
            elif feature_row is not None and header in univariate["headers"]:
                values.append(feature_row[_index(univariate["headers"], header)])
            else:
                values.append(None)
        rows.append(values)
    return {"headers": headers, "rows": rows}


def _add_top_iv(
    workbook: xlsxwriter.Workbook, table: dict[str, Any], univariate: dict[str, Any]
) -> int:
    ws = table["sheet"]
    start = table["data_row"] + len(table["rows"]) + 4
    ws.merge_range(
        start, 0, start, 2, "Top 10 特征 IV（辅助图例）", workbook._formats["section"]
    )
    ws.write_row(start + 1, 0, ["特征", "IV"], workbook._formats["header"])
    feature_index = _index(univariate["headers"], "feature")
    iv_index = _index(univariate["headers"], "iv")
    top_rows = sorted(
        univariate["rows"], key=lambda row: float(row[iv_index] or 0), reverse=True
    )[:10]
    for offset, row in enumerate(top_rows):
        ws.write(start + 2 + offset, 0, row[feature_index], workbook._formats["body"])
        ws.write(start + 2 + offset, 1, row[iv_index], workbook._formats["decimal"])
    if top_rows:
        ws.conditional_format(
            start + 2,
            1,
            start + 1 + len(top_rows),
            1,
            {"type": "data_bar", "bar_color": COLORS["blue"]},
        )
        chart = workbook.add_chart({"type": "bar"})
        chart.add_series(
            {
                "name": "IV",
                "categories": [ws.name, start + 2, 0, start + 1 + len(top_rows), 0],
                "values": [ws.name, start + 2, 1, start + 1 + len(top_rows), 1],
                "fill": {"color": COLORS["blue"]},
            }
        )
        chart.set_title({"name": "Top 10 特征 IV"})
        chart.set_legend({"none": True})
        chart.set_x_axis({"min": 0, "num_format": "0.00"})
        chart.set_size({"width": 650, "height": 300})
        ws.insert_chart(start, 3, chart)
    return start + 2


def _add_top_missing(
    workbook: xlsxwriter.Workbook, table: dict[str, Any]
) -> tuple[int, int]:
    """Add a compact missing-rate ranking table and return its data range."""
    ws = table["sheet"]
    headers = table["headers"]
    feature_index = _index(headers, "column_name")
    missing_index = _index(headers, "missing_rate")
    if feature_index < 0 or missing_index < 0:
        return 0, 0
    rows = sorted(
        table["rows"],
        key=lambda row: float(row[missing_index] or 0),
        reverse=True,
    )[:10]
    start = table["data_row"] + len(table["rows"]) + 24
    ws.merge_range(
        start,
        0,
        start,
        2,
        "Top 10 字段缺失率（辅助图例）",
        workbook._formats["section"],
    )
    ws.write_row(start + 1, 0, ["字段", "缺失率"], workbook._formats["header"])
    for offset, row in enumerate(rows):
        ws.write(start + 2 + offset, 0, row[feature_index], workbook._formats["body"])
        ws.write(start + 2 + offset, 1, row[missing_index], workbook._formats["percent"])
    if rows:
        ws.conditional_format(
            start + 2,
            1,
            start + 1 + len(rows),
            1,
            {"type": "data_bar", "bar_color": COLORS["orange"]},
        )
    return start + 2, len(rows)


def _add_psi_summary(
    workbook: xlsxwriter.Workbook, table: dict[str, Any]
) -> tuple[int, int]:
    """Add max-PSI-by-feature ranking used by the dashboard chart."""
    ws = table["sheet"]
    headers = table["headers"]
    feature_index = _index(headers, "feature")
    psi_index = _index(headers, "psi")
    if feature_index < 0 or psi_index < 0:
        return 0, 0
    max_by_feature: dict[str, float] = {}
    for row in table["rows"]:
        feature = str(row[feature_index])
        value = float(row[psi_index] or 0)
        max_by_feature[feature] = max(max_by_feature.get(feature, 0.0), value)
    rows = sorted(max_by_feature.items(), key=lambda item: item[1], reverse=True)[:10]
    start = table["data_row"] + len(table["rows"]) + 4
    ws.merge_range(
        start,
        7,
        start,
        9,
        "Top 10 特征最大月度 PSI（辅助图例）",
        workbook._formats["section"],
    )
    ws.write_row(start + 1, 7, ["特征", "最大PSI"], workbook._formats["header"])
    for offset, (feature, value) in enumerate(rows):
        ws.write(start + 2 + offset, 7, feature, workbook._formats["body"])
        ws.write(start + 2 + offset, 8, value, workbook._formats["decimal"])
    if rows:
        ws.conditional_format(
            start + 2,
            8,
            start + 1 + len(rows),
            8,
            {"type": "data_bar", "bar_color": COLORS["red_text"]},
        )
    return start + 2, len(rows)


def _write_eda_dashboard(workbook: xlsxwriter.Workbook, payload: dict[str, Any]) -> Any:
    ws = workbook.add_worksheet("1.EDA总览")
    _title(
        ws, workbook, "风控数据 EDA 报告", "结论优先 · 异常优先 · 完整明细可追溯", 11
    )
    for col in range(12):
        ws.set_column(col, col, 12)
    summary = dict(payload["tables"]["1.数据概况"]["rows"])
    monthly = payload["tables"].get("6.月度样本", {"headers": [], "rows": []})
    mheaders = monthly["headers"]
    mrows = monthly["rows"]
    sample_idx, bad_idx, month_idx, rate_idx = (
        _index(mheaders, name)
        for name in ("sample_count", "bad_count", "event_month", "bad_rate")
    )
    latest = (
        sorted(mrows, key=lambda row: str(row[month_idx]))[-1]
        if mrows
        else [None] * len(mheaders)
    )
    psi_rows = payload["tables"].get("7.稳定性PSI", {}).get("rows", [])
    psi_idx = _index(payload["tables"].get("7.稳定性PSI", {}).get("headers", []), "psi")
    max_psi = max((float(row[psi_idx] or 0) for row in psi_rows), default=0.0)
    unstable = (
        len({str(row[0]) for row in psi_rows if float(row[psi_idx] or 0) >= 0.25})
        if psi_idx >= 0
        else 0
    )
    status = (
        "需重点关注"
        if (latest[sample_idx] if sample_idx >= 0 else 0) < 30 or unstable
        else "整体正常"
    )

    def card(
        row: int,
        col: int,
        label: str,
        value: Any,
        number_format: str | None = None,
        bad: bool = False,
    ) -> None:
        ws.merge_range(row, col, row, col + 2, label, workbook._formats["card_label"])
        fmt = workbook._formats["card_bad"] if bad else workbook._formats["card_value"]
        if number_format:
            fmt = workbook.add_format(
                {
                    "bold": True,
                    "font_size": 18,
                    "font_color": COLORS["red_text"] if bad else COLORS["navy"],
                    "bg_color": COLORS["red"] if bad else COLORS["white"],
                    "border": 1,
                    "border_color": COLORS["line"],
                    "align": "center",
                    "valign": "vcenter",
                    "num_format": number_format,
                }
            )
        ws.merge_range(row + 1, col, row + 2, col + 2, value, fmt)

    card(3, 0, "数据质量状态", status, bad=status != "整体正常")
    card(3, 3, "原始样本数", summary.get("原始样本数"), "#,##0")
    card(3, 6, "原始坏样本率", summary.get("原始坏样本率"), "0.00%")
    card(3, 9, "保留特征数", summary.get("保留特征数"), "#,##0")
    card(
        7,
        0,
        "最新月份样本数",
        latest[sample_idx] if sample_idx >= 0 else None,
        "#,##0",
        bad=(latest[sample_idx] if sample_idx >= 0 else 0) < 30,
    )
    card(7, 3, "PSI异常特征数", unstable, "#,##0", bad=unstable > 0)
    card(7, 6, "最大PSI", max_psi, "0.0000", bad=max_psi >= 0.25)
    ws.merge_range(11, 0, 11, 11, "自动诊断与建议", workbook._formats["section"])
    ws.write_row(
        12,
        0,
        ["等级", "", "发现", "", "", "", "", "", "建议", "", "", ""],
        workbook._formats["header"],
    )
    findings = [
        [
            "重点" if (latest[sample_idx] if sample_idx >= 0 else 0) < 30 else "正常",
            f"最新月份：{latest[month_idx] if month_idx >= 0 else '-'}，样本 {(latest[sample_idx] if sample_idx >= 0 else 0)} 条",
            "确认是否为未完整月份；不足时不要做稳定性结论",
        ],
        [
            "重点" if unstable else "正常",
            f"特征稳定性：{unstable} 个特征最大 PSI ≥ 0.25，最大值 {max_psi:.4f}",
            "优先复核漂移特征的数据口径和分布变化" if unstable else "当前未见明显漂移",
        ],
    ]
    for offset, (level, finding, suggestion) in enumerate(findings, start=13):
        ws.merge_range(
            offset,
            0,
            offset,
            1,
            level,
            workbook._formats["card_bad"]
            if level == "重点"
            else workbook._formats["card_good"],
        )
        ws.merge_range(offset, 2, offset, 7, finding, workbook._formats["body_wrap"])
        ws.merge_range(
            offset, 8, offset, 11, suggestion, workbook._formats["body_wrap"]
        )
        ws.set_row(offset, 28)
    ws.merge_range(18, 0, 18, 5, "月度坏样本率趋势", workbook._formats["section"])
    ws.merge_range(18, 6, 18, 11, "Top 10 特征 IV", workbook._formats["section"])
    ws.merge_range(34, 0, 34, 5, "Top 10 特征最大月度 PSI", workbook._formats["section"])
    ws.merge_range(34, 6, 34, 11, "Top 10 字段缺失率", workbook._formats["section"])
    ws.merge_range(
        50,
        0,
        50,
        11,
        "阈值口径：PSI < 0.10 稳定，0.10～0.25 关注，≥ 0.25 明显漂移；绝对相关系数 ≥ 0.80 视为高相关。",
        workbook._formats["note"],
    )
    return ws


def write_eda_report(payload_path: str | Path, output_path: str | Path) -> Path:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output), {"nan_inf_to_errors": False})
    _init_formats(workbook)
    dashboard = _write_eda_dashboard(workbook, payload)
    quality = _write_table(
        workbook,
        "2.数据质量",
        "数据质量与字段处理",
        "异常优先查看；完整处理结果保留在本表",
        payload["tables"]["3.特征质量"],
    )
    _conditional_quality(quality)
    monthly = _write_table(
        workbook,
        "3.月度样本",
        "月度样本与标签分布",
        "同时观察样本规模和坏样本率，避免对小样本月份过度解读",
        payload["tables"]["6.月度样本"],
    )
    field = _write_table(
        workbook,
        "4.字段与单变量分析",
        "字段画像与单变量分析",
        "字段质量、候选角色、IV/KS 和分箱信息集中查看；先看异常，再看预测能力",
        _merge_field_tables(
            payload["tables"]["2.字段画像"], payload["tables"]["4.单变量概览"]
        ),
        legend=True,
    )
    _conditional_quality(field)
    top_start = _add_top_iv(workbook, field, payload["tables"]["4.单变量概览"])
    missing_start, missing_count = _add_top_missing(workbook, field)
    monthly_source = payload["tables"].get("6.月度样本", {"headers": [], "rows": []})
    mheaders = monthly_source.get("headers", [])
    mrows = monthly_source.get("rows", [])
    month_idx = _index(mheaders, "event_month")
    sample_idx = _index(mheaders, "sample_count")
    rate_idx = _index(mheaders, "bad_rate")
    stability = _write_table(
        workbook,
        "5.稳定性分析",
        "特征稳定性分析（PSI）",
        "固定阈值着色：绿色稳定，黄色关注，红色明显漂移",
        payload["tables"]["7.稳定性PSI"],
    )
    psi_idx = _index(stability["headers"], "psi")
    if psi_idx >= 0 and stability["rows"]:
        stability["sheet"].conditional_format(
            stability["data_row"],
            psi_idx,
            stability["data_row"] + len(stability["rows"]) - 1,
            psi_idx,
            {
                "type": "3_color_scale",
                "min_color": COLORS["green"],
                "mid_color": COLORS["yellow"],
                "max_color": COLORS["red"],
            },
        )
    psi_start, psi_count = _add_psi_summary(workbook, stability)
    _write_table(
        workbook,
        "6.相关性分析",
        "特征相关性分析",
        "优先查看绝对相关系数较高的字段对",
        payload["tables"]["8.相关性"],
    )
    binning = _write_table(
        workbook,
        "7.分箱明细",
        "特征分箱明细",
        "主报告关注单变量结论；本页保留完整分箱、WOE、IV 和 KS 计算结果",
        payload["tables"]["5.分箱明细"],
    )
    _add_binning_chart(binning)
    monthly_discrimination = payload["tables"].get("9.月度区分度")
    if monthly_discrimination:
        monthly_disc = _write_table(
            workbook,
            "8.月度区分度",
            "变量月度区分度",
            "按月份查看特征 KS、AUC、PSI 和前 10% Lift；缺少正/负样本的月份会显示为空",
            monthly_discrimination,
            legend=True,
        )
        for metric in ("ks", "auc", "psi", "lift_10"):
            index = _index(monthly_disc["headers"], metric)
            if index >= 0 and monthly_disc["rows"]:
                monthly_disc["sheet"].conditional_format(
                    monthly_disc["data_row"],
                    index,
                    monthly_disc["data_row"] + len(monthly_disc["rows"]) - 1,
                    index,
                    {
                        "type": "3_color_scale",
                        "min_color": COLORS["green"],
                        "mid_color": COLORS["yellow"],
                        "max_color": COLORS["red"],
                    },
                )
    ks_bucket = payload["tables"].get("10.KS十分位")
    if ks_bucket:
        ks_table = _write_table(
            workbook,
            "9.KS十分位",
            "变量 KS 十分位与 Lift",
            "按风险排序的分位桶，保留坏样本捕获率、KS、Lift 和累计 Lift，便于检查排序性",
            ks_bucket,
            legend=True,
        )
        for metric in ("ks", "lift", "cumulative_lift"):
            index = _index(ks_table["headers"], metric)
            if index >= 0 and ks_table["rows"]:
                ks_table["sheet"].conditional_format(
                    ks_table["data_row"],
                    index,
                    ks_table["data_row"] + len(ks_table["rows"]) - 1,
                    index,
                    {
                        "type": "3_color_scale",
                        "min_color": COLORS["green"],
                        "mid_color": COLORS["yellow"],
                        "max_color": COLORS["red"],
                    },
                )
    if mrows and sample_idx >= 0 and rate_idx >= 0:
        chart = workbook.add_chart({"type": "line"})
        chart.add_series(
            {
                "name": "坏样本率",
                "categories": [
                    monthly["sheet"].name,
                    4,
                    month_idx,
                    3 + len(mrows),
                    month_idx,
                ],
                "values": [
                    monthly["sheet"].name,
                    4,
                    rate_idx,
                    3 + len(mrows),
                    rate_idx,
                ],
                "line": {"color": COLORS["blue"]},
            }
        )
        chart.set_title({"name": "月度坏样本率"})
        chart.set_y_axis({"num_format": "0.0%", "min": 0})
        chart.set_size({"width": 430, "height": 270})
        dashboard.insert_chart(19, 0, chart)
    top_count = min(10, len(payload["tables"]["4.单变量概览"]["rows"]))
    if top_count:
        chart = workbook.add_chart({"type": "bar"})
        chart.add_series(
            {
                "name": "IV",
                "categories": [
                    field["sheet"].name,
                    top_start,
                    0,
                    top_start + top_count - 1,
                    0,
                ],
                "values": [
                    field["sheet"].name,
                    top_start,
                    1,
                    top_start + top_count - 1,
                    1,
                ],
                "fill": {"color": COLORS["blue"]},
            }
        )
        chart.set_title({"name": "Top 10 特征 IV"})
        chart.set_legend({"none": True})
        chart.set_x_axis({"min": 0, "num_format": "0.00"})
        chart.set_size({"width": 430, "height": 270})
        dashboard.insert_chart(19, 6, chart)
    if psi_count:
        chart = workbook.add_chart({"type": "bar"})
        chart.add_series(
            {
                "name": "最大月度 PSI",
                "categories": [stability["sheet"].name, psi_start, 7, psi_start + psi_count - 1, 7],
                "values": [stability["sheet"].name, psi_start, 8, psi_start + psi_count - 1, 8],
                "fill": {"color": COLORS["red_text"]},
            }
        )
        chart.set_title({"name": "Top 10 特征最大月度 PSI"})
        chart.set_legend({"none": True})
        chart.set_x_axis({"min": 0, "num_format": "0.00"})
        chart.set_size({"width": 430, "height": 270})
        dashboard.insert_chart(35, 0, chart)
    if missing_count:
        chart = workbook.add_chart({"type": "bar"})
        chart.add_series(
            {
                "name": "缺失率",
                "categories": [field["sheet"].name, missing_start, 0, missing_start + missing_count - 1, 0],
                "values": [field["sheet"].name, missing_start, 1, missing_start + missing_count - 1, 1],
                "fill": {"color": COLORS["orange"]},
            }
        )
        chart.set_title({"name": "Top 10 字段缺失率"})
        chart.set_legend({"none": True})
        chart.set_x_axis({"min": 0, "num_format": "0%"})
        chart.set_size({"width": 430, "height": 270})
        dashboard.insert_chart(35, 6, chart)
    workbook.close()
    logger.info("EDA xlsx report created: %s", output)
    return output


def _reliability(bad_count: Any, sample_count: Any) -> str:
    try:
        bad = int(bad_count)
        good = int(sample_count) - bad
    except (TypeError, ValueError):
        return "样本不足"
    if bad < 10 or good < 10:
        return "样本不足"
    if bad < 30 or good < 30:
        return "仅方向性"
    return "可评价"


def _augment_model_table(name: str, table: dict[str, Any]) -> dict[str, Any]:
    headers = list(table["headers"])
    rows = [list(row) for row in table["rows"]]
    if (
        name in {"2.数据切分", "6.效果指标", "8.月度表现"}
        and "reliability" not in headers
    ):
        headers += ["reliability", "interpretation"]
        dataset_idx, bad_idx, sample_idx = (
            _index(table["headers"], key)
            for key in ("dataset", "bad_count", "sample_count")
        )
        new_rows = []
        for row in rows:
            level = _reliability(
                row[bad_idx] if bad_idx >= 0 else 0,
                row[sample_idx] if sample_idx >= 0 else 0,
            )
            dataset = row[dataset_idx] if dataset_idx >= 0 else ""
            note = (
                "OOT 独立时间外验证；不得用于调参和早停"
                if dataset == "oot"
                else "Test 用于开发期泛化判断"
                if dataset == "test"
                else "Train 拟合基准"
            )
            new_rows.append(row + [level, note])
        rows = new_rows
    return {"headers": headers, "rows": rows}


def _augment_lift_table(table: dict[str, Any]) -> dict[str, Any]:
    """Add cumulative capture fields used by the Lift chart."""
    headers = list(table.get("headers", []))
    rows = [list(row) for row in table.get("rows", [])]
    additions = [
        "cumulative_sample_rate",
        "cumulative_bad_capture",
        "cumulative_lift",
        "reliability",
    ]
    if all(name in headers for name in additions):
        return {"headers": headers, "rows": rows}
    dataset_idx = _index(headers, "dataset")
    sample_idx = _index(headers, "sample_count")
    bad_idx = _index(headers, "bad_count")
    grouped: dict[Any, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row[dataset_idx] if dataset_idx >= 0 else "all", []).append(index)
    output = [None] * len(rows)
    for indices in grouped.values():
        total_samples = sum(int(rows[index][sample_idx] or 0) for index in indices) if sample_idx >= 0 else 0
        total_bad = sum(int(rows[index][bad_idx] or 0) for index in indices) if bad_idx >= 0 else 0
        cumulative_samples = 0
        cumulative_bad = 0
        for index in indices:
            sample_count = int(rows[index][sample_idx] or 0) if sample_idx >= 0 else 0
            bad_count = int(rows[index][bad_idx] or 0) if bad_idx >= 0 else 0
            cumulative_samples += sample_count
            cumulative_bad += bad_count
            sample_rate = cumulative_samples / total_samples if total_samples else None
            bad_capture = cumulative_bad / total_bad if total_bad else None
            cumulative_lift = (
                bad_capture / sample_rate
                if bad_capture is not None and sample_rate
                else None
            )
            output[index] = rows[index] + [
                sample_rate,
                bad_capture,
                cumulative_lift,
                _reliability(bad_count, sample_count),
            ]
    return {"headers": headers + additions, "rows": output}


def _add_effect_chart(effect: dict[str, Any]) -> None:
    if not effect["rows"]:
        return
    workbook = effect["workbook"]
    ws = effect["sheet"]
    headers = effect["headers"]
    dataset = _index(headers, "dataset")
    auc = _index(headers, "auc")
    ks = _index(headers, "ks")
    if dataset < 0 or (auc < 0 and ks < 0):
        return
    chart = workbook.add_chart({"type": "column"})
    end_row = effect["data_row"] + len(effect["rows"]) - 1
    categories = [ws.name, effect["data_row"], dataset, end_row, dataset]
    if auc >= 0:
        chart.add_series(
            {
                "name": "AUC",
                "categories": categories,
                "values": [ws.name, effect["data_row"], auc, end_row, auc],
                "fill": {"color": COLORS["blue"]},
            }
        )
    if ks >= 0:
        chart.add_series(
            {
                "name": "KS",
                "categories": categories,
                "values": [ws.name, effect["data_row"], ks, end_row, ks],
                "fill": {"color": COLORS["orange"]},
            }
        )
    chart.set_title({"name": "Train / Test / OOT 效果对比"})
    chart.set_y_axis({"min": 0, "max": 1, "num_format": "0.00"})
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 620, "height": 320})
    ws.insert_chart(3, max(len(headers) + 1, 8), chart)


def _add_lift_chart(lift: dict[str, Any]) -> None:
    if not lift["rows"]:
        return
    workbook = lift["workbook"]
    ws = lift["sheet"]
    headers = lift["headers"]
    bucket = _index(headers, "bucket")
    dataset = _index(headers, "dataset")
    capture = _index(headers, "cumulative_bad_capture")
    if bucket < 0 or capture < 0:
        return
    # Chart the first dataset (normally Train) and keep the complete table for
    # Test/OOT auditability below.
    first_dataset = lift["rows"][0][dataset] if dataset >= 0 else None
    chart_rows = [
        (row_index, row)
        for row_index, row in enumerate(lift["rows"])
        if dataset < 0 or row[dataset] == first_dataset
    ]
    if not chart_rows:
        return
    chart = workbook.add_chart({"type": "line"})
    start = lift["data_row"] + chart_rows[0][0]
    end = lift["data_row"] + chart_rows[-1][0]
    chart.add_series(
        {
            "name": f"{first_dataset or '样本'} 累计坏样本捕获率",
            "categories": [ws.name, start, bucket, end, bucket],
            "values": [ws.name, start, capture, end, capture],
            "line": {"color": COLORS["blue"], "width": 2.25},
            "marker": {"type": "circle", "size": 5},
        }
    )
    chart.set_title({"name": "Lift 累计坏样本捕获曲线"})
    chart.set_y_axis({"min": 0, "max": 1, "num_format": "0%"})
    chart.set_x_axis({"name": "风险十分位（1=最高风险）"})
    chart.set_size({"width": 700, "height": 320})
    ws.insert_chart(3, max(len(headers) + 1, 9), chart)


def _add_binning_chart(binning: dict[str, Any]) -> None:
    if not binning["rows"]:
        return
    workbook = binning["workbook"]
    ws = binning["sheet"]
    headers = binning["headers"]
    feature = _index(headers, "feature")
    bin_label = _index(headers, "bin")
    bad_rate = _index(headers, "bad_rate")
    woe = _index(headers, "woe")
    iv_bin = _index(headers, "iv_bin")
    if feature < 0 or bin_label < 0 or bad_rate < 0:
        return
    feature_rows: dict[Any, list[int]] = {}
    for index, row in enumerate(binning["rows"]):
        feature_rows.setdefault(row[feature], []).append(index)
    selected_feature = max(
        feature_rows,
        key=(
            lambda name: sum(
                float(binning["rows"][row_index][iv_bin] or 0)
                for row_index in feature_rows[name]
            )
            if iv_bin >= 0
            else len(feature_rows[name])
        ),
    )
    chart_rows = feature_rows[selected_feature]
    start = binning["data_row"] + chart_rows[0]
    end = binning["data_row"] + chart_rows[-1]
    chart = workbook.add_chart({"type": "column"})
    chart.add_series(
        {
            "name": "坏样本率",
            "categories": [ws.name, start, bin_label, end, bin_label],
            "values": [ws.name, start, bad_rate, end, bad_rate],
            "fill": {"color": COLORS["orange"]},
        }
    )
    if woe >= 0:
        line = workbook.add_chart({"type": "line"})
        line.add_series(
            {
                "name": "WOE",
                "categories": [ws.name, start, bin_label, end, bin_label],
                "values": [ws.name, start, woe, end, woe],
                "line": {"color": COLORS["blue"], "width": 2},
                "marker": {"type": "circle", "size": 5},
                "y2_axis": True,
            }
        )
        chart.combine(line)
    chart.set_title({"name": f"分箱坏样本率 / WOE：{selected_feature}"})
    chart.set_y_axis({"min": 0, "num_format": "0%"})
    if woe >= 0:
        chart.set_y2_axis({"num_format": "0.00"})
    chart.set_size({"width": 760, "height": 360})
    ws.insert_chart(3, max(len(headers) + 1, 10), chart)


def _write_model_dashboard(
    workbook: xlsxwriter.Workbook, payload: dict[str, Any]
) -> Any:
    ws = workbook.add_worksheet("1.模型总览")
    _title(
        ws, workbook, "LightGBM 风控模型报告", "效果、稳定性和可靠性结论优先展示", 11
    )
    for col in range(12):
        ws.set_column(col, col, 12)
    metrics = payload["tables"].get("6.效果指标", {"headers": [], "rows": []})
    review = payload.get("model_review", {})
    records = [dict(zip(metrics["headers"], row)) for row in metrics["rows"]]
    test = next((record for record in records if record.get("dataset") == "test"), {})
    oot = next((record for record in records if record.get("dataset") == "oot"), {})
    train = next((record for record in records if record.get("dataset") == "train"), {})
    auc_gap = float(train.get("auc") or 0) - float(test.get("auc") or 0)
    review_status = review.get("status")
    status = {
        "blocked": "需重点优化",
        "needs_review": "需人工复核",
        "ready": "可继续验证",
    }.get(
        review_status,
        (
            "需重点优化"
            if auc_gap >= 0.1
            or _reliability(oot.get("bad_count"), oot.get("sample_count")) == "样本不足"
            else "可继续验证"
        ),
    )

    def card(
        row: int,
        col: int,
        label: str,
        value: Any,
        fmt: str | None = None,
        bad: bool = False,
    ) -> None:
        ws.merge_range(row, col, row, col + 2, label, workbook._formats["card_label"])
        value_fmt = workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": COLORS["red_text"] if bad else COLORS["navy"],
                "bg_color": COLORS["red"] if bad else COLORS["white"],
                "border": 1,
                "border_color": COLORS["line"],
                "align": "center",
                "valign": "vcenter",
                **({"num_format": fmt} if fmt else {}),
            }
        )
        ws.merge_range(row + 1, col, row + 2, col + 2, value, value_fmt)

    card(3, 0, "模型状态", status, bad=status != "可继续验证")
    card(
        3,
        3,
        "Test AUC",
        test.get("auc"),
        "0.0000",
        bad=float(test.get("auc") or 0) < 0.65,
    )
    card(
        3, 6, "Test KS", test.get("ks"), "0.0000", bad=float(test.get("ks") or 0) < 0.2
    )
    card(3, 9, "Train-Test AUC差距", auc_gap, "0.0000", bad=auc_gap >= 0.08)
    ws.merge_range(11, 0, 11, 11, "自动诊断与建议", workbook._formats["section"])
    ws.merge_range(
        12,
        0,
        12,
        11,
        f"Train AUC {float(train.get('auc') or 0):.4f}，Test AUC {float(test.get('auc') or 0):.4f}，AUC差距 {auc_gap:.4f}；OOT可靠性：{_reliability(oot.get('bad_count'), oot.get('sample_count'))}；诊断发现 {review.get('finding_count', 0)} 项（阻断 {review.get('blocker_count', 0)}、警告 {review.get('warning_count', 0)}），改进建议 {review.get('recommendation_count', 0)} 项。",
        workbook._formats["note"],
    )
    ws.merge_range(
        18, 0, 18, 5, "Train / Test / OOT 效果对比", workbook._formats["section"]
    )
    ws.merge_range(18, 6, 18, 11, "Top 10 Gain 重要性", workbook._formats["section"])
    return ws


def write_model_report(payload_path: str | Path, output_path: str | Path) -> Path:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output), {"nan_inf_to_errors": False})
    _init_formats(workbook)
    _write_model_dashboard(workbook, payload)
    tables = payload.get("tables", {})
    if "2.数据切分" in tables:
        _write_table(
            workbook,
            "2.样本切分",
            "Train / Test / OOT 样本切分",
            "可靠性同时考虑坏样本数和好样本数",
            _augment_model_table("2.数据切分", tables["2.数据切分"]),
        )
    if "14.特征预处理" in tables:
        _write_table(
            workbook,
            "3.特征预处理",
            "Train-only 特征类型与预处理",
            "类别字典和变量去留只在 Train 上拟合；Test/OOT 仅应用冻结规则",
            tables["14.特征预处理"],
        )
    if "3.特征筛选" in tables:
        selection = _write_table(
            workbook,
            "4.特征筛选",
            "特征筛选结果",
            "质量、IV、月度稳定性和相关性筛选只使用 Train",
            tables["3.特征筛选"],
        )
        _conditional_quality(selection)
    if "4.特征重要性" in tables:
        source = tables["4.特征重要性"]
        gain_idx = _index(source["headers"], "gain_importance")
        total = (
            sum(float(row[gain_idx] or 0) for row in source["rows"])
            if gain_idx >= 0
            else 0.0
        )
        cumulative = 0.0
        rows = []
        for row in source["rows"]:
            share = float(row[gain_idx] or 0) / total if total else 0.0
            cumulative += share
            rows.append(list(row) + [share, cumulative])
        _write_table(
            workbook,
            "4.特征重要性",
            "特征重要性",
            "Gain 占比说明模型依赖程度，不代表因果关系或风险方向",
            {
                "headers": source["headers"] + ["gain_share", "cumulative_gain_share"],
                "rows": rows,
            },
        )
    if "5.训练过程" in tables:
        training = _write_table(
            workbook,
            "9.训练明细",
            "完整训练过程明细",
            "最佳轮次已单独标记；本页用于追溯每一轮 Train/Test AUC",
            tables["5.训练过程"],
        )
        diagnostic = workbook.add_worksheet("5.训练诊断")
        _title(
            diagnostic,
            workbook,
            "模型训练与拟合诊断",
            "重点观察最佳轮次、验证集峰值以及 Train-Test 差距",
            13,
        )
        diagnostic.set_column(0, 13, 11)
        chart = workbook.add_chart({"type": "line"})
        it = _index(training["headers"], "iteration")
        ta = _index(training["headers"], "train_auc")
        va = _index(training["headers"], "test_auc")
        if it >= 0 and ta >= 0:
            chart.add_series(
                {
                    "name": "Train AUC",
                    "categories": [
                        training["sheet"].name,
                        training["data_row"],
                        it,
                        training["data_row"] + len(training["rows"]) - 1,
                        it,
                    ],
                    "values": [
                        training["sheet"].name,
                        training["data_row"],
                        ta,
                        training["data_row"] + len(training["rows"]) - 1,
                        ta,
                    ],
                    "line": {"color": COLORS["blue"]},
                }
            )
        if it >= 0 and va >= 0:
            chart.add_series(
                {
                    "name": "Test AUC",
                    "categories": [
                        training["sheet"].name,
                        training["data_row"],
                        it,
                        training["data_row"] + len(training["rows"]) - 1,
                        it,
                    ],
                    "values": [
                        training["sheet"].name,
                        training["data_row"],
                        va,
                        training["data_row"] + len(training["rows"]) - 1,
                        va,
                    ],
                    "line": {"color": COLORS["orange"]},
                }
            )
        chart.set_title({"name": "LightGBM 训练拟合曲线（AUC）"})
        chart.set_y_axis({"min": 0.5, "max": 1, "num_format": "0.00"})
        chart.set_size({"width": 800, "height": 320})
        diagnostic.insert_chart(9, 0, chart)
        ks = workbook.add_chart({"type": "line"})
        tks = _index(training["headers"], "train_ks")
        vks = _index(training["headers"], "test_ks")
        if it >= 0 and tks >= 0:
            ks.add_series(
                {
                    "name": "Train KS",
                    "categories": [
                        training["sheet"].name,
                        training["data_row"],
                        it,
                        training["data_row"] + len(training["rows"]) - 1,
                        it,
                    ],
                    "values": [
                        training["sheet"].name,
                        training["data_row"],
                        tks,
                        training["data_row"] + len(training["rows"]) - 1,
                        tks,
                    ],
                    "line": {"color": COLORS["blue"]},
                }
            )
        if it >= 0 and vks >= 0:
            ks.add_series(
                {
                    "name": "Test KS",
                    "categories": [
                        training["sheet"].name,
                        training["data_row"],
                        it,
                        training["data_row"] + len(training["rows"]) - 1,
                        it,
                    ],
                    "values": [
                        training["sheet"].name,
                        training["data_row"],
                        vks,
                        training["data_row"] + len(training["rows"]) - 1,
                        vks,
                    ],
                    "line": {"color": COLORS["orange"]},
                }
            )
        if tks >= 0 or vks >= 0:
            ks.set_title({"name": "LightGBM 训练拟合曲线（KS）"})
            ks.set_y_axis({"min": 0, "max": 1, "num_format": "0.00"})
            ks.set_size({"width": 800, "height": 320})
            diagnostic.insert_chart(9, 9, ks)
    if "5.分箱明细" in tables:
        binning = _write_table(
            workbook,
            "5.分箱分析",
            "特征分箱分析",
            "展示代表性特征的坏样本率和 WOE 走势；完整分箱明细同时保留在本页",
            tables["5.分箱明细"],
        )
        _add_binning_chart(binning)
    if "6.效果指标" in tables:
        effect = _write_table(
            workbook,
            "6.效果评估",
            "模型效果与分数稳定性",
            "同时比较效果差距和样本可靠性，不单看单一指标",
            _augment_model_table("6.效果指标", tables["6.效果指标"]),
        )
        _add_effect_chart(effect)
    if "7.Lift明细" in tables:
        lift = _write_table(
            workbook,
            "7.Lift分析",
            "风险排序与坏样本捕获",
            "十分位 1 为最高风险；重点观察累计坏样本捕获率",
            _augment_lift_table(tables["7.Lift明细"]),
        )
        _add_lift_chart(lift)
    if "8.月度表现" in tables:
        _write_table(
            workbook,
            "8.月度表现",
            "月度模型表现",
            "样本不足月份仅作方向性参考",
            _augment_model_table("8.月度表现", tables["8.月度表现"]),
        )
    for key, sheet_name, title in [
        ("10.候选模型", "10.候选模型", "Baseline 与调参候选模型"),
        ("11.调参试验", "11.调参试验", "Optuna Trial 调参明细"),
        ("12.CV折表现", "12.CV折表现", "Train 内部交叉验证明细"),
        ("13.AI模型审查", "13.AI模型审查", "AI 模型红队审查"),
        ("13.诊断建议", "13.诊断建议", "模型诊断改进建议"),
    ]:
        if key in tables:
            _write_table(
                workbook, sheet_name, title, "完整结果明细与审计证据", tables[key]
            )
    methodology = {
        "headers": [
            "检查项",
            "正常范围",
            "正常解释",
            "关注范围",
            "关注解释",
            "重点风险",
        ],
        "rows": [
            [
                "Train-Test AUC差距",
                "< 0.08",
                "一般较稳定",
                "0.08～0.15",
                "需要关注",
                "≥ 0.15 重点检查过拟合",
            ],
            [
                "Train-Test KS差距",
                "< 0.08",
                "一般较稳定",
                "0.08～0.15",
                "需要关注",
                "≥ 0.15 重点检查泛化能力",
            ],
            [
                "Test→OOT指标下降",
                "< 0.05",
                "时间外表现稳定",
                "0.05～0.10",
                "需要关注",
                "≥ 0.10 重点检查时间窗口和样本漂移",
            ],
            ["分数PSI", "< 0.10", "稳定", "0.10～0.25", "需要关注", "≥ 0.25 明显漂移"],
            [
                "月度AUC/KS波动",
                "差值 < 0.10",
                "月度表现稳定",
                "差值 ≥ 0.10",
                "需要核对波动月份",
                "结合样本量和坏样本数判断可靠性",
            ],
            [
                "Lift排序",
                "相邻分箱基本单调",
                "风险排序有效",
                "逆序分箱超过30%",
                "排序性需要复核",
                "检查标签方向、分箱和特征质量",
            ],
            [
                "指标可靠性",
                "坏/好样本均≥30",
                "可评价",
                "任一为10～29",
                "仅方向性",
                "任一<10 样本不足",
            ],
            [
                "OOT使用原则",
                "独立时间窗口",
                "只做最终验证",
                "不得用于调参",
                "不得用于早停",
                "不得用于特征筛选",
            ],
        ],
    }
    _write_table(
        workbook,
        "14.报告口径",
        "报告指标与可靠性口径",
        "阈值用于自动提示，不替代业务、合规和独立模型验证",
        methodology,
    )
    workbook.close()
    logger.info("Model xlsx report created: %s", output)
    return output

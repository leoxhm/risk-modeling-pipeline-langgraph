"""Complete structured-data EDA implementation for the LangGraph Skill.

The module is intentionally independent from the previous pipeline. Large
sources are scanned lazily and each feature is collected separately, so EDA
does not need to hold every column in memory at once.
"""

from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import polars as pl
import toad

from core.data_read.contract import DataContract, feature_columns
from core.data_read.loader import scan_table
from core.sample_diagnosis.diagnostics import build_sample_diagnosis


EPS = 0.5

_HEADER_LABELS = {
    "name": "诊断项",
    "value": "当前结果",
    "threshold": "判断标准",
    "severity": "风险级别",
    "recommendation": "处理建议",
    "feature": "特征",
    "feature_1": "特征1",
    "feature_2": "特征2",
    "dtype": "类型",
    "missing_rate": "缺失率",
    "mean": "均值",
    "std": "STD",
    "iv": "信息价值（IV）",
    "ks": "KS",
    "auc": "区分度（AUC）",
    "average_psi": "平均 PSI",
    "max_psi": "最大 PSI",
    "psi": "PSI",
    "psi_warning": "PSI 预警",
    "bin_count": "分箱数",
    "bin_order": "分箱序号",
    "bin": "分箱区间",
    "sample_count": "样本数",
    "sample_rate": "样本占比",
    "bad_count": "坏样本数",
    "good_count": "好样本数",
    "bad_rate": "坏账率",
    "bad_distribution": "坏样本占比",
    "good_distribution": "好样本占比",
    "woe": "WOE",
    "iv_bin": "分箱 IV",
    "cumulative_iv": "累计 IV",
    "cumulative_ks": "累计 KS",
    "correlation": "相关系数",
    "abs_correlation": "绝对相关系数",
    "event_month": "月份",
    "baseline_month": "基准月份",
    "lift_10": "Top10% Lift",
}

_RATE_COLUMNS = {"missing_rate", "sample_rate", "bad_rate", "bad_distribution", "good_distribution"}


def _display_name(name: str) -> str:
    """将内部英文字段名转换为报告使用的中文表头。"""
    return _HEADER_LABELS.get(name, name)


def _display_value(name: str, value: Any) -> str:
    """按字段类型格式化报告数值，指标统一保留五位小数。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return "—"
        if name in _RATE_COLUMNS:
            return f"{value:.5%}"
        return f"{value:.5f}"
    return str(value)


def _round_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """将结果表中的浮点列统一四舍五入到五位小数。"""
    expressions = [pl.col(name).round(5).alias(name) for name, dtype in frame.schema.items() if dtype.is_float()]
    return frame.with_columns(expressions) if expressions else frame


def _collect(lazy: pl.LazyFrame) -> pl.DataFrame:
    """执行 LazyFrame 收集；优先使用 Polars 流式引擎以降低峰值内存。"""
    try:
        return lazy.collect(engine="streaming")
    except (TypeError, RuntimeError):
        return lazy.collect()


def _is_numeric(dtype: pl.DataType) -> bool:
    """判断 Polars 类型是否为非布尔数值类型。"""
    return dtype.is_numeric() and dtype != pl.Boolean


def _auc(values: list[float], target: list[int]) -> float | None:
    """使用 toad 计算单变量 AUC；样本不足或只有单一标签时返回空值。"""
    if not values or len(set(target)) < 2:
        return None
    try:
        # toad 返回原始方向 AUC；风控单变量画像通常报告方向无关的 AUC。
        raw = float(toad.metrics.AUC(np.asarray(values, dtype=float), np.asarray(target, dtype=int)))
        return max(raw, 1.0 - raw)
    except (TypeError, ValueError, IndexError):
        return None


def _ks(values: list[float], target: list[int]) -> float | None:
    """使用 toad 计算单变量 KS；仅保留非空值并检查正负样本是否齐全。"""
    if not values or len(set(target)) < 2:
        return None
    try:
        return float(toad.KS(np.asarray(values, dtype=float), np.asarray(target, dtype=int)))
    except (TypeError, ValueError, IndexError):
        return None


def _lift10(values: list[float], target: list[int]) -> float | None:
    """计算按分数降序前 10% 样本的 lift。"""
    if not values or not target or not sum(target):
        return None
    size = max(1, math.ceil(len(values) * 0.1))
    top = sorted(zip(values, target), key=lambda pair: pair[0], reverse=True)[:size]
    return (sum(label for _, label in top) / size) / (sum(target) / len(target))


def _edges(values: list[float], count: int) -> list[float]:
    """根据数值分位点生成稳定且不重复的分箱切点。"""
    clean = sorted(set(value for value in values if math.isfinite(value)))
    if len(clean) <= 1:
        return []
    result = []
    for index in range(1, count):
        position = round(index * (len(clean) - 1) / count)
        value = clean[position]
        if not result or value > result[-1]:
            result.append(value)
    return result


def _make_bins(values: list[Any], dtype: pl.DataType, *, count: int, max_categories: int, target: list[int] | None = None) -> tuple[list[int], list[str]]:
    """为数值或类别变量生成分箱编号及展示标签。"""
    if _is_numeric(dtype):
        numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        cuts = _edges(numeric, count)
        # 优先采用 toad Combiner 的卡方分箱；无法处理异常列时退回分位点。
        if target is not None and numeric and len(target) == len(values) and len(set(target)) > 1:
            try:
                from toad.transform import Combiner

                combiner = Combiner()
                combiner.fit(
                    pd.DataFrame({"feature": values}),
                    pd.Series(target),
                    method="chi",
                    n_bins=count,
                    min_samples=0.01,
                    empty_separate=True,
                )
                candidate = combiner.rules.get("feature")
                if candidate is not None:
                    chi_cuts = [float(value) for value in candidate if value is not None and math.isfinite(float(value))]
                    if chi_cuts:
                        cuts = sorted(set(chi_cuts))
            except (ImportError, TypeError, ValueError, KeyError):
                pass
        labels = []
        for index, cut in enumerate(cuts):
            left = "-inf" if index == 0 else f"{cuts[index - 1]:.6g}"
            labels.append(f"({left}, {cut:.6g}]")
        labels.append(f"({cuts[-1]:.6g}, inf)" if cuts else "all")
        return [(-1 if value is None else bisect_right(cuts, float(value))) for value in values], labels
    counts = Counter(str(value) for value in values if value is not None)
    categories = [name for name, _ in counts.most_common(max(1, max_categories - 1))]
    labels = categories + (["OTHER"] if len(counts) > len(categories) else [])
    mapping = {name: index for index, name in enumerate(categories)}
    other = len(categories)
    return [(-1 if value is None else mapping.get(str(value), other)) for value in values], labels


def _bin_detail(feature: str, values: list[Any], target: list[int], dtype: pl.DataType, *, count: int, max_categories: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成分箱明细，并以 toad 计算该变量的 IV、KS 和 AUC。"""
    bins, labels = _make_bins(values, dtype, count=count, max_categories=max_categories, target=target)
    groups: dict[int, dict[str, int]] = defaultdict(lambda: {"sample_count": 0, "bad_count": 0})
    for bin_id, label in zip(bins, target):
        groups[bin_id]["sample_count"] += 1
        groups[bin_id]["bad_count"] += int(label)
    ordered = sorted(item for item in groups if item >= 0) + ([-1] if -1 in groups else [])
    total = len(target)
    total_bad = sum(target)
    total_good = total - total_bad
    rows: list[dict[str, Any]] = []
    cumulative_iv = 0.0
    cumulative_ks = 0.0
    for order, bin_id in enumerate(ordered, start=1):
        sample_count = groups[bin_id]["sample_count"]
        bad_count = groups[bin_id]["bad_count"]
        good_count = sample_count - bad_count
        bad_dist = (bad_count + EPS) / (total_bad + EPS * len(ordered)) if total_bad else 0.0
        good_dist = (good_count + EPS) / (total_good + EPS * len(ordered)) if total_good else 0.0
        woe = math.log(bad_dist / good_dist) if bad_dist and good_dist else 0.0
        iv_bin = (bad_dist - good_dist) * woe
        cumulative_iv += iv_bin
        if bin_id >= 0 and total_bad and total_good:
            prev_bad = sum(groups[item]["bad_count"] for item in ordered if 0 <= item <= bin_id)
            prev_good = sum(groups[item]["sample_count"] - groups[item]["bad_count"] for item in ordered if 0 <= item <= bin_id)
            cumulative_ks = max(cumulative_ks, abs(prev_bad / total_bad - prev_good / total_good))
        rows.append({"feature": feature, "bin_order": order, "bin": "MISSING" if bin_id == -1 else labels[bin_id], "sample_count": sample_count, "sample_rate": sample_count / total if total else 0.0, "bad_count": bad_count, "good_count": good_count, "bad_rate": bad_count / sample_count if sample_count else 0.0, "bad_distribution": bad_count / total_bad if total_bad else 0.0, "good_distribution": good_count / total_good if total_good else 0.0, "woe": woe, "iv_bin": iv_bin, "cumulative_iv": cumulative_iv, "ks": None if bin_id == -1 else cumulative_ks, "cumulative_ks": cumulative_ks})
    non_missing = [(float(value), label) for value, label in zip(values, target) if value is not None and _is_numeric(dtype)]
    numeric_values = [value for value, _ in non_missing]
    numeric_target = [label for _, label in non_missing]
    # 使用 toad.stats.IV 对当前卡方/类别分箱直接计算 IV，保证总览和明细同口径。
    clean_values = [value for value in values if value is not None]
    clean_target = [label for value, label in zip(values, target) if value is not None]
    try:
        iv, sub = toad.stats.IV(np.asarray(bins, dtype=int), np.asarray(target, dtype=int), return_sub=True) if clean_values and len(set(clean_target)) > 1 else (None, {})
        iv = float(iv) if iv is not None else None
        sub_map = {int(key): float(value) for key, value in sub.items()}
        for row, bin_id in zip(rows, ordered):
            row["iv_bin"] = sub_map.get(bin_id, row["iv_bin"])
    except (TypeError, ValueError, IndexError):
        iv = None
    ks = _ks(numeric_values, numeric_target) if numeric_values else None
    auc = _auc(numeric_values, numeric_target) if numeric_values else None
    return rows, {"feature": feature, "dtype": str(dtype), "missing_rate": sum(value is None for value in values) / total if total else 0.0, "iv": iv, "ks": ks, "auc": auc, "bin_count": len(ordered), "metrics_backend": "toad"}


def _month(value: Any) -> str | None:
    """将常见日期表达式规整为 YYYYMM 月份键。"""
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) >= 6:
        return text[:6]
    if len(text) >= 7 and text[4] in "-/":
        return text[:7].replace("-", "")
    return None


def _psi(base: list[int], current: list[int]) -> float:
    """使用 ``toad.metrics.PSI`` 按分箱计数计算总体 PSI。"""
    if not base or not current:
        return 0.0
    base_values = np.repeat(np.arange(len(base)), np.asarray(base, dtype=int))
    current_values = np.repeat(np.arange(len(current)), np.asarray(current, dtype=int))
    if not len(base_values) or not len(current_values):
        return 0.0
    try:
        return float(toad.metrics.PSI(current_values, base_values))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _markdown_table(frame: pl.DataFrame, limit: int | None = None) -> str:
    """将 Polars DataFrame 渲染为 Markdown 表格。"""
    view = frame.head(limit) if limit else frame
    lines = ["| " + " | ".join(_display_name(name) for name in view.columns) + " |", "|" + "|".join("---" for _ in view.columns) + "|"]
    lines.extend("| " + " | ".join(_display_value(name, row.get(name)) for name in view.columns) + " |" for row in view.to_dicts())
    return "\n".join(lines)


def _html_table(frame: pl.DataFrame, limit: int = 20, *, sortable: bool = False) -> str:
    """将 Polars DataFrame 渲染为 HTML 表格，可选启用点击表头排序。"""
    view = frame.head(limit)
    if not view.height:
        return "<p>暂无数据</p>"
    header_cells = []
    for name in view.columns:
        label = html.escape(_display_name(name))
        content = f"<button class='sort-btn' onclick='sortTable(this)'>{label} ↕</button>" if sortable else label
        header_cells.append(f"<th>{content}</th>")
    head = "".join(header_cells)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_display_value(name, row.get(name)))}</td>" for name in view.columns) + "</tr>" for row in view.to_dicts())
    return f"<table class='{'sortable' if sortable else ''}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _html_binning_sections(frame: pl.DataFrame) -> str:
    """按特征生成可折叠的卡方分箱明细，避免一次展开超长表格。"""
    if not frame.height or "feature" not in frame.columns:
        return "<p>暂无分箱明细</p>"
    features = list(dict.fromkeys(str(value) for value in frame.get_column("feature").to_list()))
    sections: list[str] = []
    for feature in features:
        detail = frame.filter(pl.col("feature") == feature)
        sections.append(
            "<details class='binning-feature'><summary>"
            + html.escape(feature)
            + f"（{detail.height} 个分箱）</summary>"
            + _html_table(detail, limit=max(detail.height, 1))
            + "</details>"
        )
    return "".join(sections)


def _bar_chart(frame: pl.DataFrame, value: str, title: str) -> str:
    """生成无需外部 JavaScript 的 Top10 横向条形图 HTML。"""
    rows = frame.head(10).to_dicts()
    if not rows:
        return "<p>暂无图表数据</p>"
    maximum = max(float(row.get(value) or 0) for row in rows) or 1.0
    bars = "".join(f"<div class='bar-row'><span>{html.escape(str(row.get('feature', '')))}</span><div class='bar' style='width:{float(row.get(value) or 0) / maximum * 100:.1f}%'></div><b>{_display_value(value, row.get(value))}</b></div>" for row in rows)
    return f"<h3>{html.escape(title)}</h3>{bars}"


def _correlation_heatmap(frame: pl.DataFrame) -> str:
    """把相关系数矩阵渲染为带正负颜色和数值的 HTML 热力图。"""
    if not frame.height or not frame.columns:
        return "<p>暂无数值变量相关性数据</p>"
    row_key = "feature" if "feature" in frame.columns else frame.columns[0]
    value_columns = [name for name in frame.columns if name != row_key]
    rows = [f"<th class='sticky'>{html.escape(name)}</th>" for name in value_columns]
    header = "<tr><th class='sticky'>变量</th>" + "".join(rows) + "</tr>"
    body_parts: list[str] = []
    for row in frame.to_dicts():
        name = str(row.get(row_key, ""))
        cells = []
        for column in value_columns:
            raw = row.get(column)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            value = max(-1.0, min(1.0, value))
            if value >= 0:
                color = f"rgba(37, 99, 235, {abs(value) * 0.72:.3f})"
            else:
                color = f"rgba(220, 38, 38, {abs(value) * 0.72:.3f})"
            cells.append(f"<td style='background:{color}' title='{value:.5f}'>{value:.5f}</td>")
        body_parts.append(f"<tr><th class='sticky row-label'>{html.escape(name)}</th>{''.join(cells)}</tr>")
    return "<div class='heatmap-wrap'><table class='heatmap'><thead>" + header + "</thead><tbody>" + "".join(body_parts) + "</tbody></table></div><p class='legend'><span class='legend-blue'>正相关</span><span class='legend-red'>负相关</span>；颜色越深表示绝对相关性越高</p>"


def _distribution_grid(bins: pl.DataFrame, ranking: pl.DataFrame, *, limit: int = 8) -> str:
    """按 IV 排名前若干变量生成分箱样本分布的横向条形图网格。"""
    if not bins.height or not ranking.height:
        return "<p>暂无变量分布数据</p>"
    selected = ranking.get_column("feature").head(limit).to_list()
    blocks: list[str] = []
    for feature in selected:
        detail = bins.filter(pl.col("feature") == feature).sort("bin_order")
        if not detail.height:
            continue
        maximum = max(float(value or 0) for value in detail.get_column("sample_count").to_list()) or 1.0
        bars = []
        for row in detail.to_dicts():
            label = str(row.get("bin", ""))
            width = float(row.get("sample_count") or 0) / maximum * 100
            bars.append(f"<div class='dist-row'><span>{html.escape(label[:18])}</span><div class='dist-bar' style='width:{width:.1f}%'></div><b>{int(row.get('sample_count') or 0):,}</b></div>")
        blocks.append(f"<div class='dist-card'><h3>{html.escape(str(feature))}</h3>{''.join(bars)}</div>")
    return "<div class='dist-grid'>" + "".join(blocks) + "</div>"


def _monthly_sample_chart(frame: pl.DataFrame) -> str:
    """生成月度样本量和坏账率的双轴风格 CSS 图示。"""
    if not frame.height:
        return "<p>暂无月份字段或月份数据</p>"
    rows = frame.to_dicts()
    max_count = max(float(row.get("sample_count") or 0) for row in rows) or 1.0
    items = []
    for row in rows:
        month = html.escape(str(row.get("event_month", "")))
        count = int(row.get("sample_count") or 0)
        rate = float(row.get("bad_rate") or 0.0)
        height = count / max_count * 150
        rate_width = min(100.0, rate * 1000.0)
        items.append(
            f"<div class='month-col'><div class='month-value'>{count:,}</div><div class='month-bar' style='height:{height:.1f}px'></div><div class='month-rate' style='width:{rate_width:.1f}%'></div><div class='month-label'>{month}</div><small>坏率 {rate:.5%}</small></div>"
        )
    return "<div class='chart-legend'><span class='sample-legend'>■ 样本量</span><span class='rate-legend'>━ 坏账率（放大显示）</span></div><div class='month-chart'>" + "".join(items) + "</div>"


def run_eda(*, data_path: str | Path, output_dir: str | Path, contract: DataContract, data_frame: pl.DataFrame | None = None, bin_count: int = 10, max_categories: int = 20, export_details: bool = False) -> dict[str, Any]:
    """执行完整 EDA；默认只落盘报告和摘要，详细 CSV 需显式开启。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lazy = data_frame.lazy() if data_frame is not None else scan_table(data_path)
    schema = lazy.collect_schema()
    row_count = int(_collect(lazy.select(pl.len())).item())
    features = feature_columns(_collect(lazy.head(1)), contract)
    overview_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    per_feature_month: list[dict[str, Any]] = []
    variable_psi_rows: list[dict[str, Any]] = []
    date_target = None
    if contract.date_col and contract.target_col and contract.date_col in schema and contract.target_col in schema:
        date_target = _collect(lazy.select([contract.date_col, contract.target_col]))
    duplicate_groups = 0
    missing_labels = 0
    if contract.target_col and contract.target_col in schema:
        missing_labels = int(_collect(lazy.select(pl.col(contract.target_col).is_null().sum())).item() or 0)
    valid_id_cols = [name for name in contract.id_cols if name in schema]
    if valid_id_cols:
        id_frame = _collect(lazy.select(valid_id_cols))
        duplicate_groups = int(id_frame.group_by(valid_id_cols).len().filter(pl.col("len") > 1).height)
    monthly_rows: list[dict[str, Any]] = []
    if date_target is not None:
        months = [_month(value) for value in date_target.get_column(contract.date_col).to_list()]
        labels = [int(value == contract.bad_label) for value in date_target.get_column(contract.target_col).to_list()]
        grouped: dict[str, list[int]] = defaultdict(list)
        for month, label in zip(months, labels):
            if month:
                grouped[month].append(label)
        ordered_months = sorted(grouped)
        baseline_counts = [len(grouped[month]) for month in ordered_months]
        for month in ordered_months:
            current = grouped[month]
            monthly_rows.append({"event_month": month, "sample_count": len(current), "bad_count": sum(current), "good_count": len(current) - sum(current), "bad_rate": sum(current) / len(current) if current else 0.0, "sample_rate": len(current) / row_count if row_count else 0.0, "psi": _psi(baseline_counts, [len(grouped[item]) if item == month else 0 for item in ordered_months]) if ordered_months else 0.0})
    for feature in features:
        cols = [feature]
        if contract.target_col and contract.target_col in schema:
            cols.append(contract.target_col)
        if contract.date_col and contract.date_col in schema:
            cols.append(contract.date_col)
        frame = _collect(lazy.select(cols))
        values = frame.get_column(feature).to_list()
        target = [int(value == contract.bad_label) for value in frame.get_column(contract.target_col).to_list()] if contract.target_col and contract.target_col in frame.columns else [0] * len(values)
        detail, metrics = _bin_detail(feature, values, target, schema[feature], count=bin_count, max_categories=max_categories)
        bin_rows.extend(detail)
        numeric_stats = {}
        if _is_numeric(schema[feature]):
            numeric_stats = _collect(lazy.select([pl.col(feature).mean().alias("mean"), pl.col(feature).std().alias("std")])).row(0, named=True)
        overview_rows.append({**metrics, "mean": numeric_stats.get("mean"), "std": numeric_stats.get("std")})
        if contract.date_col and contract.date_col in frame.columns:
            months = [_month(value) for value in frame.get_column(contract.date_col).to_list()]
            by_month: dict[str, tuple[list[Any], list[int]]] = {}
            for month, value, label in zip(months, values, target):
                if month:
                    current_values, current_labels = by_month.setdefault(month, ([], []))
                    current_values.append(value)
                    current_labels.append(label)
            bin_ids, _ = _make_bins(values, schema[feature], count=bin_count, max_categories=max_categories, target=target)
            bins_by_month: dict[str, list[int]] = defaultdict(list)
            for month, bin_id in zip(months, bin_ids):
                if month:
                    bins_by_month[month].append(bin_id)
            ordered_months = sorted(bins_by_month)
            baseline_month = ordered_months[0] if ordered_months else None
            if baseline_month:
                bin_count_total = max(bin_ids, default=-1) + 2
                base_counts = [bins_by_month[baseline_month].count(index - 1) for index in range(bin_count_total)]
                for psi_month in ordered_months:
                    current_counts = [bins_by_month[psi_month].count(index - 1) for index in range(bin_count_total)]
                    variable_psi_rows.append({"feature": feature, "event_month": psi_month, "baseline_month": baseline_month, "psi": _psi(base_counts, current_counts)})
            for month, (month_values, month_labels) in sorted(by_month.items()):
                valid = [(float(value), label) for value, label in zip(month_values, month_labels) if value is not None and _is_numeric(schema[feature])]
                scores = [value for value, _ in valid]
                labels = [label for _, label in valid]
                per_feature_month.append({"feature": feature, "event_month": month, "sample_count": len(month_values), "bad_count": sum(month_labels), "good_count": len(month_labels) - sum(month_labels), "bad_rate": sum(month_labels) / len(month_labels) if month_labels else 0.0, "sample_rate": len(month_values) / row_count if row_count else 0.0, "ks": _ks(scores, labels), "auc": _auc(scores, labels), "lift_10": _lift10(scores, labels)})

    overview = pl.DataFrame(overview_rows) if overview_rows else pl.DataFrame()
    bins = pl.DataFrame(bin_rows) if bin_rows else pl.DataFrame()
    monthly = pl.DataFrame(monthly_rows) if monthly_rows else pl.DataFrame()
    monthly_discrimination = pl.DataFrame(per_feature_month) if per_feature_month else pl.DataFrame()
    monthly_psi = pl.DataFrame(variable_psi_rows) if variable_psi_rows else pl.DataFrame()
    if monthly_discrimination.height and monthly_psi.height:
        monthly_discrimination = monthly_discrimination.join(monthly_psi, on=["feature", "event_month"], how="left")
    if overview.height:
        overview = overview.with_columns(pl.col("missing_rate").alias("missing_rate"))

    psi_rows: list[dict[str, Any]] = []
    for feature in features:
        values_for_feature = [row["psi"] for row in variable_psi_rows if row["feature"] == feature]
        psi_rows.append({"feature": feature, "average_psi": sum(values_for_feature) / len(values_for_feature) if values_for_feature else 0.0, "max_psi": max(values_for_feature, default=0.0), "psi_warning": max(values_for_feature, default=0.0) >= 0.25})
    psi_summary = pl.DataFrame(psi_rows) if psi_rows else pl.DataFrame()
    correlation_matrix = pl.DataFrame()
    numeric_features = [name for name in features if _is_numeric(schema[name])]
    if numeric_features:
        # 相关性按成对完整样本计算；先用各列中位数填补缺失，避免 Polars
        # 在含 null 的列上返回整列 NaN。
        correlation_frame = _collect(
            lazy.select([pl.col(name).fill_null(pl.col(name).median()).alias(name) for name in numeric_features])
        )
        raw_correlation = correlation_frame.corr()
        correlation_matrix = raw_correlation.with_columns(pl.Series("feature", numeric_features)).select(["feature", *numeric_features])
    pair_rows = []
    if correlation_matrix.height:
        for left_index, left in enumerate(numeric_features):
            for right in numeric_features[left_index + 1:]:
                value = correlation_matrix.get_column(right)[left_index]
                if value is not None:
                    pair_rows.append({"feature_1": left, "feature_2": right, "correlation": float(value), "abs_correlation": abs(float(value))})
    correlation_pairs = pl.DataFrame(pair_rows).sort("abs_correlation", descending=True) if pair_rows else pl.DataFrame()

    diagnosis = build_sample_diagnosis(
        overview=overview,
        monthly=monthly,
        psi_summary=psi_summary,
        correlation_pairs=correlation_pairs,
        monthly_discrimination=monthly_discrimination,
        duplicate_groups=duplicate_groups,
        missing_labels=missing_labels,
        total_rows=row_count,
    )

    outputs = {
        "univariate_overview": overview,
        "binning_detail": bins,
        "monthly_sample": monthly,
        "monthly_psi": monthly_psi,
        "psi_summary": psi_summary,
        "monthly_discrimination": monthly_discrimination,
        "ks_bucket": bins.select([column for column in ["feature", "bin_order", "bin", "sample_count", "bad_count", "bad_rate", "cumulative_ks"] if column in bins.columns]) if bins.height else pl.DataFrame(),
        "correlation_pairs": correlation_pairs,
        "correlation_matrix": correlation_matrix,
    }
    outputs["variable_distributions"] = bins.select([column for column in ["feature", "bin_order", "bin", "sample_count", "sample_rate", "bad_count", "bad_rate"] if column in bins.columns]) if bins.height else pl.DataFrame()
    detail_artifacts: list[str] = []
    if export_details:
        for name, frame in outputs.items():
            # CSV 保持机器可读的英文列名，同时将所有浮点指标统一保留五位小数。
            path = output / f"{name}.csv"
            _round_frame(frame).write_csv(path, float_precision=5)
            detail_artifacts.append(str(path))
    # 对话返回精简的建模摘要；完整诊断明细仍写入 eda_conclusion.md 和 HTML。
    summary = diagnosis.get("brief_summary_markdown") or f"EDA 完成：分析 {row_count:,} 行、{len(features):,} 个特征。"
    conclusion = output / "eda_conclusion.md"
    top_iv = overview.sort("iv", descending=True).head(10) if overview.height else overview
    conclusion.write_text("# EDA 结果概述\n\n" + summary + "\n\n## 完整诊断明细\n\n" + diagnosis["summary_markdown"] + "\n", encoding="utf-8")
    (output / "eda_summary.json").write_text(json.dumps({"node_id": "eda-analysis", "status": "success", "row_count": row_count, "feature_count": len(features), "metrics_backend": "toad", "summary": summary, "diagnosis": diagnosis, "conclusion": str(conclusion)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = output / "data_eda_report.html"
    top_ks = overview.sort("ks", descending=True).head(10) if overview.height and "ks" in overview.columns else overview
    top_psi = psi_summary.sort("max_psi", descending=True).head(10) if psi_summary.height else psi_summary
    top_iv_table = top_iv.select([column for column in ["feature", "iv", "ks", "auc", "missing_rate"] if column in top_iv.columns]) if top_iv.height else top_iv
    top_ks_table = top_ks.select([column for column in ["feature", "ks", "iv", "auc", "missing_rate"] if column in top_ks.columns]) if top_ks.height else top_ks
    iv_max = next((float(value) for value in (top_iv.get_column("iv").to_list() if top_iv.height and "iv" in top_iv.columns else []) if value is not None), 0.0)
    feature_table = overview
    if overview.height and psi_summary.height:
        feature_table = overview.join(psi_summary.select(["feature", "average_psi", "max_psi"]), on="feature", how="left")
    feature_columns_for_report = [column for column in ["feature", "missing_rate", "mean", "std", "iv", "ks", "auc", "average_psi", "max_psi", "bin_count"] if column in feature_table.columns]
    feature_table = feature_table.select(feature_columns_for_report) if feature_columns_for_report else feature_table
    cards = (
        f"<div class='cards'><div class='card'><small>样本量</small><strong>{row_count:,}</strong></div>"
        f"<div class='card'><small>特征数</small><strong>{len(features):,}</strong></div>"
        f"<div class='card'><small>IV最高</small><strong>{iv_max:.5f}</strong></div>"
        f"<div class='card'><small>PSI预警特征</small><strong>{sum(1 for row in psi_rows if row['psi_warning'])}</strong></div></div>"
        if overview.height and "iv" in overview.columns else ""
    )
    diagnosis_frame = pl.DataFrame(diagnosis["findings"]) if diagnosis["findings"] else pl.DataFrame()
    report_html = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>风控数据 EDA 报告</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#172033;background:#f5f7fb;max-width:1400px;margin:0 auto;padding:28px}h1{margin:0 0 8px;font-size:28px}h2{margin-top:30px;font-size:19px;border-left:4px solid #2563eb;padding-left:10px}.subtitle{color:#5c6780;margin:0 0 18px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#fff;border:1px solid #e6eaf2;border-radius:12px;padding:16px;box-shadow:0 2px 8px #1d3b6a0d}.card small{display:block;color:#6b7280}.card strong{display:block;font-size:25px;margin-top:6px;color:#1d4ed8}.panel{background:#fff;border:1px solid #e6eaf2;border-radius:12px;padding:16px;margin-top:12px;overflow:auto}.bar-row{display:flex;align-items:center;gap:10px;margin:8px 0}.bar-row span{width:105px;font-size:12px;white-space:nowrap}.bar{height:15px;background:linear-gradient(90deg,#2563eb,#60a5fa);border-radius:8px}.bar-row b{font-size:12px;color:#374151}table{border-collapse:collapse;width:100%;margin:4px 0}th,td{border-bottom:1px solid #edf0f5;padding:8px;font-size:12px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f8fafc;color:#475569;font-weight:600}.sort-btn{border:0;background:transparent;color:#475569;font:inherit;cursor:pointer}.sort-btn:hover{color:#2563eb}.month-chart{display:flex;align-items:flex-end;gap:9px;min-height:220px;padding:20px 8px 4px;border-bottom:1px solid #dbe3ef;overflow-x:auto}.month-col{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;min-width:58px;height:205px}.month-value{font-size:10px;color:#475569;margin-bottom:3px}.month-bar{width:30px;background:linear-gradient(#60a5fa,#2563eb);border-radius:5px 5px 0 0;min-height:2px}.month-rate{height:3px;background:#ef4444;border-radius:3px;margin-top:-2px}.month-label{font-size:10px;color:#475569;margin-top:7px}.month-col small{font-size:10px;color:#b91c1c}.chart-legend{font-size:12px;color:#64748b}.sample-legend{color:#2563eb;margin-right:12px}.rate-legend{color:#dc2626}.heatmap-wrap{overflow:auto}.heatmap{width:auto;min-width:650px}.heatmap th,.heatmap td{text-align:center;border:1px solid #fff;padding:6px;min-width:44px}.heatmap .sticky{position:sticky;left:0;background:#f8fafc;z-index:1}.heatmap .row-label{text-align:left}.dist-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.dist-card{border:1px solid #edf0f5;border-radius:10px;padding:12px}.dist-card h3{margin:0 0 8px;font-size:14px}.dist-row{display:flex;align-items:center;gap:8px;margin:5px 0}.dist-row span{width:100px;font-size:11px;overflow:hidden;text-overflow:ellipsis}.dist-bar{height:11px;background:#93c5fd;border-radius:6px}.dist-row b{font-size:11px;color:#475569}.legend{color:#64748b;font-size:12px}.legend-blue{color:#2563eb;margin-right:12px}.legend-red{color:#dc2626;margin-right:12px}.note{font-size:12px;color:#667085}</style></head><body><h1>风控数据 EDA 报告</h1><p class='subtitle'>""" + html.escape(summary) + " · 指标计算后端：toad</p>" + cards + "<h2>样本诊断</h2><div class='panel'>" + _html_table(diagnosis_frame, limit=100) + "</div><h2>特征指标总览</h2><div class='panel'>" + _html_table(feature_table, limit=500, sortable=True) + "</div><h2>区分度概览</h2><div class='panel'>" + _bar_chart(top_iv, "iv", "IV Top10") + _html_table(top_iv) + _bar_chart(top_ks, "ks", "KS Top10") + _html_table(top_ks) + "</div><h2>稳定性与时间分布</h2><div class='panel'><h3>月度样本与坏账率</h3>" + _monthly_sample_chart(monthly) + _html_table(monthly) + "<h3>PSI Top10</h3>" + _html_table(top_psi) + "</div><h2>变量卡方分箱明细</h2><div class='panel'><details><summary>展开查看全部变量分箱明细（共 " + str(bins.height) + " 行）</summary>" + _html_table(bins, limit=500) + "</details></div><h2>变量分布</h2><div class='panel'>" + _distribution_grid(bins, top_iv) + "</div><h2>变量相关性热力图</h2><div class='panel'>" + _correlation_heatmap(correlation_matrix) + "</div><h2>高相关字段对</h2><div class='panel'>" + _html_table(correlation_pairs) + "</div><p class='note'>报告由 LangGraph Skill 生成；CSV 文件保留完整明细，可下载后进一步分析。</p><script>function sortTable(button){const table=button.closest('table');const th=button.closest('th');const index=[...th.parentElement.children].indexOf(th);const body=table.tBodies[0];const rows=[...body.rows];const asc=button.dataset.asc!=='1';rows.sort((a,b)=>{const av=a.cells[index].innerText.trim(),bv=b.cells[index].innerText.trim(),an=parseFloat(av.replace(/,/g,'')),bn=parseFloat(bv.replace(/,/g,''));if(!Number.isNaN(an)&&!Number.isNaN(bn))return asc?an-bn:bn-an;return asc?av.localeCompare(bv):bv.localeCompare(av)});rows.forEach(row=>body.appendChild(row));button.dataset.asc=asc?'1':'0'}</script></body></html>"
    # Top10 排名表只展示指标，避免把内部 dtype、metrics_backend 等实现字段暴露给用户。
    report_html = report_html.replace(_html_table(top_iv), _html_table(top_iv_table), 1)
    report_html = report_html.replace(_html_table(top_ks), _html_table(top_ks_table), 1)
    # 报告顶部不再重复展示运行摘要，改用固定导航快速定位各分析章节。
    report_html = report_html.replace("<p class='subtitle'>" + html.escape(summary) + " · 指标计算后端：toad</p>", "", 1)
    report_html = report_html.replace("<h1>风控数据 EDA 报告</h1>", "<h1>风控数据 EDA 报告</h1><nav class='report-nav'><a href='#section-diagnosis'>样本诊断</a><a href='#section-features'>特征指标</a><a href='#section-discrimination'>区分度</a><a href='#section-stability'>稳定性与时间</a><a href='#section-binning'>分箱明细</a><a href='#section-distribution'>变量分布</a><a href='#section-correlation'>相关性</a></nav>", 1)
    report_html = report_html.replace("<h2>样本诊断</h2>", "<h2 id='section-diagnosis'>样本诊断</h2>", 1)
    report_html = report_html.replace("<h2>特征指标总览</h2>", "<h2 id='section-features'>特征指标总览</h2>", 1)
    report_html = report_html.replace("<h2>区分度概览</h2>", "<h2 id='section-discrimination'>区分度概览</h2>", 1)
    report_html = report_html.replace("<h2>稳定性与时间分布</h2>", "<h2 id='section-stability'>稳定性与时间分布</h2>", 1)
    report_html = report_html.replace("<h2>变量卡方分箱明细</h2>", "<h2 id='section-binning'>变量卡方分箱明细</h2>", 1)
    report_html = report_html.replace("<h2>变量分布</h2>", "<h2 id='section-distribution'>变量分布</h2>", 1)
    report_html = report_html.replace("<h2>变量相关性热力图</h2>", "<h2 id='section-correlation'>变量相关性热力图</h2>", 1)
    report_html = report_html.replace("</style>", ".report-nav{position:fixed;top:0;left:0;right:0;z-index:20;display:flex;gap:4px;align-items:center;overflow-x:auto;padding:10px 28px;background:rgba(255,255,255,.96);border-bottom:1px solid #dbe3ef;box-shadow:0 2px 8px #1d3b6a12}.report-nav a{color:#334155;text-decoration:none;white-space:nowrap;padding:7px 11px;border-radius:7px;font-size:12px}.report-nav a:hover{background:#eff6ff;color:#1d4ed8}body{padding-top:76px}</style>", 1)
    # 将原先整张分箱大表替换为“每个变量一个折叠面板”。
    legacy_binning = "<details><summary>展开查看全部变量分箱明细（共 " + str(bins.height) + " 行）</summary>" + _html_table(bins, limit=500) + "</details>"
    report_html = report_html.replace(legacy_binning, _html_binning_sections(bins), 1)
    report_html = report_html.replace("</style>", ".binning-feature{margin:8px 0;border:1px solid #e6eaf2;border-radius:8px;padding:8px;background:#fbfcfe}.binning-feature summary{cursor:pointer;color:#1d4ed8;font-weight:600;padding:4px}</style>", 1)
    report_html = report_html.replace("报告由 LangGraph Skill 生成；CSV 文件保留完整明细，可下载后进一步分析。", "报告由 LangGraph Skill 生成；详细指标已内嵌在本报告中。")
    report.write_text(report_html, encoding="utf-8")
    eda_summary_path = output / "eda_summary.json"
    return {"status": "success", "node_id": "eda-analysis", "summary": summary, "eda_summary": summary, "diagnosis": diagnosis, "metrics_backend": "toad", "report": str(report), "summary_markdown": str(conclusion), "diagnosis_markdown": str(conclusion), "diagnosis_json": str(eda_summary_path), "artifacts": detail_artifacts + [str(report), str(conclusion), str(eda_summary_path)], "display_files": [str(report), str(conclusion)]}

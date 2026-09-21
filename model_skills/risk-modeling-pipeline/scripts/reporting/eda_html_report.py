"""Offline HTML report for the complete structured-data EDA evidence.

The report is intentionally self-contained: all tables and SVG charts are
written into one HTML file so it can be opened in an air-gapped environment.
Calculations are performed by :mod:`eda.analytics`; this module only renders
the payload and never changes metric values.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable


_LABELS = {
    "feature": "变量",
    "variable_type": "变量类型",
    "mean": "平均值",
    "std": "标准差",
    "missing_rate": "缺失率",
    "iv": "IV",
    "ks": "KS",
    "auc": "AUC",
    "auc_adjusted": "调整后 AUC",
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
    "event_month": "月份",
    "baseline_month": "基准月",
    "psi": "PSI",
    "average_psi": "平均 PSI",
    "max_psi": "最大 PSI",
    "feature_1": "变量一",
    "feature_2": "变量二",
    "method": "方法",
    "correlation": "相关系数",
    "abs_correlation": "绝对相关系数",
    "pair_count": "有效样本对数",
    "lift_10": "前 10% Lift",
    "score_min": "分箱下界",
    "score_max": "分箱上界",
    "cumulative_bad_capture": "累计坏样本捕获率",
    "cumulative_good_capture": "累计好样本捕获率",
    "cumulative_sample_rate": "累计样本占比",
}


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "-"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _render(value: Any, header: str) -> str:
    if value is None:
        return "-"
    if header.endswith("rate") or header.endswith("_rate") or header in {
        "bad_distribution",
        "good_distribution",
        "cumulative_bad_capture",
        "cumulative_good_capture",
        "cumulative_sample_rate",
    }:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return escape(str(value))
    if header in {
        "iv",
        "ks",
        "auc",
        "auc_adjusted",
        "psi",
        "average_psi",
        "max_psi",
        "woe",
        "iv_bin",
        "cumulative_iv",
        "cumulative_ks",
        "lift_10",
        "correlation",
        "abs_correlation",
    }:
        return _num(value)
    if isinstance(value, float):
        return _num(value)
    return escape(str(value))


def _table(table: dict[str, Any], *, limit: int | None = None, class_name: str = "") -> str:
    headers = list(table.get("headers", []))
    rows = list(table.get("rows", []))
    if limit is not None:
        rows = rows[:limit]
    if not headers:
        return "<p class='muted'>暂无数据</p>"
    head = "".join(f"<th>{escape(_LABELS.get(header, header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_render(value, header)}</td>" for header, value in zip(headers, row)) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap {escape(class_name)}'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _group_rows(table: dict[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    headers = table.get("headers", [])
    if key not in headers:
        return {}
    index = headers.index(key)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.get("rows", []):
        grouped[str(row[index])].append(dict(zip(headers, row)))
    return grouped


def _svg_bars(rows: Iterable[dict[str, Any]], label: str, value: str, title: str, *, percent: bool = False) -> str:
    rows = list(rows)
    if not rows:
        return "<p class='muted'>暂无图表数据</p>"
    width, height = 760, 270
    left, right, top, bottom = 150, 30, 34, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max((float(row.get(value) or 0) for row in rows), default=0.0) or 1.0
    bar_h = max(8, min(24, plot_h / len(rows) - 4))
    elements = [f"<text x='{width/2:.0f}' y='20' text-anchor='middle' class='chart-title'>{escape(title)}</text>"]
    for index, row in enumerate(rows):
        y = top + index * (plot_h / len(rows)) + 4
        number = float(row.get(value) or 0)
        bar_w = max(1.0, number / maximum * plot_w)
        text_value = f"{number:.2%}" if percent else f"{number:.4f}"
        elements.append(
            f"<text x='{left-8}' y='{y+bar_h-2:.1f}' text-anchor='end' class='chart-label'>{escape(str(row.get(label, '')))}</text>"
            f"<rect x='{left}' y='{y:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' class='bar'></rect>"
            f"<text x='{left+bar_w+6:.1f}' y='{y+bar_h-2:.1f}' class='chart-value'>{text_value}</text>"
        )
    return f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>{''.join(elements)}</svg>"


def _svg_line(rows: list[dict[str, Any]], x_key: str, series: list[tuple[str, str, str]], title: str) -> str:
    if not rows:
        return "<p class='muted'>暂无图表数据</p>"
    width, height = 760, 280
    left, right, top, bottom = 58, 24, 34, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    maxima = [float(row.get(key) or 0) for row in rows for _, key, _ in series]
    maximum = max(maxima, default=0.0) or 1.0
    # Bad-rate and PSI are both ratios; use a shared 0..max scale for a clear
    # month-to-month comparison and annotate the unit in the legend.
    x_count = max(1, len(rows) - 1)
    points = []
    legends = []
    for series_index, (name, key, color) in enumerate(series):
        coords = []
        for index, row in enumerate(rows):
            x = left + plot_w * index / x_count
            y = top + plot_h * (1 - float(row.get(key) or 0) / maximum)
            coords.append((x, y))
        path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
        legends.append(f"<text x='{width-right-115+series_index*78}' y='22' class='legend' style='fill:{color}'>{escape(name)}</text>")
        points.append(f"<path d='{path}' class='line' style='stroke:{color}'></path>")
        points.extend(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' class='dot' style='fill:{color}'><title>{escape(str(row.get(x_key)))}：{escape(_render(row.get(key), key))}</title></circle>"
            for row, (x, y) in zip(rows, coords)
        )
    ticks = []
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_h * (1 - tick / 4)
        ticks.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' class='grid'></line><text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='chart-label'>{value:.2%}</text>")
    labels = []
    for index, row in enumerate(rows):
        if len(rows) <= 12 or index in {0, len(rows)-1} or index % max(1, len(rows)//8) == 0:
            x = left + plot_w * index / x_count
            labels.append(f"<text x='{x:.1f}' y='{height-12}' text-anchor='middle' class='chart-label'>{escape(str(row.get(x_key)))}</text>")
    return f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'><text x='{width/2:.0f}' y='20' text-anchor='middle' class='chart-title'>{escape(title)}</text>{''.join(legends)}{''.join(ticks)}{''.join(labels)}{''.join(points)}</svg>"


def _svg_monthly_samples(rows: list[dict[str, Any]], title: str) -> str:
    """Render readable monthly volume bars with a separate bad-rate axis."""
    if not rows:
        return "<p class='muted'>暂无图表数据</p>"
    width, height = 900, 330
    left, right, top, bottom = 64, 64, 42, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    count_max = max((float(row.get("sample_count") or 0) for row in rows), default=0.0) or 1.0
    rate_max = max((float(row.get("bad_rate") or 0) for row in rows), default=0.0)
    rate_max = max(0.05, rate_max * 1.25)
    slot = plot_w / max(1, len(rows))
    bar_width = max(10.0, min(42.0, slot * 0.62))
    elements = [
        f"<text x='{width/2:.0f}' y='20' text-anchor='middle' class='chart-title'>{escape(title)}</text>",
        f"<text x='{left}' y='34' class='chart-label'>样本数</text>",
        f"<text x='{width-right}' y='34' text-anchor='end' class='chart-label'>坏账率</text>",
    ]
    for tick in range(5):
        fraction = tick / 4
        count_value = count_max * fraction
        y = top + plot_h * (1 - fraction)
        rate_value = rate_max * fraction
        elements.append(
            f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' class='grid'></line>"
            f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='chart-label'>{count_value:.0f}</text>"
            f"<text x='{width-right+8}' y='{y+4:.1f}' class='chart-label'>{rate_value:.1%}</text>"
        )
    for index, row in enumerate(rows):
        x_center = left + slot * (index + 0.5)
        good = float(row.get("good_count") or 0)
        bad = float(row.get("bad_count") or 0)
        good_h = good / count_max * plot_h
        bad_h = bad / count_max * plot_h
        base_y = top + plot_h
        elements.extend(
            [
                f"<rect x='{x_center-bar_width/2:.1f}' y='{base_y-good_h:.1f}' width='{bar_width:.1f}' height='{good_h:.1f}' class='monthly-good'><title>{escape(str(row.get('event_month')))} 好样本：{good:.0f}</title></rect>",
                f"<rect x='{x_center-bar_width/2:.1f}' y='{base_y-good_h-bad_h:.1f}' width='{bar_width:.1f}' height='{bad_h:.1f}' class='monthly-bad'><title>{escape(str(row.get('event_month')))} 坏样本：{bad:.0f}</title></rect>",
            ]
        )
        if len(rows) <= 18 or index in {0, len(rows) - 1} or index % max(1, len(rows) // 10) == 0:
            elements.append(
                f"<text x='{x_center:.1f}' y='{height-18}' text-anchor='middle' class='chart-label'>{escape(str(row.get('event_month')))}</text>"
            )
    rate_points = []
    for index, row in enumerate(rows):
        x = left + slot * (index + 0.5)
        rate = float(row.get("bad_rate") or 0)
        y = top + plot_h * (1 - rate / rate_max)
        rate_points.append((x, y))
        elements.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' class='monthly-rate-dot'><title>{escape(str(row.get('event_month')))} 坏账率：{rate:.2%}</title></circle>"
        )
    path = " ".join(("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}" for index, (x, y) in enumerate(rate_points))
    elements.append(f"<path d='{path}' class='monthly-rate-line'></path>")
    elements.append(
        f"<rect x='{left+10}' y='{height-38}' width='12' height='12' class='monthly-good'></rect><text x='{left+28}' y='{height-28}' class='chart-label'>好样本</text>"
        f"<rect x='{left+88}' y='{height-38}' width='12' height='12' class='monthly-bad'></rect><text x='{left+106}' y='{height-28}' class='chart-label'>坏样本</text>"
        f"<line x1='{left+168}' x2='{left+184}' y1='{height-32}' y2='{height-32}' class='monthly-rate-line'></line><text x='{left+192}' y='{height-28}' class='chart-label'>坏账率</text>"
    )
    return f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>{''.join(elements)}</svg>"


def _correlation_heatmap(table: dict[str, Any]) -> str:
    """Render the wide correlation matrix as a signed red/blue heatmap."""
    headers = list(table.get("headers", []))
    rows = [dict(zip(headers, row)) for row in table.get("rows", [])]
    features = [header for header in headers if header != "feature"]
    if not rows or not features:
        return "<p class='muted'>暂无数值型变量相关性矩阵</p>"
    head = "<th>变量</th>" + "".join(f"<th title='{escape(feature)}'>{escape(feature)}</th>" for feature in features)
    body = []
    for row in rows:
        feature = str(row.get("feature", ""))
        cells = [f"<th title='{escape(feature)}'>{escape(feature)}</th>"]
        for column in features:
            value = row.get(column)
            try:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError
                intensity = min(1.0, abs(numeric))
                rgb = "37,99,235" if numeric >= 0 else "220,38,38"
                style = f"background:rgba({rgb},{0.06 + intensity * 0.78:.3f})"
                rendered = f"{numeric:.2f}"
            except (TypeError, ValueError):
                style = "background:#f8fafc"
                rendered = "-"
            cells.append(f"<td style='{style}' title='{escape(column)} / {escape(feature)}'>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<h3>相关性热力图</h3><div class='table-wrap corr-heatmap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div><p class='muted'>蓝色为正相关，红色为负相关，颜色越深表示绝对相关系数越高。</p>"


def _heatmap(table: dict[str, Any], row_key: str, col_key: str, value_key: str, title: str) -> str:
    headers = table.get("headers", [])
    rows = [dict(zip(headers, row)) for row in table.get("rows", [])]
    if not rows:
        return "<p class='muted'>暂无数据</p>"
    columns = sorted({str(row.get(col_key)) for row in rows})
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        grouped[str(row.get(row_key))][str(row.get(col_key))] = row.get(value_key)
    maximum = max((float(value or 0) for row in grouped.values() for value in row.values()), default=0.0) or 1.0
    head = "<th>变量</th>" + "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for key in sorted(grouped):
        cells = [f"<th>{escape(key)}</th>"]
        for column in columns:
            value = grouped[key].get(column)
            numeric = float(value or 0)
            intensity = min(1.0, numeric / maximum)
            cells.append(f"<td style='background:rgba(220,38,38,{0.08 + intensity*0.72:.3f})' title='{escape(_render(value, value_key))}'>{_render(value, value_key)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<h3>{escape(title)}</h3><div class='table-wrap heatmap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _distribution_sections(
    binning: dict[str, Any], *, include_table: bool = True, include_chart: bool = True
) -> str:
    headers = binning.get("headers", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in binning.get("rows", []):
        grouped[str(row[headers.index("feature")])].append(dict(zip(headers, row)))
    sections = []
    for feature in sorted(grouped):
        rows = grouped[feature]
        maximum = max((float(row.get("sample_count") or 0) for row in rows), default=1.0) or 1.0
        bars = []
        if include_chart:
            for row in rows:
                height = max(2.0, float(row.get("sample_count") or 0) / maximum * 120)
                bars.append(
                    f"<div class='dist-col'><div class='dist-value'>{_num(row.get('bad_rate'), 2)}</div><div class='dist-bar' style='height:{height:.1f}px'></div><div class='dist-label'>{escape(str(row.get('bin', '')))}</div></div>"
                )
        chart = (
            f"<div class='distribution'><div class='dist-axis'>柱高=样本数；柱顶数字=坏账率</div>{''.join(bars)}</div>"
            if include_chart
            else ""
        )
        detail_table = (
            _table({'headers': headers, 'rows': [[row.get(header) for header in headers] for row in rows]})
            if include_table
            else ""
        )
        sections.append(
            f"<details><summary>{escape(feature)}</summary>{chart}{detail_table}</details>"
        )
    return "".join(sections) or "<p class='muted'>暂无分布数据</p>"


def write_eda_html_report(payload_path: str | Path, output_path: str | Path) -> Path:
    """Render the complete EDA payload into one offline HTML report."""
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    tables = payload.get("tables", {})
    monthly = tables.get("6.月度样本", {})
    monthly_rows = [dict(zip(monthly.get("headers", []), row)) for row in monthly.get("rows", [])]
    overview = tables.get("4.单变量概览", {})
    overview_rows = [dict(zip(overview.get("headers", []), row)) for row in overview.get("rows", [])]
    overview_top10 = sorted(overview_rows, key=lambda row: float(row.get("iv") or 0), reverse=True)[:10]
    psi_summary = tables.get("7b.变量 PSI 汇总", {})
    psi_summary_rows = [dict(zip(psi_summary.get("headers", []), row)) for row in psi_summary.get("rows", [])]
    html = f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>风控数据 EDA 报告</title><style>
:root {{ color-scheme:light; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:#172033; background:#f5f7fb; }}
body {{ margin:0; }} main {{ max-width:1440px; margin:auto; padding:28px; }} h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ margin:0 0 14px; font-size:20px; }} h3 {{ margin:18px 0 10px; font-size:16px; }} .subtitle,.muted {{ color:#667085; }} nav {{ position:sticky; top:0; z-index:3; background:#172b4d; padding:12px 28px; overflow:auto; white-space:nowrap; }} nav a {{ color:#fff; text-decoration:none; margin-right:18px; font-size:13px; }} section {{ background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:20px; margin:18px 0; box-shadow:0 2px 8px #1018280a; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }} .table-wrap {{ overflow:auto; max-height:620px; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ padding:8px 9px; border-bottom:1px solid #eaecf0; text-align:left; white-space:nowrap; }} th {{ position:sticky; top:0; background:#f9fafb; color:#475467; z-index:1; }} .chart {{ width:100%; min-height:260px; }} .chart-title {{ fill:#172033; font-size:14px; font-weight:600; }} .chart-label,.chart-value,.legend {{ fill:#667085; font-size:11px; }} .bar {{ fill:#2f75b5; }} .line {{ fill:none; stroke-width:2.2; }} .dot {{ stroke:#fff; stroke-width:1; }} .grid {{ stroke:#eaecf0; stroke-width:1; }} .heatmap td,.corr-heatmap td {{ text-align:center; font-variant-numeric:tabular-nums; }} .corr-heatmap table {{ min-width:900px; }} .corr-heatmap th,.corr-heatmap td {{ min-width:54px; text-align:center; }} .monthly-good {{ fill:#5b8ff9; }} .monthly-bad {{ fill:#f08a5d; }} .monthly-rate-line {{ fill:none; stroke:#d64545; stroke-width:2.5; }} .monthly-rate-dot {{ fill:#d64545; stroke:#fff; stroke-width:1; }} details {{ border-top:1px solid #eaecf0; padding:10px 0; }} summary {{ cursor:pointer; font-weight:600; }} .distribution {{ display:flex; align-items:flex-end; gap:8px; overflow-x:auto; padding:24px 8px 34px; border-bottom:1px solid #eaecf0; }} .dist-col {{ min-width:64px; text-align:center; }} .dist-bar {{ background:linear-gradient(180deg,#4e8cff,#155eef); border-radius:4px 4px 0 0; }} .dist-value {{ height:18px; font-size:11px; color:#c2410c; }} .dist-label {{ margin-top:6px; font-size:10px; writing-mode:vertical-rl; max-height:100px; overflow:hidden; }} .dist-axis {{ color:#667085; font-size:12px; position:absolute; margin-top:-22px; }} .note {{ background:#f8fafc; border-left:4px solid #2f75b5; padding:12px 14px; color:#475467; }}
@media(max-width:720px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} section {{ padding:14px; }} }}
</style></head><body><nav><a href='#overview'>总览</a><a href='#monthly'>月度样本</a><a href='#variables'>变量概览</a><a href='#bins'>卡方/分箱明细</a><a href='#psi'>变量 PSI</a><a href='#corr'>相关性</a><a href='#monthly-disc'>月度区分度</a><a href='#dist'>变量分布</a><a href='#conclusion'>结论</a></nav><main>
<h1>风控数据 EDA 报告</h1><div class='subtitle'>离线报告 · 所有指标由本次 EDA 计算生成 · 可在无网络环境直接打开</div>
<section id='overview'><h2>1. 数据总览</h2>{_table(tables.get('1.数据概况', {}))}<div class='note'>PSI 口径：变量 PSI 按变量分箱后与基准月比较；月度样本 PSI 按好/坏标签构成与基准月比较。小样本月份的 KS、AUC 和 PSI 需要结合样本量解读。</div></section>
<section id='monthly'><h2>2. 月度样本统计</h2><p class='muted'>柱形按好/坏样本堆叠显示样本规模，红色折线使用右侧坐标显示坏账率；PSI 保留在下方明细表中，避免不同量纲叠加误读。</p>{_svg_monthly_samples(monthly_rows, '月度样本规模与坏账率')}{_table(monthly)}</section>
<section id='variables'><h2>3. 变量概览（Top 10）</h2><p class='muted'>按 IV 从高到低展示 Top 10；完整变量明细仍保留在 outputs/eda-analysis/univariate_overview.csv。</p><div class='grid'><div>{_svg_bars(overview_top10, 'feature', 'iv', 'Top 10 IV')}</div><div>{_svg_bars(sorted(overview_rows, key=lambda row: float(row.get('ks') or 0), reverse=True)[:10], 'feature', 'ks', 'Top 10 KS')}</div></div>{_table({'headers': overview.get('headers', []), 'rows': [[row.get(header) for header in overview.get('headers', [])] for row in overview_top10]})}</section>
<section id='bins'><h2>4. 变量卡方分箱明细</h2><p class='muted'>每个变量展开后可查看分箱区间、样本数、坏样本数、坏账率、坏样本占比、WOE、分箱 IV、累计 IV、累计 KS。</p>{_distribution_sections(tables.get('5.分箱明细', {}), include_chart=False)}</section>
<section id='psi'><h2>5. 变量 PSI 稳定性</h2>{_table(psi_summary)}{_heatmap(tables.get('7.稳定性PSI', {}), 'feature', 'event_month', 'psi', '各变量各月份 PSI')}{_table(tables.get('7.稳定性PSI', {}))}</section>
<section id='corr'><h2>6. 变量相关性</h2><p class='muted'>相关性矩阵仅对数值型变量计算；成对明细同时保留有效样本对数和相关性方法。</p>{_correlation_heatmap(tables.get('8b.相关性矩阵', {}))}{_table(tables.get('8.相关性', {}))}</section>
<section id='monthly-disc'><h2>7. 变量月度区分度</h2><p class='muted'>每个变量展开后可查看分月 KS、AUC、PSI、前 10% Lift、好/坏样本数、总样本数、坏账率和样本占比。</p>{''.join(f"<details><summary>{escape(feature)}</summary>{_table({'headers': tables.get('9.月度区分度', {}).get('headers', []), 'rows': [[row.get(header) for header in tables.get('9.月度区分度', {}).get('headers', [])] for row in rows]})}</details>" for feature, rows in sorted(_group_rows(tables.get('9.月度区分度', {}), 'feature').items()))}</section>
<section id='dist'><h2>8. 各变量分布图</h2><p class='muted'>柱高表示分箱样本数量，柱顶数字表示该分箱坏账率；完整数值见分箱明细。</p>{_distribution_sections(tables.get('5.分箱明细', {}), include_table=False)}</section>
<section id='conclusion'><h2>9. 结果概述与建议</h2>{_table(tables.get('11.结果概述', {}))}</section>
</main></body></html>"""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output

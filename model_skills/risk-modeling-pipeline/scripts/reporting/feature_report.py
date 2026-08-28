"""Self-contained HTML report for feature-processing evidence.

The report intentionally has no CDN, browser plugin, or plotting dependency.
It is generated from the authoritative feature statistics/correlation tables
and can be opened directly from an offline OpenCode workspace.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import polars as pl


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _bar_rows(stats: pl.DataFrame, metric: str, title: str, top_n: int = 20) -> str:
    if metric not in stats.columns or stats.height == 0:
        return f"<section><h2>{escape(title)}</h2><p class='muted'>暂无数据</p></section>"
    rows = (
        stats.select(["feature", metric])
        .drop_nulls(metric)
        .sort(metric, descending=True)
        .head(top_n)
        .to_dicts()
    )
    if not rows:
        return f"<section><h2>{escape(title)}</h2><p class='muted'>暂无数据</p></section>"
    maximum = max(float(row[metric]) for row in rows) or 1.0
    bars = []
    for row in rows:
        feature = escape(str(row["feature"]))
        value = float(row[metric])
        width = max(1.0, min(100.0, value / maximum * 100.0))
        bars.append(
            f"<div class='bar-row'><span class='bar-label' title='{feature}'>{feature}</span>"
            f"<span class='bar-track'><span class='bar-fill' style='width:{width:.2f}%'></span></span>"
            f"<span class='bar-value'>{_number(value)}</span></div>"
        )
    return f"<section><h2>{escape(title)}（Top {len(rows)}）</h2>{''.join(bars)}</section>"


def _stats_table(stats: pl.DataFrame) -> str:
    columns = [
        ("feature", "字段"),
        ("variable_type", "变量类型"),
        ("missing_rate", "缺失率"),
        ("iv", "IV"),
        ("ks", "KS"),
        ("max_monthly_psi", "最大月度 PSI"),
    ]
    available = [(key, label) for key, label in columns if key in stats.columns]
    head = "".join(f"<th>{escape(label)}</th>" for _, label in available)
    body = []
    for row in stats.to_dicts():
        cells = []
        for key, _ in available:
            value = row.get(key)
            if key == "feature":
                rendered = escape(str(value))
            elif key == "variable_type":
                rendered = escape(str(value))
            else:
                rendered = _number(value)
            cells.append(f"<td>{rendered}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<section><h2>字段级指标明细</h2><div class='table-wrap'>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        "</div></section>"
    )


def _correlation_table(correlations: pl.DataFrame, top_n: int = 30) -> str:
    if correlations.height == 0:
        return "<section><h2>高相关变量对</h2><p class='muted'>暂无相关变量对</p></section>"
    rows = correlations.head(top_n).to_dicts()
    body = []
    for row in rows:
        left = escape(str(row.get("feature_a", row.get("feature_1", ""))))
        right = escape(str(row.get("feature_b", row.get("feature_2", ""))))
        value = row.get("abs_correlation", row.get("correlation"))
        body.append(f"<tr><td>{left}</td><td>{right}</td><td>{_number(value)}</td></tr>")
    return (
        "<section><h2>高相关变量对</h2><div class='table-wrap'><table>"
        "<thead><tr><th>字段 A</th><th>字段 B</th><th>|相关系数|</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )


def write_feature_report(
    stats: pl.DataFrame,
    correlations: pl.DataFrame,
    output_path: str | Path,
    *,
    title: str = "特征预处理与筛选分析",
) -> Path:
    """Write a standalone feature evidence dashboard and return its path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_count = stats.height
    numeric_count = sum(
        str(value).lower().startswith(("int", "uint", "float", "decimal"))
        for value in stats.get_column("variable_type").to_list()
    ) if "variable_type" in stats.columns else 0
    cards = (
        f"<div class='cards'><div><strong>{feature_count}</strong><span>特征数</span></div>"
        f"<div><strong>{numeric_count}</strong><span>数值型字段</span></div>"
        f"<div><strong>{correlations.height}</strong><span>相关性记录</span></div></div>"
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root {{ color-scheme: light; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172033; background:#f5f7fb; }}
body {{ margin:0; padding:28px; }} main {{ max-width:1180px; margin:auto; }}
h1 {{ margin:0 0 6px; font-size:25px; }} h2 {{ margin:0 0 16px; font-size:17px; }}
.subtitle,.muted {{ color:#667085; }} section {{ background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:20px; margin:18px 0; box-shadow:0 2px 8px #1018280a; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:22px 0; }} .cards div {{ min-width:150px; background:#fff; border:1px solid #e4e7ec; border-radius:10px; padding:14px 18px; }}
.cards strong {{ display:block; font-size:24px; color:#175cd3; }} .cards span {{ color:#667085; font-size:13px; }}
.bar-row {{ display:grid; grid-template-columns:minmax(120px,220px) 1fr 72px; gap:10px; align-items:center; margin:8px 0; font-size:13px; }}
.bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .bar-track {{ height:10px; background:#eef2f6; border-radius:8px; overflow:hidden; }} .bar-fill {{ display:block; height:100%; background:linear-gradient(90deg,#4e8cff,#155eef); border-radius:8px; }} .bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }}
.table-wrap {{ overflow:auto; max-height:560px; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:9px 10px; border-bottom:1px solid #eaecf0; text-align:left; white-space:nowrap; }} th {{ position:sticky; top:0; background:#f9fafb; color:#475467; }}
@media(max-width:700px) {{ body {{ padding:14px; }} .bar-row {{ grid-template-columns:100px 1fr 62px; }} }}
</style></head><body><main><h1>{escape(title)}</h1><div class="subtitle">由 feature-processing 节点生成；指标来自当前已确认的数据配置，供筛选阈值确认使用。</div>{cards}
{_bar_rows(stats, "missing_rate", "缺失率")}
{_bar_rows(stats, "iv", "IV 信息价值")}
{_bar_rows(stats, "ks", "KS 区分度")}
{_bar_rows(stats, "max_monthly_psi", "最大月度 PSI / 稳定性")}
{_correlation_table(correlations)}
{_stats_table(stats)}
</main></body></html>"""
    output.write_text(html, encoding="utf-8")
    return output


"""Self-contained HTML model report for OpenCode/Linux environments."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


_LABELS = {
    "dataset": "样本集", "auc": "AUC", "ks": "KS", "pr_auc": "PR-AUC", "brier_score": "Brier Score", "sample_count": "样本数", "bad_count": "坏样本数",
    "bad_rate": "坏样本率", "score_psi_vs_train": "相对 Train 分数 PSI", "metrics_backend": "指标后端",
    "candidate": "候选模型", "best_iteration": "最佳轮次", "train_auc": "Train AUC", "test_auc": "Test AUC",
    "test_ks": "Test KS", "parameters": "模型参数", "auc_gap": "AUC 差距", "selected_candidate": "是否最终采用",
    "improvement_vs_baseline": "相对基线提升", "feature": "特征", "train_iv": "Train IV", "test_iv": "Test IV",
    "validation_iv": "Validation IV（Test）", "oot_iv": "OOT IV", "train_missing_rate": "Train 缺失率",
    "test_missing_rate": "Test 缺失率", "validation_missing_rate": "Validation 缺失率（Test）", "oot_missing_rate": "OOT 缺失率",
    "train_gini": "Train Gini", "test_gini": "Test Gini", "validation_gini": "Validation Gini（Test）", "oot_gini": "OOT Gini",
    "train_entropy": "Train 信息熵", "test_entropy": "Test 信息熵", "validation_entropy": "Validation 信息熵（Test）", "oot_entropy": "OOT 信息熵",
    "gain_importance": "Gain 重要性", "split_importance": "分裂次数", "gain_share": "Gain 占比", "cumulative_gain_share": "累计 Gain 占比",
    "event_month": "月份", "reliability": "可靠性", "population_rate": "累计样本占比", "cumulative_sample_rate": "累计样本占比", "cumulative_bad_capture": "累计坏样本捕获率", "cumulative_bad_rate": "累计坏样本率", "cumulative_good_rate": "累计好样本率", "threshold": "阈值",
    "fpr": "假阳性率", "tpr": "真阳性率", "mean_predicted": "平均预测概率", "observed_bad_rate": "实际坏样本率", "absolute_error": "校准绝对误差", "estimate": "原始估计", "lower": "置信区间下限", "upper": "置信区间上限", "rounds": "Bootstrap轮数", "seed": "随机种子", "dimension": "分群维度", "group_value": "分群值", "train_permutation_importance": "Train置换重要性", "test_permutation_importance": "Test置换重要性", "oot_permutation_importance": "OOT置换重要性",
    "cv_strategy": "CV策略", "train_month_start": "训练起始月", "train_month_end": "训练结束月", "validation_month_start": "验证起始月", "validation_month_end": "验证结束月",
}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _table(table: dict[str, Any], limit: int = 300) -> str:
    headers = table.get("headers", [])
    rows = table.get("rows", [])[:limit]
    if not headers:
        return "<p class='muted'>暂无数据</p>"
    head = "".join(f"<th>{_esc(_LABELS.get(value, value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _records(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    table = payload.get("tables", {}).get(name, {})
    return [dict(zip(table.get("headers", []), row)) for row in table.get("rows", [])]


def _points(table: dict[str, Any], x: str, y: str, series: str | None = None, series_value: Any = None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in table.get("rows", []):
        record = dict(zip(table.get("headers", []), row))
        if series is not None and record.get(series) != series_value:
            continue
        try:
            xv, yv = float(record[x]), float(record[y])
        except (TypeError, ValueError, KeyError):
            continue
        if xv == xv and yv == yv:
            points.append((xv, yv))
    return points


def _line_chart(title: str, series: list[tuple[str, list[tuple[float, float]], str]], *, x_label: str, y_label: str, x_max: float = 1.0, y_max: float = 1.0) -> str:
    width, height, left, top, right, bottom = 720, 300, 58, 32, 18, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#2F75B5", "#ED7D31", "#70AD47", "#8064A2"]
    def xy(x: float, y: float) -> tuple[float, float]:
        return left + max(0.0, min(1.0, x / x_max)) * plot_w, top + (1 - max(0.0, min(1.0, y / y_max))) * plot_h
    paths = []
    legend = []
    for idx, (name, points, color) in enumerate(series):
        if not points:
            continue
        points = sorted(points)
        step = max(1, len(points) // 240)
        sampled = points[::step]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        d = " ".join(("M" if i == 0 else "L") + f" {xy(x, y)[0]:.1f},{xy(x, y)[1]:.1f}" for i, (x, y) in enumerate(sampled))
        paths.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>" )
        legend.append(f"<span><i style='background:{color}'></i>{_esc(name)}</span>")
    grid = "".join(f"<line x1='{left}' y1='{top + plot_h * frac:.1f}' x2='{left + plot_w}' y2='{top + plot_h * frac:.1f}' stroke='#e5e7eb'/><text x='{left - 8}' y='{top + plot_h * frac + 4:.1f}' text-anchor='end'>{1-frac:.1f}</text>" for frac in (0, .25, .5, .75, 1))
    return f"<section class='chart'><h3>{_esc(title)}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title)}'><rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='#cbd5e1'/>{grid}<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#64748b'/><line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#64748b'/>" + "".join(paths) + f"<text x='{left + plot_w/2}' y='{height - 8}' text-anchor='middle'>{_esc(x_label)}</text><text x='14' y='{top + plot_h/2}' transform='rotate(-90 14 {top + plot_h/2})' text-anchor='middle'>{_esc(y_label)}</text></svg><div class='legend'>{''.join(legend)}</div></section>"


def _bar_chart(title: str, records: list[dict[str, Any]], category: str, value: str, *, limit: int = 15) -> str:
    values: list[tuple[str, float]] = []
    for record in records[:limit]:
        try:
            values.append((str(record.get(category, "")), float(record.get(value))))
        except (TypeError, ValueError):
            continue
    if not values:
        return ""
    max_value = max(abs(value) for _, value in values) or 1.0
    bars = []
    row_h = 24
    width = 720
    for index, (name, number) in enumerate(values):
        y = index * row_h + 4
        bar_w = max(1, abs(number) / max_value * 470)
        bars.append(f"<text x='0' y='{y+15}'>{_esc(name)[:28]}</text><rect x='190' y='{y+3}' width='{bar_w:.1f}' height='15' fill='#2F75B5'/><text x='{200+bar_w:.1f}' y='{y+15}'>{number:.4f}</text>")
    height = max(100, len(values) * row_h + 18)
    return f"<section class='chart'><h3>{_esc(title)}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title)}'>{''.join(bars)}</svg></section>"


def write_model_html_report(payload_path: str | Path, output_path: str | Path) -> Path:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    tables = payload.get("tables", {})
    metrics = _records(payload, "6.效果指标")
    overview = _records(payload, "1.模型概览")
    candidate = _records(payload, "10.候选模型")
    variable = _records(payload, "15.入模变量信息")
    stability = _records(payload, "9.模型稳定性")
    train = next((row for row in metrics if row.get("dataset") == "train"), {})
    test = next((row for row in metrics if row.get("dataset") == "test"), {})
    oot = next((row for row in metrics if row.get("dataset") == "oot"), {})
    cards = "".join(f"<div class='metric'><small>{_esc(row.get('dataset'))} AUC</small><strong>{_esc(row.get('auc'))}</strong><small>KS {_esc(row.get('ks'))}</small></div>" for row in (train, test, oot))
    roc = tables.get("16.ROC曲线", {})
    ks = tables.get("17.KS曲线", {})
    lift = tables.get("7.Lift明细", {})
    calibration = tables.get("18.校准曲线", {})
    roc_series = [
        (dataset, _points(roc, "fpr", "tpr", "dataset", dataset), color)
        for dataset, color in (("train", "#2F75B5"), ("test", "#ED7D31"), ("oot", "#70AD47"))
    ]
    lift_series = []
    for dataset, color in (("train", "#2F75B5"), ("test", "#ED7D31"), ("oot", "#70AD47")):
        lift_series.append((dataset, _points(lift, "cumulative_sample_rate", "cumulative_bad_capture", "dataset", dataset), color))
    lift_bad_rate_series = []
    for dataset, color in (("train", "#2F75B5"), ("test", "#ED7D31"), ("oot", "#70AD47")):
        points = []
        for row in _records(payload, "7.Lift明细"):
            if row.get("dataset") != dataset:
                continue
            try:
                points.append((float(row["bucket"]) / 10.0, float(row["bad_rate"])))
            except (TypeError, ValueError, KeyError):
                continue
        lift_bad_rate_series.append((dataset, points, color))
    html_report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LightGBM 模型报告</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#203040;background:#f5f7fa;margin:0}}main{{max-width:1280px;margin:0 auto;padding:28px}}h1{{color:#17365D;margin:0 0 6px}}h2{{color:#17365D;border-bottom:2px solid #d9e2ea;padding-bottom:8px;margin-top:34px}}h3{{margin:8px 0;color:#17365D}}.subtitle,.muted{{color:#66788A}}.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}}.metric{{background:#fff;border:1px solid #d9e2ea;padding:14px 20px;min-width:130px;box-shadow:0 1px 2px #0000000d}}.metric small{{display:block;color:#66788A;margin:3px 0}}.metric strong{{font-size:25px;color:#17365D}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px}}.chart{{background:#fff;border:1px solid #d9e2ea;padding:12px;overflow:auto}}svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:#334155}}.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#475569}}.legend i{{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}}.table-wrap{{background:#fff;border:1px solid #d9e2ea;overflow:auto;max-height:560px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid #e5e7eb;padding:7px 9px;text-align:left;white-space:nowrap}}th{{background:#eef5fb;color:#17365D;position:sticky;top:0}}@media(max-width:620px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>LightGBM 风控模型报告</h1><div class='subtitle'>模型效果、训练过程、入模变量和稳定性分析（所有指标由确定性代码计算）</div><div class='cards'>{cards}</div>
<h2>1.模型基本信息与最佳参数</h2>{_table({"headers":["指标","值"],"rows":[[row.get("指标"),row.get("值")] for row in overview]})}
<h2>2.模型综合效果</h2><div class='grid'>{_line_chart('Train / Test / OOT ROC 曲线', roc_series + [('随机基线', [(0,0),(1,1)], '#94A3B8')], x_label='False Positive Rate', y_label='True Positive Rate')}{_line_chart('训练集 KS 曲线', [('累计坏样本率', _points(ks, 'population_rate', 'cumulative_bad_rate'), '#ED7D31'), ('累计好样本率', _points(ks, 'population_rate', 'cumulative_good_rate'), '#2F75B5')], x_label='风险排序累计样本占比', y_label='累计占比')}{_line_chart('Train / Test / OOT Lift / Gain', lift_series, x_label='累计样本占比', y_label='累计坏样本捕获率')}{_line_chart('十分位坏样本率趋势', lift_bad_rate_series, x_label='十分位（1=高风险）', y_label='坏样本率')}</div><h3>Train / Test / OOT 效果指标</h3>{_table(tables.get('6.效果指标', {}))}
<h2>3.模型训练情况与参数候选对比</h2>{_table(tables.get('10.候选模型', {}))}<div class='grid'>{_bar_chart('候选模型 Test KS', sorted(candidate, key=lambda row: float(row.get('test_ks') or 0), reverse=True), 'candidate', 'test_ks')}{_bar_chart('候选模型 Test AUC', sorted(candidate, key=lambda row: float(row.get('test_auc') or 0), reverse=True), 'candidate', 'test_auc')}</div>
<h2>4.入模变量信息</h2>{_table(tables.get('15.入模变量信息', {}))}<div class='grid'>{_bar_chart('Gain 特征重要性', _records(payload, '4.特征重要性'), 'feature', 'gain_share')}{_bar_chart('Train IV', sorted(variable, key=lambda row: float(row.get('train_iv') or 0), reverse=True), 'feature', 'train_iv')}</div>
<h2>5.模型稳定性与校准</h2>{_table(tables.get('9.模型稳定性', {}))}{_table(tables.get('8.月度表现', {}), 500)}<div class='grid'>{_line_chart('概率校准曲线', [('Train', _points(calibration, 'mean_predicted', 'observed_bad_rate', 'dataset', 'train'), '#2F75B5'), ('Test', _points(calibration, 'mean_predicted', 'observed_bad_rate', 'dataset', 'test'), '#ED7D31'), ('OOT', _points(calibration, 'mean_predicted', 'observed_bad_rate', 'dataset', 'oot'), '#70AD47'), ('理想校准', [(0,0),(1,1)], '#94A3B8')], x_label='平均预测概率', y_label='实际坏样本率')}</div>{_table(tables.get('19.Bootstrap置信区间', {}))}{_table(tables.get('20.随机种子稳定性', {}))}
<h2>6.分群与特征稳定性</h2>{_table(tables.get('21.分群效果', {}), 500)}{_table(tables.get('22.跨数据集特征重要性', {}), 300)}
<h2>7.模型审查与改进建议</h2>{_table(tables.get('13.AI模型审查', {}), 100)}{_table(tables.get('13.诊断建议', {}), 100)}
</main></body></html>"""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_report, encoding="utf-8")
    return output


__all__ = ["write_model_html_report"]

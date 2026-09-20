"""从训练监控 JSONL 生成离线模型报告。

报告只消费已经持久化的训练证据，不重新训练模型，也不依赖 OpenCode、LangGraph
服务或第三方前端库。这样即使调参进程被超时中断，也能生成一份“已完成轮次”的报告。
"""

from __future__ import annotations

import csv
import html
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

from .evaluation import model_quality_components, model_quality_score


def _esc(value: Any) -> str:
    """把任意值安全转换为 HTML 文本。"""
    return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int = 5) -> str:
    """格式化指标，缺失或非法值显示为短横线。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    """格式化比例字段为百分比。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}%}"


def _metric(event: dict[str, Any], dataset: str, name: str) -> float | None:
    """从一轮事件中读取指定数据集指标。"""
    value = ((event.get("metrics") or {}).get(dataset) or {}).get(name)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _event_quality_score(event: dict[str, Any]) -> float | None:
    """读取或计算单轮 Train/Validate/OOT 综合质量分。"""
    metrics = event.get("metrics") or {}
    raw = metrics.get("model_quality_score")
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return model_quality_score(metrics)


def _looks_degenerate(event: dict[str, Any]) -> bool:
    """识别历史日志中的退化模型，兼容修复前没有 score_unique_count 的日志。"""
    validate = (event.get("metrics") or {}).get("validate") or {}
    try:
        if int(validate.get("score_unique_count", 2)) <= 1:
            return True
    except (TypeError, ValueError):
        pass
    auc = _metric(event, "validate", "auc")
    ks = _metric(event, "validate", "ks")
    # 修复前的 tie 处理会把“全部预测相同”记录成 AUC=0.5、KS=1。
    return auc is not None and ks is not None and abs(auc - 0.5) < 1e-9 and ks >= 0.99


def _gini_from_auc(value: Any) -> float | None:
    """将 AUC 转换为方向无关的 Gini 系数。"""
    try:
        auc = float(value)
    except (TypeError, ValueError):
        return None
    return 2.0 * auc - 1.0 if math.isfinite(auc) else None


def _read_feature_records(path: Path) -> list[dict[str, Any]]:
    """读取特征筛选结果，兼容旧版字段名和新版指标字段。"""
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError):
        return []


def _load_feature_importance(path: Path | None, feature_names: list[str] | None = None) -> dict[str, float]:
    """从已保存的 LightGBM pickle 读取 gain 重要性；读取失败时返回空结果。"""
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            model = pickle.load(handle)
        booster = getattr(model, "booster_", None)
        if booster is None:
            return {}
        values = booster.feature_importance(importance_type="gain")
        names = list(booster.feature_name())
        # 当前 LightGBM 训练器接收 numpy 矩阵时会生成 Column_0 等通用列名，
        # 此时使用训练事件中持久化的 features 列表恢复真实字段名。
        if feature_names and len(feature_names) == len(values) and all(str(name).startswith("Column_") for name in names):
            names = [str(name) for name in feature_names]
        return {str(name): float(value) for name, value in zip(names, values)}
    except (OSError, EOFError, ImportError, AttributeError, TypeError, ValueError, pickle.PickleError):
        return {}


def _feature_information(path: Path | None, model_path: Path | None, feature_names: list[str] | None = None) -> tuple[list[list[Any]], str]:
    """组合入模变量质量、区分度、稳定性和模型重要性字段。"""
    records = _read_feature_records(path) if path else []
    importance = _load_feature_importance(model_path, feature_names)
    total_gain = sum(value for value in importance.values() if value > 0)
    rows: list[list[Any]] = []
    for item in records:
        feature = item.get("feature", "")
        raw_importance = importance.get(str(feature))
        importance_pct = raw_importance / total_gain if raw_importance is not None and total_gain > 0 else None
        # 旧版文件使用 max_monthly_psi；新版使用 max_psi。
        max_psi = item.get("max_psi") or item.get("max_monthly_psi")
        rows.append(
            [
                feature,
                {"keep": "保留", "review": "待复核", "delete": "删除候选"}.get(item.get("action", item.get("selected", "")), item.get("action", item.get("selected", "—"))),
                item.get("iv", "—"),
                item.get("gini", _gini_from_auc(item.get("auc"))),
                item.get("information_entropy", "—"),
                _pct(item.get("missing_rate")),
                raw_importance if raw_importance is not None else "—",
                _pct(importance_pct) if importance_pct is not None else "—",
                item.get("validate_psi", "—"),
                item.get("oot_psi", "—"),
                max_psi if max_psi is not None else "—",
                item.get("max_correlation", "—"),
                item.get("reason", "—"),
            ]
        )
    rows.sort(key=lambda row: float(row[6]) if isinstance(row[6], (int, float)) else -1.0, reverse=True)
    note = "重要性来自已保存模型的 LightGBM gain；Gini=2×AUC−1；信息熵为 Train 分箱分布的 Shannon entropy（bit）；未生成或无法读取模型时重要性显示为“—”。"
    if not records:
        note = "未找到 feature_selection.csv，变量级指标暂无法生成。"
    elif not importance:
        note += " 当前模型文件不可用，因此重要性列为空。"
    elif total_gain <= 0:
        note += " 当前模型的 LightGBM Gain 全为 0，疑似没有产生有效分裂；请勿直接使用该模型文件并重新训练。"
    return rows, note


def _read_events(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，忽略训练中尚未写完整的最后一行。"""
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("iteration"), (int, float)):
            events.append(value)
    return sorted(events, key=lambda item: int(item.get("iteration", 0)))


def _parameter_summary(event: dict[str, Any]) -> str:
    """压缩展示每轮最影响模型复杂度的参数，避免候选表无法追溯。"""
    parameters = event.get("parameters") or {}
    names = ("learning_rate", "num_leaves", "max_depth", "min_child_samples", "n_estimators", "reg_alpha", "reg_lambda")
    parts = [f"{name}={parameters[name]}" for name in names if name in parameters]
    return "; ".join(parts) or "—"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    """生成无外部依赖的 HTML 表格。"""
    header_html = "".join(f"<th>{_esc(value)}</th>" for value in headers)
    body_html = "".join("<tr>" + "".join(f"<td>{_esc(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{header_html}</tr></thead><tbody>{body_html or '<tr><td colspan=99 class=muted>暂无数据</td></tr>'}</tbody></table></div>"


def _metric_chart_legacy(events: list[dict[str, Any]]) -> str:
    """绘制每轮 Train/Validate/OOT KS 和 AUC 的 SVG 折线。"""
    if not events:
        return "<p class='muted'>暂无迭代指标。</p>"
    width, height, left, top, right, bottom = 860, 330, 58, 30, 18, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for event in events for dataset in ("train", "validate", "oot") for key in ("ks", "auc") if (v := _metric(event, dataset, key)) is not None]
    if not values:
        return "<p class='muted'>暂无可绘制指标。</p>"
    y_min, y_max = max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.05)
    if y_max - y_min < 0.1:
        middle = (y_max + y_min) / 2
        y_min, y_max = max(0.0, middle - 0.05), min(1.0, middle + 0.05)
    y_span = y_max - y_min or 1.0
    x_max = max(int(event.get("iteration", 0)) for event in events) or 1
    colors = {"train_ks": "#2563eb", "validate_ks": "#f59e0b", "oot_ks": "#16a34a", "train_auc": "#60a5fa", "validate_auc": "#fbbf24", "oot_auc": "#4ade80"}
    lines: list[str] = []
    legend: list[str] = []
    for key, color in colors.items():
        dataset, metric_name = key.rsplit("_", 1)
        points: list[str] = []
        for event in events:
            value = _metric(event, dataset, metric_name)
            if value is None:
                continue
            x = left + int(event.get("iteration", 0)) / x_max * plot_w
            y = top + (y_max - value) / y_span * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
            legend.append(f"<span><i style='background:{color}'></i>{_esc(dataset.title() + ' ' + metric_name.upper())}</span>")
    grid = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * plot_h
        value = y_max - ratio * y_span
        grid.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e5e7eb'/><text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end'>{value:.2f}</text>")
    return f"<div class='chart-wrap'><svg viewBox='0 0 {width} {height}' role='img' aria-label='每轮模型指标'><rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#fff' stroke='#cbd5e1'/>{''.join(grid)}<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#64748b'/><line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#64748b'/>{''.join(lines)}<text x='{left + plot_w / 2}' y='{height - 10}' text-anchor='middle'>迭代轮次</text><text x='15' y='{top + plot_h / 2}' transform='rotate(-90 15 {top + plot_h / 2})' text-anchor='middle'>指标值</text></svg><div class='legend'>{''.join(legend)}</div><p class='muted'>纵轴范围：{_num(y_min, 3)} – {_num(y_max, 3)}；坐标按实际指标自动缩放。</p></div>"


def _metric_chart(events: list[dict[str, Any]]) -> str:
    """分别绘制 KS 和 AUC，避免不同量纲的六条线叠在一张图中。"""
    if not events:
        return "<p class='muted'>暂无迭代指标。</p>"
    valid_events = [event for event in events if not _looks_degenerate(event)]
    excluded_count = len(events) - len(valid_events) if valid_events else 0
    plotted_events = valid_events or events
    colors = {"train": "#2563eb", "validate": "#f59e0b", "oot": "#16a34a"}

    def one_chart(metric_name: str, title: str) -> str:
        width, height, left, top, right, bottom = 520, 280, 48, 24, 18, 42
        plot_w, plot_h = width - left - right, height - top - bottom
        x_max = max(int(event.get("iteration", 0)) for event in plotted_events) or 1
        lines: list[str] = []
        legend: list[str] = []
        for dataset, color in colors.items():
            points: list[str] = []
            for event in plotted_events:
                value = _metric(event, dataset, metric_name)
                if value is None:
                    continue
                x = left + int(event.get("iteration", 0)) / x_max * plot_w
                y = top + (1.0 - max(0.0, min(1.0, value))) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
            if points:
                lines.append(f"<polyline points='{ ' '.join(points)}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
                legend.append(f"<span><i style='background:{color}'></i>{_esc(dataset.title())}</span>")
        grid = "".join(
            f"<line x1='{left}' y1='{top + index / 4 * plot_h:.1f}' x2='{left + plot_w}' y2='{top + index / 4 * plot_h:.1f}' stroke='#e5e7eb'/><text x='{left - 7}' y='{top + index / 4 * plot_h + 4:.1f}' text-anchor='end'>{1 - index / 4:.1f}</text>"
            for index in range(5)
        )
        return f"<div class='chart-wrap'><h3>{title}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{title}'><rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#fff' stroke='#cbd5e1'/>{grid}<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#64748b'/><line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#64748b'/>{''.join(lines)}<text x='{left + plot_w / 2}' y='{height - 8}' text-anchor='middle'>迭代轮次</text><text x='14' y='{top + plot_h / 2}' transform='rotate(-90 14 {top + plot_h / 2})' text-anchor='middle'>{metric_name.upper()}</text></svg><div class='legend'>{''.join(legend)}</div><p class='muted'>纵轴固定为 0–1，便于跨轮次比较。</p></div>"

    note = f"<p class='muted'>已隐藏 {excluded_count} 个预测分数退化的 trial（AUC≈0.5 且 KS=1 的历史记录不作为有效排序结果）。</p>" if excluded_count else ""
    return "<div class='grid'>" + one_chart("ks", "每轮 KS") + one_chart("auc", "每轮 AUC") + "</div>" + note


def _fit_chart(event: dict[str, Any]) -> str:
    """绘制最佳轮次的 binary_logloss 拟合曲线。"""
    fit = event.get("fit_history") or {}
    series: list[tuple[str, list[float], str]] = []
    for dataset, color in (("train", "#2563eb"), ("validate", "#f59e0b"), ("oot", "#16a34a")):
        values = fit.get(dataset) or {}
        if not isinstance(values, dict):
            continue
        key = next((name for name, value in values.items() if isinstance(value, list) and value), None)
        if key:
            series.append((f"{dataset.title()} {key}", [float(v) for v in values[key] if isinstance(v, (int, float))], color))
    if not series:
        return "<p class='muted'>最佳轮次没有 fit_history。</p>"
    all_values = [value for _, values, _ in series for value in values]
    y_min, y_max = min(all_values), max(all_values)
    pad = (y_max - y_min) * 0.12 or max(abs(y_min) * 0.01, 0.001)
    y_min, y_max = y_min - pad, y_max + pad
    width, height, left, top, right, bottom = 860, 280, 58, 25, 18, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    y_span = y_max - y_min or 1.0
    x_max = max(len(values) for _, values, _ in series) - 1 or 1
    paths = []
    legend = []
    for name, values, color in series:
        points = " ".join(f"{left + i / x_max * plot_w:.1f},{top + (y_max - value) / y_span * plot_h:.1f}" for i, value in enumerate(values))
        paths.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
        legend.append(f"<span><i style='background:{color}'></i>{_esc(name)}</span>")
    grid = "".join(f"<line x1='{left}' y1='{top + i / 4 * plot_h:.1f}' x2='{left + plot_w}' y2='{top + i / 4 * plot_h:.1f}' stroke='#e5e7eb'/><text x='{left - 8}' y='{top + i / 4 * plot_h + 4:.1f}' text-anchor='end'>{_num(y_max - i / 4 * y_span, 4)}</text>" for i in range(5))
    return f"<div class='chart-wrap'><svg viewBox='0 0 {width} {height}' role='img' aria-label='最佳轮次拟合曲线'><rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#fff' stroke='#cbd5e1'/>{grid}<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#64748b'/><line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#64748b'/>{''.join(paths)}<text x='{left + plot_w / 2}' y='{height - 8}' text-anchor='middle'>Boosting 迭代</text><text x='15' y='{top + plot_h / 2}' transform='rotate(-90 15 {top + plot_h / 2})' text-anchor='middle'>损失值</text></svg><div class='legend'>{''.join(legend)}</div><p class='muted'>最佳轮次拟合指标范围：{_num(y_min, 5)} – {_num(y_max, 5)}。</p></div>"


def _classification_curve_chart(event: dict[str, Any]) -> str:
    """绘制最佳轮次 Validate/OOT ROC 和 PR 曲线。"""
    colors = {"validate": "#f59e0b", "oot": "#16a34a"}
    blocks: list[str] = []
    for curve_name, x_key, y_key, title, x_label, y_label in (
        ("roc_curve", "fpr", "tpr", "ROC 曲线", "FPR", "TPR"),
        ("pr_curve", "recall", "precision", "PR 曲线", "Recall", "Precision"),
    ):
        width, height, left, top, right, bottom = 430, 270, 48, 24, 18, 42
        plot_w, plot_h = width - left - right, height - top - bottom
        paths: list[str] = []
        legend: list[str] = []
        for dataset, color in colors.items():
            points = (event.get("metrics") or {}).get(dataset, {}).get(curve_name) or []
            if not points:
                continue
            coords = " ".join(
                f"{left + max(0.0, min(1.0, float(point.get(x_key, 0.0)))) * plot_w:.1f},{top + (1.0 - max(0.0, min(1.0, float(point.get(y_key, 0.0))))) * plot_h:.1f}"
                for point in points
            )
            paths.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
            legend.append(f"<span><i style='background:{color}'></i>{dataset.title()}</span>")
        if not paths:
            blocks.append(f"<div class='chart-wrap'><h3>{title}</h3><p class='muted'>暂无曲线数据，请使用新训练日志生成报告。</p></div>")
            continue
        grid = "".join(
            f"<line x1='{left}' y1='{top + index / 4 * plot_h:.1f}' x2='{left + plot_w}' y2='{top + index / 4 * plot_h:.1f}' stroke='#e5e7eb'/>"
            f"<text x='{left - 7}' y='{top + index / 4 * plot_h + 4:.1f}' text-anchor='end'>{1 - index / 4:.1f}</text>"
            for index in range(5)
        )
        diagonal = f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top}' stroke='#cbd5e1' stroke-dasharray='4 4'/>" if curve_name == "roc_curve" else ""
        blocks.append(
            f"<div class='chart-wrap'><h3>{title}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{title}'><rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#fff' stroke='#cbd5e1'/>{grid}{diagonal}{''.join(paths)}<text x='{left + plot_w / 2}' y='{height - 8}' text-anchor='middle'>{x_label}</text><text x='14' y='{top + plot_h / 2}' transform='rotate(-90 14 {top + plot_h / 2})' text-anchor='middle'>{y_label}</text></svg><div class='legend'>{''.join(legend)}</div></div>"
        )
    return "<div class='grid'>" + "".join(blocks) + "</div>"


def _monthly_rows(event: dict[str, Any]) -> list[list[Any]]:
    """合并 Validate/OOT 月度效果，便于快速定位时间退化月份。"""
    result: list[list[Any]] = []
    for dataset in ("validate", "oot"):
        rows = ((event.get("metrics") or {}).get(dataset) or {}).get("monthly") or []
        for row in rows:
            result.append([dataset.title(), row.get("month", "—"), row.get("sample_count", "—"), row.get("bad_count", "—"), _pct(row.get("bad_rate")), _num(row.get("auc")), _num(row.get("ks")), _num(row.get("pr_auc"))])
    return result


def _feature_rows(path: Path) -> list[list[Any]]:
    """读取特征筛选 CSV 的前 30 行，用于报告附录。"""
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for item in reader:
                rows.append([item.get(key, "") for key in ("feature", "selected", "iv", "ks", "max_monthly_psi", "missing_rate", "reason")])
                if len(rows) >= 30:
                    break
            return rows
    except (OSError, UnicodeError):
        return []


def write_model_report(*, iteration_log: str | Path, output_path: str | Path, run_meta_path: str | Path | None = None, split_manifest_path: str | Path | None = None, feature_selection_path: str | Path | None = None, model_path: str | Path | None = None) -> Path:
    """根据训练日志生成模型报告并返回 HTML 路径。"""
    log_path = Path(iteration_log).expanduser().resolve()
    events = _read_events(log_path)
    scored = [event for event in events if _metric(event, "validate", "ks") is not None]
    # 优先在非退化轮次中选最佳；如果整份日志都退化，仍展示最后一轮并明确提示。
    valid_scored = [event for event in scored if not _looks_degenerate(event)]
    rejected_statuses = {"skipped", "failed", "pruned", "rejected_bounds", "rejected_metric", "rejected_degenerate", "stopped_timeout"}
    selectable = [event for event in (valid_scored or scored) if str(event.get("status", "complete")) not in rejected_statuses]
    quality_scored = [event for event in (selectable or valid_scored or scored) if _event_quality_score(event) is not None]
    if quality_scored:
        best = max(quality_scored, key=lambda event: _event_quality_score(event) or float("-inf"))
    else:
        # 兼容旧版没有三分区 KS 的日志；新日志一定优先使用综合质量分。
        best = max(valid_scored or scored, key=lambda event: _metric(event, "validate", "ks") or float("-inf")) if scored else (events[-1] if events else {})
    best_quality = _event_quality_score(best)
    train_ks, validate_ks, oot_ks = (_metric(best, dataset, "ks") for dataset in ("train", "validate", "oot"))
    accepted = sum(1 for event in events if event.get("accepted") is True and not _looks_degenerate(event))
    train_validate_gap = train_ks - validate_ks if train_ks is not None and validate_ks is not None else None
    validate_oot_gap = validate_ks - oot_ks if validate_ks is not None and oot_ks is not None else None
    findings = []
    if train_validate_gap is not None and train_validate_gap > 0.10:
        findings.append(f"训练/验证 KS 差距 {_num(train_validate_gap)}，存在过拟合风险。")
    if validate_oot_gap is not None and validate_oot_gap > 0.10:
        findings.append(f"Validate/OOT KS 差距 {_num(validate_oot_gap)}，需要关注时间稳定性。")
    if oot_ks is not None and oot_ks < 0.20:
        findings.append("OOT KS 低于 0.20，建议复核时间窗口、样本量和特征稳定性。")
    if not findings:
        findings.append("未发现超过当前规则阈值的明显风险，仍需结合业务审批标准复核。")
    if _looks_degenerate(best):
        findings.insert(0, "当前最佳候选为退化模型：Validate 预测分数几乎没有变化，历史 KS=1 不可信，建议修复评估后重新运行 Bayesian。")
    split_info = {}
    if split_manifest_path and Path(split_manifest_path).is_file():
        try:
            split_info = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            split_info = {}
    split_rows = [[name.title(), (split_info.get(f"{name}_months") or []), ((split_info.get("datasets") or {}).get(name) or {}).get("sample_count", "—"), _pct(((split_info.get("datasets") or {}).get(name) or {}).get("bad_rate"))] for name in ("train", "validate", "oot")]
    param_rows = [[key, value] for key, value in (best.get("parameters") or {}).items() if key != "features"]
    candidate_rows = [[event.get("iteration"), "退化（已排除）" if _looks_degenerate(event) else event.get("status", "complete"), _num(_event_quality_score(event)), _parameter_summary(event), _num(_metric(event, "train", "ks")), _num(_metric(event, "validate", "ks")), _num(_metric(event, "oot", "ks")), _num(_metric(event, "train", "auc")), _num(_metric(event, "validate", "auc")), _num(_metric(event, "oot", "auc")), _num(_metric(event, "train", "pr_auc")), _num(_metric(event, "validate", "pr_auc")), _num(_metric(event, "oot", "pr_auc")), "是" if event.get("accepted") and not _looks_degenerate(event) else "否", (str(event.get("reason", "")) + ("；历史评估显示为 AUC=0.5/KS=1，已按退化模型处理" if _looks_degenerate(event) else ""))] for event in events]
    feature_rows = _feature_rows(Path(feature_selection_path).expanduser().resolve()) if feature_selection_path else []
    meta = {}
    if run_meta_path and Path(run_meta_path).is_file():
        try:
            meta = json.loads(Path(run_meta_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    model_file_path = Path(model_path).expanduser().resolve() if model_path and Path(model_path).is_file() else None
    model_file = str(model_file_path) if model_file_path else "未生成"
    feature_rows, feature_note = _feature_information(
        Path(feature_selection_path).expanduser().resolve() if feature_selection_path else None,
        model_file_path,
        [str(value) for value in (best.get("parameters") or {}).get("features", [])],
    )
    model_importance = _load_feature_importance(model_file_path, [str(value) for value in (best.get("parameters") or {}).get("features", [])])
    model_gain_degenerate = bool(model_file_path and model_importance and not any(value > 0 for value in model_importance.values()))
    if model_gain_degenerate:
        findings.insert(0, "已保存模型的 LightGBM Gain 全为 0，模型文件与有效训练结果不一致，建议重新保存模型。")
    effect_rows = []
    for name in ("train", "validate", "oot"):
        metric_values = (best.get("metrics") or {}).get(name) or {}
        auc = metric_values.get("auc")
        effect_rows.append(
            [
                name.title(),
                metric_values.get("sample_count", "—"),
                _pct(metric_values.get("bad_rate")),
                _num(auc),
                _num(metric_values.get("gini", _gini_from_auc(auc))),
                _num(metric_values.get("ks")),
                _num(metric_values.get("pr_auc")),
                _num(metric_values.get("score_psi_vs_train")),
            ]
        )
    stability_rows = [
        ["Train - Validate KS", _num(train_validate_gap), "差距越大，过拟合风险越高"],
        ["Validate - OOT KS", _num(validate_oot_gap), "差距越大，时间稳定性越弱"],
        ["Train - Validate AUC", _num((_metric(best, "train", "auc") or 0.0) - (_metric(best, "validate", "auc") or 0.0)) if _metric(best, "train", "auc") is not None and _metric(best, "validate", "auc") is not None else "—", "辅助判断泛化差距"],
        ["Validate - OOT AUC", _num((_metric(best, "validate", "auc") or 0.0) - (_metric(best, "oot", "auc") or 0.0)) if _metric(best, "validate", "auc") is not None and _metric(best, "oot", "auc") is not None else "—", "辅助判断时间退化"],
    ]
    feature_psi_values = []
    for item in _read_feature_records(Path(feature_selection_path).expanduser().resolve()) if feature_selection_path else []:
        raw = item.get("max_psi") or item.get("max_monthly_psi")
        try:
            feature_psi_values.append((str(item.get("feature", "")), float(raw)))
        except (TypeError, ValueError):
            continue
    stability_rows.extend(
        [
            ["Validate Score PSI", _num(_metric(best, "validate", "score_psi_vs_train")), "相对 Train 的模型分数分布漂移"],
            ["OOT Score PSI", _num(_metric(best, "oot", "score_psi_vs_train")), "相对 Train 的模型分数分布漂移"],
            ["Validate 分数唯一值数", ((best.get("metrics") or {}).get("validate") or {}).get("score_unique_count", "—"), "只有 1 个值表示模型没有产生有效排序"],
            ["OOT 分数唯一值数", ((best.get("metrics") or {}).get("oot") or {}).get("score_unique_count", "—"), "只有 1 个值表示模型没有产生有效排序"],
            ["特征 PSI≥0.25 数量", sum(value >= 0.25 for _, value in feature_psi_values), "建议复核时间口径和变量漂移"],
            ["特征最大 PSI", _num(max((value for _, value in feature_psi_values), default=None)), "PSI 越高，跨时间分布越不稳定"],
        ]
    )
    quality = model_quality_components(best.get("metrics") or {})
    stability_rows.extend(
        [
            ["模型质量得分", _num(best_quality), "综合 Train/Validate/OOT KS、分区差距和 OOT 分数 PSI，越高越好"],
            ["质量分加权 KS", _num(quality.get("weighted_ks")), "OOT 权重 45%，Validate 权重 30%，Train 权重 25%"],
            ["质量分 Train-Validate gap", _num(quality.get("train_validate_gap")), "绝对差距越小越稳健"],
            ["质量分 Validate-OOT gap", _num(quality.get("validate_oot_gap")), "绝对差距越小越接近未来样本表现"],
        ]
    )
    # 生成一张可直接用于审批的“基线 vs 最佳轮次”对比表。
    baseline = events[0] if events else {}
    comparison_rows: list[list[Any]] = []
    for dataset in ("train", "validate", "oot"):
        for metric_name, label in (("ks", "KS"), ("auc", "AUC"), ("pr_auc", "PR-AUC"), ("score_psi_vs_train", "Score PSI")):
            baseline_value = _metric(baseline, dataset, metric_name)
            best_value = _metric(best, dataset, metric_name)
            delta = best_value - baseline_value if baseline_value is not None and best_value is not None else None
            comparison_rows.append([dataset.title(), label, _num(baseline_value), _num(best_value), _num(delta)])

    # 结论卡片使用和报告诊断一致的规则，明确区分“通过 / 需复核 / 不建议直接使用”。
    if not events:
        model_decision = "未完成"
        model_decision_reason = "没有可用的训练轮次记录。"
    elif _looks_degenerate(best) or model_gain_degenerate:
        model_decision = "不建议直接使用"
        model_decision_reason = "最佳候选或已保存模型没有有效分裂，必须重新训练。"
    elif train_validate_gap is not None and train_validate_gap > 0.10:
        model_decision = "需复核"
        model_decision_reason = f"Train/Validate KS gap={_num(train_validate_gap)}，超过 0.10。"
    elif validate_oot_gap is not None and validate_oot_gap > 0.10:
        model_decision = "需复核"
        model_decision_reason = f"Validate/OOT KS gap={_num(validate_oot_gap)}，超过 0.10。"
    elif oot_ks is not None and oot_ks < 0.20:
        model_decision = "不建议直接使用"
        model_decision_reason = f"OOT KS={_num(oot_ks)}，低于 0.20。"
    elif _metric(best, "oot", "score_psi_vs_train") is not None and (_metric(best, "oot", "score_psi_vs_train") or 0.0) > 0.25:
        model_decision = "需复核"
        model_decision_reason = f"OOT Score PSI={_num(_metric(best, 'oot', 'score_psi_vs_train'))}，超过 0.25。"
    else:
        model_decision = "通过"
        model_decision_reason = "未触发当前报告的主要风险阈值；仍需结合业务审批线复核。"
    summary_banner = (
        f"<div class='summary-banner'><strong>模型结论：{_esc(model_decision)}</strong>"
        f"<span>　最佳轮次 {_esc(best.get('iteration', '—'))} · 综合质量分 {_num(best_quality)} · Train/Validate/OOT KS {_num(train_ks)} / {_num(validate_ks)} / {_num(oot_ks)} · { _esc(model_decision_reason) }</span>"
        "<p class='muted'>模型选择优先最大化综合质量分：OOT KS 权重最高，同时惩罚 Train-Validate 和 Validate-OOT 差距；OOT KS < 0.20、KS gap > 0.10 或 OOT Score PSI > 0.25 时仍需复核。</p></div>"
    )
    curve_chart = _classification_curve_chart(best)
    monthly_rows = _monthly_rows(best)
    html_report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>风控模型报告</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#203040;background:#f5f7fa;margin:0;padding-top:64px}}main{{max-width:1320px;margin:0 auto;padding:28px}}h1,h2,h3{{color:#17365d}}h1{{margin:0 0 8px}}h2{{border-bottom:2px solid #d9e2ea;padding-bottom:8px;margin-top:32px;scroll-margin-top:76px}}.subtitle,.muted{{color:#66788a;font-size:13px;line-height:1.6}}.report-nav{{position:fixed;top:0;left:0;right:0;z-index:20;display:flex;gap:4px;align-items:center;overflow-x:auto;padding:10px 28px;background:rgba(255,255,255,.97);border-bottom:1px solid #d9e2ea;box-shadow:0 2px 8px #00000012}}.report-nav a{{color:#334155;text-decoration:none;white-space:nowrap;padding:7px 11px;border-radius:7px;font-size:12px}}.report-nav a:hover{{background:#eaf2fb;color:#1d4ed8}}.summary-banner{{background:#fff;border:1px solid #d9e2ea;border-left:5px solid #2563eb;border-radius:10px;padding:14px 16px;margin:16px 0}}.summary-banner strong{{font-size:18px;color:#17365d}}.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.card{{background:#fff;border:1px solid #d9e2ea;padding:14px 20px;min-width:145px;box-shadow:0 1px 2px #0000000d}}.card small{{display:block;color:#66788a}}.card strong{{display:block;font-size:25px;color:#17365d;margin-top:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}.chart-wrap{{background:#fff;border:1px solid #d9e2ea;padding:12px;overflow:auto}}svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:#334155}}.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#475569;margin-top:5px}}.legend i{{display:inline-block;width:13px;height:3px;margin-right:5px;vertical-align:middle}}.legend-blue{{color:#2563eb;margin-right:10px}}.legend-red{{color:#dc2626;margin-right:10px}}.score-row{{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:11px}}.score-row>span{{width:52px;color:#475569}}.score-bar{{display:flex;flex:1;height:14px;background:#eef2f7;border-radius:5px;overflow:hidden}}.score-bar i,.score-bar b{{display:block;height:100%}}.score-row em{{width:92px;text-align:right;color:#64748b;font-style:normal}}.table-wrap{{background:#fff;border:1px solid #d9e2ea;overflow:auto;max-height:520px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;white-space:nowrap;vertical-align:top}}th{{background:#eef5fb;color:#17365d;position:sticky;top:0}}.finding{{padding:10px 13px;background:#fff8e6;border-left:4px solid #f59e0b;margin:7px 0}}code{{background:#eef2f7;padding:2px 4px;border-radius:3px}}@media(max-width:680px){{main{{padding:14px}}.report-nav{{padding:9px 12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>风控模型报告</h1><div class='subtitle'>调参方法：{_esc(meta.get('method', best.get('method', '—')))} · 监控日志：<code>{_esc(log_path)}</code> · 报告基于已完成的真实训练轮次生成。</div>{summary_banner}<div class='cards'><div class='card'><small>已记录轮次</small><strong>{len(events)}</strong></div><div class='card'><small>最佳轮次</small><strong>{_esc(best.get('iteration', '—'))}</strong></div><div class='card'><small>最佳模型 Validate KS</small><strong>{_num(validate_ks)}</strong></div><div class='card'><small>最佳模型 OOT KS</small><strong>{_num(oot_ks)}</strong></div><div class='card'><small>接受轮次</small><strong>{accepted}</strong></div></div>
<h2>1. 模型基本信息与最佳参数</h2>{_table(['项目','值'], [['模型','LightGBM 二分类'],['调参方法',meta.get('method', best.get('method','—'))],['模型文件',model_file],['训练状态',meta.get('status','—')],['最佳轮次',best.get('iteration','—')],['综合质量得分',_num(best_quality)]])}<h3>最佳参数</h3>{_table(['参数','值'],param_rows)}
<h2>2. 模型综合效果</h2><h3>最佳模型拟合曲线</h3>{_fit_chart(best)}<h3>最佳模型 ROC / PR 曲线</h3>{curve_chart}{_table(['数据集','样本数','坏账率','AUC','Gini','KS','PR-AUC','Score PSI'], effect_rows)}<p class='muted'>本节只展示综合质量分最高且通过退化检查的模型。Gini = 2 × AUC − 1；PR-AUC 使用 Average Precision 计算，适合坏样本比例较低的场景；Score PSI 为当前分数分布相对 Train 分布的 PSI。</p>
<h2>3. 训练轮次与参数候选对比</h2><h3>Baseline 与最佳轮次对比</h3>{_table(['数据集','指标','Baseline','最佳轮次','变化'],comparison_rows)}<p class='muted'>KS、AUC、PR-AUC 增加表示效果改善；Score PSI 下降表示分布更稳定。综合质量得分用于最终模型选择，不是单看 Validate KS。</p><h3>每轮参数与最终效果</h3>{_table(['轮次','状态','综合质量分','关键参数','Train KS','Validate KS','OOT KS','Train AUC','Validate AUC','OOT AUC','Train PR-AUC','Validate PR-AUC','OOT PR-AUC','采用','原因'],candidate_rows)}
<h2>4. 样本切分与模型稳定性</h2>{_table(['分区','月份','样本数','坏账率'],split_rows)}<div class='grid'><div><h3>模型诊断</h3>{''.join(f"<div class='finding'>{_esc(item)}</div>" for item in findings)}</div><div><h3>稳定性指标</h3>{_table(['指标','结果','解释'],stability_rows)}</div></div><h3>月度稳定性</h3>{_table(['数据集','月份','样本数','坏样本数','坏账率','AUC','KS','PR-AUC'],monthly_rows)}
<h2>5. 入模变量信息</h2><p class='muted'>{_esc(feature_note)}</p>{_table(['字段','状态','IV','Gini','信息熵','缺失率','重要性（Gain）','重要性占比','Validate PSI','OOT PSI','最大 PSI','最大相关系数','处理原因'],feature_rows)}
</main></body></html>"""
    # 通过统一替换注入固定导航，避免报告内容变长时导航和章节标题脱节。
    navigation = "<nav class='report-nav'><a href='#section-overview'>基本信息</a><a href='#section-effect'>综合效果</a><a href='#section-candidates'>训练对比</a><a href='#section-stability'>稳定性</a><a href='#section-features'>入模变量</a></nav>"
    html_report = html_report.replace("<body><main>", "<body>" + navigation + "<main>", 1)
    html_report = html_report.replace("<h2>1. 模型基本信息与最佳参数</h2>", "<h2 id='section-overview'>1. 模型基本信息与最佳参数</h2>", 1)
    html_report = html_report.replace("<h2>2. 模型综合效果</h2>", "<h2 id='section-effect'>2. 模型综合效果</h2>", 1)
    html_report = html_report.replace("<h2>3. 训练轮次与参数候选对比</h2>", "<h2 id='section-candidates'>3. 训练轮次与参数候选对比</h2>", 1)
    html_report = html_report.replace("<h2>4. 样本切分与模型稳定性</h2>", "<h2 id='section-stability'>4. 样本切分与模型稳定性</h2>", 1)
    html_report = html_report.replace("<h2>5. 入模变量信息</h2>", "<h2 id='section-features'>5. 入模变量信息</h2>", 1)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_report, encoding="utf-8")
    return destination


__all__ = ["write_model_report"]

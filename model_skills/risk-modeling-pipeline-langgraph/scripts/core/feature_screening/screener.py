"""基于 Train 指标和跨时间稳定性的特征筛选。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import toad
import yaml

from core.data_read.contract import DataContract, feature_columns
from core.data_read.loader import load_table
from core.eda_analysis.metrics import _bin_detail, _is_numeric
from core.sample_split.splitter import _month_key


def load_feature_config(path: str | Path) -> dict[str, Any]:
    """读取特征筛选与预处理配置。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(raw)


def _clean_values(frame: pl.DataFrame, feature: str, dtype: pl.DataType) -> list[Any]:
    """提取字段值并把 NaN 统一为缺失。"""
    values = frame.get_column(feature).to_list()
    if _is_numeric(dtype):
        return [None if value is None or (isinstance(value, float) and not math.isfinite(value)) else value for value in values]
    return values


def _missing_rate(frame: pl.DataFrame, feature: str, dtype: pl.DataType) -> float:
    """计算字段缺失率，数值 NaN 也计为缺失。"""
    column = pl.col(feature)
    expression = column.is_null() | column.is_nan() if _is_numeric(dtype) else column.is_null()
    return float(frame.select(expression.mean()).item() or 0.0)


def _psi(train_values: list[Any], current_values: list[Any], dtype: pl.DataType, bins: int = 10) -> float:
    """使用 toad.metrics.PSI 比较 Train 与其他分区的字段分布。"""
    if not train_values or not current_values:
        return 0.0
    if _is_numeric(dtype):
        train_numeric = np.asarray([float(value) for value in train_values if value is not None and math.isfinite(float(value))], dtype=float)
        current_numeric = np.asarray([float(value) for value in current_values if value is not None and math.isfinite(float(value))], dtype=float)
        if not len(train_numeric) or not len(current_numeric):
            return 0.0
        cuts = np.unique(np.nanquantile(train_numeric, np.linspace(0, 1, bins + 1)[1:-1]))
        train_bucket = np.digitize(train_numeric, cuts, right=True)
        current_bucket = np.digitize(current_numeric, cuts, right=True)
    else:
        categories = {str(value) for value in train_values if value is not None}
        categories = set(sorted(categories, key=lambda value: str(value))[: max(1, bins - 1)])
        mapping = {value: index for index, value in enumerate(categories)}
        other = len(mapping)
        train_bucket = np.asarray([mapping.get(str(value), other) for value in train_values if value is not None], dtype=int)
        current_bucket = np.asarray([mapping.get(str(value), other) for value in current_values if value is not None], dtype=int)
    if not len(train_bucket) or not len(current_bucket):
        return 0.0
    try:
        return float(toad.metrics.PSI(current_bucket, train_bucket))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _correlation_max(train: pl.DataFrame, features: list[str], schema: dict[str, pl.DataType]) -> dict[str, float]:
    """计算数值特征与其他数值特征的最大绝对相关系数。"""
    numeric = [feature for feature in features if _is_numeric(schema[feature])]
    if len(numeric) < 2:
        return {feature: 0.0 for feature in features}
    filled = train.select([pl.col(feature).fill_nan(None).fill_null(pl.col(feature).median()).alias(feature) for feature in numeric])
    corr = filled.corr()
    result = {feature: 0.0 for feature in features}
    for index, feature in enumerate(numeric):
        values = [float(value) for value in corr.row(index) if value is not None and math.isfinite(float(value))]
        result[feature] = max((abs(value) for value in values if abs(value) < 0.999999), default=0.0)
    return result


def _entropy_from_bins(bin_rows: list[dict[str, Any]]) -> float | None:
    """根据 Train 分箱分布计算信息熵（以 bit 为单位）。"""
    counts = [float(row.get("sample_count") or 0) for row in bin_rows]
    total = sum(counts)
    if total <= 0:
        return None
    probabilities = [count / total for count in counts if count > 0]
    return float(-sum(probability * math.log(probability, 2) for probability in probabilities))


def run_feature_screening(*, split_dir: str | Path, output_dir: str | Path, contract: DataContract, data_path: str | Path | None = None, source_frame: pl.DataFrame | None = None, config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    """计算字段质量、IV/KS、PSI、相关性并生成筛选后的建模数据。"""
    settings = {
        "quality": {"missing_rate_delete": 0.80, "constant_delete": True},
        "univariate": {"iv_min": 0.02, "ks_min": 0.05, "bin_count": 10},
        "stability": {"psi_warning": 0.10, "psi_delete": 0.25},
        "correlation": {"abs_threshold": 0.70},
    }
    for section, values in (config or {}).items():
        if isinstance(values, dict) and isinstance(settings.get(section), dict):
            settings[section].update(values)
        else:
            settings[section] = values
    split = Path(split_dir).expanduser().resolve()
    manifest = json.loads((split / "split_manifest.json").read_text(encoding="utf-8"))
    source_path = Path(data_path or manifest.get("data_path", "")).expanduser().resolve()
    if source_frame is None and not source_path.is_file():
        raise FileNotFoundError(f"找不到切分对应的原始数据：{source_path}")
    source = source_frame if source_frame is not None else load_table(source_path).data
    if not contract.date_col or contract.date_col not in source.columns:
        raise ValueError("特征筛选需要 data-read 确认的日期字段")
    months = [_month_key(value) for value in source.get_column(contract.date_col).to_list()]
    train = source.filter(pl.Series("_split_month", [month in manifest.get("train_months", []) for month in months]))
    validate = source.filter(pl.Series("_split_month", [month in manifest.get("validate_months", []) for month in months]))
    oot = source.filter(pl.Series("_split_month", [month in manifest.get("oot_months", []) for month in months]))
    schema = {name: dtype for name, dtype in zip(train.columns, train.dtypes)}
    features = feature_columns(train, contract)
    target = [int(value == contract.bad_label) for value in train.get_column(contract.target_col).to_list()] if contract.target_col and contract.target_col in train.columns else [0] * train.height
    rows: list[dict[str, Any]] = []
    for feature in features:
        dtype = schema[feature]
        train_values = _clean_values(train, feature, dtype)
        try:
            bin_rows, metrics = _bin_detail(feature, train_values, target, dtype, count=int(settings["univariate"].get("bin_count", 10)), max_categories=20)
        except (TypeError, ValueError, IndexError):
            bin_rows = []
            metrics = {"iv": None, "ks": None, "auc": None}
        val_values = _clean_values(validate, feature, dtype)
        oot_values = _clean_values(oot, feature, dtype)
        missing = _missing_rate(train, feature, dtype)
        iv = metrics.get("iv")
        ks = metrics.get("ks")
        auc = metrics.get("auc")
        rows.append({"feature": feature, "missing_rate": missing, "iv": iv, "ks": ks, "auc": auc, "gini": (2.0 * float(auc) - 1.0) if auc is not None else None, "information_entropy": _entropy_from_bins(bin_rows), "validate_psi": _psi(train_values, val_values, dtype), "oot_psi": _psi(train_values, oot_values, dtype), "constant": train.select(pl.col(feature).n_unique()).item() <= 1})
    correlation_max = _correlation_max(train, features, schema)
    for row in rows:
        row["max_psi"] = max(float(row["validate_psi"]), float(row["oot_psi"]))
        row["max_correlation"] = correlation_max.get(row["feature"], 0.0)
        reasons: list[str] = []
        action = "keep"
        if settings["quality"].get("constant_delete", True) and row["constant"]:
            action, reasons = "delete", ["常量字段"]
        elif not _is_numeric(schema[row["feature"]]) and settings.get("preprocessing", {}).get("categorical", {}).get("method") == "drop":
            action, reasons = "delete", ["按配置删除字符型变量"]
        elif row["missing_rate"] >= float(settings["quality"].get("missing_rate_delete", 0.80)):
            action, reasons = "delete", ["Train 缺失率超过阈值"]
        elif row["iv"] is not None and float(row["iv"]) < float(settings["univariate"].get("iv_min", 0.02)):
            action, reasons = "delete", ["IV 低于阈值"]
        if row["max_psi"] >= float(settings["stability"].get("psi_delete", 0.25)):
            action = "review" if action == "keep" else action
            reasons.append("Validate/OOT PSI 超过阈值")
        elif row["max_psi"] >= float(settings["stability"].get("psi_warning", 0.10)):
            reasons.append("Validate/OOT PSI 需要关注")
            if action == "keep":
                action = "review"
        if row["max_correlation"] >= float(settings["correlation"].get("abs_threshold", 0.70)):
            reasons.append("与其他数值特征高度相关")
            if action == "keep":
                action = "review"
        row["action"] = action
        row["reason"] = "；".join(reasons) or "通过质量、区分度、稳定性和相关性检查"
    # 对高相关字段按 Train IV 择优，将较弱字段标记为复核，不直接删除。
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    statistics_frame = pl.DataFrame(rows)
    selection_frame = statistics_frame.select(["feature", "missing_rate", "iv", "ks", "auc", "gini", "information_entropy", "validate_psi", "oot_psi", "max_psi", "max_correlation", "action", "reason"])
    selection_frame.write_csv(output / "feature_selection.csv", float_precision=5)
    kept = [str(row["feature"]) for row in rows if row["action"] != "delete"]
    reserved = [name for name in (*contract.id_cols, contract.date_col, contract.target_col) if name and name in train.columns]
    processed_columns = list(dict.fromkeys(reserved + kept))
    summary_counts = {action: sum(row["action"] == action for row in rows) for action in ("keep", "review", "delete")}
    action_labels = {"keep": "保留", "review": "待复核", "delete": "删除候选"}
    action_order = {"delete": 0, "review": 1, "keep": 2}
    display_limit = max(1, int(settings.get("summary_top_n", 10)))
    # 对话只展示最需要人工关注的字段，完整字段级结果仍写入 feature_selection.csv。
    summary_rows = sorted(
        rows,
        key=lambda row: (
            action_order.get(str(row["action"]), 9),
            -float(row.get("max_psi") or 0.0),
            -float(row.get("max_correlation") or 0.0),
        ),
    )
    lines = [
        "## 特征筛选反馈",
        "",
        f"共分析 **{len(features)}** 个特征：保留 **{summary_counts['keep']}** 个，待复核 **{summary_counts['review']}** 个，删除候选 **{summary_counts['delete']}** 个。",
        "",
        "| 筛选状态 | 数量 | 处理原则 |",
        "|---|---:|---|",
        f"| 保留 | {summary_counts['keep']} | 当前指标未触发删除/复核规则 |",
        f"| 待复核 | {summary_counts['review']} | 结合 PSI、相关性和业务解释性确认，不自动删除 |",
        f"| 删除候选 | {summary_counts['delete']} | 由用户确认后才从建模字段中移除 |",
        "",
        f"### 重点字段摘要（最多展示 {min(display_limit, len(summary_rows))} 个）",
        "",
        "| 特征 | 状态 | 缺失率 | IV | KS | Validate PSI | OOT PSI | 最大相关系数 | 主要原因 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows[:display_limit]:
        fmt = lambda value: "—" if value is None else f"{float(value):.5f}"
        lines.append(f"| {row['feature']} | {action_labels.get(row['action'], row['action'])} | {float(row['missing_rate']):.5%} | {fmt(row['iv'])} | {fmt(row['ks'])} | {fmt(row['validate_psi'])} | {fmt(row['oot_psi'])} | {fmt(row['max_correlation'])} | {row['reason']} |")
    if len(summary_rows) > display_limit:
        lines.extend(["", f"其余 {len(summary_rows) - display_limit} 个字段的完整指标见 `feature_selection.csv`。"])
    lines.extend(
        [
            "",
            "### 调整建议",
            "- 优先处理“删除候选”字段，并由用户确认后再移除。",
            "- 对“待复核”字段，优先核对 OOT PSI、高相关字段对和业务可解释性；默认保留，避免仅凭单一指标误删。",
            "- 样本切分中的 Excluded 月份只是不进入建模，不会修改原始数据。",
        ]
    )
    summary_path = output / "feature_selection_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"node_id": "feature-screening", "config_path": str(config_path) if config_path else None, "source_data": str(source_path), "selection": str(output / "feature_selection.csv"), "summary": str(summary_path), "selected_columns": processed_columns, "counts": summary_counts}
    (output / "feature_processing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "success", "node_id": "feature-screening", "summary": "\n".join(lines), "summary_markdown": str(summary_path), "selection": str(output / "feature_selection.csv"), "manifest": str(output / "feature_processing_manifest.json"), "selected_columns": processed_columns, "counts": summary_counts, "artifacts": [str(output / name) for name in ("feature_selection.csv", "feature_selection_summary.md", "feature_processing_manifest.json")]}
